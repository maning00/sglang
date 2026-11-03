"""One-sided I/O on the PD engine. Memory is retained until every status drains."""

from __future__ import annotations

import logging
import time

import torch

from sglang.srt.mem_cache.allocator.quarantine import QuarantinedPages

from ..channels import LendOwnerChannel
from ..rdma_io import Submission, UnsafeTransfer, submit_batches
from ..session import PeerSession

logger = logging.getLogger(__name__)


class LendTransport:
    def __init__(self, manager, *, memory_owner=None):
        self.manager = manager
        self.memory_owner = memory_owner
        self.failed_attempts = []
        self.engine = manager.engine
        self.lengths = list(manager.kv_args.kv_item_lens)
        self.staging = None

    @staticmethod
    def validate(manager):
        from sglang.srt.disaggregation.mori.conn import MoriKVManager

        if not manager.enable_deferred_decode_kv_release:
            raise ValueError(
                "decode lend requires deferred decode KV release on prefill"
            )
        if (
            not isinstance(manager, MoriKVManager)
            or manager.attn_tp_size != 1
            or manager.dcp_size != 1
        ):
            raise ValueError("decode lend requires MoRI with attention TP=1 and CP=1")

    def transfer(self, info, local_pages, remote_pages, *, write=False, staging=False):
        if len(local_pages) != len(remote_pages) or not local_pages:
            raise ValueError("Lend I/O needs matching, nonempty page lists")
        remote = info.dst_kv_mem_descs
        local = (
            [self.staging.desc] * len(remote) if staging else self.manager.kv_mem_descs
        )
        if not local or len(local) != len(remote) or len(remote) != len(self.lengths):
            raise ValueError("Decode lend requires identical PD KV layouts")
        offsets = self.staging.offsets if staging else [0] * len(local)
        local_offsets = [
            [offset + p * size for p in local_pages]
            for offset, size in zip(offsets, self.lengths)
        ]
        remote_offsets = [[p * size for p in remote_pages] for size in self.lengths]
        sizes = [[size] * len(local_pages) for size in self.lengths]
        submission = Submission(
            engine=self.engine, local=local, remote=remote, owner=self
        )
        try:
            submit_batches(
                submission,
                local_offsets=local_offsets,
                remote_offsets=remote_offsets,
                sizes=sizes,
                write=write,
            )
            ok = self.manager._wait_transfer_completion(submission.statuses) is None
            if not ok or not all(status.Succeeded() for status in submission.statuses):
                raise RuntimeError("RDMA status failed or did not complete")
            return True
        except BaseException as exc:
            if not submission.submitted:
                raise
            from sglang.srt.runtime_context import get_resources

            self.failed_attempts.append(submission)
            get_resources().rdma_quarantines.append(submission)
            raise UnsafeTransfer(submission) from exc

    def read(self, peer, records, local_pages, *, staging=False):
        if peer.session is None:
            peer.session = PeerSession(
                endpoint=f"tcp://{peer.info.endpoint}:{peer.info.lend_port}",
                owner=LendOwnerChannel(peer),
                mover=self,
                mapping=peer.info,
                cost="rdma",
            )
        session = peer.session
        grant = session.owner.acquire_exact([r[2] for r in records])
        if grant is None:
            return False
        try:
            ok = session.mover.transfer(
                session.mapping, local_pages, grant.rows.tolist(), staging=staging
            )
            return ok and time.monotonic() - grant.acquired_before < grant.lease_s / 2
        finally:
            session.owner.release(grant)


class DemotionStaging:
    def __init__(self, transport, lower, pages):
        from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
            DevicePoolEntry,
        )

        self.pages, self.transport = pages, None
        self.desc = None
        self.offsets, total = [], 0
        for size in transport.lengths:
            self.offsets.append(total)
            total += pages * size
        # GPU memory, like the KV pool: on a multi-NIC PD engine, RDMA reads into
        # registered host memory hit local protection errors where GPU reads did not.
        self.device = transport.manager.kv_args.gpu_id
        self.tensor = torch.empty(
            total, dtype=torch.uint8, device=f"cuda:{self.device}"
        )
        self.total = total
        self.quarantine = QuarantinedPages(page_size=1, owner=self.tensor)
        if not lower.storage.client.register_memory(self.tensor.data_ptr(), total):
            raise RuntimeError("UMBP rejected decode lend demotion staging")
        buffers = {}
        for pointer, size, offset in zip(
            transport.manager.kv_args.kv_data_ptrs, transport.lengths, self.offsets
        ):
            buffers[pointer] = self.tensor[offset : offset + pages * size].view(
                pages, size
            )
        self.entries = {}
        for name, source in lower.pools.items():
            components = []
            for component in source.buffer_meta:
                target = []
                for pointer, stride, size in component:
                    if (
                        pointer not in buffers
                        or stride != size
                        or buffers[pointer].shape[1] != size
                    ):
                        raise ValueError(
                            "Demotion staging requires matching unified KV page regions"
                        )
                    target.append(buffers[pointer])
                components.append(target)
            self.entries[name] = DevicePoolEntry(
                name=name,
                indices_from_pool=source.indices_from_pool,
                device_pool=None,
                components=components,
                layer_mapping=source.layer_mapping,
                page_size=source.page_size,
                rows_are_pages=True,
                packed=source.packed,
            )
        self.bind(transport)

    def bind(self, transport):
        from mori.io import MemoryLocationType

        if self.desc is not None:
            raise RuntimeError("Demotion staging is already bound")
        self.desc = transport.engine.register_memory(
            self.tensor.data_ptr(), self.total, self.device, MemoryLocationType.GPU
        )
        self.transport = transport

    def put(self, lower, records, *, slots=None):
        from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
        from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import (
            _object_sizes_per_page,
        )

        slots = list(range(len(records))) if slots is None else slots
        indices = (
            torch.tensor(slots, dtype=torch.int64)[:, None] * lower.page_size
            + torch.arange(lower.page_size)
        ).flatten()
        expanded = lower.pool_group.resolve_transfers(
            [
                PoolTransfer(
                    name=PoolName.KV,
                    keys=[r[2] for r in records],
                    device_indices=indices,
                )
            ]
        )
        plans = lower._build_load_plans([expanded])
        for plan in plans:
            entry = self.entries[plan.name]
            ptrs, sizes, offsets = lower._all_layer_ranges(plan, entry=entry)
            object_sizes = _object_sizes_per_page(entry) * len(plan.locations)
            step = lower._entries_per_call(sizes)
            for start in range(0, len(plan.keys), step):
                end = start + step
                result = list(
                    lower.storage.client.batch_put_ranges_from_ptr(
                        plan.keys[start:end],
                        object_sizes[start:end],
                        ptrs[start:end],
                        sizes[start:end],
                        offsets[start:end],
                    )
                )
                if len(result) != len(plan.keys[start:end]) or not all(result):
                    return False
        return True

    def close(self):
        if self.quarantine.pages:
            raise RuntimeError("Quarantined demotion staging requires process restart")
        if self.desc is not None:
            self.transport.engine.deregister_memory(self.desc)
            self.desc = self.transport = None
        # UMBP unregisters all its regions on lower.close(). Keep the buffer
        # alive until that client has stopped referencing it.

    def release(self):
        self.tensor = None
