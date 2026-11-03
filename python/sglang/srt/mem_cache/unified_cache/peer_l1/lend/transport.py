"""One-sided I/O on the PD engine. Memory is retained until every status drains."""

from __future__ import annotations

import logging
import time

import torch

logger = logging.getLogger(__name__)


class LendTransport:
    def __init__(self, manager):
        self.manager = manager
        self.engine = manager.engine
        self.lengths = list(manager.kv_args.kv_item_lens)
        self.staging = None

    def transfer(self, info, local_pages, remote_pages, *, write=False, staging=False):
        if len(local_pages) != len(remote_pages) or not local_pages:
            raise ValueError("Lend I/O needs matching, nonempty page lists")
        remote = info.dst_kv_mem_descs
        local = (
            [self.staging.desc] * len(remote) if staging else self.manager.kv_mem_descs
        )
        if len(local) != len(remote) or len(remote) != len(self.lengths):
            raise ValueError("Decode lend requires identical PD KV layouts")
        offsets = self.staging.offsets if staging else [0] * len(local)
        local_offsets = [
            [offset + p * size for p in local_pages]
            for offset, size in zip(offsets, self.lengths)
        ]
        remote_offsets = [[p * size for p in remote_pages] for size in self.lengths]
        sizes = [[size] * len(local_pages) for size in self.lengths]
        statuses = []
        success = False
        try:
            operation = self.engine.batch_write if write else self.engine.batch_read
            # Record each returned batch before submitting the next one: partial
            # submission failures must drain the batches already posted.
            for start in range(0, len(local), 16):
                end = start + 16
                statuses.extend(
                    operation(
                        local[start:end],
                        local_offsets[start:end],
                        remote[start:end],
                        remote_offsets[start:end],
                        sizes[start:end],
                        [self.engine.allocate_transfer_uid() for _ in local[start:end]],
                    )
                )
            success = self.manager._wait_transfer_completion(statuses) is None
        finally:
            # wait_all can return on the first error while other entries run.
            # An indefinite drain intentionally retains the caller's memory.
            for status in statuses:
                while True:
                    try:
                        if status.InProgress():
                            status.Wait()
                        break
                    except Exception:
                        logger.exception(
                            "Decode lend drain failed; retaining its buffers and retrying"
                        )
                        time.sleep(0.01)
        return success and all(s.Succeeded() for s in statuses)

    def read(self, peer, records, local_pages, *, staging=False):
        started = time.monotonic()  # BEFORE ACQUIRE, including control-plane delay.
        reply = peer.call("ACQUIRE", [r[2] for r in records])
        lease = reply.data[0] if reply is not None and reply.data else 0
        if not lease:
            return False
        try:
            granted = reply.data[1]
            if len(granted) != len(records):
                return False
            records = granted
            ok = self.transfer(
                peer.info, local_pages, [r[0] for r in records], staging=staging
            )
            return ok and time.monotonic() - started < peer.lease_s / 2
        finally:
            peer.cleanup("UNPIN", lease)


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

    def put(self, lower, records):
        from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
        from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import (
            _object_sizes_per_page,
        )

        indices = torch.arange(len(records) * lower.page_size, dtype=torch.int64)
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
        if self.desc is not None:
            self.transport.engine.deregister_memory(self.desc)
            self.desc = self.transport = None
        # UMBP unregisters all its regions on lower.close(). Keep the buffer
        # alive until that client has stopped referencing it.

    def release(self):
        self.tensor = None
