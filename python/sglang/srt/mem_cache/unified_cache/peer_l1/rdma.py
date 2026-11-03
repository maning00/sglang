"""Registered peer reads with request-local failure and permanent target quarantine."""

from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import msgspec
import torch

from sglang.srt.mem_cache.unified_cache.unified_cache_linker import LayerWiseLoadCounter
from sglang.srt.runtime_context import get_resources

from .copier import distinct_stream
from .rdma_io import Submission, UnsafeTransfer, execute

logger = logging.getLogger(__name__)


class MemoryAdvertisement(msgspec.Struct, frozen=True):
    engine: bytes
    regions: tuple[bytes, ...]
    geometry: tuple


class RemoteMemory(msgspec.Struct, frozen=True):
    engine: object
    regions: list


class ReadBatch(msgspec.Struct):
    rids: list[str]
    index: int
    keys: list[str]
    futures: list
    failed: set[str] = msgspec.field(default_factory=set)
    poisoned: bool = False

    def resolved_layers(self):
        return [i for i, future in enumerate(self.futures) if future.done()]

    def failed_requests(self):
        return self.failed

    def quiesced(self):
        return not self.poisoned and all(f.done() for f in self.futures)


class RdmaReadMover:
    def __init__(self, *, memory, host, quarantine, stats, on_failure=None):
        from mori.io import BackendType, IOEngine, IOEngineConfig, MemoryLocationType

        self.memory, self.quarantine, self.stats = memory, quarantine, stats
        self.on_failure = on_failure
        self.engine = IOEngine(
            "peer-l1-" + uuid.uuid4().hex, IOEngineConfig(host=host, port=0)
        )
        self.engine.create_backend(BackendType.RDMA)
        self.regions = [
            self.engine.register_memory(
                b.data_ptr(),
                b.numel() * b.element_size(),
                memory.device.index,
                MemoryLocationType.GPU,
            )
            for b in memory.buffers
        ]
        self.widths = [b[0].numel() * b.element_size() for b in memory.buffers]
        self.geometry = tuple(
            (g[0], g[1], tuple(g[2][1:]), g[3], g[4]) for g in memory.geometry
        )
        self.remotes = {}
        self.stream = torch.cuda.Stream(device=memory.device, priority=-1)
        self.counter = LayerWiseLoadCounter(len(memory.layer_offsets) - 1)
        self._io = ThreadPoolExecutor(max_workers=1, thread_name_prefix="peer-rdma")
        # MoRI cannot prove revocation of every remote access at pool teardown.
        # Fixed KV registrations and their allocations live until process exit.
        get_resources().rdma_quarantines.append(self)

    def export(self):
        return msgspec.msgpack.encode(
            MemoryAdvertisement(
                self.engine.get_engine_desc().pack(),
                tuple(r.pack() for r in self.regions),
                self.geometry,
            )
        )

    def connect(self, data):
        from mori.io import EngineDesc, MemoryDesc

        advertised = msgspec.msgpack.decode(data, type=MemoryAdvertisement)
        # Decode into tuples explicitly: array-like geometry has nested dimensions.
        geometry = tuple(
            (g[0], g[1], tuple(g[2]), g[3], g[4]) for g in advertised.geometry
        )
        if geometry != self.geometry or len(advertised.regions) != len(self.regions):
            raise ValueError("peer RDMA KV layouts differ")
        engine = EngineDesc.unpack(advertised.engine)
        if engine.key not in self.remotes:
            self.engine.register_remote_engine(engine)
            self.remotes[engine.key] = RemoteMemory(
                engine, [MemoryDesc.unpack(r) for r in advertised.regions]
            )
        return self.remotes[engine.key]

    def read(self, *, remote, grant, targets):
        deadline = (
            float("inf")
            if grant.lease_s is None
            else grant.acquired_before + grant.lease_s / 2
        )
        if time.monotonic() >= deadline:
            return False
        local_offsets = [[p * width for p in targets] for width in self.widths]
        remote_pages = grant.rows.tolist()
        remote_offsets = [[p * width for p in remote_pages] for width in self.widths]
        sizes = [[width] * len(targets) for width in self.widths]
        submission = Submission(
            engine=self.engine, local=self.regions, remote=remote.regions, owner=self
        )

        def wait(statuses):
            for status in statuses:
                if deadline == float("inf"):
                    status.Wait()
                else:
                    status.WaitFor(max(0, int((deadline - time.monotonic()) * 1000)))
                if not status.Succeeded():
                    return False
            return time.monotonic() < deadline

        return execute(
            submission,
            local_offsets=local_offsets,
            remote_offsets=remote_offsets,
            sizes=sizes,
            write=False,
            wait=wait,
        )

    def submit(self, plans, forward_stream):
        schedule = torch.cuda.current_stream()
        self.stream = distinct_stream(self.stream, (schedule, forward_stream))
        ready = (torch.cuda.Event(), torch.cuda.Event())
        ready[0].record(schedule)
        ready[1].record(forward_stream)
        index = self.counter.update_producer()
        batch = ReadBatch(
            [p.rid for p in plans],
            index,
            [k for p in plans for k in p.keys],
            self.counter.futures[index],
        )
        self._io.submit(self._copy, plans, ready, batch)
        return batch

    def _copy(self, plans, ready, batch):
        released = set()
        try:
            torch.cuda.set_device(self.memory.device)
            with torch.cuda.stream(self.stream):
                for event in ready:
                    event.synchronize()
                for plan in plans:
                    try:
                        targets = (
                            plan.indices.reshape(-1, 256)[:, 0].cpu() // 256
                        ).tolist()
                        for begin in range(0, len(targets), 64):
                            end = begin + 64
                            if plan.session.owner.peer.retired:
                                batch.failed.add(plan.rid)
                                break
                            grant = msgspec.structs.replace(
                                plan.grant, rows=plan.rows[begin:end]
                            )
                            if not self.read(
                                remote=plan.session.mapping,
                                grant=grant,
                                targets=targets[begin:end],
                            ):
                                batch.failed.add(plan.rid)
                                break
                            self.stats["rdma_read_pages"] += len(targets[begin:end])
                    except UnsafeTransfer as exc:
                        added = self.quarantine.mark(targets[begin:end], exc.submission)
                        self.stats["rdma_quarantined_pages"] += added
                        batch.poisoned = True
                        batch.failed.add(plan.rid)
                        logger.error(
                            "Peer RDMA targets quarantined: rid=%s pages=%d; restart required",
                            plan.rid,
                            added,
                        )
                    except BaseException:
                        batch.failed.add(plan.rid)
                        logger.exception("Peer RDMA read failed before submission")
                    finally:
                        plan.session.owner.release(plan.grant)
                        released.add(plan.rid)
        except BaseException:
            batch.failed.update(batch.rids)
            logger.exception("Peer RDMA batch preparation failed")
        finally:
            for plan in plans:
                if plan.rid not in released:
                    plan.session.owner.release(plan.grant)
            for plan in plans:
                if plan.rid in batch.failed and self.on_failure is not None:
                    self.on_failure(plan.session)
            self.stats["rdma_failed_requests"] += len(batch.failed)
            for layer in range(self.counter.num_layers):
                self.counter.complete(batch.index, layer)

    def close(self):
        self._io.shutdown(wait=True)


class MixedBatch(msgspec.Struct):
    rids: list[str]
    index: int
    keys: list[str]
    children: list

    def resolved_layers(self):
        return sorted(
            set.intersection(*(set(c.resolved_layers()) for c in self.children))
        )

    def failed_requests(self):
        return set().union(*(c.failed_requests() for c in self.children))

    def quiesced(self):
        return all(c.quiesced() for c in self.children)


class MixedMover:
    def __init__(self, ipc, rdma):
        from types import SimpleNamespace

        from .composite import CompositeLayerCounter

        self.ipc, self.rdma = ipc, rdma
        self.counter = CompositeLayerCounter(
            SimpleNamespace(
                layer_done_counter=ipc.counter,
                finish_layer_wise_loading=ipc.counter.finish,
            ),
            SimpleNamespace(
                layer_done_counter=rdma.counter,
                finish_layer_wise_loading=rdma.counter.finish,
            ),
        )

    @property
    def stream(self):
        return self.ipc.stream

    def submit(self, plans, forward_stream):
        local = [p for p in plans if p.session.cost == "local"]
        remote = [p for p in plans if p.session.cost == "rdma"]
        ipc = self.ipc.submit(local, forward_stream) if local else None
        rdma = self.rdma.submit(remote, forward_stream) if remote else None
        index = self.counter.register(
            ipc.index if ipc else -1, rdma.index if rdma else -1
        )
        return MixedBatch(
            [p.rid for p in plans],
            index,
            [k for p in plans for k in p.keys],
            [b for b in (ipc, rdma) if b is not None],
        )
