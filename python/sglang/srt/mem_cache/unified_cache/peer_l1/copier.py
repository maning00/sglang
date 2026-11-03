"""IPC mappings and layer-gated, consumer-side HIP copies."""

from __future__ import annotations

import ctypes
import glob
import os
import socket
from pathlib import Path

import msgspec
import numpy as np
import torch

from sglang.kernels.jit.utils import cache_once, load_jit

from .control import native


def validate_ipc_backend(*, server_args, params, pool, world):
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.runtime_context import get_memory

    if not torch.version.hip or params.page_size != 256:
        raise ValueError("peer_l1 requires ROCm and page_size=256")
    if get_memory().enable_unified_memory or server_args.enable_memory_saver:
        raise ValueError("peer_l1 requires fixed KV allocations without memory saver")
    allocator_config = os.environ.get(
        "PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    )
    if (
        "expandable_segments:True" in allocator_config
        or "backend:cudaMallocAsync" in allocator_config
    ):
        raise ValueError(
            "peer_l1 requires the native, non-expandable PyTorch allocator"
        )
    if not isinstance(pool, DeepSeekV4TokenToKVPool) or not pool._unified_kv:
        raise ValueError("peer_l1 currently supports DeepSeek V4 unified KV only")
    if (
        not 2 <= world <= 8
        or params.pp_size != 1
        or params.attn_cp_size != 1
        or torch.distributed.get_world_size(group=params.tp_cache_group) != 1
    ):
        raise ValueError(
            "peer_l1 requires 2–8 local DP workers with attention TP=1, CP=1, PP=1"
        )
    if torch.cuda.device_count() < world:
        raise ValueError("Every peer_l1 worker must see every participating GPU")


@cache_once
def copy_module():
    if not torch.version.hip:
        raise RuntimeError("peer_l1 currently requires ROCm")
    return load_jit(
        "peer_l1_copy",
        cuda_files=["peer_l1/copy.cuh"],
        cuda_wrappers=[
            ("copy", "peer_l1::launch_copy"),
            ("cleanup", "peer_l1::enqueue_cleanup"),
        ],
    )


@cache_once
def sdma_module():
    import mori

    # The image's mori checkout: headers under include/, libraries under build/src/.
    root = Path(mori.__file__).resolve().parents[2]
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    return load_jit(
        "peer_l1_sdma",
        cuda_files=["peer_l1/sdma.cuh"],
        cuda_wrappers=[
            ("connect", "peer_l1::sdma_connect"),
            ("alloc_signals", "peer_l1::sdma_alloc_signals"),
            ("submit", "peer_l1::sdma_submit"),
            ("wait", "peer_l1::sdma_wait"),
        ],
        extra_include_paths=[str(root / "include")],
        extra_ldflags=[
            f"-L{root}/build/src/application",
            "-lmori_application",
            f"-L{rocm}/lib",
            "-lhsa-runtime64",
            *(f"-Wl,-rpath,{d}" for d in sorted(glob.glob(f"{root}/build/src/*/"))),
        ],
    )


class SdmaEngine:
    """mori anvil SDMA queues, one per peer, that pull peer rows into this GPU."""

    def __init__(self, memory, peer_devices, channels):
        self.module = sdma_module()
        device = memory.device
        # Queues are peer-major in peer rank order, `channels` per peer; the copier
        # sorts jobs by peer and splits each peer's jobs across its channels.
        self.peers = sorted(peer_devices)
        self.channels = channels
        handles = torch.empty(len(self.peers) * channels, dtype=torch.int64)
        self.module.connect(
            device.index,
            torch.tensor([peer_devices[p] for p in self.peers], dtype=torch.int64),
            channels,
            handles,
        )
        signals = torch.empty(len(handles), dtype=torch.int64)
        self.module.alloc_signals(signals)
        self.signal_ptrs = signals.tolist()
        self.handles = handles.to(device)
        self.signals = signals.to(device)
        # Every queue's signal rises by one per layer group of every batch.
        self.base = 0


class LayerCounter:
    """The HiCache counter interface, with completion after native cleanup."""

    def __init__(self, layers: int):
        self.num_layers = layers
        self.events = [
            ([torch.cuda.Event() for _ in range(layers)], torch.cuda.Event())
            for _ in range(3)
        ]
        self.producer_index = self.consumer_index = -1

    def update_producer(self):
        index = (self.producer_index + 1) % len(self.events)
        if not self.events[index][1].query():
            raise RuntimeError("peer L1 copy producer slot is still in flight")
        self.producer_index = index
        return index

    def set_consumer(self, index):
        self.consumer_index = index

    def wait_until(self, layer):
        if self.consumer_index >= 0:
            torch.cuda.current_stream().wait_event(
                self.events[self.consumer_index][0][layer]
            )

    def finish(self, index):
        if index >= 0:
            self.events[index][1].synchronize()
        return True

    def reset(self):
        self.producer_index = self.consumer_index = -1


class PeerMemoryMap:
    def __init__(self, group, rank: int, world: int, device, cpu_group, directory_info):
        self.device = device
        self.storages = []
        self.buffers = []
        self.layer_offsets = [0]
        storage_ids = {}
        views = []
        geometry = []
        for layer in range(group.num_layers):
            for entry in group.entries:
                mapping = entry.layer_mapping.get(layer)
                if mapping is None:
                    continue
                indices = [mapping] if isinstance(mapping, int) else mapping
                for component in entry.components:
                    for index in indices:
                        buffer = component[index]
                        row_bytes = buffer[0].numel() * buffer.element_size()
                        if (
                            not buffer.is_contiguous()
                            or row_bytes % 16
                            or buffer.data_ptr() % 16
                            or entry._row_span != 1
                            or entry._index_mapper is not None
                        ):
                            raise ValueError(
                                "peer_l1 requires contiguous, 16-byte-aligned page rows"
                            )
                        storage = buffer.untyped_storage()
                        identity = (storage.data_ptr(), storage.nbytes())
                        if identity not in storage_ids:
                            storage_ids[identity] = len(self.storages)
                            self.storages.append(storage)
                        views.append(
                            (
                                storage_ids[identity],
                                buffer.data_ptr() - storage.data_ptr(),
                            )
                        )
                        geometry.append(
                            (
                                str(entry.name),
                                layer,
                                tuple(buffer.shape),
                                str(buffer.dtype),
                                row_bytes,
                            )
                        )
                        self.buffers.append(buffer)
            self.layer_offsets.append(len(views))

        # _share_cuda_ allocates a reference-counter entry for ONE receiver.
        # Export once per receiving process, while PyTorch reuses IPC mappings
        # for storages that belong to the same caching-allocator allocation.
        handles = [
            (
                [storage._share_cuda_() for storage in self.storages]
                if peer != rank
                else None
            )
            for peer in range(world)
        ]
        packet = (socket.gethostname(), directory_info, geometry, views, handles)
        packets = [None] * world
        torch.distributed.all_gather_object(packets, packet, group=cpu_group)
        self.directory_infos = [packet[1] for packet in packets]
        self.peer_storages = []
        addresses = [[0] * 8 for _ in views]
        try:
            for peer, packet in enumerate(packets):
                host, _, peer_geometry, peer_views, peer_handles = packet
                if host != socket.gethostname() or peer_geometry != geometry:
                    raise ValueError(
                        "peer_l1 requires identical device layouts on one node"
                    )
                if peer == rank:
                    continue
                mapped = [
                    torch.UntypedStorage._new_shared_cuda(device.index, *handle[1:])
                    for handle in peer_handles[rank]
                ]
                self.peer_storages.extend(mapped)
                for index, (storage, offset) in enumerate(peer_views):
                    addresses[index][peer] = mapped[storage].data_ptr() + offset
            desc = [
                [buffer.data_ptr(), geometry[i][-1], *addresses[i]]
                for i, buffer in enumerate(self.buffers)
            ]
            self.descriptors = torch.tensor(desc, dtype=torch.int64, device=device)
        except BaseException:
            self.peer_storages.clear()
            raise


class CopyBatch(msgspec.Struct):
    rids: list[str]
    index: int
    finish: object
    keys: list[str]
    error_cpu: torch.Tensor | None
    # Own every temporary and source grants through the batch
    # event, which is recorded after the cleanup callback returns.
    keepalive: list = msgspec.field(default_factory=list)
    layer_events: list = msgspec.field(default_factory=list)

    def resolved_layers(self):
        return [i for i, event in enumerate(self.layer_events) if event.query()]

    def failed_requests(self):
        return (
            set(self.rids)
            if self.error_cpu is not None and self.error_cpu.item()
            else set()
        )

    def quiesced(self):
        return self.finish.query()


def distinct_stream(stream, avoid):
    """Return ``stream``, or a same-priority replacement if it aliases one of ``avoid``
    (PyTorch draws streams from a round-robin pool)."""
    if stream.cuda_stream not in {other.cuda_stream for other in avoid}:
        return stream
    from sglang.srt.managers.utils import allocate_distinct_stream

    # A normal-priority replacement would share a hardware queue with the
    # scheduler and stall its syncs while waiting on the forward stream.
    return allocate_distinct_stream(torch.cuda, avoid, priority=stream.priority)


class L1Copier:
    def __init__(
        self,
        memory: PeerMemoryMap,
        *,
        layers: int,
        layer_group: int,
        blocks: int,
        verify: bool,
        priority: int = -1,
        engine: str = "kernel",
        peer_devices: dict | None = None,
        sdma_channels: int = 2,
    ):
        self.memory = memory
        self.module = copy_module()
        # The kernel engine copies on CUs; sdma moves rows on the DMA engines and
        # keeps the kernel only for byte verification.
        self.cleanup_callback = ctypes.cast(
            native().l1_cleanup_enqueue, ctypes.c_void_p
        ).value
        self.sdma = (
            SdmaEngine(memory, peer_devices, sdma_channels)
            if engine == "sdma"
            else None
        )
        # Normal-priority streams share a few hardware queues (GPU_MAX_HW_QUEUES), so
        # this stream's wait on the forward stream can stall the scheduler's syncs.
        self.stream = torch.cuda.Stream(device=memory.device, priority=priority)
        self.counter = LayerCounter(layers)
        self.layer_group, self.blocks, self.verify = layer_group, blocks, verify
        groups = list(range(0, layers, layer_group)) + [layers]
        self.group_first = torch.tensor(
            [memory.layer_offsets[g] for g in groups],
            dtype=torch.int64,
            device=memory.device,
        )

    def submit(self, plans, forward_stream):
        schedule_stream = torch.cuda.current_stream()
        self.stream = distinct_stream(self.stream, (schedule_stream, forward_stream))
        index = self.counter.update_producer()
        events, finish = self.counter.events[index]
        self.stream.wait_stream(schedule_stream)
        self.stream.wait_stream(forward_stream)
        if self.sdma is not None:
            # Each SDMA queue reads one peer: group the jobs by peer, in queue order.
            plans = sorted(plans, key=lambda plan: plan.peer)
        # Each plan has one peer, source rows and request-private destinations.
        keys = [key for plan in plans for key in plan.keys]
        sources = torch.empty(
            (len(keys), 2), dtype=torch.int64, device="cpu", pin_memory=True
        )
        cleanups = []
        offset = 0
        for plan in plans:
            count = len(plan.keys)
            sources[offset : offset + count, 0] = plan.peer
            sources[offset : offset + count, 1].copy_(torch.from_numpy(plan.rows))
            offset += count
            cleanups.append(plan.cleanup)
        cleanups = torch.tensor(cleanups, dtype=torch.int64, device="cpu")
        with torch.cuda.stream(self.stream):
            destinations = (
                torch.cat([plan.indices.reshape(-1, 256)[:, 0] for plan in plans])
                // 256
            )
            source_gpu = sources.to(self.memory.device, non_blocking=True)
            jobs = torch.cat((source_gpu, destinations[:, None]), dim=1)
            error = (torch.zeros if self.verify else torch.empty)(
                1, dtype=torch.int64, device=self.memory.device
            )
            offsets = self.memory.layer_offsets
            if self.sdma is not None:
                job_begin = self._submit_sdma(plans, jobs, events)
            else:
                job_begin = None
                for first in range(0, self.counter.num_layers, self.layer_group):
                    last = min(first + self.layer_group, self.counter.num_layers)
                    self.module.copy(
                        self.memory.descriptors,
                        jobs,
                        offsets[first],
                        offsets[last],
                        self.blocks,
                        error,
                        False,
                        self.stream.cuda_stream,
                    )
                    for layer in range(first, last):
                        events[layer].record(self.stream)
            error_cpu = None
            if self.verify:
                self.module.copy(
                    self.memory.descriptors,
                    jobs,
                    0,
                    offsets[-1],
                    self.blocks,
                    error,
                    True,
                    self.stream.cuda_stream,
                )
                error_cpu = torch.empty(
                    1, dtype=torch.int64, device="cpu", pin_memory=True
                )
                error_cpu.copy_(error, non_blocking=True)
            self.module.cleanup(
                cleanups, self.cleanup_callback, self.stream.cuda_stream
            )
            finish.record(self.stream)
        return CopyBatch(
            [plan.rid for plan in plans],
            index,
            finish,
            keys,
            error_cpu,
            [
                plans,
                sources,
                cleanups,
                destinations,
                source_gpu,
                jobs,
                error,
                job_begin,
            ],
            events,
        )

    def _submit_sdma(self, plans, jobs, events):
        """Queue the batch on the SDMA engines; the copy stream then waits on every
        queue's signal per layer group before recording that group's layer events."""
        sdma, stream = self.sdma, self.stream.cuda_stream
        per_peer = dict.fromkeys(sdma.peers, 0)
        for plan in plans:
            per_peer[plan.peer] += len(plan.keys)
        # A peer's jobs are contiguous; split them evenly across its channels.
        counts = [
            n // sdma.channels + (c < n % sdma.channels)
            for n in per_peer.values()
            for c in range(sdma.channels)
        ]
        job_begin = torch.empty(len(counts) + 1, dtype=torch.int64, pin_memory=True)
        job_begin[0] = 0
        job_begin[1:] = torch.from_numpy(np.cumsum(counts))
        job_begin_gpu = job_begin.to(self.memory.device, non_blocking=True)
        groups = len(self.group_first) - 1
        sdma.module.submit(
            sdma.handles,
            job_begin_gpu,
            self.memory.descriptors,
            jobs,
            self.group_first,
            groups,
            sdma.signals,
            stream,
        )
        for g, first in enumerate(range(0, self.counter.num_layers, self.layer_group)):
            for signal in sdma.signal_ptrs:
                sdma.module.wait(stream, signal, sdma.base + g + 1)
            for layer in range(
                first, min(first + self.layer_group, self.counter.num_layers)
            ):
                events[layer].record(self.stream)
        sdma.base += groups
        return [job_begin, job_begin_gpu]
