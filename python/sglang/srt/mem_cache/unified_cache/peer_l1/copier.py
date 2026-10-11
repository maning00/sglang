"""IPC mappings and layer-gated, consumer-side HIP copies."""

from __future__ import annotations

import socket
from dataclasses import dataclass, field

import torch
from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def copy_module():
    if not torch.version.hip:
        raise RuntimeError("peer_l1 currently requires ROCm")
    return load_jit(
        "peer_l1_copy",
        cuda_files=["peer_l1/copy.cuh"],
        cuda_wrappers=[
            ("copy", "peer_l1::launch_copy"),
            ("unpin", "peer_l1::enqueue_unpin"),
        ],
    )


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


@dataclass
class CopyBatch:
    rids: list[str]
    index: int
    finish: object
    keys: list[str]
    rows_cpu: torch.Tensor
    error_cpu: torch.Tensor | None
    # Own every temporary and all source directory mappings through the batch
    # event, which is recorded after the unpin callback returns.
    keepalive: list = field(default_factory=list)


class L1Copier:
    def __init__(
        self,
        memory: PeerMemoryMap,
        *,
        layers: int,
        layer_group: int,
        blocks: int,
        verify: bool,
        priority: int = 0,
    ):
        self.memory = memory
        self.module = copy_module()
        # A non-default priority draws from a separate stream pool, which can map
        # to its own hardware queue instead of one shared with forward or EP.
        self.stream = torch.cuda.Stream(device=memory.device, priority=priority)
        self.counter = LayerCounter(layers)
        self.layer_group, self.blocks, self.verify = layer_group, blocks, verify

    def submit(self, plans, forward_stream):
        schedule_stream = torch.cuda.current_stream()
        # PyTorch streams come from a round-robin pool. A fresh stream is not
        # necessarily distinct from the scheduler's subsequently drawn stream.
        if self.stream.cuda_stream in (
            schedule_stream.cuda_stream,
            forward_stream.cuda_stream,
        ):
            from sglang.srt.managers.utils import allocate_distinct_stream

            self.stream = allocate_distinct_stream(
                torch.cuda, (schedule_stream, forward_stream)
            )
        index = self.counter.update_producer()
        events, finish = self.counter.events[index]
        self.stream.wait_stream(schedule_stream)
        self.stream.wait_stream(forward_stream)
        # Each plan has one peer, selected source rows, destination indices,
        # and the full reservation (which can include deduplicated pages).
        keys = [key for plan in plans for key in plan.keys]
        sources = torch.empty(
            (len(keys), 2), dtype=torch.int64, device="cpu", pin_memory=True
        )
        unpins = []
        offset = 0
        for plan in plans:
            count = len(plan.keys)
            sources[offset : offset + count, 0] = plan.peer
            sources[offset : offset + count, 1].copy_(torch.from_numpy(plan.rows))
            offset += count
            unpins.extend(
                (plan.directory.address, int(row)) for row in plan.reserved_rows
            )
        unpins = torch.tensor(unpins, dtype=torch.int64, device="cpu").reshape(-1, 2)
        rows_cpu = torch.empty(
            len(keys), dtype=torch.int64, device="cpu", pin_memory=True
        )
        with torch.cuda.stream(self.stream):
            destinations = (
                torch.cat([plan.indices.reshape(-1, 256)[:, 0] for plan in plans])
                // 256
            )
            source_gpu = sources.to(self.memory.device, non_blocking=True)
            jobs = torch.cat((source_gpu, destinations[:, None]), dim=1)
            rows_cpu.copy_(destinations, non_blocking=True)
            error = (torch.zeros if self.verify else torch.empty)(
                1, dtype=torch.int64, device=self.memory.device
            )
            offsets = self.memory.layer_offsets
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
            self.module.unpin(unpins, self.stream.cuda_stream)
            finish.record(self.stream)
        return CopyBatch(
            [plan.rid for plan in plans],
            index,
            finish,
            keys,
            rows_cpu,
            error_cpu,
            [plans, sources, unpins, destinations, source_gpu, jobs, error],
        )
