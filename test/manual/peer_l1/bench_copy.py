"""Two/eight-process IPC copy validation and a small bandwidth sweep.

PYTHONPATH=python python test/manual/peer_l1/bench_copy.py --ranks 2 --pages 256
--skip-copy is the negative control: it must fail byte verification.
"""

import argparse
import hashlib
import json
import socket
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
)
from sglang.srt.mem_cache.unified_cache.peer_l1.copier import L1Copier, PeerMemoryMap
from sglang.srt.mem_cache.unified_cache.peer_l1.directory import L1Directory, hash_words


def keys(rank, count):
    return [
        hashlib.sha256(f"{rank}:{page}".encode()).hexdigest() for page in range(count)
    ]


def worker(rank, args, port):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=args.ranks
    )
    count = args.pages
    rows = count * 2 + 1
    entries = []
    # Actual unified fp8 Pro row geometry: 30 C4 and 31 C128 layers.
    layouts = [
        (PoolName.DEEPSEEK_V4_C4, 32768, list(range(0, 60, 2))),
        (PoolName.DEEPSEEK_V4_C4_ROPE, 8192, list(range(0, 60, 2))),
        (PoolName.DEEPSEEK_V4_C128, 1024, list(range(1, 61, 2)) + [60]),
        (PoolName.DEEPSEEK_V4_C128_ROPE, 256, list(range(1, 61, 2)) + [60]),
    ]
    # Indexer enum differs between model families; the descriptor only needs a
    # distinct entry name, and uses the production C4 indexer width.
    layouts.append((PoolName.KV, 8448, list(range(0, 60, 2))))
    for name, width, layers in layouts:
        buffers = []
        for layer in layers:
            buffer = torch.empty((rows, width), dtype=torch.uint8, device=device)
            row_pattern = (
                torch.arange(rows, device=device) * 17 + rank * 31 + layer * 7
            ) % 251
            buffer.copy_(row_pattern[:, None].to(torch.uint8).expand(rows, width))
            buffer[count + 1 :].fill_(255)
            buffers.append(buffer)
        entries.append(
            DevicePoolEntry(
                name=name,
                indices_from_pool=PoolName.KV,
                device_pool=None,
                components=[buffers],
                layer_mapping=dict(zip(layers, range(len(layers)))),
                page_size=256,
                rows_are_pages=True,
            )
        )
    group = DevicePoolGroup(entries, 61, 256, rank_replicated=True)
    path = tempfile.mktemp(prefix="sglang_peer_l1_", dir="/dev/shm")
    directory = L1Directory(path, rows, count, create=True)
    local_keys = keys(rank, count)
    directory.publish(local_keys, np.arange(1, count + 1))
    memory = PeerMemoryMap(
        group, rank, args.ranks, device, dist.group.WORLD, (path, rows, count)
    )
    peer = (rank + 1) % args.ranks
    remote = L1Directory(*memory.directory_infos[peer], create=False)
    copier = L1Copier(
        memory, layers=61, layer_group=args.layer_group, blocks=args.blocks, verify=True
    )
    forward = torch.cuda.current_stream()
    remote_keys = keys(peer, count)
    source_rows, gens = remote.probe(hash_words(remote_keys))
    destination = torch.arange((count + 1) * 256, (2 * count + 1) * 256, device=device)
    torch.cuda.synchronize()
    dist.barrier()
    timings = []
    for iteration in range(args.iterations + 1):
        assert remote.pin(source_rows, gens)
        plan = SimpleNamespace(
            rid=str(iteration),
            peer=peer,
            keys=remote_keys,
            rows=source_rows,
            indices=destination,
            directory=remote,
            reserved_rows=source_rows,
        )
        if args.skip_copy:
            jobs = torch.stack(
                (
                    torch.full_like(
                        source_rows_t := torch.from_numpy(source_rows).to(device), peer
                    ),
                    source_rows_t,
                    destination.reshape(-1, 256)[:, 0] // 256,
                ),
                dim=1,
            )
            error = torch.zeros(1, dtype=torch.int64, device=device)
            copier.module.copy(
                memory.descriptors,
                jobs,
                0,
                memory.layer_offsets[-1],
                args.blocks,
                error,
                True,
                forward.cuda_stream,
            )
            failed = bool(error.item())
            remote.unpin(source_rows)
            dist.barrier()
            memory.peer_storages.clear()
            dist.barrier()
            remote.close()
            directory.close()
            dist.destroy_process_group()
            assert not failed, "negative control: skipped copy detected"
            return
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        # Validate once, then measure the production path without the extra
        # full read/compare kernel. Drain the independent oracle first.
        copier.verify = iteration == 0
        forward.synchronize()
        start.record(copier.stream)
        batch = copier.submit([plan], forward)
        end.record(copier.stream)
        end.synchronize()
        if batch.error_cpu is not None:
            assert batch.error_cpu.item() == 0
        assert batch.rows_cpu.tolist() == list(range(count + 1, 2 * count + 1))
        # Independent expected-row oracle, not the same source pointer that
        # the copy/verification kernel used.
        for entry in entries:
            for layer, index in entry.layer_mapping.items():
                got = entry.components[0][index][count + 1 :, :1].flatten()
                expected = (
                    (
                        torch.arange(1, count + 1, device=device) * 17
                        + peer * 31
                        + layer * 7
                    )
                    % 251
                ).to(torch.uint8)
                assert torch.equal(got, expected)
        if iteration:
            timings.append(start.elapsed_time(end))
    dist.barrier()
    assert directory.pinned_refs == 0
    amount = count * sum(width * len(layers) for _, width, layers in layouts)
    ms = float(np.median(timings))
    print(
        json.dumps(
            dict(
                rank=rank,
                pages=count,
                blocks=args.blocks,
                layer_group=args.layer_group,
                ms=ms,
                GBps=amount / ms / 1e6,
            )
        ),
        flush=True,
    )
    memory.peer_storages.clear()
    dist.barrier()
    remote.close()
    directory.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--pages", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--layer-group", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--skip-copy", action="store_true")
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(args, port), nprocs=args.ranks)
