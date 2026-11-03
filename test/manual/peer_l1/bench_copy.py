"""Two/eight-process IPC copy validation and a small bandwidth sweep.

PYTHONPATH=python python test/manual/peer_l1/bench_copy.py --ranks 2 --pages 256
--skip-copy is the negative control: it must fail byte verification.
--engine sdma copies through mori anvil SDMA queues instead of the copy kernel.
Exercises native grants and callback cleanup with private owner state,
including a pause beyond the test lease before any device read.
"""

import argparse
import hashlib
import json
import socket
import time
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
from sglang.srt.mem_cache.unified_cache.peer_l1.channels import NativeOwnerChannel
from sglang.srt.mem_cache.unified_cache.peer_l1.control import (
    CleanupQueue,
    OwnerPeer,
    OwnerService,
    full_keys,
)
from sglang.srt.mem_cache.unified_cache.peer_l1.copier import L1Copier, PeerMemoryMap
from sglang.srt.mem_cache.unified_cache.peer_l1.directory import L1Directory


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
    directory = L1Directory(rows, count)
    owner = OwnerService(directory, lease_s=0.02)
    cleanup = CleanupQueue(timeout_s=0.2)
    local_keys = keys(rank, count)
    owner.publish(local_keys, np.arange(1, count + 1))
    memory = PeerMemoryMap(
        group,
        rank,
        args.ranks,
        device,
        dist.group.WORLD,
        (owner.endpoint, owner.epoch),
    )
    peer = (rank + 1) % args.ranks
    control = OwnerPeer(
        endpoint=memory.directory_infos[peer][0],
        epoch=memory.directory_infos[peer][1],
        pin_cap=count,
        cleanup=cleanup,
        timeout_s=0.2,
    )
    dist.barrier()
    copier = L1Copier(
        memory,
        layers=61,
        layer_group=args.layer_group,
        blocks=args.blocks,
        verify=True,
        engine=args.engine,
        peer_devices={p: p for p in range(args.ranks) if p != rank},
        sdma_channels=args.sdma_channels,
    )
    channel = NativeOwnerChannel(peer=control, worker_id=f"owner-{peer}")
    forward = torch.cuda.current_stream()
    remote_keys = keys(peer, count)
    destination = torch.arange((count + 1) * 256, (2 * count + 1) * 256, device=device)
    torch.cuda.synchronize()
    dist.barrier()
    timings = []
    for iteration in range(args.iterations + 1):
        identities = full_keys(remote_keys)
        split = max(1, count // 2)
        grants = [
            channel.acquire_exact(part)
            for part in (identities[:split], identities[split:])
            if len(part)
        ]
        assert all(grant is not None for grant in grants)
        source_rows = np.concatenate([grant.rows for grant in grants])
        item = grants[0].cleanup
        dist.barrier()
        # RPC connections have closed, and the lease has elapsed before any
        # device read. Neither event may release the source rows.
        time.sleep(0.04)
        assert directory.pinned_refs == count
        assert not directory.freeze()
        dist.barrier()
        plan = SimpleNamespace(
            rid=str(iteration),
            peer=peer,
            keys=remote_keys,
            rows=source_rows,
            indices=destination,
            cleanup=item,
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
            for grant in grants:
                item = grant.cleanup
                cleanup.enqueue(item)
            dist.barrier()
            memory.peer_storages.clear()
            dist.barrier()
            cleanup.close()
            owner.close()
            directory.close()
            dist.destroy_process_group()
            assert not failed, "negative control: skipped copy detected"
            return
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        # Validate once, then measure the production path without the extra
        # full read/compare kernel. Drain the independent oracle first.
        copier.verify = iteration == 0
        forward.synchronize()
        start.record(copier.stream)
        plans, offset = [], 0
        for grant in grants:
            grant_rows, grant_cleanup = grant.rows, grant.cleanup
            stop = offset + len(grant_rows)
            plans.append(
                SimpleNamespace(
                    **{
                        **vars(plan),
                        "rid": f"{iteration}:{offset}",
                        "keys": remote_keys[offset:stop],
                        "rows": grant_rows,
                        "indices": destination[offset * 256 : stop * 256],
                        "cleanup": grant_cleanup,
                    }
                )
            )
            offset = stop
        batch = copier.submit(plans, forward)
        end.record(copier.stream)
        end.synchronize()
        assert batch.quiesced()
        assert len(batch.resolved_layers()) == group.num_layers
        assert not batch.failed_requests()
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
        deadline = time.monotonic() + 5
        while cleanup.pending:
            assert time.monotonic() < deadline, "source cleanup did not drain"
            time.sleep(0.001)
        dist.barrier()
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
                engine=args.engine,
                sdma_channels=args.sdma_channels,
                ms=ms,
                GBps=amount / ms / 1e6,
            )
        ),
        flush=True,
    )
    memory.peer_storages.clear()
    dist.barrier()
    cleanup.close()
    owner.close()
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
    parser.add_argument("--engine", default="kernel", choices=["kernel", "sdma"])
    parser.add_argument("--sdma-channels", type=int, default=2)
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(args, port), nprocs=args.ranks)
