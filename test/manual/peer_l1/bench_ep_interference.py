"""Phase 0a: the peer L1 copy against real mori EP dispatch/combine and a GEMM.

PYTHONPATH=python python test/manual/peer_l1/bench_ep_interference.py --blocks 8,16,32,64

Eight ranks, each reading --pages pages from rank+1 (ring) with the production
copier, while dispatch+combine runs at DSv4-Pro DEP8 shapes (bf16, hidden 7168,
top-6 of 384 experts) on another stream. Reports, per blocks value: EP alone, copy
alone, both together, copy next to a bf16 GEMM, and a peer memcpy reference.
Working sets (256 pages = 389 MB per copy, 117 MB EP/GEMM inputs) exceed L2 and
the 256 MB Infinity Cache, so the loops are not cache-hot.
"""

import argparse
import hashlib
import json
import os
import socket
import tempfile
from types import SimpleNamespace

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "16G")

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

# Unified fp8 Pro row geometry, as in bench_copy.py.
LAYOUTS = [
    (PoolName.DEEPSEEK_V4_C4, 32768, list(range(0, 60, 2))),
    (PoolName.DEEPSEEK_V4_C4_ROPE, 8192, list(range(0, 60, 2))),
    (PoolName.DEEPSEEK_V4_C128, 1024, list(range(1, 61, 2)) + [60]),
    (PoolName.DEEPSEEK_V4_C128_ROPE, 256, list(range(1, 61, 2)) + [60]),
    (PoolName.KV, 8448, list(range(0, 60, 2))),
]
PAGE_BYTES = sum(width * len(layers) for _, width, layers in LAYOUTS)
HIDDEN, TOPK, EXPERTS = 7168, 6, 384


def page_keys(rank, count):
    return [hashlib.sha256(f"{rank}:{p}".encode()).hexdigest() for p in range(count)]


def build_copy(rank, world, pages, device, inflight):
    rows = 2 * pages + 1
    # Copies are queued back to back and unpin asynchronously.
    pin_cap = pages * inflight
    entries = []
    for name, width, layers in LAYOUTS:
        buffers = [
            torch.randint(0, 255, (rows, width), dtype=torch.uint8, device=device)
            for _ in layers
        ]
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
    path = tempfile.mktemp(prefix="sglang_peer_l1_bench_", dir="/dev/shm")
    directory = L1Directory(path, rows, pin_cap, create=True)
    directory.publish(page_keys(rank, pages), np.arange(1, pages + 1))
    memory = PeerMemoryMap(
        group, rank, world, device, dist.group.WORLD, (path, rows, pin_cap)
    )
    peer = (rank + 1) % world
    remote = L1Directory(*memory.directory_infos[peer], create=False)
    source_rows, gens = remote.probe(hash_words(page_keys(peer, pages)))
    destination = torch.arange((pages + 1) * 256, (2 * pages + 1) * 256, device=device)
    plan = SimpleNamespace(
        peer=peer,
        keys=page_keys(peer, pages),
        rows=source_rows,
        indices=destination,
        directory=remote,
        reserved_rows=source_rows,
    )
    return memory, directory, remote, plan, gens


def build_ep(rank, world, tokens):
    import mori

    torch._C._distributed_c10d._register_process_group("mori_bench", dist.group.WORLD)
    mori.shmem.shmem_torch_process_group_init("mori_bench")
    config = mori.ops.EpDispatchCombineConfig(
        data_type=torch.bfloat16,
        rank=rank,
        world_size=world,
        hidden_dim=HIDDEN,
        scale_dim=0,
        scale_type_size=4,
        max_token_type_size=2,
        max_num_inp_token_per_rank=8192,
        num_experts_per_rank=EXPERTS // world,
        num_experts_per_token=TOPK,
        warp_num_per_block=16,
        block_num=80,
        max_total_recv_tokens=0,
        use_external_inp_buf=True,
        kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode,
        gpu_per_node=8,
        rdma_block_num=0,
        num_qp_per_pe=2,
        quant_type="none",
    )
    op = mori.ops.EpDispatchCombineOp(config)
    x = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
    ids = torch.rand(tokens, EXPERTS, device="cuda").topk(TOPK, dim=1).indices
    weights = torch.rand(tokens, TOPK, dtype=torch.float32, device="cuda")

    def step():
        out, _, _, _, _ = op.dispatch(x, weights, None, ids.to(torch.int32))
        op.combine(out, None, ids.to(torch.int32), call_reset=False)

    return step


def timed(stream, fn, iterations):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    with torch.cuda.stream(stream):
        start.record(stream)
        for _ in range(iterations):
            fn()
        end.record(stream)
    return start, end


def worker(rank, args, port):
    world = args.ranks
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group(
        "cpu:gloo,cuda:nccl", rank=rank, world_size=world, device_id=device
    )
    memory, directory, remote, plan, gens = build_copy(
        rank, world, args.pages, device, args.copy_iters + 2
    )
    ep_step = build_ep(rank, world, args.tokens)
    ep_stream = torch.cuda.Stream()
    gemm_stream = torch.cuda.Stream()
    a = torch.randn(8192, HIDDEN, dtype=torch.bfloat16, device=device)
    b = torch.randn(HIDDEN, 4096, dtype=torch.bfloat16, device=device)

    def gemm():
        torch.mm(a, b)

    results = []
    for blocks in args.blocks:
        copier = L1Copier(memory, layers=61, layer_group=8, blocks=blocks, verify=False)
        forward = torch.cuda.current_stream()

        def copy_once(tag):
            assert remote.pin(plan.rows, gens)
            p = SimpleNamespace(rid=tag, **vars(plan))
            return copier.submit([p], forward)

        def copy_loop(count=None, until=None):
            """Back-to-back copies, at most two in flight (the counter has three
            slots); stops after `count` copies or once event `until` completes."""
            start = torch.cuda.Event(enable_timing=True)
            start.record(copier.stream)
            outstanding, done = [], 0
            while True:
                if len(outstanding) == 2:
                    outstanding.pop(0).finish.synchronize()
                outstanding.append(copy_once(f"{blocks}:{done}"))
                done += 1
                if count is not None and done >= count:
                    break
                if until is not None and done >= 2 and until.query():
                    break
            end = torch.cuda.Event(enable_timing=True)
            end.record(copier.stream)
            end.synchronize()
            return start.elapsed_time(end) / done

        def against(stream, fn, iterations, key):
            """Time `fn` x iterations while copies run for its whole duration.
            Also records whether the two actually overlapped: streams that land
            on one hardware queue serialize, which reads as zero interference."""
            s, e = timed(stream, fn, iterations)
            copies_start = torch.cuda.Event(enable_timing=True)
            copies_start.record(copier.stream)
            copy_ms = copy_loop(until=e)
            e.synchronize()
            fn_end = s.elapsed_time(e)
            # Copies that started only after fn ended never overlapped it.
            row[f"overlap_{key}"] = s.elapsed_time(copies_start) < 0.5 * fn_end
            return fn_end / iterations, copy_ms

        # Warm everything once.
        copy_loop(count=1)
        timed(ep_stream, ep_step, 2)[1].synchronize()
        timed(gemm_stream, gemm, 2)[1].synchronize()
        dist.barrier()

        row = dict(rank=rank, blocks=blocks, pages=args.pages, tokens=args.tokens)
        s, e = timed(ep_stream, ep_step, args.ep_iters)
        e.synchronize()
        row["ep_ms"] = s.elapsed_time(e) / args.ep_iters
        dist.barrier()
        row["copy_ms"] = copy_loop(count=args.copy_iters)
        dist.barrier()
        row["ep_ms_with_copy"], row["copy_ms_with_ep"] = against(
            ep_stream, ep_step, args.ep_iters, "ep"
        )
        dist.barrier()
        s, e = timed(gemm_stream, gemm, args.gemm_iters)
        e.synchronize()
        row["gemm_ms"] = s.elapsed_time(e) / args.gemm_iters
        dist.barrier()
        row["gemm_ms_with_copy"], row["copy_ms_with_gemm"] = against(
            gemm_stream, gemm, args.gemm_iters, "gemm"
        )
        dist.barrier()
        for key in ("copy_ms", "copy_ms_with_ep", "copy_ms_with_gemm"):
            row[key.replace("ms", "GBps")] = args.pages * PAGE_BYTES / row[key] / 1e6
        results.append(row)

    # Peer memcpy reference: one large copy from the ring neighbour, all ranks at once.
    peer = (rank + 1) % world
    size = 1 << 30
    src = torch.empty(size, dtype=torch.uint8, device=torch.device("cuda", peer))
    dst = torch.empty(size, dtype=torch.uint8, device=device)
    dst.copy_(src)
    torch.cuda.synchronize()
    dist.barrier()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(5):
        dst.copy_(src, non_blocking=True)
    end.record()
    end.synchronize()
    memcpy_gbps = 5 * size / start.elapsed_time(end) / 1e6
    del src, dst

    torch.cuda.synchronize()
    dist.barrier()
    # Every reader's host callbacks have unpinned this rank's rows.
    assert directory.pinned_refs == 0, directory.pinned_refs
    gathered = [None] * world
    dist.all_gather_object(gathered, (results, memcpy_gbps))
    if rank == 0:
        for index in range(len(args.blocks)):
            rows = [g[0][index] for g in gathered]
            summary = {
                key: float(np.median([r[key] for r in rows]))
                for key in rows[0]
                if key not in ("rank",)
            }
            summary["copy_GBps_min_rank"] = min(r["copy_GBps"] for r in rows)
            summary["ep_slowdown"] = summary["ep_ms_with_copy"] / summary["ep_ms"]
            summary["gemm_slowdown"] = summary["gemm_ms_with_copy"] / summary["gemm_ms"]
            print(json.dumps(summary), flush=True)
        print(
            json.dumps(
                dict(memcpy_peer_GBps_median=float(np.median([g[1] for g in gathered])))
            ),
            flush=True,
        )
    memory.peer_storages.clear()
    dist.barrier()
    remote.close()
    directory.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--pages", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument(
        "--blocks", type=lambda s: [int(v) for v in s.split(",")], default=[32]
    )
    parser.add_argument("--ep-iters", type=int, default=20)
    parser.add_argument("--copy-iters", type=int, default=10)
    parser.add_argument("--gemm-iters", type=int, default=50)
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(args, port), nprocs=args.ranks)
