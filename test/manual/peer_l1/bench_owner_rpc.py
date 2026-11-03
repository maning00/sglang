"""Owner RPC round trips with separate readers; TCP reuses each reader connection."""

import argparse
import hashlib
import json
import multiprocessing as mp
import time

import numpy as np

from sglang.srt.mem_cache.unified_cache.peer_l1.control import (
    Operation,
    OwnerPeer,
    OwnerService,
    full_keys,
)
from sglang.srt.mem_cache.unified_cache.peer_l1.directory import L1Directory


def keys(count):
    return [hashlib.sha256(str(i).encode()).hexdigest() for i in range(count)]


def owner(connection, transport, readers, pages):
    directory = L1Directory(pages + 1, readers * pages)
    service = OwnerService(
        directory, endpoint="tcp://127.0.0.1:0" if transport == "tcp" else None
    )
    try:
        service.publish(keys(pages), range(pages))
        connection.send((service.endpoint, service.epoch))
        connection.recv()
        assert directory.pinned_refs == 0
    finally:
        service.close()
        directory.close()


def reader(connection, barrier, endpoint, epoch, sizes, iterations, warmup):
    peer = OwnerPeer(
        endpoint=endpoint, epoch=epoch, pin_cap=max(sizes), cleanup=None, timeout_s=0.05
    )
    empty = full_keys([])
    seq = 0
    for pages in sizes:
        batch = full_keys(keys(pages))
        samples = {
            op.name: []
            for op in (
                Operation.ACQUIRE_EXACT,
                Operation.RELEASE,
                Operation.LOOKUP,
                Operation.CONTAINS,
            )
        }
        barrier.wait(30)
        for iteration in range(-warmup, iterations):
            if iteration == 0:
                barrier.wait(30)
                start = time.perf_counter_ns()
            seq += 1
            for op, payload, expected in (
                (Operation.ACQUIRE_EXACT, batch, pages),
                (Operation.RELEASE, empty, 0),
                (Operation.LOOKUP, batch, pages),
                (Operation.CONTAINS, batch, pages),
            ):
                before = time.perf_counter_ns()
                result = peer.request(op, payload, seq=seq)
                elapsed = time.perf_counter_ns() - before
                if result is None or len(result) != expected:
                    raise RuntimeError(f"{op.name} failed at {pages} pages")
                if op == Operation.CONTAINS and not result.all():
                    raise RuntimeError("CONTAINS lost a published page")
                if iteration >= 0:
                    samples[op.name].append(elapsed / 1000)
        end = time.perf_counter_ns()
        connection.send((pages, start, end, samples))


def run(args, transport):
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    source = ctx.Process(
        target=owner, args=(child, transport, args.readers, max(args.pages))
    )
    source.start()
    processes, pipes = [], []
    try:
        assert parent.poll(60), "owner did not start"
        endpoint, epoch = parent.recv()
        barrier = ctx.Barrier(args.readers)
        for _ in range(args.readers):
            read, write = ctx.Pipe()
            process = ctx.Process(
                target=reader,
                args=(
                    write,
                    barrier,
                    endpoint,
                    epoch,
                    args.pages,
                    args.iterations,
                    args.warmup,
                ),
            )
            process.start()
            processes.append(process)
            pipes.append(read)
        for pages in args.pages:
            for pipe in pipes:
                assert pipe.poll(120), "reader did not finish"
            results = [pipe.recv() for pipe in pipes]
            span = (max(r[2] for r in results) - min(r[1] for r in results)) / 1e9
            latencies = {}
            for op in results[0][3]:
                values = np.concatenate([r[3][op] for r in results])
                latencies[op] = dict(
                    zip(
                        ("p50_us", "p99_us", "max_us"),
                        np.percentile(values, [50, 99, 100]).tolist(),
                    )
                )
            print(
                json.dumps(
                    dict(
                        transport=transport,
                        readers=args.readers,
                        pages=pages,
                        iterations=args.iterations,
                        rpc_per_s=4 * args.readers * args.iterations / span,
                        latency=latencies,
                    )
                ),
                flush=True,
            )
        for process in processes:
            process.join(10)
            assert process.exitcode == 0
        parent.send("stop")
        source.join(10)
        assert source.exitcode == 0
    finally:
        for process in [*processes, source]:
            if process.is_alive():
                process.terminate()
                process.join()
        for pipe in [parent, child, *pipes]:
            pipe.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport", nargs="+", choices=["unix", "tcp"], default=["unix", "tcp"]
    )
    parser.add_argument("--readers", type=int, default=1)
    parser.add_argument("--pages", type=int, nargs="+", default=[64, 256, 2048])
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()
    for transport in args.transport:
        run(args, transport)


if __name__ == "__main__":
    main()
