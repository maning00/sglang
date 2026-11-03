"""Manual two-process last-replica stress; deterministic ordering lives in unit tests."""

import argparse
import hashlib
import multiprocessing as mp

from sglang.srt.mem_cache.unified_cache.peer_l1.control import (
    CleanupQueue,
    OwnerPeer,
    OwnerService,
    full_keys,
)
from sglang.srt.mem_cache.unified_cache.peer_l1.directory import L1Directory


def key(i):
    return hashlib.sha256(str(i).encode()).hexdigest()


def _retire_process(connection, barrier, endpoint):
    directory = L1Directory(32, 4)
    service = OwnerService(directory, endpoint=endpoint)
    queue = CleanupQueue(timeout_s=0.1)
    try:
        connection.send((service.endpoint, service.epoch))
        endpoint, epoch = connection.recv()
        peer = OwnerPeer(
            endpoint=endpoint, epoch=epoch, pin_cap=4, cleanup=queue, timeout_s=0.2
        )
        writes = []
        for _ in range(64):
            service.publish([key(1)], [3])
            barrier.wait(10)
            service.retire([key(1)])
            found = peer.contains(full_keys([key(1)]))
            writes.append(found is None or not found[0])
            barrier.wait(10)
        connection.send(writes)
    finally:
        queue.close()
        service.close()
        directory.close()
        connection.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tcp", action="store_true")
    args = parser.parse_args()
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    pipes = [context.Pipe() for _ in range(2)]
    workers = [
        context.Process(
            target=_retire_process,
            args=(child, barrier, "tcp://127.0.0.1:0" if args.tcp else None),
        )
        for _, child in pipes
    ]
    try:
        for worker in workers:
            worker.start()
        for parent, _ in pipes:
            assert parent.poll(30)
        endpoints = [parent.recv() for parent, _ in pipes]
        for i, (parent, _) in enumerate(pipes):
            parent.send(endpoints[1 - i])
        for parent, _ in pipes:
            assert parent.poll(30)
        results = [parent.recv() for parent, _ in pipes]
        assert all(a or b for a, b in zip(*results))
        for worker in workers:
            worker.join(10)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
        for pair in pipes:
            for connection in pair:
                connection.close()


if __name__ == "__main__":
    main()
