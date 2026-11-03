"""Two instances, two Gloo ranks each, with real HTTP snapshots and native owners."""

import hashlib
import json
import multiprocessing as mp
import socket
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import msgspec

from sglang.srt.mem_cache.unified_cache.peer_l1.control import (
    OwnerPeer,
    OwnerService,
    full_keys,
)
from sglang.srt.mem_cache.unified_cache.peer_l1.directory import L1Directory
from sglang.srt.mem_cache.unified_cache.peer_l1.registry import (
    DomainConfig,
    OwnerHello,
    OwnerManifest,
    PeerDescriptor,
    Resolver,
    topology,
)

KEY = hashlib.sha256(b"registry-check").hexdigest()


def free_ports(n):
    sockets = [socket.socket() for _ in range(n)]
    try:
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def worker(pipe, instance, rank, group_port, api_port, seeds):
    import torch.distributed as dist
    import uvicorn
    from fastapi import FastAPI

    from sglang.srt.entrypoints.peer_l1 import owner_snapshot, router
    from sglang.srt.runtime_context import get_resources

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{group_port}", rank=rank, world_size=2
    )
    parallel = SimpleNamespace(world_group=SimpleNamespace(cpu_group=dist.group.WORLD))
    with (
        patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
        patch("socket.gethostname", return_value=f"node-{rank}"),
    ):
        assert topology(DomainConfig()).instance_id == "node-0"
        get_resources().peer_l1_topology = None
        identity = topology(
            DomainConfig(
                enabled=True,
                instance_id=instance,
                kv_compatibility_id="weights",
                seeds=seeds,
            )
        )
    assert identity.instance_id == instance and identity.hosts == ("node-0", "node-1")
    directory = L1Directory(16, 4)
    service = OwnerService(directory, endpoint="tcp://127.0.0.1:0")
    snapshot = [None]
    http = thread = None

    def describe():
        descriptor = PeerDescriptor(
            worker_id=f"check/{instance}/prefill_l1/dp{rank}",
            instance_id=instance,
            rank=rank,
            role="prefill_l1",
            epoch=service.epoch,
            kv_compatibility_id="weights",
            restart_group=instance,
            pin_cap=4,
            control_endpoints=(service.endpoint,),
            data_capabilities=("ipc",),
            l2_domain="host",
            preservation="umbp/last_replica",
        )
        service.describe(OwnerHello(descriptor=descriptor))
        service.publish([KEY], [rank + 1])
        return descriptor

    descriptor = describe()

    def refresh_snapshot():
        owners = [None, None]
        dist.all_gather_object(owners, descriptor)
        snapshot[0] = msgspec.json.encode(
            OwnerManifest(
                deployment="check", instance_id=instance, owners=tuple(owners)
            )
        )

    try:
        refresh_snapshot()
        if rank == 0:
            app = FastAPI()
            app.include_router(router)
            app.dependency_overrides[owner_snapshot] = lambda: snapshot[0]
            http = uvicorn.Server(
                uvicorn.Config(app, host="127.0.0.1", port=api_port, log_level="error")
            )
            thread = threading.Thread(target=http.run)
            thread.start()
            deadline = time.monotonic() + 10
            while not http.started:
                assert time.monotonic() < deadline
                time.sleep(0.01)
        pipe.send(snapshot[0])
        while (command := pipe.recv()) != "stop":
            if command == "restart":
                endpoint = service.endpoint
                service.close()
                directory.close()
                directory = L1Directory(16, 4)
                service = OwnerService(directory, endpoint=endpoint)
                descriptor = describe()
            elif command == "refresh":
                refresh_snapshot()
            pipe.send("ok")
    finally:
        if http is not None:
            http.should_exit = True
            thread.join(10)
        service.close()
        directory.close()
        dist.destroy_process_group()
        pipe.close()


def wait_for(predicate):
    deadline = time.monotonic() + 10
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.02)


def main():
    ctx = mp.get_context("spawn")
    group_ports, api_ports = free_ports(2), free_ports(2)
    seeds = tuple(f"http://127.0.0.1:{p}" for p in api_ports)
    processes, pipes = [], []
    resolver = None
    try:
        for instance, group, port in zip(("a", "b"), group_ports, api_ports):
            for rank in range(2):
                parent, child = ctx.Pipe()
                process = ctx.Process(
                    target=worker,
                    args=(child, instance, rank, group, port, seeds),
                )
                process.start()
                child.close()
                processes.append(process)
                pipes.append(parent)
        for pipe in pipes:
            assert pipe.poll(60)
        ready = [pipe.recv() for pipe in pipes]
        assert ready[0] == ready[1] and ready[2] == ready[3]

        def connect(desc, previous=None):
            peer = OwnerPeer(
                endpoint=desc.control_endpoints[0],
                epoch=desc.epoch,
                pin_cap=desc.pin_cap,
                cleanup=None,
                timeout_s=0.05,
            )
            return peer if peer.hello(desc) is not None else None

        resolver = Resolver(
            seeds=seeds,
            deployment="check",
            instance_id="a",
            compatibility_id="weights",
            connect=connect,
        )
        wait_for(lambda: len(resolver.snapshot) == 2)
        old = {worker: desc.epoch for worker, (desc, _) in resolver.snapshot.items()}
        for desc, peer in resolver.snapshot.values():
            assert peer.request(1, full_keys([KEY])).tolist() == [desc.rank + 1]
        for pipe in pipes[2:]:
            pipe.send("restart")
        for pipe in pipes[2:]:
            assert pipe.poll(10) and pipe.recv() == "ok"
        for worker_id in old:
            resolver.refresh(worker_id)
        wait_for(lambda: not resolver.snapshot)
        for pipe in pipes[2:]:
            pipe.send("refresh")
        for pipe in pipes[2:]:
            assert pipe.poll(10) and pipe.recv() == "ok"
        resolver.refresh()
        wait_for(lambda: len(resolver.snapshot) == 2)
        assert all(
            old[worker] != desc.epoch for worker, (desc, _) in resolver.snapshot.items()
        )
        print(
            json.dumps(
                {
                    "result": "PASS",
                    "instances": 2,
                    "ranks_per_instance": 2,
                    "rank0_hostname_default": True,
                    "complete_manifests": True,
                    "stale_epoch_rejected": True,
                    "restart_rediscovered": True,
                    "real_http_and_owner_rpc": True,
                    "model_servers_started": False,
                }
            )
        )
    finally:
        if resolver is not None:
            resolver.close()
        for pipe, process in zip(pipes, processes):
            if process.is_alive():
                try:
                    pipe.send("stop")
                except (BrokenPipeError, EOFError):
                    pass
        for process in processes:
            process.join(10)
            if process.is_alive():
                process.terminate()
                process.join()
        for pipe in pipes:
            pipe.close()
    assert all(process.exitcode == 0 for process in processes)


if __name__ == "__main__":
    main()
