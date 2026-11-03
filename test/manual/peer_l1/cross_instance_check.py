"""Model-free two-process peer L1 read through real Indexer, HTTP, TCP and MoRI.

Run on two visible GPUs with the unified fp8 layout environment from the lend
round-trip check. Faults combine native submissions/statuses with gated GPU writes;
no NIC firmware or physical link is modified.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import socket
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch
from decode_lend_roundtrip_check import PAGE, PAGES, make_pool

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.peer_l1 import linker as linker_module
from sglang.srt.runtime_context import get_context

KEYS = [hashlib.sha256(f"cross-page-{i}".encode()).hexdigest() for i in range(4)]


def free_ports(count):
    sockets = [socket.socket() for _ in range(count)]
    try:
        for s in sockets:
            s.bind(("127.0.0.1", 0))
        return [s.getsockname()[1] for s in sockets]
    finally:
        for s in sockets:
            s.close()


def wait_for(predicate, seconds=20):
    deadline = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.02)


def worker(
    pipe,
    device,
    api_port,
    other_port,
    group_port,
    indexer,
    deployment,
    same_instance,
    isolated,
):
    import torch.distributed as dist
    import uvicorn
    from fastapi import FastAPI

    from sglang.srt.entrypoints.peer_l1 import owner_snapshot, router

    ordinal = 0 if isolated else device
    torch.cuda.set_device(ordinal)
    get_context().override_server_args(
        dsv4_attn_backend="flashmla", enable_deepseek_v4_fp4_indexer=True
    ).install()
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{group_port}",
        rank=device if same_instance else 0,
        world_size=2 if same_instance else 1,
    )
    cache_group = (
        [dist.new_group([i]) for i in range(2)][device]
        if same_instance
        else dist.group.WORLD
    )
    pool = make_pool(ordinal)
    alloc = PagedTokenToKVPoolAllocator(
        PAGES * PAGE, PAGE, torch.int64, f"cuda:{ordinal}", pool, True
    )
    params = SimpleNamespace(
        page_size=PAGE,
        token_to_kv_pool_allocator=alloc,
        tp_cache_group=cache_group,
        mtp_draft_device_pools=None,
        req_to_token_pool=None,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
    )
    parallel = SimpleNamespace(world_group=SimpleNamespace(cpu_group=dist.group.WORLD))
    config = {
        "indexer": {"endpoint": indexer, "deployment": deployment},
        "domain": {
            "enabled": True,
            "instance_id": f"instance-{device}",
            "kv_compatibility_id": "byte-check",
            "listen_host": "127.0.0.1",
            "seeds": [f"http://127.0.0.1:{other_port}"],
        },
    }
    if same_instance:
        config["domain"] = {"listen_host": "127.0.0.1"}
    with (
        patch.object(linker_module, "get_parallel", return_value=parallel),
        patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
    ):
        link = linker_module.PeerL1Linker(
            SimpleNamespace(enable_memory_saver=False),
            params,
            components={ComponentType.FULL, ComponentType.SWA},
            extra_config=config,
        )
    link.forward_stream = torch.cuda.current_stream()
    app = FastAPI()
    app.include_router(router)
    from sglang.srt.utils.auth import add_api_key_middleware

    add_api_key_middleware(app, api_key="discovery-check", admin_api_key=None)
    serving = [True]
    app.dependency_overrides[owner_snapshot] = lambda: (
        msgspec.json.encode(link.manifest) if serving[0] else None
    )
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=api_port, log_level="error")
    )
    api = threading.Thread(target=server.run)
    api.start()
    wait_for(lambda: server.started)
    if device == 0:
        assert alloc.alloc(len(KEYS) * PAGE) is not None
        for region, buffer in enumerate(link.memory.buffers):
            raw = buffer.view(torch.uint8).reshape(buffer.shape[0], -1)
            for page in range(1, len(KEYS) + 1):
                raw[page].fill_((region * 7 + page * 17) % 251)
        torch.cuda.synchronize()
        link.control.publish(KEYS, list(range(1, len(KEYS) + 1)))
        link.publisher.publish(
            [(key, KEYS[i - 1] if i else None) for i, key in enumerate(KEYS)]
        )
    pipe.send(
        {
            "epoch": link.control.epoch,
            "regions": len(link.memory.buffers),
            "gpus": [d.gpu for d in link.devices],
        }
    )
    sequence = 0
    queued = None
    try:
        while (command := pipe.recv()) != "stop":
            if command in ("seed-down", "seed-up", "evict", "republish"):
                if command.startswith("seed-"):
                    serving[0] = command == "seed-up"
                elif command == "evict":
                    link.control.retire(KEYS)
                else:
                    link.control.publish(KEYS, list(range(1, len(KEYS) + 1)))
                pipe.send({"command": command})
                continue
            if command == "miss":
                probe = PoolTransfer(name=PoolName.KV, keys=KEYS, indexer_eligible=True)
                assert len(link.lookup("miss", [probe])) == len(KEYS)
                before = dict(link.resolver.snapshot)
                assert not link.reserve("miss", len(KEYS))
                assert not link.resolver._invalid
                time.sleep(2.2)
                assert all(
                    link.resolver.snapshot[w][1] is s for w, (_, s) in before.items()
                )
                assert all(not s.owner.peer.retired for _, s in before.values())
                pipe.send({"command": command, "sessions_preserved": True})
                continue
            if command == "seed-flap":
                before = dict(link.resolver.snapshot)
                failures = []
                original = link.resolver._read

                def observed_read(seed):
                    try:
                        return original(seed)
                    except Exception:
                        failures.append(seed)
                        raise

                link.resolver._read = observed_read
                wait_for(lambda: len(failures) >= 2)
                assert all(
                    link.resolver.snapshot[w][1] is s for w, (_, s) in before.items()
                )
                assert all(not s.owner.peer.retired for _, s in before.values())
                link.resolver._read = original
                pipe.send({"command": command, "sessions_preserved": True})
                continue
            if command == "reserve":
                rid = f"read-{sequence}"
                sequence += 1
                probe = PoolTransfer(name=PoolName.KV, keys=KEYS, indexer_eligible=True)
                wait_for(lambda: len(link.lookup(rid, [probe])) == len(KEYS))
                assert link.reserve(rid, len(KEYS))
                queued = (rid, alloc.alloc(len(KEYS) * PAGE))
                pipe.send({"owner_epoch": link._reservations[rid].grant.epoch})
                continue
            if command in ("read", "timeout", "partial", "late-success", "old-grant"):
                started = time.monotonic()
                if queued is None:
                    rid = f"read-{sequence}"
                    sequence += 1
                    probe = PoolTransfer(
                        name=PoolName.KV, keys=KEYS, indexer_eligible=True
                    )
                    wait_for(lambda: len(link.lookup(rid, [probe])) == len(KEYS))
                    assert link.reserve(rid, len(KEYS))
                    queued = (rid, alloc.alloc(len(KEYS) * PAGE))
                rid, target = queued
                queued = None
                pages = (target.reshape(-1, PAGE)[:, 0] // PAGE).cpu().tolist()
                real_engine = link.rdma.engine if link.rdma is not None else None
                released = threading.Event()
                late = None
                quarantined_before = len(link.quarantine.pages)
                if command in ("timeout", "partial", "late-success"):
                    from mori import cpp

                    receipt = link._reservations[rid]
                    receipt.grant = msgspec.structs.replace(
                        receipt.grant, acquired_before=time.monotonic(), lease_s=0.4
                    )

                    def post(local, lo, remote, ro, sizes, statuses, ids):
                        if command == "late-success":
                            real_engine._engine.BatchRead(
                                local, lo, remote, ro, sizes, statuses, ids
                            )
                            if len(local) < 16:
                                for status in statuses:
                                    status.Wait()
                                    assert status.Succeeded()
                                time.sleep(0.3)
                        elif command == "partial":
                            real_engine._engine.BatchRead(
                                local[:1],
                                lo[:1],
                                remote[:1],
                                ro[:1],
                                sizes[:1],
                                statuses[:1],
                                ids[:1],
                            )
                            statuses[1].Update(
                                cpp.StatusCode.ERR_RDMA_OP,
                                "injected sibling QP failure",
                            )
                        else:
                            for status in statuses:
                                status.SetCode(cpp.StatusCode.IN_PROGRESS)

                    link.rdma.engine = SimpleNamespace(
                        _engine=SimpleNamespace(BatchRead=post),
                        allocate_transfer_uid=real_engine.allocate_transfer_uid,
                    )

                    def delayed_write():
                        torch.cuda.set_device(ordinal)
                        released.wait()
                        with torch.cuda.stream(torch.cuda.Stream(priority=-1)):
                            for buffer in link.memory.buffers:
                                raw = buffer.view(torch.uint8).reshape(
                                    buffer.shape[0], -1
                                )
                                for page in pages:
                                    raw[page].fill_(7)
                            torch.cuda.synchronize()

                    if command != "late-success":
                        late = threading.Thread(target=delayed_write)
                        late.start()
                link.load(
                    rid,
                    [
                        PoolTransfer(
                            name=PoolName.KV,
                            keys=KEYS,
                            device_indices=target,
                            indexer_eligible=True,
                        )
                    ],
                )
                batch_index = link.start_layer_wise_loading()
                failures = link.finish_loads(batch_index, [rid])
                if link.rdma is not None:
                    link.rdma.engine = real_engine
                if command == "read":
                    assert not failures
                    for region, buffer in enumerate(link.memory.buffers):
                        raw = buffer.view(torch.uint8).reshape(buffer.shape[0], -1)
                        for i, page in enumerate(pages, 1):
                            assert torch.all(raw[page] == (region * 7 + i * 17) % 251)
                else:
                    assert failures == {rid}
                if command == "late-success":
                    assert len(link.quarantine.pages) == quarantined_before
                    assert not link.resolver._invalid
                alloc.free(target)
                if late is not None:
                    assert set(pages) <= link.quarantine.pages
                    fresh = alloc.alloc(len(KEYS) * PAGE)
                    fresh_pages = (fresh.reshape(-1, PAGE)[:, 0] // PAGE).cpu().tolist()
                    assert not set(fresh_pages) & set(pages)
                    for buffer in link.memory.buffers:
                        raw = buffer.view(torch.uint8).reshape(buffer.shape[0], -1)
                        for page in fresh_pages:
                            raw[page].fill_(99)
                    torch.cuda.synchronize()
                    released.set()
                    late.join(10)
                    assert not late.is_alive()
                    for buffer in link.memory.buffers:
                        raw = buffer.view(torch.uint8).reshape(buffer.shape[0], -1)
                        for page in fresh_pages:
                            assert torch.all(raw[page] == 99)
                    alloc.free(fresh)
                pipe.send(
                    {
                        "command": command,
                        "failed": sorted(failures),
                        "quarantined_pages": len(link.quarantine.pages),
                        "elapsed_ms": (time.monotonic() - started) * 1000,
                    }
                )
    finally:
        server.should_exit = True
        api.join(10)
        link.close()
        dist.destroy_process_group()
        pipe.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indexer", required=True)
    parser.add_argument("--isolated-gpus", action="store_true")
    parser.add_argument("--same-instance", action="store_true")
    args = parser.parse_args()
    context = mp.get_context("spawn")
    api_ports = free_ports(2)
    group_ports = free_ports(2)
    deployment = f"cross-check-{time.time_ns()}"
    workers = []

    def start(device, wait=True):
        parent, child = context.Pipe()
        process = context.Process(
            target=worker,
            args=(
                child,
                device,
                api_ports[device],
                api_ports[1 - device],
                group_ports[0] if args.same_instance else group_ports[device],
                args.indexer,
                deployment,
                args.same_instance,
                args.isolated_gpus,
            ),
        )
        visible = os.environ.get("ROCR_VISIBLE_DEVICES")
        if args.isolated_gpus:
            assert visible is not None and len(visible.split(",")) >= 2
            os.environ["ROCR_VISIBLE_DEVICES"] = visible.split(",")[device]
        try:
            process.start()
        finally:
            if visible is not None:
                os.environ["ROCR_VISIBLE_DEVICES"] = visible
        child.close()
        if not wait:
            return process, parent, None
        assert parent.poll(120), "startup timed out"
        ready = parent.recv()
        return process, parent, ready

    try:
        if args.same_instance:
            workers.extend([start(0, wait=False), start(1, wait=False)])
            for i, (process, pipe, _) in enumerate(workers):
                assert pipe.poll(120), "instance startup timed out"
                workers[i] = (process, pipe, pipe.recv())
        else:
            workers.append(start(0))
            workers.append(start(1))
        reader = workers[1][1]
        results = []
        if not args.same_instance:
            for endpoint, command in (
                (reader, "read"),
                (workers[0][1], "evict"),
                (reader, "miss"),
                (workers[0][1], "republish"),
                (reader, "read"),
                (workers[0][1], "seed-down"),
                (reader, "seed-flap"),
                (workers[0][1], "seed-up"),
                (reader, "read"),
                (reader, "late-success"),
            ):
                endpoint.send(command)
                assert endpoint.poll(60), command
                results.append(endpoint.recv())
        commands = (
            ("read",)
            if args.same_instance
            else ("read", "timeout", "partial", "read", "reserve")
        )
        for command in commands:
            reader.send(command)
            assert reader.poll(60), command
            results.append(reader.recv())
        if not args.same_instance:
            old_epoch = workers[0][2]["epoch"]
            workers[0][1].send("stop")
            workers[0][0].join(30)
            assert workers[0][0].exitcode == 0
            workers[0][1].close()
            workers[0] = start(0)
            assert workers[0][2]["epoch"] != old_epoch
            for command in ("old-grant", "read"):
                reader.send(command)
                assert reader.poll(60), command
                results.append(reader.recv())
        print(
            json.dumps(
                {
                    "result": "PASS",
                    "regions": workers[0][2]["regions"],
                    "gpu_uuids": [w[2]["gpus"] for w in workers],
                    "results": results,
                    "physical_nic_fault_injected": False,
                    "same_instance": args.same_instance,
                    "isolated_gpus": args.isolated_gpus,
                }
            ),
            flush=True,
        )
    finally:
        for process, pipe, _ in workers:
            if process.is_alive():
                try:
                    pipe.send("stop")
                except (BrokenPipeError, EOFError):
                    pass
        for process, pipe, _ in workers:
            process.join(30)
            if process.is_alive():
                process.terminate()
                process.join()
            pipe.close()


if __name__ == "__main__":
    main()
