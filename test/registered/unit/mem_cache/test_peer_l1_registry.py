"""Discovery validates epoch/compatibility and never queries a scheduler on HTTP GET."""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import msgspec
import pytest

from sglang.srt.mem_cache.unified_cache.peer_l1.control import OwnerPeer, OwnerService
from sglang.srt.mem_cache.unified_cache.peer_l1.directory import L1Directory
from sglang.srt.mem_cache.unified_cache.peer_l1.registry import (
    DomainConfig,
    OwnerHello,
    OwnerManifest,
    PeerDescriptor,
    Resolver,
    merge_manifests,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def descriptor(service, rank=0, compatibility="weights-v1"):
    return PeerDescriptor(
        worker_id=f"test/p/prefill_l1/dp{rank}",
        instance_id="p",
        rank=rank,
        role="prefill_l1",
        epoch=service.epoch,
        kv_compatibility_id=compatibility,
        restart_group="p",
        pin_cap=4,
        control_endpoints=(service.endpoint,),
        data_capabilities=("ipc",),
        l2_domain="host",
        preservation="umbp/last_replica",
    )


def eventually(fn):
    deadline = time.monotonic() + 5
    while not fn():
        assert time.monotonic() < deadline
        time.sleep(0.01)


def test_resolver_checks_complete_manifest_against_authoritative_hello():
    directory = L1Directory(16, 4)
    service = OwnerService(directory, endpoint="tcp://127.0.0.1:0")
    desc = descriptor(service)
    service.describe(OwnerHello(descriptor=desc, data=b"private registration"))
    manifest = OwnerManifest(deployment="test", instance_id="p", owners=(desc,))
    state = [None]
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(503 if state[0] is None else 200)
            self.end_headers()
            if state[0] is not None:
                self.wfile.write(state[0])

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()

    def connect(d):
        calls.append(d)
        peer = OwnerPeer(
            endpoint=d.control_endpoints[0],
            epoch=d.epoch,
            pin_cap=d.pin_cap,
            cleanup=None,
            timeout_s=0.05,
        )
        return peer.hello(d)

    resolver = Resolver(
        seeds=[f"http://127.0.0.1:{server.server_port}"],
        deployment="test",
        instance_id="reader",
        compatibility_id="weights-v1",
        connect=connect,
    )
    try:
        assert not resolver.snapshot
        state[0] = msgspec.json.encode(manifest)
        resolver.refresh()
        eventually(lambda: desc.worker_id in resolver.snapshot)
        assert resolver.snapshot[desc.worker_id][1].data == b"private registration"
        # The HTTP record can be newer than the owner endpoint, or outright stale.
        bad = msgspec.structs.replace(desc, epoch=desc.epoch ^ 1)
        state[0] = msgspec.json.encode(msgspec.structs.replace(manifest, owners=(bad,)))
        resolver.refresh()
        eventually(lambda: calls[-1] == bad and not resolver.snapshot)
        compatible = msgspec.structs.replace(desc, preservation="umbp/new-policy")
        service.describe(OwnerHello(descriptor=compatible))
        incompatible = msgspec.structs.replace(
            desc,
            worker_id="test/p/prefill_l1/dp1",
            rank=1,
            kv_compatibility_id="weights-v2",
        )
        state[0] = msgspec.json.encode(
            msgspec.structs.replace(manifest, owners=(compatible, incompatible))
        )
        resolver.refresh()
        eventually(
            lambda: resolver.snapshot.get(desc.worker_id, (None,))[0] == compatible
        )
        assert len(resolver.snapshot) == 1
        assert incompatible not in calls
    finally:
        resolver.close()
        server.shutdown()
        thread.join()
        server.server_close()
        service.close()
        directory.close()


def test_frontend_snapshot_preserves_all_ranks_and_multi_tokenizer_payload(monkeypatch):
    import uuid

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from sglang.srt.entrypoints.peer_l1 import owner_snapshot, router
    from sglang.srt.managers.multi_tokenizer_mixin import (
        read_from_shared_memory,
        write_to_shared_memory,
    )

    directory = L1Directory(16, 4)
    service = OwnerService(directory)
    try:
        manifest = OwnerManifest(
            deployment="test",
            instance_id="p",
            owners=tuple(descriptor(service, i) for i in range(8)),
        )
        blob = msgspec.json.encode(manifest)
        info = {"peer_l1_owners": merge_manifests([{"peer_l1_owners": blob}] * 4)}
        name = "peer-test-" + uuid.uuid4().hex
        shm = write_to_shared_memory((None, None, info), name)
        try:
            _, _, received = read_from_shared_memory(name)
        finally:
            shm.close()
            shm.unlink()
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[owner_snapshot] = lambda: received["peer_l1_owners"]
        with TestClient(app) as client:
            assert client.get("/peer_l1/owners").content == blob
            received["peer_l1_owners"] = None
            assert client.get("/peer_l1/owners").status_code == 503
        with pytest.raises(ValueError):
            merge_manifests([info, {"peer_l1_owners": b"different"}])
    finally:
        service.close()
        directory.close()


def test_cross_instance_configuration_requires_explicit_identity():
    assert not DomainConfig().instance_id
    with pytest.raises(ValueError):
        DomainConfig(enabled=True, seeds=("http://peer",), kv_compatibility_id="w")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
