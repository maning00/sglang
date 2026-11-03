"""Ownership and discovery failure properties, using real host sockets."""

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from sglang.srt.mem_cache.unified_cache.peer_l1.control import (
    CleanupQueue,
    Operation,
    OwnerPeer,
    OwnerService,
    full_keys,
)
from sglang.srt.mem_cache.unified_cache.peer_l1.directory import L1Directory
from sglang.srt.mem_cache.unified_cache.peer_l1.indexer import (
    Placements,
    report_batches,
)
from sglang.srt.mem_cache.unified_cache.peer_l1.lend.pool import LendPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def key(i):
    return hashlib.sha256(str(i).encode()).hexdigest()


def eventually(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.005)


def cleanup_drained(peer, pool):
    peer.flush_cleanup()
    while peer.socket.poll(0):
        peer.poll()
    return not peer._cleanup and pool.pinned == 0


@pytest.fixture(params=[None, "tcp://127.0.0.1:0", "tcp://[::1]:0"])
def owner(request):
    directory = L1Directory(32, 4)
    service = OwnerService(directory, lease_s=0.02, endpoint=request.param)
    queue = CleanupQueue(timeout_s=0.02)
    peer = OwnerPeer(
        endpoint=service.endpoint,
        epoch=service.epoch,
        pin_cap=4,
        cleanup=queue,
        timeout_s=0.05,
    )
    try:
        yield directory, service, queue, peer
    finally:
        queue.close()
        service.close()
        directory.close()


def test_owner_timeout_and_disconnect_do_not_release_source(owner):
    directory, service, queue, peer = owner
    service.publish([key(1)], [3])
    rows = peer.request(Operation.ACQUIRE_EXACT, full_keys([key(1)]), seq=1)
    assert rows.tolist() == [3]
    assert peer.request(
        Operation.ACQUIRE_EXACT, full_keys([key(1)]), seq=1
    ).tolist() == [3]
    assert directory.pinned_refs == 1
    time.sleep(0.04)
    # Each RPC closes its connection. Neither close nor expiry authorizes reuse.
    assert (
        peer.request(Operation.ACQUIRE_EXACT, full_keys([key(1)]), seq=2).tolist() == []
    )
    service.retire([key(1)])
    assert not peer.contains(full_keys([key(1)]))[0]
    assert not directory.freeze()
    with pytest.raises(RuntimeError, match="pinned"):
        service.publish([key(2)], [3])
    queue.enqueue(queue.prepare(peer, 1))
    eventually(lambda: queue.pending == 0 and directory.pinned_refs == 0)
    assert directory.freeze()
    directory.thaw()
    service.publish([key(2)], [3])
    assert peer.lookup(full_keys([key(2)])) == 1


def test_cancel_before_acquire_and_repeated_cleanup(owner):
    directory, service, queue, peer = owner
    service.publish([key(1)], [3])
    assert peer.request(4, full_keys([]), seq=8).tolist() == []
    assert (
        peer.request(Operation.ACQUIRE_EXACT, full_keys([key(1)]), seq=8).tolist() == []
    )
    assert directory.pinned_refs == 0
    assert peer.request(
        Operation.ACQUIRE_EXACT, full_keys([key(1)]), seq=9
    ).tolist() == [3]
    for _ in range(2):
        queue.enqueue(queue.prepare(peer, 9, cancel=True))
    eventually(lambda: queue.pending == 0 and directory.pinned_refs == 0)
    assert (
        peer.request(Operation.ACQUIRE_EXACT, full_keys([key(1)]), seq=9).tolist() == []
    )


def test_full_key_authority_and_pin_cap(owner):
    directory, service, _queue, peer = owner
    keys = [key(i) for i in range(5)]
    service.publish(keys, range(5))
    assert peer.lookup(full_keys(keys)) == 4
    collision = keys[0][:32] + ("0" if keys[0][32] != "0" else "1") + keys[0][33:]
    assert peer.lookup(full_keys([collision])) == 0
    assert peer.acquire(full_keys(keys)) is None
    assert directory.pinned_refs == 0
    assert _queue.pending == 0


def test_collisions_churn_and_cap(owner):
    directory, service, queue, peer = owner
    # Same hash bucket, but distinct full keys; repeated retirement reuses slots.
    keys = ["0" * 16 + f"{i:016x}" + "0" * 32 for i in range(5)]
    for _ in range(100):
        service.publish(keys, range(5))
        assert peer.lookup(full_keys(keys)) == 4
        rows, grant = peer.acquire(full_keys(keys[:4]))
        assert rows.tolist() == list(range(4))
        assert peer.acquire(full_keys(keys[4:])) is None
        service.retire(keys)
        assert peer.lookup(full_keys(keys)) == 0
        queue.enqueue(grant)
        eventually(lambda: queue.pending == 0 and directory.pinned_refs == 0)


def test_retire_order_requires_at_least_one_writer(owner):
    directory, service, _queue, peer = owner
    other_directory = L1Directory(32, 4)
    other = OwnerService(other_directory)
    other_peer = OwnerPeer(
        endpoint=other.endpoint,
        epoch=other.epoch,
        pin_cap=4,
        cleanup=None,
        timeout_s=0.2,
    )
    try:
        service.publish([key(1)], [3])
        other.publish([key(1)], [3])
        service.retire([key(1)])
        assert other_peer.contains(full_keys([key(1)])).tolist() == [True]
        other.retire([key(1)])
        assert peer.contains(full_keys([key(1)])).tolist() == [False]
    finally:
        other.close()
        other_directory.close()


def test_lend_acquire_arbitrary_pages_cancel_and_expiry():
    pool = LendPool(1000, clock=lambda: 0)
    pool.add_free(range(1, 10))
    records = pool.allocate("writer", 4)
    for i, record in enumerate(records):
        record[2] = key(i)
    assert all(pool.commit("writer", [[*r, None] for r in records]))
    lease, granted = pool.acquire("reader", 3, [key(0), key(2)])
    assert [r[2] for r in granted] == [key(0), key(2)]
    assert pool.acquire("reader", 3, [key(0), key(2)]) == [lease, granted]
    assert pool.pinned == 2
    pool.cancel_acquire("reader", 4)
    assert pool.acquire("reader", 4, [key(1)]) == [0, []]
    pool.unpin("reader", lease)
    assert pool.acquire("reader", 3, [key(0)]) == [0, []]
    lease, _ = pool.acquire("reader", 5, [key(0)])
    pool.clock = lambda: 31
    pool.expire()
    assert pool.pinned == 0 and not pool._acquires


def test_report_split_preserves_parent_edges():
    records = [(key(i), key(i - 1) if i else None) for i in range(17000)]
    # F3 can report sparse interior pages, including parents absent from this owner.
    records.extend((key(20000 + i), key(19000 + i)) for i in range(300))
    from sglang.srt.mem_cache.unified_cache.peer_l1.indexer import block_hash

    expected = {block_hash(k): None if p is None else block_hash(p) for k, p in records}
    actual = {}
    for batch in report_batches(records):
        assert len(batch) <= 256 and sum(len(a.hashes) for a in batch) <= 16384
        for action in batch:
            parent = (
                action.parent_block_hash
                if action.HasField("parent_block_hash")
                else None
            )
            for child in action.hashes:
                actual[child] = parent
                parent = child
    assert actual == expected


def test_cleanup_survives_lost_grant_send_failure_and_lost_ack(owner):
    import socket
    import struct
    import threading
    import uuid
    from urllib.parse import urlsplit

    directory, service, queue, _peer = owner
    endpoint = "l1-fault-" + uuid.uuid4().hex
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind("\0" + endpoint)
    listener.listen()
    listener.settimeout(0.05)
    allow_cleanup, stop = threading.Event(), threading.Event()
    lost_ack, lost_grant = [False], [False]

    def receive(connection, n):
        result = b""
        while len(result) < n:
            chunk = connection.recv(n - len(result))
            if not chunk:
                raise EOFError
            result += chunk
        return result

    def proxy():
        while not stop.is_set():
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                continue
            with connection:
                header = receive(connection, 48)
                _, op, _, _, _, n = struct.unpack("=6Q", header)
                payload = receive(connection, n * 32)
                if op == Operation.CLEANUPS and not allow_cleanup.is_set():
                    continue
                address = urlsplit(service.endpoint)
                family = (
                    socket.AF_INET6
                    if address.hostname and ":" in address.hostname
                    else socket.AF_INET
                )
                with socket.socket(
                    family if address.scheme == "tcp" else socket.AF_UNIX,
                    socket.SOCK_STREAM,
                ) as target:
                    target.connect(
                        (address.hostname, address.port)
                        if address.scheme == "tcp"
                        else "\0" + service.endpoint
                    )
                    target.sendall(header + payload)
                    reply = receive(target, 16)
                    _, count = struct.unpack("=2Q", reply)
                    reply += receive(target, count * 8)
                if op == Operation.ACQUIRE_EXACT and not lost_grant[0]:
                    lost_grant[0] = True
                    continue
                if op == Operation.CLEANUPS and not lost_ack[0]:
                    lost_ack[0] = True
                    continue
                connection.sendall(reply)

    thread = threading.Thread(target=proxy)
    thread.start()
    peer = OwnerPeer(
        endpoint=endpoint, epoch=service.epoch, pin_cap=4, cleanup=queue, timeout_s=0.1
    )
    try:
        service.publish([key(1)], [3])
        assert peer.acquire(full_keys([key(1)])) is None
        eventually(lambda: queue.pending > 0)
        assert directory.pinned_refs == 1
        allow_cleanup.set()
        eventually(lambda: queue.pending == 0 and directory.pinned_refs == 0)
        assert lost_ack[0]
        assert (
            peer.request(
                Operation.ACQUIRE_EXACT, full_keys([key(1)]), seq=peer.seq
            ).tolist()
            == []
        )
    finally:
        stop.set()
        thread.join()
        listener.close()


def test_publisher_recovers_missing_revoke_restart_and_concurrent_retire():
    import threading

    import grpc

    from sglang.srt.grpc import kv_indexer_pb2 as pb
    from sglang.srt.mem_cache.unified_cache.peer_l1.indexer import (
        IndexerClient,
        IndexerConfig,
        PlacementPublisher,
        block_hash,
    )

    records, parents = {}, {}
    lock = threading.Lock()
    fail_revoke, fail_report = [False], [False]
    pause_clear = [False]
    entered, resume = threading.Event(), threading.Event()

    def apply(request, context):
        assert not request.worker_address
        assert len(request.actions) <= 256
        assert sum(len(a.hashes) for a in request.actions) <= 16384
        with lock:
            for action in request.actions:
                if action.type == pb.ACTION_REPORT:
                    if fail_report[0]:
                        fail_report[0] = False
                        context.abort(
                            grpc.StatusCode.UNAVAILABLE, "interrupted recovery"
                        )
                    parent = (
                        action.parent_block_hash
                        if action.HasField("parent_block_hash")
                        else None
                    )
                    for child in action.hashes:
                        if child in parents and parents[child] != parent:
                            context.abort(
                                grpc.StatusCode.INVALID_ARGUMENT, "conflicting parent"
                            )
                        parents[child] = parent
                        records[child] = parent
                        parent = child
                elif action.type == pb.ACTION_REVOKE:
                    if fail_revoke[0]:
                        fail_revoke[0] = False
                        context.abort(grpc.StatusCode.UNAVAILABLE, "lost revoke")
                    for child in action.hashes:
                        records.pop(child, None)
                else:
                    records.clear()
        if pause_clear[0] and any(
            a.type == pb.ACTION_CLEAR_ALL_AT_TIER for a in request.actions
        ):
            entered.set()
            assert resume.wait(5)
        return pb.ApplyExternalKvBatchResponse()

    def match(request, context):
        assert not request.count_as_hit
        with lock:
            held = [h for h in request.hashes if h in records]
        return pb.MatchExternalKvResponse(
            matches=[
                pb.ExternalKvNodeMatch(
                    worker_id="test/lend/node/dp0",
                    hashes_by_tier=[pb.TierHashes(tier=pb.TIER_HBM, hashes=held)],
                )
            ]
        )

    server = grpc.server(ThreadPoolExecutor(2))
    server.add_generic_rpc_handlers(
        [
            grpc.method_handlers_generic_handler(
                "kv_indexer.v1.KVIndexer",
                {
                    "ApplyExternalKvBatch": grpc.unary_unary_rpc_method_handler(
                        apply,
                        request_deserializer=pb.ApplyExternalKvBatchRequest.FromString,
                        response_serializer=pb.ApplyExternalKvBatchResponse.SerializeToString,
                    ),
                    "MatchExternalKv": grpc.unary_unary_rpc_method_handler(
                        match,
                        request_deserializer=pb.MatchExternalKvRequest.FromString,
                        response_serializer=pb.MatchExternalKvResponse.SerializeToString,
                    ),
                },
            )
        ]
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    config = IndexerConfig(
        endpoint=f"127.0.0.1:{port}",
        deployment="test",
        recovery_s=0.2,
        cooldown_s=0.01,
        timeout_s=0.1,
    )
    publisher = PlacementPublisher(config=config, worker_id="test/lend/node/dp0")
    client = IndexerClient(config)

    def snapshot():
        with lock:
            return set(records)

    try:
        publisher.publish([(key(1), None), (key(2), key(1))])
        eventually(lambda: snapshot() == {block_hash(key(1)), block_hash(key(2))})
        assert (
            client.lookup([block_hash(key(1)), block_hash(key(2))])[publisher.worker_id]
            == snapshot()
        )
        fail_revoke[0] = True
        publisher.retire([key(1)])
        eventually(lambda: snapshot() == {block_hash(key(2))})
        # Indexer restart with no failed client RPC: the periodic recovery heals it.
        with lock:
            records.clear()
        eventually(lambda: snapshot() == {block_hash(key(2))})
        pause_clear[0] = True
        publisher.recover()
        assert entered.wait(5)
        publisher.retire([key(2)])
        publisher.publish([(key(3), key(2))])
        pause_clear[0] = False
        fail_report[0] = True
        resume.set()
        eventually(lambda: snapshot() == {block_hash(key(3))})
    finally:
        resume.set()
        publisher.close()
        client.close()
        server.stop(0).wait()


def test_indexer_query_stats_distinguish_failure_cooldown_and_chunk_latency(
    monkeypatch,
):
    from types import SimpleNamespace

    import grpc

    from sglang.srt.mem_cache.unified_cache.peer_l1 import indexer

    client = indexer.IndexerClient(
        indexer.IndexerConfig(endpoint="unused", deployment="test")
    )
    now = [100.0]
    monkeypatch.setattr(indexer, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(indexer, "MAX_HASHES", 2)
    # Also exercise the latency ring: the interval max must survive truncation.
    from collections import deque

    client._latencies_ms = deque(maxlen=3)
    outcomes = iter(
        [
            (1, None),
            (51, grpc.StatusCode.DEADLINE_EXCEEDED),
            (5, grpc.StatusCode.UNAVAILABLE),
            (3, None),
            (9, None),
        ]
    )

    class Failure(grpc.RpcError):
        def __init__(self, status):
            self.status = status

        def code(self):
            return self.status

    def match(request, timeout):
        assert len(request.hashes) <= 2 and timeout == 0.05
        elapsed, status = next(outcomes)
        now[0] += elapsed / 1000
        if status:
            raise Failure(status)
        return client.pb.MatchExternalKvResponse()

    client.match = match
    try:
        assert client.lookup([1, 2, 3]) is None  # Partial result is discarded.
        assert client.lookup([1, 2, 3]) is None  # Cooldown makes no RPC.
        now[0] += 0.101
        assert client.lookup([1, 2, 3]) is None
        now[0] += 0.101
        assert client.lookup([1, 2, 3]) == {}
        stats = client.stats_snapshot()
        for status in ("ok", "timeout", "error", "cooldown_skip"):
            assert stats[f"indexer_query_{status}"] == 1
        assert stats["indexer_query_calls"] == 4
        assert stats["indexer_query_hashes"] == 12
        assert stats["indexer_query_unavailable_hashes"] == 9
        assert stats["indexer_lookup_rpcs"] == stats["indexer_rpc_latency_count"] == 5
        assert stats["indexer_rpc_latency_samples"] == 3
        assert stats["indexer_rpc_ms_p50"] == pytest.approx(5)
        assert stats["indexer_rpc_ms_p99"] == pytest.approx(9)
        assert stats["indexer_rpc_ms_max"] == pytest.approx(51)
        assert stats["indexer_rpc_ms_total"] == pytest.approx(69)
        empty = client.stats_snapshot()
        assert empty["indexer_rpc_latency_count"] == 0
        assert empty["indexer_rpc_ms_max"] == 0
        assert empty["indexer_query_timeout"] == 1  # Counters are cumulative.
    finally:
        client.close()


def test_publisher_stats_include_failed_recovery_and_retry_gap(monkeypatch):
    from types import SimpleNamespace

    from sglang.srt.grpc import kv_indexer_pb2 as pb
    from sglang.srt.mem_cache.unified_cache.peer_l1 import indexer

    now, fail = [100.0], [False]
    monkeypatch.setattr(indexer, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(indexer.PlacementPublisher, "_run", lambda self: None)

    def apply(request, timeout):
        now[0] += 0.2
        if fail[0] and request.actions[0].type == pb.ACTION_REPORT:
            fail[0] = False
            raise RuntimeError("lost snapshot")

    client = SimpleNamespace(
        pb=pb,
        apply=apply,
        on_reconnect=lambda callback: callback,
        channel=SimpleNamespace(unsubscribe=lambda callback: None),
    )
    publisher = indexer.PlacementPublisher(
        config=indexer.IndexerConfig(endpoint="unused", deployment="test"),
        worker_id="test/lend/dp0",
        client=client,
    )
    try:
        publisher.publish([(key(1), None)])
        publisher._flush()
        publisher._subscription()  # READY alone does not prove state loss.
        publisher._flush()
        assert publisher.stats_snapshot()["recovery_started"] == 1
        publisher.recover()
        fail[0] = True
        with pytest.raises(RuntimeError, match="lost snapshot"):
            publisher._flush()
        now[0] += 1.0  # Retry cooldown remains part of the visibility gap.
        assert publisher.stats_snapshot()["recovery_active_ms"] == pytest.approx(1400)
        publisher.recover()
        publisher._flush()
        stats = publisher.stats_snapshot()
        assert stats["recovery_started"] == 3
        assert stats["recovery_ok"] == 2 and stats["recovery_failed"] == 1
        assert stats["reconnects"] == 1
        assert stats["recovery_requested_manual"] == 2
        assert stats["recovery_ms_total"] == pytest.approx(1200)
        assert stats["recovery_ms_max"] == pytest.approx(400)
        assert stats["recovery_gap_ms_total"] == pytest.approx(2200)
        assert stats["recovery_gap_ms_max"] == pytest.approx(1800)
        assert stats["recovery_active_ms"] == 0
        assert stats["report_rpcs"] == 6
        assert stats["live_records"] == 1 and stats["pending_records"] == 0
    finally:
        publisher.close()


@pytest.mark.parametrize("operation", ["snapshot", "report", "revoke"])
def test_publisher_retries_exact_batch_before_newer_deltas(monkeypatch, operation):
    from types import SimpleNamespace

    import grpc

    from sglang.srt.grpc import kv_indexer_pb2 as pb
    from sglang.srt.mem_cache.unified_cache.peer_l1 import indexer

    monkeypatch.setattr(indexer.PlacementPublisher, "_run", lambda self: None)
    monkeypatch.setattr(indexer, "MAX_HASHES", 1)
    sent, held, waits = [], set(), []
    fail = [False]

    class LostAck(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.DEADLINE_EXCEEDED

    def apply(request, timeout):
        sent.append(request.SerializeToString())
        action = request.actions[0]
        if action.type == pb.ACTION_CLEAR_ALL_AT_TIER:
            held.clear()
        elif action.type == pb.ACTION_REPORT:
            held.update(action.hashes)
        else:
            held.difference_update(action.hashes)
        expected = pb.ACTION_REVOKE if operation == "revoke" else pb.ACTION_REPORT
        if fail[0] and action.type == expected:
            fail[0] = False
            # A conflicting delta arrives after application but before the ACK.
            if operation == "revoke":
                publisher.publish([(key(1), None)])
            else:
                publisher.retire([key(1)])
            raise LostAck()

    client = SimpleNamespace(
        pb=pb,
        apply=apply,
        on_reconnect=lambda callback: callback,
        channel=SimpleNamespace(unsubscribe=lambda callback: None),
    )
    publisher = indexer.PlacementPublisher(
        config=indexer.IndexerConfig(endpoint="unused", deployment="test"),
        worker_id="test/lend/dp0",
        client=client,
    )
    publisher._stop = SimpleNamespace(
        wait=lambda delay: waits.append(delay) or False, set=lambda: None
    )
    try:
        if operation != "snapshot":
            publisher._flush()
        publisher.publish([(key(1), None), (key(2), key(1))])
        if operation == "revoke":
            publisher._flush()
            publisher.retire([key(1)])
        fail[0] = True
        publisher._flush()
        publisher._flush()  # Apply the delta queued during the lost ACK.
        requests = [pb.ApplyExternalKvBatchRequest.FromString(data) for data in sent]
        assert (
            sum(r.actions[0].type == pb.ACTION_CLEAR_ALL_AT_TIER for r in requests) == 1
        )
        assert sum(a == b for a, b in zip(sent, sent[1:])) == 1
        assert held == {indexer.block_hash(key(2))} | (
            {indexer.block_hash(key(1))} if operation == "revoke" else set()
        )
        assert len(waits) == 1 and 0.075 <= waits[0] <= 0.1
        assert publisher.stats_snapshot()["report_retries"] == 1
        assert publisher.stats_snapshot()["recovery_started"] == 1
    finally:
        publisher.close()


def test_publisher_outage_bounds_retries_and_pending_until_periodic_recovery(
    monkeypatch,
):
    from types import SimpleNamespace

    import grpc

    from sglang.srt.grpc import kv_indexer_pb2 as pb
    from sglang.srt.mem_cache.unified_cache.peer_l1 import indexer

    offset, outage, sent, held = [0.0], [False], [], set()
    monkeypatch.setattr(
        indexer, "time", SimpleNamespace(monotonic=lambda: time.monotonic() + offset[0])
    )
    monkeypatch.setattr(indexer, "_PENDING_LIMIT", 2)

    class Unavailable(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.UNAVAILABLE

    def apply(request, timeout):
        sent.append(request.SerializeToString())
        if outage[0]:
            raise Unavailable()
        for action in request.actions:
            if action.type == pb.ACTION_CLEAR_ALL_AT_TIER:
                held.clear()
            elif action.type == pb.ACTION_REPORT:
                held.update(action.hashes)
            else:
                held.difference_update(action.hashes)

    client = SimpleNamespace(
        pb=pb,
        apply=apply,
        on_reconnect=lambda callback: callback,
        channel=SimpleNamespace(unsubscribe=lambda callback: None),
    )
    publisher = indexer.PlacementPublisher(
        config=indexer.IndexerConfig(
            endpoint="unused", deployment="test", cooldown_s=0.001, recovery_s=60
        ),
        worker_id="test/lend/dp0",
        client=client,
    )
    try:
        eventually(lambda: publisher.stats_snapshot()["recovery_ok"] == 1)
        outage[0] = True
        publisher.publish([(key(1), None)])
        eventually(lambda: publisher.stats_snapshot().get("report_pauses") == 1)
        assert len(sent) == 1 + indexer._REPORT_RETRIES + 1
        assert len(set(sent[1:])) == 1
        publisher.retire([key(1)])
        for i in range(2, 10):
            publisher.publish([(key(i), None)])
            publisher.retire([key(i)])
        publisher.publish([(key(10), None)])
        stats = publisher.stats_snapshot()
        assert stats["pending_overflow"] and stats["pending_records"] <= 2
        assert stats["live_records"] == 1
        outage[0] = False
        publisher._subscription()  # Reconnect must not reset the retry budget.
        assert publisher.stats_snapshot()["recovery_started"] == 1
        offset[0] += 60
        eventually(lambda: publisher.stats_snapshot()["recovery_ok"] == 2)
        assert held == {indexer.block_hash(key(10))}
        assert not publisher.stats_snapshot()["pending_overflow"]
        assert publisher.stats_snapshot()["recovery_requested_periodic"] == 1
    finally:
        publisher.close()


def test_old_epoch_cleanup_does_not_touch_current_grant(owner):
    directory, service, queue, peer = owner
    service.publish([key(1)], [3])
    assert peer.request(
        Operation.ACQUIRE_EXACT, full_keys([key(1)]), seq=1
    ).tolist() == [3]
    current_epoch = peer.epoch
    peer.epoch ^= 1
    queue.enqueue(queue.prepare(peer, 1))
    eventually(lambda: queue.pending == 0)
    assert directory.pinned_refs == 1
    peer.epoch = current_epoch
    queue.enqueue(queue.prepare(peer, 1))
    eventually(lambda: directory.pinned_refs == 0)


def test_lend_cleanup_retries_send_failure_and_missing_ack():
    from types import SimpleNamespace

    import zmq

    from sglang.srt.mem_cache.unified_cache.peer_l1.lend.protocol import (
        LendPeer,
        LendService,
        Message,
        decoder,
        encode,
    )

    pool = LendPool(1000)
    pool.add_free(range(1, 10))
    record = pool.allocate("writer", 1)[0]
    record[2] = key(1)
    assert pool.commit("writer", [[*record, None]]) == [True]
    service = LendService(pool, "127.0.0.1")
    peer = LendPeer(
        SimpleNamespace(endpoint="127.0.0.1", lend_port=service.port),
        lambda *_: None,
        timeout_s=0.02,
    )

    class FaultSocket:
        def __init__(self, socket):
            self.socket = socket
            self.fail_send = True
            self.lose_ack = True

        def send(self, payload, flags=0):
            if decoder.decode(payload).op == "CANCEL_ACQUIRE" and self.fail_send:
                raise zmq.Again()
            return self.socket.send(payload, flags)

        def poll(self, timeout):
            return self.socket.poll(timeout)

        def recv(self):
            reply = decoder.decode(self.socket.recv())
            if reply.op == "ACQUIRE" or (
                reply.op == "CANCEL_ACQUIRE" and self.lose_ack
            ):
                if reply.op == "CANCEL_ACQUIRE":
                    self.lose_ack = False
                return encode(Message(0, "ignored", reply.epoch))
            return encode(reply)

        def close(self):
            self.socket.close()

    peer.socket = FaultSocket(peer.socket)
    try:
        assert peer.call("ACQUIRE", [key(1)]) is None
        assert pool.pinned == 1 and peer._cleanup
        peer.socket.fail_send = False

        eventually(lambda: cleanup_drained(peer, pool))
        assert not peer.socket.lose_ack
    finally:
        peer.close()
        service.close()


def test_discovery_keeps_peer_prefix_and_lend_scattered_coverage_distinct():
    from collections import Counter
    from types import SimpleNamespace

    from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
    from sglang.srt.mem_cache.unified_cache.peer_l1.indexer import (
        Placements,
        block_hash,
    )
    from sglang.srt.mem_cache.unified_cache.peer_l1.lend.tiered import TieredL2
    from sglang.srt.mem_cache.unified_cache.peer_l1.linker import PeerL1Linker

    keys = [key(i) for i in range(130)]
    l1 = PeerL1Linker.__new__(PeerL1Linker)
    l1.controls = {
        0: SimpleNamespace(pin_cap=4, available=lambda: True),
        1: SimpleNamespace(pin_cap=3, available=lambda: True),
    }
    l1.worker_ids = {0: "peer0", 1: "peer1"}
    matches = {"peer0": set(map(block_hash, keys)), "peer1": set(map(block_hash, keys))}
    matches.update(
        {
            "lend0": set(map(block_hash, keys[::2])),
            "lend1": set(map(block_hash, keys[1::2])),
            "unknown": set(map(block_hash, keys)),
        }
    )
    l1.placements = Placements(SimpleNamespace(lookup=lambda hashes: matches))
    l1.placement = l1.placements.get("r", keys)
    assert len(l1.discover(keys, start=1).rows) == 4
    l2 = TieredL2.__new__(TieredL2)
    l2.l1, l2.manager, l2.lower = l1, object(), None
    l2._peers = {
        0: SimpleNamespace(worker_id="lend0"),
        1: SimpleNamespace(worker_id="lend1"),
    }
    l2._lookups, l2._pending, l2.stats = {}, {}, Counter()
    transfer = PoolTransfer(name=PoolName.KV, keys=keys, indexer_eligible=True)
    assert l2.lookup("r", [transfer])[-1] == 130
    assert l2.load("r", [transfer])
    _, grouped = l2._pending["r"]
    assert len(grouped[0]) == len(grouped[1]) == 65
    assert {record[2] for _, record in grouped[0]} == set(keys[::2])
    assert l2.stats["lookup_calls"] == 1
    assert l2.stats["lookup_lend_pages"] == l2.stats["lookup_pages"] == 130
    # Reusing an unavailable placement must count every affected lookup,
    # although only the first one asks the client (or hits its cooldown).
    calls = []
    l1.placements.client.lookup = lambda hashes: calls.append(hashes)
    assert l2.lookup("unavailable", [transfer]) == []
    assert l2.lookup("unavailable", [transfer]) == []
    assert len(calls) == 1
    assert l2.stats["lookup_discovery_unavailable_calls"] == 2
    assert l2.stats["lookup_discovery_unavailable_pages"] == 260
    assert l2.stats["lookup_lend_pages"] == 130


def test_lend_reads_more_than_owner_pin_limit_in_chunks():
    from types import SimpleNamespace

    from sglang.srt.mem_cache.unified_cache.peer_l1.lend.protocol import (
        LendPeer,
        LendService,
    )
    from sglang.srt.mem_cache.unified_cache.peer_l1.lend.tiered import TieredL2
    from sglang.srt.mem_cache.unified_cache.peer_l1.lend.transport import LendTransport

    pool = LendPool(12800, limit=0.01)
    pool.add_free(range(1, 200))
    records = []
    for start in range(0, 130, 64):
        allocated = pool.allocate("writer", min(64, 130 - start))
        for i, record in enumerate(allocated, start):
            record[2] = key(i)
        assert all(pool.commit("writer", [[*r, None] for r in allocated]))
        records.extend(allocated)
    assert pool.pin_limit == 64 and len(records) == 130
    service = LendService(pool, "127.0.0.1")
    peer = LendPeer(
        SimpleNamespace(endpoint="127.0.0.1", lend_port=service.port), lambda *_: None
    )
    manager = SimpleNamespace(engine=None, kv_args=SimpleNamespace(kv_item_lens=[]))
    transport = LendTransport(manager)
    sizes = []

    def copy(_info, local, remote, **_kwargs):
        assert remote == [records[i][0] for i in local]
        sizes.append(len(remote))
        return True

    transport.transfer = copy
    l2 = TieredL2.__new__(TieredL2)
    l2.transport = transport
    try:
        assert l2._read_pinned(peer, None, list(enumerate(records)), list(range(130)))
        assert sizes == [64, 64, 2]

        eventually(lambda: cleanup_drained(peer, pool))
    finally:
        peer.close()
        service.close()


@pytest.mark.parametrize("submitted", [False, True])
def test_cancel_keeps_submitted_grant_until_completion(owner, submitted):
    from collections import Counter

    from sglang.srt.mem_cache.unified_cache.peer_l1.linker import (
        PeerL1Linker,
        Reservation,
    )

    directory, service, queue, peer = owner
    service.publish([key(1)], [3])
    from sglang.srt.mem_cache.unified_cache.peer_l1.channels import NativeOwnerChannel
    from sglang.srt.mem_cache.unified_cache.peer_l1.session import PeerSession

    channel = NativeOwnerChannel(peer=peer, worker_id="owner")
    grant = channel.acquire_exact(full_keys([key(1)]))
    linker = PeerL1Linker.__new__(PeerL1Linker)
    linker.cleanup = queue
    linker.sessions = {
        0: PeerSession(
            endpoint=service.endpoint,
            owner=channel,
            mover=None,
            mapping=None,
            cost="local",
        )
    }
    linker.stats = Counter()
    linker.placements = Placements(None)
    linker._lookups = {}
    linker._queued = {} if submitted else {"r": object()}
    linker._inflight = {"r"} if submitted else set()
    linker._reservations = {"r": Reservation(0, [key(1)], grant.rows, grant=grant)}
    linker._collective_guard = None
    linker.release_reservation("r")
    linker.release_reservation("r")
    assert directory.pinned_refs == 1 and queue.pending == 0
    if submitted:
        queue.enqueue(grant.cleanup)  # The real host callback is checked on GPU.
        linker._forget_reservation("r")
    else:
        assert linker.cancel_queued_load("r")
        assert not linker.cancel_queued_load("r")
    eventually(lambda: directory.pinned_refs == 0 and queue.pending == 0)


@pytest.mark.parametrize("elsewhere", [False, True])
def test_f3_retires_whole_batch_before_query_and_drops_retired_metadata(
    owner, elsewhere
):
    from collections import Counter
    from types import SimpleNamespace

    import numpy as np

    from sglang.srt.mem_cache.unified_cache.peer_l1.owner import L1Owner

    directory, service, _queue, peer = owner
    service.publish([key(1), key(2)], [3, 4])
    cache_owner = L1Owner.__new__(L1Owner)
    events = [([key(i)], None, key(i - 1), i != 2) for i in (1, 2, 3)]
    cache_owner.cache = SimpleNamespace(
        tree_core=SimpleNamespace(take_l1_events=lambda: events)
    )
    cache_owner.directory, cache_owner.authority = directory, service
    cache_owner._pending, cache_owner._metadata = {}, {}
    cache_owner._published, cache_owner._hold, cache_owner._writing = (
        set(),
        set(),
        set(),
    )
    queries, writes = [], []

    def contains(keys):
        assert not peer.contains(full_keys([key(1), key(2)])).any()
        queries.append(keys)
        return np.array([elsewhere, False], dtype=bool)

    cache_owner.linker = SimpleNamespace(
        publisher=SimpleNamespace(retire=lambda keys: None),
        stats=Counter(),
        contains_peers=contains,
    )
    cache_owner.last_replica_writer = lambda keys, rows, **_metadata: (
        writes.append(
            (keys, rows.tolist(), _metadata["first_parent"], _metadata["eligible"])
        )
        or True
    )
    cache_owner.consume_events()
    assert queries == [[key(1), key(2)]]
    expected = [([key(1)], [3], key(0), True), ([key(2)], [4], key(1), False)]
    assert writes == (expected[1:] if elsewhere else expected)
    assert cache_owner._writing == cache_owner._hold == ({4} if elsewhere else {3, 4})
    assert cache_owner.linker.stats["f3_contains_pages"] == 2
    assert cache_owner.linker.stats["f3_contains_elsewhere_pages"] == int(elsewhere)
    assert cache_owner._metadata == {}


def test_waiting_placement_is_bounded_reused_and_refreshed(monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    from sglang.srt.mem_cache.unified_cache.peer_l1 import indexer

    now, calls = [0.0], []
    monkeypatch.setattr(indexer.time, "monotonic", lambda: now[0])

    def lookup(hashes):
        calls.append(hashes.copy())
        return {"peer": set(hashes)}

    cache = indexer.Placements(SimpleNamespace(lookup=lookup))
    keys = [key(1), key(2)]
    first = cache.get("waiting", keys)
    assert cache.get("waiting", keys) is first and len(calls) == 1
    assert np.array_equal(first.full, full_keys(keys))
    assert list(first.hashes) == list(map(indexer.block_hash, keys))
    now[0] += indexer.PLACEMENT_TTL_S + 0.001
    assert cache.get("waiting", keys) is first and len(calls) == 2
    # Requests which never load or reserve cannot leave unbounded hints.
    for rid in range(indexer.PLACEMENT_CAPACITY * 2):
        cache.get(str(rid), keys)
    assert len(cache.entries) == indexer.PLACEMENT_CAPACITY
    assert "waiting" not in cache.entries
    cache.pop(str(rid))
    assert len(cache.entries) == indexer.PLACEMENT_CAPACITY - 1


def test_lookup_failure_cools_owner_and_discovery_skips_it(owner, monkeypatch):
    from types import SimpleNamespace

    from sglang.srt.mem_cache.unified_cache.peer_l1.indexer import (
        Placements,
    )
    from sglang.srt.mem_cache.unified_cache.peer_l1.linker import PeerL1Linker

    _directory, _service, _queue, peer = owner
    calls = []
    monkeypatch.setattr(
        peer, "request", lambda *args, **kwargs: calls.append(args) or None
    )
    keys = [key(1)]
    assert peer.lookup(full_keys(keys)) == peer.lookup(full_keys(keys)) == 0
    assert len(calls) == 1
    linker = PeerL1Linker.__new__(PeerL1Linker)
    linker.controls = {0: peer}
    linker.worker_ids = {0: "peer"}
    cache = Placements(SimpleNamespace(lookup=lambda hashes: {"peer": set(hashes)}))
    linker.placement = cache.get("r", keys)
    assert linker.discover(keys, start=0) is None
    linker.placement.matches = None
    assert linker.discover(keys, start=0) is None
    assert len(calls) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
