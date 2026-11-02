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
from sglang.srt.mem_cache.unified_cache.peer_l1.indexer import report_batches
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


@pytest.fixture
def owner(tmp_path):
    directory = L1Directory(str(tmp_path / "directory"), 32, 4, create=True)
    service = OwnerService(directory, lease_s=0.02)
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
    rows = peer.request(2, full_keys([key(1)]), seq=1)
    assert rows.tolist() == [3]
    assert peer.request(2, full_keys([key(1)]), seq=1).tolist() == [3]
    assert directory.pinned_refs == 1
    time.sleep(0.04)
    # Each RPC closes its connection. Neither close nor expiry authorizes reuse.
    assert peer.request(2, full_keys([key(1)]), seq=2).tolist() == []
    mapping = L1Directory(directory.path, 32, 4, create=False)
    try:
        service.retire([key(1)])
        from sglang.srt.mem_cache.unified_cache.peer_l1.directory import hash_words

        assert not mapping.contains(hash_words([key(1)]))[0]
    finally:
        mapping.close()
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
    assert peer.request(2, full_keys([key(1)]), seq=8).tolist() == []
    assert directory.pinned_refs == 0
    assert peer.request(2, full_keys([key(1)]), seq=9).tolist() == [3]
    for _ in range(2):
        queue.enqueue(queue.prepare(peer, 9, cancel=True))
    eventually(lambda: queue.pending == 0 and directory.pinned_refs == 0)
    assert peer.request(2, full_keys([key(1)]), seq=9).tolist() == []


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


def test_retire_order_requires_at_least_one_writer(owner):
    directory, service, _queue, peer = owner
    other_directory = L1Directory(None, 32, 4, create=True)
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
    assert all(pool.commit("writer", records))
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
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as target:
                    target.connect("\0" + service.endpoint)
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
        assert peer.request(2, full_keys([key(1)]), seq=peer.seq).tolist() == []
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
        stage=1,
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


def test_old_epoch_cleanup_does_not_touch_current_grant(owner):
    directory, service, queue, peer = owner
    service.publish([key(1)], [3])
    assert peer.request(2, full_keys([key(1)]), seq=1).tolist() == [3]
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
    assert pool.commit("writer", [record]) == [True]
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
        IndexerConfig,
        Placements,
        block_hash,
    )
    from sglang.srt.mem_cache.unified_cache.peer_l1.lend.tiered import TieredL2
    from sglang.srt.mem_cache.unified_cache.peer_l1.linker import PeerL1Linker

    keys = [key(i) for i in range(130)]
    l1 = PeerL1Linker.__new__(PeerL1Linker)
    l1.indexer_config = IndexerConfig(stage=3, endpoint="unused", deployment="test")
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
    l1.sources = l1.controls
    l1.placements = Placements(SimpleNamespace(lookup=lambda hashes: matches))
    l1.placement = l1.placements.get("r", keys)
    assert len(l1.discover(keys, start=1).rows) == 4
    l2 = TieredL2.__new__(TieredL2)
    l2.indexer_stage, l2.l1, l2.manager, l2.lower = 1, l1, object(), None
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
        assert all(pool.commit("writer", allocated))
        records.extend(allocated)
    assert pool.pin_limit == 64 and len(records) == 130
    service = LendService(pool, "127.0.0.1")
    peer = LendPeer(
        SimpleNamespace(endpoint="127.0.0.1", lend_port=service.port), lambda *_: None
    )
    manager = SimpleNamespace(engine=None, kv_args=SimpleNamespace(kv_item_lens=[]))
    transport = LendTransport(manager, acquire=True)
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
    rows, cleanup = peer.acquire(full_keys([key(1)]))
    linker = PeerL1Linker.__new__(PeerL1Linker)
    linker.cleanup = queue
    linker.stats = Counter()
    linker.placements = None
    linker._lookups = {}
    linker._queued = {} if submitted else {"r": object()}
    linker._inflight = {"r"} if submitted else set()
    linker._reservations = {"r": Reservation(0, [key(1)], rows, None, cleanup)}
    linker._collective_guard = None
    linker.release_reservation("r")
    linker.release_reservation("r")
    assert directory.pinned_refs == 1 and queue.pending == 0
    if submitted:
        queue.enqueue(cleanup)  # The real host callback is checked on GPU.
        linker._forget_reservation("r")
    else:
        assert linker.cancel_queued_load("r")
        assert not linker.cancel_queued_load("r")
    eventually(lambda: directory.pinned_refs == 0 and queue.pending == 0)


def test_f3_retires_whole_batch_before_query_and_drops_retired_metadata(owner):
    from collections import Counter
    from types import SimpleNamespace

    import numpy as np

    from sglang.srt.mem_cache.unified_cache.peer_l1.indexer import IndexerConfig
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
        return np.zeros(len(keys), dtype=bool)

    cache_owner.linker = SimpleNamespace(
        publisher=None,
        stats=Counter(),
        contains_peers=contains,
        indexer_config=IndexerConfig(stage=4, endpoint="unused", deployment="test"),
    )
    cache_owner.last_replica_writer = lambda keys, rows, **_metadata: (
        writes.append(
            (keys, rows.tolist(), _metadata["first_parent"], _metadata["eligible"])
        )
        or True
    )
    cache_owner.consume_events()
    assert queries == [[key(1), key(2)]]
    assert writes == [([key(1)], [3], key(0), True), ([key(2)], [4], key(1), False)]
    assert cache_owner._writing == cache_owner._hold == {3, 4}
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
        IndexerConfig,
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
    linker.indexer_config = IndexerConfig(stage=3, endpoint="unused", deployment="test")
    linker.sources = {0: peer}
    linker.worker_ids = {0: "peer"}
    cache = Placements(SimpleNamespace(lookup=lambda hashes: {"peer": set(hashes)}))
    linker.placement = cache.get("r", keys)
    assert linker.discover(keys, start=0) is None
    linker.placement.matches = None
    assert linker.discover(keys, start=0) is None
    assert len(calls) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
