"""Event-driven publisher repair; no GPU or simulated network is required."""

import math
from types import SimpleNamespace

import pytest

from sglang.srt.grpc import kv_indexer_pb2 as pb
from sglang.srt.mem_cache.unified_cache.peer_l1 import indexer


class FixedRandom:
    def __init__(self, fraction=0.5):
        self.fraction = fraction

    def uniform(self, low, high):
        return low + (high - low) * self.fraction


@pytest.fixture
def publisher(monkeypatch):
    now = [100.0]
    held, requests = set(), []
    monkeypatch.setattr(indexer, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(indexer.random, "Random", lambda: FixedRandom(0.5))
    monkeypatch.setattr(indexer.PlacementPublisher, "_run", lambda self: None)

    def apply(request, timeout):
        requests.append(request)
        for a in request.actions:
            if a.type == pb.ACTION_CLEAR_ALL_AT_TIER:
                held.clear()
            elif a.type == pb.ACTION_REPORT:
                held.update(a.hashes)
            elif a.type == pb.ACTION_REVOKE:
                held.difference_update(a.hashes)

    client = SimpleNamespace(
        pb=pb,
        apply=apply,
        on_reconnect=lambda cb: cb,
        channel=SimpleNamespace(unsubscribe=lambda cb: None),
    )
    p = indexer.PlacementPublisher(
        config=indexer.IndexerConfig(endpoint="test", deployment="test"),
        worker_id="test/prefill/dp0",
        client=client,
    )
    yield p, now, held, requests
    p.close()


def key(i):
    return f"{i:016x}" + "0" * 48


def test_defaults_and_disabled_fallback_validation():
    c = indexer.IndexerConfig(endpoint="test", deployment="test")
    assert c.recovery_s == 3600 and c.recovery_jitter_s == 5
    assert (
        indexer.IndexerConfig(
            endpoint="test", deployment="test", recovery_s=0
        ).recovery_s
        == 0
    )
    for field in ["timeout_s", "cooldown_s", "recovery_s", "recovery_jitter_s"]:
        for value in [-1, math.inf, math.nan]:
            with pytest.raises(ValueError):
                indexer.IndexerConfig(endpoint="x", deployment="x", **{field: value})
    for field in ["timeout_s", "cooldown_s"]:
        with pytest.raises(ValueError):
            indexer.IndexerConfig(endpoint="x", deployment="x", **{field: 0})


def test_channel_only_notifies_ready_after_a_disconnect_and_connects_idle(monkeypatch):
    import grpc

    calls = []

    class Channel:
        def unary_unary(self, *a, **k):
            return lambda *a, **k: None

        def subscribe(self, callback, try_to_connect=False):
            self.callback = callback
            self.try_to_connect = try_to_connect

        def close(self):
            pass

    channel = Channel()
    monkeypatch.setattr(grpc, "insecure_channel", lambda _: channel)
    c = indexer.IndexerClient(indexer.IndexerConfig(endpoint="test", deployment="test"))
    try:
        cb = c.on_reconnect(lambda: calls.append("reconnect"))
        assert channel.try_to_connect
        for state in [
            grpc.ChannelConnectivity.IDLE,
            grpc.ChannelConnectivity.CONNECTING,
            grpc.ChannelConnectivity.READY,
            grpc.ChannelConnectivity.READY,
        ]:
            cb(state)
        assert not calls
        for state in [
            grpc.ChannelConnectivity.TRANSIENT_FAILURE,
            grpc.ChannelConnectivity.CONNECTING,
            grpc.ChannelConnectivity.READY,
            grpc.ChannelConnectivity.READY,
        ]:
            cb(state)
        assert calls == ["reconnect"]
        cb(grpc.ChannelConnectivity.IDLE)
        cb(grpc.ChannelConnectivity.READY)
        assert len(calls) == 2
    finally:
        c.close()


def test_confirmed_reconnect_rebuilds_state_without_periodic(publisher):
    p, now, held, _ = publisher
    p.config = indexer.IndexerConfig(
        endpoint="x", deployment="x", recovery_s=0, recovery_jitter_s=0
    )
    p.publish([(key(1), None), (key(2), key(1))])
    p._flush()
    held.clear()
    p._subscription()
    assert p._generation != p._completed_generation
    p._flush()
    assert held == {1, 2}
    assert p.stats_snapshot()["recovery_requested_reconnect"] == 1
    assert p._fallback_at is None


def test_per_worker_jitter_coalesces_without_postponing(publisher):
    p, now, held, _ = publisher
    assert p._ready_delay(now[0]) == pytest.approx(2.5)
    p._flush()
    p._rng = FixedRandom(0.2)
    p.recover("reconnect")
    first = p._recover_at
    assert first == pytest.approx(now[0] + 1)
    now[0] += 0.1
    p._rng = FixedRandom(0.8)
    p.recover("reconnect")
    assert p._recover_at == first
    assert p._ready_delay(now[0]) == pytest.approx(0.9)
    now[0] = first
    assert p._ready_delay(now[0]) == 0
    p._flush()
    assert p.stats_snapshot()["recovery_ok"] == 2


def test_fallback_is_bounded_and_does_not_add_event_jitter(publisher):
    p, now, _, _ = publisher
    p._flush()
    assert p._fallback_at == pytest.approx(now[0] + 3240)
    due = p._fallback_at
    now[0] = due - 0.001
    p._ready_delay(now[0])
    assert p.stats.get("recovery_requested_periodic", 0) == 0
    now[0] = due
    assert p._ready_delay(now[0]) == 0
    assert p.stats["recovery_requested_periodic"] == 1
    p._flush()
    assert p._fallback_at == pytest.approx(now[0] + 3240)


def test_zero_fallback_still_allows_event_repairs(publisher):
    p, now, held, _ = publisher
    p.config = indexer.IndexerConfig(
        endpoint="x", deployment="x", recovery_s=0, recovery_jitter_s=0
    )
    p.publish([(key(1), None)])
    p._flush()
    now[0] += 1e7
    assert p._fallback_at is None
    p._ready_delay(now[0])
    assert p._generation == p._completed_generation
    held.clear()
    p.recover("reconnect")
    p._flush()
    assert held == {1}


def test_failure_retry_backoff_is_bounded_and_independent_of_hour_timer(publisher):
    p, now, held, _ = publisher
    p.config = indexer.IndexerConfig(endpoint="x", deployment="x", recovery_jitter_s=0)
    p.publish([(key(1), None)])
    p._flush()
    held.clear()
    delays = []
    for _ in range(15):
        p._report_failed(RuntimeError("rejected"))
        delays.append(p._ready_delay(now[0]))
        now[0] = p._paused_until
    assert delays[0] == pytest.approx(0.075)
    assert 0 < max(delays) <= 30 and delays[-1] == pytest.approx(22.5)
    assert p._ready_delay(now[0]) == 0
    p._flush()
    assert held == {1}
    assert p.stats["recovery_requested_report_failure"] == 15


def test_overflow_schedules_immediate_full_repair(publisher, monkeypatch):
    p, now, held, _ = publisher
    p.config = indexer.IndexerConfig(
        endpoint="x", deployment="x", recovery_s=0, recovery_jitter_s=0
    )
    monkeypatch.setattr(indexer, "_PENDING_LIMIT", 2)
    p._flush()
    p.publish([(key(i), key(i - 1) if i else None) for i in range(3)])
    assert p._pending_overflow and not p._pending
    assert p._generation != p._completed_generation
    p._flush()
    assert held == {0, 1, 2}
    assert not p._pending_overflow
    assert p.stats["recovery_requested_pending_overflow"] == 1


def test_reconnect_during_snapshot_is_not_lost(publisher):
    p, now, held, _ = publisher
    p.config = indexer.IndexerConfig(
        endpoint="x", deployment="x", recovery_s=0, recovery_jitter_s=0
    )
    p.publish([(key(1), None)])
    p._flush()
    original = p.client.apply
    once = [False]

    def apply(request, timeout):
        original(request, timeout)
        if not once[0] and request.actions[0].type == pb.ACTION_CLEAR_ALL_AT_TIER:
            once[0] = True
            p.retire([key(1)])
            p.publish([(key(2), key(1))])
            p._on_reconnect()

    p.client.apply = apply
    p.recover()
    p._flush()
    assert p._generation != p._completed_generation
    p._flush()
    assert held == {2}
    assert p._generation == p._completed_generation


def test_idle_grpc_restart_restores_without_any_new_publication():
    import time
    from concurrent.futures import ThreadPoolExecutor

    import grpc

    state = [set()]

    def apply(request, context):
        for a in request.actions:
            if a.type == pb.ACTION_CLEAR_ALL_AT_TIER:
                state[0].clear()
            elif a.type == pb.ACTION_REPORT:
                state[0].update(a.hashes)
            else:
                state[0].difference_update(a.hashes)
        return pb.ApplyExternalKvBatchResponse()

    def start(port):
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
                        )
                    },
                )
            ]
        )
        actual = server.add_insecure_port(f"127.0.0.1:{port}")
        assert actual
        server.start()
        return server, actual

    def until(predicate):
        deadline = time.monotonic() + 10
        while not predicate():
            assert time.monotonic() < deadline
            time.sleep(0.01)

    server, port = start(0)
    publisher = indexer.PlacementPublisher(
        config=indexer.IndexerConfig(
            endpoint=f"127.0.0.1:{port}",
            deployment="test",
            recovery_s=0,
            recovery_jitter_s=0,
            cooldown_s=0.001,
        ),
        worker_id="test/prefill/dp0",
    )
    try:
        publisher.publish([(key(1), None), (key(2), key(1))])
        until(lambda: state[0] == {1, 2})
        server.stop(0).wait()
        state[0] = set()
        time.sleep(0.2)
        server, again = start(port)
        assert again == port
        # No lookup(), publish(), recover(), or periodic fallback drives this.
        until(lambda: state[0] == {1, 2})
        assert publisher.stats_snapshot()["recovery_requested_reconnect"] >= 1
        assert publisher.stats_snapshot().get("recovery_requested_periodic", 0) == 0
    finally:
        publisher.close()
        server.stop(0).wait()


def test_rejected_publication_triggers_background_full_repair():
    import time

    import grpc

    held, sent, reject = set(), [], [True]

    class Rejected(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.INVALID_ARGUMENT

    def apply(request, timeout):
        sent.append(request)
        for action in request.actions:
            if action.type == pb.ACTION_REPORT and reject[0]:
                reject[0] = False
                raise Rejected()
            if action.type == pb.ACTION_CLEAR_ALL_AT_TIER:
                held.clear()
            elif action.type == pb.ACTION_REPORT:
                held.update(action.hashes)

    client = SimpleNamespace(
        pb=pb,
        apply=apply,
        on_reconnect=lambda cb: cb,
        channel=SimpleNamespace(unsubscribe=lambda cb: None),
    )
    p = indexer.PlacementPublisher(
        config=indexer.IndexerConfig(
            endpoint="x",
            deployment="x",
            recovery_s=0,
            recovery_jitter_s=0,
            cooldown_s=0.01,
        ),
        worker_id="test/dp0",
        client=client,
    )

    def until(predicate):
        deadline = time.monotonic() + 3
        while not predicate():
            assert time.monotonic() < deadline
            time.sleep(0.005)

    try:
        until(lambda: p.stats_snapshot()["recovery_ok"] == 1)
        p.publish([(key(1), None), (key(2), key(1))])
        until(lambda: held == {1, 2} and p.stats_snapshot()["recovery_ok"] == 2)
        assert p.stats_snapshot()["recovery_requested_report_failure"] == 1
        assert p.stats_snapshot().get("recovery_requested_periodic", 0) == 0
        assert [r.actions[0].type for r in sent] == [
            pb.ACTION_CLEAR_ALL_AT_TIER,
            pb.ACTION_REPORT,
            pb.ACTION_CLEAR_ALL_AT_TIER,
            pb.ACTION_REPORT,
        ]
    finally:
        p.close()


def test_worker_jitter_uses_independent_random_streams(monkeypatch):
    monkeypatch.setattr(indexer.PlacementPublisher, "_run", lambda self: None)
    monkeypatch.setattr(indexer, "time", SimpleNamespace(monotonic=lambda: 100.0))
    fractions = iter([0.2, 0.8])
    monkeypatch.setattr(
        indexer, "random", SimpleNamespace(Random=lambda: FixedRandom(next(fractions)))
    )
    client = SimpleNamespace(
        on_reconnect=lambda cb: cb, channel=SimpleNamespace(unsubscribe=lambda cb: None)
    )
    config = indexer.IndexerConfig(endpoint="x", deployment="x")
    a = indexer.PlacementPublisher(config=config, worker_id="a", client=client)
    b = indexer.PlacementPublisher(config=config, worker_id="b", client=client)
    try:
        assert (a._recover_at, b._recover_at) == (101.0, 104.0)
    finally:
        a.close()
        b.close()


def test_idle_connectivity_keeps_only_one_readiness_future(monkeypatch):
    import grpc

    futures = []
    rpc_calls = []

    class Future:
        def __init__(self):
            self.cancelled = False

        def done(self):
            return self.cancelled

        def cancel(self):
            self.cancelled = True

    class Channel:
        def unary_unary(self, *args, **kwargs):
            return lambda *args, **kwargs: rpc_calls.append(1)

        def subscribe(self, callback, try_to_connect=False):
            pass

        def close(self):
            pass

    channel = Channel()
    monkeypatch.setattr(grpc, "insecure_channel", lambda _: channel)

    def ready_future(_):
        f = Future()
        futures.append(f)
        return f

    monkeypatch.setattr(grpc, "channel_ready_future", ready_future)
    c = indexer.IndexerClient(indexer.IndexerConfig(endpoint="x", deployment="x"))
    try:
        cb = c.on_reconnect(lambda: None)
        for _ in range(10):
            c.ensure_connected()
        assert len(futures) == 1 and not rpc_calls
        cb(grpc.ChannelConnectivity.READY)
        assert futures[0].cancelled
        cb(grpc.ChannelConnectivity.IDLE)
        for _ in range(10):
            c.ensure_connected()
        assert len(futures) == 2 and not rpc_calls
    finally:
        c.close()
    assert futures[1].cancelled
