"""Ownership, real control sockets/shared directory, and RDMA completion boundaries."""

import random
import threading
from collections import Counter
from contextlib import nullcontext
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.mem_cache.unified_cache.peer_l1.lend.allocator import (
    LendablePagedAllocator,
)
from sglang.srt.mem_cache.unified_cache.peer_l1.lend.directory import LendDirectory
from sglang.srt.mem_cache.unified_cache.peer_l1.lend.lifecycle import unbind_decode_lend
from sglang.srt.mem_cache.unified_cache.peer_l1.lend.pool import LendPool
from sglang.srt.mem_cache.unified_cache.peer_l1.lend.protocol import (
    LendPeer,
    LendService,
)
from sglang.srt.mem_cache.unified_cache.peer_l1.lend.tiered import TieredL2
from sglang.srt.mem_cache.unified_cache.peer_l1.lend.transport import LendTransport
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def key(n):
    return f"{n:064x}"


def pool():
    result = LendPool(1000, limit=0.02, clock=lambda: 0)
    result.add_free(range(1, 101))
    return result


def committed(p, count=2):
    records = p.allocate("writer", count)
    for i, r in enumerate(records):
        r[2] = key(i)
    assert all(p.commit("writer", records))
    return records


def test_multiple_readers_survive_demoted_and_expiry():
    p = pool()
    records = committed(p)
    a = p.pin("a", records)
    b = p.pin("b", records)
    assert a and b and p.pinned == 2
    p.request_demotions(2)
    p.demoted("writer", records)
    assert not p.pin("c", records)
    p.unpin("a", a)
    assert p.pinned == 2
    assert not {r[0] for r in records} & set(p.reclaim(1000))
    p.unpin("b", b)
    assert p.pinned == 0 and p.reclaimable == 2
    assert set(p.reclaim(2)) == {r[0] for r in records}


def test_writing_is_never_timeout_reclaimed_and_stale_generation_is_rejected():
    p = pool()
    records = p.allocate("writer", 10)
    assert len(records) == p.write_limit
    assert not p.allocate("writer", 1)
    p.clock = lambda: 100000
    p.expire()
    assert not {r[0] for r in records} & set(p.reclaim(1000))
    p.abort("other", records)
    assert p.writing == 10
    p.abort("writer", records)
    assert p.writing == 0
    new = p.allocate("writer", 10)
    for r in new:
        r[2] = key(r[0])
    assert all(p.commit("writer", new))
    assert not p.pin("reader", [[r[0], r[1] - 1, r[2]] for r in new])


def test_unique_pinned_pages_budget_and_growth_gate():
    p = pool()
    records = committed(p, 10)
    a = p.pin("a", records)
    b = p.pin("b", records)
    assert a and b and p.pinned == 10
    extra = p.allocate("writer", 1)
    extra[0][2] = key(100)
    assert p.commit("writer", extra) == [True]
    assert not p.pin("c", extra)
    p.decode_used = 901
    assert not p.allocate("writer", 1)
    p.unpin("a", a)
    p.clock = lambda: 31
    p.expire()
    assert p.pinned == 0 and not p.leases


def test_native_directory_conditional_removal_and_shared_mapping(tmp_path):
    path = str(tmp_path / "lend")
    first = LendDirectory(path, 17, create=True)
    second = LendDirectory(path, 17)
    try:
        first.put(3, [[7, 1, key(1)]])
        assert second.query([key(1), key(2)]) == [[3, 7, 1], [-1, 0, 0]]
        second.put(3, [[7, 2, key(1)]])
        first.remove(3, [[7, 1, key(1)]])
        assert first.query([key(1)]) == [[3, 7, 2]]
        first.remove(3, [[7, 2, key(1)]])
        assert second.query([key(1)])[0][0] == -1
        # Deletion must preserve probe chains, including a fully populated table.
        records = [[i, 1, key(i)] for i in range(17)]
        assert first.put(1, records) == 17
        first.remove(1, records)
        assert first.put(2, records) == 17
    finally:
        second.close()
        first.close()


def test_native_directory_churn_preserves_colliding_and_wrapped_entries(tmp_path):
    directory = LendDirectory(str(tmp_path / "lend"), 17, create=True)
    rng = random.Random(7)
    expected = {}
    try:
        for _ in range(400):
            n = rng.randrange(35)
            record = [n + 1, rng.randrange(1, 8), key(n)]
            if rng.randrange(2):
                accepted = directory.put(2, [record])
                if accepted:
                    expected[key(n)] = [2, *record[:2]]
            else:
                directory.remove(2, [record])
                if expected.get(key(n)) == [2, *record[:2]]:
                    del expected[key(n)]
            keys = [key(i) for i in range(35)]
            assert directory.query(keys) == [expected.get(k, [-1, 0, 0]) for k in keys]
    finally:
        directory.close()


def test_real_control_service_roundtrip():
    p = pool()
    service = LendService(p, "127.0.0.1")
    peer = LendPeer(
        SimpleNamespace(endpoint="127.0.0.1", lend_port=service.port), lambda *_: None
    )
    try:
        assert peer.epoch == p.epoch
        record = peer.call("ALLOC", [1]).data[0]
        record[2] = key(7)
        assert peer.call("COMMIT", [record]).data == [True]
        lease = peer.call("PIN", [record]).data[0]
        assert lease
        p.request_demotions(1)
        peer.call("DEMOTED", [record])
        assert p.pages[record[0]].state == "retired"
        peer.call("UNPIN", [lease])
        assert p.pages[record[0]].state == "free"
    finally:
        peer.close()
        service.close()


def test_lost_alloc_reply_cancels_only_its_unseen_grant(monkeypatch):
    p = pool()
    other = p.allocate("other-writer", 1, seq=2)
    service = LendService(p, "127.0.0.1")
    peer = LendPeer(
        SimpleNamespace(endpoint="127.0.0.1", lend_port=service.port),
        lambda *_: None,
        timeout_s=0.05,
    )
    poll = peer.poll

    def drop_alloc(timeout=0):
        reply = poll(timeout)
        return None if reply is not None and reply.op == "ALLOC" else reply

    try:
        monkeypatch.setattr(peer, "poll", drop_alloc)
        assert peer.call("ALLOC", [2]) is None
        assert p.writing == 1
        assert p.pages[other[0][0]].state == "writing"
        monkeypatch.setattr(peer, "poll", poll)
        record = peer.call("ALLOC", [1]).data[0]
        # A delayed duplicate cancellation cannot touch the next allocation.
        peer.call("CANCEL_ALLOC", [2])
        assert p.pages[record[0]].state == "writing"
        peer.call("ABORT", [record])
        assert p.writing == 1
    finally:
        peer.close()
        service.close()


def test_unresponsive_writer_fails_unbind_without_reusing_its_pages():
    p = pool()
    records = p.allocate("unresponsive", 1)
    a = allocator(p)
    service = SimpleNamespace(pool=p, close=MagicMock())
    outer = SimpleNamespace(full_attn_allocator=a)
    scheduler = SimpleNamespace(
        decode_lend_service=service, token_to_kv_pool_allocator=outer
    )
    with pytest.raises(TimeoutError, match="remain protected"):
        unbind_decode_lend(scheduler, timeout_s=0)
    service.close.assert_not_called()
    assert scheduler.decode_lend_service is service
    assert p.draining and p.writing == 1
    assert records[0][0] not in p.reclaim(1000)


def allocator(p):
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

    original = PagedTokenToKVPoolAllocator(
        1000 * 256, 256, torch.int64, "cpu", None, True
    )
    original.use_triton_kernels = False
    # Pages in p have already been removed from the real allocator.
    original.free_pages = original.free_pages[100:]
    return LendablePagedAllocator.install(original, p)


def test_allocator_all_entrypoints_reclaim_before_kernels():
    for method in ("alloc", "alloc_extend", "alloc_decode"):
        p = pool()
        a = allocator(p)
        # Leave one lent page beyond the lend I/O limit and no actual free pages.
        p.reclaim(99 - p.limit)
        a.free_pages = torch.empty(0, dtype=torch.int64)
        assert a.available_size() == 256
        if method == "alloc":
            result = a.alloc(256)
        elif method == "alloc_extend":
            result = a.alloc_extend(
                torch.tensor([0]),
                torch.tensor([0]),
                torch.tensor([256]),
                torch.tensor([256]),
                torch.tensor([-1]),
                256,
            )
        else:
            result = a.alloc_decode(
                torch.tensor([1]), torch.tensor([1]), torch.tensor([-1])
            )
        assert result is not None and len(p.pages) == p.limit
        assert result[0].item() == (100 - p.limit) * 256


@pytest.mark.parametrize("return_to_decode", [False, True])
def test_pending_lend_needs_forward_and_copy_and_remains_allocatable(return_to_decode):
    p = pool()
    a = allocator(p)
    pages = a.free_pages[:2]
    a.free_pages = a.free_pages[2:]
    copied, forwarded = MagicMock(), MagicMock()
    copied.query.return_value = False
    forwarded.query.return_value = True
    a._pending_lend = (pages, pages.clone(), copied, forwarded)
    assert a.available_size() == (1000 - p.limit) * 256
    assert a.get_all_free_pages().unique().numel() == 900
    a._publish_ready_lend()
    assert pages[0].item() not in p.pages
    copied.query.return_value = True
    forwarded.query.return_value = False
    a._publish_ready_lend()
    assert pages[0].item() not in p.pages
    if return_to_decode:
        assert a.alloc(900 * 256) is not None
        assert a.available_size() == (100 - p.limit) * 256
    forwarded.query.return_value = True
    a._publish_ready_lend()
    assert (pages[0].item() in p.pages) != return_to_decode
    assert a._pending_lend is None
    assert a.available_size() == ((100 if return_to_decode else 1000) - p.limit) * 256


def test_admission_budget_survives_concurrent_lend_io():
    """Decode can allocate what available_size() reported after lend I/O pins or
    writes lent pages up to the limit in between."""
    p = pool()
    a = allocator(p)
    a.free_pages = torch.empty(0, dtype=torch.int64)
    budget = a.available_size()
    records = committed(p, p.pin_limit)
    assert p.pin("reader", records)
    assert p.allocate("writer", p.write_limit)
    assert a.alloc(budget) is not None


def test_restore_bypasses_full_write_queue(monkeypatch):
    from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
        LayerWiseLoadCounter,
    )

    l2 = TieredL2.__new__(TieredL2)
    l2._tasks = Queue(maxsize=1)
    l2._loads = Queue()
    l2._wake = threading.Event()
    l2._stop = threading.Event()
    l2._pending = {"r": (None, {0: []})}
    l2._demotions = []
    l2._peers = {}
    l2._load_counter = LayerWiseLoadCounter(1)
    l2.layer_done_counter = l2._load_counter
    l2.num_layers, l2.lower = 1, None
    l2.manager = SimpleNamespace(kv_args=SimpleNamespace(gpu_id=0))
    l2._next_stats_log = float("inf")
    l2.stats = Counter()
    l2._refresh_peers = lambda: None
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(
        "sglang.srt.mem_cache.unified_cache.peer_l1.lend.tiered.get_device_module",
        lambda: SimpleNamespace(
            Event=MagicMock, Stream=MagicMock, stream=lambda _: nullcontext()
        ),
    )
    order = []
    l2._read = lambda _: order.append("read")

    def write(_):
        order.append("write")
        l2._stop.set()
        return True

    l2._write = write
    l2._results = Queue()
    l2._tasks.put_nowait(("write", None, MagicMock()))
    index = l2.start_layer_wise_loading()
    l2._run()
    assert order == ["read", "write"]
    assert l2.finish_layer_wise_loading(index)
    assert l2._results.get_nowait() is True


def test_failed_peer_discovery_cools_down_and_new_registration_retries(monkeypatch):
    l2 = TieredL2.__new__(TieredL2)
    info = SimpleNamespace(engine_key="old", decode_dp_rank=0, lend_port=1)
    l2.manager = SimpleNamespace(decode_kv_args_table={"old": info})
    l2.transport = SimpleNamespace(lengths=[])
    l2._peers, l2._peer_retry_after, l2._incompatible_peers = {}, {}, set()
    clock = [0.0]
    peer = MagicMock(return_value=SimpleNamespace(epoch="", close=lambda: None))
    monkeypatch.setattr(
        "sglang.srt.mem_cache.unified_cache.peer_l1.lend.tiered.LendPeer", peer
    )
    monkeypatch.setattr(
        "sglang.srt.mem_cache.unified_cache.peer_l1.lend.tiered.time.monotonic",
        lambda: clock[0],
    )
    l2._refresh_peers()
    l2._refresh_peers()
    assert peer.call_count == 1
    clock[0] = 6
    l2._refresh_peers()
    assert peer.call_count == 2
    info.engine_key = "new"
    l2._refresh_peers()
    assert peer.call_count == 3 and "old" not in l2._peer_retry_after


def test_restore_runs_while_lower_write_is_pending():
    from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
    from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
        LayerWiseLoadCounter,
    )

    l2 = TieredL2.__new__(TieredL2)
    l2.page_size, l2.num_layers = 256, 1
    l2._peers, l2._loads = {}, Queue()
    l2.stats = Counter()
    l2._load_counter = LayerWiseLoadCounter(1)
    index = l2._load_counter.update_producer()
    l2._read = MagicMock()
    l2._loads.put_nowait(("load", (index, []), MagicMock()))
    l2.lower = MagicMock()
    l2.lower.num_completed_offloads.side_effect = [0, 0, 1]
    l2.lower.pop_completed_offload.return_value = True
    kv = PoolTransfer(name=PoolName.KV, keys=[key(1)], device_indices=torch.arange(256))
    assert l2._write(kv)
    l2._read.assert_called_once_with([])
    assert l2._load_counter.finish(index)


def test_drain_finishes_every_status_after_wait_all_failure():
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    status = MagicMock()
    status.InProgress.return_value = True
    status.Wait.side_effect = lambda: (entered.set(), release.wait(5))
    engine = MagicMock()
    engine.batch_read.return_value = [status]
    manager = SimpleNamespace(
        engine=engine,
        kv_mem_descs=[object()],
        kv_args=SimpleNamespace(kv_item_lens=[64]),
        _wait_transfer_completion=lambda _: "first status failed",
    )
    transport = LendTransport(manager)
    outcome = []

    def read():
        outcome.append(
            transport.transfer(SimpleNamespace(dst_kv_mem_descs=[object()]), [1], [2])
        )
        returned.set()

    worker = threading.Thread(target=read)
    worker.start()
    assert entered.wait(5) and not returned.is_set()
    release.set()
    worker.join(5)
    assert outcome == [False]


def test_expired_pin_data_is_discarded_after_drain(monkeypatch):
    manager = SimpleNamespace(engine=object(), kv_args=SimpleNamespace(kv_item_lens=[]))
    transport = LendTransport(manager)
    clock = [0.0]
    monkeypatch.setattr(
        "sglang.srt.mem_cache.unified_cache.peer_l1.lend.transport.time.monotonic",
        lambda: clock[0],
    )
    peer = MagicMock(lease_s=30)
    peer.call.return_value = SimpleNamespace(data=[1])

    def copy(*_, **__):
        clock[0] = 16
        return True

    transport.transfer = copy
    assert not transport.read(peer, [[1, 1, key(1)]], [2])
    assert peer.call.call_args.args == ("UNPIN", [1])
