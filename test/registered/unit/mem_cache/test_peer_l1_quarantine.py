"""Failed RDMA destinations must survive release, reset, and late completion."""

from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.quarantine import (
    QuarantinedPages,
    full_page_quarantine,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.unified_cache.peer_l1.lend.allocator import (
    LendablePagedAllocator,
)
from sglang.srt.mem_cache.unified_cache.peer_l1.lend.pool import LendPool
from sglang.srt.mem_cache.unified_cache.peer_l1.lend.protocol import (
    LendPeer,
    LendService,
)
from sglang.srt.mem_cache.unified_cache.peer_l1.lend.tiered import TieredL2
from sglang.srt.mem_cache.unified_cache.peer_l1.lend.transport import LendTransport
from sglang.srt.mem_cache.unified_cache.peer_l1.rdma_io import (
    Submission,
    UnsafeTransfer,
)
from sglang.srt.runtime_context import (
    get_resources,
    reset_context,
    restore_context,
    snapshot_context,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def allocator(pages=8, need_sort=False):
    return PagedTokenToKVPoolAllocator(
        pages * 4, 4, torch.float32, "cpu", torch.zeros(pages + 1, 4), need_sort
    )


def test_release_filters_on_device_and_observation_accounts_lazily(monkeypatch):
    """A quarantined page must not force every later free to copy IDs to the host."""
    quarantine = QuarantinedPages(page_size=4, owner=object())
    quarantine.mark([2], object())
    with monkeypatch.context() as patch:
        patch.setattr(
            torch.Tensor, "cpu", lambda _: pytest.fail("release copied IDs to CPU")
        )
        assert quarantine.filter_released(torch.tensor([1, 2])).tolist() == [1]
        quarantine.mark([3], object())
        assert quarantine.filter_released(torch.tensor([3, 4])).tolist() == [4]
    assert quarantine.withheld == {2, 3}
    assert quarantine.retained_tokens == 8


def test_ipc_only_does_not_require_a_quarantine_compatible_allocator():
    wrapper = object()
    assert full_page_quarantine(wrapper, required=False) is None
    with pytest.raises(ValueError, match="paged full allocator"):
        full_page_quarantine(wrapper)


@pytest.mark.parametrize("need_sort", [False, True])
def test_late_write_cannot_reach_reused_or_reset_request_pages(need_sort):
    alloc = allocator(need_sort=need_sort)
    old = alloc.alloc(8)
    poisoned = int(old[0]) // 4
    state = snapshot_context()
    alloc.quarantine.mark([poisoned], object())
    alloc.free_group_begin()
    alloc.free(old[:4])
    alloc.free_segment(old[4:], start_pos=4)
    alloc.free_group_end()
    assert alloc.quarantine.retained_tokens == 4
    fresh = alloc.alloc(28)
    assert fresh is not None and poisoned not in (fresh // 4).tolist()
    alloc._kvcache[fresh // 4] = 99
    alloc._kvcache[poisoned] = 7  # A late DMA reaches only the retired allocation.
    assert torch.all(alloc._kvcache[fresh // 4] == 99)
    alloc.clear()
    assert alloc.available_size() == 28 and poisoned not in alloc.free_pages
    restore_context(state)
    reset_context()
    assert alloc.quarantine in get_resources().rdma_quarantines


def test_poisoned_decode_allocation_stays_out_of_reclaim_and_clear():
    original = allocator(1000)
    stolen = original.alloc(8)[::4] // 4
    pool = LendPool(1000)
    pool.add_free(stolen.tolist())
    alloc = LendablePagedAllocator.install(original, pool)
    records = pool.allocate("writer", 1, seq=7)
    page = records[0][0]
    pool.poison_allocation("writer", 7)
    pool.poison_allocation("writer", 7)
    pool.abort("writer", records)
    pool.cancel_allocation("writer", 7)
    pool.expire()
    assert pool.poisoned == 1 and pool.writing == 0
    assert page not in pool.reclaim(1000)
    alloc.clear()
    assert pool.pages[page].state == "poisoned"
    assert alloc.available_size() == 3996 and page not in alloc.free_pages
    assert alloc.alloc(4000) is None
    assert pool.stats["quarantined_pages"] == 1


def test_partial_native_submission_retains_statuses_and_precheck_does_not(monkeypatch):
    from sglang.srt.mem_cache.unified_cache.peer_l1 import rdma_io

    statuses = []

    def status():
        value = SimpleNamespace(Succeeded=lambda: False)
        statuses.append(value)
        return value

    monkeypatch.setattr(rdma_io, "new_status", status)

    def post(*args):
        assert len(args[-2]) == 2
        raise RuntimeError("one QP posted, sibling post failed")

    manager = SimpleNamespace(
        engine=SimpleNamespace(
            _engine=SimpleNamespace(BatchRead=post), allocate_transfer_uid=lambda: 1
        ),
        kv_mem_descs=[object(), object()],
        kv_args=SimpleNamespace(kv_item_lens=[4, 4]),
    )
    transfer = LendTransport(manager)
    with pytest.raises(ValueError):
        transfer.transfer(SimpleNamespace(dst_kv_mem_descs=[]), [1], [1])
    assert not transfer.failed_attempts and not statuses
    with pytest.raises(UnsafeTransfer) as error:
        transfer.transfer(
            SimpleNamespace(dst_kv_mem_descs=[object(), object()]), [1], [1]
        )
    assert error.value.submission.statuses == statuses
    assert (
        len(statuses) == 2
        and error.value.submission in get_resources().rdma_quarantines
    )


def test_write_failure_never_aborts_when_poison_notification_is_delayed():
    pool = LendPool(1000)
    pool.add_free([1])
    service = LendService(pool, "127.0.0.1", worker_id="decode")
    peer = LendPeer(
        SimpleNamespace(endpoint="127.0.0.1", lend_port=service.port), lambda *_: None
    )
    pending = []
    cleanup = peer.cleanup
    peer.cleanup = lambda op, seq: pending.append((op, seq))
    attempt = Submission(engine=None, local=[], remote=[], submitted=True)

    def failed(*args, **kwargs):
        raise UnsafeTransfer(attempt)

    l2 = TieredL2.__new__(TieredL2)
    l2.page_size, l2._next_write_peer = 4, 0
    l2._peers, l2.lower, l2.stats = {0: peer}, None, Counter()
    l2.transport = SimpleNamespace(transfer=failed)
    try:
        assert not l2._write(
            PoolTransfer(
                name=PoolName.KV,
                keys=["key"],
                device_indices=torch.arange(4, 8),
                indexer_eligible=True,
            )
        )
        assert pool.writing == 1 and not pool.free
        assert pending[0][0] == "POISON"
        peer.cleanup = cleanup
        cleanup(*pending[0])
        peer.call("HELLO")  # Same socket orders the poison before this reply.
        assert pool.poisoned == 1 and not pool.reclaim(1)
    finally:
        peer.close()
        service.close()


def test_uncertain_read_cannot_fallback_into_the_same_destination():
    alloc = allocator()
    target = alloc.alloc(8)
    pages = (target[::4] // 4).tolist()
    l2 = TieredL2.__new__(TieredL2)
    l2.l1 = SimpleNamespace(quarantine=alloc.quarantine)
    l2.stats = Counter()
    attempt = Submission(engine=None, local=[], remote=[], submitted=True)

    def fail(*args, **kwargs):
        raise UnsafeTransfer(attempt)

    l2.transport = SimpleNamespace(read=fail)
    fallback = []
    l2._read_lower = lambda *args: fallback.append(args) or True
    items = [(i, [0, 0, str(i)]) for i in range(2)]
    assert not l2._read_pinned(object(), None, items, pages)
    assert not fallback and alloc.quarantine.pages == set(pages)
    l2.transport.read = lambda *args, **kwargs: False  # ACQUIRE miss, no submission.
    fresh_pages = (alloc.alloc(8)[::4] // 4).tolist()
    assert l2._read_pinned(object(), None, items, fresh_pages)
    assert len(fallback) == 1


def test_staging_skips_quarantined_slots_for_later_demotions():
    staging = SimpleNamespace(
        pages=4, quarantine=QuarantinedPages(page_size=1, owner=object())
    )
    writes = []
    staging.put = lambda lower, records, *, slots: writes.append(slots) or True
    calls = []

    def read(peer, records, slots, **kwargs):
        calls.append(slots)
        if len(calls) == 1:
            raise UnsafeTransfer(
                Submission(engine=None, local=[], remote=[], submitted=True)
            )
        return True

    l2 = TieredL2.__new__(TieredL2)
    l2.staging, l2.lower, l2.stats = staging, None, Counter()
    l2.transport = SimpleNamespace(read=read)
    peer = SimpleNamespace(call=lambda *args: None)
    l2._demote(peer, [[1, 1, "a"], [2, 1, "b"]])
    l2._demote(peer, [[3, 1, "c"], [4, 1, "d"]])
    assert calls == [[0, 1], [2, 3]] and writes == [[2, 3]]
    assert l2.stats["quarantined_staging_pages"] == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
