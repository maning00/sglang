"""The shared-memory protocol, including a real cross-process reader."""

import multiprocessing as mp

import pytest
import torch
import torch.distributed as dist
from sglang.srt.mem_cache.unified_cache.peer_l1.debug import CollectiveGuard
from sglang.srt.mem_cache.unified_cache.peer_l1.directory import L1Directory, hash_words
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize("name", ["all_reduce", "all_gather_into_tensor"])
def test_collective_guard_catches_preimported_alias(name):
    collective = getattr(dist, name)  # Capture the alias before enabling checks.
    tensor = torch.zeros(1)
    guard = CollectiveGuard()
    guard.enable()
    try:
        dist.is_initialized()  # Metadata queries must remain usable.
        with pytest.raises(AssertionError, match="unsubmitted reservation"):
            if name == "all_reduce":
                collective(tensor)
            else:
                collective(tensor, tensor)
        assert not guard.enabled
    finally:
        guard.disable()


def _key(value):
    # Deliberately collide in the table's low hash word.
    return "0" * 16 + f"{value:016x}" + "0" * 32


def _reader(path, acquired, release):
    directory = L1Directory(path, 16, 4, create=False)
    rows, gens = directory.probe(hash_words([_key(1)]))
    assert rows.tolist() == [3]
    assert directory.pin(rows, gens)
    acquired.set()
    assert release.wait(10)
    directory.unpin(rows)
    directory.close()


def test_pin_rollback_and_generation(tmp_path):
    directory = L1Directory(str(tmp_path / "directory"), 16, 4, create=True)
    try:
        directory.publish([_key(1), _key(2)], [3, 4])
        rows, gens = directory.probe(hash_words([_key(1), _key(2)]))
        directory.retire([_key(2)])
        assert not directory.pin(rows, gens)
        assert directory.pinned_refs == 0
        assert directory.states([3, 4])[0].tolist() == [0, 0]
        directory.publish([_key(3)], [4])
        assert not directory.pin(rows, gens)
        assert directory.pinned_refs == 0
        assert directory.freeze()
        assert not directory.pin(rows[:1], gens[:1])
        directory.thaw()
        assert directory.pin(rows[:1], gens[:1])
        directory.unpin(rows[:1])
    finally:
        directory.close()


def test_unlinked_name_keeps_mappings_shared(tmp_path):
    """The owner drops the name after peers map it; both mappings must stay one
    shared table, and close must tolerate the missing name."""
    path = tmp_path / "directory"
    directory = L1Directory(str(path), 16, 4, create=True)
    peer = L1Directory(str(path), 16, 4, create=False)
    try:
        directory.unlink()
        assert not path.exists()
        directory.publish([_key(1)], [3])
        rows, gens = peer.probe(hash_words([_key(1)]))
        assert rows.tolist() == [3]
        assert peer.pin(rows, gens)
        assert directory.states([3])[0].tolist() == [1]
        peer.unpin(rows)
        assert directory.pinned_refs == 0
    finally:
        peer.close()
        directory.close()


def test_reader_survives_retirement(tmp_path):
    path = str(tmp_path / "directory")
    directory = L1Directory(path, 16, 4, create=True)
    ctx = mp.get_context("spawn")
    acquired, release = ctx.Event(), ctx.Event()
    reader = ctx.Process(target=_reader, args=(path, acquired, release))
    try:
        directory.publish([_key(1)], [3])
        reader.start()
        assert acquired.wait(10)
        assert directory.retire([_key(1)]).tolist() == [3]
        assert directory.states([3])[0].tolist() == [1]
        assert not directory.freeze()
        with pytest.raises(RuntimeError, match="pinned"):
            directory.publish([_key(2)], [3])
        release.set()
        reader.join(10)
        assert reader.exitcode == 0
        assert directory.pinned_refs == 0
        directory.publish([_key(2)], [3])
        assert directory.probe(hash_words([_key(2)]))[0].tolist() == [3]
    finally:
        release.set()
        if reader.is_alive():
            reader.terminate()
            reader.join()
        directory.close()


def test_collisions_churn_and_cap(tmp_path):
    directory = L1Directory(str(tmp_path / "directory"), 16, 2, create=True)
    try:
        for generation in range(100):
            keys = [_key(generation * 3 + i) for i in range(3)]
            directory.publish(keys, [1, 2, 3])
            # Lookup clips the promise before reserve, rather than discarding
            # the entire long prefix when the hard cap would reject it.
            rows, gens = directory.probe(hash_words(keys))
            assert rows.tolist() == [1, 2]
            assert directory.pin(rows, gens)
            assert not directory.pin(rows[:1], gens[:1])
            assert directory.retire(keys).tolist() == [1, 2, 3]
            directory.unpin(rows)
            assert directory.pinned_refs == 0
    finally:
        directory.close()


@pytest.fixture
def peer_linker(tmp_path):
    from collections import Counter

    from sglang.srt.mem_cache.unified_cache.peer_l1.linker import PeerL1Linker

    directory = L1Directory(str(tmp_path / "source"), 16, 4, create=True)
    directory.publish([_key(1), _key(2)], [3, 4])
    peer = PeerL1Linker.__new__(PeerL1Linker)
    peer.peers = {1: directory}
    peer.stats = Counter()
    peer.min_pages = 1
    peer.pin_cap = 4
    peer.stitch_stats = False
    peer._collective_guard = None
    peer._lookups, peer._reservations, peer._queued, peer._batches = {}, {}, {}, {}
    peer._inflight = set()
    try:
        yield peer, directory
    finally:
        directory.close()


def _queue_peer(peer):
    import torch

    from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer

    transfer = PoolTransfer(
        name=PoolName.KV, keys=[_key(1), _key(2)], device_indices=torch.arange(8)
    )
    assert peer.lookup("r", [transfer]) == [1, 2]
    assert peer.reserve("r", 2)
    assert peer.load("r", [transfer])


def test_peer_cancel_unpins_only_a_queued_load(peer_linker):
    peer, directory = peer_linker
    _queue_peer(peer)
    # Releasing a request's hint must not unpin a still-queued transfer.
    peer.release_reservation("r")
    assert directory.pinned_refs == 2
    assert peer.cancel_queued_load("r")
    assert not peer.cancel_queued_load("r")
    assert directory.pinned_refs == 0
    assert not peer.has_loads and not peer._lookups


@pytest.mark.parametrize("error", [0, 1])
def test_peer_finish_waits_before_retiring_even_on_failure(peer_linker, error):
    from types import SimpleNamespace

    import torch

    peer, directory = peer_linker
    _queue_peer(peer)
    plan = peer._queued.pop("r")
    peer._inflight.add("r")
    peer._batches[0] = SimpleNamespace(rids=["r"], error_cpu=torch.tensor(error))

    def finish(index):
        assert index == 0 and peer.has_loads
        assert directory.pinned_refs == 2
        directory.unpin(plan.rows)  # Completion event follows the native callback.
        return True

    peer.layer_done_counter = SimpleNamespace(finish=finish)
    assert not peer.cancel_queued_load("r")
    peer.release_reservation("r")
    assert directory.pinned_refs == 2
    assert peer.finish_layer_wise_loading(0) == (error == 0)
    assert not peer.has_loads and not peer._inflight
    assert directory.pinned_refs == 0
    assert peer.stats["failed_load_batches"] == bool(error)


class _Source:
    """Queue contract with real CPU layer futures; no device or storage IO."""

    def __init__(self):
        from collections import Counter

        from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
            LayerWiseLoadCounter,
        )

        self.page_size = 4
        self.hits, self.pin, self.accept_load = 2, True, True
        self.stats = Counter()
        self.layer_done_counter = LayerWiseLoadCounter(2)
        self.pending, self.batches = {}, {}
        self.reserved, self.active, self.hints = set(), set(), set()
        self.finishes = []

    def bind_cache(self, cache):
        pass

    def lookup(self, rid, transfers):
        self.hints.add(rid)
        return list(range(1, self.hits + 1))

    def reserve(self, rid, pages):
        self.hints.discard(rid)
        if self.pin:
            self.reserved.add(rid)
        return self.pin

    def release_reservation(self, rid):
        self.hints.discard(rid)
        if rid not in self.active:
            self.reserved.discard(rid)

    def load(self, rid, transfers):
        if not self.accept_load:
            return False
        self.pending[rid] = transfers
        self.active.add(rid)
        return True

    def start_layer_wise_loading(self):
        if not self.pending:
            return -1
        index = self.layer_done_counter.update_producer()
        self.batches[index] = list(self.pending)
        self.pending.clear()
        return index

    def finish_layer_wise_loading(self, index):
        self.finishes.append(index)
        landed = self.layer_done_counter.finish(index)
        for rid in self.batches.pop(index, ()):
            self.active.remove(rid)
            self.reserved.discard(rid)
        return landed

    def cancel_queued_load(self, rid):
        if self.pending.pop(rid, None) is None:
            return False
        self.active.remove(rid)
        self.release_reservation(rid)
        return True

    def reset(self):
        self.pending.clear()
        self.batches.clear()
        self.active.clear()
        self.reserved.clear()
        self.hints.clear()
        self.layer_done_counter.reset()

    close = reset


@pytest.fixture
def composite():
    from sglang.srt.mem_cache.unified_cache.peer_l1.composite import PeerL1PlusL2Linker

    return PeerL1PlusL2Linker(_Source(), _Source())


def _transfers(pages=2):
    import torch

    from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer

    return [
        PoolTransfer(
            name=PoolName.KV,
            keys=[_key(i) for i in range(pages)],
            device_indices=torch.arange(pages * 4),
        )
    ]


@pytest.mark.parametrize(
    "a,b,pin,source",
    [
        (4, 2, True, "l1"),
        (2, 2, True, "l1"),
        (2, 4, True, "l2"),
        (4, 4, False, "l2"),
        (4, 2, False, None),
    ],
)
def test_composite_selects_one_source_and_releases_unused_reservation(
    composite, a, b, pin, source
):
    composite.l1.hits, composite.l2.hits, composite.l1.pin = a, b, pin
    pages = max(a, b)
    assert composite.lookup("r", _transfers(pages)) == list(range(1, pages + 1))
    assert composite.reserve("r", pages) == (source is not None)
    if source is not None:
        assert composite._requests["r"].source == source
    composite.release_reservation("r")  # Allocation failed before load().
    assert not composite._requests and not composite._lookups
    assert not composite.l1.reserved and not composite.l1.hints


@pytest.mark.parametrize("source", ["l1", "l2"])
@pytest.mark.parametrize("cancel", [False, True], ids=["rejected", "cancelled"])
def test_composite_cleans_a_load_that_never_started(composite, source, cancel):
    composite.l1.hits = 2 if source == "l1" else 0
    backend = getattr(composite, source)
    backend.accept_load = cancel
    transfers = _transfers()
    composite.lookup("r", transfers)
    assert composite.reserve("r", 2)
    assert composite.load("r", transfers) == cancel
    if cancel:
        assert "r" in backend.pending  # L2 must be queued during load(), too.
        assert composite.cancel_queued_load("r")
    composite.release_reservation("r")
    assert composite.start_layer_wise_loading() == -1
    assert not composite._requests and not composite._queued
    assert not backend.active and not composite.l1.reserved


@pytest.mark.parametrize("failed_source", [None, "l1", "l2"])
def test_composite_finishes_both_sources_before_batch_verdict(composite, failed_source):
    from types import SimpleNamespace

    from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
        UnifiedCacheLinkerWrapper,
    )
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from test_unified_cache_linker import _cache_for_wrapper

    transfers = _transfers()
    for rid, hits in (("a", 2), ("b", 0)):
        composite.l1.hits = hits
        composite.lookup(rid, transfers)
        assert composite.reserve(rid, 2)
        assert composite.load(rid, transfers)
    index = composite.start_layer_wise_loading()
    assert not composite.cancel_queued_load("a")
    for name in ("l1", "l2"):
        source = getattr(composite, name)
        counter = source.layer_done_counter
        if name == failed_source:
            counter.fail(counter.producer_index, RuntimeError("load failed"))
        else:
            for layer in range(2):
                counter.complete(counter.producer_index, layer)
    composite.layer_done_counter.set_consumer(index)
    composite.layer_done_counter.wait_until(1)  # No exception or early retirement.
    assert index in composite.layer_done_counter._indices

    cache = _cache_for_wrapper(_all_reduce_attn_groups=lambda *args: None)
    wrapper = UnifiedCacheLinkerWrapper(cache, composite)
    cache.linker = wrapper
    wrapper.inflight_load_rids = ["a", "b"]
    wrapper.inflight_load_index = index
    wrapper.private_load_indices = {
        rid: transfers[0].device_indices for rid in ("a", "b")
    }
    reqs = [
        SimpleNamespace(rid=rid, skip_radix_cache_insert=False) for rid in ("a", "b")
    ]
    failed = UnifiedRadixCache.finish_external_linker_loads(cache, reqs)
    assert failed == (reqs if failed_source else [])
    if failed_source:
        assert all(
            req.skip_radix_cache_insert and req.discard_output_reason.status_code == 503
            for req in reqs
        )
    assert not wrapper.private_load_indices and not wrapper.inflight_load_rids
    assert not composite._requests and not composite._batches
    assert not composite.layer_done_counter._indices
    assert composite.l1.finishes == [0] and composite.l2.finishes == [0]
    assert not composite.l1.reserved and not composite.l2.active
    composite.layer_done_counter.set_consumer(index)  # A later decode keeps this index.
    composite.layer_done_counter.wait_until(0)


@pytest.mark.parametrize("operation", ["reset", "close"])
def test_composite_reset_clears_unused_requests(composite, operation):
    composite.lookup("r", _transfers())
    composite.reserve("r", 2)
    getattr(composite, operation)()
    assert not composite._lookups and not composite._requests
    assert not composite._queued and not composite._batches
    assert not composite.layer_done_counter._indices
    assert not composite.l1.reserved


class _HeldKeys(_Source):
    """An L2 that answers lookups from the keys it holds."""

    def __init__(self, held):
        super().__init__()
        self.held, self.probed = set(held), []

    def lookup(self, rid, transfers):
        (transfer,) = transfers
        self.probed.append(list(transfer.keys))
        hits = 0
        while hits < len(transfer.keys) and transfer.keys[hits] in self.held:
            hits += 1
        return list(range(1, hits + 1))


@pytest.mark.parametrize("l2_accepts", [True, False])
def test_last_replica_restores_a_peer_head_with_its_l2_tail(l2_accepts):
    """Last-replica writes keep a page out of L2 while some HBM holds it, so a
    prefix whose head is still in a peer has its tail only in L2. The restore
    takes the head from the peer and the tail from L2, each into its own slots;
    a rejected L2 part leaves the peer part neither queued nor pinned."""
    from sglang.srt.mem_cache.unified_cache.peer_l1.composite import PeerL1PlusL2Linker

    l1, l2 = _Source(), _HeldKeys({_key(2), _key(3)})
    l2.accept_load = l2_accepts
    linker = PeerL1PlusL2Linker(l1, l2, l2_write="last_replica")
    assert linker.lookup("r", _transfers(5)) == [1, 2, 3, 4]
    assert l2.probed == [[_key(2), _key(3), _key(4)]]
    assert linker.reserve("r", 4)
    assert linker.load("r", _transfers(4)) == l2_accepts
    if not l2_accepts:
        assert not l1.pending and not l1.reserved and not linker._requests
        return
    ((head,),), ((tail,),) = l1.pending.values(), l2.pending.values()
    assert head.keys == [_key(0), _key(1)] and tail.keys == [_key(2), _key(3)]
    assert head.device_indices.tolist() == list(range(8))
    assert tail.device_indices.tolist() == list(range(8, 16))
    index = linker.start_layer_wise_loading()
    for source in (l1, l2):
        counter = source.layer_done_counter
        for layer in range(2):
            counter.complete(counter.producer_index, layer)
    assert linker.finish_layer_wise_loading(index)
    assert (l1.stats["l1_loaded_pages"], l1.stats["l2_loaded_pages"]) == (2, 2)
    assert l1.stats["stitched_loads"] == 1 and not l1.reserved


def test_last_replica_completions_release_rows_in_submission_order():
    """L2 completions are positional: each must release the rows of the write
    submitted in the same order, and none may reach the linker wrapper, which
    has no offload of its own pending under the last-replica policy."""
    from types import SimpleNamespace

    import numpy as np
    from sglang.srt.mem_cache.hicache_storage import PoolName
    from sglang.srt.mem_cache.unified_cache.peer_l1.composite import PeerL1PlusL2Linker

    l1, l2 = _Source(), _Source()
    released = []
    l1.owner = SimpleNamespace(release_written=released.append)
    l1.bind_cache = lambda cache: None
    offloads, results = [], []
    l2.offload = lambda transfers: offloads.append(transfers) or True
    l2.num_completed_offloads = lambda: len(results)
    l2.pop_completed_offload = lambda: results.pop(0)
    linker = PeerL1PlusL2Linker(l1, l2, l2_write="last_replica")
    assert not linker.persists
    linker.bind_cache(None)
    first, second = np.array([3, 5]), np.array([7])
    assert l1.owner.last_replica_writer(["a", "b"], first)
    assert l1.owner.last_replica_writer(["c"], second)
    (transfer,) = offloads[0]
    assert transfer.name == PoolName.KV and transfer.keys == ["a", "b"]
    assert transfer.device_indices.device.type == "cpu"
    assert transfer.device_indices.tolist() == list(range(12, 16)) + list(range(20, 24))
    results.extend([True, False])
    assert linker.num_completed_offloads() == 0
    assert [rows.tolist() for rows in released] == [[3, 5], [7]]
    assert l1.stats["last_replica_failed_batches"] == 1


def test_last_replica_writes_are_bounded_while_in_flight(monkeypatch):
    """Rows held for in-flight L2 writes cannot grow without bound: a write that
    would exceed the in-flight cap is refused, and after one write outlives the
    timeout every new write is refused until it lands. A late write keeps its rows
    held, since the L2 copy may still be reading them."""
    from types import SimpleNamespace

    import numpy as np
    from sglang.srt.mem_cache.unified_cache.peer_l1 import composite as composite_module

    clock = [0.0]
    monkeypatch.setattr(composite_module.time, "monotonic", lambda: clock[0])
    l1, l2 = _Source(), _Source()
    released = []
    l1.owner = SimpleNamespace(release_written=released.append)
    results = []
    l2.offload = lambda transfers: True
    l2.num_completed_offloads = lambda: len(results)
    l2.pop_completed_offload = lambda: results.pop(0)
    linker = composite_module.PeerL1PlusL2Linker(
        l1, l2, l2_write="last_replica", max_inflight_write_pages=3, write_timeout_s=10
    )
    linker.bind_cache(None)
    write = l1.owner.last_replica_writer

    assert write(["a", "b"], np.array([3, 5]))
    assert not write(["c", "d"], np.array([7, 8]))
    assert l1.stats["last_replica_capped_pages"] == 2
    results.append(True)
    linker.num_completed_offloads()
    assert write(["c", "d"], np.array([7, 8]))

    clock[0] = 11.0
    linker.num_completed_offloads()
    assert not write(["e"], np.array([9]))
    assert l1.stats["last_replica_stalled_pages"] == 1
    assert l1.stats["last_replica_write_timeouts"] == 1
    assert [rows.tolist() for rows in released] == [[3, 5]]
    results.append(True)
    linker.num_completed_offloads()
    assert [rows.tolist() for rows in released] == [[3, 5], [7, 8]]
    assert write(["e"], np.array([9]))
