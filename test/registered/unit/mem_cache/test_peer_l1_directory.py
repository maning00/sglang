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


# The composition tests use CPU tensors and the real native directory. Child
# transports expose only their queue contract; no GPU-module import stubs.
class _Source:
    def __init__(self, name, hits=0, pin=True):
        from collections import Counter, deque
        from queue import Queue
        from types import SimpleNamespace

        self.name, self.hits, self.pin = name, hits, pin
        self.page_size = 4
        self.loading_keys, self.stats = Counter(), Counter()
        self.pending, self.reserved = {}, {}
        self.waiting, self._completed_loads = deque(), Queue()
        self.active, self.hints = set(), set()
        self.index = -1
        self.waits = []
        self.layer_done_counter = SimpleNamespace(
            num_layers=2, set_consumer=self._consumer, wait_until=self.waits.append
        )

    def _consumer(self, index):
        self.consumer = index

    @property
    def has_loads(self):
        return bool(self.reserved or self.active)

    def lookup(self, rid, transfers):
        self.hints.add(rid)
        self.keys = transfers[0].keys
        return list(range(1, self.hits + 1))

    def reserve(self, rid, pages):
        self.hints.discard(rid)
        if not self.pin:
            return False
        self.reserved[rid] = self.keys[:pages]
        self.loading_keys.update(self.reserved[rid])
        return True

    def release_reservation(self, rid):
        self.hints.discard(rid)
        if rid not in self.active and rid in self.reserved:
            self.loading_keys.subtract(self.reserved.pop(rid))
            self.loading_keys += {}  # Drop zeros, like the peer backend.

    def load(self, rid, transfers):
        self.active.add(rid)
        self.pending[rid] = transfers
        return True

    def start_layer_wise_loading(self):
        if not self.pending:
            return -1
        self.index += 1
        self.waiting.append(list(self.pending))
        self.pending = {}
        return self.index % 3 if self.name == "l1" else self.index

    def complete(self, error=None):
        self._completed_loads.put((self.waiting.popleft(), error))

    def num_completed_loads(self):
        return self._completed_loads.qsize()

    def pop_completed_load(self):
        from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import (
            UMBPDirectLinker,
        )

        rids = UMBPDirectLinker.pop_completed_load(self)
        for rid in rids:
            self.active.remove(rid)
            if self.name == "l1":
                self.release_reservation(rid)
        return rids


@pytest.fixture
def composite(tmp_path):
    from types import SimpleNamespace

    from sglang.srt.mem_cache.unified_cache.peer_l1.composite import PeerL1PlusL2Linker
    from sglang.srt.mem_cache.unified_cache.peer_l1.owner import L1Owner

    l1, l2 = _Source("l1"), _Source("l2")
    directory = L1Directory(str(tmp_path / "composite"), 16, 8, create=True)
    owner = L1Owner.__new__(L1Owner)
    owner.directory, owner.linker = directory, l1
    owner._version, owner._pending, owner._attaches = 0, {}, []
    owner._published, owner._hold, owner._zombies = set(), set(), set()
    events = []

    def take_events():
        result = events[:]
        events.clear()
        return result

    wrapper = SimpleNamespace(pending_loads={}, pending_offloads=[])
    owner.cache = SimpleNamespace(
        tree_core=SimpleNamespace(take_l1_events=take_events), linker=wrapper
    )
    l1.owner = owner
    try:
        yield PeerL1PlusL2Linker(l1, l2), directory, events
    finally:
        directory.close()


def _transfers(pages):
    from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer

    return [
        PoolTransfer(
            name=PoolName.KV,
            keys=[_key(i + 1) for i in range(pages)],
            device_indices=torch.arange(pages * 4, dtype=torch.int64),
        )
    ]


@pytest.mark.parametrize(
    "a,b,pin,source,stored",
    [
        (4, 2, True, "l1", False),
        (2, 2, True, "l1", True),
        (2, 4, True, "l2", True),
        (4, 4, False, "l2", True),
        (4, 2, False, None, None),
    ],
)
def test_composite_selects_one_source_and_rolls_back(
    composite, a, b, pin, source, stored
):
    combo, _, _ = composite
    combo.l1.hits, combo.l2.hits, combo.l1.pin = a, b, pin
    assert combo.lookup("req", _transfers(max(a, b))) == list(range(1, max(a, b) + 1))
    assert combo.reserve("req", max(a, b)) == (source is not None)
    if source is not None:
        assert combo._requests["req"].source == source
        assert combo.marks_loaded_as_stored("req") == stored
    combo.release_reservation("req")  # Allocation failed or COMMIT adopted no page.
    assert not combo.l1.loading_keys and not combo.l1.hints
    assert not combo._requests


def test_l2_cancel_keeps_publication_state_and_consumes_late_attach(composite):
    combo, directory, events = composite
    combo.l2.hits = 2
    transfers = _transfers(2)
    combo.lookup("req", transfers)
    assert combo.reserve("req", 2)
    events.append((transfers[0].keys, transfers[0].device_indices))
    assert combo.load("req", transfers)
    combo.release_reservation("req")  # Request cancellation does not cancel IO.
    assert combo.l1.loading_keys and combo._requests
    combo.start_layer_wise_loading()
    assert not len(directory.probe(hash_words(transfers[0].keys))[0])
    combo.l2.complete()
    assert combo.num_completed_loads() == 1
    assert combo.pop_completed_load() == ["req"]
    assert directory.probe(hash_words(transfers[0].keys))[0].tolist() == [0, 1]
    assert not combo.l1.loading_keys and not combo._requests
    assert not combo.l1.owner._attaches  # No second publication was scheduled.


def test_failed_l2_batch_never_publishes(composite):
    combo, directory, _ = composite
    combo.l2.hits = 1
    transfers = _transfers(1)
    combo.lookup("req", transfers)
    assert combo.reserve("req", 1)
    assert combo.load("req", transfers)
    combo.start_layer_wise_loading()
    combo.l2.complete(RuntimeError("injected IO failure"))
    with pytest.raises(RuntimeError, match="UMBP layer-wise"):
        combo.pop_completed_load()
    assert not len(directory.probe(hash_words(transfers[0].keys))[0])


def test_composite_fifo_does_not_reuse_one_child_completion(composite):
    combo, _, _ = composite
    for rid, a, b in [("l1", 1, 0), ("both-l2", 0, 1)]:
        combo.l1.hits, combo.l2.hits = a, b
        combo.lookup(rid, _transfers(1))
        assert combo.reserve(rid, 1)
        assert combo.load(rid, _transfers(1))
    first = combo.start_layer_wise_loading()
    # A second batch uses only L2, on a different page so publications differ.
    combo.l1.hits, combo.l2.hits = 0, 1
    transfer = _transfers(2)[0]
    transfer.keys, transfer.device_indices = (
        transfer.keys[1:],
        transfer.device_indices[4:],
    )
    combo.lookup("later", [transfer])
    assert combo.reserve("later", 1)
    assert combo.load("later", [transfer])
    second = combo.start_layer_wise_loading()
    combo.l1.complete()
    assert combo.num_completed_loads() == 0
    combo.l2.complete()
    assert combo.num_completed_loads() == 1  # One L2 completion cannot cover both.
    assert set(combo.pop_completed_load()) == {"l1", "both-l2"}
    assert combo.num_completed_loads() == 0
    combo.l2.complete()
    assert combo.pop_completed_load() == ["later"]
    combo.layer_done_counter.set_consumer(first)  # Already complete on decode reuse.
    assert combo.l1.consumer == combo.l2.consumer == -1
    assert second > first


def test_owner_flush_sees_l2_loads_and_offloads(composite, caplog):
    combo, directory, _ = composite
    owner = combo.l1.owner
    owner.cache.linker.pending_loads["l2"] = object()
    assert not owner.begin_flush()
    owner.cache.linker.pending_loads.clear()
    owner.cache.linker.pending_offloads.append(object())
    assert not owner.begin_flush()
    assert "UMBP offloads" in caplog.text
    owner.cache.linker.pending_offloads.clear()
    assert owner.begin_flush()
    directory.thaw()


def test_umbp_failure_survives_future_consumption():
    from queue import Queue
    from types import SimpleNamespace

    from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import (
        LayerWiseLoadCounter,
        UMBPDirectLinker,
    )

    def fail():
        raise RuntimeError("injected ready-event failure")

    backend = UMBPDirectLinker.__new__(UMBPDirectLinker)
    backend.layer_done_counter = LayerWiseLoadCounter(1)
    index = backend.layer_done_counter.update_producer()
    backend._load_queue, backend._completed_loads = Queue(), Queue()
    backend._load_queue.put((index, ["req"], [], SimpleNamespace(synchronize=fail)))
    backend._load_queue.put(None)
    backend._load_thread_func()  # Run the real worker, deterministically.
    backend.layer_done_counter.set_consumer(index)
    with pytest.raises(RuntimeError, match="UMBP layer-wise"):
        backend.layer_done_counter.wait_until(0)
    assert not backend.layer_done_counter._futures
    with pytest.raises(RuntimeError, match="UMBP layer-wise"):
        backend.pop_completed_load()
    assert backend.num_completed_loads() == 0
