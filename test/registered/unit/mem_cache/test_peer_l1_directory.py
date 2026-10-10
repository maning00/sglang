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
