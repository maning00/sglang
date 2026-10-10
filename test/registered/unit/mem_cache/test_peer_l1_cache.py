"""Exercise the new lifecycle through real tree/allocator operations."""

import ctypes
import threading
from array import array
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.peer_l1.directory import L1Directory, hash_words
from sglang.srt.mem_cache.unified_cache.peer_l1.owner import L1Owner
from sglang.test.ci.ci_register import register_cuda_ci
from test_unified_radix_cache_unittest import CacheConfig, build_fixture

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


def test_hip_event_waits_for_host_callback():
    if not torch.version.hip:
        pytest.skip("HIP callback contract")
    hip = ctypes.CDLL("libamdhip64.so")
    callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    hip.hipLaunchHostFunc.argtypes = [ctypes.c_void_p, callback_type, ctypes.c_void_p]
    hip.hipLaunchHostFunc.restype = ctypes.c_int
    entered, release = threading.Event(), threading.Event()

    @callback_type
    def callback(_):
        entered.set()
        release.wait(10)

    stream, finish = torch.cuda.Stream(), torch.cuda.Event()
    try:
        assert hip.hipLaunchHostFunc(stream.cuda_stream, callback, None) == 0
        finish.record(stream)
        assert entered.wait(10)
        assert not finish.query()
    finally:
        release.set()
        finish.synchronize()


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_pinned_eviction_compensates_and_reclaims(tmp_path, backend):
    cache, allocator, _ = build_fixture(
        CacheConfig(page_size=4, kv_size=64, num_layers=2), tree_core_backend=backend
    )
    if backend == "rust":
        assert type(cache.tree_core).__name__.startswith(
            "Rust"
        ), "Rust coverage must not silently fall back"
    cache.tree_core.enable_external_cache_linker = True
    cache.write_through_threshold = (1 << 63) - 1
    directory = L1Directory(str(tmp_path / "directory"), 17, 4, create=True)
    linker = SimpleNamespace(
        directory=directory,
        device=allocator.device,
        loading_keys=Counter(),
        check_hooks=True,
        stats=Counter(),
        has_unstarted_reservations=False,
        wait_seconds=0,
    )
    owner = L1Owner(cache, linker)
    linker.owner = owner
    cache.peer_l1 = linker
    try:
        first = allocator.alloc(8)
        cache.insert(InsertParams(key=RadixKey(array("q", range(8))), value=first))
        second = allocator.alloc(8)
        cache.insert(
            InsertParams(key=RadixKey(array("q", range(100, 108))), value=second)
        )
        owner.poll(torch.cuda.current_stream())
        owner.stream.synchronize()
        owner.poll(torch.cuda.current_stream())
        first_rows = (first.reshape(-1, 4)[:, 0] // 4).cpu().numpy()
        first_keys = [
            key
            for key in owner._published
            if directory.probe(hash_words([key]))[0][0] in first_rows
        ]
        rows_and_gens = [directory.probe(hash_words([key])) for key in first_keys]
        # Pin separately so key-set ordering cannot affect the assertion.
        for rows, gens in rows_and_gens:
            assert directory.pin(rows, gens)
        # The older leaf is pinned; its replacement must be compensated by
        # another eviction, without making those still-readable rows reusable.
        result = cache.evict(EvictParams(num_tokens=8))
        assert result.num_tokens_evicted == 16
        assert allocator.available_size() == 56
        assert owner.deferred_tokens == 8
        assert linker.stats["compensation_rounds"] >= 1
        for rows, _ in rows_and_gens:
            directory.unpin(rows)
        assert owner.reclaim() == 8
        assert allocator.available_size() == 64
        assert owner.deferred_tokens == 0
        # A genuine shortage (no remote reader can help) keeps best-effort
        # eviction semantics; the caller can still retract or reject a load.
        result = cache.evict_for_alloc(EvictParams(num_tokens=128))
        assert result.num_tokens_evicted == 0
        assert linker.stats["capacity_waits"] == 0

        # Fill the allocator with active pages plus one evictable leaf. A local
        # unstarted reservation must prevent waiting on remote readers.
        occupied = allocator.alloc(64)
        cache.insert(
            InsertParams(key=RadixKey(array("q", range(200, 208))), value=occupied[:8])
        )
        owner.poll(torch.cuda.current_stream())
        owner.stream.synchronize()
        owner.poll(torch.cuda.current_stream())
        rows, gens = directory.probe(hash_words(sorted(owner._published)))
        assert len(rows) == 2 and directory.pin(rows, gens)
        linker.has_unstarted_reservations = True
        cache.evict_for_alloc(EvictParams(num_tokens=8))
        assert owner.deferred_tokens == 8 and allocator.available_size() == 0
        assert linker.stats["capacity_waits"] == 0

        # Once this worker has submitted its copies, a genuine delayed unpin
        # can satisfy the shortage without another eviction or a collective.
        linker.has_unstarted_reservations = False
        linker.wait_seconds = 2
        stop = threading.Event()

        def release_when_waiting():
            while not stop.wait(0.001):
                if linker.stats["capacity_waits"]:
                    directory.unpin(rows)
                    return

        reader = threading.Thread(target=release_when_waiting)
        reader.start()
        try:
            cache.evict_for_alloc(EvictParams(num_tokens=8))
            assert linker.stats["capacity_waits"] == 1
            assert owner.deferred_tokens == 0 and allocator.available_size() == 8
        finally:
            stop.set()
            reader.join()
        allocator.free(occupied[8:])
    finally:
        owner.stream.synchronize()
        directory.close()


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_split_preserves_publications_and_eviction_retires_them(backend):
    cache, allocator, _ = build_fixture(
        CacheConfig(page_size=4, kv_size=64, num_layers=2), tree_core_backend=backend
    )
    if backend == "rust":
        assert type(cache.tree_core).__name__.startswith("Rust")
    core = cache.tree_core
    core.enable_external_cache_linker = True
    cache.write_through_threshold = (1 << 63) - 1
    core.enable_l1_events = True
    value = allocator.alloc(12)
    cache.insert(InsertParams(key=RadixKey(array("q", range(12))), value=value))
    events = core.take_l1_events()
    assert len(events) == 1 and len(events[0][0]) == 3
    keys, attached = events[0]
    assert torch.equal(attached, value)
    cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", range(4)))))
    assert core.take_l1_events() == []
    cache.evict(EvictParams(num_tokens=12))
    retired = core.take_l1_events()
    assert all(value is None for _, value in retired)
    assert sorted(key for hashes, _ in retired for key in hashes) == sorted(keys)
