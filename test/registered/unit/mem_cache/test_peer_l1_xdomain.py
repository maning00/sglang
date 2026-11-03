"""Cross-domain policy, lease replay, and backend-local IPC slot invariants."""

import hashlib
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from sglang.srt.mem_cache.unified_cache.peer_l1.control import (
    Operation,
    OwnerPeer,
    OwnerService,
    full_keys,
)
from sglang.srt.mem_cache.unified_cache.peer_l1.directory import L1Directory
from sglang.srt.mem_cache.unified_cache.peer_l1.linker import PeerL1Linker
from sglang.srt.mem_cache.unified_cache.peer_l1.topology import DeviceInfo, ipc_sets
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_replayed_leased_grant_does_not_extend_its_lifetime():
    directory = L1Directory(16, 8)
    service = OwnerService(directory, endpoint="tcp://127.0.0.1:0", lease_s=0.5)
    key = hashlib.sha256(b"leased").hexdigest()
    peer = OwnerPeer(
        endpoint=service.endpoint,
        epoch=service.epoch,
        pin_cap=8,
        cleanup=None,
        timeout_s=1,
    )
    keys = full_keys([key])
    try:
        service.publish([key], [3])
        start = time.monotonic()
        assert peer.request(Operation.ACQUIRE_LEASED, keys, seq=1).tolist() == [3]
        time.sleep(0.35)
        assert peer.request(Operation.ACQUIRE_LEASED, keys, seq=1).tolist() == [3]
        assert directory.pinned_refs == 1
        while directory.pinned_refs and time.monotonic() < start + 0.7:
            time.sleep(0.005)
        assert directory.pinned_refs == 0
        assert peer.request(Operation.ACQUIRE_LEASED, keys, seq=1).tolist() == []
    finally:
        service.close()
        directory.close()


def test_remote_prefix_threshold_never_discards_the_only_hit():
    linker = PeerL1Linker.__new__(PeerL1Linker)
    linker.domain = SimpleNamespace(min_gain_pages=3)
    linker.controls = {
        p: SimpleNamespace(available=lambda: True, pin_cap=8) for p in (0, 1)
    }
    linker.sessions = {
        0: SimpleNamespace(cost="local"),
        1: SimpleNamespace(cost="rdma"),
    }
    linker.worker_ids = {0: "local", 1: "remote"}
    linker.placement = SimpleNamespace(
        matches={"local": {0, 1}, "remote": set(range(4))},
        hashes=list(range(8)),
        full=np.zeros((8, 4), dtype=np.uint64),
    )
    assert linker.discover(list(range(8)), start=0).peer == 0
    linker.placement.matches["remote"].add(4)
    assert linker.discover(list(range(8)), start=0).peer == 1
    linker.placement.matches = {"local": set(), "remote": {0}}
    assert linker.discover(list(range(8)), start=0).peer == 1


def test_f3_excludes_foreign_instances_domains_and_policies():
    linker = PeerL1Linker.__new__(PeerL1Linker)

    def desc(instance="self", domain="dram-a", policy="last"):
        return SimpleNamespace(
            instance_id=instance, l2_domain=domain, preservation=policy
        )

    linker.descriptor = desc()
    linker._descriptors = {
        0: desc(),
        1: desc(instance="foreign"),
        2: desc(domain="dram-b"),
        3: desc(policy="none"),
    }
    calls = []

    def contains(p):
        calls.append(p)
        return np.array([p != 0])

    linker.controls = {
        p: SimpleNamespace(contains=lambda _, p=p: contains(p)) for p in range(4)
    }
    linker.stats = Counter()
    linker._rpc_pool = ThreadPoolExecutor(4)
    try:
        assert not linker.contains_peers([hashlib.sha256(b"key").hexdigest()])[0]
        assert calls == [0]
    finally:
        linker._rpc_pool.shutdown()


def test_global_rank_count_does_not_overflow_ipc_slots():
    devices = [
        DeviceInfo(
            str(i // 4),
            f"gpu-{i}",
            tuple(f"gpu-{j}" for j in range(i // 4 * 4, i // 4 * 4 + 4)),
        )
        for i in range(16)
    ]
    peers = ipc_sets(devices)
    assert peers[12] == (13, 14, 15) and peers[0] == (1, 2, 3)
    devices[1] = DeviceInfo("0", "gpu-1", ("gpu-1",))
    assert 1 not in ipc_sets(devices)[0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
