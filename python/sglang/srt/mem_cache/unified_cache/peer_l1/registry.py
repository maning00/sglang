"""Instance-owned discovery snapshots; resolution never runs on request ingress."""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
import urllib.request

import msgspec

logger = logging.getLogger(__name__)


class DomainConfig(msgspec.Struct, frozen=True, kw_only=True):
    enabled: bool = False
    instance_id: str = ""
    kv_compatibility_id: str = ""
    seeds: tuple[str, ...] = ()
    listen_host: str = ""
    restart_group: str = ""
    l2_domain: str = ""
    min_gain_pages: int = 1

    def __post_init__(self):
        if self.enabled and not (
            self.instance_id and self.kv_compatibility_id and self.seeds
        ):
            raise ValueError(
                "cross-instance sharing requires instance_id, kv_compatibility_id and seeds"
            )
        if "/" in self.instance_id or len(self.seeds) > 128 or self.min_gain_pages < 1:
            raise ValueError("invalid peer domain configuration")
        if self.seeds and not self.enabled:
            raise ValueError("peer domain seeds require enabled=true")


class Topology(msgspec.Struct, frozen=True):
    instance_id: str
    hosts: tuple[str, ...]


def topology(config: DomainConfig) -> Topology:
    import torch.distributed as dist

    from sglang.srt.runtime_context import get_parallel, get_resources

    resources = get_resources()
    if resources.peer_l1_topology is None:
        group = get_parallel().world_group.cpu_group
        entries = [None] * dist.get_world_size(group)
        identity = (config.enabled, config.instance_id, config.kv_compatibility_id)
        dist.all_gather_object(entries, (socket.gethostname(), identity), group=group)
        if any(entry[1] != identity for entry in entries):
            raise ValueError("peer domain identity must agree across instance ranks")
        resources.peer_l1_topology = Topology(
            config.instance_id or entries[0][0], tuple(e[0] for e in entries)
        )
    return resources.peer_l1_topology


def initialize_identity():
    from sglang.srt.runtime_context import get_disagg, get_memory

    memory = get_memory()
    if not (
        get_disagg().disaggregation_decode_lend_kv
        or (
            memory.enable_unified_cache_external_linker
            and memory.unified_cache_external_linker_backend
            in ("peer_l1", "mori_peer_l1")
        )
    ):
        return
    raw = memory.hicache_storage_backend_extra_config
    config = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
    topology(DomainConfig(**config.get("peer_l1", config).get("domain", {})))


class PeerDescriptor(msgspec.Struct, frozen=True, kw_only=True):
    worker_id: str
    instance_id: str
    rank: int
    role: str
    epoch: int
    kv_compatibility_id: str
    restart_group: str
    pin_cap: int
    control_endpoints: tuple[str, ...]
    data_capabilities: tuple[str, ...]
    l2_domain: str
    preservation: str


class OwnerManifest(msgspec.Struct, frozen=True, kw_only=True):
    deployment: str
    instance_id: str
    owners: tuple[PeerDescriptor, ...]
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError("unsupported owner manifest schema")
        seen = set()
        for owner in self.owners:
            expected = (
                f"{self.deployment}/{self.instance_id}/{owner.role}/dp{owner.rank}"
            )
            if (
                owner.instance_id != self.instance_id
                or owner.worker_id != expected
                or expected in seen
            ):
                raise ValueError("inconsistent owner manifest")
            seen.add(expected)


class OwnerHello(msgspec.Struct, frozen=True, kw_only=True):
    descriptor: PeerDescriptor
    data: bytes = b""
    lease_s: float = 30.0


def cache_manifest(cache):
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    if isinstance(cache, UnifiedRadixCache) and cache.peer_l1 is not None:
        return msgspec.json.encode(cache.peer_l1.manifest)
    return None


def merge_manifests(infos):
    values = [i["peer_l1_owners"] for i in infos if i.get("peer_l1_owners") is not None]
    if values and any(value != values[0] for value in values):
        raise ValueError("inconsistent instance owner snapshots")
    return values[0] if values else None


class Resolver:
    def __init__(
        self, *, seeds, deployment, instance_id, compatibility_id, connect, retire=None
    ):
        self.seeds = tuple(dict.fromkeys(seeds))
        self.deployment, self.instance_id = deployment, instance_id
        self.compatibility_id, self.connect = compatibility_id, connect
        self.retire = retire
        self.snapshot = {}
        self._wake, self._stop = threading.Event(), threading.Event()
        self._invalid = set()
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run, name="peer-resolver", daemon=True
        )
        self._thread.start()

    def refresh(self, worker=None):
        if worker is not None:
            with self._lock:
                self._invalid.add(worker)
        self._wake.set()

    def _read(self, seed):
        with urllib.request.urlopen(
            seed.rstrip("/") + "/peer_l1/owners", timeout=1.0
        ) as response:
            payload = response.read((1 << 20) + 1)
        if len(payload) > 1 << 20:
            raise ValueError("owner manifest exceeds 1 MiB")
        manifest = msgspec.json.decode(payload, type=OwnerManifest)
        if manifest.deployment != self.deployment:
            raise ValueError("peer belongs to another deployment")
        return manifest

    def _run(self):
        previous, delay = {}, 1.0
        while not self._stop.is_set():
            candidates, conflicts, failed = {}, set(), False
            for seed in self.seeds:
                if self._stop.is_set():
                    return
                try:
                    manifest = self._read(seed)
                    if manifest.instance_id == self.instance_id:
                        continue
                    for owner in manifest.owners:
                        if owner.kv_compatibility_id != self.compatibility_id:
                            continue
                        if (
                            owner.worker_id in candidates
                            and candidates[owner.worker_id] != owner
                        ):
                            conflicts.add(owner.worker_id)
                        candidates[owner.worker_id] = owner
                except Exception:
                    failed = True
                    logger.debug(
                        "Peer manifest refresh failed for %s", seed, exc_info=True
                    )
            with self._lock:
                invalid, self._invalid = self._invalid, set()
            resolved = {}
            for worker, descriptor in candidates.items():
                if self._stop.is_set():
                    return
                if worker in conflicts:
                    continue
                old = previous.get(worker)
                try:
                    value = (
                        old[1]
                        if old is not None
                        and old[0] == descriptor
                        and worker not in invalid
                        else self.connect(descriptor)
                    )
                    if value is not None:
                        resolved[worker] = (descriptor, value)
                    else:
                        failed = True
                except Exception:
                    failed = True
                    logger.debug("Peer connection failed for %s", worker, exc_info=True)
            if self.retire is not None:
                for worker, (_, old) in previous.items():
                    current = resolved.get(worker)
                    if current is None or current[1] is not old:
                        self.retire(old)
            previous = resolved
            self.snapshot = resolved
            delay = min(delay * 2, 10.0) if failed else 1.0
            # Refresh requests coalesce but cannot bypass retry backoff.
            deadline = time.monotonic() + delay
            while not self._stop.is_set() and time.monotonic() < deadline:
                self._wake.wait(max(0, deadline - time.monotonic()))
                self._wake.clear()

    def close(self):
        self._stop.set()
        self._wake.set()
        self._thread.join()
