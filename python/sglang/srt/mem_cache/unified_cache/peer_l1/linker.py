"""Direct peer HBM prefix restoration for single-node DP attention."""

from __future__ import annotations

import json
import logging
import socket
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import msgspec
import numpy as np
import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinker
from sglang.srt.runtime_context import get_memory, get_parallel

from .channels import NativeGrant, NativeOwnerChannel
from .control import CleanupQueue, OwnerPeer, OwnerService, full_keys
from .copier import L1Copier, PeerMemoryMap, validate_ipc_backend
from .debug import CollectiveGuard
from .directory import L1Directory
from .indexer import IndexerClient, IndexerConfig, PlacementPublisher, Placements
from .owner import L1Owner
from .session import PeerSession

logger = logging.getLogger(__name__)

# Arbitrary; often enough to difference counters at benchmark phase boundaries.
_STATS_LOG_SECONDS = 60.0


class PeerInfo(msgspec.Struct, frozen=True):
    pin_cap: int
    device: int
    worker_id: str
    endpoint: str
    epoch: int


class Reservation(msgspec.Struct):
    peer: int
    keys: list[str]
    rows: np.ndarray
    grant: NativeGrant | None = None
    identities: np.ndarray | None = None


class LoadPlan(msgspec.Struct):
    rid: str
    peer: int
    keys: list[str]
    rows: np.ndarray
    indices: torch.Tensor
    cleanup: int


class PeerL1Linker(UnifiedCacheLinker):
    persists = False

    def __init__(self, server_args, params, *, components, extra_config=None):
        memory = get_memory()
        pool = params.token_to_kv_pool_allocator.get_kvcache()
        parallel = get_parallel()
        self.cpu_group = parallel.world_group.cpu_group
        self.rank = torch.distributed.get_rank(group=self.cpu_group)
        world = torch.distributed.get_world_size(group=self.cpu_group)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.page_size = params.page_size
        validate_ipc_backend(
            server_args=server_args, params=params, pool=pool, world=world
        )
        config = (
            memory.hicache_storage_backend_extra_config
            if extra_config is None
            else extra_config
        )
        config = json.loads(config) if isinstance(config, str) else dict(config or {})
        # Measured on DEP8 AgentX: 2048 caused reserve failures and truncated long
        # prefixes up to c128; 8192 cleared both at c64, 16384 adds headroom.
        self.pin_cap = int(config.get("pin_cap", 16384))
        self.wait_seconds = float(config.get("wait_timeout_ms", 200)) / 1000
        self.min_pages = int(config.get("min_hit_pages", 1))
        self.check_hooks = bool(config.get("check_hooks", False))
        self._collective_guard = CollectiveGuard() if self.check_hooks else None
        copy_engine = str(config.get("copy_engine", "kernel"))
        if copy_engine not in ("kernel", "sdma"):
            raise ValueError(f"Unknown peer_l1 copy_engine: {copy_engine!r}")
        layer_group, blocks = (
            int(config.get("layer_group", 8)),
            int(config.get("blocks", 32)),
        )
        if (
            min(self.pin_cap, self.min_pages, layer_group, blocks) <= 0
            or self.wait_seconds < 0
        ):
            raise ValueError(
                "peer_l1 sizes must be positive and wait_timeout_ms nonnegative"
            )
        self.indexer_config = IndexerConfig(**config.get("indexer", {}))
        self.indexer = IndexerClient(self.indexer_config)
        self.publisher = self.control = self.cleanup = self._rpc_pool = None
        self.placements = Placements(self.indexer)
        self.placement = None
        self.stats = Counter()
        self._next_stats_log = time.monotonic() + _STATS_LOG_SECONDS
        self._lookups = {}
        self._reservations = {}
        self._queued = {}
        self._batches = {}
        self._inflight = set()
        self._closed = False
        self._flush_frozen = False
        self.owner = None
        self.forward_stream = None
        self.pool_group = resolve_hybrid_device_pool_group(
            kvcache=pool,
            page_size=params.page_size,
            params=params,
            components=components,
        )
        rows = min(entry._row_count for entry in self.pool_group.entries)
        self.worker_id = f"{self.indexer_config.deployment}/prefill/{socket.gethostname()}/dp{self.rank}"
        self.directory = L1Directory(rows, self.pin_cap)
        self.control = OwnerService(self.directory)
        self.cleanup = CleanupQueue(timeout_s=self.indexer_config.timeout_s)
        self._rpc_pool = ThreadPoolExecutor(
            max_workers=world - 1, thread_name_prefix="l1-query"
        )
        self.controls = {}
        try:
            self.memory = PeerMemoryMap(
                self.pool_group,
                self.rank,
                world,
                self.device,
                self.cpu_group,
                PeerInfo(
                    self.pin_cap,
                    self.device.index,
                    self.worker_id,
                    self.control.endpoint,
                    self.control.epoch,
                ),
            )
            device_ids = [info.device for info in self.memory.directory_infos]
            if len(set(device_ids)) != world:
                raise ValueError("peer_l1 requires a distinct GPU per DP worker")
            self.worker_ids = {
                peer: info.worker_id
                for peer, info in enumerate(self.memory.directory_infos)
                if peer != self.rank
            }
            self.controls = {
                peer: OwnerPeer(
                    endpoint=info.endpoint,
                    epoch=info.epoch,
                    pin_cap=info.pin_cap,
                    cleanup=self.cleanup,
                    timeout_s=self.indexer_config.timeout_s,
                    cooldown_s=self.indexer_config.cooldown_s,
                )
                for peer, info in enumerate(self.memory.directory_infos)
                if peer != self.rank
            }
            self.publisher = PlacementPublisher(
                config=self.indexer_config,
                worker_id=self.worker_id,
                client=self.indexer,
            )
            torch.distributed.barrier(group=self.cpu_group)
            self.copier = L1Copier(
                self.memory,
                layers=self.pool_group.num_layers,
                layer_group=layer_group,
                blocks=blocks,
                verify=bool(config.get("verify_bytes", False)),
                priority=int(config.get("stream_priority", -1)),
                engine=copy_engine,
                sdma_channels=int(config.get("sdma_channels", 2)),
                peer_devices={
                    peer: info.device
                    for peer, info in enumerate(self.memory.directory_infos)
                    if peer != self.rank
                },
            )
            self.sessions = {
                peer: PeerSession(
                    endpoint=info.endpoint,
                    owner=NativeOwnerChannel(
                        peer=self.controls[peer], worker_id=info.worker_id
                    ),
                    mover=self.copier,
                    mapping=self.memory,
                    cost="local",
                )
                for peer, info in enumerate(self.memory.directory_infos)
                if peer != self.rank
            }
            self.layer_done_counter = self.copier.counter
        except BaseException:
            self._close_control()
            raise
        logger.info(
            "peer_l1 rank=%d: %d page rows, %d buffers, %d peers, pin_cap=%d",
            self.rank,
            rows,
            len(self.memory.buffers),
            world - 1,
            self.pin_cap,
        )

    def bind_cache(self, cache):
        self.owner = L1Owner(cache, self)
        cache.peer_l1 = self

    @property
    def has_unstarted_reservations(self):
        # Derived from all requests; aborting one must not clear another's guard.
        return any(rid not in self._inflight for rid in self._reservations)

    @property
    def has_loads(self):
        return bool(self._reservations or self._queued or self._batches)

    def lookup(self, rid, transfers):
        self._lookups.pop(rid, None)
        # A request can have only one reservation at a time.
        if rid in self._reservations:
            return []
        kv = next(t for t in transfers if t.name == PoolName.KV)
        keys = kv.keys
        self.placement = self.placements.get(rid, keys, eligible=kv.indexer_eligible)
        self.stats["indexer_lookup_rpcs"] = self.indexer.rpc_count
        start = self.stats["lookups"] % len(self.controls)
        best = self.discover(keys, start=start)
        self.stats["lookups"] += 1
        self.stats["lookup_tail_pages"] += len(keys)
        if best is None or len(best.rows) < self.min_pages:
            return []
        self.stats["lookup_hits"] += 1
        self.stats["lookup_hit_pages"] += len(best.rows)
        selected_cap = self.controls[best.peer].pin_cap
        if len(best.rows) == selected_cap < len(keys):
            self.stats["lookup_capped"] += 1
        self._lookups[rid] = best
        return list(range(1, len(best.rows) + 1))

    def discover(self, keys, *, start):
        order = list(self.controls)
        order = order[start:] + order[:start]
        matches = self.placement.matches
        order = [p for p in order if self.controls[p].available()]
        if not order:
            return None
        if matches is not None:
            lengths = {}
            for peer in order:
                held = matches.get(self.worker_ids[peer], set())
                cap = min(len(keys), self.controls[peer].pin_cap)
                lengths[peer] = next(
                    (i for i in range(cap) if self.placement.hashes[i] not in held), cap
                )
        else:
            queries = {
                p: self._rpc_pool.submit(self.controls[p].lookup, self.placement.full)
                for p in order
            }
            lengths = {p: f.result() for p, f in queries.items()}
        peer = max(order, key=lambda p: lengths[p])
        count = lengths[peer]
        if not count:
            return None
        return Reservation(
            peer,
            list(keys[:count]),
            np.empty(count, dtype=np.int64),
            identities=self.placement.full[:count],
        )

    def contains_peers(self, keys):
        found = np.zeros(len(keys), dtype=bool)
        keys = full_keys(keys)
        queries = [
            self._rpc_pool.submit(peer.contains, keys)
            for peer in self.controls.values()
        ]
        answers = [query.result() for query in queries]
        if any(answer is None for answer in answers):
            self.stats["contains_failures"] += 1
            return found
        for answer in answers:
            found |= answer
        return found

    def reserve(self, rid, num_pages):
        found = self._lookups.pop(rid, None)
        if rid in self._reservations or found is None or num_pages > len(found.rows):
            return False
        found = Reservation(
            found.peer,
            found.keys[:num_pages],
            found.rows[:num_pages],
            identities=found.identities[:num_pages],
        )
        # Install before pinning so a conflicting debugger/profiler cannot
        # leave an untracked reservation behind.
        if self._collective_guard is not None:
            self._collective_guard.enable()
        try:
            self.stats["reserve_attempts"] += 1
            granted = self.sessions[found.peer].owner.acquire_exact(found.identities)
            if granted is None:
                self.stats["reserve_failures"] += 1
                self.stats["reserve_failed_pages"] += num_pages
                return False
            found.rows, found.grant = granted.rows, granted
            self.stats["reserved_pages"] += num_pages
            self._reservations[rid] = found
            return True
        finally:
            self._stop_collective_guard_if_submitted()

    def _stop_collective_guard_if_submitted(self):
        if self._collective_guard is not None and not self.has_unstarted_reservations:
            self._collective_guard.disable()

    def _forget_reservation(self, rid):
        return self._reservations.pop(rid)

    def release_reservation(self, rid, *, keep_placement=False):
        if not keep_placement:
            self.placements.pop(rid)
        self._lookups.pop(rid, None)
        if (
            rid in self._queued
            or rid in self._inflight
            or rid not in self._reservations
        ):
            return
        reservation = self._forget_reservation(rid)
        self.sessions[reservation.peer].owner.release(reservation.grant)
        self._stop_collective_guard_if_submitted()

    def load(self, rid, transfers):
        transfer = next(t for t in transfers if t.name == PoolName.KV)
        reservation = self._reservations[rid]
        if list(transfer.keys) != reservation.keys:
            raise ValueError("peer_l1 load does not match its reservation")
        self._queued[rid] = LoadPlan(
            rid,
            reservation.peer,
            list(transfer.keys),
            reservation.rows,
            transfer.device_indices,
            reservation.grant.cleanup,
        )
        return True

    def start_layer_wise_loading(self):
        if not self._queued:
            return -1
        if self.forward_stream is None:
            raise RuntimeError("peer_l1 forward stream was not registered")
        plans = list(self._queued.values())
        # A failed submission may already have queued source reads or callbacks.
        # Keep those reservations protected until engine teardown.
        self._inflight.update(plan.rid for plan in plans)
        batch = self.copier.submit(plans, self.forward_stream)
        self._batches[batch.index] = batch
        self._queued.clear()
        self._stop_collective_guard_if_submitted()
        self.stats["loaded_pages"] += len(batch.keys)
        return batch.index

    def cancel_queued_load(self, rid):
        if self._queued.pop(rid, None) is None:
            return False
        self.release_reservation(rid)
        return True

    def poll(self):
        self.owner.poll(self.forward_stream)
        now = time.monotonic()
        if now >= self._next_stats_log:
            self._next_stats_log = now + _STATS_LOG_SECONDS
            self._log_stats()

    def _log_stats(self):
        # These counters are cumulative snapshots, not increments.
        for key, value in self.indexer.stats_snapshot().items():
            self.stats[key] = value
        logger.info(
            "peer_l1 rank=%d stats: %s deferred_tokens=%d own_pinned_refs=%d",
            self.rank,
            dict(sorted(self.stats.items())),
            self.owner.deferred_tokens if self.owner is not None else 0,
            self.directory.pinned_refs,
        )

    def finish_layer_wise_loading(self, counter_index):
        batch = self._batches.get(counter_index)
        if batch is None:
            return True
        self.layer_done_counter.finish(counter_index)
        failed = batch.failed_requests()
        if failed:
            logger.error(
                "peer_l1 byte verification failed at copy work item %d",
                batch.error_cpu.item() - 1,
            )
            self.stats["failed_load_batches"] += 1
        for rid in batch.rids:
            self._inflight.remove(rid)
            self._forget_reservation(rid)
        del self._batches[counter_index]
        return not failed

    def offload(self, transfers):
        return False

    def num_completed_offloads(self):
        return 0

    def pop_completed_offload(self):
        raise RuntimeError("peer_l1 does not offload")

    def begin_flush(self):
        # A paused scheduler skips the usual offload poll. Loads have already
        # been finished by run_batch, before result processing can free slots.
        wrapper = self.owner.cache.linker
        wrapper.commit_completed_offloads(
            wrapper.take_completed_offloads(wrapper.num_completed_offloads())
        )
        self._flush_frozen = self.owner.begin_flush()
        return self._flush_frozen

    def end_flush(self):
        if self._flush_frozen:
            self.directory.thaw()
            self._flush_frozen = False

    def reset(self):
        owns_freeze = not self._flush_frozen
        if owns_freeze and not self.begin_flush():
            raise RuntimeError("peer_l1 reset requires a quiescent cache")
        try:
            self.owner.reset()
            self._lookups.clear()
            self.placements.clear()
            self.layer_done_counter.reset()
        finally:
            if owns_freeze:
                self.end_flush()

    def close(self):
        if self._closed:
            return
        self._log_stats()
        self.copier.stream.synchronize()
        for batch in self._batches.values():
            for rid in batch.rids:
                self._inflight.remove(rid)
                self._forget_reservation(rid)
        self._batches.clear()
        self._queued.clear()
        for rid in list(self._reservations):
            self.release_reservation(rid)
        if self.owner is not None:
            self.owner.stream.synchronize()
        # Process shutdown is collective; producers must outlive IPC consumers.
        torch.distributed.barrier(group=self.cpu_group)
        self.memory.peer_storages.clear()
        torch.distributed.barrier(group=self.cpu_group)
        self._close_control()
        self._closed = True

    def _close_control(self):
        if self.publisher:
            self.publisher.close()
        if self._rpc_pool:
            self._rpc_pool.shutdown()
        if self.cleanup:
            self.cleanup.close()
        if self.control:
            self.control.close()
        self.directory.close()
        if self.indexer:
            self.indexer.close()
