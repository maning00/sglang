"""Direct peer HBM prefix restoration for single-node DP attention."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinker
from sglang.srt.runtime_context import get_memory, get_parallel

from .copier import L1Copier, PeerMemoryMap
from .debug import CollectiveGuard
from .directory import L1Directory, hash_words
from .owner import L1Owner

logger = logging.getLogger(__name__)

# Arbitrary; often enough to difference counters at benchmark phase boundaries.
_STATS_LOG_SECONDS = 60.0


@dataclass
class Reservation:
    peer: int
    keys: list[str]
    rows: np.ndarray
    gens: np.ndarray


@dataclass
class LoadPlan:
    rid: str
    peer: int
    keys: list[str]
    rows: np.ndarray
    indices: torch.Tensor
    directory: L1Directory
    reserved_rows: np.ndarray


class PeerL1Linker(UnifiedCacheLinker):
    persists = False

    def __init__(self, server_args, params, *, components, extra_config=None):
        if not torch.version.hip or params.page_size != 256:
            raise ValueError("peer_l1 requires ROCm and page_size=256")
        memory = get_memory()
        if memory.enable_unified_memory or server_args.enable_memory_saver:
            raise ValueError(
                "peer_l1 requires fixed KV allocations without memory saver"
            )
        allocator_config = os.environ.get(
            "PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        )
        if (
            "expandable_segments:True" in allocator_config
            or "backend:cudaMallocAsync" in allocator_config
        ):
            raise ValueError(
                "peer_l1 requires the native, non-expandable PyTorch allocator"
            )
        pool = params.token_to_kv_pool_allocator.get_kvcache()
        if not getattr(pool, "_unified_kv", False):
            raise ValueError("peer_l1 currently supports DeepSeek V4 unified KV only")
        parallel = get_parallel()
        self.cpu_group = parallel.world_group.cpu_group
        self.rank = torch.distributed.get_rank(group=self.cpu_group)
        world = torch.distributed.get_world_size(group=self.cpu_group)
        if (
            not 2 <= world <= 8
            or params.pp_size != 1
            or params.attn_cp_size != 1
            or torch.distributed.get_world_size(group=params.tp_cache_group) != 1
        ):
            raise ValueError(
                "peer_l1 requires 2–8 local DP workers with attention TP=1, CP=1, PP=1"
            )
        self.device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.device_count() < world:
            raise ValueError("Every peer_l1 worker must see every participating GPU")
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
        self.stitch_stats = bool(config.get("stitch_stats", False))
        self._collective_guard = CollectiveGuard() if self.check_hooks else None
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
        path = f"/dev/shm/sglang_l1_{uuid.uuid4().hex}_{self.rank}"
        self.directory = L1Directory(path, rows, self.pin_cap, create=True)
        self.peers = {}
        try:
            self.memory = PeerMemoryMap(
                self.pool_group,
                self.rank,
                world,
                self.device,
                self.cpu_group,
                (path, rows, self.pin_cap, self.device.index),
            )
            device_ids = [info[3] for info in self.memory.directory_infos]
            if len(set(device_ids)) != world:
                raise ValueError("peer_l1 requires a distinct GPU per DP worker")
            self.peers = {
                peer: L1Directory(*info[:3], create=False)
                for peer, info in enumerate(self.memory.directory_infos)
                if peer != self.rank
            }
            self.copier = L1Copier(
                self.memory,
                layers=self.pool_group.num_layers,
                layer_group=layer_group,
                blocks=blocks,
                verify=bool(config.get("verify_bytes", False)),
                priority=int(config.get("stream_priority", -1)),
            )
            self.layer_done_counter = self.copier.counter
        except BaseException:
            for directory in self.peers.values():
                directory.close()
            self.directory.close()
            raise
        logger.info(
            "peer_l1 rank=%d: %d page rows, %d buffers, %d peers, pin_cap=%d",
            self.rank,
            rows,
            len(self.memory.buffers),
            len(self.peers),
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
        keys = next(
            transfer.keys for transfer in transfers if transfer.name == PoolName.KV
        )
        hashes = hash_words(keys)
        best = None
        # Rotate the first peer so ties distribute load without an RNG call.
        peers = list(self.peers)
        start = self.stats["lookups"] % len(peers)
        for peer in peers[start:] + peers[:start]:
            rows, gens = self.peers[peer].probe(hashes)
            if len(rows) and (best is None or len(rows) > len(best.rows)):
                best = Reservation(peer, list(keys[: len(rows)]), rows, gens)
        self.stats["lookups"] += 1
        self.stats["lookup_tail_pages"] += len(keys)
        if self.stitch_stats and best is not None:
            self._count_stitchable(hashes, start=len(best.rows))
        if best is None or len(best.rows) < self.min_pages:
            return []
        self.stats["lookup_hits"] += 1
        self.stats["lookup_hit_pages"] += len(best.rows)
        if len(best.rows) == self.pin_cap < len(keys):
            self.stats["lookup_capped"] += 1
        self._lookups[rid] = best
        return list(range(1, len(best.rows) + 1))

    def _count_stitchable(self, hashes, *, start):
        """Diagnostic: the prefix restorable if consecutive runs could come
        from different peers, against the single-peer prefix ending at start."""
        position = start
        while position < len(hashes):
            run = max(
                len(peer.probe(hashes[position:])[0]) for peer in self.peers.values()
            )
            if run == 0:
                break
            position += run
        self.stats["lookup_single_peer_pages"] += start
        self.stats["lookup_stitched_pages"] += position

    def reserve(self, rid, num_pages):
        found = self._lookups.pop(rid, None)
        if rid in self._reservations or found is None or num_pages > len(found.rows):
            return False
        found = Reservation(
            found.peer,
            found.keys[:num_pages],
            found.rows[:num_pages],
            found.gens[:num_pages],
        )
        # Install before pinning so a conflicting debugger/profiler cannot
        # leave an untracked reservation behind.
        if self._collective_guard is not None:
            self._collective_guard.enable()
        try:
            if not self.peers[found.peer].pin(found.rows, found.gens):
                self.stats["reserve_failures"] += 1
                self.stats["reserve_failed_pages"] += num_pages
                return False
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

    def release_reservation(self, rid):
        self._lookups.pop(rid, None)
        if (
            rid in self._queued
            or rid in self._inflight
            or rid not in self._reservations
        ):
            return
        reservation = self._forget_reservation(rid)
        self.peers[reservation.peer].unpin(reservation.rows)
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
            self.peers[reservation.peer],
            reservation.rows,
        )
        return True

    def start_layer_wise_loading(self):
        if not self._queued:
            return -1
        if self.forward_stream is None:
            raise RuntimeError("peer_l1 forward stream was not registered")
        plans = list(self._queued.values())
        batch = self.copier.submit(plans, self.forward_stream)
        self._batches[batch.index] = batch
        self._inflight.update(batch.rids)
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
        error = batch.error_cpu.item() if batch.error_cpu is not None else 0
        if error:
            logger.error(
                "peer_l1 byte verification failed at copy work item %d", error - 1
            )
            self.stats["failed_load_batches"] += 1
        for rid in batch.rids:
            self._inflight.remove(rid)
            self._forget_reservation(rid)
        del self._batches[counter_index]
        return not error

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
        for directory in self.peers.values():
            directory.close()
        self.directory.close()
        self._closed = True
