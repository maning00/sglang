"""Restore a request from peer HBM, from UMBP, or from a peer prefix that UMBP
continues (last-replica writes only)."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, replace

import numpy as np
import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinker

logger = logging.getLogger(__name__)


class CompositeLayerCounter:
    def __init__(self, l1, l2):
        if l1.layer_done_counter.num_layers != l2.layer_done_counter.num_layers:
            raise ValueError("L1 and L2 must cover the same logical layers")
        self.l1, self.l2 = l1, l2
        self.num_layers = l1.layer_done_counter.num_layers
        self.reset()

    def register(self, l1_index, l2_index):
        index = self._next_index
        self._next_index += 1
        self._indices[index] = (l1_index, l2_index)
        return index

    def set_consumer(self, index):
        if index >= self._next_index:
            raise ValueError("Unknown composite load index")
        # Decode can retain an old index whose loads have already finished.
        a, b = self._indices.get(index, (-1, -1))
        self.l1.layer_done_counter.set_consumer(a)
        self.l2.layer_done_counter.set_consumer(b)

    def wait_until(self, layer):
        self.l1.layer_done_counter.wait_until(layer)
        self.l2.layer_done_counter.wait_until(layer)

    def finish(self, index):
        indices = self._indices.get(index)
        if indices is None:
            return True
        a, b = indices
        # L1's backend also checks bytes and releases its batch state.
        # Never short-circuit the second completion.
        landed_l1 = self.l1.finish_layer_wise_loading(a)
        landed_l2 = self.l2.finish_layer_wise_loading(b)
        del self._indices[index]
        return landed_l1 and landed_l2

    def reset(self):
        self._indices = {}
        self._next_index = 0


@dataclass
class _Restore:
    l1_pages: int
    l2_pages: int
    queued: bool = False

    @property
    def source(self):
        if not self.l1_pages:
            return "l2"
        return "l1+l2" if self.l2_pages else "l1"


class PeerL1PlusL2Linker(UnifiedCacheLinker):
    def __init__(
        self,
        l1,
        l2,
        *,
        l2_write="write_through",
        max_inflight_write_pages=None,
        write_timeout_s=60.0,
        stitch_l2=True,
    ):
        if l2_write not in ("write_through", "last_replica"):
            raise ValueError(f"Unknown peer_l1 l2_write policy: {l2_write!r}")
        self.l1, self.l2 = l1, l2
        self.page_size = l2.page_size
        # last_replica: L2 receives a page only when its last HBM copy on the node
        # is evicted (the wrapper writes through only for a persisting linker).
        self.persists = l2_write == "write_through"
        # Last-replica writes leave a page out of L2 while any HBM holds it, so a
        # prefix whose head is still in a peer has its evicted tail only in L2.
        self.stitch = stitch_l2 and not self.persists
        # In-flight last-replica writes, oldest first: (rows, submit time). Their
        # rows stay allocated until the write lands, so both bounds cap held HBM.
        self._writes = deque()
        self._inflight_write_pages = 0
        self._max_inflight_write_pages = max_inflight_write_pages
        self._write_timeout_s = write_timeout_s
        self._writes_stalled = False
        self.layer_done_counter = CompositeLayerCounter(l1, l2)
        self._lookups = {}
        self._requests = {}
        self._queued = {}
        self._batches = {}

    @classmethod
    def from_config(cls, server_args, params, *, components):
        from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import (
            UMBPDirectLinker,
            _parse_storage_extra_config,
        )
        from sglang.srt.runtime_context import get_memory

        from .linker import PeerL1Linker

        config = _parse_storage_extra_config(
            get_memory().hicache_storage_backend_extra_config
        )
        l1_config = config.pop("peer_l1", {})
        l1 = PeerL1Linker(
            server_args, params, components=components, extra_config=l1_config
        )
        backend = l1_config.get("l2_backend", "umbp")
        if backend not in ("umbp", "decode_lend", "decode_lend+umbp"):
            raise ValueError(f"Unknown peer_l1 l2_backend: {backend!r}")
        if backend != "umbp" and not getattr(l1.pool_group, "rank_replicated", False):
            raise ValueError("decode lend requires rank-replicated unified KV")
        l2 = (
            UMBPDirectLinker(
                server_args, params, components=components, extra_config=config
            )
            if backend != "decode_lend"
            else None
        )
        if backend != "umbp":
            from .lend.tiered import TieredL2

            l2 = TieredL2(l1, l2)
        return cls(
            l1,
            l2,
            l2_write=l1_config.get("l2_write", "write_through"),
            # Arbitrary: 15% of the page rows, above the ~9% per-rank hold peak
            # measured with last_replica at a 0.75 TB tier.
            max_inflight_write_pages=int(
                l1_config.get(
                    "l2_write_max_inflight_pages", l1.directory.num_rows * 15 // 100
                )
            ),
            write_timeout_s=float(l1_config.get("l2_write_timeout_s", 60.0)),
            stitch_l2=bool(l1_config.get("l2_stitch", True)),
        )

    def bind_pd(self, manager):
        if hasattr(self.l2, "bind_pd"):
            self.l2.bind_pd(manager)

    def unbind_pd(self):
        if hasattr(self.l2, "unbind_pd"):
            self.l2.unbind_pd()

    def bind_cache(self, cache):
        self.l1.bind_cache(cache)
        self.l2.bind_cache(cache)
        if not self.persists:
            self.l1.owner.last_replica_writer = self._write_last_replicas

    def _write_last_replicas(self, keys, rows, *, parents=None, eligible=True):
        # A refused page is freed without an L2 copy: L2 coverage, not HBM, pays.
        if self._writes_stalled:
            self.l1.stats["last_replica_stalled_pages"] += len(rows)
            return False
        if (
            self._max_inflight_write_pages is not None
            and self._inflight_write_pages + len(rows) > self._max_inflight_write_pages
        ):
            self.l1.stats["last_replica_capped_pages"] += len(rows)
            return False
        page = self.page_size
        # CPU indices: the L2 offload snapshots them without a device sync.
        indices = torch.from_numpy((rows[:, None] * page + np.arange(page)).reshape(-1))
        transfer = PoolTransfer(
            name=PoolName.KV,
            device_indices=indices,
            keys=list(keys),
            parent_keys=parents,
            indexer_eligible=eligible,
        )
        if not self.l2.offload([transfer]):
            return False
        self._writes.append((rows, time.monotonic()))
        self._inflight_write_pages += len(rows)
        return True

    def _check_write_timeout(self):
        # A late write keeps its rows: the L2 copy may still be reading them.
        # New writes are refused until it lands, so the held rows stop growing.
        stalled = (
            bool(self._writes)
            and time.monotonic() - self._writes[0][1] > self._write_timeout_s
        )
        if stalled and not self._writes_stalled:
            self.l1.stats["last_replica_write_timeouts"] += 1
            logger.warning(
                "peer_l1: an L2 write has been in flight for over %.0f s; refusing "
                "new last-replica writes until it lands (%d pages in flight)",
                self._write_timeout_s,
                self._inflight_write_pages,
            )
        elif self._writes_stalled and not stalled:
            logger.info("peer_l1: L2 writes landed again; last-replica writes resume")
        self._writes_stalled = stalled

    def lookup(self, rid, transfers):
        self._lookups.pop(rid, None)
        if rid in self._requests:
            return []
        kv = next(t for t in transfers if t.name == PoolName.KV)
        keys = kv.keys
        # Both sources restore contiguous prefixes in the supported unified KV
        # layout: pages [0, a) from one peer, [start, b) from L2.
        l1_hits = self.l1.lookup(rid, transfers)
        a = l1_hits[-1] if l1_hits else 0
        if a == len(keys):
            self.l1.stats["lookup_l1_covers_tail"] += 1
        if self.stitch and a:
            # L2 continues the peer prefix. It also covers any longer L2 prefix:
            # L2 holding [0, n) holds [a, n).
            start = a
            l2_hits = (
                self.l2.lookup(rid, [replace(kv, keys=keys[a:])])
                if a < len(keys)
                else []
            )
            b = a + (l2_hits[-1] if l2_hits else 0)
            if b > a:
                self.l1.stats["stitch_lookups"] += 1
                self.l1.stats["stitch_lookup_l2_pages"] += b - a
        else:
            start = 0
            l2_hits = self.l2.lookup(rid, transfers)
            b = l2_hits[-1] if l2_hits else 0
        self._lookups[rid] = (a, b, start)
        return list(range(1, max(a, b) + 1))

    def reserve(self, rid, num_pages):
        hit = self._lookups.pop(rid, None)
        if rid in self._requests or hit is None:
            return False
        a, b, start = hit
        if num_pages <= a and self.l1.reserve(rid, num_pages):
            l1_pages = num_pages
        elif start and num_pages <= b and self.l1.reserve(rid, start):
            l1_pages = start
        else:
            self.l1.release_reservation(rid)
            # A continuation is useless without its peer prefix.
            if start or num_pages > b:
                self.l1.stats["reserve_fallback_to_recompute"] += 1
                self.l1.stats["reserve_recompute_tokens"] += num_pages * self.page_size
                return False
            l1_pages = 0
            if num_pages <= a:
                self.l1.stats["l1_reserve_fallback_to_l2"] += 1
        self._requests[rid] = _Restore(l1_pages, num_pages - l1_pages)
        return True

    def release_reservation(self, rid):
        self._lookups.pop(rid, None)
        self.l1.release_reservation(rid)
        self.l2.release_reservation(rid)
        state = self._requests.get(rid)
        if state is not None and not state.queued:
            del self._requests[rid]

    def _pages(self, transfer, start, end):
        tokens = slice(start * self.page_size, end * self.page_size)
        return replace(
            transfer,
            keys=transfer.keys[start:end],
            device_indices=transfer.device_indices[tokens],
        )

    def load(self, rid, transfers):
        state = self._requests[rid]
        if len(transfers) != 1 or transfers[0].name != PoolName.KV:
            raise ValueError("mori_peer_l1 supports unified KV transfers only")
        if not state.l2_pages:
            parts = [(self.l1, transfers)]
        elif not state.l1_pages:
            parts = [(self.l2, transfers)]
        else:
            (kv,) = transfers
            split, end = state.l1_pages, len(kv.keys)
            parts = [
                (self.l1, [self._pages(kv, 0, split)]),
                (self.l2, [self._pages(kv, split, end)]),
            ]
        for index, (backend, part) in enumerate(parts):
            if not backend.load(rid, part):
                if index:  # The peer prefix is already queued.
                    self.l1.cancel_queued_load(rid)
                self.release_reservation(rid)
                return False
        state.queued = True
        self._queued[rid] = state
        return True

    def start_layer_wise_loading(self):
        # Submit peer reads before UMBP's CPU index snapshot can block this rank.
        i1 = self.l1.start_layer_wise_loading()
        i2 = self.l2.start_layer_wise_loading()
        if i1 < 0 and i2 < 0:
            return -1
        index = self.layer_done_counter.register(i1, i2)
        self._batches[index] = list(self._queued)
        self._queued.clear()
        if i1 >= 0 and i2 >= 0:
            self.l1.stats["mixed_load_batches"] += 1
        return index

    def finish_layer_wise_loading(self, counter_index):
        landed = self.layer_done_counter.finish(counter_index)
        for rid in self._batches.pop(counter_index, ()):
            state = self._requests.pop(rid)
            if landed:
                self.l1.stats["l1_loaded_pages"] += state.l1_pages
                self.l1.stats["l2_loaded_pages"] += state.l2_pages
                if state.l1_pages and state.l2_pages:
                    self.l1.stats["stitched_loads"] += 1
        if not landed:
            self.l1.stats["failed_composite_load_batches"] += 1
        return landed

    def cancel_queued_load(self, rid):
        state = self._queued.get(rid)
        if state is None:
            return False
        # Both parts are queued and submitted together; L2's cancel has no side
        # effect to undo if L1's then fails.
        if state.l2_pages and not self.l2.cancel_queued_load(rid):
            return False
        if state.l1_pages and not self.l1.cancel_queued_load(rid):
            raise RuntimeError(f"peer_l1: only the L2 part of {rid} was still queued")
        del self._queued[rid]
        del self._requests[rid]
        return True

    def offload(self, transfers):
        return self.l2.offload(transfers)

    def num_completed_offloads(self):
        if self.persists:
            return self.l2.num_completed_offloads()
        # The wrapper offloads nothing; every completion is a last-replica write.
        for _ in range(self.l2.num_completed_offloads()):
            landed = self.l2.pop_completed_offload()
            rows, _ = self._writes.popleft()
            self._inflight_write_pages -= len(rows)
            self.l1.owner.release_written(rows)
            if not landed:
                self.l1.stats["last_replica_failed_batches"] += 1
        self._check_write_timeout()
        return 0

    def pop_completed_offload(self):
        return self.l2.pop_completed_offload()

    def _clear(self):
        self._lookups.clear()
        self._requests.clear()
        self._queued.clear()
        self._batches.clear()
        self.layer_done_counter.reset()

    def reset(self):
        self.l1.reset()
        self.l2.reset()
        self._clear()

    def close(self):
        self.l2.close()
        self.l1.close()
        self._clear()
