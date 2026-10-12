"""Choose one source per request: peer HBM when it covers the hit, else UMBP."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

import numpy as np
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinker


class CompositeLayerCounter:
    def __init__(self, l1, l2):
        if l1.num_layers != l2.num_layers:
            raise ValueError("L1 and L2 must cover the same logical layers")
        self.l1, self.l2 = l1, l2
        self.num_layers = l1.num_layers
        self.reset()

    def register(self, l1_index, l2_index):
        index = self._next_index
        self._next_index += 1
        self._indices[index] = (l1_index, l2_index)
        return index

    def set_consumer(self, index):
        # A batch can keep its old index when it becomes a decode batch. A
        # retired mapping means both sources are already readable.
        if index >= self._next_index:
            raise ValueError("Unknown composite load index")
        self.consumer_index = index
        a, b = self._indices.get(index, (-1, -1))
        self.l1.set_consumer(a)
        self.l2.set_consumer(b)

    def wait_until(self, layer):
        self.l1.wait_until(layer)
        self.l2.wait_until(layer)
        if layer == self.num_layers - 1:
            self.retire(self.consumer_index)

    def retire(self, index):
        self._indices.pop(index, None)

    def reset(self):
        self._indices = {}
        self._next_index = 0
        self.consumer_index = -1


@dataclass
class _Restore:
    source: str
    stored: bool
    keys: list[str]
    queued: bool = False
    transfer: PoolTransfer | None = None
    rows: np.ndarray | None = None


class PeerL1PlusL2Linker(UnifiedCacheLinker):
    persists = True

    def __init__(self, l1, l2):
        self.l1, self.l2 = l1, l2
        self.page_size = l2.page_size
        self.layer_done_counter = CompositeLayerCounter(
            l1.layer_done_counter, l2.layer_done_counter
        )
        self._lookups = {}
        self._requests = {}
        self._queued = {}
        self._batches = deque()

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
        l2 = UMBPDirectLinker(
            server_args, params, components=components, extra_config=config
        )
        return cls(l1, l2)

    def bind_cache(self, cache):
        self.l1.bind_cache(cache)
        self.l2.bind_cache(cache)

    def lookup(self, rid, transfers):
        self._lookups.pop(rid, None)
        if rid in self._requests:
            return []
        keys = next(t.keys for t in transfers if t.name == PoolName.KV)
        # Both backends return contiguous prefixes for the supported unified
        # KV layout. Keep lengths rather than another set of page indices.
        a = len(self.l1.lookup(rid, transfers))
        b = len(self.l2.lookup(rid, transfers))
        self._lookups[rid] = (keys, a, b)
        if a == len(keys):
            self.l1.stats["lookup_l1_covers_tail"] += 1
        return list(range(1, max(a, b) + 1))

    def reserve(self, rid, num_pages):
        hit = self._lookups.pop(rid, None)
        if rid in self._requests or hit is None:
            return False
        keys, a, b = hit
        if num_pages <= a and self.l1.reserve(rid, num_pages):
            source = "l1"
        else:
            self.l1.release_reservation(rid)  # Includes an unused lookup hint.
            if num_pages > b:
                self.l1.stats["reserve_fallback_to_recompute"] += 1
                return False
            source = "l2"
            if num_pages <= a:
                self.l1.stats["l1_reserve_fallback_to_l2"] += 1
            self.l1.loading_keys.update(keys[:num_pages])
        stored = num_pages <= b
        self._requests[rid] = _Restore(source, stored, list(keys[:num_pages]))
        return True

    def marks_loaded_as_stored(self, rid):
        state = self._requests[rid]
        metric = "stored_marked_pages" if state.stored else "l1_unmarked_pages"
        self.l1.stats[metric] += len(state.keys)
        return state.stored

    def _forget(self, rid):
        state = self._requests.pop(rid)
        if state.source == "l2":
            for key in state.keys:
                self.l1.loading_keys[key] -= 1
                if not self.l1.loading_keys[key]:
                    del self.l1.loading_keys[key]
        return state

    def release_reservation(self, rid):
        self._lookups.pop(rid, None)
        self.l1.release_reservation(rid)
        state = self._requests.get(rid)
        if state is not None and not state.queued:
            self._forget(rid)

    def load(self, rid, transfers):
        state = self._requests[rid]
        if len(transfers) != 1 or transfers[0].name != PoolName.KV:
            raise ValueError("mori_peer_l1 supports unified KV transfers only")
        if state.source == "l1" and not self.l1.load(rid, transfers):
            return False
        state.transfer = transfers[0]
        state.queued = True
        self._queued[rid] = state
        return True

    def start_layer_wise_loading(self):
        # Submit every peer read before the UMBP plan's existing CPU index
        # snapshot can block this scheduler. This preserves the pin wait rule.
        i1 = self.l1.start_layer_wise_loading()
        for rid, state in self._queued.items():
            if state.source == "l2":
                transfer = state.transfer
                indices = transfer.device_indices.to(device="cpu")
                state.rows = indices[:: self.page_size].numpy() // self.page_size
                # Reuse this snapshot for UMBP's pointer planning, avoiding a
                # second D2H and any GPU synchronization at completion time.
                if not self.l2.load(rid, [replace(transfer, device_indices=indices)]):
                    raise RuntimeError(f"Failed to queue UMBP load for {rid!r}")
        i2 = self.l2.start_layer_wise_loading()
        self._queued.clear()
        if i1 < 0 and i2 < 0:
            return -1
        index = self.layer_done_counter.register(i1, i2)
        self._batches.append((index, i1 >= 0, i2 >= 0))
        if i1 >= 0 and i2 >= 0:
            self.l1.stats["mixed_load_batches"] += 1
        return index

    def num_completed_loads(self):
        a, b = self.l1.num_completed_loads(), self.l2.num_completed_loads()
        count = 0
        for _, has_l1, has_l2 in self._batches:
            if a < has_l1 or b < has_l2:
                break
            a -= has_l1
            b -= has_l2
            count += 1
        return count

    def pop_completed_load(self):
        index, has_l1, has_l2 = self._batches.popleft()
        # UMBP pop raises on a failed batch before any of its rows are published.
        l2_rids = self.l2.pop_completed_load() if has_l2 else []
        l1_rids = self.l1.pop_completed_load() if has_l1 else []
        if l2_rids:
            self.l1.owner.consume_events()  # All loading_keys are still held.
            for rid in l2_rids:
                state = self._requests[rid]
                self.l1.owner.publish_loaded(state.transfer.keys, state.rows)
        for rid in l1_rids + l2_rids:
            state = self._forget(rid)
            pages = len(state.transfer.keys)
            self.l1.stats[f"{state.source}_loaded_pages"] += pages
        self.layer_done_counter.retire(index)
        return l1_rids + l2_rids

    def cancel_queued_load(self, rid):
        return False

    def offload(self, transfers):
        return self.l2.offload(transfers)

    def num_completed_offloads(self):
        return self.l2.num_completed_offloads()

    def pop_completed_offload(self):
        return self.l2.pop_completed_offload()

    def _clear(self):
        for rid in list(self._requests):
            self._forget(rid)
        self._lookups.clear()
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
