"""Choose one source per request: peer HBM when it covers the hit, else UMBP."""

from __future__ import annotations

from dataclasses import dataclass

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinker


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
    source: str
    pages: int
    queued: bool = False


class PeerL1PlusL2Linker(UnifiedCacheLinker):
    def __init__(self, l1, l2):
        self.l1, self.l2 = l1, l2
        self.page_size = l2.page_size
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
        # Both sources restore contiguous prefixes in the supported unified KV
        # layout. A request uses exactly one source for the selected boundary.
        l1_hits = self.l1.lookup(rid, transfers)
        l2_hits = self.l2.lookup(rid, transfers)
        a = l1_hits[-1] if l1_hits else 0
        b = l2_hits[-1] if l2_hits else 0
        self._lookups[rid] = (a, b)
        if a == len(keys):
            self.l1.stats["lookup_l1_covers_tail"] += 1
        return list(range(1, max(a, b) + 1))

    def reserve(self, rid, num_pages):
        hit = self._lookups.pop(rid, None)
        if rid in self._requests or hit is None:
            return False
        a, b = hit
        if num_pages <= a and self.l1.reserve(rid, num_pages):
            source = "l1"
        else:
            self.l1.release_reservation(rid)
            if num_pages > b:
                self.l1.stats["reserve_fallback_to_recompute"] += 1
                return False
            source = "l2"
            if num_pages <= a:
                self.l1.stats["l1_reserve_fallback_to_l2"] += 1
        self._requests[rid] = _Restore(source, num_pages)
        return True

    def release_reservation(self, rid):
        self._lookups.pop(rid, None)
        self.l1.release_reservation(rid)
        state = self._requests.get(rid)
        if state is not None and not state.queued:
            del self._requests[rid]

    def load(self, rid, transfers):
        state = self._requests[rid]
        if len(transfers) != 1 or transfers[0].name != PoolName.KV:
            raise ValueError("mori_peer_l1 supports unified KV transfers only")
        backend = self.l1 if state.source == "l1" else self.l2
        if not backend.load(rid, transfers):
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
                self.l1.stats[f"{state.source}_loaded_pages"] += state.pages
        if not landed:
            self.l1.stats["failed_composite_load_batches"] += 1
        return landed

    def cancel_queued_load(self, rid):
        state = self._queued.get(rid)
        if state is None:
            return False
        backend = self.l1 if state.source == "l1" else self.l2
        if not backend.cancel_queued_load(rid):
            return False
        del self._queued[rid]
        del self._requests[rid]
        return True

    def offload(self, transfers):
        return self.l2.offload(transfers)

    def num_completed_offloads(self):
        return self.l2.num_completed_offloads()

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
