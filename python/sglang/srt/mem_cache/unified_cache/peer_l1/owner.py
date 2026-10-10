"""Publish completed tree pages and delay physical reuse for remote readers."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class Publication:
    event: object
    keys: list[str]
    versions: list[int]
    rows: torch.Tensor
    keepalive: list


class L1Owner:
    def __init__(self, cache, linker):
        self.cache, self.linker = cache, linker
        self.directory = linker.directory
        self.stream = torch.cuda.Stream(device=linker.device)
        self._version = 0
        self._pending = {}
        self._attaches = []
        self._publications = deque()
        self._published = set()
        # Retired rows awaiting their free action, then rows withheld from it.
        self._hold = set()
        self._zombies = set()
        self.withheld_total = 0
        cache.tree_core.enable_l1_events = True

    @property
    def deferred_tokens(self):
        return len(self._zombies) * self.cache.page_size

    @property
    def pinned_live_tokens(self):
        refs = self.directory.pinned_refs
        if self._zombies:
            refs -= int(self.directory.states(list(self._zombies))[0].sum())
        return max(0, refs) * self.cache.page_size

    def consume_events(self):
        for keys, value in self.cache.tree_core.take_l1_events():
            if value is None:
                for key in keys:
                    self._pending.pop(key, None)
                    self._published.discard(key)
                rows = self.directory.retire(keys)
                if len(rows):
                    pins, _ = self.directory.states(rows)
                    self._hold.update(int(row) for row in rows[pins > 0])
            else:
                self._version += 1
                versions = []
                for key in keys:
                    if self.linker.loading_keys[key]:
                        versions.append(-1)
                    else:
                        self._pending[key] = self._version
                        versions.append(self._version)
                if any(version >= 0 for version in versions):
                    self._attaches.append((keys, versions, value))

    def filter_free(self, indices):
        """Run before every FULL allocator free; normal frees do no D2H."""
        if not self._hold and not self.linker.check_hooks:
            return indices
        page = self.cache.page_size
        cpu = indices.reshape(-1, page).to("cpu")
        rows = cpu[:, 0].numpy() // page
        if self.linker.check_hooks:
            _, live = self.directory.states(rows)
            if np.any(live):
                raise AssertionError(
                    "freeing a published peer L1 page: missing retire hook"
                )
        keep = np.array([int(row) not in self._hold for row in rows], dtype=bool)
        withheld = set(map(int, rows[~keep]))
        if not withheld:
            return indices
        self._hold.difference_update(withheld)
        self._zombies.update(withheld)
        self.withheld_total += len(withheld) * page
        self.linker.stats["withheld_pages"] += len(withheld)
        self.linker.stats["max_deferred_pages"] = max(
            self.linker.stats["max_deferred_pages"], len(self._zombies)
        )
        # This is the explicitly rare, already-synchronized eviction path.
        return cpu[torch.from_numpy(keep)].flatten().to(indices.device)

    def reclaim(self):
        if not self._zombies:
            return 0
        rows = np.fromiter(self._zombies, dtype=np.int64)
        pins, _ = self.directory.states(rows)
        ready = rows[pins == 0]
        if not len(ready):
            return 0
        page = self.cache.page_size
        cpu = torch.from_numpy(ready * page)
        starts = cpu.to(self.linker.device, non_blocking=True)
        indices = (
            starts[:, None] + torch.arange(page, device=self.linker.device)
        ).flatten()
        allocator = self.cache.token_to_kv_pool_allocator
        full = getattr(allocator, "full_attn_allocator", allocator)
        full.free_segment(indices, start_pos=0)
        self._zombies.difference_update(map(int, ready))
        return len(ready) * page

    def wait_for_capacity(self, target, available):
        self.reclaim()
        if (
            self.linker.has_unstarted_reservations
            or not self._zombies
            or available() + self.deferred_tokens < target
        ):
            return
        self.linker.stats["capacity_waits"] += 1
        start = time.monotonic()
        deadline = start + self.linker.wait_seconds
        while available() < target and self._zombies and time.monotonic() < deadline:
            self.reclaim()
            if available() < target:
                time.sleep(0.00005)
        self.linker.stats["capacity_wait_seconds"] += time.monotonic() - start

    def poll(self, forward_stream):
        self.consume_events()
        self.reclaim()
        while self._publications and self._publications[0].event.query():
            publication = self._publications.popleft()
            selected = [
                i
                for i, (key, version) in enumerate(
                    zip(publication.keys, publication.versions)
                )
                if version >= 0 and self._pending.get(key) == version
            ]
            keys = [publication.keys[i] for i in selected]
            rows = publication.rows.numpy()[selected]
            if keys:
                self.publish_loaded(keys, rows)
        if not self._attaches:
            return
        schedule_stream = torch.cuda.current_stream()
        if self.stream.cuda_stream in (
            schedule_stream.cuda_stream,
            forward_stream.cuda_stream,
        ):
            from sglang.srt.managers.utils import allocate_distinct_stream

            self.stream = allocate_distinct_stream(
                torch.cuda, (schedule_stream, forward_stream)
            )
        attaches, self._attaches = self._attaches, []
        keys = [key for item in attaches for key in item[0]]
        versions = [version for item in attaches for version in item[1]]
        rows_cpu = torch.empty(
            len(keys), dtype=torch.int64, device="cpu", pin_memory=True
        )
        event = torch.cuda.Event()
        self.stream.wait_stream(schedule_stream)
        self.stream.wait_stream(forward_stream)
        with torch.cuda.stream(self.stream):
            rows = (
                torch.cat(
                    [
                        item[2].reshape(-1, self.cache.page_size)[:, 0]
                        for item in attaches
                    ]
                )
                // self.cache.page_size
            )
            rows_cpu.copy_(rows, non_blocking=True)
            event.record(self.stream)
        self._publications.append(
            Publication(event, keys, versions, rows_cpu, [attaches, rows])
        )

    def publish_loaded(self, keys, rows):
        self.directory.publish(keys, rows)
        self._published.update(keys)
        for key in keys:
            self._pending.pop(key, None)

    def begin_flush(self):
        self.consume_events()
        self.reclaim()
        if self.linker.has_loads:
            logger.warning("peer_l1 flush busy: local loads have not completed")
            return False
        if self._hold or self._zombies:
            logger.warning(
                "peer_l1 flush busy: %d retired pages awaiting free, %d deferred pages",
                len(self._hold),
                len(self._zombies),
            )
            return False
        if not self.directory.freeze():
            logger.warning(
                "peer_l1 flush busy: directory is pinned or frozen (remote refs=%d)",
                self.directory.pinned_refs,
            )
            return False
        return True

    def reset(self):
        # Called only after begin_flush, or after the whole worker quiesces.
        if self._published:
            self.directory.retire(list(self._published))
        self._published.clear()
        self._pending.clear()
        self._attaches.clear()
        # The gather may still reference value tensors and pinned CPU buffers.
        # Reset is cold; finish it before releasing those allocations.
        self.stream.synchronize()
        self._publications.clear()
