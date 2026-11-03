"""Decode-owned lent pages. No GPU or allocator calls occur under this lock."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from itertools import chain, islice

logger = logging.getLogger(__name__)


@dataclass
class Page:
    generation: int
    state: str = "free"
    key: str = ""
    writer: str = ""
    alloc_seq: int = 0
    pins: set[int] = field(default_factory=set)
    demote_deadline: float = 0.0


class LendPool:
    def __init__(
        self,
        capacity,
        *,
        idle=0.05,
        limit=0.02,
        growth=0.10,
        watermark=0.02,
        lease_s=30.0,
        clock=time.monotonic,
    ):
        if capacity < 1 or not (
            0 < limit < 1
            and 0 <= idle < 1
            and 0 <= growth < 1
            and 0 < watermark < 1
            and lease_s > 0
        ):
            raise ValueError("Invalid decode lend limits")
        self.capacity, self.idle, self.growth = capacity, idle, growth
        self.limit = max(1, int(capacity * limit))
        self.write_limit = max(1, self.limit // 2)
        self.pin_limit = max(1, self.limit - self.write_limit)
        self.watermark, self.lease_s, self.clock = watermark, lease_s, clock
        self.epoch = uuid.uuid4().hex
        self.lock = threading.RLock()
        self.pages: dict[int, Page] = {}
        self.generations = Counter()
        self.free = OrderedDict()
        self.lru = OrderedDict()
        self.keys = {}
        self.leases = {}
        self._next_lease = 0
        self.events = deque()
        self.stats = Counter()
        self.decode_used = 0
        self.draining = False
        self.memory_owner = None
        self.poisoned = 0
        self.writing = 0
        self.pinned = 0
        self.demoting = OrderedDict()
        self.publisher = None
        self._acquires = {}
        self._lease_requests = {}
        self._completed = Counter()
        self._revoked = []

    def _unavailable(self):
        return self.writing + self.pinned + self.poisoned

    @property
    def reclaimable(self):
        with self.lock:
            return len(self.pages) - self._unavailable()

    @property
    def assured_reclaimable(self):
        # Service threads may still take reclaimable pages for writes or pins
        # up to the limit before the scheduler allocates; budgets exclude them.
        with self.lock:
            return max(0, len(self.pages) - max(self.limit, self._unavailable()))

    def _admit(self, extra):
        allowed = (
            not self.draining
            and self._unavailable() + extra <= self.limit
            and self.decode_used
            + self._unavailable()
            + extra
            + int(self.capacity * self.growth)
            <= self.capacity
        )
        if not allowed:
            self.stats["gate_refused"] += extra
        return allowed

    def add_free(self, pages):
        with self.lock:
            for page in pages:
                page = int(page)
                if not 0 < page <= self.capacity or page in self.pages:
                    raise ValueError("Lent page is invalid or already owned")
                self.pages[page] = Page(self.generations[page])
                self.free[page] = None

    def _record(self, page):
        p = self.pages[page]
        return [page, p.generation, p.key]

    def _matches(self, page, gen):
        return page in self.pages and self.pages[page].generation == gen

    def allocate(self, writer, count, seq=0):
        with self.lock:
            count = max(0, min(count, len(self.free), self.write_limit - self.writing))
            if not count or not self._admit(count):
                return []
            result = []
            for _ in range(count):
                page, _ = self.free.popitem(last=False)
                self.generations[page] += 1
                self.pages[page] = Page(
                    self.generations[page], "writing", writer=writer, alloc_seq=seq
                )
                result.append(self._record(page))
                self.writing += 1
            return result

    def cancel_allocation(self, writer, seq):
        """The caller timed out before receiving this grant, so posted no writes."""
        with self.lock:
            for page, p in list(self.pages.items()):
                if p.state == "writing" and p.writer == writer and p.alloc_seq == seq:
                    self._discard(page)

    def poison_allocation(self, writer, seq):
        with self.lock:
            before = self.poisoned
            for p in self.pages.values():
                if p.writer == writer and p.alloc_seq == seq and p.state == "writing":
                    p.state = "poisoned"
                    self.writing -= 1
                    self.poisoned += 1
                    self.stats["quarantined_pages"] += 1
            if self.poisoned != before:
                if not before:
                    from sglang.srt.runtime_context import get_resources

                    get_resources().rdma_quarantines.append(self)
                logger.error(
                    "Decode lend allocation quarantined: pages=%d total=%d; restart required",
                    self.poisoned - before,
                    self.poisoned,
                )

    def commit(self, writer, records):
        with self.lock:
            result, published = [], []
            for page, gen, key, parent in records:
                if not self._matches(page, gen):
                    result.append(False)
                    continue
                p = self.pages[page]
                # A lost reply may be retried without republishing another key.
                if (
                    p.writer == writer
                    and p.key == key
                    and p.state in ("valid", "demoting")
                ):
                    result.append(True)
                elif p.state != "writing" or p.writer != writer:
                    result.append(False)
                elif key in self.keys:
                    self._discard(page)
                    result.append(False)
                else:
                    self.writing -= 1
                    p.state, p.key = "valid", key
                    self.keys[key] = page
                    self.lru[page] = None
                    if self.publisher:
                        published.append((key, parent))
                    result.append(True)
            if published:
                self.publisher.publish(published)
            return result

    def abort(self, writer, records):
        with self.lock:
            for page, gen, *_ in records:
                if self._matches(page, gen):
                    p = self.pages[page]
                    if p.state == "writing" and p.writer == writer:
                        self._discard(page)

    def acquire(self, reader, seq, keys):
        with self.lock:
            self.expire()
            request = (reader, seq)
            if request in self._acquires:
                return self._acquires[request]
            if seq <= self._completed[reader]:
                return [0, []]
            records = []
            for key in keys:
                page = self.keys.get(key)
                if page is None or self.pages[page].state not in ("valid", "demoting"):
                    break
                records.append(self._record(page))
            lease = self.pin(reader, records)
            if not lease:
                self._completed[reader] = max(seq, self._completed[reader])
                return [0, []]
            result = [lease, records]
            self._acquires[request] = result
            self._lease_requests[lease] = request
            return result

    def cancel_acquire(self, reader, seq):
        with self.lock:
            saved = self._acquires.get((reader, seq))
            if saved:
                self.unpin(reader, saved[0])
            self._completed[reader] = max(seq, self._completed[reader])

    def pin(self, reader, records):
        with self.lock:
            self.expire()
            selected = []
            for page, gen, key in records:
                if not self._matches(page, gen):
                    return 0
                p = self.pages[page]
                if p.state not in ("valid", "demoting") or p.key != key:
                    return 0
                selected.append(page)
            selected = list(dict.fromkeys(selected))
            extra = sum(not self.pages[p].pins for p in selected)
            if (
                not selected
                or self.pinned + extra > self.pin_limit
                or not self._admit(extra)
            ):
                return 0
            self.pinned += extra
            self._next_lease += 1
            lease = self._next_lease
            self.leases[lease] = (reader, selected, self.clock() + self.lease_s)
            for page in selected:
                self.pages[page].pins.add(lease)
                self.lru.move_to_end(page)
            return lease

    def unpin(self, reader, lease, *, report=True):
        with self.lock:
            saved = self.leases.get(lease)
            if saved is None or saved[0] != reader:
                return
            del self.leases[lease]
            request = self._lease_requests.pop(lease, None)
            if request is not None:
                self._acquires.pop(request)
                self._completed[reader] = max(request[1], self._completed[reader])
            for page in saved[1]:
                p = self.pages.get(page)
                if p is None:
                    continue
                p.pins.remove(lease)
                if not p.pins:
                    self.pinned -= 1
                if p.state == "retired" and not p.pins:
                    self._discard(page)
            if report:
                self._report_revokes()

    def _report_revokes(self):
        if self._revoked:
            self.publisher.retire(self._revoked)
            self._revoked.clear()

    def _discard(self, page):
        p = self.pages[page]
        assert not p.pins and p.state != "poisoned"
        if p.state == "writing":
            self.writing -= 1
        self.demoting.pop(page, None)
        if p.key:
            if self.publisher:
                self._revoked.append(p.key)
            if self.keys.get(p.key) == page:
                del self.keys[p.key]
        self.lru.pop(page, None)
        self.pages[page] = Page(p.generation)
        self.free[page] = None

    def demoted(self, writer, records):
        with self.lock:
            for page, gen, key in records:
                if self._matches(page, gen):
                    p = self.pages[page]
                    if p.writer == writer and p.key == key and p.state == "demoting":
                        self.demoting.pop(page, None)
                        if p.pins:
                            p.state = "retired"
                            if self.publisher:
                                self._revoked.append(p.key)
                            self.lru.pop(page, None)
                        else:
                            self._discard(page)
            self._report_revokes()

    def expire(self):
        with self.lock:
            now = self.clock()
            for lease, (reader, _, deadline) in list(self.leases.items()):
                if now >= deadline:
                    self.unpin(reader, lease, report=False)
            for page, deadline in list(self.demoting.items()):
                if now < deadline:
                    break
                self.demoting.pop(page)
                p = self.pages[page]
                if p.state == "demoting":
                    if p.pins:
                        p.state = "retired"
                        if self.publisher:
                            self._revoked.append(p.key)
                        self.lru.pop(page, None)
                    else:
                        self._discard(page)
            self._report_revokes()

    def request_demotions(self, count):
        with self.lock:
            if count <= 0:
                return
            records = {}
            for page in self.lru:
                if count <= 0:
                    break
                p = self.pages[page]
                if p.state != "valid":
                    continue
                p.state, p.demote_deadline = "demoting", self.clock() + self.lease_s
                self.demoting[page] = p.demote_deadline
                records.setdefault(p.writer, []).append(self._record(page))
                count -= 1
            for writer, pages in records.items():
                self.events.append(("DEMOTE", writer, pages))

    def reclaim(self, count):
        """Called only by the scheduler; service threads never return pages."""
        with self.lock:
            self.expire()
            candidates = list(
                islice(
                    chain(self.free, (p for p in self.lru if not self.pages[p].pins)),
                    max(0, count),
                )
            )
            result = []
            for page in candidates:
                p = self.pages[page]
                if p.state != "free":
                    self._discard(page)
                    self.stats["sync_discard"] += 1
                self.free.pop(page)
                del self.pages[page]
                result.append(page)
            self._report_revokes()
            return result
