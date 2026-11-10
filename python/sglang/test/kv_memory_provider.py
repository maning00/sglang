"""CPU rank-local engine substitute for provider integration tests.

This models ownership, not copying: publication's cancel/send point is locked,
Retire has operation identity, I/O survives row reuse, and reset stays sealed
until commit. Production tests still execute the real allocators/tree/storage.
"""

from collections import Counter
from dataclasses import dataclass, replace
from threading import RLock

import torch

from sglang.srt.mem_cache.kv_memory_provider import (
    Busy,
    CompletedReceipt,
    JoinedReceipt,
    PageAccounting,
    PageHandle,
    PoolEpoch,
    ProviderFailStop,
    Short,
)


class ManualReceipt:
    def __init__(self, ready=False):
        self.done = ready

    def query(self):
        return self.done

    def wait_on(self, stream):
        stream.wait_event(self)

    def wait(self):
        self.synchronize()

    def synchronize(self):
        if not self.done:
            raise RuntimeError("test receipt has not completed")


@dataclass(frozen=True)
class FakeResetLease:
    old_epoch: PoolEpoch
    new_epoch: PoolEpoch
    approved_pages: tuple[int, ...]


class FakeReservation:
    def __init__(self, provider, shape, pages):
        self.provider, self.shape = provider, shape
        self.pages = {
            "load": list(pages[: shape.load_pages]),
            "extend": list(pages[shape.load_pages :]),
        }
        self.closed = False

    def remaining(self, purpose):
        return len(self.pages[purpose])

    def covers(self, shape):
        return not self.closed and shape == self.shape

    def take(self, purpose, n):
        with self.provider.lock:
            self.provider.check()
            if self.closed or n > self.remaining(purpose):
                raise RuntimeError("reservation over-consumed")
            pages, self.pages[purpose] = (
                self.pages[purpose][:n],
                self.pages[purpose][n:],
            )
            handles = []
            for p in pages:
                assert self.provider.states[p] == "reserved"
                self.provider.generations[p] += 1
                self.provider.states[p] = "in_use"
                handles.append(self.provider.handle(p))
            self.provider.takes.append((self.shape.request_id, purpose, tuple(handles)))
            return tuple(handles)

    def revise(self, shape):
        with self.provider.lock:
            self.provider.check()
            if (
                self.closed
                or shape.request_id != self.shape.request_id
                or shape.revision <= self.shape.revision
            ):
                return False
            wanted = {"load": shape.load_pages, "extend": shape.extend_pages}
            extra = sum(max(0, wanted[k] - len(self.pages[k])) for k in wanted)
            cached = [
                p for p, state in self.provider.states.items() if state == "cached"
            ]
            if extra > len(cached):
                return False
            for purpose, n in wanted.items():
                kept = self.pages[purpose]
                for page in kept[n:]:
                    self.provider.states[page] = "cached"
                del kept[n:]
                while len(kept) < n:
                    page = cached.pop(0)
                    self.provider.states[page] = "reserved"
                    kept.append(page)
            self.shape = shape
            return True

    def refund_unused(self):
        with self.provider.lock:
            if self.closed:
                return
            for pages in self.pages.values():
                for p in pages:
                    assert self.provider.states[p] == "reserved"
                    self.provider.states[p] = "cached"
                pages.clear()
            self.closed = True


class FakeFrozen:
    def __init__(self, provider, handles):
        self.provider, self.by_page = provider, {h.page_id: h for h in handles}

    def handles(self, indices=None):
        return (
            tuple(self.by_page.values())
            if indices is None
            else tuple(self.by_page[p] for p in self.provider.page_ids(indices))
        )

    def ready(self, handles):
        return JoinedReceipt(
            tuple(self.provider.forwards.get(h, CompletedReceipt()) for h in handles)
        )


class FakeKVMemoryProvider:
    def __init__(self, *, device="cpu", options=None, pages=16, page_size=4):
        options = options or {}
        pages, page_size = (
            options.get("pages", pages),
            options.get("page_size", page_size),
        )
        self.device, self.page_size = device, page_size
        self.lock = RLock()
        self.epoch = PoolEpoch(b"fake-node-boot", 0)
        self.states = {p: "cached" for p in range(1, pages + 1)}
        self.generations = dict.fromkeys(self.states, 0)
        self.pending_publish, self.forwards, self.guards = {}, {}, {}
        self.records, self.io, self.quarantine_resources = {}, {}, []
        self.retires, self.commands, self.takes, self.shapes = {}, [], [], []
        self.quarantine_returned = set()
        self.sent_retires = set()
        self.released_ops = set()
        self.events, self.event_results = [], []
        self.next_op = 1
        self.closed = False
        self.lease = None
        self.auto_release = True
        self.before_send = None
        self.mem_pool_object = object()

    def check(self):
        if self.closed:
            raise ProviderFailStop("fake control channel disconnected")
        if self.lease is not None:
            raise RuntimeError("pool sealed for reset")

    def mem_pool(self, device):
        return self.mem_pool_object

    def create_linker(self, server_args, params, *, components, forward_stream):
        return make_fake_linker(self)

    def register_pool(self, geometry, request_regions):
        geometry.validate()
        if geometry.num_pages != len(self.states):
            raise ValueError("fake geometry capacity mismatch")
        self.geometry, self.request_regions = geometry, tuple(request_regions)
        self.page_size = geometry.page_tokens
        return "fake-pool"

    def handle(self, p):
        return PageHandle("fake-pool", self.epoch, p, self.generations[p])

    def page_ids(self, indices):
        from sglang.srt.mem_cache.kv_memory_provider import allocation_page_ids

        return allocation_page_ids(indices, self.page_size)

    def freeze_allocations(self, indices, *, context=None, page_ids=None):
        with self.lock:
            self.check()
            pages = self.page_ids(indices) if page_ids is None else tuple(page_ids)
            if any(
                p not in self.states
                or self.states[p] not in ("in_use", "publish_sent", "published")
                for p in pages
            ):
                raise RuntimeError("indices do not name live allocations")
            return FakeFrozen(
                self,
                tuple(
                    context.by_page[p]
                    if context is not None and p in context.by_page
                    else self.handle(p)
                    for p in pages
                ),
            )

    def reserve(self, shape):
        with self.lock:
            self.check()
            self.shapes.append(shape)
            cached = [p for p, state in self.states.items() if state == "cached"]
            n = shape.load_pages + shape.extend_pages
            if len(cached) < n:
                return Short(n - len(cached))
            pages = cached[:n]
            for p in pages:
                self.states[p] = "reserved"
            return FakeReservation(self, shape, pages)

    def record_forward(self, handles, done):
        for h in handles:
            assert h == self.handle(h.page_id)
            self.forwards[h] = done

    @property
    def has_pending_io(self):
        with self.lock:
            return bool(self.io)

    def publish(self, pages, *, ready):
        with self.lock:
            self.check()
            for page in pages:
                h = page.handle
                if h != self.handle(h.page_id) or self.states[h.page_id] not in (
                    "in_use",
                    "publish_sent",
                    "published",
                ):
                    raise RuntimeError("publish requires a current live allocation")
                if self.records.get(h) == page:
                    continue
                if h in self.records:
                    raise RuntimeError("published allocation cannot change its key")
                self.pending_publish[h] = (page, ready)

    def held(self, h):
        return any(h in handles for handles, _, _ in self.io.values())

    def safe(self, h, ready):
        return (
            not self.held(h)
            and ready.query()
            and self.forwards.get(h, CompletedReceipt()).query()
        )

    def release_pages(self, handles, *, ready):
        with self.lock:
            self.check()
            for h in handles:
                if h != self.handle(h.page_id):
                    raise RuntimeError("stale physical return")
                state = self.states[h.page_id]
                if state == "quarantined" and h not in self.quarantine_returned:
                    self.quarantine_returned.add(h)
                    continue
                if state not in ("in_use", "publish_sent", "published"):
                    raise RuntimeError("duplicate physical return")
                self.pending_publish.pop(h, None)
                if state == "in_use":
                    self.states[h.page_id] = (
                        "cached" if self.safe(h, ready) else "pending_free"
                    )
                    self.guards[h] = ready
                else:
                    op = self.next_op
                    self.next_op += 1
                    self.retires[op] = h, ready
                    self.states[h.page_id] = "retiring"
                    self._send_retire(op, h)

    def release_page_ids(self, page_ids, *, ready):
        with self.lock:
            self.check()
            self.release_pages(tuple(self.handle(p) for p in page_ids), ready=ready)

    def _send_retire(self, op, h):
        if (
            op not in self.sent_retires
            and not self.held(h)
            and self.states[h.page_id] == "retiring"
        ):
            self.commands.append(("retire", op, h))
            self.sent_retires.add(op)

    def released(self, op, h):
        with self.lock:
            entry = self.retires.get(op)
            if entry is None or entry[0] != h or h != self.handle(h.page_id):
                return False
            if (
                self.states[h.page_id] != "retiring"
                or self.held(h)
                or op not in self.sent_retires
            ):
                return False
            del self.retires[op]
            self.released_ops.add(op)
            self.states[h.page_id] = (
                "cached" if self.safe(h, entry[1]) else "pending_free"
            )
            self.guards[h] = entry[1]
            self.records.pop(h, None)
            return True

    def hold_io(self, handles, *, kind, writes, receipt):
        with self.lock:
            self.check()
            if len(set(handles)) != len(handles) or any(
                h.page_id not in self.states
                or h != self.handle(h.page_id)
                or self.states[h.page_id] not in ("in_use", "publish_sent", "published")
                or (writes and self.states[h.page_id] != "in_use")
                for h in handles
            ):
                raise RuntimeError("I/O hold requires current live allocations")
            ticket = self.next_op
            self.next_op += 1
            self.io[ticket] = tuple(handles), receipt, kind
            return ticket

    def complete_io(self, ticket, *, outcome):
        with self.lock:
            if outcome not in ("completed", "never_submitted", "unknown"):
                raise ValueError(f"invalid physical I/O outcome: {outcome}")
            held = self.io.get(ticket)
            if held is None:
                return
            if outcome == "unknown":
                for h in held[0]:
                    self.states[h.page_id] = "quarantined"
                    self.pending_publish.pop(h, None)
                # Retain backing and dependency receipts, but no active hold
                # that would block logical reset. Isolation is irreversible.
                guards = tuple(
                    (self.forwards.get(h), self.guards.get(h)) for h in held[0]
                )
                self.quarantine_resources.append((*held, guards))
            del self.io[ticket]

    def progress(self):
        with self.lock:
            self.check()
            for h, (page, ready) in list(self.pending_publish.items()):
                if ready.query() and self.forwards.get(h, CompletedReceipt()).query():
                    # Cancellation, state change and enqueue share this lock.
                    if self.before_send is not None:
                        self.before_send()
                    self.states[h.page_id] = "publish_sent"
                    self.records[h] = page
                    self.commands.append(("publish", self.next_op, h))
                    self.next_op += 1
                    del self.pending_publish[h]
            for h, ready in list(self.guards.items()):
                if self.states[h.page_id] == "pending_free" and self.safe(h, ready):
                    self.states[h.page_id] = "cached"
            for op, (h, _) in list(self.retires.items()):
                self._send_retire(op, h)
                if self.auto_release:
                    self.released(op, h)
            events, self.events = self.events, []
            return events

    def complete_tree_event(self, event, result):
        self.event_results.append((event, result))

    def quiesce_reset(self, *, deadline_ns):
        self.progress()
        with self.lock:
            if (
                self.io
                or any(
                    s in ("reserved", "pending_free", "retiring")
                    for s in self.states.values()
                )
                or any(
                    not f.query()
                    for h, f in self.forwards.items()
                    if self.states[h.page_id] != "quarantined"
                )
            ):
                self.commands.append(("reset_cancel_ack", self.epoch))
                return Busy("live grant/I/O/guard/reservation")
            self.lease = FakeResetLease(
                self.epoch,
                replace(self.epoch, reset_seq=self.epoch.reset_seq + 1),
                tuple(p for p, s in self.states.items() if s != "quarantined"),
            )
            return self.lease

    def validate_reset(self, lease):
        if self.lease is not lease:
            raise RuntimeError("invalid reset lease")

    def commit_reset(self, lease):
        self.validate_reset(lease)
        self.epoch = lease.new_epoch
        for p in lease.approved_pages:
            self.states[p] = "cached"
        self.pending_publish.clear()
        self.records.clear()
        self.forwards.clear()
        self.guards.clear()
        self.retires.clear()
        self.sent_retires.clear()
        self.lease = None

    def abort_reset(self, lease):
        self.validate_reset(lease)
        self.lease = None

    def accounting(self):
        with self.lock:
            blocked = {
                h
                for handles, _, _ in self.io.values()
                for h in handles
                if self.states[h.page_id] in ("publish_sent", "published")
            }
            return PageAccounting(
                capacity=len(self.states),
                blocked_reclaim_estimate=len(blocked),
                **Counter(self.states.values()),
            )

    def accounting_violations(self):
        return self.accounting().violations()

    def validate_projection(self, indices, *, page_ids=None):
        pages = self.page_ids(indices) if page_ids is None else tuple(page_ids)
        expected = {
            p
            for p, s in self.states.items()
            if s in ("in_use", "publish_sent", "published")
        }
        quarantined = {p for p, s in self.states.items() if s == "quarantined"}
        if not expected <= set(pages) <= expected | quarantined:
            return [
                f"provider projection mismatch: projected={set(pages)}, live={expected}"
            ]
        # Tree/request values must contain each token once; distinct tokens of
        # an unfinished tail legitimately name the same physical page.
        if indices is None:
            return []
        from sglang.srt.mem_cache.provider_indices import index_cpu

        ids = index_cpu(indices).tolist()
        return ["duplicate projected KV token"] if len(ids) != len(set(ids)) else []

    def close(self):
        with self.lock:
            self.closed = True


def build_provider_fixture(
    *,
    pages=16,
    page_size=4,
    backend="python",
    provider_enabled=True,
    ring=True,
    components=None,
    provider_override=None,
):
    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.static_kv_storage import kv_storage_of
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.test.tree_slots import tree_slots_of

    provider = provider_override or FakeKVMemoryProvider(
        pages=pages, page_size=page_size
    )
    pool = SWAKVPool(
        size=pages * page_size,
        size_swa=pages * page_size,
        page_size=page_size,
        dtype=torch.float32,
        head_num=1,
        head_dim=8,
        swa_attention_layer_ids=[1],
        full_attention_layer_ids=[0],
        device="cpu",
    )
    if provider_enabled:
        pool.kv_memory_provider = provider
    # The approved DSv4 layout keeps SWA in per-request rings, outside FULL
    # objects. CPU tensors stand in for their bytes; the real ring accounting
    # and allocator paths are still used.
    pool.swa_req_ring_size = page_size * 2 if ring else None
    req_pool = ReqToTokenPool(
        size=8, max_context_len=128, device="cpu", enable_memory_saver=False
    )
    allocator = SWATokenToKVPoolAllocator(
        size=pages * page_size,
        size_swa=pages * page_size,
        page_size=page_size,
        dtype=torch.float32,
        device="cpu",
        kvcache=pool,
        need_sort=True,
        req_to_token_pool=req_pool,
    )
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            tree_slots=tree_slots_of(
                components or (ComponentType.FULL,), window=page_size * 2
            ),
            sliding_window_size=page_size * 2,
            swa_gates_prefix_match=not ring,
            tree_core_backend=backend,
        )
    )
    if provider_enabled:
        cache.tree_core.enable_external_cache_linker = True
    return provider, allocator, cache, kv_storage_of(req_pool, allocator)


def take_pages(source, provider, n, *, purpose="extend", request_id="test"):
    from sglang.srt.mem_cache.kv_memory_provider import CapacityShape

    shape = CapacityShape(
        request_id,
        1,
        n if purpose == "load" else 0,
        n if purpose == "extend" else 0,
        0,
        n * source.page_size,
    )
    reservation = provider.reserve(shape)
    with source.reservation_scope(reservation, purpose):
        return source.alloc(n * source.page_size)


def make_fake_linker(provider):
    from sglang.srt.mem_cache.hicache_storage import PoolName
    from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
        LayerWiseLoadCounter,
        UnifiedCacheLinker,
    )

    class FakeLinker(UnifiedCacheLinker):
        def __init__(self):
            self.provider = provider
            self.pools = {PoolName.KV: object()}
            self.layer_done_counter = LayerWiseLoadCounter(2)
            self.queued = {}
            self.reject = False
            self.outcome = "completed"

        def lookup(self, rid, transfers):
            return list(range(1, len(transfers[0].keys) + 1))

        def load(self, rid, transfers):
            if self.reject:
                return False
            indices = transfers[0].device_indices
            handles = provider.freeze_allocations(indices).handles(indices)
            ticket = provider.hold_io(
                handles, kind="load-target", writes=True, receipt=self
            )
            self.queued[rid] = ticket
            return True

        def cancel_queued_load(self, rid):
            ticket = self.queued.pop(rid, None)
            if ticket is not None:
                provider.complete_io(ticket, outcome="never_submitted")
            return True

        def start_layer_wise_loading(self):
            index = self.layer_done_counter.update_producer()
            self.started = self.queued
            self.queued = {}
            return index

        def finish_layer_wise_loading(self, index):
            for ticket in self.started.values():
                provider.complete_io(
                    ticket,
                    outcome="unknown" if self.outcome == "unknown" else "completed",
                )
            self.started = {}
            if self.outcome in ("failed", "unknown"):
                self.layer_done_counter.fail(index, RuntimeError("batch load failed"))
            else:
                for layer in range(2):
                    self.layer_done_counter.complete(index, layer)
            return self.layer_done_counter.finish(index)

        def offload(self, transfers, session_id=None):
            raise AssertionError("provider does not offload through the linker")

        def num_completed_offloads(self):
            return 0

        def pop_completed_offload(self):
            raise AssertionError("no provider offload queue")

        def reset(self):
            if self.queued or getattr(self, "started", {}):
                raise ProviderFailStop("reset with unfinished loads")

        def close(self):
            self.reset()

    return FakeLinker()
