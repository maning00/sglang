"""Optional ownership of static KV bytes and FULL pages.

These are rank-local contracts, independent of a storage implementation. Only
the provider owns page generations, publication queues, holds and retire ACKs.
Tree projections and device free-page tensors never grant reuse permission.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class PoolEpoch:
    boot_id: bytes
    reset_seq: int


@dataclass(frozen=True)
class PageHandle:
    pool_id: str
    epoch: PoolEpoch
    page_id: int
    generation: int


@dataclass(frozen=True)
class PageRegion:
    region_id: str
    base: int
    length: int
    row_bytes: int
    rows_per_page: int
    logical_offset: int
    layer_id: int
    component: str

    @property
    def page_bytes(self) -> int:
        return self.row_bytes * self.rows_per_page


@dataclass(frozen=True)
class PageGeometry:
    page_tokens: int
    num_pages: int
    object_bytes: int
    layout_id: str
    regions: tuple[PageRegion, ...]

    def validate(self) -> None:
        if self.page_tokens <= 0 or self.num_pages <= 0 or not self.regions:
            raise ValueError("empty KV page geometry")
        offset, names = 0, set()
        for region in self.regions:
            if region.region_id in names or region.logical_offset != offset:
                raise ValueError("KV regions must cover the object exactly once")
            if region.base <= 0 or region.page_bytes <= 0:
                raise ValueError("invalid KV region")
            if region.length < (self.num_pages + 1) * region.page_bytes:
                raise ValueError("KV region does not include all pages and padding")
            names.add(region.region_id)
            offset += region.page_bytes
        if offset != self.object_bytes or not self.layout_id:
            raise ValueError("inconsistent KV object geometry")


@dataclass(frozen=True)
class RequestRegion:
    region_id: str
    base: int
    length: int
    kind: str


@dataclass(frozen=True)
class CapacityShape:
    request_id: str
    revision: int
    load_pages: int
    extend_pages: int
    prefix_tokens: int
    end_tokens: int


class ReadyReceipt(Protocol):
    """Thread-safe rank-local completion proof, queried by engine workers.

    query must be non-blocking and never issue RPCs; wait blocks off the
    scheduler thread (native/event waits release the GIL), without polling.
    """

    def query(self) -> bool: ...
    def wait_on(self, stream: Any) -> None: ...
    def wait(self) -> None: ...


@dataclass(frozen=True)
class EventReceipt:
    """Keep the native event alive, including its stream-ordering operation."""

    event: Any

    def query(self) -> bool:
        return self.event.query()

    def wait_on(self, stream: Any) -> None:
        stream.wait_event(self.event)

    def wait(self):
        self.event.synchronize()


@dataclass(frozen=True)
class CompletedReceipt:
    def query(self) -> bool:
        return True

    def wait_on(self, stream: Any) -> None:
        pass

    def wait(self):
        pass


@dataclass(frozen=True)
class JoinedReceipt:
    receipts: tuple[ReadyReceipt, ...]

    def query(self):
        return all(r.query() for r in self.receipts)

    def wait_on(self, stream):
        for receipt in self.receipts:
            receipt.wait_on(stream)

    def wait(self):
        for receipt in self.receipts:
            receipt.wait()


def indices_ready(indices) -> ReadyReceipt:
    """An index tensor may have been produced on the scheduler stream."""
    if indices.device.type == "cpu":
        return CompletedReceipt()
    import torch

    return EventReceipt(torch.cuda.current_stream(indices.device).record_event())


class FrozenAllocationContext(Protocol):
    def handles(self, indices: Any = None) -> tuple[PageHandle, ...]: ...
    def ready(self, handles: Sequence[PageHandle]) -> ReadyReceipt: ...


def allocation_page_ids(indices, page_size):
    """CPU page-ID sideband, not an ownership or generation mirror."""
    hint = getattr(indices, "_kv_provider_page_ids", None)
    if hint is not None:
        return tuple(hint)
    from sglang.srt.mem_cache.provider_indices import index_cpu

    return tuple(
        dict.fromkeys((index_cpu(indices.detach()).reshape(-1) // page_size).tolist())
    )


class Reservation(Protocol):
    shape: CapacityShape

    def take(self, purpose: str, n: int) -> tuple[PageHandle, ...]: ...
    def remaining(self, purpose: str) -> int: ...
    def covers(self, shape: CapacityShape) -> bool: ...
    def refund_unused(self) -> None: ...
    def revise(self, shape: CapacityShape) -> bool: ...

    # In-place remaining-purpose adjustment. Consumed handles/I/O survive;
    # failure preserves the old ticket; no unreserve/re-reserve gap.


@dataclass(frozen=True)
class PageSourceSpec:
    size: int
    page_size: int
    dtype: Any
    device: str
    kvcache: Any
    need_sort: bool


class PageSource(Protocol):
    page_size: int
    free_pages: Any

    def available_size(self) -> int: ...
    def alloc(self, need_size: int) -> Any: ...
    def reservation_scope(self, reservation: Reservation, purpose: str) -> Any: ...
    def batch_reservation_scope(
        self, reqs: Sequence[Any], prefix_lens: Any, seq_lens: Any
    ) -> Any: ...
    def _release_page_ids(self, *page_ids: Any) -> None: ...
    def clear(self, *, reset_lease: Any = None) -> None: ...


class IOTicket(Protocol):
    """Opaque identity owned by the provider, independent of request rows."""


@dataclass(frozen=True)
class Short:
    missing_pages: int
    retryable: bool = True


@dataclass(frozen=True)
class Busy:
    # Returning Busy requires acknowledged cancellation of the reset seal.
    reason: str


class ProviderFailStop(RuntimeError):
    """Reuse or completion cannot be established; the worker must stop."""


class ResetLease(Protocol):
    old_epoch: PoolEpoch
    new_epoch: PoolEpoch
    approved_pages: tuple[int, ...]


@dataclass(frozen=True)
class ResidentPage:
    key: str
    parent_key: str | None
    handle: PageHandle


@dataclass(frozen=True)
class DetachCandidate:
    node_id: int
    revision: int
    keys: tuple[str, ...]
    handles: tuple[PageHandle, ...]

    def __post_init__(self):
        object.__setattr__(self, "keys", tuple(self.keys))
        object.__setattr__(self, "handles", tuple(self.handles))


@dataclass(frozen=True)
class DetachResult:
    candidate: DetachCandidate
    status: Literal["detached", "stale", "busy"]


@dataclass(frozen=True)
class ProviderTreeEvent:
    operation_id: Any
    kind: Literal["offer", "detach"]
    n_pages: int = 0
    candidates: tuple[DetachCandidate, ...] = ()


@dataclass(frozen=True)
class PageAccounting:
    capacity: int
    cached: int = 0
    reserved: int = 0
    in_use: int = 0
    pending_free: int = 0
    publish_sent: int = 0
    published: int = 0
    retiring: int = 0
    quarantined: int = 0
    free: int = 0
    # Latest node aggregate of published slots blocked by grants/I/O. This is
    # an overlapping estimate, never a partition or a reuse permission.
    blocked_reclaim_estimate: int = 0

    def violations(self) -> list[str]:
        counts = tuple(
            v
            for k, v in vars(self).items()
            if k not in ("capacity", "blocked_reclaim_estimate")
        )
        if min((self.capacity, *counts)) < 0 or sum(counts) != self.capacity:
            return [f"provider page conservation: {self}"]
        if not 0 <= self.blocked_reclaim_estimate <= self.capacity:
            return [f"invalid provider reclaim estimate: {self}"]
        return []


@runtime_checkable
class ByteMemoryProvider(Protocol):
    def mem_pool(self, device: str) -> Any: ...


@runtime_checkable
class KVMemoryProvider(ByteMemoryProvider, Protocol):
    """A complete provider. All operations must reject a failed session.

    reserve/take/refund are local. progress never re-enters the tree. release
    serializes pre-send cancellation with publication's atomic enqueue point;
    sent pages require the original retire operation's Released ACK. Unknown
    I/O retains native resources. Unknown writes quarantine immediately;
    unknown reads may keep their native hold open while the source is owned,
    preserving reads/publication and quarantining on final physical return.
    These open holds keep reset Busy and prevent network deregistration.
    Quarantine persists across logical reset.
    Quarantined holds do not make reset Busy: exclude their pages from the
    approval set and retain backing through commit. Busy requires acknowledged
    cancellation; Ready remains sealed until commit/abort with a full approval
    set. Accounting is a local snapshot, never an RPC.

    On published pages, local PD/I/O holds delay Retire. Released alone does
    not imply reuse: an unfinished reuse guard leaves the page PENDING_FREE.
    freeze only accepts live allocations; use CPU page_ids whenever available,
    avoiding D2H per action. Returning a quarantined allocation uses the same
    release entry, not a new live-allocation freeze.
    """

    def register_pool(
        self, geometry: PageGeometry, request_regions: Sequence[RequestRegion]
    ) -> str: ...
    def create_linker(
        self, server_args: Any, params: Any, *, components: Any, forward_stream: Any
    ) -> Any: ...
    def reserve(self, shape: CapacityShape) -> Reservation | Short: ...
    def freeze_allocations(
        self,
        indices: Any,
        *,
        context: FrozenAllocationContext | None = None,
        page_ids: Sequence[int] | None = None,
    ) -> FrozenAllocationContext: ...
    def release_page_ids(
        self, page_ids: Sequence[int], *, ready: ReadyReceipt
    ) -> None: ...
    def release_pages(
        self, handles: Sequence[PageHandle], *, ready: ReadyReceipt
    ) -> None: ...
    def publish(
        self, pages: Sequence[ResidentPage], *, ready: ReadyReceipt
    ) -> None: ...
    def hold_io(
        self, handles: Sequence[PageHandle], *, kind: str, writes: bool, receipt: Any
    ) -> Any: ...
    def complete_io(self, ticket: Any, *, outcome: str) -> None: ...
    @property
    def has_pending_io(self) -> bool:
        """Open local holds, including completed receipts parked until return."""
        ...

    def record_forward(
        self, handles: Sequence[PageHandle], done: ReadyReceipt
    ) -> None: ...
    def quiesce_reset(self, *, deadline_ns: int) -> ResetLease | Busy: ...
    def validate_reset(self, lease: ResetLease) -> None: ...
    def commit_reset(self, lease: ResetLease) -> None: ...
    def abort_reset(self, lease: ResetLease) -> None: ...
    def progress(self) -> Sequence[ProviderTreeEvent]: ...
    def complete_tree_event(
        self, event: ProviderTreeEvent, result: Sequence[Any]
    ) -> None: ...
    def accounting(self) -> PageAccounting: ...
    def accounting_violations(self) -> list[str]: ...
    def validate_projection(
        self, indices: Any, *, page_ids: Sequence[int] | None = None
    ) -> list[str]: ...
    def close(self) -> None: ...


class CapacityScope:
    """One admission's reservation, including every shape revision.

    The scheduler owns tree traversal. A wait only advances provider receipts;
    it cannot wait on the batch's not-yet-started loads. The deadline bounds all
    retire rounds together, rather than restarting the timeout for each leaf.
    """

    def __init__(
        self,
        provider,
        tree,
        shape,
        *,
        deadline_ns,
        verdict=lambda ok: ok,
        stats=None,
        can_wait=True,
    ):
        self.provider, self.tree = provider, tree
        self.shape, self.deadline_ns = shape, deadline_ns
        self.verdict = verdict
        self.reservation = None
        self.committed = False
        self.stats = stats if stats is not None else {}
        self._wait_before = self.stats.get("wait_ns", 0)
        self.can_wait = can_wait
        self.consumed_load_pages = 0

    def __enter__(self):
        try:
            self.revise(self.shape)
            return self
        except BaseException:
            self._refund()
            raise

    @property
    def ready(self):
        return self.reservation is not None

    def _refund(self):
        if self.reservation is not None:
            self.reservation.refund_unused()
            self.reservation = None
        self.consumed_load_pages = 0

    def revise(self, shape, *, require_coverage=False):
        if self.reservation is not None:
            consumed = self.shape.load_pages - self.reservation.remaining("load")
            adjusted = self.reservation.revise(shape)
            if not self.verdict(adjusted):
                raise ProviderFailStop(
                    "loaded request reservation cannot cover its revised shape"
                )
            self.consumed_load_pages += consumed
            self.shape = shape
            return True
        if require_coverage:
            raise ProviderFailStop("loaded request lost its reservation")
        self.shape = shape
        started = time.monotonic_ns()
        attempts = 0
        while True:
            result = self.provider.reserve(shape)
            attempts += 1
            short = isinstance(result, Short)
            if not short:
                self.reservation = result
            if self.verdict(not short):
                break
            missing = self.tree.provider_shortage(result.missing_pages if short else 0)
            self._refund()
            if not self.verdict(
                (not short or result.retryable)
                and time.monotonic_ns() < self.deadline_ns
            ):
                break
            # Whole-leaf confirmation finishes before receipt progress. Never
            # suspend an eviction walk across a provider/control call.
            candidates = self.tree.offer_candidates(missing)
            detached = self.tree.confirm_detach(candidates)
            self.tree.check_provider_events()
            pages = sum(
                len(r.candidate.handles) for r in detached if r.status == "detached"
            )
            self.stats["detached_pages"] = self.stats.get("detached_pages", 0) + pages
            self.stats["over_detached_pages"] = self.stats.get(
                "over_detached_pages", 0
            ) + max(0, pages - missing)
            self.stats["retire_rounds"] = self.stats.get("retire_rounds", 0) + 1
            self.stats["candidates"] = self.stats.get("candidates", 0) + len(candidates)
            # ACK/I/O processing can already have refilled CACHED. One fast
            # retry does not wait for the queued batch or another scheduler turn.
            result = self.provider.reserve(shape)
            attempts += 1
            if not isinstance(result, Short):
                self.reservation = result
            if self.verdict(self.ready):
                break
            self._refund()
            snapshot = self.provider.accounting()
            progressing = (
                snapshot.cached >= shape.load_pages + shape.extend_pages
                or pages > 0
                or snapshot.retiring > 0
                or snapshot.pending_free > 0
            )
            # A queued batch must run before its loads/retirement can advance.
            # With no physical work progressing, yield to the scheduler now.
            if not self.verdict(self.can_wait and progressing):
                break
            if not pages:
                time.sleep(0.0005)
        for key, value in (
            ("prepare", 1),
            ("attempts", attempts),
            ("short", int(not self.ready)),
            ("fast", int(self.ready and attempts == 1)),
            ("wait_ns", time.monotonic_ns() - started),
        ):
            self.stats[key] = self.stats.get(key, 0) + value
        return self.ready

    def commit_to(self, req):
        if not self.ready or req.kv.reservation is not None:
            raise RuntimeError("invalid KV reservation handoff")
        req.kv.reservation = self.reservation
        self.committed = True

    def __exit__(self, exc_type, exc, tb):
        if not self.committed or exc_type is not None:
            self._refund()
        self.stats["requests"] = self.stats.get("requests", 0) + 1
        self.stats.setdefault("request_wait_ns", []).append(
            self.stats.get("wait_ns", 0) - self._wait_before
        )


def refund_request_reservation(req) -> None:
    reservation = getattr(req.kv, "reservation", None)
    if reservation is not None:
        reservation.refund_unused()
        req.kv.reservation = None


def reservation_request_id(req):
    """An instance identity that cannot repeat after Python object/row reuse."""
    if req.kv.reservation_identity is None:
        import uuid

        req.kv.reservation_identity = uuid.uuid4().hex
    return f"{req.rid}:{req.kv.reservation_identity}"


@contextmanager
def no_capacity_scope():
    yield None


def describe_dsv4_pool(pool, *, num_pages):
    """Use the PD byte geometry and native layer/component descriptions.

    get_contiguous_buf_infos already skips the unified SWA ring. Match it to
    get_device_pool_infos instead of reconstructing the model's 182 segments.
    """
    import hashlib
    import json

    metadata = {}
    for info in pool.get_device_pool_infos():
        if str(info.indices_from_pool.value) != "kv":
            continue
        for layer, buffer in zip(info.layer_ids, info.buffer_info.buffers, strict=True):
            metadata[buffer.data_ptr()] = (
                layer,
                info.pool_name.value,
                str(buffer.dtype),
                info.buffer_info.layout,
            )
    pointers, lengths, items = pool.get_contiguous_buf_infos()
    regions, layout, offset = [], [], 0
    for index, (base, length, item) in enumerate(
        zip(pointers, lengths, items, strict=True)
    ):
        if base not in metadata:
            raise ValueError("KV provider cannot describe a FULL page region")
        layer, component, dtype, format_ = metadata.pop(base)
        regions.append(
            PageRegion(
                f"{component}:{layer}", base, length, item, 1, offset, layer, component
            )
        )
        layout.append((layer, component, dtype, format_, item))
        offset += item
    if metadata:
        raise ValueError("KV provider geometry omitted a page component")
    layout_id = hashlib.sha256(
        json.dumps((pool.page_size, layout), separators=(",", ":")).encode()
    ).hexdigest()
    geometry = PageGeometry(
        pool.page_size, num_pages, offset, layout_id, tuple(regions)
    )
    geometry.validate()
    requests = []
    for kind in (
        "get_state_buf_infos",
        "get_request_state_buf_infos",
        "get_unified_swa_ring_buf_infos",
        "get_swa_valid_from_buf_infos",
    ):
        bases, sizes, _ = getattr(pool, kind)()
        for i, (base, size) in enumerate(zip(bases, sizes, strict=True)):
            if size:
                requests.append(
                    RequestRegion(f"{kind}:{i}", base, size, "request-only")
                )
    return geometry, tuple(requests)
