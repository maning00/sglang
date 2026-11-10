"""SGLang adapter for the rank-local mori.umbp.engine ownership authority.

Only explicit provider construction imports mori. Handles come from Take;
this adapter keeps capabilities/receipts, never an independent free list or
page-state machine. Native client/executor and torch MemPool are shared with
PageSource and the engine linker.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
import uuid
import weakref
from dataclasses import dataclass
from threading import RLock

from sglang.srt.mem_cache.kv_memory_provider import (
    Busy,
    CompletedReceipt,
    JoinedReceipt,
    PageAccounting,
    PageHandle,
    PoolEpoch,
    ProviderFailStop,
    Short,
    allocation_page_ids,
)
from sglang.srt.mem_cache.utils import storage_namespace_seed
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)


@dataclass
class _ResetLease:
    native: object
    old_epoch: PoolEpoch
    new_epoch: PoolEpoch
    approved_pages: tuple[int, ...]


@dataclass
class _IOTicket:
    native: object
    handles: tuple[PageHandle, ...]
    receipt: object
    kind: str
    writes: bool
    parked: bool = False


@dataclass
class _Publication:
    ticket: int
    records: tuple
    ready: object
    armed: bool


class _FrozenAllocations:
    def __init__(self, provider, handles, indices, page_ids):
        self.provider = provider
        self.indices, self.page_ids = indices, tuple(page_ids)
        self.by_page = {h.page_id: h for h in handles}
        self.receipts = {
            h: provider._forwards.get(h, CompletedReceipt()) for h in handles
        }

    def handles(self, indices=None):
        if indices is None or indices is self.indices:
            return tuple(self.by_page.values())
        return tuple(
            self.by_page[p]
            for p in allocation_page_ids(indices, self.provider.geometry.page_tokens)
        )

    def ready(self, handles):
        return JoinedReceipt(tuple(self.receipts[h] for h in handles))


def _rank_slot(options, device_index):
    """Node-local topology slot, independent of CUDA visibility/remapping."""
    if "rank" in options:
        raise ValueError(
            "options.rank is shared by workers; use rank_base or worker-scoped UMBP_ENGINE_RANK"
        )
    base = options.get("rank_base", 0)
    if type(base) is not int or not 0 <= base < 8:
        raise ValueError("rank_base must be in the fixed eight-rank topology")
    override = os.environ.get("UMBP_ENGINE_RANK")
    if override is not None:
        try:
            rank = int(override)
        except ValueError as exc:
            raise ValueError(
                "UMBP_ENGINE_RANK must be a node-local integer slot"
            ) from exc
    elif device_index < 0:
        rank = base  # CPU control tests have no serving parallel context.
    else:
        parallel = get_parallel()
        local_tp = parallel.tp_size // parallel.nnodes
        if not local_tp or parallel.tp_size % parallel.nnodes:
            raise ValueError("engine rank requires an integral TP placement per node")
        # dp_rank is a replica offset only across separate WORLD groups. Under
        # DP attention one WORLD group spans the node and dp_rank aliases the
        # attention-DP rank, so tp_rank alone is the node-local position.
        replica = (
            (parallel.dp_rank or 0) if getattr(parallel, "attn_dp_size", 1) <= 1 else 0
        )
        rank = base + replica * local_tp + parallel.tp_rank % local_tp
    if not 0 <= rank < 8:
        raise ValueError("engine rank must be in the fixed eight-rank topology")
    return rank


class _Reservation:
    """Purpose quotas over native local reservations, without a refund gap."""

    def __init__(self, provider, shape, native_id=None):
        self.provider, self.shape = provider, shape
        self.parts = []
        if native_id is not None:
            self.parts.append([native_id, shape.load_pages, shape.extend_pages])
        self.left = {"load": shape.load_pages, "extend": shape.extend_pages}
        self.closed = False

    def remaining(self, purpose):
        return self.left[purpose] if not self.closed else 0

    def covers(self, shape):
        return not self.closed and self.shape == shape

    def take(self, purpose, n):
        p = self.provider
        with p._lock:
            p._check()
            if (
                self.closed
                or purpose not in self.left
                or not 0 <= n <= self.left[purpose]
            ):
                raise RuntimeError("engine reservation over-consumed")
            index = 1 if purpose == "load" else 2
            kind = getattr(p.engine.ReservationPurpose, purpose.title())
            native, remaining = [], n
            for part in self.parts:
                count = min(part[index], remaining)
                if count:
                    taken = p._call(p.client.take, part[0], kind, count)
                    p._require(
                        len(taken) == count, "Take did not cover its reservation"
                    )
                    native.extend(taken)
                    part[index] -= count
                    remaining -= count
                if not remaining:
                    break
            p._require(remaining == 0, "native purpose quota missing")
            self.left[purpose] -= n
            return tuple(p._adopt(h) for h in native)

    def revise(self, shape):
        p = self.provider
        with p._lock:
            p._check()
            if min(shape.load_pages, shape.extend_pages) < 0:
                raise ValueError("negative capacity shape")
            if (
                self.closed
                or shape.request_id != self.shape.request_id
                or shape.revision <= self.shape.revision
            ):
                return False
            wanted = {"load": shape.load_pages, "extend": shape.extend_pages}
            available = {
                k: sum(part[i] for part in self.parts)
                for k, i in (("load", 1), ("extend", 2))
            }
            extra = {k: max(0, wanted[k] - available[k]) for k in wanted}
            if any(extra.values()):
                result = p._call(
                    p.client.reserve,
                    f"{shape.request_id}:{shape.revision}",
                    extra["load"],
                    extra["extend"],
                )
                if not result.success:
                    p._require(
                        result.shortfall > 0,
                        "reservation revise failed without a shortfall",
                    )
                    return False
                self.parts.append([result.id, extra["load"], extra["extend"]])
            # Native Refund is whole-ticket only. Reduced unused quotas remain
            # RESERVED until the existing scope's refund_unused, never issued.
            self.left, self.shape = wanted, shape
            return True

    def refund_unused(self):
        p = self.provider
        with p._lock:
            if self.closed:
                return
            p._check()
            for native_id, _, _ in self.parts:
                p._require(
                    p._call(p.client.refund, native_id), "reservation Refund rejected"
                )
            self.closed = True
            self.left = {"load": 0, "extend": 0}
            self.parts.clear()


class UMBPEngineProvider:
    # Enforced by the existing sole factory, including distinct device keys.
    process_singleton = True

    def __init__(self, *, device, options):
        import torch

        device = torch.device(device)
        if device.type not in ("cpu", "cuda") or (
            device.type == "cuda" and device.index is None
        ):
            raise ValueError("engine provider requires cpu or an explicit cuda ordinal")
        unknown = set(options) - {
            "node_address",
            "namespace",
            "rank",
            "rank_base",
            "heartbeat_ms",
            "heartbeat_timeout_ms",
            "dram",
            "io_host",
            "io_port",
            "staging_bytes",
            "capacity_timeout_ms",
            "reset_timeout_ms",
        }
        if unknown:
            raise ValueError(f"unknown engine provider options: {sorted(unknown)}")
        address, namespace = options.get("node_address"), options.get("namespace")
        if not isinstance(address, str) or not address.startswith("unix:///"):
            raise ValueError("engine node_address must be an absolute unix:/// address")
        if not isinstance(namespace, dict) or not all(
            namespace.get(k) for k in ("model", "weights", "kv_format", "parallel")
        ):
            raise ValueError(
                "engine namespace requires model/weights revision/kv_format/parallel"
            )
        self.namespace = json.dumps(
            namespace, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        self.namespace_seed = storage_namespace_seed(self.namespace, None)
        self.device, self.device_index = (
            str(device),
            device.index if device.type == "cuda" else -1,
        )
        self.rank = _rank_slot(options, self.device_index)
        heartbeat = options.get("heartbeat_ms", 500)
        timeout = options.get("heartbeat_timeout_ms", 5000)
        if (
            type(heartbeat) is not int
            or type(timeout) is not int
            or not 0 < heartbeat <= timeout
        ):
            raise ValueError(
                "heartbeat requires 0 < heartbeat_ms <= heartbeat_timeout_ms"
            )
        self.dram_options = options.get("dram", {})
        if not isinstance(self.dram_options, dict) or set(self.dram_options) - {
            "min_bytes",
            "region_ids",
        }:
            raise ValueError("dram accepts min_bytes and required region_ids")
        if (
            type(self.dram_options.get("min_bytes", 0)) is not int
            or self.dram_options.get("min_bytes", 0) < 0
        ):
            raise ValueError("dram.min_bytes must be nonnegative")
        if not isinstance(self.dram_options.get("region_ids", []), list) or not all(
            isinstance(x, str) for x in self.dram_options.get("region_ids", [])
        ):
            raise ValueError("dram.region_ids must be a list of region names")
        self.capacity_timeout_ms = options.get("capacity_timeout_ms", 50)
        self.reset_timeout_ms = options.get("reset_timeout_ms", timeout)
        if (
            not 0 < self.capacity_timeout_ms <= timeout
            or not 0 < self.reset_timeout_ms <= timeout
        ):
            raise ValueError(
                "capacity/reset deadlines must be positive and bounded by the heartbeat timeout"
            )
        # No import-time dependency on mori for provider-off serving.
        from mori.umbp import engine

        self.engine = engine
        self.executor_id = f"sglang:{socket.gethostname()}:{os.getpid()}:{self.device}"
        self.pool_id = uuid.uuid4().hex
        self.client = engine.EngineClient(address, self.executor_id, heartbeat, timeout)
        if any(
            not callable(getattr(self.client, name, None))
            for name in ("acknowledge_publish_receipt", "consume_event_overflow")
        ):
            self.client.close()
            raise ValueError(
                "UMBP engine requires publish receipt ACK and event overflow bindings"
            )
        self.executor = engine.EngineExecutor(
            self.executor_id,
            self.device_index,
            options.get("io_host", ""),
            options.get("io_port", 0),
            options.get("staging_bytes", 0),
        )
        self.client.attach_executor(self.executor)
        self._lock = RLock()
        self._pool = self.geometry = self.registration = self._lease = None
        self._closed = False
        self._failed = None
        self._by_page, self._native, self._forwards = {}, {}, {}
        self._owned, self._quarantined = set(), set()
        self._io, self._guards, self._publications, self._records = {}, {}, {}, {}
        self._publication_by_handle, self._unready_publications = {}, {}
        self._publication_results, self._record_tickets = {}, {}
        self._retained, self.vmm_exports = [], []
        self._reservations = weakref.WeakSet()
        self._linker = None
        logger.info(
            "UMBP engine provider created device=%s rank=%d node=%s executor=%s namespace=%s",
            self.device,
            self.rank,
            address,
            self.executor_id,
            self.namespace_seed,
        )

    def _fail(self, message):
        self._failed = message
        raise ProviderFailStop(message)

    def _check(self, *, registered=True, sealed=False):
        if self._closed or self._failed:
            raise ProviderFailStop(self._failed or "engine provider closed")
        if self.registration is not None and not self.client.healthy:
            self._fail(
                "UMBP engine session disconnected; no local allocation/reconnect allowed"
            )
        if registered:
            if self.registration is None:
                raise RuntimeError("engine pool is not registered")
        if self._lease is not None and not sealed:
            raise RuntimeError("engine pool sealed for reset")

    def _call(self, method, *args):
        try:
            result = method(*args)
        except Exception as exc:
            self._failed = f"UMBP engine {method.__name__} failed: {exc}"
            raise ProviderFailStop(self._failed) from exc
        if self.registration is not None and not self.client.healthy:
            self._fail("UMBP engine control channel failed")
        return result

    def _require(self, value, message):
        if not value:
            self._fail(message)

    @staticmethod
    def _epoch(native):
        return PoolEpoch(bytes(native.boot_id), native.reset_seq)

    def _adopt(self, native):
        handle = PageHandle(
            native.pool_id, self._epoch(native.epoch), native.page_id, native.generation
        )
        self._require(
            handle.pool_id == self.pool_id
            and handle.epoch == self.epoch
            and 1 <= handle.page_id <= self.geometry.num_pages,
            "Take returned a foreign allocation",
        )
        previous = self._by_page.get(handle.page_id)
        self._require(
            previous not in self._owned, "Take reissued a logically owned page"
        )
        if previous is not None:
            self._require(
                handle.generation > previous.generation,
                "Take did not advance native generation",
            )
            self._native.pop(previous, None)
            self._forwards.pop(previous, None)
            self._records.pop(previous, None)
        self._by_page[handle.page_id], self._native[handle] = handle, native
        self._owned.add(handle)
        return handle

    def _live(self, handles):
        if len(set(handles)) != len(handles) or any(
            h not in self._owned
            or h in self._quarantined
            or self._by_page.get(h.page_id) != h
            for h in handles
        ):
            raise RuntimeError(
                "engine operation requires current live allocation handles"
            )

    def mem_pool(self, device):
        with self._lock:
            self._check(registered=False)
            if str(device) != self.device or self.device_index < 0:
                raise ValueError("engine MemPool requires its configured GPU device")
            if self._pool is None:
                self._pool = self.engine.kv_mem_pool(self.device)
                logger.info(
                    "UMBP engine MemPool active device=%s allocator=KvVmmAllocator",
                    self.device,
                )
            return self._pool

    def register_pool(self, geometry, request_regions):
        with self._lock:
            self._check(registered=False)
            if self.registration is not None:
                raise RuntimeError("engine pool already registered")
            geometry.validate()
            self.geometry, self.request_regions = geometry, tuple(request_regions)
            native = self.engine.PageGeometry()
            for name in ("page_tokens", "num_pages", "object_bytes", "layout_id"):
                setattr(native, name, getattr(geometry, name))
            # mori::io::MemoryLocationType: Unknown=0, CPU=1, GPU=2. Unknown made every
            # rank-side HBM<->DRAM copy plan unroutable (F3 rejected on all ranks).
            location = 2 if self.device_index >= 0 else 1
            native.regions = [
                self.engine.PageRegion(
                    r.region_id,
                    r.base,
                    r.length,
                    r.row_bytes,
                    r.rows_per_page,
                    r.logical_offset,
                    r.layer_id,
                    r.component,
                    location,
                    self.device_index,
                )
                for r in geometry.regions
            ]
            for region in (*geometry.regions, *self.request_regions):
                if self.device_index >= 0:
                    allocation = self.engine.lookup_allocation(
                        region.base, region.length
                    )
                    self._require(
                        allocation is not None
                        and allocation["device"] == self.device_index,
                        f"KV region {region.region_id} is not owned by the VMM MemPool",
                    )
                    exported = self.engine.export_allocation(region.base)
                    self._require(
                        exported is not None,
                        f"VMM export failed for {region.region_id}",
                    )
                    self.vmm_exports.append((region.region_id, allocation, exported))
                self._require(
                    self._call(
                        self.executor.register_original_region,
                        region.region_id,
                        region.base,
                        region.length,
                        location,
                        self.device_index,
                    ),
                    f"executor registration rejected {region.region_id}",
                )
            registration = self._call(
                self.client.register_pool,
                self.rank,
                self.pool_id,
                self.engine.PoolEpoch(uuid.uuid4().bytes),
                native,
            )
            self._require(
                registration.success,
                f"engine pool registration rejected: {registration.error}",
            )
            self.registration, self.epoch = (
                registration,
                self._epoch(registration.epoch),
            )
            regions = registration.dram_regions
            ids = {r.region_id for r in regions}
            self._require(
                sum(r.length for r in regions) >= self.dram_options.get("min_bytes", 0)
                and set(self.dram_options.get("region_ids", ())) <= ids,
                "node DRAM regions do not satisfy provider configuration",
            )
            self._call(self.executor.map_shared_dram, registration)
            self._require(
                not self.accounting_violations(),
                "initial delegation does not conserve pages",
            )
            logger.info(
                "UMBP engine pool registered pool=%s backend=%d pages=%d page_tokens=%d object_bytes=%d layout=%s dram_regions=%d dram_bytes=%d vmm_exports=%d",
                self.pool_id,
                self.client.backend_id,
                geometry.num_pages,
                geometry.page_tokens,
                geometry.object_bytes,
                geometry.layout_id,
                len(regions),
                sum(r.length for r in regions),
                len(self.vmm_exports),
            )
            return self.pool_id

    def create_linker(self, server_args, params, *, components, forward_stream):
        from sglang.srt.mem_cache.storage.umbp.umbp_engine_linker import (
            UMBPEngineLinker,
        )

        with self._lock:
            self._check()
            if self._linker is None:
                self._linker = UMBPEngineLinker(
                    self, params, components=components, forward_stream=forward_stream
                )
            return self._linker

    def get_load_stats(self):
        return self._linker.stats if self._linker is not None else {}

    def storage_key(self, key):
        """Shared by publish and the future linker; rank is not a key namespace."""
        return f"{self.namespace_seed}:{self.geometry.layout_id}:{key}"

    def reserve(self, shape):
        with self._lock:
            self._check()
            if min(shape.load_pages, shape.extend_pages) < 0:
                raise ValueError("negative capacity shape")
            if shape.load_pages + shape.extend_pages == 0:
                result = _Reservation(self, shape)
            else:
                native = self._call(
                    self.client.reserve,
                    f"{shape.request_id}:{shape.revision}",
                    shape.load_pages,
                    shape.extend_pages,
                )
                if not native.success:
                    self._require(
                        native.shortfall > 0,
                        "Reserve failed without a capacity shortfall",
                    )
                    return Short(native.shortfall)
                result = _Reservation(self, shape, native.id)
            self._reservations.add(result)
            return result

    def freeze_allocations(self, indices, *, context=None, page_ids=None):
        with self._lock:
            self._check()
            ids = (
                allocation_page_ids(indices, self.geometry.page_tokens)
                if page_ids is None
                else tuple(page_ids)
            )
            try:
                handles = tuple(
                    context.by_page[p]
                    if context is not None and p in context.by_page
                    else self._by_page[p]
                    for p in ids
                )
            except KeyError as exc:
                raise RuntimeError("freeze does not name an issued allocation") from exc
            self._live(handles)
            return _FrozenAllocations(self, handles, indices, ids)

    def record_forward(self, handles, done):
        with self._lock:
            self._check()
            self._live(handles)
            # All SGLang forward receipts use the same ordered forward stream.
            self._forwards.update((h, done) for h in handles)

    def publish(self, pages, *, ready):
        with self._lock:
            self._check()
            self._live(tuple(p.handle for p in pages))
            fresh = []
            for page in pages:
                old = self._records.get(page.handle)
                if old is not None:
                    if old.key != page.key or old.parent_key != page.parent_key:
                        raise RuntimeError(
                            "an immutable allocation cannot publish another key"
                        )
                    continue
                fresh.append(page)
            if fresh:
                self._queue_publication(tuple(fresh), ready)

    def _queue_publication(self, pages, ready, *, arm=True):
        records = []
        for page in pages:
            record = self.engine.ResidentRecord()
            record.key = self.storage_key(page.key)
            record.parent_key = (
                self.storage_key(page.parent_key) if page.parent_key is not None else ""
            )
            record.handle = self._native[page.handle]
            records.append(record)
        ticket = self._call(self.client.queue_publish, records, False)
        self._require(ticket != 0, "native publication queue rejected live allocations")
        task = _Publication(ticket, pages, ready, False)
        self._publications[ticket] = task
        self._publication_results[ticket] = pages
        self._record_tickets.update((p.handle, ticket) for p in pages)
        self._unready_publications[ticket] = task
        self._publication_by_handle.update((p.handle, ticket) for p in pages)
        self._records.update((p.handle, p) for p in pages)
        if arm:
            self._arm_publication(task)

    def _drop_publication(self, task, *, forget_records):
        """Drop cancel/readiness data; terminal correlation survives Sent."""
        self._publications.pop(task.ticket, None)
        self._unready_publications.pop(task.ticket, None)
        for page in task.records:
            if self._publication_by_handle.get(page.handle) == task.ticket:
                del self._publication_by_handle[page.handle]
                if (
                    forget_records
                    and self._record_tickets.get(page.handle) == task.ticket
                ):
                    self._records.pop(page.handle, None)
                    self._record_tickets.pop(page.handle, None)
        if forget_records:
            self._publication_results.pop(task.ticket, None)

    def _live_publication_pages(self, pages):
        return tuple(
            p
            for p in pages
            if p.handle in self._owned
            and p.handle not in self._quarantined
            and self._by_page.get(p.handle.page_id) == p.handle
        )

    def _ack_publish_receipt(self, ticket):
        if not self._call(self.client.acknowledge_publish_receipt, ticket):
            # A duplicate terminal event may arrive after its first ACK.
            self._require(
                self._call(self.client.publish_status, ticket) is None,
                "native terminal publication acknowledgement rejected",
            )

    def _check_event_overflow(self):
        # The shared flag also covers lost quarantine/retire notices. A
        # publication-status rescan alone cannot restore that ownership view.
        if self._call(self.client.consume_event_overflow):
            self._fail("native engine event queue overflow; ownership notices lost")

    def _drain_publish_receipts(self):
        states = self.engine.PublishReceiptState
        terminal = (states.Accepted, states.Rejected, states.Cancelled, states.Unknown)
        for receipt in self._call(self.client.poll_publish_receipts):
            pages = self._publication_results.get(receipt.ticket)
            if pages is None:
                if receipt.state in terminal:
                    self._ack_publish_receipt(receipt.ticket)
                continue  # duplicate, previous epoch, or locally canceled task
            state = receipt.state
            if state in (states.Queued, states.Ready):
                continue
            task = self._publications.get(receipt.ticket)
            if state == states.Sent:
                if task is not None:
                    self._drop_publication(task, forget_records=False)
                continue
            if state == states.Unknown:
                self._quarantined.update(
                    p.handle
                    for p in pages
                    if self._by_page.get(p.handle.page_id) == p.handle
                )
                self._retained.append(task if task is not None else pages)
                self._ack_publish_receipt(receipt.ticket)
                self._fail("native publication has Unknown completion")
            if state in (states.Accepted, states.Rejected):
                accepted = tuple(receipt.accepted)
                if state == states.Accepted and (
                    len(accepted) != len(pages) or not all(accepted)
                ):
                    self._fail(
                        "native Accepted receipt has inconsistent per-record results"
                    )
                for i, page in enumerate(pages):
                    if self._record_tickets.get(page.handle) != receipt.ticket:
                        continue
                    if state == states.Rejected and not (
                        i < len(accepted) and accepted[i]
                    ):
                        self._records.pop(page.handle, None)
                        self._record_tickets.pop(page.handle, None)
                if task is not None:
                    self._drop_publication(task, forget_records=False)
                self._publication_results.pop(receipt.ticket, None)
            elif state == states.Cancelled:
                if task is not None:
                    self._drop_publication(task, forget_records=True)
                    live = self._live_publication_pages(pages)
                    if live:
                        self._queue_publication(live, task.ready, arm=False)
                else:
                    for page in pages:
                        if self._record_tickets.get(page.handle) == receipt.ticket:
                            self._records.pop(page.handle, None)
                            self._record_tickets.pop(page.handle, None)
                    self._publication_results.pop(receipt.ticket, None)
            if state in terminal:
                self._ack_publish_receipt(receipt.ticket)

    def _drain_retire_events(self):
        # I10's background thread owns Ready completion and ACKs. This is a
        # local event drain, never a permission to reuse.
        for ticket in self._call(self.client.poll_retires):
            if ticket.state != self.engine.RetireState.Quarantined:
                continue
            for native in ticket.handles:
                handle = PageHandle(
                    native.pool_id,
                    self._epoch(native.epoch),
                    native.page_id,
                    native.generation,
                )
                current = self._by_page.get(handle.page_id)
                if handle.pool_id == self.pool_id and (
                    current is None or current == handle
                ):
                    self._quarantined.add(handle)
            self._retained.append(ticket)

    def _arm_publication(self, task):
        if task.armed:
            return
        try:
            ready = task.ready.query()
        except Exception as exc:
            self._quarantined.update(p.handle for p in task.records)
            self._retained.append(task)
            self._call(
                self.client.mark_publish_ready,
                task.ticket,
                self.engine.CompletionProof.Unknown,
            )
            self._fail(f"publication dependency has unknown completion: {exc}")
        if ready:
            marked_ready = self._call(
                self.client.mark_publish_ready,
                task.ticket,
                self.engine.CompletionProof.Completed,
            )
            if not marked_ready:
                # Reset may cancel an unsent native ticket. Its tree-owned
                # pages still need publication after Busy or abort. Requeue
                # once, without recursively spinning on a canceled ticket.
                self._drop_publication(task, forget_records=True)
                live = self._live_publication_pages(task.records)
                if live:
                    self._queue_publication(live, task.ready, arm=False)
                return
            task.armed = True
            self._unready_publications.pop(task.ticket, None)

    def _finish_guard(self, guard, handles, receipt):
        try:
            ready = receipt.query()
        except Exception as exc:
            self._quarantined.update(handles)
            self._retained.append((guard, handles, receipt))
            self._call(
                self.client.complete_reuse_guard,
                guard,
                self.engine.CompletionProof.Unknown,
            )
            self._fail(f"reuse guard has unknown completion: {exc}")
        if ready:
            self._require(
                self._call(
                    self.client.complete_reuse_guard,
                    guard,
                    self.engine.CompletionProof.Completed,
                ),
                "reuse guard completion rejected",
            )
        return ready

    def release_page_ids(self, page_ids, *, ready):
        with self._lock:
            self._check()
            try:
                handles = tuple(self._by_page[p] for p in page_ids)
            except KeyError as exc:
                raise RuntimeError("return does not name issued pages") from exc
            self.release_pages(handles, ready=ready)

    def release_pages(self, handles, *, ready):
        with self._lock:
            self._check()
            handles = tuple(handles)
            if len(set(handles)) != len(handles) or any(
                h not in self._owned for h in handles
            ):
                raise RuntimeError("duplicate/stale engine physical return")
            if not handles:
                return
            live = tuple(h for h in handles if h not in self._quarantined)
            dependency = JoinedReceipt(
                (ready, *(self._forwards.get(h, CompletedReceipt()) for h in live))
            )
            if live:
                guard = self._call(
                    self.client.hold_reuse_guard, [self._native[h] for h in live]
                )
                self._require(guard is not None, "reuse guard rejected")
                if not self._finish_guard(guard, live, dependency):
                    self._guards[guard.value] = (guard, live, dependency)
            returned = set(handles)
            tickets = dict.fromkeys(
                self._publication_by_handle[h]
                for h in handles
                if h in self._publication_by_handle
            )
            for ticket in tickets:
                task = self._publications[ticket]
                canceled = self._call(self.client.cancel_publish, ticket)
                canceled = canceled or not task.armed
                self._drop_publication(task, forget_records=canceled)
                # A native batch cancellation also covers untouched siblings.
                # Requeue them only after CancelPublish proves it never sent.
                kept = tuple(p for p in task.records if p.handle not in returned)
                if canceled and kept:
                    self._queue_publication(kept, task.ready)
            self._require(
                self._call(
                    self.client.return_pages, [self._native[h] for h in handles]
                ),
                "native page return/retirement rejected",
            )
            self._owned.difference_update(handles)
            for h in handles:
                self._records.pop(h, None)
                self._record_tickets.pop(h, None)
            # Unknown readers may still access returned pages. Native holds
            # protect them until every member of the original ticket is back.
            for ticket in tuple(self._io.values()):
                if ticket.parked and self._owned.isdisjoint(ticket.handles):
                    self._finish_io(ticket, self.engine.CompletionProof.Unknown)

    def hold_io(self, handles, *, kind, writes, receipt):
        with self._lock:
            self._check()
            handles = tuple(handles)
            self._live(handles)
            if writes and any(h in self._records for h in handles):
                raise RuntimeError("published allocations cannot be I/O write targets")
            native = self._call(self.client.hold_io, [self._native[h] for h in handles])
            self._require(native is not None, "native I/O hold rejected")
            ticket = _IOTicket(native, handles, receipt, kind, writes)
            self._io[id(ticket)] = ticket
            return ticket

    @property
    def has_pending_io(self):
        with self._lock:
            return bool(self._io)

    def _finish_io(self, ticket, proof):
        if proof == self.engine.CompletionProof.Unknown:
            self._quarantined.update(ticket.handles)
            self._retained.append(
                (ticket, tuple(self._forwards.get(h) for h in ticket.handles))
            )
        self._require(
            self._call(self.client.complete_io, ticket.native, proof),
            "native I/O completion rejected",
        )
        del self._io[id(ticket)]

    def complete_io(self, ticket, *, outcome):
        with self._lock:
            self._check()
            proof = {
                "completed": self.engine.CompletionProof.Completed,
                "never_submitted": self.engine.CompletionProof.NeverSubmitted,
                "unknown": self.engine.CompletionProof.Unknown,
            }.get(outcome)
            if proof is None:
                raise ValueError("invalid I/O completion proof")
            if self._io.get(id(ticket)) is not ticket or ticket.parked:
                raise RuntimeError("foreign or already completed I/O ticket")
            if outcome == "unknown" and not ticket.writes:
                # An unresolved read cannot corrupt a still-owned shared
                # prefix. Keep the existing native hold open to forbid reuse;
                # settle Unknown only at the final physical return.
                ticket.parked = True
                if not self._owned.isdisjoint(ticket.handles):
                    return
            self._finish_io(ticket, proof)

    def progress(self):
        with self._lock:
            self._check()
            self._check_event_overflow()
            for key, (guard, handles, ready) in list(self._guards.items()):
                if self._finish_guard(guard, handles, ready):
                    del self._guards[key]
            for task in tuple(self._unready_publications.values()):
                self._arm_publication(task)
            self._drain_publish_receipts()
            self._drain_retire_events()
            self._check_event_overflow()
            # I10 has no node Offer/Detach push or reply binding (the initial
            # proactive-clean window is zero). Local admission uses the real
            # tree's offer/confirm directly.
            return ()

    def complete_tree_event(self, event, result):
        self._fail("mori does not bind node offer/confirm requests and replies")

    def quiesce_reset(self, *, deadline_ns):
        with self._lock:
            self._check()
            self.progress()
            if time.monotonic_ns() >= deadline_ns:
                return Busy("reset deadline elapsed before sealing")
            native = self._call(self.client.quiesce_reset)
            if not native.ready:
                # EngineHbmBackend::BeginReset clears reset_sealed_ before
                # returning Busy; the ordered reset reply IS the cancel ACK.
                # A local Busy (id=0) never installed a node seal. FinishReset
                # on a Busy lease would be rejected: there is no active lease.
                return Busy("native pool has active holds/reservations")
            if time.monotonic_ns() >= deadline_ns:
                self._require(
                    self._call(self.client.finish_reset, native, False),
                    "late reset could not be canceled",
                )
                return Busy("reset deadline elapsed; node cancellation acknowledged")
            try:
                approved = tuple(h.page_id for h in native.approved)
                if not (
                    len(set(approved)) == len(approved)
                    and self._epoch(native.old_epoch) == self.epoch
                    and all(
                        h.pool_id == self.pool_id
                        and self._epoch(h.epoch) == self.epoch
                        and h.backend_id == self.client.backend_id
                        and 1 <= h.page_id <= self.geometry.num_pages
                        for h in native.approved
                    )
                ):
                    raise ValueError("invalid native reset approval set")
            except Exception as exc:
                self._require(
                    self._call(self.client.finish_reset, native, False),
                    "failed to cancel invalid reset approval",
                )
                self._fail(f"invalid native reset approval set: {exc}")
            # A quarantine notification can race the Ready snapshot. Approval
            # may shrink while sealed; never revive a known isolated page.
            isolated = {h.page_id for h in self._quarantined}
            self._lease = _ResetLease(
                native,
                self._epoch(native.old_epoch),
                self._epoch(native.new_epoch),
                tuple(p for p in approved if p not in isolated),
            )
            logger.info(
                "UMBP engine reset ready pool=%s epoch=%d approved_pages=%d quarantined=%d",
                self.pool_id,
                self.epoch.reset_seq,
                len(self._lease.approved_pages),
                len(isolated),
            )
            return self._lease

    def validate_reset(self, lease):
        self._check(sealed=True)
        if self._lease is not lease:
            raise RuntimeError("foreign/stale engine reset lease")
        self._check_event_overflow()
        self._drain_retire_events()
        self._check_event_overflow()
        isolated = {h.page_id for h in self._quarantined}
        lease.approved_pages = tuple(
            p for p in lease.approved_pages if p not in isolated
        )

    def commit_reset(self, lease):
        with self._lock:
            self.validate_reset(lease)
            self._require(
                self._call(self.client.finish_reset, lease.native, True),
                "reset commit rejected",
            )
            self.epoch = lease.new_epoch
            for reservation in self._reservations:
                reservation.closed = True
                reservation.parts.clear()
            self._owned.clear()
            self._publications.clear()
            self._publication_by_handle.clear()
            self._unready_publications.clear()
            self._records.clear()
            self._record_tickets.clear()
            self._publication_results.clear()
            self._forwards.clear()
            self._lease = None
            logger.info(
                "UMBP engine reset committed pool=%s epoch=%d approved_pages=%d",
                self.pool_id,
                self.epoch.reset_seq,
                len(lease.approved_pages),
            )

    def abort_reset(self, lease):
        with self._lock:
            self.validate_reset(lease)
            self._require(
                self._call(self.client.finish_reset, lease.native, False),
                "reset cancellation rejected",
            )
            self._lease = None

    def accounting(self):
        with self._lock:
            return self._accounting_snapshot()[0]

    def _accounting_snapshot(self):
        self._check(sealed=True)
        native = self._call(self.client.accounting)
        blocked = {
            h
            for ticket in self._io.values()
            for h in ticket.handles
            if h in self._records and h not in self._quarantined
        }
        account = PageAccounting(
            capacity=self.geometry.num_pages,
            **{
                k: getattr(native, k)
                for k in (
                    "cached",
                    "reserved",
                    "in_use",
                    "pending_free",
                    "publish_sent",
                    "published",
                    "retiring",
                    "quarantined",
                )
            },
            # RegisterPool delegates the entire fixed pool. A missing
            # native state is a leak, never an invented FREE allowance.
            free=0,
            blocked_reclaim_estimate=len(blocked),
        )
        return account, native.total

    def accounting_violations(self):
        with self._lock:
            account, total = self._accounting_snapshot()
            errors = account.violations()
            if total != self.geometry.num_pages:
                errors.append(
                    f"native page total {total} != registered capacity {self.geometry.num_pages}"
                )
            if (
                sum(
                    value
                    for name, value in vars(account).items()
                    if name not in ("capacity", "blocked_reclaim_estimate")
                )
                != total
            ):
                errors.append("native page state counts != native total")
            return errors

    def validate_projection(self, indices, *, page_ids=None):
        with self._lock:
            self._check()
            pages = (
                allocation_page_ids(indices, self.geometry.page_tokens)
                if page_ids is None
                else tuple(page_ids)
            )
            expected = {h.page_id for h in self._owned if h not in self._quarantined}
            optional = {h.page_id for h in self._quarantined}
            if (
                len(set(pages)) != len(pages)
                or not expected <= set(pages)
                or not set(pages) <= expected | optional
            ):
                return [
                    "engine provider request/tree projection differs from issued live pages"
                ]
            return []

    def close(self):
        with self._lock:
            if self._closed:
                return
            if self._linker is not None:
                self._linker.close()
            if (
                self._failed
                or (self.registration is not None and not self.client.healthy)
                or (
                    self.registration is not None
                    and self.client.accounting().quarantined
                )
                or self._io
                or self._guards
                or self._retained
                or any(not ready.query() for ready in self._forwards.values())
            ):
                self._fail(
                    "engine close cannot release active/unknown I/O or reuse-guard resources"
                )
            self.client.close()
            self.executor.close()
            for _, _, exported in self.vmm_exports:
                if exported.get("fd", -1) >= 0:
                    os.close(exported["fd"])
            self.vmm_exports.clear()
            self._closed = True
