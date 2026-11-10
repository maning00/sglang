"""Layer-wise UMBP reads through the pool provider's existing native session."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import InvalidStateError
from dataclasses import dataclass, field

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.kv_memory_provider import (
    CompletedReceipt,
    EventReceipt,
    JoinedReceipt,
    PoolEpoch,
    ProviderFailStop,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    LayerWiseLoadCounter,
    UnifiedCacheLinker,
)

logger = logging.getLogger(__name__)


class _EngineLoadCounter(LayerWiseLoadCounter):
    def complete(self, index, layer):
        try:
            super().complete(index, layer)
        except InvalidStateError:
            # Future's atomic transition decides a race with fail-stop. A
            # later completion must not overwrite the terminal result.
            pass

    def fail(self, index, error):
        for future in self.futures.get(index, ()):
            try:
                future.set_exception(error)
            except InvalidStateError:
                pass  # a physical completion or another failure won

    def wait_until(self, threshold):
        futures = self.futures.get(self.consumer_index)
        if futures is None:
            return
        error = futures[threshold].exception()
        if isinstance(error, ProviderFailStop):
            # A lost control channel or unprovable submit still stops forward.
            # Native terminal Unknown uses a normal batch failure instead.
            raise error
        if self.on_layer_ready is not None:
            self.on_layer_ready(self.consumer_index, threshold)


@dataclass
class _LookupPlan:
    keys: tuple[str, ...]
    native: object
    boundaries: frozenset[int]
    created_at: float
    epoch: PoolEpoch


@dataclass
class _Load:
    rid: str
    prepared: object
    handles: tuple
    indices: object
    ready: object
    layer_ids: tuple[int, ...]
    sources: dict[str, int]
    native: object = None
    submitting: bool = False
    batch_index: int | None = None
    submission_error: Exception | None = None
    copy_cursor: int = 0
    layer_failed: bool = False
    terminal: bool = False
    outcome: object = None
    isolated: bool = False


@dataclass
class _Batch:
    index: int
    jobs: tuple[_Load, ...]
    layer_copies: tuple | None = None
    remaining: list[int] = field(default_factory=list)
    next_layer: int = 0
    done: threading.Event = field(default_factory=threading.Event)
    landed: bool = False
    fatal: ProviderFailStop | None = None


class UMBPEngineLinker(UnifiedCacheLinker):
    # TODO(mori I11): enable only after every nonzero layer_event is recorded
    # before SubmitLoad returns. CPU layer_done polling remains the default.
    use_layer_events = False
    lookup_plan_ttl_s = 2.0
    load_plan_ttl_s = 5.0
    _max_load_pages = 256
    _poll_interval_s = 0.002
    _copies_per_turn = 16

    def __init__(self, provider, params, *, components, forward_stream=None):
        if self.use_layer_events:
            raise ValueError("layer_event stream waits require the mori I11 contract")
        for owner, names in (
            (provider.client, ("lookup", "prepare_load", "cancel_load")),
            (provider.executor, ("submit_load", "poll_loads")),
        ):
            if any(not callable(getattr(owner, name, None)) for name in names):
                raise ValueError("UMBP engine linker requires the mori I9 read API")
        self.provider, self.client, self.executor = (
            provider,
            provider.client,
            provider.executor,
        )
        self.forward_stream = forward_stream
        self.page_size = provider.geometry.page_tokens
        pool = params.token_to_kv_pool_allocator.get_kvcache()
        self.pools = {PoolName.KV: pool}
        self.start_layer = getattr(pool, "start_layer", 0)
        # The configured stage includes layers with no FULL bytes. Region
        # maxima cannot describe the model's layer-wait counter.
        self.num_layers = getattr(pool, "layer_num", None)
        if not isinstance(self.num_layers, int) or self.num_layers <= 0:
            raise ValueError("UMBP engine linker requires configured pool layer_num")
        if any(
            not 0 <= r.layer_id - self.start_layer < self.num_layers
            for r in provider.geometry.regions
        ):
            raise ValueError("UMBP engine geometry has invalid logical layers")
        self.layer_done_counter = _EngineLoadCounter(self.num_layers)
        self._plans, self._pending, self._batches = {}, {}, {}
        self._active_rids = set()
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._closed = self._stopping = False
        self._stats = dict(
            lookups=0,
            lookup_hit_pages=0,
            prepare_rejected=0,
            submitted=0,
            completed=0,
            failed=0,
            unknown=0,
            canceled=0,
        )
        self._thread = threading.Thread(
            target=self._run, name="umbp-layer-completions", daemon=True
        )
        self._thread.start()
        logger.info(
            "UMBP engine linker active pool=%s layers=%d layer_readiness=cpu",
            provider.pool_id,
            self.num_layers,
        )

    @property
    def stats(self):
        with self._lock:
            return dict(self._stats)

    def _check(self):
        if self._closed or self._stopping:
            raise RuntimeError("UMBP engine linker closed")
        self.provider._check()

    @staticmethod
    def _kv(transfers):
        if len(transfers) != 1 or transfers[0].name != PoolName.KV:
            raise ValueError("engine linker transfers one whole FULL object per page")
        return transfers[0]

    def lookup(self, rid, transfers):
        self._check()
        transfer = self._kv(transfers)
        keys = tuple(self.provider.storage_key(key) for key in transfer.keys or ())
        with self._lock:
            self._expire_plans()
            cached = self._plans.get(rid)
            if (
                cached is not None
                and cached.keys == keys
                and time.monotonic() - cached.created_at < self.lookup_plan_ttl_s
                and cached.epoch == self.provider.epoch
            ):
                return sorted(cached.boundaries)
            self._plans.pop(rid, None)
        if not keys:
            return []
        plan = self.provider._call(
            self.client.lookup, list(keys), self.provider.namespace_seed
        )
        with self._lock:
            self._stats["lookups"] += 1
        if plan.success and (
            not (0 <= plan.valid_prefix <= len(keys))
            or tuple(plan.keys) != keys
            or plan.layout_id != self.provider.geometry.layout_id
            or self.provider._epoch(plan.epoch) != self.provider.epoch
            or plan.namespace != self.provider.namespace_seed
        ):
            self.provider._fail(
                "native lookup returned a foreign/inconsistent read plan"
            )
        boundaries = sorted(
            {
                int(n)
                for n in plan.legal_boundaries
                if 0 < n <= min(plan.valid_prefix, self._max_load_pages)
            }
            if plan.success
            else ()
        )
        with self._changed:
            # Preserve creation order for amortized O(expired) cleanup. Cache
            # misses too: LPM can rematch every waiting request each round.
            self._plans.pop(rid, None)
            self._plans[rid] = _LookupPlan(
                keys,
                plan if plan.success else None,
                frozenset(boundaries),
                time.monotonic(),
                self.provider.epoch,
            )
            if boundaries:
                self._stats["lookup_hit_pages"] += boundaries[-1]
            self._changed.notify_all()
        logger.info(
            "UMBP engine lookup rid=%s requested_pages=%d valid_prefix=%d legal_boundaries=%s",
            rid,
            len(keys),
            plan.valid_prefix,
            boundaries,
        )
        # The existing wrapper intersects these exact sets across ranks.
        return boundaries

    def _expire_plans(self):
        cutoff = time.monotonic() - self.load_plan_ttl_s
        while self._plans:
            rid = next(iter(self._plans))
            if self._plans[rid].created_at > cutoff:
                break
            del self._plans[rid]

    def load(self, rid, transfers):
        self._check()
        transfer = self._kv(transfers)
        with self._lock:
            if rid in self._pending or rid in self._active_rids:
                raise RuntimeError("engine load already queued for this request")
            hit = self._plans.pop(rid, None)
        keys = tuple(self.provider.storage_key(key) for key in transfer.keys or ())
        if (
            hit is None
            or time.monotonic() - hit.created_at >= self.load_plan_ttl_s
            or hit.epoch != self.provider.epoch
            or len(keys) not in hit.boundaries
            or keys != hit.keys[: len(keys)]
        ):
            return False
        indices = transfer.device_indices
        context = self.provider.freeze_allocations(indices)
        handles = context.handles()
        if len(handles) != len(keys) or indices.numel() != len(keys) * self.page_size:
            raise ValueError("load targets are not whole reserved FULL pages")
        prepared = self.provider._call(
            self.client.prepare_load,
            hit.native,
            rid,
            [self.provider._native[h] for h in handles],
            len(handles),
        )
        if not prepared.success:
            # Includes grant failure and I11 size/fd-limit rejection. Native
            # PrepareLoad unwinds its holds; wrapper ABORT frees taken targets.
            with self._lock:
                self._stats["prepare_rejected"] += 1
            logger.info(
                "UMBP engine load rejected rid=%s pages=%d reason=%s",
                rid,
                len(handles),
                prepared.error,
            )
            return False
        try:
            layer_ids = tuple(int(layer) for layer in prepared.layer_ids)
            if (
                prepared.layer_count != len(layer_ids)
                or not layer_ids
                or any(
                    not 0 <= layer - self.start_layer < self.num_layers
                    for layer in layer_ids
                )
            ):
                raise ValueError("native prepared load has invalid layer geometry")
            ready = CompletedReceipt()
            if self.provider.device_index >= 0:
                ready = JoinedReceipt(
                    tuple(
                        EventReceipt(event)
                        for event in self.record_transfer_ready().events
                    )
                )
            sources = {}
            for source in hit.native.sources[: len(handles)]:
                kind = getattr(source.kind, "name", str(source.kind))
                sources[kind] = sources.get(kind, 0) + 1
        except Exception:
            result = self.provider._call(self.client.cancel_load, prepared)
            if result != self.provider.engine.CancelLoadResult.NeverSubmitted:
                raise self._fatal(
                    None,
                    ProviderFailStop(
                        "prepared load setup could not prove cancellation"
                    ),
                    (prepared, handles, indices),
                )
            raise
        job = _Load(rid, prepared, handles, indices, ready, layer_ids, sources)
        with self._lock:
            self._pending[rid] = job
        return True

    def cancel_queued_load(self, rid):
        with self._lock:
            self._plans.pop(rid, None)
            if rid in self._active_rids:
                return False  # batch owns it; native holds protect its targets
            job = self._pending.get(rid)
            if job is None:
                return True
            result = self.provider._call(self.client.cancel_load, job.prepared)
            if result != self.provider.engine.CancelLoadResult.NeverSubmitted:
                return False
            del self._pending[rid]
            self._stats["canceled"] += 1
            return True

    def start_layer_wise_loading(self):
        with self._changed:
            self._check()
            if not self._pending:
                return -1
            index = self.layer_done_counter.update_producer()
            jobs = tuple(self._pending.values())
            self._pending.clear()
            for job in jobs:
                job.batch_index = index
                self._active_rids.add(job.rid)
            self._batches[index] = _Batch(index, jobs)
            self._changed.notify_all()
            return index

    def _fatal(self, batch, error, resources=()):
        with self._changed:
            self.provider._retained.append(batch if batch is not None else resources)
            self.provider._failed = str(error)
            if batch is not None:
                if batch.fatal is None:
                    self._stats["unknown"] += 1
                batch.fatal = error
                batch.done.set()
                self.layer_done_counter.fail(batch.index, error)
            self._changed.notify_all()
        return error

    def _submit(self, batch, job):
        job.submitting = True
        try:
            # A throwing executor call is classified by CancelLoad's physical
            # proof below, rather than prematurely poisoning a healthy client.
            job.native = self.executor.submit_load(job.prepared)
            if not self.client.healthy:
                raise ProviderFailStop("engine control channel failed during submit")
            if job.native is None or job.native.layer_count != len(job.layer_ids):
                raise ProviderFailStop(
                    "native load submission has no valid completion handle"
                )
            with self._lock:
                self._stats["submitted"] += 1
        except Exception as exc:
            # A throwing submit may have posted work. Only native cancellation
            # proving NeverSubmitted can turn this into an ordinary failure.
            try:
                result = self.provider._call(self.client.cancel_load, job.prepared)
            except Exception:
                result = None
            if result == self.provider.engine.CancelLoadResult.NeverSubmitted:
                job.submission_error = exc
                return
            error = ProviderFailStop(f"load submission cannot prove quiescence: {exc}")
            self._fatal(batch, error)

    def _advance(self, batch):
        if batch.done.is_set():
            return False
        outcomes = self.provider.engine.LoadOutcome
        progressed = False
        for job in batch.jobs:
            if job.native is None and not job.submitting:
                if not job.ready.query():
                    continue
                self._submit(batch, job)
                progressed = True
                if batch.fatal:
                    return progressed
        if any(
            job.native is None and job.submission_error is None for job in batch.jobs
        ):
            return progressed
        if batch.layer_copies is None:
            copies = [[] for _ in range(self.num_layers)]
            for job in batch.jobs:
                if job.native is not None:
                    for i, layer in enumerate(job.layer_ids):
                        copies[layer - self.start_layer].append((job.native, i))
            batch.layer_copies = tuple(tuple(layer) for layer in copies)
            batch.remaining = [len(layer) for layer in batch.layer_copies]
        for job in batch.jobs:
            if job.submission_error is not None:
                continue
            if not job.terminal:
                job.outcome = job.native.outcome
                job.terminal = job.outcome != outcomes.Pending
                if job.terminal:
                    progressed = True
            if job.terminal and job.outcome != outcomes.Completed:
                if job.outcome == outcomes.Unknown and not job.isolated:
                    # FinishLoad(Unknown) has isolated these native targets
                    # before publishing its terminal outcome. Keep resources
                    # alive, fail this batch, and leave healthy requests usable.
                    with self.provider._lock:
                        self.provider._quarantined.update(job.handles)
                        self.provider._retained.append(job)
                    job.isolated = True
                    with self._lock:
                        self._stats["unknown"] += 1
                continue
            if job.layer_failed:
                if job.terminal:
                    raise ProviderFailStop(
                        "native load completed after a failed layer receipt"
                    )
                continue
            # Native executes in index order. Never probe a later copy while
            # this cursor is pending, or query an already consumed receipt.
            # Bound each request's turn, then yield the GIL before more work.
            for _ in range(self._copies_per_turn):
                i = job.copy_cursor
                if i == len(job.layer_ids):
                    break
                if not job.native.layer_done(i):
                    if job.terminal:
                        raise ProviderFailStop(
                            "native load completed without physical layer receipts"
                        )
                    break
                if job.native.layer_outcome(i) != outcomes.Completed:
                    job.layer_failed = True
                    if job.terminal:
                        raise ProviderFailStop(
                            "native load completed after a failed layer receipt"
                        )
                    break
                job.copy_cursor += 1
                batch.remaining[job.layer_ids[i] - self.start_layer] -= 1
                progressed = True
        terminal = all(
            job.submission_error is not None
            or (
                job.terminal
                and (
                    job.outcome != outcomes.Completed
                    or job.copy_cursor == len(job.layer_ids)
                )
            )
            for job in batch.jobs
        )
        landed = terminal and all(
            job.submission_error is None and job.outcome == outcomes.Completed
            for job in batch.jobs
        )
        if terminal:
            # Publish batch terminal state before waking the last forward layer.
            with self._lock:
                batch.landed = landed
                batch.done.set()
                self._stats["completed" if landed else "failed"] += len(batch.jobs)
                if landed:
                    for job in batch.jobs:
                        for kind, count in job.sources.items():
                            key = f"completed_{kind}_pages"
                            self._stats[key] = self._stats.get(key, 0) + count
                for job in batch.jobs:
                    logger.info(
                        "UMBP engine load %s rid=%s pages=%d layer_copies=%d plan_sources=%s",
                        "completed" if landed else "failed",
                        job.rid,
                        len(job.handles),
                        len(job.layer_ids),
                        job.sources,
                    )
            if not landed:
                self.layer_done_counter.fail(
                    batch.index, RuntimeError("UMBP batch load failed after submit")
                )
                return True
        if not any(
            job.submission_error is not None
            or job.layer_failed
            or (job.terminal and job.outcome != outcomes.Completed)
            for job in batch.jobs
        ):
            while batch.next_layer < self.num_layers:
                layer = batch.next_layer
                if batch.remaining[layer] or (
                    layer == self.num_layers - 1 and not terminal
                ):
                    break
                self.layer_done_counter.complete(batch.index, layer)
                batch.next_layer += 1
                progressed = True
        return progressed

    def _run(self):
        while True:
            with self._changed:
                # Expire orphaned plans even when no batch/request ever loads
                # them (zero rank intersection or a successful request retire).
                while not self._stopping:
                    self._expire_plans()
                    if any(not b.done.is_set() for b in self._batches.values()):
                        break
                    first = next(iter(self._plans.values()), None)
                    timeout = (
                        max(
                            0,
                            first.created_at + self.load_plan_ttl_s - time.monotonic(),
                        )
                        if first is not None
                        else None
                    )
                    self._changed.wait(timeout=timeout)
                if self._stopping:
                    return
                batches = tuple(
                    b for b in self._batches.values() if not b.done.is_set()
                )
            progressed = False
            for batch in batches:
                try:
                    if self.provider._failed:
                        self._fatal(batch, ProviderFailStop(self.provider._failed))
                        continue
                    progressed |= self._advance(batch)
                except Exception as exc:
                    self._fatal(
                        batch, ProviderFailStop(f"UMBP load progress failed: {exc}")
                    )
            try:
                self.provider._call(self.executor.poll_loads)
            except Exception as exc:
                for batch in batches:
                    if not batch.done.is_set():
                        self._fatal(
                            batch,
                            ProviderFailStop(
                                f"UMBP load completion channel failed: {exc}"
                            ),
                        )
            if progressed:
                # Drain already-complete copies promptly, with a GIL handoff
                # between bounded turns rather than a sleep for every layer.
                time.sleep(0)
            else:
                with self._changed:
                    self._changed.wait(timeout=self._poll_interval_s)

    def finish_layer_wise_loading(self, counter_index):
        if counter_index < 0:
            return True
        with self._lock:
            batch = self._batches[counter_index]
            if batch.fatal is not None:
                raise batch.fatal
            if not batch.done.is_set() or any(
                not f.done() for f in self.layer_done_counter.futures[counter_index]
            ):
                error = ProviderFailStop(
                    "forward finished before all UMBP load layers quiesced"
                )
                raise self._fatal(batch, error)
            landed = self.layer_done_counter.finish(counter_index)
            del self._batches[counter_index]
            self._active_rids.difference_update(job.rid for job in batch.jobs)
            return landed and batch.landed

    def offload(self, transfers, session_id=None):
        raise RuntimeError(
            "engine KV publication uses residency actions, not linker offload"
        )

    def num_completed_offloads(self):
        return 0

    def pop_completed_offload(self):
        raise RuntimeError("engine linker has no offload queue")

    def reset(self):
        with self._lock:
            if self._batches:
                raise ProviderFailStop(
                    "cannot reset an engine linker with active batches"
                )
            for rid in tuple(self._pending):
                if not self.cancel_queued_load(rid):
                    raise ProviderFailStop("queued engine load could not be canceled")
            self._plans.clear()
            self.layer_done_counter.reset()

    def close(self):
        with self._changed:
            if self._closed:
                return
            if self._batches:
                raise ProviderFailStop(
                    "finish engine load batches before closing the linker"
                )
            self.reset()
            self._stopping = True
            self._changed.notify_all()
        self._thread.join(timeout=self.provider.reset_timeout_ms / 1000)
        if self._thread.is_alive():
            raise ProviderFailStop("engine linker completion thread did not stop")
        self._closed = True
        # Provider owns the shared client/executor/MemPool; do not close them.
