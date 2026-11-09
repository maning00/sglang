"""Soft placement discovery; source ownership remains with the cache owner."""

from __future__ import annotations

import logging
import math
import random
import threading
import time
import zlib
from collections import Counter, OrderedDict, deque

import msgspec
import numpy as np

from .control import full_keys

logger = logging.getLogger(__name__)

# upstream sgl-kv-indexer/src/service.rs batch/query limits.
MAX_HASHES = 16384
MAX_ACTIONS = 256
# Bounded hints for waiting requests; authority is checked at ACQUIRE.
PLACEMENT_CAPACITY = 128
PLACEMENT_TTL_S = 0.1
_STATS_LOG_SECONDS = 60.0
_LATENCY_SAMPLES = 4096
_REPORT_RETRIES = 5
_REPORT_BACKOFF_MAX_S = 30.0
_PENDING_LIMIT = 65536


class Placement(msgspec.Struct):
    keys: tuple[str, ...]
    full: np.ndarray
    hashes: list[int]
    by_key: dict[str, int] | None = None
    matches: dict | None = None
    expires: float = 0.0
    eligible: bool = False


class Placements:
    def __init__(self, client):
        self.client = client
        self.entries = OrderedDict()

    def get(self, rid, keys, *, eligible=True):
        item = self.entries.pop(rid, None)
        if item is None or item.keys != tuple(keys):
            full = full_keys(keys)
            hashes = full.view(">i8").reshape(-1)[::4].tolist()
            item = Placement(tuple(keys), full, hashes)
        if item.eligible != eligible:
            item.expires = 0.0
            item.eligible = eligible
        if time.monotonic() >= item.expires:
            item.matches = self.client.lookup(item.hashes) if eligible else None
            item.expires = time.monotonic() + PLACEMENT_TTL_S
        self.entries[rid] = item
        if len(self.entries) > PLACEMENT_CAPACITY:
            self.entries.popitem(last=False)
        return item

    def pop(self, rid):
        self.entries.pop(rid, None)

    def clear(self):
        self.entries.clear()


class IndexerConfig(msgspec.Struct, frozen=True, kw_only=True):
    endpoint: str = ""
    deployment: str = ""
    timeout_s: float = 0.05
    cooldown_s: float = 0.1
    # Maximum sparse full-repair interval; independent phases use 80–100%.
    # Zero disables only this fallback, not event-driven recovery.
    recovery_s: float = 3600.0
    # Full-repair event/startup staggering; coalesced requests keep earliest due.
    recovery_jitter_s: float = 5.0

    def __post_init__(self):
        if not self.endpoint or not self.deployment:
            raise ValueError("indexer requires endpoint and deployment")
        if not all(
            math.isfinite(v) and v > 0 for v in (self.timeout_s, self.cooldown_s)
        ):
            raise ValueError("indexer timeouts must be finite and positive")
        if not all(
            math.isfinite(v) and v >= 0
            for v in (self.recovery_s, self.recovery_jitter_s)
        ):
            raise ValueError(
                "indexer recovery intervals must be finite and nonnegative"
            )


def block_hash(key):
    value = int(key[:16], 16)
    return value if value < 1 << 63 else value - (1 << 64)


def report_batches(records):
    from sglang.srt.grpc.kv_indexer_pb2 import ACTION_REPORT, TIER_HBM, ExternalKvAction

    actions, count = [], 0
    for key, parent in records:
        value = block_hash(key)
        parent = block_hash(parent) if parent is not None else None
        if count == MAX_HASHES or len(actions) == MAX_ACTIONS:
            yield actions
            actions, count = [], 0
        if actions and actions[-1].hashes[-1] == parent:
            actions[-1].hashes.append(value)
        else:
            action = ExternalKvAction(type=ACTION_REPORT, tier=TIER_HBM, hashes=[value])
            if parent is not None:
                action.parent_block_hash = parent
            actions.append(action)
        count += 1
    if actions:
        yield actions


class IndexerClient:
    def __init__(self, config):
        import grpc

        from sglang.srt.grpc import kv_indexer_pb2 as pb

        self.config, self.pb = config, pb
        self.channel = grpc.insecure_channel(config.endpoint)
        self._ready = threading.Event()
        self._connect_future = None
        self.apply = self.channel.unary_unary(
            "/kv_indexer.v1.KVIndexer/ApplyExternalKvBatch",
            request_serializer=pb.ApplyExternalKvBatchRequest.SerializeToString,
            response_deserializer=pb.ApplyExternalKvBatchResponse.FromString,
        )
        self.match = self.channel.unary_unary(
            "/kv_indexer.v1.KVIndexer/MatchExternalKv",
            request_serializer=pb.MatchExternalKvRequest.SerializeToString,
            response_deserializer=pb.MatchExternalKvResponse.FromString,
        )
        self._retry_after = 0.0
        self.rpc_count = 0
        self.stats = Counter(
            query_ok=0, query_timeout=0, query_error=0, query_cooldown_skip=0
        )
        # Percentiles cover the most recent samples since the last stats log;
        # retain the whole interval's count and max even if the ring fills.
        self._latencies_ms = deque(maxlen=_LATENCY_SAMPLES)
        self._latency_count = 0
        self._latency_max_ms = 0.0

    def lookup(self, hashes):
        import grpc

        self.stats["query_calls"] += 1
        self.stats["query_hashes"] += len(hashes)
        if time.monotonic() < self._retry_after:
            self.stats["query_cooldown_skip"] += 1
            self.stats["query_unavailable_hashes"] += len(hashes)
            return None
        result = {}
        try:
            for start in range(0, len(hashes), MAX_HASHES):
                self.rpc_count += 1
                started = time.monotonic()
                try:
                    reply = self.match(
                        self.pb.MatchExternalKvRequest(
                            hashes=hashes[start : start + MAX_HASHES],
                            count_as_hit=False,
                        ),
                        timeout=self.config.timeout_s,
                    )
                finally:
                    elapsed_ms = (time.monotonic() - started) * 1000
                    self._latencies_ms.append(elapsed_ms)
                    self._latency_count += 1
                    self._latency_max_ms = max(self._latency_max_ms, elapsed_ms)
                    self.stats["rpc_ms_total"] += elapsed_ms
                for match in reply.matches:
                    held = result.setdefault(match.worker_id, set())
                    for tier in match.hashes_by_tier:
                        if tier.tier == self.pb.TIER_HBM:
                            held.update(tier.hashes)
        except grpc.RpcError as exc:
            status = (
                "timeout"
                if exc.code() == grpc.StatusCode.DEADLINE_EXCEEDED
                else "error"
            )
            self.stats[f"query_{status}"] += 1
            self.stats["query_unavailable_hashes"] += len(hashes)
            self._retry_after = time.monotonic() + self.config.cooldown_s
            return None
        self.stats["query_ok"] += 1
        return result

    def stats_snapshot(self):
        """Cumulative counters; bounded RPC latency window reset by the caller."""
        values = sorted(self._latencies_ms)
        count = len(values)
        result = dict(self.stats)
        result.update(
            lookup_rpcs=self.rpc_count,
            rpc_latency_count=self._latency_count,
            rpc_latency_samples=count,
            rpc_ms_p50=values[(count - 1) // 2] if count else 0.0,
            rpc_ms_p99=values[(99 * count + 99) // 100 - 1] if count else 0.0,
            rpc_ms_max=self._latency_max_ms,
        )
        self._latencies_ms.clear()
        self._latency_count = 0
        self._latency_max_ms = 0.0
        return {f"indexer_{key}": value for key, value in result.items()}

    def on_reconnect(self, callback):
        import grpc

        seen_ready, was_ready = False, False
        lock = threading.Lock()

        def changed(state):
            nonlocal seen_ready, was_ready
            pending = None
            with lock:
                ready = state == grpc.ChannelConnectivity.READY
                if ready:
                    self._ready.set()
                    pending, self._connect_future = self._connect_future, None
                else:
                    self._ready.clear()
                reconnect = ready and seen_ready and not was_ready
                seen_ready = seen_ready or ready
                was_ready = ready
            if pending is not None:
                pending.cancel()
            if reconnect:
                callback()

        # Keep reconnection discovery active even with no pending publications.
        # First READY is handled by the publisher's initial snapshot, not twice.
        self.channel.subscribe(changed, try_to_connect=True)
        return changed

    def ensure_connected(self):
        if not self._ready.is_set():
            import grpc

            # The persistent watch can return to IDLE without an RPC. Keep one
            # readiness future per outage to request a connection, without a
            # new subscription/thread on every publisher tick. No MATCH is sent
            # and transport reconnect backoff is not reset.
            pending = self._connect_future
            if pending is None or pending.done():
                ready = grpc.channel_ready_future(self.channel)
                self._connect_future = ready
                if self._ready.is_set():
                    ready.cancel()
                    if self._connect_future is ready:
                        self._connect_future = None

    def close(self):
        if self._connect_future is not None:
            self._connect_future.cancel()
        self.channel.close()


class PlacementPublisher:
    def __init__(self, *, config, worker_id, client=None):
        self.config, self.worker_id = config, worker_id
        self._owns_client = client is None
        self.client = client if client is not None else IndexerClient(config)
        self._records = {}
        self._pending = {}
        self._pending_overflow = False
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._generation = 0
        self._completed_generation = -1
        self._seq = 0
        self.stats = Counter(recovery_started=0, recovery_ok=0, recovery_failed=0)
        self._recovery_since = None
        self._paused_until = 0.0
        self._failures = 0
        self._rng = random.Random()
        self._recover_at = time.monotonic() + self._rng.uniform(
            0, config.recovery_jitter_s
        )
        self._fallback_at = None
        self._recovery_reasons = Counter(startup=1)
        self._next_stats_log = time.monotonic() + _STATS_LOG_SECONDS
        self._subscription = self.client.on_reconnect(self._on_reconnect)
        self._thread = threading.Thread(
            target=self._run, name="kv-placement", daemon=True
        )
        self._thread.start()

    def recover(self, reason="manual"):
        with self._lock:
            self._recover_locked(reason, time.monotonic())
        self._wake.set()

    def _recover_locked(self, reason, now, *, jitter=True):
        self._generation += 1
        self.stats[f"recovery_requested_{reason}"] += 1
        self._recovery_reasons[reason] += 1
        due = now + (
            self._rng.uniform(0, self.config.recovery_jitter_s) if jitter else 0
        )
        # Coalesce requests without postponing an already scheduled repair.
        self._recover_at = (
            due if self._recover_at is None else min(self._recover_at, due)
        )

    def _fallback_locked(self, now):
        # Bounded, independent worker phases: no silent-state hole waits longer
        # than recovery_s before the next full repair starts (when reachable).
        self._fallback_at = (
            now + self.config.recovery_s * self._rng.uniform(0.8, 1.0)
            if self.config.recovery_s
            else None
        )

    def _on_reconnect(self):
        if self._stop.is_set():
            return
        with self._lock:
            self.stats["reconnects"] += 1
            self._recover_locked("reconnect", time.monotonic())
        self._wake.set()

    def _queue(self, key, value):
        # Called under _lock. Keep current ownership in _records regardless of
        # publication outages; coalesce deltas and bound obsolete-key buildup.
        if self._pending_overflow:
            return
        self._pending[key] = value
        if len(self._pending) > _PENDING_LIMIT:
            self._pending.clear()
            self._pending_overflow = True
            self.stats["pending_overflows"] += 1
            self._recover_locked("pending_overflow", time.monotonic())

    def publish(self, records):
        with self._lock:
            for key, parent in records:
                self._records[key] = parent
                self._queue(key, (True, parent))
        self._wake.set()

    def retire(self, keys):
        with self._lock:
            for key in keys:
                self._records.pop(key, None)
                self._queue(key, (False, None))
        self._wake.set()

    def _send(self, actions):
        import grpc

        self._seq += 1
        request = self.client.pb.ApplyExternalKvBatchRequest(
            worker_id=self.worker_id, seq=self._seq, worker_address="", actions=actions
        )
        phase = zlib.crc32(self.worker_id.encode()) / (1 << 32)
        for attempt in range(_REPORT_RETRIES + 1):
            started = time.monotonic()
            try:
                self.client.apply(request, timeout=max(1.0, self.config.timeout_s))
                return
            except grpc.RpcError as exc:
                if attempt == _REPORT_RETRIES or exc.code() not in (
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    grpc.StatusCode.UNAVAILABLE,
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                ):
                    raise
            finally:
                elapsed_ms = (time.monotonic() - started) * 1000
                with self._lock:
                    self.stats["report_rpcs"] += 1
                    self.stats["report_ms_total"] += elapsed_ms
                    self.stats["report_ms_max"] = max(
                        self.stats["report_ms_max"], elapsed_ms
                    )
            delay = min(_REPORT_BACKOFF_MAX_S, self.config.cooldown_s * 2**attempt) * (
                0.75 + 0.25 * phase
            )
            with self._lock:
                self.stats["report_retries"] += 1
                self.stats["report_backoff_ms"] += delay * 1000
            # Retain this exact batch, including seq, until acknowledged. Never
            # replay an old REPORT after a newer REVOKE or restart a snapshot.
            if self._stop.wait(delay):
                raise RuntimeError("placement publisher stopped during retry")

    def _flush(self):
        with self._lock:
            generation = self._generation
            recover = generation != self._completed_generation
            if self._pending_overflow and not recover:
                return
            records = list(self._records.items()) if recover else None
            pending, self._pending = self._pending, {}
            if recover:
                self._pending_overflow = False
                self._recover_at = None
                reasons, self._recovery_reasons = self._recovery_reasons, Counter()
        if recover:
            started = time.monotonic()
            first_seq = self._seq + 1
            logger.info(
                "KV placement recovery start worker=%s generation=%d reasons=%s records=%d seq=%d",
                self.worker_id,
                generation,
                dict(reasons),
                len(records),
                first_seq,
            )
            with self._lock:
                self.stats["recovery_started"] += 1
                self.stats["recovery_records"] += len(records)
                if self._recovery_since is None:
                    self._recovery_since = started
            succeeded = False
            try:
                self._send(
                    [
                        self.client.pb.ExternalKvAction(
                            type=self.client.pb.ACTION_CLEAR_ALL_AT_TIER,
                            tier=self.client.pb.TIER_HBM,
                        )
                    ]
                )
                for batch in report_batches(records):
                    self._send(batch)
                succeeded = True
            finally:
                now = time.monotonic()
                elapsed_ms = (now - started) * 1000
                with self._lock:
                    self.stats["recovery_ok" if succeeded else "recovery_failed"] += 1
                    self.stats["recovery_ms_total"] += elapsed_ms
                    self.stats["recovery_ms_max"] = max(
                        self.stats["recovery_ms_max"], elapsed_ms
                    )
                    if succeeded:
                        # Upper bound on the CLEAR/snapshot visibility gap,
                        # including failed attempts and their retry cooldown.
                        gap_ms = (now - self._recovery_since) * 1000
                        self.stats["recovery_gap_ms_total"] += gap_ms
                        self.stats["recovery_gap_ms_max"] = max(
                            self.stats["recovery_gap_ms_max"], gap_ms
                        )
                        self._recovery_since = None
                        self._fallback_locked(now)
                logger.info(
                    "KV placement recovery done worker=%s generation=%d ok=%s seq_first=%d seq_last=%d ms=%.3f",
                    self.worker_id,
                    generation,
                    succeeded,
                    first_seq,
                    self._seq,
                    elapsed_ms,
                )
        else:
            removed = [block_hash(k) for k, (live, _) in pending.items() if not live]
            for start in range(0, len(removed), MAX_HASHES):
                self._send(
                    [
                        self.client.pb.ExternalKvAction(
                            type=self.client.pb.ACTION_REVOKE,
                            tier=self.client.pb.TIER_HBM,
                            hashes=removed[start : start + MAX_HASHES],
                        )
                    ]
                )
            records = [(k, parent) for k, (live, parent) in pending.items() if live]
            for batch in report_batches(records):
                self._send(batch)
        with self._lock:
            self._completed_generation = generation

    def stats_snapshot(self):
        with self._lock:
            result = dict(self.stats)
            result.update(
                live_records=len(self._records),
                pending_records=len(self._pending),
                pending_overflow=int(self._pending_overflow),
                report_pause_remaining_ms=max(
                    0.0, self._paused_until - time.monotonic()
                )
                * 1000,
                recovery_wait_remaining_ms=max(
                    0.0,
                    max(self._recover_at or 0.0, self._paused_until) - time.monotonic(),
                )
                * 1000,
                fallback_remaining_ms=(
                    max(0.0, self._fallback_at - time.monotonic()) * 1000
                    if self._fallback_at is not None
                    else None
                ),
                recovery_active_ms=(time.monotonic() - self._recovery_since) * 1000
                if self._recovery_since is not None
                else 0.0,
            )
        return result

    def _maybe_log_stats(self):
        now = time.monotonic()
        if now >= self._next_stats_log:
            self._next_stats_log = now + _STATS_LOG_SECONDS
            logger.info(
                "KV placement publisher worker=%s stats: %s",
                self.worker_id,
                self.stats_snapshot(),
            )

    def _ready_delay(self, now):
        with self._lock:
            if self._fallback_at is not None and now >= self._fallback_at:
                # An outstanding full repair already covers the fallback.
                if self._generation == self._completed_generation:
                    self._recover_locked("periodic", now, jitter=False)
                self._fallback_at = None
            due = self._paused_until
            if self._generation != self._completed_generation:
                due = max(due, self._recover_at or now)
            return max(0.0, due - now)

    def _report_failed(self, error):
        now = time.monotonic()
        with self._lock:
            self._failures += 1
            cap = min(
                _REPORT_BACKOFF_MAX_S,
                self.config.cooldown_s * 2 ** min(self._failures - 1, 30),
            )
            delay = cap * self._rng.uniform(0.5, 1.0)
            self._paused_until = now + delay
            self.stats["report_pauses"] += 1
            self.stats["recovery_retry_backoff_ms"] += delay * 1000
            self._recover_locked("report_failure", now)
        logger.warning(
            "KV placement report failed worker=%s; full recovery after backoff %.3fs: %s",
            self.worker_id,
            delay,
            error,
        )

    def _run(self):
        while not self._stop.is_set():
            self._wake.clear()
            ensure_connected = getattr(self.client, "ensure_connected", None)
            if ensure_connected is not None:
                ensure_connected()
            now = time.monotonic()
            delay = self._ready_delay(now)
            if delay:
                self._maybe_log_stats()
                self._wake.wait(min(1.0, delay))
                continue
            try:
                self._flush()
            except Exception as error:
                if self._stop.is_set():
                    break
                self._report_failed(error)
            else:
                with self._lock:
                    self._failures = 0
                    self._paused_until = 0.0
            finally:
                self._maybe_log_stats()
            with self._lock:
                due = [
                    x for x in (self._recover_at, self._fallback_at) if x is not None
                ]
            self._wake.wait(
                min(1.0, max(0.0, min(due) - time.monotonic())) if due else 1.0
            )

    def close(self):
        self._stop.set()
        self._wake.set()
        self._thread.join()
        self.client.channel.unsubscribe(self._subscription)
        if self._owns_client:
            self.client.close()
