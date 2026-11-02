"""Soft placement discovery; source ownership remains with the cache owner."""

from __future__ import annotations

import logging
import threading
import time
import zlib
from collections import OrderedDict

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


class Placement(msgspec.Struct):
    keys: tuple[str, ...]
    full: np.ndarray
    hashes: np.ndarray
    by_key: dict[str, int]
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
            hashes = full.view(">i8").reshape(-1)[::4]
            item = Placement(tuple(keys), full, hashes, dict(zip(keys, hashes)))
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
    stage: int = 0
    endpoint: str = ""
    deployment: str = ""
    timeout_s: float = 0.01
    cooldown_s: float = 1.0
    recovery_s: float = 600.0

    def __post_init__(self):
        if not 0 <= self.stage <= 4:
            raise ValueError("indexer stage must be between 0 and 4")
        if self.stage and (not self.endpoint or not self.deployment):
            raise ValueError("indexer requires endpoint and deployment")
        if min(self.timeout_s, self.cooldown_s, self.recovery_s) <= 0:
            raise ValueError("indexer timeouts must be positive")


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

    def lookup(self, hashes):
        import grpc

        if time.monotonic() < self._retry_after:
            return None
        result = {}
        try:
            for start in range(0, len(hashes), MAX_HASHES):
                self.rpc_count += 1
                reply = self.match(
                    self.pb.MatchExternalKvRequest(
                        hashes=hashes[start : start + MAX_HASHES],
                        count_as_hit=False,
                    ),
                    timeout=self.config.timeout_s,
                )
                for match in reply.matches:
                    held = result.setdefault(match.worker_id, set())
                    for tier in match.hashes_by_tier:
                        if tier.tier == self.pb.TIER_HBM:
                            held.update(tier.hashes)
        except grpc.RpcError:
            self._retry_after = time.monotonic() + self.config.cooldown_s
            return None
        return result

    def on_reconnect(self, callback):
        import grpc

        def changed(state):
            if state == grpc.ChannelConnectivity.READY:
                callback()

        self.channel.subscribe(changed)
        return changed

    def close(self):
        self.channel.close()


class PlacementPublisher:
    def __init__(self, *, config, worker_id, client=None):
        self.config, self.worker_id = config, worker_id
        self._owns_client = client is None
        self.client = client if client is not None else IndexerClient(config)
        self._records = {}
        self._pending = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._generation = 0
        self._completed_generation = -1
        self._seq = 0
        self._subscription = self.client.on_reconnect(self.recover)
        self._thread = threading.Thread(
            target=self._run, name="kv-placement", daemon=True
        )
        self._thread.start()

    def recover(self):
        with self._lock:
            self._generation += 1
        self._wake.set()

    def publish(self, records):
        with self._lock:
            for key, parent in records:
                self._records[key] = parent
                self._pending[key] = (True, parent)
        self._wake.set()

    def retire(self, keys):
        with self._lock:
            for key in keys:
                self._records.pop(key, None)
                self._pending[key] = (False, None)
        self._wake.set()

    def _send(self, actions):
        self._seq += 1
        self.client.apply(
            self.client.pb.ApplyExternalKvBatchRequest(
                worker_id=self.worker_id,
                seq=self._seq,
                worker_address="",
                actions=actions,
            ),
            timeout=max(1.0, self.config.timeout_s),
        )

    def _flush(self):
        with self._lock:
            generation = self._generation
            recover = generation != self._completed_generation
            records = list(self._records.items()) if recover else None
            pending, self._pending = self._pending, {}
        if recover:
            self._send(
                [
                    self.client.pb.ExternalKvAction(
                        type=self.client.pb.ACTION_CLEAR_ALL_AT_TIER,
                        tier=self.client.pb.TIER_HBM,
                    )
                ]
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
        self._completed_generation = generation

    def _run(self):
        phase = zlib.crc32(self.worker_id.encode()) / (1 << 32)
        next_recovery = time.monotonic() + self.config.recovery_s * (0.5 + 0.5 * phase)
        while not self._stop.is_set():
            self._wake.clear()
            now = time.monotonic()
            if now >= next_recovery:
                self.recover()
                next_recovery = now + self.config.recovery_s
            try:
                self._flush()
            except Exception:
                logger.warning(
                    "KV placement report failed; resnapshotting", exc_info=True
                )
                self.recover()
                self._stop.wait(self.config.cooldown_s)
                continue
            self._wake.wait(min(1.0, max(0, next_recovery - time.monotonic())))

    def close(self):
        self._stop.set()
        self._wake.set()
        self._thread.join()
        self.client.channel.unsubscribe(self._subscription)
        if self._owns_client:
            self.client.close()
