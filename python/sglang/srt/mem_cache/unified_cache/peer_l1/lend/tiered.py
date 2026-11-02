"""L2 with decode HBM above the existing UMBP DRAM linker."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import Counter, defaultdict, deque
from dataclasses import replace
from queue import Empty, Full, Queue
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    LayerWiseLoadCounter,
    UnifiedCacheLinker,
)
from sglang.srt.utils import get_device_module

from ..indexer import block_hash
from .directory import LendDirectory
from .protocol import LendPeer
from .transport import DemotionStaging, LendTransport

logger = logging.getLogger(__name__)

# Restore pages per PIN lease; see _read_pinned.
_PIN_CHUNK_PAGES = 64
# Same cadence as the peer_l1 stats line.
_STATS_LOG_SECONDS = 60.0
# One non-preemptible demotion and its GPU staging: 16 pages, ~22 MB for DSv4.
_DEMOTE_BATCH_PAGES = 16


def select_pages(transfer, positions, page_size):
    indices = transfer.device_indices
    if indices is not None:
        rows = indices.reshape(-1, page_size)
        index = torch.tensor(positions, dtype=torch.int64)
        if rows.device.type != "cpu":
            # A pageable H2D would make the caller's stream sync on the host.
            index = index.pin_memory().to(rows.device, non_blocking=True)
        indices = rows[index].reshape(-1)
    return replace(
        transfer,
        keys=[transfer.keys[i] for i in positions],
        device_indices=indices,
        parent_keys=[transfer.parent_keys[i] for i in positions]
        if transfer.parent_keys is not None
        else None,
    )


class TieredL2(UnifiedCacheLinker):
    indexer_stage = 0

    def __init__(self, l1, lower):
        from ..composite import CompositeLayerCounter

        self.page_size, self.num_layers = l1.page_size, l1.pool_group.num_layers
        self.lower = lower
        self.indexer_stage = l1.indexer_config.stage
        self.l1 = l1
        self._lookups = {}
        self.directory = self._open_directory(l1) if not self.indexer_stage else None
        rank = l1.rank
        self._load_counter = LayerWiseLoadCounter(self.num_layers)
        self.layer_done_counter = (
            CompositeLayerCounter(
                SimpleNamespace(
                    layer_done_counter=self._load_counter,
                    finish_layer_wise_loading=self._load_counter.finish,
                ),
                lower,
            )
            if lower
            else self._load_counter
        )
        self.stats = Counter()
        self._pending, self._results = {}, Queue()
        self._tasks = Queue(maxsize=256)
        # Forward bounds the number of outstanding loads. They must never wait
        # for write queue capacity or sit behind a backlog of F3 writes.
        self._loads = Queue()
        self._wake = threading.Event()
        self._demotions = deque()
        self._peers = {}
        self.manager = self.transport = self.staging = None
        self._thread = None
        self._stop = threading.Event()
        self._closed = False
        self._unbound_writes = 0
        self._incompatible_peers = set()
        self._peer_retry_after = {}
        self._next_write_peer = rank
        self._rank = rank
        self._next_stats_log = time.monotonic() + _STATS_LOG_SECONDS

    @staticmethod
    def _open_directory(l1):
        rank, group = l1.rank, l1.cpu_group
        path = [
            f"/dev/shm/sglang_decode_lend_{uuid.uuid4().hex}" if rank == 0 else None
        ]
        distributed = torch.distributed.is_initialized()
        if distributed:
            torch.distributed.broadcast_object_list(path, src=0, group=group)
        # Prefill can have a much smaller KV pool than decode (e.g. the 186 GB
        # prefill limit). Size hints for the node's possible decode HBM instead.
        page_bytes = sum(
            size
            for entry in l1.pool_group.entries
            for component in entry.buffer_meta
            for _, _, size in component
        )
        world = torch.distributed.get_world_size(group) if distributed else 1
        capacity = max(
            1024,
            2
            * world
            * (
                torch.cuda.get_device_properties(
                    torch.cuda.current_device()
                ).total_memory
                // page_bytes
            ),
        )
        if rank == 0:
            directory = LendDirectory(path[0], capacity, create=True)
        if distributed:
            torch.distributed.barrier(group=group)
        if rank != 0:
            directory = LendDirectory(path[0], capacity)
        # Every rank has mapped the directory, so a crash after this point leaves
        # no named segment behind in /dev/shm.
        if distributed:
            torch.distributed.barrier(group=group)
        if rank == 0:
            directory.unlink()
        return directory

    def bind_cache(self, cache):
        if self.lower:
            self.lower.bind_cache(cache)

    def bind_pd(self, manager):
        if not manager.enable_deferred_decode_kv_release:
            raise ValueError(
                "decode lend requires deferred decode KV release on prefill"
            )
        if (
            not hasattr(manager, "kv_mem_descs")
            or manager.attn_tp_size != 1
            or manager.dcp_size != 1
        ):
            raise ValueError("decode lend requires MoRI with attention TP=1 and CP=1")
        if self.manager is not None:
            raise RuntimeError("Decode lend is already bound")
        while self._unbound_writes:
            if self.lower.num_completed_offloads():
                self._results.put(self.lower.pop_completed_offload())
                self._unbound_writes -= 1
            else:
                time.sleep(0.001)
        self.manager = manager
        self.transport = LendTransport(manager, acquire=bool(self.indexer_stage))
        if self.lower:
            if self.staging is None:
                self.staging = DemotionStaging(
                    self.transport, self.lower, _DEMOTE_BATCH_PAGES
                )
            else:
                self.staging.bind(self.transport)
            self.transport.staging = self.staging
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="prefill-lend-io", daemon=True
        )
        self._thread.start()

    def unbind_pd(self):
        if self._thread is None:
            return
        self._tasks.join()
        self._loads.join()
        self._stop.set()
        self._wake.set()
        self._thread.join()  # never deregister memory while I/O is in flight
        self._thread = None
        for peer in self._peers.values():
            peer.close()
        self._peers.clear()
        self._demotions.clear()
        self._incompatible_peers.clear()
        self._peer_retry_after.clear()
        if self.staging:
            self.staging.close()
        self.manager = self.transport = None

    def discover(self, keys, *, rid=None):
        if not self.indexer_stage:
            return self.directory.query(keys)
        query_rid, matches = self.l1.placement
        if query_rid != rid or matches is None:
            return [[-1, 0, 0]] * len(keys)
        candidates = [
            (rank, matches[peer.worker_id])
            for rank, peer in list(self._peers.items())
            if peer.worker_id in matches
        ]
        return [
            [
                next(
                    (rank for rank, held in candidates if block_hash(key) in held), -1
                ),
                0,
                0,
            ]
            for key in keys
        ]

    def release_reservation(self, rid):
        if self.indexer_stage:
            self._lookups.pop(rid, None)

    def lookup(self, rid, transfers):
        kv = next(t for t in transfers if t.name == PoolName.KV)
        # The directory is node-wide; a page on a decode rank this prefill rank has
        # no live peer for cannot be read here, and lend holds its only copy.
        values = (
            self.discover(kv.keys, rid=rid)
            if self.manager
            else [[-1, 0, 0]] * len(kv.keys)
        )
        if self.indexer_stage:
            self._lookups[rid] = dict(zip(kv.keys, values))
        present = [v[0] >= 0 and v[0] in self._peers for v in values]
        missing = [i for i, hit in enumerate(present) if not hit]
        if missing and self.lower:
            # Existence probes need keys only; no device indices at lookup.
            coverage = self.lower.page_exists(
                [replace(kv, keys=[kv.keys[i] for i in missing])]
            )
            for i, hit in zip(missing, coverage):
                present[i] = hit
        count = next((i for i, hit in enumerate(present) if not hit), len(present))
        return list(range(1, count + 1))

    def load(self, rid, transfers):
        if rid in self._pending:
            raise RuntimeError(f"Decode lend load already queued: {rid}")
        kv = next(t for t in transfers if t.name == PoolName.KV)
        if self.indexer_stage:
            saved = self._lookups.pop(rid, {})
            values = [saved.get(key, [-1, 0, 0]) for key in kv.keys]
        else:
            values = (
                self.discover(kv.keys) if self.manager else [[-1, 0, 0]] * len(kv.keys)
            )
        lower = []
        grouped = defaultdict(list)
        for i, (rank, page, gen) in enumerate(values):
            if rank < 0:
                lower.append(i)
            else:
                grouped[rank].append((i, [page, gen, kv.keys[i]]))
        if lower and (
            self.lower is None
            or not self.lower.load(rid, [select_pages(kv, lower, self.page_size)])
        ):
            return False
        self._pending[rid] = (kv, grouped)
        return True

    def cancel_queued_load(self, rid):
        if rid not in self._pending:
            return False
        self._pending.pop(rid)
        if self.lower:
            self.lower.cancel_queued_load(rid)
        return True

    def start_layer_wise_loading(self):
        pending, self._pending = self._pending, {}
        batches = [(kv, grouped) for kv, grouped in pending.values() if grouped]
        index = -1
        if batches:
            index = self._load_counter.update_producer()
            event = get_device_module().Event()
            event.record()
            self._loads.put_nowait(("load", (index, batches), event))
            self._wake.set()
        lower_index = self.lower.start_layer_wise_loading() if self.lower else -1
        if index == lower_index == -1:
            return -1
        return (
            self.layer_done_counter.register(index, lower_index)
            if self.lower
            else index
        )

    def finish_layer_wise_loading(self, index):
        return self.layer_done_counter.finish(index)

    def offload(self, transfers):
        if self.manager is None:
            accepted = self.lower is not None and self.lower.offload(transfers)
            self._unbound_writes += int(accepted)
            return accepted
        kv = next(t for t in transfers if t.name == PoolName.KV)
        # F3 passes CPU indices. Other callers snapshot before queuing too.
        kv = replace(
            kv,
            keys=list(kv.keys),
            device_indices=kv.device_indices.detach().cpu().clone(),
        )
        event = get_device_module().Event()
        event.record()
        try:
            self._tasks.put_nowait(("write", kv, event))
            self._wake.set()
            return True
        except Full:
            self.stats["write_refused_pages"] += len(kv.keys)
            return False

    def num_completed_offloads(self):
        return self._results.qsize() + (
            self.lower.num_completed_offloads() if self._unbound_writes else 0
        )

    def pop_completed_offload(self):
        if not self._results.empty():
            return self._results.get_nowait()
        if self._unbound_writes:
            self._unbound_writes -= 1
            return self.lower.pop_completed_offload()
        raise Empty

    def _refresh_peers(self):
        # A restarted decoder can leave its old registration in the PD table.
        # For the first-step 1P1D topology, the newest registration owns a rank.
        latest = {
            info.decode_dp_rank: info
            for info in list(self.manager.decode_kv_args_table.values())
            if getattr(info, "lend_port", None)
        }
        live_keys = {info.engine_key for info in latest.values()}
        self._peer_retry_after = {
            key: deadline
            for key, deadline in self._peer_retry_after.items()
            if key in live_keys
        }
        for info in latest.values():
            if info.engine_key in self._incompatible_peers:
                continue
            if time.monotonic() < self._peer_retry_after.get(info.engine_key, 0):
                continue
            previous = self._peers.get(info.decode_dp_rank)
            if previous is not None:
                if previous.info.engine_key == info.engine_key:
                    continue
                previous.close()
                del self._peers[info.decode_dp_rank]
            try:
                peer = LendPeer(info, self._on_event, layout=self.transport.lengths)
            except ValueError:
                self._incompatible_peers.add(info.engine_key)
                logger.exception("Ignoring decode lend peer with a different KV layout")
                continue
            if not peer.epoch:
                peer.close()
                self._peer_retry_after[info.engine_key] = time.monotonic() + 5.0
                continue
            self._peer_retry_after.pop(info.engine_key, None)
            self._peers[info.decode_dp_rank] = peer

    def _on_event(self, peer, message):
        rank = peer.info.decode_dp_rank
        if message.op == "EVICTED":
            if self.directory is not None:
                self.directory.remove(rank, message.data)
        elif message.op == "DEMOTE":
            for start in range(0, len(message.data), _DEMOTE_BATCH_PAGES):
                self._demotions.append(
                    (peer, message.data[start : start + _DEMOTE_BATCH_PAGES])
                )

    def _serve_load(self):
        try:
            _, (index, batches), event = self._loads.get_nowait()
        except Empty:
            return False
        started = time.monotonic()
        try:
            event.synchronize()
            self._read(batches)
            for layer in range(self.num_layers):
                self._load_counter.complete(index, layer)
        except Exception as exc:
            self._load_counter.fail(index, exc)
            logger.exception("Decode lend restore failed")
        finally:
            self._loads.task_done()
            self.stats["load_ms"] += int((time.monotonic() - started) * 1000)
        return True

    def _maybe_log_stats(self):
        now = time.monotonic()
        if now >= self._next_stats_log:
            self._next_stats_log = now + _STATS_LOG_SECONDS
            logger.info(
                "decode lend L2 rank=%d stats: %s",
                self._rank,
                dict(sorted(self.stats.items())),
            )

    def _run(self):
        torch.cuda.set_device(self.manager.kv_args.gpu_id)
        # With few HIP hardware queues (GPU_MAX_HW_QUEUES), normal-priority streams
        # share them with the forward; a sync here must not wait behind a forward.
        device = get_device_module()
        with device.stream(device.Stream(priority=-1)):
            self._loop()

    def _loop(self):
        while not self._stop.is_set():
            try:
                if self._serve_load():
                    continue
                self._maybe_log_stats()
                self._refresh_peers()
                for peer in self._peers.values():
                    peer.flush_cleanup()
                    while peer.socket.poll(0):
                        peer.poll()
                if self._serve_load():
                    continue
                try:
                    _, payload, event = self._tasks.get_nowait()
                except Empty:
                    if self._demotions:
                        self._demote(*self._demotions.popleft())
                    else:
                        self._wake.wait(0.02)
                        self._wake.clear()
                    continue
                started = time.monotonic()
                try:
                    event.synchronize()
                    self._results.put(self._write(payload))
                except Exception:
                    self._results.put(False)
                    logger.exception("Decode lend write failed")
                finally:
                    self._tasks.task_done()
                    self.stats["write_ms"] += int((time.monotonic() - started) * 1000)
                # Continuous F3 writes must not starve demotions: reclaim one
                # staging batch between tasks as well as when the queue is idle.
                if self._demotions and self._loads.empty():
                    self._demote(*self._demotions.popleft())
            except Exception:
                logger.exception("Decode lend control operation failed")
                self._stop.wait(0.05)

    def _read(self, batches):
        for kv, grouped in batches:
            pages = (
                kv.device_indices.reshape(-1, self.page_size)[:, 0].cpu()
                // self.page_size
            ).tolist()
            for rank, items in grouped.items():
                if not self._read_pinned(self._peers.get(rank), kv, items, pages):
                    if self.directory is not None:
                        self.directory.remove(rank, [r for _, r in items])
                    self.stats["pin_or_read_miss_pages"] += len(items)
                    raise RuntimeError("Decode lend PIN or read failed")
                self.stats["read_pages"] += len(items)

    def _read_pinned(self, peer, kv, items, pages):
        # A decode rank pins at most 1% of its pages; one PIN for a long prefix is
        # refused. Each prefill I/O thread holds one lease at a time, so 8 x 64 fits.
        for start in range(0, len(items), _PIN_CHUNK_PAGES):
            chunk = items[start : start + _PIN_CHUNK_PAGES]
            if peer is not None and self.transport.read(
                peer, [r for _, r in chunk], [pages[i] for i, _ in chunk]
            ):
                continue
            # A page demoted after lookup is gone from decode but already in the
            # lower tier; only pages decode dropped outright fail the load.
            if not self._read_lower(kv, [i for i, _ in chunk]):
                return False
            self.stats["lower_fallback_pages"] += len(chunk)
        return True

    def _read_lower(self, kv, positions):
        lower = self.lower
        if lower is None:
            return False
        expanded = lower.pool_group.resolve_transfers(
            [select_pages(kv, positions, self.page_size)]
        )
        for plan in lower._build_load_plans([expanded]):
            ptrs, sizes, offsets = lower._all_layer_ranges(plan)
            step = lower._entries_per_call(sizes)
            for start in range(0, len(plan.keys), step):
                end = start + step
                keys = plan.keys[start:end]
                results = list(
                    lower.storage.client.batch_get_ranges_into_ptr(
                        keys, ptrs[start:end], sizes[start:end], offsets[start:end]
                    )
                )
                if len(results) != len(keys) or not all(results):
                    return False
        return True

    def _write(self, kv):
        pages = (
            kv.device_indices.reshape(-1, self.page_size)[:, 0] // self.page_size
        ).tolist()
        remaining = list(range(len(pages)))
        if self.indexer_stage and (not kv.indexer_eligible or kv.parent_keys is None):
            peers = []
        else:
            peers = list(self._peers.items())
        # Rotate the first decode rank so writes, and later restores, spread over
        # all decode NICs instead of filling the first registered rank.
        if self.indexer_stage:
            peers = [(rank, peer) for rank, peer in peers if peer.worker_id]
        if peers:
            start = self._next_write_peer % len(peers)
            self._next_write_peer += 1
            peers = peers[start:] + peers[:start]
        for rank, peer in peers:
            if not remaining:
                break
            reply = peer.call("ALLOC", [len(remaining)])
            records = reply.data if reply is not None else []
            if not records:
                continue
            selected, remaining = remaining[: len(records)], remaining[len(records) :]
            records = [[p, g, kv.keys[i]] for (p, g, _), i in zip(records, selected)]
            committed = []
            try:
                if self.transport.transfer(
                    peer.info,
                    [pages[i] for i in selected],
                    [r[0] for r in records],
                    write=True,
                ):
                    commits = (
                        [
                            r + [[kv.parent_keys[i], kv.indexer_eligible]]
                            for r, i in zip(records, selected)
                        ]
                        if self.indexer_stage
                        else records
                    )
                    reply = peer.call("COMMIT", commits)
                    if reply is not None:
                        committed = [i for i, ok in enumerate(reply.data) if ok]
                        published = (
                            self.directory.put(rank, [records[i] for i in committed])
                            if self.directory is not None
                            else len(committed)
                        )
                        self.stats["directory_refused_pages"] += (
                            len(committed) - published
                        )
            finally:
                # Also handles a lost COMMIT reply: ABORT is idempotent and
                # only touches WRITING; already committed pages stay valid.
                peer.call("ABORT", records)
            remaining.extend(
                selected[i] for i in range(len(selected)) if i not in committed
            )
            self.stats["write_pages"] += len(committed)
        if not remaining:
            return True
        if self.lower is None:
            return False
        if not self.lower.offload([select_pages(kv, remaining, self.page_size)]):
            return False
        while not self.lower.num_completed_offloads():
            # The lower tier owns an asynchronous write. Its pending F3 result
            # keeps source rows alive, while this thread can restore other rows.
            if not self._serve_load():
                time.sleep(0.001)
        return self.lower.pop_completed_offload()

    def _demote(self, peer, records):
        # DEMOTE is a hint. Every read still needs its own PIN.
        started = time.monotonic()
        try:
            if self.staging and self.transport.read(
                peer, records, list(range(len(records))), staging=True
            ):
                read = time.monotonic()
                self.stats["demote_read_ms"] += int((read - started) * 1000)
                if self.staging.put(self.lower, records):
                    if self.directory is not None:
                        self.directory.remove(peer.info.decode_dp_rank, records)
                    self.stats["demote_pages"] += len(records)
                self.stats["demote_put_ms"] += int((time.monotonic() - read) * 1000)
        finally:
            peer.call("DEMOTED", records)
            self.stats["demote_ms"] += int((time.monotonic() - started) * 1000)

    def reset(self):
        self._tasks.join()
        self._loads.join()
        self._pending.clear()
        self._lookups.clear()
        if self.lower:
            self.lower.reset()
        self._load_counter.reset()
        if self.lower:
            self.layer_done_counter.reset()

    def close(self):
        if self._closed:
            return
        self.unbind_pd()
        if self.lower:
            self.lower.close()
        if self.staging:
            self.staging.release()
        if self.directory is not None:
            self.directory.close()
        logger.info("decode lend L2 stats: %s", self.stats)
        self._closed = True
