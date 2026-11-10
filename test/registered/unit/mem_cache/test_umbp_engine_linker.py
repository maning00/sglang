"""Engine read adapter: real reservation/wrapper paths, fake native I/O boundary."""

import threading
import time
import unittest
from array import array
from enum import Enum, auto
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from test_provider_prefill_reservation import adder, request
from test_umbp_engine_provider import Client, PublishState

from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle, InsertParams
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.kv_memory_provider import (
    CapacityShape,
    CompletedReceipt,
    PageGeometry,
    PageRegion,
    PoolEpoch,
    ProviderFailStop,
    reservation_request_id,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.umbp.engine_provider import UMBPEngineProvider
from sglang.srt.mem_cache.storage.umbp.umbp_engine_linker import UMBPEngineLinker
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    ExternalCacheHitMarker,
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kv_memory_provider import ManualReceipt, build_provider_fixture
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class Outcome(Enum):
    Pending = auto()
    Completed = auto()
    Failed = auto()
    Unknown = auto()
    Canceled = auto()


class Cancel(Enum):
    Rejected = auto()
    NeverSubmitted = auto()
    InFlight = auto()
    Completed = auto()


class Source(Enum):
    LocalDram = auto()
    LocalHbm = auto()


class ReadClient(Client):
    def __init__(self, *args):
        super().__init__(*args)
        self.next_page = 1
        self.lookups, self.prepares, self.cancels = [], [], []
        self.boundaries, self.valid_prefix = None, None
        self.reject = False
        self.cancel_result = None
        self.layers = (0, 0, 1)  # two component copies in logical layer zero
        self.source_kinds = None

    def attach_executor(self, executor):
        super().attach_executor(executor)
        executor.client = self

    def take(self, rid, purpose, n):
        result = super().take(rid, purpose, n)
        for h in result:
            h.page_id = self.next_page
            self.next_page += 1
        return result

    def accounting(self):
        snapshot = super().accounting()
        snapshot.cached = 16 - len(self.quarantined)
        snapshot.total = 16
        return snapshot

    def lookup(self, keys, namespace):
        self.lookups.append((keys, namespace))
        valid = len(keys) if self.valid_prefix is None else self.valid_prefix
        boundaries = (
            list(range(1, valid + 1)) if self.boundaries is None else self.boundaries
        )
        return SimpleNamespace(
            success=valid > 0,
            id=len(self.lookups),
            keys=keys,
            namespace=namespace,
            valid_prefix=valid,
            legal_boundaries=boundaries,
            layout_id="layout",
            epoch=SimpleNamespace(boot_id=b"a" * 16, reset_seq=0),
            requester_backend_id=self.backend_id,
            sources=[
                SimpleNamespace(
                    kind=self.source_kinds[i] if self.source_kinds else Source.LocalDram
                )
                for i in range(len(keys))
            ],
            error="",
        )

    def prepare_load(self, plan, request_id, targets, n_pages):
        self.prepares.append((plan, request_id, targets, n_pages))
        return SimpleNamespace(
            success=not self.reject,
            id=len(self.prepares),
            request_id=request_id,
            targets=targets,
            layer_count=len(self.layers),
            layer_ids=self.layers,
            started=False,
            target_hold=None if self.reject else self.hold_io(targets),
            hold_completed=False,
            error="Rejected: grant or size/fd limit" if self.reject else "",
        )

    def cancel_load(self, prepared):
        self.cancels.append(prepared)
        result = self.cancel_result or (
            Cancel.InFlight if prepared.started else Cancel.NeverSubmitted
        )
        if result == Cancel.NeverSubmitted and not prepared.hold_completed:
            self.complete_io(prepared.target_hold, "never_submitted")
            prepared.hold_completed = True
        return result


class LoadHandle:
    def __init__(self, prepared, client):
        self.id, self.layer_count = prepared.id, prepared.layer_count
        self.prepared, self.client = prepared, client
        self.outcome = Outcome.Pending
        self.done = [False] * self.layer_count
        self.results = [Outcome.Pending] * self.layer_count
        self.done_calls, self.outcome_calls = [], []

    def layer_done(self, i):
        self.done_calls.append(i)
        return self.done[i]

    def layer_outcome(self, i):
        self.outcome_calls.append(i)
        return self.results[i]

    def layer_event(self, i):
        raise AssertionError("I9 event must not be used before I11 record guarantees")

    def wait(self):
        raise AssertionError("linker must not wait for whole native load")

    def complete(self, layer, outcome=Outcome.Completed, *, terminal=True):
        self.results[layer], self.done[layer] = outcome, True
        if not terminal:
            return  # the native FinishLoad/target hold has not resolved yet
        if outcome != Outcome.Completed:
            self.outcome = outcome  # Failed is physically quiesced; Unknown is not.
        elif all(self.done):
            self.outcome = Outcome.Completed
        if self.outcome != Outcome.Pending and not self.prepared.hold_completed:
            self.client.complete_io(
                self.prepared.target_hold,
                "unknown" if self.outcome == Outcome.Unknown else "completed",
            )
            self.prepared.hold_completed = True

    def complete_all(self):
        for layer in range(self.layer_count):
            self.complete(layer)


class Executor:
    def __init__(self, *args):
        self.handles = []
        self.submit_entered, self.submit_gate = threading.Event(), threading.Event()
        self.submit_gate.set()
        self.polls = 0
        self.raise_on_submit = False
        self.raise_after_start = False

    def submit_load(self, prepared):
        self.submit_entered.set()
        assert self.submit_gate.wait(3)
        if self.raise_on_submit:
            prepared.started = self.raise_after_start
            raise RuntimeError("native submission exception")
        prepared.started = True
        handle = LoadHandle(prepared, self.client)
        self.handles.append(handle)
        return handle

    def poll_loads(self):
        self.polls += 1
        return []

    def close(self):
        pass


class TestUMBPEngineLinker(CustomTestCase):
    def setUp(self):
        self.enterContext(get_context().override_server_args())
        engine = SimpleNamespace(
            EngineClient=ReadClient,
            EngineExecutor=Executor,
            ReservationPurpose=SimpleNamespace(Load="load", Extend="extend"),
            CompletionProof=SimpleNamespace(Completed="completed", Unknown="unknown"),
            ResidentRecord=SimpleNamespace,
            PublishReceiptState=PublishState,
            RetireState=SimpleNamespace(
                Quarantined="quarantined", Ready="ready", Released="released"
            ),
            LoadOutcome=Outcome,
            CancelLoadResult=Cancel,
        )
        mori, umbp = ModuleType("mori"), ModuleType("mori.umbp")
        mori.umbp, umbp.engine = umbp, engine
        with patch.dict("sys.modules", {"mori": mori, "mori.umbp": umbp}):
            p = self.p = UMBPEngineProvider(
                device="cpu",
                options={
                    "node_address": "unix:///tmp/g2.sock",
                    "namespace": {
                        "model": "m",
                        "weights": "rev",
                        "kv_format": "fp8",
                        "parallel": "dep8",
                    },
                },
            )
        p.geometry = PageGeometry(
            4,
            16,
            32,
            "layout",
            (
                PageRegion("a", 4096, 272, 16, 1, 0, 0, "kv"),
                PageRegion("b", 8192, 272, 16, 1, 16, 1, "kv"),
            ),
        )
        p.registration, p.epoch = object(), PoolEpoch(b"a" * 16, 0)
        p.client.pool_id = p.pool_id
        self.p, self.a, self.c, self.storage = build_provider_fixture(
            provider_override=p
        )
        self.linker = p.create_linker(
            None,
            SimpleNamespace(token_to_kv_pool_allocator=self.a),
            components={ComponentType.FULL},
            forward_stream=None,
        )
        self.addCleanup(self._close)

    def _close(self):
        self.p.executor.submit_gate.set()
        try:
            self.linker.close()
        except ProviderFailStop:
            # Fake native handles have no hardware resources; stop only this
            # fixture's monitor when a test deliberately exercises fail-stop.
            with self.linker._changed:
                self.linker._stopping = True
                self.linker._changed.notify_all()
            self.linker._thread.join(3)

    def until(self, predicate):
        end = time.monotonic() + 3
        while time.monotonic() < end:
            if predicate():
                return
            time.sleep(0.001)
        self.fail("fake load progress did not reach the asserted state")

    def queued(self, rid="r", keys=("a",)):
        self.linker.lookup(rid, [PoolTransfer(PoolName.KV, keys=list(keys))])
        ticket = self.p.reserve(CapacityShape(rid, 1, len(keys), 0, 0, len(keys) * 4))
        with self.a.full_attn_allocator.reservation_scope(ticket, "load"):
            target = self.storage.take_units("full", len(keys) * 4)
        # Use the canonical token sub-pool name from the allocator.
        self.assertTrue(
            self.linker.load(
                rid, [PoolTransfer(PoolName.KV, device_indices=target, keys=list(keys))]
            )
        )
        ticket.refund_unused()
        return target

    def test_lookup_returns_exact_sparse_boundaries_and_one_namespaced_rpc(self):
        self.p.client.valid_prefix, self.p.client.boundaries = 3, [1, 3]
        result = self.linker.lookup(
            "r", [PoolTransfer(PoolName.KV, keys=["a", "b", "c", "missing"])]
        )
        self.assertEqual(result, [1, 3])
        self.assertEqual(
            self.p.client.lookups,
            [
                (
                    [self.p.storage_key(k) for k in ("a", "b", "c", "missing")],
                    self.p.namespace_seed,
                )
            ],
        )
        wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)

        def intersect(mask, op):
            mask.copy_(torch.minimum(mask, torch.tensor([0, 0, 1, 0, 0])))

        wrapper.cache = SimpleNamespace(_all_reduce_attn_groups=intersect)
        self.assertEqual(
            wrapper._sync_restorable_prefix(result, num_pages=4, device_hit_pages=0), 0
        )
        self.p.client.valid_prefix, self.p.client.boundaries = 0, []
        self.assertEqual(
            self.linker.lookup("r", [PoolTransfer(PoolName.KV, keys=["missing"])]), []
        )
        # Negative lookup plans are short-lived too, so repeated misses avoid
        # one synchronous node/master probe per scheduler round.
        self.assertEqual(
            self.linker.lookup("r", [PoolTransfer(PoolName.KV, keys=["missing"])]), []
        )
        self.assertEqual(len(self.p.client.lookups), 2)

    def test_g4_repeated_match_reuses_lookup_until_keys_or_age_change(self):
        self.c.linker = UnifiedCacheLinkerWrapper(self.c, self.linker)
        req = request(self.c, 16)
        now = [100.0]
        with patch(
            "sglang.srt.mem_cache.storage.umbp.umbp_engine_linker.time",
            SimpleNamespace(monotonic=lambda: now[0]),
        ):
            for _ in range(4):
                req.init_next_round_input(self.c, cow_mamba=False)
            self.assertEqual(len(self.p.client.lookups), 1)
            now[0] += 1.9
            req.init_next_round_input(self.c, cow_mamba=False)
            self.assertEqual(len(self.p.client.lookups), 1)
            now[0] += 0.2
            req.init_next_round_input(self.c, cow_mamba=False)
            self.assertEqual(len(self.p.client.lookups), 2)
            self.linker.lookup(req.rid, [PoolTransfer(PoolName.KV, keys=["changed"])])
            self.assertEqual(len(self.p.client.lookups), 3)

    def test_g4_idle_expiry_drops_plans_without_request_release(self):
        self.c.linker = UnifiedCacheLinkerWrapper(self.c, self.linker)
        req = request(self.c, 8)
        now = [100.0]
        with patch(
            "sglang.srt.mem_cache.storage.umbp.umbp_engine_linker.time",
            SimpleNamespace(monotonic=lambda: now[0]),
        ):
            with patch.object(
                self.c,
                "_all_reduce_attn_groups",
                side_effect=lambda mask, op: mask.zero_(),
            ):
                req.init_next_round_input(self.c, cow_mamba=False)
            self.assertNotIn(req.rid, self.c.linker.hit_markers)
            self.linker.lookup("finished", [PoolTransfer(PoolName.KV, keys=["a"])])
            self.assertIn(req.rid, self.linker._plans)
            self.assertIn("finished", self.linker._plans)
            with self.linker._changed:
                now[0] += 5.1
                self.linker._changed.notify_all()
            self.until(lambda: not self.linker._plans)

    def _stop_monitor(self):
        # Deterministically drive the real _advance below without concurrent
        # native-handle queries from the fixture's monitor.
        with self.linker._changed:
            self.linker._stopping = True
            self.linker._changed.notify_all()
        self.linker._thread.join(3)
        self.assertFalse(self.linker._thread.is_alive())
        self.linker._stopping = False

    def test_g6_failed_miss_plan_is_cached_across_real_matches(self):
        self.c.linker = UnifiedCacheLinkerWrapper(self.c, self.linker)
        self.p.client.valid_prefix, self.p.client.boundaries = 0, []
        req = request(self.c, 16)
        now = [100.0]
        with patch(
            "sglang.srt.mem_cache.storage.umbp.umbp_engine_linker.time",
            SimpleNamespace(monotonic=lambda: now[0]),
        ):
            for _ in range(4):
                req.init_next_round_input(self.c, cow_mamba=False)
            self.assertEqual(req.host_hit_length, 0)
            self.assertEqual(len(self.p.client.lookups), 1)
            now[0] += 1.99
            req.init_next_round_input(self.c, cow_mamba=False)
            self.assertEqual(len(self.p.client.lookups), 1)
            now[0] += 0.02
            req.init_next_round_input(self.c, cow_mamba=False)
            self.assertEqual(len(self.p.client.lookups), 2)

    def test_g6_reused_plan_can_load_after_two_second_lookup_window(self):
        now = [100.0]
        transfer = PoolTransfer(PoolName.KV, keys=["a"])
        with patch(
            "sglang.srt.mem_cache.storage.umbp.umbp_engine_linker.time",
            SimpleNamespace(monotonic=lambda: now[0]),
        ):
            self.assertEqual(self.linker.lookup("r", [transfer]), [1])
            now[0] = 101.99
            self.assertEqual(self.linker.lookup("r", [transfer]), [1])
            self.assertEqual(len(self.p.client.lookups), 1)
            with self.linker._changed:
                now[0] = 102.01
                self.linker._expire_plans()  # the same expiry used by the monitor
            reservation = self.p.reserve(CapacityShape("r", 1, 1, 0, 0, 4))
            with self.a.full_attn_allocator.reservation_scope(reservation, "load"):
                target = self.storage.take_units(self.storage.owner, 4)
            self.assertTrue(
                self.linker.load(
                    "r", [PoolTransfer(PoolName.KV, keys=["a"], device_indices=target)]
                )
            )
            self.assertEqual(self.p.client.prepares[-1][0].id, 1)
            self.assertTrue(self.linker.cancel_queued_load("r"))
            reservation.refund_unused()
            self.a.full_attn_allocator.free(target)

    def test_g6_lookup_keeps_only_legal_boundaries_within_prepare_limit(self):
        self.p.client.boundaries = [1, 255, 256, 257, 300]
        boundaries = self.linker.lookup(
            "large", [PoolTransfer(PoolName.KV, keys=[str(i) for i in range(300)])]
        )
        self.assertEqual(boundaries, [1, 255, 256])
        wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
        wrapper.cache = SimpleNamespace(_all_reduce_attn_groups=lambda *args: None)
        self.assertEqual(
            wrapper._sync_restorable_prefix(
                boundaries, num_pages=300, device_hit_pages=0
            ),
            256,
        )

    def test_g6_scheduler_fatal_can_win_a_layer_completion_race(self):
        self._stop_monitor()
        self.queued()
        index = self.linker.start_layer_wise_loading()
        batch = self.linker._batches[index]
        self.linker._advance(batch)
        self.p.executor.handles[0].complete(0)
        self.p.executor.handles[0].complete(1)
        future = self.linker.layer_done_counter.futures[index][0]
        original = future.set_result
        entered, release = threading.Event(), threading.Event()
        errors = []

        def set_result(value):
            entered.set()
            assert release.wait(3)
            original(value)

        def advance():
            try:
                self.linker._advance(batch)
            except Exception as exc:
                errors.append(exc)

        fatal = ProviderFailStop("scheduler observed incomplete forward")
        with patch.object(future, "set_result", side_effect=set_result):
            worker = threading.Thread(target=advance)
            worker.start()
            try:
                self.assertTrue(entered.wait(3))
                self.linker._fatal(batch, fatal)
            finally:
                release.set()
                worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertIs(future.exception(), fatal)

    def test_g6_layer_completion_can_win_a_counter_failure_race(self):
        self._stop_monitor()
        counter = self.linker.layer_done_counter
        index = counter.update_producer()
        future = counter.futures[index][0]
        original = future.set_exception
        entered, release = threading.Event(), threading.Event()
        errors = []

        def set_exception(error):
            entered.set()
            assert release.wait(3)
            original(error)

        fatal = ProviderFailStop("completion raced fail-stop")

        def fail():
            try:
                counter.fail(index, fatal)
            except Exception as exc:
                errors.append(exc)

        with patch.object(future, "set_exception", side_effect=set_exception):
            worker = threading.Thread(target=fail)
            worker.start()
            try:
                self.assertTrue(entered.wait(3))
                counter.complete(index, 0)
            finally:
                release.set()
                worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertIsNone(future.exception())
        self.assertIs(counter.futures[index][1].exception(), fatal)
        self.assertFalse(counter.finish(index))

    def test_g4_cursor_queries_only_next_copy_and_never_rescans_completed_copies(self):
        self._stop_monitor()
        self.p.client.layers = (0,) * 64 + (1,) * 64
        self.queued()
        index = self.linker.start_layer_wise_loading()
        batch = self.linker._batches[index]
        self.linker._advance(batch)
        handle = self.p.executor.handles[0]
        self.assertEqual(set(handle.done_calls), {0})
        handle.complete(0)
        self.linker._advance(batch)
        handle.done_calls.clear()
        self.linker._advance(batch)
        self.assertEqual(handle.done_calls, [1])
        handle.complete_all()
        for _ in range(16):
            self.linker._advance(batch)
            if batch.done.is_set():
                break
        self.assertTrue(batch.done.is_set())
        self.assertEqual(sorted(handle.outcome_calls), list(range(128)))
        self.assertTrue(self.linker.finish_layer_wise_loading(index))

    def test_g4_layer_count_comes_from_pool_even_with_no_full_in_last_layer(self):
        self.linker.close()
        pool = SimpleNamespace(start_layer=0, layer_num=3)
        params = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: pool)
        )
        self.linker = UMBPEngineLinker(self.p, params, components={ComponentType.FULL})
        self.assertEqual(self.linker.num_layers, 3)
        self.queued()
        index = self.linker.start_layer_wise_loading()
        self.until(lambda: bool(self.p.executor.handles))
        self.p.executor.handles[0].complete_all()
        self.linker.layer_done_counter.set_consumer(index)
        self.linker.layer_done_counter.wait_until(2)
        self.assertTrue(self.linker.finish_layer_wise_loading(index))

    def test_g4_partial_hit_prepare_preserves_target_order_and_page_count(self):
        self.p.client.valid_prefix, self.p.client.boundaries = 2, [1, 2]
        keys = ["page-a", "page-b", "missing"]
        self.assertEqual(
            self.linker.lookup("partial", [PoolTransfer(PoolName.KV, keys=keys)]),
            [1, 2],
        )
        reservation = self.p.reserve(CapacityShape("partial", 1, 2, 0, 0, 8))
        with self.a.full_attn_allocator.reservation_scope(reservation, "load"):
            indices = self.storage.take_units(self.storage.owner, 8)
        expected = tuple(reversed(self.p.freeze_allocations(indices).handles()))
        indices = indices.reshape(2, 4).flip(0).reshape(-1)
        indices._kv_provider_page_ids = tuple(h.page_id for h in expected)
        self.assertTrue(
            self.linker.load(
                "partial",
                [PoolTransfer(PoolName.KV, device_indices=indices, keys=keys[:2])],
            )
        )
        plan, rid, targets, n_pages = self.p.client.prepares[-1]
        self.assertEqual(plan.keys, [self.p.storage_key(k) for k in keys])
        self.assertEqual(rid, "partial")
        self.assertEqual(targets, [self.p._native[h] for h in expected])
        self.assertEqual(n_pages, 2)
        self.assertTrue(self.linker.cancel_queued_load("partial"))
        reservation.refund_unused()
        self.a.full_attn_allocator.free(indices)

    def test_g4_cached_prepare_rejected_frees_targets_and_adder_reselects_recompute(
        self,
    ):
        self.c.linker = UnifiedCacheLinkerWrapper(self.c, self.linker)
        req = request(self.c)
        self.linker.lookup(req.rid, [PoolTransfer(PoolName.KV, keys=["a"])])
        self.linker.lookup(req.rid, [PoolTransfer(PoolName.KV, keys=["a"])])
        self.assertEqual(len(self.p.client.lookups), 1)
        self.c.linker.hit_markers[req.rid] = ExternalCacheHitMarker(["a"], 0)
        req.host_hit_length = 4
        self.p.client.reject = True
        add = adder(self.a, self.c)
        add.add_one_req(req, False, None)
        self.assertEqual(add.can_run_list, [req])
        self.assertEqual(req.host_loaded_length, 0)
        self.assertEqual(req.kv.reservation.remaining("load"), 0)
        self.assertEqual(req.kv.reservation.remaining("extend"), 2)
        self.assertFalse(self.p._owned)
        self.assertEqual(self.linker.stats["prepare_rejected"], 1)
        self.assertEqual(self.p.client.prepares[0][0].id, 1)
        req.kv.reservation.refund_unused()

    def test_counter_waits_for_every_component_and_request_without_blocking_start(self):
        self.queued("a")
        self.queued("b")
        self.p.executor.submit_gate.clear()
        index = self.linker.start_layer_wise_loading()
        self.assertTrue(self.p.executor.submit_entered.wait(1))
        self.assertFalse(self.linker.layer_done_counter.futures[index][0].done())
        self.p.executor.submit_gate.set()
        self.until(lambda: len(self.p.executor.handles) == 2)
        a, b = self.p.executor.handles
        a.complete(0)
        a.complete(1)
        b.complete(0)
        before = self.p.executor.polls
        self.until(lambda: self.p.executor.polls > before + 1)
        self.assertFalse(self.linker.layer_done_counter.futures[index][0].done())
        b.complete(1)
        self.until(lambda: self.linker.layer_done_counter.futures[index][0].done())
        self.assertFalse(self.linker.layer_done_counter.futures[index][1].done())
        a.complete(2)
        b.complete(2)
        self.linker.layer_done_counter.set_consumer(index)
        self.linker.layer_done_counter.wait_until(1)
        self.assertTrue(self.linker.finish_layer_wise_loading(index))
        self.assertEqual(self.linker.stats["completed_LocalDram_pages"], 2)

    def _wrapper_load(self, rid):
        req = request(self.c, 4)
        req.rid = rid
        req.kv.reservation_revision = 1
        req.kv.reservation = self.p.reserve(
            CapacityShape(reservation_request_id(req), 1, 1, 0, 0, 4)
        )
        self.linker.lookup(rid, [PoolTransfer(PoolName.KV, keys=[rid])])
        self.c.linker.hit_markers[rid] = ExternalCacheHitMarker([rid], 0)
        self.assertEqual(self.c.linker.load_back(req)[0], 4)
        req.kv.reservation.refund_unused()
        req.kv.reservation = None
        return req

    def test_post_submit_failure_discards_the_whole_batch_without_tree_insert(self):
        self.c.linker = UnifiedCacheLinkerWrapper(self.c, self.linker)
        reqs = [self._wrapper_load("a"), self._wrapper_load("b")]
        index = self.c.linker.start_layer_wise_loading()
        self.until(lambda: len(self.p.executor.handles) == 2)
        self.p.executor.handles[0].complete_all()
        self.p.executor.handles[1].complete(0, Outcome.Failed)
        self.linker.layer_done_counter.set_consumer(index)
        self.linker.layer_done_counter.wait_until(1)
        failed = self.c.finish_external_linker_loads(reqs)
        self.assertEqual(failed, reqs)
        self.assertTrue(
            all(
                r.skip_radix_cache_insert and r.discard_output_reason.status_code == 503
                for r in reqs
            )
        )
        self.assertEqual(self.c.all_values_flatten().numel(), 0)

    def test_loaded_pages_publish_only_after_successful_batch_checkpoint(self):
        self.c.linker = UnifiedCacheLinkerWrapper(self.c, self.linker)
        req = self._wrapper_load("success")
        loaded = self.c.linker.private_load_indices[req.rid]
        index = self.c.linker.start_layer_wise_loading()
        self.assertEqual(self.c.all_values_flatten().numel(), 0)
        self.until(lambda: bool(self.p.executor.handles))
        self.p.executor.handles[0].complete_all()
        self.linker.layer_done_counter.set_consumer(index)
        self.linker.layer_done_counter.wait_until(1)
        self.assertEqual(self.c.finish_external_linker_loads([req]), [])
        self.assertFalse(self.p.client.published)
        self.c.insert(InsertParams(key=RadixKey(array("q", range(4))), value=loaded))
        self.assertEqual(self.c.all_values_flatten().tolist(), loaded.tolist())
        self.assertEqual(len(self.p.client.published), 1)

    def test_g4_unknown_isolates_targets_and_aborts_whole_batch_without_fail_stop(self):
        self.c.linker = UnifiedCacheLinkerWrapper(self.c, self.linker)
        reqs = [self._wrapper_load("unknown"), self._wrapper_load("healthy")]
        unknown_targets = self.c.linker.private_load_indices[reqs[0].rid]
        healthy_targets = self.c.linker.private_load_indices[reqs[1].rid]
        isolated = self.p.freeze_allocations(unknown_targets).handles()
        index = self.c.linker.start_layer_wise_loading()
        self.until(lambda: len(self.p.executor.handles) == 2)
        unknown, healthy = self.p.executor.handles
        before = self.p.executor.polls
        unknown.complete(0, Outcome.Unknown, terminal=False)
        self.until(lambda: self.p.executor.polls > before + 1)
        self.assertIsNone(self.p._failed)
        self.assertFalse(self.p._quarantined)
        self.assertFalse(self.linker.layer_done_counter.futures[index][0].done())
        unknown.complete(0, Outcome.Unknown)
        self.until(lambda: bool(self.p._quarantined))
        self.assertFalse(self.linker._batches[index].done.is_set())
        healthy.complete_all()
        self.linker.layer_done_counter.set_consumer(index)
        self.linker.layer_done_counter.wait_until(1)
        self.assertEqual(self.c.finish_external_linker_loads(reqs), reqs)
        self.assertTrue(all(r.skip_radix_cache_insert for r in reqs))
        self.assertTrue(all(r.discard_output_reason.status_code == 503 for r in reqs))
        self.assertEqual(self.c.all_values_flatten().numel(), 0)
        self.assertEqual(self.p._quarantined, set(isolated))
        self.assertTrue(self.p._retained)
        self.assertIsNone(self.p._failed)
        self.assertEqual(self.linker.stats["unknown"], 1)
        self.a.full_attn_allocator.free(unknown_targets)
        self.a.full_attn_allocator.free(healthy_targets)
        self.linker.lookup("next", [PoolTransfer(PoolName.KV, keys=["a"])])
        self.assertEqual(len(self.p.client.lookups), 3)
        next_targets = self.queued("next")
        next_index = self.linker.start_layer_wise_loading()
        self.until(lambda: len(self.p.executor.handles) == 3)
        self.p.executor.handles[2].complete_all()
        self.linker.layer_done_counter.set_consumer(next_index)
        self.linker.layer_done_counter.wait_until(1)
        self.assertTrue(self.linker.finish_layer_wise_loading(next_index))
        self.assertEqual(self.linker.stats["completed_LocalDram_pages"], 1)
        self.a.full_attn_allocator.free(next_targets)

    def test_mixed_peer_hbm_and_dram_sources_count_only_completed_reads(self):
        self.p.client.source_kinds = (Source.LocalHbm, Source.LocalDram)
        self.queued(keys=("peer", "dram"))
        self.assertNotIn("completed_LocalHbm_pages", self.linker.stats)
        index = self.linker.start_layer_wise_loading()
        self.until(lambda: bool(self.p.executor.handles))
        self.p.executor.handles[0].complete_all()
        self.linker.layer_done_counter.set_consumer(index)
        self.linker.layer_done_counter.wait_until(1)
        self.assertTrue(self.linker.finish_layer_wise_loading(index))
        self.assertEqual(self.p.get_load_stats()["completed_LocalHbm_pages"], 1)
        self.assertEqual(self.p.get_load_stats()["completed_LocalDram_pages"], 1)

    def test_g4_unknown_batch_does_not_stop_other_waiting_batches(self):
        self.queued("a")
        first = self.linker.start_layer_wise_loading()
        self.until(lambda: len(self.p.executor.handles) == 1)
        self.queued("b")
        ready = self.linker._pending["b"].ready = ManualReceipt()
        second = self.linker.start_layer_wise_loading()
        self.p.executor.handles[0].complete(0, Outcome.Unknown)
        self.linker.layer_done_counter.set_consumer(first)
        self.linker.layer_done_counter.wait_until(1)
        self.assertFalse(self.linker.finish_layer_wise_loading(first))
        self.assertIsNone(self.p._failed)
        ready.done = True
        self.until(lambda: len(self.p.executor.handles) == 2)
        self.p.executor.handles[1].complete_all()
        self.linker.layer_done_counter.set_consumer(second)
        self.linker.layer_done_counter.wait_until(1)
        self.assertTrue(self.linker.finish_layer_wise_loading(second))
        self.assertEqual(len(self.p.executor.handles), 2)

    def test_cancel_before_start_requires_never_submitted_and_drops_lookup_plan(self):
        self.queued()
        self.assertTrue(self.linker.cancel_queued_load("r"))
        self.assertEqual(self.linker.start_layer_wise_loading(), -1)
        self.linker.lookup("plan-only", [PoolTransfer(PoolName.KV, keys=["a"])])
        self.assertTrue(self.linker.cancel_queued_load("plan-only"))
        self.assertNotIn("plan-only", self.linker._plans)

    def test_cancel_after_start_cannot_claim_never_submitted(self):
        self.queued()
        index = self.linker.start_layer_wise_loading()
        self.until(lambda: bool(self.p.executor.handles))
        self.assertFalse(self.linker.cancel_queued_load("r"))
        self.p.executor.handles[0].complete_all()
        self.linker.layer_done_counter.set_consumer(index)
        self.linker.layer_done_counter.wait_until(1)
        self.assertTrue(self.linker.finish_layer_wise_loading(index))

    def test_request_retirement_discards_lookup_plan_even_without_queued_load(self):
        self.c.linker = UnifiedCacheLinkerWrapper(self.c, self.linker)
        self.linker.lookup("retired", [PoolTransfer(PoolName.KV, keys=["a"])])
        self.c.linker.release_request(CacheRequestHandle("retired", 1))
        self.assertNotIn("retired", self.linker._plans)

    def test_invalid_prepared_layers_cancel_holds_before_wrapper_aborts(self):
        self.linker.lookup("r", [PoolTransfer(PoolName.KV, keys=["a"])])
        ticket = self.p.reserve(CapacityShape("r", 1, 1, 0, 0, 4))
        with self.a.full_attn_allocator.reservation_scope(ticket, "load"):
            targets = self.storage.take_units(self.storage.owner, 4)
        self.p.client.layers = (999,)
        with self.assertRaisesRegex(ValueError, "layer geometry"):
            self.linker.load(
                "r", [PoolTransfer(PoolName.KV, device_indices=targets, keys=["a"])]
            )
        self.assertEqual(len(self.p.client.cancels), 1)
        self.assertFalse(self.linker._pending)
        self.p.release_page_ids(targets._kv_provider_page_ids, ready=CompletedReceipt())
        ticket.refund_unused()

    def test_provider_factory_reuses_client_executor_and_one_linker(self):
        again = self.p.create_linker(
            None,
            SimpleNamespace(token_to_kv_pool_allocator=self.a),
            components={ComponentType.FULL},
            forward_stream=None,
        )
        self.assertIs(again, self.linker)
        self.assertIs(again.client, self.p.client)
        self.assertIs(again.executor, self.p.executor)
        self.assertFalse(again.use_layer_events)

    def test_submit_exception_with_never_submitted_proof_is_quiesced_batch_failure(
        self,
    ):
        self.queued()
        self.p.executor.raise_on_submit = True
        index = self.linker.start_layer_wise_loading()
        self.linker.layer_done_counter.set_consumer(index)
        self.linker.layer_done_counter.wait_until(1)
        self.assertFalse(self.linker.finish_layer_wise_loading(index))
        self.assertIsNone(self.p._failed)
        self.assertEqual(len(self.p.client.cancels), 1)

    def test_submit_exception_without_quiescence_proof_retains_targets(self):
        self.queued()
        self.p.executor.raise_on_submit = self.p.executor.raise_after_start = True
        index = self.linker.start_layer_wise_loading()
        self.linker.layer_done_counter.set_consumer(index)
        with self.assertRaises(ProviderFailStop):
            self.linker.layer_done_counter.wait_until(0)
        self.assertTrue(self.p._retained)

    def test_linker_close_does_not_close_provider_client_or_executor(self):
        with (
            patch.object(
                self.p.client,
                "close",
                side_effect=AssertionError("shared client closed"),
            ),
            patch.object(
                self.p.executor,
                "close",
                side_effect=AssertionError("shared executor closed"),
            ),
        ):
            self.linker.close()
        self.assertFalse(self.linker._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
