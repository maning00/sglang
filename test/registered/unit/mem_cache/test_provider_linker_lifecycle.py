"""Provider load receipts, post-start failure and scheduler reset ordering."""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.kv_memory_provider import (
    CapacityShape,
    reservation_request_id,
)
from sglang.srt.mem_cache.static_kv_storage import StaticPoolShape
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    ExternalCacheHitMarker,
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kv_memory_provider import (
    ManualReceipt,
    build_provider_fixture,
    make_fake_linker,
    take_pages,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestProviderLinkerLifecycle(CustomTestCase):
    def setUp(self):
        scope = get_context().override_server_args()
        scope.install()
        self.addCleanup(scope.restore)

    def _load(self, backend, outcome):
        p, a, c, storage = build_provider_fixture(backend=backend)
        linker = make_fake_linker(p)
        linker.outcome = outcome
        c.linker = UnifiedCacheLinkerWrapper(c, linker)
        reqs = []
        for i in range(2):
            req = Req(
                rid=f"load-{i}",
                origin_input_text=None,
                origin_input_ids=array("q", range(4)),
                sampling_params=SamplingParams(max_new_tokens=0),
            )
            req.refresh_fill_ids()
            req.last_node = req.best_match_node = c.root_node_handle()
            req.kv.reservation_revision = 1
            req.kv.reservation = p.reserve(
                CapacityShape(reservation_request_id(req), 1, 1, 0, 4, 4)
            )
            c.linker.hit_markers[req.rid] = ExternalCacheHitMarker([f"{i + 1:064x}"], 0)
            loaded, node = c.linker.load_back(req)
            self.assertEqual(loaded, 4)
            reqs.append(req)
        self.assertEqual(p.accounting().in_use, 2)
        c.linker.start_layer_wise_loading()
        failed = c.finish_external_linker_loads(reqs)
        if outcome == "unknown":
            self.assertEqual(p.accounting().quarantined, 2)
        self.assertEqual(failed, reqs)
        self.assertTrue(all(req.skip_radix_cache_insert for req in failed))
        self.assertTrue(
            all(req.discard_output_reason.status_code == 503 for req in failed)
        )
        self.assertEqual(c.all_values_flatten().numel(), 0)
        return p.accounting()

    def test_c24_t02_post_start_batch_failure_and_unknown_match_both_cores(self):
        for outcome in ("failed", "unknown"):
            self.assertEqual(self._load("python", outcome), self._load("rust", outcome))

    def test_forward_receipt_gates_publication_with_unified_memory_disabled(self):
        p, a, _, storage = build_provider_fixture()
        ids = take_pages(a.full_attn_allocator, p, 1)
        done = ManualReceipt()
        storage.record_forward(
            done,
            written_tokens=ids,
            token_writes_unknown=False,
            req_pool_indices=None,
            state_slots=(),
        )
        handles = p.freeze_allocations(ids).handles(ids)
        from sglang.srt.mem_cache.kv_memory_provider import (
            CompletedReceipt,
            ResidentPage,
        )

        p.publish([ResidentPage("key", None, handles[0])], ready=CompletedReceipt())
        p.progress()
        self.assertFalse(p.commands)
        done.done = True
        p.progress()
        self.assertEqual(p.commands[0][0], "publish")

    def test_c09_scheduler_busy_reset_does_not_mutate_any_local_state(self):
        p, a, c, storage = build_provider_fixture()
        ids = take_pages(a.full_attn_allocator, p, 1)
        handles = p.freeze_allocations(ids).handles(ids)
        p.hold_io(handles, kind="remote-reader", writes=False, receipt=object())
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.token_to_kv_pool_allocator = a
        scheduler.req_to_token_pool = storage.requests
        scheduler.tree_cache = c
        scheduler.waiting_queue = []
        scheduler.running_batch = SimpleNamespace(reqs=[])
        scheduler.pause_state = SimpleNamespace(paused=False)
        scheduler.is_fully_idle = lambda **kwargs: True
        original_tree = c.tree_core
        scheduler.cur_batch_for_debug = marker = object()
        self.assertFalse(scheduler.flush_cache(empty_cache=False))
        self.assertIs(scheduler.cur_batch_for_debug, marker)
        self.assertIs(c.tree_core, original_tree)
        self.assertEqual(p.epoch.reset_seq, 0)

    def test_c10_scheduler_ready_reset_uses_approved_subset(self):
        p, a, c, storage = build_provider_fixture(pages=3)
        isolated = take_pages(a.full_attn_allocator, p, 1)
        h = p.freeze_allocations(isolated).handles(isolated)
        ticket = p.hold_io(h, kind="load-target", writes=True, receipt=object())
        a.full_attn_allocator.free(isolated)
        p.complete_io(ticket, outcome="unknown")
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.token_to_kv_pool_allocator = a
        scheduler.req_to_token_pool = storage.requests
        scheduler.tree_cache = c
        scheduler.waiting_queue = []
        scheduler.running_batch = SimpleNamespace(reqs=[])
        scheduler.pause_state = SimpleNamespace(paused=False)
        scheduler.is_fully_idle = lambda **kwargs: True
        scheduler.uocr_runtime = None
        scheduler.draft_worker = None
        scheduler.grammar_manager = SimpleNamespace(clear=lambda: None)
        scheduler.metrics_reporter = SimpleNamespace(
            reset_metrics=lambda: None, is_stats_logging_rank=False
        )
        scheduler.invariant_checker = SimpleNamespace(run=lambda phase: None)
        self.assertTrue(scheduler.flush_cache(empty_cache=False))
        self.assertEqual(a.full_attn_allocator.available_size(), 8)
        self.assertEqual(p.epoch.reset_seq, 1)
        self.assertEqual(p.accounting().quarantined, 1)
        self.assertIsNone(p.lease)

    def test_c28_physical_accounting_and_projection_detect_lost_or_duplicate_pages(
        self,
    ):
        from sglang.srt.environ import envs

        self.enterContext(envs.SGLANG_KV_PROVIDER_AUDIT_INTERVAL_SECONDS.override(0.0))
        from sglang.srt.disaggregation.mode import DisaggregationMode
        from sglang.srt.managers.scheduler_components.invariant_checker import (
            SchedulerInvariantChecker,
        )
        from sglang.srt.managers.scheduler_components.pool_stats_observer import (
            SchedulerPoolStatsObserver,
        )
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams
        from sglang.srt.mem_cache.radix_cache import RadixKey

        p, a, c, storage = build_provider_fixture()
        storage.bind_shape(
            StaticPoolShape(
                has_window=True,
                has_state=False,
                token_capacity=64,
                full_tokens_per_layer=64,
                swa_tokens_per_layer=64,
                window=8,
            )
        )
        observer = SchedulerPoolStatsObserver(
            tree_cache=c,
            storage=storage,
            token_to_kv_pool_allocator=a,
            session_controller=SimpleNamespace(sessions={}),
            hisparse_coordinator=None,
            enable_hisparse=False,
        )
        checker = SchedulerInvariantChecker(
            disaggregation_mode=DisaggregationMode.PREFILL,
            page_size=4,
            tree_cache=c,
            token_to_kv_pool_allocator=a,
            req_to_token_pool=storage.requests,
            pool_stats_observer=observer,
            scheduler_stage_metrics=SimpleNamespace(),
        )
        ids = take_pages(a.full_attn_allocator, p, 1)
        c.insert(InsertParams(key=RadixKey(array("q", range(4))), value=ids))
        p.progress()
        ticket = p.hold_io(
            p.freeze_allocations(ids).handles(ids),
            kind="read",
            writes=False,
            receipt=object(),
        )
        self.assertEqual(p.accounting().blocked_reclaim_estimate, 1)
        self.assertEqual(a.create_prefill_budget(c).remaining_total, 60)
        self.assertEqual(observer.get_pool_stats().provider_tree_evictable_tokens, 0)
        c.confirm_detach(c.offer_candidates(1))
        stats = observer.get_pool_stats()
        self.assertEqual(stats.full_available_size, 60)
        self.assertEqual(stats.provider_pages["retiring"], 1)
        from sglang.srt.observability.metrics_collector import SchedulerStats

        metrics = SchedulerStats()
        stats.update_scheduler_stats(metrics)
        self.assertEqual(metrics.kv_provider_pages["retiring"], 1)
        self.assertEqual(
            metrics.kv_available_tokens
            + metrics.kv_used_tokens
            + metrics.kv_evictable_tokens,
            64,
        )
        self.assertEqual(checker._check_full_pool(stats), (False, ""))
        p.complete_io(ticket, outcome="completed")
        p.progress()
        lost = take_pages(a.full_attn_allocator, p, 1)
        self.assertTrue(checker._check_full_pool(observer.get_pool_stats())[0])
        self.assertTrue(p.validate_projection(torch.cat([lost, lost])))

    def test_provider_admission_metrics_keep_empty_rounds_and_request_waits(self):
        from functools import partial

        from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

        from sglang.srt.managers.scheduler_components.metrics_reporter import (
            SchedulerMetricsReporter,
        )
        from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector

        registry = CollectorRegistry()
        collector = SchedulerMetricsCollector.__new__(SchedulerMetricsCollector)
        collector.labels = {"model": "provider-cpu"}
        collector._counter_cls = partial(Counter, registry=registry)
        collector._gauge_cls = partial(Gauge, registry=registry)
        collector._histogram_cls = partial(Histogram, registry=registry)
        collector.kv_provider_admission = None
        reporter = SchedulerMetricsReporter.__new__(SchedulerMetricsReporter)
        reporter.enable_metrics = True
        reporter.metrics_collector = collector
        reporter.record_provider_admission_round(
            {
                "short": 1,
                "requests": 2,
                "wait_ns": 3000,
                "request_wait_ns": [1000, 2000],
            },
            0,
        )
        reporter.record_provider_admission_round({}, 0)
        self.assertEqual(
            registry.get_sample_value(
                "sglang:kv_provider_empty_rounds", collector.labels
            ),
            2,
        )
        reporter.record_provider_admission_round(
            {"fast": 1, "requests": 1, "wait_ns": 500, "request_wait_ns": [500]}, 1
        )
        self.assertEqual(
            registry.get_sample_value(
                "sglang:kv_provider_empty_rounds", collector.labels
            ),
            0,
        )
        self.assertEqual(
            registry.get_sample_value(
                "sglang:kv_provider_round_wait_seconds_count", collector.labels
            ),
            3,
        )
        self.assertEqual(
            registry.get_sample_value(
                "sglang:kv_provider_request_wait_seconds_count", collector.labels
            ),
            3,
        )
        self.assertEqual(
            registry.get_sample_value(
                "sglang:kv_provider_admission_total",
                dict(collector.labels, event="admitted"),
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
