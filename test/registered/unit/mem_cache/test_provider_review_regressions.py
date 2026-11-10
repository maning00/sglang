"""S1F regressions: preserve real admission, allocator, tree and receipt paths."""

import itertools
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch
from test_kv_memory_provider import CONFIG
from test_provider_prefill_reservation import adder, request

from sglang.srt.arg_groups.hicache_hook import handle_hicache
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import PhysicalSendReceipt, SendRecord
from sglang.srt.mem_cache.allocation import _room_in
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.kv_memory_provider import (
    CapacityScope,
    CapacityShape,
    ProviderFailStop,
    reservation_request_id,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components import LinkerTransferPhase
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    ExternalCacheHitMarker,
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kv_memory_provider import (
    build_provider_fixture,
    make_fake_linker,
    take_pages,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestProviderReviewRegressions(CustomTestCase):
    def setUp(self):
        self.enterContext(get_context().override_server_args())

    def test_s_p1_extend_uses_real_room_in_without_evicting_reserved_full(self):
        for enabled in (True, False):
            with self.subTest(provider=enabled):
                p, a, c, storage = build_provider_fixture(provider_enabled=enabled)
                source = a.full_attn_allocator
                cached = take_pages(source, p, 1) if enabled else source.alloc(4)
                c.insert(
                    InsertParams(
                        key=RadixKey(array("q", range(100, 104))), value=cached
                    )
                )
                req = request(c)
                if enabled:
                    req.kv.reservation_revision = 1
                    req.kv.reservation = p.reserve(
                        CapacityShape(reservation_request_id(req), 1, 0, 2, 0, 8)
                    )
                before = c.all_values_flatten().clone()
                out = storage.take_extend(
                    rows=None,
                    prefix_lens=torch.tensor([0]),
                    prefix_lens_cpu=torch.tensor([0]),
                    seq_lens=torch.tensor([8]),
                    seq_lens_cpu=torch.tensor([8]),
                    last_loc=torch.tensor([-1]),
                    num_tokens=8,
                    make_room=_room_in(c),
                    tree_cache=c,
                    batch=SimpleNamespace(reqs=[req]),
                )
                self.assertEqual(out.numel(), 8)
                self.assertTrue(torch.equal(c.all_values_flatten(), before))

    def test_s_p1_load_short_ticket_does_not_evict_tree(self):
        p, a, c, storage = build_provider_fixture()
        ids = take_pages(a.full_attn_allocator, p, 1)
        c.insert(InsertParams(key=RadixKey(array("q", range(100, 104))), value=ids))
        ticket = p.reserve(CapacityShape("load", 1, 1, 0, 0, 4))
        before = c.all_values_flatten().clone()
        with a.full_attn_allocator.reservation_scope(ticket, "load"):
            with self.assertRaises(RuntimeError):
                c.components[ComponentType.FULL].build_external_linker_transfer(
                    LinkerTransferPhase.LOAD, None, ["a", "b"]
                )
        self.assertTrue(torch.equal(c.all_values_flatten(), before))
        ticket.refund_unused()

    def test_s_p1_non_ring_extend_reclaims_only_the_short_window(self):
        p, a, c, storage = build_provider_fixture(
            pages=4, ring=False, components=(ComponentType.FULL, ComponentType.SWA)
        )
        old = take_pages(a.full_attn_allocator, p, 1)
        a.set_full_to_swa_mapping(old, a.swa_attn_allocator.alloc(4))
        c.insert(InsertParams(key=RadixKey(array("q", range(100, 104))), value=old))
        occupied = a.swa_attn_allocator.alloc(12)
        req = request(c, 4)
        req.kv.reservation_revision = 1
        req.kv.reservation = p.reserve(
            CapacityShape(reservation_request_id(req), 1, 0, 1, 0, 4)
        )
        with patch.object(c, "evict_for_alloc", wraps=c.evict_for_alloc) as evict:
            out = storage.take_extend(
                rows=None,
                prefix_lens=torch.tensor([0]),
                prefix_lens_cpu=torch.tensor([0]),
                seq_lens=torch.tensor([4]),
                seq_lens_cpu=torch.tensor([4]),
                last_loc=torch.tensor([-1]),
                num_tokens=4,
                make_room=_room_in(c),
                tree_cache=c,
                batch=SimpleNamespace(reqs=[req]),
            )
        self.assertEqual(out.numel(), 4)
        self.assertEqual(evict.call_count, 1)
        demand = evict.call_args.args[0]
        self.assertEqual(demand.full_tokens, 0)
        self.assertEqual(tuple(demand.window_tokens.values()), (4,))
        # The existing SWA leaf cascade may free its co-owned FULL page;
        # the allocation request itself must charge only the short window.
        self.assertEqual(a.swa_available_size(), 0)
        a.swa_attn_allocator.free(occupied)

    def test_s_p2_1_short_without_progress_does_not_spin(self):
        p, a, c, _ = build_provider_fixture(pages=1)
        held = p.reserve(CapacityShape("other", 1, 0, 1, 0, 4))
        ticks = itertools.count()
        stats = {}
        with (
            patch(
                "sglang.srt.mem_cache.kv_memory_provider.time.monotonic_ns",
                side_effect=lambda: next(ticks),
            ),
            patch("sglang.srt.mem_cache.kv_memory_provider.time.sleep") as sleep,
        ):
            with CapacityScope(
                p, c, CapacityShape("new", 1, 0, 1, 0, 4), deadline_ns=20, stats=stats
            ) as scope:
                self.assertFalse(scope.ready)
            self.assertLessEqual(stats["attempts"], 2)
            self.assertEqual(sleep.call_count, 0)
        held.refund_unused()

    def test_s_p2_2_chunked_insert_publishes_and_freezes_only_new_pages(self):
        for backend in ("python", "rust"):
            with self.subTest(backend=backend):
                p, a, c, _ = build_provider_fixture(backend=backend)
                records, frozen = [], []
                publish, freeze = p.publish, p.freeze_allocations

                def publishing(pages, **kw):
                    records.extend(pages)
                    return publish(pages, **kw)

                def freezing(indices, **kw):
                    if kw.get("context") is None:
                        frozen.append(indices.numel())
                    return freeze(indices, **kw)

                values = torch.empty(0, dtype=torch.int64)
                with (
                    patch.object(p, "publish", side_effect=publishing),
                    patch.object(p, "freeze_allocations", side_effect=freezing),
                ):
                    for end in (4, 8, 12):
                        fresh = take_pages(a.full_attn_allocator, p, 1)
                        values = torch.cat((values, fresh))
                        c.insert(
                            InsertParams(
                                key=RadixKey(array("q", range(end))),
                                value=values,
                                prev_prefix_len=end - 4,
                            )
                        )
                        p.progress()
                self.assertEqual(len(records), 3)
                self.assertEqual(frozen, [4, 4, 4])

    def test_s_p2_3_failed_room_before_worker_seal_is_not_unknown(self):
        record = SendRecord()
        manager = SimpleNamespace(
            request_status={7: KVPoll.Transferring}, _send_records={7: record}
        )
        receipt = PhysicalSendReceipt(manager, 7, record, 0)
        receipt.bind()
        receipt.finish_enqueue()
        manager.request_status[7] = KVPoll.Failed
        self.assertIsNone(receipt.poll())
        receipt.seal()
        self.assertEqual(receipt.poll(), "never_submitted")

    def test_s_p2_4_unknown_io_allows_reset_and_keeps_quarantine_backing(self):
        p, a, c, storage = build_provider_fixture(pages=2)
        ids = take_pages(a.full_attn_allocator, p, 1)
        h = p.freeze_allocations(ids).handles(ids)
        resource = object()
        ticket = p.hold_io(h, kind="load-target", writes=True, receipt=resource)
        a.full_attn_allocator.free(ids)
        p.complete_io(ticket, outcome="unknown")
        with storage.reset_scope() as ready:
            self.assertTrue(ready)
            self.assertNotIn(h[0].page_id, p.lease.approved_pages)
            c.reset()
            a.clear()
        self.assertEqual(p.accounting().quarantined, 1)
        self.assertEqual(a.full_available_size(), 4)
        self.assertIs(p.quarantine_resources[0][1], resource)

    def test_s_p2_5_handle_hicache_rejects_explicit_mooncake(self):
        args = ServerArgs(model_path="dummy", kv_memory_provider="mooncake")
        with self.assertRaisesRegex(ValueError, "explicit|environment"):
            handle_hicache(args)

    def test_p3_reset_verdict_failure_aborts_sealed_lease(self):
        p, a, c, storage = build_provider_fixture()

        def fail(_):
            raise RuntimeError("collective failure")

        with self.assertRaises(ProviderFailStop):
            with storage.reset_scope(verdict=fail):
                self.fail("reset body entered")
        self.assertIsNone(p.lease)

    def test_p3_reset_abort_failure_is_fail_stop(self):
        p, a, c, storage = build_provider_fixture()
        with (
            patch.object(p, "abort_reset", side_effect=RuntimeError("channel lost")),
            self.assertRaises(ProviderFailStop),
        ):
            with storage.reset_scope(
                verdict=lambda _: (_ for _ in ()).throw(RuntimeError("collective"))
            ):
                self.fail("body entered")
        self.assertIsNone(a.full_attn_allocator._reset_lease)

    def test_p3_empty_provider_poll_does_not_gather_objects(self):
        _, _, c, _ = build_provider_fixture()
        with patch.object(c, "_provider_gather", wraps=c._provider_gather) as gather:
            c.check_provider_events()
        self.assertEqual(gather.call_count, 0)

    def test_p3_budget_collective_is_snapshotted_once(self):
        _, a, c, _ = build_provider_fixture()
        with patch.object(
            c, "provider_min_tokens", wraps=c.provider_min_tokens
        ) as minimum:
            budget = a.create_prefill_budget(c)
            for _ in range(4):
                _ = budget.remaining_total, budget.remaining_current
        self.assertEqual(minimum.call_count, 1)

    def test_p3_successful_load_keeps_its_extend_ticket(self):
        p, a, c, _ = build_provider_fixture()
        c.linker = UnifiedCacheLinkerWrapper(c, make_fake_linker(p))
        req = request(c)
        req.host_hit_length = 4
        c.linker.hit_markers[req.rid] = ExternalCacheHitMarker(["a" * 64], 0)
        add = adder(a, c)
        add.add_one_req(req, False, None)
        self.assertEqual(add.can_run_list, [req])
        self.assertEqual(len(p.shapes), 1)
        self.assertEqual(req.kv.reservation.remaining("extend"), 1)

    def test_p3_layout_guards_run_through_handle_hicache(self):
        for kw in (
            {"enable_encoder_swa_bounded_replay": True},
            {"enable_memory_saver": True},
        ):
            with self.subTest(flags=kw):
                args = ServerArgs(
                    model_path="dummy",
                    kv_memory_provider="dynamic",
                    kv_memory_provider_config=CONFIG,
                    disaggregation_mode="prefill",
                    disaggregation_transfer_backend="mori",
                    page_size=256,
                    **kw,
                )
                with self.assertRaisesRegex(ValueError, "encoder|memory.saver"):
                    handle_hicache(args)

    def test_g8_cp_layout_guard_runs_through_handle_hicache(self):
        args = ServerArgs(
            model_path="dummy",
            kv_memory_provider="dynamic",
            kv_memory_provider_config=CONFIG,
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="mori",
            page_size=256,
            enable_prefill_cp=True,
        )
        with self.assertRaisesRegex(ValueError, "CP padded|CP.*layout"):
            handle_hicache(args)

    def test_s_p2_2_swa_replacement_bumps_revision_and_plain_overlap_is_quiet(self):
        from sglang.srt.mem_cache.unified_cache.cache_action import ResidencyAction

        sequences = []
        for backend in ("python", "rust"):
            p, a, c, _ = build_provider_fixture(
                backend=backend,
                ring=False,
                components=(ComponentType.FULL, ComponentType.SWA),
            )
            source = a.full_attn_allocator
            old = take_pages(source, p, 2)
            swa = a.swa_attn_allocator.alloc(8)
            a.set_full_to_swa_mapping(old, swa)
            a.free_swa_segment(old, start_pos=0)
            key = RadixKey(array("q", range(8)))
            c.insert(
                InsertParams(
                    key=key, value=old, component_evicted_seqlens={ComponentType.SWA: 8}
                )
            )
            p.progress()
            fresh = take_pages(source, p, 2)
            a.set_full_to_swa_mapping(fresh, a.swa_attn_allocator.alloc(8))
            actions = []
            original = c._apply_cache_actions

            def apply(items):
                actions.extend(
                    (x.reason, x.node_revision, x.keys, x.indices.tolist())
                    for x in items
                    if isinstance(x, ResidencyAction)
                )
                return original(items)

            with patch.object(c, "_apply_cache_actions", side_effect=apply):
                c.insert(InsertParams(key=key, value=fresh))
                p.progress()
                c.insert(InsertParams(key=key, value=fresh, prev_prefix_len=8))
            dedup = [x for x in actions if x[0] == "dedup"]
            self.assertEqual(len(dedup), 1)
            self.assertEqual(dedup[0][1], 1)
            self.assertEqual(c.all_values_flatten().tolist(), fresh.tolist())
            sequences.append(actions)
        self.assertEqual(sequences[0], sequences[1])

    def test_p3_live_freeze_hold_and_publish_reject_returned_or_quarantined_pages(self):
        from sglang.srt.mem_cache.kv_memory_provider import (
            CompletedReceipt,
            ResidentPage,
        )

        p, a, c, _ = build_provider_fixture()
        source = a.full_attn_allocator
        ids = take_pages(source, p, 1)
        h = p.freeze_allocations(ids).handles(ids)
        ticket = p.hold_io(h, kind="pd-send", writes=False, receipt=object())
        source.free(ids)
        for action in (
            lambda: p.freeze_allocations(ids),
            lambda: p.hold_io(h, kind="read", writes=False, receipt=object()),
        ):
            with self.assertRaises(RuntimeError):
                action()
        p.complete_io(ticket, outcome="unknown")
        for action in (
            lambda: p.freeze_allocations(ids),
            lambda: p.hold_io(h, kind="read", writes=False, receipt=object()),
            lambda: p.publish(
                [ResidentPage("bad", None, h[0])], ready=CompletedReceipt()
            ),
        ):
            with self.assertRaises(RuntimeError):
                action()

    def test_p3_forward_and_pd_use_existing_cpu_page_ids(self):
        from sglang.test.kv_memory_provider import ManualReceipt

        p, a, c, storage = build_provider_fixture()
        ids = take_pages(a.full_attn_allocator, p, 1)
        page = int(ids[0]) // 4

        class DeviceIndices:
            _kv_provider_page_ids = (page,)

            def detach(self):
                raise AssertionError("unnecessary D2H")

        done = ManualReceipt(True)
        storage.record_forward(
            done,
            written_tokens=DeviceIndices(),
            token_writes_unknown=False,
            req_pool_indices=None,
            state_slots=(),
        )
        self.assertEqual(getattr(ids, "_kv_provider_page_ids", None), (page,))
        with patch.object(
            torch, "as_tensor", wraps=torch.as_tensor
        ) as device_roundtrip:
            storage.record_network_transfer(
                request(c), pages=[page], page_size=4, writes=False
            )
        self.assertEqual(device_roundtrip.call_count, 0)

    def test_p3_confirm_verdict_is_one_vector_collective(self):
        p, a, c, _ = build_provider_fixture()
        for start in (0, 100):
            ids = take_pages(a.full_attn_allocator, p, 1)
            c.insert(
                InsertParams(
                    key=RadixKey(array("q", range(start, start + 4))), value=ids
                )
            )
        candidates = c.offer_candidates(2)
        self.assertEqual(len(candidates), 2)
        with patch.object(
            c, "_all_reduce_attn_groups", wraps=c._all_reduce_attn_groups
        ) as reduction:
            results = c.confirm_detach(candidates)
        self.assertTrue(all(r.status == "detached" for r in results))
        self.assertEqual(reduction.call_count, 1)

    def test_p3_reset_authorization_failure_aborts_before_mutation(self):
        p, a, c, storage = build_provider_fixture()
        with patch.object(
            a.full_attn_allocator,
            "authorize_reset",
            side_effect=RuntimeError("invalid approval"),
        ):
            with self.assertRaises(ProviderFailStop):
                with storage.reset_scope():
                    self.fail("body entered")
        self.assertIsNone(p.lease)

    def test_p3_cuda_device_without_ordinal_resolves_current_device(self):
        from sglang.srt.mem_cache.kv_memory_provider_factory import (
            get_or_create_kv_memory_provider,
        )
        from sglang.srt.runtime_context import get_resources

        with get_resources().override(kv_memory_providers={}):
            with patch("torch.cuda.current_device", return_value=0):
                provider = get_or_create_kv_memory_provider(
                    device="cuda", config=CONFIG
                )
                self.assertEqual(provider.device, "cuda:0")
                self.assertIs(
                    provider,
                    get_or_create_kv_memory_provider(device="cuda:0", config=CONFIG),
                )

    def test_g1b_kv_and_bytes_provider_use_nonzero_current_cuda_device(self):
        from sglang.srt.mem_cache.kv_memory_provider_factory import (
            get_or_create_kv_memory_provider,
            provider_mem_pool,
        )
        from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool
        from sglang.srt.runtime_context import get_resources

        for kind in ("kv", "bytes"):
            with (
                self.subTest(kind=kind),
                get_resources().override(kv_memory_providers={}),
            ):
                config = {**CONFIG, "kind": kind}
                with patch("torch.cuda.current_device", return_value=5):
                    provider = get_or_create_kv_memory_provider(
                        device="cuda", config=config
                    )
                    explicit = get_or_create_kv_memory_provider(
                        device="cuda:5", config=config
                    )
                    self.assertIs(provider, explicit)
                    self.assertEqual(provider.device, "cuda:5")
                    self.assertEqual(
                        set(get_resources().kv_memory_providers), {"cuda:5"}
                    )
                    self.assertIs(
                        provider_mem_pool(provider, "cuda"),
                        provider_mem_pool(provider, "cuda:5"),
                    )
                    with get_context().override_server_args(
                        kv_memory_provider="dynamic", kv_memory_provider_config=config
                    ):
                        enabled, pool, _ = maybe_init_custom_mem_pool("cuda")
                    self.assertTrue(enabled)
                    self.assertIs(pool, provider_mem_pool(provider, "cuda:5"))
                with patch("torch.cuda.current_device", return_value=3):
                    other = get_or_create_kv_memory_provider(
                        device="cuda", config=config
                    )
                    self.assertEqual(other.device, "cuda:3")
                    self.assertIsNot(other, provider)
                    self.assertEqual(
                        set(get_resources().kv_memory_providers), {"cuda:3", "cuda:5"}
                    )

    def test_p3_off_flush_never_constructs_static_storage(self):
        from sglang.srt.managers.scheduler import Scheduler

        p, a, c, storage = build_provider_fixture(provider_enabled=False)
        sched = Scheduler.__new__(Scheduler)
        sched.token_to_kv_pool_allocator, sched.req_to_token_pool, sched.tree_cache = (
            a,
            storage.requests,
            c,
        )
        sched.pause_state = SimpleNamespace(paused=False)
        sched.is_fully_idle = lambda **kw: True
        sched.uocr_runtime = sched.draft_worker = None
        sched.grammar_manager = SimpleNamespace(clear=lambda: None)
        sched.metrics_reporter = SimpleNamespace(
            reset_metrics=lambda: None, is_stats_logging_rank=False
        )
        sched.invariant_checker = SimpleNamespace(run=lambda phase: None)
        with patch(
            "sglang.srt.managers.scheduler.kv_storage_of",
            side_effect=AssertionError("provider-off storage construction"),
        ):
            self.assertTrue(sched.flush_cache(empty_cache=False))

    def test_s_p2_1_admitted_batch_never_waits_for_retiring_pages(self):
        from dataclasses import replace

        p, a, c, _ = build_provider_fixture(pages=6)
        ids = take_pages(a.full_attn_allocator, p, 4)
        c.insert(InsertParams(key=RadixKey(array("q", range(100, 116))), value=ids))
        p.progress()
        h = p.freeze_allocations(ids).handles(ids)
        io = p.hold_io(h, kind="remote-reader", writes=False, receipt=object())
        accounting = p.accounting
        # The node summary is only an estimate and may lag an Acquire.
        with patch.object(
            p,
            "accounting",
            side_effect=lambda: replace(accounting(), blocked_reclaim_estimate=0),
        ):
            add = adder(a, c)
            first = request(c, 4)
            add.add_one_req(first, False, None)
            self.assertEqual(add.can_run_list, [first])
            with patch(
                "sglang.srt.mem_cache.kv_memory_provider.time.sleep",
                wraps=__import__("time").sleep,
            ) as sleep:
                add.add_one_req(request(c, 8), False, None)
            self.assertEqual(sleep.call_count, 0)
            self.assertEqual(add.can_run_list, [first])
        first.kv.reservation.refund_unused()
        p.complete_io(io, outcome="completed")

    def test_p3_c16_admission_decisions_match_main(self):
        outcomes = []
        for enabled in (False, True):
            p, a, c, _ = build_provider_fixture(pages=8, provider_enabled=enabled)
            add = adder(a, c)
            results = [add.add_one_req(request(c, n), False, None) for n in (8, 8, 4)]
            outcomes.append(
                (
                    results,
                    len(add.can_run_list),
                    add.rem_input_tokens,
                    add.memory_budget.remaining_total,
                )
            )
            for req in add.can_run_list:
                if req.kv.reservation:
                    req.kv.reservation.refund_unused()
        self.assertEqual(outcomes[0], outcomes[1])

    def test_p3_preserves_tree_ledger_checks_with_valid_held_pages(self):
        from sglang.srt.environ import envs

        self.enterContext(envs.SGLANG_KV_PROVIDER_AUDIT_INTERVAL_SECONDS.override(0.0))
        from sglang.srt.disaggregation.mode import DisaggregationMode
        from sglang.srt.managers.scheduler_components.invariant_checker import (
            SchedulerInvariantChecker,
        )
        from sglang.srt.managers.scheduler_components.pool_stats_observer import (
            SchedulerPoolStatsObserver,
        )
        from sglang.srt.mem_cache.static_kv_storage import StaticPoolShape

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
        candidate = c.offer_candidates(1)[0]
        lock = c.lock(candidate.node_id)
        self.assertFalse(checker._check_full_pool(observer.get_pool_stats())[0])
        c.unlock(lock)
        c.tree_core.component_evictable_size_[ComponentType.FULL] += 4
        self.assertTrue(checker._check_full_pool(observer.get_pool_stats())[0])

    def test_p3_published_local_hold_delays_retire_and_released_waits_guard(self):
        from sglang.srt.mem_cache.kv_memory_provider import (
            CompletedReceipt,
            ResidentPage,
        )
        from sglang.test.kv_memory_provider import ManualReceipt

        p, a, c, _ = build_provider_fixture()
        ids = take_pages(a.full_attn_allocator, p, 1)
        h = p.freeze_allocations(ids).handles(ids)
        p.publish([ResidentPage("key", None, h[0])], ready=CompletedReceipt())
        p.progress()
        io = p.hold_io(h, kind="pd-send", writes=False, receipt=object())
        guard = ManualReceipt()
        a.full_attn_allocator.set_reuse_guard_event(guard)
        a.full_attn_allocator.free(ids)
        self.assertFalse([x for x in p.commands if x[0] == "retire"])
        p.complete_io(io, outcome="completed")
        p.progress()
        self.assertEqual(p.accounting().pending_free, 1)
        self.assertEqual(a.full_available_size(), 60)
        guard.done = True
        p.progress()
        self.assertEqual(a.full_available_size(), 64)

    def test_s_p2_1_waits_only_for_existing_physical_progress(self):
        from sglang.srt.mem_cache.kv_memory_provider import (
            CompletedReceipt,
            ResidentPage,
        )
        from sglang.test.kv_memory_provider import ManualReceipt

        p, a, c, _ = build_provider_fixture(pages=1)
        ids = take_pages(a.full_attn_allocator, p, 1)
        h = p.freeze_allocations(ids).handles(ids)
        p.publish([ResidentPage("key", None, h[0])], ready=CompletedReceipt())
        p.progress()
        guard = ManualReceipt()
        a.full_attn_allocator.set_reuse_guard_event(guard)
        a.full_attn_allocator.free(ids)

        def finish(_):
            guard.done = True

        with patch(
            "sglang.srt.mem_cache.kv_memory_provider.time.sleep", side_effect=finish
        ):
            with CapacityScope(
                p,
                c,
                CapacityShape("next", 2, 0, 1, 0, 4),
                deadline_ns=__import__("time").monotonic_ns() + 100_000_000,
            ) as scope:
                self.assertTrue(scope.ready)

    def test_p3_successful_load_adjustment_failure_is_fail_stop(self):
        from sglang.srt.mem_cache.kv_memory_provider import Short

        p, a, c, _ = build_provider_fixture()
        c.linker = UnifiedCacheLinkerWrapper(c, make_fake_linker(p))
        req = request(c)
        req.host_hit_length = 4
        c.linker.hit_markers[req.rid] = ExternalCacheHitMarker(["a" * 64], 0)
        reserve, calls = p.reserve, []

        def refusing(shape):
            calls.append(shape)
            if len(calls) > 1:
                return Short(1, False)
            result = reserve(shape)
            result.revise = lambda shape: False
            return result

        add = adder(a, c)
        with patch.object(p, "reserve", side_effect=refusing):
            with self.assertRaises(ProviderFailStop):
                add.add_one_req(req, False, None)
        self.assertFalse(add.can_run_list)
        self.assertTrue(p.io)  # queued target remains held until fail-stop

    def test_p3_wire_candidate_keys_are_immutable_before_event_dedup(self):
        from sglang.srt.mem_cache.kv_memory_provider import (
            DetachCandidate,
            ProviderTreeEvent,
        )

        p, a, c, _ = build_provider_fixture()
        ids = take_pages(a.full_attn_allocator, p, 1)
        c.insert(InsertParams(key=RadixKey(array("q", range(4))), value=ids))
        candidate = c.offer_candidates(1)[0]
        wire = DetachCandidate(
            candidate.node_id,
            candidate.revision,
            list(candidate.keys),
            candidate.handles,
        )
        p.events.append(ProviderTreeEvent("detach", "detach", candidates=(wire,)))
        c.check_provider_events()
        self.assertIsInstance(wire.keys, tuple)
        self.assertEqual(p.event_results[0][1][0].status, "detached")

    def test_p3_physical_conservation_is_reported_once(self):
        from dataclasses import replace

        from sglang.srt.disaggregation.mode import DisaggregationMode
        from sglang.srt.managers.scheduler_components.invariant_checker import (
            SchedulerInvariantChecker,
        )
        from sglang.srt.managers.scheduler_components.pool_stats_observer import (
            SchedulerPoolStatsObserver,
        )
        from sglang.srt.mem_cache.static_kv_storage import StaticPoolShape

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
        messages = []
        checker.reporter = SimpleNamespace(
            report=lambda severity, message: messages.append(message)
        )
        broken = replace(p.accounting(), cached=15)
        with patch.object(p, "accounting", return_value=broken):
            checker._check_idle_pools()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].count("provider page conservation"), 1)


if __name__ == "__main__":
    unittest.main()
