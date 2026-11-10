"""FULL release convergence, local engine ownership and reset approvals."""

import unittest
from array import array
from dataclasses import replace
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.kv_memory_provider import (
    CapacityShape,
    CompletedReceipt,
    ResidentPage,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kv_memory_provider import (
    ManualReceipt,
    build_provider_fixture,
    take_pages,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestProviderPageSource(CustomTestCase):
    def setUp(self):
        scope = get_context().override_server_args()
        scope.install()
        self.addCleanup(scope.restore)

    def test_c11_release_family_converges_once(self):
        for method in (
            "free",
            "free_segment",
            "free_page_ids",
            "free_group_end",
            "free_full_segments",
            "release_taken",
        ):
            with self.subTest(method=method):
                p, a, _, storage = build_provider_fixture()
                source = a.full_attn_allocator
                ids = take_pages(source, p, 2)
                handles = p.freeze_allocations(ids).handles(ids)
                p.publish(
                    [ResidentPage(str(i), None, h) for i, h in enumerate(handles)],
                    ready=CompletedReceipt(),
                )
                p.progress()
                p.auto_release = False
                if method == "free":
                    source.free(ids)
                elif method == "free_segment":
                    source.free_segment(ids, start_pos=0)
                elif method == "free_page_ids":
                    source.free_page_ids(ids[::4] // 4)
                elif method == "free_group_end":
                    source.free_group_begin()
                    source.free(ids[:4])
                    source.free_page_ids(ids[4::4] // 4)
                    source.free_group_end()
                elif method == "free_full_segments":
                    a.free_full_segments([(ids[:4], 0), (ids[4:], 4)])
                else:
                    storage.release_taken("kv", ids)
                self.assertEqual(p.accounting().retiring, 2)
                self.assertEqual(source.num_staged_pages, 0)
                retires = [(op, h) for tag, op, h in p.commands if tag == "retire"]
                self.assertEqual(len(retires), 2)
                for op, h in retires:
                    self.assertTrue(p.released(op, h))
                    self.assertFalse(p.released(op, h))
                self.assertEqual(source.available_size(), a.size_full)
                with self.assertRaisesRegex(RuntimeError, "duplicate physical return"):
                    source.free(ids)

    def test_c03_c07_cancel_before_send_pending_io_and_guard(self):
        p, a, _, storage = build_provider_fixture(pages=2)
        source = a.full_attn_allocator
        ids = take_pages(source, p, 1)
        handles = p.freeze_allocations(ids).handles(ids)
        ready, guard = ManualReceipt(), ManualReceipt()
        p.publish([ResidentPage("key", None, handles[0])], ready=ready)
        ticket = p.hold_io(handles, kind="load-target", writes=True, receipt=object())
        source.set_reuse_guard_event(guard)
        storage.release_taken("kv", ids)
        self.assertEqual(p.accounting().pending_free, 1)
        self.assertEqual(source.available_size(), 4)
        ready.done = True
        p.complete_io(ticket, outcome="completed")
        p.progress()
        self.assertEqual(p.accounting().pending_free, 1)
        guard.done = True
        p.progress()
        self.assertEqual(p.accounting().cached, 2)
        self.assertEqual(p.commands, [])
        old = handles[0]
        new_ids = take_pages(source, p, 1)
        new = p.freeze_allocations(new_ids).handles(new_ids)[0]
        self.assertGreater(new.generation, old.generation)

    def test_publish_enqueue_and_return_are_one_atomic_handoff(self):
        p, a, _, _ = build_provider_fixture(pages=1)
        source = a.full_attn_allocator
        ids = take_pages(source, p, 1)
        handle = p.freeze_allocations(ids).handles(ids)[0]
        p.publish([ResidentPage("key", None, handle)], ready=CompletedReceipt())
        entered, proceed, returning = Event(), Event(), Event()

        def sending():
            entered.set()
            assert proceed.wait(5)

        def freeing():
            returning.set()
            source.free(ids)

        p.before_send = sending
        p.auto_release = False
        publisher = Thread(target=p.progress)
        publisher.start()
        self.assertTrue(entered.wait(5))
        freer = Thread(target=freeing)
        freer.start()
        self.assertTrue(returning.wait(5))
        proceed.set()
        publisher.join(5)
        freer.join(5)
        self.assertFalse(publisher.is_alive() or freer.is_alive())
        self.assertEqual([x[0] for x in p.commands], ["publish", "retire"])
        self.assertEqual(source.available_size(), 0)
        op, (h, _) = next(iter(p.retires.items()))
        self.assertTrue(p.released(op, h))
        self.assertFalse(p.released(op, h))

    def test_failed_control_session_stops_reserved_and_cached_allocations(self):
        from sglang.srt.mem_cache.kv_memory_provider import ProviderFailStop

        p, a, _, _ = build_provider_fixture()
        ticket = p.reserve(CapacityShape("request", 1, 1, 0, 0, 4))
        p.close()
        with a.full_attn_allocator.reservation_scope(ticket, "load"):
            with self.assertRaises(ProviderFailStop):
                a.full_attn_allocator.alloc(4)

    def test_c09_c10_reset_busy_and_approved_set_before_clear(self):
        p, a, cache, storage = build_provider_fixture(pages=3)
        source = a.full_attn_allocator
        ids = take_pages(source, p, 1)
        h = p.freeze_allocations(ids).handles(ids)
        ticket = p.hold_io(h, kind="pd-send", writes=False, receipt=object())
        before = a.full_to_swa_index_mapping.clone()
        old_epoch = p.epoch
        with storage.reset_scope() as ready:
            self.assertFalse(ready)
        self.assertEqual(p.epoch, old_epoch)
        self.assertIsNone(p.lease)
        for operation in (
            a.clear,
            lambda: source.resize(SimpleNamespace(max_total_num_tokens=4)),
            lambda: a.resize(
                SimpleNamespace(full_max_total_num_tokens=4, swa_max_total_num_tokens=4)
            ),
        ):
            with self.assertRaises(RuntimeError):
                operation()
            self.assertEqual(a.size_full, 12)
            self.assertTrue(torch.equal(before, a.full_to_swa_index_mapping))
        from sglang.srt.disaggregation.role_switch import teardown_disaggregation

        with self.assertRaisesRegex(RuntimeError, "role switch"):
            teardown_disaggregation(SimpleNamespace(token_to_kv_pool_allocator=a))
        p.complete_io(ticket, outcome="completed")
        source.free(ids)
        isolated = take_pages(source, p, 1)
        isolated_handle = p.freeze_allocations(isolated).handles(isolated)
        isolated_ticket = p.hold_io(
            isolated_handle, kind="load-target", writes=True, receipt=object()
        )
        source.free(isolated)
        p.complete_io(isolated_ticket, outcome="unknown")
        with storage.reset_scope() as ready:
            self.assertTrue(ready)
            self.assertEqual(
                set(p.lease.approved_pages),
                set(range(1, 4)) - {isolated_handle[0].page_id},
            )
            cache.reset()
            a.clear()
        self.assertEqual(p.epoch.reset_seq, old_epoch.reset_seq + 1)
        self.assertEqual(source.available_size(), 8)
        self.assertEqual(p.accounting().quarantined, 1)
        self.assertNotIn(0, p.states)

    def test_requires_ticket_and_detects_duplicate_page_ids(self):
        p, a, _, _ = build_provider_fixture()
        source = a.full_attn_allocator
        with self.assertRaisesRegex(RuntimeError, "reservation"):
            source.alloc(4)
        ids = take_pages(source, p, 1)
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            source.free_page_ids(torch.tensor([int(ids[0]) // 4] * 2))
        source.free(ids)
        self.assertEqual(p.accounting().cached, 16)
        ticket = p.reserve(CapacityShape("malformed", 1, 1, 0, 0, 4))
        with patch.object(
            ticket, "take", return_value=(replace(p.handle(1), page_id=0),)
        ):
            with source.reservation_scope(ticket, "load"):
                from sglang.srt.mem_cache.kv_memory_provider import ProviderFailStop

                with self.assertRaises(ProviderFailStop):
                    source.alloc(4)
        ticket.refund_unused()
        self.assertEqual(p.accounting().cached, 16)

    def test_c11_request_retirer_finish_abort_retract_share_return_path(self):
        from sglang.srt.managers.req_types import FINISH_LENGTH
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.test.scheduler_test_utils import release_only_retirer

        for verb in ("finish", "abort", "retract"):
            with self.subTest(verb=verb):
                p, a, cache, storage = build_provider_fixture()
                ids = take_pages(a.full_attn_allocator, p, 2)
                handles = p.freeze_allocations(ids).handles(ids)
                p.publish(
                    [ResidentPage(str(i), None, h) for i, h in enumerate(handles)],
                    ready=CompletedReceipt(),
                )
                p.progress()
                p.auto_release = False
                req = Req(
                    rid="retire",
                    origin_input_text=None,
                    origin_input_ids=array("q", range(8)),
                    sampling_params=SamplingParams(max_new_tokens=0),
                )
                req.refresh_fill_ids()
                row = storage.acquire([req])[0]
                storage.requests.req_to_token[row, :8] = ids
                req.kv.allocate_prealloc(8)
                req.kv.commit_prealloc(8)
                req.kv.reservation = p.reserve(CapacityShape("unused", 1, 0, 1, 8, 12))
                req.skip_radix_cache_insert = True
                retirer = release_only_retirer(cache)
                retirer.send_abort_output = lambda *args: None
                if verb == "finish":
                    req.finish_outside_step(FINISH_LENGTH(length=0))
                    retirer.release(req, checkpoint=False)
                elif verb == "abort":
                    retirer.abort(req)
                else:
                    retirer.retract(req, offload_kv=False)
                self.assertFalse(req.kv.holds_kv)
                self.assertIsNone(req.kv.reservation)
                self.assertEqual(p.accounting().reserved, 0)
                self.assertEqual(p.accounting().retiring, 2)
                self.assertEqual(len([x for x in p.commands if x[0] == "retire"]), 2)


if __name__ == "__main__":
    unittest.main()
