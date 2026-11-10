"""Real prefill admission, request tickets, load rollback and batch allocation."""

import os
import tempfile
import time
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.allocation import _room_in
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.kv_memory_provider import (
    CapacityScope,
    CapacityShape,
    refund_request_reservation,
    reservation_request_id,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    ExternalCacheHitMarker,
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kv_memory_provider import build_provider_fixture, take_pages
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def request(cache, n=8, *, ignore_eos=False):
    req = Req(
        rid="request",
        origin_input_text=None,
        origin_input_ids=array("q", range(n)),
        sampling_params=SamplingParams(max_new_tokens=0, ignore_eos=ignore_eos),
    )
    req.refresh_fill_ids()
    req.prefix_len = 0
    req.last_node = req.best_match_node = cache.root_node_handle()
    return req


def adder(allocator, cache, *, chunk=None):
    return PrefillAdder(
        page_size=allocator.page_size,
        tree_cache=cache,
        token_to_kv_pool_allocator=allocator,
        running_batch=None,
        new_token_ratio=1.0,
        rem_input_tokens=128,
        rem_chunk_tokens=chunk,
    )


def _attention_rank(rank, rendezvous):
    import torch.distributed as dist

    from sglang.test.kv_memory_provider import make_fake_linker

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        with get_context().override_server_args():
            p, a, c, storage = build_provider_fixture(pages=4)
            c.attn_tp_group = c.tp_group = dist.group.WORLD
            c.tp_world_size = 2
            held = p.reserve(CapacityShape("other", 1, 0, 3, 0, 12)) if rank else None
            req = request(c)
            shape = CapacityShape(reservation_request_id(req), 1, 0, 2, 0, 8)
            with CapacityScope(
                p,
                c,
                shape,
                deadline_ns=time.monotonic_ns() + 100_000_000,
                verdict=c.provider_verdict,
            ) as capacity:
                assert not capacity.ready
            assert p.accounting().reserved == (3 if rank else 0)
            if held:
                held.refund_unused()
            # Rank 0 queues its plan; rank 1 rejects the source. The queued
            # plan is cancelled before both ranks roll their target pages back.
            backend = make_fake_linker(p)
            backend.reject = bool(rank)
            c.linker = UnifiedCacheLinkerWrapper(c, backend)
            c.linker.hit_markers[req.rid] = ExternalCacheHitMarker(["a" * 64], 0)
            req.kv.reservation_revision = 2
            req.kv.reservation = p.reserve(
                CapacityShape(reservation_request_id(req), 2, 1, 0, 4, 4)
            )
            assert c.linker.load_back(req)[0] == 0
            req.kv.reservation.refund_unused()
            assert p.accounting().cached == 4 and not backend.queued
            # A Ready rank aborts its sealed lease if another attention rank
            # reports Busy; neither rank changes epoch or its local tree.
            if rank:
                source_ids = take_pages(a.full_attn_allocator, p, 1)
                h = p.freeze_allocations(source_ids).handles(source_ids)
                ticket = p.hold_io(h, kind="pd-send", writes=False, receipt=object())
            with storage.reset_scope(verdict=c.provider_verdict) as ready:
                assert not ready
            assert p.epoch.reset_seq == 0 and p.lease is None
            if rank:
                p.complete_io(ticket, outcome="completed")
                a.full_attn_allocator.free(source_ids)
            # Independent DP workers have no attention group: a Short in one
            # does not turn the other worker's local Ready into a refusal.
            c.attn_tp_group = c.tp_group = None
            c.tp_world_size = 1
            held = p.reserve(CapacityShape("other", 3, 0, 3, 0, 12)) if rank else None
            with CapacityScope(
                p, c, shape, deadline_ns=time.monotonic_ns(), verdict=c.provider_verdict
            ) as capacity:
                assert capacity.ready == (rank == 0)
            Path(f"{rendezvous}.{rank}.ok").write_text(
                "ready/short/load rollback/DP passed"
            )
    finally:
        dist.destroy_process_group()


class RefusingLinker:
    load_mode, prefetch_policy = "layer_wise", "timeout"
    pools = {}

    def __init__(self):
        self.targets = []

    def load(self, rid, transfers):
        self.targets.append(transfers[0].device_indices.clone())
        return False


class TestProviderPrefillReservation(CustomTestCase):
    def setUp(self):
        scope = get_context().override_server_args()
        scope.install()
        self.addCleanup(scope.restore)

    def test_c12_other_and_exception_refund_before_commit(self):
        for outcome in (None, RuntimeError("load boundary failure")):
            with self.subTest(outcome=outcome):
                p, a, c, _ = build_provider_fixture()
                req = request(c)
                req.host_hit_length = 4
                add = adder(a, c)
                with patch.object(
                    c,
                    "init_load_back",
                    side_effect=outcome if isinstance(outcome, Exception) else None,
                    return_value=None,
                ):
                    if isinstance(outcome, Exception):
                        with self.assertRaisesRegex(RuntimeError, "load boundary"):
                            add.add_one_req(req, False, None)
                    else:
                        self.assertEqual(
                            add.add_one_req(req, False, None), AddReqResult.OTHER
                        )
                self.assertEqual(add.can_run_list, [])
                self.assertIsNone(req.kv.reservation)
                self.assertEqual(p.accounting().cached, 16)

    def test_c12_short_keeps_waiting_request_uncommitted(self):
        p, a, c, _ = build_provider_fixture(pages=8)
        ids = take_pages(a.full_attn_allocator, p, 8)
        c.insert(InsertParams(key=RadixKey(array("q", range(100, 132))), value=ids))
        p.progress()
        handles = p.freeze_allocations(ids).handles(ids)
        p.hold_io(handles, kind="remote-reader", writes=False, receipt=object())
        req = request(c)
        waiting = [req]
        add = adder(a, c)
        self.assertEqual(add.add_one_req(req, False, None), AddReqResult.NO_TOKEN)
        self.assertEqual(waiting, [req])
        self.assertEqual(add.can_run_list, [])
        self.assertIsNone(req.kv.reservation)
        self.assertEqual(p.accounting().cached, 0)

    def test_c14_c24_refused_load_revises_one_load_one_extend_to_two_extend(self):
        p, a, c, _ = build_provider_fixture()
        backend = RefusingLinker()
        c.linker = UnifiedCacheLinkerWrapper(c, backend)
        req = request(c)
        req.host_hit_length = 4
        c.linker.hit_markers[req.rid] = ExternalCacheHitMarker(["a" * 64], 0)
        add = adder(a, c)
        add.add_one_req(req, False, None)
        self.assertEqual(add.can_run_list, [req])
        self.assertEqual(
            [(s.load_pages, s.extend_pages) for s in p.shapes], [(1, 1), (0, 2)]
        )
        self.assertEqual(req.kv.reservation.shape.revision, 2)
        self.assertEqual(p.accounting().reserved, 2)
        self.assertEqual(p.accounting().in_use, 0)
        self.assertEqual(len(backend.targets), 1)
        refund_request_reservation(req)
        self.assertEqual(p.accounting().cached, 16)

    def test_c13_chunk_shrink_continuation_short_keeps_prefix_and_ring(self):
        p, a, c, _ = build_provider_fixture(pages=16)
        req = request(c, 12)
        add = adder(a, c, chunk=4)
        add.add_one_req(req, False, None)
        self.assertIs(add.new_chunked_req, req)
        self.assertEqual(req.extend_len, 4)
        self.assertEqual(req.kv.reservation.shape.extend_pages, 1)
        # Consume the first chunk through the real paged extend kernel.
        with a.full_attn_allocator.batch_reservation_scope(
            [req], torch.tensor([0]), torch.tensor([4])
        ):
            first = a.full_attn_allocator.alloc_extend(
                torch.tensor([0]),
                torch.tensor([0]),
                torch.tensor([4]),
                torch.tensor([4]),
                torch.tensor([-1]),
                4,
            )
        req.prefix_len = 4
        held_prefix = first.clone()
        before_ring = a.swa_available_size()
        # Remaining physical pages are independently reserved; no tree leaf can
        # be retired to cover the continuing chunk.
        other = p.reserve(CapacityShape("other", 1, 0, 15, 0, 60))
        continued = adder(a, c, chunk=4)
        self.assertIs(continued.add_chunked_req(req), req)
        self.assertTrue(continued.chunked_req_waits)
        self.assertFalse(continued.can_run_list)
        self.assertEqual(req.prefix_len, 4)
        self.assertTrue(torch.equal(first, held_prefix))
        self.assertEqual(a.swa_available_size(), before_ring)
        other.refund_unused()

    def test_c15_two_request_load_and_partial_tail_batch(self):
        p, a, c, storage = build_provider_fixture(pages=3)
        r1 = SimpleNamespace(rid="one", kv=ReqKvInfo(reservation_revision=1))
        r2 = SimpleNamespace(rid="two", kv=ReqKvInfo(reservation_revision=1))
        first = take_pages(a.full_attn_allocator, p, 1)
        # Req1 already owns an incomplete page; only its tail is extended.
        r1.kv.reservation = p.reserve(
            CapacityShape(reservation_request_id(r1), 1, 0, 0, 2, 3)
        )
        r2.kv.reservation = p.reserve(
            CapacityShape(reservation_request_id(r2), 1, 1, 1, 4, 8)
        )
        with a.full_attn_allocator.reservation_scope(r2.kv.reservation, "load"):
            loaded = storage.take_units("kv", 4)
        out = storage.take_extend(
            rows=None,
            prefix_lens=torch.tensor([2, 4]),
            prefix_lens_cpu=torch.tensor([2, 4]),
            seq_lens=torch.tensor([3, 8]),
            seq_lens_cpu=torch.tensor([3, 8]),
            last_loc=torch.tensor([first[1], loaded[-1]]),
            num_tokens=5,
            make_room=_room_in(c),
            tree_cache=c,
            batch=SimpleNamespace(reqs=[r1, r2]),
        )
        self.assertEqual(out.tolist(), [int(first[2]), 12, 13, 14, 15])
        self.assertEqual(p.accounting().in_use, 3)
        self.assertIsNone(r1.kv.reservation)
        self.assertIsNone(r2.kv.reservation)

    def test_c15_stale_revision_rejected_and_all_batch_tickets_refunded(self):
        p, a, c, _ = build_provider_fixture()
        req = request(c)
        req.kv.reservation_revision = 2
        req.kv.reservation = p.reserve(
            CapacityShape(reservation_request_id(req), 1, 0, 2, 0, 8)
        )
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            with a.full_attn_allocator.batch_reservation_scope(
                [req], torch.tensor([0]), torch.tensor([8])
            ):
                self.fail("stale reservation entered allocation")
        self.assertIsNone(req.kv.reservation)
        self.assertEqual(p.accounting().cached, 16)

    def test_ignore_eos_entry_reserves_selected_chunk_and_charges_compute(self):
        p, a, c, _ = build_provider_fixture()
        c.disable = True
        req = request(c, 8, ignore_eos=True)
        add = adder(a, c, chunk=4)
        add.add_one_req(req, False, None)
        self.assertEqual(add.can_run_list, [req])
        self.assertEqual(req.kv.reservation.shape.extend_pages, 1)
        self.assertEqual(p.accounting().reserved, 1)
        self.assertEqual(add.rem_chunk_tokens, 0)
        self.assertEqual(add.rem_input_tokens, 124)
        refund_request_reservation(req)

    def test_c16_credit_only_extend_memory_preserves_future_alignment_and_ring(self):
        p, a, c, _ = build_provider_fixture()
        req = request(c)
        budget = a.create_prefill_budget(c)
        with budget.capacity_scope(
            req, prefix_tokens=4, end_tokens=8, load_tokens=4
        ) as scope:
            self.assertTrue(scope.ready)
            budget.reserve(
                4, 7, extra_tokens=2, unfinished_input_len=8, deferred_max_new_tokens=9
            )
            self.assertEqual(budget.current_offset, 6)  # alignment + extras
            self.assertEqual(budget.total_offset, 13)  # future headroom intact
            self.assertEqual(budget.swa_offset, a.swa_ring_cost_tokens)
        self.assertEqual(p.accounting().cached, 16)

    def test_c17_attention_group_short_and_load_rejection_rollback_dp_independent(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            rendezvous = str(Path(directory) / "gloo")
            torch.multiprocessing.spawn(
                _attention_rank, (rendezvous,), nprocs=2, join=True
            )
            self.assertTrue(
                all(Path(f"{rendezvous}.{rank}.ok").is_file() for rank in range(2))
            )


if __name__ == "__main__":
    unittest.main()
