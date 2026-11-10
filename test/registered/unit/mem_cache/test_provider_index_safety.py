"""G8: chunked-prefill provenance, row registry safety and device audits."""

import unittest
from array import array
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.mem_cache.allocation import _prefix_kv_indices, write_cache_indices
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.kv_memory_provider import CapacityShape
from sglang.srt.mem_cache.provider_indices import (
    attach_index_cpu,
    attach_page_ids,
    find_index_cpu,
    index_cpu,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.runtime_context import get_context, get_resources, reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kv_memory_provider import build_provider_fixture, take_pages
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _CudaIndices(torch.Tensor):
    @property
    def device(self):
        return torch.device("cuda:0")  # CPU backing; only metadata emulates CUDA


@contextmanager
def copy_probe(*, forbid_wait=False):
    empty = torch.empty
    original_copy = torch.Tensor.copy_
    pinned_destinations = set()
    copies = Mock()
    wait = Mock(
        side_effect=AssertionError("fallback synchronized") if forbid_wait else None
    )

    def pinned(*args, **kwargs):
        is_copy_buffer = kwargs.get("pin_memory")
        if is_copy_buffer:
            kwargs["pin_memory"] = False
        result = empty(*args, **kwargs)
        if is_copy_buffer:
            pinned_destinations.add(id(result))
        return result

    def copy(dst, source, *args, **kwargs):
        if id(dst) in pinned_destinations and source.device.type == "cuda":
            copies()
        return original_copy(dst, source, *args, **kwargs)

    with (
        patch("torch.empty", side_effect=pinned),
        patch.object(torch.Tensor, "copy_", copy),
        patch(
            "torch.cuda.current_stream",
            return_value=SimpleNamespace(
                record_event=lambda: SimpleNamespace(synchronize=wait)
            ),
        ),
    ):
        yield copies, wait


class TestProviderIndexSafety(CustomTestCase):
    def setUp(self):
        self.enterContext(get_context().override_server_args())
        self.enterContext(get_resources().override(kv_provider_index_mirrors={}))
        self.p, self.a, self.c, self.s = build_provider_fixture(pages=8)

    def row(self, n=5):
        from test_provider_prefill_reservation import request

        req = request(self.c, 16)
        row = self.s.acquire([req])[0]
        ids = take_pages(self.a.full_attn_allocator, self.p, 2)[:n]
        z, end, rows = torch.tensor([0]), torch.tensor([n]), torch.tensor([row])
        write_cache_indices(
            ids,
            rows,
            rows,
            z,
            z,
            end,
            end,
            end,
            end,
            [torch.empty(0, dtype=torch.int64)],
            self.s.requests,
        )
        req.kv.kv_allocated_len = req.kv.kv_committed_len = n
        req.prefix_len = n
        return req, row, ids

    def checker(self, reqs=()):
        return SchedulerInvariantChecker(
            disaggregation_mode=DisaggregationMode.PREFILL,
            page_size=4,
            tree_cache=self.c,
            token_to_kv_pool_allocator=self.a,
            req_to_token_pool=self.s.requests,
            pool_stats_observer=SimpleNamespace(storage=self.s),
            scheduler_stage_metrics=SimpleNamespace(),
            get_request_lists=lambda: {"active": reqs},
        )

    def test_g8_holds_kv_chunk_cast_has_no_fallback_copy_or_event_wait(self):
        req, row, _ = self.row()
        self.assertTrue(req.kv.holds_kv)
        self.s.requests.req_to_token = self.s.requests.req_to_token.as_subclass(
            _CudaIndices
        )
        batch = SimpleNamespace(
            req_to_token_pool=self.s.requests,
            token_to_kv_pool_allocator=self.a,
            tree_cache=self.c,
        )
        with copy_probe(forbid_wait=True) as (copies, wait):
            prefix = _prefix_kv_indices(batch, req)
            self.assertEqual(prefix.dtype, torch.int64)
            index_cpu(prefix[-1:])  # last_loc in the real extend preparation
            ticket = self.p.reserve(CapacityShape("next-chunk", 1, 0, 1, 5, 10))
            five, ten = torch.tensor([5]), torch.tensor([10])
            with self.a.full_attn_allocator.reservation_scope(ticket, "extend"):
                tail = self.a.full_attn_allocator.alloc_extend(
                    five, five, ten, ten, prefix[-1:], 5
                )
            self.s.requests._provider_row_mirror.write(row, prefix, tail)
            ticket.refund_unused()
            self.assertIsNotNone(find_index_cpu(prefix))
        copies.assert_not_called()
        wait.assert_not_called()

    def test_g8_fallback_read_does_not_replace_live_row_mirror(self):
        _, row, ids = self.row()
        mirror = self.s.requests._provider_row_mirror
        table = self.s.requests.req_to_token.as_subclass(_CudaIndices)
        with copy_probe() as (copies, wait):
            index_cpu(table)  # whole 2D table cannot be resolved as one CPU row
        self.assertTrue(copies.called and wait.called)
        replacement = ids.flip(0)
        mirror.write(row, replacement, torch.empty(0, dtype=torch.int64))
        self.assertEqual(find_index_cpu(table[row, :5]).tolist(), replacement.tolist())

    def test_g8_explicit_dense_attach_cannot_replace_mutable_row_mirror(self):
        _, row, ids = self.row()
        table = self.s.requests.req_to_token
        attach_index_cpu(table, table.to(torch.int64, copy=True))
        replacement = ids.flip(0)
        self.s.requests._provider_row_mirror.write(
            row, replacement, torch.empty(0, dtype=torch.int64)
        )
        self.assertEqual(find_index_cpu(table[row, :5]).tolist(), replacement.tolist())

    def test_g8_row_views_reregister_after_runtime_resource_reset(self):
        _, row, ids = self.row()
        reset_context()
        cpu = find_index_cpu(self.s.requests.req_to_token[row, :5])
        self.assertIsNotNone(cpu)
        self.assertEqual(cpu.tolist(), ids.tolist())

    def test_g8_raw_table_writers_are_rejected_before_mutation(self):
        from sglang.srt.beam_search.fork import (
            alias_members_prompt_kv,
            remap_kv_mapping,
        )
        from sglang.srt.mem_cache.allocation import assign_req_to_token_pool_func
        from sglang.srt.models.unlimited_ocr_v2_support.document_reuse import DocReuse

        _, row, _ = self.row()
        table = self.s.requests.req_to_token
        before = table.clone()
        calls = (
            lambda: alias_members_prompt_kv(table, torch.tensor([row + 1]), row, 4),
            lambda: remap_kv_mapping(
                table, torch.tensor([row]), torch.tensor([0]), 0, 4
            ),
            lambda: assign_req_to_token_pool_func(
                torch.tensor([row]),
                table,
                torch.tensor([0]),
                torch.tensor([1]),
                torch.tensor([99]),
                1,
            ),
            lambda: DocReuse._splice_text(
                SimpleNamespace(_cache=self.c), None, 0, None
            ),
        )
        for call in calls:
            with (
                self.subTest(writer=call),
                self.assertRaisesRegex(RuntimeError, "provider.*unmirrored"),
            ):
                call()
            self.assertTrue(torch.equal(table, before))

    def test_g8_canary_pointer_perturbation_rejected_with_provider(self):
        from sglang.srt.kv_canary.perturb import req_to_token

        self.row()
        with (
            patch.object(req_to_token, "should_run_perturbation", return_value=True),
            patch.object(req_to_token, "collect_active_slots") as slots,
        ):
            with self.assertRaisesRegex(RuntimeError, "provider.*unmirrored"):
                req_to_token.run(
                    maybe_inaccurate_forward_batch=None,
                    config=SimpleNamespace(req_to_token_prob=1),
                    req_to_token_pool=self.s.requests,
                    warmup_gate=None,
                )
            slots.assert_not_called()

    def test_g8_cp_padded_provider_rejected_before_allocation(self):
        z, end = torch.tensor([0]), torch.tensor([4])
        with (
            patch("sglang.srt.layers.cp.base.is_cp_sharded_padded", return_value=True),
            self.assertRaisesRegex(ValueError, "CP padded"),
        ):
            self.s.take_extend(
                rows=None,
                prefix_lens=z,
                prefix_lens_cpu=z,
                seq_lens=end,
                seq_lens_cpu=end,
                last_loc=torch.tensor([-1]),
                num_tokens=4,
                make_room=lambda _: None,
                batch=SimpleNamespace(reqs=[]),
            )

    def test_g8_vectorized_geometry_does_not_build_python_token_ranges(self):
        ticket = self.p.reserve(CapacityShape("vector", 1, 0, 2, 0, 8))
        z, end = torch.tensor([0]), torch.tensor([8])
        with patch(
            "sglang.srt.mem_cache.allocator.provider.range",
            side_effect=AssertionError("Python token range"),
            create=True,
        ):
            with self.a.full_attn_allocator.reservation_scope(ticket, "extend"):
                indices = self.a.full_attn_allocator.alloc_extend(
                    z, z, end, end, torch.tensor([-1]), 8
                )
        ticket.refund_unused()
        self.assertEqual(index_cpu(indices).tolist(), list(range(4, 12)))

    def test_g8_page_hint_must_equal_preserved_cpu_geometry(self):
        ids = take_pages(self.a.full_attn_allocator, self.p, 2)
        with self.assertRaisesRegex(
            (AssertionError, ValueError, RuntimeError), "mirror|geometry"
        ):
            attach_page_ids(ids, (7, 8), 4)
        self.assertEqual(index_cpu(ids).tolist(), ids.tolist())

    def test_g8_first_idle_audit_reads_device_values_then_obeys_interval(self):
        checker = self.checker()
        with (
            envs.SGLANG_KV_PROVIDER_AUDIT_INTERVAL_SECONDS.override(60.0),
            patch.object(
                self.c, "all_values_flatten", wraps=self.c.all_values_flatten
            ) as census,
        ):
            self.assertEqual(checker._check_full_pool(None), (False, ""))
            self.assertEqual(census.call_count, 1)
            self.assertEqual(checker._check_full_pool(None), (False, ""))
            self.assertEqual(census.call_count, 1)

    def test_g8_audit_detects_tree_device_values_diverging_from_cpu_geometry(self):
        for backend in ("python", "rust"):
            with self.subTest(backend=backend):
                p, a, c, s = build_provider_fixture(pages=4, backend=backend)
                ids = take_pages(a.full_attn_allocator, p, 2)
                key = RadixKey(array("q", range(8)))
                c.insert(InsertParams(key=key, value=ids))
                node = c.match_prefix(MatchPrefixParams(key=key)).last_device_node
                from sglang.srt.mem_cache.unified_cache.component_type import (
                    ComponentType,
                )

                value = c.tree_core.get_component_device_value(node, ComponentType.FULL)
                value[0] += (
                    1  # simulate a device-side indexing bug, bypass mirror writers
                )
                checker = SchedulerInvariantChecker(
                    disaggregation_mode=DisaggregationMode.PREFILL,
                    page_size=4,
                    tree_cache=c,
                    token_to_kv_pool_allocator=a,
                    req_to_token_pool=s.requests,
                    pool_stats_observer=SimpleNamespace(storage=s),
                    scheduler_stage_metrics=SimpleNamespace(),
                )
                with envs.SGLANG_KV_PROVIDER_AUDIT_INTERVAL_SECONDS.override(0):
                    bad, message = checker._check_full_pool(None)
                self.assertTrue(bad)
                self.assertIn("mirror", message)

    def test_g8_audit_detects_request_row_device_value_drift(self):
        req, row, ids = self.row()
        self.c.insert(InsertParams(key=RadixKey(array("q", range(4))), value=ids[:4]))
        req.kv.cache_protected_len = 4
        self.s.requests.req_to_token[row, 0] += 1
        with envs.SGLANG_KV_PROVIDER_AUDIT_INTERVAL_SECONDS.override(0):
            bad, message = self.checker([req])._check_full_pool(None)
        self.assertTrue(bad)
        self.assertIn("mirror", message)

    def test_g8_device_audit_matches_cpu_geometry_order_on_branched_trees(self):
        for backend in ("python", "rust"):
            with self.subTest(backend=backend):
                p, a, c, s = build_provider_fixture(pages=4, backend=backend)
                for tokens, pages in ((range(8), 2), (range(100, 104), 1)):
                    ids = take_pages(a.full_attn_allocator, p, pages)
                    c.insert(InsertParams(key=RadixKey(array("q", tokens)), value=ids))
                checker = SchedulerInvariantChecker(
                    disaggregation_mode=DisaggregationMode.PREFILL,
                    page_size=4,
                    tree_cache=c,
                    token_to_kv_pool_allocator=a,
                    req_to_token_pool=s.requests,
                    pool_stats_observer=SimpleNamespace(storage=s),
                    scheduler_stage_metrics=SimpleNamespace(),
                )
                self.assertEqual(checker._check_full_pool(None), (False, ""))


if __name__ == "__main__":
    unittest.main()
