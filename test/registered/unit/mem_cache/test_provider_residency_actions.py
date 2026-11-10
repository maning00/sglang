"""Both real tree cores adopt identical pages without additional barriers."""

import unittest
from array import array
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.kv_memory_provider import allocation_page_ids
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.cache_action import ResidencyAction
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kv_memory_provider import (
    ManualReceipt,
    build_provider_fixture,
    take_pages,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestProviderResidencyActions(CustomTestCase):
    def setUp(self):
        scope = get_context().override_server_args()
        scope.install()
        self.addCleanup(scope.restore)

    def test_g1c_page_ids_survive_suffix_slice_split_and_offer_confirm_on_both_cores(
        self,
    ):
        for backend in ("python", "rust"):
            with self.subTest(backend=backend):
                p, a, c, _ = build_provider_fixture(backend=backend)
                original = take_pages(a.full_attn_allocator, p, 2)
                fresh = take_pages(a.full_attn_allocator, p, 1)
                branch = torch.cat((original[:4], fresh))
                branch._kv_provider_page_ids = (
                    original._kv_provider_page_ids[0],
                    fresh._kv_provider_page_ids[0],
                )
                actions = []
                apply = c._apply_cache_actions

                def applying(items):
                    actions.extend(x for x in items if isinstance(x, ResidencyAction))
                    return apply(items)

                with (
                    patch.object(p, "page_ids", wraps=p.page_ids) as reads,
                    patch(
                        "sglang.srt.mem_cache.kv_memory_provider.allocation_page_ids",
                        wraps=allocation_page_ids,
                    ) as suffix_reads,
                    patch.object(c, "_apply_cache_actions", side_effect=applying),
                ):
                    c.insert(
                        InsertParams(key=RadixKey(array("q", range(8))), value=original)
                    )
                    c.insert(
                        InsertParams(
                            key=RadixKey(array("q", [0, 1, 2, 3, 40, 41, 42, 43])),
                            value=branch,
                            prev_prefix_len=4,
                        )
                    )
                    candidates = c.offer_candidates(1)
                    self.assertTrue(candidates)
                    confirmed = c.confirm_detach(candidates)
                    self.assertTrue(all(r.status == "detached" for r in confirmed))
                self.assertEqual(reads.call_count, 0)
                self.assertEqual(suffix_reads.call_count, 0)
                for action in actions:
                    self.assertEqual(
                        action.page_ids, tuple((action.indices[::4] // 4).tolist())
                    )

    def test_g1c_swa_full_replacements_keep_cpu_page_ids_on_both_cores(self):
        for backend in ("python", "rust"):
            for window_start in (0, 4):
                with self.subTest(backend=backend, window_start=window_start):
                    p, a, c, _ = build_provider_fixture(
                        backend=backend,
                        ring=False,
                        components=(ComponentType.FULL, ComponentType.SWA),
                    )
                    old = take_pages(a.full_attn_allocator, p, 2)
                    a.set_full_to_swa_mapping(old, a.swa_attn_allocator.alloc(8))
                    a.free_swa_segment(old, start_pos=0)
                    key = RadixKey(array("q", range(8)))
                    c.insert(
                        InsertParams(
                            key=key,
                            value=old,
                            component_evicted_seqlens={ComponentType.SWA: 8},
                        )
                    )
                    fresh = take_pages(a.full_attn_allocator, p, 2)
                    a.set_full_to_swa_mapping(fresh, a.swa_attn_allocator.alloc(8))
                    apply, actions = c._apply_cache_actions, []

                    def applying(items):
                        actions.extend(
                            x for x in items if isinstance(x, ResidencyAction)
                        )
                        return apply(items)

                    with (
                        patch.object(p, "page_ids", wraps=p.page_ids) as reads,
                        patch(
                            "sglang.srt.mem_cache.kv_memory_provider.allocation_page_ids",
                            wraps=allocation_page_ids,
                        ) as suffix_reads,
                        patch.object(c, "_apply_cache_actions", side_effect=applying),
                    ):
                        c.insert(
                            InsertParams(
                                key=key,
                                value=fresh,
                                component_evicted_seqlens={
                                    ComponentType.SWA: window_start
                                },
                            )
                        )
                        candidates = c.offer_candidates(1)
                    self.assertEqual(reads.call_count, 0)
                    self.assertEqual(suffix_reads.call_count, 0)
                    self.assertTrue(any(x.reason == "dedup" for x in actions))
                    for action in actions:
                        self.assertEqual(
                            action.page_ids, tuple((action.indices[::4] // 4).tolist())
                        )
                    self.assertEqual(
                        candidates[0].handles[-1].page_id,
                        fresh._kv_provider_page_ids[-1],
                    )

    def test_g1c_partial_prefix_cpu_geometry_starts_at_the_containing_page(self):
        params = InsertParams(prev_prefix_len=5, provider_page_ids=(19, 23))
        self.assertEqual(params.page_ids_for(8, 12, 4), (23,))

    def _trajectory(self, backend):
        p, a, c, _ = build_provider_fixture(backend=backend)
        source = a.full_attn_allocator
        actions, barriers = [], []
        original = c._apply_cache_actions

        def apply(items):
            for x in items:
                if isinstance(x, ResidencyAction):
                    actions.append((x.reason, x.keys, x.parent_key, x.indices.tolist()))
            return original(items)

        def insert(tokens, values):
            params = InsertParams(key=RadixKey(array("q", tokens)), value=values)
            # An adoption is deferrable: a FULL-only insert returns its result
            # in the first call for both native and Python mechanisms.
            step = c.tree_core.begin_insert(params)
            count = 1
            try:
                while True:
                    apply(step.actions)
                    if step.result is not None:
                        break
                    step = c.tree_core.resume_insert()
                    count += 1
            finally:
                apply(c.tree_core.end_insert())
            barriers.append(count)

        first = take_pages(source, p, 2)
        done = ManualReceipt()
        context = p.freeze_allocations(first)
        p.record_forward(context.handles(first), done)
        insert(range(8), first)
        self.assertFalse(p.commands)
        p.progress()
        self.assertFalse(p.commands)
        done.done = True
        p.progress()
        # A split caused by prefix matching also travels through typed actions.
        match = c.tree_core.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", range(4))))
        )
        apply(match.cache_actions)
        duplicate = take_pages(source, p, 2)
        insert(range(8), duplicate)
        p.progress()
        candidates = c.offer_candidates(2)
        self.assertEqual(len(candidates), 1)
        stale = candidates[0]
        lock = c.lock(stale.node_id)
        self.assertEqual(c.confirm_detach([stale])[0].status, "stale")
        c.unlock(lock)
        self.assertEqual(c.confirm_detach(c.offer_candidates(2))[0].status, "detached")
        p.progress()
        self.assertEqual(p.accounting().violations(), [])
        # Restore a host-only tombstone through the real insertion mechanism.
        host_key = RadixKey(array("q", range(200, 204)))
        hosted = c.tree_core.insert_host(
            c.root_node_handle(), host_key, torch.arange(100, 104), ["b" * 64]
        )
        apply(hosted.cache_actions)
        restored = take_pages(source, p, 1)
        insert(range(200, 204), restored)
        p.progress()
        return actions, barriers

    def test_t01_t03_native_python_adoption_split_dedup(self):
        python = self._trajectory("python")
        rust = self._trajectory("rust")
        self.assertEqual(python, rust)
        self.assertEqual(python[1], [1, 1, 1])
        self.assertIn("restore", [a[0] for a in python[0]])
        from sglang.srt.mem_cache.rust_tree_core.adapter import RustUnifiedTreeCore
        from sglang.srt.mem_cache.rust_tree_core.extension import bindings

        _, _, cache, _ = build_provider_fixture(backend="rust")
        self.assertIsInstance(cache.tree_core, RustUnifiedTreeCore)
        self.assertIn("umbp-engine-hbm/sglang/cache", bindings.__file__)

    def test_t02_abort_at_existing_component_barrier_drains_once(self):
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

        outcomes = []
        for backend in ("python", "rust"):
            p, a, c, _ = build_provider_fixture(
                backend=backend,
                ring=False,
                components=(ComponentType.FULL, ComponentType.SWA),
            )
            ids = take_pages(a.full_attn_allocator, p, 2)
            swa = a.swa_attn_allocator.alloc(8)
            a.set_full_to_swa_mapping(ids, swa)
            gate = ManualReceipt()
            p.record_forward(p.freeze_allocations(ids).handles(ids), gate)
            step = c.tree_core.begin_insert(
                InsertParams(key=RadixKey(array("q", range(8))), value=ids)
            )
            self.assertIsNone(step.result)  # existing SWARebuild barrier
            actions = [
                (x.reason, x.keys, x.indices.tolist())
                for x in step.actions
                if isinstance(x, ResidencyAction)
            ]
            # Stop the walk at its public abort boundary; already emitted
            # actions plus finally's remainder are consumed only once.
            c._apply_cache_actions(step.actions)
            c._apply_cache_actions(c.tree_core.end_insert())
            self.assertFalse(c.tree_core.has_ongoing_insert())
            c.confirm_detach(c.offer_candidates(2))
            gate.done = True
            p.progress()
            self.assertEqual(p.accounting().cached, 16)
            self.assertFalse(any(command[0] == "publish" for command in p.commands))
            outcomes.append((actions, p.accounting()))
        self.assertEqual(outcomes[0], outcomes[1])

    def test_c25_disabled_provider_has_no_residency_or_extra_barrier(self):
        for backend in ("python", "rust"):
            _, a, c, _ = build_provider_fixture(backend=backend, provider_enabled=False)
            values = a.full_attn_allocator.alloc(4)
            step = c.tree_core.begin_insert(
                InsertParams(key=RadixKey(array("q", range(4))), value=values)
            )
            self.assertIsNotNone(step.result)
            self.assertFalse(any(isinstance(x, ResidencyAction) for x in step.actions))
            c._apply_cache_actions(step.actions)
            self.assertEqual(c.tree_core.end_insert(), [])


if __name__ == "__main__":
    unittest.main()
