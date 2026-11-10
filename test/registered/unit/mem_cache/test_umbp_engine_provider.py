"""Engine adapter boundaries, without importing mori or starting a node."""

import os
import time
import unittest
from enum import Enum, auto
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.common.conn import PhysicalSendReceipt, SendRecord
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.kv_memory_provider import (
    CapacityShape,
    CompletedReceipt,
    PageGeometry,
    PageRegion,
    PoolEpoch,
    ProviderFailStop,
    ResidentPage,
)
from sglang.srt.mem_cache.kv_memory_provider_factory import (
    get_or_create_kv_memory_provider,
    provider_mem_pool,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.umbp.engine_provider import (
    UMBPEngineProvider,
    _rank_slot,
)
from sglang.srt.runtime_context import get_context, get_parallel, get_resources
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kv_memory_provider import ManualReceipt, build_provider_fixture
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _NoHostIndices(torch.Tensor):
    def cpu(self, *args, **kwargs):
        raise AssertionError("device index copied to host")

    def tolist(self):
        raise AssertionError("device index converted to list")


class PublishState(Enum):
    Queued = auto()
    Ready = auto()
    Sent = auto()
    Accepted = auto()
    Rejected = auto()
    Cancelled = auto()
    Unknown = auto()


class Client:
    def __init__(self, *args):
        self.healthy = True
        self.refunds, self.reserves, self.calls = [], [], []
        self.backend_id = 8
        self.finish_ok = True
        self.publications = {}
        self.published = []
        self.next_publication = 1
        self.cancel_unready_on_reset = False
        self.publish_events, self.retire_events, self.statuses = [], [], {}
        self.next_accepted = None
        self.terminal_on_mark = True
        self.reset_seq = 0
        self.generations = {}
        self.quarantined = set()
        self.holds = {}
        self.publish_acks = []
        self.event_overflow = False

    def attach_executor(self, executor):
        self.executor = executor

    def reserve(self, rid, load, extend):
        self.reserves.append((rid, load, extend))
        return SimpleNamespace(success=True, shortfall=0, id=len(self.reserves))

    def take(self, rid, purpose, n):
        self.calls.append((rid, purpose, n))
        pages = [i for i in range(1, 5) if i not in self.quarantined][:n]
        for page in pages:
            self.generations[page] = self.generations.get(page, 6) + 1
        return [
            SimpleNamespace(
                pool_id=self.pool_id,
                epoch=SimpleNamespace(boot_id=b"a" * 16, reset_seq=self.reset_seq),
                page_id=i,
                generation=self.generations[i],
            )
            for i in pages
        ]

    def refund(self, rid):
        self.refunds.append(rid)
        return True

    def return_pages(self, handles):
        return True

    def hold_reuse_guard(self, handles):
        return SimpleNamespace(value=1)

    def complete_reuse_guard(self, guard, proof):
        self.calls.append(("guard", proof))
        return True

    def quiesce_reset(self):
        if self.cancel_unready_on_reset:
            self.publications = {
                key: task for key, task in self.publications.items() if task.armed
            }
        if hasattr(self, "lease"):
            return self.lease
        epoch = SimpleNamespace(boot_id=b"a" * 16, reset_seq=self.reset_seq)
        return SimpleNamespace(
            ready=not self.holds,
            id=SimpleNamespace(seq=12),
            old_epoch=epoch,
            new_epoch=SimpleNamespace(boot_id=b"a" * 16, reset_seq=self.reset_seq + 1),
            approved=[
                SimpleNamespace(
                    pool_id=self.pool_id,
                    epoch=epoch,
                    backend_id=self.backend_id,
                    page_id=i,
                    generation=self.generations.get(i, 0),
                )
                for i in range(1, 5)
                if i not in self.quarantined
            ],
        )

    def queue_publish(self, records, ready):
        ticket = self.next_publication
        self.next_publication += 1
        self.publications[ticket] = SimpleNamespace(records=records, armed=ready)
        self.emit_publish(ticket, PublishState.Ready if ready else PublishState.Queued)
        return ticket

    def mark_publish_ready(self, ticket, proof):
        task = self.publications.get(ticket)
        if task is None:
            return False
        task.armed = True
        self.emit_publish(ticket, PublishState.Ready)
        self.emit_publish(ticket, PublishState.Sent)
        self.published.extend(task.records)
        if self.terminal_on_mark:
            accepted = self.next_accepted or [True] * len(task.records)
            self.next_accepted = None
            self.emit_publish(
                ticket,
                PublishState.Accepted if all(accepted) else PublishState.Rejected,
                accepted,
            )
        return True

    def cancel_publish(self, ticket):
        task = self.publications.get(ticket)
        if task is None or task.armed:
            return False
        del self.publications[ticket]
        self.emit_publish(ticket, PublishState.Cancelled)
        return True

    def emit_publish(self, ticket, state, accepted=()):
        receipt = SimpleNamespace(ticket=ticket, state=state, accepted=list(accepted))
        self.statuses[ticket] = receipt
        self.publish_events.append(receipt)

    def poll_publish_receipts(self):
        events, self.publish_events = self.publish_events, []
        return events

    def publish_status(self, ticket):
        return self.statuses.get(ticket)

    def acknowledge_publish_receipt(self, ticket):
        receipt = self.statuses.get(ticket)
        if receipt is None or receipt.state not in (
            PublishState.Accepted,
            PublishState.Rejected,
            PublishState.Cancelled,
            PublishState.Unknown,
        ):
            return False
        self.publish_acks.append(ticket)
        del self.statuses[ticket]
        return True

    def consume_event_overflow(self):
        overflow, self.event_overflow = self.event_overflow, False
        return overflow

    def poll_retires(self):
        events, self.retire_events = self.retire_events, []
        return events

    def acknowledge_released(self, operations):
        raise AssertionError("scheduler issued a Released ACK")

    def hold_io(self, handles):
        hold = SimpleNamespace(seq=len(self.holds) + 1)
        self.holds[id(hold)] = handles
        return hold

    def complete_io(self, hold, proof):
        handles = self.holds.pop(id(hold))
        if proof == "unknown":
            self.quarantined.update(h.page_id for h in handles)
            self.retire_events.append(
                SimpleNamespace(state="quarantined", handles=handles)
            )
        return True

    def finish_reset(self, lease, commit):
        self.calls.append(("finish_reset", lease, commit))
        if commit and self.finish_ok:
            self.reset_seq = lease.new_epoch.reset_seq
        return self.finish_ok

    def accounting(self):
        return SimpleNamespace(
            cached=4 - len(self.quarantined),
            reserved=0,
            in_use=0,
            pending_free=0,
            publish_sent=0,
            published=0,
            retiring=0,
            quarantined=len(self.quarantined),
            total=4,
        )

    def close(self):
        pass


class TestUMBPEngineProvider(CustomTestCase):
    def setUp(self):
        self.enterContext(get_context().override_server_args())
        self.enterContext(get_resources().override(kv_memory_providers={}))
        self.options = {
            "node_address": "unix:///tmp/engine.sock",
            "namespace": {
                "model": "model",
                "weights": "sha256:revision1",
                "kv_format": "fp8",
                "parallel": "dep8",
            },
        }
        self.engine = SimpleNamespace(
            EngineClient=Client,
            EngineExecutor=lambda *a: SimpleNamespace(close=lambda: None),
            kv_mem_pool=lambda device: object(),
            ReservationPurpose=SimpleNamespace(Load="load", Extend="extend"),
            CompletionProof=SimpleNamespace(
                Completed="completed",
                NeverSubmitted="never_submitted",
                Unknown="unknown",
            ),
            ResidentRecord=SimpleNamespace,
            PublishReceiptState=PublishState,
            RetireState=SimpleNamespace(
                Quarantined="quarantined", Ready="ready", Released="released"
            ),
        )
        mori, umbp = ModuleType("mori"), ModuleType("mori.umbp")
        mori.umbp, umbp.engine = umbp, self.engine
        self.enterContext(patch.dict("sys.modules", {"mori": mori, "mori.umbp": umbp}))

    def provider(self, options=None):
        p = UMBPEngineProvider(device="cpu", options=options or self.options)
        p.geometry = PageGeometry(
            4, 4, 32, "layout", (PageRegion("kv", 4096, 160, 32, 1, 0, 0, "kv"),)
        )
        p.epoch = PoolEpoch(b"a" * 16, 0)
        p.registration = object()
        p.client.pool_id = p.pool_id
        return p

    def _approval(self, provider, pages):
        return [
            SimpleNamespace(
                pool_id=provider.pool_id,
                backend_id=provider.client.backend_id,
                epoch=SimpleNamespace(
                    boot_id=b"a" * 16, reset_seq=provider.epoch.reset_seq
                ),
                page_id=page,
                generation=0,
            )
            for page in pages
        ]

    def test_factory_owns_one_client_and_pool_across_all_consumers(self):
        config = {
            "module": UMBPEngineProvider.__module__,
            "class": "UMBPEngineProvider",
            "options": self.options,
        }
        with (
            get_context().override_server_args(
                kv_memory_provider="dynamic", kv_memory_provider_config=config
            ),
            get_parallel().override(tp_rank=0, dp_rank=None),
        ):
            p = get_or_create_kv_memory_provider(device="cuda:0")
            self.assertIs(get_or_create_kv_memory_provider(device="cuda:0"), p)
            first = provider_mem_pool(p, "cuda:0")
            self.assertIs(first, provider_mem_pool(p, "cuda:0"))
            self.assertIs(first, p.mem_pool("cuda:0"))
            self.assertIs(p.client.executor, p.executor)
            with self.assertRaisesRegex(ValueError, "one device/client"):
                get_or_create_kv_memory_provider(device="cuda:1")
            self.assertEqual(len(get_resources().kv_memory_providers), 1)

    def test_cpu_page_sideband_preserves_native_generation_and_rejects_stale_context(
        self,
    ):
        p = self.provider()
        ticket = p.reserve(CapacityShape("req", 1, 0, 1, 0, 4))
        handles = ticket.take("extend", 1)
        self.assertEqual(handles[0].generation, 7)

        class Indices:
            _kv_provider_page_ids = (1,)

            def detach(self):
                raise AssertionError("unnecessary device roundtrip")

        frozen = p.freeze_allocations(Indices())
        self.assertEqual(frozen.handles(), handles)
        p.release_pages(handles, ready=CompletedReceipt())
        with self.assertRaises(RuntimeError):
            p.freeze_allocations(Indices(), context=frozen)

    def test_post_load_revision_keeps_native_extend_quota_reserved(self):
        p = self.provider()
        ticket = p.reserve(CapacityShape("req", 1, 1, 2, 0, 12))
        consumed = ticket.take("load", 1)
        self.assertTrue(ticket.revise(CapacityShape("req", 2, 0, 1, 4, 8)))
        self.assertEqual(ticket.remaining("extend"), 1)
        self.assertEqual(p.client.refunds, [])
        self.assertEqual(len(p.client.reserves), 1)
        ticket.refund_unused()
        self.assertEqual(p.client.refunds, [1])
        self.assertIn(consumed[0], p._owned)

    def test_reset_busy_reply_is_the_node_cancellation_acknowledgement(self):
        p = self.provider()
        p.client.lease = SimpleNamespace(ready=False, id=SimpleNamespace(seq=12))
        result = p.quiesce_reset(deadline_ns=2**63)
        self.assertEqual(type(result).__name__, "Busy")
        self.assertEqual(p.client.calls, [])
        self.assertIsNone(p._lease)
        ticket = p.reserve(CapacityShape("req", 1, 0, 1, 0, 4))
        ticket.refund_unused()

    def test_revise_shortfall_preserves_old_ticket_and_unused_quota(self):
        p = self.provider()
        old = CapacityShape("req", 1, 0, 2, 0, 8)
        ticket = p.reserve(old)
        p.client.reserve = lambda *args: SimpleNamespace(success=False, shortfall=1)
        self.assertFalse(ticket.revise(CapacityShape("req", 2, 0, 3, 0, 12)))
        self.assertTrue(ticket.covers(old))
        self.assertEqual(ticket.remaining("extend"), 2)
        self.assertEqual(p.client.refunds, [])
        ticket.refund_unused()
        self.assertEqual(p.client.refunds, [1])

    def test_reset_busy_without_a_healthy_reply_is_fail_stop(self):
        p = self.provider()

        def disconnected():
            p.client.healthy = False
            return SimpleNamespace(ready=False, id=SimpleNamespace(seq=12))

        p.client.quiesce_reset = disconnected
        with self.assertRaises(ProviderFailStop):
            p.quiesce_reset(deadline_ns=2**63)

    def test_invalid_precommit_approval_cancels_before_failure(self):
        p = self.provider()
        p.client.lease = SimpleNamespace(
            ready=True,
            id=SimpleNamespace(seq=12),
            approved=self._approval(p, (0, 1, 1)),
            old_epoch=SimpleNamespace(boot_id=b"a" * 16, reset_seq=0),
        )
        with self.assertRaisesRegex(ProviderFailStop, "approval"):
            p.quiesce_reset(deadline_ns=2**63)
        self.assertEqual(p.client.calls, [("finish_reset", p.client.lease, False)])
        self.assertIsNone(p._lease)

    def test_ready_lease_keeps_gate_sealed_until_native_commit(self):
        p = self.provider()
        p.client.lease = SimpleNamespace(
            ready=True,
            id=SimpleNamespace(seq=12),
            approved=self._approval(p, (1, 2, 3)),
            old_epoch=SimpleNamespace(boot_id=b"a" * 16, reset_seq=0),
            new_epoch=SimpleNamespace(boot_id=b"a" * 16, reset_seq=1),
        )
        lease = p.quiesce_reset(deadline_ns=2**63)
        self.assertEqual(lease.approved_pages, (1, 2, 3))
        with self.assertRaisesRegex(RuntimeError, "sealed"):
            p.reserve(CapacityShape("req", 1, 0, 1, 0, 4))
        p.commit_reset(lease)
        self.assertEqual(p.epoch.reset_seq, 1)
        self.assertIsNone(p._lease)
        self.assertEqual(p.client.calls, [("finish_reset", p.client.lease, True)])

    def test_failed_reuse_receipt_reports_unknown_and_retains_backing(self):
        p = self.provider()
        reservation = p.reserve(CapacityShape("req", 1, 0, 1, 0, 4))
        handles = reservation.take("extend", 1)

        class FailedReceipt:
            def query(self):
                raise RuntimeError("device completion unavailable")

        receipt = FailedReceipt()
        with self.assertRaisesRegex(ProviderFailStop, "unknown completion"):
            p.release_pages(handles, ready=receipt)
        self.assertIn(("guard", "unknown"), p.client.calls)
        self.assertTrue(p._retained)
        with self.assertRaises(ProviderFailStop):
            p.close()

    def test_namespace_includes_weights_layout_without_process_rank(self):
        p = self.provider()
        original_key = p.storage_key("key")
        other = self.provider()
        other.rank = 7
        self.assertEqual(p.storage_key("key"), other.storage_key("key"))
        other = self.provider(
            {
                **self.options,
                "namespace": {**self.options["namespace"], "weights": "revision2"},
            }
        )
        self.assertNotEqual(p.storage_key("key"), other.storage_key("key"))
        p.namespace_seed = other.namespace_seed
        other.geometry = SimpleNamespace(layout_id="another-layout")
        self.assertNotEqual(p.storage_key("key"), other.storage_key("key"))
        # Same JSON members in a different order resolve the same namespace.
        reordered = self.provider(
            {
                **self.options,
                "namespace": dict(reversed(list(self.options["namespace"].items()))),
            }
        )
        self.assertEqual(reordered.storage_key("key"), original_key)

    def test_missing_weights_revision_rejected_before_native_construction(self):
        self.options["namespace"]["weights"] = ""
        with self.assertRaisesRegex(ValueError, "namespace"):
            UMBPEngineProvider(device="cuda:0", options=self.options)

    def test_g1c_busy_or_abort_canceled_publication_recovers_without_fail_stop(self):
        for ready in (False, True):
            with self.subTest(ready_then_abort=ready):
                p = self.provider()
                ticket = p.reserve(CapacityShape("req", 1, 0, 2, 0, 8))
                handles = ticket.take("extend", 2)
                receipt = ManualReceipt()
                records = tuple(
                    ResidentPage(f"key-{h.page_id}", None, h) for h in handles
                )
                p.publish(records, ready=receipt)
                canceled = next(iter(p._publications))
                p.client.cancel_unready_on_reset = True
                p.client.lease = SimpleNamespace(
                    ready=ready,
                    id=SimpleNamespace(seq=12),
                    approved=self._approval(p, (1, 2, 3, 4)),
                    old_epoch=SimpleNamespace(boot_id=b"a" * 16, reset_seq=0),
                    new_epoch=SimpleNamespace(boot_id=b"a" * 16, reset_seq=1),
                )
                lease = p.quiesce_reset(deadline_ns=2**63)
                if ready:
                    p.abort_reset(lease)
                else:
                    self.assertEqual(type(lease).__name__, "Busy")
                receipt.done = True
                p.progress()
                p.progress()  # a replacement ticket may be armed on the next round
                self.assertIsNone(p._failed)
                self.assertNotIn(canceled, p._publications)
                self.assertEqual(
                    {r.key for r in p.client.published},
                    {p.storage_key(r.key) for r in records},
                )
                following = p.reserve(CapacityShape("next", 1, 0, 1, 0, 4))
                following.refund_unused()

    def _published_pages(self):
        p = self.provider()
        ticket = p.reserve(CapacityShape("req", 1, 0, 4, 0, 16))
        handles = ticket.take("extend", 4)
        for h in handles:
            p.publish(
                (ResidentPage(str(h.page_id), None, h),), ready=CompletedReceipt()
            )
        return p, handles

    def test_g1c_release_uses_handle_index_without_history_scan(self):
        p, handles = self._published_pages()

        class IndexedOnly(dict):
            def items(self):
                raise AssertionError("release scanned unrelated publication history")

        p._publications = IndexedOnly(p._publications)
        p.release_pages((handles[0],), ready=CompletedReceipt())
        self.assertNotIn(handles[0], p._owned)
        self.assertEqual(len(p._owned), 3)

    def test_g1c_progress_visits_only_unready_publications(self):
        p, handles = self._published_pages()

        class NoHistoryWalk(dict):
            def values(self):
                raise AssertionError("progress scanned armed publication history")

        p._publications = NoHistoryWalk(p._publications)
        p.progress()
        self.assertEqual(len(p._owned), 4)

    def test_g1c_frozen_handles_reuse_the_initial_cpu_page_ids(self):
        import torch

        p = self.provider()
        ticket = p.reserve(CapacityShape("req", 1, 0, 1, 0, 4))
        issued = ticket.take("extend", 1)

        class Indices:
            reads = 0

            def detach(self):
                self.reads += 1
                return torch.tensor([4, 5, 6, 7])

        indices = Indices()
        frozen = p.freeze_allocations(indices)
        self.assertEqual(frozen.handles(indices), issued)
        self.assertEqual(frozen.handles(indices), issued)
        self.assertEqual(indices.reads, 1)

    def test_g1c_accounting_does_not_invent_free_pages_to_balance_missing_states(self):
        p = self.provider()
        original = p.client.accounting()
        broken = SimpleNamespace(**vars(original))
        broken.cached = broken.total = 3
        p.client.accounting = lambda: broken
        self.assertEqual(p.accounting().free, 0)
        self.assertTrue(p.accounting_violations())
        # State counters can conserve geometry while the separately reported
        # native total is corrupt. The second check must still reject that.
        broken.cached = 4
        self.assertFalse(p.accounting().violations())
        errors = p.accounting_violations()
        self.assertTrue(any("native page total" in error for error in errors))
        self.assertIn("native page state counts != native total", errors)

    def _parallel_placement(self, width, rank, replica):
        return get_parallel().override(
            tp_size=width,
            tp_rank=rank,
            dp_rank=replica,
            dp_size=replica + 1,
            attn_tp_size=width,
            attn_tp_rank=rank,
            attn_dp_size=1,
            attn_dp_rank=0,
            attn_cp_size=1,
            attn_cp_rank=0,
            moe_tp_size=width,
            moe_ep_size=1,
            moe_dp_size=1,
            nnodes=1,
            node_rank=0,
        )

    def test_g1c_rank_uses_parallel_placement_under_cuda_remapping(self):
        with (
            patch.dict(
                os.environ, {"CUDA_VISIBLE_DEVICES": "6", "HIP_VISIBLE_DEVICES": "6"}
            ),
            self._parallel_placement(2, 1, 1),
        ):
            p = UMBPEngineProvider(device="cuda:0", options=self.options)
        self.assertEqual(p.rank, 3)

    def test_g1c_rank_base_separates_instances_and_worker_env_overrides_slot(self):
        with self._parallel_placement(4, 2, 0):
            first = UMBPEngineProvider(device="cuda:0", options=self.options)
            second = UMBPEngineProvider(
                device="cuda:0", options={**self.options, "rank_base": 4}
            )
            with patch.dict(os.environ, {"UMBP_ENGINE_RANK": "7"}):
                worker = UMBPEngineProvider(device="cuda:0", options=self.options)
        self.assertEqual((first.rank, second.rank, worker.rank), (2, 6, 7))

    def test_g3_sent_receipt_drops_heavy_ticket_and_rejected_result_can_republish(self):
        p = self.provider()
        reservation = p.reserve(CapacityShape("req", 1, 0, 2, 0, 8))
        handles = reservation.take("extend", 2)
        records = tuple(ResidentPage(str(h.page_id), None, h) for h in handles)
        p.client.terminal_on_mark = False
        p.publish(records, ready=CompletedReceipt())
        ticket = next(iter(p._publications))

        class IndexedOnly(dict):
            def __iter__(self):
                raise AssertionError("receipt drain scanned ticket history")

            def items(self):
                raise AssertionError("receipt drain scanned ticket history")

            def values(self):
                raise AssertionError("receipt drain scanned ticket history")

        p._publications = IndexedOnly(p._publications)
        p._publication_results = IndexedOnly(p._publication_results)
        p.client.publish_status = lambda _: self.fail("per-step ticket polling")
        p.progress()
        self.assertNotIn(ticket, p._publications)
        self.assertFalse(p._unready_publications)
        self.assertFalse(p._publication_by_handle)
        p.client.emit_publish(ticket, PublishState.Rejected, [True, False])
        p.progress()
        self.assertIn(handles[0], p._records)
        self.assertNotIn(handles[1], p._records)
        p.client.terminal_on_mark = True
        p.publish(records, ready=CompletedReceipt())
        replacement = p.client.next_publication - 1
        p.progress()
        self.assertNotIn(replacement, p._publications)
        self.assertIn(handles[1], p._records)
        p.client.emit_publish(ticket, PublishState.Rejected, [False, False])
        p.progress()
        self.assertIn(handles[1], p._records)  # late old ticket cannot erase retry

    def test_g3_ready_reset_uses_native_handles_and_preserves_unknown_isolation(self):
        from array import array

        p = self.provider()
        p, allocator, cache, storage = build_provider_fixture(
            pages=4, provider_override=p
        )
        source = allocator.full_attn_allocator
        ticket = p.reserve(CapacityShape("old", 1, 0, 2, 0, 8))
        with source.reservation_scope(ticket, "extend"):
            ids = source.alloc(8)
        ticket.refund_unused()
        cache.insert(InsertParams(key=RadixKey(array("q", range(4))), value=ids[:4]))
        h = p.freeze_allocations(ids[4:]).handles()
        io = p.hold_io(h, kind="pd-send", writes=False, receipt=object())
        source.free(ids[4:])
        p.complete_io(io, outcome="unknown")
        with storage.reset_scope() as ready:
            self.assertTrue(ready)
            self.assertNotIn(h[0].page_id, p._lease.approved_pages)
            cache.reset()
            allocator.clear()
        self.assertEqual(p.epoch.reset_seq, 1)
        self.assertEqual(p.accounting().quarantined, 1)
        new = p.reserve(CapacityShape("new", 1, 0, 3, 0, 12))
        with source.reservation_scope(new, "extend"):
            allocated = source.alloc(12)
        new.refund_unused()
        self.assertNotIn(h[0].page_id, allocated._kv_provider_page_ids)
        self.assertEqual(cache.all_values_flatten().numel(), 0)
        self.assertTrue(p._retained)

    def test_g3_progress_never_acknowledges_released_on_scheduler_thread(self):
        p = self.provider()
        p.client.retire_events.append(
            SimpleNamespace(state="ready", id="op", handles=[])
        )
        snapshot = p.client.accounting()
        snapshot.cached, snapshot.retiring = 3, 1
        p.client.accounting = lambda: snapshot
        p.progress()  # fake acknowledge_released raises if called

    def test_g3_local_quarantine_events_are_consumed_without_retiring_pages(self):
        p = self.provider()
        reservation = p.reserve(CapacityShape("req", 1, 0, 1, 0, 4))
        handles = reservation.take("extend", 1)
        reservation.refund_unused()
        # Model a native-owned load hold completing Unknown outside the
        # adapter. Isolation comes from hold/complete, never a state write.
        native = p.client.hold_io([p._native[h] for h in handles])
        p.client.complete_io(native, self.engine.CompletionProof.Unknown)
        self.assertFalse(p._quarantined)
        self.assertEqual(p.accounting().retiring, 0)
        p.progress()
        self.assertIn(handles[0], p._quarantined)
        with self.assertRaisesRegex(RuntimeError, "current live"):
            p.freeze_allocations(None, page_ids=(handles[0].page_id,))
        lease = p.quiesce_reset(deadline_ns=time.monotonic_ns() + 1_000_000_000)
        self.assertNotIn(handles[0].page_id, lease.approved_pages)
        p.abort_reset(lease)

    def test_g5_unknown_pd_read_keeps_shared_tree_prefix_live_until_all_returned(self):
        from array import array

        p, allocator, cache, storage = build_provider_fixture(
            pages=4, provider_override=self.provider()
        )
        source = allocator.full_attn_allocator
        reservation = p.reserve(CapacityShape("shared", 1, 0, 2, 0, 8))
        with source.reservation_scope(reservation, "extend"):
            ids = source.alloc(8)
        reservation.refund_unused()
        cache.insert(InsertParams(key=RadixKey(array("q", range(4))), value=ids[:4]))
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", range(8))), value=ids, prev_prefix_len=4
            )
        )
        handles = p.freeze_allocations(ids).handles()
        record = SendRecord()
        manager = SimpleNamespace(request_status={}, _send_records={1: record})
        receipt = PhysicalSendReceipt(manager, 1, record, 0)
        receipt.bind()
        storage.record_network_transfer(
            None,
            pages=[h.page_id for h in handles],
            page_size=4,
            writes=False,
            peer=receipt,
        )
        first = next(iter(p._io.values()))
        receipt.before_post()
        receipt.seal(unknown=True)
        self.assertEqual(receipt.poll(), "unknown")
        storage.poll_provider_io()
        self.assertFalse(storage._provider_io)
        # Another request uses the exact prefix still owned by the real tree.
        self.assertEqual(cache.all_values_flatten().numel(), 8)
        frozen = p.freeze_allocations(ids)
        second = p.hold_io(
            frozen.handles(), kind="pd-send", writes=False, receipt=object()
        )
        self.assertFalse(p._quarantined)
        self.assertIn(id(first), p._io)
        self.assertIn(id(first.native), p.client.holds)
        p.publish(tuple(p._records[h] for h in handles), ready=CompletedReceipt())
        p.record_forward(handles, CompletedReceipt())
        p.complete_io(second, outcome="completed")
        # Unknown is latched: a late caller cannot turn the old ticket into a
        # completed proof and allow reuse.
        with self.assertRaisesRegex(RuntimeError, "already completed"):
            p.complete_io(first, outcome="completed")
        cache.confirm_detach(cache.offer_candidates(1))
        self.assertEqual(cache.all_values_flatten().numel(), 4)
        self.assertFalse(p._quarantined)
        self.assertIn(id(first), p._io)
        cache.confirm_detach(cache.offer_candidates(1))
        self.assertEqual(cache.all_values_flatten().numel(), 0)
        self.assertEqual(p._quarantined, set(handles))
        self.assertNotIn(id(first), p._io)
        self.assertNotIn(id(first.native), p.client.holds)
        self.assertIs(p._retained[0][0].receipt, receipt)
        with self.assertRaisesRegex(RuntimeError, "current live"):
            p.freeze_allocations(ids)

    def test_g5_parked_read_blocks_network_reset_and_close_but_write_unknown_is_immediate(
        self,
    ):
        p, allocator, _, storage = build_provider_fixture(
            pages=4, provider_override=self.provider()
        )
        reservation = p.reserve(CapacityShape("io", 1, 0, 2, 0, 8))
        source = allocator.full_attn_allocator
        with source.reservation_scope(reservation, "extend"):
            ids = source.alloc(8)
        reservation.refund_unused()
        handles = p.freeze_allocations(ids).handles()
        manager = SimpleNamespace()
        storage.bind_network_manager(manager)
        read = p.hold_io(handles[:1], kind="pd-send", writes=False, receipt=object())
        p.complete_io(read, outcome="unknown")
        self.assertFalse(p._quarantined)
        self.assertFalse(manager.physical_send_quiescence())
        from sglang.srt.mem_cache.kv_memory_provider import Busy

        self.assertIsInstance(
            p.quiesce_reset(deadline_ns=time.monotonic_ns() + 1_000_000_000), Busy
        )
        write = p.hold_io(
            handles[1:], kind="load-target", writes=True, receipt=object()
        )
        p.complete_io(write, outcome="unknown")
        self.assertEqual(p._quarantined, {handles[1]})
        self.assertNotIn(id(write), p._io)
        self.assertIn(id(read), p._io)
        with (
            patch.object(p.client, "close") as close_client,
            patch.object(p.executor, "close") as close_executor,
        ):
            with self.assertRaises(ProviderFailStop):
                p.close()
            close_client.assert_not_called()
            close_executor.assert_not_called()
        self.assertIn(id(read), p._io)
        self.assertIn(id(read.native), p.client.holds)

    def test_g5_rank_slot_dp_attention_uses_world_tp_rank_without_replica_offset(self):
        with patch.dict(os.environ):
            os.environ.pop("UMBP_ENGINE_RANK", None)
            for rank in range(8):
                with (
                    self.subTest(rank=rank),
                    self._parallel_placement(8, rank, rank),
                    get_parallel().override(
                        dp_size=8,
                        attn_tp_size=1,
                        attn_tp_rank=0,
                        attn_dp_size=8,
                        attn_dp_rank=rank,
                    ),
                ):
                    self.assertEqual(_rank_slot({}, 0), rank)
            with (
                self._parallel_placement(8, 3, 3),
                get_parallel().override(
                    dp_size=8,
                    attn_tp_size=1,
                    attn_tp_rank=0,
                    attn_dp_size=8,
                    attn_dp_rank=3,
                ),
            ):
                self.assertEqual(_rank_slot({"rank_base": 2}, 0), 5)

    def test_g5_rank_slot_non_dp_attention_adds_replica_and_node_local_tp(self):
        with patch.dict(os.environ):
            os.environ.pop("UMBP_ENGINE_RANK", None)
            for tp_rank, replica, nodes, base, expected in (
                (1, 1, 1, 2, 7),
                (3, 1, 2, 1, 4),
                (2, 0, 2, 3, 3),
            ):
                with (
                    self.subTest(tp=tp_rank, dp=replica, nodes=nodes),
                    self._parallel_placement(4, tp_rank, replica),
                    get_parallel().override(
                        nnodes=nodes, node_rank=tp_rank // (4 // nodes)
                    ),
                ):
                    self.assertEqual(_rank_slot({"rank_base": base}, 0), expected)

    def test_g6_publication_terminal_events_are_acked_even_after_local_cancel(self):
        p = self.provider()
        reservation = p.reserve(CapacityShape("pub", 1, 0, 2, 0, 8))
        handles = reservation.take("extend", 2)
        reservation.refund_unused()
        p.publish(
            (ResidentPage("accepted", None, handles[0]),), ready=CompletedReceipt()
        )
        p.progress()
        self.assertEqual(p.client.publish_acks, [1])
        self.assertIsNone(p.client.publish_status(1))
        self.assertIn(handles[0], p._records)
        p.client.next_accepted = [False]
        p.publish(
            (ResidentPage("rejected", None, handles[1]),), ready=CompletedReceipt()
        )
        p.progress()
        self.assertIsNone(p.client.publish_status(2))
        self.assertNotIn(handles[1], p._records)
        p.publish((ResidentPage("cancelled", None, handles[1]),), ready=ManualReceipt())
        p.release_pages(handles[1:], ready=CompletedReceipt())
        self.assertNotIn(3, p._publication_results)
        p.progress()
        self.assertEqual(p.client.publish_acks, [1, 2, 3])
        self.assertFalse(p.client.statuses)
        # Native cancellation can report the terminal state after local
        # cleanup/reset; it still needs ACK, with no retained ticket history.
        p.client.emit_publish(99, PublishState.Cancelled)
        p.client.emit_publish(99, PublishState.Cancelled)
        p.progress()
        self.assertFalse(p.client.statuses)

    def test_g6_publication_unknown_retains_resources_and_acks_before_fail_stop(self):
        p = self.provider()
        reservation = p.reserve(CapacityShape("pub", 1, 0, 1, 0, 4))
        handles = reservation.take("extend", 1)
        reservation.refund_unused()
        p.client.terminal_on_mark = False
        p.publish(
            (ResidentPage("unknown", None, handles[0]),), ready=CompletedReceipt()
        )
        p.progress()
        self.assertFalse(p.client.publish_acks)  # Sent isn't a terminal receipt
        native_hold = p.client.hold_io([p._native[h] for h in handles])
        p.client.complete_io(native_hold, self.engine.CompletionProof.Unknown)
        p.client.emit_publish(1, PublishState.Unknown)
        with self.assertRaises(ProviderFailStop):
            p.progress()
        self.assertTrue(p._retained)
        self.assertEqual(p._quarantined, set(handles))
        self.assertIsNone(p.client.publish_status(1))

    def test_g6_event_overflow_fail_stops_without_discarding_held_resources(self):
        p = self.provider()
        reservation = p.reserve(CapacityShape("held", 1, 0, 1, 0, 4))
        handles = reservation.take("extend", 1)
        reservation.refund_unused()
        ticket = p.hold_io(handles, kind="pd-send", writes=False, receipt=object())
        p.client.event_overflow = True
        with self.assertRaisesRegex(ProviderFailStop, "event.*overflow"):
            p.progress()
        self.assertFalse(p.client.event_overflow)
        self.assertIn(id(ticket), p._io)
        self.assertIn(id(ticket.native), p.client.holds)
        with self.assertRaises(ProviderFailStop):
            p.reserve(CapacityShape("after-gap", 1, 0, 1, 0, 4))

    def test_g6_old_receipt_bindings_are_rejected_before_executor_creation(self):
        with (
            patch.object(Client, "acknowledge_publish_receipt", None),
            patch.object(Client, "close") as close_client,
            patch.object(self.engine, "EngineExecutor") as executor,
        ):
            with self.assertRaisesRegex(ValueError, "receipt ACK.*overflow"):
                self.provider()
            close_client.assert_called_once()
            executor.assert_not_called()

    def test_g7_prefill_forward_insert_freeze_and_tree_release_do_not_copy_to_host(
        self,
    ):
        from array import array

        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams

        for backend in ("python", "rust"):
            with self.subTest(backend=backend):
                p, allocator, cache, storage = build_provider_fixture(
                    pages=4, backend=backend, provider_override=self.provider()
                )
                source = allocator.full_attn_allocator
                reservation = p.reserve(CapacityShape("step", 1, 0, 2, 0, 8))
                with source.reservation_scope(reservation, "extend"):
                    ids = source.alloc(8)
                reservation.refund_unused()
                view = ids[:].as_subclass(_NoHostIndices)
                with patch.object(
                    torch.Tensor,
                    "cpu",
                    side_effect=AssertionError("device-to-host sync"),
                ):
                    storage.record_forward(
                        ManualReceipt(True),
                        written_tokens=view,
                        token_writes_unknown=False,
                        req_pool_indices=None,
                        state_slots=(),
                    )
                    from sglang.srt.mem_cache.provider_indices import find_index_cpu

                    self.assertIsNotNone(find_index_cpu(view))
                    frozen = p.freeze_allocations(view)
                    self.assertEqual(len(frozen.handles(view[4:])), 1)
                    cache.insert(
                        InsertParams(key=RadixKey(array("q", range(8))), value=view)
                    )
                    match = cache.match_prefix(
                        MatchPrefixParams(key=RadixKey(array("q", range(8))))
                    )
                    prefix = cache.path_device_indices(match.last_device_node)
                    self.assertIsNotNone(find_index_cpu(prefix))
                    self.assertEqual(
                        p.freeze_allocations(prefix).handles(), frozen.handles()
                    )
                    cache.confirm_detach(cache.offer_candidates(2))
                self.assertFalse(p._owned)

    def test_g7_all_full_free_entries_use_cpu_geometry_and_keep_duplicate_checks(self):
        p, allocator, _, _ = build_provider_fixture(
            pages=4, provider_override=self.provider()
        )
        source = allocator.full_attn_allocator
        reservation = p.reserve(CapacityShape("step", 1, 0, 4, 0, 16))
        with source.reservation_scope(reservation, "extend"):
            ids = source.alloc(16)
        reservation.refund_unused()
        with patch.object(
            torch.Tensor, "cpu", side_effect=AssertionError("device-to-host sync")
        ):
            source.free(ids[:4])
            source.free_segment(ids[4:8], start_pos=4)
            source.free_group_begin()
            source.free(ids[8:12])
            source.free_page_ids(torch.tensor([4]))
            source.free_group_end()
            with self.assertRaisesRegex(RuntimeError, "duplicate|stale"):
                source.free_segment(ids[:4], start_pos=0)
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                source.free_page_ids(torch.tensor([1, 1]))
        self.assertFalse(p._owned)

    def test_g7_checkpoint_preserves_request_row_cpu_mirror(self):
        from test_provider_prefill_reservation import request

        from sglang.srt.mem_cache.allocation import write_cache_indices

        p, allocator, cache, storage = build_provider_fixture(
            pages=4, provider_override=self.provider()
        )
        req = request(cache, 8)
        row = storage.acquire([req])[0]
        reservation = p.reserve(CapacityShape("step", 1, 0, 2, 0, 8))
        with allocator.full_attn_allocator.reservation_scope(reservation, "extend"):
            ids = allocator.full_attn_allocator.alloc(8)
        reservation.refund_unused()
        req.kv.kv_allocated_len = req.kv.kv_committed_len = 8
        rows, zero, end = torch.tensor([row]), torch.tensor([0]), torch.tensor([8])
        with patch.object(
            torch.Tensor, "cpu", side_effect=AssertionError("device-to-host sync")
        ):
            write_cache_indices(
                ids,
                rows,
                rows,
                zero,
                zero,
                end,
                end,
                end,
                end,
                [torch.empty(0, dtype=torch.int64)],
                storage.requests,
            )
            cache.checkpoint_into_tree(req, up_to=8)
        self.assertEqual(cache.all_values_flatten().numel(), 8)

    def test_g7_partial_extend_and_concat_cpu_geometry_matches_kernel_indices(self):
        from sglang.srt.mem_cache.provider_indices import cat_indices, index_cpu

        p, allocator, _, _ = build_provider_fixture(pages=4)
        source = allocator.full_attn_allocator
        first = p.reserve(CapacityShape("first", 1, 0, 2, 0, 5))
        zero, five, ten = torch.tensor([0]), torch.tensor([5]), torch.tensor([10])
        with source.reservation_scope(first, "extend"):
            indices = source.alloc_extend(zero, zero, five, five, torch.tensor([-1]), 5)
        first.refund_unused()
        second = p.reserve(CapacityShape("second", 1, 0, 1, 5, 10))
        with patch.object(
            torch.Tensor, "cpu", side_effect=AssertionError("device-to-host sync")
        ):
            with source.reservation_scope(second, "extend"):
                tail = source.alloc_extend(five, five, ten, ten, indices[-1:], 5)
            row = cat_indices([indices, tail])
            self.assertEqual(len(p.freeze_allocations(row).handles(row)), 3)
        second.refund_unused()
        self.assertEqual(index_cpu(tail).tolist(), tail.tolist())
        self.assertEqual(index_cpu(row[3:9]).tolist(), row[3:9].tolist())

    def test_g7_default_invariant_check_skips_census_but_forced_audit_finds_leak(self):
        from sglang.srt.environ import envs
        from sglang.srt.managers.scheduler_components.invariant_checker import (
            SchedulerInvariantChecker,
        )

        p, allocator, cache, storage = build_provider_fixture(
            pages=4, provider_override=self.provider()
        )
        from sglang.srt.disaggregation.mode import DisaggregationMode

        checker = SchedulerInvariantChecker(
            disaggregation_mode=DisaggregationMode.PREFILL,
            page_size=4,
            token_to_kv_pool_allocator=allocator,
            req_to_token_pool=storage.requests,
            tree_cache=cache,
            pool_stats_observer=SimpleNamespace(storage=storage),
            scheduler_stage_metrics=SimpleNamespace(),
        )
        with (
            patch.object(
                cache, "all_values_flatten", wraps=cache.all_values_flatten
            ) as census,
            patch.object(
                torch.Tensor, "cpu", side_effect=AssertionError("device-to-host sync")
            ),
            envs.SGLANG_KV_PROVIDER_AUDIT_INTERVAL_SECONDS.override(60.0),
        ):
            self.assertEqual(checker._check_full_pool(None), (False, ""))
            self.assertEqual(census.call_count, 1)
            ticket = p.reserve(CapacityShape("leaked", 1, 0, 1, 0, 4))
            with allocator.full_attn_allocator.reservation_scope(ticket, "extend"):
                allocator.full_attn_allocator.alloc(4)
            ticket.refund_unused()
            self.assertEqual(checker._check_full_pool(None), (False, ""))
            self.assertEqual(census.call_count, 1)
            with envs.SGLANG_KV_PROVIDER_AUDIT_INTERVAL_SECONDS.override(0.0):
                failed, message = checker._check_full_pool(None)
            self.assertTrue(failed)
            self.assertIn("projection", message)
            self.assertEqual(census.call_count, 2)

    def test_g7_row_rebind_updates_geometry_without_mutating_index_snapshots(self):
        from test_provider_prefill_reservation import request

        from sglang.srt.mem_cache.allocation import write_cache_indices
        from sglang.srt.mem_cache.provider_indices import copy_index_metadata, index_cpu

        p, allocator, cache, storage = build_provider_fixture(pages=4)
        snapshots = []
        for i in range(2):
            req = request(cache, 8)
            req.rid = f"same-prefix-{i}"
            row = storage.acquire([req])[0]
            ticket = p.reserve(CapacityShape(req.rid, 1, 0, 2, 0, 8))
            with allocator.full_attn_allocator.reservation_scope(ticket, "extend"):
                ids = allocator.full_attn_allocator.alloc(8)
            ticket.refund_unused()
            req.kv.kv_allocated_len = req.kv.kv_committed_len = 8
            rows, zero, end = torch.tensor([row]), torch.tensor([0]), torch.tensor([8])
            write_cache_indices(
                ids,
                rows,
                rows,
                zero,
                zero,
                end,
                end,
                end,
                end,
                [torch.empty(0, dtype=torch.int64)],
                storage.requests,
            )
            value = storage.requests.req_to_token[row, :8]
            snapshots.append(
                copy_index_metadata(value, value.to(dtype=torch.int64, copy=True))
            )
            with patch.object(
                torch.Tensor, "cpu", side_effect=AssertionError("device-to-host sync")
            ):
                cache.checkpoint_into_tree(req, up_to=8)
            self.assertEqual(index_cpu(value).tolist(), value.tolist())
        self.assertNotEqual(snapshots[0].tolist(), snapshots[1].tolist())
        self.assertEqual(index_cpu(snapshots[1]).tolist(), snapshots[1].tolist())
        self.assertEqual(index_cpu(value).tolist(), snapshots[0].tolist())

    def test_g7_split_and_regular_eviction_preserve_node_index_geometry(self):
        from array import array

        from sglang.srt.mem_cache.base_prefix_cache import (
            EvictParams,
            MatchPrefixParams,
        )

        for backend in ("python", "rust"):
            with self.subTest(backend=backend):
                p, allocator, cache, _ = build_provider_fixture(
                    pages=4, backend=backend, provider_override=self.provider()
                )
                ticket = p.reserve(CapacityShape("split", 1, 0, 2, 0, 8))
                with allocator.full_attn_allocator.reservation_scope(ticket, "extend"):
                    ids = allocator.full_attn_allocator.alloc(8)
                ticket.refund_unused()
                with patch.object(
                    torch.Tensor,
                    "cpu",
                    side_effect=AssertionError("device-to-host sync"),
                ):
                    cache.insert(
                        InsertParams(key=RadixKey(array("q", range(8))), value=ids)
                    )
                    cache.match_prefix(
                        MatchPrefixParams(key=RadixKey(array("q", range(4))))
                    )
                    cache.evict(EvictParams(num_tokens=8))
                self.assertFalse(p._owned)


if __name__ == "__main__":
    unittest.main()
