"""Physical sends retain allocation handles across failed/reused rooms."""

import unittest
from contextlib import nullcontext
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import PhysicalSendReceipt, SendRecord
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.static_kv_storage import kv_storage_of
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kv_memory_provider import build_provider_fixture, take_pages
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class Status:
    def __init__(self, state="pending"):
        self.state = state

    def Succeeded(self):
        return self.state == "succeeded"

    def Failed(self):
        return self.state == "failed"


class TestProviderPDSendHold(CustomTestCase):
    def setUp(self):
        scope = get_context().override_server_args()
        scope.install()
        self.addCleanup(scope.restore)

    def _setup_send(self):
        p, a, _, storage = build_provider_fixture(pages=2)
        source = a.full_attn_allocator
        ids = take_pages(source, p, 1)
        record = SendRecord()
        manager = SimpleNamespace(
            _send_records={7: record}, request_status={7: KVPoll.Transferring}
        )
        receipt = PhysicalSendReceipt(manager, 7, record, 0)
        storage.record_network_transfer(
            SimpleNamespace(),
            pages=ids[::4] // 4,
            page_size=4,
            writes=False,
            peer=receipt,
        )
        return p, source, storage, ids, manager, receipt

    def test_c23_unpublished_send_holds_survive_row_release_until_all_statuses_succeed(
        self,
    ):
        p, source, storage, ids, manager, receipt = self._setup_send()
        receipt.bind()
        first, second = Status(), Status()
        receipt.before_post()
        receipt.posted([first, second])
        receipt.seal()
        receipt.finish_enqueue()
        source.free(ids)
        self.assertEqual(p.accounting().pending_free, 1)
        first.state = "succeeded"
        storage.end_network_transfer(SimpleNamespace())
        storage.poll_provider_io()
        p.progress()
        self.assertEqual(p.accounting().pending_free, 1)
        second.state = "succeeded"
        storage.poll_provider_io()
        storage.poll_provider_io()  # duplicate polling cannot release twice
        p.progress()
        self.assertEqual(p.accounting().cached, 2)

    def test_c23_partial_post_failure_room_clear_and_late_success_stay_isolated(self):
        for failure in ("failed_status", "failed_room", "clear", "partial_raise"):
            with self.subTest(failure=failure):
                p, source, storage, ids, manager, receipt = self._setup_send()
                receipt.bind()
                status = Status()
                receipt.before_post()
                receipt.posted([status])
                receipt.finish_enqueue()
                if failure == "failed_status":
                    status.state = "failed"
                    receipt.seal()
                elif failure == "failed_room":
                    manager.request_status[7] = KVPoll.Failed
                elif failure == "clear":
                    manager._send_records[7] = SendRecord()
                else:
                    receipt.seal(unknown=True)
                receipt.seal()
                source.free(ids)
                storage.poll_provider_io()
                self.assertEqual(p.accounting().quarantined, 1)
                self.assertEqual(source.available_size(), 4)
                self.assertIs(p.quarantine_resources[0][1], receipt)
                status.state = "succeeded"
                storage.poll_provider_io()
                p.progress()
                self.assertEqual(p.accounting().quarantined, 1)

    def test_c23_never_enqueued_can_cancel_hold(self):
        p, source, storage, ids, _, receipt = self._setup_send()
        receipt.finish_enqueue()
        source.free(ids)
        storage.poll_provider_io()
        p.progress()
        self.assertEqual(p.accounting().cached, 2)

    def test_g1b_rebound_storages_share_pending_io_through_repeated_polls(self):
        p, source, original, first_ids, manager, first = self._setup_send()
        first.bind()
        first.before_post()
        first_status = Status()
        first.posted([first_status])
        first.seal()
        first.finish_enqueue()
        source.free(first_ids)
        table = ReqToTokenPool(
            size=2, max_context_len=8, device="cpu", enable_memory_saver=False
        )
        rebound = kv_storage_of(table, original._allocator)
        shared = original._provider_io
        self.assertIs(rebound._provider_io, shared)
        original.poll_provider_io()
        self.assertIs(original._provider_io, shared)
        self.assertIs(rebound._provider_io, shared)
        self.assertEqual(len(shared), 1)

        second_ids = take_pages(source, p, 1)
        second = PhysicalSendReceipt(manager, 7, first.record, 1)
        rebound.record_network_transfer(
            SimpleNamespace(),
            pages=second_ids[::4] // 4,
            page_size=4,
            writes=False,
            peer=second,
        )
        second.bind()
        second.before_post()
        second_status = Status()
        second.posted([second_status])
        second.seal()
        second.finish_enqueue()
        source.free(second_ids)
        with patch.object(p, "complete_io", wraps=p.complete_io) as completed:
            original.poll_provider_io()
            self.assertEqual(len(shared), 2)
            first_status.state = "succeeded"
            rebound.poll_provider_io()
            original.poll_provider_io()
            p.progress()
            self.assertEqual(completed.call_count, 1)
            self.assertEqual(len(shared), 1)
            self.assertEqual(p.accounting().pending_free, 1)
            self.assertEqual(p.accounting().cached, 1)
            second_status.state = "succeeded"
            original.poll_provider_io()
            rebound.poll_provider_io()
            p.progress()
            self.assertEqual(completed.call_count, 2)
            self.assertFalse(shared)
            self.assertEqual(p.accounting().cached, 2)
        self.assertIs(original._provider_io, rebound._provider_io)

    def test_c23_mori_enqueue_partial_submission_retains_native_resources(self):
        import numpy as np

        from sglang.srt.disaggregation.mode import DisaggregationMode
        from sglang.test.mori_stub import mori_stub

        stub = mori_stub()
        with stub if stub is not None else nullcontext():
            from sglang.srt.disaggregation.mori.conn import (
                BatchTransferPlan,
                MoriKVManager,
                _MoriTransferSubmissionError,
                _SubmissionLocal,
            )

            p, a, _, storage = build_provider_fixture()
            ids = take_pages(a.full_attn_allocator, p, 1)
            manager = MoriKVManager.__new__(MoriKVManager)
            manager.disaggregation_mode = DisaggregationMode.PREFILL
            manager.enable_deferred_decode_kv_release = False
            manager._submission_local = _SubmissionLocal()
            manager.request_status = {7: KVPoll.Transferring}
            manager.check_status = lambda room: manager.request_status[room]
            manager.transfer_infos = {7: {}}
            manager._num_shards = 1
            manager._transfer_queues = [Queue()]
            storage.bind_network_manager(manager)
            status = Status()
            posted = []

            def native_post(*args):
                posted.append(args)
                if len(posted) == 2:
                    raise RuntimeError("partial native post")
                return [status]

            manager.engine = native = SimpleNamespace(
                batch_write=native_post, allocate_transfer_uid=lambda: 1
            )
            receipt = manager.prepare_physical_send(7)
            storage.record_network_transfer(
                SimpleNamespace(),
                pages=ids[::4] // 4,
                page_size=4,
                writes=False,
                peer=receipt,
            )
            manager.add_transfer_request(
                7, np.array([1], dtype=np.int32), slice(0, 1), False
            )
            receipt.finish_enqueue()
            chunk = manager._transfer_queues[0].get_nowait()
            self.assertIs(chunk.physical_receipt, receipt)
            self.assertEqual(receipt.sequence, 0)
            plan = BatchTransferPlan(local_offsets=[0], remote_offsets=[0], sizes=[4])

            def submit(*args, **kwargs):
                first = manager._submit_batch_transfer_plan(object(), object(), plan)
                return first + manager._submit_batch_transfer_plan(
                    object(), object(), plan
                )

            manager._submit_kv_transfer = submit
            with self.assertRaises(_MoriTransferSubmissionError):
                manager._submit_transfer_chunk(chunk)
            self.assertEqual(receipt.statuses, [status])
            a.full_attn_allocator.free(ids)
            storage.poll_provider_io()
            self.assertEqual(p.accounting().quarantined, 1)
            from sglang.srt.mem_cache.kv_memory_provider import ProviderFailStop

            with self.assertRaises(ProviderFailStop):
                manager.teardown()
            self.assertIs(manager.engine, native)
            self.assertFalse(getattr(manager, "_stopped", False))


if __name__ == "__main__":
    unittest.main()
