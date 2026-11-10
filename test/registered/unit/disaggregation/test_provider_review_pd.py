"""S1F physical-send regressions through real prefill/sender/worker entry points."""

import threading
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import PhysicalSendReceipt
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kv_memory_provider import (
    ManualReceipt,
    build_provider_fixture,
    take_pages,
)
from sglang.test.mori_stub import mori_stub
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestProviderReviewPD(CustomTestCase):
    def setUp(self):
        self.enterContext(get_context().override_server_args())
        stub = mori_stub()
        if stub is not None:
            self.enterContext(stub)
        from sglang.srt.disaggregation.mori.conn import MoriKVManager, _SubmissionLocal

        self.manager = MoriKVManager.__new__(MoriKVManager)
        m = self.manager
        m.disaggregation_mode = DisaggregationMode.PREFILL
        m.enable_deferred_decode_kv_release = False
        m._submission_local = _SubmissionLocal()
        m.transfer_lock = threading.Lock()
        m.request_status = {7: KVPoll.Transferring}
        m.transfer_infos = {7: {}}
        m._num_shards = 1
        m._transfer_queues = [Queue()]
        m.failure_lock, m.failure_records = threading.Lock(), {}
        m.send_kv_status_message = lambda **kw: None  # external network boundary
        m.attn_cp_size = 1
        m._dcp_pack_max_tokens = None
        m.kv_shard_size = 1
        m.enable_all_cp_ranks_for_transfer = m.is_dummy_cp_rank = False
        self.p, self.a, self.c, self.storage = build_provider_fixture()

    def _chunk(self):
        ids = take_pages(self.a.full_attn_allocator, self.p, 1)
        receipt = self.manager.prepare_physical_send(7)
        self.storage.record_network_transfer(
            SimpleNamespace(),
            pages=ids[::4] // 4,
            page_size=4,
            writes=False,
            peer=receipt,
        )
        receipt.bind()
        receipt.finish_enqueue()
        chunk = TransferKVChunk(
            room=7,
            prefill_kv_indices=np.array([1], dtype=np.int32),
            index_slice=slice(0, 1),
            is_last_chunk=False,
            prefill_aux_index=None,
            state_indices=None,
            physical_receipt=receipt,
        )
        return ids, receipt, chunk

    def test_s_p2_3_error_before_first_post_is_never_submitted(self):
        from sglang.srt.disaggregation.mori.conn import _MoriTransferSubmissionError

        ids, receipt, chunk = self._chunk()

        # Real _submit_kv_transfer rejects the missing transfer-info record
        # before building any native plan or issuing a post.
        with self.assertRaisesRegex(_MoriTransferSubmissionError, "No transfer info"):
            self.manager._submit_transfer_chunk(chunk)
        self.assertEqual(receipt.attempts, 0)
        self.assertEqual(receipt.poll(), "never_submitted")

    def test_s_p2_3_generic_worker_exception_seals_never_posted_send(self):
        ids, receipt, chunk = self._chunk()
        chunk.wait_event = SimpleNamespace(
            synchronize=lambda: (_ for _ in ()).throw(
                ValueError("GPU dependency failure")
            )
        )
        queue = Queue()
        queue.put(chunk)
        queue.put(None)
        self.manager._transfer_worker(queue)
        self.assertTrue(receipt.sealed)
        self.assertEqual(receipt.poll(), "never_submitted")
        self.assertEqual(self.manager.request_status[7], KVPoll.Failed)

    def test_s_p2_3_worker_skip_after_failed_room_preserves_never_submitted(self):
        ids, receipt, chunk = self._chunk()
        self.a.full_attn_allocator.free(ids)
        self.manager.request_status[7] = KVPoll.Failed
        self.storage.poll_provider_io()
        self.assertEqual(self.p.accounting().pending_free, 1)
        self.assertTrue(self.manager._process_transfer_chunk(chunk))
        self.storage.poll_provider_io()
        self.p.progress()
        self.assertEqual(self.p.accounting().cached, 16)
        self.assertEqual(self.p.accounting().quarantined, 0)

    def test_p3_worker_wait_ready_uses_one_blocking_wait(self):
        ids, receipt, chunk = self._chunk()
        waiting, release = threading.Event(), threading.Event()

        class Completion(ManualReceipt):
            waits = 0

            def wait(self):
                self.waits += 1
                waiting.set()
                assert release.wait(5)
                self.done = True

        dependency = Completion()
        receipt.ready = dependency
        # The real submit path handles a dummy CP destination without native I/O.
        self.manager.transfer_infos = {
            7: {0: SimpleNamespace(engine_key="dummy", is_dummy=True)}
        }
        self.manager.decode_kv_args_table = {"dummy": SimpleNamespace()}
        result, errors = [], []

        def run():
            try:
                result.append(self.manager._process_transfer_chunk(chunk))
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=run)
        with patch(
            "sglang.srt.disaggregation.common.conn.time.sleep",
            side_effect=lambda _: release.wait(0.001),
        ) as sleep:
            worker.start()
            observed = waiting.wait(1)
            release.set()
            dependency.done = True  # allow the pre-fix polling worker to exit, too
            worker.join(5)
            self.assertTrue(observed)
            self.assertEqual(sleep.call_count, 0)
        self.assertFalse(worker.is_alive())
        self.assertFalse(errors)
        self.assertEqual(dependency.waits, 1)
        self.assertEqual(result, [True])

    def _prefill_chunk(self):
        from array import array

        from sglang.srt.disaggregation.mori.conn import MoriKVSender
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.observability.trace import TraceNullContext
        from sglang.srt.sampling.sampling_params import SamplingParams

        ids = take_pages(self.a.full_attn_allocator, self.p, 2)
        req = Req(
            rid="send",
            origin_input_text=None,
            origin_input_ids=array("q", range(8)),
            sampling_params=SamplingParams(max_new_tokens=0),
        )
        req.refresh_fill_ids()
        row = self.storage.acquire([req])[0]
        self.storage.requests.req_to_token[row, :8] = ids
        req.kv.set_row_len(8)
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams
        from sglang.srt.mem_cache.radix_cache import RadixKey

        self.c.insert(InsertParams(key=RadixKey(array("q", range(8))), value=ids))
        req.kv.protect_prefix(8)
        req.extend_end = 8
        sender = MoriKVSender.__new__(MoriKVSender)
        sender.kv_mgr, sender.bootstrap_room = self.manager, 7
        sender.conclude_state, sender.curr_idx, sender.num_kv_indices = None, 0, 2
        sender.aux_index = None
        sender._early_send_wait_event = sender._transfer_start_time = None
        sender._transfer_num_kv_indices = sender._transfer_num_state_indices = 0
        sender.trace_ctx = TraceNullContext()
        req.disagg_kv_sender = sender
        scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=self.a,
            req_to_token_pool=self.storage.requests,
            tree_cache=self.c,
            enable_staging=False,
        )
        return ids, req, sender, scheduler

    def test_p3_real_prefill_prepare_hold_send_finish_order(self):
        from sglang.srt.disaggregation.prefill import (
            SchedulerDisaggregationPrefillMixin,
        )

        ids, req, sender, scheduler = self._prefill_chunk()
        order = []
        prepare, hold, send, finish = (
            self.manager.prepare_physical_send,
            self.storage.record_network_transfer,
            sender.send,
            PhysicalSendReceipt.finish_enqueue,
        )

        def preparing(*args):
            order.append("prepare")
            return prepare(*args)

        def holding(*args, **kw):
            order.append("hold")
            return hold(*args, **kw)

        def sending(*args, **kw):
            order.append("send")
            self.assertTrue(self.p.io)
            return send(*args, **kw)

        def finishing(receipt):
            order.append("finish")
            return finish(receipt)

        with (
            patch.object(self.manager, "prepare_physical_send", side_effect=preparing),
            patch.object(self.storage, "record_network_transfer", side_effect=holding),
            patch.object(sender, "send", side_effect=sending),
            patch.object(PhysicalSendReceipt, "finish_enqueue", new=finishing),
        ):
            SchedulerDisaggregationPrefillMixin._send_kv_chunk(
                scheduler, req, end_idx=4
            )
        self.assertEqual(order, ["prepare", "hold", "send", "finish"])
        self.assertEqual(req.start_send_idx, 4)
        chunk = self.manager._transfer_queues[0].get_nowait()
        self.assertTrue(
            chunk.physical_receipt.bound and chunk.physical_receipt.enqueue_finished
        )
        self.assertEqual(chunk.prefill_kv_indices.tolist(), [int(ids[0]) // 4])

    def test_g1b_scheduler_exception_after_enqueue_retains_source_before_late_post(
        self,
    ):
        from sglang.srt.disaggregation.prefill import (
            SchedulerDisaggregationPrefillMixin,
        )

        ids, req, sender, scheduler = self._prefill_chunk()
        send = sender.send

        def enqueue_then_raise(*args, **kwargs):
            send(*args, **kwargs)
            raise RuntimeError("scheduler failed after worker enqueue")

        with patch.object(sender, "send", side_effect=enqueue_then_raise):
            with self.assertRaisesRegex(RuntimeError, "after worker enqueue"):
                SchedulerDisaggregationPrefillMixin._send_kv_chunk(
                    scheduler, req, end_idx=4
                )
        chunk = self.manager._transfer_queues[0].get_nowait()
        receipt = chunk.physical_receipt
        self.assertTrue(receipt.bound and receipt.sealed and receipt.enqueue_finished)
        self.assertEqual(receipt.attempts, 0)  # worker has not started yet
        self.assertEqual(receipt.poll(), "unknown")
        self.c.confirm_detach(self.c.offer_candidates(2))
        self.storage.poll_provider_io()
        self.p.progress()
        self.assertEqual(self.p.accounting().quarantined, 1)
        self.assertEqual(self.p.accounting().cached, 15)
        self.assertIs(self.p.quarantine_resources[0][1], receipt)

        # Execute the worker's real post wrapper after the scheduler exception.
        # Only the native engine boundary is replaced; late success cannot
        # turn this unresolved source hold into permission to reuse the page.
        status = SimpleNamespace(Succeeded=lambda: True, Failed=lambda: False)
        self.manager.engine = SimpleNamespace(batch_write=lambda *args: [status])
        self.manager._submission_local.receipt = receipt
        try:
            self.manager._engine_batch_write()
        finally:
            self.manager._submission_local.receipt = None
        self.assertEqual(receipt.attempts, 1)
        self.assertEqual(receipt.poll(), "unknown")
        self.storage.poll_provider_io()
        self.p.progress()
        self.assertEqual(self.p.accounting().quarantined, 1)
        self.assertEqual(self.p.accounting().cached, 15)


if __name__ == "__main__":
    unittest.main()
