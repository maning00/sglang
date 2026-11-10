"""Real node/EngineClient CPU integration; no GPU or native dependency in CPU CI.

Run with UMBP_STANDALONE_BIN and the staged mori package in PYTHONPATH.
Host bytes replace HBM; the node, transport, executor, DRAM/F3 and ownership
state machine are real. Missing native protocol capabilities are explicit.
"""

import ctypes
import json
import mmap
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.allocator.provider import ProviderPageSource
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.kv_memory_provider import (
    Busy,
    CapacityShape,
    CompletedReceipt,
    KVMemoryProvider,
    PageGeometry,
    PageRegion,
    ProviderFailStop,
    ResidentPage,
)
from sglang.srt.mem_cache.storage.umbp.engine_provider import UMBPEngineProvider
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.kv_memory_provider import ManualReceipt
from sglang.test.test_utils import CustomTestCase


class _EngineFixture(CustomTestCase):
    retained_unknown = []
    row_bytes = (32,)
    model_layers = 1
    layout_id = "cpu-layout-v1"

    @classmethod
    def setUpClass(cls):
        binary = os.environ.get("UMBP_STANDALONE_BIN")
        if not binary or not Path(binary).is_file():
            raise unittest.SkipTest("requires the external UMBP standalone build")
        try:
            from mori.umbp import engine
        except ImportError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        cls.engine, cls.binary = engine, binary

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(
            prefix="sgl-engine-", dir=os.environ.get("TMPDIR")
        )
        self.addCleanup(self.tmp.cleanup)
        self.address = "unix://" + self.tmp.name + "/node.sock"
        self.log = open(Path(self.tmp.name) / "node.log", "w+")
        self.addCleanup(self.log.close)
        env = dict(os.environ)
        for key in tuple(env):
            if key.startswith("UMBP_"):
                del env[key]
        env.update(
            UMBP_ENGINE_HBM_RANKS="8",
            UMBP_DISTRIBUTED_MEDIUM="DRAM",
            UMBP_DRAM_CAPACITY="1048576",
            UMBP_DISTRIBUTED_DRAM_PAGE_SIZE="4096",
            UMBP_DRAM_USE_HUGEPAGES="0",
            UMBP_DRAM_PREFAULT="0",
            UMBP_HBM_GATHER_PREWARM_STREAMS="0",
            MORI_IO_SIM_ENABLE="1",
            MORI_IO_SIM_BACKEND="RDMA",
        )
        self.server = subprocess.Popen(
            [self.binary, self.address],
            env=env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        self.addCleanup(self._stop_server)
        for _ in range(500):
            if Path(self.tmp.name + "/node.sock.engine.sock").exists():
                break
            if self.server.poll() is not None:
                self.log.seek(0)
                self.fail(self.log.read())
            time.sleep(0.01)
        else:
            self.fail("node engine socket not ready")
        self.p = UMBPEngineProvider(
            device="cpu",
            options={
                "node_address": self.address,
                "namespace": {
                    "model": "cpu-dsv4-geometry",
                    "weights": "revision-1",
                    "kv_format": "fp8",
                    "parallel": "dep8",
                },
                "heartbeat_ms": 10,
                "heartbeat_timeout_ms": 1000,
                "dram": {"min_bytes": 4096},
            },
        )
        self.addCleanup(self._close_provider)
        self.backing = ctypes.create_string_buffer(5 * sum(self.row_bytes))
        regions, offset = [], 0
        for layer, row_bytes in enumerate(self.row_bytes):
            regions.append(
                PageRegion(
                    "kv" if layer == 0 else f"kv{layer}",
                    ctypes.addressof(self.backing) + 5 * offset,
                    5 * row_bytes,
                    row_bytes,
                    1,
                    offset,
                    layer,
                    "kv",
                )
            )
            offset += row_bytes
        geometry = PageGeometry(
            4,
            4,
            sum(self.row_bytes),
            self.layout_id,
            tuple(regions),
        )
        self.p.register_pool(geometry, ())
        self.a = ProviderPageSource(
            size=16,
            page_size=4,
            dtype=torch.float32,
            device="cpu",
            kvcache=SimpleNamespace(layer_num=self.model_layers, start_layer=0),
            need_sort=True,
            provider=self.p,
        )
        self.assertIsInstance(self.p, KVMemoryProvider)

    def _stop_server(self):
        if self.server.poll() is None:
            self.server.terminate()
            try:
                self.server.wait(5)
            except subprocess.TimeoutExpired:
                self.server.kill()
                self.server.wait()

    def _close_provider(self):
        try:
            self.p.close()
        except ProviderFailStop:
            # Unknown resources are intentionally process-lifetime retained.
            self.retained_unknown.append((self.p, self.backing))

    def wait_for(self, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            self.p.progress()
            if predicate():
                return
            time.sleep(0.005)
        self.fail(f"state did not progress: {self.p.accounting()}")

    def take(self, n=1):
        ticket = self.p.reserve(CapacityShape("req", 1, 0, n, 0, n * 4))
        with self.a.reservation_scope(ticket, "extend"):
            indices = self.a.alloc(n * 4)
        ticket.refund_unused()
        return indices, self.p.freeze_allocations(indices).handles()

    def publish(self, handles, ready=CompletedReceipt()):
        records = tuple(
            ResidentPage(f"key-{h.page_id}-{h.generation}", None, h) for h in handles
        )
        self.p.publish(records, ready=ready)
        return records


class TestUMBPEngineProvider(_EngineFixture):
    def test_register_take_publish_retire_released_reuse(self):
        indices, handles = self.take(4)
        patterns = [bytes([31 + h.page_id]) * 32 for h in handles]
        for handle, pattern in zip(handles, patterns):
            ctypes.memmove(
                ctypes.addressof(self.backing) + handle.page_id * 32, pattern, 32
            )
        self.publish(handles)
        self.wait_for(lambda: self.p.accounting().published == 4)
        self.a.free(indices)
        self.wait_for(lambda: self.p.accounting().cached == 4)
        self.assertFalse(self.p.accounting_violations())
        dram = []
        for region in self.p.registration.dram_regions:
            with mmap.mmap(
                region.fd, region.mapped_length, access=mmap.ACCESS_READ
            ) as mapped:
                dram.append(mapped[region.fd_offset : region.fd_offset + region.length])
        for pattern in patterns:
            self.assertTrue(any(pattern in payload for payload in dram))
        again, new = self.take(4)
        before = {h.page_id: h.generation for h in handles}
        self.assertTrue(all(h.generation > before[h.page_id] for h in new))
        self.a.free(again)
        self.assertEqual(self.p.accounting().cached, 4)

    def test_g2_dram_readback_uses_real_lookup_prepare_and_layered_executor(self):
        if not hasattr(self.p.client, "lookup"):
            self.skipTest("requires the staged mori I9 read API")
        source_indices, handles = self.take(2)
        patterns = [bytes((17 * i + j) % 256 for j in range(32)) for i in (1, 2)]
        for h, pattern in zip(handles, patterns):
            ctypes.memmove(ctypes.addressof(self.backing) + h.page_id * 32, pattern, 32)
        self.publish(handles)
        self.wait_for(lambda: self.p.accounting().published == 2)
        self.a.free(source_indices)
        self.wait_for(lambda: self.p.accounting().cached == 4)
        ctypes.memset(ctypes.addressof(self.backing), 0, len(self.backing))
        linker = self.p.create_linker(
            None,
            SimpleNamespace(token_to_kv_pool_allocator=self.a),
            components={ComponentType.FULL},
            forward_stream=None,
        )
        keys = [f"key-{h.page_id}-{h.generation}" for h in handles]
        self.assertEqual(
            linker.lookup("dram-readback", [PoolTransfer(PoolName.KV, keys=keys)]),
            [1, 2],
        )
        reservation = self.p.reserve(CapacityShape("dram-readback", 1, 2, 0, 0, 8))
        with self.a.reservation_scope(reservation, "load"):
            targets = self.a.alloc(8)
        self.assertTrue(
            linker.load(
                "dram-readback",
                [PoolTransfer(PoolName.KV, device_indices=targets, keys=keys)],
            )
        )
        reservation.refund_unused()
        index = linker.start_layer_wise_loading()
        linker.layer_done_counter.set_consumer(index)
        linker.layer_done_counter.wait_until(0)
        self.assertTrue(linker.finish_layer_wise_loading(index))
        target_handles = self.p.freeze_allocations(targets).handles()
        for handle, expected in zip(target_handles, patterns):
            self.assertEqual(
                ctypes.string_at(
                    ctypes.addressof(self.backing) + handle.page_id * 32, 32
                ),
                expected,
            )
        self.assertEqual(linker.stats["completed_LocalDram_pages"], 2)
        self.assertIs(linker.client, self.p.client)
        self.assertIs(linker.executor, self.p.executor)
        self.a.free(targets)
        self.assertEqual(self.p.accounting().cached, 4)

    def test_pending_batch_cancellation_keeps_unreturned_siblings_published(self):
        indices, handles = self.take(4)
        ready = ManualReceipt()
        self.publish(handles, ready)
        self.a.free(indices[:4])
        self.assertEqual(self.p.accounting().cached, 1)
        ready.done = True
        self.wait_for(lambda: self.p.accounting().published == 3)
        self.assertEqual(self.p.accounting().cached, 1)
        self.a.free(indices[4:])
        self.wait_for(lambda: self.p.accounting().cached == 4)

    def test_forward_receipt_delays_native_publication_without_device_roundtrip(self):
        indices, handles = self.take()
        done = ManualReceipt()
        self.p.record_forward(handles, done)
        frozen = self.p.freeze_allocations(None, page_ids=(handles[0].page_id,))
        self.publish(handles, frozen.ready(handles))
        self.p.progress()
        self.assertEqual(self.p.accounting().published, 0)
        self.assertEqual(self.p.accounting().in_use, 1)
        done.done = True
        self.wait_for(lambda: self.p.accounting().published == 1)
        self.a.free(indices)
        self.wait_for(lambda: self.p.accounting().cached == 4)

    def test_load_ticket_revise_grows_extend_without_refunding_inflight_target(self):
        ticket = self.p.reserve(CapacityShape("load-request", 1, 1, 1, 0, 8))
        with self.a.reservation_scope(ticket, "load"):
            target = self.a.alloc(4)
        handles = self.p.freeze_allocations(target).handles()
        hold = self.p.hold_io(
            handles, kind="load-target", writes=True, receipt=object()
        )
        self.assertTrue(ticket.revise(CapacityShape("load-request", 2, 0, 2, 4, 12)))
        self.assertEqual(self.p.accounting().reserved, 2)
        self.assertEqual(self.p.accounting().in_use, 1)
        with self.a.reservation_scope(ticket, "extend"):
            extend = self.a.alloc(8)
        ticket.refund_unused()
        self.assertEqual(self.p.accounting().in_use, 3)
        self.p.complete_io(hold, outcome="completed")
        self.a.free(torch.cat((target, extend)))
        self.assertEqual(self.p.accounting().cached, 4)

    def test_pd_hold_and_published_reuse_guard_wait_for_both_proofs(self):
        indices, handles = self.take()
        self.publish(handles)
        self.wait_for(lambda: self.p.accounting().published == 1)
        receipt = object()
        io = self.p.hold_io(handles, kind="pd-send", writes=False, receipt=receipt)
        guard = ManualReceipt()
        self.a.set_reuse_guard_event(guard)
        self.a.free(indices)
        self.assertEqual(self.p.accounting().cached, 3)
        # The safety contract admits an early Retire (old runtime) or a
        # delayed Retire (I10); both must keep the guarded page unavailable.
        snapshot = self.p.accounting()
        self.assertEqual(snapshot.published + snapshot.retiring, 1)
        self.p.complete_io(io, outcome="completed")
        self.wait_for(lambda: self.p.accounting().retiring == 0)
        self.assertEqual(self.p.accounting().cached, 3)
        snapshot = self.p.accounting()
        self.assertEqual(snapshot.published + snapshot.pending_free, 1)
        from sglang.srt.mem_cache.kv_memory_provider import Short

        self.assertIsInstance(
            self.p.reserve(CapacityShape("guarded", 1, 0, 4, 0, 16)), Short
        )
        guard.done = True
        self.wait_for(lambda: self.p.accounting().cached == 4)

    def test_busy_reset_preserves_epoch_and_reopens_allocation(self):
        indices, handles = self.take()
        ticket = self.p.hold_io(handles, kind="pd-send", writes=False, receipt=object())
        epoch = self.p.epoch
        lease = self.p.quiesce_reset(deadline_ns=time.monotonic_ns() + 1_000_000_000)
        self.assertIsInstance(lease, Busy)
        self.assertEqual(self.p.epoch, epoch)
        reservation = self.p.reserve(CapacityShape("after-busy", 1, 0, 1, 0, 4))
        reservation.refund_unused()
        self.p.complete_io(ticket, outcome="never_submitted")
        self.a.free(indices)
        native = self.p.client.quiesce_reset()
        self.assertTrue(native.ready)
        self.assertTrue(self.p.client.finish_reset(native, False))
        self.assertEqual(self.p.accounting().cached, 4)

    def test_unknown_isolation_is_not_native_reset_busy(self):
        indices, handles = self.take()
        resource = object()
        ticket = self.p.hold_io(handles, kind="pd-send", writes=False, receipt=resource)
        self.a.free(indices)
        self.p.complete_io(ticket, outcome="unknown")
        self.assertEqual(self.p.accounting().quarantined, 1)
        self.assertEqual(len(self.p._io), 0)
        old_epoch = self.p.epoch
        lease = self.p.quiesce_reset(deadline_ns=time.monotonic_ns() + 1_000_000_000)
        self.assertNotIsInstance(lease, Busy)
        self.assertNotIn(handles[0].page_id, lease.approved_pages)
        self.a.clear(reset_lease=lease)
        self.p.commit_reset(lease)
        self.a.finish_reset()
        self.assertEqual(self.p.epoch.reset_seq, old_epoch.reset_seq + 1)
        self.assertEqual(self.p.accounting().cached, 3)
        self.assertEqual(self.p.accounting().quarantined, 1)
        reused, new = self.take(3)
        self.assertNotIn(handles[0].page_id, {h.page_id for h in new})
        self.publish(new)
        self.wait_for(lambda: self.p.accounting().published == 3)
        self.a.free(reused)
        self.wait_for(lambda: self.p.accounting().cached == 3)
        self.assertIs(self.p._retained[0][0].receipt, resource)
        with self.assertRaises(ProviderFailStop):
            self.p.close()

    def test_g5_unknown_read_preserves_live_pages_until_all_native_returns(self):
        from sglang.srt.mem_cache.kv_memory_provider import Short

        for published in (False, True):
            with self.subTest(published=published):
                indices, handles = self.take(2)
                if published:
                    self.publish(handles)
                    self.wait_for(lambda: self.p.accounting().published == 2)
                io = self.p.hold_io(
                    handles, kind="pd-send", writes=False, receipt=object()
                )
                self.p.complete_io(io, outcome="unknown")
                self.assertTrue(self.p.has_pending_io)
                self.assertEqual(self.p.accounting().quarantined, 2 if published else 0)
                # The previous subcase's quarantined pages are permanent. The
                # current source capabilities remain valid for another reader.
                self.assertEqual(self.p.freeze_allocations(indices).handles(), handles)
                reader = self.p.hold_io(
                    handles, kind="pd-send", writes=False, receipt=object()
                )
                self.p.complete_io(reader, outcome="completed")
                self.assertIsInstance(
                    self.p.quiesce_reset(
                        deadline_ns=time.monotonic_ns() + 1_000_000_000
                    ),
                    Busy,
                )
                self.a.free(indices[:4])
                self.assertTrue(self.p.has_pending_io)
                state = self.p.accounting()
                self.assertEqual(state.cached, 0 if published else 2)
                self.assertEqual(state.pending_free, 0 if published else 1)
                self.assertEqual(state.published, 2 if published else 0)
                self.assertEqual(state.retiring, 0)
                self.a.free(indices[4:])
                self.assertFalse(self.p.has_pending_io)
                self.assertEqual(self.p.accounting().quarantined, 4 if published else 2)
                self.assertFalse(self.p.accounting_violations())
        self.assertIsInstance(
            self.p.reserve(CapacityShape("cannot-reissue", 1, 0, 1, 0, 4)), Short
        )

    def test_g5_parked_read_allows_native_publication_then_isolates_on_return(self):
        indices, handles = self.take()
        ticket = self.p.hold_io(handles, kind="pd-send", writes=False, receipt=object())
        self.p.complete_io(ticket, outcome="unknown")
        self.assertEqual(self.p.accounting().quarantined, 0)
        self.p.record_forward(handles, CompletedReceipt())
        self.publish(handles)
        self.wait_for(lambda: self.p.accounting().published == 1)
        self.assertEqual(self.p.freeze_allocations(indices).handles(), handles)
        self.a.free(indices)
        self.assertFalse(self.p.has_pending_io)
        self.assertEqual(self.p.accounting().quarantined, 1)
        self.assertEqual(self.p.accounting().cached, 3)

    def test_g3_aborted_ready_reset_resumes_pending_publication(self):
        indices, handles = self.take()
        ready = ManualReceipt()
        self.publish(handles, ready)
        ticket = next(iter(self.p._publications))
        epoch = self.p.epoch
        lease = self.p.quiesce_reset(deadline_ns=time.monotonic_ns() + 1_000_000_000)
        self.assertNotIsInstance(lease, Busy)
        self.assertEqual(
            self.p.client.publish_status(ticket).state,
            self.engine.PublishReceiptState.Queued,
        )
        self.p.abort_reset(lease)
        self.assertEqual(self.p.epoch, epoch)
        ready.done = True
        self.wait_for(lambda: self.p.accounting().published == 1)
        self.assertNotIn(ticket, self.p._publications)
        self.assertFalse(self.p._publication_results)
        self.a.free(indices)
        self.wait_for(lambda: self.p.accounting().cached == 4)

    def test_g3_busy_reset_resumes_pending_publication_and_drains_receipts(self):
        indices, handles = self.take()
        ready = ManualReceipt()
        self.publish(handles, ready)
        ticket = next(iter(self.p._publications))
        io = self.p.hold_io(handles, kind="pd-send", writes=False, receipt=object())
        epoch = self.p.epoch
        lease = self.p.quiesce_reset(deadline_ns=time.monotonic_ns() + 1_000_000_000)
        self.assertIsInstance(lease, Busy)
        self.assertEqual(self.p.epoch, epoch)
        self.assertEqual(
            self.p.client.publish_status(ticket).state,
            self.engine.PublishReceiptState.Queued,
        )
        ready.done = True
        self.wait_for(lambda: self.p.accounting().published == 1)
        self.assertNotIn(ticket, self.p._publications)
        self.assertFalse(self.p._unready_publications)
        self.assertFalse(self.p._publication_by_handle)
        self.assertFalse(self.p._publication_results)
        self.assertIsNone(self.p.client.publish_status(ticket))  # terminal ACK
        self.a.free(indices)
        self.assertEqual(self.p.accounting().cached, 3)
        self.assertEqual(self.p.accounting().published, 1)  # local hold delays Retire
        self.p.complete_io(io, outcome="completed")
        self.wait_for(lambda: self.p.accounting().cached == 4)

    def test_g3_rejected_publication_removes_manifest_and_allows_retry(self):
        indices, handles = self.take(2)
        first = ResidentPage("collision", None, handles[0])
        retry = ResidentPage("collision", None, handles[1])
        self.p.publish((first,), ready=CompletedReceipt())
        self.wait_for(lambda: self.p.accounting().published == 1)
        self.p.publish((retry,), ready=CompletedReceipt())
        ticket = next(iter(self.p._publications))
        self.wait_for(lambda: handles[1] not in self.p._records)
        self.assertIsNone(self.p.client.publish_status(ticket))  # terminal ACK
        self.assertNotIn(ticket, self.p._publications)
        self.assertFalse(self.p._publication_results)
        self.a.free(indices[:4])
        self.wait_for(lambda: self.p.accounting().cached == 3)
        self.p.publish((retry,), ready=CompletedReceipt())
        self.wait_for(lambda: self.p.accounting().published == 1)
        self.assertIn(handles[1], self.p._records)
        self.assertFalse(self.p._publications)
        self.assertFalse(self.p._publication_results)
        self.a.free(indices[4:])
        self.wait_for(lambda: self.p.accounting().cached == 4)

    def test_g3_node_busy_from_peer_grant_resumes_pending_publication(self):
        indices, handles = self.take(2)
        source = self.publish(handles[:1])[0]
        self.wait_for(lambda: self.p.accounting().published == 1)
        # A separate process preserves one EngineClient per rank process. Its
        # prepared (never submitted) read holds a node grant on our source.
        script = """
import ctypes,json,sys
from pathlib import Path
from mori.umbp import engine
from sglang.srt.mem_cache.kv_memory_provider import CapacityShape,CompletedReceipt,PageGeometry,PageRegion
from sglang.srt.mem_cache.storage.umbp.engine_provider import UMBPEngineProvider
p=UMBPEngineProvider(device='cpu',options=json.loads(sys.argv[1]))
backing=ctypes.create_string_buffer(160)
p.register_pool(PageGeometry(4,4,32,'cpu-layout-v1',(
 PageRegion('kv',ctypes.addressof(backing),160,32,1,0,0,'kv'),)),())
reservation=p.reserve(CapacityShape('peer-reader',1,1,0,0,4))
targets=reservation.take('load',1)
reservation.refund_unused()
plan=p.client.lookup([p.storage_key(sys.argv[2])],p.namespace_seed)
assert plan.valid_prefix==1
prepared=p.client.prepare_load(plan,'peer-reader',[p._native[h] for h in targets],1)
assert prepared.success,prepared.error
Path(sys.argv[3]).touch()
assert sys.stdin.readline().strip()=='cancel'
assert p.client.cancel_load(prepared)==engine.CancelLoadResult.NeverSubmitted
p.release_pages(targets,ready=CompletedReceipt())
p.close()
"""
        ready_file = Path(self.tmp.name) / "peer-prepared"
        peer_log = self.enterContext(open(Path(self.tmp.name) / "peer.log", "w+"))
        peer = subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                json.dumps(
                    {
                        "node_address": self.address,
                        "namespace": json.loads(self.p.namespace),
                        "rank_base": 1,
                    }
                ),
                source.key,
                str(ready_file),
            ],
            stdin=subprocess.PIPE,
            stdout=peer_log,
            stderr=subprocess.STDOUT,
            text=True,
        )

        def stop_peer():
            if peer.poll() is None:
                peer.kill()
                peer.wait(5)
            peer.stdin.close()

        self.addCleanup(stop_peer)
        deadline = time.monotonic() + 20
        while not ready_file.exists() and peer.poll() is None:
            if time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        peer_log.seek(0)
        self.assertTrue(ready_file.exists(), peer_log.read())
        ready = ManualReceipt()
        self.publish(handles[1:], ready)
        ticket = max(self.p._publications)
        # No local IO/reservation blocks the owner: BeginReset must visit the
        # real node, where the peer's grant makes it Busy and unseals it.
        self.assertFalse(self.p._io)
        self.assertEqual(self.p.accounting().reserved, 0)
        lease = self.p.quiesce_reset(deadline_ns=time.monotonic_ns() + 1_000_000_000)
        self.assertIsInstance(lease, Busy)
        ready.done = True
        self.wait_for(lambda: self.p.accounting().published == 2)
        self.assertNotIn(ticket, self.p._publications)
        self.assertFalse(self.p._publication_results)
        peer.communicate("cancel\n", timeout=10)
        peer_log.seek(0)
        self.assertEqual(peer.returncode, 0, peer_log.read())
        self.a.free(indices)
        self.wait_for(lambda: self.p.accounting().cached == 4)

    def test_ready_provider_reset_uses_complete_native_approval_set(self):
        indices, handles = self.take(2)
        self.publish(handles)
        self.wait_for(lambda: self.p.accounting().published == 2)
        old_epoch = self.p.epoch
        lease = self.p.quiesce_reset(deadline_ns=time.monotonic_ns() + 1_000_000_000)
        self.assertNotIsInstance(lease, Busy)
        self.assertEqual(set(lease.approved_pages), {1, 2, 3, 4})
        self.assertEqual({h.page_id for h in lease.native.approved}, {1, 2, 3, 4})
        self.a.clear(reset_lease=lease)
        self.p.commit_reset(lease)
        self.a.finish_reset()
        self.assertEqual(self.p.epoch.reset_seq, old_epoch.reset_seq + 1)
        self.assertEqual(self.p.accounting().cached, 4)
        again, new = self.take(4)
        self.assertTrue(all(h.epoch == self.p.epoch for h in new))
        self.publish(new)
        self.wait_for(lambda: self.p.accounting().published == 4)
        self.a.free(again)
        self.wait_for(lambda: self.p.accounting().cached == 4)

    def test_node_kill_stops_local_take_without_reconnect(self):
        ticket = self.p.reserve(CapacityShape("inflight", 1, 0, 1, 0, 4))
        self.server.kill()
        self.server.wait(5)
        deadline = time.monotonic() + 2
        while self.p.client.healthy and time.monotonic() < deadline:
            time.sleep(0.005)
        with self.assertRaises(ProviderFailStop):
            ticket.take("extend", 1)
        with self.assertRaises(ProviderFailStop):
            self.p.reserve(CapacityShape("after-disconnect", 1, 0, 1, 0, 4))

    def test_g6_real_failed_lookup_is_cached(self):
        linker = self.p.create_linker(
            None,
            SimpleNamespace(token_to_kv_pool_allocator=self.a),
            components={ComponentType.FULL},
            forward_stream=None,
        )
        keys = ["not-published"]
        raw = self.p.client.lookup([self.p.storage_key(keys[0])], self.p.namespace_seed)
        self.assertFalse(raw.success)
        for _ in range(3):
            self.assertEqual(
                linker.lookup("cold", [PoolTransfer(PoolName.KV, keys=keys)]), []
            )
        self.assertEqual(linker.stats["lookups"], 1)
        self.assertIsNone(linker._plans["cold"].native)

    def test_g6_native_publication_receipts_are_removed_by_ack(self):
        for _ in range(16):
            indices, handles = self.take()
            self.publish(handles)
            ticket = self.p._publication_by_handle[handles[0]]
            self.wait_for(
                lambda: (
                    self.p.accounting().published == 1
                    and not self.p._publication_results
                )
            )
            self.assertIsNone(self.p.client.publish_status(ticket))
            self.a.free(indices)
            self.wait_for(lambda: self.p.accounting().cached == 4)
        self.assertFalse(self.p._publications)

    def test_g7_cpu_geometry_free_group_preserves_index_readiness_guard(self):
        from unittest.mock import patch

        indices, _ = self.take(2)
        ready = ManualReceipt()
        self.a.free_group_begin()
        with patch(
            "sglang.srt.mem_cache.allocator.provider.indices_ready", return_value=ready
        ):
            self.a.free_segment(indices[:4], start_pos=0)
            self.a.free_segment(indices[4:], start_pos=4)
        self.a.free_group_end()
        self.assertEqual(self.p.accounting().pending_free, 2)
        self.assertEqual(self.p.accounting().cached, 2)
        ready.done = True
        self.wait_for(lambda: self.p.accounting().cached == 4)

    def test_g6_native_event_overflow_is_detected_before_reuse(self):
        _, handles = self.take()
        record = self.engine.ResidentRecord()
        record.key = self.p.storage_key("overflow")
        record.parent_key = ""
        record.handle = self.p._native[handles[0]]
        for _ in range(4097):
            ticket = self.p.client.queue_publish([record], False)
            self.assertNotEqual(ticket, 0)
            self.assertTrue(self.p.client.cancel_publish(ticket))
        with self.assertRaisesRegex(ProviderFailStop, "event.*overflow"):
            self.p.progress()
        self.assertEqual(self.p._owned, set(handles))
        with self.assertRaises(ProviderFailStop):
            self.p.reserve(CapacityShape("after-overflow", 1, 0, 1, 0, 4))

    def test_gpu_vmm_bytes_and_exports(self):
        if not torch.cuda.is_available():
            self.skipTest(
                "requires GPU: actual VMM allocation/export and torch MemPool"
            )
        pool = self.engine.kv_mem_pool("cuda:0")
        self.assertIs(pool, self.engine.kv_mem_pool("cuda:0"))
        with torch.cuda.use_mem_pool(pool):
            data = torch.empty(4096, dtype=torch.uint8, device="cuda:0")
        data.fill_(0x5A)
        torch.cuda.synchronize()
        allocation = self.engine.lookup_allocation(data.data_ptr(), data.numel())
        self.assertIsNotNone(allocation)
        self.assertEqual(allocation["base"], data.data_ptr())
        exported = self.engine.export_allocation(data.data_ptr())
        self.assertIsNotNone(exported)
        try:
            self.assertTrue(exported["fd"] >= 0 or exported["fabric"])
            self.assertTrue(torch.all(data.cpu() == 0x5A))
        finally:
            if exported["fd"] >= 0:
                os.close(exported["fd"])


class TestUMBPMultiLayerRead(_EngineFixture):
    row_bytes = (8, 8, 16)
    model_layers = 4  # the last logical layer has no FULL bytes
    layout_id = "cpu-multi-layer-v1"

    def test_g4_partial_dram_readback_preserves_pages_across_all_layers(self):
        source_indices, sources = self.take(3)
        expected = {}
        for i, handle in enumerate(sources):
            for region in self.p.geometry.regions:
                pattern = bytes(
                    (i * 71 + region.layer_id * 19 + j) % 256
                    for j in range(region.row_bytes)
                )
                expected[i, region.layer_id] = pattern
                ctypes.memmove(
                    region.base + handle.page_id * region.row_bytes,
                    pattern,
                    len(pattern),
                )
        records = self.publish(sources)
        self.wait_for(lambda: self.p.accounting().published == 3)
        self.a.free(source_indices)
        self.wait_for(lambda: self.p.accounting().cached == 4)
        ctypes.memset(ctypes.addressof(self.backing), 0, len(self.backing))
        linker = self.p.create_linker(
            None,
            SimpleNamespace(token_to_kv_pool_allocator=self.a),
            components={ComponentType.FULL},
            forward_stream=None,
        )
        keys = [record.key for record in records] + ["missing"]
        for _ in range(2):
            self.assertEqual(
                linker.lookup("partial-dram", [PoolTransfer(PoolName.KV, keys=keys)]),
                [1, 2, 3],
            )
        self.assertEqual(linker.stats["lookups"], 1)
        reservation = self.p.reserve(CapacityShape("partial-dram", 1, 2, 0, 0, 8))
        with self.a.reservation_scope(reservation, "load"):
            targets = self.a.alloc(8)
        ids = tuple(reversed(targets._kv_provider_page_ids))
        targets = targets.reshape(2, 4).flip(0).reshape(-1)
        targets._kv_provider_page_ids = ids
        self.assertTrue(
            linker.load(
                "partial-dram",
                [PoolTransfer(PoolName.KV, device_indices=targets, keys=keys[:2])],
            )
        )
        job = linker._pending["partial-dram"]
        self.assertEqual([h.page_id for h in job.prepared.targets], list(ids))
        self.assertEqual(job.layer_ids, (0, 1, 2))
        self.assertEqual(linker.num_layers, 4)
        reservation.refund_unused()
        index = linker.start_layer_wise_loading()
        linker.layer_done_counter.set_consumer(index)
        for layer in range(4):
            linker.layer_done_counter.wait_until(layer)
        self.assertTrue(linker.finish_layer_wise_loading(index))
        for i, target in enumerate(self.p.freeze_allocations(targets).handles()):
            for region in self.p.geometry.regions:
                self.assertEqual(
                    ctypes.string_at(
                        region.base + target.page_id * region.row_bytes,
                        region.row_bytes,
                    ),
                    expected[i, region.layer_id],
                )
        self.assertEqual(linker.stats["completed_LocalDram_pages"], 2)
        self.a.free(targets)
        self.assertEqual(self.p.accounting().cached, 4)


if __name__ == "__main__":
    unittest.main()
