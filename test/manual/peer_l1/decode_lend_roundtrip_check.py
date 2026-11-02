"""GPU byte checks for DSv4 decode lending, mixed L2 reads and DRAM demotion.

SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton SGLANG_DSV4_UNIFIED_KV_FP8=1 \
    MORI_RDMA_DEVICES=bnxt_re0 python test/manual/peer_l1/decode_lend_roundtrip_check.py

Uses the production 61-layer/182-region unified fp8 layout on two GPUs. No model
weights are needed. Control plane, RDMA and UMBP are real; the scheduler and PD
registration are represented by their already-registered manager descriptors.
--indexer HOST:PORT enables real gRPC discovery/reporting and key-based ACQUIRE.
"""

import argparse
import ctypes
import hashlib
import json
import threading
import time
from collections import defaultdict
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from f3_l2_roundtrip_check import fill, ranges, read
from mori.io import BackendType, IOEngine, IOEngineConfig, MemoryLocationType

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.checksum import KvChecksumComputer
from sglang.srt.disaggregation.mori.conn import MoriKVManager, _SubmissionLocal
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    get_kv_transfer_buf_infos,
)
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4TokenToKVPool,
    select_dsv4_kv_layout,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import UMBPDirectLinker
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.peer_l1.indexer import (
    IndexerClient,
    IndexerConfig,
    PlacementPublisher,
    Placements,
    block_hash,
)
from sglang.srt.mem_cache.unified_cache.peer_l1.lend.allocator import (
    LendablePagedAllocator,
)
from sglang.srt.mem_cache.unified_cache.peer_l1.lend.pool import LendPool
from sglang.srt.mem_cache.unified_cache.peer_l1.lend.protocol import LendService
from sglang.srt.mem_cache.unified_cache.peer_l1.lend.tiered import TieredL2
from sglang.srt.mem_cache.unified_cache.peer_l1.lend.transport import LendTransport
from sglang.srt.runtime_context import get_context

PAGE, PAGES = 256, 128
RATIOS = [128] + [128, 4] * 30


def make_pool(device):
    layout, compressed = select_dsv4_kv_layout()
    full = PAGES * PAGE
    pool = DeepSeekV4TokenToKVPool(
        max_num_reqs=4,
        swa_size=full,
        c4_size=full // 4,
        c128_size=full // 128,
        c4_state_pool_size=0,
        c128_state_pool_size=0,
        page_size=PAGE,
        swa_page_size=PAGE,
        dtype=torch.float8_e4m3fn,
        c4_state_dtype=torch.float32,
        c128_state_dtype=torch.float32,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        indexer_head_dim=128,
        layer_num=len(RATIOS),
        device=f"cuda:{device}",
        enable_memory_saver=False,
        compression_ratios=RATIOS,
        start_layer=0,
        end_layer=len(RATIOS),
        unified_fp8=True,
        full_size=full,
        kv_layout=layout,
        compressed_kv_layout=compressed,
    )
    pool.register_mapping(
        torch.arange(full, device=f"cuda:{device}", dtype=torch.int64)
    )
    return pool


def register(engine, pool, device):
    pointers, lengths, items = get_kv_transfer_buf_infos(pool)
    descs = [
        engine.register_memory(ptr, length, device, MemoryLocationType.GPU)
        for ptr, length in zip(pointers, lengths)
    ]
    return SimpleNamespace(
        kv_data_ptrs=pointers, kv_data_lens=lengths, kv_item_lens=items, gpu_id=device
    ), descs


def transfer(keys, pages):
    return PoolTransfer(
        name=PoolName.KV,
        keys=keys,
        first_parent=None,
        indexer_eligible=True,
        device_indices=torch.tensor([p * PAGE + i for p in pages for i in range(PAGE)]),
    )


def wait_write(l2):
    deadline = time.monotonic() + 60
    while not l2.num_completed_offloads():
        assert time.monotonic() < deadline, "write did not complete"
        time.sleep(0.01)
    assert l2.pop_completed_offload(), "write failed"


def restore(l2, keys, pages):
    if l2.indexer_stage:
        l2.l1.placements.get("restore", keys)
    assert l2.lookup("restore", [PoolTransfer(name=PoolName.KV, keys=keys)])[-1] == len(
        keys
    )
    assert l2.load("restore", [transfer(keys, pages)])
    index = l2.start_layer_wise_loading()
    l2.layer_done_counter.set_consumer(index)
    l2.layer_done_counter.wait_until(0)
    assert l2.finish_layer_wise_loading(index)
    torch.cuda.synchronize()


def check_failed_load_source_hold(manager, info, target_args, lower, rng):
    """Inject a load failure while an early PD send waits in the real worker.

    Request bookkeeping stands in for a model scheduler; PD queue/worker/send,
    failed-load requeue and the production GPU checksum implementation are real.
    """
    room, src, dst = 707, [40, 41, 42, 43], [100, 101, 102, 103]
    fill(ranges(lower, src), rng)
    spans = ranges(lower, src[:2])
    computer = KvChecksumComputer(
        "cuda:0", manager.kv_args.kv_data_ptrs, manager.kv_args.kv_item_lens
    )
    expected = computer.compute(torch.tensor(src, device="cuda:0"))
    manager.disaggregation_mode = DisaggregationMode.PREFILL
    manager.request_status = {room: KVPoll.Transferring}
    manager.transfer_lock = threading.Lock()
    manager._abort_ack_lock = threading.Lock()
    manager._staging_outstanding = defaultdict(int)
    manager._deferred_ack_targets = {}
    manager._submission_local = _SubmissionLocal()
    manager._num_shards = 1
    manager._transfer_queues = [Queue()]
    manager.is_mla_backend = True
    manager.is_hybrid_mla_backend = False
    manager.state_mem_descs = []
    info.requires_dcp_relayout = False
    info.dst_kv_layer_ids = []
    manager.transfer_infos = {
        room: {
            0: SimpleNamespace(
                engine_key=info.engine_key,
                dst_kv_indices=np.array(dst, np.int32),
                is_dummy=False,
                decode_prefix_len=0,
            )
        }
    }
    entered, release = threading.Event(), threading.Event()

    def wait_source():
        entered.set()
        assert release.wait(30)

    manager.add_transfer_request(
        room,
        np.array(src[:2], np.int32),
        slice(0, 2),
        False,
        wait_event=SimpleNamespace(synchronize=wait_source),
    )
    assert not manager.transfer_quiescent(room), "queued source is unprotected"
    worker = threading.Thread(
        target=manager._transfer_worker, args=(manager._transfer_queues[0],)
    )
    worker.start()
    try:
        assert entered.wait(30)
        scheduler = SimpleNamespace(
            disagg_prefill_bootstrap_queue=SimpleNamespace(kv_manager=manager),
            _release_aborted_request=Mock(),
            tree_cache=None,
            waiting_queue=[],
            processed_tokens_counter=0,
        )
        req = SimpleNamespace(
            rid="failed-load-during-early-send",
            bootstrap_room=room,
            reset_for_retract=Mock(),
            advance_cache_request_handle=Mock(),
            time_stats=SimpleNamespace(
                reset_prefill_retry_time=Mock(), set_wait_queue_entry_time=Mock()
            ),
        )
        # Releasing the request immediately recycles its old source pages.
        with patch(
            "sglang.srt.disaggregation.prefill.release_kv_cache",
            side_effect=lambda *_a, **_k: fill(spans, rng),
        ) as free:
            SchedulerDisaggregationPrefillMixin.requeue_after_failed_load(
                scheduler, req
            )
            free.assert_not_called()
            assert scheduler.waiting_queue == []
            release.set()
            manager._transfer_queues[0].put(None)
            worker.join(30)
            assert not worker.is_alive() and manager.transfer_quiescent(room)
            SchedulerDisaggregationPrefillMixin.requeue_after_failed_load(
                scheduler, req
            )
            free.assert_called_once()
            assert scheduler.waiting_queue == [req]
        # The retry retains start_send_idx: only its recomputed tail is sent.
        manager.add_transfer_request(
            room, np.array(src[2:], np.int32), slice(2, 4), False
        )
        chunk = manager._transfer_queues[0].get_nowait()
        assert manager._process_transfer_chunk(chunk)
        manager._mark_transfer_quiescent(chunk)
        torch.cuda.set_device(1)
        actual = KvChecksumComputer(
            "cuda:1", target_args.kv_data_ptrs, target_args.kv_item_lens
        ).compute(torch.tensor(dst, device="cuda:1"))
        assert actual == expected, (
            f"early PD source corrupted: {expected:#x} != {actual:#x}"
        )
        return f"{actual:#x}"
    finally:
        release.set()
        if worker.is_alive():
            manager._transfer_queues[0].put(None)
            worker.join(30)
        torch.cuda.set_device(0)


def check_gpu_reclaim():
    for method in ("alloc", "alloc_extend", "alloc_decode"):
        pool = LendPool(PAGES)
        # One lent page beyond what lend I/O may still pin or write.
        pool.add_free(range(7, 8 + pool.limit))
        original = PagedTokenToKVPoolAllocator(
            PAGES * PAGE, PAGE, torch.int64, "cuda:0", None, True
        )
        original.free_pages = torch.empty(0, dtype=torch.int64, device="cuda:0")
        allocator = LendablePagedAllocator.install(
            original, pool, forward_stream=torch.cuda.current_stream()
        )
        assert allocator.available_size() == PAGE
        if method == "alloc":
            result = allocator.alloc(PAGE)
        elif method == "alloc_extend":
            result = allocator.alloc_extend(
                torch.tensor([0], device="cuda:0"),
                torch.tensor([0]),
                torch.tensor([PAGE], device="cuda:0"),
                torch.tensor([PAGE]),
                torch.tensor([-1], device="cuda:0"),
                PAGE,
            )
        else:
            result = allocator.alloc_decode(
                torch.tensor([1], device="cuda:0"),
                torch.tensor([1]),
                torch.tensor([-1], device="cuda:0"),
            )
        assert result is not None and result[0].item() == 7 * PAGE
        assert len(pool.pages) == pool.limit


def check_overlap_lending(prefill, decode):
    """Hold a redundant forward before its KV write; lend only after it drains.

    Two stream priorities keep the scheduler copy independent of the blocked
    forward on ROCm. The final guest write uses the real MoRI RDMA engine.
    """
    hip = ctypes.CDLL("libamdhip64.so")
    ptr, u64, uint = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint
    hip.hipExtMallocWithFlags.argtypes = [ctypes.POINTER(ptr), ctypes.c_size_t, uint]
    hip.hipMemset.argtypes = [ptr, ctypes.c_int, ctypes.c_size_t]
    hip.hipStreamWaitValue64.argtypes = [ptr, ptr, u64, uint, u64]
    hip.hipStreamWriteValue64.argtypes = [ptr, ptr, u64, uint]
    hip.hipFree.argtypes = [ptr]

    def checked(result):
        assert result == 0, f"HIP signal operation failed: {result}"

    torch.cuda.set_device(0)
    guest = torch.full((1, 16), 9, dtype=torch.int32, device="cuda:0")
    guest_desc = prefill.register_memory(
        guest.data_ptr(), 64, 0, MemoryLocationType.GPU
    )
    torch.cuda.synchronize()
    torch.cuda.set_device(1)
    kv = torch.zeros((1025, 16), dtype=torch.int32, device="cuda:1")
    kv_desc = decode.register_memory(
        kv.data_ptr(), kv.numel() * 4, 1, MemoryLocationType.GPU
    )
    forward = torch.cuda.Stream(priority=0)
    schedule = torch.cuda.Stream(priority=-1)
    pool = LendPool(1024)
    with torch.cuda.stream(schedule):
        original = PagedTokenToKVPoolAllocator(
            1024 * PAGE, PAGE, torch.int64, "cuda:1", None, True
        )
        allocator = LendablePagedAllocator.install(
            original, pool, forward_stream=forward
        )
    signal = ptr()
    checked(
        hip.hipExtMallocWithFlags(ctypes.byref(signal), 8, 0x2)
    )  # hipMallocSignalMemory
    checked(hip.hipMemset(signal, 0, 8))
    torch.cuda.synchronize()
    try:
        checked(
            hip.hipStreamWaitValue64(forward.cuda_stream, signal, 1, 0, (1 << 64) - 1)
        )
        with torch.cuda.stream(forward):
            kv[1:257].fill_(7)  # the redundant forward still owns these freed pages
        with torch.cuda.stream(schedule):
            allocator.rebalance()
            assert allocator._pending_lend is not None
            deadline = time.monotonic() + 10
            while not allocator._pending_lend[2].query():
                assert time.monotonic() < deadline, "scheduler id copy did not complete"
                time.sleep(0.001)
            allocator._publish_ready_lend()
            assert not pool.allocate("guest", 1), "lent KV before the forward finished"
            checked(hip.hipStreamWriteValue64(schedule.cuda_stream, signal, 1, 0))
            forward.synchronize()
            allocator._publish_ready_lend()
            records = pool.allocate("guest", 1)
            assert records

        def wait(statuses):
            for status in statuses:
                status.Wait()

        manager = SimpleNamespace(
            engine=prefill,
            kv_mem_descs=[guest_desc],
            kv_args=SimpleNamespace(kv_item_lens=[64]),
            _wait_transfer_completion=wait,
        )
        assert LendTransport(manager).transfer(
            SimpleNamespace(dst_kv_mem_descs=[kv_desc]),
            [0],
            [records[0][0]],
            write=True,
        )
        torch.cuda.synchronize()
        assert torch.all(kv[records[0][0]] == 9).item(), (
            "forward overwrote the guest KV"
        )
        pool.abort("guest", records)
    finally:
        checked(hip.hipStreamWriteValue64(schedule.cuda_stream, signal, 1, 0))
        torch.cuda.synchronize()
        checked(hip.hipFree(signal))
        prefill.deregister_memory(guest_desc)
        decode.deregister_memory(kv_desc)
        torch.cuda.set_device(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indexer", help="Upstream Indexer gRPC endpoint")
    args = parser.parse_args()
    config = (
        IndexerConfig(stage=1, endpoint=args.indexer, deployment="lend-roundtrip")
        if args.indexer
        else IndexerConfig()
    )
    get_context().override_server_args(
        dsv4_attn_backend="flashmla", enable_deepseek_v4_fp4_indexer=True
    ).install()
    torch.cuda.set_device(0)
    source = make_pool(0)
    torch.cuda.set_device(1)
    target = make_pool(1)
    engines = []
    for name in ("prefill", "decode"):
        engine = IOEngine(
            f"lend-check-{name}", IOEngineConfig(host="127.0.0.1", port=0)
        )
        engine.create_backend(BackendType.RDMA)
        engines.append(engine)
    prefill, decode = engines
    prefill.register_remote_engine(decode.get_engine_desc())
    decode.register_remote_engine(prefill.get_engine_desc())
    source_args, source_descs = register(prefill, source, 0)
    target_args, target_descs = register(decode, target, 1)
    assert source_args.kv_item_lens == target_args.kv_item_lens
    assert len(source_descs) == 182
    assert sum(source_args.kv_item_lens) == 1399040
    torch.cuda.set_device(0)
    params = SimpleNamespace(
        page_size=PAGE,
        token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: source),
        tp_cache_group=None,
        mtp_draft_device_pools=None,
        req_to_token_pool=None,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
    )
    lower = UMBPDirectLinker(
        None,
        params,
        components={ComponentType.FULL, ComponentType.SWA},
        extra_config={"dram_capacity_bytes": 1 << 30, "ssd_enabled": False},
    )
    l1 = SimpleNamespace(
        page_size=PAGE,
        pool_group=lower.pool_group,
        rank=0,
        cpu_group=None,
        directory=SimpleNamespace(num_rows=PAGES),
        indexer_config=config,
        indexer=IndexerClient(config) if args.indexer else None,
        placement=(None, None),
    )
    l1.placements = Placements(l1.indexer) if args.indexer else None
    # 16 writes admitted; first eight pages in HBM, next eight in DRAM.
    pool = LendPool(PAGES, limit=0.25)
    pool.add_free(range(1, 9))
    worker_id = "lend-roundtrip/lend/dp0"
    if args.indexer:
        pool.publisher = PlacementPublisher(config=config, worker_id=worker_id)
    service = LendService(
        pool, "127.0.0.1", layout=source_args.kv_item_lens, worker_id=worker_id
    )
    info = SimpleNamespace(
        endpoint="127.0.0.1",
        lend_port=service.port,
        decode_dp_rank=0,
        engine_key=decode.get_engine_desc().key,
        dst_kv_mem_descs=target_descs,
    )
    manager = MoriKVManager.__new__(MoriKVManager)
    manager.engine, manager.kv_args, manager.kv_mem_descs = (
        prefill,
        source_args,
        source_descs,
    )
    manager.decode_kv_args_table = {info.engine_key: info}
    manager.enable_deferred_decode_kv_release = True
    manager.attn_tp_size = manager.dcp_size = 1
    manager._wait_poll_ms, manager._transfer_timeout_ms = 10, 30000
    l2 = TieredL2(l1, lower)
    rng = np.random.default_rng(28)
    keys = [hashlib.sha256(f"lend-{i}".encode()).hexdigest() for i in range(16)]
    written, restored = list(range(16, 32)), list(range(64, 80))
    original_spans = ranges(lower, written)
    fill(original_spans, rng)
    original = read(original_spans)
    try:
        l2.bind_pd(manager)
        deadline = time.monotonic() + 30
        while not l2._peers:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert l2.offload([transfer(keys, written)])
        wait_write(l2)
        assert l2.stats["write_pages"] == 8
        if args.indexer:
            deadline = time.monotonic() + 10
            while (
                len(
                    (l1.indexer.lookup([block_hash(k) for k in keys]) or {}).get(
                        worker_id, set()
                    )
                )
                != 8
            ):
                assert time.monotonic() < deadline, "lending publication missing"
                time.sleep(0.01)
        fill(original_spans, rng)  # original source slots have now been reused
        restore(l2, keys, restored)
        got = read(ranges(lower, restored))
        assert len(got) == len(original)
        assert all(np.array_equal(a, b) for a, b in zip(original, got)), (
            "mixed L2 bytes differ"
        )
        with pool.lock:
            pool.request_demotions(8)
        deadline = time.monotonic() + 30
        while l2.stats["demote_pages"] < 8:
            assert time.monotonic() < deadline, "demotion did not complete"
            time.sleep(0.01)
        if args.indexer:
            deadline = time.monotonic() + 10
            while (l1.indexer.lookup([block_hash(k) for k in keys]) or {}).get(
                worker_id
            ):
                assert time.monotonic() < deadline, "demoted placement still published"
                time.sleep(0.01)
        else:
            assert all(v[0] < 0 for v in l2.directory.query(keys))
        fill(ranges(lower, restored), rng)
        restore(l2, keys, restored)
        assert all(
            np.array_equal(a, b)
            for a, b in zip(original, read(ranges(lower, restored)))
        ), "demoted bytes differ"
        staging_ptr = l2.staging.tensor.data_ptr()
        l2.unbind_pd()
        l2.bind_pd(manager)
        assert l2.staging.tensor.data_ptr() == staging_ptr, "role switch leaked staging"
        restore(l2, keys, restored)
        l2.unbind_pd()
        checksum = check_failed_load_source_hold(manager, info, target_args, lower, rng)
        check_gpu_reclaim()
        check_overlap_lending(prefill, decode)
        print(
            json.dumps(
                {
                    "result": "PASS",
                    "regions": len(source_descs),
                    "page_bytes": sum(source_args.kv_item_lens),
                    "pages": len(keys),
                    "stats": dict(l2.stats),
                    "early_pd_checksum": checksum,
                    "staging_rebind": True,
                    "gpu_reclaim": True,
                    "overlap_lending": True,
                    "indexer_stage": config.stage,
                }
            )
        )
    finally:
        l2.close()
        service.close()
        if l1.indexer:
            l1.indexer.close()
        for engine, descs in ((prefill, source_descs), (decode, target_descs)):
            for desc in descs:
                engine.deregister_memory(desc)


if __name__ == "__main__":
    main()
