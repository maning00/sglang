"""Last-replica L2 writes keep every DSv4 object byte-exact through slot reuse.

    SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton SGLANG_DSV4_UNIFIED_KV_FP8=1 \\
        python test/manual/peer_l1/f3_l2_roundtrip_check.py

A DeepSeek-V4-Pro-shaped unified fp8 pool (production row widths, layout and
compression pattern, six layers) and a UMBP direct linker on an in-process DRAM
tier. Pages are written through the composite linker's last-replica path while
other pages' slots are rewritten; once the write lands the written slots are
rewritten too, as the allocator reusing them would. The pages are then restored
into fresh slots the way the SWA-ring wrapper restores them (KV only), and every
object range the restore wrote is compared with the original bytes.

With --control the written slots are rewritten right after the write is
submitted, as if the rows were not held; the check must then report mismatches,
or it could not catch a broken hold.
"""

import argparse
import ctypes
import json
import os
import time
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4TokenToKVPool,
    select_dsv4_kv_layout,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import UMBPDirectLinker
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.peer_l1.composite import PeerL1PlusL2Linker
from sglang.srt.runtime_context import get_context

PAGE = 256
PAGES = 64
# DeepSeek-V4-Pro's compress_ratios start 128, 128, 4, 128, 4, ...
RATIOS = [128, 128, 4, 128, 4, 128]
WRITTEN = list(range(8, 24))
RESTORED = list(range(40, 56))
OTHERS = [page for page in range(PAGES) if page not in WRITTEN + RESTORED]

hip = ctypes.CDLL("libamdhip64.so")
hip.hipMemcpy.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
]
H2D, D2H = 1, 2


def ranges(linker, pages):
    """(pool, object index, ptr, size) for every byte range a KV transfer of
    ``pages`` moves, from the linker's own plan."""
    indices = torch.tensor([p * PAGE + i for p in pages for i in range(PAGE)])
    expanded = linker.pool_group.resolve_transfers(
        [
            PoolTransfer(
                name=PoolName.KV, keys=[f"k{p}" for p in pages], device_indices=indices
            )
        ],
        allow_partial=True,
    )
    out = []
    for plan in linker._build_load_plans([expanded]):
        ptrs, sizes, _ = linker._all_layer_ranges(plan)
        for obj, (obj_ptrs, obj_sizes) in enumerate(zip(ptrs, sizes)):
            out.extend((plan.name, obj, p, s) for p, s in zip(obj_ptrs, obj_sizes))
    return out


def fill(spans, rng):
    for _, _, ptr, size in spans:
        data = rng.integers(0, 256, size, dtype=np.uint8)
        assert hip.hipMemcpy(ptr, data.ctypes.data, size, H2D) == 0


def read(spans):
    out = []
    for _, _, ptr, size in spans:
        data = np.empty(size, dtype=np.uint8)
        assert hip.hipMemcpy(data.ctypes.data, ptr, size, D2H) == 0
        out.append(data)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", action="store_true")
    control = parser.parse_args().control
    torch.cuda.set_device(0)
    get_context().override_server_args(dsv4_attn_backend="flashmla").install()
    kv_layout, compressed_kv_layout = select_dsv4_kv_layout()
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
        device="cuda",
        enable_memory_saver=False,
        compression_ratios=RATIOS,
        start_layer=0,
        end_layer=len(RATIOS),
        unified_fp8=True,
        full_size=full,
        kv_layout=kv_layout,
        compressed_kv_layout=compressed_kv_layout,
    )
    pool.register_mapping(torch.arange(full, dtype=torch.int64, device="cuda"))
    params = SimpleNamespace(
        page_size=PAGE,
        token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: pool),
        tp_cache_group=None,
        mtp_draft_device_pools=None,
        req_to_token_pool=None,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
    )
    l2 = UMBPDirectLinker(
        None,
        params,
        components={ComponentType.FULL, ComponentType.SWA},
        extra_config={"dram_capacity_bytes": 4 << 30, "ssd_enabled": False},
    )
    released = []
    l1 = SimpleNamespace(
        layer_done_counter=SimpleNamespace(num_layers=l2.layer_done_counter.num_layers),
        owner=SimpleNamespace(release_written=released.append),
        stats=Counter(),
    )
    composite = PeerL1PlusL2Linker(l1, l2, l2_write="last_replica")

    rng = np.random.default_rng(0)
    written, restored, others = (
        ranges(l2, pages) for pages in (WRITTEN, RESTORED, OTHERS)
    )
    fill(written, rng)
    fill(others, rng)
    original = read(written)
    keys = [f"f3check-{page:04d}" for page in WRITTEN]
    assert composite._write_last_replicas(keys, np.array(WRITTEN))
    if control:
        fill(written, rng)
    # Allocator pressure on slots that are not held while the write is in flight.
    rewrites = 0
    deadline = time.monotonic() + 60
    while not released:
        fill(others, rng)
        rewrites += 1
        composite.num_completed_offloads()
        assert time.monotonic() < deadline, "the L2 write never landed"
    assert l1.stats["last_replica_failed_batches"] == 0, "the L2 write failed"
    # The rows are free again: reuse them.
    fill(written, rng)

    indices = torch.tensor([p * PAGE + i for p in RESTORED for i in range(PAGE)])
    lookup = [PoolTransfer(name=PoolName.KV, keys=keys)]
    hit = l2.lookup("check", lookup)
    assert hit and hit[-1] == len(WRITTEN), (
        f"lookup found {hit[-1] if hit else 0} of {len(WRITTEN)} pages"
    )
    assert l2.load(
        "check", [PoolTransfer(name=PoolName.KV, keys=keys, device_indices=indices)]
    )
    index = l2.start_layer_wise_loading()
    assert l2.finish_layer_wise_loading(index), "the L2 load failed"
    torch.cuda.synchronize()

    got = read(restored)
    assert len(got) == len(original)
    bad = Counter()
    for (pool_name, _, _, _), want, have in zip(written, original, got):
        if not np.array_equal(want, have):
            bad[str(pool_name)] += 1
    pools = Counter(str(span[0]) for span in written)
    print(
        json.dumps(
            {
                "kv_layout": str(kv_layout),
                "compressed_kv_layout": str(compressed_kv_layout),
                "pools": dict(pools),
                "bytes": int(sum(span[3] for span in written)),
                "rewrites_during_write": rewrites,
                "mismatched_ranges": dict(bad),
                "control": control,
            }
        )
    )
    if control:
        assert bad, "control: slots rewritten mid-write still restored intact"
        print("f3 l2 roundtrip control ok: the check catches an unheld write")
        return
    assert not bad, f"restored bytes differ: {dict(bad)}"
    assert rewrites > 0, "the write landed before any concurrent rewrite"
    l2.close()
    print("f3 l2 roundtrip check ok")


if __name__ == "__main__":
    main()
