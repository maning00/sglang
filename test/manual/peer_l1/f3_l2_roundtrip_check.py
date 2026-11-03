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

With --stitch only the pages past HEAD are written, as when the head is still in
a peer's HBM: the composite must restore the head from the peer (a stand-in that
writes the original bytes into the slots it is given) and the rest from L2.
"""

import argparse
import ctypes
import json
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
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    LayerWiseLoadCounter,
)
from sglang.srt.runtime_context import get_context

PAGE = 256
PAGES = 64
# DeepSeek-V4-Pro's compress_ratios start 128, 128, 4, 128, 4, ...
RATIOS = [128, 128, 4, 128, 4, 128]
WRITTEN = list(range(8, 24))
RESTORED = list(range(40, 56))
OTHERS = [page for page in range(PAGES) if page not in WRITTEN + RESTORED]
# --stitch: pages of WRITTEN the peer still holds.
HEAD = 6

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


class Peer:
    """Peer L1 stand-in for the pages in ``held``: its load writes each page's
    original bytes into the slots the composite hands it."""

    def __init__(self, linker, num_layers, held):
        self.linker, self.held = linker, held
        self.layer_done_counter = LayerWiseLoadCounter(num_layers)
        self.owner = None
        self.stats = Counter()
        self.queued = []

    def lookup(self, rid, transfers):
        (transfer,) = transfers
        hits = 0
        while hits < len(transfer.keys) and transfer.keys[hits] in self.held:
            hits += 1
        return list(range(1, hits + 1))

    def reserve(self, rid, num_pages):
        return True

    def release_reservation(self, rid, *, keep_placement=False):
        pass

    def load(self, rid, transfers):
        self.queued.extend(transfers)
        return True

    def cancel_queued_load(self, rid):
        return False

    def start_layer_wise_loading(self):
        if not self.queued:
            return -1
        for transfer in self.queued:
            slots = transfer.device_indices.tolist()
            for i, key in enumerate(transfer.keys):
                dest = ranges(self.linker, [slots[i * PAGE] // PAGE])
                for (_, _, ptr, size), data in zip(dest, self.held[key]):
                    assert size == data.size
                    assert hip.hipMemcpy(ptr, data.ctypes.data, size, H2D) == 0
        self.queued = []
        index = self.layer_done_counter.update_producer()
        for layer in range(self.layer_done_counter.num_layers):
            self.layer_done_counter.complete(index, layer)
        return index

    def finish_layer_wise_loading(self, index):
        return self.layer_done_counter.finish(index)

    def finish_loads(self, index, rids):
        return set() if self.finish_layer_wise_loading(index) else set(rids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--stitch", action="store_true")
    args = parser.parse_args()
    control, head = args.control, HEAD if args.stitch else 0
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
    rng = np.random.default_rng(0)
    written, restored, others = (
        ranges(l2, pages) for pages in (WRITTEN, RESTORED, OTHERS)
    )
    fill(written, rng)
    fill(others, rng)
    original = read(written)
    keys = [f"f3check-{page:04d}" for page in WRITTEN]
    held = {keys[i]: read(ranges(l2, [WRITTEN[i]])) for i in range(head)}
    released = []
    l1 = Peer(l2, l2.layer_done_counter.num_layers, held)
    l1.owner = SimpleNamespace(release_written=released.append)
    composite = PeerL1PlusL2Linker(l1, l2, l2_write="last_replica")
    assert composite._write_last_replicas(keys[head:], np.array(WRITTEN[head:]))
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
    hit = composite.lookup("check", lookup)
    assert hit and hit[-1] == len(WRITTEN), (
        f"lookup found {hit[-1] if hit else 0} of {len(WRITTEN)} pages"
    )
    assert composite.reserve("check", len(WRITTEN))
    assert composite.load(
        "check", [PoolTransfer(name=PoolName.KV, keys=keys, device_indices=indices)]
    )
    index = composite.start_layer_wise_loading()
    assert composite.finish_layer_wise_loading(index), "the load failed"
    torch.cuda.synchronize()
    loaded = (l1.stats["l1_loaded_pages"], l1.stats["l2_loaded_pages"])
    assert loaded == (head, len(WRITTEN) - head), f"loaded (L1, L2) pages {loaded}"

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
                "peer_head_pages": head,
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
