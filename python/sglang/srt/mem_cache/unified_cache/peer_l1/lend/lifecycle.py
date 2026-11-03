"""Scheduler hooks: initialize only after PD has registered its KV regions."""

from __future__ import annotations

import json
import logging
import time

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_memory, get_parallel

from .allocator import LendablePagedAllocator
from .pool import LendPool
from .protocol import LendService

logger = logging.getLogger(__name__)


def install_decode_lend(scheduler):
    from ..indexer import IndexerConfig, PlacementPublisher
    from ..registry import DomainConfig, topology

    raw = get_memory().hicache_storage_backend_extra_config
    config = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
    indexer = IndexerConfig(**config.get("peer_l1", config).get("indexer", {}))

    manager = scheduler.disagg_decode_prealloc_queue.kv_manager
    outer = scheduler.token_to_kv_pool_allocator
    cache = outer.get_kvcache()
    if (
        not manager.enable_deferred_decode_kv_release
        or not hasattr(manager, "kv_mem_descs")
        or manager.attn_tp_size != 1
        or manager.dcp_size != 1
        or not getattr(cache, "_unified_kv", False)
        or not getattr(outer, "swa_req_ring", False)
        or outer.page_size != 256
        or get_parallel().pp_size != 1
        or scheduler.enable_unified_memory
    ):
        raise ValueError(
            "decode lend requires MoRI, deferred KV release, unified KV, "
            "request SWA rings, page_size=256, attention TP=CP=1"
        )
    inner = outer.full_attn_allocator
    pool = LendPool(
        inner.num_pages,
        idle=envs.SGLANG_DISAGGREGATION_DECODE_LEND_IDLE_FRACTION.get(),
        limit=envs.SGLANG_DISAGGREGATION_DECODE_LEND_UNRECLAIMABLE_FRACTION.get(),
        growth=envs.SGLANG_DISAGGREGATION_DECODE_LEND_GROWTH_FRACTION.get(),
        watermark=envs.SGLANG_DISAGGREGATION_DECODE_LEND_FREE_WATERMARK.get(),
        lease_s=envs.SGLANG_DISAGGREGATION_DECODE_LEND_LEASE_S.get(),
    )
    pool.memory_owner = (cache, manager)
    outer.full_attn_allocator = LendablePagedAllocator.install(
        inner, pool, forward_stream=scheduler.forward_stream
    )
    domain = DomainConfig(**config.get("peer_l1", config).get("domain", {}))
    worker_id = f"{indexer.deployment}/{topology(domain).instance_id}/decode_lend/dp{get_parallel().attn_dp_rank}"
    pool.publisher = PlacementPublisher(config=indexer, worker_id=worker_id)
    scheduler.decode_lend_service = LendService(
        pool,
        manager.local_ip,
        layout=list(manager.kv_args.kv_item_lens),
        worker_id=worker_id,
        compatibility_id=domain.kv_compatibility_id,
    )
    manager.lend_port = scheduler.decode_lend_service.port
    manager.lend_dp_rank = get_parallel().attn_dp_rank


def unbind_decode_lend(scheduler, *, timeout_s=30.0):
    service = getattr(scheduler, "decode_lend_service", None)
    if service is None:
        return
    allocator = scheduler.token_to_kv_pool_allocator.full_attn_allocator
    pool = service.pool
    with pool.lock:
        if pool.poisoned:
            raise RuntimeError("Quarantined decode lend pages require process restart")
        pool.draining = True
    # A writer may still own a posted RDMA. Only its COMMIT/ABORT (after
    # draining) permits reuse. Timeout FAILS teardown; it never frees those pages.
    deadline = time.monotonic() + timeout_s
    while True:
        with pool.lock:
            pool.expire()
            writing, pinned = pool.writing, pool.pinned
        if not writing and not pinned:
            break
        if time.monotonic() >= deadline:
            message = (
                f"Decode lend teardown timed out: {writing} writing, {pinned} pinned "
                "pages remain protected; restart required"
            )
            logger.error(message)
            raise TimeoutError(message)
        time.sleep(0.01)
    service.close()
    allocator._ensure_pages(allocator.num_pages)
    if allocator._pending_lend is not None:
        allocator._pending_lend[2].synchronize()
        allocator._pending_lend = None
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

    original = PagedTokenToKVPoolAllocator.__new__(PagedTokenToKVPoolAllocator)
    original.__dict__.update(allocator.__dict__)
    del original.lend_pool
    scheduler.token_to_kv_pool_allocator.full_attn_allocator = original
    scheduler.decode_lend_service = None


def prefill_linker(scheduler):
    wrapper = getattr(scheduler.tree_cache, "linker", None)
    return getattr(wrapper, "cache_linker", None)
