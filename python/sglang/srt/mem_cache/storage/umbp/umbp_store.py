"""UMBPStore — HiCache L3 storage backend using UMBP (local DRAM + SSD).

Follows the same pattern as MooncakeStore:
- Zero-copy v1 interface (batch_get_v1 / batch_set_v1)
- Uses mem_pool_host.get_page_buffer_meta() for pointer/size extraction
- Key suffix generation per TP rank / PP rank
"""

import logging
import os
from typing import Any, List, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache

logger = logging.getLogger(__name__)


def _import_umbp_client():
    """Import UMBPClient from mori.umbp (requires mori built with BUILD_UMBP=ON)."""
    import mori.umbp as umbp_mod

    UMBPClient = umbp_mod.UMBPClient
    UMBPConfig = umbp_mod.UMBPConfig
    UMBPRole = umbp_mod.UMBPRole
    UMBPIoBackend = getattr(umbp_mod, "UMBPIoBackend", None)
    UMBPDurabilityMode = getattr(umbp_mod, "UMBPDurabilityMode", None)
    UMBPDistributedConfig = getattr(umbp_mod, "UMBPDistributedConfig", None)

    return (
        UMBPClient,
        UMBPConfig,
        UMBPRole,
        UMBPIoBackend,
        UMBPDurabilityMode,
        UMBPDistributedConfig,
    )


class UMBPStore(HiCacheStorage):
    """Local DRAM+SSD storage backend for HiCache L3 caching.

    Compatible with the zero-copy v1 interface used by CacheController.
    """

    def __init__(
        self,
        storage_config: HiCacheStorageConfig = None,
        mem_pool_host: HostKVCache = None,
    ):
        (
            UMBPClient,
            UMBPConfig,
            UMBPRole,
            UMBPIoBackend,
            UMBPDurabilityMode,
            UMBPDistributedConfig,
        ) = _import_umbp_client()

        cfg = UMBPConfig()

        # Load settings from extra_config if available
        extra = getattr(storage_config, "extra_config", None) or {}
        if "dram_capacity_bytes" in extra:
            cfg.dram.capacity_bytes = int(extra["dram_capacity_bytes"])
        if "ssd_enabled" in extra:
            cfg.ssd.enabled = bool(extra["ssd_enabled"])
        if "ssd_storage_dir" in extra:
            cfg.ssd.storage_dir = str(extra["ssd_storage_dir"])
        if "ssd_capacity_bytes" in extra:
            cfg.ssd.capacity_bytes = int(extra["ssd_capacity_bytes"])
        if "copy_to_ssd_async" in extra:
            cfg.copy_pipeline.async_enabled = bool(extra["copy_to_ssd_async"])
        if "copy_to_ssd_queue_depth" in extra:
            cfg.copy_pipeline.queue_depth = int(extra["copy_to_ssd_queue_depth"])
        if "ssd_segment_size_bytes" in extra:
            cfg.ssd.segment_size_bytes = int(extra["ssd_segment_size_bytes"])
        if "ssd_batch_max_ops" in extra:
            cfg.copy_pipeline.batch_max_ops = int(extra["ssd_batch_max_ops"])
        if "ssd_queue_depth" in extra:
            cfg.ssd.io.queue_depth = int(extra["ssd_queue_depth"])
        if "ssd_writer_threads" in extra:
            cfg.copy_pipeline.worker_threads = int(extra["ssd_writer_threads"])
        if "ssd_enable_background_gc" in extra:
            cfg.ssd.durability.enable_background_gc = bool(
                extra["ssd_enable_background_gc"]
            )
        if "auto_promote_on_read" in extra:
            cfg.eviction.auto_promote_on_read = bool(extra["auto_promote_on_read"])
        if "eviction_policy" in extra:
            cfg.eviction.policy = str(extra["eviction_policy"])
        if "eviction_candidate_window" in extra:
            cfg.eviction.candidate_window = int(extra["eviction_candidate_window"])
        if "ssd_io_backend" in extra and UMBPIoBackend is not None:
            backend = str(extra["ssd_io_backend"]).lower()
            if backend in ("pthread", "posix"):
                cfg.ssd.io.backend = UMBPIoBackend.PThread
            elif backend in ("io_uring", "uring"):
                cfg.ssd.io.backend = UMBPIoBackend.IoUring
        if "ssd_durability_mode" in extra and UMBPDurabilityMode is not None:
            durability = str(extra["ssd_durability_mode"]).lower()
            if durability in ("strict", "sync"):
                cfg.ssd.durability.mode = UMBPDurabilityMode.Strict
            elif durability in ("relaxed", "async"):
                cfg.ssd.durability.mode = UMBPDurabilityMode.Relaxed

        # Distributed mode: when master_address is provided, construct
        # UMBPDistributedConfig and assign to cfg.distributed.  This enables
        # the PoolClient inside UMBPClient to connect to the Master for
        # cross-node KV cache sharing via RDMA.
        # Default node_id to node_address:<rank> so that the id is globally
        # unique across nodes.  In DP+EP mode the attention tp_rank is 0 for
        # every rank, so we fall back to dp_rank to keep node_ids distinct.
        tp_rank = storage_config.tp_rank if storage_config is not None else 0
        unique_rank = tp_rank
        try:
            from sglang.srt.layers.dp_attention import (
                get_attention_dp_rank,
                is_dp_attention_enabled,
            )

            if is_dp_attention_enabled():
                dp_rank = get_attention_dp_rank()
                unique_rank = dp_rank if tp_rank == 0 and dp_rank > 0 else tp_rank
        except (ImportError, AssertionError):
            pass
        if (
            "master_address" in extra
            and UMBPDistributedConfig is not None
        ):
            dist_cfg = UMBPDistributedConfig()
            dist_cfg.master_address = str(extra["master_address"])
            node_address = str(extra.get("node_address", ""))
            dist_cfg.node_id = str(extra.get("node_id", f"{node_address}:{unique_rank}"))
            dist_cfg.node_address = node_address
            if "auto_heartbeat" in extra:
                dist_cfg.auto_heartbeat = bool(extra["auto_heartbeat"])
            if "io_engine_host" in extra:
                dist_cfg.io_engine_host = str(extra["io_engine_host"])
            if "io_engine_port" in extra:
                port_list = extra["io_engine_port"]
                dist_cfg.io_engine_port = int(port_list[unique_rank])
            if "staging_buffer_size" in extra:
                dist_cfg.staging_buffer_size = int(extra["staging_buffer_size"])
            if "peer_service_port" in extra:
                dist_cfg.peer_service_port = int(extra["peer_service_port"])
            if "cache_remote_fetches" in extra:
                dist_cfg.cache_remote_fetches = bool(extra["cache_remote_fetches"])
            cfg.distributed = dist_cfg
            logger.info(
                "UMBPStore distributed mode: master=%s, node_id=%s, node_addr=%s",
                dist_cfg.master_address,
                dist_cfg.node_id,
                dist_cfg.node_address,
            )

        self.storage_config = storage_config

        # TP/PP rank info for key suffix generation
        if storage_config is not None:
            self.is_mla_backend = storage_config.is_mla_model
            self.local_rank = storage_config.tp_rank
            self.pp_rank = storage_config.pp_rank
            self.pp_size = storage_config.pp_size
        else:
            self.is_mla_backend = False
            self.local_rank = 0
            self.pp_rank = 0
            self.pp_size = 1

        # MLA + TP > 1: shared SSD mode
        self.is_mla_follower = False
        tp_size = getattr(storage_config, "tp_size", 1) if storage_config else 1
        if self.is_mla_backend and tp_size > 1:
            cfg.ssd.enabled = True
            if self.local_rank == 0:
                # Leader: copy every DRAM write to shared SSD.
                if UMBPRole is not None and hasattr(cfg, "role"):
                    cfg.role = UMBPRole.SharedSSDLeader
                cfg.force_ssd_copy_on_write = True
            else:
                # Follower: read-only, SSD filesystem fallback.
                if UMBPRole is not None and hasattr(cfg, "role"):
                    cfg.role = UMBPRole.SharedSSDFollower
                cfg.follower_mode = True
                self.is_mla_follower = True
            logger.info(
                "UMBPStore MLA+TP>1: rank=%d, role=%s, shared_ssd=%s",
                self.local_rank,
                "leader" if self.local_rank == 0 else "follower",
                cfg.ssd.storage_dir,
            )

        try:
            from sglang.srt.layers.dp_attention import (
                get_attention_dp_rank,
                is_dp_attention_enabled,
            )

            if is_dp_attention_enabled():
                dp_rank = get_attention_dp_rank()
                if dp_rank > 0 and cfg.ssd.enabled:
                    cfg.ssd.storage_dir = f"{cfg.ssd.storage_dir}/dp{dp_rank}"
                    logger.info(
                        "UMBPStore DP isolation: dp_rank=%d, ssd_dir=%s",
                        dp_rank,
                        cfg.ssd.storage_dir,
                    )
        except (ImportError, AssertionError):
            pass

        # Pin RDMA NIC per rank so each rank uses a dedicated NIC,
        # avoiding QPN collisions from round-robin selection.
        os.environ["MORI_IO_RDMA_NIC_IDX"] = str(unique_rank)
        logger.info("UMBPStore: pinning RDMA NIC to index %d for unique_rank %d", unique_rank, unique_rank)

        self.client = UMBPClient(cfg)

        self.enable_pp = self.pp_size > 1
        if self.enable_pp:
            self.mha_suffix = f"{self.local_rank}_{self.pp_rank}"
            self.mla_suffix = f"{self.pp_rank}"
        else:
            self.mha_suffix = f"{self.local_rank}"
            self.mla_suffix = ""

        self.split_factor = 0
        if storage_config and storage_config.should_split_heads:
            self.split_factor = storage_config.tp_lcm_size // storage_config.tp_size
            base_rank = self.local_rank * self.split_factor
            target_ranks = [base_rank + i for i in range(self.split_factor)]
            if self.enable_pp:
                self.mha_suffix = [f"{rank}_{self.pp_rank}" for rank in target_ranks]
            else:
                self.mha_suffix = [f"{rank}" for rank in target_ranks]

        logger.info(
            "UMBPStore initialized: dram=%d MB, ssd=%s, mla=%s, rank=%d",
            cfg.dram.capacity_bytes // (1024 * 1024),
            cfg.ssd.enabled,
            self.is_mla_backend,
            self.local_rank,
        )

    # ------------------------------------------------------------------
    # Host memory pool registration
    # ------------------------------------------------------------------
    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        super().register_mem_pool_host(mem_pool_host)
        assert self.mem_pool_host.layout in [
            "page_first",
            "page_first_direct",
            "page_head",
        ], "UMBP store only supports page_first, page_first_direct, or page_head layout"

    # ------------------------------------------------------------------
    # Key suffix generation — mirrors MooncakeStore
    # ------------------------------------------------------------------
    def _get_mha_buffer_meta(self, keys, indices):
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)
        key_list = []
        for key_ in keys:
            key_list.append(f"{key_}_{self.mha_suffix}_k")
            key_list.append(f"{key_}_{self.mha_suffix}_v")
        assert len(key_list) == len(ptr_list)
        return key_list, ptr_list, element_size_list

    def _get_mha_split_heads_buffer_meta(self, keys, indices):
        ptr_list, element_size_list = (
            self.mem_pool_host.get_split_heads_page_buffer_meta(
                indices, self.split_factor
            )
        )
        key_list = []
        for key_ in keys:
            for suffix in self.mha_suffix:
                key_list.append(f"{key_}_{suffix}_k")
                key_list.append(f"{key_}_{suffix}_v")
        assert len(key_list) == len(ptr_list)
        return key_list, ptr_list, element_size_list

    def _get_mla_buffer_meta(self, keys, indices):
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)
        key_list = []
        for key_ in keys:
            key_list.append(f"{key_}_{self.mla_suffix}_k")
        assert len(key_list) == len(ptr_list)
        return key_list, ptr_list, element_size_list

    def _batch_preprocess(self, keys, host_indices):
        assert len(keys) > 0
        assert len(keys) == len(host_indices) // self.mem_pool_host.page_size
        if self.is_mla_backend:
            return self._get_mla_buffer_meta(keys, host_indices)
        else:
            if self.storage_config and self.storage_config.should_split_heads:
                return self._get_mha_split_heads_buffer_meta(keys, host_indices)
            else:
                return self._get_mha_buffer_meta(keys, host_indices)

    def _batch_postprocess(self, results: List[bool], is_set_operate=False):
        """Convert per-key-component results to per-page results.

        For MHA: each page has K+V → group pairs.
        For MLA: each page has K only.
        """
        if self.is_mla_backend:
            return list(results)
        else:
            if self.storage_config and self.storage_config.should_split_heads:
                group_size = self.split_factor * 2
                groups = [
                    results[i : i + group_size]
                    for i in range(0, len(results), group_size)
                ]
                return [all(g) for g in groups]
            else:
                # Group K/V pairs
                kv_pairs = zip(results[::2], results[1::2])
                return [k and v for k, v in kv_pairs]

    # ------------------------------------------------------------------
    # Zero-copy v1 interface
    # ------------------------------------------------------------------
    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(keys, host_indices)

        # Normalize sizes to list of per-key sizes
        if isinstance(buffer_sizes, int):
            sizes = [buffer_sizes] * len(key_strs)
        elif isinstance(buffer_sizes, list) and len(buffer_sizes) == 1:
            sizes = buffer_sizes * len(key_strs)
        else:
            sizes = list(buffer_sizes)

        get_results = self.client.batch_get_into_ptr(key_strs, list(buffer_ptrs), sizes)
        return self._batch_postprocess(get_results)

    def _compute_expanded_depths(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo]
    ) -> List[int]:
        """Compute per-expanded-key depth values from prefix_keys metadata.

        depth = len(prefix_keys) + page_index_within_node.
        All key variants of the same page (K, V, multi-rank) share the same depth.
        Returns an empty list if no metadata is available (caller falls back to plain LRU).
        """
        prefix_keys = getattr(extra_info, "prefix_keys", None) if extra_info else None
        if prefix_keys is None:
            return []

        prefix_len = len(prefix_keys)
        depths_per_page = [prefix_len + i for i in range(len(keys))]

        # Expand to match the key_strs layout produced by _batch_preprocess.
        expanded = []
        for d in depths_per_page:
            if self.is_mla_backend:
                expanded.append(d)  # MLA: 1 key per page
            elif self.storage_config and self.storage_config.should_split_heads:
                # split heads: 2 keys per split rank, split_factor ranks per page
                for _ in range(self.split_factor):
                    expanded.append(d)
                    expanded.append(d)
            else:
                expanded.append(d)  # K
                expanded.append(d)  # V
        return expanded

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        # Follower never writes (CacheController also sets backup_skip, but guard here too)
        if self.is_mla_follower:
            page_count = len(host_indices) // self.mem_pool_host.page_size
            return [True] * page_count

        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(keys, host_indices)

        if isinstance(buffer_sizes, int):
            sizes = [buffer_sizes] * len(key_strs)
        elif isinstance(buffer_sizes, list) and len(buffer_sizes) == 1:
            sizes = buffer_sizes * len(key_strs)
        else:
            sizes = list(buffer_sizes)

        expanded_depths = self._compute_expanded_depths(keys, extra_info)

        if expanded_depths:
            put_results = self.client.batch_put_from_ptr_with_depth(
                key_strs, list(buffer_ptrs), sizes, expanded_depths
            )
        else:
            put_results = self.client.batch_put_from_ptr(
                key_strs, list(buffer_ptrs), sizes
            )

        return self._batch_postprocess(put_results, is_set_operate=True)

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        """Return count of consecutive existing keys from start."""
        if self.is_mla_backend:
            query_keys = [f"{key}_{self.mla_suffix}_k" for key in keys]
            key_multiplier = 1
        else:
            query_keys = []
            if self.storage_config and self.storage_config.should_split_heads:
                for key in keys:
                    for suffix in self.mha_suffix:
                        query_keys.append(f"{key}_{suffix}_k")
                        query_keys.append(f"{key}_{suffix}_v")
                key_multiplier = 2 * self.split_factor
            else:
                for key in keys:
                    query_keys.append(f"{key}_{self.mha_suffix}_k")
                    query_keys.append(f"{key}_{self.mha_suffix}_v")
                key_multiplier = 2

        hit_count = self.client.batch_exists_consecutive(query_keys)
        return hit_count // key_multiplier

    # ------------------------------------------------------------------
    # Legacy ABC interface (required by HiCacheStorage)
    # ------------------------------------------------------------------
    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        if target_location is None or target_sizes is None:
            return None
        ok = self.client.get_into_ptr(key, target_location, target_sizes)
        return target_location if ok else None

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> int:
        if not keys:
            return 0
        assert len(keys) == len(target_locations) == len(target_sizes)
        results = self.client.batch_get_into_ptr(
            keys, list(target_locations), list(target_sizes)
        )
        for i, ok in enumerate(results):
            if not ok:
                return i
        return len(keys)

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if self.is_mla_follower:
            return True
        if target_location is None or target_sizes is None:
            return False
        return self.client.put_from_ptr(key, target_location, target_sizes)

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if not keys:
            return False
        if self.is_mla_follower:
            return True
        assert len(keys) == len(target_locations) == len(target_sizes)
        results = self.client.batch_put_from_ptr(
            keys, list(target_locations), list(target_sizes)
        )
        return all(results)

    def exists(self, key: str) -> bool:
        return self.client.exists(key)

    def clear(self) -> None:
        self.client.clear()
