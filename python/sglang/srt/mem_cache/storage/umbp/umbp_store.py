"""UMBPStore — HiCache L3 storage backend using UMBP (local DRAM + SSD).

Follows the same pattern as MooncakeStore:
- Zero-copy v1 interface (batch_get_v1 / batch_set_v1)
- Uses mem_pool_host.get_page_buffer_meta() for pointer/size extraction
- Key suffix generation per TP rank / PP rank
"""

import logging
import os
import sys
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
    """Import UMBPClient from the compiled C++ module."""
    try:
        # Try direct import first (if installed or in sys.path)
        from _umbp_core import UMBPClient, UMBPConfig

        return UMBPClient, UMBPConfig
    except ImportError:
        # Fall back to looking in the build directory
        # Navigate from sglang/python/sglang/srt/mem_cache/storage/umbp/ up to KVManager/
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        # Try common locations
        candidates = [
            os.path.normpath(os.path.join(_this_dir, "..", "..", "..", "..", "..", "..", "..", "..", "umbp", "build")),
            os.path.normpath(os.path.join(_this_dir, "..", "..", "..", "..", "..", "..", "umbp", "build")),
        ]
        # Also check UMBP_BUILD_DIR env var
        if os.environ.get("UMBP_BUILD_DIR"):
            candidates.insert(0, os.environ["UMBP_BUILD_DIR"])

        umbp_build = None
        for candidate in candidates:
            if os.path.isdir(candidate):
                umbp_build = candidate
                break

        if umbp_build is None:
            raise ImportError(
                "Cannot import _umbp_core. Build the UMBP C++ module first: "
                "cd umbp && mkdir build && cd build && "
                "cmake .. -Dpybind11_DIR=$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())') "
                "&& make -j"
            )
        sys.path.insert(0, umbp_build)
        from _umbp_core import UMBPClient, UMBPConfig

        return UMBPClient, UMBPConfig


class UMBPStore(HiCacheStorage):
    """Local DRAM+SSD storage backend for HiCache L3 caching.

    Compatible with the zero-copy v1 interface used by CacheController.
    """

    def __init__(
        self,
        storage_config: HiCacheStorageConfig = None,
        mem_pool_host: HostKVCache = None,
    ):
        UMBPClient, UMBPConfig = _import_umbp_client()

        cfg = UMBPConfig()

        # Load settings from extra_config if available
        extra = getattr(storage_config, "extra_config", None) or {}
        if "dram_capacity_bytes" in extra:
            cfg.dram_capacity_bytes = int(extra["dram_capacity_bytes"])
        if "ssd_enabled" in extra:
            cfg.ssd_enabled = bool(extra["ssd_enabled"])
        if "ssd_storage_dir" in extra:
            cfg.ssd_storage_dir = str(extra["ssd_storage_dir"])
        if "ssd_capacity_bytes" in extra:
            cfg.ssd_capacity_bytes = int(extra["ssd_capacity_bytes"])
        if "auto_promote_on_read" in extra:
            cfg.auto_promote_on_read = bool(extra["auto_promote_on_read"])

        self.client = UMBPClient(cfg)
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

        self.enable_pp = self.pp_size > 1
        if self.enable_pp:
            self.mha_suffix = f"{self.local_rank}_{self.pp_rank}"
            self.mla_suffix = f"{self.pp_rank}"
        else:
            self.mha_suffix = f"{self.local_rank}"
            self.mla_suffix = ""

        self.split_factor = 0
        if storage_config and storage_config.should_split_heads:
            self.split_factor = (
                storage_config.tp_lcm_size // storage_config.tp_size
            )
            base_rank = self.local_rank * self.split_factor
            target_ranks = [base_rank + i for i in range(self.split_factor)]
            if self.enable_pp:
                self.mha_suffix = [
                    f"{rank}_{self.pp_rank}" for rank in target_ranks
                ]
            else:
                self.mha_suffix = [f"{rank}" for rank in target_ranks]

        logger.info(
            "UMBPStore initialized: dram=%d MB, ssd=%s, mla=%s, rank=%d",
            cfg.dram_capacity_bytes // (1024 * 1024),
            cfg.ssd_enabled,
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
        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(
            keys, host_indices
        )

        # Normalize sizes to list of per-key sizes
        if isinstance(buffer_sizes, int):
            sizes = [buffer_sizes] * len(key_strs)
        elif isinstance(buffer_sizes, list) and len(buffer_sizes) == 1:
            sizes = buffer_sizes * len(key_strs)
        else:
            sizes = list(buffer_sizes)

        get_results = self.client.batch_get_into_ptr(
            key_strs, list(buffer_ptrs), sizes
        )
        return self._batch_postprocess(get_results)

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(
            keys, host_indices
        )

        if isinstance(buffer_sizes, int):
            sizes = [buffer_sizes] * len(key_strs)
        elif isinstance(buffer_sizes, list) and len(buffer_sizes) == 1:
            sizes = buffer_sizes * len(key_strs)
        else:
            sizes = list(buffer_sizes)

        # Dedup: skip already-existing keys
        exist_results = self.client.batch_exists(key_strs)

        set_keys = []
        set_ptrs = []
        set_sizes = []
        set_indices = []
        final_results = [False] * len(key_strs)

        for i in range(len(key_strs)):
            if exist_results[i]:
                final_results[i] = True  # Already exists
            else:
                set_keys.append(key_strs[i])
                set_ptrs.append(buffer_ptrs[i])
                set_sizes.append(sizes[i])
                set_indices.append(i)

        if set_keys:
            put_results = self.client.batch_put_from_ptr(set_keys, set_ptrs, set_sizes)
            for idx, result in zip(set_indices, put_results):
                final_results[idx] = result

        return self._batch_postprocess(final_results, is_set_operate=True)

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

        exist_results = self.client.batch_exists(query_keys)
        for i in range(len(query_keys)):
            if not exist_results[i]:
                return i // key_multiplier
        return len(query_keys) // key_multiplier

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
        assert len(keys) == len(target_locations) == len(target_sizes)
        results = self.client.batch_put_from_ptr(
            keys, list(target_locations), list(target_sizes)
        )
        return all(results)

    def exists(self, key: str) -> bool:
        return self.client.exists(key)

    def clear(self) -> None:
        self.client.clear()
