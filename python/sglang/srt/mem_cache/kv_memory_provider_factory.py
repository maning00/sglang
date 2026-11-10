"""The sole constructor for explicit KV memory providers.

The legacy Mooncake environment hook deliberately stays uncached. Explicit
providers have one instance and one MemPool per device.
"""

from __future__ import annotations

import importlib
import inspect
from copy import deepcopy

from sglang.srt.mem_cache.kv_memory_provider import ByteMemoryProvider, KVMemoryProvider
from sglang.srt.runtime_context import get_context, get_memory, get_resources


class _ProviderSlot:
    def __init__(self, identity, provider):
        self.identity, self.provider = identity, provider
        self.pool = None


def _device_key(device):
    import torch

    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        # Serving configuration keeps only the device type. ModelRunner sets
        # the rank's current device before constructing the pools/provider.
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type != "cpu" and device.index is None:
        raise ValueError("KV provider requires an explicit device ordinal, e.g. cuda:0")
    return str(device)


def provider_config(config):
    if not isinstance(config, dict):
        raise ValueError("--kv-memory-provider-config must be a JSON object")
    unknown = set(config) - {"module", "class", "options", "kind"}
    if unknown or config.get("kind", "kv") not in ("kv", "bytes"):
        raise ValueError("invalid KV memory provider configuration")
    if not isinstance(config.get("options", {}), dict):
        raise ValueError("KV memory provider options must be an object")
    return deepcopy(config)


def provider_manages_pages(cfg):
    return (
        cfg.kv_memory_provider == "dynamic"
        and (cfg.kv_memory_provider_config or {}).get("kind", "kv") == "kv"
    )


def validate_provider_args(cfg):
    """Resolution-time guard: never import a plugin or construct a GPU pool."""
    name = cfg.kv_memory_provider
    if name is None:
        if cfg.kv_memory_provider_config is not None:
            raise ValueError("provider config requires --kv-memory-provider")
        return
    from sglang.srt.environ import envs

    if envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get() is not None:
        raise ValueError("explicit provider conflicts with the Mooncake allocator hook")
    if name == "mooncake":
        raise ValueError(
            "explicit Mooncake provider is unsupported; use the main Mooncake environment path"
        )
    if name != "dynamic":
        raise ValueError(f"unknown KV memory provider {name!r}")
    config = provider_config(cfg.kv_memory_provider_config)
    if not all(
        isinstance(config.get(k), str) and config[k] for k in ("module", "class")
    ):
        raise ValueError("dynamic provider requires module and class")
    if config.get("kind", "kv") == "bytes":
        return
    if cfg.disaggregation_mode != "prefill":
        raise ValueError("KV page provider requires PD prefill")
    if cfg.disaggregation_transfer_backend != "mori":
        raise ValueError("KV page provider requires PD physical send receipts (mori)")
    if cfg.enable_encoder_swa_bounded_replay or cfg.enable_memory_saver:
        raise ValueError(
            "KV page provider does not support encoder SWA bounded replay or memory saver"
        )
    if cfg.enable_prefill_cp or cfg.cp_kv_layout is not None:
        raise ValueError("KV page provider does not support CP padded/sharded layouts")
    if cfg.enable_unified_memory or cfg.enable_pd_role_switch:
        raise ValueError(
            "KV page provider does not support unified memory or role switch"
        )
    if (
        cfg.enable_unified_cache_external_linker
        or cfg.enable_hierarchical_cache
        or cfg.hicache_storage_backend is not None
        or cfg.enable_lmcache
        or cfg.unified_cache_external_linker_config is not None
        or cfg.unified_cache_external_linker_backend != "mooncake"
    ):
        raise ValueError(
            "KV page provider owns its linker; a second storage/linker is unsupported"
        )
    if cfg.disable_radix_cache or cfg.radix_cache_backend is not None:
        raise ValueError("KV page provider requires UnifiedRadixCache")
    if (
        cfg.page_size != 256
        or cfg.enable_hisparse
        or cfg.dllm_algorithm is not None
        or cfg.speculative_algorithm is not None
        or cfg.pp_size != 1
        or cfg.attn_cp_size != 1
        or cfg.dcp_size != 1
    ):
        raise ValueError("unsupported KV page provider layout/parallel configuration")
    namespace = config.get("options", {}).get("namespace", {})
    if not isinstance(namespace, dict) or not all(
        namespace.get(k) for k in ("model", "weights", "kv_format", "parallel")
    ):
        raise ValueError(
            "KV page provider requires model/weights/kv_format/parallel namespace"
        )


def get_or_create_kv_memory_provider(*, device: str, config=None):
    if config is None:
        if not get_context().is_config_namespace_published("memory"):
            return None
        memory = get_memory()
        name = memory.kv_memory_provider
        if name is None:
            return None
        config = memory.kv_memory_provider_config
    else:
        name = "dynamic"
    if name == "mooncake":
        raise ValueError(
            "explicit Mooncake provider is unsupported; use the environment path"
        )
    parsed = provider_config(config or {})
    providers = get_resources().kv_memory_providers
    # RuntimeContext is process-local; a worker owns its device in that process.
    key = _device_key(device)
    identity = (name, parsed)
    if key in providers:
        slot = providers[key]
        if slot.identity != identity:
            raise RuntimeError(
                "a different KV memory provider already owns this device"
            )
        return slot.provider
    cls = getattr(importlib.import_module(parsed["module"]), parsed["class"])
    if not inspect.isclass(cls) or inspect.isabstract(cls):
        raise TypeError("KV memory provider must be a concrete class")
    if getattr(cls, "process_singleton", False) and any(
        getattr(slot.provider, "process_singleton", False)
        for slot in providers.values()
    ):
        raise ValueError("this KV provider supports only one device/client per process")
    provider = cls(device=key, options=deepcopy(parsed.get("options", {})))
    protocol = ByteMemoryProvider if parsed.get("kind") == "bytes" else KVMemoryProvider
    if not isinstance(provider, protocol):
        raise TypeError(f"{cls.__name__} does not implement {protocol.__name__}")
    providers[key] = _ProviderSlot(identity, provider)
    return provider


def provider_mem_pool(provider, device):
    key = _device_key(device)
    slot = get_resources().kv_memory_providers[key]
    if slot.provider is not provider:
        raise RuntimeError("MemPool belongs to another KV provider")
    if slot.pool is None:
        slot.pool = provider.mem_pool(key)
        if slot.pool is None:
            raise TypeError("KV memory provider did not return a MemPool")
    return slot.pool


def full_provider_of(allocator):
    return getattr(allocator, "__dict__", {}).get("kv_memory_provider")
