"""Factory singleton, geometry, disabled-provider parity and boot guards."""

import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.mem_cache.kv_memory_provider import describe_dsv4_pool
from sglang.srt.mem_cache.kv_memory_provider_factory import (
    get_or_create_kv_memory_provider,
    provider_mem_pool,
    validate_provider_args,
)
from sglang.srt.runtime_context import get_context, get_resources
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

CONFIG = {
    "module": "sglang.test.kv_memory_provider",
    "class": "FakeKVMemoryProvider",
    "options": {
        "namespace": {
            "model": "test-model",
            "weights": "revision-1",
            "kv_format": "fp8",
            "parallel": "dep8",
        }
    },
}


class TestKVMemoryProvider(CustomTestCase):
    def setUp(self):
        self.enterContext(get_resources().override(kv_memory_providers={}))

    def test_singleton_in_resources_and_distinct_devices(self):
        with get_context().override_server_args():
            first = get_or_create_kv_memory_provider(device="cpu:0", config=CONFIG)
            again = get_or_create_kv_memory_provider(device="cpu:0", config=CONFIG)
            other = get_or_create_kv_memory_provider(device="cpu:1", config=CONFIG)
            self.assertIs(first, again)
            self.assertIs(first.mem_pool("cpu:0"), again.mem_pool("cpu:0"))
            self.assertIsNot(first, other)
            with patch.object(first, "mem_pool", side_effect=[object(), object()]):
                pool = provider_mem_pool(first, "cpu:0")
                self.assertIs(pool, provider_mem_pool(first, "cpu:0"))
            self.assertEqual(len(get_resources().kv_memory_providers), 2)
            with self.assertRaisesRegex(RuntimeError, "already owns"):
                get_or_create_kv_memory_provider(
                    device="cpu:0", config={**CONFIG, "options": {"pages": 2}}
                )

    def test_c25_disabled_provider_does_not_import_mori_or_create_resources(self):
        script = """
import sys
from unittest.mock import patch
from sglang.srt.runtime_context import get_context, get_resources
from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool
with get_context().override_server_args():
    assert maybe_init_custom_mem_pool('cpu') == (False, None, None)
    assert not get_resources().kv_memory_providers
    assert not any(x == 'mori' or x.startswith('mori.') for x in sys.modules)
"""
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_legacy_mooncake_hook_uncached_and_explicit_provider_cached(self):
        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool

        pools = [object(), object()]
        with (
            get_context().override_server_args(),
            envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.override("NVLINK"),
        ):
            with patch(
                "sglang.srt.disaggregation.mooncake.utils.init_mooncake_custom_mem_pool",
                side_effect=[(True, pool, "NVLINK") for pool in pools],
            ):
                self.assertIs(maybe_init_custom_mem_pool("cpu")[1], pools[0])
                self.assertIs(maybe_init_custom_mem_pool("cpu")[1], pools[1])
        with get_context().override_server_args(kv_memory_provider="mooncake"):
            with self.assertRaisesRegex(ValueError, "explicit.*unsupported"):
                maybe_init_custom_mem_pool("cpu")

    def test_c26_rejects_unsupported_configuration_before_plugin_import(self):
        base = dict(
            kv_memory_provider="dynamic",
            kv_memory_provider_config=CONFIG,
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="mori",
            page_size=256,
        )
        invalid = [
            dict(disaggregation_mode="decode"),
            dict(disaggregation_mode="null"),
            dict(enable_unified_memory=True),
            dict(enable_pd_role_switch=True),
            dict(enable_hierarchical_cache=True),
            dict(enable_unified_cache_external_linker=True),
            dict(hicache_storage_backend="mori"),
            dict(unified_cache_external_linker_config={"linker": "second"}),
            dict(unified_cache_external_linker_backend="mori"),
            dict(page_size=128),
            dict(disable_radix_cache=True),
            dict(dllm_algorithm="test"),
            dict(attn_cp_size=2),
        ]
        for fields in invalid:
            with (
                self.subTest(fields=fields),
                get_context().override_server_args(**(base | fields)),
            ):
                with self.assertRaises(ValueError):
                    validate_provider_args(resolving_view(get_context().server_args))
                self.assertEqual(get_resources().kv_memory_providers, {})
        with get_context().override_server_args(**base):
            validate_provider_args(resolving_view(get_context().server_args))

    def test_c26_model_layout_guard_precedes_provider_constructor_and_gpu_pools(self):
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

        # Only the model boundary is substituted. The real pool entry must
        # reject before it can instantiate the provider or allocate any pool.
        config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"])
            )
        )
        with get_context().override_server_args(
            kv_memory_provider="dynamic", kv_memory_provider_config=CONFIG
        ):
            with self.assertRaisesRegex(ValueError, "DSv4"):
                KVCacheConfigurator._init_pools(
                    config,
                    sizes=None,
                    req_to_token_pool=None,
                    token_to_kv_pool_allocator=None,
                )
            self.assertFalse(get_resources().kv_memory_providers)

    def test_geometry_182_regions_layout_independent_of_addresses(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4IndexerPool,
            DeepSeekV4TokenToKVPool,
        )

        def pool():
            # Real DSv4 export methods over CPU byte backing: 61 compressed
            # layers, 30 split FP4 indexers, two FULL pages plus padding. No
            # GPU constructor, VMM allocator or copy kernel is substituted into
            # production. Ring rows precede the compressed page-zero row.
            result = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
            result.page_size = 256
            result._stage_start, result._stage_end = 0, 61
            result.compression_ratios = [4] * 30 + [128] * 31
            result.sources_by_ratio = {4: list(range(30)), 128: list(range(30, 61))}
            result._unified_kv = result._unified_kv_fp8 = True
            result.kv_pools = {4: None, 128: None}
            result.unified_kv_pool = SimpleNamespace(
                swa_pages=3,
                kv_buffer=[
                    torch.empty((3 + 3 * (256 // ratio), 512), dtype=torch.uint8)
                    for ratio in result.compression_ratios
                ],
                kv_buffer_rope=[
                    torch.empty((3 + 3 * (256 // ratio), 64), dtype=torch.bfloat16)
                    for ratio in result.compression_ratios
                ],
            )
            indexer = DeepSeekV4IndexerPool.__new__(DeepSeekV4IndexerPool)
            indexer.index_k_with_scale_buffer = None
            indexer.index_k_payload_buffer = [
                torch.empty((3, 1, 1, 64, 64), dtype=torch.uint8) for _ in range(30)
            ]
            indexer.index_k_scale_buffer = [
                torch.empty((3, 1, 1, 64, 4), dtype=torch.uint8) for _ in range(30)
            ]
            result.index_pools = {4: indexer}
            result.request_window = result.swa_kv_pool = result.swa_valid_from = None
            result.compress_state_pools = result.indexer_compress_state_pools = []
            return result

        original = pool()
        first, request_regions = describe_dsv4_pool(original, num_pages=2)
        second, _ = describe_dsv4_pool(pool(), num_pages=2)
        self.assertEqual(first.layout_id, second.layout_id)
        self.assertEqual(len(first.regions), 182)
        self.assertEqual(first.object_bytes, 1_399_040)
        first_buffer = original.unified_kv_pool.kv_buffer[0]
        self.assertEqual(first.regions[0].base, first_buffer.data_ptr() + 3 * 512)
        self.assertEqual(
            first.regions[0].base + first.regions[0].page_bytes,
            first_buffer.data_ptr() + (3 + 64) * 512,
        )
        self.assertEqual(len(request_regions), 122)
        self.assertEqual(
            first.regions[-1].logical_offset + first.regions[-1].page_bytes,
            first.object_bytes,
        )


if __name__ == "__main__":
    unittest.main()
