import json
import os
import threading
import unittest
from enum import Enum
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
)
from sglang.srt.mem_cache.storage.umbp import umbp_direct_linker
from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import (
    UMBPDirectLinker,
    _object_sizes_per_page,
    _PoolRangePlan,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinkerWrapper,
    with_direct_linker_cache_layout_tag,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _DeploymentMode(Enum):
    Local = 0
    StandaloneProcess = 1
    Distributed = 2


class TestUMBPDirectLinkerRegressions(unittest.TestCase):
    page_size = 2
    num_layers = 3

    def setUp(self):
        self.kv_buffers = [
            torch.zeros((32, 1, 4), dtype=torch.uint8) for _ in range(self.num_layers)
        ]
        self.indexer_buffers = [
            torch.zeros((16, 6), dtype=torch.uint8) for _ in range(self.num_layers)
        ]
        identity = {layer: layer for layer in range(self.num_layers)}
        self.pool_group = DevicePoolGroup(
            [
                DevicePoolEntry(
                    name=PoolName.KV,
                    indices_from_pool=PoolName.KV,
                    device_pool=None,
                    components=[self.kv_buffers],
                    layer_mapping=identity,
                    page_size=self.page_size,
                    rows_are_pages=False,
                ),
                DevicePoolEntry(
                    name=PoolName.INDEXER,
                    indices_from_pool=PoolName.KV,
                    device_pool=None,
                    components=[self.indexer_buffers],
                    layer_mapping=identity,
                    page_size=self.page_size,
                    rows_are_pages=True,
                ),
            ],
            self.num_layers,
            self.page_size,
        )
        self.pools = self.pool_group.entry_map

        self.client = MagicMock()
        self.client.is_distributed.return_value = True
        self.client.get_deployment_mode.return_value = _DeploymentMode.StandaloneProcess
        self.client.get_backend_mode.return_value = _DeploymentMode.Local
        self.client.supports_ranged_io.return_value = True
        self.client.register_memory.return_value = True
        self.client.batch_exists.side_effect = lambda keys: [True] * len(keys)
        self.client.batch_put_ranges_from_ptr.side_effect = lambda keys, *args: [
            True
        ] * len(keys)
        self.client.batch_get_ranges_into_ptr.side_effect = lambda keys, *args: [
            True
        ] * len(keys)
        self.client.report_external_kv_blocks.return_value = True
        self.client.revoke_external_kv_blocks.return_value = True
        self.client.revoke_all_external_kv_blocks_at_tier.return_value = True

        self.storage = MagicMock()
        self.storage.client = self.client
        self.storage._disable_zero_copy_register = False
        self.storage._get_hybrid_page_component_keys.side_effect = (
            lambda keys, transfer, rank_suffix=None: (
                [f"{key}_{rank_suffix or 'rank'}_{transfer.name}" for key in keys],
                1,
            )
        )

        self.freeze_gc_patcher = patch(
            "sglang.srt.mem_cache.storage.umbp.umbp_direct_linker.freeze_gc"
        )
        self.freeze_gc_mock = self.freeze_gc_patcher.start()
        self.addCleanup(self.freeze_gc_patcher.stop)
        self.event_patcher = patch(
            "sglang.srt.mem_cache.storage.umbp.umbp_direct_linker.device_module.Event",
            side_effect=lambda: SimpleNamespace(
                record=lambda: None, synchronize=lambda: None
            ),
        )
        self.event_patcher.start()
        self.addCleanup(self.event_patcher.stop)

        self.server_args = SimpleNamespace(
            hicache_storage_backend_extra_config=None,
            tp_size=1,
            pp_size=1,
            attn_cp_size=1,
            enable_dp_attention=False,
            model_path="test-model",
        )
        self.params = SimpleNamespace(
            page_size=self.page_size,
            req_to_token_pool=MagicMock(),
            token_to_kv_pool_allocator=MagicMock(),
            tp_cache_group=None,
            attn_cp_cache_group=None,
            attn_tp_cache_group=None,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
        )
        self.connectors = []

    def tearDown(self):
        for connector in self.connectors:
            connector.close()

    def make_connector(
        self,
        extra_config=None,
        pool_group=None,
    ):
        pool_group = pool_group or self.pool_group
        raw_config = json.dumps(extra_config) if extra_config is not None else None
        self.server_args.hicache_storage_backend_extra_config = raw_config
        with (
            # A sibling test leaves gloo initialized without sglang parallel groups.
            patch("torch.distributed.is_initialized", return_value=False),
            patch.object(
                umbp_direct_linker,
                "get_memory",
                return_value=SimpleNamespace(
                    hicache_storage_backend_extra_config=raw_config
                ),
            ),
            patch.object(
                umbp_direct_linker,
                "get_model",
                return_value=SimpleNamespace(model_path="test-model"),
            ),
            patch.object(
                umbp_direct_linker,
                "resolve_hybrid_device_pool_group",
                return_value=pool_group,
            ),
            patch(
                "sglang.srt.mem_cache.storage.umbp.umbp_direct_linker._parse_storage_extra_config",
                return_value=dict(extra_config or {}),
            ),
        ):
            connector = UMBPDirectLinker(
                self.server_args,
                self.params,
                components={ComponentType.FULL},
                _storage=self.storage,
            )
        self.connectors.append(connector)
        return connector

    def transfer(self, pages=2):
        starts = torch.arange(pages, dtype=torch.int64) * self.page_size
        offsets = torch.arange(self.page_size, dtype=torch.int64)
        indices = (starts[:, None] + offsets).flatten()
        return PoolTransfer(
            name=PoolName.KV,
            device_indices=indices,
            keys=[f"page-{index}" for index in range(pages)],
        )

    def test_lookup_narrows_the_probe_to_the_surviving_boundary(self):
        """Every pool after the first probes only up to the longest candidate.

        The probe is a synchronous round trip inside the scheduler's prefill
        batch build, and DP-attention ranks are lockstep, so a pool that
        re-probes the full key list charges every rank for keys whose answer
        cannot change the boundary.
        """
        connector = self.make_connector()
        pages = 8
        kv_present = 3

        probes = []

        def batch_exists(keys):
            probes.append(list(keys))
            if keys and keys[0].endswith(str(PoolName.KV)):
                return [index < kv_present for index in range(len(keys))]
            return [True] * len(keys)

        self.client.batch_exists.side_effect = batch_exists

        valid_pages = connector.lookup("rid", [self.transfer(pages=pages)])

        self.assertEqual(valid_pages, list(range(1, kv_present + 1)))
        self.assertEqual(len(probes), 2)
        self.assertEqual(len(probes[0]), pages)
        self.assertEqual(len(probes[1]), kv_present)

    def test_lookup_probe_narrowing_does_not_change_the_boundary(self):
        """Narrowing is an optimisation: the answer matches an unnarrowed probe."""
        connector = self.make_connector()
        pages = 8

        for kv_present in range(pages + 1):
            for indexer_present in range(pages + 1):

                def batch_exists(keys, kv=kv_present, indexer=indexer_present):
                    limit = kv if keys and keys[0].endswith(str(PoolName.KV)) else indexer
                    return [index < limit for index in range(len(keys))]

                self.client.batch_exists.side_effect = batch_exists
                got = connector.lookup("rid", [self.transfer(pages=pages)])
                expected = list(range(1, min(kv_present, indexer_present) + 1))
                self.assertEqual(got, expected, f"kv={kv_present} indexer={indexer_present}")
    def dsv4_pool_group(self):
        entry = DevicePoolEntry(
            name=PoolName.DEEPSEEK_V4_C4,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[self.kv_buffers],
            layer_mapping={layer: layer for layer in range(self.num_layers)},
            page_size=self.page_size,
            rows_are_pages=False,
        )
        return DevicePoolGroup(
            [entry], self.num_layers, self.page_size, rank_replicated=True
        )

    def test_dsv4_layout_tag_is_appended_to_umbp_storage_config(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )

        kvcache = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        kvcache._unified_kv = True
        kvcache.c4_indexer_kv_pool = SimpleNamespace(use_fp4_indexer=False)
        self.params.token_to_kv_pool_allocator.get_kvcache.return_value = kvcache
        self.params.pp_rank = 1
        self.params.pp_size = 4

        captured = {}

        def make_storage_config(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(**kwargs)

        with (
            patch.dict(os.environ, {}, clear=False),
            patch.object(
                umbp_direct_linker,
                "HiCacheStorageConfig",
                side_effect=make_storage_config,
            ),
        ):
            os.environ.pop("SGLANG_PP_LAYER_PARTITION", None)
            self.make_connector(
                {"extra_backend_tag": "tenant-a", "custom_option": "kept"},
                pool_group=self.dsv4_pool_group(),
            )

        self.assertEqual(captured["extra_config"]["custom_option"], "kept")
        self.assertEqual(
            captured["extra_config"]["extra_backend_tag"],
            "tenant-a__ucdl-dsv4-v1-layout-unified-bf16-indexer-int8-"
            "pp4-layers-auto",
        )

    def test_dsv4_layout_tag_is_identical_across_pp_ranks_for_extkv(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )

        kvcache = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        kvcache._unified_kv = True
        kvcache.c4_indexer_kv_pool = SimpleNamespace(use_fp4_indexer=False)
        with patch.dict(
            os.environ, {"SGLANG_PP_LAYER_PARTITION": "30,31"}, clear=False
        ):
            tags = [
                with_direct_linker_cache_layout_tag(
                    {"extra_backend_tag": "tenant-a"},
                    kvcache=kvcache,
                    pool_group=self.dsv4_pool_group(),
                    pp_rank=rank,
                    pp_size=2,
                )["extra_backend_tag"]
                for rank in range(2)
            ]

        self.assertEqual(tags[0], tags[1])
        self.assertTrue(tags[0].endswith("-pp2-layers-30.31"))

    @staticmethod
    def wait_for_offloads(connector):
        connector._offload_queue.join()

    def test_offload_publishes_result_even_if_extkv_raises(self):
        connector = self.make_connector()
        with patch.object(
            connector,
            "_extkv_page_hashes",
            side_effect=RuntimeError("extkv failed"),
        ):
            self.assertTrue(connector.offload([self.transfer(pages=1)]))
            self.wait_for_offloads(connector)

        self.assertEqual(connector.num_completed_offloads(), 1)
        self.assertFalse(connector.pop_completed_offload())
        self.client.batch_put_ranges_from_ptr.assert_not_called()

    def test_offload_results_are_fifo(self):
        def result_for_key(keys, *args):
            return [not keys[0].startswith("fail-")] * len(keys)

        self.client.batch_put_ranges_from_ptr.side_effect = result_for_key
        connector = self.make_connector()
        # One task per batch. A coalesced batch resolves as a unit, so the
        # per-task outcomes asserted below are only well defined while the two
        # tasks cannot travel together.
        connector._offload_coalesce_pages = 1
        success = self.transfer(pages=1)
        success.keys = ["success-page"]
        failure = self.transfer(pages=1)
        failure.keys = ["fail-page"]

        self.assertTrue(connector.offload([success]))
        self.assertTrue(connector.offload([failure]))
        self.wait_for_offloads(connector)

        self.assertEqual(connector.num_completed_offloads(), 2)
        self.assertTrue(connector.pop_completed_offload())
        self.assertFalse(connector.pop_completed_offload())

    def offset_transfer(self, first_page, pages=1):
        """A transfer over its own page range, so tasks never overlap."""
        starts = (torch.arange(pages, dtype=torch.int64) + first_page) * self.page_size
        offsets = torch.arange(self.page_size, dtype=torch.int64)
        indices = (starts[:, None] + offsets).flatten()
        return PoolTransfer(
            name=PoolName.KV,
            device_indices=indices,
            keys=[f"page-{first_page + index}" for index in range(pages)],
        )

    def resolved_task(self, connector, transfer):
        return (
            connector.pool_group.resolve_transfers([transfer], allow_partial=True),
            SimpleNamespace(synchronize=lambda: None),
        )

    def test_coalesced_offload_puts_one_batch_per_pool_not_one_per_task(self):
        """Tasks already queued travel together, and each still gets a result.

        The tree offloads once per node walked during insert, so tasks arrive a
        page or two at a time and each one used to pay the per-call cost on its
        own. A DSv4-Pro agentic trace put 1.9 keys per call on this path against
        the load path's 149.
        """
        entered = threading.Event()
        gate = threading.Event()
        calls = []

        def gated_put(keys, *args):
            calls.append(list(keys))
            if len(calls) == 1:
                # Hold the worker past its first drain so the remaining tasks
                # pile up behind it; otherwise this races the thread.
                entered.set()
                gate.wait(timeout=10)
            return [True] * len(keys)

        self.client.batch_put_ranges_from_ptr.side_effect = gated_put
        connector = self.make_connector()
        pools = len(
            connector.pool_group.resolve_transfers(
                [self.offset_transfer(0)], allow_partial=True
            )
        )

        self.assertTrue(connector.offload([self.offset_transfer(0)]))
        self.assertTrue(entered.wait(timeout=10))
        for page in range(1, 5):
            self.assertTrue(connector.offload([self.offset_transfer(page)]))
        gate.set()
        self.wait_for_offloads(connector)

        # Two batches, not five: the lone task, then the four that queued up.
        self.assertEqual(len(calls), 2 * pools)
        self.assertEqual(
            sorted(len(keys) for keys in calls), sorted([1] * pools + [4] * pools)
        )
        # The tree pairs results positionally with its own pending list, so the
        # count must follow tasks and not batches.
        self.assertEqual(connector.num_completed_offloads(), 5)
        # The ratio of these two is the coalescing a real workload achieves --
        # the number that says whether this path is worth anything there.
        self.assertEqual(connector._stats["offload"], 5)
        self.assertEqual(connector._stats["offload_batches"], 2)

    def test_coalescing_puts_the_same_ranges_as_one_task_at_a_time(self):
        """A coalesced batch must put exactly what the tasks put separately.

        Merging concatenates the pointer/size/offset lists that address device
        memory. An off-by-one here still succeeds -- it just writes real KV
        bytes at the wrong offset -- so compare the two forms item by item.
        """
        transfers = [self.offset_transfer(page) for page in range(4)]

        def put_args_for(batches):
            self.client.batch_put_ranges_from_ptr.reset_mock()
            connector = self.make_connector()
            for batch in batches:
                tasks = [self.resolved_task(connector, item) for item in batch]
                self.assertTrue(connector._run_offload(tasks))
            return [
                call.args
                for call in self.client.batch_put_ranges_from_ptr.call_args_list
            ]

        separate = put_args_for([[transfer] for transfer in transfers])
        merged = put_args_for([transfers])

        # Both arms visit the pools in plan order, so striding the per-task
        # calls by the pool count recovers each pool's own sequence.
        self.assertEqual(len(separate), len(merged) * len(transfers))
        for pool_index, merged_args in enumerate(merged):
            per_task = separate[pool_index :: len(merged)]
            expected = [
                [item for args in per_task for item in args[position]]
                for position in range(len(merged_args))
            ]
            actual = [list(value) for value in merged_args]
            self.assertEqual(actual, expected, f"pool_index={pool_index}")

    def take_offload_batch(self, queued, budget):
        """Drive the drain with no worker thread competing for the queue."""
        stub = SimpleNamespace(_offload_queue=Queue(), _offload_coalesce_pages=budget)
        for item in queued:
            stub._offload_queue.put(item)
        return UMBPDirectLinker._take_offload_batch(stub)

    def test_offload_batch_stops_at_the_page_budget(self):
        tasks = [
            ([self.offset_transfer(page, pages=2)], object()) for page in range(0, 8, 2)
        ]
        taken, stopping = self.take_offload_batch(tasks, budget=4)

        self.assertFalse(stopping)
        # Two 2-page tasks reach the budget. The rest wait for the next batch
        # instead of growing one plan without bound.
        self.assertEqual(len(taken), 2)

    def test_offload_batch_reports_the_stop_sentinel_with_its_tasks(self):
        task = ([self.offset_transfer(0)], object())
        taken, stopping = self.take_offload_batch([task, None], budget=64)

        # The sentinel must not strand the task that preceded it: it is already
        # off the queue, and the tree is waiting for its result.
        self.assertTrue(stopping)
        self.assertEqual(taken, [task])

    def offload_pages(self, connector, count):
        for _ in range(count):
            self.assertTrue(connector.offload([self.transfer(pages=1)]))
        self.wait_for_offloads(connector)

    def test_completed_offload_outcomes_are_local(self):
        connector = self.make_connector()
        connector._offload_results.put(True)
        connector._offload_results.put(False)

        with patch("torch.distributed.all_reduce") as all_reduce:
            self.assertEqual(connector.num_completed_offloads(), 2)
            self.assertTrue(connector.pop_completed_offload())
            self.assertFalse(connector.pop_completed_offload())
        all_reduce.assert_not_called()

    def test_idle_steps_issue_no_collective(self):
        """Counting completed loads and offloads never issues a collective."""
        connector = self.make_connector()

        with patch("torch.distributed.all_reduce") as all_reduce:
            self.assertEqual(connector.num_completed_loads(), 0)
            self.assertEqual(connector.num_completed_offloads(), 0)

        all_reduce.assert_not_called()

    def test_wrapper_drain_releases_nodes_with_agreed_outcomes(self):
        connector = self.make_connector()
        self.offload_pages(connector, 3)
        nodes = [SimpleNamespace(id=i, external_cache_stored=False) for i in range(3)]
        cache = SimpleNamespace(
            released=[],
        )
        cache.resolve_node_handle = lambda node_id: nodes[node_id]
        cache.dec_lock_ref = lambda node_id, params: cache.released.append(node_id)
        wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
        wrapper.cache = cache
        wrapper.cache_linker = connector
        wrapper.pending_offloads = [(node.id, object()) for node in nodes]

        local_results = wrapper.take_completed_offloads(1)
        self.assertEqual(local_results, [True])
        wrapper.commit_completed_offloads([False])

        self.assertEqual(cache.released, [0])
        self.assertEqual(len(wrapper.pending_offloads), 2)
        self.assertFalse(nodes[0].external_cache_stored)

    def test_layerwise_load_waits_for_recording_stream(self):
        calls = []

        class _Event:
            def record(self):
                calls.append("record")

            def synchronize(self):
                calls.append("synchronize")

        def get_ranges(keys, *args):
            calls.append("get")
            return [True] * len(keys)

        connector = self.make_connector()
        self.client.batch_get_ranges_into_ptr.side_effect = get_ranges
        with patch.object(umbp_direct_linker.device_module, "Event", _Event):
            self.assertTrue(connector.load("rid", [self.transfer(pages=1)]))
            counter = connector.start_layer_wise_loading()
            connector.layer_done_counter.set_consumer(counter)
            connector.layer_done_counter.wait_until(self.num_layers - 1)

        self.assertGreater(calls.count("get"), 0)
        self.assertEqual(calls[:3], ["record", "synchronize", "get"])

    def test_layerwise_load_sync_failure_reaches_consumer(self):
        connector = self.make_connector()
        expanded = connector.pool_group.resolve_transfers([self.transfer(pages=1)])
        plans = connector._build_load_plans([expanded])
        counter = connector.layer_done_counter.update_producer()
        connector.layer_done_counter.set_consumer(counter)
        ready_event = MagicMock()
        ready_event.synchronize.side_effect = RuntimeError("event failed")
        self.client.batch_get_ranges_into_ptr.reset_mock()

        with self.assertLogs(umbp_direct_linker.logger, level="ERROR"):
            connector._run_layer_wise_batch(counter, plans, ready_event)

        with self.assertRaisesRegex(
            RuntimeError, "UMBP layer-wise KV load failed"
        ) as raised:
            connector.layer_done_counter.wait_until(self.num_layers - 1)
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        self.assertEqual(str(raised.exception.__cause__), "event failed")
        ready_event.synchronize.assert_called_once_with()
        self.client.batch_get_ranges_into_ptr.assert_not_called()

    def test_load_publishes_ids_only_after_transfer(self):
        worker_started = threading.Event()
        release_worker = threading.Event()

        def blocked_get(keys, *args):
            worker_started.set()
            if not release_worker.wait(timeout=5):
                raise TimeoutError("test did not release load worker")
            return [True] * len(keys)

        self.client.batch_get_ranges_into_ptr.side_effect = blocked_get
        connector = self.make_connector()
        self.assertTrue(connector.load("rid-a", [self.transfer(pages=1)]))
        self.assertTrue(connector.load("rid-b", [self.transfer(pages=1)]))
        counter = connector.start_layer_wise_loading()
        connector.layer_done_counter.set_consumer(counter)
        self.assertTrue(worker_started.wait(timeout=5))
        try:
            self.assertEqual(connector.num_completed_loads(), 0)
        finally:
            release_worker.set()

        connector._load_queue.join()
        connector.layer_done_counter.wait_until(self.num_layers - 1)
        self.assertEqual(connector.num_completed_loads(), 1)
        self.assertEqual(connector.pop_completed_load(), ["rid-a", "rid-b"])

    def test_worker_publishes_ids_even_if_batch_raises(self):
        connector = UMBPDirectLinker.__new__(UMBPDirectLinker)
        connector._load_queue = Queue()
        connector._completed_loads = Queue()
        connector._run_layer_wise_batch = MagicMock(
            side_effect=RuntimeError("batch failed")
        )
        connector._load_queue.put(
            (7, ["rid"], [object()], SimpleNamespace(synchronize=lambda: None))
        )

        with self.assertRaisesRegex(RuntimeError, "batch failed"):
            connector._load_thread_func()

        self.assertEqual(connector._completed_loads.get(timeout=1), ["rid"])
        self.assertEqual(connector._load_queue.unfinished_tasks, 0)

    def test_cancel_queued_load_never_cancels(self):
        connector = self.make_connector()
        self.assertTrue(connector.load("queued", [self.transfer(pages=1)]))
        queued = connector._pending["queued"]

        self.assertFalse(connector.cancel_queued_load("queued"))
        self.assertIs(connector._pending["queued"], queued)

        connector.start_layer_wise_loading()
        self.assertFalse(connector.cancel_queued_load("queued"))
        self.assertFalse(connector.cancel_queued_load("missing"))
        self.assertEqual(connector._pending, {})
        connector._load_queue.join()

    def test_queue_failure_keeps_pending_for_retry(self):
        for failure_point in ("build", "put"):
            with self.subTest(failure_point=failure_point):
                connector = self.make_connector()
                rid = f"rid-{failure_point}"
                self.assertTrue(connector.load(rid, [self.transfer(pages=1)]))
                target = (
                    patch.object(
                        connector,
                        "_build_load_plans",
                        side_effect=RuntimeError("plan failed"),
                    )
                    if failure_point == "build"
                    else patch.object(
                        connector._load_queue,
                        "put",
                        side_effect=RuntimeError("queue failed"),
                    )
                )

                with target, self.assertRaisesRegex(RuntimeError, "failed"):
                    connector.start_layer_wise_loading()

                self.assertIn(rid, connector._pending)
                self.assertEqual(connector.num_completed_loads(), 0)

    def test_reset_discards_completed_loads(self):
        connector = self.make_connector()
        connector._completed_loads.put(["stale"])

        connector.reset()

        self.assertEqual(connector.num_completed_loads(), 0)

    def test_queue_load_releases_anchor_on_failure(self):
        for mode in ("raise", "false"):
            with self.subTest(mode=mode):
                lock_params = object()
                locks = []
                unlocks = []

                def inc_lock_ref(node_id):
                    locks.append(node_id)
                    return SimpleNamespace(to_dec_params=lambda: lock_params)

                wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
                wrapper.pending_loads = {}
                wrapper.cache = SimpleNamespace(
                    inc_lock_ref=inc_lock_ref,
                    dec_lock_ref=lambda node_id, params: unlocks.append(
                        (node_id, params)
                    ),
                )
                wrapper.cache_linker = SimpleNamespace(
                    load=(
                        (lambda rid, transfers: False)
                        if mode == "false"
                        else MagicMock(side_effect=RuntimeError("load failed"))
                    )
                )

                with self.assertRaises(RuntimeError):
                    wrapper._queue_load("rid", 7, [object()])

                self.assertEqual(locks, [7])
                self.assertEqual(unlocks, [(7, lock_params)])
                self.assertEqual(wrapper.pending_loads, {})

    def test_offload_ranges_tile_each_object_exactly(self):
        """The offload builder must still tile every object, gap-free.

        Offload sends one object's ranges in a single call and the tier writes
        them into an object of a size the connector declares separately, so a
        gap or an overlap corrupts the stored page rather than failing. This
        used to be guaranteed by accumulating layer by layer in order; the
        column-wise builder produces the same order by construction, and that
        is exactly the kind of property a rewrite can lose silently.
        """
        connector = self.make_connector()
        for pages in (1, 3):
            indices = self.transfer(pages=pages).device_indices
            for name, entry in connector.pools.items():
                per_page = _object_sizes_per_page(entry)
                plan = _PoolRangePlan(
                    name=name,
                    keys=[f"object-{index}" for index in range(pages * len(per_page))],
                    locations=entry.prepare_locations(indices),
                    entries_per_page=len(per_page),
                )
                _, sizes, offsets = connector._all_layer_ranges(plan)
                where = f"pool={name} pages={pages}"
                self.assertEqual(len(sizes), len(plan.keys), where)
                for index, (object_sizes, object_offsets) in enumerate(
                    zip(sizes, offsets)
                ):
                    cursor = 0
                    for size, offset in zip(object_sizes, object_offsets):
                        self.assertEqual(offset, cursor, f"{where} object={index}")
                        cursor += size
                    self.assertEqual(
                        cursor,
                        per_page[index % len(per_page)],
                        f"{where} object={index} does not tile its declared size",
                    )

    def test_offload_ranges_match_the_pools_own_range_api(self):
        """Offload's ranges must agree with the per-layer accumulation.

        `_all_layer_ranges` now delegates to the column-wise builder instead of
        accumulating `get_prepared_layer_range_meta` layer by layer. Same
        dependency on the pool's layout as the load path, so it gets the same
        guard: the expected value here is built the way the old code built it.
        """
        connector = self.make_connector()
        for pages in (1, 3):
            indices = self.transfer(pages=pages).device_indices
            for name, entry in connector.pools.items():
                per_page = _object_sizes_per_page(entry)
                plan = _PoolRangePlan(
                    name=name,
                    keys=[f"object-{index}" for index in range(pages * len(per_page))],
                    locations=entry.prepare_locations(indices),
                    entries_per_page=len(per_page),
                )
                expected = tuple([[] for _ in plan.keys] for _ in range(3))
                for layer in connector.pool_layers[name]:
                    meta = entry.get_prepared_layer_range_meta(plan.locations, layer)
                    if meta is None:
                        continue
                    for column, values in zip(expected, meta):
                        for index in range(len(plan.keys)):
                            column[index].extend(values[index])
                actual = connector._all_layer_ranges(plan)
                where = f"pool={name} pages={pages}"
                for label, got, want in zip(
                    ("ptrs", "sizes", "offsets"), actual, expected
                ):
                    self.assertEqual(
                        [list(entry) for entry in got], want, f"{where} {label}"
                    )

    def test_vectorized_ranges_match_the_pools_own_range_api(self):
        """The fast builder must agree with the pool's public range API.

        It reads `buffer_meta` and `_component_offsets` off the pool and expands
        them itself, which is what makes the vectorized form possible but also
        makes it depend on a layout the pool owns. Comparing against
        `get_prepared_layer_range_meta` -- accumulated the way the per-object
        builder used to -- is what keeps that dependency honest: a layout change
        upstream fails here instead of silently pointing ranges at the wrong
        offsets.
        """
        connector = self.make_connector()
        for pages in (1, 3):
            indices = self.transfer(pages=pages).device_indices
            for name, entry in connector.pools.items():
                plan = _PoolRangePlan(
                    name=name,
                    keys=[f"object-{index}" for index in range(pages)],
                    locations=entry.prepare_locations(indices),
                    entries_per_page=1,
                )
                for layers in ([0], [0, 1], [0, 1, 2], [1, 2]):
                    expected_ptrs = [[] for _ in plan.keys]
                    expected_sizes = [[] for _ in plan.keys]
                    expected_offsets = [[] for _ in plan.keys]
                    for layer in layers:
                        meta = entry.get_prepared_layer_range_meta(
                            plan.locations, layer
                        )
                        if meta is None:
                            continue
                        for index in range(len(plan.keys)):
                            expected_ptrs[index].extend(meta[0][index])
                            expected_sizes[index].extend(meta[1][index])
                            expected_offsets[index].extend(meta[2][index])
                    ptrs, sizes, offsets = connector._layer_group_ranges(plan, layers)
                    where = f"pool={plan.name} pages={pages} layers={layers}"
                    self.assertEqual(ptrs, expected_ptrs, where)
                    self.assertEqual(
                        [list(entry) for entry in sizes], expected_sizes, where
                    )
                    self.assertEqual(
                        [list(entry) for entry in offsets], expected_offsets, where
                    )

    def test_background_load_failure_reaches_consumer(self):
        connector = self.make_connector()
        # One call per layer, so the failure names a single layer.
        connector.layer_group = 1
        call_index = 0

        def fail_second_layer(keys, *args):
            nonlocal call_index
            call_index += 1
            if call_index == 3:
                return [False] * len(keys)
            return [True] * len(keys)

        self.client.batch_get_ranges_into_ptr.side_effect = fail_second_layer
        self.assertTrue(connector.load("rid", [self.transfer(pages=2)]))
        counter = connector.start_layer_wise_loading()
        connector.layer_done_counter.set_consumer(counter)
        with self.assertRaisesRegex(
            RuntimeError, "UMBP layer-wise KV load failed"
        ) as raised:
            connector.layer_done_counter.wait_until(1)
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        self.assertEqual(
            str(raised.exception.__cause__),
            "UMBP get failed for pool=kv, layer=1: success=0/2.",
        )
        connector._load_queue.join()
        self.assertEqual(connector.num_completed_loads(), 1)
        self.assertEqual(connector.pop_completed_load(), ["rid"])

    def test_accepts_any_deployment_mode_that_advertises_ranged_io(self):
        """Admit on the capability, not on which mode happens to have it.

        The gate used to be `mode == StandaloneProcess`, which was true of the
        only client that implemented ranged I/O at the time. mori now implements
        it in the in-process client too, so a mode test would reject a client
        that can do the job. Pinning every mode here keeps the next person from
        reintroducing the proxy.
        """
        for mode in (
            _DeploymentMode.Local,
            _DeploymentMode.StandaloneProcess,
            _DeploymentMode.Distributed,
        ):
            with self.subTest(mode=mode):
                self.client.get_deployment_mode.return_value = mode
                connector = self.make_connector()
                self.assertEqual(connector.deployment_mode, mode)
                # Only StandaloneProcess deregisters client-wide at close; the
                # in-process client's DeregisterMemory is a base-class no-op.
                self.assertEqual(
                    connector._standalone_process_mode,
                    mode == _DeploymentMode.StandaloneProcess,
                )
