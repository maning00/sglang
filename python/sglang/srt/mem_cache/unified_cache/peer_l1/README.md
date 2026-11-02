# Peer L1 linker

An experimental, single-node ROCm backend for restoring DeepSeek V4 unified KV
directly from another DP worker's HBM. It supports 2–8 workers with attention
TP=1, CP=1 and PP=1. Every worker must see all participating GPUs.

Add these options to a DEP8 launch:

```bash
--enable-unified-cache-external-linker \
--unified-cache-external-linker-backend peer_l1 \
--hicache-storage-backend-extra-config '{"blocks":32,"layer_group":8}'
```

Use page size 256 and the `unified_kv_triton` attention layout. For the fp8
two-pool layout, set `SGLANG_DSV4_UNIFIED_KV_FP8=1` as well as
`--kv-cache-dtype fp8_e4m3`. The actual pool views determine copy geometry.
Memory saver, movable unified-memory allocations, and expandable/async
PyTorch allocators are unsupported. The backend creates no L2 backup.

The extra-config object accepts:

| Option | Default | Meaning |
|---|---:|---|
| `blocks` | 32 | Maximum blocks in a fused copy launch |
| `layer_group` | 8 | Logical layers copied before recording their ready events |
| `copy_engine` | `kernel` | `sdma` moves restored rows with mori anvil SDMA queues (one per peer, DMA engines instead of CUs; needs `hipStreamWaitValue64`); `kernel` uses the copy kernel. Byte verification always uses the kernel |
| `sdma_channels` | 2 | SDMA queues per peer; a peer's jobs in a batch are split evenly across them |
| `stream_priority` | -1 | Copy stream priority. The default high priority keeps the copy stream off the hardware queues that normal-priority streams share; at 0 its wait on the forward stream can stall the scheduler stream's syncs until that forward ends (measured on MI355 with `GPU_MAX_HW_QUEUES=5`) |
| `pin_cap` | 16384 | Maximum remote page references held on one owner |
| `min_hit_pages` | 1 | Minimum prefix worth restoring |
| `wait_timeout_ms` | 200 | Bounded capacity wait; 0 disables waiting |
| `verify_bytes` | false | Compare every copied row before releasing its source pin |
| `check_hooks` | false | Check allocator frees and reject collectives between pin and copy submission |
| `stitch_stats` | false | Count the prefix pages restorable if runs could come from different peers (`lookup_stitched_pages` against `lookup_single_peer_pages`) |
| `l2_write` | `write_through` | With an L2 (`PeerL1PlusL2Linker`): `write_through` keeps the wrapper's L2 write-through; `last_replica` writes a page to L2 only when its last HBM copy on the node is evicted, holding the row until the write lands |
| `l2_write_max_inflight_pages` | 15% of page rows | `last_replica` only: a write that would put more pages in flight is refused (the page is freed without an L2 copy) |
| `l2_write_timeout_s` | 60 | `last_replica` only: once a write is in flight this long, new writes are refused until it lands; the late write keeps its rows |
| `l2_stitch` | true | `last_replica` only: UMBP continues a peer prefix (the peer's pages, then UMBP's from where they end), since UMBP lacks the pages still in some HBM |

SDMA detects MoRI Anvil's `nodeForHipDevice()` at compile time: MoRI #623
(`7cba7e1975b6c77e7093b726e5b9e93cbd3c347b`) uses KFD node IDs; older
versions keep their HIP-ordinal `connect()` / `getSdmaQueue()` API.

Both verification options add work and should be disabled for performance
measurements. A hard cap truncates the advertised prefix before admission;
a reservation race returns a cache miss through the existing load-back flow.
The collective check uses the current thread's Python profile hook only while
reservations are unsubmitted, including calls through previously imported
aliases. It cannot run alongside another Python profiler on that thread.

## Ownership and completion

The shared directory contains atomic row identities and pin counts. Retirement
changes the generation before allocator reuse. Pinned rows leave the tree but
stay out of the allocator until the reader releases them. Allocation-driven
eviction checks actual available space and reclaims completed holds between
passes. Genuine shortages retain the caller's existing retract/ABORT behavior.

Tree publications wait for both the scheduler's index writes and the forward's
KV writes. The consumer queues layer events for forward and a separate batch
event after verification and the native unpin callback. Batch cleanup and event
slot reuse use the latter. Loads stay request-private until the forward finishes;
checkpoint insertion publishes them through the same events as computed KV.
A failed batch is aborted by the scheduler without publishing its pages. Queued
loads can be cancelled before the request frees its private slots.

Flush checks local activity, atomically freezes new remote reservations, then
withdraws the directory before clearing the pool. Completed offloads are reaped
when flushing a paused engine; pending loads, offloads or remote pins return busy.
Weight updates use the upstream failure response when a flush is refused.

IPC mappings live for the engine's lifetime. Close is collective across the
worker group so producers outlive consumers. Recovery from individual worker
crashes is outside this PoC's scope.

## Peer L1 plus UMBP

`--unified-cache-external-linker-backend mori_peer_l1` combines peer HBM reads
with the existing UMBP write-through store. UMBP configuration stays at the top
level and peer options go in `peer_l1`, for example:

```json
{"ssd_enabled":false,"peer_l1":{"blocks":32,"layer_group":8,"pin_cap":16384}}
```

This requires the same UMBP standalone setup as the `mori` backend. With
`write_through`, each request uses one source: peer HBM when it covers the
selected prefix and can reserve it, otherwise UMBP when it covers the whole
prefix. A failed peer reservation with only a shorter UMBP prefix falls back to
recomputation. With `last_replica` and `l2_stitch`, UMBP is probed only past the
peer prefix, and a request may take its head from the peer and the rest from
UMBP (`stitched_loads`); if the peer reservation then fails, it recomputes.
Writes always go to UMBP.
Restores stay private until their whole load succeeds and checkpoint inserts them.
An L1 or L2 failure aborts every restoring request in that batch.
Flush checks both loaders and UMBP offloads; it clears device directories while
preserving the UMBP tier. The periodic peer stats distinguish `l1_loaded_pages`
and `l2_loaded_pages` for successful batches, and `failed_composite_load_batches`.
Restored tails follow upstream write-through behavior when checkpoint inserts them,
including a repeated write for a tail loaded from UMBP.

## Verification

The focused tests are `test_peer_l1_directory.py` and `test_peer_l1_cache.py` in
`test/registered/unit/mem_cache/`. The second exercises Python and Rust tree
hooks, compensation/reclaim through the real allocator, and HIP callback/event
ordering. It needs a supported Rust/PyTorch environment to cover the Rust cases.

`test/manual/peer_l1/bench_copy.py` exercises real IPC mappings with the unified
fp8 Pro row geometry. It validates a warmup copy against both the remote bytes
and an independent row oracle, then times the production copy path. Its
`--skip-copy` negative control must fail. `stress_directory.cpp` runs the native
protocol across forked processes without a GPU.

With a real-weight DEP8 server and both verification options enabled, run
`test/manual/peer_l1/restore_smoke.py`. It checks cross-rank restores, local reuse,
concurrent readers, and cold attribution after flush. Unified KV output text
can differ across cache paths; the byte checks and cache-source attribution
establish which data path was exercised.
Pass `--combined` for `mori_peer_l1`: after flush it requires an L2 hit, followed
by another peer read of that restored prefix. Check both loaded-page counters
in the server log to confirm the selected sources.

## Indexer migration

`peer_l1.indexer` selects cumulative migration stages. For the standalone
`peer_l1` backend, put `indexer` directly in the extra-config object. Use the
same stage, endpoint, and deployment on all prefill and decode ranks:

```json
{"peer_l1":{"l2_backend":"decode_lend+umbp","l2_write":"last_replica","indexer":{"stage":4,"endpoint":"127.0.0.1:50051","deployment":"example"}}}
```

| Stage | Discovery and ownership |
|---|---|
| 0 (default) | Original shm discovery and pins; directory names are unlinked after every rank maps them |
| 1 | Decode lending reports to Indexer and grants pages by complete key with ACQUIRE |
| 2 | Peer L1 also reports to Indexer; the selected candidate's shm provides rows and generations |
| 3 | Native Unix socket owner RPCs grant pins; HIP callbacks enqueue reliable batched cleanup; shm is retained for F3 only |
| 4 | Batched authoritative CONTAINS replaces F3 shm reads; peer directories use private anonymous memory |

Stage 0 restores the original path for comparisons. These are engine startup
options; changing a stage requires restarting all ranks. Stage 1 and later do
not create the decode lending shm directory. Stage 4 creates neither directory.
The deprecated multi-peer stitching diagnostic is disabled at stages 3 and 4,
where remote shm probes are no longer permitted.

Run `kv-indexer-server` separately, using the wire protocol from SGLang upstream
commit `2c362af4cc`. The protobuf source is `experimental/sgl-router/sgl-kv-indexer/proto/kv_indexer.proto`;
regenerate the Python binding with `python -m grpc_tools.protoc
-I experimental/sgl-router/sgl-kv-indexer/proto --python_out=python/sglang/srt/grpc
experimental/sgl-router/sgl-kv-indexer/proto/kv_indexer.proto`. Clients report directly by gRPC, without
a bridge. Worker addresses are empty so these entries do not become Router
prefix destinations. A deployment must have a unique name and consistent hash,
page size, model, and KV layout. Discovery returns known peers only and never
authorizes access to a row.

`timeout_s` defaults to 0.01 for discovery and peer L1 control, `cooldown_s` to
1.0, and `recovery_s` to 600. Reports have a minimum one-second RPC deadline.
Each owner restores soft state with CLEAR, a snapshot, then buffered deltas on
failure or reconnection, and periodically with staggered phases. The publisher
runs outside the scheduler thread. A request shares one discovery result between
peer L1 and lending; requests over 16,384 pages require multiple bounded RPCs.
Indexer failures use local shm probes before stage 3 and parallel LOOKUP from
stage 3 onwards; lending falls back to DRAM. Namespaced requests skip Indexer,
using local discovery and the DRAM tier. Partial peer L1 grants are released and
remain misses; they do not introduce a different L2 stitching boundary.

The 60-second stats include discovery diagnostics. Counters are cumulative;
take per-rank window deltas before summing them:

- `peer_l1`: `indexer_query_{ok,timeout,error,cooldown_skip}` count logical
  queries; `indexer_lookup_rpcs` counts actual RPC chunks, including failures.
  Cached placements make no query. `indexer_rpc_ms_p50/p99/max` cover RPCs
  since the previous log, including failures. Percentiles use the most recent
  4,096 samples; `indexer_rpc_latency_samples/count` show truncation. Max covers
  the full interval; `indexer_rpc_ms_total` is cumulative. Do not difference
  percentiles or average them across ranks.
- `decode lend L2`: `lookup_lend_pages` counts pages found on connected lenders
  before DRAM probing. `lookup_pages` and `lookup_restorable_pages` provide its
  denominator and the eventual contiguous prefix. Repeated lookups count again.
  `lookup_discovery_unavailable_calls/pages` include cached failed discovery.
- `KV placement publisher worker=...`: emitted for both prefill and decode
  owners. `recovery_started/ok/failed`, `recovery_requested_*`, and
  `recovery_ms_total/max` describe attempts and triggers. `recovery_gap_ms_total/max`
  bounds CLEAR-to-snapshot gaps, including failed attempts and retry cooldown;
  `recovery_active_ms` shows an unfinished gap. These are time bounds, not lost
  page counts. `live_records/pending_records` are gauges; `report_rpcs` and
  `report_ms_total/max` measure publication RPCs.
- `peer_l1`: `f3_contains_pages/elsewhere_pages` count retired pages examined
  and pages for which F3 skips writing because another owner holds a copy.
  `f3_contains_ms_total` measures the query cost. A failed CONTAINS still writes
  conservatively and increments the existing `contains_failures` counter.

Peer L1 never releases source pins on a control disconnect or a timer. Old
outstanding grants stop new grants to that reader after 30 seconds, but stay
pinned until RELEASE or whole-engine teardown. CANCEL and RELEASE are retried
until acknowledgment; an owner epoch change makes old cleanup obsolete. A
per-reader completion watermark rejects late cancelled ACQUIRE requests without
retaining a tombstone per request. Flush retains busy/retry behavior. Lending
keeps its original expiry policy and checks elapsed read time before using data.

CPU protocol checks are in `test_peer_l1_indexer.py`, alongside the existing
peer and lending tests. They do not establish HIP callback ordering, RDMA byte
correctness, or end-to-end performance. Validate both copy engines, flush and
cancellation, owner disconnect during copying, and the approved throughput,
TTFT, hit-rate and reserve-failure budgets on matching hardware before rollout.

An owner that stays dead leaves cleanup retrying until collective engine shutdown;
individual-owner recovery remains outside this PoC. Placement hints are shared
by peer and lending lookup, bounded to 128 request IDs and refreshed after 100 ms.
With more than 128 waiting request IDs, eviction can reduce reuse to one RPC per
match. Stage 4 F3 splits writes by chain; the additional offloads can fill
TieredL2's 256-task queue and count toward `write_refused_pages`. This retains
the per-node granularity of stages 0–3; tune only after measurement.
