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
| `stream_priority` | -1 | Copy stream priority. The default high priority keeps the copy stream off the hardware queues that normal-priority streams share; at 0 its wait on the forward stream can stall the scheduler stream's syncs until that forward ends (measured on MI355 with `GPU_MAX_HW_QUEUES=5`) |
| `pin_cap` | 16384 | Maximum remote page references held on one owner |
| `min_hit_pages` | 1 | Minimum prefix worth restoring |
| `wait_timeout_ms` | 200 | Bounded capacity wait; 0 disables waiting |
| `verify_bytes` | false | Compare every copied row before releasing its source pin |
| `check_hooks` | false | Check allocator frees and reject collectives between pin and copy submission |
| `stitch_stats` | false | Count the prefix pages restorable if runs could come from different peers (`lookup_stitched_pages` against `lookup_single_peer_pages`) |

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

This requires the same UMBP standalone setup as the `mori` backend. Each request
uses one source: peer HBM when it covers the selected prefix and can reserve it,
otherwise UMBP when it covers the whole prefix. A failed peer reservation with
only a shorter UMBP prefix falls back to recomputation. Writes always go to UMBP.
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
