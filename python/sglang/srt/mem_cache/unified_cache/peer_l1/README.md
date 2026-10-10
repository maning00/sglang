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
| `pin_cap` | 2048 | Maximum remote page references held on one owner |
| `min_hit_pages` | 1 | Minimum prefix worth restoring |
| `wait_timeout_ms` | 200 | Bounded capacity wait; 0 disables waiting |
| `verify_bytes` | false | Compare every copied row before releasing its source pin |
| `check_hooks` | false | Check allocator frees and reject collectives between pin and copy submission |

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
slot reuse use the latter. Flush first drains completed loads and their tree
locks, including when the scheduler is paused. It then checks local activity,
atomically freezes new remote reservations, and withdraws the directory before
clearing the pool. Busy flushes report their reason and return immediately.
Weight updates can retry for `wait_timeout_ms` only after every worker confirms
it is paused and idle; this confirmation precedes loading weights so failed
loads do not skip a collective. Expiry still rejects the stale cache.

IPC mappings live for the engine's lifetime. Close is collective across the
worker group so producers outlive consumers. Recovery from individual worker
crashes is outside this PoC's scope.

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
