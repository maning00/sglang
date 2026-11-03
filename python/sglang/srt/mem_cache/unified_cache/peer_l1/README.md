# Peer L1 linker

An experimental, single-node ROCm backend for restoring DeepSeek V4 unified KV
directly from another DP worker's HBM. It supports 2–8 workers with attention
TP=1, CP=1 and PP=1. Every worker must see all participating GPUs.

Add these options to a DEP8 launch:

```bash
--enable-unified-cache-external-linker \
--unified-cache-external-linker-backend peer_l1 \
--hicache-storage-backend-extra-config '{"blocks":32,"layer_group":8,"indexer":{"endpoint":"127.0.0.1:50051","deployment":"example"}}'
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

The owner keeps row identities and pin counts in private memory. Retirement
changes the generation before allocator reuse. Pinned rows leave the tree but
stay out of the allocator until the reader releases them. Allocation-driven
eviction checks actual available space and reclaims completed holds between
passes. Genuine shortages retain the caller's existing retract/ABORT behavior.

Tree publications wait for both the scheduler's index writes and the forward's
KV writes. The consumer queues layer events for forward and a separate batch
event after verification and the native cleanup callback. Batch cleanup and event
slot reuse use the latter. Loads stay request-private until the forward finishes;
checkpoint insertion publishes them through the same events as computed KV.
Failed loads are discarded before publication. A missing lending page that
DRAM cannot restore fails only its request; other requests in the batch finish
restoring. Failure masks are reduced per request across attention ranks. The
existing prefill retry path recomputes the failed request only after PD source
I/O quiesces. Unknown batch I/O failures remain conservative. Queued loads can
be cancelled before the request frees its private slots.

Flush checks local activity, atomically freezes new remote reservations, then
withdraws owner records before clearing the pool. Completed offloads are reaped
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
{"ssd_enabled":false,"peer_l1":{"blocks":32,"layer_group":8,"pin_cap":16384,"indexer":{"endpoint":"127.0.0.1:50051","deployment":"example"}}}
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
A batch-level L1 or L2 failure aborts requests using that source; lending misses
can identify and fail individual requests.
Flush checks both loaders and UMBP offloads; it clears owner records while
preserving the UMBP tier. The periodic peer stats distinguish `l1_loaded_pages`
and `l2_loaded_pages` for successful requests, and
`failed_composite_load_batches/requests`. Lending also reports
`failed_restore_requests` for missing-page restores.
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
`--skip-copy` negative control must fail. `stress_owner.py` exercises concurrent
retirement across owner processes without a GPU.

With a real-weight DEP8 server and both verification options enabled, run
`test/manual/peer_l1/restore_smoke.py`. It checks cross-rank restores, local reuse,
concurrent readers, and cold attribution after flush. Unified KV output text
can differ across cache paths; the byte checks and cache-source attribution
establish which data path was exercised.
Pass `--combined` for `mori_peer_l1`: after flush it requires an L2 hit, followed
by another peer read of that restored prefix. Check both loaded-page counters
in the server log to confirm the selected sources.

## Indexer discovery and owner services

Peer L1 and decode lending use KV Indexer for discovery. Configure
`peer_l1.indexer` with the same endpoint and deployment on all prefill and decode
ranks. For the standalone `peer_l1` backend, put `indexer` directly in the
extra-config object. Both fields are required; missing values fail at startup.

```json
{"peer_l1":{"l2_backend":"decode_lend+umbp","l2_write":"last_replica","indexer":{"endpoint":"127.0.0.1:50051","deployment":"example"}}}
```

Ownership remains authoritative on each owner. Peer L1 uses native Unix socket
ACQUIRE, cancellation, RELEASE and LOOKUP; HIP callbacks enqueue reliable batched
cleanup after device reads finish. F3 queries owners with batched CONTAINS before
writing last replicas. Decode lending grants pages by complete key with ACQUIRE.
Both owners keep their state in private memory; neither creates or maps a
`/dev/shm` hint directory. The original shm implementation remains at `7ad884811e`.

Run `kv-indexer-server` separately, using the wire protocol from SGLang upstream
commit `2c362af4cc`. The protobuf source is `experimental/sgl-router/sgl-kv-indexer/proto/kv_indexer.proto`;
regenerate the Python binding with `python -m grpc_tools.protoc
-I experimental/sgl-router/sgl-kv-indexer/proto --python_out=python/sglang/srt/grpc
experimental/sgl-router/sgl-kv-indexer/proto/kv_indexer.proto`. Clients report directly by gRPC, without
a bridge. Worker addresses are empty so these entries do not become Router
prefix destinations. A deployment must have a unique name and consistent hash,
page size, model, and KV layout. Discovery returns known peers only and never
authorizes access to a row.

`timeout_s` defaults to 0.05 for discovery and peer L1 control, `cooldown_s` to
0.1, and `recovery_s` to 600. Reports have a minimum one-second RPC deadline.
The discovery defaults were selected from the T2b 1P1D runs on n09-29 + n09-33,
with Indexer on the decode node. Indexer runs 4144 / 4155 (`95f73e4907`) with
the former 0.01 / 1.0 defaults averaged 3.6% less throughput and 0.1 percentage
point less cache hit rate than runs 4139 / 4156 using the original shm path
from `7ad884811e`. Run 4158 (`6157478408`) used 0.05 / 0.1: throughput was 1,243,024 versus the shm baseline mean of 1,242,820
tok/s (+0.0%), TTFT p50 / p95 changed by +3.6% / -2.4%, and cache hit rate was
96.14% versus 96.15%, with no errors or recomputations. A discovery timeout
disables lending discovery during cooldown; peer L1 has a local fallback.
The longer deadline and shorter cooldown reduce this loss of lending hits.
Evidence: UMBP indexer migration evidence 55, `runs/cap_compare_all_pairs_4158.txt`
and `runs/indexer_diag_4158.txt`.

Each owner restores soft state with CLEAR, a snapshot, then buffered deltas at
startup and periodically with staggered phases. READY alone does not prove a
server restart: the protocol has no server incarnation ID, so lost state after
a restart is repaired by the periodic fallback (600 s by default).
Timeout, UNAVAILABLE, and RESOURCE_EXHAUSTED retry the same batch and sequence
up to five times, preserving order with newer deltas. Backoff doubles from
`cooldown_s`, is capped at 30 s, and is staggered by worker identity. A partially
sent snapshot resumes at its failed batch without repeating CLEAR.
After retry exhaustion or a non-transient rejection, publication pauses until
the next periodic recovery. Pending deltas coalesce by key; above 65,536 keys,
they are discarded until periodic recovery rebuilds from current ownership.
This bounds outage backlog while retaining authoritative local records. Discovery
can remain stale during an outage; ACQUIRE still checks complete keys and pins.
The publisher runs outside the scheduler thread. A request shares one discovery
result between peer L1 and lending; requests over 16,384 pages require multiple
bounded RPCs.
Indexer failures use parallel LOOKUP from node-local owners; lending falls back
to DRAM. Namespaced requests skip Indexer, using local discovery and the DRAM
tier. A peer must grant the complete requested prefix; refusal remains a miss
and does not introduce a different L2 stitching boundary.

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
  `report_ms_total/max` measure publication RPCs. `report_retries/backoff_ms`,
  `report_pauses`, `report_pause_remaining_ms`, and `pending_overflows/overflow`
  distinguish retry, pause, and backlog saturation. `reconnects` counts READY
  notifications, independently of recovery attempts.
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
match. F3 splits writes by chain; these offloads can fill TieredL2's 256-task
queue and count toward `write_refused_pages`.

## Reader contracts and owner transport

`PeerSession` binds an owner channel to its data mover and imported memory.
`Grant` records the owner epoch, source rows and authorization start time;
`lease_s=None` keeps native source pins until completion cleanup. Lend grants
retain their finite lease and are acquired by the I/O thread in 64-page chunks.
The two owner adapters preserve their existing cancellation and release queues.
`CopyBatch` and `LendTransfer` expose resolved layers, failed requests and
quiescence separately. Layer resolution still precedes the post-forward failure
verdict; destination reuse requires completed device accesses.

Native owners accept abstract Unix endpoints (bare names or `unix:name`) and
`tcp://IP:port` (`tcp://[IPv6]:port` for IPv6). A zero TCP port binds an ephemeral
port, returned by `OwnerService.endpoint`. Endpoints use numeric IPs so DNS cannot
block a bounded owner RPC. TCP uses the same frames and acknowledged cleanup as
Unix, with TCP_NODELAY on both ends. TCP reuses one connection per reader thread and endpoint, discarding it on errors
or an epoch mismatch. Unix retains one connection per RPC. TCP uses a separate
nonblocking service loop with bounded per-connection work; an incomplete frame or
blocked reply cannot hold up other sockets. Both listeners share the owner lock.
`OwnerService.listen_tcp()` can add a TCP listener to an existing Unix owner.
Production same-host sessions continue to use Unix; TCP forcing is test-only.

```bash
PYTHONPATH=python python test/manual/peer_l1/stress_owner.py --tcp
PYTHONPATH=python python test/manual/peer_l1/bench_owner_rpc.py --readers 1
PYTHONPATH=python python test/manual/peer_l1/bench_owner_rpc.py --readers 7
```

The benchmark reports round-trip latency (TCP setup is amortized), and aggregate
RPC/s for repeated ACQUIRE_EXACT / RELEASE / LOOKUP / CONTAINS cycles. RELEASE
is measured through its acknowledgement, not just cleanup-queue insertion. It is
a loopback control-plane measurement, not an inference or RDMA throughput result.

## Cross-instance discovery

The instance API serves a cached, read-only `GET /peer_l1/owners` manifest.
Python and Rust frontends use the same schema. The startup handshake carries the
complete instance snapshot, including all ranks on other nodes; multi-tokenizer
workers receive the same snapshot. Before publication the route returns 503.
Background resolvers fetch configured instance URLs and validate each owner with
HELLO. Identity, epoch and `kv_compatibility_id` must match before a session is
usable. HTTP is never called from an inference request.

The `peer_l1.domain` object accepts `enabled`, `instance_id`, `seeds` (instance API
URLs) and `kv_compatibility_id`. When enabled all three identity/discovery fields
are required. Otherwise the instance uses rank 0's hostname, gathered at startup;
no new configuration is required. Two instances on one host must not reuse a
deployment without explicit instance identities. Worker IDs are
`deployment/instance_id/role/dpN`; they contain no addresses. A restarted logical
instance keeps its ID; replacement with a new ID requires the deployment to CLEAR
old worker records explicitly. Upgrading from the older worker-ID format also
requires clearing the old records or using fresh Indexer state.

Owner identity, KV compatibility, membership and capability changes require a
restart in this version. This includes weight or adapter-generation changes
that would change compatibility. Decoder lending keeps its existing PD
registration; its HELLO validates a configured compatibility ID too. Set the same
`domain.kv_compatibility_id` on the lending decoder when the prefill uses one.

## RDMA failure quarantine

Once a MoRI native submission call has been entered, failure is conservatively
classified as possibly posted. Preflight errors and ACQUIRE misses do not poison
pages. Status objects are allocated and retained before invoking the native batch,
including when that call raises after a partial post. Terminal failure does not
mean DMA has stopped.

Failed read destinations are excluded by the full paged allocator's release and
clear paths. They cannot be used for DRAM fallback. Failed lending writes issue a
reliable, epoch-bound POISON for the original allocation sequence and never ABORT;
if notification is lost, WRITING still prevents reuse. Poisoned decode pages count
against the unreclaimable budget and cannot be reclaimed or cleared back into the
allocator. Demotion staging skips poisoned slots. Native statuses, registrations
and allocations survive runtime context resets; affected PD transports cannot be
unbound without restarting the process.

The 60-second lending stats report `quarantined_read_pages`,
`quarantined_write_pages`, and `quarantined_staging_pages`. Peer L1 stats include
`rdma_quarantined_pages` and `rdma_quarantined_retained_pages`; the latter counts
pages already removed from request ownership. Decode `lend pages` includes
`quarantined`. Quarantine emits an error log and has no automatic release timer.

## Cross-domain reads

Peer identities are global; IPC kernels use at most eight backend-local slots.
GPU UUIDs, host identity and mutual device visibility select IPC peers. Other
intra-instance peers use TCP and RDMA, as do discovered cross-instance owners.
The source pool is registered once with a dedicated MoRI engine, independent of
PD; registrations and fixed allocations are retained until process exit because
remote-access revocation cannot otherwise be established.

RDMA sessions use finite native grants (30 seconds by default). Reserve still
acquires the exact prefix; reads are submitted in 64-page chunks and each check
includes time elapsed since ACQUIRE. Replay does not renew a grant. A retired
session cannot authorize or submit new reads, while an in-flight read keeps its
original epoch and registration. Failed posted reads use the same quarantine as
lending and fail only the affected requests after forward.

Sources are ordered local, fabric, RDMA. A farther source must add at least
`domain.min_gain_pages` (default 1) beyond an existing nearer hit; this is a
functional default, not a cross-machine performance tuning result. F3 CONTAINS
is restricted to the same instance, effective `l2_domain` and preservation
policy. The L2 domain defaults to hostname; an explicit domain must represent
an actually shared backing keyspace. Foreign copies never suppress this
instance's F3 writes. Indexer fallback remains within the instance restart group.

Example for one of two instances (use a distinct ID and reciprocal seed on the
other; hostnames here are examples):

```json
{"domain":{"enabled":true,"instance_id":"prefill-a","kv_compatibility_id":"weights-layout-v1","seeds":["http://prefill-b:6100"]}}
```

Merge `domain` into the existing peer L1 config; the Indexer config remains
required. Cross-tray fabric mapping is not implemented; unsupported IPC pairs
use RDMA. Run `test/manual/peer_l1/cross_instance_check.py --indexer HOST:PORT`
for the model-free read/restart/quarantine check. `--same-instance` checks one
instance and `--isolated-gpus` limits each child to its own GPU. Fault injection
uses native statuses/submissions and gated GPU writes, without changing NICs.
