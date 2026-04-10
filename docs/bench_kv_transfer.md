# Mori KV Transfer Benchmark Tool

Standalone benchmark for the mori-io RDMA/XGMI transfer engine, without model loading.

## Overview

`bench_kv_transfer` creates lightweight mock prefill/decode servers that perform **real GPU memory transfers** via the mori engine. It plugs directly into the existing Rust router and `bench_serving` — no modifications needed to either.

**What it does:**
- Allocates real GPU KV cache buffers (configurable geometry)
- Starts mori IOEngine with RDMA/XGMI backends
- Handles the full PD disaggregation lifecycle (bootstrap, handshake, KV transfer)
- Exposes `/generate` endpoint compatible with the Rust router
- Supports SSE streaming with configurable output length

**What it skips:**
- Model loading (saves 10+ minutes)
- Tokenization
- Inference compute

## Architecture

```
bench_serving ──> Rust Router ──> POST /generate
                       |                    |
                       v                    v
              MockPrefillServer      MockDecodeServer
              (GPU 0..N-1)           (GPU N..2N-1 or remote)
                       |                    |
              MoriKVManager(PREFILL) MoriKVManager(DECODE)
              IOEngine + RDMA/XGMI  IOEngine + RDMA/XGMI
                       |                    |
                       +--- real mori KV ---+
                            transfer
```

## Quick Start

### Single-Machine TP=1

```bash
HOST_IP=$(hostname -I | awk '{print $1}')

# Terminal 1: Prefill (GPU 0)
python -m sglang.bench_kv_transfer --mode prefill --port 30100 \
    --tp-size 1 --base-gpu-id 0 --advertise-host $HOST_IP \
    --num-layers 2 --head-dim 128 --dtype fp8_e4m3 --kv-cache-type mla \
    --max-concurrency 32 --max-input-len 1024 --max-output-len 1

# Terminal 2: Decode (GPU 1)
python -m sglang.bench_kv_transfer --mode decode --port 30200 \
    --tp-size 1 --base-gpu-id 1 --advertise-host $HOST_IP \
    --num-layers 2 --head-dim 128 --dtype fp8_e4m3 --kv-cache-type mla \
    --max-concurrency 32 --max-input-len 1024 --max-output-len 1

# Terminal 3: Router
python -m sglang_router.launch_router --pd-disaggregation \
    --host 0.0.0.0 --port 30000 --policy random \
    --prefill http://$HOST_IP:30100 8998 \
    --decode http://$HOST_IP:30200

# Terminal 4: Benchmark
python -m sglang.bench_serving --host $HOST_IP --port 30000 \
    --model mock-kv-bench --tokenizer <any-hf-tokenizer> \
    --dataset-name random --random-input-len 1024 --random-output-len 1 \
    --num-prompts 64 --max-concurrency 8
```

### Single-Machine TP=4 (XGMI)

```bash
HOST_IP=$(hostname -I | awk '{print $1}')
IB="mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9"

# Prefill (GPU 0-3)
python -m sglang.bench_kv_transfer --mode prefill --port 30100 \
    --tp-size 4 --base-gpu-id 0 --advertise-host $HOST_IP \
    --num-layers 2 --head-dim 128 --dtype fp8_e4m3 --kv-cache-type mla \
    --max-concurrency 64 --max-input-len 1024 --max-output-len 1 \
    --ib-device "$IB"

# Decode (GPU 4-7)
python -m sglang.bench_kv_transfer --mode decode --port 30200 \
    --tp-size 4 --base-gpu-id 4 --advertise-host $HOST_IP \
    --num-layers 2 --head-dim 128 --dtype fp8_e4m3 --kv-cache-type mla \
    --max-concurrency 64 --max-input-len 1024 --max-output-len 1 \
    --ib-device "$IB"

# Router + Benchmark (same as TP=1)
```

### Cross-Machine TP=8 (RDMA)

```bash
# Machine A (prefill): 10.1.1.1, GPU 0-7
# Machine B (decode):  10.1.1.2, GPU 0-7

IB="mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9"

# On Machine A:
python -m sglang.bench_kv_transfer --mode prefill --port 30100 \
    --tp-size 8 --base-gpu-id 0 --advertise-host 10.1.1.1 \
    --num-layers 2 --head-dim 128 --dtype fp8_e4m3 --kv-cache-type mla \
    --max-concurrency 256 --max-input-len 1024 --max-output-len 1 \
    --ib-device "$IB"

# On Machine B:
python -m sglang.bench_kv_transfer --mode decode --port 30200 \
    --tp-size 8 --base-gpu-id 0 --advertise-host 10.1.1.2 \
    --num-layers 2 --head-dim 128 --dtype fp8_e4m3 --kv-cache-type mla \
    --max-concurrency 256 --max-input-len 1024 --max-output-len 1 \
    --ib-device "$IB"

# Router (on either machine):
python -m sglang_router.launch_router --pd-disaggregation \
    --host 0.0.0.0 --port 30000 --policy random \
    --prefill http://10.1.1.1:30100 8998 \
    --decode http://10.1.1.2:30200 \
    --max-concurrent-requests 512 --queue-size 4096 \
    --cb-failure-threshold 100

# Benchmark:
python -m sglang.bench_serving --host <router-ip> --port 30000 \
    --model mock-kv-bench --tokenizer <any-hf-tokenizer> \
    --dataset-name random --random-input-len 1024 --random-output-len 1 \
    --num-prompts 2048 --max-concurrency 256
```

## CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | (required) | `prefill` or `decode` |
| `--host` | `0.0.0.0` | HTTP bind address |
| `--advertise-host` | auto | Routable IP for mori/ZMQ. Auto-detected if omitted. Set explicitly for cross-machine. |
| `--port` | (required) | HTTP port |
| `--tp-size` | `1` | Tensor parallel size |
| `--base-gpu-id` | `0` | First GPU index (rank i uses GPU `base-gpu-id + i`) |
| `--num-layers` | `8` | Number of KV cache layers |
| `--kv-head-num` | `1` | Number of KV heads per layer (MHA only) |
| `--head-dim` | `128` | Head dimension |
| `--kv-cache-type` | `mla` | `mla` (1 combined KV per layer) or `mha` (separate K+V) |
| `--page-size` | `1` | Tokens per page |
| `--dtype` | `fp8_e4m3` | KV cache dtype: `fp8_e4m3`, `fp8_e5m2`, `bf16`, `fp16`, `fp32` |
| `--bootstrap-port` | `8998` | Bootstrap server port (prefill side) |
| `--ib-device` | `""` | Comma-separated IB devices for RDMA (e.g. `mlx5_0,mlx5_2,...`) |
| `--model-name` | `mock-kv-bench` | Model name reported to router |
| `--max-concurrency` | `1024` | Max concurrent requests (for pool size auto-calc and aux allocation) |
| `--max-input-len` | `8192` | Max input tokens per request (for pool size auto-calc) |
| `--max-output-len` | `1024` | Max output tokens per request (for pool size auto-calc) |
| `--pool-size` | `0` (auto) | Total KV token slots. `0` = `max_concurrency * (input + output)` |
| `--aux-size` | `512` | Aux metadata buffer size per slot (bytes) |
| `--log-level` | `INFO` | Logging level |

## KV Cache Geometry

The benchmark does not load a model — you configure the KV cache geometry directly. This allows testing any model's transfer characteristics without its compute.

### Memory per token per GPU

| Config | Formula | Bytes/token |
|--------|---------|-------------|
| DeepSeek-R1 MLA, FP8 | 61 layers × 576 dim × 1 | 35,136 |
| DeepSeek-R1 MLA, BF16 | 61 layers × 576 dim × 2 | 70,272 |
| Llama-70B MHA, FP8 | 80 layers × 2(K+V) × 8 heads × 128 dim × 1 | 163,840 |
| Lightweight bench, FP8 | 8 layers × 128 dim × 1 | 1,024 |
| Minimal bench, FP8 | 2 layers × 128 dim × 1 | 256 |

### Pool Size

`pool_size` determines the total KV token slots in GPU memory. Auto-calculated as:

```
pool_size = max_concurrency × ceil((max_input_len + max_output_len) / page_size) × page_size
```

**GPU memory** = `pool_size × bytes_per_token`

Example: `--max-concurrency 256 --max-input-len 1024 --max-output-len 1 --num-layers 2 --head-dim 128 --dtype fp8_e4m3`
- pool_size = 256 × 1025 = 262,400 tokens
- bytes/token = 2 × 128 × 1 = 256 bytes
- GPU memory = 262,400 × 256 = **67 MB/GPU**

For high-concurrency targets (c=1024+), use lightweight geometry to fit in memory:

```bash
# 1024 concurrent × 8K input → ~8.4M tokens × 256 bytes = 2.1 GB/GPU
--num-layers 2 --head-dim 128 --dtype fp8_e4m3 --kv-cache-type mla \
--max-concurrency 1024 --max-input-len 8192 --max-output-len 1
```

## Interpreting Results

With `--random-output-len 1`, `bench_serving` reports:
- **TTFT** = bootstrap handshake + real mori KV transfer time
- **Throughput** = requests/s the mori engine can sustain

With `--random-output-len N` (N > 1), additionally:
- **ITL** = simulated decode latency (~5ms/token, not real compute)
- Pages are held during streaming to simulate decode memory pressure

### Comparison with theoretical limits

```
Transfer size per request = input_len × bytes_per_token × tp_size
Theoretical TTFT = transfer_size / link_bandwidth

Example: 8K tokens × 256 bytes × 8 TP = 16 MB
XGMI (MI300X): ~800 GB/s → 0.02 ms
RDMA (400 Gbps): ~50 GB/s → 0.3 ms
```

Actual TTFT will be higher due to bootstrap handshake, ZMQ latency, and concurrency contention.

## Router Configuration

For high-concurrency benchmarks, the Rust router needs matching limits:

```bash
python -m sglang_router.launch_router --pd-disaggregation \
    --host 0.0.0.0 --port 30000 --policy random \
    --prefill http://<prefill-ip>:30100 8998 \
    --decode http://<decode-ip>:30200 \
    --max-concurrent-requests 512 \  # >= max_concurrency
    --queue-size 4096 \              # >= num_prompts
    --cb-failure-threshold 100 \     # prevent circuit breaker cascade
    --retry-max-retries 3
```

Without these, the router's circuit breaker trips at ~200+ concurrent connections, causing `503 No available decode workers`.

## Notes on `bench_serving` Tokenizer

`bench_serving` requires a tokenizer to generate random prompts. Since `mock-kv-bench` is not a real model, pass any HuggingFace tokenizer:

```bash
--tokenizer meta-llama/Llama-3.1-8B
# or a local path:
--tokenizer /path/to/any/model
```

The tokenizer is only used client-side to generate random text; the mock server estimates token count from text length.

## Tested Configurations

| Topology | TP | Concurrency | Requests | Failures | Transfer |
|----------|-----|-------------|----------|----------|----------|
| Single-machine | 1+1 | 1-16 | 256 | 0 | XGMI |
| Single-machine | 4+4 | 1-16 | 320 | 0 | XGMI |
| Cross-machine | 8+8 | 1-256 | 3,952 | 0 | RDMA |

## Internals

### Page Allocator

Uses a numpy int32 stack (`pool_size × 4 bytes`). O(n) alloc/free where n = pages per request.
No temporary arrays. 37M pages = 150 MB constant memory.

### TP > 1 Architecture

- `torch.multiprocessing.spawn` creates one process per GPU
- Only rank 0 runs the HTTP server
- Rank 0 broadcasts requests to non-zero ranks via per-rank ZMQ PUSH/PULL (reliable, ordered)
- Port allocation: rank 0 binds to OS-assigned ports, broadcasts actual ports via `torch.distributed.broadcast`
- `_SyncPoller`: single poll thread per non-zero rank drives all transfers (prefill: two-phase poll+send; decode: poll until completion). Calls `clear()` on terminal state.
- Release barrier: non-zero ranks wait for their own transfer to complete (`wait_done()`) before freeing pages

### Bootstrap Lifecycle

1. Prefill rank 0 starts `CommonKVBootstrapServer` on `--bootstrap-port`
2. All prefill ranks register with the bootstrap server
3. On each request, decode side calls `try_ensure_parallel_info()` to discover prefill topology
4. Decode receiver sends metadata (destination buffer indices) to prefill sender via ZMQ
5. Prefill sender polls until `WaitingForInput` (metadata received), then calls `send()` for real GPU-to-GPU transfer
6. Both sides poll until `Success`, then clean up
