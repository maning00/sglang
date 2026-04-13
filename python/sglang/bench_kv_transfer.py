"""
Standalone Mori KV Transfer Benchmark Tool.

Benchmarks mori-io transfer engine throughput/latency without model loading.
Creates lightweight mock prefill/decode servers that perform real GPU memory
transfers via mori RDMA/XGMI, plugging into the existing Rust router and
bench_serving infrastructure.

Usage:
    # Prefill (GPU 0-3)
    python -m sglang.bench_kv_transfer --mode prefill --port 30100 \
        --tp-size 4 --base-gpu-id 0 --num-layers 8 --head-dim 128 \
        --dtype fp8_e4m3 --kv-cache-type mla --max-concurrency 1024

    # Decode (GPU 4-7)
    python -m sglang.bench_kv_transfer --mode decode --port 30200 \
        --tp-size 4 --base-gpu-id 4 --num-layers 8 --head-dim 128 \
        --dtype fp8_e4m3 --kv-cache-type mla --max-concurrency 1024
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
import torch.multiprocessing as mp
import uvicorn
import zmq
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.utils import DisaggregationMode

logger = logging.getLogger("bench_kv_transfer")

# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

DTYPE_MAP = {
    "fp8_e4m3": (torch.float8_e4m3fn, 1),
    "fp8_e5m2": (torch.float8_e5m2, 1),
    "bf16": (torch.bfloat16, 2),
    "fp16": (torch.float16, 2),
    "fp32": (torch.float32, 4),
}


def _resolve_dtype(name: str) -> Tuple[torch.dtype, int]:
    if name not in DTYPE_MAP:
        raise ValueError(f"Unknown dtype {name!r}, choose from {list(DTYPE_MAP)}")
    return DTYPE_MAP[name]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mori KV Transfer Benchmark (no model loading)"
    )
    p.add_argument(
        "--mode",
        choices=["prefill", "decode"],
        required=True,
        help="Server role",
    )
    p.add_argument("--host", default="0.0.0.0", help="Bind address for HTTP")
    p.add_argument(
        "--advertise-host",
        default=None,
        help="Routable IP for mori/ZMQ (auto-detected if omitted)",
    )
    p.add_argument("--port", type=int, required=True, help="HTTP port")
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--base-gpu-id", type=int, default=0)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--kv-head-num", type=int, default=1)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument(
        "--kv-cache-type", choices=["mla", "mha"], default="mla"
    )
    p.add_argument("--page-size", type=int, default=1)
    p.add_argument("--dtype", default="fp8_e4m3", choices=list(DTYPE_MAP))
    p.add_argument("--bootstrap-port", type=int, default=8998)
    p.add_argument("--ib-device", default="")
    p.add_argument("--model-name", default="mock-kv-bench")
    p.add_argument("--max-concurrency", type=int, default=1024)
    p.add_argument("--max-input-len", type=int, default=8192)
    p.add_argument("--max-output-len", type=int, default=1024)
    p.add_argument(
        "--pool-size",
        type=int,
        default=0,
        help="Total KV token slots. 0 = auto from max-concurrency * (input+output)",
    )
    p.add_argument(
        "--aux-size",
        type=int,
        default=512,
        help="Aux metadata buffer size per slot (bytes)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# SimplePageAllocator
# ---------------------------------------------------------------------------


class SimplePageAllocator:
    """Thread-safe page allocator using numpy stack. O(n) alloc/free, zero temp arrays.

    Memory: 37M pages × 4 bytes = 150 MB constant. No per-request temporaries.
    """

    def __init__(self, pool_size: int, page_size: int):
        self.page_size = page_size
        self.num_pages = pool_size // page_size
        # Stack of free page indices: _stack[:_top] are available
        self._stack = np.arange(self.num_pages, dtype=np.int32)
        self._top = self.num_pages
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)

    def alloc(self, num_pages: int, timeout: float = 30.0) -> npt.NDArray[np.int32]:
        """Pop pages from stack top. O(num_pages) via numpy slice."""
        deadline = time.monotonic() + timeout
        with self.not_empty:
            while self._top < num_pages:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"Page allocation timeout: need {num_pages}, "
                        f"have {self._top}/{self.num_pages}"
                    )
                self.not_empty.wait(timeout=min(remaining, 1.0))
            self._top -= num_pages
            return self._stack[self._top : self._top + num_pages].copy()

    def reserve(self, indices: npt.NDArray[np.int32]) -> None:
        """Non-zero ranks: decrement available count (rank 0 chose the indices)."""
        with self.lock:
            self._top -= len(indices)

    def free(self, indices: npt.NDArray[np.int32]) -> None:
        """Push pages back onto stack."""
        with self.not_empty:
            idx = np.asarray(indices, dtype=np.int32)
            n = len(idx)
            self._stack[self._top : self._top + n] = idx
            self._top += n
            self.not_empty.notify_all()

    release = free


# ---------------------------------------------------------------------------
# PollNotifier — single thread polls all active transfers
# ---------------------------------------------------------------------------


class PollNotifier:
    """One poll thread per rank; polls all active sender/receiver objects."""

    def __init__(self):
        self._active: Dict[
            int, Tuple[Any, Callable, asyncio.Event, asyncio.AbstractEventLoop]
        ] = {}
        self._lock = threading.Lock()
        t = threading.Thread(target=self._loop, daemon=True, name="poll-notifier")
        t.start()

    def register(
        self,
        key: int,
        pollable: Any,
        condition_fn: Callable,
        event: asyncio.Event,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        with self._lock:
            self._active[key] = (pollable, condition_fn, event, loop)

    def unregister(self, key: int) -> None:
        with self._lock:
            self._active.pop(key, None)

    def _loop(self) -> None:
        while True:
            with self._lock:
                snapshot = list(self._active.items())
            if not snapshot:
                time.sleep(0.001)
                continue
            for key, (pollable, cond_fn, event, loop) in snapshot:
                try:
                    status = pollable.poll()
                    if cond_fn(status):
                        loop.call_soon_threadsafe(event.set)
                        with self._lock:
                            self._active.pop(key, None)
                except Exception:
                    loop.call_soon_threadsafe(event.set)
                    with self._lock:
                        self._active.pop(key, None)


async def poll_until(
    pollable: Any,
    poll_notifier: PollNotifier,
    key: Optional[int] = None,
    target_not: Optional[int] = None,
    target_in: Optional[Tuple[int, ...]] = None,
    timeout: float = 60.0,
) -> int:
    """Async wait until poll() meets condition."""
    # Quick check
    status = pollable.poll()
    if target_not is not None and status != target_not:
        return status
    if target_in is not None and status in target_in:
        return status

    event = asyncio.Event()
    loop = asyncio.get_running_loop()
    poll_key = key if key is not None else id(pollable)

    if target_not is not None:
        cond = lambda s: s != target_not
    else:
        cond = lambda s: s in target_in

    poll_notifier.register(poll_key, pollable, cond, event, loop)
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        return KVPoll.Failed
    finally:
        poll_notifier.unregister(poll_key)
    return pollable.poll()


# ---------------------------------------------------------------------------
# Parallel state + buffer allocation + MoriKVManager setup
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _init_parallel_state(tp_rank: int, tp_size: int, master_port: int) -> None:
    """Initialize torch.distributed (gloo) + model parallel groups."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)

    from sglang.srt.distributed.parallel_state import (
        initialize_model_parallel,
        init_distributed_environment,
    )

    init_distributed_environment(
        world_size=tp_size,
        rank=tp_rank,
        distributed_init_method=f"tcp://127.0.0.1:{master_port}",
        local_rank=tp_rank,
        backend="gloo",
    )

    initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
    )

    from sglang.srt.layers.dp_attention import initialize_dp_attention

    mock_server_args = SimpleNamespace(
        enable_dp_attention=False,
        dp_size=1,
        moe_dense_tp_size=None,
        attn_cp_size=1,
        device="cuda",
    )
    mock_model_config = SimpleNamespace(
        hidden_size=1,
        dtype=torch.bfloat16,
    )
    initialize_dp_attention(mock_server_args, mock_model_config)


def _validate_memory(
    pool_size: int,
    num_layers: int,
    kv_head_num: int,
    head_dim: int,
    dtype_size: int,
    kv_cache_type: str,
    gpu_id: int,
) -> None:
    """Check that KV pool fits in GPU memory."""
    if kv_cache_type == "mla":
        bytes_per_token = num_layers * 1 * head_dim * dtype_size
    else:
        bytes_per_token = num_layers * 2 * kv_head_num * head_dim * dtype_size
    total_bytes = pool_size * bytes_per_token
    avail = torch.cuda.mem_get_info(gpu_id)[0]
    if total_bytes > avail * 0.9:
        raise ValueError(
            f"KV pool requires {total_bytes / 1e9:.1f} GB but GPU {gpu_id} has "
            f"{avail / 1e9:.1f} GB free.\n"
            f"Reduce with: --num-layers, --head-dim, --kv-head-num, "
            f"--dtype fp8_e4m3, or --pool-size"
        )


def _allocate_kv_buffers(
    pool_size: int,
    num_layers: int,
    kv_head_num: int,
    head_dim: int,
    torch_dtype: torch.dtype,
    kv_cache_type: str,
    gpu_id: int,
) -> List[torch.Tensor]:
    """Allocate GPU KV cache buffers and return flat list."""
    device = f"cuda:{gpu_id}"
    if kv_cache_type == "mla":
        # MLA: one combined KV buffer per layer, shape [pool_size, 1, head_dim]
        return [
            torch.empty(pool_size, 1, head_dim, dtype=torch_dtype, device=device)
            for _ in range(num_layers)
        ]
    else:
        # MHA: separate K and V per layer, shape [pool_size, kv_head_num, head_dim]
        k_bufs = [
            torch.empty(
                pool_size, kv_head_num, head_dim, dtype=torch_dtype, device=device
            )
            for _ in range(num_layers)
        ]
        v_bufs = [
            torch.empty(
                pool_size, kv_head_num, head_dim, dtype=torch_dtype, device=device
            )
            for _ in range(num_layers)
        ]
        return k_bufs + v_bufs  # K layers first, then V layers


def _allocate_aux_buffers(
    num_slots: int, aux_size: int
) -> torch.Tensor:
    """Allocate CPU-pinned aux metadata buffer."""
    return torch.empty(num_slots * aux_size, dtype=torch.uint8).pin_memory()


def _build_kv_args(
    args: argparse.Namespace,
    tp_rank: int,
    gpu_id: int,
    page_size: int,
    kv_buffers: List[torch.Tensor],
    aux_buffer: torch.Tensor,
    aux_size: int,
    max_aux_slots: int,
) -> KVArgs:
    """Construct KVArgs from allocated buffers."""
    kv_args = KVArgs()
    kv_args.engine_rank = tp_rank
    kv_args.kv_data_ptrs = [buf.data_ptr() for buf in kv_buffers]
    kv_args.kv_data_lens = [buf.nelement() * buf.element_size() for buf in kv_buffers]

    # kv_item_lens: bytes per page (page_size tokens)
    if args.kv_cache_type == "mla":
        item_len = page_size * 1 * args.head_dim * _resolve_dtype(args.dtype)[1]
    else:
        item_len = (
            page_size * args.kv_head_num * args.head_dim * _resolve_dtype(args.dtype)[1]
        )
    kv_args.kv_item_lens = [item_len] * len(kv_buffers)

    # Aux buffers (sized by max_concurrency, not pool_size)
    kv_args.aux_data_ptrs = [aux_buffer.data_ptr()]
    kv_args.aux_data_lens = [aux_buffer.nelement()]
    kv_args.aux_item_lens = [aux_size]

    # No state tensors
    kv_args.state_data_ptrs = []
    kv_args.state_data_lens = []
    kv_args.state_item_lens = []
    kv_args.state_type = "none"
    kv_args.state_dim_per_tensor = []

    kv_args.ib_device = args.ib_device
    kv_args.ib_traffic_class = ""
    kv_args.gpu_id = gpu_id
    kv_args.kv_head_num = args.kv_head_num
    kv_args.total_kv_head_num = args.kv_head_num * args.tp_size
    kv_args.page_size = page_size
    kv_args.pp_rank = 0
    kv_args.prefill_start_layer = 0
    kv_args.system_dp_rank = 0

    return kv_args


def _build_mock_server_args(args: argparse.Namespace) -> SimpleNamespace:
    """Build a mock ServerArgs with fields that CommonKVManager reads."""
    from sglang.srt.utils.network import get_local_ip_auto

    advertise_host = args.advertise_host or get_local_ip_auto()

    return SimpleNamespace(
        host=advertise_host,
        port=args.port,
        disaggregation_bootstrap_port=args.bootstrap_port,
        dp_size=1,
        pp_size=1,
        enable_dp_attention=False,
        kv_cache_dtype=args.dtype,
        load_balance_method="round_robin",
        dist_init_addr=None,
        enable_hisparse=False,
    )


# ---------------------------------------------------------------------------
# HTTP server helpers
# ---------------------------------------------------------------------------


def _estimate_input_len(body: dict) -> int:
    """Estimate token count from request body.

    Priority:
    1. input_ids (exact token list from bench_serving --tokenize-prompt)
    2. __kv_num_tokens (explicit override via --extra-request-body)
    3. text length heuristic (~4 chars/token)
    """
    if "input_ids" in body:
        return len(body["input_ids"])
    if "__kv_num_tokens" in body:
        return int(body["__kv_num_tokens"])
    text = body.get("text", "")
    if not text:
        messages = body.get("messages", [])
        text = " ".join(
            m.get("content", "") for m in messages if isinstance(m, dict)
        )
    # Rough estimate: 1 token per 4 chars, minimum 1
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# SSE streaming
# ---------------------------------------------------------------------------


async def _sse_stream(
    room: int,
    input_len: int,
    output_len: int,
    transfer_ms: float,
    dst_indices: npt.NDArray[np.int32],
    dst_aux: int,
    page_allocator: SimplePageAllocator,
    aux_allocator: SimplePageAllocator,
    broadcast_release_fn: Optional[Callable] = None,
):
    """Sparse SSE: first-token chunk + final chunk. Pages held during streaming."""
    import orjson

    try:
        # Chunk 1: first token (triggers TTFT measurement in bench_serving)
        chunk1 = {
            "text": "x",
            "meta_info": {
                "id": str(room),
                "prompt_tokens": input_len,
                "completion_tokens": 1,
                "finish_reason": None,
            },
        }
        yield b"data: " + orjson.dumps(chunk1) + b"\n\n"

        if output_len > 1:
            # Simulate decode time: ~5ms per token
            decode_delay = (output_len - 1) * 0.005
            await asyncio.sleep(decode_delay)

            # Chunk 2: final
            chunk_final = {
                "text": "x" * output_len,
                "meta_info": {
                    "id": str(room),
                    "prompt_tokens": input_len,
                    "completion_tokens": output_len,
                    "finish_reason": {"type": "length", "length": output_len},
                },
            }
            yield b"data: " + orjson.dumps(chunk_final) + b"\n\n"

        yield b"data: [DONE]\n\n"
    finally:
        page_allocator.free(dst_indices)
        aux_allocator.free(np.array([dst_aux], dtype=np.int32))
        if broadcast_release_fn is not None:
            broadcast_release_fn(room)


# ---------------------------------------------------------------------------
# Worker — runs on each TP rank
# ---------------------------------------------------------------------------


def run_worker(
    tp_rank: int,
    args: argparse.Namespace,
    master_port: int,
) -> None:
    """Main worker function for each TP rank."""
    gpu_id = args.base_gpu_id + tp_rank
    torch.cuda.set_device(gpu_id)

    # Set advertise host before anything touches mori/ZMQ
    from sglang.srt.utils.network import get_local_ip_auto

    advertise_host = args.advertise_host or get_local_ip_auto()
    os.environ["SGLANG_HOST_IP"] = advertise_host

    # Initialize parallel state
    _init_parallel_state(tp_rank, args.tp_size, master_port)

    torch_dtype, dtype_size = _resolve_dtype(args.dtype)
    page_size = args.page_size

    # Compute pool size
    if args.pool_size > 0:
        pool_size = args.pool_size
    else:
        pool_size = args.max_concurrency * (
            math.ceil((args.max_input_len + args.max_output_len) / page_size)
            * page_size
        )

    # Validate memory
    _validate_memory(
        pool_size,
        args.num_layers,
        args.kv_head_num,
        args.head_dim,
        dtype_size,
        args.kv_cache_type,
        gpu_id,
    )

    # Allocate buffers
    kv_buffers = _allocate_kv_buffers(
        pool_size,
        args.num_layers,
        args.kv_head_num,
        args.head_dim,
        torch_dtype,
        args.kv_cache_type,
        gpu_id,
    )
    # Aux is per-request metadata, not per-token — size by max_concurrency
    max_aux_slots = args.max_concurrency
    aux_buffer = _allocate_aux_buffers(max_aux_slots, args.aux_size)

    # Build KVArgs
    kv_args = _build_kv_args(
        args, tp_rank, gpu_id, page_size, kv_buffers, aux_buffer, args.aux_size,
        max_aux_slots,
    )

    # Build mock server args
    mock_server_args = _build_mock_server_args(args)

    mode = (
        DisaggregationMode.PREFILL
        if args.mode == "prefill"
        else DisaggregationMode.DECODE
    )
    is_mla = args.kv_cache_type == "mla"

    # Bootstrap server (prefill rank 0 only)
    bootstrap_server = None
    bootstrap_ready = threading.Event()
    if args.mode == "prefill" and tp_rank == 0:
        from sglang.srt.disaggregation.mori.conn import MoriKVBootstrapServer

        bootstrap_server = MoriKVBootstrapServer(advertise_host, args.bootstrap_port)
        logger.info(
            "Bootstrap server started on %s:%d", advertise_host, args.bootstrap_port
        )

    # Synchronize: all ranks wait for bootstrap server to be up (prefill only)
    if args.mode == "prefill":
        torch.distributed.barrier()

    # Create MoriKVManager
    from sglang.srt.disaggregation.mori.conn import MoriKVManager

    kv_manager = MoriKVManager(kv_args, mode, mock_server_args, is_mla_backend=is_mla)
    logger.info(
        "Rank %d: MoriKVManager created (mode=%s, gpu=%d)", tp_rank, args.mode, gpu_id
    )

    # Wait for bootstrap to be ready (prefill only)
    if args.mode == "prefill" and tp_rank == 0 and bootstrap_server is not None:
        for _ in range(300):
            if bootstrap_server._is_ready():
                break
            time.sleep(1)
        if not bootstrap_server._is_ready():
            raise RuntimeError("Bootstrap server not ready after 300s")
        bootstrap_ready.set()
        logger.info("Bootstrap server ready — all %d ranks registered", args.tp_size)

    if args.mode == "prefill":
        torch.distributed.barrier()

    # Page allocators
    page_allocator = SimplePageAllocator(pool_size, page_size)
    aux_allocator = SimplePageAllocator(args.max_concurrency, 1)

    # Poll notifier
    poll_notifier = PollNotifier()

    # TP>1: ZMQ PUSH/PULL — rank 0 binds to port 0, broadcasts actual ports
    push_sockets = None
    broadcast_release_fn = None

    if args.tp_size > 1:
        ctx = zmq.Context()
        num_peers = args.tp_size - 1
        # Rank 0: bind to OS-assigned ports, then broadcast actual ports
        port_tensor = torch.zeros(num_peers, dtype=torch.long)
        if tp_rank == 0:
            push_sockets = []
            for i in range(num_peers):
                sock = ctx.socket(zmq.PUSH)
                sock.setsockopt(zmq.SNDHWM, 0)
                sock.bind("tcp://127.0.0.1:0")  # OS assigns port
                actual_port = int(
                    sock.getsockopt(zmq.LAST_ENDPOINT).decode().rsplit(":", 1)[-1]
                )
                port_tensor[i] = actual_port
                push_sockets.append(sock)

            def _broadcast_msg(msg):
                for sock in push_sockets:
                    sock.send_pyobj(msg)

            def _broadcast_release(bootstrap_room: int):
                _broadcast_msg(("release", bootstrap_room))

            broadcast_release_fn = _broadcast_release

        # Broadcast actual ports from rank 0 to all ranks
        torch.distributed.broadcast(port_tensor, src=0)

        if tp_rank != 0:
            my_port = int(port_tensor[tp_rank - 1].item())
            pull_socket = ctx.socket(zmq.PULL)
            pull_socket.connect(f"tcp://127.0.0.1:{my_port}")
            threading.Thread(
                target=_nonzero_rank_loop,
                args=(
                    tp_rank,
                    args,
                    kv_manager,
                    page_allocator,
                    aux_allocator,
                    pull_socket,
                ),
                daemon=True,
                name=f"tp-rank-{tp_rank}-worker",
            ).start()

        # Barrier ensures all PULL sockets connected before rank 0 sends
        torch.distributed.barrier()

    # Only rank 0 runs HTTP server
    if tp_rank == 0:
        _run_http_server(
            args,
            kv_manager,
            page_allocator,
            aux_allocator,
            poll_notifier,
            bootstrap_ready if args.mode == "prefill" else None,
            push_sockets,
            broadcast_release_fn,
            pool_size,
        )
    else:
        # Non-zero ranks block forever (worker loop runs in daemon thread)
        threading.Event().wait()


# ---------------------------------------------------------------------------
# Non-zero rank worker loop (TP>1)
# ---------------------------------------------------------------------------


class _SyncPoller:
    """Single thread polls all active pollables to completion.

    For prefill senders: Bootstrapping → WaitingForInput → send() → terminal → clear().
    For decode receivers: poll until terminal → clear().
    Tracks completed rooms so release can wait for transfer to finish.
    """

    def __init__(self, tp_rank: int):
        self._tp_rank = tp_rank
        self._active: Dict[int, dict] = {}
        self._done: set = set()  # rooms that reached terminal + cleared
        self._lock = threading.Lock()
        self._done_cond = threading.Condition(self._lock)
        threading.Thread(
            target=self._loop, daemon=True, name=f"sync-poller-{tp_rank}"
        ).start()

    def add_sender(
        self, room: int, sender: Any, kv_indices: npt.NDArray[np.int32]
    ) -> None:
        with self._lock:
            self._done.discard(room)
            self._active[room] = {
                "pollable": sender,
                "kv_indices": kv_indices,
                "phase": "bootstrap",
            }

    def add_receiver(self, room: int, receiver: Any) -> None:
        with self._lock:
            self._done.discard(room)
            self._active[room] = {
                "pollable": receiver,
                "phase": "wait_transfer",
            }

    def wait_done(self, room: int, timeout: float = 60.0) -> bool:
        """Block until room reaches terminal state. Returns True if done."""
        deadline = time.monotonic() + timeout
        with self._done_cond:
            while room not in self._done:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._done_cond.wait(timeout=min(remaining, 0.5))
            return True

    def remove(self, room: int) -> None:
        with self._lock:
            self._active.pop(room, None)
            self._done.discard(room)

    def _mark_done(self, room: int, pollable: Any) -> None:
        try:
            pollable.clear()
        except Exception:
            pass
        with self._done_cond:
            self._active.pop(room, None)
            self._done.add(room)
            self._done_cond.notify_all()

    def _loop(self) -> None:
        while True:
            with self._lock:
                snapshot = list(self._active.items())
            if not snapshot:
                time.sleep(0.001)
                continue
            for room, info in snapshot:
                try:
                    pollable = info["pollable"]
                    phase = info["phase"]
                    if phase == "done":
                        continue
                    status = pollable.poll()
                    if phase == "bootstrap":
                        if status == KVPoll.Failed:
                            self._mark_done(room, pollable)
                        elif status != KVPoll.Bootstrapping:
                            pollable.send(info["kv_indices"], state_indices=None)
                            info["phase"] = "transfer"
                    elif phase == "transfer":
                        if status in (KVPoll.Success, KVPoll.Failed):
                            self._mark_done(room, pollable)
                    elif phase == "wait_transfer":
                        if status in (KVPoll.Success, KVPoll.Failed):
                            self._mark_done(room, pollable)
                except Exception:
                    logger.exception(
                        "Rank %d: poller error room %d", self._tp_rank, room
                    )
                    self._mark_done(room, info["pollable"])


def _nonzero_rank_loop(
    tp_rank: int,
    args: argparse.Namespace,
    kv_manager: Any,
    page_allocator: SimplePageAllocator,
    aux_allocator: SimplePageAllocator,
    pull_socket: zmq.Socket,
) -> None:
    """Non-zero TP rank: listen for broadcasts, handle transfers."""
    from sglang.srt.disaggregation.mori.conn import MoriKVReceiver, MoriKVSender

    active_rooms: Dict[int, dict] = {}
    mode = args.mode
    # Single poller drives transfer lifecycle for both prefill and decode
    poller = _SyncPoller(tp_rank)

    while True:
        try:
            msg = pull_socket.recv_pyobj()
        except Exception:
            logger.exception("Non-zero rank %d: recv error", tp_rank)
            continue

        cmd = msg[0]
        if cmd == "request":
            _, bootstrap_room, bootstrap_addr, kv_indices_bytes, aux_idx, num_pages = (
                msg
            )
            kv_indices = np.frombuffer(kv_indices_bytes, dtype=np.int32).copy()
            try:
                page_allocator.reserve(kv_indices)
                aux_allocator.reserve(np.array([aux_idx], dtype=np.int32))
            except Exception:
                logger.warning(
                    "Rank %d: reserve failed for room %d", tp_rank, bootstrap_room
                )
                continue

            active_rooms[bootstrap_room] = {
                "kv_indices": kv_indices,
                "aux_idx": aux_idx,
            }

            if mode == "prefill":
                sender = MoriKVSender(
                    kv_manager,
                    bootstrap_addr,
                    bootstrap_room,
                    dest_tp_ranks=list(range(args.tp_size)),
                    pp_rank=0,
                )
                sender.init(num_kv_indices=num_pages, aux_index=int(aux_idx))
                poller.add_sender(bootstrap_room, sender, kv_indices)
            elif mode == "decode":
                kv_manager.try_ensure_parallel_info(bootstrap_addr)
                receiver = MoriKVReceiver(
                    kv_manager, bootstrap_addr, bootstrap_room
                )
                receiver.init(prefill_dp_rank=0)
                receiver.send_metadata(
                    kv_indices=kv_indices,
                    aux_index=int(aux_idx),
                    state_indices=None,
                )
                poller.add_receiver(bootstrap_room, receiver)

        elif cmd == "release":
            bootstrap_room = msg[1]
            # Wait for this rank's transfer to complete before freeing
            poller.wait_done(bootstrap_room, timeout=60)
            poller.remove(bootstrap_room)
            info = active_rooms.pop(bootstrap_room, None)
            if info:
                page_allocator.free(info["kv_indices"])
                aux_allocator.free(np.array([info["aux_idx"]], dtype=np.int32))


# ---------------------------------------------------------------------------
# HTTP Server (rank 0 only)
# ---------------------------------------------------------------------------


def _run_http_server(
    args: argparse.Namespace,
    kv_manager: Any,
    page_allocator: SimplePageAllocator,
    aux_allocator: SimplePageAllocator,
    poll_notifier: PollNotifier,
    bootstrap_ready: Optional[threading.Event],
    push_sockets: Optional[List[zmq.Socket]],
    broadcast_release_fn: Optional[Callable],
    pool_size: int,
) -> None:
    """Create and run FastAPI app on rank 0."""
    from sglang.srt.disaggregation.mori.conn import (
        MoriKVReceiver,
        MoriKVSender,
    )

    app = FastAPI()
    mode = args.mode
    page_size = args.page_size
    tp_size = args.tp_size

    # Helpers for TP>1 broadcast (PUSH/PULL — reliable)
    def broadcast_msg(msg):
        if push_sockets is not None:
            for sock in push_sockets:
                sock.send_pyobj(msg)

    def broadcast_request(
        bootstrap_room: int,
        bootstrap_addr: str,
        kv_indices: npt.NDArray[np.int32],
        aux_idx: int,
        num_pages: int,
    ):
        broadcast_msg((
            "request",
            bootstrap_room,
            bootstrap_addr,
            kv_indices.tobytes(),
            aux_idx,
            num_pages,
        ))

    def broadcast_release(bootstrap_room: int):
        if broadcast_release_fn is not None:
            broadcast_release_fn(bootstrap_room)

    # --- Endpoints ---

    @app.get("/health")
    async def health():
        if mode == "prefill" and bootstrap_ready is not None:
            if not bootstrap_ready.is_set():
                return Response(status_code=503)
        return Response(status_code=200)

    @app.get("/v1/models")
    async def models():
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": args.model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "sglang",
                    }
                ],
            }
        )

    @app.get("/get_model_info")
    async def model_info():
        return JSONResponse(
            {"model_path": args.model_name, "is_generation": True}
        )

    @app.get("/server_info")
    async def server_info():
        return JSONResponse(
            {
                "model_path": args.model_name,
                "tokenizer_path": args.model_name,
                "is_generation": True,
                "tp_size": args.tp_size,
                "dp_size": 1,
                "load_balance_method": "round_robin",
                "disaggregation_mode": args.mode,
                "version": "0.0.1-kv-bench",
                "internal_states": [{"last_gen_throughput": 0.0}],
            }
        )

    @app.post("/generate")
    async def generate(request: Request):
        body = await request.json()

        # Extract bootstrap info injected by Rust router
        bootstrap_host = body.get("bootstrap_host", "")
        bootstrap_port = body.get("bootstrap_port", args.bootstrap_port)
        bootstrap_room = body.get("bootstrap_room")

        if bootstrap_room is None:
            return JSONResponse(
                {"error": "missing bootstrap_room"}, status_code=400
            )

        bootstrap_room = int(bootstrap_room)
        bootstrap_addr = f"{bootstrap_host}:{bootstrap_port}"

        if mode == "prefill":
            return await _handle_prefill(
                body,
                bootstrap_addr,
                bootstrap_room,
                kv_manager,
                page_allocator,
                aux_allocator,
                poll_notifier,
                page_size,
                tp_size,
                broadcast_request,
                broadcast_release,
            )
        else:
            return await _handle_decode(
                body,
                bootstrap_addr,
                bootstrap_room,
                kv_manager,
                page_allocator,
                aux_allocator,
                poll_notifier,
                page_size,
                tp_size,
                broadcast_request,
                broadcast_release,
            )

    @app.api_route("/v1/chat/completions", methods=["POST"])
    async def chat_completions(request: Request):
        return await generate(request)

    logger.info("Starting HTTP server on %s:%d (mode=%s)", args.host, args.port, mode)
    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        access_log=False,
    )
    server = uvicorn.Server(config)
    server.run()


# ---------------------------------------------------------------------------
# Prefill handler
# ---------------------------------------------------------------------------


async def _handle_prefill(
    body: dict,
    bootstrap_addr: str,
    bootstrap_room: int,
    kv_manager: Any,
    page_allocator: SimplePageAllocator,
    aux_allocator: SimplePageAllocator,
    poll_notifier: PollNotifier,
    page_size: int,
    tp_size: int,
    broadcast_request: Callable,
    broadcast_release: Callable,
) -> Response:
    from sglang.srt.disaggregation.mori.conn import MoriKVSender

    input_len = _estimate_input_len(body)
    num_pages = math.ceil(input_len / page_size)

    kv_indices = page_allocator.alloc(num_pages)
    aux_idx_arr = aux_allocator.alloc(1)
    aux_idx = int(aux_idx_arr[0])
    sender = None

    try:
        # Broadcast to non-zero ranks (they self-drive via SyncPoller)
        if tp_size > 1:
            broadcast_request(
                bootstrap_room, bootstrap_addr, kv_indices, aux_idx, num_pages
            )

        sender = MoriKVSender(
            kv_manager,
            bootstrap_addr,
            bootstrap_room,
            dest_tp_ranks=list(range(tp_size)),
            pp_rank=0,
        )
        sender.init(num_kv_indices=num_pages, aux_index=aux_idx)

        # Phase 1: Wait for decode to register (Bootstrapping -> WaitingForInput)
        status = await poll_until(
            sender,
            poll_notifier,
            key=bootstrap_room,
            target_not=KVPoll.Bootstrapping,
            timeout=60,
        )
        if status == KVPoll.Failed:
            logger.error(
                "Prefill room %d: bootstrap failed (input_len=%d)",
                bootstrap_room, input_len,
            )
            return JSONResponse({"error": "bootstrap timeout"}, status_code=500)

        # Phase 2: Send KV and wait for completion
        t0 = time.perf_counter()
        sender.send(kv_indices, state_indices=None)

        status = await poll_until(
            sender,
            poll_notifier,
            key=bootstrap_room + 1_000_000_000,
            target_in=(KVPoll.Success, KVPoll.Failed),
            timeout=60,
        )
        transfer_ms = (time.perf_counter() - t0) * 1000

    except Exception:
        logger.exception("Prefill room %d: unexpected error", bootstrap_room)
        raise
    finally:
        page_allocator.free(kv_indices)
        aux_allocator.free(aux_idx_arr)
        if tp_size > 1:
            broadcast_release(bootstrap_room)
        if sender is not None:
            sender.clear()

    if status == KVPoll.Failed:
        logger.error(
            "Prefill room %d: transfer failed after %.1fms (input_len=%d, pages=%d)",
            bootstrap_room, transfer_ms, input_len, num_pages,
        )
        return JSONResponse({"error": "transfer failed"}, status_code=500)

    logger.info(
        "Prefill room %d: OK %.1fms (input_len=%d, pages=%d)",
        bootstrap_room, transfer_ms, input_len, num_pages,
    )
    return JSONResponse(
        {
            "text": "",
            "meta_info": {
                "id": str(bootstrap_room),
                "prompt_tokens": input_len,
                "completion_tokens": 0,
                "finish_reason": {"type": "length", "length": 0},
            },
        }
    )


# ---------------------------------------------------------------------------
# Decode handler
# ---------------------------------------------------------------------------


async def _handle_decode(
    body: dict,
    bootstrap_addr: str,
    bootstrap_room: int,
    kv_manager: Any,
    page_allocator: SimplePageAllocator,
    aux_allocator: SimplePageAllocator,
    poll_notifier: PollNotifier,
    page_size: int,
    tp_size: int,
    broadcast_request: Callable,
    broadcast_release: Callable,
) -> Response:
    from sglang.srt.disaggregation.mori.conn import MoriKVReceiver

    is_stream = body.get("stream", True)
    input_len = _estimate_input_len(body)
    output_len = body.get("sampling_params", {}).get("max_new_tokens", 1)
    num_pages = math.ceil((input_len + output_len) / page_size)

    dst_indices = page_allocator.alloc(num_pages)
    dst_aux_arr = aux_allocator.alloc(1)
    dst_aux = int(dst_aux_arr[0])
    pages_owned = True

    try:
        # Broadcast to non-zero ranks
        if tp_size > 1:
            broadcast_request(
                bootstrap_room, bootstrap_addr, dst_indices, dst_aux, num_pages
            )

        if not kv_manager.try_ensure_parallel_info(bootstrap_addr):
            logger.error(
                "Decode room %d: cannot reach bootstrap at %s",
                bootstrap_room, bootstrap_addr,
            )
            return JSONResponse(
                {"error": "cannot reach bootstrap"}, status_code=503
            )

        receiver = MoriKVReceiver(kv_manager, bootstrap_addr, bootstrap_room)
        receiver.init(prefill_dp_rank=0)
        receiver.send_metadata(
            kv_indices=dst_indices, aux_index=dst_aux, state_indices=None
        )

        t0 = time.perf_counter()
        status = await poll_until(
            receiver,
            poll_notifier,
            key=bootstrap_room,
            target_in=(KVPoll.Success, KVPoll.Failed),
            timeout=60,
        )
        transfer_ms = (time.perf_counter() - t0) * 1000
        receiver.clear()

        if status == KVPoll.Failed:
            logger.error(
                "Decode room %d: transfer failed after %.1fms (input_len=%d, pages=%d)",
                bootstrap_room, transfer_ms, input_len, num_pages,
            )
            return JSONResponse({"error": "transfer failed"}, status_code=500)

        logger.info(
            "Decode room %d: OK %.1fms (input_len=%d, pages=%d)",
            bootstrap_room, transfer_ms, input_len, num_pages,
        )
        if is_stream:
            pages_owned = False
            return StreamingResponse(
                _sse_stream(
                    bootstrap_room,
                    input_len,
                    output_len,
                    transfer_ms,
                    dst_indices,
                    dst_aux,
                    page_allocator,
                    aux_allocator,
                    broadcast_release if tp_size > 1 else None,
                ),
                media_type="text/event-stream",
            )
        else:
            return JSONResponse(
                {
                    "text": "x" * output_len,
                    "meta_info": {
                        "id": str(bootstrap_room),
                        "prompt_tokens": input_len,
                        "completion_tokens": output_len,
                        "finish_reason": {"type": "length", "length": output_len},
                    },
                }
            )
    except Exception:
        logger.exception("Decode room %d: unexpected error", bootstrap_room)
        raise
    finally:
        if pages_owned:
            page_allocator.free(dst_indices)
            aux_allocator.free(dst_aux_arr)
            if tp_size > 1:
                broadcast_release(bootstrap_room)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Set advertise host early
    from sglang.srt.utils.network import get_local_ip_auto

    advertise_host = args.advertise_host or get_local_ip_auto()
    os.environ["SGLANG_HOST_IP"] = advertise_host

    master_port = _find_free_port()

    _, dtype_size = _resolve_dtype(args.dtype)
    if args.kv_cache_type == "mla":
        bytes_per_token = args.num_layers * 1 * args.head_dim * dtype_size
    else:
        bytes_per_token = (
            args.num_layers * 2 * args.kv_head_num * args.head_dim * dtype_size
        )

    pool_size = args.pool_size
    if pool_size == 0:
        pool_size = args.max_concurrency * (
            math.ceil((args.max_input_len + args.max_output_len) / args.page_size)
            * args.page_size
        )
    total_mem = pool_size * bytes_per_token
    logger.info(
        "Config: mode=%s, tp=%d, gpu_base=%d, layers=%d, head_dim=%d, "
        "kv_heads=%d, dtype=%s, type=%s",
        args.mode,
        args.tp_size,
        args.base_gpu_id,
        args.num_layers,
        args.head_dim,
        args.kv_head_num,
        args.dtype,
        args.kv_cache_type,
    )
    logger.info(
        "Pool: %d tokens, %d bytes/token, %.2f GB/GPU total",
        pool_size,
        bytes_per_token,
        total_mem / 1e9,
    )

    if args.tp_size == 1:
        run_worker(0, args, master_port)
    else:
        mp.spawn(
            run_worker,
            args=(args, master_port),
            nprocs=args.tp_size,
            join=True,
        )


if __name__ == "__main__":
    main()
