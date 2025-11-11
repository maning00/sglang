from __future__ import annotations

import importlib.util
import shutil
from typing import Optional

import torch


def has_flashinfer() -> bool:
    """Return True if FlashInfer is importable and nvcc is present."""
    if importlib.util.find_spec("flashinfer") is None:
        return False
    if shutil.which("nvcc") is None:
        return False
    return True


@torch.library.custom_op(
    "sglang::trtllm_fp4_block_scale_moe",
    mutates_args=["output"],
    device_types="cuda",
)
def trtllm_fp4_block_scale_moe(  # type: ignore[no-redef]
    output: torch.Tensor,
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_bias: Optional[torch.Tensor],
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm2_bias: Optional[torch.Tensor],
    output1_scale_scalar: Optional[torch.Tensor],
    output1_scale_gate_scalar: Optional[torch.Tensor],
    output2_scale_scalar: Optional[torch.Tensor],
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[torch.Tensor],
    tile_tokens_dim: Optional[int],
    routing_method_type: int,
    do_finalize: bool,
    tune_max_num_tokens: Optional[int] = None,
) -> torch.Tensor:
    """Runtime implementation that calls FlashInfer TRTLLM MoE.

    Returns the output tensor (FlashInfer returns a tuple; we return the first element).
    """
    try:
        from flashinfer.fused_moe import trtllm_fp4_block_scale_moe as _impl
    except Exception as e:
        raise RuntimeError(
            "FlashInfer TRTLLM MoE is not available; cannot execute custom op."
        ) from e

    result_tuple = _impl(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm1_bias=gemm1_bias,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        gemm2_bias=gemm2_bias,
        output1_scale_scalar=output1_scale_scalar,
        output1_scale_gate_scalar=output1_scale_gate_scalar,
        output2_scale_scalar=output2_scale_scalar,
        num_experts=num_experts,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        intermediate_size=intermediate_size,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        routed_scaling_factor=routed_scaling_factor,
        tile_tokens_dim=tile_tokens_dim,
        routing_method_type=routing_method_type,
        do_finalize=do_finalize,
        output=output,
        tune_max_num_tokens=tune_max_num_tokens,
    )
    # FlashInfer returns (output, maybe_workspace, ...)
    return result_tuple[0]


@torch.library.register_fake("sglang::trtllm_fp4_block_scale_moe")
def trtllm_fp4_block_scale_moe_fake(  # type: ignore[no-redef]
    output: torch.Tensor,
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_bias: Optional[torch.Tensor],
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm2_bias: Optional[torch.Tensor],
    output1_scale_scalar: Optional[torch.Tensor],
    output1_scale_gate_scalar: Optional[torch.Tensor],
    output2_scale_scalar: Optional[torch.Tensor],
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[torch.Tensor],
    tile_tokens_dim: Optional[int],
    routing_method_type: int,
    do_finalize: bool,
    tune_max_num_tokens: Optional[int] = None,
) -> torch.Tensor:
    # Use provided output tensor as the fake return; this preserves shape/dtype/device.
    return output
