"""MiniMax-H3's CUDA arithmetic, installed on the shared transformer after strict loading.

The shared ``MiniMaxH3Transformer3DModel`` keeps its default arithmetic for the other
backends. On CUDA the per-row elementwise work follows the reference's FP32 formulas, with
one BF16 rounding where the reference stores:

- every standalone RMS norm (token refiner, Q/K norms without rotary, output norm) in FP32,
  as the ``difflet_h3::rms_norm`` custom op;
- in each main block, RMS norm + AdaLN modulation, the gated residuals and the SwiGLU gate
  as fused ``difflet_h3::`` kernels (``fused_ops``), and per-head Q/K RMS norm + rotary
  embedding as one more;
- the output norm's modulation in FP32.

Compiled blocks call the same custom ops, so eager and compiled runs share these kernels;
Inductor would otherwise fuse the norms' reductions in its own order.
"""

from __future__ import annotations

import types

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_utils import get_parameter_dtype
from torch import nn

from difflet.backends.cuda.minimax_h3 import fused_ops  # noqa: F401  (registers difflet_h3::)
from difflet.models.minimax_h3.modeling_minimax_h3 import (
    MiniMaxH3AttnProcessor,
    MiniMaxH3Transformer3DModel,
)


@torch.library.custom_op("difflet_h3::rms_norm", mutates_args=())
def _rms_norm_op(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Statistics, scaling and the weight product in FP32; one cast back."""
    value = hidden_states.float()
    normalized = value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + eps)
    return (weight.float() * normalized).to(hidden_states.dtype)


@_rms_norm_op.register_fake
def _(hidden_states, weight, eps):
    return torch.empty_like(hidden_states)


def _fp32_rms_norm(self: nn.RMSNorm, hidden_states: torch.Tensor) -> torch.Tensor:
    return torch.ops.difflet_h3.rms_norm(hidden_states, self.weight, self.eps)


class CudaMiniMaxH3AttnProcessor(MiniMaxH3AttnProcessor):
    """With rotary embedding (main blocks), Q/K norm and rotation are one fused kernel."""

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        if rotary_emb is None:
            query, key = attn.norm_q(query), attn.norm_k(key)
        else:
            head_norm_rotary = torch.ops.difflet_h3.head_norm_rotary
            query = head_norm_rotary(query, attn.norm_q.weight, attn.norm_q.eps, *rotary_emb)
            key = head_norm_rotary(key, attn.norm_k.weight, attn.norm_k.eps, *rotary_emb)
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = attn.to_out[0](hidden_states.flatten(2, 3).type_as(query))
        return attn.to_out[1](hidden_states)


def _block_forward(
    self,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    adaln_indices: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    block_modulation: torch.Tensor | None = None,
) -> torch.Tensor:
    """``MiniMaxH3TransformerBlock.forward`` with the fused per-row kernels."""
    ops = torch.ops.difflet_h3
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)
    normed = ops.rms_norm_modulate(
        hidden_states, self.norm1.weight, self.norm1.eps, shift_msa, scale_msa, adaln_indices
    )
    attn_output = self.attn(normed, rotary_emb, attention_mask)
    hidden_states = ops.gated_residual(hidden_states, gate_msa, adaln_indices, attn_output)
    normed = ops.rms_norm_modulate(
        hidden_states, self.norm2.weight, self.norm2.eps, shift_mlp, scale_mlp, adaln_indices
    )
    ff_output = ops.swiglu(self.ff.net[0].proj(normed))
    for module in self.ff.net[1:]:
        ff_output = module(ff_output)
    return ops.gated_residual(hidden_states, gate_mlp, adaln_indices, ff_output)


def _norm_out_forward(
    self, hidden_states: torch.Tensor, temb: torch.Tensor, timestep_indices: torch.Tensor
) -> torch.Tensor:
    """``MiniMaxH3AdaLayerNormOut.forward`` with the modulation in FP32."""
    shift, scale = self.linear(nn.functional.silu(temb).to(get_parameter_dtype(self.linear))).chunk(
        2, dim=-1
    )
    hidden_states = self.norm(hidden_states)
    scale = scale.index_select(0, timestep_indices).float()
    shift = shift.index_select(0, timestep_indices).float()
    return (hidden_states.float() * (1.0 + scale) + shift).to(hidden_states.dtype)


def install_cuda_numerics(model: MiniMaxH3Transformer3DModel) -> None:
    """Install the CUDA arithmetic on a loaded transformer, before any compile."""
    for module in model.modules():
        if isinstance(module, nn.RMSNorm):
            module.forward = types.MethodType(_fp32_rms_norm, module)
    processor = CudaMiniMaxH3AttnProcessor()
    for block in model.token_refiner.refiner_blocks:
        block.attn.processor = processor
    for block in model.transformer_blocks:
        block.attn.processor = processor
        block.forward = types.MethodType(_block_forward, block)
    model.norm_out.forward = types.MethodType(_norm_out_forward, model.norm_out)


__all__ = ["CudaMiniMaxH3AttnProcessor", "install_cuda_numerics"]
