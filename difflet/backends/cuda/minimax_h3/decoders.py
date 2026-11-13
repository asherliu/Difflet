"""MiniMax-H3's VAE decoders on CUDA, FP32 weights.

The video decoder runs its transformer blocks under FP16 autocast, as the reference does,
with the reference's arithmetic around them (``_CudaViTDecoder``): the token grid and rotary
tables in the activation dtype, and the input/output projections in FP32 outside autocast.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from difflet.backends._torch_megatron.checkpoint import load_checkpoint_into
from difflet.backends._torch_megatron.weights import materialize_meta_
from difflet.models.minimax_h3.modeling_vae import (
    MiniMaxH3AudioBigVGANDecoder,
    MiniMaxH3VideoViTDecoder3d,
)


def _fp32_linear(linear: nn.Linear, hidden_states: torch.Tensor) -> torch.Tensor:
    """``linear`` in its FP32 weights with autocast off, cast back to the input dtype."""
    with torch.autocast(hidden_states.device.type, enabled=False):
        output = linear(hidden_states.to(linear.weight.dtype))
    return output.to(hidden_states.dtype)


class _CudaViTDecoder(MiniMaxH3VideoViTDecoder3d):
    """``MiniMaxH3VideoViTDecoder3d.forward`` (no padding rows) with the reference's dtypes."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        dtype = hidden_states.dtype
        hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(
            batch_size, num_frames * height * width, num_channels
        )
        hidden_states = _fp32_linear(self.proj_in, hidden_states)
        num_patches = hidden_states.shape[1]
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        cls_token = torch.zeros_like(hidden_states[:, :1, :])
        hidden_states = torch.cat([hidden_states, register_tokens, cls_token], dim=1)

        grids = [
            2.0 * (torch.arange(0.5, size, dtype=dtype, device=hidden_states.device) / size) - 1.0
            for size in (num_frames, height, width)
        ]
        position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1, -1)
        suffix_ids = position_ids.new_zeros((batch_size, self.num_register_tokens + 1, 3))
        position_ids = torch.cat([position_ids, suffix_ids], dim=1)
        cos, sin = self.rope(position_ids)
        rotary_emb = (cos.to(dtype), sin.to(dtype))
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, rotary_emb, None)

        hidden_states = _fp32_linear(self.proj_out, self.norm_out(hidden_states))
        hidden_states = hidden_states[:, :num_patches, :]
        patch_size, patch_size_t = self.patch_size, self.patch_size_t
        hidden_states = hidden_states.view(
            batch_size,
            num_frames,
            height,
            width,
            self.out_channels,
            patch_size_t,
            patch_size,
            patch_size,
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return hidden_states.reshape(
            batch_size,
            self.out_channels,
            num_frames * patch_size_t,
            height * patch_size,
            width * patch_size,
        )


class _VideoDecoder(nn.Module):
    """Latent tile (on any device) to pixels on the decoder's device."""

    def __init__(self, config):
        super().__init__()
        config = dict(config)
        for kind in ("spatial", "temporal"):
            factors = config.get(f"{kind}_downsample_factors")
            if factors is not None:
                config[f"{kind}_compression_ratio"] = math.prod(factors)
        self.config = SimpleNamespace(**config)
        c = self.config
        self.post_quant_conv = nn.Conv3d(c.latent_channels, c.latent_channels, 1)
        self.decoder = _CudaViTDecoder(
            in_channels=c.latent_channels,
            out_channels=c.out_channels,
            patch_size=c.spatial_compression_ratio,
            patch_size_t=c.temporal_compression_ratio,
            num_layers=c.decoder_num_layers,
            num_attention_heads=c.decoder_num_attention_heads,
            attention_head_dim=c.decoder_attention_head_dim,
            num_register_tokens=c.decoder_num_register_tokens,
            ffn_mult=c.decoder_ffn_mult,
            rope_theta=c.decoder_rope_theta,
            rope_dim_ratio=c.decoder_rope_dim_ratio,
            norm_eps=c.decoder_norm_eps,
        )

    def forward(self, latents):
        latents = latents.to(self.post_quant_conv.weight.device)
        with torch.autocast(latents.device.type, dtype=torch.float16):
            return self.decoder(self.post_quant_conv(latents))


class _AudioDecoder(nn.Module):
    """Stereo-as-batch latents (on any device) to waveforms on the decoder's device."""

    def __init__(self, config):
        super().__init__()
        self.config = SimpleNamespace(**config)
        c = self.config
        self.dec_in_proj = nn.Conv1d(c.latent_channels, c.latent_dim, 1)
        # Keep the checkpoint's weight_g/weight_v representation. Its forward
        # hooks recompute the weight from the loaded FP32 parameters.
        self.decoder = MiniMaxH3AudioBigVGANDecoder(
            in_channels=c.latent_dim,
            upsample_initial_channel=c.decoder_dim,
            upsample_rates=tuple(c.decoder_rates),
            upsample_kernel_sizes=tuple(c.decoder_kernel_sizes),
            resblock_kernel_sizes=tuple(c.resblock_kernel_sizes),
            resblock_dilation_sizes=tuple(tuple(v) for v in c.resblock_dilation_sizes),
            use_weight_norm=True,
        )

    def forward(self, latents):
        latents = latents.to(self.dec_in_proj.weight.device)
        return self.decoder(self.dec_in_proj(latents))


def _load(module_cls, model_path: str, device) -> nn.Module:
    from accelerate import init_empty_weights

    config = json.loads((Path(model_path) / "config.json").read_text())
    with init_empty_weights(include_buffers=False):
        module = module_cls(config)
    materialize_meta_(module, device=device, dtype=torch.float32)
    load_checkpoint_into(module, model_path, tp_size=1, tp_rank=0, strict=True)
    return module.to(device).eval().requires_grad_(False)


def load_video_decoder(model_path: str, *, device) -> nn.Module:
    return _load(_VideoDecoder, model_path, device)


def load_audio_decoder(model_path: str, *, device) -> nn.Module:
    return _load(_AudioDecoder, model_path, device)


__all__ = ["load_audio_decoder", "load_video_decoder"]
