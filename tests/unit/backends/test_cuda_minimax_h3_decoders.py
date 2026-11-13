"""MiniMax-H3's VAE modules on CUDA, on CPU: strict FP32 loading, the video decoder's
reference arithmetic around its FP16 blocks, and the shared tiled decode they plug into."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from difflet.backends.cuda.minimax_h3 import decoders
from difflet.models.minimax_h3.modeling_vae import MiniMaxH3VideoViTDecoder3d
from difflet.models.minimax_h3.pipeline import decode_minimax_h3_video


def _video_config():
    return dict(
        latent_channels=2,
        out_channels=3,
        spatial_downsample_factors=[2],
        temporal_downsample_factors=[2],
        decoder_num_layers=1,
        decoder_num_attention_heads=1,
        decoder_attention_head_dim=12,
        decoder_num_register_tokens=1,
        decoder_ffn_mult=2,
        decoder_rope_theta=100.0,
        decoder_rope_dim_ratio=0.5,
        decoder_norm_eps=1e-5,
        latents_mean=[0.0, 0.0],
        latents_std=[1.0, 1.0],
    )


def _audio_config():
    return dict(
        latent_channels=2,
        latent_dim=4,
        decoder_dim=8,
        decoder_rates=[2, 2],
        decoder_kernel_sizes=[4, 4],
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1, 3, 5]],
        latents_mean=[0.0, 0.0],
        latents_std=[1.0, 1.0],
    )


@pytest.mark.parametrize(
    "load,module,config,shape,expected",
    [
        ("load_video_decoder", "_VideoDecoder", _video_config(), (1, 2, 2, 2, 2), (1, 3, 4, 4, 4)),
        ("load_audio_decoder", "_AudioDecoder", _audio_config(), (2, 2, 8), (2, 1, 32)),
    ],
)
def test_decoder_checkpoint_roundtrip(tmp_path, load, module, config, shape, expected):
    torch.manual_seed(3)
    original = getattr(decoders, module)(config).eval()
    (tmp_path / "config.json").write_text(json.dumps(config))
    weights = {k: v.contiguous() for k, v in original.state_dict().items()}
    save_file(weights, str(tmp_path / "model.safetensors"))
    loaded = getattr(decoders, load)(str(tmp_path), device="cpu")
    x = torch.randn(shape)
    with torch.no_grad():
        actual = loaded(x)
        assert actual.shape == expected
        torch.testing.assert_close(actual, original(x), rtol=0, atol=0)
    assert all(p.dtype == torch.float32 for p in loaded.parameters())
    assert loaded.config.latents_mean == config["latents_mean"]


def test_the_video_config_derives_its_compression_from_the_factors():
    config = _video_config()
    config.update(spatial_downsample_factors=[2, 2, 2, 2], temporal_downsample_factors=[1, 2, 2])
    decoder = decoders._VideoDecoder(config)
    assert (
        decoder.config.spatial_compression_ratio,
        decoder.config.temporal_compression_ratio,
    ) == (
        16,
        4,
    )
    assert (decoder.decoder.patch_size, decoder.decoder.patch_size_t) == (16, 4)


def _vit(cls):
    torch.manual_seed(4)
    return cls(
        in_channels=4,
        out_channels=3,
        patch_size=2,
        patch_size_t=2,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=24,
        num_register_tokens=2,
        ffn_mult=2,
    ).eval()


def test_in_fp32_the_cuda_decoder_is_the_shared_decoder():
    x = torch.randn(1, 4, 3, 4, 5)
    with torch.no_grad():
        got = _vit(decoders._CudaViTDecoder)(x)
        want = _vit(MiniMaxH3VideoViTDecoder3d)(x)
    assert torch.equal(got, want)


def test_under_autocast_the_projections_stay_fp32_and_the_tables_follow_the_activations(
    monkeypatch,
):
    vit = _vit(decoders._CudaViTDecoder)
    seen = []
    linear = decoders._fp32_linear

    def spy(module, hidden_states):
        out = linear(module, hidden_states)
        seen.append((module.weight.dtype, hidden_states.dtype, out.dtype))
        return out

    monkeypatch.setattr(decoders, "_fp32_linear", spy)
    rope = vit.rope.forward
    tables = []
    monkeypatch.setattr(vit.rope, "forward", lambda ids: tables.append(ids.dtype) or rope(ids))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = vit(torch.randn(1, 4, 3, 4, 5, dtype=torch.bfloat16))
    # proj_in on the BF16 activations and proj_out on the FP32 norm output, both in FP32.
    assert seen == [
        (torch.float32, torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32, torch.float32),
    ]
    assert tables == [torch.bfloat16] and out.dtype == torch.float32


def test_tiles_reach_the_decoder_device_through_the_shared_tiled_decode():
    config = _video_config()
    config.update(clip_length=17, token_drop=3)
    seen = []

    class Decoder(torch.nn.Module):
        def forward(self, tile):
            seen.append(tile.shape)
            b, _, t, h, w = tile.shape
            return torch.zeros(b, 3, t * 4, h * 16, w * 16)

    latents = torch.randn(1, 24, 9, 16, 32)
    video = decode_minimax_h3_video(
        Decoder(), latents, latents_mean=[0.0] * 24, latents_std=[1.0] * 24
    )
    assert video.shape == (1, 3, 30, 256, 512) and video.device.type == "cpu"
    assert all(shape[-2:] == (16, 16) for shape in seen)
