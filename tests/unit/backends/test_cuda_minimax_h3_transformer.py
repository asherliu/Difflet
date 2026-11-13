"""MiniMax-H3's CUDA transformer on CPU: the checkpoint lifecycle, the CUDA arithmetic it
installs after loading (``numerics``) and its compile policy. The fused ops run their plain
formulas as CPU kernels (``minimax_h3_reference_ops``)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from difflet.backends.cuda.minimax_h3.numerics import (
    CudaMiniMaxH3AttnProcessor,
    install_cuda_numerics,
)
from difflet.backends.cuda.ops_impl import parallel_mesh
from difflet.models.minimax_h3.modeling_minimax_h3 import MiniMaxH3Transformer3DModel
from tests.unit.backends import minimax_h3_reference_ops as reference

TINY = dict(
    num_attention_heads=2,
    attention_head_dim=32,
    hidden_size=64,
    num_layers=2,
    num_refiner_layers=1,
    ffn_dim=64,
    in_channels=2,
    audio_in_channels=3,
    patch_size=(1, 1, 1),
    text_dim=4,
    freq_dim=4,
    time_embed_hidden_dim=16,
    time_embed_dim=8,
    rope_freq_dim=4,
)
FP32_MODULES = ("proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out")


@pytest.fixture(autouse=True)
def _cpu_ops_and_mesh():
    reference.register_cpu_kernels()
    parallel_mesh.destroy_parallel_mesh()
    yield
    parallel_mesh.destroy_parallel_mesh()


def _model(seed=0):
    torch.manual_seed(seed)
    model = MiniMaxH3Transformer3DModel(**TINY).eval()
    frequencies = model.rope.inv_freq.clone()
    model.to(torch.bfloat16)
    for name in FP32_MODULES:
        model.get_submodule(name).float()
    model.rope.inv_freq = frequencies  # the real FP32 frequencies, not BF16-rounded ones
    return model


def _checkpoint(path):
    model = _model()
    model.save_pretrained(path, max_shard_size="4KB")
    return model


def _inputs():
    torch.manual_seed(1)
    return (
        torch.randn(1, 2, 2),  # video rows, FP32
        torch.randn(1, 2, 3),  # audio rows, FP32
        torch.randn(1, 2, 4, dtype=torch.bfloat16),  # text
        torch.tensor([0.3, 0.7]),  # the step's unique timesteps
        torch.tensor([1, 0, 0, 1, 1, 1]),  # each row's entry
        torch.tensor([1, 2, 2, 0, 0, 1]),  # token tags
        torch.arange(6).unsqueeze(1).expand(-1, 3).float(),
        torch.tensor([3, 4]),
        torch.tensor([1, 2]),
        torch.tensor([0, 5]),
        None,
        None,
    )


def _component(path, **kwargs):
    from difflet.backends.cuda.minimax_h3.transformer import CudaMiniMaxH3TransformerApplication

    return CudaMiniMaxH3TransformerApplication(model_path=str(path), device="cpu", **kwargs)


def _loaded(path, **kwargs):
    component = _component(path, **kwargs)
    component.compile(path / "artifacts")
    component.load(path / "artifacts")
    return component


def test_loads_the_sharded_checkpoint_with_its_fp32_modules(tmp_path):
    original = _checkpoint(tmp_path)
    module = _loaded(tmp_path).module
    for name, tensor in original.state_dict().items():
        loaded = module.state_dict()[name]
        assert loaded.dtype == tensor.dtype and torch.equal(loaded, tensor), name
    assert module.proj_in.weight.dtype == torch.float32
    assert module.transformer_blocks[0].attn.to_q.weight.dtype == torch.bfloat16
    assert torch.equal(module.rope.inv_freq, original.rope.inv_freq)


def test_a_missing_checkpoint_tensor_fails_the_load(tmp_path):
    from safetensors.torch import load_file, save_file

    _checkpoint(tmp_path)
    shard = sorted(tmp_path.glob("*.safetensors"))[0]
    tensors = load_file(shard)
    tensors.pop(next(iter(tensors)))
    save_file(tensors, shard)
    with pytest.raises(Exception):
        _loaded(tmp_path)


def test_the_cuda_arithmetic_is_installed_after_loading(tmp_path):
    _checkpoint(tmp_path)
    module = _loaded(tmp_path).module
    attentions = [b.attn for b in module.token_refiner.refiner_blocks]
    attentions += [b.attn for b in module.transformer_blocks]
    assert all(isinstance(a.processor, CudaMiniMaxH3AttnProcessor) for a in attentions)
    norms = [m for m in module.modules() if isinstance(m, nn.RMSNorm)]
    assert norms and all("forward" in vars(m) for m in norms)
    assert all("forward" in vars(b) for b in module.transformer_blocks)
    assert "forward" in vars(module.norm_out)


def _counting(monkeypatch):
    counts = {}
    for name in ("rms_norm_modulate", "gated_residual", "head_norm_rotary", "swiglu"):
        op = getattr(torch.ops.difflet_h3, name)

        def wrapped(*args, _op=op, _name=name):
            counts[_name] = counts.get(_name, 0) + 1
            return _op(*args)

        monkeypatch.setattr(torch.ops.difflet_h3, name, wrapped, raising=False)
    return counts


def test_each_main_block_runs_the_fused_ops(monkeypatch):
    model = _model()
    install_cuda_numerics(model)
    counts = _counting(monkeypatch)
    with torch.no_grad():
        model(*_inputs(), return_dict=False)
    blocks = TINY["num_layers"]
    assert counts == {
        "rms_norm_modulate": 2 * blocks,
        "gated_residual": 2 * blocks,
        "head_norm_rotary": 2 * blocks,  # Q and K; the refiner has no rotary embedding
        "swiglu": blocks,
    }


def test_the_cuda_arithmetic_tracks_an_fp32_forward():
    """Not bit-exact (BF16 stores), but the same function: wiring errors such as swapped
    shift/scale or a dropped gate would be far outside this tolerance."""
    model = _model()
    exact = MiniMaxH3Transformer3DModel(**TINY).eval()
    exact.load_state_dict({k: v.float() for k, v in model.state_dict().items()})
    exact.rope.inv_freq = model.rope.inv_freq
    install_cuda_numerics(model)
    inputs = _inputs()
    with torch.no_grad():
        got = model(*inputs, return_dict=False)
        want = exact(*(t.float() if t is inputs[2] else t for t in inputs), return_dict=False)
    for g, w in zip(got, want):
        torch.testing.assert_close(g.float(), w.float(), atol=0.05, rtol=0.05)


def test_fp32_rms_norm_multiplies_the_weight_before_one_cast():
    norm = nn.RMSNorm(8, eps=1e-6).to(torch.bfloat16)
    nn.init.normal_(norm.weight)
    model = _model()
    model.norm_out.norm = norm
    install_cuda_numerics(model)
    x = torch.randn(5, 8, dtype=torch.bfloat16)
    value = x.float()
    want = norm.weight.float() * (value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + 1e-6))
    assert torch.equal(norm(x), want.to(torch.bfloat16))


def test_the_refiner_blocks_compile_whole_with_the_norm_op():
    model = _model()
    install_cuda_numerics(model)
    block = model.token_refiner.refiner_blocks[0]
    compiled = torch.compile(block, backend="aot_eager", fullgraph=True, dynamic=True)
    hidden = torch.randn(1, 5, TINY["hidden_size"], dtype=torch.bfloat16)
    with torch.no_grad():
        assert torch.equal(compiled(hidden), block(hidden))


def test_the_blocks_compile_whole_with_the_custom_ops():
    model = _model()
    install_cuda_numerics(model)
    block = model.transformer_blocks[0]
    compiled = torch.compile(block, backend="aot_eager", fullgraph=True, dynamic=True)
    hidden = torch.randn(1, 6, TINY["hidden_size"], dtype=torch.bfloat16)
    temb = torch.randn(2, TINY["time_embed_dim"])
    indices = torch.tensor([3, 0, 3, 4, 4, 1])
    cos, sin = torch.randn(6, 24), torch.randn(6, 24)
    with torch.no_grad():
        assert torch.equal(
            compiled(hidden, temb, indices, (cos, sin)), block(hidden, temb, indices, (cos, sin))
        )


@pytest.mark.parametrize("torch_compile", [False, True])
def test_compile_wraps_the_refiner_and_main_blocks_in_inductor_default_mode(
    tmp_path, monkeypatch, torch_compile
):
    _checkpoint(tmp_path)
    calls = []
    monkeypatch.setattr(torch, "compile", lambda block, **kw: calls.append(kw) or block)
    _loaded(tmp_path, torch_compile=torch_compile)
    if not torch_compile:
        assert calls == []
        return
    assert len(calls) == TINY["num_refiner_layers"] + TINY["num_layers"]
    options = torch._inductor.list_mode_options("max-autotune-no-cudagraphs")
    options.update(
        emulate_precision_casts=True, max_autotune=False, coordinate_descent_tuning=False
    )
    assert all(c == {"dynamic": True, "fullgraph": False, "options": options} for c in calls)
