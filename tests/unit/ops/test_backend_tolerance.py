"""One tolerance table for the op surface, run against every backend present on the host.

Level 1 of docs/plans/2026-10-21-cuda-generic-layer.md. Each row names an op, a dtype, its
reference and what "equal" means for it: ``exact`` rows compare with ``torch.equal`` and also
assert dtype, shape and strides (``torch.equal`` checks neither); ``tolerance`` rows use the
row's ``atol``/``rtol`` against the CPU reference backend computed in fp32 on the same seeded
inputs. A row whose tolerance is ``None`` has not been measured on a device yet and is
``xfail(strict=True)``: filling it in flips the test, so an unmeasured gap cannot be forgotten.

The fp32 tolerances are the ones ``tests/unit/backends/test_cuda_*.py`` already use. The bf16
rows are filled from a GPU run of this file with ``DIFFLET_TOLERANCE_REPORT=1``, which prints
the measured max abs and rel error per row instead of asserting; the tolerance is then the
smallest power of ten above the measured error (Step A of the design).

Backends: ``cpu`` is the reference and always runs. ``cuda`` always runs too — its ops are
plain torch and run on CPU tensors when no GPU is present (the existing CUDA unit tests rely
on that) — and runs on the GPU when one is available, where the attention rows also assert
which SDPA kernel torch picked (never ``math`` for a model shape: G11). The TPU backend is not
in the table yet: its ops need XLA tensors and no host in this series runs them.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Callable

import pytest
import torch

from difflet.pipeline.parallel_mesh import MeshSpec

REPORT = os.environ.get("DIFFLET_TOLERANCE_REPORT") == "1"


def _backend_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


BACKENDS = ("cpu", "cuda")
# Ops whose contract takes no fp32 input: their reference row is the cpu backend in bf16.
BF16_ONLY_OPS = frozenset({"quantize_mx"})


def _ops(backend: str, module: str):
    return importlib.import_module(f"difflet.backends.{backend}.ops_impl.{module}")


@pytest.fixture(autouse=True)
def _tp1_mesh():
    """The CUDA layers refuse to build without a mesh; tp=1 makes every collective the identity."""
    pm = _ops("cuda", "parallel_mesh")
    pm.destroy_parallel_mesh()
    pm.init_parallel_mesh(MeshSpec(tp=1))
    yield
    pm.destroy_parallel_mesh()


# ----------------------------------------------------------------------------- the inputs


def _seeded(*shape, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def _qkv(b=1, h=2, s=4, d=8):
    return tuple(_seeded(b, h, s, d, seed=i) for i in range(3))


# Model shapes the attention rows assert the kernel on when a GPU is present: Qwen-Image's
# joint sequence (text 1024 + image 4096 at 1024², 24 heads of 128), Flux's (512 + 4096),
# Wan's (480×832×9 → 3510 tokens, 40 heads of 128).
MODEL_ATTENTION_SHAPES = {
    "qwen_image": (1, 24, 1024 + 4096, 128),
    "flux": (1, 24, 512 + 4096, 128),
    "wan": (1, 40, 3510, 128),
}


# ------------------------------------------------------------------------------- the rows


@dataclass(frozen=True)
class Row:
    op: str
    dtype: torch.dtype
    kind: str  # "exact" | "tolerance"
    atol: float | None
    rtol: float | None
    note: str

    @property
    def id(self) -> str:
        return f"{self.op}-{str(self.dtype).split('.')[-1]}"


TOLERANCE = [
    Row("attention", torch.float32, "tolerance", 1e-5, 0.0, "SDPA vs the CPU softmax"),
    Row("attention", torch.bfloat16, "tolerance", None, None, "measured in Step A"),
    Row("cross_attention", torch.float32, "tolerance", 1e-5, 0.0, ""),
    Row(
        "attention_bounds",
        torch.float32,
        "tolerance",
        1e-5,
        0.0,
        "bound_min/bound_max key windows: HunyuanVideo's joint attention, (B*H, S, D) inputs",
    ),
    Row(
        "RMSNorm",
        torch.float32,
        "tolerance",
        1e-6,
        0.0,
        "fused F.rms_norm vs the seven-pass reference",
    ),
    Row("RMSNorm", torch.bfloat16, "tolerance", None, None, "measured in Step A"),
    Row("rms_norm_fp32", torch.float32, "tolerance", 1e-6, 0.0, ""),
    Row("LayerNorm", torch.float32, "tolerance", 1e-6, 0.0, ""),
    Row(
        "apply_rotary_emb",
        torch.float32,
        "exact",
        None,
        None,
        "pure torch, the same code on cpu, tpu and cuda",
    ),
    Row("ColumnParallelLinear", torch.float32, "tolerance", 1e-6, 0.0, "world size 1"),
    Row("RowParallelLinear", torch.float32, "tolerance", 1e-6, 0.0, "world size 1"),
    Row("ParallelEmbedding", torch.float32, "tolerance", 1e-6, 0.0, ""),
    Row(
        "quantize_mx",
        torch.bfloat16,
        "tolerance",
        None,
        None,
        "the op takes bf16/fp16 only; raises NotImplementedError on cuda, the cpu reference checks itself",
    ),
]


# ---------------------------------------------------------------------------- the cases
#
# Each case takes (backend name, dtype, device) and returns (backend output, fp32 CPU reference).
# The reference is always the cpu backend in fp32 on the same seeded inputs; the backend runs
# in the row's dtype on its device and the result is compared in fp32.


def _run_attention(backend, dtype, device, *, bounds=False, cross=False):
    q, k, v = _qkv()
    cpu_a = _ops("cpu", "attention")
    a = _ops(backend, "attention")
    if bounds:
        b, h, s, d = q.shape
        q3, k3, v3 = (t.reshape(b * h, s, d) for t in (q, k, v))
        bound_max = torch.full((b * h, s, 1), 3, dtype=torch.int32)  # keys [0, 3) valid
        bound_min = torch.zeros_like(bound_max)
        ref = cpu_a.attention(q3, k3, v3, scale=0.3, bound_min=bound_min, bound_max=bound_max)
        out = a.attention(
            q3.to(device, dtype),
            k3.to(device, dtype),
            v3.to(device, dtype),
            scale=0.3,
            bound_min=bound_min.to(device),
            bound_max=bound_max.to(device),
        )
        return out, ref
    fn_ref = cpu_a.cross_attention if cross else cpu_a.attention
    fn = a.cross_attention if cross else a.attention
    ref = fn_ref(q, k, v, scale=0.3)
    out = fn(q.to(device, dtype), k.to(device, dtype), v.to(device, dtype), scale=0.3)
    return out, ref


def _run_norm(backend, dtype, device, *, which: str):
    x = _seeded(3, 7, 32)
    weight = torch.rand(32, generator=torch.Generator().manual_seed(1)) + 0.5
    cpu_n = _ops("cpu", "norm")
    n = _ops(backend, "norm")
    if which == "rms_norm_fp32":
        ref = cpu_n.rms_norm_fp32(x, weight, 1e-6)
        out = n.rms_norm_fp32(x.to(device, dtype), weight.to(device, dtype), 1e-6)
        return out, ref
    cls_ref, cls = getattr(cpu_n, which), getattr(n, which)
    ref_m, m = cls_ref(32), cls(32)
    with torch.no_grad():
        for mod in (ref_m, m):
            mod.weight.copy_(weight)
            if getattr(mod, "bias", None) is not None:
                mod.bias.zero_()
    m = m.to(device, dtype)
    return m(x.to(device, dtype)), ref_m(x)


def _run_rotary(backend, dtype, device):
    x = _seeded(1, 2, 4, 8)
    cos = torch.cos(_seeded(4, 8, seed=3))
    sin = torch.sin(_seeded(4, 8, seed=3))
    ref = _ops("cpu", "embeddings").apply_rotary_emb(x, cos, sin)
    out = _ops(backend, "embeddings").apply_rotary_emb(
        x.to(device, dtype), cos.to(device, dtype), sin.to(device, dtype)
    )
    return out, ref


def _run_linear(backend, dtype, device, *, which: str):
    x = _seeded(2, 16) if which == "RowParallelLinear" else _seeded(2, 8)
    cpu_l = _ops("cpu", "linear")
    layer_l = _ops(backend, "linear")
    kwargs = {"input_is_parallel": True} if which == "RowParallelLinear" else {}
    in_f, out_f = (16, 8) if which == "RowParallelLinear" else (8, 16)
    ref_m = getattr(cpu_l, which)(in_f, out_f, bias=True, **kwargs)
    m = getattr(layer_l, which)(in_f, out_f, bias=True, **kwargs)
    weight = _seeded(out_f, in_f, seed=4)
    bias = _seeded(out_f, seed=5)
    with torch.no_grad():
        ref_m.weight.copy_(weight)
        ref_m.bias.copy_(bias)
        m.weight.copy_(weight)
        m.bias.copy_(bias)
    m = m.to(device, dtype)
    out = m(x.to(device, dtype))
    return out, ref_m(x)


def _run_embedding(backend, dtype, device):
    ids = torch.tensor([[1, 4, 9], [0, 2, 7]])
    weight = _seeded(10, 4, seed=6)
    ref_m = _ops("cpu", "linear").ParallelEmbedding(10, 4)
    m = _ops(backend, "linear").ParallelEmbedding(10, 4)
    with torch.no_grad():
        ref_m.weight.copy_(weight)
        m.weight.copy_(weight)
    m = m.to(device, dtype)
    return m(ids.to(device)), ref_m(ids)


def _run_quantize_mx(backend, dtype, device):
    x = _seeded(8, 32).to(torch.bfloat16)  # quantize_mx needs multiples of (8, 4)
    ref = _ops("cpu", "mx").quantize_mx(x)
    out = _ops(backend, "mx").quantize_mx(x.to(device, dtype))
    return out, ref


CASES: dict[str, Callable] = {
    "attention": _run_attention,
    "cross_attention": lambda b, d, dev: _run_attention(b, d, dev, cross=True),
    "attention_bounds": lambda b, d, dev: _run_attention(b, d, dev, bounds=True),
    "RMSNorm": lambda b, d, dev: _run_norm(b, d, dev, which="RMSNorm"),
    "rms_norm_fp32": lambda b, d, dev: _run_norm(b, d, dev, which="rms_norm_fp32"),
    "LayerNorm": lambda b, d, dev: _run_norm(b, d, dev, which="LayerNorm"),
    "apply_rotary_emb": _run_rotary,
    "ColumnParallelLinear": lambda b, d, dev: _run_linear(b, d, dev, which="ColumnParallelLinear"),
    "RowParallelLinear": lambda b, d, dev: _run_linear(b, d, dev, which="RowParallelLinear"),
    "ParallelEmbedding": _run_embedding,
    "quantize_mx": _run_quantize_mx,
}


def test_every_row_has_a_case_and_every_case_a_row():
    assert set(r.op for r in TOLERANCE) == set(CASES)


def _as_fp32_cpu(t):
    if isinstance(t, tuple):
        t = t[0]
    return t.detach().float().cpu()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("row", TOLERANCE, ids=lambda r: r.id)
def test_backend_meets_the_tolerance_table(backend, row):
    device = _backend_device(backend)
    if backend == "cpu" and row.dtype is not torch.float32 and row.op not in BF16_ONLY_OPS:
        pytest.skip("the reference backend runs the fp32 rows only")
    case = CASES[row.op]
    if row.op == "quantize_mx" and backend == "cuda":
        with pytest.raises(NotImplementedError):
            case(backend, row.dtype, device)
        return
    out, ref = case(backend, row.dtype, device)
    out_f, ref_f = _as_fp32_cpu(out), _as_fp32_cpu(ref)
    if row.op in BF16_ONLY_OPS and backend == "cpu":
        assert torch.equal(out_f, ref_f), f"{row.id}: the reference backend disagrees with itself"
        return
    assert out_f.shape == ref_f.shape, row
    if row.kind == "exact":
        out_t = out[0] if isinstance(out, tuple) else out
        ref_t = ref[0] if isinstance(ref, tuple) else ref
        assert out_t.dtype == row.dtype and out_t.stride() == ref_t.stride(), row
        assert torch.equal(out_f, ref_f), f"{row.id} on {backend}: not bit-equal"
        return
    err = (out_f - ref_f).abs()
    max_abs = err.max().item()
    max_rel = (err / ref_f.abs().clamp_min(1e-12)).max().item()
    if REPORT:
        print(f"\nTOLERANCE {backend:5s} {row.id:32s} max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
        return
    if row.atol is None:
        pytest.xfail(f"{row.id}: tolerance not measured yet ({row.note})")
    assert torch.allclose(out_f, ref_f, atol=row.atol, rtol=row.rtol), (
        f"{row.id} on {backend}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
        f"exceeds atol={row.atol} rtol={row.rtol}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernel choice is a GPU fact")
@pytest.mark.parametrize("model", sorted(MODEL_ATTENTION_SHAPES))
def test_cuda_attention_never_falls_back_to_the_math_kernel_at_model_shapes(model):
    """G11: a test that claims a fast path asserts that it ran."""
    platform = _ops("cuda", "platform")
    b, h, s, d = MODEL_ATTENTION_SHAPES[model]
    q, k, v = (torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    report = platform.sdpa_kernel_report(q, k, v)
    print(f"\nSDPA {model}: {report}")
    assert not report.startswith("math"), report
    assert not report.startswith("none"), report
