"""Numerical coverage of the CUDA dense MMQ backend against dequantized references.

Weights are real tensors taken from a GGUF checkpoint (``GGUF_CUDA_MMQ_TEST_MODEL``,
default: the Qwen3.8-27B UD-Q4_K_XL file). The reference dequantizer is the
``gguf`` package's NumPy implementation, which is independent of both this
package and the transformers GGUF fork.

References are computed in exact FP32 math (TF32 disabled), not with cuBLAS
BF16 GEMMs: with PyTorch's default reduced-precision BF16 reduction, cuBLAS was
measured to be *less* accurate than this kernel (NRMSE 2.35e-3 vs 1.66e-3
against exact math), so it is not a usable oracle at this resolution.

* ``bf16`` reference: the weight dequantized to FP32 and rounded once to BF16
  (exactly what the kernel stages in shared memory), multiplied in FP32.
* ``fp32`` reference: the FP32 dequantized weight, multiplied in FP32.

Tolerances, as normalized RMSE = ||actual - ref|| / ||ref||:

* Per element vs ``bf16``: at most one BF16 ulp (8 significant bits) of the
  exact value, plus ``CANCEL_SLACK`` = 1e-4 x RMS(reference) absolute for
  outputs near zero, where FP32 summation-order differences exceed the ulp at
  that magnitude. A correctly rounded result is within half an ulp; one ulp
  admits near-tie rounding flips. This criterion does not dilute with output
  size, unlike NRMSE.
* vs ``bf16``: ``BF16_NRMSE`` = 2e-3. The unavoidable output-rounding floor
  was measured at 1.63e-3..1.69e-3 for Q4_K/Q5_K/Q6_K/Q8_0 (random normal inputs);
  an indexing or scale bug produces errors of order 1.
* vs ``fp32``: ``FP32_NRMSE`` = 5e-3. Rounding decoded weights to BF16 adds
  another ~1.65e-3 (measured) on top of the output rounding.
"""

import glob
import os
from functools import cache

import gguf
import numpy as np
import pytest
import torch

import torch_ggml_ops
from torch_ggml_ops.runtime_contract import quant_workspace_elements

torch.backends.cuda.matmul.allow_tf32 = False

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="CUDA backend only",
)

CANCEL_SLACK = 1e-4
BF16_NRMSE = 2e-3
FP32_NRMSE = 5e-3

_DEFAULT_MODEL = (
    "~/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/*/"
    "Qwen3.8-27B-UD-Q4_K_XL.gguf"
)

# One real tensor per quant type; (name, rows to take). Row counts are chosen to
# leave partial 128-row tiles.
_TENSORS = {
    "Q4_K": "blk.0.ffn_gate.weight",
    "Q5_K": "blk.0.ffn_down.weight",
    "Q6_K": "output.weight",
    "Q8_0": "blk.0.attn_v.weight",
    "IQ4_NL": "blk.0.ffn_up.weight",
    "IQ4_XS": "blk.0.ffn_gate.weight",
    "Q3_K": "blk.0.ffn_up.weight",
    "IQ3_S": "blk.0.ffn_down.weight",
}


@cache
def _reader() -> gguf.GGUFReader:
    pattern = os.path.expanduser(
        os.environ.get("GGUF_CUDA_MMQ_TEST_MODEL", _DEFAULT_MODEL)
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        pytest.skip(f"GGUF test model is unavailable: {pattern}")
    return gguf.GGUFReader(matches[0])


@cache
def _tensor_by_type(quant_name: str) -> gguf.ReaderTensor:
    reader = _reader()
    wanted = gguf.GGMLQuantizationType[quant_name]
    preferred = _TENSORS[quant_name]
    tensors = [t for t in reader.tensors if t.tensor_type == wanted and t.data.ndim == 2]
    if not tensors:
        pytest.skip(f"model has no 2D {quant_name} tensor")
    for tensor in tensors:
        if tensor.name == preferred:
            return tensor
    return max(tensors, key=lambda t: t.data.shape[0])


def _weights(quant_name: str, rows: int, row_offset: int = 0):
    tensor = _tensor_by_type(quant_name)
    packed_np = np.ascontiguousarray(tensor.data[row_offset : row_offset + rows])
    rows = packed_np.shape[0]
    dense = gguf.quants.dequantize(packed_np, tensor.tensor_type)
    in_features = dense.shape[-1]
    packed = torch.from_numpy(packed_np.copy()).cuda()
    w32 = torch.from_numpy(np.ascontiguousarray(dense, dtype=np.float32)).cuda()
    return packed, int(tensor.tensor_type), rows, in_features, w32


def _nrmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a = actual.float()
    e = expected.float()
    assert torch.isfinite(a).all()
    return ((a - e).square().mean().sqrt() / e.square().mean().sqrt()).item()


def _randn(*shape: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, generator=g, device="cuda", dtype=torch.bfloat16)


CASES = [
    (q, m, n)
    for q in _TENSORS
    for (m, n) in ((1, 37), (37, 300), (300, 129), (1000, 512))
]


@pytest.mark.parametrize("quant_name,rows,out_features", CASES)
def test_forward_and_input_gradient(quant_name, rows, out_features):
    packed, quant_type, n, k, w32 = _weights(quant_name, out_features, row_offset=7)
    w16 = w32.to(torch.bfloat16)
    x = _randn(rows, k, seed=rows * 7 + n).requires_grad_(True)
    grad = _randn(rows, n, seed=rows * 11 + n)

    y = torch_ggml_ops.mmq(x, packed, quant_type, n)
    y.backward(grad)

    with torch.no_grad():
        y_bf16 = x.float() @ w16.float().T
        y_fp32 = x.float() @ w32.T
        gx_bf16 = grad.float() @ w16.float()
        gx_fp32 = grad.float() @ w32

    assert y.shape == (rows, n) and y.dtype == torch.bfloat16
    assert x.grad.shape == (rows, k) and x.grad.dtype == torch.bfloat16
    _check(y, y_bf16, y_fp32)
    _check(x.grad, gx_bf16, gx_fp32)


def _bf16_ulp(values: torch.Tensor) -> torch.Tensor:
    """Spacing of BF16 numbers (8 significant bits) at each magnitude."""
    exponent = torch.floor(torch.log2(values.abs().clamp_min(torch.finfo(torch.float32).tiny)))
    return torch.exp2(exponent - 7)


def _check(actual, bf16_ref, fp32_ref):
    a = actual.float()
    # Per element: within one BF16 ulp of the exact value. The slack of
    # CANCEL_SLACK x RMS covers outputs near zero, where FP32 summation-order
    # differences exceed the (tiny) ulp at that magnitude.
    excess = (a - bf16_ref).abs() - _bf16_ulp(bf16_ref)
    worst = (excess / bf16_ref.square().mean().sqrt()).max().item()
    errors = (worst, _nrmse(actual, bf16_ref), _nrmse(actual, fp32_ref))
    assert worst <= CANCEL_SLACK, errors
    assert errors[1] < BF16_NRMSE, errors
    assert errors[2] < FP32_NRMSE, errors


@pytest.mark.parametrize("quant_name", list(_TENSORS))
def test_tolerance_rejects_a_single_corrupted_block(quant_name):
    """Negative control: the criteria must catch one wrong super-block scale."""
    packed, quant_type, n, k, w32 = _weights(quant_name, 64)
    corrupted = packed.clone()
    # The fp16 super-block scale "d" is little-endian at bytes 0-1 (Q4_K, Q5_K,
    # Q8_0, IQ4_NL, IQ4_XS, IQ3_S) or 208-209 (Q6_K) or 108-109 (Q3_K) of a block. XOR 0x04 on its high byte flips the
    # lowest exponent bit, i.e. doubles or halves one block's scale in one row.
    offset = {"Q6_K": 209, "Q3_K": 109}.get(quant_name, 1)
    corrupted[5, offset] ^= 0x04
    x = _randn(16, k, seed=12)
    y = torch_ggml_ops.mmq(x, corrupted, quant_type, n)
    w16 = w32.to(torch.bfloat16)
    with pytest.raises(AssertionError):
        _check(y, x.float() @ w16.float().T, x.float() @ w32.T)


@pytest.mark.parametrize("quant_name", list(_TENSORS))
def test_full_width_projection(quant_name):
    """A full real projection (all rows of the tensor, capped at 6144)."""
    tensor = _tensor_by_type(quant_name)
    packed, quant_type, n, k, w32 = _weights(quant_name, min(tensor.data.shape[0], 6144))
    w16 = w32.to(torch.bfloat16)
    x = _randn(2, 256, k, seed=5).requires_grad_(True)
    grad = _randn(2, 256, n, seed=6)
    y = torch_ggml_ops.mmq(x, packed, quant_type, n)
    y.backward(grad)
    with torch.no_grad():
        _check(y, x.float() @ w16.float().T, x.float() @ w32.T)
        _check(x.grad, grad.float() @ w16.float(), grad.float() @ w32)


def test_inplace_entry_points_match_autograd():
    packed, quant_type, n, k, _ = _weights("Q6_K", 256)
    x = _randn(64, k, seed=1)
    g = _randn(64, n, seed=2)
    out = torch.empty(64, n, device="cuda", dtype=torch.bfloat16)
    ws = torch.empty(quant_workspace_elements(x.numel()), device="cuda", dtype=torch.uint8)
    assert ws.numel() == 0
    torch_ggml_ops.mmq_inplace(x, packed, quant_type, n, out, ws)
    gx = torch.empty(64, k, device="cuda", dtype=torch.bfloat16)
    torch_ggml_ops.mmq_grad_input_inplace(g, packed, quant_type, k, gx)
    xr = x.clone().requires_grad_(True)
    yr = torch_ggml_ops.mmq(xr, packed, quant_type, n)
    yr.backward(g)
    torch.testing.assert_close(out, yr, rtol=0, atol=0)
    torch.testing.assert_close(gx, xr.grad, rtol=0, atol=0)


def test_repeatable_and_input_dependent():
    packed, quant_type, n, k, _ = _weights("Q4_K", 200)
    x = _randn(130, k, seed=3)
    a = torch_ggml_ops.mmq(x, packed, quant_type, n)
    b = torch_ggml_ops.mmq(x, packed, quant_type, n)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    x2 = x.clone()
    x2[129].neg_()
    c = torch_ggml_ops.mmq(x2, packed, quant_type, n)
    torch.testing.assert_close(a[:129], c[:129], rtol=0, atol=0)
    torch.testing.assert_close(c[129], -a[129], rtol=0, atol=0)


def test_compiles_fullgraph():
    packed, quant_type, n, _, _ = _weights("Q4_K", 64)
    x = _randn(48, packed.shape[1] // 144 * 256, seed=4)

    @torch.compile(fullgraph=True)
    def f(x, packed):
        return torch_ggml_ops.mmq(x, packed, quant_type, n)

    torch.testing.assert_close(f(x, packed), torch_ggml_ops.mmq(x, packed, quant_type, n), rtol=0, atol=0)


def test_exported_type_set_matches_the_backend():
    assert torch_ggml_ops.DENSE_MMQ_QUANT_TYPES == {
        int(gguf.GGMLQuantizationType[name]) for name in _TENSORS
    }


def test_unsupported_quant_type_fails_clearly():
    k = 256
    packed = torch.zeros(4, k // 256 * 84, dtype=torch.uint8, device="cuda")  # Q2_K geometry
    x = _randn(8, k, seed=9)
    with pytest.raises(RuntimeError, match="not implemented by the CUDA MMQ backend"):
        torch_ggml_ops.mmq(x, packed, int(gguf.GGMLQuantizationType.Q2_K), 4)


def test_grouped_ops_fail_clearly():
    x = _randn(4, 256, seed=1)
    with pytest.raises(RuntimeError, match="not implemented by the CUDA backend"):
        torch_ggml_ops.grouped_mmq(
            x,
            torch.zeros(1, dtype=torch.uint8, device="cuda"),
            torch.zeros(1, dtype=torch.int64, device="cuda"),
            torch.zeros(2, dtype=torch.int32, device="cuda"),
            12,
            4,
        )
