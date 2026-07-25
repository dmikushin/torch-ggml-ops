import os
from pathlib import Path

import gguf
import numpy as np
import pytest
import torch
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor
from transformers.integrations.moe import _grouped_linear

import torch_ggml_ops

_MODEL = Path(
    os.environ.get(
        "GGUF_DEEPSEEK_MMQ_TEST_MODEL",
        os.path.expanduser("~/models/ds4/DeepSeek-V4-Flash-IQ2XXS.gguf"),
    )
)
_ROUTED_PROJECTIONS = {
    "IQ2_XXS": "blk.0.ffn_gate_exps.weight",
    "Q2_K": "blk.0.ffn_down_exps.weight",
}


@pytest.fixture(scope="module")
def reader() -> gguf.GGUFReader:
    if not _MODEL.is_file():
        pytest.skip("DeepSeek-V4-Flash GGUF model is unavailable")
    return gguf.GGUFReader(_MODEL)


def _tensor(reader: gguf.GGUFReader, name: str) -> gguf.ReaderTensor:
    return next(tensor for tensor in reader.tensors if tensor.name == name)


def _packed_experts(
    reader: gguf.GGUFReader,
    tensor_name: str,
    *,
    num_experts: int = 8,
    out_features: int = 37,
) -> tuple[torch.Tensor, gguf.GGMLQuantizationType, int]:
    tensor = _tensor(reader, tensor_name)
    host = np.array(
        tensor.data[:num_experts, :out_features],
        dtype=np.uint8,
        copy=True,
        order="C",
    )
    return (
        torch.from_numpy(host).to("cuda"),
        tensor.tensor_type,
        int(tensor.shape[0]),
    )


def _routing() -> tuple[torch.Tensor, torch.Tensor]:
    experts = torch.tensor([0, 2, 5, 7], device="cuda", dtype=torch.int64)
    offsets = torch.tensor([1, 4, 8, 10], device="cuda", dtype=torch.int32)
    return experts, offsets


def _reference_grouped(
    input: torch.Tensor,
    packed: torch.Tensor,
    experts: torch.Tensor,
    offsets: torch.Tensor,
    quant_type: gguf.GGMLQuantizationType,
    out_features: int,
) -> torch.Tensor:
    logical = dequantize_gguf_tensor(
        packed.index_select(0, experts),
        quant_type,
        dtype=input.dtype,
        device=input.device,
    ).reshape(experts.numel(), out_features, input.shape[1])
    return _grouped_linear(input, logical, offsets)


def _reference_grouped_grad_input(
    grad_output: torch.Tensor,
    packed: torch.Tensor,
    experts: torch.Tensor,
    offsets: torch.Tensor,
    quant_type: gguf.GGMLQuantizationType,
    in_features: int,
    out_features: int,
) -> torch.Tensor:
    logical = dequantize_gguf_tensor(
        packed.index_select(0, experts),
        quant_type,
        dtype=grad_output.dtype,
        device=grad_output.device,
    ).reshape(experts.numel(), out_features, in_features)
    expected = torch.empty(
        grad_output.shape[0], in_features, device=grad_output.device, dtype=grad_output.dtype
    )
    for group, (row_start, row_end) in enumerate(
        zip((0, *offsets[:-1].tolist()), offsets.tolist(), strict=True)
    ):
        expected[row_start:row_end] = grad_output[row_start:row_end] @ logical[group]
    return expected


def _reference_fixed_grad_input(
    grad_output: torch.Tensor,
    packed: torch.Tensor,
    quant_type: gguf.GGMLQuantizationType,
    in_features: int,
) -> torch.Tensor:
    expected = torch.empty(
        *grad_output.shape[:-1], in_features,
        device=grad_output.device,
        dtype=grad_output.dtype,
    )
    for group in range(8):
        logical = dequantize_gguf_tensor(
            packed[group],
            quant_type,
            dtype=grad_output.dtype,
            device=grad_output.device,
        ).reshape(grad_output.shape[-1], in_features)
        expected[:, group] = grad_output[:, group] @ logical
    return expected


def _assert_q8_activation_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    error = actual.float() - expected.float()
    normalized_rmse = (
        error.square().mean().sqrt() / expected.float().square().mean().sqrt()
    )
    assert torch.isfinite(actual).all()
    assert normalized_rmse.item() < 0.04


@pytest.mark.parametrize("qname", tuple(_ROUTED_PROJECTIONS))
def test_deepseek_routed_forward_matches_transformers_dequantization(
    reader: gguf.GGUFReader,
    qname: str,
) -> None:
    packed, quant_type, in_features = _packed_experts(
        reader, _ROUTED_PROJECTIONS[qname]
    )
    assert quant_type.name == qname
    experts, offsets = _routing()
    generator = torch.Generator(device="cuda").manual_seed(1234)
    input = torch.randn(
        10,
        in_features,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )

    expected = _reference_grouped(input, packed, experts, offsets, quant_type, 37)
    actual = torch_ggml_ops.grouped_mmq(
        input, packed, experts, offsets, int(quant_type), 37
    )

    assert actual.shape == (10, 37)
    assert actual.dtype == torch.bfloat16
    assert actual.is_contiguous()
    _assert_q8_activation_error(actual, expected)


def test_deepseek_iq2_xxs_pair_matches_transformers_dequantization(
    reader: gguf.GGUFReader,
) -> None:
    gate, quant_type, in_features = _packed_experts(
        reader, "blk.0.ffn_gate_exps.weight"
    )
    up, up_quant_type, up_in_features = _packed_experts(
        reader, "blk.0.ffn_up_exps.weight"
    )
    assert quant_type.name == "IQ2_XXS"
    assert up_quant_type == quant_type
    assert up_in_features == in_features == 4096
    experts, offsets = _routing()
    generator = torch.Generator(device="cuda").manual_seed(5678)
    input = torch.randn(
        10,
        in_features,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )

    expected_gate = _reference_grouped(input, gate, experts, offsets, quant_type, 37)
    expected_up = _reference_grouped(input, up, experts, offsets, quant_type, 37)
    actual_gate, actual_up = torch_ggml_ops.grouped_mmq_pair(
        input, gate, up, experts, offsets, int(quant_type), 37
    )

    _assert_q8_activation_error(actual_gate, expected_gate)
    _assert_q8_activation_error(actual_up, expected_up)


@pytest.mark.parametrize("qname", tuple(_ROUTED_PROJECTIONS))
def test_deepseek_routed_backward_matches_transformers_dequantization(
    reader: gguf.GGUFReader,
    qname: str,
) -> None:
    packed, quant_type, in_features = _packed_experts(
        reader, _ROUTED_PROJECTIONS[qname]
    )
    experts, offsets = _routing()
    generator = torch.Generator(device="cuda").manual_seed(4321)
    grad_output = torch.randn(
        10, 37, generator=generator, device="cuda", dtype=torch.bfloat16
    )

    expected = _reference_grouped_grad_input(
        grad_output,
        packed,
        experts,
        offsets,
        quant_type,
        in_features,
        37,
    )
    actual = torch.ops.torch_ggml_ops.grouped_mmq_grad_input.default(
        grad_output,
        packed,
        experts,
        offsets,
        int(quant_type),
        in_features,
    )

    assert actual.shape == (10, in_features)
    assert actual.dtype == torch.bfloat16
    _assert_q8_activation_error(actual, expected)

    input = torch.randn(
        10, in_features, generator=generator, device="cuda", dtype=torch.bfloat16,
        requires_grad=True,
    )
    torch_ggml_ops.grouped_mmq(
        input, packed, experts, offsets, int(quant_type), 37
    ).backward(grad_output)
    _assert_q8_activation_error(input.grad, expected)


def test_deepseek_iq2_xxs_pair_backward_matches_transformers_dequantization(
    reader: gguf.GGUFReader,
) -> None:
    gate, quant_type, in_features = _packed_experts(
        reader, "blk.0.ffn_gate_exps.weight"
    )
    up, up_quant_type, up_in_features = _packed_experts(
        reader, "blk.0.ffn_up_exps.weight"
    )
    assert up_quant_type == quant_type
    assert up_in_features == in_features
    experts, offsets = _routing()
    generator = torch.Generator(device="cuda").manual_seed(6789)
    first_grad = torch.randn(
        10, 37, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    second_grad = torch.randn(
        10, 37, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    gate_logical = dequantize_gguf_tensor(
        gate.index_select(0, experts), quant_type,
        dtype=torch.bfloat16, device="cuda",
    ).reshape(experts.numel(), 37, in_features)
    up_logical = dequantize_gguf_tensor(
        up.index_select(0, experts), quant_type,
        dtype=torch.bfloat16, device="cuda",
    ).reshape(experts.numel(), 37, in_features)
    expected = torch.empty(10, in_features, device="cuda", dtype=torch.bfloat16)
    for group, (row_start, row_end) in enumerate(
        zip((0, *offsets[:-1].tolist()), offsets.tolist(), strict=True)
    ):
        expected[row_start:row_end] = (
            first_grad[row_start:row_end] @ gate_logical[group] +
            second_grad[row_start:row_end] @ up_logical[group]
        )

    actual = torch.ops.torch_ggml_ops.grouped_mmq_pair_grad_input.default(
        first_grad, second_grad, gate, up, experts, offsets, int(quant_type), in_features
    )
    _assert_q8_activation_error(actual, expected)

    input = torch.randn(
        10, in_features, generator=generator, device="cuda", dtype=torch.bfloat16,
        requires_grad=True,
    )
    first, second = torch_ggml_ops.grouped_mmq_pair(
        input, gate, up, experts, offsets, int(quant_type), 37
    )
    torch.autograd.backward((first, second), (first_grad, second_grad))
    _assert_q8_activation_error(input.grad, expected)


@pytest.mark.parametrize("out_features", [37, 1024])
def test_deepseek_fixed_q8_0_forward_matches_transformers_dequantization(
    reader: gguf.GGUFReader,
    out_features: int,
) -> None:
    tensor = _tensor(reader, "blk.0.attn_output_a.weight")
    assert tensor.tensor_type.name == "Q8_0"
    assert tuple(tensor.data.shape) == (8192, 4352)
    grouped_data = tensor.data.reshape(8, 1024, 4352)
    packed = torch.from_numpy(
        np.array(
            grouped_data[:, :out_features],
            dtype=np.uint8,
            copy=True,
            order="C",
        )
    ).to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(9012)
    input = torch.randn(
        2,
        8,
        4096,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )

    expected = torch.empty(2, 8, out_features, device="cuda", dtype=torch.bfloat16)
    for group in range(8):
        logical = dequantize_gguf_tensor(
            packed[group],
            tensor.tensor_type,
            dtype=input.dtype,
            device=input.device,
        ).reshape(out_features, 4096)
        expected[:, group] = input[:, group] @ logical.T

    actual = torch_ggml_ops.fixed_grouped_mmq(input, packed)

    assert actual.shape == (2, 8, out_features)
    assert actual.dtype == torch.bfloat16
    assert actual.is_contiguous()
    _assert_q8_activation_error(actual, expected)


def test_deepseek_fixed_q8_0_opcheck_compile_and_backward(
    reader: gguf.GGUFReader,
) -> None:
    tensor = _tensor(reader, "blk.0.attn_output_a.weight")
    grouped_data = tensor.data.reshape(8, 1024, 4352)
    packed = torch.from_numpy(
        np.array(grouped_data[:, :37], dtype=np.uint8, copy=True, order="C")
    ).to("cuda")
    input = torch.randn(
        1,
        8,
        4096,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    result = torch.library.opcheck(
        torch.ops.torch_ggml_ops.fixed_grouped_mmq.default,
        (input.detach(), packed),
        test_utils=(
            "test_schema",
            "test_autograd_registration",
            "test_faketensor",
            "test_aot_dispatch_dynamic",
        ),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result

    @torch.compile(fullgraph=True)
    def compiled(input: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
        return torch_ggml_ops.fixed_grouped_mmq(input, packed)

    expected = torch_ggml_ops.fixed_grouped_mmq(input.detach(), packed)
    actual = compiled(input.detach(), packed)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    grad_output = torch.randn(
        1, 8, 37, device="cuda", dtype=torch.bfloat16, generator=torch.Generator(device="cuda").manual_seed(3456)
    )
    expected_grad = _reference_fixed_grad_input(
        grad_output, packed, tensor.tensor_type, 4096
    )
    actual_grad = torch.ops.torch_ggml_ops.fixed_grouped_mmq_grad_input.default(
        grad_output, packed
    )
    _assert_q8_activation_error(actual_grad, expected_grad)

    input = input.detach().requires_grad_()
    torch_ggml_ops.fixed_grouped_mmq(input, packed).backward(grad_output)
    _assert_q8_activation_error(input.grad, expected_grad)
