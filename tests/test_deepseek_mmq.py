import os
from pathlib import Path

import gguf
import pytest
import torch
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor

import torch_ggml_ops
from bench.mmq_benchmark_common import (
    DEEPSEEK_DENSE_TENSOR_CASES,
    DenseTensorCase,
    load_packed_tensor,
)

_MODEL = Path(
    os.environ.get(
        "GGUF_DEEPSEEK_MMQ_TEST_MODEL",
        os.path.expanduser("~/models/ds4/DeepSeek-V4-Flash-IQ2XXS.gguf"),
    )
)
_DENSE_CASES = DEEPSEEK_DENSE_TENSOR_CASES
_CASE_IDS = tuple(case.name for case in _DENSE_CASES)
_CHECKED_OUT_FEATURES = 129


@pytest.fixture(scope="module")
def reader() -> gguf.GGUFReader:
    if not _MODEL.is_file():
        pytest.skip("DeepSeek-V4-Flash GGUF model is unavailable")
    return gguf.GGUFReader(_MODEL)


def _tensor(reader: gguf.GGUFReader, name: str) -> gguf.ReaderTensor:
    return next(tensor for tensor in reader.tensors if tensor.name == name)


def _checked_weight(
    reader: gguf.GGUFReader,
    case: DenseTensorCase,
) -> tuple[torch.Tensor, torch.Tensor, gguf.GGMLQuantizationType]:
    tensor = _tensor(reader, case.tensor_name)
    packed = load_packed_tensor(tensor, _CHECKED_OUT_FEATURES)
    logical = dequantize_gguf_tensor(
        packed,
        tensor.tensor_type,
        dtype=torch.bfloat16,
        device="cuda",
    ).reshape(_CHECKED_OUT_FEATURES, case.in_features)
    return packed, logical, tensor.tensor_type


@pytest.mark.parametrize("case", _DENSE_CASES, ids=_CASE_IDS)
def test_deepseek_dense_q8_0_inventory(
    reader: gguf.GGUFReader,
    case: DenseTensorCase,
) -> None:
    matches = [
        tensor
        for tensor in reader.tensors
        if (
            tensor.name == case.tensor_suffix
            or tensor.name.endswith(case.tensor_suffix)
        )
        and tensor.tensor_type.name == "Q8_0"
    ]

    assert len(matches) == case.tensor_count
    for tensor in matches:
        assert tuple(reversed(tuple(int(value) for value in tensor.shape))) == (
            case.out_features,
            case.in_features,
        )
        assert tuple(tensor.data.shape) == (
            case.out_features,
            case.in_features // 32 * 34,
        )
    assert _tensor(reader, case.tensor_name) in matches


@pytest.mark.parametrize(
    "case_index,case",
    tuple(enumerate(_DENSE_CASES)),
    ids=_CASE_IDS,
)
def test_deepseek_dense_q8_0_forward_matches_gguf_reference(
    reader: gguf.GGUFReader,
    case_index: int,
    case: DenseTensorCase,
) -> None:
    packed, logical_weight, quant_type = _checked_weight(reader, case)
    generator = torch.Generator(device="cuda").manual_seed(12000 + case_index)
    input = torch.randn(
        3,
        43,
        case.in_features,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )

    expected = torch.nn.functional.linear(input, logical_weight)
    actual = torch_ggml_ops.mmq(
        input,
        packed,
        int(quant_type),
        _CHECKED_OUT_FEATURES,
    )
    error = actual.float() - expected.float()
    normalized_rmse = (
        error.square().mean().sqrt() / expected.float().square().mean().sqrt()
    )

    assert actual.shape == (3, 43, _CHECKED_OUT_FEATURES)
    assert actual.dtype == torch.bfloat16
    assert actual.is_contiguous()
    assert torch.isfinite(actual).all()
    assert normalized_rmse.item() < 0.04


@pytest.mark.parametrize("case", _DENSE_CASES, ids=_CASE_IDS)
def test_deepseek_dense_q8_0_backward_decodes_selected_weight_row(
    reader: gguf.GGUFReader,
    case: DenseTensorCase,
) -> None:
    packed, logical_weight, quant_type = _checked_weight(reader, case)
    grad_output = torch.zeros(
        1, _CHECKED_OUT_FEATURES, device="cuda", dtype=torch.bfloat16
    )
    grad_output[0, 17] = 1

    actual = torch.ops.torch_ggml_ops.mmq_grad_input.default(
        grad_output,
        packed,
        int(quant_type),
        case.in_features,
    )

    torch.testing.assert_close(actual[0], logical_weight[17], rtol=0, atol=0)


def test_deepseek_dense_q8_0_backward_crosses_row_tile(
    reader: gguf.GGUFReader,
) -> None:
    case = _DENSE_CASES[1]
    packed, logical_weight, quant_type = _checked_weight(reader, case)
    grad_output = torch.randn(
        65,
        _CHECKED_OUT_FEATURES,
        generator=torch.Generator(device="cuda").manual_seed(12567),
        device="cuda",
        dtype=torch.bfloat16,
    )
    expected = torch.mm(grad_output, logical_weight)

    actual = torch.ops.torch_ggml_ops.mmq_grad_input.default(
        grad_output,
        packed,
        int(quant_type),
        case.in_features,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "case_index,case",
    tuple(enumerate(_DENSE_CASES)),
    ids=_CASE_IDS,
)
def test_deepseek_dense_q8_0_backward_and_autograd_match_gguf_reference(
    reader: gguf.GGUFReader,
    case_index: int,
    case: DenseTensorCase,
) -> None:
    packed, logical_weight, quant_type = _checked_weight(reader, case)
    generator = torch.Generator(device="cuda").manual_seed(13000 + case_index)
    input = torch.randn(
        2,
        5,
        case.in_features,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    grad_output = torch.randn(
        2,
        5,
        _CHECKED_OUT_FEATURES,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    expected = torch.mm(
        grad_output.reshape(-1, _CHECKED_OUT_FEATURES), logical_weight
    ).reshape_as(input)

    actual = torch.ops.torch_ggml_ops.mmq_grad_input.default(
        grad_output,
        packed,
        int(quant_type),
        case.in_features,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    torch_ggml_ops.mmq(
        input,
        packed,
        int(quant_type),
        _CHECKED_OUT_FEATURES,
    ).backward(grad_output)
    torch.testing.assert_close(input.grad, expected, rtol=0, atol=0)
    assert packed.grad is None
