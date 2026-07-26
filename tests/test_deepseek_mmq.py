import pytest
import torch
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor

import torch_ggml_ops  # noqa: F401 Register native operators before torch.ops use.
from tests.deepseek_dense_cases import (
    DEEPSEEK_DENSE_TENSOR_CASES,
    DenseTensorCase,
)
from tests.mmq_test_support import find_tensor, load_packed_tensor, random_bf16
from tests.model_test_cases import DEEPSEEK_MODEL
from tests.model_test_support import model_reader

_CHECKED_OUT_FEATURES = 129


@pytest.mark.parametrize(
    "case",
    DEEPSEEK_DENSE_TENSOR_CASES,
    ids=tuple(case.name for case in DEEPSEEK_DENSE_TENSOR_CASES),
)
def test_deepseek_dense_q8_0_inventory(case: DenseTensorCase) -> None:
    reader = model_reader(DEEPSEEK_MODEL)
    matches = [
        tensor
        for tensor in reader.tensors
        if (
            tensor.name == case.tensor_suffix
            or tensor.name.endswith(case.tensor_suffix)
        )
        and tensor.tensor_type.name == case.quant_type
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
    assert find_tensor(reader, case.tensor_name) in matches


def test_deepseek_dense_q8_0_backward_crosses_row_tile() -> None:
    case = DEEPSEEK_DENSE_TENSOR_CASES[1]
    tensor = find_tensor(model_reader(DEEPSEEK_MODEL), case.tensor_name)
    packed = load_packed_tensor(tensor, _CHECKED_OUT_FEATURES)
    logical_weight = dequantize_gguf_tensor(
        packed,
        tensor.tensor_type,
        dtype=torch.bfloat16,
        device="cuda",
    ).reshape(_CHECKED_OUT_FEATURES, case.in_features)
    grad_output = random_bf16(65, _CHECKED_OUT_FEATURES, seed=12567)
    expected = torch.mm(grad_output, logical_weight)

    actual = torch.ops.torch_ggml_ops.mmq_grad_input.default(
        grad_output,
        packed,
        int(tensor.tensor_type),
        case.in_features,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
