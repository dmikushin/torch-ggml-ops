import gguf
import pytest
import torch
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor

import torch_ggml_ops
from tests.mmq_test_support import (
    assert_normalized_rmse,
    find_tensor,
    load_packed_fixed_groups,
    random_bf16,
)
from tests.model_test_cases import DEEPSEEK_MODEL
from tests.model_test_support import model_reader


@pytest.fixture(scope="module")
def reader() -> gguf.GGUFReader:
    return model_reader(DEEPSEEK_MODEL)


def _reference_fixed_grad_input(
    grad_output: torch.Tensor,
    packed: torch.Tensor,
    quant_type: gguf.GGMLQuantizationType,
    in_features: int,
) -> torch.Tensor:
    expected = torch.empty(
        *grad_output.shape[:-1],
        in_features,
        device=grad_output.device,
        dtype=grad_output.dtype,
    )
    for group in range(grad_output.shape[1]):
        logical = dequantize_gguf_tensor(
            packed[group],
            quant_type,
            dtype=grad_output.dtype,
            device=grad_output.device,
        ).reshape(grad_output.shape[-1], in_features)
        expected[:, group] = grad_output[:, group] @ logical
    return expected


@pytest.mark.parametrize("out_features", [37, 1024])
def test_deepseek_fixed_q8_0_forward_matches_dequantized_reference(
    reader: gguf.GGUFReader,
    out_features: int,
) -> None:
    tensor = find_tensor(reader, "blk.0.attn_output_a.weight")
    assert tensor.tensor_type.name == "Q8_0"
    assert tuple(tensor.data.shape) == (8192, 4352)
    packed = load_packed_fixed_groups(
        tensor,
        groups=8,
        group_out_features=1024,
        out_features=out_features,
    )
    input = random_bf16(2, 8, 4096, seed=9012)

    expected = torch.empty(
        *input.shape[:-1],
        out_features,
        device=input.device,
        dtype=input.dtype,
    )
    for group in range(input.shape[1]):
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
    assert_normalized_rmse(actual, expected)


def test_deepseek_fixed_q8_0_opcheck_compile_and_backward(
    reader: gguf.GGUFReader,
) -> None:
    tensor = find_tensor(reader, "blk.0.attn_output_a.weight")
    packed = load_packed_fixed_groups(
        tensor,
        groups=8,
        group_out_features=1024,
        out_features=37,
    )
    input = random_bf16(1, 8, 4096, seed=10000, requires_grad=True)

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

    grad_output = random_bf16(1, 8, 37, seed=3456)
    expected_grad = _reference_fixed_grad_input(
        grad_output, packed, tensor.tensor_type, 4096
    )
    actual_grad = torch.ops.torch_ggml_ops.fixed_grouped_mmq_grad_input.default(
        grad_output, packed
    )
    assert_normalized_rmse(actual_grad, expected_grad, maximum=5e-5)

    input = input.detach().requires_grad_()
    torch_ggml_ops.fixed_grouped_mmq(input, packed).backward(grad_output)
    assert input.grad is not None
    assert_normalized_rmse(input.grad, expected_grad, maximum=5e-5)
