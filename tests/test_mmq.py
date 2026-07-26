import gguf
import pytest
import torch
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor

import torch_ggml_ops
from tests.mmq_test_support import (
    assert_normalized_rmse,
    find_tensor,
    load_packed_rows,
    random_bf16,
)
from tests.model_test_cases import (
    DENSE_MMQ_TEST_CASE_IDS,
    DENSE_MMQ_TEST_CASES,
    QWEN_Q4_DENSE_MMQ_TEST_CASE,
    DenseMMQTestCase,
)
from tests.model_test_support import model_reader


def _packed_case(
    case: DenseMMQTestCase,
) -> tuple[torch.Tensor, gguf.GGMLQuantizationType]:
    tensor = find_tensor(model_reader(case.model), case.tensor_name)
    assert tensor.tensor_type.name == case.quant_type
    return load_packed_rows(tensor, case.out_features), tensor.tensor_type


def _checked_weight(
    case: DenseMMQTestCase,
) -> tuple[torch.Tensor, torch.Tensor, gguf.GGMLQuantizationType]:
    packed, quant_type = _packed_case(case)
    logical = dequantize_gguf_tensor(
        packed,
        quant_type,
        dtype=torch.bfloat16,
        device="cuda",
    ).reshape(case.out_features, case.in_features)
    return packed, logical, quant_type


@pytest.mark.parametrize(
    "case_index,case",
    tuple(enumerate(DENSE_MMQ_TEST_CASES)),
    ids=DENSE_MMQ_TEST_CASE_IDS,
)
def test_dense_forward_matches_dequantized_reference(
    case_index: int,
    case: DenseMMQTestCase,
) -> None:
    packed, logical_weight, quant_type = _checked_weight(case)
    input = random_bf16(3, 43, case.in_features, seed=10000 + case_index)

    expected = torch.nn.functional.linear(input, logical_weight)
    actual = torch_ggml_ops.mmq(
        input,
        packed,
        int(quant_type),
        case.out_features,
    )

    assert actual.shape == (3, 43, case.out_features)
    assert actual.dtype == torch.bfloat16
    assert actual.is_contiguous()
    assert_normalized_rmse(actual, expected)


@pytest.mark.parametrize("case", DENSE_MMQ_TEST_CASES, ids=DENSE_MMQ_TEST_CASE_IDS)
def test_dense_backward_decodes_selected_weight_row(case: DenseMMQTestCase) -> None:
    packed, logical_weight, quant_type = _checked_weight(case)
    grad_output = torch.zeros(
        1,
        case.out_features,
        device="cuda",
        dtype=torch.bfloat16,
    )
    grad_output[0, 17] = 1

    actual = torch.ops.torch_ggml_ops.mmq_grad_input.default(
        grad_output,
        packed,
        int(quant_type),
        case.in_features,
    )

    torch.testing.assert_close(actual[0], logical_weight[17], rtol=0, atol=0)


@pytest.mark.parametrize(
    "case_index,case",
    tuple(enumerate(DENSE_MMQ_TEST_CASES)),
    ids=DENSE_MMQ_TEST_CASE_IDS,
)
def test_dense_backward_and_autograd_match_dequantized_reference(
    case_index: int,
    case: DenseMMQTestCase,
) -> None:
    packed, logical_weight, quant_type = _checked_weight(case)
    input = random_bf16(
        2,
        5,
        case.in_features,
        seed=11000 + case_index,
        requires_grad=True,
    )
    grad_output = random_bf16(
        2,
        5,
        case.out_features,
        seed=12000 + case_index,
    )
    expected = torch.mm(
        grad_output.reshape(-1, case.out_features), logical_weight
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
        case.out_features,
    ).backward(grad_output)
    torch.testing.assert_close(input.grad, expected, rtol=0, atol=0)
    assert packed.grad is None


def test_dense_zero_input_and_current_stream() -> None:
    case = QWEN_Q4_DENSE_MMQ_TEST_CASE
    packed, quant_type = _packed_case(case)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        input = torch.zeros(129, case.in_features, device="cuda", dtype=torch.bfloat16)
        output = torch_ggml_ops.mmq(
            input,
            packed,
            int(quant_type),
            case.out_features,
        )
        grad_output = torch.ones(
            129,
            case.out_features,
            device="cuda",
            dtype=torch.bfloat16,
        )
        grad_input = torch.ops.torch_ggml_ops.mmq_grad_input.default(
            grad_output,
            packed,
            int(quant_type),
            case.in_features,
        )
        forward_checksum = output.float().abs().sum()
        backward_checksum = grad_input.float().abs().sum()
    stream.synchronize()

    assert forward_checksum.item() == 0.0
    assert backward_checksum.item() > 0.0
    assert torch.isfinite(grad_input).all()


def test_dense_opcheck_and_compile() -> None:
    case = QWEN_Q4_DENSE_MMQ_TEST_CASE
    packed, quant_type = _packed_case(case)
    input = random_bf16(2, case.in_features, seed=13000, requires_grad=True)
    result = torch.library.opcheck(
        torch.ops.torch_ggml_ops.mmq.default,
        (input, packed, int(quant_type), case.out_features),
        test_utils=(
            "test_schema",
            "test_autograd_registration",
            "test_faketensor",
            "test_aot_dispatch_dynamic",
        ),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in result.values()), result

    grad_output = random_bf16(2, case.out_features, seed=13001)
    grad_result = torch.library.opcheck(
        torch.ops.torch_ggml_ops.mmq_grad_input.default,
        (grad_output, packed, int(quant_type), case.in_features),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in grad_result.values()), grad_result

    @torch.compile(fullgraph=True)
    def compiled(input: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
        return torch_ggml_ops.mmq(
            input,
            packed,
            int(quant_type),
            case.out_features,
        )

    expected = torch_ggml_ops.mmq(
        input.detach(),
        packed,
        int(quant_type),
        case.out_features,
    )
    actual = compiled(input.detach(), packed)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_dense_invalid_inputs_fail_without_hidden_copies() -> None:
    case = QWEN_Q4_DENSE_MMQ_TEST_CASE
    packed, quant_type = _packed_case(case)
    input = random_bf16(4, case.in_features, seed=14000)

    with pytest.raises(RuntimeError, match="contiguous"):
        torch_ggml_ops.mmq(input[:, ::2], packed, int(quant_type), case.out_features)
    with pytest.raises(RuntimeError, match="zero storage offset"):
        torch_ggml_ops.mmq(input[1:], packed, int(quant_type), case.out_features)
    with pytest.raises(RuntimeError, match="expected"):
        torch_ggml_ops.mmq(
            input,
            packed[:-1].clone(),
            int(quant_type),
            case.out_features,
        )
    with pytest.raises(RuntimeError, match="unsupported quant_type"):
        torch_ggml_ops.mmq(input, packed, 999, case.out_features)
    with pytest.raises(RuntimeError, match="zero-row"):
        torch_ggml_ops.mmq(input[:0], packed, int(quant_type), case.out_features)


def test_dense_grad_input_rejects_higher_order_gradients() -> None:
    case = QWEN_Q4_DENSE_MMQ_TEST_CASE
    packed, quant_type = _packed_case(case)
    grad_output = random_bf16(
        2,
        case.out_features,
        seed=15000,
        requires_grad=True,
    )
    grad_input = torch.ops.torch_ggml_ops.mmq_grad_input.default(
        grad_output,
        packed,
        int(quant_type),
        case.in_features,
    )

    with pytest.raises(RuntimeError, match="does not support higher-order"):
        grad_input.sum().backward()


def test_dense_invalid_grad_input_operands_fail_without_hidden_copies() -> None:
    case = QWEN_Q4_DENSE_MMQ_TEST_CASE
    packed, quant_type = _packed_case(case)
    grad_output = random_bf16(4, case.out_features, seed=16000)
    op = torch.ops.torch_ggml_ops.mmq_grad_input.default

    with pytest.raises(RuntimeError, match="contiguous"):
        op(
            grad_output[:, ::2],
            packed,
            int(quant_type),
            case.in_features,
        )
    with pytest.raises(RuntimeError, match="zero storage offset"):
        op(grad_output[1:], packed, int(quant_type), case.in_features)
    with pytest.raises(RuntimeError, match="expected"):
        op(
            grad_output,
            packed[:-1].clone(),
            int(quant_type),
            case.in_features,
        )
    with pytest.raises(RuntimeError, match="unsupported quant_type"):
        op(grad_output, packed, 999, case.in_features)
    with pytest.raises(RuntimeError, match="zero-row"):
        op(grad_output[:0], packed, int(quant_type), case.in_features)
