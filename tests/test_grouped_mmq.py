import gguf
import pytest
import torch

import torch_ggml_ops
from tests.grouped_mmq_test_support import (
    dequantize_experts,
    fused_pair_grad_input_reference,
    grouped_forward_reference,
    grouped_grad_input_reference,
    load_expert_weight,
    small_route,
)
from tests.mmq_test_support import (
    assert_fused_pair_close,
    assert_normalized_rmse,
    random_bf16,
)
from tests.model_test_cases import (
    PAIRED_MMQ_TEST_CASE_IDS,
    PAIRED_MMQ_TEST_CASES,
    QWEN_MODEL,
    ROUTED_MMQ_TEST_CASE_IDS,
    ROUTED_MMQ_TEST_CASES,
    PairedMMQTestCase,
    RoutedMMQTestCase,
)
from tests.model_test_support import model_reader


@pytest.fixture(scope="module")
def reader() -> gguf.GGUFReader:
    return model_reader(QWEN_MODEL)


def _paired_weights(
    case: PairedMMQTestCase,
    *,
    out_features: int = 37,
) -> tuple[torch.Tensor, torch.Tensor, gguf.GGMLQuantizationType, int]:
    reader = model_reader(case.model)
    first, quant_type, in_features = load_expert_weight(
        reader, case.first_tensor_name, out_features=out_features
    )
    second, second_quant_type, second_in_features = load_expert_weight(
        reader,
        case.second_tensor_name,
        out_features=out_features,
        row_offset=(
            out_features if case.second_tensor_name == case.first_tensor_name else 0
        ),
    )
    assert quant_type.name == case.quant_type
    assert second_quant_type == quant_type
    assert second_in_features == in_features
    return first, second, quant_type, in_features


def _group_metadata() -> tuple[torch.Tensor, torch.Tensor]:
    experts = torch.tensor([0, 2, 5, 7], device="cuda", dtype=torch.int64)
    offsets = torch.tensor([1, 128, 257, 262], device="cuda", dtype=torch.int32)
    return experts, offsets


@pytest.mark.parametrize(
    "case_index,case",
    tuple(enumerate(ROUTED_MMQ_TEST_CASES)),
    ids=ROUTED_MMQ_TEST_CASE_IDS,
)
def test_grouped_forward_matches_dequantized_reference(
    case_index: int,
    case: RoutedMMQTestCase,
) -> None:
    packed, quant_type, in_features = load_expert_weight(
        model_reader(case.model), case.tensor_name
    )
    assert quant_type.name == case.quant_type
    experts, offsets = small_route()
    input = random_bf16(10, in_features, seed=20000 + case_index)

    expected = grouped_forward_reference(
        input, packed, experts, offsets, quant_type, 37
    )
    actual = torch_ggml_ops.grouped_mmq(
        input, packed, experts, offsets, int(quant_type), 37
    )
    assert actual.shape == (10, 37)
    assert actual.dtype == torch.bfloat16
    assert actual.is_contiguous()
    assert_normalized_rmse(actual, expected)


@pytest.mark.parametrize(
    "case_index,case",
    tuple(enumerate(PAIRED_MMQ_TEST_CASES)),
    ids=PAIRED_MMQ_TEST_CASE_IDS,
)
def test_grouped_pair_forward_matches_single_and_dequantized_references(
    case_index: int,
    case: PairedMMQTestCase,
) -> None:
    first, second, quant_type, in_features = _paired_weights(case)
    experts, offsets = small_route()
    input = random_bf16(10, in_features, seed=23000 + case_index)

    actual_first, actual_second = torch_ggml_ops.grouped_mmq_pair(
        input, first, second, experts, offsets, int(quant_type), 37
    )
    packed_first = torch_ggml_ops.grouped_mmq(
        input, first, experts, offsets, int(quant_type), 37
    )
    packed_second = torch_ggml_ops.grouped_mmq(
        input, second, experts, offsets, int(quant_type), 37
    )
    logical_first = grouped_forward_reference(
        input, first, experts, offsets, quant_type, 37
    )
    logical_second = grouped_forward_reference(
        input, second, experts, offsets, quant_type, 37
    )

    torch.testing.assert_close(actual_first, packed_first, rtol=0, atol=0)
    torch.testing.assert_close(actual_second, packed_second, rtol=0, atol=0)
    assert_normalized_rmse(actual_first, logical_first)
    assert_normalized_rmse(actual_second, logical_second)


def test_grouped_pair_production_row_tasks_match_dense(
    reader: gguf.GGUFReader,
) -> None:
    gate, quant_type, in_features = load_expert_weight(
        reader,
        "blk.0.ffn_gate_exps.weight",
        out_features=512,
    )
    up, up_quant_type, up_in_features = load_expert_weight(
        reader,
        "blk.0.ffn_up_exps.weight",
        out_features=512,
    )
    assert up_quant_type == quant_type
    assert up_in_features == in_features == 2048

    experts = torch.tensor([0, 2, 5, 7], device="cuda", dtype=torch.int64)
    offsets = torch.tensor([65, 194, 323, 512], device="cuda", dtype=torch.int32)
    input = random_bf16(512, in_features, seed=2468)

    actual_gate, actual_up = torch_ggml_ops.grouped_mmq_pair(
        input, gate, up, experts, offsets, int(quant_type), 512
    )

    expected_gate_parts = []
    expected_up_parts = []
    row_begin = 0
    for expert, row_end in zip(experts.cpu().tolist(), offsets.cpu().tolist()):
        group_input = input[row_begin:row_end].clone()
        expected_gate_parts.append(
            torch_ggml_ops.mmq(group_input, gate[expert].clone(), int(quant_type), 512)
        )
        expected_up_parts.append(
            torch_ggml_ops.mmq(group_input, up[expert].clone(), int(quant_type), 512)
        )
        row_begin = row_end

    torch.testing.assert_close(
        actual_gate, torch.cat(expected_gate_parts), rtol=0, atol=0
    )
    torch.testing.assert_close(actual_up, torch.cat(expected_up_parts), rtol=0, atol=0)


@pytest.mark.parametrize(
    "case_index,case",
    tuple(enumerate(ROUTED_MMQ_TEST_CASES)),
    ids=ROUTED_MMQ_TEST_CASE_IDS,
)
def test_grouped_backward_and_autograd_match_dequantized_reference(
    case_index: int,
    case: RoutedMMQTestCase,
) -> None:
    packed, quant_type, in_features = load_expert_weight(
        model_reader(case.model), case.tensor_name
    )
    experts, offsets = small_route()
    grad_output = random_bf16(10, 37, seed=21000 + case_index)
    expected = grouped_grad_input_reference(
        grad_output,
        packed,
        experts,
        offsets,
        quant_type,
        in_features,
    )

    actual = torch.ops.torch_ggml_ops.grouped_mmq_grad_input.default(
        grad_output,
        packed,
        experts,
        offsets,
        int(quant_type),
        in_features,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    input = random_bf16(
        10,
        in_features,
        seed=22000 + case_index,
        requires_grad=True,
    )
    torch_ggml_ops.grouped_mmq(
        input, packed, experts, offsets, int(quant_type), 37
    ).backward(grad_output)
    torch.testing.assert_close(input.grad, expected, rtol=0, atol=0)
    assert packed.grad is None


@pytest.mark.parametrize(
    "case_index,case",
    tuple(enumerate(PAIRED_MMQ_TEST_CASES)),
    ids=PAIRED_MMQ_TEST_CASE_IDS,
)
def test_grouped_pair_backward_and_autograd_match_fused_reference(
    case_index: int,
    case: PairedMMQTestCase,
) -> None:
    first, second, quant_type, in_features = _paired_weights(case)
    experts, offsets = small_route()
    first_grad = random_bf16(10, 37, seed=24000 + case_index)
    second_grad = random_bf16(10, 37, seed=25000 + case_index)
    logical_first = dequantize_experts(first, experts, quant_type, 37, in_features)
    logical_second = dequantize_experts(second, experts, quant_type, 37, in_features)
    expected = fused_pair_grad_input_reference(
        first_grad,
        second_grad,
        logical_first,
        logical_second,
        offsets,
    )

    actual = torch.ops.torch_ggml_ops.grouped_mmq_pair_grad_input.default(
        first_grad,
        second_grad,
        first,
        second,
        experts,
        offsets,
        int(quant_type),
        in_features,
    )
    assert_fused_pair_close(actual, expected)

    input = random_bf16(
        10,
        in_features,
        seed=26000 + case_index,
        requires_grad=True,
    )
    first_output, second_output = torch_ggml_ops.grouped_mmq_pair(
        input, first, second, experts, offsets, int(quant_type), 37
    )
    torch.autograd.backward((first_output, second_output), (first_grad, second_grad))
    assert input.grad is not None
    assert_fused_pair_close(input.grad, expected)
    assert first.grad is None
    assert second.grad is None


def test_grouped_backward_route_group_boundaries(
    reader: gguf.GGUFReader,
) -> None:
    packed, quant_type, in_features = load_expert_weight(
        reader,
        "blk.2.ffn_down_exps.weight",
        num_experts=12,
        out_features=37,
    )
    group_size_values = (1, 15, 16, 17, 63, 64, 65, 127, 128, 129)
    group_sizes = torch.tensor(
        group_size_values,
        device="cuda",
        dtype=torch.int32,
    )
    offsets = group_sizes.cumsum(0).to(torch.int32).contiguous()
    experts = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 11],
        device="cuda",
        dtype=torch.int64,
    )
    grad_output = random_bf16(sum(group_size_values), 37, seed=8642)

    actual = torch.ops.torch_ggml_ops.grouped_mmq_grad_input.default(
        grad_output,
        packed,
        experts,
        offsets,
        int(quant_type),
        in_features,
    )
    expected = grouped_grad_input_reference(
        grad_output,
        packed,
        experts,
        offsets,
        quant_type,
        in_features,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_grouped_grad_input_direct_ops_compose_and_reject_higher_order(
    reader: gguf.GGUFReader,
) -> None:
    gate, quant_type, in_features = load_expert_weight(
        reader, "blk.10.ffn_gate_exps.weight", out_features=64
    )
    up, _, _ = load_expert_weight(reader, "blk.10.ffn_up_exps.weight", out_features=64)
    experts = torch.tensor([0, 2], device="cuda", dtype=torch.int64)
    offsets = torch.tensor([2, 4], device="cuda", dtype=torch.int32)
    first_grad = random_bf16(4, 64, seed=30000, requires_grad=True)
    second_grad = random_bf16(4, 64, seed=30001, requires_grad=True)

    single_result = torch.library.opcheck(
        torch.ops.torch_ggml_ops.grouped_mmq_grad_input.default,
        (first_grad.detach(), gate, experts, offsets, int(quant_type), in_features),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in single_result.values()), single_result

    pair_result = torch.library.opcheck(
        torch.ops.torch_ggml_ops.grouped_mmq_pair_grad_input.default,
        (
            first_grad.detach(),
            second_grad.detach(),
            gate,
            up,
            experts,
            offsets,
            int(quant_type),
            in_features,
        ),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
        raise_exception=False,
    )
    assert all(value == "SUCCESS" for value in pair_result.values()), pair_result

    single_grad_input = torch.ops.torch_ggml_ops.grouped_mmq_grad_input.default(
        first_grad,
        gate,
        experts,
        offsets,
        int(quant_type),
        in_features,
    )
    with pytest.raises(RuntimeError, match="does not support higher-order"):
        single_grad_input.sum().backward()

    pair_grad_input = torch.ops.torch_ggml_ops.grouped_mmq_pair_grad_input.default(
        first_grad,
        second_grad,
        gate,
        up,
        experts,
        offsets,
        int(quant_type),
        in_features,
    )
    with pytest.raises(RuntimeError, match="does not support higher-order"):
        pair_grad_input.sum().backward()


def test_grouped_grad_input_uses_current_stream_and_rejects_invalid_operands(
    reader: gguf.GGUFReader,
) -> None:
    packed, quant_type, in_features = load_expert_weight(
        reader, "blk.2.ffn_down_exps.weight", out_features=37
    )
    experts = torch.tensor([0, 2], device="cuda", dtype=torch.int64)
    offsets = torch.tensor([2, 4], device="cuda", dtype=torch.int32)
    grad_output = random_bf16(4, 37, seed=10000)
    op = torch.ops.torch_ggml_ops.grouped_mmq_grad_input.default

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        stream_grad = torch.ones(4, 37, device="cuda", dtype=torch.bfloat16)
        stream_result = op(
            stream_grad,
            packed,
            experts,
            offsets,
            int(quant_type),
            in_features,
        )
        checksum = stream_result.float().abs().sum()
    stream.synchronize()
    assert checksum.item() > 0

    with pytest.raises(RuntimeError, match="contiguous"):
        op(
            random_bf16(4, 74, seed=10001)[:, ::2],
            packed,
            experts,
            offsets,
            int(quant_type),
            in_features,
        )
    with pytest.raises(RuntimeError, match="zero storage offset"):
        op(
            random_bf16(5, 37, seed=10002)[1:],
            packed,
            experts,
            offsets,
            int(quant_type),
            in_features,
        )
    with pytest.raises(RuntimeError, match="torch.int32"):
        op(
            grad_output,
            packed,
            experts,
            offsets.long(),
            int(quant_type),
            in_features,
        )
    with pytest.raises(RuntimeError, match="bytes per row"):
        op(
            grad_output,
            packed[..., :-1].contiguous(),
            experts,
            offsets,
            int(quant_type),
            in_features,
        )


def test_grouped_pair_opcheck_and_compile(reader: gguf.GGUFReader) -> None:
    gate, quant_type, in_features = load_expert_weight(
        reader, "blk.0.ffn_gate_exps.weight", out_features=64
    )
    up, _, _ = load_expert_weight(reader, "blk.0.ffn_up_exps.weight", out_features=64)
    experts = torch.tensor([0, 2], device="cuda", dtype=torch.int64)
    offsets = torch.tensor([2, 4], device="cuda", dtype=torch.int32)
    input = random_bf16(4, in_features, seed=31000, requires_grad=True)
    result = torch.library.opcheck(
        torch.ops.torch_ggml_ops.grouped_mmq_pair.default,
        (input, gate, up, experts, offsets, int(quant_type), 64),
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
    def compiled(
        input: torch.Tensor,
        gate: torch.Tensor,
        up: torch.Tensor,
        experts: torch.Tensor,
        offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch_ggml_ops.grouped_mmq_pair(
            input, gate, up, experts, offsets, int(quant_type), 64
        )

    expected = torch_ggml_ops.grouped_mmq_pair(
        input.detach(), gate, up, experts, offsets, int(quant_type), 64
    )
    actual = compiled(input.detach(), gate, up, experts, offsets)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_grouped_opcheck_compile_and_invalid_metadata(
    reader: gguf.GGUFReader,
) -> None:
    packed, quant_type, in_features = load_expert_weight(
        reader, "blk.2.ffn_down_exps.weight"
    )
    experts = torch.tensor([0, 2], device="cuda", dtype=torch.int64)
    offsets = torch.tensor([2, 4], device="cuda", dtype=torch.int32)
    input = random_bf16(4, in_features, seed=32000, requires_grad=True)
    result = torch.library.opcheck(
        torch.ops.torch_ggml_ops.grouped_mmq.default,
        (input, packed, experts, offsets, int(quant_type), 37),
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
    def compiled(
        input: torch.Tensor,
        packed: torch.Tensor,
        experts: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        return torch_ggml_ops.grouped_mmq(
            input, packed, experts, offsets, int(quant_type), 37
        )

    expected = torch_ggml_ops.grouped_mmq(
        input.detach(), packed, experts, offsets, int(quant_type), 37
    )
    actual = compiled(input.detach(), packed, experts, offsets)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    with pytest.raises(RuntimeError, match="torch.int32"):
        torch_ggml_ops.grouped_mmq(
            input.detach(), packed, experts, offsets.long(), int(quant_type), 37
        )
    with pytest.raises(RuntimeError, match="zero storage offset"):
        torch_ggml_ops.grouped_mmq(
            input.detach(),
            packed,
            experts,
            torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32)[1:],
            int(quant_type),
            37,
        )
