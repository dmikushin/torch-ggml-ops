#!/usr/bin/env python3
"""Benchmark production routed and fixed-group MMQ forward.

The packed path measures the complete public operator, including Q8_1 activation
quantization and grouped multiplication. Routed gate/up uses grouped_mmq_pair so
both projections share one Q8_1 workspace and compares with BF16 AITER GMM.
Fixed output-A uses fixed_grouped_mmq and a BF16 strided-batched GEMM reference.
"""

import argparse
import gc
import json
from pathlib import Path

import gguf
import torch
from aiter.ops.triton.gmm import gmm
from grouped_mmq_benchmark_common import (
    GroupedMMQCase as GroupedForwardCase,
)
from grouped_mmq_benchmark_common import (
    MODEL_FAMILY_CHOICES,
    RouteDistribution,
    benchmark_function,
    bf16_fixed_mmq_reference,
    device_metadata,
    distribution_summary,
    error_metrics,
    fixed_group_distribution,
    parse_name_list,
    resolve_model_family,
    route_distributions,
    select_cases,
    truncate_distribution,
)
from mmq_benchmark_common import (
    DEFAULT_MODEL,
    load_packed_tensor,
    make_bf16_input,
    parse_int_list,
    synchronize,
)
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor

import torch_ggml_ops
from torch_ggml_ops.aiter_gmm_heuristics import gmm_config as aiter_gmm_config

DEFAULT_OUTPUT = Path("/tmp/torch_ggml_ops_grouped_mmq_fwd_benchmark.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="override the selected model family's routed top-k",
    )
    parser.add_argument(
        "--model-family",
        choices=MODEL_FAMILY_CHOICES,
        default="auto",
        help="case family; auto detects DeepSeek from attn_output_a",
    )
    parser.add_argument("--batches", type=parse_int_list, default=(1, 4, 16))
    parser.add_argument(
        "--distributions",
        type=parse_name_list,
        default=("uniform", "skewed", "sparse", "boundary"),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--correctness-rows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument(
        "--cases",
        type=str,
        default="",
        help="comma-separated case names; empty selects every production case",
    )
    parser.add_argument("--primary-only", action="store_true")
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"GGUF model not found: {args.model}")
    if args.sequence_length <= 0 or (args.top_k is not None and args.top_k <= 0):
        parser.error("--sequence-length and --top-k must be positive")
    if args.warmup < 0 or args.repeats <= 0 or args.correctness_rows <= 0:
        parser.error(
            "warmup must be nonnegative; repeats/correctness rows must be positive"
        )
    known_distributions = {"uniform", "skewed", "sparse", "boundary"}
    unknown = sorted(set(args.distributions) - known_distributions)
    if unknown:
        parser.error(
            f"unknown distributions {unknown}; expected {sorted(known_distributions)}"
        )
    return args


def dense_grouped_mmq_reference(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    distribution: RouteDistribution,
    quant_type: int,
    out_features: int,
) -> torch.Tensor:
    outputs = []
    row_begin = 0
    for expert, size in zip(
        distribution.expert_indices_cpu, distribution.group_sizes_cpu, strict=True
    ):
        row_end = row_begin + size
        input_group = input[row_begin:row_end].clone()
        expert_weight = packed_weight[expert].clone()
        outputs.append(
            torch_ggml_ops.mmq(input_group, expert_weight, quant_type, out_features)
        )
        row_begin = row_end
    return torch.cat(outputs, dim=0)


def dense_fixed_mmq_reference(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    quant_type: int,
    out_features: int,
) -> torch.Tensor:
    return torch.stack(
        tuple(
            torch_ggml_ops.mmq(
                input[:, group].clone(),
                packed_weight[group].clone(),
                quant_type,
                out_features,
            )
            for group in range(input.shape[1])
        ),
        dim=1,
    )


def correctness_metrics(
    case: GroupedForwardCase,
    input: torch.Tensor,
    packed_weights: tuple[torch.Tensor, ...],
    logical_weights: tuple[torch.Tensor, ...],
    distribution: RouteDistribution,
    quant_type: int,
    gmm_config: dict[str, int],
) -> dict:
    if case.kind == "fixed":
        with torch.inference_mode():
            actual = torch_ggml_ops.fixed_grouped_mmq(input, packed_weights[0])
            dense_reference = dense_fixed_mmq_reference(
                input,
                packed_weights[0],
                quant_type,
                case.expected_out_features,
            )
            bf16_reference = bf16_fixed_mmq_reference(input, logical_weights[0])
        return {
            "rows": input.shape[0],
            "same_q8_dense": error_metrics(actual, dense_reference),
            "bf16_bmm": error_metrics(actual, bf16_reference),
        }

    checked_distribution = truncate_distribution(distribution, input.shape[0])
    expert_indices, expert_offsets, group_sizes = device_metadata(checked_distribution)
    selected_logical = tuple(
        weight.index_select(0, expert_indices).transpose(1, 2)
        for weight in logical_weights
    )

    with torch.inference_mode():
        if case.kind == "pair":
            actual = torch_ggml_ops.grouped_mmq_pair(
                input,
                packed_weights[0],
                packed_weights[1],
                expert_indices,
                expert_offsets,
                quant_type,
                case.expected_out_features,
            )
            dense_reference = tuple(
                dense_grouped_mmq_reference(
                    input,
                    weight,
                    checked_distribution,
                    quant_type,
                    case.expected_out_features,
                )
                for weight in packed_weights
            )
            bf16_reference = tuple(
                gmm(
                    input,
                    weight,
                    group_sizes,
                    preferred_element_type=input.dtype,
                    config=gmm_config,
                )
                for weight in selected_logical
            )
            return {
                "rows": input.shape[0],
                "same_q8_dense": [
                    error_metrics(actual[index], dense_reference[index])
                    for index in range(2)
                ],
                "bf16_aiter": [
                    error_metrics(actual[index], bf16_reference[index])
                    for index in range(2)
                ],
            }

        actual_single = torch_ggml_ops.grouped_mmq(
            input,
            packed_weights[0],
            expert_indices,
            expert_offsets,
            quant_type,
            case.expected_out_features,
        )
        dense_single = dense_grouped_mmq_reference(
            input,
            packed_weights[0],
            checked_distribution,
            quant_type,
            case.expected_out_features,
        )
        bf16_single = gmm(
            input,
            selected_logical[0],
            group_sizes,
            preferred_element_type=input.dtype,
            config=gmm_config,
        )
        return {
            "rows": input.shape[0],
            "same_q8_dense": error_metrics(actual_single, dense_single),
            "bf16_aiter": error_metrics(actual_single, bf16_single),
        }


def print_result(result: dict) -> None:
    mmq = result["grouped_mmq"]
    if result["kind"] == "fixed":
        reference = result["bf16_bmm"]
        reference_label = "BMM"
        nrmse = result["correctness"]["bf16_bmm"]["normalized_rmse"]
        row_label = "M"
    else:
        reference = result["aiter_bf16"]
        reference_label = "AITER"
        if result["kind"] == "pair":
            nrmse = max(
                metric["normalized_rmse"]
                for metric in result["correctness"]["bf16_aiter"]
            )
        else:
            nrmse = result["correctness"]["bf16_aiter"]["normalized_rmse"]
        row_label = "R"
    print(
        f"{result['case']:<24} B={result['batch']:>2} "
        f"{result['distribution']:<8} {row_label}={result['rows']:>6} "
        f"G={result['group_summary']['active_experts']:>3} "
        f"{result['quant_type']:<8} "
        f"MMQ={mmq['median_ms']:>8.3f} ms {mmq['logical_tflops']:>6.2f} TF "
        f"{reference_label}={reference['median_ms']:>8.3f} ms "
        f"{reference['logical_tflops']:>6.2f} TF "
        f"ratio={result['mmq_to_reference_tflops_ratio']:>5.2f}x "
        f"NRMSE={nrmse:.3e}",
        flush=True,
    )


def benchmark_fixed_case(
    args: argparse.Namespace,
    case: GroupedForwardCase,
    case_index: int,
    tensor: gguf.ReaderTensor,
    quant_type: int,
    quant_name: str,
    report: dict,
) -> None:
    flat_packed = load_packed_tensor(tensor)
    expected_flat_shape = (
        case.fixed_groups * case.expected_out_features,
        case.packed_row_bytes,
    )
    if tuple(flat_packed.shape) != expected_flat_shape:
        raise RuntimeError(
            f"{tensor.name} has packed shape {tuple(flat_packed.shape)}, "
            f"expected {expected_flat_shape}"
        )
    logical_shape = tuple(int(value) for value in reversed(tensor.shape))
    expected_logical_shape = (
        case.fixed_groups * case.expected_out_features,
        case.expected_in_features,
    )
    if logical_shape != expected_logical_shape:
        raise RuntimeError(
            f"{tensor.name} has logical shape {logical_shape}, "
            f"expected {expected_logical_shape}"
        )

    packed_weight = flat_packed.view(
        case.fixed_groups, case.expected_out_features, flat_packed.shape[-1]
    )
    logical_weight = dequantize_gguf_tensor(
        packed_weight,
        tensor.tensor_type,
        dtype=torch.bfloat16,
        device="cuda",
    ).reshape(
        case.fixed_groups,
        case.expected_out_features,
        case.expected_in_features,
    )

    for batch_index, batch in enumerate(args.batches):
        rows = batch * args.sequence_length
        input = make_bf16_input(
            rows * case.fixed_groups,
            case.expected_in_features,
            args.seed + case_index * 10000 + batch_index * 100,
        ).view(rows, case.fixed_groups, case.expected_in_features)

        def mmq_function(
            input=input,
            packed_weight=packed_weight,
        ):
            return torch_ggml_ops.fixed_grouped_mmq(input, packed_weight)

        def bf16_function(
            input=input,
            logical_weight=logical_weight,
        ):
            return bf16_fixed_mmq_reference(input, logical_weight)

        mmq_result = benchmark_function(
            mmq_function,
            rows,
            case.expected_out_features,
            case.expected_in_features,
            case.projections,
            args.warmup,
            args.repeats,
        )
        bf16_result = benchmark_function(
            bf16_function,
            rows,
            case.expected_out_features,
            case.expected_in_features,
            case.projections,
            args.warmup,
            args.repeats,
        )

        correctness_rows = min(rows, args.correctness_rows)
        correctness_input = input[:correctness_rows].clone()
        fixed_distribution = fixed_group_distribution(rows, case.fixed_groups)
        correctness = correctness_metrics(
            case,
            correctness_input,
            (packed_weight,),
            (logical_weight,),
            fixed_distribution,
            quant_type,
            {},
        )

        workspace_bytes = (
            rows * case.fixed_groups * (case.expected_in_features // (4 * 32)) * 144
        )
        output_bytes = (
            rows
            * case.fixed_groups
            * case.expected_out_features
            * torch.bfloat16.itemsize
        )
        group_summary = distribution_summary(fixed_distribution)
        model_calls = case.model_layer_count
        ratio = mmq_result["logical_tflops"] / bf16_result["logical_tflops"]
        result = {
            "case": case.name,
            "model_family": case.model_family,
            "kind": case.kind,
            "description": case.description,
            "priority": case.priority,
            "batch": batch,
            "rows": rows,
            "logical_group_rows": rows * case.fixed_groups,
            "n": case.expected_out_features,
            "k": case.expected_in_features,
            "projections": case.projections,
            "quant_type": quant_name,
            "quant_type_id": quant_type,
            "distribution": "fixed",
            "group_summary": group_summary,
            "expert_indices": list(range(case.fixed_groups)),
            "group_sizes": [rows] * case.fixed_groups,
            "packed_weight_shapes": [list(packed_weight.shape)],
            "reference_kind": "BF16 torch.bmm with public-layout conversion",
            "q8_workspace_bytes": workspace_bytes,
            "expected_output_bytes": output_bytes,
            "pair_shares_one_q8_workspace": False,
            "fixed_groups_share_one_q8_workspace": True,
            "model_calls_per_forward": model_calls,
            "checkpointed_calls_per_optimizer_step": 2 * model_calls,
            "grouped_mmq": mmq_result,
            "bf16_bmm": bf16_result,
            "mmq_to_reference_tflops_ratio": ratio,
            "mmq_to_bf16_bmm_tflops_ratio": ratio,
            "estimated_mmq_optimizer_step_ms": (
                2 * model_calls * mmq_result["median_ms"]
            ),
            "estimated_reference_optimizer_step_ms": (
                2 * model_calls * bf16_result["median_ms"]
            ),
            "estimated_bf16_bmm_optimizer_step_ms": (
                2 * model_calls * bf16_result["median_ms"]
            ),
            "correctness": correctness,
            "metadata_device_resident": True,
            "host_group_descriptor_build_in_timed_path": False,
            "current_stream_operator_contract_tested": True,
        }
        report["results"].append(result)
        print_result(result)

        del mmq_function, bf16_function
        del input, correctness_input
        gc.collect()
        torch.cuda.empty_cache()
        synchronize()

    del logical_weight, packed_weight, flat_packed
    gc.collect()
    torch.cuda.empty_cache()
    synchronize()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("a HIP/CUDA device is required")

    reader = gguf.GGUFReader(args.model)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    model_family, detected_model_family = resolve_model_family(
        args.model_family,
        tensors,
    )
    cases = select_cases(args.cases, args.primary_only, model_family)
    quant_names = {int(value): value.name for value in gguf.GGMLQuantizationType}
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())

    missing = [
        name for case in cases for name in case.tensor_names if name not in tensors
    ]
    if missing:
        raise RuntimeError(f"checkpoint is missing benchmark tensors: {missing}")

    routed_top_ks = {
        args.top_k if args.top_k is not None else case.top_k
        for case in cases
        if case.routed
    }
    report_top_k = next(iter(routed_top_ks)) if len(routed_top_ks) == 1 else None

    report = {
        "model": str(args.model),
        "model_family": model_family,
        "detected_model_family": detected_model_family,
        "device": {
            "name": properties.name,
            "gcn_arch_name": getattr(properties, "gcnArchName", None),
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
        },
        "configuration": {
            "sequence_length": args.sequence_length,
            "top_k": report_top_k,
            "top_k_override": args.top_k,
            "batches": list(args.batches),
            "cases": [case.name for case in cases],
            "distributions": list(args.distributions),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "correctness_rows": args.correctness_rows,
            "aiter_heuristic": "torch_ggml_ops.aiter_gmm_heuristics.gmm_config",
            "reference": (
                "BF16 AITER gmm with project-owned gmm_config"
                if all(case.routed for case in cases)
                else "case-specific BF16 reference; see routed_reference and fixed_reference"
            ),
            "routed_reference": "BF16 AITER gmm with project-owned gmm_config",
            "fixed_reference": "BF16 torch.bmm including public-layout conversion",
            "aiter_work_stealing": False,
        },
        "results": [],
    }

    print(
        f"device={properties.name} arch={getattr(properties, 'gcnArchName', None)} "
        f"torch={torch.__version__} hip={torch.version.hip}",
        flush=True,
    )
    print(
        f"model={args.model} family={model_family} cases={','.join(case.name for case in cases)}",
        flush=True,
    )

    with torch.inference_mode():
        for case_index, case in enumerate(cases):
            case_tensors = tuple(tensors[name] for name in case.tensor_names)
            quant_types = {int(tensor.tensor_type) for tensor in case_tensors}
            if len(quant_types) != 1:
                raise RuntimeError(f"{case.name} paired tensors have different qtypes")
            quant_type = quant_types.pop()
            quant_name = quant_names.get(quant_type, str(quant_type))
            if quant_name != case.expected_quant_type:
                raise RuntimeError(
                    f"{case.name} has quant type {quant_name}, "
                    f"expected {case.expected_quant_type}"
                )

            if case.kind == "fixed":
                benchmark_fixed_case(
                    args,
                    case,
                    case_index,
                    case_tensors[0],
                    quant_type,
                    quant_name,
                    report,
                )
                continue

            packed_weights = tuple(
                load_packed_tensor(tensor) for tensor in case_tensors
            )
            for tensor, packed in zip(case_tensors, packed_weights, strict=True):
                logical_shape = tuple(
                    int(value) for value in reversed(tensor.shape[:-1])
                )
                if logical_shape != (
                    case.expected_out_features,
                    case.expected_in_features,
                ):
                    raise RuntimeError(
                        f"{tensor.name} has logical per-expert shape {logical_shape}, "
                        f"expected {(case.expected_out_features, case.expected_in_features)}"
                    )
                if packed.shape[0] != 256:
                    raise RuntimeError(
                        f"{tensor.name} has {packed.shape[0]} experts, expected 256"
                    )
                if packed.shape[2] != case.packed_row_bytes:
                    raise RuntimeError(
                        f"{tensor.name} has {packed.shape[2]} packed bytes per row, "
                        f"expected {case.packed_row_bytes} for {case.expected_quant_type}"
                    )

            logical_weights = tuple(
                dequantize_gguf_tensor(
                    packed,
                    tensor.tensor_type,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                .reshape(256, case.expected_out_features, case.expected_in_features)
                .contiguous()
                for tensor, packed in zip(case_tensors, packed_weights, strict=True)
            )
            gmm_config = aiter_gmm_config(
                case.expected_in_features, case.expected_out_features
            )

            top_k = args.top_k if args.top_k is not None else case.top_k
            for batch_index, batch in enumerate(args.batches):
                rows = batch * args.sequence_length * top_k
                available_distributions = route_distributions(rows, batch)
                for distribution_index, distribution_name in enumerate(
                    args.distributions
                ):
                    distribution = available_distributions[distribution_name]
                    expert_indices, expert_offsets, group_sizes = device_metadata(
                        distribution
                    )
                    selected_logical = tuple(
                        weight.index_select(0, expert_indices).transpose(1, 2)
                        for weight in logical_weights
                    )
                    input_seed = (
                        args.seed
                        + case_index * 10000
                        + batch_index * 100
                        + distribution_index
                    )
                    input = make_bf16_input(rows, case.expected_in_features, input_seed)

                    if case.kind == "pair":

                        def mmq_function(
                            input=input,
                            packed_weights=packed_weights,
                            expert_indices=expert_indices,
                            expert_offsets=expert_offsets,
                            quant_type=quant_type,
                            out_features=case.expected_out_features,
                        ):
                            return torch_ggml_ops.grouped_mmq_pair(
                                input,
                                packed_weights[0],
                                packed_weights[1],
                                expert_indices,
                                expert_offsets,
                                quant_type,
                                out_features,
                            )

                        def aiter_function(
                            input=input,
                            selected_logical=selected_logical,
                            group_sizes=group_sizes,
                            gmm_config=gmm_config,
                        ):
                            return tuple(
                                gmm(
                                    input,
                                    weight,
                                    group_sizes,
                                    preferred_element_type=input.dtype,
                                    config=gmm_config,
                                )
                                for weight in selected_logical
                            )
                    else:

                        def mmq_function(
                            input=input,
                            packed_weights=packed_weights,
                            expert_indices=expert_indices,
                            expert_offsets=expert_offsets,
                            quant_type=quant_type,
                            out_features=case.expected_out_features,
                        ):
                            return torch_ggml_ops.grouped_mmq(
                                input,
                                packed_weights[0],
                                expert_indices,
                                expert_offsets,
                                quant_type,
                                out_features,
                            )

                        def aiter_function(
                            input=input,
                            selected_logical=selected_logical,
                            group_sizes=group_sizes,
                            gmm_config=gmm_config,
                        ):
                            return gmm(
                                input,
                                selected_logical[0],
                                group_sizes,
                                preferred_element_type=input.dtype,
                                config=gmm_config,
                            )

                    mmq_result = benchmark_function(
                        mmq_function,
                        rows,
                        case.expected_out_features,
                        case.expected_in_features,
                        case.projections,
                        args.warmup,
                        args.repeats,
                    )
                    aiter_result = benchmark_function(
                        aiter_function,
                        rows,
                        case.expected_out_features,
                        case.expected_in_features,
                        case.projections,
                        args.warmup,
                        args.repeats,
                    )

                    checked_distribution = truncate_distribution(
                        distribution, args.correctness_rows
                    )
                    correctness_input = input[: checked_distribution.rows].clone()
                    correctness = correctness_metrics(
                        case,
                        correctness_input,
                        packed_weights,
                        logical_weights,
                        checked_distribution,
                        quant_type,
                        gmm_config,
                    )

                    workspace_bytes = (
                        rows * (case.expected_in_features // (4 * 32)) * 144
                    )
                    output_bytes = (
                        case.projections
                        * rows
                        * case.expected_out_features
                        * torch.bfloat16.itemsize
                    )
                    model_calls = case.model_layer_count
                    ratio = (
                        mmq_result["logical_tflops"] / aiter_result["logical_tflops"]
                    )
                    result = {
                        "case": case.name,
                        "model_family": case.model_family,
                        "kind": case.kind,
                        "description": case.description,
                        "priority": case.priority,
                        "batch": batch,
                        "top_k": top_k,
                        "rows": rows,
                        "n": case.expected_out_features,
                        "k": case.expected_in_features,
                        "projections": case.projections,
                        "quant_type": quant_name,
                        "quant_type_id": quant_type,
                        "distribution": distribution.name,
                        "group_summary": distribution_summary(distribution),
                        "expert_indices": list(distribution.expert_indices_cpu),
                        "group_sizes": list(distribution.group_sizes_cpu),
                        "packed_weight_shapes": [
                            list(weight.shape) for weight in packed_weights
                        ],
                        "reference_kind": "BF16 AITER gmm",
                        "aiter_config": dict(gmm_config),
                        "q8_workspace_bytes": workspace_bytes,
                        "expected_output_bytes": output_bytes,
                        "pair_shares_one_q8_workspace": case.kind == "pair",
                        "model_calls_per_forward": model_calls,
                        "checkpointed_calls_per_optimizer_step": 2 * model_calls,
                        "grouped_mmq": mmq_result,
                        "aiter_bf16": aiter_result,
                        "mmq_to_reference_tflops_ratio": ratio,
                        "mmq_to_aiter_tflops_ratio": ratio,
                        "estimated_mmq_optimizer_step_ms": (
                            2 * model_calls * mmq_result["median_ms"]
                        ),
                        "estimated_reference_optimizer_step_ms": (
                            2 * model_calls * aiter_result["median_ms"]
                        ),
                        "estimated_aiter_optimizer_step_ms": (
                            2 * model_calls * aiter_result["median_ms"]
                        ),
                        "correctness": correctness,
                        "metadata_device_resident": True,
                        "host_group_descriptor_build_in_timed_path": False,
                        "current_stream_operator_contract_tested": True,
                    }
                    report["results"].append(result)
                    print_result(result)

                    del mmq_function, aiter_function
                    del (
                        input,
                        correctness_input,
                        selected_logical,
                        expert_indices,
                        expert_offsets,
                        group_sizes,
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                    synchronize()

            del logical_weights, packed_weights
            gc.collect()
            torch.cuda.empty_cache()
            synchronize()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"report={args.output}", flush=True)


if __name__ == "__main__":
    main()
