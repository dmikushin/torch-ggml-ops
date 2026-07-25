"""Shared utilities for dense MMQ forward and backward benchmarks."""

import argparse
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import gguf
import numpy as np
import torch

DEFAULT_MODEL = Path(
    os.environ.get(
        "GGUF_MMQ_BENCH_MODEL",
        os.path.expanduser("~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf"),
    )
)
MODEL_FAMILY_CHOICES = ("auto", "qwen", "deepseek")
DEFAULT_LM_HEAD_CHUNKS = {
    "qwen": (64, 128, 256),
    "deepseek": (32, 64, 128, 256, 512),
}


@dataclass(frozen=True)
class WeightCase:
    name: str
    tensor_name: str
    expected_out_features: int
    expected_in_features: int
    model_tensor_count: int
    priority: str
    description: str
    lm_head: bool = False
    model_family: str = "qwen"
    expected_quant_type: str | None = None


@dataclass(frozen=True)
class DenseTensorCase:
    name: str
    tensor_name: str
    tensor_suffix: str
    out_features: int
    in_features: int
    tensor_count: int
    lm_head: bool = False


# One real checkpoint tensor represents each (N, K, quant_type) combination
# dispatched by dense MMQ in the model. model_tensor_count records how often
# that exact geometry and quantization appears among the 160 ordinary weights.
CASES = (
    WeightCase(
        "attn_q_q3_k",
        "blk.3.attn_q.weight",
        8192,
        2048,
        9,
        "primary",
        "full-attention query plus query-gate projection",
    ),
    WeightCase(
        "attn_q_q4_k",
        "blk.39.attn_q.weight",
        8192,
        2048,
        1,
        "secondary",
        "final-layer full-attention query plus query-gate projection",
    ),
    WeightCase(
        "narrow_q4_k",
        "blk.5.ffn_gate_shexp.weight",
        512,
        2048,
        70,
        "primary",
        "dominant k/v/shared-gate/shared-up geometry",
    ),
    WeightCase(
        "narrow_q5_k",
        "blk.0.ffn_gate_shexp.weight",
        512,
        2048,
        21,
        "secondary",
        "q5 k/v/shared-gate/shared-up geometry",
    ),
    WeightCase(
        "narrow_q3_k",
        "blk.3.attn_k.weight",
        512,
        2048,
        9,
        "secondary",
        "q3 full-attention key geometry",
    ),
    WeightCase(
        "attn_output_q4_k",
        "blk.3.attn_output.weight",
        2048,
        4096,
        10,
        "primary",
        "full-attention output projection",
    ),
    WeightCase(
        "shared_down_q4_k",
        "blk.5.ffn_down_shexp.weight",
        2048,
        512,
        30,
        "primary",
        "dominant shared-expert down projection",
    ),
    WeightCase(
        "shared_down_q5_k",
        "blk.0.ffn_down_shexp.weight",
        2048,
        512,
        10,
        "secondary",
        "q5 shared-expert down projection",
    ),
    WeightCase(
        "lm_head_q6_k",
        "output.weight",
        248320,
        2048,
        1,
        "primary",
        "chunked language-model head",
        lm_head=True,
    ),
)

DEEPSEEK_DENSE_TENSOR_CASES = (
    DenseTensorCase(
        "attention_q_a", "blk.0.attn_q_a.weight", ".attn_q_a.weight",
        1024, 4096, 43,
    ),
    DenseTensorCase(
        "attention_q_b", "blk.0.attn_q_b.weight", ".attn_q_b.weight",
        32768, 1024, 43,
    ),
    DenseTensorCase(
        "attention_kv", "blk.0.attn_kv.weight", ".attn_kv.weight",
        512, 4096, 43,
    ),
    DenseTensorCase(
        "attention_output_b", "blk.0.attn_output_b.weight",
        ".attn_output_b.weight", 4096, 8192, 43,
    ),
    DenseTensorCase(
        "shared_gate", "blk.0.ffn_gate_shexp.weight",
        ".ffn_gate_shexp.weight", 2048, 4096, 43,
    ),
    DenseTensorCase(
        "shared_up", "blk.0.ffn_up_shexp.weight",
        ".ffn_up_shexp.weight", 2048, 4096, 43,
    ),
    DenseTensorCase(
        "shared_down", "blk.0.ffn_down_shexp.weight",
        ".ffn_down_shexp.weight", 4096, 2048, 43,
    ),
    DenseTensorCase(
        "lm_head", "output.weight", "output.weight", 129280, 4096, 1,
        lm_head=True,
    ),
)
_DEEPSEEK_TENSORS = {case.name: case for case in DEEPSEEK_DENSE_TENSOR_CASES}


def _deepseek_weight_case(
    tensor_case_name: str,
    benchmark_name: str,
    description: str,
    *,
    model_tensor_count: int | None = None,
) -> WeightCase:
    tensor_case = _DEEPSEEK_TENSORS[tensor_case_name]
    return WeightCase(
        benchmark_name,
        tensor_case.tensor_name,
        tensor_case.out_features,
        tensor_case.in_features,
        tensor_case.tensor_count if model_tensor_count is None else model_tensor_count,
        "primary",
        description,
        lm_head=tensor_case.lm_head,
        model_family="deepseek",
        expected_quant_type="Q8_0",
    )


DEEPSEEK_CASES = (
    _deepseek_weight_case(
        "attention_q_a", "ds4_attn_q_a_q8_0", "attention query A projection"
    ),
    _deepseek_weight_case(
        "attention_q_b", "ds4_attn_q_b_q8_0", "attention query B projection"
    ),
    _deepseek_weight_case(
        "attention_kv", "ds4_attn_kv_q8_0", "attention key/value projection"
    ),
    _deepseek_weight_case(
        "attention_output_b", "ds4_attn_output_b_q8_0",
        "attention output B projection",
    ),
    _deepseek_weight_case(
        "shared_gate", "ds4_shared_gate_up_q8_0",
        "shared-expert gate/up geometry", model_tensor_count=86,
    ),
    _deepseek_weight_case(
        "shared_down", "ds4_shared_down_q8_0", "shared-expert down projection"
    ),
    _deepseek_weight_case(
        "lm_head", "ds4_lm_head_q8_0", "chunked language-model head"
    ),
)

DEEPSEEK_FORWARD_CASES = DEEPSEEK_CASES
FORWARD_CASES = CASES + DEEPSEEK_CASES


def parse_int_list(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive integers"
        )
    return result


def synchronize() -> None:
    torch.cuda.synchronize()


def cuda_event_times_ms(
    function: Callable[[], object], warmup: int, repeats: int
) -> list[float]:
    for _ in range(warmup):
        output = function()
        del output
    synchronize()

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
        del output
    return times


def incremental_peak_bytes(function: Callable[[], object]) -> tuple[int, int]:
    synchronize()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    output = function()
    synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    del output
    return (
        peak_allocated - baseline_allocated,
        max(0, peak_reserved - baseline_reserved),
    )


def summarize_timing(times_ms: list[float], m: int, n: int, k: int) -> dict:
    median_ms = statistics.median(times_ms)
    logical_flops = 2 * m * n * k
    return {
        "samples_ms": times_ms,
        "median_ms": median_ms,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "logical_tflops": logical_flops / (median_ms * 1.0e9),
    }


def make_bf16_input(rows: int, features: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(
        rows,
        features,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )


def load_packed_tensor(
    tensor: gguf.ReaderTensor,
    out_features: int | None = None,
) -> torch.Tensor:
    data = tensor.data if out_features is None else tensor.data[:out_features]
    host = np.array(data, dtype=np.uint8, copy=True, order="C")
    packed = torch.from_numpy(host).to("cuda")
    del host
    return packed


def resolve_model_family(
    requested: str,
    tensor_names: set[str],
) -> tuple[str, str]:
    detected = (
        "deepseek" if "blk.0.attn_output_a.weight" in tensor_names else "qwen"
    )
    return (detected if requested == "auto" else requested), detected


def resolve_lm_head_chunks(
    requested: tuple[int, ...] | None,
    model_family: str,
) -> tuple[int, ...]:
    return DEFAULT_LM_HEAD_CHUNKS[model_family] if requested is None else requested


def validate_weight_case(
    tensor: gguf.ReaderTensor,
    case: WeightCase,
) -> tuple[int, str, tuple[int, ...]]:
    logical_shape = tuple(int(value) for value in reversed(tensor.shape))
    expected_shape = (case.expected_out_features, case.expected_in_features)
    if logical_shape != expected_shape:
        raise RuntimeError(
            f"{case.tensor_name} has logical shape {logical_shape}, expected "
            f"{expected_shape}"
        )
    quant_type = int(tensor.tensor_type)
    quant_name = tensor.tensor_type.name
    if case.expected_quant_type is not None and quant_name != case.expected_quant_type:
        raise RuntimeError(
            f"{case.tensor_name} has quant type {quant_name}, expected "
            f"{case.expected_quant_type}"
        )
    physical_shape = tuple(int(value) for value in tensor.data.shape)
    return quant_type, quant_name, physical_shape


def make_row_specs(
    case: WeightCase,
    batches: tuple[int, ...],
    sequence_length: int,
    lm_head_chunks: tuple[int, ...],
) -> tuple[list[dict[str, int]], tuple[int, ...]]:
    if case.lm_head:
        specs = [
            {
                "batch": batch,
                "m": chunk,
                "model_rows": batch * sequence_length,
                "calls": math.ceil(batch * sequence_length / chunk),
            }
            for chunk in lm_head_chunks
            for batch in batches
        ]
        return specs, lm_head_chunks
    specs = [
        {
            "batch": batch,
            "m": batch * sequence_length,
            "model_rows": batch * sequence_length,
            "calls": 1,
        }
        for batch in batches
    ]
    return specs, tuple(spec["m"] for spec in specs)


def _select_cases(
    case_names: str,
    primary_only: bool,
    available_cases: tuple[WeightCase, ...],
) -> tuple[WeightCase, ...]:
    by_name = {case.name: case for case in available_cases}
    if case_names:
        names = tuple(name.strip() for name in case_names.split(",") if name.strip())
        unknown = sorted(set(names) - set(by_name))
        if unknown:
            raise ValueError(
                f"unknown cases {unknown}; available cases are {sorted(by_name)}"
            )
        cases = tuple(by_name[name] for name in names)
    else:
        cases = available_cases
    if primary_only:
        cases = tuple(case for case in cases if case.priority == "primary")
    if not cases:
        raise ValueError("no benchmark cases selected")
    return cases


def select_cases(
    case_names: str,
    primary_only: bool,
    model_family: str,
) -> tuple[WeightCase, ...]:
    available_cases = tuple(
        case for case in FORWARD_CASES if case.model_family == model_family
    )
    return _select_cases(case_names, primary_only, available_cases)


def select_forward_cases(
    case_names: str,
    primary_only: bool,
    model_family: str,
) -> tuple[WeightCase, ...]:
    return select_cases(case_names, primary_only, model_family)
