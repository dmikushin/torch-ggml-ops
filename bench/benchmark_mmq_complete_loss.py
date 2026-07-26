#!/usr/bin/env python3
"""Benchmark the complete packed LM-head loss loop across chunk schedules."""

import argparse
import gc
import importlib
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

import gguf
import torch

from mmq_benchmark_common import (
    DEFAULT_MODEL,
    load_packed_tensor,
    make_bf16_input,
    parse_int_list,
    resolve_lm_head_chunks,
    resolve_model_family,
    select_cases,
    validate_weight_case,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-family",
        choices=("auto", "qwen", "deepseek"),
        default="auto",
    )
    parser.add_argument("--batches", type=parse_int_list, default=(1, 4, 16))
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--chunks", type=parse_int_list)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--ignore-index", type=int, default=-100)
    parser.add_argument(
        "--loss-module-root",
        type=Path,
        required=True,
        help="directory containing the production gguf_liger_loss module",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sequence_length <= 0:
        parser.error("--sequence-length must be positive")
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be nonnegative and --repeats must be positive")
    return args


def _load_loss_function(root: Path) -> Callable[..., tuple[torch.Tensor, ...]]:
    if not (root / "gguf_liger_loss.py").is_file():
        raise FileNotFoundError(f"missing {root / 'gguf_liger_loss.py'}")
    sys.path.insert(0, str(root))
    module = importlib.import_module("gguf_liger_loss")
    return module._packed_q8_linear_cross_entropy_forward


def _summarize(values: list[float]) -> dict[str, object]:
    return {
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "mean_ms": statistics.fmean(values),
        "samples_ms": values,
    }


def _event_times_ms(
    function: Callable[[], object],
    repeats: int,
) -> list[float]:
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


def _incremental_peak_bytes(function: Callable[[], object]) -> tuple[int, int]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    output = function()
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    del output
    return (
        peak_allocated - baseline_allocated,
        max(0, peak_reserved - baseline_reserved),
    )


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    args = _parse_args()
    loss_function = _load_loss_function(args.loss_module_root)

    reader = gguf.GGUFReader(args.model)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    model_family, detected_family = resolve_model_family(
        args.model_family, set(tensors)
    )
    chunks = resolve_lm_head_chunks(args.chunks, model_family)
    phase_order = chunks + tuple(reversed(chunks))
    lm_cases = tuple(
        case for case in select_cases("", False, model_family) if case.lm_head
    )
    if len(lm_cases) != 1:
        raise RuntimeError(
            f"expected one {model_family} LM-head case, found {len(lm_cases)}"
        )
    case = lm_cases[0]
    tensor = tensors[case.tensor_name]
    quant_type, quant_name, physical_shape = validate_weight_case(tensor, case)
    packed_weight = load_packed_tensor(tensor, case.expected_out_features)

    report: dict[str, object] = {
        "model": str(args.model),
        "model_family": model_family,
        "detected_model_family": detected_family,
        "device": torch.cuda.get_device_name(),
        "sequence_length": args.sequence_length,
        "batches": list(args.batches),
        "hidden_size": case.expected_in_features,
        "out_features": case.expected_out_features,
        "physical_weight_shape": list(physical_shape),
        "quant_type": quant_name,
        "quant_type_id": quant_type,
        "chunks": list(chunks),
        "phase_order": list(phase_order),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "scope": (
            "MMQ forward + in-place Liger cross entropy + packed MMQ "
            "grad-input"
        ),
        "batches_result": [],
    }

    for batch in args.batches:
        rows = batch * args.sequence_length
        input_tensor = make_bf16_input(rows, case.expected_in_features, args.seed)
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        target = torch.randint(
            0,
            case.expected_out_features,
            (rows,),
            dtype=torch.long,
            device="cuda",
            generator=generator,
        )
        target[::257] = args.ignore_index

        def run(chunk_size: int) -> tuple[torch.Tensor, ...]:
            return loss_function(
                input_tensor,
                packed_weight,
                target,
                quant_type,
                case.expected_out_features,
                chunk_size,
                args.ignore_index,
                0.0,
                0.0,
                "mean",
                None,
                False,
                False,
            )

        for chunk_size in chunks:
            for _ in range(args.warmup):
                output = run(chunk_size)
                del output
            torch.cuda.synchronize()

        correctness: dict[str, dict[str, float | bool]] = {}
        reference_loss: float | None = None
        reference_grad: torch.Tensor | None = None
        for chunk_size in chunks:
            loss, _, _, grad_input = run(chunk_size)
            torch.cuda.synchronize()
            loss_value = float(loss)
            if reference_loss is None:
                reference_loss = loss_value
                reference_grad = grad_input.clone()
            assert reference_loss is not None and reference_grad is not None
            grad_float = grad_input.float()
            reference_float = reference_grad.float()
            correctness[str(chunk_size)] = {
                "loss": loss_value,
                "loss_relative_to_first_chunk": (
                    abs(loss_value - reference_loss) / abs(reference_loss)
                ),
                "grad_exact_to_first_chunk": bool(
                    torch.equal(grad_input, reference_grad)
                ),
                "grad_cosine_to_first_chunk": float(
                    torch.nn.functional.cosine_similarity(
                        grad_float.flatten(), reference_float.flatten(), dim=0
                    )
                ),
                "grad_relative_l2_to_first_chunk": float(
                    torch.linalg.vector_norm(grad_float - reference_float)
                    / torch.linalg.vector_norm(reference_float)
                ),
            }
            del loss, grad_input, grad_float, reference_float
        del reference_grad

        peak: dict[str, dict[str, int]] = {}
        for chunk_size in chunks:
            allocated, reserved = _incremental_peak_bytes(
                lambda chunk_size=chunk_size: run(chunk_size)
            )
            peak[str(chunk_size)] = {
                "incremental_peak_allocated_bytes": allocated,
                "incremental_peak_reserved_bytes": reserved,
            }

        phases = []
        for phase_index, chunk_size in enumerate(phase_order):
            times = _event_times_ms(
                lambda chunk_size=chunk_size: run(chunk_size), args.repeats
            )
            phase = {
                "phase": phase_index,
                "chunk_size": chunk_size,
                **_summarize(times),
            }
            phases.append(phase)
            print(
                f"B={batch} phase={phase_index} M={chunk_size} "
                f"median={phase['median_ms']:.3f} ms",
                flush=True,
            )

        by_chunk = {}
        for chunk_size in chunks:
            chunk_phases = [
                phase for phase in phases if phase["chunk_size"] == chunk_size
            ]
            phase_medians = [float(phase["median_ms"]) for phase in chunk_phases]
            by_chunk[str(chunk_size)] = {
                "bracket_median_ms": statistics.fmean(phase_medians),
                "phase_medians_ms": phase_medians,
                **peak[str(chunk_size)],
                **correctness[str(chunk_size)],
            }

        batch_result = {
            "batch": batch,
            "rows": rows,
            "by_chunk": by_chunk,
            "phases": phases,
        }
        report["batches_result"].append(batch_result)
        _write_report(args.output, report)
        gc.collect()
        torch.cuda.empty_cache()

    _write_report(args.output, report)
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
