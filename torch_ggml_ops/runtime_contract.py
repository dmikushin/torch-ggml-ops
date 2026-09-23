"""Shared Python facts for the native MMQ launch contracts."""

import torch

QUANT_WORKSPACE_BLOCK_VALUES = 128
QUANT_WORKSPACE_BLOCK_BYTES = 144
PAIRED_ROW_TASK_ROWS = 64
PAIRED_ROW_TASK_QUANT_TYPES = frozenset({11, 22})
# The ROCm kernels quantize activations to Q8_1 in a caller-owned workspace.
# The CUDA kernels multiply BF16 activations directly and take an empty one.
QUANTIZES_ACTIVATIONS = torch.version.hip is not None
# GGML quant type ids that the dense ``mmq`` operators accept on this backend.
# ROCm additionally requires an exact deployed (M, N, K) key; CUDA accepts any
# shape whose input width is a multiple of 256.
DENSE_MMQ_QUANT_TYPES = (
    frozenset({11, 12, 13, 14, 8})  # Q3_K, Q4_K, Q5_K, Q6_K, Q8_0
    if QUANTIZES_ACTIVATIONS
    # Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ3_S, IQ4_XS
    else frozenset({8, 11, 12, 13, 14, 20, 21, 23})
)


def quant_workspace_elements(numel: int) -> int:
    """Return the uint8 workspace elements required by quantization."""
    assert numel >= 0
    if not QUANTIZES_ACTIVATIONS:
        return 0
    return (numel // QUANT_WORKSPACE_BLOCK_VALUES) * QUANT_WORKSPACE_BLOCK_BYTES


def paired_row_task_rows(quant_type: int) -> int:
    """Return the row-task tile for the paired deployment, or zero."""
    return PAIRED_ROW_TASK_ROWS if quant_type in PAIRED_ROW_TASK_QUANT_TYPES else 0


def paired_row_task_capacity(
    aggregate_rows: int, route_entries: int, row_task_rows: int
) -> int:
    """Return the descriptor capacity required by paired row-task setup."""
    assert aggregate_rows > 0
    assert route_entries > 0
    assert row_task_rows > 0
    return (aggregate_rows + row_task_rows - 1) // row_task_rows + route_entries
