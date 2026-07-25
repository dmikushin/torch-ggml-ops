import os
from pathlib import Path

import gguf
import numpy as np
import pytest
import torch
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor

import torch_ggml_ops

_MODEL = Path(
    os.environ.get(
        "GGUF_DEEPSEEK_MMQ_TEST_MODEL",
        os.path.expanduser("~/models/ds4/DeepSeek-V4-Flash-IQ2XXS.gguf"),
    )
)
_DENSE_CASES = (
    ("attention_q_a", "blk.0.attn_q_a.weight", ".attn_q_a.weight", 1024, 4096, 43),
    ("attention_q_b", "blk.0.attn_q_b.weight", ".attn_q_b.weight", 32768, 1024, 43),
    ("attention_kv", "blk.0.attn_kv.weight", ".attn_kv.weight", 512, 4096, 43),
    (
        "attention_output_b",
        "blk.0.attn_output_b.weight",
        ".attn_output_b.weight",
        4096,
        8192,
        43,
    ),
    (
        "shared_gate",
        "blk.0.ffn_gate_shexp.weight",
        ".ffn_gate_shexp.weight",
        2048,
        4096,
        43,
    ),
    (
        "shared_up",
        "blk.0.ffn_up_shexp.weight",
        ".ffn_up_shexp.weight",
        2048,
        4096,
        43,
    ),
    (
        "shared_down",
        "blk.0.ffn_down_shexp.weight",
        ".ffn_down_shexp.weight",
        4096,
        2048,
        43,
    ),
    ("lm_head", "output.weight", "output.weight", 129280, 4096, 1),
)
_CASE_IDS = tuple(case[0] for case in _DENSE_CASES)


@pytest.fixture(scope="module")
def reader() -> gguf.GGUFReader:
    if not _MODEL.is_file():
        pytest.skip("DeepSeek-V4-Flash GGUF model is unavailable")
    return gguf.GGUFReader(_MODEL)


def _tensor(reader: gguf.GGUFReader, name: str) -> gguf.ReaderTensor:
    return next(tensor for tensor in reader.tensors if tensor.name == name)


@pytest.mark.parametrize(
    "_case_name,tensor_name,tensor_suffix,out_features,in_features,tensor_count",
    _DENSE_CASES,
    ids=_CASE_IDS,
)
def test_deepseek_dense_q8_0_inventory(
    reader: gguf.GGUFReader,
    _case_name: str,
    tensor_name: str,
    tensor_suffix: str,
    out_features: int,
    in_features: int,
    tensor_count: int,
) -> None:
    matches = [
        tensor
        for tensor in reader.tensors
        if (tensor.name == tensor_suffix or tensor.name.endswith(tensor_suffix))
        and tensor.tensor_type.name == "Q8_0"
    ]

    assert len(matches) == tensor_count
    for tensor in matches:
        assert tuple(reversed(tuple(int(value) for value in tensor.shape))) == (
            out_features,
            in_features,
        )
        assert tuple(tensor.data.shape) == (
            out_features,
            in_features // 32 * 34,
        )
    assert _tensor(reader, tensor_name) in matches


@pytest.mark.parametrize(
    "case_index,case",
    tuple(enumerate(_DENSE_CASES)),
    ids=_CASE_IDS,
)
def test_deepseek_dense_q8_0_forward_matches_gguf_reference(
    reader: gguf.GGUFReader,
    case_index: int,
    case: tuple[str, str, str, int, int, int],
) -> None:
    _case_name, tensor_name, _tensor_suffix, _out_features, in_features, _count = (
        case
    )
    tensor = _tensor(reader, tensor_name)
    checked_out_features = 129
    packed = torch.from_numpy(
        np.array(
            tensor.data[:checked_out_features],
            dtype=np.uint8,
            copy=True,
            order="C",
        )
    ).to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(12000 + case_index)
    input = torch.randn(
        3,
        43,
        in_features,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    logical_weight = dequantize_gguf_tensor(
        packed,
        tensor.tensor_type,
        dtype=torch.bfloat16,
        device="cuda",
    ).reshape(checked_out_features, in_features)

    expected = torch.nn.functional.linear(input, logical_weight)
    actual = torch_ggml_ops.mmq(
        input,
        packed,
        int(tensor.tensor_type),
        checked_out_features,
    )
    error = actual.float() - expected.float()
    normalized_rmse = (
        error.square().mean().sqrt() / expected.float().square().mean().sqrt()
    )

    assert actual.shape == (3, 43, checked_out_features)
    assert actual.dtype == torch.bfloat16
    assert actual.is_contiguous()
    assert torch.isfinite(actual).all()
    assert normalized_rmse.item() < 0.04
