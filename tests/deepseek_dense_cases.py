from dataclasses import dataclass


@dataclass(frozen=True)
class DenseTensorCase:
    name: str
    tensor_name: str
    tensor_suffix: str
    out_features: int
    in_features: int
    tensor_count: int
    quant_type: str = "Q8_0"
    lm_head: bool = False


DEEPSEEK_DENSE_TENSOR_CASES = (
    DenseTensorCase(
        "attention_q_a",
        "blk.0.attn_q_a.weight",
        ".attn_q_a.weight",
        1024,
        4096,
        43,
    ),
    DenseTensorCase(
        "attention_q_b",
        "blk.0.attn_q_b.weight",
        ".attn_q_b.weight",
        32768,
        1024,
        43,
    ),
    DenseTensorCase(
        "attention_kv",
        "blk.0.attn_kv.weight",
        ".attn_kv.weight",
        512,
        4096,
        43,
    ),
    DenseTensorCase(
        "attention_output_b",
        "blk.0.attn_output_b.weight",
        ".attn_output_b.weight",
        4096,
        8192,
        43,
    ),
    DenseTensorCase(
        "shared_gate",
        "blk.0.ffn_gate_shexp.weight",
        ".ffn_gate_shexp.weight",
        2048,
        4096,
        43,
    ),
    DenseTensorCase(
        "shared_up",
        "blk.0.ffn_up_shexp.weight",
        ".ffn_up_shexp.weight",
        2048,
        4096,
        43,
    ),
    DenseTensorCase(
        "shared_down",
        "blk.0.ffn_down_shexp.weight",
        ".ffn_down_shexp.weight",
        4096,
        2048,
        43,
    ),
    DenseTensorCase(
        "lm_head",
        "output.weight",
        "output.weight",
        129280,
        4096,
        1,
        lm_head=True,
    ),
)

DEEPSEEK_DENSE_TENSORS_BY_NAME = {
    case.name: case for case in DEEPSEEK_DENSE_TENSOR_CASES
}
