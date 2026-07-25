import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSRC = ROOT / "csrc"
PACKAGE_DIR = ROOT / "torch_ggml_ops" / "kernels" / "gfx1151"
GENERATED_HEADER = CSRC / "generated" / "mmq_bundle_table.cuh"
ARCH = "gfx1151"
ABI_PREFIX = "torch_ggml_ops_mmq_gfx1151_v1_"

FORWARD_WRAPPER = "mmq_bundle_forward_kernel.cu"
DENSE_BACKWARD_WRAPPER = "mmq_bundle_dense_backward_kernel.cu"
GROUPED_BACKWARD_WRAPPER = "mmq_bundle_grouped_backward_kernel.cu"
WRAPPERS = (
    CSRC / FORWARD_WRAPPER,
    CSRC / DENSE_BACKWARD_WRAPPER,
    CSRC / GROUPED_BACKWARD_WRAPPER,
)

QUANT_TYPES = (
    ("Q8_0", 8),
    ("Q2_K", 10),
    ("Q3_K", 11),
    ("Q4_K", 12),
    ("Q5_K", 13),
    ("Q6_K", 14),
    ("IQ2_XXS", 16),
    ("IQ2_S", 22),
)
BACKWARD_QUANT_TYPES = tuple(
    item for item in QUANT_TYPES
    if item[0] in {"Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ2_XXS", "IQ2_S"}
)
ROW_TASK_TYPES = tuple(
    item for item in QUANT_TYPES if item[0] in {"Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ2_S"}
)


@dataclass(frozen=True)
class KernelSpec:
    cpp_id: str
    suffix: str
    wrapper: str
    defines: tuple[tuple[str, int], ...]
    enforce_resource_gate: bool = False

    @property
    def symbol(self) -> str:
        return ABI_PREFIX + self.suffix

    @property
    def filename(self) -> str:
        return f"{self.symbol}.hsaco"

    @property
    def cuid(self) -> str:
        return hashlib.sha256(self.symbol.encode()).hexdigest()[:16]

    def compiler_defines(self) -> list[str]:
        return [
            f"-DMMQ_BUNDLE_KERNEL_SYMBOL={self.symbol}",
            *(f"-D{name}={value}" for name, value in self.defines),
        ]


def _quant_suffix(name: str) -> str:
    return name.lower()


def _forward_spec(
    cpp_id: str,
    suffix: str,
    kind: int,
    *,
    quant_type: int = 0,
    j: int = 0,
    nrows_weight: int = 0,
    blocks_per_weight_row: int = 0,
    groups: int = 0,
    fallback: bool = False,
    rolled_q2: bool = False,
    mixed_iq2_s: bool = False,
    mixed_q2_k: bool = False,
    enforce_resource_gate: bool = False,
) -> KernelSpec:
    defines = [
        ("MMQ_BUNDLE_FORWARD_KIND", kind),
        ("MMQ_BUNDLE_QUANT_TYPE", quant_type),
        ("MMQ_BUNDLE_J", j),
        ("MMQ_BUNDLE_NROWS_WEIGHT", nrows_weight),
        ("MMQ_BUNDLE_BLOCKS_PER_WEIGHT_ROW", blocks_per_weight_row),
        ("MMQ_BUNDLE_GROUPS", groups),
        ("MMQ_BUNDLE_FALLBACK", int(fallback)),
    ]
    if rolled_q2:
        defines.append(("MMQ_USE_ROLLED_Q2_K", 1))
    if mixed_iq2_s:
        defines.append(("MMQ_USE_MIXED_IQ2_S_TAILS", 1))
    if mixed_q2_k:
        defines.append(("MMQ_USE_MIXED_Q2_K_TAILS", 1))
    return KernelSpec(
        cpp_id,
        suffix,
        FORWARD_WRAPPER,
        tuple(defines),
        enforce_resource_gate,
    )


def _grouped_forward_specs() -> list[KernelSpec]:
    specs = [
        _forward_spec(
            "GroupedFwdFixedQ80G8K4096J64Full",
            "grouped_fwd_fixed_q8_0_g8_k4096_j64_full",
            5,
            quant_type=8,
            j=64,
            blocks_per_weight_row=16,
            groups=8,
            enforce_resource_gate=True,
        ),
        _forward_spec(
            "GroupedFwdFixedQ80G8K4096J64Bounded",
            "grouped_fwd_fixed_q8_0_g8_k4096_j64_bounded",
            5,
            quant_type=8,
            j=64,
            blocks_per_weight_row=16,
            groups=8,
            fallback=True,
        ),
    ]
    for quant_name, quant_type in QUANT_TYPES:
        label = quant_name.replace("_", "")
        suffix = _quant_suffix(quant_name)
        specs.append(
            _forward_spec(
                f"GroupedFwdSerial{label}GenericJ128",
                f"grouped_fwd_serial_{suffix}_generic_j128",
                3,
                quant_type=quant_type,
                j=128,
            )
        )
    for quant_name, quant_type in QUANT_TYPES:
        label = quant_name.replace("_", "")
        suffix = _quant_suffix(quant_name)
        specs.append(
            _forward_spec(
                f"GroupedFwdSerial{label}N512K2048J64",
                f"grouped_fwd_serial_{suffix}_n512_k2048_j64",
                3,
                quant_type=quant_type,
                j=64,
                nrows_weight=512,
                blocks_per_weight_row=8,
                enforce_resource_gate=quant_name in {"Q3_K", "IQ2_S"},
            )
        )
    for quant_name, quant_type in QUANT_TYPES:
        label = quant_name.replace("_", "")
        suffix = _quant_suffix(quant_name)
        specs.append(
            _forward_spec(
                f"GroupedFwdSerial{label}N2048K512J64",
                f"grouped_fwd_serial_{suffix}_n2048_k512_j64",
                3,
                quant_type=quant_type,
                j=64,
                nrows_weight=2048,
                blocks_per_weight_row=2,
                enforce_resource_gate=quant_name in {"Q4_K", "Q5_K", "IQ2_S"},
            )
        )
    for quant_name, quant_type in ROW_TASK_TYPES:
        label = quant_name.replace("_", "")
        suffix = _quant_suffix(quant_name)
        specs.append(
            _forward_spec(
                f"GroupedFwdRowTask{label}N512K2048J64",
                f"grouped_fwd_row_task_{suffix}_n512_k2048_j64",
                4,
                quant_type=quant_type,
                j=64,
                nrows_weight=512,
                blocks_per_weight_row=8,
                enforce_resource_gate=quant_name in {"Q3_K", "IQ2_S"},
            )
        )
    specs.extend(
        (
            _forward_spec(
                "GroupedFwdSerialIQ2SN2048K512J64J32",
                "grouped_fwd_serial_iq2_s_n2048_k512_j64_j32",
                3,
                quant_type=22,
                j=64,
                nrows_weight=2048,
                blocks_per_weight_row=2,
                mixed_iq2_s=True,
                enforce_resource_gate=True,
            ),
            _forward_spec(
                "GroupedFwdSerialIQ2XXSN2048K4096J64",
                "grouped_fwd_serial_iq2_xxs_n2048_k4096_j64",
                3,
                quant_type=16,
                j=64,
                nrows_weight=2048,
                blocks_per_weight_row=16,
                enforce_resource_gate=True,
            ),
            _forward_spec(
                "GroupedFwdSerialIQ2XXSN2048K4096J80",
                "grouped_fwd_serial_iq2_xxs_n2048_k4096_j80",
                3,
                quant_type=16,
                j=80,
                nrows_weight=2048,
                blocks_per_weight_row=16,
                enforce_resource_gate=True,
            ),
            _forward_spec(
                "GroupedFwdSerialQ2KN4096K2048J32",
                "grouped_fwd_serial_q2_k_n4096_k2048_j32",
                3,
                quant_type=10,
                j=32,
                nrows_weight=4096,
                blocks_per_weight_row=8,
                rolled_q2=True,
                enforce_resource_gate=True,
            ),
            _forward_spec(
                "GroupedFwdSerialQ2KN4096K2048J32J16",
                "grouped_fwd_serial_q2_k_n4096_k2048_j32_j16",
                3,
                quant_type=10,
                j=32,
                nrows_weight=4096,
                blocks_per_weight_row=8,
                rolled_q2=True,
                mixed_q2_k=True,
                enforce_resource_gate=True,
            ),
            _forward_spec(
                "GroupedFwdSerialQ5KN2048K512J32",
                "grouped_fwd_serial_q5_k_n2048_k512_j32",
                3,
                quant_type=13,
                j=32,
                nrows_weight=2048,
                blocks_per_weight_row=2,
                enforce_resource_gate=True,
            ),
        )
    )
    assert len(specs) == 37
    return specs


def _dense_backward_spec(
    cpp_id: str,
    suffix: str,
    quant_type: int,
    n_tiles: int,
    k_iteration: int,
    *,
    group_m: int = 2,
    m_tiles_per_wave: int = 1,
    decoder_width: int = 0,
    prefetch_local: bool = False,
    full_tiles: bool = False,
    prefetch_packed: bool = False,
    lds_padding: int = 0,
    vector_local_load: bool = False,
    lds_swizzle_chunk: int = 0,
    pack_q5_quant_bytes: bool = False,
    pack_q6_quant_bytes: bool = False,
) -> KernelSpec:
    defines = (
        ("MMQ_BUNDLE_QUANT_TYPE", quant_type),
        ("MMQ_BUNDLE_N_TILES", n_tiles),
        ("MMQ_BUNDLE_K_ITERATION", k_iteration),
        ("MMQ_BUNDLE_GROUP_M", group_m),
        ("MMQ_BUNDLE_M_TILES_PER_WAVE", m_tiles_per_wave),
        ("MMQ_BUNDLE_DECODER_WIDTH", decoder_width),
        ("MMQ_BUNDLE_PREFETCH_LOCAL", int(prefetch_local)),
        ("MMQ_BUNDLE_FULL_TILES", int(full_tiles)),
        ("MMQ_BUNDLE_PREFETCH_PACKED", int(prefetch_packed)),
        ("MMQ_BUNDLE_LDS_PADDING", lds_padding),
        ("MMQ_BUNDLE_VECTOR_LOCAL_LOAD", int(vector_local_load)),
        ("MMQ_BUNDLE_LDS_SWIZZLE_CHUNK", lds_swizzle_chunk),
        ("MMQ_BUNDLE_PACK_Q5_QUANT_BYTES", int(pack_q5_quant_bytes)),
        ("MMQ_BUNDLE_PACK_Q6_QUANT_BYTES", int(pack_q6_quant_bytes)),
    )
    return KernelSpec(
        cpp_id,
        suffix,
        DENSE_BACKWARD_WRAPPER,
        defines,
        enforce_resource_gate=True,
    )


def _dense_backward_specs() -> list[KernelSpec]:
    specs: list[KernelSpec] = []

    def generic(
        label: str,
        quant_type: int,
        n_tiles: int,
        group_m: int,
    ) -> None:
        nt = n_tiles * 16
        cpp_label = label.replace("_", "")
        specs.append(
            _dense_backward_spec(
                f"DenseBwd{cpp_label}NT{nt}KI16G{group_m}",
                f"dense_bwd_{_quant_suffix(label)}_nt{nt}_ki16_g{group_m}",
                quant_type,
                n_tiles,
                16,
                group_m=group_m,
            )
        )

    for label, quant_type, variants in (
        ("Q3_K", 11, ((1, 0), (4, 0), (4, 2), (8, 2), (12, 2), (16, 2))),
        ("Q4_K", 12, ((1, 0), (4, 0), (8, 2), (12, 2), (16, 2))),
        ("Q5_K", 13, ((1, 0), (4, 0), (8, 2), (12, 2), (16, 2))),
        ("IQ2_S", 22, ((1, 0), (4, 0), (4, 2), (12, 2), (16, 2))),
    ):
        for n_tiles, group_m in variants:
            generic(label, quant_type, n_tiles, group_m)

    specs.extend(
        (
            _dense_backward_spec(
                "DenseBwdQ3KFullWide",
                "dense_bwd_q3_k_mt128_nt128_ki32_full_wide",
                11,
                8,
                32,
                group_m=1,
                m_tiles_per_wave=2,
                decoder_width=16,
                prefetch_local=True,
                full_tiles=True,
                prefetch_packed=True,
                vector_local_load=True,
                lds_swizzle_chunk=8,
            ),
            _dense_backward_spec(
                "DenseBwdQ3KFullNarrow",
                "dense_bwd_q3_k_mt128_nt128_ki32_full_narrow",
                11,
                8,
                32,
                group_m=1,
                m_tiles_per_wave=2,
                decoder_width=16,
                prefetch_local=True,
                full_tiles=True,
                lds_padding=8,
                vector_local_load=True,
            ),
            _dense_backward_spec(
                "DenseBwdQ4KFullK4096",
                "dense_bwd_q4_k_mt128_nt128_ki32_full_k4096",
                12,
                8,
                32,
                group_m=1,
                m_tiles_per_wave=2,
                decoder_width=16,
                prefetch_local=True,
                full_tiles=True,
                prefetch_packed=True,
                lds_padding=8,
                vector_local_load=True,
            ),
            _dense_backward_spec(
                "DenseBwdQ4KFullK2048",
                "dense_bwd_q4_k_mt128_nt128_ki32_full_k2048",
                12,
                8,
                32,
                group_m=1,
                m_tiles_per_wave=2,
                decoder_width=16,
                prefetch_local=True,
                full_tiles=True,
                prefetch_packed=True,
                vector_local_load=True,
                lds_swizzle_chunk=8,
            ),
            _dense_backward_spec(
                "DenseBwdQ4KFullK512",
                "dense_bwd_q4_k_mt128_nt128_ki32_full_k512",
                12,
                8,
                32,
                group_m=1,
                m_tiles_per_wave=2,
                decoder_width=16,
                prefetch_local=True,
                full_tiles=True,
                prefetch_packed=True,
                vector_local_load=True,
                lds_swizzle_chunk=16,
            ),
            _dense_backward_spec(
                "DenseBwdQ5KFullK2048",
                "dense_bwd_q5_k_mt128_nt128_ki32_full_k2048",
                13,
                8,
                32,
                group_m=1,
                m_tiles_per_wave=2,
                decoder_width=16,
                prefetch_local=True,
                full_tiles=True,
                prefetch_packed=True,
                vector_local_load=True,
                lds_swizzle_chunk=8,
                pack_q5_quant_bytes=True,
            ),
            _dense_backward_spec(
                "DenseBwdQ5KFullK512",
                "dense_bwd_q5_k_mt128_nt128_ki32_full_k512",
                13,
                8,
                32,
                group_m=1,
                m_tiles_per_wave=2,
                decoder_width=16,
                prefetch_local=True,
                full_tiles=True,
                prefetch_packed=True,
                vector_local_load=True,
                lds_swizzle_chunk=4,
            ),
        )
    )

    for rows, n_tiles, k_iteration, m_tiles, swizzle, pack_q6 in (
        (64, 2, 64, 1, 16, False),
        (128, 4, 32, 2, 8, False),
        (256, 4, 32, 2, 8, True),
    ):
        for full in (False, True):
            specs.append(
                _dense_backward_spec(
                    f"DenseBwdQ6KM{rows}{'Full' if full else 'Bounded'}",
                    f"dense_bwd_q6_k_m{rows}_nt{n_tiles * 16}_ki{k_iteration}_"
                    f"{'full' if full else 'bounded'}",
                    14,
                    n_tiles,
                    k_iteration,
                    group_m=0,
                    m_tiles_per_wave=m_tiles,
                    prefetch_local=True,
                    full_tiles=full,
                    vector_local_load=True,
                    lds_swizzle_chunk=swizzle,
                    pack_q6_quant_bytes=pack_q6,
                )
            )
    specs.extend(
        (
            _dense_backward_spec(
                "DenseBwdQ6KNT128KI16G2",
                "dense_bwd_q6_k_nt128_ki16_g2",
                14,
                8,
                16,
            ),
            _dense_backward_spec(
                "DenseBwdQ6KNT256KI16G2",
                "dense_bwd_q6_k_nt256_ki16_g2",
                14,
                16,
                16,
            ),
        )
    )
    assert len(specs) == 36
    return specs


def _grouped_backward_spec(
    cpp_id: str,
    suffix: str,
    kind: int,
    quant_type: int = 0,
    enforce_resource_gate: bool = False,
) -> KernelSpec:
    return KernelSpec(
        cpp_id,
        suffix,
        GROUPED_BACKWARD_WRAPPER,
        (
            ("MMQ_BUNDLE_GROUPED_BWD_KIND", kind),
            ("MMQ_BUNDLE_QUANT_TYPE", quant_type),
        ),
        enforce_resource_gate,
    )


def _grouped_backward_specs() -> list[KernelSpec]:
    specs: list[KernelSpec] = []
    for quant_name, quant_type in BACKWARD_QUANT_TYPES:
        label = quant_name.replace("_", "")
        specs.append(
            _grouped_backward_spec(
                f"GroupedBwdSingle{label}Generic",
                f"grouped_bwd_single_{_quant_suffix(quant_name)}_generic",
                1,
                quant_type,
            )
        )
    for quant_name, quant_type in BACKWARD_QUANT_TYPES:
        label = quant_name.replace("_", "")
        specs.append(
            _grouped_backward_spec(
                f"GroupedBwdPair{label}Generic",
                f"grouped_bwd_pair_{_quant_suffix(quant_name)}_generic",
                2,
                quant_type,
            )
        )
    specs.extend(
        (
            _grouped_backward_spec(
                "GroupedBwdFixedQ80G8K4096",
                "grouped_bwd_fixed_q8_0_g8_k4096",
                15,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdSingleQ2KN4096K2048M64N64",
                "grouped_bwd_single_q2_k_n4096_k2048_mt64_nt64",
                16,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdSingleQ2KN4096K2048M128N64",
                "grouped_bwd_single_q2_k_n4096_k2048_mt128_nt64",
                17,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdPairIQ2XXSN2048K4096M64N64",
                "grouped_bwd_pair_iq2_xxs_n2048_k4096_mt64_nt64",
                18,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdSingleQ2KN4096K2048M128N64U2",
                "grouped_bwd_single_q2_k_n4096_k2048_mt128_nt64_u2",
                19,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdSingleQ4KN2048K512M64N64",
                "grouped_bwd_single_q4_k_n2048_k512_mt64_nt64",
                3,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdSingleQ4KN2048K512M128N64",
                "grouped_bwd_single_q4_k_n2048_k512_mt128_nt64",
                4,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdPairQ3KN512K2048M64N64",
                "grouped_bwd_pair_q3_k_n512_k2048_mt64_nt64",
                5,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdPairQ3KN512K2048M128N64",
                "grouped_bwd_pair_q3_k_n512_k2048_mt128_nt64",
                6,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdSingleQ5KN2048K512M64N64",
                "grouped_bwd_single_q5_k_n2048_k512_mt64_nt64",
                7,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdSingleIQ2SN2048K512M64N64",
                "grouped_bwd_single_iq2_s_n2048_k512_mt64_nt64",
                8,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdSingleIQ2SN2048K512M128N64",
                "grouped_bwd_single_iq2_s_n2048_k512_mt128_nt64",
                9,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdPairIQ2SN512K2048M64N64",
                "grouped_bwd_pair_iq2_s_n512_k2048_mt64_nt64",
                10,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdPairIQ2SN512K2048M128N64",
                "grouped_bwd_pair_iq2_s_n512_k2048_mt128_nt64",
                11,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdRowTaskQ4KN2048K512M128N128",
                "grouped_bwd_row_task_q4_k_n2048_k512_mt128_nt128",
                12,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdRowTaskQ5KN2048K512M128N128",
                "grouped_bwd_row_task_q5_k_n2048_k512_mt128_nt128",
                13,
                enforce_resource_gate=True,
            ),
            _grouped_backward_spec(
                "GroupedBwdRowTaskIQ2SN2048K512M128N128",
                "grouped_bwd_row_task_iq2_s_n2048_k512_mt128_nt128",
                14,
                enforce_resource_gate=True,
            ),
        )
    )
    assert len(specs) == 31
    return specs


def kernel_specs() -> tuple[KernelSpec, ...]:
    specs: list[KernelSpec] = [
        _forward_spec(
            "QuantizeQ81D4",
            "quantize_bf16_q8_1_d4",
            1,
            quant_type=8,
            enforce_resource_gate=True,
        ),
        _forward_spec(
            "QuantizeQ81DS4",
            "quantize_bf16_q8_1_ds4",
            1,
            quant_type=12,
            enforce_resource_gate=True,
        ),
        _forward_spec(
            "QuantizeQ81D2S6",
            "quantize_bf16_q8_1_d2s6",
            1,
            quant_type=10,
            enforce_resource_gate=True,
        ),
    ]
    for quant_name, quant_type in QUANT_TYPES:
        label = quant_name.replace("_", "")
        specs.append(
            _forward_spec(
                f"DenseFwd{label}J128",
                f"dense_fwd_{_quant_suffix(quant_name)}_j128",
                2,
                quant_type=quant_type,
                j=128,
            )
        )
    specs.append(
        _forward_spec(
            "DenseFwdQ6KJ64",
            "dense_fwd_q6_k_j64",
            2,
            quant_type=14,
            j=64,
            enforce_resource_gate=True,
        )
    )
    specs.append(
        _forward_spec(
            "GroupedRowTaskSetup",
            "grouped_row_task_setup",
            6,
        )
    )
    specs.extend(_grouped_forward_specs())
    specs.extend(_dense_backward_specs())
    specs.extend(_grouped_backward_specs())
    assert len(specs) == 117
    assert len({spec.cpp_id for spec in specs}) == len(specs)
    assert len({spec.symbol for spec in specs}) == len(specs)
    return tuple(specs)


def _find_tool(hipcc: Path, name: str) -> Path:
    candidates = (
        hipcc.with_name(name),
        hipcc.parent.parent / "lib" / "llvm" / "bin" / name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    found = shutil.which(name)
    if found:
        return Path(found)
    raise FileNotFoundError(f"cannot locate {name} beside {hipcc}")


def _compiler_identity(hipcc: Path) -> str:
    return subprocess.run(
        [str(hipcc), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _common_args(hipcc: Path) -> list[str]:
    return [
        str(hipcc),
        "--genco",
        "--no-gpu-bundle-output",
        "-O3",
        "-std=c++17",
        f"--offload-arch={ARCH}",
        f"-I{CSRC}",
        f"-ffile-prefix-map={ROOT}=.",
    ]


def _build_input_digest(
    hipcc: Path, compiler_identity: str, specs: tuple[KernelSpec, ...]
) -> str:
    digest = hashlib.sha256()
    digest.update(compiler_identity.encode())
    digest.update("\0".join(_common_args(hipcc)[1:]).encode())
    digest.update(json.dumps([asdict(spec) for spec in specs], sort_keys=True).encode())
    device_headers = [
        path
        for path in sorted(CSRC.rglob("*.cuh"))
        if "generated" not in path.parts and not path.name.endswith("_hip.cuh")
    ]
    for path in (Path(__file__), *WRAPPERS, *device_headers):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _header_text(specs: tuple[KernelSpec, ...], build_input: str) -> str:
    enum_values = "\n".join(
        f"    {spec.cpp_id} = {index}," for index, spec in enumerate(specs)
    )
    records = "\n".join(
        f'    MMQKernelSpec{{"{spec.symbol}", "{spec.filename}"}},' for spec in specs
    )
    return f"""// Generated by tools/build_mmq_bundle.py. Do not edit directly.
// Build input: {build_input}
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace torch_ggml_ops::mmq_bundle {{

enum class MMQKernelId : std::uint16_t {{
{enum_values}
    Count = {len(specs)},
}};

struct MMQKernelSpec {{
    const char * symbol;
    const char * filename;
}};

inline constexpr std::array<MMQKernelSpec, {len(specs)}> kMMQKernelSpecs{{{{
{records}
}}}};

inline constexpr const MMQKernelSpec & mmq_kernel_spec(MMQKernelId id) {{
    return kMMQKernelSpecs[static_cast<std::size_t>(id)];
}}

}} // namespace torch_ggml_ops::mmq_bundle
"""


def _verify_artifact(artifact: Path, spec: KernelSpec, readelf: Path) -> bytes:
    data = artifact.read_bytes()
    if not data.startswith(b"\x7fELF"):
        raise RuntimeError(f"{artifact} is not an ELF code object")
    if ARCH.encode() not in data:
        raise RuntimeError(f"{artifact} does not identify target {ARCH}")
    symbols = subprocess.run(
        [str(readelf), "--symbols", "--wide", str(artifact)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    exported_functions = {
        line.split()[-1]
        for line in symbols.splitlines()
        if " FUNC " in line and " GLOBAL " in line
    }
    if exported_functions != {spec.symbol}:
        raise RuntimeError(
            f"{artifact} exports {sorted(exported_functions)}, expected only {spec.symbol}"
        )
    notes = subprocess.run(
        [str(readelf), "--notes", str(artifact)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    def metadata_value(name: str) -> str:
        match = re.search(rf"\.{name}:\s+([^\s]+)", notes)
        if match is None:
            raise RuntimeError(f"{artifact} has no {name} metadata")
        return match.group(1)

    resources: dict[str, int | bool] = {
        "private_segment_bytes": int(metadata_value("private_segment_fixed_size")),
        "sgpr_spills": int(metadata_value("sgpr_spill_count")),
        "uses_dynamic_stack": metadata_value("uses_dynamic_stack") == "true",
        "vgpr_spills": int(metadata_value("vgpr_spill_count")),
    }
    if spec.enforce_resource_gate and (
        resources["private_segment_bytes"] != 0
        or resources["sgpr_spills"] != 0
        or resources["vgpr_spills"] != 0
        or resources["uses_dynamic_stack"]
    ):
        raise RuntimeError(f"kernel {spec.cpp_id} fails the resource gate: {resources}")
    return data


def _compile_one(
    spec: KernelSpec,
    output_dir: Path,
    hipcc: Path,
    readelf: Path,
    env: dict[str, str],
) -> tuple[str, bytes]:
    temporary = output_dir / f"{spec.cpp_id}.hsaco"
    command = [
        *_common_args(hipcc),
        f"-cuid={spec.cuid}",
        *spec.compiler_defines(),
        str(CSRC / spec.wrapper),
        "-o",
        str(temporary),
    ]
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"failed to compile {spec.cpp_id}\ncommand: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    data = _verify_artifact(temporary, spec, readelf)
    temporary.chmod(0o644)
    temporary.rename(output_dir / spec.filename)
    return spec.cpp_id, data


def _compile_all(
    specs: tuple[KernelSpec, ...], hipcc: Path, jobs: int
) -> tuple[Path, list[bytes]]:
    readelf = _find_tool(hipcc, "llvm-readelf")
    PACKAGE_DIR.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".mmq-gfx1151-", dir=PACKAGE_DIR.parent))
    env = os.environ.copy()
    env.update({"LC_ALL": "C", "LANG": "C", "SOURCE_DATE_EPOCH": "0"})
    try:
        images_by_id: dict[str, bytes] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(_compile_one, spec, staging, hipcc, readelf, env): spec
                for spec in specs
            }
            for future in concurrent.futures.as_completed(futures):
                cpp_id, image = future.result()
                images_by_id[cpp_id] = image
                print(f"built {cpp_id}", flush=True)
        return staging, [images_by_id[spec.cpp_id] for spec in specs]
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _bundle_is_current(build_input: str, specs: tuple[KernelSpec, ...]) -> bool:
    if not PACKAGE_DIR.is_dir() or not GENERATED_HEADER.is_file():
        return False
    expected_files = {spec.filename for spec in specs}
    actual_files = {path.name for path in PACKAGE_DIR.glob("*.hsaco")}
    return (
        actual_files == expected_files
        and GENERATED_HEADER.read_text() == _header_text(specs, build_input)
    )


def _install_bundle(
    staging: Path,
    specs: tuple[KernelSpec, ...],
    build_input: str,
) -> None:
    generated_staging = GENERATED_HEADER.with_suffix(".cuh.tmp")
    generated_staging.parent.mkdir(parents=True, exist_ok=True)
    generated_staging.write_text(_header_text(specs, build_input))

    PACKAGE_DIR.parent.mkdir(parents=True, exist_ok=True)
    old_dir = PACKAGE_DIR.with_name(PACKAGE_DIR.name + ".old")
    shutil.rmtree(old_dir, ignore_errors=True)
    if PACKAGE_DIR.exists():
        PACKAGE_DIR.rename(old_dir)
    staging.rename(PACKAGE_DIR)
    shutil.rmtree(old_dir, ignore_errors=True)
    os.replace(generated_staging, GENERATED_HEADER)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hipcc", type=Path)
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--verify-reproducible", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")

    hipcc_value = args.hipcc or (
        Path(shutil.which("hipcc")) if shutil.which("hipcc") else None
    )
    if hipcc_value is None or not hipcc_value.is_file():
        raise FileNotFoundError("hipcc is required to build the MMQ bundle")
    hipcc = hipcc_value.resolve()
    specs = kernel_specs()
    build_input = _build_input_digest(hipcc, _compiler_identity(hipcc), specs)

    if args.check:
        if not _bundle_is_current(build_input, specs):
            raise SystemExit("MMQ gfx1151 bundle is stale")
        print(f"MMQ gfx1151 bundle is current ({len(specs)} kernels)")
        return
    if (
        not args.force
        and not args.verify_reproducible
        and _bundle_is_current(build_input, specs)
    ):
        print(f"MMQ gfx1151 bundle is current ({len(specs)} kernels)")
        return

    first_dir, first_images = _compile_all(specs, hipcc, args.jobs)
    try:
        if args.verify_reproducible:
            second_dir, second_images = _compile_all(specs, hipcc, args.jobs)
            try:
                if first_images != second_images:
                    mismatches = [
                        spec.cpp_id
                        for spec, first, second in zip(
                            specs, first_images, second_images, strict=True
                        )
                        if first != second
                    ]
                    raise RuntimeError(
                        "non-reproducible MMQ artifacts: " + ", ".join(mismatches)
                    )
            finally:
                shutil.rmtree(second_dir, ignore_errors=True)
        _install_bundle(first_dir, specs, build_input)
    except Exception:
        shutil.rmtree(first_dir, ignore_errors=True)
        raise
    print(f"installed {len(specs)} MMQ kernels in {PACKAGE_DIR}")


if __name__ == "__main__":
    main()
