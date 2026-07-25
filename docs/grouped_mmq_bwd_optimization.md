# Grouped MMQ backward optimization

## Current status

Grouped MMQ backward is production-tuned for the current Qwen and DeepSeek GGUF representations on gfx1151. The local kernel optimization pass is complete.

Current source-of-record artifacts:

```text
Qwen final acceptance:     /tmp/grouped_mmq_bwd_qwen_tail_predicate_final_acceptance.json
DeepSeek final acceptance: /tmp/grouped_mmq_bwd_ds4_tail_predicate_final_acceptance.json
Qwen historical baseline:  /tmp/grouped_mmq_bwd_baseline_full.json
Qwen pre-tail final:       /tmp/grouped_mmq_bwd_final_full.json
```

Latest outcome:

- Qwen packed kernels win 35/60 individual case, batch, and routing points against predecoded BF16 AITER GMM.
- Both Qwen fused gate/up families win all 24 points. The checkpoint-weighted Qwen estimate wins all 12 batch/routing combinations by `1.255-2.159x`.
- DeepSeek wins all 27 references: fixed Q8_0 wins 3/3 against BF16 BMM, routed IQ2_XXS wins 12/12, and routed Q2_K wins 12/12 against predecoded BF16 AITER.
- The checkpoint-weighted DeepSeek estimate wins all 12 batch/routing combinations by `1.453-2.067x`.
- Every retained specialized kernel has zero private storage, zero VGPR spills, zero SGPR spills, and no dynamic stack.
- The complete gfx1151 bundle contains 118 kernels, including 32 grouped-backward entries. HSACOs are generated from source and are not committed.
- Remaining work is limited to a separate model-owned compact representation for Qwen Q4_K/IQ2_S single-down weights. No local kernel experiment remains pending, and DeepSeek should stay packed.

## Latest results

### Qwen

The latest Qwen matrix uses `~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf` and includes physical batches 1, 4, and 16 with uniform, skewed, sparse, and boundary routing.

| Family | Wins | Reference/packed latency range | Median | Packed latency by batch |
| --- | ---: | ---: | ---: | --- |
| Fused Q3_K gate/up | 12/12 | `1.596-3.006x` | `2.236x` | B1 `3.650-5.701 ms`; B4 `10.736-14.242 ms`; B16 `46.298-49.892 ms` |
| Fused IQ2_S gate/up | 12/12 | `1.451-2.926x` | `2.175x` | B1 `4.076-6.171 ms`; B4 `10.885-14.703 ms`; B16 `46.758-49.303 ms` |
| IQ2_S down | 1/12 | `0.813-1.294x` | `0.936x` | B1 `3.483-4.496 ms`; B4 `9.779-10.481 ms`; B16 `33.925-35.199 ms` |
| Q4_K down | 4/12 | `0.795-1.360x` | `0.940x` | B1 `3.551-4.257 ms`; B4 `9.701-9.917 ms`; B16 `33.034-33.824 ms` |
| Q5_K down | 6/12 | `0.814-1.310x` | `1.008x` | B1 `3.737-4.011 ms`; B4 `9.498-10.096 ms`; B16 `33.128-33.437 ms` |

Q4_K and Q5_K include the final inactive-M row-task suppression. Relative to sequential 25-repeat false controls:

- Q4_K improves every B4/B16 point by `5.27-11.60%`, with a `9.16%` geometric gain.
- Q5_K improves every B4/B16 point by `1.05-6.34%`, with a `3.68%` geometric gain.
- IQ2_S suppression was rejected. It improved geometrically by only `1.06%` and regressed B4 uniform and B16 boundary.

Checkpoint-weighted Qwen estimate, covering the five grouped-backward families and their checkpoint call counts:

| Batch | Route | Packed ms | AITER ms | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 1 | uniform | 307.4 | 574.2 | 1.868x |
| 1 | skewed | 411.6 | 579.1 | 1.407x |
| 1 | sparse | 381.2 | 478.5 | 1.255x |
| 1 | boundary | 397.7 | 582.6 | 1.465x |
| 4 | uniform | 826.7 | 1,339.3 | 1.620x |
| 4 | skewed | 977.5 | 1,589.8 | 1.626x |
| 4 | sparse | 958.6 | 1,511.8 | 1.577x |
| 4 | boundary | 985.8 | 1,618.7 | 1.642x |
| 16 | uniform | 3,200.5 | 6,908.3 | 2.159x |
| 16 | skewed | 3,353.8 | 7,002.3 | 2.088x |
| 16 | sparse | 3,351.2 | 7,088.9 | 2.115x |
| 16 | boundary | 3,350.2 | 6,979.1 | 2.083x |

The isolated single-down deficits do not overturn the checkpoint-weighted result. The fused pair families dominate the aggregate win because each combines two projections, keeps one FP32 accumulator set, rounds once, and allocates one output.

### DeepSeek

The latest DeepSeek matrix uses `~/models/ds4/DeepSeek-V4-Flash-IQ2XXS.gguf`. Routed cases use top-six routing with uniform, skewed, sparse, and boundary distributions. Fixed output-A uses its dedicated `[tokens, 8, features]` layout and has no fabricated routing metadata.

| Family | Wins | Reference/packed latency range | Median | Packed latency by batch |
| --- | ---: | ---: | ---: | --- |
| Fixed Q8_0 output-A | 3/3 | `1.154-1.225x` | `1.193x` | B1 `6.146 ms`; B4 `25.460 ms`; B16 `97.206 ms` |
| Fused IQ2_XXS gate/up | 12/12 | `1.531-2.401x` | `1.993x` | B1 `30.966-35.551 ms`; B4 `104.728-116.506 ms`; B16 `436.487-450.460 ms` |
| Q2_K down | 12/12 | `1.267-2.157x` | `1.614x` | B1 `18.934-21.931 ms`; B4 `50.027-51.971 ms`; B16 `173.735-186.494 ms` |

Checkpoint-weighted DeepSeek estimate includes 43 calls each of fixed output-A, fused gate/up, and routed down:

| Batch | Route | Packed ms | BF16 reference ms | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 1 | uniform | 2,410.0 | 4,981.5 | 2.067x |
| 1 | skewed | 2,736.0 | 4,981.1 | 1.821x |
| 1 | sparse | 2,596.0 | 3,909.4 | 1.506x |
| 1 | boundary | 2,566.0 | 4,986.2 | 1.943x |
| 4 | uniform | 7,806.5 | 11,342.8 | 1.453x |
| 4 | skewed | 8,286.1 | 13,060.8 | 1.576x |
| 4 | sparse | 8,149.6 | 12,799.8 | 1.571x |
| 4 | boundary | 8,339.3 | 12,548.6 | 1.505x |
| 16 | uniform | 30,419.4 | 59,413.1 | 1.953x |
| 16 | skewed | 31,335.3 | 59,868.4 | 1.911x |
| 16 | sparse | 31,477.9 | 58,700.3 | 1.865x |
| 16 | boundary | 31,545.3 | 61,703.8 | 1.956x |

DeepSeek does not need an expanded decoded representation. Every packed family already beats its ideal predecoded BF16 arithmetic reference before decode allocation or cache management is added.

## Scope and contracts

This document covers three input-gradient operators:

- `grouped_mmq_grad_input` for one routed frozen packed projection.
- `grouped_mmq_pair_grad_input` for fused routed gate/up input gradients.
- `fixed_grouped_mmq_grad_input` for DeepSeek's fixed eight-group output-A layout.

For one routed expert group:

```text
forward:  Y[M, N]  = X[M, K]  @ W[N, K].T
backward: dX[M, K] = dY[M, N] @ W[N, K]
```

The public cotangent and result are BF16. WMMA accumulation is FP32. The authoritative weights remain packed GGUF tensors.

Production invariants:

- Use the current CUDA/HIP stream.
- Do not materialize a logical dense weight matrix in the ordinary path.
- Do not launch arithmetic workgroups for inactive experts.
- Keep route metadata device-resident.
- Do not add `.item()`, host descriptors, device-to-host route copies, or hidden synchronization.
- Preserve the fused pair's one FP32 accumulator set, one BF16 rounding, and output-only allocation.
- Preserve fixed output-A's `[tokens, 8, features]` layout. It is not a routed operator.

Routed metadata ABI:

```text
expert_indices: contiguous CUDA int64, shape [G]
expert_offsets: contiguous CUDA int32, shape [G]
expert_offsets[-1] = total routed rows R
G <= 256
```

Main implementation files:

```text
csrc/ck/grouped_mmq_backward.cuh
csrc/ck/grouped_mmq_backward_tiled.cuh
csrc/ck/gguf_decode.cuh
csrc/mmq_bundle.cpp
csrc/generated/mmq_bundle_table.cuh
tools/build_mmq_bundle.py
tools/mmq_bundle_wrapper_source.py
```

Project-specific decode logic stays outside `csrc/vendor/llama_cpp/*`.

## Workloads and references

### Qwen workload

| Case | Weight `(N, K)` | GGUF type | Calls/layers | Operator |
| --- | ---: | --- | ---: | --- |
| Gate/up outer | `(512, 2048)` | Q3_K | 20 pairs | fused pair |
| Gate/up middle | `(512, 2048)` | IQ2_S | 20 pairs | fused pair |
| Down middle | `(2048, 512)` | IQ2_S | 20 | single |
| Down outer main | `(2048, 512)` | Q4_K | 18 | single |
| Down outer edge | `(2048, 512)` | Q5_K | 2 | single |

Qwen sequence length is 2,048 and top-k is 8.

| Physical batch | Routed rows | Uniform rows/expert |
| ---: | ---: | ---: |
| 1 | 16,384 | 64 |
| 4 | 65,536 | 256 |
| 16 | 262,144 | 1,024 |

### DeepSeek workload

| Case | Weight `(N, K)` | GGUF type | Calls | Operator |
| --- | ---: | --- | ---: | --- |
| Fixed output-A, 8 groups | `(1024, 4096)` per group | Q8_0 | 43 | fixed-group single |
| Routed gate/up, 256 experts | `(2048, 4096)` | IQ2_XXS | 43 pairs | fused pair |
| Routed down, 256 experts | `(4096, 2048)` | Q2_K | 43 | single |

DeepSeek sequence length is 2,048 and top-k is 6.

| Physical batch | Token rows | Routed rows | Uniform rows/expert |
| ---: | ---: | ---: | ---: |
| 1 | 2,048 | 12,288 | 48 |
| 4 | 8,192 | 49,152 | 192 |
| 16 | 32,768 | 196,608 | 768 |

### Routing distributions

Routed benchmarks cover four deterministic distributions:

- `uniform`: all experts have equal group sizes.
- `skewed`: all experts are active with deterministic nonuniform sizes.
- `sparse`: 192, 224, and 240 active experts at batches 1, 4, and 16.
- `boundary`: includes sizes 1, 15, 16, 17, 63, 64, 65, 127, 128, and 129.

These distributions are mandatory. Uniform-only movement is not sufficient to retain a dispatch rule because it does not expose inactive experts, rounded tails, or route imbalance.

For Qwen 128-row tasks, expert-local task counts grow from 192-257 at B1 to 512-660 at B4 and 2,048-2,174 at B16, depending on route distribution. This is why task ordering and nonuniform launch cost were measured directly rather than inferred from uniform routing.

### BF16 references

Routed performance uses AITER Triton `gmm` configured by `torch_ggml_ops.aiter_gmm_heuristics.gmm_config`. Fixed Q8_0 uses BF16 BMM in the public fixed-group layout.

The timed references start with independently dequantized BF16 weights. Dequantization and active-expert selection are setup costs and are not included. This makes AITER an ideal predecoded arithmetic reference, not a complete packed-weight alternative.

The Qwen heuristic selects M128/N128/K64 with 256 threads and 256 persistent programs for gate/up, and M64/N128/K64 with 256 threads and 256 persistent programs for down. AITER is a production reference, not a performance ceiling.

For fused pairs, AITER runs two GMM calls and adds two BF16 outputs. It is the production performance reference but not a bitwise numerical oracle because the packed pair accumulates both projections in FP32 and rounds once.

Correctness references:

- Qwen supported types use the independent dense packed path.
- DeepSeek Q2_K and IQ2_XXS use independently dequantized BF16 weights.
- Fixed Q8_0 uses independent GGUF decode and BF16 BMM in `[tokens, 8, K]` layout.

### Measurement rules

- Run GPU benchmarks sequentially with real nonzero tensors.
- Warm module loading before timing.
- Use three warmups and nine repeats for matrix coverage.
- Any fresh median movement above 1% requires a sequential 25-repeat A/B control.
- Include route-task setup, allocation, decode, arithmetic, and layout handling in public latency.
- Retain a local candidate only when a sequential control shows at least one material target gain, normally 2% or more, without a repeatable family regression above 1%.
- Use `PYTHONPATH=.` so benchmarks load the in-tree extension.
- Build with all CPU cores:

```bash
python tools/build_mmq_bundle.py --force --jobs "$(nproc)"
```

Latest acceptance commands:

```bash
PYTHONPATH=. python bench/benchmark_grouped_mmq_bwd.py \
  --model ~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf \
  --model-family qwen \
  --batches 1,4,16 \
  --distributions uniform,skewed,sparse,boundary \
  --warmup 3 --repeats 9 \
  --output /tmp/grouped_mmq_bwd_qwen_tail_predicate_final_acceptance.json

PYTHONPATH=. python bench/benchmark_grouped_mmq_bwd.py \
  --model ~/models/ds4/DeepSeek-V4-Flash-IQ2XXS.gguf \
  --model-family deepseek \
  --batches 1,4,16 \
  --distributions uniform,skewed,sparse,boundary \
  --warmup 3 --repeats 9 \
  --output /tmp/grouped_mmq_bwd_ds4_tail_predicate_final_acceptance.json
```

## Production implementation

### Dispatch

Dispatch uses only host-visible shape, quant type, total rows, and group count. It does not inspect device offsets or use online autotuning.

| Model/family | Production dispatch |
| --- | --- |
| Qwen Q3_K pair | M64/N64/K32 below `128 * num_groups`; M128/N64/K32 otherwise |
| Qwen IQ2_S pair | M64/N64/K32 below `128 * num_groups`; M128/N64/K32 otherwise |
| Qwen Q4_K down | M64/N64 below 80 rows/group; M128/N64 at 80-127; M128/N128 row tasks at 128+ |
| Qwen Q5_K down | M64/N64 below 128 rows/group; M128/N128 row tasks at 128+ |
| Qwen IQ2_S down | M64/N64 below 80 rows/group; M128/N64 at 80-127; M128/N128 row tasks at 128+ |
| DeepSeek IQ2_XXS pair | M64/N64/K32, width-16 decode, swizzle4 at all route sizes |
| DeepSeek Q2_K down | M64/U1 below 128 rows/group; M128/U2 at 128-511; M128/U1 at 512+ |
| DeepSeek fixed Q8_0 | M256/N64/K32, width-16 decode, no swizzle at all batches |
| Unsupported type or shape | Generic grouped compatibility kernel |

Every selected kernel remains bounded-correct for nonuniform groups. Average rows per group is only a host-visible dispatch hint.

### Retained arithmetic mechanisms

Common choices:

- Four wave32 waves and 128 threads.
- K32 reduction tiles.
- Cooperative width-16 packed decode matched to natural quant metadata sharing.
- FP32 WMMA accumulation and BF16 stores.
- Exact production shapes and static template arguments.
- Short decode-temporary lifetimes.
- Separate pair and down LDS layouts where measured behavior differs.
- Zero private storage and zero spills as hard retention gates.

Qwen-specific choices:

- Q3_K pair uses padded LDS rows.
- Q4_K down uses a sixteen-BF16 XOR layout.
- Q5_K uses swizzle4 for M64 and swizzle8 for row tasks.
- IQ2_S down uses a sixteen-BF16 XOR layout.
- IQ2_S pair uses a four-BF16 XOR layout.
- Large down kernels use device-built M-major 128-row tasks. All four adjacent N workgroups for a row task remain contiguous.
- Q4_K and Q5_K row tasks suppress wholly inactive 16-row M minitiles while keeping decode and barriers unconditional.
- IQ2_S row-task suppression is disabled because its route-level result was unstable.

DeepSeek-specific choices:

- IQ2_XXS pair uses two separate weight LDS tiles, width-16 decode, swizzle4, and inactive-M consumer suppression.
- Q2_K uses width-16 decode that shares each scale/min group and packed shift. Inactive-M consumer suppression is enabled in all three production wrappers.
- Fixed Q8_0 preserves token-major public layout and stages one unswizzled N64/K32 weight tile per fixed group.

### Device row tasks

Qwen large single-down paths use an atomics-free 256-thread prefix-sum setup to build device-resident 128-row tasks. Setup averages approximately `0.004 ms`, so caching route tasks is not a meaningful optimization target.

M-major ordering is required. N-major ordering nearly doubled B16 latency. Pair and small-row paths stay serial. DeepSeek Q2_K stays serial because it already exposes 32 N workgroups per expert and thousands of workgroups overall; adding row tasks would not remove rounded tail arithmetic.

### Code-object resources

Current specialized resources:

| Kernel | VGPR | SGPR | LDS bytes |
| --- | ---: | ---: | ---: |
| Qwen Q3_K pair M64/N64 | 183 | 26 | 10,240 |
| Qwen Q3_K pair M128/N64 | 206 | 26 | 10,240 |
| Qwen IQ2_S pair M64/N64 | 194 | 54 | 8,192 |
| Qwen IQ2_S pair M128/N64 | 219 | 54 | 8,192 |
| Qwen Q4_K down M64/N64 | 96 | 22 | 4,096 |
| Qwen Q4_K down M128/N64 | 173 | 30 | 4,096 |
| Qwen Q4_K row task M128/N128 | 216 | 30 | 8,192 |
| Qwen Q5_K down M64/N64 | 115 | 22 | 4,096 |
| Qwen Q5_K row task M128/N128 | 234 | 26 | 8,192 |
| Qwen IQ2_S down M64/N64 | 90 | 22 | 4,096 |
| Qwen IQ2_S down M128/N64 | 161 | 30 | 4,096 |
| Qwen IQ2_S row task M128/N128 | 238 | 24 | 8,192 |
| DeepSeek IQ2_XXS pair M64/N64 | 209 | 54 | 8,192 |
| DeepSeek Q2_K M64/N64/U1 | 102 | 30 | 4,096 |
| DeepSeek Q2_K M128/N64/U1 | 146 | 31 | 4,096 |
| DeepSeek Q2_K M128/N64/U2 | 160 | 31 | 4,096 |
| DeepSeek fixed Q8_0 M256/N64 | 209 | 23 | 4,096 |

Every listed kernel has zero private bytes, zero VGPR/SGPR spills, and `uses_dynamic_stack: false`.

### Correctness and allocation

Qwen and DeepSeek single-projection samples are exact against their packed or independently dequantized BF16 references. Q2_K has zero NRMSE; differing-element counts with zero absolute error are signed-zero differences.

Fused Qwen and DeepSeek pairs remain in the expected one-rounding envelope, approximately `0.00286` NRMSE against two separately rounded BF16 projections plus addition. Fixed Q8_0 is approximately `5e-5` NRMSE against BF16 BMM.

Qwen fused pair incremental allocation remains output-only:

| Batch | Packed pair | AITER pair |
| ---: | ---: | ---: |
| 1 | 64 MiB | 192 MiB |
| 4 | 256 MiB | 768 MiB |
| 16 | 1,024 MiB | 3,072 MiB |

Qwen row-task metadata adds only about 9-10 KiB at batch 4 and 27-28 KiB at batch 16.

## Remaining work

### Local kernel work

No evidence-backed local grouped-backward experiment remains pending.

The following neighborhoods are closed by direct controls:

- Runtime full/tail branching and split full/tail task lists.
- N-major tasks, fixed persistent traversal, and broad row-task geometry sweeps.
- K64, two-LDS buffering, GSU, split-K, grouped Stream-K, and direct-to-VGPR variants.
- Broad swizzle, decoder-width, prefetch, and extraction sweeps.
- DeepSeek Q2_K N128 and row-task variants.
- DeepSeek IQ2_XXS width32, M128, and larger swizzles.
- Fixed Q8_0 wider decode, swizzle4, and M512.
- Qwen IQ2_S inactive-M row-task suppression.

A local kernel experiment should be reopened only if new profiler evidence contradicts the current diagnosis. Raw instruction ordering, code-object offsets, and timing movement from byte-identical binaries are not sufficient evidence.

### Representation-level Qwen work

The only substantive remaining optimization is a separate model-owned compact representation for Qwen Q4_K and IQ2_S single-down weights. This is not an operator-internal cache.

A viable project must define:

- A prepare API and the exact lossless integer-plus-scale bytes stored per projection.
- Ownership and lifetime across forward and backward.
- Mutation/version invalidation.
- Model-load and peak-memory policy.
- Cold-build and steady-state latency.
- Forward/backward sharing that repays representation construction.
- Behavior for inactive experts and route changes.

The packed GGUF tensor remains authoritative. The likely direction is a WMMA-friendly tile-major integer-plus-scale representation, not a BF16 shadow.

Already rejected representation controls:

- A transient BF16 materialization floor was `1.15-1.34x` slower for IQ2_S and `1.08-1.60x` slower for Q4_K before real packed reads and decode were added.
- One transient projection needs 512 MiB of BF16 workspace and 528-768 MiB incremental peak memory.
- Persistent BF16 shadows require 19 GiB for the remaining Q4_K/IQ2_S down tensors and 60 GiB for all benchmarked Qwen expert projections.
- Persistent expansion is 6.24x for IQ2_S and 3.56x for Q4_K over GGUF storage.
- Ideal predecoded AITER saves only 22.1 ms model-wide at B1 and 43.9 ms at B4 for the two down families, then loses 401.1 ms at B16.

Acceptance gates for a compact representation:

1. Preserve lossless GGUF semantics and BF16 public inputs/outputs.
2. Include preparation, workspace, synchronization, and invalidation costs.
3. Preserve sparse inactive-expert behavior.
4. Improve grouped and dense shared-down controls, proving that common decode cost was removed.
5. Improve all four route distributions or provide a legal shape-based dispatch boundary.
6. Preserve pair one-rounding and output allocation when pair families use the representation.
7. State cold-call and steady-state memory/latency explicitly.

DeepSeek should remain packed. Its kernels already beat ideal predecoded references at every measured point.

## Optimization logs

All timings below are historical step measurements. They explain production choices and should not replace the latest acceptance artifacts.

### DeepSeek P0-P1: harness, baseline, and diagnosis

DeepSeek support added exact Q8_0, Q2_K, and IQ2_XXS decoding, independent BF16 references, routed single/pair APIs, and a dedicated fixed-group operator. Fixed output-A was not represented with fabricated routing metadata.

Generic B1/B4 baseline:

| Family | B1 packed | B4 packed | Reference ratio |
| --- | ---: | ---: | ---: |
| Fixed Q8_0 | 51.038 ms | 212.396 ms | `0.140-0.151x` |
| IQ2_XXS pair | 172.158-189.815 ms | 1,046.712-1,075.508 ms | `0.155-0.411x` |
| Q2_K down | 72.464-94.062 ms | 475.348-519.821 ms | `0.142-0.372x` |

Generic resources were spill-free: Q2_K used 46 VGPRs/24 SGPRs/512-byte LDS, IQ2_XXS pair used 65/40/3,072, and fixed Q8_0 used 35/18/512. The defect was narrow eight-wave N16/K16 ownership and serial row work, not spills. This justified spending resources on four-wave N64/K32 reuse.

Artifacts:

```text
/tmp/grouped_mmq_bwd_ds4_baseline_b1_b4.json
/tmp/grouped_mmq_bwd_qwen_pre_ds4_control.json
/tmp/grouped_mmq_bwd_ds4_fixed_harness_check.json
/tmp/grouped_mmq_bwd_ds4_routed_harness_check.json
```

### DeepSeek P2: fixed Q8_0

P2 replaced the generic body with an exact four-wave N64/K32 kernel.

| M tile | B1 | B4 | B16 |
| ---: | ---: | ---: | ---: |
| 64 | 10.693 ms | 43.965 ms | 173.436 ms |
| 128 | 6.400 ms | 26.059 ms | 101.364 ms |
| 256 | 6.237 ms | 25.427 ms | 97.204 ms |

A sequential 25-repeat control confirmed M256 over M128 by `3.91-4.42%`. Width32 regressed `2.4-4.3%` and raised VGPR use. Swizzle4 regressed `3.8-5.1%`. M512 was not compiled because doubling the M256 accumulator set would cross the practical 256-VGPR warning boundary without adding weight reuse.

The retained M256/N64/K32 width16 unswizzled body provides `8.09x`, `8.33x`, and `8.75x` speedups over the generic fixed kernel at B1/B4/B16.

Artifacts:

```text
/tmp/grouped_mmq_bwd_ds4_q80_m64_full.json
/tmp/grouped_mmq_bwd_ds4_q80_m128_full.json
/tmp/grouped_mmq_bwd_ds4_q80_m256_full.json
/tmp/grouped_mmq_bwd_ds4_q80_m256_control_25.json
/tmp/grouped_mmq_bwd_ds4_q80_m128_control_25.json
/tmp/grouped_mmq_bwd_ds4_q80_m256_w32_full.json
/tmp/grouped_mmq_bwd_ds4_q80_m256_s4_full.json
/tmp/grouped_mmq_bwd_ds4_q80_generic_control_25.json
```

### DeepSeek P3: IQ2_XXS pair

P3 introduced exact `(N,K)=(2048,4096)` M64/N64/K32 pair ownership and cooperative width16 decode. The decoder shares each packed word, two grid lookups, parity-adjusted signs, and scale across sixteen values.

An early decoder took the address of a local packed word and created an 8-byte private segment. Shift/mask extraction restored a register-only body. M128 remained at 8 private bytes and was rejected before timing.

Layout controls:

- Width32 had mixed movement: two B1 wins but sparse/boundary and B4 regressions with no legal host-visible separator. Width16 was retained.
- Swizzle4 improved all 12 points by about 20-28% over unswizzled.
- Swizzle16 regressed 24-39% relative to swizzle4.
- Inactive-M suppression improved every point: B1 `5.45-8.44%`, B4 `0.89-4.44%`, B16 `0.58-1.60%`, and `4.08%` geometrically.

Artifacts:

```text
/tmp/grouped_mmq_bwd_ds4_tiled_focus.json
/tmp/grouped_mmq_bwd_ds4_iq2xxs_width32_focus.json
/tmp/grouped_mmq_bwd_ds4_iq2xxs_swizzle4_full.json
/tmp/grouped_mmq_bwd_ds4_iq2xxs_swizzle16_focus.json
/tmp/grouped_mmq_bwd_ds4_iq2xxs_tail_predicate_baseline_25.json
/tmp/grouped_mmq_bwd_ds4_iq2xxs_tail_predicate_candidate_25.json
```

### DeepSeek P4: Q2_K down

P4 introduced M64 and M128 N64/K32 bodies. Width16 decode shares one Q2_K scale/min group and packed shift across sixteen values.

Reduction unroll controls:

- U2 improved all B4 routes but regressed B1 and one B16 boundary point, so it is dispatched only for 128-511 rows/group.
- Sequential 25-repeat B4 controls confirmed U2 over U1 by `1.83-2.83%` across all four routes.
- U4 lost to U2 by `1.2-4.3%` despite remaining spill-free at 174 VGPRs.

Inactive-M suppression keeps decode and barriers unconditional and skips only cotangent loads, WMMA, and stores for wholly inactive 16-row minitiles. It improved B1 by `8.69-15.15%`, kept every B4 movement within `0.67%`, and improved the complete 12-point matrix geometrically by `4.73%`.

The bounded N128/U1 control compiled at 255 VGPRs and 8 KiB LDS with no spills, but a valid sequential B16 bracket lost every route by `1.18-4.18%`, or `3.06%` geometrically. The temporary wrapper and N-width generalization were removed.

Q2_K row tasks were not added. N64 already launches 32 N workgroups per expert and thousands overall, while row tasks would not eliminate rounded tail arithmetic.

Artifacts:

```text
/tmp/grouped_mmq_bwd_ds4_q2k_u2_dispatch_control_25.json
/tmp/grouped_mmq_bwd_ds4_q2k_u1_dispatch_control_25.json
/tmp/grouped_mmq_bwd_ds4_q2k_tail_predicate_baseline_25.json
/tmp/grouped_mmq_bwd_ds4_q2k_tail_predicate_candidate_25.json
/tmp/grouped_mmq_bwd_ds4_q2k_n128_candidate_valid_b16_25.json
/tmp/grouped_mmq_bwd_ds4_q2k_n64_bracket_b16_25.json
```

### Qwen GB0: baseline and diagnosis

The historical generic baseline won 0/60 points and was `2.39-6.07x` slower than AITER by checkpoint-weighted latency.

| Family | Mean packed/reference throughput | Best | Worst |
| --- | ---: | ---: | ---: |
| Q3_K pair | 0.316x | 0.511x | 0.193x |
| IQ2_S pair | 0.281x | 0.458x | 0.166x |
| Q4_K down | 0.185x | 0.313x | 0.114x |
| Q5_K down | 0.181x | 0.308x | 0.109x |
| IQ2_S down | 0.168x | 0.308x | 0.066x |

The old kernel used eight waves, N16/K16 ownership, one accumulator per wave, scalar decode, and serial 128-row chunks. It was spill-free at 46-52 VGPRs for singles and 58-89 VGPRs for pairs. Profiling showed very low LDS stalls and high Q4_K L2 hit rate, ruling out spills and LDS banking as first-order causes.

A one-expert 1,024-row diagnostic showed dense packed speedups of `5.08x` for Q3_K pair, `4.15x` for Q4_K, and `1.95x` for IQ2_S. At 64 rows, grouped and dense Q4_K/IQ2_S were approximately equal. This established that small and large groups needed different bodies and that metadata overhead was not the large-group bottleneck.

Artifacts:

```text
/tmp/grouped_mmq_bwd_baseline_full.json
/tmp/grouped_mmq_bwd_one_expert.txt
/tmp/rocprof_grouped_bwd_baseline_gate_b16_packed
/tmp/rocprof_grouped_bwd_baseline_down_q4_b4_packed
/tmp/rocprof_grouped_bwd_baseline_down_iq2_b4_sparse_packed
```

### Qwen GB1-GB3: tiled Q4_K, fused Q3_K, and small bodies

GB1 retained Q4_K M128/N128/K32 with width16 decode and a sixteen-BF16 XOR layout. It improved representative B4/B16 points by 5-8x over baseline.

GB2 retained a fused Q3_K pair with separate padded weight tiles, one accumulator set, and one final BF16 rounding. Representative B4/B16 points improved by 8-14x over baseline and beat AITER by about 2.1-2.9x.

GB3 introduced exact M64/N64/K32 S1 bodies. Universal M128 S2 was rejected because 64-row uniform groups became half-empty bounded tiles: Q3_K uniform regressed from 3.614 to 6.936 ms and Q4_K from 3.590 to 4.185 ms. Q4_K S2 was retained only at 80-127 rows/group; a 25-repeat sparse control measured 4.037 ms S2 versus 5.201 ms S1.

Artifacts:

```text
/tmp/grouped_mmq_bwd_step1_q4_matrix.json
/tmp/grouped_mmq_bwd_step2_q3_pair_matrix.json
/tmp/grouped_mmq_bwd_step3_s1.json
/tmp/grouped_mmq_bwd_step3_s2.json
/tmp/grouped_mmq_bwd_q4_sparse_s2_25.json
/tmp/grouped_mmq_bwd_q4_sparse_s1_25.json
```

### Qwen GB4-GB6: row tasks, Q5_K, and IQ2_S

Large Q4_K, Q5_K, and IQ2_S down paths retained device-built M-major 128-row tasks.

Representative task speedups over serial:

| Family/point | Serial | M-major tasks | Speedup |
| --- | ---: | ---: | ---: |
| Q4_K B16 uniform | 40.270 ms | 35.821 ms | 1.12x |
| Q5_K B16 uniform | 37.421 ms | 34.074 ms | 1.10x |
| IQ2_S B16 uniform | 38.992 ms | 32.806 ms | 1.19x |

Rejected scheduling controls:

- N-major order nearly doubled B16 latency.
- Fixed 1,024-program traversal regressed Q4_K/IQ2_S and gave Q5_K 10 spills plus a 44-byte private segment.
- Runtime full/tail branching produced private segments and 2-4 spills.
- Split full/tail lists improved uniform B16 modestly but regressed every nonuniform B4 point by 6-14% because of the second launch.

Q5_K reused the Q4_K framework with width16 low/high decode and bounded packed prefetch. Prefetch improved the initial port by 12-24%. Q5_K S2 was rejected: 4.065 ms versus 3.913 ms S1 on sparse B1.

The project-owned IQ2_S decoder reconstructs sixteen aligned values from two grid entries, two sign bytes, one shared scale nibble, and one `d` factor. M256/N64 large down was rejected despite no spills because doubling N workgroups outweighed M reuse. Selective M128/N64 S2 was retained at 80-127 rows/group, improving B1 sparse from 5.537 to 3.618 ms.

Artifacts:

```text
/tmp/grouped_mmq_bwd_step4_row_tasks_mmajor.json
/tmp/grouped_mmq_bwd_step4_row_tasks_nmajor.json
/tmp/grouped_mmq_bwd_step4_persistent1024.json
/tmp/grouped_mmq_bwd_step7_split_tasks.json
/tmp/grouped_mmq_bwd_step5_q5_prefetch.json
/tmp/grouped_mmq_bwd_q5_sparse_s2_25.json
/tmp/grouped_mmq_bwd_step6_iq2.json
/tmp/grouped_mmq_bwd_step6_iq2_n64_reuse.json
```

### Qwen GB7 and final local controls: pair geometry, layouts, and tails

IQ2_S down retained swizzle16; IQ2_S pair retained swizzle4. Swizzle4 regressed down by 20-35% but improved pair by 15-25%, proving that pair and single layouts must remain separate.

IQ2_S and Q3_K large pairs retained M128/N64/K32. Reducing N relieved pair resource pressure and improved representative points by 1-7%. Width8 IQ2_S decode was rejected because it duplicated scale work and doubled loader groups; pair B16 uniform regressed from 44.101 to 50.274 ms and down B16 uniform from 32.959 to 42.949 ms.

Q5_K swizzle8 was retained only for row tasks. A universal swizzle8 regressed M64/B1 by 5.6-22.5%, while sequential controls improved all row-task routes by `4.2-6.4%` at B4 and `5.2-6.6%` at B16. Row-task VGPR use fell from 256 to 233 before final inactive-M suppression.

Final inactive-M row-task controls retained Q4_K and Q5_K and rejected IQ2_S, as summarized in the latest Qwen results.

Artifacts:

```text
/tmp/grouped_mmq_bwd_step7_iq2_swizzle4.json
/tmp/grouped_mmq_bwd_step7_iq2_pair_n64.json
/tmp/grouped_mmq_bwd_step7_q3_pair_n64.json
/tmp/grouped_mmq_bwd_step7_iq2_width8.json
/tmp/grouped_mmq_bwd_qwen_q5_swizzle8_rowtask_control_25.json
/tmp/grouped_mmq_bwd_qwen_q5_swizzle4_rowtask_control_25.json
/tmp/grouped_mmq_bwd_qwen_rowtask_tail_predicate_control_false_25.json
/tmp/grouped_mmq_bwd_qwen_rowtask_tail_predicate_control_true_25.json
```

### Qwen representation controls

The remaining Q4_K/IQ2_S deficit appears in dense shared-down controls as well as grouped kernels:

```text
Q4_K dense shared-down: 5.148 ms packed versus 4.223 ms BF16
Q5_K dense shared-down: 5.547 ms packed versus 4.210 ms BF16
```

Representative profiling found approximately `0.004 ms` task construction, low LDS stalls, high Q4_K L2 hit rate, and zero scratch/private storage. This closes task construction, LDS buffering, and spill removal as explanations.

| Historical profile point | OccupancyPercent | L2CacheHit | ALUStalledByLDS |
| --- | ---: | ---: | ---: |
| Q3_K pair B16 uniform | 43.45 | 58.24 | 0.16 |
| Q4_K down B4 skewed | 30.85 | 82.45 | 0.77 |

Profiler timing is intentionally not the latency source of record because collection perturbs execution. Event medians in the JSON matrices remain authoritative.

The transient BF16 and persistent shadow controls were rejected for the latency and memory reasons recorded under Remaining work.

Artifacts:

```text
/tmp/mmq_bwd_grouped_final_dense_control.json
/tmp/grouped_mmq_bwd_qwen_transient_bf16_floor.json
/tmp/rocprof_grouped_bwd_final_down_q4_b4
/tmp/rocprof_grouped_bwd_final_down_iq2_b4
```

## Rejected and closed experiments

| Family | Candidate | Decision and reason |
| --- | --- | --- |
| Qwen common | Universal S2 | Half-empty 64-row groups regressed uniform B1 |
| Qwen Q5_K | Selective S2 | 4.065 ms versus 3.913 ms S1 on sparse B1 |
| Qwen IQ2_S | M256/N64 | More state and twice the N workgroups regressed B4/B16 |
| Qwen row tasks | N-major order | Nearly doubled B16 latency |
| Qwen row tasks | Fixed 1,024-program traversal | Slower and spilled Q5_K |
| Qwen row tasks | Runtime full/tail branch | Private storage and 2-4 spills |
| Qwen row tasks | Split full/tail lists | Second launch regressed nonuniform B4 by 6-14% |
| Qwen IQ2_S | Width8 decode | Duplicated scale work and loader groups |
| Qwen IQ2_S | Shared pair/down swizzle | Opposite measured preferences; keep layouts separate |
| Qwen Q5_K | Universal swizzle8 | Regressed B1; retain only for row tasks |
| Qwen IQ2_S | Inactive-M row-task suppression | Mixed route movement and two regressions |
| DeepSeek IQ2_XXS | M128/N64 | 8-byte private segment failed the gate |
| DeepSeek IQ2_XXS | Width32 | Mixed route movement with no legal host separator |
| DeepSeek IQ2_XXS | Swizzle0 or swizzle16 | Swizzle4 was uniformly faster; swizzle16 lost 24-39% |
| DeepSeek Q2_K | U4 | Lost to U2 by 1.2-4.3% |
| DeepSeek Q2_K | N128/U1 | 255-VGPR cliff and 3.06% geometric regression |
| DeepSeek Q2_K | Row tasks | Sufficient N-grid parallelism; no reduction in rounded tail work |
| DeepSeek fixed Q8_0 | M64/M128 | M256 won sequential controls |
| DeepSeek fixed Q8_0 | Width32 or swizzle4 | Regressed all target batches |
| DeepSeek fixed Q8_0 | M512 | Accumulator growth would exceed the VGPR warning budget |
| Representation | Transient BF16 | Slower optimistic floor plus 512 MiB per projection |
| Representation | Persistent BF16 | 19-60 GiB residency and poor model-wide latency trade |

Do not restart wholesale direct-to-VGPR, two-LDS pipelines, GSU, split-K, grouped Stream-K, custom LDS barriers, ordinary K64, broad packed prefetch, or architecture-gated direct-global-to-LDS without new profiler evidence. Dense and grouped passes already found these neutral, slower, or resource-invalid.

Do not use compiler-managed local arrays as prefetch state or keep next-iteration packed fragments alive across current WMMA. Longer VGPR lifetimes lost to the retained bounded decode.

Do not replace fused pairs with two public outputs plus `torch.add`. That changes rounding and triples peak pair output allocation.

## Durable design rules

- Judge complete public latency and checkpoint-weighted latency, not isolated instruction counts.
- Use real nonzero cotangents and authoritative packed weights.
- Run GPU measurements sequentially.
- Preserve device-resident routing and inactive-expert sparsity.
- Match cooperative decode width to actual metadata sharing boundaries.
- Keep decode temporaries dead before long WMMA phases.
- Specialize and shorten state before reducing N; never retain spills.
- Keep pair, single-down, and fixed layouts separate when semantics or measurements differ.
- Keep adjacent N workgroups for one row task contiguous in M-major order.
- Keep decode and barriers unconditional when suppressing only inactive consumer minitiles.
- Use normalized disassembly and source semantics. Raw ELF offsets and instruction ordering are not optimization evidence.
- Require a dense shared-primitive control when grouped and dense kernels share decode or arithmetic helpers.
- Use shape, quant type, token count, and rows/group for dispatch. Do not inspect device routes on the host.
- Treat the 256-VGPR range as a warning boundary even when the compiler reports no spills.

## Validation and packaging

Final validation after the retained tail controls:

- `67 passed, 14 warnings` from `PYTHONPATH=. pytest -q`.
- Ruff and compileall pass.
- `git diff --check` passes.
- The in-tree extension builds successfully.
- All resource-gated kernels pass zero-private, zero-spill, and no-dynamic-stack checks.
- Two independent all-core builds pass `--verify-reproducible`.
- Bundle freshness reports 118 current kernels.
- Qwen 60-point and DeepSeek 27-point final acceptance matrices pass correctness.

Validation commands:

```bash
PYTHONPATH=. pytest -q
ruff check .
python -m compileall -q bench tools torch_ggml_ops tests
python tools/build_mmq_bundle.py --check
python tools/build_mmq_bundle.py --force --jobs "$(nproc)" --verify-reproducible
git diff --check
```

The concrete-wrapper conversion established byte-identical HSACOs for all 118 then-current symbolic-selector controls. Subsequent tail-predication changes are intentional HIP-semantic changes and remain reproducible and resource-gated.

Earlier apparent B1 movements above 1% were checked against rebuilt byte-identical Q4_K, Q5_K, and IQ2_S artifacts. Sequential warmed 25-repeat controls still ranged from `-2.21%` to `+0.61%`, confirming timing variance rather than an ISA regression. Byte-identical timing movement must not reopen a semantic optimization decision.

The current bundle has 32 grouped-backward entries:

- Seven generic singles.
- Seven generic pairs.
- One generic fixed Q8_0 entry.
- Twelve Qwen geometry-specific entries.
- Five DeepSeek geometry-specific entries.

`setup.py build_ext` generates the source-derived bundle and prunes stale artifacts. Local wheels may contain generated HSACOs. Git and source distributions do not.

## Artifact index

### Latest acceptance

```text
/tmp/grouped_mmq_bwd_qwen_tail_predicate_final_acceptance.json
/tmp/grouped_mmq_bwd_ds4_tail_predicate_final_acceptance.json
/tmp/grouped_mmq_bwd_qwen_rowtask_tail_predicate_control_false_25.json
/tmp/grouped_mmq_bwd_qwen_rowtask_tail_predicate_control_true_25.json
/tmp/grouped_mmq_bwd_ds4_iq2xxs_tail_predicate_baseline_25.json
/tmp/grouped_mmq_bwd_ds4_iq2xxs_tail_predicate_candidate_25.json
/tmp/grouped_mmq_bwd_ds4_q2k_tail_predicate_baseline_25.json
/tmp/grouped_mmq_bwd_ds4_q2k_tail_predicate_candidate_25.json
```

### DeepSeek retained and rejected controls

```text
/tmp/grouped_mmq_bwd_ds4_baseline_b1_b4.json
/tmp/grouped_mmq_bwd_ds4_q80_m256_control_25.json
/tmp/grouped_mmq_bwd_ds4_q80_m128_control_25.json
/tmp/grouped_mmq_bwd_ds4_q80_generic_control_25.json
/tmp/grouped_mmq_bwd_ds4_iq2xxs_swizzle4_full.json
/tmp/grouped_mmq_bwd_ds4_iq2xxs_swizzle16_focus.json
/tmp/grouped_mmq_bwd_ds4_q2k_u2_dispatch_control_25.json
/tmp/grouped_mmq_bwd_ds4_q2k_u1_dispatch_control_25.json
/tmp/grouped_mmq_bwd_ds4_q2k_n128_candidate_valid_b16_25.json
/tmp/grouped_mmq_bwd_ds4_q2k_n64_bracket_b16_25.json
```

### Qwen historical and retained controls

```text
/tmp/grouped_mmq_bwd_baseline_full.json
/tmp/grouped_mmq_bwd_final_full.json
/tmp/grouped_mmq_bwd_one_expert.txt
/tmp/grouped_mmq_bwd_step1_q4_matrix.json
/tmp/grouped_mmq_bwd_step2_q3_pair_matrix.json
/tmp/grouped_mmq_bwd_step3_s1.json
/tmp/grouped_mmq_bwd_step4_row_tasks_mmajor.json
/tmp/grouped_mmq_bwd_step5_q5_prefetch.json
/tmp/grouped_mmq_bwd_step6_iq2.json
/tmp/grouped_mmq_bwd_step7_iq2_pair_n64.json
/tmp/grouped_mmq_bwd_step7_q3_pair_n64.json
/tmp/grouped_mmq_bwd_qwen_q5_swizzle8_rowtask_control_25.json
/tmp/grouped_mmq_bwd_qwen_q5_swizzle4_rowtask_control_25.json
/tmp/mmq_bwd_grouped_final_dense_control.json
/tmp/grouped_mmq_bwd_qwen_transient_bf16_floor.json
/tmp/grouped_mmq_bwd_qwen_short_identical_control_a_25.json
/tmp/grouped_mmq_bwd_qwen_short_identical_control_b_25.json
/tmp/grouped_bwd_final_readobj.txt
/tmp/grouped_bwd_final_disasm.txt
```

### Rejected Qwen controls

```text
/tmp/grouped_mmq_bwd_step3_s2.json
/tmp/grouped_mmq_bwd_q5_sparse_s2_25.json
/tmp/grouped_mmq_bwd_step6_iq2_n64_reuse.json
/tmp/grouped_mmq_bwd_step4_row_tasks_nmajor.json
/tmp/grouped_mmq_bwd_step4_persistent1024.json
/tmp/grouped_mmq_bwd_step7_split_tasks.json
/tmp/grouped_mmq_bwd_step7_iq2_width8.json
/tmp/grouped_mmq_bwd_step7_iq2_swizzle0.json
/tmp/grouped_mmq_bwd_step7_iq2_swizzle4.json
```

Related documents:

- `docs/mmq_bwd_optimization.md` for dense backward.
- `docs/grouped_mmq_fwd_optimization.md` for grouped forward.
- `docs/kernel_bundle.md` for source-only bundle generation and packaging.
