# Dense MMQ forward optimization status and plan

## Scope

This document covers dense `torch_ggml_ops::mmq` forward on gfx1151.

Included:
- BF16 activations.
- internal Q8_1 activation quantization.
- packed GGUF Q3_K, Q4_K, Q5_K, Q6_K, IQ2_S, and Q8_0 weights.
- the DeepSeek-V4-Flash persistent ordinary Q8_0 projections.
- BF16 outputs.
- the 160 ordinary Qwen projections and packed Q6_K language-model head.
- the 301 ordinary DeepSeek Q8_0 projections and packed Q8_0 language-model head.
- production batch sizes 1, 4, and 16 at sequence length 2048.

Excluded:
- `grouped_mmq` multiplication and routed-expert scheduling.
- GatedDeltaNet physical-layout permutations.
- LoRA GEMMs and residual accumulation.
- public operator-schema changes.
- changes to `csrc/vendor/llama_cpp/*`.

The grouped path shares the rewritten activation quantizer but now has its own independently optimized multiplication and scheduling implementation, documented in `docs/grouped_mmq_fwd_optimization.md`.

## Current status

Dense forward has a stable Qwen production baseline and complete DeepSeek-V4-Flash dense `Q8_0` correctness and workload coverage. The next optimization project is defined below: first specialize DeepSeek's exact production shapes, then reuse same-input Q8_1 activations, and only then consider representation-level work for the remaining Qwen and DeepSeek margins.

Done:
- replaced the excessive small-workgroup Q8_1 launch with one 512-thread workgroup per real activation row.
- added compile-time row-tile selection and retained `J=64` only for the 64-row Q6_K fallback.
- kept the measured ordinary `I=64, J=128`, 128-thread geometry.
- validated zero private segment for the retained quantizer and dense forward specializations.
- selected 256 rows as the production Qwen LM-head loss chunk at the scheduling layer.
- validated all seven DeepSeek dense `Q8_0` geometry families, including the LM head, against independent GGUF dequantization through the direct packed-weight path.
- added model-family-aware dense-forward benchmark cases without flattening the fixed eight-group output-A projection into dense semantics.

The Qwen source-of-record benchmark remains `/tmp/mmq_fwd_final_full.json`; DeepSeek has correctness coverage but no accepted performance source of record yet. The production Qwen LM-head decision must be evaluated with the complete loss loop: its M=256 forward call is slower than BF16 in isolation, but the 2,048-row packed loss is faster because M=256 sharply reduces call count and uses the optimized backward kernel.

Remaining work is deliberately bounded:
- specialize exact full-tile DeepSeek `Q8_0` bodies and remove the J128 over-computation at LM-head M=32 and M=64.
- reuse Q8_1 activation workspaces across verified same-input projections in both models.
- revisit Qwen's narrow Q3_K/Q5_K margins only with exact-shape or profiler-supported decode/LDS changes.
- treat Qwen Q6_K M=256 and any residual DeepSeek Q8_0 deficit as explicit model-owned representation projects, not another global tile sweep.

Production dispatch and LM-head chunk choices remain unchanged until the acceptance matrix and complete-loss controls in this document pass.

## Hardware and measurement rules

Measurements were taken on:

```text
GPU: Radeon 8060S Graphics
architecture: gfx1151, wave32, 40 CUs
PyTorch: 2.12.0+rocm7.15.0a20260701
HIP: 7.14.60850
```

The performance reference is `torch.mm` using the same BF16 activation and the logical GGUF weight dequantized to BF16. PyTorch normally dispatches these references to hipBLASLt.

The first forward and backward baselines were mistakenly run concurrently and contended for the GPU. They were discarded. All accepted numbers come from sequential runs with no concurrent GPU benchmark or profiler.

On gfx1151, int8 WMMA is approximately as fast as BF16 WMMA. Packed forward wins through lower weight traffic, compact staging, or better geometry rather than a nominal 2x arithmetic advantage.

## Benchmark harness and artifacts

The forward benchmark is:

```bash
source ~/venv_torch/bin/activate
PYTHONPATH=. python bench/benchmark_mmq_fwd.py
```

A focused example is:

```bash
PYTHONPATH=. python bench/benchmark_mmq_fwd.py \
  --cases narrow_q4_k --batches 1,4,16 --warmup 3 --repeats 9
```

Primary artifacts:

```text
Sequential baseline: /tmp/mmq_fwd_baseline_primary_sequential.json
Final full forward:  /tmp/mmq_fwd_final_full.json
```

The `/tmp` paths record measurement provenance and are not repository inputs.

## Production shapes

For ordinary projections, `M = batch * 2048`:

| Batch | M |
| ---: | ---: |
| 1 | 2,048 |
| 4 | 8,192 |
| 16 | 32,768 |

Representative shapes:

| Family | `(N, K)` | Weight types | Model tensors |
| --- | ---: | --- | ---: |
| Query plus query gate | `(8192, 2048)` | Q3_K, Q4_K | 10 |
| Key/value/shared gate/up | `(512, 2048)` | Q3_K, Q4_K, Q5_K | 100 |
| Attention output | `(2048, 4096)` | Q4_K | 10 |
| Shared-expert down | `(2048, 512)` | Q4_K, Q5_K | 40 |

The LM head uses:

```text
N = 248320
K = 2048
weight = Q6_K
production chunk M = 256
```

Comparison chunks are `M = 64, 128, 256`.

## DeepSeek-V4-Flash expansion

Status: correctness implementation and workload coverage are complete; the performance plan is ready but not yet executed. The target remains gfx1151 with sequence length 2,048 and physical batch sizes 1, 4, and 16. Batch coverage is part of the production contract, not a gradient-accumulation substitute.

For full-sequence dense projections:

| Physical batch | M |
| ---: | ---: |
| 1 | 2,048 |
| 4 | 8,192 |
| 16 | 32,768 |

The generic dense `Q8_0` specialization now covers every persistent dense matrix family in DeepSeek-V4-Flash:

| Family | `(N, K)` | GGUF type | Tensors | Forward execution |
| --- | ---: | --- | ---: | --- |
| Attention Q-A | `(1024, 4096)` | Q8_0 | 43 | dense |
| Attention Q-B | `(32768, 1024)` | Q8_0 | 43 | dense |
| Attention KV | `(512, 4096)` | Q8_0 | 43 | dense |
| Attention output B | `(4096, 8192)` | Q8_0 | 43 | dense |
| Shared gate/up | `(2048, 4096)` | Q8_0 | 86 | 43 same-input pairs |
| Shared down | `(4096, 2048)` | Q8_0 | 43 | dense |
| LM head | `(129280, 4096)` | Q8_0 | 1 | packed loss chunks |

The LM-head chunk candidates are `M = 32, 64, 128, 256, 512`. Tune them by complete packed-loss-loop time and peak allocation separately at physical batch sizes 1, 4, and 16; isolated MMQ throughput is not sufficient.

The implemented correctness contract is:
- BF16 input and output.
- Q8_1 dynamic activation quantization.
- direct packed `Q8_0` decode with no logical weight materialization.
- runtime M/N/K coverage through the existing dense specialization.
- independent GGUF-reference tests for all seven ordinary projection families.

The planned optimization contract is:
- one reusable Q8_1 activation workspace for each shared gate/up pair.
- explicit production lookup entries keyed by quant type, `(M, N, K)` geometry, direction, and batch/chunk bucket.
- allocation and event-timed benchmarks for all three physical batch sizes.
- complete packed-loss-loop selection for the LM-head chunk.

The complete new quant inventory also contains routed `IQ2_XXS` gate/up weights and routed `Q2_K` down weights. Those types are not dense-MMQ targets; their exact expert shapes and schedules are specified in `docs/grouped_mmq_fwd_optimization.md`. The frozen eight-group output-A projection is also handled by the grouped plan rather than flattened into dense `(8192, 4096)` semantics.

## Current implementation

The dense forward device bodies are implemented in project-owned `csrc/mmq_core.cuh`. Production entry points are independently compiled by `tools/build_mmq_bundle.py` and selected and launched through `csrc/mmq_bundle.cpp`; `csrc/mmq_hip.cu` retains only operator validation and tensor/workspace ownership.

The ordinary multiplication geometry remains:

```text
I = 64
J = 128
threads = 128, four wave32 waves
K iteration = 256
```

Q6_K calls whose padded row count is 64 use:

```text
I = 64
J = 64
threads = 128
K iteration = 256
```

Q6_K calls above 64 rows retain `J=128`.

## Accepted changes

### One activation-quantization workgroup per real row

The original Q8_1 launch used:

```text
grid = [rows_padded, K / 256]
block = 64 threads
```

At `K=2048`, it created eight workgroups per activation row and also quantized padded rows. At `M=32768`, it launched 262,144 small workgroups and dominated narrow-N calls.

The accepted launch is:

```text
grid = [rows, 1]
block = 512 threads
```

Each workgroup owns one real row. Every thread processes four BF16 values per loop iteration, and the block loops only when `K > 2048`.

The Q8_1 D4 and DS4 layouts, 32-value reductions, rounding, and workspace representation are unchanged. Padded workspace rows are not written because their corresponding MMQ outputs are bounds-masked.

### Compile-time `J` and the Q6_K `J=64` specialization

Dense forward was generalized from fixed `J=128` to compile-time `J` in project-owned code.

The 64-row Q6_K fallback specialization removes 64 padded rows and halves the accumulator and activation-tile footprint. The production 256-row chunk uses `J=128` because a global `J=64` policy regressed ordinary and larger-row workloads.

## Final retained results

The following compares the valid sequential baseline with `/tmp/mmq_fwd_final_full.json`. Ratio means packed-MMQ throughput divided by BF16 throughput.

| Case | M | Baseline ms | Final ms | Speedup | Final ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Query Q3_K | 2,048 | 3.213 | 3.054 | 1.05x | 1.05x |
| Query Q3_K | 8,192 | 13.324 | 12.403 | 1.07x | 1.03x |
| Query Q3_K | 32,768 | 52.589 | 48.828 | 1.08x | 1.03x |
| Narrow Q4_K | 2,048 | 0.235 | 0.214 | 1.10x | 1.48x |
| Narrow Q4_K | 8,192 | 1.990 | 0.899 | 2.21x | 1.04x |
| Narrow Q4_K | 32,768 | 7.636 | 3.590 | 2.13x | 0.99x |
| Attention output Q4_K | 2,048 | 1.811 | 1.410 | 1.28x | 1.26x |
| Attention output Q4_K | 8,192 | 7.638 | 5.717 | 1.34x | 1.23x |
| Attention output Q4_K | 32,768 | 30.624 | 22.986 | 1.33x | 1.20x |
| Shared down Q4_K | 2,048 | 0.256 | 0.257 | 0.99x | 5.59x |
| Shared down Q4_K | 8,192 | 1.053 | 0.985 | 1.07x | 5.35x |
| Shared down Q4_K | 32,768 | 4.150 | 3.939 | 1.05x | 5.26x |
| LM head Q6_K | 64 | 7.413 | 4.202 | 1.76x | 2.05x |
| LM head Q6_K | 128 | 8.001 | 7.988 | 1.00x | 1.56x |
| LM head Q6_K | 256 | 16.097 | 16.127 | 1.00x | 0.81x |

Secondary final ratios:

| Case | M=2,048 | M=8,192 | M=32,768 |
| --- | ---: | ---: | ---: |
| Query Q4_K | 1.13x | 1.12x | 1.10x |
| Narrow Q5_K | 1.21x | 1.00x | 0.95x |
| Narrow Q3_K | 1.38x | 0.97x | 0.92x |
| Shared down Q5_K | 5.28x | 5.15x | 5.13x |

The dominant 70-tensor narrow Q4_K path reaches BF16 parity at large M. The 64-row LM-head fallback reaches about 15.49 logical TFLOP/s and 2.05x BF16 throughput.

## Profiling and generated ISA

### Narrow Q4_K at `M=32768, N=512, K=2048`

`rocprofv3` tracing reported:

| Kernel | Baseline average | Final average | Final resources |
| --- | ---: | ---: | --- |
| Q8_1 quantizer | 5,105.979 us | 886.928 us | 32 allocated VGPRs, no LDS, no private segment |
| Dense MMQ | 2,565.848 us | 2,645.314 us | 256 allocated VGPRs, 38,400-byte LDS, no private segment |

The quantizer is 5.76x faster. Combined traced time fell from about 7.67 ms to 3.53 ms.

The multiplication kernel itself did not improve in this experiment. The end-to-end gain came from eliminating quantization scheduling overhead.

Code-object metadata reports 254 architectural VGPRs for Q4_K `J=128`. rocprofv3 rounds the allocation to 256.

### Q6_K LM head at `M=64`

The `J=64` multiplication kernel averages about 4,190 us and uses:

```text
176 allocated VGPRs
28,928-byte LDS
0-byte private segment
```

The Q8_1 quantizer is about 3.4 us at this row count. The call is almost entirely multiplication time, and removing padded-row multiplication produced the 1.76x speedup.

### LDS vectorization and layout

Generated gfx1151 ISA shows that the main forward LDS path is already substantially vectorized.

Q4_K and Q5_K `J=128` use `ds_load_b128` for their principal LDS reads. Their staged operands use dual-address 32-bit stores such as `ds_store_2addr_b32` and `ds_store_2addr_stride64_b32`.

Q3_K and Q6_K use paired 32-bit or 64-bit LDS operations where their packed layouts permit them. Some scale, metadata, and packed fields remain 32-bit because they are not naturally contiguous per thread.

Forward retains the inherited `GGML_PAD(...)` separation and type-specific SRAM strides used by the selectively vendored MMQ templates. This pass did not perform a controlled forward bank-swizzle sweep, so the layout should not be described as proven optimal.

There is no current evidence that forward LDS layout is a large production bottleneck. The dominant shapes already match or beat BF16, and the observed narrow improvement came from the quantizer rather than multiplication.

## Experiment log

### Accepted

| Experiment | Result |
| --- | --- |
| One 512-thread Q8_1 block per real row | Removed padded-row work and excessive small workgroups. First-order narrow gain |
| Compile-time forward `J` | Enabled bounded row-tile specialization without vendor changes |
| Q6_K `J=64` for padded rows `<=64` | 7.413 to 4.202 ms on the low-memory LM-head fallback |
| Ordinary `I=64`, `J=128`, 128 threads | Best measured general production configuration |

### Rejected or not generalized

| Experiment | Reason |
| --- | --- |
| Global `J=64` | Regressed ordinary and larger-row workloads |
| `I=128` | Regressed the measured shape mix |
| Smaller forward thread/tile combinations | Did not improve the production aggregate |
| A major WMMA representation rewrite | No large remaining production margin and high implementation risk |
| DirectToLds/DirectToVgpr-style rewrite | Packed reconstruction remains in the path. Selected hipBLASLt references also disable these modes |
| gfx1250 WMMA arb-stall programming | The capability is not available on gfx1151 |

## Relevant architecture lessons

TensileLite and shipped hipBLASLt remain useful as records of measured gfx1151 geometry and scheduling. They are not directly reusable generators for packed GGUF MMQ.

Relevant forward lessons:
- four wave32 waves remain a sound workgroup size.
- conventional global-to-VGPR-to-LDS staging is competitive on gfx1151.
- one and two LDS buffers both win in different dense shapes, so dual buffering is not a universal rule.
- wide local reads and aligned LDS layouts are desirable, but must be judged by end-to-end time.
- source-swap and transposed LDS concepts are useful orientation references rather than drop-in code.
- generated dense-GEMM solution databases do not model Q8 activation workspaces or GGUF decode.

FeatherOps reinforces several measurement rules:
- use controlled ablations rather than PC samples alone.
- inspect generated ISA and resource metadata after layout or vectorization changes.
- use real nonzero operands because zero WMMA inputs can mislead.
- keep explicit prefetch state in fixed scalar/vector VGPR values rather than compiler-managed arrays.
- optimize whole-call time, including quantization and workspace allocation.

## Optimization plan

The retained Qwen implementation has already bounded the global launch-shape neighborhood. This plan does not reopen the rejected `I=128`, global `J=64`, 64/256-thread, split-K, persistent-workgroup, decoded-weight-LDS-cache, or speculative prefetch sweeps. It applies later evidence from dense backward and grouped forward/backward only where dense-forward ownership and tile reuse make the hypothesis transferable.

Evidence carried into this plan:

| Log | Transferable result | Boundary |
| --- | --- | --- |
| Dense forward | The 512-thread row quantizer and I64/J128 ordinary body are the production controls; global geometry and speculative prefetch sweeps are closed. | Q6_K M=256 and narrow Q3_K/Q5_K remain measured deficits. |
| Dense backward | Exact shapes, quant-specific packed extraction, and body-specific LDS layouts can matter materially. | Backward ownership, reuse, and swizzle constants do not select a forward body. |
| Grouped forward | Compile-time N/K, separate full/tail bodies, bounded decoder unrolling, and Q8_0 J64 produced real gains. | Fixed/routed scheduling, group-major workspaces, and rejected Q8_0 scale staging do not transfer directly. |
| Grouped backward | Resource cliffs, exact per-body dispatch, and explicit representation ownership are first-class constraints. | Inactive-M suppression, row tasks, and pair accumulation apply to irregular grouped work, not dense full tiles. |
| Kernel bundle | Warm modules, normalized ISA, concrete static wrappers, resource verification, and independent reproducibility are required controls. | Code-object offsets, symbol order, and launcher movement are not arithmetic evidence. |

The priority order is:

| Phase | Scope | Primary hypothesis | Completion condition |
| --- | --- | --- | --- |
| P0 | Rebaseline and decompose | The benchmark must separate quantization, packed arithmetic, allocation, and complete-call costs before choosing a target. | Sequential source-of-record matrices, resources, and representative profiler counters exist for both models. |
| P1 | DeepSeek exact Q8_0 bodies | Exact N/K and full-tile M/N bodies remove runtime bounds and pointer state; J32/J64 remove LM-head row over-computation. | A simple static Q8_0 dispatch beats or matches the generic body across all production shapes with no protected regression. |
| P2 | DeepSeek profile-guided controls | A bounded J64/J128 or K-loop change can lower the current 248-VGPR/38,400-byte-LDS pressure where occupancy or loop overhead is exposed. | Retain only changes with a measured shape-specific gain and a matching ISA/resource explanation. |
| P3 | Same-input activation reuse | Quantizing one BF16 activation once per metadata layout removes duplicate Q8_1 work and allocation without changing arithmetic. | Shared gate/up pairs and other audited same-input families use explicit prepared workspaces with exact output parity. |
| P4 | Remaining Qwen local work | Exact full-tile bodies or profiler-supported Q3_K/Q5_K producer changes may close the narrow-projection margin. | The Qwen matrix improves without reopening broad geometry sweeps or regressing the established LM-head schedule. |
| P5 | Representation projects | Q6_K M=256 and any residual Q8_0 margin require reusable prepared weights, not more local scheduling guesses. | A separately owned representation wins after preparation, memory, forward, and backward reuse are included. |
| P6 | Loss-loop and release acceptance | Single-call gains matter only if production-weighted and complete-loss behavior also improves. | Correctness, resources, 25-repeat controls, full matrices, and loss-loop checks all pass. |

### P0: establish source-of-record baselines

Run GPU jobs sequentially with warmed modules and real nonzero tensors. Record a nine-repeat baseline first; any fresh median movement above 1% requires a sequential 25-repeat baseline/candidate/bracket control before a semantic decision.

The DeepSeek matrix is:
- six ordinary Q8_0 shapes at M=2,048, 8,192, and 32,768.
- LM-head Q8_0 at M=32, 64, 128, 256, and 512.
- report the 43 shared gate/up pairs as 86 model calls while retaining one unique matrix shape.
- report checkpoint-weighted latency separately from the unweighted geometric median.

The Qwen control matrix remains every existing forward case at physical batches 1, 4, and 16, plus Q6_K LM-head M=64, 128, and 256. Do not infer arithmetic movement from raw code-object offsets or bundle symbol order; compare normalized ISA and HIP semantics.

Create explicit timing components for:
- Q8_1 workspace allocation.
- Q8_1 quantization for each required metadata layout.
- packed multiplication with a prepared workspace.
- the complete public `mmq` call.
- BF16 reference multiplication.

Use these baseline commands as the starting controls:

```bash
PYTHONPATH=. python bench/benchmark_mmq_fwd.py \
  --model /home/wd/models/ds4/DeepSeek-V4-Flash-IQ2XXS.gguf \
  --model-family deepseek --batches 1,4,16 \
  --lm-head-chunks 32,64,128,256,512 \
  --warmup 3 --repeats 9 \
  --output /tmp/mmq_fwd_ds4_plan_baseline_9.json

PYTHONPATH=. python bench/benchmark_mmq_fwd.py \
  --model /home/wd/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf \
  --model-family qwen --batches 1,4,16 \
  --lm-head-chunks 64,128,256 \
  --warmup 3 --repeats 9 \
  --output /tmp/mmq_fwd_qwen_plan_control_9.json
```

Before changing a kernel, record normalized disassembly, VGPR/SGPR counts, static and dynamic LDS, private storage, spills, and dynamic stack. Profile one narrow, one wide, and one long-K DeepSeek shape plus Q8_0 LM-head M=32 and M=256. Counters should decide whether P2 investigates occupancy, global/LDS traffic, bank conflicts, or loop/control overhead.

P0 stops when repeated controls are stable enough to distinguish a 1% change. If they are not, increase repetitions and bracket each candidate; do not compensate by widening the candidate set.

P0 completed on 2026-07-25. Source artifacts are `/tmp/mmq_fwd_ds4_plan_baseline_9.json`, `/tmp/mmq_fwd_qwen_plan_control_9.json`, and `/tmp/mmq_fwd_p0_components.txt`. The 33-point DeepSeek report established:
- ordinary Q8_0 sustains about 18.4-23.4 logical TFLOP/s; KV at B16 is the only ordinary point below BF16, at `0.985x`, while Q-B remains close at `1.008-1.045x` for B4/B16.
- LM-head M=32, M=64, and M=128 take `5.395`, `5.630`, and `5.859 ms`; the nearly flat latency confirms that the generic J128 body over-computes the two smaller chunks.
- profiler decomposition attributes only `0.3%` of LM-head M32 and `0.4%` of Q-B B1 to Q8_1 quantization, but `26.3%` of KV B16 and `3.2%` of output-B B1.
- Qwen narrow Q3_K B16 spends `23.8%` in Q8_1 quantization, while Q6_K M256 spends less than `0.1%`; the latter is a packed arithmetic/representation limit.

The public call owns allocation and does not expose a prepared arithmetic entry point, so allocation remains included in whole-call event timing and peak-memory reporting. Separate prepared-workspace arithmetic timing is deferred to P3, where such an internal entry point would be real production code rather than a synthetic benchmark ABI.

### P1: specialize exact DeepSeek Q8_0 production bodies

The generic dense Q8_0 kernel is correct but uses one J128 runtime-M/N/K body for every DeepSeek matrix. Its current code object uses 248 VGPRs, 29 SGPRs, and 38,400 bytes of dynamic LDS, with zero private storage, spills, and dynamic stack. That is a valid baseline but leaves little resource headroom and computes four 32-row minitiles for LM-head M=32 and two for M=64.

Implement a small Q8_0-only specialization family with compile-time K, compile-time row tile, and full-I/full-J body flags. All production N values are multiples of I64, ordinary M values are multiples of J128, and LM-head candidates are multiples of 32, so production wrappers do not require mixed output tails.

The bounded candidates are:

| Production class | Required candidates | Reason |
| --- | --- | --- |
| Ordinary M=2,048/8,192/32,768 | I64/J128 full body; I64/J64 as one control | J128 maximizes weight reuse; J64 may lower VGPR/LDS pressure and expose more workgroups for N=512/1024. |
| LM head M=32 | I64/J32 full body; spill-free I64/J64 bounded fallback | Prefer no row over-computation, but retain the smallest body that passes the resource gate. |
| LM head M=64 | I64/J64 full body | Avoid 2x row-tile over-computation. |
| LM head M=128 | I64/J128 and I64/J64 full bodies | Establish the crossover once rather than assuming it. |
| LM head M=256/512 | I64/J128 full body | Preserve packed-weight reuse unless measurements reject it. |

Do not add I128. Dense Qwen and grouped-forward experiments already show that wider output tiles increase reuse only by accepting occupancy and scheduling costs, and DeepSeek Q8_0 has no evidence that changes that tradeoff.

Key the host lookup only by `Q8_0`, K bucket, and M row-tile/full-body class. N should select a different body only if profiling proves that narrow N=512/1024 needs J64. Do not generate one unique wrapper per tensor name when several shapes share the same semantics.

The first implementation should make only these semantic changes:
- compile out M/N load and store predicates for exact full tiles.
- replace runtime K-derived bounds and offsets with typed compile-time values.
- use affine pointer increments where they shorten live ranges.
- retain the existing Q8_1 metadata layout, Q8_0 decode, WMMA operation, output type, and accumulation order.

P1 acceptance requires exact parity with the current packed implementation where the accumulation order is unchanged, plus the independent BF16/GGUF-reference envelope already enforced by `tests/test_deepseek_mmq.py`. Retain a specialization only if it improves its intended production class without a greater than 1% regression in another class that shares the wrapper.

P1 completed on 2026-07-25. The first I64/J32/K4096 full body was rejected before installation because it used 48 private bytes and spilled 11 VGPRs. The retained M=32 body is bounded I64/J64; M=64 uses full I64/J64, and every other production shape uses an exact-K full I64/J128 body.

Retained resources are:
- generic Q8_0 J128: 248 VGPRs and 29 SGPRs.
- exact Q8_0 J128: 216 VGPRs and 28 SGPRs.
- exact/bounded Q8_0 J64: 132 VGPRs and 28 SGPRs.
- all retained bodies: zero private storage, zero spills, no dynamic stack; LDS is 38,400 bytes for J128 and 28,928 bytes for J64.

The sequential 25-repeat generic/candidate/generic bracket is recorded in `/tmp/mmq_fwd_ds4_p1_generic_before_25.json`, `/tmp/mmq_fwd_ds4_p1_exact_25.json`, `/tmp/mmq_fwd_ds4_p1_generic_after_25.json`, and `/tmp/mmq_fwd_ds4_p1_exact_bracket_25.txt`. Against the generic midpoint:
- all 18 ordinary points improved by `9.18-21.75%`.
- LM-head M=32 and M=64 improved by `47.40%` and `49.47%`.
- LM-head M=128/256/512 improved by `12.02-12.96%`.
- the full 33-point geometric latency improved by `21.62%`.
- checkpoint-weighted latency improved by `14.64-25.36%`, depending on batch and provisional LM chunk.

All affected correctness tests passed (`36 passed`). A clean detached build of the pre-change source produced byte-identical Qwen Q8_1, Q3_K, Q4_K, Q5_K, and Q6_K HSACOs, as well as a byte-identical generic Q8_0 fallback. The existing Qwen cases therefore retain identical device code; `/tmp/mmq_fwd_qwen_p1_control_9.json` is retained only as a runtime-variance control.

### P2: run only profile-supported DeepSeek controls

P2 is conditional on P0/P1 evidence. It is not a generic tuning sweep.

Allowed controls are:
- J64 versus J128 for ordinary Q8_0 shapes, one exact body at a time.
- K-loop unroll factors 1, 2, and at most 4 for K=1,024/2,048/4,096/8,192, only when loop/control instructions are exposed.
- one Q8_0 LDS producer-layout or swizzle comparison when bank-conflict counters identify a problem.
- shorter-lived Q8 scale/decode values when VGPR lifetime, rather than memory latency, is the measured occupancy limiter.

Transfer the grouped evidence conservatively. Fixed-group Q8_0 forward established J64 as the only spill-free useful local body in that ownership model, while grouped backward showed that Q8_0 can profit from a large M tile in a different decode-to-BF16 algorithm. Neither result selects a dense-forward geometry by itself. The dense P1 A/B decides.

Do not repeat grouped Q8_0 scale staging, group-major workspace repacking, or generic swizzle sweeps: they regressed there, and there is no dense-specific hypothesis until counters say otherwise. Inactive-M suppression and row-task scheduling also do not transfer: every dense production tile is active and the regular M/N grid already exposes work directly.

A retained P2 kernel must have zero private storage, zero spills, no dynamic stack, and LDS within the gfx1151 budget. A faster candidate at an occupancy cliff is accepted only after a sequential 25-repeat control across every dispatch class that would use it.

### P3: reuse Q8_1 activations explicitly

Same-input activation reuse is the highest-confidence whole-call opportunity. DeepSeek has 43 Q8_0 shared gate/up pairs with one common D4 Q8_1 layout. Audit the model call graph for attention Q-A/KV and any other same-input families before claiming reuse; tensor shapes alone are not proof that activations are identical.

Implement either an internal prepared-activation entry point or a pair/multi-projection entry point. It must:
- quantize once per required Q8_1 metadata layout.
- launch the existing statically selected arithmetic body for each weight.
- own workspace lifetime explicitly for the current stream.
- preserve ordinary `mmq` as the fallback and preserve the public operator schema.
- avoid a hidden pointer cache.

For Qwen, first inventory actual same-input projection groups. Q4_K/Q5_K require the scale-plus-sum layout; Q3_K/Q6_K/IQ2_S require the scale-only layout. A layer that consumes both classes may prepare two workspaces, but never more than one per layout.

Benchmark single calls, pairs, and the checkpoint-weighted layer schedule. Include allocation and quantization in the candidate timing. Acceptance requires bitwise equality with two independent ordinary calls, unchanged autograd behavior, no extra persistent weight memory, and a positive whole-layer gain at physical batches 1, 4, and 16. If quantization plus allocation is below 1% of the paired calls, stop this phase rather than complicating ownership.

### P4: bounded remaining Qwen work

The old global fused-kernel sweep remains closed. The only local Qwen experiment reopened by later logs is exact production specialization, because grouped forward repeatedly benefited when runtime bounds and K-derived state became compile-time facts.

Start with full-I/full-J, compile-time-K bodies for:
- narrow Q3_K `(N, K) = (512, 2048)` at M=32,768, currently about 8% behind BF16.
- narrow Q5_K `(512, 2048)` at M=32,768, currently about 5% behind BF16.
- Q6_K LM head `(248320, 2048)` at M=256 as a protected control, not as a new geometry sweep.

If exact specialization does not move normalized ISA or latency, stop it. For Q3_K/Q5_K, one quant-specific packed extraction or LDS producer-layout control is allowed only after profiling shows exposed decode instructions or bank conflicts. Dense-backward swizzles are hypotheses, not reusable constants: forward stores packed tiles and has different consumers and bank mappings.

Do not retry global J64, I128, split-K, persistent scheduling, decoded-weight LDS caching, or speculative deep prefetch. Do not treat the approximately 3.2% bundle-only movements in attention-query Q3_K or attention-output Q4_K as optimization evidence; those controls had no HIP-semantic change.

Qwen acceptance is the complete existing forward matrix, not only the two narrow points. Preserve Q6_K M=64/M=128 fallbacks and the current M=256 production loss schedule.

### P5: representation-level projects

Enter P5 only after exact bodies and shared activation reuse have been measured. Representation ownership must be explicit at model preparation time, with documented memory cost, lifetime, invalidation, device placement, and forward/backward reuse. Do not add hidden persistent BF16 shadows.

For Qwen Q6_K M=256, the leading candidate is a compact integer-plus-scale representation that is lossless relative to the existing packed quantized values and laid out for the WMMA consumer. It must avoid repeated 6-bit reconstruction while remaining materially smaller than BF16 and usable by both forward and backward. Measure model-load preparation separately and report packed bytes, prepared bytes, and peak bytes.

For Qwen Q3_K/Q5_K, add a prepared representation only if P4 proves packed decode is the remaining limit and the checkpoint-weighted reuse repays its memory. Do not generalize from one narrow tensor type to every quant family.

DeepSeek Q8_0 is already an integer-plus-scale format. If P1/P2 still lose materially, compare two floors before designing storage:
- a one-time tile-major Q8_0 repack that preserves values and scales.
- an optimistic BF16 dense floor with decode and allocation accounted for separately.

A tile-major Q8_0 representation is preferred over BF16 if it closes the layout cost while remaining close to packed size. Any BF16 or decoded dense stage must include preparation in first-use latency and peak allocation, and must beat direct packed execution over its real reuse count. Keep routed and fixed-group DeepSeek weights on their existing packed paths; dense evidence does not authorize changing grouped representations.

### P6: complete-loss and release acceptance

Forward timing alone may propose an LM-head chunk, but it cannot select production scheduling. Qwen must rerun the complete 2,048-row packed-loss loop for M=64, 128, and 256 after any retained forward or representation change. Its current control is 229.958 ms at M=256 versus 312.690 ms at M=64.

DeepSeek must measure M=32, 64, 128, 256, and 512 at physical batches 1, 4, and 16, including peak allocation. Final chunk selection is deferred until the Q8_0 dense-backward path and complete packed-loss loop exist; isolated forward results should be recorded but must not set production dispatch.

Every retained phase must pass:
- independent GGUF-reference correctness for every affected quant/shape family.
- exact current-output comparison where semantics and accumulation order are unchanged.
- the complete Qwen and DeepSeek forward benchmark matrices with three warmups and at least nine measured repeats.
- sequential 25-repeat A/B controls for any median movement above 1%.
- checkpoint-weighted latency and unweighted per-point summaries.
- zero private storage, zero spills, no dynamic stack, and acceptable LDS for every new resource-gated kernel.
- `python tools/build_mmq_bundle.py --force --jobs "$(nproc)"`, bundle freshness, symbol/resource checks, and independent `--verify-reproducible` validation.
- the complete project test suite, Ruff, compileall, and `git diff --check`.

Static dispatch may depend only on quant type, M row/chunk bucket, N/K shape class, or an explicitly prepared activation/weight representation. No online autotuning or pointer-identity cache is permitted. A candidate that fails its phase gate is removed before the next phase so the final dispatch table contains only independently justified bodies.

## Correctness and compatibility

The current complete project suite passes:

```text
pytest -q tests/
83 passed
```

The historical downstream integration suite also passed 9 tests, including the production 256-row packed-loss schedule. Current project validation is self-contained.

Forward normalized RMSE remains within the existing Q8_1 envelope:
- approximately 0.6% for Q3_K and Q6_K.
- approximately 1.1-2.0% for Q4_K and Q5_K.

The resource-gated quantizer specializations have zero-byte private segments and no spills. Dense-forward compatibility entries retain their historical compiler resource profiles; in particular, `Q2_K` J128 uses 112 private bytes and 27 VGPR spills and is deliberately not resource-gated. No file under `csrc/vendor/llama_cpp/*` was modified by the bundle conversion.

## Architecture-specific bundle conversion

Dense forward and its activation quantizers moved from the extension fatbinary into the generalized gfx1151 package documented in `docs/kernel_bundle.md`. The conversion retains D4/DS4/D2S6 workspace selection, J128 ordinary geometry, and the J64 `Q6_K` small-row threshold. It removes direct launches and device entry points from `csrc/mmq_hip.cu`.

The initial nine-repeat before/after comparison measured `-0.23%` geometric dense-forward latency. Sequential embedded/bundle/embedded 25-repeat controls measured the bundle at `+0.87%` geometrically, `+0.76%` by median point, and `+1.05%` by estimated model latency against the bracket midpoint. The embedded controls themselves drifted by `+2.18%`.

Notable midpoint comparisons are about `+3.2%` for attention-query `Q3_K` batch 4 and attention-output `Q4_K` batch 4, and `-1.5%` for narrow `Q5_K` batch 1. These are code-object layout effects, not retained tile or dispatch changes. Source artifacts are `/tmp/mmq_fwd_pre_bundle.json`, `/tmp/mmq_fwd_post_bundle.json`, and the three `/tmp/mmq_fwd_*_control_25.json` files.

## Tool notes

- PC sampling heavily perturbs short kernels, so use it qualitatively.
- `roc-obj-ls` is broken in the active environment because of a `rocm_sdk_core._cli` import error.
- Code-object inspection uses `.hip_fatbin`, `clang-offload-bundler`, `llvm-readobj`, `llvm-nm`, and `llvm-objdump`.
