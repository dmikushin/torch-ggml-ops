# Grouped MMQ forward optimization

## Scope

This document covers routed and fixed-group grouped MMQ forward on gfx1151.

The production operators are:
- `grouped_mmq_pair` for routed gate and up.
- `grouped_mmq` for routed down.
- `fixed_grouped_mmq` for DeepSeek output A.

Dense MMQ forward and backward are documented separately in `docs/mmq_fwd_optimization.md` and `docs/mmq_bwd_optimization.md`.

The grouped optimization pass is complete for the current Qwen and DeepSeek-V4-Flash packed representations. The original spill-heavy baselines and all experiment logs remain below for provenance. Qwen's source of record remains the spill-free G11 dispatch; the accepted DeepSeek artifact is `/tmp/grouped_mmq_fwd_ds4_final_isolated_full.json`, with its same-build Qwen control at `/tmp/grouped_mmq_fwd_qwen_post_ds4_isolated_final.json`.

## Qwen current status

Done:
- specialized production gate/up and down shapes at compile time with `I=64`, `J=64`, and 128 threads.
- removed all production private segments and register spills.
- split exact full-row and bounded tail bodies.
- retained a two-block fixed-K down schedule and pointer-increment gate/up traversal.
- added atomics-free device row-task descriptors for large gate/up groups and reused them across paired projections.
- retained serial row ownership for down after descriptors and a complete decoded-weight LDS cache regressed.
- added contiguous bounded down-tail activation loads.
- validated all 60 production benchmark points exactly against dense MMQ.

Final outcome:
- packed MMQ wins 54 of 60 individual points against BF16 AITER.
- every gate/up point and every Q4_K/Q5_K down point wins.
- checkpoint-weighted packed grouped projections are 1.53-2.82x faster than AITER across the measured batch/distribution matrix.
- the only remaining individual losses are nonuniform IQ2_S down at batch 1 and batch 4.

Remaining work for the current Qwen workload is representation-level: compact lossless IQ2_S decode caching, cross-call decoded-weight reuse, or a transient project-owned decoded dense stage. Its local tile, scheduler, K-loop, bounds, cache, and synchronization neighborhoods are closed.

## Qwen production contract

The public activation and output dtype is BF16.

Activations are quantized internally to Q8_1. Packed GGUF weights remain the authoritative representation.

`expert_indices` and `expert_offsets` remain device-resident. The optimized path must not inspect group sizes through `.item()`, a device-to-host copy, a CPU descriptor, or an implicit synchronization.

The metadata ABI is:
- `expert_indices`: contiguous CUDA `torch.int64`, shape `[G]`.
- `expert_offsets`: contiguous CUDA `torch.int32`, shape `[G]`.
- `expert_offsets[-1] = R`.
- `G <= 256`.

The sequence length is 2,048 and top-k is 8. The routed row count is therefore:

| Physical batch | Routed rows |
|---:|---:|
| 1 | 16,384 |
| 4 | 65,536 |
| 16 | 262,144 |

Batch 1 commonly has about 150-256 active experts per layer. The observed mean was about 198.5 active experts.

One representative batch-1 layer had 192 active experts and group sizes around 60-106 rows. Batch 4 and batch 16 usually activate all 256 experts, although the sizes remain skewed.

## Qwen production checkpoint matrix

The benchmark uses real tensors from `Qwen3.6-35B-A3B-APEX-I-Mini.gguf`.

| Case | Logical expert shape | GGUF type | Layers | Operator |
|---|---:|---|---:|---|
| Gate/up outer | `512 x 2048` | Q3_K | 20 | `grouped_mmq_pair` |
| Gate/up middle | `512 x 2048` | IQ2_S | 20 | `grouped_mmq_pair` |
| Down outer edge | `2048 x 512` | Q5_K | 2 | `grouped_mmq` |
| Down outer main | `2048 x 512` | Q4_K | 18 | `grouped_mmq` |
| Down middle | `2048 x 512` | IQ2_S | 20 | `grouped_mmq` |

Gate and up use one shared Q8_1 activation workspace. The two packed projections still execute as two grouped multiplication launches.

## DeepSeek-V4-Flash expansion and optimization plan

### Implementation status

Correctness support is implemented for two routed expert formats and one semantically distinct fixed-group projection. Performance tuning has not started. The production checkpoint matrix is:

| Benchmark case | Groups/experts | Logical weight per group/expert `(N, K)` | Physical packed shape | GGUF type | Tensors | Operator |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| `ds4_output_a_q8_0` | 8 fixed groups | `(1024, 4096)` | raw `(8192, 4352)`, operator view `(8, 1024, 4352)` | Q8_0 | 43 | `fixed_grouped_mmq` |
| `ds4_gate_up_iq2_xxs` | 256 experts | `(2048, 4096)` | `(256, 2048, 1056)` each | IQ2_XXS | 43 pairs | `grouped_mmq_pair` |
| `ds4_down_q2_k` | 256 experts | `(4096, 2048)` | `(256, 4096, 672)` | Q2_K | 43 | `grouped_mmq` |

DeepSeek uses top-six routing. The exact batch targets are:

| Physical batch | Token rows M | Routed rows R | Mean rows/expert if all 256 are active |
| ---: | ---: | ---: | ---: |
| 1 | 2,048 | 12,288 | 48 |
| 4 | 8,192 | 49,152 | 192 |
| 16 | 32,768 | 196,608 | 768 |

The fixed output-A operator consumes logical input `[..., 8, 4096]` and produces `[..., 8, N]`, where `N` is carried by the packed logical view. Production uses `N=1024`. This mirrors llama.cpp reshaping the raw logical `(8192, 4096)` weight to eight `(1024, 4096)` matrices before MMQ. It uses token rows `M`, not routed rows `R`; flattening it into an ordinary `4096 -> 8192` projection is invalid.

The implemented forward contract is:
- paired `IQ2_XXS` gate/up execution with one shared Q8_1 activation workspace and independent packed weights and outputs.
- direct packed `Q2_K` down execution without selected logical expert matrices.
- one Q8_1 activation workspace shared by all eight fixed output-A groups.
- device-resident route metadata with no `.item()`, CPU descriptor, or hidden synchronization.
- forward-only dispatch for the three new formats until their independent backward decoders exist.
- Transformers GGUF dequantization as the independent numerical reference.

Focused real-checkpoint validation is bitwise exact against separate dense packed MMQ calls. Observed normalized RMSE against independent BF16 references is approximately `0.0061` for `Q8_0` and `IQ2_XXS`, and `0.0121` for `Q2_K`. These are correctness checks, not accepted production performance baselines.

### Benchmark matrix

`bench/benchmark_grouped_mmq_fwd.py` is model-family-aware. Auto detection keeps the Qwen checkpoint on its original five cases and selects the three cases above when `blk.0.attn_output_a.weight` is present. An explicit `--model-family` override is available. Routed top-k defaults to eight for Qwen and six for DeepSeek; `--top-k` is an explicit override.

The complete DeepSeek matrix has 27 points:
- fixed output A: three physical batches, one fixed distribution.
- routed gate/up: three physical batches by four route distributions, 12 points.
- routed down: three physical batches by four route distributions, 12 points.
- the routed pair contributes one complete public-operator timing per point while accounting for two logical projections.

Routed cases retain BF16 AITER GMM with the project-owned heuristic as their timed reference. Fixed output A uses BF16 `torch.bmm` over eight strided groups and includes conversion back to the public token-major contiguous output layout. Setup-only dequantization is excluded from both references.

Every point records:
- complete packed public-operator latency, including Q8_1 quantization and workspace allocation.
- logical throughput, allocation growth, physical packed shape, quant type, and group summary.
- bitwise comparison with separate dense packed MMQ calls.
- error against independently dequantized BF16 AITER or BF16 batched GEMM.
- checkpoint-weighted forward and recomputation estimates using 43 layers.

Baseline commands are:

```bash
python bench/benchmark_grouped_mmq_fwd.py \
  --model ~/models/ds4/DeepSeek-V4-Flash-IQ2XXS.gguf \
  --model-family deepseek \
  --warmup 3 \
  --repeats 9 \
  --correctness-rows 256 \
  --output /tmp/grouped_mmq_fwd_ds4_baseline_full.json

python bench/benchmark_grouped_mmq_fwd.py \
  --model ~/models/qwen3.6/Qwen3.6-35B-A3B-APEX-I-Mini.gguf \
  --model-family qwen \
  --warmup 3 \
  --repeats 9 \
  --correctness-rows 256 \
  --output /tmp/grouped_mmq_fwd_qwen_pre_ds4_control.json
```

The four deterministic route distributions remain useful for stable comparisons. Before final dispatch selection, add captured hash-router and learned-router distributions from real DeepSeek runs as an additional route set. Captures must preserve inactive experts and exact offsets while remaining device-resident in the timed path.

### Non-regression contract

The existing Qwen dispatch is a production control, not a starting point to rewrite. `/tmp/grouped_mmq_fwd_final_full_v2.json` remains its historical source of record. Before changing kernel source, produce the fresh Qwen control above on the same build and machine used for the DeepSeek baseline.

A retained DeepSeek optimization must satisfy all of the following:
- all 60 Qwen case/batch/distribution points remain bitwise equal to dense packed MMQ.
- all 27 DeepSeek points remain bitwise equal to dense packed MMQ and inside the established independent-reference error envelope.
- no Qwen point has a repeatable latency regression. Treat a median movement above 1% as requiring a 25-repeat sequential A/B control; reject a repeatable regression above measurement noise even if the checkpoint-weighted total improves.
- the Qwen checkpoint-weighted estimate does not regress.
- existing Qwen arithmetic kernels retain zero private segment, zero spills, and no dynamic stack.
- new production arithmetic kernels also require zero private segment, zero spills, and no dynamic stack.
- changes to the shared quantizer or MMQ core run dense forward controls in addition to both grouped matrices.
- no concurrent benchmark or profiler run is accepted.

Improvements to existing Qwen cases are welcome, but they must pass the same complete matrix. Do not replace an exact Qwen branch with a broader heuristic based only on a DeepSeek win.

### Dispatch policy

Keep dispatch static, small, and explainable. It may depend on:
- quant type.
- operator kind: fixed, routed pair, or routed single.
- exact `(N, K)` production geometry.
- a coarse host-visible row bucket such as mean rows per active group.

It must not depend on a route distribution name, environment variable, online autotuning, host reads of device offsets, or a large per-shape table. The intended order is:

```text
1. Existing exact Qwen production branches, unchanged unless independently improved.
2. Exact DeepSeek Q8_0 fixed-group branch.
3. Exact DeepSeek IQ2_XXS (2048, 4096) pair branch.
4. Exact DeepSeek Q2_K (4096, 2048) single branch.
5. General bounds-safe grouped MMQ fallback for every other supported shape.
```

Start with at most two row buckets per routed DeepSeek family: small groups and large groups. Add a third bucket only when a broad route matrix demonstrates a repeatable win. An average-row threshold is only a launch hint; every selected kernel must still handle skew, inactive experts, and tails correctly.

### Phase D0: lock baselines and diagnose

1. Run the fresh Qwen 60-point control and DeepSeek 27-point baseline sequentially.
2. Trace one representative point for each new family at batches 1 and 16. Separate quantization, task setup, and multiplication time.
3. Extract code-object metadata and disassembly for `Q8_0`, `IQ2_XXS`, and `Q2_K`. Record VGPRs, SGPRs, LDS, private bytes, spills, and dynamic stack.
4. Record one-expert controls at representative group sizes 48, 192, and 768 for routed cases. Compare grouped packed execution with the corresponding sequence of dense packed calls.
5. Profile counters only after decomposition. Prior logs show that spill removal and address/control cleanup can dominate apparent cache or LDS hypotheses.

The current routed DeepSeek paths use the general `J=128` kernel because their shapes do not match the Qwen exact branches. The first question is therefore whether generic shape state and `J=128` create spills or excessive full/tail control, not whether a new persistent scheduler is needed.

#### D0 baseline checkpoint: complete

The same-session baseline artifacts are:
- DeepSeek 27-point baseline: `/tmp/grouped_mmq_fwd_ds4_baseline_full.json`.
- Qwen 60-point pre-tuning control: `/tmp/grouped_mmq_fwd_qwen_pre_ds4_control.json`.
- source commit: `aa3ebd4`.

DeepSeek baseline summary:

| Family | Batch 1 packed/reference | Batch 4 packed/reference | Batch 16 packed/reference | Main observation |
| --- | ---: | ---: | ---: | --- |
| Fixed Q8_0 | `10.990/8.162 ms` | `44.336/32.874 ms` | `175.484/129.540 ms` | stable `0.74x`; throughput-bound rather than launch-bound |
| IQ2_XXS pair | `66.104-77.549/56.055-73.039 ms` | `135.713-148.871/134.315-138.103 ms` | `335.363-400.420/654.188-735.394 ms` | small groups lag AITER; large uniform groups already reach `1.95x` |
| Q2_K down | `62.426-76.312/31.472-40.433 ms` | `155.004-162.014/81.587-82.465 ms` | `421.356-482.628/434.581-477.845 ms` | roughly `0.5x` at batches 1/4; approaches parity only at batch 16 |

Every DeepSeek point remained bitwise exact against dense packed MMQ. Independent-reference NRMSE stayed near `0.0061` for Q8_0/IQ2_XXS and `0.0107-0.0114` for Q2_K.

The fresh Qwen control is the acceptance baseline for this tuning session. Relative to the older July G11 artifact, unchanged Qwen code measured a median `4.9%` slower with a `-0.3%` to `+17.1%` point range. That historical movement is too large for per-point acceptance; all retained changes therefore use sequential fresh pre/post controls, while `/tmp/grouped_mmq_fwd_final_full_v2.json` remains qualitative history.

D0 code-object and operator-only trace results:

| Kernel | J | VGPR | SGPR | Private bytes | VGPR spills | Batch-1 quantize ms | Batch-1 arithmetic ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed Q8_0 production full-N | 64 | 213 | 48 | 0 | 0 | 0.964 | 10.052 |
| IQ2_XXS pair, per projection | 128 | 256 | 72 | 360 | 117 | 0.723 total | 38.279 each |
| Q2_K down | 128 | 256 | 61 | 576 | 230 | 0.365 | 75.731 |

All three quantizers are spill-free. Quantization is only about 9% of fixed Q8_0, 1% of the IQ2_XXS pair, and 0.5% of Q2_K at batch 1. The routed baseline bottleneck is therefore the spilled generic arithmetic kernel. D1-D3 start with exact `J=64` specializations; quantizer restructuring is deferred unless the arithmetic fixes expose it as material.

### Phase D1: fixed-group Q8_0 output A

Production geometry is eight always-active groups, `N=1024`, `K=4096`, and 16 MMQ K iterations. Start from the current llama.cpp-style Q8_1/Q8_0 implementation and compare it with the relevant DwarfStar grouped-Q8 implementation on this machine; do not assume either source is faster.

Measure a bounded candidate set:
- compile-time `N=1024`, `K=4096`, group count eight, and fixed K trip count.
- `I=64, J=64` as the first specialized control.
- `I=64, J=128` only if code-object resources remain spill-free.
- `I=128` only after profiling shows insufficient output reuse or launch parallelism; G9 already showed that wider output tiles can lose despite fewer VGPRs.
- current Q8_1 prequantization versus a direct BF16/Q8_0 DwarfStar-style path, judged by complete public-operator latency and workspace allocation.

Keep group as an explicit grid dimension and preserve token-major contiguous output. Avoid a host loop over eight public MMQ calls. Output-A has no routing imbalance, so route descriptors and persistent expert traversal are non-priorities. Its likely first-order choices are activation quantization cost at `8*M` rows, fixed K traversal, and Q8_0 weight/scale staging.

#### D1.1 J=128 fixed token tile: rejected

`J=128` increased the fixed production full-N kernel from 213 to 256 VGPRs, introduced 88 private bytes and 21 reported VGPR spills, and regressed complete latency at every physical batch: `10.990 -> 15.153 ms`, `44.336 -> 60.563 ms`, and `175.484 -> 242.127 ms`. The approximately 38% loss shows that halving Q8_0 weight reloads does not repay accumulator pressure. Spill-free `J=64` is restored.

#### D1.2 DwarfStar grouped-Q8 schedule: rejected for this contract

DwarfStar's retained grouped output-A kernel stages F32 activations and computes directly against dequantized Q8_0 values with per-wave DP4A/scalar accumulation. It does not apply this operator's Q8_1 activation quantization, so it cannot preserve bitwise equality with dense packed MMQ. Its prequantized Q8 path assigns one output row/token to a wave and provides no 64-token by 64-output weight reuse; the current spill-free WMMA kernel is structurally stronger for physical batches 1/4/16. A direct port is therefore rejected on semantics and reuse before source integration. The comparison can be reopened only if quantization becomes dominant; D0 measured it at 9% of fixed batch-1 time.

#### D1.3 rolled Q8_0 dot loop: rejected

Keeping the Q8_0/IQ2_XXS `k01` loop rolled did not clean up J96 IQ2_XXS or improve the production fixed kernel's resource counts. Fixed latency also moved from `10.990/44.336/175.484 ms` to `11.066/44.595/177.793 ms`. The original compiler schedule is retained.

#### D1.4 J=32 fixed token tile: rejected

J32 unexpectedly reaches 256 VGPRs, 344 private bytes, and 85 reported spills despite its smaller accumulator. It fails the production resource gate before timing. The tested J32/J64/J128 family therefore retains spill-free J64.

### Phase D2: routed IQ2_XXS gate/up

Production geometry is a pair of `N=2048`, `K=4096` experts with 16 packed blocks per row. Mean rows per expert are 48, 192, and 768.

Start with:
- compile-time `N=2048`, `K=4096`, and fixed 16-block pointer traversal.
- `I=64, J=64` with exact full-row and one bounded tail body.
- a `J=32` small-group control only for batch-1 nonuniform routes, where mean group size is 48.
- `J=128` only as a spill-free measured control; prior Qwen G3 materially regressed this geometry.
- serial expert ownership as the initial scheduler because 32 output tiles per active expert already expose much more N parallelism than Qwen gate/up.
- row-task descriptors only for large groups if traces show serial row imbalance. Reuse one descriptor build across gate and up.

`IQ2_XXS` optimization should focus on the natural decode-sharing unit: grid lookup, sign unpacking, scale formation, and aligned packed loads. Keep project-specific cooperative decode helpers outside vendored source. Do not fuse the two forward arithmetic outputs unless a resource analysis shows two accumulator sets remain spill-free; unlike backward, forward must produce two separate tensors and cannot share one accumulator.

#### D2.1 exact J=64 specialization: retained candidate

Specializing production `(N,K)=(2048,4096)` with `J=64` reduced the arithmetic kernel from 256 to 229 VGPRs, removed all 360 private bytes and all 117 reported VGPR spills, and preserved zero dynamic stack. Batch-1 uniform complete pair latency improved from `77.549 ms` to `33.581 ms`, a `2.31x` packed-path improvement. Correctness remained bitwise exact.

The complete 12-point route matrix has a `1.52x` geometric-mean speedup. Batches 1 and 4 improve by `29.3-56.7%`; batch-16 skewed/sparse/boundary improve by `8.9-9.4%`. Batch-16 uniform regresses `2.8%`, from `335.363 ms` to `344.635 ms`, because perfectly full 768-row groups make the doubled J=64 weight-tile traversal visible.

#### D2.2 J=96 large-row specialization: rejected

`J=96` uses 256 VGPRs, 76 private bytes, and 18 reported VGPR spills. It is `58-62%` slower than J64 at batch 1 and loses on batch-4 skew. It improves batch-16 uniform strongly but only improves the other batch-16 distributions by `0.3-1.7%`. Rolling the shared Q8_0/IQ2_XXS dot loop did not remove any J96 spill state. J96 is rejected by the production resource gate.

#### D2.3 J=80 large-row specialization: retained

`J=80` uses 253 VGPRs, 51 SGPRs, zero private bytes, zero spills, and no dynamic stack. At batch 16 it improves every distribution over J64: uniform `344.635 -> 333.184 ms`, skewed `358.908 -> 338.649 ms`, sparse `360.531 -> 342.558 ms`, and boundary `363.622 -> 344.836 ms`. At batch 4 it is consistently `1.2-2.2%` slower than J64. The final static heuristic therefore uses J80 only when `rows >= 512 * num_groups`, selecting batch 16 from coarse host-visible geometry without reading device offsets; batch 1/4 retain J64.

#### D2.4 J=32 small-group specialization: rejected

Exact J32 reaches 256 VGPRs, 156 private bytes, and 48 reported spills. It fails the resource gate before batch-1 timing. J64 remains the smallest retained IQ2_XXS tile.

### Phase D3: routed Q2_K down

Production geometry is `N=4096`, `K=2048` with eight packed blocks per row and 64 output tiles per active expert. Its activation metadata uses the distinct Q8_1 `D2S6` layout.

Start with:
- compile-time `N=4096`, `K=2048`, and fixed eight-block pointer traversal.
- `I=64, J=64`, exact full rows, and one contiguous bounded tail.
- serial expert row ownership as the initial scheduler. The 64 N tiles per expert make the Qwen G8 down result directly relevant: extra row descriptors are unlikely to repay setup or locality loss.
- `J=32` only for batch-1 tail utilization and `J=128` only as a resource-clean control.
- separate measurement of D2S6 activation quantization and packed multiplication before changing the shared quantizer.

Tune Q2_K scale/min reconstruction and LDS placement independently of IQ2_XXS. A common schedule may be selected only if both complete matrices support it; a common decoder abstraction is not a performance goal.

#### D3.1 exact J=64 specialization: rejected

The direct `J=64`, `(N,K)=(4096,2048)` specialization increased private storage from 576 to 2,784 bytes and reported VGPR spills from 230 to 1,188. Batch-1 uniform latency regressed from `76.312 ms` to `104.589 ms` (`37.1%`). The branch was removed. Q2_K cannot use the IQ2_XXS tile rule; its decode and 64 output tiles require a different resource reduction.

#### D3.2 J=64 with runtime shape: rejected

Leaving N and K-block count runtime-valued did not isolate the spill problem. The resulting J=64 kernel used 3,068 private bytes and 1,132 VGPR spills, and batch-1 uniform regressed `9.9%`, from `76.312 ms` to `83.862 ms`. This branch was also removed.

#### D3.3 J=32 runtime-shape specialization: retained candidate

`J=32` reduced the generic Q2_K kernel from 576 to 500 private bytes and from 230 to 135 reported VGPR spills. Despite fourfold weight-tile traversal relative to J=128, batch-1 uniform improved `76.312 -> 37.029 ms` (`2.06x`).

The complete route matrix improves every baseline point: `48-51%` at batch 1, `33-40%` at batch 4, and `8-17%` at batch 16. Packed Q2_K now beats AITER at 7 of 12 points. The branch remains bounds-safe and bitwise exact, but 500 private bytes still fail the final resource target. Further D3 work keeps J=32 and targets Q2 scale/min correction lifetime or a distinct decoder body.

#### D3.4 split full/tail launches: rejected

Splitting J32 into separate compiler entry points reduced the full-body kernel to 72 private bytes/17 spills, while the bounded tail remained at 460 bytes/114 spills. The extra launch regressed batch 1 by `4.9-5.6%`, was neutral to 1% slower at batch 4, and improved batch 16 by only `0.5-1.1%`. This is insufficient to retain the split. It does establish that tail bounds, not the full Q2 arithmetic body, account for most remaining compiler spill state.

#### D3.5 replicated tail activation lanes: rejected

For Q2_K tails only, clamping invalid lanes to the last valid activation reduced the tail entry point to 168 private bytes/41 spills without changing valid output fragments. Runtime nevertheless regressed further: batch-1 uniform/boundary reached `41.509/41.710 ms`, boundary batch 4 reached `112.189 ms`, and batch 16 did not improve. Extra clamp/address work dominates the lower scratch count. Replication and split launches were both removed; zero-filled bounded loads in the single J32 kernel remain faster.

#### D3.6 exact J32 production geometry: retained

Adding compile-time `N=4096` and eight K blocks to J32 reduced private storage from 500 to 404 bytes and reported spills from 135 to 109. It improved all six uniform/boundary probes by `5.6-7.6%`, then improved every point in the complete matrix. Candidate ranges were `30.897-35.053 ms` at batch 1, `92.034-101.358 ms` at batch 4, and `366.258-376.947 ms` at batch 16.

#### D3.7 rolled Q2_K scale loop: retained

The gfx1151 compiler fully unrolled the eight Q2_K `k01` scale/min phases, generating a 34 KiB kernel with 404 private bytes and 109 spills even at J32. A generator-owned `#pragma unroll 1` on that Q2_K AMD WMMA loop reduces the exact kernel to 14 KiB, 122 VGPRs, 30 SGPRs, zero private bytes, zero spills, and no dynamic stack. It improves all 12 route points again: `24.052-26.861 ms` at batch 1, `79.920-87.187 ms` at batch 4, and `324.683-335.600 ms` at batch 16. Packed beats AITER at nine points, is within 6% at the other three batch-4 distributions, and remains bitwise exact. This is the first Q2_K candidate satisfying the production arithmetic resource gate.

#### D3.8 partial Q2_K scale-loop unroll: factor 4 retained

A fresh same-session `unroll 1` control at `/tmp/grouped_mmq_fwd_ds4_q2_unroll1_control.json` measured `23.954-26.719 ms`, `78.069-85.444 ms`, and `319.110-333.089 ms` for physical batches 1, 4, and 16. Explicit `unroll 2` remained spill-free at 158 VGPRs and 28 SGPRs, with a roughly 19 KiB arithmetic symbol. Its artifact `/tmp/grouped_mmq_fwd_ds4_q2_unroll2.json` is bitwise exact and improves the 12-point geometric mean by `1.28%`, but is rejected because factor 4 is uniformly faster.

Explicit `unroll 4` uses 208 VGPRs and 38 SGPRs, has a roughly 24 KiB arithmetic symbol, and retains zero private bytes, zero spills, and no dynamic stack. `/tmp/grouped_mmq_fwd_ds4_q2_unroll4.json` is bitwise exact and improves every route: `17.5-19.1%` at batch 1 and `10.7-13.9%` at batches 4/16. Its 12-point geometric speedup is `1.145x` over factor 1 and `1.130x` over factor 2. Factor 4 is retained; the already-rejected automatic full unroll remains over the spill cliff at 404 private bytes and 109 spills.

### Phase D4: simple heuristics and integration

After the three families have independent winners:
1. Encode only the measured exact-shape branches and coarse row thresholds.
2. Keep the general bounds-safe fallback for all other supported shapes and output ranks.
3. Run the full Qwen and DeepSeek matrices after every shared-core change.
4. Inspect code-object resources after each retained structural change, not only at the end.
5. Run `torch.compile`, FakeTensor, allocation, current-stream, sparse-route, and partial-tile tests.
6. Produce `/tmp/grouped_mmq_fwd_ds4_final_full.json` and a same-build `/tmp/grouped_mmq_fwd_qwen_post_ds4_control.json`.
7. Compare fresh pre/post Qwen controls as well as the historical G11 artifact.

#### D4.1 final dispatch and DeepSeek matrix: complete

The accepted static dispatch is:
- fixed Q8_0 `(1024,4096)`: J64.
- routed IQ2_XXS `(2048,4096)`: exact J64 below `rows = 512 * num_groups`, exact J80 at and above that threshold.
- routed Q2_K `(4096,2048)`: exact J32 with factor-4 Q2_K scale/min-loop unrolling; use a J16 tail body only when `rows < 64 * num_groups`.
- every other supported shape: the existing bounds-safe generic path.

The final pre-bundle 27-point artifact is `/tmp/grouped_mmq_fwd_ds4_last_version_baseline.json`. All 39 fixed and paired projection checks remain bitwise exact against dense packed MMQ. Independent BF16 NRMSE is `0.006022-0.006062` for Q8_0/IQ2_XXS and `0.010685-0.011403` for Q2_K.

| Family | Batch 1 packed ms | Batch 4 packed ms | Batch 16 packed ms | Baseline-to-final geometric speedup |
| --- | ---: | ---: | ---: | ---: |
| Fixed Q8_0 | `11.030` | `44.488` | `176.170` | `1.00x` |
| IQ2_XXS pair | `31.884-35.055` | `84.010-99.779` | `334.909-344.324` | `1.54x` |
| Q2_K down | `18.421-21.496` | `72.219-76.700` | `290.493-295.563` | `2.27x` |

Across all 27 points the geometric-mean baseline-to-final speedup is `1.75x`. IQ2_XXS and Q2_K beat AITER at all 24 routed points. Fixed Q8_0 remains approximately `0.74-0.76x` the independently dequantized BF16 BMM reference.

#### D4.2 Qwen code-object isolation and acceptance: complete

Adding the new routed specializations to the monolithic HIP translation unit produced a repeatable `2-4%` Q5_K batch-1 regression even though the Q5 arithmetic kernel, Q5 quantizer, and specialized host launcher had instruction-identical hashes. The regression followed code-object layout. The candidate was rejected in that form.

The retained implementation places DeepSeek-only J64/J80 IQ2_XXS and rolled-Q2 J32 instantiations in `csrc/deepseek_mmq_hip.cu`. The original generated Q2_K dot function and all Qwen arithmetic remain in `csrc/mmq_hip.cu`; a separate generator-owned rolled Q2_K helper is included only by the DeepSeek translation unit. The sensitive 25-repeat Q5_K batch-1 medians then matched baseline within `0.2%`, and the complete 25-repeat Q5 family control had no repeatable regression.

The final 60-point Qwen artifact is `/tmp/grouped_mmq_fwd_qwen_post_ds4_isolated_final.json`. Relative to `/tmp/grouped_mmq_fwd_qwen_pre_ds4_control.json`, median point movement is `+0.11%` and geometric-mean movement is `+0.16%`. Checkpoint-weighted bucket movement is `-0.22%` to `+0.88%`. The two nine-repeat movements above 1% were cleared by sequential 25-repeat controls: Q5_K batch-16 sparse improved to `24.459 ms` versus `24.914-25.336 ms` baseline, and down IQ2_S batch-16 sparse measured `30.645 ms` versus `30.665 ms` baseline. All 60 points remain bitwise exact.

#### D4.3 final production resources

| Kernel | J | VGPR | SGPR | Private bytes | VGPR spills | Dynamic stack |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Fixed Q8_0 full-N | 64 | 213 | 48 | 0 | 0 | no |
| IQ2_XXS small/medium rows | 64 | 229 | 77 | 0 | 0 | no |
| IQ2_XXS large rows | 80 | 253 | 51 | 0 | 0 | no |
| Q2_K fixed-tail production | 32 | 208 | 38 | 0 | 0 | no |
| Q2_K J32/J16 small-row production | 32/16 | 213 | 43 | 0 | 0 | no |

All retained DeepSeek arithmetic kernels satisfy the zero-private, zero-spill, zero-dynamic-stack gate.

#### D4.4 measured remaining bottleneck

Fresh accepted-build traces are `/tmp/rocprof_ds4_final_q2k_b1/q2k_final_results.db` and `/tmp/rocprof_ds4_final_iq2xxs_b16/iq2xxs_final_results.db`:
- Q2_K batch 1 averages `25.946 ms` arithmetic and `0.351 ms` quantization; arithmetic is about `98.7%` of those operator kernels.
- IQ2_XXS batch 16 averages `157.216 ms` per projection and `11.469 ms` for shared quantization; the two arithmetic launches are about `96.5%` of operator kernel time.
- The unchanged fixed-Q8 trace averages `10.052 ms` arithmetic and `0.964 ms` quantization; arithmetic is about `91%`.

The remaining limitation is packed arithmetic and repeated decode, not activation quantization, spilling, or launch setup. Q2_K nonuniform batch-4 routes pay one bounded J32 tail per active expert and repeat packed scale/min reconstruction across 64 output tiles. Fixed Q8_0 repeats Q8_0 scale staging while the BF16 BMM reference starts from already dequantized weights; the tested J32/J128 and direct DwarfStar schedules do not improve this contract. IQ2_XXS is already well ahead of AITER, and J80 is at the spill-free VGPR limit.

The bounded local schedule space is exhausted: fixed J32/J128, IQ J32/J96, Q2 J64, exact/runtime J64, split full/tail launches, tail replication, and shared Q8 loop rolling all failed either resource or complete-operator timing gates. More headroom requires a representation-level design such as a compact reusable decoded-weight cache or a transient dense stage, with separate memory and end-to-end evaluation.

### Closed neighborhoods carried forward from prior logs

Do not begin with the following mechanisms. They were neutral, invalid, or materially slower in the dense/grouped forward and backward passes:
- a global `J=64`, `J=128`, or `I=128` rule across quant types.
- eight-wave ownership of an `I=64` output tile without a valid reduction design.
- complete decoded-weight LDS caching at the current 64-row output tile.
- N-major row tasks, fixed-size persistent traversal, split full/tail launches, or host-built grouped descriptors.
- wholesale direct-to-VGPR, both-operands direct-to-VGPR, two-LDS pipelines, GSU, split-K, or grouped Stream-K.
- compiler-managed local arrays as packed prefetch state.
- keeping next-iteration packed fragments live across the current WMMA phase.
- broad swizzle or padding sweeps without a measured LDS-bank or fragment-load problem.
- assuming int8 WMMA has a raw arithmetic advantage over BF16 WMMA on gfx1151.

These mechanisms may be reopened only with new profile evidence specific to `Q8_0`, `IQ2_XXS`, or `Q2_K`. The plan deliberately starts with exact specialization, bounded lifetimes, full/tail separation, and simple scheduling because those were the durable wins across all previous logs.

## Benchmark infrastructure

`bench/benchmark_grouped_mmq_fwd.py` follows the dense benchmark conventions.

It records:
- complete public packed-operator latency.
- logical throughput.
- incremental peak allocation and reservation growth.
- model family, quant type, and physical packed shapes.
- AITER configuration for routed cases.
- routing metadata and distribution statistics.
- checkpoint-weighted forward and optimizer-step estimates.
- grouped-versus-dense-MMQ exactness.
- grouped-versus-independently-dequantized-BF16 error.

The packed timing includes Q8_1 quantization and grouped multiplication.

The routed BF16 performance reference is AITER Triton `gmm`, using the project-owned `torch_ggml_ops.aiter_gmm_heuristics.gmm_config`. AITER receives independently dequantized BF16 versions of the same logical GGUF experts. Dequantization and active-weight selection are setup costs and are not inside the timed GMM call. The reference uses production transposed weight metadata and keeps `work_stealing` disabled.

The fixed-group BF16 performance reference is `torch.bmm` over eight independently dequantized matrices. Its timed function includes conversion from the group-major batched-GEMM output to the public token-major contiguous layout.

The full baseline command was:

```bash
python bench/benchmark_grouped_mmq_fwd.py \
  --warmup 2 \
  --repeats 5 \
  --correctness-rows 128 \
  --output /tmp/grouped_mmq_fwd_baseline_full.json
```

The historical pre-optimization artifact is:

```text
/tmp/grouped_mmq_fwd_baseline_full.json
```

The final retained command uses the same arguments and writes:

```text
/tmp/grouped_mmq_fwd_final_full_v2.json
```

GPU benchmarks and profiler runs were sequential.

## Routing distributions

The benchmark covers four deterministic distributions.

`uniform` activates all 256 experts with equal group sizes.

`skewed` activates all 256 experts with non-multiple group sizes centered around the production mean.

`sparse` uses sparse active-expert IDs. It uses 192, 224, and 240 active experts for batches 1, 4, and 16.

`boundary` includes sizes 1, 15, 16, 17, 63, 64, 65, 127, 128, and 129 before filling the remaining groups. It exercises edge handling without making it the only performance distribution.

The following representative summaries are for the historical Qwen top-eight rows:

| Batch | Distribution | Active experts | Minimum | Maximum | Mean |
|---:|---|---:|---:|---:|---:|
| 1 | uniform | 256 | 64 | 64 | 64.0 |
| 1 | skewed | 256 | 42 | 86 | 64.0 |
| 1 | sparse | 192 | 64 | 106 | 85.3 |
| 1 | boundary | 256 | 1 | 129 | 64.0 |
| 4 | uniform | 256 | 256 | 256 | 256.0 |
| 4 | skewed | 256 | 192 | 320 | 256.0 |
| 4 | sparse | 224 | 221 | 367 | 292.6 |
| 16 | uniform | 256 | 1,024 | 1,024 | 1,024.0 |
| 16 | skewed | 256 | 768 | 1,280 | 1,024.0 |
| 16 | sparse | 240 | 821 | 1,367 | 1,092.3 |

DeepSeek top-six deterministic summaries are:

| Batch | Distribution | Active experts | Minimum | Maximum | Mean |
|---:|---|---:|---:|---:|---:|
| 1 | uniform | 256 | 48 | 48 | 48.0 |
| 1 | skewed | 256 | 26 | 70 | 48.0 |
| 1 | sparse | 192 | 48 | 80 | 64.0 |
| 1 | boundary | 256 | 1 | 129 | 48.0 |
| 4 | uniform | 256 | 192 | 192 | 192.0 |
| 4 | skewed | 256 | 128 | 256 | 192.0 |
| 4 | sparse | 224 | 165 | 275 | 219.4 |
| 4 | boundary | 256 | 1 | 217 | 192.0 |
| 16 | uniform | 256 | 768 | 768 | 768.0 |
| 16 | skewed | 256 | 512 | 1,024 | 768.0 |
| 16 | sparse | 240 | 616 | 1,026 | 819.2 |
| 16 | boundary | 256 | 1 | 877 | 768.0 |

The fixed output-A case does not use a routing distribution. All eight groups receive exactly `M` rows.

## Historical pre-G1 baseline results

The speedup column is `AITER time / packed MMQ time`. Values above 1.0 mean the packed path is faster.

### Uniform routing

| Case | Batch | Packed ms | AITER ms | Packed logical TFLOP/s | Speedup |
|---|---:|---:|---:|---:|---:|
| Gate/up Q3_K | 1 | 11.185 | 13.245 | 6.14 | 1.18x |
| Gate/up Q3_K | 4 | 32.572 | 33.679 | 8.44 | 1.03x |
| Gate/up Q3_K | 16 | 130.408 | 83.293 | 8.43 | 0.64x |
| Gate/up IQ2_S | 1 | 11.453 | 13.212 | 6.00 | 1.15x |
| Gate/up IQ2_S | 4 | 33.408 | 35.066 | 8.23 | 1.05x |
| Gate/up IQ2_S | 16 | 133.528 | 87.740 | 8.23 | 0.66x |
| Down IQ2_S | 1 | 6.326 | 3.578 | 5.43 | 0.57x |
| Down IQ2_S | 4 | 18.226 | 7.707 | 7.54 | 0.42x |
| Down IQ2_S | 16 | 69.996 | 44.702 | 7.85 | 0.64x |
| Down Q4_K | 1 | 6.874 | 3.524 | 5.00 | 0.51x |
| Down Q4_K | 4 | 21.977 | 7.730 | 6.25 | 0.35x |
| Down Q4_K | 16 | 83.663 | 44.266 | 6.57 | 0.53x |
| Down Q5_K | 1 | 6.953 | 3.575 | 4.94 | 0.51x |
| Down Q5_K | 4 | 21.990 | 7.777 | 6.25 | 0.35x |
| Down Q5_K | 16 | 84.674 | 43.794 | 6.49 | 0.52x |

### Sparse observed-style routing

| Case | Batch | Packed ms | AITER ms | Packed logical TFLOP/s | Speedup |
|---|---:|---:|---:|---:|---:|
| Gate/up Q3_K | 1 | 9.678 | 14.412 | 7.10 | 1.49x |
| Gate/up Q3_K | 4 | 35.091 | 36.875 | 7.83 | 1.05x |
| Gate/up Q3_K | 16 | 133.880 | 100.466 | 8.21 | 0.75x |
| Gate/up IQ2_S | 1 | 9.953 | 14.225 | 6.90 | 1.43x |
| Gate/up IQ2_S | 4 | 35.966 | 36.829 | 7.64 | 1.02x |
| Gate/up IQ2_S | 16 | 136.851 | 96.521 | 8.03 | 0.71x |
| Down IQ2_S | 1 | 5.544 | 2.839 | 6.20 | 0.51x |
| Down IQ2_S | 4 | 19.089 | 8.909 | 7.20 | 0.47x |
| Down IQ2_S | 16 | 71.064 | 44.474 | 7.74 | 0.63x |
| Down Q4_K | 1 | 6.342 | 2.821 | 5.42 | 0.44x |
| Down Q4_K | 4 | 22.196 | 8.817 | 6.19 | 0.40x |
| Down Q4_K | 16 | 84.262 | 44.430 | 6.52 | 0.53x |
| Down Q5_K | 1 | 6.394 | 2.842 | 5.37 | 0.44x |
| Down Q5_K | 4 | 22.310 | 8.805 | 6.16 | 0.39x |
| Down Q5_K | 16 | 84.894 | 43.764 | 6.48 | 0.52x |

At the historical baseline, gate/up was strong only for batch 1, approximately tied at batch 4, and behind at batch 16. Down was the dominant deficit for every batch. G1-G11 resolve those deficits except for six nonuniform IQ2_S down points documented in the final evaluation.

### Routing sensitivity

Across all four distributions, the average per-case speedups were:

| Case | Batch 1 | Batch 4 | Batch 16 |
|---|---:|---:|---:|
| Gate/up Q3_K | 1.35x | 1.04x | 0.71x |
| Gate/up IQ2_S | 1.31x | 1.03x | 0.69x |
| Down IQ2_S | 0.54x | 0.45x | 0.62x |
| Down Q4_K | 0.49x | 0.39x | 0.52x |
| Down Q5_K | 0.49x | 0.39x | 0.51x |

Packed gate/up benefits from sparse batch-1 routing because inactive experts do not launch grouped workgroups. AITER's fixed persistent grid still scans the active group list.

The historical pre-G1 kernel was mildly sensitive to skew at batch 4. Each `(expert, output tile)` workgroup serially processed every 128-row chunk for its expert, so large groups created longer-lived workgroups. G4 reduced the row tile to 64, and G8 exposed large gate/up row tiles as independent device tasks.

### Checkpoint-weighted grouped base-projection estimate

The estimate applies the checkpoint layer counts and two executions per optimizer step under activation checkpointing. It covers grouped base projections only, not routing, activation functions, LoRA, or other model work.

| Batch | Distribution | Packed seconds | AITER seconds | Packed speedup |
|---:|---|---:|---:|---:|
| 1 | uniform | 1.434 | 1.343 | 0.94x |
| 1 | skewed | 1.430 | 1.500 | 1.05x |
| 1 | sparse | 1.261 | 1.372 | 1.09x |
| 1 | boundary | 1.439 | 1.501 | 1.04x |
| 4 | uniform | 4.247 | 3.367 | 0.79x |
| 4 | skewed | 4.546 | 3.754 | 0.83x |
| 4 | sparse | 4.494 | 3.657 | 0.81x |
| 4 | boundary | 4.626 | 3.643 | 0.79x |
| 16 | uniform | 16.708 | 10.398 | 0.62x |
| 16 | skewed | 17.022 | 11.405 | 0.67x |
| 16 | sparse | 17.045 | 11.433 | 0.67x |
| 16 | boundary | 17.015 | 10.861 | 0.64x |

This table is the historical checkpoint-weighted baseline that motivated the implementation pass. The final checkpoint-weighted results are reported in `Final retained evaluation` and are faster than AITER for every batch and distribution.

AITER is the production BF16 reference. It is not a performance ceiling for a packed kernel with much smaller authoritative weights.

## Correctness

Every one of the 60 measured production points matched the concatenation of per-group dense MMQ outputs exactly in BF16.

This checks the grouped scheduler against the same Q8_1 activation semantics and the same packed GGUF decode path.

Against independently dequantized BF16 weights and BF16 AITER GMM, normalized RMSE ranges were:

| Type | Minimum NRMSE | Maximum NRMSE |
|---|---:|---:|
| Q3_K | 0.00599 | 0.00613 |
| IQ2_S | 0.00600 | 0.00611 |
| Q4_K | 0.01120 | 0.01381 |
| Q5_K | 0.01300 | 0.01650 |

These errors include the intended internal activation quantization.

## Incremental memory

The pair path uses one Q8_1 workspace for both outputs.

| Operator shape | Batch | Packed peak | AITER peak | Q8_1 workspace | Output bytes |
|---|---:|---:|---:|---:|---:|
| Pair `R x 2048 -> 2 x R x 512` | 1 | 68 MiB | 32 MiB | 36 MiB | 32 MiB |
| Pair `R x 2048 -> 2 x R x 512` | 4 | 272 MiB | 128 MiB | 144 MiB | 128 MiB |
| Pair `R x 2048 -> 2 x R x 512` | 16 | 1,088 MiB | 512 MiB | 576 MiB | 512 MiB |
| Down `R x 512 -> R x 2048` | 1 | 73 MiB | 64 MiB | 9 MiB | 64 MiB |
| Down `R x 512 -> R x 2048` | 4 | 292 MiB | 256 MiB | 36 MiB | 256 MiB |
| Down `R x 512 -> R x 2048` | 16 | 1,168 MiB | 1,024 MiB | 144 MiB | 1,024 MiB |

The measurements exclude resident packed or BF16 reference weights.

The pair workspace saving is material: two independent packed calls would need two quantization launches and two workspace lifetimes.

## Final packed kernel structure

The retained production arithmetic geometry is:

```text
I = 64 output columns
J = 64 routed rows
threads = 128, four wave32 waves
K iteration = 256 packed values
```

Both production families are compile-time specialized:
- gate/up: `NRowsWeight=512`, `BlocksPerWeightRow=8`.
- down: `NRowsWeight=2048`, `BlocksPerWeightRow=2`.

Common behavior:
- exact output tiles have no output-row fallback.
- full row tiles use contiguous Q8_1 loads and unmasked BF16 stores.
- only the final partial row tile uses bounded zero fill and masked stores.
- packed weights and Q8_1 activations are staged in LDS.
- public inputs and outputs remain BF16 and packed GGUF weights remain authoritative.

Gate/up scheduling is shape- and row-count-specific:
- batch-1-sized groups retain one `(expert, output tile)` workgroup with a serial row loop, preserving sparse launch behavior.
- when the host-visible average reaches at least two 64-row tiles, one GPU setup workgroup builds atomics-free `(expert, row_start, row_end)` task descriptors.
- `grouped_mmq_pair` builds descriptors once and reuses them for gate and up.

Down retains serial expert row ownership because it already launches 32 output tiles per active expert. Its two fixed K blocks are emitted as two explicit calls, and partial activation tiles use one contiguous integer-span predicate.

Dynamic LDS is 30,976 bytes for Q3_K/IQ2_S and 28,928 bytes for Q4_K/Q5_K. A general `J=128` fallback remains only for tests and non-production shapes.

## Historical pre-G1 packed-kernel profiling

The following traces describe the original spill-heavy baseline and are retained to explain why G1 was prioritized. Kernel-trace artifacts are:

```text
/tmp/rocprof_grouped_gate_b1
/tmp/rocprof_grouped_gate_b16
/tmp/rocprof_grouped_down_q4_b4
```

### Gate/up Q3_K, batch 1, uniform

The profiled pair decomposed into:
- Q8_1 quantization: 0.829 ms.
- first grouped projection: 5.408 ms.
- second grouped projection: 5.359 ms.

The grouped projections account for about 93% of the packed operator time.

### Gate/up Q3_K, batch 16, uniform

The profiled pair decomposed into:
- Q8_1 quantization: 7.894 ms.
- two grouped projections: 122.445 ms total.

The grouped projections account for about 94% of the packed operator time.

### Down Q4_K, batch 4, uniform

The profiled single projection decomposed into:
- Q8_1 quantization: 0.916 ms.
- grouped projection: 21.964 ms.

The multiplication kernel is the first-order bottleneck. More quantizer work is not justified before fixing it.

## Historical pre-G1 packed code-object resources

The pre-G1 extension code object was extracted from `.hip_fatbin` and inspected with `clang-offload-bundler`, `llvm-readobj`, `llvm-nm`, and `llvm-objdump`. These resources are historical. Final retained resources are reported later.

| Grouped kernel | VGPRs | SGPRs | Private bytes/thread | VGPR spills |
|---|---:|---:|---:|---:|
| Q3_K | 256 | 74 | 124 | 30 |
| Q4_K | 256 | 76 | 512 | 127 |
| Q5_K | 256 | 74 | 520 | 129 |
| Q6_K | 256 | 74 | 272 | 67 |
| IQ2_S | 255 | 78 | 132 | 32 |

The corresponding dense J=128 kernels have zero private segment and zero spills:

| Dense kernel | VGPRs | SGPRs | Private bytes/thread | VGPR spills |
|---|---:|---:|---:|---:|
| Q3_K | 216 | 29 | 0 | 0 |
| Q4_K | 254 | 30 | 0 | 0 |
| Q5_K | 230 | 29 | 0 | 0 |
| IQ2_S | 208 | 34 | 0 | 0 |

The packed arithmetic body is not inherently forced to spill. The grouped expert metadata and dynamic row-chunk loop push the inherited dense body over the register limit.

Static disassembly reinforces this conclusion. The grouped Q4_K function has 121 scratch-load and 76 scratch-store instruction sites, compared with 7 and 6 in the profiled AITER down kernel.

Q4_K and Q5_K spill roughly half a kilobyte per thread. This is the clearest explanation for the severe down-projection gap.

## AITER source, lowering, and historical pre-G1 comparison profile

The inspected AITER source is:

```text
~/venv_torch/lib/python3.14/site-packages/aiter/ops/triton/gmm.py
~/venv_torch/lib/python3.14/site-packages/aiter/ops/triton/_triton_kernels/gmm.py
```

AITER uses a 256-program persistent grid.

Each program starts from its program ID and advances through logical GMM tiles by `GRID_DIM`. It walks the device-resident group-size array and never requires a host-side group descriptor.

The tile mapping uses XCD remapping. Edge tiles wrap input row and output-column load offsets with modulo arithmetic, then mask only the final store.

The K loop uses direct BF16 loads and `tl.dot`. Generated gfx1151 ISA contains `v_wmma_f32_16x16x16_bf16` instructions.

The production heuristic selects:

| Shape | M tile | N tile | K tile | Threads | Persistent programs |
|---|---:|---:|---:|---:|---:|
| Gate/up `K=2048, N=512` | 64 | 128 | 64 | 256 | 256 |
| Down `K=512, N=2048` | 128 | 128 | 64 | 256 | 256 |

The generated gate/up kernel uses 176 VGPRs, 58 SGPRs, no private segment, and no fixed LDS.

The generated down kernel uses 255 VGPRs, 57 SGPRs, and 48 private bytes per thread. It is near the register limit but its spill footprint is much smaller than packed Q4_K/Q5_K.

The gate/up AITER profile launched 256 workgroups of 256 threads. One profiled projection took 7.849 ms under kernel tracing.

The down AITER profile used the same persistent grid and took 8.156 ms under kernel tracing for the batch-4 uniform case.

Profiler time is perturbed and is used only for decomposition and resource comparison. CUDA-event medians remain the latency source of record.

Counter runs on down Q4_K batch 4 reported:

| Kernel | Occupancy | L2 hit rate |
|---|---:|---:|
| Packed grouped Q4_K | 18.44% | 69.73% |
| BF16 AITER GMM | 23.20% | 71.68% |

Packed `ALUStalledByLDS` was only 0.0154%. LDS bank stalls are therefore not the primary problem.

The similar L2 hit rates argued against treating cache hit rate as the first optimization target in the pre-G1 kernel. Register spills, tile shape, and the amount of scheduled work were stronger baseline explanations.

`MemUnitBusy` was unavailable through dispatch-windowed counter collection on this gfx1151 profiler configuration and is not treated as a result.

## TensileLite, CK, and dense-MMQ mechanism study

This study predates G1-G11 and is retained for mechanism and design provenance. The outcomes are stated explicitly so that the original hypotheses are not mistaken for remaining tasks.

### TensileLite evidence

Principal sources:

```text
~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/Tests/common/groupedgemm/grouped_gemm.yaml
~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/Tests/common/groupedgemm/gfx11/grouped_gemm_gfx11.yaml
~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/Tests/common/groupedgemm/gfx11/grouped_gemm_userargs_gfx11.yaml
~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/SolutionStructs/Solution.py
~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/KernelWriterAssembly.py
~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/Components/SIA.py
~/rocm-libraries/projects/hipblaslt/tensilelite/client/src/ClientProblemFactory.cpp
~/rocm-libraries/projects/hipblaslt/tensilelite/client/src/SolutionIterator.cpp
~/rocm-libraries/projects/hipblaslt/tensilelite/src/ContractionSolution.cpp
```

TensileLite can tune a complete grouped BF16 workload, but every GEMM in one group uses one static solution. That model matches separate fixed-shape gate/up and down families, but it cannot directly express Q8_1 activation handling, packed GGUF decode, scale application, device routing, and direct BF16 output.

The grouped client combines exact GEMMs into one `ContractionProblemGroupedGemm`, checks a candidate against every member, and times the complete enqueue. Its displayed grouped GFLOP/s is not a valid aggregate throughput because `BenchmarkTimer.cpp` uses only `problem->gemms[0].flopCount()` as the numerator. Historical grouped client results therefore remain capability evidence, not a performance bound.

TensileLite's runtime helper normally constructs grouped user-argument records on the host and copies them to the device. That setup is incompatible with the production routing ABI. The useful transferable idea was compact cumulative device metadata, which G8 implemented with one GPU prefix setup and direct row-task indexing.

| Mechanism | Durable lesson | Final outcome |
|---|---|---|
| Static macro-tile and matrix-instruction selection | Measure compile-time `I`, `J`, and thread-count families | `I=64`, `J=64`, 128 threads retained. G3/G9 rejected wider tiles |
| Fixed `DepthU` and assertions | Specialize eight-block gate/up and two-block down traversal | Retained in G1, G6, and G7 |
| SGPR/immediate global-read offsets | Prefer affine pointer increments and scalar fixed offsets on exact tiles | Gate/up pointer increments retained in G7 |
| Algorithm-3 issue scheduling | Split read/decode/commit lifetimes before adding pacing | Deeper staged prefetch closed because final kernels are already near the VGPR limit |
| Wave-separated reads | Keep weight ownership wave-local where geometry divides | Useful layout principle. No separate retained launch variant |
| `DirectToVgpr` | At most one immediately consumed decoded-weight fragment could be plausible | Wholesale and both-operands paths rejected. Not pursued after resource closure |
| One/two LDS buffers | Extra staging needs a measured residency argument | Complete down decoded cache lost 27-40%. Second-stage work remains closed |
| Store remapping | C-shuffle can trade LDS and synchronization for coalesced output | Not retained. Direct BF16 writeback is not the final bottleneck |
| GSU, split-K, Stream-K | Duplicates packed decode or requires reduction | Rejected for fixed K=512/2048. Grouped Stream-K is unsupported |

`UseSgprForGRO` remains a useful code-generation lesson: one per-lane VGPR base plus scalar offsets can reduce address state for affine exact tiles. Shift-pointer edge handling is incompatible with this form, which reinforces separate full and tail bodies. Explicit `readfirstlane` is justified only when final ISA proves that a wave-uniform value remained in VGPRs.

Algorithm-3 scheduling also established an ordering rule that remains useful for any future representation:
- read a bounded packed fragment.
- decode with temporary metadata and bit-extraction state.
- commit decoded BF16 values to LDS or form one immediately consumed register operand.
- end decode-temporary lifetimes.
- issue LDS reads and WMMA while only bounded next-fragment state is live.

Adding barriers to an unchanged monolithic decoder does not reproduce this dataflow. G5 confirmed that removing one existing write barrier is neutral, while the final VGPR allocations make a deeper live prefetch window unattractive.

TensileLite direct-to-VGPR assumes ordinary layout conversion. GGUF formats require scale reconstruction, metadata interpretation, arbitrary bit extraction, and BF16 formation. Direct global-to-LDS likewise cannot perform GGUF decode. Activations must remain in LDS because all four waves reuse them.

### Composable Kernel evidence

Relevant sources:

```text
~/rocm-libraries/projects/composablekernel/example/15_grouped_gemm/grouped_gemm_wmma_fixed_nk_fp16.cpp
~/rocm-libraries/projects/composablekernel/example/15_grouped_gemm/grouped_gemm_wmma_splitk_bf16.cpp
~/rocm-libraries/projects/composablekernel/example/ck_tile/17_grouped_gemm/grouped_gemm.cpp
~/rocm-libraries/projects/composablekernel/include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_fixed_nk_common.hpp
~/rocm-libraries/projects/composablekernel/include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_wmma_fixed_nk.hpp
~/rocm-libraries/projects/composablekernel/include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_wmma_cshuffle_tile_loop_v3.hpp
~/rocm-libraries/projects/composablekernel/include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v1.hpp
~/rocm-libraries/projects/composablekernel/include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v3.hpp
~/rocm-libraries/projects/composablekernel/include/ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp
```

The useful CK lessons were structural rather than directly reusable kernels:
- fixed-NK specialization removes dynamic N/K scheduler state.
- load, decode, LDS commit, LDS read, and WMMA phases need bounded lifetimes.
- tile ordering should preserve locality without adding excessive persistent control state.
- C-shuffle and extra pipeline stages are worthwhile only with a resource argument.
- direct global-to-LDS is not a gfx1151 CK mechanism.

The fixed-NK hypothesis produced the largest retained change. Production shapes are exact and narrow:

| Projection | Output rows per expert | Input features | GGUF blocks per weight row |
|---|---:|---:|---:|
| Gate/up | 512 | 2,048 | 8 |
| Down | 2,048 | 512 | 2 |

G1 specialized these dimensions at compile time and removed output-tile edge predicates. G6 retained an explicit two-block down schedule, while G7 retained a rolled eight-block gate/up loop with pointer increments.

The complete decoded-down-weight cache was also a direct CK-inspired hypothesis. Its estimated dynamic LDS was 48,384 bytes for Q4_K/Q5_K and 52,480 bytes for IQ2_S at `J=64`. G2 measured 27-40% regressions because the additional 19-22 KiB LDS cost outweighed cross-row decode reuse. This design must not be retried without a substantially more compact cached representation.

The device scheduling study led to G8. With `G <= 256`, one 256-thread workgroup computes task counts, performs an atomics-free prefix sum, and writes device-resident `(expert, row_start, row_end)` arrays with capacity:

```text
ceil(R / 64) + G
```

Output tile is the fastest-changing launch dimension. Large gate/up groups gain approximately 1-2%, and paired projections reuse the setup. Down descriptors lost 2-14% because 32 output tiles per expert already expose enough parallelism. Persistent grid-stride traversal was not pursued because the nonpersistent gain was small and extra control state would threaten sparse behavior and register headroom.

G4 retained separate full and tail bodies in one public launch. Full rows have no activation bounds handling and use unmasked BF16 stores. G11 further reduced down-tail address work to one contiguous integer-span predicate. A second full-tile launch, clamped tail arithmetic, and C-shuffle do not have enough remaining headroom to justify more local work.

### Dense-MMQ lessons carried into grouped MMQ

- Inspect the final code object after every structural change. Source-level register intuition was insufficient.
- Use compile-time typed variants and measured dispatch because decode and resource behavior differ by quant type.
- Optimize complete public-operator latency. Final multiplication still accounts for approximately 87-91% of retained kernel time.
- Larger cooperative tiles help only when reuse exceeds their LDS, accumulator, and workgroup-residency costs. G3 and G9 show that this condition does not hold here.
- Type-specific packed extraction can win, but the final grouped limit is IQ2_S representation cost rather than a universal decoder schedule.
- gfx1151 int8 WMMA has no decisive raw throughput advantage over BF16 WMMA. Packed MMQ must win through authoritative-weight compression, reuse, scheduling, and lower traffic.

## Historical baseline diagnosis and final resolution

### Register spilling was the first blocker

The original grouped Q4_K/Q5_K kernels reached 256 VGPRs and spilled 127-129 VGPRs per thread. G1's compile-time `J=64` and fixed production shapes removed every production private segment and spill. Final kernels remain spill-free.

### Larger output or row tiles did not provide the next gain

The original packed tile was narrower than AITER, which made `I=128` and `J=128` reasonable hypotheses. G3 and G9 tested those neighborhoods and both regressed despite reduced or eliminated spilling. The final `I=64, J=64` geometry is retained.

### Scheduling must remain shape-specific

The original serial expert loop limited gate/up balance at larger batches. G8's atomics-free row-task descriptors provide a small but repeatable 1-2% gain for large gate/up groups and are reused by paired projections.

Down already exposes 32 output tiles per expert. Device row descriptors regressed it by 2-14%, so down retains serial row ownership. A complete decoded-weight LDS cache also regressed by 27-40% because its extra 19-22 KiB LDS cost outweighed reuse.

### Quantization and pair fusion were not the main limit

Gate/up already shares one Q8_1 workspace. Final profiling still assigns approximately 87-91% of operator kernel time to packed multiplication. A fused two-weight arithmetic kernel would require a second live accumulator set and is not justified by the remaining headroom.

### Final bottleneck

The remaining six losses are nonuniform IQ2_S down points at batch 1 and batch 4. They are caused by IQ2_S decode and zero-padded partial `J=64` work repeated across 32 output tiles, not by spills, descriptor setup, or generic metadata traversal. Further worthwhile work requires a changed representation or cross-call decode reuse.

## Optimization experiment log

### G1: compile-time `J=64` and fixed production shapes

Status: retained.

The first implementation combined the two highest-priority spill-removal changes:
- compile-time `J=64` for both production shape families.
- fixed gate/up `NRowsWeight=512, BlocksPerWeightRow=8` and down `NRowsWeight=2048, BlocksPerWeightRow=2` variants.
- no output-row fallback in the fixed variants because every production output tile is complete.
- a fixed-trip K loop while retaining the general `J=128` fallback for tests and non-production shapes.

The focused timing artifact is:

```text
/tmp/grouped_step1_j64_fixed.json
```

| Point | Baseline ms | G1 ms | Speedup |
|---|---:|---:|---:|
| Gate/up Q3_K batch 1 uniform | 11.185 | 6.170 | 1.81x |
| Gate/up Q3_K batch 1 sparse | 9.678 | 7.166 | 1.35x |
| Gate/up Q3_K batch 4 boundary | 35.929 | 24.609 | 1.46x |
| Gate/up Q3_K batch 16 uniform | 130.408 | 94.200 | 1.38x |
| Gate/up IQ2_S batch 1 sparse | 9.953 | 7.566 | 1.32x |
| Down Q4_K batch 1 uniform | 6.874 | 2.842 | 2.42x |
| Down Q4_K batch 4 uniform | 21.977 | 10.599 | 2.07x |
| Down Q4_K batch 4 boundary | 22.934 | 11.418 | 2.01x |
| Down Q4_K batch 16 uniform | 83.663 | 42.110 | 1.99x |
| Down Q5_K batch 4 uniform | 21.990 | 10.688 | 2.06x |
| Down IQ2_S batch 16 uniform | 69.996 | 45.963 | 1.52x |

The fixed production kernels are spill-free:

| Production kernel | VGPRs | SGPRs | Private bytes/thread | Dynamic LDS |
|---|---:|---:|---:|---:|
| Gate/up Q3_K | 164 | 46 | 0 | 30,976 bytes |
| Gate/up IQ2_S | 190 | 47 | 0 | 30,976 bytes |
| Down IQ2_S | 190 | 49 | 0 | 30,976 bytes |
| Down Q4_K | 168 | 49 | 0 | 28,928 bytes |
| Down Q5_K | 181 | 48 | 0 | 28,928 bytes |

This removes the original 124-520 byte private segments and 30-129 VGPR spills without introducing dynamic stack use. The retained kernels also reduce dynamic LDS by 9,472 bytes relative to the original `J=128` grouped path.

Production-shape correctness was checked against dense MMQ:
- gate/up Q3_K batch 1 sparse: both outputs had zero differing BF16 elements.
- down Q4_K batch 4 uniform: zero differing BF16 elements.

The complete public gate/up point measured 7.155 ms versus 14.155 ms AITER, or 1.98x faster. The complete public down Q4_K batch-4 point measured 10.617 ms versus 7.805 ms AITER, or 0.74x. Spill removal therefore explains and resolves most of the original deficit, but batch-4 down and batch-16 gate/up remain the primary performance gaps.

### G1 decision

Keep the combined specialization. Separating fixed shape from `J=64` is not necessary for acceptance because the combined variant is spill-free and materially faster at every focused production point. Future variants must compare against G1 rather than the original baseline.

### G2: complete decoded-weight LDS cache for down

Status: rejected and reverted.

G2 decoded both fixed `K=512` weight blocks into immutable LDS before the serial row loop and reused them across all `J=64` row chunks. The cache was dispatched only when the host-visible average was at least two row tiles, preserving the G1 batch-1 path.

The artifact is:

```text
/tmp/grouped_step2_down_cache.json
```

| Point | G1 ms | G2 ms | Relative |
|---|---:|---:|---:|
| Down Q4_K batch 4 uniform | 10.599 | 15.427 | 0.69x |
| Down Q4_K batch 4 boundary | 11.418 | 15.705 | 0.73x |
| Down Q4_K batch 16 uniform | 42.110 | 70.334 | 0.60x |
| Down Q5_K batch 4 uniform | 10.688 | 15.503 | 0.69x |
| Down IQ2_S batch 16 uniform | 45.963 | 72.384 | 0.64x |

The additional immutable weight tile increased dynamic LDS from 28,928 to 48,384 bytes for Q4_K/Q5_K and from 30,976 to 52,480 bytes for IQ2_S. The saved packed loads and decode did not compensate for the resulting resource and residency loss. This also shows that the original down deficit is no longer caused primarily by repeated decode once G1 has removed scratch traffic.

Do not retry the complete cache with the same 64-row output tile. Any future cross-row weight reuse must either use a more compact decoded representation or change the output/workgroup geometry enough to recover residency.

### G3: fixed-shape `J=128`

Status: rejected and reverted.

G3 kept fixed production N/K specialization but restored `J=128` to determine whether halving the serial row-loop count could beat G1 after dynamic shape state was removed.

The artifact is:

```text
/tmp/grouped_step3_j128_fixed.json
```

Every focused point regressed by 25-50% relative to G1. Representative results were:

| Point | G1 `J=64` ms | G3 `J=128` ms | Relative |
|---|---:|---:|---:|
| Gate/up Q3_K batch 1 uniform | 6.170 | 11.167 | 0.55x |
| Gate/up Q3_K batch 16 uniform | 94.200 | 129.555 | 0.73x |
| Down Q4_K batch 4 uniform | 10.599 | 16.764 | 0.63x |
| Down Q4_K batch 16 uniform | 42.110 | 65.776 | 0.64x |
| Down Q5_K batch 4 uniform | 10.688 | 17.727 | 0.60x |
| Down IQ2_S batch 16 uniform | 45.963 | 66.939 | 0.69x |

Fixed specialization reduced but did not eliminate `J=128` pressure for the main down types: Q4_K used 256 VGPRs and 184 private bytes/thread, while Q5_K used 256 VGPRs and 232 private bytes/thread. Q3_K retained 12 private bytes/thread. IQ2_S was spill-free at 241 VGPRs but still lost heavily, showing that the larger accumulator and LDS footprint is itself unfavorable even without scratch traffic.

Keep `J=64` for all production grouped-forward types. The fixed `I=64, J=128` neighborhood is closed.

### G4: separate exact full-row and bounded tail bodies

Status: retained.

G4 split the serial expert loop into compile-time full-row and tail helpers. A full `J=64` row tile now:
- loads the Q8_1 activation region as one contiguous integer span with no row division, remainder, or predicate.
- writes a complete BF16 row tile with no row predicate.
- retains the bounded zero-fill and masked-store path only for the final partial tile.

The focused artifact is:

```text
/tmp/grouped_step4_full_rows.json
```

| Point | G1 ms | G4 ms | Speedup |
|---|---:|---:|---:|
| Gate/up Q3_K batch 1 uniform | 6.170 | 3.735 | 1.65x |
| Gate/up Q3_K batch 1 sparse | 7.166 | 5.427 | 1.32x |
| Gate/up Q3_K batch 4 boundary | 24.609 | 16.755 | 1.47x |
| Gate/up Q3_K batch 16 uniform | 94.200 | 57.956 | 1.63x |
| Gate/up IQ2_S batch 1 sparse | 7.566 | 6.065 | 1.25x |
| Down Q4_K batch 1 uniform | 2.842 | 1.599 | 1.78x |
| Down Q4_K batch 4 uniform | 10.599 | 6.063 | 1.75x |
| Down Q4_K batch 4 boundary | 11.418 | 6.889 | 1.66x |
| Down Q4_K batch 16 uniform | 42.110 | 24.292 | 1.73x |
| Down Q5_K batch 4 uniform | 10.688 | 6.067 | 1.76x |
| Down IQ2_S batch 16 uniform | 45.963 | 32.846 | 1.40x |

The production kernels remain spill-free. Q3_K uses 168 VGPRs, Q4_K uses 168, Q5_K uses 175 for down, and IQ2_S uses 225. The extra compile-time full/tail code raises some register counts but does not create a private segment.

Production correctness remained exact against dense MMQ for both paths:
- gate/up Q3_K batch 1 sparse: zero differing BF16 elements for both outputs.
- down Q4_K batch 4 boundary: zero differing BF16 elements.

The complete gate/up Q3_K batch-1 sparse point measured 5.368 ms versus 14.131 ms AITER, or 2.63x. The complete down Q4_K batch-4 boundary point measured 6.962 ms versus 9.286 ms AITER, or 1.33x. G4 moves every focused production class ahead of its baseline AITER reference.

The magnitude of this gain shows that integer row decomposition and per-load tail control, not only register spilling, were first-order costs in the inherited grouped loop.

### G5: remove the post-write row barrier

Status: rejected and reverted.

The barrier after BF16 writeback is not required for shared-memory correctness because the preceding post-dot barrier already ends the decoded-weight and activation LDS lifetime. Removing it, however, produced only noise-level changes from 0.975x to 1.004x across the focused matrix. The artifact is:

```text
/tmp/grouped_step5_no_write_barrier.json
```

Keep the barrier in the retained source. It is not a measurable bottleneck, and retaining the simpler row-iteration synchronization structure is preferable to a neutral change.

### G6: compile-time two-block down unroll

Status: retained for the fixed `K=512` down shape.

G6 factored one packed K-block operation into a force-inlined helper and emits two explicit calls for down's exact `BlocksPerWeightRow=2`. Gate/up and the general fallback retain the original fixed-trip loop, avoiding full unrolling of the eight-block gate/up path.

The retained artifact is:

```text
/tmp/grouped_step6b_down_unroll.json
```

| Point | G4 ms | G6 ms | Speedup |
|---|---:|---:|---:|
| Down Q4_K batch 1 uniform | 1.599 | 1.565 | 1.02x |
| Down Q4_K batch 4 uniform | 6.063 | 5.877 | 1.03x |
| Down Q4_K batch 4 boundary | 6.889 | 6.882 | 1.00x |
| Down Q4_K batch 16 uniform | 24.292 | 23.057 | 1.05x |
| Down Q5_K batch 4 uniform | 6.067 | 5.849 | 1.04x |
| Down IQ2_S batch 16 uniform | 32.846 | 27.102 | 1.21x |

The gate/up body remained neutral after restoring its original loop inline: Q3_K batch 16 changed from 57.956 to 58.107 ms and batch-1 sparse from 5.427 to 5.383 ms.

All down variants remain spill-free. The explicit two-block schedule raises VGPR allocation to 207 for Q3_K, 223 for Q4_K, 230 for Q5_K, 217 for Q6_K, and 218 for IQ2_S, with 46 SGPRs and zero private segment. The register increase is acceptable because the fixed-shape kernels remain below the spill threshold and the complete operator improves.

Production correctness remained exact against dense MMQ for down Q4_K batch-4 boundary and down IQ2_S batch-16 uniform, with zero differing BF16 elements. Complete timings were 6.931 versus 9.278 ms AITER for Q4_K and 28.042 versus 45.382 ms AITER for IQ2_S.

The large IQ2_S gain and smaller Q4_K/Q5_K gains show that fixed-trip branch/address cleanup remains useful after G4, but its value is type-dependent and bounded by register growth.

### G7: pointer-increment gate/up K traversal

Status: retained.

G7 keeps the eight-block gate/up loop rolled but replaces per-iteration weight-block and activation-plane multiplication with incremented packed-weight indices and Q8_1 pointers. The full-row path now loads directly from the current and next activation plane.

The main artifact is:

```text
/tmp/grouped_step7_gate_pointers.json
```

Relative to G4, gate/up Q3_K improved by 0.4-1.7% across the focused matrix:

| Point | G4 ms | G7 ms | Speedup |
|---|---:|---:|---:|
| Gate/up Q3_K batch 1 uniform | 3.735 | 3.713 | 1.01x |
| Gate/up Q3_K batch 1 sparse | 5.427 | 5.358 | 1.01x |
| Gate/up Q3_K batch 4 boundary | 16.755 | 16.480 | 1.02x |
| Gate/up Q3_K batch 16 uniform | 57.956 | 57.708 | 1.00x |

The batch-1 IQ2_S sparse point was neutral, but separate large-group A/B measurements showed the pointer path improving IQ2_S from 17.306 to 16.930 ms at batch 4 and from 67.928 to 66.909 ms at batch 16. Those artifacts are `/tmp/grouped_step6_iq2_large.json` and `/tmp/grouped_step7_iq2_large.json`.

Production gate/up kernels remain spill-free. The pointer state raises VGPR allocation to 177 for Q3_K/Q4_K, 188 for Q5_K, 190 for Q6_K, and 240 for IQ2_S while reducing Q3_K SGPR allocation from 49 to 44. The complete gains are small but consistent at the important Q3_K and large IQ2_S points, so the affine pointer form is retained.

### G8: atomics-free row-task descriptors for large gate/up groups

Status: retained for gate/up when the host-visible average is at least two `J=64` row tiles. Rejected for down.

G8 adds a one-workgroup GPU setup pass. Each of at most 256 threads computes one group's task count, participates in a shared-memory prefix sum, and writes compact `(expert, row_start, row_end)` records without atomics. The capacity remains bounded by:

```text
ceil(R / 64) + G
```

The task count and three descriptor arrays remain device-resident. The compute grid indexes descriptors directly with output tile as the fastest-changing launch dimension. `grouped_mmq_pair` builds descriptors once and reuses them for gate and up. Batch-1 keeps G7's sparse serial dispatch because the descriptor threshold is not met.

The focused artifacts are:

```text
/tmp/grouped_step8_gate_descriptors.json
/tmp/grouped_step8_gate_descriptors_15.json
/tmp/grouped_step7_gate_serial_15.json
/tmp/grouped_step8_iq2_descriptors.json
```

The fair 15-repeat sequential A/B comparison, including setup, descriptor allocation, excess bounded workgroups, quantization, and both gate/up projections, measured:

| Point | G7 serial ms | G8 descriptors ms | Speedup |
|---|---:|---:|---:|
| Gate/up Q3_K batch 4 boundary | 16.576 | 16.348 | 1.01x |
| Gate/up Q3_K batch 16 uniform | 57.660 | 56.329 | 1.02x |
| Gate/up IQ2_S batch 16 uniform | 66.966 | 66.314 | 1.01x |

The descriptor arithmetic kernel remains spill-free and uses fewer VGPRs than the serial G7 body: 173 for Q3_K and 213 for IQ2_S, with 54/50 SGPRs. The setup kernel uses 13 VGPRs, 23 SGPRs, 1 KiB LDS, and no private segment. Production gate/up Q3_K batch-4 boundary remained exactly equal to dense MMQ for both outputs.

A down-only descriptor dispatch was also tested and reverted. The artifact is:

```text
/tmp/grouped_step8b_down_descriptors.json
```

Down already launches 32 output tiles per active expert and saturates the GPU without row descriptors. Extra task parallelism regressed batch-4 Q4_K by 2.8%, boundary Q4_K by 4.4%, batch-16 Q4_K by 1.8%, batch-4 Q5_K by 1.9%, and batch-16 IQ2_S by 13.9%. Keep G6's serial row ownership for down.

### G8 correctness coverage

`tests/test_grouped_mmq.py::test_grouped_pair_production_row_tasks_match_dense` exercises the production `512 x 2048` paired descriptor path with both full and tail row tasks and requires bit-exact BF16 equality against concatenated dense MMQ calls.

### G9: `I=128, J=64`, 256 threads

Status: rejected and reverted.

G9 doubled the output tile and workgroup size globally for the focused production experiment. This kept 32 FP32 accumulators per thread and all selected production kernels remained spill-free, but dynamic LDS rose to approximately 50-55 KiB and workgroup-level scheduling became less favorable.

The artifact is:

```text
/tmp/grouped_step9_i128.json
```

Every focused point regressed relative to the retained `I=64` dispatch. Representative comparisons were:

| Point | Retained ms | G9 ms | Relative |
|---|---:|---:|---:|
| Gate/up Q3_K batch 1 uniform | 3.52-3.71 | 3.918 | 0.90-0.95x |
| Gate/up Q3_K batch 4 boundary | 16.348 | 17.068 | 0.96x |
| Gate/up Q3_K batch 16 uniform | 56.329 | 61.084 | 0.92x |
| Gate/up IQ2_S batch 16 uniform | 66.314 | 69.075 | 0.96x |
| Down Q4_K batch 4 uniform | 5.877 | 6.744 | 0.87x |
| Down Q4_K batch 16 uniform | 23.057 | 26.287 | 0.88x |
| Down Q5_K batch 4 uniform | 5.849 | 6.853 | 0.85x |
| Down IQ2_S batch 16 uniform | 27.102 | 29.160 | 0.93x |

The wider task kernel actually reduced Q3_K/IQ2_S VGPR allocation to 150/201, confirming that register pressure was not the cause. The regression comes from the larger LDS footprint, eight-wave workgroups, reduced workgroup residency/flexibility, and less favorable balance between packed-weight work and activation reuse. The `I=128, J=64` neighborhood is closed.

### G10: 256 threads with `I=64, J=64`

Status: invalid and reverted. Timings are non-results.

G10 changed only the workgroup from four to eight waves while retaining the 64-row output tile. The inherited MMQ wave mapping assigns output fragments by `threadIdx.y`, so the additional four waves mapped beyond the logical 64-row output tile and overlapped neighboring output work. The focused test appeared implausibly fast for that reason.

`tests/test_grouped_mmq.py::test_grouped_pair_production_row_tasks_match_dense` rejected the variant with 15,624 mismatched elements out of 262,144 and NaNs in the output. The artifact `/tmp/grouped_step10_256_threads.json` must not be used as performance evidence.

A correct eight-wave kernel would require a different decomposition such as split-K or duplicated output ownership and reduction. Those mechanisms add packed decode work or reduction overhead and are already outside the accepted design space. Keep 128 threads.

### G11: contiguous bounded activation-tail loads for down

Status: retained for the fixed two-block down body.

G11 observes that each valid partial Q8_1 row tile is still one contiguous integer span. The down helper now computes `(j_max + 1) * q8_block_ints` once and uses a single `l < valid_activation_ints` predicate, eliminating per-load row division, remainder, source-row reconstruction, and nested row bounds. Gate/up retains its G7/G8 tail code because applying the same source change there produced mixed noise-level results.

The retained 15-repeat artifact is:

```text
/tmp/grouped_step11b_down_contiguous_tails_15.json
```

| Point | Previous ms | G11 ms | Speedup |
|---|---:|---:|---:|
| Down Q4_K batch 1 sparse | 2.304 | 2.270 | 1.02x |
| Down Q4_K batch 4 boundary | 6.704 | 6.592 | 1.02x |
| Down Q4_K batch 16 uniform | 23.153 | 22.742 | 1.02x |
| Down Q5_K batch 4 boundary | 6.897 | 6.608 | 1.04x |
| Down IQ2_S batch 1 skewed | 3.836 | 3.659 | 1.05x |
| Down IQ2_S batch 1 sparse | 3.889 | 3.730 | 1.04x |
| Down IQ2_S batch 1 boundary | 3.988 | 3.764 | 1.06x |
| Down IQ2_S batch 4 skewed | 9.816 | 9.580 | 1.02x |
| Down IQ2_S batch 4 sparse | 9.525 | 9.336 | 1.02x |
| Down IQ2_S batch 4 boundary | 10.068 | 9.683 | 1.04x |
| Down IQ2_S batch 16 uniform | 26.832 | 26.478 | 1.01x |

The source cleanup increases down VGPR allocation to 229 for Q3_K, 244 for Q4_K, 248 for Q5_K, 239 for Q6_K, and 232 for IQ2_S, but every production kernel remains spill-free with zero private segment. The complete operator improves despite the higher allocation.

Production down IQ2_S batch-1 sparse and Q5_K batch-4 boundary remained bit-exact against dense MMQ. IQ2_S sparse improved to 3.745 ms in the correctness run but still trails 2.846 ms AITER, identifying the remaining final deficit.

### G12: partial IQ2_S dot-loop unroll

Status: rejected.

The exact Qwen down `(N,K,J)=(2048,512,64)` candidate is isolated in a separate HIP translation unit so the established Qwen device code object remains unchanged. The fresh automatic-unroll control is `/tmp/grouped_mmq_fwd_qwen_iq2s_auto_control.json`. Explicit `unroll 1` remains spill-free at 232 VGPRs and 46 SGPRs and is bitwise exact in `/tmp/grouped_mmq_fwd_qwen_iq2s_unroll1.json`. Its 12-point geometric speedup is `1.018x`, but most of the movement is batch-1 uniform (`13.46%`); the target nonuniform routes move only `-0.29%` to `+2.16%`.

Explicit `unroll 2` is rejected before timing. It reaches 256 VGPRs, 8 private bytes, and one VGPR spill, while growing the arithmetic symbol from about 61 KiB to 64 KiB. Factor 4 is also rejected before timing at 256 VGPRs, 352 private bytes, and 94 VGPR spills. Because factor 1 had several apparent movements above 1%, final acceptance used a sequential 25-repeat comparison against automatic unrolling in the same isolated translation unit, removing code-object placement as a confounder. The matched artifacts are `/tmp/grouped_mmq_fwd_qwen_iq2s_auto_isolated_25.json` and `/tmp/grouped_mmq_fwd_qwen_iq2s_unroll1_isolated_25.json`. Factor 1 is `0.41%` slower geometrically, including regressions of `1.33%` at batch-1 uniform and `1.29%` at batch-16 sparse; only batch-4 skew improves just over 1%. The initial apparent gain came from moving the kernel into a different code object, not loop rolling. The rolled IQ2_S helper is removed.

### G13: same-launch mixed-size tails

Status: complete; bounded two-size policies retained for small rows.

The literal IQ2_S variant keeps J64 for full tiles and selects J16/J32/J48 for the final partial expert tile inside the same kernel launch. It fails the resource gate before timing: the arithmetic symbol grows from about 61 KiB to 149 KiB, reaches 256 VGPRs, allocates 1,644 private bytes, and reports 455 VGPR spills. Compiling all four accumulator/decode bodies behind the device tail branch is not viable.

The narrower IQ2_S J64/J32 candidate passes at 231 VGPRs and 47 SGPRs with zero private bytes, zero spills, and no dynamic stack. Matched 25-repeat artifacts are `/tmp/grouped_mmq_fwd_qwen_iq2s_auto_isolated_post_tail_25.json` and `/tmp/grouped_mmq_fwd_qwen_iq2s_j64_j32_tail_25.json`. The candidate improves batch-1 nonuniform routes by `2.8-9.0%` and batch-1 uniform by `0.45%`. Applying it to all rows also regresses batch-4 uniform by `1.01%` and batch-16 uniform by `1.05%`. The retained static policy therefore uses J64/J32 only when `rows < 128 * num_groups`, selecting Qwen physical batch 1, and keeps the established J64 body otherwise.

The DeepSeek Q2_K J32/J16 candidate passes the resource gate at 213 VGPRs and 43 SGPRs with zero private bytes, zero spills, and no dynamic stack; its symbol is about 31 KiB. The initial nine-repeat artifact `/tmp/grouped_mmq_fwd_ds4_q2_unroll4_mixed_tail.json` is bitwise exact and improves all 12 routes over factor-4 J32 alone.

The matched 25-repeat artifacts are `/tmp/grouped_mmq_fwd_ds4_q2_unroll4_fixed_j32_25.json` and `/tmp/grouped_mmq_fwd_ds4_q2_unroll4_mixed_tail_25.json`. Mixed tails improve batch 1 by `5.1-15.7%`, but batch-4 uniform regresses `1.64%` and batch 16 ranges from `0.8%` slower to `0.1%` faster. The retained static policy therefore uses J32/J16 only when `rows < 64 * num_groups`, selecting DeepSeek physical batch 1, and factor-4 fixed J32 otherwise. This keeps the strong small-row gain without accepting the repeatable large-row regression.

### G14: pre-bundle last-version checkpoint

Status: benchmarked and committed as the comparison point for standalone kernel packaging. It is not an acceptable long-term artifact-layout contract.

This checkpoint preserves the established Qwen object first, keeps an unreachable factor-1 DeepSeek Q2_K instantiation as a layout anchor, and links the final factor-4 Q2_K kernels after the Qwen object. The small-row Qwen IQ2_S specialization also has its own translation unit. This arrangement makes the arithmetic experiments reproducible enough to hand off, but it intentionally records the translation-unit and link-order dependency that the standalone HSACO bundle must remove.

The exact benchmarked extension at `build/lib.linux-x86_64-cpython-314/torch_ggml_ops/_C.abi3.so` has SHA-256 `2b3e5f9e222ded44bd27e06408794d2bb96aacf9bf3034f4ed0738d09970fddc`.

Fresh nine-repeat source-of-record artifacts are:

```text
/tmp/grouped_mmq_fwd_ds4_last_version_baseline.json
/tmp/grouped_mmq_fwd_qwen_last_version_baseline.json
```

Their SHA-256 digests are `607bb3b0e6d7b2efd8a2851a55bc0787d7526bcb4bcfab75d849ba45c03b79fc` and `306f6bb3009604c76147d99fa8835fbf6fb4d42282b3423eac3ccb595dacd98f`. DeepSeek completes 27 points with 39/39 exact packed-reference checks and wins all 24 routed comparisons. Qwen completes its five-case, 60-point matrix with 84/84 exact checks and wins 54/60 AITER comparisons.

The adjacent full Qwen 25-repeat A/B artifacts are:

```text
/tmp/grouped_mmq_fwd_qwen_last_version_pre_control_25.json
/tmp/grouped_mmq_fwd_qwen_last_version_post_control_25.json
```

Relative to detached `efca259`, the checkpoint has `+0.54%` median and `+0.23%` geometric-mean latency movement. The retained small-row IQ2_S path improves batch-1 uniform/sparse by `10.8%/9.6%` in this packaged comparison. Conversely, unchanged Q3_K gate/up batch-4 points regress `1.2-4.3%`, and unchanged IQ2_S gate/up batch-4 points regress `0.9-2.5%`. Twenty-five of 60 individual points move by more than 1% despite unchanged arithmetic for most of them. This confirms that translation-unit separation, filename order, and the unreachable layout anchor are not valid performance contracts. The post-control artifact is the authoritative last-version Qwen baseline for the standalone module conversion; the packaged implementation must be compared against it in both production sequence and cold-instruction-cache conditions.

## Final retained evaluation

The complete retained 60-point artifact after G11 is:

```text
/tmp/grouped_mmq_fwd_final_full_v2.json
```

All 60 points match concatenated dense MMQ exactly with zero differing BF16 elements. Normalized RMSE against independently dequantized BF16 AITER remains within the original expected ranges:
- Q3_K: 0.00599-0.00613.
- IQ2_S: 0.00600-0.00611.
- Q4_K: 0.01120-0.01381.
- Q5_K: 0.01300-0.01650.

The packed operator wins 54 of 60 individual points against AITER. All gate/up Q3_K and IQ2_S points win, all down Q4_K and Q5_K points win, and all batch-16 down IQ2_S points win. The only remaining losses are nonuniform down IQ2_S at batch 1 and batch 4.

Representative final complete-operator results are:

| Point | Packed ms | AITER ms | Packed/AITER speedup |
|---|---:|---:|---:|
| Gate/up Q3_K batch 1 sparse | 5.307 | 14.215 | 2.68x |
| Gate/up Q3_K batch 4 boundary | 16.197 | 37.887 | 2.34x |
| Gate/up Q3_K batch 16 uniform | 56.601 | 89.336 | 1.58x |
| Gate/up IQ2_S batch 16 uniform | 66.798 | 89.102 | 1.33x |
| Down Q4_K batch 4 uniform | 5.644 | 7.878 | 1.40x |
| Down Q4_K batch 16 uniform | 22.816 | 45.213 | 1.98x |
| Down Q5_K batch 4 uniform | 5.761 | 7.820 | 1.36x |
| Down IQ2_S batch 16 uniform | 26.299 | 44.365 | 1.69x |
| Down IQ2_S batch 1 sparse | 3.736 | 2.851 | 0.76x |
| Down IQ2_S batch 4 boundary | 9.570 | 9.271 | 0.97x |

### Final checkpoint-weighted estimates

The following estimates multiply each complete public-operator latency by the checkpoint's projection-call count and by two for checkpointed forward recomputation during an optimizer step:

| Batch | Distribution | Original packed ms | Final packed ms | Final AITER ms | Final versus original | Final versus AITER |
|---:|---|---:|---:|---:|---:|---:|
| 1 | uniform | 1,433.8 | 473.5 | 1,336.4 | 3.03x | 2.82x |
| 1 | skewed | 1,429.5 | 739.1 | 1,510.4 | 1.93x | 2.04x |
| 1 | sparse | 1,260.9 | 700.0 | 1,370.0 | 1.80x | 1.96x |
| 1 | boundary | 1,439.5 | 733.1 | 1,510.8 | 1.96x | 2.06x |
| 4 | uniform | 4,247.4 | 1,750.0 | 3,384.0 | 2.43x | 1.93x |
| 4 | skewed | 4,546.0 | 2,051.5 | 3,792.7 | 2.22x | 1.85x |
| 4 | sparse | 4,494.2 | 2,012.8 | 3,649.8 | 2.23x | 1.81x |
| 4 | boundary | 4,625.6 | 2,053.5 | 3,761.4 | 2.25x | 1.83x |
| 16 | uniform | 16,707.8 | 6,899.6 | 10,725.2 | 2.42x | 1.55x |
| 16 | skewed | 17,021.7 | 7,067.5 | 11,370.6 | 2.41x | 1.61x |
| 16 | sparse | 17,044.8 | 7,051.0 | 11,084.8 | 2.42x | 1.57x |
| 16 | boundary | 17,015.1 | 7,086.6 | 10,818.2 | 2.40x | 1.53x |

### Final code-object resources

All retained production arithmetic kernels have zero private segment, zero VGPR spills, zero SGPR spills, and no dynamic stack:
- large-group gate/up row-task Q3_K/IQ2_S: 173/213 VGPRs and 54/50 SGPRs.
- batch-1 serial gate/up Q3_K/IQ2_S: 177/240 VGPRs and 44/56 SGPRs.
- down Q4_K/Q5_K/IQ2_S: 244/248/232 VGPRs and 46 SGPRs.
- descriptor setup: 13 VGPRs, 23 SGPRs, and 1 KiB LDS.

Dynamic LDS remains 28,928 bytes for Q4_K/Q5_K and 30,976 bytes for Q3_K/IQ2_S at `I=64, J=64`.

## Final bottleneck

Final sequential kernel traces are under:

```text
/tmp/rocprof_grouped_final_gate_b16_csv
/tmp/rocprof_grouped_final_v2_down_q4_b4
/tmp/rocprof_grouped_final_v2_down_iq2_b4_skewed
```

Ignoring benchmark input-generation kernels, the packed multiplication remains dominant:
- gate/up Q3_K batch 16 uniform: 50.875 ms across two row-task projections, 7.703 ms quantization, and 0.009 ms descriptor setup. Multiplication is approximately 87% of operator kernel time.
- down Q4_K batch 4 uniform: 5.555 ms multiplication and 0.522 ms quantization. Multiplication is approximately 91%.
- down IQ2_S batch 4 skewed: 9.576 ms multiplication and 0.945 ms quantization. Multiplication is approximately 91%.

The remaining deficit is not register spilling, launch setup, LDS banking, or generic metadata traversal. It is the packed IQ2_S arithmetic representation on irregular down groups:
- IQ2_S packed metadata interpretation, grid lookup, scale formation, and BF16 decoded-weight construction remain inside every output-tile workgroup.
- Down launches 32 output tiles per active expert, so the same expert tail and packed decode structure is repeated many times.
- Nonuniform groups require one partial `J=64` tile per expert. G11 removed row division and source reconstruction, but the WMMA tile still computes zero-padded rows and the packed weights still must be decoded for that partial tile.
- BF16 AITER has no packed decode cost and therefore retains a narrow advantage on the six short/nonuniform IQ2_S down points.

The tested same-representation alternatives do not offer more headroom:
- a complete decoded-weight LDS cache loses 27-40%.
- down row descriptors lose 2-14%.
- `J=128` and `I=128` lose materially.
- the post-write barrier is neutral.
- eight-wave `I=64` ownership is invalid without split-K-style reduction.
- split-K, larger LDS, a second stage, and persistent control state conflict with measured regressions and resource constraints.

No further local tile, scheduler, bounds, or synchronization change is worthwhile for the current packed representation. A higher ceiling requires a representation-level change, most plausibly a compact lossless IQ2_S decoded cache that is substantially smaller than BF16, cross-call decoded-weight reuse, or a transient project-owned decoded dense stage. Those are separate architectural projects and should be judged against the now-strong complete-operator baseline rather than added to this kernel as more control state.

## Completed plan and remaining work

| Phase | Outcome |
|---|---|
| Compile-time row and fixed-shape specialization | Completed in G1. Removed all production spills |
| Down complete decoded-weight LDS cache | Rejected in G2. 27-40% slower |
| `J=128` and wider output tiles | Rejected in G3/G9 |
| Exact full-row and bounded tail paths | Retained in G4 and refined for down in G11 |
| Fixed two-block down and gate/up pointer traversal | Retained in G6/G7 |
| Device row-task descriptors | Retained for large gate/up in G8. Rejected for down |
| Barrier removal | Neutral and reverted in G5 |
| Eight-wave `I=64` workgroup | Invalid output ownership in G10 |
| End-to-end validation | Complete: 60/60 exact against dense MMQ, 39 project tests, 9 integration tests |

### What remains

No additional local tile, scheduler, bounds, prefetch-toggle, LDS-cache, or synchronization sweep is planned for the current representation.

The only material remaining per-point deficit is nonuniform IQ2_S down at batch 1 and batch 4. Higher-ceiling follow-up projects are:
- a compact lossless IQ2_S decoded cache substantially smaller than BF16.
- cross-call decoded-weight reuse.
- a transient project-owned decoded dense stage.

Any future implementation must preserve device-resident routing metadata, batch-1 sparse launch behavior, exact grouped-versus-dense BF16 output, and complete-operator timing. Representation-level Qwen work still compares against `/tmp/grouped_mmq_fwd_final_full_v2.json`. The standalone bundle conversion compares against the G14 DeepSeek source-of-record matrix and the 25-repeat Qwen post-control artifact.

If grouped source changes resume, validation remains:

```bash
pytest -q tests/
python -m compileall -q bench
git diff --check
```

Then rerun the complete 60-point matrix and the local project test suite.

## Acceptance criteria

Status: satisfied for the retained production dispatch.

Every retained kernel must preserve exact grouped-versus-dense-MMQ BF16 output.

Production grouped arithmetic kernels require zero private segment and zero spills. The original Q4_K/Q5_K 512-520 bytes per thread was not acceptable. Every retained production kernel satisfies the zero-private-segment requirement.

Batch-1 sparse gate/up must retain a clear advantage over AITER.

Batch-4 down and batch-16 gate/up are the primary performance gates. A local improvement that regresses the model-weighted optimizer-step estimate should be rejected.

AITER remains the reference, not the ceiling. The final target is the best end-to-end packed grouped latency that preserves the production contract.

## Non-priorities and rejected directions

- Do not replace AITER GMM with `torch.matmul` as the performance reference.
- Do not link packed kernels against AITER or hipBLASLt.
- Do not construct full logical BF16 expert matrices in the packed production path.
- Do not introduce CPU expert descriptors or metadata synchronization.
- Do not prioritize another quantizer rewrite: production spills are already removed, and final multiplication still accounts for approximately 87-91% of retained kernel time.
- Do not use runtime environment variables or online autotuning for dispatch.
- Do not assume raw int8 WMMA throughput will overcome decode, LDS, scratch, or scheduling overhead.
- Do not treat PC sampling on these short kernels as quantitative latency data.
- Do not use shipped TensileLite grouped YAML choices or their displayed grouped GFLOP/s as a performance bound. Use them only as generator and interface evidence.
- Do not copy TensileLite's host-built grouped user-argument setup or per-workgroup GEMM search into the production routing path. Build compact metadata on the device and index it directly.
- Do not retry a wholesale direct-to-VGPR conversion, both-operands direct-to-VGPR, a two-LDS pipeline, GSU, split-K, or grouped Stream-K for the current packed representation.

## Final status

G1 completed the compile-time `J=64` and fixed-production-shape work and removed all production grouped-forward spills.

The complete fixed-`K=512` decoded-weight LDS cache was rejected in G2 because its LDS residency cost overwhelmed decode reuse.

G3 rejected fixed-shape `J=128`. `J=64` remains the production row tile. G4 then removed full-row address decomposition and moved every focused production class ahead of AITER.

G5 showed that removing the post-write barrier is neutral. G6 retained a compile-time two-block down schedule, and G7 retained pointer-increment traversal for gate/up.

G8 retained device row-task descriptors for large gate/up groups and rejected them for down. Batch-1 sparse routing keeps the serial G7 path.

G9 rejected the final high-value wider-output geometry despite zero spills. G10 confirmed that 256 threads cannot be applied to `I=64` without redesigning output ownership or introducing split-K-style reduction. G11 retained contiguous bounded activation-tail loads for down.

The current tile, scheduler, addressing, K-loop, synchronization, cache, and tail-control neighborhoods are exhausted. Further worthwhile work must change IQ2_S packed decode or representation rather than add geometry or control state.
