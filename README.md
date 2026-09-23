# torch-ggml-ops

This package provides PyTorch bindings of GGML operators with quantized weights. Currently it provides dense MMQ, grouped MMQ, and grouped MMQ pair for fused gate-up. All ops support gradients on inputs. See https://github.com/woct0rdho/transformers5-qwen3.5-recipe for example usage.

Currently all ops support bf16 input and output activations. and quant types in `IQ2_XXS, IQ2_S, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0`.

The kernels are tested on Strix Halo (gfx1151), and they should also work on RDNA3 GPUs. Dense MMQ also has an NVIDIA CUDA backend, see below.

The kernel parameters are tuned for typical input and weight shapes of Qwen3.5-35B-A3B and DeepSeek-V4-Flash. An autotune system is possible but not yet implemented.

The forward kernels are modified from llama.cpp . The GGUF format is only optimized for forward where each matmul tile needs only one scale, but when doing backward each matmul tile crosses multiple quantized blocks and requires multiple scales. So we do not use int8 MMA, but dequantize each tile into bf16 and run bf16 MMA. This is still faster and saves most of the VRAM compared to dequantizing the whole weights into bf16.

The backward kernels are implemented with CK Tile, and it should be straightforward to port them to CuTe on Nvidia GPUs.

## Installation

The C++ extension uses Python 3.10 ABI3 and libtorch 2.10 stable ABI. It requires PyTorch >= 2.10 .

Extract source code from the llama.cpp repo:

```bash
python3 tools/generate_vendor.py --llama-cpp=/path/to/llama.cpp/
```

Build the package with ROCm and PyTorch in the current environment, and install in place:

```bash
pip install --no-build-isolation --no-deps -e .
```

Currently my forked [transformers with GGUF quantizer](https://github.com/woct0rdho/transformers/tree/gguf) is required to run the tests. Install it, and download the example GGUF models:
- https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF/blob/main/Qwen3.6-35B-A3B-APEX-I-Mini.gguf
- https://huggingface.co/antirez/deepseek-v4-gguf/blob/main/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf

Then run the tests:

```bash
pytest tests/
```

## NVIDIA CUDA backend

`csrc/mmq_cuda.cu` + `csrc/cuda/mmq_mma.cuh` implement the dense `mmq` operators (forward and input gradient) for NVIDIA GPUs with BF16 tensor cores (sm_80+). It was developed and tested on an RTX 3090 (sm_86).

Design: one tiled kernel serves both directions. Each CTA decodes only the GGUF tile it is about to multiply into shared memory as BF16 and runs `mma.sync.m16n8k16` BF16 MMA with FP32 accumulation (128x128x32 tiles, 8 warps, `cp.async` double buffering for activations). The backward reads the same weight tile through `ldmatrix.trans`. This is the scheme the HIP backward kernels use; on CUDA the forward uses it too, so activations are never quantized to Q8_1 and the forward workspace is empty (`runtime_contract.quant_workspace_elements` returns 0). There is no exact-key deployment table: any shape whose input width is a multiple of 256 is served. Grouped / paired / fixed-grouped (MoE) operators are registered with the same schemas but raise "not implemented by the CUDA backend".

Quant types: `Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, IQ3_S, IQ4_NL, IQ4_XS` (`torch_ggml_ops.DENSE_MMQ_QUANT_TYPES`). Other types raise a clear error.

Build (CUDA toolkit matching the torch CUDA major version, e.g. torch 2.14+cu130 with nvcc 13.x):

```bash
TORCH_CUDA_ARCH_LIST=8.6 pip install --no-build-isolation --no-deps -e .
pytest tests/test_cuda_dense_mmq.py
```

The CUDA tests take real tensors from a GGUF checkpoint (`GGUF_CUDA_MMQ_TEST_MODEL`, a glob; default the Qwen3.8-27B UD-Q4_K_XL file in the Hugging Face cache) and compare against the `gguf` package's NumPy dequantizer with exact FP32 math. Criteria and their measured justification are in the test module docstring: every output element within one BF16 ulp of the exact product of the BF16-rounded weight (plus 1e-4 x RMS slack for cancellation), NRMSE < 2e-3 against that reference (measured floor 1.63e-3..1.69e-3, the output rounding itself) and < 5e-3 against the unrounded FP32 weight. A negative control corrupts one block scale and requires the criteria to fail.

`bench/benchmark_cuda_dense_mmq.py` compares against cuBLAS on a pre-dequantized BF16 weight.

## TODO

- fp16
- other quant types
- other GPUs
- autotune
