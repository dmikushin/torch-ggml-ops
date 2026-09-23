// NVIDIA CUDA backend of the torch_ggml_ops operator library.
//
// Operator schemas are identical to the HIP backend (csrc/mmq_hip.cu). Dense
// MMQ forward and input-gradient are implemented by csrc/cuda/mmq_mma.cuh for
// Q4_K, Q5_K, Q6_K and Q8_0 on sm_80+. Unlike the HIP backend there is no
// exact-shape deployment table: every shape that passes validation is served
// by the same tiled kernel. Grouped (MoE) operators are registered with the
// same schemas but are not implemented on CUDA and fail with a clear error.

#include "vendor/llama_cpp/common.cuh"
#include "cuda/mmq_mma.cuh"

#include <cuda_runtime.h>
#include <Python.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <cstdint>
#include <limits>

namespace {

#include "mmq_dense_validation.cuh"

void check_launch(cudaError_t error, const char * what) {
    STD_TORCH_CHECK(error == cudaSuccess, what, " failed: ", cudaGetErrorString(error));
}

template <ggml_type type, bool backward>
void launch_mma(
        const void * a,
        const void * w,
        void * c,
        int rows,
        int contraction,
        int columns,
        int w_rows,
        int64_t w_row_bytes,
        cudaStream_t stream) {
    using cfg = torch_ggml_ops::cuda_mma::tile_config<backward>;
    const dim3 grid(
        static_cast<unsigned>((columns + cfg::BN - 1) / cfg::BN),
        static_cast<unsigned>((rows + cfg::BM - 1) / cfg::BM));
    STD_TORCH_CHECK(grid.y <= 65535u, "MMQ row count exceeds the CUDA grid limit");
    const auto * a_ptr = static_cast<const __nv_bfloat16 *>(a);
    const auto * w_ptr = static_cast<const uint8_t *>(w);
    auto * c_ptr = static_cast<__nv_bfloat16 *>(c);
    static_assert(cfg::SMEM_BYTES <= 48 * 1024, "tile needs opt-in dynamic shared memory");
    auto run = [&](auto kernel) {
        kernel<<<grid, cfg::THREADS, cfg::SMEM_BYTES, stream>>>(
            a_ptr, w_ptr, c_ptr, rows, contraction, columns, w_rows, w_row_bytes);
        check_launch(cudaGetLastError(), "MMQ kernel launch");
    };
    // 16-byte cp.async needs every A row to start on a 16-byte boundary.
    if (contraction % 8 == 0) {
        run(torch_ggml_ops::cuda_mma::mmq_mma_kernel<type, backward, true>);
    } else {
        run(torch_ggml_ops::cuda_mma::mmq_mma_kernel<type, backward, false>);
    }
}

template <bool backward>
void dispatch_mma(
        int64_t quant_type,
        const void * a,
        const void * w,
        void * c,
        int rows,
        int contraction,
        int columns,
        int w_rows,
        int64_t w_row_bytes,
        cudaStream_t stream) {
    switch (quant_type) {
        case GGML_TYPE_Q4_K:
            return launch_mma<GGML_TYPE_Q4_K, backward>(a, w, c, rows, contraction, columns, w_rows, w_row_bytes, stream);
        case GGML_TYPE_Q5_K:
            return launch_mma<GGML_TYPE_Q5_K, backward>(a, w, c, rows, contraction, columns, w_rows, w_row_bytes, stream);
        case GGML_TYPE_Q6_K:
            return launch_mma<GGML_TYPE_Q6_K, backward>(a, w, c, rows, contraction, columns, w_rows, w_row_bytes, stream);
        case GGML_TYPE_Q8_0:
            return launch_mma<GGML_TYPE_Q8_0, backward>(a, w, c, rows, contraction, columns, w_rows, w_row_bytes, stream);
        default:
            STD_TORCH_CHECK(
                false,
                "quant_type ", quant_type,
                " is not implemented by the CUDA MMQ backend (supported: Q4_K=12, Q5_K=13, Q6_K=14, Q8_0=8)");
    }
}

void mmq_launch_cuda(
        const Tensor & input,
        const Tensor & packed_weight,
        int64_t quant_type,
        int64_t out_features,
        Tensor output,
        Tensor workspace) {
    const DenseMMQShape shape =
        validate_dense_mmq(input, packed_weight, quant_type, out_features);
    validate_explicit_buffer(
        output,
        input,
        ScalarType::BFloat16,
        static_cast<int64_t>(shape.rows) * shape.out_features,
        "output");
    validate_replaced_final_dimension(output, input, shape.out_features, "output");
    validate_explicit_vector(
        workspace, input, ScalarType::Byte, shape.workspace_bytes, "workspace");
    torch::stable::accelerator::DeviceGuard guard(input.get_device_index());
    dispatch_mma<false>(
        quant_type,
        input.const_data_ptr(),
        packed_weight.const_data_ptr(),
        output.mutable_data_ptr(),
        shape.rows,
        shape.in_features,
        shape.out_features,
        shape.out_features,
        packed_row_bytes(quant_type, shape.in_features),
        current_stream(input));
}

void mmq_grad_input_launch_cuda(
        const Tensor & grad_output,
        const Tensor & packed_weight,
        int64_t quant_type,
        int64_t in_features,
        Tensor grad_input) {
    const DenseMMQShape shape = validate_dense_mmq_backward(
        grad_output, packed_weight, quant_type, in_features);
    validate_explicit_buffer(
        grad_input,
        grad_output,
        ScalarType::BFloat16,
        static_cast<int64_t>(shape.rows) * shape.in_features,
        "grad_input");
    validate_replaced_final_dimension(
        grad_input, grad_output, shape.in_features, "grad_input");
    torch::stable::accelerator::DeviceGuard guard(grad_output.get_device_index());
    dispatch_mma<true>(
        quant_type,
        grad_output.const_data_ptr(),
        packed_weight.const_data_ptr(),
        grad_input.mutable_data_ptr(),
        shape.rows,
        shape.out_features,
        shape.in_features,
        shape.out_features,
        packed_row_bytes(quant_type, shape.in_features),
        current_stream(grad_output));
}

[[noreturn]] void grouped_unsupported(const char * op) {
    STD_TORCH_CHECK(false, op, " is not implemented by the CUDA backend (only dense MMQ is)");
    __builtin_unreachable();
}

void fixed_grouped_mmq_launch_cuda(const Tensor &, const Tensor &, Tensor, Tensor) {
    grouped_unsupported("fixed_grouped_mmq");
}
void fixed_grouped_mmq_grad_input_launch_cuda(const Tensor &, const Tensor &, Tensor) {
    grouped_unsupported("fixed_grouped_mmq backward");
}
void grouped_mmq_launch_cuda(
        const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, Tensor, Tensor) {
    grouped_unsupported("grouped_mmq");
}
void grouped_mmq_grad_input_launch_cuda(
        const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, Tensor) {
    grouped_unsupported("grouped_mmq backward");
}
void grouped_mmq_pair_launch_cuda(
        const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t,
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
    grouped_unsupported("grouped_mmq_pair");
}
void grouped_mmq_pair_grad_input_launch_cuda(
        const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &,
        int64_t, int64_t, Tensor) {
    grouped_unsupported("grouped_mmq_pair backward");
}

} // namespace

STABLE_TORCH_LIBRARY(torch_ggml_ops, m) {
    m.def("_mmq_launch(Tensor input, Tensor packed_weight, int quant_type, int out_features, "
          "Tensor(a!) output, Tensor(b!) workspace) -> ()");
    m.def("_mmq_grad_input_launch(Tensor grad_output, Tensor packed_weight, int quant_type, int in_features, "
          "Tensor(a!) grad_input) -> ()");
    m.def("_fixed_grouped_mmq_launch(Tensor input, Tensor packed_weight, Tensor(a!) output, "
          "Tensor(b!) workspace) -> ()");
    m.def("_fixed_grouped_mmq_grad_input_launch(Tensor grad_output, Tensor packed_weight, "
          "Tensor(a!) grad_input) -> ()");
    m.def("_grouped_mmq_launch(Tensor input, Tensor packed_weight, Tensor expert_indices, "
          "Tensor expert_offsets, int quant_type, int out_features, Tensor(a!) output, "
          "Tensor(b!) workspace) -> ()");
    m.def("_grouped_mmq_grad_input_launch(Tensor grad_output, Tensor packed_weight, "
          "Tensor expert_indices, Tensor expert_offsets, int quant_type, int in_features, "
          "Tensor(a!) grad_input) -> ()");
    m.def("_grouped_mmq_pair_launch(Tensor input, Tensor first_packed_weight, "
          "Tensor second_packed_weight, Tensor expert_indices, Tensor expert_offsets, "
          "int quant_type, int out_features, Tensor(a!) first_output, "
          "Tensor(b!) second_output, Tensor(c!) workspace, Tensor(d!) task_count, "
          "Tensor(e!) task_experts, Tensor(f!) task_row_starts, Tensor(g!) task_row_ends) -> ()");
    m.def("_grouped_mmq_pair_grad_input_launch(Tensor first_grad_output, Tensor second_grad_output, "
          "Tensor first_packed_weight, Tensor second_packed_weight, Tensor expert_indices, "
          "Tensor expert_offsets, int quant_type, int in_features, Tensor(a!) grad_input) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(torch_ggml_ops, CUDA, m) {
    m.impl("_mmq_launch", TORCH_BOX(&mmq_launch_cuda));
    m.impl("_mmq_grad_input_launch", TORCH_BOX(&mmq_grad_input_launch_cuda));
    m.impl("_fixed_grouped_mmq_launch", TORCH_BOX(&fixed_grouped_mmq_launch_cuda));
    m.impl("_fixed_grouped_mmq_grad_input_launch", TORCH_BOX(&fixed_grouped_mmq_grad_input_launch_cuda));
    m.impl("_grouped_mmq_launch", TORCH_BOX(&grouped_mmq_launch_cuda));
    m.impl("_grouped_mmq_grad_input_launch", TORCH_BOX(&grouped_mmq_grad_input_launch_cuda));
    m.impl("_grouped_mmq_pair_launch", TORCH_BOX(&grouped_mmq_pair_launch_cuda));
    m.impl("_grouped_mmq_pair_grad_input_launch", TORCH_BOX(&grouped_mmq_pair_grad_input_launch_cuda));
}

extern "C" PyObject * PyInit__C(void) {
    static PyModuleDef module = {
        PyModuleDef_HEAD_INIT,
        "_C",
        nullptr,
        -1,
        nullptr,
    };
    return PyModule_Create(&module);
}
