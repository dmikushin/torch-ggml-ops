// Load final DeepSeek Q2_K kernels after the established Qwen device code object.
#ifdef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_OPERATORS__
#endif
#ifdef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_CONVERSIONS__
#endif

#define MMQ_USE_ROLLED_Q2_K
#define MMQ_USE_MIXED_Q2_K_TAILS
#include "mmq_core.cuh"

#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

__launch_bounds__(MMQ_NTHREADS, 2)
static __global__ void deepseek_q2_fixed_bf16_kernel(
        const char * __restrict__ weights,
        const int * __restrict__ activations,
        __hip_bfloat16 * __restrict__ dst,
        const int64_t * __restrict__ expert_indices,
        const int32_t * __restrict__ expert_offsets,
        int num_experts,
        int nrows_activation,
        int64_t bytes_per_expert) {
    constexpr ggml_type type = GGML_TYPE_Q2_K;
    constexpr int J = MMQ_J_TINY;
    constexpr int nrows_weight = 4096;
    constexpr int blocks_per_weight_row = 8;
    const int tile_i = blockIdx.x;
    const int group = blockIdx.y;
    const int row_begin = group == 0 ? 0 : expert_offsets[group - 1];
    const int row_end = expert_offsets[group];
    const int64_t expert = expert_indices[group];
    if (
        expert < 0 || expert >= num_experts || row_begin < 0 ||
        row_end <= row_begin || row_end > nrows_activation
    ) {
        return;
    }

    const char * expert_weights = weights + expert * bytes_per_expert;
    extern __shared__ int shared[];
    int * tile_y = shared + J;
    int * tile_x = tile_y + GGML_PAD(J * MMQ_TILE_Y_K, MMQ_NTHREADS);

    int row_start = row_begin;
    for (; row_start + J <= row_end; row_start += J) {
        grouped_mmq_row_tile<type, J, nrows_weight, blocks_per_weight_row, true>(
            expert_weights,
            activations,
            dst,
            tile_x,
            tile_y,
            tile_i,
            row_start,
            row_end,
            nrows_weight,
            nrows_activation,
            blocks_per_weight_row);
    }
    if (row_start < row_end) {
        grouped_mmq_row_tile<type, J, nrows_weight, blocks_per_weight_row, false>(
            expert_weights,
            activations,
            dst,
            tile_x,
            tile_y,
            tile_i,
            row_start,
            row_end,
            nrows_weight,
            nrows_activation,
            blocks_per_weight_row);
    }
}

void launch_q2_mixed_projection(
        const char * packed,
        const int * activations,
        __hip_bfloat16 * output,
        const int64_t * expert_indices,
        const int32_t * expert_offsets,
        int num_experts,
        int num_groups,
        int rows,
        int64_t bytes_per_expert,
        hipStream_t stream) {
    constexpr ggml_type type = GGML_TYPE_Q2_K;
    constexpr int J = MMQ_J_TINY;
    constexpr int nrows_weight = 4096;
    constexpr int blocks_per_weight_row = 8;
    const dim3 grid(nrows_weight / MMQ_I, num_groups, 1);
    const dim3 block(WARP_SIZE, MMQ_NWARPS, 1);
    const int shared_ints = J + GGML_PAD(J * MMQ_TILE_Y_K, MMQ_NTHREADS) +
        MMQ_I * mmq_sram_stride(type);
    grouped_mmq_bf16_kernel<type, J, nrows_weight, blocks_per_weight_row>
        <<<grid, block, shared_ints * sizeof(int), stream>>>(
            packed,
            activations,
            output,
            expert_indices,
            expert_offsets,
            num_experts,
            nrows_weight,
            rows,
            blocks_per_weight_row,
            bytes_per_expert);
}

void launch_q2_fixed_projection(
        const char * packed,
        const int * activations,
        __hip_bfloat16 * output,
        const int64_t * expert_indices,
        const int32_t * expert_offsets,
        int num_experts,
        int num_groups,
        int rows,
        int64_t bytes_per_expert,
        hipStream_t stream) {
    constexpr int J = MMQ_J_TINY;
    const dim3 grid(4096 / MMQ_I, num_groups, 1);
    const dim3 block(WARP_SIZE, MMQ_NWARPS, 1);
    const int shared_ints = J + GGML_PAD(J * MMQ_TILE_Y_K, MMQ_NTHREADS) +
        MMQ_I * mmq_sram_stride(GGML_TYPE_Q2_K);
    deepseek_q2_fixed_bf16_kernel
        <<<grid, block, shared_ints * sizeof(int), stream>>>(
            packed,
            activations,
            output,
            expert_indices,
            expert_offsets,
            num_experts,
            rows,
            bytes_per_expert);
}

} // namespace

void launch_deepseek_q2_final_projection(
        const char * packed,
        const int * activations,
        __hip_bfloat16 * output,
        const int64_t * expert_indices,
        const int32_t * expert_offsets,
        int num_experts,
        int num_groups,
        int rows,
        int64_t bytes_per_expert,
        hipStream_t stream) {
    if (rows < num_groups * MMQ_J_SMALL) {
        launch_q2_mixed_projection(
            packed,
            activations,
            output,
            expert_indices,
            expert_offsets,
            num_experts,
            num_groups,
            rows,
            bytes_per_expert,
            stream);
    } else {
        launch_q2_fixed_projection(
            packed,
            activations,
            output,
            expert_indices,
            expert_offsets,
            num_experts,
            num_groups,
            rows,
            bytes_per_expert,
            stream);
    }
}
