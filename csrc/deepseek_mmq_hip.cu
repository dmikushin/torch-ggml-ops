// Keep DeepSeek-only kernel instantiations out of the Qwen code object.
#ifdef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_OPERATORS__
#endif
#ifdef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_CONVERSIONS__
#endif

#define MMQ_USE_ROLLED_Q2_K
#include "mmq_core.cuh"

#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

template <ggml_type type, int J, int fixed_nrows_weight, int fixed_blocks_per_weight_row>
void launch_projection(
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
    const dim3 grid((fixed_nrows_weight + MMQ_I - 1) / MMQ_I, num_groups, 1);
    const dim3 block(WARP_SIZE, MMQ_NWARPS, 1);
    const int shared_ints = J + GGML_PAD(J * MMQ_TILE_Y_K, MMQ_NTHREADS) +
        MMQ_I * mmq_sram_stride(type);
    grouped_mmq_bf16_kernel<type, J, fixed_nrows_weight, fixed_blocks_per_weight_row>
        <<<grid, block, shared_ints * sizeof(int), stream>>>(
            packed,
            activations,
            output,
            expert_indices,
            expert_offsets,
            num_experts,
            fixed_nrows_weight,
            rows,
            fixed_blocks_per_weight_row,
            bytes_per_expert);
}

} // namespace

void launch_deepseek_grouped_projection(
        ggml_type type,
        const char * packed,
        const int * activations,
        __hip_bfloat16 * output,
        const int64_t * expert_indices,
        const int32_t * expert_offsets,
        int num_experts,
        int num_groups,
        int rows,
        int in_features,
        int out_features,
        int64_t bytes_per_expert,
        hipStream_t stream) {
    if (type == GGML_TYPE_IQ2_XXS && out_features == 2048 && in_features == 4096) {
        if (rows >= num_groups * 512) {
            launch_projection<GGML_TYPE_IQ2_XXS, MMQ_J_MEDIUM, 2048, 16>(
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
            launch_projection<GGML_TYPE_IQ2_XXS, MMQ_J_SMALL, 2048, 16>(
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
    } else if (type == GGML_TYPE_Q2_K && out_features == 4096 && in_features == 2048) {
        launch_projection<GGML_TYPE_Q2_K, MMQ_J_TINY, 4096, 8>(
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
