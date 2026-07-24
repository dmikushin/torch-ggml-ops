// Keep the small-row Qwen IQ2_S specialization out of the established Qwen code object.
#ifdef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_OPERATORS__
#endif
#ifdef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_CONVERSIONS__
#endif

#define MMQ_USE_MIXED_IQ2_S_TAILS
#include "mmq_core.cuh"

#include <hip/hip_runtime.h>

#include <cstdint>

void launch_qwen_iq2_s_down_projection(
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
    constexpr ggml_type type = GGML_TYPE_IQ2_S;
    constexpr int J = MMQ_J_SMALL;
    constexpr int nrows_weight = 2048;
    constexpr int blocks_per_weight_row = 2;
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
