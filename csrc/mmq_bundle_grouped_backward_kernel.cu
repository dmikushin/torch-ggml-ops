// Canonical wrappers for independently packaged grouped MMQ backward kernels.
#ifdef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_OPERATORS__
#endif
#ifdef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_CONVERSIONS__
#endif

#include "ck/grouped_mmq_backward.cuh"

#ifndef MMQ_BUNDLE_KERNEL_SYMBOL
#error "MMQ_BUNDLE_KERNEL_SYMBOL must name the exported kernel"
#endif

#define MMQ_GROUPED_SINGLE_ARGUMENTS \
        const __hip_bfloat16 * __restrict__ grad_output, \
        const char * __restrict__ packed_weight, \
        __hip_bfloat16 * __restrict__ grad_input, \
        const int64_t * __restrict__ expert_indices, \
        const int32_t * __restrict__ expert_offsets, \
        int num_experts, \
        int rows, \
        int64_t bytes_per_expert

#define MMQ_GROUPED_SINGLE_VALUES \
        grad_output, packed_weight, grad_input, expert_indices, expert_offsets, \
        num_experts, rows, bytes_per_expert

#define MMQ_GROUPED_PAIR_ARGUMENTS \
        const __hip_bfloat16 * __restrict__ first_grad_output, \
        const __hip_bfloat16 * __restrict__ second_grad_output, \
        const char * __restrict__ first_packed_weight, \
        const char * __restrict__ second_packed_weight, \
        __hip_bfloat16 * __restrict__ grad_input, \
        const int64_t * __restrict__ expert_indices, \
        const int32_t * __restrict__ expert_offsets, \
        int num_experts, \
        int rows, \
        int64_t bytes_per_expert

#define MMQ_GROUPED_PAIR_VALUES \
        first_grad_output, second_grad_output, first_packed_weight, \
        second_packed_weight, grad_input, expert_indices, expert_offsets, \
        num_experts, rows, bytes_per_expert

#define MMQ_GROUPED_ROW_TASK_ARGUMENTS \
        const __hip_bfloat16 * __restrict__ grad_output, \
        const char * __restrict__ packed_weight, \
        __hip_bfloat16 * __restrict__ grad_input, \
        const int32_t * __restrict__ task_count, \
        const int32_t * __restrict__ task_experts, \
        const int32_t * __restrict__ task_row_starts, \
        const int32_t * __restrict__ task_row_ends, \
        int64_t bytes_per_expert

#define MMQ_GROUPED_ROW_TASK_VALUES \
        grad_output, packed_weight, grad_input, task_count, task_experts, \
        task_row_starts, task_row_ends, bytes_per_expert

#define MMQ_GROUPED_DEEPSEEK_SINGLE_ARGUMENTS \
        const __hip_bfloat16 * __restrict__ grad_output, \
        const char * __restrict__ packed_weight, \
        __hip_bfloat16 * __restrict__ grad_input, \
        const int64_t * __restrict__ expert_indices, \
        const int32_t * __restrict__ expert_offsets, \
        int num_experts, \
        int rows, \
        int64_t bytes_per_expert

#define MMQ_GROUPED_DEEPSEEK_SINGLE_VALUES \
        grad_output, packed_weight, grad_input, expert_indices, expert_offsets, \
        num_experts, rows, bytes_per_expert

#define MMQ_GROUPED_DEEPSEEK_PAIR_ARGUMENTS \
        const __hip_bfloat16 * __restrict__ first_grad_output, \
        const __hip_bfloat16 * __restrict__ second_grad_output, \
        const char * __restrict__ first_packed_weight, \
        const char * __restrict__ second_packed_weight, \
        __hip_bfloat16 * __restrict__ grad_input, \
        const int64_t * __restrict__ expert_indices, \
        const int32_t * __restrict__ expert_offsets, \
        int num_experts, \
        int rows, \
        int64_t bytes_per_expert

#define MMQ_GROUPED_DEEPSEEK_PAIR_VALUES \
        first_grad_output, second_grad_output, first_packed_weight, \
        second_packed_weight, grad_input, expert_indices, expert_offsets, \
        num_experts, rows, bytes_per_expert

#define MMQ_FIXED_GROUPED_BWD_ARGUMENTS \
        const __hip_bfloat16 * __restrict__ grad_output, \
        const char * __restrict__ packed_weight, \
        __hip_bfloat16 * __restrict__ grad_input, \
        int tokens, \
        int out_features, \
        int64_t bytes_per_group

#define MMQ_FIXED_GROUPED_BWD_VALUES \
        grad_output, packed_weight, grad_input, tokens, bytes_per_group

#if defined(MMQ_BUNDLE_GROUPED_BWD_GENERIC_SINGLE)

#ifndef MMQ_BUNDLE_QUANT_TYPE
#error "generic grouped backward kernels require MMQ_BUNDLE_QUANT_TYPE"
#endif

extern "C" __launch_bounds__(torch_ggml_ops::ck::GROUPED_BACKWARD_THREADS, 1) __global__
void MMQ_BUNDLE_KERNEL_SYMBOL(
        const __hip_bfloat16 * __restrict__ grad_output,
        const char * __restrict__ packed_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        const int64_t * __restrict__ expert_indices,
        const int32_t * __restrict__ expert_offsets,
        int num_experts,
        int rows,
        int out_features,
        int in_features,
        int blocks_per_weight_row,
        int64_t bytes_per_expert) {
    torch_ggml_ops::ck::grouped_mmq_grad_input_body<
        static_cast<ggml_type>(MMQ_BUNDLE_QUANT_TYPE)>(
            grad_output,
            packed_weight,
            grad_input,
            expert_indices,
            expert_offsets,
            num_experts,
            rows,
            out_features,
            in_features,
            blocks_per_weight_row,
            bytes_per_expert);
}

#elif defined(MMQ_BUNDLE_GROUPED_BWD_GENERIC_PAIR)

#ifndef MMQ_BUNDLE_QUANT_TYPE
#error "generic grouped pair backward kernels require MMQ_BUNDLE_QUANT_TYPE"
#endif

extern "C" __launch_bounds__(torch_ggml_ops::ck::GROUPED_BACKWARD_THREADS, 1) __global__
void MMQ_BUNDLE_KERNEL_SYMBOL(
        const __hip_bfloat16 * __restrict__ first_grad_output,
        const __hip_bfloat16 * __restrict__ second_grad_output,
        const char * __restrict__ first_packed_weight,
        const char * __restrict__ second_packed_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        const int64_t * __restrict__ expert_indices,
        const int32_t * __restrict__ expert_offsets,
        int num_experts,
        int rows,
        int out_features,
        int in_features,
        int blocks_per_weight_row,
        int64_t bytes_per_expert) {
    torch_ggml_ops::ck::grouped_mmq_pair_grad_input_body<
        static_cast<ggml_type>(MMQ_BUNDLE_QUANT_TYPE)>(
            first_grad_output,
            second_grad_output,
            first_packed_weight,
            second_packed_weight,
            grad_input,
            expert_indices,
            expert_offsets,
            num_experts,
            rows,
            out_features,
            in_features,
            blocks_per_weight_row,
            bytes_per_expert);
}

#elif defined(MMQ_BUNDLE_GROUPED_BWD_Q4_SINGLE_M64)
extern "C" __launch_bounds__(torch_ggml_ops::ck::BACKWARD_THREADS, 2) __global__
void MMQ_BUNDLE_KERNEL_SYMBOL(MMQ_GROUPED_SINGLE_ARGUMENTS) {
    torch_ggml_ops::ck::grouped_mmq_grad_input_q4_small_body(
        MMQ_GROUPED_SINGLE_VALUES);
}

#elif defined(MMQ_BUNDLE_GROUPED_BWD_Q4_SINGLE_M128)
extern "C" __launch_bounds__(torch_ggml_ops::ck::BACKWARD_THREADS, 2) __global__
void MMQ_BUNDLE_KERNEL_SYMBOL(MMQ_GROUPED_SINGLE_ARGUMENTS) {
    torch_ggml_ops::ck::grouped_mmq_grad_input_q4_small_s2_body(
        MMQ_GROUPED_SINGLE_VALUES);
}

#elif defined(MMQ_BUNDLE_GROUPED_BWD_Q3_PAIR_M64)
extern "C" __launch_bounds__(torch_ggml_ops::ck::BACKWARD_THREADS, 2) __global__
void MMQ_BUNDLE_KERNEL_SYMBOL(MMQ_GROUPED_PAIR_ARGUMENTS) {
    torch_ggml_ops::ck::grouped_mmq_pair_grad_input_q3_small_body(
        MMQ_GROUPED_PAIR_VALUES);
}

#elif defined(MMQ_BUNDLE_GROUPED_BWD_Q3_PAIR_M128)
extern "C" __launch_bounds__(torch_ggml_ops::ck::BACKWARD_THREADS, 2) __global__
void MMQ_BUNDLE_KERNEL_SYMBOL(MMQ_GROUPED_PAIR_ARGUMENTS) {
    torch_ggml_ops::ck::grouped_mmq_pair_grad_input_q3_n64_large_body(
        MMQ_GROUPED_PAIR_VALUES);
}

#elif defined(MMQ_BUNDLE_GROUPED_BWD_Q5_SINGLE_M64)
extern "C" __launch_bounds__(torch_ggml_ops::ck::BACKWARD_THREADS, 2) __global__
void MMQ_BUNDLE_KERNEL_SYMBOL(MMQ_GROUPED_SINGLE_ARGUMENTS) {
    torch_ggml_ops::ck::grouped_mmq_grad_input_q5_small_body(
        MMQ_GROUPED_SINGLE_VALUES);
}

#elif defined(MMQ_BUNDLE_GROUPED_BWD_IQ2_S_SINGLE_M64)
extern "C" __launch_bounds__(torch_ggml_ops::ck::BACKWARD_THREADS, 2) __global__
void MMQ_BUNDLE_KERNEL_SYMBOL(MMQ_GROUPED_SINGLE_ARGUMENTS) {
    torch_ggml_ops::ck::grouped_mmq_grad_input_iq2_tiled_body<true>(
        MMQ_GROUPED_SINGLE_VALUES);
}

#elif defined(MMQ_BUNDLE_GROUPED_BWD_IQ2_S_SINGLE_M128)
extern "C" __launch_bounds__(torch_ggml_ops::ck::BACKWARD_THREADS, 2) __global__
void MMQ_BUNDLE_KERNEL_SYMBOL(MMQ_GROUPED_SINGLE_ARGUMENTS) {
    torch_ggml_ops::ck::grouped_mmq_grad_input_iq2_s2_body(
        MMQ_GROUPED_SINGLE_VALUES);
}

#elif defined(MMQ_BUNDLE_GROUPED_BWD_IQ2_S_PAIR_M64)
extern "C" __launch_bounds__(torch_ggml_ops::ck::BACKWARD_THREADS, 2) __global__
void MMQ_BUNDLE_KERNEL_SYMBOL(MMQ_GROUPED_PAIR_ARGUMENTS) {
    torch_ggml_ops::ck::grouped_mmq_pair_grad_input_iq2_tiled_body<true>(
        MMQ_GROUPED_PAIR_VALUES);
}

#elif defined(MMQ_BUNDLE_GROUPED_BWD_IQ2_S_PAIR_M128)
extern "C" __launch_bounds__(torch_ggml_ops::ck::BACKWARD_THREADS, 2) __global__
void MMQ_BUNDLE_KERNEL_SYMBOL(MMQ_GROUPED_PAIR_ARGUMENTS) {
    torch_ggml_ops::ck::grouped_mmq_pair_grad_input_iq2_n64_large_body(
        MMQ_GROUPED_PAIR_VALUES);
}

#elif defined(MMQ_BUNDLE_GROUPED_BWD_Q4_ROW_TASK)
extern "C" __launch_bounds__(torch_ggml_ops::ck::BACKWARD_THREADS, 2) __global__
void MMQ_BUNDLE_KERNEL_SYMBOL(MMQ_GROUPED_ROW_TASK_ARGUMENTS) {
    torch_ggml_ops::ck::grouped_mmq_grad_input_q4_row_task_body(
        MMQ_GROUPED_ROW_TASK_VALUES);
}

#elif defined(MMQ_BUNDLE_GROUPED_BWD_Q5_ROW_TASK)
extern "C" __launch_bounds__(torch_ggml_ops::ck::BACKWARD_THREADS, 2) __global__
void MMQ_BUNDLE_KERNEL_SYMBOL(MMQ_GROUPED_ROW_TASK_ARGUMENTS) {
    torch_ggml_ops::ck::grouped_mmq_grad_input_q5_row_task_body(
        MMQ_GROUPED_ROW_TASK_VALUES);
}

#elif defined(MMQ_BUNDLE_GROUPED_BWD_IQ2_S_ROW_TASK)
extern "C" __launch_bounds__(torch_ggml_ops::ck::BACKWARD_THREADS, 2) __global__
void MMQ_BUNDLE_KERNEL_SYMBOL(MMQ_GROUPED_ROW_TASK_ARGUMENTS) {
    torch_ggml_ops::ck::grouped_mmq_grad_input_iq2_row_task_body(
        MMQ_GROUPED_ROW_TASK_VALUES);
}

#elif defined(MMQ_BUNDLE_GROUPED_BWD_FIXED_Q8_0_GENERIC)
extern "C" __launch_bounds__(torch_ggml_ops::ck::GROUPED_BACKWARD_THREADS, 1) __global__
void MMQ_BUNDLE_KERNEL_SYMBOL(MMQ_FIXED_GROUPED_BWD_ARGUMENTS) {
    torch_ggml_ops::ck::fixed_grouped_q8_0_grad_input_body(
        grad_output,
        packed_weight,
        grad_input,
        tokens,
        out_features,
        bytes_per_group);
}

#elif defined(MMQ_BUNDLE_GROUPED_BWD_Q2_K_SINGLE_M64_U1)
extern "C" __launch_bounds__(torch_ggml_ops::ck::BACKWARD_THREADS, 2) __global__
void MMQ_BUNDLE_KERNEL_SYMBOL(MMQ_GROUPED_DEEPSEEK_SINGLE_ARGUMENTS) {
    torch_ggml_ops::ck::grouped_mmq_grad_input_deepseek_body<
        GGML_TYPE_Q2_K, 4096, 2048, 8, 1, 1>(
            MMQ_GROUPED_DEEPSEEK_SINGLE_VALUES);
}

#elif defined(MMQ_BUNDLE_GROUPED_BWD_Q2_K_SINGLE_M128_U1)
extern "C" __launch_bounds__(torch_ggml_ops::ck::BACKWARD_THREADS, 2) __global__
void MMQ_BUNDLE_KERNEL_SYMBOL(MMQ_GROUPED_DEEPSEEK_SINGLE_ARGUMENTS) {
    torch_ggml_ops::ck::grouped_mmq_grad_input_deepseek_body<
        GGML_TYPE_Q2_K, 4096, 2048, 8, 2, 1>(
            MMQ_GROUPED_DEEPSEEK_SINGLE_VALUES);
}

#elif defined(MMQ_BUNDLE_GROUPED_BWD_IQ2_XXS_PAIR_M64)
extern "C" __launch_bounds__(torch_ggml_ops::ck::BACKWARD_THREADS, 2) __global__
void MMQ_BUNDLE_KERNEL_SYMBOL(MMQ_GROUPED_DEEPSEEK_PAIR_ARGUMENTS) {
    torch_ggml_ops::ck::grouped_mmq_pair_grad_input_deepseek_body<
        GGML_TYPE_IQ2_XXS, 2048, 4096, 16, 1>(
            MMQ_GROUPED_DEEPSEEK_PAIR_VALUES);
}

#elif defined(MMQ_BUNDLE_GROUPED_BWD_Q2_K_SINGLE_M128_U2)
extern "C" __launch_bounds__(torch_ggml_ops::ck::BACKWARD_THREADS, 2) __global__
void MMQ_BUNDLE_KERNEL_SYMBOL(MMQ_GROUPED_DEEPSEEK_SINGLE_ARGUMENTS) {
    torch_ggml_ops::ck::grouped_mmq_grad_input_deepseek_body<
        GGML_TYPE_Q2_K, 4096, 2048, 8, 2, 2>(
            MMQ_GROUPED_DEEPSEEK_SINGLE_VALUES);
}

#elif defined(MMQ_BUNDLE_GROUPED_BWD_FIXED_Q8_0_M256)
extern "C" __launch_bounds__(torch_ggml_ops::ck::BACKWARD_THREADS, 1) __global__
void MMQ_BUNDLE_KERNEL_SYMBOL(MMQ_FIXED_GROUPED_BWD_ARGUMENTS) {
    torch_ggml_ops::ck::fixed_grouped_q8_0_grad_input_tiled_body<4, 16, 0>(
        MMQ_FIXED_GROUPED_BWD_VALUES);
}

#else
#error "a symbolic MMQ grouped backward selector is required"
#endif
