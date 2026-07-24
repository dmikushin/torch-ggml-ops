// Canonical wrappers for independently packaged MMQ forward and setup kernels.
#ifdef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_OPERATORS__
#endif
#ifdef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_CONVERSIONS__
#endif

#include "mmq_core.cuh"

#ifndef MMQ_BUNDLE_KERNEL_SYMBOL
#error "MMQ_BUNDLE_KERNEL_SYMBOL must name the exported kernel"
#endif
#ifndef MMQ_BUNDLE_FORWARD_KIND
#error "MMQ_BUNDLE_FORWARD_KIND must select a forward kernel family"
#endif

#if MMQ_BUNDLE_FORWARD_KIND == 1

#ifndef MMQ_BUNDLE_QUANT_TYPE
#error "quantize kernels require MMQ_BUNDLE_QUANT_TYPE"
#endif

extern "C" __launch_bounds__(512, 1) __global__
void MMQ_BUNDLE_KERNEL_SYMBOL(
        const __hip_bfloat16 * __restrict__ input,
        block_q8_1_mmq * __restrict__ output,
        int64_t rows,
        int64_t rows_padded,
        int64_t k) {
    quantize_bf16_mmq_q8_1_body<
        static_cast<ggml_type>(MMQ_BUNDLE_QUANT_TYPE)>(
            input, output, rows, rows_padded, k);
}

#elif MMQ_BUNDLE_FORWARD_KIND == 2

#ifndef MMQ_BUNDLE_QUANT_TYPE
#error "dense forward kernels require MMQ_BUNDLE_QUANT_TYPE"
#endif
#ifndef MMQ_BUNDLE_J
#error "dense forward kernels require MMQ_BUNDLE_J"
#endif

extern "C" __launch_bounds__(MMQ_NTHREADS, 2) __global__
void MMQ_BUNDLE_KERNEL_SYMBOL(
        const char * __restrict__ weights,
        const int * __restrict__ activations,
        __hip_bfloat16 * __restrict__ dst,
        int nrows_weight,
        int nrows_activation,
        int nrows_activation_padded,
        int blocks_per_weight_row) {
    dense_mmq_bf16_body<
        static_cast<ggml_type>(MMQ_BUNDLE_QUANT_TYPE), MMQ_BUNDLE_J>(
            weights,
            activations,
            dst,
            nrows_weight,
            nrows_activation,
            nrows_activation_padded,
            blocks_per_weight_row);
}

#elif MMQ_BUNDLE_FORWARD_KIND == 3

#ifndef MMQ_BUNDLE_QUANT_TYPE
#error "grouped serial kernels require MMQ_BUNDLE_QUANT_TYPE"
#endif
#ifndef MMQ_BUNDLE_J
#error "grouped serial kernels require MMQ_BUNDLE_J"
#endif
#ifndef MMQ_BUNDLE_NROWS_WEIGHT
#error "grouped serial kernels require MMQ_BUNDLE_NROWS_WEIGHT"
#endif
#ifndef MMQ_BUNDLE_BLOCKS_PER_WEIGHT_ROW
#error "grouped serial kernels require MMQ_BUNDLE_BLOCKS_PER_WEIGHT_ROW"
#endif

extern "C" __launch_bounds__(MMQ_NTHREADS, 2) __global__
void MMQ_BUNDLE_KERNEL_SYMBOL(
        const char * __restrict__ weights,
        const int * __restrict__ activations,
        __hip_bfloat16 * __restrict__ dst,
        const int64_t * __restrict__ expert_indices,
        const int32_t * __restrict__ expert_offsets,
        int num_experts,
        int nrows_weight,
        int nrows_activation,
        int blocks_per_weight_row,
        int64_t bytes_per_expert) {
    grouped_mmq_bf16_body<
        static_cast<ggml_type>(MMQ_BUNDLE_QUANT_TYPE),
        MMQ_BUNDLE_J,
        MMQ_BUNDLE_NROWS_WEIGHT,
        MMQ_BUNDLE_BLOCKS_PER_WEIGHT_ROW>(
            weights,
            activations,
            dst,
            expert_indices,
            expert_offsets,
            num_experts,
            nrows_weight,
            nrows_activation,
            blocks_per_weight_row,
            bytes_per_expert);
}

#elif MMQ_BUNDLE_FORWARD_KIND == 4

#ifndef MMQ_BUNDLE_QUANT_TYPE
#error "grouped row-task kernels require MMQ_BUNDLE_QUANT_TYPE"
#endif
#ifndef MMQ_BUNDLE_J
#error "grouped row-task kernels require MMQ_BUNDLE_J"
#endif
#ifndef MMQ_BUNDLE_NROWS_WEIGHT
#error "grouped row-task kernels require MMQ_BUNDLE_NROWS_WEIGHT"
#endif
#ifndef MMQ_BUNDLE_BLOCKS_PER_WEIGHT_ROW
#error "grouped row-task kernels require MMQ_BUNDLE_BLOCKS_PER_WEIGHT_ROW"
#endif

extern "C" __launch_bounds__(MMQ_NTHREADS, 2) __global__
void MMQ_BUNDLE_KERNEL_SYMBOL(
        const char * __restrict__ weights,
        const int * __restrict__ activations,
        __hip_bfloat16 * __restrict__ dst,
        const int32_t * __restrict__ task_count,
        const int32_t * __restrict__ task_experts,
        const int32_t * __restrict__ task_row_starts,
        const int32_t * __restrict__ task_row_ends,
        int nrows_activation,
        int64_t bytes_per_expert) {
    grouped_mmq_row_task_body<
        static_cast<ggml_type>(MMQ_BUNDLE_QUANT_TYPE),
        MMQ_BUNDLE_J,
        MMQ_BUNDLE_NROWS_WEIGHT,
        MMQ_BUNDLE_BLOCKS_PER_WEIGHT_ROW>(
            weights,
            activations,
            dst,
            task_count,
            task_experts,
            task_row_starts,
            task_row_ends,
            nrows_activation,
            bytes_per_expert);
}

#elif MMQ_BUNDLE_FORWARD_KIND == 5

#ifndef MMQ_BUNDLE_J
#error "fixed grouped kernels require MMQ_BUNDLE_J"
#endif
#ifndef MMQ_BUNDLE_GROUPS
#error "fixed grouped kernels require MMQ_BUNDLE_GROUPS"
#endif
#ifndef MMQ_BUNDLE_BLOCKS_PER_WEIGHT_ROW
#error "fixed grouped kernels require MMQ_BUNDLE_BLOCKS_PER_WEIGHT_ROW"
#endif
#ifndef MMQ_BUNDLE_FALLBACK
#error "fixed grouped kernels require MMQ_BUNDLE_FALLBACK"
#endif

extern "C" __launch_bounds__(MMQ_NTHREADS, 2) __global__
void MMQ_BUNDLE_KERNEL_SYMBOL(
        const char * __restrict__ weights,
        const int * __restrict__ activations,
        __hip_bfloat16 * __restrict__ dst,
        int tokens,
        int nrows_weight,
        int64_t bytes_per_group) {
    fixed_grouped_q8_0_mmq_bf16_body<
        MMQ_BUNDLE_J,
        MMQ_BUNDLE_GROUPS,
        MMQ_BUNDLE_BLOCKS_PER_WEIGHT_ROW,
        MMQ_BUNDLE_FALLBACK>(
            weights,
            activations,
            dst,
            tokens,
            nrows_weight,
            bytes_per_group);
}

#elif MMQ_BUNDLE_FORWARD_KIND == 6

extern "C" __launch_bounds__(256, 1) __global__
void MMQ_BUNDLE_KERNEL_SYMBOL(
        const int64_t * __restrict__ expert_indices,
        const int32_t * __restrict__ expert_offsets,
        int32_t * __restrict__ task_count,
        int32_t * __restrict__ task_experts,
        int32_t * __restrict__ task_row_starts,
        int32_t * __restrict__ task_row_ends,
        int num_experts,
        int num_groups,
        int nrows_activation,
        int row_tile) {
    grouped_mmq_build_row_tasks_body(
        expert_indices,
        expert_offsets,
        task_count,
        task_experts,
        task_row_starts,
        task_row_ends,
        num_experts,
        num_groups,
        nrows_activation,
        row_tile);
}

#else
#error "unsupported MMQ_BUNDLE_FORWARD_KIND"
#endif
