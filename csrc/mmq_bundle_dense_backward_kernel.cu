// Canonical wrapper for independently packaged dense MMQ backward kernels.
#ifdef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_OPERATORS__
#endif
#ifdef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_CONVERSIONS__
#endif

#include "ck/mmq_backward.cuh"

#ifndef MMQ_BUNDLE_KERNEL_SYMBOL
#error "MMQ_BUNDLE_KERNEL_SYMBOL must name the exported kernel"
#endif
#ifndef MMQ_BUNDLE_QUANT_TYPE
#error "dense backward kernels require MMQ_BUNDLE_QUANT_TYPE"
#endif
#ifndef MMQ_BUNDLE_N_TILES
#error "dense backward kernels require MMQ_BUNDLE_N_TILES"
#endif
#ifndef MMQ_BUNDLE_K_ITERATION
#error "dense backward kernels require MMQ_BUNDLE_K_ITERATION"
#endif
#ifndef MMQ_BUNDLE_GROUP_M
#error "dense backward kernels require MMQ_BUNDLE_GROUP_M"
#endif
#ifndef MMQ_BUNDLE_M_TILES_PER_WAVE
#error "dense backward kernels require MMQ_BUNDLE_M_TILES_PER_WAVE"
#endif
#ifndef MMQ_BUNDLE_DECODER_WIDTH
#error "dense backward kernels require MMQ_BUNDLE_DECODER_WIDTH"
#endif
#ifndef MMQ_BUNDLE_PREFETCH_LOCAL
#error "dense backward kernels require MMQ_BUNDLE_PREFETCH_LOCAL"
#endif
#ifndef MMQ_BUNDLE_FULL_TILES
#error "dense backward kernels require MMQ_BUNDLE_FULL_TILES"
#endif
#ifndef MMQ_BUNDLE_PREFETCH_PACKED
#error "dense backward kernels require MMQ_BUNDLE_PREFETCH_PACKED"
#endif
#ifndef MMQ_BUNDLE_LDS_PADDING
#error "dense backward kernels require MMQ_BUNDLE_LDS_PADDING"
#endif
#ifndef MMQ_BUNDLE_VECTOR_LOCAL_LOAD
#error "dense backward kernels require MMQ_BUNDLE_VECTOR_LOCAL_LOAD"
#endif
#ifndef MMQ_BUNDLE_LDS_SWIZZLE_CHUNK
#error "dense backward kernels require MMQ_BUNDLE_LDS_SWIZZLE_CHUNK"
#endif
#ifndef MMQ_BUNDLE_PACK_Q5_QUANT_BYTES
#error "dense backward kernels require MMQ_BUNDLE_PACK_Q5_QUANT_BYTES"
#endif
#ifndef MMQ_BUNDLE_PACK_Q6_QUANT_BYTES
#error "dense backward kernels require MMQ_BUNDLE_PACK_Q6_QUANT_BYTES"
#endif

extern "C" __launch_bounds__(torch_ggml_ops::ck::BACKWARD_THREADS, 2) __global__
void MMQ_BUNDLE_KERNEL_SYMBOL(
        const __hip_bfloat16 * __restrict__ grad_output,
        const char * __restrict__ packed_weight,
        __hip_bfloat16 * __restrict__ grad_input,
        int rows,
        int out_features,
        int in_features,
        int blocks_per_weight_row) {
    torch_ggml_ops::ck::dense_mmq_grad_input_body<
        static_cast<ggml_type>(MMQ_BUNDLE_QUANT_TYPE),
        MMQ_BUNDLE_N_TILES,
        MMQ_BUNDLE_K_ITERATION,
        MMQ_BUNDLE_GROUP_M,
        MMQ_BUNDLE_M_TILES_PER_WAVE,
        MMQ_BUNDLE_DECODER_WIDTH,
        MMQ_BUNDLE_PREFETCH_LOCAL,
        MMQ_BUNDLE_FULL_TILES,
        MMQ_BUNDLE_PREFETCH_PACKED,
        MMQ_BUNDLE_LDS_PADDING,
        MMQ_BUNDLE_VECTOR_LOCAL_LOAD,
        MMQ_BUNDLE_LDS_SWIZZLE_CHUNK,
        MMQ_BUNDLE_PACK_Q5_QUANT_BYTES,
        MMQ_BUNDLE_PACK_Q6_QUANT_BYTES>(
            grad_output,
            packed_weight,
            grad_input,
            rows,
            out_features,
            in_features,
            blocks_per_weight_row);
}
