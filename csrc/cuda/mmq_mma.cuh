#pragma once

// Dense quantized matmul for NVIDIA Ampere and newer (sm_80+).
//
// The weight is never materialized. Each CTA decodes only the GGUF tile it is
// about to multiply into shared memory as BF16, and the product runs on BF16
// tensor cores (mma.sync.m16n8k16, FP32 accumulation). This is the same
// numerical scheme as the upstream HIP backward kernels (tile dequant to BF16,
// BF16 MMA), used here for both directions.
//
// One kernel template serves both directions as C[M, Nc] = A[M, Kc] * B[Kc, Nc]:
//   forward:   A = input       [M, K],  B[k][n] = W[n][k],  C = output     [M, N]
//   backward:  A = grad_output [M, N],  B[n][k] = W[n][k],  C = grad_input [M, K]
// W is the packed GGUF matrix of logical shape [N, K] (row n holds K / QK packed
// blocks). In both directions the shared-memory B tile holds a W sub-block in
// W's own row-major order; only the fragment load differs (ldmatrix vs
// ldmatrix.trans).

#include "../vendor/llama_cpp/common.cuh"
#include "../ck/gguf_decode.cuh"

#include <cuda_bf16.h>
#include <cstdint>

namespace torch_ggml_ops::cuda_mma {

using torch_ggml_ops::ck::k_min;
using torch_ggml_ops::ck::k_scale;

// ---------------------------------------------------------------------------
// GGUF block geometry
// ---------------------------------------------------------------------------

template <ggml_type type>
struct quant_traits;

template <>
struct quant_traits<GGML_TYPE_Q8_0> {
    static constexpr int block_values = QK8_0;
    static constexpr int block_bytes = sizeof(block_q8_0);
};
template <>
struct quant_traits<GGML_TYPE_Q4_K> {
    static constexpr int block_values = QK_K;
    static constexpr int block_bytes = sizeof(block_q4_K);
};
template <>
struct quant_traits<GGML_TYPE_Q5_K> {
    static constexpr int block_values = QK_K;
    static constexpr int block_bytes = sizeof(block_q5_K);
};
template <>
struct quant_traits<GGML_TYPE_Q6_K> {
    static constexpr int block_values = QK_K;
    static constexpr int block_bytes = sizeof(block_q6_K);
};

static __device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
    const __nv_bfloat162 pair = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&pair);
}

// Four naturally 2-byte-aligned uint16 loads. Q6_K (210 B) and Q8_0 (34 B)
// blocks are only 2-byte aligned inside a row.
static __device__ __forceinline__ uint2 load8_u16(const uint8_t * p) {
    const uint16_t * q = reinterpret_cast<const uint16_t *>(p);
    uint2 r;
    r.x = uint32_t(__ldg(q + 0)) | (uint32_t(__ldg(q + 1)) << 16);
    r.y = uint32_t(__ldg(q + 2)) | (uint32_t(__ldg(q + 3)) << 16);
    return r;
}

static __device__ __forceinline__ float ldg_half(const void * p) {
    const unsigned short bits = __ldg(reinterpret_cast<const unsigned short *>(p));
    return __half2float(__ushort_as_half(bits));
}

static __device__ __forceinline__ int byte_of(uint2 v, int i) {
    return int(((i < 4 ? v.x : v.y) >> (8 * (i & 3))) & 0xff);
}

// Decode the 8 consecutive logical values [k, k + 8) of one packed row into
// eight BF16 values. k is a multiple of 8, so the 8 values always share one
// scale (Q6_K scales cover 16 values, the others 32).
template <ggml_type type>
static __device__ __forceinline__ uint4 decode8(const uint8_t * row, int k);

template <>
__device__ __forceinline__ uint4 decode8<GGML_TYPE_Q4_K>(const uint8_t * row, int k) {
    const block_q4_K * block = reinterpret_cast<const block_q4_K *>(row) + (k >> 8);
    const int j = k & 255;
    const int group = j >> 5;
    const float d = ldg_half(&block->d) * float(k_scale(block->scales, group));
    const float m = ldg_half(&block->dmin) * float(k_min(block->scales, group));
    const uint2 q = __ldg(reinterpret_cast<const uint2 *>(block->qs + (group >> 1) * 32 + (j & 31)));
    const int shift = 4 * (group & 1);
    float v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = d * float((byte_of(q, i) >> shift) & 0x0f) - m;
    }
    return make_uint4(pack_bf16x2(v[0], v[1]), pack_bf16x2(v[2], v[3]),
                      pack_bf16x2(v[4], v[5]), pack_bf16x2(v[6], v[7]));
}

template <>
__device__ __forceinline__ uint4 decode8<GGML_TYPE_Q5_K>(const uint8_t * row, int k) {
    const block_q5_K * block = reinterpret_cast<const block_q5_K *>(row) + (k >> 8);
    const int j = k & 255;
    const int group = j >> 5;
    const float d = ldg_half(&block->d) * float(k_scale(block->scales, group));
    const float m = ldg_half(&block->dmin) * float(k_min(block->scales, group));
    const uint2 ql = __ldg(reinterpret_cast<const uint2 *>(block->qs + (group >> 1) * 32 + (j & 31)));
    const uint2 qh = __ldg(reinterpret_cast<const uint2 *>(block->qh + (j & 31)));
    const int shift = 4 * (group & 1);
    float v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int q = ((byte_of(ql, i) >> shift) & 0x0f) | (((byte_of(qh, i) >> group) & 1) << 4);
        v[i] = d * float(q) - m;
    }
    return make_uint4(pack_bf16x2(v[0], v[1]), pack_bf16x2(v[2], v[3]),
                      pack_bf16x2(v[4], v[5]), pack_bf16x2(v[6], v[7]));
}

template <>
__device__ __forceinline__ uint4 decode8<GGML_TYPE_Q6_K>(const uint8_t * row, int k) {
    const uint8_t * block = row + (k >> 8) * int(sizeof(block_q6_K));
    const int j = k & 255;
    const int chunk = j >> 7;
    const int r = j & 127;
    const uint2 ql = load8_u16(block + chunk * 64 + (r & 63));
    const uint2 qh = load8_u16(block + 128 + chunk * 32 + (r & 31));
    const int low_shift = 4 * (r >> 6);
    const int high_shift = 2 * (r >> 5);
    const float scale = ldg_half(block + 208) *
        float(static_cast<int8_t>(__ldg(block + 192 + (j >> 4))));
    float v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int q = ((byte_of(ql, i) >> low_shift) & 0x0f) |
            (((byte_of(qh, i) >> high_shift) & 0x03) << 4);
        v[i] = scale * float(q - 32);
    }
    return make_uint4(pack_bf16x2(v[0], v[1]), pack_bf16x2(v[2], v[3]),
                      pack_bf16x2(v[4], v[5]), pack_bf16x2(v[6], v[7]));
}

template <>
__device__ __forceinline__ uint4 decode8<GGML_TYPE_Q8_0>(const uint8_t * row, int k) {
    const uint8_t * block = row + (k >> 5) * int(sizeof(block_q8_0));
    const float d = ldg_half(block);
    const uint2 q = load8_u16(block + 2 + (k & 31));
    float v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = d * float(static_cast<int8_t>(byte_of(q, i)));
    }
    return make_uint4(pack_bf16x2(v[0], v[1]), pack_bf16x2(v[2], v[3]),
                      pack_bf16x2(v[4], v[5]), pack_bf16x2(v[6], v[7]));
}

// ---------------------------------------------------------------------------
// PTX wrappers
// ---------------------------------------------------------------------------

static __device__ __forceinline__ uint32_t smem_u32(const void * p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// 16-byte async copy; src_bytes == 0 zero-fills the destination.
static __device__ __forceinline__ void cp_async16(void * dst, const void * src, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(smem_u32(dst)), "l"(src), "r"(src_bytes));
}

static __device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

static __device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}

static __device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void * p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_u32(p)));
}

static __device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const void * p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_u32(p)));
}

static __device__ __forceinline__ void mma_bf16_16816(
        float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

template <bool backward>
struct tile_config {
    static constexpr int BM = 128;   // rows of A / C per CTA
    static constexpr int BN = 128;   // columns of C per CTA
    static constexpr int BK = 32;    // contraction step
    static constexpr int WARPS_M = 2;
    static constexpr int WARPS_N = 4;
    static constexpr int THREADS = 32 * WARPS_M * WARPS_N;
    static constexpr int WM = BM / WARPS_M;   // 64
    static constexpr int WN = BN / WARPS_N;   // 32
    static constexpr int MT = WM / 16;        // m16 tiles per warp
    static constexpr int NT = WN / 8;         // n8 tiles per warp
    static constexpr int PAD = 8;             // BF16 padding: conflict-free ldmatrix
    static constexpr int A_STRIDE = BK + PAD;
    // W sub-block in W order: forward [BN n-rows][BK k-cols], backward [BK n-rows][BN k-cols].
    static constexpr int W_ROWS = backward ? BK : BN;
    static constexpr int W_COLS = backward ? BN : BK;
    static constexpr int W_STRIDE = W_COLS + PAD;
    static constexpr int A_ELEMS = BM * A_STRIDE;
    static constexpr int W_ELEMS = W_ROWS * W_STRIDE;
    static constexpr int SMEM_BYTES = 2 * (A_ELEMS + W_ELEMS) * 2;
};

// a_vec: the A row stride is a multiple of 8 elements, so 16-byte cp.async is legal.
template <ggml_type type, bool backward, bool a_vec>
__global__ void __launch_bounds__(tile_config<backward>::THREADS, 2)
mmq_mma_kernel(
        const __nv_bfloat16 * __restrict__ a,
        const uint8_t * __restrict__ w,
        __nv_bfloat16 * __restrict__ c,
        int M,
        int Kc,              // contraction length (K forward, N backward)
        int Nc,              // output columns    (N forward, K backward)
        int w_rows,          // logical N of W
        int64_t w_row_bytes) {
    using cfg = tile_config<backward>;
    extern __shared__ __align__(16) uint8_t smem_raw[];
    __nv_bfloat16 * a_smem = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
    __nv_bfloat16 * w_smem = a_smem + 2 * cfg::A_ELEMS;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = warp / cfg::WARPS_N;
    const int warp_n = warp % cfg::WARPS_N;

    const int n0 = blockIdx.x * cfg::BN;
    const int m0 = blockIdx.y * cfg::BM;

    // --- tile loaders -------------------------------------------------------
    auto load_a = [&](int stage, int kc0) {
        __nv_bfloat16 * dst = a_smem + stage * cfg::A_ELEMS;
        // BM x BK = 128 x 32 BF16 = 512 chunks of 8 values; 2 per thread.
#pragma unroll
        for (int it = 0; it < (cfg::BM * cfg::BK / 8) / cfg::THREADS; ++it) {
            const int chunk = tid + it * cfg::THREADS;
            const int row = chunk / (cfg::BK / 8);
            const int col = (chunk % (cfg::BK / 8)) * 8;
            const int gm = m0 + row;
            const int gk = kc0 + col;
            __nv_bfloat16 * s = dst + row * cfg::A_STRIDE + col;
            if constexpr (a_vec) {
                // Kc is a multiple of 8 here, so a chunk is entirely in or out.
                const bool inside = gm < M && gk < Kc;
                const __nv_bfloat16 * g = a + (inside ? int64_t(gm) * Kc + gk : 0);
                cp_async16(s, g, inside ? 16 : 0);
            } else {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    s[e] = (gm < M && gk + e < Kc)
                        ? a[int64_t(gm) * Kc + gk + e]
                        : __float2bfloat16(0.0f);
                }
            }
        }
    };

    // W sub-block rows [wr0, wr0 + W_ROWS) x cols [wc0, wc0 + W_COLS), decoded to BF16.
    auto load_w = [&](int stage, int wr0, int wc0) {
        __nv_bfloat16 * dst = w_smem + stage * cfg::W_ELEMS;
        constexpr int chunks_per_row = cfg::W_COLS / 8;
#pragma unroll
        for (int it = 0; it < (cfg::W_ROWS * cfg::W_COLS / 8) / cfg::THREADS; ++it) {
            const int chunk = tid + it * cfg::THREADS;
            const int row = chunk / chunks_per_row;
            const int col = (chunk % chunks_per_row) * 8;
            const int gr = wr0 + row;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (gr < w_rows) {
                v = decode8<type>(w + int64_t(gr) * w_row_bytes, wc0 + col);
            }
            *reinterpret_cast<uint4 *>(dst + row * cfg::W_STRIDE + col) = v;
        }
    };

    auto load_stage = [&](int stage, int kc0) {
        load_a(stage, kc0);
        if constexpr (backward) {
            load_w(stage, kc0, n0);   // W rows = contraction (N), cols = output (K)
        } else {
            load_w(stage, n0, kc0);   // W rows = output (N), cols = contraction (K)
        }
        if constexpr (a_vec) {
            cp_async_commit();
        }
    };

    float acc[cfg::MT][cfg::NT][4];
#pragma unroll
    for (int i = 0; i < cfg::MT; ++i)
#pragma unroll
        for (int j = 0; j < cfg::NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

    const int k_tiles = (Kc + cfg::BK - 1) / cfg::BK;

    load_stage(0, 0);
    if constexpr (a_vec) cp_async_wait_all();
    __syncthreads();

    for (int kt = 0; kt < k_tiles; ++kt) {
        const int stage = kt & 1;
        if (kt + 1 < k_tiles) {
            load_stage(stage ^ 1, (kt + 1) * cfg::BK);
        }

        const __nv_bfloat16 * as = a_smem + stage * cfg::A_ELEMS;
        const __nv_bfloat16 * ws = w_smem + stage * cfg::W_ELEMS;
#pragma unroll
        for (int kk = 0; kk < cfg::BK; kk += 16) {
            uint32_t af[cfg::MT][4];
#pragma unroll
            for (int i = 0; i < cfg::MT; ++i) {
                const int r = warp_m * cfg::WM + i * 16 + (lane & 15);
                const int col = kk + (lane >> 4) * 8;
                ldmatrix_x4(af[i], as + r * cfg::A_STRIDE + col);
            }
            uint32_t bf[cfg::NT][2];
#pragma unroll
            for (int j = 0; j < cfg::NT; j += 2) {
                uint32_t t[4];
                const int nb = warp_n * cfg::WN + j * 8;
                if constexpr (backward) {
                    // ws is [k][n] (n contiguous): transpose while loading.
                    const int kr = kk + (lane & 7) + ((lane >> 3) & 1) * 8;
                    const int nc = nb + (lane >> 4) * 8;
                    ldmatrix_x4_trans(t, ws + kr * cfg::W_STRIDE + nc);
                } else {
                    // ws is [n][k] (k contiguous): already the "col" B operand.
                    const int nr = nb + (lane & 7) + (lane >> 4) * 8;
                    const int kc = kk + ((lane >> 3) & 1) * 8;
                    ldmatrix_x4(t, ws + nr * cfg::W_STRIDE + kc);
                }
                bf[j][0] = t[0];
                bf[j][1] = t[1];
                bf[j + 1][0] = t[2];
                bf[j + 1][1] = t[3];
            }
#pragma unroll
            for (int i = 0; i < cfg::MT; ++i)
#pragma unroll
                for (int j = 0; j < cfg::NT; ++j)
                    mma_bf16_16816(acc[i][j], af[i], bf[j][0], bf[j][1]);
        }

        if constexpr (a_vec) cp_async_wait_all();
        __syncthreads();
    }

    // --- epilogue -------------------------------------------------------------
    const int g = lane >> 2;
    const int t = lane & 3;
    const bool pair_store = (Nc & 1) == 0;
#pragma unroll
    for (int i = 0; i < cfg::MT; ++i) {
#pragma unroll
        for (int j = 0; j < cfg::NT; ++j) {
            const int col = n0 + warp_n * cfg::WN + j * 8 + 2 * t;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + warp_m * cfg::WM + i * 16 + g + 8 * h;
                if (row >= M) continue;
                __nv_bfloat16 * out = c + int64_t(row) * Nc + col;
                const float v0 = acc[i][j][2 * h];
                const float v1 = acc[i][j][2 * h + 1];
                if (pair_store && col + 1 < Nc) {
                    *reinterpret_cast<__nv_bfloat162 *>(out) = __floats2bfloat162_rn(v0, v1);
                } else {
                    if (col < Nc) out[0] = __float2bfloat16(v0);
                    if (col + 1 < Nc) out[1] = __float2bfloat16(v1);
                }
            }
        }
    }
}

} // namespace torch_ggml_ops::cuda_mma
