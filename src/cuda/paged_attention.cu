//! Kernels CUDA para PagedAttention sobre el memory-pool de `BlockAllocator`.
//! Layout de cada bloque físico (`block_bytes_half` en halfs):
//!   [ K region: block_size * num_kv_heads * head_dim ]
//!   [ V region: block_size * num_kv_heads * head_dim ]
//! El índice de un elemento K(t, kv_head, d) dentro del bloque físico `phys` es:
//!   base = phys * block_bytes_half
//!   k = base + (t * num_kv_heads + kv_head) * head_dim + d
//!   v = base + (block_size * num_kv_heads + t * num_kv_heads + kv_head) * head_dim + d
//! Idéntico al layout que lee la referencia CPU (`attention.zig`).
//!
//! Optimizaciones aplicadas:
//!  - Q en shared memory (tile).
//!  - Acc en shared memory (accumuladores FP32).
//!  - Online softmax (no materializa S completo).
//!  - Warp-level reduction via __shfl_xor_sync.
//!  - FP16 accumulation en FP32.
//!  - Loads vectorizados (LDST.128): 8 halfs por acceso vía float4/union.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

// Permite leer 8 halfs (16 bytes = LDST.128) manteniendo acceso a cada half.
union H8 {
    float4 v;
    half h[8];
};

// Carga 8 halfs contiguos y acumula el producto punto contra sq[0..7].
__device__ __forceinline__ float dot8(
    const float* sq, const half* __restrict__ kp, int g8
) {
    H8 kv;
    kv.v = *reinterpret_cast<const float4*>(kp + g8 * 8);
    const int s = g8 * 8;
    return sq[s + 0] * __half2float(kv.h[0])
         + sq[s + 1] * __half2float(kv.h[1])
         + sq[s + 2] * __half2float(kv.h[2])
         + sq[s + 3] * __half2float(kv.h[3])
         + sq[s + 4] * __half2float(kv.h[4])
         + sq[s + 5] * __half2float(kv.h[5])
         + sq[s + 6] * __half2float(kv.h[6])
         + sq[s + 7] * __half2float(kv.h[7]);
}

// Acumula en acc[0..7] el producto de V (8 halfs) por un escalar.
__device__ __forceinline__ void axpy8(
    float* acc, const half* __restrict__ vp, int g8, float scale
) {
    H8 vv;
    vv.v = *reinterpret_cast<const float4*>(vp + g8 * 8);
    const int s = g8 * 8;
    acc[s + 0] += __half2float(vv.h[0]) * scale;
    acc[s + 1] += __half2float(vv.h[1]) * scale;
    acc[s + 2] += __half2float(vv.h[2]) * scale;
    acc[s + 3] += __half2float(vv.h[3]) * scale;
    acc[s + 4] += __half2float(vv.h[4]) * scale;
    acc[s + 5] += __half2float(vv.h[5]) * scale;
    acc[s + 6] += __half2float(vv.h[6]) * scale;
    acc[s + 7] += __half2float(vv.h[7]) * scale;
}

// Carga Q (8 halfs) en sq[0..7] y pone a cero acc[0..7].
__device__ __forceinline__ void loadQ8(
    float* sq, float* acc, const half* __restrict__ qp, int g8
) {
    H8 qv;
    qv.v = *reinterpret_cast<const float4*>(qp + g8 * 8);
    const int s = g8 * 8;
    sq[s + 0] = __half2float(qv.h[0]);
    sq[s + 1] = __half2float(qv.h[1]);
    sq[s + 2] = __half2float(qv.h[2]);
    sq[s + 3] = __half2float(qv.h[3]);
    sq[s + 4] = __half2float(qv.h[4]);
    sq[s + 5] = __half2float(qv.h[5]);
    sq[s + 6] = __half2float(qv.h[6]);
    sq[s + 7] = __half2float(qv.h[7]);
    acc[s + 0] = 0.0f;
    acc[s + 1] = 0.0f;
    acc[s + 2] = 0.0f;
    acc[s + 3] = 0.0f;
    acc[s + 4] = 0.0f;
    acc[s + 5] = 0.0f;
    acc[s + 6] = 0.0f;
    acc[s + 7] = 0.0f;
}

// Reescala 8 valores de acc[0..7] por un escalar (necesario en online softmax
// cuando cambia el máximo corriente).
__device__ __forceinline__ void scal8(
    float* acc, int g8, float scale
) {
    const int s = g8 * 8;
    acc[s + 0] *= scale;
    acc[s + 1] *= scale;
    acc[s + 2] *= scale;
    acc[s + 3] *= scale;
    acc[s + 4] *= scale;
    acc[s + 5] *= scale;
    acc[s + 6] *= scale;
    acc[s + 7] *= scale;
}

// Escribe 8 halfs del output desde acc[0..7] normalizado por exp_sum.
__device__ __forceinline__ void storeOut8(
    half* __restrict__ op, int g8, const float* acc, float inv_sum
) {
    H8 o;
    const int s = g8 * 8;
    o.h[0] = __float2half(acc[s + 0] * inv_sum);
    o.h[1] = __float2half(acc[s + 1] * inv_sum);
    o.h[2] = __float2half(acc[s + 2] * inv_sum);
    o.h[3] = __float2half(acc[s + 3] * inv_sum);
    o.h[4] = __float2half(acc[s + 4] * inv_sum);
    o.h[5] = __float2half(acc[s + 5] * inv_sum);
    o.h[6] = __float2half(acc[s + 6] * inv_sum);
    o.h[7] = __float2half(acc[s + 7] * inv_sum);
    *reinterpret_cast<float4*>(op + g8 * 8) = o.v;
}

extern "C" __global__ void paged_attention_decode_f16_kernel(
    half* __restrict__ out,
    const half* __restrict__ query,
    const half* __restrict__ cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    int num_seqs,
    int max_num_blocks,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size
) {
    const int seq_idx = blockIdx.x;
    const int q_head = blockIdx.y;
    if (seq_idx >= num_seqs) return;
    const int seq_len = seq_lens[seq_idx];
    if (seq_len == 0) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int block_bytes_half = block_size * num_kv_heads * head_dim * 2;
    const int q_offset = (seq_idx * num_q_heads + q_head) * head_dim;
    const bool use_vec = (head_dim % 8 == 0);

    extern __shared__ float smem[];
    float* sq = smem;              // head_dim
    float* acc = smem + head_dim;  // head_dim

    if (use_vec) {
        const int vd = head_dim / 8;
        for (int g = tid; g < vd; g += nthreads) loadQ8(sq, acc, query + q_offset, g);
    } else {
        for (int d = tid; d < head_dim; d += nthreads) {
            sq[d] = __half2float(query[q_offset + d]);
            acc[d] = 0.0f;
        }
    }
    __syncthreads();

    float max_val = -1e30f;
    float exp_sum = 0.0f;
    const float scale_factor = 1.0f / sqrtf((float)head_dim);
    const int num_blocks = (seq_len + block_size - 1) / block_size;

    for (int b = 0; b < num_blocks; b++) {
        const int phys = block_tables[seq_idx * max_num_blocks + b];
        if (phys < 0) continue;
        const int tokens_in_block = (b == num_blocks - 1) ? (seq_len - b * block_size) : block_size;
        const int base = phys * block_bytes_half;

        for (int t = 0; t < tokens_in_block; t++) {
            const int k_base = base + (t * num_kv_heads + kv_head) * head_dim;
            float partial = 0.0f;
            if (use_vec) {
                const int vd = head_dim / 8;
                for (int g = tid; g < vd; g += nthreads) partial += dot8(sq, cache + k_base, g);
            } else {
                for (int d = tid; d < head_dim; d += nthreads) {
                    partial += sq[d] * __half2float(cache[k_base + d]);
                }
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                partial += __shfl_xor_sync(0xffffffffu, partial, off);
            }
            const float score = partial * scale_factor;

            const float new_max = fmaxf(max_val, score);
            const float scale = expf(max_val - new_max);
            exp_sum *= scale;
            if (use_vec) {
                const int vd = head_dim / 8;
                for (int g = tid; g < vd; g += nthreads) scal8(acc, g, scale);
            } else {
                for (int d = tid; d < head_dim; d += nthreads) acc[d] *= scale;
            }
            max_val = new_max;
            const float exp_score = expf(score - new_max);
            exp_sum += exp_score;

            const int v_base = base + (block_size * num_kv_heads + t * num_kv_heads + kv_head) * head_dim;
            if (use_vec) {
                const int vd = head_dim / 8;
                for (int g = tid; g < vd; g += nthreads) axpy8(acc, cache + v_base, g, exp_score);
            } else {
                for (int d = tid; d < head_dim; d += nthreads) {
                    acc[d] += __half2float(cache[v_base + d]) * exp_score;
                }
            }
        }
    }

    const int out_offset = q_offset;
    const float inv_sum = 1.0f / exp_sum;
    if (use_vec) {
        const int vd = head_dim / 8;
        for (int g = tid; g < vd; g += nthreads) storeOut8(out + out_offset, g, acc, inv_sum);
    } else {
        for (int d = tid; d < head_dim; d += nthreads) {
            out[out_offset + d] = __float2half(acc[d] * inv_sum);
        }
    }
}

// Prefill batch: un bloque por (token, q_head). Computa atención causal del
// token `blockIdx.x` sobre los tokens [0..blockIdx.x] del bloque físico actual
// hacia atrás, con online softmax. Out/queries layout: [seq_len, num_q_heads, head_dim].
extern "C" __global__ void paged_attention_prefill_f16_kernel(
    half* __restrict__ out,
    const half* __restrict__ queries,
    const half* __restrict__ cache,
    const int* __restrict__ block_tables,
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size
) {
    const int token = blockIdx.x;
    const int q_head = blockIdx.y;
    if (token >= seq_len) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int block_bytes_half = block_size * num_kv_heads * head_dim * 2;
    const int q_stride = num_q_heads * head_dim;
    const int q_offset = token * q_stride + q_head * head_dim;
    const bool use_vec = (head_dim % 8 == 0);

    extern __shared__ float smem[];
    float* sq = smem;              // head_dim
    float* acc = smem + head_dim;  // head_dim

    if (use_vec) {
        const int vd = head_dim / 8;
        for (int g = tid; g < vd; g += nthreads) loadQ8(sq, acc, queries + q_offset, g);
    } else {
        for (int d = tid; d < head_dim; d += nthreads) {
            sq[d] = __half2float(queries[q_offset + d]);
            acc[d] = 0.0f;
        }
    }
    __syncthreads();

    float max_val = -1e30f;
    float exp_sum = 0.0f;
    const float scale_factor = 1.0f / sqrtf((float)head_dim);
    // Causal: solo bloques que contienen tokens <= `token`.
    const int last_block = token / block_size;

    for (int b = 0; b <= last_block; b++) {
        const int phys = block_tables[b];
        if (phys < 0) continue;
        const int tokens_in_block = (b == last_block) ? (token % block_size) + 1 : block_size;
        const int base = phys * block_bytes_half;

        for (int t = 0; t < tokens_in_block; t++) {
            const int k_base = base + (t * num_kv_heads + kv_head) * head_dim;
            float partial = 0.0f;
            if (use_vec) {
                const int vd = head_dim / 8;
                for (int g = tid; g < vd; g += nthreads) partial += dot8(sq, cache + k_base, g);
            } else {
                for (int d = tid; d < head_dim; d += nthreads) {
                    partial += sq[d] * __half2float(cache[k_base + d]);
                }
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                partial += __shfl_xor_sync(0xffffffffu, partial, off);
            }
            const float score = partial * scale_factor;

            const float new_max = fmaxf(max_val, score);
            const float scale = expf(max_val - new_max);
            exp_sum *= scale;
            if (use_vec) {
                const int vd = head_dim / 8;
                for (int g = tid; g < vd; g += nthreads) scal8(acc, g, scale);
            } else {
                for (int d = tid; d < head_dim; d += nthreads) acc[d] *= scale;
            }
            max_val = new_max;
            const float exp_score = expf(score - new_max);
            exp_sum += exp_score;

            const int v_base = base + (block_size * num_kv_heads + t * num_kv_heads + kv_head) * head_dim;
            if (use_vec) {
                const int vd = head_dim / 8;
                for (int g = tid; g < vd; g += nthreads) axpy8(acc, cache + v_base, g, exp_score);
            } else {
                for (int d = tid; d < head_dim; d += nthreads) {
                    acc[d] += __half2float(cache[v_base + d]) * exp_score;
                }
            }
        }
    }

    const int out_offset = q_offset;
    const float inv_sum = 1.0f / exp_sum;
    if (use_vec) {
        const int vd = head_dim / 8;
        for (int g = tid; g < vd; g += nthreads) storeOut8(out + out_offset, g, acc, inv_sum);
    } else {
        for (int d = tid; d < head_dim; d += nthreads) {
            out[out_offset + d] = __float2half(acc[d] * inv_sum);
        }
    }
}

extern "C" __global__ void reshape_and_block_write_f16_kernel(
    half* __restrict__ cache,
    const half* __restrict__ new_keys,
    const half* __restrict__ new_values,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int* __restrict__ token_offsets,
    int num_seqs,
    int max_num_blocks,
    int num_kv_heads,
    int head_dim,
    int block_size
) {
    const int seq_idx = blockIdx.x;
    if (seq_idx >= num_seqs) return;

    const int seq_len = seq_lens[seq_idx];
    const int token_start = token_offsets[seq_idx];
    const int num_new_tokens = seq_len - token_start;
    const int kv_dim = num_kv_heads * head_dim;
    const int block_bytes_half = block_size * kv_dim * 2;

    const int total = num_new_tokens * kv_dim;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int t = idx / kv_dim;
        const int kv = idx % kv_dim;
        const int kv_head = kv / head_dim;
        const int d = kv % head_dim;
        const int global_token = token_start + t;
        const int block_idx = global_token / block_size;
        const int block_offset = global_token % block_size;
        const int phys = block_tables[seq_idx * max_num_blocks + block_idx];
        if (phys < 0) continue;

        const int base = phys * block_bytes_half;
        const int src = global_token * kv_dim + kv;
        const int k_idx = base + (block_offset * num_kv_heads + kv_head) * head_dim + d;
        const int v_idx = base + (block_size * num_kv_heads + block_offset * num_kv_heads + kv_head) * head_dim + d;
        cache[k_idx] = new_keys[src];
        cache[v_idx] = new_values[src];
    }
}

extern "C" __global__ void block_copy_f16_kernel(
    half* __restrict__ dst_cache,
    const half* __restrict__ src_cache,
    const int2* __restrict__ copy_map,
    int num_copies,
    int block_bytes
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_copies) return;
    const int2 mapping = copy_map[idx];
    const int dst_block = mapping.x;
    const int src_block = mapping.y;
    const half* src = src_cache + (src_block * block_bytes / sizeof(half));
    half* dst = dst_cache + (dst_block * block_bytes / sizeof(half));
    const int num_elems = block_bytes / sizeof(half);
    for (int i = threadIdx.x; i < num_elems; i += blockDim.x) {
        dst[i] = src[i];
    }
}