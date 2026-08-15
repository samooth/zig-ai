//! Kernels CUDA para PagedAttention sobre el memory-pool de `BlockAllocator`.
//! Layout de cada bloque físico (`block_bytes_half` en halfs):
//!   [ K region: block_size * num_kv_heads * head_dim ]
//!   [ V region: block_size * num_kv_heads * head_dim ]
//! El índice de un elemento K(t, kv_head, d) dentro del bloque físico `phys` es:
//!   base = phys * block_bytes_half
//!   k = base + (t * num_kv_heads + kv_head) * head_dim + d
//!   v = base + (block_size * num_kv_heads + t * num_kv_heads + kv_head) * head_dim + d
//! Idéntico al layout que lee la referencia CPU (`attention.zig`).
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

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

    extern __shared__ float smem[];
    float* sq = smem;              // head_dim
    float* acc = smem + head_dim;  // head_dim

    for (int d = tid; d < head_dim; d += nthreads) {
        sq[d] = __half2float(query[q_offset + d]);
        acc[d] = 0.0f;
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
            for (int d = tid; d < head_dim; d += nthreads) {
                partial += sq[d] * __half2float(cache[k_base + d]);
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                partial += __shfl_xor_sync(0xffffffffu, partial, off);
            }
            const float score = partial * scale_factor;

            const float new_max = fmaxf(max_val, score);
            const float scale = expf(max_val - new_max);
            exp_sum *= scale;
            max_val = new_max;
            const float exp_score = expf(score - new_max);
            exp_sum += exp_score;

            const int v_base = base + (block_size * num_kv_heads + t * num_kv_heads + kv_head) * head_dim;
            for (int d = tid; d < head_dim; d += nthreads) {
                acc[d] += __half2float(cache[v_base + d]) * exp_score;
            }
        }
    }

    const int out_offset = q_offset;
    for (int d = tid; d < head_dim; d += nthreads) {
        out[out_offset + d] = __float2half(acc[d] / exp_sum);
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