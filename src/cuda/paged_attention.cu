#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif
#define WARP_SIZE 32

struct SoftmaxState {
    float max_val;
    float exp_sum;
    float acc[128];
};

__global__ void paged_attention_decode_f16_kernel(
    half* __restrict__ out,
    const half* __restrict__ query,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    int num_seqs,
    int max_num_blocks,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int q_per_kv
) {
    const int seq_idx = blockIdx.x;
    const int q_head = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    if (seq_idx >= num_seqs) return;
    const int seq_len = seq_lens[seq_idx];
    if (seq_len == 0) return;

    const int q_offset = ((seq_idx * num_q_heads) + q_head) * head_dim;
    extern __shared__ float shared_q[];
    if (tid < head_dim) {
        shared_q[tid] = __half2float(query[q_offset + tid]);
    }
    __syncthreads();

    SoftmaxState state;
    state.max_val = -1e30f;
    state.exp_sum = 0.0f;
    #pragma unroll
    for (int d = 0; d < head_dim; d++) state.acc[d] = 0.0f;

    const int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int b = warp_id; b < num_blocks; b += blockDim.x / WARP_SIZE) {
        const int phys_block = block_tables[seq_idx * max_num_blocks + b];
        if (phys_block < 0) continue;

        const int tokens_in_block = (b == num_blocks - 1)
            ? (seq_len - b * BLOCK_SIZE)
            : BLOCK_SIZE;

        for (int t = lane; t < tokens_in_block; t += WARP_SIZE) {
            const int kv_head = q_head / q_per_kv;
            const int kv_offset = ((phys_block * BLOCK_SIZE + t) * num_kv_heads + kv_head) * head_dim;

            float score = 0.0f;
            #pragma unroll
            for (int d = 0; d < head_dim; d++) {
                score += shared_q[d] * __half2float(key_cache[kv_offset + d]);
            }
            score /= sqrtf((float)head_dim);

            float new_max = fmaxf(state.max_val, score);
            float scale = expf(state.max_val - new_max);
            state.exp_sum *= scale;
            state.max_val = new_max;
            float exp_score = expf(score - new_max);
            state.exp_sum += exp_score;

            #pragma unroll
            for (int d = 0; d < head_dim; d++) {
                state.acc[d] += __half2float(value_cache[kv_offset + d]) * exp_score;
            }
        }
    }

    const int out_offset = q_offset;
    if (tid < head_dim) {
        out[out_offset + tid] = __float2half(state.acc[tid] / state.exp_sum);
    }
}

__global__ void reshape_and_block_write_f16_kernel(
    half* __restrict__ key_cache,
    half* __restrict__ value_cache,
    const half* __restrict__ new_keys,
    const half* __restrict__ new_values,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int* __restrict__ token_offsets,
    int num_seqs,
    int max_num_blocks,
    int num_kv_heads,
    int head_dim
) {
    const int seq_idx = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int tid = threadIdx.x;
    if (seq_idx >= num_seqs) return;

    const int seq_len = seq_lens[seq_idx];
    const int token_start = token_offsets[seq_idx];
    const int num_new_tokens = seq_len - token_start;

    for (int t = tid; t < num_new_tokens; t += blockDim.x) {
        const int global_token = token_start + t;
        const int block_idx = global_token / BLOCK_SIZE;
        const int block_offset = global_token % BLOCK_SIZE;
        const int phys_block = block_tables[seq_idx * max_num_blocks + block_idx];
        if (phys_block < 0) continue;

        const int cache_offset = ((phys_block * BLOCK_SIZE + block_offset) * num_kv_heads + kv_head) * head_dim;
        const int src_offset = ((token_start + t) * num_kv_heads + kv_head) * head_dim;

        for (int d = 0; d < head_dim; d++) {
            key_cache[cache_offset + d] = new_keys[src_offset + d];
            value_cache[cache_offset + d] = new_values[src_offset + d];
        }
    }
}

__global__ void block_copy_f16_kernel(
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
