#include "online_softmax.cuh"
#include "matmul_utils.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <assert.h>

template<int Bq, int Bkv, int D>
__global__ void flash_attention_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    int N, int num_heads, float scale, bool causal
) {
    int bh = blockIdx.x;
    int q_tile = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;

    int head_offset = bh * N * D;
    const __half* q_ptr = Q + head_offset;
    const __half* k_ptr = K + head_offset;
    const __half* v_ptr = V + head_offset;
    __half* o_ptr = O + head_offset;

    extern __shared__ char smem[];
    __half* q_smem = (__half*)smem;
    __half* k_smem = q_smem + Bq * D;
    __half* v_smem = k_smem + Bkv * D;
    float* s_smem = (float*)(v_smem + Bkv * D);

    int rows_per_warp = (Bq + num_warps - 1) / num_warps;
    int q_row_start = warp_id * rows_per_warp;
    int q_row_end = min(q_row_start + rows_per_warp, Bq);

    OnlineSoftmaxState<float> state[Bq];
    float o_accum[Bq * D];

    #pragma unroll
    for (int i = q_row_start; i < q_row_end; i++) {
        state[i].init();
        #pragma unroll
        for (int d = 0; d < D; d++) o_accum[i * D + d] = 0.0f;
    }

    int q_global_start = q_tile * Bq;
    if (q_global_start >= N) return;
    int q_len = min(Bq, N - q_global_start);

    for (int idx = tid; idx < Bq * D; idx += blockDim.x) {
        int row = idx / D, col = idx % D;
        int global_row = q_global_start + row;
        if (global_row < N) q_smem[idx] = q_ptr[global_row * D + col];
    }
    __syncthreads();

    int num_kv_tiles = (N + Bkv - 1) / Bkv;
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        int kv_global_start = kv_tile * Bkv;
        int kv_len = min(Bkv, N - kv_global_start);

        for (int idx = tid; idx < Bkv * D; idx += blockDim.x) {
            int row = idx / D, col = idx % D;
            int global_row = kv_global_start + row;
            if (global_row < N) {
                k_smem[idx] = k_ptr[global_row * D + col];
                v_smem[idx] = v_ptr[global_row * D + col];
            }
        }
        __syncthreads();

        for (int q_row = q_row_start; q_row < q_row_end; q_row++) {
            if (q_row >= q_len) continue;
            int global_q_row = q_global_start + q_row;
            float s_local[Bkv];

            for (int k_col = lane_id; k_col < kv_len; k_col += 32) {
                float dot = 0.0f;
                #pragma unroll
                for (int d = 0; d < D; d++)
                    dot += __half2float(q_smem[q_row * D + d]) * __half2float(k_smem[k_col * D + d]);
                float score = dot * scale;
                int global_k_col = kv_global_start + k_col;
                if (causal && global_q_row < global_k_col) score = -CUDART_INF_F;
                s_local[k_col] = score;
            }
            for (int k_col = kv_len + lane_id; k_col < Bkv; k_col += 32)
                s_local[k_col] = -CUDART_INF_F;

            float m_local = -CUDART_INF_F;
            for (int k = 0; k < Bkv; k++) m_local = fmaxf(m_local, s_local[k]);
            m_local = warp_reduce_max(m_local);

            float m_new = fmaxf(state[q_row].m, m_local);
            float alpha = expf(state[q_row].m - m_new);
            float l_local = 0.0f;
            for (int k = 0; k < kv_len; k++) {
                float p = expf(s_local[k] - m_new);
                s_local[k] = p;
                l_local += p;
            }
            l_local = warp_reduce_sum(l_local);
            float l_new = state[q_row].l * alpha + l_local;

            for (int d = 0; d < D; d++) {
                float o_new = o_accum[q_row * D + d] * alpha;
                for (int k = 0; k < kv_len; k++)
                    o_new += s_local[k] * __half2float(v_smem[k * D + d]);
                o_accum[q_row * D + d] = o_new;
            }
            state[q_row].m = m_new;
            state[q_row].l = l_new;
        }
        __syncthreads();
    }

    for (int q_row = q_row_start; q_row < q_row_end; q_row++) {
        if (q_row >= q_len) continue;
        int global_row = q_global_start + q_row;
        for (int d = lane_id; d < D; d += 32) {
            float val = o_accum[q_row * D + d] / state[q_row].l;
            o_ptr[global_row * D + d] = __float2half(val);
        }
    }
}

extern "C" {
    void launch_flash_attention(
        const void* q, const void* k, const void* v, void* o,
        int N, int num_heads, float scale, int causal,
        int Bq, int Bkv, int D, cudaStream_t stream
    ) {
        dim3 grid(num_heads, (N + Bq - 1) / Bq);
        dim3 block(256);
        size_t smem_size = (Bq * D + 2 * Bkv * D + Bq * Bkv) * sizeof(float);
        bool launched = false;

        if (Bq == 64 && Bkv == 64 && D == 128) {
            flash_attention_kernel<64, 64, 128><<<grid, block, smem_size, stream>>>(
                (const __half*)q, (const __half*)k, (const __half*)v, (__half*)o,
                N, num_heads, scale, causal);
            launched = true;
        } else if (Bq == 64 && Bkv == 64 && D == 64) {
            flash_attention_kernel<64, 64, 64><<<grid, block, smem_size, stream>>>(
                (const __half*)q, (const __half*)k, (const __half*)v, (__half*)o,
                N, num_heads, scale, causal);
            launched = true;
        } else if (Bq == 128 && Bkv == 128 && D == 128) {
            flash_attention_kernel<128, 128, 128><<<grid, block, smem_size, stream>>>(
                (const __half*)q, (const __half*)k, (const __half*)v, (__half*)o,
                N, num_heads, scale, causal);
            launched = true;
        }

        if (!launched) {
            fprintf(stderr, "[FATAL] No kernel for Bq=%d, Bkv=%d, D=%d\n", Bq, Bkv, D);
            assert(false && "Unsupported FlashAttention config");
        }
    }
}
