#include "online_softmax.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <assert.h>

template<int Bq, int Bkv, int D, int num_warps>
__global__ void flash_attention_v2_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    int N, int d, float scale, bool causal
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warps_per_block = num_warps;

    const int q_tile_idx = blockIdx.y;
    const int batch_head = blockIdx.x;
    const int q_global_start = q_tile_idx * Bq;
    if (q_global_start >= N) return;
    const int q_len = min(Bq, N - q_global_start);

    const int rows_per_warp = (Bq + warps_per_block - 1) / warps_per_block;
    const int q_row_start = warp_id * rows_per_warp;
    const int q_row_end = min(q_row_start + rows_per_warp, q_len);

    const int head_offset = batch_head * N * d;
    const __half* q_base = Q + head_offset;
    const __half* k_base = K + head_offset;
    const __half* v_base = V + head_offset;
    __half* o_base = O + head_offset;

    extern __shared__ char smem[];
    __half* q_smem = (__half*)smem;
    __half* k_smem = q_smem + Bq * D;
    __half* v_smem = k_smem + Bkv * D;

    for (int idx = tid; idx < Bq * D; idx += blockDim.x) {
        int row = idx / D, col = idx % D;
        int global_row = q_global_start + row;
        if (global_row < N) q_smem[idx] = q_base[global_row * d + col];
    }
    __syncthreads();

    float m[Bq]; float l[Bq]; float o_accum[Bq * D];
    #pragma unroll
    for (int i = q_row_start; i < q_row_end; i++) {
        m[i] = -CUDART_INF_F; l[i] = 0.0f;
        #pragma unroll
        for (int j = 0; j < D; j++) o_accum[i * D + j] = 0.0f;
    }

    const int num_kv_tiles = (N + Bkv - 1) / Bkv;
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        const int kv_global_start = kv_tile * Bkv;
        const int kv_len = min(Bkv, N - kv_global_start);

        for (int idx = tid; idx < Bkv * D; idx += blockDim.x) {
            int row = idx / D, col = idx % D;
            int global_row = kv_global_start + row;
            if (global_row < N) {
                k_smem[idx] = k_base[global_row * d + col];
                v_smem[idx] = v_base[global_row * d + col];
            }
        }
        __syncthreads();

        for (int q_row = q_row_start; q_row < q_row_end; q_row++) {
            const int global_q_row = q_global_start + q_row;
            float s_local[Bkv];
            for (int k_col = lane_id; k_col < kv_len; k_col += 32) {
                float dot = 0.0f;
                #pragma unroll
                for (int di = 0; di < D; di++)
                    dot += __half2float(q_smem[q_row * D + di]) * __half2float(k_smem[k_col * D + di]);
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

            float m_new = fmaxf(m[q_row], m_local);
            float alpha = expf(m[q_row] - m_new);
            float l_local = 0.0f;
            for (int k = 0; k < kv_len; k++) {
                float p = expf(s_local[k] - m_new);
                s_local[k] = p;
                l_local += p;
            }
            l_local = warp_reduce_sum(l_local);
            float l_new = l[q_row] * alpha + l_local;

            for (int di = 0; di < D; di++) {
                float o_new = o_accum[q_row * D + di] * alpha;
                for (int k = 0; k < kv_len; k++)
                    o_new += s_local[k] * __half2float(v_smem[k * D + di]);
                o_accum[q_row * D + di] = o_new;
            }
            m[q_row] = m_new; l[q_row] = l_new;
        }
        __syncthreads();
    }

    for (int q_row = q_row_start; q_row < q_row_end; q_row++) {
        int global_row = q_global_start + q_row;
        for (int di = lane_id; di < D; di += 32) {
            float val = o_accum[q_row * D + di] / l[q_row];
            o_base[global_row * d + di] = __float2half(val);
        }
    }
}

extern "C" {
    void launch_flash_attention_v2(
        const void* q, const void* k, const void* v, void* o,
        int N, int d, float scale, int causal, cudaStream_t stream
    ) {
        const int Bq = 64, Bkv = 64, D = 128, num_warps = 8;
        dim3 grid(8, (N + Bq - 1) / Bq);
        dim3 block(num_warps * 32);
        size_t smem_size = (Bq * D + 2 * Bkv * D) * sizeof(__half);
        flash_attention_v2_kernel<64, 64, 128, 8><<<grid, block, smem_size, stream>>>(
            (const __half*)q, (const __half*)k, (const __half*)v, (__half*)o,
            N, d, scale, causal);
    }
}
