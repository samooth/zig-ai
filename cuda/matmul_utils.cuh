#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template<int BM, int BN, int BK>
__device__ void matmul_smem_tile(
    const __half* A, const __half* B, float* C,
    int M, int N, int K
) {
    __shared__ __half As[BM * BK];
    __shared__ __half Bs[BK * BN];
    const int tid = threadIdx.x;
    const int row = tid / BN;
    const int col = tid % BN;
    float acc = 0.0f;
    for (int tile_k = 0; tile_k < K; tile_k += BK) {
        for (int i = tid; i < BM * BK; i += blockDim.x) {
            int a_row = i / BK, a_col = i % BK;
            int global_k = tile_k + a_col;
            As[i] = (a_row < M && global_k < K) ? A[a_row * K + global_k] : __float2half(0.0f);
        }
        for (int i = tid; i < BK * BN; i += blockDim.x) {
            int b_row = i / BN, b_col = i % BN;
            int global_k = tile_k + b_row;
            Bs[i] = (global_k < K && b_col < N) ? B[b_col * K + global_k] : __float2half(0.0f);
        }
        __syncthreads();
        if (row < M && col < N) {
            #pragma unroll
            for (int k = 0; k < BK; k++)
                acc += __half2float(As[row * BK + k]) * __half2float(Bs[k * BN + col]);
        }
        __syncthreads();
    }
    if (row < M && col < N) C[row * BN + col] = acc;
}
