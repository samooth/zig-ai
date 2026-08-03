#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>

template<typename T>
struct OnlineSoftmaxState {
    T m; T l;
    __device__ __forceinline__ void init() { m = -CUDART_INF_F; l = 0.0f; }
    __device__ __forceinline__ void update(T m_new_tile, T l_new_tile) {
        T m_new = fmaxf(m, m_new_tile);
        T alpha = expf(m - m_new);
        l = l * alpha + l_new_tile;
        m = m_new;
    }
    __device__ __forceinline__ T normalize(T o) { return o / l; }
};

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val, float* shared_mem) {
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp_id = tid / 32;
    val = warp_reduce_max(val);
    if (lane == 0) shared_mem[warp_id] = val;
    __syncthreads();
    if (warp_id == 0) {
        val = (lane < blockDim.x / 32) ? shared_mem[lane] : -CUDART_INF_F;
        val = warp_reduce_max(val);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* shared_mem) {
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp_id = tid / 32;
    val = warp_reduce_sum(val);
    if (lane == 0) shared_mem[warp_id] = val;
    __syncthreads();
    if (warp_id == 0) {
        val = (lane < blockDim.x / 32) ? shared_mem[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;
}
