//! Grouped RMSNorm CUDA kernel for K2-Horizon.
//!
//! Operation: split hidden_dim into n_groups, normalize each group independently,
//! then apply x * inv_rms * gamma.
//!
//! Input:  [N, hidden_dim] f32
//! Gamma:  [hidden_dim] f32
//! Output: [N, hidden_dim] f32
//!
//! Grid:  (N, n_groups)
//! Block: (group_dim) threads, 1 warp (or multiple warps if group_dim > 32)

#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void groupedRmsNormKernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    float* __restrict__ output,
    int N,
    int hidden_dim,
    int n_groups,
    float eps
) {
    const int row = blockIdx.x;
    const int group = blockIdx.y;
    const int group_dim = hidden_dim / n_groups;

    if (row >= N || group >= n_groups) return;

    const float* row_input = input + row * hidden_dim;
    const float* row_gamma = gamma + group * group_dim;
    float* row_output = output + row * hidden_dim;

    // Cada thread procesa un elemento del grupo
    const int tid = threadIdx.x;
    if (tid >= group_dim) return;

    // Cargar input
    const float x = row_input[group * group_dim + tid];

    // Calcular x^2 en smem para reducción
    extern __shared__ float sdata[];
    sdata[tid] = x * x;
    __syncthreads();

    // Warp reduction para calcular sum(x^2)
    float sum_sq = sdata[tid];
    #pragma unroll
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    // Primer thread del warp calcula inv_rms
    float inv_rms = 1.0f;
    if (tid == 0) {
        const float mean_sq = sum_sq / (float)group_dim;
        inv_rms = 1.0f / sqrtf(mean_sq + eps);
    }

    // Broadcast inv_rms a todos los threads del warp
    inv_rms = __shfl_sync(0xffffffff, inv_rms, 0);

    // Aplicar normalización y gamma
    const float w = row_gamma[tid];
    row_output[group * group_dim + tid] = x * inv_rms * w;
}
