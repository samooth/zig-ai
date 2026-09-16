// FP8 Block-Scaled Linear Kernels
// Implements per-token-group quantization and block-scaled GEMM/GEMV
// Based on FreeToken's fp8_block_linear.py patterns

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "e4m3_compat.cu"

// ─── per_token_group_quant_fp8 ───────────────────────────────────────────────
// Quantizes input [M, K] bf16/f16 to FP8 with per-token, per-128-group scales
// Output: a_fp8 [M, K] fp8, a_scales [M, K/128] fp32

extern "C" __global__ void per_token_group_quant_fp8_kernel(
    const half* __restrict__ input,   // [M, K] bf16/f16
    fp8_e4m3* __restrict__ output,    // [M, K] fp8
    float* __restrict__ scales,       // [M, K/128] fp32
    int M, int K
) {
    const int group_size = 128;
    const int num_groups = K / group_size;

    int row = blockIdx.x;
    int group = blockIdx.y;
    int tid = threadIdx.x;

    if (row >= M || group >= num_groups) return;

    const half* row_ptr = input + (size_t)row * K;
    fp8_e4m3* out_row = output + (size_t)row * K;
    float* scale_row = scales + (size_t)row * num_groups;

    int base_idx = group * group_size;

    // Phase 1: Find max abs in group (block-level reduction)
    float local_max = 0.0f;
    for (int i = tid; i < group_size; i += blockDim.x) {
        float val = __half2float(row_ptr[base_idx + i]);
        local_max = fmaxf(local_max, fabsf(val));
    }

    // Block reduction
    __shared__ float s_max[256];
    s_max[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }

    float group_max = s_max[0];
    __syncthreads();

    // Compute scale (FP8 max is 224 for normal, but we use 448 for clamp as in FreeToken)
    // Actually FP8 E4M3 max normal is 224, but FreeToken uses 448 as clamp (includes subnormals)
    float scale = (group_max > 0.0f) ? (group_max / 448.0f) : 1.0f;
    float inv_scale = 1.0f / scale;

    if (tid == 0) {
        scale_row[group] = scale;
    }
    __syncthreads();

    // Phase 2: Quantize with scale
    for (int i = tid; i < group_size; i += blockDim.x) {
        float val = __half2float(row_ptr[base_idx + i]) * inv_scale;
        // Clamp to FP8 E4M3 range [-448, 448]
        val = fmaxf(-448.0f, fminf(448.0f, val));
        out_row[base_idx + i] = fp32_to_fp8_e4m3(val);
    }
}

// ─── 2.1 (lane-f): variantes f32 del quantizer ───────────────────────────────
// El engine pasa activaciones f32 (el original lee half*). MISMA matemática:
// amax por (fila, grupo-128) → scale = amax/448 (E4M3 clamp ±448, el mismo
// convenio que el original de FreeToken) → x/scale → e4m3.

extern "C" __global__ void per_token_group_quant_fp8_f32_kernel(
    const float* __restrict__ input,  // [M, K] f32
    fp8_e4m3* __restrict__ output,     // [M, K] fp8
    float* __restrict__ scales,        // [M, K/128] fp32
    int M, int K
) {
    const int group_size = 128;
    const int num_groups = K / group_size;

    int row = blockIdx.x;
    int group = blockIdx.y;
    int tid = threadIdx.x;

    if (row >= M || group >= num_groups) return;

    const float* row_ptr = input + (size_t)row * K;
    fp8_e4m3* out_row = output + (size_t)row * K;
    float* scale_row = scales + (size_t)row * num_groups;

    int base_idx = group * group_size;

    float local_max = 0.0f;
    for (int i = tid; i < group_size; i += blockDim.x)
        local_max = fmaxf(local_max, fabsf(row_ptr[base_idx + i]));

    __shared__ float s_max[256];
    s_max[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        __syncthreads();
    }
    float group_max = s_max[0];
    __syncthreads();

    float scale = (group_max > 0.0f) ? (group_max / 448.0f) : 1.0f;
    float inv_scale = 1.0f / scale;

    if (tid == 0) scale_row[group] = scale;
    __syncthreads();

    for (int i = tid; i < group_size; i += blockDim.x) {
        float val = row_ptr[base_idx + i] * inv_scale;
        val = fmaxf(-448.0f, fminf(448.0f, val));
        out_row[base_idx + i] = fp32_to_fp8_e4m3(val);
    }
}

// ─── 2.1 (lane-f): cuantizador de PESOS [N, K] ───────────────────────────────
// gemmFp8Block/gemvSplitK exigen W_fp8 [N,K] + W_scales [N, K/128] pre-cuantizados
// (nadie lo producía). Filas = filas de salida N (layout B del gemm), grupos
// de 128 en K. MISMO kernel que las activaciones (fila = "token"): el launcher
// simplemente lo llama con M=N — se documenta aquí para el descubrimiento.

// ─── block_fp8_gemm ──────────────────────────────────────────────────────────
// Block-scaled FP8 GEMM: C = A @ B^T with block scales
// A: [M, K] fp8, a_scales: [M, K/128] fp32
// B: [N, K] fp8, b_scales: [N, K/128] fp32
// C: [M, N] fp32 (accumulate in fp32)

// Simplified CUDA core implementation for sm_80+ compatibility
// Production version would use tensor cores via WMMA/PTX
extern "C" __global__ void block_fp8_gemm_kernel(
    const fp8_e4m3* __restrict__ A,
    const float* __restrict__ a_scales,
    const fp8_e4m3* __restrict__ B,
    const float* __restrict__ b_scales,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int BLOCK_M = 64;
    const int BLOCK_N = 64;
    const int BLOCK_K = 128;

    int m_start = blockIdx.y * BLOCK_M;
    int n_start = blockIdx.x * BLOCK_N;

    int m_end = min(m_start + BLOCK_M, M);
    int n_end = min(n_start + BLOCK_N, N);

    for (int k_start = 0; k_start < K; k_start += BLOCK_K) {
        int k_end = min(k_start + BLOCK_K, K);

        for (int m = m_start + threadIdx.y; m < m_end; m += blockDim.y) {
            for (int n = n_start + threadIdx.x; n < n_end; n += blockDim.x) {
                float sum = 0.0f;
                int a_scale_idx = (size_t)m * (K / 128) + k_start / 128;
                int b_scale_idx = (size_t)n * (K / 128) + k_start / 128;
                float a_scale = a_scales[a_scale_idx];
                float b_scale = b_scales[b_scale_idx];

                for (int k = k_start; k < k_end; k++) {
                    float a_val = fp8_e4m3_to_fp32(A[(size_t)m * K + k]) * a_scale;
                    float b_val = fp8_e4m3_to_fp32(B[(size_t)n * K + k]) * b_scale;
                    sum += a_val * b_val;
                }
                atomicAdd(&C[(size_t)m * N + n], sum);
            }
        }
    }
}

// ─── block_fp8_gemv_splitk ───────────────────────────────────────────────────
// Split-K FP8 GEMV for M=1 decode
// x: [K] fp8, x_scale: [K/128] fp32
// w: [N, K] fp8, w_scales: [N, K/128] fp32
// out: [N] fp32

extern "C" __global__ void block_fp8_gemv_splitk_kernel(
    const fp8_e4m3* __restrict__ x,
    const float* __restrict__ x_scale,
    const fp8_e4m3* __restrict__ w,
    const float* __restrict__ w_scales,
    float* __restrict__ out,
    int N, int K,
    int num_splits
) {
    int split = blockIdx.y;
    int n_start = blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    int n = n_start + tid;
    if (n >= N) return;

    int k_groups = K / 128;
    int k_per_split = (k_groups + num_splits - 1) / num_splits;
    int k_start_group = split * k_per_split;
    int k_end_group = min(k_start_group + k_per_split, k_groups);

    float sum = 0.0f;

    for (int kg = k_start_group; kg < k_end_group; kg++) {
        int k_base = kg * 128;
        float x_s = x_scale[kg];
        float w_s = w_scales[(size_t)n * (K / 128) + kg];

        for (int k = 0; k < 128; k++) {
            int k_idx = k_base + k;
            if (k_idx >= K) break;
            float x_val = fp8_e4m3_to_fp32(x[k_idx]) * x_s;
            float w_val = fp8_e4m3_to_fp32(w[(size_t)n * K + k_idx]) * w_s;
            sum += x_val * w_val;
        }
    }

    // Atomic add for split-K reduction
    atomicAdd(&out[n], sum);
}

// ─── Launch helpers ──────────────────────────────────────────────────────────

extern "C" void launch_per_token_group_quant_fp8(
    const half* input, fp8_e4m3* output, float* scales,
    int M, int K, cudaStream_t stream
) {
    const int group_size = 128;
    int num_groups = K / group_size;
    dim3 grid(M, num_groups);
    dim3 block(256);
    per_token_group_quant_fp8_kernel<<<grid, block, 0, stream>>>(input, output, scales, M, K);
}

extern "C" void launch_block_fp8_gemm(
    const fp8_e4m3* A, const float* a_scales,
    const fp8_e4m3* B, const float* b_scales,
    float* C, int M, int N, int K, cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((N + 63) / 64, (M + 63) / 64);
    block_fp8_gemm_kernel<<<grid, block, 0, stream>>>(A, a_scales, B, b_scales, C, M, N, K);
}

extern "C" void launch_block_fp8_gemv_splitk(
    const fp8_e4m3* x, const float* x_scale,
    const fp8_e4m3* w, const float* w_scales,
    float* out, int N, int K, int num_splits, cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((N + 255) / 256, num_splits);
    block_fp8_gemv_splitk_kernel<<<grid, block, 0, stream>>>(x, x_scale, w, w_scales, out, N, K, num_splits);
}