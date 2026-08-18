// kernels/dequant_q4_k.cu
// Q4_K dequant: super-bloque 256, 144 bytes
// val = d*sc*q - min*sm  (ref ggml dequantize_row_q4_K)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void dequant_q4_k_kernel(
    const uint8_t* __restrict__ raw,
    float* __restrict__ out,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    const int qk = 256, bs = 144;
    int nb = idx / qk;
    const uint8_t* blk = raw + nb * bs;
    float d = __half2float(*(const __half*)(blk));
    float min = __half2float(*(const __half*)(blk + 2));
    const uint8_t* scales = blk + 4;
    const uint8_t* qs = blk + 16;
    int in = idx % qk;
    int g = in / 64;          // 0..3
    int l = in % 64;          // 0..63
    int si = 2 * g + (l < 32 ? 0 : 1);
    int sd, sm;
    if (si < 4) {
        sd = scales[si] & 63;
        sm = scales[si + 4] & 63;
    } else {
        sd = (scales[si + 4] & 0xF) | ((scales[si - 4] >> 6) << 4);
        sm = (scales[si + 4] >> 4) | ((scales[si] >> 6) << 4);
    }
    float dl = d * (float)sd;
    float ml = min * (float)sm;
    uint8_t qb = qs[g * 32 + (l % 32)];
    int qv = (l < 32) ? (qb & 0xF) : ((qb >> 4) & 0xF);
    out[idx] = dl * (float)qv - ml;
}

extern "C" void dequant_q4_k_launcher(
    float* out, const uint8_t* in, int num_elements, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    dequant_q4_k_kernel<<<blocks, threads, 0, stream>>>(in, out, num_elements);
}