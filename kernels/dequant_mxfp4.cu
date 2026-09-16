// kernels/dequant_mxfp4.cu
#include "tables.cuh"
#include <stdint.h>

extern "C" __global__ void dequant_mxfp4_kernel(
    const uint8_t* __restrict__ raw, float* __restrict__ out, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    const int qk = 32, bs = 17;
    int nb = idx / qk, k = idx % qk;
    const uint8_t* blk = raw + nb * bs;
    float d = e8m0Half(blk[0]);
    const uint8_t* qs = blk + 1;
    int q = (k < 16) ? (qs[k] & 0xF) : (qs[k - 16] >> 4);
    out[idx] = (float)kvalues_fp4[q] * d;
}

extern "C" void dequant_mxfp4_launcher(
    float* out, const uint8_t* in, int num_elements, cudaStream_t stream) {
    int threads = 256, blocks = (num_elements + threads - 1) / threads;
    dequant_mxfp4_kernel<<<blocks, threads, 0, stream>>>(in, out, num_elements);
}