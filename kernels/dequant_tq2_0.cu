// kernels/dequant_tq2_0.cu
#include "tables.cuh"
#include <stdint.h>

extern "C" __global__ void dequant_tq2_0_kernel(
    const uint8_t* __restrict__ raw, float* __restrict__ out, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    const int qk = 256, bs = 66;
    int nb = idx / qk, in = idx % qk;
    int seg = in / 128, rem = in % 128;
    int l = rem / 32, m = rem % 32;
    const uint8_t* blk = raw + nb * bs;
    float d = rdh(blk + 64);
    const uint8_t* qs = blk;
    int q = (qs[seg * 32 + m] >> (2 * l)) & 3;
    out[idx] = (float)(q - 1) * d;
}

extern "C" void dequant_tq2_0_launcher(
    float* out, const uint8_t* in, int num_elements, cudaStream_t stream) {
    int threads = 256, blocks = (num_elements + threads - 1) / threads;
    dequant_tq2_0_kernel<<<blocks, threads, 0, stream>>>(in, out, num_elements);
}