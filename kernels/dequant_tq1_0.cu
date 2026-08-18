// kernels/dequant_tq1_0.cu
#include "tables.cuh"
#include <stdint.h>

extern "C" __global__ void dequant_tq1_0_kernel(
    const uint8_t* __restrict__ raw, float* __restrict__ out, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    const int qk = 256, bs = 54;
    int nb = idx / qk, in = idx % qk;
    const uint8_t* blk = raw + nb * bs;
    float d = rdh(blk + 52);
    const uint8_t* qs = blk;
    const uint8_t* qh = blk + 48;
    uint8_t q;
    if (in < 160) { int local = in; int j = local / 5; int n = local % 5; q = (uint8_t)((uint16_t)qs[j] * pow3[n]); }
    else if (in < 240) { int local = in - 160; int j = local / 5; int n = local % 5; q = (uint8_t)((uint16_t)qs[32 + j] * pow3[n]); }
    else { int local = in - 240; int j = local / 4; int n = local % 4; q = (uint8_t)((uint16_t)qh[j] * pow3[n]); }
    int xi = (int)((q * 3) >> 8) - 1;
    out[idx] = (float)xi * d;
}

extern "C" void dequant_tq1_0_launcher(
    float* out, const uint8_t* in, int num_elements, cudaStream_t stream) {
    int threads = 256, blocks = (num_elements + threads - 1) / threads;
    dequant_tq1_0_kernel<<<blocks, threads, 0, stream>>>(in, out, num_elements);
}