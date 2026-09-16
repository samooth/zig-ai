// kernels/dequant_iq1_s.cu
#include "tables.cuh"
#include <stdint.h>

extern "C" __global__ void dequant_iq1_s_kernel(
    const uint8_t* __restrict__ raw, float* __restrict__ out, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    const int qk = 256, bs = 50;
    int nb = idx / qk, in = idx % qk;
    int ib = in / 32, rem = in % 32;
    int l = rem / 8, j = rem % 8;
    const uint8_t* blk = raw + nb * bs;
    float d = rdh(blk);
    const uint8_t* qs = blk + 2;
    const uint8_t* qh = blk + 34;
    uint16_t qhb = rd16(qh + ib * 2);
    float dl = d * (2.0f * (float)((qhb >> 12) & 7) + 1.0f);
    float dd = (qhb & 0x8000) ? -0.125f : 0.125f;
    int idxg = qs[ib * 4 + l] | (((qhb >> (3 * l)) & 7) << 8);
    uint64_t g = iq1s_grid[idxg];
    float gv = (float)(int8_t)((g >> (8 * j)) & 0xFF);
    out[idx] = dl * (gv + dd);
}

extern "C" void dequant_iq1_s_launcher(
    float* out, const uint8_t* in, int num_elements, cudaStream_t stream) {
    int threads = 256, blocks = (num_elements + threads - 1) / threads;
    dequant_iq1_s_kernel<<<blocks, threads, 0, stream>>>(in, out, num_elements);
}