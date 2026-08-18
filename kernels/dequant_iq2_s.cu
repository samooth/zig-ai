// kernels/dequant_iq2_s.cu
#include "tables.cuh"
#include <stdint.h>

extern "C" __global__ void dequant_iq2_s_kernel(
    const uint8_t* __restrict__ raw, float* __restrict__ out, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    const int qk = 256, bs = 82;
    int nb = idx / qk, in = idx % qk;
    int ib = in / 32, rem = in % 32;
    int l = rem / 8, j = rem % 8;
    const uint8_t* blk = raw + nb * bs;
    float d = rdh(blk);
    const uint8_t* qs = blk + 2;
    const uint8_t* signs = blk + 34;
    const uint8_t* qh = blk + 66;
    const uint8_t* scales = blk + 74;
    float db0 = d * (0.5f + (float)(scales[ib] & 0xF)) * 0.25f;
    float db1 = d * (0.5f + (float)(scales[ib] >> 4)) * 0.25f;
    int idxg = qs[ib * 4 + l] | ((qh[ib] << (8 - 2 * l)) & 0x300);
    uint64_t g = iq2s_grid[idxg];
    float db = (l < 2) ? db0 : db1;
    int s = (signs[ib * 4 + l] & kmask_iq2xs[j]) ? -1 : 1;
    float gv = (float)(uint8_t)((g >> (8 * j)) & 0xFF);
    out[idx] = db * gv * (float)s;
}

extern "C" void dequant_iq2_s_launcher(
    float* out, const uint8_t* in, int num_elements, cudaStream_t stream) {
    int threads = 256, blocks = (num_elements + threads - 1) / threads;
    dequant_iq2_s_kernel<<<blocks, threads, 0, stream>>>(in, out, num_elements);
}