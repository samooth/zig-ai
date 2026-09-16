// kernels/dequant_iq2_xs.cu
#include "tables.cuh"
#include <stdint.h>

extern "C" __global__ void dequant_iq2_xs_kernel(
    const uint8_t* __restrict__ raw, float* __restrict__ out, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    const int qk = 256, bs = 74;
    int nb = idx / qk, in = idx % qk;
    int ib = in / 32, rem = in % 32;
    int l = rem / 8, j = rem % 8;
    const uint8_t* blk = raw + nb * bs;
    float d = rdh(blk);
    const uint8_t* qs = blk + 2;
    const uint8_t* scales = blk + 66;
    float db0 = d * (0.5f + (float)(scales[ib] & 0xF)) * 0.25f;
    float db1 = d * (0.5f + (float)(scales[ib] >> 4)) * 0.25f;
    uint16_t v = rd16(qs + ib * 8 + l * 2);
    int idxg = v & 511;
    uint8_t signs = ksigns_iq2xs[v >> 9];
    uint64_t g = iq2xs_grid[idxg];
    float db = (l < 2) ? db0 : db1;
    int s = (signs & kmask_iq2xs[j]) ? -1 : 1;
    float gv = (float)(uint8_t)((g >> (8 * j)) & 0xFF);
    out[idx] = db * gv * (float)s;
}

extern "C" void dequant_iq2_xs_launcher(
    float* out, const uint8_t* in, int num_elements, cudaStream_t stream) {
    int threads = 256, blocks = (num_elements + threads - 1) / threads;
    dequant_iq2_xs_kernel<<<blocks, threads, 0, stream>>>(in, out, num_elements);
}