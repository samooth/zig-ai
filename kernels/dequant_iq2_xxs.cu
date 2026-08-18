// kernels/dequant_iq2_xxs.cu
#include "tables.cuh"
#include <stdint.h>

extern "C" __global__ void dequant_iq2_xxs_kernel(
    const uint8_t* __restrict__ raw, float* __restrict__ out, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    const int qk = 256, bs = 66;
    int nb = idx / qk, in = idx % qk;
    int ib = in / 32, rem = in % 32;
    int l = rem / 8, j = rem % 8;
    const uint8_t* blk = raw + nb * bs;
    float d = rdh(blk);
    const uint8_t* qs = blk + 2;
    uint32_t aux0 = rd32(qs + ib * 8);
    uint32_t aux1 = rd32(qs + ib * 8 + 4);
    float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
    int idxg = (int)((aux0 >> (8 * l)) & 0xFF);
    uint8_t signs = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
    uint64_t g = iq2xxs_grid[idxg];
    int s = (signs & kmask_iq2xs[j]) ? -1 : 1;
    float gv = (float)(uint8_t)((g >> (8 * j)) & 0xFF);
    out[idx] = db * gv * (float)s;
}

extern "C" void dequant_iq2_xxs_launcher(
    float* out, const uint8_t* in, int num_elements, cudaStream_t stream) {
    int threads = 256, blocks = (num_elements + threads - 1) / threads;
    dequant_iq2_xxs_kernel<<<blocks, threads, 0, stream>>>(in, out, num_elements);
}