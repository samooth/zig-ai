// kernels/dequant_iq3_xxs.cu
#include "tables.cuh"
#include <stdint.h>

extern "C" __global__ void dequant_iq3_xxs_kernel(
    const uint8_t* __restrict__ raw, float* __restrict__ out, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    const int qk = 256, bs = 98;
    int nb = idx / qk, in = idx % qk;
    int ib = in / 32, pos = in % 32;
    int l = pos / 8, sub = pos % 8;
    const uint8_t* blk = raw + nb * bs;
    float d = rdh(blk);
    const uint8_t* qs = blk + 2;
    const uint8_t* ss = blk + 66;
    uint32_t aux = rd32(ss + ib * 4);
    float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
    uint8_t signs = ksigns_iq2xs[(aux >> (7 * l)) & 127];
    int j; uint64_t g; uint8_t kmask;
    if (sub < 4) { j = sub; g = iq3xxs_grid[qs[ib * 8 + 2 * l]]; kmask = kmask_iq2xs[j]; }
    else { j = sub - 4; g = iq3xxs_grid[qs[ib * 8 + 2 * l + 1]]; kmask = kmask_iq2xs[j + 4]; }
    int s = (signs & kmask) ? -1 : 1;
    float gv = (float)(uint8_t)((g >> (8 * j)) & 0xFF);
    out[idx] = db * gv * (float)s;
}

extern "C" void dequant_iq3_xxs_launcher(
    float* out, const uint8_t* in, int num_elements, cudaStream_t stream) {
    int threads = 256, blocks = (num_elements + threads - 1) / threads;
    dequant_iq3_xxs_kernel<<<blocks, threads, 0, stream>>>(in, out, num_elements);
}