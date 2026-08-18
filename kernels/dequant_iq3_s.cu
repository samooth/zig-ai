// kernels/dequant_iq3_s.cu
#include "tables.cuh"
#include <stdint.h>

// IQ3_S: super-bloques de 256. Cada bloque (110 bytes):
//   d f16 (0), qs[64] (2), qh[8] (66), signs[32] (74), scales[4] (106).
//   db1 = d*(1+2*(sc&0xf)); db2 = d*(1+2*(sc>>4))
//   idx3s(q,h,l) = q | ((h << (8-2l)) & 256)
//   idx3s2(q,h,l) = q | ((h << (7-2l)) & 256)
//   val = db * byte_j(iq3s_grid[idx]) * sign
__device__ static inline uint32_t idx3s(uint8_t q, uint8_t h, int l) {
    return (uint32_t)q | (((uint32_t)h << (8 - 2 * l)) & 256u);
}
__device__ static inline uint32_t idx3s2(uint8_t q, uint8_t h, int l) {
    return (uint32_t)q | (((uint32_t)h << (7 - 2 * l)) & 256u);
}

extern "C" __global__ void dequant_iq3_s_kernel(
    const uint8_t* __restrict__ raw, float* __restrict__ out, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    const int qk = 256, bs = 110;
    int nb = idx / qk;
    int in = idx % qk;
    int it = in / 64;
    int rem = in % 64;
    int half = rem / 32;
    int pos = rem % 32;
    int l = pos / 8;       // 0..3
    int col = pos % 8;     // 0..7
    const uint8_t* blk = raw + nb * bs;
    float d = rdh(blk);
    const uint8_t* qs = blk + 2;
    const uint8_t* qh = blk + 66;
    const uint8_t* signs = blk + 74;
    const uint8_t* scales = blk + 106;
    uint8_t sc = scales[it];
    float db = (half == 0) ? d * (1.0f + 2.0f * (float)(sc & 0xF))
                           : d * (1.0f + 2.0f * (float)(sc >> 4));
    const uint8_t* q = (half == 0) ? (qs + it * 16) : (qs + it * 16 + 8);
    uint8_t hb = qh[2 * it + half];
    uint8_t sm = signs[it * 8 + half * 4 + l];
    uint32_t e = (col < 4) ? iq3s_grid[idx3s(q[2 * l], hb, l)]
                           : iq3s_grid[idx3s2(q[2 * l + 1], hb, l)];
    int j = (col < 4) ? col : (col - 4);
    int kmask = (col < 4) ? kmask_iq2xs[j] : kmask_iq2xs[j + 4];
    int s = (sm & kmask) ? -1 : 1;
    float gv = (float)(uint8_t)((e >> (8 * j)) & 0xFF);
    out[idx] = db * gv * (float)s;
}

extern "C" void dequant_iq3_s_launcher(
    float* out, const uint8_t* in, int num_elements, cudaStream_t stream) {
    int threads = 256, blocks = (num_elements + threads - 1) / threads;
    dequant_iq3_s_kernel<<<blocks, threads, 0, stream>>>(in, out, num_elements);
}