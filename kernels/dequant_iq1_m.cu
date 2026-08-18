// kernels/dequant_iq1_m.cu
#include "tables.cuh"
#include <stdint.h>

extern "C" __global__ void dequant_iq1_m_kernel(
    const uint8_t* __restrict__ raw, float* __restrict__ out, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    const int qk = 256, bs = 56;
    int nb = idx / qk, in = idx % qk;
    int ib = in / 32, pos = in % 32;
    int l = pos / 8, j = pos % 8;
    const uint8_t* blk = raw + nb * bs;
    const uint8_t* sc = blk;
    const uint8_t* qs = blk;
    const uint8_t* qh = blk + 32;
    // d es un f16 ensamblado desde bits dispersos de scales (ref ggml iq1_m):
    // scales se lee como 4 u16 little-endian y se reensambla el f16.
    uint16_t sc0 = rd16(sc + 0);
    uint16_t sc1 = rd16(sc + 2);
    uint16_t sc2 = rd16(sc + 4);
    uint16_t sc3 = rd16(sc + 6);
    uint16_t scale_u16 = (uint16_t)((sc0 >> 12) | ((sc1 >> 8) & 0xF0) | ((sc2 >> 4) & 0xF00) | (sc3 & 0xF000));
    float d = __half2float(*(const __half*)&scale_u16);
    uint16_t sc16 = rd16(sc + (ib / 2) * 2);
    float dl1 = d * (2.0f * (float)((sc16 >> (6 * (ib % 2) + 0)) & 7) + 1.0f);
    float dl2 = d * (2.0f * (float)((sc16 >> (6 * (ib % 2) + 3)) & 7) + 1.0f);
    const uint8_t* q0 = qs + ib * 4;
    const uint8_t* qh0 = qh + (ib / 2) * 2;
    int idxarr[4] = {
        q0[0] | ((qh0[0] << 8) & 0x700),
        q0[1] | ((qh0[0] << 4) & 0x700),
        q0[2] | ((qh0[1] << 8) & 0x700),
        q0[3] | ((qh0[1] << 4) & 0x700),
    };
    float ddarr[4] = {
        (qh0[0] & 0x08) ? -0.125f : 0.125f,
        (qh0[0] & 0x80) ? -0.125f : 0.125f,
        (qh0[1] & 0x08) ? -0.125f : 0.125f,
        (qh0[1] & 0x80) ? -0.125f : 0.125f,
    };
    int idxg = idxarr[l];
    float dd = ddarr[l];
    float dl = (l < 2) ? dl1 : dl2;
    uint64_t g = iq1s_grid[idxg];
    float gv = (float)(int8_t)((g >> (8 * j)) & 0xFF);
    out[idx] = dl * (gv + dd);
}

extern "C" void dequant_iq1_m_launcher(
    float* out, const uint8_t* in, int num_elements, cudaStream_t stream) {
    int threads = 256, blocks = (num_elements + threads - 1) / threads;
    dequant_iq1_m_kernel<<<blocks, threads, 0, stream>>>(in, out, num_elements);
}