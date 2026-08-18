// kernels/dequant_iq4_xs.cu
// IQ4_XS dequant: super-bloque 256, 136 bytes
// dl = d * (ls - 32); val = dl * kvalues_iq4nl[q]

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

__device__ const int8_t kvalues_iq4nl[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

extern "C" __global__ void dequant_iq4_xs_kernel(
    const uint8_t* __restrict__ raw,
    float* __restrict__ out,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    const int qk = 256, bs = 136;
    int nb = idx / qk;
    const uint8_t* blk = raw + nb * bs;
    float d = __half2float(*(const __half*)(blk));
    unsigned short scales_h = *(const unsigned short*)(blk + 2);
    const uint8_t* scales_l = blk + 4;
    const uint8_t* qs = blk + 8;
    int in = idx % qk;
    int ib = in / 32; // 0..7
    int j = in % 32;  // 0..31
    int ls = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF;
    ls |= ((scales_h >> (2 * ib)) & 3) << 4;
    float dl = d * (float)(ls - 32);
    int qidx = (j < 16) ? j : (j - 16);
    uint8_t qb = qs[ib * 16 + qidx];
    int qv = (j < 16) ? (qb & 0xF) : ((qb >> 4) & 0xF);
    out[idx] = dl * (float)kvalues_iq4nl[qv];
}

extern "C" void dequant_iq4_xs_launcher(
    float* out, const uint8_t* in, int num_elements, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    dequant_iq4_xs_kernel<<<blocks, threads, 0, stream>>>(in, out, num_elements);
}