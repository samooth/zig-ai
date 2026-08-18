// kernels/dequant_iq4_nl.cu
// IQ4_NL: bloques de 32. Cada bloque (18 bytes): d f16 (0), qs[16] (2).
// val = d * kvalues_iq4nl[q]   (ref: ggml dequantize_row_iq4_nl — sin signos)
#include "tables.cuh"
#include <stdint.h>

extern "C" __global__ void dequant_iq4_nl_kernel(
    const uint8_t* __restrict__ raw, float* __restrict__ out, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    const int qk = 32, bs = 18;
    int nb = idx / qk, k = idx % qk;
    const uint8_t* blk = raw + nb * bs;
    float d = rdh(blk);
    const uint8_t* qs = blk + 2;
    int q = (k < 16) ? (qs[k] & 0xF) : (qs[k - 16] >> 4);
    out[idx] = d * (float)kvalues_iq4nl[q];
}

extern "C" void dequant_iq4_nl_launcher(
    float* out, const uint8_t* in, int num_elements, cudaStream_t stream) {
    int threads = 256, blocks = (num_elements + threads - 1) / threads;
    dequant_iq4_nl_kernel<<<blocks, threads, 0, stream>>>(in, out, num_elements);
}