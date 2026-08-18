// kernels/dequant_q6_k.cu
// Q6_K dequant: super-bloque 256, 210 bytes
// val = d * sc * (q - 32)  (ref ggml dequantize_row_q6_K)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void dequant_q6_k_kernel(
    const uint8_t* __restrict__ raw,
    float* __restrict__ out,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    const int qk = 256, bs = 210;
    int nb = idx / qk;
    const uint8_t* blk = raw + nb * bs;
    float d = __half2float(*(const __half*)(blk + 208));
    const uint8_t* ql = blk;
    const uint8_t* qh = blk + 128;
    const int8_t* sc = (const int8_t*)(blk + 192);
    int in = idx % qk;
    int n2 = in / 128;   // 0..1
    int rem = in % 128;  // 0..127
    int quad = rem / 32; // 0..3
    int l = rem % 32;    // 0..31
    int is = l / 16;     // 0..1
    const uint8_t* ql2 = ql + n2 * 64;
    const uint8_t* qh2 = qh + n2 * 32;
    const int8_t* sc2 = sc + n2 * 8;
    int qv, scv;
    if (quad == 0) {
        qv = (ql2[l] & 0xF) | ((qh2[l] >> 0) & 3) << 4;
        scv = sc2[is + 0];
    } else if (quad == 1) {
        qv = (ql2[l + 32] & 0xF) | ((qh2[l] >> 2) & 3) << 4;
        scv = sc2[is + 2];
    } else if (quad == 2) {
        qv = (ql2[l] >> 4) | ((qh2[l] >> 4) & 3) << 4;
        scv = sc2[is + 4];
    } else {
        qv = (ql2[l + 32] >> 4) | ((qh2[l] >> 6) & 3) << 4;
        scv = sc2[is + 6];
    }
    qv -= 32;
    out[idx] = d * (float)scv * (float)qv;
}

extern "C" void dequant_q6_k_launcher(
    float* out, const uint8_t* in, int num_elements, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    dequant_q6_k_kernel<<<blocks, threads, 0, stream>>>(in, out, num_elements);
}