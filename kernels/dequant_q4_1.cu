// kernels/dequant_q4_1.cu
// Q4_1 dequant: block 32, 20 bytes
// val = d * q + m
// Layout: qs[0] = (elem0, elem16), qs[1] = (elem1, elem17), ..., qs[15] = (elem15, elem31)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void dequant_q4_1_kernel(
    const uint8_t* __restrict__ raw,
    float* __restrict__ out,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    const int blk_size = 32, bs = 20;
    int nb = idx / blk_size;
    const uint8_t* blk = raw + nb * bs;
    float d = __half2float(*(const __half*)(blk));
    float m = __half2float(*(const __half*)(blk + 2));
    const uint8_t* qs = blk + 4;
    int in = idx % blk_size;
    int q;
    if (in < 16) {
        // First 16 elements: low nibbles
        q = qs[in] & 0xF;
    } else {
        // Next 16 elements: high nibbles
        q = qs[in - 16] >> 4;
    }
    out[idx] = d * (float)q + m;
}

extern "C" void dequant_q4_1_launcher(
    float* out, const uint8_t* in, int num_elements, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    dequant_q4_1_kernel<<<blocks, threads, 0, stream>>>(in, out, num_elements);
}