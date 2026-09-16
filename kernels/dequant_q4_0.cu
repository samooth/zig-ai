// kernels/dequant_q4_0.cu
// Q4_0 dequant: block 32, 18 bytes
// val = d * (nibble - 8)
// Layout: qs[0] = (elem0, elem16), qs[1] = (elem1, elem17), ..., qs[15] = (elem15, elem31)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void dequant_q4_0_kernel(
    const uint8_t* __restrict__ raw,
    float* __restrict__ out,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    const int blk_size = 32, bs = 18;
    int nb = idx / blk_size;
    const uint8_t* blk = raw + nb * bs;
    float d = __half2float(*(const __half*)(blk));
    const uint8_t* qs = blk + 2;
    int in = idx % blk_size;
    int q;
    if (in < 16) {
        // First 16 elements: low nibbles
        q = (qs[in] & 0xF) - 8;
    } else {
        // Next 16 elements: high nibbles
        q = (qs[in - 16] >> 4) - 8;
    }
    out[idx] = d * (float)q;
}

extern "C" void dequant_q4_0_launcher(
    float* out, const uint8_t* in, int num_elements, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    dequant_q4_0_kernel<<<blocks, threads, 0, stream>>>(in, out, num_elements);
}