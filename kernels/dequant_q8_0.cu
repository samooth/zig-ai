// kernels/dequant_q8_0.cu
// Q8_0 dequant: block 32, 34 bytes
// d f16 + 32 int8

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void dequant_q8_0_kernel(
    const uint8_t* __restrict__ raw,
    float* __restrict__ out,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    const int blk_size = 32, bs = 34;
    int nb = idx / blk_size;
    const uint8_t* blk = raw + nb * bs;
    float d = __half2float(*(const __half*)(blk));
    const int8_t* qs = (const int8_t*)(blk + 2);
    int in = idx % blk_size;
    out[idx] = d * (float)qs[in];
}

extern "C" void dequant_q8_0_launcher(
    float* out, const uint8_t* in, int num_elements, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    dequant_q8_0_kernel<<<blocks, threads, 0, stream>>>(in, out, num_elements);
}