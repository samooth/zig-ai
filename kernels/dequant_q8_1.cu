// kernels/dequant_q8_1.cu
// Q8_1 dequant: block 32, 36 bytes
// d f16 + m f16 + 32 int8

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void dequant_q8_1_kernel(
    const uint8_t* __restrict__ raw,
    float* __restrict__ out,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    const int blk_size = 32, bs = 36;
    int nb = idx / blk_size;
    const uint8_t* blk = raw + nb * bs;
    float d = __half2float(*(const __half*)(blk));
    float m = __half2float(*(const __half*)(blk + 2));
    const int8_t* qs = (const int8_t*)(blk + 4);
    int in = idx % blk_size;
    out[idx] = d * (float)qs[in] + m;
}

extern "C" void dequant_q8_1_launcher(
    float* out, const uint8_t* in, int num_elements, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    dequant_q8_1_kernel<<<blocks, threads, 0, stream>>>(in, out, num_elements);
}