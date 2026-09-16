// kernels/dequant_q8_k.cu
// Q8_K dequant: super-block 256, 292 bytes
// d f32 + qs[256] i8

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void dequant_q8_k_kernel(
    const uint8_t* __restrict__ raw,
    float* __restrict__ out,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    const int qk = 256, bs = 292;
    int nb = idx / qk;
    const uint8_t* blk = raw + nb * bs;
    float d = *(const float*)(blk);
    const int8_t* qs = (const int8_t*)(blk + 4);
    int in = idx % qk;
    out[idx] = d * (float)qs[in];
}

extern "C" void dequant_q8_k_launcher(
    float* out, const uint8_t* in, int num_elements, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    dequant_q8_k_kernel<<<blocks, threads, 0, stream>>>(in, out, num_elements);
}