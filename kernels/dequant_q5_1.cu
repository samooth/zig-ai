// kernels/dequant_q5_1.cu
// Q5_1 dequant: block 32, 24 bytes
// d f16 + m f16 + qh u32 + qs[16] nibbles
// Layout: qs[0] = (elem0, elem16), qs[1] = (elem1, elem17), ..., qs[15] = (elem15, elem31)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void dequant_q5_1_kernel(
    const uint8_t* __restrict__ raw,
    float* __restrict__ out,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    const int blk_size = 32, bs = 24;
    int nb = idx / blk_size;
    const uint8_t* blk = raw + nb * bs;
    float d = __half2float(*(const __half*)(blk));
    float m = __half2float(*(const __half*)(blk + 2));
    uint32_t qh = *(const uint32_t*)(blk + 4);
    const uint8_t* qs = blk + 8;
    int in = idx % blk_size;
    int q;
    if (in < 16) {
        q = qs[in] & 0xF;
    } else {
        q = qs[in - 16] >> 4;
    }
    int xh = (qh >> in) & 1;
    q = q | (xh << 4);
    out[idx] = d * (float)q + m;
}

extern "C" void dequant_q5_1_launcher(
    float* out, const uint8_t* in, int num_elements, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    dequant_q5_1_kernel<<<blocks, threads, 0, stream>>>(in, out, num_elements);
}