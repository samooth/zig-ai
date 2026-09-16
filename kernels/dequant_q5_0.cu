// kernels/dequant_q5_0.cu
// Q5_0 dequant: block 32, 22 bytes
// d f16 + qh u32 + qs[16] nibbles
// high bit in qh for each element
// Layout: qs[0] = (elem0, elem16), qs[1] = (elem1, elem17), ..., qs[15] = (elem15, elem31)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void dequant_q5_0_kernel(
    const uint8_t* __restrict__ raw,
    float* __restrict__ out,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    const int blk_size = 32, bs = 22;
    int nb = idx / blk_size;
    const uint8_t* blk = raw + nb * bs;
    float d = __half2float(*(const __half*)(blk));
    // qh is at offset 2 (unaligned for uint32_t), read byte-by-byte
    uint32_t qh = (uint32_t)blk[2] | ((uint32_t)blk[3] << 8) | ((uint32_t)blk[4] << 16) | ((uint32_t)blk[5] << 24);
    const uint8_t* qs = blk + 6;
    int in = idx % blk_size;
    int q;
    if (in < 16) {
        q = qs[in] & 0xF;
    } else {
        q = qs[in - 16] >> 4;
    }
    int xh = (qh >> in) & 1;
    q = (q | (xh << 4)) - 16;
    out[idx] = d * (float)q;
}

extern "C" void dequant_q5_0_launcher(
    float* out, const uint8_t* in, int num_elements, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    dequant_q5_0_kernel<<<blocks, threads, 0, stream>>>(in, out, num_elements);
}