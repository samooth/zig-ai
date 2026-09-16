// kernels/dequant_int4.cu
// INT4 asymmetric dequant: block 64, scale f32 + zp f32 per block, 2 values per byte

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void dequant_int4_kernel(
    const uint8_t* __restrict__ raw,
    float* __restrict__ out,
    int num_elements,
    int block_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    int blk = idx / block_size;
    int in = idx % block_size;
    const float* meta = (const float*)(raw + blk * (block_size / 2 + 8));
    float scale = meta[0];
    float zp = meta[1];
    int byte_idx = blk * (block_size / 2) + in / 2;
    uint8_t byte = raw[byte_idx];
    int nibble = (in % 2 == 0) ? (byte & 0xF) : (byte >> 4);
    out[idx] = scale * ((float)nibble - zp);
}

extern "C" void dequant_int4_launcher(
    float* out, const uint8_t* in, int num_elements, int block_size, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    dequant_int4_kernel<<<blocks, threads, 0, stream>>>(in, out, num_elements, block_size);
}