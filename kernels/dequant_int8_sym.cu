// kernels/dequant_int8_sym.cu
// INT8 symmetric dequant: block 64, scale f32 per block

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void dequant_int8_sym_kernel(
    const uint8_t* __restrict__ raw,
    float* __restrict__ out,
    int num_elements,
    int block_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    int blk = idx / block_size;
    int in = idx % block_size;
    float scale = *(const float*)(raw + blk * (block_size + 4));
    int8_t q = ((const int8_t*)raw)[blk * block_size + in];
    out[idx] = scale * (float)q;
}

extern "C" void dequant_int8_sym_launcher(
    float* out, const uint8_t* in, int num_elements, int block_size, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    dequant_int8_sym_kernel<<<blocks, threads, 0, stream>>>(in, out, num_elements, block_size);
}