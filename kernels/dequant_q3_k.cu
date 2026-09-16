// kernels/dequant_q3_k.cu
// Q3_K dequant: super-block 256, 110 bytes
// hmask[32] + qs[64] (2-bit) + scales[12] (reordered to 16 i8) + d f16
// Matches llama.cpp dequantize_row_q3_K

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void dequant_q3_k_kernel(
    const uint8_t* __restrict__ raw,
    float* __restrict__ out,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    const int qk = 256, bs = 110;
    int nb = idx / qk;
    const uint8_t* blk = raw + nb * bs;
    float d = __half2float(*(const __half*)(blk + 108));
    const uint8_t* hmask = blk;
    const uint8_t* qs = blk + 32;
    const uint8_t* scales = blk + 96;
    
    // Reorder scales (same as CPU)
    uint32_t aux[4];
    uint8_t* aux_b = (uint8_t*)aux;
    aux_b[0] = scales[0]; aux_b[1] = scales[1]; aux_b[2] = scales[2]; aux_b[3] = scales[3];
    aux_b[4] = scales[4]; aux_b[5] = scales[5]; aux_b[6] = scales[6]; aux_b[7] = scales[7];
    aux_b[8] = scales[8]; aux_b[9] = scales[9]; aux_b[10] = scales[10]; aux_b[11] = scales[11];
    uint32_t tmp = aux[2];
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    int8_t scales16[16];
    for (int i = 0; i < 16; i++) scales16[i] = ((int8_t*)aux)[i];
    
    int in = idx % qk;
    int n = in / 128;  // 0 or 1 (half)
    int in_half = in % 128;
    
    // CPU writes to: n*128 + j*32 + l
    // j = in_half / 32 (0..3)
    // l = in_half % 32 (0..31)
    // For l=0..15: use q[l], hmask[l], first scale (is)
    // For l=16..31: use q[l], hmask[l], second scale (is+1)
    int j = in_half / 32;
    int l = in_half % 32;
    int is_first = (l < 16);
    int l_inner = l % 16;
    
    int is = n * 8 + j * 2 + (is_first ? 0 : 1);
    int8_t sc = scales16[is];
    float dl = d * (float)(sc - 32);
    
    int shift = j * 2;
    int q_idx = n * 32 + l;
    int q = (qs[q_idx] >> shift) & 3;
    
    int hm_idx = is_first ? l_inner : l_inner + 16;
    int bit_idx = n * 4 + j;
    int hm = (hmask[hm_idx] & (1 << bit_idx)) ? 0 : 4;
    
    out[idx] = dl * (float)(q - hm);
}

extern "C" void dequant_q3_k_launcher(
    float* out, const uint8_t* in, int num_elements, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    dequant_q3_k_kernel<<<blocks, threads, 0, stream>>>(in, out, num_elements);
}