// kernels/dequant_q2_k.cu
// Q2_K dequant: super-block 256, 84 bytes
// scales[16] (4-bit scale + 4-bit min) + qs[64] (2-bit) + d f16 + dmin f16
// Layout matches CPU dequantQ2_K

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void dequant_q2_k_kernel(
    const uint8_t* __restrict__ raw,
    float* __restrict__ out,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    const int qk = 256, bs = 84;
    int nb = idx / qk;
    const uint8_t* blk = raw + nb * bs;
    float d = __half2float(*(const __half*)(blk + 80));
    float min = __half2float(*(const __half*)(blk + 82));
    const uint8_t* scales = blk;
    const uint8_t* qs = blk + 16;
    int in = idx % qk;
    int n = in / 128;  // 0 or 1 (which half)
    int in_half = in % 128;

    int group = in_half / 16;  // 0..7 (each group = 16 elements)
    int j = group / 2;         // 0..3 (scale pair index)
    int l = in_half % 16;      // 0..15 (element within group)
    int is_first = (group % 2) == 0;
    int shift = j * 2;

    int is = n * 8 + j * 2;
    uint8_t sc1 = scales[is];
    uint8_t sc2 = scales[is + 1];

    float dl = d * (float)(sc1 & 0xF);
    float ml = min * (float)(sc1 >> 4);
    float dl2 = d * (float)(sc2 & 0xF);
    float ml2 = min * (float)(sc2 >> 4);

    // qs slice for this half: n * 32 (SAME for all j, only shift changes)
    int q_base = n * 32;

    int q1 = (qs[q_base + l] >> shift) & 3;
    int q2 = (qs[q_base + l + 16] >> shift) & 3;

    float val;
    if (is_first) {
        val = dl * (float)q1 - ml;
    } else {
        val = dl2 * (float)q2 - ml2;
    }
    out[idx] = val;
}

extern "C" void dequant_q2_k_launcher(
    float* out, const uint8_t* in, int num_elements, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    dequant_q2_k_kernel<<<blocks, threads, 0, stream>>>(in, out, num_elements);
}