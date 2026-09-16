// cuda/dequantize_kernels.cu
// De-cuantizacion on-the-fly de K/V cuantizados a FP16 en GPU
// Soporta: INT8 simetrico/asimetrico, INT4, Q4_0, Q8_0

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// ─── INT8 Simetrico ───
// raw: []i8, scales: []f32 (un scale por bloque), block_size: 32/64/256
__global__ void dequant_int8_symmetric_kernel(
    const int8_t* __restrict__ raw,
    const float* __restrict__ scales,
    __half* __restrict__ out,
    int num_elements,
    int block_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    int block_idx = idx / block_size;
    float scale = scales[block_idx];
    int8_t q = raw[idx];
    out[idx] = __float2half((float)q * scale);
}

// ─── INT8 Asimetrico ───
__global__ void dequant_int8_asymmetric_kernel(
    const uint8_t* __restrict__ raw,
    const float* __restrict__ scales,
    const float* __restrict__ zero_points,
    __half* __restrict__ out,
    int num_elements,
    int block_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    int block_idx = idx / block_size;
    float scale = scales[block_idx];
    float zp = zero_points[block_idx];
    float q = (float)raw[idx];
    out[idx] = __float2half((q - zp) * scale);
}

// ─── INT4 (2 valores por byte) ───
__global__ void dequant_int4_kernel(
    const uint8_t* __restrict__ raw,
    const float* __restrict__ scales,
    const float* __restrict__ zero_points,
    __half* __restrict__ out,
    int num_elements,
    int block_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 2 >= num_elements) return;

    int block_idx = (idx * 2) / block_size;
    float scale = scales[block_idx];
    float zp = zero_points[block_idx];

    uint8_t byte = raw[idx];
    float q0 = (float)(byte & 0x0F);
    float q1 = (float)((byte >> 4) & 0x0F);

    int base = idx * 2;
    out[base] = __float2half((q0 - zp) * scale);
    if (base + 1 < num_elements)
        out[base + 1] = __float2half((q1 - zp) * scale);
}

// ─── Q4_0 (GGUF) ───
// Bloque de 32: [scale_f16: 2 bytes] + [32 nibbles: 16 bytes] = 18 bytes
__global__ void dequant_q4_0_kernel(
    const uint8_t* __restrict__ raw,
    __half* __restrict__ out,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    int block_idx = idx / 32;
    int block_offset = block_idx * 18;  // 18 bytes por bloque

    // Leer scale (f16 en bytes 0-1)
    __half scale_h = *(__half*)(raw + block_offset);
    float scale = __half2float(scale_h);

    int in_block = idx % 32;
    int byte_idx = block_offset + 2 + (in_block / 2);
    uint8_t byte = raw[byte_idx];

    int nibble = (in_block % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
    int q = nibble - 8;  // Q4_0 usa offset 8

    out[idx] = __float2half((float)q * scale);
}

// ─── Q8_0 (GGUF) ───
// Bloque de 32: [scale_f16: 2 bytes] + [32 bytes] = 34 bytes
__global__ void dequant_q8_0_kernel(
    const uint8_t* __restrict__ raw,
    __half* __restrict__ out,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    int block_idx = idx / 32;
    int block_offset = block_idx * 34;  // 34 bytes por bloque

    __half scale_h = *(__half*)(raw + block_offset);
    float scale = __half2float(scale_h);

    int in_block = idx % 32;
    int8_t q = (int8_t)raw[block_offset + 2 + in_block];

    out[idx] = __float2half((float)q * scale);
}

// ─── Launcher FFI ───
extern "C" {

void launch_dequant_int8_sym(
    const void* raw, const void* scales, void* out,
    int num_elements, int block_size, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    dequant_int8_symmetric_kernel<<<blocks, threads, 0, stream>>>(
        (const int8_t*)raw, (const float*)scales, (__half*)out,
        num_elements, block_size
    );
}

void launch_dequant_int8_asym(
    const void* raw, const void* scales, const void* zero_points,
    void* out, int num_elements, int block_size, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    dequant_int8_asymmetric_kernel<<<blocks, threads, 0, stream>>>(
        (const uint8_t*)raw, (const float*)scales, (const float*)zero_points,
        (__half*)out, num_elements, block_size
    );
}

void launch_dequant_int4(
    const void* raw, const void* scales, const void* zero_points,
    void* out, int num_elements, int block_size, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_elements / 2 + threads - 1) / threads;
    dequant_int4_kernel<<<blocks, threads, 0, stream>>>(
        (const uint8_t*)raw, (const float*)scales, (const float*)zero_points,
        (__half*)out, num_elements, block_size
    );
}

void launch_dequant_q4_0(
    const void* raw, void* out, int num_elements, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    dequant_q4_0_kernel<<<blocks, threads, 0, stream>>>(
        (const uint8_t*)raw, (__half*)out, num_elements
    );
}

void launch_dequant_q8_0(
    const void* raw, void* out, int num_elements, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    dequant_q8_0_kernel<<<blocks, threads, 0, stream>>>(
        (const uint8_t*)raw, (__half*)out, num_elements
    );
}

} // extern "C"

// ═══════════════════════════════════════════════════════════════════════════════
// Dequant de tensores GGUF (pesos de modelo) — Q4_K, Q6_K, IQ4_XS, IQ3_S.
// Cada kernel descomprime UN elemento por thread, bit-exact vs
// gguf.dequantQ4_K / dequantQ6_K / dequantIq4_xs / dequantIq3_s (src/loader/gguf.zig).
// Escriben en f32 para comparación directa con la referencia CPU.
// ═══════════════════════════════════════════════════════════════════════════════

// ─── Q4_K ───
// Super-bloque 256: [d f16][dmin f16][scales 12][qs 128] = 144 bytes
// val = d*sc*q - min*sm  (ref ggml dequantize_row_q4_K)
extern "C" __global__ void dequant_q4_k_kernel(
    const uint8_t* __restrict__ raw,
    float* __restrict__ out,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    const int qk = 256, bs = 144;
    int nb = idx / qk;
    const uint8_t* blk = raw + nb * bs;
    float d = __half2float(*(const __half*)(blk));
    float min = __half2float(*(const __half*)(blk + 2));
    const uint8_t* scales = blk + 4;
    const uint8_t* qs = blk + 16;
    int in = idx % qk;
    int g = in / 64;          // 0..3
    int l = in % 64;          // 0..63
    int si = 2 * g + (l < 32 ? 0 : 1);
    int sd, sm;
    if (si < 4) {
        sd = scales[si] & 63;
        sm = scales[si + 4] & 63;
    } else {
        sd = (scales[si + 4] & 0xF) | ((scales[si - 4] >> 6) << 4);
        sm = (scales[si + 4] >> 4) | ((scales[si] >> 6) << 4);
    }
    float dl = d * (float)sd;
    float ml = min * (float)sm;
    uint8_t qb = qs[g * 32 + (l % 32)];
    int qv = (l < 32) ? (qb & 0xF) : ((qb >> 4) & 0xF);
    out[idx] = dl * (float)qv - ml;
}

// ─── Q6_K ───
// Super-bloque 256: [ql 128][qh 64][sc i8 16][d f16] = 210 bytes
// val = d * sc * (q - 32)  (ref ggml dequantize_row_q6_K)
extern "C" __global__ void dequant_q6_k_kernel(
    const uint8_t* __restrict__ raw,
    float* __restrict__ out,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    const int qk = 256, bs = 210;
    int nb = idx / qk;
    const uint8_t* blk = raw + nb * bs;
    float d = __half2float(*(const __half*)(blk + 208));
    const uint8_t* ql = blk;
    const uint8_t* qh = blk + 128;
    const int8_t* sc = (const int8_t*)(blk + 192);
    int in = idx % qk;
    int n2 = in / 128;   // 0..1
    int rem = in % 128;  // 0..127
    int quad = rem / 32; // 0..3
    int l = rem % 32;    // 0..31
    int is = l / 16;     // 0..1
    const uint8_t* ql2 = ql + n2 * 64;
    const uint8_t* qh2 = qh + n2 * 32;
    const int8_t* sc2 = sc + n2 * 8;
    int qv, scv;
    if (quad == 0) {
        qv = (ql2[l] & 0xF) | ((qh2[l] >> 0) & 3) << 4;
        scv = sc2[is + 0];
    } else if (quad == 1) {
        qv = (ql2[l + 32] & 0xF) | ((qh2[l] >> 2) & 3) << 4;
        scv = sc2[is + 2];
    } else if (quad == 2) {
        qv = (ql2[l] >> 4) | ((qh2[l] >> 4) & 3) << 4;
        scv = sc2[is + 4];
    } else {
        qv = (ql2[l + 32] >> 4) | ((qh2[l] >> 6) & 3) << 4;
        scv = sc2[is + 6];
    }
    qv -= 32;
    out[idx] = d * (float)scv * (float)qv;
}

// ─── IQ4_XS ───
// Super-bloque 256: [d f16][scales_h u16][scales_l 4][qs 128] = 136 bytes
// dl = d * (ls - 32); val = dl * kvalues_iq4nl[q]
__device__ const int8_t kvalues_iq4nl[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

extern "C" __global__ void dequant_iq4_xs_kernel(
    const uint8_t* __restrict__ raw,
    float* __restrict__ out,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    const int qk = 256, bs = 136;
    int nb = idx / qk;
    const uint8_t* blk = raw + nb * bs;
    float d = __half2float(*(const __half*)(blk));
    unsigned short scales_h = *(const unsigned short*)(blk + 2);
    const uint8_t* scales_l = blk + 4;
    const uint8_t* qs = blk + 8;
    int in = idx % qk;
    int ib = in / 32; // 0..7
    int j = in % 32;  // 0..31
    int ls = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF;
    ls |= ((scales_h >> (2 * ib)) & 3) << 4;
    float dl = d * (float)(ls - 32);
    int qidx = (j < 16) ? j : (j - 16);
    uint8_t qb = qs[ib * 16 + qidx];
    int qv = (j < 16) ? (qb & 0xF) : ((qb >> 4) & 0xF);
    out[idx] = dl * (float)kvalues_iq4nl[qv];
}

// ─── IQ3_S ───
// Super-bloque 256: [d f16][qs 64][qh 8][signs 32][scales 4] = 110 bytes
// idx grid de 9 bits; db = d * (1 + 2*sc); val = db * byte_j(iq3s_grid[idx]) * sign
__device__ const uint32_t iq3s_grid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

extern "C" __global__ void dequant_iq3_s_kernel(
    const uint8_t* __restrict__ raw,
    float* __restrict__ out,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    const int qk = 256, bs = 110;
    int nb = idx / qk;
    const uint8_t* blk = raw + nb * bs;
    float d = __half2float(*(const __half*)(blk));
    const uint8_t* qs = blk + 2;
    const uint8_t* qh = blk + 66;
    const uint8_t* signs = blk + 74;
    const uint8_t* scales = blk + 106;
    int in = idx % qk;
    int it = in / 64;     // 0..3
    int rem = in % 64;    // 0..63
    int half = rem / 32;  // 0..1
    int l = (rem % 32) / 8; // 0..3
    int j = rem % 8;      // 0..7
    int sc = scales[it];
    int scv = (half == 0) ? (sc & 0xF) : (sc >> 4);
    float db = d * (1 + 2 * (float)scv);
    const uint8_t* qq = qs + it * 16 + (half == 0 ? 0 : 8);
    int qh_h = qh[2 * it + half];
    int sm = signs[it * 8 + half * 4 + l];
    int jj = j % 4;
    uint32_t e;
    if (j < 4) {
        int qv = qq[2 * l] | ((qh_h << (8 - 2 * l)) & 256);
        e = iq3s_grid[qv];
    } else {
        int qv = qq[2 * l + 1] | ((qh_h << (7 - 2 * l)) & 256);
        e = iq3s_grid[qv];
    }
    int grid_byte = (e >> (8 * jj)) & 0xFF;
    float sgn = (sm & (1 << j)) ? -1.0f : 1.0f;
    out[idx] = db * (float)grid_byte * sgn;
}
