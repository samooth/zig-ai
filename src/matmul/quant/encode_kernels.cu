// Quantization (encode) CUDA kernels for matmul weights/activations —
// Phase 3 (FreeToken Technique 7: "quantization as bandwidth lever").
//
// Encodes f32 rows to GGUF wire formats on-device so weights never round-
// trip through BF16 on load. Bit-identical to the CPU encoders in
// src/kv_cache/kv_quant.zig (encodeMXFP4 / encodeQ8_0):
//   MXFP4: 17B per 32-elem block [scale u8 E8M0][qs[16] split-16];
//          value = 2^(scale−127) · kvalues_fp4[nibble]; minimal scale found
//          by doubling (no log2f ⇒ bit-identical GPU↔CPU).
//   Q8_0:  34B per 32-elem block [d:f16 LE][qs[32] i8]; d = amax/127.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// ─── MXFP4 encode (weight/activation rows) ───────────────────────────────────

// LUT FP4 canónica (misma que kv_quant.zig / layer_kernels.cu).
__device__ __constant__ int8_t d_kvalues_fp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12,
    0, -1, -2, -3, -4, -6, -8, -12
};

// One thread per 32-elem block. out layout: 17B blocks, contiguous.
extern "C" __global__ void quant_mxfp4_kernel(
    const float* __restrict__ src,
    uint8_t* __restrict__ dst,
    int num_blocks   // 32-elem blocks; total elems = num_blocks*32
) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= num_blocks) return;

    const float* blk = src + (size_t)b * 32;
    uint8_t* out = dst + (size_t)b * 17;

    // amax del bloque.
    float amax = 0.0f;
#pragma unroll
    for (int c = 0; c < 32; ++c) {
        amax = fmaxf(amax, fabsf(blk[c]));
    }

    // Escala MÍNIMA e tal que 2^(e-127)·12 >= amax (búsqueda por doblar —
    // bit-idéntica a encodeMXFP4 CPU, sin log2f/exp2f).
    uint8_t e = 127;
    float cover = 12.0f;
    while (cover < amax && e < 254) {
        e += 1;
        cover *= 2.0f;
    }
    out[0] = e;
    const float d = cover * (1.0f / 12.0f);

    // Nibbles: nearest LUT ascendente, tie → menor índice (igual que CPU:
    // solo reemplaza en diff estrictamente menor).
#pragma unroll 1
    for (int b4 = 0; b4 < 16; ++b4) out[1 + b4] = 0;
#pragma unroll 1
    for (int c = 0; c < 32; ++c) {
        const float xv = blk[c];
        uint8_t best = 0;
        float best_diff = fabsf(xv - d * (float)d_kvalues_fp4[0]);
#pragma unroll
        for (int id = 1; id < 16; ++id) {
            const float diff = fabsf(xv - d * (float)d_kvalues_fp4[id]);
            if (diff < best_diff) {
                best_diff = diff;
                best = (uint8_t)id;
            }
        }
        if (c < 16) {
            out[1 + c] = (uint8_t)((out[1 + c] & 0xF0u) | best);
        } else {
            out[1 + c - 16] |= (uint8_t)(best << 4);
        }
    }
}

// ─── Q8_0 encode (weight/activation rows) ─────────────────────────────────────

// One thread per 32-elem block. out layout: 34B blocks, contiguous.
extern "C" __global__ void quant_q8_0_kernel(
    const float* __restrict__ src,
    uint8_t* __restrict__ dst,
    int num_blocks
) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= num_blocks) return;

    const float* blk = src + (size_t)b * 32;
    uint8_t* out = dst + (size_t)b * 34;

    float amax = 0.0f;
#pragma unroll
    for (int c = 0; c < 32; ++c) {
        amax = fmaxf(amax, fabsf(blk[c]));
    }

    // d = amax/127 (0 → 1.0), f16 LE — espejo de encodeQ8_0 CPU
    // (writeF16: bits de @floatCast(f16), little-endian).
    // NOTA: store del u16 completo (bloque 34B ⇒ out siempre 2-alineado).
    // El byte-split (out[0]=bits&0xFF; out[1]=bits>>8) baja a
    // `st.global.u8 <b16-reg>` en compute_86 y ptxas pierde el byte bajo
    // (verificado empiricamente); el u16 store es correcto.
    const float d = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
    *reinterpret_cast<unsigned short*>(out) =
        __half_as_ushort(__float2half_rn(d));

    const float inv = 1.0f / d;
#pragma unroll 1
    for (int c = 0; c < 32; ++c) {
        // round-to-nearest-even vía nearbyintf (empate .5 → par, igual que
        // la conversión @round de Zig).
        const float q = nearbyintf(blk[c] * inv);
        out[2 + c] = (uint8_t)__float2int_rn(fmaxf(-127.0f, fminf(127.0f, q)));
    }
}

// ─── Launchers (Runtime API, C linkage) ──────────────────────────────────────

extern "C" void quant_mxfp4_launcher(
    const float* src, uint8_t* dst, int num_blocks, cudaStream_t stream
) {
    if (num_blocks <= 0) return;
    const int threads = 256;
    const int blocks = (num_blocks + threads - 1) / threads;
    quant_mxfp4_kernel<<<blocks, threads, 0, stream>>>(src, dst, num_blocks);
}

extern "C" void quant_q8_0_launcher(
    const float* src, uint8_t* dst, int num_blocks, cudaStream_t stream
) {
    if (num_blocks <= 0) return;
    const int threads = 256;
    const int blocks = (num_blocks + threads - 1) / threads;
    quant_q8_0_kernel<<<blocks, threads, 0, stream>>>(src, dst, num_blocks);
}
