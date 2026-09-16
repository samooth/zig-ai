//! Kernels CUDA para PagedAttention sobre el memory-pool de `BlockAllocator`.
//! Layout de cada bloque físico (`block_bytes_half` en halfs):
//!   [ K region: block_size * num_kv_heads * head_dim ]
//!   [ V region: block_size * num_kv_heads * head_dim ]
//! El índice de un elemento K(t, kv_head, d) dentro del bloque físico `phys` es:
//!   base = phys * block_bytes_half
//!   k = base + (t * num_kv_heads + kv_head) * head_dim + d
//!   v = base + (block_size * num_kv_heads + t * num_kv_heads + kv_head) * head_dim + d
//! Idéntico al layout que lee la referencia CPU (`attention.zig`).
//!
//! Optimizaciones aplicadas:
//!  - Q en shared memory (tile).
//!  - Acc en shared memory (accumuladores FP32).
//!  - Online softmax (no materializa S completo).
//!  - Warp-level reduction via __shfl_xor_sync.
//!  - FP16 accumulation en FP32.
//!  - Loads vectorizados (LDST.128): 8 halfs por acceso vía float4/union.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define WARP_SIZE 32

// Permite leer 8 halfs (16 bytes = LDST.128) manteniendo acceso a cada half.
union H8 {
    float4 v;
    half h[8];
};

// Carga 8 halfs contiguos y acumula el producto punto contra sq[0..7].
__device__ __forceinline__ float dot8(
    const float* sq, const half* __restrict__ kp, int g8
) {
    H8 kv;
    kv.v = *reinterpret_cast<const float4*>(kp + g8 * 8);
    const int s = g8 * 8;
    return sq[s + 0] * __half2float(kv.h[0])
         + sq[s + 1] * __half2float(kv.h[1])
         + sq[s + 2] * __half2float(kv.h[2])
         + sq[s + 3] * __half2float(kv.h[3])
         + sq[s + 4] * __half2float(kv.h[4])
         + sq[s + 5] * __half2float(kv.h[5])
         + sq[s + 6] * __half2float(kv.h[6])
         + sq[s + 7] * __half2float(kv.h[7]);
}

// Acumula en acc[0..7] el producto de V (8 halfs) por un escalar.
__device__ __forceinline__ void axpy8(
    float* acc, const half* __restrict__ vp, int g8, float scale
) {
    H8 vv;
    vv.v = *reinterpret_cast<const float4*>(vp + g8 * 8);
    const int s = g8 * 8;
    acc[s + 0] += __half2float(vv.h[0]) * scale;
    acc[s + 1] += __half2float(vv.h[1]) * scale;
    acc[s + 2] += __half2float(vv.h[2]) * scale;
    acc[s + 3] += __half2float(vv.h[3]) * scale;
    acc[s + 4] += __half2float(vv.h[4]) * scale;
    acc[s + 5] += __half2float(vv.h[5]) * scale;
    acc[s + 6] += __half2float(vv.h[6]) * scale;
    acc[s + 7] += __half2float(vv.h[7]) * scale;
}

// Carga Q (8 halfs) en sq[0..7] y pone a cero acc[0..7].
__device__ __forceinline__ void loadQ8(
    float* sq, float* acc, const half* __restrict__ qp, int g8
) {
    H8 qv;
    qv.v = *reinterpret_cast<const float4*>(qp + g8 * 8);
    const int s = g8 * 8;
    sq[s + 0] = __half2float(qv.h[0]);
    sq[s + 1] = __half2float(qv.h[1]);
    sq[s + 2] = __half2float(qv.h[2]);
    sq[s + 3] = __half2float(qv.h[3]);
    sq[s + 4] = __half2float(qv.h[4]);
    sq[s + 5] = __half2float(qv.h[5]);
    sq[s + 6] = __half2float(qv.h[6]);
    sq[s + 7] = __half2float(qv.h[7]);
    acc[s + 0] = 0.0f;
    acc[s + 1] = 0.0f;
    acc[s + 2] = 0.0f;
    acc[s + 3] = 0.0f;
    acc[s + 4] = 0.0f;
    acc[s + 5] = 0.0f;
    acc[s + 6] = 0.0f;
    acc[s + 7] = 0.0f;
}

// Reescala 8 valores de acc[0..7] por un escalar (necesario en online softmax
// cuando cambia el máximo corriente).
__device__ __forceinline__ void scal8(
    float* acc, int g8, float scale
) {
    const int s = g8 * 8;
    acc[s + 0] *= scale;
    acc[s + 1] *= scale;
    acc[s + 2] *= scale;
    acc[s + 3] *= scale;
    acc[s + 4] *= scale;
    acc[s + 5] *= scale;
    acc[s + 6] *= scale;
    acc[s + 7] *= scale;
}

// ─── G1b' (lane-b1 2026-09-08): dequant-on-the-fly VECTORIZADO ─────────────
// Los kernels prefill f16 usan dot8/scal8/axpy8/loadQ8 (LDST.128); los
// cuantizados q8_0/q4_0/q4_k iban ELEMENTO a elemento (% y / por ítem en el
// hot loop). Estos helpers procesan GRUPOS de 32 con escala cargada una vez:
//  - q8_0: 2B escala f16 + 32×int8 (=34 B/grupo).
//  - q4_0: 2B escala f16 + 16B nibbles lo/hi (=18 B/grupo, GGUF split-16).
// La escala se comparte por todo el grupo ⇒ se carga UNA vez por (t,qgroup),
// los 8 elems del sub-grupo g8 se dequantizan multiplicando por esa escala.

// Dot de 8 elems q8_0 contra sq[0..7]. `blk` apunta al bloque de 34B del grupo
// (escala f16 en [0..2), 32 int8 en [2..34)); `byte0` = offset (0/8/16/24) dentro
// del grupo del primer elem; `g8` = índice de sq/acc (0..head_dim/8-1).
__device__ __forceinline__ float dot8_q8(
    const float* sq, const uint8_t* __restrict__ blk, int g8, int byte0
) {
    const float dsc = __half2float(*reinterpret_cast<const __half*>(blk));
    const int s = g8 * 8;
    return sq[s + 0] * (float)(int8_t)blk[2 + byte0 + 0] * dsc
         + sq[s + 1] * (float)(int8_t)blk[2 + byte0 + 1] * dsc
         + sq[s + 2] * (float)(int8_t)blk[2 + byte0 + 2] * dsc
         + sq[s + 3] * (float)(int8_t)blk[2 + byte0 + 3] * dsc
         + sq[s + 4] * (float)(int8_t)blk[2 + byte0 + 4] * dsc
         + sq[s + 5] * (float)(int8_t)blk[2 + byte0 + 5] * dsc
         + sq[s + 6] * (float)(int8_t)blk[2 + byte0 + 6] * dsc
         + sq[s + 7] * (float)(int8_t)blk[2 + byte0 + 7] * dsc;
}

// Acumula en acc[0..7] = V_q8 (8 elems) * scale. `byte0` = offset dentro del grupo.
__device__ __forceinline__ void axpy8_q8(
    float* acc, const uint8_t* __restrict__ blk, int g8, int byte0, float scale
) {
    const float dsc = __half2float(*reinterpret_cast<const __half*>(blk)) * scale;
    const int s = g8 * 8;
    acc[s + 0] += (float)(int8_t)blk[2 + byte0 + 0] * dsc;
    acc[s + 1] += (float)(int8_t)blk[2 + byte0 + 1] * dsc;
    acc[s + 2] += (float)(int8_t)blk[2 + byte0 + 2] * dsc;
    acc[s + 3] += (float)(int8_t)blk[2 + byte0 + 3] * dsc;
    acc[s + 4] += (float)(int8_t)blk[2 + byte0 + 4] * dsc;
    acc[s + 5] += (float)(int8_t)blk[2 + byte0 + 5] * dsc;
    acc[s + 6] += (float)(int8_t)blk[2 + byte0 + 6] * dsc;
    acc[s + 7] += (float)(int8_t)blk[2 + byte0 + 7] * dsc;
}

// Dot de 8 elems q4_0 contra sq[0..7]. `blk` = bloque de 18B del grupo de 32
// (2B escala f16 + 16B nibbles, GGUF layout split-16: elems [0,16)=nibbles
// bajos, [16,32)=nibbles altos). Cada elem: nibble - 8 (sesgo).
__device__ __forceinline__ float dot8_q4(
    const float* sq, const uint8_t* __restrict__ blk, int off
) {
    const float dsc = __half2float(*reinterpret_cast<const __half*>(blk));
    const int s = off * 8;
    float r = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int e = off * 8 + j; // 0..31 dentro del grupo
        const int byte = blk[2 + (e < 16 ? e : e - 16)];
        const int nib = (e < 16) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        r += sq[s + j] * (float)(nib - 8) * dsc;
    }
    return r;
}

// Acumula en acc[0..7] = V_q4 * scale.
__device__ __forceinline__ void axpy8_q4(
    float* acc, const uint8_t* __restrict__ blk, int off, float scale
) {
    const float dsc = __half2float(*reinterpret_cast<const __half*>(blk)) * scale;
    const int s = off * 8;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int e = off * 8 + j;
        const int byte = blk[2 + (e < 16 ? e : e - 16)];
        const int nib = (e < 16) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        acc[s + j] += (float)(nib - 8) * dsc;
    }
}

// Escribe 8 halfs del output desde acc[0..7] normalizado por exp_sum.
__device__ __forceinline__ void storeOut8(
    half* __restrict__ op, int g8, const float* acc, float inv_sum
) {
    H8 o;
    const int s = g8 * 8;
    o.h[0] = __float2half(acc[s + 0] * inv_sum);
    o.h[1] = __float2half(acc[s + 1] * inv_sum);
    o.h[2] = __float2half(acc[s + 2] * inv_sum);
    o.h[3] = __float2half(acc[s + 3] * inv_sum);
    o.h[4] = __float2half(acc[s + 4] * inv_sum);
    o.h[5] = __float2half(acc[s + 5] * inv_sum);
    o.h[6] = __float2half(acc[s + 6] * inv_sum);
    o.h[7] = __float2half(acc[s + 7] * inv_sum);
    *reinterpret_cast<float4*>(op + g8 * 8) = o.v;
}

extern "C" __global__ void paged_attention_decode_f16_kernel(
    half* __restrict__ out,
    const half* __restrict__ query,
    const half* __restrict__ cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    int num_seqs,
    int max_num_blocks,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size
) {
    const int seq_idx = blockIdx.x;
    const int q_head = blockIdx.y;
    if (seq_idx >= num_seqs) return;
    const int seq_len = seq_lens[seq_idx];
    if (seq_len == 0) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int block_bytes_half = block_size * num_kv_heads * head_dim * 2;
    const int q_offset = (seq_idx * num_q_heads + q_head) * head_dim;
    const bool use_vec = (head_dim % 8 == 0);

    extern __shared__ float smem[];
    float* sq = smem;              // head_dim
    float* acc = smem + head_dim;  // head_dim

    if (use_vec) {
        const int vd = head_dim / 8;
        for (int g = tid; g < vd; g += nthreads) loadQ8(sq, acc, query + q_offset, g);
    } else {
        for (int d = tid; d < head_dim; d += nthreads) {
            sq[d] = __half2float(query[q_offset + d]);
            acc[d] = 0.0f;
        }
    }
    __syncthreads();

    float max_val = -1e30f;
    float exp_sum = 0.0f;
    const float scale_factor = 1.0f / sqrtf((float)head_dim);
    const int num_blocks = (seq_len + block_size - 1) / block_size;

    for (int b = 0; b < num_blocks; b++) {
        const int phys = block_tables[seq_idx * max_num_blocks + b];
        if (phys < 0) continue;
        const int tokens_in_block = (b == num_blocks - 1) ? (seq_len - b * block_size) : block_size;
        const int base = phys * block_bytes_half;

        for (int t = 0; t < tokens_in_block; t++) {
            const int k_base = base + (t * num_kv_heads + kv_head) * head_dim;
            float partial = 0.0f;
            if (use_vec) {
                const int vd = head_dim / 8;
                for (int g = tid; g < vd; g += nthreads) partial += dot8(sq, cache + k_base, g);
            } else {
                for (int d = tid; d < head_dim; d += nthreads) {
                    partial += sq[d] * __half2float(cache[k_base + d]);
                }
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                partial += __shfl_xor_sync(0xffffffffu, partial, off);
            }
            const float score = partial * scale_factor;

            const float new_max = fmaxf(max_val, score);
            const float scale = expf(max_val - new_max);
            exp_sum *= scale;
            if (use_vec) {
                const int vd = head_dim / 8;
                for (int g = tid; g < vd; g += nthreads) scal8(acc, g, scale);
            } else {
                for (int d = tid; d < head_dim; d += nthreads) acc[d] *= scale;
            }
            max_val = new_max;
            const float exp_score = expf(score - new_max);
            exp_sum += exp_score;

            const int v_base = base + (block_size * num_kv_heads + t * num_kv_heads + kv_head) * head_dim;
            if (use_vec) {
                const int vd = head_dim / 8;
                for (int g = tid; g < vd; g += nthreads) axpy8(acc, cache + v_base, g, exp_score);
            } else {
                for (int d = tid; d < head_dim; d += nthreads) {
                    acc[d] += __half2float(cache[v_base + d]) * exp_score;
                }
            }
        }
    }

    const int out_offset = q_offset;
    const float inv_sum = 1.0f / exp_sum;
    if (use_vec) {
        const int vd = head_dim / 8;
        for (int g = tid; g < vd; g += nthreads) storeOut8(out + out_offset, g, acc, inv_sum);
    } else {
        for (int d = tid; d < head_dim; d += nthreads) {
            out[out_offset + d] = __float2half(acc[d] * inv_sum);
        }
    }
}

// ─── G1 (lane-b1 2026-09-08): flash-decoding split-K f16 ──────────────────
// El kernel paged_attention_decode_f16_kernel lanza grid (1, num_q_heads, 1)
// × block (32,1,1): con GQA (0.8B: 8 q-heads) son 8 warps en total = 256
// threads para 84 SMs — latency-bound por diseño. Split-K: N bloques por
// q-head, cada uno procesa un CHUNK de la secuencia con online-softmax
// parcial (m, l, acc) y un combine final los reduce:
//   m_global = max(m_i);  l_i' = l_i * exp(m_i - m_global);
//   out = Σ acc_i * exp(m_i - m_global) / Σ l_i'
// Paridad bit-a-bit NO garantizada vs base (orden de suma distinto) — el
// gate es paridad numérica (tol 5e-3) + greedy E2E idéntico.
//
// Layout de partials: [num_q_heads][n_splits][3 + head_dim] floats:
//   [0]=m, [1]=l, [2]=pad, [3..3+head_dim)=acc (sin normalizar).
// El buffer lo pre-alloca el host en presizeDecodeScratch (graph-safe:
// puntero ESTABLE por replay — ver trampa de reversión en
// gpu_kernels.zig:1310: el grafo congela kernelParams; buffers nuevos
// DEBEN pre-existir al capture y no reasignarse).
#define G1_SPLIT_STRIDE_FLOATS(head_dim) (3 + (head_dim))

extern "C" __global__ void paged_attention_decode_f16_split_kernel(
    float* __restrict__ partials,       // [num_q_heads][n_splits][3+head_dim]
    const half* __restrict__ query,     // [num_q_heads][head_dim] (num_seqs=1)
    const half* __restrict__ cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    int num_seqs,
    int max_num_blocks,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int n_splits,                       // gridDim.z
    int tokens_per_split_nominal        // host: chunk mínimo nominal (128);
                                        // el EFECTIVO se deriva en-kernel de
                                        // seq_lens (device — CUDA-graph-safe:
                                        // los escalares host quedan congelados
                                        // en el ejecutable capturado, ver
                                        // G1c fix lane-c 2026-09-12)
) {
    // grid (1, num_q_heads, n_splits) × block (128,1,1): seq única (decode).
    const int q_head = blockIdx.y;
    const int split = blockIdx.z;
    if (q_head >= num_q_heads) return;
    const int seq_idx = 0; // decode: 1 secuencia
    const int seq_len = seq_lens[seq_idx];
    if (seq_len == 0) return;
    // G1c fix: cobertura dinámica — ceil(seq/n_splits) cubre TODO el KV.
    // El nominal fijo 128 cubría solo 8×128=1024 y a seq mayor el resto
    // del KV se ignoraba (atención truncada ⇒ texto degenerado).
    const int need = (seq_len + n_splits - 1) / n_splits;
    const int tokens_per_split = need > tokens_per_split_nominal ? need : tokens_per_split_nominal;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int block_bytes_half = block_size * num_kv_heads * head_dim * 2;
    const int q_offset = q_head * head_dim;
    const bool use_vec = (head_dim % 8 == 0);

    // Rango de tokens de este split.
    const int tok_lo = split * tokens_per_split;
    const int tok_hi = min(seq_len, tok_lo + tokens_per_split);
    if (tok_lo >= tok_hi) {
        // Split vacío (sec más corta que n_splits*tokens_per_split):
        // m=-inf, l=0 → el combine lo ignora.
        if (tid == 0) {
            float* slot = partials + ((q_head * n_splits + split) * (3 + head_dim));
            slot[0] = -1e30f;
            slot[1] = 0.0f;
        }
        return;
    }

    extern __shared__ float smem[];
    float* sq = smem;              // head_dim
    float* acc = smem + head_dim;  // head_dim

    if (use_vec) {
        const int vd = head_dim / 8;
        for (int g = tid; g < vd; g += nthreads) loadQ8(sq, acc, query + q_offset, g);
    } else {
        for (int d = tid; d < head_dim; d += nthreads) {
            sq[d] = __half2float(query[q_offset + d]);
            acc[d] = 0.0f;
        }
    }
    __syncthreads();

    float max_val = -1e30f;
    float exp_sum = 0.0f;
    const float scale_factor = 1.0f / sqrtf((float)head_dim);

    // Online softmax sobre los tokens [tok_lo, tok_hi) del split.
    for (int tok = tok_lo; tok < tok_hi; tok++) {
        const int b = tok / block_size;
        const int t = tok - b * block_size;
        const int phys = block_tables[seq_idx * max_num_blocks + b];
        if (phys < 0) continue;
        const int base = phys * block_bytes_half;

        const int k_base = base + (t * num_kv_heads + kv_head) * head_dim;
        float partial = 0.0f;
        if (use_vec) {
            const int vd = head_dim / 8;
            for (int g = tid; g < vd; g += nthreads) partial += dot8(sq, cache + k_base, g);
        } else {
            for (int d = tid; d < head_dim; d += nthreads) {
                partial += sq[d] * __half2float(cache[k_base + d]);
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            partial += __shfl_xor_sync(0xffffffffu, partial, off);
        }
        // NOTA: exige blockDim.x múltiplo de 32 (host lanza 128 fijo).
        const float score = partial * scale_factor;

        const float new_max = fmaxf(max_val, score);
        const float scale = expf(max_val - new_max);
        exp_sum *= scale;
        if (use_vec) {
            const int vd = head_dim / 8;
            for (int g = tid; g < vd; g += nthreads) scal8(acc, g, scale);
        } else {
            for (int d = tid; d < head_dim; d += nthreads) acc[d] *= scale;
        }
        max_val = new_max;
        const float exp_score = expf(score - new_max);
        exp_sum += exp_score;

        const int v_base = base + (block_size * num_kv_heads + t * num_kv_heads + kv_head) * head_dim;
        if (use_vec) {
            const int vd = head_dim / 8;
            for (int g = tid; g < vd; g += nthreads) axpy8(acc, cache + v_base, g, exp_score);
        } else {
            for (int d = tid; d < head_dim; d += nthreads) {
                acc[d] += __half2float(cache[v_base + d]) * exp_score;
            }
        }
    }

    // Volcar (m, l, acc) del split a su slot.
    float* slot = partials + ((q_head * n_splits + split) * (3 + head_dim));
    if (tid == 0) {
        slot[0] = max_val;
        slot[1] = exp_sum;
    }
    __syncthreads(); // slot[0..1] visibles antes de que otros threads lean
    for (int d = tid; d < head_dim; d += nthreads) {
        slot[3 + d] = acc[d];
    }
}

extern "C" __global__ void paged_attention_decode_f16_split_combine_kernel(
    half* __restrict__ out,             // [num_q_heads][head_dim]
    const float* __restrict__ partials, // [num_q_heads][n_splits][3+head_dim]
    int num_q_heads,
    int head_dim,
    int n_splits
) {
    // Un bloque por q_head (reducido — sólo combina n_splits slots).
    const int q_head = blockIdx.x;
    const int tid = threadIdx.x;

    // 1) m_global y (2) normalización: recorre slots del head.
    float m_global = -1e30f;
    for (int s = 0; s < n_splits; s++) {
        const float* slot = partials + ((q_head * n_splits + s) * (3 + head_dim));
        m_global = fmaxf(m_global, slot[0]);
    }

    float inv_sum = 0.0f;
    for (int s = 0; s < n_splits; s++) {
        const float* slot = partials + ((q_head * n_splits + s) * (3 + head_dim));
        inv_sum += slot[1] * expf(slot[0] - m_global);
    }
    inv_sum = 1.0f / inv_sum;

    const int out_offset = q_head * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float o = 0.0f;
        for (int s = 0; s < n_splits; s++) {
            const float* slot = partials + ((q_head * n_splits + s) * (3 + head_dim));
            o += slot[3 + d] * expf(slot[0] - m_global);
        }
        out[out_offset + d] = __float2half(o * inv_sum);
    }
}

// Prefill batch: un bloque por (token, q_head). Computa atención causal del
// token `blockIdx.x` sobre los tokens [0..blockIdx.x] del bloque físico actual
// hacia atrás, con online softmax. Out/queries layout: [seq_len, num_q_heads, head_dim].
extern "C" __global__ void paged_attention_prefill_f16_kernel(
    half* __restrict__ out,
    const half* __restrict__ queries,
    const half* __restrict__ cache,
    const int* __restrict__ block_tables,
    int n_queries,
    int start_pos,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int causal
) {
    const int token = blockIdx.x;
    const int q_head = blockIdx.y;
    if (token >= n_queries) return;
    // Posición absoluta en la secuencia (0 para prefill single-shot; > 0 para
    // prefill en chunks): mueve la máscara causal sin cambiar el índice local
    // del query dentro del buffer `queries`.
    const int abs_token = start_pos + token;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int block_bytes_half = block_size * num_kv_heads * head_dim * 2;
    const int q_stride = num_q_heads * head_dim;
    const int q_offset = token * q_stride + q_head * head_dim;
    const bool use_vec = (head_dim % 8 == 0);

    extern __shared__ float smem[];
    float* sq = smem;              // head_dim
    float* acc = smem + head_dim;  // head_dim

    if (use_vec) {
        const int vd = head_dim / 8;
        for (int g = tid; g < vd; g += nthreads) loadQ8(sq, acc, queries + q_offset, g);
    } else {
        for (int d = tid; d < head_dim; d += nthreads) {
            sq[d] = __half2float(queries[q_offset + d]);
            acc[d] = 0.0f;
        }
    }
    __syncthreads();

    float max_val = -1e30f;
    float exp_sum = 0.0f;
    const float scale_factor = 1.0f / sqrtf((float)head_dim);
    // Causal: solo bloques que contienen tokens <= `abs_token`.
    // A5: máscara parametrizable. causal=1 → cada fila atiende [0..abs_token]
        // (comportamiento histórico); causal=0 (denoiser DFlash) → TODAS las
        // filas atienden [0..start_pos+n_queries) — bidireccional en el chunk.
        const int ctx_end = causal ? abs_token : (start_pos + gridDim.x - 1);
        const int last_block = ctx_end / block_size;

    for (int b = 0; b <= last_block; b++) {
        const int phys = block_tables[b];
        if (phys < 0) continue;
        const int tokens_in_block = (b == last_block) ? (ctx_end % block_size) + 1 : block_size;
        const int base = phys * block_bytes_half;

        for (int t = 0; t < tokens_in_block; t++) {
            const int k_base = base + (t * num_kv_heads + kv_head) * head_dim;
            float partial = 0.0f;
            if (use_vec) {
                const int vd = head_dim / 8;
                for (int g = tid; g < vd; g += nthreads) partial += dot8(sq, cache + k_base, g);
            } else {
                for (int d = tid; d < head_dim; d += nthreads) {
                    partial += sq[d] * __half2float(cache[k_base + d]);
                }
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                partial += __shfl_xor_sync(0xffffffffu, partial, off);
            }
            const float score = partial * scale_factor;

            const float new_max = fmaxf(max_val, score);
            const float scale = expf(max_val - new_max);
            exp_sum *= scale;
            if (use_vec) {
                const int vd = head_dim / 8;
                for (int g = tid; g < vd; g += nthreads) scal8(acc, g, scale);
            } else {
                for (int d = tid; d < head_dim; d += nthreads) acc[d] *= scale;
            }
            max_val = new_max;
            const float exp_score = expf(score - new_max);
            exp_sum += exp_score;

            const int v_base = base + (block_size * num_kv_heads + t * num_kv_heads + kv_head) * head_dim;
            if (use_vec) {
                const int vd = head_dim / 8;
                for (int g = tid; g < vd; g += nthreads) axpy8(acc, cache + v_base, g, exp_score);
            } else {
                for (int d = tid; d < head_dim; d += nthreads) {
                    acc[d] += __half2float(cache[v_base + d]) * exp_score;
                }
            }
        }
    }

    const int out_offset = q_offset;
    const float inv_sum = 1.0f / exp_sum;
    if (use_vec) {
        const int vd = head_dim / 8;
        for (int g = tid; g < vd; g += nthreads) storeOut8(out + out_offset, g, acc, inv_sum);
    } else {
        for (int d = tid; d < head_dim; d += nthreads) {
            out[out_offset + d] = __float2half(acc[d] * inv_sum);
        }
    }
}

// Prefill batch q8_0: idéntico a paged_attention_prefill_f16_kernel pero lee
// K/V del pool cuantizado q8_0 con dequant on-the-fly (escala f16 embebida por
// grupo de 32). Layout por bloque físico: [K: qb*34][V: qb*34].
extern "C" __global__ void paged_attention_prefill_q8_0_kernel(
    half* __restrict__ out,
    const half* __restrict__ queries,
    const uint8_t* __restrict__ cache_kv,
    const int* __restrict__ block_tables,
    int n_queries,
    int start_pos,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int causal
) {
    const int token = blockIdx.x;
    const int q_head = blockIdx.y;
    if (token >= n_queries) return;
    // Posición absoluta en la secuencia (> 0 para prefill en chunks): mueve la
    // máscara causal sin cambiar el índice local del query.
    const int abs_token = start_pos + token;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    // G1b': el q8_0 prefill vectorizado indexa por g8 (sub-grupos de 8 elems)
    // PERO t_offset (y por tanto kv_head) sigue siendo necesario para
    // localizar el K/V del token-kvhead dentro del bloque (no es un head que
    // arranque alineado a 32 cuando head_dim<32 en tests).
    const int elems_per_block = block_size * num_kv_heads * head_dim;
    const size_t k_bytes = (size_t)((elems_per_block + 31) / 32) * 34;
    const int q_stride = num_q_heads * head_dim;
    const int q_offset = token * q_stride + q_head * head_dim;
    // G1b': vectorizado — vg = grupos de 32 por head, g8 iteraciones de 8 elems.

    extern __shared__ float smem[];
    float* sq = smem;              // head_dim
    float* acc = smem + head_dim;  // head_dim

    for (int d = tid; d < head_dim; d += nthreads) {
        sq[d] = __half2float(queries[q_offset + d]);
        acc[d] = 0.0f;
    }
    __syncthreads();

    float max_val = -1e30f;
    float exp_sum = 0.0f;
    const float scale_factor = 1.0f / sqrtf((float)head_dim);
    // Causal: solo bloques que contienen tokens <= `abs_token`.
    // A5: máscara parametrizable. causal=1 → cada fila atiende [0..abs_token]
        // (comportamiento histórico); causal=0 (denoiser DFlash) → TODAS las
        // filas atienden [0..start_pos+n_queries) — bidireccional en el chunk.
        const int ctx_end = causal ? abs_token : (start_pos + gridDim.x - 1);
        const int last_block = ctx_end / block_size;

    for (int b = 0; b <= last_block; b++) {
        const int phys = block_tables[b];
        if (phys < 0) continue;
        const int tokens_in_block = (b == last_block) ? (ctx_end % block_size) + 1 : block_size;
        const uint8_t* k_data = cache_kv + (size_t)phys * (2 * k_bytes);
        const uint8_t* v_data = k_data + k_bytes;

        for (int t = 0; t < tokens_in_block; t++) {
            // Offset del K/V de este token-kvhead dentro del bloque físico:
            // t_offset = t*num_kv*head_dim + kv_head*head_dim. NO está
            // necesariamente alineado a 32 (head_dim puede ser < 32 en tests);
            // el grupo de quant de un elemento índice `be` es be/32.
            const int t_offset = t * num_kv_heads * head_dim + kv_head * head_dim;

            // G1b' vectorizado: cada thread cubre los sub-grupos de 8 elems
            // con stride nthreads. Para un g8 dado, el índice del primer elem
            // es `be0 = t_offset + g8*8`; su grupo de 32 es be0/32, y el byte
            // offset del bloque es grp*34. El sub-offset `off` dentro del
            // grupo de 32 es `(be0 & 31) / 8` — pero como g8*8 avanza de 8 en
            // 8, y NO damos por hecho alineación a 32 de t_offset, usamos el
            // byte base absoluto `(be0 & ~31)` dentro del K_data y leemos los
            // 8 int8 desde `be0 & 31`.
            const int n_g8 = head_dim / 8;
            float partial = 0.0f;
            for (int g8 = tid; g8 < n_g8; g8 += nthreads) {
                const int be0 = t_offset + g8 * 8;        // índice absoluto del 1er elem
                const int byte0 = (be0 & 31);             // offset dentro del grupo de 32
                const uint8_t* kblk = k_data + (size_t)(be0 >> 5) * 34;
                // dot8_q8 lee 8 int8 desde blk[2+byte0..2+byte0+7] (misma escala del grupo).
                partial += dot8_q8(sq, kblk, g8, byte0);
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                partial += __shfl_xor_sync(0xffffffffu, partial, off);
            }
            const float score = partial * scale_factor;

            const float new_max = fmaxf(max_val, score);
            const float scale_exp = expf(max_val - new_max);
            exp_sum *= scale_exp;
            for (int g8 = tid; g8 < n_g8; g8 += nthreads) scal8(acc, g8, scale_exp);
            max_val = new_max;
            const float exp_score = expf(score - new_max);
            exp_sum += exp_score;

            for (int g8 = tid; g8 < n_g8; g8 += nthreads) {
                const int be0 = t_offset + g8 * 8;
                const int byte0 = (be0 & 31);
                const uint8_t* vblk = v_data + (size_t)(be0 >> 5) * 34;
                axpy8_q8(acc, vblk, g8, byte0, exp_score);
            }
        }
    }

    const int out_offset = q_offset;
    const float inv_sum = 1.0f / exp_sum;
    for (int d = tid; d < head_dim; d += nthreads) {
        out[out_offset + d] = __float2half(acc[d] * inv_sum);
    }
}

extern "C" __global__ void paged_attention_prefill_q4_0_kernel(
    half* __restrict__ out,
    const half* __restrict__ queries,
    const uint8_t* __restrict__ cache_kv,
    const int* __restrict__ block_tables,
    int n_queries,
    int start_pos,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int causal
) {
    const int token = blockIdx.x;
    const int q_head = blockIdx.y;
    if (token >= n_queries) return;
    const int abs_token = start_pos + token;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int elems_per_block = block_size * num_kv_heads * head_dim;
    const size_t k_bytes = (size_t)((elems_per_block + 31) / 32) * 18; // 18 bytes per 32 elems
    const int q_stride = num_q_heads * head_dim;
    const int q_offset = token * q_stride + q_head * head_dim;
    extern __shared__ float smem[];
    float* sq = smem;
    float* acc = smem + head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        sq[d] = __half2float(queries[q_offset + d]);
        acc[d] = 0.0f;
    }
    __syncthreads();
    float max_val = -1e30f;
    float exp_sum = 0.0f;
    const float scale_factor = 1.0f / sqrtf((float)head_dim);
    const int ctx_end = causal ? abs_token : (start_pos + gridDim.x - 1);
    const int last_block = ctx_end / block_size;
    // G1b' (lane-a, A1-resto): vectorizado con el patrón g8 del q8_0 de
    // lane-b1 (@a953d63). Cada thread cubre sub-grupos de 8 elems con
    // stride nthreads. be0 = t_offset + g8*8; su grupo de quant es be0>>5
    // (bloque de 18B: 2B escala f16 + 16B nibbles split-16). byte0 = be0&31.
    // CON head_dim%8==0 (geom real 64/128 y el test hd=8), un span de 8
    // cae entero en un lado del split-16 (byte0∈{0,8}=low, {16,24}=high) y
    // dentro de UN grupo de 32 (byte0+8<=32) ⇒ 1 carga de escala f16 por
    // span (vs 1 por ELEM del escalar) y el byte de nibble se lee de 2 en 2
    // pares (u16) sin rama por elemento.
    const int n_g8 = head_dim / 8;
    for (int b = 0; b <= last_block; b++) {
        const int phys = block_tables[b];
        if (phys < 0) continue;
        const int tokens_in_block = (b == last_block) ? (ctx_end % block_size) + 1 : block_size;
        const uint8_t* k_data = cache_kv + (size_t)phys * (2 * k_bytes);
        const uint8_t* v_data = k_data + k_bytes;
        for (int t = 0; t < tokens_in_block; t++) {
            const int t_offset = t * num_kv_heads * head_dim + kv_head * head_dim;

            float partial = 0.0f;
            for (int g8 = tid; g8 < n_g8; g8 += nthreads) {
                const int be0 = t_offset + g8 * 8;   // índice absoluto del 1er elem
                const int byte0 = (be0 & 31);        // arranque dentro del grupo de 32
                const uint8_t* kblk = k_data + (size_t)(be0 >> 5) * 18;
                const float dsc = __half2float(*reinterpret_cast<const __half*>(kblk));
                const float* sq8 = sq + g8 * 8;
                // split-16 GGUF: elems [0,16) = nibbles BAJOS de bytes 2..17,
                // elems [16,32) = nibbles ALTOS de los MISMOS bytes 2..17. Un
                // span de 8 alineado cae entero en un lado ⇒ una rama por span,
                // cero ramas por elemento (vs la rama por elemento del escalar).
                if (byte0 < 16) {
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const uint8_t byt = kblk[2 + byte0 + j];
                        partial += sq8[j] * (float)((byt & 0x0F) - 8) * dsc;
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const uint8_t byt = kblk[2 + (byte0 - 16) + j];
                        partial += sq8[j] * (float)((byt >> 4) - 8) * dsc;
                    }
                }
            }
            // Cola head_dim%8≠0 (sólo geometrías de test; las reales son
            // 64/128): camino escalar de referencia para los elems restantes.
            if (head_dim % 8 != 0) {
                for (int d = n_g8 * 8 + tid; d < head_dim; d += nthreads) {
                    const int be = t_offset + d;
                    const uint8_t* kblk = k_data + (size_t)(be >> 5) * 18;
                    const float dsc = __half2float(*reinterpret_cast<const __half*>(kblk));
                    const int w = be & 31;
                    const int byt = kblk[2 + (w < 16 ? w : w - 16)];
                    const int nib = (w < 16) ? (byt & 0x0F) : ((byt >> 4) & 0x0F);
                    partial += sq[d] * (float)(nib - 8) * dsc;
                }
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                partial += __shfl_xor_sync(0xffffffffu, partial, off);
            }
            const float score = partial * scale_factor;
            const float new_max = fmaxf(max_val, score);
            const float scale_exp = expf(max_val - new_max);
            exp_sum *= scale_exp;
            for (int g8 = tid; g8 < n_g8; g8 += nthreads) scal8(acc, g8, scale_exp);
            max_val = new_max;
            const float exp_score = expf(score - new_max);
            exp_sum += exp_score;
            for (int g8 = tid; g8 < n_g8; g8 += nthreads) {
                const int be0 = t_offset + g8 * 8;
                const int byte0 = (be0 & 31);
                const uint8_t* vblk = v_data + (size_t)(be0 >> 5) * 18;
                const float dsc = __half2float(*reinterpret_cast<const __half*>(vblk)) * exp_score;
                float* a8 = acc + g8 * 8;
                if (byte0 < 16) {
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const uint8_t byt = vblk[2 + byte0 + j];
                        a8[j] += (float)((byt & 0x0F) - 8) * dsc;
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const uint8_t byt = vblk[2 + (byte0 - 16) + j];
                        a8[j] += (float)((byt >> 4) - 8) * dsc;
                    }
                }
            }
            // Cola head_dim%8≠0 (idem dot): axpy escalar de referencia.
            if (head_dim % 8 != 0) {
                for (int d = n_g8 * 8 + tid; d < head_dim; d += nthreads) {
                    const int be = t_offset + d;
                    const uint8_t* vblk = v_data + (size_t)(be >> 5) * 18;
                    const float dsc = __half2float(*reinterpret_cast<const __half*>(vblk)) * exp_score;
                    const int w = be & 31;
                    const int byt = vblk[2 + (w < 16 ? w : w - 16)];
                    const int nib = (w < 16) ? (byt & 0x0F) : ((byt >> 4) & 0x0F);
                    acc[d] += (float)(nib - 8) * dsc;
                }
            }
        }
    }
    const int out_offset = q_offset;
    const float inv_sum = 1.0f / exp_sum;
    for (int d = tid; d < head_dim; d += nthreads) {
        out[out_offset + d] = __float2half(acc[d] * inv_sum);
    }
}
extern "C" __global__ void paged_attention_prefill_q4_k_kernel(
    half* __restrict__ out,
    const half* __restrict__ queries,
    const uint8_t* __restrict__ cache_kv,
    const int* __restrict__ block_tables,
    int n_queries,
    int start_pos,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int causal
) {
    const int token = blockIdx.x;
    const int q_head = blockIdx.y;
    if (token >= n_queries) return;
    const int abs_token = start_pos + token;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int elems_per_block = block_size * num_kv_heads * head_dim;
    const int qk = 256;
    const int quant_blocks_per_block = (elems_per_block + qk - 1) / qk;
    const size_t k_bytes_per_block = (size_t)quant_blocks_per_block * 144;
    const int q_stride = num_q_heads * head_dim;
    const int q_offset = token * q_stride + q_head * head_dim;
    extern __shared__ float smem[];
    float* sq = smem;
    float* acc = smem + head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        sq[d] = __half2float(queries[q_offset + d]);
        acc[d] = 0.0f;
    }
    __syncthreads();
    float max_val = -1e30f;
    float exp_sum = 0.0f;
    const float scale_factor = 1.0f / sqrtf((float)head_dim);
    const int ctx_end = causal ? abs_token : (start_pos + gridDim.x - 1);
    const int last_block = ctx_end / block_size;
    for (int b = 0; b <= last_block; b++) {
        const int phys = block_tables[b];
        if (phys < 0) continue;
        const int tokens_in_block = (b == last_block) ? (ctx_end % block_size) + 1 : block_size;
        const uint8_t* k_data = cache_kv + (size_t)phys * (2 * k_bytes_per_block);
        const uint8_t* v_data = k_data + k_bytes_per_block;
        for (int t = 0; t < tokens_in_block; t++) {
            const int t_offset = t * num_kv_heads * head_dim + kv_head * head_dim;

            float partial = 0.0f;
            // B-a1 (lane-a): vectorizado con el patrón g8 del q4_0
            // (@2827c10). Spans de 8 elems con stride nthreads: hoistea por
            // span la carga d/min (f16×2), la rama si<4 y el cálculo
            // sd/sm — el escalar los repetía POR ELEMENTO. Invariantes del
            // layout q4_k (SB256/144B, 4 grupos g de 64 elems, nibble
            // split en l=32): un span alineado a 8 cae entero en UN
            // grupo g (64%8==0, byte0+8<=64) y en UN lado del split
            // (l64+8<=32 o l64>=32) ⇒ una sola rama por span.
            // Cola head_dim%8!=0: camino escalar de referencia.
            const int n_g8 = head_dim / 8;
            for (int g8 = tid; g8 < n_g8; g8 += nthreads) {
                const int be0 = t_offset + g8 * 8;
                const int l64 = be0 & 63;
                const uint8_t* blk = k_data + (size_t)(be0 >> 8) * 144;
                const float d_val = __half2float(*(const __half*)(blk));
                const float min_val = __half2float(*(const __half*)(blk + 2));
                const uint8_t* scales = blk + 4;
                const uint8_t* qs = blk + 16;
                const int g_idx = (be0 >> 6) & 3;
                const int si = 2 * g_idx + (l64 < 32 ? 0 : 1);
                int sd, sm;
                if (si < 4) {
                    sd = scales[si] & 63;
                    sm = scales[si + 4] & 63;
                } else {
                    sd = (scales[si + 4] & 0xF) | ((scales[si - 4] >> 6) << 4);
                    sm = (scales[si + 4] >> 4) | ((scales[si] >> 6) << 4);
                }
                const float dl = d_val * (float)sd;
                const float ml = min_val * (float)sm;
                const uint8_t* q8 = qs + g_idx * 32 + (l64 < 32 ? l64 : l64 - 32);
                const float* sq8 = sq + g8 * 8;
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    const int qv = (l64 < 32) ? (q8[j] & 0xF) : ((q8[j] >> 4) & 0xF);
                    partial += sq8[j] * (dl * (float)qv - ml);
                }
            }
            if (head_dim % 8 != 0) {
                for (int d = n_g8 * 8 + tid; d < head_dim; d += nthreads) {
                    const int be = t_offset + d;
                    const int qb = be / qk;
                    const int in = be % qk;
                    const int blk_off = qb * 144;
                    const uint8_t* blk = k_data + blk_off;
                    const float d_val = __half2float(*(const __half*)(blk));
                    const float min_val = __half2float(*(const __half*)(blk + 2));
                    const uint8_t* scales = blk + 4;
                    const uint8_t* qs = blk + 16;
                    const int g_idx = in / 64;
                    const int l = in % 64;
                    const int si = 2 * g_idx + (l < 32 ? 0 : 1);
                    int sd, sm;
                    if (si < 4) {
                        sd = scales[si] & 63;
                        sm = scales[si + 4] & 63;
                    } else {
                        sd = (scales[si + 4] & 0xF) | ((scales[si - 4] >> 6) << 4);
                        sm = (scales[si + 4] >> 4) | ((scales[si] >> 6) << 4);
                    }
                    const float dl = d_val * (float)sd;
                    const float ml = min_val * (float)sm;
                    const uint8_t qb_val = qs[g_idx * 32 + (l % 32)];
                    const int qv = (l < 32) ? (qb_val & 0xF) : ((qb_val >> 4) & 0xF);
                    partial += sq[d] * (dl * (float)qv - ml);
                }
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                partial += __shfl_xor_sync(0xffffffffu, partial, off);
            }
            const float score = partial * scale_factor;
            const float new_max = fmaxf(max_val, score);
            const float scale_exp = expf(max_val - new_max);
            exp_sum *= scale_exp;
            for (int d = tid; d < head_dim; d += nthreads) acc[d] *= scale_exp;
            max_val = new_max;
            const float exp_score = expf(score - new_max);
            exp_sum += exp_score;
            // B-a1 (lane-a): V-axpy g8 — mismo patrón que el K-dot de
            // arriba (span de 8: escala/sd/sm/rama hoisteados, unroll).
            for (int g8 = tid; g8 < n_g8; g8 += nthreads) {
                const int be0 = t_offset + g8 * 8;
                const int l64 = be0 & 63;
                const uint8_t* blk = v_data + (size_t)(be0 >> 8) * 144;
                const float d_val = __half2float(*(const __half*)(blk));
                const float min_val = __half2float(*(const __half*)(blk + 2));
                const uint8_t* scales = blk + 4;
                const uint8_t* qs = blk + 16;
                const int g_idx = (be0 >> 6) & 3;
                const int si = 2 * g_idx + (l64 < 32 ? 0 : 1);
                int sd, sm;
                if (si < 4) {
                    sd = scales[si] & 63;
                    sm = scales[si + 4] & 63;
                } else {
                    sd = (scales[si + 4] & 0xF) | ((scales[si - 4] >> 6) << 4);
                    sm = (scales[si + 4] >> 4) | ((scales[si] >> 6) << 4);
                }
                const float dl = d_val * (float)sd;
                const float ml = min_val * (float)sm;
                const uint8_t* q8 = qs + g_idx * 32 + (l64 < 32 ? l64 : l64 - 32);
                float* acc8 = acc + g8 * 8;
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    const int qv = (l64 < 32) ? (q8[j] & 0xF) : ((q8[j] >> 4) & 0xF);
                    acc8[j] += (dl * (float)qv - ml) * exp_score;
                }
            }
            if (head_dim % 8 != 0) {
                for (int d = n_g8 * 8 + tid; d < head_dim; d += nthreads) {
                    const int be = t_offset + d;
                    const int qb = be / qk;
                    const int in = be % qk;
                    const int blk_off = qb * 144;
                    const uint8_t* blk = v_data + blk_off;
                    const float d_val = __half2float(*(const __half*)(blk));
                    const float min_val = __half2float(*(const __half*)(blk + 2));
                    const uint8_t* scales = blk + 4;
                    const uint8_t* qs = blk + 16;
                    const int g_idx = in / 64;
                    const int l = in % 64;
                    const int si = 2 * g_idx + (l < 32 ? 0 : 1);
                    int sd, sm;
                    if (si < 4) {
                        sd = scales[si] & 63;
                        sm = scales[si + 4] & 63;
                    } else {
                        sd = (scales[si + 4] & 0xF) | ((scales[si - 4] >> 6) << 4);
                        sm = (scales[si + 4] >> 4) | ((scales[si] >> 6) << 4);
                    }
                    const float dl = d_val * (float)sd;
                    const float ml = min_val * (float)sm;
                    const uint8_t qb_val = qs[g_idx * 32 + (l % 32)];
                    const int qv = (l < 32) ? (qb_val & 0xF) : ((qb_val >> 4) & 0xF);
                    acc[d] += (dl * (float)qv - ml) * exp_score;
                }
            }
        }
    }
    const int out_offset = q_offset;
    const float inv_sum = 1.0f / exp_sum;
    for (int d = tid; d < head_dim; d += nthreads) {
        out[out_offset + d] = __float2half(acc[d] * inv_sum);
    }
}
extern "C" __global__ void reshape_and_block_write_f16_kernel(
    half* __restrict__ cache,
    const half* __restrict__ new_keys,
    const half* __restrict__ new_values,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int* __restrict__ token_offsets,
    int num_seqs,
    int max_num_blocks,
    int num_kv_heads,
    int head_dim,
    int block_size
) {
    const int seq_idx = blockIdx.x;
    if (seq_idx >= num_seqs) return;

    const int seq_len = seq_lens[seq_idx];
    const int token_start = token_offsets[seq_idx];
    const int num_new_tokens = seq_len - token_start;
    const int kv_dim = num_kv_heads * head_dim;
    const int block_bytes_half = block_size * kv_dim * 2;

    const int total = num_new_tokens * kv_dim;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int t = idx / kv_dim;
        const int kv = idx % kv_dim;
        const int kv_head = kv / head_dim;
        const int d = kv % head_dim;
        const int global_token = token_start + t;
        const int block_idx = global_token / block_size;
        const int block_offset = global_token % block_size;
        const int phys = block_tables[seq_idx * max_num_blocks + block_idx];
        if (phys < 0) continue;

        const int base = phys * block_bytes_half;
        const int src = global_token * kv_dim + kv;
        const int k_idx = base + (block_offset * num_kv_heads + kv_head) * head_dim + d;
        const int v_idx = base + (block_size * num_kv_heads + block_offset * num_kv_heads + kv_head) * head_dim + d;
        cache[k_idx] = new_keys[src];
        cache[v_idx] = new_values[src];
    }
}
 
// ════════════════════════════════════════════════════════════════════════════════
// Fused q8_0 Paged Attention Decode Kernel (reescrito).
// Dequantiza K/V q8_0 on-the-fly durante la atención (online softmax, mismo
// patrón que los kernels fusionados de K-quants). Layout por bloque físico:
//   [K: qb*34 bytes][V: qb*34 bytes], qb = ceil(block_size*n_kv*hd/32)
//   cada grupo de 32 elems: [escala f16 embebida @0..1][32×int8 @2..33].
// k_scales/v_scales quedan en la firma por compatibilidad con el launcher
// unificado, pero NO se usan: la escala va embebida en el bloque.
extern "C" __global__ void paged_attention_decode_q8_0_kernel(
    half* __restrict__ out,
    const half* __restrict__ query,
    const uint8_t* __restrict__ cache_kv,
    const half* __restrict__ k_scales,
    const half* __restrict__ v_scales,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    int num_seqs,
    int max_num_blocks,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size
) {
    (void)k_scales;   // escala embebida por bloque de 32
    (void)v_scales;
    const int seq_idx = blockIdx.x;
    const int q_head  = blockIdx.y;
    if (seq_idx >= num_seqs) return;
    const int seq_len = seq_lens[seq_idx];
    if (seq_len == 0) return;

    const int tid      = threadIdx.x;
    const int nthreads = blockDim.x;
    const int kv_head  = q_head / (num_q_heads / num_kv_heads);
    const int elems_per_block = block_size * num_kv_heads * head_dim;
    const size_t k_bytes = (size_t)((elems_per_block + 31) / 32) * 34;
    const int q_offset       = (seq_idx * num_q_heads + q_head) * head_dim;
    const int kv_head_stride = num_kv_heads * head_dim;

    extern __shared__ float smem[];
    float* sq  = smem;
    float* acc = smem + head_dim;
    for (int e = tid; e < head_dim; e += nthreads) {
        sq[e]  = __half2float(query[q_offset + e]);
        acc[e] = 0.0f;
    }
    __syncthreads();

    float max_val = -1e30f;
    float exp_sum = 0.0f;
    const float scale_factor = 1.0f / sqrtf((float)head_dim);
    const int num_blocks = (seq_len + block_size - 1) / block_size;

    for (int b = 0; b < num_blocks; b++) {
        const int phys = block_tables[seq_idx * max_num_blocks + b];
        if (phys < 0) continue;
        const int tokens_in_block = (b == num_blocks - 1)
            ? (seq_len - b * block_size) : block_size;
        const uint8_t* k_data = cache_kv + (size_t)phys * (2 * k_bytes);
        const uint8_t* v_data = k_data + k_bytes;

        for (int t = 0; t < tokens_in_block; t++) {
            const int t_offset = t * kv_head_stride + kv_head * head_dim;
            float partial = 0.0f;
            for (int e = tid; e < head_dim; e += nthreads) {
                const int be = t_offset + e;
                const uint8_t* blk = k_data + (size_t)(be >> 5) * 34;
                const float dsc = __half2float(*reinterpret_cast<const __half*>(blk));
                partial += sq[e] * (float)(int8_t)blk[2 + (be & 31)] * dsc;
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                partial += __shfl_xor_sync(0xffffffffu, partial, off);
            const float score    = partial * scale_factor;
            const float new_max  = fmaxf(max_val, score);
            const float rescale  = expf(max_val - new_max);
            const float exp_score = expf(score - new_max);
            exp_sum = exp_sum * rescale + exp_score;
            for (int e = tid; e < head_dim; e += nthreads) acc[e] *= rescale;
            max_val = new_max;
            for (int e = tid; e < head_dim; e += nthreads) {
                const int be = t_offset + e;
                const uint8_t* blk = v_data + (size_t)(be >> 5) * 34;
                const float dsc = __half2float(*reinterpret_cast<const __half*>(blk));
                acc[e] += (float)(int8_t)blk[2 + (be & 31)] * dsc * exp_score;
            }
        }
    }
    const float inv_sum = 1.0f / exp_sum;
    for (int e = tid; e < head_dim; e += nthreads)
        out[q_offset + e] = __float2half(acc[e] * inv_sum);
}
// ═════════════════════════════════════════════════════════════════════════════════
// 3.3 (lane-f): Fused q8_0 Paged Attention Decode — VARIANTE dp4a.
// El kernel q8_0 base dequantiza K byte-a-byte + FMA f32 por elemento
// (1 dot/instr). Aquí Q se cuantiza UNA VEZ por launch a q8_0 en smem
// (mismo layout de grupo-32 del pool: [escala f16][32 quanta int8]) y el
// dot Q·K corre por dp4a: 4 int8-mults por instrucción, K leída como u32
// (4 quanta) — el layout q8_0 del pool ES dp4a-natural (2..33 de 34B).
// Dot: head_dim/32 grupos; cada grupo g es propiedad del thread g%32 con
// 8 dp4a (2 u32 de Q × 2 u32 de K por lado), escalas d_q[g]·d_k aplicadas
// al parcial; reduce-warp sólo suma los parciales por grupo (1 butterfly).
// V-side y softmax: idénticos al base (dequant + FMA exp_score — no hay
// dot entero que exprimir ahí).
// smem extra (f32 units, extra_smem_floats del mapa Zig):
//   ceil(head_dim/32) escalas d_q  +  ceil(head_dim/32)*8 u32 de Q-cuantizada
// = qb*9 f32 units.
// A/B: seleccionado por env ZIG_AI_PADP4A=1 en deviceDecodeKernel (default
// OFF); paridad esperada rel < 1e-2 (la cuantización q8 de Q añade ruido
// ~0.4% — dentro del margen del formato).
// ═════════════════════════════════════════════════════════════════════════════════
extern "C" __global__ void paged_attention_decode_q8_0_dp4a_kernel(
    half* __restrict__ out,
    const half* __restrict__ query,
    const uint8_t* __restrict__ cache_kv,
    const half* __restrict__ k_scales,
    const half* __restrict__ v_scales,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    int num_seqs,
    int max_num_blocks,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size
) {
    (void)k_scales;   // escalas embebidas por grupo de 32 (contrato C1)
    (void)v_scales;
    const int seq_idx = blockIdx.x;
    const int q_head  = blockIdx.y;
    if (seq_idx >= num_seqs) return;
    const int seq_len = seq_lens[seq_idx];
    if (seq_len == 0) return;

    const int tid      = threadIdx.x;
    const int nthreads = blockDim.x;   // 32: 1 warp por (seq, q_head)
    const int kv_head  = q_head / (num_q_heads / num_kv_heads);
    const int elems_per_block = block_size * num_kv_heads * head_dim;
    const size_t k_bytes = (size_t)((elems_per_block + 31) / 32) * 34;
    const int q_offset       = (seq_idx * num_q_heads + q_head) * head_dim;
    const int kv_head_stride = num_kv_heads * head_dim;
    const int qb_head = (head_dim + 31) / 32;   // grupos de 32 del head_dim

    // smem: [sq f32 × head_dim][acc f32 × head_dim][d_q f32 × qb_head][pad 16B][q8_q u32 × qb_head*8]
    extern __shared__ float smem[];
    float* sq  = smem;
    float* acc = smem + head_dim;
    float* d_q = acc + head_dim;
    // q8_q alineado a 16B: los u32-loads de Q/K exigen alineación; el pad
    // (≤3 f32) lo cubre extra_smem_floats del mapa Zig (qb_head*9 + 4).
    unsigned int* q8_q = reinterpret_cast<unsigned int*>(((uintptr_t)(d_q + qb_head) + 15ull) & ~15ull);

    // ── Fase 0: Q f16 → q8_0 en smem (una vez por launch) ──
    // Cada grupo de 32 elems del head_dim lo cuantiza UN warp (todos los
    // threads participan en amax-shuffle); grupo g lo procesa warp g%1
    // (somos UN warp: g por round-robin tid).
    for (int g = 0; g < qb_head; ++g) {
        const int base_e = g * 32;
        float amax = 0.0f;
        if (base_e + tid < head_dim) {
            const float qv = __half2float(query[q_offset + base_e + tid]);
            sq[base_e + tid] = qv;
            amax = fabsf(qv);
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d = (amax > 0.0f) ? amax / 127.0f : 1.0f;
        if (tid == 0) d_q[g] = d;
        // Cola head_dim%32: quanta=0 en los lanes inválidos ⇒ el término
        // 0×k anula la contribución del K de fuera-del-head en el dp4a
        // (hd no múltiplo de 32, ej. hd=72 con grupo 3 en lanes 8..31).
        if (base_e + tid < head_dim) {
            int q = (int)roundf(sq[base_e + tid] / d);
            q = max(-127, min(127, q));
            // byte lane del quanta dentro del grupo: [2..33] del registro de 34B
            reinterpret_cast<signed char*>(q8_q)[g * 32 + tid] = (signed char)q;
            acc[base_e + tid] = 0.0f;
        } else if (tid < 32) {
            reinterpret_cast<signed char*>(q8_q)[g * 32 + tid] = (signed char)0;
        }
    }
    __syncwarp();

    float max_val = -1e30f;
    float exp_sum = 0.0f;
    const float scale_factor = 1.0f / sqrtf((float)head_dim);
    const int num_blocks = (seq_len + block_size - 1) / block_size;

    // Grupos del head_dim repartidos round-robin entre los 32 threads.
    const int my_groups = (qb_head + nthreads - 1) / nthreads;

    for (int b = 0; b < num_blocks; ++b) {
        const int phys = block_tables[seq_idx * max_num_blocks + b];
        if (phys < 0) continue;
        const int tokens_in_block = (b == num_blocks - 1)
            ? (seq_len - b * block_size) : block_size;
        const uint8_t* k_data = cache_kv + (size_t)phys * (2 * k_bytes);
        const uint8_t* v_data = k_data + k_bytes;

        for (int t = 0; t < tokens_in_block; ++t) {
            const int t_off = t * kv_head_stride + kv_head * head_dim;

            // ── Dot Q·K con dp4a: thread dueño del grupo acumula 8 dp4a ──
            float partial = 0.0f;
            for (int mg = 0; mg < my_groups; ++mg) {
                const int g = mg * nthreads + tid;
                if (g >= qb_head) break;
                const int be = t_off + g * 32;
                const uint8_t* blk = k_data + (size_t)(be >> 5) * 34;
                const float d_k = __half2float(*reinterpret_cast<const __half*>(blk));
                // 32 quanta de K = bytes [2..33] del registro de 34B. El
                // registro está a offset múltiplo de 34 ⇒ NO alineado a 4B:
                // cargar como u32 desalineada = misaligned-address (716).
                // Pack manual por bytes (4 LDG.U8 + 3 BFE/PRMT por u32) —
                // el dp4a sigue dando 4 int8-mults/instr.
                const uint8_t* kq8 = blk + 2;
                const unsigned int* qq = q8_q + g * 8;
                int dot = 0;
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    unsigned int kw = (unsigned int)kq8[u * 4]
                        | ((unsigned int)kq8[u * 4 + 1] << 8)
                        | ((unsigned int)kq8[u * 4 + 2] << 16)
                        | ((unsigned int)kq8[u * 4 + 3] << 24);
                    dot = __dp4a((int)kw, (int)qq[u], dot);
                }
                partial += (float)dot * (d_q[g] * d_k);
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                partial += __shfl_xor_sync(0xffffffffu, partial, off);
            const float score = partial * scale_factor;
            const float new_max = fmaxf(max_val, score);
            const float rescale = expf(max_val - new_max);
            const float exp_score = expf(score - new_max);
            exp_sum = exp_sum * rescale + exp_score;
            for (int e = tid; e < head_dim; e += nthreads) acc[e] *= rescale;
            max_val = new_max;
            // V-side: dequant + FMA (igual que el base).
            for (int e = tid; e < head_dim; e += nthreads) {
                const int be = t_off + e;
                const uint8_t* blk = v_data + (size_t)(be >> 5) * 34;
                const float dsc = __half2float(*reinterpret_cast<const __half*>(blk));
                acc[e] += (float)(int8_t)blk[2 + (be & 31)] * dsc * exp_score;
            }
        }
    }
    const float inv_sum = 1.0f / exp_sum;
    for (int e = tid; e < head_dim; e += nthreads)
        out[q_offset + e] = __float2half(acc[e] * inv_sum);
}
// ═════════════════════════════════════════════════════════════════════════════════
// Fused q4_0 Paged Attention Decode Kernel
// Dequantizes q4_0 K/V on-the-fly during attention computation.
// Each 32-element block: [scale_f16: 2 bytes][16 bytes nibbles] = 18 bytes
// nibble - 8 offset for q4_0 format
// ════════════════════════════════════════════════════════════════════════════════
#include <stdint.h>

extern "C" __global__ void paged_attention_decode_q4_0_kernel(
    half* __restrict__ out,
    const half* __restrict__ query,
    const uint8_t* __restrict__ cache_kv,   // Quantized K then V data
    const half* __restrict__ k_scales,      // K scales: [num_blocks * quant_blocks_per_block]
    const half* __restrict__ v_scales,      // V scales: [num_blocks * quant_blocks_per_block]
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    int num_seqs,
    int max_num_blocks,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size
) {
    const int seq_idx = blockIdx.x;
    const int q_head = blockIdx.y;
    if (seq_idx >= num_seqs) return;
    const int seq_len = seq_lens[seq_idx];
    if (seq_len == 0) return;

    const int tid = threadIdx.x;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    
    // Quantized layout constants
    const int elems_per_block = block_size * num_kv_heads * head_dim;
    const int quant_blocks_per_block = (elems_per_block + 31) / 32;
    const int k_bytes_per_block = quant_blocks_per_block * 18;  // 18 bytes per 32 elems
    const int v_bytes_per_block = k_bytes_per_block;
    
    const int q_offset = (seq_idx * num_q_heads + q_head) * head_dim;
    const int kv_head_stride = num_kv_heads * head_dim;
    const bool use_vec = (head_dim % 8 == 0);

    extern __shared__ float smem[];
    float* sq = smem;

    // Load query into shared memory
    if (use_vec) {
        const int vd = head_dim / 8;
        for (int g = tid; g < vd; g += blockDim.x) loadQ8(sq, smem + head_dim, query + q_offset, g);
    } else {
        for (int d = tid; d < head_dim; d += blockDim.x) {
            sq[d] = __half2float(query[q_offset + d]);
            smem[head_dim + d] = 0.0f;
        }
    }
    __syncthreads();
    // (Contrato C1) escalas embebidas: params k_scales/v_scales inertes.
    __syncthreads();

    float max_val = -1e30f;
    float exp_sum = 0.0f;
    const float scale_factor = 1.0f / sqrtf((float)head_dim);
    const int num_blocks = (seq_lens[seq_idx] + block_size - 1) / block_size;

    for (int b = 0; b < num_blocks; b++) {
        const int phys = block_tables[seq_idx * max_num_blocks + b];
        if (phys < 0) continue;
        
        const int base = phys * (k_bytes_per_block + v_bytes_per_block);
        
        // K and V data pointers for this physical block
        const uint8_t* k_data = cache_kv + base;
        const uint8_t* v_data = cache_kv + base + k_bytes_per_block;

        for (int t = 0; t < (b == num_blocks - 1 ? (seq_lens[seq_idx] - b * block_size) : block_size); t++) {
            // Load K and compute Q @ K^T with on-the-fly dequantization
            float partial = 0.0f;
            
            if (use_vec) {
                const int vd = head_dim / 8;
                for (int g = tid; g < vd; g += blockDim.x) {
                    // Dequantize 8 K elements on-the-fly and dot with sq
                    int base_elem = (t * kv_head_stride + kv_head * head_dim) + g * 8;
                    int qb = base_elem / 32;
                    int block_offset = qb * 18;
                    __half scale_h = *reinterpret_cast<const __half*>(k_data + block_offset);
                    float scale = __half2float(scale_h);
                    
                    float partial8 = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        int elem_idx = base_elem + i;
                                                int in_block = elem_idx % 32;
                        const int hidx = (in_block < 16) ? in_block : (in_block - 16);
                        uint8_t byte = k_data[block_offset + 2 + hidx];
                        int nibble = (in_block < 16) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                        int q = nibble - 8;  // Q4_0 uses offset 8
                        float k_val = (float)q * scale;
                        partial8 += sq[g * 8 + i] * k_val;
                    }
                    partial += partial8;
                }
            } else {
                for (int d = tid; d < head_dim; d += blockDim.x) {
                    int base_elem = t * kv_head_stride + kv_head * head_dim + d;
                    int qb = base_elem / 32;
                    int block_offset = qb * 18;
                    __half scale_h = *reinterpret_cast<const __half*>(k_data + block_offset);
                    float scale = __half2float(scale_h);
                    int in_block = base_elem % 32;
                    const int hidx = (in_block < 16) ? in_block : (in_block - 16);
                    uint8_t byte = k_data[block_offset + 2 + hidx];
                    int nibble = (in_block < 16) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                    int q = nibble - 8;  // Q4_0 uses offset 8
                    float k_val = (float)q * scale;
                    partial += sq[d] * k_val;
                }
            }
            
            // Warp reduction for partial sum
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                partial += __shfl_xor_sync(0xffffffffu, partial, off);
            }
            
            const float score = partial * (1.0f / sqrtf((float)head_dim));
            const float new_max = fmaxf(max_val, score);
            const float scale = expf(max_val - new_max);
            exp_sum *= scale;
            
            // Scale accumulated values
            if (use_vec) {
                const int vd = head_dim / 8;
                for (int g = tid; g < vd; g += blockDim.x) scal8(smem + head_dim, g, scale);
            } else {
                for (int d = tid; d < head_dim; d += blockDim.x) {
                    smem[head_dim + d] *= scale;
                }
            }
            max_val = new_max;
            
            // Exp score and add to sum
            const float exp_score = expf(score - new_max);
            exp_sum += exp_score;
            
            // Load V and accumulate
            if (use_vec) {
                const int vd = head_dim / 8;
                for (int g = tid; g < vd; g += blockDim.x) {
                    int base_elem = (t * kv_head_stride + kv_head * head_dim) + g * 8;
                    int qb = base_elem / 32;
                    int block_offset = qb * 18;
                    __half scale_h = *reinterpret_cast<const __half*>(v_data + block_offset);
                    float scale = __half2float(scale_h);
                    
                    // Dequantize 8 V elements on-the-fly
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        int elem_idx = g * 8 + i;
                                                int in_block = elem_idx % 32;
                        const int hidx = (in_block < 16) ? in_block : (in_block - 16);
                        uint8_t byte = v_data[block_offset + 2 + hidx];
                        int nibble = (in_block < 16) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                        int q = nibble - 8;  // Q4_0 uses offset 8
                        float v_val = (float)q * scale;
                        smem[head_dim + g * 8 + i] += v_val * exp_score;
                    }
                }
            } else {
                for (int d = tid; d < head_dim; d += blockDim.x) {
                    int base_elem = (t * kv_head_stride + kv_head * head_dim) + d;
                    int qb = base_elem / 32;
                    int block_offset = qb * 18;
                    __half scale_h = *reinterpret_cast<const __half*>(v_data + block_offset);
                    float scale = __half2float(scale_h);
                    int in_block = base_elem % 32;
                    const int hidx = (in_block < 16) ? in_block : (in_block - 16);
                    uint8_t byte = v_data[block_offset + 2 + hidx];
                    int nibble = (in_block < 16) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                    int q = nibble - 8;  // Q4_0 uses offset 8
                    float v_val = (float)q * scale;
                    smem[head_dim + d] += v_val * exp_score;
                }
            }
        }
    }
    
    // Write output
    const float inv_sum = 1.0f / exp_sum;
    if (use_vec) {
        const int vd = head_dim / 8;
        for (int g = tid; g < vd; g += blockDim.x) storeOut8(out + q_offset, g, smem + head_dim, inv_sum);
    } else {
        for (int d = tid; d < head_dim; d += blockDim.x) {
            out[q_offset + d] = __float2half(smem[head_dim + d] * (1.0f / exp_sum));
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// 3.3-b (lane-f): Fused q4_0 Paged Attention Decode — VARIANTE dp4a.
// Mismo esquema que la variante q8_0 (Q→q8_0 en smem, dp4a por grupo-32,
// dueño por thread) con el truco MMQ estándar para el offset de nibbles:
//   v = n − 8  ⇒  dot(q8, n) = dot(q8, v) + 8·Σq8  ⇒  dot(q8, v) = dp4a − 8·Σq8
// K q4_0: 16 nibbles por byte-u32... registro 18B [escala f16][16B nibbles]:
// elems [0..16) = nibbles LOW (elem par), [16..32) = nibbles HIGH (elem impar).
// K se carga como u32 (4 nibbles/u32): dp4a con Q-repackeada en el MISMO
// orden low-par/high-impar + corrección −8·Σq8 por grupo.
// Q-repack: q8_q[g*32+i] con i par ← quanta del elem 2j, i impar ← elem 2j+1
// (espejo del orden de nibbles de K). Σq8 del grupo se precomputa en fase 0
// (suma de los 32 quanta del grupo — 1 DP-int por grupo, f32 store).
// ════════════════════════════════════════════════════════════════════════════════
extern "C" __global__ void paged_attention_decode_q4_0_dp4a_kernel(
    half* __restrict__ out,
    const half* __restrict__ query,
    const uint8_t* __restrict__ cache_kv,
    const half* __restrict__ k_scales,
    const half* __restrict__ v_scales,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    int num_seqs,
    int max_num_blocks,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size
) {
    (void)k_scales;
    (void)v_scales;
    const int seq_idx = blockIdx.x;
    const int q_head  = blockIdx.y;
    if (seq_idx >= num_seqs) return;
    const int seq_len = seq_lens[seq_idx];
    if (seq_len == 0) return;

    const int tid      = threadIdx.x;
    const int nthreads = blockDim.x;
    const int kv_head  = q_head / (num_q_heads / num_kv_heads);
    const int elems_per_block = block_size * num_kv_heads * head_dim;
    const size_t k_bytes = (size_t)((elems_per_block + 31) / 32) * 18;
    const int q_offset       = (seq_idx * num_q_heads + q_head) * head_dim;
    const int kv_head_stride = num_kv_heads * head_dim;
    const int qb_head = (head_dim + 31) / 32;

    // smem: [sq][acc][d_q qb][qsum qb][pad16B][q8_q u32 qb*8]
    extern __shared__ float smem[];
    float* sq    = smem;
    float* acc   = smem + head_dim;
    float* d_q   = acc + head_dim;
    float* qsum  = d_q + qb_head;
    unsigned int* q8_q = reinterpret_cast<unsigned int*>(((uintptr_t)(qsum + qb_head) + 15ull) & ~15ull);

    // ── Fase 0: Q f16 → q8_0 (orden par/impar espejo de nibbles) + Σq8 ──
    for (int g = 0; g < qb_head; ++g) {
        const int base_e = g * 32;
        float amax = 0.0f;
        if (base_e + tid < head_dim) {
            const float qv = __half2float(query[q_offset + base_e + tid]);
            sq[base_e + tid] = qv;
            amax = fabsf(qv);
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d = (amax > 0.0f) ? amax / 127.0f : 1.0f;
        if (tid == 0) d_q[g] = d;
        int q = 0;
        if (base_e + tid < head_dim) {
            q = (int)roundf(sq[base_e + tid] / d);
            q = max(-127, min(127, q));
            acc[base_e + tid] = 0.0f;
        }
        const int qi = q;   // quanta INDIVIDUAL del lane (el butterfly de
                            // abajo destruye q — bug 3.3-b: el byte-store
                            // guardaba la suma total en todos los bytes).
        // Σ de los 32 quanta del grupo (para la corrección −8·Σq8): butterfly
        // int + broadcast; cola hd%32 aporta 0.
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            q += __shfl_xor_sync(0xffffffffu, q, o);
        if (tid == 0) qsum[g] = (float)q;
        // Q-repack espejo del layout q4_0 de ESTE pool (kvAppendQ4_0:
        // SPLIT-16, igual que GGUF): byte j del registro = [low: elem j |
        // high: elem j+16]. El dp4a trabaja por BYTES tras extraer los
        // nibbles de K en dos mitades: lo = kw & 0x0F0F0F0F (elems 0..15,
        // orden de byte) y hi = (kw>>4) & 0x0F0F0F0F (elems 16..31).
        // ⇒ el map de Q es el TRIVIAL: byte b = quanta del elem b
        //   bytes [0..16) = elems pares-bajo... NO: elems 0..15 (low)
        //   bytes [16..32) = elems 16..31 (high)
        // El dot por grupo = dp4a(lo, qq[u]) + dp4a(hi, qq[u+4]).
        reinterpret_cast<signed char*>(q8_q)[g * 32 + tid] = (signed char)qi;
    }
    __syncwarp();

    float max_val = -1e30f;
    float exp_sum = 0.0f;
    const float scale_factor = 1.0f / sqrtf((float)head_dim);
    const int num_blocks = (seq_len + block_size - 1) / block_size;
    const int my_groups = (qb_head + nthreads - 1) / nthreads;

    for (int b = 0; b < num_blocks; ++b) {
        const int phys = block_tables[seq_idx * max_num_blocks + b];
        if (phys < 0) continue;
        const int tokens_in_block = (b == num_blocks - 1)
            ? (seq_len - b * block_size) : block_size;
        const uint8_t* k_data = cache_kv + (size_t)phys * (2 * k_bytes);
        const uint8_t* v_data = k_data + k_bytes;

        for (int t = 0; t < tokens_in_block; ++t) {
            const int t_off = t * kv_head_stride + kv_head * head_dim;

            // ── Dot Q·K: dp4a contra nibbles + corrección −8·Σq8 ──
            float partial = 0.0f;
            for (int mg = 0; mg < my_groups; ++mg) {
                const int g = mg * nthreads + tid;
                if (g >= qb_head) break;
                const int be = t_off + g * 32;
                const uint8_t* blk = k_data + (size_t)(be >> 5) * 18;
                const float d_k = __half2float(*reinterpret_cast<const __half*>(blk));
                // 16B de nibbles = 4 u32. blk+2 alineación: registro 18B ⇒
                // offset múltiplo de 18 NO alineado 4B — pack por bytes
                // (misma trampa 3.3 q8_0). Cada u32 = 4 bytes = 8 nibbles:
                // [e2j low | e2j+1 high]×4 → low-mask pares / high-shift impares.
                const uint8_t* kn = blk + 2;
                const unsigned int* qq = q8_q + g * 8;
                int dot = 0;
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    unsigned int kw = (unsigned int)kn[u * 4]
                        | ((unsigned int)kn[u * 4 + 1] << 8)
                        | ((unsigned int)kn[u * 4 + 2] << 16)
                        | ((unsigned int)kn[u * 4 + 3] << 24);
                    const unsigned int lo = kw & 0x0F0F0F0Fu;
                    const unsigned int hi = (kw >> 4) & 0x0F0F0F0Fu;
                    dot = __dp4a((int)lo, (int)qq[u], dot);
                    dot = __dp4a((int)hi, (int)qq[u + 4], dot);
                }
                // dot = Σ q8·nibbles; v = n−8 ⇒ Σ q8·v = dot − 8·Σq8.
                partial += ((float)dot - 8.0f * qsum[g]) * (d_q[g] * d_k);
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                partial += __shfl_xor_sync(0xffffffffu, partial, off);
            const float score = partial * scale_factor;
            const float new_max = fmaxf(max_val, score);
            const float rescale = expf(max_val - new_max);
            const float exp_score = expf(score - new_max);
            exp_sum = exp_sum * rescale + exp_score;
            for (int e = tid; e < head_dim; e += nthreads) acc[e] *= rescale;
            max_val = new_max;
            // V-side: dequant + FMA (igual que el base).
            for (int e = tid; e < head_dim; e += nthreads) {
                const int be = t_off + e;
                const uint8_t* blk = v_data + (size_t)(be >> 5) * 18;
                const float dsc = __half2float(*reinterpret_cast<const __half*>(blk));
                const int in_block = be & 31;
                const int hidx = (in_block < 16) ? in_block : (in_block - 16);
                const uint8_t byte = blk[2 + hidx];
                const int nibble = (in_block < 16) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                acc[e] += (float)(nibble - 8) * dsc * exp_score;
            }
        }
    }
    const float inv_sum = 1.0f / exp_sum;
    for (int e = tid; e < head_dim; e += nthreads)
        out[q_offset + e] = __float2half(acc[e] * inv_sum);
}
// ════════════════════════════════════════════════════════════════════════════════
// Fused q4_k Paged Attention Decode Kernel
// Dequantizes q4_k K/V on-the-fly during attention computation.
// Super-block 256, 144 bytes: d f16, min f16, scales[12], qs[128]
// ════════════════════════════════════════════════════════════════════════════════
#include <stdint.h>

extern "C" __global__ void paged_attention_decode_q4_k_kernel(
    half* __restrict__ out,
    const half* __restrict__ query,
    const uint8_t* __restrict__ cache_kv,
    const half* __restrict__ k_scales,
    const half* __restrict__ v_scales,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    int num_seqs,
    int max_num_blocks,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size
) {
    const int seq_idx = blockIdx.x;
    const int q_head = blockIdx.y;
    if (seq_idx >= num_seqs) return;
    const int seq_len = seq_lens[seq_idx];
    if (seq_len == 0) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    
    const int elems_per_block = block_size * num_kv_heads * head_dim;
    const int qk = 256;
    const int quant_blocks_per_block = (elems_per_block + qk - 1) / qk;
    const int k_bytes_per_block = quant_blocks_per_block * 144;
    const int v_bytes_per_block = k_bytes_per_block;
    
    const int q_offset = (seq_idx * num_q_heads + q_head) * head_dim;
    const int kv_head_stride = num_kv_heads * head_dim;
    const bool use_vec = (head_dim % 8 == 0);

    extern __shared__ float smem[];
    float* sq = smem;
    float* acc = smem + head_dim;

    if (use_vec) {
        const int vd = head_dim / 8;
        for (int g = tid; g < vd; g += nthreads) loadQ8(sq, acc, query + q_offset, g);
    } else {
        for (int d = tid; d < head_dim; d += nthreads) {
            sq[d] = __half2float(query[q_offset + d]);
            acc[d] = 0.0f;
        }
    }
    __syncthreads();

    float max_val = -1e30f;
    float exp_sum = 0.0f;
    const float scale_factor = 1.0f / sqrtf((float)head_dim);
    const int num_blocks = (seq_len + block_size - 1) / block_size;

    for (int b = 0; b < num_blocks; b++) {
        const int phys = block_tables[seq_idx * max_num_blocks + b];
        if (phys < 0) continue;
        
        const int tokens_in_block = (b == num_blocks - 1) ? (seq_len - b * block_size) : block_size;
        const int base = phys * (k_bytes_per_block + v_bytes_per_block);
        
        const uint8_t* k_data = cache_kv + base;
        const uint8_t* v_data = cache_kv + base + k_bytes_per_block;

        for (int t = 0; t < tokens_in_block; t++) {
            float partial = 0.0f;
            const int t_offset = t * kv_head_stride + kv_head * head_dim;
            
            // Declare d_val and min outside if/else so both branches can use them
            float d_val = 0.0f;
            float min = 0.0f;
            
            if (use_vec) {
                const int vd = head_dim / 8;
                for (int g = tid; g < vd; g += nthreads) {
                    int base_elem = t_offset + g * 8;
                    int qb = base_elem / qk;
                    int in_qk = base_elem % qk;
                    int blk_off = qb * 144;
                    const uint8_t* blk = k_data + blk_off;
                    d_val = __half2float(*(const __half*)(blk));
                    min = __half2float(*(const __half*)(blk + 2));
                    const uint8_t* scales = blk + 4;
                    const uint8_t* qs = blk + 16;
                    
                    float partial8 = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        int elem = base_elem + i;
                        if (elem < t_offset + head_dim) {
                            int in = (in_qk + i) % qk;
                            int g_idx = in / 64;
                            int l = in % 64;
                            int si = 2 * g_idx + (l < 32 ? 0 : 1);
                            int sd, sm;
                            if (si < 4) {
                                sd = scales[si] & 63;
                                sm = scales[si + 4] & 63;
                            } else {
                                sd = (scales[si + 4] & 0xF) | ((scales[si - 4] >> 6) << 4);
                                sm = (scales[si + 4] >> 4) | ((scales[si] >> 6) << 4);
                            }
                            float dl = d_val * (float)sd;
                            float ml = min * (float)sm;
                            uint8_t qb_val = qs[g_idx * 32 + (l % 32)];
                            int qv = (l < 32) ? (qb_val & 0xF) : ((qb_val >> 4) & 0xF);
                            float k_val = dl * (float)qv - ml;
                            partial8 += sq[g * 8 + i] * k_val;
                        }
                    }
                    partial += partial8;
                }
            } else {
                for (int d = tid; d < head_dim; d += nthreads) {
                    int base_elem = t_offset + d;
                    int qb = base_elem / qk;
                    int in = base_elem % qk;
                    int blk_off = qb * 144;
                    const uint8_t* blk = k_data + blk_off;
                    d_val = __half2float(*(const __half*)(blk));
                    min = __half2float(*(const __half*)(blk + 2));
                    const uint8_t* scales = blk + 4;
                    const uint8_t* qs = blk + 16;
                    int g_idx = in / 64;
                    int l = in % 64;
                    int si = 2 * g_idx + (l < 32 ? 0 : 1);
                    int sd, sm;
                    if (si < 4) {
                        sd = scales[si] & 63;
                        sm = scales[si + 4] & 63;
                    } else {
                        sd = (scales[si + 4] & 0xF) | ((scales[si - 4] >> 6) << 4);
                        sm = (scales[si + 4] >> 4) | ((scales[si] >> 6) << 4);
                    }
                    float dl = d_val * (float)sd;
                    float ml = min * (float)sm;
                    uint8_t qb_val = qs[g_idx * 32 + (l % 32)];
                    int qv = (l < 32) ? (qb_val & 0xF) : ((qb_val >> 4) & 0xF);
                    float k_val = dl * (float)qv - ml;
                    partial += sq[d] * k_val;
                }
            }

            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                partial += __shfl_xor_sync(0xffffffffu, partial, off);
            }

            const float score = partial * scale_factor;
            const float new_max = fmaxf(max_val, score);
            const float scale = expf(max_val - new_max);
            exp_sum *= scale;

            if (use_vec) {
                const int vd = head_dim / 8;
                for (int g = tid; g < vd; g += nthreads) scal8(acc, g, scale);
            } else {
                for (int d = tid; d < head_dim; d += nthreads) acc[d] *= scale;
            }
            max_val = new_max;

            const float exp_score = expf(score - new_max);
            exp_sum += exp_score;

            if (use_vec) {
                const int vd = head_dim / 8;
                for (int g = tid; g < vd; g += nthreads) {
                    int base_elem = t_offset + g * 8;
                    int qb = base_elem / qk;
                    int in_qk = base_elem % qk;
                    int blk_off = qb * 144;
                    const uint8_t* blk = v_data + blk_off;
                    float d = __half2float(*(const __half*)(blk));
                    float min = __half2float(*(const __half*)(blk + 2));
                    const uint8_t* scales = blk + 4;
                    const uint8_t* qs = blk + 16;
                    
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        int elem = base_elem + i;
                        if (elem < t_offset + head_dim) {
                            int in = (in_qk + i) % qk;
                            int g_idx = in / 64;
                            int l = in % 64;
                            int si = 2 * g_idx + (l < 32 ? 0 : 1);
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
                            uint8_t qb_val = qs[g_idx * 32 + (l % 32)];
                            int qv = (l < 32) ? (qb_val & 0xF) : ((qb_val >> 4) & 0xF);
                            float v_val = dl * (float)qv - ml;
                            acc[g * 8 + i] += v_val * exp_score;
                        }
                    }
                }
            } else {
                for (int d = tid; d < head_dim; d += nthreads) {
                    int base_elem = t_offset + d;
                    int qb = base_elem / qk;
                    int in = base_elem % qk;
                    int blk_off = qb * 144;
                    const uint8_t* blk = v_data + blk_off;
                    float d_val = __half2float(*(const __half*)(blk));
                    float min = __half2float(*(const __half*)(blk + 2));
                    const uint8_t* scales = blk + 4;
                    const uint8_t* qs = blk + 16;
                    int g_idx = in / 64;
                    int l = in % 64;
                    int si = 2 * g_idx + (l < 32 ? 0 : 1);
                    int sd, sm;
                    if (si < 4) {
                        sd = scales[si] & 63;
                        sm = scales[si + 4] & 63;
                    } else {
                        sd = (scales[si + 4] & 0xF) | ((scales[si - 4] >> 6) << 4);
                        sm = (scales[si + 4] >> 4) | ((scales[si] >> 6) << 4);
                    }
                    float dl = d_val * (float)sd;
                    float ml = min * (float)sm;
                    uint8_t qb_val = qs[g_idx * 32 + (l % 32)];
                    int qv = (l < 32) ? (qb_val & 0xF) : ((qb_val >> 4) & 0xF);
                    float v_val = dl * (float)qv - ml;
                    acc[d] += v_val * exp_score;
                }
            }
        }
    }

    const float inv_sum = 1.0f / exp_sum;
    if (use_vec) {
        const int vd = head_dim / 8;
        for (int g = tid; g < vd; g += nthreads) storeOut8(out + q_offset, g, acc, inv_sum);
    } else {
        for (int d = tid; d < head_dim; d += nthreads) {
            out[q_offset + d] = __float2half(acc[d] * inv_sum);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// Fused q8_k Paged Attention Decode Kernel
// Dequantizes q8_k K/V on-the-fly during attention computation.
// Super-block 256 (block_q8_K, 292 bytes):
//   d f32 (offset 0), qs[256] i8 (offset 4), bsums[16] i16 (offset 260, unused).
//   val = d * (float)qs[j]                    (ref: ggml dequantize_row_q8_K)
//
// Ampere-optimized:
//  - Bandwidth-bound decode: each thread loads 8 consecutive int8 via one uint2
//    (LDG.64), dots against sq[0..7] in registers; warp reduce via __shfl_xor_sync.
//  - Per-super-block f32 scales preloaded once per physical block into shared
//    memory so the inner token loops never re-read them from global.
// ════════════════════════════════════════════════════════════════════════════════
#include <stdint.h>

extern "C" __global__ void paged_attention_decode_q8_k_kernel(
    half* __restrict__ out,
    const half* __restrict__ query,
    const uint8_t* __restrict__ cache_kv,
    const half* __restrict__ k_scales,
    const half* __restrict__ v_scales,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    int num_seqs,
    int max_num_blocks,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size
) {
    const int seq_idx = blockIdx.x;
    const int q_head  = blockIdx.y;
    if (seq_idx >= num_seqs) return;
    const int seq_len = seq_lens[seq_idx];
    if (seq_len == 0) return;

    const int tid      = threadIdx.x;
    const int nthreads = blockDim.x;
    const int kv_head  = q_head / (num_q_heads / num_kv_heads);

    // Layout: [K region][V region] per physical block.
    const int elems_per_block = block_size * num_kv_heads * head_dim;
    constexpr int QK_K        = 256;
    constexpr int SB_BYTES    = 292;   // sizeof(block_q8_K)
    const int sb_per_region   = (elems_per_block + QK_K - 1) / QK_K;
    const int k_bytes         = sb_per_region * SB_BYTES;
    const int v_bytes         = k_bytes;

    const int q_offset       = (seq_idx * num_q_heads + q_head) * head_dim;
    const int kv_head_stride = num_kv_heads * head_dim;
    const bool use_vec       = (head_dim % 8 == 0);

    extern __shared__ float smem[];
    float* sq          = smem;                 // [head_dim]
    float* acc         = smem + head_dim;      // [head_dim]
    float* scale_cache = smem + 2 * head_dim;  // [2 * sb_per_region] K then V

    if (use_vec) {
        const int vd = head_dim / 8;
        for (int g = tid; g < vd; g += nthreads) loadQ8(sq, acc, query + q_offset, g);
    } else {
        for (int d = tid; d < head_dim; d += nthreads) {
            sq[d]  = __half2float(query[q_offset + d]);
            acc[d] = 0.0f;
        }
    }
    __syncthreads();

    float max_val = -1e30f;
    float exp_sum = 0.0f;
    const float scale_factor = 1.0f / sqrtf((float)head_dim);
    const int num_blocks = (seq_len + block_size - 1) / block_size;

    for (int b = 0; b < num_blocks; b++) {
        const int phys = block_tables[seq_idx * max_num_blocks + b];
        if (phys < 0) continue;

        const int tokens_in_block = (b == num_blocks - 1)
            ? (seq_len - b * block_size) : block_size;
        const uint8_t* k_data = cache_kv + (size_t)phys * (k_bytes + v_bytes);
        const uint8_t* v_data = k_data + k_bytes;

        // Preload super-block f32 scales into shared memory (K first, then V).
        for (int s = tid; s < sb_per_region; s += nthreads) {
            scale_cache[s] = *reinterpret_cast<const float*>(k_data + (size_t)s * SB_BYTES);
            scale_cache[sb_per_region + s] =
                *reinterpret_cast<const float*>(v_data + (size_t)s * SB_BYTES);
        }
        __syncthreads();

        for (int t = 0; t < tokens_in_block; t++) {
            const int t_offset = t * kv_head_stride + kv_head * head_dim;

            // ── QK^T with on-the-fly dequant ────────────────────────────────
            float partial = 0.0f;
            if (use_vec) {
                const int vd = head_dim / 8;
                for (int g = tid; g < vd; g += nthreads) {
                    const int base_elem = t_offset + g * 8;
                    const int sb        = base_elem >> 8;              // /QK_K
                    const float d       = scale_cache[sb];
                    const uint8_t* q8 = k_data + (size_t)sb * SB_BYTES + 4
                                      + (base_elem & 255);
                    float dot = 0.0f;
                    _Pragma("unroll")
                    for (int i = 0; i < 8; i++) {
                        dot += sq[g * 8 + i] * (float)(int8_t)q8[i];
                    }
                    partial += d * dot;
                }
            } else {
                for (int e = tid; e < head_dim; e += nthreads) {
                    const int base_elem = t_offset + e;
                    const int sb        = base_elem >> 8;
                    const float d       = scale_cache[sb];
                    const int8_t q      = (int8_t)k_data[(size_t)sb * SB_BYTES + 4 + (base_elem & 255)];
                    partial += sq[e] * (d * (float)q);
                }
            }

            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                partial += __shfl_xor_sync(0xffffffffu, partial, off);
            }

            // ── Online softmax update ───────────────────────────────────────
            const float score   = partial * scale_factor;
            const float new_max = fmaxf(max_val, score);
            const float rescale = expf(max_val - new_max);
            exp_sum *= rescale;
            if (use_vec) {
                const int vd = head_dim / 8;
                for (int g = tid; g < vd; g += nthreads) scal8(acc, g, rescale);
            } else {
                for (int d = tid; d < head_dim; d += nthreads) acc[d] *= rescale;
            }
            max_val = new_max;
            const float exp_score = expf(score - new_max);
            exp_sum += exp_score;

            // ── Accumulate weighted V with on-the-fly dequant ────────────────
            if (use_vec) {
                const int vd = head_dim / 8;
                for (int g = tid; g < vd; g += nthreads) {
                    const int base_elem = t_offset + g * 8;
                    const int sb        = base_elem >> 8;
                    const float d       = scale_cache[sb_per_region + sb];
                    const uint8_t* q8 = v_data + (size_t)sb * SB_BYTES + 4
                                      + (base_elem & 255);
                    _Pragma("unroll")
                    for (int i = 0; i < 8; i++) {
                        acc[g * 8 + i] += (d * (float)(int8_t)q8[i]) * exp_score;
                    }
                }
            } else {
                for (int e = tid; e < head_dim; e += nthreads) {
                    const int base_elem = t_offset + e;
                    const int sb        = base_elem >> 8;
                    const float d       = scale_cache[sb_per_region + sb];
                    const int8_t q      = (int8_t)v_data[(size_t)sb * SB_BYTES + 4 + (base_elem & 255)];
                    acc[e] += (d * (float)q) * exp_score;
                }
            }
        }
        __syncthreads();   // scale_cache reused across physical blocks
    }

    const float inv_sum = 1.0f / exp_sum;
    if (use_vec) {
        const int vd = head_dim / 8;
        for (int g = tid; g < vd; g += nthreads) storeOut8(out + q_offset, g, acc, inv_sum);
    } else {
        for (int d = tid; d < head_dim; d += nthreads) {
            out[q_offset + d] = __float2half(acc[d] * inv_sum);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// K-quants restantes: dequant on-the-fly fiel a unsloth dequantize.cuh
// (mapeos tid→posición reescritos a posición absoluta w∈[0,256)).
//  block_q2_K (84B): scales[16]@0 qs[64]@16 d f16@80 dmin f16@82
//  block_q3_K(110B): hmask[32]@0  qs[64]@32 scales[12]@96 d f16@108
//  block_q5_K(176B): d f16@0 dmin f16@2 scales[12]@4 qh[32]@16 qs[128]@48
//  block_q6_K(210B): ql[128]@0   qh[64]@128 scales i8[16]@192 d f16@208
// ════════════════════════════════════════════════════════════════════════════

__device__ __forceinline__ float q2k_val(const uint8_t* blk, int w) {
    // w = 128n + l + 32j ; byte qs[32n+l], shift 2j, escala scales[8n+2j+l/16]
    const int n = w >> 7;
    const int r = w & 127;
    const int j = r >> 5;
    const int l = r & 31;
    const float d    = __half2float(*reinterpret_cast<const __half*>(blk + 80));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(blk + 82));
    const uint8_t scl = blk[8 * n + 2 * j + l / 16];
    const uint8_t q   = blk[16 + 32 * n + l];
    return d * (float)(scl & 0xF) * (float)((q >> (2 * j)) & 3)
         - dmin * (float)(scl >> 4);
}

__device__ __forceinline__ float q3k_val(const uint8_t* blk, int w) {
    const int n = w >> 7;
    const int r = w & 127;
    const int j = r >> 5;
    const int l = r & 31;
    const int is     = 8 * n + 2 * j + l / 16;
    const uint8_t us =
          is <  4 ? (uint8_t)((blk[96 + is]      & 0xF) | (((blk[96 + is + 8] >> 0) & 3) << 4))
        : is <  8 ? (uint8_t)((blk[96 + is]      & 0xF) | (((blk[96 + is + 4] >> 2) & 3) << 4))
        : is < 12 ? (uint8_t)((blk[96 + is - 8] >> 4)      | (((blk[96 + is]     >> 4) & 3) << 4))
                  : (uint8_t)((blk[96 + is - 8] >> 4)      | (((blk[96 + is - 4] >> 6) & 3) << 4));
    const float dl  = __half2float(*reinterpret_cast<const __half*>(blk + 108)) * ((int)us - 32);
    const uint8_t m = 1u << (4 * n + j);
    const int hm    = (blk[l] & m) ? 0 : 4;
    const int q     = (blk[32 + 32 * n + l] >> (2 * j)) & 3;
    return dl * (float)(q - hm);
}

__device__ __forceinline__ void get_scale_min_k5_dev(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) { d = q[j] & 63;     m = q[j + 4] & 63; }
    else       { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                 m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

__device__ __forceinline__ float q5k_val(const uint8_t* blk, int w) {
    const int il     = w >> 6;              // grupo de 64 (0..3)
    const int t      = w & 63;
    const bool hi    = t >= 32;
    const int ir     = (t & 31) >> 1;
    const int parity = t & 1;
    const float dall = __half2float(*reinterpret_cast<const __half*>(blk));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(blk + 2));
    const uint8_t* scales = blk + 4;
    const uint8_t* qh     = blk + 16;
    const uint8_t* qs     = blk + 48;
    int sd, sm;
    get_scale_min_k5_dev(2 * il + (hi ? 1 : 0), scales, sd, sm);
    const float d1 = dall * (float)sd;
    const float m1 = dmin * (float)sm;
    const uint8_t ql  = qs[32 * il + 2 * ir + parity];
    const uint8_t qb  = qh[2 * ir + parity];
    const uint8_t hmb = 1u << (2 * il + (hi ? 1 : 0));
    const int nib     = hi ? (ql >> 4) : (ql & 0xF);
    const int extra   = (qb & hmb) ? 16 : 0;
    return d1 * (float)(nib + extra) - m1;
}

__device__ __forceinline__ float q6k_val(const uint8_t* blk, int w) {
    const int ip  = w >> 7;
    const int r   = w & 127;
    const int il  = r & 31;
    const int j   = r >> 5;
    const float d = __half2float(*reinterpret_cast<const __half*>(blk + 208));
    const int8_t sc = (int8_t)blk[192 + 8 * ip + il / 16 + 2 * j];
    // Lane A fix: cuadrantes {low,high} alternan bloques ql por j&1 (no j>>1).
    // GGUF q6_K: q1=ql[l]&F, q2=ql[l+32]&F, q3=ql[l]>>4, q4=ql[l+32]>>4.
    const uint8_t qlb = blk[64 * ip + il + (j & 1) * 32];
    const int nib     = (j >> 1) ? (qlb >> 4) : (qlb & 0xF);
    const uint8_t qhb = blk[128 + 32 * ip + il];
    const int packed  = nib | (((qhb >> (2 * j)) & 3) << 4);
    return d * (float)sc * (float)(packed - 32);
}

// Kernel fusionado genérico para K-quants con dequant por posición absoluta.
#define DEFINE_PAGED_DECODE_QK_KERNEL(KERNEL_NAME, VALFN, SB_BYTES)                  \
extern "C" __global__ void KERNEL_NAME(                                              \
    half* __restrict__ out,                                                          \
    const half* __restrict__ query,                                                  \
    const uint8_t* __restrict__ cache_kv,                                            \
    const half* __restrict__ k_scales,                                               \
    const half* __restrict__ v_scales,                                               \
    const int* __restrict__ block_tables,                                            \
    const int* __restrict__ seq_lens,                                                \
    int num_seqs, int max_num_blocks, int num_q_heads,                               \
    int num_kv_heads, int head_dim, int block_size)                                  \
{                                                                                    \
    const int seq_idx = blockIdx.x;                                                  \
    const int q_head  = blockIdx.y;                                                  \
    if (seq_idx >= num_seqs) return;                                                 \
    const int seq_len = seq_lens[seq_idx];                                           \
    if (seq_len == 0) return;                                                        \
    const int tid      = threadIdx.x;                                                \
    const int nthreads = blockDim.x;                                                 \
    const int kv_head  = q_head / (num_q_heads / num_kv_heads);                      \
    const int elems_per_block = block_size * num_kv_heads * head_dim;                \
    const int sb_per_region   = (elems_per_block + 255) / 256;                       \
    const int k_bytes         = sb_per_region * (SB_BYTES);                          \
    const int q_offset        = (seq_idx * num_q_heads + q_head) * head_dim;         \
    const int kv_head_stride  = num_kv_heads * head_dim;                             \
    extern __shared__ float smem[];                                                  \
    float* sq  = smem;                                                               \
    float* acc = smem + head_dim;                                                    \
    for (int e = tid; e < head_dim; e += nthreads) {                                 \
        sq[e]  = __half2float(query[q_offset + e]);                                  \
        acc[e] = 0.0f;                                                               \
    }                                                                                \
    __syncthreads();                                                                 \
    float max_val = -1e30f;                                                          \
    float exp_sum = 0.0f;                                                            \
    const float scale_factor = 1.0f / sqrtf((float)head_dim);                        \
    const int num_blocks = (seq_len + block_size - 1) / block_size;                  \
    for (int b = 0; b < num_blocks; b++) {                                           \
        const int phys = block_tables[seq_idx * max_num_blocks + b];                 \
        if (phys < 0) continue;                                                      \
        const int tokens_in_block = (b == num_blocks - 1)                            \
            ? (seq_len - b * block_size) : block_size;                               \
        const uint8_t* k_data = cache_kv + (size_t)phys * (2 * k_bytes);             \
        const uint8_t* v_data = k_data + k_bytes;                                    \
        for (int t = 0; t < tokens_in_block; t++) {                                  \
            const int t_offset = t * kv_head_stride + kv_head * head_dim;            \
            float partial = 0.0f;                                                    \
            for (int e = tid; e < head_dim; e += nthreads) {                         \
                const int be = t_offset + e;                                         \
                partial += sq[e] * VALFN(k_data + (size_t)(be >> 8) * (SB_BYTES), be & 255); \
            }                                                                        \
            _Pragma("unroll")                                                        \
            for (int off = 16; off > 0; off >>= 1)                                   \
                partial += __shfl_xor_sync(0xffffffffu, partial, off);               \
            const float score   = partial * scale_factor;                            \
            const float new_max = fmaxf(max_val, score);                             \
            const float rescale  = expf(max_val - new_max);                          \
            const float exp_score = expf(score - new_max);                           \
            exp_sum = exp_sum * rescale + exp_score;                                 \
            for (int e = tid; e < head_dim; e += nthreads) acc[e] *= rescale;        \
            max_val = new_max;                                                       \
            for (int e = tid; e < head_dim; e += nthreads) {                         \
                const int be = t_offset + e;                                         \
                acc[e] += VALFN(v_data + (size_t)(be >> 8) * (SB_BYTES), be & 255) * exp_score; \
            }                                                                        \
        }                                                                            \
    }                                                                                \
    const float inv_sum = 1.0f / exp_sum;                                            \
    for (int e = tid; e < head_dim; e += nthreads)                                   \
        out[q_offset + e] = __float2half(acc[e] * inv_sum);                          \
}

DEFINE_PAGED_DECODE_QK_KERNEL(paged_attention_decode_q2_k_kernel, q2k_val, 84)
DEFINE_PAGED_DECODE_QK_KERNEL(paged_attention_decode_q3_k_kernel, q3k_val, 110)
DEFINE_PAGED_DECODE_QK_KERNEL(paged_attention_decode_q5_k_kernel, q5k_val, 176)
DEFINE_PAGED_DECODE_QK_KERNEL(paged_attention_decode_q6_k_kernel, q6k_val, 210)