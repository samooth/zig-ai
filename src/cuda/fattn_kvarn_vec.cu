//! KVarN native FA — vec path (lane-b1 Dev-B, B6 of TODO_B1_DEV_B).
//!
//! ## Scope (B6, TODO_B1_DEV_B §B6)
//!
//!   D=256 + SWA + GQA==2 + n_q==1 + fast-pairs (D5). Block (32, SLICES,
//!   DIM_GROUPS=4/SLICES) = 4 warps por block. Vec path es LA ruta
//!   especializada con mejor perf para D=256 + SWA en sm_80+; el
//!   dispatch master (B7) la elige cuando el geometry analyzer confirma
//!   la elegibilidad.
//!
//! ## Implementation status (B6, 1ª pasada)
//!
//!   PRAGMÁTICO: la primera pasada de B6 entrega:
//!     (a) Eligibility gate puro (B1) ⇒ `vecEligible` (B7 consume).
//!     (b) Kernel skeleton con la estructura del upstream (block, smem,
//!         reducción), UN SOLO template instantation: D=256, TPS=16,
//!         MAX_GQA=2, K/V bits=4 (el primer fast-pair k4v4). El resto de
//!         las 16 combinaciones default de D5 se irán añadiendo en
//!         iteraciones siguientes — el dispatch master sabrá cuál
//!         instanciar según el spec del usuario.
//!     (c) Zig launcher en fattn_kvarn.zig (mismo archivo) — wrapper
//!         de la única instanciación.
//!     (d) Test: SKIP hasta que Dev-A A3 (Sinkhorn bit-exacto) aterrice,
//!         porque sin eso no podemos probar el camino record.
//!
//!   El kernel consume los contratos de A0 (KvarnDesc + KvarnDesc::stage
//!   + kvarn_record_value) y la firma del portable. El Q NO se rota
//!   in-kernel (D3 in-kernel rotation es exclusiva del portable; vec
//!   asume pre-rotación por el caller, como upstream).

#include "kvarn_desc.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cfloat>

// ----------------------------------------------------------------------------
// Constantes y enums
// ----------------------------------------------------------------------------

enum {
    FKVEC_INVALID = 0,
    FKVEC_STAGE   = 1,
    FKVEC_RECORD  = 2,
};

struct FkvecRef {
    int source;       // FKVEC_*
    int pos;          // posición dentro del grupo (0..127)
    int stage_pos;    // índice absoluto en stage
    int record_group; // grupo record (no-SWA: stream*groups+group; SWA: group)
};

// Resolución de token → fuente (stage / record / invalid). Para SWA
// usamos desc.indices[token] (read_indirect siempre true bajo SWA).
// Transcripción del upstream `ggml_cuda_fattn_kvarn_vec_resolve`.
__device__ __forceinline__ FkvecRef fkvec_resolve(
    const KvarnDesc& d, int token)
{
    FkvecRef r;
    r.source = FKVEC_INVALID;
    r.pos = 0;
    r.stage_pos = 0;
    r.record_group = 0;
    if (token < 0) return r;

    int group;
    if (d.swa || d.read_indirect) {
        const int64_t enc = d.indices[token];
        if (enc == -1) return r;
        bool es = enc < -1;
        int aslot = -1;
        if (es) {
            const uint64_t payload = kvarn_index_payload(enc);
            const uint32_t packed = (uint32_t)(payload >> 32);
            aslot = packed == 0 ? -1 : (int)(packed - 1u);
        }
        const int64_t cell = (int64_t)(uint32_t)kvarn_index_payload(enc);
        group = (int)(cell / 128);
        r.pos = (int)(cell - (int64_t)group * 128);
        const bool from_stage = es || kvarn_group_from_stage(d, group);
        const bool from_record = !es && (d.read_indirect && !d.swa ?
            true : kvarn_group_from_record(d, group));
        if (from_stage) {
            r.source = FKVEC_STAGE;
            r.stage_pos = kvarn_stage_pos(d, group, r.pos, aslot);
        } else if (from_record) {
            r.source = FKVEC_RECORD;
            r.record_group = d.swa ? group % d.groups_per_stream :
                d.stream * d.groups_per_stream + group;
        }
        return r;
    }

    group = token / 128;
    r.pos = token - group * 128;
    if (kvarn_group_from_stage(d, group)) {
        r.source = FKVEC_STAGE;
        const int stage_base = d.stream * 128 * d.stage_groups;
        r.stage_pos = stage_base + (group == 0 ? r.pos :
            128 + ((group - 1) % d.tail_groups) * 128 + r.pos);
    } else if (kvarn_group_from_record(d, group)) {
        r.source = FKVEC_RECORD;
        r.record_group = d.stream * d.groups_per_stream + group;
    }
    return r;
}

// Load dequant: por ahora un STUB que sólo cubre K_BITS=4 / V_BITS=4 /
// D=256. La forma definitiva iterará sobre BITS ∈ {2,3,4,5,6,8} y
// value ∈ {K,V} con `kvarn_unpack` del A0. Tests E2E para el
// k4v4 únicamente — el resto de pares D5 son follow-up.
__device__ __forceinline__ float fkvec_load(
    const KvarnDesc& d, const FkvecRef& r, int slice, int dim)
{
    const int record_head = d.head_base + slice;
    if (r.source == FKVEC_STAGE) {
        // C2v2: fila K = 2*record_head. (D=256 ⇒ hd=128 per slice.)
        const int64_t idx = (int64_t)r.stage_pos * (2 * d.n_record_heads) * 128 +
            (int64_t)(2 * record_head) * 128 + dim;
        return __half2float(d.stage[idx]);
    }
    if (r.source == FKVEC_RECORD) {
        // A0 helper; usa d.bits/record_group/etc.
        return kvarn_record_value(d, r.record_group, r.pos, dim, slice);
    }
    return 0.0f;
}

// ----------------------------------------------------------------------------
// Unpack + load genérico por bits (lane-b1 Dev-B, B6 iter 3).
// ----------------------------------------------------------------------------
//
// Misma semántica que `kvarn_unpack` (A0): row-major, LSB-first, fast
// paths para 8/4/2, ventana genérica para 3/5/6. K tile: dim * 128 + pos
// (la fila es dim, la columna es token dentro del grupo).
template<int BITS>
__device__ __forceinline__ uint32_t fkvec_unpack(
    const uint8_t* payload, int index)
{
    if (BITS == 8) return payload[index];
    if (BITS == 4) {
        const uint8_t packed = payload[index >> 1];
        return (packed >> ((index & 1) * 4)) & 0x0Fu;
    }
    if (BITS == 2) {
        const uint8_t packed = payload[index >> 2];
        return (packed >> ((index & 3) * 2)) & 0x03u;
    }
    // Genérico (3/5/6).
    const int bit_off = index * BITS;
    const int byte_off = bit_off >> 3;
    const int shift = bit_off & 7;
    uint32_t window = payload[byte_off];
    if (shift + BITS > 8) window |= (uint32_t)payload[byte_off + 1] << 8;
    return (window >> shift) & ((1u << BITS) - 1u);
}

// Load dequant parametrizado por bits. Para STAGE es f16 puro (los
// bits no aplican — el stage es f16 rotado). Para RECORD decodifica
// vía C1 con la tabla de `kvarn_record_value` (espejo del kernel
// D128 de A0), pero reimplementado localmente con fkvec_unpack
// porque la versión A0 sólo cubre el caso slice=0.
template<int BITS>
__device__ __forceinline__ float fkvec_load_bits(
    const KvarnDesc& d, const FkvecRef& r, int slice, int dim)
{
    if (r.source == FKVEC_STAGE) {
        const int record_head = d.head_base + slice;
        // C2v2: fila K = 2*record_head, fila V = 2*record_head+1. El
        // lado lo codifica d.value (0=K, 1=V) — NO hardcodear +1.
        const int64_t idx = (int64_t)r.stage_pos * (2 * d.n_record_heads) * 128 +
            (int64_t)(2 * record_head + d.value) * 128 + dim;
        return __half2float(d.stage[idx]);
    }
    if (r.source != FKVEC_RECORD) return 0.0f;

    // Compute C1 record base.
    const uint8_t* rec = d.records +
        ((int64_t)r.record_group * d.n_record_heads + d.head_base + slice) * d.record_bytes;
    if (d.value == 0) {
        // K: row=dim, col=pos. Region layout: payload (D*128*BITS/8)
        // + k_s_col (hd*2) + k_zp (hd*2) + k_s_row (128*2).
        const int hd = 128; // slice-local
        const int payload_bytes = 128 * 128 * BITS / 8;
        const uint8_t* payload = rec;
        const __half* scale = (const __half*)(rec + payload_bytes);
        const __half* zp    = (const __half*)(rec + payload_bytes + hd * 2);
        const __half* other = (const __half*)(rec + payload_bytes + hd * 4);
        const int q = (int)fkvec_unpack<BITS>(payload, dim * 128 + r.pos);
        return ((float)q * __half2float(scale[dim]) + __half2float(zp[dim])) *
            __half2float(other[r.pos]);
    } else {
        // V: row=pos, col=dim. offset_v = k_region bytes.
        const int hd = 128;
        const int k_payload_bytes = 128 * 128 * BITS / 8;
        const int k_region = k_payload_bytes + hd * 2 + hd * 2 + 128 * 2;
        const uint8_t* v = rec + k_region;
        const int v_payload_bytes = 128 * 128 * BITS / 8;
        const uint8_t* payload = v;
        const __half* scale = (const __half*)(v + v_payload_bytes);
        const __half* zp    = (const __half*)(v + v_payload_bytes + 128 * 2);
        const __half* other = (const __half*)(v + v_payload_bytes + 128 * 4);
        const int q = (int)fkvec_unpack<BITS>(payload, r.pos * 128 + dim);
        return ((float)q * __half2float(scale[r.pos]) + __half2float(zp[r.pos])) *
            __half2float(other[dim]);
    }
}

// ----------------------------------------------------------------------------
// Kernel D=256, TPS=16, MAX_GQA=2, K/V bits=4
// ----------------------------------------------------------------------------
//
// Block: (32, 1, 1) = 32 threads = 1 warp. Por warp: la mitad
// izquierda cubre dims 0..63 del slice 0, la mitad derecha dims 64..127
// del slice 0. Para D=256, el bloque trabaja SOLO el slice 0 (SLICES=2
// requeriría 64 threads, fuera del alcance de un warp). En B6 +
// iteración siguiente se amplía a (32, 2) para cubrir ambos slices.

template<int TPS>
__device__ __forceinline__ void fkvec_dot_warp(
    const float* q_sh,        // [MAX_GQA=2][D=256] compartido
    const float* k_slice,     // [TPS][D=128] del slice actual
    float* score_partial,     // [MAX_GQA][TPS] salida
    int h, int tid)
{
    float acc = 0.0f;
    // Cada lane del warp cubre dims pares (h, h+32, h+64, h+96) del
    // slice. Q-rotado ya está en q_sh.
    #pragma unroll
    for (int lane_off = 0; lane_off < 128; lane_off += 32) {
        const int d = lane_off + tid;
        acc += q_sh[h * 128 + d] * k_slice[d];
    }
    // Reducir entre los TPS lanes (TPS=16 ⇒ lanes 0..15 hacen el
    // sum; lanes 16..31 = 0).
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off, 32);
    }
    if (tid < TPS) {
        score_partial[h * TPS + tid] = acc;
    }
}

// ----------------------------------------------------------------------------
// D=256 vec kernel — body refactor (B6 iter 3, lane-b1 Dev-B)
// ----------------------------------------------------------------------------
//
// El body del kernel vec vive en `fkvec_d256_body<BITS_K, BITS_V>` (función
// `__device__`). Las 15 wrappers `__global__` restantes de D5 (más el
// k4v4 canónico) sólo pasan los bits como template params y delegan.
//
// Esto resuelve el problema de B6 iter 2: un `__global__` no puede
// llamar a otro `__global__` con template args explícitos, pero SÍ
// puede llamar a un `__device__` con template args.
//
// BUG B6_ITER3_BUG (iter 10, fix): la reducción warp del score
// aplicaba `__shfl_xor_sync(..., 16, 32)` que CRUZA slices. En este
// kernel, lanes 0..15 cubren slice 0 (tokens 0..15) y lanes 16..31
// cubren slice 1 (también tokens 0..15 del otro slice). El XOR16
// mezclaba scores de tokens DISTINTOS entre slices, produciendo
// "uncorrelated values" en el output (observado por Dev-A en su
// absorbed test M1). Fix: el XOR 16 SE OMITE. La reducción es
// xor 8/4/2/1 sobre los 16 lanes DE CADA SLICE (no sobre el warp
// entero). 4 niveles (2^4 = 16) ⇒ suficiente para sumar
// completamente.
//
// Online softmax cross-chunk: la cadena es la del iter 3 anterior
// (factor = exp(m_old - m_new), rescale out_local, V-pass).
// Después de este fix de lanes, la cadena queda bien con n_kv
// múltiplo de TPS (16); pendiente para n_kv arbitrario.

template<int BITS_K, int BITS_V>
__device__ __forceinline__ void fkvec_d256_body(
    const float* q_data,
    KvarnDesc* k_descs,
    KvarnDesc* v_descs,
    float* dst_data,
    int n_kv, int n_q_heads, int n_kv_heads, int n_stream,
    float scale)
{
    constexpr int D = 256;
    constexpr int THREADS = 32;
    constexpr int TPS = 16;
    constexpr int DIMS_PER_LANE = D / 32; // 8
    constexpr int SLICES = 2;             // D/128

    const int query = (int) blockIdx.x;
    const int query_head = (int) blockIdx.y;
    const int stream = (int) blockIdx.z;
    const int tid = (int) threadIdx.x;
    if (query >= 1) return;            // vec eligibility: n_q=1
    if (query_head >= n_q_heads) return;
    if (tid >= THREADS) return;
    if (n_q_heads != n_kv_heads * 2) return; // GQA=2 estricto

    const int gqa = 2;
    const int kv_head = query_head / gqa;

    const float* q = q_data
        + ((size_t) query * n_q_heads + query_head) * n_stream * D
        + (size_t) stream * D;

    const int my_slice = tid / TPS;       // 0 or 1
    const int my_t = tid - my_slice * TPS; // 0..15 dentro de mi slice
    const int slice_base = my_slice * 128;

    __shared__ float q_sh[D];
    // C2v2 K/V layout: 2 slices × TPS tokens × 128 dims. Cada lane cubre
    // UNA slice (lanes 0..15 ⇒ slice 0; lanes 16..31 ⇒ slice 1) y la
    // softmax se mantiene por slice ⇒ sólo necesitamos los 128 dims de
    // nuestra slice en smem.
    __shared__ float k_sm[2 * TPS * 128];
    __shared__ float v_sm[2 * TPS * 128];
    __shared__ float score[2 * TPS];
    __shared__ float m_shared[2];
    __shared__ float l_shared[2];
    __shared__ float old_scale_shared[2];
    __shared__ float acc_out[D];

    // Cargar Q y aplicar WHT-128 in-kernel (D3) — 2 slices.
    if (tid < 32) {
        q_sh[tid]        = q[tid];
        q_sh[tid + 32]   = q[tid + 32];
        q_sh[tid + 64]   = q[tid + 64];
        q_sh[tid + 96]   = q[tid + 96];
        q_sh[tid + 128]  = q[tid + 128];
        q_sh[tid + 160]  = q[tid + 160];
        q_sh[tid + 192]  = q[tid + 192];
        q_sh[tid + 224]  = q[tid + 224];
    }
    __syncthreads();

    // WHT-128 in-place sobre q_sh (2 pasadas, 1 por slice). La
    // normalización 1/sqrt(128) se aplica a los 128 dims de cada
    // slice: 16 lanes × DIMS_PER_LANE=8 dims/lane = 128 dims.
    #pragma unroll
    for (int slice = 0; slice < SLICES; ++slice) {
        const int base = slice * 128;
        #pragma unroll
        for (int stride = 1; stride < 128; stride <<= 1) {
            for (int pair = tid; pair < 64; pair += 32) {
                const int j = (pair / stride) * (2 * stride) + (pair % stride);
                const float a = q_sh[base + j];
                const float b = q_sh[base + j + stride];
                q_sh[base + j] = a + b;
                q_sh[base + j + stride] = a - b;
            }
            __syncthreads();
        }
        if (tid < 2 * TPS) {
            const int d_base = base + my_t * DIMS_PER_LANE;
            #pragma unroll
            for (int d = 0; d < DIMS_PER_LANE; ++d) {
                q_sh[d_base + d] *= 0.08838834764831845f;
            }
        }
        __syncthreads();
    }

    // Init softmax + acc_out.
    if (tid < SLICES) {
        m_shared[tid] = -FLT_MAX;
        l_shared[tid] = 0.0f;
        old_scale_shared[tid] = 1.0f;
    }
    if (tid < D) {
        acc_out[tid] = 0.0f;
    }
    __syncthreads();

    const KvarnDesc& k_desc = k_descs[(size_t) stream * n_kv_heads + kv_head];
    const KvarnDesc& v_desc = v_descs[(size_t) stream * n_kv_heads + kv_head];

    float out_local[DIMS_PER_LANE];
    #pragma unroll
    for (int d = 0; d < DIMS_PER_LANE; ++d) {
        out_local[d] = 0.0f;
    }

    for (int token_base = 0; token_base < n_kv; token_base += TPS) {
        // 1) Carga K/V al smem con fkvec_load_bits<BITS_K/BITS_V>.
        // Cada lane (tid) cubre UNA slice (my_slice = tid / TPS) y carga
        // los 128 dims de su slice para TPS tokens.
        for (int t = 0; t < TPS; ++t) {
            const int tok = token_base + t;
            if (tok >= n_kv) break;
            const FkvecRef kr = fkvec_resolve(k_desc, tok);
            const FkvecRef vr = fkvec_resolve(v_desc, tok);
            for (int d_off = 0; d_off < 128; d_off += 32) {
                const int d = d_off + tid;
                k_sm[my_slice * TPS * 128 + t * 128 + d] =
                    fkvec_load_bits<BITS_K>(k_desc, kr, my_slice, d);
                v_sm[my_slice * TPS * 128 + t * 128 + d] =
                    fkvec_load_bits<BITS_V>(v_desc, vr, my_slice, d);
            }
        }
        __syncthreads();

        // 2) Score per (slice, t). Warp shuffle reduction 32→TPS.
        // BUG B6_ITER3_BUG: el XOR 16 cruza slices. La distribución de
        // lanes en el warp es: lanes 0..15 cubren slice 0, lanes 16..31
        // cubren slice 1. Una reducción xor16 entre slices mezcla
        // scores de tokens DISTINTOS, corrompiendo la salida
        // ("uncorrelated values" observado por Dev-A en el E2E).
        // Fix: xor 8/4/2/1 sobre los 16 lanes DE CADA SLICE. El xor16
        // se omite a propósito.
        float my_score = 0.0f;
        #pragma unroll
        for (int d = 0; d < 128; d += 32) {
            my_score += q_sh[slice_base + d + tid] *
                k_sm[my_slice * TPS * 128 + my_t * 128 + d + tid];
        }
        // Reducción DENTRO de cada slice (16 lanes). xor 16 NO
        // se aplica — cruza slices.
        my_score += __shfl_xor_sync(0xFFFFFFFFu, my_score, 8, 32);
        my_score += __shfl_xor_sync(0xFFFFFFFFu, my_score, 4, 32);
        my_score += __shfl_xor_sync(0xFFFFFFFFu, my_score, 2, 32);
        my_score += __shfl_xor_sync(0xFFFFFFFFu, my_score, 1, 32);
        if (tid < 2 * TPS) {
            score[my_slice * TPS + my_t] = my_score * scale;
        }
        __syncthreads();

        // Combina los scores parciales de ambos slices: el score FULL
        // de D=256 es s0[t] + s1[t] (la WHT es ortonormal POR slice ⇒
        // el producto parcial de cada slice es exactamente Qs·Ks). La
        // softmax debe operar sobre el score COMBINADO — dos softmax
        // independientes NO son equivalentes a la atención estándar.
        if (tid < TPS) {
            score[tid] += score[TPS + tid];
        }
        __syncthreads();

        // 3) Softmax online (m, l, old_scale) — UN solo estado global
        // (tid 0). Los scores combinados viven en score[0..TPS).
        if (tid == 0) {
            const float m_shared_s = m_shared[0];
            float m_local = m_shared_s;
            float l_local = l_shared[0];
            for (int t = 0; t < TPS; ++t) {
                const float sc = score[t];
                const float m_new = fmaxf(m_local, sc);
                const float old_s = (m_local == -FLT_MAX) ? 0.0f : __expf(m_local - m_new);
                const float w = __expf(sc - m_new);
                l_local = l_local * old_s + w;
                m_local = m_new;
            }
            const float factor = (m_shared_s == -FLT_MAX) ? 1.0f : __expf(m_shared_s - m_local);
            m_shared[0] = m_local;
            l_shared[0] = l_local;
            old_scale_shared[0] = factor;
        }
        __syncthreads();
        // Todos los lanes (ambos slices) rescalean su out_local con el
        // ÚNICO factor global.
        const float my_factor = old_scale_shared[0];
        #pragma unroll
        for (int d = 0; d < DIMS_PER_LANE; ++d) {
            out_local[d] *= my_factor;
        }

        // 4) V-pass + output accumulation. AMBOS slices usan los
        // MISMOS pesos (score combinado) — cada slice acumula su mitad
        // de V con los pesos de la atención completa.
        for (int t = 0; t < TPS; ++t) {
            const float sc = score[t];
            const float m_local = m_shared[0];
            float weight = 0.0f;
            if (sc > -FLT_MAX / 2.0f) {
                weight = __expf(sc - m_local);
            }
            #pragma unroll
            for (int d = 0; d < DIMS_PER_LANE; ++d) {
                out_local[d] += weight *
                    v_sm[my_slice * TPS * 128 + t * 128 + my_t * DIMS_PER_LANE + d];
            }
        }

        __syncthreads();
    }

    // 5) Output write: copiar out_local a acc_out + denom + WHT⁻¹.
    #pragma unroll
    for (int d = 0; d < DIMS_PER_LANE; ++d) {
        const int d_global = slice_base + my_t * DIMS_PER_LANE + d;
        acc_out[d_global] = out_local[d];
    }
    __syncthreads();

    __syncthreads();

    if (tid < 2 * TPS) {
        // AMBOS slices dividen por el MISMO denom (softmax única
        // sobre el score combinado): 32 lanes × 8 dims = 256 dims.
        const float denom = l_shared[0] > 0.0f ? l_shared[0] : 1.0f;
        const int d_base = my_slice * 128 + my_t * DIMS_PER_LANE;
        #pragma unroll
        for (int d = 0; d < DIMS_PER_LANE; ++d) {
            acc_out[d_base + d] /= denom;
        }
    }
    __syncthreads();

    // WHT⁻¹ (mismo butterfly, self-inverse). Normalización por 16
    // lanes × DIMS_PER_LANE=8 dims/lane = 128 dims por slice.
    #pragma unroll
    for (int slice = 0; slice < SLICES; ++slice) {
        const int base = slice * 128;
        #pragma unroll
        for (int stride = 1; stride < 128; stride <<= 1) {
            for (int pair = tid; pair < 64; pair += 32) {
                const int j = (pair / stride) * (2 * stride) + (pair % stride);
                const float a = acc_out[base + j];
                const float b = acc_out[base + j + stride];
                acc_out[base + j] = a + b;
                acc_out[base + j + stride] = a - b;
            }
            __syncthreads();
        }
        if (tid < 2 * TPS) {
            const int d_base = base + my_t * DIMS_PER_LANE;
            #pragma unroll
            for (int d = 0; d < DIMS_PER_LANE; ++d) {
                acc_out[d_base + d] *= 0.08838834764831845f;
            }
        }
        __syncthreads();
    }

    float* output = dst_data
        + ((size_t) query_head * 1 + query) * n_stream * D
        + (size_t) stream * D;
    // Cada lane escribe su(s) dim(s) del accumulator a output: 16
    // lanes por slice × DIMS_PER_LANE=8 dims = 128 dims por slice.
    {
        const int d_base = my_slice * 128 + my_t * DIMS_PER_LANE;
        #pragma unroll
        for (int d = 0; d < DIMS_PER_LANE; ++d) {
            output[d_base + d] = acc_out[d_base + d];
        }
    }
}

// ----------------------------------------------------------------------------
// Wrappers `__global__` de D5 (lane-b1 Dev-B, B6 iter 3)
// ----------------------------------------------------------------------------
//
// Cada par k_bits/v_bits del spec D5 (16 pares) es un wrapper thin que
// llama al `__device__` body parametrizado. El dispatcher (B7) elige
// el wrapper por nombre.
//
// El k4v4 sigue siendo el original `fattn_kvarn_vec_d256_k4v4_kernel`
// (línea ~244); las 15 nuevas wrappers siguen el mismo patrón de
// signature + forwarding.

#define FKVEC_LAUNCHER(NAME, BITS_K, BITS_V) \
    extern "C" __global__ void NAME( \
        const float* q_data, KvarnDesc* k_descs, KvarnDesc* v_descs, \
        const __half* mask_data, float* dst_data, \
        int n_kv, int n_q_heads, int n_kv_heads, int n_stream, float scale) \
    { \
        (void)mask_data; /* vec path no usa mask_data en B6 iter 1-3 */ \
        fkvec_d256_body<BITS_K, BITS_V>(q_data, k_descs, v_descs, dst_data, \
            n_kv, n_q_heads, n_kv_heads, n_stream, scale); \
    }

FKVEC_LAUNCHER(fattn_kvarn_vec_d256_k8v8_kernel, 8, 8)
FKVEC_LAUNCHER(fattn_kvarn_vec_d256_k8v6_kernel, 8, 6)
FKVEC_LAUNCHER(fattn_kvarn_vec_d256_k8v5_kernel, 8, 5)
FKVEC_LAUNCHER(fattn_kvarn_vec_d256_k6v6_kernel, 6, 6)
FKVEC_LAUNCHER(fattn_kvarn_vec_d256_k6v5_kernel, 6, 5)
FKVEC_LAUNCHER(fattn_kvarn_vec_d256_k6v4_kernel, 6, 4)
FKVEC_LAUNCHER(fattn_kvarn_vec_d256_k5v5_kernel, 5, 5)
FKVEC_LAUNCHER(fattn_kvarn_vec_d256_k5v4_kernel, 5, 4)
FKVEC_LAUNCHER(fattn_kvarn_vec_d256_k5v3_kernel, 5, 3)
FKVEC_LAUNCHER(fattn_kvarn_vec_d256_k4v3_kernel, 4, 3)
FKVEC_LAUNCHER(fattn_kvarn_vec_d256_k4v2_kernel, 4, 2)
FKVEC_LAUNCHER(fattn_kvarn_vec_d256_k3v3_kernel, 3, 3)
FKVEC_LAUNCHER(fattn_kvarn_vec_d256_k3v2_kernel, 3, 2)
FKVEC_LAUNCHER(fattn_kvarn_vec_d256_k2v2_kernel, 2, 2)

#undef FKVEC_LAUNCHER

// k4v4: el original `fattn_kvarn_vec_d256_k4v4_kernel` queda — los
// 14 macros de arriba son los pares restantes D5 (más k4v4 = 15 total
// wrappers; el canónico k4v4 es la implementación que ya existía).

// Wrapper de k4v4 (lane-b1 Dev-B, B6 iter 3): pasa por el mismo
// `__device__` body que el resto. El kernel canónico k4v4 (línea ~244
// en este fichero) se mantiene por compatibilidad pero ya no se usa
// desde el wrapper — la macro de abajo lo delega al body.
extern "C" __global__ void fattn_kvarn_vec_d256_k4v4_kernel(
    const float* q_data,
    KvarnDesc* k_descs,
    KvarnDesc* v_descs,
    const __half* mask_data,
    float* dst_data,
    int n_kv, int n_q_heads, int n_kv_heads, int n_stream, float scale)
{
    (void)mask_data; // B6 iter 1-3 vec path no usa mask
    fkvec_d256_body<4, 4>(q_data, k_descs, v_descs, dst_data,
        n_kv, n_q_heads, n_kv_heads, n_stream, scale);
}
