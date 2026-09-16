// KVarN FA — decode-split MMA (lane-b1 A10, Dev A).
//
// Transcrito del análisis upstream (fattn-mma-kvarn-decode.cuh) adaptado a
// C1 y a las primitivas de mma_kvarn.cuh. Dos fases:
//   1) kvarn_decode_mma_kernel: partials por split (SPLIT_TOKENS=64) +
//      meta (m, denom) por (stream, q, head, split).
//   2) kvarn_decode_combine_kernel: reduce global → dst final.
//
// Fragmentos (tile<16,8> Turing, elementos 32-bit):
//   A (K o V): get_i(l) = (l/2)·8 + lane/4  → token_local (K) / token (V)
//              get_j(l) = l·4 + lane%4      → PAR de dims (2·get_j)
//   B (Q o P): ldmatrix desde smem half2-stride.
// Q se pre-proyecta (truco zq) cuando el split cae en un solo record-group:
//   q' = scale_axis[dim]·q ; zq = Σ zp_axis[dim]·q ;
//   score = other[pos]·(mma + zq)  (absorbe zp sin materializar K).

#include "kvarn_desc.cuh"
#include "mma_kvarn.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cfloat>

#define KVARN_DECODE_THREADS 256
#define KVARN_DECODE_CHUNK 16

// Fallback per-element (rotated domain, stage o record).
__device__ __forceinline__ float kvarn_decode_load_rotated(
    const KvarnDesc& d, int token, int slice, int dim)
{
    const int group = token / KVARN_DIM;
    const int pos = token - group * KVARN_DIM;
    if (kvarn_group_from_stage(d, group)) {
        const int sp = kvarn_stage_pos(d, group, pos, -1);
        // C2v2: fila K=2*head, V=2*head+1 (el desc sabe su lado por
        // d.value). El record_head de kvarn_stage_row NO distingue
        // lados: el V leia la fila K (outputs mezclados).
        const int row = d.head_base + slice;
        const int stage_row = d.value ? 2 * row + 1 : 2 * row;
        return __half2float(d.stage[
            ((int64_t)sp * (2 * d.n_record_heads) + stage_row) * KVARN_DIM + dim]);
    }
    if (!kvarn_group_from_record(d, group)) return 0.0f;
    const int rg = d.swa ? (group % d.groups_per_stream)
        : (d.stream * d.groups_per_stream + group);
    return kvarn_record_value(d, rg, pos, dim, slice);
}


// WHT-128 in-place sobre smem (butterfly + 1/√128; involutiva).
// Requiere blockDim.x == 128 en el eje x de esta fase (bloque cooperativo).
static __device__ __forceinline__ void kvarn_wht_128_shared(float* v, int tid)
{
    // Todos los threads del bloque participan; el reparto de pares debe
    // cubrir 64 pares POR STRIDE independiente del blockDim (el combine
    // usa 256 threads: con `pair += nthreads` solo hacía el primer stride
    // ⇒ WHT incompleta ⇒ output corrupto).
    const int nthreads = blockDim.x * blockDim.y;
    for (int stride = 1; stride < KVARN_DIM; stride <<= 1) {
        const int npairs = KVARN_DIM / 2;
        // Solo los primeros npairs threads participan (con 128 threads
        // y 64 pairs, tid>=64 duplicaría pares ⇒ doble aplicación).
        for (int pair = tid; pair < npairs; pair += nthreads) {
            const int j = (pair / stride) * (2 * stride) + (pair % stride);
            const float a = v[j];
            const float b = v[j + stride];
            v[j] = a + b;
            v[j + stride] = a - b;
        }
        __syncthreads();
    }
    if (tid < KVARN_DIM) v[tid] *= 0.08838834764831845f;
}

template <int D, int MAX_GQA, int SPLIT_TOKENS, int NWARPS,
          int K_BITS, int V_BITS>
__device__ void kvarn_decode_mma_kernel(
    const float* Q,
    const KvarnDesc* k_descs,
    const KvarnDesc* v_descs,
    const __half* mask,
    float* partial,
    float2* partial_meta,
    float scale,
    int n_kv,
    int n_q,
    int n_q_heads,
    int n_kv_heads,
    int gqa_ratio,
    int n_gqa_blocks,
    int n_splits)
{
    constexpr int SLICES = D / KVARN_DIM;
    constexpr int TOKENS_PER_CHUNK = KVARN_DECODE_CHUNK;
    constexpr int TOKEN_CHUNKS = SPLIT_TOKENS / TOKENS_PER_CHUNK;
    constexpr int WARPS_PER_CHUNK = SLICES;
    constexpr int CHUNKS_PER_PASS = NWARPS / WARPS_PER_CHUNK;

    using T_A = kvarn_mma::TileA;
    using T_B = kvarn_mma::TileB;
    using T_C = kvarn_mma::TileC;

    const int split = blockIdx.x;
    const int q_index = blockIdx.y % n_q;
    const int gqa_block = (blockIdx.y / n_q) % n_gqa_blocks;
    const int kv_head = blockIdx.y / (n_q * n_gqa_blocks);
    const int stream = blockIdx.z;
    const int lane = (int)threadIdx.x;
    const int warp = (int)threadIdx.y;
    const int tid = warp * 32 + lane;

    const KvarnDesc& k_desc = k_descs[(size_t)stream * n_kv_heads + kv_head];
    const KvarnDesc& v_desc = v_descs[(size_t)stream * n_kv_heads + kv_head];

    const int q_head0 = kv_head * gqa_ratio + gqa_block * MAX_GQA;
    const int gqa_head_count = min(MAX_GQA, gqa_ratio - gqa_block * MAX_GQA);
    const int token_begin = split * SPLIT_TOKENS;
    const int token_end = min(n_kv, token_begin + SPLIT_TOKENS);
    const int group = token_begin / KVARN_DIM;
    const int group_pos_begin = token_begin - group * KVARN_DIM;

    // Plan tile (non-SWA direct): fast si el grupo está sellado y el split
    // cabe entero en el grupo.
    const bool k_from_record = !k_desc.swa && kvarn_group_from_record(k_desc, group);
    const bool k_split_in_group =
        k_from_record && (group_pos_begin + SPLIT_TOKENS) <= KVARN_DIM;
    const int record_group_k = k_desc.stream * k_desc.groups_per_stream + group;
    const bool v_from_record = !v_desc.swa && kvarn_group_from_record(v_desc, group);
    const bool v_split_in_group =
        v_from_record && (group_pos_begin + SPLIT_TOKENS) <= KVARN_DIM;
    const int record_group_v = v_desc.stream * v_desc.groups_per_stream + group;

    const uint8_t* k_rec = k_desc.records
        + ((int64_t)record_group_k * k_desc.n_record_heads + k_desc.head_base)
            * k_desc.record_bytes;

    // Ejes C1 del record (D=128 ⇒ hd regions = 128):
    //   K: [k_payload][k_s_col hd][k_zp hd][k_s_row 128]
    //   V: [v_payload][v_s_col hd][v_s_row 128][v_zp 128]
    const int k_payload_bytes = KVARN_DIM * D * K_BITS / 8;
    const __half* k_scale_ax = (const __half*)(k_rec + k_payload_bytes);
    const __half* k_zp_ax = k_scale_ax + D;
    const __half* k_other_ax = k_zp_ax + D;

    __shared__ __align__(16) __half q_sh[D][8]; // [dim][8 heads] 16B rows
    __shared__ __align__(16) float score_sh[MAX_GQA][SPLIT_TOKENS];
    // BUG OOB-softmax: con n_kv < n_splits*SPLIT, los slots inválidos
    // quedaban en 0 y el softmax los sumaba como e^(0-m) ⇒ dilución
    // (rel 12 con n_kv=16). Init a -inf; solo slots < n_kv reciben score.
    {
        const int n_init = MAX_GQA * SPLIT_TOKENS;
        for (int i = tid; i < n_init; i += NWARPS * 32) {
            score_sh[i / SPLIT_TOKENS][i % SPLIT_TOKENS] = -FLT_MAX / 2.0f;
        }
    }
    __syncthreads();
    __shared__ __align__(16) __half p_sh[SPLIT_TOKENS][8]; // [token][8 heads] 16B rows
    // Staging tiles MMA: [warp][...]. k_stage: [token m][dim k] (score);
    // v_stage: [dim m][token k] (V-pass). 4 warps → sin races.
    __shared__ __align__(16) __half k_stage[4][16][16];
    __shared__ __align__(16) __half v_stage[4][16][16];
    __shared__ float k_scale_s[SLICES][KVARN_DIM];
    __shared__ float k_zp_s[SLICES][KVARN_DIM];
    __shared__ float k_other_s[SLICES][KVARN_DIM];
    __shared__ float zq_s[SLICES][MAX_GQA];
    __shared__ float m_sh[MAX_GQA];
    __shared__ float denom_sh[MAX_GQA];

    // ---- 0) Init q_sh a 0: cols ghost (heads >= gqa_head_count)
    // contenían basura de smem no inicializada ⇒ el MMA acumulaba ruido
    // en los fragmentos y contaminaba los scores de TODOS los heads.
    {
        const int n_init = D * 8;
        for (int i = tid; i < n_init; i += NWARPS * 32) {
            ((__half*)q_sh)[i] = __float2half(0.0f);
        }
        __syncthreads();
    }

    // ---- 1) Q load → q_sh [dim][8] (B-tile: k=dim, n=head) + WHT⁻¹ (D3).
    // El split trabaja en dominio ROTADO (K/V de stage/records ya lo están):
    // Q se rota al cargar con WHT-128 (involutiva) POR HEAD; el combine
    // de-rotará el output. Todo el dot product queda en rotado y la
    // rotación es ortogonal ⇒ softmax equivale al dominio original.
    //
    // FIX HANG: el bucle ANTERIOR tenía `continue` que saltaba
    // __syncthreads() en threads con head inválido ⇒ barrier divergence
    // ⇒ deadlock in-device (40+ min de hang). Reescrito: TODOS los
    // threads ejecutan el MISMO número de barriers; el guard de head
    // solo controla la ESCRITURA, nunca el control de flujo.
    __shared__ float qrow[128];
    for (int h = 0; h < gqa_head_count; ++h) {
        const bool valid_head = (q_head0 + h < n_q_heads);
        const float* q = Q
            + ((size_t)stream * n_q_heads + q_head0 + h) * (size_t)n_q * D
            + (size_t)q_index * D;
        // Carga estratificada: tid cubre [0, NWARPS*32) = [0,128).
        for (int d = tid; d < D; d += NWARPS * 32) {
            // Scale en el LOAD (semántica upstream): con el zq trick el
            // scale DEBE entrar en Q antes de zp_ax·q — si se aplica en
            // el score-store, zq queda sin escalar y los splits record
            // divergen (m ~0.57 pero scores ~2x).
            qrow[d] = valid_head ? q[d] : 0.0f;
        }
        __syncthreads();
        kvarn_wht_128_shared(qrow, tid);
        __syncthreads();
        if (valid_head) {
            for (int d = tid; d < D; d += NWARPS * 32) {
                q_sh[d][h] = __float2half(qrow[d]);
            }
        }
        __syncthreads();
    }

    // ---- 2) Pre-proyección de ejes (zq trick) ----
    if (k_split_in_group) {
        for (int i = tid; i < SLICES * KVARN_DIM; i += NWARPS * 32) {
            const int s = i / KVARN_DIM;
            const int ax = i % KVARN_DIM;
            k_scale_s[s][ax] = __half2float(k_scale_ax[s * KVARN_DIM + ax]);
            k_zp_s[s][ax] = __half2float(k_zp_ax[s * KVARN_DIM + ax]);
            k_other_s[s][ax] = __half2float(k_other_ax[ax]);
        }
    }

    __syncthreads();
#ifdef KVARN_SPLIT_DEBUG
    __syncthreads();
    if (blockIdx.y == 0 && blockIdx.z == 0 &&
            threadIdx.x == 0 && threadIdx.y == 0) {
        printf("[split] blk=%d grp=%d posb=%d k_rec=%d k_sig=%d v_rec=%d v_sig=%d\n",
            (int)blockIdx.x, group, group_pos_begin,
            (int)k_from_record, (int)k_split_in_group,
            (int)v_from_record, (int)v_split_in_group);
    }
    __syncthreads();
#endif
    if (k_split_in_group) {
        const int n_targets = SLICES * gqa_head_count;
        for (int t = tid; t < n_targets; t += NWARPS * 32) {
            const int s = t / gqa_head_count;
            const int h = t % gqa_head_count;
            float zq = 0.0f;
            for (int dim = 0; dim < KVARN_DIM; ++dim) {
                const float qv = __half2float(q_sh[s * KVARN_DIM + dim][h]);
                zq += k_zp_s[s][dim] * qv;
                q_sh[s * KVARN_DIM + dim][h] =
                    __float2half(k_scale_s[s][dim] * qv);
            }
            zq_s[s][h] = zq;
#ifdef KVARN_SPLIT_DEBUG
            if (s == 0 && h == 0 && tid == 0) {
                printf("[split] zq s0h0=%.6f k_scale[0..3]=%.6f %.6f zp=%.6f %.6f other[0..3]=%.6f %.6f %.6f %.6f\n",
                    zq, k_scale_s[0][0], k_scale_s[0][1],
                    k_zp_s[0][0], k_zp_s[0][1],
                    k_other_s[0][0], k_other_s[0][1], k_other_s[0][2], k_other_s[0][3]);
            }
#endif
        }
    }
    __syncthreads();

    const int local_chunk = warp / WARPS_PER_CHUNK;
    const int warp_in_chunk = warp % WARPS_PER_CHUNK; // slice
    const int chunk_of_warp = local_chunk;
    const __half* mask_h = (mask != nullptr)
        ? (const __half*)mask + (size_t)stream * n_kv
        : nullptr;

#ifdef KVARN_SPLIT_DEBUG
    if (tid == 0 && blockIdx.x >= 2) {
        // V y K dequant del record group (splits 2,3) — comparar con
        // portable/kvarn_record_value host-side.
        const float kv0 = kvarn_record_value(k_desc, record_group_k, 0, 0, 0);
        const float kv1 = kvarn_record_value(k_desc, record_group_k, 1, 0, 0);
        const float vv0 = kvarn_record_value(v_desc, record_group_v, 0, 0, 0);
        const float vv1 = kvarn_record_value(v_desc, record_group_v, 1, 0, 0);
        printf("[split] s=%d rg_k=%d rg_v=%d kv(0,0)=%.6f kv(1,0)=%.6f vv(0,0)=%.6f vv(1,0)=%.6f\n",
            (int)blockIdx.x, record_group_k, record_group_v, kv0, kv1, vv0, vv1);
    }
#endif
    // ---- 3) Score chunks (K·Qᵀ via MMA) ----
    #pragma unroll 1
    for (int chunk_base = 0; chunk_base < TOKEN_CHUNKS; chunk_base += CHUNKS_PER_PASS) {
        const int chunk = chunk_base + chunk_of_warp;
        const bool chunk_active = chunk < TOKEN_CHUNKS;
        const int token0 = token_begin + chunk * TOKENS_PER_CHUNK;
        T_C scores;
        #pragma unroll
        for (int l = 0; l < T_C::ne; ++l) scores.x[l] = 0.0f;

        if (chunk_active) {
            // Staging K por paso: 16 tokens × 16 dims en smem row-major
            // (token fila, dim col) + ldmatrix_a x4 — el MISMO camino
            // validado por el smoke A9. El fill manual anterior con
            // frag_i/frag_j asumía el layout del wrapper GGML (que hace
            // swap interno de regs) que mi mma() asm directo no replica
            // ⇒ scores corruptos.
            // (k_stage/v_stage declaradas al scope del kernel, junto a
            // p_sh: el V-pass también usa v_stage.)
            #pragma unroll 1
            for (int dim0 = warp_in_chunk * KVARN_DIM;
                    dim0 < (warp_in_chunk + 1) * KVARN_DIM;
                    dim0 += 16) { // 16 dims por MMA
                for (int e = lane; e < 16 * 16; e += 32) {
                    const int trow = e / 16; // token 0..15
                    const int tcol = e % 16; // dim 0..15
                    // RACE FIX: k_stage la compartian los 4 warps
                    // (chunks en paralelo) con solo __syncwarp: overwrites
                    // cruzados. Buffer por warp, indexado por local_chunk.
                    const int my_stage = local_chunk;
                    const int dim = dim0 + tcol;
                    const int token = token0 + trow;
                    float x;
                    const int pos = group_pos_begin
                        + chunk * TOKENS_PER_CHUNK + trow;
                    if (k_split_in_group) {
                        // A/B DEBUG: dequant completo (path portable,
                        // sin zq trick) para aislar el bug del zq path.
                        x = kvarn_record_value(
                            k_desc, record_group_k, pos,
                            dim - warp_in_chunk * KVARN_DIM, warp_in_chunk);
                    } else {
                        x = (token < token_end)
                            ? kvarn_decode_load_rotated(k_desc, token, warp_in_chunk,
                                dim - warp_in_chunk * KVARN_DIM)
                            : 0.0f;
                    }
                    k_stage[my_stage][trow][tcol] = __float2half(x);
                }
                __syncwarp();
                T_A k_a;
                kvarn_mma::load_ldmatrix_a(
                    k_a, &k_stage[local_chunk][0][0], 16);
                T_B q_b;
                // Q B-tile: k=dims (16 desde dim0), n=heads (8).
                // [dim][8] halfs ⇒ filas 16B alineadas.
                kvarn_mma::load_ldmatrix_b(
                    q_b, (__half*)q_sh + dim0 * 8, 8);
                kvarn_mma::mma(scores, k_a, q_b);
            }

            // Aplicar other + zq y descargar scores a smem.
            // Layout C PTX: lane L reg l → row=L/4+8·(l/2), col=2·(L%4)+l%2.
            #pragma unroll
            for (int l = 0; l < T_C::ne; ++l) {
                const int j = l / 2 * 8 + lane / 4;   // token local
                const int h = 2 * (lane % 4) + (l % 2); // head
                if (h < MAX_GQA) {
                    // El scale ya está en Q (load) — aquí raw. En path
                    // record: other·(raw + zq) como upstream.
                    float v = scores.x[l] * scale;
                    if (false && k_split_in_group && h < gqa_head_count) {
                        const int pos = group_pos_begin
                            + chunk * TOKENS_PER_CHUNK + j;
                        v = k_other_s[warp_in_chunk][pos] * (v + zq_s[warp_in_chunk][h]);
                    }
                    if (mask_h != nullptr) {
                        const int token = token0 + j;
                        if (token < n_kv) {
                            v += __half2float(mask_h[token]);
                        }
                    }
                    if (token_begin + chunk * TOKENS_PER_CHUNK + j < n_kv) {
                        score_sh[h][chunk * TOKENS_PER_CHUNK + j] = v;
                    }
                }
            }
        }
        __syncthreads();
    }

    // ---- 4) Softmax split-local por head ----
    if (tid < MAX_GQA) {
        const int h = tid;
        float m = -FLT_MAX / 2.0f;
        for (int t = 0; t < SPLIT_TOKENS; ++t) {
            m = fmaxf(m, score_sh[h][t]);
        }
        float denom = 0.0f;
        for (int t = 0; t < SPLIT_TOKENS; ++t) {
            const float diff = score_sh[h][t] - m;
            const float w = (diff >= -15.0f) ? __expf(diff) : 0.0f;
            denom += w;
            p_sh[t][h] = __float2half(w);
        }
        m_sh[h] = m;
        denom_sh[h] = denom;
    }
    __syncthreads();

    // ---- 5) V-pass: out = Σ_chunks MMA(V_a, P_b) ----
    // V record: fila=token ⇒ payload[pos·128 + ldim].
    // Cada warp cubre dims [warp·(D/NWARPS), +D/NWARPS).
    const int dims_per_warp = D / NWARPS;
    // FIX: el tile A del V-pass es m=16 pero dims_per_warp=32 (D=128,
    // NWARPS=4): el loop exterior (un dim0 por warp) dejaba 16 de cada
    // 32 dims SIN calcular. Ahora: dos sub-tiles por warp (dim0 y
    // dim0+16), escribiendo todos los D dims.
    for (int dim_base = warp * dims_per_warp; dim_base < warp * dims_per_warp + dims_per_warp; dim_base += 16) {
        const int dim0 = dim_base;
        T_C out;
        #pragma unroll
        for (int l = 0; l < T_C::ne; ++l) out.x[l] = 0.0f;
        #pragma unroll 1
        for (int chunk = 0; chunk < TOKEN_CHUNKS; ++chunk) {
            // A = V con m=DIM, k=TOKEN. Fill manual frag_i/frag_j
            // ELIMINADO: asumía el reg-swap del wrapper GGML que nuestro
            // mma.sync asm crudo no hace (misma lección que el A-tile del
            // score-path). Staging smem + ldmatrix: v_stage[m=dim][k=token]
            // row-major, 16B rows.
            #pragma unroll
            for (int e = lane; e < 16 * 16; e += 32) {
                const int m = e / 16;   // dim local 0..15
                const int k = e % 16;   // token local 0..15
                const int pos = group_pos_begin + chunk * TOKENS_PER_CHUNK + k;
                const int token = token_begin + chunk * TOKENS_PER_CHUNK + k;
                float x;
                if (v_split_in_group && pos < KVARN_DIM) {
                    // Dequant V via loader canónico (kvarn_record_value,
                    // mismo que portable): (q·s_row[pos]+zp[pos])·s_col[dim].
                    x = kvarn_record_value(
                        v_desc, record_group_v, pos, dim0 + m, 0);
                } else {
                    x = (token < token_end)
                        ? kvarn_decode_load_rotated(v_desc, token, 0, dim0 + m)
                        : 0.0f;
                }
                v_stage[warp][m][k] = __float2half(x);
            }
#ifdef KVARN_SPLIT_DEBUG
            if (blockIdx.x == 2 && blockIdx.y == 0 && warp == 0 &&
                    chunk == 0 && lane < 4) {
                printf("[split] Vst blk2 w0 c0 m%d k%d = %.4f\n",
                    lane, lane, __half2float(v_stage[0][lane][lane]));
            }
#endif
            __syncwarp();
            T_A v_a;
            kvarn_mma::load_ldmatrix_a(
                v_a, &v_stage[warp][0][0], 16);
            T_B p_b;
            // P B-tile: k=tokens del chunk (16), n=heads (8). [token][8].
            kvarn_mma::load_ldmatrix_b(
                p_b, (__half*)p_sh + chunk * TOKENS_PER_CHUNK * 8, 8);
            kvarn_mma::mma(out, v_a, p_b);
        }

        // Descarga partial: lane/reg → (dim, head).
        #pragma unroll
        for (int l = 0; l < T_C::ne; ++l) {
            const int i = l / 2 * 8 + lane / 4;   // dim local
            const int h = 2 * (lane % 4) + (l % 2); // head
            const int q_head = q_head0 + h;
            if (h < gqa_head_count && q_head < n_q_heads) {
                const size_t base = (((size_t)stream * n_q + q_index)
                    * n_q_heads + q_head) * n_splits + split;
                partial[base * D + dim0 + i] = out.x[l];
            }
        }
        __syncwarp();
    }
    __syncthreads();

    if (tid < gqa_head_count && q_head0 + tid < n_q_heads) {
        const int q_head = q_head0 + tid;
        const size_t base = (((size_t)stream * n_q + q_index)
            * n_q_heads + q_head) * n_splits + split;
        partial_meta[base] = make_float2(m_sh[tid], denom_sh[tid]);
    }
}

// ---------------------------------------------------------------------------
// Combine: reduce los partials (m, denom, out) a la salida final.
// ---------------------------------------------------------------------------

template <int D>
__device__ void kvarn_decode_combine_kernel(
    const float* partial,
    const float2* partial_meta,
    float* dst,
    int n_splits,
    int n_q,
    int n_q_heads)
{
    const int q_head = blockIdx.x;
    const int q_index = blockIdx.y;
    const int stream = blockIdx.z;
    const int tid = (int)threadIdx.x;

    __shared__ float reduce_sh[KVARN_DECODE_THREADS];
    extern __shared__ float split_weights[];

    const size_t row = ((size_t)stream * n_q + q_index) * n_q_heads + q_head;

    float local_max = -FLT_MAX / 2.0f;
    for (int split = tid; split < n_splits; split += blockDim.x) {
        const float2 meta = partial_meta[row * n_splits + split];
        if (meta.y > 0.0f) local_max = fmaxf(local_max, meta.x);
    }
    reduce_sh[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_sh[tid] = fmaxf(reduce_sh[tid], reduce_sh[tid + stride]);
        }
        __syncthreads();
    }
    const float m = reduce_sh[0];

    float local_denom = 0.0f;
    for (int split = tid; split < n_splits; split += blockDim.x) {
        const float2 meta = partial_meta[row * n_splits + split];
        float weight = 0.0f;
        if (meta.y > 0.0f) {
            weight = __expf(meta.x - m);
            local_denom += weight * meta.y;
        }
        split_weights[split] = weight;
    }
    reduce_sh[tid] = local_denom;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_sh[tid] += reduce_sh[tid + stride];
        }
        __syncthreads();
    }
    const float denom = reduce_sh[0];

    if (tid == 0 && denom > 0.0f) {
        // (dst_meta se omite en esta variante mínima)
    }

    // Weighted-sum por dim.
    __shared__ float out_s[D];
    for (int dim = tid; dim < D; dim += blockDim.x) {
        float out = 0.0f;
        if (denom > 0.0f) {
            for (int split = 0; split < n_splits; ++split) {
                out += split_weights[split]
                    * partial[row * n_splits * D + (size_t)split * D + dim];
            }
            out /= denom;
        }
        out_s[dim] = out;
    }
    __syncthreads();
    // De-rotación (D3): el pipeline entero corrió en dominio ROTADO;
    // el output del softmax es invariante bajo la ortogonal WHT aplicada
    // a K y V ⇒ el output ROTADO de-rota con WHT⁻¹ (involutiva).
    kvarn_wht_128_shared(out_s, tid);
    __syncthreads();
    for (int dim = tid; dim < D; dim += blockDim.x) {
        dst[row * D + dim] = out_s[dim];
    }
}

// ---- Kernels wrapper extern "C" (NO-template): cargables desde cubin
// por nombre vía cuModuleGetFunction (Driver API, patrón zig-ai). ----

extern "C" __global__ void kvarn_decode_mma_d128_gqa6_s64_w4_k4v4_kernel(
    const float* Q, const KvarnDesc* k_descs, const KvarnDesc* v_descs,
    const __half* mask, float* partial, float2* partial_meta,
    float scale, int n_kv, int n_q, int n_q_heads, int n_kv_heads,
    int gqa_ratio, int n_gqa_blocks, int n_splits)
{
#ifdef KVARN_SPLIT_DEBUG
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        printf("[split] blk=%d n_kv=%d nq=%d gqa=%d nspl=%d\n",
            (int)blockIdx.x, n_kv, n_q, gqa_ratio, n_splits);
        const KvarnDesc& kd = k_descs[0];
        printf("[split] blk=%d kdesc: gps=%d rb=%d sg=%d tg=%d live=%d/%d eager=%d bits=%d nheads=%d\n",
            (int)blockIdx.x, kd.groups_per_stream, kd.record_bytes, kd.stage_groups, kd.tail_groups,
            kd.live_group, kd.live_pos, kd.eager_records, kd.bits, kd.n_record_heads);
    }
#endif
    kvarn_decode_mma_kernel<128, 6, 64, 4, 4, 4>(
        Q, k_descs, v_descs, mask, partial, partial_meta, scale,
        n_kv, n_q, n_q_heads, n_kv_heads, gqa_ratio, n_gqa_blocks,
        n_splits);
}

extern "C" __global__ void kvarn_decode_combine_d128_kernel(
    const float* partial, const float2* partial_meta, float* dst,
    int n_splits, int n_q, int n_q_heads)
{
    kvarn_decode_combine_kernel<128>(
        partial, partial_meta, dst, n_splits, n_q, n_q_heads);
}
