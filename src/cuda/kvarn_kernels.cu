//! Kernels GPU KVarN — store (encode) + materialize (decode) sobre records C1.
//!
//! Lane-b1 (Dev A). Transcripción de los algoritmos upstream al layout C1
//! (record fusionado K+V, sectores 32B).
//!
//! Estructura:
//!   - A1: WHT-128 device (butterfly normalizado, involutivo)
//!   - B3 (Dev-B): init_descs (live_group/live_pos + fill de descs K/V)
//!   - A2-A4: store monolítico hi-shmem (WHT → stage f16 → seal C1)
//!     con Sinkhorn-as-B2 determinista (A3) y pack LSB-first a offsets C1.

#include "kvarn_desc.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math_constants.h>
#include <cfloat>

// ============================================================================
// A1: WHT-128 normalizada (butterfly Cooley-Tukey, in-place en shared).
// ============================================================================

__device__ __forceinline__ void kvarn_wht_128_impl(float* values, int tid)
{
    const int nthreads = blockDim.x > 0 ? blockDim.x : 128;
    for (int stride = 1; stride < KVARN_DIM; stride <<= 1) {
        const int half = stride;
        for (int pair = tid; pair < KVARN_DIM / 2; pair += nthreads) {
            const int j = (pair / half) * (2 * half) + (pair % half);
            const float a = values[j];
            const float b = values[j + half];
            values[j] = a + b;
            values[j + half] = a - b;
        }
        __syncthreads();
    }
    if (tid < KVARN_DIM) {
        values[tid] *= 0.08838834764831845f; // 1/sqrt(128)
    }
}

extern "C" __global__ void kvarn_wht_128_kernel(float* values)
{
    kvarn_wht_128_impl(values, (int)threadIdx.x);
}

extern "C" __global__ void kvarn_wht_128_rows_kernel(
    float* rows, int n_rows)
{
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    const int tid = (int)threadIdx.x;
    for (int d = tid; d < KVARN_DIM; d += blockDim.x) {
        smem[d] = rows[(size_t)row * KVARN_DIM + d];
    }
    __syncthreads();
    kvarn_wht_128_impl(smem, tid);
    __syncthreads();
    for (int d = tid; d < KVARN_DIM; d += blockDim.x) {
        rows[(size_t)row * KVARN_DIM + d] = smem[d];
    }
}

// ============================================================================
// A1b: WHT-64 normalizada (butterfly Cooley-Tukey, in-place en shared).
//       9.12 (lane-cuda) F2 GPU: espejo CPU hadamard64InPlace (kvarn.zig:349).
//       N=64, 6 etapas butterfly (1→2→4→8→16→32→64), 1/√64 = 0.125.
//       128 threads cooperan: stride llega a 32 < 128, paralelización por
//       tid natural (patrón idéntico a kvarn_wht_128_impl con KVARN_DIM→64).
// ============================================================================

static const int KVARN_D64 = 64;

__device__ __forceinline__ void kvarn_wht_64_impl(float* values, int tid)
{
    const int nthreads = blockDim.x > 0 ? blockDim.x : 128;
    for (int stride = 1; stride < KVARN_D64; stride <<= 1) {
        const int half = stride;
        for (int pair = tid; pair < KVARN_D64 / 2; pair += nthreads) {
            const int j = (pair / half) * (2 * half) + (pair % half);
            const float a = values[j];
            const float b = values[j + half];
            values[j] = a + b;
            values[j + half] = a - b;
        }
        __syncthreads();
    }
    if (tid < KVARN_D64) {
        values[tid] *= 0.125f; // 1/sqrt(64)
    }
}

extern "C" __global__ void kvarn_wht_64_kernel(float* values)
{
    kvarn_wht_64_impl(values, (int)threadIdx.x);
}

extern "C" __global__ void kvarn_wht_64_rows_kernel(
    float* rows, int n_rows)
{
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    const int tid = (int)threadIdx.x;
    for (int d = tid; d < KVARN_D64; d += blockDim.x) {
        smem[d] = rows[(size_t)row * KVARN_D64 + d];
    }
    __syncthreads();
    kvarn_wht_64_impl(smem, tid);
    __syncthreads();
    for (int d = tid; d < KVARN_D64; d += blockDim.x) {
        rows[(size_t)row * KVARN_D64 + d] = smem[d];
    }
}

// ============================================================================
// B3 (Dev-B): kvarnInitDescsKernel
// ============================================================================
//
// Lane-b1 Dev-B (B3, TODO_B1_DEV_B.md §B3). Grid (n_stream), block 128. Por
// cada stream: calcula live_group/live_pos con un árbol de reducción sobre
// los `indices` (i64) y rellena un KvarnDesc para value=0/1 (K/V).
//
// `live_group/live_pos` = max índice válido de los tokens del stream
// (índices >= 0 ó staged; -1 = skip). El valor se descompone en
//   group_global = idx / 128, pos = idx % 128
// y se reporta el máximo lexicográfico (group, pos) — el grupo "vivo" es
// el más alto con tokens válidos, y dentro de él, la posición más alta.
//
// El kernel se invoca DOS veces desde el wrapper Zig (una por side, K y V)
// porque cada side tiene su propio `bits`. Los demás campos (records,
// stage, indices, layout) son los mismos del fused record C1 — son punteros
// compartidos del módulo B2.
//
// Block 128 = 1 thread por dim WHT, lo que nos da una reducción
// `__shfl_xor` natural de 128→1 sin smem dinámico (4 warps × 32 lanes
// hacen la reducción por warp; una pasada final combina 4 partials).
//
// Requisitos de launch (validados en el wrapper):
//   - blockDim.x == 128
//   - n_stream > 0, n_stream <= 65535
//   - indices no nulo, n_indices >= tokensPerStream() del stream
//   - desc_out no nulo, n_desc >= n_stream
//
// No hay `__syncthreads` (no usamos smem); las reducciones son puramente
// por warp vía `__shfl_xor_sync` (4 warps) y un final a través de smem
// estática de 4 floats (la unión de los 4 partials del warp). La
// coherencia del `desc_out` no es problema: el único escritor es el lane 0
// del último warp reducido, y la escritura a `desc_out[s]` no compite con
// otros streams (cada stream es un bloque distinto).

template<int NWARPS>
__device__ __forceinline__ void kvarn_reduce_max_group_pos(
    const int64_t* indices, int n_indices, int s, int* out_group, int* out_pos)
{
    // Cada thread del bloque recorre una porción de los índices del stream.
    // Inicializa local_max_group=−1, local_max_pos=−1.
    int local_group = -1;
    int local_pos = -1;
    for (int i = threadIdx.x; i < n_indices; i += blockDim.x) {
        const int64_t enc = indices[(size_t)s * n_indices + i];
        if (enc == -1) continue; // skip
        // Decodifica: el "idx" efectivo es el campo cell del payload (los
        // staged se mapean a su cell; el grouping se hace siempre por el
        // valor absoluto de cell, no por slot).
        const int64_t cell = (int64_t)(uint32_t)kvarn_index_payload(enc);
        if (cell < 0) continue;
        const int g = (int)(cell / KVARN_DIM);
        const int p = (int)(cell - (int64_t)g * KVARN_DIM);
        // Max lexicográfico (group, pos): g>p group siempre gana; si
        // mismo group, gana la pos mayor.
        if (g > local_group || (g == local_group && p > local_pos)) {
            local_group = g;
            local_pos = p;
        }
    }

    // Reducción intra-warp (__shfl_xor) — patrón butterfly 32→1.
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1) {
        const int g_other = __shfl_xor_sync(0xFFFFFFFFu, local_group, off, 32);
        const int p_other = __shfl_xor_sync(0xFFFFFFFFu, local_pos, off, 32);
        // Max lexicográfico: mayor group gana; empate en group ⇒ mayor pos.
        if (g_other > local_group || (g_other == local_group && p_other > local_pos)) {
            local_group = g_other;
            local_pos = p_other;
        }
    }

    // Cross-warp: el primer lane de cada warp escribe a smem[4]; el
    // primer warp combina los 4 partials y escribe el resultado.
    __shared__ int s_g[NWARPS];
    __shared__ int s_p[NWARPS];
    if (lane == 0) {
        s_g[warp] = local_group;
        s_p[warp] = local_pos;
    }
    __syncthreads();
    if (warp == 0 && lane < NWARPS) {
        int g = s_g[lane];
        int p = s_p[lane];
        for (int off = NWARPS / 2; off > 0; off >>= 1) {
            const int g_o = __shfl_xor_sync(0xFFFFFFFFu, g, off, 32);
            const int p_o = __shfl_xor_sync(0xFFFFFFFFu, p, off, 32);
            if (g_o > g || (g_o == g && p_o > p)) {
                g = g_o;
                p = p_o;
            }
        }
        if (lane == 0) {
            *out_group = g;
            *out_pos = p;
        }
    }
}

extern "C" __global__ void kvarn_init_descs_kernel(
    int n_stream, int n_indices,
    const int64_t* indices,
    KvarnDesc* descs_out, int desc_stride,
    uint8_t* records, __half* stage,
    int n_record_heads, int groups_per_stream, int record_bytes,
    int stage_groups, int tail_groups,
    int k_bits, int v_bits,
    int head_slices, int eager_records, int read_indirect,
    int original_domain, int swa)
{
    const int s = blockIdx.x;
    if (s >= n_stream) return;
    if (threadIdx.x >= 128) return; // block must be 128

    // 1) Reducción: live_group / live_pos para este stream.
    // FIX A3-bis v2 (M3): el reduce publica *out_group/pos SOLO desde
    // (warp0, lane0) — los demás threads NO ven esos registros. El
    // write de descs corre EXCLUSIVAMENTE en threadIdx.x==0 con loop
    // sobre kv_heads (serial pero trivial: ≤8 heads × 2 descs): usa
    // los registros del MISMO thread que el reduce publicó ⇒
    // coherencia garantizada sin barriers ni smem extra.
    int live_group = -1;
    int live_pos = -1;
    kvarn_reduce_max_group_pos<4>(
        indices, n_indices, s, &live_group, &live_pos);

    // 2) Los descriptores: layout POR-LADO — [K(h0..hn), V(h0..hn)] por
    //    stream, contiguos por lado (el portable indexa
    //    k_descs[stream·n_kv_heads + kv_head] y v_descs idem con la base
    //    V ya avanzada por el caller). 9.4 (lane-b): el layout anterior
    //    intercalado [K0,V0,K1,V1] con desc_stride=2 hacia el portable
    //    leer K1 donde esperaba V0 (n_kv_heads>1: Qwen3.5 8Q/2KV leía el
    //    desc equivocado ⇒ degradación desde el 1er decode). Con 1
    //    kv_head ambos layouts coinciden ⇒ tests d256/m1seed invariante.
    //    head_base = base de heads FÍSICAS de cada kv_head LÓGICA:
    //    D≥256 despliega head_slices físicas por lógica (beellama
    //    head0 = blockIdx.x·slices; D=128: head_slices=1 ⇒ invariante).
    const int n_kv_heads_logical = n_record_heads / (head_slices > 0 ? head_slices : 1);
    if (threadIdx.x == 0) {
        for (int kv_head = 0; kv_head < n_kv_heads_logical; ++kv_head) {
        // ── K (value = 0) ── bloque K: offset kv_head ──
        const int off_k = s * desc_stride * n_kv_heads_logical
            + kv_head;
        KvarnDesc dk;
        dk.records         = records;
        dk.stage           = stage;
        dk.indices         = indices;
        dk.n_record_heads  = n_record_heads;
        dk.live_group      = live_group;
        dk.live_pos        = live_pos;
        dk.stream          = s;
        // 9.4 (lane-b) D2: head_base es la base de las heads FÍSICAS de
        // esta kv_head — D≥256 despliega `head_slices` físicas por
        // lógica (beellama head0 = blockIdx.x·slices). D=128:
        // head_slices=1 ⇒ head_base = kv_head (invariante).
        dk.head_base       = kv_head * head_slices;
        dk.groups_per_stream = groups_per_stream;
        dk.record_bytes    = record_bytes;
        dk.stage_groups    = stage_groups;
        dk.tail_groups     = tail_groups;
        dk.bits            = k_bits;
        dk.value           = 0;
        dk.swa             = swa;
        dk.head_slices     = head_slices;
        dk.eager_records   = eager_records;
        dk.read_indirect   = read_indirect;
        dk.original_domain = original_domain;
        descs_out[off_k] = dk;

        // ── V (value = 1) ── bloque V tras TODAS las K del stream ──
        const int off_v = s * desc_stride * n_kv_heads_logical
            + n_kv_heads_logical + kv_head;
        KvarnDesc dv;
        dv = dk;
        dv.bits = v_bits;
        dv.value = 1;
        descs_out[off_v] = dv;
        } // fin loop kv_head
    }
}

// Lanzador host Zig-equivalente: llena un array [n_stream × 2] de KvarnDesc.
// Esta declaración `extern "C"` se invoca desde el wrapper Zig. No se usa
// desde .cu (los tests lo hacen vía el wrapper).
extern "C" void kvarn_init_descs_launcher(
    int n_stream, int n_indices,
    const int64_t* d_indices,
    KvarnDesc* d_descs, int desc_stride,
    uint8_t* d_records, __half* d_stage,
    int n_record_heads, int groups_per_stream, int record_bytes,
    int stage_groups, int tail_groups,
    int k_bits, int v_bits,
    int head_slices, int eager_records, int read_indirect,
    int original_domain, int swa,
    cudaStream_t stream)
{
    // Validación mínima (lanzamientos inválidos ⇒ no-op silencioso).
    if (n_stream <= 0 || n_indices <= 0) return;
    kvarn_init_descs_kernel<<<n_stream, 128, 0, stream>>>(
        n_stream, n_indices,
        d_indices,
        d_descs, desc_stride,
        d_records, d_stage,
        n_record_heads, groups_per_stream, record_bytes,
        stage_groups, tail_groups,
        k_bits, v_bits,
        head_slices, eager_records, read_indirect,
        original_domain, swa);
}

// ============================================================================
// A2-A4 (Dev-A): store monolítico hi-shmem — encode C1 bit-exacto vs B2
// ============================================================================
//
// Un bloque (128 threads) por cabeza; loop serial sobre los tokens del
// stream. Por token: decode índice → (group, pos, slot); WHT-128 del vector
// entrante; escritura al stage. Seal cuando la ventana lo exige:
//   - delayed flush (non-eager): al ENTRAR en un grupo nuevo (pos==0),
//     sellar el record del grupo que sale de la ventana tail.
//   - eager seal: al COMPLETAR un grupo (pos==127), quantizar inmediato
//     (non-SWA solo group>0; el grupo 0 sink queda al delayed-flush).
//
// Seal = 2× Sinkhorn-as-B2 (A3: transcendentales deterministas — MISMA
// secuencia IEEE que el CPU reference) + quantize f32 por filas + pack
// LSB-first a los offsets C1 del record fusionado K+V.
//
// K usa Sinkhorn DIM-major (como encodeKTile de B2), V TOKEN-major (como
// encodeVTile): las orientaciones DIFFIEREN por lado.
//
// Shared dinámica (69,704B): tile[16384] | log_s_col[128] | log_s_row[128]
// | s_col[128] | s_row[128] | col_std[128] | row_std[128]
// | best_imbalance[1] | better[1] | reduce[16]

// ---- Transcendentales deterministas (mirror EXACTO de kvarn.zig B2) ----

__device__ __forceinline__ double kvarn_log_f64(double x)
{
    const double LN2 = 0.693147180559945309417232121458176568;
    const double SQRT_HALF_HI = 0.70710678118654752440;
    int k = 0;
    double m = x;
    if (m >= 1.0) {
        while (m >= 1.0) { m *= 0.5; k += 1; }
    } else {
        while (m < 0.5) { m *= 2.0; k -= 1; }
    }
    if (m < SQRT_HALF_HI) { m *= 2.0; k -= 1; }
    const double t = (m - 1.0) / (m + 1.0);
    const double t2 = t * t;
    const double atanh = t * (1.0
        + t2 * (1.0 / 3.0
            + t2 * (1.0 / 5.0
                + t2 * (1.0 / 7.0
                    + t2 * (1.0 / 9.0
                        + t2 * (1.0 / 11.0
                            + t2 * (1.0 / 13.0
                                + t2 * (1.0 / 15.0
                                    + t2 * (1.0 / 17.0
                                        + t2 * (1.0 / 19.0
                                            + t2 * (1.0 / 21.0
                                                + t2 * (1.0 / 23.0
                                                    + t2 * (1.0 / 25.0)))))))))))));
    const double kf = (double)k;
    return 2.0 * atanh + kf * LN2;
}

__device__ __forceinline__ double kvarn_exp_f64(double x)
{
    const double LN2 = 0.693147180559945309417232121458176568;
    const double kf = round(x / LN2);
    const int k = (int)kf;
    const double r = x - kf * LN2;
    const double r2 = r * r;
    const double r3 = r2 * r;
    const double r4 = r2 * r2;
    const double r5 = r3 * r2;
    const double r6 = r3 * r3;
    const double r7 = r4 * r3;
    const double r8 = r4 * r4;
    const double r9 = r5 * r4;
    const double r10 = r5 * r5;
    const double r11 = r6 * r5;
    const double r12 = r6 * r6;
    const double r13 = r7 * r6;
    const double poly = 1.0
        + r
        + r2 * (1.0 / 2.0)
        + r3 * (1.0 / 6.0)
        + r4 * (1.0 / 24.0)
        + r5 * (1.0 / 120.0)
        + r6 * (1.0 / 720.0)
        + r7 * (1.0 / 5040.0)
        + r8 * (1.0 / 40320.0)
        + r9 * (1.0 / 362880.0)
        + r10 * (1.0 / 3628800.0)
        + r11 * (1.0 / 39916800.0)
        + r12 * (1.0 / 479001600.0)
        + r13 * (1.0 / 6227020800.0);
    double scale = 1.0;
    if (k > 0) {
        for (int i = 0; i < k; ++i) scale *= 2.0;
    } else if (k < 0) {
        for (int i = 0; i > k; --i) scale *= 0.5;
    }
    return poly * scale;
}

// ---- sampleStd f64 (mirror B2: den N−1, max(0,·)) sobre 128 f32 + stride ----

__device__ __forceinline__ double kvarn_std_128(
    const float* tile, int base, int stride)
{
    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < 128; ++i) {
        const double v = (double)tile[base + i * stride];
        sum += v;
        sum_sq += v * v;
    }
    const double nf = 128.0;
    const double mean = sum / nf;
    const double variance = fmax(0.0, (sum_sq - nf * mean * mean) / (nf - 1.0));
    return sqrt(variance);
}

// Orientaciones del Sinkhorn sobre el tile TOKEN-major del stage:
//   transpose=1 (K): row=dim (tile[t*128+i]), col=token (tile[j*128+d]).
//   transpose=0 (V): row=token (tile[i*128+c]), col=dim (tile[t*128+j]).
__device__ __forceinline__ double kvarn_std_row(
    const float* tile, int transpose, int i)
{
    return transpose ? kvarn_std_128(tile, i, 128) : kvarn_std_128(tile, i * 128, 1);
}

__device__ __forceinline__ double kvarn_std_col(
    const float* tile, int transpose, int j)
{
    return transpose ? kvarn_std_128(tile, j * 128, 1) : kvarn_std_128(tile, j, 128);
}

// std con rebalance f32 on-the-fly (mirror rebuildCur f32 + sampleStd
// f64-sobre-f32 — sin el cast f32 intermedio el std diverge del CPU).
__device__ __forceinline__ double kvarn_balanced_std_row(
    const float* tile, int transpose, int i,
    const float* log_s_col, const float* log_s_row)
{
    double sum = 0.0, sum_sq = 0.0;
    for (int j = 0; j < 128; ++j) {
        const int idx = transpose ? (j * 128 + i) : (i * 128 + j);
        const double sc = kvarn_exp_f64((double)log_s_col[j]);
        const double sr = kvarn_exp_f64((double)log_s_row[i]);
        const float v32 = (float)((double)tile[idx] / (sc * sr));
        const double v = (double)v32;
        sum += v;
        sum_sq += v * v;
    }
    const double nf = 128.0;
    const double mean = sum / nf;
    const double variance = fmax(0.0, (sum_sq - nf * mean * mean) / (nf - 1.0));
    return sqrt(variance);
}

__device__ __forceinline__ double kvarn_balanced_std_col(
    const float* tile, int transpose, int j,
    const float* log_s_col, const float* log_s_row)
{
    // tr=1 (col j = token): elementos tile[j*128+i] con i=dim variable.
    // tr=0 (col j = dim):   elementos tile[i*128+j] con i=token variable.
    // (Antes tenía los índices cruzados en tr — el std no veía el rebalance
    // del row-pass y el Sinkhorn quedaba congelado.)
    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < 128; ++i) {
        const int idx = transpose ? (j * 128 + i) : (i * 128 + j);
        const double sc = kvarn_exp_f64((double)log_s_col[j]);
        const double sr = kvarn_exp_f64((double)log_s_row[i]);
        const float v32 = (float)((double)tile[idx] / (sc * sr));
        const double v = (double)v32;
        sum += v;
        sum_sq += v * v;
    }
    const double nf = 128.0;
    const double mean = sum / nf;
    const double variance = fmax(0.0, (sum_sq - nf * mean * mean) / (nf - 1.0));
    return sqrt(variance);
}

// Min/max de stds (shfl warp + fold) — mirror del imbalance B2.
__device__ __forceinline__ void kvarn_std_ranges(
    const float* col_std, const float* row_std,
    float* reduce,
    float* col_min_out, float* col_max_out,
    float* row_min_out, float* row_max_out)
{
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    float cmin = 3.4e38f, cmax = 0.0f;
    float rmin = 3.4e38f, rmax = 0.0f;
    if (tid < 128) {
        cmin = col_std[tid]; cmax = col_std[tid];
        rmin = row_std[tid]; rmax = row_std[tid];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        cmin = fminf(cmin, __shfl_down_sync(0xFFFFFFFFu, cmin, off));
        cmax = fmaxf(cmax, __shfl_down_sync(0xFFFFFFFFu, cmax, off));
        rmin = fminf(rmin, __shfl_down_sync(0xFFFFFFFFu, rmin, off));
        rmax = fmaxf(rmax, __shfl_down_sync(0xFFFFFFFFu, rmax, off));
    }
    if (lane == 0 && warp < 4) {
        reduce[warp * 4 + 0] = cmin;
        reduce[warp * 4 + 1] = cmax;
        reduce[warp * 4 + 2] = rmin;
        reduce[warp * 4 + 3] = rmax;
    }
    __syncthreads();
    if (tid < 4) {
        float cm = reduce[0], cx = reduce[1], rm = reduce[2], rx = reduce[3];
        for (int w = 1; w < 4; ++w) {
            cm = fminf(cm, reduce[w * 4 + 0]);
            cx = fmaxf(cx, reduce[w * 4 + 1]);
            rm = fminf(rm, reduce[w * 4 + 2]);
            rx = fmaxf(rx, reduce[w * 4 + 3]);
        }
        reduce[0] = cm; reduce[1] = cx; reduce[2] = rm; reduce[3] = rx;
    }
    __syncthreads();
    *col_min_out = reduce[0];
    *col_max_out = reduce[1];
    *row_min_out = reduce[2];
    *row_max_out = reduce[3];
}

// ---- Sinkhorn un lado (mirror op a op de varianceNormalize B2) ----

__device__ void kvarn_sinkhorn_one_side(
    const float* tile,
    int transpose,
    float* log_s_col, float* log_s_row,
    float* s_col, float* s_row,
    float* col_std, float* row_std,
    float* best_imbalance, float* better, float* reduce,
    int sinkhorn_iters)
{
    const int tid = (int)threadIdx.x;

    if (tid < 128) {
        log_s_col[tid] = 0.0f;
        log_s_row[tid] = 0.0f;
        s_col[tid] = 1.0f;
        s_row[tid] = 1.0f;
    }
    __syncthreads();
    if (tid == 0) *best_imbalance = 3.4e38f;
    __syncthreads();

    // eval0: stds del tile crudo (balanced == tile con s=1).
    if (tid < 128) {
        row_std[tid] = (float)kvarn_std_row(tile, transpose, tid);
        col_std[tid] = (float)kvarn_std_col(tile, transpose, tid);
    }
    __syncthreads();
    {
        float cmin, cmax, rmin, rmax;
        kvarn_std_ranges(col_std, row_std, reduce, &cmin, &cmax, &rmin, &rmax);
        const float cm = cmin < 1e-8f ? 1e-8f : cmin;
        const float rm = rmin < 1e-8f ? 1e-8f : rmin;
        if (tid == 0) *best_imbalance = cmax / cm + rmax / rm;
        __syncthreads();
    }

    for (int iter = 0; iter < sinkhorn_iters; ++iter) {
        // Column pass (clamp std → log-accum f64 → clamp f64 → cast f32).
        if (tid < 128) {
            float std_c = col_std[tid];
            if (std_c < 1e-3f) std_c = 1e-3f;
            if (std_c > 1e3f) std_c = 1e3f;
            const double new_log = (double)log_s_col[tid] + kvarn_log_f64((double)std_c);
            double clamped = new_log;
            if (clamped < -0.3) clamped = -0.3;
            if (clamped > 10.0) clamped = 10.0;
            log_s_col[tid] = (float)clamped;
        }
        __syncthreads();
        if (tid < 128) {
            row_std[tid] = (float)kvarn_balanced_std_row(tile, transpose, tid, log_s_col, log_s_row);
            col_std[tid] = (float)kvarn_balanced_std_col(tile, transpose, tid, log_s_col, log_s_row);
        }
        __syncthreads();

        // Row pass.
        if (tid < 128) {
            float std_r = row_std[tid];
            if (std_r < 1e-3f) std_r = 1e-3f;
            if (std_r > 1e3f) std_r = 1e3f;
            const double new_log = (double)log_s_row[tid] + kvarn_log_f64((double)std_r);
            double clamped = new_log;
            if (clamped < -0.3) clamped = -0.3;
            if (clamped > 10.0) clamped = 10.0;
            log_s_row[tid] = (float)clamped;
        }
        __syncthreads();
        if (tid < 128) {
            row_std[tid] = (float)kvarn_balanced_std_row(tile, transpose, tid, log_s_col, log_s_row);
            col_std[tid] = (float)kvarn_balanced_std_col(tile, transpose, tid, log_s_col, log_s_row);
        }
        __syncthreads();

        // Imbalance + snapshot best-so-far (tie <=, como B2).
        float cmin, cmax, rmin, rmax;
        kvarn_std_ranges(col_std, row_std, reduce, &cmin, &cmax, &rmin, &rmax);
        const float cm = cmin < 1e-8f ? 1e-8f : cmin;
        const float rm = rmin < 1e-8f ? 1e-8f : rmin;
        if (tid == 0) {
            const float imb = cmax / cm + rmax / rm;
            *better = (imb <= *best_imbalance) ? 1.0f : 0.0f;
            if (*better > 0.0f) *best_imbalance = imb;
        }
        __syncthreads();
        __syncthreads();
        if (tid < 128 && *better > 0.0f) {
            s_col[tid] = (float)kvarn_exp_f64((double)log_s_col[tid]);
            s_row[tid] = (float)kvarn_exp_f64((double)log_s_row[tid]);
        }
        __syncthreads();
    }
}

// ---- Quantize + pack un lado (mirror encodeK/VTile B2, f32 puro) ----
// fila r; balanced[r][j] = tile[...] / (s_col[j]·s_row[r]) con división f64
// y cast f32 (mirror de rebuildCur); lo/hi/scale/round/clamp f32; pack
// LSB-first bit a bit (mirror packBit). Axes f16 con absorb=s_row[r].

__device__ __forceinline__ void kvarn_quantize_pack_side(
    const float* tile,
    int transpose,          // 1=K: fila=dim, j=token; 0=V: fila=token, j=dim
    const float* s_col, const float* s_row,
    uint8_t* record,
    int payload_off,
    int bits,
    int scale_axis_off,     // indexed by fila r
    int zp_axis_off,        // indexed by fila r
    int other_axis_off)     // indexed by j
{
    const int tid = (int)threadIdx.x;
    const int r = tid; // fila
    if (r >= 128) return;
    const float absorb = s_row[r];

    // Mirror EXACTO B2: balanced = tile / (s_col[j] * s_row[r]) con
    // producto y división en f32 PURO (el f64 solo vive en kvarnLog/Exp).
    float lo = 1e30f, hi = -1e30f;
    for (int j = 0; j < 128; ++j) {
        const int idx = transpose ? (j * 128 + r) : (r * 128 + j);
        const float v = tile[idx] / (s_col[j] * absorb);
        if (v < lo) lo = v;
        if (v > hi) hi = v;
    }
    const float qmax = (float)((1 << bits) - 1);
    const float range = hi - lo;
    const float scale = range > 0.0f ? range / qmax : 1e-10f;

    for (int j = 0; j < 128; ++j) {
        const int idx = transpose ? (j * 128 + r) : (r * 128 + j);
        const float v = tile[idx] / (s_col[j] * absorb);
        const float valf = roundf((v - lo) / scale);
        float cl = valf; if (cl < 0.0f) cl = 0.0f;
        if (cl > qmax) cl = qmax;
        const uint32_t q = (uint32_t)cl;
        const int bit_index = r * 128 + j;
        for (int b = 0; b < bits; ++b) {
            const int dst_bit = bit_index * bits + b;
            const uint32_t bit_val = (q >> b) & 1u;
            uint8_t* bytep = record + payload_off + (dst_bit >> 3);
            const uint8_t mask = (uint8_t)(1u << (dst_bit & 7));
            if (bit_val) *bytep |= mask; else *bytep &= (uint8_t)~mask;
        }
    }
    __half* scale_axis = (__half*)(record + scale_axis_off);
    __half* zp_axis = (__half*)(record + zp_axis_off);
    scale_axis[r] = __float2half(absorb * scale);
    zp_axis[r] = __float2half(absorb * lo);
}

// ---- Seal completo: 2× Sinkhorn (K dim-major, V token-major) + pack ----

__device__ void kvarn_seal_k_side(
    const float* tile,      // token-major K (stage fila 2h)
    float* log_s_col, float* log_s_row,
    float* s_col, float* s_row,
    float* col_std, float* row_std,
    float* best_imbalance, float* better, float* reduce,
    uint8_t* record,
    int k_bits,
    int k_payload_off, int k_s_col_off, int k_zp_off, int k_s_row_off,
    int sinkhorn_iters)
{
    const int tid = (int)threadIdx.x;
    // K: Sinkhorn DIM-major (transpose=1).
    kvarn_sinkhorn_one_side(tile, 1,
        log_s_col, log_s_row, s_col, s_row, col_std, row_std,
        best_imbalance, better, reduce, sinkhorn_iters);
    kvarn_quantize_pack_side(tile, 1, s_col, s_row, record,
        k_payload_off, k_bits, k_s_col_off, k_zp_off, k_s_row_off);
    __syncthreads();
    if (tid < 128) {
        // other axis K = s_col[j] con j=token.
        __half* ksr = (__half*)(record + k_s_row_off);
        ksr[tid] = __float2half(s_col[tid]);
    }
    __syncthreads();
}

__device__ void kvarn_seal_v_side(
    const float* tile_v,    // token-major V (stage fila 2h+1)
    float* log_s_col, float* log_s_row,
    float* s_col, float* s_row,
    float* col_std, float* row_std,
    float* best_imbalance, float* better, float* reduce,
    uint8_t* record,
    int v_bits,
    int v_payload_off, int v_s_col_off, int v_s_row_off, int v_zp_off,
    int sinkhorn_iters)
{
    const int tid = (int)threadIdx.x;
    // V: Sinkhorn TOKEN-major (transpose=0).
    kvarn_sinkhorn_one_side(tile_v, 0,
        log_s_col, log_s_row, s_col, s_row, col_std, row_std,
        best_imbalance, better, reduce, sinkhorn_iters);
    kvarn_quantize_pack_side(tile_v, 0, s_col, s_row, record,
        v_payload_off, v_bits, v_s_row_off, v_zp_off, v_s_col_off);
    __syncthreads();
    if (tid < 128) {
        // other axis V = s_col[j] con j=dim.
        __half* vsc = (__half*)(record + v_s_col_off);
        vsc[tid] = __float2half(s_col[tid]);
    }
    __syncthreads();
}

// ---- Helpers de slot ----

__device__ __forceinline__ int kvarn_stage_slot_for_group_std(
    int swa, int group, int stage_groups, int tail_groups)
{
    if (swa) return group % stage_groups;
    return group == 0 ? 0 : 1 + ((group - 1) % tail_groups);
}

__device__ __forceinline__ int kvarn_stage_slot_for_group_ext(
    int swa, int group, int stage_groups, int tail_groups, int assigned_slot)
{
    if (assigned_slot >= 0) return assigned_slot;
    return kvarn_stage_slot_for_group_std(swa, group, stage_groups, tail_groups);
}

// ---- Carga del tile desde el stage a shared (token-major) ----

__device__ __forceinline__ void kvarn_load_tile_from_stage(
    __half* stage, int n_record_heads, int head,
    int stream, int slot, int stage_groups,
    float* tile, int tid)
{
    const int stage_base = stream * KVARN_DIM * stage_groups + slot * KVARN_DIM;
    const int stage_heads = 2 * n_record_heads;
    for (int i = tid; i < KVARN_DIM * KVARN_DIM; i += blockDim.x) {
        const int tok = i / KVARN_DIM;
        const int dim = i % KVARN_DIM;
        // Layout C2v2: [stage_pos][2*heads][128]; `head` llega ya como
        // 2h (fila K) o 2h+1 (fila V) según el lado que carga.
        tile[i] = __half2float(stage[
            ((int64_t)(stage_base + tok) * stage_heads + head) * KVARN_DIM + dim]);
    }
}

// ---- A2: kernel store monolítico ----

extern "C" __global__ void kvarn_store_kernel(
    const float*   current,      // [n_tokens, n_record_heads, 128] K ORIGINALES
    const float*   current_v,    // idem V (nullptr ⇒ mismo buffer que K)
    const int64_t* indices,      // [n_tokens] celdas codificadas (este stream)
    __half*        stage,        // stage f16 (rotated) global
    uint8_t*       records,      // records C1 global
    int n_tokens,
    int n_record_heads,
    int stream,
    int groups_per_stream,
    int record_bytes,
    int k_payload_off, int k_s_col_off, int k_zp_off, int k_s_row_off,
    int v_payload_off, int v_s_col_off, int v_s_row_off, int v_zp_off,
    int k_bits, int v_bits,
    int sinkhorn_iters,
    int stage_groups,
    int tail_groups,
    int swa,
    int eager_records)
{
    extern __shared__ float smem[];
    float* tile = smem;                       // 16384 (K)
    float* log_s_col = tile + 16384;           // 128
    float* log_s_row = log_s_col + 128;        // 128
    float* s_col = log_s_row + 128;            // 128
    float* s_row = s_col + 128;                // 128
    float* col_std = s_row + 128;              // 128
    float* row_std = col_std + 128;            // 128
    float* best_imbalance = row_std + 128;     // 1
    float* better = best_imbalance + 1;        // 1
    float* reduce = better + 1;                // 16

    const int head = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    if (head >= n_record_heads) return;

    for (int t = 0; t < n_tokens; ++t) {
        const int64_t enc = indices[t];
        if (enc == -1) continue;
        bool explicitly_staged = false;
        int assigned_slot = -1;
        const int64_t cell = kvarn_read_cell(enc, explicitly_staged, &assigned_slot);
        const int group = (int)(cell / KVARN_DIM);
        const int pos = (int)(cell - (int64_t)group * KVARN_DIM);
        const int record_group = swa
            ? (group % groups_per_stream)
            : (stream * groups_per_stream + group);

        // ---- Delayed flush (non-eager): sellar el grupo que sale de la
        //      ventana tail al ENTRAR en un grupo nuevo (pos==0).
        if (!eager_records && pos == 0 &&
                (swa ? group >= tail_groups : group > tail_groups)) {
            const int flush_group = group - tail_groups;
            const int flush_record_group = swa
                ? (flush_group % groups_per_stream)
                : (stream * groups_per_stream + flush_group);
            const int flush_slot = kvarn_stage_slot_for_group_std(
                swa, flush_group, stage_groups, tail_groups);
            uint8_t* rec = records
                + ((int64_t)flush_record_group * n_record_heads + head) * record_bytes;
            // K: cargar fila 2h → seal K.
            kvarn_load_tile_from_stage(
                stage, n_record_heads, 2 * head, stream, flush_slot, stage_groups,
                tile, tid);
            __syncthreads();
            kvarn_seal_k_side(
                tile, log_s_col, log_s_row, s_col, s_row, col_std, row_std,
                best_imbalance, better, reduce,
                rec, k_bits,
                k_payload_off, k_s_col_off, k_zp_off, k_s_row_off,
                sinkhorn_iters);
            __syncthreads();
            // V: recargar el MISMO smem con la fila 2h+1 → seal V.
            kvarn_load_tile_from_stage(
                stage, n_record_heads, 2 * head + 1, stream, flush_slot, stage_groups,
                tile, tid);
            __syncthreads();
            kvarn_seal_v_side(
                tile, log_s_col, log_s_row, s_col, s_row, col_std, row_std,
                best_imbalance, better, reduce,
                rec, v_bits,
                v_payload_off, v_s_col_off, v_s_row_off, v_zp_off,
                sinkhorn_iters);
            __syncthreads();
        } // fin delayed flush

        // ---- Stage write: WHT-128 de K y V; filas K(2h)/V(2h+1) del
        //      stage [stage_pos][2*n_record_heads][128] (C2v2).
        {
            __shared__ float vec[128];
            __shared__ float vec_v[128];
            const float* cur_v = (current_v != nullptr) ? current_v : current;
            if (tid < 128) {
                vec[tid] = current[((int64_t)t * n_record_heads + head) * KVARN_DIM + tid];
                vec_v[tid] = cur_v[((int64_t)t * n_record_heads + head) * KVARN_DIM + tid];
            }
            __syncthreads();
            kvarn_wht_128_impl(vec, tid);
            kvarn_wht_128_impl(vec_v, tid);
            __syncthreads();
            const int slot = kvarn_stage_slot_for_group_ext(
                swa, group, stage_groups, tail_groups, assigned_slot);
            const int stage_pos = stream * KVARN_DIM * stage_groups
                + slot * KVARN_DIM + pos;
            const int stage_heads = 2 * n_record_heads;
            if (tid < 128) {
                stage[((int64_t)stage_pos * stage_heads + 2 * head) * KVARN_DIM + tid] =
                    __float2half(vec[tid]);
                stage[((int64_t)stage_pos * stage_heads + 2 * head + 1) * KVARN_DIM + tid] =
                    __float2half(vec_v[tid]);
            }
            __syncthreads();
        }

        // ---- Eager seal: al COMPLETAR un grupo (pos==127), sellar YA.
        if (eager_records && pos == KVARN_DIM - 1
                && (swa || group > 0)) {
            const int slot = kvarn_stage_slot_for_group_std(
                swa, group, stage_groups, tail_groups);
            uint8_t* rec = records
                + ((int64_t)record_group * n_record_heads + head) * record_bytes;
            kvarn_load_tile_from_stage(
                stage, n_record_heads, 2 * head, stream, slot, stage_groups,
                tile, tid);
            __syncthreads();
            kvarn_seal_k_side(
                tile, log_s_col, log_s_row, s_col, s_row, col_std, row_std,
                best_imbalance, better, reduce,
                rec, k_bits,
                k_payload_off, k_s_col_off, k_zp_off, k_s_row_off,
                sinkhorn_iters);
            __syncthreads();
            kvarn_load_tile_from_stage(
                stage, n_record_heads, 2 * head + 1, stream, slot, stage_groups,
                tile, tid);
            __syncthreads();
            kvarn_seal_v_side(
                tile, log_s_col, log_s_row, s_col, s_row, col_std, row_std,
                best_imbalance, better, reduce,
                rec, v_bits,
                v_payload_off, v_s_col_off, v_s_row_off, v_zp_off,
                sinkhorn_iters);
            __syncthreads();
        }
    }
}


// ============================================================================
// A5 (Dev-A): store LOW-SHMEM — sellado re-leyendo el stage half global.
//   smem = 788 floats estáticos (3,152 B < 48KB default) ⇒ sin
//   cuFuncSetAttribute opt-in, coexiste con el split kernel, y habilita
//   sm_75 (64KB/SM: el hishmem de 69.7KB no cabe ahí).
//   ⚠ NO bit-exacto con hishmem: el Sinkhorn ve f16-rounding del stage
//   pre-escala (upstream idem — ruta smpbo < KVAR_N_SHARED_BYTES).
// ============================================================================

// ctx de lectura stage C2v2 [pos][2·heads][128]; fila 2h(K)/2h+1(V).
struct KvarnLowCtx {
    const __half* stage;
    int n_heads, head, side_v, stream, slot, stage_groups;
};

// tile lógico [r][j]: K(dim-major) fila=dim r, col=token j;
// V(token-major) fila=token r, col=dim j — MIRROR de la transposición
// que usa el hishmem (kvarn_seal_k_side transpose=1 / v_side 0).
__device__ __forceinline__ float kvarn_low_tile(
    const KvarnLowCtx* c, int r, int j)
{
    const int tok = c->side_v ? r : j;
    const int dim = c->side_v ? j : r;
    const int stage_base = c->stream * KVARN_DIM * c->stage_groups
                        + c->slot * KVARN_DIM;
    const int heads2 = 2 * c->n_heads;
    const int row = 2 * c->head + c->side_v;
    return __half2float(c->stage[
        ((int64_t)(stage_base + tok) * heads2 + row) * KVARN_DIM + dim]);
}

// std/balanced re-leyendo el stage (f32-cast del half, igual que
// upstream kvarn_std_{col,row}_lowshmem; el rebalance f64→f32 cast
// mirror de kvarn_balanced_std_*).
__device__ __forceinline__ double kvarn_low_std_row(
    const KvarnLowCtx* c, int i)
{
    double sum = 0.0, sum_sq = 0.0;
    for (int j = 0; j < KVARN_DIM; ++j) {
        const float v32 = kvarn_low_tile(c, i, j);
        sum += v32; sum_sq += (double)v32 * v32;
    }
    const double nf = (double)KVARN_DIM;
    const double mean = sum / nf;
    const double var = fmax(0.0, (sum_sq - nf * mean * mean) / (nf - 1.0));
    return sqrt(var);
}

__device__ __forceinline__ double kvarn_low_std_col(
    const KvarnLowCtx* c, int j)
{
    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < KVARN_DIM; ++i) {
        const float v32 = kvarn_low_tile(c, i, j);
        sum += v32; sum_sq += (double)v32 * v32;
    }
    const double nf = (double)KVARN_DIM;
    const double mean = sum / nf;
    const double var = fmax(0.0, (sum_sq - nf * mean * mean) / (nf - 1.0));
    return sqrt(var);
}

__device__ __forceinline__ double kvarn_low_balanced_std_row(
    const KvarnLowCtx* c, int i,
    const float* log_s_col, const float* log_s_row)
{
    double sum = 0.0, sum_sq = 0.0;
    for (int j = 0; j < KVARN_DIM; ++j) {
        const double sc = kvarn_exp_f64((double)log_s_col[j]);
        const double sr = kvarn_exp_f64((double)log_s_row[i]);
        const float v32 = (float)((double)kvarn_low_tile(c, i, j) / (sc * sr));
        sum += v32; sum_sq += (double)v32 * v32;
    }
    const double nf = (double)KVARN_DIM;
    const double mean = sum / nf;
    const double var = fmax(0.0, (sum_sq - nf * mean * mean) / (nf - 1.0));
    return sqrt(var);
}

__device__ __forceinline__ double kvarn_low_balanced_std_col(
    const KvarnLowCtx* c, int j,
    const float* log_s_col, const float* log_s_row)
{
    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < KVARN_DIM; ++i) {
        const double sc = kvarn_exp_f64((double)log_s_col[j]);
        const double sr = kvarn_exp_f64((double)log_s_row[i]);
        const float v32 = (float)((double)kvarn_low_tile(c, i, j) / (sc * sr));
        sum += v32; sum_sq += (double)v32 * v32;
    }
    const double nf = (double)KVARN_DIM;
    const double mean = sum / nf;
    const double var = fmax(0.0, (sum_sq - nf * mean * mean) / (nf - 1.0));
    return sqrt(var);
}

// Sinkhorn one-side sobre stage (mismo schedule que kvarn_sinkhorn_one_side
// pero tile=kvarn_low_tile): log init 0, eval0 crudo, passes col→row con
// clamp [1e-3,1e3] y log clamp [-0.3,10], imbalance tie-<=, snapshot best.
__device__ void kvarn_low_sinkhorn(
    const KvarnLowCtx* c,
    float* log_s_col, float* log_s_row,
    float* s_col, float* s_row,
    float* col_std, float* row_std,
    float* best_imbalance, float* better, float* reduce,
    int sinkhorn_iters)
{
    const int tid = (int)threadIdx.x;
    if (tid < KVARN_DIM) {
        log_s_col[tid] = 0.0f;
        log_s_row[tid] = 0.0f;
        s_col[tid] = 1.0f;
        s_row[tid] = 1.0f;
    }
    __syncthreads();
    if (tid == 0) *best_imbalance = 3.4e38f;
    __syncthreads();
    if (tid < KVARN_DIM) {
        row_std[tid] = (float)kvarn_low_std_row(c, tid);
        col_std[tid] = (float)kvarn_low_std_col(c, tid);
    }
    __syncthreads();
    {
        float cmin, cmax, rmin, rmax;
        kvarn_std_ranges(col_std, row_std, reduce, &cmin, &cmax, &rmin, &rmax);
        const float cm = cmin < 1e-8f ? 1e-8f : cmin;
        const float rm = rmin < 1e-8f ? 1e-8f : rmin;
        if (tid == 0) *best_imbalance = cmax / cm + rmax / rm;
        __syncthreads();
    }
    for (int iter = 0; iter < sinkhorn_iters; ++iter) {
        if (tid < KVARN_DIM) {
            float std_c = col_std[tid];
            if (std_c < 1e-3f) std_c = 1e-3f;
            if (std_c > 1e3f) std_c = 1e3f;
            const double new_log = (double)log_s_col[tid] + kvarn_log_f64((double)std_c);
            double clamped = new_log;
            if (clamped < -0.3) clamped = -0.3;
            if (clamped > 10.0) clamped = 10.0;
            log_s_col[tid] = (float)clamped;
        }
        __syncthreads();
        if (tid < KVARN_DIM) {
            row_std[tid] = (float)kvarn_low_balanced_std_row(c, tid, log_s_col, log_s_row);
            col_std[tid] = (float)kvarn_low_balanced_std_col(c, tid, log_s_col, log_s_row);
        }
        __syncthreads();
        if (tid < KVARN_DIM) {
            float std_r = row_std[tid];
            if (std_r < 1e-3f) std_r = 1e-3f;
            if (std_r > 1e3f) std_r = 1e3f;
            const double new_log = (double)log_s_row[tid] + kvarn_log_f64((double)std_r);
            double clamped = new_log;
            if (clamped < -0.3) clamped = -0.3;
            if (clamped > 10.0) clamped = 10.0;
            log_s_row[tid] = (float)clamped;
        }
        __syncthreads();
        if (tid < KVARN_DIM) {
            row_std[tid] = (float)kvarn_low_balanced_std_row(c, tid, log_s_col, log_s_row);
            col_std[tid] = (float)kvarn_low_balanced_std_col(c, tid, log_s_col, log_s_row);
        }
        __syncthreads();
        float cmin, cmax, rmin, rmax;
        kvarn_std_ranges(col_std, row_std, reduce, &cmin, &cmax, &rmin, &rmax);
        const float cm = cmin < 1e-8f ? 1e-8f : cmin;
        const float rm = rmin < 1e-8f ? 1e-8f : rmin;
        if (tid == 0) {
            const float imb = cmax / cm + rmax / rm;
            *better = (imb <= *best_imbalance) ? 1.0f : 0.0f;
            if (*better > 0.0f) *best_imbalance = imb;
        }
        __syncthreads();
        if (tid < KVARN_DIM && *better > 0.0f) {
            s_col[tid] = (float)kvarn_exp_f64((double)log_s_col[tid]);
            s_row[tid] = (float)kvarn_exp_f64((double)log_s_row[tid]);
        }
        __syncthreads();
    }
}

// Quantize+pack un lado leyendo stage (mirror kvarn_quantize_pack_side
// con tile=kvarn_low_tile; f32 puro en balanced, LSB-first bit a bit).
__device__ __forceinline__ void kvarn_low_quantize_pack(
    const KvarnLowCtx* c,
    const float* s_col, const float* s_row,
    uint8_t* record,
    int payload_off, int bits,
    int scale_axis_off, int zp_axis_off, int other_axis_off)
{
    const int tid = (int)threadIdx.x;
    const int r = tid;
    if (r >= KVARN_DIM) return;
    const float absorb = s_row[r];
    float lo = 1e30f, hi = -1e30f;
    for (int j = 0; j < KVARN_DIM; ++j) {
        const float v = kvarn_low_tile(c, r, j) / (s_col[j] * absorb);
        if (v < lo) lo = v;
        if (v > hi) hi = v;
    }
    const float qmax = (float)((1 << bits) - 1);
    const float range = hi - lo;
    const float scale = range > 0.0f ? range / qmax : 1e-10f;
    for (int j = 0; j < KVARN_DIM; ++j) {
        const float v = kvarn_low_tile(c, r, j) / (s_col[j] * absorb);
        const float valf = roundf((v - lo) / scale);
        float cl = valf; if (cl < 0.0f) cl = 0.0f;
        if (cl > qmax) cl = qmax;
        const uint32_t q = (uint32_t)cl;
        const int bit_index = r * KVARN_DIM + j;
        for (int b = 0; b < bits; ++b) {
            const int dst_bit = bit_index * bits + b;
            const uint32_t bit_val = (q >> b) & 1u;
            uint8_t* bytep = record + payload_off + (dst_bit >> 3);
            const uint8_t mask = (uint8_t)(1u << (dst_bit & 7));
            if (bit_val) *bytep |= mask; else *bytep &= (uint8_t)~mask;
        }
    }
    __half* scale_axis = (__half*)(record + scale_axis_off);
    __half* zp_axis = (__half*)(record + zp_axis_off);
    scale_axis[r] = __float2half(absorb * scale);
    zp_axis[r] = __float2half(absorb * lo);
    // (other axis se escribe en kvarn_low_seal_side tras el pack)
}

// Seal un lado low-shmem (K o V según side_v) sobre el slot dado.
__device__ void kvarn_low_seal_side(
    const KvarnLowCtx* c,
    float* log_s_col, float* log_s_row,
    float* s_col, float* s_row,
    float* col_std, float* row_std,
    float* best_imbalance, float* better, float* reduce,
    uint8_t* record,
    int bits,
    int payload_off, int scale_axis_off, int zp_axis_off, int other_axis_off,
    int sinkhorn_iters)
{
    const int tid = (int)threadIdx.x;
    kvarn_low_sinkhorn(c, log_s_col, log_s_row, s_col, s_row,
        col_std, row_std, best_imbalance, better, reduce, sinkhorn_iters);
    __syncthreads();
    kvarn_low_quantize_pack(c, s_col, s_row, record,
        payload_off, bits, scale_axis_off, zp_axis_off, other_axis_off);
    __syncthreads();
    // other axis = s_col[j] (K: j=token; V: j=dim) — mirror seal sides.
    if (tid < KVARN_DIM) {
        __half* other_axis = (__half*)(record + other_axis_off);
        other_axis[tid] = __float2half(s_col[tid]);
    }
    __syncthreads();
}

// ---- A5 kernel: misma firma/semántica que kvarn_store_kernel ----
extern "C" __global__ void kvarn_store_lowshmem_kernel(
    const float*   current,
    const float*   current_v,
    const int64_t* indices,
    __half*        stage,
    uint8_t*       records,
    int n_tokens,
    int n_record_heads,
    int stream,
    int groups_per_stream,
    int record_bytes,
    int k_payload_off, int k_s_col_off, int k_zp_off, int k_s_row_off,
    int v_payload_off, int v_s_col_off, int v_s_row_off, int v_zp_off,
    int k_bits, int v_bits,
    int sinkhorn_iters,
    int stage_groups,
    int tail_groups,
    int swa,
    int eager_records)
{
    // 788 floats estáticos (< 48KB: sin dyn smem ni opt-in).
    __shared__ float log_s_col[KVARN_DIM];
    __shared__ float log_s_row[KVARN_DIM];
    __shared__ float s_col[KVARN_DIM];
    __shared__ float s_row[KVARN_DIM];
    __shared__ float col_std[KVARN_DIM];
    __shared__ float row_std[KVARN_DIM];
    __shared__ float best_imbalance[1];
    __shared__ float better[1];
    __shared__ float reduce[16];
    __shared__ float vec[128];
    __shared__ float vec_v[128];

    const int head = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    if (head >= n_record_heads) return;

    for (int t = 0; t < n_tokens; ++t) {
        const int64_t enc = indices[t];
        if (enc == -1) continue;
        bool explicitly_staged = false;
        int assigned_slot = -1;
        const int64_t cell = kvarn_read_cell(enc, explicitly_staged, &assigned_slot);
        const int group = (int)(cell / KVARN_DIM);
        const int pos = (int)(cell - (int64_t)group * KVARN_DIM);
        const int record_group = swa
            ? (group % groups_per_stream)
            : (stream * groups_per_stream + group);

        // ---- Delayed flush (no-eager): sellar el grupo que sale de la
        //      ventana tail al ENTRAR en uno nuevo (pos==0), re-leyendo
        //      el stage global (low-shmem: sin tile smem).
        if (!eager_records && pos == 0 &&
                (swa ? group >= tail_groups : group > tail_groups)) {
            const int flush_group = group - tail_groups;
            const int flush_record_group = swa
                ? (flush_group % groups_per_stream)
                : (stream * groups_per_stream + flush_group);
            const int flush_slot = kvarn_stage_slot_for_group_std(
                swa, flush_group, stage_groups, tail_groups);
            uint8_t* rec = records
                + ((int64_t)flush_record_group * n_record_heads + head) * record_bytes;
            KvarnLowCtx ck = { stage, n_record_heads, head, 0, stream,
                               flush_slot, stage_groups };
            KvarnLowCtx cv = { stage, n_record_heads, head, 1, stream,
                               flush_slot, stage_groups };
            kvarn_low_seal_side(&ck, log_s_col, log_s_row, s_col, s_row,
                col_std, row_std, best_imbalance, better, reduce,
                rec, k_bits, k_payload_off, k_s_col_off, k_zp_off, k_s_row_off,
                sinkhorn_iters);
            kvarn_low_seal_side(&cv, log_s_col, log_s_row, s_col, s_row,
                col_std, row_std, best_imbalance, better, reduce,
                rec, v_bits, v_payload_off, v_s_row_off, v_zp_off, v_s_col_off,
                sinkhorn_iters);
        }

        // ---- Stage write (idéntico hishmem): WHT-128 K/V, filas 2h/2h+1.
        {
            const float* cur_v = (current_v != nullptr) ? current_v : current;
            if (tid < 128) {
                vec[tid] = current[((int64_t)t * n_record_heads + head) * KVARN_DIM + tid];
                vec_v[tid] = cur_v[((int64_t)t * n_record_heads + head) * KVARN_DIM + tid];
            }
            __syncthreads();
            kvarn_wht_128_impl(vec, tid);
            kvarn_wht_128_impl(vec_v, tid);
            __syncthreads();
            const int slot = kvarn_stage_slot_for_group_ext(
                swa, group, stage_groups, tail_groups, assigned_slot);
            const int stage_pos = stream * KVARN_DIM * stage_groups
                + slot * KVARN_DIM + pos;
            const int stage_heads = 2 * n_record_heads;
            if (tid < 128) {
                stage[((int64_t)stage_pos * stage_heads + 2 * head) * KVARN_DIM + tid] =
                    __float2half(vec[tid]);
                stage[((int64_t)stage_pos * stage_heads + 2 * head + 1) * KVARN_DIM + tid] =
                    __float2half(vec_v[tid]);
            }
            __syncthreads();
        }

        // ---- Eager seal: al COMPLETAR un grupo (pos==127), sellar YA
        //      re-leyendo el stage.
        if (eager_records && pos == KVARN_DIM - 1
                && (swa || group > 0)) {
            const int slot = kvarn_stage_slot_for_group_std(
                swa, group, stage_groups, tail_groups);
            uint8_t* rec = records
                + ((int64_t)record_group * n_record_heads + head) * record_bytes;
            KvarnLowCtx ck = { stage, n_record_heads, head, 0, stream,
                               slot, stage_groups };
            KvarnLowCtx cv = { stage, n_record_heads, head, 1, stream,
                               slot, stage_groups };
            kvarn_low_seal_side(&ck, log_s_col, log_s_row, s_col, s_row,
                col_std, row_std, best_imbalance, better, reduce,
                rec, k_bits, k_payload_off, k_s_col_off, k_zp_off, k_s_row_off,
                sinkhorn_iters);
            kvarn_low_seal_side(&cv, log_s_col, log_s_row, s_col, s_row,
                col_std, row_std, best_imbalance, better, reduce,
                rec, v_bits, v_payload_off, v_s_row_off, v_zp_off, v_s_col_off,
                sinkhorn_iters);
        }
    }
}

// ============================================================================
// 9.4 (lane-b) D2-store: kvarn_store_d256_kernel — D=256 con cross-slice
// ============================================================================
// Port de beellama kvarn.cu:1055-1210 (store por cabezas físicas): una
// cabeza LÓGICA D=256 son 2 cabezas FÍSICAS de 128 (head0 = blockIdx.x·2).
// El stage write aplica WHT-128 intra-slice a cada slice del vector lógico
// + butterfly cross-slice (a,b)→((a+b),(a-b))·1/√2 — espejo del CPU ref
// hadamardSlicesRows (kvarn.zig, D2 ratificada) y del Q-rot del portable
// d256 (fattn_kvarn_portable.cu:820-824). El sellado de records es el
// A4 estándar per cabeza física (records [n_kv_heads·2]).
//
// `current` es el K/V lógico: [n_tokens, n_logical_heads, 256]; cada
// bloque saca los 2 slices del head lógico blockIdx.x.
// `n_record_heads` = heads FÍSICAS TOTALES (n_logical_heads·2) — igual que
// beellama (`head0 + head_slices > n_heads` con n_heads físicas).
extern "C" __global__ void kvarn_store_d256_kernel(
    const float*   current,      // [n_tokens, n_logical_heads, 256] K
    const float*   current_v,    // idem V (nullptr ⇒ mismo buffer)
    const int64_t* indices,      // [n_tokens]
    __half*        stage,        // stage f16 rotated [pos][2·n_record_heads][128]
    uint8_t*       records,      // records C1 [groups][n_record_heads]
    int n_tokens,
    int n_logical_heads,         // heads LÓGICAS D=256
    int n_record_heads,          // FÍSICAS totales (= n_logical_heads·2)
    int stream,
    int groups_per_stream,
    int record_bytes,
    int k_payload_off, int k_s_col_off, int k_zp_off, int k_s_row_off,
    int v_payload_off, int v_s_col_off, int v_s_row_off, int v_zp_off,
    int k_bits, int v_bits,
    int sinkhorn_iters,
    int stage_groups,
    int tail_groups,
    int swa,
    int eager_records)
{
    extern __shared__ float smem[];
    float* tile = smem;                       // 16384 (K)
    float* log_s_col = tile + 16384;           // 128
    float* log_s_row = log_s_col + 128;        // 128
    float* s_col = log_s_row + 128;            // 128
    float* s_row = s_col + 128;                // 128
    float* col_std = s_row + 128;              // 128
    float* row_std = col_std + 128;            // 128
    float* best_imbalance = row_std + 128;     // 1
    float* better = best_imbalance + 1;        // 1
    float* reduce = better + 1;                // 16

    const int head0 = (int)blockIdx.x * 2;     // 2 cabezas físicas
    if (head0 + 2 > n_record_heads) return;
    const int tid = (int)threadIdx.x;

    for (int t = 0; t < n_tokens; ++t) {
        const int64_t enc = indices[t];
        if (enc == -1) continue;
        bool explicitly_staged = false;
        int assigned_slot = -1;
        const int64_t cell = kvarn_read_cell(enc, explicitly_staged, &assigned_slot);
        const int group = (int)(cell / KVARN_DIM);
        const int pos = (int)(cell - (int64_t)group * KVARN_DIM);
        const int record_group = swa
            ? (group % groups_per_stream)
            : (stream * groups_per_stream + group);

        // ---- Delayed flush (idéntico al store 128: sellar K y V de cada
        //      cabeza física head0/head0+1 al rotar la ventana).
        if (!eager_records && pos == 0 &&
                (swa ? group >= tail_groups : group > tail_groups)) {
            const int flush_group = group - tail_groups;
            const int flush_record_group = swa
                ? (flush_group % groups_per_stream)
                : (stream * groups_per_stream + flush_group);
            const int flush_slot = kvarn_stage_slot_for_group_std(
                swa, flush_group, stage_groups, tail_groups);
            for (int h = head0; h < head0 + 2; ++h) {
                uint8_t* rec = records
                    + ((int64_t)flush_record_group * n_record_heads + h) * record_bytes;
                // K side (fila 2h del stage).
                kvarn_load_tile_from_stage(
                    stage, n_record_heads, 2 * h, stream, flush_slot, stage_groups,
                    tile, tid);
                __syncthreads();
                kvarn_seal_k_side(
                    tile, log_s_col, log_s_row, s_col, s_row, col_std, row_std,
                    best_imbalance, better, reduce,
                    rec, k_bits,
                    k_payload_off, k_s_col_off, k_zp_off, k_s_row_off,
                    sinkhorn_iters);
                __syncthreads();
                // V side (fila 2h+1).
                kvarn_load_tile_from_stage(
                    stage, n_record_heads, 2 * h + 1, stream, flush_slot, stage_groups,
                    tile, tid);
                __syncthreads();
                kvarn_seal_v_side(
                    tile, log_s_col, log_s_row, s_col, s_row, col_std, row_std,
                    best_imbalance, better, reduce,
                    rec, v_bits,
                    v_payload_off, v_s_col_off, v_zp_off, v_s_row_off,
                    sinkhorn_iters);
                __syncthreads();
            }
        }

        // ---- Stage write D=256: WHT-128 intra-slice ×2 + cross-slice
        //      butterfly ×1/√2 (beellama kvarn.cu:1086-1105).
        //      K y V se cross-mezclan INDEPENDIENTEMENTE: el head lógico h
        //      tiene slices K(h0,h0+1) y V(h0,h0+1) como heads FÍSICAS
        //      — C2v2: fila K(2·f), V(2·f+1) por física f.
        {
            __shared__ float vec_k[2][128];
            __shared__ float vec_v[2][128];
            const float* cur_v = (current_v != nullptr) ? current_v : current;
            if (tid < 128) {
                // K slices del head lógico head0/2.
                vec_k[0][tid] = current[((int64_t)t * n_logical_heads + head0 / 2) * 256 + tid];
                vec_k[1][tid] = current[((int64_t)t * n_logical_heads + head0 / 2) * 256 + 128 + tid];
                // V slices.
                vec_v[0][tid] = cur_v[((int64_t)t * n_logical_heads + head0 / 2) * 256 + tid];
                vec_v[1][tid] = cur_v[((int64_t)t * n_logical_heads + head0 / 2) * 256 + 128 + tid];
            }
            __syncthreads();
            // WHT-128 intra-slice de cada slice (K y V).
            for (int s = 0; s < 2; ++s) {
                kvarn_wht_128_impl(vec_k[s], tid);
                kvarn_wht_128_impl(vec_v[s], tid);
                __syncthreads();
            }
            // Cross-slice K: (a+b, a-b)·1/√2 → heads físicas h0/h0+1.
            // Cross-slice V: ídem sobre los slices V.
            if (tid < 128) {
                const float inv_sqrt2 = 0.707106781186547524f;
                {
                    const float a = vec_k[0][tid];
                    const float b = vec_k[1][tid];
                    vec_k[0][tid] = (a + b) * inv_sqrt2;
                    vec_k[1][tid] = (a - b) * inv_sqrt2;
                }
                {
                    const float a = vec_v[0][tid];
                    const float b = vec_v[1][tid];
                    vec_v[0][tid] = (a + b) * inv_sqrt2;
                    vec_v[1][tid] = (a - b) * inv_sqrt2;
                }
            }
            __syncthreads();

            const int slot = kvarn_stage_slot_for_group_ext(
                swa, group, stage_groups, tail_groups, assigned_slot);
            const int stage_pos = stream * KVARN_DIM * stage_groups
                + slot * KVARN_DIM + pos;
            const int stage_heads = 2 * n_record_heads;
            if (tid < 128) {
                // C2v2: K física f en fila 2f, V física f en fila 2f+1.
                // K slices → físicas head0/head0+1 (bloque lógico head0/2).
                stage[((int64_t)stage_pos * stage_heads + 2 * head0) * KVARN_DIM + tid] =
                    __float2half(vec_k[0][tid]);
                stage[((int64_t)stage_pos * stage_heads + 2 * (head0 + 1)) * KVARN_DIM + tid] =
                    __float2half(vec_k[1][tid]);
                // V slices → físicas head0/head0+1.
                stage[((int64_t)stage_pos * stage_heads + 2 * head0 + 1) * KVARN_DIM + tid] =
                    __float2half(vec_v[0][tid]);
                stage[((int64_t)stage_pos * stage_heads + 2 * (head0 + 1) + 1) * KVARN_DIM + tid] =
                    __float2half(vec_v[1][tid]);
            }
            __syncthreads();
        }

        // ---- Eager seal (pos==127): sellar AMBAS cabezas físicas.
        if (eager_records && pos == KVARN_DIM - 1
                && (swa || group > 0)) {
            const int slot = kvarn_stage_slot_for_group_std(
                swa, group, stage_groups, tail_groups);
            for (int h = head0; h < head0 + 2; ++h) {
                uint8_t* rec = records
                    + ((int64_t)record_group * n_record_heads + h) * record_bytes;
                kvarn_load_tile_from_stage(
                    stage, n_record_heads, 2 * h, stream, slot, stage_groups,
                    tile, tid);
                __syncthreads();
                kvarn_seal_k_side(
                    tile, log_s_col, log_s_row, s_col, s_row, col_std, row_std,
                    best_imbalance, better, reduce,
                    rec, k_bits,
                    k_payload_off, k_s_col_off, k_zp_off, k_s_row_off,
                    sinkhorn_iters);
                __syncthreads();
                kvarn_load_tile_from_stage(
                    stage, n_record_heads, 2 * h + 1, stream, slot, stage_groups,
                    tile, tid);
                __syncthreads();
                kvarn_seal_v_side(
                    tile, log_s_col, log_s_row, s_col, s_row, col_std, row_std,
                    best_imbalance, better, reduce,
                    rec, v_bits,
                    v_payload_off, v_s_col_off, v_zp_off, v_s_row_off,
                    sinkhorn_iters);
                __syncthreads();
            }
        }
    }
}

// ============================================================================
// A6 (Dev-A): materialize — records/stage C1 → filas f16 (dominio original)
// ============================================================================
//
// Grid (n_tokens, n_heads), block 128 (un thread por dim). Un stream por
// launch (mismo convenio que el store). Por token: resolver (group, pos) y
// la fuente (stage caliente o record sellado); por dim: decodificar
// val = (q·scale[row] + zp[row])·other[col] con el mapping C1 del lado
// (K: fila=dim,col=tok; V: fila=tok,col=dim). Si !emit_rotated, WHT⁻¹
// (involutiva) sobre la fila antes de escribir f16.
//
// Huecos (enc==-1): el bloque retorna temprano y la salida queda INTACTA
// (semántica upstream — el caller pre-rellena o enmascara).
// Celdas sin fuente (ni stage ni record): se escribe 0 (mirror de
// load_rotated).

extern "C" __global__ void kvarn_materialize_kernel(
    const uint8_t* records,
    const __half* stage,
    const int64_t* indices,      // no se lee si !swa && !read_indirect
    __half* out,                 // [n_tokens, n_heads, KVARN_DIM]
    int n_tokens,
    int n_heads,
    int stream,
    int groups_per_stream,
    int record_bytes,
    int payload_off,
    int scale_off,               // eje por FILA del tile
    int zp_off,                  // eje por FILA del tile
    int other_off,               // eje por COLUMNA del tile
    int bits,
    int value,                   // 0=K, 1=V
    int stage_groups,
    int tail_groups,
    int swa,
    int eager_records,
    int read_indirect,
    int live_group,
    int live_pos,
    int emit_rotated)
{
    const int token = (int)blockIdx.x;
    const int head = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    if (token >= n_tokens || head >= n_heads) return;
    if (tid >= KVARN_DIM) return;

    // Desc local para reusar los helpers de membresía de kvarn_desc.cuh.
    KvarnDesc d;
    d.records = records;
    d.stage = stage;
    d.indices = indices;
    d.n_record_heads = n_heads;
    d.live_group = live_group;
    d.live_pos = live_pos;
    d.stream = stream;
    d.head_base = head;
    d.groups_per_stream = groups_per_stream;
    d.record_bytes = record_bytes;
    d.stage_groups = stage_groups;
    d.tail_groups = tail_groups;
    d.bits = bits;
    d.value = value;
    d.swa = swa;
    d.head_slices = 1; // D=128 (D>=256 gated D2)
    d.eager_records = eager_records;
    d.read_indirect = read_indirect;
    d.original_domain = !emit_rotated;

    int group, pos;
    bool from_stage, from_record;
    int aslot = -1;
    if (swa || read_indirect) {
        const int64_t enc = indices[token];
        if (enc == -1) return; // hueco: salida intacta
        bool es = false;
        const int64_t cell = kvarn_read_cell(enc, es, &aslot);
        group = (int)(cell / KVARN_DIM);
        pos = (int)(cell - (int64_t)group * KVARN_DIM);
        from_stage = es
            || (!(read_indirect && !swa) && kvarn_group_from_stage(d, group));
        from_record = !es && (read_indirect && !swa
            ? true : kvarn_group_from_record(d, group));
    } else {
        group = token / KVARN_DIM;
        pos = token - group * KVARN_DIM;
        from_stage = kvarn_group_from_stage(d, group);
        from_record = kvarn_group_from_record(d, group);
    }

    float v = 0.0f;
    if (from_stage) {
        const int stage_pos = kvarn_stage_pos(d, group, pos, aslot);
        // C2v2: fila K=2h / V=2h+1 según `value`.
        const int row = 2 * head + (value ? 1 : 0);
        v = __half2float(stage[
            ((int64_t)stage_pos * (2 * n_heads) + row) * KVARN_DIM + tid]);
    } else if (from_record) {
        const int rg = swa
            ? (group % groups_per_stream)
            : (stream * groups_per_stream + group);
        const uint8_t* rec = records
            + ((int64_t)rg * n_heads + head) * record_bytes;
        const __half* scale_ax = (const __half*)(rec + scale_off);
        const __half* zp_ax = (const __half*)(rec + zp_off);
        const __half* other_ax = (const __half*)(rec + other_off);
        if (!value) {
            // K: fila=dim(tid), col=token(pos).
            const uint32_t q = kvarn_unpack(rec + payload_off, tid * KVARN_DIM + pos, bits);
            v = ((float)q * __half2float(scale_ax[tid]) + __half2float(zp_ax[tid]))
                * __half2float(other_ax[pos]);
        } else {
            // V: fila=token(pos), col=dim(tid).
            const uint32_t q = kvarn_unpack(rec + payload_off, pos * KVARN_DIM + tid, bits);
            v = ((float)q * __half2float(scale_ax[pos]) + __half2float(zp_ax[pos]))
                * __half2float(other_ax[tid]);
        }
    } // else: celda no respaldada ⇒ 0.

    __shared__ float row[KVARN_DIM];
    row[tid] = v;
    __syncthreads();
    if (!emit_rotated) {
        kvarn_wht_128_impl(row, tid); // involutiva → dominio original
        __syncthreads();
    }
    out[((int64_t)token * n_heads + head) * KVARN_DIM + tid] =
        __float2half(row[tid]);
}
