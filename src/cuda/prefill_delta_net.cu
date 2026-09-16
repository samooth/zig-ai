// ─── Chunked Batched ΔNet Prefill Kernel v5 (STUDY §5.2) ─────────────────────
// Processes K tokens per v-head in a single kernel launch, keeping the S_V×S_V
// state in registers across the K-token loop.
//
// v5: K is a RUNTIME parameter (not template). The engine passes
// K = min(64, n - t_start) for each chunk. This avoids buffer overflow when
// the engine's buffer is sized for n < 64 (e.g., n=30).
//
// Layout (zig-ai interleaved):
//   conv_out: [n, qkv_dim] with q(0)|k(key_dim)|v(2·key_dim)
//   gate:     [n, dt_rank]
//   beta:     [n, dt_rank]
//   state:    [n_v_heads, S_V, S_V] contiguous per-head (INOUT)
//   attn_out: [n, d_inner] interleaved
//
// Grid:  (n_v_heads, n_seqs, (S_V + 3) / 4)
// Block: (warp_size=32, 4, 1) — 128 threads, 4 warps

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

#define WARP 32
#define S_V 128
#define MAX_K 512  // 1.11 (lane-c): fused CH ubatch-wide — vestigial para el
                   // unroll (v5 loopea por n_tokens runtime), cota doc del K512

template <int warp_size>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    // FIX §5.2 (coordinador): __shfl_down_sync deja la suma completa SOLO en
    // lane 0 — y las lanes altas leen su PROPIO valor cuando el offset sale del
    // warp (semántica shfl-down: fuente out-of-range = valor propio), lo que
    // infla delta_col/o_col en 31/32 de las columnas (repro: /tmp/chunk_repro).
    // Butterfly XOR (unsloth common.cuh:456): TODAS las lanes terminan con la
    // suma completa — sin broadcast posterior y sin lanes corruptas.
    #pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset /= 2)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

template <int K, bool KDA>
__device__ void deltaNetChunkLoop(
    float* __restrict__ head_state,
    float* __restrict__ attn_out,
    int d_inner,
    const float* __restrict__ conv_out, int q_off, int k_off, int v_off, int qkv_stride,
    const float* __restrict__ gate, int dt_stride,
    const float* __restrict__ beta, int dt_stride_b,
    int col, int lane, int hv, float scale, int n_v_heads, int n_k_heads, int head_v_dim,
    int n_tokens)  // Runtime: actual tokens to process (<= K)
{
    constexpr int warp_size = WARP;
    constexpr int rows_per_lane = S_V / warp_size;

    const int hk = hv % n_k_heads; // 7.1b: módulo (ggml_repeat_4d), no bloque
    const int q_offset = q_off + hk * head_v_dim;
    const int k_offset = k_off + hk * head_v_dim;
    const int v_offset = v_off + hv * head_v_dim;

    float s_shard[rows_per_lane];
    #pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r] = head_state[i * S_V + col];
    }

    for (int t = 0; t < n_tokens; t++) {
        const int base = t * qkv_stride;
        const float* q_t = conv_out + base + q_offset;
        const float* k_t = conv_out + base + k_offset;
        const float* v_t = conv_out + base + v_offset;
        const float b_val = beta[t * dt_stride_b + hv];

        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
        #pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(gate[t * dt_stride + hv]);

            #pragma unroll
            for (int r = 0; r < rows_per_lane; r++)
                s_shard[r] *= g_val;

            float sk_shard = 0.0f;
            #pragma unroll
            for (int r = 0; r < rows_per_lane; r++)
                sk_shard += s_shard[r] * k_reg[r];
            const float sk_col = warp_reduce_sum<warp_size>(sk_shard);

            const float delta_col = b_val * (v_t[col] - sk_col);

            #pragma unroll
            for (int r = 0; r < rows_per_lane; r++)
                s_shard[r] += k_reg[r] * delta_col;

            float o_shard = 0.0f;
            #pragma unroll
            for (int r = 0; r < rows_per_lane; r++)
                o_shard += s_shard[r] * q_reg[r];
            const float o_col = warp_reduce_sum<warp_size>(o_shard) * scale;

            if (lane == 0)
                attn_out[t * d_inner + hv * S_V + col] = o_col;
        } else {
            const float* g_t = gate + (size_t)t * S_V;

            float sk_shard = 0.0f;
            #pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                const float gv = expf(g_t[i]);
                s_shard[r] *= gv;
                sk_shard += s_shard[r] * k_reg[r];
            }
            const float sk_col = warp_reduce_sum<warp_size>(sk_shard);

            const float delta_col = b_val * (v_t[col] - sk_col);

            #pragma unroll
            for (int r = 0; r < rows_per_lane; r++)
                s_shard[r] += k_reg[r] * delta_col;

            float o_shard = 0.0f;
            #pragma unroll
            for (int r = 0; r < rows_per_lane; r++)
                o_shard += s_shard[r] * q_reg[r];
            const float o_col = warp_reduce_sum<warp_size>(o_shard) * scale;

            if (lane == 0)
                attn_out[t * d_inner + hv * S_V + col] = o_col;
        }
    }

    #pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        head_state[i * S_V + col] = s_shard[r];
    }
}

// ─── Global wrappers ──────────────────────────────────────────────────────────
// Each wrapper takes n_tokens as a runtime parameter. The caller (engine or test)
// passes min(K, n - t_start) for the last chunk.

extern "C" __global__ void prefillDeltaNetChunk_nkda_K64(
    const float* conv_out, int q_off, int k_off, int v_off, int qkv_stride,
    const float* gate, int dt_stride, const float* beta, int dt_stride_b,
    float* state, float* attn_out, int d_inner, int t_start, float scale,
    int n_v_heads, int n_k_heads, int head_v_dim, int n_tokens)
{
    const int col = blockIdx.z * blockDim.y + threadIdx.y;
    if (col >= S_V) return;
    const int hv = blockIdx.x;
    if (hv >= n_v_heads) return;
    (void)t_start; // ABI: host passes it; v5 loop uses runtime n_tokens instead.
    float* head_state = state + hv * S_V * S_V;
    deltaNetChunkLoop<64, false>(head_state, attn_out, d_inner,
        conv_out, q_off, k_off, v_off, qkv_stride, gate, dt_stride, beta, dt_stride_b,
        col, threadIdx.x, hv, scale, n_v_heads, n_k_heads, head_v_dim, n_tokens);
}

extern "C" __global__ void prefillDeltaNetChunk_nkda_K128(
    const float* conv_out, int q_off, int k_off, int v_off, int qkv_stride,
    const float* gate, int dt_stride, const float* beta, int dt_stride_b,
    float* state, float* attn_out, int d_inner, int t_start, float scale,
    int n_v_heads, int n_k_heads, int head_v_dim, int n_tokens)
{
    const int col = blockIdx.z * blockDim.y + threadIdx.y;
    if (col >= S_V) return;
    const int hv = blockIdx.x;
    if (hv >= n_v_heads) return;
    (void)t_start; // ABI: host passes it; v5 loop uses runtime n_tokens instead.
    float* head_state = state + hv * S_V * S_V;
    deltaNetChunkLoop<128, false>(head_state, attn_out, d_inner,
        conv_out, q_off, k_off, v_off, qkv_stride, gate, dt_stride, beta, dt_stride_b,
        col, threadIdx.x, hv, scale, n_v_heads, n_k_heads, head_v_dim, n_tokens);
}

extern "C" __global__ void prefillDeltaNetChunk_kda_K64(
    const float* conv_out, int q_off, int k_off, int v_off, int qkv_stride,
    const float* gate, int dt_stride, const float* beta, int dt_stride_b,
    float* state, float* attn_out, int d_inner, int t_start, float scale,
    int n_v_heads, int n_k_heads, int head_v_dim, int n_tokens)
{
    const int col = blockIdx.z * blockDim.y + threadIdx.y;
    if (col >= S_V) return;
    const int hv = blockIdx.x;
    if (hv >= n_v_heads) return;
    (void)t_start; // ABI: host passes it; v5 loop uses runtime n_tokens instead.
    float* head_state = state + hv * S_V * S_V;
    deltaNetChunkLoop<64, true>(head_state, attn_out, d_inner,
        conv_out, q_off, k_off, v_off, qkv_stride, gate, dt_stride, beta, dt_stride_b,
        col, threadIdx.x, hv, scale, n_v_heads, n_k_heads, head_v_dim, n_tokens);
}

extern "C" __global__ void prefillDeltaNetChunk_kda_K128(
    const float* conv_out, int q_off, int k_off, int v_off, int qkv_stride,
    const float* gate, int dt_stride, const float* beta, int dt_stride_b,
    float* state, float* attn_out, int d_inner, int t_start, float scale,
    int n_v_heads, int n_k_heads, int head_v_dim, int n_tokens)
{
    const int col = blockIdx.z * blockDim.y + threadIdx.y;
    if (col >= S_V) return;
    const int hv = blockIdx.x;
    if (hv >= n_v_heads) return;
    (void)t_start;
    float* head_state = state + hv * S_V * S_V;
    deltaNetChunkLoop<128, true>(head_state, attn_out, d_inner,
        conv_out, q_off, k_off, v_off, qkv_stride, gate, dt_stride, beta, dt_stride_b,
        col, threadIdx.x, hv, scale, n_v_heads, n_k_heads, head_v_dim, n_tokens);
}

extern "C" __global__ void prefillDeltaNetChunk_nkda_K256(
    const float* conv_out, int q_off, int k_off, int v_off, int qkv_stride,
    const float* gate, int dt_stride, const float* beta, int dt_stride_b,
    float* state, float* attn_out, int d_inner, int t_start, float scale,
    int n_v_heads, int n_k_heads, int head_v_dim, int n_tokens)
{
    const int col = blockIdx.z * blockDim.y + threadIdx.y;
    if (col >= S_V) return;
    const int hv = blockIdx.x;
    if (hv >= n_v_heads) return;
    (void)t_start;
    float* head_state = state + hv * S_V * S_V;
    deltaNetChunkLoop<256, false>(head_state, attn_out, d_inner,
        conv_out, q_off, k_off, v_off, qkv_stride, gate, dt_stride, beta, dt_stride_b,
        col, threadIdx.x, hv, scale, n_v_heads, n_k_heads, head_v_dim, n_tokens);
}

extern "C" __global__ void prefillDeltaNetChunk_nkda_K512(
    const float* conv_out, int q_off, int k_off, int v_off, int qkv_stride,
    const float* gate, int dt_stride, const float* beta, int dt_stride_b,
    float* state, float* attn_out, int d_inner, int t_start, float scale,
    int n_v_heads, int n_k_heads, int head_v_dim, int n_tokens)
{
    const int col = blockIdx.z * blockDim.y + threadIdx.y;
    if (col >= S_V) return;
    const int hv = blockIdx.x;
    if (hv >= n_v_heads) return;
    (void)t_start;
    float* head_state = state + hv * S_V * S_V;
    deltaNetChunkLoop<512, false>(head_state, attn_out, d_inner,
        conv_out, q_off, k_off, v_off, qkv_stride, gate, dt_stride, beta, dt_stride_b,
        col, threadIdx.x, hv, scale, n_v_heads, n_k_heads, head_v_dim, n_tokens);
}

extern "C" __global__ void prefillDeltaNetChunk_kda_K256(
    const float* conv_out, int q_off, int k_off, int v_off, int qkv_stride,
    const float* gate, int dt_stride, const float* beta, int dt_stride_b,
    float* state, float* attn_out, int d_inner, int t_start, float scale,
    int n_v_heads, int n_k_heads, int head_v_dim, int n_tokens)
{
    const int col = blockIdx.z * blockDim.y + threadIdx.y;
    if (col >= S_V) return;
    const int hv = blockIdx.x;
    if (hv >= n_v_heads) return;
    (void)t_start;
    float* head_state = state + hv * S_V * S_V;
    deltaNetChunkLoop<256, true>(head_state, attn_out, d_inner,
        conv_out, q_off, k_off, v_off, qkv_stride, gate, dt_stride, beta, dt_stride_b,
        col, threadIdx.x, hv, scale, n_v_heads, n_k_heads, head_v_dim, n_tokens);
}

extern "C" __global__ void prefillDeltaNetChunk_kda_K512(
    const float* conv_out, int q_off, int k_off, int v_off, int qkv_stride,
    const float* gate, int dt_stride, const float* beta, int dt_stride_b,
    float* state, float* attn_out, int d_inner, int t_start, float scale,
    int n_v_heads, int n_k_heads, int head_v_dim, int n_tokens)
{
    const int col = blockIdx.z * blockDim.y + threadIdx.y;
    if (col >= S_V) return;
    const int hv = blockIdx.x;
    if (hv >= n_v_heads) return;
    (void)t_start;
    float* head_state = state + hv * S_V * S_V;
    deltaNetChunkLoop<512, true>(head_state, attn_out, d_inner,
        conv_out, q_off, k_off, v_off, qkv_stride, gate, dt_stride, beta, dt_stride_b,
        col, threadIdx.x, hv, scale, n_v_heads, n_k_heads, head_v_dim, n_tokens);
}

// 1.11 (lane-c): fused CH ubatch-wide K=512 — removido: los wrappers K512
// eran vestigiales (v5 loop por n_tokens runtime) y no hay dispatch desde
// Zig; el engine llama solo a K64/K128 según chunk size. Esto elimina
// compilación muerta y reduce superficie de warnings ptxas.
//
// extern "C" __global__ void prefillDeltaNetChunk_nkda_K512(...) { ... }
// extern "C" __global__ void prefillDeltaNetChunk_kda_K512(...) { ... }

// ─── STUDY §5.2 WY (1.4, lane-b): representación WY — prefill batched ─────────
// Oráculo de referencia: src/transformer/prefill_wy.zig (paridad rel<1e-3 vs
// per-token; los 3 bugs de orientación del WY están documentados ahí).
//
//  K1 prefillWYSolve (grid: n_chunks × n_v_heads, block 256):
//    Todo lo INDEPENDIENTE del estado entrante, batcheado:
//      g_cs cumsum → T=tri_strict(kb)/kq (dots S por par) → A por
//      sustitución (T+A en smem, columna-thread) → kg/q_g/k_cd/A·v_b
//      a SCRATCH GLOBAL.
//  K2 prefillWYState (grid: S_v × n_v_heads, block S_v):
//    Lo DEPENDIENTE del estado — secuencial en chunks, paralelo en
//    columnas s del estado (grid.x = S_v; cada block una columna):
//      v_new = (A·v_b) − k_cdᵀ·S_col
//      o     = S_col·q_g + Σ_{j≤t} v_new[j]·kq[t][j]   (×scale)
//      S_col = S_col·exp(g_last) + Σ_t kg[t]·v_new[t]
//
// Scratch global f32 (launcher lo aloca — layouts por (chunk,head)):
//   wy_g_cs [nC][nvh][CS] · wy_attn(A) [nC][nvh][CS][CS] · wy_kq idem
//   wy_kg [nC][nvh][CS][S] · wy_q_g idem · wy_k_cd [nC][nvh][S][CS]
//   wy_avb [nC][nvh][CS][S] (A·v_b precomputado en K1) · wy_g_last [nC][nvh]
// Entradas: layouts idénticos a v5 (conv_out intercalado q|k|v, gate/beta
// [n, dt_rank]). CS=64 (nkda, Qwen3.5); S = head_v_dim (≤128).

#define WY_CS 64
#define WY_MAX_S 128

extern "C" __global__ void prefillWYSolve(
    const float* __restrict__ conv_out, int q_off, int k_off, int v_off, int qkv_stride,
    const float* __restrict__ gate, int dt_stride, const float* __restrict__ beta, int dt_stride_b,
    int n_v_heads, int n_k_heads, int head_v_dim, int n_tokens, int n_chunks,
    float* __restrict__ wy_g_cs, float* __restrict__ wy_attn, float* __restrict__ wy_kq,
    float* __restrict__ wy_kg, float* __restrict__ wy_q_g, float* __restrict__ wy_k_cd,
    float* __restrict__ wy_avb, float* __restrict__ wy_g_last)
{
    const int chunk = blockIdx.x;
    const int hv = blockIdx.y;
    if (chunk >= n_chunks || hv >= n_v_heads) return;
    const int hk = hv % n_k_heads;
    const int S = head_v_dim;
    const int t0 = chunk * WY_CS;
    const int cs = min(WY_CS, n_tokens - t0);
    if (cs <= 0) return;

    const int tid = threadIdx.x;
    const int nthr = blockDim.x;      // 256

    const size_t base2 = (size_t)chunk * n_v_heads + hv;
    float* g_cs = wy_g_cs + base2 * WY_CS;
    float* A    = wy_attn + base2 * WY_CS * WY_CS;
    float* kq   = wy_kq + base2 * WY_CS * WY_CS;
    float* kg   = wy_kg + base2 * WY_CS * S;
    float* q_g  = wy_q_g + base2 * WY_CS * S;
    float* k_cd = wy_k_cd + base2 * S * WY_CS;
    float* avb  = wy_avb + base2 * WY_CS * S;

    // smem: T tri-estricto (para el solve) + A de trabajo + g_cs.
    // [65] padding anti bank-conflict. 64×65×4×2 ≈ 33KB.
    __shared__ float T_s[WY_CS][WY_CS + 1];
    __shared__ float A_s[WY_CS][WY_CS + 1];
    __shared__ float gcs_s[WY_CS];

    // ── 1. g_cs cumsum (clamp 50) — thread 0, cs≤64.
    if (tid == 0) {
        float acc = 0.f;
        for (int i = 0; i < cs; ++i) {
            acc += gate[(size_t)(t0 + i) * dt_stride + hv];
            gcs_s[i] = fminf(acc, 50.f);
        }
        wy_g_last[base2] = __expf(gcs_s[cs - 1]);
    }
    __syncthreads();

    // ── 2. T[i][j] = b_i·(k_i·k_j)·decay[i][j] (j<i) ; kq[t][j] =
    //       (q_t·k_j)·decay[t][j] (j≤t). Un thread por par (i,j).
    //       NB: el dot S=128 por par — 128 FMA/par, 4096 pares / 256 thr.
    for (int idx = tid; idx < cs * cs; idx += nthr) {
        const int i = idx / cs;
        const int j = idx % cs;
        const float decay_ij = (j <= i) ? __expf(gcs_s[i] - gcs_s[j]) : 0.f;
        const float* ki = conv_out + (size_t)(t0 + i) * qkv_stride + k_off + hk * S;
        const float* kj = conv_out + (size_t)(t0 + j) * qkv_stride + k_off + hk * S;
        const float* qi = conv_out + (size_t)(t0 + i) * qkv_stride + q_off + hk * S;
        float dot_kb = 0.f, dot_kq = 0.f;
        for (int c = 0; c < S; ++c) {
            const float kjc = kj[c];
            dot_kb += ki[c] * kjc;
            dot_kq += qi[c] * kjc;
        }
        const float b_i = beta[(size_t)(t0 + i) * dt_stride_b + hv];
        T_s[i][j] = (j < i) ? b_i * dot_kb * decay_ij : 0.f;
        kq[i * cs + j] = (j <= i) ? dot_kq * decay_ij : 0.f;  // tri-DIAG
    }
    __syncthreads();

    // ── 3. A = M⁻¹·(−T), M = I + T (unit-lower-tri). Sustitución por
    //       columna: UN thread por columna col (cs ≤ 64 ≤ 256 thr).
    //       In-place en A_s; T_s intacto:
    //         i=col:  A[col][col] = −T[col][col] = 0
    //         i>col:  A[i][col] = −T[i][col] − Σ_{j=col..i−1} T[i][j]·A[j][col]
    //       (+I después: A[i][i] += 1 → uso el CASO j==col anidado con el
    //       término −T[i][col] absorbido: empiezo x=−T[i][col] y sumo desde
    //       j=col — A[col][col] es 0, correcto).
    for (int col = tid; col < cs; col += nthr) {
        for (int i = col; i < cs; ++i) {
            float x = -T_s[i][col];
            for (int j = col; j < i; ++j)
                x -= T_s[i][j] * A_s[j][col];
            A_s[i][col] = x;
        }
    }
    __syncthreads();
    for (int idx = tid; idx < cs * cs; idx += nthr) {
        const int i = idx / cs;
        const int j = idx % cs;
        // +I en el PROPIO A_s compartido: avb/k_cd (pasos 4b/5) leen A_s y
        // necesitan A=X+I (bug de paridad state ~2e-3: sin el término propio
        // i==t, v_new pierde el own-delta β_t·(v_t−exp(g_cs_t)·k_t·S)).
        // 1.4 (lane-f) E2E-divergence FIX: el solve tri escribe SOLO el
        // triángulo i≥col — el upper de A_s queda con RESTOS de smem de
        // kernels previos (en test aislado la smem virgen≈0 lo ocultaba;
        // en E2E conv1d/qgemm dejan ~1e3 ⇒ avb/k_cd leen basura ⇒ out
        // ~1e13, sumabs post-K1 3.28e13). A=X+I es 0 en upper-stricto:
        // escribirlo EXPLÍCITO.
        if (i == j) A_s[i][j] += 1.f;
        else if (i < j) A_s[i][j] = 0.f;
        A[i * cs + j] = A_s[i][j];
    }
    __syncthreads();

    // ── 4. kg/q_g (elementwise, dots de longitud 1) + v_b·β → avb = A·v_b.
    //       avb[t][s] = Σ_j A[t][j]·v[j][s]·β_j — un thread por (t,s):
    //       64×128 = 8192 elems / 256 thr = 32 iters de dots cs≤64.
    for (int idx = tid; idx < cs * S; idx += nthr) {
        const int i = idx / S;
        const int c = idx % S;
        const float gi = __expf(gcs_s[i]);
        kg[i * S + c] = conv_out[(size_t)(t0 + i) * qkv_stride + k_off + hk * S + c] * __expf(gcs_s[cs - 1] - gcs_s[i]);
        q_g[i * S + c] = conv_out[(size_t)(t0 + i) * qkv_stride + q_off + hk * S + c] * gi;
    }
    __syncthreads();

    for (int idx = tid; idx < cs * S; idx += nthr) {
        const int t = idx / S;
        const int s = idx % S;
        float acc = 0.f;
        for (int j = 0; j < cs; ++j) {
            const float vj = conv_out[(size_t)(t0 + j) * qkv_stride + v_off + hv * S + s];
            const float bj = beta[(size_t)(t0 + j) * dt_stride_b + hv];
            acc += A_s[t][j] * vj * bj;
        }
        avb[t * S + s] = acc;
    }
    __syncthreads();

    // ── 5. k_cd[c][j] = Σ_i A[j][i]·k[i][c]·β_i·exp(g_cs[i]) — un thread
    //       por (c,j): 128×64 = 8192 / 256 = 32 iters, dot cs≤64.
    //       Reusa kq·decay ya hecho? No — k_cd necesita β·exp(gcs)·A.
    for (int idx = tid; idx < S * cs; idx += nthr) {
        const int c = idx / cs;
        const int j = idx % cs;
        float acc = 0.f;
        for (int i = 0; i < cs; ++i) {
            const float ki_c = conv_out[(size_t)(t0 + i) * qkv_stride + k_off + hk * S + c];
            const float b_i = beta[(size_t)(t0 + i) * dt_stride_b + hv];
            acc += A_s[j][i] * ki_c * b_i * __expf(gcs_s[i]);
        }
        k_cd[c * cs + j] = acc;
    }
    // g_cs a global (K2 no lo recomputa).
    for (int i = tid; i < cs; i += nthr) g_cs[i] = gcs_s[i];
}

// K2: un block por (columna s del estado, v-head). El estado S[c][s] vive
// en smem (128 f32); v_new[t] en smem (64); loop secuencial de chunks.
extern "C" __global__ void prefillWYState(
    const float* __restrict__ conv_out, int q_off, int k_off, int v_off, int qkv_stride,
    const float* __restrict__ beta, int dt_stride_b,
    float* __restrict__ state, float* __restrict__ attn_out, int d_inner, float scale,
    int n_v_heads, int n_k_heads, int head_v_dim, int n_tokens, int n_chunks,
    const float* __restrict__ wy_attn, const float* __restrict__ wy_kq,
    const float* __restrict__ wy_kg, const float* __restrict__ wy_q_g,
    const float* __restrict__ wy_k_cd, const float* __restrict__ wy_avb,
    const float* __restrict__ wy_g_last)
{
    const int s = blockIdx.x;          // columna del estado [0, S)
    const int hv = blockIdx.y;
    if (s >= head_v_dim || hv >= n_v_heads) return;
    const int S = head_v_dim;
    const int tid = threadIdx.x;       // S threads (128)

    // Estado de esta (hv, s): columna s = S[0..S)[c][s] para todo c.
    float* head_state = state + (size_t)hv * S * S;
    __shared__ float Scol[WY_MAX_S];
    for (int c = tid; c < S; c += blockDim.x)
        Scol[c] = head_state[(size_t)c * S + s];
    __shared__ float v_new[WY_CS];

    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        const int t0 = chunk * WY_CS;
        const int cs = min(WY_CS, n_tokens - t0);
        if (cs <= 0) break;

        const size_t base2 = (size_t)chunk * n_v_heads + hv;
        const float* kq   = wy_kq + base2 * WY_CS * WY_CS;
        const float* kg   = wy_kg + base2 * WY_CS * S;
        const float* q_g  = wy_q_g + base2 * WY_CS * S;
        const float* k_cd = wy_k_cd + base2 * S * WY_CS;
        const float* avb  = wy_avb + base2 * WY_CS * S;
        const float g_last_exp = wy_g_last[base2];

        // ── a) v_new[t] = avb[t][s] − Σ_c k_cd[c][t]·Scol[c]  (split-c:
        //       cada thread acumula S/threads c's; butterfly al final).
        for (int t = tid; t < cs; t += blockDim.x) {
            float vp = 0.f;
            for (int c = 0; c < S; ++c)
                vp += k_cd[(size_t)c * cs + t] * Scol[c];
            v_new[t] = avb[t * S + s] - vp;
        }
        __syncthreads();

        // ── b) o[t][s] = (Σ_c Scol[c]·q_g[t][c] + Σ_{j≤t} v_new[j]·kq[t][j])·scale
        //       Cada thread UN t: dot S=128 serial (128 FMA — trivial)
        //       + kq-dot serial (≤t+1 términos).
        for (int t = tid; t < cs; t += blockDim.x) {
            const float* q_g_t = q_g + (size_t)t * S;
            const float* kq_t = kq + (size_t)t * cs;
            float o = 0.f;
            for (int c = 0; c < S; ++c) o += Scol[c] * q_g_t[c];
            float kq_sum = 0.f;
            for (int j = 0; j <= t; ++j)
                kq_sum += v_new[j] * kq_t[j];
            attn_out[(size_t)(t0 + t) * d_inner + hv * S + s] = (o + kq_sum) * scale;
        }
        __syncthreads();

        // ── c) Scol[c] = Scol[c]·exp(g_last) + Σ_t kg[t][c]·v_new[t]
        //       thread c: 64 FMAs (kg[t][c] — stride S entre t's: NO
        //       coalesced pero es [t][c] con c fijo — L2-friendly).
        for (int c = tid; c < S; c += blockDim.x) {
            float a = Scol[c] * g_last_exp;
            for (int t = 0; t < cs; ++t)
                a += kg[(size_t)t * S + c] * v_new[t];
            Scol[c] = a;
        }
        __syncthreads();
    }

    // Estado de vuelta a global.
    for (int c = tid; c < S; c += blockDim.x)
        head_state[(size_t)c * S + s] = Scol[c];
}
