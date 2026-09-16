// Elementwise / normalization CUDA kernels for the GPU-resident hybrid layer.
// Compiled by the build into a cubin and launched from layer_kernels.zig via the
// CUDA driver API (same pattern as paged_attention.cu).
//
// Convention: all pointers are device pointers. Kernels do NOT sync; the caller
// synchronizes the stream once per token.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <mma.h>
#include "error_flag.cuh" // lane-cuda UC-2.3: zaSetError/zaCheckFloat (device-only)

#define WARP 32

// WMMA tile dimensions for INT8 (sm_80+)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// ─── RMSNorm: out = x * gamma / sqrt(mean(x^2) + eps), per row ────────────────
// x, gamma, out: [rows, n]. Launched grid=(rows), block=min(n, 256).
extern "C" __global__ void rmsNormKernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    float* __restrict__ out,
    int n, float eps)
{
    int row = blockIdx.x;
    const float* xr = x + (size_t)row * n;
    float* or_ = out + (size_t)row * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        ss += v * v;
    }
    // reduction within block
    __shared__ float reds[256];
    reds[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) reds[threadIdx.x] += reds[threadIdx.x + s];
        __syncthreads();
    }
    float inv = 1.0f / sqrtf(reds[0] / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        or_[i] = xr[i] * inv * gamma[i];
    }
}

// ─── Elementwise add: out = a + b (or out = a, with b) ────────────────────────
extern "C" __global__ void addKernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
extern "C" __global__ void addInplaceKernel(
    float* __restrict__ a, const float* __restrict__ b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

// ─── Mul element-wise: out[i] = a[i] * b[i] ───────────────────────────────────
// 8.3 (LFM2.5): gating GDN del ShortConv real (b·x y c·conv_out).
extern "C" __global__ void mulKernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

// ─── ViT kernels (PLAN_MMPROJ 10.2 device-resident, lane-mmproj) ────────────

// LayerNorm con gamma/beta (ViT Qwen2/3-VL — clip_block.layerNormSlice GPU).
// rows = n_pos (1 block por token), n = n_embd.
extern "C" __global__ void layerNormKernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int n, float eps)
{
    int row = blockIdx.x;
    const float* xr = x + (size_t)row * n;
    float* or_ = out + (size_t)row * n;
    float sum = 0.0f, ssq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v;
        ssq += v * v;
    }
    __shared__ float reds[2][256];
    reds[0][threadIdx.x] = sum;
    reds[1][threadIdx.x] = ssq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            reds[0][threadIdx.x] += reds[0][threadIdx.x + s];
            reds[1][threadIdx.x] += reds[1][threadIdx.x + s];
        }
        __syncthreads();
    }
    const float mean = reds[0][0] / n;
    const float var_ = reds[1][0] / n - mean * mean;
    const float inv_std = rsqrtf(var_ + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float b = beta ? beta[i] : 0.0f;
        or_[i] = (xr[i] - mean) * inv_std * gamma[i] + b;
    }
}

// GELU tanh standalone (ViT FFN; misma fórmula que clip_block .gelu path).
extern "C" __global__ void geluKernel(
    const float* __restrict__ x,
    float* __restrict__ out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float c = v * v * v;
        float inner = 0.7978845608f * (v + 0.044715f * c);
        out[i] = v * 0.5f * (1.0f + tanhf(inner));
    }
}

// Bias ADD in-place: x[i] += b[i % b_len] (proyecciones QKV/FFN con bias
// broadcast por fila — b_len = out_dim).
extern "C" __global__ void biasAddKernel(
    float* __restrict__ x,
    const float* __restrict__ b,
    int n, int b_len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += b[i % b_len];
}

// Pack Q/K/V por head (PLAN_MMPROJ 10.2): extrae el head h del tensor
// interleaved [n_pos, n_head, hd] (qkv proyectado) al buffer head-major
// [n_head][n_pos·hd] (el que consume vitAttnHeadKernel).
// src: base del segmento (q/k/v) en el qkv [n_pos·n_head·hd interleaved]
// dst: [n_head][n_pos·hd]
extern "C" __global__ void packHeadKernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n_pos, int n_head, int hd, int src_row_stride)
{
    // Un thread por elemento: idx global sobre dst [n_head·n_pos·hd]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_head * n_pos * hd;
    if (idx >= total) return;
    int d = idx % hd;
    int rest = idx / hd;
    int p = rest % n_pos;
    int h = rest / n_pos;
    // src row-interleaved [p][src_row_stride elems] con el head-slice del
    // bloque (Q/K/V) YA en el offset base del caller: el elemento (p,h,d)
    // vive en p·src_row_stride + h·hd + d. Para el QKV interleaved
    // [p][3·n_embd], Q base=0, K base=n_embd, V base=2·n_embd (el caller
    // los pasa con +seg_dev — el stride INTERNO del bloque es n_embd).
    // 10.2-bisect: el stride por fila era n_head·hd (=n_embd) hardcoded,
    // VÁLIDO sólo para el bloque Q (offset 0). K/V leían filas de Q.
    dst[idx] = src[(size_t)p * src_row_stride + h * hd + d];
}

// Atención ViT por (query, head): un block por query — TODOS los threads
// colaboran en cada dot(Q,K[j]) (paralelismo intra-dot), softmax bidireccional
// completo en registers/shared, luego O = Σ softmax·V acumulado por threads.
// Simplito pero suficiente: hd ≤ 128, cada thread maneja d = threadIdx.x si
// blockDim = hd (llamamos con 128 threads y guard d < hd).
// q/k/v/o: [n_pos, hd] contiguos DEL HEAD (pack por head ya hecho en host).
// Unpack por head (inverso del packHeadKernel): head-major [n_head][n_pos·hd]
// → interleaved [n_pos, n_head·hd] (lo que consume la O-proj).
extern "C" __global__ void unpackHeadKernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n_pos, int n_head, int hd)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_head * n_pos * hd;
    if (idx >= total) return;
    int d = idx % hd;
    int rest = idx / hd;
    int p = rest % n_pos;
    int h = rest / n_pos;
    dst[(size_t)p * n_head * hd + h * hd + d] = src[idx];
}

extern "C" __global__ void vitAttnHeadKernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ o,
    int n_pos, int hd, float kq_scale)
{
    int qi = blockIdx.x;
    if (qi >= n_pos) return;
    const int d = threadIdx.x; // d < hd (blockDim >= hd; guard abajo)
    const bool active = d < hd;
    const float* qrow = q + (size_t)qi * hd;

    // scores en shared: cada thread posee la fila de scores para UN d-slice?
    // NO: computamos score[j] con reducción por block (dot completo).
    // Para evitar O(n_pos) shared por block, hacemos 2 pasadas:
    //   1) max y sum por j (dot reducido, j secuencial — todos los threads
    //      en el dot)
    //   2) acumulación O con p_j = exp(s_j - max)/sum
    __shared__ float s_dot[128];

    // Pass 1a: max
    float m = -FLT_MAX; // lane-cuda: INFINITY indefinido bajo NVRTC (sin __STDC_VERSION__); -FLT_MAX es init equivalente
    for (int j = 0; j < n_pos; j++) {
        float partial = 0.0f;
        if (active) {
            const float* krow = k + (size_t)j * hd;
            partial = qrow[d] * krow[d];
        }
        s_dot[d] = partial;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (d < s) s_dot[d] += s_dot[d + s];
            __syncthreads();
        }
        float dot = s_dot[0] * kq_scale;
        __syncthreads();
        if (d == 0) m = fmaxf(m, dot);
    }
    __shared__ float s_max;
    if (d == 0) s_max = m;
    __syncthreads();
    m = s_max;

    // Pass 1b: sum de exp y acumulación O (una sola pasada más: p_j y acc)
    float l = 0.0f;
    float acc = 0.0f; // cada thread acumula SU dim d
    for (int j = 0; j < n_pos; j++) {
        float partial = 0.0f;
        if (active) {
            const float* krow = k + (size_t)j * hd;
            partial = qrow[d] * krow[d];
        }
        s_dot[d] = partial;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (d < s) s_dot[d] += s_dot[d + s];
            __syncthreads();
        }
        float p = __expf(s_dot[0] * kq_scale - m);
        __syncthreads();
        l += p;
        if (active) {
            const float* vrow = v + (size_t)j * hd;
            acc += p * vrow[d];
        }
    }
    __shared__ float s_l;
    if (d == 0) s_l = l;
    __syncthreads();
    if (active) {
        o[(size_t)qi * hd + d] = acc / (s_l + 1e-30f);
    }
}

// ─── SwiGLU (in-place on gate): gate = silu(gate) * up ────────────────────────
extern "C" __global__ void swigluKernel(
    float* __restrict__ gate, const float* __restrict__ up, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        gate[i] = g / (1.0f + expf(-g)) * up[i];
    }
}

// ─── sigmoid in place ─────────────────────────────────────────────────────────
extern "C" __global__ void sigmoidKernel(float* __restrict__ x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 1.0f / (1.0f + expf(-x[i]));
}

// ─── gate = ssm_a * softplus(x + dt_bias)  [per v-head column] ────────────────
// x: [N, dt_rank], dt_bias/ssm_a: [dt_rank]. gate[h] = ssm_a[h]*softplus(x[h]+dt_bias[h]).
extern "C" __global__ void gateComputeKernel(
    float* __restrict__ x,
    const float* __restrict__ dt_bias,
    const float* __restrict__ ssm_a,
    int n, int dt_rank)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int h = i % dt_rank;
        float v = x[i] + dt_bias[h];
        x[i] = ssm_a[h] * logf(1.0f + expf(v));
    }
}

// ─── fused sigmoid(beta) + gateCompute(gate) in one launch ───────────────────
// beta and gate are both [N, dt_rank]; the per-column h = i % dt_rank.
extern "C" __global__ void sigmoidGateKernel(
    float* __restrict__ beta,
    float* __restrict__ gate,
    const float* __restrict__ dt_bias,
    const float* __restrict__ ssm_a,
    int n, int dt_rank)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        beta[i] = 1.0f / (1.0f + expf(-beta[i]));
        int h = i % dt_rank;
        float v = gate[i] + dt_bias[h];
        gate[i] = ssm_a[h] * logf(1.0f + expf(v));
    }
}

// ─── fused beta=x·W_beta, alpha=x·W_alpha, sigmoid(beta), gateCompute ────────
// One block per (h, n): computes both projections of the same x row, applies
// sigmoid(beta) and gate = ssm_a*softplus(alpha + dt_bias), and writes the two
// [N, dt_rank] outputs. Replaces 2 cublas SGEMM + sigmoidGate.
extern "C" __global__ void sigmoidGateProjKernel(
    const float* __restrict__ x,
    const float* __restrict__ w_beta,
    const float* __restrict__ w_alpha,
    const float* __restrict__ dt_bias,
    const float* __restrict__ ssm_a,
    float* __restrict__ beta,
    float* __restrict__ gate,
    int N, int K, int dt_rank)
{
    const int h = blockIdx.x;
    const int n = blockIdx.y;
    if (h >= dt_rank || n >= N) return;
    extern __shared__ float sx[];
    const float* xrow = x + (size_t)n * K;
    for (int i = threadIdx.x; i < K; i += blockDim.x) sx[i] = xrow[i];
    __syncthreads();
    const float* wb = w_beta + (size_t)h * K;
    const float* wa = w_alpha + (size_t)h * K;
    float s_b = 0.0f, s_a = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        s_b += sx[i] * wb[i];
        s_a += sx[i] * wa[i];
    }
    __shared__ float rb[256], ra[256];
    rb[threadIdx.x] = s_b;
    ra[threadIdx.x] = s_a;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) {
            rb[threadIdx.x] += rb[threadIdx.x + o];
            ra[threadIdx.x] += ra[threadIdx.x + o];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const int off = n * dt_rank + h;
        beta[off] = 1.0f / (1.0f + expf(-rb[0]));
        gate[off] = ssm_a[h] * logf(1.0f + expf(ra[0] + dt_bias[h]));
    }
}

// ─── L2-normalize Q and K heads in conv_out (fiel a FLA l2norm.py:69) ────────
// conv_out: [N, qkv_dim]; Q in [0,key_dim), K in [key_dim, 2*key_dim).
// scale = 1 / sqrt(sum x^2 + eps) per head (eps DENTRO de la raíz — 1.14).
extern "C" __global__ void l2NormHeadsKernel(
    float* __restrict__ conv_out,
    int N, int qkv_dim, int key_dim, int n_k_heads, int head_v_dim, float eps)
{
    int t = blockIdx.y;
    int h = blockIdx.x;
    if (t >= N || h >= n_k_heads) return;
    for (int part = 0; part < 2; ++part) {
        int base = (int)((size_t)t * qkv_dim) + (part == 0 ? h * head_v_dim : (key_dim + h * head_v_dim));
        float ss = 0.0f;
        for (int i = threadIdx.x; i < head_v_dim; i += blockDim.x)
            ss += conv_out[base + i] * conv_out[base + i];
        __shared__ float reds[256];
        reds[threadIdx.x] = ss;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) reds[threadIdx.x] += reds[threadIdx.x + s];
            __syncthreads();
        }
        float scale = 1.0f / sqrtf(reds[0] + eps);
        for (int i = threadIdx.x; i < head_v_dim; i += blockDim.x)
            conv_out[base + i] *= scale;
    }
}

// ─── conv1d (causal) + silu on qkv → conv_out, with direct state access ──────
// Reads conv_state [(d_conv-1), qkv_dim] and qkv [N, qkv_dim] as the combined
// input rows [(d_conv-1)+N, qkv_dim] (state first), computes causal conv + silu
// into conv_out [N, qkv_dim], and writes the shifted state (new_state[i] =
// input row N+i) into state_out. state_out must NOT alias conv_state or qkv.
// ─── §5.1 FUSED GDN: l2 + ΔNet(registros) + rmsNorm·silu(z) — 3→1 ────────────
// Port faithful de unsloth gated_delta_net.cu (§3.1 del study) + la semántica
// de l2NormHeads/rmsNormGateMul de zig-ai. UN bloque por (v-head, token):
// grid (n_v_heads, N, 1), block (32, 4, 1) = 128 threads; cada warp es dueño
// de una COLUMNA j del estado (j = warp 0..3 del bloque cubre 4 cols vía
// blockIdx.z*4 — NO: aquí un bloque = UNA head completa ⇒ 128 cols / 4 warps
// = 32 cols por warp… para S_v=128 se necesitan 32 warps. AJUSTE: block
// (32, 32, 1) = 1024 threads = 32 warps por head ⇒ col = threadIdx.y.
// S-state (128×128 f32 = 64 KB) distribuido: cada lane guarda
// rows_per_lane = S_v/32 = 4 floats EN REGISTROS (cero tráfico DRAM del
// estado en la recurrencia; el clásico lo tocaba 5×, §5.6 2×, esto 0×).
//
// Etapas (barriers SÓLO entre etapas — la recurrencia hot-loop no tiene):
//   A) l2 K/Q head slices de conv_out (in-place) — patrón strided +
//      árbol shared EXACTO de l2NormHeadsKernel ⇒ bit-idéntico.
//      Solo lo hace el bloque de la head hk correspondiente (los bloques de
//      heads v!=hk con el mismo hk lo saltan — guard).
//   B) ΔNet warp-per-column (semántica EXACTA de deltaNetWarpKernel §5.6:
//      sk = g·S₀ᵀk broadcast, δ = β(v − g·sk), S = g·S + k⊗δ, o = Sᵀq·scale).
//   C) rmsNormGateMul sobre attn_out[hv·dim..] × silu(z) — patrón strided +
//      árbol EXACTO de rmsNormGateMulKernel ⇒ bit-idéntico.
//
// REQUISITO: head_v_dim % 32 == 0 y S_v <= 128 (regs). S_v=128 ⇒ 4 f32/lane.
// OPT-IN: DNFUSED=1 (ssm.zig). Fallback: kernels separados (§5.6/§4.2).
// ─── §5.1 FUSED GDN: l2 + ΔNet(S en registros) + rmsNorm·silu(z) — 3→1 ──────
// Port faithful de unsloth gated_delta_net.cu (study §3.1) + semántica
// l2NormHeads/rmsNormGateMul de zig-ai. UN BLOQUE POR (v-head, token):
// grid (n_v_heads, N, 1); block (32, 32, 1) = 1024 threads = 32 warps.
// Cada warp posee cpw = dim/32 columnas (S_v=128 ⇒ 4); S-state por columna
// distribuido en registros: rows_per_lane = dim/32 = 4 floats/lane/columna
// ⇒ 16 regs de estado por lane (64 KB cero DRAM durante la recurrencia;
// 1 lectura + 1 escritura de persistencia — igual que unsloth).
//
// Etapas (barriers SÓLO entre etapas; la recurrencia per-columna es
// barrier-free dentro del warp):
//   A) l2 K/Q slices de la head hk IN-PLACE — la replica EXACTAMENTE el
//      warp 0 (threadIdx.y==0, tid==threadIdx.x ∈ [0,32))… NO: para
//      bit-paridad con l2NormHeadsKernel (blockDim=256, reds[256]) la
//      acumulación strided la hacen los primeros 256 lanes del bloque
//      (flat tid < 256), con el MISMO patrón i=tid; i<dim; i+=256 y el
//      MISMO árbol reds[256]. El resto espera en el barrier.
//   B) ΔNet: por columna c (del warp), per-lane strided rows + shfl tree
//      + broadcast — semántica EXACTA de deltaNetWarpKernel §5.6
//      (sk=g·S₀ᵀk, δ=β(v−g·sk), S=g·S+k⊗δ, o=Sᵀq·scale).
//   C) rmsNorm·silu(z) sobre attn_out[hv·dim..] — patrón strided + árbol
//      reds[256] EXACTO de rmsNormGateMulKernel ⇒ bit-idéntico.
//
// REQUISITOS: head_v_dim % 32 == 0; dim <= 128 (16 regs estado/lane);
// n_v_heads % n_k_heads == 0. OPT-IN: DNFUSED=1 (ssm.zig).
// ─── §5.1 FUSED GDN: l2 + ΔNet(S en registros) + rmsNorm·silu(z) — 3→1 ──────
// Port faithful de unsloth gated_delta_net.cu (study §3.1) + semántica
// l2NormHeads/rmsNormGateMul de zig-ai. UN BLOQUE POR (v-head, token):
// grid (n_v_heads, N, 1); block (32, 32, 1) = 1024 threads = 32 warps.
// Cada warp posee cpw = dim/32 columnas (S_v=128 ⇒ 4); el S-state por
// columna vive en registros: rows_per_lane = dim/32 = 4 f32/lane/columna.
//
// PARIDAD BARRIER: l2 (etapa A) y rmsNorm (etapa C) replican el patrón
// strided + árbol del kernel original (blockDim=256, reds[256]): los
// primeros 256 lanes hacen la acumulación, el árbol lo recorren TODOS
// (barrier UNIFORME fuera de los guards — mismo nº de __syncthreads()
// para owner y no-owner de l2; los no-owners acumulan 0 y su escala
// no se aplica: idempotente).
//
//   A) l2 K/Q de la head hk IN-PLACE — la hace el bloque owner (primer
//      v-head de la k-head); los demás pasan por los mismos barriers.
//   B) ΔNet por columna: shfl tree + broadcast — semántica §5.6 exacta.
//   C) rmsNorm·silu(z) sobre la slice de ESTA v-head — patrón rmsNormGateMul.
//
    // REQUISITOS: head_v_dim % 32 == 0; dim <= 128; n_v_heads % n_k_heads == 0.
    // OPT-IN: DNFUSED=1 (ssm.zig). Fallback: kernels separados.
    extern "C" __global__ __launch_bounds__(1024, 2) void deltaNetFusedKernel(
        float* __restrict__ conv_out,         // [N, qkv_dim] — l2 K/Q IN-PLACE
        const float* __restrict__ gate,       // [N, dt_rank]
        const float* __restrict__ beta,       // [N, dt_rank]
        const float* __restrict__ z,          // [N, n_v_heads*dim]
        const float* __restrict__ ssm_norm,   // [n_v_heads*dim]
        float* __restrict__ attn_out,          // [N, n_v_heads*dim] — OUT
        float* __restrict__ state,             // [n_v_heads, dim*dim] persistente
        int N, int qkv_dim, int key_dim, int n_k_heads, int n_v_heads,
        int head_v_dim, int dt_rank, float eps)
    {
        const int t  = blockIdx.y;
        const int hv = blockIdx.x;
        if (t >= N || hv >= n_v_heads) return;
        const int dim = head_v_dim;
        const int hk = hv % n_k_heads;
        const int warp = threadIdx.y;                    // 0..31
        const int lane = threadIdx.x;                    // 0..31
        const int flat_tid = warp * blockDim.x + lane;   // 0..1023
        const int cpw = dim / (int)blockDim.y;           // columnas por warp (4)
        const int rows_per_lane = (dim + blockDim.x - 1) / blockDim.x; // 4

        const float g = expf(gate[(size_t)t * dt_rank + hv]);
        const float b = beta[(size_t)t * dt_rank + hv];
        const float scale = 1.0f / sqrtf((float)dim);
        const int q_base = (int)((size_t)t * qkv_dim) + hk * dim;
        const int k_base = (int)((size_t)t * qkv_dim) + key_dim + hk * dim;
        const int v_base = (int)((size_t)t * qkv_dim) + 2 * key_dim + hv * dim;
        const int s_base = hv * dim * dim;
        const int hd_base = (int)((size_t)t * (n_v_heads * dim)) + hv * dim;

        // Shared memory for 256-thread reduction (matching l2NormHeadsKernel/rmsNormGateMulKernel)
        __shared__ float reds[256];
        // ══ ETAPA A: l2 K/Q IN-PLACE — 256-thread tree reduction (BIT-IDENTICAL) ════
        // Dueño de l2: el primer v-head de cada k-head (hv % ratio == 0)
        const int ratio = n_v_heads / n_k_heads;
        const bool l2_owner = (hv % ratio == 0);
        for (int part = 0; part < 2; ++part) {
            const int base = (int)((size_t)t * qkv_dim) + (part == 0 ? hk * dim : (key_dim + hk * dim));
            // Phase 1: first 256 threads accumulate partial sum (matching l2NormHeadsKernel)
            float ss = 0.0f;
            if (l2_owner && flat_tid < 256) {
                for (int i = flat_tid; i < dim; i += 256)
                    ss += conv_out[base + i] * conv_out[base + i];
            }
            // Phase 2: 256-thread tree reduction (EXACT same as l2NormHeadsKernel)
            if (flat_tid < 256) reds[flat_tid] = ss;
            __syncthreads();
            for (int s = 128; s > 0; s >>= 1) {
                if (flat_tid < s) reds[flat_tid] += reds[flat_tid + s];
                __syncthreads();
            }
            const float l2_scale = 1.0f / sqrtf(reds[0] + eps);
            if (l2_owner && flat_tid < 256) {
                for (int i = flat_tid; i < dim; i += 256)
                    conv_out[base + i] *= l2_scale;
            }
            __syncthreads(); // ensure writes visible before next part
        }
        // Total Stage A barriers: 2 parts × 2 = 4 (matching l2NormHeadsKernel)

    // ══ ETAPA B: ΔNet — por columna del warp, S en registros (§5.6 exacta) ═
    // Unchanged: already barrier-free within warp (shuffles only).
    const int c0 = warp * cpw;
    #pragma unroll
    for (int cc = 0; cc < cpw; ++cc) {
        const int c = c0 + cc;
        float s_shard[8];
        #pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            const int i = lane + r * blockDim.x;
            s_shard[r] = state[s_base + (size_t)i * dim + c];
        }
        float sk = 0.0f;
        #pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            const int i = lane + r * blockDim.x;
            sk += s_shard[r] * conv_out[k_base + i];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            sk += __shfl_down_sync(0xffffffffu, sk, off);
        sk = __shfl_sync(0xffffffffu, sk, 0);
        const float d = b * (conv_out[v_base + c] - g * sk);
        float attn_partial = 0.0f;
        #pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            const int i = lane + r * blockDim.x;
            s_shard[r] = g * s_shard[r] + conv_out[k_base + i] * d;
            attn_partial += s_shard[r] * conv_out[q_base + i];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            attn_partial += __shfl_down_sync(0xffffffffu, attn_partial, off);
        attn_partial = __shfl_sync(0xffffffffu, attn_partial, 0);
        #pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            const int i = lane + r * blockDim.x;
            state[s_base + (size_t)i * dim + c] = s_shard[r];
        }
        if (lane == 0)
            attn_out[hd_base + c] = attn_partial * scale;
    }

    __syncthreads(); // attn_out de la head completo antes de la norma. (1 barrier)

    // ══ ETAPA C: rmsNorm·silu(z) — 256-thread tree reduction (BIT-IDENTICAL) ═══
    // Matches rmsNormGateMulKernel exactly: first 256 threads do reduction
    {
        float ss = 0.0f;
        if (flat_tid < 256) {
            for (int i = flat_tid; i < dim; i += 256) {
                const float v = attn_out[hd_base + i];
                ss += v * v;
            }
        }
        // 256-thread tree reduction (EXACT same as rmsNormGateMulKernel)
        if (flat_tid < 256) reds[flat_tid] = ss;
        __syncthreads();
        for (int s = 128; s > 0; s >>= 1) {
            if (flat_tid < s) reds[flat_tid] += reds[flat_tid + s];
            __syncthreads();
        }
        const float rscale = 1.0f / sqrtf(reds[0] / (float)dim + eps);
        if (flat_tid < 256) {
            for (int i = flat_tid; i < dim; i += 256) {
                const float zn = z[hd_base + i];
                const float silu = zn / (1.0f + expf(-zn));
                attn_out[hd_base + i] = attn_out[hd_base + i] * rscale * ssm_norm[i] * silu;
            }
        }
    }
    // Total Stage C barriers: 2 (matching rmsNormGateMulKernel)
    // GRAND TOTAL: 4 + 1 + 2 = 7 barriers (was 27)
}



extern "C" __global__ void conv1dSiluKernel(    const float* __restrict__ conv_state,
    const float* __restrict__ qkv,
    const float* __restrict__ conv_w,
    float* __restrict__ conv_out,
    float* __restrict__ state_out,
    int N, int qkv_dim, int d_conv)
{
    int rows = (N > d_conv - 1 ? N : d_conv - 1);
    int total = rows * qkv_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int t = idx / qkv_dim;
    int c = idx % qkv_dim;
    if (t < N) {
        float sum = 0.0f;
        for (int k = 0; k < d_conv; ++k) {
            int r = t + k;
            int rr = r < (d_conv - 1) ? r : r - (d_conv - 1);
            const float* src = (r < d_conv - 1) ? conv_state : qkv;
            sum += src[(size_t)rr * qkv_dim + c] * conv_w[(size_t)c * d_conv + k];
        }
        conv_out[(size_t)t * qkv_dim + c] = sum / (1.0f + expf(-sum));
    }
    if (t < d_conv - 1) {
        int ri = N + t;
        int srr = ri < (d_conv - 1) ? ri : ri - (d_conv - 1);
        const float* ssrc = (ri < d_conv - 1) ? conv_state : qkv;
        state_out[(size_t)t * qkv_dim + c] = ssrc[(size_t)srr * qkv_dim + c];
    }
}

// ─── Conv1d lineal (SIN silu) — 8.3 LFM2: ggml_ssm_conv es lineal ==========
// Espejo exacto del conv1dSiluKernel sin el silu final. El LFM2 ShortConv
// usa conv lineal pura (lfm2.cpp ssm_conv); el silu era un bug del zig-ai.
// ─── GDN gate mul con layout [T, 3*D] interleaved (8.3 fix) ──────────────────
// in_proj GEMM produce row-major [T, 3*D]: por token t, b/c/x son columnas
// interleaved (b: t*3D + [0,D), c: [D,2D), x: [2D,3D)). El split anterior
// asumía bloques GLOBALES [3][T][D] (punteros planos) — solo correcto si el
// GEMM hubiera producido [3D, T] col-major. Este kernel hace el mul per-token
// con los offsets correctos: out[t*D + c] = src[t*3D + off_a*D + c] *
// src[t*3D + off_b*D + c].
// mode 0: out = b·x  (off_a=0, off_b=2)
// mode 1: out = c·conv_out (src2 = conv_out [T,D] separado)
extern "C" __global__ void gdnGateKernel(
    const float* __restrict__ src,
    const float* __restrict__ src2, // mode 1: conv_out [T, D]; mode 0: unused
    float* __restrict__ out,
    int T, int D, int mode)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * D;
    if (i >= total) return;
    int t = i / D;
    int c = i % D;
    if (mode == 0) {
        out[i] = src[t * 3 * D + c] * src[t * 3 * D + 2 * D + c]; // b · x
    } else {
        out[i] = src[t * 3 * D + D + c] * src2[i]; // c · conv_out
    }
}

extern "C" __global__ void conv1dLinearKernel(
    const float* __restrict__ conv_state,
    const float* __restrict__ input,
    const float* __restrict__ conv_w,
    float* __restrict__ conv_out,
    float* __restrict__ state_out,
    int N, int in_dim, int d_conv)
{
    int rows = (N > d_conv - 1 ? N : d_conv - 1);
    int total = rows * in_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int t = idx / in_dim;
    int c = idx % in_dim;
    if (t < N) {
        float sum = 0.0f;
        for (int k = 0; k < d_conv; ++k) {
            int r = t + k;
            int rr = r < (d_conv - 1) ? r : r - (d_conv - 1);
            const float* src = (r < d_conv - 1) ? conv_state : input;
            sum += src[(size_t)rr * in_dim + c] * conv_w[(size_t)c * d_conv + k];
        }
        conv_out[(size_t)t * in_dim + c] = sum; // LFM2: lineal, SIN silu
    }
    if (t < d_conv - 1) {
        int ri = N + t;
        int srr = ri < (d_conv - 1) ? ri : ri - (d_conv - 1);
        const float* ssrc = (ri < d_conv - 1) ? conv_state : input;
        state_out[(size_t)t * in_dim + c] = ssrc[(size_t)srr * in_dim + c];
    }
}

// ─── rmsnorm(attn_out) * silu(z) * ssm_norm, per v-head block ────────────────
// attn_out: [N, d_inner], z: [N, d_inner], ssm_norm: [head_v_dim].
extern "C" __global__ void rmsNormGateMulKernel(
    float* __restrict__ attn_out,
    const float* __restrict__ z,
    const float* __restrict__ ssm_norm,
    int N, int d_inner, int n_v_heads, int head_v_dim, float eps)
{
    int t = blockIdx.y;
    int hv = blockIdx.x;
    if (t >= N || hv >= n_v_heads) return;
    int base = (int)((size_t)t * d_inner) + hv * head_v_dim;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < head_v_dim; i += blockDim.x) {
        float v = attn_out[base + i];
        ss += v * v;
    }
    __shared__ float reds[256];
    reds[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) reds[threadIdx.x] += reds[threadIdx.x + s];
        __syncthreads();
    }
    float rscale = 1.0f / sqrtf(reds[0] / (float)head_v_dim + eps);
    for (int i = threadIdx.x; i < head_v_dim; i += blockDim.x) {
        float zn = z[base + i];
        float silu = zn / (1.0f + expf(-zn));
        attn_out[base + i] = attn_out[base + i] * rscale * ssm_norm[i] * silu;
    }
}

// ─── DeltaNet recurrence, one (t, hv) per block; state updated in place ───────
// conv_out: [N, qkv_dim]; gate/beta: [N, dt_rank]; state: [n_v_heads, hv_d, hv_d]
// q in [0,key_dim), k in [key_dim,2*key_dim), v in [2*key_dim, qkv_dim).
extern "C" __global__ void deltaNetKernel(
    const float* __restrict__ conv_out,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ attn_out,
    float* __restrict__ state,
    int N, int qkv_dim, int key_dim, int n_k_heads, int n_v_heads,
    int head_v_dim, int dt_rank, float eps)
{
    int t = blockIdx.y;
    int hv = blockIdx.x;
    if (t >= N || hv >= n_v_heads) return;
    int hk = hv % n_k_heads; // 7.1b: módulo (ggml_repeat_4d), no bloque
    int dim = head_v_dim;
    int q_base = (int)((size_t)t * qkv_dim) + hk * dim;
    int k_base = (int)((size_t)t * qkv_dim) + key_dim + hk * dim;
    int v_base = (int)((size_t)t * qkv_dim) + 2 * key_dim + hv * dim;
    int s_base = (hv * dim * dim); // state is per-v-head
    float g = expf(gate[(size_t)t * dt_rank + hv]);
    float b = beta[(size_t)t * dt_rank + hv];
    float scale = 1.0f / sqrtf((float)dim);

    // decay S *= exp(g)
    for (int i = threadIdx.x; i < dim * dim; i += blockDim.x)
        state[s_base + i] *= g;
    __syncthreads();

    // sk[j] = sum_i S[i][j]*k[i]; d[j] = b*(v[j] - sk[j])
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        float sk = 0.0f;
        for (int i = 0; i < dim; ++i)
            sk += state[s_base + i * dim + j] * conv_out[k_base + i];
        float d = b * (conv_out[v_base + j] - sk);
        for (int i = 0; i < dim; ++i)
            state[s_base + i * dim + j] += conv_out[k_base + i] * d;
    }
    __syncthreads();
    // o[j] = sum_i S[i][j]*q[i]*scale
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        float o = 0.0f;
        for (int i = 0; i < dim; ++i)
            o += state[s_base + i * dim + j] * conv_out[q_base + i];
        attn_out[(size_t)t * (n_v_heads * dim) + hv * dim + j] = o * scale;
    }
}

// ─── ΔNet warp-shuffle (STUDY §5.6) ───────────────────────────────────────────
// Una COLUMNA del estado (j) por warp: los 128 elementos de la columna viven
// distribuidos en 4 filas/lane ⇒ las reducciones S^T·k y S^T·q son árboles
// __shfl_down_sync (sin __syncthreads, sin bucles seriales de 128). El decay
// S *= exp(g) se FUSIONA en la actualización (se ahorra una pasada completa
// de lectura+escritura del estado). Cada warp es dueño exclusivo de su
// columna ⇒ no hay carras intra-block. Grid: (n_v_heads, n_seqs, dim/4);
// block: (32, 4, 1) = 128 threads — 4 columnas por bloque.
// Semántica bit-comparable al deltaNetKernel clásico (mismo orden de suma
// por lane: i = lane, lane+32, ...).
extern "C" __global__ void deltaNetWarpKernel(
    const float* __restrict__ conv_out,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ attn_out,
    float* __restrict__ state,
    int N, int qkv_dim, int key_dim, int n_k_heads, int n_v_heads,
    int head_v_dim, int dt_rank, float eps)
{
    const int t  = blockIdx.y;
    const int hv = blockIdx.x;
    if (t >= N || hv >= n_v_heads) return;
    const int dim = head_v_dim;
    const int cols_per_block = blockDim.y;              // 4
    const int j  = blockIdx.z * cols_per_block + threadIdx.y;
    if (j >= dim) return;
    // 7.1b: mapeo MÓDULO (semántica ggml_repeat_4d del oráculo): hv=1→k1.
    // Bloque (hv/ratio) solo coincide cuando n_v==n_k (0.8B) — el 9B
    // (n_v=32, n_k=16) leía Q/K equivocados en cada v-head impar.
    const int hk = hv % n_k_heads;
    const int lane = threadIdx.x;                        // 0..31
    const int rows_per_lane = (dim + blockDim.x - 1) / blockDim.x; // 4

    const int q_base = (int)((size_t)t * qkv_dim) + hk * dim;
    const int k_base = (int)((size_t)t * qkv_dim) + key_dim + hk * dim;
    const int v_base = (int)((size_t)t * qkv_dim) + 2 * key_dim + hv * dim;
    const int s_base = (hv * dim * dim);
    const float g = expf(gate[(size_t)t * dt_rank + hv]);
    const float b = beta[(size_t)t * dt_rank + hv];
    const float scale = 1.0f / sqrtf((float)dim);

    // sk = Σ_i S[i][j]·k[i] — árbol shfl sobre los dim elementos de la columna.
    // Lectura del estado PRE-decay: el factor g· se aplica aquí (semántica
    // del clásico: decay-then-sk ⇒ sk = g·S₀ᵀk; espejo de unsloth
    // gated_delta_net.cu:95-96: delta = (v − g·kv)·β).
    float sk = 0.0f;
    #pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
        const int i = lane + r * blockDim.x;
        sk += state[s_base + (size_t)i * dim + j] * conv_out[k_base + i];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sk += __shfl_down_sync(0xffffffffu, sk, off);
    // BROADCAST: el árbol down deja el total sólo en lane 0 — sin esto,
    // lanes 1..31 computan δ con sumas parciales (corrupten 31/32 de la
    // actualización). Todas las lanes necesitan el sk completo.
    sk = __shfl_sync(0xffffffffu, sk, 0);

    // δ_j = β·(v[j] − g·sk); actualización fusionada con el decay:
    // S[i][j] = g·S[i][j] + k[i]·δ  (por-lane, sin barreras: columna privada).
    const float d = b * (conv_out[v_base + j] - g * sk);
    #pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
        const int i = lane + r * blockDim.x;
        state[s_base + (size_t)i * dim + j] =
            g * state[s_base + (size_t)i * dim + j] + conv_out[k_base + i] * d;
    }

    // o_j = Σ_i S'[i][j]·q[i]·scale — mismo árbol shfl sobre el estado NUEVO.
    float o = 0.0f;
    #pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
        const int i = lane + r * blockDim.x;
        o += state[s_base + (size_t)i * dim + j] * conv_out[q_base + i];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        o += __shfl_down_sync(0xffffffffu, o, off);

    if (lane == 0)
        attn_out[(size_t)t * (n_v_heads * dim) + hv * dim + j] = o * scale;
}

// ─── f32 <-> f16 copies ──────────────────────────────────────────────────────
extern "C" __global__ void copyF32toF16Kernel(const float* __restrict__ src, half* __restrict__ dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}
extern "C" __global__ void copyF16toF32Kernel(const half* __restrict__ src, float* __restrict__ dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

// ─── Split Q|G interleaved: qg [N, n_head*head_dim*2] → q, g [N, n_head*head_dim] ──
// Q_h at base h*(2*head_dim), G_h at base+head_dim (igual que el forward CPU).
extern "C" __global__ void splitQGKernel(
    const float* __restrict__ qg,
    float* __restrict__ q,
    float* __restrict__ g,
    int N, int n_head, int head_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * n_head * head_dim;
    if (i >= total) return;
    int t = i / (n_head * head_dim);
    int rem = i % (n_head * head_dim);
    int h = rem / head_dim;
    int d = rem % head_dim;
    int src = t * (n_head * head_dim * 2) + h * (2 * head_dim) + d;
    q[i] = qg[src];
    g[i] = qg[src + head_dim];
}

// ─── MRoPE (NEOX half-split) — fiel a la referencia CPU applyRoPEMultiSection.
// Para texto los 4 position ids son iguales → theta = global_pos * scale^ic
// en todos los sectores. data: [N, n_head, head_dim] TOKEN-MAJOR; rows =
// n_head (o n_kv_head) * N; pos = row / heads (U2-fix lane-b1 — antes
// row % N asumía head-major y scrambleaba el prefill batched).
extern "C" __global__ void mropeKernel(
    float* __restrict__ data,
    const int* __restrict__ start_pos,
    int rows, int N, int head_dim, int n_rot,
    float base)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    // U2-fix (lane-b1): data llega TOKEN-MAJOR [N, heads, head_dim] del
    // caller (g_q/g_k de splitQG/qgemm: row = t*heads + h) ⇒ pos = row/heads.
    // El row % N anterior asumía head-major [heads, N, dim] — correcto sólo
    // en decode (N=1 ⇒ siempre 0); en prefill batched scrambleaba pos
    // (t*H+h)%N (confirmado: kernel_pos=9 real_pos=1 con H=8; rel 0.21
    // batched-vs-unrolled del diag U2). mropeVision NO cambia: ese sí
    // recibe pack head-major.
    const int heads = (N > 0) ? rows / N : 1;
    int pos = row / heads;
    float global_pos = (float)(*start_pos + pos);
    int half_rot = n_rot / 2;
    float scale = powf(base, -2.0f / (float)n_rot);
    float* d = data + (size_t)row * head_dim;
    float theta = global_pos;
    for (int ic = 0; ic < half_rot; ++ic) {
        float c = cosf(theta);
        float s = sinf(theta);
        float q0 = d[ic];
        float q1 = d[ic + half_rot];
        d[ic] = q0 * c - q1 * s;
        d[ic + half_rot] = q0 * s + q1 * c;
        theta *= scale;
    }
}

// ─── MRoPE con position-ids PER-TOKEN (PLAN_MMPROJ Fase B) ──────────────────
// Port CUDA de rope.zig::applyRoPEMultiSectionPosIds: 4 ids (t,h,w,e) por
// token para embeddings de imagen (mtmd-helper.cpp:142, n_pos_per_embd=4).
// pos_ids: [N][4] i32 device (row-major). Misma rotación NEOX half-split
// (pares (ic, ic+half_rot)) que mropeKernel; sólo cambian las thetas:
// sector = ic % sect_dims elige entre t/h/w/e, y cada theta escala
// independiente: theta_k *= scale cada par (ops.cpp:5862 ggml_mrope_cache_init).
// sections: [4] i32 en kernel params (p.ej. qwen35: [11,11,10,0]).
// data: [N, heads, head_dim] TOKEN-MAJOR (layout del caller forwardGPU —
// U2-fix lane-b1; antes row%N head-major).
// Con ids secuenciales (t==h==w==e==pos+start) es bit-equivalente al
// mropeKernel clásico — ver test de paridad.
extern "C" __global__ void mropePosIdsKernel(
    float* __restrict__ data,
    const int* __restrict__ pos_ids,   // [N][4]
    int rows, int N, int head_dim, int n_rot,
    const int* __restrict__ sections,  // [4]
    float base)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    // U2-fix (lane-b1): mismo fix que mropeKernel — data TOKEN-MAJOR
    // [N, heads, head_dim] del caller (forwardGPU pasa g_q/g_k planos).
    // El row % N anterior era correcto sólo con pack head-major (tests
    // mmproj, que copian flat head-major) — scrambleaba los callers reales.
    const int heads = (N > 0) ? rows / N : 1;
    int pos = row / heads;
    const int* ids = pos_ids + (size_t)pos * 4;
    int half_rot = n_rot / 2;
    float scale = powf(base, -2.0f / (float)n_rot);
    int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    int sec_w = sections[0] + sections[1];
    int sec_e = sections[2] + sec_w;
    float* d = data + (size_t)row * head_dim;
    float theta_t = (float)ids[0];
    float theta_h = (float)ids[1];
    float theta_w = (float)ids[2];
    float theta_e = (float)ids[3];
    for (int ic = 0; ic < half_rot; ++ic) {
        int sector = ic % sect_dims;
        float theta = theta_t;
        if (sector >= sections[0] && sector < sec_w) {
            theta = theta_h;
        } else if (sector >= sec_w && sector < sec_e) {
            theta = theta_w;
        } else if (sector >= sec_e) {
            theta = theta_e;
        }
        float c = cosf(theta);
        float s = sinf(theta);
        float q0 = d[ic];
        float q1 = d[ic + half_rot];
        d[ic] = q0 * c - q1 * s;
        d[ic + half_rot] = q0 * s + q1 * c;
        // Escalado independiente por sección (fiel a ggml: cada theta
        // multiplica por scale en cada par de SU sección).
        theta_t *= scale;
        theta_h *= scale;
        theta_w *= scale;
        theta_e *= scale;
    }
}

// ─── M-RoPE VISION interleaved (PLAN_MMPROJ 10.2-bisect, lane-mmproj) ───────
// Port exacto de mrope_vision.buildVisionCache + applyMRopeVision:
// rotación por pares ADYACENTES (2i, 2i+1), frecuencia
// 10000^(-2·ip/n_pairs) con n_pairs = head_dim/2, sector = ip % sect_dims.
// El mropePosIdsKernel (NEOX half-split, theta iterativo) NO es
// intercambiable: los thetas/frecuencias difieren ⇒ el encoder GPU
// divergía cos 0.4 del CPU con ids 2D de imagen.
extern "C" __global__ void mropeVisionKernel(
    float* __restrict__ data,          // [rows][head_dim] (pack por head)
    const int* __restrict__ pos_ids,   // [N][4]
    int rows, int N, int head_dim,
    const int* __restrict__ sections)  // [4]
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    int pos = row % N;
    const int* ids = pos_ids + (size_t)pos * 4;
    const int n_pairs = head_dim / 2;
    int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    int sec_w = sections[0] + sections[1];
    int sec_e = sections[2] + sec_w;
    float base_t = (float)ids[0];
    float base_h = (float)ids[1];
    float base_w = (float)ids[2];
    float base_e = (float)ids[3];
    float* d = data + (size_t)row * head_dim;
    for (int ip = 0; ip < n_pairs; ++ip) {
        int sector = ip % sect_dims;
        float p = base_t;
        if (sector < sections[0]) {
            p = base_t;
        } else if (sector < sec_w) {
            p = base_h;
        } else if (sector < sec_e) {
            p = base_w;
        } else {
            p = base_e;
        }
        float inv_freq = powf(10000.0f, -2.0f * (float)ip / (float)n_pairs);
        float theta = p * inv_freq;
        float c = cosf(theta);
        float s = sinf(theta);
        float q0 = d[2 * ip];
        float q1 = d[2 * ip + 1];
        d[2 * ip] = q0 * c - q1 * s;
        d[2 * ip + 1] = q0 * s + q1 * c;
    }
}

// ─── KV-append f16: escribe K/V del token al pool paginado (d_cache).
// Fiel a reshape_and_block_write_f16_kernel: por bloque físico, K en
// [0, block_size*kv_dim), V en [block_size*kv_dim, 2*block_size*kv_dim).
// k/v: [N, kv_dim] f32; bt: block table device (phys por bloque lógico).
extern "C" __global__ void kvAppendF16Kernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    half* __restrict__ cache,
    const int* __restrict__ bt,
    const int* __restrict__ start_pos,
    int n,
    int kv_dim, int n_kv_head, int head_dim, int block_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * kv_dim;
    if (i >= total) return;
    int t = i / kv_dim;
    int c = i % kv_dim;
    int h = c / head_dim;
    int d = c % head_dim;
    long long global_pos = (long long)*start_pos + t;
    int block_idx = (int)(global_pos / block_size);
    int off = (int)(global_pos % block_size);
    int phys = bt[block_idx];
    if (phys < 0) return;
    long long base = (long long)phys * block_size * kv_dim * 2;
    long long k_idx = base + ((long long)off * kv_dim + h * head_dim + d);
    long long v_idx = base + ((long long)block_size * kv_dim + (long long)off * kv_dim + h * head_dim + d);
    cache[k_idx] = __float2half(k[i]);
    cache[v_idx] = __float2half(v[i]);
}

// ─── KV-append q8_0: cuantiza K/V float→q8_0 canónico al escribir al pool
// paginado. Layout idéntico al que lee paged_attention_decode_q8_0_kernel:
//   por bloque físico: [K: qb*34 bytes][V: qb*34 bytes], qb = elems_región/32,
//   cada grupo de 32 elems: [escala f16 @0..1][32×int8 @2..33].
//
// PARALELIZACIÓN RESTRINGIDA A PROPÓSITO: un solo CUDA-block (1 warp) escribe
// TODO el launch, serializando bloques físicos → lado (K,V) → grupos. Motivo:
// los grupos de 34B no están alineados a sectores de 32B y regiones contiguas
// comparten sector; con escritores concurrentes en distintos CUDA-blocks el
// write-combining pierde actualizaciones parciales del sector (escala
// corrupta, no reproducible con printf activo). Dentro de un mismo bloque,
// __syncwarp/__syncthreads ordenan las tiendas de forma fiable. El append es
// despreciable frente a la atención del prefill; si algún día hace falta
// paralelizar, primero alinear el stride físico a 32B en pool Y kernels.
extern "C" __global__ void kvAppendQ8_0Kernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    uint8_t* __restrict__ cache,
    const int* __restrict__ bt,
    const int* __restrict__ start_pos,
    int n,
    int kv_dim, int n_kv_head, int head_dim, int block_size,
    int dbg /* breadcrumb DUMP_KVQUANT: 1 = emitir diagnóstico interno */)
{
    const int tid = threadIdx.x;
    const int first_block = *start_pos / block_size;
    const int last_block = (*start_pos + n - 1) / block_size;

    const int elems_region = block_size * kv_dim;
    const int qb_per_region = (elems_region + 31) / 32;
    const size_t k_bytes = (size_t)qb_per_region * 34;
    const size_t phys_stride = 2 * k_bytes;

    for (int lb = first_block; lb <= last_block; ++lb) {
        const int phys = bt[lb];
        if (phys < 0) continue;
        for (int side = 0; side < 2; ++side) {
            const float* __restrict__ src = (side == 1) ? v : k;
            uint8_t* region = cache + (size_t)phys * phys_stride
                            + (size_t)side * k_bytes;
            for (int qb = 0; qb < qb_per_region; ++qb) {
                // Grupo qb: elementos be ∈ [qb*32, qb*32+32) de la región,
                // be = off*kv_dim + c. Con kv_dim % 32 == 0 cada grupo cae
                // íntegro en UN token (off constante), así que sólo se
                // escriben los grupos del chunk actual [start_pos, start_pos+n):
                // k/v SOLO contienen esos tokens (en decode n=1, el buffer es
                // de 1×kv_dim — leer tokens previos sería OOB).
                const int be = qb * 32 + tid;
                const int off = be / kv_dim;
                const int c = be % kv_dim;
                const int t_abs = lb * block_size + off;
                // Sólo grupos del chunk actual: k/v contiene EXACTAMENTE los
                // n tokens del chunk en índices RELATIVOS ([0..n)) — indexar
                // con t_abs leía más allá del buffer en decode (start_pos>0):
                // OOB silencioso que crece por paso hasta el 700 sticky.
                if (t_abs < *start_pos || t_abs >= *start_pos + n) continue;
                const int t_rel = t_abs - *start_pos;
                const float src_val = src[(size_t)t_rel * kv_dim + c];

                float amax = fabsf(src_val);
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1)
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                const float d = (amax > 0.0f) ? amax / 127.0f : 1.0f;
                int q = (int)roundf(src_val / d);
                q = max(-127, min(127, q));

                uint8_t* dst = region + (size_t)qb * 34;

                // Breadcrumb DUMP_KVQUANT (gated en runtime vía `dbg`): amax
                // shuffle vs recálculo serial + bits f16 de la escala.
                if (dbg != 0 && side == 0 && qb < 2 && tid == 0) {
                    float m2 = 0.0f;
                    for (int j = 0; j < 32; j++) {
                        const int bj = qb * 32 + j;
                        const int oj = bj / kv_dim, cj = bj % kv_dim;
                        const int tj = lb * block_size + oj;
                        if (tj < *start_pos) continue;
                        if (tj >= *start_pos + n) break;
                        const float vj = src[(size_t)(tj - *start_pos) * kv_dim + cj];
                        m2 = fmaxf(m2, fabsf(vj));
                    }
                    printf("[q80dbg] phys=%d qb=%d dstrel=%lld amax_shfl=%f amax_serial=%f f16=%04hx\n",
                           phys, qb, (long long)(dst - cache), amax, m2,
                           (unsigned short)__half_as_ushort(__float2half(d)));
                }

                // DUMP_KVQUANT=2: modo sólo-lecturas (aislar si el fallo
                // asíncrono viene de leer k/v/bt o de escribir al pool).
                if (dbg != 2) {
                    __stcg(&dst[2 + tid], (uint8_t)(int8_t)q);
                    // Ordena las tiendas de quanta del warp antes de la escala.
                    __syncwarp();
                    if (tid == 0) {
                        const unsigned short bits = __half_as_ushort(__float2half(d));
                        __stcg(&dst[0], (uint8_t)(bits & 0xFF));
                        __stcg(&dst[1], (uint8_t)(bits >> 8));
                    if (dbg != 0 && side == 0 && qb < 2) {
                        const volatile uint8_t* p = dst;
                        printf("[q80w] phys=%d qb=%d escribió %02x %02x releído %02x %02x\n",
                               phys, qb, (unsigned)(bits & 0xFF),
                               (unsigned)(bits >> 8),
                               (unsigned)p[0], (unsigned)p[1]);
                    }
                    }
                }
            }
            // Sector frontera K|V del mismo phys: separar fases dentro del
            // bloque antes de empezar a escribir la región contigua.
            __syncthreads();
        }
    }
}

// ─── KV-append q4_0: cuantiza K/V float→q4_0 canónico al pool paginado.
// Layout por bloque físico: [K: qb*18 bytes][V: qb*18 bytes], cada grupo de
// 32 elems: [escala f16 @0..1][16 bytes de nibbles @2..17] (low = elem par,
// high = elem impar; q = clamp(round(v/d), -8..7)+8, d = amax/7).
// Mismo patrón serializado que kvAppendQ8_0Kernel (contrato single-writer por
// sector de 32B — ver TODO_NO_OOM.md). Un warp itera grupos; el packing de
// nibbles lo hacen las lanes pares vía shuffle del q de la lane impar.
extern "C" __global__ void kvAppendQ4_0Kernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    uint8_t* __restrict__ cache,
    const int* __restrict__ bt,
    const int* __restrict__ start_pos,
    int n,
    int kv_dim, int n_kv_head, int head_dim, int block_size,
    int dbg /* breadcrumb DUMP_KVQUANT */)
{
    const int tid = threadIdx.x;
    const int first_block = *start_pos / block_size;
    const int last_block = (*start_pos + n - 1) / block_size;

    const int elems_region = block_size * kv_dim;
    const int qb_per_region = (elems_region + 31) / 32;
    const size_t k_bytes = (size_t)qb_per_region * 18;
    const size_t phys_stride = 2 * k_bytes;

    for (int lb = first_block; lb <= last_block; ++lb) {
        const int phys = bt[lb];
        if (phys < 0) continue;
        for (int side = 0; side < 2; ++side) {
            const float* __restrict__ src = (side == 1) ? v : k;
            uint8_t* region = cache + (size_t)phys * phys_stride
                            + (size_t)side * k_bytes;
            for (int qb = 0; qb < qb_per_region; ++qb) {
                const int be = qb * 32 + tid;
                const int off = be / kv_dim;
                const int c = be % kv_dim;
                const int t_abs = lb * block_size + off;
                if (t_abs < *start_pos || t_abs >= *start_pos + n) continue;
                const int t_rel = t_abs - *start_pos;
                const float src_val = src[(size_t)t_rel * kv_dim + c];

                float amax = fabsf(src_val);
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1)
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                const float d = (amax > 0.0f) ? amax / 7.0f : 1.0f;
                int q = (int)roundf(src_val / d);
                q = max(-8, min(7, q)) + 8;   // [0..15]

                uint8_t* dst = region + (size_t)qb * 18;

                if (dbg != 0 && side == 0 && qb < 2 && tid == 0) {
                    printf("[q40dbg] phys=%d qb=%d dstrel=%lld amax=%f f16=%04hx\n",
                           phys, qb, (long long)(dst - cache), amax,
                           (unsigned short)__half_as_ushort(__float2half(d)));
                }

                // Empaquetado SPLIT-16 canónico GGML (espejo encodeQ4_0 fix
                // en kv_quant.zig): elems [0,16) = nibbles BAJOS de qs[0..16),
                // [16,32) = ALTOS. La lane tid<16 escribe SU byte completo
                // (low=elem tid, high=elem tid+16) — único escritor por byte.
                // El shuffle se ejecuta UNIFORME antes de la rama (nombrar
                // lanes fuera del mask cuelga el warp Volta+).
                const int q_hi16 = __shfl_xor_sync(0xffffffffu, q, 16);
                __syncwarp();
                if (tid < 16) {
                    const uint8_t packed =
                        (uint8_t)((q & 0x0F) | ((q_hi16 & 0x0F) << 4));
                    __stcg(&dst[2 + tid], packed);
                }
                __syncwarp();
                if (tid == 0) {
                    const unsigned short bits = __half_as_ushort(__float2half(d));
                    __stcg(&dst[0], (uint8_t)(bits & 0xFF));
                    __stcg(&dst[1], (uint8_t)(bits >> 8));
                    if (dbg != 0 && side == 0 && qb < 2) {
                        const volatile uint8_t* p = dst;
                        printf("[q40w] phys=%d qb=%d escribió %02x %02x releído %02x %02x\n",
                               phys, qb, (unsigned)(bits & 0xFF),
                               (unsigned)(bits >> 8),
                               (unsigned)p[0], (unsigned)p[1]);
                    }
                }
            }
            __syncthreads();
        }
    }
}

// ─── Estadísticas de un sub-bloque q4_K: deriva sd/sm de 6 bits y sus
// escalas efectivas a partir de span/offset del sub-bloque (espejo exacto
// del codificador Zig encodeQ4_K).
__device__ __forceinline__ void q4k_sub_stats(
    float span, float off, float d, float dmin,
    uint8_t* sd, uint8_t* sm, float* dl, float* ml)
{
    int isd = (int)lroundf(span / (15.0f * d));
    isd = max(0, min(63, isd));
    int ism = (int)lroundf(off / dmin);
    ism = max(0, min(63, ism));
    *sd = (uint8_t)isd;
    *sm = (uint8_t)ism;
    *dl = d * (float)isd;
    *ml = dmin * (float)ism;
}


// ─── KV-append IQ4_NL: cuantiza K/V float→IQ4_NL canónico al pool paginado.
// Layout por bloque de 32 elems (18B): [d f16][qs[16] split-16]; valor =
// d·kvalues_iq4nl[nibble]. ESPEJO EXACTO de encodeIQ4_NL: d=amax/127;
// nearest LUT ascendente tie→menor índice (con d full-precision). Gran 32 ⇒
// solo exige kv_dim % 32 == 0. Guard continue por grupo fuera del chunk
// (patrón q8_0) + tiendas stcg single-writer por byte (lane tid<16 escribe
// SU byte completo vía shfl_xor 16).
extern "C" __global__ void kvAppendIQ4_NLKernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    uint8_t* __restrict__ cache,
    const int* __restrict__ bt,
    const int* __restrict__ start_pos,
    int n,
    int kv_dim, int n_kv_head, int head_dim, int block_size,
    int dbg)
{
    const int tid = threadIdx.x;
    const int first_block = *start_pos / block_size;
    const int last_block = (*start_pos + n - 1) / block_size;

    const int elems_region = block_size * kv_dim;
    const int qb_per_region = (elems_region + 31) / 32;
    const size_t k_bytes = (size_t)qb_per_region * 18;
    const size_t phys_stride = 2 * k_bytes;

    // LUT kvalues_iq4nl canónica (misma que encodeIQ4_NL/kv_quant.zig).
    const int8_t kv_nl[16] = { -127, -104, -83, -65, -49,
        -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

    for (int lb = first_block; lb <= last_block; ++lb) {
        const int phys = bt[lb];
        if (phys < 0) continue;
        for (int side = 0; side < 2; ++side) {
            const float* __restrict__ src = (side == 1) ? v : k;
            uint8_t* region = cache + (size_t)phys * phys_stride
                            + (size_t)side * k_bytes;
            for (int qb = 0; qb < qb_per_region; ++qb) {
                const int be = qb * 32 + tid;
                const int off = be / kv_dim;
                const int c = be % kv_dim;
                const int t_abs = lb * block_size + off;
                if (t_abs < *start_pos || t_abs >= *start_pos + n) continue;
                const float src_val = src[(size_t)(t_abs - *start_pos) * kv_dim + c];

                float amax = fabsf(src_val);
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1)
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                const float d = (amax > 0.0f) ? amax / 127.0f : 1.0f;

                uint8_t best = 0;
                float best_diff = fabsf(src_val - d * (float)kv_nl[0]);
                #pragma unroll
                for (int id = 1; id < 16; ++id) {
                    const float diff = fabsf(src_val - d * (float)kv_nl[id]);
                    if (diff < best_diff) { best_diff = diff; best = (uint8_t)id; }
                }

                uint8_t* dst = region + (size_t)qb * 18;

                // Split-16 single-writer: lane tid<16 escribe SU byte
                // (low=elem tid, high=elem tid+16 vía shfl uniforme).
                const int q_hi16 = __shfl_xor_sync(0xffffffffu, best, 16);
                __syncwarp();
                if (tid < 16) {
                    __stcg(&dst[2 + tid],
                           (uint8_t)((best & 0x0F) | ((q_hi16 & 0x0F) << 4)));
                }
                __syncwarp();
                if (tid == 0) {
                    const unsigned short bits = __half_as_ushort(__float2half(d));
                    __stcg(&dst[0], (uint8_t)(bits & 0xFF));
                    __stcg(&dst[1], (uint8_t)(bits >> 8));
                }
            }
            __syncthreads();
        }
    }
}
// ─── KV-append q4_K: cuantiza K/V float→q4_K canónico al pool paginado.
// Layout por SB de 256 elems (144B): [d f16@0][dmin f16@2][scales[12]@4]
// [qs[128]@16]; elemento w: g=w/64, l=w%64, escala si=2g+(l≥32), nibble en
// qs[g*32+(l%32)] low/high — exactamente lo que leen el decode q4_k y el
// dequant CPU canónico. Matemática espejo de encodeQ4_K (kv_quant.zig).
//
// REQUISITO: kv_dim % 256 == 0 — así cada SB cae íntegro en UN token y el
// append del chunk nunca necesita valores de tokens previos (con SB parciales
// habría que reescribir datos ya publicados: imposible sin leer el pool).
// Patrón serializado del contrato single-writer por sector (TODO_NO_OOM.md):
// un CUDA-block recorre lb → K/V → super-bloques; stats por sub-bloque vía
// reducciones warp + staging en shared memory; tiendas __stcg.
extern "C" __global__ void kvAppendQ4_KKernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    uint8_t* __restrict__ cache,
    const int* __restrict__ bt,
    const int* __restrict__ start_pos,
    int n,
    int kv_dim, int n_kv_head, int head_dim, int block_size,
    int dbg /* breadcrumb DUMP_KVQUANT */)
{
    const int tid = threadIdx.x;
    const int first_block = *start_pos / block_size;
    const int last_block = (*start_pos + n - 1) / block_size;

    const int elems_region = block_size * kv_dim;
    const int sb_per_region = (elems_region + 255) / 256;
    const size_t k_bytes = (size_t)sb_per_region * 144;
    const size_t phys_stride = 2 * k_bytes;

    __shared__ float s_span[8];
    __shared__ float s_off[8];
    __shared__ uint8_t s_sd[8];
    __shared__ uint8_t s_sm[8];
    __shared__ uint8_t s_sc[12];

    for (int lb = first_block; lb <= last_block; ++lb) {
        const int phys = bt[lb];
        if (phys < 0) continue;
        for (int side = 0; side < 2; ++side) {
            const float* __restrict__ src = (side == 1) ? v : k;
            uint8_t* region = cache + (size_t)phys * phys_stride
                            + (size_t)side * k_bytes;

            for (int sb = 0; sb < sb_per_region; ++sb) {
                uint8_t* dst = region + (size_t)sb * 144;
                const int sb_base_elem = sb * 256;

                // Preservación: con kv_dim%256==0 el SB cae íntegro en el
                // token off_sb. Si ese token no pertenece al chunk actual NO
                // se escribe — reescribirlo pisaría la cuantización previa de
                // tokens ya válidos de este bloque físico (decode-step
                // mid-bloque / prefill sp>0). Uniforme en el bloque (sb lo
                // es) ⇒ seguro ante __syncthreads del cuerpo.
                {
                    const int t_sb = lb * block_size + sb_base_elem / kv_dim;
                    if (t_sb < *start_pos || t_sb >= *start_pos + n) continue;
                }

                // ── Pass 1: stats por sub-bloque de 32 (8 por SB). Elementos
                // fuera de región o fuera del chunk → 0 (padding determinista
                // idéntico al codificador CPU; con kv_dim%256==0 todo SB del
                // chunk pertenece a un único token así que esto sólo afecta
                // al padding final de la región).
                if (tid < 8) { s_span[tid] = 0.0f; s_off[tid] = 0.0f; }
                __syncwarp();
                for (int s2 = 0; s2 < 8; ++s2) {
                    const int w = sb_base_elem + s2 * 32 + tid;
                    const int off_r = w / kv_dim;
                    const int c = w % kv_dim;
                    const int t_abs = lb * block_size + off_r;
                    float val = 0.0f;
                    const bool in_region = w < elems_region;
                    const bool in_chunk = (t_abs >= *start_pos) && (t_abs < *start_pos + n);
                    if (in_region && in_chunk)
                        val = src[(size_t)(t_abs - *start_pos) * kv_dim + c];
                    float mn = val, mx = val;
                    #pragma unroll
                    for (int o = 16; o > 0; o >>= 1) {
                        mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, o));
                        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
                    }
                    if (tid == 0) {
                        s_span[s2] = mx - mn;
                        s_off[s2] = fmaxf(-mn, 0.0f);
                    }
                }
                __syncwarp();

                // Super-escalas del SB (todos los hilos las calculan igual
                // leyendo smem: valor uniforme, sin divergencia posterior).
                float max_span = 0.0f, max_off = 0.0f;
                #pragma unroll
                for (int s2 = 0; s2 < 8; ++s2) {
                    max_span = fmaxf(max_span, s_span[s2]);
                    max_off = fmaxf(max_off, s_off[s2]);
                }
                const float d = (max_span > 0.0f) ? max_span / (15.0f * 63.0f) : 1.0f;
                const float dmin = (max_off > 0.0f) ? max_off / 63.0f : 1.0f;

                // ── Pass 2: cuantiza por grupo-pair de 64 (sub-bloques 2g y
                // 2g+1 comparten byte de nibble: low/high respectivamente).
                for (int g = 0; g < 4; ++g) {
                    uint8_t sd0, sm0, sd1, sm1;
                    float dl0, ml0, dl1, ml1;
                    q4k_sub_stats(s_span[2 * g], s_off[2 * g], d, dmin, &sd0, &sm0, &dl0, &ml0);
                    q4k_sub_stats(s_span[2 * g + 1], s_off[2 * g + 1], d, dmin, &sd1, &sm1, &dl1, &ml1);
                    if (tid == 0) {
                        s_sd[2 * g] = sd0; s_sm[2 * g] = sm0;
                        s_sd[2 * g + 1] = sd1; s_sm[2 * g + 1] = sm1;
                    }

                    // Grupo-pair g cubre elementos [64g, 64(g+1)): sub-bloque
                    // bajo = 2g (elems 64g..64g+31), alto = 2g+1 (+32).
                    const int w0 = sb_base_elem + 2 * g * 32 + tid;
                    const int w1 = w0 + 32;
                    float v0 = 0.0f, v1 = 0.0f;
                    {
                        const int orr0 = w0 / kv_dim, c0 = w0 % kv_dim;
                        const int ta0 = lb * block_size + orr0;
                        if (w0 < elems_region && ta0 >= *start_pos && ta0 < *start_pos + n)
                            v0 = src[(size_t)(ta0 - *start_pos) * kv_dim + c0];
                        const int orr1 = w1 / kv_dim, c1 = w1 % kv_dim;
                        const int ta1 = lb * block_size + orr1;
                        if (w1 < elems_region && ta1 >= *start_pos && ta1 < *start_pos + n)
                            v1 = src[(size_t)(ta1 - *start_pos) * kv_dim + c1];
                    }
                    int q0 = 0, q1 = 0;
                    if (dl0 > 0.0f) q0 = max(0, min(15, (int)lroundf((v0 + ml0) / dl0)));
                    if (dl1 > 0.0f) q1 = max(0, min(15, (int)lroundf((v1 + ml1) / dl1)));
                    const uint8_t byte = (uint8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
                    if (sb == 0 && dbg != 0 && ((g == 0 && tid == 0) || (g == 1 && tid < 2))) {
                        if (g == 1) {
                            printf("[q4kdbg] tid=%d v0=%f v1=%f q0=%d q1=%d dl0=%f ml0=%f dl1=%f ml1=%f sd0=%d sm0=%d\n",
                                   tid, v0, v1, q0, q1, dl0, ml0, dl1, ml1, (int)sd0, (int)sm0);
                        } else {
                        printf("[q4kdbg] phys=%d sb=%d d=%f dmin=%f sd0=%d sm0=%d\n",
                               phys, sb, d, dmin, (int)sd0, (int)sm0);
                        }
                    }
                    __stcg(&dst[16 + g * 32 + tid], byte);
                }

                // ── Escalas empaquetadas (12B) + super-escalas f16, por lane0.
                __syncwarp();
                if (tid == 0) {
                    #pragma unroll
                    for (int si = 0; si < 4; ++si) {
                        s_sc[si] = s_sd[si] & 63;
                        s_sc[si + 4] = s_sm[si] & 63;
                    }
                    #pragma unroll
                    for (int si = 4; si < 8; ++si) {
                        s_sc[si + 4] = (s_sd[si] & 0xF) | ((s_sm[si] & 0xF) << 4);
                        s_sc[si - 4] |= ((s_sd[si] >> 4) & 3) << 6;
                        s_sc[si] |= ((s_sm[si] >> 4) & 3) << 6;
                    }
                    #pragma unroll
                    for (int j = 0; j < 12; ++j) __stcg(&dst[4 + j], s_sc[j]);
                    const unsigned short bd = __half_as_ushort(__float2half(d));
                    const unsigned short bmn = __half_as_ushort(__float2half(dmin));
                    __stcg(&dst[0], (uint8_t)(bd & 0xFF));
                    __stcg(&dst[1], (uint8_t)(bd >> 8));
                    __stcg(&dst[2], (uint8_t)(bmn & 0xFF));
                    __stcg(&dst[3], (uint8_t)(bmn >> 8));
                }
                __syncwarp(); // smem reutilizado por el siguiente SB
            }
            __syncthreads();
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// B3 — MMQ GEMV: activaciones cuantizadas a q8_0 + dp4a contra pesos q4_0.
// Patrón mmq-vec-dot (unsloth/llama.cpp) adaptado al layout de bloques de
// este proyecto. Sólo decode (M≤4); M grande sigue en cuBLAS/dequant.
//
// Paso 1 — quantizeAQ8Kernel: A[M,K] f32 → {a_i8[M*K], d_f16[M*KB], sa_i32[M*KB]}
//   un warp por bloque de 32 elems (mismo patrón amax que kvAppendQ8_0).
//   sa = Σ round(v/d) (i32): corrección del truco unsigned-nibble del paso 2,
//   pues dot_true = Σ a·q = Σ a·n − 8·sa con n=q+8 ∈ [0,15].
// Paso 2 — mmqQ4_0GEMVKernel: C[M,N] = A_q8 · W_q4_0.
//   Un warp por columna j; thread lane procesa el bloque kb=it·32+lane
//   completo: 4×u32 de W → 8 dp4a contra 8 ints de A. Escalas aplicadas en
//   f32 por bloque (dA[kb]·dB[j,kb]). Contrato single-writer no aplica
//   (escritura coalescida disjunta por elemento de salida).
// ════════════════════════════════════════════════════════════════════════════
extern "C" __global__ void mmqQuantizeAQ8Kernel(
    const float* __restrict__ a,      // [M*K]
    int8_t* __restrict__ aq,          // [M*K]
    __half* __restrict__ ad,          // [M*KB] escala f16 por bloque
    int* __restrict__ asa,            // [M*KB] suma de quanta (corrección −8)
    int m, int k)
{
    const int kb_total = k / 32;
    const int g = blockIdx.x;          // bloque lógico global (m,kb)
    const int kb = g % kb_total;
    const int mi = g / kb_total;
    if (mi >= m) return;
    const int tid = threadIdx.x;
    const int base = mi * k + kb * 32 + tid;
    const float v = a[base];

    float amax = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = (amax > 0.0f) ? amax / 127.0f : 1.0f;
    int q = (int)roundf(v / d);
    q = max(-127, min(127, q));

    aq[base] = (int8_t)q;

    // Σ q del bloque vía reducción warp (para la corrección -8·sa).
    int ssum = q;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        ssum += __shfl_xor_sync(0xffffffffu, ssum, o);

    if (tid == 0) {
        ad[mi * kb_total + kb] = __float2half(d);
        asa[mi * kb_total + kb] = ssum;
    }
}

extern "C" __global__ void mmqQ4_0GEMVKernel(
    const int8_t* __restrict__ aq,    // [M*K]
    const __half* __restrict__ ad,    // [M*KB]
    const int* __restrict__ asa,      // [M*KB]
    const uint8_t* __restrict__ w,    // [N][KB*18] pesos q4_0 crudos GGUF
    float* __restrict__ c,            // [M*N] (acumulado con atomics: pre-cero)
    int m, int k, int n)
{
    const int kb_total = k / 32;
    // Split-K: blockIdx.z reparte los bloques K entre SK warps por columna
    // (ocupación ×SK; parciales atómicos f32 sobre C pre-cerado).
    const int sk = gridDim.z;
    const int kb_lo = (int)(((long long)kb_total * blockIdx.z) / sk);
    const int kb_hi = (int)(((long long)kb_total * (blockIdx.z + 1)) / sk);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int j = blockIdx.x * (blockDim.x >> 5) + warp; // columna de salida
    if (j >= n) return;

    // Layout dinámico EN BYTES: aq [M*K pad16] | d f32 [M*KB] | sa f32 [M*KB].
    // OJO: aritmética sobre puntero float avanza ×4 — usar base byte.
    extern __shared__ unsigned char smem_raw[];
    int8_t* s_aq = (int8_t*)smem_raw;
    float* s_d = (float*)(smem_raw + ((m * k + 15) & ~15));
    float* s_sa = s_d + m * kb_total;

    for (int i = tid; i < m * k; i += blockDim.x) s_aq[i] = aq[i];
    for (int i = tid; i < m * kb_total; i += blockDim.x) {
        s_d[i] = __half2float(ad[i]);
        s_sa[i] = (float)asa[i];
    }
    __syncthreads();

    float accf[4] = {0.f, 0.f, 0.f, 0.f};
    const uint8_t* wrow = w + (size_t)j * kb_total * 18;

    for (int kb = kb_lo + lane; kb < kb_hi; kb += 32) {
        const uint8_t* blk = wrow + (size_t)kb * 18;

        // 16 bytes de quanta como 4×u32 (lecturas escalares L1: el stride
        // 18B impide vectorizar; optimización futura con padding).
        // 16 bytes de quanta via 5 cargas u32 alineadas + funnel shift:
        // el stride 18B hace la alineacion ciclica mod-4 y las lecturas
        // escalares eran el cuello principal del kernel.
        uint32_t u[4];
        {
            const uint32_t* vp = (const uint32_t*)((uintptr_t)(blk + 2) & ~(uintptr_t)3);
            const uint32_t sh = (uint32_t)(((uintptr_t)(blk + 2) & 3) * 8);
            const uint32_t r0 = vp[0], r1 = vp[1], r2 = vp[2], r3 = vp[3], r4 = vp[4];
            u[0] = __funnelshift_r(r0, r1, sh);
            u[1] = __funnelshift_r(r1, r2, sh);
            u[2] = __funnelshift_r(r2, r3, sh);
            u[3] = __funnelshift_r(r3, r4, sh);
        }
        const float db = __half2float(*(const __half*)blk);

        for (int mi = 0; mi < m; ++mi) {
            const int8_t* abase = s_aq + mi * k + kb * 32;
            // Emparejamiento fijo: u[0]→elems 0..7 (ints @0,@4), u[1]→8..15,
            // u[2]→16..23, u[3]→24..31 (cada u[t] = elems 8t..8t+7).
            const int a0 = *(const int*)(abase);
            const int a1 = *(const int*)(abase + 4);
            const int a2 = *(const int*)(abase + 8);
            const int a3 = *(const int*)(abase + 12);
            const int a4 = *(const int*)(abase + 16);
            const int a5 = *(const int*)(abase + 20);
            const int a6 = *(const int*)(abase + 24);
            const int a7 = *(const int*)(abase + 28);

            // SPLIT-16 canónico GGML (fix conjunto q4_0): elems [0,16) =
            // LOW nibbles de qs[0..16); [16,32) = HIGH de los MISMOS bytes.
            // u[t] = bytes 4t..4t+3 ⇒ lows de u[t] = elems 4t..4t+3,
            // highs de u[t] = elems 16+4t..16+4t+3.
            // SPLIT-16: lows = máscara directa por byte (0x0F0F0F0F); highs =
            // shift word >>4 + misma máscara (el cruce entre bytes se
            // recorta). Sin reempaquetado — el canon es dp4a-natural.
            const uint32_t lo_m = 0x0F0F0F0Fu;
            int sn = 0;
            sn = __dp4a((int)(u[0] & lo_m), a0, sn);
            sn = __dp4a((int)(u[1] & lo_m), a1, sn);
            sn = __dp4a((int)(u[2] & lo_m), a2, sn);
            sn = __dp4a((int)(u[3] & lo_m), a3, sn);
            sn = __dp4a((int)((u[0] >> 4) & lo_m), a4, sn);
            sn = __dp4a((int)((u[1] >> 4) & lo_m), a5, sn);
            sn = __dp4a((int)((u[2] >> 4) & lo_m), a6, sn);
            sn = __dp4a((int)((u[3] >> 4) & lo_m), a7, sn);
            // Corrección unsigned-nibble: Σ a·q = Σ a·n − 8·Σa
            const int dot_true = sn - 8 * (int)s_sa[mi * kb_total + kb];
            accf[mi] += db * s_d[mi * kb_total + kb] * (float)dot_true;
        }
    }

    // Reducción entre lanes del warp (cada lane acumuló su sub-conjunto de kb)
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        #pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            float v = accf[mi];
            v += __shfl_xor_sync(0xffffffffu, v, o);
            accf[mi] = (mi < m) ? v : 0.f;
        }
    }
    // Parciales atómicos: cada slice añade su suma de bloques.
    if (lane == 0) {
        for (int mi = 0; mi < m; ++mi)
            atomicAdd(&c[(size_t)mi * n + j], accf[mi]);
    }
}

// ─── KV-append q8_K: cuantiza K/V float→q8_K canónico al pool paginado.
// Layout por SB de 256 elems (292B): [d f32 LE @0..3][qs i8×256 @4..259]
// [bsums i16×16 @260..291 = ceros, unused por el decode]. Espejo exacto de
// encodeQ8_K. REQUIERE kv_dim % 256 == 0 (SB íntegro por token). Patrón
// serializado del contrato single-writer por sector.
extern "C" __global__ void kvAppendQ8_KKernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    uint8_t* __restrict__ cache,
    const int* __restrict__ bt,
    const int* __restrict__ start_pos,
    int n,
    int kv_dim, int n_kv_head, int head_dim, int block_size,
    int dbg)
{
    const int tid = threadIdx.x;
    const int first_block = *start_pos / block_size;
    const int last_block = (*start_pos + n - 1) / block_size;

    const int elems_region = block_size * kv_dim;
    const int sb_per_region = (elems_region + 255) / 256;
    const size_t k_bytes = (size_t)sb_per_region * 292;
    const size_t phys_stride = 2 * k_bytes;

    for (int lb = first_block; lb <= last_block; ++lb) {
        const int phys = bt[lb];
        if (phys < 0) continue;
        for (int side = 0; side < 2; ++side) {
            const float* __restrict__ src = (side == 1) ? v : k;
            uint8_t* region = cache + (size_t)phys * phys_stride
                            + (size_t)side * k_bytes;
            for (int sb = 0; sb < sb_per_region; ++sb) {
                uint8_t* dst = region + (size_t)sb * 292;

                // Preservación: mismo guard que Q4_K (SB íntegro por token;
                // fuera del chunk ⇒ no reescribir). Uniforme en el bloque.
                {
                    const int t_sb = lb * block_size + (sb * 256) / kv_dim;
                    if (t_sb < *start_pos || t_sb >= *start_pos + n) continue;
                }

                // amax de los 256 elems: lane acumula sus 8 strided + butterfly.
                float amax = 0.0f;
                #pragma unroll
                for (int kk = 0; kk < 8; ++kk) {
                    const int w = sb * 256 + tid + kk * 32;
                    const bool ok = (w < elems_region);
                    const int off_r = w / kv_dim;
                    const int t_abs = lb * block_size + off_r;
                    float val = 0.0f;
                    if (ok && t_abs >= *start_pos && t_abs < *start_pos + n)
                        val = src[(size_t)(t_abs - *start_pos) * kv_dim + (w % kv_dim)];
                    amax = fmaxf(amax, fabsf(val));
                }
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1)
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                const float d = (amax > 0.0f) ? amax / 127.0f : 1.0f;

                #pragma unroll
                for (int kk = 0; kk < 8; ++kk) {
                    const int w = sb * 256 + tid + kk * 32;
                    const bool ok = (w < elems_region);
                    const int off_r = w / kv_dim;
                    const int t_abs = lb * block_size + off_r;
                    float val = 0.0f;
                    if (ok && t_abs >= *start_pos && t_abs < *start_pos + n)
                        val = src[(size_t)(t_abs - *start_pos) * kv_dim + (w % kv_dim)];
                    int q = (int)roundf(val / d);
                    q = max(-128, min(127, q));
                    __stcg(&dst[4 + tid + kk * 32], (uint8_t)(int8_t)q);
                }
                if (dbg != 0 && sb == 0 && tid == 0) {
                    printf("[q8kdbg] phys=%d d=%f\n", phys, d);
                }
                // bsums a cero (unused) + escala f32 LE por lane0.
                __syncwarp();
                if (tid == 0) {
                    const uint32_t bits = __float_as_uint(d);
                    __stcg(&dst[0], (uint8_t)(bits & 0xFF));
                    __stcg(&dst[1], (uint8_t)((bits >> 8) & 0xFF));
                    __stcg(&dst[2], (uint8_t)((bits >> 16) & 0xFF));
                    __stcg(&dst[3], (uint8_t)(bits >> 24));
                }
                if (tid < 32) __stcg(&dst[260 + tid], (uint8_t)0);
                __syncwarp();
            }
            __syncthreads();
        }
    }
}

// ─── LUT canónica IQ4_NL (kernels/tables.cuh, ggml-common.h).
__device__ __forceinline__ uint8_t iq4nl_nearest(float xv, float dl) {
    const int8_t kv[16] = { -127, -104, -83, -65, -49, -35, -22, -10,
                             1, 13, 25, 38, 53, 69, 89, 113 };
    uint8_t best = 0;
    float best_diff = fabsf(xv - dl * (float)kv[0]);
    #pragma unroll
    for (int id = 1; id < 16; ++id) {
        const float diff = fabsf(xv - dl * (float)kv[id]);
        if (diff < best_diff) { best_diff = diff; best = (uint8_t)id; }
    }
    return best;
}


// ─── KV-append Q2_K: cuantiza K/V float→Q2_K canónico al pool paginado.
// Layout por SB de 256 elems (84B): [scales[16]@0][qs[64]@16][d f16@80]
// [min f16@82]. Sub-bloques de 16 elems (16/SB): byte scales[s] = dcode
// nibble low + mcode high; quanta 2-bit: byte qs[nh*32+col] campo shift=2j.
// ESPEJO EXACTO de encodeQ2_K (kv_quant.zig): paso=(mx-mn)/3;
// d=max_span/45, min_s=max_neg/15; dcode=round(paso/d), mcode=round(neg/min_s)
// (clamped 0..15); q=clamp(round((x+ml)/dl),0,3) con ml=min_s·mcode.
// Serial en tid==0 (cuantización directa sin búsqueda ⇒ µs). REQUIERE
// kv_dim % 256 == 0. Guard preservación SB íntegro por token.
extern "C" __global__ void kvAppendQ2_KKernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    uint8_t* __restrict__ cache,
    const int* __restrict__ bt,
    const int* __restrict__ start_pos,
    int n,
    int kv_dim, int n_kv_head, int head_dim, int block_size,
    int dbg)
{
    const int tid = threadIdx.x;
    if (tid != 0) return;
    const int first_block = *start_pos / block_size;
    const int last_block = (*start_pos + n - 1) / block_size;

    const int elems_region = block_size * kv_dim;
    const int sb_per_region = (elems_region + 255) / 256;
    const size_t k_bytes = (size_t)sb_per_region * 84;
    const size_t phys_stride = 2 * k_bytes;

    for (int lb = first_block; lb <= last_block; ++lb) {
        const int phys = bt[lb];
        if (phys < 0) continue;
        for (int side = 0; side < 2; ++side) {
            const float* __restrict__ src = (side == 1) ? v : k;
            uint8_t* region = cache + (size_t)phys * phys_stride
                            + (size_t)side * k_bytes;

            auto ld = [&](int w) -> float {
                const int off_r = w / kv_dim;
                const int t_abs = lb * block_size + off_r;
                if (w >= elems_region || t_abs < *start_pos || t_abs >= *start_pos + n) return 0.0f;
                return src[(size_t)(t_abs - *start_pos) * kv_dim + (w % kv_dim)];
            };

            for (int sb = 0; sb < sb_per_region; ++sb) {
                uint8_t* dst = region + (size_t)sb * 84;
                const int sb_base = sb * 256;

                // Preservación: SB íntegro por token (mismo guard que Q4_K).
                {
                    const int t_sb = lb * block_size + sb_base / kv_dim;
                    if (t_sb < *start_pos || t_sb >= *start_pos + n) continue;
                }

                // Zero de escalas+qs (los OR posteriores asumen base 0;
                // la memoria del pool puede ser stale). d/min van aparte.
                for (int z = 0; z < 80; ++z) __stcg(&dst[z], 0);

                // Pass 1: stats por sub-bloque de 16.
                float span[16], neg[16];
                float max_span = 0.0f, max_neg = 0.0f;
                for (int s = 0; s < 16; ++s) {
                    float mn = FLT_MAX, mx = -FLT_MAX; // lane-cuda: 1e308f fuera de rango f32 (NVRTC lo rechaza; nvcc degrada a inf)
                    for (int c = 0; c < 16; ++c) {
                        const float vv = ld(sb_base + s * 16 + c);
                        mn = fminf(mn, vv);
                        mx = fmaxf(mx, vv);
                    }
                    span[s] = mx - mn;
                    neg[s] = (mn < 0.0f) ? -mn : 0.0f;
                    max_span = fmaxf(max_span, span[s]);
                    max_neg = fmaxf(max_neg, neg[s]);
                }
                const float d = (max_span > 0.0f) ? max_span / 45.0f : 1.0f;
                const float min_s = (max_neg > 0.0f) ? max_neg / 15.0f : 1.0f;

                // Pass 2: códigos + quanta.
                for (int s = 0; s < 16; ++s) {
                    int dcode = (int)roundf(span[s] / 3.0f / d);
                    dcode = max(0, min(15, dcode));
                    int mcode = (int)roundf(neg[s] / min_s);
                    mcode = max(0, min(15, mcode));
                    dst[s] = (uint8_t)(dcode | (mcode << 4));

                    const float dl = d * (float)dcode;
                    const float ml = min_s * (float)mcode;
                    const float inv = (dl > 0.0f) ? 1.0f / dl : 0.0f;
                    const int nh = s >> 3;
                    const int jj = (s >> 1) & 3;
                    const int shift = 2 * jj;
                    for (int c = 0; c < 16; ++c) {
                        int q = (int)roundf((ld(sb_base + s * 16 + c) + ml) * inv);
                        q = max(0, min(3, q));
                        const int col = c + (s & 1) * 16;
                        dst[16 + nh * 32 + col] |= (uint8_t)(q << shift);
                    }
                }

                const unsigned short bd = __half_as_ushort(__float2half(d));
                __stcg(&dst[80], (uint8_t)(bd & 0xFF));
                __stcg(&dst[81], (uint8_t)(bd >> 8));
                const unsigned short md = __half_as_ushort(__float2half(min_s));
                __stcg(&dst[82], (uint8_t)(md & 0xFF));
                __stcg(&dst[83], (uint8_t)(md >> 8));
            }
        }
    }
}

// ─── KV-append Q3_K: cuantiza K/V float→Q3_K canónico al pool paginado.
// Layout por SB de 256 elems (110B): [hmask[32]@0][qs[64]@32][scales[12]@96]
// [d f16@108]. Sub-bloques de 16: dl=d·(i8(s16[s])−32); rango {−4..3}·dl.
// ESPEJO EXACTO de encodeQ3_K: dl_target=span/7; s16[s]=clamp(round(dl/d)+32,
// 32,63) — el reorden kmask solo preserva 6 bits/escala; d=max_span/(7·31).
// Per-elem: x≥0 → bit hmask SET (global nh*4+j), q=round(x/dl); x<0 → CLEAR,
// q=clamp(round(x/dl)+4,0,3). Escalas escritas vía INVERSA del reorden kmask
// (transcrita de encodeQ3_K, verificada biyectiva ≤63). Serial en tid==0.
extern "C" __global__ void kvAppendQ3_KKernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    uint8_t* __restrict__ cache,
    const int* __restrict__ bt,
    const int* __restrict__ start_pos,
    int n,
    int kv_dim, int n_kv_head, int head_dim, int block_size,
    int dbg)
{
    const int tid = threadIdx.x;
    if (tid != 0) return;
    const int first_block = *start_pos / block_size;
    const int last_block = (*start_pos + n - 1) / block_size;

    const int elems_region = block_size * kv_dim;
    const int sb_per_region = (elems_region + 255) / 256;
    const size_t k_bytes = (size_t)sb_per_region * 110;
    const size_t phys_stride = 2 * k_bytes;

    for (int lb = first_block; lb <= last_block; ++lb) {
        const int phys = bt[lb];
        if (phys < 0) continue;
        for (int side = 0; side < 2; ++side) {
            const float* __restrict__ src = (side == 1) ? v : k;
            uint8_t* region = cache + (size_t)phys * phys_stride
                            + (size_t)side * k_bytes;

            auto ld = [&](int w) -> float {
                const int off_r = w / kv_dim;
                const int t_abs = lb * block_size + off_r;
                if (w >= elems_region || t_abs < *start_pos || t_abs >= *start_pos + n) return 0.0f;
                return src[(size_t)(t_abs - *start_pos) * kv_dim + (w % kv_dim)];
            };

            for (int sb = 0; sb < sb_per_region; ++sb) {
                uint8_t* dst = region + (size_t)sb * 110;
                const int sb_base = sb * 256;

                // Preservación: mismo guard que Q4_K.
                {
                    const int t_sb = lb * block_size + sb_base / kv_dim;
                    if (t_sb < *start_pos || t_sb >= *start_pos + n) continue;
                }

                float max_span = 0.0f;
                for (int s = 0; s < 16; ++s) {
                    float mn = FLT_MAX, mx = -FLT_MAX; // lane-cuda: 1e308f fuera de rango f32 (NVRTC lo rechaza; nvcc degrada a inf)
                    for (int c = 0; c < 16; ++c) {
                        const float vv = ld(sb_base + s * 16 + c);
                        mn = fminf(mn, vv);
                        mx = fmaxf(mx, vv);
                    }
                    max_span = fmaxf(max_span, mx - mn);
                }
                const float d = (max_span > 0.0f) ? max_span / (7.0f * 31.0f) : 1.0f;

                // Zero de hmask+qs+scales (OR posterior asume base 0).
                for (int z = 0; z < 108; ++z) __stcg(&dst[z], 0);

                uint8_t s16[16];
                for (int s = 0; s < 16; ++s) {
                    float mn = FLT_MAX, mx = -FLT_MAX; // lane-cuda: 1e308f fuera de rango f32 (NVRTC lo rechaza; nvcc degrada a inf)
                    for (int c = 0; c < 16; ++c) {
                        const float vv = ld(sb_base + s * 16 + c);
                        mn = fminf(mn, vv);
                        mx = fmaxf(mx, vv);
                    }
                    const float dl_t = (mx - mn) / 7.0f;
                    int b_val = (int)roundf(dl_t / d) + 32;
                    b_val = max(32, min(63, b_val)); // ≥32 ⇒ dl≥0; ≤63 (6-bit)
                    s16[s] = (uint8_t)b_val;

                    const float dl = d * (float)((int8_t)b_val - 32);
                    const float inv = (dl > 0.0f) ? 1.0f / dl : 0.0f;
                    const int nh = s >> 3;
                    const int j = (s >> 1) & 3;
                    const int shift = 2 * j;
                    for (int c = 0; c < 16; ++c) {
                        const float xv = ld(sb_base + s * 16 + c);
                        const int col = c + (s & 1) * 16;
                        int q;
                        if (xv >= 0.0f) {
                            q = max(0, min(3, (int)roundf(xv * inv)));
                            dst[col] |= (uint8_t)(1u << (nh * 4 + j)); // bit GLOBAL
                        } else {
                            q = (int)roundf(xv * inv) + 4;
                            q = max(0, min(3, q));
                        }
                        dst[32 + nh * 32 + col] |= (uint8_t)(q << shift);
                    }
                }

                // INVERSA del reorden kmask → scales[12] (dst[96..108]).
                uint32_t aux[4] = {0, 0, 0, 0};
                for (int i = 0; i < 16; ++i)
                    aux[i / 4] |= (uint32_t)s16[i] << ((i % 4) * 8);
                for (int b = 0; b < 4; ++b) {
                    const uint8_t o0 = (uint8_t)(aux[0] >> (b * 8));
                    const uint8_t o1 = (uint8_t)(aux[1] >> (b * 8));
                    const uint8_t o2 = (uint8_t)(aux[2] >> (b * 8));
                    const uint8_t o3 = (uint8_t)(aux[3] >> (b * 8));
                    dst[96 + b] = (o0 & 0xF) | ((o2 & 0xF) << 4);
                    dst[100 + b] = (o1 & 0xF) | ((o3 & 0xF) << 4);
                    dst[104 + b] = ((o0 >> 4) & 3) | (((o1 >> 4) & 3) << 2)
                                 | (((o2 >> 4) & 3) << 4) | (((o3 >> 4) & 3) << 6);
                }

                const unsigned short bd = __half_as_ushort(__float2half(d));
                __stcg(&dst[108], (uint8_t)(bd & 0xFF));
                __stcg(&dst[109], (uint8_t)(bd >> 8));
            }
        }
    }
}

// ─── KV-append IQ4_XS: cuantiza K/V float→IQ4_XS canónico al pool paginado.
// Layout por SB de 256 elems (136B): [d f16@0][scales_h u16@2][scales_l[4]@4
// (pares nibble: sb par→low, impar→high)][qs[128]@8; byte ib*16+j' con
// low=elem j<16 / high=j≥16 del sub-bloque ib]. Espejo exacto de
// encodeIQ4_XS. REQUIERE kv_dim % 256 == 0 (SB íntegro por token). Patrón
// serializado single-writer por sector (TODO_NO_OOM.md §Contrato).
extern "C" __global__ void kvAppendIQ4_XSKernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    uint8_t* __restrict__ cache,
    const int* __restrict__ bt,
    const int* __restrict__ start_pos,
    int n,
    int kv_dim, int n_kv_head, int head_dim, int block_size,
    int dbg)
{
    const int tid = threadIdx.x;
    const int first_block = *start_pos / block_size;
    const int last_block = (*start_pos + n - 1) / block_size;

    const int elems_region = block_size * kv_dim;
    const int sb_per_region = (elems_region + 255) / 256;
    const size_t k_bytes = (size_t)sb_per_region * 136;
    const size_t phys_stride = 2 * k_bytes;

    __shared__ float s_amax[8];
    __shared__ int s_ls[8];

    for (int lb = first_block; lb <= last_block; ++lb) {
        const int phys = bt[lb];
        if (phys < 0) continue;
        for (int side = 0; side < 2; ++side) {
            const float* __restrict__ src = (side == 1) ? v : k;
            uint8_t* region = cache + (size_t)phys * phys_stride
                            + (size_t)side * k_bytes;

            auto ld = [&](int w) -> float {
                const int off_r = w / kv_dim;
                const int t_abs = lb * block_size + off_r;
                if (w >= elems_region || t_abs < *start_pos || t_abs >= *start_pos + n) return 0.0f;
                return src[(size_t)(t_abs - *start_pos) * kv_dim + (w % kv_dim)];
            };

            for (int sb = 0; sb < sb_per_region; ++sb) {
                uint8_t* dst = region + (size_t)sb * 136;
                const int sb_base = sb * 256;

                // Preservación: mismo guard que Q4_K (SB íntegro por token;
                // fuera del chunk ⇒ no reescribir). Uniforme en el bloque.
                {
                    const int t_sb = lb * block_size + sb_base / kv_dim;
                    if (t_sb < *start_pos || t_sb >= *start_pos + n) continue;
                }

                // Pass 1: amax por sub-bloque de 32 (reducción warp).
                if (dbg != 0 && sb < 2 && tid < 8)
                    printf("[ld] sb=%d tid=%d val=%.6f\n", sb, tid, ld(sb_base + tid));
                for (int s2 = 0; s2 < 8; ++s2) {
                    const int w = sb_base + s2 * 32 + tid;
                    float amax = fabsf(ld(w));
                    #pragma unroll
                    for (int o = 16; o > 0; o >>= 1)
                        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                    if (tid == 0) s_amax[s2] = amax;
                }
                __syncwarp();

                float amax_all = 0.0f;
                #pragma unroll
                for (int s2 = 0; s2 < 8; ++s2) amax_all = fmaxf(amax_all, s_amax[s2]);
                const float d = (amax_all > 0.0f) ? amax_all / (113.0f * 31.0f) : 1.0f;

                // Pass 2: ls por sub-bloque + quanta empaquetados.
                // OJO layout: qs[ib*16 + j'] con j'=j%16 — CADA sub-bloque
                // posee solo 16 bytes; byte j' empaqueta elem j'(low) y
                // elem j'+16(high). Lane tid<16 escribe SU byte completo
                // (único escritor). Antes, tid 0..31 escribían 32 bytes ⇒
                // desbordamiento al SB contiguo (cabecera de escalas).
                for (int ib = 0; ib < 8; ++ib) {
                    int ls = 32;
                    const float am = s_amax[ib];
                    if (am > 0.0f) {
                        ls = 32 + (int)ceilf(am / (113.0f * d));
                        ls = max(33, min(63, ls));
                    }
                    if (tid == 0) s_ls[ib] = ls;
                    const float dl = d * (float)(ls - 32);

                    if (tid < 16) {
                        const int w_lo = sb_base + ib * 32 + tid;
                        const int w_hi = w_lo + 16;
                        const uint8_t packed = (uint8_t)(
                            (uint8_t)iq4nl_nearest(ld(w_lo), dl) |
                            ((uint8_t)iq4nl_nearest(ld(w_hi), dl) << 4));
                        __stcg(&dst[8 + ib * 16 + tid], packed);
                    }
                }
                __syncwarp();

                if (tid == 0) {
                    uint8_t sl[4] = {0, 0, 0, 0};
                    uint16_t sh = 0;
                    for (int sbi = 0; sbi < 8; ++sbi) {
                        const int ls = s_ls[sbi];
                        sl[sbi / 2] |= (uint8_t)((ls & 0xF) << (4 * (sbi % 2)));
                        sh |= (uint16_t)((ls >> 4) & 3) << (2 * sbi);
                    }
                    #pragma unroll
                    for (int t2 = 0; t2 < 4; ++t2) __stcg(&dst[4 + t2], sl[t2]);
                    const unsigned short bd = __half_as_ushort(__float2half(d));
                    __stcg(&dst[0], (uint8_t)(bd & 0xFF));
                    __stcg(&dst[1], (uint8_t)(bd >> 8));
                    __stcg(&dst[2], (uint8_t)(sh & 0xFF));
                    __stcg(&dst[3], (uint8_t)(sh >> 8));
                    if (dbg != 0 && sb < 2) {
                        printf("[q4xsd] ls=[%d %d %d %d %d %d %d %d] d=%f amax=[",
                               s_ls[0], s_ls[1], s_ls[2], s_ls[3],
                               s_ls[4], s_ls[5], s_ls[6], s_ls[7], d);
                        #pragma unroll
                        for (int s2 = 0; s2 < 8; ++s2)
                            printf("%s%.6f", s2 ? " " : "", s_amax[s2]);
                        printf("]\n");
                    }
                }
                __syncwarp();
            }
            __syncthreads();
        }
    }
}

// ─── B6 — MMQ GEMV con PESOS q8_0: C[M,N] = A_q8 · W_q8_0ᵀ.
// W layout GGUF por fila j (34B × KB): [d f16][i8×32]. Ambos lados signed
// ⇒ dp4a directo sin corrección unsigned. Para lm_head grande cuantizado
// on-load (bloquea 27B NEO-MTP: output.weight BF16 2.4GB no cabe).
extern "C" __global__ void mmqQ8_0WGEMVKernel(
    const int8_t* __restrict__ aq,    // [M*K]
    const __half* __restrict__ ad,    // [M*KB]
    const uint8_t* __restrict__ w,    // [N][KB*34]
    float* __restrict__ c,            // [M*N] pre-cero + atomics
    int m, int k, int n,
    int dbg_mmq, int dbg_kb, int dbg_lane)
{
    const int kb_total = k / 32;
    const int sk = gridDim.z;
    const int kb_lo = (int)(((long long)kb_total * blockIdx.z) / sk);
    const int kb_hi = (int)(((long long)kb_total * (blockIdx.z + 1)) / sk);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int j = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= n) return;

    extern __shared__ unsigned char smem_raw[];
    int8_t* s_aq = (int8_t*)smem_raw;
    float* s_d = (float*)(smem_raw + ((m * k + 15) & ~15));

    for (int i = tid; i < m * k; i += blockDim.x) s_aq[i] = aq[i];
    for (int i = tid; i < m * kb_total; i += blockDim.x)
        s_d[i] = __half2float(ad[i]);
    __syncthreads();

    float accf[8] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    const uint8_t* wrow = w + (size_t)j * kb_total * 34;

    for (int kb = kb_lo + lane; kb < kb_hi; kb += 32) {
        const uint8_t* blk = wrow + (size_t)kb * 34;
        // 32 bytes de quanta alineables a bloques de 2 (stride 34): 8 cargas
        // u16 + empaque a u32, o 8 escalares. Stride 34 ≡ 2 mod 4 ⇒ paridad
        // alterna; usamos 2×u32 vía par de u16 para mantener simple.
        const float db = __half2float(*(const __half*)blk);

        for (int mi = 0; mi < m; ++mi) {
            const int8_t* abase = s_aq + mi * k + kb * 32;
            // u[t] = w_i8 elems [4t..4t+3]; A int en abase+t*4 = mismos elems.
            // Ambos lados signed ⇒ dp4a directo, sin corrección.
            int sn = 0;
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                const uint32_t b = (uint32_t)blk[2 + t * 4]
                    | ((uint32_t)blk[2 + t * 4 + 1] << 8)
                    | ((uint32_t)blk[2 + t * 4 + 2] << 16)
                    | ((uint32_t)blk[2 + t * 4 + 3] << 24);
                sn = __dp4a((int)b, *(const int*)(abase + t * 4), sn);
            }
            if (dbg_mmq != 0 && (dbg_kb < 0 || kb == dbg_kb) && mi == 0 && j == dbg_lane) {
                printf("[mmqd] j=%d kb=%d sn=%d db=%.6e da=%.6e bytes=", j, kb, sn, db, s_d[mi * kb_total + kb]);
                for (int h = 0; h < 6; ++h) printf("%02x ", blk[h]);
                printf("\n");
                if (lane == 0) {
                    int sn_dbg = 0;
                    for (int t = 0; t < 8; ++t) {
                        const uint32_t b = (uint32_t)blk[2 + t * 4]
                            | ((uint32_t)blk[2 + t * 4 + 1] << 8)
                            | ((uint32_t)blk[2 + t * 4 + 2] << 16)
                            | ((uint32_t)blk[2 + t * 4 + 3] << 24);
                        const int av = *(const int*)(abase + t * 4);
                        const int before = sn_dbg;
                        sn_dbg = __dp4a((int)b, av, sn_dbg);
                        printf("[mmqt] t=%d b=%08x a=%08x dp4a_delta=%d\n",
                               t, b, av, sn_dbg - before);
                    }
                    printf("[mmqt] sn_sum=%d\n", sn_dbg);
                }
            }
            accf[mi] += db * s_d[mi * kb_total + kb] * (float)sn;
        }
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        #pragma unroll
        for (int mi = 0; mi < 8; ++mi) {
            float v = accf[mi];
            v += __shfl_xor_sync(0xffffffffu, v, o);
            if (mi < m) accf[mi] = v;
        }
    }
    if (lane == 0) {
        for (int mi = 0; mi < m; ++mi) atomicAdd(&c[(size_t)mi * n + j], accf[mi]);
    }
}

// ─── B3-v3 — GEMV q8_0W FUSIONADO: cuantiza A en-kernel (fase 1, smem,
// solo el rango kb de ESTE split) + GEMV dp4a (fase 2) en UN launch.
// Ataca el diagnóstico B3-v2 (latency-bound: 3 launches ≈15µs + roundtrip
// global aq/ad). La escala d pasa por __float2half→half2float para
// reproducir EXACTAMENTE la aritmética del camino no-fusionado (ad f16).
// Split-K z + atomics sobre C pre-cerado (igual que mmqQ8_0WGEMV).
// OJO warps con j≥n: NO hacen return temprano — participan en fase 1
// (smem compartido por bloque; un hueco rompería a los warps activos) y
// solo saltan la tienda final.
extern "C" __global__ void mmqQ8_0WFusedKernel(
    const float* __restrict__ a,      // [M*K] activaciones crudas
    const uint8_t* __restrict__ w,    // [N][KB*34]
    float* __restrict__ c,            // [M*N] pre-cero + atomics
    int m, int k, int n,
    int dbg_mmq, int dbg_kb, int dbg_lane)
{
    const int kb_total = k / 32;
    const int sk = gridDim.z;
    const int kb_lo = (int)(((long long)kb_total * blockIdx.z) / sk);
    const int kb_hi = (int)(((long long)kb_total * (blockIdx.z + 1)) / sk);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int j = blockIdx.x * (blockDim.x >> 5) + warp;

    extern __shared__ unsigned char smem_raw[];
    int8_t* s_aq = (int8_t*)smem_raw;
    float* s_d = (float*)(smem_raw + ((m * k + 15) & ~15));

    // ── Fase 1: cuantizar A (un warp por bloque (mi,kb) del rango split).
    {
        const int kbs = kb_hi - kb_lo;
        const int nw = blockDim.x >> 5;
        for (int g2 = warp; g2 < m * kbs; g2 += nw) {
            const int mi = g2 / kbs;
            const int kb = kb_lo + (g2 % kbs);
            const int base = mi * k + kb * 32 + lane;
            const float v = a[base];
            float amax = fabsf(v);
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
            const float df = (amax > 0.0f) ? amax / 127.0f : 1.0f;
            int q = (int)roundf(v / df);
            q = max(-127, min(127, q));
            s_aq[base] = (int8_t)q;
            if (lane == 0)
                s_d[mi * kb_total + kb] = __half2float(__float2half(df));
        }
    }
    __syncthreads();

    // ── Fase 2: GEMV dp4a idéntico a mmqQ8_0WGEMV (lee smem ya cuantizado).
    if (j >= n) return;
    float accf[8] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    const uint8_t* wrow = w + (size_t)j * kb_total * 34;

    for (int kb = kb_lo + lane; kb < kb_hi; kb += 32) {
        const uint8_t* blk = wrow + (size_t)kb * 34;
        const float db = __half2float(*(const __half*)blk);

        for (int mi = 0; mi < m; ++mi) {
            const int8_t* abase = s_aq + mi * k + kb * 32;
            int sn = 0;
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                const uint32_t b = (uint32_t)blk[2 + t * 4]
                    | ((uint32_t)blk[2 + t * 4 + 1] << 8)
                    | ((uint32_t)blk[2 + t * 4 + 2] << 16)
                    | ((uint32_t)blk[2 + t * 4 + 3] << 24);
                sn = __dp4a((int)b, *(const int*)(abase + t * 4), sn);
            }
            accf[mi] += db * s_d[mi * kb_total + kb] * (float)sn;
        }
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        #pragma unroll
        for (int mi = 0; mi < 8; ++mi) {
            float v = accf[mi];
            v += __shfl_xor_sync(0xffffffffu, v, o);
            if (mi < m) accf[mi] = v;
        }
    }
    if (lane == 0 && j < n) {
        for (int mi = 0; mi < m; ++mi) atomicAdd(&c[(size_t)mi * n + j], accf[mi]);
    }
}

__device__ const unsigned long long dev_iq1s_grid[2048] = {
    18446744073709551615, 18446744073709551361, 18446744073709486080, 18446744073709486591,
    18446744073709486337, 18446744073692839680, 18446744073692774400, 18446744073692905471,
    18446744073692905217, 18446744073692840447, 18446744073692840193, 18446744069431296000,
    18446744069414649600, 18446744069414584575, 18446744069414584321, 18446744069414649856,
    18446744069448138751, 18446744069448138497, 18446744069448073727, 18446744069448073473,
    18446744069431361536, 18446744069431492607, 18446744069431492353, 18446744069431427583,
    18446744069431427329, 18446742978492825855, 18446742978492825600, 18446742978476179200,
    18446742978476114175, 18446742978476113921, 18446742978476114176, 18446742978476114177,
    18446742978476179456, 18446742974214700800, 18446742974214635521, 18446742974214635776,
    18446742974197989121, 18446742974197923840, 18446742974197924097, 18446742974198054656,
    18446742974197989631, 18446742974197989377, 18446742974197989887, 18446742974231412736,
    18446742974214766336, 18446742974214701311, 18446742974214701057, 18446742974214766592,
    18446742982787858431, 18446742982787858177, 18446742982787793407, 18446742982787793153,
    18446742982771081216, 18446742982771212287, 18446742982771212033, 18446742982771147263,
    18446742982771147009, 18446742978509602816, 18446742978492956416, 18446742978492891392,
    18446742978492956927, 18446742978492956928, 18446742978526445567, 18446742978526445313,
    18446742978526380543, 18446742978526380289, 18446742978509733632, 18446742978509668352,
    18446742978509668608, 18446742978509799423, 18446742978509799169, 18446742978509734399,
    18446742978509734145, 18446463698227756800, 18446463698227691775, 18446463698227691521,
    18446463698227757056, 18446463693966278400, 18446463693966213376, 18446463693949501440,
    18446463693949501697, 18446463693949567231, 18446463693949566976, 18446463693966343936,
    18446463693966278912, 18446463693966344192, 18446462603027808000, 18446462603027742975,
    18446462603027742720, 18446462603027742721, 18446462603011031040, 18446462603011031551,
    18446462603011031297, 18446462603011096832, 18446462598749618175, 18446462598749552640,
    18446462598749552897, 18446462598732906495, 18446462598732906240, 18446462598732841215,
    18446462598732840960, 18446462598732840961, 18446462598732841216, 18446462598732972031,
    18446462598732971777, 18446462598732906496, 18446462598732907007, 18446462598732906753,
    18446462598766395136, 18446462598749683456, 18446462598749618176, 18446462598749618687,
    18446462598749618433, 18446462598749748992, 18446462598749683967, 18446462598749683712,
    18446462598749683713, 18446462598749683968, 18446462607305998591, 18446462607305998592,
    18446462603044585216, 18446462603044520191, 18446462603027873791, 18446462603027873537,
    18446462603027808256, 18446462603027808767, 18446462603027939327, 18446462603027939072,
    18446462603027873793, 18446462603027874048, 18446462603061297152, 18446462603044650752,
    18446462603044585727, 18446462603044585728, 18446464797756096511, 18446464797756096257,
    18446464797756031487, 18446464797756031233, 18446464797739319296, 18446464797739450367,
    18446464797739450113, 18446464797739385343, 18446464797739385089, 18446464793477840896,
    18446464793461194496, 18446464793461129217, 18446464793461194752, 18446464793494683647,
    18446464793494683393, 18446464793494618623, 18446464793494618369, 18446464793477906432,
    18446464793478037503, 18446464793478037249, 18446464793477972479, 18446464793477972225,
    18446463702539370496, 18446463702522724096, 18446463702522659071, 18446463702522659072,
    18446463702522724607, 18446463702522724352, 18446463698261245696, 18446463698244534271,
    18446463698244534016, 18446463698244468736, 18446463698244599552, 18446463698244534527,
    18446463698244534528, 18446463698261311232, 18446463698261246207, 18446463698261245953,
    18446463698261246208, 18446463698261311488, 18446463706834403327, 18446463706834403073,
    18446463706834338303, 18446463706834338049, 18446463706817626112, 18446463706817757183,
    18446463706817756929, 18446463706817692159, 18446463706817691905, 18446463702556147712,
    18446463702539501312, 18446463702539436288, 18446463702539566848, 18446463702539501568,
    18446463702572990463, 18446463702572990209, 18446463702572924928, 18446463702572925439,
    18446463702572925185, 18446463702556213248, 18446463702556344319, 18446463702556344065,
    18446463702556279295, 18446463702556279041, 18374967954631622655, 18374967954631622400,
    18374967954631557375, 18374967954631557376, 18374967954631622911, 18374967954631622656,
    18374967950370144000, 18374967950370078975, 18374967950353432575, 18374967950353367040,
    18374967950353367551, 18374967950353497856, 18374967950353432831, 18374967950353432576,
    18374967950353432832, 18374967950370209536, 18374967950370144511, 18374967950370144257,
    18374967950370275072, 18374967950370209792, 18374966859431673600, 18374966859431608575,
    18374966859431608321, 18374966859431608576, 18374966859414962175, 18374966859414961921,
    18374966859414896640, 18374966859414897151, 18374966859415027456, 18374966859414962431,
    18374966859414962432, 18374966855153418240, 18374966855153418497, 18374966855136772095,
    18374966855136771840, 18374966855136771841, 18374966855136706815, 18374966855136706560,
    18374966855136706561, 18374966855136706816, 18374966855136837631, 18374966855136772096,
    18374966855170195711, 18374966855153549057, 18374966855153483776, 18374966855153614592,
    18374966855153549567, 18374966863709929216, 18374966863709864191, 18374966863709863937,
    18374966863709929472, 18374966859448451071, 18374966859448385537, 18374966859448385792,
    18374966859431739137, 18374966859431673856, 18374966859431674367, 18374966859431674113,
    18374966859431739647, 18374966859431739393, 18374966859465162752, 18374966859448516352,
    18374966859448451327, 18374966859448451073, 18374966859448516608, 18374687579183251200,
    18374687579183185921, 18374687579183186176, 18374687579166474495, 18374687579166474240,
    18374687579166474751, 18374687579166474496, 18374687579166605056, 18374687579166539777,
    18374687574905061120, 18374687574904995840, 18374687574904995841, 18374687574904996351,
    18374687574904996097, 18374687574888349440, 18374687574888284415, 18374687574888284160,
    18374687574888284161, 18374687574888284416, 18374687574888414977, 18374687574888349696,
    18374687574888350207, 18374687574921773311, 18374687574921773312, 18374687574905126911,
    18374687574905061631, 18374687574905061376, 18374687574905061887, 18374687574905061632,
    18374687574905061633, 18374687574905192192, 18374687574905127167, 18374687574905126912,
    18374687574905127168, 18374686483966590721, 18374686483966525440, 18374686483966525697,
    18374686483949879040, 18374686483949814015, 18374686483949813760, 18374686483949813761,
    18374686483949814016, 18374686483949944831, 18374686483949944577, 18374686483949879296,
    18374686483949879807, 18374686483949879553, 18374686479688400640, 18374686479688335615,
    18374686479688335360, 18374686479688335361, 18374686479671688960, 18374686479671688961,
    18374686479671623935, 18374686479671623680, 18374686479671623681, 18374686479671623936,
    18374686479671623937, 18374686479671754496, 18374686479671689471, 18374686479671689216,
    18374686479671689217, 18374686479671689472, 18374686479705178111, 18374686479705177857,
    18374686479705112831, 18374686479705112576, 18374686479705113087, 18374686479705112833,
    18374686479688466431, 18374686479688466176, 18374686479688401151, 18374686479688400896,
    18374686479688400897, 18374686479688401152, 18374686479688401153, 18374686479688531967,
    18374686479688531713, 18374686479688466432, 18374686488261558016, 18374686488261492991,
    18374686488261492736, 18374686488261492737, 18374686488244781056, 18374686488244781057,
    18374686488244781567, 18374686488244781313, 18374686488244911872, 18374686488244846593,
    18374686483983368191, 18374686483983367937, 18374686483983302911, 18374686483983302656,
    18374686483983303167, 18374686483983302913, 18374686483966656256, 18374686483966590976,
    18374686483966590977, 18374686483966591487, 18374686483966591232, 18374686483966721792,
    18374686483966656767, 18374686483966656512, 18374686483966657023, 18374686483966656768,
    18374686483966656769, 18374686484000079873, 18374686484000080129, 18374686483983433473,
    18374686483983368192, 18374686483983433983, 18374686483983433984, 18374688678678167296,
    18374688678678102017, 18374688678678167552, 18374688674416688896, 18374688674416623871,
    18374688674416623617, 18374688674416623872, 18374688674399977471, 18374688674399911936,
    18374688674399912447, 18374688674399912193, 18374688674400043007, 18374688674400042752,
    18374688674399977727, 18374688674399977473, 18374688674399977728, 18374688674433400832,
    18374688674416754432, 18374688674416689407, 18374688674416754688, 18374687583461507071,
    18374687583461506817, 18374687583461441536, 18374687583461441793, 18374687583461572352,
    18374687583461507072, 18374687579200028417, 18374687579199963391, 18374687579199963136,
    18374687579199963647, 18374687579183316736, 18374687579183251711, 18374687579183251456,
    18374687579183251457, 18374687579183251712, 18374687579183251713, 18374687579183382527,
    18374687579183316992, 18374687579183317249, 18374687579216740608, 18374687579200093952,
    18374687579200093953, 18374687579200028672, 18374687579200029183, 18374687579200159488,
    18374687579200094209, 18374687579200094464, 18374687587773120768, 18374687587756408833,
    18374687587756474623, 18374687587756474369, 18374687583494930687, 18374687583494930433,
    18374687583494930688, 18374687583478284287, 18374687583478284033, 18374687583478218752,
    18374687583478219263, 18374687583478349568, 18374687583478284289, 18374687583478284544,
    18374687583511707648, 18374687583495061248, 18374687583494995969, 18374687583494996225,
    18375249429625044991, 18375249429625044737, 18375249429624979967, 18375249429624979713,
    18375249429608267776, 18375249429608398847, 18375249429608398593, 18375249429608333312,
    18375249429608333823, 18375249429608333569, 18375249425346789376, 18375249425330142976,
    18375249425330077952, 18375249425330208512, 18375249425330143232, 18375249425363632127,
    18375249425363631873, 18375249425363567103, 18375249425363566849, 18375249425346854912,
    18375249425346985983, 18375249425346985729, 18375249425346920448, 18375249425346920959,
    18375249425346920705, 18375248334408318976, 18375248334391672576, 18375248334391607551,
    18375248334391607552, 18375248334391672832, 18375248330130194177, 18375248330130129151,
    18375248330130129152, 18375248330113417216, 18375248330113417727, 18375248330113417473,
    18375248330113548032, 18375248330113483007, 18375248330113482752, 18375248330113482753,
    18375248330146906112, 18375248330130259967, 18375248330130194433, 18375248330130194688,
    18375248330130259968, 18375248338703351552, 18375248338703286783, 18375248338703286529,
    18375248338686639872, 18375248338686574592, 18375248338686705663, 18375248338686705409,
    18375248338686640639, 18375248338686640385, 18375248334425096192, 18375248334408449792,
    18375248334408384513, 18375248334408384768, 18375248334408450048, 18375248334441938688,
    18375248334441873919, 18375248334441873665, 18375248334425227008, 18375248334425161728,
    18375248334425292799, 18375248334425292545, 18375248334425227775, 18375248334425227521,
    18374969054159896576, 18374969054143185151, 18374969054143184897, 18374969054143185152,
    18374969054143250432, 18374969049881706751, 18374969049881706496, 18374969049881706497,
    18374969049881706752, 18374969049865060097, 18374969049864994816, 18374969049864995327,
    18374969049864995073, 18374969049865060353, 18374969049898483712, 18374969049881837312,
    18374969049881772287, 18374969049881772288, 18374969049881902848, 18374969049881837568,
    18374967958943236352, 18374967958926524416, 18374967958926655232, 18374967958926590208,
    18374967954665111551, 18374967954665046016, 18374967954665046527, 18374967954648399616,
    18374967954648334591, 18374967954648334336, 18374967954648334592, 18374967954648465153,
    18374967954648399872, 18374967954648400383, 18374967954681823488, 18374967954665177087,
    18374967954665111807, 18374967954665111552, 18374967954665112063, 18374967954665111809,
    18374967954665242368, 18374967954665177343, 18374967954665177089, 18374967954665177344,
    18374967963238203392, 18374967963221557247, 18374967963221556993, 18374967963221491968,
    18374967963221557248, 18374967958960078592, 18374967958960013568, 18374967958943301632,
    18374967958943432703, 18374967958943432448, 18374967958943367424, 18374967958976790783,
    18374967958976790529, 18374967958960144383, 18374967958960079105, 18374970153671589887,
    18374970153671589633, 18374970153671524863, 18374970153671524609, 18374970153654812672,
    18374970153654943743, 18374970153654943489, 18374970153654878719, 18374970153654878465,
    18374970149393334272, 18374970149376687872, 18374970149376622847, 18374970149376688128,
    18374970149410177023, 18374970149410176769, 18374970149410111999, 18374970149410111745,
    18374970149393530879, 18374970149393530625, 18374970149393465855, 18374970149393465601,
    18374969058454864128, 18374969058438217472, 18374969058438152447, 18374969058438152448,
    18374969058438217728, 18374969054176673793, 18374969054176674048, 18374969054160027393,
    18374969054159962112, 18374969054160092928, 18374969054160027903, 18374969054160027649,
    18374969054160027904, 18374969054193451008, 18374969054176804863, 18374969054176739329,
    18374969054176739584, 18374969054176805119, 18374969054176804864, 18374969062749896703,
    18374969062749896449, 18374969062749831679, 18374969062749831425, 18374969062733250559,
    18374969062733250305, 18374969062733185535, 18374969062733185281, 18374969058471641088,
    18374969058454994688, 18374969058454929409, 18374969058454929664, 18374969058454994944,
    18374969058488483839, 18374969058488483585, 18374969058488418815, 18374969058488418561,
    18374969058471706624, 18374969058471837695, 18374969058471837441, 18374969058471772671,
    18374969058471772417, 72057594037862400, 72057594021216000, 72057594021150721,
    72057594021216256, 72057589759672576, 72057589743025921, 72057589742960640,
    72057589742961151, 72057589742960897, 72057589743091456, 72057589743026431,
    72057589743026177, 72057589759738111, 72057589759738112, 72057589759868672,
    72057589759803393, 72056498821267455, 72056498821267200, 72056498821202175,
    72056498821201921, 72056498821202176, 72056498804555521, 72056498804490240,
    72056498804490241, 72056498804490751, 72056498804490497, 72056498804621056,
    72056498804555777, 72056498804556032, 72056494543011840, 72056494543012351,
    72056494543012097, 72056494526365440, 72056494526300415, 72056494526300160,
    72056494526300161, 72056494526300416, 72056494526300417, 72056494526365696,
    72056494526366207, 72056494526365953, 72056494559854336, 72056494559789311,
    72056494559789057, 72056494543142911, 72056494543142657, 72056494543077376,
    72056494543208447, 72056494543208192, 72056494543208193, 72056503116169216,
    72056503099522816, 72056503099457791, 72056503099457537, 72056503099523072,
    72056498838044416, 72056498821332737, 72056498821267456, 72056498821267713,
    72056498821333247, 72056498821333248, 72056498854756608, 72056498838044927,
    72056498838110208, 71777218572844800, 71777218556067840, 71777218556068096,
    71777218556133632, 71777214294589440, 71777214294589951, 71777214294589697,
    71777214277943040, 71777214277878015, 71777214277877760, 71777214277877761,
    71777214278008576, 71777214278008577, 71777214277943296, 71777214277943807,
    71777214277943553, 71777214311431936, 71777214311366657, 71777214311366912,
    71777214294720511, 71777214294720257, 71777214294654976, 71777214294786047,
    71777214294785792, 71777214294720768, 71776123356184320, 71776123356184321,
    71776123356119040, 71776123356119297, 71776123339472640, 71776123339407615,
    71776123339407360, 71776123339407361, 71776123339407616, 71776123339538431,
    71776123339472896, 71776123339473153, 71776119077994240, 71776119077929215,
    71776119077928960, 71776119077928961, 71776119077929216, 71776119061282815,
    71776119061282560, 71776119061217535, 71776119061217280, 71776119061217281,
    71776119061217791, 71776119061217536, 71776119061348096, 71776119061283071,
    71776119061282816, 71776119061282817, 71776119061283072, 71776119094771457,
    71776119094706431, 71776119094706176, 71776119094706687, 71776119078059776,
    71776119077994751, 71776119077994496, 71776119077994497, 71776119077994752,
    71776119077994753, 71776119078060032, 71776119078060543, 71776119078060289,
    71776127651151616, 71776127651086336, 71776127651086592, 71776127634374911,
    71776127634374656, 71776127634375167, 71776127634374913, 71776127634505472,
    71776127634440447, 71776127634440448, 71776123372961791, 71776123372961537,
    71776123372896256, 71776123372896767, 71776123356250111, 71776123356249856,
    71776123356249857, 71776123356184576, 71776123356184577, 71776123356184832,
    71776123356315393, 71776123356250112, 71776123356250623, 71776123389738752,
    71776123389673472, 71776123389673729, 71776123372962047, 71776123372961792,
    71776123373092608, 71776123373027583, 71776123373027329, 71778318084407296,
    71778318067760896, 71778318067695616, 71778318067695873, 71778318067761152,
    71778313806282497, 71778313806217472, 71778313789571071, 71778313789505536,
    71778313789506047, 71778313789636352, 71778313789571327, 71778313789571073,
    71778313789571328, 71778313822994432, 71778313806348032, 71778313806283007,
    71778313806282753, 71778313806283008, 71778313806348288, 71777222867812096,
    71777222867746816, 71777222867746817, 71777222867747073, 71777222851100671,
    71777222851035391, 71777222851035136, 71777222851035647, 71777222851165952,
    71777222851100927, 71777222851100673, 71777218589622271, 71777218589556736,
    71777218589556993, 71777218572910336, 71777218572910337, 71777218572845311,
    71777218572845056, 71777218572845057, 71777218572845312, 71777218572976127,
    71777218572975873, 71777218572910592, 71777218572910593, 71777218572910849,
    71777218606333953, 71777218606334208, 71777218589687553, 71777218589622272,
    71777218589622273, 71777218589622783, 71777218589753088, 71777218589688063,
    71777218589687809, 71777218589688064, 71777227146002433, 71777222884524287,
    71777222884524033, 71777222884524288, 71777222867812352, 71777222867812863,
    71777222867812609, 71777222867878143, 71777222867878144, 71777222884589823,
    71777222884655104, 281474976710400, 281474976645375, 281474976645120,
    281474976645121, 281474976645376, 281474959998721, 281474959933440,
    281474959933697, 281474960064256, 281474959999231, 281474959999232,
    281470698520575, 281470698455040, 281470698455551, 281470681808640,
    281470681743615, 281470681743360, 281470681743361, 281470681743616,
    281470681808896, 281470681809407, 281470715232257, 281470715232512,
    281470698520576, 281470698521087, 281470698651647, 281470698651392,
    281470698586113, 281470698586368, 280379759984640, 280379759985151,
    280379759984896, 280379759984897, 280379743338240, 280379743273215,
    280379743272960, 280379743272961, 280379743273471, 280379743273216,
    280379743404031, 280379743338496, 280379743338497, 280379743339007,
    280379743338753, 280375481859840, 280375481794815, 280375481794560,
    280375481794561, 280375481794816, 280375465148415, 280375465148160,
    280375465148161, 280375465083135, 280375465082880, 280375465082881,
    280375465083391, 280375465083136, 280375465083137, 280375465213696,
    280375465148671, 280375465148416, 280375465148417, 280375465148672,
    280375498637057, 280375498571776, 280375481925376, 280375481860351,
    280375481860096, 280375481860097, 280375481860352, 280375481991167,
    280375481925632, 280375481925889, 280384055017216, 280384054951937,
    280384038305537, 280384038240256, 280384038240513, 280384038371072,
    280384038306047, 280379776827137, 280379776761856, 280379776762113,
    280379760115456, 280379760050431, 280379760050176, 280379760050177,
    280379760050432, 280379760180993, 280379760115712, 280379793539072,
    280379776892927, 280379776892673, 280379776827392, 280379776827648,
    280379776827649, 280379776893183, 1099511562495, 1099511562240,
    1099494915840, 1099494850815, 1099494850560, 1099494850561,
    1099494851071, 1099494850816, 1099494981376, 1099494916096,
    1099494916607, 1099494916353, 1095233437440, 1095233372415,
    1095233372160, 1095233372161, 1095233372416, 1095233372417,
    1095216726015, 1095216725760, 1095216660735, 1095216660480,
    1095216660481, 1095216660991, 1095216660736, 1095216660737,
    1095216791296, 1095216791297, 1095216726271, 1095216726016,
    1095216726017, 1095216726272, 1095250214911, 1095250149887,
    1095250149633, 1095233502976, 1095233437951, 1095233437696,
    1095233437697, 1095233437952, 1095233568512, 1095233503487,
    1095233503232, 1095233503489, 4294967040, 4294967041,
    4294902015, 4294901760, 4294901761, 4294902016,
    4278255615, 4278255360, 4278255361, 4278190335,
    4278190080, 4278190081, 4278190336, 4278190337,
    4278320896, 4278255871, 4278255616, 4278255617,
    4278255872, 16777215, 16776960, 16776961,
    16711935, 16711680, 16711681, 16712191,
    16711936, 65535, 65280, 65281,
    255, 0, 1, 511,
    256, 257, 131071, 130816,
    65791, 65536, 65537, 66047,
    65792, 65793, 33554176, 33489151,
    33488896, 33489152, 33489153, 16842751,
    16842496, 16777471, 16777216, 16777217,
    16777727, 16777472, 16908032, 16843007,
    16842752, 16842753, 16843008, 8589934591,
    8589934336, 8589934337, 8589869311, 8589869057,
    8589869567, 8589869312, 8573222656, 8573157631,
    8573157376, 8573157887, 8573157632, 8573288447,
    8573288192, 8573288193, 8573223167, 8573222912,
    8573222913, 8573223423, 8573223168, 4311744256,
    4311678976, 4311678977, 4311679487, 4311679232,
    4311679233, 4295032831, 4295032576, 4295032577,
    4294967551, 4294967296, 4294967297, 4294967807,
    4294967552, 4294967553, 4295098112, 4295033087,
    4295032832, 4295033088, 4328521473, 4328456192,
    4328456193, 4328456703, 4328456448, 4328456449,
    4311809792, 4311744512, 4311744769, 4311875329,
    4311810048, 4311810049, 4311810559, 4311810304,
    2199023190271, 2199023190016, 2199023190017, 2199023190272,
    2199006543871, 2199006478336, 2199006478847, 2199006609152,
    2199006544129, 2194744999936, 2194745000447, 2194745000193,
    2194728353536, 2194728288511, 2194728288256, 2194728288257,
    2194728288767, 2194728288512, 2194728419327, 2194728419073,
    2194728354047, 2194728353792, 2194761842433, 2194761777408,
    2194745131007, 2194745130753, 2194745065472, 2194745065983,
    2194745196288, 2194745131264, 1103806594816, 1103806594817,
    1103806529536, 1103806529793, 1103789883136, 1103789818111,
    1103789817856, 1103789817857, 1103789818112, 1103789883392,
    1099528404736, 1099528339711, 1099528339456, 1099528339457,
    1099528339712, 1099511693311, 1099511693056, 1099511693057,
    1099511628031, 1099511627776, 1099511627777, 1099511628287,
    1099511628032, 1099511628033, 1099511758592, 1099511693567,
    1099511693312, 1099511693313, 1099511693568, 1099545181952,
    1099545116672, 1099545116928, 1099528470272, 1099528405247,
    1099528404992, 1099528404993, 1099528405503, 1099528405248,
    1099528470528, 1108101497087, 1108101497343, 1108084785152,
    1108084785409, 1108084916223, 1108084850688, 1108084850689,
    1108084850944, 1103823306752, 1103823307263, 1103823307008,
    1103806660352, 1103806595072, 1103806595073, 1103806595583,
    1103806595328, 1103806725889, 1103806660608, 1103806661119,
    1103806660865, 1103840149248, 1103840084225, 1103823437569,
    1103823372288, 1103823372289, 1103823372799, 1103823372545,
    1103823503104, 562949953355776, 562949936644351, 562949936644097,
    562949936644352, 562949936709632, 562945675165951, 562945658519551,
    562945658454016, 562945658454017, 562945658454527, 562945658454273,
    562945658584832, 562945658519807, 562945658519553, 562945658519808,
    562945692008192, 562945675231233, 562945675296768, 561854736760576,
    561854736695551, 561854736695297, 561854736695552, 561854720048897,
    561854719983616, 561854720114432, 561854720114433, 561854720049153,
    561854720049408, 561850458505216, 561850458505472, 561850441858816,
    561850441793536, 561850441793537, 561850441793792, 561850441859072,
    561850441859073, 561850441859329, 561850475282687, 561850475282689,
    561850458636033, 561850458570752, 561850458701568, 561850458636289,
    561850458636544, 561859015016192, 561859014950913, 561859014951168,
    561854753538047, 561854753537792, 561854753472513, 561854736760832,
    561854736760833, 561854736761343, 561854736891903, 561854770249728,
    561854753603328, 561854753538049, 561854753603584, 282574471626496,
    282574471626497, 282574471561216, 282574471561217, 282574471561473,
    282574471692032, 282574471626753, 282574471627008, 282570210148351,
    282570210148097, 282570210082816, 282570210083327, 282570210083073,
    282570193436416, 282570193371391, 282570193371136, 282570193371137,
    282570193371392, 282570193371393, 282570193502207, 282570193501953,
    282570193436672, 282570226860287, 282570226860032, 282570226860288,
    282570210213887, 282570210213633, 282570210148607, 282570210148352,
    282570210148353, 282570210148863, 282570210148609, 282570210279168,
    282570210214143, 282570210214144, 281479271612416, 281479271612927,
    281479271612673, 281479254966016, 281479254900736, 281479254900737,
    281479254900992, 281474993422591, 281474993422336, 281474993422337,
    281474993422592, 281474976776191, 281474976775936, 281474976710911,
    281474976710656, 281474976710657, 281474976710912, 281474976841472,
    281474976776447, 281474976776192, 281474976776193, 281474976776448,
    281475010199553, 281475010199808, 281475010199809, 281474993553152,
    281474993487872, 281474993487873, 281474993488128, 281474993488129,
    281474993618689, 281474993553408, 281474993553409, 281474993553919,
    281483566644993, 281483566579968, 281483549868032, 281483549999103,
    281483549933569, 281483549934079, 281483549933824, 281479288455167,
    281479288389632, 281479288390143, 281479288389889, 281479271743232,
    281479271678207, 281479271677952, 281479271677953, 281479271678463,
    281479271678209, 281479271809023, 281479271743488, 281479271743999,
    281479305232383, 281479305232129, 281479305166848, 281479305167105,
    281479288455423, 281479288455169, 281479288455679, 281479288455424,
    281479288586239, 281479288520959, 281479288520705, 281479288520961,
    283673983188993, 283673983189248, 283673983254528, 283669721775872,
    283669705064193, 283669704998912, 283669704999169, 283669705129728,
    283669705064704, 283669738487808, 283669721841408, 283669721776639,
    283669721841665, 282578783305472, 282578783240447, 282578766594047,
    282578766528512, 282578766659328, 282578766594303, 282578766594049,
    282578766594304, 282574505115647, 282574505115392, 282574505050112,
    282574505050113, 282574505050623, 282574488403712, 282574488338687,
    282574488338432, 282574488338433, 282574488338688, 282574488469503,
    282574488403968, 282574488404225, 282574521892609, 282574521827583,
    282574521827585, 282574505115648, 282574505246464, 282574505181439,
    282574505181184, 282574505181440, 282583061561088, 282583061495809,
    282583061496319, 282578800082688, 282578800017663, 282578800017664,
    282578783371263, 282578783305728, 282578783306239, 282578783305985,
    282578783371519, 282578783371264, 282578783371520, 282578816794625,
    282578800083199, 282578800083455, 282578800083201, 282578800148481,
    144115188075855871, 144115188075855617, 144115188075790847, 144115188075790593,
    144115188059209727, 144115188059209473, 144115188059144703, 144115188059144449,
    144115183797600256, 144115183780954111, 144115183780953856, 144115183780888831,
    144115183780888577, 144115183780888832, 144115183780954112, 144115183814443007,
    144115183814442753, 144115183814377983, 144115183814377729, 144115183797665792,
    144115183797796863, 144115183797796609, 144115183797731839, 144115183797731585,
    144114092859129856, 144114092842483456, 144114092842418431, 144114092842418177,
    144114092842418432, 144114092842483712, 144114088581005056, 144114088580940031,
    144114088580940032, 144114088564293631, 144114088564293377, 144114088564228096,
    144114088564228097, 144114088564228607, 144114088564228352, 144114088564293887,
    144114088564293633, 144114088564293888, 144114088597716992, 144114088597717248,
    144114088581005567, 144114088581005313, 144114088581005568, 144114088581070848,
    144114097154162687, 144114097154162433, 144114097154097663, 144114097154097409,
    144114097137385472, 144114097137516543, 144114097137516289, 144114097137451519,
    144114097137451265, 144114092859260672, 144114092859195647, 144114092859195648,
    144114092859260928, 144114092892749823, 144114092892749569, 144114092892684799,
    144114092892684545, 144114092875972608, 144114092876103679, 144114092876103425,
    144114092876038655, 144114092876038401, 143834812593996031, 143834812593996032,
    143834808332582656, 143834808332517631, 143834808315870976, 143834808315805696,
    143834808315805953, 143834808315936512, 143834808315871487, 143834808315871488,
    143834808332583167, 143834808332583168, 143833717394112256, 143833717394047232,
    143833717377400577, 143833717377335296, 143833717377335553, 143833717377400833,
    143833717377401088, 143833713115922431, 143833713115922176, 143833713115856896,
    143833713115857407, 143833713099210496, 143833713099145471, 143833713099145216,
    143833713099145217, 143833713099145472, 143833713099145473, 143833713099210752,
    143833713099210753, 143833713099211263, 143833713099211009, 143833713132699392,
    143833713132634367, 143833713132634113, 143833713132634368, 143833713115987967,
    143833713115987713, 143833713115922432, 143833713115922943, 143833713115987969,
    143833721672367872, 143833721672302593, 143833721672302848, 143833721672368128,
    143833717410889472, 143833717410824447, 143833717410824448, 143833717410824449,
    143833717394178047, 143833717394112512, 143833717394112768, 143833717394112769,
    143833717394243328, 143833717394178049, 143833717394178305, 143833717427601408,
    143833717410955008, 143833717410889985, 143833717410955519, 143835912122400767,
    143835912122400513, 143835912122335743, 143835912122335489, 143835912105623552,
    143835912105754623, 143835912105754369, 143835912105689599, 143835912105689345,
    143835907844210432, 143835907844145152, 143835907827498752, 143835907827433727,
    143835907827433728, 143835907827499008, 143835907827499264, 143835907860987903,
    143835907860987649, 143835907860922879, 143835907860922625, 143835907844210688,
    143835907844341759, 143835907844341505, 143835907844276735, 143835907844276481,
    143834816905674752, 143834816905674753, 143834816889028352, 143834816888963327,
    143834816888963073, 143834816889028608, 143834812627549952, 143834812627484927,
    143834812627484673, 143834812627484928, 143834812610838527, 143834812610838273,
    143834812610772992, 143834812610773249, 143834812610903808, 143834812610838783,
    143834812644261888, 143834812627550209, 143834812627550464, 143834812627615744,
    143834821200707583, 143834821200707329, 143834821200642559, 143834821200642305,
    143834821183930368, 143834821184061439, 143834821184061185, 143834821183996415,
    143834821183996161, 143834816922451968, 143834816905805568, 143834816905740543,
    143834816905740289, 143834816939294719, 143834816939294465, 143834816939229695,
    143834816939229441, 143834816922517504, 143834816922648575, 143834816922648321,
    143834816922583551, 143834816922583297, 72339069014573056, 72339068997926656,
    72339068997861377, 72339068997861887, 72339068997861632, 72339068997926912,
    72339064736448256, 72339064736382977, 72339064736383232, 72339064719671296,
    72339064719671807, 72339064719671553, 72339064719737088, 72339064719737089,
    72339064753160192, 72339064736513792, 72339064736448767, 72339064736448513,
    72339064736448768, 72339064736514048, 72337973797977856, 72337973797912831,
    72337973797912577, 72337973797912832, 72337973781266431, 72337973781200896,
    72337973781201407, 72337973781201153, 72337973781331712, 72337973781266687,
    72337973781266433, 72337973781266688, 72337969519788031, 72337969519722496,
    72337969503076351, 72337969503076096, 72337969503011071, 72337969503010816,
    72337969503010817, 72337969503011072, 72337969503141633, 72337969503076352,
    72337969536499967, 72337969536499713, 72337969519853313, 72337969519788032,
    72337969519788543, 72337969519918848, 72337969519853823, 72337969519853569,
    72337969519853824, 72337978092879872, 72337978076233472, 72337978076168447,
    72337978076168448, 72337978076233728, 72337973814690047, 72337973814689793,
    72337973814690048, 72337973798043647, 72337973798043393, 72337973797978112,
    72337973797978623, 72337973798043649, 72337973798043904, 72337973831467008,
    72337973814755583, 72337973814755329, 72337973814821120, 72058693549555456,
    72058693549490431, 72058693549490177, 72058693532844031, 72058693532778496,
    72058693532779007, 72058693532844033, 72058689271365631, 72058689271300353,
    72058689254653696, 72058689254588671, 72058689254588416, 72058689254588417,
    72058689254588927, 72058689254588672, 72058689254719487, 72058689254719232,
    72058689254719233, 72058689254653952, 72058689288077567, 72058689288077313,
    72058689271430913, 72058689271365887, 72058689271365632, 72058689271366143,
    72058689271496448, 72058689271431424, 72057598332895231, 72057598332829696,
    72057598332830207, 72057598332829953, 72057598316183551, 72057598316183296,
    72057598316118271, 72057598316118016, 72057598316118017, 72057598316118272,
    72057598316248832, 72057598316183552, 72057598316183808, 72057598316183809,
    72057594054704896, 72057594054639871, 72057594054639616, 72057594054639617,
    72057594054639872, 72057594037993471, 72057594037993216, 72057594037993217,
    72057594037928191, 72057594037927936, 72057594037927937, 72057594037928447,
    72057594037928192, 72057594037928193, 72057594038058752, 72057594037993727,
    72057594037993472, 72057594037993473, 72057594037993728, 72057594071482112,
    72057594071416832, 72057594071417343, 72057594054770432, 72057594054770433,
    72057594054705407, 72057594054705152, 72057594054705153, 72057594054705408,
    72057594054705409, 72057594054836223, 72057594054835969, 72057594054770688,
    72057594054771199, 72057594054770945, 72057602627862272, 72057602627797247,
    72057602611150847, 72057602611085312, 72057602611085568, 72057602611216383,
    72057602611150849, 72057602611151104, 72057598349606912, 72057598349607423,
    72057598349607168, 72057598332960512, 72057598332960513, 72057598332895232,
    72057598332895233, 72057598332895488, 72057598332960768, 72057598332961279,
    72057598366449409, 72057598366384383, 72057598366384384, 72057598366384385,
    72057598349737729, 72057598349672703, 72057598349672448, 72057598349738239,
    72057598349737985, 72057598349738240, 72059793061117952, 72059793044406273,
    72059793044406528, 72059793044471808, 72059788782993152, 72059788782927873,
    72059788766281727, 72059788766281473, 72059788766216192, 72059788766216193,
    72059788766216449, 72059788766281983, 72059788766281728, 72059788799705088,
    72059788783058688, 72059788782993409, 72059788782993664, 72059788783058944,
    72058697844457727, 72058697844457473, 72058697844457728, 72058697827811327,
    72058697827811073, 72058697827745792, 72058697827746303, 72058697827746049,
    72058697827876863, 72058697827876608, 72058697827811583, 72058697827811329,
    72058693566332927, 72058693566332673, 72058693566267392, 72058693566267903,
    72058693566267649, 72058693549620992, 72058693549555967, 72058693549555712,
    72058693549555713, 72058693549555968, 72058693549686529, 72058693549621248,
    72058693549621249, 72058693549621505, 72058693583109888, 72058693583044863,
    72058693566398463, 72058693566398209, 72058693566332928, 72058693566333185,
    72058693566463744, 72058693566398465, 72058702139424768, 72058702122713088,
    72058702122778624, 72058697861234943, 72058697861234689, 72058697861234944,
    72058697844588543, 72058697844523008, 72058697844523519, 72058697844653824,
    72058697878011904, 72058697861365504, 72058697861300479, 72058697861300224,
    72058697861300225, 72620543991349247, 72620543991348993, 72620543991284223,
    72620543991283969, 72620543974572032, 72620543974703103, 72620543974702849,
    72620543974638079, 72620543974637825, 72620539713093632, 72620539696447232,
    72620539696382207, 72620539696381953, 72620539696382208, 72620539729936383,
    72620539729936129, 72620539729871359, 72620539729871105, 72620539713159168,
    72620539713290239, 72620539713289985, 72620539713225215, 72620539713224961,
    72619448774623232, 72619448774623488, 72619448757976832, 72619448757911807,
    72619448757911553, 72619448757911808, 72619448757911809, 72619444496433153,
    72619444496433408, 72619444479786752, 72619444479721472, 72619444479721983,
    72619444479721729, 72619444479852288, 72619444479787263, 72619444513210368,
    72619444496564223, 72619444496563969, 72619444496498689, 72619444496498944,
    72619453069655809, 72619453069591039, 72619453069590785, 72619453052944383,
    72619453052879104, 72619453053009665, 72619453052944895, 72619453052944641,
    72619448791400448, 72619448774754048, 72619448774688769, 72619448774689024,
    72619448774754304, 72619448808243199, 72619448808242945, 72619448808178175,
    72619448808177921, 72619448791465984, 72619448791597055, 72619448791596801,
    72619448791532031, 72619448791531777, 72340168509489408, 72340168509554688,
    72340164248076032, 72340164248011007, 72340164231364607, 72340164231299327,
    72340164231299072, 72340164231299583, 72340164231299329, 72340164231429888,
    72340164231364608, 72340164231364609, 72340164231365119, 72340164231364864,
    72340164264787968, 72339073309540353, 72339073309540608, 72339073292894207,
    72339073292893953, 72339073292828672, 72339073292829183, 72339073292894209,
    72339073292894464, 72339069031415553, 72339069031350272, 72339069014703872,
    72339069014638847, 72339069014638592, 72339069014638593, 72339069014638848,
    72339069014704128, 72339069014704385, 72339069048192768, 72339069048127743,
    72339069048127488, 72339069048127489, 72339069048127744, 72339069031481089,
    72339069031415808, 72339069031416319, 72339077604507648, 72339077587861248,
    72339077587795969, 72339077587796225, 72339077587926784, 72339077587861504,
    72339073326317823, 72339073326317569, 72339073326317825, 72339073309671169,
    72339073309605888, 72339073309605889, 72339073309606399, 72339073309736959,
    72339073309736705, 72339073343094785, 72339073326448639, 72339073326383104,
    72339073326383105, 72339073326383360, 72339073326513920, 72339073326448895,
    72339073326448641, 72341268037894143, 72341268037893889, 72341268037829119,
    72341268037828865, 72341268021247999, 72341268021247745, 72341268021182975,
    72341268021182721, 72341263742992128, 72341263742927103, 72341263742926849,
    72341263742927104, 72341263776481279, 72341263776481025, 72341263776416255,
    72341263776416001, 72341263759704064, 72341263759835135, 72341263759834881,
    72341263759770111, 72341263759769857, 72340172821168128, 72340172804456703,
    72340172804456704, 72340172804587264, 72340172804521984, 72340168543043328,
    72340168526331903, 72340168526266368, 72340168526266625, 72340168526397184,
    72340168526331905, 72340168526332160, 72340168543109119, 72340168543043585,
    72340177116200959, 72340177116200705, 72340177116135935, 72340177116135681,
    72340177099554815, 72340177099554561, 72340177099489791, 72340177099489537,
    72340172821298944, 72340172821233919, 72340172821233665, 72340172854788095,
    72340172854787841, 72340172854723071, 72340172854722817, 72340172838010880,
    72340172838141951, 72340172838141697, 72340172838076927, 72340172838076673,
};

__device__ const unsigned int dev_iq3s_grid[512] = {
    16843009u, 16843011u, 16843013u, 16843019u,
    16843023u, 16843521u, 16843523u, 16843525u,
    16843529u, 16843533u, 16844033u, 16844035u,
    16844043u, 16844551u, 16845057u, 16845061u,
    16845067u, 16845071u, 16845571u, 16845575u,
    16846081u, 16846085u, 16846595u, 16846601u,
    16846607u, 16974081u, 16974083u, 16974085u,
    16974089u, 16974593u, 16974595u, 16974603u,
    16975105u, 16975111u, 16975119u, 16975619u,
    16975627u, 16976137u, 16977155u, 16977163u,
    16977669u, 17105153u, 17105155u, 17105163u,
    17105167u, 17105665u, 17105671u, 17105677u,
    17106179u, 17106187u, 17106689u, 17106697u,
    17107205u, 17107211u, 17107215u, 17107715u,
    17107719u, 17108737u, 17108743u, 17236231u,
    17236739u, 17236747u, 17237249u, 17237253u,
    17237763u, 17237767u, 17237773u, 17238281u,
    17238785u, 17238789u, 17239311u, 17239811u,
    17239819u, 17367297u, 17367815u, 17367823u,
    17368323u, 17368329u, 17368837u, 17369345u,
    17369351u, 17369859u, 17370881u, 17498373u,
    17498377u, 17499393u, 17499397u, 17499405u,
    17499911u, 17500419u, 17500427u, 17500431u,
    17501453u, 17501959u, 17629453u, 17629955u,
    17629959u, 17630979u, 17632005u, 17633027u,
    17760513u, 17760517u, 17760521u, 17761537u,
    17761541u, 17761549u, 17762055u, 17763073u,
    17763081u, 50397441u, 50397443u, 50397445u,
    50397449u, 50397953u, 50397955u, 50397959u,
    50397963u, 50397967u, 50398465u, 50398469u,
    50398979u, 50398985u, 50398989u, 50400009u,
    50400013u, 50400515u, 50401029u, 50528513u,
    50528515u, 50528519u, 50528525u, 50529025u,
    50529033u, 50529539u, 50530049u, 50530055u,
    50530563u, 50531073u, 50531077u, 50532097u,
    50532109u, 50659585u, 50660101u, 50660107u,
    50660111u, 50660609u, 50660617u, 50661125u,
    50661633u, 50661639u, 50662155u, 50662657u,
    50663173u, 50790659u, 50790665u, 50790671u,
    50791169u, 50791175u, 50791683u, 50791695u,
    50792193u, 50792201u, 50792707u, 50793733u,
    50794241u, 50921735u, 50921739u, 50922245u,
    50922249u, 50923267u, 50923271u, 50923781u,
    50923789u, 50924289u, 50924297u, 51052803u,
    51053313u, 51053319u, 51053827u, 51054337u,
    51054341u, 51055363u, 51184897u, 51184905u,
    51184911u, 51185929u, 51185933u, 51314947u,
    51314951u, 51315457u, 51315461u, 51315971u,
    51316491u, 51316995u, 51318021u, 51318529u,
    83951873u, 83951875u, 83951879u, 83951883u,
    83951887u, 83952385u, 83952389u, 83952393u,
    83952397u, 83952899u, 83952903u, 83952911u,
    83953409u, 83953413u, 83953923u, 83953927u,
    83953931u, 83954433u, 83954437u, 83954959u,
    83955457u, 83955463u, 83955467u, 84082945u,
    84082949u, 84083457u, 84083463u, 84083471u,
    84083973u, 84083979u, 84084483u, 84084489u,
    84084997u, 84085507u, 84214019u, 84214025u,
    84214031u, 84215043u, 84215047u, 84215553u,
    84215567u, 84216067u, 84216583u, 84216591u,
    84217603u, 84217609u, 84345089u, 84345093u,
    84345099u, 84345603u, 84346117u, 84346121u,
    84346627u, 84346631u, 84347141u, 84347649u,
    84348173u, 84476163u, 84476175u, 84477185u,
    84477191u, 84477701u, 84477707u, 84478211u,
    84479749u, 84479755u, 84607241u, 84607747u,
    84608261u, 84608783u, 84609281u, 84609799u,
    84610817u, 84738305u, 84738309u, 84738319u,
    84739331u, 84740875u, 84741379u, 84869387u,
    84869891u, 84870413u, 84870913u, 84871431u,
    84871937u, 117506309u, 117506819u, 117506823u,
    117506827u, 117506831u, 117507333u, 117507843u,
    117507847u, 117507851u, 117508357u, 117508361u,
    117508367u, 117508867u, 117509383u, 117509891u,
    117637379u, 117637383u, 117637387u, 117637897u,
    117638403u, 117638407u, 117639425u, 117640449u,
    117640965u, 117640973u, 117768449u, 117768965u,
    117769473u, 117769989u, 117769993u, 117771009u,
    117899523u, 117900033u, 117900041u, 117900547u,
    117900551u, 117900559u, 117901057u, 117901571u,
    117901575u, 117901583u, 117902091u, 117903111u,
    118030599u, 118031107u, 118031117u, 118031621u,
    118032131u, 118033157u, 118033665u, 118033673u,
    118161667u, 118162177u, 118162181u, 118162699u,
    118163205u, 118163721u, 118164237u, 118165255u,
    118293261u, 118294787u, 118423811u, 118423815u,
    118424833u, 118424837u, 118425355u, 151060737u,
    151060745u, 151061253u, 151061761u, 151061769u,
    151061775u, 151062277u, 151062787u, 151063297u,
    151064321u, 151191813u, 151191823u, 151192323u,
    151192327u, 151192837u, 151193345u, 151193355u,
    151193863u, 151194371u, 151194379u, 151322883u,
    151322887u, 151323393u, 151323403u, 151323907u,
    151324423u, 151324929u, 151325455u, 151325957u,
    151326465u, 151453961u, 151454467u, 151454471u,
    151454977u, 151454981u, 151455491u, 151455499u,
    151585025u, 151585029u, 151586057u, 151586575u,
    151587073u, 151588611u, 151716107u, 151716111u,
    151717123u, 151719173u, 151847687u, 151848713u,
    151850241u, 151978753u, 151978763u, 151979777u,
    151980295u, 151980803u, 184615173u, 184615681u,
    184615689u, 184616197u, 184617217u, 184617225u,
    184617231u, 184617733u, 184618253u, 184618761u,
    184746243u, 184746247u, 184746251u, 184746757u,
    184747267u, 184747781u, 184749829u, 184877313u,
    184877827u, 184878343u, 184878849u, 184878861u,
    184879879u, 185008389u, 185008399u, 185008897u,
    185009423u, 185010441u, 185010947u, 185011467u,
    185011975u, 185139459u, 185139465u, 185140481u,
    185140997u, 185141517u, 185271045u, 185271565u,
    185273091u, 185273095u, 185403653u, 185532677u,
    185532681u, 185533701u, 218170115u, 218170119u,
    218170123u, 218171139u, 218171143u, 218172673u,
    218300673u, 218301697u, 218301711u, 218303753u,
    218432261u, 218433289u, 218433797u, 218434315u,
    218434821u, 218435329u, 218562817u, 218563337u,
    218563843u, 218564865u, 218694923u, 218695943u,
    218696965u, 218824961u, 218824967u, 218826505u,
    218828033u, 218956043u, 218958081u, 219087619u,
    219087623u, 251724033u, 251724041u, 251724047u,
    251725057u, 251725061u, 251725581u, 251726081u,
    251726601u, 251727109u, 251855109u, 251855619u,
    251856137u, 251857159u, 251857163u, 251986179u,
    251986185u, 251986689u, 251986701u, 251987203u,
    251987713u, 251988739u, 252117253u, 252118789u,
    252118795u, 252119815u, 252248323u, 252248331u,
    252248839u, 252249345u, 252250881u, 252380421u,
    252381445u, 252510469u, 252512003u, 252641537u,
};

__device__ const uint32_t dev_iq3xxs_grid[256] = {
    67372036u, 67372052u, 67372068u, 67374092u, 67374108u, 67374142u, 67376132u, 67376148u,
    67378188u, 67380244u, 67386908u, 67386924u, 67896332u, 67896348u, 67898372u, 67898388u,
    67900428u, 67900460u, 67902468u, 67902484u, 67904524u, 67906596u, 67911172u, 68420612u,
    68420628u, 68420644u, 68422668u, 68424708u, 68424724u, 68426764u, 68426780u, 68426814u,
    68430860u, 68430910u, 68435500u, 68944908u, 68944958u, 68946948u, 68946964u, 68949036u,
    68959748u, 69471260u, 69475390u, 69477412u, 69479486u, 69484060u, 69484076u, 69993484u,
    69993534u, 69999636u, 70003732u, 70523948u, 70530084u, 71175172u, 71175204u, 71175220u,
    71181340u, 71185420u, 201589772u, 201589788u, 201591812u, 201591828u, 201593868u, 201593884u,
    201595908u, 201595924u, 201595940u, 201598014u, 201600004u, 202114052u, 202114068u, 202116108u,
    202118148u, 202118164u, 202638348u, 202638364u, 202640388u, 202640404u, 202642444u, 202644484u,
    202653204u, 203162628u, 203162644u, 203166724u, 203168780u, 203170868u, 203174964u, 203686924u,
    203686956u, 203697156u, 204215300u, 204215332u, 204219444u, 204226060u, 204735532u, 205394964u,
    205399044u, 335807492u, 335807508u, 335809548u, 335809564u, 335811588u, 335811604u, 335811636u,
    335813644u, 335815700u, 336331788u, 336331804u, 336331820u, 336333828u, 336333844u, 336335884u,
    336337924u, 336344092u, 336344126u, 336346628u, 336856068u, 336856084u, 336858124u, 336858174u,
    336860164u, 336860180u, 336862270u, 336864260u, 336866348u, 337380364u, 337382404u, 337382436u,
    337395204u, 337395236u, 337910828u, 337914908u, 338428956u, 338433086u, 338437132u, 338443812u,
    339608588u, 339608604u, 339610676u, 339616812u, 470025228u, 470027268u, 470027284u, 470029324u,
    470029340u, 470035460u, 470037548u, 470040084u, 470549508u, 470549524u, 470553604u, 470555660u,
    470557732u, 470557748u, 471073804u, 471073820u, 471075844u, 471077932u, 471084052u, 471088660u,
    471600140u, 471604252u, 472128516u, 472130622u, 472137236u, 472646660u, 472646708u, 472650772u,
    472656940u, 473173028u, 473177140u, 473183260u, 473832476u, 473838596u, 604242980u, 604245054u,
    604249132u, 604249150u, 604253212u, 604253246u, 604782116u, 605295620u, 605297726u, 605299716u,
    605303812u, 605303860u, 605815870u, 605824044u, 606340132u, 606350348u, 606352420u, 606868524u,
    606872604u, 606879236u, 608044076u, 608046084u, 608046100u, 608050180u, 738462740u, 738468876u,
    738475524u, 738984964u, 738985012u, 738989108u, 738995244u, 739511332u, 739515412u, 739524116u,
    740033556u, 740043804u, 740559876u, 740561948u, 740561982u, 740572692u, 741082132u, 741088268u,
    741616644u, 742265892u, 742269972u, 872682532u, 872686628u, 872686644u, 872690724u, 873206796u,
    873214988u, 873729086u, 873739300u, 874257412u, 874257460u, 874783780u, 875299884u, 875310100u,
    875830300u, 876479516u, 876483596u, 1040450588u, 1040450604u, 1040450622u, 1040452612u, 1040456724u,
    1040460820u, 1040978996u, 1040983044u, 1041501204u, 1041507372u, 1041509396u, 1042023428u, 1042025516u,
    1042029596u, 1042035716u, 1042551820u, 1042555916u, 1043072004u, 1043072020u, 1043076132u, 1043602436u,
};
__device__ const uint8_t dev_ksigns_iq2xs[128] = {
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15,
    144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159,
    160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175,
    48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207,
    80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
    96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

__device__ const unsigned long long dev_iq2xxs_grid[256] = {
    578721382704613384ull, 578721382704613419ull, 578721382704617753ull, 578721382704622344ull,
    578721382704622379ull, 578721382705727513ull, 578721382705731848ull, 578721382706907144ull,
    578721382706907179ull, 578721382706916104ull, 578721382706916139ull, 578721382989826073ull,
    578721382989830408ull, 578721382990940168ull, 578721382990949128ull, 578721382992119833ull,
    578721382992124168ull, 578721383291815944ull, 578721383291815979ull, 578721383291824939ull,
    578721383294109739ull, 578721455719057433ull, 578721455719061768ull, 578721455720171528ull,
    578721455720175897ull, 578721456004270088ull, 578721456306264328ull, 578721456307383048ull,
    578721533028468744ull, 578721533028468779ull, 578721533030762539ull, 578721533615671339ull,
    578740074402285593ull, 578740074402289928ull, 578740074403399688ull, 578740074404579353ull,
    578740074404583688ull, 578740074687498248ull, 578740074687498283ull, 578740074687507208ull,
    578740074689792008ull, 578740074989488153ull, 578740074989492488ull, 578740074990602248ull,
    578740074991786248ull, 578740147416729608ull, 578740147416729643ull, 578740147416738568ull,
    578740147419023368ull, 578740147701946667ull, 578740147704245017ull, 578740148003932168ull,
    578740148005046297ull, 578740224726149913ull, 578740224727255048ull, 578740225011353608ull,
    578740225313347848ull, 578740225315641608ull, 578759865611585544ull, 578759865611589913ull,
    578759865611594504ull, 578759865612704008ull, 578759865613888264ull, 578759865896798233ull,
    578759865896802568ull, 578759865897912328ull, 578759865897912363ull, 578759866198797064ull,
    578759938626033928ull, 578759938911242248ull, 578760015935440939ull, 578760015936559368ull,
    583506457308694553ull, 583506457308698888ull, 583506457309808648ull, 583506457310988313ull,
    583506457593907208ull, 583506457596200968ull, 583506457895901448ull, 583506457897011208ull,
    583506457897015577ull, 583506530323138568ull, 583506530323147528ull, 583506530325432328ull,
    583506530609465352ull, 583506530609474347ull, 583506530910341128ull, 583506607634848008ull,
    583506607917766937ull, 583525149006366728ull, 583525149006375688ull, 583525149008660488ull,
    583525149008664857ull, 583525149291588377ull, 583525149593569288ull, 583525222021933832ull,
    583525222308317227ull, 583525299330222088ull, 583525299331340587ull, 583544940215666713ull,
    583544940215671048ull, 583544940216780808ull, 583544940500879368ull, 583544940802869273ull,
    583545013230110728ull, 583545013230115097ull, 583545013819607048ull, 583545090825848857ull,
    588573006889486344ull, 588573006889486379ull, 588573006889495339ull, 588573007174703368ull,
    588573007176992793ull, 588573007476688904ull, 588573007476688939ull, 588573079906233113ull,
    588573080189152008ull, 588573157213341704ull, 588573157213341739ull, 588591698587158553ull,
    588591698587162888ull, 588591698588272648ull, 588591698872371208ull, 588591698873489707ull,
    588591771601602568ull, 588591771886815257ull, 588591771889113352ull, 588591849499330568ull,
    588611489796467464ull, 588611489798752264ull, 588611490384779528ull, 588611640405530888ull,
    1803700481349388313ull, 1803700481349392648ull, 1803700481350502408ull, 1803700481350511368ull,
    1803700481351682073ull, 1803700481351686408ull, 1803700481634600968ull, 1803700481634609928ull,
    1803700481635719467ull, 1803700481636894728ull, 1803700481936590873ull, 1803700481936595208ull,
    1803700481937704968ull, 1803700554363832328ull, 1803700554366126088ull, 1803700554651338777ull,
    1803700554951034888ull, 1803700554951039257ull, 1803700631673243673ull, 1803700631674357768ull,
    1803700631958465288ull, 1803700631959574827ull, 1803700631960759048ull, 1803719173047060488ull,
    1803719173047069448ull, 1803719173049354248ull, 1803719173634263048ull, 1803719173635386137ull,
    1803719246062618667ull, 1803719246063802632ull, 1803719323370915848ull, 1803738964256360473ull,
    1803738964256364808ull, 1803738964257474568ull, 1803738964541573128ull, 1803738964541577497ull,
    1803739037270804488ull, 1803739037557140232ull, 1803739037558310937ull, 1803739037858007083ull,
    1803739114865432857ull, 1803739115168532488ull, 1808485555953469448ull, 1808485555953478408ull,
    1808485555954583577ull, 1808485555954592537ull, 1808485555955763208ull, 1808485556540672008ull,
    1808485556540680968ull, 1808485628967917832ull, 1808485629253126187ull, 1808485629557414152ull,
    1808485706865641497ull, 1808504248239458312ull, 1808504248239458347ull, 1808504320665594667ull,
    1808504397974997017ull, 1808504398261328136ull, 1808524038860441608ull, 1808524038861555737ull,
    1808524038861564697ull, 1808524039147952392ull, 1808524112160098312ull, 1808524189184305928ull,
    1813552105534265608ull, 1813552105535375368ull, 1813552105819473928ull, 1813552105821776648ull,
    1813552178548705288ull, 1813552178835036441ull, 1813552255859239688ull, 1813552256145623048ull,
    1813570797231933448ull, 1813570797231937817ull, 1813570870247491592ull, 1813570870247491627ull,
    1813570870833584392ull, 1813590588726446123ull, 3100737174032091144ull, 3100737174032091179ull,
    3100737174032100139ull, 3100737174317303833ull, 3100737174619293739ull, 3100737247046539528ull,
    3100737247047658248ull, 3100737247331747848ull, 3100737324357060633ull, 3100755865729763353ull,
    3100755865729767688ull, 3100755865730877448ull, 3100755865730881817ull, 3100755866014976008ull,
    3100755866017269768ull, 3100755938744207368ull, 3100755939029424427ull, 3100755939332528392ull,
    3100756016053627673ull, 3100756016338831368ull, 3100756016341125128ull, 3100775656939063339ull,
    3100775729953511688ull, 3100775807264032793ull, 3105522248636176648ull, 3105522248637286408ull,
    3105522248638470408ull, 3105522248921384968ull, 3105522249225668633ull, 3105522321651734827ull,
    3105522322237818888ull, 3105522399245244697ull, 3105540940333844488ull, 3105540940336138283ull,
    3105540940619061512ull, 3105541013634615321ull, 3105560732130347033ull, 3105560804559882248ull,
    3110588798216964139ull, 3110588798503290888ull, 3110588798804171033ull, 3110588871231417113ull,
    3110588948540819464ull, 3110607489915759368ull, 3110627281410263048ull, 3110627354138384648ull,
};

__device__ const unsigned long long dev_iq2xs_grid[512] = {
    578721382704613384ull, 578721382704613419ull, 578721382704617753ull, 578721382704622344ull,
    578721382704622379ull, 578721382705727513ull, 578721382705731848ull, 578721382705731883ull,
    578721382705736473ull, 578721382706907144ull, 578721382706907179ull, 578721382706911513ull,
    578721382706916104ull, 578721382989826073ull, 578721382989830408ull, 578721382989830443ull,
    578721382989835033ull, 578721382990940168ull, 578721382990940203ull, 578721382990944537ull,
    578721382990949128ull, 578721382992119833ull, 578721382992124168ull, 578721383291815944ull,
    578721383291815979ull, 578721383291820313ull, 578721383291824904ull, 578721383292930073ull,
    578721383292934408ull, 578721383292939033ull, 578721383294109704ull, 578721455719057433ull,
    578721455719061768ull, 578721455719061803ull, 578721455719066393ull, 578721455720171528ull,
    578721455720171563ull, 578721455720175897ull, 578721455720180488ull, 578721455720180523ull,
    578721455721351193ull, 578721455721355528ull, 578721456004270088ull, 578721456004270123ull,
    578721456004274457ull, 578721456004279048ull, 578721456005384217ull, 578721456005388552ull,
    578721456006563848ull, 578721456006572808ull, 578721456306259993ull, 578721456306264328ull,
    578721456307374088ull, 578721533028468744ull, 578721533028468779ull, 578721533028473113ull,
    578721533028477704ull, 578721533029582873ull, 578721533029587208ull, 578721533030762504ull,
    578721533313681433ull, 578721533313685768ull, 578721533314795528ull, 578721533314799897ull,
    578721533615671304ull, 578721533615680299ull, 578740074402285593ull, 578740074402289928ull,
    578740074402289963ull, 578740074402294553ull, 578740074403399688ull, 578740074403399723ull,
    578740074403404057ull, 578740074403408648ull, 578740074404579353ull, 578740074404583688ull,
    578740074687498248ull, 578740074687498283ull, 578740074687502617ull, 578740074687507208ull,
    578740074688612377ull, 578740074688616712ull, 578740074688616747ull, 578740074689792008ull,
    578740074989488153ull, 578740074989492488ull, 578740074990602248ull, 578740147416729608ull,
    578740147416729643ull, 578740147416733977ull, 578740147416738568ull, 578740147417843737ull,
    578740147417848072ull, 578740147419023368ull, 578740147701942297ull, 578740147701946632ull,
    578740147703056392ull, 578740147704236057ull, 578740148003932168ull, 578740224726140953ull,
    578740224726145288ull, 578740224727255048ull, 578740224728439083ull, 578740225011353608ull,
    578740225011353643ull, 578740225313347848ull, 578759865611585544ull, 578759865611585579ull,
    578759865611589913ull, 578759865611594504ull, 578759865611594539ull, 578759865612699673ull,
    578759865612704008ull, 578759865613879304ull, 578759865613883673ull, 578759865896798233ull,
    578759865896802568ull, 578759865897912328ull, 578759865897921288ull, 578759866198788104ull,
    578759866201081864ull, 578759866201090859ull, 578759938626029593ull, 578759938626033928ull,
    578759938627143688ull, 578759938911242248ull, 578759939213232153ull, 578759939213241113ull,
    578760015935440904ull, 578760015937734664ull, 578760015937743624ull, 578760016523761963ull,
    578760016524937224ull, 583506457308694553ull, 583506457308698888ull, 583506457308698923ull,
    583506457308703513ull, 583506457309808648ull, 583506457309808683ull, 583506457309813017ull,
    583506457309817608ull, 583506457310988313ull, 583506457310992648ull, 583506457593907208ull,
    583506457593907243ull, 583506457593911577ull, 583506457593916168ull, 583506457595021337ull,
    583506457595025672ull, 583506457596200968ull, 583506457596209963ull, 583506457895897113ull,
    583506457895901448ull, 583506457897011208ull, 583506530323138568ull, 583506530323138603ull,
    583506530323142937ull, 583506530323147528ull, 583506530324252697ull, 583506530324257032ull,
    583506530325432328ull, 583506530608351257ull, 583506530608355592ull, 583506530609465352ull,
    583506530910341128ull, 583506530911459592ull, 583506530911459627ull, 583506607632549913ull,
    583506607632554248ull, 583506607632554283ull, 583506607633664008ull, 583506607917762568ull,
    583506607920056328ull, 583525149006366728ull, 583525149006366763ull, 583525149006371097ull,
    583525149006375688ull, 583525149007480857ull, 583525149007485192ull, 583525149008660488ull,
    583525149291579417ull, 583525149291583752ull, 583525149291588377ull, 583525149292693512ull,
    583525149293877512ull, 583525149593569288ull, 583525222020810777ull, 583525222020815112ull,
    583525222021924872ull, 583525222306023432ull, 583525299330222088ull, 583525299331340552ull,
    583525299615443737ull, 583544940215666713ull, 583544940215671048ull, 583544940216780808ull,
    583544940216780843ull, 583544940500879368ull, 583544940501997832ull, 583544940802873643ull,
    583545013230110728ull, 583545013230115097ull, 583545013517621547ull, 583545090825848857ull,
    583545091129027353ull, 588573006889486344ull, 588573006889486379ull, 588573006889490713ull,
    588573006889495304ull, 588573006889495339ull, 588573006890600473ull, 588573006890604808ull,
    588573006891780104ull, 588573007174699033ull, 588573007174703368ull, 588573007175813128ull,
    588573007476688904ull, 588573007478982664ull, 588573079903930393ull, 588573079903934728ull,
    588573079905044488ull, 588573080189143048ull, 588573080189152008ull, 588573080191441177ull,
    588573157213341704ull, 588573157215635499ull, 588573157800544264ull, 588573157802846984ull,
    588591698587158553ull, 588591698587162888ull, 588591698588272648ull, 588591698589461273ull,
    588591698872371208ull, 588591771601602568ull, 588591771886815257ull, 588591771887929387ull,
    588591772189928217ull, 588591848911013913ull, 588591848912137003ull, 588591849500514603ull,
    588611489796458504ull, 588611489796467464ull, 588611489796467499ull, 588611489798752264ull,
    588611490082789657ull, 588611490383670024ull, 588611490385954859ull, 588611563098417928ull,
    588611563399219208ull, 588611640120322824ull, 588611640122607624ull, 588611640707516459ull,
    588611640707525384ull, 588611640707525419ull, 1803700481349388313ull, 1803700481349392648ull,
    1803700481349392683ull, 1803700481349397273ull, 1803700481350502408ull, 1803700481350502443ull,
    1803700481350506777ull, 1803700481350511368ull, 1803700481351682073ull, 1803700481351686408ull,
    1803700481634600968ull, 1803700481634601003ull, 1803700481634605337ull, 1803700481634609928ull,
    1803700481634609963ull, 1803700481635715097ull, 1803700481635719432ull, 1803700481636894728ull,
    1803700481636899097ull, 1803700481936590873ull, 1803700481936595208ull, 1803700481937704968ull,
    1803700554363832328ull, 1803700554363832363ull, 1803700554363836697ull, 1803700554363841288ull,
    1803700554364946457ull, 1803700554364950792ull, 1803700554366126088ull, 1803700554649045017ull,
    1803700554649049352ull, 1803700554650159112ull, 1803700554951034888ull, 1803700554951039257ull,
    1803700554953328683ull, 1803700631673243673ull, 1803700631673248008ull, 1803700631674357768ull,
    1803700631674357803ull, 1803700631675546393ull, 1803700631958456328ull, 1803719173047060488ull,
    1803719173047060523ull, 1803719173047064857ull, 1803719173047069448ull, 1803719173048174617ull,
    1803719173048178952ull, 1803719173048183577ull, 1803719173049354248ull, 1803719173332273177ull,
    1803719173332277512ull, 1803719173333387272ull, 1803719173634263048ull, 1803719173635381512ull,
    1803719246061504537ull, 1803719246061508872ull, 1803719246062618632ull, 1803719246063802632ull,
    1803719246346717192ull, 1803719246649830187ull, 1803719323370915848ull, 1803719323370924843ull,
    1803719323656132872ull, 1803719323657242632ull, 1803738964256360473ull, 1803738964256364808ull,
    1803738964257474568ull, 1803738964541573128ull, 1803738964541577497ull, 1803738964542691592ull,
    1803738964543866923ull, 1803739037270804488ull, 1803739037271918617ull, 1803739037556021512ull,
    1803739037557131272ull, 1803739037558319897ull, 1803739114580220168ull, 1808485555953469448ull,
    1808485555953469483ull, 1808485555953473817ull, 1808485555953478408ull, 1808485555954583577ull,
    1808485555954587912ull, 1808485555955763208ull, 1808485555955772168ull, 1808485556238682137ull,
    1808485556238686472ull, 1808485556239796232ull, 1808485556540672008ull, 1808485628967913497ull,
    1808485628967917832ull, 1808485628969027592ull, 1808485628969031961ull, 1808485629253126152ull,
    1808485629253126187ull, 1808485706277324808ull, 1808485706562541832ull, 1808485706866830123ull,
    1808504247651141657ull, 1808504247651145992ull, 1808504247652255752ull, 1808504247653435417ull,
    1808504247936354312ull, 1808504247938648072ull, 1808504248238344217ull, 1808504248240637977ull,
    1808504320665585672ull, 1808504320665594632ull, 1808504321252788232ull, 1808504321252797192ull,
    1808504397977290777ull, 1808504398262512392ull, 1808504398564493337ull, 1808524038860441608ull,
    1808524038861560072ull, 1808524039145654297ull, 1808524039146768392ull, 1808524039448767257ull,
    1808524111876008747ull, 1808524112160098312ull, 1808524112160098347ull, 1808524189771503897ull,
    1813552105534261273ull, 1813552105534265608ull, 1813552105535375368ull, 1813552105819473928ull,
    1813552105820592392ull, 1813552105821767723ull, 1813552106121468203ull, 1813552106123766553ull,
    1813552178548705288ull, 1813552255860414728ull, 1813552256143338283ull, 1813552256446433323ull,
    1813570797231933448ull, 1813570797233051947ull, 1813570870247491592ull, 1813570870531590152ull,
    1813570870531594521ull, 1813570870835878152ull, 1813590588441233433ull, 1813590588728748843ull,
    1813590661457975577ull, 1813590738765093163ull, 1813590739051419912ull, 1813590739052595243ull,
    3100737174032091144ull, 3100737174032091179ull, 3100737174032095513ull, 3100737174032100104ull,
    3100737174033205273ull, 3100737174033209608ull, 3100737174034384904ull, 3100737174034393899ull,
    3100737174317303833ull, 3100737174317308168ull, 3100737174318417928ull, 3100737174619293704ull,
    3100737174619293739ull, 3100737174621596424ull, 3100737174621596459ull, 3100737247046535193ull,
    3100737247046539528ull, 3100737247046539563ull, 3100737247047649288ull, 3100737247331747848ull,
    3100737247332861977ull, 3100737247332870937ull, 3100737324355946504ull, 3100737324358240264ull,
    3100737324943149064ull, 3100737324943149099ull, 3100737324945442824ull, 3100737324945451784ull,
    3100755865729763353ull, 3100755865729767688ull, 3100755865730877448ull, 3100755865730877483ull,
    3100755865730881817ull, 3100755866014976008ull, 3100755866017269768ull, 3100755866316974873ull,
    3100755938744207368ull, 3100755939029424392ull, 3100755939333708057ull, 3100756016054741768ull,
    3100756016341134123ull, 3100775656939063304ull, 3100775656939072264ull, 3100775656941361433ull,
    3100775657225399083ull, 3100775657526265864ull, 3100775657526265899ull, 3100775657528568584ull,
    3100775729953511723ull, 3100775807265212459ull, 3100775807850121224ull, 3100775807850130184ull,
    3100775807851239723ull, 3100775807852423944ull, 3105522248636172313ull, 3105522248636176648ull,
    3105522248637286408ull, 3105522248921384968ull, 3105522248922503467ull, 3105522249223379208ull,
    3105522321650616328ull, 3105522321652910123ull, 3105522321938127112ull, 3105522399246358827ull,
    3105522399547239193ull, 3105540940333844488ull, 3105540940333848857ull, 3105540940619061512ull,
    3105540940620171272ull, 3105540940620180232ull, 3105541013350591257ull, 3105541013936605192ull,
    3105541013936605227ull, 3105541090942912537ull, 3105560731829471257ull, 3105560732132645163ull,
    3105560804842810137ull, 3105560881868118297ull, 3105560882154506248ull, 3110588798216964104ull,
    3110588798216964139ull, 3110588798216973064ull, 3110588798216973099ull, 3110588798219257864ull,
    3110588798219266859ull, 3110588798806460424ull, 3110588871517734937ull, 3110588871517743897ull,
    3110588871820908843ull, 3110588948540819464ull, 3110588948540819499ull, 3110588948540828424ull,
    3110588948543122219ull, 3110588949128022024ull, 3110588949130315784ull, 3110607490199848968ull,
    3110607490502957337ull, 3110607640526002457ull, 3110607640826817288ull, 3110627281123945259ull,
    3110627281126230024ull, 3110627281126230059ull, 3110627281126238984ull, 3110627281713432584ull,
    3110627281713441544ull, 3110627354138384648ull, 3110627354725587208ull, 3110627354725587243ull,
    3110627431450094344ull, 3110627431450094379ull, 3110627432036108313ull, 3110627432037296939ull,
};

__device__ const unsigned long long dev_iq2s_grid[1024] = {
    578721382704613384ull, 578721382704613419ull, 578721382704617753ull, 578721382704622344ull,
    578721382704622379ull, 578721382705727513ull, 578721382705731848ull, 578721382705731883ull,
    578721382705736473ull, 578721382706907144ull, 578721382706907179ull, 578721382706911513ull,
    578721382706916104ull, 578721382989826073ull, 578721382989830408ull, 578721382989830443ull,
    578721382989835033ull, 578721382990940168ull, 578721382990940203ull, 578721382990944537ull,
    578721382990949128ull, 578721382992119833ull, 578721382992124168ull, 578721382992124203ull,
    578721382992128793ull, 578721383291815944ull, 578721383291815979ull, 578721383291820313ull,
    578721383291824904ull, 578721383292930073ull, 578721383292934408ull, 578721383294109704ull,
    578721383294114073ull, 578721383294118699ull, 578721455719057433ull, 578721455719061768ull,
    578721455719061803ull, 578721455719066393ull, 578721455720171528ull, 578721455720171563ull,
    578721455720175897ull, 578721455720180488ull, 578721455721351193ull, 578721455721355528ull,
    578721456004270088ull, 578721456004270123ull, 578721456004274457ull, 578721456004279048ull,
    578721456005384217ull, 578721456005388552ull, 578721456005388587ull, 578721456005393177ull,
    578721456006563848ull, 578721456006568217ull, 578721456006572808ull, 578721456306259993ull,
    578721456306264328ull, 578721456307374088ull, 578721456307374123ull, 578721456307378457ull,
    578721456308553753ull, 578721456308558088ull, 578721533028468744ull, 578721533028468779ull,
    578721533028473113ull, 578721533028477704ull, 578721533029582873ull, 578721533029587208ull,
    578721533030762504ull, 578721533030771499ull, 578721533313681433ull, 578721533313685768ull,
    578721533313685803ull, 578721533313690393ull, 578721533314795528ull, 578721533314799897ull,
    578721533615671304ull, 578721533615675673ull, 578721533615680299ull, 578721533616789768ull,
    578721533617965099ull, 578740074402285593ull, 578740074402289928ull, 578740074402289963ull,
    578740074402294553ull, 578740074403399688ull, 578740074403399723ull, 578740074403404057ull,
    578740074403408648ull, 578740074404579353ull, 578740074404583688ull, 578740074404583723ull,
    578740074404588313ull, 578740074687498248ull, 578740074687498283ull, 578740074687502617ull,
    578740074687507208ull, 578740074687507243ull, 578740074688612377ull, 578740074688616712ull,
    578740074688616747ull, 578740074688621337ull, 578740074689792008ull, 578740074689792043ull,
    578740074689796377ull, 578740074989488153ull, 578740074989492488ull, 578740074989492523ull,
    578740074989497113ull, 578740074990602248ull, 578740074990606617ull, 578740074990611208ull,
    578740074991781913ull, 578740074991786248ull, 578740147416729608ull, 578740147416729643ull,
    578740147416733977ull, 578740147416738568ull, 578740147416738603ull, 578740147417843737ull,
    578740147417848072ull, 578740147417848107ull, 578740147417852697ull, 578740147419023368ull,
    578740147419027737ull, 578740147419032328ull, 578740147701942297ull, 578740147701946632ull,
    578740147701946667ull, 578740147701951257ull, 578740147703056392ull, 578740147703056427ull,
    578740147703060761ull, 578740147703065352ull, 578740147704236057ull, 578740147704240392ull,
    578740148003932168ull, 578740148003932203ull, 578740148003936537ull, 578740148003941128ull,
    578740148005046297ull, 578740148005050632ull, 578740148006225928ull, 578740224726140953ull,
    578740224726145288ull, 578740224726145323ull, 578740224726149913ull, 578740224727255048ull,
    578740224727259417ull, 578740225011353608ull, 578740225011357977ull, 578740225011362568ull,
    578740225012467737ull, 578740225012472072ull, 578740225013647368ull, 578740225313343513ull,
    578740225313347848ull, 578740225314457608ull, 578759865611585544ull, 578759865611585579ull,
    578759865611589913ull, 578759865611594504ull, 578759865612699673ull, 578759865612704008ull,
    578759865612704043ull, 578759865612708633ull, 578759865613879304ull, 578759865613883673ull,
    578759865613888299ull, 578759865896798233ull, 578759865896802568ull, 578759865896802603ull,
    578759865896807193ull, 578759865897912328ull, 578759865897912363ull, 578759865897916697ull,
    578759865897921288ull, 578759865899091993ull, 578759865899096328ull, 578759866198788104ull,
    578759866198792473ull, 578759866199906568ull, 578759866201090859ull, 578759938626029593ull,
    578759938626033928ull, 578759938627143688ull, 578759938627143723ull, 578759938627148057ull,
    578759938627152648ull, 578759938628323353ull, 578759938911242248ull, 578759938911246617ull,
    578759938911251208ull, 578759938912356377ull, 578759938912360712ull, 578759938913536008ull,
    578759939213232153ull, 578759939214346248ull, 578760015935440904ull, 578760015936555033ull,
    578760015936559368ull, 578760015937734699ull, 578760015937743624ull, 578760015937743659ull,
    578760016221767688ull, 578760016523766553ull, 583506457308694553ull, 583506457308698888ull,
    583506457308698923ull, 583506457308703513ull, 583506457309808648ull, 583506457309808683ull,
    583506457309813017ull, 583506457309817608ull, 583506457310988313ull, 583506457310992648ull,
    583506457310992683ull, 583506457593907208ull, 583506457593907243ull, 583506457593911577ull,
    583506457593916168ull, 583506457595021337ull, 583506457595025672ull, 583506457595025707ull,
    583506457595030297ull, 583506457596200968ull, 583506457596201003ull, 583506457596205337ull,
    583506457596209928ull, 583506457895897113ull, 583506457895901448ull, 583506457895901483ull,
    583506457897011208ull, 583506457897015577ull, 583506457897020168ull, 583506457898190873ull,
    583506457898195208ull, 583506530323138568ull, 583506530323138603ull, 583506530323142937ull,
    583506530323147528ull, 583506530323147563ull, 583506530324252697ull, 583506530324257032ull,
    583506530324257067ull, 583506530324261657ull, 583506530325432328ull, 583506530325432363ull,
    583506530325436697ull, 583506530325441288ull, 583506530608351257ull, 583506530608355592ull,
    583506530608355627ull, 583506530608360217ull, 583506530609465352ull, 583506530609465387ull,
    583506530609469721ull, 583506530609474312ull, 583506530610645017ull, 583506530610649352ull,
    583506530910341128ull, 583506530910341163ull, 583506530910345497ull, 583506530910350088ull,
    583506530911455257ull, 583506530911459592ull, 583506607632549913ull, 583506607632554248ull,
    583506607632558873ull, 583506607633664008ull, 583506607633668377ull, 583506607634843673ull,
    583506607634848008ull, 583506607917762568ull, 583506607917766937ull, 583506607918876697ull,
    583506607918881032ull, 583506608219752473ull, 583506608219756808ull, 583506608220866568ull,
    583525149006366728ull, 583525149006366763ull, 583525149006371097ull, 583525149006375688ull,
    583525149007480857ull, 583525149007485192ull, 583525149007485227ull, 583525149007489817ull,
    583525149008660488ull, 583525149008664857ull, 583525149008669448ull, 583525149291579417ull,
    583525149291583752ull, 583525149291583787ull, 583525149291588377ull, 583525149292693512ull,
    583525149292693547ull, 583525149292697881ull, 583525149292702472ull, 583525149293873177ull,
    583525149293877512ull, 583525149593569288ull, 583525149593569323ull, 583525149593573657ull,
    583525149593578248ull, 583525149594683417ull, 583525149594687752ull, 583525149595863048ull,
    583525222020810777ull, 583525222020815112ull, 583525222020815147ull, 583525222020819737ull,
    583525222021924872ull, 583525222021924907ull, 583525222021929241ull, 583525222021933832ull,
    583525222023104537ull, 583525222023108872ull, 583525222306023432ull, 583525222306023467ull,
    583525222306027801ull, 583525222306032392ull, 583525222307137561ull, 583525222307141896ull,
    583525222308317192ull, 583525222608013337ull, 583525222608017672ull, 583525222609127432ull,
    583525299330222088ull, 583525299330226457ull, 583525299330231048ull, 583525299331336217ull,
    583525299331340552ull, 583525299332515848ull, 583525299615434777ull, 583525299615439112ull,
    583525299616548872ull, 583525299917424648ull, 583525299919727403ull, 583544940215666713ull,
    583544940215671048ull, 583544940215671083ull, 583544940215675673ull, 583544940216780808ull,
    583544940216785177ull, 583544940216789768ull, 583544940217960473ull, 583544940500879368ull,
    583544940500879403ull, 583544940500883737ull, 583544940500888328ull, 583544940501993497ull,
    583544940501997832ull, 583544940503173128ull, 583544940802869273ull, 583544940802873608ull,
    583545013230110728ull, 583545013230110763ull, 583545013230115097ull, 583545013230119688ull,
    583545013231224857ull, 583545013231229192ull, 583545013232404488ull, 583545013515323417ull,
    583545013515327752ull, 583545013516437512ull, 583545013517626137ull, 583545013819607083ull,
    583545090539526408ull, 583545090540636168ull, 583545090824734728ull, 583545090825853227ull,
    588573006889486344ull, 588573006889486379ull, 588573006889490713ull, 588573006889495304ull,
    588573006890600473ull, 588573006890604808ull, 588573006890604843ull, 588573006890609433ull,
    588573006891780104ull, 588573006891784473ull, 588573006891789099ull, 588573007174699033ull,
    588573007174703368ull, 588573007175813128ull, 588573007175813163ull, 588573007175817497ull,
    588573007176997128ull, 588573007476688904ull, 588573007476697899ull, 588573007477807368ull,
    588573007478991659ull, 588573079903930393ull, 588573079903934728ull, 588573079905044488ull,
    588573079905044523ull, 588573079905048857ull, 588573079906224153ull, 588573080189143048ull,
    588573080189143083ull, 588573080189147417ull, 588573080190257177ull, 588573080190261512ull,
    588573080191436808ull, 588573080491132953ull, 588573080491137288ull, 588573080492247048ull,
    588573157213341704ull, 588573157213350699ull, 588573157215635499ull, 588573157215644424ull,
    588573157215644459ull, 588573157498558728ull, 588573157499668488ull, 588573157800553224ull,
    588573157800553259ull, 588573157802846984ull, 588591698587158553ull, 588591698587162888ull,
    588591698587162923ull, 588591698587167513ull, 588591698588272648ull, 588591698588277017ull,
    588591698588281608ull, 588591698589452313ull, 588591698589456648ull, 588591698872371208ull,
    588591698872371243ull, 588591698872375577ull, 588591698872380168ull, 588591698873485337ull,
    588591698873489672ull, 588591698874664968ull, 588591699174361113ull, 588591699174365448ull,
    588591699175475208ull, 588591771601602568ull, 588591771601606937ull, 588591771601611528ull,
    588591771602716697ull, 588591771602721032ull, 588591771603896328ull, 588591771886815257ull,
    588591771886819592ull, 588591771887929352ull, 588591771889113387ull, 588591772188805128ull,
    588591848911013913ull, 588591848911018248ull, 588591848912128008ull, 588591849196226568ull,
    588591849197349657ull, 588611489796458504ull, 588611489796462873ull, 588611489797572633ull,
    588611489797576968ull, 588611490081671193ull, 588611490081675528ull, 588611490082785288ull,
    588611490383670059ull, 588611490385963819ull, 588611562810902553ull, 588611562810906888ull,
    588611562812016648ull, 588611563399223577ull, 588611640120322859ull, 588611640122607659ull,
    588611640407824648ull, 588611640707525384ull, 588611640707525419ull, 1803700481349388313ull,
    1803700481349392648ull, 1803700481349392683ull, 1803700481349397273ull, 1803700481350502408ull,
    1803700481350502443ull, 1803700481350506777ull, 1803700481350511368ull, 1803700481350511403ull,
    1803700481351682073ull, 1803700481351686408ull, 1803700481351686443ull, 1803700481634600968ull,
    1803700481634601003ull, 1803700481634605337ull, 1803700481634609928ull, 1803700481634609963ull,
    1803700481635715097ull, 1803700481635719432ull, 1803700481635719467ull, 1803700481635724057ull,
    1803700481636894728ull, 1803700481636894763ull, 1803700481636899097ull, 1803700481936590873ull,
    1803700481936595208ull, 1803700481937704968ull, 1803700481937709337ull, 1803700481937713928ull,
    1803700481938884633ull, 1803700481938888968ull, 1803700554363832328ull, 1803700554363832363ull,
    1803700554363836697ull, 1803700554363841288ull, 1803700554364946457ull, 1803700554364950792ull,
    1803700554364950827ull, 1803700554364955417ull, 1803700554366126088ull, 1803700554366126123ull,
    1803700554366130457ull, 1803700554649045017ull, 1803700554649049352ull, 1803700554649049387ull,
    1803700554649053977ull, 1803700554650159112ull, 1803700554650159147ull, 1803700554650163481ull,
    1803700554650168072ull, 1803700554651338777ull, 1803700554651343112ull, 1803700554951034888ull,
    1803700554951034923ull, 1803700554951039257ull, 1803700554951043848ull, 1803700554952149017ull,
    1803700554952153352ull, 1803700554953328648ull, 1803700631673243673ull, 1803700631673248008ull,
    1803700631674357768ull, 1803700631674357803ull, 1803700631674362137ull, 1803700631674366728ull,
    1803700631675541768ull, 1803700631958456328ull, 1803700631958460697ull, 1803700631958465288ull,
    1803700631959570457ull, 1803700631959574792ull, 1803700631960750088ull, 1803700632260446233ull,
    1803700632260450568ull, 1803719173047060488ull, 1803719173047060523ull, 1803719173047064857ull,
    1803719173047069448ull, 1803719173047069483ull, 1803719173048174617ull, 1803719173048178952ull,
    1803719173048178987ull, 1803719173048183577ull, 1803719173049354248ull, 1803719173049354283ull,
    1803719173049358617ull, 1803719173049363208ull, 1803719173332273177ull, 1803719173332277512ull,
    1803719173332277547ull, 1803719173332282137ull, 1803719173333387272ull, 1803719173333387307ull,
    1803719173333391641ull, 1803719173333396232ull, 1803719173334566937ull, 1803719173334571272ull,
    1803719173634263048ull, 1803719173634263083ull, 1803719173634267417ull, 1803719173634272008ull,
    1803719173635377177ull, 1803719173635381512ull, 1803719173636556808ull, 1803719246061504537ull,
    1803719246061508872ull, 1803719246061508907ull, 1803719246061513497ull, 1803719246062618632ull,
    1803719246062618667ull, 1803719246062623001ull, 1803719246062627592ull, 1803719246063798297ull,
    1803719246063802632ull, 1803719246346717192ull, 1803719246346717227ull, 1803719246346721561ull,
    1803719246346726152ull, 1803719246347831321ull, 1803719246347835656ull, 1803719246349010952ull,
    1803719246349019947ull, 1803719246648707097ull, 1803719246648711432ull, 1803719246649821192ull,
    1803719323370915848ull, 1803719323370915883ull, 1803719323370920217ull, 1803719323370924808ull,
    1803719323372029977ull, 1803719323372034312ull, 1803719323373209608ull, 1803719323656128537ull,
    1803719323656132872ull, 1803719323657242632ull, 1803719323958118408ull, 1803719323960416537ull,
    1803738964256360473ull, 1803738964256364808ull, 1803738964256369433ull, 1803738964257474568ull,
    1803738964257474603ull, 1803738964257478937ull, 1803738964257483528ull, 1803738964258654233ull,
    1803738964258658568ull, 1803738964541573128ull, 1803738964541573163ull, 1803738964541577497ull,
    1803738964541582088ull, 1803738964542687257ull, 1803738964542691592ull, 1803738964543866888ull,
    1803738964843567368ull, 1803738964844677128ull, 1803739037270804488ull, 1803739037270804523ull,
    1803739037270808857ull, 1803739037270813448ull, 1803739037271918617ull, 1803739037271922952ull,
    1803739037273098248ull, 1803739037556017177ull, 1803739037556021512ull, 1803739037557131272ull,
    1803739037858007048ull, 1803739037859125547ull, 1803739114580215833ull, 1803739114580220168ull,
    1803739114581329928ull, 1803739114865428488ull, 1808485555953469448ull, 1808485555953469483ull,
    1808485555953473817ull, 1808485555953478408ull, 1808485555954583577ull, 1808485555954587912ull,
    1808485555954587947ull, 1808485555954592537ull, 1808485555955763208ull, 1808485555955763243ull,
    1808485555955767577ull, 1808485555955772168ull, 1808485556238682137ull, 1808485556238686472ull,
    1808485556238686507ull, 1808485556238691097ull, 1808485556239796232ull, 1808485556239796267ull,
    1808485556239800601ull, 1808485556239805192ull, 1808485556240975897ull, 1808485556240980232ull,
    1808485556540672008ull, 1808485556540672043ull, 1808485556540676377ull, 1808485556540680968ull,
    1808485556541786137ull, 1808485556541790472ull, 1808485628967913497ull, 1808485628967917832ull,
    1808485628967917867ull, 1808485628967922457ull, 1808485628969027592ull, 1808485628969027627ull,
    1808485628969031961ull, 1808485628969036552ull, 1808485628970207257ull, 1808485628970211592ull,
    1808485629253126152ull, 1808485629253126187ull, 1808485629253130521ull, 1808485629253135112ull,
    1808485629254240281ull, 1808485629254244616ull, 1808485629255419912ull, 1808485629555116057ull,
    1808485629555120392ull, 1808485629556230152ull, 1808485706277324808ull, 1808485706277329177ull,
    1808485706277333768ull, 1808485706278438937ull, 1808485706278443272ull, 1808485706279618568ull,
    1808485706562537497ull, 1808485706562541832ull, 1808485706563651592ull, 1808485706564840217ull,
    1808485706864527368ull, 1808504247651141657ull, 1808504247651145992ull, 1808504247651146027ull,
    1808504247651150617ull, 1808504247652255752ull, 1808504247652255787ull, 1808504247652260121ull,
    1808504247652264712ull, 1808504247653435417ull, 1808504247653439752ull, 1808504247936354312ull,
    1808504247936354347ull, 1808504247936358681ull, 1808504247936363272ull, 1808504247937468441ull,
    1808504247937472776ull, 1808504247938648072ull, 1808504248238344217ull, 1808504248238348552ull,
    1808504248239458312ull, 1808504320665585672ull, 1808504320665585707ull, 1808504320665590041ull,
    1808504320665594632ull, 1808504320666699801ull, 1808504320666704136ull, 1808504320667879432ull,
    1808504320950798361ull, 1808504320950802696ull, 1808504320951912456ull, 1808504321252788232ull,
    1808504397974997017ull, 1808504397975001352ull, 1808504397976111112ull, 1808504397977295147ull,
    1808504398260209672ull, 1808524038860441608ull, 1808524038860441643ull, 1808524038860445977ull,
    1808524038860450568ull, 1808524038861555737ull, 1808524038861560072ull, 1808524038862735368ull,
    1808524039145654297ull, 1808524039145658632ull, 1808524039146768392ull, 1808524039146777387ull,
    1808524039447644168ull, 1808524111874885657ull, 1808524111874889992ull, 1808524111875999752ull,
    1808524112160098312ull, 1808524189184296968ull, 1808524189185420057ull, 1808524189771503897ull,
    1808524189773802248ull, 1813552105534261273ull, 1813552105534265608ull, 1813552105534265643ull,
    1813552105535375368ull, 1813552105535375403ull, 1813552105535379737ull, 1813552105535384328ull,
    1813552105536555033ull, 1813552105536559368ull, 1813552105819473928ull, 1813552105819478297ull,
    1813552105819482888ull, 1813552105820588057ull, 1813552105820592392ull, 1813552105821767688ull,
    1813552106121468168ull, 1813552106122577928ull, 1813552178548705288ull, 1813552178548705323ull,
    1813552178548709657ull, 1813552178548714248ull, 1813552178549819417ull, 1813552178549823752ull,
    1813552178550999048ull, 1813552178833917977ull, 1813552178833922312ull, 1813552178835032072ull,
    1813552179135907848ull, 1813552179137030937ull, 1813552255858120968ull, 1813552255859230728ull,
    1813552256143329288ull, 1813552256144447787ull, 1813552256447612953ull, 1813570797231933448ull,
    1813570797231937817ull, 1813570797231942408ull, 1813570797233047577ull, 1813570797233051912ull,
    1813570797234227208ull, 1813570797517146137ull, 1813570797517150472ull, 1813570797518260232ull,
    1813570797819136008ull, 1813570870246377497ull, 1813570870246381832ull, 1813570870247491592ull,
    1813570870531590152ull, 1813570870531599147ull, 1813570870533892872ull, 1813570870834694187ull,
    1813570947555788808ull, 1813570948144109832ull, 1813590588441233433ull, 1813590588441237768ull,
    1813590588442347528ull, 1813590588728744217ull, 1813590589029559048ull, 1813590661455677448ull,
    1813590661457980203ull, 1813590739050301483ull, 1813590739354585113ull, 3100737174032091144ull,
    3100737174032091179ull, 3100737174032095513ull, 3100737174032100104ull, 3100737174033205273ull,
    3100737174033209608ull, 3100737174033214233ull, 3100737174034384904ull, 3100737174034389273ull,
    3100737174317303833ull, 3100737174317308168ull, 3100737174318417928ull, 3100737174318417963ull,
    3100737174318422297ull, 3100737174318426888ull, 3100737174319597593ull, 3100737174619293704ull,
    3100737174619298073ull, 3100737174620407833ull, 3100737174620412168ull, 3100737247046535193ull,
    3100737247046539528ull, 3100737247046544153ull, 3100737247047649288ull, 3100737247047649323ull,
    3100737247047653657ull, 3100737247047658248ull, 3100737247048828953ull, 3100737247048833288ull,
    3100737247331747848ull, 3100737247331747883ull, 3100737247331752217ull, 3100737247331756808ull,
    3100737247332861977ull, 3100737247332866312ull, 3100737247633737753ull, 3100737247633742088ull,
    3100737247634851848ull, 3100737247636040473ull, 3100737324355946504ull, 3100737324355950873ull,
    3100737324355955499ull, 3100737324357060633ull, 3100737324357064968ull, 3100737324641159193ull,
    3100737324641163528ull, 3100737324642273288ull, 3100755865729763353ull, 3100755865729767688ull,
    3100755865729767723ull, 3100755865729772313ull, 3100755865730877448ull, 3100755865730877483ull,
    3100755865730881817ull, 3100755865730886408ull, 3100755865732057113ull, 3100755866014976008ull,
    3100755866014976043ull, 3100755866014980377ull, 3100755866014984968ull, 3100755866016090137ull,
    3100755866016094472ull, 3100755866017269768ull, 3100755866316965913ull, 3100755866316970248ull,
    3100755866318080008ull, 3100755938744207368ull, 3100755938744207403ull, 3100755938744211737ull,
    3100755938744216328ull, 3100755938745321497ull, 3100755938745325832ull, 3100755938746501128ull,
    3100755939029420057ull, 3100755939029424392ull, 3100755939030534152ull, 3100755939331409928ull,
    3100755939331418923ull, 3100756016053618713ull, 3100756016053623048ull, 3100756016054732808ull,
    3100756016055921433ull, 3100756016338831368ull, 3100775656939063304ull, 3100775656939067673ull,
    3100775656940177433ull, 3100775656940181768ull, 3100775657224275993ull, 3100775657224280328ull,
    3100775657225390088ull, 3100775657528559659ull, 3100775729953507353ull, 3100775729953511688ull,
    3100775730238720008ull, 3100775730241018137ull, 3100775807265212459ull, 3100775807549254408ull,
    3100775807549254443ull, 3100775807850121259ull, 3100775807852415019ull, 3105522248636172313ull,
    3105522248636176648ull, 3105522248636181273ull, 3105522248637286408ull, 3105522248637286443ull,
    3105522248637290777ull, 3105522248637295368ull, 3105522248638470408ull, 3105522248921384968ull,
    3105522248921385003ull, 3105522248921389337ull, 3105522248921393928ull, 3105522248922499097ull,
    3105522248922503432ull, 3105522248923678728ull, 3105522249223374873ull, 3105522249223379208ull,
    3105522249224488968ull, 3105522321650616328ull, 3105522321650620697ull, 3105522321651730457ull,
    3105522321651734792ull, 3105522321935829017ull, 3105522321935833352ull, 3105522321936943112ull,
    3105522321936952107ull, 3105522398960027673ull, 3105522398960032008ull, 3105522398961141768ull,
    3105522399245240328ull, 3105522399549528363ull, 3105540940333844488ull, 3105540940333844523ull,
    3105540940333848857ull, 3105540940333853448ull, 3105540940334958617ull, 3105540940334962952ull,
    3105540940336138248ull, 3105540940619057177ull, 3105540940619061512ull, 3105540940620171272ull,
    3105540940921047048ull, 3105540940922165547ull, 3105541013348288537ull, 3105541013348292872ull,
    3105541013349402632ull, 3105541013633501192ull, 3105541013936614152ull, 3105541013937784857ull,
    3105541090657699848ull, 3105541090942916907ull, 3105541090945210632ull, 3105560731543144473ull,
    3105560731543148808ull, 3105560731544258568ull, 3105560731545442603ull, 3105560731828357128ull,
    3105560732132649753ull, 3105560804557588488ull, 3105560804842810137ull, 3105560804843915307ull,
    3105560882455316488ull, 3110588798216964104ull, 3110588798216968473ull, 3110588798216973099ull,
    3110588798218082568ull, 3110588798219257899ull, 3110588798219266859ull, 3110588798502176793ull,
    3110588798502181128ull, 3110588798503290888ull, 3110588798806460459ull, 3110588798806469419ull,
    3110588871516620808ull, 3110588871518918937ull, 3110588948540819499ull, 3110588948540828459ull,
    3110588948543113259ull, 3110588948543122184ull, 3110588948543122219ull, 3110588949128022059ull,
    3110588949128030984ull, 3110588949128031019ull, 3110588949130324744ull, 3110607489914636313ull,
    3110607489914640648ull, 3110607489915750408ull, 3110607490199848968ull, 3110607490501847833ull,
    3110607490504136968ull, 3110607562929080328ull, 3110607562930203417ull, 3110607640524818457ull,
    3110627281123945259ull, 3110627281126238984ull, 3110627281713432619ull, 3110627354424711432ull,
    3110627354725587243ull, 3110627431447800584ull, 3110627431447800619ull, 3110627431450085384ull,
    3110627431450085419ull, 3110627431450094344ull, 3110627432035003144ull, 3110627432037296939ull,
};







// ─── KV-append IQ1_S: cuantiza K/V float→IQ1_S canónico.
// Layout por SB de 256 elems (50B): [d f16@0][qs[32]@2][qh[16]@34].
// Sub-bloque ib (32 elems): dl=d·(2δ+1), dd=±0.125;
// idxg(l) = qs[ib*4+l] | ((qhb>>(3l))&7)<<8; gv=i8(grid[idxg]>>(8j));
// val = dl*(gv+dd). d del SB = amax_sb/(9·15) sobre SUS 256 elems.
// BÚSQUEDA COOPERATIVA (blockDim 512): el warp w lleva el combo
// (delta=w>>1, neg=w&1) — mismo orden delta-major que el encoder serial —
// y reparte los 512 candidatos gi entre sus 32 lanes con reduce argmin
// (err asc, tie→índice MENOR), equivalente exacto del escaneo serial
// strict-< ascendente ⇒ selección bit-idéntica a encodeIQ1_S. El bloque
// elige combo por (err_total asc, warp asc) = mismo desempate del serial.
// Acumulación double (e/be/err_total; término dv*dv f32 ensanchado).
// REQUIERE kv_dim % 256 == 0. UN block por región: contrato single-writer
// por sector (TODO_NO_OOM.md §Contrato) + tiendas __stcg.
extern "C" __global__ void kvAppendIQ1_SKernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    uint8_t* __restrict__ cache,
    const int* __restrict__ bt,
    const int* __restrict__ start_pos,
    int n,
    int kv_dim, int n_kv_head, int head_dim, int block_size,
    int dbg)
{
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    if (nthreads != 512) return; // diseño: 16 warps ↔ 16 combos (delta,neg)
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int first_block = *start_pos / block_size;
    const int last_block = (*start_pos + n - 1) / block_size;

    const int elems_region = block_size * kv_dim;
    const int sb_per_region = (elems_region + 255) / 256;
    const size_t k_bytes = (size_t)sb_per_region * 50;
    const size_t phys_stride = 2 * k_bytes;

    __shared__ float s_amax[16];
    __shared__ float s_d;
    __shared__ float s_err[16];
    __shared__ int s_idx[16][4];

    for (int lb = first_block; lb <= last_block; ++lb) {
        const int phys = bt[lb];
        if (phys < 0) continue;
        for (int side = 0; side < 2; ++side) {
            const float* __restrict__ src = (side == 1) ? v : k;
            uint8_t* region = cache + (size_t)phys * phys_stride
                            + (size_t)side * k_bytes;

            auto ld = [&](int w) -> float {
                const int off_r = w / kv_dim;
                const int t_abs = lb * block_size + off_r;
                if (w >= elems_region || t_abs < *start_pos || t_abs >= *start_pos + n) return 0.0f;
                return src[(size_t)(t_abs - *start_pos) * kv_dim + (w % kv_dim)];
            };

            for (int sb = 0; sb < sb_per_region; ++sb) {
                uint8_t* dst = region + (size_t)sb * 50;
                const int sb_base = sb * 256;

                // Preservación: mismo guard que Q4_K (SB íntegro por token;
                // fuera del chunk ⇒ no reescribir). Uniforme en el bloque.
                {
                    const int t_sb = lb * block_size + sb_base / kv_dim;
                    if (t_sb < *start_pos || t_sb >= *start_pos + n) continue;
                }

                // d del SB: amax de SUS 256 elems (reducción de bloque).
                float amax_p = 0.0f;
                for (int w = sb_base + tid; w < sb_base + 256; w += nthreads)
                    amax_p = fmaxf(amax_p, fabsf(ld(w)));
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1)
                    amax_p = fmaxf(amax_p, __shfl_xor_sync(0xffffffffu, amax_p, o));
                if (lane == 0) s_amax[warp] = amax_p;
                __syncthreads();
                if (tid == 0) {
                    float m = 0.0f;
                    for (int wq = 0; wq < 16; ++wq) m = fmaxf(m, s_amax[wq]);
                    s_d = (m > 0.0f) ? m / (9.0f * 15.0f) : 1.0f;
                }
                __syncthreads();
                const float d = s_d;

                // Brute-force cooperativo: 8 sub-bloques × 16 combos.
                for (int ib = 0; ib < 8; ++ib) {
                    double err_total = 0.0;
                    int idxs[4] = {0, 0, 0, 0};
                    if (warp < 16) {
                        const int delta = warp >> 1;
                        const int neg_i = warp & 1;
                        const float dl = d * (2.0f * (float)delta + 1.0f);
                        const float dd = (neg_i == 1) ? -0.125f : 0.125f;
                        for (int l = 0; l < 4; ++l) {
                        // Preload del grupo (8 elems): independientes del
                        // candidato ⇒ fuera del scan (evita 512 ld() con
                        // div/mod por evaluación).
                        float xv[8];
                        #pragma unroll
                        for (int j = 0; j < 8; ++j)
                            xv[j] = ld(sb_base + ib * 32 + l * 8 + j);
                        // Scan 512 candidatos repartido 16/lane. Acumulación
                        // f32 MISMO orden que el encoder (e += dv*dv secuencial
                        // ascendente) ⇒ selección idéntica bit-a-bit.
                        float be = 3.0e38f;
                        int bi = lane;
                        // Mapeo intercalado kk*32+lane: lanes leen entradas
                        // consecutivas ⇒ cargas coalescidas. El desempate
                        // por índice del reduce preserva la semántica serial.
                        for (int kk = 0; kk < 16; ++kk) {
                            const int gi = kk * 32 + lane;
                            const unsigned long long g = dev_iq1s_grid[gi];
                            float e = 0.0f;
                            for (int j = 0; j < 8; ++j) {
                                const float gv = (float)(int8_t)((g >> (8 * j)) & 0xFF);
                                const float dv = xv[j] - dl * (gv + dd);
                                e += dv * dv;
                            }
                            if (e < be) { be = e; bi = gi; }
                        }
                        // Reduce argmin (be asc, tie→bi menor): idéntico
                        // al escaneo serial strict-< en orden ascendente.
                        #pragma unroll
                        for (int o = 16; o > 0; o >>= 1) {
                            const float eo = __shfl_xor_sync(0xffffffffu, be, o);
                            const int io = __shfl_xor_sync(0xffffffffu, bi, o);
                            if (eo < be || (eo == be && io < bi)) { be = eo; bi = io; }
                        }
                        idxs[l] = bi;
                        err_total += be;
                    }
                        if (lane == 0) {
                            s_err[warp] = err_total;
                            for (int l = 0; l < 4; ++l) s_idx[warp][l] = idxs[l];
                        }
                    }
                    __syncthreads();

                    if (tid == 0) {
                        // Ganador: err asc, tie→warp asc (= delta-major/neg-
                        // minor del encoder serial). Escrituras directas al
                        // dst con __stcg (single writer por byte).
                        int bw = 0;
                        for (int wq = 1; wq < 16; ++wq)
                            if (s_err[wq] < s_err[bw]) bw = wq;
                        const int best_delta = bw >> 1;
                        const int best_neg = bw & 1;
                        uint16_t qhb = (uint16_t)(best_delta << 12);
                        if (best_neg != 0) qhb |= 0x8000;
                        for (int l = 0; l < 4; ++l) {
                            const int bi = s_idx[bw][l];
                            __stcg(&dst[2 + ib * 4 + l], (uint8_t)(bi & 0xFF));
                            qhb |= (uint16_t)((bi >> 8) & 7) << (3 * l);
                        }
                        __stcg(&dst[34 + ib * 2], (uint8_t)(qhb & 0xFF));
                        __stcg(&dst[34 + ib * 2 + 1], (uint8_t)(qhb >> 8));
                        if (dbg != 0 && sb < 2 && ib < 2)
                            printf("[iq1sd] sb=%d ib=%d delta=%d neg=%d err=%.6e\n",
                                   sb, ib, best_delta, best_neg, s_err[bw]);
                    }
                    __syncthreads();
                }

                // Escala embebida (bytes disjuntos del resto).
                if (tid == 0) {
                    const unsigned short bd = __half_as_ushort(__float2half(d));
                    __stcg(&dst[0], (uint8_t)(bd & 0xFF));
                    __stcg(&dst[1], (uint8_t)(bd >> 8));
                }
                __syncthreads();
            }
        }
    }
}

// ─── KV-append IQ3_S: cuantiza K/V float→IQ3_S canónico.
// Layout por SB de 256 elems (110B): [d f16@0][qs[64]@2][qh[8]@66]
// [signs[32]@74][scales[4]@106]. Elemento in: it=in/64, half=(in%64)/32,
// l=(in%32)/8, col=in%8; db=d·(1+2·nibble(sc,half)); idx a/b = qs-byte |
// bit hb<<8 (a: bit 2l, b: bit 2l+1); valor = db · byte_j(grid[idx]) ·
// signo(bit col de sm). Espejo EXACTO de encodeIQ3_S.
// BÚSQUEDA COOPERATIVA (blockDim 512): por mitad (it,half), el warp w lleva
// sc=w y reparte los 512 candidatos entre lanes con reduce argmin
// (err asc, tie→índice menor) — equivalente exacto del escaneo serial
// strict-< ascendente ⇒ bit-exacto. Bloque elige sc por (err asc, warp asc).
// Signos sc-independientes: tid0 los escribe tras elegir sc. Acumulación
// double. REQUIERE kv_dim % 256 == 0. UN block + tiendas __stcg (contrato).
extern "C" __global__ void kvAppendIQ3_SKernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    uint8_t* __restrict__ cache,
    const int* __restrict__ bt,
    const int* __restrict__ start_pos,
    int n,
    int kv_dim, int n_kv_head, int head_dim, int block_size,
    int dbg)
{
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    if (nthreads != 512) return; // diseño: 16 warps ↔ 16 códigos sc
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int first_block = *start_pos / block_size;
    const int last_block = (*start_pos + n - 1) / block_size;

    const int elems_region = block_size * kv_dim;
    const int sb_per_region = (elems_region + 255) / 256;
    const size_t k_bytes = (size_t)sb_per_region * 110;
    const size_t phys_stride = 2 * k_bytes;

    __shared__ float s_amax[16];
    __shared__ float s_d;
    __shared__ float s_err[16];
    __shared__ uint8_t s_qa[16][4];
    __shared__ uint8_t s_qb[16][4];
    __shared__ uint8_t s_hb[16];

    for (int lb = first_block; lb <= last_block; ++lb) {
        const int phys = bt[lb];
        if (phys < 0) continue;
        for (int side = 0; side < 2; ++side) {
            const float* __restrict__ src = (side == 1) ? v : k;
            uint8_t* region = cache + (size_t)phys * phys_stride
                            + (size_t)side * k_bytes;

            auto ld = [&](int w) -> float {
                const int off_r = w / kv_dim;
                const int t_abs = lb * block_size + off_r;
                if (w >= elems_region || t_abs < *start_pos || t_abs >= *start_pos + n) return 0.0f;
                return src[(size_t)(t_abs - *start_pos) * kv_dim + (w % kv_dim)];
            };

            for (int sb = 0; sb < sb_per_region; ++sb) {
                uint8_t* dst = region + (size_t)sb * 110;
                const int sb_base = sb * 256;

                // Preservación: mismo guard que Q4_K (uniforme en el bloque).
                {
                    const int t_sb = lb * block_size + sb_base / kv_dim;
                    if (t_sb < *start_pos || t_sb >= *start_pos + n) continue;
                }

                // d del SB: amax de SUS 256 elems (reducción de bloque).
                float amax_p = 0.0f;
                for (int w = sb_base + tid; w < sb_base + 256; w += nthreads)
                    amax_p = fmaxf(amax_p, fabsf(ld(w)));
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1)
                    amax_p = fmaxf(amax_p, __shfl_xor_sync(0xffffffffu, amax_p, o));
                if (lane == 0) s_amax[warp] = amax_p;
                __syncthreads();
                if (tid == 0) {
                    float m = 0.0f;
                    for (int wq = 0; wq < 16; ++wq) m = fmaxf(m, s_amax[wq]);
                    s_d = (m > 0.0f) ? m / (15.0f * 31.0f) : 1.0f;
                }
                __syncthreads();
                const float d = s_d;

                // Por mitad: 16 warps ↔ 16 códigos sc; ganador por bloque.
                // scales se acumula en registro local (el byte del pool puede
                // ser stale: nunca leer-modificar-escribir el destino).
                uint8_t scales_acc[4] = {0, 0, 0, 0};
                for (int it = 0; it < 4; ++it) {
                    for (int half = 0; half < 2; ++half) {
                        const int h_base = sb_base + it * 64 + half * 32;

                        float err_total = 0.0f;
                        uint8_t qa[4] = {0, 0, 0, 0}, qb[4] = {0, 0, 0, 0};
                        uint8_t hbb = 0;
                        if (warp < 16) {
                            const float db = d * (2.0f * (float)warp + 1.0f);
                            for (int l = 0; l < 4; ++l) {
                                // a-índice (cols 0..3) y b-índice (cols 4..7)
                                // independientes: preload del cuarteto (no
                                // depende del candidato) + scan 16/lane.
                                for (int is_b = 0; is_b < 2; ++is_b) {
                                    float xv[4];
                                    #pragma unroll
                                    for (int j = 0; j < 4; ++j)
                                        xv[j] = fabsf(ld(h_base + l * 8 + j + (is_b ? 4 : 0)));
                                    // Acumulación f32 MISMO orden que encoder
                                    // (e += dv*dv secuencial ascendente).
                                    float be = 3.0e38f;
                                    int bi = lane;
                                    // Mapeo intercalado coalescido (ver iq1_s).
                                    for (int kk = 0; kk < 16; ++kk) {
                                        const int gi = kk * 32 + lane;
                                        const unsigned int g = dev_iq3s_grid[gi];
                                        float e = 0.0f;
                                        for (int j = 0; j < 4; ++j) {
                                            const float bv = (float)((g >> (8 * j)) & 0xFF);
                                            const float dv = xv[j] - db * bv;
                                            e += dv * dv;
                                        }
                                        if (e < be) { be = e; bi = gi; }
                                    }
                                    #pragma unroll
                                    for (int o = 16; o > 0; o >>= 1) {
                                        const float eo = __shfl_xor_sync(0xffffffffu, be, o);
                                        const int io = __shfl_xor_sync(0xffffffffu, bi, o);
                                        if (eo < be || (eo == be && io < bi)) { be = eo; bi = io; }
                                    }
                                    err_total += be;
                                    if (is_b == 0) {
                                        qa[l] = (uint8_t)(bi & 0xFF);
                                        hbb |= (uint8_t)(((bi >> 8) & 1) << (2 * l));
                                    } else {
                                        qb[l] = (uint8_t)(bi & 0xFF);
                                        hbb |= (uint8_t)(((bi >> 8) & 1) << (2 * l + 1));
                                    }
                                }
                            }
                            if (lane == 0) {
                                s_err[warp] = err_total;
                                for (int l = 0; l < 4; ++l) { s_qa[warp][l] = qa[l]; s_qb[warp][l] = qb[l]; }
                                s_hb[warp] = hbb;
                            }
                        }
                        __syncthreads();

                        if (tid == 0) {
                            // Ganador: err asc, tie→warp asc (= sc asc serial).
                            int bw = 0;
                            for (int wq = 1; wq < 16; ++wq)
                                if (s_err[wq] < s_err[bw]) bw = wq;
                            for (int l = 0; l < 4; ++l) {
                                __stcg(&dst[2 + it * 16 + half * 8 + 2 * l], s_qa[bw][l]);
                                __stcg(&dst[2 + it * 16 + half * 8 + 2 * l + 1], s_qb[bw][l]);
                                // Signos sc-independientes: óptimo sgn(x).
                                uint8_t sm = 0;
                                for (int j = 0; j < 4; ++j) {
                                    if (ld(h_base + l * 8 + j) < 0.0f)
                                        sm |= (uint8_t)(1u << j);
                                    if (ld(h_base + l * 8 + 4 + j) < 0.0f)
                                        sm |= (uint8_t)(1u << (j + 4));
                                }
                                __stcg(&dst[74 + it * 8 + half * 4 + l], sm);
                            }
                            __stcg(&dst[66 + 2 * it + half], s_hb[bw]);
                            scales_acc[it] |= (uint8_t)(half == 0 ? bw : bw << 4);
                        }
                        __syncthreads();
                    }
                }

                // Escala embebida + bytes de scales acumulados localmente.
                if (tid == 0) {
                    const unsigned short bd = __half_as_ushort(__float2half(d));
                    __stcg(&dst[0], (uint8_t)(bd & 0xFF));
                    __stcg(&dst[1], (uint8_t)(bd >> 8));
                    for (int i = 0; i < 4; ++i) __stcg(&dst[106 + i], scales_acc[i]);
                }
                __syncthreads();
            }
        }
    }
}

// ─── KV-append IQ3_XXS: cuantiza K/V float→IQ3_XXS canónico al pool.
// Layout por SB de 256 elems (98B): [d f16][qs[64]@2][ss[32]@66]. Sub-bloque
// ib de 32: aux u32 LE = ss[ib*4..] con sc en bits[28,32) e idx_l (signos
// cols 0-6) en [7l,7l+7); db=d·(0.5+sc)·0.5; signs=ksigns[idx_l] (bit c =
// signo elem l*8+c; ⚠️ bit7 de col7 lo impone la paridad de los otros).
// ESPEJO EXACTO de encodeIQ3_XXS: d=max_span/(4·62); scan sc ascendente;
// grids iq3xxs por cuarteto independientes (tie→menor índice); f64.
// Serial tid==0 + guard preservación SB íntegro + zero previo a los OR.
extern "C" __global__ void kvAppendIQ3_XXSKernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    uint8_t* __restrict__ cache,
    const int* __restrict__ bt,
    const int* __restrict__ start_pos,
    int n,
    int kv_dim, int n_kv_head, int head_dim, int block_size,
    int dbg)
{
    const int tid = threadIdx.x;
    // 7.5 FIX (lane-f): era serial tid==0 — ver kvAppendIQ2_SKernel.
    const int first_block = *start_pos / block_size;
    const int last_block = (*start_pos + n - 1) / block_size;

    const int elems_region = block_size * kv_dim;
    const int sb_per_region = (elems_region + 255) / 256;
    const size_t k_bytes = (size_t)sb_per_region * 98;
    const size_t phys_stride = 2 * k_bytes;

    for (int lb = first_block; lb <= last_block; ++lb) {
        const int phys = bt[lb];
        if (phys < 0) continue;
        for (int side = 0; side < 2; ++side) {
            const float* __restrict__ src = (side == 1) ? v : k;
            uint8_t* region = cache + (size_t)phys * phys_stride
                            + (size_t)side * k_bytes;

            auto ld0 = [&](int w) -> float {
                const int off_r = w / kv_dim;
                const int t_abs = lb * block_size + off_r;
                if (w >= elems_region || t_abs < *start_pos || t_abs >= *start_pos + n) return 0.0f;
                return src[(size_t)(t_abs - *start_pos) * kv_dim + (w % kv_dim)];
            };

            // 7.5 FIX: paralelización 1:1 thread↔SB (bit-exact por SB).
            const int sb = tid;
            if (sb < sb_per_region) {
                uint8_t* dst = region + (size_t)sb * 98;
                const int sb_base = sb * 256;

                // 7.5 FIX-2 (lane-f): los grid-searches re-leían ld() O(4M)
                // veces por SB — global uncoalesced en el inner loop ⇒ ~225ms
                // por append. Cache del SB en array local (guards aplicados
                // UNA vez — bit-exact: mismos valores que ld0).
                float x[256];
                for (int w = 0; w < 256; ++w) x[w] = ld0(sb_base + w);
                auto ld = [&](int w) -> float { return x[w - sb_base]; };

                // Preservación: mismo guard que Q4_K.
                {
                    const int t_sb = lb * block_size + sb_base / kv_dim;
                    if (t_sb < *start_pos || t_sb >= *start_pos + n) continue;
                }

                // Zero previo (los OR posteriores asumen base 0).
                for (int z = 0; z < 98; ++z) __stcg(&dst[z], 0);

                float max_span = 0.0f;
                for (int s = 0; s < 8; ++s) {
                    float mn = FLT_MAX, mx = -FLT_MAX; // lane-cuda: 1e308f fuera de rango f32 (NVRTC lo rechaza; nvcc degrada a inf)
                    for (int c = 0; c < 32; ++c) {
                        const float vv = ld(sb_base + s * 32 + c);
                        mn = fminf(mn, vv);
                        mx = fmaxf(mx, vv);
                    }
                    max_span = fmaxf(max_span, mx - mn);
                }
                const float d = (max_span > 0.0f) ? max_span / (4.0f * 62.0f) : 1.0f;

                for (int ib = 0; ib < 8; ++ib) {
                    // Signos fijos del sub-bloque (cols 0-6 deseados; col7 paridad).
                    int sgn[8];
                    unsigned idx7b = 0;
                    for (int c = 0; c < 7; ++c)
                        if (ld(sb_base + ib * 32 + c) < 0.0f) idx7b |= 1u << c;
                    const uint8_t sm = dev_ksigns_iq2xs[idx7b];
                    for (int c = 0; c < 8; ++c)
                        sgn[c] = ((sm >> c) & 1) ? -1.0f : 1.0f;

                    double best_err = 1e308;
                    uint8_t best_sc = 0;
                    uint8_t best_bytes[8] = {0, 0, 0, 0, 0, 0, 0, 0};

                    for (int sc = 0; sc < 16; ++sc) {
                        const float db = d * (0.5f + (float)sc) * 0.5f;
                        double err_total = 0.0;
                        uint8_t bytes[8];

                        for (int l = 0; l < 4; ++l) {
                            for (int is_b = 0; is_b < 2; ++is_b) {
                                double be = 1e308;
                                int bi = 0;
                                for (int gi = 0; gi < 256; ++gi) {
                                    const unsigned int g = dev_iq3xxs_grid[gi];
                                    double e = 0.0;
                                    for (int jj = 0; jj < 4; ++jj) {
                                        const int col = jj + (is_b ? 4 : 0);
                                        const float gv = (float)((g >> (8 * jj)) & 0xFF);
                                        const float dv = ld(sb_base + ib * 32 + l * 8 + col)
                                                       - sgn[col] * db * gv;
                                        e += (double)(dv * dv);
                                    }
                                    if (e < be) { be = e; bi = gi; }
                                }
                                err_total += be;
                                bytes[l * 2 + is_b] = (uint8_t)bi;
                            }
                        }
                        if (err_total < best_err) {
                            best_err = err_total;
                            best_sc = (uint8_t)sc;
                            for (int z = 0; z < 8; ++z) best_bytes[z] = bytes[z];
                        }
                    }

                    for (int l = 0; l < 4; ++l) {
                        __stcg(&dst[2 + ib * 8 + l * 2], best_bytes[l * 2]);
                        __stcg(&dst[2 + ib * 8 + l * 2 + 1], best_bytes[l * 2 + 1]);
                    }
                    unsigned aux = (unsigned)best_sc << 28;
                    for (int l = 0; l < 4; ++l) aux |= idx7b << (7 * l);
                    __stcg(&dst[66 + ib * 4 + 0], (uint8_t)(aux & 0xFF));
                    __stcg(&dst[66 + ib * 4 + 1], (uint8_t)((aux >> 8) & 0xFF));
                    __stcg(&dst[66 + ib * 4 + 2], (uint8_t)((aux >> 16) & 0xFF));
                    __stcg(&dst[66 + ib * 4 + 3], (uint8_t)((aux >> 24) & 0xFF));
                }

                const unsigned short bd = __half_as_ushort(__float2half(d));
                __stcg(&dst[0], (uint8_t)(bd & 0xFF));
                __stcg(&dst[1], (uint8_t)(bd >> 8));
            }
        }
    }
}
// ─── KV-append MXFP4: cuantiza K/V float→MXFP4 canónico al pool paginado.
// Layout por bloque de 32 elems (17B): [escala u8 E8M0][qs[16] split-16];
// valor = 2^(escala−127)·kvalues_fp4[nibble]. ESPEJO EXACTO de encodeMXFP4:
// escala MÍNIMA e tal que 2^(e−127)·12 ≥ amax (búsqueda por doblar — sin
// log2f/exp2f ⇒ bit-idéntico). Gran 32 ⇒ solo kv_dim % 32 == 0.
extern "C" __global__ void kvAppendMXFP4Kernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    uint8_t* __restrict__ cache,
    const int* __restrict__ bt,
    const int* __restrict__ start_pos,
    int n,
    int kv_dim, int n_kv_head, int head_dim, int block_size,
    int dbg)
{
    const int tid = threadIdx.x;
    const int first_block = *start_pos / block_size;
    const int last_block = (*start_pos + n - 1) / block_size;

    const int elems_region = block_size * kv_dim;
    const int qb_per_region = (elems_region + 31) / 32;
    const size_t k_bytes = (size_t)qb_per_region * 17;
    const size_t phys_stride = 2 * k_bytes;

    // LUT FP4 canónica (misma que encodeMXFP4/kv_quant.zig).
    const int8_t kv_fp4[16] = { 0, 1, 2, 3, 4, 6, 8, 12,
        0, -1, -2, -3, -4, -6, -8, -12 };

    for (int lb = first_block; lb <= last_block; ++lb) {
        const int phys = bt[lb];
        if (phys < 0) continue;
        for (int side = 0; side < 2; ++side) {
            const float* __restrict__ src = (side == 1) ? v : k;
            uint8_t* region = cache + (size_t)phys * phys_stride
                            + (size_t)side * k_bytes;
            for (int qb = 0; qb < qb_per_region; ++qb) {
                const int be_idx = qb * 32 + tid;
                const int off = be_idx / kv_dim;
                const int c = be_idx % kv_dim;
                const int t_abs = lb * block_size + off;
                if (t_abs < *start_pos || t_abs >= *start_pos + n) continue;
                const float src_val = src[(size_t)(t_abs - *start_pos) * kv_dim + c];

                float amax = fabsf(src_val);
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1)
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));

                // Escala E8M0 mínima por doblar (bit-idéntica al encoder).
                uint8_t e = 127;
                float cover = 12.0f;
                while (cover < amax && e < 254) {
                    e += 1;
                    cover *= 2.0f;
                }
                const float d = cover / 12.0f;

                uint8_t best = 0;
                float best_diff = fabsf(src_val - d * (float)kv_fp4[0]);
                #pragma unroll
                for (int id = 1; id < 16; ++id) {
                    const float diff = fabsf(src_val - d * (float)kv_fp4[id]);
                    if (diff < best_diff) { best_diff = diff; best = (uint8_t)id; }
                }

                uint8_t* dst = region + (size_t)qb * 17;

                // Split-16 single-writer: lane tid<16 escribe SU byte.
                const int q_hi16 = __shfl_xor_sync(0xffffffffu, best, 16);
                __syncwarp();
                if (tid < 16) {
                    __stcg(&dst[1 + tid],
                           (uint8_t)((best & 0x0F) | ((q_hi16 & 0x0F) << 4)));
                }
                __syncwarp();
                if (tid == 0) {
                    __stcg(&dst[0], e);
                }
            }
            __syncthreads();
        }
    }
}
// ─── KV-append IQ2_XXS: cuantiza K/V float→IQ2_XXS canónico al pool.
// Layout por SB de 256 elems (66B): [d f16][qs[64]]: por sub-bloque ib de
// 32: aux0 u32 LE = 4 índices grid (grupo l), aux1 u32 LE = sc bits[28,32)
// + idx-signos 7 bits/l. db=d·(0.5+sc)·0.25; gv=byte j de iq2xxs_grid[idx];
// sg=±1 vía ksigns (bit7 col7 impuesto — igual que iq3_xxs).
// ESPEJO EXACTO de encodeIQ2_XXS: d=max_span/(15.5·43); scan sc ascendente;
// grid search por grupo con signos fijos; f64. Serial tid==0 + guard
// preservación + zero previo a los OR.
extern "C" __global__ void kvAppendIQ2_XXSKernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    uint8_t* __restrict__ cache,
    const int* __restrict__ bt,
    const int* __restrict__ start_pos,
    int n,
    int kv_dim, int n_kv_head, int head_dim, int block_size,
    int dbg)
{
    const int tid = threadIdx.x;
    // 7.5 FIX (lane-f): era serial tid==0 — ver kvAppendIQ2_SKernel.
    const int first_block = *start_pos / block_size;
    const int last_block = (*start_pos + n - 1) / block_size;

    const int elems_region = block_size * kv_dim;
    const int sb_per_region = (elems_region + 255) / 256;
    const size_t k_bytes = (size_t)sb_per_region * 66;
    const size_t phys_stride = 2 * k_bytes;

    for (int lb = first_block; lb <= last_block; ++lb) {
        const int phys = bt[lb];
        if (phys < 0) continue;
        for (int side = 0; side < 2; ++side) {
            const float* __restrict__ src = (side == 1) ? v : k;
            uint8_t* region = cache + (size_t)phys * phys_stride
                            + (size_t)side * k_bytes;

            auto ld0 = [&](int w) -> float {
                const int off_r = w / kv_dim;
                const int t_abs = lb * block_size + off_r;
                if (w >= elems_region || t_abs < *start_pos || t_abs >= *start_pos + n) return 0.0f;
                return src[(size_t)(t_abs - *start_pos) * kv_dim + (w % kv_dim)];
            };

            // 7.5 FIX: paralelización 1:1 thread↔SB (bit-exact por SB).
            const int sb = tid;
            if (sb < sb_per_region) {
                uint8_t* dst = region + (size_t)sb * 66;
                const int sb_base = sb * 256;

                // 7.5 FIX-2 (lane-f): los grid-searches re-leían ld() O(4M)
                // veces por SB — global uncoalesced en el inner loop ⇒ ~225ms
                // por append. Cache del SB en array local (guards aplicados
                // UNA vez — bit-exact: mismos valores que ld0).
                float x[256];
                for (int w = 0; w < 256; ++w) x[w] = ld0(sb_base + w);
                auto ld = [&](int w) -> float { return x[w - sb_base]; };

                // Preservación: mismo guard que Q4_K.
                {
                    const int t_sb = lb * block_size + sb_base / kv_dim;
                    if (t_sb < *start_pos || t_sb >= *start_pos + n) continue;
                }

                // Zero previo (OR posterior asume base 0).
                for (int z = 0; z < 66; ++z) __stcg(&dst[z], 0);

                float max_span = 0.0f;
                for (int s = 0; s < 8; ++s) {
                    float mn = FLT_MAX, mx = -FLT_MAX; // lane-cuda: 1e308f fuera de rango f32 (NVRTC lo rechaza; nvcc degrada a inf)
                    for (int c = 0; c < 32; ++c) {
                        const float vv = ld(sb_base + s * 32 + c);
                        mn = fminf(mn, vv);
                        mx = fmaxf(mx, vv);
                    }
                    max_span = fmaxf(max_span, mx - mn);
                }
                const float d = (max_span > 0.0f) ? max_span / (15.5f * 43.0f) : 1.0f;

                for (int ib = 0; ib < 8; ++ib) {
                    // Signos fijos por grupo l.
                    float sgn[4][8];
                    unsigned idxv[4] = {0, 0, 0, 0};
                    for (int l = 0; l < 4; ++l) {
                        unsigned i7b = 0;
                        for (int c = 0; c < 7; ++c)
                            if (ld(sb_base + ib * 32 + l * 8 + c) < 0.0f) i7b |= 1u << c;
                        idxv[l] = i7b;
                        const uint8_t sm = dev_ksigns_iq2xs[i7b];
                        for (int c = 0; c < 8; ++c)
                            sgn[l][c] = ((sm >> c) & 1) ? -1.0f : 1.0f;
                    }

                    double best_err = 1e308;
                    uint8_t best_sc = 0;
                    unsigned best_idx0 = 0;

                    for (int sc = 0; sc < 16; ++sc) {
                        const float db = d * (0.5f + (float)sc) * 0.25f;
                        double err_total = 0.0;
                        unsigned idx0 = 0;

                        for (int l = 0; l < 4; ++l) {
                            double be = 1e308;
                            unsigned bi = 0;
                            for (int gi = 0; gi < 256; ++gi) {
                                const unsigned long long g = dev_iq2xxs_grid[gi];
                                double e = 0.0;
                                for (int j = 0; j < 8; ++j) {
                                    const float gv = (float)((g >> (8 * j)) & 0xFF);
                                    const float dv = ld(sb_base + ib * 32 + l * 8 + j)
                                                   - sgn[l][j] * db * gv;
                                    e += (double)(dv * dv);
                                }
                                if (e < be) { be = e; bi = gi; }
                            }
                            err_total += be;
                            idx0 |= bi << (8 * l);
                        }
                        if (err_total < best_err) {
                            best_err = err_total;
                            best_sc = (uint8_t)sc;
                            best_idx0 = idx0;
                        }
                    }

                    for (int l = 0; l < 4; ++l) {
                        const unsigned byte_l = (best_idx0 >> (8 * l)) & 0xFF;
                        __stcg(&dst[2 + ib * 8 + l], (uint8_t)byte_l);
                    }
                    unsigned aux1 = (unsigned)best_sc << 28;
                    for (int l = 0; l < 4; ++l) aux1 |= idxv[l] << (7 * l);
                    __stcg(&dst[2 + ib * 8 + 4], (uint8_t)(aux1 & 0xFF));
                    __stcg(&dst[2 + ib * 8 + 5], (uint8_t)((aux1 >> 8) & 0xFF));
                    __stcg(&dst[2 + ib * 8 + 6], (uint8_t)((aux1 >> 16) & 0xFF));
                    __stcg(&dst[2 + ib * 8 + 7], (uint8_t)((aux1 >> 24) & 0xFF));
                }

                const unsigned short bd = __half_as_ushort(__float2half(d));
                __stcg(&dst[0], (uint8_t)(bd & 0xFF));
                __stcg(&dst[1], (uint8_t)(bd >> 8));
            }
        }
    }
}
// ─── KV-append IQ2_XS: cuantiza K/V float→IQ2_XS canónico al pool.
// Layout por SB de 256 elems (74B): [d f16][qs[64]][scales[8]]. Sub-bloque
// ib de 32, 4 grupos l de 8: v u16 LE en qs[ib*8+2l..]: bits[0,9)=índice
// iq2xs_grid(512), bits[9,16)=idx-signos ksigns; db=d·(0.5+nibble(scales
// [ib],l/2))·0.25. ESPEJO EXACTO de encodeIQ2_XS: d=max_span/15.5; scan
// nibble ascendente POR MITAD (grupos comparten nibble); grid search con
// signos fijos; f64. Serial tid==0 + guard preservación + zero previo.
extern "C" __global__ void kvAppendIQ2_XSKernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    uint8_t* __restrict__ cache,
    const int* __restrict__ bt,
    const int* __restrict__ start_pos,
    int n,
    int kv_dim, int n_kv_head, int head_dim, int block_size,
    int dbg)
{
    const int tid = threadIdx.x;
    // 7.5 FIX (lane-f): era serial tid==0 — ver kvAppendIQ2_SKernel.
    const int first_block = *start_pos / block_size;
    const int last_block = (*start_pos + n - 1) / block_size;

    const int elems_region = block_size * kv_dim;
    const int sb_per_region = (elems_region + 255) / 256;
    const size_t k_bytes = (size_t)sb_per_region * 74;
    const size_t phys_stride = 2 * k_bytes;

    for (int lb = first_block; lb <= last_block; ++lb) {
        const int phys = bt[lb];
        if (phys < 0) continue;
        for (int side = 0; side < 2; ++side) {
            const float* __restrict__ src = (side == 1) ? v : k;
            uint8_t* region = cache + (size_t)phys * phys_stride
                            + (size_t)side * k_bytes;

            auto ld0 = [&](int w) -> float {
                const int off_r = w / kv_dim;
                const int t_abs = lb * block_size + off_r;
                if (w >= elems_region || t_abs < *start_pos || t_abs >= *start_pos + n) return 0.0f;
                return src[(size_t)(t_abs - *start_pos) * kv_dim + (w % kv_dim)];
            };

            // 7.5 FIX: paralelización 1:1 thread↔SB (bit-exact por SB).
            const int sb = tid;
            if (sb < sb_per_region) {
                uint8_t* dst = region + (size_t)sb * 74;
                const int sb_base = sb * 256;

                // 7.5 FIX-2 (lane-f): los grid-searches re-leían ld() O(4M)
                // veces por SB — global uncoalesced en el inner loop ⇒ ~225ms
                // por append. Cache del SB en array local (guards aplicados
                // UNA vez — bit-exact: mismos valores que ld0).
                float x[256];
                for (int w = 0; w < 256; ++w) x[w] = ld0(sb_base + w);
                auto ld = [&](int w) -> float { return x[w - sb_base]; };

                // Preservación: mismo guard que Q4_K.
                {
                    const int t_sb = lb * block_size + sb_base / kv_dim;
                    if (t_sb < *start_pos || t_sb >= *start_pos + n) continue;
                }

                // Zero previo (OR no usado aquí, pero qs/scales se escriben
                // completos; el zero cubre cualquier hueco).
                for (int z = 0; z < 74; ++z) __stcg(&dst[z], 0);

                float max_span = 0.0f;
                for (int s = 0; s < 8; ++s) {
                    float mn = FLT_MAX, mx = -FLT_MAX; // lane-cuda: 1e308f fuera de rango f32 (NVRTC lo rechaza; nvcc degrada a inf)
                    for (int c = 0; c < 32; ++c) {
                        const float vv = ld(sb_base + s * 32 + c);
                        mn = fminf(mn, vv);
                        mx = fmaxf(mx, vv);
                    }
                    max_span = fmaxf(max_span, mx - mn);
                }
                const float d = (max_span > 0.0f) ? max_span / 15.5f : 1.0f;

                for (int ib = 0; ib < 8; ++ib) {
                    // Signos fijos por grupo l.
                    float sgn[4][8];
                    unsigned idxv[4] = {0, 0, 0, 0};
                    for (int l = 0; l < 4; ++l) {
                        unsigned i7b = 0;
                        for (int c = 0; c < 7; ++c)
                            if (ld(sb_base + ib * 32 + l * 8 + c) < 0.0f) i7b |= 1u << c;
                        idxv[l] = i7b;
                        const uint8_t sm = dev_ksigns_iq2xs[i7b];
                        for (int c = 0; c < 8; ++c)
                            sgn[l][c] = ((sm >> c) & 1) ? -1.0f : 1.0f;
                    }

                    uint8_t nibbles = 0;
                    for (int half = 0; half < 2; ++half) {
                        double best_err = 1e308;
                        uint8_t best_n = 0;
                        unsigned best_gi[2] = {0, 0};

                        for (int nn = 0; nn < 16; ++nn) {
                            const float db = d * (0.5f + (float)nn) * 0.25f;
                            double err_total = 0.0;
                            unsigned gis[2] = {0, 0};
                            for (int go = 0; go < 2; ++go) {
                                const int l = half * 2 + go;
                                double be = 1e308;
                                unsigned bi = 0;
                                for (int gi = 0; gi < 512; ++gi) {
                                    const unsigned long long g = dev_iq2xs_grid[gi];
                                    double e = 0.0;
                                    for (int j = 0; j < 8; ++j) {
                                        const float gv = (float)((g >> (8 * j)) & 0xFF);
                                        const float dv = ld(sb_base + ib * 32 + l * 8 + j)
                                                       - sgn[l][j] * db * gv;
                                        e += (double)(dv * dv);
                                    }
                                    if (e < be) { be = e; bi = gi; }
                                }
                                err_total += be;
                                gis[go] = bi;
                            }
                            if (err_total < best_err) {
                                best_err = err_total;
                                best_n = (uint8_t)nn;
                                best_gi[0] = gis[0];
                                best_gi[1] = gis[1];
                            }
                        }

                        nibbles |= (half == 0) ? best_n : (uint8_t)(best_n << 4);
                        for (int go = 0; go < 2; ++go) {
                            const int l = half * 2 + go;
                            const unsigned short vv16 =
                                (unsigned short)(best_gi[go] & 511)
                              | ((unsigned short)(idxv[l] & 127) << 9);
                            __stcg(&dst[2 + ib * 8 + l * 2], (uint8_t)(vv16 & 0xFF));
                            __stcg(&dst[2 + ib * 8 + l * 2 + 1], (uint8_t)((vv16 >> 8) & 0xFF));
                        }
                    }
                    __stcg(&dst[66 + ib], nibbles);
                }

                const unsigned short bd = __half_as_ushort(__float2half(d));
                __stcg(&dst[0], (uint8_t)(bd & 0xFF));
                __stcg(&dst[1], (uint8_t)(bd >> 8));
            }
        }
    }
}
// ─── KV-append IQ2_S: cuantiza K/V float→IQ2_S canónico al pool.
// Layout por SB de 256 elems (82B): [d f16][qs[32]][signs[32]][qh[8]]
// [scales[8]]. Sub-bloque ib de 32, 4 grupos l de 8: idxg=qs[ib*4+l] |
// ((qh[ib]<<(8−2l))&0x300) (10 bits); signs byte DEDICADO LIBRE;
// db=d·(0.5+nibble(scales[ib],l/2))·0.25.
// ESPEJO EXACTO de encodeIQ2_S: d=max_span/(3.875·43); scan nibble por
// mitad; grid search con signos fijos sign(x); par de bits altos del
// índice escrito en qh posición 2l (inverso del shift del decode).
// ── 7.5 FIX (lane-f): era `tid!=0 return` — UN thread serializaba los 32
// SB × 2 lados × grid-search O(100k ops/SB) ⇒ prefill 20s/tok (degenerado
// del ticket 7.5). Paralelización 1:1 thread↔SB (blockDim=32, sb=32 en
// la geometría 0.8B): mismo encode serial POR SB (bit-exact), 32× de ancho
// de banda. Guard de preservación y zero previo intactos (por SB).
extern "C" __global__ void kvAppendIQ2_SKernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    uint8_t* __restrict__ cache,
    const int* __restrict__ bt,
    const int* __restrict__ start_pos,
    int n,
    int kv_dim, int n_kv_head, int head_dim, int block_size,
    int dbg)
{
    // F3 (lane-f): warp-collectivo. UN WARP por SB (warp w → SB w), 8 warps
    // = 256 threads. El grid-search de 1024 candidatos se reparte 32/lane y
    // el argmin es warp-reduce DETERMINISTA (tie → índice menor = misma
    // primera-ocurrencia del escaneo serial estricto-<). Bit-exact por
    // construcción con el encoder CPU (validado test-kvq).
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int first_block = *start_pos / block_size;
    const int last_block = (*start_pos + n - 1) / block_size;

    const int elems_region = block_size * kv_dim;
    const int sb_per_region = (elems_region + 255) / 256;
    const size_t k_bytes = (size_t)sb_per_region * 82;
    const size_t phys_stride = 2 * k_bytes;

    for (int lb = first_block; lb <= last_block; ++lb) {
        const int phys = bt[lb];
        if (phys < 0) continue;
        for (int side = 0; side < 2; ++side) {
            const float* __restrict__ src = (side == 1) ? v : k;
            uint8_t* region = cache + (size_t)phys * phys_stride
                            + (size_t)side * k_bytes;

            auto ld0 = [&](int w) -> float {
                const int off_r = w / kv_dim;
                const int t_abs = lb * block_size + off_r;
                if (w >= elems_region || t_abs < *start_pos || t_abs >= *start_pos + n) return 0.0f;
                return src[(size_t)(t_abs - *start_pos) * kv_dim + (w % kv_dim)];
            };

            // F3-next: mapeo block×warp — sb = blockIdx*nwarps + warp. Con
            // gridDim=1 el kernel entero corría en UN SM (f64 de 1 SM ≈
            // 250M FMA/s ⇒ 8.4M ops = 33ms — exacto lo medido). El launcher
            // ahora lanza ceil(sb_per_region/8) blocks ⇒ cada warp su SB,
            // repartidos por TODOS los SMs.
            const int nwarps = (int)(blockDim.x >> 5);
            const int sb = blockIdx.x * nwarps + warp;
            if (sb < sb_per_region) {
                uint8_t* dst = region + (size_t)sb * 82;
                const int sb_base = sb * 256;

                // Cache del SB: 256 elems / 32 lanes = 8 cada una (coalesced).
                float x[8];
                const int c0 = lane * 8;
                for (int w = 0; w < 8; ++w) x[w] = ld0(sb_base + c0 + w);
                // Broadcast correcto: TODAS las lanes evalúan x[q%8] con el
                // MISMO q; shfl toma el registro del lane q/8 (dueño del
                // elem). (La versión previa `owner==lane ? x : 0` solo portaba
                // el valor del propio lane — los shfl cross-lane daban 0.0f y
                // el span quedaba truncado ⇒ d ~60% menor ⇒ mismatch.)
                auto ldg = [&](int w) -> float {
                    const int q = w - sb_base;
                    return __shfl_sync(0xffffffffu, x[q & 7], (q >> 3) & 31);
                };

                // Preservación: mismo guard que Q4_K (todo el warp comparte
                // la decisión — el t_sb es igual para lanes 0..31 del SB).
                {
                    const int t_sb = lb * block_size + sb_base / kv_dim;
                    if (t_sb < *start_pos || t_sb >= *start_pos + n) continue;
                }

                if (lane == 0) {
                    // Zero previo (OR de qh asume base 0) — single writer.
                    for (int z = 0; z < 82; ++z) __stcg(&dst[z], 0);
                }

                // Span: árbol warp SIN ldg (todos los valores ya viven en
                // x[8] local). Sub-bloque s = elems [s*32,s*32+32) = x[] de
                // lanes 4s..4s+3. (1) mn/mx local de 8 elems; (2) reduce
                // segmento-de-4 lanes (shfl_xor o=1,2); (3) span por lane —
                // todas las lanes del segmento comparten el MISMO valor ⇒
                // el reduce max sobre el warp es exacto y SIN shfl-divergente.
                float mn = FLT_MAX, mx = -FLT_MAX; // lane-cuda: 1e308f fuera de rango f32 (NVRTC lo rechaza; nvcc degrada a inf)
                for (int c = 0; c < 8; ++c) {
                    mn = fminf(mn, x[c]);
                    mx = fmaxf(mx, x[c]);
                }
                #pragma unroll
                for (int o = 1; o <= 2; o <<= 1) {
                    mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, o));
                    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
                }
                float max_span = mx - mn;
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1)
                    max_span = fmaxf(max_span, __shfl_xor_sync(0xffffffffu, max_span, o));
                const float d = (max_span > 0.0f) ? max_span / (3.875f * 43.0f) : 1.0f;
                if (dbg != 0 && sb == 0 && lane == 0 && lb == 0 && side == 0)
                    printf("[iq2s-dbg] sb=0 max_span=%e d=%e\n", max_span, d);

                for (int ib = 0; ib < 8; ++ib) {
                    // Signos por grupo l: lane l∈[0,4) computa SU grupo.
                    // FIX deadlock: ldg (shfl full-mask) NUNCA bajo branch
                    // divergente — TODAS las lanes llaman ldg con el MISMO
                    // índice (lane<4 usa el resultado; lanes ≥4 lo descartan).
                    float srow[8];
                    for (int c = 0; c < 8; ++c)
                        srow[c] = ldg(sb_base + ib * 32 + lane * 8 + c);
                    float sgn_loc[8];
                    uint8_t sbits_l = 0;
                    if (lane < 4) {
                        for (int c = 0; c < 8; ++c)
                            if (srow[c] < 0.0f) sbits_l |= 1u << c;
                        for (int c = 0; c < 8; ++c)
                            sgn_loc[c] = ((sbits_l >> c) & 1) ? -1.0f : 1.0f;
                    }
                    float sgn[4][8];
                    for (int l = 0; l < 4; ++l) {
                        for (int c = 0; c < 8; ++c) {
                            const float sv = (lane == l) ? sgn_loc[c] : 0.0f;
                            sgn[l][c] = __shfl_sync(0xffffffffu, sv, l);
                        }
                        const unsigned sbv = __shfl_sync(0xffffffffu, (unsigned)sbits_l, l);
                        if (lane == 0) dst[34 + ib * 4 + l] = (uint8_t)sbv;
                    }

                    uint8_t best_nlo = 0, best_nhi = 0;
                    unsigned best_gi[4] = {0, 0, 0, 0};

                    // half UNROLLED (2 iters) — con half runtime, sgn[half*2]
                    // no es indexable estáticamente y el array entero vivía
                    // en LOCAL (208B stack frame, 168B spills). Con el unroll
                    // sgn[x][j] queda estático → registros.
                    #pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        // F3-next: reduce-paralelo-por-nn. Antes: nn outer ⇒
                        // 16×2 argmin×5-shfl SERIALIZADOS por ib (la cadena de
                        // dependencia de 160 stalls dominaba: 35.5ms/append).
                        // Ahora: cada lane escanea SU rango de 32 gi UNA vez
                        // por go, acumulando best-(e,gi) por nn en registros
                        // estáticos (nn UNROLLED); UN butterfly de 5 pasos
                        // reduce los 16 nn a la vez (las 16 cadenas cortas se
                        // solapan ⇒ ~5 rounds dependientes, no 80).
                        // ptxas con esta forma: 255 regs + 168B spills —
                        // dbn/gv se recalculan INLINE (CSE del compilador
                        // dentro de cada nn) para bajar pressure.
                        float xv[2][8];
                        for (int go = 0; go < 2; ++go) {
                            const int l = half * 2 + go;
                            for (int j = 0; j < 8; ++j)
                                xv[go][j] = ldg(sb_base + ib * 32 + l * 8 + j);
                        }
                        double beA[16], beB[16];
                        unsigned biA[16], biB[16];
                        #pragma unroll
                        for (int nn = 0; nn < 16; ++nn) {
                            beA[nn] = 1e308; biA[nn] = 0;
                            beB[nn] = 1e308; biB[nn] = 0;
                        }
                        for (int gsub = 0; gsub < 32; ++gsub) {
                            const int gi = lane * 32 + gsub;
                            const unsigned long long g = dev_iq2s_grid[gi];
                            #pragma unroll
                            for (int nn = 0; nn < 16; ++nn) {
                                const float db = d * (0.5f + (float)nn) * 0.25f;
                                // go=0
                                double e = 0.0;
                                #pragma unroll
                                for (int j = 0; j < 8; ++j) {
                                    const float gv = (float)((g >> (8 * j)) & 0xFF);
                                    const float dv = xv[0][j] - sgn[half * 2][j] * db * gv;
                                    e += (double)(dv * dv);
                                }
                                if (e < beA[nn]) { beA[nn] = e; biA[nn] = (unsigned)gi; }
                                // go=1
                                e = 0.0;
                                #pragma unroll
                                for (int j = 0; j < 8; ++j) {
                                    const float gv = (float)((g >> (8 * j)) & 0xFF);
                                    const float dv = xv[1][j] - sgn[half * 2 + 1][j] * db * gv;
                                    e += (double)(dv * dv);
                                }
                                if (e < beB[nn]) { beB[nn] = e; biB[nn] = (unsigned)gi; }
                            }
                        }
                        // Butterfly 5-pasos × 16 nn en paralelo (solapados).
                        #pragma unroll
                        for (int o = 16; o > 0; o >>= 1) {
                            #pragma unroll
                            for (int nn = 0; nn < 16; ++nn) {
                                const double eoA = __shfl_xor_sync(0xffffffffu, beA[nn], o);
                                const unsigned ioA = __shfl_xor_sync(0xffffffffu, biA[nn], o);
                                if (eoA < beA[nn] || (eoA == beA[nn] && ioA < biA[nn])) { beA[nn] = eoA; biA[nn] = ioA; }
                                const double eoB = __shfl_xor_sync(0xffffffffu, beB[nn], o);
                                const unsigned ioB = __shfl_xor_sync(0xffffffffu, biB[nn], o);
                                if (eoB < beB[nn] || (eoB == beB[nn] && ioB < biB[nn])) { beB[nn] = eoB; biB[nn] = ioB; }
                            }
                        }
                        // Selección del nn ganador (misma semántica serial:
                        // err_half asc, primera ocurrencia — el bucle nn es
                        // asc y err < half_best estricto).
                        double half_best = 1e308;
                        uint8_t hn = 0;
                        unsigned hgi[2] = {0, 0};
                        #pragma unroll
                        for (int nn = 0; nn < 16; ++nn) {
                            const double err_half = beA[nn] + beB[nn];
                            if (err_half < half_best) {
                                half_best = err_half;
                                hn = (uint8_t)nn;
                                hgi[0] = biA[nn];
                                hgi[1] = biB[nn];
                            }
                        }
                        best_gi[half * 2] = hgi[0];
                        best_gi[half * 2 + 1] = hgi[1];
                        if (half == 0) best_nlo = hn;
                        else best_nhi = hn;
                    }

                    if (lane == 0) {
                        for (int l = 0; l < 4; ++l)
                            dst[2 + ib * 4 + l] = (uint8_t)(best_gi[l] & 0xFF);
                        uint8_t sc_byte = best_nlo | (best_nhi << 4);
                        __stcg(&dst[74 + ib], sc_byte);
                        for (int l = 0; l < 4; ++l) {
                            const uint8_t hi2 = (uint8_t)((best_gi[l] >> 8) & 3);
                            dst[66 + ib] |= (uint8_t)(hi2 << (2 * l)); // inverso del decode
                        }
                    }
                }

                if (lane == 0) {
                    const unsigned short bd = __half_as_ushort(__float2half(d));
                    __stcg(&dst[0], (uint8_t)(bd & 0xFF));
                    __stcg(&dst[1], (uint8_t)(bd >> 8));
                }
            }
        }
    }
}
// ─── KV-append TQ2_0: cuantiza K/V float→TQ2_0 canónico al pool.
// 66B/SB256: [qs 64B][d f16@64]. Seg=in/128, l=(in%128)/32, m=in%32;
// byte dst[seg*32+m], shift=2l; q∈{0..3}; val=d(q−1).
// Serial tid==0 + guard preservación.
extern "C" __global__ void kvAppendTQ2_0Kernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    uint8_t* __restrict__ cache,
    const int* __restrict__ bt,
    const int* __restrict__ start_pos,
    int n,
    int kv_dim, int n_kv_head, int head_dim, int block_size,
    int dbg)
{
    const int tid = threadIdx.x;
    if (tid != 0) return;
    const int first_block = *start_pos / block_size;
    const int last_block = (*start_pos + n - 1) / block_size;

    const int elems_region = block_size * kv_dim;
    const int sb_per_region = (elems_region + 255) / 256;
    const size_t k_bytes = (size_t)sb_per_region * 66;
    const size_t phys_stride = 2 * k_bytes;

    for (int lb = first_block; lb <= last_block; ++lb) {
        const int phys = bt[lb];
        if (phys < 0) continue;
        for (int side = 0; side < 2; ++side) {
            const float* __restrict__ src = (side == 1) ? v : k;
            uint8_t* region = cache + (size_t)phys * phys_stride
                            + (size_t)side * k_bytes;

            auto ld = [&](int w) -> float {
                const int off_r = w / kv_dim;
                const int t_abs = lb * block_size + off_r;
                if (w >= elems_region || t_abs < *start_pos || t_abs >= *start_pos + n) return 0.0f;
                return src[(size_t)(t_abs - *start_pos) * kv_dim + (w % kv_dim)];
            };

            for (int sb = 0; sb < sb_per_region; ++sb) {
                uint8_t* dst = region + (size_t)sb * 66;
                const int sb_base = sb * 256;

                {
                    const int t_sb = lb * block_size + sb_base / kv_dim;
                    if (t_sb < *start_pos || t_sb >= *start_pos + n) continue;
                }

                // Zero previo (OR posterior).
                for (int z = 0; z < 64; ++z) __stcg(&dst[z], 0);

                // FIX (lane-b, regresión latente): el escaneo anterior era
                // `for (w=tid; w<256; w+=32)` + __shfl_xor_sync(0xffffffff)
                // PERO con `if (tid!=0) return` arriba ⇒ lane 0 reducía contra
                // 31 lanes EXITED (UB): leía basura del register file y d
                // salía corrupta según el codegen del cubin (fallaba append/
                // decode-step/preservación en esta GPU). Contrato serial
                // honesto: escaneo completo 256 elems SIN shuffles — idéntico
                // a encodeTQ2_0 CPU.
                float mn = FLT_MAX, mx = -FLT_MAX; // lane-cuda: 1e308f fuera de rango f32 (NVRTC lo rechaza; nvcc degrada a inf)
                for (int w = 0; w < 256; ++w) {
                    const float vv = ld(sb_base + w);
                    mn = fminf(mn, vv);
                    mx = fmaxf(mx, vv);
                }
                const float span = mx - mn;
                const float d = (span > 0.0f) ? span / 2.0f : 1.0f;

                for (int in = 0; in < 256; ++in) {
                    const int seg = in / 128;
                    const int rem = in % 128;
                    const int l = rem / 32;
                    const int m = rem % 32;
                    int q = (int)roundf(ld(sb_base + in) / d) + 1;
                    q = max(0, min(3, q));
                    dst[seg * 32 + m] |= (uint8_t)(q << (2 * l));
                }

                const unsigned short bd = __half_as_ushort(__float2half(d));
                __stcg(&dst[64], (uint8_t)(bd & 0xFF));
                __stcg(&dst[65], (uint8_t)(bd >> 8));
            }
        }
    }
}

// ─── KV-append TQ1_0: cuantiza K/V float→TQ1_0 canónico al pool.
// 54B/SB256: [qs 32B elems 0..159, 5 dígitos/byte][qs2 16B elems 160..239,
// 5/byte][qh 4B elems 240..255, 4/byte][d f16@52]. d=amax, nivel=q−1.
// Extracción de dígitos (espejo val_tq1_0 fused_decode_extra.cu:497):
// q=(u8)(byte·3^n); xi=(q·3)>>8 ∈{0,1,2}. El pack ingenuo Σt·3^n NO es
// la inversa (803/1215 estados fallan — verificado por enumeración);
// inversa por LUT runtime: por cada estado idx se busca el primer byte
// cuyo unpack lo reproduce (exacto por construcción, espejo de TQ1_LUT5/
// TQ1_LUT4 comptime de kv_quant.zig). Serial tid==0 + guard
// preservación + __stcg — contrato single-writer por sector.
// REQUIERE kv_dim % 256 == 0.
extern "C" __global__ void kvAppendTQ1_0Kernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    uint8_t* __restrict__ cache,
    const int* __restrict__ bt,
    const int* __restrict__ start_pos,
    int n,
    int kv_dim, int n_kv_head, int head_dim, int block_size,
    int dbg)
{
    const int tid = threadIdx.x;
    if (tid != 0) return;
    (void)dbg;
    const int first_block = *start_pos / block_size;
    const int last_block = (*start_pos + n - 1) / block_size;

    const int elems_region = block_size * kv_dim;
    const int sb_per_region = (elems_region + 255) / 256;
    const size_t k_bytes = (size_t)sb_per_region * 54;
    const size_t phys_stride = 2 * k_bytes;

    for (int lb = first_block; lb <= last_block; ++lb) {
        const int phys = bt[lb];
        if (phys < 0) continue;
        for (int side = 0; side < 2; ++side) {
            const float* __restrict__ src = (side == 1) ? v : k;
            uint8_t* region = cache + (size_t)phys * phys_stride
                            + (size_t)side * k_bytes;

            for (int sb = 0; sb < sb_per_region; ++sb) {
                uint8_t* dst = region + (size_t)sb * 54;
                const int sb_base = sb * 256;

                {
                    const int t_sb = lb * block_size + sb_base / kv_dim;
                    if (t_sb < *start_pos || t_sb >= *start_pos + n) continue;
                }

                for (int z = 0; z < 52; ++z) __stcg(&dst[z], 0);

                float amax = 0.0f;
                for (int w = 0; w < 256; ++w) {
                    const int off_r = (sb_base + w) / kv_dim;
                    const int t_abs = lb * block_size + off_r;
                    const int wi = sb_base + w;
                    float vv = 0.0f;
                    if (wi < elems_region && t_abs >= *start_pos && t_abs < *start_pos + n)
                        vv = src[(size_t)(t_abs - *start_pos) * kv_dim + (wi % kv_dim)];
                    amax = fmaxf(amax, fabsf(vv));
                }
                const float d = (amax > 0.0f) ? amax : 1.0f;

                // dígitos ternarios t[in] = clamp(round(v/d)+1, 0..2)
                uint8_t t[256];
                for (int in = 0; in < 256; ++in) {
                    const int off_r = (sb_base + in) / kv_dim;
                    const int t_abs = lb * block_size + off_r;
                    float vv = 0.0f;
                    if (sb_base + in < elems_region && t_abs >= *start_pos && t_abs < *start_pos + n)
                        vv = src[(size_t)(t_abs - *start_pos) * kv_dim + ((sb_base + in) % kv_dim)];
                    int q = (int)roundf(vv / d) + 1;
                    q = max(0, min(2, q));
                    t[in] = (uint8_t)q;
                }

                // pack con LUT inversa runtime (espejo tq1BuildLut): primer
                // byte cuyo unpack reproduce los dígitos. unpack(b,n):
                // q=(u8)(b·3^n); xi=(q·3)>>8.
                auto pack5 = [&](int e0) -> uint8_t {
                    for (int b = 0; b < 256; ++b) {
                        if (((int)(((uint8_t)((uint16_t)b * 1)) * 3) >> 8) != (int)t[e0]) continue;
                        if (((int)(((uint8_t)((uint16_t)b * 3)) * 3) >> 8) != (int)t[e0 + 1]) continue;
                        if (((int)(((uint8_t)((uint16_t)b * 9)) * 3) >> 8) != (int)t[e0 + 2]) continue;
                        if (((int)(((uint8_t)((uint16_t)b * 27)) * 3) >> 8) != (int)t[e0 + 3]) continue;
                        if (((int)(((uint8_t)((uint16_t)b * 81)) * 3) >> 8) != (int)t[e0 + 4]) continue;
                        return (uint8_t)b;
                    }
                    return 0; // inalcanzable: la LUT cubre los 243 estados
                };
                auto pack4 = [&](int e0) -> uint8_t {
                    for (int b = 0; b < 256; ++b) {
                        if (((int)(((uint8_t)((uint16_t)b * 1)) * 3) >> 8) != (int)t[e0]) continue;
                        if (((int)(((uint8_t)((uint16_t)b * 3)) * 3) >> 8) != (int)t[e0 + 1]) continue;
                        if (((int)(((uint8_t)((uint16_t)b * 9)) * 3) >> 8) != (int)t[e0 + 2]) continue;
                        if (((int)(((uint8_t)((uint16_t)b * 27)) * 3) >> 8) != (int)t[e0 + 3]) continue;
                        return (uint8_t)b;
                    }
                    return 0;
                };

                for (int j = 0; j < 32; ++j) __stcg(&dst[j], pack5(j * 5));
                for (int j = 0; j < 16; ++j) __stcg(&dst[32 + j], pack5(160 + j * 5));
                for (int j = 0; j < 4; ++j) __stcg(&dst[48 + j], pack4(240 + j * 4));

                const unsigned short bd = __half_as_ushort(__float2half(d));
                __stcg(&dst[52], (uint8_t)(bd & 0xFF));
                __stcg(&dst[53], (uint8_t)(bd >> 8));
            }
        }
    }
}

// ─── KV-append IQ1_M: cuantiza K/V float→IQ1_M layout entrelazado KV-path
// (56B/SB256, padding [48..56)=0). Espejo EXACTO de encodeIQ1_M (kv_quant):
//   · SIN campo f16 propio: d coarse-f16 (mantissa baja siempre 0) se
//     reensambla desde los nibbles ALTOS de los bytes impares [1,3,5,7].
//   · bytes [0..8): sc16 del par p=ib>>1 — códigos dl 3b×4 en bits [0,12)
//     (dl1-par bit0, dl2-par bit3, dl1-impar bit6, dl2-impar bit9) +
//     nibble-d en bits [12,16). ESTOS bytes SOLAPAN qb[0..8): los grupos
//     ib<2 tienen qb IMPUESTO (el índice grid incorpora los bits de escala)
//     ⇒ sólo se busca dd ∈ {±0.125}. Grupos ib≥2: qb libre ⇒ 512 candidatos
//     (qb 0..255 × dd), orden de escaneo qb-major/dd-minor.
//   · qh@32 (+sc_off): 3 bits altos del idx — l par <<8, l impar <<4 — y
//     bit de signo dd (0x08 l par / 0x80 l impar) por mitad-l.
// BÚSQUEDA COOPERATIVA (512 hilos = 1 candidato/hilo en grupos libres):
// error por candidato acumulado en f64 como (double)(dv*dv) secuencial
// ascendente — idéntico al encoder serial — y reduce argmin con desempate
// strict-< por índice de candidato ascendente ⇒ selección bit-exacta.
// Escalas/packaging en tid==0. Guard amax_all==0 (SB de padding todo-ceros):
// d_bits=0, codes=0 — evita el NaN 0/0 del camino sin guardar.
// REQUIERE kv_dim % 256 == 0. UN block + tiendas __stcg (contrato single-
// writer). Ops aritméticas con _rn explícito: sin contracción FMA ni
// aproximación de división (paridad estricta vs encoder CPU).
extern "C" __global__ void kvAppendIQ1_MKernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    uint8_t* __restrict__ cache,
    const int* __restrict__ bt,
    const int* __restrict__ start_pos,
    int n,
    int kv_dim, int n_kv_head, int head_dim, int block_size,
    int dbg)
{
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    if (nthreads != 512) return; // diseño: 512 hilos ↔ 512 candidatos
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int first_block = *start_pos / block_size;
    const int last_block = (*start_pos + n - 1) / block_size;

    const int elems_region = block_size * kv_dim;
    const int sb_per_region = (elems_region + 255) / 256;
    const size_t k_bytes = (size_t)sb_per_region * 56;
    const size_t phys_stride = 2 * k_bytes;

    __shared__ float s_amaxh[16];      // amax por mitad-grupo (ib,h)
    __shared__ int s_codes[8][2];
    __shared__ float s_dll[8][2];
    __shared__ unsigned char s_pre[8]; // bytes [0..8) empaquetados
    __shared__ float s_xv[8];          // elems del grupo en curso
    __shared__ double s_berr[16];
    __shared__ int s_bid[16];

    for (int lb = first_block; lb <= last_block; ++lb) {
        const int phys = bt[lb];
        if (phys < 0) continue;
        for (int side = 0; side < 2; ++side) {
            const float* __restrict__ src = (side == 1) ? v : k;
            uint8_t* region = cache + (size_t)phys * phys_stride
                            + (size_t)side * k_bytes;

            auto ld = [&](int w) -> float {
                const int off_r = w / kv_dim;
                const int t_abs = lb * block_size + off_r;
                if (w >= elems_region || t_abs < *start_pos || t_abs >= *start_pos + n) return 0.0f;
                return src[(size_t)(t_abs - *start_pos) * kv_dim + (w % kv_dim)];
            };

            for (int sb = 0; sb < sb_per_region; ++sb) {
                uint8_t* dst = region + (size_t)sb * 56;
                const int sb_base = sb * 256;

                // Preservación: mismo guard que Q4_K/IQ1_S (SB íntegro por
                // token; fuera del chunk ⇒ no reescribir).
                {
                    const int t_sb = lb * block_size + sb_base / kv_dim;
                    if (t_sb < *start_pos || t_sb >= *start_pos + n) continue;
                }

                // Fase A: amax por mitad-grupo (tid<16 → mitad tid).
                if (tid < 16) {
                    float mx = 0.0f;
                    #pragma unroll
                    for (int c = 0; c < 16; ++c)
                        mx = fmaxf(mx, fabsf(ld(sb_base + (tid >> 1) * 32 + (tid & 1) * 16 + c)));
                    s_amaxh[tid] = mx;
                }
                __syncthreads();

                // Fase B (serial tid0): d coarse-f16 + códigos dl + empaquetado
                // de bytes [0..8) y cero del padding [48..56).
                if (tid == 0) {
                    float amax_all = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) amax_all = fmaxf(amax_all, s_amaxh[i]);
                    unsigned short d_bits = 0;
                    float d = 0.0f;
                    if (amax_all > 0.0f) {
                        unsigned short b16 = __half_as_ushort(
                            __float2half(__fdiv_rn(amax_all, 15.0f * 1.125f)));
                        b16 = (unsigned short)(b16 + 15u);
                        b16 &= (unsigned short)0xFFF0u;
                        if (b16 >= 0x7C00u) b16 = (unsigned short)0x7BF0u;
                        d_bits = b16;
                        d = __half2float(__ushort_as_half(b16));
                    }
                    for (int i = 0; i < 8; ++i) {
                        for (int h = 0; h < 2; ++h) {
                            int cc = 0;
                            if (d > 0.0f) {
                                const float rr = roundf(__fsub_rn(
                                    __fdiv_rn(s_amaxh[i * 2 + h], __fmul_rn(2.25f, d)), 0.5f));
                                cc = (rr < 0.0f) ? 0 : ((rr > 7.0f) ? 7 : (int)rr);
                            }
                            s_codes[i][h] = cc;
                            s_dll[i][h] = __fmul_rn(
                                d, __fadd_rn(__fmul_rn(2.0f, (float)cc), 1.0f));
                        }
                    }
                    unsigned char pre[8];
                    #pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        const unsigned short sc16 = (unsigned short)(
                            ((unsigned short)s_codes[p * 2][0]) |
                            ((unsigned short)s_codes[p * 2][1] << 3) |
                            ((unsigned short)s_codes[p * 2 + 1][0] << 6) |
                            ((unsigned short)s_codes[p * 2 + 1][1] << 9));
                        pre[p * 2] = (unsigned char)(sc16 & 0xFF);
                        pre[p * 2 + 1] = (unsigned char)(
                            ((sc16 >> 8) & 0xF) | ((((unsigned)d_bits >> (4 * p)) & 0xF) << 4));
                    }
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        s_pre[i] = pre[i];
                        __stcg(&dst[i], pre[i]);
                    }
                    #pragma unroll
                    for (int i = 48; i < 56; ++i) __stcg(&dst[i], (unsigned char)0);
                }
                __syncthreads();

                // Imagen local de qh (la materializa tid0 al cierre del SB).
                unsigned char qh_img[16];
                if (tid == 0) {
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) qh_img[i] = 0;
                }

                // Fase C: 32 grupos × búsqueda cooperativa del quanta.
                for (int g = 0; g < 32; ++g) {
                    const int ib = g >> 2;
                    const int l = g & 3;
                    const int qb_pos = ib * 4 + l;
                    const bool free_g = qb_pos >= 8;
                    const int ncand = free_g ? 512 : 2;

                    if (tid < 8) s_xv[tid] = ld(sb_base + ib * 32 + l * 8 + tid);
                    __syncthreads();

                    const float dll = s_dll[ib][l >> 1];
                    const int shift_amt = (l & 1) ? 4 : 8;
                    const int fixed_qb = free_g ? 0 : (int)s_pre[qb_pos];

                    // Sentinel: err infinito + índice máximo ⇒ cualquier
                    // candidato activo gana (empate por índice ascendente).
                    double be = 1e308; // lane-cuda: INFINITY indefinido bajo NVRTC; 1e308 < DBL_MAX f64 legal
                    int bi = 0x7fffffff;
                    if ((int)tid < ncand) {
                        const int h = free_g ? (int)(tid & 1) : (int)tid;
                        const int qb_val = free_g ? (int)(tid >> 1) : fixed_qb;
                        const float dd = (h == 0) ? 0.125f : -0.125f;
                        const int idxg = qb_val | ((h << shift_amt) & 0x700);
                        const unsigned long long gg = dev_iq1s_grid[idxg];
                        double e = 0.0;
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            const float raw =
                                (float)(int)((signed char)((gg >> (8 * j)) & 0xFFULL));
                            const float dv = __fsub_rn(
                                s_xv[j], __fmul_rn(dll, __fadd_rn(raw, dd)));
                            e += (double)__fmul_rn(dv, dv);
                        }
                        be = e;
                        bi = (int)tid;
                    }
                    // Reduce argmin (err asc, tie→índice menor): equivalente
                    // exacto del escaneo serial strict-< ascendente.
                    #pragma unroll
                    for (int o = 16; o > 0; o >>= 1) {
                        const double eo = __shfl_xor_sync(0xffffffffu, be, o);
                        const int io = __shfl_xor_sync(0xffffffffu, bi, o);
                        if (eo < be || (eo == be && io < bi)) { be = eo; bi = io; }
                    }
                    if (lane == 0) { s_berr[warp] = be; s_bid[warp] = bi; }
                    __syncthreads();
                    if (tid == 0) {
                        double we = s_berr[0];
                        int wi = s_bid[0];
                        for (int w = 1; w < 16; ++w) {
                            if (s_berr[w] < we || (s_berr[w] == we && s_bid[w] < wi)) {
                                we = s_berr[w];
                                wi = s_bid[w];
                            }
                        }
                        s_bid[0] = wi;
                    }
                    __syncthreads();

                    // Fase D: escrituras (single writer).
                    if (tid == 0) {
                        const int win = s_bid[0];
                        int qb_val, hb;
                        if (free_g) { qb_val = win >> 1; hb = win & 1; }
                        else        { qb_val = fixed_qb; hb = win; }
                        if (free_g) __stcg(&dst[qb_pos], (unsigned char)qb_val);
                        const int qh_i = (ib >> 1) * 2 + (l >> 1);
                        if ((l & 1) == 0) {
                            qh_img[qh_i] |= (unsigned char)(hb & 1);
                            if (hb != 0) qh_img[qh_i] |= 0x08;
                        } else {
                            qh_img[qh_i] |= (unsigned char)((hb & 1) << 4);
                            if (hb != 0) qh_img[qh_i] |= 0x80;
                        }
                        if (dbg != 0 && sb < 2 && g < 2)
                            printf("[iq1md] sb=%d g=%d cand=%d\n", sb, g, win);
                    }
                    __syncthreads();
                }

                // qh embebido (bytes disjuntos del resto).
                if (tid == 0) {
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) __stcg(&dst[32 + i], qh_img[i]);
                }
                __syncthreads();
            }
        }
    }
}
// ─── Gate: attn[i] *= sigmoid(g[i]) ─────────────────────────────────────────
extern "C" __global__ void gateKernel(
    float* __restrict__ attn,
    const float* __restrict__ g,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) attn[i] *= 1.0f / (1.0f + expf(-g[i]));
}

// ─── embedding gather: out[1,n_embd] = emb[token, :] (f16) ────────────────────
extern "C" __global__ void embeddingGatherKernel(
    const half* __restrict__ emb, const int token, half* __restrict__ out, int n_embd)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_embd) out[i] = emb[(size_t)token * n_embd + i];
}

// ─── Q4_0 M=1 GEMM: C[1,N] = A[1,K] * B_q4[K,N] ─────────────────────────────
// B es el peso Q4_0 tal cual está en el GGUF (layout [in,out] = [K,N], bloques
// de 32 a lo largo del dim contiguo K). Cada fila de salida j ocupa bytes
// contiguos: (K/32) bloques de 18 bytes. A: f32 [K], C: f32 [N].
// Un warp por fila de salida; grid cubre N/8 filas.
// K debe ser múltiplo de 32.
// ─── STUDY §5.8: M=1 q4_0 GEMV — A-quantize FUSED + dp4a, UN lanzamiento ─────
// Perfil (nsys, NOGRAPH): los qgemm GEMVs son 68% del decode a ~118 GB/s
// (26% HBM). El clásico q4gemmM1 serializa la extracción de nibbles por
// lane (FMA escalar); el MMQ 2-launch paga 2 lanzamientos + memset que en
// formas SSM (1-3 MB) cuestan más que el cómputo. Este kernel FUSIONA:
//   fase 1: warp 0 cuantiza A[K] f32 → s_aq i8 (dp4a-natural, GGUF
//           super-bloques de 32: amax shfl → d f16 → q = round(v/d))
//           + s_d (d f32) + s_sa (Σq i32 → corrección unsigned-nibble)
//   fase 2: cada warp = una fila N; recorre TODOS los bloques K con las
//           4×u32 funnel-shift (stride 18B) + 8 dp4a — MISMA matemática
//           del mmqQ4_0GEMV (bit-paridad dp4a), SIN split-K ni atomics:
//           a N≥1024 un warp por fila satura mejor que 8 filas/bloque.
// Requisitos: K % 32 == 0, blockDim 256 (8 warps), N ≥ gridDim que llene.
// Gate rel<1e-3 vs q4gemmM1 (mismo A cuantizado ⇒ mismo valor, distinto
// orden de reducción por el árbol shfl — como §5.6).
extern "C" __global__ void q4gemmM1Dp4aKernel(
    const float* __restrict__ a,
    const unsigned char* __restrict__ b,
    float* __restrict__ c,
    int K, int N)
{
    extern __shared__ unsigned char smem_raw[];
    const int kb_total = K >> 5;
    // Layout shared: s_aq [K pad16] | s_d f32 [KB] | s_sa f32 [KB].
    int8_t* s_aq = (int8_t*)smem_raw;
    float* s_d = (float*)(smem_raw + ((K + 15) & ~15));
    float* s_sa = s_d + kb_total;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    // ── Fase 1: cuantizar A (8 warps reparten los KB bloques — 1 warp/bloque,
    // amax por shfl; idéntico al mmqQuantizeAQ8Kernel pero dentro del launch).
    for (int kb = warp; kb < kb_total; kb += (blockDim.x >> 5)) {
        const int base = kb * 32 + lane;
        const float v = a[base];
        float amax = fabsf(v);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float df = (amax > 0.0f) ? amax / 127.0f : 1.0f;
        int q = (int)roundf(v / df);
        q = max(-127, min(127, q));
        s_aq[base] = (int8_t)q;
        // Σq del bloque: TODAS las lanes participan en la reducción (un
        // __shfl con máscara full ejecutado por un subconjunto es UB/hang —
        // bug del primer WIP); lane 0 escribe el resultado.
        int sq = q;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if (lane == 0) {
            s_d[kb] = __half2float(__float2half(df));
            s_sa[kb] = (float)sq;
        }
    }
    __syncthreads();

    // ── Fase 2: GEMV dp4a — un warp por fila (grid: (N+7)/8 bloques).
    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= N) return;
    const unsigned char* rowb = b + (size_t)row * ((size_t)kb_total * 18);

    float acc = 0.0f;
    for (int kb = lane; kb < kb_total; kb += 32) {
        const unsigned char* blk = rowb + (size_t)kb * 18;
        // 16B de quanta como 4×u32 alineadas + funnel-shift (stride 18B).
        uint32_t u[4];
        {
            const uint32_t* vp = (const uint32_t*)((uintptr_t)(blk + 2) & ~(uintptr_t)3);
            const uint32_t sh = (uint32_t)(((uintptr_t)(blk + 2) & 3) * 8);
            const uint32_t r0 = vp[0], r1 = vp[1], r2 = vp[2], r3 = vp[3], r4 = vp[4];
            u[0] = __funnelshift_r(r0, r1, sh);
            u[1] = __funnelshift_r(r1, r2, sh);
            u[2] = __funnelshift_r(r2, r3, sh);
            u[3] = __funnelshift_r(r3, r4, sh);
        }
        const float db = __half2float(*(const __half*)blk);
        const int8_t* abase = s_aq + kb * 32;
        const int a0 = *(const int*)(abase);
        const int a1 = *(const int*)(abase + 4);
        const int a2 = *(const int*)(abase + 8);
        const int a3 = *(const int*)(abase + 12);
        const int a4 = *(const int*)(abase + 16);
        const int a5 = *(const int*)(abase + 20);
        const int a6 = *(const int*)(abase + 24);
        const int a7 = *(const int*)(abase + 28);
        // SPLIT-16 canónico: lows de u[t] = elems 4t..4t+3; highs = 16+4t...
        const uint32_t lo_m = 0x0F0F0F0Fu;
        int sn = 0;
        sn = __dp4a((int)(u[0] & lo_m), a0, sn);
        sn = __dp4a((int)(u[1] & lo_m), a1, sn);
        sn = __dp4a((int)(u[2] & lo_m), a2, sn);
        sn = __dp4a((int)(u[3] & lo_m), a3, sn);
        sn = __dp4a((int)((u[0] >> 4) & lo_m), a4, sn);
        sn = __dp4a((int)((u[1] >> 4) & lo_m), a5, sn);
        sn = __dp4a((int)((u[2] >> 4) & lo_m), a6, sn);
        sn = __dp4a((int)((u[3] >> 4) & lo_m), a7, sn);
        // Corrección unsigned-nibble: Σ a·q = Σ a·n − 8·Σa.
        const int dot_true = sn - 8 * (int)s_sa[kb];
        acc += db * s_d[kb] * (float)dot_true;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) c[row] = acc;
}

// ─── TODO 1.3 (lane-f): M1 q4_0 dp4a sobre peso REPACKED dual-view ───────────
// Variante del §5.8 M1 que consume el layout repackeado on-load
// (q4PackedWeight): payload 16B uint4-aligned contiguo por fila + escalas
// d en array propio. El lane lee 1×uint4 por KB (16B) en vez del
// funnel-shift de 5 loads sobre el SB de 18B desalineado — menos
// transacciones por warp-iter (9→8 sectores ideales, sin overlaps).
// Matemática IDÉNTICA al §5.8 (dp4a split-16 + corrección −8·Σa):
// bit-exact por construcción (mismo contenido, otra dirección).
//   b_payload: [N][kb_total*16] u8 | b_d: [N*kb_total] f16 (fila r en b_d + r*kb_total)
extern "C" __global__ void q4gemmM1Dp4aPackedKernel(
    const float* __restrict__ a,
    const unsigned char* __restrict__ b_payload,
    const __half* __restrict__ b_d,
    float* __restrict__ c,
    int K, int N)
{
    extern __shared__ unsigned char smem_raw[];
    const int kb_total = K >> 5;
    int8_t* s_aq = (int8_t*)smem_raw;
    float* s_d = (float*)(smem_raw + ((K + 15) & ~15));
    float* s_sa = s_d + kb_total;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    // Fase 1: cuantizar A (idéntico §5.8 — un warp por KB, amax shfl).
    for (int kb = warp; kb < kb_total; kb += (blockDim.x >> 5)) {
        const int base = kb * 32 + lane;
        const float v = a[base];
        float amax = fabsf(v);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float df = (amax > 0.0f) ? amax / 127.0f : 1.0f;
        int q = (int)roundf(v / df);
        q = max(-127, min(127, q));
        s_aq[base] = (int8_t)q;
        int sq = q;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if (lane == 0) {
            s_d[kb] = __half2float(__float2half(df));
            s_sa[kb] = (float)sq;
        }
    }
    __syncthreads();

    // Fase 2: GEMV dp4a packed — un warp por fila; el lane lee SU uint4.
    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= N) return;
    const uint4* rowp = (const uint4*)(b_payload + (size_t)row * (size_t)kb_total * 16);
    const __half* rowd = b_d + (size_t)row * kb_total;

    float acc = 0.0f;
    for (int kb = lane; kb < kb_total; kb += 32) {
        const uint4 u = rowp[kb];              // 16B en UNA transacción
        const float db = __half2float(rowd[kb]);
        const int8_t* abase = s_aq + kb * 32;
        const int a0 = *(const int*)(abase);
        const int a1 = *(const int*)(abase + 4);
        const int a2 = *(const int*)(abase + 8);
        const int a3 = *(const int*)(abase + 12);
        const int a4 = *(const int*)(abase + 16);
        const int a5 = *(const int*)(abase + 20);
        const int a6 = *(const int*)(abase + 24);
        const int a7 = *(const int*)(abase + 28);
        const uint32_t lo_m = 0x0F0F0F0Fu;
        int sn = 0;
        sn = __dp4a((int)(u.x & lo_m), a0, sn);
        sn = __dp4a((int)(u.y & lo_m), a1, sn);
        sn = __dp4a((int)(u.z & lo_m), a2, sn);
        sn = __dp4a((int)(u.w & lo_m), a3, sn);
        sn = __dp4a((int)((u.x >> 4) & lo_m), a4, sn);
        sn = __dp4a((int)((u.y >> 4) & lo_m), a5, sn);
        sn = __dp4a((int)((u.z >> 4) & lo_m), a6, sn);
        sn = __dp4a((int)((u.w >> 4) & lo_m), a7, sn);
        const int dot_true = sn - 8 * (int)s_sa[kb];
        acc += db * s_d[kb] * (float)dot_true;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) c[row] = acc;
}

// ─── STUDY §5.9: M≤32 q4_0 GEMM dp4a — prefill tier (fase-1 quantize ×M) ────
// Spreen-2 assignment #1. Perfil del coordinator: qgemmKernel con grid.y=m
// re-lee la MATRIZ DE PESO COMPLETA desde HBM por cada token (512× en pp512).
// Este kernel cuantiza las M activaciones en shared (una sola vez) y cada
// warp computa UNA fila de salida para TODOS los M tokens leyendo el peso
// UNA vez desde HBM (los M dp4a del mismo bloque pesan desde L1/L2).
// Matemática = §5.8 (dp4a + funnel-shift 18B + corrección −8·Σa) por (fila, m).
//
// Layout shared: s_aq [M*K pad16] | s_d f32 [M*KB] | s_sa f32 [M*KB]
//   M=32, K=2048: 64KB + 8KB + 8KB = 80KB (≤99KB opt-in sm_86)
//   M=32, K=1024: 32KB + 4KB + 4KB = 40KB
// Fase 1: M*K bloques de 32 repartidos entre los 8 warps (un warp por bloque:
//        amax shfl → d → q por elemento; TODAS las lanes en cada reducción).
// Fase 2: grid ((N+7)/8), block 256; warp → fila j; por cada m < M_real el
//        warp recorre los KB bloques (peso compartido por los M — L1 hit) y
//        acumula acc[m] con el MISMO convenio dp4a del §5.8.
// M_real llega como parámetro (m_chunk): el dispatch trocea m>32 en
// lanzamientos de ≤32 (512 → 16 launches por proyección).
// ─── STUDY §5.9 v2: M q4_0 GEMM dp4a — warp-per-token, peso 1× por m-tile ──
// Rediseño tras NO-GO de la v1 (m-loop serializado por warp: 0.56× — el
// paralelismo de tokens importaba más que el ahorro de lectura). v2:
//   grid = ((N+7)/8, ceil(M/8)); block = (32, 8) — UN WARP POR TOKEN.
//   fase 1: el bloque cuantiza sus 8 tokens [M_blk*K] a shared (cooperativo,
//           un warp por bloque (m,kb) — igual que v1).
//   fase 2: los 8 warps recorren JUNTOS los mismos kb del mismo row (lockstep):
//           el peso se lee de HBM 1× por (row, m-tile) y de L1 para los 8
//           warps (misma línea); cada warp dp4a-su token con registers.
//   Tráfico de peso: M/8× menos que el clásico (64 m-tiles a 512 tok vs 512
//   bloques-y clásicos). Paralelismo: full (token por warp, fila por blockIdx.x).
// Matemática dp4a idéntica a §5.8/v1 (funnel-shift 18B, −8·Σa, SPLIT-16).
// ─── STUDY §5.9 v3: GEMM q4_0 M dp4a — grid clásico (m en blockIdx.y) ────────
// Verdad estructural tras v0 (0.65×, m serial/warp) y v2 (0.20×, 1 fila/bloque
// sub-ocupado): el grid del CLÁSICO ((N+7)/8 × M, block 256, warp=fila) ya es
// la forma correcta — L2 absorbe la mayor parte del re-read m× del peso. La
// ÚNICA ineficiencia real es su inner-loop de FMA escalar por lane.
// v3 = clásico EXACTO + fase-1 de cuantización (el bloque cuantiza SU token a
// shared, un warp por kb) + inner-loop dp4a (los mismos 8 elems por lane que
// el clásico lee como f32, ahora i8-packed) + corrección −8·Σa y escalas del
// bloque. Tráfico de peso IDÉNTICO al clásico (m× con L2); cómputo ~2-4×.
extern "C" __global__ void q4gemmMDp4aKernel(
    const float* __restrict__ a,          // [M*K]
    const unsigned char* __restrict__ b,  // [N][KB*18] q4_0
    float* __restrict__ c,                // [M*N]
    int K, int N, int M)
{
    extern __shared__ unsigned char smem_raw[];
    const int kb_total = K >> 5;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int nwarps = blockDim.x >> 5;   // 8
    const int m = blockIdx.y;
    const float* am = a + (size_t)m * K;

    int8_t* s_aq = (int8_t*)smem_raw;           // [K]
    float* s_d = (float*)(smem_raw + ((K + 15) & ~15));
    float* s_sa = s_d + kb_total;

    // ── Fase 1: cuantizar EL token m de este bloque (un warp por kb) ──────
    for (int kb = warp; kb < kb_total; kb += nwarps) {
        const float v = am[kb * 32 + lane];
        float amax = fabsf(v);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float df = (amax > 0.0f) ? amax / 127.0f : 1.0f;
        int q = (int)roundf(v / df);
        q = max(-127, min(127, q));
        s_aq[kb * 32 + lane] = (int8_t)q;
        int sq = q;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if (lane == 0) {
            s_d[kb] = __half2float(__float2half(df));
            s_sa[kb] = (float)sq;
        }
    }
    __syncthreads();

    // ── Fase 2: EXACTAMENTE el grid del clásico — warp = fila ─────────────
    const int row = blockIdx.x * nwarps + warp;
    if (row >= N) return;
    const unsigned char* rowb = b + (size_t)row * ((size_t)kb_total * 18);

    float acc = 0.0f;
    for (int kb = lane; kb < kb_total; kb += 32) {
        const unsigned char* blk = rowb + (size_t)kb * 18;
        uint32_t u[4];
        {
            const uint32_t* vp = (const uint32_t*)((uintptr_t)(blk + 2) & ~(uintptr_t)3);
            const uint32_t sh = (uint32_t)(((uintptr_t)(blk + 2) & 3) * 8);
            const uint32_t r0 = vp[0], r1 = vp[1], r2 = vp[2], r3 = vp[3], r4 = vp[4];
            u[0] = __funnelshift_r(r0, r1, sh);
            u[1] = __funnelshift_r(r1, r2, sh);
            u[2] = __funnelshift_r(r2, r3, sh);
            u[3] = __funnelshift_r(r3, r4, sh);
        }
        const float db = __half2float(*(const __half*)blk);
        const int8_t* abase = s_aq + kb * 32;
        const int a0 = *(const int*)(abase);
        const int a1 = *(const int*)(abase + 4);
        const int a2 = *(const int*)(abase + 8);
        const int a3 = *(const int*)(abase + 12);
        const int a4 = *(const int*)(abase + 16);
        const int a5 = *(const int*)(abase + 20);
        const int a6 = *(const int*)(abase + 24);
        const int a7 = *(const int*)(abase + 28);
        const uint32_t lo_m = 0x0F0F0F0Fu;
        int sn = 0;
        sn = __dp4a((int)(u[0] & lo_m), a0, sn);
        sn = __dp4a((int)(u[1] & lo_m), a1, sn);
        sn = __dp4a((int)(u[2] & lo_m), a2, sn);
        sn = __dp4a((int)(u[3] & lo_m), a3, sn);
        sn = __dp4a((int)((u[0] >> 4) & lo_m), a4, sn);
        sn = __dp4a((int)((u[1] >> 4) & lo_m), a5, sn);
        sn = __dp4a((int)((u[2] >> 4) & lo_m), a6, sn);
        sn = __dp4a((int)((u[3] >> 4) & lo_m), a7, sn);
        const int dot_true = sn - 8 * (int)s_sa[kb];
        acc += db * s_d[kb] * (float)dot_true;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    // REGRESIÓN c817c4a (lane-a, reportada por lane-b 2026-09-11): mi fix
    // del reduce equivocado (edit de fase 2 que matcheó este tail por el
    // shuffle idéntico) reescribió el write como c[row] — todas las slices
    // m (blockIdx.y) pisaban la slice 0: write crow, prefill q4_0 M>1 roto.
    // Restaurado el índice [M][N] row-major ORIGINAL.
    if (lane == 0) c[(size_t)m * N + row] = acc;
}


// Q5_K scale decoder (espejo de lane_get_scale_min_k5 en fused_decode_extra.cu):
// scales[i]: bits 0-5 = d (6 bits), bits 6-7 = nibble alto compartido con
// scales[i+4]. Ver GGML q5_K get_scale_min_k5.
__device__ __forceinline__ static void lane_get_scale_min_k5(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) { d = q[j] & 63;     m = q[j + 4] & 63; }
    else       { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                 m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// ─── STUDY 1.1: GEMV q5_k M=1 — ssm_out del 0.8B (2048→1024) ────────────────
// Layout canónico Q5_K: SB 256 elems / 176B. [d f16@0][m f16@2][scales 12B@4
// (4 grupos × 3B: 6-bit d1 + 6-bit m1, ver lane_get_scale_min_k5)][qh 32B@16
// (1 bit/elem)][qs 128B@48 (nibble/2 elems, par/impar)].
// Grupo = 64 elems: 32B qs + 8B qh; sub-escalas lo/hi por 32 elems.
//
// M=1 GEMV: warp=fila; lane procesa 2 elems del grupo (par/impar del MISMO
// byte de qs). A se lee en f32 desde smem (el dp4a no amortiza en M=1 con
// el layout intercalado de Q5_K — el inner es FMA escalar como el clásico;
// el win viene del smem en vez de re-reads DRAM de A y del shfl único).
// ESPEJO EXACTO de la semántica de lane_q5k_val (case 2 del qgemmKernel):
//   elem global w: g=w>>6, t=w&63, hi=t>=32, ir=(t&31)>>1, par=t&1
//   qs[32*g + 2*ir + par], qh bit (2*g + hi), d1=dall*sd/2, m1=dmin*sm/2
//   val = d1*(nib + 16*extra) - m1
extern "C" __global__ void q5gemmM1Kernel(
    const float* __restrict__ a,
    const unsigned char* __restrict__ b,
    float* __restrict__ c,
    int K, int N)
{
    extern __shared__ float sa[];
    const int sb_total = K >> 8;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    for (int i = tid; i < K; i += blockDim.x) sa[i] = a[i];
    __syncthreads();

    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= N) return;
    const unsigned char* rowb = b + (size_t)row * ((size_t)sb_total * 176);

    float acc = 0.0f;
    for (int sb = 0; sb < sb_total; sb++) {
        const unsigned char* blk = rowb + (size_t)sb * 176;
        const float dall = __half2float(*(const __half*)(blk));
        const float dmin = __half2float(*(const __half*)(blk + 2));
        const unsigned char* scales = blk + 4;
        const unsigned char* qh = blk + 16;
        const unsigned char* qs = blk + 48;
        // Un SB = 256 elems; el lane cubre 8 elems (2 por grupo × 4 grupos).
        // w = g*64 + hi*32 + lane: hi=0 (elems 0..31, escala 2*g) y
        // hi=1 (elems 32..63, escala 2*g+1). Ambos leen qs[32*g+2*ir+par]
        // con el MISMO byte (nibble bajo/alto) — 1 lectura/byte útil.
        for (int g = 0; g < 4; g++) {
            const int ir = lane >> 1;
            const int parity = lane & 1;
            const unsigned char qbyte = qs[32 * g + 2 * ir + parity];
            const unsigned char qbh = qh[2 * ir + parity];
            // sub-lo: t=lane (hi=0) → nib BAJO; bit qh (2*g).
            {
                int sd, sm;
                lane_get_scale_min_k5(2 * g, scales, sd, sm);
                const float d1 = dall * (float)sd;
                const float m1 = dmin * (float)sm;
                const int nib = qbyte & 0xF;
                const int extra = (qbh >> (2 * g)) & 1;
                const int w = g * 64 + lane;
                acc += sa[sb * 256 + w] * (d1 * (float)(nib + 16 * extra) - m1);
            }
            // sub-hi: t=32+lane (hi=1) → nib ALTO; bit qh (2*g+1).
            {
                int sd, sm;
                lane_get_scale_min_k5(2 * g + 1, scales, sd, sm);
                const float d1 = dall * (float)sd;
                const float m1 = dmin * (float)sm;
                const int nib = qbyte >> 4;
                const int extra = (qbh >> (2 * g + 1)) & 1;
                const int w = g * 64 + 32 + lane;
                acc += sa[sb * 256 + w] * (d1 * (float)(nib + 16 * extra) - m1);
            }
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) c[row] = acc;
}

// ─── STUDY 1.2: GEMV q6_k M=1 — lm_head/FFN-down del 0.8B ────────────────────
// Layout Q6_K: SB 256 elems / 210B. [ql 128B@0][qh 32B@128][sc 16B@192][d f16@208].
// Elem w: ip=w>>7, r=w&127, il=r&31, j=r>>5.
//   qlb = ql[64*ip + il + (j&1)*32], nib = (j>>1)?(qlb>>4):(qlb&0xF)
//   qhb = qh[32*ip + il], qbits = (qhb>>(2j))&3, packed = nib|qbits<<4 ∈ [0,63]
//   sc  = sc[192 + 8*ip + il/16 + 2*j] (i8), d = f16@208
//   val = d * sc * (packed − 32)
// M=1: warp=fila; lane fija il (0..31) y recorre (ip,j) — 8 elems/lane/SB.
// A en smem compartido entre las 8 filas del bloque (mismo win que 1.1).
// ESPEJO EXACTO de lane_q6k_val (case 3 del qgemmKernel) — bit-paridad.
extern "C" __global__ void q6gemmM1Kernel(
    const float* __restrict__ a,
    const unsigned char* __restrict__ b,
    float* __restrict__ c,
    int K, int N)
{
    extern __shared__ float sa[];
    const int sb_total = K >> 8;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    for (int i = tid; i < K; i += blockDim.x) sa[i] = a[i];
    __syncthreads();

    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= N) return;
    const unsigned char* rowb = b + (size_t)row * ((size_t)sb_total * 210);

    float acc = 0.0f;
    const int il = lane;
    for (int sb = 0; sb < sb_total; sb++) {
        const unsigned char* blk = rowb + (size_t)sb * 210;
        const float d = __half2float(*(const __half*)(blk + 208));
        for (int ip = 0; ip < 2; ip++) {
            for (int j = 0; j < 4; j++) {
                const uint8_t qlb = blk[64 * ip + il + (j & 1) * 32];
                const int nib = (j >> 1) ? (qlb >> 4) : (qlb & 0xF);
                const uint8_t qhb = blk[128 + 32 * ip + il];
                const int packed = nib | (((qhb >> (2 * j)) & 3) << 4);
                const int8_t sc = (int8_t)blk[192 + 8 * ip + il / 16 + 2 * j];
                const int w = sb * 256 + ip * 128 + j * 32 + il;
                acc += sa[w] * (d * (float)sc * (float)(packed - 32));
            }
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) c[row] = acc;
}

extern "C" __global__ void q4gemmM1Kernel(
    const float* __restrict__ a,
    const unsigned char* __restrict__ b,
    float* __restrict__ c,
    int K, int N)
{
    extern __shared__ float sa[];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    for (int i = tid; i < K; i += blockDim.x) sa[i] = a[i];
    __syncthreads();

    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= N) return;

    const int nblocks = K >> 5;
    const unsigned char* rowb = b + (size_t)row * (nblocks * 18);

    float acc = 0.0f;
    for (int blk = 0; blk < nblocks; blk++) {
        const unsigned char* blkptr = rowb + (size_t)blk * 18;
        const float d = __half2float(*(const __half*)blkptr);
        const unsigned char q = blkptr[2 + (lane & 15)];
        const int nibble = (lane < 16) ? (q & 0x0F) : (q >> 4);
        acc += sa[blk * 32 + lane] * (d * (float)(nibble - 8));
    }
    for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) c[row] = acc;
}

// ─── U3 (lane-a, eje §12): GEMV q4_k M=1 — hueco del Q4_K_XL/gate 3B ───────
// PERF_STAGE Q4_K_XL: ssm 67.9ms = 97% del token; las proyecciones q4_k
// caían al qgemm GENÉRICO (28 crumbs [§12-generic] qtype=4 m=1 n=3584).
// Q5_K_S todo-especializado corre a 153 t/s vs 50 del Q4_K_XL ⇒ éste es
// EL eslabón del gate 3B (Q3_K_S = q4_k/q3_k dominante en attn_*/ffn_*).
//
// Layout Q4_K: SB 256 elems / 144B. [d f16@0][dmin f16@2][scales 12B@4
// (8 sub-escalas 6-bit con spill, get_scale_min_k4)][qs 128B@16 (nibble/
// 2 elems, par/impar)]. Sub-bloque = 32 elems (8 por SB), escala is:
//   dv = d * (is<4 ? sc[is]&63 : (sc[is+4]&0xF)|((sc[is-4]>>6)<<4))
//   mv = mn * (is<4 ? sc[is+4]&63 : (sc[is+4]>>4)|((sc[is]>>6)<<4))
//   val = dv*(nib) - mv, nib = is par? low : high de qs[(is>>1)*32 + lane]
// M=1: warp=fila; lane cubre los 32 elems del sub-bloque (acceso
// coalescido, escalas uniformes en el warp — el unpack ramificado es
// POR SUB-BLOQUE, no por elemento). A en smem compartido entre las 8
// filas del bloque (mismo win que 1.1-q5/1.2-q6).
// ESPEJO EXACTO de la semántica del case 4 del qgemmKernel — bit-paridad
// por construcción (misma aritmética f32, mismo orden de acumulación).
extern "C" __global__ void q4kGemmM1Kernel(
    const float* __restrict__ a,
    const unsigned char* __restrict__ b,
    float* __restrict__ c,
    int K, int N)
{
    extern __shared__ float sa[];
    const int sb_total = K >> 8;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    for (int i = tid; i < K; i += blockDim.x) sa[i] = a[i];
    __syncthreads();

    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= N) return;
    const unsigned char* rowb = b + (size_t)row * ((size_t)sb_total * 144);

    float acc = 0.0f;
    for (int sb = 0; sb < sb_total; sb++) {
        const unsigned char* blk = rowb + (size_t)sb * 144;
        const float d = __half2float(*(const __half*)blk);
        const float mn = __half2float(*(const __half*)(blk + 2));
        const unsigned char* sc = blk + 4;
        const unsigned char* qs = blk + 16;
        const float* sa_base = sa + (size_t)sb * 256;
        for (int is = 0; is < 8; ++is) {
            const int c64 = is >> 1;
            const float dv = d * (float)((is < 4) ? (sc[is] & 63)
                : ((sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4)));
            const float mv = mn * (float)((is < 4) ? (sc[is + 4] & 63)
                : ((sc[is + 4] >> 4) | ((sc[is] >> 6) << 4)));
            if ((is & 1) == 0) {
                acc += sa_base[is * 32 + lane]
                     * (dv * (float)(qs[c64 * 32 + lane] & 0xF) - mv);
            } else {
                acc += sa_base[is * 32 + lane]
                     * (dv * (float)(qs[c64 * 32 + lane] >> 4) - mv);
            }
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) c[row] = acc;
}

// ─── U3 (lane-a, eje §12): GEMV q8_0 M=1 — 2º hueco del Q4_K_XL ──────────────
// Tras el q4_k M=1, el residual genérico m=1 era q8_0 (18 crumbs
// [§12-generic] qtype=5 n=1024 k=2048/token: bancos ssm_* del ΔNet).
// Layout Q8_0: bloque 32 elems / 34B [d f16@0][i8×32@2]. Lane cubre los
// 32 elems — acceso coalescido 1 byte/lane, FMA escalar. Espejo EXACTO
// del case 5 del qgemmKernel (misma aritmética, mismo orden). A en smem
// compartido entre las 8 filas del bloque (patrón 1.1/1.2/q4k).
extern "C" __global__ void q8kGemmM1Kernel(
    const float* __restrict__ a,
    const unsigned char* __restrict__ b,
    float* __restrict__ c,
    int K, int N)
{
    extern __shared__ float sa[];
    const int nb_total = K >> 5;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    for (int i = tid; i < K; i += blockDim.x) sa[i] = a[i];
    __syncthreads();

    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= N) return;
    const unsigned char* rowb = b + (size_t)row * ((size_t)nb_total * 34);

    float acc = 0.0f;
    for (int blk = 0; blk < nb_total; blk++) {
        const unsigned char* bp = rowb + (size_t)blk * 34;
        const float d = __half2float(*(const __half*)bp);
        acc += sa[blk * 32 + lane]
             * (d * (float)(int8_t)bp[2 + lane]);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) c[row] = acc;
}

// a-U3: load u32 desde dirección posiblemente 2B-alineada (stride 110B).
// Patrón §5.8 (stride 18B): dos loads alineados + funnel-shift.
__device__ __forceinline__ uint32_t ld32_unaligned(const unsigned char* p) {
    const uint32_t* vp = (const uint32_t*)((uintptr_t)p & ~(uintptr_t)3);
    return __funnelshift_r(vp[0], vp[1], (uint32_t)(((uintptr_t)p & 3) * 8));
}

// ─── a-U3 (lane-a, eje §12): GEMV q3_k M=1 dp4a — EL cuello del gate 3B ───────
// Llama-3.2-3B-Q3_K_S es TODO q3_k (attn_*/ffn_*): el case 6 escalar del
// qgemmKernel explica los 712µs/capa medidos (≈10 instr/elem de extracción
// 2-bit + hmask + escalas a 326 Ginstr/s ⇒ compute-bound, 16% del techo HBM
// — PERF_STAGE 2026-09-10, análisis lane-a). Formulación vec_dot_q3_K_q8_1
// de llama.cpp (vecdotq.cuh:450) adaptada a NUESTRO layout 110B/SB256
// [hmask32][qs64][scales12][d f16@108] con escalas reordenadas a 16×i8
// (kmask spill, mismo patrón que el case 6):
//
//   val(idx) = d · (s16[is]−32) · (q2 − (hmask_bit ? 0 : 4))
//   idx = nh·128 + col (nh 0..1, col 0..31) ⇒ por lane-iter: j=0 fijo,
//   is_lo = nh·8 (col 0..15), is_hi = nh·8+1 (col 16..31), hmask bit
//   uniforme = nh·4. Con u = A cuantizada q8_1 en smem (32 elems/lane·iter):
//
//   dot = sc_lo·dp4a(vi_lo16, u_lo16) + sc_hi·dp4a(vi_hi16, u_hi16)
//   vi = quanta−4·(1−hmask_bit) en BYTES (0..3 − 4|0 → −4..3, cabe en i8:
//   __vsubss4 satura) — exactamente el truco llama.cpp (~hmask >> shift,
//   0x03030303/0x04040404). sc = s16[is]−32 se aplica FUERA del dp4a por
//   mitades de 16 (las 2 escalas viven en bytes distintos del mismo u32).
//   A q8_1: quanta i8 s_aq + escala d8 f32 por sub-bloque de 32 (1 por
//   lane-iter) ⇒ dot_final = Σ_sb d8·(sc_lo·lo16 + sc_hi·hi16).
//
// Fase 1 cuantiza A a q8_1 en smem (8 warps reparten KB, amax por shfl —
// patrón §5.8). Fase 2: warp por fila, lane estride 32 SBs... no: cada
// lane-iter cubre 32 elems de un SB, 8 iters (t) cubren el SB completo
// (nh alterna por t: idx = lane + t·32 recorre 0..255 con nh = idx>>7).
extern "C" __global__ void q3kGemmM1Dp4aKernel(
    const float* __restrict__ a,
    const unsigned char* __restrict__ b,
    float* __restrict__ c,
    int K, int N, int s_param)
{
    extern __shared__ unsigned char smem_raw[];
    const int sb_total = K >> 8;          // SB de 256 elems
    const int kb_total = K >> 5;         // KB q8 de 32 elems
    // Layout smem: s_aq i8 [K pad16] | s_d8 f32 [KB]  (escala q8 por KB)
    int8_t* s_aq = (int8_t*)smem_raw;
    float* s_d8 = (float*)(smem_raw + ((K + 15) & ~15));

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    // ── Fase 1: A → q8_1 (quanta + escala por KB de 32). Patrón §5.8.
    for (int kb = warp; kb < kb_total; kb += (blockDim.x >> 5)) {
        const int base = kb * 32 + lane;
        const float v = a[base];
        float amax = fabsf(v);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d = (amax > 0.0f) ? amax / 127.0f : 1.0f;
        int q = (int)roundf(v / d);
        q = max(-127, min(127, q));
        s_aq[base] = (int8_t)q;
        if (lane == 0) s_d8[kb] = d;
    }
    __syncthreads();

    // ── Fase 2: GEMV — S warps por fila. s_param > 0: INTER-block (grid
    // (N*S+7)/8, fila=row/S — MÁS bloques, re-paga fase 1 ×S: NEGATIVO,
    // microbench 2026-09-10 0.74-0.92×). s_param < 0: INTRA-block (grid
    // (N·8/S+7)/8, los 8 warps del bloque cubren 8/S filas × S partes —
    // fase 1 se paga 1×/bloque, critical path por fila ÷S). S>1 ⇒ c se
    // pre-inicializa a 0 (memset async) y reduce con atomicAdd.
    const int S = (s_param < 0) ? -s_param : s_param;
    int rrow, part;
    if (S == 1) {
        rrow = blockIdx.x * 8 + warp;
        part = 0;
    } else if (s_param < 0) {
        rrow = blockIdx.x * (8 / S) + warp / S;   // intra (8%S==0: S∈{2,4})
        part = warp % S;
    } else {
        const int row = blockIdx.x * 8 + warp;    // inter
        rrow = row / S;
        part = row % S;
    }
    if (rrow >= N) return;
    const unsigned char* rowb = b + (size_t)rrow * ((size_t)sb_total * 110);

    // ── Cuerpo de UN SB (helper inline: también usado por el ILP-2 del
    // bucle principal — mismo código, 2 llamadas con acumuladores distintos).
    auto sbAccum = [&](int sb, float& acc_l) {
        const unsigned char* bp = rowb + (size_t)sb * 110;
        const float d3 = __half2float(*(const __half*)(bp + 108));
        const unsigned char* hm = bp;          // hmask 32B
        const unsigned char* qs = bp + 32;     // quanta 2-bit 64B
        const unsigned char* sc12 = bp + 96;   // escalas crudas 12B
        // A q8 de ESTE SB: elems globales sb·256..sb·256+255 (TRAMPA: s_aq
        // es índice GLOBAL — el bug original re-usaba los quanta del SB 0).
        const int8_t* sb_aq = s_aq + sb * 256;

        // Escalas reordenadas a 16×i8 (idéntico al case 6 del qgemmKernel).
        uint32_t aux[4] = {0, 0, 0, 0};
        aux[0] = (uint32_t)sc12[0] | ((uint32_t)sc12[1] << 8)
               | ((uint32_t)sc12[2] << 16) | ((uint32_t)sc12[3] << 24);
        aux[1] = (uint32_t)sc12[4] | ((uint32_t)sc12[5] << 8)
               | ((uint32_t)sc12[6] << 16) | ((uint32_t)sc12[7] << 24);
        aux[2] = (uint32_t)sc12[8] | ((uint32_t)sc12[9] << 8)
               | ((uint32_t)sc12[10] << 16) | ((uint32_t)sc12[11] << 24);
        const uint32_t tmp = aux[2];
        const uint32_t kmask1 = 0x03030303u;
        const uint32_t kmask2 = 0x0f0f0f0fu;
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const uint8_t* s16 = (const uint8_t*)aux;

        // ── Mapeo warp (verificado contra dequantQ3_K — AUTORIDAD):
        // elem idx = nh·128 + j·32 + c (c 0..31):
        //   quanta = byte qs[nh·32 + c] >> 2j    (contiguo: lane=chunk de
        //           4 bytes → 4 elems consecutivos, 1 u32, coalesced)
        //   hmask  = bit (nh·4+j) del byte hm[c]  (hmask[l+16] en la mitad
        //           alta del dequant ⇒ byte = c, completo 0..31)
        //   escala = s16[(nh·4+j)·2 + (c>>4)] — c = chunk·4+r con r 0..3
        //           ⇒ mitad de 16 = chunk>>2 (chunk 0..3 baja, 4..7 alta);
        //           los 4 elems del lane comparten UNA escala.
        // dp4a: vi = q2 − (bit?0:4) en BYTES vía vsubss4(qv, (~hm>>bit & 1)<<2)
        // — truco llama.cpp (satura −4..3, i8). 2 dp4a/lane·SB (nh 0|1).
        const int j = lane >> 3;
        const int chunk = lane & 7;
        const int is_lane = chunk >> 2;
        // ld32 misalineado (stride 110B no múltiplo de 4): funnel-shift
        // helper (patrón §5.8, misma trampa que 3.3 con 34B).
        const uint32_t hm_u32 = ld32_unaligned(hm + chunk * 4);
        const uint32_t qs0_u32 = ld32_unaligned(qs + chunk * 4);
        const uint32_t qs1_u32 = ld32_unaligned(qs + 32 + chunk * 4);

        // ── nh=0: elems (sb·256) + j·32 + chunk·4 + r
        // OJO: escala = (int8_t)s16[is] − 32 — el cast CON SIGNO (i8) es
        // el convenio dequantQ3_K (scales16 puede tener bytes ≥0x80).
        {
            const int sc = (int)(int8_t)s16[(0 * 4 + j) * 2 + is_lane] - 32;
            const uint32_t qv = (qs0_u32 >> (2 * j)) & 0x03030303u;
            const uint32_t vh = ((~hm_u32 >> (0 * 4 + j)) & 0x01010101u) << 2;
            const int vi = __vsubss4((int)qv, (int)vh);
            const int u = *(const int*)(sb_aq + j * 32 + chunk * 4);
            acc_l += d3 * s_d8[sb * 8 + j] * (float)sc * (float)__dp4a(vi, u, 0);
        }
        // ── nh=1: elems (sb·256) + 128 + j·32 + chunk·4 + r
        {
            const int sc = (int)(int8_t)s16[(1 * 4 + j) * 2 + is_lane] - 32;
            const uint32_t qv = (qs1_u32 >> (2 * j)) & 0x03030303u;
            const uint32_t vh = ((~hm_u32 >> (1 * 4 + j)) & 0x01010101u) << 2;
            const int vi = __vsubss4((int)qv, (int)vh);
            const int u = *(const int*)(sb_aq + 128 + j * 32 + chunk * 4);
            acc_l += d3 * s_d8[sb * 8 + 4 + j] * (float)sc * (float)__dp4a(vi, u, 0);
        }
    };

    // ILP-2 (Opción A del plan): 2 SBs en vuelo por warp — par e impar con
    // acumuladores separados para romper la cadena de dependencia de
    // `acc` (latencia FMA/LSU de la fase de memoria se solapa entre pares).
    // S=1: todos los SBs; S>1: el warp `part` cubre {part, part+S, ...}
    // en cada paridad (interleaved split-K, reduce via atomicAdd).
    float acc = 0.0f;
    float acc_b = 0.0f;
    int sb = part;
    for (; sb + S < sb_total; sb += 2 * S) {
        sbAccum(sb, acc);
        sbAccum(sb + S, acc_b);
    }
    for (; sb < sb_total; sb += S) sbAccum(sb, acc);
    acc += acc_b;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) {
        if (S == 1) c[rrow] = acc;
        else atomicAdd(c + rrow, acc);   // c pre-inicializada a 0 (memset async)
    }
}

// ─── a-U3 fase 3 (lane-a): GEMV q3_k M=1 sobre REPACK alineado 128B/SB ────
// Hipótesis (fase 2, 2026-09-10): el cuello del GEMV no es latencia/warps
// sino la AMPLIFICACIÓN DE SECTORES del stride 110B — ld32_unaligned hace
// byte-loads no coalesced (~2.5× sectores vs bytes útiles). Layout packed
// 128B/SB (idéntico contenido, re-ordenado en load):
//   [0..31]   hmask 32B          (igual)
//   [32..95]  quanta 2-bit 64B   (igual)
//   [96..111] escalas s16 16B PRE-DECODIFICADAS (kmask-spill hecho en pack)
//   [112..113] d f16             (offset par ⇒ ld u16 alineado)
//   [114..127] pad (SB múltiplo de 16 ⇒ uint4 loads)
// El kernel consume: cada SB son 8×uint4 (128B) — lanes 0..7 cargan el
// SB completo en 2 iteraciones coalesced; sin funnel-shift (escalas ya
// decodificadas). Ref numérica = MISMA que el 110B (dequantQ3_K es
// biyectivo en el repack — paridad bit a bit esperada vs q3kGemmM1Dp4a).
extern "C" __global__ void q3kGemmM1Dp4aPackedKernel(
    const float* __restrict__ a,
    const unsigned char* __restrict__ b,   // N filas × sb_total×128B
    float* __restrict__ c,
    int K, int N)
{
    extern __shared__ unsigned char smem_raw[];
    const int sb_total = K >> 8;
    const int kb_total = K >> 5;
    int8_t* s_aq = (int8_t*)smem_raw;
    float* s_d8 = (float*)(smem_raw + ((K + 15) & ~15));

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    // ── Fase 1: A → q8_1 (idéntica al kernel 110B — el repack es solo del
    // peso B; A no cambia).
    for (int kb = warp; kb < kb_total; kb += (blockDim.x >> 5)) {
        const int base = kb * 32 + lane;
        const float v = a[base];
        float amax = fabsf(v);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d = (amax > 0.0f) ? amax / 127.0f : 1.0f;
        int q = (int)roundf(v / d);
        q = max(-127, min(127, q));
        s_aq[base] = (int8_t)q;
        if (lane == 0) s_d8[kb] = d;
    }
    __syncthreads();

    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= N) return;
    const unsigned char* rowb = b + (size_t)row * ((size_t)sb_total * 128);

    float acc = 0.0f;
    for (int sb = 0; sb < sb_total; sb++) {
        const unsigned char* bp = rowb + (size_t)sb * 128;
        const float d3 = __half2float(*(const __half*)(bp + 112));
        const int j = lane >> 3;
        const int chunk = lane & 7;
        const int is_lane = chunk >> 2;
        // Loads coalesced: chunk 0..7 ⇒ 4 bytes contiguos cada 16B-lane —
        // con SB alineado a 16 TODOS son uint4/ld.32 alineados (el hw
        // fusiona el warp en sectores mínimos, sin amplificación).
        const uint32_t hm_u32 = *(const uint32_t*)(bp + chunk * 4);
        const uint32_t qs0_u32 = *(const uint32_t*)(bp + 32 + chunk * 4);
        const uint32_t qs1_u32 = *(const uint32_t*)(bp + 64 + chunk * 4);
        // Escalas pre-decodificadas: s16[(nh*4+j)*2 + is] en [96..111].
        // TRAMPA: (nh*4+j)*2 + is_lane con is_lane=1 es IMPAR — u32 misaligned
        // = error async. Leo el u16 PAR (offset 96+(nh*4+j)*2) y extraigo el
        // byte is_lane con shift (u16 alineado a 2 es legal).
        const uint16_t sc0_u16 = *(const uint16_t*)(bp + 96 + (0 * 4 + j) * 2);
        const uint16_t sc1_u16 = *(const uint16_t*)(bp + 96 + (1 * 4 + j) * 2);
        const int8_t* sb_aq = s_aq + sb * 256;

        // nh=0
        {
            const int sc = (int)(int8_t)((sc0_u16 >> (8 * is_lane)) & 0xFF) - 32;
            const uint32_t qv = (qs0_u32 >> (2 * j)) & 0x03030303u;
            const uint32_t vh = ((~hm_u32 >> (0 * 4 + j)) & 0x01010101u) << 2;
            const int vi = __vsubss4((int)qv, (int)vh);
            const int u = *(const int*)(sb_aq + j * 32 + chunk * 4);
            acc += d3 * s_d8[sb * 8 + j] * (float)sc * (float)__dp4a(vi, u, 0);
        }
        // nh=1
        {
            const int sc = (int)(int8_t)((sc1_u16 >> (8 * is_lane)) & 0xFF) - 32;
            const uint32_t qv = (qs1_u32 >> (2 * j)) & 0x03030303u;
            const uint32_t vh = ((~hm_u32 >> (1 * 4 + j)) & 0x01010101u) << 2;
            const int vi = __vsubss4((int)qv, (int)vh);
            const int u = *(const int*)(sb_aq + 128 + j * 32 + chunk * 4);
            acc += d3 * s_d8[sb * 8 + 4 + j] * (float)sc * (float)__dp4a(vi, u, 0);
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) c[row] = acc;
}

// ─── a-U3 fase 3c (lane-a): fusión QKV — UN launch para las 3 proyecciones ──
// El decode llama q/k/v con la MISMA A (x del token): 3 launches + 3 fase-1
// (A→q8 idéntica ×3) + 2 launches extra por capa. FUSIÓN: grid cubre
// n_q+n_k+n_v filas, fase-1 UNA vez, y la fila `row` elige peso/salida:
//   row < n_q        ⇒ peso bq, out cq, fila local row
//   row < n_q+n_k    ⇒ peso bk, out ck, fila local row-n_q
//   resto            ⇒ peso bv, out cv, fila local row-n_q-n_k
// Los 3 pesos DEBEN ser packed 128B/SB (q3kPackedWeight) con MISMO K y
// qtype q3_k. Los outs siguen siendo los buffers separados del attn
// (g_qg/g_k/g_v) — cero cambios en consumers.
extern "C" __global__ void q3kGemmM1Dp4aPackedQKVKernel(
    const float* __restrict__ a,
    const unsigned char* __restrict__ bq,   // packed [n_q × sb*128]
    const unsigned char* __restrict__ bk,   // packed [n_k × sb*128]
    const unsigned char* __restrict__ bv,   // packed [n_v × sb*128]
    float* __restrict__ cq,
    float* __restrict__ ck,
    float* __restrict__ cv,
    int K, int n_q, int n_k, int n_v)
{
    extern __shared__ unsigned char smem_raw[];
    const int sb_total = K >> 8;
    const int kb_total = K >> 5;
    int8_t* s_aq = (int8_t*)smem_raw;
    float* s_d8 = (float*)(smem_raw + ((K + 15) & ~15));

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    // ── Fase 1: A → q8_1 UNA VEZ (vs 3× en el camino separado).
    for (int kb = warp; kb < kb_total; kb += (blockDim.x >> 5)) {
        const int base = kb * 32 + lane;
        const float v = a[base];
        float amax = fabsf(v);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d = (amax > 0.0f) ? amax / 127.0f : 1.0f;
        int q = (int)roundf(v / d);
        q = max(-127, min(127, q));
        s_aq[base] = (int8_t)q;
        if (lane == 0) s_d8[kb] = d;
    }
    __syncthreads();

    // ── Fase 2: selección de (peso, out, fila local) por row.
    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    const int n_qk = n_q + n_k;
    const int N = n_qk + n_v;
    if (row >= N) return;
    const unsigned char* b;
    float* c;
    int lrow;
    if (row < n_q) {
        b = bq; c = cq; lrow = row;
    } else if (row < n_qk) {
        b = bk; c = ck; lrow = row - n_q;
    } else {
        b = bv; c = cv; lrow = row - n_qk;
    }
    const unsigned char* rowb = b + (size_t)lrow * ((size_t)sb_total * 128);

    float acc = 0.0f;
    for (int sb = 0; sb < sb_total; sb++) {
        const unsigned char* bp = rowb + (size_t)sb * 128;
        const float d3 = __half2float(*(const __half*)(bp + 112));
        const int j = lane >> 3;
        const int chunk = lane & 7;
        const int is_lane = chunk >> 2;
        const uint32_t hm_u32 = *(const uint32_t*)(bp + chunk * 4);
        const uint32_t qs0_u32 = *(const uint32_t*)(bp + 32 + chunk * 4);
        const uint32_t qs1_u32 = *(const uint32_t*)(bp + 64 + chunk * 4);
        const uint16_t sc0_u16 = *(const uint16_t*)(bp + 96 + (0 * 4 + j) * 2);
        const uint16_t sc1_u16 = *(const uint16_t*)(bp + 96 + (1 * 4 + j) * 2);
        const int8_t* sb_aq = s_aq + sb * 256;

        {
            const int sc = (int)(int8_t)((sc0_u16 >> (8 * is_lane)) & 0xFF) - 32;
            const uint32_t qv = (qs0_u32 >> (2 * j)) & 0x03030303u;
            const uint32_t vh = ((~hm_u32 >> (0 * 4 + j)) & 0x01010101u) << 2;
            const int vi = __vsubss4((int)qv, (int)vh);
            const int u = *(const int*)(sb_aq + j * 32 + chunk * 4);
            acc += d3 * s_d8[sb * 8 + j] * (float)sc * (float)__dp4a(vi, u, 0);
        }
        {
            const int sc = (int)(int8_t)((sc1_u16 >> (8 * is_lane)) & 0xFF) - 32;
            const uint32_t qv = (qs1_u32 >> (2 * j)) & 0x03030303u;
            const uint32_t vh = ((~hm_u32 >> (1 * 4 + j)) & 0x01010101u) << 2;
            const int vi = __vsubss4((int)qv, (int)vh);
            const int u = *(const int*)(sb_aq + 128 + j * 32 + chunk * 4);
            acc += d3 * s_d8[sb * 8 + 4 + j] * (float)sc * (float)__dp4a(vi, u, 0);
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) c[lrow] = acc;
}

// ─── Decodificación de escala/min (6 bits) para Q5_K (ref: get_scale_min_k4) ──
__device__ __forceinline__ int scaleMinD(int j, const unsigned char* sc) {
    if (j < 4) return sc[j] & 63;
    return (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
}
__device__ __forceinline__ int scaleMinM(int j, const unsigned char* sc) {
    if (j < 4) return sc[j + 4] & 63;
    return (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
}

// ─── GEMM cuantizado batched (M filas) ────────────────────────────────────────
// C[M,N] = A[M,K] * B^T, B = peso cuantizado GGUF [N,K] fila por fila.
// type: 0=q4_0, 1=q4_1, 2=q5_k, 3=q6_k. Un warp por (m, row); A[m] en smem.
// K múltiplo de 32 (q4_0/q4_1) o 256 (q5_k/q6_k).
extern "C" __global__ void qgemmKernel(
    const float* __restrict__ a,
    const unsigned char* __restrict__ b,
    float* __restrict__ c,
    int M, int K, int N, int type,
    unsigned int* ef) // lane-cuda UC-2.3: ErrorFlag (null = off)
{
    extern __shared__ float sa[];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int m = blockIdx.y;
    const float* am = a + (size_t)m * K;
    for (int i = tid; i < K; i += blockDim.x) sa[i] = am[i];
    __syncthreads();

    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= N) return;

    const int nblocks = K >> 5;
    const int nbig = K >> 8;
    const size_t rowstride =
        type == 0 ? (size_t)nblocks * 18 :
        type == 1 ? (size_t)nblocks * 20 :
        type == 2 ? (size_t)nbig * 176 :
        type == 4 ? (size_t)nbig * 144 :
        type == 5 ? (size_t)nblocks * 34 :
        type == 6 ? (size_t)nbig * 110 :
        type == 7 ? (size_t)nbig * 84 :
        type == 8 ? (size_t)nbig * 110 :
        type == 9 ? (size_t)nbig * 82 :
        type == 10 ? (size_t)nblocks * 18 :
        type == 11 ? (size_t)nblocks * 17 :
        type == 12 ? (size_t)nbig * 98 :
        type == 13 ? (size_t)nbig * 66 :
        type == 14 ? (size_t)nbig * 74 :
        type == 15 ? (size_t)nbig * 66 :
        type == 16 ? (size_t)nbig * 56 :
        type == 17 ? (size_t)nbig * 50 :
        type == 18 ? (size_t)(K >> 7) * 18 : // q1_0: 128 elems/block, 18B
        type == 19 ? (size_t)(K >> 6) * 18 :  // q2_0: 64 elems/block, 18B
                    (size_t)nbig * 210;
    const unsigned char* rowb = b + (size_t)row * rowstride;

    // lane-cuda UC-2.3 (piloto ErrorFlag): guards baratos (1 por fila,
    // thread lane==0), ABI-estables (ef==null = off). Detectan la clase
    // de bug que hangueaba kvarn-store en silencio: K no alineado con el
    // layout del tipo (sa OOB) y NaN del acumulado.
    // - nblocks/nbig NO cubren K residual: si K%32!=0 (o K%256!=0 en los
    //   tipos super-bloque) el loop lee/escibe sa fuera de [0,K).
    if (ef != nullptr && lane == 0) {
        const int superblock = (type == 2 || type == 4 || (type >= 6 && type <= 17) || type == 18 || type == 19);
        const int mod = superblock ? (type == 18 ? 128 : (type == 19 ? 64 : 256)) : 32;
        if (K % mod != 0) zaSetError(ef, ZA_EF_OOB);
    }

    float acc = 0.0f;
    switch (type) {
    case 0: { // q4_0: d = q*sc - 8
        for (int blk = 0; blk < nblocks; blk++) {
            const unsigned char* bp = rowb + (size_t)blk * 18;
            const float d = __half2float(*(const __half*)bp);
            const unsigned char q = bp[2 + (lane & 15)];
            const int nib = (lane < 16) ? (q & 0x0F) : (q >> 4);
            acc += sa[blk * 32 + lane] * (d * (float)(nib - 8));
        }
        break; }
    case 1: { // q4_1: val = d*q + m
        for (int blk = 0; blk < nblocks; blk++) {
            const unsigned char* bp = rowb + (size_t)blk * 20;
            const float d = __half2float(*(const __half*)bp);
            const float mm = __half2float(*(const __half*)(bp + 2));
            const unsigned char q = bp[4 + (lane & 15)];
            const int nib = (lane < 16) ? (q & 0x0F) : (q >> 4);
            acc += sa[blk * 32 + lane] * (d * (float)nib + mm);
        }
        break; }
    case 2: { // q5_k: super-bloque 256, 4 grupos de 64, escalas de 6 bits
        for (int blk = 0; blk < nbig; blk++) {
            const unsigned char* bp = rowb + (size_t)blk * 176;
            const float d = __half2float(*(const __half*)bp);
            const float mn = __half2float(*(const __half*)(bp + 2));
            const unsigned char* sc = bp + 4;
            const unsigned char* qh = bp + 16;
            const unsigned char* qs = bp + 48;
            const float* sa_base = sa + (size_t)blk * 256;
            for (int t = 0; t < 8; t++) {
                const int idx = lane + t * 32;
                const int g = idx >> 6;
                const int sub = idx & 63;
                const int is = g * 2;
                float d1, m1; int qv;
                if (sub < 32) {
                    const int l = sub;
                    const int bit = 1 << (2 * g);
                    d1 = d * (float)scaleMinD(is, sc);
                    m1 = mn * (float)scaleMinM(is, sc);
                    qv = (qs[g * 32 + l] & 0xF) + ((qh[l] & bit) ? 16 : 0);
                } else {
                    const int l = sub - 32;
                    const int bit = 2 << (2 * g);
                    d1 = d * (float)scaleMinD(is + 1, sc);
                    m1 = mn * (float)scaleMinM(is + 1, sc);
                    qv = (qs[g * 32 + l] >> 4) + ((qh[l] & bit) ? 16 : 0);
                }
                acc += sa_base[idx] * (d1 * (float)qv - m1);
            }
        }
        break; }
    case 9: { // iq2_s: 82B/SB256 [d f16][qs32][signs32][qh8][scales8].
              // Espejo val_iq2_s (kernel VERDE lane-a): idxg 10 bits con 2
              // bits altos de qh, signs byte libre por grupo, nibble escala
              // por mitad.
        for (int blk = 0; blk < nbig; ++blk) {
            const unsigned char* bp = rowb + (size_t)blk * 82;
            const float d = __half2float(*(const __half*)bp);
            const unsigned char* qs = bp + 2;
            const unsigned char* signs = bp + 34;
            const unsigned char* qh = bp + 66;
            const unsigned char* scales = bp + 74;
            const float* sa_base = sa + (size_t)blk * 256;
            for (int t = 0; t < 8; ++t) {
                const int in = lane + t * 32;
                const int ib = in >> 5;
                const int rem = in & 31;
                const int l = rem >> 3;
                const int j = rem & 7;
                const float db0 = d * (0.5f + (float)(scales[ib] & 0xF)) * 0.25f;
                const float db1 = d * (0.5f + (float)(scales[ib] >> 4)) * 0.25f;
                const float db = (l < 2) ? db0 : db1;
                const uint32_t idxg = (uint32_t)qs[ib * 4 + l]
                    | (((uint32_t)qh[ib] << (8 - 2 * l)) & 0x300u);
                const unsigned long long g = dev_iq2s_grid[idxg];
                const int sgn = (signs[ib * 4 + l] & (1u << j)) ? -1 : 1;
                const float gv = (float)((g >> (8 * j)) & 0xFF);
                acc += sa_base[in] * (db * gv * (float)sgn);
            }
        }
        break; }
    case 8: { // iq3_s: 110B/SB256 [d f16][qs64][qh8][signs32][scales4].
              // Espejo val_iq3_s (kernel VERDE lane-a): it=in/64,
              // half=(in%64)/32, l=(in%32)/8, col=in%8; db=d·(1+2·nibble);
              // grid via idx3s_a/b (bit hb global), signo bit col de sm.
              // ⚠️ MISMO layout que mi append IQ3_S (bit-exacto verificado).
        for (int blk = 0; blk < nbig; ++blk) {
            const unsigned char* bp = rowb + (size_t)blk * 110;
            const float d = __half2float(*(const __half*)bp);
            const unsigned char* qs = bp + 2;
            const unsigned char* qh = bp + 66;
            const unsigned char* signs = bp + 74;
            const unsigned char* scales = bp + 106;
            const float* sa_base = sa + (size_t)blk * 256;
            for (int t = 0; t < 8; ++t) {
                const int in = lane + t * 32;
                const int it = in >> 6;
                const int rem = in & 63;
                const int half = rem >> 5;
                const int l = (rem & 31) >> 3;
                const int col = rem & 7;
                const unsigned char sc = scales[it];
                const float db = d * (1.0f + 2.0f * (float)(
                    (half == 0) ? (sc & 0xF) : (sc >> 4)));
                const unsigned char* q = qs + it * 16 + (half != 0 ? 8 : 0);
                const unsigned char hb = qh[2 * it + half];
                const unsigned char sm = signs[it * 8 + half * 4 + l];
                const uint32_t idx = (col < 4)
                    ? (uint32_t)q[2 * l] | (((uint32_t)hb << (8 - 2 * l)) & 256u)
                    : (uint32_t)q[2 * l + 1] | (((uint32_t)hb << (7 - 2 * l)) & 256u);
                const uint32_t e = dev_iq3s_grid[idx];
                const int jx = (col < 4) ? col : col - 4;
                const int sgn = (sm & (1u << col)) ? -1 : 1;
                const float gv = (float)((e >> (8 * jx)) & 0xFF);
                acc += sa_base[in] * (db * gv * (float)sgn);
            }
        }
        break; }
    case 10: { // iq4_nl: 18B/bloque32 [d f16][qs16 split-16]; val=d·LUT[nib].
        const int8_t kv_nl[16] = { -127, -104, -83, -65, -49,
            -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };
        for (int blk = 0; blk < nblocks; ++blk) {
            const unsigned char* bp = rowb + (size_t)blk * 18;
            const float dd = __half2float(*(const __half*)bp);
            const unsigned char nib = bp[2 + (lane & 15)];
            const int qv = (lane < 16) ? (nib & 0xF) : (nib >> 4);
            acc += sa[blk * 32 + lane] * (dd * (float)kv_nl[qv]);
        }
        break; }
    case 11: { // mxfp4: 17B/bloque32 [E8M0 u8][qs16 split-16];
               // val=2^(e−127)·kvalues_fp4[nib].
        const int8_t kv_fp4[16] = { 0, 1, 2, 3, 4, 6, 8, 12,
            0, -1, -2, -3, -4, -6, -8, -12 };
        for (int blk = 0; blk < nblocks; ++blk) {
            const unsigned char* bp = rowb + (size_t)blk * 17;
            float dd = 1.0f;
            const unsigned char eb = bp[0];
            if (eb >= 127) { for (unsigned z = 127; z < eb; ++z) dd *= 2.0f; }
            else { for (unsigned z = eb; z < 127; ++z) dd *= 0.5f; }
            const unsigned char nib = bp[1 + (lane & 15)];
            const int qv = (lane < 16) ? (nib & 0xF) : (nib >> 4);
            acc += sa[blk * 32 + lane] * (dd * (float)kv_fp4[qv]);
        }
        break; }
    case 12: { // iq3_xxs: 98B/SB256 [d f16][qs64][ss32]. Sub-bloque ib:
               // aux u32 LE = ss[ib*4]: sc bits[28,32), signos-grupo l en
               // bits[7l,7l+7) vía ksigns; grupo l: cols 0-3 ← grid[qs[ib*8+
               // 2l]], 4-7 ← grid[qs[ib*8+2l+1]]; gv=byte jx; sg bit col sm.
        for (int blk = 0; blk < nbig; ++blk) {
            const unsigned char* bp = rowb + (size_t)blk * 98;
            const float dd = __half2float(*(const __half*)bp);
            const unsigned char* qs = bp + 2;
            const unsigned char* ssc = bp + 66;
            const float* sa_base = sa + (size_t)blk * 256;
            for (int t = 0; t < 8; ++t) {
                const int in = lane + t * 32;
                const int ib = in >> 5;
                const int pos = in & 31;
                const int l = pos >> 3;
                const int sub = pos & 7;
                const uint32_t aux = (uint32_t)ssc[ib * 4] |
                    ((uint32_t)ssc[ib * 4 + 1] << 8) |
                    ((uint32_t)ssc[ib * 4 + 2] << 16) |
                    ((uint32_t)ssc[ib * 4 + 3] << 24);
                const float db = dd * (0.5f + (float)(aux >> 28)) * 0.5f;
                const uint8_t signs = dev_ksigns_iq2xs[(aux >> (7 * l)) & 127];
                const int jx = (sub < 4) ? sub : sub - 4;
                const unsigned char qb = (sub < 4) ? qs[ib * 8 + 2 * l]
                                                   : qs[ib * 8 + 2 * l + 1];
                const uint32_t gx = dev_iq3xxs_grid[qb];
                // Máscara de signo = BIT DIRECTO col (kmask_iq2xs[col]);
                // ⚠️ no confundir con la tabla ksigns (valores de byte).
                const int sg = (signs & (int)(1u << sub)) ? -1 : 1;
                const float gv = (float)((gx >> (8 * jx)) & 0xFF);
                acc += sa_base[in] * (db * gv * (float)sg);
            }
        }
        break; }
    case 13: { // iq2_xxs: 66B/SB256 [d f16][qs64]. Sub-bloque ib: aux0 u32
               // LE qs[ib*8..+4]=4 índices grid por grupo l; aux1 u32 LE
               // qs[ib*8+4..+8]: sc bits[28,32), signos-grupo 7 bits/l;
               // db=d·(0.5+sc)·0.25; g=iq2xxs_grid[(aux0>>(8l))&FF];
               // gv=byte j; sg bit j de ksigns-byte.
        for (int blk = 0; blk < nbig; ++blk) {
            const unsigned char* bp = rowb + (size_t)blk * 66;
            const float dd = __half2float(*(const __half*)bp);
            const unsigned char* qs = bp + 2;
            const float* sa_base = sa + (size_t)blk * 256;
            for (int t = 0; t < 8; ++t) {
                const int in = lane + t * 32;
                const int ib = in >> 5;
                const int pos = in & 31;
                const int l = pos >> 3;
                const int j = pos & 7;
                const uint32_t aux0 = (uint32_t)qs[ib * 8] |
                    ((uint32_t)qs[ib * 8 + 1] << 8) |
                    ((uint32_t)qs[ib * 8 + 2] << 16) |
                    ((uint32_t)qs[ib * 8 + 3] << 24);
                const uint32_t aux1 = (uint32_t)qs[ib * 8 + 4] |
                    ((uint32_t)qs[ib * 8 + 5] << 8) |
                    ((uint32_t)qs[ib * 8 + 6] << 16) |
                    ((uint32_t)qs[ib * 8 + 7] << 24);
                const float db = dd * (0.5f + (float)(aux1 >> 28)) * 0.25f;
                const int idxg = (int)((aux0 >> (8 * l)) & 0xFF);
                const uint8_t signs = dev_ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
                const unsigned long long g = dev_iq2xxs_grid[idxg];
                // Máscara de signo = BIT DIRECTO j (kmask), no tabla ksigns.
                const int sg = (signs & (int)(1u << j)) ? -1 : 1;
                const float gv = (float)((g >> (8 * j)) & 0xFF);
                acc += sa_base[in] * (db * gv * (float)sg);
            }
        }
        break; }
    case 14: { // iq2_xs: 74B/SB256 [d f16][qs64][scales8]. Grupo l: v u16 LE
               // qs[ib*8+2l..]: bits[0,9)=índice iq2xs_grid, [9,16)=signos
               // ksigns; db por nibble de scales[ib] (low→grupos 0-1).
        for (int blk = 0; blk < nbig; ++blk) {
            const unsigned char* bp = rowb + (size_t)blk * 74;
            const float dd = __half2float(*(const __half*)bp);
            const unsigned char* qs = bp + 2;
            const unsigned char* scales = bp + 66;
            const float* sa_base = sa + (size_t)blk * 256;
            for (int t = 0; t < 8; ++t) {
                const int in = lane + t * 32;
                const int ib = in >> 5;
                const int rem = in & 31;
                const int l = rem >> 3;
                const int j = rem & 7;
                const float db0 = dd * (0.5f + (float)(scales[ib] & 0xF)) * 0.25f;
                const float db1 = dd * (0.5f + (float)(scales[ib] >> 4)) * 0.25f;
                const float db = (l < 2) ? db0 : db1;
                const uint16_t v = (uint16_t)qs[ib * 8 + l * 2] |
                    ((uint16_t)qs[ib * 8 + l * 2 + 1] << 8);
                const uint8_t signs = dev_ksigns_iq2xs[v >> 9];
                const unsigned long long g = dev_iq2xs_grid[v & 511];
                const int sg = (signs & (int)(1u << j)) ? -1 : 1;
                const float gv = (float)((g >> (8 * j)) & 0xFF);
                acc += sa_base[in] * (db * gv * (float)sg);
            }
        }
        break; }

    case 15: { // tq2_0: 66B/SB256 [qs 64B][d f16@64]. val=d(q−1).
        for (int blk = 0; blk < nbig; ++blk) {
            const unsigned char* bp = rowb + (size_t)blk * 66;
            const float dd = __half2float(*(const __half*)(bp + 64));
            const float* sa_base = sa + (size_t)blk * 256;
            for (int t = 0; t < 8; ++t) {
                const int in = lane + t * 32;
                const int seg = in >> 7;
                const int rem = in & 127;
                const int l = rem >> 5;
                const int m = rem & 31;
                const int q = (bp[seg * 32 + m] >> (2 * l)) & 3;
                acc += sa_base[in] * dd * ((float)q - 1.0f);
            }
        }
        break; }

    case 16: { // iq1_m: 56B/SB256 layout entrelazado KV-path. SIN campo d
               // propio: f16 reensamblado de nibbles ALTOS de bytes impares
               // [1,3,5,7]; sc16 del par p=ib>>1 en bytes [2p,2p+2) con dl
               // codes 3b×4 en bits [0,12) — SOLAPADOS con qb[0..8); qh@32:
               // 3 bits altos idx (l par <<8 / impar <<4) + bit dd ±0.125.
               // Espejo val_iq1_m (kernel VERDE lane-a) == mi append IQ1_M.
        for (int blk = 0; blk < nbig; ++blk) {
            const unsigned char* bp = rowb + (size_t)blk * 56;
            const unsigned short sc0 = (unsigned short)bp[0] | ((unsigned short)bp[1] << 8);
            const unsigned short sc1 = (unsigned short)bp[2] | ((unsigned short)bp[3] << 8);
            const unsigned short sc2 = (unsigned short)bp[4] | ((unsigned short)bp[5] << 8);
            const unsigned short sc3 = (unsigned short)bp[6] | ((unsigned short)bp[7] << 8);
            const float dd_scale = __half2float(__ushort_as_half((unsigned short)(
                (sc0 >> 12) | ((sc1 >> 8) & 0xF0) |
                ((sc2 >> 4) & 0xF00) | (sc3 & 0xF000))));
            const float* sa_base = sa + (size_t)blk * 256;
            for (int t = 0; t < 8; ++t) {
                const int in = lane + t * 32;
                const int ib = in >> 5;
                const int rem = in & 31;
                const int l = rem >> 3;
                const int j = rem & 7;
                const int sc_off = (ib >> 1) * 2;
                const unsigned short sc16 =
                    (unsigned short)bp[sc_off] | ((unsigned short)bp[sc_off + 1] << 8);
                const float dl1 = dd_scale * (2.0f * (float)((sc16 >> (6 * (ib & 1))) & 7) + 1.0f);
                const float dl2 = dd_scale * (2.0f * (float)((sc16 >> (6 * (ib & 1) + 3)) & 7) + 1.0f);
                const unsigned char qb = bp[ib * 4 + l]; // ⚠️ qb en base[0..32)
                const unsigned char qhb = bp[32 + sc_off + (l >> 1)];
                const int idxg = qb | ((((int)qhb << ((l & 1) ? 4 : 8))) & 0x700);
                const float ddt = (qhb & ((l & 1) ? 0x80 : 0x08)) ? -0.125f : 0.125f;
                const unsigned long long g = dev_iq1s_grid[idxg];
                const float gv = (float)(int)((signed char)((g >> (8 * j)) & 0xFFULL));
                acc += sa_base[in] * ((l < 2 ? dl1 : dl2) * (gv + ddt));
            }
        }
        break; }

    case 17: { // iq1_s (B-a4, lane-a): 50B/SB256 [d f16@0][qs 4x8B@2 (1B
               // por par l=0..3, byte índice grid LOW)][qh 16B@34 (u16 por
               // ib: 3 bits idx HIGH en 12+3*l, signo dd en bit 15)].
               // Espejo EXACTO de val_iq1_s (fused_decode_extra.cu:437,
               // kernels KV VERDES): mismo grid dev_iq1s_grid, misma
               // aritmética f32, mismo orden de acumulación ⇒ bit-paridad.
        for (int blk = 0; blk < nbig; ++blk) {
            const unsigned char* bp = rowb + (size_t)blk * 50;
            const float d = __half2float(__ushort_as_half((unsigned short)(
                (unsigned short)bp[0] | ((unsigned short)bp[1] << 8))));
            const float* sa_base = sa + (size_t)blk * 256;
            for (int t = 0; t < 8; ++t) {
                const int in = lane + t * 32;
                const int ib = in >> 5;
                const int rem = in & 31;
                const int l = rem >> 3;
                const int j = rem & 7;
                const unsigned char* qs = bp + 2;
                const unsigned char* qh = bp + 34;
                const unsigned short qhb =
                    (unsigned short)qh[ib * 2] | ((unsigned short)qh[ib * 2 + 1] << 8);
                const float dl = d * (2.0f * (float)((qhb >> 12) & 7) + 1.0f);
                const float dd = (qhb & 0x8000) ? -0.125f : 0.125f;
                const int idxg = qs[ib * 4 + l] | ((((int)qhb >> (3 * l)) & 7) << 8);
                const unsigned long long g = dev_iq1s_grid[idxg];
                const float gv = (float)(int)((signed char)((g >> (8 * j)) & 0xFFULL));
                acc += sa_base[in] * (dl * (gv + dd));
            }
        }
        break; }

    case 18: { // q1_0: 1-bit sign, QK=128, 18B/block [d f16][qs 16B].
               // val = sign ? d : -d
        for (int blk = 0; blk < (K >> 7); ++blk) {
            const unsigned char* bp = rowb + (size_t)blk * 18;
            const float d = __half2float(*(const __half*)bp);
            const unsigned char* qs = bp + 2;
            const int base = blk * 128;
            for (int j = 0; j < 128; ++j) {
                const int idx = base + j;
                if (idx >= K) break;
                const int byte_idx = j >> 3;
                const int bit_idx = j & 7;
                const int bit = (qs[byte_idx] >> bit_idx) & 1;
                const float val = bit ? d : -d;
                acc += sa[idx] * val;
            }
        }
        break; }

    case 19: { // q2_0: 2-bit codes, QK=64, 18B/block [d f16][qs 16B].
               // q ∈ {0,1,2,3} → val = (q - 1) * d => {-1, 0, +1, +2}
        for (int blk = 0; blk < (K >> 6); ++blk) {
            const unsigned char* bp = rowb + (size_t)blk * 18;
            const float d = __half2float(*(const __half*)bp);
            const unsigned char* qs = bp + 2;
            const int base = blk * 64;
            for (int j = 0; j < 64; ++j) {
                const int idx = base + j;
                if (idx >= K) break;
                const int byte_idx = j >> 2;
                const int bit_shift = (j & 3) * 2;
                const int q = (qs[byte_idx] >> bit_shift) & 3;
                const float val = (float)(q - 1) * d;
                acc += sa[idx] * val;
            }
        }
        break; }

    case 6: { // q3_k: 110B/SB256 [hmask32][qs64][scales12][d f16@108].
              // Escalas reordenadas a 16 i8 (kmask spill, espejo
              // dequantQ3_K); val = d·(sc−32)·(q2 − (hmask_bit?0:4)).
        for (int blk = 0; blk < nbig; blk++) {
            const unsigned char* bp = rowb + (size_t)blk * 110;
            const float d = __half2float(*(const __half*)(bp + 108));
            const unsigned char* hm = bp;
            const unsigned char* qs = bp + 32;
            const unsigned char* sc12 = bp + 96;
            uint32_t aux[4] = {0, 0, 0, 0};
            aux[0] = (uint32_t)sc12[0] | ((uint32_t)sc12[1] << 8)
                   | ((uint32_t)sc12[2] << 16) | ((uint32_t)sc12[3] << 24);
            aux[1] = (uint32_t)sc12[4] | ((uint32_t)sc12[5] << 8)
                   | ((uint32_t)sc12[6] << 16) | ((uint32_t)sc12[7] << 24);
            aux[2] = (uint32_t)sc12[8] | ((uint32_t)sc12[9] << 8)
                   | ((uint32_t)sc12[10] << 16) | ((uint32_t)sc12[11] << 24);
            const uint32_t tmp = aux[2];
            const uint32_t kmask1 = 0x03030303u;
            const uint32_t kmask2 = 0x0f0f0f0fu;
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            const uint8_t* s16 = (const uint8_t*)aux;
            const float* sa_base = sa + (size_t)blk * 256;
            for (int t = 0; t < 8; t++) {
                const int idx = lane + t * 32;
                const int nh = idx >> 7;
                const int rem = idx & 127;
                const int j = rem >> 5;
                const int col = rem & 31;
                const int shift = 2 * j;
                const int is = (nh * 4 + j) * 2 + (col >> 4);
                const int qv = (int)((qs[nh * 32 + col] >> shift) & 3);
                // OJO: el bit de hmask NO se reinicia por mitad de 128 —
                // m acumula globalmente (dequantQ3_K: m<<=1 fuera del while
                // de mitades) ⇒ bit = nh*4+j.
                const int hv = (hm[col] & (1u << (nh * 4 + j))) ? 0 : 4;
                acc += sa_base[idx]
                     * (d * (float)((int8_t)s16[is] - 32)) * (float)(qv - hv);
            }
        }
        break; }
    case 7: { // q2_k: 84B/SB256 [scales16][qs64][d f16@80][min f16@82].
              // Sub-bloques de 16: nibble low=d-scale/high=min-scale,
              // quanta 2-bit con shift 2j; espejo dequantQ2_K.
        for (int blk = 0; blk < nbig; blk++) {
            const unsigned char* bp = rowb + (size_t)blk * 84;
            const float d = __half2float(*(const __half*)(bp + 80));
            const float mn = __half2float(*(const __half*)(bp + 82));
            const unsigned char* sc = bp;
            const unsigned char* qs = bp + 16;
            const float* sa_base = sa + (size_t)blk * 256;
            for (int t = 0; t < 8; t++) {
                const int idx = lane + t * 32;
                const int nh = idx >> 7;
                const int rem = idx & 127;
                const int j = rem >> 5;
                const int col = rem & 31;
                const int shift = 2 * j;
                const int is = (nh * 4 + j) * 2 + (col >> 4);
                const float dl = d * (float)(sc[is] & 0xF);
                const float ml = mn * (float)(sc[is] >> 4);
                const int qv = (int)((qs[nh * 32 + col] >> shift) & 3);
                acc += sa_base[idx] * (dl * (float)qv - ml);
            }
        }
        break; }
    case 5: { // q8_0: 34B/bloque32 [d f16][i8×32] — lane cubre los 32 elems.
        for (int blk = 0; blk < nblocks; blk++) {
            const unsigned char* bp = rowb + (size_t)blk * 34;
            const float d = __half2float(*(const __half*)bp);
            acc += sa[blk * 32 + lane]
                 * (d * (float)(int8_t)bp[2 + lane]);
        }
        break; }
    case 4: { // q4_k: 144B/SB256 [d f16][dmin f16][scales[12]][qs[128]];
              // packing 6-bit con spill (get_scale_min_k4) — espejo exacto de
              // dequantQ4_K/getScaleMinK4Canon (kv_quant.zig, bit-verde).
              // Escalas precomputadas UNA vez por sub-bloque (uniforme en el
              // warp); lane cubre los 32 elems del sub-bloque con acceso
              // coalescido (antes: unpack ramificado POR ELEMENTO ⇒ 40% más
              // lento que q6_k moviendo MENOS bytes).
        for (int blk = 0; blk < nbig; blk++) {
            const unsigned char* bp = rowb + (size_t)blk * 144;
            const float d = __half2float(*(const __half*)bp);
            const float mn = __half2float(*(const __half*)(bp + 2));
            const unsigned char* sc = bp + 4;
            const unsigned char* qs = bp + 16;
            const float* sa_base = sa + (size_t)blk * 256;
            for (int is = 0; is < 8; ++is) {
                const int c64 = is >> 1;
                const float dv = d * (float)((is < 4) ? (sc[is] & 63)
                    : ((sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4)));
                const float mv = mn * (float)((is < 4) ? (sc[is + 4] & 63)
                    : ((sc[is + 4] >> 4) | ((sc[is] >> 6) << 4)));
                if ((is & 1) == 0) {
                    acc += sa_base[is * 32 + lane]
                         * (dv * (float)(qs[c64 * 32 + lane] & 0xF) - mv);
                } else {
                    acc += sa_base[is * 32 + lane]
                         * (dv * (float)(qs[c64 * 32 + lane] >> 4) - mv);
                }
            }
        }
        break; }
    case 3: { // q6_k: ql/qh + escalas i8, val = d*sc*(q - 32)
        for (int blk = 0; blk < nbig; blk++) {
            const unsigned char* bp = rowb + (size_t)blk * 210;
            const float d = __half2float(*(const __half*)(bp + 208));
            const unsigned char* ql = bp;
            const unsigned char* qh = bp + 128;
            const unsigned char* sc = bp + 192;
            const float* sa_base = sa + (size_t)blk * 256;
            for (int t = 0; t < 8; t++) {
                const int idx = lane + t * 32;
                const int n2 = idx >> 7;
                const int sub = idx & 127;
                const int l = sub & 31;
                const int part = sub >> 5;
                const int is = l >> 4;
                const unsigned char qlb = ql[n2 * 64 + l + ((part & 1) ? 32 : 0)];
                const unsigned char qv0 = (part < 2) ? (qlb & 0x0F) : (qlb >> 4);
                const unsigned char qv = qv0 | ((qh[n2 * 32 + l] >> (2 * part)) & 3) << 4;
                const float s = (float)((signed char)sc[n2 * 8 + is + 2 * part]);
                acc += sa_base[idx] * (d * s * (float)((int)qv - 32));
            }
        }
        break; }
    }
    for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    // lane-cuda UC-2.3: NaN/Inf del resultado (post-reduce, pre-store) —
    // graba el PRIMER error y sigue (no trappea). 1 instr de predicado.
    if (ef != nullptr && lane == 0) (void)zaCheckFloat(acc, ef);
    if (lane == 0) c[(size_t)m * N + row] = acc;
}

// ─── Argmax (G2 / TODO 1.7): argmax por fila, warp-shuffle f32→i32 ─────────
// Port de `argmax.cu:8-40` del fork unsloth (ggml/src/ggml-cuda/argmax.cu).
// Hoy el decode copia TODO el vector de logits a host (vocab·4B ≈ 993 KB,
// ~257 µs) y hace el argmax en CPU (~84 µs). Con este kernel el argmax
// ocurre en device y el D2H es de 4 bytes (un i32) en modo greedy.
//   ~340 µs/token ⇒ +6% decode, y es prerequisito del acceptance-en-device
//   de spec decoding.
//
// Semántica de empates: gana la PRIMERA aparición del máximo (mismo criterio
// que Sampler.greedyArgmax de pipeline.zig) — la reducción usa `>` estricto
// para comparar, así que un valor igual posterior NO desplaza al primero.
// Layout: x [nrows, ncols] contiguo; dst [nrows] i32.
// Lanzamiento: grid=(nrows), block=min(1024, round_up(ncols, WARP)).
// Desempate: ¿debe (val,col) desplazar a (maxval,argmax)? Valor
// estrictamente mayor, o empate EXACTO con índice MENOR.
//
// Por qué hace falta el desempate por índice: el barrido usa stride
// (el hilo t procesa t, t+B, t+2B…), así que el orden de descubrimiento
// NO es monotónico en el índice de columna. Sin desempate, un máximo
// repetido podría resolverse a una posición posterior y divergir del
// argmax host (que devuelve la PRIMERA aparición = índice mínimo entre
// los que empatan). Con esto el kernel es bit-exacto con
// Sampler.greedyArgmax (pipeline.zig).
__device__ __forceinline__ bool argmaxWins(float val, int col, float maxval, int argmax) {
    if (val > maxval) return true;
    if (val == maxval && (argmax < 0 || col < argmax)) return true;
    return false;
}

extern "C" __global__ void argmaxF32Kernel(
    const float* __restrict__ x,
    int32_t* __restrict__ dst,
    int ncols)
{
    const int row = blockIdx.x;
    const float* rowx = x + (size_t)row * ncols;

    float maxval = -FLT_MAX;
    int   argmax = -1;

    // Barrido con stride: cada hilo salta blockDim.x (ncols >> blockDim.x
    // en vocabularios reales: 128256 vs 1024 hilos). Dentro del hilo las
    // columnas son crecientes ⇒ `>` ya deja el índice menor.
    for (int col = threadIdx.x; col < ncols; col += blockDim.x) {
        const float val = rowx[col];
        if (argmaxWins(val, col, maxval, argmax)) {
            maxval = val;
            argmax = col;
        }
    }

    // Reducción intra-warp (xor-shuffle: todos los lanes acaban con el
    // máximo del warp; no necesita sincronización).
    #pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xffffffffu, maxval, offset, WARP);
        const int   col = __shfl_xor_sync(0xffffffffu, argmax, offset, WARP);
        if (argmaxWins(val, col, maxval, argmax)) {
            maxval = val;
            argmax = col;
        }
    }

    const int n_warps = blockDim.x / WARP;
    const int lane_id = threadIdx.x % WARP;
    const int warp_id = threadIdx.x / WARP;

    if (n_warps > 1) {
        constexpr int max_warps = 1024 / WARP;
        __shared__ float shared_maxval[max_warps];
        __shared__ int   shared_argmax[max_warps];
        if (lane_id == 0) {
            shared_maxval[warp_id] = maxval;
            shared_argmax[warp_id] = argmax;
        }
        __syncthreads();

        // El warp 0 reduce los n_warps parciales. Los lanes sobrantes
        // (lane_id >= n_warps) aportan -FLT_MAX/-1 para no contaminar
        // el desempate por índice.
        if (warp_id == 0) {
            if (lane_id < n_warps) {
                maxval = shared_maxval[lane_id];
                argmax = shared_argmax[lane_id];
            } else {
                maxval = -FLT_MAX;
                argmax = -1;
            }
            #pragma unroll
            for (int offset = WARP / 2; offset > 0; offset >>= 1) {
                const float val = __shfl_xor_sync(0xffffffffu, maxval, offset, WARP);
                const int   col = __shfl_xor_sync(0xffffffffu, argmax, offset, WARP);
                if (argmaxWins(val, col, maxval, argmax)) {
                    maxval = val;
                    argmax = col;
                }
            }
        }
    }

    if (warp_id == 0 && lane_id == 0) {
        dst[row] = argmax;
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// MMQ Q4_0 GEMM — tensor-core WMMA INT8 for prefill M>=128
//
// Stage 1: caller provides pre-quantized A as q8_0 [M*K] + scales + ssums
// MMQ Q4_0 batched GEMM — tensor cores WMMA INT8 (16x16x16), lane-f §5.10.
//
// C[M,N] = A[M,K]_q8 · B_q4_0[N,K]^T  (pesos GGUF [out,in] fila-major)
// A: q8 simétrico por bloques K de 32 → aq i8 [M,K], ad half [M,KB];
//    asa se IGNORA (B se dequanta a simétrico -8..7 → sin corrección).
// Por paso kb: contrib(r,n) = dA(r,kb)·dB(n,kb)·Σ_{k∈kb} aq·wq. Como
// dA/dB varían por paso, el fragmento se escala tras cada mma mediante
// store_matrix_sync a SHARED (colectivo del warp — jamás a memoria
// privada per-thread, semántica inválida). El mapeo lane→posición del
// fragmento NO se usa: cada lane acumula bajo su propio mapeo
// p = lane + 32*e del tile 16×16; el store row-major a shared es la
// única pieza hardware-definida.
//
// Grid: ((N+BN-1)/BN, (M+BM-1)/BM)  Block: 8 warps = 256 threads
// Warp map: warp>>2 (2 filas) × warp&3 (4 cols) → cada warp es dueño
// EXCLUSIVO de un tile 64×32 de C (sin solape, sin atomics, sin memset).
// ══════════════════════════════════════════════════════════════════════════════
#define MMQ_GEMM_BM 128
#define MMQ_GEMM_BN 128
#define MMQ_GEMM_BK 32
#define MMQ_GEMM_NWARPS 8
#define MMQ_GEMM_THREADS (MMQ_GEMM_NWARPS * 32)
#define MMQ_GEMM_BK_PAD (MMQ_GEMM_BK + 16)

__device__ __forceinline__ void mmq_dequant_q4_0(
    const uint8_t* blk, int8_t out[32], float* scale_out)
{
    *scale_out = __half2float(*(const __half*)blk);
    uint32_t u[4];
    {
        const uint32_t* vp = (const uint32_t*)((uintptr_t)(blk + 2) & ~(uintptr_t)3);
        const uint32_t sh = (uint32_t)(((uintptr_t)(blk + 2) & 3) * 8);
        const uint32_t r0 = vp[0], r1 = vp[1], r2 = vp[2], r3 = vp[3], r4 = vp[4];
        u[0] = __funnelshift_r(r0, r1, sh);
        u[1] = __funnelshift_r(r1, r2, sh);
        u[2] = __funnelshift_r(r2, r3, sh);
        u[3] = __funnelshift_r(r3, r4, sh);
    }
    const uint32_t lo_m = 0x0F0F0F0Fu;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int8_t* lp = (const int8_t*)(&u[i]);
        const int blo = i * 4, bhi = 16 + i * 4;
        out[blo + 0] = (lp[0] & 0x0F) - 8;
        out[blo + 1] = (lp[1] & 0x0F) - 8;
        out[blo + 2] = (lp[2] & 0x0F) - 8;
        out[blo + 3] = (lp[3] & 0x0F) - 8;
        out[bhi + 0] = ((lp[0] >> 4) & 0x0F) - 8;
        out[bhi + 1] = ((lp[1] >> 4) & 0x0F) - 8;
        out[bhi + 2] = ((lp[2] >> 4) & 0x0F) - 8;
        out[bhi + 3] = ((lp[3] >> 4) & 0x0F) - 8;
    }
}

extern "C" __global__ __launch_bounds__(MMQ_GEMM_THREADS)
void mmqQ4_0GEMMKernel(
    const int8_t* __restrict__ aq,     // [M*K] q8 simétrico
    const __half* __restrict__ ad,    // [M*KB] escala por (fila, bloque K)
    const int* __restrict__ asa,      // [M*KB] (no usado aquí)
    const uint8_t* __restrict__ w,    // [N][KB*18] q4_0 crudo GGUF
    float* __restrict__ c,            // [M*N] salida (escritura única)
    int M, int K, int N)
{
    const int kb_total = K >> 5;
    const int bm = blockIdx.y * MMQ_GEMM_BM;
    const int bn = blockIdx.x * MMQ_GEMM_BN;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    extern __shared__ unsigned char smem_raw[];
    // Bytes: s_aq[128*48] | s_ad[128 f32] | s_bwr[128*32 i8] | s_dB[128 f32]
    //        | s_cW[8 warps × 256 i32] (staging exclusivo por warp)
    int8_t* s_aq  = (int8_t*)smem_raw;                                // 6144
    float*  s_ad  = (float*)(s_aq + MMQ_GEMM_BM * MMQ_GEMM_BK_PAD);   //  512
    int8_t* s_bwr = (int8_t*)(s_ad + MMQ_GEMM_BM);                    // 4096
    float*  s_dB  = (float*)(s_bwr + MMQ_GEMM_BN * 32);               //  512
    int*    s_cW  = (int*)(s_dB + MMQ_GEMM_BN);                       // 8192

    const int aq_ld = MMQ_GEMM_BK_PAD;

    // Tile exclusivo del warp: 64 filas × 32 cols del bloque 128×128
    const int warp_row_off = (warp >> 2) * 64;   // offset fila dentro del bloque
    const int warp_col_off = (warp & 3) * 32;    // offset col  dentro del bloque
    const int warp_rows = bm + warp_row_off;
    const int warp_cols = bn + warp_col_off;
    int* s_c = s_cW + warp * 256;               // staging EXCLUSIVO de este warp

    // 8 tiles (4 ri × 2 ci) × 8 posiciones/lane = 64 acumuladores f32
    float acc[8][8] = {};

    for (int kb = 0; kb < kb_total; ++kb) {
        // ── Stage 1: peso [BN×32] q4_0 → s_bwr i8 simétrico + s_dB ──
        // (filas ≥ N: s_dB=0 anula su contribución aunque s_bwr sea basura)
        for (int i = tid; i < MMQ_GEMM_BN; i += blockDim.x) {
            const int nrow = bn + i;
            if (nrow >= N) { s_dB[i] = 0.0f; continue; }
            const uint8_t* blk = w + (size_t)nrow * (size_t)kb_total * 18
                                  + (size_t)kb * 18;
            float wscale;
            int8_t wq[32];
            mmq_dequant_q4_0(blk, wq, &wscale);
            s_dB[i] = wscale;
            const int bwr_off = i * 32;
            #pragma unroll
            for (int j = 0; j < 32; ++j)
                s_bwr[bwr_off + j] = wq[j];
        }

        // ── Stage 2a: dA del paso (por fila M) → s_ad (0 si fila ≥ M) ──
        for (int i = tid; i < MMQ_GEMM_BM; i += blockDim.x) {
            const int gi = bm + i;
            s_ad[i] = (gi < M) ? __half2float(ad[gi * kb_total + kb]) : 0.0f;
        }

        // ── Stage 2b: A tile [BM×32] → s_aq (fila stride 48; 0 si fila ≥ M) ──
        for (int flat = tid; flat < MMQ_GEMM_BM * MMQ_GEMM_BK; flat += blockDim.x) {
            const int gi = bm + flat / MMQ_GEMM_BK;
            const int koff = kb * 32 + (flat % MMQ_GEMM_BK);
            const int dst = (flat / MMQ_GEMM_BK) * aq_ld + (flat % MMQ_GEMM_BK);
            s_aq[dst] = (gi < M) ? aq[gi * K + koff] : (int8_t)0;
        }
        __syncthreads();

        // ── Stage 3: WMMA + escala por paso ──
        #pragma unroll
        for (int ri = 0; ri < 4; ++ri) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                                   int8_t, nvcuda::wmma::row_major> a0, a1;
            // Filas del warp dentro del bloque: warp_row_off + ri*16 + [0..16)
            const int aoff = (warp_row_off + ri * WMMA_M) * aq_ld;
            nvcuda::wmma::load_matrix_sync(a0, s_aq + aoff, aq_ld);
            nvcuda::wmma::load_matrix_sync(a1, s_aq + aoff + 16, aq_ld);
            #pragma unroll
            for (int ci = 0; ci < 2; ++ci) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                                       int8_t, nvcuda::wmma::col_major> b0, b1;
                // col_major: (k,n) en ptr[n*ldm+k]; s_bwr fila=n (32 k) ⇒ ldm=32.
                // Columnas del warp: warp_col_off + ci*16 + [0..16)
                const int boff = (warp_col_off + ci * WMMA_N) * 32;
                nvcuda::wmma::load_matrix_sync(b0, s_bwr + boff, 32);
                nvcuda::wmma::load_matrix_sync(b1, s_bwr + boff + 16, 32);

                nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> x;
                nvcuda::wmma::fill_fragment(x, 0);
                nvcuda::wmma::mma_sync(x, a0, b0, x);   // K[0..16)
                nvcuda::wmma::mma_sync(x, a1, b1, x);   // K[16..32)

                // Staging colectivo del warp → shared, luego escala con el
                // mapeo PROPIO p = lane + 32*e (r=p/16, cc=p%16).
                __syncwarp();
                nvcuda::wmma::store_matrix_sync(s_c, x, WMMA_N, nvcuda::wmma::mem_row_major);
                __syncwarp();
                const int ti = ri * 2 + ci;
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int p = lane + 32 * e;         // 0..255
                    const int r = p >> 4, cc = p & 15;
                    const float da = s_ad[warp_row_off + ri * WMMA_M + r];
                    const float db = s_dB[warp_col_off + ci * WMMA_N + cc];
                    acc[ti][e] += (float)s_c[p] * da * db;
                }
                __syncwarp();
            }
        }
        __syncthreads();
    }

    // ── Epílogo: cada posición de C escrita EXACTAMENTE una vez ──
    (void)asa;
    #pragma unroll
    for (int ri = 0; ri < 4; ++ri) {
        #pragma unroll
        for (int ci = 0; ci < 2; ++ci) {
            const int ti = ri * 2 + ci;
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int p = lane + 32 * e;
                const int r = p >> 4, cc = p & 15;
                const int grow = warp_rows + ri * WMMA_M + r;
                const int gcol = warp_cols + ci * WMMA_N + cc;
                if (grow < M && gcol < N)
                    c[(size_t)grow * N + gcol] = acc[ti][e];
            }
        }
    }
}

// ─── C6.1 DFlash encoder (5.1, lane-c): gather de taps device-resident ───────
// El encoder dflash consume features del target: concat de los hidden de
// las target_layers (dflash.target_layers GGUF — p.ej. {2,6,10,14,18,22,
// 26,30} del 9B ⇒ [8·n_embd] = 32768). Los hidden YA viven en device en
// el decode path (g.g_norm / g.g_mixer del target); este kernel los
// concatena SIN D2H — el requisito central del TODO 5.1.
//   taps_ptrs: array device de n_taps punteros device (cada uno → n_embd f32)
//   out: [n_taps · n_embd] f32 (fc_input del encoder)
// Grid: (n_embd/256, n_taps) — coalesced por fila; 1 launch por ronda spec.
extern "C" __global__ void dflashTapsGatherKernel(
    const float* __restrict__ const* taps_ptrs,
    float* __restrict__ out,
    int n_embd, int n_taps)
{
    const int tap = blockIdx.y;
    if (tap >= n_taps) return;
    const float* src = taps_ptrs[tap];
    float* dst = out + (size_t)tap * n_embd;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_embd;
         i += gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}

// ─── C6.1 DFlash encoder (5.1, lane-c): GEMV q8_0 fc con split-K por tiles ──
// El fc del sidecar es [n_embd, n_taps·n_embd] con K=32768 (9B: 8 taps×4096)
// — el M1 genérico (smem k·4B = 128KB) excede el límite sm_86 (99KB).
// Este kernel procesa K en tiles de DFLASH_FC_TILE (8192 => 32KB smem),
// acumulando en registros; smem reutilizada entre tiles, un solo launch.
// Layout q8_0 GGUF: 34B/bloque32 = [d f16][i8*32] — espejo del case 5 del
// qgemmKernel (lane = 1 elem del bloque).
#define DFLASH_FC_TILE 8192

extern "C" __global__ void dflashFcGemmM1Kernel(
    const float* __restrict__ a,          // [K] (fc_in: taps concat)
    const unsigned char* __restrict__ b,  // q8_0 [N][K] GGUF row-major
    float* __restrict__ out,              // [N]
    int K, int N)
{
    const int warp = threadIdx.x >> 5;
    const int n_warps = blockDim.x >> 5;
    const int row = blockIdx.x * n_warps + warp;
    if (row >= N) return;
    const int lane = threadIdx.x & 31;
    const int nblocks_total = K >> 5;                 // 32 elems/bloque
    const int nblocks_tile = DFLASH_FC_TILE >> 5;      // 256 bloques/tile
    const size_t rowstride = (size_t)nblocks_total * 34;
    const unsigned char* brow = b + (size_t)row * rowstride;

    __shared__ float a_tile[DFLASH_FC_TILE];

    float acc = 0.0f;
    for (int t0 = 0; t0 < K; t0 += DFLASH_FC_TILE) {
        for (int i = threadIdx.x; i < DFLASH_FC_TILE && t0 + i < K; i += blockDim.x) {
            a_tile[i] = a[t0 + i];
        }
        __syncthreads();
        // CADA lane itera TODOS los bloques del tile leyendo SU elem (case-5
        // pattern del qgemmKernel) — el round-robin blk+=32 que puse primero
        // dejaba cada bloque con UN solo producto (lane==blk): rel 71.
        float wacc = 0.0f;
        const int blk0 = (t0 >> 5);
        for (int blk = 0; blk < nblocks_tile; ++blk) {
            const unsigned char* qb = brow + (size_t)(blk0 + blk) * 34;
            const float d = __half2float(*(const __half*)qb);
            wacc += d * (float)(int8_t)qb[2 + lane]
                  * a_tile[blk * 32 + lane];
        }
        #pragma unroll
        for (int s = 16; s > 0; s >>= 1)
            wacc += __shfl_down_sync(0xffffffffu, wacc, s);
        if (lane == 0) acc += wacc;
        __syncthreads();
    }
    if (lane == 0) out[row] = acc;
}

// ─── 1.15 path-A (lane-a): sampler GPU Gumbel-trick ────────────────────────
// temp>0 + rep_penalty EN DEVICE, graph-capturable: elimina el softmax 128k
// CPU (~1.3ms/tok) + D2H de logits completos (501KB/tok) del camino default
// temp=1.0 (gap medido 2.3 t/s en 3B Q3_K_S).
//
// Gumbel-max trick (Adam & Yellin): token = argmax(logits/temp + g_i) con
// g_i = -ln(-ln(u_i)) u.i.d. ⇒ sampleo EXACTO de softmax(logits/temp) sin
// sort ni cumsum. Coste ≈ argmax (una reducción + exp/log por elemento).
// Philox 4x32-10 (Random123, Salmon et al.) — counter-based, 0 estado
// mutante en host: el counter vive en device (uint4) y se incrementa al
// final del kernel (determinista, replay-safe: cada replay consume streams
// nuevos). Verificación contra curand: counter/key=0 ⇒ x0=0x6627e8d5 (los
// words x1..x3 difieren del doc por orden de bump — variante igualmente
// válida como PRNG; el gate del test 1.15 es la UNIFORMIDAD estadística,
// no el valor exacto). Mi variante previa 2x32-7 casera colapsaba (no
// cruzaba pares — p=1.0 medido en el test).

__device__ __forceinline__ uint4 philox4x32round(uint4 ctr, uint2 key) {
    // Random123 EXACTO: M0 multiplica a ctr.X, M1 a ctr.Z (con los Y/W
    // como multiplicandos el estado degenera cuando y=w=0 — bug medido).
    const uint32_t M0 = 0xD2511F53u;
    const uint32_t M1 = 0xCD9E8D57u;
    const uint32_t hi0 = __umulhi(M0, ctr.x);
    const uint32_t lo0 = M0 * ctr.x;
    const uint32_t hi1 = __umulhi(M1, ctr.z);
    const uint32_t lo1 = M1 * ctr.z;
    return make_uint4(
        hi1 ^ ctr.y ^ key.x,
        lo1,
        hi0 ^ ctr.w ^ key.y,
        lo0);
}

__device__ __forceinline__ uint4 philox4x32_10(uint4 c, uint2 k) {
    #pragma unroll
    for (int r = 0; r < 10; ++r) {
        if (r > 0) {
            k.x += 0x9E3779B9u;
            k.y += 0xBB67AE85u;
        }
        c = philox4x32round(c, k);
    }
    return c;
}

__device__ __forceinline__ float gumbelFromUint(uint32_t u) {
    // u → (0,1) exclusivo (evitar log(0)); g = -ln(-ln(p)).
    const float p = (u & 0x00FFFFFFu) * (1.0f / 16777216.0f) + (1.0f / 33554432.0f);
    return -__logf(-__logf(p));
}

// Muestreo Gumbel: 1 bloque, grid (rows). Cada thread computa score de sus
// columnas (stride blockDim) con SU philox stream derivado del counter
// global + columna (determinista por posición, no por hilo ⇒ replay-stable
// en orden de ejecución). rep_penalty: scatter de los últimos ring_n tokens
// (ring circular en device) ANTES del score — búsqueda lineal por columna
// (ring_n ≤ 64 típico: coste ≤64 comparaciones por elemento, barato vs
// exp/log). Reduce final = argmaxWins (desempate índice menor).
// ring_n vive EN DEVICE (ring[0]): el graph captura el PUNTERO, el host
// actualiza el tamaño por replay (ring[0] = len, ring[1..] = tokens) — un
// kernel-param int quedaría congelado en la captura.
extern "C" __global__ void sampleF32GumbelKernel(
    const float* __restrict__ logits,
    int32_t* __restrict__ dst,
    const uint4* __restrict__ philox_counter,   // [1] estado (pre-alocado)
    const uint32_t* __restrict__ ring,         // [0]=n, [1..n]=tokens (device)
    float temp,
    float penalty,
    int ncols)
{
    const int row = blockIdx.x;
    const float* rowx = logits + (size_t)row * ncols;
    const uint4 ctr = *philox_counter;
    const int ring_n = (ring != nullptr) ? (int)ring[0] : 0;

    float maxval = -FLT_MAX;
    int   argmax = -1;
    const float inv_t = 1.0f / temp;

    for (int col = threadIdx.x; col < ncols; col += blockDim.x) {
        // RNG determinista por (counter, col): stream por columna.
        uint4 c = make_uint4(ctr.x, ctr.y, ctr.z + (uint32_t)col, ctr.w);
        const uint4 r = philox4x32_10(c, make_uint2(0x0B7E1516u, 0x152FBC0Bu));
        const float g = gumbelFromUint(r.x);
        float v = rowx[col];
        // rep_penalty (llama.cpp convenio): pos/neg div/mult. ring[0]=n,
        // tokens en ring[1..n].
        if (penalty != 1.0f && ring_n > 0) {
            for (int i = 1; i <= ring_n; ++i) {
                if ((int)ring[i] == col) {
                    if (v > 0.0f) v /= penalty;
                    else if (v < 0.0f) v *= penalty;
                    break;
                }
            }
        }
        const float score = v * inv_t + g;
        if (argmaxWins(score, col, maxval, argmax)) {
            maxval = score;
            argmax = col;
        }
    }

    // Reducción intra-warp + inter-warp (idéntica al argmaxF32Kernel).
    #pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xffffffffu, maxval, offset, WARP);
        const int   col = __shfl_xor_sync(0xffffffffu, argmax, offset, WARP);
        if (argmaxWins(val, col, maxval, argmax)) {
            maxval = val;
            argmax = col;
        }
    }
    const int n_warps = blockDim.x / WARP;
    const int lane_id = threadIdx.x % WARP;
    const int warp_id = threadIdx.x / WARP;
    if (n_warps > 1) {
        constexpr int max_warps = 1024 / WARP;
        __shared__ float shared_maxval[max_warps];
        __shared__ int   shared_argmax[max_warps];
        if (lane_id == 0) {
            shared_maxval[warp_id] = maxval;
            shared_argmax[warp_id] = argmax;
        }
        __syncthreads();
        if (warp_id == 0) {
            if (lane_id < n_warps) {
                maxval = shared_maxval[lane_id];
                argmax = shared_argmax[lane_id];
            } else {
                maxval = -FLT_MAX;
                argmax = -1;
            }
            #pragma unroll
            for (int offset = WARP / 2; offset > 0; offset >>= 1) {
                const float val = __shfl_xor_sync(0xffffffffu, maxval, offset, WARP);
                const int   col = __shfl_xor_sync(0xffffffffu, argmax, offset, WARP);
                if (argmaxWins(val, col, maxval, argmax)) {
                    maxval = val;
                    argmax = col;
                }
            }
            if (lane_id == 0) dst[row] = argmax;
        }
    } else {
        if (lane_id == 0) dst[row] = argmax;
    }

    // Avance del counter SOLO una vez por fila (warp 0, lane 0): la llamada
    // N+1 usa streams distintos (ctr.x+1). Replay-safe: dentro del graph
    // el incremento persiste en device entre replays.
    if (threadIdx.x == 0) {
        // atomicAdd sobre x del counter (uint4 como int*): +1 por sample.
        atomicAdd((int*)philox_counter, 1);
    }
}

// ─── RLT: Recurrent Looped Transformer gated merge ───────────────────────────
// u_t = e_t + α * σ(W_g [e_t; RMSNorm(s_{t-1})]) ⊙ W_s RMSNorm(s_{t-1})
// Single-row kernel: one block processes one row (d elements).
// Grid: (1,), Block: (min(d, 256),)
//
// Params:
//   encoder_rep  [d]     — e_t (input, read-only)
//   prev_state   [d]     — s_{t-1} (input, read-only)
//   w_gate       [d, 2d] — gate projection [2d, d] row-major
//   w_gate_bias  [d]     — gate bias
//   w_state      [d, d]  — state projection [d, d] row-major
//   out          [d]     — output u_t
//   d            int     — embedding dimension
//   alpha        float   — feedback scale
// ─── dev-IQ P0-5/P0-6: dp4a M1 kernels ────────────────────────────────────────

// ─── dev-IQ P0-5 (2026-09-14): dp4a para iq3_s M=1 (case 8) ──────────────────
// Layout 110B/SB256 [d f16][qs64][qh8][signs32][scales4].
extern "C" __global__ void iq3sGemmM1Dp4aKernel(
    const float* __restrict__ a,
    const unsigned char* __restrict__ b,
    float* __restrict__ c,
    int K, int N)
{
    extern __shared__ unsigned char smem_raw[];
    const int sb_total = K >> 8;
    const int kb_total = K >> 5;
    int8_t* s_aq = (int8_t*)smem_raw;
    float* s_d8 = (float*)(smem_raw + ((K + 15) & ~15));

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    for (int kb = warp; kb < kb_total; kb += (blockDim.x >> 5)) {
        const int base = kb * 32 + lane;
        const float v = a[base];
        float amax = fabsf(v);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d = (amax > 0.0f) ? amax / 127.0f : 1.0f;
        int q = (int)roundf(v / d);
        q = max(-127, min(127, q));
        s_aq[base] = (int8_t)q;
        if (lane == 0) s_d8[kb] = d;
    }
    __syncthreads();

    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= N) return;
    const unsigned char* rowb = b + (size_t)row * ((size_t)sb_total * 110);

    float acc = 0.0f;
    for (int sb = 0; sb < sb_total; sb++) {
        const unsigned char* bp = rowb + (size_t)sb * 110;
        const float d = __half2float(*(const __half*)bp);
        const unsigned char* qs = bp + 2;
        const unsigned char* qh = bp + 66;
        const unsigned char* signs = bp + 74;
        const unsigned char* scales = bp + 106;
        const int8_t* sb_aq = s_aq + sb * 256;

        for (int t = 0; t < 8; t++) {
            const int in = lane + t * 32;
            const int it = in >> 6;
            const int rem = in & 63;
            const int half = rem >> 5;
            const int l = (rem & 31) >> 3;
            const int col = rem & 7;
            const unsigned char sc = scales[it];
            const float db = d * (1.0f + 2.0f * (float)(
                (half == 0) ? (sc & 0xF) : (sc >> 4)));
            const unsigned char* q = qs + it * 16 + (half != 0 ? 8 : 0);
            const unsigned char hb = qh[2 * it + half];
            const unsigned char sm = signs[it * 8 + half * 4 + l];
            const uint32_t idx = (col < 4)
                ? (uint32_t)q[2 * l] | (((uint32_t)hb << (8 - 2 * l)) & 256u)
                : (uint32_t)q[2 * l + 1] | (((uint32_t)hb << (7 - 2 * l)) & 256u);
            const uint32_t e = dev_iq3s_grid[idx];
            const int jx = (col < 4) ? col : col - 4;
            const int sgn = (sm & (int)(1u << col)) ? -1 : 1;
            const int gv_signed = (int)((e >> (8 * jx)) & 0xFF) * sgn;
            const int u = *(const int*)(sb_aq + in);
            acc += db * (float)__dp4a(gv_signed, u, 0);
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) c[row] = acc;
}

// ─── dev-IQ P0-5 (2026-09-14): dp4a para iq2_s M=1 (case 9) ──────────────────
// Layout 82B/SB256 [d f16][qs32][signs32][qh8][scales8].
extern "C" __global__ void iq2sGemmM1Dp4aKernel(
    const float* __restrict__ a,
    const unsigned char* __restrict__ b,
    float* __restrict__ c,
    int K, int N)
{
    extern __shared__ unsigned char smem_raw[];
    const int sb_total = K >> 8;
    const int kb_total = K >> 5;
    int8_t* s_aq = (int8_t*)smem_raw;
    float* s_d8 = (float*)(smem_raw + ((K + 15) & ~15));

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    for (int kb = warp; kb < kb_total; kb += (blockDim.x >> 5)) {
        const int base = kb * 32 + lane;
        const float v = a[base];
        float amax = fabsf(v);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d = (amax > 0.0f) ? amax / 127.0f : 1.0f;
        int q = (int)roundf(v / d);
        q = max(-127, min(127, q));
        s_aq[base] = (int8_t)q;
        if (lane == 0) s_d8[kb] = d;
    }
    __syncthreads();

    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= N) return;
    const unsigned char* rowb = b + (size_t)row * ((size_t)sb_total * 82);

    float acc = 0.0f;
    for (int sb = 0; sb < sb_total; sb++) {
        const unsigned char* bp = rowb + (size_t)sb * 82;
        const float d = __half2float(*(const __half*)bp);
        const unsigned char* qs = bp + 2;
        const unsigned char* signs = bp + 34;
        const unsigned char* qh = bp + 66;
        const unsigned char* scales = bp + 74;
        const int8_t* sb_aq = s_aq + sb * 256;

        for (int t = 0; t < 8; t++) {
            const int in = lane + t * 32;
            const int ib = in >> 5;
            const int rem = in & 31;
            const int l = rem >> 3;
            const int j = rem & 7;
            const float db0 = d * (0.5f + (float)(scales[ib] & 0xF)) * 0.25f;
            const float db1 = d * (0.5f + (float)(scales[ib] >> 4)) * 0.25f;
            const float db = (l < 2) ? db0 : db1;
            const uint32_t idxg = (uint32_t)qs[ib * 4 + l]
                | (((uint32_t)qh[ib] << (8 - 2 * l)) & 0x300u);
            const unsigned long long g = dev_iq2s_grid[idxg];
            const int sgn = (signs[ib * 4 + l] & (1u << j)) ? -1 : 1;
            const int gv_signed = (int)((g >> (8 * j)) & 0xFF) * sgn;
            const int u = *(const int*)(sb_aq + in);
            acc += db * (float)__dp4a(gv_signed, u, 0);
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) c[row] = acc;
}

// ─── dev-IQ P0-5 (2026-09-14): dp4a para iq4_xs M=1 (case 18) ────────────────
// Layout 136B/SB256 [d f16][scales_h u16@2][scales_l[4]@4][qs[128]@8].
extern "C" __global__ void iq4xsGemmM1Dp4aKernel(
    const float* __restrict__ a,
    const unsigned char* __restrict__ b,
    float* __restrict__ c,
    int K, int N)
{
    extern __shared__ unsigned char smem_raw[];
    const int sb_total = K >> 8;
    const int kb_total = K >> 5;
    int8_t* s_aq = (int8_t*)smem_raw;
    float* s_d8 = (float*)(smem_raw + ((K + 15) & ~15));

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    for (int kb = warp; kb < kb_total; kb += (blockDim.x >> 5)) {
        const int base = kb * 32 + lane;
        const float v = a[base];
        float amax = fabsf(v);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d = (amax > 0.0f) ? amax / 127.0f : 1.0f;
        int q = (int)roundf(v / d);
        q = max(-127, min(127, q));
        s_aq[base] = (int8_t)q;
        if (lane == 0) s_d8[kb] = d;
    }
    __syncthreads();

    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= N) return;
    const unsigned char* rowb = b + (size_t)row * ((size_t)sb_total * 136);

    const int8_t kv_nl[16] = { -127, -104, -83, -65, -49,
        -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

    float acc = 0.0f;
    for (int sb = 0; sb < sb_total; sb++) {
        const unsigned char* bp = rowb + (size_t)sb * 136;
        const float d = __half2float(*(const __half*)bp);
        const unsigned short scales_h = *(const unsigned short*)(bp + 2);
        const unsigned char* scales_l = bp + 4;
        const unsigned char* qs = bp + 8;
        const int8_t* sb_aq = s_aq + sb * 256;

        for (int t = 0; t < 8; t++) {
            const int in = lane + t * 32;
            const int ib = in >> 5;
            const int rem = in & 31;
            const int j = rem & 15;
            const int ls = ((scales_l[ib >> 1] >> ((ib & 1) << 2)) & 0xF)
                         | (((scales_h >> (ib << 1)) & 3) << 4);
            const float dl = d * (float)(ls - 32);
            const int qb_off = ib * 16 + j;
            const unsigned char qb = qs[qb_off];
            const int q = (rem < 16) ? (qb & 0xF) : (qb >> 4);
            const int lut_val = (int)kv_nl[q];
            const int u = *(const int*)(sb_aq + in);
            acc += dl * (float)__dp4a(lut_val, u, 0);
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) c[row] = acc;
}

// ─── dev-IQ P0-6 (2026-09-14): dp4a para q4_1 M=1 (case 1) ──────────────────
// Layout 20B/SB256 [d f16@0][mm f16@2][qs64@4].
extern "C" __global__ void q41GemmM1Dp4aKernel(
    const float* __restrict__ a,
    const unsigned char* __restrict__ b,
    float* __restrict__ c,
    int K, int N)
{
    extern __shared__ unsigned char smem_raw[];
    const int kb_total = K >> 5;
    int8_t* s_aq = (int8_t*)smem_raw;
    float* s_d8 = (float*)(smem_raw + ((K + 15) & ~15));
    float* s_sa = s_d8 + kb_total;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    for (int kb = warp; kb < kb_total; kb += (blockDim.x >> 5)) {
        const int base = kb * 32 + lane;
        const float v = a[base];
        float amax = fabsf(v);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d = (amax > 0.0f) ? amax / 127.0f : 1.0f;
        int q = (int)roundf(v / d);
        q = max(-127, min(127, q));
        s_aq[base] = (int8_t)q;
        int sq = q;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if (lane == 0) {
            s_d8[kb] = d;
            s_sa[kb] = (float)sq;
        }
    }
    __syncthreads();

    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= N) return;
    const unsigned char* rowb = b + (size_t)row * ((size_t)kb_total * 20);

    float acc = 0.0f;
    for (int kb = lane; kb < kb_total; kb += 32) {
        const unsigned char* blk = rowb + (size_t)kb * 20;
        const float d = __half2float(*(const __half*)blk);
        const float mm = __half2float(*(const __half*)(blk + 2));
        const unsigned char* qs = blk + 4;
        const int8_t* abase = s_aq + kb * 32;
        uint32_t u[4];
        {
            const uint32_t* vp = (const uint32_t*)((uintptr_t)qs & ~(uintptr_t)3);
            const uint32_t sh = (uint32_t)(((uintptr_t)qs & 3) * 8);
            const uint32_t r0 = vp[0], r1 = vp[1], r2 = vp[2], r3 = vp[3], r4 = vp[4];
            u[0] = __funnelshift_r(r0, r1, sh);
            u[1] = __funnelshift_r(r1, r2, sh);
            u[2] = __funnelshift_r(r2, r3, sh);
            u[3] = __funnelshift_r(r3, r4, sh);
        }
        const int a0 = *(const int*)(abase);
        const int a1 = *(const int*)(abase + 4);
        const int a2 = *(const int*)(abase + 8);
        const int a3 = *(const int*)(abase + 12);
        const int a4 = *(const int*)(abase + 16);
        const int a5 = *(const int*)(abase + 20);
        const int a6 = *(const int*)(abase + 24);
        const int a7 = *(const int*)(abase + 28);
        const uint32_t lo_m = 0x0F0F0F0Fu;
        int sn = 0;
        sn = __dp4a((int)(u[0] & lo_m), a0, sn);
        sn = __dp4a((int)(u[1] & lo_m), a1, sn);
        sn = __dp4a((int)(u[2] & lo_m), a2, sn);
        sn = __dp4a((int)(u[3] & lo_m), a3, sn);
        sn = __dp4a((int)((u[0] >> 4) & lo_m), a4, sn);
        sn = __dp4a((int)((u[1] >> 4) & lo_m), a5, sn);
        sn = __dp4a((int)((u[2] >> 4) & lo_m), a6, sn);
        sn = __dp4a((int)((u[3] >> 4) & lo_m), a7, sn);
        acc += s_d8[kb] * (d * (float)sn + mm * s_sa[kb]);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) c[row] = acc;
}

// ─── dev-IQ P0-6 (2026-09-14): dp4a para q2_k M=1 (case 7) ──────────────────
// Layout 84B/SB256 [scales16][qs64][d f16@80][min f16@82].
extern "C" __global__ void q2kGemmM1Dp4aKernel(
    const float* __restrict__ a,
    const unsigned char* __restrict__ b,
    float* __restrict__ c,
    int K, int N)
{
    extern __shared__ unsigned char smem_raw[];
    const int sb_total = K >> 8;
    const int kb_total = K >> 5;
    int8_t* s_aq = (int8_t*)smem_raw;
    float* s_d8 = (float*)(smem_raw + ((K + 15) & ~15));

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    for (int kb = warp; kb < kb_total; kb += (blockDim.x >> 5)) {
        const int base = kb * 32 + lane;
        const float v = a[base];
        float amax = fabsf(v);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d = (amax > 0.0f) ? amax / 127.0f : 1.0f;
        int q = (int)roundf(v / d);
        q = max(-127, min(127, q));
        s_aq[base] = (int8_t)q;
        if (lane == 0) s_d8[kb] = d;
    }
    __syncthreads();

    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= N) return;
    const unsigned char* rowb = b + (size_t)row * ((size_t)sb_total * 84);

    float acc = 0.0f;
    for (int sb = 0; sb < sb_total; sb++) {
        const unsigned char* bp = rowb + (size_t)sb * 84;
        const float d = __half2float(*(const __half*)(bp + 80));
        const float mn = __half2float(*(const __half*)(bp + 82));
        const unsigned char* sc = bp;
        const unsigned char* qh = bp + 16;
        const unsigned char* qs = bp + 32;
        const int8_t* sb_aq = s_aq + sb * 256;

        for (int t = 0; t < 8; t++) {
            const int in = lane + t * 32;
            const int g = in >> 6;
            const int sub = in & 63;
            const int is = g * 2;
            const float d1 = d * (float)scaleMinD(is, sc);
            const float m1 = mn * (float)scaleMinM(is, sc);
            const int bit = (sub < 32) ? (1 << (2 * g)) : (2 << (2 * g));
            const int l = sub >> 3;
            const int qv_base = (sub < 32)
                ? (qs[g * 32 + l] & 0xF)
                : (qs[g * 32 + l] >> 4);
            const int qv = qv_base + ((qh[l] & bit) ? 16 : 0);
            const int u = *(const int*)(sb_aq + in);
            acc += s_d8[sb * 8 + (in >> 5)] * ((float)d1 * (float)__dp4a(qv, u, 0) - m1 * (float)u);
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) c[row] = acc;
}

// ─── dev-IQ P0-6 (2026-09-14): dp4a para iq4_nl M=1 (case 10) ───────────────
// Layout 18B/SB32 [d f16@0][qs16@2]. LUT kvalues_iq4nl.
extern "C" __global__ void iq4nlGemmM1Dp4aKernel(
    const float* __restrict__ a,
    const unsigned char* __restrict__ b,
    float* __restrict__ c,
    int K, int N)
{
    extern __shared__ unsigned char smem_raw[];
    const int kb_total = K >> 5;
    int8_t* s_aq = (int8_t*)smem_raw;
    float* s_d8 = (float*)(smem_raw + ((K + 15) & ~15));

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    for (int kb = warp; kb < kb_total; kb += (blockDim.x >> 5)) {
        const int base = kb * 32 + lane;
        const float v = a[base];
        float amax = fabsf(v);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d = (amax > 0.0f) ? amax / 127.0f : 1.0f;
        int q = (int)roundf(v / d);
        q = max(-127, min(127, q));
        s_aq[base] = (int8_t)q;
        if (lane == 0) s_d8[kb] = d;
    }
    __syncthreads();

    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= N) return;
    const unsigned char* rowb = b + (size_t)row * ((size_t)kb_total * 18);

    const int8_t kv_nl[16] = { -127, -104, -83, -65, -49,
        -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

    float acc = 0.0f;
    for (int kb = lane; kb < kb_total; kb += 32) {
        const unsigned char* blk = rowb + (size_t)kb * 18;
        const float d = __half2float(*(const __half*)blk);
        const unsigned char* qs = blk + 2;
        const int8_t* abase = s_aq + kb * 32;
        int sn = 0;
        for (int i = 0; i < 32; i++) {
            const unsigned char byte = qs[i >> 1];
            const int nib = (i & 1) ? (byte >> 4) : (byte & 0xF);
            const int lut_val = (int)kv_nl[nib];
            sn += lut_val * (int)abase[i];
        }
        acc += d * s_d8[kb] * (float)sn;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) c[row] = acc;
}

// ─── dev-IQ P0-6 (2026-09-14): dp4a para iq3_xxs M=1 (case 12) ──────────────
// Layout 98B/SB256 [d f16][qs64][ss32].
extern "C" __global__ void iq3xxsGemmM1Dp4aKernel(
    const float* __restrict__ a,
    const unsigned char* __restrict__ b,
    float* __restrict__ c,
    int K, int N)
{
    extern __shared__ unsigned char smem_raw[];
    const int sb_total = K >> 8;
    const int kb_total = K >> 5;
    int8_t* s_aq = (int8_t*)smem_raw;
    float* s_d8 = (float*)(smem_raw + ((K + 15) & ~15));

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    for (int kb = warp; kb < kb_total; kb += (blockDim.x >> 5)) {
        const int base = kb * 32 + lane;
        const float v = a[base];
        float amax = fabsf(v);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d = (amax > 0.0f) ? amax / 127.0f : 1.0f;
        int q = (int)roundf(v / d);
        q = max(-127, min(127, q));
        s_aq[base] = (int8_t)q;
        if (lane == 0) s_d8[kb] = d;
    }
    __syncthreads();

    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= N) return;
    const unsigned char* rowb = b + (size_t)row * ((size_t)sb_total * 98);

    float acc = 0.0f;
    for (int sb = 0; sb < sb_total; sb++) {
        const unsigned char* bp = rowb + (size_t)sb * 98;
        const float dd = __half2float(*(const __half*)bp);
        const unsigned char* qs = bp + 2;
        const unsigned char* ssc = bp + 66;
        const int8_t* sb_aq = s_aq + sb * 256;

        for (int t = 0; t < 8; t++) {
            const int in = lane + t * 32;
            const int ib = in >> 5;
            const int pos = in & 31;
            const int l = pos >> 3;
            const int sub = pos & 7;
            const uint32_t aux = (uint32_t)ssc[ib * 4] |
                ((uint32_t)ssc[ib * 4 + 1] << 8) |
                ((uint32_t)ssc[ib * 4 + 2] << 16) |
                ((uint32_t)ssc[ib * 4 + 3] << 24);
            const float db = dd * (0.5f + (float)(aux >> 28)) * 0.5f;
            const uint8_t signs = dev_ksigns_iq2xs[(aux >> (7 * l)) & 127];
            const int jx = (sub < 4) ? sub : sub - 4;
            const unsigned char qb = (sub < 4)
                ? qs[ib * 8 + 2 * l]
                : qs[ib * 8 + 2 * l + 1];
            const uint32_t gx = dev_iq3xxs_grid[qb];
            const int sg = (signs & (int)(1u << sub)) ? -1 : 1;
            const int gv_signed = (int)((gx >> (8 * jx)) & 0xFF) * sg;
            const int u = *(const int*)(sb_aq + in);
            acc += db * (float)__dp4a(gv_signed, u, 0);
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) c[row] = acc;
}

// ─── dev-IQ P0-6 (2026-09-14): dp4a para iq2_xxs M=1 (case 13) ──────────────
// Layout 66B/SB256 [d f16][qs64].
extern "C" __global__ void iq2xxsGemmM1Dp4aKernel(
    const float* __restrict__ a,
    const unsigned char* __restrict__ b,
    float* __restrict__ c,
    int K, int N)
{
    extern __shared__ unsigned char smem_raw[];
    const int sb_total = K >> 8;
    const int kb_total = K >> 5;
    int8_t* s_aq = (int8_t*)smem_raw;
    float* s_d8 = (float*)(smem_raw + ((K + 15) & ~15));

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    for (int kb = warp; kb < kb_total; kb += (blockDim.x >> 5)) {
        const int base = kb * 32 + lane;
        const float v = a[base];
        float amax = fabsf(v);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d = (amax > 0.0f) ? amax / 127.0f : 1.0f;
        int q = (int)roundf(v / d);
        q = max(-127, min(127, q));
        s_aq[base] = (int8_t)q;
        if (lane == 0) s_d8[kb] = d;
    }
    __syncthreads();

    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= N) return;
    const unsigned char* rowb = b + (size_t)row * ((size_t)sb_total * 66);

    float acc = 0.0f;
    for (int sb = 0; sb < sb_total; sb++) {
        const unsigned char* bp = rowb + (size_t)sb * 66;
        const float dd = __half2float(*(const __half*)bp);
        const unsigned char* qs = bp + 2;
        const int8_t* sb_aq = s_aq + sb * 256;

        for (int t = 0; t < 8; t++) {
            const int in = lane + t * 32;
            const int ib = in >> 5;
            const int pos = in & 31;
            const int l = pos >> 3;
            const int j = pos & 7;
            const uint32_t aux0 = (uint32_t)qs[ib * 8] |
                ((uint32_t)qs[ib * 8 + 1] << 8) |
                ((uint32_t)qs[ib * 8 + 2] << 16) |
                ((uint32_t)qs[ib * 8 + 3] << 24);
            const uint32_t aux1 = (uint32_t)qs[ib * 8 + 4] |
                ((uint32_t)qs[ib * 8 + 5] << 8) |
                ((uint32_t)qs[ib * 8 + 6] << 16) |
                ((uint32_t)qs[ib * 8 + 7] << 24);
            const float db = dd * (0.5f + (float)(aux1 >> 28)) * 0.25f;
            const int idxg = (int)((aux0 >> (8 * l)) & 0xFF);
            const uint8_t signs = dev_ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
            const unsigned long long g = dev_iq2xxs_grid[idxg];
            const int sg = (signs & (int)(1u << j)) ? -1 : 1;
            const int gv_signed = (int)((g >> (8 * j)) & 0xFF) * sg;
            const int u = *(const int*)(sb_aq + in);
            acc += db * (float)__dp4a(gv_signed, u, 0);
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) c[row] = acc;
}

// ─── dev-IQ P0-6 (2026-09-14): dp4a para iq2_xs M=1 (case 14) ───────────────
// Layout 74B/SB256 [d f16][qs64][scales8].
extern "C" __global__ void iq2xsGemmM1Dp4aKernel(
    const float* __restrict__ a,
    const unsigned char* __restrict__ b,
    float* __restrict__ c,
    int K, int N)
{
    extern __shared__ unsigned char smem_raw[];
    const int sb_total = K >> 8;
    const int kb_total = K >> 5;
    int8_t* s_aq = (int8_t*)smem_raw;
    float* s_d8 = (float*)(smem_raw + ((K + 15) & ~15));

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    for (int kb = warp; kb < kb_total; kb += (blockDim.x >> 5)) {
        const int base = kb * 32 + lane;
        const float v = a[base];
        float amax = fabsf(v);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d = (amax > 0.0f) ? amax / 127.0f : 1.0f;
        int q = (int)roundf(v / d);
        q = max(-127, min(127, q));
        s_aq[base] = (int8_t)q;
        if (lane == 0) s_d8[kb] = d;
    }
    __syncthreads();

    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= N) return;
    const unsigned char* rowb = b + (size_t)row * ((size_t)sb_total * 74);

    float acc = 0.0f;
    for (int sb = 0; sb < sb_total; sb++) {
        const unsigned char* bp = rowb + (size_t)sb * 74;
        const float dd = __half2float(*(const __half*)bp);
        const unsigned char* qs = bp + 2;
        const unsigned char* scales = bp + 66;
        const int8_t* sb_aq = s_aq + sb * 256;

        for (int t = 0; t < 8; t++) {
            const int in = lane + t * 32;
            const int ib = in >> 5;
            const int rem = in & 31;
            const int l = rem >> 3;
            const int j = rem & 7;
            const float db0 = dd * (0.5f + (float)(scales[ib] & 0xF)) * 0.25f;
            const float db1 = dd * (0.5f + (float)(scales[ib] >> 4)) * 0.25f;
            const float db = (l < 2) ? db0 : db1;
            const uint16_t v = (uint16_t)qs[ib * 8 + l * 2] |
                ((uint16_t)qs[ib * 8 + l * 2 + 1] << 8);
            const uint8_t signs = dev_ksigns_iq2xs[v >> 9];
            const unsigned long long g = dev_iq2xs_grid[v & 511];
            const int sg = (signs & (int)(1u << j)) ? -1 : 1;
            const int gv_signed = (int)((g >> (8 * j)) & 0xFF) * sg;
            const int u = *(const int*)(sb_aq + in);
            acc += db * (float)__dp4a(gv_signed, u, 0);
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) c[row] = acc;
}

extern "C" __global__ void mergeFeedbackKernel(
    const float* __restrict__ encoder_rep,
    const float* __restrict__ prev_state,
    const float* __restrict__ w_gate,       // [d, 2*d]
    const float* __restrict__ w_state,      // [d, d]
    float* __restrict__ out,
    int d, float alpha)
{
    // Shared memory layout:
    // [0, d)          = r (RMSNorm of prev_state)
    // [d, 2*d)        = gate_preact (pre-activation gate)
    // [2*d, 3*d)      = state_proj (W_state @ r)
    extern __shared__ float smem[];
    float* r = smem;                // [d]
    float* gate_preact = smem + d;  // [d]
    float* state_proj = smem + 2*d; // [d]

    // Step 1: RMSNorm(prev_state) → r
    float ss = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float v = prev_state[i];
        ss += v * v;
    }
    // Warp-level reduction
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        ss += __shfl_xor_sync(0xffffffffu, ss, offset, WARP);
    }
    // Inter-warp reduction via shared memory
    __shared__ float warp_sums[32];
    int warp_id = threadIdx.x / WARP;
    int lane_id = threadIdx.x % WARP;
    if (lane_id == 0) warp_sums[warp_id] = ss;
    __syncthreads();
    int num_warps = (blockDim.x + WARP - 1) / WARP;
    if (warp_id == 0) {
        ss = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        for (int offset = WARP / 2; offset > 0; offset >>= 1) {
            ss += __shfl_xor_sync(0xffffffffu, ss, offset, WARP);
        }
    }
    __shared__ float s_rms;
    if (threadIdx.x == 0) s_rms = rsqrtf(ss / (float)d + 1e-6f);
    __syncthreads();
    float inv_rms = s_rms;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        r[i] = prev_state[i] * inv_rms;
    }
    __syncthreads();

    // Step 2: gate_preact = W_gate @ [encoder_rep; r]
    // W_gate is [d, 2*d] row-major: row i, col j → w_gate[i*(2*d) + j]
    // concat = [encoder_rep(0..d-1), r(0..d-1)]
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float acc = 0.0f;
        int row_off = i * (2 * d);
        // encoder_rep portion
        for (int j = 0; j < d; j++) {
            acc += w_gate[row_off + j] * encoder_rep[j];
        }
        // r portion
        for (int j = 0; j < d; j++) {
            acc += w_gate[row_off + d + j] * r[j];
        }
        gate_preact[i] = acc;
    }
    __syncthreads();

    // Step 3: Sigmoid gate
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        gate_preact[i] = 1.0f / (1.0f + __expf(-gate_preact[i]));
    }
    __syncthreads();

    // Step 4: state_proj = W_state @ r
    // W_state is [d, d] row-major
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float acc = 0.0f;
        int row_off = i * d;
        for (int j = 0; j < d; j++) {
            acc += w_state[row_off + j] * r[j];
        }
        state_proj[i] = acc;
    }
    __syncthreads();

    // Step 5: out = encoder_rep + alpha * gate * state_proj
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        out[i] = encoder_rep[i] + alpha * gate_preact[i] * state_proj[i];
    }
}
