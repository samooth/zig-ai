// Elementwise / normalization CUDA kernels for the GPU-resident hybrid layer.
// Compiled by the build into a cubin and launched from layer_kernels.zig via the
// CUDA driver API (same pattern as paged_attention.cu).
//
// Convention: all pointers are device pointers. Kernels do NOT sync; the caller
// synchronizes the stream once per token.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>

#define WARP 32

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

// ─── L2-normalize Q and K heads in conv_out (fiel a ggml_l2_norm) ─────────────
// conv_out: [N, qkv_dim]; Q in [0,key_dim), K in [key_dim, 2*key_dim).
// scale = 1 / max(sqrt(sum x^2), eps) per head.
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
        float scale = 1.0f / fmaxf(sqrtf(reds[0]), eps);
        for (int i = threadIdx.x; i < head_v_dim; i += blockDim.x)
            conv_out[base + i] *= scale;
    }
}

// ─── conv1d (causal) + silu on qkv → conv_out ────────────────────────────────
// conv_in: [(d_conv-1)+N, qkv_dim] (history rows then current); conv_w: [qkv_dim, d_conv].
extern "C" __global__ void conv1dSiluKernel(
    const float* __restrict__ conv_in,
    const float* __restrict__ conv_w,
    float* __restrict__ conv_out,
    int N, int qkv_dim, int d_conv)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * qkv_dim) return;
    int t = idx / qkv_dim;
    int c = idx % qkv_dim;
    float sum = 0.0f;
    for (int k = 0; k < d_conv; ++k)
        sum += conv_in[((size_t)(t + k) * qkv_dim) + c] * conv_w[(size_t)c * d_conv + k];
    conv_out[(size_t)t * qkv_dim + c] = sum / (1.0f + expf(-sum));
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
    int hk = hv / (n_v_heads / n_k_heads);
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
// El CPU copia flat [N, heads, dim] en un tensor [1, heads, N, dim] y rota con
// row_offset = (h*N + pos)*head_dim; replicamos esa semántica exacta (incluida
// la transposición) para que decode GPU == prefill CPU. Para texto los 4
// position ids son iguales → theta = global_pos * scale^ic en todos los sectores.
// data: [N, n_head, head_dim] flat; rows = n_head (o n_kv_head) * N.
extern "C" __global__ void mropeKernel(
    float* __restrict__ data,
    int rows, int N, int head_dim, int n_rot,
    float base, long long start_pos)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    int pos = row % N;
    float global_pos = (float)(start_pos + pos);
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

// ─── KV-append f16: escribe K/V del token al pool paginado (d_cache).
// Fiel a reshape_and_block_write_f16_kernel: por bloque físico, K en
// [0, block_size*kv_dim), V en [block_size*kv_dim, 2*block_size*kv_dim).
// k/v: [N, kv_dim] f32; bt: block table device (phys por bloque lógico).
extern "C" __global__ void kvAppendF16Kernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    half* __restrict__ cache,
    const int* __restrict__ bt,
    int n, int start_pos,
    int kv_dim, int n_kv_head, int head_dim, int block_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * kv_dim;
    if (i >= total) return;
    int t = i / kv_dim;
    int c = i % kv_dim;
    int h = c / head_dim;
    int d = c % head_dim;
    long long global_pos = (long long)start_pos + t;
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
