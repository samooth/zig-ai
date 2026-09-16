// fused_decode_extra.cu — Lane A
// Kernels fused-decode para formatos aún no presentes en paged_attention.cu.
// MISMA FIRMA de 13 parámetros que los kernels existentes (out, query,
// cache_kv, k_scales/v_scales RESERVADOS por Contrato C1, block_tables,
// seq_lens, num_seqs, max_num_blocks, num_q_heads, num_kv_heads, head_dim,
// block_size); grid (1, num_q_heads), bloque 32 hilos.
//
// Layout de región por bloque físico: [K: k_bytes][V: k_bytes] con escalas
// EMBEBIDAS canónicas GGUF (Contrato C1, PLAN_MAESTRO). Los dequants siguen
// EXACTAMENTE las referencias verificadas en src/loader/gguf.zig /
// kernels/dequant_*.cu (bit-exactos en test_dequant_gpu).
//
// Propiedad: Lane A (ver LANE_A.md). No tocar desde otros lanes.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// ─── LUT canónica IQ4_NL (de kernels/dequant_iq4_xs.cu verificado) ──────────
__device__ const int8_t kvalues_iq4nl_lut[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

// ─── Value-fns legacy 32-elementos: base = grupo, in = elemento en grupo ────

extern "C" __device__ __forceinline__ float val_q4_1(const uint8_t* base, int in) {
    const float d = __half2float(__ushort_as_half((unsigned short)((uint16_t)base[0] | ((uint16_t)base[1] << 8))));
    const float m = __half2float(__ushort_as_half((unsigned short)((uint16_t)base[2] | ((uint16_t)base[3] << 8))));
    const int j = (in < 16) ? in : (in - 16);
    const uint8_t byte = base[4 + j];
    const int nib = (in < 16) ? (byte & 0x0F) : (byte >> 4);
    return d * (float)nib + m;
}

extern "C" __device__ __forceinline__ float val_q5_0(const uint8_t* base, int in) {
    const float d = __half2float(*reinterpret_cast<const __half*>(base));
    const uint32_t qh = (uint32_t)base[2] | ((uint32_t)base[3] << 8) |
                        ((uint32_t)base[4] << 16) | ((uint32_t)base[5] << 24);
    const int j = (in < 16) ? in : (in - 16);
    const int xh = (in < 16) ? (((qh >> j) & 1) << 4) : (((qh >> (j + 16)) & 1) << 4);
    const uint8_t byte = base[6 + j];
    const int nib = (in < 16) ? (byte & 0x0F) : (byte >> 4);
    return d * (float)((nib | xh) - 16);
}

extern "C" __device__ __forceinline__ float val_q5_1(const uint8_t* base, int in) {
    const float d = __half2float(__ushort_as_half((unsigned short)((uint16_t)base[0] | ((uint16_t)base[1] << 8))));
    const float m = __half2float(__ushort_as_half((unsigned short)((uint16_t)base[2] | ((uint16_t)base[3] << 8))));
    const uint32_t qh = (uint32_t)base[4] | ((uint32_t)base[5] << 8) |
                        ((uint32_t)base[6] << 16) | ((uint32_t)base[7] << 24);
    const int j = (in < 16) ? in : (in - 16);
    const int xh = (in < 16) ? (((qh >> j) & 1) << 4) : (((qh >> (j + 16)) & 1) << 4);
    const uint8_t byte = base[8 + j];
    const int nib = (in < 16) ? (byte & 0x0F) : (byte >> 4);
    return d * (float)((nib | xh)) + m;
}

extern "C" __device__ __forceinline__ float val_q8_1(const uint8_t* base, int in) {
    const float d = __half2float(__ushort_as_half((unsigned short)((uint16_t)base[0] | ((uint16_t)base[1] << 8))));
    const float s = __half2float(__ushort_as_half((unsigned short)((uint16_t)base[2] | ((uint16_t)base[3] << 8))));
    return d * (float)(int8_t)base[4 + in] + s;
}

// ─── IQ4_XS: super-bloque 256 elems, 136 bytes ──────────────────────────────
// dl = d * (ls - 32); val = dl * kvalues[q]; ls empaquetado 6-bit
// (4-bit low en scales_l[ib/2] + 2-bit high en scales_h).
extern "C" __device__ __forceinline__ float val_iq4_xs(const uint8_t* base, int in) {
    const float d = __half2float(*reinterpret_cast<const __half*>(base));
    const uint16_t scales_h = *(const uint16_t*)(base + 2);
    const uint8_t* scales_l = base + 4;
    const uint8_t* qs = base + 8;
    const int ib = in / 32;   // 0..7 sub-bloques
    const int j  = in % 32;
    int ls = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF;
    ls |= ((scales_h >> (2 * ib)) & 3) << 4;
    const float dl = d * (float)(ls - 32);
    const int qidx = (j < 16) ? j : (j - 16);
    const uint8_t qb = qs[ib * 16 + qidx];
    const int qv = (j < 16) ? (qb & 0xF) : (qb >> 4);
    return dl * (float)kvalues_iq4nl_lut[qv];
}

// ─── Macro generadora: cuerpo idéntico al decode q8_0 escalar probado ───────
// VALFN(base_ptr_expr, local_idx) — se expande por formato.
#define DEFINE_EXTRA_DECODE_KERNEL(NAME, SB_SHIFT, STRIDE_CONST, VALFN)       \
extern "C" __global__ void NAME(                                               \
    half* __restrict__ out,                                                    \
    const half* __restrict__ query,                                            \
    const uint8_t* __restrict__ cache_kv,                                      \
    const half* __restrict__ k_scales,                                         \
    const half* __restrict__ v_scales,                                         \
    const int* __restrict__ block_tables,                                      \
    const int* __restrict__ seq_lens,                                          \
    int num_seqs, int max_num_blocks, int num_q_heads,                         \
    int num_kv_heads, int head_dim, int block_size)                            \
{                                                                              \
    (void)k_scales; /* Contrato C1: escalas embebidas */                       \
    (void)v_scales;                                                            \
    const int seq_idx = blockIdx.x;                                            \
    const int q_head  = blockIdx.y;                                            \
    if (seq_idx >= num_seqs) return;                                           \
    const int seq_len = seq_lens[seq_idx];                                     \
    if (seq_len == 0) return;                                                  \
    const int tid = threadIdx.x;                                               \
    const int kv_head = q_head / (num_q_heads / num_kv_heads);                 \
    const int elems_per_block = block_size * num_kv_heads * head_dim;          \
    const int sb_elems = 1 << (SB_SHIFT);                                      \
    const int n_sb = (elems_per_block + sb_elems - 1) >> (SB_SHIFT);           \
    const size_t k_bytes = (size_t)n_sb * (STRIDE_CONST);                      \
    const size_t phys_stride = 2 * k_bytes;                                    \
    const int q_offset = (seq_idx * num_q_heads + q_head) * head_dim;          \
    const int kv_head_stride = num_kv_heads * head_dim;                        \
                                                                               \
    extern __shared__ float smem[];                                            \
    float* sq  = smem;                                                         \
    float* acc = smem + head_dim;                                              \
    for (int e = tid; e < head_dim; e += blockDim.x) {                         \
        sq[e]  = __half2float(query[q_offset + e]);                            \
        acc[e] = 0.0f;                                                         \
    }                                                                          \
    __syncthreads();                                                           \
                                                                               \
    float max_val = -1e30f;                                                    \
    float exp_sum = 0.0f;                                                      \
    const float scale_factor = rsqrtf((float)head_dim);                        \
    const int num_blocks = (seq_len + block_size - 1) / block_size;            \
                                                                               \
    for (int b = 0; b < num_blocks; b++) {                                     \
        const int phys = block_tables[seq_idx * max_num_blocks + b];           \
        if (phys < 0) continue;                                                \
        const int tokens_in_block = (b == num_blocks - 1)                      \
            ? (seq_len - b * block_size) : block_size;                         \
        const uint8_t* k_data = cache_kv + (size_t)phys * phys_stride;         \
        const uint8_t* v_data = k_data + k_bytes;                              \
                                                                               \
        for (int t = 0; t < tokens_in_block; t++) {                            \
            const int t_offset = t * kv_head_stride + kv_head * head_dim;      \
            float partial = 0.0f;                                              \
            for (int e = tid; e < head_dim; e += blockDim.x) {                 \
                const int be = t_offset + e;                                   \
                const uint8_t* sb = k_data + ((size_t)(be >> (SB_SHIFT)) * (STRIDE_CONST)); \
                partial += sq[e] * VALFN(sb, be & (sb_elems - 1));             \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int off = 16; off > 0; off >>= 1)                             \
                partial += __shfl_xor_sync(0xffffffffu, partial, off);         \
            const float score     = partial * scale_factor;                    \
            const float new_max   = fmaxf(max_val, score);                     \
            const float rescale   = expf(max_val - new_max);                   \
            const float exp_score = expf(score - new_max);                     \
            exp_sum = exp_sum * rescale + exp_score;                           \
            for (int e = tid; e < head_dim; e += blockDim.x) acc[e] *= rescale;\
            max_val = new_max;                                                 \
            for (int e = tid; e < head_dim; e += blockDim.x) {                 \
                const int be = t_offset + e;                                   \
                const uint8_t* sb = v_data + ((size_t)(be >> (SB_SHIFT)) * (STRIDE_CONST)); \
                acc[e] += VALFN(sb, be & (sb_elems - 1)) * exp_score;          \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    const float inv_sum = 1.0f / exp_sum;                                      \
    for (int e = tid; e < head_dim; e += blockDim.x)                           \
        out[q_offset + e] = __float2half(acc[e] * inv_sum);                    \
}

// Legacy 32-elementos: grupos de 32, strides canónicos por formato
DEFINE_EXTRA_DECODE_KERNEL(paged_attention_decode_q4_1_kernel, 5, 20, val_q4_1)
DEFINE_EXTRA_DECODE_KERNEL(paged_attention_decode_q5_0_kernel, 5, 22, val_q5_0)
DEFINE_EXTRA_DECODE_KERNEL(paged_attention_decode_q5_1_kernel, 5, 24, val_q5_1)
DEFINE_EXTRA_DECODE_KERNEL(paged_attention_decode_q8_1_kernel, 5, 36, val_q8_1)

// IQ4_XS: super-bloque 256 elems, 136 bytes
DEFINE_EXTRA_DECODE_KERNEL(paged_attention_decode_iq4_xs_kernel, 8, 136, val_iq4_xs)

#define LO16(p) ((unsigned short)((unsigned long long)(p) & 0xFFFFull))
#define HI16(p) ((unsigned short)(((unsigned long long)(p) >> 16) & 0xFFFFull))
// ─────────────────────────────────────────────────────────────────────────────
// PREFILL (Lane A): variante causal del patrón prefill_q8_0_kernel.
// Firma 10-param idéntica: out, queries, cache_kv, block_tables,
// n_queries, start_pos, num_q_heads, num_kv_heads, head_dim, block_size.
// Grid (n_queries, num_q_heads), bloque 32, smem 2*hd floats.
// ─────────────────────────────────────────────────────────────────────────────

extern "C" __device__ __forceinline__ float valpref_q4_0(const uint8_t* base, int in) {
    const float d = __half2float(__ushort_as_half((unsigned short)(
        (uint16_t)base[0] | ((uint16_t)base[1] << 8))));
    const int j = (in < 16) ? in : (in - 16);
    const uint8_t byte = base[2 + j];
    const int nib = (in < 16) ? (byte & 0x0F) : (byte >> 4);
    return d * (float)((int)nib - 8);
}

extern "C" __device__ __forceinline__ float valpref_iq4_xs(const uint8_t* base, int in) {
    const float d = __half2float(__ushort_as_half((unsigned short)(
        (uint16_t)base[0] | ((uint16_t)base[1] << 8))));
    const uint16_t scales_h = (uint16_t)base[2] | ((uint16_t)base[3] << 8);
    const uint8_t* scales_l = base + 4;
    const uint8_t* qs = base + 8;
    const int ib = in / 32;
    const int j  = in % 32;
    int ls = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF;
    ls |= ((scales_h >> (2 * ib)) & 3) << 4;
    const float dl = d * (float)(ls - 32);
    const int qidx = (j < 16) ? j : (j - 16);
    const uint8_t qb = qs[ib * 16 + qidx];
    const int qv = (j < 16) ? (qb & 0xF) : (qb >> 4);
    return dl * (float)kvalues_iq4nl_lut[qv];
}

// A5: el parámetro `causal` de este macro es la máscara parametrizable
// (documentado en paged_attention_prefill_f16_kernel).
// ─── Block-level reduction helper for arbitrary block sizes ───────────────────
// Performs sum reduction across all threads in the block using shared memory.
// Works for any block size (multiple of 32).
__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float shared[32];  // Max 32 warps (1024 threads)
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    int num_warps = (blockDim.x + 31) / 32;

    // Warp-level reduction
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_xor_sync(0xffffffffu, val, off);

    // First thread in each warp writes to shared memory
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // First warp reduces the per-warp results
    val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;
    if (wid == 0) {
        for (int off = 16; off > 0; off >>= 1)
            val += __shfl_xor_sync(0xffffffffu, val, off);
    }
    __syncthreads();

    return val;
}

#define DEFINE_EXTRA_PREFILL_KERNEL(NAME, SB_SHIFT, STRIDE_CONST, VALFN)       \
extern "C" __global__ void NAME(                                               \
    half* __restrict__ out,                                                    \
    const half* __restrict__ queries,                                          \
    const uint8_t* __restrict__ cache_kv,                                      \
    const int* __restrict__ block_tables,                                      \
    int n_queries, int start_pos, int num_q_heads,                             \
    int num_kv_heads, int head_dim, int block_size,                            \
    int causal)                                                                \
{                                                                              \
    const int token = blockIdx.x;                                              \
    const int q_head  = blockIdx.y;                                            \
    if (token >= n_queries) return;                                            \
    const int abs_token = start_pos + token;                                   \
    const int tid = threadIdx.x;                                               \
    const int kv_head = q_head / (num_q_heads / num_kv_heads);                 \
    if (head_dim == 31336) {                                                   \
        /* PROBE-K Lane A: volcado CRUDO de K-dequant del bloque fisico 0.     \
         * out[token,q_head,d] = K[tok=token,kv_head,d] sin softmax. */        \
        const int hd_p = 8;                                                    \
        const int eb_p = ((4 * num_kv_heads * hd_p) + 31) >> 5;                \
        const size_t kb_p = (size_t)eb_p * (STRIDE_CONST);                     \
        const uint8_t* kp_ = cache_kv + (size_t)block_tables[0] * 2 * kb_p;    \
        const int toff_p = token * num_kv_heads * hd_p + kv_head * hd_p;       \
        const int qo_p = token * num_q_heads * hd_p + q_head * hd_p;           \
        for (int d = tid; d < hd_p; d += blockDim.x) {                         \
            const int be = toff_p + d;                                         \
            out[qo_p + d] = __float2half(                                      \
                VALFN(kp_ + (size_t)(be >> (SB_SHIFT)) * (STRIDE_CONST),       \
                      be & ((1 << (SB_SHIFT)) - 1)));                          \
        }                                                                      \
        return;                                                                \
    }                                                                          \
    if (head_dim == 31338) {                                                   \
        /* PROBE-S Lane A: escribe el SCORE de t=0 (dot(q,K[0])*rsqrt(hd))     \
         * en out[token,q_head,0..7]. Diagnóstico camino K. */                 \
        const int hd_p = 8;                                                    \
        const int eb_p = ((4 * num_kv_heads * hd_p) + 31) >> 5;                \
        const size_t kb_p = (size_t)eb_p * (STRIDE_CONST);                     \
        const uint8_t* kp_ = cache_kv + (size_t)block_tables[0] * 2 * kb_p;    \
        const int toff_p = 0 * num_kv_heads * hd_p + kv_head * hd_p;           \
        const int qo_p = token * num_q_heads * hd_p + q_head * hd_p;           \
        extern __shared__ float sp[];                                          \
        for (int d = tid; d < hd_p; d += blockDim.x)                           \
            sp[d] = __half2float(queries[qo_p + d]);                           \
        __syncthreads();                                                       \
        float part = 0.0f;                                                     \
        for (int d = tid; d < hd_p; d += blockDim.x) {                         \
            const int be = toff_p + d;                                         \
            part += sp[d] * VALFN(kp_ + (size_t)(be >> (SB_SHIFT)) * (STRIDE_CONST), \
                                  be & ((1 << (SB_SHIFT)) - 1));               \
        }                                                                      \
        part = blockReduceSum(part);                                           \
        if (tid < hd_p) out[qo_p + tid] = __float2half(part * rsqrtf((float)hd_p)); \
        return;                                                                \
    }                                                                          \
    if (head_dim == 31337) {                                                   \
        /* PROBE Lane A: escribe V-dequant crudo del bloque físico 0.          \
         * out[token,q_head,d] = V[tok=token,kv_head,d] sin softmax.           \
         * Diagnóstico: si esto coincide con la referencia CPU, la lectura     \
         * y dequant son correctas y el bug vive en el camino K/score. */      \
        const int hd_p = 8;                                                    \
        const int eb_p = ((4 * num_kv_heads * hd_p) + 31) >> 5;                \
        const size_t kb_p = (size_t)eb_p * (STRIDE_CONST);                     \
        const uint8_t* vp = cache_kv + (size_t)block_tables[0] * 2 * kb_p      \
                          + kb_p;                                              \
        const int toff_p = token * num_kv_heads * hd_p + kv_head * hd_p;       \
        const int qo_p = token * num_q_heads * hd_p + q_head * hd_p;           \
        for (int d = tid; d < hd_p; d += blockDim.x) {                         \
            const int be = toff_p + d;                                         \
            out[qo_p + d] = __float2half(                                      \
                VALFN(vp + (size_t)(be >> (SB_SHIFT)) * (STRIDE_CONST),        \
                      be & ((1 << (SB_SHIFT)) - 1)));                          \
        }                                                                      \
        return;                                                                \
    }                                                                          \
    if (head_dim == 31340) {                                                   \
        /* PROBE-ECHO Lane A: escribe los params recibidos tal cual.           \
         * [0..1]=lo/hi cache_kv, [2..3]=lo/hi block_tables,                    \
         * [4..5]=lo/hi queries, [6]=nq+sp*100, [7]=qh*100+kvh,                \
         * [8]=block_tables[0], [9]=block_tables[1] (LEÍDOS DEL DEVICE) */     \
        const int qo_e = token * num_q_heads * 8 + q_head * 8;                 \
        if (tid == 0) out[qo_e + 0] = __ushort_as_half(LO16(cache_kv));        \
        if (tid == 0) out[qo_e + 1] = __ushort_as_half(HI16(cache_kv));        \
        if (tid == 0) out[qo_e + 2] = __ushort_as_half(LO16(block_tables));    \
        if (tid == 0) out[qo_e + 3] = __ushort_as_half(HI16(block_tables));    \
        if (tid == 0) out[qo_e + 4] = __ushort_as_half(LO16(queries));         \
        if (tid == 0) out[qo_e + 5] = __ushort_as_half(HI16(queries));         \
        if (tid == 0) out[qo_e + 6] = __float2half((float)(n_queries           \
                                 + start_pos * 100));                          \
        if (tid == 0) out[qo_e + 7] = __float2half((float)(num_q_heads * 100   \
                                 + num_kv_heads));                             \
        if (tid == 0) out[qo_e + 8] = __float2half((float)block_tables[0]);    \
        if (tid == 1) out[qo_e + 9] = __float2half((float)block_tables[1]);    \
        return;                                                                \
    }                                                                          \
    const bool diag = (head_dim == 31339);                                     \
    const int hd_real = diag ? 8 : head_dim;                                   \
    const int elems_per_block = block_size * num_kv_heads * hd_real;           \
    const int sb_elems = 1 << (SB_SHIFT);                                      \
    const int n_sb = (elems_per_block + sb_elems - 1) >> (SB_SHIFT);           \
    const size_t k_bytes = (size_t)n_sb * (STRIDE_CONST);                      \
    const size_t phys_stride = 2 * k_bytes;                                    \
    const int q_stride = num_q_heads * hd_real;                                \
    const int q_offset = token * q_stride + q_head * hd_real;                  \
                                                                               \
    extern __shared__ float smem[];                                            \
    float* sq  = smem;                                                         \
    float* acc = smem + hd_real;                                               \
    if (!diag)                                                                 \
    for (int d = tid; d < hd_real; d += blockDim.x) {                          \
        sq[d]  = __half2float(queries[q_offset + d]);                          \
        acc[d] = 0.0f;                                                         \
    }                                                                          \
    if (diag)                                                                  \
    for (int d = tid; d < hd_real; d += blockDim.x) {                          \
        sq[d]  = __half2float(queries[q_offset + d]);                          \
        acc[d] = 0.0f;                                                         \
    }                                                                          \
    __syncthreads();                                                           \
                                                                               \
    float max_val = -1e30f;                                                    \
    float exp_sum = 0.0f;                                                      \
    const float scale_factor = rsqrtf((float)hd_real);                        \
    const int ctx_end = causal ? abs_token : (start_pos + gridDim.x - 1);      \
    const int last_block = ctx_end / block_size;                               \
                                                                               \
    for (int b = 0; b <= last_block; b++) {                                    \
        const int phys = block_tables[b];                                      \
        if (phys < 0) continue;                                                \
        const int tokens_in_block = (b == last_block)                          \
            ? (ctx_end % block_size) + 1 : block_size;                       \
        const uint8_t* k_data = cache_kv + (size_t)phys * phys_stride;         \
        const uint8_t* v_data = k_data + k_bytes;                              \
                                                                               \
        for (int t = 0; t < tokens_in_block; t++) {                            \
            const int t_offset = t * num_kv_heads * hd_real                   \
                               + kv_head * hd_real;                           \
            float partial = 0.0f;                                              \
            for (int d = tid; d < hd_real; d += blockDim.x) {                 \
                const int be = t_offset + d;                                   \
                const uint8_t* sb = k_data + ((size_t)(be >> (SB_SHIFT)) * (STRIDE_CONST)); \
                partial += sq[d] * VALFN(sb, be & (sb_elems - 1));             \
            }                                                                  \
            partial = blockReduceSum(partial);                                 \
            const float score     = partial * scale_factor;                    \
            const float new_max   = fmaxf(max_val, score);                     \
            const float rescale   = expf(max_val - new_max);                   \
            exp_sum *= rescale;                                                \
            for (int d = tid; d < hd_real; d += blockDim.x) acc[d] *= rescale;\
            max_val = new_max;                                                 \
            const float exp_score = expf(score - new_max);                     \
            exp_sum += exp_score;                                              \
            for (int d = tid; d < hd_real; d += blockDim.x) {                 \
                const int be = t_offset + d;                                   \
                const uint8_t* sb = v_data + ((size_t)(be >> (SB_SHIFT)) * (STRIDE_CONST)); \
                acc[d] += VALFN(sb, be & (sb_elems - 1)) * exp_score;          \
            }                                                                  \
        }                                                                      \
    }                                                                          \
                                                                               \
    if (diag) {                                                                \
        /* PROBE-D: estado interno tras el bucle completo */                   \
        if (tid == 0) out[q_offset + 0] = __float2half(max_val);               \
        if (tid == 1) out[q_offset + 1] = __float2half(exp_sum);               \
        for (int d = tid + 2; d < hd_real; d += blockDim.x)                    \
            out[q_offset + d] = __float2half(acc[d - 2]);                      \
        return;                                                                \
    }                                                                          \
    const float inv_sum = 1.0f / exp_sum;                                      \
    for (int d = tid; d < hd_real; d += blockDim.x)                           \
        out[q_offset + d] = __float2half(acc[d] * inv_sum);                    \
}

// q4_0: petición formal de lane-b para desbloquear su E2E q4_0
DEFINE_EXTRA_PREFILL_KERNEL(paged_attention_prefill_q4_0_kernel, 5, 18, valpref_q4_0)
// iq4_xs: formato local (NEO-MTP-IQ4_XS)
DEFINE_EXTRA_PREFILL_KERNEL(paged_attention_prefill_iq4_xs_kernel, 8, 136, valpref_iq4_xs)

// ─── I-quants (Lane A): requieren tables.cuh (grids LUT) ────────────────────
#include "tables.cuh"

// IQ3_S: SB 256 elems, 110 bytes
// d f16@0 | qs[64]@2 | qh[8]@66 | signs[32]@74 | scales[4]@106
// db = d*(1+2*(sc nibbles)); val = db * byte_j(grid[idx]) * sign(kmask)
__device__ __forceinline__ uint32_t idx3s_a(uint8_t q, uint8_t h, int l) {
    return (uint32_t)q | (((uint32_t)h << (8 - 2 * l)) & 256u);
}
__device__ __forceinline__ uint32_t idx3s_b(uint8_t q, uint8_t h, int l) {
    return (uint32_t)q | (((uint32_t)h << (7 - 2 * l)) & 256u);
}
extern "C" __device__ __forceinline__ float val_iq3_s(const uint8_t* base, int in) {
    const float d = __half2float(__ushort_as_half((unsigned short)(
        (uint16_t)base[0] | ((uint16_t)base[1] << 8))));
    const int it   = in / 64;
    const int rem  = in % 64;
    const int half = rem / 32;
    const int pos  = rem % 32;
    const int l    = pos / 8;
    const int col  = pos % 8;
    const uint8_t* qs     = base + 2;
    const uint8_t* qh     = base + 66;
    const uint8_t* signs  = base + 74;
    const uint8_t* scales = base + 106;
    const uint8_t sc = scales[it];
    const float db = (half == 0) ? d * (1.0f + 2.0f * (float)(sc & 0xF))
                                 : d * (1.0f + 2.0f * (float)(sc >> 4));
    const uint8_t* q  = (half == 0) ? (qs + it * 16) : (qs + it * 16 + 8);
    const uint8_t hb  = qh[2 * it + half];
    const uint8_t sm  = signs[it * 8 + half * 4 + l];
    const uint32_t e  = (col < 4) ? iq3s_grid[idx3s_a(q[2 * l], hb, l)]
                                  : iq3s_grid[idx3s_b(q[2 * l + 1], hb, l)];
    const int j = (col < 4) ? col : (col - 4);
    const int kmask = (col < 4) ? kmask_iq2xs[j] : kmask_iq2xs[j + 4];
    const int sgn = (sm & kmask) ? -1 : 1;
    const float gv = (float)(uint8_t)((e >> (8 * j)) & 0xFF);
    return db * gv * (float)sgn;
}

// IQ1_S: SB 256 elems, 50 bytes
// d f16@0 | qs[16]@2 | qh[16]@34 (u16 por ib: delta3b + sign flag)
// dl = d*(1+2*(3-bit delta)); dd = ±0.125; val = dl*(grid_byte + dd)
extern "C" __device__ __forceinline__ float val_iq1_s(const uint8_t* base, int in) {
    const float d = __half2float(__ushort_as_half((unsigned short)(
        (uint16_t)base[0] | ((uint16_t)base[1] << 8))));
    const int ib  = in / 32;
    const int rem = in % 32;
    const int l   = rem / 8;
    const int j   = rem % 8;
    const uint8_t* qs = base + 2;
    const uint8_t* qh = base + 34;
    const uint16_t qhb = (uint16_t)qh[ib * 2] | ((uint16_t)qh[ib * 2 + 1] << 8);
    const float dl = d * (2.0f * (float)((qhb >> 12) & 7) + 1.0f);
    const float dd = (qhb & 0x8000) ? -0.125f : 0.125f;
    const int idxg = qs[ib * 4 + l] | (((qhb >> (3 * l)) & 7) << 8);
    const unsigned long long g = iq1s_grid[idxg];
    const float gv = (float)(int8_t)((g >> (8 * j)) & 0xFF);
    return dl * (gv + dd);
}

DEFINE_EXTRA_DECODE_KERNEL(paged_attention_decode_iq3_s_kernel, 8, 110, val_iq3_s)
DEFINE_EXTRA_DECODE_KERNEL(paged_attention_decode_iq1_s_kernel, 8, 50, val_iq1_s)

// IQ1_M: SB 256 elems, 56 bytes (completa el set A2-A4; espejo dequantIq1_m)
// sc[8]@0 (solapado con qs[32]@0!), qh[16]@32.
// d f16 reensamblado desde bits dispersos: (sc0>>12)|(sc1>>8 &F0)|(sc2>>4 &F00)|(sc3 &F000).
// Por ib de 8 sub-bloques×32: dl1/dl2 desde campo 6b de sc16; idx=qs|qh&0x700;
// dd=±0.125 por bit qh; l<2 usa dl1, l>=2 usa dl2.
extern "C" __device__ __forceinline__ float val_iq1_m(const uint8_t* base, int in) {
    const int ib = in / 32;
    const int rem = in % 32;
    const int l   = rem / 8;
    const int j   = rem % 8;
    const uint16_t sc0 = (uint16_t)base[0] | ((uint16_t)base[1] << 8);
    const uint16_t sc1 = (uint16_t)base[2] | ((uint16_t)base[3] << 8);
    const uint16_t sc2 = (uint16_t)base[4] | ((uint16_t)base[5] << 8);
    const uint16_t sc3 = (uint16_t)base[6] | ((uint16_t)base[7] << 8);
    const uint16_t scale_u16 = (uint16_t)((sc0 >> 12) | ((sc1 >> 8) & 0xF0) |
                                          ((sc2 >> 4) & 0xF00) | (sc3 & 0xF000));
    const float d = __half2float(__ushort_as_half(scale_u16));
    const int sc_off = (ib >> 1) * 2;
    const uint16_t sc16 = (uint16_t)base[sc_off] | ((uint16_t)base[sc_off + 1] << 8);
    const float dl1 = d * (2.0f * (float)((sc16 >> (6 * (ib & 1) + 0)) & 7) + 1.0f);
    const float dl2 = d * (2.0f * (float)((sc16 >> (6 * (ib & 1) + 3)) & 7) + 1.0f);
    const uint8_t qb      = base[ib * 4 + l];
    const uint8_t qh_byte = base[32 + sc_off + (l >> 1)];
    const int idxg   = qb | ((((int)qh_byte << ((l & 1) ? 4 : 8))) & 0x700);
    const float dd   = (qh_byte & ((l & 1) ? 0x80 : 0x08)) ? -0.125f : 0.125f;
    const unsigned long long g = iq1s_grid[idxg];
    const float gv = (float)(int8_t)((g >> (8 * j)) & 0xFF);
    return (l < 2 ? dl1 : dl2) * (gv + dd);
}

DEFINE_EXTRA_DECODE_KERNEL(paged_attention_decode_iq1_m_kernel, 8, 56, val_iq1_m)

// ─── T-quants + MXFP4 + IQ4_NL + IQ2 family + IQ3_XXS (Lane A) ──────────────

// TQ1_0: SB 256, 54 bytes — ternario base-3 empacado
__device__ __forceinline__ uint16_t tq1_pow3(int n) {
    const uint16_t p3[6] = { 1, 3, 9, 27, 81, 243 };
    return p3[n];
}
extern "C" __device__ __forceinline__ float val_tq1_0(const uint8_t* base, int in) {
    const float d = __half2float(__ushort_as_half((unsigned short)(
        (uint16_t)base[52] | ((uint16_t)base[53] << 8))));
    const uint8_t* qs = base;
    const uint8_t* qh = base + 48;
    uint8_t q;
    if (in < 160) { int j = in / 5, n = in % 5; q = (uint8_t)((uint16_t)qs[j] * tq1_pow3(n)); }
    else if (in < 240) { int l = in - 160; int j = l / 5, n = l % 5; q = (uint8_t)((uint16_t)qs[32 + j] * tq1_pow3(n)); }
    else { int l = in - 240; int j = l / 4, n = l % 4; q = (uint8_t)((uint16_t)qh[j] * tq1_pow3(n)); }
    return d * (float)((int)(((q * 3) >> 8) - 1));
}

// TQ2_0: SB 256, 66 bytes — 2-bit plano
extern "C" __device__ __forceinline__ float val_tq2_0(const uint8_t* base, int in) {
    const float d = __half2float(__ushort_as_half((unsigned short)(
        (uint16_t)base[64] | ((uint16_t)base[65] << 8))));
    const int seg = in / 128, rem = in % 128;
    const int l = rem / 32, m = rem % 32;
    const int q = (base[seg * 32 + m] >> (2 * l)) & 3;
    return d * (float)(q - 1);
}

// MXFP4: 32 elems, 17 bytes — e8m0 exponent + fp4 LUT
extern "C" __device__ __forceinline__ float val_mxfp4(const uint8_t* base, int in) {
    // e8m0: exponente sin mantisa → 2^(x-127)
    const float d = exp2f((float)(int)base[0] - 127.0f);
    const uint8_t* qs = base + 1;
    const int q = (in < 16) ? (qs[in] & 0xF) : (qs[in - 16] >> 4);
    return d * (float)kvalues_fp4[q];
}

// IQ4_NL: 32 elems, 18 bytes — misma LUT iq4nl que iq4_xs
extern "C" __device__ __forceinline__ float val_iq4_nl(const uint8_t* base, int in) {
    const float d = __half2float(__ushort_as_half((unsigned short)(
        (uint16_t)base[0] | ((uint16_t)base[1] << 8))));
    const int q = (in < 16) ? (base[2 + in] & 0xF) : (base[2 + in - 16] >> 4);
    return d * (float)kvalues_iq4nl_lut[q];
}

// IQ2_XXS: SB 256, 66 bytes
extern "C" __device__ __forceinline__ float val_iq2_xxs(const uint8_t* base, int in) {
    const float d = __half2float(__ushort_as_half((unsigned short)(
        (uint16_t)base[0] | ((uint16_t)base[1] << 8))));
    const uint8_t* qs = base + 2;
    const int ib = in / 32, rem = in % 32;
    const int l = rem / 8, j = rem % 8;
    const uint32_t aux0 = (uint32_t)qs[ib * 8] | ((uint32_t)qs[ib * 8 + 1] << 8)
                        | ((uint32_t)qs[ib * 8 + 2] << 16) | ((uint32_t)qs[ib * 8 + 3] << 24);
    const uint32_t aux1 = (uint32_t)qs[ib * 8 + 4] | ((uint32_t)qs[ib * 8 + 5] << 8)
                        | ((uint32_t)qs[ib * 8 + 6] << 16) | ((uint32_t)qs[ib * 8 + 7] << 24);
    const float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
    const int idxg = (int)((aux0 >> (8 * l)) & 0xFF);
    const uint8_t signs = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
    const unsigned long long g = iq2xxs_grid[idxg];
    const int sg = (signs & kmask_iq2xs[j]) ? -1 : 1;
    const float gv = (float)(uint8_t)((g >> (8 * j)) & 0xFF);
    return db * gv * (float)sg;
}

// IQ2_XS: SB 256, 74 bytes
extern "C" __device__ __forceinline__ float val_iq2_xs(const uint8_t* base, int in) {
    const float d = __half2float(__ushort_as_half((unsigned short)(
        (uint16_t)base[0] | ((uint16_t)base[1] << 8))));
    const uint8_t* qs = base + 2;
    const uint8_t* scales = base + 66;
    const int ib = in / 32, rem = in % 32;
    const int l = rem / 8, j = rem % 8;
    const float db0 = d * (0.5f + (float)(scales[ib] & 0xF)) * 0.25f;
    const float db1 = d * (0.5f + (float)(scales[ib] >> 4)) * 0.25f;
    const uint16_t v = (uint16_t)qs[ib * 8 + l * 2] | ((uint16_t)qs[ib * 8 + l * 2 + 1] << 8);
    const int idxg = v & 511;
    const uint8_t signs = ksigns_iq2xs[v >> 9];
    const unsigned long long g = iq2xs_grid[idxg];
    const float db = (l < 2) ? db0 : db1;
    const int sg = (signs & kmask_iq2xs[j]) ? -1 : 1;
    const float gv = (float)(uint8_t)((g >> (8 * j)) & 0xFF);
    return db * gv * (float)sg;
}

// IQ2_S: SB 256, 82 bytes
extern "C" __device__ __forceinline__ float val_iq2_s(const uint8_t* base, int in) {
    const float d = __half2float(__ushort_as_half((unsigned short)(
        (uint16_t)base[0] | ((uint16_t)base[1] << 8))));
    const uint8_t* qs = base + 2;
    const uint8_t* signs = base + 34;
    const uint8_t* qh = base + 66;
    const uint8_t* scales = base + 74;
    const int ib = in / 32, rem = in % 32;
    const int l = rem / 8, j = rem % 8;
    const float db0 = d * (0.5f + (float)(scales[ib] & 0xF)) * 0.25f;
    const float db1 = d * (0.5f + (float)(scales[ib] >> 4)) * 0.25f;
    const int idxg = qs[ib * 4 + l] | (((uint32_t)qh[ib] << (8 - 2 * l)) & 0x300);
    const unsigned long long g = iq2s_grid[idxg];
    const float db = (l < 2) ? db0 : db1;
    const int sg = (signs[ib * 4 + l] & kmask_iq2xs[j]) ? -1 : 1;
    const float gv = (float)(uint8_t)((g >> (8 * j)) & 0xFF);
    return db * gv * (float)sg;
}

// IQ3_XXS: SB 256, 98 bytes
extern "C" __device__ __forceinline__ float val_iq3_xxs(const uint8_t* base, int in) {
    const float d = __half2float(__ushort_as_half((unsigned short)(
        (uint16_t)base[0] | ((uint16_t)base[1] << 8))));
    const uint8_t* qs = base + 2;
    const uint8_t* ss = base + 66;
    const int ib = in / 32, pos = in % 32;
    const int l = pos / 8, sub = pos % 8;
    const uint32_t aux = (uint32_t)ss[ib * 4] | ((uint32_t)ss[ib * 4 + 1] << 8)
                       | ((uint32_t)ss[ib * 4 + 2] << 16) | ((uint32_t)ss[ib * 4 + 3] << 24);
    const float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux >> (7 * l)) & 127];
    int jx; unsigned long long gx; int kmx;
    if (sub < 4) {
        jx = sub;
        gx = iq3xxs_grid[qs[ib * 8 + 2 * l]];
        kmx = kmask_iq2xs[jx];
    } else {
        jx = sub - 4;
        gx = iq3xxs_grid[qs[ib * 8 + 2 * l + 1]];
        kmx = kmask_iq2xs[jx + 4];
    }
    const int sg = (signs & kmx) ? -1 : 1;
    const float gv = (float)(uint8_t)((gx >> (8 * jx)) & 0xFF);
    return db * gv * (float)sg;
}

DEFINE_EXTRA_DECODE_KERNEL(paged_attention_decode_tq1_0_kernel, 8, 54, val_tq1_0)
DEFINE_EXTRA_DECODE_KERNEL(paged_attention_decode_tq2_0_kernel, 8, 66, val_tq2_0)
DEFINE_EXTRA_DECODE_KERNEL(paged_attention_decode_mxfp4_kernel, 5, 17, val_mxfp4)
DEFINE_EXTRA_DECODE_KERNEL(paged_attention_decode_iq4_nl_kernel, 5, 18, val_iq4_nl)
DEFINE_EXTRA_DECODE_KERNEL(paged_attention_decode_iq2_xxs_kernel, 8, 66, val_iq2_xxs)
DEFINE_EXTRA_DECODE_KERNEL(paged_attention_decode_iq2_xs_kernel, 8, 74, val_iq2_xs)
DEFINE_EXTRA_DECODE_KERNEL(paged_attention_decode_iq2_s_kernel, 8, 82, val_iq2_s)
DEFINE_EXTRA_DECODE_KERNEL(paged_attention_decode_iq3_xxs_kernel, 8, 98, val_iq3_xxs)

// ─── Q8_K prefill (request lane-b, desbloquea gate .q8_k del modelo Q8_K_XL)
// Layout SB 256 elems, 292 bytes: d f32@0 | qs[256] int8@4 | bsums[16] i16@260
extern "C" __device__ __forceinline__ float valpref_q8_k(const uint8_t* base, int in) {
    // f32 @base[0..4]: ensamblar byte-a-byte para alineación segura
    const float d = __uint_as_float((uint32_t)base[0] | ((uint32_t)base[1] << 8)
                  | ((uint32_t)base[2] << 16) | ((uint32_t)base[3] << 24));
    return d * (float)(int8_t)base[4 + in];
}
DEFINE_EXTRA_PREFILL_KERNEL(paged_attention_prefill_q8_k_kernel, 8, 292, valpref_q8_k)

// ─────────────────────────────────────────────────────────────────────────────
// Lane A: PREFILL UNIVERSAL — instanciaciones del macro para TODOS los
// formatos restantes. Las VALFN de decode ya son bit-exacto vs CPU ref
// (harness 21/21) y comparten firma (base,in) con el macro de prefill.
// K-quants: valfns portadas desde paged_attention.cu (q6k_val incluye fix
// j&1 de alternancia ql low/high).
// ─────────────────────────────────────────────────────────────────────────────

// ── helpers K-quants (portados; TU separada, sin colision) ──
__device__ __forceinline__ void lane_get_scale_min_k5(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) { d = q[j] & 63;     m = q[j + 4] & 63; }
    else       { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                 m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

__device__ __forceinline__ float lane_q2k_val(const uint8_t* blk, int w) {
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

__device__ __forceinline__ float lane_q3k_val(const uint8_t* blk, int w) {
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

__device__ __forceinline__ float lane_q4k_val(const uint8_t* blk, int w) {
    // Portado de gguf.zig dequantQ4_K: sub-grupos de 64 con escalas k5.
    const int j64 = w >> 6;                 // sub-grupo de 64 (0..3)
    const int l   = w & 63;
    const bool hi = l >= 32;
    const int li  = l & 31;
    const float dall = __half2float(*reinterpret_cast<const __half*>(blk));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(blk + 2));
    const uint8_t* scales = blk + 4;
    const uint8_t* qs     = blk + 16;
    int sd, sm;
    lane_get_scale_min_k5(2 * j64 + (hi ? 1 : 0), scales, sd, sm);
    // FIX: indice qs incluye el termino del sub-grupo (qs[g_idx*32 + l%32]).
    const uint8_t nib = hi ? (qs[32 * j64 + li] >> 4) : (qs[32 * j64 + li] & 0xF);
    return dall * (float)sd * (float)nib - dmin * (float)sm;
}

__device__ __forceinline__ float lane_q5k_val(const uint8_t* blk, int w) {
    const int il     = w >> 6;
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
    lane_get_scale_min_k5(2 * il + (hi ? 1 : 0), scales, sd, sm);
    const float d1 = dall * (float)sd;
    const float m1 = dmin * (float)sm;
    const uint8_t ql  = qs[32 * il + 2 * ir + parity];
    const uint8_t qb  = qh[2 * ir + parity];
    const uint8_t hmb = 1u << (2 * il + (hi ? 1 : 0));
    const int nib     = hi ? (ql >> 4) : (ql & 0xF);
    const int extra   = (qb & hmb) ? 16 : 0;
    return d1 * (float)(nib + extra) - m1;
}

__device__ __forceinline__ float lane_q6k_val(const uint8_t* blk, int w) {
    const int ip  = w >> 7;
    const int r   = w & 127;
    const int il  = r & 31;
    const int j   = r >> 5;
    const float d = __half2float(*reinterpret_cast<const __half*>(blk + 208));
    const int8_t sc = (int8_t)blk[192 + 8 * ip + il / 16 + 2 * j];
    // Fix j&1: cuadrantes alternan bloques ql low/high (no j>>1).
    // Fix: (j&1) alterna el bloque ql low/high; (j>>1) elige el nibble del
    // par de cuadrantes {q1,q2}=low / {q3,q4}=high (espejo de q6k_val verde).
    const uint8_t qlb = blk[64 * ip + il + (j & 1) * 32];
    const int nib     = (j >> 1) ? (qlb >> 4) : (qlb & 0xF);
    const uint8_t qhb = blk[128 + 32 * ip + il];
    const int packed  = nib | (((qhb >> (2 * j)) & 3) << 4);
    return d * (float)sc * (float)(packed - 32);
}

// ── formatos extra-cubin ──
DEFINE_EXTRA_PREFILL_KERNEL(paged_attention_prefill_q4_1_kernel,   5,  20, val_q4_1)
DEFINE_EXTRA_PREFILL_KERNEL(paged_attention_prefill_q5_0_kernel,   5,  22, val_q5_0)
DEFINE_EXTRA_PREFILL_KERNEL(paged_attention_prefill_q5_1_kernel,   5,  24, val_q5_1)
DEFINE_EXTRA_PREFILL_KERNEL(paged_attention_prefill_q8_1_kernel,   5,  36, val_q8_1)
DEFINE_EXTRA_PREFILL_KERNEL(paged_attention_prefill_iq3_s_kernel,  8, 110, val_iq3_s)
DEFINE_EXTRA_PREFILL_KERNEL(paged_attention_prefill_iq1_s_kernel,  8,  50, val_iq1_s)
DEFINE_EXTRA_PREFILL_KERNEL(paged_attention_prefill_iq1_m_kernel, 8, 56, val_iq1_m)
DEFINE_EXTRA_PREFILL_KERNEL(paged_attention_prefill_tq1_0_kernel,  8,  54, val_tq1_0)
DEFINE_EXTRA_PREFILL_KERNEL(paged_attention_prefill_tq2_0_kernel,  8,  66, val_tq2_0)
DEFINE_EXTRA_PREFILL_KERNEL(paged_attention_prefill_mxfp4_kernel,  5,  17, val_mxfp4)
DEFINE_EXTRA_PREFILL_KERNEL(paged_attention_prefill_iq4_nl_kernel, 5,  18, val_iq4_nl)
DEFINE_EXTRA_PREFILL_KERNEL(paged_attention_prefill_iq2_xxs_kernel,8,  66, val_iq2_xxs)
DEFINE_EXTRA_PREFILL_KERNEL(paged_attention_prefill_iq2_xs_kernel, 8,  74, val_iq2_xs)
DEFINE_EXTRA_PREFILL_KERNEL(paged_attention_prefill_iq2_s_kernel,  8,  82, val_iq2_s)
DEFINE_EXTRA_PREFILL_KERNEL(paged_attention_prefill_iq3_xxs_kernel,8,  98, val_iq3_xxs)

// ── K-quants (valfns portadas) ──
DEFINE_EXTRA_PREFILL_KERNEL(paged_attention_prefill_q2_k_kernel,   8,  84, lane_q2k_val)
DEFINE_EXTRA_PREFILL_KERNEL(paged_attention_prefill_q3_k_kernel,   8, 110, lane_q3k_val)
DEFINE_EXTRA_PREFILL_KERNEL(paged_attention_prefill_q4_k_kernel,   8, 144, lane_q4k_val)
DEFINE_EXTRA_PREFILL_KERNEL(paged_attention_prefill_q5_k_kernel,   8, 176, lane_q5k_val)
DEFINE_EXTRA_PREFILL_KERNEL(paged_attention_prefill_q6_k_kernel,   8, 210, lane_q6k_val)
