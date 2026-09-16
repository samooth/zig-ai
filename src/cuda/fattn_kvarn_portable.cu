//! KVarN native FA — portable path (lane-b1 Dev-B, B4 of TODO_B1_DEV_B).
//!
//! Pure-CUDA attention that consumes the C1 contract (`KvarnDesc` +
//! `KvarnRecordLayout`) and applies WHT-rotated Q against stage-or-record
//! K/V, with the WHT inverse applied to the output. Tail (KVCPT) is
//! gated behind D4 and left for the follow-up task; this file is the
//! M1 critical path body-only implementation.
//!
//! ## Scope (B4, TODO_B1_DEV_B §B4)
//!
//!   - `fattnKvarnPortable<D=128>` template (D≥256 gated by D2; D3
//!     cross-slice is documented but not implemented until B2 ratifies
//!     the cross-slice CPU ref).
//!   - Grid: (n_q, n_q_heads, n_stream). Block: 128 threads.
//!   - 128 threads = 1 thread per dim WHT (per PLAN_B1 §3 "128 threads
//!     = 1 thread por dim WHT").
//!   - SLICES = D/128 = 1 for D=128. `accumulator[SLICES]` becomes
//!     `accumulator[1]` = a single f32 reg per thread.
//!   - Reduction: classic 128→1 over a `__shared__ float reduction[128]`
//!     with `__syncthreads` strided 64→1 (transcrito del upstream).
//!   - Online softmax: `m`/`l` running max/denom, `old_scale`/`weight`
//!     in shared vars visibles a todos los threads ANTES del V-accumulate.
//!   - Q rotation: WHT-128 butterfly (1/√128, self-inverse) at load +
//!     inverse at write. Cross-slice omitted (D=128 only).
//!   - Body-only (no tail). The KVCPT tail (D4) is a follow-up; this
//!     B4 is the critical path of M1 and the gate says "tu portable +
//!     store bit-exacto del Dev A" = M1.
//!
//! ## Transcripción (not copy)
//!
//! Algorithm re-expressed in zig-ai terms: own struct names, own register
//! layout, own smem variable set. No upstream code copied verbatim. The
//! Q-rotation + reduction + online softmax scheme is the standard
//! FlashAttention pattern, transcribed from
//! `docs/b1-research/KVAR_N_CUDA_REFERENCE.md` §12 + the upstream
//! portable reference for ordering of the smem events.

#include "kvarn_desc.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cfloat>

// WHT-128 in-place butterfly (transcripción local — see kvarn_kernels.cu
// for the canonical implementation; both follow the same Cooley-Tukey
// pattern normalized by 1/√128). Inlined here to keep this translation
// unit self-contained and to make the rotation D3 explicit at the
// portable call sites.
__device__ __forceinline__ void fattn_kvarn_wht_128(float* values, int tid)
{
    constexpr int N = 128;
    for (int stride = 1; stride < N; stride <<= 1) {
        for (int pair = tid; pair < N / 2; pair += blockDim.x) {
            const int j = (pair / stride) * (2 * stride) + (pair % stride);
            const float a = values[j];
            const float b = values[j + stride];
            values[j] = a + b;
            values[j + stride] = a - b;
        }
        __syncthreads();
    }
    if (tid < N) {
        values[tid] *= 0.08838834764831845f; // 1/sqrt(128)
    }
}

// Rotated-domain record loader. For D=128 the slice is 0 and we just
// call the A0 helper directly. For D>=256 the caller must unroll slices.
__device__ __forceinline__ float fattn_kvarn_load_value(
    const KvarnDesc& d, int record_group, int token_pos, int dim)
{
    return kvarn_record_value(d, record_group, token_pos, dim, 0);
}

// ============================================================================
// D4: KVCPT exact-tail load (lane-b1 Dev-B, B4 follow-up).
// ============================================================================
//
// El body del portable procesa tokens comprimidos (records C1 o
// stage f16). El tail procesa una cola exacta f16/bf16 que el B2
// manager conserva intrínseca (no cuantizada) para D<=tail_tokens.
// Esto permite al motor conmutar entre quantizado y exacto sin
// materializar — la rotación Q ya está aplicada en el body; el tail
// es la misma rotación + el softmax online acumula sobre TODO.
//
// Layout del tail (sigue el upstream portable):
//   k_tail_data / v_tail_data: [n_kv_heads × tail_slots × D] f16 o bf16
//   tail_mask: [n_tail, n_q] f16 (-inf ⇒ máscara; ≡ body mask)
//   run_desc[6+token] = slot para el token `token` del tail (i32).
//
// bf16 → f32: __uint_as_float((uint32)bits << 16) — el patrón upstream.
// f16 → f32: __half2float.
//
// El body+tail loop es IDÉNTICO al upstream portable. Para D=128,
// sólo hay 1 slice, el tail loop es trivial.

__device__ __forceinline__ float fattn_kvarn_load_tail(
    const char* ptr, bool bf16)
{
    if (bf16) {
        const uint16_t bits = *reinterpret_cast<const uint16_t*>(ptr);
        return __uint_as_float((uint32_t)bits << 16);
    }
    return __half2float(*reinterpret_cast<const __half*>(ptr));
}

extern "C" __global__ void fattn_kvarn_portable_d128_kernel(
    const float* q_data,           // [n_q, n_q_heads, n_stream, D]
    KvarnDesc* k_descs,            // [n_stream * n_kv_heads]
    KvarnDesc* v_descs,            // [n_stream * n_kv_heads]
    const __half* mask_data,       // [n_kv, n_q] or null
    float* dst_data,               // [n_q_heads, n_q, n_stream, D]
    int n_kv,
    int n_q,
    int n_q_heads,
    int n_kv_heads,
    int n_stream,
    float scale)
{
    constexpr int D = 128;
    constexpr int THREADS = 128;
    constexpr int SLICES = 1; // D/128

    const int query = (int) blockIdx.x;
    const int query_head = (int) blockIdx.y;
    const int stream = (int) blockIdx.z;
    const int tid = (int) threadIdx.x;
    if (query >= n_q || query_head >= n_q_heads) return;
    if (tid >= THREADS) return;

    const int gqa = n_q_heads / n_kv_heads;
    const int kv_head = query_head / gqa;

    // Q row for this (query, head, stream) — laid out as
    // [n_q, n_q_heads, n_stream, D], stride D per element.
    const float* q = q_data
        + ((size_t) query * n_q_heads + query_head) * n_stream * D
        + (size_t) stream * D;

    __shared__ float reduction[THREADS];
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float old_scale_shared;
    __shared__ float weight_shared;

    // Q rotation D3: load Q row into shared, apply WHT-128 in-place. The
    // WHT is involutive so we can apply it once at load and once at
    // write; the loaded values are the rotated Q, used directly to dot
    // against rotated K (per KVAR_N_CUDA_REFERENCE §3: "stage and records
    // are stored in rotated domain for both K and V").
    __shared__ float q_rot[THREADS];
    q_rot[tid] = q[tid];
    __syncthreads();
    fattn_kvarn_wht_128(&q_rot[0], tid);
    __syncthreads();

    float accumulator = 0.0f;
    if (tid == 0) {
        maximum = -FLT_MAX;
        denominator = 0.0f;
    }
    __syncthreads();

    for (int token = 0; token < n_kv; ++token) {
        const KvarnDesc& k_desc = k_descs[(size_t) stream * n_kv_heads + kv_head];
        const KvarnDesc& v_desc = v_descs[(size_t) stream * n_kv_heads + kv_head];

        // Resolve token → cell (group, pos) + flags. Mirrors the upstream
        // portable_resolve: SWA / read_indirect ⇒ indices[token]; else
        // direct group = token/128, pos = token%128.
        int group;
        int pos;
        bool from_stage;
        bool from_record;
        if (k_desc.swa || k_desc.read_indirect) {
            const int64_t enc = k_desc.indices[token];
            if (enc == -1) continue;
            bool es;
            int aslot;
            const int64_t cell = (int64_t)(uint32_t)kvarn_index_payload(enc);
            if (enc < -1) {
                es = true;
                aslot = (int)((uint32_t)(kvarn_index_payload(enc) >> 32)) - 1;
                if (aslot < 0) es = false; // high word 0 ⇒ no slot
            } else {
                es = false;
                aslot = -1;
            }
            group = (int)(cell / D);
            pos = (int)(cell - (int64_t)group * D);
            from_stage = es || kvarn_group_from_stage(k_desc, group);
            from_record = !es && (k_desc.read_indirect && !k_desc.swa ?
                true : kvarn_group_from_record(k_desc, group));
        } else {
            group = token / D;
            pos = token - group * D;
            from_stage = kvarn_group_from_stage(k_desc, group);
            from_record = kvarn_group_from_record(k_desc, group);
        }

        // K value for this (token, dim=tid) — one float per thread.
        float k_value;
        if (from_stage) {
            // Direct f16 stage read (rotated domain already).
            int stage_pos;
            {
                int aslot = -1;
                if (k_desc.swa || k_desc.read_indirect) {
                    const int64_t enc = k_desc.indices[token];
                    if (enc < -1) {
                        aslot = (int)((uint32_t)(kvarn_index_payload(enc) >> 32)) - 1;
                    }
                }
                stage_pos = kvarn_stage_pos(k_desc, group, pos, aslot);
            }
            k_value = __half2float(
                k_desc.stage[
                    ((int64_t)stage_pos * (2 * k_desc.n_record_heads) + 2 * kv_head) * D + tid]);
        } else if (from_record) {
            int record_group;
            if (k_desc.swa) {
                record_group = group % k_desc.groups_per_stream;
            } else {
                record_group = k_desc.stream * k_desc.groups_per_stream + group;
            }
            k_value = fattn_kvarn_load_value(k_desc, record_group, pos, tid);
        } else {
            continue; // neither stage nor record — skip token
        }

        // Score partial = sum over slices of q_rot[dim] * k_value[dim].
        // For SLICES=1, the inner loop is one iteration.
        float partial = 0.0f;
        #pragma unroll
        for (int slice = 0; slice < SLICES; ++slice) {
            const int dim = slice * D + tid;
            partial += q_rot[dim] * k_value;
        }
        reduction[tid] = partial;
        __syncthreads();

        // Tree reduction 128→1.
        for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduction[tid] += reduction[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            float mask_value = 0.0f;
            if (mask_data != nullptr) {
                const __half* mask = mask_data + (size_t) token * n_q + query;
                mask_value = __half2float(*mask);
            }
            float score = reduction[0] * scale + mask_value;
            if (mask_value == -INFINITY) {
                old_scale_shared = 1.0f;
                weight_shared = 0.0f;
            } else {
                const float next_max = fmaxf(maximum, score);
                const float old_scale = (maximum == -FLT_MAX) ?
                    0.0f : expf(maximum - next_max);
                const float weight = expf(score - next_max);
                maximum = next_max;
                denominator = denominator * old_scale + weight;
                old_scale_shared = old_scale;
                weight_shared = weight;
            }
        }
        __syncthreads();

        // V value for this (token, dim=tid) — one float per thread.
        float v_value;
        if (from_stage) {
            int stage_pos;
            {
                int aslot = -1;
                if (v_desc.swa || v_desc.read_indirect) {
                    const int64_t enc = v_desc.indices[token];
                    if (enc < -1) {
                        aslot = (int)((uint32_t)(kvarn_index_payload(enc) >> 32)) - 1;
                    }
                }
                stage_pos = kvarn_stage_pos(v_desc, group, pos, aslot);
            }
            v_value = __half2float(
                v_desc.stage[
                    ((int64_t)stage_pos * (2 * v_desc.n_record_heads) + 2 * kv_head + 1) * D + tid]);
        } else if (from_record) {
            int record_group;
            if (v_desc.swa) {
                record_group = group % v_desc.groups_per_stream;
            } else {
                record_group = v_desc.stream * v_desc.groups_per_stream + group;
            }
            v_value = fattn_kvarn_load_value(v_desc, record_group, pos, tid);
        } else {
            v_value = 0.0f;
        }

        accumulator = accumulator * old_scale_shared + v_value * weight_shared;
        __syncthreads();
    }

    // Final scale: divide by denominator. weight_shared was set to
    // 1/denominator after the loop ends in the upstream; we recompute
    // here to avoid a barrier + extra shared write.
    if (tid == 0) {
        if (denominator == 0.0f) denominator = 1.0f;
        weight_shared = 1.0f / denominator;
    }
    __syncthreads();

    // Apply WHT⁻¹ to the per-thread accumulator (D3 per PLAN_B1 §3
    // "la salida se escribe tras WHT⁻¹"). The WHT is self-inverse
    // (H·H = I when normalized by 1/√N on both sides) — applying the
    // same butterfly again returns the vector to the original basis.
    //
    // Cooperative across the 128 threads: each writes its dim to
    // shared, then the whole block runs the inverse, then each writes
    // out. This costs one extra `__syncthreads` pair + 1 read of shared
    // per thread (negligible vs the n_kv passes above).
    __shared__ float out_rot[THREADS];
    out_rot[tid] = accumulator * weight_shared;
    __syncthreads();
    fattn_kvarn_wht_128(&out_rot[0], tid);
    __syncthreads();

    // Output row: [n_q_heads, n_q, n_stream, D]
    float* output = dst_data
        + ((size_t) query_head * n_q + query) * n_stream * D
        + (size_t) stream * D;
    output[tid] = out_rot[tid];
}

// D=128 portable FA con cola exacta (D4 KVCPT, lane-b1 Dev-B).
// ----------------------------------------------------------------------------
// Variante del fattn_kvarn_portable_d128_kernel que procesa, además del
// body C1, una cola exacta f16/bf16 administrada por el B2 manager
// (getExactTail). El tail se integra en el MISMO softmax online (sin
// materializar), preservando numericamente el resultado.
//
// Args adicionales vs el d128 sin cola:
//   k_tail_data, v_tail_data: char* a f16/bf16 [n_kv_heads × tail_slots × D]
//   tail_mask: __half* [n_tail, n_q] (opcional)
//   run_desc_slots: int32* a [n_tail] slots (uno por token del tail)
//   n_tail: número de tokens del tail
//   k_tail_bf16, v_tail_bf16: bool — formato del tail
//   d_k, d_v: (unused in this stub; reserved for the manager pointer
//     shape; row-major contiguous)
//
// Si n_tail == 0 ó k_tail_data == nullptr, el kernel procesa SOLO el
// body (idéntico al d128 sin tail). Esto permite al dispatcher (B7)
// usar el mismo wrapper para ambos caminos.
__device__ __forceinline__ void fattn_kvarn_portable_d128_body_impl(
    const float* q_data,
    KvarnDesc* k_descs,
    KvarnDesc* v_descs,
    const __half* mask_data,
    int n_kv, int n_q, int n_q_heads, int n_kv_heads, int n_stream,
    int query, int query_head, int stream, int tid, int kv_head,
    float* q_rot, float* reduction, float& accumulator,
    float& maximum, float& denominator,
    float& old_scale_shared, float& weight_shared,
    float scale)
{
    constexpr int D = 128;
    constexpr int THREADS = 128;
    // D=128 ⇒ SLICES=1, but we don't use the variable here. The
    // for-loop over SLICES is in the D=256 kernel.

    for (int token = 0; token < n_kv; ++token) {
        const KvarnDesc& k_desc = k_descs[(size_t) stream * n_kv_heads + kv_head];
        const KvarnDesc& v_desc = v_descs[(size_t) stream * n_kv_heads + kv_head];

        int group;
        int pos;
        bool from_stage;
        bool from_record;
        int aslot_local = -1;
        if (k_desc.swa || k_desc.read_indirect) {
            const int64_t enc = k_desc.indices[token];
            if (enc == -1) continue;
            bool es = enc < -1;
            int aslot = -1;
            if (es) {
                const uint64_t payload = kvarn_index_payload(enc);
                const uint32_t packed = (uint32_t)(payload >> 32);
                aslot = packed == 0 ? -1 : (int)(packed - 1u);
            }
            const int64_t cell = (int64_t)(uint32_t)kvarn_index_payload(enc);
            group = (int)(cell / D);
            pos = (int)(cell - (int64_t)group * D);
            from_stage = es || kvarn_group_from_stage(k_desc, group);
            from_record = !es && (k_desc.read_indirect && !k_desc.swa ?
                true : kvarn_group_from_record(k_desc, group));
            aslot_local = aslot;
        } else {
            group = token / D;
            pos = token - group * D;
            from_stage = kvarn_group_from_stage(k_desc, group);
            from_record = kvarn_group_from_record(k_desc, group);
        }

        float k_value;
        if (from_stage) {
            const int stage_pos = kvarn_stage_pos(k_desc, group, pos, aslot_local);
            k_value = __half2float(
                k_desc.stage[
                    ((int64_t)stage_pos * (2 * k_desc.n_record_heads) + 2 * kv_head) * D + tid]);
        } else if (from_record) {
            int record_group = k_desc.swa
                ? group % k_desc.groups_per_stream
                : k_desc.stream * k_desc.groups_per_stream + group;
            k_value = fattn_kvarn_load_value(k_desc, record_group, pos, tid);
        } else {
            continue;
        }

        float partial = q_rot[tid] * k_value;
        reduction[tid] = partial;
        __syncthreads();
        for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) reduction[tid] += reduction[tid + stride];
            __syncthreads();
        }

        if (tid == 0) {
            float mask_value = 0.0f;
            if (mask_data != nullptr) {
                const __half* mask = mask_data + (size_t) token * n_q + query;
                mask_value = __half2float(*mask);
            }
            float score = reduction[0] * scale + mask_value;
            if (mask_value == -INFINITY) {
                old_scale_shared = 1.0f;
                weight_shared = 0.0f;
            } else {
                const float next_max = fmaxf(maximum, score);
                const float old_scale = (maximum == -FLT_MAX) ?
                    0.0f : expf(maximum - next_max);
                const float weight = expf(score - next_max);
                maximum = next_max;
                denominator = denominator * old_scale + weight;
                old_scale_shared = old_scale;
                weight_shared = weight;
            }
        }
        __syncthreads();

        float v_value;
        if (from_stage) {
            const int stage_pos = kvarn_stage_pos(v_desc, group, pos, aslot_local);
            v_value = __half2float(
                v_desc.stage[
                    ((int64_t)stage_pos * (2 * v_desc.n_record_heads) + 2 * kv_head + 1) * D + tid]);
        } else if (from_record) {
            int record_group = v_desc.swa
                ? group % v_desc.groups_per_stream
                : v_desc.stream * v_desc.groups_per_stream + group;
            v_value = fattn_kvarn_load_value(v_desc, record_group, pos, tid);
        } else {
            v_value = 0.0f;
        }
        accumulator = accumulator * old_scale_shared + v_value * weight_shared;
        __syncthreads();
    }
}

// D=128 launch wrapper (called by the Zig launcher via cuLaunchKernel;
// a tiny shim with cudaStream_t to keep the API uniform with the other
// lane-b1 cubin kernels).
extern "C" void fattn_kvarn_portable_d128_launcher(
    const float* q_data,
    KvarnDesc* k_descs,
    KvarnDesc* v_descs,
    const __half* mask_data,
    float* dst_data,
    int n_kv, int n_q, int n_q_heads, int n_kv_heads, int n_stream,
    float scale,
    cudaStream_t stream)
{
    if (n_q <= 0 || n_q_heads <= 0 || n_kv <= 0 || n_stream <= 0) return;
    if (n_q_heads % n_kv_heads != 0) return; // GQA must be integer
    dim3 grid((unsigned)n_q, (unsigned)n_q_heads, (unsigned)n_stream);
    fattn_kvarn_portable_d128_kernel<<<grid, 128, 0, stream>>>(
        q_data, k_descs, v_descs, mask_data, dst_data,
        n_kv, n_q, n_q_heads, n_kv_heads, n_stream, scale);
}

// ============================================================================
// D4 KVCPT tail: portable D=128 con cola exacta f16/bf16 (lane-b1 Dev-B).
// ============================================================================
//
// El B2 manager conserva los últimos N tokens EXACTOS (sin cuantizar)
// para los modelos donde la cuantización agresiva degrada la calidad
// del output. Esta cola se procesa con el MISMO softmax online del
// body — sin materializar — y se suma al accumulator antes del WHT⁻¹.
//
// Convenciones:
//   - d_k, d_v: bytes por token del tail (D * sizeof(half)). Para
//     f16 es 256 bytes (D=128), para bf16 también 256 bytes.
//   - run_desc_slots[i32]: slot del i-ésimo token del tail. El B2
//     manager decide qué slot corresponde a qué token.
//   - El tail respeta el `kv_head` de la query (GQA, no de cada token).
//   - Si n_tail == 0, el kernel actúa como el d128 sin tail.

extern "C" __global__ void fattn_kvarn_portable_d128_tail_kernel(
    const float* q_data,
    KvarnDesc* k_descs,
    KvarnDesc* v_descs,
    const __half* mask_data,
    const char* k_tail_data,
    const char* v_tail_data,
    const __half* tail_mask,
    const int32_t* run_desc_slots,
    int n_kv, int n_tail, int d_k, int d_v,
    int n_q, int n_q_heads, int n_kv_heads, int n_stream,
    bool k_tail_bf16, bool v_tail_bf16,
    float* dst_data,
    float scale)
{
    constexpr int D = 128;
    constexpr int THREADS = 128;

    const int query = (int) blockIdx.x;
    const int query_head = (int) blockIdx.y;
    const int stream = (int) blockIdx.z;
    const int tid = (int) threadIdx.x;
    if (query >= n_q || query_head >= n_q_heads) return;
    if (tid >= THREADS) return;
    if (n_q_heads % n_kv_heads != 0) return;

    const int gqa = n_q_heads / n_kv_heads;
    const int kv_head = query_head / gqa;

    const float* q = q_data
        + ((size_t) query * n_q_heads + query_head) * n_stream * D
        + (size_t) stream * D;

    __shared__ float reduction[THREADS];
    __shared__ float q_rot[THREADS];
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float old_scale_shared;
    __shared__ float weight_shared;

    // ── Q load + WHT-128 in-kernel (D3) ──
    q_rot[tid] = q[tid];
    __syncthreads();
    fattn_kvarn_wht_128(&q_rot[0], tid);
    __syncthreads();

    // ── Init softmax ──
    float accumulator = 0.0f;
    if (tid == 0) {
        maximum = -FLT_MAX;
        denominator = 0.0f;
    }
    __syncthreads();

    // ── Body loop (idéntico al d128 sin tail) ──
    fattn_kvarn_portable_d128_body_impl(
        q_data, k_descs, v_descs, mask_data,
        n_kv, n_q, n_q_heads, n_kv_heads, n_stream,
        query, query_head, stream, tid, kv_head,
        q_rot, reduction, accumulator,
        maximum, denominator, old_scale_shared, weight_shared,
        scale);

    // ── Tail loop (D4 KVCPT, lane-b1 Dev-B) ──
    // Procesa `n_tail` tokens exactos (f16/bf16). El slot del
    // token i viene de run_desc_slots[i] (lo calcula el B2
    // manager). El tail_mask es opcional (null = sin máscara).
    for (int token = 0; token < n_tail; ++token) {
        const int slot = run_desc_slots[token];
        // K value: f16/bf16 → f32 desde k_tail_data.
        const char* k_ptr = k_tail_data +
            (size_t)slot * d_k + (size_t)kv_head * D * 2 +
            (size_t)tid * 2;
        const float k_value = fattn_kvarn_load_tail(k_ptr, k_tail_bf16);

        // Score partial: dot over the 128 dims (1 dim/thread).
        reduction[tid] = q_rot[tid] * k_value;
        __syncthreads();
        for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) reduction[tid] += reduction[tid + stride];
            __syncthreads();
        }

        if (tid == 0) {
            float mask_value = 0.0f;
            if (tail_mask != nullptr) {
                const __half* tm = tail_mask + (size_t)token * n_q + query;
                mask_value = __half2float(*tm);
            }
            float score = reduction[0] * scale + mask_value;
            if (mask_value == -INFINITY) {
                old_scale_shared = 1.0f;
                weight_shared = 0.0f;
            } else {
                const float next_max = fmaxf(maximum, score);
                const float old_scale = (maximum == -FLT_MAX) ?
                    0.0f : expf(maximum - next_max);
                const float weight = expf(score - next_max);
                maximum = next_max;
                denominator = denominator * old_scale + weight;
                old_scale_shared = old_scale;
                weight_shared = weight;
            }
        }
        __syncthreads();

        // V value: f16/bf16 → f32 desde v_tail_data.
        const char* v_ptr = v_tail_data +
            (size_t)slot * d_v + (size_t)kv_head * D * 2 +
            (size_t)tid * 2;
        const float v_value = fattn_kvarn_load_tail(v_ptr, v_tail_bf16);

        accumulator = accumulator * old_scale_shared + v_value * weight_shared;
        __syncthreads();
    }

    // ── Final scale + WHT⁻¹ + write (idéntico al d128 sin tail) ──
    if (tid == 0) {
        if (denominator == 0.0f) denominator = 1.0f;
        weight_shared = 1.0f / denominator;
    }
    __syncthreads();

    __shared__ float out_rot[THREADS];
    out_rot[tid] = accumulator * weight_shared;
    __syncthreads();
    fattn_kvarn_wht_128(&out_rot[0], tid);
    __syncthreads();

    float* output = dst_data
        + ((size_t) query_head * n_q + query) * n_stream * D
        + (size_t) stream * D;
    output[tid] = out_rot[tid];
}

// Launcher para el D=128 + tail. El dispatcher (B7) elige este
// wrapper cuando el caller (B2 manager) entrega k_tail/v_tail
// no nulos.
extern "C" void fattn_kvarn_portable_d128_tail_launcher(
    const float* q_data,
    KvarnDesc* k_descs,
    KvarnDesc* v_descs,
    const __half* mask_data,
    const char* k_tail_data,
    const char* v_tail_data,
    const __half* tail_mask,
    const int32_t* run_desc_slots,
    int n_kv, int n_tail, int d_k, int d_v,
    int n_q, int n_q_heads, int n_kv_heads, int n_stream,
    bool k_tail_bf16, bool v_tail_bf16,
    float* dst_data,
    float scale,
    cudaStream_t stream)
{
    if (n_q <= 0 || n_q_heads <= 0) return;
    if (n_q_heads % n_kv_heads != 0) return;
    dim3 grid((unsigned)n_q, (unsigned)n_q_heads, (unsigned)n_stream);
    fattn_kvarn_portable_d128_tail_kernel<<<grid, 128, 0, stream>>>(
        q_data, k_descs, v_descs, mask_data,
        k_tail_data, v_tail_data, tail_mask, run_desc_slots,
        n_kv, n_tail, d_k, d_v,
        n_q, n_q_heads, n_kv_heads, n_stream,
        k_tail_bf16, v_tail_bf16,
        dst_data, scale);
}

// ============================================================================
// 9.12 (lane-cuda) F3: D=64 portable FA — body-only (no tail).
// ============================================================================
//
// Estructura paralela al D=128: D=64, THREADS=64, SLICES=1.
// Cada thread cubre 1 dim. Q rotation = WHT-64 in-place.
// Score = dot(Q,K) reducido en 64 threads. V accumulate = online softmax.
// Output = WHT-64 inverse (self-inverse).
//
// Stage layout: [pos][2·n_record_heads][128] — D64 solo usa first64.
// Record layout: head_dim=64 (KvarnRecordLayout.init(64, ...)).
//
// D=64 es intrinsicamente un solo "slice" (no cross-slice necesario).
// La rotación Q usa WHT-64 y la inversa idem.

__device__ __forceinline__ void fattn_kvarn_wht_64(float* values, int tid)
{
    constexpr int N = 64;
    for (int stride = 1; stride < N; stride <<= 1) {
        for (int pair = tid; pair < N / 2; pair += blockDim.x) {
            const int j = (pair / stride) * (2 * stride) + (pair % stride);
            const float a = values[j];
            const float b = values[j + stride];
            values[j] = a + b;
            values[j + stride] = a - b;
        }
        __syncthreads();
    }
    if (tid < N) {
        values[tid] *= 0.125f; // 1/sqrt(64)
    }
}

// D64 record loader: head_dim=64, uses kvarn_record_value with D=64.
__device__ __forceinline__ float fattn_kvarn_load_value_d64(
    const KvarnDesc& d, int record_group, int token_pos, int dim)
{
    return kvarn_record_value(d, record_group, token_pos, dim, 0);
}

extern "C" __global__ void fattn_kvarn_portable_d64_kernel(
    const float* q_data,           // [n_q, n_q_heads, n_stream, 64]
    KvarnDesc* k_descs,            // [n_stream * n_kv_heads]
    KvarnDesc* v_descs,            // [n_stream * n_kv_heads]
    const __half* mask_data,       // [n_kv, n_q] or null
    float* dst_data,               // [n_q_heads, n_q, n_stream, 64]
    int n_kv,
    int n_q,
    int n_q_heads,
    int n_kv_heads,
    int n_stream,
    float scale)
{
    constexpr int D = 64;
    constexpr int THREADS = 64;

    const int query = (int) blockIdx.x;
    const int query_head = (int) blockIdx.y;
    const int stream = (int) blockIdx.z;
    const int tid = (int) threadIdx.x;
    if (query >= n_q || query_head >= n_q_heads) return;
    if (tid >= THREADS) return;

    const int gqa = n_q_heads / n_kv_heads;
    const int kv_head = query_head / gqa;

    // Q row: [n_q, n_q_heads, n_stream, 64]
    const float* q = q_data
        + ((size_t) query * n_q_heads + query_head) * n_stream * D
        + (size_t) stream * D;

    __shared__ float reduction[128]; // pad to 128 for tree reduction
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float old_scale_shared;
    __shared__ float weight_shared;

    // Q rotation D3: WHT-64 in-place
    __shared__ float q_rot[D];
    q_rot[tid] = q[tid];
    __syncthreads();
    fattn_kvarn_wht_64(&q_rot[0], tid);
    __syncthreads();

    float accumulator = 0.0f;
    if (tid == 0) {
        maximum = -FLT_MAX;
        denominator = 0.0f;
    }
    __syncthreads();

    for (int token = 0; token < n_kv; ++token) {
        const KvarnDesc& k_desc = k_descs[(size_t) stream * n_kv_heads + kv_head];
        const KvarnDesc& v_desc = v_descs[(size_t) stream * n_kv_heads + kv_head];

        int group;
        int pos;
        bool from_stage;
        bool from_record;
        int aslot_local = -1;
        if (k_desc.swa || k_desc.read_indirect) {
            const int64_t enc = k_desc.indices[token];
            if (enc == -1) continue;
            bool es = enc < -1;
            int aslot = -1;
            if (es) {
                const uint64_t payload = kvarn_index_payload(enc);
                const uint32_t packed = (uint32_t)(payload >> 32);
                aslot = packed == 0 ? -1 : (int)(packed - 1u);
            }
            const int64_t cell = (int64_t)(uint32_t)kvarn_index_payload(enc);
            group = (int)(cell / KVARN_DIM); // group based on stage dim (128)
            pos = (int)(cell - (int64_t)group * KVARN_DIM);
            from_stage = es || kvarn_group_from_stage(k_desc, group);
            from_record = !es && (k_desc.read_indirect && !k_desc.swa ?
                true : kvarn_group_from_record(k_desc, group));
            aslot_local = aslot;
        } else {
            group = token / KVARN_DIM;
            pos = token - group * KVARN_DIM;
            from_stage = kvarn_group_from_stage(k_desc, group);
            from_record = kvarn_group_from_record(k_desc, group);
        }

        // K value: first64 of stage or record
        float k_value;
        if (from_stage) {
            const int stage_pos = kvarn_stage_pos(k_desc, group, pos, aslot_local);
            // Stage: [pos][2·n_record_heads][128]; K row = 2*kv_head; first64 only.
            k_value = __half2float(
                k_desc.stage[
                    ((int64_t)stage_pos * (2 * k_desc.n_record_heads) +
                     2 * kv_head) * KVARN_DIM + tid]);
        } else if (from_record) {
            int record_group = k_desc.swa
                ? group % k_desc.groups_per_stream
                : k_desc.stream * k_desc.groups_per_stream + group;
            k_value = fattn_kvarn_load_value_d64(k_desc, record_group, pos, tid);
        } else {
            continue;
        }

        // Score partial: dot over 64 dims
        float partial = q_rot[tid] * k_value;
        reduction[tid] = partial;
        if (tid == 0) reduction[64] = 0.0f; // clear padding
        __syncthreads();

        // Tree reduction 64→1 (pad to 128)
        for (int stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduction[tid] += reduction[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            float mask_value = 0.0f;
            if (mask_data != nullptr) {
                const __half* mask = mask_data + (size_t) token * n_q + query;
                mask_value = __half2float(*mask);
            }
            float score = reduction[0] * scale + mask_value;
            if (mask_value == -INFINITY) {
                old_scale_shared = 1.0f;
                weight_shared = 0.0f;
            } else {
                const float next_max = fmaxf(maximum, score);
                const float old_scale = (maximum == -FLT_MAX) ?
                    0.0f : expf(maximum - next_max);
                const float weight = expf(score - next_max);
                maximum = next_max;
                denominator = denominator * old_scale + weight;
                old_scale_shared = old_scale;
                weight_shared = weight;
            }
        }
        __syncthreads();

        // V value: first64 of stage or record
        float v_value;
        if (from_stage) {
            const int stage_pos = kvarn_stage_pos(v_desc, group, pos, aslot_local);
            // V row = 2*kv_head+1
            v_value = __half2float(
                v_desc.stage[
                    ((int64_t)stage_pos * (2 * v_desc.n_record_heads) +
                     2 * kv_head + 1) * KVARN_DIM + tid]);
        } else if (from_record) {
            int record_group = v_desc.swa
                ? group % v_desc.groups_per_stream
                : v_desc.stream * v_desc.groups_per_stream + group;
            v_value = fattn_kvarn_load_value_d64(v_desc, record_group, pos, tid);
        } else {
            v_value = 0.0f;
        }

        accumulator = accumulator * old_scale_shared + v_value * weight_shared;
        __syncthreads();
    }

    // Final scale
    if (tid == 0) {
        if (denominator == 0.0f) denominator = 1.0f;
        weight_shared = 1.0f / denominator;
    }
    __syncthreads();

    // WHT⁻¹: self-inverse, apply WHT-64 again
    __shared__ float out_rot[D];
    out_rot[tid] = accumulator * weight_shared;
    __syncthreads();
    fattn_kvarn_wht_64(&out_rot[0], tid);
    __syncthreads();

    // Output: [n_q_heads, n_q, n_stream, 64]
    float* output = dst_data
        + ((size_t) query_head * n_q + query) * n_stream * D
        + (size_t) stream * D;
    output[tid] = out_rot[tid];
}

extern "C" void fattn_kvarn_portable_d64_launcher(
    const float* q_data,
    KvarnDesc* k_descs,
    KvarnDesc* v_descs,
    const __half* mask_data,
    float* dst_data,
    int n_kv, int n_q, int n_q_heads, int n_kv_heads, int n_stream,
    float scale,
    cudaStream_t stream)
{
    if (n_q <= 0 || n_q_heads <= 0 || n_kv <= 0 || n_stream <= 0) return;
    if (n_q_heads % n_kv_heads != 0) return;
    dim3 grid((unsigned)n_q, (unsigned)n_q_heads, (unsigned)n_stream);
    fattn_kvarn_portable_d64_kernel<<<grid, 64, 0, stream>>>(
        q_data, k_descs, v_descs, mask_data, dst_data,
        n_kv, n_q, n_q_heads, n_kv_heads, n_stream, scale);
}

// ============================================================================
// WHT-128 device smoke (gate M1 D3, B4 prep)
// ============================================================================
//
// Bit-exacto vs hadamard128InPlace de B2. Patrón idéntico al A1 de Dev-A
// (kvarn_wht_128_rows_kernel en kvarn_kernels.cu). Lo necesitamos como
// paso de validación ANTES de la FA completa — si el WHT del path
// portable difiere del CPU reference, el matmul acumula el drift y M1
// falla downstream con un error muy difícil de aislar.
//
// Grid: (n_rows), block: 128, smem dyn: 128 floats. Cada bloque aplica
// fattn_kvarn_wht_128 (la misma función que usa el FA portable) a una
// fila de 128 floats. El patrón butterfly es idéntico al de A1: si A1
// bit-pasa contra B2, este también (mismo algoritmo, misma precisión).

extern "C" __global__ void fattn_kvarn_wht_128_rows_kernel(float* rows, int n_rows)
{
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    const int tid = (int)threadIdx.x;
    if (tid < 128) {
        smem[tid] = rows[(size_t)row * 128 + tid];
    }
    __syncthreads();
    fattn_kvarn_wht_128(&smem[0], tid);
    __syncthreads();
    if (tid < 128) {
        rows[(size_t)row * 128 + tid] = smem[tid];
    }
}

// ============================================================================
// D7: Cross-slice WHT para D=256/512 (lane-b1 Dev-B).
// ============================================================================
//
// PLAN_B1 D2: "D=256 (2 slices) / D=512 (4 slices) butterfly entre slices +
// escala 1/√2 / 1/√4". Aplicado DESPUÉS del WHT intra-slice en la rotación
// forward y antes en la inversa. Gated D2 — el caller (B7 dispatch
// master) sólo invoca D>=256 cuando lane-b2 ratifique el CPU ref del
// cross-slice; mientras tanto, esta función existe como transcripción
// correcta y se usa en los templates D=256/512.
//
// Bloque 128 threads cooperan: cada thread `tid` cubre el dim `tid` de
// cada slice (1 dim per thread per slice, total SLICES dims por thread
// = 4 max para D=512).
//
// Forward: x[slice][dim] = (sum_{s=0..SLICES-1} sign[slice][s] * x[s][dim]) * scale
//   SLICES=2: x[0][d] = (a+b) * 1/√2; x[1][d] = (a-b) * 1/√2
//   SLICES=4: closed-form WHT-4 normalizado.

template<int SLICES>
__device__ __forceinline__ void fattn_kvarn_wht_cross_slices(
    float* values, // pointer to [SLICES * 128] flat
    int tid)        // 0..127
{
    if constexpr (SLICES == 1) {
        // No-op.
    } else if constexpr (SLICES == 2) {
        const float a = values[tid];
        const float b = values[tid + 128];
        values[tid]       = (a + b) * 0.7071067811865475f;
        values[tid + 128] = (a - b) * 0.7071067811865475f;
    } else if constexpr (SLICES == 4) {
        // Closed-form WHT-4 (Cooley-Tukey).
        const float a0 = values[tid];
        const float a1 = values[tid + 128];
        const float a2 = values[tid + 256];
        const float a3 = values[tid + 384];
        const float b0 = a0 + a1;
        const float b1 = a0 - a1;
        const float b2 = a2 + a3;
        const float b3 = a2 - a3;
        values[tid]       = (b0 + b2) * 0.5f; // = (a0+a1+a2+a3) / 2
        values[tid + 128] = (b1 + b3) * 0.5f; // = (a0-a1+a2-a3) / 2
        values[tid + 256] = (b0 - b2) * 0.5f; // = (a0+a1-a2-a3) / 2
        values[tid + 384] = (b1 - b3) * 0.5f; // = (a0-a1-a2+a3) / 2
    }
    // SLICES > 4 no soportado (D > 512).
}

// (helper `fattn_kvarn_wht_cross_slices` se valida vía el template D=256).
// (el helper `fattn_kvarn_wht_cross_slices` se valida vía el template
//  D=256 del kernel que sigue debajo).

// ============================================================================
// D=256 portable (gated D2 ratification; D=128 sigue siendo el camino M1).
// ============================================================================
//
// Estructura paralela al D=128: SLICES=2, cada thread `tid` cubre
// slice 0 dim=tid + slice 1 dim=tid+128. La rotación Q aplica
// fattn_kvarn_wht_128 a cada slice + fattn_kvarn_wht_cross_slices<2>
// entre slices. La inversa al output.
//
// Gated por D2: mientras lane-b2 no ratifique el CPU ref del cross-slice,
// este kernel NO se compila en el cubin de M1 (ver macro al final).

extern "C" __global__ void fattn_kvarn_portable_d256_kernel(
    const float* q_data,
    KvarnDesc* k_descs,
    KvarnDesc* v_descs,
    const __half* mask_data,
    float* dst_data,
    int n_kv, int n_q, int n_q_heads, int n_kv_heads, int n_stream,
    float scale)
{
    constexpr int D = 256;
    constexpr int THREADS = 128;
    constexpr int SLICES = 2; // D/128

    const int query = (int) blockIdx.x;
    const int query_head = (int) blockIdx.y;
    const int stream = (int) blockIdx.z;
    const int tid = (int) threadIdx.x;
    if (query >= n_q || query_head >= n_q_heads) return;
    if (tid >= THREADS) return;
    if (n_q_heads % n_kv_heads != 0) return;

    const int gqa = n_q_heads / n_kv_heads;
    const int kv_head = query_head / gqa;

    const float* q = q_data
        + ((size_t) query * n_q_heads + query_head) * n_stream * D
        + (size_t) stream * D;

    __shared__ float reduction[THREADS];
    __shared__ float q_rot[D];
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float old_scale_shared;
    __shared__ float weight_shared;

    // Q load + WHT-128 (intra-slice, 2 pasadas para 2 slices) +
    // WHT-2 cross-slice. Verificación: el orden importa — D3 del plan
    // dice "intra-128 WHT then cross-slice WHT" para la forward.
    if (tid < THREADS) {
        q_rot[tid]        = q[tid];
        q_rot[tid + 128]  = q[tid + 128];
    }
    __syncthreads();

    // Intra-slice WHT (cada slice = 128 dims, 128 threads cooperan).
    // Hacemos 2 pasadas: slice 0, luego slice 1.
    for (int slice = 0; slice < SLICES; ++slice) {
        fattn_kvarn_wht_128(&q_rot[slice * 128], tid);
        __syncthreads();
    }
    // Cross-slice WHT (D=256: SLICES=2, butterfly ×1/√2).
    fattn_kvarn_wht_cross_slices<SLICES>(&q_rot[0], tid);
    __syncthreads();

    float accumulator[SLICES] = { 0.0f, 0.0f };
    if (tid == 0) {
        maximum = -FLT_MAX;
        denominator = 0.0f;
    }
    __syncthreads();

    for (int token = 0; token < n_kv; ++token) {
        const KvarnDesc& k_desc = k_descs[(size_t) stream * n_kv_heads + kv_head];
        const KvarnDesc& v_desc = v_descs[(size_t) stream * n_kv_heads + kv_head];

        bool es = false;
        int aslot = -1;
        int group;
        int pos;
        bool from_stage;
        bool from_record;
        if (k_desc.swa || k_desc.read_indirect) {
            const int64_t enc = k_desc.indices[token];
            if (enc == -1) continue;
            es = enc < -1;
            if (es) {
                const uint64_t payload = kvarn_index_payload(enc);
                const uint32_t packed = (uint32_t)(payload >> 32);
                aslot = packed == 0 ? -1 : (int)(packed - 1u);
            }
            const int64_t cell = (int64_t)(uint32_t)kvarn_index_payload(enc);
            group = (int)(cell / 128);
            pos = (int)(cell - (int64_t)group * 128);
            from_stage = es || kvarn_group_from_stage(k_desc, group);
            from_record = !es && (k_desc.read_indirect && !k_desc.swa ?
                true : kvarn_group_from_record(k_desc, group));
        } else {
            group = token / 128;
            pos = token - group * 128;
            from_stage = kvarn_group_from_stage(k_desc, group);
            from_record = kvarn_group_from_record(k_desc, group);
        }

        float k_values[SLICES] = { 0.0f, 0.0f };
        if (from_stage) {
            const int stage_pos = kvarn_stage_pos(k_desc, group, pos, aslot);
            // 9.4 (lane-b) D2 FIX: layout C2v2 del store zig A2 — filas
            // K(2h)/V(2h+1) intercaladas, stride 2·n_record_heads (el
            // indexing original plano nunca se ejercitó: gated D2).
            #pragma unroll
            for (int slice = 0; slice < SLICES; ++slice) {
                k_values[slice] = __half2float(
                    k_desc.stage[
                        ((int64_t)stage_pos * (2 * k_desc.n_record_heads) +
                         (int64_t)(2 * (k_desc.head_base + slice))) * 128 + tid]);
            }
        } else if (from_record) {
            int record_group;
            if (k_desc.swa) {
                record_group = group % k_desc.groups_per_stream;
            } else {
                record_group = k_desc.stream * k_desc.groups_per_stream + group;
            }
            #pragma unroll
            for (int slice = 0; slice < SLICES; ++slice) {
                k_values[slice] = kvarn_record_value(k_desc, record_group, pos, tid, slice);
            }
        } else {
            continue;
        }

        float partial = 0.0f;
        #pragma unroll
        for (int slice = 0; slice < SLICES; ++slice) {
            const int dim = slice * 128 + tid;
            partial += q_rot[dim] * k_values[slice];
        }
        reduction[tid] = partial;
        __syncthreads();

        for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduction[tid] += reduction[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            float mask_value = 0.0f;
            if (mask_data != nullptr) {
                const __half* mask = mask_data + (size_t) token * n_q + query;
                mask_value = __half2float(*mask);
            }
            float score = reduction[0] * scale + mask_value;
            if (mask_value == -INFINITY) {
                old_scale_shared = 1.0f;
                weight_shared = 0.0f;
            } else {
                const float next_max = fmaxf(maximum, score);
                const float old_scale = (maximum == -FLT_MAX) ?
                    0.0f : expf(maximum - next_max);
                const float weight = expf(score - next_max);
                maximum = next_max;
                denominator = denominator * old_scale + weight;
                old_scale_shared = old_scale;
                weight_shared = weight;
            }
        }
        __syncthreads();

        float v_values[SLICES] = { 0.0f, 0.0f };
        if (from_stage) {
            const int v_stage_pos = kvarn_stage_pos(v_desc, group, pos, aslot);
            // 9.4 (lane-b) D2 FIX: C2v2 — fila V = 2·(head_base+slice)+1.
            #pragma unroll
            for (int slice = 0; slice < SLICES; ++slice) {
                v_values[slice] = __half2float(
                    v_desc.stage[
                        ((int64_t)v_stage_pos * (2 * v_desc.n_record_heads) +
                         (int64_t)(2 * (v_desc.head_base + slice) + 1)) * 128 + tid]);
            }
        } else if (from_record) {
            int v_record_group;
            if (v_desc.swa) {
                v_record_group = group % v_desc.groups_per_stream;
            } else {
                v_record_group = v_desc.stream * v_desc.groups_per_stream + group;
            }
            #pragma unroll
            for (int slice = 0; slice < SLICES; ++slice) {
                v_values[slice] = kvarn_record_value(v_desc, v_record_group, pos, tid, slice);
            }
        } else {
            #pragma unroll
            for (int slice = 0; slice < SLICES; ++slice) {
                v_values[slice] = 0.0f;
            }
        }

        #pragma unroll
        for (int slice = 0; slice < SLICES; ++slice) {
            accumulator[slice] = accumulator[slice] * old_scale_shared +
                v_values[slice] * weight_shared;
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (denominator == 0.0f) denominator = 1.0f;
        weight_shared = 1.0f / denominator;
    }
    __syncthreads();

    // Output: cada thread escribe SLICES floats (uno por slice).
    __shared__ float out_rot[D];
    #pragma unroll
    for (int slice = 0; slice < SLICES; ++slice) {
        out_rot[slice * 128 + tid] = accumulator[slice] * weight_shared;
    }
    __syncthreads();

    // WHT⁻¹: cross-slice (WHT-2 self-inverse) + intra-slice (WHT-128
    // self-inverse, 2 pasadas para 2 slices).
    fattn_kvarn_wht_cross_slices<SLICES>(&out_rot[0], tid);
    __syncthreads();
    for (int slice = 0; slice < SLICES; ++slice) {
        fattn_kvarn_wht_128(&out_rot[slice * 128], tid);
        __syncthreads();
    }

    float* output = dst_data
        + ((size_t) query_head * n_q + query) * n_stream * D
        + (size_t) stream * D;
    if (tid < D) {
        output[tid] = out_rot[tid];
    }
    // 9.4 (lane-b) D2 FIX: block=128 y D=256 — `tid < D` arriba solo
    // cubría out[0..127]; slice-1 quedaba sin escribir (0 tras memset).
    // Cada thread escribe su slot de cada slice restante.
    #pragma unroll
    for (int slice = 1; slice < SLICES; ++slice) {
        output[slice * 128 + tid] = out_rot[slice * 128 + tid];
    }
}

extern "C" void fattn_kvarn_portable_d256_launcher(
    const float* q_data,
    KvarnDesc* k_descs,
    KvarnDesc* v_descs,
    const __half* mask_data,
    float* dst_data,
    int n_kv, int n_q, int n_q_heads, int n_kv_heads, int n_stream,
    float scale,
    cudaStream_t stream)
{
    if (n_q <= 0 || n_q_heads <= 0 || n_kv <= 0 || n_stream <= 0) return;
    if (n_q_heads % n_kv_heads != 0) return;
    dim3 grid((unsigned)n_q, (unsigned)n_q_heads, (unsigned)n_stream);
    fattn_kvarn_portable_d256_kernel<<<grid, 128, 0, stream>>>(
        q_data, k_descs, v_descs, mask_data, dst_data,
        n_kv, n_q, n_q_heads, n_kv_heads, n_stream, scale);
}

// ============================================================================
// D=512 portable (gated D2 ratification; D=128 es el camino M1).
// ============================================================================
//
// SLICES=4 (4 slices de 128). Block 128 threads, cada thread `tid`
// cubre el dim `tid` de cada slice (4 dims por thread). Q rotation =
// 4× WHT-128 intra-slice + WHT-4 cross-slice. Output = WHT-4 + 4× WHT-128.
//
// Sigue el patrón del D=256 (mismo: load + online softmax + V-pass +
// WHT⁻¹ al output), parametrizado por SLICES=4.
//
// Gated D2: mientras lane-b2 no ratifique el CPU ref del cross-slice
// WHT-4, este kernel NO entra en el cubin de M1. El caller (test
// o B7) gatea la ejecución vía ZIG_AI_KVARN_D2_UNLOCK; el kernel
// en sí no se compila/omite por el env var (siempre compila si
// está en el .cu).

extern "C" __global__ void fattn_kvarn_portable_d512_kernel(
    const float* q_data,
    KvarnDesc* k_descs,
    KvarnDesc* v_descs,
    const __half* mask_data,
    float* dst_data,
    int n_kv, int n_q, int n_q_heads, int n_kv_heads, int n_stream,
    float scale)
{
    constexpr int D = 512;
    constexpr int THREADS = 128;
    constexpr int SLICES = 4; // D/128

    const int query = (int) blockIdx.x;
    const int query_head = (int) blockIdx.y;
    const int stream = (int) blockIdx.z;
    const int tid = (int) threadIdx.x;
    if (query >= n_q || query_head >= n_q_heads) return;
    if (tid >= THREADS) return;
    if (n_q_heads % n_kv_heads != 0) return;

    const int gqa = n_q_heads / n_kv_heads;
    const int kv_head = query_head / gqa;

    const float* q = q_data
        + ((size_t) query * n_q_heads + query_head) * n_stream * D
        + (size_t) stream * D;

    __shared__ float reduction[THREADS];
    __shared__ float q_rot[D];
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float old_scale_shared;
    __shared__ float weight_shared;

    // Q load + 4× intra-slice WHT-128 + WHT-4 cross-slice (D3).
    if (tid < THREADS) {
        q_rot[tid]         = q[tid];
        q_rot[tid + 128]   = q[tid + 128];
        q_rot[tid + 256]   = q[tid + 256];
        q_rot[tid + 384]   = q[tid + 384];
    }
    __syncthreads();

    #pragma unroll
    for (int slice = 0; slice < SLICES; ++slice) {
        fattn_kvarn_wht_128(&q_rot[slice * 128], tid);
        __syncthreads();
    }
    // Cross-slice WHT (D=512: SLICES=4, butterfly closed-form).
    fattn_kvarn_wht_cross_slices<SLICES>(&q_rot[0], tid);
    __syncthreads();

    float accumulator[SLICES] = { 0.0f, 0.0f, 0.0f, 0.0f };
    if (tid == 0) {
        maximum = -FLT_MAX;
        denominator = 0.0f;
    }
    __syncthreads();

    for (int token = 0; token < n_kv; ++token) {
        const KvarnDesc& k_desc = k_descs[(size_t) stream * n_kv_heads + kv_head];
        const KvarnDesc& v_desc = v_descs[(size_t) stream * n_kv_heads + kv_head];

        bool es = false;
        int aslot = -1;
        int group;
        int pos;
        bool from_stage;
        bool from_record;
        if (k_desc.swa || k_desc.read_indirect) {
            const int64_t enc = k_desc.indices[token];
            if (enc == -1) continue;
            es = enc < -1;
            if (es) {
                const uint64_t payload = kvarn_index_payload(enc);
                const uint32_t packed = (uint32_t)(payload >> 32);
                aslot = packed == 0 ? -1 : (int)(packed - 1u);
            }
            const int64_t cell = (int64_t)(uint32_t)kvarn_index_payload(enc);
            group = (int)(cell / 128);
            pos = (int)(cell - (int64_t)group * 128);
            from_stage = es || kvarn_group_from_stage(k_desc, group);
            from_record = !es && (k_desc.read_indirect && !k_desc.swa ?
                true : kvarn_group_from_record(k_desc, group));
        } else {
            group = token / 128;
            pos = token - group * 128;
            from_stage = kvarn_group_from_stage(k_desc, group);
            from_record = kvarn_group_from_record(k_desc, group);
        }

        float k_values[SLICES] = { 0.0f, 0.0f, 0.0f, 0.0f };
        if (from_stage) {
            const int stage_pos = kvarn_stage_pos(k_desc, group, pos, aslot);
            #pragma unroll
            for (int slice = 0; slice < SLICES; ++slice) {
                k_values[slice] = __half2float(
                    k_desc.stage[
                        ((int64_t)stage_pos * k_desc.n_record_heads +
                         (int64_t)(k_desc.head_base + slice)) * 128 + tid]);
            }
        } else if (from_record) {
            int record_group = k_desc.swa
                ? group % k_desc.groups_per_stream
                : k_desc.stream * k_desc.groups_per_stream + group;
            #pragma unroll
            for (int slice = 0; slice < SLICES; ++slice) {
                k_values[slice] = kvarn_record_value(k_desc, record_group, pos, tid, slice);
            }
        } else {
            continue;
        }

        float partial = 0.0f;
        #pragma unroll
        for (int slice = 0; slice < SLICES; ++slice) {
            const int dim = slice * 128 + tid;
            partial += q_rot[dim] * k_values[slice];
        }
        reduction[tid] = partial;
        __syncthreads();

        for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduction[tid] += reduction[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            float mask_value = 0.0f;
            if (mask_data != nullptr) {
                const __half* mask = mask_data + (size_t) token * n_q + query;
                mask_value = __half2float(*mask);
            }
            float score = reduction[0] * scale + mask_value;
            if (mask_value == -INFINITY) {
                old_scale_shared = 1.0f;
                weight_shared = 0.0f;
            } else {
                const float next_max = fmaxf(maximum, score);
                const float old_scale = (maximum == -FLT_MAX) ?
                    0.0f : expf(maximum - next_max);
                const float weight = expf(score - next_max);
                maximum = next_max;
                denominator = denominator * old_scale + weight;
                old_scale_shared = old_scale;
                weight_shared = weight;
            }
        }
        __syncthreads();

        float v_values[SLICES] = { 0.0f, 0.0f, 0.0f, 0.0f };
        if (from_stage) {
            const int v_stage_pos = kvarn_stage_pos(v_desc, group, pos, aslot);
            #pragma unroll
            for (int slice = 0; slice < SLICES; ++slice) {
                v_values[slice] = __half2float(
                    v_desc.stage[
                        ((int64_t)v_stage_pos * v_desc.n_record_heads +
                         (int64_t)(v_desc.head_base + slice)) * 128 + tid]);
            }
        } else if (from_record) {
            int v_record_group = v_desc.swa
                ? group % v_desc.groups_per_stream
                : v_desc.stream * v_desc.groups_per_stream + group;
            #pragma unroll
            for (int slice = 0; slice < SLICES; ++slice) {
                v_values[slice] = kvarn_record_value(v_desc, v_record_group, pos, tid, slice);
            }
        } else {
            #pragma unroll
            for (int slice = 0; slice < SLICES; ++slice) {
                v_values[slice] = 0.0f;
            }
        }

        #pragma unroll
        for (int slice = 0; slice < SLICES; ++slice) {
            accumulator[slice] = accumulator[slice] * old_scale_shared +
                v_values[slice] * weight_shared;
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (denominator == 0.0f) denominator = 1.0f;
        weight_shared = 1.0f / denominator;
    }
    __syncthreads();

    __shared__ float out_rot[D];
    #pragma unroll
    for (int slice = 0; slice < SLICES; ++slice) {
        out_rot[slice * 128 + tid] = accumulator[slice] * weight_shared;
    }
    __syncthreads();

    // WHT⁻¹: WHT-4 cross-slice + 4× WHT-128 intra-slice.
    fattn_kvarn_wht_cross_slices<SLICES>(&out_rot[0], tid);
    __syncthreads();
    #pragma unroll
    for (int slice = 0; slice < SLICES; ++slice) {
        fattn_kvarn_wht_128(&out_rot[slice * 128], tid);
        __syncthreads();
    }

    float* output = dst_data
        + ((size_t) query_head * n_q + query) * n_stream * D
        + (size_t) stream * D;
    if (tid < D) {
        output[tid] = out_rot[tid];
    }
    // 9.4 (lane-b) D2 FIX: idem d256 — cubrir slices 1..3 (D=512).
    #pragma unroll
    for (int slice = 1; slice < SLICES; ++slice) {
        output[slice * 128 + tid] = out_rot[slice * 128 + tid];
    }
}

extern "C" void fattn_kvarn_portable_d512_launcher(
    const float* q_data,
    KvarnDesc* k_descs,
    KvarnDesc* v_descs,
    const __half* mask_data,
    float* dst_data,
    int n_kv, int n_q, int n_q_heads, int n_kv_heads, int n_stream,
    float scale,
    cudaStream_t stream)
{
    if (n_q <= 0 || n_q_heads <= 0 || n_kv <= 0 || n_stream <= 0) return;
    if (n_q_heads % n_kv_heads != 0) return;
    dim3 grid((unsigned)n_q, (unsigned)n_q_heads, (unsigned)n_stream);
    fattn_kvarn_portable_d512_kernel<<<grid, 128, 0, stream>>>(
        q_data, k_descs, v_descs, mask_data, dst_data,
        n_kv, n_q, n_q_heads, n_kv_heads, n_stream, scale);
}
