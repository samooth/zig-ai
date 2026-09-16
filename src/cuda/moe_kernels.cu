// moe_kernels.cu — Lane E (P2b/E3): ensure_experts_moe graph-safe.
//
// Puerto 1:1 del espejo CPU `ensureExpertsMirror` (src/moe/offload_cache.zig),
// que a su vez replica `_ensure_experts_hybrid_kernel` de FreeToken
// (offload_kernels.py:290-410). Toda la decisión es device-side y en punto
// fijo Q16 (cero floats) ⇒ capturable en CUDA graph con buffers de shape fija.
//
// DECISIÓN v1: UN hilo ejecuta el algoritmo completo (grid 1×1×1, block
// 1×1×1). El trabajo es O(num_fetch·C + E) entero (≤ ~10^5 ops con C~4k,
// fetch≤8), corre una vez por capa MoE y paso de decode, y el objetivo de esta
// versión es paridad bit-exacta con el espejo (oráculo de tests) + captura en
// grafo. Si algún día midiera, la paralelización debe preservar EXACTAMENTE:
//   - argmin/argmax primera-ocurrencia (tie-break índice menor),
//   - owner_active calculado UNA sola vez ANTES del bucle de fetch,
//   - scratch_usage con INF en protegidos/yas-elegidas (usage real intacto),
//   - el punto exacto de cada tienda (orden de reescrituras).
//
// Semántica Contrato 7 (PLAN_MAESTRO): expert_ids se REESCRIBE in-place
// (id de experto → slot del pool ó −1 = overflow al executor CPU de lane-f);
// fetch_frac_q16 > 0 sustituye el cap fijo por la fracción Q16
//   lo=(M*frac)>>16; elegir vecino que minimiza max(F·(65536−frac),(M−F)·frac)
// con frac=0 → cap fijo max_fetch (offload puro).

extern "C" __global__ void ensureExpertsMoeKernel(
    int* __restrict__ expert_ids,        // [num_active] in/out (Contrato 7)
    int* __restrict__ slot_for_id,       // [num_layers*num_experts], -1 libre
    int* __restrict__ id_of_slot,        // [cache_size], -1 vacío
    long long* __restrict__ usage,       // [cache_size]
    long long* __restrict__ step_ptr,    // [1]
    int* __restrict__ active_mask,       // [num_experts]
    int* __restrict__ evict_slots,       // [plan] slots elegidos este paso
    int* __restrict__ src_indices,       // [plan] expertos (layer-local)
    long long* __restrict__ num_indices, // [1] fetches capped
    long long* __restrict__ num_missing_full, // [1] misses pre-cap (stats)
    long long* __restrict__ expert_recency,   // [num_layers*num_experts], -1 init
    long long* __restrict__ scratch_usage,    // [cache_size] copia del paso
    long long* __restrict__ scratch_score,    // [num_experts] score del paso
    long long* __restrict__ stat_active_layer,     // [num_layers] += num_active
    long long* __restrict__ stat_missing_layer,    // [num_layers] += misses
    long long* __restrict__ stat_fetched_layer,    // [num_layers] += fetches
    long long* __restrict__ stat_steps_layer,      // [num_layers] += 1
    long long* __restrict__ decode_freq,           // [num_layers*num_experts] histograma routing
    const int layer_id,
    const int num_active,
    const int max_fetch,
    const unsigned int fetch_frac_q16,
    const int num_experts,
    const int cache_size)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const long long E = num_experts;
    const long long base = (long long)layer_id * E;
    const long long SENTINEL = -1152921504606846976LL; // −2^60, idéntico al espejo
    const long long UINF = 9223372036854775807LL;      // i64 max

    // ── step global ──
    const long long step = *step_ptr + 1;
    *step_ptr = step;

    // ── Fase 1: activos + misses ──
    for (int e = 0; e < num_experts; ++e) active_mask[e] = 0;
    for (int i = 0; i < num_active; ++i) {
        const int id = expert_ids[i];
        if (id >= 0 && (long long)id < E) {
            active_mask[id] = 1;
            decode_freq[base + id] += 1;
        }
    }
    long long num_missing = 0;
    for (int e = 0; e < num_experts; ++e) {
        const int sidx = slot_for_id[base + e];
        const bool is_active = active_mask[e] != 0;
        const bool is_missing = is_active && sidx == -1;
        if (is_missing) ++num_missing;
        scratch_score[e] = is_missing
            ? expert_recency[base + e] * E + (E - (long long)e - 1)
            : SENTINEL;
    }

    // Cap: fracción Q16 o fijo (parámetro NO especializado — Contrato 7).
    long long eff_max_fetch = max_fetch;
    if (fetch_frac_q16 > 0u) {
        const long long frac = (long long)fetch_frac_q16;
        const long long lo = (num_missing * frac) >> 16;
        const long long a_lo = lo * ((1LL << 16) - frac);
        const long long b_lo = (num_missing - lo) * frac;
        const long long cost_lo = a_lo > b_lo ? a_lo : b_lo;
        const long long a_hi = (lo + 1) * ((1LL << 16) - frac);
        const long long b_hi = (num_missing - lo - 1) * frac;
        const long long cost_hi = a_hi > b_hi ? a_hi : b_hi;
        eff_max_fetch = (cost_lo <= cost_hi) ? lo : lo + 1;
    }
    long long num_fetch = num_missing < eff_max_fetch ? num_missing : eff_max_fetch;

    *num_missing_full = num_missing;
    *num_indices = num_fetch;

    // Hits: bump usage.
    for (int e = 0; e < num_experts; ++e) {
        if (!active_mask[e]) continue;
        const int sidx = slot_for_id[base + e];
        if (sidx >= 0) usage[sidx] = step;
    }

    // ── Stats acumuladas device-side (E6): += capturable en grafo, lectura
    //    host diferida única. Histograma de routing ya acumulado en fase 1.
    stat_active_layer[layer_id] += num_active;
    stat_missing_layer[layer_id] += num_missing;
    stat_fetched_layer[layer_id] += num_fetch;
    stat_steps_layer[layer_id] += 1;

    // ── Fase 2: evict argmin(usage) protegiendo activos; selección de misses
    //    por score estricto recency·E+(E−1−id) ──
    if (num_fetch > 0) {
        // owner_active UNA vez antes del bucle (las asignaciones nuevas no
        // re-protegen: su usage=step ya las excluye del argmin).
        for (int c = 0; c < cache_size; ++c) {
            const long long oid = id_of_slot[c];
            bool owned = false;
            if (oid >= 0 && oid >= base && oid < base + E)
                owned = active_mask[(int)(oid - base)] != 0;
            scratch_usage[c] = owned ? UINF : usage[c];
        }

        for (long long i = 0; i < num_fetch; ++i) {
            // víctima = primera ocurrencia del mínimo.
            int victim = 0;
            long long best = UINF;
            for (int c = 0; c < cache_size; ++c) {
                if (scratch_usage[c] < best) { best = scratch_usage[c]; victim = c; }
            }
            const long long old_id = id_of_slot[victim];
            if (old_id >= 0) slot_for_id[old_id] = -1;

            // ganador = argmax(score), primera ocurrencia.
            int winner = 0;
            long long bs = (-9223372036854775807LL - 1); // i64 min
            for (int e = 0; e < num_experts; ++e) {
                if (scratch_score[e] > bs) { bs = scratch_score[e]; winner = e; }
            }

            id_of_slot[victim] = (int)(base + winner);
            slot_for_id[base + winner] = victim;
            usage[victim] = step;
            evict_slots[i] = victim;
            src_indices[i] = winner;
            scratch_usage[victim] = UINF;
            scratch_score[winner] = SENTINEL;
        }
    }

    // ── Fase 3: rewrite ids → slot ó −1; bump recencia de activos ──
    for (int i = 0; i < num_active; ++i) {
        const int e = expert_ids[i];
        if (e >= 0 && (long long)e < E)
            expert_ids[i] = slot_for_id[base + e];
        else
            expert_ids[i] = -1;
    }
    for (int e = 0; e < num_experts; ++e) {
        if (active_mask[e]) expert_recency[base + e] = step;
    }
}

// ── E4: gather zero-copy multi-banco fusionado ────────────────────────────
// Estilo fast_index_copy_multi de FreeToken (fast_index_copy.cuh:487-512):
// copia las MISMAS filas (evict_slots←src_indices) en TODOS los bancos en UN
// solo launch. Las fuentes son VAs de memoria host PINNED (visibles desde
// device vía UVA en Linux — Contrato 5/D2 las registra; el interim usa
// cuMemAllocHost+pinnedAlloc que ya es page-locked). Restricción del layout:
// feat_bytes[b] múltiplo de 16 (unidades uint4); se valida en el wrapper.
//
// Grid = blocks_per_bank × num_banks, block = kThreads: el cuello es PCIe, no
// SM (~4096 hilos/banco es la rodilla medida por FreeToken; 16×256=4096).
constexpr int kGatherThreads = 256;

extern "C" __global__ void gatherMissingRowsKernel(
    const unsigned long long* __restrict__ dst_ptrs,   // [num_banks] VRAM
    const unsigned long long* __restrict__ src_ptrs,   // [num_banks] host VA
    const unsigned long long* __restrict__ feat_bytes, // [num_banks] %16==0
    const int* __restrict__ dst_rows,                  // [plan] = evict_slots
    const int* __restrict__ src_rows,                  // [plan] = src_indices
    const long long* __restrict__ num_indices_ptr,     // [1]
    const int num_banks,
    const int blocks_per_bank)
{
    const int b = blockIdx.x / blocks_per_bank;
    if (b >= num_banks) return;
    const int blk = blockIdx.x % blocks_per_bank;
    const unsigned char* src = (const unsigned char*)(size_t)src_ptrs[b];
    unsigned char* dst = (unsigned char*)(size_t)dst_ptrs[b];
    const long long feat = (long long)feat_bytes[b];
    const long long n = *num_indices_ptr;
    if (n <= 0) return;
    const long long units = feat >> 4;
    const long long total = n * units;
    const long long stride = (long long)blocks_per_bank * kGatherThreads;
    for (long long u = (long long)blk * kGatherThreads + threadIdx.x; u < total;
         u += stride) {
        const long long row = u / units;
        const long long col = (u - row * units) << 4;
        const long long dr = dst_rows[row];
        const long long sr = src_rows[row];
        const uint4 v = *(const uint4*)(src + sr * feat + col);
        *(uint4*)(dst + dr * feat + col) = v;
    }
}


// ── E5: router GEMV + softmax + top-k (bs=1 decode) ───────────────────────
// Logits en shared memory (smem dinámico = E·4B): fase paralela de dots +
// fase serial de softmax/top-k con máscara real (tie-break id menor,
// primera ocurrencia). Pesos re-normalizados entre los seleccionados
// (convención qwen-moe); variantes por familia (sigmoid gpt-oss, etc.) se
// añaden al validar contra el modelo real.
extern "C" __global__ void routerTopKKernel(
    const float* __restrict__ x,            // [n_embd]
    const unsigned char* __restrict__ router_bytes, // [E][n_embd] f32 GGUF
    float* __restrict__ weights_out,        // [top_k]
    int* __restrict__ ids_out,              // [top_k]
    const int n_embd,
    const int num_experts,
    const int top_k)
{
    extern __shared__ float logits[]; // [num_experts]
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        const float* row = (const float*)(router_bytes + (size_t)e * n_embd * 4);
        float acc = 0.0f;
        for (int i = 0; i < n_embd; ++i) acc += row[i] * x[i];
        logits[e] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float mx = -3.402823466e38f;
        for (int e = 0; e < num_experts; ++e) mx = logits[e] > mx ? logits[e] : mx;
        float denom = 0.0f;
        for (int e = 0; e < num_experts; ++e) denom += __expf(logits[e] - mx);
        const float inv = 1.0f / denom;
        for (int k = 0; k < top_k; ++k) {
            float best_p = -1.0f;
            int best_e = -1;
            for (int e = 0; e < num_experts; ++e) {
                if (logits[e] <= -3.30e38f) continue; // ya elegido (marcado)
                const float p = __expf(logits[e] - mx) * inv;
                if (p > best_p) { best_p = p; best_e = e; }
            }
            if (best_e < 0) best_e = k < num_experts ? k : 0;
            ids_out[k] = best_e;
            weights_out[k] = best_p;
            logits[best_e] = -3.402823466e38f;
        }
        float s = 0.0f;
        for (int k = 0; k < top_k; ++k) s += weights_out[k];
        const float sinv = s > 0.0f ? 1.0f / s : 1.0f;
        for (int k = 0; k < top_k; ++k) weights_out[k] *= sinv;
    }
}

// ── E5: out += alpha * in (sum-reduce ponderado del MoE) ──────────────────
extern "C" __global__ void axpyMulKernel(
    const float* __restrict__ in_,
    float* __restrict__ out,
    const float alpha,
    const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += alpha * in_[i];
}

// ── E5: copia device→device de buffers f32 (resultado MoE al buffer caller)
extern "C" __global__ void copyF32Kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
