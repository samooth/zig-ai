# PagedAttention Integration TODO

> **Objetivo**: Integrar PagedAttention (vLLM-style) en zig-ai para soportar contextos largos, batching continuo, y offloading de KV-cache a CPU.

---

## 📋 Estado Actual

| Componente | Estado |
|---|---|
| `src/paged_attention/` | ✅ Implementación completa (CPU reference + CUDA stubs) |
| `src/paged_attention/attention.zig` | ✅ CPU reference con online softmax |
| `src/paged_attention/attention.zig` | ✅ `decode` / `prefill` / `decodeBatch` |
| `src/paged_attention/scheduler.zig` | ✅ Scheduler con preemption + prefix cache |
| `src/paged_attention/allocator.zig` | ✅ Block allocator con COW + swap a CPU |
| `src/paged_attention/block_table.zig` | ✅ Mapeo lógico→físico |
| `src/paged_attention/prefix_cache.zig` | ✅ Prefix cache por hash |
| `src/paged_attention/*.cu` | ✅ Kernels CUDA (flash_attention_v2 + paged kernels) |
| `src/kv_cache/` | ⚠️ Implementación legacy (contigua) - **reemplazar** |
| `src/transformer/hybrid_attn.zig` | ✅ AttentionLayer con KV-cache contiguo |
| `src/main.zig` | ⚠️ Usa `KVCacheManager` legacy + `TransformerLayer` |

---

## 🎯 Objetivo

Reemplazar `KVCacheManager` + `TransformerLayer` attention path por **PagedAttention** completo:
- KV-cache paginado (bloques de 16 tokens)
- Batching continuo con scheduler (prefill + decode interleaved)
- Prefix caching (deduplicación de prefijos comunes)
- CPU offloading de bloques fríos
- Prefix caching para prompts compartidos
- Soporte futuro: chunked prefill, chunked decode

---

## 📋 TODO Detallado

---

## 🟢 FASE 1: Integración Básica (1-2 sem)

### 1.1 Integrar `PagedKVCache` en `runHybridInference`

**Archivo**: `src/main.zig` → `runHybridInference`

```zig
// Reemplazar KVCacheManager por PagedKVCache
const paged = @import("paged_attention");

var paged_config = pa.PagedConfig{
    .block_size = 16,
    .num_blocks = 8192,           // 8192 * 16 = 131k tokens ≈ 8GB @ f16
    .head_dim = head_dim,
    .num_kv_heads = cfg.head_count_kv,
    .num_q_heads = cfg.head_count,
    .dtype = .f16,
    .enable_prefix_cache = true,
    .enable_cpu_offload = true,
    .max_seq_len = max_seq_len,
    .max_batch_size = 32,
};

var paged_kv = try pa.PagedKVCache.init(allocator, paged_config);
defer paged_kv.deinit();
```

**Tareas:**
- [ ] Crear `PagedKVCache` en `runHybridInference`
- [ ] Pasar `paged_kv` a `HybridLayer.init` (nuevo parámetro)
- [ ] Eliminar `KVCacheManager` legacy

---

### 1.2 Actualizar `HybridLayer` para usar `PagedKVCache`

**Archivo**: `src/transformer/hybrid_attn.zig`

Cambios en `AttentionLayer`:
```zig
// ANTES: k_cache / v_cache contiguos
k_cache: []f32, v_cache: []f32, cache_len: usize

// DESPUÉS: PagedKVCache + block_table
paged_kv: *PagedKVCache,
block_table: *BlockTable,
sequence_id: u64,
```

**Cambios en `AttentionLayer.init`:**
```zig
pub fn init(
    allocator: std.mem.Allocator,
    layer_idx: usize,
    params: HybridAttnParams,
    backend: matmul.Backend,
    paged_kv: *PagedKVCache,     // NUEVO
    sequence_id: u64,            // NUEVO
) !Self
```

**Cambios en `AttentionLayer.forward`:**
```zig
pub fn forward(
    self: *Self,
    x: Tensor(f16),           // [N, seq_len, n_embd]
    out: *Tensor(f16),        // [N, seq_len, n_embd]
    block_table: *BlockTable, // NUEVO: tabla de bloques
    start_pos: usize,
    n: usize,
) !void
```

---

## 🟡 FASE 2: Kernel de Atención Paginada (2-3 sem)

### 2.1 Kernel CUDA: `paged_attention_decode`

**Archivo**: `src/cuda/paged_attention.cu` (ya existe stub)

Implementar kernel optimizado:
```cuda
__global__ void paged_attention_decode_kernel(
    const half* __restrict__ query,      // [batch, num_q_heads, head_dim]
    const half* __restrict__ key_cache,  // [num_blocks, block_size, num_kv_heads, head_dim]
    const half* __restrict__ value_cache,// [num_blocks, block_size, num_kv_heads, head_dim]
    half* __restrict__ output,           // [batch, num_q_heads, head_dim]
    const int* __restrict__ block_table, // [batch, max_blocks]
    const int* __restrict__ seq_lens,    // [batch]
    int num_q_heads, int num_kv_heads,
    int head_dim, int block_size,
    float scale
)
```

**Optimizaciones:**
- [ ] Shared memory para Q/K/V tiles
- [ ] Online softmax (iterativo, sin materializar S completo)
- [ ] Warp-level reduction para softmax
- [ ] FP16 accumulation en FP32
- [ ] Vectorized loads (LDST.128)
- [ ] Async copy (TMA en Hopper+)

---

### 2.2 Kernel: `paged_attention_prefill`

**Diferencias vs decode:**
- Procesa secuencia completa de una vez
- Usa causal mask (triangular)
- Puede usar FlashAttention internamente
- Output: todos los tokens de la secuencia

---

### 2.3 Kernel: `paged_attention_reshape_and_cache`

Para mover K/V de layout contiguo (output de linear) a layout paginado:
```cuda
__global__ void reshape_and_cache_kernel(
    const half* __restrict__ kv_input,   // [batch, seq_len, num_kv_heads * head_dim * 2]
    half* __restrict__ key_cache,        // [num_blocks, block_size, num_kv_heads, head_dim]
    half* __restrict__ value_cache,
    const int* __restrict__ block_table,
    int num_kv_heads, int head_dim, int block_size,
    int seq_len, int batch_size
)
```

---

## 🟠 FASE 3: Integración en Pipeline Híbrido (1-2 sem)

### 3.1 `HybridLayer.forward` con PagedAttention

```zig
pub fn forward(self: *Self, x: Tensor(f16), out: *Tensor(f16), 
               block_table: *BlockTable, 
               start_pos: usize, n: usize) !void
{
    // 1. RMSNorm
    norm.rmsNorm(f16, f16, x, self.attn_norm, rms_eps, &normed);
    
    // 2. QKV projection (fused)
    // qkv = [N, seq_len, qg_dim] = [N, seq, qg_dim]
    try self.w_qkv.dequantToF16(self.scratch_f16);  // o f32 path
    try self.matmul_engine.linearProjection(f16, normed, self.w_qkv, &qkv);
    
    // Split Q/K/V + RoPE
    // Q: [N, seq_len, q_dim], K: [N, seq_len, kv_dim], V: [N, seq_len, kv_dim]
    
    // 3. Reshape + Cache (PagedAttention reshape_and_cache)
    try self.reshape_and_cache(q, k, v, block_table, seq_len);
    
    // 4. PagedAttention decode
    try self.paged_attention.decode(
        query, output, block_table, &self.block_alloc
    );
    
    // 5. Output projection
    try self.matmul_engine.linearProjection(f16, attn_out, self.w_o, out);
}
```

---

### 3.2 `runHybridInference` con Scheduler

```zig
fn runHybridInference(...) !void {
    // ... setup ...
    
    // PagedKVCache + Scheduler
    var paged_kv = try pa.PagedKVCache.init(allocator, paged_config);
    defer paged_kv.deinit();
    
    var sched = pa.Scheduler.init(allocator, paged_config, &paged_kv);
    defer sched.deinit();
    
    // Prefill batch
    var req_id = try sched.submit(.{ .prompt_tokens = prompt_ids, ... });
    var batch = try sched.schedule();
    
    while (sched.numRunning() > 0) {
        const batch = try sched.schedule();
        
        // Prefill o Decode según phase
        for (seq_id in batch) {
            const seq = sched.getSequence(seq_id);
            if (seq.phase == .prefill) {
                try paged_attn.prefill(...);
            } else {
                try paged_attn.decode(...);
            }
        }
        
        // Sample tokens
        for (seq_id in batch) {
            const token = sample(logits);
            sched.appendToken(seq_id, token);
        }
    }
}
```

---

## 🔴 FASE 4: Prefix Caching + CPU Offload (2-3 sem)

### 4.1 Prefix Cache

```zig
// En PagedKVCache
pub fn matchPrefix(self: *Self, tokens: []const u32) !usize {
    if (!self.config.enable_prefix_cache or tokens.len == 0) return 0;
    return self.prefix_cache.longestPrefixMatch(tokens);
}

pub fn cachePrefix(self: *Self, seq_id: u64, tokens: []const u32) !void {
    // Hash de bloques completos → insertar en PrefixCache
}
```

**Uso en Scheduler.admitRequests:**
```zig
fn admitRequests(self: *Self) !void {
    for (waiting_queue) |req| {
        const prefix_len = try kv_cache.matchPrefix(req.prompt_tokens);
        // Solo allocar bloques para tokens NUEVOS
        blocks_needed = (new_tokens + block_size - 1) / block_size;
    }
}
```

---

### 4.2 CPU Offloading

**En `BlockAllocator`:**
```zig
pub fn swapToCpu(self: *Self, phys_id: usize) !void {
    const block = &self.blocks[phys_id];
    if (block.on_gpu) {
        // Copia GPU → Host (pinned memory)
        try cudaMemcpyAsync(host_ptr, gpu_ptr, block_bytes, D2H, stream);
        block.on_gpu = false;
        block.cpu_ptr = host_ptr;
    }
}

pub fn swapToGpu(self: *Self, phys_id: usize) !void {
    const block = &self.blocks[phys_id];
    if (!block.on_gpu) {
        // Host → GPU
        try cudaMemcpyAsync(gpu_ptr, cpu_ptr, block_bytes, H2D, stream);
        block.on_gpu = true;
    }
}
```

**En `Scheduler.preemptIfNeeded`:**
```zig
fn preemptIfNeeded(self: *Self) !void {
    while (kv_cache.freeBlocks() == 0 and running.len > 0) {
        victim = select_victim(); // LRU / lowest priority
        for (victim_block_table) |phys_id| {
            try block_alloc.swapToCpu(phys_id);
        }
        // Mover a preempted queue
    }
}
```

---

## 🟢 FASE 5: Tests y Validación (1-2 sem)

### 5.1 Tests Unitarios

```bash
# Tests existentes
zig build test

# Tests específicos PagedAttention
zig build test -- test_paged_attention
```

**Tests requeridos:**
- [ ] `test_paged_attention_decode` - decode single token vs CPU reference
- [ ] `test_paged_attention_prefill` - prefill vs CPU reference
- [ ] `test_prefix_cache` - prefix match / cache hit rate
- [ ] `test_block_allocator_cow` - COW en fork
- [ ] `test_cpu_offload` - swap to CPU + restore
- [ ] `test_preempt_restore` - preempt + restore sequence
- [ ] `test_prefix_cache_hit` - shared prefix reutilizado

### 5.2 Benchmarks

```zig
// benchmarks/bench_paged_attention.zig
pub fn bench_paged_attention() !void {
    // Latencia decode (ms/token) vs batch size
    // Throughput (tokens/s) vs batch size
    // Memoria VRAM vs num_blocks
    // Prefix cache hit rate
    // CPU offload latency
}
```

---

## 📋 Checklist de Integración

| Tarea | Estado | Responsable |
|---|---|---|
| Integrar `PagedKVCache` en `runHybridInference` | ⬜ | |
| Actualizar `HybridLayer` + `AttentionLayer` para PagedKVCache | ⬜ | |
| Implementar `reshape_and_cache` kernel | ⬜ | |
| Implementar `paged_attention_decode` kernel | ⬜ | |
| Implementar `paged_attention_prefill` kernel | ⬜ | |
| Integrar `Scheduler` en `runHybridInference` | ⬜ | |
| Prefix Cache + `matchPrefix` en `admitRequests` | ⬜ | |
| CPU Offload (`swapToCpu` / `swapToGpu`) | ⬜ | |
| Preemption + restore | ⬜ | |
| Prefix cache hit rate metrics | ⬜ | |
| Tests unitarios + benchmarks | ⬜ | |

---

## 📝 Notas de Implementación

### Layout de Memoria (Column-Major para cuBLAS/cuBLASLt)

```
Key Cache:   [num_blocks, block_size, num_kv_heads, head_dim]  (column-major)
Value Cache: [num_blocks, block_size, num_kv_heads, head_dim]  (column-major)
Block Table: [batch_size, max_blocks]  (int32)
```

### Layout de Q/K/V para reshape_and_cache

```
qkv_input: [batch, seq_len, num_kv_heads * head_dim * 2]  // K + V concatenados
// o separado:
// K: [batch, seq_len, num_kv_heads * head_dim]
// V: [batch, seq_len, num_kv_heads * head_dim]
```

### Online Softmax (Iterativo)

```zig
// En attention.zig (CPU reference) - líneas 36-102
// Implementar igual en CUDA:
// 1. max_score por head
// 2. exp(score - max) acumulativo
// 3. Normalización final
```

---

## 📋 Próximos Pasos Inmediatos

1. **Esta semana**: Integrar `PagedKVCache` en `runHybridInference` + compilar
2. **Próxima semana**: `reshape_and_cache` + `decode` kernels + test unitario
3. **Siguiente semana**: `Scheduler` + `prefill`/`decode` + test E2E
4. **Luego**: Prefix cache + CPU offload

---

> **Nota**: El módulo `paged_attention` ya tiene 90% de la lógica CPU + stubs CUDA. El trabajo principal es **wiring** (conectar piezas) + **kernels CUDA** para decode/prefill/reshape.