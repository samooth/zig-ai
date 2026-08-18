# AirLLM Layer Streaming — Implementation Plan

> **Fecha:** Agosto 2026
> **Estado:** En progreso (Phase 1: ✅ | Phase 2: ⏳ | Phase 3: ⬜ | Phase 4: ⬜ | Phase 5: ⬜)

## Contexto

Portar la arquitectura AirLLM (inferencia por capas con streaming) al motor Zig existente. El
objetivo es ejecutar modelos que no caben en VRAM mediante carga perezosa de pesos capa a capa,
prefetch asíncrono, y offloading de activaciones/KV-cache.

---

## Codebase Current State vs Docs

| Componente | Docs Status | Código Actual |
|---|---|---|
| Lazy weight loading (`get_subtensor`) | ❌ Planned | `QuantWeight` existe (mmap, dequant por bloque) pero `get_subtensor()` **falta** |
| `LayerStreamer` | ❌ Planned | **No existe** — `PrefetchPipeline` y `KVCacheManager.prefetchLayer()` son stubs |
| `ActivationPool` | ❌ Planned | **No existe** |
| `VramBudget` | ❌ Planned | **No existe** — `KVPoolAllocator` tiene `lru_evict` + callback `on_evict` |
| GPU dequant kernels (IQ4_XS, IQ3_S, etc.) | ❌ Planned | **Implementado** — `kernels/*.cu` + `gguf_dequant_gpu.zig` |
| Quantized KV-cache | ❌ Planned | **Implementado** — `src/kv_cache/` |
| PagedAttention CPU + GPU | — | **Implementado** — `src/paged_attention/` |
| CUDA graphs (decode) | — | **Implementado** — `src/cuda/decode_graph.zig` |
| Matmul weight cache | — | **Implementado** — `src/matmul/root.zig` (`gemmCuBlasF32Resident`) |
| Layer pipeline (CPU/GPU) | ✅ | `src/transformer/hybrid_layer.zig` + `main.zig:runHybridInference` |

### Discrepancias Docs vs Código

1. **GPU attention está ENABLED**: `main.zig:653` → `gpu_attention_enabled = true` (docs dicen `false`)
2. **Q4_K/Q5_K/IQ kernels existen** — los docs marcan como ❌ pero el código los tiene
3. **`PrefetchPipeline` existe** pero es un stub (marca streams busy, no carga pesos)
4. **`KVCacheManager.prefetchLayer()`** — stub vacío

---

## Phase 1: Lazy Subtensor Loading (2-3 días)

**Status:** ✅ Completado (2026-08-18)

### Objetivo
Permite dequantizar SOLO el slice pedido de un peso GGUF, evitando la dequantización
completa cuando solo se necesita una parte (e.g., `w_q` del `w_qkv`).

### Cambios realizados

- [x] **`src/loader/quant_weight.zig`**: Agregado `get_subtensor(row_start, row_end, col_start, col_end)` 
  - Lee solo los bloques que cubren el rango solicitado del mmap (no dequantiza todo)
  - Dequantiza mediante `gguf.dequantBlock()` existente, reutiliza lógica de transpose
  - Para 2D: retorna `Tensor(f16)` con shape [rows, cols] en layout [out, in]
  - Para 1D: retorna `Tensor(f16)` con los elementos [col_start, col_end)
  - `build.zig`: agregado import `core` → `quant_weight_mod`

- [x] **`src/loader/quant_weight.zig`**: 2 tests agregados
  - `test QuantWeight q8_0 get_subtensor 1D` — verifica slice parcial coincide con dequant completo
  - `test QuantWeight q8_0 get_subtensor 2D` — verifica rows parciales coinciden con transpuesta completa

### Validación
```
zig build test  →  89/94 tests passed (2 new +72 existing), 2 crashes pre-existing en hybrid_attn
```

---

## Phase 2: LayerStreamer (5-7 días)

**Status:** ⏳ En progreso — core file created, integration pending

### Objetivo
Orquesta la carga asíncrona de capas: mientras la GPU computa la capa `i`, la capa `i+1`
se descarga/carga en background. Usa `std.Thread.Pool` + `std.Thread.Queue`.

### Nuevo archivo: `src/transformer/layer_streamer.zig`

```zig
pub const LayerStreamer = struct {
    allocator: std.mem.Allocator,
    model: *gguf.GgufFile,
    pool: std.Thread.Pool,
    prefetch_queue: std.Thread.Queue(usize),
    loaded_layers: std.AutoHashMap(usize, *HybridLayer),
    max_resident: usize,  // LRU eviction threshold
    lru: std.DoublyLinkedList(usize),
    // ...
};
```

### Métodos
- `init(allocator, gguf_file, num_workers, max_resident_layers)`
- `prefetch_layer(layer_idx)` — dequantiza + sube pesos al GPU en worker thread
- `get_layer(layer_idx)` — bloquea hasta listo
- `unload_layer(layer_idx)` — libera GPU buffers
- `set_max_resident_layers(n)` — LRU eviction

### Integración en `main.zig`

Reemplazar el loop secuencial de carga (`for 0..cfg.block_count` → `loadWeightsFromGguf`)
con `LayerStreamer`, y agregar overlap en el forward:
```zig
// En prefill/decode loops:
for (layers, 0..) |*layer, li| {
    streamer.prefetchNext(li + 1);  // async load layer li+1
    try HybridLayer.forwardGPU(layer, &lk, ...);  // compute layer li
    streamer.unloadLayer(li - 1);   // free old layer weights
}
```

### Wire-up en `build.zig`
- Agregar `layer_streamer` module import al `exe_mod` y `hybrid_transformer_mod`

### Validación
```bash
# Test de overlap (instrumentar con timing)
zig build test --filter test_prefetch_overlap
```

---

## Phase 3: Activation Pool (3-4 días)

**Status:** ⬜ Pendiente

### Objetivo
Pool de memoria para tensores intermedios (activaciones) con LRU eviction.
Actualmente cada capa hace alloc/free por token → churn de memoria GPU.

### Nuevo archivo: `src/transformer/activation_pool.zig`
```zig
pub const ActivationPool = struct {
    arena: std.heap.ArenaAllocator,
    max_bytes: usize,
    lru: std.DoublyLinkedList(ActivationEntry),
    // alloc(shape), free(tensor), maybeEvict()
};
```

### Integración en `HybridLayer.forwardGPU()`
- Reemplazar `cublas.GpuTensor.alloc` transient con `ActivationPool.alloc` reutilizable

---

## Phase 4: VRAM Budget Manager (2-3 días)

**Status:** ⬜ Pendiente

### Objetivo
Presupuesto dinámico de VRAM: weights / activations / KV-cache.
`LayerStreamer.setMaxResidentLayers()` usa `vram_budget.weights_budget`.

### Nuevo archivo: `src/transformer/vram_budget.zig`
```zig
pub const VramBudget = struct {
    total_vram: usize,
    weights_budget: usize,
    activations_budget: usize,
    kv_budget: usize,
    safety_margin: usize,
    fn canAlloc(cat, bytes) bool
    fn reserve(cat, bytes) !void
    fn release(cat, bytes) void
    fn maybeEvict() !void
};
```

---

## Phase 5: Integration Tests (2-3 días)

**Status:** ⬜ Pendiente

| Test | Descripción |
|---|---|
| `test_lazy_subtensor` | `get_subtensor()` returns correct slice, minimal I/O |
| `test_prefetch_overlap` | Layer i+1 loads mientras GPU computa layer i (≥80% overlap) |
| `test_vram_budget_eviction` | Forzar >VRAM → LRU evict sin crash |
| `test_kv_offload_4k` | 4K contexto en 8GB VRAM sin OOM |
| `test_qwen35_4b_q4k` | Qwen3.5-0.8B Q4_K corre en 8GB VRAM |

---

## Risk Assessment

| Risk | Level | Mitigation |
|---|---|---|
| Thread safety de `HybridLayer` (Send+Sync) | Medium | Verificar Tensor/GpuBuffer son Send+Sync; usar `*Mutex` si necesario |
| CUDA graph interaction con streaming | Medium | Disable graph mode durante streaming; re-capturar al terminar |
| `get_subtensor` correctness | Low | Reusar `dequantBlock()` + transpose logic; tests bit-exact vs `dequantToF16Transposed` |
| Thread pool contention (prefetch vs compute) | Medium | Configurable `num_prefetch_threads` (default 2) |
