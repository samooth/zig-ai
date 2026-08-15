# AIRLLM Implementation TODO — Layer Streaming para Modelos > VRAM

> **Objetivo**: Ejecutar modelos que no caben en VRAM (ej. Qwen3.5-4B 5.6GB en 8GB VRAM) mediante **layer streaming** estilo AirLLM: cargar pesos capa a capa desde disco, prefetch asíncrono, offloading de activaciones/KV-cache.

---

## 📋 Estado Actual (Resumen)

| Componente | Estado |
|---|---|
| Carga GGUF + mmap | ✅ `GgufModel.load` |
| Desquant CPU (Q4_0, Q4_1, Q2_K, Q8_K) | ✅ `QuantWeight::dequantToF32Transposed` |
| Matmuls GPU (cuBLAS) | ✅ Pearson 1.0 vs CPU |
| Pipeline híbrido CPU/GPU | ✅ `runHybridInference` |
| KV Cache | ✅ Básico |

**Gaps críticos para AirLLM:**
- ❌ Carga perezosa de sub-tensores (solo lo que se usa)
- ❌ Prefetch asíncrono de capas (capa i+1 mientras GPU computa i)
- ❌ Cuantizaciones Q4_K/Q5_K/IQ4_XS/IQ3_S/FP8
- ❌ Offloading de activaciones/KV-cache a RAM
- ❌ Gestión de presupuesto VRAM (pesos + activaciones + KV)

---

## 📋 TODO Detallado por Fases

---

## 🟢 FASE 1: Lazy Weight Loading (2-3 sem)

### 1.1 `QuantWeight::get_subtensor()` — Carga perezosa de sub-tensores

**Archivo**: `src/loader/quant_weight.zig`

```zig
// Nuevo método: desquantiza SOLO el slice pedido
pub fn get_subtensor(
    self: *Self,
    allocator: std.mem.Allocator,
    tensor_name: []const u8,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
) !Tensor(f32)
```

**Tareas:**
- [ ] Parsear `tensor_info` (offset, shape, strides) desde GGUF metadata
- [ ] Implementar `read_subtensor()` que lea solo los bytes necesarios del mmap
- [ ] Desquantizar solo el slice pedido → `Tensor(f32)` temporal
- [ ] Cache opcional LRU para sub-tensores frecuentes

**Tests:**
- [ ] Cargar solo `w_q` (primer tercio de `wqkv`) → shape correcto
- [ ] Verificar que no se lee todo el tensor del mmap (instrumentar I/O)

---

### 1.2 `LayerStreamer` — Prefetch asíncrono de capas

**Archivo nuevo**: `src/transformer/layer_streamer.zig`

```zig
pub const LayerStreamer = struct {
    allocator: std.mem.Allocator,
    model: *GgufModel,
    pool: std.Thread.Pool,
    prefetch_queue: std.Thread.Queue(usize), // índices de capa
    loaded_layers: std.AutoHashMap(usize, LayerWeights),
    // ...
};
```

**Métodos clave:**
```zig
fn init(allocator, model, num_prefetch_threads) !LayerStreamer
fn prefetch_layer(layer_idx: usize) !void        // Encola carga async
fn get_layer(layer_idx: usize) !LayerWeights    // Bloquea hasta listo
fn unload_layer(layer_idx: usize) void          // Libera memoria
fn set_max_resident_layers(max: usize) void     // LRU eviction
```

**Tasks:**
- [ ] Thread pool con `std.Thread.Pool` (configurable workers)
- [ ] Cola de prefetch con `std.Thread.Queue` (bounded)
- [ ] Cache LRU de capas cargadas (`max_resident_layers`)
- [ ] `prefetch_layer(i+1)` llamado al terminar capa `i`
- [ ] Métricas: hit rate, latency de carga, memoria usada

---

## 🟡 FASE 2: Cuantizaciones Avanzadas (3-4 sem)

### 2.1 Formatos GGUF faltantes

| Formato | Archivo | Estado |
|---|---|---|
| **Q4_K** | `gguf.zig` + `gguf_dequant_gpu.zig` | ❌ |
| **Q5_K** | `gguf.zig` + `gguf_dequant_gpu.zig` | ❌ |
| **Q6_K** | `gguf.zig` + `gguf_dequant_gpu.zig` | ❌ |
| **Q8_0** | ✅ (existe) | — |
| **IQ4_XS** | ❌ | — |
| **IQ3_S** | ❌ | — |
| **FP8 (E4M3/E5M2)** | `gguf_dequant_gpu.zig` + kernel CUDA | ❌ |
| **FP8 (E5M2)** | Para modelos nuevos (Nemotron, etc.) | ❌ |

**Referencia**: `llama.cpp` `ggml-quants.c` / `ggml-quants.h` — implementaciones de referencia exactas.

**Tasks por formato:**
- [ ] Parsear `quantization_type` en `gguf.zig`
- [ ] Implementar `dequantQ4_K`, `dequantQ5_K`, etc. en `gguf.zig` (CPU)
- [ ] Kernels CUDA en `gguf_dequant_gpu.zig` + PTX
- [ ] Tests bit-exact vs `llama.cpp` / `ggml` reference

---

## 🟠 FASE 3: Offloading de Activaciones + KV-Cache (2-3 sem)

### 3.1 Activation Pool + LRU Eviction

**Archivo nuevo**: `src/transformer/activation_pool.zig`

```zig
pub const ActivationPool = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    max_bytes: usize,
    lru: std.DoublyLinkedList(ActivationEntry),
    // ...
    fn alloc(shape: []usize) !Tensor(f16)
    fn free(tensor: Tensor(f16)) void
    fn maybe_evict() !void  // LRU hasta < max_bytes
};
```

**Integración en `hybrid_layer.zig`:**
```zig
// Antes de forward:
var activation = try activation_pool.alloc(shape);
// ... compute ...
// Al terminar capa:
activation_pool.free(activation);
```

### 3.2 KV-Cache Paging / Offloading

**Archivo**: `src/kv_cache/root.zig` (extender `KVCache`)

```zig
pub fn offload_to_host(self: *Self, layer_idx: usize, start_pos: usize, len: usize) !void {
    // Copia K/V de device → host buffer (mmap anónimo o archivo temporal)
}

pub fn prefetch_from_host(self: *Self, layer_idx: usize, start_pos: usize, len: usize) !void {
    // Copia host → device async
}
```

**Estrategia:**
- KV-cache en VRAM solo para ventanas recientes (últimos N tokens)
- Resto en RAM (mmap anónimo) o disco (archivo temporal)
- Prefetch async de la ventana siguiente mientras GPU computa

---

## 🟢 FASE 4: VRAM Budget Manager (2-3 sem)

### 4.1 `VramBudget` — Presupuesto dinámico VRAM

**Archivo nuevo**: `src/transformer/vram_budget.zig`

```zig
pub const VramBudget = struct {
    total_vram: usize,
    weights_budget: usize,      // Pesos residentes
    activations_budget: usize,  // Pool activaciones
    kv_budget: usize,           // KV cache
    safety_margin: usize,       // 5-10%
    
    fn can_alloc(category: Category, bytes: usize) bool
    fn reserve(category: Category, bytes: usize) !void
    fn release(category: Category, bytes: usize) void
    fn maybe_evict() !void  // LRU hasta caber
};
```

**Integración:**
- `LayerStreamer.set_max_resident_layers()` usa `vram_budget.weights_budget`
- `ActivationPool` consulta `vram_budget.activations_budget`
- `KVCache` consulta `vram_budget.kv_budget`

---

## 📋 Tests de Integración Obligatorios

| Test | Descripción | Criterio de éxito |
|---|---|---|
| `test_lazy_qkv_loading` | Cargar solo `w_q` + `w_k` + `w_v` por separado | Pearson ≥ 0.999 vs CPU full |
| `test_prefetch_overlap` | Capa i computa mientras capa i+1 carga | Overlap ≥ 80% |
| `test_vram_budget_eviction` | Forzar > VRAM → LRU evict sin crash | 0 OOM |
| `test_kv_offload_4k` | 4K contexto en 8GB VRAM | Sin OOM, latencia < 2x |
| `test_q4k_4b_model` | Qwen3.5-4B Q4_K en 8GB VRAM | Genera texto coherente |

---

## 📅 Cronograma Sugerido (12-18 semanas)

| Semana | Hito |
|---|---|
| 1-2 | `get_subtensor()` + tests |
| 2-3 | `LayerStreamer` + prefetch overlap |
| 3-5 | Q4_K / Q5_K / Q6_K dequant CPU + GPU |
| 5-6 | Activation Pool + LRU |
| 6-7 | KV-cache paging + offloading |
| 7-8 | VRAM Budget + integración |
| 9-10 | Test E2E: Qwen3.5-4B Q4_K en 8GB VRAM |
| 9-10 | Documentación + benchmarks |

---

## 📁 Archivos a Crear/Modificar (Resumen)

| Archivo | Acción |
|---|---|
| `src/loader/quant_weight.zig` | `get_subtensor()`, lazy dequant |
| `src/transformer/layer_streamer.zig` | **NUEVO** — `LayerStreamer` |
| `src/loader/gguf.zig` | Parseo Q4_K/Q5_K/Q6_K/IQ4_XS/IQ3_S |
| `src/loader/gguf_dequant_gpu.zig` | Kernels Q4_K/Q5_K/IQ4_XS |
| `src/transformer/hybrid_layer.zig` | Integración `LayerStreamer` + `VramBudget` |
| `src/transformer/activation_pool.zig` | **NUEVO** — `ActivationPool` |
| `src/kv_cache/root.zig` | `offload_to_host()` / `prefetch()` |
| `src/transformer/vram_budget.zig` | **NUEVO** — `VramBudget` |
| `tests/test_airllm.zig` | **NUEVO** — Tests de integración |

---

## 📝 Notas de Implementación Clave

1. **Thread safety**: `LayerStreamer` usa `std.Thread.Pool` + `std.Thread.Queue` (thread-safe). `LayerWeights` debe ser `Send` + `Sync` (Zig: sin punteros a memoria no-Send).

2. **Memory mapping**: Usar `std.os.mmap` con `MAP_POPULATE` + `MADV_WILLNEED` / `MADV_DONTNEED` para hint de prefetch.

3. **Error handling**: Todos los `try` propagados; `defer` para limpieza; `errdefer` en caminos de error.

4. **Métricas**: Contadores atómicos para `prefetch_hits`, `prefetch_misses`, `evictions`, `vram_used_bytes` — expuestos via `std.debug.print` o export Prometheus futuro.

---

## 📚 Referencias Implementación

| Componente | Referencia |
|---|---|
| Q4_K/Q5_K dequant | `llama.cpp/ggml/src/ggml-quants.c` |
| Layer streaming | AirLLM paper + `llama.cpp` `llama_model_quantize.cpp` |
| KV-cache paging | `llama.cpp` `llama_kv_cache_unload` / `llama_kv_cache_load` |
| VRAM budgeting | `vllm` `BlockManager` + `BlockManager` |
| `io_uring` | `std.os.linux.io_uring` (Zig 0.13+) / `io_uring` crate |

---

## ✅ Criterios de Éxito (Definition of Done)

- [ ] **Qwen3.5-4B Q4_K** corre en **8GB VRAM** sin OOM
- [ ] **Pearson ≥ 0.999** vs CPU en logits first-token
- [ ] **Prefetch overlap** ≥ 80% (GPU compute ∥ disk I/O)
- [ ] **VRAM peak** ≤ 7.5GB en 8GB VRAM (margen 0.5GB)
- [ ] **Latencia primer token** ≤ 1.5x CPU baseline
- [ ] **Throughput** ≥ 0.8x CPU baseline (meta: 1.5-3x eventual)

---

> **Nota**: Este documento es vivo. Actualizar conforme avancen las fases. Cada PR debe referenciar la tarea correspondiente (ej. `feat: quant-weight-get-subtensor #AIRLLM-1.1`).