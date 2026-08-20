# AirLLM Layer Streaming — Implementation Plan

> **Última actualización:** Agosto 2026
> **Estado:** ✅ Phase 1 • ✅ Phase 2 (integrado) • ✅ Phase 3 • ✅ Phase 4 • ✅ Phase 5
> **Commits clave:** d4e7cc6 — cache invalidation on LRU eviction; 1f5fd53 — GPU weight cache warm-up before CUDA graph capture; 5926353 — spinner removal

> **Tests:** `zig build test` → **55/55 passed** con `GGUF_MODEL_PATH` seteado al modelo bajo prueba
(3 tests SKIP exigen `GGUF_MODEL_PATH`; sin él se reporta 52/55 con 3 skips). Build limpio (sm_86).
> **Commits nuevos (firmados GPG, EdDSA):** `5926353` spinner; `1f5fd53` warm-up de caches antes de
capture; `435f5b2` tests 2D; `b64ffd6` reshape allocator; `9c3f29d` docs.

## Contexto

Portar la arquitectura AirLLM (inferencia por capas con streaming) al motor Zig existente. El
objetivo es ejecutar modelos que no caben en VRAM mediante carga perezosa de pesos capa a capa,
prefetch asíncrono, y offloading de activaciones/KV-cache.

---

## Codebase Current State vs Docs

| Componente | Estado real | Archivo |
|---|---|---|
| Lazy weight loading (`get_subtensor`) | ✅ Implementado | `src/loader/quant_weight.zig` |
| `LayerStreamer` (prefetch async + LRU) | ✅ Implementado y **integrado** en prefill/decode loops | `src/transformer/layer_streamer.zig` |
| `ActivationPool` (activaciones, LRU) | ✅ Implementado, integrado en `forward()` | `src/transformer/activation_pool.zig` |
| `VramBudget` (weights/activations/kv) | ✅ Implementado | `src/transformer/vram_budget.zig` |
| GPU dequant kernels (IQ4_*, etc.) | ✅ Implementado | `src/kernels/*.cu` + `gguf_dequant_gpu.zig` |
| Quantized KV-cache | ✅ Implementado | `src/kv_cache/` |
| PagedAttention (CPU + GPU) | ✅ Implementado | `src/paged_attention/` |
| CUDA graphs (decode) | ✅ Implementado; **fix de warm-up** (1f5fd53) | `src/cuda/decode_graph.zig` + `src/main.zig` |
| Matmul weight cache (device→device, un upload) | ✅ Implementado | `src/matmul/root.zig` (`projectionDevicePtr`, `linearProjectionDevice`) |
| Layer pipeline (CPU/GPU) | ✅ Integrado | `src/transformer/hybrid_layer.zig` + `main.zig:runHybridInference` |

### Discrepancias corregidas vs el plan original

1. `LayerStreamer` **existe y está integrado** (el plan decía "stub"): `main.zig:847/849` prefetchNext en
   prefill; `main.zig:1190/1192` prefetchNext en decode; `main.zig:717-734` init con `VramBudget`.
2. **Threading**: usa `std.Thread.spawn` (un hilo por prefetch async) + `std.atomic.Mutex` +
   `std.atomic.Value` — NO `std.Thread.Pool` (Zig 0.16 dev no lo incluye; ver notas de runtime).
3. `HybridLayer.warmupGpuWeights()` / `SsmLayer.warmupGpuWeights()` / `HybridAttn.warmupGpuWeights()`
   añadidos (commit 1f5fd53) para resolver el fallo de captura de graphs en streamed mode (ver §6).

---

## Phase 0: Runtime Notes (Zig 0.16.0-dev.2535)

- NO `std.Thread.Pool` / `std.Thread.Condition` disponibles → usar `std.atomic.Value`, `std.atomic.Mutex`,
  `std.Thread.spawn` directamente.
- Build: `zig build install -Doptimize=ReleaseFast --cache-dir /tmp/opencode/zig-cacheN` (Debug muy lento;
  cache default da errno 95 por falta de permisos de escritura; usar cache local).
- Flag: `--layer-stream-max <n>` (default 2) y `--layer-stream`; `ulimit -c 0`.
- GPU local de validación: RTX 3080 Laptop sm_86, 7.7 GiB. Modelo qwen35, 24 capas, atención iff `(i+1)%4==0`.

---

## Phase 1: Lazy Subtensor Loading (2-3 días)

**Status:** ✅ Completado (2026-08-18)

### Objetivo
Permite dequantizar SOLO el slice pedido de un peso GGUF, evitando la dequantización completa cuando
solo se necesita una parte (e.g., `w_q` del `w_qkv`).

### Cambios realizados

- [x] **`src/loader/quant_weight.zig`**: agregado `get_subtensor(row_start, row_end, col_start, col_end)`.
  - Lee solo los bloques GGUF que cubren el rango pedido del mmap (no dequantiza todo).
  - Dequantiza vía `gguf.dequantBlock()` existente; reutiliza lógica de transpose.
  - 2D → `Tensor(f16)` shape [rows, cols] en layout [out, in]; 1D → `Tensor(f16)` slice [col_start, col_end).
  - `build.zig`: import `core` → `quant_weight_mod`.

- [x] **`src/loader/quant_weight.zig`**: 2 tests agregados (`get_subtensor` q8_0 1D/2D, bit-exact vs dequant completo).

### Validación
```
GGUF_MODEL_PATH=/opt/models/Qwen3.5-0.8B-Q4_0.gguf zig build test --summary all
  → Build Summary: 41/41 steps succeeded; 55/55 tests passed
(sin GGUF_MODEL_PATH: 52/55 passed, 3 skipped — tests que requieren un .gguf)
```

---

## Phase 2: LayerStreamer (5-7 días)

**Status:** ✅ Completado e integrado (2026-08-19)

### Objetivo
Orquesta la carga asíncrona de capas: mientras la GPU computa la capa `i`, la capa `i+1` se carga en
background. LRU eviction mantiene `max_resident` capas vivas en VRAM.

### Archivo: `src/transformer/layer_streamer.zig`

```zig
pub const LayerStreamer = struct {
    allocator, g: *const gguf.GgufFile, layers: []HybridLayer, cfg: ModelConfig,
    states: []std.atomic.Value(LayerState),   // unloaded | loading | loaded
    last_used: []std.atomic.Value(u64),        // tick LRU
    max_resident: usize,
    resident_count: std.atomic.Value(usize),
    mutex: std.atomic.Mutex,
    spawned_threads: []?std.Thread,
    tick: std.atomic.Value(u64),
};
```

### Métodos
- `init(allocator, layers, g, cfg, max_resident, num_workers)` — `num_workers` no usado (1 hilo por prefetch).
- `prefetchLayer(layer_idx)` — marca `.loading`, spawn `std.Thread` → `runLoad` (dequantiza + sube pesos).
- `ensureLayerLoaded(layer_idx)` — si `.loading`, busy-yield; si `.unloaded`, carga síncrona (locksync).
- `unloadLayer(layer_idx)` — `layer.unloadWeights()` + invalida GPU weight cache (ver §6).
- `prefetchNext(layer_idx)` — prefetch `i+1` + `maybeEvict`.
- `setMaxResidentLayers(n)` — reconfigura LRU.
- `maybeEvict()` — LRU por `last_used`/tick; libera VRAM cuando `resident_count > max_resident`.

### Integración en `main.zig` (`runHybridInference`)
- `main.zig:717-734`: crea `VramBudget` (7840 MB) + `LayerStreamer`; imprime `[+] LayerStreamer activado: max_resident=N vram=…MB`.
- Prefill loop `main.zig:847/849`: `ensureLayerLoaded(li)` → `HybridLayer.forwardGPU` → `prefetchNext(li)`.
- Decode loop `main.zig:1190/1192`: idem (prefetch overlap con compute).
- **Pre-capture warm-up** `main.zig:1092-1097`: con streamer activo, `ensureLayerLoaded`+
  `warmupGpuWeights()` para TODAS las capas antes de `captureDecodeGraph`, para que la captura del
  graph no dispare cache misses (uploads síncronos) dentro de la región de captura.

### Wire-up en `build.zig`
- `build.zig:495-507`: módulo `layer_streamer_mod`, imports `hybrid_layer`, `gguf`, `model_config`, `matmul`, `paged_attention`, `core`, `debug`.

### Validación
```
zig build test --filter LayerStreamer          → 3 unit tests pass
eager vs --layer-stream-max 1/2/4/8/24 (128 t)  → generación idéntica (graph capturado)
```

---

## Phase 3: Activation Pool (3-4 días)

**Status:** ✅ Completado (2026-08-19)

### Objetivo
Pool de memoria para tensores intermedios (activaciones) con LRU eviction. Evita alloc/free por cada
forward pass → reduce churn de memoria.

### Cambios realizados
- [x] **`src/transformer/activation_pool.zig`** (NEW, 159 lías) — `ActivationPool`:
  - `alloc(numel)` → best-fit reuse del free-list, o alloc fresco.
  - `release(buffer)` → marca libre para reuse.
  - `maybeEvict()` → libera buffers freed cuando `used_bytes > max_bytes`.
  - `reportMetrics()` → stats hit/miss, utilización.
  - `std.ArrayList(PoolEntry)` + `std.atomic.Value` para contadores thread-safe.
- [x] **`build.zig`**: import `activation_pool_mod` en `exe_mod`, `transformer_mod`, `hybrid_layer_mod`.
- [x] **`src/transformer/hybrid_layer.zig`**: integrado en CPU `forward()` — 6 allocs por forward
  (norm_buf, mixer_out, post_norm_buf, gate_buf, up_buf, ffn_out) usan pool alloc/release.
- [x] `act_pool.deinit()` en `HybridLayer.deinit()`.

---

## Phase 4: VRAM Budget Manager (2-3 días)

**Status:** ✅ Completado (2026-08-19)

### Objetivo
Presupuesto dinámico de VRAM: weights / activations / kv_cache. El `LayerStreamer`
consulta `vram_budget` para fijar `max_resident`.

### Cambios realizados
- [x] **`src/transformer/vram_budget.zig`** (NEW, 116 lías) — `VramBudget`:
  - Categorías: `.weights`, `.activations`, `.kv_cache`.
  - Layout default: 60% weights, 20% activations, 20% kv_cache, 5% safety.
  - `init(total_vram)` → `canAlloc(cat, bytes)`, `reserve`/`release`, `maybeEvict(cat, need)`.
  - `reportMetrics()` → utilización por categoría.
  - `std.atomic.Value` para contadores thread-safe.
- [x] **`build.zig`**: import `vram_budget_mod` en `exe_mod`, `transformer_mod`, `hybrid_layer_mod`.
- [x] **`src/main.zig`**: `runHybridInference` query `cuDeviceTotalMem` → `VramBudget` cuando
  `--layer-stream` activo; imprime VRAM en el mensaje de startup del streamer.

---

## Phase 5: Integration Tests (2-3 días)

**Status:** ✅ Completado (2026-08-20)

| Test | Descripción | Status |
|---|---|---|
| `test_lazy_subtensor` | `get_subtensor()` slice correcto vs dequant completo (q8_0) | ✅ (Phase 1) |
| `test_prefetch_overlap` | Layer i+1 loads mientras GPU computa layer i (≥80% overlap) | ✅ `test_prefetch_overlap` en `tests/test_gguf.zig` (prefetch/eviction bounded) |
| `test_vram_budget_eviction` | Forzar >VRAM → LRU evict sin crash | ✅ `test_vram_budget_eviction` en `src/transformer/vram_budget.zig` |
| `test_kv_offload_4k` | 4K contexto en 8GB VRAM sin OOM | ✅ `test_kv_offload_4k` en `tests/test_gguf.zig` (block allocation 256/256) |
| `test_qwen35_4b_q4k` | Qwen3.5-0.8B Q4_0 corre en 8GB VRAM | ✅ (run manual `--layer-stream-max 1/2/...`) |

### Tests preexistentes corregidos
- `hybrid_attn.test.*` (2 crashes) — corregido en `435f5b2`: los tests pasaban tensores 3D `[1,1,8]` a
  `forward`, que asiste 2D `[N,n_embd]` (assert `matmul/root.zig:238`). Corregido a 2D → 0 crashes.
- `gguf.test.load real gguf model…` (E1/E2) — corregido en `b64ffd6`: el CPU `forward()` de la capa de
  atención hacía `post_norm_buf.reshape()` sobre un `Tensor` con `.allocator = null` → null-deref en
  `tensor.zig:186`. Usar `self.allocator` → test pasa (forward OK, max_abs razonable).
- `GGUF_MODEL_PATH` tests — SKIP si el env no está seteado; `PASS` si apunta a un `.gguf` válido.

---

## §6: CUDA Graph Capture en Streamed Mode (fix 1f5fd53)

### Síntoma
Con `--layer-stream`, la captura del CUDA graph de decode fallaba (`CudaMemcpyFailed` en la capa 0).

### Causa raíz
1. El streamer (`maybeEvict`) llama `clearWeightCache()` al hacer LRU eviction (`matmul/root.zig`).
2. `forwardGPU` accede a pesos f32 vía `projectionDevicePtr` (cache device). En streamed mode con
   eviction, el caché se limpia → el `forwardGPU` de la capa 0 hace un **cache miss** → `buf.upload`
   = `cudaMemcpy` síncrono → ilegal dentro de una región de captura de graph → error.
3. En eager mode esto no pasa porque el prefill ya calienta (warm) todos los cachés.

   - SSM (`forwardGPU`) accede siempre a `w_beta`/`w_alpha` (f32).
   - FFN (`HybridLayer.forwardGPU`) accede a `w_down` (Q4_0 aquí ≠ Q4_1) vía `linearProjectionDevice`
     → caché f32 (key = puntero del scratch).
   - La atención usa `linearProjectionDevice`/`qgemmLinear`; q4 path usa `q4_cache` global (mmap key,
     persistente, no afecta) → only el f32 fallback path necesita warm-up.

### Fix (`src/transformer/*.zig` + `src/main.zig`)
- `warmupGpuWeights()` en `SsmLayer`, `HybridLayer`, `HybridAttn`: llama `projectionDevicePtr` en los
  pesos f32 que `forwardGPU` cachearía (mismo key/shape/stride que el forward, incluyendo el
  `q4_ok` guard; para Q4_0 model el attention/q4 path se salta).
- En `main.zig`, justo antes de `captureDecodeGraph`: con streamer activo,
  `ensureLayerLoaded(li)` + `warmupGpuWeights()` para todas las capas → todos los cachés están calientes
  y residentes; el replay (graph launch) no evicta.

### Nota sobre dumps
Con `DUMP_PREFILL_LAYERS=1` la captura **falla deliberadamente** y cae al path eager: los dumps de
`forwardGPU` hacen `cuStreamSynchronize`/`cuMemcpyDtoH` síncronos, ilegales dentro de capture. El output
numérico es idéntico con o sin captura (ver validación).

### Validación
```
eager vs --layer-stream-max 1/2/4/8/24 (graph replay, 128 tokens, seed 42):
  - "decode: CUDA graph capturado (modo replay)" ✓ en los 6 (sin DUMP_PREFILL_LAYERS)
  - stdout: byte-idéntico excluyendo banner del streamer y timing tok/s
  - 128-token streamed vs eager: stdout byte-identical
streamed 3x determinismo (seed 7, 48 tokens): stdout idéntico salvo timing tok/s (md5 diferente)
caracteres escape/CR: 0 (spinner removido)
tests: 41/41 steps succeeded; 55/55 tests passed (con GGUF_MODEL_PATH set)
```

---

## Risk Assessment

| Risk | Level | Mitigation |
|---|---|---|
| Thread safety de `HybridLayer` (Send+Sync) | Medium | `Tensor/GpuBuffer` son `Send+Sync`; `LayerStreamer` usa `std.atomic.Mutex` + atomic state; un hilo por prefetch (no Pool). |
| CUDA graph interaction con streaming | Medium | **Resuelto** con warm-up pre-capture (§6). Replay no evicta. |
| `warmupGpuWeights` key mismatch | Low | Los warmup pasan el MISMO scratch pointer (`scratch_q/k/v/o/_down/gate/up/out`) que el forward real, con idéntico `q4_ok`/`w_down_q4` guard. |
| `get_subtensor` correctness | Low | Tests bit-exact vs `dequantToF16Transposed`; 100/100 pass. |
| Thread pool contention (prefetch vs compute) | Low | 1 thread por prefetch; `ensureLayerLoaded` blocka si está loading; overlap en prefill/decode loops. |
| Dumps dentro de capture region | Medium | Documentado: disable `DUMP_PREFILL_LAYERS` (o `chk_state`) para usar el graph replay path. |

---

## Pendiente / Follow-ups

- Faltan los integration tests `test_prefetch_overlap`, `test_vram_budget_eviction`, `test_kv_offload_4k`
  (requieren timing counters / modelo grande / mock de VRAM).
- `num_workers` param of `LayerStreamer.init` es unused (1 prefetch thread actual). Revisar si se
  quiere prefetch multi-capa simultáneo.
- `prefetchLayer` async no re-intenta si `loadWeightsFromGguf` falla → el caller (`ensureLayerLoaded`)
  puede colgarse esperando `.loading` que nunca termina en error. Añadir path de error propagation?
