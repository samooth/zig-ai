# Análisis: zig-kv-cache-gpu + zig-ai-engine

> **Fecha:** 2026-08-02  
> **Estado:** Módulo KV-Cache implementado. Pendiente integración y GQA.

---

## 1. Inventario de lo que ya tienes (zig-kv-cache-gpu)

### 1.1 CUDA / GPU

| Componente | Archivo | Estado | Notas |
|------------|---------|--------|-------|
| Kernels de de-cuantización | `cuda/dequantize_kernels.cu` | ✅ Completo | INT8 sym/asym, INT4, Q4_0, Q8_0 |
| FFI launchers | `gpu_dequant.zig` | ✅ Completo | Wrappers Zig→CUDA con stream async |
| Buffers GPU persistentes | `GpuDequantBuffers` | ✅ Completo | Reutilización, auto-resize |

### 1.2 Core KV-Cache

| Componente | Archivo | Estado | Notas |
|------------|---------|--------|-------|
| Tipos de cuantización | `quant_types.zig` | ✅ Completo | 6 formatos, metadatos por bloque |
| Pool allocator | `allocator.zig` | ✅ Completo | bump, free_list, LRU_evict, compactación |
| KVCacheManager | `kv_cache_manager.zig` | ✅ Completo | Múltiples secuencias, métricas, CPU fallback |
| StreamRing | `stream.zig` | ✅ Completo | 3 streams, overlap compute/memcpy |
| PrefetchPipeline | `stream.zig` | ✅ Completo | Prefetch capa N+1 |

### 1.3 Integración FlashAttention

| Componente | Archivo | Estado | Notas |
|------------|---------|--------|-------|
| KVFlashAttention | `flash_attention.zig` | 🚧 Placeholder | Estructura lista, falta kernel real |
| QuantizedGPUTensor | `flash_attention.zig` | ✅ Completo | Descriptor de tensor cuantizado en GPU |

---

## 2. Gap Analysis — Qué falta para el motor completo

### 2.1 Crítico (Bloqueante)

| # | Gap | Dónde | Impacto |
|---|-----|-------|---------|
| **G1** | **GQA / MQA nativo** | `quant_types.zig`, `kv_cache_manager.zig` | Llama-3, Qwen, Mistral usan GQA. Sin esto, KV-Cache 4-8x más grande |
| **G2** | **Integración TransformerLayer ↔ KVCacheManager** | `src/transformer/layer.zig` | `forward()` no usa el manager, no cuantiza K/V |
| **G3** | **Pipeline autoregresivo** | `src/main.zig` | No hay bucle prefill → generateToken |
| **G4** | **Cuantización de K/V en append** | `kv_cache_manager.zig` | `appendTokens` recibe `[]const u8` ya cuantizado, pero nadie cuantiza |
| **G5** | **De-cuantización en kernel FA** | `flash_attention.zig` | `launchFlashAttention` es placeholder vacío |

### 2.2 Alto (Performance)

| # | Gap | Dónde |
|---|-----|-------|
| **G6** | K/V persistentes en GPU | `kv_cache_manager.zig` — ahora todo pasa por host |
| **G7** | Tests unitarios del módulo kv_cache | Solo hay benchmark, 0 tests |
| **G8** | Configuración por capa funcional | `layer_formats` existe pero no se usa en alloc real |

### 2.3 Medio (Features)

| # | Gap | Dónde |
|---|-----|-------|
| **G9** | Sliding window attention | No implementado |
| **G10** | Prefix caching | No implementado |
| **G11** | Continuous batching | No implementado |

---

## 3. Plan de Integración Actualizado

### Paso 1: GQA / MQA (1-2 días)
**Por qué primero:** Sin GQA, el cache es inusable para modelos modernos.

**Cambios necesarios:**
```
quant_types.zig:
  - KVCacheConfig: añadir num_kv_heads: u32
  - SequenceState: k_slots/v_slots de [layer][head] → [layer][kv_head]

kv_cache_manager.zig:
  - createSequence: allocar num_kv_heads en lugar de num_heads
  - appendTokens: recibir head_idx como kv_head_idx
  - retrieveForAttention: broadcast de kv_head a q_heads en de-cuantización

flash_attention.zig:
  - FlashAttentionOp: num_q_heads vs num_kv_heads
  - GQA broadcast en el kernel (o en Zig antes de lanzar)
```

### Paso 2: Cuantización en append (2-3 días)
**Por qué:** Ahora `appendTokens` recibe bytes ya cuantizados. Necesita recibir FP16 y cuantizar.

**Cambios:**
```
kv_cache_manager.zig:
  - appendTokensF16(seq_id, layer, head, k_f16[], v_f16[]) → cuantiza internamente
  - Añadir quant_ops.zig con funciones de cuantización CPU
  - Soporte por formato: Q4_0, Q8_0, INT8 sym

quant_ops.zig (nuevo):
  - quantizeQ4_0(src: []f16, dst: []u8, block_size: usize) → scale + nibbles
  - quantizeQ8_0(src: []f16, dst: []u8, block_size: usize) → scale + int8
  - quantizeInt8Symmetric(src: []f16, dst: []u8, block_size: usize)
```

### Paso 3: Integración TransformerLayer (2-3 días)
**Conectar el zig-ai-engine con zig-kv-cache-gpu.**

**Cambios en `src/transformer/layer.zig`:**
```zig
pub const TransformerLayer = struct {
    // ... campos existentes ...

    // NUEVO:
    kv_manager: ?*kvcache.KVCacheManager = null,
    seq_id: u64 = 0,

    pub fn forward(
        self: *Self,
        hidden_state: Tensor(f16),
        output: *Tensor(f16),
        position: usize,      // ← NUEVO
        is_prefill: bool,     // ← NUEVO
    ) !void {
        // 1. Proyecciones Q/K/V (igual)

        // 2. RoPE (igual)

        // 3. KV-CACHE: cuantizar y almacenar
        if (self.kv_manager) |mgr| {
            // Cuantizar k_proj y v_proj
            // append al manager
        }

        // 4. Recuperar K_full/V_full desde cache
        //    de-cuantizar (GPU si disponible)

        // 5. FlashAttention con K_full/V_full

        // 6. O-proj + residual + FFN
    }
};
```

### Paso 4: Pipeline Autoregresivo (1-2 días)
**Bucle de generación en `src/main.zig`.**

```zig
// Prefill
for (layers) |*layer| {
    try layer.forward(hidden, &output, 0, true);
}

// Generación token a token
for (0..max_tokens) |_| {
    for (layers) |*layer| {
        try layer.forward(token_hidden, &out, current_pos, false);
    }
    const next_token = sample(logits);
    current_pos += 1;
}
```

### Paso 5: Tests y Benchmarks (2 días)
- Tests unitarios para cada formato de cuantización
- Test de round-trip: f16 → cuantizar → de-cuantizar → comparar
- Test de integración: prefill 10 tokens → generate 5 tokens → verificar coherencia
- Benchmark de memoria: FP16 vs Q4_0 vs mixto

---

## 4. Arquitectura de Integración Propuesta

```
┌─────────────────────────────────────────────────────────────────┐
│                    zig-ai-engine (monorepo)                      │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   matmul/   │  │    fa/      │  │      transformer/       │ │
│  │  (6 backends)│  │ (FA GPU+CPU)│  │   layer.zig (usa KV)   │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│         │                │                      │               │
│         └────────────────┴──────────────────────┘               │
│                          │                                      │
│  ┌───────────────────────▼───────────────────────────────┐     │
│  │              kv_cache/ (módulo independiente)          │     │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │     │
│  │  │quant_types│ │allocator │ │gpu_dequant│ │manager   │  │     │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │     │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐               │     │
│  │  │  stream  │ │quant_ops │ │flash_att │               │     │
│  │  └──────────┘ └──────────┘ └──────────┘               │     │
│  └───────────────────────────────────────────────────────┘     │
│                          │                                      │
│  ┌───────────────────────▼───────────────────────────────┐     │
│  │              cuda/ (kernels compilados)                │     │
│  │  flash_attention.cu  │  dequantize_kernels.cu         │     │
│  └───────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Decisiones Técnicas para la Integración

| Decisión | Recomendación | Justificación |
|----------|---------------|---------------|
| **¿Módulo separado o merge?** | **Merge en monorepo** | El zig-ai-engine ya es monorepo. Añadir `src/kv_cache/` y `cuda/dequantize_kernels.cu` |
| **¿Cudaz compartido?** | **Sí** | El `cudaz_stub.zig` del zig-ai-engine sirve para ambos. Unificar en `src/cuda/` |
| **¿Build.zig unificado?** | **Sí** | Un solo `build.zig` que compile PTX de FA + de-cuantización |
| **¿Quién cuantiza?** | **KVCacheManager** | La capa transformer genera FP16, el manager cuantiza al append. Separación de responsabilidades |
| **¿GQA en Zig o CUDA?** | **Zig** | Más simple: en `retrieveForAttention`, repetir K/V heads para cada Q head antes de FA |

---

## 6. Próximos Pasos Inmediatos (Ordenados por Prioridad)

### Hoy — Paso 1: GQA
1. Añadir `num_kv_heads` a `KVCacheConfig`
2. Modificar `SequenceState` para usar `num_kv_heads`
3. En `retrieveForAttention`, implementar broadcast de kv_heads

### Mañana — Paso 2: Cuantización CPU
1. Crear `src/kv_cache/quant_ops.zig`
2. Implementar `quantizeQ4_0` y `quantizeQ8_0`
3. Modificar `appendTokens` para recibir FP16 y cuantizar internamente

### Día 3-4 — Paso 3: Integración Transformer
1. Modificar `TransformerLayer` para usar `KVCacheManager`
2. Añadir parámetros `position` e `is_prefill` a `forward()`
3. Implementar bucle de prefill + generación en `main.zig`

### Día 5 — Paso 4: Tests
1. Test round-trip de cuantización (todos los formatos)
2. Test de secuencia: create → append 10 → retrieve → verificar
3. Test de integración end-to-end (con pesos dummy)

---

## 7. Código Clave para la Integración

### 7.1 Broadcast GQA en retrieve

```zig
// En kv_cache_manager.zig::retrieveForAttention

// Si num_kv_heads < num_q_heads, repetir cada kv_head
const kv_head_idx = head_idx % self.config.num_kv_heads;
const k_slot = seq.k_slots[layer_idx][kv_head_idx];
const v_slot = seq.v_slots[layer_idx][kv_head_idx];
```

### 7.2 Cuantización Q4_0 (pseudocódigo)

```zig
fn quantizeQ4_0(src: []const f16, dst: []u8, block_size: usize) void {
    const blocks = src.len / block_size;
    var dst_offset: usize = 0;
    for (0..blocks) |b| {
        // Encontrar max_abs del bloque
        var max_abs: f32 = 0;
        for (0..block_size) |i| {
            max_abs = @max(max_abs, @abs(@as(f32, src[b*block_size + i])));
        }
        const scale = max_abs / 8.0;

        // Escribir scale f16
        const scale_f16: f16 = @floatCast(scale);
        @memcpy(dst[dst_offset..dst_offset+2], std.mem.asBytes(&scale_f16));
        dst_offset += 2;

        // Cuantizar y empaquetar nibbles
        for (0..block_size/2) |i| {
            const q0 = @min(15, @max(0, @round(@as(f32, src[b*block_size + i*2]) / scale) + 8));
            const q1 = @min(15, @max(0, @round(@as(f32, src[b*block_size + i*2 + 1]) / scale) + 8));
            dst[dst_offset] = @as(u8, @intCast(q0)) | (@as(u8, @intCast(q1)) << 4);
            dst_offset += 1;
        }
    }
}
```

### 7.3 Modificación mínima de TransformerLayer

```zig
// En layer.zig, forward():

// Después de projectK y projectV:
if (self.kv_manager) |mgr| {
    // Cuantizar k_proj y v_proj (o pasar directamente si el manager cuantiza)
    try mgr.appendTokensF16(self.seq_id, self.layer_idx, h, 
        self.k_proj.data, self.v_proj.data);
}

// Antes de FA:
var k_full = try Tensor(f16).alloc(...);
var v_full = try Tensor(f16).alloc(...);
defer { k_full.deinit(); v_full.deinit(); }

if (self.kv_manager) |mgr| {
    for (0..self.num_heads) |h| {
        try mgr.retrieveForAttention(self.seq_id, self.layer_idx, h,
            k_full_slice, v_full_slice);
    }
} else {
    // Fallback: usar K, V locales
    k_full = self.k_proj;
    v_full = self.v_proj;
}

// FA con k_full, v_full
try self.fa_engine.forward(self.q_proj, k_full, v_full, &self.attn_out);
```

---

## 8. Métricas de Éxito para la Integración

| Métrica | Target | Cómo medir |
|---------|--------|------------|
| Compresión KV (Q4_0 K + Q8_0 V) | ~3.5x vs FP16 | `manager.metrics.bytes_saved` |
| Round-trip error (Q4_0) | < 2% RMSE | Test unitario |
| Round-trip error (Q8_0) | < 0.5% RMSE | Test unitario |
| Memoria por token (Llama-3-8B, 32 layers, GQA) | ~35 KB/token | `estimatedSize()` |
| Secuencia máxima (8GB pool) | ~240K tokens | `pool_capacity / bytes_per_token` |

---

*Análisis generado tras revisión del código fuente completo de zig-kv-cache-gpu.*
