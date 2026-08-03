# Integración Completa: Zig AI Engine + Matmul v2 + FlashAttention + KV-Cache Cuantizado

> **Documento técnico de integración** — Agosto 2026  
> **Stack:** Zig 0.13.0+, CUDA 12.x (opcional), cuBLAS (opcional)

---

## 1. Visión de Arquitectura

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         APLICACIÓN (main.zig)                               │
│  Carga modelo → Tokeniza → Genera tokens → Detokeniza                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LLM INFERENCE PIPELINE                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │  Tokenizer  │  │  Embedding  │  │   Decoder   │  │     Sampler     │   │
│  │  (BPE/SP)   │  │   Lookup    │  │  (N capas)  │  │  (greedy/top-p) │   │
│  └─────────────┘  └─────────────┘  └──────┬──────┘  └─────────────────┘   │
│                                           │                                 │
│                    ┌──────────────────────┘                                 │
│                    ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    TRANSFORMER LAYER (N veces)                      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │   │
│  │  │  Matmul  │  │  Matmul  │  │  Matmul  │  │   FlashAttention │   │   │
│  │  │  Q-proj  │  │  K-proj  │  │  V-proj  │  │   (GPU/CPU)      │   │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘   │   │
│  │       │             │             │                  │             │   │
│  │       └─────────────┴─────────────┘                  │             │   │
│  │                     │                                │             │   │
│  │              ┌──────▼──────┐                  ┌──────▼──────┐     │   │
│  │              │  KV-Cache   │◄─────────────────│  RoPE + FA  │     │   │
│  │              │ Cuantizado  │                  │   Output    │     │   │
│  │              │  + Paging   │                  └─────────────┘     │   │
│  │              └─────────────┘                                      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │   │
│  │  │  Matmul  │  │   SwiGLU │  │  Matmul  │  │   Residual +     │   │   │
│  │  │  O-proj  │  │   FFN    │  │  FFN-out │  │   LayerNorm      │   │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MOTORES SUBYACENTES                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  Matmul Engine  │  │ FlashAttention  │  │     KV-Cache Manager      │  │
│  │    (v2)         │  │   (CUDA/CPU)    │  │   (Cuantizado + Paging)   │  │
│  │                 │  │                 │  │                           │  │
│  │ • naive/simd/   │  │ • Kernel CUDA   │  │ • Bloques 256 tokens      │  │
│  │   tiled/parallel│  │   v1/v2         │  │ • FP16/INT8/INT4/Q4_0     │  │
│  │ • openblas      │  │ • Online softmax│  │ • GQA/MQA nativo          │  │
│  │ • cuBLAS async  │  │ • Warp reduce   │  │ • Prefix caching          │  │
│  │ • GemmEx f16/   │  │ • CPU ref O(N²) │  │ • Evicción LRU            │  │
│  │   bf16          │  │                 │  │ • De-cuantización on-fly  │  │
│  │ • Batch/strided │  │                 │  │                           │  │
│  │ • INT8/INT4     │  │                 │  │                           │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Módulos y Sus Responsabilidades

### 2.1 Matmul Engine v2 (`src/matmul/`)

**Responsabilidad:** Todas las multiplicaciones de matrices del modelo.

| Función | Uso en Transformer | Backend recomendado |
|---------|-------------------|---------------------|
| `linearProjection()` | Q, K, V, O, FFN gate/up/out | cuBLAS (GPU) / parallel (CPU) |
| `gemmBatch()` | Batch de secuencias | cuBLAS |
| `gemmStrided()` | GQA (heads compartidos) | cuBLAS |
| `gemmExF16()` | Proyecciones con Tensor Cores | cuBLAS + Hopper+ |
| `gemmQuantized()` | FFN con pesos INT8/INT4 | CPU (cuBLAS Lt para INT8 GPU) |

**Re-exporta:** `QuantConfig`, `QuantizedTensor`, `tensorF32ToF16`, `BF16`, etc.

### 2.2 FlashAttention (`src/fa/`)

**Responsabilidad:** Computar atención con complejidad O(N) en memoria.

| Componente | Rol |
|-----------|-----|
| `FlashAttention` (GPU) | Kernel CUDA con online softmax, warp/block reduce |
| `FlashAttentionCpu` | Referencia O(N²) para validación y fallback |
| `fa_utils` | RoPE, softmax, online softmax, inicialización |
| `fa_kernels` | Launchers CUDA con config de grid/block/smem |

### 2.3 KV-Cache Cuantizado (`src/kv_cache/`)

**Responsabilidad:** Almacenar K y V de todas las capas y secuencias con compresión.

| Componente | Rol |
|-----------|-----|
| `KVCacheManager` | Múltiples secuencias, memoria global, stats |
| `KVCacheSequence` | Una secuencia = lista de bloques paginados |
| `KVCacheBlock` | 256 tokens de K+V cuantizados |
| `QuantizedTensor` | Datos crudos + metadatos de escala |
| `quant_ops` | Cuantizar/de-cuantizar entre FP16 y formatos |

---

## 3. Flujo de Datos Detallado

### 3.1 Prefill (primera pasada del prompt)

```
Prompt tokens: [t0, t1, t2, ..., tN-1]

1. Embedding → hidden_state [batch=1, seq=N, hidden_dim]

2. Para cada capa L = 0..num_layers-1:

   a. Proyecciones Q/K/V:
      Q = hidden @ W_Q^T   → MatmulEngine.linearProjection(f16)
      K = hidden @ W_K^T   → MatmulEngine.linearProjection(f16)
      V = hidden @ W_V^T   → MatmulEngine.linearProjection(f16)

   b. RoPE sobre Q y K (posiciones 0..N-1)

   c. Almacenar K, V en KV-Cache:
      kv_cache.append(seq_id, K, V)
      → Cuantiza K/V según formato configurado
      → Almacena en bloques paginados

   d. FlashAttention:
      - Si GPU: forward(Q, K, V, O) con kernels CUDA
      - Si CPU: forward(Q, K, V, O) con referencia

   e. Proyección O:
      attn_out = O_proj @ O  → MatmulEngine.linearProjection(f16)

   f. Residual:
      hidden = hidden + attn_out

   g. FFN (SwiGLU):
      gate = hidden @ W_gate^T
      up   = hidden @ W_up^T
      hidden = hidden + swish(gate) * up @ W_down^T

3. Head (LM Head):
   logits = hidden @ W_lm^T

4. Samplear último token → tN
```

### 3.2 Generación Autoregresiva (token a token)

```
Token nuevo: tN (posición N)

1. Embedding de tN → hidden [batch=1, seq=1, hidden_dim]

2. Para cada capa L:

   a. Proyecciones Q/K/V del ÚNICO token nuevo:
      q_new = hidden @ W_Q^T  → [1, num_heads, 1, head_dim]
      k_new = hidden @ W_K^T  → [1, num_kv_heads, 1, head_dim]
      v_new = hidden @ W_V^T  → [1, num_kv_heads, 1, head_dim]

   b. RoPE sobre q_new y k_new (posición N)

   c. Append al KV-Cache:
      kv_cache.append(seq_id, k_new, v_new)
      → Solo se cuantizan 1 token (rápido)
      → Se añade al último bloque o se crea uno nuevo

   d. Recuperar K_full y V_full del cache:
      - De-cuantizar TODOS los bloques de la secuencia
      - K_full = [1, num_kv_heads, N+1, head_dim]
      - V_full = [1, num_kv_heads, N+1, head_dim]

   e. FlashAttention con q_new y K_full/V_full:
      - q_new atiende a TODAS las posiciones 0..N
      - Mask causal automática (posición N solo ve ≤ N)

   f. Proyección O + Residual + FFN (igual que prefill)

3. Head → logits → samplear tN+1

4. Repetir hasta EOS o max_tokens
```

### 3.3 Batch de Secuencias (múltiples prompts)

```
Secuencias: [seq_0, seq_1, ..., seq_B-1]

1. Cada secuencia tiene su propio KVCacheSequence
   → Mismo manager, IDs distintos

2. Prefill puede ser paralelo (batch GEMM):
   - Concatenar tokens de todas las secuencias
   - O procesar una a una si longitudes muy distintas

3. En generación, cada secuencia avanza a su ritmo:
   - seq_0 puede tener 100 tokens, seq_1 solo 10
   - KV-Cache independiente por secuencia
   - Bloques no compartidos (a menos que prefix caching)

4. Prefix Caching (optimización):
   - Si dos secuencias comparten prefijo (system prompt):
   - Los bloques del prefijo se comparten (read-only)
   - Copy-on-write si una secuencia modifica
```

---

## 4. Cambios en el Código Existente

### 4.1 `src/transformer/layer.zig` — Integrar KV-Cache cuantizado

**ANTES (actual):**
```zig
pub const TransformerLayer = struct {
    // ... pesos, FA engine, buffers ...

    pub fn forward(self: *Self, hidden_state: Tensor(f16), output: *Tensor(f16)) !void {
        // 1. Proyecciones Q, K, V
        try self.projectQ(X_2d);
        try self.projectK(X_2d);
        try self.projectV(X_2d);

        // 2. RoPE
        fa_utils.applyRoPE(&self.q_proj, &self.k_proj, 0, self.head_dim);

        // 3. FlashAttention
        try self.fa_engine.forward(self.q_proj, self.k_proj, self.v_proj, &self.attn_out);

        // 4. O-proj + residual
        // ...
    }
};
```

**DESPUÉS (con KV-Cache cuantizado):**
```zig
const kvcache = @import("kv_cache");

pub const TransformerLayer = struct {
    // ... pesos, FA engine, buffers ...

    // NUEVO: Referencia al KV-Cache manager y secuencia
    kv_manager: ?*kvcache.KVCacheManager = null,
    seq_id: usize = 0,
    layer_idx: usize,  // Ya existe, ahora se usa para indexar cache por capa

    pub fn forward(
        self: *Self,
        hidden_state: Tensor(f16),
        output: *Tensor(f16),
        position: usize,        // Posición actual (0 en prefill, N en generación)
        is_prefill: bool,       // true = primer paso, false = autoregresivo
    ) !void {
        const batch_size = hidden_state.shape[0];
        const seq_len = hidden_state.shape[1];

        // 1. Proyecciones Q, K, V
        try self.projectQ(X_2d);
        try self.projectK(X_2d);
        try self.projectV(X_2d);

        // 2. RoPE (posición relativa al inicio de la secuencia)
        fa_utils.applyRoPE(&self.q_proj, &self.k_proj, position, self.head_dim);

        // 3. KV-CACHE: almacenar K, V cuantizados
        if (self.kv_manager) |mgr| {
            // Obtener la secuencia
            var seq = &mgr.sequences.items[self.seq_id];

            // Cuantizar y append
            try seq.append(self.k_proj, self.v_proj);
        }

        // 4. Reconstruir K_full y V_full desde el cache
        var k_full: Tensor(f16) = undefined;
        var v_full: Tensor(f16) = undefined;

        if (self.kv_manager) |mgr| {
            const seq = mgr.sequences.items[self.seq_id];
            const total_len = seq.current_len;

            // Allocar tensores para K_full y V_full
            k_full = try Tensor(f16).alloc(self.allocator, &.{
                batch_size, self.num_heads, total_len, self.head_dim
            });
            v_full = try Tensor(f16).alloc(self.allocator, &.{
                batch_size, self.num_heads, total_len, self.head_dim
            });

            // De-cuantizar TODO el cache
            try seq.dequantizeFull(&k_full, &v_full);
        } else {
            // Fallback: usar K, V locales (sin cache)
            k_full = self.k_proj;
            v_full = self.v_proj;
        }

        // 5. FlashAttention con K_full y V_full
        try self.fa_engine.forward(self.q_proj, k_full, v_full, &self.attn_out);

        // 6. Liberar K_full/V_full si vinieron del cache
        if (self.kv_manager != null) {
            k_full.deinit();
            v_full.deinit();
        }

        // 7. O-proj + residual + FFN
        // ... (igual que antes)
    }
};
```

### 4.2 `src/main.zig` — Pipeline de generación

**ANTES:**
```zig
pub fn main() !void {
    var layer = try TransformerLayer.init(...);
    try layer.forward(hidden_state, &output);
}
```

**DESPUÉS:**
```zig
pub fn main() !void {
    // 1. Configurar KV-Cache
    const kv_config = kvcache.KVCacheConfig{
        .num_layers = 32,
        .num_heads = 32,
        .num_kv_heads = 8,        // GQA: 8 heads KV para 32 heads Q
        .head_dim = 128,
        .max_seq_len = 32768,
        .block_size = 256,
        .k_format = .q4_0,        // K en Q4_0 (GGUF)
        .v_format = .int8_symmetric, // V en INT8 (menos pérdida que K)
    };

    var kv_mgr = try kvcache.KVCacheManager.init(allocator, kv_config);
    defer kv_mgr.deinit();

    const seq_id = try kv_mgr.createSequence();

    // 2. Inicializar capas con referencia al KV-Cache
    var layers = try allocator.alloc(TransformerLayer, kv_config.num_layers);
    defer allocator.free(layers);

    for (0..kv_config.num_layers) |l| {
        layers[l] = try TransformerLayer.init(allocator, l, fa_config, ptx_path, hidden_dim, precision);
        layers[l].kv_manager = &kv_mgr;
        layers[l].seq_id = seq_id;
    }
    defer for (0..kv_config.num_layers) |l| layers[l].deinit();

    // 3. Prefill del prompt
    const prompt = "El futuro de la IA es";
    const prompt_tokens = try tokenize(prompt);

    var hidden = try embedding_lookup(prompt_tokens);

    for (0..kv_config.num_layers) |l| {
        var layer_output = try Tensor(f16).alloc(allocator, hidden.shape);
        defer layer_output.deinit();
        try layers[l].forward(hidden, &layer_output, 0, true);
        hidden = layer_output;  // En realidad se reutiliza buffer
    }

    // 4. Generación autoregresiva
    var generated = std.ArrayList(u32).init(allocator);
    defer generated.deinit();

    var current_pos = prompt_tokens.len;

    for (0..max_new_tokens) |_| {
        // Forward de UNA capa con UN token
        var token_hidden = try embedding_lookup(&[_]u32{last_token});

        for (0..kv_config.num_layers) |l| {
            var layer_output = try Tensor(f16).alloc(allocator, token_hidden.shape);
            defer layer_output.deinit();
            try layers[l].forward(token_hidden, &layer_output, current_pos, false);
            token_hidden = layer_output;
        }

        // Samplear
        const next_token = sampleGreedy(token_hidden);
        try generated.append(next_token);
        current_pos += 1;

        if (next_token == EOS_TOKEN) break;
    }

    // 5. Stats
    std.log.info("Tokens generados: {d}", .{generated.items.len});
    std.log.info("Memoria KV-Cache: {d:.2} MB (FP16 sería {d:.2} MB)", .{
        kv_mgr.totalMemoryMB(),
        kv_mgr.totalMemoryFp16MB(),
    });
    std.log.info("Ratio compresión: {d:.2}x", .{kv_mgr.globalCompressionRatio()});
}
```

### 4.3 `build.zig` — Añadir módulo kv_cache

```zig
// === Módulo kv_cache ===
const kv_cache_mod = b.addModule("kv_cache", .{
    .root_source_file = b.path("src/kv_cache/kv_cache.zig"),
    .target = target,
    .optimize = optimize,
});
kv_cache_mod.addImport("core", core_mod);
kv_cache_mod.addImport("quant_types", quant_types_mod);

// === Módulo transformer (actualizado) ===
transformer_mod.addImport("kv_cache", kv_cache_mod);

// === Ejecutable ===
exe.root_module.addImport("kv_cache", kv_cache_mod);
```

---

## 5. API de Integración

### 5.1 Inicialización

```zig
const matmul = @import("matmul");
const fa = @import("fa");
const transformer = @import("transformer");
const kvcache = @import("kv_cache");

// 1. Motor matmul
var engine = try matmul.MatmulEngine.init(allocator, .cublas, .f16);
defer engine.deinit();

// 2. Configuración FlashAttention
const fa_config = fa.fa_config.FlashAttentionConfig{
    .N = 512, .d = 128, .num_heads = 8,
    .batch_size = 1, .dtype = .f16, .causal = true,
};

// 3. Configuración KV-Cache
const kv_config = kvcache.KVCacheConfig{
    .num_layers = 32,
    .num_heads = 32,
    .num_kv_heads = 8,     // GQA
    .head_dim = 128,
    .max_seq_len = 32768,
    .block_size = 256,
    .k_format = .q4_0,
    .v_format = .int8_symmetric,
};

var kv_mgr = try kvcache.KVCacheManager.init(allocator, kv_config);
defer kv_mgr.deinit();
```

### 5.2 Prefill

```zig
/// Procesa el prompt completo en una pasada
pub fn prefill(
    layers: []TransformerLayer,
    kv_mgr: *kvcache.KVCacheManager,
    prompt_tokens: []const u32,
    embeddings: Tensor(f16),  // [1, prompt_len, hidden_dim]
) !Tensor(f16) {
    const seq_id = try kv_mgr.createSequence();

    var hidden = embeddings;

    for (layers, 0..) |*layer, l| {
        layer.kv_manager = kv_mgr;
        layer.seq_id = seq_id;

        var output = try Tensor(f16).alloc(allocator, hidden.shape);
        defer if (l < layers.len - 1) output.deinit();

        try layer.forward(hidden, &output, 0, true);
        hidden = output;
    }

    return hidden;
}
```

### 5.3 Generación de un token

```zig
/// Genera el siguiente token dado el último
pub fn generateToken(
    layers: []TransformerLayer,
    kv_mgr: *kvcache.KVCacheManager,
    last_token_embedding: Tensor(f16),  // [1, 1, hidden_dim]
    position: usize,
) !Tensor(f16) {
    var hidden = last_token_embedding;

    for (layers) |*layer| {
        var output = try Tensor(f16).alloc(allocator, hidden.shape);
        defer output.deinit();

        try layer.forward(hidden, &output, position, false);
        hidden = output;
    }

    return hidden;
}
```

### 5.4 Cuantización de Pesos del Modelo

```zig
/// Cuantiza todos los pesos de proyección a INT8 para CPU
pub fn quantizeModelWeights(
    allocator: std.mem.Allocator,
    layers: []TransformerLayer,
    config: matmul.QuantConfig,
) !void {
    for (layers) |*layer| {
        // Cuantizar W_Q, W_K, W_V, W_O
        if (layer.w_q_t) |w| {
            layer.w_q_t_q = try matmul.quantizeInt8PerChannel(allocator, w, config);
        }
        if (layer.w_k_t) |w| {
            layer.w_k_t_q = try matmul.quantizeInt8PerChannel(allocator, w, config);
        }
        // ... etc
    }
}
```

---

## 6. Optimizaciones Avanzadas

### 6.1 De-cuantización en GPU (FlashAttention kernel)

En lugar de de-cuantizar en host y hacer HtoD de K_full/V_full:

```
OPCIÓN A (actual): Host de-cuantiza → cudaMemcpy HtoD → FA kernel
OPCIÓN B (óptima):  Pasar K_cache cuantizado (device ptr) al kernel
                    → El kernel de-cuantiza en shared memory tile por tile
                    → Menor ancho de banda HBM, más compute (ganancia en GPU)
```

**Implementación kernel CUDA modificada:**
```cuda
// En flash_attention_v2.cu, añadir:
__global__ void flash_attention_v2_dequant_kernel(
    const __half* Q,
    const uint8_t* K_q,      // K cuantizado INT4/INT8
    const uint8_t* V_q,      // V cuantizado
    const float* k_scales,   // Escalas por bloque
    const float* v_scales,
    __half* O,
    int N, int d, float scale, bool causal
) {
    // Al cargar tile de K/V desde HBM:
    // 1. Leer bytes cuantizados
    // 2. De-cuantizar a f16 en shared memory
    // 3. Computar atención con valores f16
}
```

### 6.2 PagedAttention (vLLM-style)

```zig
// En lugar de bloques fijos por secuencia:
// - Bloques de 256 tokens en un pool global
// - Secuencias son listas de punteros a bloques
// - Bloques no utilizados se evictan a CPU/disco

pub const PagedKVCache = struct {
    block_pool: std.ArrayList(KVCacheBlock),  // Todos los bloques
    free_blocks: std.ArrayList(usize),         // Índices libres
    block_table: std.AutoHashMap(usize, []usize), // seq_id -> [block_idx, ...]

    pub fn allocateBlock(self: *Self) !usize {
        if (self.free_blocks.items.len > 0) {
            return self.free_blocks.pop();
        }
        // Evict LRU si necesario
        try self.evictLRU();
        return self.free_blocks.pop();
    }
};
```

### 6.3 Continuous Batching

```zig
// Múltiples secuencias en el mismo batch, cada una en su fase:
// - Seq A: prefill de 100 tokens
// - Seq B: generación, token 50
// - Seq C: generación, token 200

// En cada iteración:
// 1. Agrupar secuencias que necesitan forward
// 2. Para prefill: batch GEMM de todas las secuencias en prefill
// 3. Para generación: si hay múltiples, batch sus q_new
// 4. FlashAttention con K/V de todas las secuencias (atención cruzada no, self-attention por secuencia)
```

### 6.4 Speculative Decoding

```zig
// 1. Modelo pequeño (draft) genera 4 tokens candidatos
// 2. Modelo grande (target) verifica todos de una vez con batch
// 3. Aceptar los que coinciden, rechazar el resto

// Requiere:
// - KV-Cache separado para draft y target
// - Batch forward del target con 4 posiciones
// - FlashAttention con Q = [4, 1, d] y K/V = cache completo
```

---

## 7. Esquema de Directorios Final

```
zig-ai-engine/
├── build.zig
├── build.zig.zon
├── README.md
│
├── src/
│   ├── main.zig                    # CLI / demo
│   ├── tensor.zig                  # Tensor(T) core
│   │
│   ├── matmul/                     # Motor matmul v2
│   │   ├── root.zig                # MatmulEngine (API pública)
│   │   ├── types.zig               # Layout, TileConfig, SimdInfo, Timer
│   │   ├── naive.zig               # GEMM naive
│   │   ├── simd.zig                # GEMM @Vector
│   │   ├── tiled.zig               # GEMM tiled
│   │   ├── parallel.zig            # GEMM multi-hilo
│   │   ├── openblas.zig            # FFI OpenBLAS
│   │   ├── cublas.zig              # FFI cuBLAS (async, batch, strided, GemmEx)
│   │   ├── f16bf16.zig             # Conversión FP16/BF16
│   │   └── quant.zig               # Cuantización INT8/INT4
│   │
│   ├── fa/                         # FlashAttention
│   │   ├── flash_attention.zig     # Motor FA (GPU + CPU ref)
│   │   ├── fa_config.zig           # Configuración y validación
│   │   ├── fa_utils.zig            # RoPE, softmax, online softmax
│   │   └── fa_kernels.zig          # Launchers CUDA
│   │
│   ├── transformer/                # Capa transformer
│   │   ├── layer.zig               # TransformerLayer + KVCache simple
│   │   ├── model.zig               # Modelo completo (N capas)
│   │   └── pipeline.zig            # Prefill + generación
│   │
│   ├── kv_cache/                   # KV-Cache cuantizado (NUEVO)
│   │   ├── quant_types.zig         # QuantFormat, QuantizedTensor
│   │   ├── quant_ops.zig           # Cuantizar/de-cuantizar
│   │   ├── kv_cache.zig            # KVCacheBlock, Sequence, Manager
│   │   ├── paged.zig               # PagedKVCache (vLLM-style)
│   │   └── fa_integration.zig      # De-cuantización en FA
│   │
│   ├── cuda/
│   │   └── cudaz_stub.zig          # Bindings CUDA Driver API
│   │
│   ├── tokenizer/
│   │   └── bpe.zig                 # Tokenizador BPE (futuro)
│   │
│   └── utils/
│       └── sampling.zig            # Greedy, top-k, top-p
│
├── cuda/                           # Kernels CUDA
│   ├── online_softmax.cuh
│   ├── matmul_utils.cuh
│   ├── flash_attention.cu          # FA v1
│   ├── flash_attention_v2.cu       # FA v2
│   └── flash_attention_dequant.cu  # FA con de-cuantización en kernel (futuro)
│
├── tests/
│   ├── test_tensor.zig
│   ├── test_matmul.zig
│   ├── test_flash_attention.zig
│   ├── test_online_softmax.zig
│   ├── test_transformer.zig
│   ├── test_kv_cache.zig           # Tests KV-Cache (NUEVO)
│   ├── test_quant.zig              # Tests cuantización
│   └── benchmark.zig
│
├── scripts/
│   ├── convert_weights.py          # Convierte checkpoints a formato .bin
│   └── quantize_model.py           # Cuantiza pesos offline
│
└── models/
    └── llama-3-8b/                 # Pesos del modelo
        ├── config.json
        ├── tokenizer.json
        └── weights/
            ├── layer.0.self_attn.q_proj.weight_t.bin
            ├── layer.0.self_attn.k_proj.weight_t.bin
            └── ...
```

---

## 8. Matriz de Compatibilidad

| Feature | CPU (sin CUDA) | CPU (OpenBLAS) | GPU (cuBLAS) | GPU (kernels propios) |
|---------|---------------|----------------|--------------|----------------------|
| Matmul naive | ✅ | ✅ | ✅ | ✅ |
| Matmul SIMD | ✅ | ✅ | ✅ | ✅ |
| Matmul tiled | ✅ | ✅ | ✅ | ✅ |
| Matmul parallel | ✅ | ✅ | ✅ | ✅ |
| Matmul OpenBLAS | — | ✅ | — | — |
| Matmul cuBLAS | — | — | ✅ | — |
| Matmul GemmEx | — | — | ✅ | — |
| FlashAttention CPU | ✅ | ✅ | ✅ | ✅ |
| FlashAttention GPU | — | — | — | ✅ |
| KV-Cache FP16 | ✅ | ✅ | ✅ | ✅ |
| KV-Cache INT8 | ✅ | ✅ | ✅ | ✅ |
| KV-Cache INT4 | ✅ | ✅ | ✅ | ✅ |
| KV-Cache Q4_0 | ✅ | ✅ | ✅ | ✅ |
| De-cuantización GPU | — | — | — | 🚧 |
| PagedAttention | ✅ | ✅ | ✅ | ✅ |
| Continuous Batching | ✅ | ✅ | ✅ | ✅ |

---

## 9. Roadmap de Implementación

### Fase 1: Fundación (completado)
- [x] Tensor multidimensional
- [x] Matmul 6 backends
- [x] FlashAttention CPU + CUDA
- [x] Capa transformer básica

### Fase 2: KV-Cache Cuantizado (actual)
- [ ] Tipos de cuantización (FP16, INT8, INT4, Q4_0, Q8_0)
- [ ] Paging por bloques
- [ ] GQA/MQA
- [ ] Integración con TransformerLayer
- [ ] Pipeline prefill + generación

### Fase 3: Optimización GPU
- [ ] De-cuantización en kernel CUDA
- [ ] FlashAttention con K/V cuantizados en shared memory
- [ ] PagedAttention v2 (vLLM)
- [ ] Continuous batching

### Fase 4: Producción
- [ ] Tokenizador BPE/SentencePiece
- [ ] Carga de checkpoints (GGUF, Safetensors)
- [ ] Sampling (top-k, top-p, temperature)
- [ ] Speculative decoding
- [ ] Quantization-Aware Training (QAT)

---

## 10. Referencias

- **FlashAttention-2:** Dao, Tri. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." ICLR 2024.
- **vLLM:** Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023.
- **GGUF:** Format specification: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- **Q4_0/Q8_0:** llama.cpp quantization schemes
- **GQA:** Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." EMNLP 2023.
- **Zig:** https://ziglang.org/documentation/0.13.0/

---

*Documento generado en agosto 2026.*
*Para dudas o contribuciones: revisar el repositorio zig-ai-engine.*
