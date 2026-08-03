# Plan de Desarrollo: Zig AI Engine

> **Documento vivo de planificación** — Agosto 2026  
> **Proyecto:** Motor de inferencia de LLM en Zig con FlashAttention, KV-Cache cuantizado y multi-backend  
> **Última actualización:** 2026-08-02

---

## 1. Estado Actual del Proyecto

### 1.1 Inventario de Componentes Implementados

| Componente | Archivo(s) | Estado | Complejidad |
|------------|-----------|--------|-------------|
| **Tensor multidimensional** | `src/tensor.zig` | ✅ Completo | Core |
| **Matmul Engine v2** | `src/matmul/` | ✅ Completo | Core |
| ├─ Naive GEMM | `naive.zig` | ✅ | |
| ├─ SIMD GEMM | `simd.zig` | ✅ | |
| ├─ Tiled GEMM | `tiled.zig` | ✅ | |
| ├─ Parallel GEMM | `parallel.zig` | ✅ | |
| ├─ OpenBLAS FFI | `openblas.zig` | ✅ | |
| ├─ cuBLAS FFI | `cublas.zig` | ✅ | |
| ├─ FP16/BF16 utils | `f16bf16.zig` | ✅ | |
| └─ Cuantización INT8/INT4 | `quant.zig` | ✅ | |
| **FlashAttention** | `src/fa/` | ✅ GPU + CPU | Core |
| ├─ Kernel CUDA v1 | `flash_attention.cu` | ✅ | |
| ├─ Kernel CUDA v2 | `flash_attention_v2.cu` | ✅ | |
| ├─ CPU reference | `flash_attention.zig` | ✅ | |
| ├─ Launchers CUDA | `fa_kernels.zig` | ✅ | |
| ├─ Config/validación | `fa_config.zig` | ✅ | |
| └─ RoPE, softmax, utils | `fa_utils.zig` | ✅ | |
| **Transformer Layer** | `src/transformer/layer.zig` | ✅ Base | Core |
| ├─ Proyecciones Q/K/V/O | | ✅ | |
| ├─ FlashAttention integration | | ✅ | |
| ├─ Pesos cuantizados (CPU) | | ✅ | |
| ├─ Carga de pesos desde disco | | ✅ | |
| └─ KV-Cache simple (FP16) | `layer.zig` (KVCache) | ✅ Básico | |
| **CUDA Bindings** | `src/cuda/cudaz_stub.zig` | ✅ Driver API | Infra |
| **Build System** | `build.zig` / `build.zig.zon` | ✅ Auto-detect CUDA | Infra |
| **Tests** | `tests/` | ✅ 30+ tests | QA |
| **Benchmarks** | `tests/benchmark.zig` | ✅ Matmul + FA CPU | QA |

### 1.2 Métricas de Cobertura

```
Core Engine:        ████████████████████░░░░░  80%
GPU/CUDA:           ██████████████░░░░░░░░░░░  56%
KV-Cache:           ██████░░░░░░░░░░░░░░░░░░░  25%
Tokenizer:          ░░░░░░░░░░░░░░░░░░░░░░░░░   0%
Model Loader:       ░░░░░░░░░░░░░░░░░░░░░░░░░   0%
Pipeline Completo:  ████░░░░░░░░░░░░░░░░░░░░░  15%
Optimizaciones:     ███░░░░░░░░░░░░░░░░░░░░░░  12%
```

---

## 2. Gap Analysis — Qué Falta

### 2.1 Crítico (Bloqueante para inferencia real)

| # | Gap | Impacto | Bloquea |
|---|-----|---------|---------|
| G1 | **KV-Cache cuantizado con paging** | Memoria OOM en secuencias largas | Inferencia >4K tokens |
| G2 | **GQA / MQA nativo** | KV-Cache 4-8x más grande de lo necesario | Modelos modernos (Llama-3, Qwen) |
| G3 | **Generación autoregresiva** | No hay pipeline de token-a-token | Chat/completion |
| G4 | **Tokenizer BPE/SentencePiece** | Sin tokenización no hay entrada de texto | Todo el pipeline |
| G5 | **Carga de checkpoints** | Pesos dummy solo, no modelos reales | Cualquier demo útil |

### 2.2 Alto (Performance y usabilidad)

| # | Gap | Impacto |
|---|-----|---------|
| G6 | De-cuantización en kernel CUDA | Evita H2D memcpy por token |
| G7 | PagedAttention (vLLM-style) | Múltiples secuencias concurrentes |
| G8 | Continuous batching | Throughput en servicio |
| G9 | Sliding window attention | Secuencias infinitas |
| G10 | Sampling (top-k, top-p, temperature) | Calidad de generación |

### 2.3 Medio (Features avanzadas)

| # | Gap | Impacto |
|---|-----|---------|
| G11 | Speculative decoding | Latencia 2-3x menor |
| G12 | Prefix caching | Reutilización de system prompts |
| G13 | Multi-GPU sharding | Modelos >70B |
| G14 | Quantization-Aware Training | Mejor calidad INT4 |
| G15 | Tensor parallelism | Modelos grandes en cluster |

---

## 3. Plan de Trabajo por Fases

### Fase 1: Fundación de Inferencia (Semanas 1-2)
**Objetivo:** Primer forward end-to-end con modelo real pequeño

- [ ] **1.1** Implementar KV-Cache cuantizado (`src/kv_cache/`)
  - [ ] Tipos de cuantización: Q4_0, Q8_0, INT8 sym/asym, INT4
  - [ ] Pool allocator con estrategias bump/free_list/lru_evict
  - [ ] GpuDequantEngine (PTX kernels de de-cuantización)
  - [ ] KVCacheManager (múltiples secuencias, métricas)
  - [ ] Tests unitarios para cada formato

- [ ] **1.2** Integrar KV-Cache cuantizado en TransformerLayer
  - [ ] Modificar `forward()` para aceptar `position` e `is_prefill`
  - [ ] Almacenar K/V cuantizados tras proyección
  - [ ] Recuperar y de-cuantizar para FlashAttention
  - [ ] Liberar tensores temporales correctamente

- [ ] **1.3** GQA / MQA nativo
  - [ ] `num_kv_heads` en config (actualmente `num_heads` para K/V)
  - [ ] Broadcast de K/V heads en FlashAttention
  - [ ] Ajustar shapes en proyecciones K/V

- [ ] **1.4** Pipeline de generación autoregresiva
  - [ ] Función `prefill()` — procesa prompt completo
  - [ ] Función `generateToken()` — un token nuevo
  - [ ] Bucle de generación con límite de tokens / EOS
  - [ ] Gestión de `current_pos` por secuencia

**Entregable:** Demo que carga un modelo pequeño (ej. TinyLlama) y genera texto.

---

### Fase 2: Tokenización y Carga de Modelos (Semanas 2-3)
**Objetivo:** Input de texto real y carga de checkpoints

- [ ] **2.1** Tokenizador BPE
  - [ ] Cargar `tokenizer.json` (formato HuggingFace)
  - [ ] Merge rules y vocabulario
  - [ ] Funciones `encode()` y `decode()`
  - [ ] Soporte de tokens especiales (BOS, EOS, PAD)

- [ ] **2.2** Carga de checkpoints
  - [ ] Parser de Safetensors (formato prioritario)
  - [ ] Parser de GGUF (formato cuantizado nativo)
  - [ ] Mapeo de tensores a estructura del modelo
  - [ ] Conversión de layouts (PyTorch → Zig)

- [ ] **2.3** Configuración del modelo
  - [ ] Parser de `config.json`
  - [ ] Auto-configuración de capas, heads, dims
  - [ ] Detección de arquitectura (Llama, Qwen, Mistral)

**Entregable:** CLI que recibe un prompt y genera texto usando un modelo descargado.

---

### Fase 3: Optimización GPU (Semanas 3-4)
**Objetivo:** Latencia competitiva con llama.cpp en GPU

- [ ] **3.1** De-cuantización en kernel CUDA
  - [ ] Kernel `flash_attention_dequant.cu`
  - [ ] Cargar K/V cuantizado directamente en shared memory
  - [ ] De-cuantizar tile-by-tile dentro del kernel FA
  - [ ] Eliminar H2D de tensores FP16 completos

- [ ] **3.2** PagedAttention (vLLM-style)
  - [ ] Bloques de 256 tokens en pool global
  - [ ] Block table por secuencia
  - [ ] Evicción LRU con fallback a CPU RAM

- [ ] **3.3** Overlap compute/memcpy
  - [ ] StreamRing para múltiples streams CUDA
  - [ ] Prefetch de capa N+1 mientras se computa N
  - [ ] Async sin `cuStreamSynchronize` por token

- [ ] **3.4** K/V persistentes en GPU
  - [ ] Almacenar cache cuantizado en device ptr
  - [ ] Evitar copias host→device en cada token

**Entregable:** Benchmark vs llama.cpp en A100/RTX 4090.

---

### Fase 4: Sampling y Calidad (Semanas 4-5)
**Objetivo:** Generación de texto de calidad comparable a transformers

- [ ] **4.1** Estrategias de sampling
  - [ ] Greedy decoding
  - [ ] Temperature scaling
  - [ ] Top-k sampling
  - [ ] Nucleus sampling (top-p)
  - [ ] Repetition penalty

- [ ] **4.2** Continuous batching
  - [ ] Múltiples secuencias en mismo batch
  - [ ] Cada secuencia avanza a su ritmo
  - [ ] Agrupación dinámica de prefill vs generation

- [ ] **4.3** Streaming de tokens
  - [ ] Callback por token generado
  - [ ] Flush parcial de buffer
  - [ ] Cancelación de generación

**Entregable:** API de servidor HTTP/WebSocket con streaming.

---

### Fase 5: Features Avanzadas (Semanas 5-8)
**Objetivo:** Paridad de features con vLLM / TGI

- [ ] **5.1** Speculative decoding
  - [ ] Modelo draft pequeño (mismo árbol de capas)
  - [ ] Verificación en batch por el modelo target
  - [ ] Aceptación/rechazo de tokens candidatos

- [ ] **5.2** Prefix caching
  - [ ] Hash de bloques de K/V
  - [ ] Compartición read-only entre secuencias
  - [ ] Copy-on-write

- [ ] **5.3** Sliding window attention
  - [ ] Atención limitada a últimos N tokens
  - [ ] Evicción automática de tokens antiguos
  - [ ] Ring buffer para K/V

- [ ] **5.4** Multi-GPU
  - [ ] Pipeline parallelism (capas en GPUs distintas)
  - [ ] Tensor parallelism (sharding de atención)
  - [ ] NCCL para comunicación GPU-GPU

- [ ] **5.5** Quantization-Aware Training (QAT)
  - [ ] Simulación de cuantización en forward de entrenamiento
  - [ ] Export a GGUF con calidad mejorada

**Entregable:** Servidor de producción comparable a vLLM.

---

## 4. Arquitectura Objetivo Final

```
┌─────────────────────────────────────────────────────────────────────┐
│                         API / CLI / Servidor                         │
│  (HTTP, WebSocket, gRPC, CLI interactivo)                           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Pipeline de Generación                            │
│  Tokenize → Prefill → Autoregressive Loop → Detokenize → Stream     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Transformer Model (N capas)                       │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌──────────────────────┐  │
│  │ Layer 0 │→ │ Layer 1 │→ │  ...   │→ │     Layer N-1       │  │
│  │ + KVCache│  │ + KVCache│  │         │  │  + KVCache          │  │
│  └─────────┘  └─────────┘  └─────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Motores de Cómputo                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Matmul Engine│  │ FlashAttention│  │   KV-Cache Manager      │  │
│  │ (cuBLAS/CPU) │  │ (CUDA/CPU)   │  │  (Cuantizado + Paged)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Infraestructura                                   │
│  CUDA Driver API  │  Memory Pool GPU  │  Stream Ring  │  Scheduler │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Decisiones Técnicas Pendientes

| Decisión | Opciones | Recomendación | Justificación |
|----------|----------|---------------|---------------|
| **Formato de pesos** | Safetensors vs GGUF vs ambos | **GGUF primero** | Cuantización nativa, ecosistema llama.cpp |
| **Tokenizer** | Implementar vs binding a C++ | **Implementar en Zig** | Menos dependencias, control total |
| **Servidor HTTP** | Zig http vs binding a C | **Zig std http** | Sin deps externas, suficiente para API |
| **Multi-GPU comm** | NCCL vs manual | **NCCL bindings** | Estándar de facto, optimizado |
| **QAT** | En Zig vs Python export | **Python export** | Ecosistema de entrenamiento en Python |

---

## 6. Próximos Pasos Inmediatos (Hoy)

### Prioridad P0 — Comenzar ahora

1. **Crear módulo `src/kv_cache/`**
   - `quant_types.zig` — enums y structs de cuantización
   - `kv_cache.zig` — KVCacheBlock, KVCacheSequence, KVCacheManager
   - `gpu_dequant.zig` — GpuDequantEngine con PTX

2. **Modificar `src/transformer/layer.zig`**
   - Añadir `kv_manager: ?*KVCacheManager`
   - Añadir parámetros `position: usize` e `is_prefill: bool` a `forward()`
   - En prefill: cuantizar y append K/V
   - En generation: recuperar K_full/V_full, de-cuantizar, pasar a FA

3. **Actualizar `src/main.zig`**
   - Inicializar KVCacheManager con config
   - Implementar bucle `prefill()` → `generateToken()`
   - Medir memoria usada vs FP16 baseline

### Prioridad P1 — Esta semana

4. Implementar tokenizer BPE básico
5. Parser de config.json para auto-configuración
6. Script Python para convertir HF checkpoints a formato .bin del engine

---

## 7. Métricas de Éxito

| Métrica | Actual | Fase 1 | Fase 2 | Fase 3 | Fase 5 |
|---------|--------|--------|--------|--------|--------|
| Tokens/seg (GPU) | N/A | 10 | 50 | 200 | 500+ |
| Memoria KV (vs FP16) | 100% | 50% | 50% | 25% | 12.5% |
| Secuencias concurrentes | 1 | 1 | 1 | 10 | 100+ |
| Modelo más grande | N/A | 1B | 7B | 70B | 405B |
| Latencia TTFT | N/A | <1s | <500ms | <200ms | <100ms |

> **TTFT** = Time To First Token

---

## 8. Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|-------------|---------|------------|
| CUDA FFI inestable en Zig | Media | Alto | Fallback a CPU robusto, tests CI |
| Memoria fragmentada en KV-Cache | Media | Alto | Compactación periódica, bump allocator para single-seq |
| Precisión INT4 insuficiente | Baja | Medio | Config por capa (capas finales INT4, iniciales INT8) |
| Falta de modelos de test | Baja | Medio | Usar TinyLlama (1.1B), Phi-2 (2.7B) |
| Compilación Zig lenta con CUDA | Media | Bajo | Cache de PTX/CUBIN, build incremental |

---

## 9. Referencias y Recursos

- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)
- [vLLM PagedAttention](https://arxiv.org/abs/2309.06180)
- [GGUF Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Llama.cpp KV-Cache](https://github.com/ggerganov/llama.cpp/blob/master/src/llama-kv-cache.cpp)
- [Zig Language Reference](https://ziglang.org/documentation/0.13.0/)
- [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/)

---

*Plan generado el 2026-08-02. Actualizar tras completar cada fase.*
