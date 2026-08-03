# Análisis del Ecosistema Zig AI/ML para zig-ai-engine

> **Fecha:** 2026-08-03  
> **Proyecto:** zig-ai-engine — Fase 1 completada  
> **Objetivo:** Identificar código maduro del ecosistema Zig AI/ML reutilizable para cerrar los gaps críticos del motor.

---

## Índice

1. [Código para reutilizar DIRECTAMENTE](#1-código-para-reutilizar-directamente)
2. [Código para estudiar y portar](#2-código-para-estudiar-y-portar)
3. [Licencias a tener en cuenta](#3-licencias-a-tener-en-cuenta)
4. [Plan de integración priorizado](#4-plan-de-integración-priorizado)
5. [Código propio a deprecar/refactorizar](#5-código-propio-a-deprecarrefactorizar)
6. [Decisiones técnicas](#6-decisiones-técnicas)
7. [Referencias](#7-referencias)

---

## 1. Código para reutilizar DIRECTAMENTE

### 1.1 SMC17/tokenizers-zig — Tokenizer de producción

| Propiedad | Detalle |
|-----------|---------|
| **Repositorio** | github.com/SMC17/tokenizers-zig |
| **Licencia** | AGPL-3.0 |
| **Estado** | Producción. Paridad total con HuggingFace tokenizers. |
| **Formatos** | BPE, WordPiece, Unigram |
| **Rendimiento** | ~5× más rápido que upstream Rust en fixtures de forma Llama |

**Gap que cierra:** Reemplaza `DummyTokenizer` en `src/main.zig`, que asigna `token[i] = i % 256` y no sirve para texto real.

**Integración:**
- Añadir dependencia en `build.zig.zon`.
- Crear módulo `tokenizer`.
- Sustituir `DummyTokenizer.encode/decode` por llamadas a la librería.
- Cargar `tokenizer.json` de cualquier modelo (Llama, Qwen, Mistral, GPT-2).

---

### 1.2 SMC17/safetensors-zig — Cargador de pesos

| Propiedad | Detalle |
|-----------|---------|
| **Repositorio** | github.com/SMC17/safetensors-zig |
| **Licencia** | AGPL-3.0 |
| **Estado** | Producción. Parser puro-Zig de Safetensors. |
| **Rendimiento** | ~5× más rápido que el upstream Rust |

**Gap que cierra:** `loadWeights()` asume archivos `.bin` crudos por capa sin metadatos. Con esto leemos directamente checkpoints de HuggingFace (`model.safetensors`) con nombres de tensores, shapes y dtypes reales.

**Integración:**
- Implementar `loadModelFromSafetensors()` que lea `model.safetensors`.
- Eliminar la necesidad de scripts Python de conversión a `.bin`.
- Mapear nombres de tensores HF (`model.layers.0.self_attn.q_proj.weight`) a nuestra estructura interna.

---

### 1.3 SMC17/vllm-zig — Kernels de inferencia (referencia)

| Propiedad | Detalle |
|-----------|---------|
| **Repositorio** | github.com/SMC17/vllm-zig |
| **Licencia** | AGPL-3.0 |
| **Estado** | Activo. Forward pass real de TinyLlama. |
| **Kernels** | RoPE, GQA, KV cache, matmul SIMD multi-hilo, streaming |

**Gap que cierra:**
- **GQA nativo:** Nuestro `expandGqa()` copia datos en Zig (O(n) extra por token). `vllm-zig` tiene GQA integrado en el forward sin copia.
- **RoPE vectorizado:** Nuestro `applyRoPE` es funcional pero naive. `vllm-zig` usa SIMD.
- **KV cache:** Podemos comparar nuestra arquitectura cuantizada con su KV cache e incorporar ideas (paging, prefix caching).
- **Matmul SIMD multi-hilo:** Referencia para optimizar `src/matmul/parallel.zig`.

**Integración:** No como dependencia directa (incompatibilidad de licencia AGPL vs MIT), sino como **referencia de implementación** para portar kernels a nuestro stack.

---

## 2. Código para estudiar y portar

| Proyecto | Repositorio | Qué aprender/portar | Relevancia |
|----------|-------------|---------------------|------------|
| **LLaMa2.zig** | cgbur/LLaMa2.zig | Implementación pura-Zig de Llama 2 en un solo archivo. FFN, RMSNorm, embedding, sampling. | **Alta** — Cierra gap #3 (FFN+Norm). |
| **zig_gpt2** | EugenHotaj/zig_gpt2 | NanoGPT inference. Pipeline autoregresivo completo. | **Alta** — Cierra gap #6 (pipeline e2e). |
| **Zigrad** | Marco-Christiani/zigrad | Framework DL con autograd. Matmul optimizado (2.5× PyTorch en Apple Silicon, 1.5× x86). | **Media** — Mejora `matmul/` CPU. |
| **ZML** | github.com/zml | Inference stack production. Compila a MLIR/OpenXLA. Soporta AMD, TPU, Trainium. | **Media** — Arquitectura "model-to-metal" como referencia a largo plazo. |
| **zolotukhin/zinc** | zolotukhin/zinc | Inference engine GGUF con Vulkan/Metal. Chat UI + API OpenAI. | **Media** — Referencia para runtime y scheduler de batching. |
| **dnns-from-scratch-in-zig** | SilasMarvin/dnns-from-scratch-in-zig | DNN simple 96% MNIST con solo stdlib. `comptime` para shapes. | **Baja** — Referencia pedagógica para backprop si añadimos training. |

---

## 3. Licencias a tener en cuenta

| Proyecto | Licencia | Implicación |
|----------|----------|-------------|
| **SMC17/*** (vllm-zig, safetensors-zig, tokenizers-zig, faiss-zig) | **AGPL-3.0** | Si se usan como dependencias o se copia código, el proyecto debe ser AGPL-3.0. **No compatible con MIT** sin relicenciar. |
| **Zigrad** | No especificada en fuentes públicas (probablemente MIT/Apache) | Verificar antes de copiar código. |
| **zig_gpt2**, **LLaMa2.zig** | Generalmente MIT | Seguro para referencia y portar. |
| **zig-ai-engine (propio)** | MIT | Mantiene libertad de uso comercial. |

### Estrategias de integración ante AGPL-3.0

| Estrategia | Descripción | Viabilidad |
|------------|-------------|------------|
| **A. Dependencia dinámica** | Usar SMC17 como binarios/librerías dinámicas sin mezclar código fuente. El core permanece MIT. | Viable para tokenizers y safetensors si se compilan como `.so`/`.dll`. |
| **B. Relicenciar a AGPL-3.0** | Convertir zig-ai-engine a AGPL-3.0 y fusionar con vllm-zig. | Posible pero restrictivo para uso comercial cerrado. |
| **C. Reimplementación limpia (recomendada)** | Estudiar los algoritmos de vllm-zig e implementarlos de cero en nuestro estilo/código. Los algoritmos (RoPE, GQA) no están protegidos por copyright, solo la expresión concreta. | **Recomendada.** Mantiene MIT y control total. |

---

## 4. Plan de integración priorizado

### Fase 1: Infraestructura del modelo (esta semana)

| # | Tarea | Código fuente | Gap que cierra |
|---|-------|---------------|----------------|
| 1.1 | Portar/integrar `tokenizers-zig` | `src/tokenizer/bpe.zig` (nuevo) | #5 Tokenizer real |
| 1.2 | Portar/integrar `safetensors-zig` | `src/loader/safetensors.zig` (nuevo) | #4 Cargador de pesos |
| 1.3 | Implementar FFN (SwiGLU) + RMSNorm | `src/transformer/norm.zig` (nuevo), modificar `layer.zig` | #3 FFN+Norm |
| 1.4 | Implementar pipeline autoregresivo robusto | `src/main.zig`, `src/transformer/pipeline.zig` | #6 Pipeline e2e |

### Fase 2: Optimización del motor (semana siguiente)

| # | Tarea | Referencia | Gap que cierra |
|---|-------|------------|----------------|
| 2.1 | Reimplementar GQA sin `expandGqa()` | `vllm-zig` kernels | #1 Shapes dinámicas / memoria |
| 2.2 | Optimizar `applyRoPE()` con SIMD | `vllm-zig` RoPE | Performance RoPE |
| 2.3 | Comparar `KVCacheManager` con `vllm-zig` | `vllm-zig` KV cache | #9 Optimización cache |
| 2.4 | De-cuantización GPU en pipeline FA | `gpu_dequant.zig` + kernels CUDA | #10 GPU dequant |

### Fase 3: Producción (futuro)

| # | Tarea | Referencia |
|---|-------|------------|
| 3.1 | Evaluar `ZML` como backend alternativo | `github.com/zml` |
| 3.2 | PagedAttention (vLLM-style) | `vllm-zig` + paper SOSP 2023 |
| 3.3 | Continuous Batching | `vllm-zig` + `zinc` scheduler |
| 3.4 | Speculative Decoding | `vllm-zig` + paper |
| 3.5 | Integrar `faiss-zig` (SMC17) si se añade RAG | `github.com/SMC17/faiss-zig` |

---

## 5. Código propio a deprecar/refactorizar

| Código actual | Problema | Acción | Reemplazo del ecosistema |
|---------------|----------|--------|--------------------------|
| `DummyTokenizer` | Asigna `token[i] = i % 256`. Inútil para texto real. | **Deprecar inmediatamente** | `tokenizers-zig` |
| `loadWeightFile()` con `.bin` crudos | Sin metadatos. Requiere scripts Python de conversión. | **Deprecar inmediatamente** | `safetensors-zig` |
| `expandGqa()` naive | Copia datos O(n) por token. Ineficiente en memoria y tiempo. | **Refactorizar** | GQA nativo tipo `vllm-zig` |
| `applyRoPE()` naive | Sin vectorización SIMD. | **Optimizar** | RoPE vectorizado tipo `vllm-zig` |
| `FlashAttention` con `q_len == cfg.N` | No soporta generación autoregresiva (`q_len=1`, `kv_len=N`). | **Refactorizar crítico** | FA variable `q_len`/`kv_len` |
| `TransformerLayer.forward()` sin position/is_prefill | No distingue prefill de generación. | **Refactorizar** | Pipeline tipo `zig_gpt2` |
| `matmul/naive.zig` | Solo para tests. | **Conservar** | Sirve para validación numérica |

---

## 6. Decisiones técnicas

### 6.1 ¿Merge en monorepo o módulos separados?

**Decisión:** Mantener monorepo `zig-ai-engine` con `src/kv_cache/`, `src/tokenizer/`, `src/loader/`. Un solo `build.zig` compila todo.

**Justificación:**
- Simplifica el desarrollo y los tests.
- Un solo `cudaz_stub.zig` compartido para CUDA.
- Versionado atómico de todo el stack.

### 6.2 ¿Quién cuantiza K/V?

**Decisión:** `KVCacheManager` recibe FP16 y cuantiza internamente en `appendTokensF16()`.

**Justificación:**
- La capa transformer genera FP16 (formato nativo de proyección).
- El manager decide el formato según `LayerQuantConfig`.
- Separación de responsabilidades: transformer computa, manager almacena.

### 6.3 ¿GQA en Zig o CUDA?

**Decisión:** Implementar broadcast de KV heads en Zig, no en el kernel CUDA.

**Justificación:**
- Más simple de mantener.
- El kernel FA recibe tensores ya expandidos (o se modifica el índice en el loop de atención CPU).
- A corto plazo, evita recompilar PTX por cada variante de GQA.

### 6.4 ¿Backend GPU: cudaz, sysgpu, zgpu o SPIR-V nativo?

| Opción | Estado | Recomendación |
|--------|--------|---------------|
| **cudaz** (actual) | Funcional, bindings CUDA driver API. | **Mantener a corto plazo.** Es lo que tenemos y funciona. |
| **sysgpu** (Mach) | En desarrollo intensivo. WebGPU nativo en Zig. | **Evaluar a medio plazo** para compute cross-platform (AMD, Intel, Apple). |
| **zgpu** (zig-gamedev) | Estable. Dawn (WebGPU nativo). | **No prioritario.** Más orientado a gráficos que a compute puro. |
| **SPIR-V nativo** | Experimental en Zig 0.13.0+. | **Seguir de cerca.** Escribir kernels en Zig puro sin CUDA es el objetivo a largo plazo. |
| **ZML** | Producción. MLIR/OpenXLA. | **Evaluar si** se necesita soporte AMD/TPU/Trainium. No para reemplazar cudaz inmediatamente. |

---

## 7. Referencias

### Repositorios mencionados

| Proyecto | URL | Licencia |
|----------|-----|----------|
| SMC17/tokenizers-zig | github.com/SMC17/tokenizers-zig | AGPL-3.0 |
| SMC17/safetensors-zig | github.com/SMC17/safetensors-zig | AGPL-3.0 |
| SMC17/vllm-zig | github.com/SMC17/vllm-zig | AGPL-3.0 |
| SMC17/faiss-zig | github.com/SMC17/faiss-zig | AGPL-3.0 |
| cgbur/LLaMa2.zig | github.com/cgbur/LLaMa2.zig | MIT (probable) |
| EugenHotaj/zig_gpt2 | github.com/EugenHotaj/zig_gpt2 | MIT (probable) |
| Marco-Christiani/zigrad | github.com/Marco-Christiani/zigrad | Por verificar |
| zml/zml | github.com/zml | Por verificar |
| zolotukhin/zinc | github.com/zolotukhin/zinc | Por verificar |
| SilasMarvin/dnns-from-scratch-in-zig | github.com/SilasMarvin/dnns-from-scratch-in-zig | MIT (probable) |

### Papers y especificaciones

- **FlashAttention-2:** Dao, Tri. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." ICLR 2024.
- **vLLM / PagedAttention:** Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023.
- **GGUF:** github.com/ggerganov/ggml/blob/master/docs/gguf.md
- **GQA:** Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." EMNLP 2023.
- **Safetensors:** huggingface.co/docs/safetensors

### Documentación Zig

- Zig 0.13.0: ziglang.org/documentation/0.13.0/
- sysgpu (Mach): github.com/hexops/mach
- zgpu: github.com/zig-gamedev/zgpu
- Awesome Zig ML: github.com/zigcc/awesome-zig

---

*Documento generado el 2026-08-03 para el proyecto zig-ai-engine.*
*Para dudas o actualizaciones: revisar los repositorios listados arriba.*
