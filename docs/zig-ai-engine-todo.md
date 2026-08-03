# Zig AI Engine — Lista de Cosas por Hacer (TODO)

> Fecha: 2026-08-03
> Estado: Fase 2 — Módulos core generados, pendiente integración y testing

---

## Leyenda

- 🔴 **Bloqueante** — Sin esto no compila o no funciona end-to-end
- 🟡 **Importante** — Necesario para modelos reales
- 🟢 **Mejora** — Optimización o feature adicional
- ⚪ **Futuro** — Fase 3+

---

## Fase 2.1 — Integración y Main (Bloqueante)

| # | Tarea | Archivo | Prioridad | Detalle |
|---|-------|---------|-----------|---------|
| 2.1.1 | Actualizar `src/main.zig` | `src/main.zig` | 🔴 | Reemplazar demo actual con pipeline real usando BPETokenizer, SafetensorsLoader, InferencePipeline |
| 2.1.2 | Corregir imports en `layer.zig` | `src/transformer/layer.zig` | 🔴 | `@import("norm.zig")` → `@import("norm")` etc. Ajustar a estructura de módulos Zig |
| 2.1.3 | Corregir imports en `pipeline.zig` | `src/transformer/pipeline.zig` | 🔴 | Mismo que arriba. `@import("transformer/embedding.zig")` no funciona como módulo, usar `@import("embedding")` |
| 2.1.4 | Actualizar `build.zig.zon` | `build.zig.zon` | 🔴 | Versión 0.2.0 → 0.3.0, añadir nuevos paths |
| 2.1.5 | Añadir `src/transformer/` a paths de build.zig.zon | `build.zig.zon` | 🔴 | `src/tokenizer/`, `src/loader/` |
| 2.1.6 | Test de compilación limpia | — | 🔴 | `zig build test` debe pasar sin errores |

---

## Fase 2.2 — Tokenizer Real (Importante)

| # | Tarea | Archivo | Prioridad | Detalle |
|---|-------|---------|-----------|---------|
| 2.2.1 | Parser de `tokenizer.json` HF | `src/tokenizer/tokenizer_json.zig` | 🟡 | Leer vocab.json, merges.txt, special_tokens_map.json del formato HuggingFace |
| 2.2.2 | Pre-tokenización regex GPT-2 | `src/tokenizer/bpe.zig` | 🟡 | Implementar regex real: `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+` |
| 2.2.3 | Byte-fallback encoding | `src/tokenizer/bpe.zig` | 🟡 | Mapear bytes 0-255 a tokens `<0x00>`...`<0xFF>` como en Llama |
| 2.2.4 | Special tokens (BOS, EOS, PAD, UNK) | `src/tokenizer/bpe.zig` | 🟡 | `<s>`, `</s>`, `<unk>`, `<pad>` configurables por modelo |
| 2.2.5 | Test con vocabulario real | `tests/test_tokenizer.zig` | 🟡 | Cargar tokenizer de Llama-3.2-1B y verificar encode/decode de frases |

---

## Fase 2.3 — Safetensors y Carga de Pesos (Importante)

| # | Tarea | Archivo | Prioridad | Detalle |
|---|-------|---------|-----------|---------|
| 2.3.1 | Mejorar parser JSON de safetensors | `src/loader/safetensors.zig` | 🟡 | Manejar escapes en strings, números negativos, arrays anidados |
| 2.3.2 | Soporte dtypes adicionales | `src/loader/safetensors.zig` | 🟡 | BF16, INT8, F32 → conversión automática a F16 |
| 2.3.3 | Mapeo de nombres HF → interno | `src/loader/model_config.zig` | 🟡 | `model.layers.0.self_attn.q_proj.weight` → `layer.0.q_proj` |
| 2.3.4 | Carga por tensores (no todo en memoria) | `src/loader/safetensors.zig` | 🟡 | mmap o streaming para modelos > 7B |
| 2.3.5 | Script de conversión HF → nuestro formato | `scripts/convert_hf.py` | 🟡 | Python script que descarga modelo HF, transpone pesos, genera `.bin` por capa |
| 2.3.6 | Config del modelo (config.json) | `src/loader/model_config.zig` | 🟡 | Leer hidden_size, num_layers, num_heads, intermediate_size, vocab_size, rope_theta |

---

## Fase 2.4 — Transformer Layer Completa (Bloqueante)

| # | Tarea | Archivo | Prioridad | Detalle |
|---|-------|---------|-----------|---------|
| 2.4.1 | Integrar `layer_v2.zig` en `layer.zig` | `src/transformer/layer.zig` | 🔴 | Sobrescribir layer.zig con la versión completa (Norm + Attn + Residual + FFN + Residual) |
| 2.4.2 | Tests de layer con FFN | `tests/test_transformer.zig` | 🔴 | Verificar que forward con FFN no produce NaN/Inf |
| 2.4.3 | Tests de layer con RMSNorm | `tests/test_transformer.zig` | 🔴 | Verificar preservación de norma |
| 2.4.4 | Tests de pipeline end-to-end | `tests/test_pipeline.zig` | 🔴 | Prefill + generación de N tokens con dummy weights |
| 2.4.5 | Verificar shapes en generación autoregresiva | `src/transformer/layer.zig` | 🔴 | `q_len=1`, `kv_len=total_len` debe funcionar correctamente |

---

## Fase 2.5 — RoPE y GQA Optimizaciones (Mejora)

| # | Tarea | Archivo | Prioridad | Detalle |
|---|-------|---------|-----------|---------|
| 2.5.1 | Precomputar sin/cos de RoPE | `src/transformer/rope.zig` | 🟢 | Cachear tabla de sin/cos por posición para evitar recalcular |
| 2.5.2 | RoPE con base configurable (rope_theta) | `src/transformer/rope.zig` | 🟢 | Llama usa 10000, Llama-3 usa 500000 |
| 2.5.3 | RoPE con scaling (NTK, YaRN) | `src/transformer/rope.zig` | 🟢 | Para contextos largos > 4096 |
| 2.5.4 | GQA sin expand físico en FA | `src/transformer/gqa.zig` | 🟢 | Modificar índices en kernel CUDA en lugar de copiar |
| 2.5.5 | GQA con broadcast en CPU FA | `src/fa/flash_attention.zig` | 🟢 | FlashAttentionCpu con índices de GQA |

---

## Fase 2.6 — Sampling y Generación (Mejora)

| # | Tarea | Archivo | Prioridad | Detalle |
|---|-------|---------|-----------|---------|
| 2.6.1 | Repetition penalty | `src/transformer/pipeline.zig` | 🟢 | Penalizar tokens repetidos en los últimos N |
| 2.6.2 | Min-P sampling | `src/transformer/pipeline.zig` | 🟢 | Filtrar tokens con probabilidad < min_p * max_prob |
| 2.6.3 | Mirostat sampling | `src/transformer/pipeline.zig` | 🟢 | Control adaptativo de perplexity |
| 2.6.4 | Beam search | `src/transformer/pipeline.zig` | ⚪ | Para tareas de traducción/resumen |
| 2.6.5 | Streaming de tokens | `src/transformer/pipeline.zig` | 🟢 | Callback por token generado para UI |

---

## Fase 3 — Optimización de Performance (Futuro)

| # | Tarea | Archivo | Prioridad | Detalle |
|---|-------|---------|-----------|---------|
| 3.1 | Matmul SIMD multi-hilo | `src/matmul/parallel.zig` | ⚪ | Mejorar con blocking y prefetch |
| 3.2 | RoPE vectorizado SIMD | `src/transformer/rope.zig` | ⚪ | @Vector para pares de dimensiones |
| 3.3 | RMSNorm vectorizado | `src/transformer/norm.zig` | ⚪ | SIMD para mean(x²) y scale |
| 3.4 | PagedAttention (vLLM) | `src/kv_cache/` | ⚪ | Bloques de 16-32 tokens, compartir entre secuencias |
| 3.5 | Continuous batching | `src/transformer/pipeline.zig` | ⚪ | Batch dinámico de múltiples requests |
| 3.6 | Speculative decoding | `src/transformer/pipeline.zig` | ⚪ | Modelo pequeño predice, grande verifica |
| 3.7 | Quantización de activaciones | `src/matmul/quant.zig` | ⚪ | INT8/INT4 para activaciones, no solo pesos |
| 3.8 | Kernel FA con q_len variable | `cuda/flash_attention.cu` | ⚪ | q_len=1 optimizado para generación |
| 3.9 | sysgpu / WebGPU backend | `src/gpu/` | ⚪ | Compute cross-platform (AMD, Intel, Apple) |
| 3.10 | ZML backend alternativo | — | ⚪ | Compilar modelos a MLIR/OpenXLA |

---

## Fase 4 — Modelos Reales (Futuro)

| # | Tarea | Archivo | Prioridad | Detalle |
|---|-------|---------|-----------|---------|
| 4.1 | Soporte Llama 2/3 | `src/models/llama.zig` | ⚪ | Config, tokenizer, arquitectura |
| 4.2 | Soporte Mistral | `src/models/mistral.zig` | ⚪ | Sliding window attention |
| 4.3 | Soporte Qwen | `src/models/qwen.zig` | ⚪ | Tie embeddings, Qwen-specific |
| 4.4 | Soporte GPT-2 | `src/models/gpt2.zig` | ⚪ | LayerNorm, GELU, sin GQA |
| 4.5 | GGUF loader | `src/loader/gguf.zig` | ⚪ | Leer modelos de llama.cpp |
| 4.6 | Chat template | `src/models/chat.zig` | ⚪ | Aplicar templates tipo "<|user|>\n{}<|assistant|>\n" |

---

## Fase 5 — Producción (Futuro)

| # | Tarea | Archivo | Prioridad | Detalle |
|---|-------|---------|-----------|---------|
| 5.1 | API HTTP REST | `src/server/` | ⚪ | Compatible con OpenAI API |
| 5.2 | WebSocket streaming | `src/server/` | ⚪ | SSE para tokens en tiempo real |
| 5.3 | Multi-GPU | `src/gpu/multi.zig` | ⚪ | Tensor parallelism, pipeline parallelism |
| 5.4 | LoRA/QLoRA inference | `src/lora/` | ⚪ | Adaptadores pequeños sobre modelo base |
| 5.5 | Tool calling | `src/tools/` | ⚪ | Function calling con JSON schema |
| 5.6 | RAG integration | `src/rag/` | ⚪ | Faiss/ANN para retrieval |

---

## Bugs Conocidos / Problemas Pendientes

| # | Problema | Archivo | Severidad |
|---|----------|---------|-----------|
| B1 | `layer.zig` importa `@import("norm.zig")` en lugar de `@import("norm")` | `layer.zig` | 🔴 No compila |
| B2 | `pipeline.zig` importa `@import("transformer/embedding.zig")` — path incorrecto | `pipeline.zig` | 🔴 No compila |
| B3 | `safetensors.zig` usa `std.heap.page_allocator` en parseo — debería usar `self.allocator` | `safetensors.zig` | 🟡 Memory leak potencial |
| B4 | `bpe.zig` no libera memoria de `name_owned` en `parseTensorInfo` | `safetensors.zig` | 🟡 Memory leak |
| B5 | FlashAttention CPU usa O(N²) memoria — limitado a N<4096 | `src/fa/flash_attention.zig` | 🟡 No crítico para demo |
| B6 | `expandGqaFallback` hace copia física — ineficiente | `gqa.zig` | 🟢 Optimizable |
| B7 | No hay manejo de errores en `embeddingLookup` para token out-of-vocab | `embedding.zig` | 🟡 Panic potencial |

---

## Checklist de Integración Inmediata

Para tener un `zig build test` que pase:

- [ ] Copiar `layer_v2.zig` → `src/transformer/layer.zig` (sobrescribir)
- [ ] Copiar `build_v2.zig` → `build.zig` (sobrescribir)
- [ ] Crear directorios: `src/tokenizer/`, `src/loader/`, `src/transformer/`
- [ ] Copiar todos los `.zig` a sus rutas correspondientes
- [ ] Corregir imports en `layer.zig`: `@import("norm.zig")` → `@import("norm")`
- [ ] Corregir imports en `pipeline.zig`: `@import("transformer/embedding.zig")` → `@import("embedding")`
- [ ] Corregir `safetensors.zig`: `parseTensorInfo` usa `std.heap.page_allocator` para `name_owned`
- [ ] Ejecutar `zig build test`
- [ ] Arreglar errores de compilación uno por uno
- [ ] Ejecutar `zig build run` para ver demo

---

## Métricas Objetivo

| Métrica | Target Fase 2 | Target Fase 3 |
|---------|---------------|---------------|
| Compilación limpia | ✅ `zig build test` pasa | ✅ |
| Demo end-to-end | ✅ Prefill + 20 tokens | ✅ |
| Tokens/seg (CPU, 4 capas) | ~10 tok/s | ~50 tok/s |
| Tokens/seg (GPU, 4 capas) | ~50 tok/s | ~200 tok/s |
| Memoria KV-cache (4K ctx) | < 512 MB (Q4_0) | < 256 MB (Q4_0 + paging) |
| Modelos soportados | Dummy / Demo | Llama-3.2-1B, Mistral-7B |

---

*Documento generado el 2026-08-03. Actualizar conforme se completan tareas.*
