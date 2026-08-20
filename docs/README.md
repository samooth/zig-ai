# Documentación — Zig AI Engine

Índice central de documentación. Los docs vigentes son la fuente de verdad; los
históricos viven en [`docs/archive/`](archive/) y se conservan por contexto.

> **Estado (2026-08-20):** `zig build test` → todos los tests pasan
> (55/55 con `GGUF_MODEL_PATH`; 52/55 sin él, 3 skips que requieren un `.gguf`).

## Guías de usuario

| Doc | Descripción |
|---|---|
| [`airllm-layer-streaming-guide.md`](airllm-layer-streaming-guide.md) | **AirLLM layer streaming** — flags (`--layer-stream`, `--layer-stream-max`), presupuesto VRAM, tuning, troubleshooting, CUDA graphs con streaming |
| [`../README.md`](../README.md) | README raíz: build, CLI, estructura del repo |

## Arquitectura y diseño

| Doc | Descripción |
|---|---|
| [`qwen35-hybrid-deltanet.md`](qwen35-hybrid-deltanet.md) | Arquitectura híbrida Qwen3.5 (Gated DeltaNet + atención) y LFM2.5 (ShortConv + atención), estrategia `QuantWeight`, recurrencia FLA |
| [`PAGED_ATTENTION_TODO.md`](PAGED_ATTENTION_TODO.md) | Desarrollo e integración de PagedAttention (CPU reference + kernels GPU + pool VMM) |
| [`AIRLLM_IMPLEMENTATION_PLAN.md`](AIRLLM_IMPLEMENTATION_PLAN.md) | Plan de implementación del layer streaming (fases 1–5), estado, riesgos, fix de captura de CUDA graphs |

## Planes y estado del proyecto

| Doc | Descripción |
|---|---|
| [`../TODO.md`](../TODO.md) | Plan de desarrollo canónico por fases (A–I): build 0.16, GGUF, tokenizer, pipeline, PagedAttention, Qwen3.5 hybrid, LFM2.5 hybrid |
| [`../STATE.md`](../STATE.md) | Estado actual del proyecto: bug-hunt de decode CUDA Graphs + optimizaciones en curso |

## Archivados (histórico)

Docs obsoletos/superados, conservados por contexto. **No usar como referencia
técnica actual** — pueden describir código que ya no existe.

| Doc | Fue |
|---|---|
| `archive/zig-ai-engine-plan.md` | Plan de desarrollo original (Fases 1–5) |
| `archive/zig-ai-engine-todo.md` | Checklist Fase 2 y pendientes (superado) |
| `archive/zig-ai-engine-fase2-plan.md` | Plan infraestructura del modelo (Fase 2) |
| `archive/zig-ai-engine-integration-analysis.md` | Análisis integración zig-kv-cache-gpu |
| `archive/integracion-zig-ai-engine.md` | Plan de integración completa (arquitectura objetivo) |
| `archive/g1-gqa-implementation.md` | Spec GQA/MQA en el KV-cache |
| `archive/kv-cache-cuantizado.md` | Diseño del KV-cache cuantizado (construido; difiere del estado real actual) |
| `archive/AIRLLM_TODO.md` | Plan original AirLLM (fases 1–4, superseded por `AIRLLM_IMPLEMENTATION_PLAN.md`) |
| `archive/zig-ai-engine-ecosystem-analysis.md` | Análisis del ecosistema Zig AI/ML |
| `archive/airllm_zig_viabilidad.md` | Estudio de viabilidad AirLLM en Zig |
| `archive/estudio_evolucion_pagedattention.md` | Estudio académico: evolución de PagedAttention |
| `archive/estudio_evolucion_pagedattention_v2.md` | Estudio complementario: técnicas adicionales |

## Conventions

- Los tests se corren con: `GGUF_MODEL_PATH=/opt/models/<modelo>.gguf zig build test`
  (en eCryptfs, usar `--cache-dir` en `/tmp`, p. ej. `--cache-dir /tmp/opencode/zig-cacheN`).
- Los commits de este repo son firmados (`git commit -S`, GPG/EdDSA).