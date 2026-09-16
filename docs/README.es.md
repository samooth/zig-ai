# Documentación — Zig AI Engine

Índice central de documentación pública.

> **Estado (2026-08-20):** `zig build test` → todos los tests pasan
> (55/55 con `GGUF_MODEL_PATH`; 52/55 sin él, 3 skips que requieren un `.gguf`).

## Guías de usuario

| Doc | Descripción |
|---|---|
| [`airllm-layer-streaming-guide.md`](airllm-layer-streaming-guide.md) | Guía AirLLM layer streaming (inglés) |
| [`airllm-layer-streaming-guide.es.md`](airllm-layer-streaming-guide.es.md) | Guía AirLLM layer streaming (español) |
| [`../README.md`](../README.md) | README raíz (inglés) |
| [`../README.es.md`](../README.es.md) | README raíz (español) |

## Arquitectura y diseño

| Doc | Descripción |
|---|---|
| [`qwen35-hybrid-deltanet.md`](qwen35-hybrid-deltanet.md) | Arquitectura híbrida Qwen3.5 y LFM2.5 (inglés) |
| [`qwen35-hybrid-deltanet.es.md`](qwen35-hybrid-deltanet.es.md) | Arquitectura híbrida Qwen3.5 y LFM2.5 (español) |

## Cuantización

| Doc | Descripción |
|---|---|
| [`quantization.md`](quantization.md) | Formatos de cuantización soportados (inglés) |
| [`quantization.es.md`](quantization.es.md) | Formatos de cuantización soportados (español) |

## Roadmap y cambios

| Doc | Descripción |
|---|---|
| [`../ROADMAP.md`](../ROADMAP.md) | Roadmap público (inglés) |
| [`../ROADMAP.es.md`](../ROADMAP.es.md) | Hoja de ruta pública (español) |
| [`../CHANGELOG.md`](../CHANGELOG.md) | Historial de cambios (inglés) |

## Servidor y API

| Doc | Descripción |
|---|---|
| [`server.md`](server.md) | Modo servidor HTTP (API OpenAI/Anthropic/Ollama), flags, streaming, batching, auth |
| [`server.es.md`](server.es.md) | Modo servidor HTTP (API OpenAI/Anthropic/Ollama) (español) |

## Convenciones

- Tests: `GGUF_MODEL_PATH=/opt/models/<modelo>.gguf zig build test`
  (en eCryptfs, usar `--cache-dir` en `/tmp`, p. ej. `--cache-dir /tmp/opencode/zig-cacheN`).
