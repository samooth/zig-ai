# Documentation — Zig AI Engine

Public documentation index.

> **Status (2026-08-20):** `zig build test` → all tests pass
> (55/55 with `GGUF_MODEL_PATH`; 52/55 without it, 3 skips require a `.gguf`).

## User guides

| Doc | Description |
|---|---|
| [`airllm-layer-streaming-guide.md`](airllm-layer-streaming-guide.md) | AirLLM layer streaming — flags (`--layer-stream`, `--layer-stream-max`), VRAM budget, tuning, troubleshooting |
| [`airllm-layer-streaming-guide.es.md`](airllm-layer-streaming-guide.es.md) | Guía de usuario: AirLLM layer streaming (español) |
| [`../README.md`](../README.md) | Root README: build, CLI, repo structure |
| [`../README.es.md`](../README.es.md) | README raíz en español |

## Architecture and design

| Doc | Description |
|---|---|
| [`qwen35-hybrid-deltanet.md`](qwen35-hybrid-deltanet.md) | Qwen3.5 hybrid architecture (Gated DeltaNet + attention) and LFM2.5 (ShortConv + attention), `QuantWeight` strategy, FLA recurrence |
| [`qwen35-hybrid-deltanet.es.md`](qwen35-hybrid-deltanet.es.md) | Arquitectura híbrida Qwen3.5 y LFM2.5 (español) |

## Quantization

| Doc | Description |
|---|---|
| [`quantization.md`](quantization.md) | Supported quantization formats (weight + KV cache) |
| [`quantization.es.md`](quantization.es.md) | Formatos de cuantización soportados (español) |

## Roadmap and changes

| Doc | Description |
|---|---|
| [`../ROADMAP.md`](../ROADMAP.md) | Public roadmap (completed + planned) |
| [`../ROADMAP.es.md`](../ROADMAP.es.md) | Hoja de ruta pública (español) |
| [`../CHANGELOG.md`](../CHANGELOG.md) | Changelog |

## Server and API

| Doc | Description |
|---|---|
| [`server.md`](server.md) | HTTP server mode (OpenAI/Anthropic/Ollama API), flags, streaming, batching, auth |
| [`server.es.md`](server.es.md) | Modo servidor HTTP (API OpenAI/Anthropic/Ollama) (español) |

## Conventions

- Tests: `GGUF_MODEL_PATH=/opt/models/<model>.gguf zig build test`
  (on eCryptfs, use `--cache-dir` in `/tmp`, e.g. `--cache-dir /tmp/opencode/zig-cacheN`).
