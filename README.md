# Zig AI Engine

[English below](#zig-ai-engine-english)

Motor de inferencia de transformers escrito en Zig con aceleración CUDA.
Carga modelos GGUF y ejecuta generación de texto end-to-end en un solo
binario, sin dependencias de Python ni frameworks externos.

## Características

- **Modelos GGUF** — Qwen2/2.5/3.x, Qwen3.5 híbrido (Gated DeltaNet),
  LFM2.5 (ShortConv), Llama, Gemma, Mistral y MoE (Qwen3-MoE, gemma-4,
  Mixtral).
- **Cuantización** — 25+ formatos GGUF (q4_0, q8_0, familia K…, familia IQ…)
  con dequantización on-the-fly dentro de los kernels: los pesos nunca se
  materializan en f32.
- **KV-cache cuantizado** — 13 formatos para K y V (`-ctk`/`-ctv`), con
  roundtrip bit-exact verificado.
- **Atención** — FlashAttention GPU, PagedAttention, GQA y M-RoPE
  (incluido el variant multi-sección de los modelos híbridos).
- **Matmul multi-backend** — naive, SIMD, tiled, parallel, OpenBLAS y
  cuBLAS, seleccionables por CLI o auto-detectados.
- **Modelos mayores que la VRAM** — layer streaming al estilo AirLLM: un
  modelo de 27B parámetros corre en una GPU de 8GB.
- **MoE offload híbrido** — expertos en CPU con fetch PCIe por demanda,
  cache LRU y memoria pinned copy-once.
- **Decodificación especulativa** — MTP y sidecars DFlash/DSpark/DFlash2
  con verificación batched.
- **Vision** — encoder CLIP ViT device-resident (merger Qwen2/2.5/3-VL),
  inyección de embeddings con M-RoPE 2D, imagen y vídeo.
- **Server** — `--serve`: API compatible OpenAI/Anthropic/Ollama con
  streaming SSE, autenticación, TLS y auditoría.

## Inicio rápido

Requisitos: Zig 0.16.0 y, opcionalmente, CUDA Toolkit 12.x con una GPU
compatible (sm_70+).

```bash
# Build (binario en zig-out/bin/zig-ai-engine)
zig build install -Doptimize=ReleaseFast

# Generación greedy
./zig-out/bin/zig-ai-engine -m modelo.gguf --prompt "Hola" -n 128

# Perplexity sobre un corpus
./zig-out/bin/zig-ai-engine -m modelo.gguf --ppl corpus.txt --ctx-size 2048

# Servidor API (compatible OpenAI)
./zig-out/bin/zig-ai-engine -m modelo.gguf --serve

# Vision: describir una imagen
./zig-out/bin/zig-ai-engine -m modelo.gguf \
  --mmproj mmproj.gguf --image foto.jpg --prompt "Describe this image"
```

Más flags y ejemplos: [`docs/README.md`](docs/README.md) ·
Estado y dirección: [`docs/ROADMAP.md`](docs/ROADMAP.md)

## Matmul backends

| Backend   | f32 | f64 | f16 | bf16 | INT8 | Async | Batch |
|-----------|-----|-----|-----|------|------|-------|-------|
| naive     | ✓   | ✓   | —   | —    | —    | —     | —     |
| simd      | ✓   | ✓   | —   | —    | —    | —     | —     |
| tiled     | ✓   | —   | —   | —    | —    | —     | —     |
| parallel  | ✓   | ✓   | —   | —    | —    | —     | —     |
| openblas  | ✓   | ✓   | —   | —    | —    | —     | —     |
| cublas    | ✓   | —   | ✓*  | ✓*   | —    | ✓     | ✓     |

\* GemmEx con acumulación f32.

## Licencia

MIT

---

# Zig AI Engine (English)

A transformer inference engine written in Zig with CUDA acceleration. It
loads GGUF models and runs end-to-end text generation in a single binary,
with no Python or external framework dependencies.

## Features

- **GGUF models** — Qwen2/2.5/3.x, Qwen3.5 hybrid (Gated DeltaNet), LFM2.5
  (ShortConv), Llama, Gemma, Mistral and MoE (Qwen3-MoE, gemma-4, Mixtral).
- **Quantization** — 25+ GGUF formats (q4_0, q8_0, K-family, IQ-family) with
  on-the-fly dequantization inside the kernels: weights are never
  materialized as f32.
- **Quantized KV cache** — 13 formats for K and V (`-ctk`/`-ctv`), with
  verified bit-exact roundtrips.
- **Attention** — GPU FlashAttention, PagedAttention, GQA and M-RoPE
  (including the multi-section variant used by hybrid models).
- **Multi-backend matmul** — naive, SIMD, tiled, parallel, OpenBLAS and
  cuBLAS, selectable via CLI or auto-detected.
- **Models larger than VRAM** — AirLLM-style layer streaming: a 27B
  parameter model runs on an 8GB GPU.
- **Hybrid MoE offload** — experts on CPU with on-demand PCIe fetch, LRU
  caching and copy-once pinned memory.
- **Speculative decoding** — MTP and DFlash/DSpark/DFlash2 sidecars with
  batched verification.
- **Vision** — device-resident CLIP ViT encoder (Qwen2/2.5/3-VL merger),
  embedding injection with 2D M-RoPE, image and video input.
- **Server** — `--serve`: OpenAI/Anthropic/Ollama-compatible API with SSE
  streaming, authentication, TLS and audit logging.

## Quick start

Requirements: Zig 0.16.0 and, optionally, CUDA Toolkit 12.x with a
compatible GPU (sm_70+).

```bash
# Build (binary at zig-out/bin/zig-ai-engine)
zig build install -Doptimize=ReleaseFast

# Greedy generation
./zig-out/bin/zig-ai-engine -m model.gguf --prompt "Hello" -n 128

# Perplexity over a corpus
./zig-out/bin/zig-ai-engine -m model.gguf --ppl corpus.txt --ctx-size 2048

# API server (OpenAI-compatible)
./zig-out/bin/zig-ai-engine -m model.gguf --serve

# Vision: describe an image
./zig-out/bin/zig-ai-engine -m model.gguf \
  --mmproj mmproj.gguf --image photo.jpg --prompt "Describe this image"
```

More flags and examples: [`docs/README.md`](docs/README.md) ·
Status and direction: [`docs/ROADMAP.md`](docs/ROADMAP.md)

## Matmul backends

| Backend   | f32 | f64 | f16 | bf16 | INT8 | Async | Batch |
|-----------|-----|-----|-----|------|------|-------|-------|
| naive     | ✓   | ✓   | —   | —    | —    | —     | —     |
| simd      | ✓   | ✓   | —   | —    | —    | —     | —     |
| tiled     | ✓   | —   | —   | —    | —    | —     | —     |
| parallel  | ✓   | ✓   | —   | —    | —    | —     | —     |
| openblas  | ✓   | ✓   | —   | —    | —    | —     | —     |
| cublas    | ✓   | —   | ✓*  | ✓*   | —    | ✓     | ✓     |

\* GemmEx with f32 accumulation.

## License

MIT
