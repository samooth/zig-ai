# Documentación — Zig AI Engine

[English below](#documentation--zig-ai-engine-english)

Guía de referencia rápida del motor. Para una visión general, build y
ejemplos completos, ver el [README raíz](../README.md).

## Índice

| Doc | Contenido |
|---|---|
| [`../README.md`](../README.md) | Overview, características, inicio rápido |
| [`ROADMAP.md`](ROADMAP.md) | Estado actual y dirección del proyecto |

## Uso diario

```bash
# Generación greedy (determinista)
./zig-out/bin/zig-ai-engine --model modelo.gguf \
  --prompt "The capital of France is" -n 4 --temperature 0

# Muestreo con parámetros
./zig-out/bin/zig-ai-engine --model modelo.gguf \
  --prompt "Escribe un poema" -n 128 --temperature 0.8 --top-p 0.95

# Perplexity (calidad del modelo)
./zig-out/bin/zig-ai-engine --model modelo.gguf --ppl corpus.txt --ctx-size 2048

# Servidor API (compatible OpenAI/Anthropic/Ollama)
./zig-out/bin/zig-ai-engine --model modelo.gguf --serve

# Vision: imagen
./zig-out/bin/zig-ai-engine --model modelo.gguf \
  --mmproj mmproj.gguf --image foto.jpg --prompt "Describe this image"

# Vision: vídeo (ffmpeg/ffprobe externos)
./zig-out/bin/zig-ai-engine --model modelo.gguf \
  --mmproj mmproj.gguf --video clip.mp4 --prompt "Resume el vídeo"
```

## Flags principales

| Flag | Descripción |
|------|-------------|
| `-m, --model <ruta>` | Modelo GGUF (obligatorio) |
| `--prompt <texto>` | Prompt de entrada |
| `-n <num>` | Tokens a generar |
| `--temperature <t>` | 0 = greedy determinista |
| `--top-k <k>` / `--top-p <p>` | Sampling |
| `--ctx-size <n>` | Tamaño de contexto (default 65536) |
| `--backend <auto\|cpu\|gpu>` | Backend de matmul |
| `-cl, --n-gpu-layers <n>` | Capas en GPU (resto CPU) |
| `-ctk <fmt>` / `-ctv <fmt>` | Cuantización del KV-cache (fp16, q8_0, q4_0, q4_k… iq1_m) |
| `--layer-stream` | Streaming de capas: modelos mayores que la VRAM |
| `--draft-model <gguf>` | Modelo draft para decodificación especulativa |
| `--spec-type <tipo>` | Especulativa: MTP / DFlash / DSpark / DFlash2 |
| `--mmproj <ruta>` | Encoder vision (GGUF del merger CLIP ViT) |
| `--image <ruta>` / `--video <ruta>` | Entrada multimodal (repetible) |
| `--ppl <fichero>` | Modo perplexity sobre un corpus |
| `--serve` | Servidor API en lugar de CLI |
| `--version` | Versión + git sha de build |

## Soporte de modelos

| Familia | Arquitectura | Notas |
|---------|--------------|-------|
| Qwen3.5 | Híbrido (Gated DeltaNet + atención) | Familia prioritaria; 0.8B–27B |
| Qwen2/2.5/3.x | Denso y MoE | Incluye Qwen3-VL (vision) |
| LFM2.5 | ShortConv híbrido | |
| Llama 3.x | Denso | Incluye Instruct |
| Gemma 3 | Denso | |
| Mistral | Denso | |
| MoE | Qwen3-MoE, gemma-4, Mixtral | Offload híbrido CPU/GPU |

25+ formatos de cuantización GGUF para pesos (q4_0…q6_k, iq1_m…iq4_nl,
tq2_0) y 13 para KV-cache.

---

# Documentation — Zig AI Engine (English)

Quick reference guide for the engine. For an overview, build and full
examples, see the [root README](../README.md).

## Index

| Doc | Contents |
|---|---|
| [`../README.md`](../README.md) | Overview, features, quick start |
| [`ROADMAP.md`](ROADMAP.md) | Current status and project direction |

## Daily use

```bash
# Greedy generation (deterministic)
./zig-out/bin/zig-ai-engine --model model.gguf \
  --prompt "The capital of France is" -n 4 --temperature 0

# Sampling with parameters
./zig-out/bin/zig-ai-engine --model model.gguf \
  --prompt "Write a poem" -n 128 --temperature 0.8 --top-p 0.95

# Perplexity (model quality)
./zig-out/bin/zig-ai-engine --model model.gguf --ppl corpus.txt --ctx-size 2048

# API server (OpenAI/Anthropic/Ollama-compatible)
./zig-out/bin/zig-ai-engine --model model.gguf --serve

# Vision: image
./zig-out/bin/zig-ai-engine --model model.gguf \
  --mmproj mmproj.gguf --image photo.jpg --prompt "Describe this image"

# Vision: video (external ffmpeg/ffprobe)
./zig-out/bin/zig-ai-engine --model model.gguf \
  --mmproj mmproj.gguf --video clip.mp4 --prompt "Summarize the video"
```

## Main flags

| Flag | Description |
|------|-------------|
| `-m, --model <path>` | GGUF model (required) |
| `--prompt <text>` | Input prompt |
| `-n <num>` | Tokens to generate |
| `--temperature <t>` | 0 = deterministic greedy |
| `--top-k <k>` / `--top-p <p>` | Sampling |
| `--ctx-size <n>` | Context size (default 65536) |
| `--backend <auto\|cpu\|gpu>` | Matmul backend |
| `-cl, --n-gpu-layers <n>` | Layers on GPU (rest on CPU) |
| `-ctk <fmt>` / `-ctv <fmt>` | KV-cache quantization (fp16, q8_0, q4_0, q4_k… iq1_m) |
| `--layer-stream` | Layer streaming: models larger than VRAM |
| `--draft-model <gguf>` | Draft model for speculative decoding |
| `--spec-type <type>` | Speculative: MTP / DFlash / DSpark / DFlash2 |
| `--mmproj <path>` | Vision encoder (CLIP ViT merger GGUF) |
| `--image <path>` / `--video <path>` | Multimodal input (repeatable) |
| `--ppl <file>` | Perplexity mode over a corpus |
| `--serve` | API server instead of CLI |
| `--version` | Version + build git sha |

## Model support

| Family | Architecture | Notes |
|--------|--------------|-------|
| Qwen3.5 | Hybrid (Gated DeltaNet + attention) | Priority family; 0.8B–27B |
| Qwen2/2.5/3.x | Dense and MoE | Includes Qwen3-VL (vision) |
| LFM2.5 | ShortConv hybrid | |
| Llama 3.x | Dense | Includes Instruct |
| Gemma 3 | Dense | |
| Mistral | Dense | |
| MoE | Qwen3-MoE, gemma-4, Mixtral | Hybrid CPU/GPU offload |

25+ GGUF weight quantization formats (q4_0…q6_k, iq1_m…iq4_nl, tq2_0) and
13 KV-cache formats.
