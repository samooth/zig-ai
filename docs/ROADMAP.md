# Roadmap — Zig AI Engine

> Estado y dirección del proyecto. Actualizado por hitos.
> [English below](#roadmap--zig-ai-engine-english)

Última actualización: 2026-09-14

## Qué soporta hoy

| Área | Estado |
|---|---|
| Modelos GGUF | Qwen2/2.5/3.x, Qwen3.5 híbrido (DeltaNet), LFM2.5 (ShortConv), Llama, Gemma, Mistral, MoE (Qwen3-MoE, gemma-4, Mixtral) |
| Cuantización de pesos | 25+ formatos GGUF con dequant on-the-fly en kernels — sin materializar pesos f32 |
| KV-cache cuantizado | 13 formatos vía `-ctk`/`-ctv` (fp16, q8_0, q4_0, q4_k…q6_k, iq1_m, iq3_s, iq4_nl…) |
| Atención | FlashAttention GPU, PagedAttention, GQA, M-RoPE, CUDA graphs |
| Matmul | 6 backends: naive / SIMD / tiled / parallel / OpenBLAS / cuBLAS |
| Modelos > VRAM | Layer streaming (AirLLM-style) y streaming v2 cuantizado — Qwen3.8-27B corre en 8GB VRAM |
| MoE offload | Híbrido CPU/GPU: cache LRU de expertos, copy-once pinned, io_uring |
| Especulativa | MTP, sidecars DFlash/DSpark/DFlash2, verify-batched |
| Vision | Encoder CLIP ViT GPU device-resident (Qwen2/2.5/3-VL merger), M-RoPE 2D, imagen y vídeo |
| Server | `--serve`: API OpenAI/Anthropic/Ollama compatible, streaming SSE, auth/TLS/audit |

## En qué trabajamos ahora

- **Rendimiento decode/prefill** — CUDA graphs, argmax en GPU, GEMMs
  batched con tensor cores, kernels cuantizados M=1 (dp4a).
- **PPL end-to-end en GPU** — perplexity con prefill batcheado (gate de
  paridad vs llama.cpp).
- **Release v0.1.0** — gates de release automatizados sobre la familia
  Qwen3.5 (0.8B→27B): generación greedy correcta en toda la familia y
  perplexity de referencia (golden vs llama.cpp).

## Dirección (medio plazo)

- **KVarN** — KV comprimido con records WHT (port BeeLlama): encode/decode
  CPU+GPU completados; queda CLI e integración en el manager.
- **MoE híbrido** — offload de expertos con prefetch pool pinned
  (RSS 20GB→9.6GB validado en gemma-4-26B).
- **Especulativa gen-2** — kernels device-resident, selector top-K,
  matriz de aceptación con prompts naturales.
- **Transferencia KV cross-model** — mapper ridge calibrado offline
  (paper 2608.03893): prefill en modelo pequeño, decode en grande.
- **Server v2** — engine persistente entre requests, continuous batching.

## Bugs conocidos

| Área | Nota | Mitigación |
|---|---|---|
| RAM con iq1_s | materializa scratch f32 (sin kernel GEMM ese dtype) | usar IQ1_M/Q4_K |
| OOM `--ctx-size` | default 65536 puede exceder RAM modesta | pasar `--ctx-size` acorde |

## No-goals (decisiones con datos)

- FP8 nativo en sm_86 (solo emulación E4M3 — más lento que f16).
- iq1_s/iq2 como KV-cache default (pérdida de coherencia intrínseca 1-2 bits).
- mmap→GPU zero-copy (imposible con driver actual).

---

# Roadmap — Zig AI Engine (English)

> Project status and direction. Updated per milestone.
> [Versión española arriba](#roadmap--zig-ai-engine-spanish)

Last updated: 2026-09-14

## Supported today

| Area | Status |
|---|---|
| GGUF models | Qwen2/2.5/3.x, Qwen3.5 hybrid (DeltaNet), LFM2.5 (ShortConv), Llama, Gemma, Mistral, MoE (Qwen3-MoE, gemma-4, Mixtral) |
| Weight quantization | 25+ GGUF formats with on-the-fly kernel dequant — no f32 materialization |
| Quantized KV cache | 13 formats via `-ctk`/`-ctv` |
| Attention | GPU FlashAttention, PagedAttention, GQA, M-RoPE, CUDA graphs |
| Matmul | 6 backends: naive / SIMD / tiled / parallel / OpenBLAS / cuBLAS |
| Models > VRAM | AirLLM-style layer streaming + quantized v2 — Qwen3.8-27B runs in 8GB VRAM |
| MoE offload | CPU/GPU hybrid: LRU expert cache, pinned copy-once, io_uring |
| Speculative | MTP, DFlash/DSpark/DFlash2 sidecars, batched verify |
| Vision | GPU device-resident CLIP ViT encoder (Qwen2/2.5/3-VL merger), 2D M-RoPE, image + video |
| Server | `--serve`: OpenAI/Anthropic/Ollama-compatible API, SSE streaming, auth/TLS/audit |

## Current focus

- **Decode/prefill performance** — CUDA graphs, GPU argmax, batched tensor-core GEMMs, M=1 quantized kernels (dp4a).
- **End-to-end GPU perplexity** — batched prefill PPL (parity gate vs llama.cpp).
- **v0.1.0 release** — automated release gates over the Qwen3.5 family
  (0.8B–27B): correct greedy generation across the family and reference
  perplexity (golden vs llama.cpp).

## Mid-term direction

- **KVarN** — WHT-record compressed KV (BeeLlama port): CPU+GPU encode/decode done; CLI + manager integration pending.
- **Hybrid MoE** — expert offload with pinned prefetch pool (RSS 20GB→9.6GB validated on gemma-4-26B).
- **Speculative gen-2** — device-resident kernels, top-K selector, natural-prompt acceptance matrix.
- **Cross-model KV transfer** — offline-calibrated ridge mapper (paper 2608.03893): prefill small model, decode large.
- **Server v2** — persistent engine across requests, continuous batching.

## Known bugs

| Area | Note | Mitigation |
|---|---|---|
| RAM with iq1_s | materializes f32 scratch (no GEMM kernel for that dtype) | use IQ1_M/Q4_K |
| `--ctx-size` OOM | default 65536 may exceed modest RAM | pass a suitable `--ctx-size` |

## Non-goals (data-backed decisions)

- Native FP8 on sm_86 (E4M3 emulation only — slower than f16).
- iq1_s/iq2 family as default KV cache (intrinsic 1-2 bit coherence loss).
- mmap→GPU zero-copy (impossible with current driver).
