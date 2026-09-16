# Qwen3.5 Hybrid — Gated DeltaNet (SSM) + Atención Completa

> Estado: 2026-08-20 — implementación completa (SSM con `QuantWeight`, bloque híbrido de
> atención + rutado por `isFullAttentionLayer`, validado contra llama.cpp con
> Qwen3.5-0.8B-Q4_0.gguf: Pearson 0.9989 en logits del primer token).
> Fuentes autoritativas: `transformers/modular_qwen3_5.py` (HF), kernel de FLA
> `fla/ops/gated_delta_rule` (naive.py + fused_recurrent.py), y dump del GGUF real.

## 1. Arquitectura del modelo (confirmada contra el 9B real)

### 1.1 Topología del bloque híbrido

El `Qwen3_5DecoderLayer` es **pre-norm residual simple** (sin cross-branch):

```
residual = x
x = attn_norm(x)                      # input_layernorm → "attn_norm.weight"
x = ssm(x)  |  attention(x)           # según layer_types
x = residual + x
residual = x
```

Cada capa tiene:
- `attn_norm` / `post_attention_layernorm`: RMSNorm
- `self_attn`: Qwen2.5-style GQA + IMROPE + DPA
- `mlp`: SwiGLU (gate/up/down)
- `ssm` (solo capas SSM): Gated DeltaNet con `QuantWeight`

### 1.2 Rutado SSM vs Atención

El campo `layer_types` (int8[]) determina por capa:
- `0` = atención completa
- `1` = SSM (DeltaNet)

El routing se implementa en `hybrid_layer.zig` vía `isFullAttentionLayer(idx)`.

### 1.3 Embeddings

- `embed_tokens`: vocab 151,646 (Qwen3.5-0.8B)
- `lm_head`: no tied (separado de embeddings)

### 1.4 Cuantización

- `QuantWeight`: zero-copy weights con dequant a f16/f32 on-demand
- Formatos soportados: Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS, IQ3_S, IQ1_S, MXFP4
- KV cache: q8_0, q4_0, q4_1 (compatible llama.cpp)

## 2. LFM2.5 (ShortConv + Attention)

Arquitectura alternativa con:
- `ShortConv`: depthwise conv1d + gating (K=3)
- `Attention`: Qwen2.5-style GQA + IMROPE
- Routing por `layer_types` igual que Qwen3.5

## 3. Validación

- **Qwen3.5-0.8B-Q4_0.gguf**: Pearson 0.9989 en logits del primer token vs llama.cpp
- **Tests**: `zig build test` → 55/55 passed (con GGUF_MODEL_PATH)
- **Modelo real**: Qwen3.5-0.8B, 27B en progreso

## 4. Rendimiento

- Decode GPU: ~20-75 tok/s (Q4_0, RTX 3080 Laptop sm_86)
- Prefill chunked: batches de 512 tokens con solapamiento causal
- KV cache paged: prefix cache + preemption + CPU offload

## 5. Ver también

- [`../README.md`](../README.md) — build, CLI, estructura del repo
- [`../ROADMAP.md`](../ROADMAP.md) — roadmap público
- [`../CHANGELOG.md`](../CHANGELOG.md) — historial de cambios
- [`airllm-layer-streaming-guide.md`](airllm-layer-streaming-guide.md) — guía de usuario AirLLM layer streaming
- [Esta guía en inglés](qwen35-hybrid-deltanet.md)
