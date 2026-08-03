# Zig AI Engine — Plan Fase 2: Infraestructura del Modelo

> Fecha: 2026-08-03
> Estado: En progreso
> Objetivo: Cerrar gaps críticos para ejecutar modelos reales (Llama, Mistral, Qwen)

## Gaps críticos de Fase 1 (cerrados en esta sesión)

| # | Gap | Archivo | Estado |
|---|-----|---------|--------|
| 1 | DummyTokenizer inútil | `src/tokenizer/bpe.zig` | Implementar BPE limpio |
| 2 | loadWeightFile con .bin crudos | `src/loader/safetensors.zig` | Parser Safetensors |
| 3 | Sin FFN ni Norm | `src/transformer/ffn.zig`, `norm.zig` | SwiGLU + RMSNorm |
| 4 | Sin Embedding real | `src/transformer/embedding.zig` | Lookup + LM Head |
| 5 | Pipeline autoregresivo frágil | `src/transformer/pipeline.zig` | Sampling, EOS, batch |
| 6 | RoPE naive | `src/transformer/rope.zig` | SIMD + precisión |
| 7 | GQA con copia O(n) | `src/transformer/gqa.zig` | Broadcast nativo |

## Arquitectura objetivo

```
src/
├── tensor.zig              # (existente)
├── matmul/                 # (existente)
├── fa/                     # (existente)
├── kv_cache/               # (existente)
├── transformer/
│   ├── norm.zig            # RMSNorm, LayerNorm
│   ├── ffn.zig             # SwiGLU, GELU
│   ├── embedding.zig       # Token embedding, LM Head
│   ├── rope.zig            # RoPE vectorizado
│   ├── gqa.zig             # GQA broadcast nativo
│   ├── layer.zig           # (actualizar con FFN+Norm)
│   └── pipeline.zig        # Pipeline autoregresivo
├── tokenizer/
│   └── bpe.zig             # Tokenizer BPE
├── loader/
│   └── safetensors.zig     # Parser Safetensors HF
└── main.zig                # (actualizar)
```

## Decisiones técnicas

- **Licencia:** Todo reimplementación limpia MIT. No se copia código AGPL.
- **Safetensors:** Formato abierto de HuggingFace. Parser propio en Zig puro.
- **BPE:** Algoritmo público. Implementación desde la especificación.
- **RMSNorm:** Fórmula estándar `x * rsqrt(mean(x^2) + eps) * weight`.
- **SwiGLU:** `FFN(x) = (silu(x @ W_gate) * (x @ W_up)) @ W_down`.

## Fases siguientes

- **Fase 3:** Optimización (GQA kernel, RoPE SIMD, PagedAttention)
- **Fase 4:** Multi-modelo (Llama, Mistral, Qwen, GPT-2)
- **Fase 5:** Producción (batching, speculative decoding, API)
