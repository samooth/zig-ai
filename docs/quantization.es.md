# Formatos de cuantización soportados

## Cuantización de pesos

| Formato | Bits por peso | Tamaño de bloque | Notas |
|---|---|---|---|
| Q4_0 | 4.5 | 32 | Simétrico, sin bias |
| Q4_1 | 5.0 | 32 | Simétrico + escala + mínimo |
| Q5_0 | 5.5 | 32 | Simétrico, 4-bit + flag 1-bit |
| Q5_1 | 6.0 | 32 | Simétrico + escala + mínimo + flag 1-bit |
| Q8_0 | 8.0 | 32 | Simétrico, solo escala |
| Q8_1 | 8.5 | 32 | Simétrico + escala + mínimo |
| Q2_K | ~3.5 | 256 | K-quant, asimétrico |
| Q3_K | ~3.5 | 256 | K-quant, asimétrico |
| Q4_K | ~4.5 | 256 | K-quant, asimétrico |
| Q5_K | ~5.5 | 256 | K-quant, asimétrico |
| Q6_K | ~6.5 | 256 | K-quant, asimétrico |
| Q8_K | 8.0 | 256 | K-quant, asimétrico |
| IQ2_XXS | 2.06 | 256 | Bit extremadamente bajo |
| IQ2_XS | 2.25 | 256 | Bit muy bajo |
| IQ3_XXS | 3.0 | 256 | Bit bajo |
| IQ1_S | 1.56 | 256 | Un bit, super-bloque |
| IQ2_S | 2.5 | 256 | Un bit |
| IQ3_S | 3.5 | 256 | Un bit |
| IQ1_M | 1.75 | 256 | Super-bloque medio |
| MXFP4 | 4.0 | 32 | 4-bit por bloque con escala por bloque |

## Cuantización de KV cache

| Formato | Bits por elemento | Notas |
|---|---|---|
| Q8_0 | 8.0 | Simétrico, escala por bloque |
| Q4_0 | 4.5 | Simétrico, sin bias |
| Q4_1 | 5.0 | Simétrico + escala + mínimo |

## Notas

- Tamaños de bloque: 32 para tipos básicos, 256 para K-quant y tipos IQ
- Todos los formatos son compatibles con la especificación GGUF de llama.cpp
- Los formatos de KV cache se usan para almacenamiento cuantizado de KV cache
- Los formatos de peso se usan para dequantización de pesos del modelo
- [This guide in English](quantization.md)
