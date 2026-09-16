# Supported Quantization Formats

## Weight quantization

| Format | Bits per weight | Block size | Notes |
|---|---|---|---|
| Q4_0 | 4.5 | 32 | Symmetric, no bias |
| Q4_1 | 5.0 | 32 | Symmetric + scale + min |
| Q5_0 | 5.5 | 32 | Symmetric, 4-bit + 1-bit flag |
| Q5_1 | 6.0 | 32 | Symmetric + scale + min + 1-bit flag |
| Q8_0 | 8.0 | 32 | Symmetric, scale only |
| Q8_1 | 8.5 | 32 | Symmetric + scale + min |
| Q2_K | ~3.5 | 256 | K-quant, asymmetric |
| Q3_K | ~3.5 | 256 | K-quant, asymmetric |
| Q4_K | ~4.5 | 256 | K-quant, asymmetric |
| Q5_K | ~5.5 | 256 | K-quant, asymmetric |
| Q6_K | ~6.5 | 256 | K-quant, asymmetric |
| Q8_K | 8.0 | 256 | K-quant, asymmetric |
| IQ2_XXS | 2.06 | 256 | Extremely low bit |
| IQ2_XS | 2.25 | 256 | Extra low bit |
| IQ3_XXS | 3.0 | 256 | Low bit |
| IQ1_S | 1.56 | 256 | Single bit, super-block |
| IQ2_S | 2.5 | 256 | Single bit |
| IQ3_S | 3.5 | 256 | Single bit |
| IQ1_M | 1.75 | 256 | Medium super-block |
| MXFP4 | 4.0 | 32 | Block-wise 4-bit with per-block scale |

## KV cache quantization

| Format | Bits per element | Notes |
|---|---|---|
| Q8_0 | 8.0 | Symmetric, scale per block |
| Q4_0 | 4.5 | Symmetric, no bias |
| Q4_1 | 5.0 | Symmetric + scale + min |

## Notes

- Block sizes: 32 for basic types, 256 for K-quant and IQ types
- All formats are compatible with llama.cpp GGUF spec
- KV cache formats are used for quantized KV cache storage
- Weight formats are used for model weight dequantization
- [Esta guía en español](quantization.es.md)
