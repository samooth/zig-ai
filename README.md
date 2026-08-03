# Zig AI Engine

Motor de inferencia de transformers en Zig con FlashAttention, matmul multi-backend y cuantización.

## Características

- **Tensor multidimensional**: shape, strides, views, iteradores
- **Matmul multi-backend**: naive, SIMD, tiled, paralelo, OpenBLAS, cuBLAS
- **FlashAttention**: kernels CUDA v1/v2 + implementación CPU de referencia
- **Cuantización**: INT8/INT4 simétrico/asimétrico, per-channel
- **Capa Transformer completa**: proyecciones Q/K/V/O + FA + residual
- **KV-Cache**: para generación autoregresiva
- **Precisión configurable**: f32, f16, bf16, INT8, INT4

## Estructura

```
zig-ai-engine/
├── build.zig              # Build system (detecta CUDA/OpenBLAS automáticamente)
├── build.zig.zon          # Manifesto del paquete
├── src/
│   ├── tensor.zig         # Tensor genérico multidimensional
│   ├── matmul/
│   │   ├── root.zig       # Motor matmul unificado (multi-backend)
│   │   ├── types.zig      # Tipos y configuraciones
│   │   ├── naive.zig      # GEMM naive
│   │   ├── simd.zig       # GEMM con vectorización
│   │   ├── tiled.zig      # GEMM tiled con micro-kernel
│   │   ├── parallel.zig   # GEMM multi-hilo
│   │   ├── openblas.zig   # Wrapper OpenBLAS
│   │   ├── cublas.zig     # Wrapper cuBLAS (async, batch, GemmEx)
│   │   ├── f16bf16.zig    # Conversión f16/bf16
│   │   └── quant.zig      # Cuantización INT8/INT4
│   ├── fa/
│   │   ├── flash_attention.zig  # Motor FA (GPU + CPU)
│   │   ├── fa_config.zig        # Configuración FA
│   │   ├── fa_utils.zig         # RoPE, softmax, utilidades
│   │   └── fa_kernels.zig       # Launchers CUDA
│   ├── transformer/
│   │   ├── layer.zig       # Capa Transformer completa
│   │   ├── pipeline.zig    # Pipeline de capas
│   │   ├── norm.zig        # RMSNorm / LayerNorm
│   │   ├── ffn.zig         # FFN SwiGLU
│   │   ├── rope.zig        # Rotatory Position Embedding
│   │   ├── gqa.zig         # Grouped Query Attention
│   │   └── embedding.zig   # Embedding / lm_head
│   ├── kv_cache/
│   │   ├── kv_cache_manager.zig  # Gestión de KV-Cache cuantizado
│   │   ├── quant_types.zig       # Formatos de cuantización
│   │   └── gpu_dequant.zig       # Dequant en GPU
│   ├── tokenizer/bpe.zig   # Tokenizer BPE
│   ├── loader/safetensors.zig    # Carga de pesos
│   ├── cuda/cudaz_stub.zig       # Bindings CUDA Driver API (stub sin GPU)
│   └── main.zig                  # CLI principal
├── cuda/
│   ├── online_softmax.cuh       # Online softmax (warp/block reduce)
│   ├── matmul_utils.cuh         # Tiles shared-memory
│   ├── flash_attention.cu       # Kernel FA v1
│   ├── flash_attention_v2.cu    # Kernel FA v2
│   └── dequantize_kernels.cu    # Kernels de dequantización
└── tests/
    ├── test_tensor.zig
    ├── test_matmul.zig
    ├── test_flash_attention.zig
    ├── test_online_softmax.zig
    ├── test_transformer.zig
    ├── test_kv_cache.zig
    └── benchmark.zig
```

## Requisitos

- Zig 0.13.0+
- CUDA Toolkit 12.x (opcional, para GPU)
- cuBLAS (opcional)
- OpenBLAS (opcional)

## Compilación

```bash
# CPU only
zig build

# Con CUDA (auto-detectado)
CUDA_PATH=/usr/local/cuda zig build

# Tests
zig build test

# Benchmarks
zig build bench

# Ejecutar
zig build run
```

## Uso

```zig
const TransformerLayer = @import("transformer").TransformerLayer;
const LayerPrecision = @import("transformer").LayerPrecision;

const precision = LayerPrecision{
    .compute = .f16,
    .weights_on_gpu = true,
    .use_quantized = false,
};

var layer = try TransformerLayer.init(allocator, 0, fa_config, "cuda/flash_attention.ptx", 1024, precision);
defer layer.deinit();

try layer.forward(hidden_state, &output);
```

## Backends Matmul

| Backend   | f32 | f64 | f16 | bf16 | INT8 | Async | Batch |
|-----------|-----|-----|-----|------|------|-------|-------|
| naive     | ✓   | ✓   | —   | —    | —    | —     | —     |
| simd      | ✓   | ✓   | —   | —    | —    | —     | —     |
| tiled     | ✓   | —   | —   | —    | —    | —     | —     |
| parallel  | ✓   | ✓   | —   | —    | —    | —     | —     |
| openblas  | ✓   | ✓   | —   | —    | —    | —     | —     |
| cublas    | ✓   | —   | ✓*  | ✓*   | —    | ✓     | ✓     |

*GemmEx con acumulación f32

## Licencia

MIT
