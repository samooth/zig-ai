# Zig AI Engine

[Zig](https://ziglang.org/) · [CUDA](https://developer.nvidia.com/cuda-toolkit) · [MIT License](LICENSE)

> Motor de inferencia de transformers en Zig con FlashAttention, matmul multi-backend y soporte de cuantización compatible con llama.cpp.

---

## Características

| | |
|---|---|
| **Tensor** | Tensor multidimensional con shape, strides, views, iteradores |
| **Matmul** | Multi-backend: naive, SIMD, tiled, paralelo, OpenBLAS, cuBLAS |
| **FlashAttention** | Kernels CUDA v1/v2 + implementación CPU de referencia |
| **Cuantización** | Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS, IQ3_S, IQ1_S, MXFP4 y más (ver [quantization.es.md](docs/quantization.es.md)) |
| **KV-Cache** | Cuantizada, paged, prefix cache, preemption, CPU offload |
| **Bloques híbridos (Qwen3.5)** | Gated DeltaNet / SSM recurrente + atención con GQA e IMROPE |
| **Bloques híbridos (LFM2.5)** | ShortConv (depthwise conv1d + gating) + atención estándar con GQA |
| **Layer streaming** | Carga asíncrona de pesos capa a capa, prefetch, LRU eviction |
| **GGUF loader** | Parser completo, metadata, dequant, ModelConfig, MMAP, tokenizer BPE |
| **GPU kernels** | Dequant GPU, GEMM cuantizado, SSM fusion, sampling vectorizado, CUDA graphs |
| **Prefill** | Prefill chunked GPU con solapamiento causal |
| **Speculative decoding** | Draft-model verification (DFlash), adaptive draft-max, lookup-fill |
| **MoE support** | Expert routing, CPU executor, bandwidth-aware offload, hybrid split |
| **Vision encoder** | CLIP ViT, Qwen-VL projector, m-rope vision, image/video preprocessing |
| **Server mode** | OpenAI/Anthropic/Ollama-compatible HTTP API, SSE streaming, batching, chat templates |
| **CI/CD** | GitHub Actions workflow for Zig 0.16 + CUDA testing |
| **Benchmarks** | Adaptive bench, controller overhead, KLD, paged attention, PP512, sweep |
| **Examples** | Adaptive bench, overhead analysis, MoE bench, stream bench |
| **Vendored deps** | stb_image for image loading |
| **Precisión** | f32, f16, bf16, INT8, INT4 |

---

## Requisitos

| Dependencia | Versión | Propósito |
|---|---|---|
| Zig | 0.16.0 | Toolchain |
| CUDA Toolkit | 12.x | Aceleración GPU (opcional) |
| cuBLAS | — | Backend cuBLAS (opcional) |
| OpenBLAS | — | Backend OpenBLAS (opcional) |

---

## Compilación

```bash
# Solo CPU
zig build

# Con CUDA (auto-detectado)
CUDA_PATH=/usr/local/cuda zig build

# Forzar arquitectura GPU
zig build -Dgpu-arch=sm_89

# Tests
zig build test

# Benchmarks
zig build bench

# Ejecutar
zig build run
```

> **Nota:** en filesystems sin soporte de `renameat2(RENAME_EXCHANGE)` (p. ej. ecryptfs),
> `zig build` falla en la etapa de opciones. Redirige la caché:
> `zig build --cache-dir /tmp/ziglocal --global-cache-dir /tmp/zigglobal`.

---

## Uso (CLI)

```bash
# Inferencia
./zig-out/bin/zig-ai-engine \
 --model modelo.gguf --prompt "Hola, mundo" -n 256 \
 --temperature 0.7 --top-k 40 --top-p 0.9 --repetition-penalty 1.1 --seed 7

# Ayuda
./zig-out/bin/zig-ai-engine --help
```

### Flags

| Flag | Default | Descripción |
|---|---|---|
| `-m`, `--model <ruta>` | — | Ruta a modelo GGUF |
| `--prompt <texto>` | `"Hola"` | Prompt de entrada |
| `-n`, `--max-tokens <n>` | 128 | Máx. tokens a generar |
| `--temperature <f>` | 1.0 | Temperatura (`≤0` = greedy) |
| `--top-k <n>` | 0 | Top-k (0 = desactivado) |
| `--top-p <f>` | 1.0 | Top-p / nucleus (1.0 = desactivado) |
| `--repetition-penalty <f>` | 1.0 | Repetition penalty |
| `--backend <auto\|cpu\|gpu>` | auto | Backend matmul |
| `--seed <n>` | 42 | Semilla del RNG |
| `-c`, `--ctx-size <n>` | 65536 | Contexto de inferencia (0 = contexto entrenado) |
| `-b`, `--batch-size <n>` | 2048 | Batch lógico de prefill (tokens) |
| `-ub`, `--ubatch-size <n>` | 512 | Batch físico por llamada GPU |
| `-ngl`, `--n-gpu-layers <n>` | auto | Capas a offload a GPU |
| `-np <n>` | 1 | Secuencias paralelas |
| `--quant <auto\|off\|fp8>` | auto | GEMM de pesos cuantizados en GPU |
| `--mmproj <path>` | — | Encoder visión GGUF (CLIP ViT) |
| `--image <path>` | — | Imagen de entrada (repetible para multi-imagen) |
| `--video <path>` | — | Video de entrada (Qwen-VL temporal merge) |
| `-ctk`, `--cache-type-k <fmt>` | fp16 | Cuantización cache K |
| `-ctv`, `--cache-type-v <fmt>` | fp16 | Cuantización cache V |
| `--spec-draft-type-k <fmt>` | fp16 | Draft K cache quantization |
| `--spec-draft-type-v <fmt>` | fp16 | Draft V cache quantization |
| `--spec-type <mode>` | none | Modo speculative decoding |
| | | `none`, `draft-mtp`/`mtp`, `draft-dflash`/`dflash`, `draft-dspark`/`dspark`, `draft-dflash2`/`dflash2` |
| `--model-draft <path>` | — | Modelo GGUF draft sidecar (DFlash/DSpark) |
| `--spec-draft-n-max <n>` | 16 | Máx. tokens de draft por round |
| `--spec-draft-n-min <n>` | 4 | Mín. tokens de draft aceptados por round |
| `--spec-draft-block-size <n>` | 0 | Block size para DFlash (0 = auto desde GGUF) |
| `--spec-p-min <f>` | 0.1 | Probabilidad mínima para aceptar draft |
| `--spec-dm-controller <off\|profit>` | profit | Controlador adaptive draft-max |
| `--spec-dm-profit-baseline-interval <n>` | 1024 | Intervalo de re-baseline para controlador |
| `--reasoning-loop-mode <off\|force-close\|warn>` | force-close | Modo loop guard |
| `--reasoning-loop-window <n>` | 64 | Ventana de detección de loop |
| `--reasoning-loop-max-period <n>` | 16 | Período máximo de loop antes de cutoff |
| `--reasoning-loop-channel <hidden\|visible\|both>` | hidden | Canal de salida del loop guard |
| `--spec-lookup-n <n>` | 5 | Prompt-lookup n-gram fill (0 = desactivado) |
| `--spec-selector-top-k <n>` | 10 | Top-k para selector de tokens especulativo |
| `--spec-selector-rank <n>` | 128 | Rank budget para selector especulativo |
| `--layer-stream` | off | Activa layer streaming (AirLLM-style) |
| `--layer-stream-max <n>` | 2 | Máx. capas residentes en VRAM |
| `--f16-max-resident <n>` | 2 | Máx. capas F16 residentes (eviction LRU) |
| `--kv-transfer <path>` | — | Pesos KV-transfer (.ktb) |
| `--rlt-sidecar <path>` | — | Sidecar GGUF RLT |
| `--rlt-feedback` | auto | Forzar feedback RLT ON |
| `--no-rlt-feedback` | auto | Forzar feedback RLT OFF |
| `--recurrent-prefill` | off | Prefill secuencial recurrente (RLT) |
| `--exact-replay` | off | Replay exacto para speculative decode |
| `--swa <n>` | 0 | Sliding window attention por capa (0 = full context) |
| `--download-dflash` | off | Auto-descargar sidecar DFlash de HF |
| `--download-dspark` | off | Auto-descargar sidecar DSPark de HF |
| `--download-dflash2` | off | Auto-descargar sidecar DFlash2 de HF |
| `-jinja`, `--jinja` | off | Usar template chat Jinja del tokenizer |
| `--preset <path.ini>` | — | Ruta a preset INI |
| `--models-dir <dir>` | — | Directorio de modelos |
| `--models-preset <path.ini>` | — | Preset dentro de --models-dir |
| `--serve` | off | Activa modo servidor HTTP (OpenAI/Anthropic/Ollama). Bind no-loopback requiere `--tls-cert`/`--tls-key` |
| `--host <ip>` | 127.0.0.1 | Bind del servidor |
| `--port <n>` | 8080 | Puerto del servidor |
| `--api-key-file <ruta>` | — | API keys (una por línea, permisos 0600) |
| `--audit-log <ruta>` | — | Audit JSON-lines (fail-soft) |
| `--rate-limit <n>` | 60 | Rate limit req/min por IP |
| `--tls-cert <ruta>` | — | Certificado TLS PEM (bind público) |
| `--tls-key <ruta>` | — | Clave privada TLS PEM (bind público) |
| `--no-warmup` | off | Salta la inferencia de warmup inicial |
| `--ppl <archivo>` | — | Eval de perplexidad |
| `--capture-rlt <ruta>` | — | Captura hidden states a .rltcap para entrenamiento RLT |
| `--dump-lm-head <ruta>` | — | Vuelca lm_head f32 a .bin para entrenamiento RLT |
| `--dump-logits-target <ruta>` | — | Vuelca logits target a .bin para entrenamiento RLT MSE |
| `-h`, `--help` | — | Muestra esta ayuda |
| `-v`, `--version` | — | Muestra versión |
El prefill corre en GPU por chunks de `--ubatch-size` tokens con solapamiento causal.
Bind no-loopback del servidor requiere `--tls-cert`/`--tls-key`.
Env vars: `NOGPU_PREFILL=1`, `NOQ4=1`, `NOQ4SSM=1`, `NOQ4ATTN=1`, `NOQ4FFN=1`.

Sampler: repetition penalty → temperature → top-k → top-p → multinomial (o greedy si `temperature ≤ 0`).

---

## Documentación

| Doc | Idioma |
|---|---|
| [README](README.md) | English |
| [README.es.md](README.es.md) | Español |
| [docs/README.md](docs/README.md) | Índice de documentación |
| [docs/airllm-layer-streaming-guide.md](docs/airllm-layer-streaming-guide.md) | Guía usuario (EN) |
| [docs/airllm-layer-streaming-guide.es.md](docs/airllm-layer-streaming-guide.es.md) | Guía usuario (ES) |
| [docs/qwen35-hybrid-deltanet.md](docs/qwen35-hybrid-deltanet.md) | Arquitectura (EN) |
| [docs/qwen35-hybrid-deltanet.es.md](docs/qwen35-hybrid-deltanet.es.md) | Arquitectura (ES) |
| [docs/quantization.md](docs/quantization.md) | Formatos de cuantización (EN) |
| [docs/quantization.es.md](docs/quantization.es.md) | Formatos de cuantización (ES) |
| [docs/server.md](docs/server.md) | Modo servidor (EN) |
| [docs/server.es.md](docs/server.es.md) | Modo servidor (ES) |
| [ROADMAP.md](ROADMAP.md) | Hoja de ruta (EN) |
| [ROADMAP.es.md](ROADMAP.es.md) | Hoja de ruta (ES) |
| [CHANGELOG.md](CHANGELOG.md) | Historial de cambios |

---

## Licencia

MIT
