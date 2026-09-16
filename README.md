# Zig AI Engine

[Zig](https://ziglang.org/) · [CUDA](https://developer.nvidia.com/cuda-toolkit) · [MIT License](LICENSE)

> High-performance transformer inference engine in Zig with FlashAttention, multi-backend matmul, and llama.cpp-compatible quantization.

---

## Features

| | |
|---|---|
| **Tensor core** | Multidimensional tensor with shape, strides, views, iterators |
| **Matmul** | Multi-backend: naive, SIMD, tiled, parallel, OpenBLAS, cuBLAS |
| **FlashAttention** | CUDA v1/v2 kernels + CPU reference |
| **Quantization** | Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS, IQ3_S, IQ1_S, MXFP4 and more (see [quantization.md](docs/quantization.md)) |
| **KV-Cache** | Quantized, paged, prefix cache, preemption, CPU offload |
| **Hybrid blocks (Qwen3.5)** | Gated DeltaNet / recurrent SSM + attention with GQA and IMROPE |
| **Hybrid blocks (LFM2.5)** | ShortConv (depthwise conv1d + gating) + standard attention with GQA |
| **Layer streaming** | AirLLM-style async layer-by-layer weight loading, prefetch, LRU eviction |
| **GGUF loader** | Full parser, metadata, dequant, ModelConfig, MMAP, BPE tokenizer |
| **GPU kernels** | GPU dequant, quantized GEMM, SSM fusion, vectorized sampling, CUDA graphs |
| **Prefill** | Chunked GPU prefill with correct causal overlap |
| **Speculative decoding** | Draft-model verification (DFlash), adaptive draft-max, lookup-fill |
| **MoE support** | Expert routing, CPU executor, bandwidth-aware offload, hybrid split |
| **Vision encoder** | CLIP ViT, Qwen-VL projector, m-rope vision, image/video preprocessing |
| **Server mode** | OpenAI/Anthropic/Ollama-compatible HTTP API, SSE streaming, batching, chat templates |
| **CI/CD** | GitHub Actions workflow for Zig 0.16 + CUDA testing |
| **Benchmarks** | Adaptive bench, controller overhead, KLD, paged attention, PP512, sweep |
| **Examples** | Adaptive bench, overhead analysis, MoE bench, stream bench |
| **Vendored deps** | stb_image for image loading |
| **Precision** | f32, f16, bf16, INT8, INT4 |

---

## Requirements

| Dependency | Version | Purpose |
|---|---|---|
| Zig | 0.16.0 | Toolchain |
| CUDA Toolkit | 12.x | GPU acceleration (optional) |
| cuBLAS | — | cuBLAS backend (optional) |
| OpenBLAS | — | OpenBLAS backend (optional) |

---

## Build

```bash
# CPU only
zig build

# With CUDA (auto-detected)
CUDA_PATH=/usr/local/cuda zig build

# Force GPU architecture
zig build -Dgpu-arch=sm_89

# Tests
zig build test

# Benchmarks
zig build bench

# Run
zig build run
```

> **Note:** on filesystems without `renameat2(RENAME_EXCHANGE)` support (e.g. ecryptfs),
> `zig build` fails at the options stage. Redirect the cache:
> `zig build --cache-dir /tmp/ziglocal --global-cache-dir /tmp/zigglobal`.

---

## CLI Usage

Run inference end-to-end with a GGUF model, or use the benchmark mode.

```bash
# Inference
./zig-out/bin/zig-ai-engine \
 --model model.gguf --prompt "Hello, world" -n 256 \
 --temperature 0.7 --top-k 40 --top-p 0.9 --repetition-penalty 1.1 --seed 7

# Help
./zig-out/bin/zig-ai-engine --help
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `-m`, `--model <path>` | — | Path to GGUF model |
| `--prompt <text>` | `"Hello"` | Input prompt |
| `-n`, `--max-tokens <n>` | 128 | Max tokens to generate |
| `--temperature <f>` | 1.0 | Temperature (`≤0` = greedy) |
| `--top-k <n>` | 0 | Top-k (0 = disabled) |
| `--top-p <f>` | 1.0 | Top-p / nucleus (1.0 = disabled) |
| `--repetition-penalty <f>` | 1.0 | Repetition penalty |
| `--seed <n>` | 42 | RNG seed |
| `--backend <auto\|cpu\|gpu>` | auto | Matmul backend |
| `-c`, `--ctx-size <n>` | 65536 | Context window size (0 = trained context) |
| `-b`, `--batch-size <n>` | 2048 | Logical prefill batch (tokens) |
| `-ub`, `--ubatch-size <n>` | 512 | Physical batch per GPU call |
| `-ngl`, `--n-gpu-layers <n>` | auto | Layers to offload to GPU |
| `-np <n>` | 1 | Parallel sequences |
| `--quant <auto\|off\|fp8>` | auto | Quantized weight GEMM on GPU |
| `--mmproj <path>` | — | Vision encoder GGUF (CLIP ViT) |
| `--image <path>` | — | Input image (repeatable for multi-image) |
| `--video <path>` | — | Input video (Qwen-VL temporal merge) |
| `-ctk`, `--cache-type-k <fmt>` | fp16 | K cache quantization |
| `-ctv`, `--cache-type-v <fmt>` | fp16 | V cache quantization |
| `--spec-draft-type-k <fmt>` | fp16 | Draft K cache quantization |
| `--spec-draft-type-v <fmt>` | fp16 | Draft V cache quantization |
| `--spec-type <mode>` | none | Speculative decoding mode |
| | | `none`, `draft-mtp`/`mtp`, `draft-dflash`/`dflash`, `draft-dspark`/`dspark`, `draft-dflash2`/`dflash2` |
| `--model-draft <path>` | — | Draft model GGUF (DFlash/DSpark) |
| `--spec-draft-n-max <n>` | 16 | Max draft tokens per round |
| `--spec-draft-n-min <n>` | 4 | Min draft tokens accepted per round |
| `--spec-draft-block-size <n>` | 0 | Block size for DFlash (0 = auto from GGUF) |
| `--spec-p-min <f>` | 0.1 | Min probability for draft acceptance |
| `--spec-dm-controller <off\|profit>` | profit | Adaptive draft-max controller |
| `--spec-dm-profit-baseline-interval <n>` | 1024 | Re-baseline interval for adaptive controller |
| `--reasoning-loop-mode <off\|force-close\|warn>` | force-close | Loop guard mode |
| `--reasoning-loop-window <n>` | 64 | Loop detection window |
| `--reasoning-loop-max-period <n>` | 16 | Max loop period before cutoff |
| `--reasoning-loop-channel <hidden\|visible\|both>` | hidden | Loop guard output channel |
| `--spec-lookup-n <n>` | 5 | Prompt-lookup n-gram fill (0 = disabled) |
| `--spec-selector-top-k <n>` | 10 | Top-k for speculative token selector |
| `--spec-selector-rank <n>` | 128 | Rank budget for speculative selector |
| `--layer-stream` | off | Enable AirLLM-style layer streaming |
| `--layer-stream-max <n>` | 2 | Max resident layers in VRAM |
| `--f16-max-resident <n>` | 2 | Max F16 resident layers (LRU eviction) |
| `--kv-transfer <path>` | — | KV-transfer weights (.ktb) |
| `--rlt-sidecar <path>` | — | RLT sidecar GGUF |
| `--rlt-feedback` | auto | Force RLT feedback ON |
| `--no-rlt-feedback` | auto | Force RLT feedback OFF |
| `--recurrent-prefill` | off | Recurrent prefill (RLT) |
| `--exact-replay` | off | Exact replay for speculative decoding |
| `--swa <n>` | 0 | Sliding window attention (0 = full context) |
| `--download-dflash` | off | Download DFlash model |
| `--download-dspark` | off | Download DSPark model |
| `--download-dflash2` | off | Download DFlash2 model |
| `-jinja`, `--jinja` | off | Use Jinja chat template |
| `--preset <path>` | — | Preset INI path |
| `--models-dir <dir>` | — | Models directory |
| `--models-preset <path>` | — | Preset within models directory |
| `--serve` | off | Enable HTTP server mode (OpenAI/Anthropic/Ollama). Non-loopback bind requires `--tls-cert`/`--tls-key` |
| `--host <addr>` | 127.0.0.1 | Server bind address |
| `--port <n>` | 8080 | Server bind port |
| `--api-key-file <path>` | — | API key file (Bearer auth) |
| `--audit-log <path>` | — | Audit log JSON-lines |
| `--rate-limit <rpm>` | 60 | Rate limit requests/min per IP |
| `--tls-cert <path>` | — | TLS certificate (PEM) |
| `--tls-key <path>` | — | TLS key (PEM) |
| `--no-warmup` | off | Disable server warmup |
| `--ppl <file>` | — | Perplexity eval file |
| `--capture-rlt <path>` | — | Capture RLT hidden states |
| `--dump-lm-head <path>` | — | Dump LM head weights |
| `--dump-logits-target <path>` | — | Dump target logits for RLT training |
| `-h`, `--help` | — | Show help |
| `-v`, `--version` | — | Show version |
Prompt prefill runs on GPU in chunks of `--ubatch-size` tokens with correct causal overlap.
Env vars: `NOGPU_PREFILL=1`, `NOQ4=1`, `NOQ4SSM=1`, `NOQ4ATTN=1`, `NOQ4FFN=1`.

Sampler order: repetition penalty → temperature → top-k → top-p → multinomial (or greedy if `temperature ≤ 0`).

---

## Documentation

| Doc | Language |
|---|---|
| [README](README.md) | English |
| [README.es.md](README.es.md) | Español |
| [docs/README.md](docs/README.md) | Documentation index |
| [docs/airllm-layer-streaming-guide.md](docs/airllm-layer-streaming-guide.md) | User guide (EN) |
| [docs/airllm-layer-streaming-guide.es.md](docs/airllm-layer-streaming-guide.es.md) | Guía de usuario (ES) |
| [docs/qwen35-hybrid-deltanet.md](docs/qwen35-hybrid-deltanet.md) | Architecture (EN) |
| [docs/qwen35-hybrid-deltanet.es.md](docs/qwen35-hybrid-deltanet.es.md) | Arquitectura (ES) |
| [docs/quantization.md](docs/quantization.md) | Quantization formats (EN) |
| [docs/quantization.es.md](docs/quantization.es.md) | Formatos de cuantización (ES) |
| [docs/server.md](docs/server.md) | Server mode (EN) |
| [docs/server.es.md](docs/server.es.md) | Modo servidor (ES) |
| [ROADMAP.md](ROADMAP.md) | Roadmap (EN) |
| [ROADMAP.es.md](ROADMAP.es.md) | Hoja de ruta (ES) |
| [CHANGELOG.md](CHANGELOG.md) | Changelog |

---

## License

MIT
