# AirLLM Layer Streaming — User Guide

## Overview

**AirLLM Layer Streaming** allows running large models that don't fit in VRAM by streaming layer weights on-demand from host RAM to GPU. The engine keeps only `N` layers resident in VRAM (configurable via `--layer-stream-max`), loading the next layer asynchronously while the GPU computes the current one, and evicting the least-recently-used layer when the budget is exceeded.

This enables running models larger than VRAM (e.g., Qwen3.5-0.8B Q4_0 on a 8GB GPU).

---

## Quick Start

```bash
# Enable layer streaming with max 2 resident layers
./zig-out/bin/zig-ai-engine \
  --model /path/to/model.gguf \
  --prompt "Tu prompt aquí" \
  --max-tokens 256 \
  --layer-stream \
  --layer-stream-max 2

# Or auto-detect VRAM budget (uses cuDeviceTotalMem)
./zig-out/bin/zig-ai-engine \
  --model /path/to/model.gguf \
  --prompt "Tu prompt aquí" \
  --layer-stream
```

### Key Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--layer-stream` | off | Activa el streaming de capas (carga asíncrona + LRU eviction) |
| `--layer-stream-max <N>` | 2 | Máximo de capas residentes simultáneamente en VRAM |

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VRAM (max 2 layers)                          │
│  ┌─────────────┐   ┌─────────────┐                                  │
│  │  Layer i    │   │  Layer i+1  │  ← Prefetched while Layer i runs │
│  │  (active)   │   │  (loaded)   │                                  │
│  └─────────────┘   └─────────────┘                                  │
│  Layer i-1 evicted (LRU)                                            │
└─────────────────────────────────────────────────────────────────────┘
                              ↑
                    Host RAM (all weights mmap'd)
```

1. **Prefetch**: While GPU computes layer `i`, the streamer spawns a thread to load layer `i+1` weights from host RAM → GPU.
2. **Compute**: GPU executes layer `i` with weights already in VRAM.
3. **Eviction**: If `resident_count > max_resident`, the least-recently-used layer is unloaded (weights freed, GPU cache invalidated).
4. **Graph Capture**: Before CUDA graph capture, all layers are pre-loaded and their GPU weight caches are warmed so capture doesn't trigger synchronous `cudaMemcpy`.

---

## Configuration

### VRAM Budget

The engine auto-detects total VRAM via `cuDeviceTotalMem` and splits it:
- **60% weights** (layer weights)
- **20% activations** (intermediate tensors)
- **20% KV-cache** (PagedAttention)
- **5% safety margin**

Override with `--layer-stream-max N` to control max resident layers (lower = less VRAM, more streaming overhead).

### Cache & Quantization Options

| Flag | Default | Effect |
|------|---------|--------|
| `--quant <auto|off>` | auto | Enable Q4_0 GEMM for attention/FFN (saves 8x VRAM bandwidth) |
| `--cache-type-k <fmt>` | fp16 | KV-cache K quantization (q4_0, q4_1, q8_0, fp16) |
| `--cache-type-v <fmt>` | fp16 | KV-cache V quantization |
| `--quant off` | — | Disable Q4 GEMM (use f32/f16 weights) |

---

## Usage Examples

### Basic Streaming (2 layers resident)

```bash
./zig-out/bin/zig-ai-engine \
  --model /models/qwen3.5-0.8b-q4_0.gguf \
  --prompt "Escribe un cuento sobre un gato" \
  --max-tokens 512 \
  --layer-stream \
  --layer-stream-max 2
```

### Aggressive Streaming (1 layer resident - minimal VRAM)

```bash
./zig-out/bin/zig-ai-engine \
  --model /models/qwen3.5-0.8b-q4_0.gguf \
  --prompt "Resume el texto: ..." \
  --max-tokens 1024 \
  --layer-stream \
  --layer-stream-max 1 \
  --cache-type-k q4_0 --cache-type-v q4_0
```

### Debug Output

Enable debug logging for the streamer:

```bash
DEBUG_LEVEL=1 ./zig-out/bin/zig-ai-engine \
  --model model.gguf --prompt "Hola" \
  --layer-stream --layer-stream-max 2
```

Output:
```
[+] LayerStreamer activado: max_resident=2 vram=7840MB
[LayerStreamer] layer 0 ensured loaded
[LayerStreamer] async load layer 1 OK (resident=2)
...
```

---

## CUDA Graph Capture with Streaming

When `--layer-stream` is active, the engine **still captures the decode graph** (same as eager mode). The capture works because:

1. **Pre-capture warm-up**: All layers are loaded + `warmupGpuWeights()` runs for every layer before `beginCapture`.
2. **Warm-up**: Calls `projectionDevicePtr` on every f32 weight that `forwardGPU` would cache, so the GPU weight cache is hot.
3. **No eviction during capture**: All layers are resident; eviction is disabled during capture.
4. **Replay**: Graph launch never evicts (streamer not used during replay).

> ⚠️ **Debug Dumps**: With `DUMP_PREFILL_LAYERS=1`, graph capture **intentionally fails** (dumps do sync `cuMemcpyDtoH` inside capture). The engine falls back to eager launch; output is numerically identical.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `CudaMemcpyFailed` during capture | Weight cache miss during graph capture | Ensure `--layer-stream` warm-up runs (auto with `--layer-stream`) |
| OOM on 8GB GPU | Too many resident layers | Reduce `--layer-stream-max` (try 1) |
| Slow generation | Too few resident layers / frequent eviction | Increase `--layer-stream-max` (try 4) |
| `LoadTimeout` error | Async load thread hung/panicked | Check `loadWeightsFromGguf` for errors; increase `max_spins` in `ensureLayerLoaded` |
| Graph capture fails | Debug dumps enabled | Unset `DUMP_PREFILL_LAYERS` |

---

## Performance Tuning

| Scenario | Recommended `--layer-stream-max` | Notes |
|----------|----------------------------------|-------|
| 8GB GPU, Q4_0 model (7B params) | 2 | Balance VRAM vs overhead |
| 8GB GPU, 13B+ params | 1 | Minimum VRAM |
| 12GB+ GPU | 4+ | More layers = less streaming overhead |
| Debugging | 24 (all layers) | Disable streaming effectively |

---

## Internals (For Developers)

Key files:
- `src/transformer/layer_streamer.zig` — `LayerStreamer` struct, async load, LRU
- `src/main.zig` — `runHybridInference`, streamer integration, capture warm-up
- `src/transformer/hybrid_layer.zig` — `warmupGpuWeights()`, `forwardGPU`
- `src/transformer/ssm.zig` / `hybrid_attn.zig` — `warmupGpuWeights()` per sub-layer
- `src/matmul/root.zig` — `projectionDevicePtr`, `clearWeightCache`

### Adding a New Layer Type
1. Implement `warmupGpuWeights()` in the layer's Zig file
2. Call `projectionDevicePtr` on all f32 weights that `forwardGPU` caches
2. Match the `q4_ok`/`w_down_q4` guards exactly as in `forwardGPU`
4. The streamer will call it automatically during pre-capture warm-up

---

## Environment Variables

| Variable | Effect |
|----------|--------|
| `GGUF_MODEL_PATH` | Required for integration tests |
| `DUMP_PREFILL_LAYERS=1` | Enable debug dumps (disables graph capture) |
| `NOQ4=1` | Disable all Q4 GEMM |
| `NOQ4SSM=1` | Disable Q4 in SSM |
| `NOQ4ATTN=1` | Disable Q4 in attention |
| `NOQ4FFN=1` | Disable Q4 in FFN |
| `NOGPU_PREFILL=1` | Force CPU prefill |

---

## References

- [Implementation Plan](AIRLLM_IMPLEMENTATION_PLAN.md) — Detailed status, risk assessment
- [Qwen3.5 Hybrid DeltaNet](qwen35-hybrid-deltanet.md) — Architecture
- [PagedAttention TODO](PAGED_ATTENTION_TODO.md) — PagedAttention roadmap