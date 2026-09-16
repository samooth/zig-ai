# Changelog

All notable changes to this project are documented here.

## [Unreleased]

- feat: add CI workflow and GitHub Actions
- feat: update build system and add config example
- feat: add benchmarks and examples
- feat: add CUDA dequantization kernels
- feat: add new engine modules (CUDA, MoE, speculative, vision, server)
- feat: update core modules (tensor, transformer, kv_cache, paged_attention, CUDA)
- feat: add vendored stb_image dependency for vision preprocessing
- docs: add public ROADMAP.md (done + planned)

### 2026-08-20
- feat: add LFM2.5 architecture (ShortConv + Attention hybrid)
- docs: add AirLLM layer streaming user guide
- docs: fix stale claims and complete README CLI table
- docs: add central documentation index
- fix(transformer): LFM2.5 warmup + swiglu/value fix + RoPE f16 generic

### 2026-08-19
- feat: AirLLM-style layer streaming (Phase 1+2)
- feat: Phase 3+4 — ActivationPool + VramBudget in hybrid inference
- feat: spinner progress, GPU name fix, Spanish messages
- fix: use cuDeviceTotalMem_v2 for VRAM reporting (>4GB)
- fix: LayerStreamer deinit scope — prevent use-after-free
- fix: invalidate GPU weight cache on LRU eviction
- feat: remove spinner from generation output
- fix: warm GPU weight caches before CUDA graph capture
- fix: pass 2D tensors to hybrid attention forward
- fix: give FFN post-norm buf real allocator for Tensor.reshape
- docs: record 100/100 test pass and 128-token capture matrix

### 2026-08-18
- fix: CUDA graph prefill UAF + cuGraphInstantiate crash; add STATE.md
- fix(decode): per-layer block table scratch for CUDA graph replay parity
- feat: IQ/q2_k/q3_k/q8_k dequant + 10 more quant types (Phase D-E)
- feat: GPU dequant for 14 GGML types (zig-cuda-agent pattern)
- fix: bench-pa stream lifetime bug + centralized debug breadcrumbs

### 2026-08-17
- feat: llama.cpp-compatible quantized KV cache + CLI (Phase 2)
- feat: real quantized K/V store + dequant-on-read (q8_0/q4_0/q4_1)
- perf(matmul): GPU weight residency cache + dequant-once (~11x faster)
- fix(paged-attention): correct online-softmax accumulator rescaling
- perf(ssm): dequantize SSM weights once at load instead of per token
- perf(hybrid): GPU-resident hybrid layer decode (~20-75 tok/s)
- feat(prefill): GPU-resident chunked prefill for hybrid Qwen3.5
- perf(prefill): batched quantized GEMM kernels (q4_0/q4_1/q5_k/q6_k)
- build: auto-detect GPU architecture instead of hardcoding sm_86
- test: fix ssm fixtures and paged attention GPU tests on Zig 0.16
- docs: mark F1/F2 done, note full-suite green with real model
- perf(decode): cache CUfunction handles + drop decodeDevice stream sync
- perf(ssm): fuse sigmoid+gateCompute and conv1d+silu
- perf(sample): vectorized greedy argmax and skip the logits copy
- perf(ssm): fuse beta/alpha projections into sigmoidGateProj

### 2026-08-16
- feat: prefix cache block reuse + preemption/restore (Phase 5)
- feat: prefix cache hit-rate metrics + CPU offload swap (Phase 5)
- feat: proactive eviction of stale prefix blocks (Phase 5)
- feat: persistent GPU block pool with block-granular stage/evict (Phase 5)
- feat: GPU cold-block eviction driven by prefix-cache hit rate (Phase 5)
- feat: GPU paginated block pool (Phase 5)
- feat: hybrid paginated path by default, benchmarks, GPU pool (Phase 5)
- feat: automatic path detection (removes --legacy) (Phase 5)

### 2026-08-15
- feat: Fase 2: decode vectorizado (LDST.128) y prefill causal batched
- fix: Q4_0 split-layout dequant + tokenizer (llama.cpp parity)
- fix: CUDA context + cuBLAS layout for hybrid inference
- feat: complete E2E pipeline (RoPE, tok/s, cuLaunchKernel cast)
- feat: integrate PagedKVCache into hybrid attention (Phase 1)
- feat: PagedAttention CUDA kernels + GPU engine + tests (Phase 2)
- feat: integrate PagedAttentionGpu.decode into AttentionLayer (Phase 3)
- feat: scheduler integration into runHybridInference + test fixes (Phase 4)

### 2026-08-14
- feat: add hybrid attention layer + fix GGUF loading (Phase H)
- feat: add CUDA bindings, runtime sampling CLI, GPU dequant kernels
- fix: small-model loading (tied lm_head fallback + Q4_1 dequant)
- fix: weight orientations for real GGUF + f16 matmul precision
- fix: SSM decay gate (ssm_a already -exp(A_log), dont double-transform)
- fix: GGUF weight transpose (dim0-contiguous layout)

### 2026-08-13
- feat: add QuantWeight zero-copy + Gated DeltaNet SSM (Phase G)
- feat: add PagedAttention evolution study notes

### 2026-08-04
- feat: integrate PagedAttention (vLLM-style) KV cache module
- refactor: port to Zig 0.16 (build system + stdlib + parallel backend)
- refactor: migrate file I/O to std.Io (readFileAlloc, Dir.access, io threading)
- feat: add GGUF loader Phase C (parser, metadata, dequant, ModelConfig)
- feat: add GGUF mmap loading, tensor mapping, real-model test
- feat: add GGUF tokenizer extraction and BPE feeding (Phase D)
- feat: add Q4_K/Q6_K dequantization (Phase E)
- feat: add GGUF weight loading and CPU forward (Phase E)

### 2026-08-03
- refactor: reorganize monorepo by components
- fix: port to Zig 0.14 (stdlib API + CUDA linking)
- docs: update README
