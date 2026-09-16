# Changelog

## v0.1.0 — 2026-09-15

**Scope**: Qwen3.5 family (0.8B/2B/4B/9B/27B)

### Gates de release
- G3 family smoke: 5/5 PASS
- G4 27B streaming: PASS (178 layers)
- G5 KT-B identity: PASS
- G6b golden llama.cpp: PPL 11.34 ± 1.09 (ctx 1024, wiki12k)
- G6 PPL golden Qwen3.5-0.8B: **9.9705** PASS

### Paridad familiar R2 (PPL wiki12k, fp16 KV)
- 0.8B Q4_0: **9.9705** ✅
- 2B Q3_K_M: **7.8429** ✅
- 4B Q4_0: **6.3025** ✅
- 9B UD-IQ2_M: **7.3512** ✅

### Rendimiento decode
- IQ dp4a kernels cases 8/9/18 (iq3_s/iq2_s/iq4_xs) — P0-5 @70adfc7
- q4_0 gate fix: dp4a ahora aplica para n>=1024 (antes n>=2048) — @b88d90d
- 0.8B decode: 6.00 → 4.69 ms/tok (+22%)
- 9B decode: 3.1 tok/s (iq2_s scalar baseline; dp4a en progresión P0-6)

### Infra
- GPU sampler VRAM/GPUutil peaks + bench-history.csv (@42a4101)
- A5 ZIG_AI_NO_LEAK_REPORT flag para train binaries (@d84db8e)
- q4_0 dp4a para n>=1024: fix @b88d90d
- IQ4_XS mapping fix: rowstride type 18 + case 18 escalar — 4B IQ4_XS 24.7 tok/s (@0a054d2)
- KVarN D64 Manager Integration: head_dim 64/128/256 + layout rect-64 (@33ba484)
- R-2 --exact-replay NOGRAPH: deterministic replay sin draft cache (@0180077)
- R2 runHybridPpl KV q4_k multi-ubatch: honor -ctk/-ctv en PPL path (@91cdf42)

### Bugs fixeados
- IQ4_XS rowstride type 18 missing → LAUNCH_FAILED (@0a054d2)
- moe/cache.zig string literal multiline roto → compilación fallaba (@049103d)
- qgemmTypeFor mapping incompleto (0..17 → 0..18) (@0a054d2)
