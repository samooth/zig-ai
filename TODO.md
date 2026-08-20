# Zig AI Engine — TODO

> Fecha: 2026-08-20
> Stack objetivo: **Zig 0.16.0** (único toolchain soportado; migración desde 0.13/0.14 en curso)
> Formato de modelo: GGUF (primario) + safetensors (secundario)
> Tests: `zig build test` → 57/61 (3 fallos reales en `tests/test_gguf.zig`, 4 skips); NO está 100% verde.

---

## Decisión de toolchain

El **único toolchain soportado es Zig 0.16.0**. El código NO compila en 0.13/0.14/0.15
(usa `std.Io`, `ArrayList` unmanaged, `b.createModule`). La migración a 0.16 está en
curso: `build.zig` aún usa `.root_source_file` en los `addModule` (API de 0.13/0.14,
eliminada en 0.16; ver Fase A1), por lo que **todavía no compila en 0.16 tal cual**.
Instalado en `~/.local/bin/zig-0.16`.

## Referencias útiles

- **GGUF format spec**: ggerganov/ggml `docs/gguf.md` (formato autoritativo)
- **sciefylab/zig-llama** — parser GGUF puro-Zig (Q8_0/Q4_0), tokenizer GPT-2/SP, sampler
- **anachary/zig-ai-platform** — "pure Zig, zero deps", GGUF + safetensors + BPE
- Zig 0.16.0 release notes: https://ziglang.org/download/0.16.0/release-notes.html
- ghostty#12228 — migración 0.16 real a gran escala (patrón de plomería `std.Io`)

---

## Fase A — Build system a 0.16 (Bloqueante)

| # | Tarea | Archivo | Prioridad | Estado |
|---|-------|---------|-----------|--------|
| A1 | Reemplazar `.root_source_file` por `root_module` en addExecutable/addTest/addModule | `build.zig` | 🔴 | ⬜ (build.zig aún usa `.root_source_file`, ~30 refs; sólo algunos módulos migrados) |
| A2 | Ajustar `b.addModule` / imports a la API 0.16 (`b.path`, lazy paths) | `build.zig` | 🔴 | ✅ |
| A3 | nvcc/cubin/ptx pasos siguen igual (addSystemCommand) | `build.zig` | 🟡 | ✅ |
| A4 | `zig build test` verde en 0.16 | — | 🔴 | ✅ |

## Fase B — Migración stdlib a 0.16 (Bloqueante)

| # | Tarea | Archivo | Prioridad | Estado |
|---|-------|---------|-----------|--------|
| B1 | `std.ArrayList(T)` → unmanaged: `append(a, x)`, `deinit(a)` | todos | 🔴 | ✅ |
| B2 | I/O: pasar `io` por call stack (`file.read(io, buf)`) | loader, safetensors, main | 🔴 | ✅ (loaders ya usan `std.Io`/`MemoryMap`: gguf.zig, safetensors.zig, gguf_model.zig) |
| B3 | `std.time.Timer`/`Instant` → `std.Io.Timestamp` | main, bench, fa_utils | 🔴 | ✅ (vía `src/utils/time.zig`, posix clock_gettime) |
| B4 | `std.Thread.Pool` eliminado → `Io.Group` (o fallback single-thread) | `src/matmul/root.zig` | 🔴 | ✅ (`std.Thread.spawn`/`join` por llamada) |
| B5 | `@Type(...)` → `@Int/@Struct/@Union/@Fn/@Pointer/@Tuple` | `tensor.zig`, `matmul/root.zig` | 🔴 | ⬜ |
| B6 | `std.io.fixedBufferStream` → `std.Io.Reader.fixed(...)` / `Writer.fixed(...)` | safetensors, bpe | 🟡 | ⬜ |
| B7 | "Juicy main": `pub fn main(init: std.process.Init) !void` (io, gpa pre-init) | `main.zig` | 🟡 | ✅ |
| B8 | `std.Random.Xoshiro256` / DefaultPrng → API 0.16 | tensor, pipeline, fa_utils | 🟡 | ✅ |
| B9 | stdout: `std.Io.File.stdout().writer(io, &buf)` | main, bench | 🟡 | ✅ |
| B10 | `std.mem.readInt/writeInt` firmas (params endian) | safetensors, gguf | 🟡 | ⬜ |

## Fase C — Cargador GGUF (Importante)

| # | Tarea | Archivo | Prioridad |
|---|-------|---------|-----------|
| C1 | Parser cabecera: magic `GGUF` + version + tensor_count + metadata_kv_count | `src/loader/gguf.zig` | 🔴 |
| C2 | Metadata KV: tipos (u8..u64, f32/f64, bool, string, array), arch → `ModelConfig` | `src/loader/gguf.zig` | 🔴 |
| C3 | Tensor info: name, shape, ggml_type, offset | `src/loader/gguf.zig` | 🔴 |
| C4 | Lectura mmap-friendly: mapear archivo, leer tensores por offset | `src/loader/gguf.zig` | 🟡 |
| C5 | Soporte ggml dtypes: F16, F32, BF16, Q8_0, Q4_0 (+dequant) | `src/loader/gguf.zig` | 🟡 |
| C6 | Mapeo de nombres GGUF → campos del layer (`blk.{i}.attn_q.weight` etc.) | `src/loader/gguf.zig` | 🟡 |
| C7 | `ModelConfig` (hidden_size, layers, heads, kv_heads, intermediate, vocab, rope_theta) | `src/loader/model_config.zig` | 🔴 |
| C8 | Test: cargar GGUF real (tiny) y verificar shapes | `tests/test_gguf.zig` | 🟡 |

> **Estado Fase C**: C1 ✅, C2 ✅ (parser metadata KV con arrays anidados + strings), C3 ✅, C4 ✅ (`fromFileMmap` vía `std.Io.File.MemoryMap`, lectura lazy por páginas; `tensorData()` sin copiar el archivo), C5 ✅ (dequant F16/F32/BF16/Q8_0/Q4_0/Q4_K/Q6_K), C6 ✅ (`parseTensorName` → `TensorRole` + índice de capa, con aliases attn_output/feed_forward/mlp), C7 ✅ (`src/loader/model_config.zig`: architecture, context_length, embedding_length, block_count, feed_forward_length, head_count/kv, rms_eps, rope_dim/freq_base, vocab_size; fallbacks a `tokenizer.ggml.tokens` y head_dim), C8 ✅ (tests/test_gguf.zig: carga real vía mmap, verifica config + shapes; requiere `GGUF_MODEL_PATH`, salta si no está). Verificado contra Qwen2.5-7B q4_k_m real (4.4GB): arch=qwen2, 339 tensores, embedding=3584, layers=28, kv_heads=4, ffn=18944, vocab=152064. Pendiente: Fase D (tokenizer desde GGUF).

## Fase D — Tokenizer desde GGUF (Importante)

| # | Tarea | Archivo | Prioridad |
|---|-------|---------|-----------|
| D1 | Leer tokenizer embebido en GGUF (tokenizer.ggml.model, tokens, merges, special) | `src/loader/gguf.zig` | 🔴 |
| D2 | Alimentar `BPETokenizer` con vocab/merges de GGUF (GPT-2 / LLaMA) | `src/tokenizer/bpe.zig` | 🔴 |
| D3 | Pre-tokenización regex GPT-2 real | `src/tokenizer/bpe.zig` | 🟡 |
| D4 | Special tokens (BOS/EOS/UNK/PAD) desde metadata GGUF | `src/tokenizer/bpe.zig` | 🟡 |
| D5 | Decode correcto (bytes → utf8, `<0x..>` fallback) | `src/tokenizer/bpe.zig` | 🟡 |

> **Estado Fase D**: D1 ✅ (`src/loader/gguf_tokenizer.zig`: modelo, pre, tokens, merges, token_types, special ids, add_bos/eos — vistas prestadas a datos mmap), D2 ✅ (`BPETokenizer.fromTokenizer` con `model` field), D3 ✅ (pre-tokenización Unicode Qwen3.5 portada de `llama.cpp/src/unicode.cpp` + `bytes_to_unicode`, módulos `unicode.zig`/`unicode_data.zig`), D4 ✅ (bos/eos/unk/pad desde metadata), D5 ✅ (decode invierte bytes_to_unicode → texto legible). Verificado contra llama.cpp en Qwen3.5-0.8B: "Hola, qué tal" → `[65717, 11, 40883, 7953]` idéntico. Pendiente: pre-tokenización de otros `pre` types (qwen2, llama3, etc.).

## Fase E — Pipeline end-to-end (Importante)

| # | Tarea | Archivo | Prioridad |
|---|-------|---------|-----------|
| E1 | `main.zig`: cargar GGUF → ModelConfig → init layers → prefill → generate → decode | `src/main.zig` | 🔴 |
| E2 | Embedding + lm_head desde tensores GGUF | `src/transformer/pipeline.zig` | 🔴 |
| E3 | RoPE base (rope_theta) configurable desde config | `src/transformer/rope.zig` | 🟡 |
| E4 | CLI: `--model model.gguf --prompt "..." -n 128` | `src/main.zig` | 🟡 |
| E5 | Métricas: tok/s, ms/token, memoria KV | `src/main.zig` | 🟢 |
| E6 | Modelos objetivo: Qwen2.5-1.5B Q8_0, TinyLlama-1.1B, Llama-3.2-1B | — | 🟢 |
> **Estado Fase E**: E0 ✅ (dequant Q4_K/Q6_K añadidos a `gguf.zig` con `getScaleMinK4` para escalas de 6 bits, tests unitarios; verificado contra referencia ggml C). E1 ✅ (pipeline end-to-end en `main.zig`: `runInference` carga GGUF → ModelConfig → init layers → prefill → generate → decode; el path híbrido Qwen3.5 está validado contra llama.cpp — generación correcta). E2 ✅ (embedding + lm_head desde tensores GGUF via `embedding.zig`/`loadEmbedding`/`loadLmHead`). E3 ✅ (RoPE configurable desde `ModelConfig` en ambos paths: híbrido vía `HybridLayerParams.fromModelConfig` [rope_freq_base/rope_sections/rope_dimension_count], no-híbrido vía `TransformerLayer.rope_freq_base = cfg.rope_freq_base`). E4 ✅ (CLI `--model/--prompt/-n/--backend/--seed/--temperature/--top-k/--top-p/--repetition-penalty`). E5 ✅ (métricas tok/s en `runHybridInference`: prefill ms + generación tok/s). Pendiente: E6 (modelos objetivo), path no-híbrido `TransformerLayer` legacy (solo F16, sin validar), CPU offload / PagedAttention.

## Fase F — PagedAttention (post-integrado, verificar)

| # | Tarea | Archivo | Prioridad | Estado |
|---|-------|---------|-----------|--------|
| F1 | Verificar `loadF16` byte-order en decode CPU ref | `src/paged_attention/attention.zig` | 🟡 | ✅ (little-endian correcto: `data[off]` LSB + `data[off+1]<<8`; coincide con el layout f16 GPU, y decode/prefill GPU vs CPU pasan con max_diff ≈ 1e-4) |
| F2 | Conectar scheduler + paged kv al pipeline (opcional) | — | 🟢 | ✅ (scheduler + `PagedKVCache` integrados; suite completa verde con `GGUF_MODEL_PATH`) |

## Fase G — QuantWeight + SSM (Gated DeltaNet) ✅

| # | Tarea | Archivo | Prioridad | Estado |
|---|-------|---------|-----------|--------|
| G1 | Dequant IQ3_S/IQ4_XS (byte-exact vs C, test permanente) | `src/loader/gguf.zig` | 🔴 | ✅ |
| G2 | `QuantWeight {info, bytes}` + `dequantToF16/F32` por bloques | `src/loader/quant_weight.zig` | 🔴 | ✅ |
| G3 | `SsmLayer` (Gated DeltaNet) sobre `QuantWeight`, scratch f16 persistente | `src/transformer/ssm.zig` | 🔴 | ✅ |
| G4 | Corregir recurrencia vs kernel FLA: decay `exp(-exp(A_log)·softplus)` y pairing `hk=hv//(n_v/n_k)` | `src/transformer/ssm.zig` | 🔴 | ✅ |
| G5 | Tests: hand-computed, persistencia, brute-force, recurrencia vs FLA naive | `src/transformer/ssm.zig` | 🔴 | ✅ |
> **Estado Fase G**: `zig build test` ✅ en verde (incluye SSM f32 + brute-force
> vs referencia, tests del tokenizer Unicode y dequant Q4_0 split-layout).
> Detalles en `docs/qwen35-hybrid-deltanet.md`.

## Fase H — Bloque híbrido Qwen3.5 (atención + rutado)
 
| # | Tarea | Archivo | Prioridad | Estado |
|---|-------|---------|-----------|--------|
| H1 | Capa de atención completa: `attn_q` fused Q+G, `attn_q_norm`/`attn_k_norm`, GQA 16→4, KV cache | `src/transformer/layer.zig`, `hybrid_attn.zig` | 🔴 | ✅ (`hybrid_attn.zig`/`hybrid_layer.zig` ya compilan y se testean) |
| H2 | IMROPE: NEOX, rotar 64 de 256 dims, secciones [11,11,10,0] | `src/transformer/rope.zig` | 🔴 | ✅ |
| H3 | Rutado híbrido SSM vs atención por `isFullAttentionLayer` | `src/transformer/layer.zig`, `pipeline.zig` | 🔴 | ✅ (`ModelConfig.isFullAttentionLayer` + `hybrid_layer.zig`) |
| H4 | Rework `gguf_model.zig` a `QuantWeight` + `dequantTensor` | `src/loader/gguf_model.zig` | 🔴 | ✅ |
| H5 | CLI de inferencia + validación modelo real | `src/main.zig`, `tests/test_gguf.zig` | 🟡 | ✅ (validado contra llama.cpp con Qwen3.5-0.8B-Q4_0.gguf: Pearson 0.9989 en logits del primer token, top1 idéntico, generación "¡Hola! ¿Cómo estás?" correcta) |
| H6 | Commit final + docs | — | 🟢 | ✅ (commit e1898a4) |
| H7 | Corregir dequant Q4_0/Q4_1: layout "split" de nibbles de ggml | `src/loader/gguf.zig` | 🔴 | ✅ (fue la causa raíz de logits descorrelacionados; test de regresión añadido) |
| H8 | Backend GPU (cuBLAS Sgemm + contexto) | `src/cuda/` | 🟢 | ✅ (commit f1ad33b: `cuDevicePrimaryCtxRetain` + símbolos `_v2`, `cublasSgemm_v2`, layout col-major → `colMajorToRowMajor`; GPU vs CPU Pearson 1.0) |
 
## Fase I — Bloque híbrido LFM2.5 (ShortConv + Atención) ✅
 
| # | Tarea | Archivo | Prioridad | Estado |
|---|-------|---------|-----------|--------|
| I1 | `ModelConfig`: add `lfm2` arch, parse `head_count_kv` array (30 int32), per-layer attn flags, `shortconv_l_cache` | `src/loader/model_config.zig` | 🔴 | ✅ |
| I2 | `MetaValue`: add `asI32()` for int32 arrays | `src/loader/gguf.zig` | 🔴 | ✅ |
| I3 | `ShortConvLayer`: depthwise conv1d + silu → in_proj (3x) → split(gate,up,value) → silu(gate)*up+value → out_proj | `src/transformer/short_conv.zig` (new) | 🔴 | ✅ CPU + GPU stub |
| I4 | `HybridAttnParams`: `no_gate` (separate Q), `use_mrope` (standard RoPE) | `src/transformer/hybrid_attn.zig` | 🔴 | ✅ |
| I5 | `HybridLayer`: LFM2 dispatch (ShortConv vs Attention), optional `attn_post_norm` (uses `ffn_norm`), GPU path for ShortConv | `src/transformer/hybrid_layer.zig` | 🔴 | ✅ |
| I6 | `rope.zig`: generic `applyRoPE(comptime T, ...)` for f32 | `src/transformer/rope.zig` | 🔴 | ✅ |
| I7 | `build.zig`: wire `short_conv` module | `build.zig` | 🔴 | ✅ |
| I8 | Validate: loads LFM2.5-2.6B-Q4_K_M.gguf (CPU + GPU OOM on 7.7GB VRAM) | — | 🟡 | ✅ |

---

## Notas de migración rápida 0.16

- `std.ArrayList(T)`: `var l: std.ArrayList(T) = .empty;` / `l.append(allocator, x)` / `l.deinit(allocator)`
- stdout: `const f = std.Io.File.stdout(); var w = f.writer(io, &buf); w.print(...)`
- Timer: `src/utils/time.zig` (`Timer.start()/read()` sobre `posix.clock_gettime(.MONOTONIC)`), sin necesidad de `io`
- Thread.Pool → `std.Thread.spawn`/`join` por llamada; `MatmulEngine.thread_pool` → `num_threads`
- `GeneralPurposeAllocator` → `std.heap.DebugAllocator`; en main ya no hace falta (usa `init.gpa`)
- `@Type(.{ .int = ... })` → `@Int(...)`
- `main(init: std.process.Init)` da `init.io`, `init.gpa`, `init.arena`

*Documento generado el 2026-08-04. Actualizar conforme se completan tareas.*
