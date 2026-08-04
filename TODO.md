# Zig AI Engine — TODO

> Fecha: 2026-08-04
> Stack objetivo: **Zig 0.16.0** (único toolchain soportado, se abandona 0.14)
> Formato de modelo: GGUF (primario) + safetensors (secundario)

---

## Decisión de toolchain

El código NO es compatible entre 0.14 / 0.15 / 0.16 (writergate + std.Io + ArrayList
unmanaged + build system). Se migra el monorepo a **Zig 0.16.0** y se eliminan los
paths solo-0.14. Instalado en `~/.local/bin/zig-0.16`.

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
| A1 | Reemplazar `.root_source_file` por `root_module` en addExecutable/addTest/addModule | `build.zig` | 🔴 | ✅ |
| A2 | Ajustar `b.addModule` / imports a la API 0.16 (`b.path`, lazy paths) | `build.zig` | 🔴 | ✅ |
| A3 | nvcc/cubin/ptx pasos siguen igual (addSystemCommand) | `build.zig` | 🟡 | ✅ |
| A4 | `zig build test` verde en 0.16 | — | 🔴 | ✅ |

## Fase B — Migración stdlib a 0.16 (Bloqueante)

| # | Tarea | Archivo | Prioridad | Estado |
|---|-------|---------|-----------|--------|
| B1 | `std.ArrayList(T)` → unmanaged: `append(a, x)`, `deinit(a)` | todos | 🔴 | ✅ |
| B2 | I/O: pasar `io` por call stack (`file.read(io, buf)`) | loader, safetensors, main | 🔴 | ⬜ (main/bench sí; loader pendiente) |
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

> **Estado Fase C**: C1 ✅, C2 ✅ (parser metadata KV con arrays anidados + strings), C3 ✅, C4 ✅ (`fromFileMmap` vía `std.Io.File.MemoryMap`, lectura lazy por páginas; `tensorData()` sin copiar el archivo), C5 ✅ parcial (dequant F16/F32/BF16/Q8_0/Q4_0), C6 ✅ (`parseTensorName` → `TensorRole` + índice de capa, con aliases attn_output/feed_forward/mlp), C7 ✅ (`src/loader/model_config.zig`: architecture, context_length, embedding_length, block_count, feed_forward_length, head_count/kv, rms_eps, rope_dim/freq_base, vocab_size; fallbacks a `tokenizer.ggml.tokens` y head_dim), C8 ✅ (tests/test_gguf.zig: carga real vía mmap, verifica config + shapes; requiere `GGUF_MODEL_PATH`, salta si no está). Verificado contra Qwen2.5-7B q4_k_m real (4.4GB): arch=qwen2, 339 tensores, embedding=3584, layers=28, kv_heads=4, ffn=18944, vocab=152064. Pendiente: Fase D (tokenizer desde GGUF).

## Fase D — Tokenizer desde GGUF (Importante)

| # | Tarea | Archivo | Prioridad |
|---|-------|---------|-----------|
| D1 | Leer tokenizer embebido en GGUF (tokenizer.ggml.model, tokens, merges, special) | `src/loader/gguf.zig` | 🔴 |
| D2 | Alimentar `BPETokenizer` con vocab/merges de GGUF (GPT-2 / LLaMA) | `src/tokenizer/bpe.zig` | 🔴 |
| D3 | Pre-tokenización regex GPT-2 real | `src/tokenizer/bpe.zig` | 🟡 |
| D4 | Special tokens (BOS/EOS/UNK/PAD) desde metadata GGUF | `src/tokenizer/bpe.zig` | 🟡 |
| D5 | Decode correcto (bytes → utf8, `<0x..>` fallback) | `src/tokenizer/bpe.zig` | 🟡 |

## Fase E — Pipeline end-to-end (Importante)

| # | Tarea | Archivo | Prioridad |
|---|-------|---------|-----------|
| E1 | `main.zig`: cargar GGUF → ModelConfig → init layers → prefill → generate → decode | `src/main.zig` | 🔴 |
| E2 | Embedding + lm_head desde tensores GGUF | `src/transformer/pipeline.zig` | 🔴 |
| E3 | RoPE base (rope_theta) configurable desde config | `src/transformer/rope.zig` | 🟡 |
| E4 | CLI: `--model model.gguf --prompt "..." -n 128` | `src/main.zig` | 🟡 |
| E5 | Métricas: tok/s, ms/token, memoria KV | `src/main.zig` | 🟢 |
| E6 | Modelos objetivo: Qwen2.5-1.5B Q8_0, TinyLlama-1.1B, Llama-3.2-1B | — | 🟢 |

## Fase F — PagedAttention (post-integrado, verificar)

| # | Tarea | Archivo | Prioridad |
|---|-------|---------|-----------|
| F1 | Verificar `loadF16` byte-order en decode CPU ref | `src/paged_attention/attention.zig` | 🟡 |
| F2 | Conectar scheduler + paged kv al pipeline (opcional) | — | 🟢 |

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
