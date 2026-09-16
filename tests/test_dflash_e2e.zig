//! 5.2 (lane-b1) — Test de integración del DflashDraftModel con el sidecar
//! REAL (qwen35-9b-dflash-Q8_0) + target Qwen3.5-9B. Sin GPU (camino CPU):
//!
//! 1. dual-load del sidecar (hereda tok_embd/lm_head del target)
//! 2. init del DflashDraftModel (6 capas blk.* como HybridLayers no-causales)
//! 3. encoderForward: taps sintéticas → fused [n_tok, 4096] (sanidad: finito)
//! 4. kvInject: fused → KV del draft crece n_tok posiciones
//! 5. denoiseDraft: [anchor, MASK×15] → 15×vocab logits finitos
//!
//! Gates: finitud (sin NaN/inf), crecimiento del KV exacto, shapes
//! correctos. La CORRECTITUD numérica vs llama.cpp requiere el E2E spec
//! (gate posterior con wiring cli.zig).
//!
//!   ZIG_AI_DFLASH_SIDECAR=<dflash.gguf> ZIG_AI_DFLASH_TARGET=<target.gguf> \
//!     zig build test-dflash-e2e

const std = @import("std");
const gguf = @import("gguf");
const gguf_model = @import("gguf_model");
const paged = @import("paged_attention");
const hybrid_layer = @import("hybrid_layer");
const dflash = @import("speculative").dflash_draft;

test "DflashDraftModel integración sidecar real: kvInject + denoise CPU" {
    const allocator = std.heap.c_allocator;
    const io = std.Io.Threaded.global_single_threaded.io();

    const sidecar_path = std.c.getenv("ZIG_AI_DFLASH_SIDECAR") orelse return error.SkipZigTest;
    const target_path = std.c.getenv("ZIG_AI_DFLASH_TARGET") orelse return error.SkipZigTest;

    // Target (Qwen3.5-9B — tok_embd/lm_head/output_norm heredados)
    var target = try gguf_model.GgufModel.load(io, allocator, std.mem.span(target_path));
    defer target.deinit();

    // Dual-load del sidecar
    var sidecar = try gguf_model.GgufModel.loadSidecarDraft(&target, io, allocator, std.mem.span(sidecar_path));
    defer sidecar.deinit();

    // Validaciones del dual-load contra el sidecar real
    try std.testing.expectEqual(@as(usize, 16), sidecar.block_size);
    try std.testing.expectEqual(@as(usize, 6), sidecar.n_layers);
    try std.testing.expectEqual(@as(usize, 8), sidecar.target_layers.len);
    try std.testing.expectEqual(@as(i32, 2), sidecar.target_layers[0]);
    try std.testing.expectEqual(@as(i32, 30), sidecar.target_layers[7]);

    // KV paginado del draft (pool propio, secuencias separadas)
    var draft_kv = try paged.PagedKVCache.init(allocator, .{
        .block_size = 16,
        .num_blocks = 512,
        .head_dim = sidecar.model.config.head_dim,
        .num_kv_heads = sidecar.model.config.head_count_kv,
        .num_q_heads = sidecar.model.config.head_count,
        .dtype = .f16,
    });
    defer draft_kv.deinit();

    // Draft-model: 6 capas no-causales + encoder fc/norm (backend parallel = CPU multi-thread)
    var model = try dflash.DflashDraftModel.init(allocator, &sidecar, &draft_kv, .parallel, 0.1);
    defer model.deinit();
    try std.testing.expectEqual(@as(usize, 6), model.layers.len);
    try std.testing.expectEqual(@as(usize, 16), model.block_size);
    // Geometría real del sidecar
    try std.testing.expectEqual(@as(usize, 4096), model.cfg.embedding_length);
    try std.testing.expectEqual(@as(usize, 32), model.cfg.head_count);
    try std.testing.expectEqual(@as(usize, 8), model.cfg.head_count_kv);
    try std.testing.expectEqual(@as(usize, 128), model.cfg.head_dim);

    // ── 1. Encoder: taps sintéticas [3 tok, 8×4096] → fused [3, 4096] ──
    const n_extract = model.target_layers.len;
    const n_embd = model.cfg.embedding_length;
    const n_taps = 3;
    const taps = try allocator.alloc(f32, n_taps * n_extract * n_embd);
    defer allocator.free(taps);
    var prng = std.Random.DefaultPrng.init(42);
    const rnd = prng.random();
    for (taps) |*t| t.* = rnd.float(f32) * 2.0 - 1.0;
    const fused = try allocator.alloc(f32, n_taps * n_embd);
    defer allocator.free(fused);
    try model.encoderForward(undefined, taps, n_taps, fused);
    for (fused) |v| try std.testing.expect(std.math.isFinite(v));

    // ── 2. kvInject: el KV del draft crece exactamente n_taps ──
    const kv_before = model.kvLen();
    try model.kvInject(fused, n_taps, kv_before);
    try std.testing.expectEqual(kv_before + n_taps, model.kvLen());

    // ── 3. denoise: [anchor, MASK×15] → 15 filas × vocab logits ──
    const vocab = model.lm_head.shape[0];
    const bs = model.block_size;
    const logits = try allocator.alloc(f32, (bs - 1) * vocab);
    defer allocator.free(logits);
    const n_rows = try model.denoiseDraft(1234, logits);
    try std.testing.expectEqual(bs - 1, n_rows);
    var n_finite: usize = 0;
    for (logits) |v| {
        if (std.math.isFinite(v)) n_finite += 1;
    }
    // Todos finitos (el forward con pesos reales q8_0 no puede dar NaN si
    // la maquinaria de buffers/RoPE/softmax es correcta)
    try std.testing.expectEqual(logits.len, n_finite);

    // ── 4. Segundo kvInject en start_pos avanzado + rollback ──
    try model.kvInject(fused, n_taps, model.kvLen());
    const after2 = model.kvLen();
    try model.rollbackTo(after2 - n_taps);
    try std.testing.expectEqual(after2 - n_taps, model.kvLen());
}
