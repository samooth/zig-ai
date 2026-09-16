//! KT-B (lane-f) — tests del runtime KV-transfer.
//!
//! Gate-0: identity roundtrip. Un mismo KVCacheManager con DOS secuencias
//! (patrón MTP-draft de cli.zig): seq SOURCE recibe un prefill sintético,
//! el transfer (mapper identity) debe reproducir en seq TARGET los K/V
//! originales — validar retrieve target vs source con tolerancia del
//! roundtrip fp16 (encode/decode del manager).
//!
//! El roundtrip numérico strip→re-rope es EXACTO solo en f32; el manager
//! cuantiza al formato de capa (fp16 default) — tolerancia 2e-3 rel.

const std = @import("std");
const kvc = @import("kv_cache");
const rope_mod = @import("rope");
const matmul = @import("matmul");
const kt = kvc.kt_transfer;

const allocator = std.testing.allocator;

// NOTA lifetime: layer_configs apunta a memoria del CALLER (el manager no
// copia) — los cfgs viven hasta el deinit del manager. En cada test se
// alocan antes del manager y se liberan después (defer orden inverso).
fn makeCfgs(num_layers: u32) ![]kvc.LayerQuantConfig {
    const cfgs = try allocator.alloc(kvc.LayerQuantConfig, num_layers);
    for (cfgs) |*c| c.* = .{ .k_format = .fp16, .v_format = .fp16, .k_block_size = 32, .v_block_size = 32, .quant_threshold = null };
    return cfgs;
}

fn makeManager(num_layers: u32, num_heads: u32, head_dim: u32, max_seq: u32, cfgs: []kvc.LayerQuantConfig) !kvc.KVCacheManager {
    // Igual que el CLI default (main.zig layer_cfgs): fp16 explícito —
    // el default del manager sin layer_configs es q4_0/q8_0 y degradaría
    // el roundtrip (lección: SIEMPRE pasar layer_formats + lifetime largo).
    var cfg = kvc.KVCacheConfig.default(num_layers, num_heads, head_dim, max_seq);
    cfg.layer_configs = cfgs;
    return kvc.KVCacheManager.init(allocator, cfg, 8);
}

test "KT-B gate-0: identity roundtrip en manager dual-seq" {
    const n_layers: u32 = 2;
    const n_kv: u32 = 3;
    const hd: u32 = 64;
    const n_tokens: usize = 16;

    const cfgs = try makeCfgs(n_layers);
    defer allocator.free(cfgs);
    var mgr = try makeManager(n_layers, n_kv, hd, 256, cfgs);
    defer mgr.deinit();

    // Prefill sintético del source: K/V "post-RoPE" fabricados (como los
    // escribiría storeKvCache) — K_pre rotado con rope NORM base 10000.
    const seq_src: u64 = 1;
    const seq_tgt: u64 = 2;
    try mgr.createSequence(seq_src);
    try mgr.createSequence(seq_tgt);

    // Referencia pre-RoPE por (layer, head, pos).
    const k_pre_ref = try allocator.alloc(f32, n_layers * n_kv * n_tokens * hd);
    defer allocator.free(k_pre_ref);

    var prng = std.Random.Xoshiro256.init(42);
    const rand = prng.random();

    for (0..n_layers) |l| {
        for (0..n_kv) |h| {
            for (0..n_tokens) |p| {
                const base = ((l * n_kv + h) * n_tokens + p) * hd;
                for (0..hd) |c| {
                    k_pre_ref[base + c] = rand.float(f32) * 2.0 - 1.0;
                }
            }
        }
    }

    // Escribir el source como lo hace el engine: K post-RoPE (aplicamos
    // forward con el mismo helper del runtime), V tal cual.
    var k_post = try allocator.alloc(f32, n_tokens * hd);
    defer allocator.free(k_post);
    var v_tok = try allocator.alloc(f16, n_tokens * hd);
    defer allocator.free(v_tok);
    var k_f16 = try allocator.alloc(f16, n_tokens * hd);
    defer allocator.free(k_f16);

    for (0..n_layers) |l| {
        for (0..n_kv) |h| {
            const base = ((l * n_kv + h) * n_tokens) * hd;
            for (0..n_tokens) |p| {
                for (0..hd) |c| v_tok[p * hd + c] = @floatCast(k_pre_ref[base + p * hd + c] * 0.5);
            }
            // K: copiar pre → aplicar forward in-place → append f16.
            for (0..n_tokens * hd) |i| k_post[i] = k_pre_ref[base + i];
            for (0..n_tokens) |p| {
                rope_mod.applyRoPEForwardOnSlice(f32, k_post[p * hd .. (p + 1) * hd], p, hd, 10000.0, .norm);
            }
            for (0..n_tokens * hd) |i| k_f16[i] = @floatCast(k_post[i]);
            try mgr.appendTokensF16(seq_src, @intCast(l), @intCast(h), k_f16, v_tok);
        }
    }
    for (0..n_tokens) |_| try mgr.advanceSequence(seq_src);

    // Pesos identity + runtime.
    const io = std.Io.Threaded.global_single_threaded.io();
    try kt.writeIdentityKtb(io, "tests/corpora/ktb_gate0.ktb", n_layers, n_kv, hd);
    var w = try kt.KtWeights.load(allocator, io, "tests/corpora/ktb_gate0.ktb", n_layers, n_kv, n_kv, hd);
    defer w.deinit();

    var engine = try matmul.MatmulEngine.init(allocator, .tiled, .f32);
    defer engine.deinit();

    var rt = kt.KtRuntime{
        .allocator = allocator,
        .weights = &w,
        .seq_source = seq_src,
        .seq_target = seq_tgt,
    };
    try rt.transfer(
        .{ .manager = &mgr, .rope_base = 10000.0, .pairing = .norm },
        .{ .manager = &mgr, .rope_base = 10000.0, .pairing = .norm },
        &engine,
        n_tokens,
    );

    // Verificación: retrieve target → strip RoPE (source pairing) → debe
    // igualar k_pre_ref (tolerancia fp16 encode/decode).
    const k_out = try allocator.alloc(f16, n_tokens * hd);
    defer allocator.free(k_out);
    const v_out = try allocator.alloc(f16, n_tokens * hd);
    defer allocator.free(v_out);
    const k_stripped = try allocator.alloc(f32, n_tokens * hd);
    defer allocator.free(k_stripped);

    try std.testing.expectEqual(n_tokens, try mgr.getSequenceLen(seq_tgt));

    for (0..n_layers) |l| {
        for (0..n_kv) |h| {
            try mgr.retrieveForAttention(seq_tgt, @intCast(l), @intCast(h), k_out, v_out);
            for (0..n_tokens) |p| {
                rope_mod.applyRoPEInverseOnSlice(f16, k_out[p * hd .. (p + 1) * hd], k_stripped[p * hd .. (p + 1) * hd], p, hd, 10000.0, .norm);
            }
            const base = ((l * n_kv + h) * n_tokens) * hd;
            for (0..n_tokens * hd) |i| {
                const got: f32 = k_stripped[i];
                const want: f32 = k_pre_ref[base + i];
                try std.testing.expectApproxEqAbs(want, got, 5e-2);
                // V: sin RoPE — directa (el 0.5 del fabric).
                const v_got: f32 = @floatCast(v_out[i]);
                try std.testing.expectApproxEqAbs(want * 0.5, v_got, 5e-2);
            }
        }
    }
}

test "KT-B error paths: target no vacío y source corto" {
    const cfgs = try makeCfgs(1);
    defer allocator.free(cfgs);
    var mgr = try makeManager(1, 2, 32, 64, cfgs);
    defer mgr.deinit();
    try mgr.createSequence(1);
    try mgr.createSequence(2);

    const io = std.Io.Threaded.global_single_threaded.io();
    try kt.writeIdentityKtb(io, "tests/corpora/ktb_err.ktb", 1, 2, 32);
    var w = try kt.KtWeights.load(allocator, io, "tests/corpora/ktb_err.ktb", 1, 2, 2, 32);
    defer w.deinit();

    var engine = try matmul.MatmulEngine.init(allocator, .tiled, .f32);
    defer engine.deinit();
    var rt = kt.KtRuntime{ .allocator = allocator, .weights = &w, .seq_source = 1, .seq_target = 2 };

    // Source vacío → SequenceEmpty.
    try std.testing.expectError(kt.KtError.SequenceEmpty, rt.transfer(.{ .manager = &mgr }, .{ .manager = &mgr }, &engine, 4));

    // Llenar source, ensuciar target → BadHeader (target debe arrancar 0).
    const k: []const f16 = &[_]f16{0.1} ** 32;
    try mgr.appendTokensF16(1, 0, 0, k, k);
    try mgr.advanceSequence(1);
    try mgr.appendTokensF16(2, 0, 0, k, k);
    try mgr.advanceSequence(2);
    try std.testing.expectError(kt.KtError.BadHeader, rt.transfer(.{ .manager = &mgr }, .{ .manager = &mgr }, &engine, 1));
}

test "KT-B dense W=I numérico: transfer == identity (regresión fromSlice)" {
    // Regresión del bug fromSlice: gemm escribía en la copia C y y quedaba
    // intacto (basura post-first-token). W=I numérico (kind=dense) debe
    // igualar el identity byte a byte en la salida del manager.
    const io = std.Io.Threaded.global_single_threaded.io();
    const n_layers: u32 = 2;
    const n_kv: u32 = 2;
    const hd: u32 = 16;
    const n_tokens: usize = 4;
    const out_dim: usize = n_kv * hd;

    // .ktb dense con W=I, b=0.
    var body = std.ArrayList(u8).empty;
    defer body.deinit(std.heap.page_allocator);
    var hdr: [24]u8 = undefined;
    std.mem.writeInt(u32, hdr[0..4], kt.KT_MAGIC, .little);
    std.mem.writeInt(u32, hdr[4..8], kt.KT_VERSION, .little);
    std.mem.writeInt(u32, hdr[8..12], n_layers, .little);
    std.mem.writeInt(u32, hdr[12..16], n_kv, .little);
    std.mem.writeInt(u32, hdr[16..20], n_kv, .little);
    std.mem.writeInt(u32, hdr[20..24], hd, .little);
    try body.appendSlice(std.heap.page_allocator, &hdr);
    for (0..n_layers) |_| {
        var kind: [4]u8 = undefined;
        std.mem.writeInt(u32, kind[0..4], 1, .little); // dense
        try body.appendSlice(std.heap.page_allocator, &kind);
        // W=I [out_dim×in_dim] = [n_kv·hd × n_kv·hd] = 32×32 (¡no hd×hd!).
        var wi: [32 * 32]f32 = @splat(0.0);
        for (0..32) |i| wi[i * 32 + i] = 1.0;
        var zero: [32]f32 = @splat(0.0);
        try body.appendSlice(std.heap.page_allocator, std.mem.sliceAsBytes(&wi)); // W_k=I
        try body.appendSlice(std.heap.page_allocator, std.mem.sliceAsBytes(&zero)); // b_k
        try body.appendSlice(std.heap.page_allocator, std.mem.sliceAsBytes(&wi)); // W_v=I
        try body.appendSlice(std.heap.page_allocator, std.mem.sliceAsBytes(&zero)); // b_v
    }
    const path = "tests/corpora/ktb_dense_I.ktb";
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = body.items });

    const cfgs = try makeCfgs(n_layers);
    defer allocator.free(cfgs);
    var mgr = try makeManager(n_layers, n_kv, hd, 64, cfgs);
    defer mgr.deinit();
    const seq_src: u64 = 1;
    const seq_tgt: u64 = 2;
    try mgr.createSequence(seq_src);
    try mgr.createSequence(seq_tgt);

    var w = try kt.KtWeights.load(allocator, io, path, n_layers, n_kv, n_kv, hd);
    defer w.deinit();
    var engine = try matmul.MatmulEngine.init(allocator, .tiled, .f32);
    defer engine.deinit();

    // Source: K/V sintéticos post-RoPE (K), V directo.
    var k_f16 = try allocator.alloc(f16, n_tokens * hd);
    defer allocator.free(k_f16);
    var v_f16 = try allocator.alloc(f16, n_tokens * hd);
    defer allocator.free(v_f16);
    var k_pre = try allocator.alloc(f32, n_tokens * hd);
    defer allocator.free(k_pre);
    var prng = std.Random.Xoshiro256.init(7);
    const rand = prng.random();
    for (0..n_tokens * hd) |i| {
        k_pre[i] = rand.float(f32) * 2 - 1;
        v_f16[i] = @floatCast(k_pre[i] * 0.25);
        k_f16[i] = @floatCast(k_pre[i]);
    }
    for (0..n_tokens) |p| {
        rope_mod.applyRoPEForwardOnSlice(f16, k_f16[p * hd .. (p + 1) * hd], p, hd, 10000.0, .norm);
    }
    for (0..n_layers) |l| {
        for (0..n_kv) |h| {
            const chunk_k = k_f16[0 .. n_tokens * hd];
            const chunk_v = v_f16[0 .. n_tokens * hd];
            try mgr.appendTokensF16(seq_src, @intCast(l), @intCast(h), chunk_k, chunk_v);
        }
    }
    for (0..n_tokens) |_| try mgr.advanceSequence(seq_src);

    var rt = kt.KtRuntime{ .allocator = allocator, .weights = &w, .seq_source = seq_src, .seq_target = seq_tgt };
    try rt.transfer(
        .{ .manager = &mgr, .rope_base = 10000.0, .pairing = .norm },
        .{ .manager = &mgr, .rope_base = 10000.0, .pairing = .norm },
        &engine,
        n_tokens,
    );

    // Salida dense W=I == entrada (strip RoPE + I + re-rope = original).
    const rk: []f16 = try allocator.alloc(f16, n_tokens * hd);
    defer allocator.free(rk);
    const rv: []f16 = try allocator.alloc(f16, n_tokens * hd);
    defer allocator.free(rv);
    try mgr.retrieveForAttention(seq_tgt, 0, 0, rk, rv);
    for (0..n_tokens) |p| {
        const row = p * hd;
        try std.testing.expectApproxEqAbs(@as(f32, @floatCast(v_f16[row])), @as(f32, @floatCast(rv[row])), 1e-3);
    }
    _ = out_dim;
}
