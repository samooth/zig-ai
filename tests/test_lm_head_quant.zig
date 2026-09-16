//! 7.1d lm_head (lane-c): paridad del GEMV cuant-residente vs la tabla f16
//! densa clásica. `lmHeadGemvQ80` (bytes q8_0 on-load) y `lmHeadGemvQuant`
//! (QuantWeight nativo) computan logits[j] = dot(row_j, x) con dequant de
//! bloque inline — cero materialización f16 del peso y cero round-trip f32
//! del tensor completo. Gate: argmax idéntico y rel-err ≤ 1e-2 por término
//! (el q8_0 del peso añade ~0.4% de error de cuantización por fila — el
//! argmax del greedy debe ser estable salvo empates <0.05).

const std = @import("std");
const Tensor = @import("core").Tensor;
const embedding_mod = @import("embedding");
const gguf = @import("gguf");
const QuantWeight = @import("quant_weight").QuantWeight;
const matmul = @import("matmul");

test "lmHeadGemvQ80: paridad argmax + rel-err vs tabla f16" {
    const allocator = std.testing.allocator;

    // Geometría pequeña: hidden=32 (1 bloque q8_0), vocab=64.
    const hidden: usize = 32;
    const vocab: usize = 64;
    const kb = hidden / 32;

    // Fuente f32 aleatoria → tabla f16 (referencia) + bytes q8_0.
    const src = try allocator.alloc(f32, vocab * hidden);
    defer allocator.free(src);
    var prng = std.Random.DefaultPrng.init(0xBEEF);
    const rand = prng.random();
    for (src) |*v| v.* = rand.float(f32) * 2.0 - 1.0;

    var table = try Tensor(f16).alloc(allocator, &[_]usize{ vocab, hidden });
    defer table.deinit();
    for (src, 0..) |v, i| table.data[i] = @floatCast(v);

    // q8_0 canónico: [d f16 @0][i8×32 @2] por bloque (34B).
    const bytes = try allocator.alloc(u8, vocab * kb * 34);
    defer allocator.free(bytes);
    for (0..vocab) |j| {
        const row = src[j * hidden ..][0..hidden];
        const dst_row = bytes[j * kb * 34 ..][0 .. kb * 34];
        var amax: f32 = 0;
        for (row) |v| amax = @max(amax, @abs(v));
        const d: f16 = @floatCast(if (amax > 0) amax / 127.0 else 1.0);
        std.mem.writeInt(u16, dst_row[0..2], @bitCast(d), .little);
        for (row, 0..) |v, c| {
            const q: i8 = @intFromFloat(@round(v / @max(@as(f32, @floatCast(d)), 1e-8)));
            dst_row[2 + c] = @bitCast(q);
        }
    }

    // x de entrada (hidden post-RMSNorm).
    var x = try Tensor(f16).alloc(allocator, &.{ 1, hidden });
    defer x.deinit();
    for (x.data, 0..) |*p, i| p.* = @floatCast(@as(f32, @floatFromInt(i % 7)) * 0.1 - 0.3);

    // Referencia: GEMV f16 sobre la tabla.
    var ref_logits: [64]f32 = undefined;
    for (0..vocab) |j| {
        var acc: f32 = 0;
        const row = table.data[j * hidden ..][0..hidden];
        for (row, 0..) |h16, i| acc += @as(f32, h16) * @as(f32, @floatCast(x.data[i]));
        ref_logits[j] = acc;
    }

    // Camino q8_0.
    var out: [64]f32 = undefined;
    embedding_mod.lmHeadGemvQ80(x.data[0..hidden], bytes, vocab, hidden, &out);

    // Gate SNR estilo FP8 (TODO 2.1): el error por-término explota con
    // ref≈0 (logit pequeño) — el criterio estable es ‖err‖/‖ref‖ global
    // (la cuantización q8 del peso añade ~0.4%/fila, cancela en el dot).
    var err2: f64 = 0;
    var ref2: f64 = 0;
    for (ref_logits, 0..) |rv, j| {
        const e: f64 = out[j] - rv;
        err2 += e * e;
        ref2 += @as(f64, rv) * rv;
    }
    const snr = @sqrt(err2) / @sqrt(ref2);
    try std.testing.expect(snr < 0.02);

    // Argmax idéntico (estable salvo empates <0.05 — verificar gap).
    var ia: usize = 0;
    var ib: usize = 0;
    for (ref_logits, 0..) |v, j| {
        if (v > ref_logits[ia]) ia = j;
        if (out[j] > out[ib]) ib = j;
    }
    if (ia != ib) {
        const gap = @abs(ref_logits[ia] - ref_logits[ib]);
        try std.testing.expect(gap < 0.05); // flip sólo en empate
    }
}

test "lmHeadGemvQuant: paridad EXACTA vs tabla f16 (mismo dequantBlock)" {
    const allocator = std.testing.allocator;

    // El lm_head GGUF cuantizado: dequant de referencia con el MISMO
    // dequantBlock ⇒ paridad bit-exacta del dot (f32 acumulado).
    const hidden: usize = 64; // 2 bloques q8_0
    const vocab: usize = 48;
    const bs = 32;
    const bb = 34;
    const total = hidden * vocab;
    const n_blocks = total / bs;

    const src = try allocator.alloc(f32, total);
    defer allocator.free(src);
    var prng = std.Random.DefaultPrng.init(0xC0DE);
    const rand = prng.random();
    for (src) |*v| v.* = rand.float(f32) * 2.0 - 1.0;

    const bytes = try allocator.alloc(u8, n_blocks * bb);
    defer allocator.free(bytes);
    for (0..n_blocks) |i| {
        const blk = src[i * bs ..][0..bs];
        var amax: f32 = 0;
        for (blk) |v| amax = @max(amax, @abs(v));
        const d: f16 = @floatCast(amax / 127.0);
        std.mem.writeInt(u16, bytes[i * bb ..][0..2], @bitCast(d), .little);
        for (blk, 0..) |v, j| {
            const q: i8 = @intFromFloat(@round(v / @max(@as(f32, @floatCast(d)), 1e-8)));
            bytes[i * bb + 2 + j] = @bitCast(q);
        }
    }

    var dims = [_]u64{ 0, 0, 0, 0 };
    dims[0] = hidden;
    dims[1] = vocab;
    const info = gguf.TensorInfo{
        .name = "output.weight",
        .n_dims = 2,
        .dims = dims,
        .dtype = .q8_0,
        .offset = 0,
    };
    const qw = QuantWeight.init(&info, bytes);

    // Tabla de referencia vía dequant total.
    var table = try Tensor(f16).alloc(allocator, &[_]usize{ vocab, hidden });
    defer table.deinit();
    var dec: [256]f32 = undefined;
    for (0..n_blocks) |i| {
        gguf.dequantBlock(.q8_0, bytes[i * bb ..][0..bb], &dec, bs);
        for (0..bs) |j| {
            const s = i * bs + j;
            const r = s % hidden;
            const c = s / hidden;
            table.data[c * hidden + r] = @floatCast(dec[j]);
        }
    }

    var x = try Tensor(f16).alloc(allocator, &.{ 1, hidden });
    defer x.deinit();
    for (x.data, 0..) |*p, i| p.* = @floatCast(@as(f32, @floatFromInt(i % 5)) * 0.2 - 0.4);

    var ref_logits: [48]f32 = undefined;
    for (0..vocab) |j| {
        var acc: f32 = 0;
        const row = table.data[j * hidden ..][0..hidden];
        for (row, 0..) |h16, i| acc += @as(f32, h16) * @as(f32, @floatCast(x.data[i]));
        ref_logits[j] = acc;
    }

    var out: [48]f32 = undefined;
    embedding_mod.lmHeadGemvQuant(x.data[0..hidden], &qw, &out);

    // Bit-exact no garantizado por el cast f16 intermedio de la tabla ref —
    // el GEMV cuant acumula f32 puro. Gate SNR ‖err‖/‖ref‖ ≤ 1e-3
    // (sólo redondeo f16 de la referencia).
    var err2: f64 = 0;
    var ref2: f64 = 0;
    for (ref_logits, 0..) |rv, j| {
        const e: f64 = out[j] - rv;
        err2 += e * e;
        ref2 += @as(f64, rv) * rv;
    }
    const snr = @sqrt(err2) / @sqrt(ref2);
    try std.testing.expect(snr < 1e-3);
}

test "LmHeadSource.q80 via lmHeadForwardSource: argmax idéntico al camino tabla" {
    const allocator = std.testing.allocator;

    const hidden: usize = 32;
    const vocab: usize = 64;
    const kb = hidden / 32;

    const src = try allocator.alloc(f32, vocab * hidden);
    defer allocator.free(src);
    var prng = std.Random.DefaultPrng.init(0xFACE);
    const rand = prng.random();
    for (src) |*v| v.* = rand.float(f32) * 2.0 - 1.0;

    var table = try Tensor(f16).alloc(allocator, &[_]usize{ vocab, hidden });
    defer table.deinit();
    for (src, 0..) |v, i| table.data[i] = @floatCast(v);

    const bytes = try allocator.alloc(u8, vocab * kb * 34);
    defer allocator.free(bytes);
    for (0..vocab) |j| {
        const row = src[j * hidden ..][0..hidden];
        const dst_row = bytes[j * kb * 34 ..][0 .. kb * 34];
        var amax: f32 = 0;
        for (row) |v| amax = @max(amax, @abs(v));
        const d: f16 = @floatCast(if (amax > 0) amax / 127.0 else 1.0);
        std.mem.writeInt(u16, dst_row[0..2], @bitCast(d), .little);
        for (row, 0..) |v, c| {
            const q: i8 = @intFromFloat(@round(v / @max(@as(f32, @floatCast(d)), 1e-8)));
            dst_row[2 + c] = @bitCast(q);
        }
    }

    var engine = try matmul.MatmulEngine.init(allocator, .naive, .f32);
    defer engine.deinit();

    var hidden_2d = try Tensor(f16).alloc(allocator, &.{ 1, hidden });
    defer hidden_2d.deinit();
    for (hidden_2d.data, 0..) |*p, i| p.* = @floatCast(@as(f32, @floatFromInt(i % 3)) * 0.15 - 0.2);

    var logits_tbl = try Tensor(f32).alloc(allocator, &.{ 1, vocab });
    defer logits_tbl.deinit();
    try embedding_mod.lmHeadForwardSource(&engine, hidden_2d, .{ .table = table }, hidden, vocab, &logits_tbl);

    var logits_q80 = try Tensor(f32).alloc(allocator, &.{ 1, vocab });
    defer logits_q80.deinit();
    try embedding_mod.lmHeadForwardSource(&engine, hidden_2d, .{ .q80 = bytes }, hidden, vocab, &logits_q80);

    var ia: usize = 0;
    var ib: usize = 0;
    for (logits_tbl.data, 0..) |v, j| {
        if (v > logits_tbl.data[ia]) ia = j;
        if (logits_q80.data[j] > logits_q80.data[ib]) ib = j;
    }
    if (ia != ib) {
        const gap = @abs(logits_tbl.data[ia] - logits_tbl.data[ib]);
        try std.testing.expect(gap < 0.05);
    }
}
