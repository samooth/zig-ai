//! 7.1d-embedding (lane-b1): paridad embeddingLookupQuant vs tabla f16
//! completa (loadEmbedding). El camino cuant-residente dequantiza SOLO las
//! filas de los tokens pedidos desde los bytes mmap del GGUF; el gate es
//! bit-exactitud contra materializar la tabla [vocab, hidden] y copiar
//! filas (embeddingLookup).

const std = @import("std");
const Tensor = @import("core").Tensor;
const embedding_mod = @import("embedding");
const gguf = @import("gguf");
const QuantWeight = @import("quant_weight").QuantWeight;

test "embeddingLookupQuant: paridad bit-exacta vs tabla f16 completa" {
    const allocator = std.testing.allocator;

    // Tabla sintética [hidden=32, vocab=64] en layout GGUF (dims[0]=hidden
    // contiguo). Quant q8_0 fabricado a mano: por bloque de 32 elems,
    // 2B f16 (escala d) + 32×int8 (quants) = 34B/bloque — layout canónico.
    const hidden: usize = 32;
    const vocab: usize = 64;
    const total = hidden * vocab;
    const bs = 32;
    const bb = 34;
    const n_blocks = total / bs; // 2048/32 = 64 (exacto)

    const src = try allocator.alloc(f32, total);
    defer allocator.free(src);
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rand = prng.random();
    for (src) |*v| v.* = rand.float(f32) * 2.0 - 1.0;

    const bytes = try allocator.alloc(u8, n_blocks * bb);
    defer allocator.free(bytes);
    for (0..n_blocks) |i| {
        const blk = src[i * bs .. (i + 1) * bs];
        // d = max|v| / 127 (convención GGUF q8_0)
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
        .name = "token_embd.weight",
        .n_dims = 2,
        .dims = dims,
        .dtype = .q8_0,
        .offset = 0,
    };
    const qw = QuantWeight.init(&info, bytes);

    // Referencia: dequant total → tabla [vocab, hidden] y embeddingLookup.
    // (bit-exacto: mismo dequantBlock de gguf en ambos caminos)
    var table = try Tensor(f16).alloc(allocator, &[_]usize{ vocab, hidden });
    defer table.deinit();
    var dec: [256]f32 = undefined;
    for (0..n_blocks) |i| {
        gguf.dequantBlock(.q8_0, bytes[i * bb .. (i + 1) * bb], &dec, bs);
        for (0..bs) |j| {
            const s = i * bs + j;
            const r = s % hidden;
            const c = s / hidden;
            table.data[c * hidden + r] = @floatCast(dec[j]);
        }
    }

    const tokens = [_]u32{ 0, 5, 63, 33, 7 };
    const seq = tokens.len;
    var out_ref = try Tensor(f16).alloc(allocator, &[_]usize{ 1, seq, hidden });
    defer out_ref.deinit();
    var out_q = try Tensor(f16).alloc(allocator, &[_]usize{ 1, seq, hidden });
    defer out_q.deinit();

    embedding_mod.embeddingLookup(table, &tokens, 1, seq, &out_ref);
    embedding_mod.embeddingLookupQuant(&qw, &tokens, 1, seq, &out_q);

    try std.testing.expectEqualSlices(f16, out_ref.data, out_q.data);
}

test "embeddingLookupQuant: cola de bloque no-alineada (hidden%bs!=0 imposible en GGUF, pero rows no-múltiplo sí)" {
    // Caso borde estructural: la fila de un token SIEMPRE es contigua en
    // el plano GGUF; el bucle de bloques de embeddingLookupQuant cubre
    // rangos cruzando límites de fila (first_block..last_block). Probar con
    // vocab=33 (número primo) para ejercitar la aritmética de división.
    const allocator = std.testing.allocator;
    const hidden: usize = 32;
    const vocab: usize = 33;
    const total = hidden * vocab;
    const bs = 32;
    const bb = 34;
    const n_blocks = (total + bs - 1) / bs; // 1056/32 = 33 exacto

    const src = try allocator.alloc(f32, total);
    defer allocator.free(src);
    var prng = std.Random.DefaultPrng.init(0xD1CE);
    const rand = prng.random();
    for (src) |*v| v.* = rand.float(f32) * 4.0 - 2.0;

    const bytes = try allocator.alloc(u8, n_blocks * bb);
    defer allocator.free(bytes);
    for (0..n_blocks) |i| {
        const rem = @min(bs, total - i * bs);
        const blk = src[i * bs .. i * bs + rem];
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
        .name = "token_embd.weight",
        .n_dims = 2,
        .dims = dims,
        .dtype = .q8_0,
        .offset = 0,
    };
    const qw = QuantWeight.init(&info, bytes);

    var table = try Tensor(f16).alloc(allocator, &[_]usize{ vocab, hidden });
    defer table.deinit();
    var dec: [256]f32 = undefined;
    for (0..n_blocks) |i| {
        const rem = @min(bs, total - i * bs);
        gguf.dequantBlock(.q8_0, bytes[i * bb .. i * bb + bb], &dec, rem);
        for (0..rem) |j| {
            const s = i * bs + j;
            const r = s % hidden;
            const c = s / hidden;
            table.data[c * hidden + r] = @floatCast(dec[j]);
        }
    }

    const tokens = [_]u32{ 0, 32, 16, 3 };
    const seq = tokens.len;
    var out_ref = try Tensor(f16).alloc(allocator, &[_]usize{ 1, seq, hidden });
    defer out_ref.deinit();
    var out_q = try Tensor(f16).alloc(allocator, &[_]usize{ 1, seq, hidden });
    defer out_q.deinit();

    embedding_mod.embeddingLookup(table, &tokens, 1, seq, &out_ref);
    embedding_mod.embeddingLookupQuant(&qw, &tokens, 1, seq, &out_q);

    try std.testing.expectEqualSlices(f16, out_ref.data, out_q.data);
}
