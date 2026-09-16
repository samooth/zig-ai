//! lane-kvc P3 — tests de QuantWeight para BitNet i2_s (escala GLOBAL al
//! tensor, tail del buffer). El layout es el del quantizador CPU de BitNet
//! (ggml-bitnet-mad.cpp quantize_i2_s, QK=128): valor j del grupo g en byte
//! g*32 + j%32, bits [2*(j/32) .. +2); mapeo 0=-1, 1=0, 2=+1; escala f32
//! tras los (numel*2+7)/8 bytes de datos.
const std = @import("std");
const gguf = @import("gguf");
const quant_weight = @import("quant_weight");
const QuantWeight = quant_weight.QuantWeight;

test "QuantWeight i2_s: dequant completo y transpuesto con escala global" {
    const testing = std.testing;
    // Fixture: 256 valores (2 grupos de 128). Grupo 0: patrón cíclico
    // -1,0,+1. Grupo 1: todo +1. Escala 0.5 en el tail.
    var data: [128]u8 = [_]u8{0} ** 128; // 64 datos + 4 escala + pad
    for (0..128) |j| {
        const code: u8 = switch (j % 3) {
            0 => 0, // -1
            1 => 1, // 0
            else => 2, // +1
        };
        const byte_idx = 0 * 32 + (j % 32);
        // Packing real (gguf.zig i2sByteAt): (c0<<6)|(c1<<4)|(c2<<2)|c3 con
        // c_k = valor j del sub-rango [32k, 32k+32) => shift = 6 - 2*(j/32).
        // El fixture original usaba 2*(j/32) (convención invertida) —
        // corregido al layout verificado contra GGUF real (lane-kvc).
        data[byte_idx] |= code << @intCast(6 - 2 * (j / 32));
    }
    for (0..32) |b| data[32 + b] = 0xAA; // grupo 1: todos code 2
    std.mem.writeInt(u32, data[64..68], @bitCast(@as(f32, 0.5)), .little);

    var info = gguf.TensorInfo{
        .name = "t",
        .n_dims = 2,
        .dims = .{ 16, 16, 0, 0 }, // [in=16, out=16] -> 256 elementos
        .dtype = .i2_s,
        .offset = 0,
    };
    const w = QuantWeight.init(&info, &data);

    // dequant completo f32
    var out32: [256]f32 = undefined;
    w.dequantToF32(&out32);
    for (0..128) |j| {
        const expect: f32 = switch (j % 3) {
            0 => -0.5,
            1 => 0.0,
            else => 0.5,
        };
        try testing.expectApproxEqAbs(expect, out32[j], 1e-6);
    }
    for (128..256) |j| try testing.expectApproxEqAbs(@as(f32, 0.5), out32[j], 1e-6);

    // dequant completo f16
    var out16: [256]f16 = undefined;
    w.dequantToF16(&out16);
    for (0..128) |j| {
        const expect: f32 = switch (j % 3) {
            0 => -0.5,
            1 => 0.0,
            else => 0.5,
        };
        try testing.expectApproxEqAbs(expect, @as(f32, out16[j]), 1e-3);
    }

    // transpuesta f32: elemento flat s=(r,c) -> out[c*16+r]
    var outT: [256]f32 = undefined;
    w.dequantToF32Transposed(&outT);
    try testing.expectApproxEqAbs(@as(f32, -0.5), outT[0], 1e-6); // s=0 (0,0)
    try testing.expectApproxEqAbs(@as(f32, 0.0), outT[1], 1e-6); // s=1 (1,0)
    try testing.expectApproxEqAbs(@as(f32, 0.5), outT[2], 1e-6); // s=2 (2,0)
    try testing.expectApproxEqAbs(@as(f32, 0.0), outT[16], 1e-6); // s=16 (0,1)
    try testing.expectApproxEqAbs(@as(f32, 0.5), outT[17], 1e-6); // s=17 (1,1)

    // get_subtensor 2D fila 0, cols [0, 6): patrón -0.5, 0, +0.5, -0.5, 0, +0.5
    // (row 0 del layout GGUF = los primeros 16 flat elems)
    var sub1 = try w.get_subtensor(std.testing.allocator, 0, 1, 0, 6);
    defer sub1.deinit();
    const exp1 = [_]f32{ -0.5, 0.0, 0.5, -0.5, 0.0, 0.5 };
    for (exp1, 0..) |e, i| try testing.expectApproxEqAbs(e, @as(f32, sub1.data[i]), 1e-3);
}

test "QuantWeight i2_s: get_subtensor 2D coincide con transpuesta completa" {
    const testing = std.testing;
    // 512 valores (4 grupos): grupo 0/1 patrón -1,0,+1; grupos 2/3 todo -1.
    var data: [256]u8 = [_]u8{0} ** 256; // 128 datos + 4 escala + pad
    for (0..256) |j| {
        const code: u8 = switch (j % 3) {
            0 => 0,
            1 => 1,
            else => 2,
        };
        const g = j / 128;
        const jj = j % 128;
        data[g * 32 + (jj % 32)] |= code << @intCast(6 - 2 * (jj / 32));
    }
    for (256..512) |j| {
        const g = j / 128;
        const jj = j % 128;
        // code 0 (-1): el byte ya es 0 — el |= es no-op, explícito por claridad
        _ = g;
        _ = jj;
    }
    std.mem.writeInt(u32, data[128..132], @bitCast(@as(f32, 1.0)), .little);

    var info = gguf.TensorInfo{
        .name = "t",
        .n_dims = 2,
        .dims = .{ 32, 16, 0, 0 }, // [in=32, out=16] -> 512 elementos
        .dtype = .i2_s,
        .offset = 0,
    };
    const w = QuantWeight.init(&info, &data);

    var full_t: [512]f16 = undefined;
    w.dequantToF16Transposed(&full_t);

    // Subtensor: rows [2, 6), cols [8, 16)
    var sub = try w.get_subtensor(std.testing.allocator, 2, 6, 8, 16);
    defer sub.deinit();
    try testing.expectEqual(@as(usize, 4), sub.shape[0]);
    try testing.expectEqual(@as(usize, 8), sub.shape[1]);
    for (0..4) |orr| {
        for (0..8) |occ| {
            const expected = full_t[(2 + orr) * 32 + (8 + occ)];
            try testing.expectApproxEqAbs(@as(f32, expected), @as(f32, sub.data[orr * 8 + occ]), 1e-3);
        }
    }
}
