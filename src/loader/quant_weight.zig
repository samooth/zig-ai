//! QuantWeight — peso GGUF cuantizado que referencia directamente los bytes
//! mmap del archivo (sin copia). Dequantiza bajo demanda bloque a bloque a
//! f16/f32 con scratch acotado (un solo bloque), la estrategia de memoria
//! elegida para el modelo: los bytes cuantizados viven en el mapa y el f16
//! dequantizado se materializa en scratch reutilizable antes de cada matmul.
const std = @import("std");
const gguf = @import("gguf");

pub const QuantWeight = struct {
    /// Préstamo al GgufFile/archivo mmap; el peso debe vivir tanto como éste.
    info: *const gguf.TensorInfo,
    bytes: []const u8,

    pub const Self = @This();

    pub fn init(info: *const gguf.TensorInfo, bytes: []const u8) Self {
        return .{ .info = info, .bytes = bytes };
    }

    pub fn name(self: *const Self) []const u8 {
        return self.info.name;
    }

    pub fn dtype(self: *const Self) gguf.GgmlType {
        return self.info.dtype;
    }

    pub fn numel(self: *const Self) u64 {
        return self.info.numel();
    }

    pub fn shape(self: *const Self) []const u64 {
        return self.info.shape();
    }

    /// Dequantiza el peso completo a f16 en `out` (debe tener numel()
    /// elementos), bloque a bloque con un scratch f32 de un bloque.
    pub fn dequantToF16(self: *const Self, out: []f16) void {
        const info = self.info;
        const bs = info.dtype.blockSize();
        const bb = info.dtype.blockBytes();
        const num_blocks = (info.numel() + bs - 1) / bs;
        var tmp: [256]f32 = undefined;
        var src: usize = 0;
        var dst: usize = 0;
        for (0..num_blocks) |_| {
            const n = @min(bs, info.numel() - dst);
            gguf.dequantBlock(info.dtype, self.bytes[src .. src + bb], &tmp, n);
            for (0..n) |j| out[dst + j] = @floatCast(tmp[j]);
            src += bb;
            dst += n;
        }
    }

    /// Dequantiza un peso 2D [in, out] (layout GGUF) escribiéndolo TRANSPUESTO
    /// [out, in] en `out`, que es la orientación que esperan los tensores de
    /// las capas ([out, in] con trans_b=true en linearProjection).
    /// Para tensores no-2D se comporta igual que dequantToF16.
    pub fn dequantToF16Transposed(self: *const Self, out: []f16) void {
        const info = self.info;
        if (info.n_dims != 2) {
            self.dequantToF16(out);
            return;
        }
        const d0: usize = @intCast(info.dims[0]);
        const d1: usize = @intCast(info.dims[1]);
        const bs = info.dtype.blockSize();
        const bb = info.dtype.blockBytes();
        const num_blocks = (info.numel() + bs - 1) / bs;
        var tmp: [256]f32 = undefined;
        var src: usize = 0;
        var dst: usize = 0;
        for (0..num_blocks) |_| {
            const n = @min(bs, info.numel() - dst);
            gguf.dequantBlock(info.dtype, self.bytes[src .. src + bb], &tmp, n);
            for (0..n) |j| {
                const s = dst + j;
                const r = s / d1;
                const c = s % d1;
                out[c * d0 + r] = @floatCast(tmp[j]);
            }
            src += bb;
            dst += n;
        }
    }

    /// Dequantiza el peso completo a f32 en `out` (numel() elementos).
    pub fn dequantToF32(self: *const Self, out: []f32) void {
        const info = self.info;
        const bs = info.dtype.blockSize();
        const bb = info.dtype.blockBytes();
        const num_blocks = (info.numel() + bs - 1) / bs;
        var tmp: [256]f32 = undefined;
        var src: usize = 0;
        var dst: usize = 0;
        for (0..num_blocks) |_| {
            const n = @min(bs, info.numel() - dst);
            gguf.dequantBlock(info.dtype, self.bytes[src .. src + bb], &tmp, n);
            @memcpy(out[dst .. dst + n], tmp[0..n]);
            src += bb;
            dst += n;
        }
    }
};

const testing = std.testing;

test "QuantWeight iq3_s: dos bloques A+B dequantizan a f16 exacto" {
    // Bloques deterministas idénticos a los del test iq3_s de gguf.zig
    // (valores esperados verificados contra la referencia C de ggml).
    var ba: [110]u8 = [_]u8{0} ** 110;
    std.mem.writeInt(u16, ba[0..2], @bitCast(@as(f16, 1.0)), .little);
    for (0..16) |i| ba[2 + i] = @intCast(i);
    ba[106] = 0x01;

    var bb: [110]u8 = [_]u8{0} ** 110;
    std.mem.writeInt(u16, bb[0..2], @bitCast(@as(f16, 1.0)), .little);
    for (0..16) |i| bb[2 + i] = @intCast(i);
    bb[66] = 0x01;
    bb[67] = 0x02;
    bb[74] = 0x01;
    bb[75] = 0x03;
    bb[76] = 0x0f;
    bb[77] = 0xff;
    bb[78] = 0x80;
    bb[79] = 0x55;
    bb[80] = 0xaa;
    bb[81] = 0x00;

    const expect_a = [_]f32{
        3, 3, 3, 3, 9, 3, 3, 3,
        15, 3, 3, 3, 33, 3, 3, 3,
        45, 3, 3, 3, 3, 9, 3, 3,
        9, 9, 3, 3, 15, 9, 3, 3,
        9, 3, 1, 1, 13, 3, 1, 1,
        1, 5, 1, 1, 3, 5, 1, 1,
        11, 5, 1, 1, 7, 7, 1, 1,
        1, 9, 1, 1, 5, 9, 1, 1,
    };
    const expect_b = [_]f32{
        -7, 5, 9, 5, 3, 1, 1, 1,
        -5, -1, 1, 1, 11, 1, 1, 1,
        -15, -1, -1, -1, 1, 3, 1, 1,
        -3, -3, -1, -1, -5, -3, -1, -1,
        9, 3, 1, 1, 15, 7, 11, -5,
        -1, 5, -1, 1, -3, 5, -1, 1,
        11, -5, 1, -1, 7, -7, 1, -1,
        1, 9, 1, 1, 5, 9, 1, 1,
    };

    var info = gguf.TensorInfo{
        .name = "test",
        .n_dims = 1,
        .dims = .{ 512, 0, 0, 0 },
        .dtype = .iq3_s,
        .offset = 0,
    };
    var data: [220]u8 = undefined;
    @memcpy(data[0..110], ba[0..]);
    @memcpy(data[110..220], bb[0..]);

    const w = QuantWeight.init(&info, &data);
    try testing.expectEqual(@as(u64, 512), w.numel());

    var out16: [512]f16 = undefined;
    w.dequantToF16(&out16);
    for (expect_a, 0..) |exp, i| {
        try testing.expectApproxEqRel(exp, out16[i], 1e-3);
    }
    for (expect_b, 0..) |exp, i| {
        try testing.expectApproxEqRel(exp, out16[256 + i], 1e-3);
    }

    var out32: [512]f32 = undefined;
    w.dequantToF32(&out32);
    for (expect_a, 0..) |exp, i| {
        try testing.expectApproxEqRel(exp, out32[i], 1e-5);
    }
    for (expect_b, 0..) |exp, i| {
        try testing.expectApproxEqRel(exp, out32[256 + i], 1e-5);
    }
}

test "QuantWeight q8_0: bloques de 32 con bloque final parcial" {
    // 70 elementos = 2 bloques completos (64) + 6 elementos del tercero.
    const bb = 34;
    var data: [3 * bb]u8 = [_]u8{0} ** (3 * bb);
    var d: [70]f32 = undefined;
    for (0..3) |b| {
        const scale: f16 = @floatFromInt(@as(i32, @intCast(b + 1)));
        std.mem.writeInt(u16, data[b * bb..][0..2], @bitCast(scale), .little);
        for (0..32) |j| data[b * bb + 2 + j] = @intCast(j % 16);
    }

    var info = gguf.TensorInfo{
        .name = "t",
        .n_dims = 1,
        .dims = .{ 70, 0, 0, 0 },
        .dtype = .q8_0,
        .offset = 0,
    };
    const w = QuantWeight.init(&info, &data);
    w.dequantToF32(&d);
    // bloque 0: d=1 -> val = 1*j, j=0..15 (i8 bitCast de 0..15 = 0..15)
    try testing.expectApproxEqRel(@as(f32, 7.0), d[7], 1e-5);
    // bloque 1: d=2 -> val = 2*j
    try testing.expectApproxEqRel(@as(f32, 2.0 * 11.0), d[32 + 11], 1e-5);
    // bloque 2 (parcial, 6 elems): d=3 -> val = 3*j
    try testing.expectApproxEqRel(@as(f32, 3.0 * 5.0), d[64 + 5], 1e-5);
}
