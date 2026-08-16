//! Codificación / decodificación de regiones KV-cache en formatos GGUF
//! (q8_0, q4_0, q4_1) para usarlos como backing store cuantizado.
//!
//! El layout coincide con los decuantizadores de `src/loader/gguf.zig`
//! (dequantQ8_0 / dequantQ4_0 / dequantQ4_1) — scale f16 (u16) por bloque de 32
//! elementos y datos en el formato GGUF canónico:
//!   q8_0: f16 scale + 32 int8            (34 bytes/bloque)   val = d * qs
//!   q4_0: f16 scale + 16 nibbles "split" (18 bytes/bloque)   val = d * (nibble - 8)
//!   q4_1: f16 scale + f16 min + 16 nib.  (20 bytes/bloque)   val = d * q + m
//! Los valores lógicos del KV-cache son f16: encode toma f16 y decode produce f16.

const std = @import("std");
const QuantFormat = @import("quant_types.zig").QuantFormat;

pub const BLOCK: usize = 32;

/// Bytes que ocupan `n_elements` lógicos en `format` (incluye metadatos).
pub fn quantBytes(format: QuantFormat, n_elements: usize) usize {
    const num_blocks = (n_elements + BLOCK - 1) / BLOCK;
    return num_blocks * format.bytesPerBlock();
}

/// Codifica `src` (f16) al `format` en un buffer recién alocado (layout canónico).
pub fn encodeToOwned(allocator: std.mem.Allocator, format: QuantFormat, src: []const f16) ![]u8 {
    const dst = try allocator.alloc(u8, quantBytes(format, src.len));
    encode(format, src, dst);
    return dst;
}

/// Codifica `src` (f16) al `format` en `dst` (tamaño `quantBytes(format, src.len)`).
pub fn encode(format: QuantFormat, src: []const f16, dst: []u8) void {
    const n = src.len;
    const num_blocks = (n + BLOCK - 1) / BLOCK;
    var blk: [BLOCK]f32 = undefined;
    var i: usize = 0;
    for (0..num_blocks) |bi| {
        for (0..BLOCK) |j| blk[j] = if (i + j < n) @as(f32, @floatCast(src[i + j])) else 0.0;
        const base = bi * format.bytesPerBlock();
        switch (format) {
            .q8_0 => encodeQ8_0(blk[0..], dst[base..][0..format.bytesPerBlock()]),
            .q4_0 => encodeQ4_0(blk[0..], dst[base..][0..format.bytesPerBlock()]),
            .q4_1 => encodeQ4_1(blk[0..], dst[base..][0..format.bytesPerBlock()]),
            .fp16 => {
                const src_bytes = std.mem.sliceAsBytes(src);
                const off = bi * BLOCK * 2;
                const take = @min(BLOCK * 2, dst[base..].len);
                @memcpy(dst[base..][0..take], src_bytes[off..][0..take]);
            },
            else => @memset(dst[base..][0..format.bytesPerBlock()], 0),
        }
        i += BLOCK;
    }
}

/// Decodifica `bytes` (formato canónico) a `out` (f16). `out.len` = num
/// elementos lógicos. Reaprovecha el layout GGUF canónico.

fn encodeQ8_0(blk: []const f32, dst: []u8) void {
    const d = if (maxAbs(blk) > 0) maxAbs(blk) / 127.0 else 1.0;
    writeF16(dst[0..2], d);
    for (blk, 0..) |v, j| {
        var q: i32 = @intFromFloat(@round(v / d));
        if (q < -127) q = -127;
        if (q > 127) q = 127;
        dst[2 + j] = @as(u8, @bitCast(@as(i8, @intCast(q))));
    }
}

fn encodeQ4_0(blk: []const f32, dst: []u8) void {
    const d = if (maxAbs(blk) > 0) maxAbs(blk) / 7.0 else 1.0;
    writeF16(dst[0..2], d);
    const half = BLOCK / 2;
    for (0..half) |j| {
        const lo: i32 = @intFromFloat(@round(blk[j] / d));
        const hi: i32 = @intFromFloat(@round(blk[j + half] / d));
        const lo_n: i32 = @min(@max(lo, -8), 7) + 8; // nibble codifica signed [-8..7]
        const hi_n: i32 = @min(@max(hi, -8), 7) + 8;
        dst[2 + j] = @as(u8, @intCast(lo_n & 0x0F)) | (@as(u8, @intCast(hi_n & 0x0F)) << 4);
    }
}

fn encodeQ4_1(blk: []const f32, dst: []u8) void {
    var max_val: f32 = -std.math.inf(f32);
    var min_val: f32 = std.math.inf(f32);
    for (blk) |v| { max_val = @max(max_val, v); min_val = @min(min_val, v); }
    const d = if (max_val - min_val > 0) (max_val - min_val) / 15.0 else 1.0;
    writeF16(dst[0..2], d);
    writeF16(dst[2..4], min_val);
    const half = BLOCK / 2;
    for (0..half) |j| {
        const lo: i32 = @intFromFloat(@round((blk[j] - min_val) / d));
        const hi: i32 = @intFromFloat(@round((blk[j + half] - min_val) / d));
        var lo_c: i32 = lo; if (lo_c < 0) lo_c = 0; if (lo_c > 15) lo_c = 15;
        var hi_c: i32 = hi; if (hi_c < 0) hi_c = 0; if (hi_c > 15) hi_c = 15;
        dst[4 + j] = @as(u8, @intCast(lo_c)) | (@as(u8, @intCast(hi_c)) << 4);
    }
}

/// Decodifica `bytes` (formato canónico) a `out` (f16). `out.len` = num
/// elementos lógicos. Reaprovecha el layout GGUF canónico.
pub fn decode(format: QuantFormat, bytes: []const u8, out: []f16) void {
    const n = out.len;
    if (n == 0) return;
    const blocks_total = (n + BLOCK - 1) / BLOCK;
    var tmp: [8192]f32 = undefined;
    const chunk_blocks = tmp.len / BLOCK;
    var bi: usize = 0;
    while (bi < blocks_total) : (bi += chunk_blocks) {
        const step_blocks = @min(chunk_blocks, blocks_total - bi);
        const step = step_blocks * BLOCK;
        const acc = bi * BLOCK;
        dequantF32(format, bytes[bi * format.bytesPerBlock()..], tmp[0..step]);
        const take = @min(step, n - acc);
        for (0..take) |j| out[acc + j] = @as(f16, @floatCast(tmp[j]));
    }
}

fn dequantF32(format: QuantFormat, bytes: []const u8, out: []f32) void {
    switch (format) {
        .q8_0 => {
            var i: usize = 0;
            while (i < out.len) : (i += BLOCK) {
                const blk_off = (i / BLOCK) * 34;
                const d_bits = std.mem.readInt(u16, bytes[blk_off..][0..2], .little);
                const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
                const qs = bytes[blk_off + 2 ..][0..BLOCK];
                for (0..@min(BLOCK, out.len - i)) |j| {
                    out[i + j] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(qs[j]))));
                }
            }
        },
        .q4_0 => {
            var i: usize = 0;
            while (i < out.len) : (i += BLOCK) {
                const blk_off = (i / BLOCK) * 18;
                const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[blk_off..][0..2], .little))));
                const qs = bytes[blk_off + 2 ..][0..(BLOCK / 2)];
                const half = @min(BLOCK / 2, out.len - i);
                for (0..half) |j| {
                    const lo: i32 = @as(i32, @intCast(qs[j] & 0x0F)) - 8;
                    const hi: i32 = @as(i32, @intCast(qs[j] >> 4)) - 8;
                    out[i + j] = d * @as(f32, @floatFromInt(lo));
                    if (j + BLOCK / 2 < out.len - i) out[i + j + BLOCK / 2] = d * @as(f32, @floatFromInt(hi));
                }
            }
        },
        .q4_1 => {
            var i: usize = 0;
            while (i < out.len) : (i += BLOCK) {
                const blk_off = (i / BLOCK) * 20;
                const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[blk_off..][0..2], .little))));
                const m_bits = std.mem.readInt(u16, bytes[blk_off + 2 ..][0..2], .little);
                const m: f32 = @floatCast(@as(f16, @bitCast(m_bits)));
                const qs = bytes[blk_off + 4 ..][0..(BLOCK / 2)];
                const half = @min(BLOCK / 2, out.len - i);
                for (0..half) |j| {
                    const lo: i32 = @as(i32, @intCast(qs[j] & 0x0F));
                    const hi: i32 = @as(i32, @intCast(qs[j] >> 4));
                    out[i + j] = d * @as(f32, @floatFromInt(lo)) + m;
                    if (j + BLOCK / 2 < out.len - i) out[i + j + BLOCK / 2] = d * @as(f32, @floatFromInt(hi)) + m;
                }
            }
        },
        .fp16 => {
            const src = std.mem.bytesAsSlice(f16, bytes);
            for (out, src[0..out.len]) |*o, s| o.* = @as(f32, @floatCast(s));
        },
        .fp32 => {
            const src = std.mem.bytesAsSlice(f32, bytes);
            @memcpy(out, src[0..out.len]);
        },
        else => @memset(out, 0),
    }
}

fn maxAbs(s: []const f32) f32 {
    var m: f32 = 0;
    for (s) |v| m = @max(m, @abs(v));
    return m;
}

fn writeF16(dst: []u8, v: f32) void {
    const bits = @as(u16, @bitCast(@as(f16, @floatCast(v))));
    std.mem.writeInt(u16, dst[0..2], bits, .little);
}

test "q8_0 roundtrip" {
    const fmt: QuantFormat = .q8_0;
    const n: usize = 256;
    var src: [n]f16 = undefined;
    for (0..n) |i| src[i] = @as(f16, @floatCast(@sin(@as(f32, @floatFromInt(i)) * 0.1) * 3.5));
    const qbytes = quantBytes(fmt, n);
    const dst = try std.testing.allocator.alloc(u8, qbytes);
    defer std.testing.allocator.free(dst);
    encode(fmt, &src, dst);
    var back: [n]f16 = undefined;
    decode(fmt, dst, &back);
    var max_err: f32 = 0;
    for (0..n) |i| max_err = @max(max_err, @abs(@as(f32, @floatCast(src[i])) - @as(f32, @floatCast(back[i]))));
    try std.testing.expect(max_err < 0.05);
}

test "q4_0 roundtrip" {
    const fmt: QuantFormat = .q4_0;
    const n: usize = 256;
    var src: [n]f16 = undefined;
    for (0..n) |i| src[i] = @as(f16, @floatCast(@cos(@as(f32, @floatFromInt(i)) * 0.13) * 2.0));
    const qbytes = quantBytes(fmt, n);
    const dst = try std.testing.allocator.alloc(u8, qbytes);
    defer std.testing.allocator.free(dst);
    encode(fmt, &src, dst);
    var back: [n]f16 = undefined;
    decode(fmt, dst, &back);
    var max_err: f32 = 0;
    for (0..n) |i| max_err = @max(max_err, @abs(@as(f32, @floatCast(src[i])) - @as(f32, @floatCast(back[i]))));
    try std.testing.expect(max_err < 0.05);
}

comptime {
    const fmt: QuantFormat = .q8_0;
    const n: usize = 256;
    const qbytes = quantBytes(fmt, n);
    const _check_q8 = qbytes == 34 * 8;
    const _check_q4 = quantBytes(.q4_0, 256) == 18 * 8;
    _ = .{ _check_q8, _check_q4 };
}

test "quantBytes / bytesPerBlock" {
    try std.testing.expectEqual(34 * 8, quantBytes(.q8_0, 256));
    try std.testing.expectEqual(18 * 8, quantBytes(.q4_0, 256));
    try std.testing.expectEqual(34, QuantFormat.q8_0.bytesPerBlock());
    try std.testing.expectEqual(18, QuantFormat.q4_0.bytesPerBlock());
    try std.testing.expectEqual("q8_0", QuantFormat.q8_0.toString());
    try std.testing.expectEqual(QuantFormat.q8_0, QuantFormat.fromString("q8_0").?);
    try std.testing.expectEqual(QuantFormat.fp16, QuantFormat.fromString("fp16").?);
    try std.testing.expectEqual(QuantFormat.fp16, QuantFormat.fromString("none").?);
    try std.testing.expect(QuantFormat.fromString("bogus") == null);
}


