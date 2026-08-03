
//! Cuantización de pesos INT8 / INT4 para inferencia eficiente
//! Implementa: simétrica, asimétrica (zero-point), y de-grupos (per-channel/per-group)

const std = @import("std");
const Tensor = @import("core").Tensor;

// ═══════════════════════════════════════════════════════════════════════════════
// Cuantización Simétrica INT8:  w_q = round(w / scale),  scale = max(|w|) / 127
// ═══════════════════════════════════════════════════════════════════════════════

pub const QuantConfig = struct {
    bits: u4,           // 8 o 4
    symmetric: bool,    // true: zero_point = 0
    per_channel: bool,  // true: un scale por fila/columna
    group_size: usize,  // 0 = sin grupos, >0 = per-group (ej: 128)
};

pub const QuantizedTensor = struct {
    data: []u8,         // bytes cuantizados (2x INT4 por byte, o 1x INT8)
    shape: []const usize,
    scales: []f32,      // un scale por canal/grupo
    zero_points: []i8,  // un zero_point por canal/grupo (si asimétrico)
    config: QuantConfig,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *QuantizedTensor) void {
        self.allocator.free(self.data);
        self.allocator.free(self.shape);
        self.allocator.free(self.scales);
        if (self.zero_points.len > 0) self.allocator.free(self.zero_points);
    }

    pub fn numel(self: QuantizedTensor) usize {
        var n: usize = 1;
        for (self.shape) |s| n *= s;
        return n;
    }
};

/// Cuantizar tensor f32 a INT8 simétrico
pub fn quantizeInt8Symmetric(
    allocator: std.mem.Allocator,
    src: Tensor(f32),
    config: QuantConfig,
) !QuantizedTensor {
    std.debug.assert(config.bits == 8);

    const num_elements = src.numel();
    const data = try allocator.alloc(u8, num_elements);
    errdefer allocator.free(data);

    // Para simétrico simple: un solo scale global
    var max_abs: f32 = 0;
    var it = src.iterator();
    while (it.next()) |p| {
        const abs_v = @abs(p.*);
        if (abs_v > max_abs) max_abs = abs_v;
    }

    const scale = max_abs / 127.0;
    const scales = try allocator.alloc(f32, 1);
    scales[0] = scale;

    // Cuantizar
    it = src.iterator();
    var idx: usize = 0;
    while (it.next()) |p| : (idx += 1) {
        const q = @round(p.* / scale);
        const clamped = @max(-127.0, @min(127.0, q));
        data[idx] = @as(u8, @bitCast(@as(i8, @intFromFloat(clamped))));
    }

    const shape_copy = try allocator.dupe(usize, src.shape);

    return QuantizedTensor{
        .data = data,
        .shape = shape_copy,
        .scales = scales,
        .zero_points = &.{},
        .config = config,
        .allocator = allocator,
    };
}

/// Cuantizar tensor f32 a INT8 asimétrico con zero-point
/// w_q = round((w - zero) / scale)
/// scale = (max - min) / 255, zero = round(-min / scale)
pub fn quantizeInt8Asymmetric(
    allocator: std.mem.Allocator,
    src: Tensor(f32),
    config: QuantConfig,
) !QuantizedTensor {
    std.debug.assert(config.bits == 8);

    const num_elements = src.numel();
    const data = try allocator.alloc(u8, num_elements);
    errdefer allocator.free(data);

    var min_val: f32 = std.math.inf(f32);
    var max_val: f32 = -std.math.inf(f32);

    var it = src.iterator();
    while (it.next()) |p| {
        if (p.* < min_val) min_val = p.*;
        if (p.* > max_val) max_val = p.*;
    }

    const scale = (max_val - min_val) / 255.0;
    const zero_point_f = @round(-min_val / scale);
    const zero_point: i8 = @intCast(@min(255, @max(0, zero_point_f)));

    const scales = try allocator.alloc(f32, 1);
    scales[0] = scale;
    const zps = try allocator.alloc(i8, 1);
    zps[0] = zero_point;

    it = src.iterator();
    var idx: usize = 0;
    while (it.next()) |p| : (idx += 1) {
        const q = @round((p.* - min_val) / scale);
        const clamped = @max(0, @min(255, q));
        data[idx] = @intCast(clamped);
    }

    const shape_copy = try allocator.dupe(usize, src.shape);

    return QuantizedTensor{
        .data = data,
        .shape = shape_copy,
        .scales = scales,
        .zero_points = zps,
        .config = config,
        .allocator = allocator,
    };
}

/// Cuantización per-channel (una escala por fila de la matriz de pesos)
/// Útil para capas lineales donde cada fila de W es un vector de salida
pub fn quantizeInt8PerChannel(
    allocator: std.mem.Allocator,
    src: Tensor(f32),
    config: QuantConfig,
) !QuantizedTensor {
    std.debug.assert(config.bits == 8);
    std.debug.assert(src.shape.len == 2);

    const rows = src.shape[0];
    const cols = src.shape[1];
    const num_elements = rows * cols;

    const data = try allocator.alloc(u8, num_elements);
    errdefer allocator.free(data);

    const scales = try allocator.alloc(f32, rows);
    errdefer allocator.free(scales);

    const zps = if (config.symmetric)
        try allocator.alloc(i8, 0)
    else
        try allocator.alloc(i8, rows);
    errdefer if (!config.symmetric) allocator.free(zps);

    // Cuantizar por fila
    for (0..rows) |r| {
        var min_val: f32 = std.math.inf(f32);
        var max_val: f32 = -std.math.inf(f32);

        for (0..cols) |c| {
            const v = src.at2(r, c);
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }

        if (config.symmetric) {
            const max_abs = @max(@abs(min_val), @abs(max_val));
            const scale = max_abs / 127.0;
            scales[r] = scale;

            for (0..cols) |c| {
                const q = @round(src.at2(r, c) / scale);
                const clamped = @max(-127.0, @min(127.0, q));
                data[r * cols + c] = @as(u8, @bitCast(@as(i8, @intFromFloat(clamped))));
            }
        } else {
            const scale = (max_val - min_val) / 255.0;
            scales[r] = scale;
            zps[r] = @intCast(@min(255, @max(0, @round(-min_val / scale))));

            for (0..cols) |c| {
                const q = @round((src.at2(r, c) - min_val) / scale);
                const clamped = @max(0, @min(255, q));
                data[r * cols + c] = @intCast(clamped);
            }
        }
    }

    const shape_copy = try allocator.dupe(usize, src.shape);

    return QuantizedTensor{
        .data = data,
        .shape = shape_copy,
        .scales = scales,
        .zero_points = if (config.symmetric) &.{} else zps,
        .config = config,
        .allocator = allocator,
    };
}

// ═══════════════════════════════════════════════════════════════════════════════
// Cuantización INT4 — 2 valores por byte
// ═══════════════════════════════════════════════════════════════════════════════

pub fn quantizeInt4Symmetric(
    allocator: std.mem.Allocator,
    src: Tensor(f32),
    config: QuantConfig,
) !QuantizedTensor {
    std.debug.assert(config.bits == 4);

    const num_elements = src.numel();
    const num_bytes = (num_elements + 1) / 2; // 2x int4 por byte

    const data = try allocator.alloc(u8, num_bytes);
    errdefer allocator.free(data);
    @memset(data, 0);

    var max_abs: f32 = 0;
    var it = src.iterator();
    while (it.next()) |p| {
        const abs_v = @abs(p.*);
        if (abs_v > max_abs) max_abs = abs_v;
    }

    const scale = max_abs / 7.0; // rango INT4 simétrico: -7..7
    const scales = try allocator.alloc(f32, 1);
    scales[0] = scale;

    it = src.iterator();
    var idx: usize = 0;
    while (it.next()) |p| {
        const q = @round(p.* / scale);
        const clamped = @max(-7.0, @min(7.0, q));
        const val: u4 = @intCast(@as(i4, @intFromFloat(clamped)) & 0xF);

        const byte_idx = idx / 2;
        if (idx % 2 == 0) {
            data[byte_idx] = val; // nibble bajo
        } else {
            data[byte_idx] |= @as(u8, val) << 4; // nibble alto
        }
        idx += 1;
    }

    const shape_copy = try allocator.dupe(usize, src.shape);

    return QuantizedTensor{
        .data = data,
        .shape = shape_copy,
        .scales = scales,
        .zero_points = &.{},
        .config = config,
        .allocator = allocator,
    };
}

// ═══════════════════════════════════════════════════════════════════════════════
// De-cuantización
// ═══════════════════════════════════════════════════════════════════════════════

pub fn dequantizeToF32(
    allocator: std.mem.Allocator,
    q: QuantizedTensor,
) !Tensor(f32) {
    var dst = try Tensor(f32).alloc(allocator, q.shape);

    if (q.config.bits == 8) {
        if (q.config.per_channel and q.shape.len == 2) {
            // Per-channel dequant
            const rows = q.shape[0];
            const cols = q.shape[1];
            for (0..rows) |r| {
                const scale = q.scales[r];
                const zp: f32 = if (q.zero_points.len > 0) @floatFromInt(q.zero_points[r]) else 0;
                for (0..cols) |c| {
                    const qval: i8 = @bitCast(q.data[r * cols + c]);
                    const val: f32 = @floatFromInt(qval);
                    if (q.zero_points.len > 0) {
                        dst.ptr2(r, c).* = (val - zp) * scale;
                    } else {
                        dst.ptr2(r, c).* = val * scale;
                    }
                }
            }
        } else {
            // Global dequant
            const scale = q.scales[0];
            const zp: f32 = if (q.zero_points.len > 0) @floatFromInt(q.zero_points[0]) else 0;

            var it = dst.iterator();
            var idx: usize = 0;
            while (it.next()) |p| : (idx += 1) {
                const qval: i8 = @bitCast(q.data[idx]);
                const val: f32 = @floatFromInt(qval);
                if (q.zero_points.len > 0) {
                    p.* = (val - zp) * scale;
                } else {
                    p.* = val * scale;
                }
            }
        }
    } else if (q.config.bits == 4) {
        const scale = q.scales[0];
        var it = dst.iterator();
        var idx: usize = 0;
        while (it.next()) |p| {
            const byte_idx = idx / 2;
            const is_high = (idx % 2) == 1;
            const nibble: u4 = if (is_high)
                @intCast(q.data[byte_idx] >> 4)
            else
                @intCast(q.data[byte_idx] & 0xF);
            const signed: i4 = @bitCast(nibble);
            p.* = @as(f32, @floatFromInt(signed)) * scale;
            idx += 1;
        }
    }

    return dst;
}

// ═══════════════════════════════════════════════════════════════════════════════
// GEMM con pesos cuantizados INT8 (CPU, de-cuantizando on-the-fly)
// ═══════════════════════════════════════════════════════════════════════════════

/// GEMM donde B está cuantizado INT8, A y C son FP32
/// C = A @ dequant(B)
/// Optimizado para inferencia: solo de-cuantizamos los elementos necesarios
pub fn gemmWithQuantizedB(
    A: Tensor(f32),
    B_q: QuantizedTensor,
    C: *Tensor(f32),
    M: usize,
    N: usize,
    K: usize,
) void {
    std.debug.assert(B_q.config.bits == 8);
    std.debug.assert(B_q.shape.len == 2);
    std.debug.assert(B_q.shape[0] == K and B_q.shape[1] == N);

    const scale = B_q.scales[0];
    const has_zp = B_q.zero_points.len > 0;
    const zp: f32 = if (has_zp) @floatFromInt(B_q.zero_points[0]) else 0;

    @memset(C.data, 0);

    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0;
            for (0..K) |k| {
                const a_val = A.at2(i, k);
                const qval: i8 = @bitCast(B_q.data[k * N + j]);
                const b_val = if (has_zp)
                    (@as(f32, @floatFromInt(qval)) - zp) * scale
                else
                    @as(f32, @floatFromInt(qval)) * scale;
                sum += a_val * b_val;
            }
            C.ptr2(i, j).* = sum;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

test "INT8 symmetric quantize-dequantize" {
    const allocator = std.testing.allocator;
    var src = try Tensor(f32).alloc(allocator, &[_]usize{ 4, 4 });
    defer src.deinit();

    var rng = std.rand.DefaultPrng.init(42);
    src.randUniform(&rng, -2.0, 2.0);

    const config = QuantConfig{ .bits = 8, .symmetric = true, .per_channel = false, .group_size = 0 };
    var q = try quantizeInt8Symmetric(allocator, src, config);
    defer q.deinit();

    var dst = try dequantizeToF32(allocator, q);
    defer dst.deinit();

    // Verificar que el error es razonable (< 2% relativo típicamente)
    var max_err: f32 = 0;
    var it_src = src.iterator();
    var it_dst = dst.iterator();
    while (it_src.next()) |ps| {
        const pd = it_dst.next().?;
        const err = @abs(ps.* - pd.*);
        if (err > max_err) max_err = err;
    }
    try std.testing.expect(max_err < 0.05);
}

test "INT8 per-channel quantize-dequantize" {
    const allocator = std.testing.allocator;
    var src = try Tensor(f32).alloc(allocator, &[_]usize{ 8, 16 });
    defer src.deinit();

    var rng = std.rand.DefaultPrng.init(42);
    src.randUniform(&rng, -3.0, 3.0);

    const config = QuantConfig{ .bits = 8, .symmetric = true, .per_channel = true, .group_size = 0 };
    var q = try quantizeInt8PerChannel(allocator, src, config);
    defer q.deinit();

    var dst = try dequantizeToF32(allocator, q);
    defer dst.deinit();

    var max_err: f32 = 0;
    var it_src = src.iterator();
    var it_dst = dst.iterator();
    while (it_src.next()) |ps| {
        const pd = it_dst.next().?;
        const err = @abs(ps.* - pd.*);
        if (err > max_err) max_err = err;
    }
    try std.testing.expect(max_err < 0.05);
}

test "INT4 symmetric quantize-dequantize" {
    const allocator = std.testing.allocator;
    var src = try Tensor(f32).alloc(allocator, &[_]usize{ 4, 4 });
    defer src.deinit();

    var rng = std.rand.DefaultPrng.init(42);
    src.randUniform(&rng, -1.0, 1.0);

    const config = QuantConfig{ .bits = 4, .symmetric = true, .per_channel = false, .group_size = 0 };
    var q = try quantizeInt4Symmetric(allocator, src, config);
    defer q.deinit();

    var dst = try dequantizeToF32(allocator, q);
    defer dst.deinit();

    var max_err: f32 = 0;
    var it_src = src.iterator();
    var it_dst = dst.iterator();
    while (it_src.next()) |ps| {
        const pd = it_dst.next().?;
        const err = @abs(ps.* - pd.*);
        if (err > max_err) max_err = err;
    }
    // INT4 tiene menos precisión, tolerancia mayor
    try std.testing.expect(max_err < 0.3);
}

test "GEMM with quantized weights" {
    const allocator = std.testing.allocator;
    const M = 8;
    const K = 16;
    const N = 8;

    var A = try Tensor(f32).alloc(allocator, &[_]usize{ M, K });
    defer A.deinit();
    var W = try Tensor(f32).alloc(allocator, &[_]usize{ K, N });
    defer W.deinit();
    var C_ref = try Tensor(f32).alloc(allocator, &[_]usize{ M, N });
    defer C_ref.deinit();
    var C_q = try Tensor(f32).alloc(allocator, &[_]usize{ M, N });
    defer C_q.deinit();

    var rng = std.rand.DefaultPrng.init(42);
    A.randUniform(&rng, -1.0, 1.0);
    W.randUniform(&rng, -1.0, 1.0);

    // Referencia FP32
    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0;
            for (0..K) |k| {
                sum += A.at2(i, k) * W.at2(k, j);
            }
            C_ref.ptr2(i, j).* = sum;
        }
    }

    // Cuantizar W
    const config = QuantConfig{ .bits = 8, .symmetric = true, .per_channel = false, .group_size = 0 };
    var W_q = try quantizeInt8Symmetric(allocator, W, config);
    defer W_q.deinit();

    // GEMM con W cuantizado
    gemmWithQuantizedB(A, W_q, &C_q, M, N, K);

    // Verificar que el error es aceptable
    var max_err: f32 = 0;
    for (0..M) |i| {
        for (0..N) |j| {
            const err = @abs(C_ref.at2(i, j) - C_q.at2(i, j));
            if (err > max_err) max_err = err;
        }
    }
    try std.testing.expect(max_err < 1.0);
}


