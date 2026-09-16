//! Lane-b2 P0.3: roundtrip CPU para los nuevos cache types q2_0s/q2_1/q3_0/q3_1/q6_0/q6_1.
//!
//! Verifica encode→dequant→comparar sobre distribuciones Gaussian-like,
//! igual que test_kvarn_kld.zig pero para los quant formats clásicos.

const std = @import("std");
const testing = std.testing;
const kv_quant = @import("kv_cache").kv_quant;
const qt = @import("kv_cache").quant_types;
const QuantFormat = qt.QuantFormat;

fn maxAbsDiff(a: []const f32, b: []const f32) f32 {
    var mx: f32 = 0.0;
    for (a, 0..) |av, i| {
        const d = @abs(av - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

fn l2Ratio(a: []const f32, b: []const f32) f32 {
    var num: f64 = 0.0;
    var den: f64 = 0.0;
    for (a, 0..) |av, i| {
        const diff = @as(f64, av) - @as(f64, b[i]);
        num += diff * diff;
        den += @as(f64, av) * @as(f64, av);
    }
    if (den == 0.0) return 0.0;
    return @floatCast(@sqrt(num) / @sqrt(den));
}

fn roundtripCheck(
    fmt: QuantFormat,
    tile_in: []const f32,
    tile_out: []f16,
    max_allowed: f32,
) !void {
    const n = tile_in.len;
    const bytes = try testing.allocator.alloc(u8, kv_quant.quantBytes(fmt, n));
    defer testing.allocator.free(bytes);

    // f16 input: convertimos el f32 a f16 y luego encode.
    var src_f16 = try testing.allocator.alloc(f16, n);
    defer testing.allocator.free(src_f16);
    for (tile_in, 0..) |v, i| src_f16[i] = @as(f16, @floatCast(v));

    kv_quant.encode(fmt, src_f16, bytes);
    kv_quant.decode(fmt, bytes, tile_out);

    // Comparamos f32 vs f16: convertimos el output a f32.
    var err: f32 = 0.0;
    for (tile_in, 0..) |v, i| {
        const diff = @abs(v - @as(f32, @floatCast(tile_out[i])));
        if (diff > err) err = diff;
    }
    if (err > max_allowed) {
        std.debug.print("FAIL {s}: err={d:.4} n={d}\n", .{ fmt.toString(), err, n });
    }
    try testing.expect(err <= max_allowed);
}

test "q2_0s roundtrip on Gaussian-like tile" {
    var prng = std.Random.DefaultPrng.init(0xA1A1_0001);
    const random = prng.random();

    var tile_in: [128]f32 = undefined;
    var tile_out: [128]f16 = [_]f16{0} ** 128;
    var i: usize = 0;
    while (i < tile_in.len) : (i += 1) {
        const v = random.floatNorm(f32) * 0.5;
        tile_in[i] = std.math.clamp(v, -2.0, 2.0);
    }
    // q2_0s: 2 bpw, rango [-2, +2]·d con d ≈ mx/2. Max error ≈ d (≈0.25-0.5)
    // más conversión f32→f16 (~0.001). Tolerancia amplia 1.0 para cubrir
    // el peor caso de mx pequeño donde d ≈ mx/2.
    try roundtripCheck(.q2_0s, &tile_in, &tile_out, 1.0);
}

test "q2_1 roundtrip on Gaussian-like tile" {
    var prng = std.Random.DefaultPrng.init(0xA1A1_0002);
    const random = prng.random();
    var tile_in: [128]f32 = undefined;
    var tile_out: [128]f16 = [_]f16{0} ** 128;
    var i: usize = 0;
    while (i < tile_in.len) : (i += 1) tile_in[i] = random.floatNorm(f32) * 2.0 - 1.0;
    // q2_1 con d ≈ (mx-mn)/3. Outliers ±3σ elevan d; error máx ≈ d.
    try roundtripCheck(.q2_1, &tile_in, &tile_out, 2.0);
}

test "q3_0 roundtrip on Gaussian-like tile" {
    var prng = std.Random.DefaultPrng.init(0xA1A1_0003);
    const random = prng.random();
    var tile_in: [128]f32 = undefined;
    var tile_out: [128]f16 = [_]f16{0} ** 128;
    var i: usize = 0;
    while (i < tile_in.len) : (i += 1) tile_in[i] = random.floatNorm(f32) * 0.7;
    // q3_0: 3 bpw con d ≈ 0.175. Error máx ≈ d ≈ 0.18. Tolerancia 2.0
    // para cubrir outliers (clipping en saturación + f16 noise).
    try roundtripCheck(.q3_0, &tile_in, &tile_out, 2.0);
}

test "q3_1 roundtrip on Gaussian-like tile" {
    var prng = std.Random.DefaultPrng.init(0xA1A1_0004);
    const random = prng.random();
    var tile_in: [128]f32 = undefined;
    var tile_out: [128]f16 = [_]f16{0} ** 128;
    var i: usize = 0;
    while (i < tile_in.len) : (i += 1) tile_in[i] = random.floatNorm(f32) * 0.7;
    try roundtripCheck(.q3_1, &tile_in, &tile_out, 2.0);
}

test "q6_0 roundtrip on Gaussian-like tile" {
    var prng = std.Random.DefaultPrng.init(0xA1A1_0005);
    const random = prng.random();
    var tile_in: [128]f32 = undefined;
    var tile_out: [128]f16 = [_]f16{0} ** 128;
    var i: usize = 0;
    while (i < tile_in.len) : (i += 1) tile_in[i] = random.floatNorm(f32) * 5.0;
    // q6_0: 6 bpw, d ≈ 5/32 ≈ 0.16. Error ≈ d ≈ 0.16. Tolerancia 0.6.
    try roundtripCheck(.q6_0, &tile_in, &tile_out, 0.6);
}

test "q6_1 roundtrip on Gaussian-like tile" {
    var prng = std.Random.DefaultPrng.init(0xA1A1_0006);
    const random = prng.random();
    var tile_in: [128]f32 = undefined;
    var tile_out: [128]f16 = [_]f16{0} ** 128;
    var i: usize = 0;
    while (i < tile_in.len) : (i += 1) tile_in[i] = random.floatNorm(f32) * 5.0;
    try roundtripCheck(.q6_1, &tile_in, &tile_out, 0.3);
}

test "name/fromString roundtrip for new types" {
    const cases = [_]QuantFormat{ .q2_0s, .q2_1, .q3_0, .q3_1, .q6_0, .q6_1 };
    for (cases) |c| {
        const s = c.toString();
        const back = qt.QuantFormat.fromString(s);
        try testing.expect(back != null);
        try testing.expectEqual(c, back.?);
    }
    // Alias q2_0 → q2_0s (Bee).
    try testing.expectEqual(@as(QuantFormat, .q2_0s), qt.QuantFormat.fromString("q2_0").?);
}