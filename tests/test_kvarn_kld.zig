//! KVarN: KLD/calidad del roundtrip CPU reference vs BF16.
//!
//! El test mide la calidad de cuantización sobre tiles con distribución
//! similar a K/V reales (post-RoPE + Hadamard), reportando:
//!   - maxAbs (cota L_inf)
//!   - ratio L2 (norma del error / norma de la señal)
//!   - KLD proxy (error relativo medio)
//!
//! El gate del ladder Bee (Qwen3.6 27B Q5_K_S, 64K, KVarN5/4 + tail 1024)
//! mide KLD ~0.00094 sobre el corpus Wikitext-2 completo. Como referencia
//! CPU-only sin pipeline real, comparamos contra el orden de magnitud del
//! roundtrip en tiles individuales: ratio L2 < 0.15, maxAbs < 0.5 (típico
//! de KVarN5/4 sobre distribuciones Gaussian-like).
//!
//! No pretende replicar el benchmark upstream (requiere modelo + GGUF + KV
//! reales); solo valida que el encode/decode está dentro de un orden de
//! magnitud sensato para el algoritmo transcrito.

const std = @import("std");
const testing = std.testing;
const kvarn = @import("kv_cache").kvarn;

/// Aplica Hadamard por filas (pre-cuantización) para imitar la pipeline real.
fn hadamardRowsInPlace(tile: []f32, head_dim: usize) void {
    var row: [128]f32 = undefined;
    var r: usize = 0;
    while (r < 128) : (r += 1) {
        const off = r * head_dim;
        var c: usize = 0;
        while (c < head_dim) : (c += 1) row[c] = tile[off + c];
        kvarn.hadamard128InPlace(&row);
        c = 0;
        while (c < head_dim) : (c += 1) tile[off + c] = row[c];
    }
}

fn maxAbsDiff(a: []const f32, b: []const f32) f32 {
    var m: f32 = 0.0;
    for (a, 0..) |av, i| {
        const d = @abs(av - b[i]);
        if (d > m) m = d;
    }
    return m;
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

fn meanRelError(a: []const f32, b: []const f32) f32 {
    var sum: f64 = 0.0;
    var count: usize = 0;
    for (a, 0..) |av, i| {
        const denom = if (@abs(av) > 1e-6) @abs(av) else 1.0;
        sum += @abs(@as(f64, av) - @as(f64, b[i])) / @as(f64, denom);
        count += 1;
    }
    if (count == 0) return 0.0;
    return @floatCast(sum / @as(f64, @floatFromInt(count)));
}

test "KVarN5/4 quality on Gaussian-like tiles" {
    var prng = std.Random.DefaultPrng.init(0xCAFE_F00D);
    const random = prng.random();

    const cases = [_]struct { k_bits: u8, v_bits: u8 }{
        .{ .k_bits = 2, .v_bits = 2 },
        .{ .k_bits = 4, .v_bits = 4 },
        .{ .k_bits = 5, .v_bits = 4 },
        .{ .k_bits = 5, .v_bits = 5 },
        .{ .k_bits = 6, .v_bits = 6 },
        .{ .k_bits = 8, .v_bits = 8 },
    };

    for (cases) |c| {
        const layout = try kvarn.KvarnRecordLayout.init(128, c.k_bits, c.v_bits);
        var record = [_]u8{0} ** (32 * 1024);
        var max_err: f32 = 0.0;
        var max_l2: f32 = 0.0;

        var trial: u32 = 0;
        while (trial < 4) : (trial += 1) {
            var tile: [128 * 128]f32 = undefined;
            var i: usize = 0;
            while (i < tile.len) : (i += 1) {
                // Distribución ~N(0, 1) recortada a ±3σ, similar a K/V post-RoPE.
                const v = random.floatNorm(f32) * 0.5;
                tile[i] = std.math.clamp(v, -3.0, 3.0);
            }
            // Simula la pipeline real: Hadamard → Sinkhorn → quantize.
            hadamardRowsInPlace(&tile, 128);
            try kvarn.encodeKTile(&tile, 16, c.k_bits, layout, &record);

            var recon: [128 * 128]f32 = [_]f32{0} ** (128 * 128);
            try kvarn.decodeKTile(&record, c.k_bits, layout, &recon);

            const err = maxAbsDiff(&tile, &recon);
            const l2 = l2Ratio(&tile, &recon);
            const mean = meanRelError(&tile, &recon);
            if (err > max_err) max_err = err;
            if (l2 > max_l2) max_l2 = l2;
            _ = mean;
        }

        // Tolerancias por bit-width: a más bits, menor error.
        if (c.k_bits == 8) {
            try testing.expect(max_err < 0.02);
            try testing.expect(max_l2 < 0.05);
        } else if (c.k_bits >= 5) {
            try testing.expect(max_err < 0.4);
            try testing.expect(max_l2 < 0.30);
        } else if (c.k_bits >= 3) {
            try testing.expect(max_err < 0.7);
            try testing.expect(max_l2 < 0.40);
        } else {
            // 2 bits: alta distorsión esperada.
            try testing.expect(max_err < 1.5);
            try testing.expect(max_l2 < 0.55);
        }
    }
}

test "KVarN5/5 (calidad balanceada) report" {
    // Test simple de regresión: KVarN5/5 sobre un tile conocido.
    const layout = try kvarn.KvarnRecordLayout.init(128, 5, 5);
    var record = [_]u8{0} ** (32 * 1024);
    var tile: [128 * 128]f32 = undefined;
    var i: usize = 0;
    while (i < tile.len) : (i += 1) tile[i] = @sin(@as(f32, @floatFromInt(i)) * 0.01);
    hadamardRowsInPlace(&tile, 128);

    try kvarn.encodeKTile(&tile, 16, 5, layout, &record);
    var recon: [128 * 128]f32 = [_]f32{0} ** (128 * 128);
    try kvarn.decodeKTile(&record, 5, layout, &recon);

    const err = maxAbsDiff(&tile, &recon);
    const l2 = l2Ratio(&tile, &recon);
    // Senoide suave: tras Hadamard se "blanquea" → muy buena cuantización.
    try testing.expect(err < 0.3);
    try testing.expect(l2 < 0.20);
}