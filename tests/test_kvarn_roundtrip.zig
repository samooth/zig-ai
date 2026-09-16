//! KVarN: roundtrip bit-exacto de encode/decode CPU reference.
//!
//! Verifica:
//!   - Hadamard 128: Parseval + involutiva.
//!   - Bit packing: roundtrip a todos los bit-widths válidos.
//!   - fp16 ↔ fp32: roundtrip en valores representativos.
//!   - KvarnType parse: roundtrip del nombre canónico.
//!   - KvarnRecordLayout: alineación a 32B (C1 freeze) y offsets coherentes.
//!   - K tile roundtrip: tile fp32 aleatorio → encodeKTile → decodeKTile y
//!     verificación de error máximo acotado (tolerancia cuantización).
//!   - V tile roundtrip: análogo a K con la codificación V.
//!   - Ring buffer exacto: writeToken / tokenAt / numValid coherentes.

const std = @import("std");
const testing = std.testing;
const kvarn = @import("kv_cache").kvarn;

test "KvarnType name and parse roundtrip for all valid pairs" {
    const kvs = [_]kvarn.KvarnType{
        .{ .key_bits = 2, .value_bits = 2 },
        .{ .key_bits = 2, .value_bits = 3 },
        .{ .key_bits = 4, .value_bits = 5 },
        .{ .key_bits = 5, .value_bits = 4 },
        .{ .key_bits = 8, .value_bits = 8 },
    };
    for (kvs) |t| {
        const s = t.name();
        try testing.expect(t.isValid());
        const back = kvarn.KvarnType.parse(s).?;
        try testing.expectEqual(t.key_bits, back.key_bits);
        try testing.expectEqual(t.value_bits, back.value_bits);
    }
}

test "KvarnType rejects invalid bit pairs" {
    try testing.expect(!kvarn.isValidBits(1));
    try testing.expect(!kvarn.isValidBits(7));
    try testing.expect(!kvarn.isValidBits(9));
    try testing.expect(kvarn.isValidBits(2));
    try testing.expect(kvarn.isValidBits(8));
    try testing.expect(kvarn.KvarnType.parse("kvarn_k1v4_g128") == null);
    try testing.expect(kvarn.KvarnType.parse("kvarn_k5v9_g128") == null);
    try testing.expect(kvarn.KvarnType.parse("garbage") == null);
}

test "KvarnRecordLayout respects 32B alignment for all head_dims" {
    const hds = [_]u32{ 128, 256, 512 };
    const bits = [_]u8{ 2, 3, 4, 5, 6, 8 };
    for (hds) |hd| {
        for (bits) |k| {
            for (bits) |v| {
                const layout = try kvarn.KvarnRecordLayout.init(hd, k, v);
                try testing.expectEqual(@as(usize, 0) % kvarn.KVAR_L2_SECTOR, layout.tile_bytes % kvarn.KVAR_L2_SECTOR);
                try testing.expect(layout.tile_bytes > layout.v_zp_off);
                try testing.expect(layout.v_payload_off > layout.k_s_row_off);
                try testing.expect(layout.k_s_col_off >= layout.k_payload_off + layout.k_payload_bytes);
            }
        }
    }
}

test "Hadamard 128 preserves Parseval on random vectors" {
    var prng = std.Random.DefaultPrng.init(0x1234_5678);
    const random = prng.random();

    var trial: u32 = 0;
    while (trial < 8) : (trial += 1) {
        var values: [128]f32 = undefined;
        var original_norm: f64 = 0.0;
        var i: usize = 0;
        while (i < 128) : (i += 1) {
            values[i] = random.float(f32) * 4.0 - 2.0;
            original_norm += @as(f64, values[i]) * @as(f64, values[i]);
        }

        kvarn.hadamard128InPlace(&values);

        var new_norm: f64 = 0.0;
        for (values) |v| new_norm += @as(f64, v) * @as(f64, v);

        try testing.expectApproxEqAbs(original_norm, new_norm, 1e-3);
    }
}

test "Hadamard 128 applied twice returns identity" {
    var prng = std.Random.DefaultPrng.init(0xCAFE_BABE);
    const random = prng.random();

    var trial: u32 = 0;
    while (trial < 4) : (trial += 1) {
        var values: [128]f32 = undefined;
        var original: [128]f32 = undefined;
        var i: usize = 0;
        while (i < 128) : (i += 1) {
            values[i] = random.float(f32) * 2.0 - 1.0;
            original[i] = values[i];
        }
        kvarn.hadamard128InPlace(&values);
        kvarn.hadamard128InPlace(&values);
        for (values, 0..) |v, idx| {
            try testing.expectApproxEqAbs(original[idx], v, 1e-3);
        }
    }
}

test "fp16 roundtrip preserves representative values" {
    const cases = [_]f32{
        0.0, 1.0,  -1.0,  0.5,    -0.5, 1.5,   -1.5, 1e-3, -1e-3,
        1e3, -1e3, 100.0, -100.0, 1e-5, -1e-5,
    };
    for (cases) |v| {
        const h = kvarn.floatToF16(v);
        const back = kvarn.f16ToFloat(h);
        const tol: f32 = @abs(v) * 5e-3 + 1e-3;
        try testing.expectApproxEqAbs(v, back, tol);
    }
}

test "bit packing roundtrip for every supported bit-width" {
    const cases = [_]struct { bits: u8, vals: []const u8 }{
        .{ .bits = 2, .vals = &[_]u8{ 0, 1, 2, 3, 0, 3, 1, 2, 2, 0, 1, 3 } },
        .{ .bits = 3, .vals = &[_]u8{ 0, 1, 7, 3, 5, 6, 2, 4 } },
        .{ .bits = 4, .vals = &[_]u8{ 0, 15, 7, 8, 1, 14, 12, 3, 9 } },
        .{ .bits = 5, .vals = &[_]u8{ 0, 31, 16, 15, 1, 30, 17 } },
        .{ .bits = 6, .vals = &[_]u8{ 0, 63, 32, 31, 1, 62, 33, 17 } },
        .{ .bits = 8, .vals = &[_]u8{ 0, 255, 128, 1, 127, 200 } },
    };
    for (cases) |c| {
        var buf: [128]u8 = [_]u8{0} ** 128;
        for (c.vals, 0..) |v, i| kvarn.packBit(&buf, i, c.bits, v);
        for (c.vals, 0..) |v, i| {
            const got = kvarn.unpackBit(&buf, i, c.bits);
            try testing.expectEqual(v, got);
        }
    }
}

test "K tile roundtrip bounded error" {
    // Tile aleatorio → encodeKTile → decodeKTile → comparar.
    var prng = std.Random.DefaultPrng.init(0xDEAD_BEEF);
    const random = prng.random();

    const bits_k: u8 = 5;
    const layout = try kvarn.KvarnRecordLayout.init(128, bits_k, 4);
    var tile: [128 * 128]f32 = undefined;
    var i: usize = 0;
    while (i < tile.len) : (i += 1) {
        tile[i] = random.float(f32) * 2.0 - 1.0;
    }
    var record = [_]u8{0} ** (32 * 1024); // holgura ≥ recordBytes(128,5,4)
    try kvarn.encodeKTile(&tile, 16, bits_k, layout, &record);

    var reconstructed: [128 * 128]f32 = [_]f32{0} ** (128 * 128);
    try kvarn.decodeKTile(&record, bits_k, layout, &reconstructed);

    // Tolerancia amplia: Sinkhorn + 5 bits + scales absorbidas.
    var max_err: f32 = 0.0;
    for (reconstructed, 0..) |r, idx| {
        const err = @abs(r - tile[idx]);
        if (err > max_err) max_err = err;
    }
    try testing.expect(max_err < 0.3);
}

test "V tile roundtrip bounded error" {
    var prng = std.Random.DefaultPrng.init(0xBEEF_C0DE);
    const random = prng.random();

    const bits_v: u8 = 4;
    const layout = try kvarn.KvarnRecordLayout.init(128, 4, bits_v);
    var tile: [128 * 128]f32 = undefined;
    var i: usize = 0;
    while (i < tile.len) : (i += 1) {
        tile[i] = random.float(f32) * 2.0 - 1.0;
    }
    var record = [_]u8{0} ** (32 * 1024);
    try kvarn.encodeVTile(&tile, 16, bits_v, layout, &record);

    var reconstructed: [128 * 128]f32 = [_]f32{0} ** (128 * 128);
    try kvarn.decodeVTile(&record, bits_v, layout, &reconstructed);

    var max_err: f32 = 0.0;
    for (reconstructed, 0..) |r, idx| {
        const err = @abs(r - tile[idx]);
        if (err > max_err) max_err = err;
    }
    try testing.expect(max_err < 0.5);
}

test "KvarnExactRing writeToken and tokenAt consistency" {
    const head_dim: u32 = 128;
    const slots = 1;
    var buffer: [slots * 128]f16 = [_]f16{0} ** (slots * 128);
    var ring = try kvarn.KvarnExactRing(slots).init(buffer[0..], head_dim);

    try testing.expectEqual(@as(u32, 0), ring.numValid());

    var tok: [128]f16 = [_]f16{0} ** 128;
    for (tok[0..], 0..) |_, i| tok[i] = @as(f16, @floatFromInt(i + 1));
    ring.writeToken(&tok);
    try testing.expectEqual(@as(u32, 1), ring.numValid());

    const back = ring.tokenAt(0);
    for (back[0..head_dim], 0..) |v, i| {
        try testing.expectEqual(@as(f16, @floatFromInt(i + 1)), v);
    }
}

test "KvarnExactRing rollover at capacity" {
    const head_dim: u32 = 128;
    const slots = 2;
    var buffer: [2 * 128]f16 = [_]f16{0} ** (2 * 128);
    var ring = try kvarn.KvarnExactRing(slots).init(buffer[0..], head_dim);

    var tok_a: [128]f16 = [_]f16{1.0} ** 128;
    var tok_b: [128]f16 = [_]f16{2.0} ** 128;
    var tok_c: [128]f16 = [_]f16{3.0} ** 128;

    ring.writeToken(&tok_a); // idx 0 → slot 0
    ring.writeToken(&tok_b); // idx 1 → slot 1
    try testing.expectEqual(@as(u32, 2), ring.numValid());

    ring.writeToken(&tok_c); // idx 2 → slot 0 (rollover)
    try testing.expectEqual(@as(u32, 2), ring.numValid());

    // tokenAt(0) = más reciente = C
    const recent = ring.tokenAt(0);
    try testing.expectEqual(@as(f16, 3.0), recent[0]);

    // tokenAt(1) = más antiguo = B
    const old = ring.tokenAt(1);
    try testing.expectEqual(@as(f16, 2.0), old[0]);
}

test "KvarnRecordLayout init rejects bad head_dim and bad bits" {
    // 9.12 F1: 64 pasó a ser VÁLIDO (antes rejected) — el nuevo límite
    // inválido es 32.
    try testing.expectError(error.UnsupportedHeadDim, kvarn.KvarnRecordLayout.init(32, 4, 4));
    try testing.expectError(error.UnsupportedHeadDim, kvarn.KvarnRecordLayout.init(1024, 4, 4));
    try testing.expectError(error.UnsupportedKvarBits, kvarn.KvarnRecordLayout.init(128, 1, 4));
    try testing.expectError(error.UnsupportedKvarBits, kvarn.KvarnRecordLayout.init(128, 4, 7));
}
// ============================================================================
// 9.4 (lane-b) D2 ratification: cross-slice WHT por slices (CPU ref)
// ============================================================================

test "D2: hadamardSlicesRows D=256 Parseval + involutiva" {
    const hd: usize = 256;
    var tile: [128 * hd]f32 = undefined;
    var original: [128 * hd]f32 = undefined;
    var rng = std.Random.Xoshiro256.init(0xD2_1001);
    for (&tile) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
    @memcpy(&original, &tile);

    var norm_o: f64 = 0;
    for (&original) |v| norm_o += @as(f64, v) * @as(f64, v);
    kvarn.hadamardSlicesRows(&tile, hd);
    var norm_t: f64 = 0;
    for (&tile) |v| norm_t += @as(f64, v) * @as(f64, v);
    try testing.expectApproxEqAbs(norm_o, norm_t, norm_o * 1e-4);

    kvarn.hadamardSlicesRows(&tile, hd);
    var max_diff: f32 = 0;
    for (&tile, &original) |t, o| max_diff = @max(max_diff, @abs(t - o));
    try testing.expect(max_diff < 1e-4);
}

test "D2: hadamardSlicesRows D=256 = wht_128×slices + butterfly manual" {
    // Espejo EXACTO de fattn_kvarn_wht_cross_slices<2> (portable.cu:728).
    const hd: usize = 256;
    var tile: [128 * hd]f32 = undefined;
    var rng = std.Random.Xoshiro256.init(0xD2_1002);
    for (&tile) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
    const orig = tile;

    kvarn.hadamardSlicesRows(&tile, hd);

    var r: usize = 0;
    while (r < 128) : (r += 1) {
        var s0: [128]f32 = undefined;
        var s1: [128]f32 = undefined;
        for (0..128) |d| {
            s0[d] = orig[r * hd + d];
            s1[d] = orig[r * hd + 128 + d];
        }
        kvarn.hadamard128InPlace(&s0);
        kvarn.hadamard128InPlace(&s1);
        for (0..128) |d| {
            const a = s0[d];
            const b = s1[d];
            try testing.expectApproxEqAbs((a + b) * 0.707106781186547524, tile[r * hd + d], 1e-5);
            try testing.expectApproxEqAbs((a - b) * 0.707106781186547524, tile[r * hd + 128 + d], 1e-5);
        }
    }
}

test "D2: roundtrip D=256 per cabezas físicas SNR gate" {
    // Cadena beellama completa: tile [128][256] → hadamardSlicesRows →
    // extract/encode/decode layout-128 per slice → inject → inversa →
    // SNR ‖err‖/‖ref‖ < 0.15 (gate manager lane-c kvarn4).
    const bits: u8 = 4;
    const layout = try kvarn.KvarnRecordLayout.init(128, bits, bits);
    const rec_bytes = kvarn.recordBytes(128, bits, bits);

    var tile: [128 * 256]f32 = undefined;
    var rng = std.Random.Xoshiro256.init(0xD2_1003);
    for (&tile) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
    const original = tile;

    kvarn.hadamardSlicesRows(&tile, 256);

    // recordBytes(128, k4v4) = 17920 + pad 32 = 17952 bytes.
    const kvarn_rec_len = 17952;
    var recs: [2][kvarn_rec_len]u8 = .{ [_]u8{0} ** kvarn_rec_len, [_]u8{0} ** kvarn_rec_len };
    var slice_buf: [128 * 128]f32 = undefined;
    for (0..2) |s| {
        kvarn.extractSlice(&tile, 256, s, &slice_buf);
        try kvarn.encodeKTile(&slice_buf, 3, bits, layout, &recs[s]);
    }
    try testing.expect(rec_bytes <= kvarn_rec_len);

    var recon: [128 * 256]f32 = undefined;
    var d_buf: [128 * 128]f32 = undefined;
    for (0..2) |s| {
        try kvarn.decodeKTile(&recs[s], bits, layout, &d_buf);
        kvarn.injectSlice(&recon, 256, s, &d_buf);
    }
    kvarn.hadamardSlicesRows(&recon, 256);

    var err: f64 = 0;
    var ref: f64 = 0;
    for (&recon, &original) |x, y| {
        err += @as(f64, x - y) * @as(f64, x - y);
        ref += @as(f64, y) * @as(f64, y);
    }
    try testing.expect(@sqrt(err / ref) < 0.15);
}

test "D2: hadamardSlicesRows D=128 degenera a hadamard128Rows" {
    var a: [128 * 128]f32 = undefined;
    var b: [128 * 128]f32 = undefined;
    var rng = std.Random.Xoshiro256.init(0xD2_1004);
    for (&a) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
    @memcpy(&b, &a);
    kvarn.hadamard128Rows(&a, 128);
    kvarn.hadamardSlicesRows(&b, 128);
    for (&a, &b) |x, y| try testing.expectEqual(x, y);
}

test "D2: hadamardSlicesRows D=512 Parseval + involutiva" {
    // SLICES=4: butterfly doble (stride 1 y 2) + escala 1/2.
    const hd: usize = 512;
    var tile: [128 * hd]f32 = undefined;
    var original: [128 * hd]f32 = undefined;
    var rng = std.Random.Xoshiro256.init(0xD2_1005);
    for (&tile) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
    @memcpy(&original, &tile);

    var norm_o: f64 = 0;
    for (&original) |v| norm_o += @as(f64, v) * @as(f64, v);
    kvarn.hadamardSlicesRows(&tile, hd);
    var norm_t: f64 = 0;
    for (&tile) |v| norm_t += @as(f64, v) * @as(f64, v);
    try testing.expectApproxEqAbs(norm_o, norm_t, norm_o * 1e-4);

    kvarn.hadamardSlicesRows(&tile, hd);
    var max_diff: f32 = 0;
    for (&tile, &original) |t, o| max_diff = @max(max_diff, @abs(t - o));
    try testing.expect(max_diff < 1e-4);
}

// 9.12 (lane-b) F1: D64 CPU ref — WHT-64 Parseval + involutiva + layout
// 64×128 admitido. Espejo kvarn_wht_64_lane beellama @e1f6d6fe6.
test "9.12 F1: hadamardSlicesRows D=64 Parseval + involutiva" {
    const hd: usize = 64;
    var tile: [128 * hd]f32 = undefined;
    var original: [128 * hd]f32 = undefined;
    var rng = std.Random.Xoshiro256.init(0x912_F001);
    for (&tile) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
    @memcpy(&original, &tile);

    var norm_o: f64 = 0;
    for (&original) |v| norm_o += @as(f64, v) * @as(f64, v);
    kvarn.hadamardSlicesRows(&tile, hd);
    var norm_t: f64 = 0;
    for (&tile) |v| norm_t += @as(f64, v) * @as(f64, v);
    try testing.expectApproxEqAbs(norm_o, norm_t, norm_o * 1e-4);

    kvarn.hadamardSlicesRows(&tile, hd);
    var max_diff: f32 = 0;
    for (&tile, &original) |t, o| max_diff = @max(max_diff, @abs(t - o));
    try testing.expect(max_diff < 1e-4);
}

test "9.12 F1: KvarnRecordLayout init(64, kb, vb) admitido y alineado" {
    inline for (.{ 3, 4, 5, 6 }) |kb| {
        const lay = try kvarn.KvarnRecordLayout.init(64, kb, 4);
        try testing.expect(lay.tile_bytes % 32 == 0);
        // k_payload: 128 tokens × 64 dims packed a kb bits.
        try testing.expect(lay.k_payload_off == 0);
        try testing.expect(lay.v_payload_off > lay.k_s_row_off);
    }
}

// Internal kvarn.zig tests (D2 added there too — refAllDecls los activa).
test {
    _ = kvarn;
}

// 9.12 (lane-b) F2: D64 rect roundtrip — encode/decode K dim-major 64×128
// y V token-major 128×64, más consistencia stage↔rect transpuesto.
test "9.12 F2: K64 roundtrip rect dim-major bounded error" {
    const bits_k: u8 = 5;
    const layout = try kvarn.KvarnRecordLayout.init(64, bits_k, 4);
    var staging: [128 * 64]f32 = undefined;
    var rng = std.Random.Xoshiro256.init(0xD64_F002);
    for (&staging) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;

    // K: pre-rotación WHT-64 por token sobre el staging (pipeline manager).
    kvarn.hadamardSlicesRows(&staging, 64);
    var rect: [64 * 128]f32 = undefined;
    kvarn.d64StageToRect(&staging, true, &rect);

    var record = [_]u8{0} ** (16 * 1024); // ≥ recordBytes(64,5,4)
    try kvarn.encodeKTile64(&rect, 16, bits_k, layout, &record);

    var rect2: [64 * 128]f32 = undefined;
    try kvarn.decodeKTile64(&record, bits_k, layout, &rect2);

    var max_err: f32 = 0.0;
    for (rect2, 0..) |v, i| max_err = @max(max_err, @abs(v - rect[i]));
    try testing.expect(max_err < 0.3);
}

test "9.12 F2: V64 roundtrip rect token-major bounded error" {
    const bits_v: u8 = 4;
    const layout = try kvarn.KvarnRecordLayout.init(64, 5, bits_v);
    var staging: [128 * 64]f32 = undefined;
    var rng = std.Random.Xoshiro256.init(0xD64_F003);
    for (&staging) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;

    var rect: [128 * 64]f32 = undefined;
    kvarn.d64StageToRect(&staging, false, &rect);

    var record = [_]u8{0} ** (16 * 1024);
    try kvarn.encodeVTile64(&rect, 16, bits_v, layout, &record);

    var rect2: [128 * 64]f32 = undefined;
    try kvarn.decodeVTile64(&record, bits_v, layout, &rect2);

    var max_err: f32 = 0.0;
    for (rect2, 0..) |v, i| max_err = @max(max_err, @abs(v - rect[i]));
    try testing.expect(max_err < 0.5);
}

test "9.12 F2: d64 stage↔rect transposition involutiva" {
    var staging: [128 * 64]f32 = undefined;
    var rng = std.Random.Xoshiro256.init(0xD64_F004);
    for (&staging) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;

    var rect: [64 * 128]f32 = undefined;
    kvarn.d64StageToRect(&staging, true, &rect);
    var back: [128 * 64]f32 = undefined;
    kvarn.d64RectToStage(&rect, true, &back);
    for (back, 0..) |v, i| try testing.expect(v == staging[i]);
}

test "9.12 F2: encodeRectSide rechaza layout D128 (BitsMismatch)" {
    const layout128 = try kvarn.KvarnRecordLayout.init(128, 5, 4);
    var rect: [64 * 128]f32 = undefined;
    var record = [_]u8{0} ** (32 * 1024);
    try testing.expectError(error.BitsMismatch, kvarn.encodeKTile64(&rect, 16, 5, layout128, &record));
}
