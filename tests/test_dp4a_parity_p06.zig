//! P0-6 dp4a parity: q4_1, q2_k, iq4_nl, iq3_xxs, iq2_xxs, iq2_xs vs scalar CPU ref.
const std = @import("std");
const cudaz = @import("cudaz");
const layer_kernels = @import("layer_kernels");
const kv_quant = @import("kv_quant");
const pa = @import("paged_attention");

fn dp4aParityTest(allocator: std.mem.Allocator, qtype: u32, fmt: pa.QuantFormat, K: usize, N: usize) !void {
    try cudaz.ensureContext();
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);
    var lk = try layer_kernels.LayerKernels.init(stream);
    defer lk.deinit();

    const row_bytes = kv_quant.quantBytesRaw(fmt, K);
    const w_bytes = try allocator.alloc(u8, N * row_bytes);
    defer allocator.free(w_bytes);
    var rng = std.Random.Xoshiro256.init(7031);
    const row = try allocator.alloc(f16, K);
    defer allocator.free(row);
    for (0..N) |j| {
        for (row) |*v| v.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);
        const enc = try kv_quant.encodeToOwned(allocator, fmt, row);
        defer allocator.free(enc);
        @memcpy(w_bytes[j * row_bytes ..][0..row_bytes], enc[0..row_bytes]);
    }

    const a1 = try allocator.alloc(f32, K);
    defer allocator.free(a1);
    var rng2 = std.Random.Xoshiro256.init(7032);
    for (a1) |*v| v.* = @floatCast(rng2.random().float(f32) * 2.0 - 1.0);

    const d_a = try cudaz.cuMemAlloc(K * @sizeOf(f32));
    defer cudaz.cuMemFree(d_a);
    const d_w = try cudaz.cuMemAlloc(w_bytes.len);
    defer cudaz.cuMemFree(d_w);
    try cudaz.cuMemcpyHtoD(d_a, @intFromPtr(a1.ptr), K * @sizeOf(f32));
    try cudaz.cuMemcpyHtoD(d_w, @intFromPtr(w_bytes.ptr), w_bytes.len);
    const d_c_dp4a = try cudaz.cuMemAlloc(N * @sizeOf(f32));
    defer cudaz.cuMemFree(d_c_dp4a);
    const d_c_ref = try cudaz.cuMemAlloc(N * @sizeOf(f32));
    defer cudaz.cuMemFree(d_c_ref);

    switch (qtype) {
        1 => try lk.q41GemmM1Dp4a(d_a, d_w, d_c_dp4a, K, N),
        7 => try lk.q2kGemmM1Dp4a(d_a, d_w, d_c_dp4a, K, N),
        10 => try lk.iq4nlGemmM1Dp4a(d_a, d_w, d_c_dp4a, K, N),
        12 => try lk.iq3xxsGemmM1Dp4a(d_a, d_w, d_c_dp4a, K, N),
        13 => try lk.iq2xxsGemmM1Dp4a(d_a, d_w, d_c_dp4a, K, N),
        14 => try lk.iq2xsGemmM1Dp4a(d_a, d_w, d_c_dp4a, K, N),
        else => unreachable,
    }
    try lk.qgemm(d_a, d_w, d_c_ref, 1, K, N, qtype);
    try cudaz.cuStreamSynchronize(stream);

    const c_ref = try allocator.alloc(f32, N);
    defer allocator.free(c_ref);
    const c_dp4a = try allocator.alloc(f32, N);
    defer allocator.free(c_dp4a);
    try cudaz.cuMemcpyDtoH(@intFromPtr(c_ref.ptr), d_c_ref, N * @sizeOf(f32));
    try cudaz.cuMemcpyDtoH(@intFromPtr(c_dp4a.ptr), d_c_dp4a, N * @sizeOf(f32));

    const w_ref = try allocator.alloc(f32, N * K);
    defer allocator.free(w_ref);
    {
        const deq = kv_quant.dequant;
        for (0..N) |j| deq(fmt, w_bytes[j * row_bytes ..][0..row_bytes], w_ref[j * K ..][0..K]);
    }
    const aq_scale = try allocator.alloc(f32, K / 32);
    defer allocator.free(aq_scale);
    const aq_val = try allocator.alloc(i32, K);
    defer allocator.free(aq_val);
    for (0..(K / 32)) |kb| {
        var amax: f32 = 0;
        for (0..32) |r| amax = @max(amax, @abs(a1[kb * 32 + r]));
        const dd: f32 = if (amax > 0) amax / 127.0 else 1.0;
        aq_scale[kb] = dd;
        for (0..32) |r| {
            const q: i32 = @max(-127, @min(127, @as(i32, @intFromFloat(@round(a1[kb * 32 + r] / dd)))));
            aq_val[kb * 32 + r] = q;
        }
    }

    var bad: usize = 0;
    var max_rel: f32 = 0;
    var max_abs: f32 = 0;
    for (0..N) |j| {
        var dot_q8: f32 = 0;
        for (0..(K / 32)) |kb| {
            var sacc: f32 = 0;
            for (0..32) |r| sacc += @as(f32, @floatFromInt(aq_val[kb * 32 + r])) * w_ref[j * K + kb * 32 + r];
            dot_q8 += sacc * aq_scale[kb];
        }
        const abs_diff = @abs(c_dp4a[j] - dot_q8);
        const rel = abs_diff / @max(@abs(dot_q8), 1.0);
        max_rel = @max(max_rel, rel);
        max_abs = @max(max_abs, abs_diff);
        if (rel > 1e-2) bad += 1;
    }
    const tag = switch (qtype) {
        1 => "q4_1",
        7 => "q2_k",
        10 => "iq4_nl",
        12 => "iq3_xxs",
        13 => "iq2_xxs",
        14 => "iq2_xs",
        else => "?",
    };
    std.debug.print("{s} dp4a M=1 k={d} n={d}: bad={d}/{d} max_rel={e} max_abs={e}\n", .{ tag, K, N, bad, N, max_rel, max_abs });
    if (bad > 0) return error.Dp4aParityFailP06;
}

test "dev-IQ P0-6 q4_1 M=1 dp4a: paridad bit-exacta vs q8_1 CPU" {
    const allocator = std.testing.allocator;
    try dp4aParityTest(allocator, 1, .q4_1, 8192, 3584);
}
test "dev-IQ P0-6 q2_k M=1 dp4a: paridad bit-exacta vs q8_1 CPU" {
    const allocator = std.testing.allocator;
    try dp4aParityTest(allocator, 7, .q2_k, 8192, 8192);
}
test "dev-IQ P0-6 iq4_nl M=1 dp4a: paridad bit-exacta vs q8_1 CPU" {
    const allocator = std.testing.allocator;
    try dp4aParityTest(allocator, 10, .iq4_nl, 8192, 3584);
}
test "dev-IQ P0-6 iq3_xxs M=1 dp4a: paridad bit-exacta vs q8_1 CPU" {
    const allocator = std.testing.allocator;
    try dp4aParityTest(allocator, 12, .iq3_xxs, 8192, 3584);
}
test "dev-IQ P0-6 iq2_xxs M=1 dp4a: paridad bit-exacta vs q8_1 CPU" {
    const allocator = std.testing.allocator;
    try dp4aParityTest(allocator, 13, .iq2_xxs, 8192, 3584);
}
test "dev-IQ P0-6 iq2_xs M=1 dp4a: paridad bit-exacta vs q8_1 CPU" {
    const allocator = std.testing.allocator;
    try dp4aParityTest(allocator, 14, .iq2_xs, 8192, 3584);
}
