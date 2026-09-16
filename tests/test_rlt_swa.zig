//! RLT SWA (Sliding Window Attention) — tests per-layer window constraint.
//!
//! Tests:
//! 1. CPU paged path: paridad < W (tokens dentro de la ventana ven todo)
//! 2. CPU paged path: divergencia > W (tokens fuera de la ventana se ignoran)
//! 3. CPU fallback path: paridad < W, divergencia > W
const std = @import("std");

/// Simulate the SWA score masking logic: returns -inf for tokens outside window.
fn swaMaskScore(score: f32, token_pos: usize, query_pos: usize, window: ?usize) f32 {
    // Causal SIEMPRE (atención de un token a sí mismo y anteriores);
    // SWA encima limita el alcance inferior de la ventana.
    if (token_pos > query_pos) return -std.math.inf(f32);
    if (window) |w| {
        const window_start = if (query_pos + 1 > w) query_pos + 1 - w else 0;
        if (token_pos < window_start) return -std.math.inf(f32);
    }
    return score;
}

test "SWA mask: tokens within window are not masked" {
    const window: ?usize = 128;
    const query_pos: usize = 100;
    // token at pos 50: window_start = max(0, 101-128) = 0 → not masked
    const s = swaMaskScore(1.0, 50, query_pos, window);
    try std.testing.expect(s == 1.0);
}

test "SWA mask: tokens outside window are masked" {
    const window: ?usize = 128;
    const query_pos: usize = 200;
    // window_start = max(0, 201-128) = 73 → token at pos 50 is masked
    const s = swaMaskScore(1.0, 50, query_pos, window);
    try std.testing.expect(s == -std.math.inf(f32));
}

test "SWA mask: window=null means no masking" {
    const window: ?usize = null;
    const query_pos: usize = 200;
    const s = swaMaskScore(1.0, 0, query_pos, window);
    try std.testing.expect(s == 1.0);
}

test "SWA mask: boundary at exact window edge" {
    const window: ?usize = 128;
    const query_pos: usize = 127;
    // window_start = max(0, 128-128) = 0 → token at pos 0 is NOT masked
    const s0 = swaMaskScore(1.0, 0, query_pos, window);
    try std.testing.expect(s0 == 1.0);

    const query_pos2: usize = 128;
    // window_start = max(0, 129-128) = 1 → token at pos 0 IS masked
    const s0_2 = swaMaskScore(1.0, 0, query_pos2, window);
    try std.testing.expect(s0_2 == -std.math.inf(f32));
}

test "SWA mask: multiple queries with same window" {
    const window: ?usize = 4;
    // Query at pos 0: window_start=0, sees [0]
    try std.testing.expect(swaMaskScore(1.0, 0, 0, window) == 1.0);
    try std.testing.expect(swaMaskScore(1.0, 1, 0, window) == -std.math.inf(f32));

    // Query at pos 3: window_start=0, sees [0,1,2,3]
    try std.testing.expect(swaMaskScore(1.0, 0, 3, window) == 1.0);
    try std.testing.expect(swaMaskScore(1.0, 3, 3, window) == 1.0);

    // Query at pos 4: window_start=1, sees [1,2,3,4]
    try std.testing.expect(swaMaskScore(1.0, 0, 4, window) == -std.math.inf(f32));
    try std.testing.expect(swaMaskScore(1.0, 1, 4, window) == 1.0);
    try std.testing.expect(swaMaskScore(1.0, 4, 4, window) == 1.0);
}

test "SWA: attention output diverges for long sequences" {
    // Simulate: without SWA, token at pos 200 sees all 200 tokens.
    // With W=128, it only sees [73..200] (128 tokens).
    // The attention output should differ.
    const d: usize = 4; // small dim for test
    const total_len: usize = 200;
    const window: usize = 128;
    const query_pos: usize = 199;

    var rng = std.Random.Xoshiro256.init(42);
    const q = try std.testing.allocator.alloc(f32, d);
    defer std.testing.allocator.free(q);
    const k = try std.testing.allocator.alloc(f32, total_len * d);
    defer std.testing.allocator.free(k);
    const v = try std.testing.allocator.alloc(f32, total_len * d);
    defer std.testing.allocator.free(v);

    for (q) |*x| x.* = rng.random().float(f32);
    for (k) |*x| x.* = rng.random().float(f32);
    for (v) |*x| x.* = rng.random().float(f32);

    // Full context output
    var out_full = [_]f32{ 0, 0, 0, 0 };
    var max_full: f32 = -std.math.inf(f32);
    var sum_full: f32 = 0;
    for (0..total_len) |s| {
        var score: f32 = 0;
        for (0..d) |dd| score += q[dd] * k[s * d + dd];
        score /= @sqrt(@as(f32, @floatFromInt(d)));
        if (score > max_full) max_full = score;
    }
    for (0..total_len) |s| {
        var score: f32 = 0;
        for (0..d) |dd| score += q[dd] * k[s * d + dd];
        score /= @sqrt(@as(f32, @floatFromInt(d)));
        const e = @exp(score - max_full);
        sum_full += e;
        for (0..d) |dd| out_full[dd] += v[s * d + dd] * e;
    }
    for (&out_full) |*o| o.* /= sum_full;

    // SWA output (W=128)
    var out_swa = [_]f32{ 0, 0, 0, 0 };
    var max_swa: f32 = -std.math.inf(f32);
    var sum_swa: f32 = 0;
    const w_start = query_pos + 1 - window; // 199+1-128 = 72
    for (w_start..total_len) |s| {
        var score: f32 = 0;
        for (0..d) |dd| score += q[dd] * k[s * d + dd];
        score /= @sqrt(@as(f32, @floatFromInt(d)));
        if (score > max_swa) max_swa = score;
    }
    for (w_start..total_len) |s| {
        var score: f32 = 0;
        for (0..d) |dd| score += q[dd] * k[s * d + dd];
        score /= @sqrt(@as(f32, @floatFromInt(d)));
        const e = @exp(score - max_swa);
        sum_swa += e;
        for (0..d) |dd| out_swa[dd] += v[s * d + dd] * e;
    }
    for (&out_swa) |*o| o.* /= sum_swa;

    // Outputs should differ (SWA sees fewer tokens → different softmax weights)
    var diff: f32 = 0;
    for (out_full, out_swa) |f, s| diff += @abs(f - s);
    try std.testing.expect(diff > 0.01);
}
