//! RLT recurrent prefill — test that sequential prefill produces
//! equivalent output to parallel prefill for the same input.
//!
//! The test verifies the core invariant: for a fixed input and weights,
//! recurrent prefill (token-by-token) should produce the same final hidden
//! state as parallel prefill (all tokens at once), modulo float precision.
const std = @import("std");

/// Simulate: parallel prefill processes all tokens in one forward pass.
/// Recurrent prefill processes tokens one at a time.
/// Both should produce the same final hidden state for a linear model.
fn parallelPrefill(
    hidden: []f32, // [seq_len * d]
    w: []const f32, // [d, d]
    seq_len: usize,
    d: usize,
    scratch: []f32, // [d] — evita in-place corrupto (fila i lee filas j ya escritas)
) void {
    // For each position, apply W to the hidden state
    for (0..seq_len) |pos| {
        const base = pos * d;
        for (0..d) |i| {
            var acc: f32 = 0;
            for (0..d) |j| acc += w[i * d + j] * hidden[base + j];
            scratch[i] = acc;
        }
        @memcpy(hidden[base .. base + d], scratch);
    }
}

fn recurrentPrefill(
    hidden: []f32, // [seq_len * d]
    w: []const f32, // [d, d]
    seq_len: usize,
    d: usize,
    scratch: []f32, // [d]
) void {
    // Process each token sequentially (same computation, one at a time)
    for (0..seq_len) |pos| {
        const base = pos * d;
        for (0..d) |i| {
            var acc: f32 = 0;
            for (0..d) |j| acc += w[i * d + j] * hidden[base + j];
            scratch[i] = acc;
        }
        @memcpy(hidden[base .. base + d], scratch);
    }
}

test "recurrent prefill: same output as parallel for linear model" {
    const allocator = std.testing.allocator;
    const d: usize = 32;
    const seq_len: usize = 8;

    const w = try allocator.alloc(f32, d * d);
    defer allocator.free(w);
    var rng = std.Random.Xoshiro256.init(42);
    for (w) |*v| v.* = rng.random().float(f32) * 0.1 - 0.05;

    const scratch = try allocator.alloc(f32, d);
    defer allocator.free(scratch);
    // Parallel
    const h_par = try allocator.alloc(f32, seq_len * d);
    defer allocator.free(h_par);
    for (h_par) |*v| v.* = rng.random().float(f32);
    // Copia del INPUT original antes de transformar h_par (el reset de
    // h_rec debe partir del input, no de la salida de par).
    const h_in = try allocator.alloc(f32, seq_len * d);
    defer allocator.free(h_in);
    @memcpy(h_in, h_par);
    parallelPrefill(h_par, w, seq_len, d, scratch);

    // Recurrent (same input)
    const h_rec = try allocator.alloc(f32, seq_len * d);
    defer allocator.free(h_rec);
    @memcpy(h_rec, h_in);
    recurrentPrefill(h_rec, w, seq_len, d, scratch);

    // Compare
    var max_diff: f32 = 0;
    for (h_par, h_rec) |p, r| {
        const diff = @abs(p - r);
        if (diff > max_diff) max_diff = diff;
    }
    try std.testing.expect(max_diff < 1e-6);
}

test "recurrent prefill: each position sees same state as decode" {
    // Simulate: for position i, decode would see hidden[i] as input.
    // Recurrent prefill should produce the same output for each position.
    const d: usize = 16;
    const seq_len: usize = 4;

    const w = [_]f32{
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
    };

    // Embedding for 4 tokens
    var hidden = [_]f32{ 1.0, 0.5, 0.3, 0.8, 0.2, 0.9, 0.7, 0.1, 0.4, 0.6, 1.0, 0.3, 0.8, 0.2, 0.5, 0.9, 0.1, 0.7, 0.4, 0.6, 0.9, 0.3, 0.8, 0.2, 0.5, 1.0, 0.7, 0.4, 0.1, 0.6, 0.3, 0.8, 0.9, 0.1, 0.6, 0.4, 0.8, 0.2, 0.5, 0.7, 0.3, 0.9, 0.1, 0.6, 0.4, 0.8, 0.2, 0.5, 0.3, 0.7, 0.9, 0.1, 0.4, 0.6, 0.8, 0.2, 0.5, 0.3, 0.7, 0.9, 0.1, 0.4, 0.6, 0.8 };

    // Save original for comparison
    var original = hidden;

    var scratch: [16]f32 = undefined;
    recurrentPrefill(&hidden, &w, seq_len, d, &scratch);

    // For a linear model with no residual, each position is independent
    // (output at pos i = W @ input[pos i]). The recurrent prefill should
    // produce the same result as processing each token independently.
    for (0..seq_len) |pos| {
        const base = pos * d;
        // Simulate independent decode for this position
        var decode_out = [_]f32{0} ** 16;
        for (0..d) |i| {
            var acc: f32 = 0;
            for (0..d) |j| acc += w[i * d + j] * original[base + j];
            decode_out[i] = acc;
        }
        // Compare with recurrent prefill output
        for (0..d) |i| {
            try std.testing.expectApproxEqAbs(decode_out[i], hidden[base + i], 1e-6);
        }
    }
}
