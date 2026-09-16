//! Gradient check tests — finite-difference vs analytic backward del merge RLT.
//!
//! Contrato: el loss de referencia es SIEMPRE L = ‖u‖²/d (misma fórmula en
//! el gradiente analítico dL/du = 2u/d y en el numérico por diferencias
//! centrales). Si cambias el loss aquí, cámbialo en ambos sitios.
const std = @import("std");
const testing = std.testing;
const rlt = @import("rlt_layer");

/// L = ‖u‖²/d — la única definición de loss de estos tests.
fn energyLoss(u: []const f32) f32 {
    var acc: f32 = 0;
    for (u) |v| acc += v * v;
    return acc / @as(f32, @floatFromInt(u.len));
}

/// Gradiente numérico central de L w.r.t. un elemento de un slice de pesos.
fn numGradWeight(
    w_arr: []f32,
    idx: usize,
    eps: f32,
    e: []const f32,
    s: []const f32,
    weights: *rlt.RltWeights,
    buf: *rlt.RltBuffers,
) f32 {
    const orig = w_arr[idx];
    w_arr[idx] = orig + eps;
    const lp = energyLoss(rlt.forward(e, s, weights, buf));
    w_arr[idx] = orig - eps;
    const lm = energyLoss(rlt.forward(e, s, weights, buf));
    w_arr[idx] = orig;
    return (lp - lm) / (2.0 * eps);
}

test "rlt backward: alpha gradient matches finite-diff" {
    const allocator = testing.allocator;
    const d: usize = 16;

    var buf = try rlt.RltBuffers.alloc(allocator, .{ .d = d });
    defer buf.deinit(allocator);

    const w_gate = try allocator.alloc(f32, d * 2 * d);
    defer allocator.free(w_gate);
    const w_state = try allocator.alloc(f32, d * d);
    defer allocator.free(w_state);

    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();
    for (w_gate) |*v| v.* = rand.float(f32) * 0.2 - 0.1;
    for (w_state) |*v| v.* = rand.float(f32) * 0.2 - 0.1;

    var weights = rlt.RltWeights{
        .w_gate = w_gate,
        .w_state = w_state,
        .alpha = 0.15,
    };

    const e = try allocator.alloc(f32, d);
    defer allocator.free(e);
    const s = try allocator.alloc(f32, d);
    defer allocator.free(s);
    for (e) |*v| v.* = rand.float(f32) * 2.0 - 1.0;
    for (s) |*v| v.* = rand.float(f32) * 2.0 - 1.0;

    const u = rlt.forward(e, s, &weights, &buf);
    const d_out = try allocator.alloc(f32, d);
    defer allocator.free(d_out);
    const two_over_d = 2.0 / @as(f32, @floatFromInt(d));
    for (0..d) |i| d_out[i] = two_over_d * u[i];
    rlt.backward(d_out, &weights, &buf);

    const analytic = buf.d_alpha;

    const eps: f32 = 1e-4;
    const orig_alpha = weights.alpha;
    weights.alpha = orig_alpha + eps;
    const lp = energyLoss(rlt.forward(e, s, &weights, &buf));
    weights.alpha = orig_alpha - eps;
    const lm = energyLoss(rlt.forward(e, s, &weights, &buf));
    weights.alpha = orig_alpha;

    const numerical = (lp - lm) / (2.0 * eps);
    const diff = @abs(analytic - numerical);
    try testing.expect(diff < 1e-3);
}

test "rlt backward: W_gate gradient matches finite-diff (sampled)" {
    const allocator = testing.allocator;
    const d: usize = 8;

    var buf = try rlt.RltBuffers.alloc(allocator, .{ .d = d });
    defer buf.deinit(allocator);

    const w_gate = try allocator.alloc(f32, d * 2 * d);
    defer allocator.free(w_gate);
    const w_state = try allocator.alloc(f32, d * d);
    defer allocator.free(w_state);

    var prng = std.Random.DefaultPrng.init(123);
    const rand = prng.random();
    for (w_gate) |*v| v.* = rand.float(f32) * 0.2 - 0.1;
    for (w_state) |*v| v.* = rand.float(f32) * 0.2 - 0.1;

    var weights = rlt.RltWeights{
        .w_gate = w_gate,
        .w_state = w_state,
        .alpha = 0.1,
    };

    const e = try allocator.alloc(f32, d);
    defer allocator.free(e);
    const s = try allocator.alloc(f32, d);
    defer allocator.free(s);
    for (e) |*v| v.* = rand.float(f32) * 2.0 - 1.0;
    for (s) |*v| v.* = rand.float(f32) * 2.0 - 1.0;

    const u = rlt.forward(e, s, &weights, &buf);
    const d_out = try allocator.alloc(f32, d);
    defer allocator.free(d_out);
    const two_over_d = 2.0 / @as(f32, @floatFromInt(d));
    for (0..d) |i| d_out[i] = two_over_d * u[i];
    rlt.backward(d_out, &weights, &buf);

    // Muestreo determinista: primera fila, mitad, y última posición absoluta.
    const indices = [_]usize{ 0, d * 2 * d / 3, d * 2 * d - 1 };
    const eps: f32 = 1e-4;

    for (indices) |idx| {
        const analytic = buf.dw_gate[idx];
        const numerical = numGradWeight(w_gate, idx, eps, e, s, &weights, &buf);
        const diff = @abs(analytic - numerical);
        try testing.expect(diff < 1e-3);
    }
}

test "rlt backward: W_state gradient matches finite-diff (sampled)" {
    const allocator = testing.allocator;
    const d: usize = 8;

    var buf = try rlt.RltBuffers.alloc(allocator, .{ .d = d });
    defer buf.deinit(allocator);

    const w_gate = try allocator.alloc(f32, d * 2 * d);
    defer allocator.free(w_gate);
    const w_state = try allocator.alloc(f32, d * d);
    defer allocator.free(w_state);

    var prng = std.Random.DefaultPrng.init(777);
    const rand = prng.random();
    for (w_gate) |*v| v.* = rand.float(f32) * 0.2 - 0.1;
    for (w_state) |*v| v.* = rand.float(f32) * 0.2 - 0.1;

    var weights = rlt.RltWeights{
        .w_gate = w_gate,
        .w_state = w_state,
        .alpha = 0.1,
    };

    const e = try allocator.alloc(f32, d);
    defer allocator.free(e);
    const s = try allocator.alloc(f32, d);
    defer allocator.free(s);
    for (e) |*v| v.* = rand.float(f32) * 2.0 - 1.0;
    for (s) |*v| v.* = rand.float(f32) * 2.0 - 1.0;

    const u = rlt.forward(e, s, &weights, &buf);
    const d_out = try allocator.alloc(f32, d);
    defer allocator.free(d_out);
    const two_over_d = 2.0 / @as(f32, @floatFromInt(d));
    for (0..d) |i| d_out[i] = two_over_d * u[i];
    rlt.backward(d_out, &weights, &buf);

    const indices = [_]usize{ 0, d * d / 2, d * d - 1 };
    const eps: f32 = 1e-4;

    for (indices) |idx| {
        const analytic = buf.dw_state[idx];
        const numerical = numGradWeight(w_state, idx, eps, e, s, &weights, &buf);
        const diff = @abs(analytic - numerical);
        try testing.expect(diff < 1e-3);
    }
}

test "rlt backward: zero gradient when e = 0 and s = 0" {
    const allocator = testing.allocator;
    const d: usize = 16;

    var buf = try rlt.RltBuffers.alloc(allocator, .{ .d = d });
    defer buf.deinit(allocator);

    const w_gate = try allocator.alloc(f32, d * 2 * d);
    defer allocator.free(w_gate);
    const w_state = try allocator.alloc(f32, d * d);
    defer allocator.free(w_state);
    @memset(w_gate, 0);
    @memset(w_state, 0);

    var weights = rlt.RltWeights{
        .w_gate = w_gate,
        .w_state = w_state,
        .alpha = 0.1,
    };

    const e = try allocator.alloc(f32, d);
    defer allocator.free(e);
    const s = try allocator.alloc(f32, d);
    defer allocator.free(s);
    @memset(e, 0);
    @memset(s, 0);

    const u = rlt.forward(e, s, &weights, &buf);
    for (u) |v| try testing.expectApproxEqAbs(@as(f32, 0), v, 1e-6);

    const d_out = try allocator.alloc(f32, d);
    defer allocator.free(d_out);
    @memset(d_out, 0);
    rlt.backward(d_out, &weights, &buf);

    for (buf.dw_gate) |v| try testing.expectApproxEqAbs(@as(f32, 0), v, 1e-6);
    for (buf.dw_state) |v| try testing.expectApproxEqAbs(@as(f32, 0), v, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0), buf.d_alpha, 1e-6);
}

test "rlt backward: gate saturates correctly" {
    const allocator = testing.allocator;
    const d: usize = 8;

    var buf = try rlt.RltBuffers.alloc(allocator, .{ .d = d });
    defer buf.deinit(allocator);

    const w_gate = try allocator.alloc(f32, d * 2 * d);
    defer allocator.free(w_gate);
    const w_state = try allocator.alloc(f32, d * d);
    defer allocator.free(w_state);
    @memset(w_gate, 10.0);
    @memset(w_state, 0.1);

    var weights = rlt.RltWeights{
        .w_gate = w_gate,
        .w_state = w_state,
        .alpha = 0.15,
    };

    const e = try allocator.alloc(f32, d);
    defer allocator.free(e);
    const s = try allocator.alloc(f32, d);
    defer allocator.free(s);
    @memset(e, 0.5);
    @memset(s, 0.5);

    _ = rlt.forward(e, s, &weights, &buf);

    for (buf.gate) |g| {
        try testing.expect(g > 0.99);
    }

    const d_out = try allocator.alloc(f32, d);
    defer allocator.free(d_out);
    @memset(d_out, 1.0);
    rlt.backward(d_out, &weights, &buf);

    // Sigmoid saturada ⇒ dL/d(gate_logits) ≈ 0 ⇒ gradiente de W_gate ≈ 0
    for (buf.dw_gate) |v| {
        try testing.expect(!std.math.isInf(v));
        try testing.expect(!std.math.isNan(v));
        try testing.expectApproxEqAbs(@as(f32, 0), v, 1e-3);
    }
    for (buf.dw_state) |v| {
        try testing.expect(!std.math.isInf(v));
        try testing.expect(!std.math.isNan(v));
    }
    try testing.expect(!std.math.isInf(buf.d_alpha));
    try testing.expect(!std.math.isNan(buf.d_alpha));
}

test "rlt backward: stride W_gate no corrompe memoria (regresión d2d bug)" {
    // Regresión del bug original: backward usaba d2d = d*2*d (=2d²) como
    // stride de fila en vez de 2*d — overflow de índice en dw_gate/w_gate
    // para todo d > 1. Este test panica (index out of bounds) si reaparece.
    const allocator = testing.allocator;
    const d: usize = 32; // d grande para asegurar que el overflow se manifeste

    var buf = try rlt.RltBuffers.alloc(allocator, .{ .d = d });
    defer buf.deinit(allocator);

    const w_gate = try allocator.alloc(f32, d * 2 * d);
    defer allocator.free(w_gate);
    const w_state = try allocator.alloc(f32, d * d);
    defer allocator.free(w_state);

    var prng = std.Random.DefaultPrng.init(7);
    const rand = prng.random();
    for (w_gate) |*v| v.* = rand.float(f32) * 0.2 - 0.1;
    for (w_state) |*v| v.* = rand.float(f32) * 0.2 - 0.1;

    var weights = rlt.RltWeights{ .w_gate = w_gate, .w_state = w_state, .alpha = 0.2 };

    const e = try allocator.alloc(f32, d);
    defer allocator.free(e);
    const s = try allocator.alloc(f32, d);
    defer allocator.free(s);
    for (e) |*v| v.* = rand.float(f32) * 2.0 - 1.0;
    for (s) |*v| v.* = rand.float(f32) * 2.0 - 1.0;

    _ = rlt.forward(e, s, &weights, &buf);
    const d_out = try allocator.alloc(f32, d);
    defer allocator.free(d_out);
    for (0..d) |i| d_out[i] = 1.0;
    rlt.backward(d_out, &weights, &buf);

    // Todos los gradientes finitos
    for (buf.dw_gate) |v| try testing.expect(std.math.isFinite(v));
    for (buf.dw_state) |v| try testing.expect(std.math.isFinite(v));
}
