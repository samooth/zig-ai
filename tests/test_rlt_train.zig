//! Smoke tests de entrenamiento RLT — loss baja + asserts de estabilidad.
//!
//! El test de alpha grad-check vive en test_rlt_backward.zig (con el loss
//! consistente); aquí solo el bucle de entrenamiento end-to-end.
const std = @import("std");
const testing = std.testing;
const rlt = @import("rlt_layer");

test "rlt train: loss decreases over 50 steps" {
    const allocator = testing.allocator;
    const d: usize = 16;
    const steps: usize = 50;

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
    const d_out = try allocator.alloc(f32, d);
    defer allocator.free(d_out);
    const target = try allocator.alloc(f32, d);
    defer allocator.free(target);

    // Problema de regresión ESTÁTICO: mismo e (prompt) y mismo target en
    // todos los steps. Sin feedback del estado (s←u tras cada paso cambia
    // el punto de partida), esto mide puramente que el gradiente empuja
    // los pesos hacia el target — el smoke correcto para backward+AdamW.
    for (e) |*v| v.* = 0.5;
    for (target, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.5) * 0.8;

    @memset(s, 0);

    const adam = rlt.AdamWConfig{
        .lr = 1e-2,
        .weight_decay = 0.0,
        .grad_clip = 1.0,
    };

    var first_loss: f32 = 0;
    var last_loss: f32 = 0;

    for (0..steps) |step| {
        const u = rlt.forward(e, s, &weights, &buf);

        // L = ‖u − target‖²/d (MSE)
        var loss: f32 = 0;
        for (u, target) |ui, ti| {
            const diff = ui - ti;
            loss += diff * diff;
        }
        loss /= @as(f32, @floatFromInt(d));

        if (step == 0) first_loss = loss;
        if (step == steps - 1) last_loss = loss;

        const two_over_d = 2.0 / @as(f32, @floatFromInt(d));
        for (u, target, 0..) |ui, ti, i| {
            d_out[i] = two_over_d * (ui - ti);
        }

        rlt.backward(d_out, &weights, &buf);
        rlt.adamwStep(w_gate, buf.dw_gate, buf.m_w_gate, buf.v_w_gate, adam, @intCast(step + 1), d * 2 * d);
        rlt.adamwStep(w_state, buf.dw_state, buf.m_w_state, buf.v_w_state, adam, @intCast(step + 1), d * d);

        // s←u: estado recurrente como en el engine
        @memcpy(s, u);
    }

    // MSE debe bajar al menos 2× en 50 steps con lr=1e-2
    try testing.expect(last_loss < first_loss * 0.5);
}

test "rlt train: weights stay finite (no NaN/Inf con grad_clip)" {
    const allocator = testing.allocator;
    const d: usize = 8;
    const steps: usize = 100;

    var buf = try rlt.RltBuffers.alloc(allocator, .{ .d = d });
    defer buf.deinit(allocator);

    const w_gate = try allocator.alloc(f32, d * 2 * d);
    defer allocator.free(w_gate);
    const w_state = try allocator.alloc(f32, d * d);
    defer allocator.free(w_state);

    var prng = std.Random.DefaultPrng.init(99);
    const rand = prng.random();
    // Init GRANDE a propósito — estresa el grad_clip
    for (w_gate) |*v| v.* = rand.float(f32) * 4.0 - 2.0;
    for (w_state) |*v| v.* = rand.float(f32) * 4.0 - 2.0;

    var weights = rlt.RltWeights{
        .w_gate = w_gate,
        .w_state = w_state,
        .alpha = 0.15,
    };

    const e = try allocator.alloc(f32, d);
    defer allocator.free(e);
    const s = try allocator.alloc(f32, d);
    defer allocator.free(s);
    const d_out = try allocator.alloc(f32, d);
    defer allocator.free(d_out);
    @memset(s, 0);

    const adam = rlt.AdamWConfig{ .lr = 1e-3, .weight_decay = 0.01, .grad_clip = 1.0 };

    for (0..steps) |step| {
        for (e) |*v| v.* = rand.float(f32) * 2.0 - 1.0;

        const u = rlt.forward(e, s, &weights, &buf);

        const two_over_d = 2.0 / @as(f32, @floatFromInt(d));
        for (0..d) |i| d_out[i] = two_over_d * u[i];

        rlt.backward(d_out, &weights, &buf);
        rlt.adamwStep(w_gate, buf.dw_gate, buf.m_w_gate, buf.v_w_gate, adam, @intCast(step + 1), d * 2 * d);
        rlt.adamwStep(w_state, buf.dw_state, buf.m_w_state, buf.v_w_state, adam, @intCast(step + 1), d * d);

        @memcpy(s, u);
    }

    for (w_gate) |v| try testing.expect(std.math.isFinite(v));
    for (w_state) |v| try testing.expect(std.math.isFinite(v));
    try testing.expect(std.math.isFinite(weights.alpha));
}
