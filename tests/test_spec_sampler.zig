//! Tests unitarios del sampler especulativo (lane-c C2).
//! Puros CPU: greedy exacto, rejection determinista por seed, propiedades.
const std = @import("std");
const spec = @import("speculative");

test "sampler greedy exacto" {
    try std.testing.expectEqual(@as(u32, 3), spec.sampler.greedy(&.{ 0.1, -2.0, 0.5, 5.0 }));
    // Empate: gana el primero.
    try std.testing.expectEqual(@as(u32, 1), spec.sampler.greedy(&.{ -1.0, 4.0, 4.0 }));
}

test "sampler rejection consistente con seed" {
    const target = [_]f32{ 4.0, 1.0, 0.2, 0.1 };
    const draft = [_]f32{ 0.5, 0.4, 0.3, 2.0 };
    var sp: [4]f32 = undefined;
    var sq: [4]f32 = undefined;

    var seq_a: [32]u32 = undefined;
    var seq_b: [32]u32 = undefined;
    {
        var prng = std.Random.DefaultPrng.init(42);
        for (&seq_a) |*tok| {
            tok.* = (try spec.sampler.rejectionStep(&target, &draft, &sp, &sq, prng.random())).token;
        }
    }
    {
        var prng = std.Random.DefaultPrng.init(42);
        for (&seq_b) |*tok| {
            tok.* = (try spec.sampler.rejectionStep(&target, &draft, &sp, &sq, prng.random())).token;
        }
    }
    try std.testing.expectEqualSlices(u32, &seq_a, &seq_b);
}

test "sampler rejection filas idénticas => acepta siempre" {
    var prng = std.Random.DefaultPrng.init(1);
    const logits = [_]f32{ 2.0, 1.0, 0.0 };
    var sp: [3]f32 = undefined;
    var sq: [3]f32 = undefined;
    for (0..16) |_| {
        const r = try spec.sampler.rejectionStep(&logits, &logits, &sp, &sq, prng.random());
        try std.testing.expect(r.accepted);
        try std.testing.expectEqual(@as(u32, 0), r.token);
    }
}

test "SpecDriver métricas de aceptación" {
    const allocator = std.testing.allocator;
    var drv = spec.SpecDriver.init(allocator, .{});
    defer drv.deinit();

    // Target argmax por posición: {1, 0}; drafts {1, 1} → acepta 1º, rechaza 2º.
    const row0 = [_]f32{ 0.0, 9.0 };
    const row1 = [_]f32{ 9.0, 1.0 };
    const rows = [_][]const f32{ &row0, &row1 };
    var out: [8]u32 = undefined;

    const res = drv.verifyGreedyRows(&.{ 1, 1 }, &rows, &out);
    try std.testing.expectEqual(@as(usize, 1), res.n_accepted); // sólo pos 0
    try std.testing.expectEqual(@as(usize, 2), res.n_total); // + bonus argmax(row1)=0
    try std.testing.expectEqual(@as(f32, 0.5), drv.metrics.acceptanceRate());

    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    drv.reportMetrics(&w) catch {};
}

test "sampler softmaxConfidence: one-hot=1, uniforme≈1/V" {
    const one_hot = [_]f32{ 0, 0, 50.0, 0 };
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), spec.sampler.softmaxConfidence(&one_hot, 2), 1e-6);
    const uni = [_]f32{ 0, 0, 0, 0 };
    const c = spec.sampler.softmaxConfidence(&uni, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), c, 1e-6);
}

test "sampler topK: orden descendente sin tocar el vocabulario completo" {
    var idx: [3]u32 = undefined;
    var val: [3]f32 = undefined;
    spec.sampler.topK(&.{ 1.0, 9.0, 3.0, 7.0, 5.0 }, &idx, &val);
    try std.testing.expectEqualSlices(u32, &.{ 1, 3, 4 }, &idx);
    try std.testing.expectEqual(@as(f32, 9.0), val[0]);
}
