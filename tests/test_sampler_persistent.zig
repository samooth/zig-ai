//! Tests unitarios de `pipeline.Sampler` con buffers persistentes.
const std = @import("std");

const pipeline = @import("pipeline");

test "sampler: initScratch y deinitScratch con vocab=256" {
    const gpa = std.testing.allocator;
    var s = pipeline.Sampler{};
    try s.initScratch(gpa, 256);
    defer s.deinitScratch(gpa);
    try std.testing.expectEqual(@as(usize, 256), s.work_buf.len);
    try std.testing.expectEqual(@as(usize, 256), s.idx_buf.len);
    try std.testing.expectEqual(@as(usize, 256), s.vals_buf.len);
    try std.testing.expectEqual(@as(usize, 256), s.probs_buf.len);
    s.deinitScratch(gpa);
    try std.testing.expectEqual(@as(usize, 0), s.work_buf.len);
    try std.testing.expectEqual(@as(usize, 0), s.idx_buf.len);
    try std.testing.expectEqual(@as(usize, 0), s.vals_buf.len);
    try std.testing.expectEqual(@as(usize, 0), s.probs_buf.len);
}

test "sampler: deterministic con seed fija y temperatura=0" {
    const gpa = std.testing.allocator;
    var s = pipeline.Sampler{ .temperature = 0 };
    try s.initScratch(gpa, 256);
    defer s.deinitScratch(gpa);

    var rng = std.Random.Xoshiro256.init(1);
    const logits = blk: {
        var ls: [256]f32 = undefined;
        for (&ls) |*v| v.* = 0.1;
        ls[42] = 5.0;
        break :blk ls;
    };

    const tok1 = s.sample(logits[0..], &rng, &[_]u32{});
    const tok2 = s.sample(logits[0..], &rng, &[_]u32{});
    try std.testing.expect(tok1 == tok2);
    try std.testing.expect(tok1 < 256);
}

test "sampler: temperatura=1.0 + top_k=10 devuelve índices válidos" {
    const gpa = std.testing.allocator;
    var s = pipeline.Sampler{ .temperature = 1.0, .top_k = 10 };
    try s.initScratch(gpa, 256);
    defer s.deinitScratch(gpa);

    var rng = std.Random.Xoshiro256.init(7);
    const logits = blk: {
        var ls: [256]f32 = undefined;
        for (&ls) |*v| v.* = 1.0;
        ls[10] = 10.0;
        ls[20] = 9.0;
        ls[30] = 8.0;
        break :blk ls;
    };

    const tok = s.sample(logits[0..], &rng, &[_]u32{});
    try std.testing.expect(tok < 256);
}

test "sampler: temperatura=0.8 + top_p=0.9 produce índices válidos" {
    const gpa = std.testing.allocator;
    var s = pipeline.Sampler{ .temperature = 0.8, .top_p = 0.9 };
    try s.initScratch(gpa, 256);
    defer s.deinitScratch(gpa);

    var rng = std.Random.Xoshiro256.init(13);
    const logits = blk: {
        var ls: [256]f32 = undefined;
        for (&ls) |*v| v.* = 0.5;
        ls[5] = 2.0;
        ls[6] = 1.5;
        ls[7] = 1.0;
        break :blk ls;
    };

    const tok = s.sample(logits[0..], &rng, &[_]u32{});
    try std.testing.expect(tok < 256);
}

test "sampler: repetition_penalty=1.2 con seed es determinista" {
    const gpa = std.testing.allocator;
    var s = pipeline.Sampler{ .temperature = 0.5, .repetition_penalty = 1.2 };
    try s.initScratch(gpa, 256);
    defer s.deinitScratch(gpa);

    var rng_a = std.Random.Xoshiro256.init(99);
    var rng_b = std.Random.Xoshiro256.init(99);

    const logits = blk: {
        var ls: [256]f32 = undefined;
        for (&ls) |*v| v.* = 1.0;
        ls[1] = 3.0;
        ls[2] = 2.0;
        break :blk ls;
    };

    const history = &[_]u32{1};
    const tok_a = s.sample(logits[0..], &rng_a, history);
    const tok_b = s.sample(logits[0..], &rng_b, history);
    try std.testing.expect(tok_a == tok_b);
    try std.testing.expect(tok_a < 256);
}
