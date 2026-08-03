const std = @import("std");
const Tensor = @import("core").Tensor;
const fa = @import("fa");

const FlashAttentionConfig = fa.fa_config.FlashAttentionConfig;
const FlashAttentionCpu = fa.FlashAttentionCpu;

test "cpu flash attention forward" {
    const allocator = std.testing.allocator;
    const config = FlashAttentionConfig{
        .N = 16, .d = 64, .num_heads = 2, .batch_size = 1,
        .dtype = .f16, .causal = true,
    };

    var fa_cpu = FlashAttentionCpu.init(allocator, config);

    var Q = try Tensor(f16).alloc(allocator, &[_]usize{ 1, 2, 16, 64 });
    defer Q.deinit();
    var K = try Tensor(f16).alloc(allocator, &[_]usize{ 1, 2, 16, 64 });
    defer K.deinit();
    var V = try Tensor(f16).alloc(allocator, &[_]usize{ 1, 2, 16, 64 });
    defer V.deinit();
    var O = try Tensor(f16).alloc(allocator, &[_]usize{ 1, 2, 16, 64 });
    defer O.deinit();

    fa.fa_utils.initUniform(&Q, -0.1, 0.1, 42);
    fa.fa_utils.initUniform(&K, -0.1, 0.1, 43);
    fa.fa_utils.initUniform(&V, -0.1, 0.1, 44);

    try fa_cpu.forward(Q, K, V, &O);

    // Verificar que la salida no es NaN y tiene valores razonables
    var it = O.iterator();
    while (it.next()) |p| {
        const v = fa.fa_utils.f16ToF32(p.*);
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
}

test "fa config validation" {
    const good = FlashAttentionConfig{
        .N = 128, .d = 64, .num_heads = 8, .batch_size = 1,
    };
    try good.validate();

    const bad = FlashAttentionConfig{
        .N = 128, .d = 100, .num_heads = 8, .batch_size = 1,
    };
    try std.testing.expectError(error.InvalidConfig, bad.validate());
}

test "fa utils softmax" {
    var scores = [_]f32{ 1.0, 2.0, 3.0 };
    var out = [_]f32{ 0, 0, 0 };
    fa.fa_utils.softmax(&scores, &out);

    var sum: f32 = 0;
    for (out) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-6);
}

test "fa utils online softmax" {
    var state = fa.fa_utils.OnlineSoftmax{};
    var scores = [_]f32{ 1.0, 2.0, 3.0 };
    state.update(&scores);

    var accum = [_]f32{ 1.0, 2.0, 3.0 };
    state.normalize(&accum);

    var sum: f32 = 0;
    for (accum) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
}
