const std = @import("std");
const fa = @import("fa");

test "online softmax state update" {
    var state = fa.fa_utils.OnlineSoftmax{};

    var tile1 = [_]f32{ 1.0, 2.0, 3.0 };
    state.update(&tile1);

    var tile2 = [_]f32{ 2.0, 3.0, 4.0 };
    state.update(&tile2);

    var accum = [_]f32{ 1.0, 2.0, 3.0, 2.0, 3.0, 4.0 };
    state.normalize(&accum);

    var sum: f32 = 0;
    for (accum) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
}

test "causal mask" {
    var scores = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    fa.fa_utils.applyCausalMask(&scores, 3);

    // Posiciones donde j > i deben ser -inf
    try std.testing.expect(!std.math.isInf(scores[0]));
    try std.testing.expect(std.math.isInf(scores[1]));  // (0,1)
    try std.testing.expect(std.math.isInf(scores[2]));  // (0,2)
    try std.testing.expect(!std.math.isInf(scores[3])); // (1,0)
    try std.testing.expect(!std.math.isInf(scores[4])); // (1,1)
    try std.testing.expect(std.math.isInf(scores[5]));  // (1,2)
}
