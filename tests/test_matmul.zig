const std = @import("std");
const Tensor = @import("core").Tensor;
const matmul = @import("matmul");

const MatmulEngine = matmul.MatmulEngine;
const Backend = matmul.Backend;
const PrecisionMode = matmul.PrecisionMode;

fn createTestMatrix(allocator: std.mem.Allocator, M: usize, N: usize, vals: []const f32) !Tensor(f32) {
    var t = try Tensor(f32).alloc(allocator, &[_]usize{ M, N });
    for (vals, 0..) |v, i| t.data[i] = v;
    return t;
}

test "naive gemm" {
    const allocator = std.testing.allocator;
    var A = try createTestMatrix(allocator, 2, 3, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    defer A.deinit();
    var B = try createTestMatrix(allocator, 3, 2, &[_]f32{ 7, 8, 9, 10, 11, 12 });
    defer B.deinit();
    var C = try Tensor(f32).alloc(allocator, &[_]usize{ 2, 2 });
    defer C.deinit();

    var engine = try MatmulEngine.init(allocator, .naive, .f32);
    defer engine.deinit();

    try engine.gemmNoTrans(f32, A, B, &C);

    try std.testing.expectApproxEqAbs(@as(f32, 58.0), C.at2(0, 0), 1e-6);  // 1*7+2*9+3*11
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), C.at2(0, 1), 1e-6);  // 1*8+2*10+3*12
    try std.testing.expectApproxEqAbs(@as(f32, 139.0), C.at2(1, 0), 1e-6); // 4*7+5*9+6*11
    try std.testing.expectApproxEqAbs(@as(f32, 154.0), C.at2(1, 1), 1e-6); // 4*8+5*10+6*12
}

test "simd gemm" {
    const allocator = std.testing.allocator;
    var A = try createTestMatrix(allocator, 2, 4, &[_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer A.deinit();
    var B = try createTestMatrix(allocator, 4, 2, &[_]f32{ 1, 0, 0, 1, 1, 0, 0, 1 });
    defer B.deinit();
    var C = try Tensor(f32).alloc(allocator, &[_]usize{ 2, 2 });
    defer C.deinit();

    var engine = try MatmulEngine.init(allocator, .simd, .f32);
    defer engine.deinit();

    try engine.gemmNoTrans(f32, A, B, &C);

    try std.testing.expectApproxEqAbs(@as(f32, 4.0), C.at2(0, 0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), C.at2(0, 1), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), C.at2(1, 0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 14.0), C.at2(1, 1), 1e-6);
}

test "linear projection" {
    const allocator = std.testing.allocator;
    var X = try createTestMatrix(allocator, 2, 3, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    defer X.deinit();
    var W_T = try createTestMatrix(allocator, 2, 3, &[_]f32{ 1, 0, 0, 0, 1, 0 });
    defer W_T.deinit();
    var Y = try Tensor(f32).alloc(allocator, &[_]usize{ 2, 2 });
    defer Y.deinit();

    var engine = try MatmulEngine.init(allocator, .naive, .f32);
    defer engine.deinit();

    try engine.linearProjection(f32, X, W_T, &Y);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), Y.at2(0, 0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), Y.at2(0, 1), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), Y.at2(1, 0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), Y.at2(1, 1), 1e-6);
}

test "quantized int8 gemm" {
    const allocator = std.testing.allocator;
    var A = try createTestMatrix(allocator, 2, 3, &[_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
    defer A.deinit();
    var B_f32 = try createTestMatrix(allocator, 3, 2, &[_]f32{ 1.0, 0.0, 0.0, 1.0, 1.0, 0.0 });
    defer B_f32.deinit();

    const qcfg = matmul.QuantConfig{ .bits = 8, .symmetric = true, .per_channel = false, .group_size = 0 };
    var B_q = try matmul.quantizeInt8Symmetric(allocator, B_f32, qcfg);
    defer B_q.deinit();

    var C = try Tensor(f32).alloc(allocator, &[_]usize{ 2, 2 });
    defer C.deinit();

    var engine = try MatmulEngine.init(allocator, .naive, .f32);
    defer engine.deinit();

    try engine.gemmQuantized(A, B_q, &C, 2, 2, 3);

    try std.testing.expectApproxEqAbs(@as(f32, 4.0), C.at2(0, 0), 0.1);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), C.at2(0, 1), 0.1);
}

test "f16/bf16 conversion" {
    const allocator = std.testing.allocator;
    var src = try createTestMatrix(allocator, 2, 2, &[_]f32{ 1.5, 2.5, 3.5, 4.5 });
    defer src.deinit();

    var f16_t = try matmul.tensorF32ToF16(allocator, src);
    defer f16_t.deinit();
    var back = try matmul.tensorF16ToF32(allocator, f16_t);
    defer back.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 1.5), back.at2(0, 0), 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 4.5), back.at2(1, 1), 1e-3);
}
