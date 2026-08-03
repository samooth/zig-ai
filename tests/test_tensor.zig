const std = @import("std");
const Tensor = @import("core").Tensor;

test "tensor alloc and basic ops" {
    const allocator = std.testing.allocator;
    var t = try Tensor(f32).alloc(allocator, &[_]usize{ 3, 4 });
    defer t.deinit();

    try std.testing.expectEqual(@as(usize, 2), t.ndim());
    try std.testing.expectEqual(@as(usize, 12), t.numel());

    t.set2(1, 2, 3.14);
    try std.testing.expectApproxEqAbs(@as(f32, 3.14), t.at2(1, 2), 1e-6);
}

test "tensor transpose" {
    const allocator = std.testing.allocator;
    var t = try Tensor(f32).alloc(allocator, &[_]usize{ 2, 3 });
    defer t.deinit();

    t.set2(0, 1, 5.0);
    t.set2(1, 2, 7.0);

    var tt = try t.transpose();
    defer {
        if (tt.allocator) |a| { a.free(tt.shape); a.free(tt.strides); }
    }

    try std.testing.expectEqual(@as(usize, 3), tt.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), tt.shape[1]);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), tt.at2(1, 0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), tt.at2(2, 1), 1e-6);
}

test "tensor iterator" {
    const allocator = std.testing.allocator;
    var t = try Tensor(f32).alloc(allocator, &[_]usize{ 2, 2 });
    defer t.deinit();

    t.set2(0, 0, 1); t.set2(0, 1, 2);
    t.set2(1, 0, 3); t.set2(1, 1, 4);

    var it = t.iterator();
    try std.testing.expectApproxEqAbs(@as(f32, 1), it.next().?.*, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2), it.next().?.*, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3), it.next().?.*, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4), it.next().?.*, 1e-6);
    try std.testing.expect(it.next() == null);
}

test "tensor reshape" {
    const allocator = std.testing.allocator;
    var t = try Tensor(f32).alloc(allocator, &[_]usize{ 2, 3 });
    defer t.deinit();

    t.set2(0, 0, 1.0); t.set2(0, 1, 2.0); t.set2(0, 2, 3.0);
    t.set2(1, 0, 4.0); t.set2(1, 1, 5.0); t.set2(1, 2, 6.0);

    var r = try t.reshape(&[_]usize{ 3, 2 });
    defer {
        if (r.allocator) |a| { a.free(r.shape); a.free(r.strides); }
    }

    try std.testing.expectEqual(@as(usize, 3), r.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), r.shape[1]);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), r.at2(2, 0), 1e-6);
}

test "tensor fill and copy" {
    const allocator = std.testing.allocator;
    var a = try Tensor(f32).alloc(allocator, &[_]usize{ 2, 2 });
    defer a.deinit();
    var b = try Tensor(f32).alloc(allocator, &[_]usize{ 2, 2 });
    defer b.deinit();

    a.fill(3.14);
    try b.copyFrom(a);

    try std.testing.expectApproxEqAbs(@as(f32, 3.14), b.at2(0, 0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.14), b.at2(1, 1), 1e-6);
}

test "tensor randUniform" {
    const allocator = std.testing.allocator;
    var t = try Tensor(f32).alloc(allocator, &[_]usize{ 10, 10 });
    defer t.deinit();

    var rng = std.Random.Xoshiro256.init(42);
    t.randUniform(&rng, -1.0, 1.0);

    var min: f32 = std.math.inf(f32);
    var max: f32 = -std.math.inf(f32);
    var it = t.iterator();
    while (it.next()) |p| {
        if (p.* < min) min = p.*;
        if (p.* > max) max = p.*;
    }
    try std.testing.expect(min >= -1.0 and min <= 1.0);
    try std.testing.expect(max >= -1.0 and max <= 1.0);
}
