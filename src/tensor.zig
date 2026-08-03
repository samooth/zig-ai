const std = @import("std");

pub const ShapeError = error{
    InvalidShape,
    DimensionMismatch,
    OutOfBounds,
};

/// Tensor multidimensional con shape, strides, offset y dtype generico
pub fn Tensor(comptime T: type) type {
    return struct {
        data: []T,
        shape: []const usize,
        strides: []usize,
        offset: usize,
        allocator: ?std.mem.Allocator,
        owns_data: bool,

        const Self = @This();

        // ─── Constructores ───

        pub fn alloc(allocator: std.mem.Allocator, shape: []const usize) !Self {
            var total: usize = 1;
            for (shape) |s| {
                if (s == 0) return ShapeError.InvalidShape;
                total *= s;
            }
            const data = try allocator.alloc(T, total);
            @memset(data, 0);

            const strides = try allocator.alloc(usize, shape.len);
            computeStrides(shape, strides);

            const shape_copy = try allocator.dupe(usize, shape);

            return Self{
                .data = data,
                .shape = shape_copy,
                .strides = strides,
                .offset = 0,
                .allocator = allocator,
                .owns_data = true,
            };
        }

        pub fn init(allocator: std.mem.Allocator, shape: []const usize) !Self {
            return try alloc(allocator, shape);
        }

        pub fn initUninitialized(allocator: std.mem.Allocator, shape: []const usize) !Self {
            var total: usize = 1;
            for (shape) |s| {
                if (s == 0) return ShapeError.InvalidShape;
                total *= s;
            }
            const data = try allocator.alloc(T, total);
            const strides = try allocator.alloc(usize, shape.len);
            computeStrides(shape, strides);
            const shape_copy = try allocator.dupe(usize, shape);

            return Self{
                .data = data,
                .shape = shape_copy,
                .strides = strides,
                .offset = 0,
                .allocator = allocator,
                .owns_data = true,
            };
        }

        pub fn fromSlice(allocator: std.mem.Allocator, data: []const T, shape: []const usize) !Self {
            var total: usize = 1;
            for (shape) |s| total *= s;
            std.debug.assert(data.len == total);

            const owned = try allocator.dupe(T, data);
            const strides = try allocator.alloc(usize, shape.len);
            computeStrides(shape, strides);
            const shape_copy = try allocator.dupe(usize, shape);

            return Self{
                .data = owned,
                .shape = shape_copy,
                .strides = strides,
                .offset = 0,
                .allocator = allocator,
                .owns_data = true,
            };
        }

        pub fn view(self: Self, shape: []const usize, strides: []usize, offset: usize) Self {
            return Self{
                .data = self.data,
                .shape = shape,
                .strides = strides,
                .offset = offset,
                .allocator = self.allocator,
                .owns_data = false,
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.owns_data) {
                if (self.allocator) |a| {
                    a.free(self.data);
                    a.free(self.shape);
                    a.free(self.strides);
                }
            }
        }

        // ─── Indexacion ───

        pub fn at(self: Self, indices: []const usize) T {
            std.debug.assert(indices.len == self.shape.len);
            var idx = self.offset;
            for (indices, 0..) |i, dim| {
                std.debug.assert(i < self.shape[dim]);
                idx += i * self.strides[dim];
            }
            return self.data[idx];
        }

        pub fn ptr(self: Self, indices: []const usize) *T {
            std.debug.assert(indices.len == self.shape.len);
            var idx = self.offset;
            for (indices, 0..) |i, dim| {
                std.debug.assert(i < self.shape[dim]);
                idx += i * self.strides[dim];
            }
            return &self.data[idx];
        }

        pub fn set(self: Self, indices: []const usize, value: T) void {
            self.ptr(indices).* = value;
        }

        pub fn at2(self: Self, i: usize, j: usize) T {
            std.debug.assert(self.shape.len == 2);
            std.debug.assert(i < self.shape[0] and j < self.shape[1]);
            return self.data[self.offset + i * self.strides[0] + j * self.strides[1]];
        }

        pub fn ptr2(self: Self, i: usize, j: usize) *T {
            std.debug.assert(self.shape.len == 2);
            std.debug.assert(i < self.shape[0] and j < self.shape[1]);
            return &self.data[self.offset + i * self.strides[0] + j * self.strides[1]];
        }

        pub fn set2(self: Self, i: usize, j: usize, value: T) void {
            self.ptr2(i, j).* = value;
        }

        // ─── Utilidades ───

        pub fn numel(self: Self) usize {
            var n: usize = 1;
            for (self.shape) |s| n *= s;
            return n;
        }

        pub fn ndim(self: Self) usize {
            return self.shape.len;
        }

        pub fn len(self: Self) usize {
            return self.data.len;
        }

        pub fn isContiguous(self: Self) bool {
            var expected: usize = 1;
            var i = self.shape.len;
            while (i > 0) : (i -= 1) {
                if (self.strides[i - 1] != expected) return false;
                expected *= self.shape[i - 1];
            }
            return true;
        }

        pub fn reshape(self: Self, new_shape: []const usize) !Self {
            var new_total: usize = 1;
            for (new_shape) |s| new_total *= s;
            std.debug.assert(new_total == self.numel());

            const new_strides = try self.allocator.?.alloc(usize, new_shape.len);
            computeStrides(new_shape, new_strides);
            const new_shape_copy = try self.allocator.?.dupe(usize, new_shape);

            return self.view(new_shape_copy, new_strides, self.offset);
        }

        pub fn transpose(self: Self) !Self {
            std.debug.assert(self.shape.len == 2);
            const new_shape = try self.allocator.?.alloc(usize, 2);
            new_shape[0] = self.shape[1];
            new_shape[1] = self.shape[0];

            const new_strides = try self.allocator.?.alloc(usize, 2);
            new_strides[0] = self.strides[1];
            new_strides[1] = self.strides[0];

            return self.view(new_shape, new_strides, self.offset);
        }

        pub fn fill(self: Self, value: T) void {
            if (self.isContiguous()) {
                @memset(self.data[self.offset..][0..self.numel()], value);
            } else {
                var it = self.iterator();
                while (it.next()) |p| p.* = value;
            }
        }

        pub fn copyFrom(self: *Self, other: Self) !void {
            std.debug.assert(self.numel() == other.numel());
            if (self.isContiguous() and other.isContiguous()) {
                @memcpy(self.data[self.offset..][0..self.numel()],
                        other.data[other.offset..][0..other.numel()]);
            } else {
                var it_dst = self.iterator();
                var it_src = other.iterator();
                while (it_dst.next()) |pd| {
                    pd.* = it_src.next().?.*;
                }
            }
        }

        pub fn approxEq(self: Self, other: Self, tolerance: T) bool {
            if (self.numel() != other.numel()) return false;
            var it_a = self.iterator();
            var it_b = other.iterator();
            while (it_a.next()) |pa| {
                const pb = it_b.next().?.*;
                const diff = if (pa.* > pb) pa.* - pb else pb - pa.*;
                if (diff > tolerance) return false;
            }
            return true;
        }

        pub fn randn(self: Self, rng: *std.Random.Xoshiro256) void {
            var it = self.iterator();
            while (it.next()) |p| {
                const r1 = rng.random().float(f32);
                const r2 = rng.random().float(f32);
                const r = @sqrt(-2.0 * @log(r1));
                const theta = 2.0 * std.math.pi * r2;
                p.* = @as(T, @floatCast(r * @cos(theta)));
            }
        }

        pub fn randUniform(self: Self, rng: *std.Random.Xoshiro256, min: f32, max: f32) void {
            var it = self.iterator();
            while (it.next()) |p| {
                const f = rng.random().float(f32);
                p.* = @as(T, @floatCast(min + f * (max - min)));
            }
        }

        pub fn printInfo(self: Self, name: []const u8) void {
            std.debug.print("Tensor '{s}': shape=[", .{name});
            for (self.shape, 0..) |s, i| {
                if (i > 0) std.debug.print(",", .{});
                std.debug.print("{d}", .{s});
            }
            std.debug.print("] strides=[", .{});
            for (self.strides, 0..) |st, i| {
                if (i > 0) std.debug.print(",", .{});
                std.debug.print("{d}", .{st});
            }
            std.debug.print("] dtype={s}, numel={d}\n", .{@typeName(T), self.numel()});
        }

        pub fn printHead(self: Self, n: usize) void {
            const limit = @min(n, self.numel());
            std.debug.print("[", .{});
            var it = self.iterator();
            var i: usize = 0;
            while (i < limit) : (i += 1) {
                if (i > 0) std.debug.print(", ", .{});
                std.debug.print("{d:.4}", .{it.next().?.*});
            }
            if (self.numel() > limit) {
                std.debug.print(", ... ({d} more)", .{self.numel() - limit});
            }
            std.debug.print("]\n", .{});
        }

        // ─── Iterador sobre elementos (maneja strides) ───

        pub fn iterator(self: Self) Iterator {
            return Iterator{ .tensor = self, .pos = 0, .total = self.numel() };
        }

        pub const Iterator = struct {
            tensor: Self,
            pos: usize,
            total: usize,

            pub fn next(self: *Iterator) ?*T {
                if (self.pos >= self.total) return null;
                var idx = self.tensor.offset;
                var rem = self.pos;
                var dim = self.tensor.shape.len;
                while (dim > 0) : (dim -= 1) {
                    const d = dim - 1;
                    const coord = rem % self.tensor.shape[d];
                    rem /= self.tensor.shape[d];
                    idx += coord * self.tensor.strides[d];
                }
                self.pos += 1;
                return &self.tensor.data[idx];
            }
        };
    };
}

fn computeStrides(shape: []const usize, strides: []usize) void {
    var stride: usize = 1;
    var i = shape.len;
    while (i > 0) : (i -= 1) {
        strides[i - 1] = stride;
        stride *= shape[i - 1];
    }
}

// ─── Tests ───

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
