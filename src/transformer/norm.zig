const std = @import("std");
const Tensor = @import("core").Tensor;

/// RMSNorm: x * rsqrt(mean(x^2) + eps) * gamma
/// Usado en Llama, Mistral, Qwen, etc.
pub fn rmsNorm(comptime T: type, input: Tensor(T), gamma: Tensor(T), eps: T, output: *Tensor(T)) void {
    std.debug.assert(input.shape.len >= 2);
    std.debug.assert(gamma.shape.len == 1);
    const hidden_dim = gamma.shape[0];
    std.debug.assert(input.shape[input.shape.len - 1] == hidden_dim);
    std.debug.assert(std.mem.eql(usize, input.shape, output.shape));

    const num_rows = input.numel() / hidden_dim;
    for (0..num_rows) |row| {
        const offset = row * hidden_dim;

        // Calcular mean(x^2)
        var mean_sq: f32 = 0.0;
        for (0..hidden_dim) |d| {
            const val = @as(f32, @floatCast(input.data[offset + d]));
            mean_sq += val * val;
        }
        mean_sq /= @as(f32, @floatFromInt(hidden_dim));

        const scale = 1.0 / @sqrt(mean_sq + @as(f32, @floatCast(eps)));

        for (0..hidden_dim) |d| {
            const x = @as(f32, @floatCast(input.data[offset + d]));
            const g = @as(f32, @floatCast(gamma.data[d]));
            output.data[offset + d] = @as(T, @floatCast(x * scale * g));
        }
    }
}

/// LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta
pub fn layerNorm(comptime T: type, input: Tensor(T), gamma: Tensor(T), beta: ?Tensor(T), eps: T, output: *Tensor(T)) void {
    std.debug.assert(input.shape.len >= 2);
    std.debug.assert(gamma.shape.len == 1);
    const hidden_dim = gamma.shape[0];
    std.debug.assert(input.shape[input.shape.len - 1] == hidden_dim);
    std.debug.assert(std.mem.eql(usize, input.shape, output.shape));

    const num_rows = input.numel() / hidden_dim;
    for (0..num_rows) |row| {
        const offset = row * hidden_dim;

        // Calcular mean
        var mean: f32 = 0.0;
        for (0..hidden_dim) |d| {
            mean += @as(f32, @floatCast(input.data[offset + d]));
        }
        mean /= @as(f32, @floatFromInt(hidden_dim));

        // Calcular var
        var var_val: f32 = 0.0;
        for (0..hidden_dim) |d| {
            const diff = @as(f32, @floatCast(input.data[offset + d])) - mean;
            var_val += diff * diff;
        }
        var_val /= @as(f32, @floatFromInt(hidden_dim));

        const inv_std = 1.0 / @sqrt(var_val + @as(f32, @floatCast(eps)));

        for (0..hidden_dim) |d| {
            const x = @as(f32, @floatCast(input.data[offset + d]));
            const g = @as(f32, @floatCast(gamma.data[d]));
            const normalized = (x - mean) * inv_std * g;
            const b = if (beta) |b_| @as(f32, @floatCast(b_.data[d])) else 0.0;
            output.data[offset + d] = @as(T, @floatCast(normalized + b));
        }
    }
}

/// Aplica RMSNorm in-place (sobrescribe input)
pub fn rmsNormInPlace(comptime T: type, input: *Tensor(T), gamma: Tensor(T), eps: T) void {
    var output = input.*;
    rmsNorm(T, input.*, gamma, eps, &output);
    input.* = output;
}

// ─── Tests ───

test "rmsNorm basic" {
    const allocator = std.testing.allocator;
    var input = try Tensor(f32).alloc(allocator, &[_]usize{ 2, 4 });
    defer input.deinit();
    var gamma = try Tensor(f32).alloc(allocator, &[_]usize{4});
    defer gamma.deinit();
    var output = try Tensor(f32).alloc(allocator, &[_]usize{ 2, 4 });
    defer output.deinit();

    input.data[0] = 1.0; input.data[1] = 2.0; input.data[2] = 3.0; input.data[3] = 4.0;
    input.data[4] = 1.0; input.data[5] = 1.0; input.data[6] = 1.0; input.data[7] = 1.0;
    @memset(gamma.data, 1.0);

    rmsNorm(f32, input, gamma, 1e-5, &output);

    // Verificar que no es NaN
    for (output.data) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
}

test "layerNorm basic" {
    const allocator = std.testing.allocator;
    var input = try Tensor(f32).alloc(allocator, &[_]usize{ 1, 4 });
    defer input.deinit();
    var gamma = try Tensor(f32).alloc(allocator, &[_]usize{4});
    defer gamma.deinit();
    var beta = try Tensor(f32).alloc(allocator, &[_]usize{4});
    defer beta.deinit();
    var output = try Tensor(f32).alloc(allocator, &[_]usize{ 1, 4 });
    defer output.deinit();

    @memset(input.data, 1.0);
    @memset(gamma.data, 1.0);
    @memset(beta.data, 0.0);

    layerNorm(f32, input, gamma, beta, 1e-5, &output);

    // Con input constante, la salida debería ser aprox beta (0)
    for (output.data) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), v, 1e-5);
    }
}
