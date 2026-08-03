const std = @import("std");
const Tensor = @import("core").Tensor;
const matmul = @import("matmul");

/// SwiGLU FFN: output = (silu(x @ W_gate) * (x @ W_up)) @ W_down
/// Usado en Llama 2+, Mistral, Qwen, etc.
pub fn swiGluForward(
    engine: *matmul.MatmulEngine,
    comptime T: type,
    x: Tensor(T),              // [batch*seq, hidden_dim]
    w_gate_t: Tensor(T),       // [intermediate_dim, hidden_dim] transpuesto
    w_up_t: Tensor(T),         // [intermediate_dim, hidden_dim] transpuesto
    w_down_t: Tensor(T),       // [hidden_dim, intermediate_dim] transpuesto
    gate_buf: *Tensor(T),      // [batch*seq, intermediate_dim]
    up_buf: *Tensor(T),        // [batch*seq, intermediate_dim]
    output: *Tensor(T),        // [batch*seq, hidden_dim]
) !void {
    // 1. Proyecciones paralelas
    try engine.linearProjection(T, x, w_gate_t, gate_buf);
    try engine.linearProjection(T, x, w_up_t, up_buf);

    // 2. Aplicar SiLU a gate y multiplicar elemento a elemento con up
    const numel = gate_buf.numel();
    for (0..numel) |i| {
        const g = @as(f32, @floatCast(gate_buf.data[i]));
        const u = @as(f32, @floatCast(up_buf.data[i]));
        // SiLU(x) = x * sigmoid(x)
        const sigmoid = 1.0 / (1.0 + @exp(-g));
        gate_buf.data[i] = @as(T, @floatCast(g * sigmoid * u));
    }

    // 3. Proyección down
    try engine.linearProjection(T, gate_buf.*, w_down_t, output);
}

/// GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
/// Usado en GPT-2, BERT, etc.
pub fn geluForward(
    engine: *matmul.MatmulEngine,
    comptime T: type,
    x: Tensor(T),
    w1_t: Tensor(T),
    w2_t: Tensor(T),
    intermediate: *Tensor(T),
    output: *Tensor(T),
) !void {
    try engine.linearProjection(T, x, w1_t, intermediate);

    const numel = intermediate.numel();
    const sqrt_2_over_pi = @sqrt(2.0 / std.math.pi);
    for (0..numel) |i| {
        const val = @as(f32, @floatCast(intermediate.data[i]));
        const cube = val * val * val;
        const inner = sqrt_2_over_pi * (val + 0.044715 * cube);
        intermediate.data[i] = @as(T, @floatCast(val * 0.5 * (1.0 + std.math.tanh(inner))));
    }

    try engine.linearProjection(T, intermediate.*, w2_t, output);
}

/// ReLU FFN simple (para modelos legacy)
pub fn reluForward(
    engine: *matmul.MatmulEngine,
    comptime T: type,
    x: Tensor(T),
    w1_t: Tensor(T),
    w2_t: Tensor(T),
    intermediate: *Tensor(T),
    output: *Tensor(T),
) !void {
    try engine.linearProjection(T, x, w1_t, intermediate);

    for (intermediate.data) |*p| {
        const val = @as(f32, @floatCast(p.*));
        p.* = @as(T, @floatCast(if (val > 0.0) val else 0.0));
    }

    try engine.linearProjection(T, intermediate.*, w2_t, output);
}

// ─── Tests ───

test "swiGlu shapes" {
    const allocator = std.testing.allocator;
    const hidden_dim: usize = 128;
    const inter_dim: usize = 256;
    const seq_len: usize = 4;

    var x = try Tensor(f32).alloc(allocator, &[_]usize{ seq_len, hidden_dim });
    defer x.deinit();
    var w_gate = try Tensor(f32).alloc(allocator, &[_]usize{ inter_dim, hidden_dim });
    defer w_gate.deinit();
    var w_up = try Tensor(f32).alloc(allocator, &[_]usize{ inter_dim, hidden_dim });
    defer w_up.deinit();
    var w_down = try Tensor(f32).alloc(allocator, &[_]usize{ hidden_dim, inter_dim });
    defer w_down.deinit();
    var gate_buf = try Tensor(f32).alloc(allocator, &[_]usize{ seq_len, inter_dim });
    defer gate_buf.deinit();
    var up_buf = try Tensor(f32).alloc(allocator, &[_]usize{ seq_len, inter_dim });
    defer up_buf.deinit();
    var output = try Tensor(f32).alloc(allocator, &[_]usize{ seq_len, hidden_dim });
    defer output.deinit();

    var engine = try matmul.MatmulEngine.init(allocator, .naive, .f32);
    defer engine.deinit();

    try swiGluForward(&engine, f32, x, w_gate, w_up, w_down, &gate_buf, &up_buf, &output);

    try std.testing.expectEqual(seq_len, output.shape[0]);
    try std.testing.expectEqual(hidden_dim, output.shape[1]);
}
