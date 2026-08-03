const std = @import("std");
const Tensor = @import("core").Tensor;

/// RoPE (Rotary Position Embedding) vectorizado
/// Aplica rotación a pares de dimensiones (d/2 pares)
/// Q/K shape: [batch, num_heads, seq_len, head_dim]
pub fn applyRoPE(
    Q: *Tensor(f16),
    K: *Tensor(f16),
    start_pos: usize,
    head_dim: usize,
    base: f32,         // theta base, típicamente 10000.0 o 1000000.0
) void {
    std.debug.assert(Q.shape.len == 4);
    std.debug.assert(K.shape.len == 4);
    std.debug.assert(head_dim % 2 == 0);

    const batch_size = Q.shape[0];
    const num_heads_q = Q.shape[1];
    const num_heads_k = K.shape[1];
    const seq_len = Q.shape[2];

    const half_dim = head_dim / 2;

    // Precomputar frecuencias una vez
    var freqs = std.heap.page_allocator.alloc(f32, half_dim) catch unreachable;
    defer std.heap.page_allocator.free(freqs);

    for (0..half_dim) |i| {
        const exponent = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim));
        freqs[i] = 1.0 / std.math.pow(f32, base, exponent);
    }

    // Aplicar a Q
    for (0..batch_size) |b| {
        for (0..num_heads_q) |h| {
            for (0..seq_len) |pos| {
                const global_pos = start_pos + pos;
                const row_offset = ((b * num_heads_q + h) * seq_len + pos) * head_dim;

                for (0..half_dim) |i| {
                    const theta = @as(f32, @floatFromInt(global_pos)) * freqs[i];
                    const cos_val = @cos(theta);
                    const sin_val = @sin(theta);

                    const idx_even = row_offset + 2 * i;
                    const idx_odd = idx_even + 1;

                    const q_even = @as(f32, @floatCast(Q.data[idx_even]));
                    const q_odd = @as(f32, @floatCast(Q.data[idx_odd]));

                    Q.data[idx_even] = @floatCast(q_even * cos_val - q_odd * sin_val);
                    Q.data[idx_odd] = @floatCast(q_even * sin_val + q_odd * cos_val);
                }
            }
        }
    }

    // Aplicar a K (mismo seq_len)
    for (0..batch_size) |b| {
        for (0..num_heads_k) |h| {
            for (0..seq_len) |pos| {
                const global_pos = start_pos + pos;
                const row_offset = ((b * num_heads_k + h) * seq_len + pos) * head_dim;

                for (0..half_dim) |i| {
                    const theta = @as(f32, @floatFromInt(global_pos)) * freqs[i];
                    const cos_val = @cos(theta);
                    const sin_val = @sin(theta);

                    const idx_even = row_offset + 2 * i;
                    const idx_odd = idx_even + 1;

                    const k_even = @as(f32, @floatCast(K.data[idx_even]));
                    const k_odd = @as(f32, @floatCast(K.data[idx_odd]));

                    K.data[idx_even] = @floatCast(k_even * cos_val - k_odd * sin_val);
                    K.data[idx_odd] = @floatCast(k_even * sin_val + k_odd * cos_val);
                }
            }
        }
    }
}

/// RoPE para un único token (generación autoregresiva)
/// Q/K shape: [batch, num_heads, 1, head_dim]
pub fn applyRoPESingle(
    Q: *Tensor(f16),
    K: *Tensor(f16),
    position: usize,
    head_dim: usize,
    base: f32,
) void {
    std.debug.assert(Q.shape[2] == 1);
    std.debug.assert(K.shape[2] == 1);
    applyRoPE(Q, K, position, head_dim, base);
}

// ─── Tests ───

test "rope preserves norm" {
    const allocator = std.testing.allocator;
    const batch: usize = 1;
    const heads: usize = 2;
    const seq: usize = 4;
    const dim: usize = 64;

    var Q = try Tensor(f16).alloc(allocator, &[_]usize{ batch, heads, seq, dim });
    defer Q.deinit();
    var K = try Tensor(f16).alloc(allocator, &[_]usize{ batch, heads, seq, dim });
    defer K.deinit();

    var rng = std.Random.Xoshiro256.init(42);
    Q.randUniform(&rng, -0.5, 0.5);
    K.randUniform(&rng, -0.5, 0.5);

    // Calcular norma antes
    var norm_before: f32 = 0;
    for (Q.data) |v| { const f = @as(f32, @floatCast(v)); norm_before += f * f; }

    applyRoPE(&Q, &K, 0, dim, 10000.0);

    // Calcular norma después (RoPE es una rotación, preserva norma)
    var norm_after: f32 = 0;
    for (Q.data) |v| { const f = @as(f32, @floatCast(v)); norm_after += f * f; }

    try std.testing.expectApproxEqAbs(norm_before, norm_after, 1e-2);
}
