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

/// MRoPE (Multi-section RoPE) para qwen35 / Qwen3.5 hybrid.
/// Fiel a llama.cpp ggml_mrope_cache_init + rotate_pairs (NEOX/MROPE mode).
/// - NEOX half-split: pares (i, i + n_rot/2) para i en 0..n_rot/2-1
/// - Solo primeros n_rot dims rotados; resto copiado sin cambios
/// - Para texto: los 4 position ids (t,h,w,e) son iguales → sectores irrelevantes,
///   equivalente a NEOX estándar sobre n_rot dims con base freq_base
/// - Q/K shape: [batch, num_heads, seq_len, head_dim]
pub fn applyRoPEMultiSection(
    Q: *Tensor(f16),
    K: *Tensor(f16),
    start_pos: usize,
    head_dim: usize,
    n_rot: usize,
    sections: [4]usize,
    base: f32,
) void {
    std.debug.assert(Q.shape.len == 4);
    std.debug.assert(K.shape.len == 4);
    std.debug.assert(n_rot % 2 == 0);
    std.debug.assert(n_rot <= head_dim);
    std.debug.assert(sections[0] + sections[1] + sections[2] + sections[3] == n_rot / 2);

    const batch_size = Q.shape[0];
    const num_heads_q = Q.shape[1];
    const num_heads_k = K.shape[1];
    const seq_len = Q.shape[2];

    const half_rot = n_rot / 2;
    const theta_scale = std.math.pow(f32, base, -2.0 / @as(f32, @floatFromInt(n_rot)));

    // Cache de cos/sin por posición y dimensión [seq_len][n_rot]
    var cache = std.heap.page_allocator.alloc(f32, seq_len * n_rot) catch unreachable;
    defer std.heap.page_allocator.free(cache);

    const sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    const sec_w = sections[0] + sections[1];
    const sec_e = sections[2] + sec_w;

    for (0..seq_len) |pos| {
        const global_pos = @as(f32, @floatFromInt(start_pos + pos));
        var theta_t = global_pos;
        var theta_h = global_pos;
        var theta_w = global_pos;
        var theta_e = global_pos;

        for (0..n_rot) |idx| {
            const i = idx / 2;
            const sector = i % sect_dims;

            var theta = theta_t;
            if (sector >= sections[0] and sector < sec_w) {
                theta = theta_h;
            } else if (sector >= sec_w and sector < sec_e) {
                theta = theta_w;
            } else if (sector >= sec_e) {
                theta = theta_e;
            }

            cache[pos * n_rot + idx] = if (idx % 2 == 0) @cos(theta) else @sin(theta);

            if (idx % 2 == 1) {
                theta_t *= theta_scale;
                theta_h *= theta_scale;
                theta_w *= theta_scale;
                theta_e *= theta_scale;
            }
        }
    }

    // Aplicar rotación NEOX half-split: pares (ic, ic + half_rot) para ic en 0..half_rot-1
    // cache index: cos=2*ic, sin=2*ic+1
    for (0..batch_size) |b| {
        for (0..num_heads_q) |h| {
            for (0..seq_len) |pos| {
                const row_offset = ((b * num_heads_q + h) * seq_len + pos) * head_dim;
                for (0..half_rot) |ic| {
                    const cos_val = cache[pos * n_rot + 2 * ic];
                    const sin_val = cache[pos * n_rot + 2 * ic + 1];

                    const idx0 = row_offset + ic;
                    const idx1 = row_offset + ic + half_rot;

                    const q0 = @as(f32, @floatCast(Q.data[idx0]));
                    const q1 = @as(f32, @floatCast(Q.data[idx1]));

                    Q.data[idx0] = @floatCast(q0 * cos_val - q1 * sin_val);
                    Q.data[idx1] = @floatCast(q0 * sin_val + q1 * cos_val);
                }
                // dims n_rot..head_dim sin cambios
            }
        }
    }

    for (0..batch_size) |b| {
        for (0..num_heads_k) |h| {
            for (0..seq_len) |pos| {
                const row_offset = ((b * num_heads_k + h) * seq_len + pos) * head_dim;
                for (0..half_rot) |ic| {
                    const cos_val = cache[pos * n_rot + 2 * ic];
                    const sin_val = cache[pos * n_rot + 2 * ic + 1];

                    const idx0 = row_offset + ic;
                    const idx1 = row_offset + ic + half_rot;

                    const k0 = @as(f32, @floatCast(K.data[idx0]));
                    const k1 = @as(f32, @floatCast(K.data[idx1]));

                    K.data[idx0] = @floatCast(k0 * cos_val - k1 * sin_val);
                    K.data[idx1] = @floatCast(k0 * sin_val + k1 * cos_val);
                }
            }
        }
    }
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
