const std = @import("std");
const Tensor = @import("core").Tensor;

pub const RopeScalingType = enum { none, linear, yarn };

pub const RopeScaling = struct {
    scaling_type: RopeScalingType = .none,
    factor: f32 = 1.0,
    orig_ctx: usize = 0,
    attn_factor: f32 = 1.0,
    yarn_ext_factor: f32 = -1.0,
    yarn_beta_fast: f32 = 32.0,
    yarn_beta_slow: f32 = 1.0,
};

/// Pairing RoPE: qué pares de canales rota applyRoPE. El oráculo es
/// llama_model_rope_type (llama-model.cpp): LLAMA/BITNET/QWEN2-3 → NORM;
/// LFM2/LFM2MOE/QWEN3NEXT → NEOX. Los callers híbridos lo pasan explícito
/// según arch; `.auto` = NORM default con A/B env ZIG_AI_ROPE_NEOX=1 (F1v7).
pub const RopePairing = enum { auto, norm, neox };

/// Generic RoPE that works with any numeric type (f32, f16, etc.)
/// Q/K shape: [batch, num_heads, seq_len, head_dim]
pub fn applyRoPE(
    comptime T: type,
    Q: *Tensor(T),
    K: *Tensor(T),
    start_pos: usize,
    head_dim: usize,
    base: f32,
    pairing: RopePairing,
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

    // F1v7 (lane-f): pairing NORM (pares consecutivos (2i, 2i+1)) —
    // llama.cpp llama_model_rope_type: LLM_ARCH_LLAMA → LLAMA_ROPE_TYPE_NORM
    // ("normal RoPE, operating on pairs of consecutive head values",
    // llama-model.cpp:2572). El half-split NEOX de antes era el bug del
    // hallazgo 11.6: contexto corto OK (paridad 6-tok "Paris"), largo
    // degradado posicionalmente (PPL 4641 vs golden 6.96 en wiki12k).
    // A/B del pairing: ZIG_AI_ROPE_NEOX=1 restaura el half-split.
    //
    // F1v8 (coord, post-merge audit): LFM2/LFM2MOE → LLAMA_ROPE_TYPE_NEOX
    // (llama-model.cpp:2682 — MISMA lista NEOX que GPT-NeoX/Falcon). El path
    // híbrido use_mrope=false (lfm2) DEBE retener NEOX: NORM aquí sería
    // regresión. El oráculo de pairing vive en el caller (HybridAttnParams
    // .rope_neox, seteado por hybrid_layer según arch); este default NORM es
    // el correcto para legacy llama-like (layer.zig:470) y BitNet (11.9).
    const use_neox = if (pairing == .auto)
        std.c.getenv("ZIG_AI_ROPE_NEOX") != null
    else
        pairing == .neox;
    const applyOne = struct {
        fn go(t: *Tensor(T), b: usize, h: usize, pos: usize, global_pos: usize, hd: usize, fr: []const f32, neox: bool) void {
            const base_off = t.offset +
                b * t.strides[0] +
                h * t.strides[1] +
                pos * t.strides[2];
            const d_stride = if (t.strides.len >= 4) t.strides[3] else 1;
            const half = hd / 2;
            if (neox) {
                for (0..half) |i| {
                    const theta = @as(f32, @floatFromInt(global_pos)) * fr[i];
                    const cos_val = @cos(theta);
                    const sin_val = @sin(theta);

                    const idx_a = base_off + i * d_stride;
                    const idx_b = base_off + (i + half) * d_stride;

                    const a = @as(f32, @floatCast(t.data[idx_a]));
                    const b2 = @as(f32, @floatCast(t.data[idx_b]));

                    t.data[idx_a] = @floatCast(a * cos_val - b2 * sin_val);
                    t.data[idx_b] = @floatCast(a * sin_val + b2 * cos_val);
                }
            } else {
                // NORM: pares consecutivos (2i, 2i+1) — freq de la pareja i
                for (0..half) |i| {
                    const theta = @as(f32, @floatFromInt(global_pos)) * fr[i];
                    const cos_val = @cos(theta);
                    const sin_val = @sin(theta);

                    const idx_a = base_off + (2 * i) * d_stride;
                    const idx_b = base_off + (2 * i + 1) * d_stride;

                    const a = @as(f32, @floatCast(t.data[idx_a]));
                    const b2 = @as(f32, @floatCast(t.data[idx_b]));

                    t.data[idx_a] = @floatCast(a * cos_val - b2 * sin_val);
                    t.data[idx_b] = @floatCast(a * sin_val + b2 * cos_val);
                }
            }
        }
    }.go;

    // Aplicar a Q
    for (0..batch_size) |b| {
        for (0..num_heads_q) |h| {
            for (0..seq_len) |pos| {
                applyOne(Q, b, h, pos, start_pos + pos, head_dim, freqs, use_neox);
            }
        }
    }

    // Aplicar a K (mismo seq_len)
    for (0..batch_size) |b| {
        for (0..num_heads_k) |h| {
            for (0..seq_len) |pos| {
                applyOne(K, b, h, pos, start_pos + pos, head_dim, freqs, use_neox);
            }
        }
    }
}

/// Legacy f16-only RoPE (kept for compatibility)
pub fn applyRoPE_f16(
    Q: *Tensor(f16),
    K: *Tensor(f16),
    start_pos: usize,
    head_dim: usize,
    base: f32,
) void {
    applyRoPE(f16, Q, K, start_pos, head_dim, base, .auto);
}

/// KT-B (lane-f): RoPE forward sobre UN slice [head_dim] de un head en una
/// posición dada — sin Tensor, para buffers planos del KV-transfer.
/// Slice layout: pares NORM (2i, 2i+1) o NEOX half-split según pairing.
/// Composición exacta con applyRoPEInverseOnSlice (rotaciones conmutan:
/// R(θ)·R(−θ) = I).
pub fn applyRoPEForwardOnSlice(
    comptime T: type,
    slice: []T,
    pos: usize,
    head_dim: usize,
    base: f32,
    pairing: RopePairing,
) void {
    std.debug.assert(slice.len == head_dim);
    std.debug.assert(head_dim % 2 == 0);
    const use_neox = if (pairing == .auto)
        std.c.getenv("ZIG_AI_ROPE_NEOX") != null
    else
        pairing == .neox;
    const half = head_dim / 2;
    const gpos: f32 = @floatFromInt(pos);
    for (0..half) |i| {
        const freq = 1.0 / std.math.pow(f32, base, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim)));
        const theta = gpos * freq;
        const cos_v: T = @floatCast(@cos(theta));
        const sin_v: T = @floatCast(@sin(theta));
        if (use_neox) {
            const a = slice[i];
            const b = slice[i + half];
            slice[i] = a * cos_v - b * sin_v;
            slice[i + half] = a * sin_v + b * cos_v;
        } else {
            const a = slice[2 * i];
            const b = slice[2 * i + 1];
            slice[2 * i] = a * cos_v - b * sin_v;
            slice[2 * i + 1] = a * sin_v + b * cos_v;
        }
    }
}

/// KT-B (lane-f): rotación INVERSA (θ → −θ) de un slice [head_dim] —
/// deshace el RoPE del cache post-RoPE para recuperar el K pre-RoPE del
/// source antes del mapper. Mismo pairing/base que aplicó el forward.
pub fn applyRoPEInverseOnSlice(
    comptime T_in: type,
    slice_in: []const T_in,
    out: []f32,
    pos: usize,
    head_dim: usize,
    base: f32,
    pairing: RopePairing,
) void {
    std.debug.assert(slice_in.len == head_dim and out.len == head_dim);
    std.debug.assert(head_dim % 2 == 0);
    const use_neox = if (pairing == .auto)
        std.c.getenv("ZIG_AI_ROPE_NEOX") != null
    else
        pairing == .neox;
    const half = head_dim / 2;
    const gpos: f32 = @floatFromInt(pos);
    for (0..half) |i| {
        const freq = 1.0 / std.math.pow(f32, base, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim)));
        const theta = gpos * freq;
        const cos_v: f32 = @cos(theta);
        const sin_v: f32 = @sin(theta);
        if (use_neox) {
            const a: f32 = @floatCast(slice_in[i]);
            const b: f32 = @floatCast(slice_in[i + half]);
            out[i] = a * cos_v + b * sin_v;
            out[i + half] = -a * sin_v + b * cos_v;
        } else {
            const a: f32 = @floatCast(slice_in[2 * i]);
            const b: f32 = @floatCast(slice_in[2 * i + 1]);
            out[2 * i] = a * cos_v + b * sin_v;
            out[2 * i + 1] = -a * sin_v + b * cos_v;
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
    applyRoPE(f16, Q, K, position, head_dim, base, .auto);
}

/// MRoPE (Multi-section RoPE) para qwen35 / Qwen3.5 hybrid.
/// Fiel a llama.cpp ggml_mrope_cache_init + rotate_pairs (NEOX/MROPE mode).
/// - NEOX half-split: pares (i, i + n_rot/2) para i en 0..n_rot/2-1
/// - Solo primeros n_rot dims rotados; resto copiado sin cambios
/// - Para texto: los 4 position ids (t,h,w,e) son iguales → sectores irrelevantes,
///   equivalente a NEOX estándar sobre n_rot dims con base freq_base
/// - Q/K shape: [batch, num_heads, seq_len, head_dim]
pub fn applyRoPEMultiSection(
    comptime T: type,
    Q: *Tensor(T),
    K: *Tensor(T),
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

/// MRoPE con position-ids PER-TOKEN (4 ids (t,h,w,e) por token) — para
/// embeddings de imagen inyectados en el target (llama.cpp
/// `llama_batch.pos` con n_pos_per_embd=4, mtmd-helper.cpp:142-213).
///
/// Idéntica rotación NEOX half-split que applyRoPEMultiSection; sólo cambia
/// la inicialización de thetas: cada token usa sus propios ids de sección
/// en lugar de `start_pos + pos` (que asume 4 ids iguales y secuenciales).
///
/// Para texto (todos los ids == posición secuencial) es bit-equivalente a
/// applyRoPEMultiSection — ver test.
///
/// Q/K shape: [batch, num_heads, seq_len, head_dim]; ids.len == seq_len.
pub fn applyRoPEMultiSectionPosIds(
    comptime T: type,
    Q: *Tensor(T),
    K: *Tensor(T),
    ids: []const [4]i32, // por token: (t, h, w, e)
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
    std.debug.assert(ids.len == seq_len);

    const half_rot = n_rot / 2;
    const theta_scale = std.math.pow(f32, base, -2.0 / @as(f32, @floatFromInt(n_rot)));

    var cache = std.heap.page_allocator.alloc(f32, seq_len * n_rot) catch unreachable;
    defer std.heap.page_allocator.free(cache);

    const sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    const sec_w = sections[0] + sections[1];
    const sec_e = sections[2] + sec_w;

    for (0..seq_len) |pos| {
        var theta_t: f32 = @floatFromInt(ids[pos][0]);
        var theta_h: f32 = @floatFromInt(ids[pos][1]);
        var theta_w: f32 = @floatFromInt(ids[pos][2]);
        var theta_e: f32 = @floatFromInt(ids[pos][3]);

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

    // Rotación NEOX half-split (idéntica a applyRoPEMultiSection)
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

    var Q = try Tensor(f32).alloc(allocator, &[_]usize{ batch, heads, seq, dim });
    defer Q.deinit();
    var K = try Tensor(f32).alloc(allocator, &[_]usize{ batch, heads, seq, dim });
    defer K.deinit();

    var rng = std.Random.Xoshiro256.init(42);
    Q.randUniform(&rng, -0.5, 0.5);
    K.randUniform(&rng, -0.5, 0.5);

    // Calcular norma antes
    var norm_before: f32 = 0;
    for (Q.data) |v| {
        const f = @as(f32, @floatCast(v));
        norm_before += f * f;
    }

    applyRoPE(f32, &Q, &K, 0, dim, 10000.0, .auto);

    // Calcular norma después (RoPE es una rotación, preserva norma)
    var norm_after: f32 = 0;
    for (Q.data) |v| {
        const f = @as(f32, @floatCast(v));
        norm_after += f * f;
    }

    try std.testing.expectApproxEqAbs(norm_before, norm_after, 1e-2);
}

test "applyRoPEMultiSectionPosIds: ids secuenciales == applyRoPEMultiSection" {
    const allocator = std.testing.allocator;
    const batch: usize = 1;
    const heads: usize = 2;
    const seq: usize = 4;
    const dim: usize = 64;
    const n_rot: usize = 32;
    const sections = [4]usize{ 8, 8, 0, 0 };

    var Q1 = try Tensor(f16).alloc(allocator, &[_]usize{ batch, heads, seq, dim });
    defer Q1.deinit();
    var K1 = try Tensor(f16).alloc(allocator, &[_]usize{ batch, heads, seq, dim });
    defer K1.deinit();
    var Q2 = try Tensor(f16).alloc(allocator, &[_]usize{ batch, heads, seq, dim });
    defer Q2.deinit();
    var K2 = try Tensor(f16).alloc(allocator, &[_]usize{ batch, heads, seq, dim });
    defer K2.deinit();

    var rng = std.Random.Xoshiro256.init(42);
    Q1.randUniform(&rng, -0.5, 0.5);
    K1.randUniform(&rng, -0.5, 0.5);
    @memcpy(Q2.data, Q1.data);
    @memcpy(K2.data, K1.data);

    // start_pos=3 en la clásica
    applyRoPEMultiSection(f16, &Q1, &K1, 3, dim, n_rot, sections, 10000.0);

    // ids secuenciales (3,4,5,6) en la nueva — 4 ids iguales por token
    const ids = [_][4]i32{
        .{ 3, 3, 3, 3 },
        .{ 4, 4, 4, 4 },
        .{ 5, 5, 5, 5 },
        .{ 6, 6, 6, 6 },
    };
    applyRoPEMultiSectionPosIds(f16, &Q2, &K2, &ids, dim, n_rot, sections, 10000.0);

    // bit-equivalencia (misma fórmula, mismas thetas)
    for (Q1.data, Q2.data) |a, b| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatCast(a)), @as(f32, @floatCast(b)), 1e-3);
    }
    for (K1.data, K2.data) |a, b| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatCast(a)), @as(f32, @floatCast(b)), 1e-3);
    }
}
