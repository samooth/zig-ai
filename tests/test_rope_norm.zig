//! Test de regresión F1v7 (lane-f, hallazgo 11.6): el pairing RoPE del
//! path legacy llama-arch debe ser **NORM (pares consecutivos (2i, 2i+1))**,
//! no NEOX half-split — llama.cpp `llama_model_rope_type`:
//! LLM_ARCH_LLAMA → LLAMA_ROPE_TYPE_NORM ("pairs of consecutive head
//! values", llama-model.cpp:2572).
//!
//! El NEOX half-split de antes producía: paridad 6-tok OK (rotaciones
//! cortas), degradación posicional creciente en contexto largo (PPL 4641
//! vs golden 6.96 en wiki12k 1B, "Berlin"→' a' tras 11 tok). A/B del
//! pairing: ZIG_AI_ROPE_NEOX=1 (aplicación NO cubierta por este test).

const std = @import("std");
const rope = @import("rope");
const Tensor = @import("core").Tensor;

/// Referencia del pairing NORM estilo ggml rope_norm: par (2i, 2i+1) rota
/// con theta = pos * base^(-2i/d). Escribimos la referencia INDEPENDIENTE
/// (fórmula directa de llama.cpp rope_norm_cache_init + rope_one_tensor)
/// para que el test no sea tautológico con applyRoPE.
fn refRopeNorm(comptime T: type, data: []T, pos: usize, head_dim: usize, base: f32) void {
    const half = head_dim / 2;
    for (0..half) |i| {
        const theta = @as(f32, @floatFromInt(pos)) * std.math.pow(f32, base, -@as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim)));
        const cos_v: T = @floatCast(@cos(theta));
        const sin_v: T = @floatCast(@sin(theta));
        const a = data[2 * i];
        const b = data[2 * i + 1];
        data[2 * i] = a * cos_v - b * sin_v;
        data[2 * i + 1] = a * sin_v + b * cos_v;
    }
}

test "rope pairing NORM (llama-arch): pares consecutivos, A/B NEOX" {
    const gpa = std.testing.allocator;

    const head_dim = 64;
    const seq_len = 5;
    const num_heads = 2;
    const base: f32 = 500000.0; // Llama-3.2 real

    // Q/K densos [1, h, seq, d] con contenido NO simétrico (para que
    // ambos pairings den resultados distintos y el test discrimine).
    var rng = std.Random.Xoshiro256.init(11);
    var q_data: [num_heads * seq_len * head_dim]f32 = undefined;
    var k_data: [num_heads * seq_len * head_dim]f32 = undefined;
    for (&q_data, &k_data) |*q, *k| {
        q.* = rng.random().float(f32) * 2 - 1;
        k.* = rng.random().float(f32) * 2 - 1;
    }

    var Q = try Tensor(f32).alloc(gpa, &.{ 1, num_heads, seq_len, head_dim });
    defer Q.deinit();
    var K = try Tensor(f32).alloc(gpa, &.{ 1, num_heads, seq_len, head_dim });
    defer K.deinit();
    @memcpy(Q.data, &q_data);
    @memcpy(K.data, &k_data);

    // Copia de referencia: aplicar la fórmula NORM independiente por
    // posición absoluta (start_pos=0 ⇒ global_pos = pos).
    var q_ref: [num_heads * seq_len * head_dim]f32 = undefined;
    var k_ref: [num_heads * seq_len * head_dim]f32 = undefined;
    @memcpy(&q_ref, &q_data);
    @memcpy(&k_ref, &k_data);
    for (0..num_heads) |h| {
        for (0..seq_len) |pos| {
            const off = (h * seq_len + pos) * head_dim;
            refRopeNorm(f32, q_ref[off..][0..head_dim], pos, head_dim, base);
            refRopeNorm(f32, k_ref[off..][0..head_dim], pos, head_dim, base);
        }
    }

    // Act: applyRoPE default (debe ser NORM post-F1v7)
    try std.testing.expect(std.c.getenv("ZIG_AI_ROPE_NEOX") == null); // A/B no activo
    rope.applyRoPE(f32, &Q, &K, 0, head_dim, base, .auto);

    // Assert: bit-cercano a la referencia NORM (f32 con cos/sin idénticos
    // ⇒ tolerancia 0 salvo orden de ops: 1e-6)
    for (0..q_ref.len) |i| {
        try std.testing.expectApproxEqAbs(q_ref[i], Q.data[i], 1e-6);
        try std.testing.expectApproxEqAbs(k_ref[i], K.data[i], 1e-6);
    }

    // Y discrimina: con NEOX el resultado DIFIERE (salvo head_dim=2
    // degenerado — aquí 64). Verificamos que no somos NEOX por accidente:
    // recalculamos la copia con pairing NEOX y comprobamos que difiere.
    var q_neox: [num_heads * seq_len * head_dim]f32 = undefined;
    @memcpy(&q_neox, &q_data);
    const half = head_dim / 2;
    for (0..num_heads) |h| {
        for (0..seq_len) |pos| {
            const off = (h * seq_len + pos) * head_dim;
            for (0..half) |i| {
                const theta = @as(f32, @floatFromInt(pos)) * std.math.pow(f32, base, -@as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim)));
                const cos_v: f32 = @cos(theta);
                const sin_v: f32 = @sin(theta);
                const a = q_neox[off + i];
                const b = q_neox[off + i + half];
                q_neox[off + i] = a * cos_v - b * sin_v;
                q_neox[off + i + half] = a * sin_v + b * cos_v;
            }
        }
    }
    var neox_differs = false;
    for (0..q_ref.len) |i| {
        if (@abs(q_ref[i] - q_neox[i]) > 1e-3) neox_differs = true;
    }
    try std.testing.expect(neox_differs);
}
