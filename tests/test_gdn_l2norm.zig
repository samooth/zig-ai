//! 1.14 (lane-c): regresión fórmula GDN l2norm Q/K — FLA vs clamp.
//!
//! Oráculo: FLA l2norm.py:39/69 `y = x / sqrt(sum(x²) + eps)` (eps DENTRO
//! de la raíz). El código pre-1.14 usaba clamp `1/max(sqrt(sum),eps)`
//! (fiel al comentario "ggml_l2_norm" pero NO a FLA): divergen cuando
//! sum(x²) < eps² — q/k casi-nulos tras silu con beta baja (decode largo).
//!
//! Corre los tests INLINE de ssm.zig (ssm forward matches brute-force,
//! deltanet recurrence matches FLA naive, y el test 1.14) + GPU paridad
//! del kernel l2NormHeadsKernel contra la misma fórmula.
const std = @import("std");
const ssm = @import("ssm");

test {
    _ = ssm; // refAllDecls: corre los tests inline de ssm.zig
}

test "1.14 GPU l2NormHeadsKernel: FLA rsqrt(sum+eps), NO clamp" {
    const layer_kernels = @import("layer_kernels");
    const cudaz = @import("cudaz");
    cudaz.ensureContext() catch return error.SkipZigTest;
    _ = cudaz.cuDeviceGet(0) catch return error.SkipZigTest;

    const stream: cudaz.CUstream = try cudaz.cuStreamCreate(0);
    var lk = try layer_kernels.LayerKernels.init(stream);
    defer lk.deinit();

    const allocator = std.testing.allocator;
    const n: usize = 1;
    const key_dim: usize = 2; // 1 k-head × head_v_dim 2
    const d_inner: usize = 4;
    const qkv_dim: usize = key_dim * 2 + d_inner;
    const eps: f32 = 1e-6;

    const host = try allocator.alloc(f32, qkv_dim * n);
    defer allocator.free(host);
    // Q head (base 0) casi-nula: sum_sq = 2e-14 < eps² = 1e-12.
    // K head (base key_dim) normal: valores 0.5 (norma sqrt(0.5)≈0.707).
    for (0..qkv_dim) |i| host[i] = 0.5;
    host[0] = 1e-7;
    host[1] = 1e-7;

    const dev = try cudaz.cuMemAlloc(host.len * 4);
    defer cudaz.cuMemFree(dev);
    try cudaz.cuMemcpyHtoD(dev, @intFromPtr(host.ptr), host.len * 4);

    try lk.l2NormHeads(dev, n, qkv_dim, key_dim, 1, 2, eps);

    const back = try allocator.alloc(f32, host.len);
    defer allocator.free(back);
    try cudaz.cuStreamSynchronize(lk.stream);
    try cudaz.cuMemcpyDtoH(@intFromPtr(back.ptr), dev, host.len * 4);

    // FLA: q = 1e-7/sqrt(2e-14 + 1e-6) ≈ 1.0005e-4 → norma ≈ 1.4142e-4.
    // (eps DOMINA el denominador cuando sum ≪ eps: FLA no fuerza unidad)
    // Clamp (fórmula vieja): q = 1e-7/1e-6 = 0.1 → norma 0.1414 — FALLARÍA.
    var q_norm: f32 = 0;
    for (0..2) |i| q_norm += back[i] * back[i];
    try std.testing.expectApproxEqAbs(@as(f32, 1.4142136e-4), @sqrt(q_norm), 1e-6);

    // K normal: norma ≈ 1 (idéntico en ambas fórmulas — sum ≫ eps²).
    var k_norm: f32 = 0;
    for (key_dim..key_dim + 2) |i| k_norm += back[i] * back[i];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), @sqrt(k_norm), 1e-4);

    // Fórmula exacta FLA en f64 para el vector casi-nulo:
    const fla_scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, 2e-14) + 1e-6));
    try std.testing.expectApproxEqAbs(host[0] * fla_scale, back[0], 1e-6);
}
