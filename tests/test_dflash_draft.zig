//! 5.2 (lane-b1) — Tests del draft-model DFlash: helpers de referencia
//! (rmsNorm, lmHead row) + truncado p_min del driver. La construcción del
//! DflashDraftModel completo (GGUF sidecar real) se valida en integración
//! con --model-draft (gate E2E); aquí la maquinaria CPU con datos sintéticos.

const std = @import("std");
const spec = @import("speculative");

test "draftSidecarGreedy: truncado p_min con logits sintéticos" {
    const allocator = std.testing.allocator;
    var drv = spec.SpecDriver.init(allocator, .{ .p_min = 0.5 });
    defer drv.deinit();

    // 3 posiciones × vocab 4: pos0 conf alta, pos1 conf media, pos2 conf baja
    const vocab = 4;
    // softmax([9,0,0,0]) ≈ p=0.999 (pasa), [0,3,0,0] p≈0.64 (pasa con p_min 0.5),
    // [1,1.2,1,1] p≈0.28 (falla p_min 0.5 ⇒ cola corta en pos 2)
    const logits = [_]f32{
        9, 0,   0, 0,
        0, 3,   0, 0,
        1, 1.2, 1, 1,
    };
    var out: [3]u32 = undefined;
    const n = try drv.draftSidecarGreedy(&logits, vocab, &out);
    // pos0 y pos1 aceptadas; pos2 truncada por p_min
    try std.testing.expectEqual(@as(usize, 2), n);
    try std.testing.expectEqual(@as(u32, 0), out[0]);
    try std.testing.expectEqual(@as(u32, 1), out[1]);
}

test "draftSidecarGreedy: conf alta en todas ⇒ cola completa" {
    const allocator = std.testing.allocator;
    var drv = spec.SpecDriver.init(allocator, .{ .p_min = 0.1 });
    defer drv.deinit();

    const vocab = 2;
    const logits = [_]f32{ 8, 0, 8, 0, 8, 0 };
    var out: [3]u32 = undefined;
    const n = try drv.draftSidecarGreedy(&logits, vocab, &out);
    try std.testing.expectEqual(@as(usize, 3), n);
    try std.testing.expectEqual(@as(u32, 0), out[2]);
}

test "draftSidecarGreedy: primera posición SIEMPRE se emite (i>0 gate)" {
    const allocator = std.testing.allocator;
    var drv = spec.SpecDriver.init(allocator, .{ .p_min = 0.99 });
    defer drv.deinit();

    // p_min absurdo: aún así pos 0 sale (gate i>0 — el ancla del bloque
    // siempre produce al menos un candidato; el oráculo trunca en n_min
    // con descarte completo, pero el driver devuelve lo que hay).
    const vocab = 2;
    const logits = [_]f32{ 0.1, 0.1, 0.1, 0.1 };
    var out: [2]u32 = undefined;
    const n = try drv.draftSidecarGreedy(&logits, vocab, &out);
    try std.testing.expectEqual(@as(usize, 1), n);
}
