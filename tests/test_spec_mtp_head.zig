//! Test estructural de la cabeza MTP (lane-c C4-tune T2).
//!
//! Requiere SPEC_MTP_MODEL apuntando a un GGUF con cabeza nextn (p.ej. el
//! 27B NEO-MTP). Sin la var se salta (CI sin modelo grande).
//! Valida el layout que asume fuseAndProject: dequantToF32Transposed produce
//! [out][in] row-major para eh_proj — si alguien refactoriza el dequant o
//! cambia el orden concat [e_norm; h_norm], esto salta antes que una
//! aceptación misteriosamente baja.
const std = @import("std");
const spec = @import("speculative");
const gguf = @import("gguf");
const gguf_model = @import("gguf_model");

test "MtpHead: layout eh_proj y normas coherentes con fuseAndProject" {
    const io = std.Io.Threaded.global_single_threaded.io();
    const path_env = std.c.getenv("SPEC_MTP_MODEL") orelse return error.SkipZigTest;
    const allocator = std.heap.c_allocator;

    var model = try gguf_model.GgufModel.load(io, allocator, std.mem.span(path_env));
    defer model.deinit();

    const info = spec.detectMtp(&model.file, model.config.block_count) orelse return error.SkipZigTest;
    var head = try spec.MtpHead.load(allocator, &model.file, info, model.config.embedding_length);
    defer head.deinit(allocator);

    const E = head.n_embd;

    // ── 1) Normas: finitas y no degeneradas ──
    for ([_][]f32{ head.enorm, head.hnorm, head.shared_head_norm }) |g| {
        try std.testing.expectEqual(@as(usize, E), g.len);
        var ssq: f64 = 0;
        for (g) |v| {
            try std.testing.expect(!std.math.isNan(v));
            ssq += @as(f64, v) * v;
        }
        // RMSNorm gamma típico ≈ sqrt(E)·media(v²)^... basta exigir no-cero.
        try std.testing.expect(ssq > 0);
    }

    // ── 2) eh_proj: dequant transpuesto ≡ dequant directo + transpose manual ──
    const tinfo = head.eh_proj.info;
    const w_t = try allocator.alloc(f32, E * 2 * E);
    defer allocator.free(w_t);
    head.eh_proj.dequantToF32Transposed(w_t);

    const direct = try allocator.alloc(f32, tinfo.numel());
    defer allocator.free(direct);
    try gguf.dequantTensor(tinfo, model.file.tensorData(tinfo), direct);

    // direct está en orden de storage [in-fastest]: elemento (i_in, j_out)
    // en flat j*2E+i. El transpuesto debe ser w_t[j*2E+i] == flat[j*2E+i]
    // cuando d0==in (identidad de posiciones) — pero el CONTENIDO debe
    // coincidir con la referencia independiente.
    var checked: usize = 0;
    const stride: usize = @max(1, tinfo.numel() / 4096);
    while (checked < tinfo.numel()) : (checked += stride) {
        try std.testing.expectEqual(direct[checked], w_t[checked]);
    }

    // ── 3) fuseAndProject vs referencia directa (orden [e;h]) ──
    const xr = try allocator.alloc(f32, 2 * E);
    defer allocator.free(xr);
    const eo = try allocator.alloc(f32, E);
    defer allocator.free(eo);
    const ho = try allocator.alloc(f32, E);
    defer allocator.free(ho);
    const out_impl = try allocator.alloc(f32, E);
    defer allocator.free(out_impl);
    const out_ref = try allocator.alloc(f32, E);
    defer allocator.free(out_ref);

    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    for (eo) |*v| v.* = rng.float(f32) - 0.5;
    for (ho) |*v| v.* = rng.float(f32) - 0.5;

    // referencia: rmsnorm manual + concat[e;h] + y_j = Σ_i W_eff[j][i]·x_i
    rmsRef(eo, head.enorm);
    rmsRef(ho, head.hnorm);
    @memcpy(xr[0..E], eo);
    @memcpy(xr[E .. 2 * E], ho);
    for (out_ref, 0..) |*oj, jj| {
        const row = w_t[jj * 2 * E ..][0 .. 2 * E];
        var acc: f64 = 0;
        for (row, xr) |wv, xv| acc += @as(f64, wv) * xv;
        oj.* = @floatCast(acc);
    }

    // implementación: MISMA entrada SIN normalizar (normaliza dentro)
    var prng2 = std.Random.DefaultPrng.init(7);
    const rng2 = prng2.random();
    for (eo) |*v| v.* = rng2.float(f32) - 0.5;
    for (ho) |*v| v.* = rng2.float(f32) - 0.5;
    head.fuseAndProject(w_t, eo, ho, out_impl, xr);

    for (out_impl, out_ref) |a, b| {
        const diff = @abs(a - b);
        const scale = @max(@abs(a), @abs(b), 1.0);
        try std.testing.expect(diff / scale < 1e-4);
    }
}

// ── helpers locales (evitan importar módulos extra al test) ─────────────────

fn rmsRef(x: []f32, gamma: []const f32) void {
    var ssq: f64 = 0;
    for (x) |v| ssq += @as(f64, v) * v;
    const inv: f32 = @floatCast(1.0 / @sqrt(ssq / @as(f64, @floatFromInt(x.len)) + 1e-5));
    for (x, gamma) |*v, gm| v.* *= inv * gm;
}
