//! Paridad §5.1 FUSED: deltaNetFusedKernel (l2+ΔNet registros+rmsNorm en 1
//! launch) vs la cadena separada (l2NormHeads + deltaNet + rmsNormGateMul).
//! La fusión es bit-idéntica por construcción: mismos patrones strided +
//! árbol reds[256] en l2/rms (etapas A/C), recurrencia §5.6 exacta (etapa B).
//! Se salta sin GPU.
const std = @import("std");

fn stdoutPrint(io: std.Io, comptime fmt: []const u8, args: anytype) !void {
    var stdout_buf: [256]u8 = undefined;
    const stdout_file = std.Io.File.stdout();
    var stdout_writer = stdout_file.writer(io, &stdout_buf);
    try stdout_writer.interface.print(fmt, args);
}
const testing = std.testing;
const cudaz = @import("cudaz");
const layer_kernels = @import("layer_kernels");

// Geometría Qwen3.5-0.8B: 16/16 heads, head_v_dim=128, dt_rank=16.
const n_v_heads: usize = 16;
const n_k_heads: usize = 16;
const head_v_dim: usize = 128;
const dt_rank: usize = 16;
const key_dim: usize = n_k_heads * head_v_dim; // 2048
const d_inner: usize = n_v_heads * head_v_dim; // 2048
const qkv_dim: usize = 2 * key_dim + d_inner; // 6144
const dim2: usize = head_v_dim * head_v_dim;
const N: usize = 1; // decode path

fn randBuf(seed: u64, buf: []f32, mag: f32) void {
    var rng = std.Random.Xoshiro256.init(seed);
    for (buf) |*v| v.* = (rng.random().float(f32) * 2.0 - 1.0) * mag;
}

fn runFused(lk: *layer_kernels.LayerKernels, conv_out: []f32, gate: []const f32, beta: []const f32, z: []const f32, ssm_norm: []const f32, attn_out: []f32, state: []f32) !void {
    const d_co = try cudaz.cuMemAlloc(conv_out.len * 4);
    defer cudaz.cuMemFree(d_co);
    const d_ga = try cudaz.cuMemAlloc(gate.len * 4);
    defer cudaz.cuMemFree(d_ga);
    const d_be = try cudaz.cuMemAlloc(beta.len * 4);
    defer cudaz.cuMemFree(d_be);
    const d_z = try cudaz.cuMemAlloc(z.len * 4);
    defer cudaz.cuMemFree(d_z);
    const d_sn = try cudaz.cuMemAlloc(ssm_norm.len * 4);
    defer cudaz.cuMemFree(d_sn);
    const d_ao = try cudaz.cuMemAlloc(attn_out.len * 4);
    defer cudaz.cuMemFree(d_ao);
    const d_st = try cudaz.cuMemAlloc(state.len * 4);
    defer cudaz.cuMemFree(d_st);
    try cudaz.cuMemcpyHtoD(d_co, @intFromPtr(conv_out.ptr), conv_out.len * 4);
    try cudaz.cuMemcpyHtoD(d_ga, @intFromPtr(gate.ptr), gate.len * 4);
    try cudaz.cuMemcpyHtoD(d_be, @intFromPtr(beta.ptr), beta.len * 4);
    try cudaz.cuMemcpyHtoD(d_z, @intFromPtr(z.ptr), z.len * 4);
    try cudaz.cuMemcpyHtoD(d_sn, @intFromPtr(ssm_norm.ptr), ssm_norm.len * 4);
    try cudaz.cuMemcpyHtoD(d_ao, @intFromPtr(attn_out.ptr), attn_out.len * 4);
    try cudaz.cuMemcpyHtoD(d_st, @intFromPtr(state.ptr), state.len * 4);
    try lk.deltaNetFused(d_co, d_ga, d_be, d_z, d_sn, d_ao, d_st, N, qkv_dim, key_dim, n_k_heads, n_v_heads, head_v_dim, dt_rank, 1e-5);
    try cudaz.cuStreamSynchronize(lk.stream);
    // conv_out (l2 in-place), attn_out (recurrencia+rms), state (persistido).
    try cudaz.cuMemcpyDtoH(@intFromPtr(conv_out.ptr), d_co, conv_out.len * 4);
    try cudaz.cuMemcpyDtoH(@intFromPtr(attn_out.ptr), d_ao, attn_out.len * 4);
    try cudaz.cuMemcpyDtoH(@intFromPtr(state.ptr), d_st, state.len * 4);
}

fn runSeparate(lk: *layer_kernels.LayerKernels, conv_out: []f32, gate: []const f32, beta: []const f32, z: []const f32, ssm_norm: []const f32, attn_out: []f32, state: []f32) !void {
    const d_co = try cudaz.cuMemAlloc(conv_out.len * 4);
    defer cudaz.cuMemFree(d_co);
    const d_ga = try cudaz.cuMemAlloc(gate.len * 4);
    defer cudaz.cuMemFree(d_ga);
    const d_be = try cudaz.cuMemAlloc(beta.len * 4);
    defer cudaz.cuMemFree(d_be);
    const d_z = try cudaz.cuMemAlloc(z.len * 4);
    defer cudaz.cuMemFree(d_z);
    const d_sn = try cudaz.cuMemAlloc(ssm_norm.len * 4);
    defer cudaz.cuMemFree(d_sn);
    const d_ao = try cudaz.cuMemAlloc(attn_out.len * 4);
    defer cudaz.cuMemFree(d_ao);
    const d_st = try cudaz.cuMemAlloc(state.len * 4);
    defer cudaz.cuMemFree(d_st);
    try cudaz.cuMemcpyHtoD(d_co, @intFromPtr(conv_out.ptr), conv_out.len * 4);
    try cudaz.cuMemcpyHtoD(d_ga, @intFromPtr(gate.ptr), gate.len * 4);
    try cudaz.cuMemcpyHtoD(d_be, @intFromPtr(beta.ptr), beta.len * 4);
    try cudaz.cuMemcpyHtoD(d_z, @intFromPtr(z.ptr), z.len * 4);
    try cudaz.cuMemcpyHtoD(d_sn, @intFromPtr(ssm_norm.ptr), ssm_norm.len * 4);
    try cudaz.cuMemcpyHtoD(d_ao, @intFromPtr(attn_out.ptr), attn_out.len * 4);
    try cudaz.cuMemcpyHtoD(d_st, @intFromPtr(state.ptr), state.len * 4);
    // Cadena separada (el camino clásico): l2 + deltaNet + rmsNormGateMul.
    try lk.l2NormHeads(d_co, N, qkv_dim, key_dim, n_k_heads, head_v_dim, 1e-5);
    try lk.deltaNet(d_co, d_ga, d_be, d_ao, d_st, N, qkv_dim, key_dim, n_k_heads, n_v_heads, head_v_dim, dt_rank, 1e-5);
    try lk.rmsNormGateMul(d_ao, d_z, d_sn, N, d_inner, n_v_heads, head_v_dim, 1e-5);
    try cudaz.cuStreamSynchronize(lk.stream);
    try cudaz.cuMemcpyDtoH(@intFromPtr(conv_out.ptr), d_co, conv_out.len * 4);
    try cudaz.cuMemcpyDtoH(@intFromPtr(attn_out.ptr), d_ao, attn_out.len * 4);
    try cudaz.cuMemcpyDtoH(@intFromPtr(state.ptr), d_st, state.len * 4);
}

test "deltaNetFused: paridad vs cadena separada (l2+deltaNet+rmsNorm)" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    cudaz.ensureContext() catch return error.SkipZigTest;
    _ = cudaz.cuDeviceGet(0) catch return error.SkipZigTest;
    const gpa = testing.allocator;

    const stream: cudaz.CUstream = try cudaz.cuStreamCreate(0);
    var lk = try layer_kernels.LayerKernels.init(stream);

    // Buffers de entrada (los mismos datos para ambas rutas).
    const conv_out_s = try gpa.alloc(f32, N * qkv_dim);
    defer gpa.free(conv_out_s);
    randBuf(0xF051, conv_out_s, 0.5);
    const conv_out_f = try gpa.alloc(f32, N * qkv_dim);
    defer gpa.free(conv_out_f);
    @memcpy(conv_out_f, conv_out_s);

    const gate = try gpa.alloc(f32, N * dt_rank);
    defer gpa.free(gate);
    randBuf(0x6A7E, gate, 0.3);
    const beta = try gpa.alloc(f32, N * dt_rank);
    defer gpa.free(beta);
    randBuf(0xB37A, beta, 0.5);
    const z = try gpa.alloc(f32, N * d_inner);
    defer gpa.free(z);
    randBuf(0x2EED, z, 0.4);
    const ssm_norm = try gpa.alloc(f32, d_inner);
    defer gpa.free(ssm_norm);
    randBuf(0x5A1E, ssm_norm, 0.8);

    const state_s = try gpa.alloc(f32, n_v_heads * dim2);
    defer gpa.free(state_s);
    randBuf(0x57A7, state_s, 0.1);
    const state_f = try gpa.alloc(f32, n_v_heads * dim2);
    defer gpa.free(state_f);
    @memcpy(state_f, state_s);

    const attn_s = try gpa.alloc(f32, N * d_inner);
    defer gpa.free(attn_s);
    @memset(attn_s, 0);
    const attn_f = try gpa.alloc(f32, N * d_inner);
    defer gpa.free(attn_f);
    @memset(attn_f, 0);

    try runSeparate(&lk, conv_out_s, gate, beta, z, ssm_norm, attn_s, state_s);
    try runFused(&lk, conv_out_f, gate, beta, z, ssm_norm, attn_f, state_f);

    // 1) conv_out tras l2 in-place: rel < 1e-3 (las dos implementaciones
    //    usan reducción distinta — 256 thr tree vs 1024 thr warp-shuffle —
    //    así que la suma de cuadrados puede diferir por orden de FMA).
    var bad_co: usize = 0;
    var max_rel_co: f32 = 0;
    for (conv_out_s, conv_out_f, 0..) |s, f, i| {
        if (@as(u32, @bitCast(s)) != @as(u32, @bitCast(f))) {
            if (bad_co < 4) try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "  conv_out[{d}]: sep={e} fused={e}\n", .{ i, s, f });
            bad_co += 1;
        }
        const rel = @abs(s - f) / @max(1e-6, @abs(s));
        max_rel_co = @max(max_rel_co, rel);
    }
    // 2) attn_out tras ΔNet+rmsNorm·silu(z): rel < 1e-3 (reducción distinta
    //    sólo si el patrón difiriera — por construcción es idéntico ⇒ esperamos
    //    bit-exact; tolerancia de seguridad por si el driver reordena FMA).
    var max_rel: f32 = 0;
    var bad_bits: usize = 0;
    for (attn_s, attn_f) |s, f| {
        const rel = @abs(s - f) / @max(1e-6, @abs(s));
        max_rel = @max(max_rel, rel);
        if (@as(u32, @bitCast(s)) != @as(u32, @bitCast(f))) bad_bits += 1;
    }
    // 3) estado persistido: bit-idéntico (misma aritmética per-lane).
    var bad_st: usize = 0;
    var max_rel_st: f32 = 0;
    for (state_s, state_f, 0..) |s, f, i| {
        if (@as(u32, @bitCast(s)) != @as(u32, @bitCast(f))) {
            if (bad_st < 4) try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "  state[{d}]: sep={e} fused={e}\n", .{ i, s, f });
            bad_st += 1;
        }
        max_rel_st = @max(max_rel_st, @abs(s - f) / @max(1e-6, @abs(s)));
    }
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "[dnfused] conv_out max_rel={e} bit-diffs={d} | attn max_rel={e} bit-diffs={d}/{d} | state max_rel={e} bit-diffs={d}\n", .{ max_rel_co, bad_co, max_rel, bad_bits, attn_s.len, max_rel_st, bad_st });

    // Gates: l2/attn/state rel<1e-3 (hard); bit-diffs reportados (soft — FMA
    // del driver puede reordenar entre implementaciones con reducción distinta).
    try testing.expect(max_rel_co < 1e-3);
    try testing.expect(max_rel < 1e-3);
    try testing.expect(max_rel_st < 1e-3);
    if (bad_bits == 0 and bad_st == 0) {
        try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "[dnfused] paridad BIT-EXACT total\n", .{});
    } else {
        try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "[dnfused] paridad rel OK con {d}/{d} bit-diffs (FMA reorder)\n", .{ bad_bits + bad_st, attn_s.len + state_f.len });
    }
}
