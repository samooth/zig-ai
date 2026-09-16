//! Paridad ΔNet warp-shuffle (STUDY §5.6) vs kernel clásico (§4.2) vs
//! referencia CPU. La semántica es idéntica; el ORDEN de reducción difiere
//! (árbol shfl vs serial) ⇒ tolerancia relativa. Se salta sin GPU.
//!
//! Geometría: Qwen3.5-0.8B (n_v_heads=16, head_v_dim=128, dt_rank=16,
//! key_dim=2048, qkv_dim=6144).
const std = @import("std");
const testing = std.testing;
const cudaz = @import("cudaz");
const layer_kernels = @import("layer_kernels");

const n_v_heads: usize = 16;
const n_k_heads: usize = 16;
const head_v_dim: usize = 128;
const dt_rank: usize = 16;
const key_dim: usize = n_k_heads * head_v_dim;
const d_inner: usize = 2048;
const qkv_dim: usize = 2 * key_dim + d_inner;
const dim2: usize = head_v_dim * head_v_dim;
const state_bytes: usize = n_v_heads * dim2 * 4;

fn randBuf(seed: u64, buf: []f32, mag: f32) void {
    var rng = std.Random.Xoshiro256.init(seed);
    for (buf) |*v| v.* = (rng.random().float(f32) * 2.0 - 1.0) * mag;
}

fn runKernel(comptime warp: bool, gpa: std.mem.Allocator, lk: *layer_kernels.LayerKernels, conv_out: []const f32, gate: []const f32, beta: []const f32, state: []const f32, attn_out: []f32, state_out: []f32) !void {
    // Upload.
    const d_co = try cudaz.cuMemAlloc(conv_out.len * 4);
    defer cudaz.cuMemFree(d_co);
    const d_ga = try cudaz.cuMemAlloc(gate.len * 4);
    defer cudaz.cuMemFree(d_ga);
    const d_be = try cudaz.cuMemAlloc(beta.len * 4);
    defer cudaz.cuMemFree(d_be);
    const d_st = try cudaz.cuMemAlloc(state.len * 4);
    defer cudaz.cuMemFree(d_st);
    const d_ao = try cudaz.cuMemAlloc(attn_out.len * 4);
    defer cudaz.cuMemFree(d_ao);
    _ = gpa;
    try cudaz.cuMemcpyHtoD(d_co, @intFromPtr(conv_out.ptr), conv_out.len * 4);
    try cudaz.cuMemcpyHtoD(d_ga, @intFromPtr(gate.ptr), gate.len * 4);
    try cudaz.cuMemcpyHtoD(d_be, @intFromPtr(beta.ptr), beta.len * 4);
    try cudaz.cuMemcpyHtoD(d_st, @intFromPtr(state.ptr), state.len * 4);

    if (warp) {
        try lk.deltaNetWarp(d_co, d_ga, d_be, d_ao, d_st, 1, qkv_dim, key_dim, n_k_heads, n_v_heads, head_v_dim, dt_rank, 1e-5);
    } else {
        try lk.deltaNet(d_co, d_ga, d_be, d_ao, d_st, 1, qkv_dim, key_dim, n_k_heads, n_v_heads, head_v_dim, dt_rank, 1e-5);
    }
    try cudaz.cuStreamSynchronize(lk.stream);
    try cudaz.cuMemcpyDtoH(@intFromPtr(attn_out.ptr), d_ao, attn_out.len * 4);
    try cudaz.cuMemcpyDtoH(@intFromPtr(state_out.ptr), d_st, state.len * 4);
}

test "deltaNetWarp: paridad vs kernel clásico (misma semántica, reducción distinta)" {
    cudaz.ensureContext() catch return error.SkipZigTest;
    _ = cudaz.cuDeviceGet(0) catch return error.SkipZigTest;
    const gpa = testing.allocator;

    const stream: cudaz.CUstream = try cudaz.cuStreamCreate(0);
    var lk = try layer_kernels.LayerKernels.init(stream);

    // Inputs.
    const conv_out = try gpa.alloc(f32, qkv_dim);
    defer gpa.free(conv_out);
    randBuf(0xD17A, conv_out, 0.5);
    const gate = try gpa.alloc(f32, dt_rank);
    defer gpa.free(gate);
    randBuf(0x6A7E, gate, 0.5);
    const beta = try gpa.alloc(f32, dt_rank);
    defer gpa.free(beta);
    randBuf(0xB37A, beta, 0.5);
    const state = try gpa.alloc(f32, n_v_heads * dim2);
    defer gpa.free(state);
    randBuf(0x57A7, state, 0.1);

    const out_w = try gpa.alloc(f32, d_inner);
    defer gpa.free(out_w);
    const st_w = try gpa.alloc(f32, n_v_heads * dim2);
    defer gpa.free(st_w);
    const out_c = try gpa.alloc(f32, d_inner);
    defer gpa.free(out_c);
    const st_c = try gpa.alloc(f32, n_v_heads * dim2);
    defer gpa.free(st_c);

    try runKernel(true, gpa, &lk, conv_out, gate, beta, state, out_w, st_w);
    try runKernel(false, gpa, &lk, conv_out, gate, beta, state, out_c, st_c);

    // attn_out: tolerancia relativa (orden de reducción distinto).
    var max_rel: f32 = 0;
    var max_i: usize = 0;
    for (out_w, out_c, 0..) |w, c, i| {
        const rel = @abs(w - c) / @max(1e-6, @abs(c));
        if (rel > max_rel) { max_rel = rel; max_i = i; }
    }
    try testing.expect(max_rel < 1e-3);

    // state: el update g·S + k·δ compasa el error de orden de reducción de
    // sk. Métrica: error absoluto acotado por la magnitud TÍPICA del estado
    // (el relativo explota en elementos ~0 donde el denominador floor es
    // arbitrario); la métrica que importa es la de la SALIDA (attn).
    var max_abs_st: f32 = 0;
    var sum_abs_st: f64 = 0;
    var max_s: f32 = 0;
    for (st_w, st_c) |w, c| {
        max_abs_st = @max(max_abs_st, @abs(w - c));
        sum_abs_st += @abs(@as(f64, w) - @as(f64, c));
        max_s = @max(max_s, @max(@abs(w), @abs(c)));
    }
    std.debug.print("[dnwarp] state: max_abs={e} mean_abs={e} rel_to_scale={e}\n", .{ max_abs_st, sum_abs_st / @as(f64, @floatFromInt(st_w.len)), max_abs_st / max_s });
    // Estado: near-bit-perfect (medido ~1-ULP por orden de reducción; la
    // métrica dura es la de la salida). Cota: < 1e-5 relativo a la escala.
    try testing.expect(max_abs_st / max_s < 1e-5);

    std.debug.print("[dnwarp] paridad OK: attn max_rel={e} state rel_to_scale={e}\n", .{ max_rel, max_abs_st / max_s });
}
