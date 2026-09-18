//! Prefill chunked ΔNet paridad (STUDY §5.2) vs referencia per-token.
//!
//! Valida que prefillDeltaNetChunkKernel produce la misma salida que el bucle
//! per-token deltaNet para n=15/30/64/75 con K=64. El kernel lee conv_out
//! intercalado y escribe attn_out [n, d_inner]; el estado es persistente INOUT.
//!
//! FIX v5: kernel usa solo el índice local `t` (no t_start + t) para evitar
//! el doble offset del caller. Añadido n_tokens runtime para evitar OOB
//! cuando n < K.
//!
//! Geometría: Qwen3.5-0.8B (n_v_heads=16, head_v_dim=128, dt_rank=16,
//! key_dim=2048, qkv_dim=6144, d_inner=2048).

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
const debugz = @import("debug");

const n_v_heads: usize = 16;
const n_k_heads: usize = 16;
const head_v_dim: usize = 128;
const dt_rank: usize = 16;
const key_dim: usize = n_k_heads * head_v_dim;
const d_inner: usize = 2048;
const qkv_dim: usize = 2 * key_dim + d_inner;
const dim2: usize = head_v_dim * head_v_dim;
const S_v: usize = head_v_dim;

fn randBuf(seed: u64, buf: []f32, mag: f32) void {
    var rng = std.Random.Xoshiro256.init(seed);
    for (buf) |*v| v.* = (rng.random().float(f32) * 2.0 - 1.0) * mag;
}

/// CPU per-token reference (matches ssm.zig deltaNetRecurrence).
fn refPerToken(
    conv_out: []const f32,
    gate: []const f32,
    beta: []const f32,
    state: []f32,
    attn_out: []f32,
    n: usize,
) void {
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_v_dim)));
    for (0..n) |t| {
        for (0..n_v_heads) |hv| {
            const hk = hv / (n_v_heads / n_k_heads);
            const g = std.math.exp(gate[t * dt_rank + hv]);
            const b = beta[t * dt_rank + hv];

            const q_base = t * qkv_dim + hk * head_v_dim;
            const k_base = t * qkv_dim + key_dim + hk * head_v_dim;
            const v_base = t * qkv_dim + 2 * key_dim + hv * head_v_dim;
            const s_base = hv * head_v_dim * head_v_dim;

            // S *= exp(g)
            for (0..head_v_dim * head_v_dim) |i| state[s_base + i] *= g;

            // sk[j] = sum_i S[i,j] * k[i]
            var d_buf: [head_v_dim]f32 = undefined;
            for (0..head_v_dim) |j| {
                var sk: f32 = 0;
                for (0..head_v_dim) |i| {
                    sk += state[s_base + i * head_v_dim + j] * conv_out[k_base + i];
                }
                d_buf[j] = b * (conv_out[v_base + j] - sk);
            }

            // S[i,j] += k[i] * d[j]
            for (0..head_v_dim) |i| {
                const kv = conv_out[k_base + i];
                for (0..head_v_dim) |j| {
                    state[s_base + i * head_v_dim + j] += kv * d_buf[j];
                }
            }

            // o[j] = sum_i S[i,j] * q[i] * scale
            for (0..head_v_dim) |j| {
                var o: f32 = 0;
                for (0..head_v_dim) |i| {
                    o += state[s_base + i * head_v_dim + j] * conv_out[q_base + i];
                }
                attn_out[t * d_inner + hv * head_v_dim + j] = o * scale;
            }
        }
    }
}

/// Combined abs+rel tolerance check.
fn checkDiff(actual: []const f32, expected: []const f32, atol: f32, rtol: f32, name: []const u8) !f32 {
    var max_combined: f32 = 0;
    for (actual, expected) |a, e| {
        const d = @abs(a - e);
        const tol = atol + rtol * @abs(e);
        const combined = d / tol;
        if (combined > max_combined) max_combined = combined;
    }
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "  {s}: max_combined={d:.3} (atol={e} rtol={e})\n", .{ name, max_combined, atol, rtol });
    return max_combined;
}

/// Run the chunked kernel directly via cuModuleGetFunction.
/// For n <= K, passes n_tokens=n so the kernel processes exactly n tokens.
fn gpuChunkedDirect(
    lk: *layer_kernels.LayerKernels,
    conv_out: []const f32,
    gate: []const f32,
    beta: []const f32,
    state_in: []const f32,
    n: usize,
    K: usize,
) !struct { attn_out: []f32, state_out: []f32, _allocator: std.mem.Allocator } {
    // Buffers sized for max(K, n) to be safe.
    const buf_size = @max(K, n);
    const d_conv = try cudaz.cuMemAlloc(buf_size * qkv_dim * 4);
    defer cudaz.cuMemFree(d_conv);
    const d_gate = try cudaz.cuMemAlloc(buf_size * dt_rank * 4);
    defer cudaz.cuMemFree(d_gate);
    const d_beta = try cudaz.cuMemAlloc(buf_size * dt_rank * 4);
    defer cudaz.cuMemFree(d_beta);
    const d_state = try cudaz.cuMemAlloc(n_v_heads * dim2 * 4);
    defer cudaz.cuMemFree(d_state);
    const d_out = try cudaz.cuMemAlloc(buf_size * d_inner * 4);
    defer cudaz.cuMemFree(d_out);

    // Use page_allocator for the zero buffer to match the returned allocator.
    const gpa = std.heap.page_allocator;
    const zero_out = try gpa.alloc(f32, buf_size * d_inner);
    defer gpa.free(zero_out);
    @memset(zero_out, 0);
    try cudaz.cuMemcpyHtoD(d_conv, @intFromPtr(conv_out.ptr), n * qkv_dim * 4);
    try cudaz.cuMemcpyHtoD(d_gate, @intFromPtr(gate.ptr), n * dt_rank * 4);
    try cudaz.cuMemcpyHtoD(d_beta, @intFromPtr(beta.ptr), n * dt_rank * 4);
    try cudaz.cuMemcpyHtoD(d_state, @intFromPtr(state_in.ptr), n_v_heads * dim2 * 4);
    try cudaz.cuMemcpyHtoD(d_out, @intFromPtr(zero_out.ptr), buf_size * d_inner * 4);

    var scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(S_v)));
    var q_off: c_int = 0;
    var k_off: c_int = @intCast(key_dim);
    var v_off: c_int = @intCast(2 * key_dim);
    var qkv_stride: c_int = @intCast(qkv_dim);
    var dt_stride: c_int = @intCast(dt_rank);
    var d_inner_c: c_int = @intCast(d_inner);
    var n_v_heads_c: c_int = @intCast(n_v_heads);
    var n_k_heads_c: c_int = @intCast(n_k_heads);
    var hv_dim_c: c_int = @intCast(head_v_dim);
    var t_start: c_int = 0;
    var n_tokens: c_int = @intCast(n);
    var d_state_var: c_ulong = @intCast(d_state);
    var co_ptr: c_ulong = @intCast(d_conv);
    var ao_ptr: c_ulong = @intCast(d_out);
    var g_ptr: c_ulong = @intCast(d_gate);
    var b_ptr: c_ulong = @intCast(d_beta);

    var kp = [_]?*anyopaque{
        @ptrCast(&co_ptr),      @ptrCast(&q_off),     @ptrCast(&k_off),    @ptrCast(&v_off),     @ptrCast(&qkv_stride),
        @ptrCast(&g_ptr),       @ptrCast(&dt_stride), @ptrCast(&b_ptr),    @ptrCast(&dt_stride), @ptrCast(&d_state_var),
        @ptrCast(&ao_ptr),      @ptrCast(&d_inner_c), @ptrCast(&t_start),  @ptrCast(&scale),     @ptrCast(&n_v_heads_c),
        @ptrCast(&n_k_heads_c), @ptrCast(&hv_dim_c),  @ptrCast(&n_tokens),
    };

    const grid_z: u32 = @intCast((S_v + 3) / 4);
    const n_v_heads_u32: u32 = @intCast(n_v_heads);
    // [coordinador] fix mínimo: obtener func del cubin de prefill ANTES del
    // launch (la línea faltaba en el WIP). Vía loadPrefillModule (layer_kernels).
    // 1.11 (lane-c): K=512 → wrapper _K512 (fused CH ubatch-wide).
    const func_name: [:0]const u8 = if (K == 512) "prefillDeltaNetChunk_nkda_K512" else if (K == 64) "prefillDeltaNetChunk_nkda_K64" else "prefillDeltaNetChunk_nkda_K128";
    const func = try cudaz.cuModuleGetFunction(try layer_kernels.loadPrefillModule(), func_name);
    try cudaz.cuLaunchKernel(func, n_v_heads_u32, 1, grid_z, 32, 4, 1, 0, lk.stream, @ptrCast(&kp), null);
    try cudaz.cuStreamSynchronize(lk.stream);

    const attn_out = try gpa.alloc(f32, n * d_inner);
    const state_out = try gpa.alloc(f32, n_v_heads * dim2);
    try cudaz.cuMemcpyDtoH(@intFromPtr(attn_out.ptr), d_out, n * d_inner * 4);
    try cudaz.cuMemcpyDtoH(@intFromPtr(state_out.ptr), d_state, n_v_heads * dim2 * 4);
    return .{ .attn_out = attn_out, .state_out = state_out, ._allocator = gpa };
}

/// GPU per-token reference via deltaNetWarp (used for n<K tail fallback).
fn gpuPerToken(
    lk: *layer_kernels.LayerKernels,
    conv_out: []const f32,
    gate: []const f32,
    beta: []const f32,
    state_in: []const f32,
    n: usize,
) !struct { attn_out: []f32, state_out: []f32, _allocator: std.mem.Allocator } {
    const gpa = std.heap.page_allocator;

    // Allocate device buffers
    const d_state = try cudaz.cuMemAlloc(n_v_heads * dim2 * 4);
    defer cudaz.cuMemFree(d_state);
    const d_out = try cudaz.cuMemAlloc(n * d_inner * 4);
    defer cudaz.cuMemFree(d_out);

    try cudaz.cuMemcpyHtoD(d_state, @intFromPtr(state_in.ptr), n_v_heads * dim2 * 4);

    for (0..n) |t| {
        const d_co = try cudaz.cuMemAlloc(qkv_dim * 4);
        defer cudaz.cuMemFree(d_co);
        const d_ga = try cudaz.cuMemAlloc(dt_rank * 4);
        defer cudaz.cuMemFree(d_ga);
        const d_be = try cudaz.cuMemAlloc(dt_rank * 4);
        defer cudaz.cuMemFree(d_be);
        const d_ao = try cudaz.cuMemAlloc(d_inner * 4);
        defer cudaz.cuMemFree(d_ao);

        try cudaz.cuMemcpyHtoD(d_co, @intFromPtr(conv_out.ptr + t * qkv_dim), qkv_dim * 4);
        try cudaz.cuMemcpyHtoD(d_ga, @intFromPtr(gate.ptr + t * dt_rank), dt_rank * 4);
        try cudaz.cuMemcpyHtoD(d_be, @intFromPtr(beta.ptr + t * dt_rank), dt_rank * 4);

        try lk.deltaNetWarp(d_co, d_ga, d_be, d_ao, d_state, 1, qkv_dim, key_dim, n_k_heads, n_v_heads, head_v_dim, dt_rank, 1e-5);
        try cudaz.cuStreamSynchronize(lk.stream);
    }

    const attn_out = try gpa.alloc(f32, n * d_inner);
    const state_out = try gpa.alloc(f32, n_v_heads * dim2);
    try cudaz.cuMemcpyDtoH(@intFromPtr(attn_out.ptr), d_out, n * d_inner * 4);
    try cudaz.cuMemcpyDtoH(@intFromPtr(state_out.ptr), d_state, n_v_heads * dim2 * 4);
    return .{ .attn_out = attn_out, .state_out = state_out, ._allocator = gpa };
}

test "prefillChunk K=64 n=64 (1 chunk exacto): paridad vs per-token" {
    cudaz.ensureContext() catch return error.SkipZigTest;
    _ = cudaz.cuDeviceGet(0) catch return error.SkipZigTest;
    const gpa = testing.allocator;

    const stream: cudaz.CUstream = try cudaz.cuStreamCreate(0);
    var lk = try layer_kernels.LayerKernels.init(stream);

    const n: usize = 64;
    const K: usize = 64;

    const conv_out = try gpa.alloc(f32, n * qkv_dim);
    defer gpa.free(conv_out);
    randBuf(0xD17A, conv_out, 0.05);
    const gate = try gpa.alloc(f32, n * dt_rank);
    defer gpa.free(gate);
    {
        var rng = std.Random.Xoshiro256.init(0x6A7E);
        for (gate) |*v| v.* = -rng.random().float(f32) * 2.0;
    }
    const beta = try gpa.alloc(f32, n * dt_rank);
    defer gpa.free(beta);
    {
        var rng = std.Random.Xoshiro256.init(0xB37A);
        for (beta) |*v| v.* = rng.random().float(f32) * 0.5 + 0.5;
    }
    const state_ref = try gpa.alloc(f32, n_v_heads * dim2);
    defer gpa.free(state_ref);
    randBuf(0x57A7, state_ref, 0.001);
    const out_ref = try gpa.alloc(f32, n * d_inner);
    defer gpa.free(out_ref);

    @memset(out_ref, 0);
    refPerToken(conv_out, gate, beta, state_ref, out_ref, n);

    const result = try gpuChunkedDirect(&lk, conv_out, gate, beta, state_ref, n, K);
    // [coordinador] FIX cross-allocator: gpuChunkedDirect aloca con
    // page_allocator (su gpa interno); liberar con testing.allocator es
    // "Invalid free" del DebugAllocator ⇒ ABRT en los 4 tests. Liberar
    // con el allocator que devuelve el helper (_allocator).
    defer result._allocator.free(result.attn_out);
    defer result._allocator.free(result.state_out);

    const out_c = try checkDiff(result.attn_out, out_ref, 1e-3, 1e-3, "attn_out");
    const st_c = try checkDiff(result.state_out, state_ref, 2e-3, 2e-3, "state");
    try testing.expect(out_c < 1.0);
    try testing.expect(st_c < 1.0);
}

test "prefillChunk K=64 n=15 (n<K, n_tokens=runtime check): paridad vs per-token" {
    cudaz.ensureContext() catch return error.SkipZigTest;
    _ = cudaz.cuDeviceGet(0) catch return error.SkipZigTest;
    const gpa = testing.allocator;

    const stream: cudaz.CUstream = try cudaz.cuStreamCreate(0);
    var lk = try layer_kernels.LayerKernels.init(stream);

    const n: usize = 15;
    const K: usize = 64;

    const conv_out = try gpa.alloc(f32, n * qkv_dim);
    defer gpa.free(conv_out);
    randBuf(0xD17A, conv_out, 0.05);
    const gate = try gpa.alloc(f32, n * dt_rank);
    defer gpa.free(gate);
    {
        var rng = std.Random.Xoshiro256.init(0x6A7E);
        for (gate) |*v| v.* = -rng.random().float(f32) * 2.0;
    }
    const beta = try gpa.alloc(f32, n * dt_rank);
    defer gpa.free(beta);
    {
        var rng = std.Random.Xoshiro256.init(0xB37A);
        for (beta) |*v| v.* = rng.random().float(f32) * 0.5 + 0.5;
    }
    const state_ref = try gpa.alloc(f32, n_v_heads * dim2);
    defer gpa.free(state_ref);
    randBuf(0x57A7, state_ref, 0.001);
    const out_ref = try gpa.alloc(f32, n * d_inner);
    defer gpa.free(out_ref);

    @memset(out_ref, 0);
    refPerToken(conv_out, gate, beta, state_ref, out_ref, n);

    // n=15 < K=64, but with n_tokens=15 runtime check, the kernel processes
    // exactly 15 tokens. Buffers are sized for max(K,n)=K=64 to be safe.
    const result = try gpuChunkedDirect(&lk, conv_out, gate, beta, state_ref, n, K);
    // [coordinador] FIX cross-allocator: gpuChunkedDirect aloca con
    // page_allocator (su gpa interno); liberar con testing.allocator es
    // "Invalid free" del DebugAllocator ⇒ ABRT en los 4 tests. Liberar
    // con el allocator que devuelve el helper (_allocator).
    defer result._allocator.free(result.attn_out);
    defer result._allocator.free(result.state_out);

    const out_c = try checkDiff(result.attn_out, out_ref, 1e-3, 1e-3, "attn_out");
    const st_c = try checkDiff(result.state_out, state_ref, 2e-3, 2e-3, "state");
    try testing.expect(out_c < 1.0);
    try testing.expect(st_c < 1.0);
}

test "prefillChunk K=64 n=30 (n<K, n_tokens=runtime check): paridad vs per-token" {
    cudaz.ensureContext() catch return error.SkipZigTest;
    _ = cudaz.cuDeviceGet(0) catch return error.SkipZigTest;
    const gpa = testing.allocator;

    const stream: cudaz.CUstream = try cudaz.cuStreamCreate(0);
    var lk = try layer_kernels.LayerKernels.init(stream);

    const n: usize = 30;
    const K: usize = 64;

    const conv_out = try gpa.alloc(f32, n * qkv_dim);
    defer gpa.free(conv_out);
    randBuf(0xD17A, conv_out, 0.05);
    const gate = try gpa.alloc(f32, n * dt_rank);
    defer gpa.free(gate);
    {
        var rng = std.Random.Xoshiro256.init(0x6A7E);
        for (gate) |*v| v.* = -rng.random().float(f32) * 2.0;
    }
    const beta = try gpa.alloc(f32, n * dt_rank);
    defer gpa.free(beta);
    {
        var rng = std.Random.Xoshiro256.init(0xB37A);
        for (beta) |*v| v.* = rng.random().float(f32) * 0.5 + 0.5;
    }
    const state_ref = try gpa.alloc(f32, n_v_heads * dim2);
    defer gpa.free(state_ref);
    randBuf(0x57A7, state_ref, 0.001);
    const out_ref = try gpa.alloc(f32, n * d_inner);
    defer gpa.free(out_ref);

    @memset(out_ref, 0);
    refPerToken(conv_out, gate, beta, state_ref, out_ref, n);

    const result = try gpuChunkedDirect(&lk, conv_out, gate, beta, state_ref, n, K);
    // [coordinador] FIX cross-allocator: gpuChunkedDirect aloca con
    // page_allocator (su gpa interno); liberar con testing.allocator es
    // "Invalid free" del DebugAllocator ⇒ ABRT en los 4 tests. Liberar
    // con el allocator que devuelve el helper (_allocator).
    defer result._allocator.free(result.attn_out);
    defer result._allocator.free(result.state_out);

    const out_c = try checkDiff(result.attn_out, out_ref, 1e-3, 1e-3, "attn_out");
    const st_c = try checkDiff(result.state_out, state_ref, 2e-3, 2e-3, "state");
    try testing.expect(out_c < 1.0);
    try testing.expect(st_c < 1.0);
}

test "prefillChunk K=64 n=75 (1 chunk + 11 tail): paridad vs per-token" {
    cudaz.ensureContext() catch return error.SkipZigTest;
    _ = cudaz.cuDeviceGet(0) catch return error.SkipZigTest;
    const gpa = testing.allocator;

    const stream: cudaz.CUstream = try cudaz.cuStreamCreate(0);
    var lk = try layer_kernels.LayerKernels.init(stream);

    const n: usize = 75;
    const K: usize = 64;

    const conv_out = try gpa.alloc(f32, n * qkv_dim);
    defer gpa.free(conv_out);
    randBuf(0xD17A, conv_out, 0.05);
    const gate = try gpa.alloc(f32, n * dt_rank);
    defer gpa.free(gate);
    {
        var rng = std.Random.Xoshiro256.init(0x6A7E);
        for (gate) |*v| v.* = -rng.random().float(f32) * 2.0;
    }
    const beta = try gpa.alloc(f32, n * dt_rank);
    defer gpa.free(beta);
    {
        var rng = std.Random.Xoshiro256.init(0xB37A);
        for (beta) |*v| v.* = rng.random().float(f32) * 0.5 + 0.5;
    }
    const state_ref = try gpa.alloc(f32, n_v_heads * dim2);
    defer gpa.free(state_ref);
    randBuf(0x57A7, state_ref, 0.001);
    const out_ref = try gpa.alloc(f32, n * d_inner);
    defer gpa.free(out_ref);

    @memset(out_ref, 0);
    refPerToken(conv_out, gate, beta, state_ref, out_ref, n);

    // For n=75, split into 1 chunk of 64 + tail of 11.
    // Chunk 0: n_tokens=64, base pointer at conv_out[0..64]
    // Tail: n_tokens=11, base pointer at conv_out[64..75]
    const result_chunk = try gpuChunkedDirect(&lk, conv_out.ptr[0 .. K * qkv_dim], gate.ptr[0 .. K * dt_rank], beta.ptr[0 .. K * dt_rank], state_ref, K, K);
    // [coordinador] FIX cross-allocator (ver n=64 test): page_allocator vs
    // testing.allocator ⇒ "Invalid free" ABRT. Liberar con el _allocator.
    errdefer result_chunk._allocator.free(result_chunk.attn_out);

    // For the tail, use the GPU per-token path (deltaNetWarp)
    // The tail's state_out becomes the final state. Don't free result_chunk.state_out
    // until after the tail completes (it's the tail's input).
    const conv_tail = conv_out.ptr[K * qkv_dim ..][0 .. (n - K) * qkv_dim];
    const gate_tail = gate.ptr[K * dt_rank ..][0 .. (n - K) * dt_rank];
    const beta_tail = beta.ptr[K * dt_rank ..][0 .. (n - K) * dt_rank];
    const result_tail = try gpuPerToken(&lk, conv_tail, gate_tail, beta_tail, result_chunk.state_out, n - K);
    defer result_chunk._allocator.free(result_chunk.state_out);
    defer result_tail._allocator.free(result_tail.attn_out);
    defer result_tail._allocator.free(result_tail.state_out);

    // Combine outputs
    const combined_out = try gpa.alloc(f32, n * d_inner);
    defer gpa.free(combined_out);
    @memcpy(combined_out[0 .. K * d_inner], result_chunk.attn_out);
    @memcpy(combined_out[K * d_inner ..][0 .. (n - K) * d_inner], result_tail.attn_out);

    const out_c = try checkDiff(combined_out, out_ref, 1e-3, 1e-3, "attn_out");
    const st_c = try checkDiff(result_tail.state_out, state_ref, 2e-3, 2e-3, "state");
    try testing.expect(out_c < 1.0);
    try testing.expect(st_c < 1.0);
}

// ─── 1.11 (lane-c): fused CH ubatch-wide K=512 ────────────────────────────────

/// Setup común de buffers aleatorios para los tests K512.
const K512Setup = struct {
    conv_out: []f32,
    gate: []f32,
    beta: []f32,
    state: []f32,
    out_ref: []f32,
    gpa: std.mem.Allocator,

    fn init(gpa: std.mem.Allocator, n: usize) !K512Setup {
        const conv_out = try gpa.alloc(f32, n * qkv_dim);
        randBuf(0xD17A, conv_out, 0.05);
        const gate = try gpa.alloc(f32, n * dt_rank);
        {
            var rng = std.Random.Xoshiro256.init(0x6A7E);
            for (gate) |*v| v.* = -rng.random().float(f32) * 2.0;
        }
        const beta = try gpa.alloc(f32, n * dt_rank);
        {
            var rng = std.Random.Xoshiro256.init(0xB37A);
            for (beta) |*v| v.* = rng.random().float(f32) * 0.5 + 0.5;
        }
        const state = try gpa.alloc(f32, n_v_heads * dim2);
        randBuf(0x57A7, state, 0.001);
        const out_ref = try gpa.alloc(f32, n * d_inner);
        @memset(out_ref, 0);
        return .{ .conv_out = conv_out, .gate = gate, .beta = beta, .state = state, .out_ref = out_ref, .gpa = gpa };
    }

    fn deinit(self: *K512Setup) void {
        self.gpa.free(self.conv_out);
        self.gpa.free(self.gate);
        self.gpa.free(self.beta);
        self.gpa.free(self.state);
        self.gpa.free(self.out_ref);
    }
};

test "prefillChunk K=512 n=512 (ubatch completo, 1 launch): paridad vs per-token" {
    cudaz.ensureContext() catch return error.SkipZigTest;
    _ = cudaz.cuDeviceGet(0) catch return error.SkipZigTest;
    const gpa = testing.allocator;

    const stream: cudaz.CUstream = try cudaz.cuStreamCreate(0);
    var lk = try layer_kernels.LayerKernels.init(stream);

    const n: usize = 512;
    const K: usize = 512;

    var s = try K512Setup.init(gpa, n);
    defer s.deinit();
    refPerToken(s.conv_out, s.gate, s.beta, s.state, s.out_ref, n);

    // Referencia GPU con 8 chunks K=64 (el camino pre-1.11 exacto).
    const st_c64 = try gpa.dupe(f32, s.state);
    defer gpa.free(st_c64);
    const out_c64 = try gpa.alloc(f32, n * d_inner);
    defer gpa.free(out_c64);
    {
        var t: usize = 0;
        while (t < n) : (t += 64) {
            const kk = @min(64, n - t);
            const r = try gpuChunkedDirect(&lk, s.conv_out.ptr[t * qkv_dim ..][0 .. kk * qkv_dim], s.gate.ptr[t * dt_rank ..][0 .. kk * dt_rank], s.beta.ptr[t * dt_rank ..][0 .. kk * dt_rank], st_c64, kk, 64);
            defer r._allocator.free(r.attn_out);
            @memcpy(out_c64[t * d_inner ..][0 .. kk * d_inner], r.attn_out);
            if (t + kk == n) {
                // el último chunk escribe el estado final
                @memcpy(st_c64, r.state_out);
            } else {
                @memcpy(st_c64, r.state_out);
            }
        }
    }

    // Fused K512: MISMO estado inicial (s.state) que la referencia CPU.
    const result = try gpuChunkedDirect(&lk, s.conv_out, s.gate, s.beta, s.state, n, K);
    defer result._allocator.free(result.attn_out);
    defer result._allocator.free(result.state_out);

    // Paridad vs CPU per-token (gate absoluto del kernel).
    const out_c = try checkDiff(result.attn_out, s.out_ref, 1e-3, 1e-3, "k512 vs cpu");
    const st_c = try checkDiff(result.state_out, s.state, 2e-3, 2e-3, "k512 state vs cpu");
    try testing.expect(out_c < 1.0);
    try testing.expect(st_c < 1.0);

    // Paridad K512 vs 8×K64 (los launches del camino clásico).
    const out_x = try checkDiff(result.attn_out, out_c64, 1e-3, 1e-3, "k512 vs 8xk64");
    try testing.expect(out_x < 1.0);
}

test "prefillChunk K=512 n=100 (cola parcial en el MISMO launch): paridad vs per-token" {
    // 1.11: n=100 < K=512 — antes eran 1×K64 + 36 per-token; ahora 1 launch
    // con n_tokens=100 (el kernel v5 soporta cola runtime sin OOB).
    cudaz.ensureContext() catch return error.SkipZigTest;
    _ = cudaz.cuDeviceGet(0) catch return error.SkipZigTest;
    const gpa = testing.allocator;

    const stream: cudaz.CUstream = try cudaz.cuStreamCreate(0);
    var lk = try layer_kernels.LayerKernels.init(stream);

    const n: usize = 100;
    const K: usize = 512;

    var s = try K512Setup.init(gpa, n);
    defer s.deinit();
    refPerToken(s.conv_out, s.gate, s.beta, s.state, s.out_ref, n);

    const result = try gpuChunkedDirect(&lk, s.conv_out, s.gate, s.beta, s.state, n, K);
    defer result._allocator.free(result.attn_out);
    defer result._allocator.free(result.state_out);

    const out_c = try checkDiff(result.attn_out, s.out_ref, 1e-3, 1e-3, "attn_out");
    const st_c = try checkDiff(result.state_out, s.state, 2e-3, 2e-3, "state");
    try testing.expect(out_c < 1.0);
    try testing.expect(st_c < 1.0);
}
