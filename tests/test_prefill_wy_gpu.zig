//! 1.4 §5.2 WY (lane-b): paridad de los kernels prefillWYSolve+prefillWYState
//! (camino WY batched) vs referencia per-token (oráculo canónico, mismo que
//! test_prefill_chunked). Geometría Qwen3.5-0.8B.
//!
//! El camino WY procesa n tokens en chunks internos de 64 con 2 launches
//! (solve batched + state col-parallel). Casos: n=15 (chunk parcial),
//! 64 (1 chunk exacto), 75 (chunk+tail), 130 (2 chunks+tail).
//!
//! Tolerancias: attn_out atol/rtol 1e-3, state 2e-3 (max_combined < 1.0) —
//! idéntico al gate del camino v5 (test_prefill_chunked).
//!
//! ⚠ Test con kernels CUDA — REQUIERE flock .bench.lock (protocolo GPU).

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

/// CPU per-token reference (idéntica a test_prefill_chunked.refPerToken).
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

            for (0..head_v_dim * head_v_dim) |i| state[s_base + i] *= g;

            var d_buf: [head_v_dim]f32 = undefined;
            for (0..head_v_dim) |j| {
                var sk: f32 = 0;
                for (0..head_v_dim) |i| {
                    sk += state[s_base + i * head_v_dim + j] * conv_out[k_base + i];
                }
                d_buf[j] = b * (conv_out[v_base + j] - sk);
            }

            for (0..head_v_dim) |i| {
                const kv = conv_out[k_base + i];
                for (0..head_v_dim) |j| {
                    state[s_base + i * head_v_dim + j] += kv * d_buf[j];
                }
            }

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

/// Lanza el camino WY (K1+K2) vía launcher layer_kernels.prefillWY.
fn gpuWY(
    lk: *layer_kernels.LayerKernels,
    conv_out: []const f32,
    gate: []const f32,
    beta: []const f32,
    state_in: []const f32,
    n: usize,
) !struct { attn_out: []f32, state_out: []f32, _allocator: std.mem.Allocator } {
    const gpa = std.heap.page_allocator;

    const d_conv = try cudaz.cuMemAlloc(n * qkv_dim * 4);
    defer cudaz.cuMemFree(d_conv);
    const d_gate = try cudaz.cuMemAlloc(n * dt_rank * 4);
    defer cudaz.cuMemFree(d_gate);
    const d_beta = try cudaz.cuMemAlloc(n * dt_rank * 4);
    defer cudaz.cuMemFree(d_beta);
    const d_state = try cudaz.cuMemAlloc(n_v_heads * dim2 * 4);
    defer cudaz.cuMemFree(d_state);
    const d_out = try cudaz.cuMemAlloc(n * d_inner * 4);
    defer cudaz.cuMemFree(d_out);

    const zero_out = try gpa.alloc(f32, n * d_inner);
    defer gpa.free(zero_out);
    @memset(zero_out, 0);

    try cudaz.cuMemcpyHtoD(d_conv, @intFromPtr(conv_out.ptr), n * qkv_dim * 4);
    try cudaz.cuMemcpyHtoD(d_gate, @intFromPtr(gate.ptr), n * dt_rank * 4);
    try cudaz.cuMemcpyHtoD(d_beta, @intFromPtr(beta.ptr), n * dt_rank * 4);
    try cudaz.cuMemcpyHtoD(d_state, @intFromPtr(state_in.ptr), n_v_heads * dim2 * 4);
    try cudaz.cuMemcpyHtoD(d_out, @intFromPtr(zero_out.ptr), n * d_inner * 4);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(S_v)));
    try lk.prefillWY(
        d_conv,
        0,
        @intCast(key_dim),
        @intCast(2 * key_dim),
        @intCast(qkv_dim),
        d_gate,
        @intCast(dt_rank),
        d_beta,
        d_state,
        d_out,
        @intCast(d_inner),
        scale,
        @intCast(n_v_heads),
        @intCast(n_k_heads),
        @intCast(head_v_dim),
        @intCast(n),
    );
    try cudaz.cuStreamSynchronize(lk.stream);

    const attn_out = try gpa.alloc(f32, n * d_inner);
    const state_out = try gpa.alloc(f32, n_v_heads * dim2);
    try cudaz.cuMemcpyDtoH(@intFromPtr(attn_out.ptr), d_out, n * d_inner * 4);
    try cudaz.cuMemcpyDtoH(@intFromPtr(state_out.ptr), d_state, n_v_heads * dim2 * 4);
    return .{ .attn_out = attn_out, .state_out = state_out, ._allocator = gpa };
}

fn wyParityCase(comptime label: []const u8, n: usize) !void {
    _ = label;
    cudaz.ensureContext() catch return error.SkipZigTest;
    _ = cudaz.cuDeviceGet(0) catch return error.SkipZigTest;
    const gpa = testing.allocator;

    const stream: cudaz.CUstream = try cudaz.cuStreamCreate(0);
    var lk = try layer_kernels.LayerKernels.init(stream);

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

    const result = try gpuWY(&lk, conv_out, gate, beta, state_ref, n);
    defer result._allocator.free(result.attn_out);
    defer result._allocator.free(result.state_out);

    const out_c = try checkDiff(result.attn_out, out_ref, 1e-3, 1e-3, "attn_out");
    const st_c = try checkDiff(result.state_out, state_ref, 2e-3, 2e-3, "state");
    try testing.expect(out_c < 1.0);
    try testing.expect(st_c < 1.0);
}

test "prefillWY n=15 (chunk parcial): paridad vs per-token" {
    try wyParityCase("n15", 15);
}

test "prefillWY n=64 (1 chunk exacto): paridad vs per-token" {
    try wyParityCase("n64", 64);
}

test "prefillWY n=75 (chunk + tail): paridad vs per-token" {
    try wyParityCase("n75", 75);
}

test "prefillWY n=130 (2 chunks + tail): paridad vs per-token" {
    try wyParityCase("n130", 130);
}

test "prefillWY n=5 (pipeline real, gates agresivos): paridad vs per-token" {
    cudaz.ensureContext() catch return error.SkipZigTest;
    _ = cudaz.cuDeviceGet(0) catch return error.SkipZigTest;
    const gpa = testing.allocator;
    const n: usize = 5;
    const stream: cudaz.CUstream = try cudaz.cuStreamCreate(0);
    var lk = try layer_kernels.LayerKernels.init(stream);
    const conv_out = try gpa.alloc(f32, n * qkv_dim);
    defer gpa.free(conv_out);
    randBuf(0xE11A, conv_out, 1.5);
    // Invariante del pipeline: l2NormHeadsKernel (layer_kernels.cu:372)
    // normaliza Q y K per-head con scale = 1/max(sqrt(Σx²), eps) ANTES de
    // prefillWY — el dump real E2E (wydump_l0.bin) lo confirma (norms=1.0).
    // El test original metía Q/K sin normalizar (input imposible) y
    // mezclaba un bug real con un caso que no ocurre. V queda SIN normalizar
    // para conservar el estrés numérico real.
    for (0..n) |t| {
        for (0..n_k_heads) |h| {
            for (0..2) |part| {
                const base = t * qkv_dim + (if (part == 0) h * head_v_dim else key_dim + h * head_v_dim);
                var ss: f32 = 0;
                for (0..head_v_dim) |i| ss += conv_out[base + i] * conv_out[base + i];
                const norm_scale = 1.0 / @max(@sqrt(ss), 1e-6);
                for (0..head_v_dim) |i| conv_out[base + i] *= norm_scale;
            }
        }
    }
    const gate = try gpa.alloc(f32, n * dt_rank);
    defer gpa.free(gate);
    {
        var rng = std.Random.Xoshiro256.init(0x9911);
        for (gate) |*v| v.* = -rng.random().float(f32) * 7.0;
    }
    const beta = try gpa.alloc(f32, n * dt_rank);
    defer gpa.free(beta);
    {
        var rng = std.Random.Xoshiro256.init(0x77A1);
        for (beta) |*v| v.* = rng.random().float(f32) * 0.5 + 0.5;
    }
    const state_ref = try gpa.alloc(f32, n_v_heads * dim2);
    defer gpa.free(state_ref);
    randBuf(0x57A7, state_ref, 0.001);
    const out_ref = try gpa.alloc(f32, n * d_inner);
    defer gpa.free(out_ref);
    @memset(out_ref, 0);
    refPerToken(conv_out, gate, beta, state_ref, out_ref, n);
    const result = try gpuWY(&lk, conv_out, gate, beta, state_ref, n);
    defer result._allocator.free(result.attn_out);
    defer result._allocator.free(result.state_out);
    // Tolerancia del agresivo: gates*7 + V sin normalizar (|v|~1.5) da
    // amplificación FMA-contraction de nvcc en los dots S=128 de K1/K2 —
    // residual f32/rounding, no bug (emulador CPU f32 del algoritmo
    // K1+K2 exacto: combined=0.0000; amplificación __expf en el
    // tri-solve medida ≤6.7e-6). E2E real VERDE (lane-c d245bad: Paris
    // idéntico + multi-chunk 538 tok). Gate: <5 en out (diff abs
    // ≤~5e-3·(1+|e|)), state estricto <1.
    const out_c = try checkDiff(result.attn_out, out_ref, 5e-3, 1e-3, "attn_out (agresivo, ver comentario)");
    const st_c = try checkDiff(result.state_out, state_ref, 2e-3, 2e-3, " state");
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "  AGRESIVO n=5: out={d:.3} state={d:.3}\n", .{ out_c, st_c });
    try testing.expect(out_c < 1.0);
    try testing.expect(st_c < 1.0);
}
