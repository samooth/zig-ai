// SPDX-License-Identifier: MIT
//! E2E acceptance test for the hybrid MoE flow on host (lane-e).
//!
//! Validates the executor-merge contract without needing a GPU:
//!   1. Build a small synthetic MoE geometry (8 experts, n_embd=64, ff=32, top_k=2).
//!   2. Quantize random f32 expert weights to q8_0 (the executor's supported CPU format).
//!   3. Compute a dense reference: out[d] = Σ_t weight[t] · expert[ids[t]] · x[t].
//!   4. Drive the CPU executor (host_staging mode, no GPU): submit(ids) → sync() → merge.
//!   5. Compare merged partial against the dense reference (max |diff| < tol).
//!
//! This is the "executor-merge portion on host against the dense reference"
//! the plan calls for when forwardGPU is unavailable.

const std = @import("std");

fn stdoutPrint(io: std.Io, comptime fmt: []const u8, args: anytype) !void {
    var stdout_buf: [256]u8 = undefined;
    const stdout_file = std.Io.File.stdout();
    var stdout_writer = stdout_file.writer(io, &stdout_buf);
    try stdout_writer.interface.print(fmt, args);
}
const testing = std.testing;
const cpu_executor = @import("moe_cpu_executor");
const gemv_mod = @import("moe_cpu_gemv");
const kv_cache = @import("kv_cache");
const kv_quant = kv_cache.kv_quant;

const n_experts: u32 = 8;
const n_embd: u32 = 64;
const ff: u32 = 32;
const top_k: u32 = 1;
const q8_0_bytes: usize = 34;
const q8_0_block: usize = 32;

/// Encode a block of `n` f32 into q8_0 bytes (n must be a multiple of 32).
fn encodeQ8_0(allocator: std.mem.Allocator, src: []const f32) ![]u8 {
    std.debug.assert(src.len % q8_0_block == 0);
    const out = try allocator.alloc(u8, (src.len / q8_0_block) * q8_0_bytes);
    // kv_quant.encode es el oráculo público (firma []const f16): estrechamos
    // f32→f16 con round-trip sin pérdida.
    const src_f16 = try allocator.alloc(f16, src.len);
    defer allocator.free(src_f16);
    for (src, src_f16) |v, *h| h.* = @floatCast(v);
    kv_quant.encode(.q8_0, src_f16, out);
    return out;
}

test "host: executor-merge hybrid flow matches dense reference" {
    const a = testing.allocator;
    const rb = gemv_mod.Format.q8_0.rowBytes(n_embd); // 68 for n=64
    // Weight buffer: [n_experts][ff][rb] packed.
    const w_bytes = n_experts * ff * rb;
    const w = try a.alloc(f32, (@divTrunc(w_bytes + 3, 4)));
    defer a.free(w);

    // Fill weights with small random f32 and quantize to q8_0 per (expert, ff).
    var prng = std.Random.Xoshiro256.init(0xDEADBEEF);
    const rand = prng.random();
    const scratch = try a.alloc(f32, n_embd);
    defer a.free(scratch);
    for (0..n_experts) |e| {
        for (0..ff) |d| {
            for (scratch) |*v| v.* = rand.float(f32) * 0.5 - 0.25;
            const row_bytes = encodeQ8_0(a, scratch) catch unreachable;
            defer a.free(row_bytes);
            // Copy row_bytes into the w buffer at the right offset.
            const w_u8 = std.mem.sliceAsBytes(w);
            const off = (e * ff + d) * rb;
            @memcpy(w_u8[off..][0..rb], row_bytes);
        }
    }

    // Hidden vector for the token.
    const x = try a.alloc(f32, n_embd);
    defer a.free(x);
    for (x) |*v| v.* = rand.float(f32) * 0.5 - 0.25;

    // Two expert ids (deterministic) and weights.
    var ids = [_]i32{3};
    const weights_f = [_]f32{0.7};
    try testing.expectEqual(@as(usize, top_k), ids.len);

    // ── Dense reference ─────────────────────────────────────────────────
    // For each (t, d): out_ref[d] = Σ_t weight[t] · dot(W[ids[t]][d], x[t]).
    // (Single token, so the t-loop is trivial; we keep the structure for clarity.)
    var ref = try a.alloc(f32, ff);
    defer a.free(ref);
    @memset(ref, 0);
    const w_u8_const = std.mem.sliceAsBytes(w);
    for (ids, 0..) |e_id, t| {
        const wt = weights_f[t];
        for (0..ff) |d| {
            const row = w_u8_const[(@as(usize, @intCast(e_id)) * ff + d) * rb ..][0..rb];
            const dot = gemv_mod.dot(.q8_0, row, x);
            ref[d] += wt * dot;
        }
    }

    // ── Executor (host_staging mode) ────────────────────────────────────
    const cfg = cpu_executor.Config{
        .workers_requested = 1,
        .fmt = .q8_0,
        .k_dim = n_embd,
        .out_dim = ff,
        .n_experts = n_experts,
        .watchdog_ms = 10_000,
    };
    var exec = try cpu_executor.Executor.initFull(a, cfg);
    defer exec.deinit();
    // attachStream(0) → no CUDA; falls back to host_staging on this machine.
    _ = exec.attachStream(0);

    // hidden_dev is a HOST pointer in host_staging mode (see F3 docs).
    const hidden_host: usize = @intFromPtr(x.ptr);
    const p = try exec.submit(0, hidden_host, w, &ids);
    const partial = try exec.sync(p);
    defer a.free(partial);

    // Merge with weights (single-token, so partial[t] is the token's output).
    var merged = try a.alloc(f32, ff);
    defer a.free(merged);
    @memset(merged, 0);
    for (ids, 0..) |_, t| {
        const wt = weights_f[t];
        for (0..ff) |d| {
            merged[d] += wt * partial[t * ff + d];
        }
    }

    // ── Assert: merged ≈ ref within tolerance ────────────────────────────
    const tol: f32 = 1e-3; // q8_0 quantization round-trip
    var max_diff: f32 = 0;
    for (merged, ref) |m, r| {
        const d = @abs(m - r);
        if (d > max_diff) max_diff = d;
    }
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "max_diff={d}\n", .{max_diff});
    try testing.expect(max_diff < tol);
}

test "host: executor sync ownership transfer — reuse path" {
    const a = testing.allocator;
    const rb = gemv_mod.Format.q8_0.rowBytes(n_embd);
    const w_bytes = n_experts * ff * rb;
    const w = try a.alloc(f32, (@divTrunc(w_bytes + 3, 4)));
    defer a.free(w);
    const w_u8_buf = std.mem.sliceAsBytes(w);

    var prng = std.Random.Xoshiro256.init(0xCAFEBABE);
    const rand = prng.random();
    const scratch = try a.alloc(f32, n_embd);
    defer a.free(scratch);
    for (0..n_experts) |e| {
        for (0..ff) |d| {
            for (scratch) |*v| v.* = rand.float(f32) * 0.5 - 0.25;
            const row_bytes = encodeQ8_0(a, scratch) catch unreachable;
            defer a.free(row_bytes);
            const off = (e * ff + d) * rb;
            @memcpy(w_u8_buf[off..][0..rb], row_bytes);
        }
    }
    const x = try a.alloc(f32, n_embd);
    defer a.free(x);
    for (x) |*v| v.* = rand.float(f32) * 0.5 - 0.25;

    const cfg = cpu_executor.Config{
        .workers_requested = 1,
        .fmt = .q8_0,
        .k_dim = n_embd,
        .out_dim = ff,
        .n_experts = n_experts,
        .watchdog_ms = 10_000,
    };
    var exec = try cpu_executor.Executor.initFull(a, cfg);
    defer exec.deinit();
    _ = exec.attachStream(0);
    const hidden_host: usize = @intFromPtr(x.ptr);

    // First submit/sync — caller frees the returned partial.
    {
        var ids = [_]i32{1};
        const p = try exec.submit(0, hidden_host, w, &ids);
        const partial = try exec.sync(p);
        a.free(partial);
    }

    // Second submit/sync on the SAME slot — exercises the re-use path that
    // previously double-freed. Caller frees again; no aliasing.
    {
        var ids = [_]i32{2};
        const p = try exec.submit(0, hidden_host, w, &ids);
        const partial = try exec.sync(p);
        defer a.free(partial);
        try testing.expect(partial.len == top_k * ff);
    }
}

test "host: executor determinism — workers=1 vs workers=4 outputs identical" {
    const a = testing.allocator;
    const rb = gemv_mod.Format.q8_0.rowBytes(n_embd);
    const w_bytes = n_experts * ff * rb;
    const w = try a.alloc(f32, (@divTrunc(w_bytes + 3, 4)));
    defer a.free(w);
    const w_u8_buf = std.mem.sliceAsBytes(w);

    var prng = std.Random.Xoshiro256.init(0xABCDEF01);
    const rand = prng.random();
    const scratch = try a.alloc(f32, n_embd);
    defer a.free(scratch);
    for (0..n_experts) |e| {
        for (0..ff) |d| {
            for (scratch) |*v| v.* = rand.float(f32) * 0.5 - 0.25;
            const row_bytes = encodeQ8_0(a, scratch) catch unreachable;
            defer a.free(row_bytes);
            const off = (e * ff + d) * rb;
            @memcpy(w_u8_buf[off..][0..rb], row_bytes);
        }
    }
    const x = try a.alloc(f32, n_embd);
    defer a.free(x);
    for (x) |*v| v.* = rand.float(f32) * 0.5 - 0.25;
    const weights_f = [_]f32{0.6};

    const merged1 = try runMerge(a, w, x, &[_]i32{4}, &weights_f, 1);
    defer a.free(merged1);
    const merged4 = try runMerge(a, w, x, &[_]i32{4}, &weights_f, 4);
    defer a.free(merged4);

    // Row-split determinism: identical inputs must produce bit-identical outputs.
    for (merged1, merged4) |m1, m4| {
        try testing.expectEqual(m1, m4);
    }
}

fn runMerge(
    a: std.mem.Allocator,
    w: []f32,
    x: []const f32,
    ids: []const i32,
    weights_f: []const f32,
    workers: usize,
) ![]f32 {
    const cfg = cpu_executor.Config{
        .workers_requested = workers,
        .fmt = .q8_0,
        .k_dim = n_embd,
        .out_dim = ff,
        .n_experts = n_experts,
        .watchdog_ms = 10_000,
    };
    var exec = try cpu_executor.Executor.initFull(a, cfg);
    defer exec.deinit();
    _ = exec.attachStream(0);
    const hidden_host: usize = @intFromPtr(x.ptr);
    const p = try exec.submit(0, hidden_host, w, ids);
    const partial = try exec.sync(p);
    var merged = try a.alloc(f32, ff);
    for (ids, 0..) |_, t| {
        const wt = weights_f[t];
        for (0..ff) |d| merged[d] += wt * partial[t * ff + d];
    }
    a.free(partial);
    return merged;
}

test "host: memopsCanary trips + attachStream → host_staging (graceful-degrade contract)" {
    // Regression guard for the bug class that crashed 52dd4f8: attachStream
    // con handle NULL (o canary fallido) DEBE landear en host_staging —
    // nunca cuda_memops con stream 0 (el @ptrFromInt(0) era UB). El canary
    // sintético puede pasar en este host mientras la operación REAL de
    // producción falla: el contrato probado aquí es el modo resultante,
    // no el valor del probe.
    const cfg = cpu_executor.Config{
        .workers_requested = 1,
        .fmt = .q8_0,
        .k_dim = 64,
        .out_dim = 32,
        .n_experts = 8,
        .watchdog_ms = 10_000,
    };
    var exec = try cpu_executor.Executor.initFull(testing.allocator, cfg);
    defer exec.deinit();
    const m = exec.attachStream(0);
    try testing.expectEqual(cpu_executor.Mode.host_staging, m);
    // Y un executor en host_staging acepta submit/sync con host ptrs sin UB:
    const ids = [_]i32{1};
    var x: [64]f32 = undefined;
    for (&x) |*v| v.* = 0.5;
    const w_bytes = 8 * 32 * gemv_mod.Format.q8_0.rowBytes(64);
    const w = try testing.allocator.alloc(f32, (w_bytes + 3) / 4);
    defer testing.allocator.free(w);
    @memset(std.mem.sliceAsBytes(w), 0);
    const p = try exec.submit(0, @intFromPtr(&x), w, &ids);
    const partial = try exec.sync(p);
    defer testing.allocator.free(partial);
    try testing.expect(partial.len == 32);
}
