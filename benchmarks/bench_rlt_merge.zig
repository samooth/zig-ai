// Micro-benchmark: RLT merge feedback overhead
// Measures ns/op for mergeFeedback (CPU) across common d_model sizes.
// No GGUF model needed — pure computation benchmark.
const builtin = @import("builtin");
const std = @import("std");

extern "kernel32" fn QueryPerformanceCounter(lpPerformanceCount: *i64) i32;
extern "kernel32" fn QueryPerformanceFrequency(lpFrequency: *i64) i32;

fn nowNs() u64 {
    if (builtin.target.os.tag == .windows) {
        var counter: i64 = undefined;
        _ = QueryPerformanceCounter(&counter);
        var freq: i64 = undefined;
        _ = QueryPerformanceFrequency(&freq);
        return @intCast(@divTrunc(counter * 1_000_000_000, freq));
    }
    var ts: std.posix.timespec = undefined;
    _ = std.posix.system.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// Reference merge: u = e + α * sigmoid(W_g [e; RMSNorm(s)]) ⊙ (W_s RMSNorm(s))
fn mergeRef(
    e: []const f32,
    s: []const f32,
    w_gate: []const f32,
    w_gate_bias: []const f32,
    w_state: []const f32,
    alpha: f32,
    out: []f32,
    r_buf: []f32,
    gate_buf: []f32,
    state_buf: []f32,
) void {
    const d = e.len;
    // RMSNorm
    var ss: f32 = 0;
    for (s) |v| ss += v * v;
    const inv_rms = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(d)) + 1e-6);
    for (s, 0..) |v, i| r_buf[i] = v * inv_rms;
    // Gate
    for (0..d) |i| {
        var acc = w_gate_bias[i];
        for (0..d) |j| {
            acc += w_gate[i * (2 * d) + j] * e[j];
            acc += w_gate[i * (2 * d) + d + j] * r_buf[j];
        }
        gate_buf[i] = 1.0 / (1.0 + @exp(-acc));
    }
    // State
    for (0..d) |i| {
        var acc: f32 = 0;
        for (0..d) |j| acc += w_state[i * d + j] * r_buf[j];
        state_buf[i] = acc;
    }
    // Merge
    for (out, e, gate_buf, state_buf) |*o, ev, g, st| {
        o.* = ev + alpha * g * st;
    }
}

/// BitNet merge: u = e + α * sigmoid(W @ s) ⊙ (W @ s)
fn mergeBitNet(
    e: []const f32,
    s: []const f32,
    w_proj: []const f32,
    alpha: f32,
    out: []f32,
) void {
    const d = e.len;
    for (0..d) |i| {
        var acc: f32 = 0;
        for (0..d) |j| acc += w_proj[i * d + j] * s[j];
        const gate = 1.0 / (1.0 + @exp(-acc));
        out[i] = e[i] + alpha * gate * acc;
    }
}

const dims = [_]usize{ 64, 128, 256, 512, 1024, 2048, 4096 };
const iterations = 1000;

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    const alloc = arena.allocator();

    std.debug.print("RLT merge micro-benchmark ({d} iters)\n", .{iterations});
    std.debug.print("{s:>8} {s:>12} {s:>12} {s:>12}\n", .{ "d_model", "rlt_merge", "bitnet_merge", "ratio" });
    std.debug.print("{s:>8} {s:>12} {s:>12} {s:>12}\n", .{ "---", "--------", "------------", "------" });

    for (dims) |d| {
        const e = try alloc.alloc(f32, d);
        defer alloc.free(e);
        const s = try alloc.alloc(f32, d);
        defer alloc.free(s);
        const w_gate = try alloc.alloc(f32, d * 2 * d);
        defer alloc.free(w_gate);
        const w_gate_bias = try alloc.alloc(f32, d);
        defer alloc.free(w_gate_bias);
        const w_state = try alloc.alloc(f32, d * d);
        defer alloc.free(w_state);
        const w_proj = try alloc.alloc(f32, d * d);
        defer alloc.free(w_proj);
        const result = try alloc.alloc(f32, d);
        defer alloc.free(result);
        const r_buf = try alloc.alloc(f32, d);
        defer alloc.free(r_buf);
        const gate_buf = try alloc.alloc(f32, d);
        defer alloc.free(gate_buf);
        const state_buf = try alloc.alloc(f32, d);
        defer alloc.free(state_buf);

        var rng = std.Random.Xoshiro256.init(42);
        for (e) |*v| v.* = rng.random().float(f32) * 2 - 1;
        for (s) |*v| v.* = rng.random().float(f32) * 2 - 1;
        for (w_gate) |*v| v.* = rng.random().float(f32) * 0.1 - 0.05;
        for (w_gate_bias) |*v| v.* = rng.random().float(f32) * 0.02 - 0.01;
        for (w_state) |*v| v.* = rng.random().float(f32) * 0.1 - 0.05;
        for (w_proj) |*v| v.* = rng.random().float(f32) * 0.1 - 0.05;

        // Benchmark RLT merge
        const t0 = nowNs();
        for (0..iterations) |_| {
            mergeRef(e, s, w_gate, w_gate_bias, w_state, 0.15, result, r_buf, gate_buf, state_buf);
        }
        const rlt_ns = nowNs() - t0;

        // Benchmark BitNet merge
        const t1 = nowNs();
        for (0..iterations) |_| {
            mergeBitNet(e, s, w_proj, 0.1, result);
        }
        const bitnet_ns = nowNs() - t1;

        const rlt_per = rlt_ns / iterations;
        const bitnet_per = bitnet_ns / iterations;
        const ratio = @as(f64, @floatFromInt(bitnet_per)) / @as(f64, @floatFromInt(rlt_per));

        std.debug.print("{d:>8} {d:>10}ns {d:>10}ns {d:>11.2}x\n", .{ d, rlt_per, bitnet_per, ratio });
    }
}
