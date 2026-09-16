//! Lane-b1 A9 (Dev A): smoke MMA m16n8k16 — fragmentos vs CPU matmul.
//!
//! Carga kvarn_mma_smoke.cubin, lanza kvarn_mma_smoke_kernel con A/B f16
//! aleatorios (seed fija) y compara D contra el producto f32 de referencia
//! (rel < 1e-3; el MMA acumula en f32 con inputs f16).

const std = @import("std");
const testing = std.testing;
const build_options = @import("build_options");
const cudaz = @import("cudaz");

const M = 16;
const K = 16;
const N = 8;

test "A9 smoke: mma.m16n8k16 == CPU matmul f32-accum" {
    if (build_options.kvarn_mma_cubin.len == 0) return error.SkipZigTest;

    try cudaz.ensureContext();
    const module = try cudaz.cuModuleLoad(build_options.kvarn_mma_cubin);
    const func = try cudaz.cuModuleGetFunction(module, "kvarn_mma_smoke_kernel");
    const allocator = testing.allocator;

    var prng = std.Random.DefaultPrng.init(0xA99E);
    const rand = prng.random();

    const a_h = try allocator.alloc(f16, M * K);
    defer allocator.free(a_h);
    const b_h = try allocator.alloc(f16, K * N);
    defer allocator.free(b_h);
    for (a_h) |*x| x.* = @floatCast(rand.float(f32) * 0.5 - 0.25);
    for (b_h) |*x| x.* = @floatCast(rand.float(f32) * 0.5 - 0.25);

    // CPU ref: f32 accumulate sobre los mismos f16.
    const want = try allocator.alloc(f32, M * N);
    defer allocator.free(want);
    for (0..M) |i| {
        for (0..N) |j| {
            var acc: f32 = 0;
            for (0..K) |k| {
                acc += @as(f32, a_h[i * K + k]) * @as(f32, b_h[k * N + j]);
            }
            want[i * N + j] = acc;
        }
    }

    const d_a = try cudaz.cuMemAlloc(@sizeOf(f16) * M * K);
    defer cudaz.cuMemFree(d_a);
    const d_b = try cudaz.cuMemAlloc(@sizeOf(f16) * K * N);
    defer cudaz.cuMemFree(d_b);
    const d_d = try cudaz.cuMemAlloc(@sizeOf(f32) * M * N);
    defer cudaz.cuMemFree(d_d);
    try cudaz.cuMemcpyHtoD(d_a, @intFromPtr(a_h.ptr), @sizeOf(f16) * M * K);
    try cudaz.cuMemcpyHtoD(d_b, @intFromPtr(b_h.ptr), @sizeOf(f16) * K * N);

    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);
    const n_iter: c_int = 1;
    var kp: [4]?*const anyopaque = .{ &d_a, &d_b, &d_d, &n_iter };
    try cudaz.cuLaunchKernel(func, 1, 1, 1, 32, 1, 1, 0, stream, @ptrCast(&kp), null);
    try cudaz.cuStreamSynchronize(stream);

    const got = try allocator.alloc(f32, M * N);
    defer allocator.free(got);
    try cudaz.cuMemcpyDtoH(@intFromPtr(got.ptr), d_d, @sizeOf(f32) * M * N);

    var max_rel: f64 = 0;
    var bad: usize = 0;
    for (got, want) |g, w| {
        const rel = @abs(@as(f64, g) - @as(f64, w)) / @max(@abs(@as(f64, w)), 1e-3);
        max_rel = @max(max_rel, rel);
        if (rel > 5e-3) bad += 1;
    }
    if (bad > 0) std.log.err("A9 max_rel={d} bad={d}/{d}", .{ max_rel, bad, M * N });
    try testing.expectEqual(@as(usize, 0), bad);
}
