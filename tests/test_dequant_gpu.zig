//! Test Fase 1 — Dequantización GPU (Q4_K, Q6_K, IQ4_XS, IQ3_S) bit-exact
//! vs la referencia CPU (gguf.dequantQ4_K / dequantQ6_K / dequantIq4_xs /
//! dequantIq3_s). Usa tensores sintéticos pequeños (sin necesidad del 9B).
//! Se salta si CUDA no está disponible.
const std = @import("std");
const gguf = @import("gguf");
const gguf_dequant = @import("gguf_dequant");
const cudaz = @import("cudaz");

fn testDtype(
    gpa: std.mem.Allocator,
    engine: *gguf_dequant.GgufDequantEngine,
    dtype: gguf_dequant.GgufDtype,
    cpu_dequant: *const fn (bytes: []const u8, out: []f32) void,
    num_blocks: usize,
    seed: u64,
) !void {
    const bs = dtype.blockBytes();
    const numel = num_blocks * dtype.blockSize();

    const bytes = try gpa.alloc(u8, num_blocks * bs);
    defer gpa.free(bytes);
    var rng = std.Random.Xoshiro256.init(seed);
    rng.random().bytes(bytes);

    const out_cpu = try gpa.alloc(f32, numel);
    defer gpa.free(out_cpu);
    const out_gpu = try gpa.alloc(f32, numel);
    defer gpa.free(out_gpu);

    cpu_dequant(bytes, out_cpu);

    try engine.dequant(dtype, bytes, out_gpu);

    var max_diff: f32 = 0;
    for (out_cpu, out_gpu, 0..) |c, g, i| {
        max_diff = @max(max_diff, @abs(c - g));
        if (c != g) {
            std.debug.print("[{s}] mismatch at {d}: cpu={d} gpu={d}\n", .{ @tagName(dtype), i, c, g });
            return error.DequantMismatch;
        }
    }
    std.debug.print("[{s}] OK: {d} elems, {d} bloques, max_diff={d}\n", .{ @tagName(dtype), numel, num_blocks, max_diff });
}

test "dequant GPU Q4_K/Q6_K/IQ4_XS/IQ3_S bit-exact vs CPU" {
    if (!cudaz.isCudaAvailable()) {
        std.debug.print("SKIP: CUDA no disponible\n", .{});
        return error.SkipZigTest;
    }
    const gpa = std.testing.allocator;
    var engine = try gguf_dequant.GgufDequantEngine.init();
    defer engine.deinit();

    try testDtype(gpa, &engine, .q4_k, gguf.dequantQ4_K, 4, 101);
    try testDtype(gpa, &engine, .q6_k, gguf.dequantQ6_K, 4, 202);
    try testDtype(gpa, &engine, .iq4_xs, gguf.dequantIq4_xs, 4, 303);
    try testDtype(gpa, &engine, .iq3_s, gguf.dequantIq3_s, 4, 404);
}
