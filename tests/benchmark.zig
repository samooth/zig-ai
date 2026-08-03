const std = @import("std");
const Tensor = @import("core").Tensor;
const matmul = @import("matmul");
const fa = @import("fa");

const MatmulEngine = matmul.MatmulEngine;
const Backend = matmul.Backend;
const PrecisionMode = matmul.PrecisionMode;
const FlashAttentionConfig = fa.fa_config.FlashAttentionConfig;
const FlashAttentionCpu = fa.FlashAttentionCpu;

fn benchmarkMatmul(allocator: std.mem.Allocator, backend: Backend, M: usize, N: usize, K: usize, iterations: usize) !f64 {
    var A = try Tensor(f32).alloc(allocator, &[_]usize{ M, K });
    defer A.deinit();
    var B = try Tensor(f32).alloc(allocator, &[_]usize{ K, N });
    defer B.deinit();
    var C = try Tensor(f32).alloc(allocator, &[_]usize{ M, N });
    defer C.deinit();

    var rng = std.Random.Xoshiro256.init(42);
    A.randUniform(&rng, -1.0, 1.0);
    B.randUniform(&rng, -1.0, 1.0);

    var engine = try MatmulEngine.init(allocator, backend, .f32);
    defer engine.deinit();

    // Warmup
    try engine.gemmNoTrans(f32, A, B, &C);

    var timer = try std.time.Timer.start();
    for (0..iterations) |_| {
        try engine.gemmNoTrans(f32, A, B, &C);
    }
    const total_ns = timer.read();
    const avg_ms = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(iterations)) / 1_000_000.0;

    const ops = @as(f64, @floatFromInt(M)) * @as(f64, @floatFromInt(N)) * @as(f64, @floatFromInt(K)) * 2.0;
    const gflops = ops / (avg_ms / 1000.0) / 1_000_000_000.0;

    return gflops;
}

fn benchmarkFlashAttentionCpu(allocator: std.mem.Allocator, N: usize, d: usize, num_heads: usize, iterations: usize) !f64 {
    const config = FlashAttentionConfig{
        .N = N, .d = d, .num_heads = num_heads, .batch_size = 1,
        .dtype = .f16, .causal = true,
    };

    var fa_cpu = FlashAttentionCpu.init(allocator, config);

    var Q = try Tensor(f16).alloc(allocator, &[_]usize{ 1, num_heads, N, d });
    defer Q.deinit();
    var K = try Tensor(f16).alloc(allocator, &[_]usize{ 1, num_heads, N, d });
    defer K.deinit();
    var V = try Tensor(f16).alloc(allocator, &[_]usize{ 1, num_heads, N, d });
    defer V.deinit();
    var O = try Tensor(f16).alloc(allocator, &[_]usize{ 1, num_heads, N, d });
    defer O.deinit();

    fa.fa_utils.initUniform(&Q, -0.1, 0.1, 42);
    fa.fa_utils.initUniform(&K, -0.1, 0.1, 43);
    fa.fa_utils.initUniform(&V, -0.1, 0.1, 44);

    // Warmup
    try fa_cpu.forward(Q, K, V, &O);

    var timer = try std.time.Timer.start();
    for (0..iterations) |_| {
        try fa_cpu.forward(Q, K, V, &O);
    }
    const total_ns = timer.read();
    const avg_ms = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(iterations)) / 1_000_000.0;

    return avg_ms;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("\n", .{});
    try stdout.print("=================================================\n", .{});
    try stdout.print("         Zig AI Engine — Benchmark Suite         \n", .{});
    try stdout.print("=================================================\n", .{});
    try stdout.print("\n", .{});

    // Matmul benchmarks
    const matmul_sizes = &[_]struct { M: usize, N: usize, K: usize, iters: usize }{
        .{ .M = 512, .N = 512, .K = 512, .iters = 100 },
        .{ .M = 1024, .N = 1024, .K = 1024, .iters = 50 },
        .{ .M = 2048, .N = 2048, .K = 2048, .iters = 20 },
    };

    const backends = &[_]Backend{ .naive, .simd, .tiled, .parallel };

    try stdout.print("--- Matmul Benchmarks ---\n", .{});
    for (matmul_sizes) |sz| {
        try stdout.print("\nM={d}, N={d}, K={d}\n", .{ sz.M, sz.N, sz.K });
        for (backends) |be| {
            const gflops = benchmarkMatmul(allocator, be, sz.M, sz.N, sz.K, sz.iters) catch |err| {
                try stdout.print("  {s}: ERROR {s}\n", .{@tagName(be), @errorName(err)});
                continue;
            };
            try stdout.print("  {s:10}: {d:.2} GFLOPS\n", .{@tagName(be), gflops});
        }
    }

    // FlashAttention CPU benchmarks
    const fa_sizes = &[_]struct { N: usize, d: usize, heads: usize, iters: usize }{
        .{ .N = 64, .d = 64, .heads = 4, .iters = 10 },
        .{ .N = 128, .d = 64, .heads = 8, .iters = 5 },
        .{ .N = 256, .d = 64, .heads = 8, .iters = 3 },
    };

    try stdout.print("\n--- FlashAttention CPU Benchmarks ---\n", .{});
    for (fa_sizes) |sz| {
        const avg_ms = benchmarkFlashAttentionCpu(allocator, sz.N, sz.d, sz.heads, sz.iters) catch |err| {
            try stdout.print("N={d}, d={d}, heads={d}: ERROR {s}\n", .{ sz.N, sz.d, sz.heads, @errorName(err) });
            continue;
        };
        try stdout.print("N={d}, d={d}, heads={d}: {d:.2} ms/iter\n", .{ sz.N, sz.d, sz.heads, avg_ms });
    }

    try stdout.print("\n=================================================\n", .{});
    try stdout.print("              Benchmark completado               \n", .{});
    try stdout.print("=================================================\n", .{});
}
