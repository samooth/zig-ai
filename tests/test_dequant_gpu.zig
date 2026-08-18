//! Verificación bit-exact (tolerancia 1e-2) de la dequantización GPU vs la
//! referencia CPU en gguf.zig. Cubre todos los dtypes con kernel CUDA.
//! Se salta si CUDA no está disponible (build_options.has_cuda == false).
const std = @import("std");
const gguf = @import("gguf");
const gguf_dequant = @import("gguf_dequant");

// Todos los dtypes con kernel CUDA enlazado (kernels/*.cu).
const dtypes = [_]gguf.GgmlType{
    .q4_k,     .q6_k,   .iq4_xs,  .iq3_s,
    .iq4_nl,   .iq2_xxs, .iq2_xs, .iq3_xxs,
    .iq1_s,    .iq2_s,   .iq1_m,  .tq1_0,
    .tq2_0,    .mxfp4,
};

fn approxEq(a: f32, b: f32) bool {
    if (a == b) return true; // también cubre +Inf/-Inf y valores exactos
    if (a != a and b != b) return true; // ambos NaN
    const diff = @abs(a - b);
    return diff <= 1e-2 + 1e-3 * @abs(a);
}

fn testDtype(gpa: std.mem.Allocator, engine: *const gguf_dequant.GgufDequantEngine, dtype: gguf.GgmlType, num_blocks: usize, seed: u64) !void {
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
    @memset(out_gpu, 0);

    gguf.dequantBlock(dtype, bytes, out_cpu, numel);
    try engine.dequant(dtype, bytes, out_gpu);

    var max_diff: f32 = 0;
    for (out_cpu, out_gpu, 0..) |c, g, i| {
        max_diff = @max(max_diff, @abs(c - g));
        if (!approxEq(c, g)) {
            std.debug.print("[{s}] mismatch at {d}: cpu={d} gpu={d}\n", .{ @tagName(dtype), i, c, g });
            return error.DequantMismatch;
        }
    }
    std.debug.print("[{s}] OK: {d} elems, {d} bloques, max_diff={d}\n", .{ @tagName(dtype), numel, num_blocks, max_diff });
}

test "dequant GPU bit-exact vs CPU (todas las variantes)" {
    const engine = blk: {
        const eng = gguf_dequant.GgufDequantEngine.init() catch |e| {
            if (e == error.CudaUnavailable) {
                std.debug.print("SKIP: CUDA no disponible\n", .{});
                return error.SkipZigTest;
            }
            return e;
        };
        break :blk eng;
    };
    defer engine.deinit();

    const gpa = std.testing.allocator;
    var seed: u64 = 100;
    inline for (dtypes) |dt| {
        try testDtype(gpa, &engine, dt, 4, seed);
        seed += 37;
    }
}
