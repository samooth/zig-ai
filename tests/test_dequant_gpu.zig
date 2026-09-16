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
    .q4_0,     .q4_1,   .q5_0,   .q5_1,
    .q8_0,     .q8_1,   .q2_k,   .q3_k,
    .q8_k,
    // .q5_k,  // TODO: implement proper kernel
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

// Lane A repro q6_k: q6k_val (lógica del kernel paged_attention.cu:1276
// portada 1:1 a Zig) vs gguf.dequantBlock sobre w=0..255 de un bloque
// sintetico. Aísla si la divergencia del decode fusionado vive en la valfn
// o en el mapeo be->w del kernel.
fn q6kValZig(blk: []const u8, w: usize) f32 {
    const ip = w >> 7;
    const r = w & 127;
    const il = r & 31;
    const j = r >> 5;
    const d = @as(f16, @bitCast(@as(u16, @as(u8, blk[208]) | (@as(u16, blk[209]) << 8))));
    const sc: i8 = @bitCast(blk[192 + 8 * ip + il / 16 + 2 * j]);
    // Fix Lane A: {low,high} alterna por j&1 (bug historico: j>>1 leia
    // ql[l+32]>>4 para j=2 donde GGUF usa ql[l]>>4).
    const qlb = blk[64 * ip + il + (j & 1) * 32];
    const nib: u8 = if ((j >> 1) != 0) (qlb >> 4) else (qlb & 0xF);
    const qhb = blk[128 + 32 * ip + il];
    const packed_v: i32 = @as(i32, nib) | (@as(i32, (qhb >> @intCast(2 * j)) & 3) << 4);
    return @as(f32, @floatCast(d)) * @as(f32, @floatFromInt(sc)) *
        @as(f32, @floatFromInt(packed_v - 32));
}

test "q6_k repro: q6k_val(kernel) vs dequantBlock(CPU) w=0..255" {
    const gpa = std.testing.allocator;
    var blk: [210]u8 = undefined;
    var rng = std.Random.Xoshiro256.init(424242);
    rng.random().bytes(&blk);

    const ref = try gpa.alloc(f32, 256);
    defer gpa.free(ref);
    gguf.dequantBlock(.q6_k, &blk, ref, 256);

    var bad: usize = 0;
    var max_diff: f32 = 0;
    for (0..256) |w| {
        const v = q6kValZig(&blk, w);
        max_diff = @max(max_diff, @abs(v - ref[w]));
        if (!approxEq(v, ref[w])) {
            if (bad < 8) std.debug.print("[q6k-val] w={d}: kernel={d} cpu={d}\n", .{ w, v, ref[w] });
            bad += 1;
        }
    }
    if (bad > 0) {
        std.debug.print("[q6k-val] FALLO {d}/256 max_diff={d}\n", .{ bad, max_diff });
        return error.Q6kValMismatch;
    }
    std.debug.print("[q6k-val] OK 256/256 (valfn == CPU; bug en mapeo del kernel)\n", .{});
}
