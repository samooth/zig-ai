//! Paridad bit-exact GPU↔CPU de los kernels de cuantización (encode)
//! Phase 3 (`src/matmul/quant/encode_kernels.cu`):
//!   - MXFP4: 17B/bloque32 — espejo de kv_quant.encodeMXFP4.
//!   - Q8_0:  34B/bloque32 — espejo de kv_quant.encodeQ8_0.
//! Se salta si CUDA no está disponible (build_options.has_cuda == false).
const std = @import("std");

fn stdoutPrint(io: std.Io, comptime fmt: []const u8, args: anytype) !void {
    var stdout_buf: [256]u8 = undefined;
    const stdout_file = std.Io.File.stdout();
    var stdout_writer = stdout_file.writer(io, &stdout_buf);
    try stdout_writer.interface.print(fmt, args);
}
const testing = std.testing;
const cudaz = @import("cudaz");
const matmul = @import("matmul");
const kv_cache = @import("kv_cache");
const kv_quant = kv_cache.kv_quant;
const quant_encode = matmul.quant_encode;

// Fallback when build_options is not provided to this test module.
const has_cuda = false;

const num_blocks: usize = 128; // 4096 elems
const elems = num_blocks * 32;

fn randF32(seed: u64, buf: []f32) void {
    var rng = std.Random.Xoshiro256.init(seed);
    // Mezcla de magnitudes para ejercitar escalas (subnormal → clip 448/127).
    for (buf) |*v| {
        v.* = switch (rng.random().intRangeAtMost(u8, 0, 3)) {
            0 => rng.random().float(f32) * 2.0 - 1.0,
            1 => (rng.random().float(f32) * 2.0 - 1.0) * 100.0,
            2 => 0.0,
            else => (rng.random().float(f32) * 2.0 - 1.0) * 0.001,
        };
    }
}

fn roundtrip(comptime fmt: kv_cache.QuantFormat, seed: u64) !void {
    // Sin build_options directo en este módulo: intentamos crear contexto;
    // si el driver no está (sin GPU), el test se salta.
    cudaz.ensureContext() catch return error.SkipZigTest;

    const gpa = testing.allocator;
    const stream: cudaz.CUstream = try cudaz.cuStreamCreate(0);

    // Fuente host + bytes de destino según formato.
    const src = try gpa.alloc(f32, elems);
    defer gpa.free(src);
    randF32(seed, src);

    const block_bytes: usize = switch (fmt) {
        .mxfp4 => 17,
        .q8_0 => 34,
        else => unreachable,
    };
    const dst_bytes = num_blocks * block_bytes;

    const dst_gpu = try gpa.alloc(u8, dst_bytes);
    defer gpa.free(dst_gpu);
    const dst_cpu = try gpa.alloc(u8, dst_bytes);
    defer gpa.free(dst_cpu);
    @memset(dst_gpu, 0xAA);
    @memset(dst_cpu, 0xAA);

    // ── Referencia CPU (oráculo): kv_quant.encode sobre f16 (su firma).
    // Para paridad bit-exact, el input se ajusta primero a la retícula f16
    // (round-trip sin pérdida): la CPU estrecha f32→f16 antes de cuantizar y
    // el kernel GPU lee f32 directo — ambos deben ver los mismos valores.
    for (src) |*v| {
        const h: f16 = @floatCast(v.*);
        v.* = @floatCast(h);
    }
    const src_f16 = try gpa.alloc(f16, elems);
    defer gpa.free(src_f16);
    for (src, src_f16) |v, *h| h.* = @floatCast(v);
    kv_quant.encode(fmt, src_f16, dst_cpu);

    // ── Camino GPU: upload f32 → kernel encode → download.
    const d_src = try cudaz.cuMemAlloc(elems * @sizeOf(f32));
    defer cudaz.cuMemFree(d_src);
    const d_dst = try cudaz.cuMemAlloc(dst_bytes);
    defer cudaz.cuMemFree(d_dst);
    try cudaz.cuMemcpyHtoD(d_src, @intFromPtr(src.ptr), elems * @sizeOf(f32));
    try cudaz.cuMemsetD8(d_dst, 0xAA, dst_bytes);

    switch (fmt) {
        .mxfp4 => try quant_encode.encodeMxfp4(d_src, d_dst, num_blocks, stream),
        .q8_0 => try quant_encode.encodeQ8_0(d_src, d_dst, num_blocks, stream),
        else => unreachable,
    }
    try cudaz.cuStreamSynchronize(stream);
    try cudaz.cuMemcpyDtoH(@intFromPtr(dst_gpu.ptr), d_dst, dst_bytes);

    // ── Paridad bit-exact de los bytes cuantizados.
    var bad: usize = 0;
    for (dst_cpu, dst_gpu, 0..) |c, g, i| {
        if (c != g) {
            if (bad < 8) try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "[{s}] byte {d}: cpu=0x{x:0>2} gpu=0x{x:0>2}\n", .{ @tagName(fmt), i, c, g });
            bad += 1;
        }
    }
    if (bad > 0) {
        try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "[{s}] FALLO {d}/{d} bytes\n", .{ @tagName(fmt), bad, dst_bytes });
        return error.EncodeMismatch;
    }
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "[{s}] OK encode GPU==CPU ({d} bloques, {d} bytes)\n", .{ @tagName(fmt), num_blocks, dst_bytes });
}

test "quant encode MXFP4: paridad bit-exact GPU vs kv_quant.encode" {
    try roundtrip(.mxfp4, 0xC0FFEE01);
}

test "quant encode Q8_0: paridad bit-exact GPU vs kv_quant.encode" {
    try roundtrip(.q8_0, 0xC0FFEE02);
}
