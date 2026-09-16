//! Lane-b1 A1 (Dev A): gate F0 — WHT-128 device bit-exacto vs CPU B2.
//!
//! Carga el cubin `kvarn_kernels.cubin`, lanza `kvarn_wht_128_rows_kernel`
//! sobre n filas de 128 floats y compara bit a bit con
//! `kv_cache.kvarn.hadamard128InPlace`. Requiere GPU (has_cuda).

const std = @import("std");
const testing = std.testing;
const build_options = @import("build_options");
const cudaz = @import("cudaz");
const kvarn = @import("kv_cache").kvarn;

const N_ROWS = 64;

fn whtRowsHost(rows: []f32) void {
    var r: usize = 0;
    while (r < rows.len / 128) : (r += 1) {
        kvarn.hadamard128InPlace(rows[r * 128 ..][0..128]);
    }
}

test "F0 smoke: WHT-128 device == hadamard128InPlace (bit-exacto)" {
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;

    try cudaz.ensureContext();
    const module = try cudaz.cuModuleLoad(build_options.kvarn_cubin);
    const func = try cudaz.cuModuleGetFunction(module, "kvarn_wht_128_rows_kernel");

    const allocator = testing.allocator;

    // Semilla fija: PRNG determinista.
    var prng = std.Random.DefaultPrng.init(0xF00D);
    const rand = prng.random();

    const rows_host = try allocator.alloc(f32, N_ROWS * 128);
    defer allocator.free(rows_host);
    for (rows_host) |*v| v.* = rand.float(f32) * 2.0 - 1.0;

    // Referencia CPU (B2).
    const ref = try allocator.dupe(f32, rows_host);
    defer allocator.free(ref);
    whtRowsHost(ref);

    // Upload, launch, download.
    const d_rows = try cudaz.cuMemAlloc(@sizeOf(f32) * rows_host.len);
    defer cudaz.cuMemFree(d_rows);
    try cudaz.cuMemcpyHtoD(d_rows, @intFromPtr(rows_host.ptr), @sizeOf(f32) * rows_host.len);

    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);

    const n_rows_c: c_int = @intCast(N_ROWS);
    var kp: [2]?*const anyopaque = .{ &d_rows, &n_rows_c };
    try cudaz.cuLaunchKernel(
        func,
        @intCast(N_ROWS), // grid x: un bloque por fila
        1,
        1,
        128, // block: 128 threads (uno por dim)
        1,
        1,
        512, // dyn shared: 128 floats
        stream,
        @ptrCast(&kp),
        null,
    );
    try cudaz.cuStreamSynchronize(stream);

    const out_host = try allocator.alloc(f32, rows_host.len);
    defer allocator.free(out_host);
    try cudaz.cuMemcpyDtoH(@intFromPtr(out_host.ptr), d_rows, @sizeOf(f32) * rows_host.len);
    try cudaz.cuStreamSynchronize(stream);

    // Bit-exacto: la CPU usa f32 y el device f32; butterfly y 1/sqrt(128)
    // idénticos => bits idénticos esperados. Si difiere, transcribir
    // operación a operación (ver PLAN_B1 D1).
    var bad: usize = 0;
    for (out_host, ref, 0..) |got, want, i| {
        if (std.math.isNan(got) and std.math.isNan(want)) continue;
        if (got != want) {
            if (bad < 5) std.log.err("F0 mismatch @{d}: got {d} want {d}", .{ i, got, want });
            bad += 1;
        }
    }
    try testing.expectEqual(@as(usize, 0), bad);
}
