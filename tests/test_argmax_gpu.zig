//! G2 (TODO 1.7, lane-a): paridad del argmax device (`argmaxF32Kernel`)
//! contra el argmax host (primera aparición del máximo — el criterio de
//! `Sampler.greedyArgmax` en pipeline.zig).
//!
//! Cubre el caso que REALMENTE importa para no romper el decode: el
//! **empate exacto**. El kernel barre con stride (el hilo t procesa
//! t, t+B, t+2B…), así que el orden de descubrimiento NO es monotónico en
//! el índice; sin desempate por índice el kernel podría elegir una
//! posición posterior con el mismo valor y el token greedy divergiría del
//! camino CPU. `argmaxWins()` resuelve a índice mínimo = primera
//! aparición, igual que el host.
//! Requiere CUDA; sin GPU sale limpio (SkipZigTest).
const std = @import("std");
const cudaz = @import("cudaz");
const matmul = @import("matmul");
const layer_kernels = @import("layer_kernels");

/// Referencia host: primera aparición del máximo (índice mínimo entre los
/// que empatan). Es la semántica exacta de Sampler.greedyArgmax.
fn hostArgmax(row: []const f32) i32 {
    var maxval: f32 = -std.math.inf(f32);
    var argmax: i32 = -1;
    for (row, 0..) |v, i| {
        if (v > maxval) {
            maxval = v;
            argmax = @intCast(i);
        }
    }
    return argmax;
}

fn sharedStream() !cudaz.CUstream {
    return @ptrCast((try matmul.MatmulEngine.sharedCudaStream()).raw);
}

/// Sube `rows`×`cols` y corre el kernel; devuelve el argmax por fila.
fn runArgmax(lk: *layer_kernels.LayerKernels, host: []const f32, rows: usize, cols: usize, out: []i32) !void {
    const d_x = try cudaz.cuMemAlloc(host.len * @sizeOf(f32));
    defer cudaz.cuMemFree(d_x);
    try cudaz.cuMemcpyHtoD(d_x, @intFromPtr(host.ptr), host.len * @sizeOf(f32));
    const d_out = try lk.argmaxOut(rows);
    try lk.argmaxF32(d_x, d_out, rows, cols);
    try cudaz.cuStreamSynchronize(lk.stream);
    try cudaz.cuMemcpyDtoH(@intFromPtr(out.ptr), d_out, rows * @sizeOf(i32));
}

test "argmax gpu: paridad vs host en varias geometrías" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    var lk = try layer_kernels.LayerKernels.init(try sharedStream());
    defer lk.deinit();

    // Geometrías deliberadas: cols < 32 (un solo warp), cols ≈ 1024 (multi
    // warp), cols no múltiplo de 32 ni de 1024 (cola + stride desalineado).
    const shapes = [_][2]usize{
        .{ 1, 1 }, .{ 1, 31 }, .{ 1, 32 }, .{ 1, 33 },
        .{ 1, 1023 }, .{ 1, 1024 }, .{ 1, 1025 },
        .{ 1, 4096 }, .{ 2, 2048 }, .{ 3, 777 },
    };
    var rng = std.Random.Xoshiro256.init(12345);
    for (shapes) |sh| {
        const rows = sh[0];
        const cols = sh[1];
        const host = try std.testing.allocator.alloc(f32, rows * cols);
        defer std.testing.allocator.free(host);
        for (host) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
        // Garantiza un máximo ÚNICO y bien separado (sin empates) para
        // aislar la paridad geométrica del desempate.
        for (0..rows) |r| {
            var mi: usize = 0;
            var mv: f32 = -std.math.inf(f32);
            for (0..cols) |c| {
                if (host[r * cols + c] > mv) {
                    mv = host[r * cols + c];
                    mi = c;
                }
            }
            host[r * cols + mi] = 50.0;
        }
        const out = try std.testing.allocator.alloc(i32, rows);
        defer std.testing.allocator.free(out);
        try runArgmax(&lk, host, rows, cols, out);
        for (0..rows) |r| {
            const want = hostArgmax(host[r * cols ..][0..cols]);
            try std.testing.expectEqual(want, out[r]);
        }
    }
}

test "argmax gpu: empate exacto → primer índice (bit-exact vs host)" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    var lk = try layer_kernels.LayerKernels.init(try sharedStream());
    defer lk.deinit();

    // El máximo (9.0) se repite en índices repartidos por TODO el vector,
    // incluyendo posiciones que caen en el mismo hilo con stride y en
    // hilos/warps distintos — el caso que el desempate por índice fija.
    const cols: usize = 5000;
    const ties = [_]usize{ 0, 1, 1023, 1024, 1025, 2047, 2048, 3001, 4999 };
    const host = try std.testing.allocator.alloc(f32, cols);
    defer std.testing.allocator.free(host);
    @memset(host, 1.0);
    for (ties) |t| host[t] = 9.0;
    // Y un valor negativo grande para descartar que gane -FLT_MAX.
    host[7] = -1.0e30;

    const out = try std.testing.allocator.alloc(i32, 1);
    defer std.testing.allocator.free(out);
    try runArgmax(&lk, host, 1, cols, out);
    // Primera aparición del máximo = índice 0.
    try std.testing.expectEqual(hostArgmax(host), out[0]);
    try std.testing.expectEqual(@as(i32, 0), out[0]);
}

test "argmax gpu: geometría de vocabulario real (128256, 1024 hilos)" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    var lk = try layer_kernels.LayerKernels.init(try sharedStream());
    defer lk.deinit();

    // La geometría del decode: vocab de Llama-3.2 (128256) con block=1024
    // ⇒ 126 pasadas de stride por hilo. Ejercita el camino multi-warp con
    // reducción por shared memory.
    const cols: usize = 128256;
    const host = try std.testing.allocator.alloc(f32, cols);
    defer std.testing.allocator.free(host);
    var rng = std.Random.Xoshiro256.init(999);
    for (host) |*v| v.* = rng.random().float(f32);
    host[65432] = 1234.5; // máximo único en una posición "del medio"
    const out = try std.testing.allocator.alloc(i32, 1);
    defer std.testing.allocator.free(out);
    try runArgmax(&lk, host, 1, cols, out);
    try std.testing.expectEqual(hostArgmax(host), out[0]);
    try std.testing.expectEqual(@as(i32, 65432), out[0]);
}
