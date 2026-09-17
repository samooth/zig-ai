//! Test quant-resident cache (Lane D): ciclo subir→hit→expulsar→re-subir con
//! contador DMA exacto y roundtrip bit-exacto. Protege la semántica de claves
//! (HOST para expulsar, DEV para lanzar) cuyo fallo silencioso root-causeamos
//! en stream_bench (2026-08-27). Requiere CUDA; sin GPU sale limpio.
const std = @import("std");
const cudaz = @import("cudaz");
const matmul = @import("matmul");

const page = std.heap.page_size_min;

fn makeTensorBytes(gpa: std.mem.Allocator, n: usize, seed: u8) ![]u8 {
    const buf = try gpa.alloc(u8, n);
    for (buf, 0..) |*b, i| b.* = @truncate(i *% 2654435761 +% seed);
    return buf;
}

test "quant-resident: upload→H2D bit-exacto→hit sin DMA→evict→re-upload" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    cudaz.ensureContext() catch return error.SkipZigTest;

    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const gpa = arena_state.allocator();

    // Estado inicial conocido: vaciar el cache compartido del proceso.
    matmul.MatmulEngine.evictQuantCacheAll();
    matmul.MatmulEngine.quantDmaBytesReset();
    try std.testing.expectEqual(@as(usize, 0), matmul.MatmulEngine.quantCacheBytes());

    const a = try makeTensorBytes(gpa, 3 * page, 1);
    const b = try makeTensorBytes(gpa, 3 * page, 2);

    // 1) Primera subida: miss ⇒ DMA + dev ptr válido.
    const dev_a1 = try matmul.MatmulEngine.quantResidentPtr(gpa, a);
    try std.testing.expect(dev_a1 != 0);
    try std.testing.expectEqual(a.len, matmul.MatmulEngine.quantCacheBytes());
    const dma_after_first = matmul.MatmulEngine.quantDmaBytes();
    try std.testing.expectEqual(a.len, dma_after_first);
    // Roundtrip bit-exacto vía D2H.
    const back = try gpa.alloc(u8, a.len);
    try cudaz.cuMemcpyDtoH(@intFromPtr(back.ptr), dev_a1, a.len);
    try std.testing.expectEqualSlices(u8, a, back);

    // 2) Misma clave HOST ⇒ hit: MISMO dev ptr, cero DMA nuevo, bytes estables.
    const dev_a2 = try matmul.MatmulEngine.quantResidentPtr(gpa, a);
    try std.testing.expectEqual(dev_a1, dev_a2);
    try std.testing.expectEqual(dma_after_first, matmul.MatmulEngine.quantDmaBytes());
    try std.testing.expectEqual(a.len, matmul.MatmulEngine.quantCacheBytes());
    // 3) Otro tensor: miss independiente.
    _ = try matmul.MatmulEngine.quantResidentPtr(gpa, b);
    try std.testing.expectEqual(a.len + b.len, matmul.MatmulEngine.quantCacheBytes());
    try std.testing.expectEqual(dma_after_first + b.len, matmul.MatmulEngine.quantDmaBytes());

    // 4) Expulsión por CLAVE HOST (la trampa: nunca pasar el dev ptr aquí).
    const freed = matmul.MatmulEngine.evictQuantCachePtr(@intFromPtr(a.ptr));
    try std.testing.expectEqual(a.len, freed);
    try std.testing.expectEqual(b.len, matmul.MatmulEngine.quantCacheBytes());
    // El DMA acumulado NO baja al expulsar (mide tráfico histórico).
    try std.testing.expectEqual(dma_after_first + b.len, matmul.MatmulEngine.quantDmaBytes());

    // 5) Re-subida tras expulsar: re-alloc + re-DMA (el ciclo del streamer).
    matmul.MatmulEngine.quantDmaBytesReset();
    const dev_a3 = try matmul.MatmulEngine.quantResidentPtr(gpa, a);
    // El driver puede REUTILIZAR la VA recién liberada (dev_a3 == dev_a1 es
    // legítimo y de hecho confirma que el free ocurrió): lo que valida el
    // re-upload es el contador DMA fresco + el roundtrip de abajo.
    try std.testing.expectEqual(a.len, matmul.MatmulEngine.quantDmaBytes());
    try cudaz.cuMemcpyDtoH(@intFromPtr(back.ptr), dev_a3, a.len);
    try std.testing.expectEqualSlices(u8, a, back);

    // 6) Expulsar b y luego evictAll deja el cache a cero bytes.
    const freed_b = matmul.MatmulEngine.evictQuantCachePtr(@intFromPtr(b.ptr));
    try std.testing.expectEqual(b.len, freed_b);
    _ = matmul.MatmulEngine.quantResidentPtr(gpa, a) catch {};
    matmul.MatmulEngine.evictQuantCacheAll();
    try std.testing.expectEqual(@as(usize, 0), matmul.MatmulEngine.quantCacheBytes());
}
