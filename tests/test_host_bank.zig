//! Test HostBank (Lane D, D2): roundtrip H2D async desde región registrada
//! pin-after-fill + error de doble registro + unreg idempotente.
//! Requiere CUDA; sin GPU sale limpio (skip).
const std = @import("std");
const cudaz = @import("cudaz");
const ext_mem = @import("cudaz_ext_mem");
const host_bank = @import("host_bank");

const testing = std.testing;
const page = std.heap.page_size_min;

test "HostBank: register→H2D async→D2H roundtrip bit-exacto + unreg" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    try cudaz.ensureContext();

    const pages: usize = 4;
    const bytes_len = pages * page;
    const raw = try std.testing.allocator.alignedAlloc(u8, .fromByteUnits(page), bytes_len);
    defer std.testing.allocator.free(raw);
    for (raw, 0..) |*b, i| b.* = @truncate(i *% 2654435761 +% (i >> 8));

    var bank = try host_bank.HostBank.fromMmapBytes(raw);
    defer bank.unreg();
    try std.testing.expect(bank.devPtr() == @intFromPtr(raw.ptr));
    try std.testing.expectEqual(bytes_len, bank.len);

    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);

    // H2D ASYNC desde la región registrada (fuente host pineada).
    const dev = try cudaz.cuMemAlloc(bytes_len);
    defer cudaz.cuMemFree(dev);
    try cudaz.cuMemcpyHtoDAsync(dev, bank.devPtr(), bytes_len, stream);

    // Evento de completación estilo pool (ready event).
    const ev = try cudaz.cuEventCreate(0);
    defer cudaz.cuEventDestroy(ev);
    try cudaz.cuEventRecord(ev, stream);
    try cudaz.cuEventSynchronize(ev);

    // D2H a buffer normal y comparación bit-exacta.
    const back = try std.testing.allocator.alloc(u8, bytes_len);
    defer std.testing.allocator.free(back);
    try cudaz.cuMemcpyDtoH(@intFromPtr(back.ptr), dev, bytes_len);
    try std.testing.expectEqualSlices(u8, raw, back);
}

test "HostBank: doble registro falla con AlreadyRegistered y unreg es idempotente" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    try cudaz.ensureContext();

    const raw = try std.testing.allocator.alignedAlloc(u8, .fromByteUnits(page), page);
    defer std.testing.allocator.free(raw);

    var bank = try host_bank.HostBank.fromMmapBytes(raw);
    defer bank.unreg();

    // Segundo registro de la MISMA región debe rechazarse limpiamente.
    try testing.expectError(error.AlreadyRegistered, ext_mem.hostRegister(@ptrCast(@constCast(raw.ptr)), page, ext_mem.CU_MEMHOSTREGISTER_DEVICEMAP));

    // Unreg doble: el segundo es no-op (registered=false).
    bank.unreg();
    bank.unreg();
    // Y un unregister directo sobre región ya liberada da NotRegistered.
    try testing.expectError(error.NotRegistered, ext_mem.hostUnregister(@ptrCast(@constCast(raw.ptr))));
}

test "HostBank: fromFileMmapWhole registra mmap completo y cubre tensores 32B-alineados internos" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    try cudaz.ensureContext();

    const pages = 3;
    const raw = try std.testing.allocator.alignedAlloc(u8, .fromByteUnits(page), pages * page);
    defer std.testing.allocator.free(raw);
    @memset(raw, 0);

    // "tensor" interno NO page-aligned (offset 512, tamaño no múltiplo de
    // página): el patrón whole-mmap lo cubre sin registros extra.
    const inner_off: usize = 512;
    const inner_len: usize = 1000;
    for (raw[inner_off .. inner_off + inner_len], 0..) |*b, i| b.* = @truncate(i +% 7);

    var bank = try host_bank.HostBank.fromFileMmapWhole(raw);
    defer bank.unreg();
    try std.testing.expectEqual(pages * page, bank.len);

    // H2D async del tensor INTERIOR usando devPtr del banco + offset.
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);
    const dev = try cudaz.cuMemAlloc(inner_len);
    defer cudaz.cuMemFree(dev);
    try cudaz.cuMemcpyHtoDAsync(dev, bank.devPtr() + inner_off, inner_len, stream);
    try cudaz.cuStreamSynchronize(stream);

    const back = try std.testing.allocator.alloc(u8, inner_len);
    defer std.testing.allocator.free(back);
    try cudaz.cuMemcpyDtoH(@intFromPtr(back.ptr), dev, inner_len);
    try std.testing.expectEqualSlices(u8, raw[inner_off .. inner_off + inner_len], back);
}
