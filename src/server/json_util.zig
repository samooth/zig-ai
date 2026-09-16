//! Helpers compartidos para serializar JSON con la API 0.16.
//!
//! En Zig 0.16 `std.json.stringifyAlloc` ya no existe. El patrón es:
//!   var aw = std.Io.Writer.Allocating.init(alloc);
//!   try aw.writer.print("{f}", .{std.json.fmt(value, .{})});
//!   const slice = try aw.toOwnedSlice();
//!
//! Aquí lo envolvemos en una sola función `jsonStringify`.

const std = @import("std");

extern "c" fn arc4random_buf(buf: [*]u8, nbytes: usize) void;

/// Devuelve un u64 aleatorio criptográficamente seguro. Reemplazo de
/// `std.crypto.random.int(u64)` que se eliminó en Zig 0.16.
pub fn rand_u64() u64 {
    var bytes: [8]u8 = undefined;
    arc4random_buf(&bytes, 8);
    return @bitCast(bytes);
}

/// Serializa un valor a JSON string. Caller owns el returned slice.
pub fn jsonStringify(allocator: std.mem.Allocator, value: anytype) ![]u8 {
    var aw = std.Io.Writer.Allocating.init(allocator);
    defer aw.deinit();
    aw.writer.print("{f}", .{std.json.fmt(value, .{})}) catch |err| switch (err) {
        error.WriteFailed => return error.OutOfMemory,
    };
    return try aw.toOwnedSlice();
}

test "rand_u64 returns different values" {
    const a = rand_u64();
    const b = rand_u64();
    try std.testing.expect(a != b or a == 0); // extraordinarily unlikely both same
}
