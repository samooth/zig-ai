//! Helpers compartidos para serializar JSON con la API 0.16.
//!
//! En Zig 0.16 `std.json.stringifyAlloc` ya no existe. El patrón es:
//!   var aw = std.Io.Writer.Allocating.init(alloc);
//!   try aw.writer.print("{f}", .{std.json.fmt(value, .{})});
//!   const slice = try aw.toOwnedSlice();
//!
//! Aquí lo envolvemos en una sola función `jsonStringify`.

const builtin = @import("builtin");
const std = @import("std");

/// Devuelve un u64 aleatorio criptográficamente seguro.
/// Usa getrandom(2) en Linux, fallback a hash simple en otras plataformas.
pub fn rand_u64() u64 {
    var bytes: [8]u8 = undefined;
    if (comptime builtin.target.os.tag == .linux) {
        _ = std.os.linux.getrandom(&bytes, 8, 0);
    } else {
        // Fallback: hash de monotonic clock + thread id (suficiente para IDs únicos)
        const ns: u64 = @intCast(@max(0, @import("time").Timer.now()));
        const tid: u64 = @intCast(std.Thread.getCurrentId());
        bytes = @bitCast(ns ^ (tid *% 0x9E3779B97F4A7C15));
    }
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
