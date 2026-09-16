//! NullWriter — Writer que descarta todo (para silenciar el CLI-legacy
//! cuando el engine corre embebido en el server).
//!
//! runHybridInferenceSink imprime ~53 banners de progreso a `stdout`.
//! En el server pasamos un NullWriter y todo el output real va por el
//! TokenSink (SSE/NDJSON). El CLI real sigue con su stdout normal.

const std = @import("std");

/// Writer con drain que acepta y tira los bytes.
pub const NullWriter = struct {
    interface: std.Io.Writer,

    pub fn init(buf: []u8) NullWriter {
        return .{ .interface = .{
            .vtable = &vtable,
            .buffer = buf,
        } };
    }

    const vtable: std.Io.Writer.VTable = .{
        .drain = drain,
    };

    fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) error{WriteFailed}!usize {
        // Descarta el buffer pendiente + data: reporta TODO consumido para
        // que el caller continúe sin reintentos.
        w.end = 0;
        var total: usize = splat;
        for (data) |d| total += d.len;
        return total;
    }
};

test "null writer discards prints without error" {
    var buf: [256]u8 = undefined;
    var nw = NullWriter.init(&buf);
    const w = &nw.interface;
    try w.print("banner {d} ignorado\n", .{123});
    try w.writeAll("más basura");
    try w.flush();
    try std.testing.expect(w.buffered().len == 0);
}
