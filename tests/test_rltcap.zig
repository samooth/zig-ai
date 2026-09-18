const std = @import("std");

fn stdoutPrint(io: std.Io, comptime fmt: []const u8, args: anytype) !void {
    var stdout_buf: [256]u8 = undefined;
    const stdout_file = std.Io.File.stdout();
    var stdout_writer = stdout_file.writer(io, &stdout_buf);
    try stdout_writer.interface.print(fmt, args);
}
const train = @import("train");

test "readRltcap: roundtrip desde fixture E2E (0.8B, wiki12k)" {
    const gpa = std.testing.allocator;
    const path = "/tmp/rltcap_test.rltcap";

    const cap = train.readRltcap(gpa, path) catch |err| {
        try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "[rltcap_test] skip: fixture no encontrada ({s}): {}\n", .{ path, err });
        return error.SkipZigTest;
    };
    defer {
        gpa.free(cap.tokens);
        gpa.free(cap.e);
    }

    try std.testing.expectEqual(cap.header.n_layers, @as(u32, 1));
    try std.testing.expect(cap.header.T > 0);
    try std.testing.expect(cap.header.d > 0);
    try std.testing.expectEqual(cap.tokens.len, cap.header.T);
    try std.testing.expectEqual(cap.e.len, cap.header.T * cap.header.d);

    // El primer token del corpus wiki12k suele ser BOS o whitespace
    try std.testing.expect(cap.tokens[0] != 0);

    // Sanity: e[0] es un f32 no-NaN
    try std.testing.expect(!std.math.isNan(cap.e[0]));
}
