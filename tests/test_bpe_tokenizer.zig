//! Tests unitarios para BPETokenizer.decodeOne (stack buffer, sin heap).
//! Ejecutar con: zig build test -Dtest-filter="bpe_tokenizer"

const std = @import("std");
const bpe = @import("tokenizer");

test "bpe_tokenizer decodeOne parity vs decode (dummy vocab)" {
    const allocator = std.testing.allocator;
    var tok = try bpe.BPETokenizer.initDummy(allocator);
    defer tok.deinit();

    const test_ids = &[_]u32{ 0, 1, 42, 127, 128, 255, 256, 257, 999 };

    for (test_ids) |id| {
        var buf: [256]u8 = undefined;
        const len = try tok.decodeOne(id, &buf);
        const piece = buf[0..len];

        const full = try tok.decode(&[_]u32{id}, allocator);
        defer allocator.free(full);

        try std.testing.expectEqualSlices(u8, full, piece);
    }
}

test "bpe_tokenizer decodeOne unk fallback" {
    const allocator = std.testing.allocator;
    var tok = try bpe.BPETokenizer.initDummy(allocator);
    defer tok.deinit();

    var buf: [256]u8 = undefined;
    const len = try tok.decodeOne(999999, &buf);
    const piece = buf[0..len];

    try std.testing.expectEqualSlices(u8, "<unk>", piece);
    try std.testing.expectEqual(@as(usize, 5), len);
}

test "bpe_tokenizer decodeOne BufferTooSmall" {
    const allocator = std.testing.allocator;
    var tok = try bpe.BPETokenizer.initDummy(allocator);
    defer tok.deinit();

    var buf: [2]u8 = undefined;
    // "<unk>" son 5 bytes → debe fallar con buffer chico
    try std.testing.expectError(error.BufferTooSmall, tok.decodeOne(999999, &buf));
}
