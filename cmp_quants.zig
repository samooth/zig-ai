const std = @import("std");
const gguf = @import("src/loader/gguf.zig");

fn compare(io: std.Io, name: []const u8, t: gguf.GgmlType, n: usize, alloc: std.mem.Allocator) !void {
    const dir = std.Io.Dir.cwd();
    const data = try dir.readFileAlloc(io, name, alloc, .unlimited);
    defer alloc.free(data);
    const qsize = data.len - n * 4;
    const q = data[0..qsize];
    const ref = std.mem.bytesAsSlice(f32, data[qsize..]);
    var out = try alloc.alloc(f32, n);
    defer alloc.free(out);
    const bs = t.blockSize();
    const bb = t.blockBytes();
    var nblk: usize = 0;
    var off: usize = 0;
    while (off + bb <= q.len) : (off += bb) {
        gguf.dequantBlock(t, q[off .. off + bb], out[nblk * bs ..][0..bs], bs);
        nblk += 1;
    }
    std.debug.print("  {s}: qsize={d} nblk={d} n={d} max_err={e} bad={d}\n", .{ tName(t), qsize, nblk, n, max_err, bad });
}

fn tName(t: gguf.GgmlType) []const u8 {
    return switch (t) {
        .iq4_nl => "iq4_nl",
        .iq2_xxs => "iq2_xxs",
        .iq2_xs => "iq2_xs",
        .iq3_xxs => "iq3_xxs",
        .iq1_s => "iq1_s",
        .iq2_s => "iq2_s",
        .iq1_m => "iq1_m",
        .tq1_0 => "tq1_0",
        .tq2_0 => "tq2_0",
        .mxfp4 => "mxfp4",
        else => "?",
    };
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();
    try compare(io, "/tmp/ref_iq4_nl.bin", .iq4_nl, 320, alloc);
    try compare(io, "/tmp/ref_iq2_xxs.bin", .iq2_xxs, 512, alloc);
    try compare(io, "/tmp/ref_iq2_xs.bin", .iq2_xs, 512, alloc);
    try compare(io, "/tmp/ref_iq3_xxs.bin", .iq3_xxs, 512, alloc);
    try compare(io, "/tmp/ref_iq1_s.bin", .iq1_s, 512, alloc);
    try compare(io, "/tmp/ref_iq2_s.bin", .iq2_s, 512, alloc);
    try compare(io, "/tmp/ref_iq1_m.bin", .iq1_m, 512, alloc);
    try compare(io, "/tmp/ref_tq1_0.bin", .tq1_0, 256, alloc);
    try compare(io, "/tmp/ref_tq2_0.bin", .tq2_0, 256, alloc);
    try compare(io, "/tmp/ref_mxfp4.bin", .mxfp4, 320, alloc);
}
