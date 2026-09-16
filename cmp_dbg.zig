const std = @import("std");
const gguf = @import("src/loader/gguf.zig");
pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const dir = std.Io.Dir.cwd();
    const data = try dir.readFileAlloc(io, "/tmp/ref_iq4_nl.bin", std.heap.page_allocator, .unlimited);
    const n: usize = 320;
    const qsize = data.len - n * 4;
    const q = data[0..qsize];
    const ref = std.mem.bytesAsSlice(f32, data[qsize..]);
    var out = try std.heap.page_allocator.alloc(f32, n);
    gguf.dequantIq4_nl(q, out);
    std.debug.print("q first bytes: {x} {x} {x} {x} {x} {x}\n", .{ q[0], q[1], q[2], q[3], q[4], q[5] });
    for (0..8) |k| {
        std.debug.print("idx={d} ref={d} out={d}\n", .{ k, ref[k], out[k] });
    }
}
