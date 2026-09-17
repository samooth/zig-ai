//! Test del merge tool — base sintético + sidecar RLT → out parseable con
//! tensores base ÍNTEGROS + tensores RLT + KVs RLT.
//!
//! Genera un "base" mínimo tipo engine (attn_q.weight q8_0-like f32 simple,
//! KVs de arquitectura) con el formato GGUF v3 a mano, lo fussiona con un
//! sidecar del export_gguf (RLT entrenado) y verifica con el parser real:
//!   1. Out parsea (magic/version/counts/offsets coherentes).
//!   2. Todos los tensores base presentes con datos BIT-EXACTOS.
//!   3. Tensores blk.0.rlt.feedback_* presentes con datos bit-exactos.
//!   4. rlt.feedback_alpha == alpha del sidecar.
const std = @import("std");
const testing = std.testing;
const gguf = @import("gguf");
const export_gguf = @import("export_gguf");
const merge_tool = @import("merge_tool");

extern "c" fn unlink(path: [*:0]const u8) c_int;

fn rmTmp(path: []const u8) void {
    var zbuf: [256]u8 = undefined;
    if (path.len >= zbuf.len) return;
    @memcpy(zbuf[0..path.len], path);
    zbuf[path.len] = 0;
    _ = unlink(zbuf[0..path.len :0].ptr);
}

fn tmpPath(buf: []u8, prefix: []const u8) ![]const u8 {
    const ts: u64 = @intCast(@max(0, @import("time").wallClockSec()));
    const rand: u32 = @truncate(ts ^ (ts >> 32));
    return std.fmt.bufPrint(buf, "/tmp/rltmerge_{s}_{d}.gguf", .{ prefix, rand });
}

fn writeStr(buf: *std.ArrayList(u8), a: std.mem.Allocator, s: []const u8) !void {
    var hdr: [8]u8 = undefined;
    std.mem.writeInt(u64, &hdr, s.len, .little);
    try buf.appendSlice(a, &hdr);
    try buf.appendSlice(a, s);
}

test "merge tool: base + sidecar RLT → out parseable con todo íntegro" {
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();

    const d: usize = 8;

    // ─── Base sintético: 1 tensor f32 [d,d] "attn_q.weight" + KVs mínimos ───
    var base_buf: std.ArrayList(u8) = .empty;
    defer base_buf.deinit(allocator);

    const base_w = try allocator.alloc(f32, d * d);
    defer allocator.free(base_w);
    for (base_w, 0..) |*v, i| v.* = @floatFromInt(50 + i); // distinguibles

    var n_kvs: u64 = 0;
    var n_tensors: u64 = 0;
    try base_buf.appendNTimes(allocator, 0, 24); // header

    // KVs base
    try writeStr(&base_buf, allocator, "general.architecture");
    var t4: [4]u8 = undefined;
    var t8: [8]u8 = undefined;
    std.mem.writeInt(u32, &t4, 8, .little); // string
    try base_buf.appendSlice(allocator, &t4);
    try writeStr(&base_buf, allocator, "qwen3");
    n_kvs += 1;

    try writeStr(&base_buf, allocator, "qwen3.block_count");
    std.mem.writeInt(u32, &t4, 10, .little); // uint64
    try base_buf.appendSlice(allocator, &t4);
    std.mem.writeInt(u64, &t8, 1, .little);
    try base_buf.appendSlice(allocator, &t8);
    n_kvs += 1;

    // Tensor info (offset 0 — data-start único)
    try writeStr(&base_buf, allocator, "attn_q.weight");
    std.mem.writeInt(u32, &t4, 2, .little);
    try base_buf.appendSlice(allocator, &t4);
    std.mem.writeInt(u64, &t8, d, .little);
    try base_buf.appendSlice(allocator, &t8);
    std.mem.writeInt(u64, &t8, d, .little);
    try base_buf.appendSlice(allocator, &t8);
    std.mem.writeInt(u32, &t4, 0, .little); // f32
    try base_buf.appendSlice(allocator, &t4);
    std.mem.writeInt(u64, &t8, 0, .little); // offset
    try base_buf.appendSlice(allocator, &t8);
    n_tensors += 1;

    // Padding + datos
    while (base_buf.items.len % 32 != 0) try base_buf.append(allocator, 0);
    try base_buf.appendSlice(allocator, std.mem.sliceAsBytes(base_w));

    // Header
    std.mem.writeInt(u32, base_buf.items[0..4], gguf.GGUF_MAGIC, .little);
    std.mem.writeInt(u32, base_buf.items[4..8], 3, .little);
    std.mem.writeInt(u64, base_buf.items[8..16], n_tensors, .little);
    std.mem.writeInt(u64, base_buf.items[16..24], n_kvs, .little);

    var base_path_buf: [256]u8 = undefined;
    const base_path = try tmpPath(&base_path_buf, "base");
    {
        const file = try std.Io.Dir.cwd().createFile(io, base_path, .{});
        defer file.close(io);
        var wbuf: [4096]u8 = undefined;
        var fw = file.writer(io, &wbuf);
        try fw.interface.writeAll(base_buf.items);
        try fw.interface.flush();
    }
    defer rmTmp(base_path);

    // ─── Sidecar RLT del export_gguf ───
    const w_gate = try allocator.alloc(f32, d * 2 * d);
    defer allocator.free(w_gate);
    const w_state = try allocator.alloc(f32, d * d);
    defer allocator.free(w_state);
    for (w_gate, 0..) |*v, i| v.* = @floatFromInt(i + 1);
    for (w_state, 0..) |*v, i| v.* = @floatFromInt(1000 + i);

    var side_path_buf: [256]u8 = undefined;
    const side_path = try tmpPath(&side_path_buf, "side");
    try export_gguf.writeRltGguf(io, allocator, side_path, .{
        .num_layers = 1,
        .d = d,
        .alpha = 0.15,
    }, &.{.{ .w_gate = w_gate, .w_state = w_state }});
    defer rmTmp(side_path);

    // ─── MERGE ───
    var out_path_buf: [256]u8 = undefined;
    const out_path = try tmpPath(&out_path_buf, "out");
    const res = try merge_tool.mergeRltIntoBase(io, allocator, base_path, side_path, out_path);
    defer rmTmp(out_path);

    try testing.expectEqual(@as(usize, 1), res.n_base_tensors);
    try testing.expectEqual(@as(usize, 2), res.n_rlt_tensors);

    // ─── Verificación con el parser real ───
    var out = try gguf.GgufFile.fromFile(io, allocator, out_path);
    defer out.deinit();

    // 4. alpha
    const alpha = out.getMeta("rlt.feedback_alpha").?.asF32().?;
    try testing.expectApproxEqAbs(@as(f32, 0.15), alpha, 1e-6);

    // 2. Tensor base bit-exacto
    const q = out.getTensor("attn_q.weight") orelse return error.BaseTensorLost;
    try testing.expect(q.dtype == .f32);
    try testing.expectEqual(@as(u64, d), q.dims[0]);
    const qf = try allocator.alloc(f32, d * d);
    defer allocator.free(qf);
    try gguf.dequantTensor(q, out.tensorData(q), qf);
    for (base_w, qf, 0..) |expected, got, i| {
        if (expected != got) {
            std.debug.print("base_w mismatch en {d}: {d} != {d}\n", .{ i, expected, got });
            return error.BaseDataCorrupt;
        }
    }

    // 3. Tensores RLT bit-exactos (state: [d,d] directo)
    const s = out.getTensor("blk.0.rlt.feedback_state.weight") orelse return error.RltTensorLost;
    const sf = try allocator.alloc(f32, d * d);
    defer allocator.free(sf);
    try gguf.dequantTensor(s, out.tensorData(s), sf);
    for (w_state, sf, 0..) |expected, got, i| {
        if (expected != got) {
            std.debug.print("w_state mismatch en {d}: {d} != {d}\n", .{ i, expected, got });
            return error.RltDataCorrupt;
        }
    }

    // gate: transpuesto por export — layout GGUF [2d,d]: gguf[r*d+c]=w_gate[c*2d+r]
    const g = out.getTensor("blk.0.rlt.feedback_gate.weight") orelse return error.RltGateLost;
    try testing.expectEqual(@as(u64, 2 * d), g.dims[0]);
    try testing.expectEqual(@as(u64, d), g.dims[1]);
    const gf = try allocator.alloc(f32, 2 * d * d);
    defer allocator.free(gf);
    try gguf.dequantTensor(g, out.tensorData(g), gf);
    for (0..d) |c| {
        for (0..2 * d) |r| {
            if (gf[r * d + c] != w_gate[c * (2 * d) + r]) return error.RltGateCorrupt;
        }
    }

    // Counts
    try testing.expectEqual(@as(u64, 3), out.tensors.count()); // 1 base + 2 RLT
}
