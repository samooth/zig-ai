//! moe_make_fixture — escribe un GGUF MoE VÁLIDO y pequeño a disco para
//! probar el pipeline offload de lane-e sin depender de descargas.
//!
//! Geometría default: 4 capas × 8 expertos × top-2, n_embd=256, ff=512,
//! expertos q4_1 (bancos fusionados `_exps`, naming estándar llama.cpp),
//! routers f32. Pesos aleatorios acotados cuantizados con encoder q4_1 propio
//! (sin NaN/Inf).
//!
//! Uso:
//!   zig build moe-make-fixture -- [/ruta/salida.gguf] [n_capas] [n_experts]
const std = @import("std");

const N_EMBD: u64 = 256;
const FF: u64 = 512;
const Q41_BS: u64 = 32;
const Q41_BB: u64 = 20;
const Q41_T: u32 = 3; // ggml q4_1
const F32_T: u32 = 0;

fn writeStr(buf: *std.ArrayList(u8), alloc: std.mem.Allocator, str: []const u8) !void {
    var hdr: [8]u8 = undefined;
    std.mem.writeInt(u64, &hdr, str.len, .little);
    try buf.appendSlice(alloc, &hdr);
    try buf.appendSlice(alloc, str);
}

var n_kvs: u64 = 0;
var n_tensors: u64 = 0;

fn kvStr(buf: *std.ArrayList(u8), alloc: std.mem.Allocator, name: []const u8, val: []const u8) !void {
    n_kvs += 1;
    try writeStr(buf, alloc, name);
    var t: [4]u8 = undefined;
    std.mem.writeInt(u32, &t, 8, .little);
    try buf.appendSlice(alloc, &t);
    try writeStr(buf, alloc, val);
}

fn kvU64(buf: *std.ArrayList(u8), alloc: std.mem.Allocator, name: []const u8, val: u64) !void {
    n_kvs += 1;
    try writeStr(buf, alloc, name);
    var t: [4]u8 = undefined;
    std.mem.writeInt(u32, &t, 10, .little);
    try buf.appendSlice(alloc, &t);
    var b: [8]u8 = undefined;
    std.mem.writeInt(u64, &b, val, .little);
    try buf.appendSlice(alloc, &b);
}

fn tensorInfo(buf: *std.ArrayList(u8), alloc: std.mem.Allocator, name: []const u8, dims: []const u64, dtype: u32, offset: u64) !void {
    n_tensors += 1;
    try writeStr(buf, alloc, name);
    var b4: [4]u8 = undefined;
    std.mem.writeInt(u32, &b4, @intCast(dims.len), .little);
    try buf.appendSlice(alloc, &b4);
    for (dims) |d| {
        var b8: [8]u8 = undefined;
        std.mem.writeInt(u64, &b8, d, .little);
        try buf.appendSlice(alloc, &b8);
    }
    std.mem.writeInt(u32, &b4, dtype, .little);
    try buf.appendSlice(alloc, &b4);
    var b8: [8]u8 = undefined;
    std.mem.writeInt(u64, &b8, offset, .little);
    try buf.appendSlice(alloc, &b8);
}

/// Codifica `vals` como bloque q4_1 (d/m f16 + nibbles) en out[0..20].
fn encodeQ41Block(vals: []const f32, out: []u8) void {
    var mn: f32 = vals[0];
    var mx: f32 = vals[0];
    for (vals) |v| {
        mn = @min(mn, v);
        mx = @max(mx, v);
    }
    if (mx == mn) mx = mn + 1e-4;
    const d: f16 = @floatCast((mx - mn) / 15.0);
    const m16: f16 = @floatCast(mn);
    std.mem.writeInt(u16, out[0..2], @bitCast(d), .little);
    std.mem.writeInt(u16, out[2..4], @bitCast(m16), .little);
    @memset(out[4..20], 0);
    for (vals, 0..) |v, j| {
        const q: u8 = @intFromFloat(std.math.clamp((v - mn) / @as(f32, d), 0, 15));
        if (j % 2 == 0) {
            out[4 + j / 2] |= q & 0xF;
        } else {
            out[4 + j / 2] |= q << 4;
        }
    }
}

/// Rellena UN experto q4_1 con pesos aleatorios acotados.
fn fillExpertQ41(dst: []u8, wtmp: []f32, rand: *std.Random.Xoshiro256) void {
    for (wtmp) |*v| v.* = (rand.random().float(f32) - 0.5) * 0.4;
    var blk: usize = 0;
    while (blk < wtmp.len / Q41_BS) : (blk += 1) {
        encodeQ41Block(wtmp[blk * Q41_BS ..][0..Q41_BS], dst[blk * Q41_BB ..][0..Q41_BB]);
    }
}

pub fn main(init: std.process.Init) !void {
    var gpa_state = std.heap.DebugAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    const io = init.io;
    const args: std.process.Args = init.minimal.args;
    var args_it = std.process.Args.Iterator.initAllocator(args, gpa) catch
        return error.OutOfMemory;
    defer args_it.deinit();
    _ = args_it.next(); // argv[0]

    const out_path = if (args_it.next()) |a| a else "moe_fixture.gguf";
    const layers: u64 = if (args_it.next()) |a| try std.fmt.parseInt(u64, a, 10) else 4;
    const experts: u64 = if (args_it.next()) |a| try std.fmt.parseInt(u64, a, 10) else 8;
    const split_mode = if (args_it.next()) |a| std.mem.eql(u8, a, "--split") else false;

    var rng = std.Random.Xoshiro256.init(0xF117BEEF);

    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);

    // Header provisional
    try buf.appendSlice(gpa, "GGUF");
    var v4: [4]u8 = undefined;
    std.mem.writeInt(u32, &v4, 3, .little);
    try buf.appendSlice(gpa, &v4);
    var zeros8: [8]u8 = .{0} ** 8;
    try buf.appendSlice(gpa, &zeros8); // tensors (parchea)
    try buf.appendSlice(gpa, &zeros8); // kvs (parchea)

    try kvStr(&buf, gpa, "general.architecture", "olmoe");
    try kvU64(&buf, gpa, "olmoe.block_count", layers);
    try kvU64(&buf, gpa, "olmoe.embedding_length", N_EMBD);
    try kvU64(&buf, gpa, "olmoe.expert_count", experts);
    try kvU64(&buf, gpa, "olmoe.expert_used_count", 2);

    // Tabla de tensores + offsets
    var offs: std.ArrayList(u64) = .empty;
    defer offs.deinit(gpa);
    var cur: u64 = 0;

    for (0..@intCast(layers)) |il| {
        var nb: [96]u8 = undefined;
        const rn = try std.fmt.bufPrint(&nb, "blk.{d}.ffn_gate_inp.weight", .{il});
        try tensorInfo(&buf, gpa, rn, &[_]u64{ N_EMBD, experts }, F32_T, cur);
        try offs.append(gpa, cur);
        cur += N_EMBD * experts * 4;
        for ([_][]const u8{ "gate", "up", "down" }) |kd| {
            const is_down = std.mem.eql(u8, kd, "down");
            const in_d: u64 = if (is_down) FF else N_EMBD;
            const out_d: u64 = if (is_down) N_EMBD else FF;
            const eb: u64 = in_d * out_d / Q41_BS * Q41_BB;
            if (split_mode) {
                // E tensores per-experto CONTIGUOS (stacking del parser exige
                // back-to-back): mismo layout que el fusionado, nombres split.
                for (0..@intCast(experts)) |e| {
                    var nb2: [128]u8 = undefined;
                    const nm = try std.fmt.bufPrint(&nb2, "blk.{d}.ffn_{s}.experts.{d}.weight", .{ il, kd, e });
                    try tensorInfo(&buf, gpa, nm, &[_]u64{ in_d, out_d }, Q41_T, cur);
                    try offs.append(gpa, cur);
                    cur += eb;
                }
            } else {
                var nb2: [96]u8 = undefined;
                const nm = try std.fmt.bufPrint(&nb2, "blk.{d}.ffn_{s}_exps.weight", .{ il, kd });
                try tensorInfo(&buf, gpa, nm, &[_]u64{ in_d, out_d, experts }, Q41_T, cur);
                try offs.append(gpa, cur);
                cur += eb * experts;
            }
        }
    }

    const eb_any: usize = @intCast(N_EMBD * FF / Q41_BS * Q41_BB); // gate/up/down comparten numel

    // Alinear a 32 y parchear counts
    while (buf.items.len % 32 != 0) try buf.append(gpa, 0);
    std.mem.writeInt(u64, buf.items[8..][0..8], n_tensors, .little);
    std.mem.writeInt(u64, buf.items[16..][0..8], n_kvs, .little);

    // Datos: routers f32 suaves + bancos q4_1 con pesos acotados
    const data_start = buf.items.len;
    try buf.resize(gpa, data_start + @as(usize, @intCast(cur)));
    const data = buf.items[data_start..];
    @memset(data, 0);

    const wtmp = try gpa.alloc(f32, @intCast(N_EMBD * FF));
    defer gpa.free(wtmp);

    var blob_i: usize = 0;
    for (0..@intCast(layers)) |il| {
        {
            const rf: [*]f32 = @ptrCast(@alignCast(data.ptr + @as(usize, @intCast(offs.items[blob_i]))));
            for (0..@intCast(N_EMBD * experts)) |i| {
                rf[i] = (rng.random().float(f32) - 0.5) * 0.2 + @as(f32, @floatFromInt(il)) * 0.01;
            }
        }
        blob_i += 1;
        for ([_][]const u8{ "gate", "up", "down" }) |kd| {
            _ = kd;
            const is_down_dummy = false;
            _ = is_down_dummy;
            // por banco: fused = 1 blob con E expertos dentro; split = E blobs
            if (split_mode) {
                for (0..@intCast(experts)) |e| {
                    const base: usize = @intCast(offs.items[blob_i]);
                    blob_i += 1;
                    fillExpertQ41(data[base..][0..eb_any], wtmp, &rng);
                    _ = e;
                }
            } else {
                const base: usize = @intCast(offs.items[blob_i]);
                blob_i += 1;
                for (0..@intCast(experts)) |e| {
                    fillExpertQ41(data[base + e * eb_any ..][0..eb_any], wtmp, &rng);
                }
            }
        }
    }

    // Escribir fichero (Writer interface 0.16: buffer + flush obligatorio)
    const dir = std.Io.Dir.cwd();
    const f = try dir.createFile(io, out_path, .{});
    defer f.close(io);
    var wbuf: [0x4000]u8 = undefined;
    var file_writer = f.writer(io, &wbuf);
    const w = &file_writer.interface;
    try w.writeAll(buf.items);
    try w.flush();
    std.debug.print("fixture{s} escrita: {s} ({d} bytes, {d} capas × {d} expertos, ~{d:.1} MB)\n", .{
        if (split_mode) " SPLIT" else "",
        out_path,
        buf.items.len,
        layers,
        experts,
        @as(f64, @floatFromInt(buf.items.len)) / 1e6,
    });
}
