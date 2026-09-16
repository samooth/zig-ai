//! GGUF writer para pesos RLT entrenados — tensores f32 + metadata.
//!
//! Formato verificado contra el parser del engine (src/loader/gguf.zig `parse`):
//!   - Header v3: "GGUF" (u32 LE) + version 3 (u32) + n_tensors (u64)
//!     + n_kvs (u64) = 24 bytes SIEMPRE al inicio.
//!   - Orden de secciones: header → KVs → tensor infos → padding a 32 → datos.
//!     (El parser lee en ese orden — no reordenar.)
//!   - KV: [len(u64)|key][MetaValueType(u32)|valor]. Tipos: string=8,
//!     float32=6, uint32=4.
//!   - Tensor info: [len|name][n_dims(u32)][dims(u64)...][ggml_type(u32)]
//!     [offset(u64)] — offset RELATIVO al data-start (post-padding).
//!   - Datos: DEFAULT_ALIGNMENT 32; f32 little-endian (@bitCast+writeInt).
//!
//! Contrato de tensores con loadRltWeights (hybrid_layer.zig):
//!   blk.{N}.rlt.feedback_gate.weight  dims [2d, d]  — GGUF [in, out]; el
//!     engine (loadGgufF32) transpone a row-major [d, 2d] = layout train.
//!     ⇒ export = transpuesta de w_gate: gguf[r*d + c] = w_gate[c*2d + r].
//!   blk.{N}.rlt.feedback_state.weight dims [d, d]   — cuadrado, sin cambio.
//!   dtype f32 (ggml_type 0). v1 sin bias (engine asume cero).
//!
//! Metadata que activa el feedback en el engine:
//!   rlt.feedback_alpha (float32) — >0 = ON, 0/ausente = OFF
//!   rlt.swa_window (uint32, opcional) — solo se escribe si > 0
const std = @import("std");
const mem = std.mem;

/// Alineación del data section (DEFAULT_ALIGNMENT del parser).
pub const GGUF_ALIGNMENT: usize = 32;

pub const RltExportConfig = struct {
    num_layers: usize,
    d: usize,
    alpha: f32,
    /// Ventana SWA opcional (rlt.swa_window); 0 = omitir el KV.
    swa_window: u32 = 0,
};

/// Pesos de UNA capa en layout de entrenamiento (row-major).
pub const LayerWeights = struct {
    w_gate: []const f32, // [d, 2*d]
    w_state: []const f32, // [d, d]
};

/// Escribe un sidecar GGUF con los pesos RLT (f32) legible por el engine.
pub fn writeRltGguf(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
    config: RltExportConfig,
    layer_weights: []const LayerWeights,
) !void {
    std.debug.assert(layer_weights.len == config.num_layers);

    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    // ─── 1. Header v3 (24 bytes, counts parchea al final) ───
    try buf.appendNTimes(allocator, 0, 24);

    // ─── 2. Metadata KVs (ANTES de tensor infos — orden del parser) ───
    var n_kvs: u64 = 0;
    {
        var kv = KvWriter{ .buf = &buf, .allocator = allocator, .count = &n_kvs };
        try kv.str("general.architecture", "rlt");
        try kv.str("general.name", "rlt-trained");
        try kv.f32v("rlt.feedback_alpha", config.alpha);
        if (config.swa_window > 0) try kv.u32v("rlt.swa_window", config.swa_window);
    }

    // ─── 3. Tensor infos + datos (contiguos: por capa, gate luego state) ───
    // Los buffers f32 (copias propias) se retienen en `copies` para liberarlos
    // con el tipo/alignment correcto (f32, align 4); `datas` solo VEE los bytes.
    var copies: std.ArrayList([]f32) = .empty;
    var datas: std.ArrayList([]const u8) = .empty;
    defer {
        for (copies.items) |c| allocator.free(c);
        copies.deinit(allocator);
        datas.deinit(allocator);
    }
    var n_tensors: u64 = 0;
    var rel: u64 = 0;
    for (layer_weights, 0..) |lw, li| {
        std.debug.assert(lw.w_gate.len == config.d * 2 * config.d);
        std.debug.assert(lw.w_state.len == config.d * config.d);

        // gate: train [d,2d] → GGUF [in=2d, out=d]: gguf[r*d+c] = w_gate[c*2d+r]
        const gate_gguf = try allocator.alloc(f32, config.d * 2 * config.d);
        for (0..2 * config.d) |r| {
            for (0..config.d) |c| {
                gate_gguf[r * config.d + c] = lw.w_gate[c * (2 * config.d) + r];
            }
        }
        try tensorInfo(&buf, allocator, li, "gate", &.{ 2 * config.d, config.d }, rel);
        n_tensors += 1;
        try copies.append(allocator, gate_gguf);
        try datas.append(allocator, mem.sliceAsBytes(gate_gguf));
        rel += gate_gguf.len * 4;

        // state: [d,d] cuadrado — layout GGUF idéntico al de train. Copia.
        const state_gguf = try allocator.alloc(f32, config.d * config.d);
        @memcpy(state_gguf, lw.w_state);
        try tensorInfo(&buf, allocator, li, "state", &.{ config.d, config.d }, rel);
        n_tensors += 1;
        try copies.append(allocator, state_gguf);
        try datas.append(allocator, mem.sliceAsBytes(state_gguf));
        rel += state_gguf.len * 4;
    }

    // ─── 4. Padding a 32 + datos de tensores ───
    while (buf.items.len % GGUF_ALIGNMENT != 0) try buf.append(allocator, 0);
    for (datas.items) |bytes| try buf.appendSlice(allocator, bytes);

    // ─── 5. Parchear header completo: magic+version (bytes 0..8) y counts ───
    std.mem.writeInt(u32, buf.items[0..4], 0x46554747, .little); // "GGUF"
    std.mem.writeInt(u32, buf.items[4..8], 3, .little); // version 3
    std.mem.writeInt(u64, buf.items[8..16], n_tensors, .little);
    std.mem.writeInt(u64, buf.items[16..24], n_kvs, .little);

    // ─── 6. Flush a disco ───
    const dir = std.Io.Dir.cwd();
    var file = try dir.createFile(io, path, .{});
    defer file.close(io);
    var wbuf: [4096]u8 = undefined;
    var fw = file.writer(io, &wbuf);
    const w = &fw.interface;
    try w.writeAll(buf.items);
    try w.flush();

    std.debug.print("[train] RLT GGUF: {s} ({d} bytes, {d} capas, d={d}, alpha={d:.4})\n", .{
        path, buf.items.len, config.num_layers, config.d, config.alpha,
    });
}

/// Escritor de KVs con contabilidad del count (el header lo necesita).
const KvWriter = struct {
    buf: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
    count: *u64,

    fn str(self: *KvWriter, name: []const u8, val: []const u8) !void {
        self.count.* += 1;
        try writeStr(self.buf, self.allocator, name);
        var t: [4]u8 = undefined;
        std.mem.writeInt(u32, &t, 8, .little); // MetaValueType.string
        try self.buf.appendSlice(self.allocator, &t);
        try writeStr(self.buf, self.allocator, val);
    }

    fn f32v(self: *KvWriter, name: []const u8, val: f32) !void {
        self.count.* += 1;
        try writeStr(self.buf, self.allocator, name);
        var t: [4]u8 = undefined;
        std.mem.writeInt(u32, &t, 6, .little); // MetaValueType.float32
        try self.buf.appendSlice(self.allocator, &t);
        var b: [4]u8 = undefined;
        std.mem.writeInt(u32, &b, @bitCast(val), .little);
        try self.buf.appendSlice(self.allocator, &b);
    }

    fn u32v(self: *KvWriter, name: []const u8, val: u32) !void {
        self.count.* += 1;
        try writeStr(self.buf, self.allocator, name);
        var t: [4]u8 = undefined;
        std.mem.writeInt(u32, &t, 4, .little); // MetaValueType.uint32
        try self.buf.appendSlice(self.allocator, &t);
        var b: [4]u8 = undefined;
        std.mem.writeInt(u32, &b, val, .little);
        try self.buf.appendSlice(self.allocator, &b);
    }
};

fn writeStr(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, s: []const u8) !void {
    var hdr: [8]u8 = undefined;
    std.mem.writeInt(u64, &hdr, s.len, .little);
    try buf.appendSlice(allocator, &hdr);
    try buf.appendSlice(allocator, s);
}

/// Un tensor info completo con nombre `blk.{N}.rlt.feedback_{kind}.weight`.
fn tensorInfo(
    buf: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
    layer_idx: usize,
    kind: []const u8, // "gate" | "state"
    dims: []const u64,
    rel_offset: u64,
) !void {
    var name_buf: [64]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "blk.{d}.rlt.feedback_{s}.weight", .{ layer_idx, kind });
    try writeStr(buf, allocator, name);

    var b4: [4]u8 = undefined;
    var b8: [8]u8 = undefined;
    std.mem.writeInt(u32, &b4, @intCast(dims.len), .little);
    try buf.appendSlice(allocator, &b4);
    for (dims) |dim| {
        std.mem.writeInt(u64, &b8, dim, .little);
        try buf.appendSlice(allocator, &b8);
    }
    std.mem.writeInt(u32, &b4, 0, .little); // GgmlType.f32
    try buf.appendSlice(allocator, &b4);
    std.mem.writeInt(u64, &b8, rel_offset, .little);
    try buf.appendSlice(allocator, &b8);
}
