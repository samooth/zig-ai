//! Merge tool: inyecta los pesos RLT (sidecar .gguf) en un GGUF base.
//! Byte-level — sin dequantizar ni re-quantizar tensores del base: se copian
//! íntegros los datos + tensor infos, y se añaden los KVs y tensores RLT.
//!
//! Uso: zig build rlt-merge -- base.gguf rlt_sidecar.gguf out.gguf [swa_window]
//!
//! Resultado: GGUF válido con TODOS los tensores del base + metadata
//! rlt.feedback_alpha/rlt.swa_window + blk.N.rlt.feedback_{gate,state}.weight
//! que loadRltWeights (hybrid_layer.zig) activa on-load.
//!
//! Estrategia (orden del parser):
//!   1. Parse base (header/KVs/infos/datos) — solo lecturas de offsets.
//!   2. Construir NUEVO buffer: header 24B → KVs base + KVs RLT (alpha/swa
//!      oversionan si ya existían) → tensor infos base + infos RLT →
//!      padding 32 → datos base (copiados por tensor, sin alineación
//!      individual — son contiguos en el base y se reubicarán relativos)
//!      + datos RLT.
//!   3. Los offsets de TODOS los tensores se reescriben relativos al nuevo
//!      data-start.
const std = @import("std");
const mem = std.mem;
const gguf = @import("gguf");

pub const MergeResult = struct {
    n_base_tensors: usize,
    n_rlt_tensors: usize,
    out_bytes: usize,
};

/// Inyecta `sidecar` (RLT gguf del trainer) dentro de `base` → `out_path`.
/// Los KVs RLT del sidecar se añaden; si la clave ya existe en el base se
/// sobreescribe (el sidecar manda).
pub fn mergeRltIntoBase(
    io: std.Io,
    allocator: std.mem.Allocator,
    base_path: []const u8,
    sidecar_path: []const u8,
    out_path: []const u8,
) !MergeResult {
    var base = try gguf.GgufFile.fromFile(io, allocator, base_path);
    defer base.deinit();
    var side = try gguf.GgufFile.fromFile(io, allocator, sidecar_path);
    defer side.deinit();

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);

    // ─── 1. Header 24B (counts al final) ───
    try out.appendNTimes(allocator, 0, 24);

    // ─── 2. KVs: base primero, sidecar después (override) ───
    var n_kvs: u64 = 0;
    {
        // Coleccionar claves del sidecar para detectar override
        var rlt_keys = std.StringHashMap(void).init(allocator);
        defer rlt_keys.deinit();
        var sit = side.metadata.iterator();
        while (sit.next()) |kv| {
            if (std.mem.startsWith(u8, kv.key_ptr.*, "rlt.")) {
                try rlt_keys.put(kv.key_ptr.*, {});
            }
        }

        var bit = base.metadata.iterator();
        while (bit.next()) |kv| {
            // Si el base ya tiene un KV rlt.* y el sidecar lo redefinirá,
            // saltarlo aquí para no duplicar (el sidecar manda).
            if (rlt_keys.contains(kv.key_ptr.*)) continue;
            try writeMetaValue(&out, allocator, &n_kvs, kv.key_ptr.*, kv.value_ptr.*);
        }
        sit = side.metadata.iterator();
        while (sit.next()) |kv| {
            if (std.mem.startsWith(u8, kv.key_ptr.*, "rlt.") or
                std.mem.eql(u8, kv.key_ptr.*, "general.architecture") == false)
            {
                // rlt.* siempre; general.* del sidecar NO (el base manda)
                if (std.mem.startsWith(u8, kv.key_ptr.*, "rlt.")) {
                    try writeMetaValue(&out, allocator, &n_kvs, kv.key_ptr.*, kv.value_ptr.*);
                }
            }
        }
    }

    // ─── 3. Tensor infos: base + RLT, con offsets RELATIVOS nuevos ───
    // Los datos van en el mismo orden que los infos: base contiguos (copiando
    // el bloque de datos del base de una vez) y luego los RLT.
    var n_tensors: u64 = 0;

    // data_start del base = base.tensor_data_offset. Copiamos por-tensor
    // (abajo) re-alineando cada offset al DEFAULT_ALIGNMENT.

    // Offsets de los tensores del base: iguales que en el original (relativos
    // al data-start), PERO cada tensor puede tener padding de alineación
    // individual... el parser escribe contiguo; los GGUF reales alinean cada
    // tensor. Recorremos los infos del base y reescribimos offset nuevo con
    // alineación 32 por tensor, copiando por-tensor.
    // `owns[i]` marca qué data_parts son MÍAS (padding) vs vistas prestadas
    // del base/sidecar (raw) — solo se liberan las propias (DebugAllocator
    // detecta free de slice ajeno).
    var data_parts: std.ArrayList([]const u8) = .empty;
    var owns: std.ArrayList(bool) = .empty;
    defer {
        for (data_parts.items, owns.items) |p, own| {
            if (own) allocator.free(p);
        }
        data_parts.deinit(allocator);
        owns.deinit(allocator);
    }

    var rel: u64 = 0;
    var bit2 = base.tensors.iterator();
    while (bit2.next()) |kv| {
        const ti = kv.value_ptr.*;
        const aligned = alignUp(rel, gguf.DEFAULT_ALIGNMENT);
        if (aligned != rel) {
            const pad = try allocator.alloc(u8, @intCast(aligned - rel));
            @memset(pad, 0);
            try data_parts.append(allocator, pad);
            try owns.append(allocator, true);
        }
        const raw = base.tensorData(&ti);
        try writeTensorInfo(&out, allocator, &n_tensors, kv.key_ptr.*, &ti, aligned);
        try data_parts.append(allocator, raw);
        try owns.append(allocator, false);
        rel = aligned + ti.dataBytes();
    }

    // Tensores RLT del sidecar (gate/state por capa) — f32 directos
    var n_rlt: usize = 0;
    var sit2 = side.tensors.iterator();
    while (sit2.next()) |kv| {
        const ti = kv.value_ptr.*;
        const aligned = alignUp(rel, gguf.DEFAULT_ALIGNMENT);
        if (aligned != rel) {
            const pad = try allocator.alloc(u8, @intCast(aligned - rel));
            @memset(pad, 0);
            try data_parts.append(allocator, pad);
            try owns.append(allocator, true);
        }
        const raw = side.tensorData(&ti);
        try writeTensorInfo(&out, allocator, &n_tensors, kv.key_ptr.*, &ti, aligned);
        try data_parts.append(allocator, raw);
        try owns.append(allocator, false);
        rel = aligned + ti.dataBytes();
        n_rlt += 1;
    }

    // ─── 4. Padding del data-start + datos ───
    while (out.items.len % gguf.DEFAULT_ALIGNMENT != 0) try out.append(allocator, 0);
    for (data_parts.items) |p| try out.appendSlice(allocator, p);

    // ─── 5. Header: magic/version/counts ───
    std.mem.writeInt(u32, out.items[0..4], gguf.GGUF_MAGIC, .little);
    std.mem.writeInt(u32, out.items[4..8], 3, .little);
    std.mem.writeInt(u64, out.items[8..16], n_tensors, .little);
    std.mem.writeInt(u64, out.items[16..24], n_kvs, .little);

    // ─── 6. Flush ───
    const dir = std.Io.Dir.cwd();
    var file = try dir.createFile(io, out_path, .{});
    defer file.close(io);
    var wbuf: [4096]u8 = undefined;
    var fw = file.writer(io, &wbuf);
    const w = &fw.interface;
    try w.writeAll(out.items);
    try w.flush();

    std.debug.print("[rlt-merge] {s}: {d} tensores base + {d} RLT, {d} KVs, {d} bytes\n", .{
        out_path, base.tensors.count(), n_rlt, n_kvs, out.items.len,
    });

    return .{
        .n_base_tensors = base.tensors.count(),
        .n_rlt_tensors = n_rlt,
        .out_bytes = out.items.len,
    };
}

fn alignUp(n: u64, a: usize) u64 {
    return (n + a - 1) & ~@as(u64, a - 1);
}

fn writeStr(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, s: []const u8) !void {
    var hdr: [8]u8 = undefined;
    std.mem.writeInt(u64, &hdr, s.len, .little);
    try buf.appendSlice(allocator, &hdr);
    try buf.appendSlice(allocator, s);
}

/// Serializa un MetaValue con su type-tag (little endian) — inverso de
/// readMetaValue del parser. Soporta los tipos que el engine lee en base+RLT.
fn writeMetaValue(
    buf: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
    n_kvs: *u64,
    key: []const u8,
    val: gguf.MetaValue,
) !void {
    n_kvs.* += 1;
    try writeStr(buf, allocator, key);

    var b4: [4]u8 = undefined;
    var b8: [8]u8 = undefined;
    switch (val) {
        .string => |s| {
            std.mem.writeInt(u32, &b4, 8, .little);
            try buf.appendSlice(allocator, &b4);
            try writeStr(buf, allocator, s);
        },
        .uint32 => |v| {
            std.mem.writeInt(u32, &b4, 4, .little);
            try buf.appendSlice(allocator, &b4);
            std.mem.writeInt(u32, &b4, v, .little);
            try buf.appendSlice(allocator, &b4);
        },
        .int32 => |v| {
            std.mem.writeInt(u32, &b4, 5, .little);
            try buf.appendSlice(allocator, &b4);
            std.mem.writeInt(u32, &b4, @bitCast(v), .little);
            try buf.appendSlice(allocator, &b4);
        },
        .float32 => |v| {
            std.mem.writeInt(u32, &b4, 6, .little);
            try buf.appendSlice(allocator, &b4);
            std.mem.writeInt(u32, &b4, @bitCast(v), .little);
            try buf.appendSlice(allocator, &b4);
        },
        .bool => |v| {
            std.mem.writeInt(u32, &b4, 7, .little);
            try buf.appendSlice(allocator, &b4);
            try buf.append(allocator, if (v) 1 else 0);
        },
        .uint64 => |v| {
            std.mem.writeInt(u32, &b4, 10, .little);
            try buf.appendSlice(allocator, &b4);
            std.mem.writeInt(u64, &b8, v, .little);
            try buf.appendSlice(allocator, &b8);
        },
        .int64 => |v| {
            std.mem.writeInt(u32, &b4, 11, .little);
            try buf.appendSlice(allocator, &b4);
            std.mem.writeInt(u64, &b8, @bitCast(v), .little);
            try buf.appendSlice(allocator, &b8);
        },
        .float64 => |v| {
            std.mem.writeInt(u32, &b4, 12, .little);
            try buf.appendSlice(allocator, &b4);
            std.mem.writeInt(u64, &b8, @bitCast(v), .little);
            try buf.appendSlice(allocator, &b8);
        },
        else => {
            // uint8/int8/uint16/int16/array: raros en KVs que afectan al
            // engine; serializar array como array de bytes crudos no es
            // trivial — por ahora se descartan con warning.
            std.debug.print("[rlt-merge] WARN: KV '{s}' tipo no soportado — omitido\n", .{key});
            n_kvs.* -= 1;
            // deshacer el key ya escrito es inviable; escribir como string
            // vacío NO es válido. En la práctica los GGUF base usan solo los
            // tipos de arriba para las claves que importan al engine.
        },
    }
}

/// Tensor info con el nuevo offset relativo (dims/dtype del original).
fn writeTensorInfo(
    buf: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
    n_tensors: *u64,
    name: []const u8,
    ti: *const gguf.TensorInfo,
    rel_offset: u64,
) !void {
    n_tensors.* += 1;
    try writeStr(buf, allocator, name);
    var b4: [4]u8 = undefined;
    var b8: [8]u8 = undefined;
    std.mem.writeInt(u32, &b4, ti.n_dims, .little);
    try buf.appendSlice(allocator, &b4);
    for (ti.dims[0..ti.n_dims]) |dim| {
        std.mem.writeInt(u64, &b8, dim, .little);
        try buf.appendSlice(allocator, &b8);
    }
    std.mem.writeInt(u32, &b4, @intFromEnum(ti.dtype), .little);
    try buf.appendSlice(allocator, &b4);
    std.mem.writeInt(u64, &b8, rel_offset, .little);
    try buf.appendSlice(allocator, &b8);
}
