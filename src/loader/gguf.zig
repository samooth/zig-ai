//! Parser GGUF (GGML Universal Format) — implementación pura en Zig.
//! Formato (little-endian):
//!   gguf_header_t { magic u32, version u32, tensor_count u64, metadata_kv_count u64 }
//!   metadata_kv[metadata_kv_count] { key gguf_string, value_type u32, value }
//!   tensor_info[tensor_count] { name gguf_string, n_dim u32, dims u64[], type u32, offset u64 }
//!   _padding a ALIGNMENT (general.alignment, default 32)
//!   tensor_data[..]  (offsets relativos al inicio de tensor_data)
//! Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
const std = @import("std");

pub const GGUF_MAGIC = 0x46554747; // "GGUF"
pub const DEFAULT_ALIGNMENT = 32;
pub const MAX_DIMS = 4;

pub const GgufError = error{
    InvalidMagic,
    UnsupportedVersion,
    InvalidData,
    UnsupportedDtype,
    TensorNotFound,
    OutOfMemory,
};

/// ggml_type (subset relevante + valores conocidos)
pub const GgmlType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    iq1_m = 29,
    bf16 = 30,
    tq1_0 = 34,
    tq2_0 = 35,
    mxfp4 = 39,

    pub fn fromRaw(raw: u32) GgufError!GgmlType {
        return std.enums.fromInt(GgmlType, raw) orelse GgufError.UnsupportedDtype;
    }

    /// Elementos por bloque (para tipos cuantizados; 1 para no cuantizados)
    pub fn blockSize(self: GgmlType) usize {
        return switch (self) {
            .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1 => 32,
            .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_k => 256,
            .iq2_xxs, .iq2_xs, .iq1_s, .iq3_s, .iq2_s => 256,
            .iq1_m => 256,
            .iq3_xxs => 256,
            .iq4_nl => 32,
            .iq4_xs => 256,
            .tq1_0 => 256,
            .tq2_0 => 256,
            .mxfp4 => 32,
            else => 1,
        };
    }

    /// Bytes por bloque
    pub fn blockBytes(self: GgmlType) usize {
        return switch (self) {
            .f32 => 4, .f16 => 2, .bf16 => 2,
            .i8 => 1, .i16 => 2, .i32 => 4, .i64 => 8, .f64 => 8,
            .q4_0 => 18, // f16 scale + 16 nibbles
            .q4_1 => 20, // f16 d + f16 m + 16 nibbles
            .q5_0 => 22, // f16 d + u32 qh + 16 nibbles
            .q5_1 => 24, // f16 d + f16 m + u32 qh + 16 nibbles
            .q8_0 => 34, // f16 d + 32 i8
            .q8_1 => 36, // f16 d + f16 s + 32 i8
            .q2_k => 84,
            .q3_k => 110,
            .q4_k => 144,
            .q5_k => 176,
            .q6_k => 210,
            .q8_k => 292,
            .iq2_xxs => 66, // f16 d + uint16 qs[32]
            .iq2_xs => 74, // f16 d + uint16 qs[32] + scales[8]
            .iq3_xxs => 98, // f16 d + qs[96]
            .iq1_s => 50, // f16 d + qs[32] + uint16 qh[8]
            .iq4_nl => 18, // f16 d + qs[16]
            .iq3_s => 110, // d f16 + qs[64] + qh[8] + signs[32] + scales[4]
            .iq2_s => 82, // f16 d + qs[64] + qh[8] + scales[8]
            .iq4_xs => 136, // f16 d + u16 scales_h + scales_l[4] + qs[128]
            .iq1_m => 56, // qs[32] + qh[16] + scales[8]
            .tq1_0 => 54, // qs[48] + qh[4] + f16 d
            .tq2_0 => 66, // qs[64] + f16 d
            .mxfp4 => 17, // e8m0 + qs[16]
        };
    }

    /// Bytes por elemento para tipos no cuantizados
    pub fn elemBytes(self: GgmlType) ?usize {
        return switch (self) {
            .f32, .i32 => 4,
            .f16, .bf16, .i16 => 2,
            .i8 => 1,
            .i64, .f64 => 8,
            else => null,
        };
    }

    pub fn isQuantized(self: GgmlType) bool {
        return self.elemBytes() == null;
    }

    pub fn name(self: GgmlType) []const u8 {
        return @tagName(self);
    }
};

/// Tipo de un valor de metadatos
pub const MetaValueType = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    bool = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,

    pub fn fromRaw(raw: u32) GgufError!MetaValueType {
        return std.enums.fromInt(MetaValueType, raw) orelse GgufError.InvalidData;
    }
};

pub const MetaValue = union(MetaValueType) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    float32: f32,
    bool: bool,
    string: []const u8,
    array: Array,
    uint64: u64,
    int64: i64,
    float64: f64,

    pub const Array = struct {
        item_type: MetaValueType,
        items: []MetaValue,
    };

    pub fn asString(self: MetaValue) ?[]const u8 {
        return switch (self) {
            .string => |s| s,
            else => null,
        };
    }

    pub fn asU32(self: MetaValue) ?u32 {
        return switch (self) {
            .uint32 => |v| v,
            .int32 => |v| @bitCast(v),
            .uint64 => |v| @intCast(v),
            .int64 => |v| @intCast(v),
            else => null,
        };
    }

    pub fn asU64(self: MetaValue) ?u64 {
        return switch (self) {
            .uint64 => |v| v,
            .int64 => |v| @intCast(v),
            .uint32 => |v| v,
            .int32 => |v| @intCast(v),
            else => null,
        };
    }

    pub fn asF32(self: MetaValue) ?f32 {
        return switch (self) {
            .float32 => |v| v,
            .float64 => |v| @floatCast(v),
            else => null,
        };
    }

    pub fn asBool(self: MetaValue) ?bool {
        return switch (self) {
            .bool => |v| v,
            .uint8 => |v| v != 0,
            else => null,
        };
    }
};

pub const TensorInfo = struct {
    name: []const u8, // préstamo a self.data
    n_dims: u32,
    dims: [MAX_DIMS]u64,
    dtype: GgmlType,
    offset: u64, // relativo al inicio de tensor_data

    pub fn shape(self: *const TensorInfo) []const u64 {
        return self.dims[0..self.n_dims];
    }

    pub fn numel(self: *const TensorInfo) u64 {
        var n: u64 = 1;
        for (self.shape()) |d| n *= d;
        return n;
    }

    /// Bytes de datos crudos en el archivo
    pub fn dataBytes(self: *const TensorInfo) usize {
        if (self.dtype.elemBytes()) |eb| {
            return @intCast(self.numel() * eb);
        }
        const bs = self.dtype.blockSize();
        const num_blocks = (self.numel() + bs - 1) / bs;
        return @intCast(num_blocks * self.dtype.blockBytes());
    }
};

pub const GgufFile = struct {
    allocator: std.mem.Allocator,
    /// Copia completa del archivo (fromBytes/fromFile) o vista mmap
    /// (fromFileMmap). El parser mantiene slices apuntando aquí, por lo que
    /// debe vivir tanto como este struct.
    data: []const u8,
    version: u32,
    alignment: u32,
    /// Offset en `data` donde empieza tensor_data
    tensor_data_offset: usize,
    tensors: std.StringHashMap(TensorInfo),
    metadata: std.StringHashMap(MetaValue),
    /// Mapeo mmap del archivo (C4). Solo válido cuando se cargó con fromFileMmap.
    io: std.Io = undefined,
    mmap: ?std.Io.File.MemoryMap = null,

    const Self = @This();

    pub fn deinit(self: *Self) void {
        destroyMetadata(self.allocator, &self.metadata);
        self.tensors.deinit();
        if (self.mmap) |*mm| {
            const file = mm.file;
            mm.destroy(self.io);
            file.close(self.io);
        } else {
            self.allocator.free(self.data);
        }
    }

    /// Cargar desde bytes (copia el buffer)
    pub fn fromBytes(allocator: std.mem.Allocator, data: []const u8) !Self {
        const owned = try allocator.dupe(u8, data);
        errdefer allocator.free(owned);
        return parse(allocator, owned);
    }

    /// Cargar desde archivo (lee el archivo completo)
    pub fn fromFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !Self {
        const dir = std.Io.Dir.cwd();
        const data = try dir.readFileAlloc(io, path, allocator, .unlimited);
        errdefer allocator.free(data);
        var self = try parse(allocator, data);
        self.io = io;
        return self;
    }

    /// Cargar desde archivo con mmap (C4): solo se leen las páginas tocadas
    /// (cabecera, metadata y tensor infos). El tensor_data se accede bajo
    /// demanda vía `tensorData()` sin copiar el archivo completo.
    pub fn fromFileMmap(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !Self {
        const dir = std.Io.Dir.cwd();
        var file = try dir.openFile(io, path, .{ .mode = .read_only });
        errdefer file.close(io);

        const size = (try file.stat(io)).size;
        var mm = try std.Io.File.MemoryMap.create(io, file, .{
            .len = @intCast(size),
            .protection = .{ .read = true },
            .populate = false,
        });
        errdefer mm.destroy(io);

        var self = try parse(allocator, mm.memory);
        self.io = io;
        self.mmap = mm;
        return self;
    }

    fn parse(allocator: std.mem.Allocator, data: []const u8) !Self {
        var r = Reader{ .data = data, .pos = 0 };

        const magic = try r.readInt(u32);
        if (magic != GGUF_MAGIC) return GgufError.InvalidMagic;
        const version = try r.readInt(u32);
        if (version < 1 or version > 3) return GgufError.UnsupportedVersion;
        const tensor_count = try r.readInt(u64);
        const metadata_kv_count = try r.readInt(u64);

        var metadata = std.StringHashMap(MetaValue).init(allocator);
        errdefer destroyMetadata(allocator, &metadata);

        var alignment: u32 = DEFAULT_ALIGNMENT;

        for (0..metadata_kv_count) |_| {
            const key = try r.readStringBorrowed();
            const value_type = try MetaValueType.fromRaw(try r.readInt(u32));
            const value = try readMetaValue(&r, value_type, allocator);
            switch (value) {
                .string => |s| {
                    if (std.mem.eql(u8, key, "general.alignment")) {
                        if (std.fmt.parseInt(u32, s, 10)) |a| {
                            alignment = a;
                        } else |_| {}
                    }
                },
                .uint32 => |v| {
                    if (std.mem.eql(u8, key, "general.alignment")) alignment = v;
                },
                else => {},
            }
            try metadata.put(key, value);
        }

        var tensors = std.StringHashMap(TensorInfo).init(allocator);
        errdefer tensors.deinit();

        for (0..tensor_count) |_| {
            const name = try r.readStringBorrowed();
            const n_dims_raw = try r.readInt(u32);
            if (n_dims_raw > MAX_DIMS) return GgufError.InvalidData;
            const n_dims: u32 = n_dims_raw;
            var dims: [MAX_DIMS]u64 = undefined;
            for (0..n_dims) |i| dims[i] = try r.readInt(u64);
            const dtype = try GgmlType.fromRaw(try r.readInt(u32));
            const offset = try r.readInt(u64);
            try tensors.put(name, .{
                .name = name,
                .n_dims = n_dims,
                .dims = dims,
                .dtype = dtype,
                .offset = offset,
            });
        }

        const tensor_data_offset = alignOffset(r.pos, alignment);
        if (tensor_count > 0 and tensor_data_offset > data.len) return GgufError.InvalidData;

        return .{
            .allocator = allocator,
            .data = data,
            .version = version,
            .alignment = alignment,
            .tensor_data_offset = tensor_data_offset,
            .tensors = tensors,
            .metadata = metadata,
        };
    }

    /// Obtener info de un tensor por nombre
    pub fn getTensor(self: *const Self, name: []const u8) ?*const TensorInfo {
        return self.tensors.getPtr(name);
    }

    /// Slice de datos crudos de un tensor (relativo al archivo completo)
    pub fn tensorData(self: *const Self, info: *const TensorInfo) []const u8 {
        const start = self.tensor_data_offset + @as(usize, @intCast(info.offset));
        const end = start + info.dataBytes();
        return self.data[start..end];
    }

    /// Obtener valor de metadatos por clave
    pub fn getMeta(self: *const Self, key: []const u8) ?MetaValue {
        return self.metadata.get(key);
    }

    pub fn arch(self: *const Self) ?[]const u8 {
        return if (self.getMeta("general.architecture")) |v| v.asString() else null;
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// C6 — Mapeo de nombres GGUF → campos del modelo
// ═══════════════════════════════════════════════════════════════════════════

/// Rol de un tensor GGUF dentro del modelo
pub const TensorRole = enum {
    token_embd,
    token_embd_norm,
    output_norm,
    output,
    attn_q,
    attn_k,
    attn_v,
    attn_o,
    attn_q_bias,
    attn_k_bias,
    attn_v_bias,
    attn_o_bias,
    attn_norm,
    ffn_norm,
    ffn_gate,
    ffn_up,
    ffn_down,
    ffn_gate_bias,
    ffn_up_bias,
    ffn_down_bias,
    rope_freqs,
    other,

    /// True si el tensor es un peso de atención (matriz)
    pub fn isAttnWeight(self: TensorRole) bool {
        return switch (self) {
            .attn_q, .attn_k, .attn_v, .attn_o => true,
            else => false,
        };
    }

    /// True si el tensor es un peso de FFN
    pub fn isFfnWeight(self: TensorRole) bool {
        return switch (self) {
            .ffn_gate, .ffn_up, .ffn_down => true,
            else => false,
        };
    }
};

pub const ParsedTensorName = struct {
    layer: ?usize,
    role: TensorRole,
};

/// Parsea un nombre de tensor GGUF y lo asigna a un rol + índice de capa.
/// Convenciones soportadas (LLaMA-like, GGUF v3):
///   token_embd.weight
///   output_norm.weight / token_embd_norm.weight
///   output.weight
///   blk.{i}.attn_q.weight|.bias      (también attn_k/v/o)
///   blk.{i}.attn_norm.weight
///   blk.{i}.ffn_norm.weight
///   blk.{i}.ffn_gate|ffn_up|ffn_down.weight|.bias
///   blk.{i}.feed_forward.w1|w2|w3.weight   (falcon)
///   blk.{i}.mlp.gate_proj|up_proj|down_proj.weight   (llama clásico)
///   rope_freqs.weight
pub fn parseTensorName(name: []const u8) ParsedTensorName {
    var rest = name;

    if (std.mem.eql(u8, rest, "token_embd.weight")) return .{ .layer = null, .role = .token_embd };
    if (std.mem.eql(u8, rest, "output_norm.weight")) return .{ .layer = null, .role = .output_norm };
    if (std.mem.eql(u8, rest, "token_embd_norm.weight")) return .{ .layer = null, .role = .token_embd_norm };
    if (std.mem.eql(u8, rest, "output.weight")) return .{ .layer = null, .role = .output };
    if (std.mem.eql(u8, rest, "rope_freqs.weight")) return .{ .layer = null, .role = .rope_freqs };

    // blk.{i}....
    if (!std.mem.startsWith(u8, rest, "blk.")) return .{ .layer = null, .role = .other };
    rest = rest["blk.".len..];

    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return .{ .layer = null, .role = .other };
    const idx_str = rest[0..dot];
    const layer = std.fmt.parseInt(usize, idx_str, 10) catch return .{ .layer = null, .role = .other };
    rest = rest[dot + 1 ..];

    // Strip trailing ".weight" / ".bias"
    const is_bias = std.mem.endsWith(u8, rest, ".bias");
    if (is_bias) rest = rest[0 .. rest.len - ".bias".len];
    if (!is_bias and std.mem.endsWith(u8, rest, ".weight")) rest = rest[0 .. rest.len - ".weight".len];

    const role: TensorRole = if (is_bias) blk: {
        break :blk if (std.mem.eql(u8, rest, "attn_q"))
            .attn_q_bias
        else if (std.mem.eql(u8, rest, "attn_k"))
            .attn_k_bias
        else if (std.mem.eql(u8, rest, "attn_v"))
            .attn_v_bias
        else if (std.mem.eql(u8, rest, "attn_o"))
            .attn_o_bias
        else if (std.mem.eql(u8, rest, "attn_output"))
            .attn_o_bias
        else if (std.mem.eql(u8, rest, "ffn_gate"))
            .ffn_gate_bias
        else if (std.mem.eql(u8, rest, "ffn_up"))
            .ffn_up_bias
        else if (std.mem.eql(u8, rest, "ffn_down"))
            .ffn_down_bias
        else
            .other;
    } else blk: {
        // aliases clásicos de FFN
        const ffn_alias: ?TensorRole = if (std.mem.eql(u8, rest, "feed_forward.w1"))
            .ffn_gate
        else if (std.mem.eql(u8, rest, "feed_forward.w3"))
            .ffn_up
        else if (std.mem.eql(u8, rest, "feed_forward.w2"))
            .ffn_down
        else if (std.mem.eql(u8, rest, "mlp.gate_proj"))
            .ffn_gate
        else if (std.mem.eql(u8, rest, "mlp.up_proj"))
            .ffn_up
        else if (std.mem.eql(u8, rest, "mlp.down_proj"))
            .ffn_down
        else
            null;
        if (ffn_alias) |r| {
            break :blk r;
        }
        break :blk if (std.mem.eql(u8, rest, "attn_q"))
            .attn_q
        else if (std.mem.eql(u8, rest, "attn_k"))
            .attn_k
        else if (std.mem.eql(u8, rest, "attn_v"))
            .attn_v
        else if (std.mem.eql(u8, rest, "attn_o") or std.mem.eql(u8, rest, "attn_output"))
            .attn_o
        else if (std.mem.eql(u8, rest, "attn_norm"))
            .attn_norm
        else if (std.mem.eql(u8, rest, "ffn_norm"))
            .ffn_norm
        else if (std.mem.eql(u8, rest, "ffn_gate"))
            .ffn_gate
        else if (std.mem.eql(u8, rest, "ffn_up"))
            .ffn_up
        else if (std.mem.eql(u8, rest, "ffn_down"))
            .ffn_down
        else
            .other;
    };

    return .{ .layer = layer, .role = role };
}

fn alignOffset(offset: usize, alignment: usize) usize {
    if (alignment == 0) return offset;
    return offset + (alignment - (offset % alignment)) % alignment;
}

/// Reader sobre el buffer con endianness little (formato GGUF estándar)
const Reader = struct {
    data: []const u8,
    pos: usize,

    fn readInt(self: *Reader, comptime T: type) GgufError!T {
        const n = @divExact(@typeInfo(T).int.bits, 8);
        if (self.pos + n > self.data.len) return GgufError.InvalidData;
        const v = std.mem.readInt(T, self.data[self.pos..][0..n], .little);
        self.pos += n;
        return v;
    }

    fn readF32(self: *Reader) GgufError!f32 {
        return @bitCast(try self.readInt(u32));
    }

    fn readF64(self: *Reader) GgufError!f64 {
        return @bitCast(try self.readInt(u64));
    }

    /// String no-null terminada; slice prestado al buffer
    fn readStringBorrowed(self: *Reader) GgufError![]const u8 {
        const len = try self.readInt(u64);
        if (self.pos + len > self.data.len) return GgufError.InvalidData;
        const s = self.data[self.pos..][0..@intCast(len)];
        self.pos += @intCast(len);
        return s;
    }
};

fn readMetaValue(r: *Reader, value_type: MetaValueType, allocator: std.mem.Allocator) GgufError!MetaValue {
    return switch (value_type) {
        .uint8 => .{ .uint8 = try r.readInt(u8) },
        .int8 => .{ .int8 = @bitCast(try r.readInt(u8)) },
        .uint16 => .{ .uint16 = try r.readInt(u16) },
        .int16 => .{ .int16 = @bitCast(try r.readInt(u16)) },
        .uint32 => .{ .uint32 = try r.readInt(u32) },
        .int32 => .{ .int32 = @bitCast(try r.readInt(u32)) },
        .float32 => .{ .float32 = try r.readF32() },
        .bool => .{ .bool = (try r.readInt(u8)) != 0 },
        .string => .{ .string = try r.readStringBorrowed() },
        .uint64 => .{ .uint64 = try r.readInt(u64) },
        .int64 => .{ .int64 = @bitCast(try r.readInt(u64)) },
        .float64 => .{ .float64 = try r.readF64() },
        .array => blk: {
            const item_type = try MetaValueType.fromRaw(try r.readInt(u32));
            const len = try r.readInt(u64);
            const items = try allocator.alloc(MetaValue, @intCast(len));
            errdefer {
                for (items) |item| freeMetaValue(allocator, item);
                allocator.free(items);
            }
            for (items) |*item| {
                item.* = try readMetaValue(r, item_type, allocator);
            }
            break :blk .{ .array = .{ .item_type = item_type, .items = items } };
        },
    };
}

fn freeMetaValue(allocator: std.mem.Allocator, value: MetaValue) void {
    switch (value) {
        .array => |arr| {
            for (arr.items) |item| freeMetaValue(allocator, item);
            if (arr.items.len > 0) allocator.free(arr.items);
        },
        else => {},
    }
}

fn destroyMetadata(allocator: std.mem.Allocator, metadata: *std.StringHashMap(MetaValue)) void {
    var it = metadata.iterator();
    while (it.next()) |entry| {
        freeMetaValue(allocator, entry.value_ptr.*);
    }
    metadata.deinit();
}

// ═══════════════════════════════════════════════════════════════════════════
// Dequantización (C5)
// ═══════════════════════════════════════════════════════════════════════════
/// F16 LE -> f32
pub fn dequantF16(bytes: []const u8, out: []f32) void {
    for (0..out.len) |i| {
        const bits = std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little);
        out[i] = @floatCast(@as(f16, @bitCast(bits)));
    }
}

/// BF16 LE -> f32
pub fn dequantBF16(bytes: []const u8, out: []f32) void {
    for (0..out.len) |i| {
        const bits = std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little);
        const f32_bits = @as(u32, @as(u32, bits) << 16);
        out[i] = @as(f32, @bitCast(f32_bits));
    }
}

/// F32 LE -> f32 (copia directa)
pub fn dequantF32(bytes: []const u8, out: []f32) void {
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. out.len * 4]);
}

/// Q8_0: bloques de 32 elementos. Cada bloque: f16 d, int8 qs[32].
/// val = d * qs[i]
pub fn dequantQ8_0(bytes: []const u8, out: []f32) void {
    const block = 32;
    const block_bytes = 34;
    var i: usize = 0;
    while (i < out.len) : (i += block) {
        const d_bits = std.mem.readInt(u16, bytes[(i / block) * block_bytes ..][0..2], .little);
        const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
        const qs = bytes[(i / block) * block_bytes + 2 ..];
        const n = @min(block, out.len - i);
        for (0..n) |j| {
            out[i + j] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(qs[j]))));
        }
    }
}

/// F64 LE -> f32 (copia directa con conversión)
pub fn dequantF64(bytes: []const u8, out: []f32) void {
    for (0..out.len) |i| {
        const bits = std.mem.readInt(u64, bytes[i * 8 ..][0..8], .little);
        out[i] = @floatCast(@as(f64, @bitCast(bits)));
    }
}

/// I8/I16/I32/I64 LE -> f32 (enteros como valor, sin escala)
pub fn dequantI8(bytes: []const u8, out: []f32) void {
    for (0..out.len) |i| {
        out[i] = @floatFromInt(@as(i8, @bitCast(bytes[i])));
    }
}

pub fn dequantI16(bytes: []const u8, out: []f32) void {
    for (0..out.len) |i| {
        out[i] = @floatFromInt(std.mem.readInt(i16, bytes[i * 2 ..][0..2], .little));
    }
}

pub fn dequantI32(bytes: []const u8, out: []f32) void {
    for (0..out.len) |i| {
        out[i] = @floatFromInt(std.mem.readInt(i32, bytes[i * 4 ..][0..4], .little));
    }
}

pub fn dequantI64(bytes: []const u8, out: []f32) void {
    for (0..out.len) |i| {
        out[i] = @floatFromInt(std.mem.readInt(i64, bytes[i * 8 ..][0..8], .little));
    }
}

/// Q5_0: bloques de 32. Cada bloque (22 bytes):
///   d f16 (offset 0), qh u32 (offset 2), qs[16] (offset 6, nibbles).
///   El bit alto de cada elemento vive en qh: bit j para elemento j (nibble
///   bajo de qs[j]), bit j+12 para elemento j+16 (nibble alto de qs[j]).
///   val = d * ((nibble | high) - 16)   (ref: ggml dequantize_row_q5_0)
pub fn dequantQ5_0(bytes: []const u8, out: []f32) void {
    const block = 32;
    const block_bytes = 22;
    var i: usize = 0;
    while (i < out.len) : (i += block) {
        const base = (i / block) * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const qh = std.mem.readInt(u32, bytes[base + 2 ..][0..4], .little);
        const qs = bytes[base + 6 ..];
        const n = @min(block, out.len - i);
        const half = @min(block / 2, n);
        for (0..half) |j| {
            const xh_0: i32 = @as(i32, @intCast((qh >> @intCast(j)) & 1)) << 4;
            const xh_1: i32 = @as(i32, @intCast((qh >> @intCast(j + 16)) & 1)) << 4;
            const lo: i32 = @as(i32, qs[j] & 0x0F) | xh_0;
            const hi: i32 = @as(i32, qs[j] >> 4) | xh_1;
            out[i + j] = d * @as(f32, @floatFromInt(lo - 16));
            if (j + 16 < n) out[i + j + 16] = d * @as(f32, @floatFromInt(hi - 16));
        }
    }
}

/// Q5_1: bloques de 32. Cada bloque (24 bytes):
///   d f16 (offset 0), m f16 (offset 2), qh u32 (offset 4), qs[16] (offset 8).
///   Igual bit layout que Q5_0 pero asimétrico: val = d*x + m.
///   (ref: ggml dequantize_row_q5_1)
pub fn dequantQ5_1(bytes: []const u8, out: []f32) void {
    const block = 32;
    const block_bytes = 24;
    var i: usize = 0;
    while (i < out.len) : (i += block) {
        const base = (i / block) * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const m: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base + 2 ..][0..2], .little))));
        const qh = std.mem.readInt(u32, bytes[base + 4 ..][0..4], .little);
        const qs = bytes[base + 8 ..];
        const n = @min(block, out.len - i);
        const half = @min(block / 2, n);
        for (0..half) |j| {
            const xh_0: i32 = @as(i32, @intCast((qh >> @intCast(j)) & 1)) << 4;
            const xh_1: i32 = @as(i32, @intCast((qh >> @intCast(j + 16)) & 1)) << 4;
            const lo: i32 = @as(i32, qs[j] & 0x0F) | xh_0;
            const hi: i32 = @as(i32, qs[j] >> 4) | xh_1;
            out[i + j] = d * @as(f32, @floatFromInt(lo)) + m;
            if (j + 16 < n) out[i + j + 16] = d * @as(f32, @floatFromInt(hi)) + m;
        }
    }
}

/// Q8_1: bloques de 32. Cada bloque (36 bytes):
///   d f16 (offset 0), s f16 (offset 2), qs[32] i8 (offset 4).
///   val = d*qs[i] + s   (ref: ggml dequantize_row_q8_1)
pub fn dequantQ8_1(bytes: []const u8, out: []f32) void {
    const block = 32;
    const block_bytes = 36;
    var i: usize = 0;
    while (i < out.len) : (i += block) {
        const base = (i / block) * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const s: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base + 2 ..][0..2], .little))));
        const qs = bytes[base + 4 ..];
        const n = @min(block, out.len - i);
        for (0..n) |j| {
            out[i + j] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(qs[j])))) + s;
        }
    }
}

/// Q4_0: bloques de 32. Cada bloque: f16 d, uint8 qs[16].
/// Layout "split" (fiel a ggml): elemento j (0..15) = nibble bajo de qs[j],
/// elemento j+16 (16..31) = nibble alto de qs[j].
/// val = d * (nibble - 8)
pub fn dequantQ4_0(bytes: []const u8, out: []f32) void {
    const block = 32;
    const block_bytes = 18;
    var i: usize = 0;
    while (i < out.len) : (i += block) {
        const d_bits = std.mem.readInt(u16, bytes[(i / block) * block_bytes ..][0..2], .little);
        const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
        const qs = bytes[(i / block) * block_bytes + 2 ..];
        const n = @min(block, out.len - i);
        const half = @min(block / 2, n);
        for (0..half) |j| {
            const lo = @as(i8, @intCast(qs[j] & 0x0F)) - 8;
            const hi = @as(i8, @intCast(qs[j] >> 4)) - 8;
            out[i + j] = d * @as(f32, @floatFromInt(lo));
            if (j + 16 < n) out[i + j + 16] = d * @as(f32, @floatFromInt(hi));
        }
    }
}

/// Q4_1: bloques de 32. Cada bloque (20 bytes):
///   d f16 (offset 0), m f16 (offset 2), qs[16] (offset 4, nibbles).
///   Layout "split": elemento j (0..15) = nibble bajo de qs[j], j+16 = alto.
///   val = d*q + m   (ref: ggml dequantize_row_q4_1)
pub fn dequantQ4_1(bytes: []const u8, out: []f32) void {
    const block = 32;
    const block_bytes = 20;
    var i: usize = 0;
    while (i < out.len) : (i += block) {
        const base = (i / block) * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const m: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base + 2 ..][0..2], .little))));
        const qs = bytes[base + 4 ..];
        const n = @min(block, out.len - i);
        const half = @min(block / 2, n);
        for (0..half) |j| {
            const lo: i32 = @intCast(qs[j] & 0x0F);
            const hi: i32 = @intCast(qs[j] >> 4);
            out[i + j] = d * @as(f32, @floatFromInt(lo)) + m;
            if (j + 16 < n) out[i + j + 16] = d * @as(f32, @floatFromInt(hi)) + m;
        }
    }
}

/// Q4_K: super-bloques de 256. Cada bloque (144 bytes):
///   d f16 (offset 0), dmin f16 (offset 2), scales[12] (offset 4, escalas
///   de 6 bits para 8 grupos de 32), qs[128] (offset 16, nibbles).
///   val = d*sc*q - min*sm   (ref: ggml dequantize_row_q4_K)
pub fn dequantQ4_K(bytes: []const u8, out: []f32) void {
    const qk = 256;
    const block_bytes = 144;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const min: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base + 2 ..][0..2], .little))));
        const scales = bytes[base + 4 .. base + 16];
        const qs = bytes[base + 16 .. base + 144];
        var is: usize = 0;
        var j: usize = 0;
        while (j < qk) : (j += 64) {
            const s1 = getScaleMinK4(is + 0, scales);
            const d1 = d * @as(f32, @floatFromInt(s1.d));
            const m1 = min * @as(f32, @floatFromInt(s1.m));
            const s2 = getScaleMinK4(is + 1, scales);
            const d2 = d * @as(f32, @floatFromInt(s2.d));
            const m2 = min * @as(f32, @floatFromInt(s2.m));
            const q = qs[(j / 64) * 32 ..];
            for (0..32) |l| {
                out[i + j + l] = d1 * @as(f32, @floatFromInt(q[l] & 0xF)) - m1;
                out[i + j + 32 + l] = d2 * @as(f32, @floatFromInt(q[l] >> 4)) - m2;
            }
            is += 2;
        }
        nb += 1;
    }
}

/// Q6_K: super-bloques de 256. Cada bloque (210 bytes):
///   ql[128] (offset 0), qh[64] (offset 128), scales[16] i8 (offset 192),
///   d f16 (offset 208).
///   val = d * sc[is] * (q - 32)   (ref: ggml dequantize_row_q6_K)
pub fn dequantQ6_K(bytes: []const u8, out: []f32) void {
    const qk = 256;
    const block_bytes = 210;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base + 208 ..][0..2], .little))));
        const ql = bytes[base .. base + 128];
        const qh = bytes[base + 128 .. base + 192];
        const sc = bytes[base + 192 .. base + 208];
        var n: usize = 0;
        while (n < qk) : (n += 128) {
            const ql2 = ql[(n / 128) * 64 ..];
            const qh2 = qh[(n / 128) * 32 ..];
            const sc2 = sc[(n / 128) * 8 ..];
            for (0..32) |l| {
                const is = l / 16;
                const q1: f32 = @floatFromInt(@as(i8, @bitCast((ql2[l] & 0xF) | ((qh2[l] >> 0) & 3) << 4)) - 32);
                const q2: f32 = @floatFromInt(@as(i8, @bitCast((ql2[l + 32] & 0xF) | ((qh2[l] >> 2) & 3) << 4)) - 32);
                const q3: f32 = @floatFromInt(@as(i8, @bitCast((ql2[l] >> 4) | ((qh2[l] >> 4) & 3) << 4)) - 32);
                const q4: f32 = @floatFromInt(@as(i8, @bitCast((ql2[l + 32] >> 4) | ((qh2[l] >> 6) & 3) << 4)) - 32);
                out[i + n + l] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc2[is + 0])))) * q1;
                out[i + n + l + 32] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc2[is + 2])))) * q2;
                out[i + n + l + 64] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc2[is + 4])))) * q3;
                out[i + n + l + 96] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc2[is + 6])))) * q4;
            }
        }
        nb += 1;
    }
}

/// Q2_K: super-bloques de 256. Cada bloque (84 bytes):
///   scales[16] (offset 0, 8 escalas de 4 bits + 8 mins de 4 bits),
///   qs[64] (offset 16), d f16 (offset 80), dmin f16 (offset 82).
///   val = dl*q - ml; q de 2 bits   (ref: ggml dequantize_row_q2_K)
pub fn dequantQ2_K(bytes: []const u8, out: []f32) void {
    const qk = 256;
    const block_bytes = 84;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base + 80 ..][0..2], .little))));
        const min: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base + 82 ..][0..2], .little))));
        const scales = bytes[base .. base + 16];
        const qs = bytes[base + 16 .. base + 80];
        var is: usize = 0;
        var n: usize = 0;
        while (n < qk) : (n += 128) {
            var shift: u8 = 0;
            for (0..4) |j| {
                const sc1 = scales[is];
                is += 1;
                const dl = d * @as(f32, @floatFromInt(sc1 & 0xF));
                const ml = min * @as(f32, @floatFromInt(sc1 >> 4));
                const sc2 = scales[is];
                is += 1;
                const dl2 = d * @as(f32, @floatFromInt(sc2 & 0xF));
                const ml2 = min * @as(f32, @floatFromInt(sc2 >> 4));
                const q = qs[(n / 128) * 32 ..];
                for (0..16) |l| {
                    const q1: i32 = @intCast((q[l] >> @as(u3, @intCast(shift))) & 3);
                    const q2: i32 = @intCast((q[l + 16] >> @as(u3, @intCast(shift))) & 3);
                    out[i + n + j * 32 + l] = dl * @as(f32, @floatFromInt(q1)) - ml;
                    out[i + n + j * 32 + 16 + l] = dl2 * @as(f32, @floatFromInt(q2)) - ml2;
                }
                shift += 2;
            }
        }
        nb += 1;
    }
}

/// Q3_K: super-bloques de 256. Cada bloque (110 bytes):
///   hmask[32] (offset 0), qs[64] (offset 32), scales[12] (offset 96),
///   d f16 (offset 108).
///   val = dl * (q - (hm?0:4)); q de 2 bits + bit alto en hmask.
///   Las 12 escalas de 6 bits se reordenan en 16 i8 (ref: dequantize_row_q3_K).
pub fn dequantQ3_K(bytes: []const u8, out: []f32) void {
    const qk = 256;
    const block_bytes = 110;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base + 108 ..][0..2], .little))));
        const hmask = bytes[base .. base + 32];
        const qs = bytes[base + 32 .. base + 96];
        const scales = bytes[base + 96 .. base + 108];
        // Reordenamiento de escalas (16 i8) fiel al C de ggml
        const kmask1: u32 = 0x03030303;
        const kmask2: u32 = 0x0f0f0f0f;
        var aux: [4]u32 = undefined;
        var aux_bytes = std.mem.sliceAsBytes(aux[0..4]);
        @memcpy(aux_bytes[0..12], scales);
        const tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const scales16 = std.mem.sliceAsBytes(aux[0..4]);
        var is: usize = 0;
        var m: u8 = 1;
        var n: usize = 0;
        while (n < qk) : (n += 128) {
            var shift: u8 = 0;
            for (0..4) |j| {
                const dl = d * @as(f32, @floatFromInt(@as(i8, @bitCast(scales16[is])) - 32));
                is += 1;
                const dl2 = d * @as(f32, @floatFromInt(@as(i8, @bitCast(scales16[is])) - 32));
                is += 1;
                const q = qs[(n / 128) * 32 ..];
                for (0..16) |l| {
                    const q1: i32 = @intCast((q[l] >> @as(u3, @intCast(shift))) & 3);
                    const q2: i32 = @intCast((q[l + 16] >> @as(u3, @intCast(shift))) & 3);
                    const h1: i32 = if (hmask[l] & m != 0) 0 else 4;
                    const h2: i32 = if (hmask[l + 16] & m != 0) 0 else 4;
                    out[i + n + j * 32 + l] = dl * @as(f32, @floatFromInt(q1 - h1));
                    out[i + n + j * 32 + 16 + l] = dl2 * @as(f32, @floatFromInt(q2 - h2));
                }
                shift += 2;
                m <<= 1;
            }
        }
        nb += 1;
    }
}

/// Q8_K: super-bloques de 256. Cada bloque (292 bytes):
///   d f32 (offset 0), qs[256] i8 (offset 4), bsums[16] i16 (offset 260,
///   ignorado al dequantizar).
///   val = d * qs[j]   (ref: ggml dequantize_row_q8_K)
pub fn dequantQ8_K(bytes: []const u8, out: []f32) void {
    const qk = 256;
    const block_bytes = 292;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const d: f32 = @as(f32, @bitCast(std.mem.readInt(u32, bytes[base..][0..4], .little)));
        const qs = bytes[base + 4 .. base + 4 + qk];
        for (0..qk) |j| {
            out[i + j] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(qs[j]))));
        }
        nb += 1;
    }
}

const ScaleMin = struct { d: u8, m: u8 };
/// Tabla de lookup no lineal para IQ4 (ref: kvalues_iq4nl en llama.cpp)
const kvalues_iq4nl = [_]i8{ -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

/// Máscaras de signo para IQ2XS/IQ3 (ref: kmask_iq2xs en ggml-common.h)
const kmask_iq2xs = [_]u8{ 1, 2, 4, 8, 16, 32, 64, 128 };

/// Grid 3x3 de 512 entradas para IQ3_S (ref: iq3s_grid en ggml-common.h).
/// Cada entrada u32 almacena 4 bytes; el byte j = (valor >> 8*j) & 0xFF
/// es el multiplicador entero del elemento j (1, 3, 5, ..., 15).
const iq3s_grid = [_]u32{
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

const ksigns_iq2xs = [_]u8{
    0, 129, 130, 3, 132, 5, 6, 135,
    136, 9, 10, 139, 12, 141, 142, 15,
    144, 17, 18, 147, 20, 149, 150, 23,
    24, 153, 154, 27, 156, 29, 30, 159,
    160, 33, 34, 163, 36, 165, 166, 39,
    40, 169, 170, 43, 172, 45, 46, 175,
    48, 177, 178, 51, 180, 53, 54, 183,
    184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71,
    72, 201, 202, 75, 204, 77, 78, 207,
    80, 209, 210, 83, 212, 85, 86, 215,
    216, 89, 90, 219, 92, 221, 222, 95,
    96, 225, 226, 99, 228, 101, 102, 231,
    232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119,
    120, 249, 250, 123, 252, 125, 126, 255,
};

const iq2xxs_grid = [_]u64{
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08, 0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819, 0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b, 0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08, 0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819, 0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808, 0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808, 0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08, 0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819, 0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808, 0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808, 0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819, 0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819, 0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19, 0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b, 0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908, 0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08, 0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819, 0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808, 0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819, 0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b, 0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808, 0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b, 0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08, 0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808, 0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819, 0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908, 0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808, 0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b, 0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19, 0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

const iq2xs_grid = [_]u64{
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08, 0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919, 0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919, 0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x080808082b080808,
    0x080808082b08082b, 0x080808082b081919, 0x080808082b082b08, 0x080808082b190819, 0x080808082b191908, 0x080808082b192b19, 0x080808082b2b0808, 0x0808081908080819,
    0x0808081908081908, 0x080808190808192b, 0x0808081908082b19, 0x0808081908190808, 0x080808190819082b, 0x0808081908191919, 0x0808081908192b08, 0x0808081908192b2b,
    0x08080819082b0819, 0x08080819082b1908, 0x0808081919080808, 0x080808191908082b, 0x0808081919081919, 0x0808081919082b08, 0x0808081919190819, 0x0808081919191908,
    0x08080819192b0808, 0x08080819192b2b08, 0x080808192b080819, 0x080808192b081908, 0x080808192b190808, 0x0808082b08080808, 0x0808082b0808082b, 0x0808082b08081919,
    0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908, 0x0808082b082b0808, 0x0808082b19080819, 0x0808082b19081908, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b082b2b, 0x0808190808080819, 0x0808190808081908, 0x080819080808192b, 0x0808190808082b19, 0x0808190808190808, 0x080819080819082b,
    0x0808190808191919, 0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819081919, 0x0808190819082b08,
    0x0808190819190819, 0x0808190819191908, 0x080819081919192b, 0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808, 0x0808191908080808,
    0x080819190808082b, 0x0808191908081919, 0x0808191908082b08, 0x0808191908190819, 0x0808191908191908, 0x08081919082b0808, 0x0808191919080819, 0x0808191919081908,
    0x0808191919190808, 0x08081919192b0819, 0x080819192b080808, 0x0808192b08080819, 0x0808192b08081908, 0x0808192b08190808, 0x0808192b082b192b, 0x0808192b19080808,
    0x0808192b1908082b, 0x0808192b2b081908, 0x08082b0808080808, 0x08082b080808082b, 0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808082b2b, 0x08082b0808190819,
    0x08082b0808191908, 0x08082b08082b0808, 0x08082b08082b1919, 0x08082b0819080819, 0x08082b0819081908, 0x08082b0819190808, 0x08082b0819192b08, 0x08082b082b080808,
    0x08082b082b2b0808, 0x08082b082b2b2b2b, 0x08082b1908080819, 0x08082b1908081908, 0x08082b1908190808, 0x08082b1919080808, 0x08082b192b080819, 0x08082b192b082b19,
    0x08082b2b08080808, 0x08082b2b082b0808, 0x08082b2b082b2b08, 0x08082b2b2b19192b, 0x08082b2b2b2b0808, 0x0819080808080819, 0x0819080808081908, 0x081908080808192b,
    0x0819080808082b19, 0x0819080808190808, 0x081908080819082b, 0x0819080808191919, 0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908, 0x0819080819080808,
    0x081908081908082b, 0x0819080819081919, 0x0819080819082b08, 0x0819080819190819, 0x0819080819191908, 0x08190808192b0808, 0x08190808192b2b2b, 0x081908082b080819,
    0x081908082b081908, 0x081908082b190808, 0x0819081908080808, 0x081908190808082b, 0x0819081908081919, 0x0819081908082b08, 0x0819081908190819, 0x0819081908191908,
    0x08190819082b0808, 0x0819081919080819, 0x0819081919081908, 0x0819081919190808, 0x081908192b080808, 0x081908192b191908, 0x081908192b19192b, 0x0819082b08080819,
    0x0819082b08081908, 0x0819082b0808192b, 0x0819082b08190808, 0x0819082b19080808, 0x0819082b192b0808, 0x0819190808080808, 0x081919080808082b, 0x0819190808081919,
    0x0819190808082b08, 0x0819190808190819, 0x0819190808191908, 0x08191908082b0808, 0x0819190819080819, 0x0819190819081908, 0x0819190819082b19, 0x0819190819190808,
    0x08191908192b1908, 0x081919082b080808, 0x0819191908080819, 0x0819191908081908, 0x0819191908190808, 0x0819191919080808, 0x0819192b08080808, 0x0819192b08191908,
    0x0819192b19082b19, 0x08192b0808080819, 0x08192b0808081908, 0x08192b0808190808, 0x08192b080819082b, 0x08192b0819080808, 0x08192b0819191908, 0x08192b082b08192b,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b19192b192b, 0x08192b2b19190819, 0x08192b2b2b2b2b19, 0x082b080808080808, 0x082b08080808082b, 0x082b080808081919,
    0x082b080808082b08, 0x082b080808082b2b, 0x082b080808190819, 0x082b080808191908, 0x082b0808082b0808, 0x082b080819080819, 0x082b080819081908, 0x082b080819190808,
    0x082b08082b080808, 0x082b08082b2b0808, 0x082b081908080819, 0x082b081908081908, 0x082b081908190808, 0x082b081919080808, 0x082b081919082b08, 0x082b0819192b1919,
    0x082b082b08080808, 0x082b082b082b082b, 0x082b082b2b080808, 0x082b082b2b2b2b08, 0x082b190808080819, 0x082b190808081908, 0x082b190808190808, 0x082b1908082b2b19,
    0x082b190819080808, 0x082b191908080808, 0x082b191919080819, 0x082b19191919082b, 0x082b19192b192b19, 0x082b192b08080819, 0x082b192b08192b2b, 0x082b192b2b2b192b,
    0x082b2b0808080808, 0x082b2b0808082b08, 0x082b2b0808082b2b, 0x082b2b08082b0808, 0x082b2b0819191919, 0x082b2b082b082b08, 0x082b2b082b2b082b, 0x082b2b19192b2b08,
    0x082b2b192b190808, 0x082b2b2b08082b08, 0x082b2b2b082b0808, 0x082b2b2b2b08082b, 0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819, 0x1908080808081908,
    0x190808080808192b, 0x1908080808082b19, 0x1908080808190808, 0x190808080819082b, 0x1908080808191919, 0x1908080808192b08, 0x19080808082b0819, 0x19080808082b1908,
    0x1908080819080808, 0x190808081908082b, 0x1908080819081919, 0x1908080819082b08, 0x1908080819082b2b, 0x1908080819190819, 0x1908080819191908, 0x19080808192b0808,
    0x19080808192b1919, 0x190808082b080819, 0x190808082b081908, 0x190808082b190808, 0x1908081908080808, 0x190808190808082b, 0x1908081908081919, 0x1908081908082b08,
    0x1908081908190819, 0x1908081908191908, 0x19080819082b0808, 0x1908081919080819, 0x1908081919081908, 0x1908081919190808, 0x190808192b080808, 0x190808192b081919,
    0x190808192b2b082b, 0x1908082b08080819, 0x1908082b08081908, 0x1908082b08190808, 0x1908082b0819082b, 0x1908082b082b2b19, 0x1908082b19080808, 0x1908190808080808,
    0x190819080808082b, 0x1908190808081919, 0x1908190808082b08, 0x1908190808190819, 0x1908190808191908, 0x1908190808192b19, 0x19081908082b0808, 0x1908190819080819,
    0x1908190819081908, 0x1908190819190808, 0x190819082b080808, 0x190819082b191908, 0x1908191908080819, 0x1908191908081908, 0x1908191908190808, 0x19081919082b1908,
    0x1908191919080808, 0x190819192b192b2b, 0x1908192b08080808, 0x1908192b08082b2b, 0x1908192b19081908, 0x1908192b19190808, 0x19082b0808080819, 0x19082b0808081908,
    0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919, 0x19082b0819191908, 0x19082b08192b082b, 0x19082b1908080808, 0x19082b1908190819, 0x19082b1919081908,
    0x19082b1919190808, 0x19082b19192b2b19, 0x19082b2b08081908, 0x1919080808080808, 0x191908080808082b, 0x1919080808081919, 0x1919080808082b08, 0x1919080808190819,
    0x1919080808191908, 0x19190808082b0808, 0x19190808082b2b08, 0x1919080819080819, 0x1919080819081908, 0x1919080819190808, 0x191908082b080808, 0x1919081908080819,
    0x1919081908081908, 0x1919081908190808, 0x1919081908191919, 0x1919081919080808, 0x191908191908082b, 0x1919082b08080808, 0x1919082b19081908, 0x1919082b2b2b2b2b,
    0x1919190808080819, 0x1919190808081908, 0x1919190808190808, 0x19191908082b0819, 0x1919190819080808, 0x19191908192b0808, 0x191919082b080819, 0x191919082b2b0819,
    0x1919191908080808, 0x1919191908082b08, 0x191919192b080808, 0x191919192b082b08, 0x1919192b082b0819, 0x1919192b192b2b08, 0x1919192b2b2b0819, 0x19192b0808080808,
    0x19192b0808191908, 0x19192b0819080819, 0x19192b0819190808, 0x19192b082b192b19, 0x19192b1908192b2b, 0x19192b1919080808, 0x19192b191908082b, 0x19192b2b2b081919,
    0x192b080808080819, 0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b080819191908, 0x192b0808192b082b, 0x192b08082b08192b, 0x192b08082b2b2b19,
    0x192b081908080808, 0x192b082b082b1908, 0x192b082b19082b2b, 0x192b082b2b19082b, 0x192b190808080808, 0x192b19080819192b, 0x192b191908190808, 0x192b191919080808,
    0x192b191919081919, 0x192b19192b2b1908, 0x192b2b0808080819, 0x192b2b08192b2b2b, 0x192b2b19082b1919, 0x192b2b2b0808192b, 0x192b2b2b19191908, 0x192b2b2b192b082b,
    0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08, 0x2b08080808190819, 0x2b08080808191908, 0x2b080808082b0808, 0x2b080808082b2b2b,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808082b080808, 0x2b0808082b08082b, 0x2b0808082b2b2b08, 0x2b0808082b2b2b2b, 0x2b08081908080819,
    0x2b08081908081908, 0x2b0808190808192b, 0x2b08081908190808, 0x2b08081919080808, 0x2b08081919190819, 0x2b08081919192b19, 0x2b08082b08080808, 0x2b08082b082b0808,
    0x2b08082b2b080808, 0x2b08082b2b08082b, 0x2b08082b2b2b0808, 0x2b08082b2b2b2b08, 0x2b08190808080819, 0x2b08190808081908, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190819080808, 0x2b081908192b0808, 0x2b0819082b082b19, 0x2b08191908080808, 0x2b08191919081908, 0x2b0819192b2b1919, 0x2b08192b08192b08,
    0x2b08192b192b2b2b, 0x2b082b0808080808, 0x2b082b0808082b08, 0x2b082b08082b1919, 0x2b082b0819192b2b, 0x2b082b082b080808, 0x2b082b082b08082b, 0x2b082b082b2b2b08,
    0x2b082b190808192b, 0x2b082b2b082b082b, 0x2b082b2b2b080808, 0x2b082b2b2b082b08, 0x2b082b2b2b19192b, 0x2b082b2b2b2b2b08, 0x2b19080808080819, 0x2b19080808081908,
    0x2b19080808190808, 0x2b19080819080808, 0x2b1908081919192b, 0x2b1908082b081908, 0x2b19081908080808, 0x2b190819082b082b, 0x2b190819192b1908, 0x2b19082b1919192b,
    0x2b19082b2b082b19, 0x2b19190808080808, 0x2b19190808081919, 0x2b19190819081908, 0x2b19190819190808, 0x2b19190819192b08, 0x2b191919082b2b19, 0x2b1919192b190808,
    0x2b1919192b19082b, 0x2b19192b19080819, 0x2b192b0819190819, 0x2b192b082b2b192b, 0x2b192b1919082b19, 0x2b192b2b08191919, 0x2b192b2b192b0808, 0x2b2b080808080808,
    0x2b2b08080808082b, 0x2b2b080808082b08, 0x2b2b080808082b2b, 0x2b2b0808082b0808, 0x2b2b0808082b2b2b, 0x2b2b08082b2b0808, 0x2b2b081919190819, 0x2b2b081919192b19,
    0x2b2b08192b2b192b, 0x2b2b082b08080808, 0x2b2b082b0808082b, 0x2b2b082b08082b08, 0x2b2b082b082b2b2b, 0x2b2b082b2b080808, 0x2b2b082b2b2b0808, 0x2b2b190819080808,
    0x2b2b19082b191919, 0x2b2b192b192b1919, 0x2b2b192b2b192b08, 0x2b2b2b0808082b2b, 0x2b2b2b08082b0808, 0x2b2b2b08082b082b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b0808,
    0x2b2b2b082b2b2b08, 0x2b2b2b1908081908, 0x2b2b2b192b081908, 0x2b2b2b192b08192b, 0x2b2b2b2b082b2b08, 0x2b2b2b2b082b2b2b, 0x2b2b2b2b2b190819, 0x2b2b2b2b2b2b2b2b,
};

const iq2s_grid = [_]u64{
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08, 0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919, 0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919, 0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x08080808192b192b,
    0x08080808192b2b19, 0x080808082b080808, 0x080808082b08082b, 0x080808082b081919, 0x080808082b082b08, 0x080808082b190819, 0x080808082b191908, 0x080808082b2b0808,
    0x080808082b2b1919, 0x080808082b2b2b2b, 0x0808081908080819, 0x0808081908081908, 0x080808190808192b, 0x0808081908082b19, 0x0808081908190808, 0x080808190819082b,
    0x0808081908191919, 0x0808081908192b08, 0x08080819082b0819, 0x08080819082b1908, 0x0808081919080808, 0x080808191908082b, 0x0808081919081919, 0x0808081919082b08,
    0x0808081919190819, 0x0808081919191908, 0x080808191919192b, 0x0808081919192b19, 0x08080819192b0808, 0x08080819192b1919, 0x08080819192b2b08, 0x080808192b080819,
    0x080808192b081908, 0x080808192b190808, 0x080808192b19082b, 0x080808192b191919, 0x080808192b2b0819, 0x080808192b2b1908, 0x0808082b08080808, 0x0808082b0808082b,
    0x0808082b08081919, 0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908, 0x0808082b082b0808, 0x0808082b082b2b2b, 0x0808082b19080819, 0x0808082b19081908,
    0x0808082b1908192b, 0x0808082b19082b19, 0x0808082b19190808, 0x0808082b19191919, 0x0808082b2b080808, 0x0808082b2b081919, 0x0808082b2b082b2b, 0x0808082b2b191908,
    0x0808082b2b2b082b, 0x0808190808080819, 0x0808190808081908, 0x080819080808192b, 0x0808190808082b19, 0x0808190808190808, 0x080819080819082b, 0x0808190808191919,
    0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908, 0x08081908082b192b, 0x08081908082b2b19, 0x0808190819080808, 0x080819081908082b, 0x0808190819081919,
    0x0808190819082b08, 0x0808190819082b2b, 0x0808190819190819, 0x0808190819191908, 0x080819081919192b, 0x0808190819192b19, 0x08081908192b0808, 0x08081908192b082b,
    0x08081908192b1919, 0x080819082b080819, 0x080819082b081908, 0x080819082b08192b, 0x080819082b082b19, 0x080819082b190808, 0x080819082b191919, 0x080819082b192b08,
    0x080819082b2b0819, 0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908081919, 0x0808191908082b08, 0x0808191908082b2b, 0x0808191908190819,
    0x0808191908191908, 0x080819190819192b, 0x0808191908192b19, 0x08081919082b0808, 0x08081919082b1919, 0x08081919082b2b08, 0x0808191919080819, 0x0808191919081908,
    0x080819191908192b, 0x0808191919082b19, 0x0808191919190808, 0x080819191919082b, 0x0808191919191919, 0x0808191919192b08, 0x08081919192b0819, 0x08081919192b1908,
    0x080819192b080808, 0x080819192b08082b, 0x080819192b081919, 0x080819192b082b08, 0x080819192b190819, 0x080819192b191908, 0x080819192b2b0808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b0808192b, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b08191919, 0x0808192b19080808, 0x0808192b19081919, 0x0808192b19082b08,
    0x0808192b19190819, 0x0808192b19191908, 0x0808192b192b0808, 0x0808192b2b080819, 0x0808192b2b081908, 0x0808192b2b190808, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808190819, 0x08082b0808191908, 0x08082b080819192b, 0x08082b0808192b19, 0x08082b08082b0808, 0x08082b08082b1919,
    0x08082b08082b2b2b, 0x08082b0819080819, 0x08082b0819081908, 0x08082b081908192b, 0x08082b0819082b19, 0x08082b0819190808, 0x08082b081919082b, 0x08082b0819191919,
    0x08082b0819192b08, 0x08082b08192b0819, 0x08082b08192b1908, 0x08082b082b080808, 0x08082b082b081919, 0x08082b082b191908, 0x08082b082b2b2b2b, 0x08082b1908080819,
    0x08082b1908081908, 0x08082b1908190808, 0x08082b190819082b, 0x08082b1908191919, 0x08082b1908192b08, 0x08082b19082b0819, 0x08082b1919080808, 0x08082b1919081919,
    0x08082b1919082b08, 0x08082b1919190819, 0x08082b1919191908, 0x08082b19192b0808, 0x08082b192b080819, 0x08082b192b190808, 0x08082b2b08080808, 0x08082b2b08190819,
    0x08082b2b08191908, 0x08082b2b082b082b, 0x08082b2b082b2b08, 0x08082b2b082b2b2b, 0x08082b2b19190808, 0x08082b2b2b192b19, 0x0819080808080819, 0x0819080808081908,
    0x081908080808192b, 0x0819080808082b19, 0x0819080808190808, 0x081908080819082b, 0x0819080808191919, 0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908,
    0x08190808082b192b, 0x0819080819080808, 0x081908081908082b, 0x0819080819081919, 0x0819080819082b08, 0x0819080819190819, 0x0819080819191908, 0x081908081919192b,
    0x0819080819192b19, 0x08190808192b0808, 0x08190808192b082b, 0x08190808192b1919, 0x08190808192b2b08, 0x081908082b080819, 0x081908082b081908, 0x081908082b08192b,
    0x081908082b190808, 0x081908082b191919, 0x081908082b192b08, 0x081908082b2b0819, 0x081908082b2b1908, 0x0819081908080808, 0x081908190808082b, 0x0819081908081919,
    0x0819081908082b08, 0x0819081908082b2b, 0x0819081908190819, 0x0819081908191908, 0x081908190819192b, 0x0819081908192b19, 0x08190819082b0808, 0x08190819082b082b,
    0x08190819082b1919, 0x08190819082b2b08, 0x0819081919080819, 0x0819081919081908, 0x081908191908192b, 0x0819081919082b19, 0x0819081919190808, 0x081908191919082b,
    0x0819081919191919, 0x0819081919192b08, 0x08190819192b0819, 0x08190819192b1908, 0x081908192b080808, 0x081908192b08082b, 0x081908192b081919, 0x081908192b082b08,
    0x081908192b190819, 0x081908192b191908, 0x0819082b08080819, 0x0819082b08081908, 0x0819082b08082b19, 0x0819082b08190808, 0x0819082b08191919, 0x0819082b082b0819,
    0x0819082b082b1908, 0x0819082b19080808, 0x0819082b19081919, 0x0819082b19190819, 0x0819082b19191908, 0x0819082b2b080819, 0x0819082b2b081908, 0x0819082b2b190808,
    0x0819190808080808, 0x081919080808082b, 0x0819190808081919, 0x0819190808082b08, 0x0819190808190819, 0x0819190808191908, 0x081919080819192b, 0x0819190808192b19,
    0x08191908082b0808, 0x08191908082b1919, 0x08191908082b2b08, 0x0819190819080819, 0x0819190819081908, 0x081919081908192b, 0x0819190819082b19, 0x0819190819190808,
    0x081919081919082b, 0x0819190819191919, 0x0819190819192b08, 0x08191908192b0819, 0x08191908192b1908, 0x081919082b080808, 0x081919082b08082b, 0x081919082b081919,
    0x081919082b082b08, 0x081919082b190819, 0x081919082b191908, 0x081919082b2b0808, 0x0819191908080819, 0x0819191908081908, 0x081919190808192b, 0x0819191908082b19,
    0x0819191908190808, 0x081919190819082b, 0x0819191908191919, 0x0819191908192b08, 0x08191919082b0819, 0x08191919082b1908, 0x0819191919080808, 0x081919191908082b,
    0x0819191919081919, 0x0819191919082b08, 0x0819191919190819, 0x0819191919191908, 0x08191919192b0808, 0x081919192b080819, 0x081919192b081908, 0x081919192b190808,
    0x0819192b08080808, 0x0819192b08081919, 0x0819192b08082b08, 0x0819192b08190819, 0x0819192b08191908, 0x0819192b082b0808, 0x0819192b19080819, 0x0819192b19081908,
    0x0819192b19190808, 0x0819192b2b080808, 0x0819192b2b2b2b2b, 0x08192b0808080819, 0x08192b0808081908, 0x08192b080808192b, 0x08192b0808082b19, 0x08192b0808190808,
    0x08192b0808191919, 0x08192b0808192b08, 0x08192b08082b0819, 0x08192b0819080808, 0x08192b081908082b, 0x08192b0819081919, 0x08192b0819082b08, 0x08192b0819190819,
    0x08192b0819191908, 0x08192b08192b0808, 0x08192b082b080819, 0x08192b082b081908, 0x08192b1908080808, 0x08192b190808082b, 0x08192b1908081919, 0x08192b1908082b08,
    0x08192b1908190819, 0x08192b1908191908, 0x08192b19082b0808, 0x08192b1919080819, 0x08192b1919081908, 0x08192b1919190808, 0x08192b19192b2b19, 0x08192b192b2b082b,
    0x08192b2b08081908, 0x08192b2b08190808, 0x08192b2b19080808, 0x08192b2b1919192b, 0x082b080808080808, 0x082b08080808082b, 0x082b080808081919, 0x082b080808082b08,
    0x082b080808190819, 0x082b080808191908, 0x082b08080819192b, 0x082b080808192b19, 0x082b0808082b0808, 0x082b0808082b1919, 0x082b0808082b2b2b, 0x082b080819080819,
    0x082b080819081908, 0x082b080819190808, 0x082b08081919082b, 0x082b080819191919, 0x082b0808192b1908, 0x082b08082b080808, 0x082b08082b082b2b, 0x082b08082b191908,
    0x082b08082b2b2b2b, 0x082b081908080819, 0x082b081908081908, 0x082b081908190808, 0x082b08190819082b, 0x082b081908191919, 0x082b0819082b0819, 0x082b081919080808,
    0x082b08191908082b, 0x082b081919081919, 0x082b081919190819, 0x082b081919191908, 0x082b0819192b0808, 0x082b08192b080819, 0x082b08192b081908, 0x082b08192b190808,
    0x082b082b08080808, 0x082b082b08082b2b, 0x082b082b082b082b, 0x082b082b082b2b08, 0x082b082b082b2b2b, 0x082b082b19081908, 0x082b082b19190808, 0x082b082b2b082b08,
    0x082b082b2b082b2b, 0x082b082b2b2b2b08, 0x082b190808080819, 0x082b190808081908, 0x082b19080808192b, 0x082b190808082b19, 0x082b190808190808, 0x082b190808191919,
    0x082b190808192b08, 0x082b1908082b0819, 0x082b1908082b1908, 0x082b190819080808, 0x082b19081908082b, 0x082b190819081919, 0x082b190819082b08, 0x082b190819190819,
    0x082b190819191908, 0x082b1908192b0808, 0x082b19082b080819, 0x082b19082b081908, 0x082b19082b190808, 0x082b191908080808, 0x082b191908081919, 0x082b191908082b08,
    0x082b191908190819, 0x082b191908191908, 0x082b1919082b0808, 0x082b191919080819, 0x082b191919081908, 0x082b191919190808, 0x082b1919192b192b, 0x082b19192b080808,
    0x082b192b08080819, 0x082b192b08081908, 0x082b192b08190808, 0x082b192b19080808, 0x082b192b19192b19, 0x082b2b0808080808, 0x082b2b0808081919, 0x082b2b0808190819,
    0x082b2b0808191908, 0x082b2b0819080819, 0x082b2b0819081908, 0x082b2b0819190808, 0x082b2b082b082b2b, 0x082b2b082b2b2b2b, 0x082b2b1908080819, 0x082b2b1908081908,
    0x082b2b1908190808, 0x082b2b192b191919, 0x082b2b2b08082b2b, 0x082b2b2b082b082b, 0x082b2b2b192b1908, 0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819,
    0x1908080808081908, 0x190808080808192b, 0x1908080808082b19, 0x1908080808190808, 0x190808080819082b, 0x1908080808191919, 0x1908080808192b08, 0x1908080808192b2b,
    0x19080808082b0819, 0x19080808082b1908, 0x19080808082b192b, 0x1908080819080808, 0x190808081908082b, 0x1908080819081919, 0x1908080819082b08, 0x1908080819082b2b,
    0x1908080819190819, 0x1908080819191908, 0x190808081919192b, 0x1908080819192b19, 0x19080808192b0808, 0x19080808192b082b, 0x19080808192b1919, 0x190808082b080819,
    0x190808082b081908, 0x190808082b190808, 0x190808082b191919, 0x190808082b192b08, 0x190808082b2b0819, 0x190808082b2b1908, 0x1908081908080808, 0x190808190808082b,
    0x1908081908081919, 0x1908081908082b08, 0x1908081908190819, 0x1908081908191908, 0x190808190819192b, 0x1908081908192b19, 0x19080819082b0808, 0x19080819082b082b,
    0x19080819082b1919, 0x1908081919080819, 0x1908081919081908, 0x190808191908192b, 0x1908081919082b19, 0x1908081919190808, 0x190808191919082b, 0x1908081919191919,
    0x1908081919192b08, 0x19080819192b0819, 0x19080819192b1908, 0x190808192b080808, 0x190808192b08082b, 0x190808192b081919, 0x190808192b082b08, 0x190808192b190819,
    0x190808192b191908, 0x190808192b2b0808, 0x1908082b08080819, 0x1908082b08081908, 0x1908082b08190808, 0x1908082b0819082b, 0x1908082b08191919, 0x1908082b08192b08,
    0x1908082b082b1908, 0x1908082b19080808, 0x1908082b19081919, 0x1908082b19082b08, 0x1908082b19190819, 0x1908082b19191908, 0x1908082b192b0808, 0x1908082b2b080819,
    0x1908082b2b081908, 0x1908190808080808, 0x190819080808082b, 0x1908190808081919, 0x1908190808082b08, 0x1908190808082b2b, 0x1908190808190819, 0x1908190808191908,
    0x190819080819192b, 0x1908190808192b19, 0x19081908082b0808, 0x19081908082b082b, 0x19081908082b1919, 0x19081908082b2b08, 0x1908190819080819, 0x1908190819081908,
    0x190819081908192b, 0x1908190819082b19, 0x1908190819190808, 0x190819081919082b, 0x1908190819191919, 0x1908190819192b08, 0x19081908192b0819, 0x19081908192b1908,
    0x190819082b080808, 0x190819082b08082b, 0x190819082b081919, 0x190819082b082b08, 0x190819082b190819, 0x190819082b191908, 0x190819082b2b0808, 0x1908191908080819,
    0x1908191908081908, 0x190819190808192b, 0x1908191908082b19, 0x1908191908190808, 0x190819190819082b, 0x1908191908191919, 0x1908191908192b08, 0x19081919082b0819,
    0x19081919082b1908, 0x1908191919080808, 0x190819191908082b, 0x1908191919081919, 0x1908191919082b08, 0x1908191919190819, 0x1908191919191908, 0x19081919192b0808,
    0x19081919192b2b2b, 0x190819192b080819, 0x190819192b081908, 0x190819192b190808, 0x1908192b08080808, 0x1908192b0808082b, 0x1908192b08081919, 0x1908192b08082b08,
    0x1908192b08190819, 0x1908192b08191908, 0x1908192b082b0808, 0x1908192b19080819, 0x1908192b19081908, 0x1908192b19190808, 0x1908192b2b080808, 0x1908192b2b2b1919,
    0x19082b0808080819, 0x19082b0808081908, 0x19082b0808082b19, 0x19082b0808190808, 0x19082b080819082b, 0x19082b0808191919, 0x19082b0808192b08, 0x19082b08082b0819,
    0x19082b08082b1908, 0x19082b0819080808, 0x19082b081908082b, 0x19082b0819081919, 0x19082b0819082b08, 0x19082b0819190819, 0x19082b0819191908, 0x19082b08192b0808,
    0x19082b082b081908, 0x19082b082b190808, 0x19082b1908080808, 0x19082b190808082b, 0x19082b1908081919, 0x19082b1908082b08, 0x19082b1908190819, 0x19082b1908191908,
    0x19082b19082b0808, 0x19082b1919080819, 0x19082b1919081908, 0x19082b1919190808, 0x19082b192b080808, 0x19082b192b19192b, 0x19082b2b08080819, 0x19082b2b08081908,
    0x19082b2b08190808, 0x19082b2b19080808, 0x1919080808080808, 0x191908080808082b, 0x1919080808081919, 0x1919080808082b08, 0x1919080808190819, 0x1919080808191908,
    0x191908080819192b, 0x1919080808192b19, 0x19190808082b0808, 0x19190808082b082b, 0x19190808082b1919, 0x19190808082b2b08, 0x1919080819080819, 0x1919080819081908,
    0x191908081908192b, 0x1919080819082b19, 0x1919080819190808, 0x191908081919082b, 0x1919080819191919, 0x1919080819192b08, 0x19190808192b0819, 0x19190808192b1908,
    0x191908082b080808, 0x191908082b08082b, 0x191908082b081919, 0x191908082b082b08, 0x191908082b190819, 0x191908082b191908, 0x1919081908080819, 0x1919081908081908,
    0x191908190808192b, 0x1919081908082b19, 0x1919081908190808, 0x191908190819082b, 0x1919081908191919, 0x1919081908192b08, 0x19190819082b0819, 0x19190819082b1908,
    0x1919081919080808, 0x191908191908082b, 0x1919081919081919, 0x1919081919082b08, 0x1919081919190819, 0x1919081919191908, 0x19190819192b0808, 0x191908192b080819,
    0x191908192b081908, 0x191908192b190808, 0x1919082b08080808, 0x1919082b08081919, 0x1919082b08082b08, 0x1919082b08190819, 0x1919082b08191908, 0x1919082b082b0808,
    0x1919082b19080819, 0x1919082b19081908, 0x1919082b19190808, 0x1919082b192b2b19, 0x1919082b2b080808, 0x1919190808080819, 0x1919190808081908, 0x191919080808192b,
    0x1919190808082b19, 0x1919190808190808, 0x191919080819082b, 0x1919190808191919, 0x1919190808192b08, 0x19191908082b0819, 0x19191908082b1908, 0x1919190819080808,
    0x191919081908082b, 0x1919190819081919, 0x1919190819082b08, 0x1919190819190819, 0x1919190819191908, 0x19191908192b0808, 0x191919082b080819, 0x191919082b081908,
    0x191919082b190808, 0x1919191908080808, 0x191919190808082b, 0x1919191908081919, 0x1919191908082b08, 0x1919191908190819, 0x1919191908191908, 0x19191919082b0808,
    0x1919191919080819, 0x1919191919081908, 0x1919191919190808, 0x191919192b080808, 0x1919192b08080819, 0x1919192b08081908, 0x1919192b08190808, 0x1919192b082b192b,
    0x1919192b19080808, 0x19192b0808080808, 0x19192b080808082b, 0x19192b0808081919, 0x19192b0808082b08, 0x19192b0808190819, 0x19192b0808191908, 0x19192b08082b0808,
    0x19192b0819080819, 0x19192b0819081908, 0x19192b0819190808, 0x19192b0819192b2b, 0x19192b082b080808, 0x19192b1908080819, 0x19192b1908081908, 0x19192b1908190808,
    0x19192b1919080808, 0x19192b2b08080808, 0x19192b2b08192b19, 0x19192b2b2b081919, 0x19192b2b2b2b2b08, 0x192b080808080819, 0x192b080808081908, 0x192b08080808192b,
    0x192b080808190808, 0x192b08080819082b, 0x192b080808191919, 0x192b080808192b08, 0x192b0808082b0819, 0x192b0808082b1908, 0x192b080819080808, 0x192b080819081919,
    0x192b080819082b08, 0x192b080819190819, 0x192b080819191908, 0x192b0808192b0808, 0x192b08082b081908, 0x192b08082b190808, 0x192b081908080808, 0x192b08190808082b,
    0x192b081908081919, 0x192b081908082b08, 0x192b081908190819, 0x192b081908191908, 0x192b0819082b0808, 0x192b081919080819, 0x192b081919081908, 0x192b081919190808,
    0x192b08192b080808, 0x192b08192b192b19, 0x192b082b08081908, 0x192b082b08190808, 0x192b082b19080808, 0x192b082b1919192b, 0x192b082b2b2b0819, 0x192b190808080808,
    0x192b190808081919, 0x192b190808082b08, 0x192b190808190819, 0x192b190808191908, 0x192b1908082b0808, 0x192b190819080819, 0x192b190819081908, 0x192b190819190808,
    0x192b19082b080808, 0x192b191908080819, 0x192b191908081908, 0x192b191908190808, 0x192b191919080808, 0x192b191919082b2b, 0x192b1919192b2b08, 0x192b19192b19082b,
    0x192b192b08080808, 0x192b192b2b191908, 0x192b2b0808080819, 0x192b2b0808081908, 0x192b2b0808190808, 0x192b2b08192b1919, 0x192b2b082b192b08, 0x192b2b1908080808,
    0x192b2b19082b2b2b, 0x192b2b2b1908082b, 0x192b2b2b2b2b0819, 0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08, 0x2b08080808190819,
    0x2b08080808191908, 0x2b08080808192b19, 0x2b080808082b0808, 0x2b080808082b1919, 0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808081919082b,
    0x2b08080819191919, 0x2b08080819192b08, 0x2b080808192b0819, 0x2b0808082b080808, 0x2b0808082b081919, 0x2b0808082b190819, 0x2b0808082b191908, 0x2b08081908080819,
    0x2b08081908081908, 0x2b08081908082b19, 0x2b08081908190808, 0x2b0808190819082b, 0x2b08081908191919, 0x2b08081908192b08, 0x2b080819082b0819, 0x2b080819082b1908,
    0x2b08081919080808, 0x2b0808191908082b, 0x2b08081919081919, 0x2b08081919082b08, 0x2b08081919190819, 0x2b08081919191908, 0x2b0808192b080819, 0x2b0808192b081908,
    0x2b0808192b190808, 0x2b0808192b2b2b19, 0x2b08082b08080808, 0x2b08082b08081919, 0x2b08082b08082b2b, 0x2b08082b08190819, 0x2b08082b08191908, 0x2b08082b19080819,
    0x2b08082b19081908, 0x2b08082b19190808, 0x2b08190808080819, 0x2b08190808081908, 0x2b0819080808192b, 0x2b08190808082b19, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190808192b08, 0x2b081908082b0819, 0x2b08190819080808, 0x2b0819081908082b, 0x2b08190819081919, 0x2b08190819082b08, 0x2b08190819190819,
    0x2b08190819191908, 0x2b081908192b0808, 0x2b0819082b080819, 0x2b0819082b081908, 0x2b0819082b190808, 0x2b08191908080808, 0x2b0819190808082b, 0x2b08191908081919,
    0x2b08191908082b08, 0x2b08191908190819, 0x2b08191908191908, 0x2b081919082b0808, 0x2b08191919080819, 0x2b08191919081908, 0x2b08191919190808, 0x2b0819192b080808,
    0x2b0819192b082b2b, 0x2b08192b08080819, 0x2b08192b08081908, 0x2b08192b08190808, 0x2b08192b082b2b19, 0x2b08192b19080808, 0x2b082b0808080808, 0x2b082b0808081919,
    0x2b082b0808190819, 0x2b082b0808191908, 0x2b082b0819080819, 0x2b082b0819081908, 0x2b082b0819190808, 0x2b082b082b2b082b, 0x2b082b1908080819, 0x2b082b1908081908,
    0x2b082b1919080808, 0x2b082b19192b1919, 0x2b082b2b082b082b, 0x2b082b2b19192b08, 0x2b082b2b19192b2b, 0x2b082b2b2b08082b, 0x2b082b2b2b2b082b, 0x2b19080808080819,
    0x2b19080808081908, 0x2b19080808082b19, 0x2b19080808190808, 0x2b1908080819082b, 0x2b19080808191919, 0x2b19080808192b08, 0x2b190808082b1908, 0x2b19080819080808,
    0x2b1908081908082b, 0x2b19080819081919, 0x2b19080819082b08, 0x2b19080819190819, 0x2b19080819191908, 0x2b190808192b0808, 0x2b1908082b080819, 0x2b1908082b081908,
    0x2b1908082b190808, 0x2b19081908080808, 0x2b19081908081919, 0x2b19081908190819, 0x2b19081908191908, 0x2b19081919080819, 0x2b19081919081908, 0x2b19081919190808,
    0x2b19081919192b2b, 0x2b19082b08080819, 0x2b19082b08081908, 0x2b19082b08190808, 0x2b19082b19080808, 0x2b19082b2b2b192b, 0x2b19190808080808, 0x2b1919080808082b,
    0x2b19190808081919, 0x2b19190808082b08, 0x2b19190808190819, 0x2b19190808191908, 0x2b191908082b0808, 0x2b19190819080819, 0x2b19190819081908, 0x2b19190819190808,
    0x2b1919082b080808, 0x2b1919082b19192b, 0x2b19191908080819, 0x2b19191908081908, 0x2b19191908190808, 0x2b19191919080808, 0x2b1919192b192b08, 0x2b1919192b2b0819,
    0x2b19192b08080808, 0x2b19192b1908192b, 0x2b19192b192b1908, 0x2b192b0808080819, 0x2b192b0808081908, 0x2b192b0808190808, 0x2b192b08082b192b, 0x2b192b0819080808,
    0x2b192b082b2b2b19, 0x2b192b1908080808, 0x2b192b1919082b19, 0x2b192b191919082b, 0x2b192b2b2b190808, 0x2b2b080808080808, 0x2b2b080808081919, 0x2b2b080808082b2b,
    0x2b2b080808191908, 0x2b2b0808082b082b, 0x2b2b0808082b2b2b, 0x2b2b080819080819, 0x2b2b080819081908, 0x2b2b080819190808, 0x2b2b08082b2b082b, 0x2b2b08082b2b2b2b,
    0x2b2b081919080808, 0x2b2b0819192b1919, 0x2b2b082b0808082b, 0x2b2b082b08082b2b, 0x2b2b082b082b082b, 0x2b2b082b082b2b08, 0x2b2b082b082b2b2b, 0x2b2b082b2b08082b,
    0x2b2b082b2b082b08, 0x2b2b082b2b082b2b, 0x2b2b082b2b2b2b08, 0x2b2b190808080819, 0x2b2b190808081908, 0x2b2b190808190808, 0x2b2b190819080808, 0x2b2b19082b082b19,
    0x2b2b19082b2b1908, 0x2b2b191908080808, 0x2b2b191908192b19, 0x2b2b192b19190819, 0x2b2b2b0808082b2b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b082b, 0x2b2b2b1919191908,
    0x2b2b2b192b08192b, 0x2b2b2b2b08082b08, 0x2b2b2b2b08082b2b, 0x2b2b2b2b082b0808, 0x2b2b2b2b082b082b, 0x2b2b2b2b082b2b08, 0x2b2b2b2b2b082b08, 0x2b2b2b2b2b2b2b2b,
};

const iq3xxs_grid = [_]u32{
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

const iq1s_grid = [_]u64{
    0xffffffffffffffff, 0xffffffffffffff01, 0xffffffffffff0000, 0xffffffffffff01ff, 0xffffffffffff0101, 0xffffffffff00ff00, 0xffffffffff000000, 0xffffffffff01ffff,
    0xffffffffff01ff01, 0xffffffffff0101ff, 0xffffffffff010101, 0xffffffff00ff0000, 0xffffffff0000ff00, 0xffffffff000000ff, 0xffffffff00000001, 0xffffffff00010000,
    0xffffffff01ffffff, 0xffffffff01ffff01, 0xffffffff01ff01ff, 0xffffffff01ff0101, 0xffffffff01000000, 0xffffffff0101ffff, 0xffffffff0101ff01, 0xffffffff010101ff,
    0xffffffff01010101, 0xffffff00ffff00ff, 0xffffff00ffff0000, 0xffffff00ff00ff00, 0xffffff00ff0000ff, 0xffffff00ff000001, 0xffffff00ff000100, 0xffffff00ff000101,
    0xffffff00ff010000, 0xffffff0000ffff00, 0xffffff0000ff0001, 0xffffff0000ff0100, 0xffffff000000ff01, 0xffffff0000000000, 0xffffff0000000101, 0xffffff000001ff00,
    0xffffff00000100ff, 0xffffff0000010001, 0xffffff00000101ff, 0xffffff0001ff0000, 0xffffff000100ff00, 0xffffff00010000ff, 0xffffff0001000001, 0xffffff0001010000,
    0xffffff01ffffffff, 0xffffff01ffffff01, 0xffffff01ffff01ff, 0xffffff01ffff0101, 0xffffff01ff000000, 0xffffff01ff01ffff, 0xffffff01ff01ff01, 0xffffff01ff0101ff,
    0xffffff01ff010101, 0xffffff0100ff0000, 0xffffff010000ff00, 0xffffff0100000100, 0xffffff01000100ff, 0xffffff0100010100, 0xffffff0101ffffff, 0xffffff0101ffff01,
    0xffffff0101ff01ff, 0xffffff0101ff0101, 0xffffff010100ff00, 0xffffff0101000000, 0xffffff0101000100, 0xffffff010101ffff, 0xffffff010101ff01, 0xffffff01010101ff,
    0xffffff0101010101, 0xffff00ffff00ff00, 0xffff00ffff0000ff, 0xffff00ffff000001, 0xffff00ffff010000, 0xffff00ff00ffff00, 0xffff00ff00ff0100, 0xffff00ff00000000,
    0xffff00ff00000101, 0xffff00ff000100ff, 0xffff00ff00010000, 0xffff00ff0100ff00, 0xffff00ff01000100, 0xffff00ff01010000, 0xffff0000ffffff00, 0xffff0000ffff00ff,
    0xffff0000ffff0000, 0xffff0000ffff0001, 0xffff0000ff000000, 0xffff0000ff0001ff, 0xffff0000ff000101, 0xffff0000ff010100, 0xffff000000ffffff, 0xffff000000ff0000,
    0xffff000000ff0101, 0xffff00000000ffff, 0xffff00000000ff00, 0xffff0000000000ff, 0xffff000000000000, 0xffff000000000001, 0xffff000000000100, 0xffff00000001ffff,
    0xffff00000001ff01, 0xffff000000010000, 0xffff0000000101ff, 0xffff000000010101, 0xffff000001ffff00, 0xffff00000100ff00, 0xffff000001000000, 0xffff0000010001ff,
    0xffff000001000101, 0xffff00000101ff00, 0xffff0000010100ff, 0xffff000001010000, 0xffff000001010001, 0xffff000001010100, 0xffff0001ff0000ff, 0xffff0001ff000100,
    0xffff000100ffff00, 0xffff000100ff00ff, 0xffff00010000ffff, 0xffff00010000ff01, 0xffff000100000000, 0xffff0001000001ff, 0xffff00010001ffff, 0xffff00010001ff00,
    0xffff000100010001, 0xffff000100010100, 0xffff000101ff0000, 0xffff00010100ff00, 0xffff0001010000ff, 0xffff000101000100, 0xffff01ffffffffff, 0xffff01ffffffff01,
    0xffff01ffffff01ff, 0xffff01ffffff0101, 0xffff01ffff000000, 0xffff01ffff01ffff, 0xffff01ffff01ff01, 0xffff01ffff0101ff, 0xffff01ffff010101, 0xffff01ff00ff0000,
    0xffff01ff0000ff00, 0xffff01ff00000001, 0xffff01ff00010000, 0xffff01ff01ffffff, 0xffff01ff01ffff01, 0xffff01ff01ff01ff, 0xffff01ff01ff0101, 0xffff01ff01000000,
    0xffff01ff0101ffff, 0xffff01ff0101ff01, 0xffff01ff010101ff, 0xffff01ff01010101, 0xffff0100ffff0000, 0xffff0100ff00ff00, 0xffff0100ff0000ff, 0xffff0100ff000100,
    0xffff0100ff0100ff, 0xffff0100ff010000, 0xffff010000ffff00, 0xffff01000000ffff, 0xffff01000000ff00, 0xffff010000000000, 0xffff01000001ff00, 0xffff0100000100ff,
    0xffff010000010100, 0xffff01000100ff00, 0xffff0100010000ff, 0xffff010001000001, 0xffff010001000100, 0xffff010001010000, 0xffff0101ffffffff, 0xffff0101ffffff01,
    0xffff0101ffff01ff, 0xffff0101ffff0101, 0xffff0101ff000000, 0xffff0101ff01ffff, 0xffff0101ff01ff01, 0xffff0101ff0101ff, 0xffff0101ff010101, 0xffff010100ff0000,
    0xffff01010000ff00, 0xffff010100000100, 0xffff01010001ff00, 0xffff010100010000, 0xffff010101ffffff, 0xffff010101ffff01, 0xffff010101ff0000, 0xffff010101ff01ff,
    0xffff010101ff0101, 0xffff010101000000, 0xffff01010101ffff, 0xffff01010101ff01, 0xffff0101010101ff, 0xffff010101010101, 0xff00ffffff00ffff, 0xff00ffffff00ff00,
    0xff00ffffff0000ff, 0xff00ffffff000100, 0xff00ffffff0100ff, 0xff00ffffff010000, 0xff00ffff00ffff00, 0xff00ffff00ff00ff, 0xff00ffff0000ffff, 0xff00ffff00000000,
    0xff00ffff000001ff, 0xff00ffff0001ff00, 0xff00ffff000100ff, 0xff00ffff00010000, 0xff00ffff00010100, 0xff00ffff0100ff00, 0xff00ffff010000ff, 0xff00ffff01000001,
    0xff00ffff0101ff00, 0xff00ffff01010000, 0xff00ff00ffffff00, 0xff00ff00ffff00ff, 0xff00ff00ffff0001, 0xff00ff00ffff0100, 0xff00ff00ff00ffff, 0xff00ff00ff00ff01,
    0xff00ff00ff000000, 0xff00ff00ff0001ff, 0xff00ff00ff01ff00, 0xff00ff00ff0100ff, 0xff00ff00ff010100, 0xff00ff0000ff0000, 0xff00ff0000ff0101, 0xff00ff000000ffff,
    0xff00ff000000ff00, 0xff00ff000000ff01, 0xff00ff00000000ff, 0xff00ff0000000000, 0xff00ff0000000001, 0xff00ff0000000100, 0xff00ff000001ffff, 0xff00ff0000010000,
    0xff00ff0001ff00ff, 0xff00ff000100ff01, 0xff00ff0001000000, 0xff00ff000101ff00, 0xff00ff00010100ff, 0xff00ff01ff00ff00, 0xff00ff01ff0000ff, 0xff00ff01ff000001,
    0xff00ff01ff010000, 0xff00ff0100ffffff, 0xff00ff0100ff0001, 0xff00ff0100ff0100, 0xff00ff010000ff01, 0xff00ff0100000000, 0xff00ff01000001ff, 0xff00ff0100000101,
    0xff00ff01000100ff, 0xff00ff0100010001, 0xff00ff0101ff0000, 0xff00ff010100ff00, 0xff00ff01010000ff, 0xff00ff0101000001, 0xff00ff0101010000, 0xff0000ffffffff00,
    0xff0000ffffff0001, 0xff0000ffffff0100, 0xff0000ffff0000ff, 0xff0000ffff000000, 0xff0000ffff0001ff, 0xff0000ffff000100, 0xff0000ffff01ff00, 0xff0000ffff010001,
    0xff0000ff00ffff00, 0xff0000ff00ff0000, 0xff0000ff00ff0001, 0xff0000ff00ff01ff, 0xff0000ff00ff0101, 0xff0000ff0000ff00, 0xff0000ff000000ff, 0xff0000ff00000000,
    0xff0000ff00000001, 0xff0000ff00000100, 0xff0000ff0001ff01, 0xff0000ff00010000, 0xff0000ff000101ff, 0xff0000ff01ff00ff, 0xff0000ff01ff0100, 0xff0000ff0100ffff,
    0xff0000ff010000ff, 0xff0000ff01000000, 0xff0000ff010001ff, 0xff0000ff01000100, 0xff0000ff01000101, 0xff0000ff0101ff00, 0xff0000ff010100ff, 0xff0000ff01010000,
    0xff0000ff01010100, 0xff000000ffffff01, 0xff000000ffff0000, 0xff000000ffff0101, 0xff000000ff00ff00, 0xff000000ff0000ff, 0xff000000ff000000, 0xff000000ff000001,
    0xff000000ff000100, 0xff000000ff01ffff, 0xff000000ff01ff01, 0xff000000ff010000, 0xff000000ff0101ff, 0xff000000ff010101, 0xff00000000ffff00, 0xff00000000ff00ff,
    0xff00000000ff0000, 0xff00000000ff0001, 0xff0000000000ff00, 0xff0000000000ff01, 0xff000000000000ff, 0xff00000000000000, 0xff00000000000001, 0xff00000000000100,
    0xff00000000000101, 0xff0000000001ff00, 0xff000000000100ff, 0xff00000000010000, 0xff00000000010001, 0xff00000000010100, 0xff00000001ffffff, 0xff00000001ffff01,
    0xff00000001ff00ff, 0xff00000001ff0000, 0xff00000001ff01ff, 0xff00000001ff0101, 0xff0000000100ffff, 0xff0000000100ff00, 0xff000000010000ff, 0xff00000001000000,
    0xff00000001000001, 0xff00000001000100, 0xff00000001000101, 0xff0000000101ffff, 0xff0000000101ff01, 0xff00000001010000, 0xff000001ffffff00, 0xff000001ffff00ff,
    0xff000001ffff0000, 0xff000001ffff0001, 0xff000001ff000000, 0xff000001ff000001, 0xff000001ff0001ff, 0xff000001ff000101, 0xff000001ff01ff00, 0xff000001ff010001,
    0xff00000100ffffff, 0xff00000100ffff01, 0xff00000100ff00ff, 0xff00000100ff0000, 0xff00000100ff01ff, 0xff00000100ff0101, 0xff0000010000ff00, 0xff00000100000000,
    0xff00000100000001, 0xff000001000001ff, 0xff00000100000100, 0xff0000010001ff00, 0xff000001000100ff, 0xff00000100010000, 0xff000001000101ff, 0xff00000100010100,
    0xff00000100010101, 0xff00000101ff0001, 0xff00000101ff0101, 0xff0000010100ff01, 0xff00000101000000, 0xff000001010100ff, 0xff00000101010100, 0xff0001ffff00ff00,
    0xff0001ffff000001, 0xff0001ffff010000, 0xff0001ff00ffff00, 0xff0001ff00ff00ff, 0xff0001ff00ff0001, 0xff0001ff00ff0100, 0xff0001ff0000ffff, 0xff0001ff00000000,
    0xff0001ff000001ff, 0xff0001ff00000101, 0xff0001ff0001ffff, 0xff0001ff0001ff00, 0xff0001ff000100ff, 0xff0001ff00010001, 0xff0001ff00010100, 0xff0001ff01ff0000,
    0xff0001ff0100ff00, 0xff0001ff010000ff, 0xff0001ff01010000, 0xff000100ff00ffff, 0xff000100ff00ff01, 0xff000100ff000000, 0xff000100ff000101, 0xff000100ff01ff00,
    0xff000100ff010000, 0xff00010000ffff01, 0xff00010000ff00ff, 0xff00010000ff0000, 0xff00010000ff01ff, 0xff0001000000ff00, 0xff000100000000ff, 0xff00010000000000,
    0xff00010000000001, 0xff00010000000100, 0xff00010000000101, 0xff0001000001ffff, 0xff00010000010000, 0xff00010000010101, 0xff00010001ff0100, 0xff0001000100ff00,
    0xff0001000100ff01, 0xff00010001000000, 0xff000100010001ff, 0xff0001000101ff00, 0xff00010001010001, 0xff00010001010100, 0xff000101ffff0100, 0xff000101ff000001,
    0xff000101ff0100ff, 0xff000101ff010001, 0xff00010100ff00ff, 0xff00010100ff0001, 0xff00010100ff0100, 0xff0001010000ffff, 0xff0001010000ff01, 0xff00010100000000,
    0xff000101000001ff, 0xff0001010001ff00, 0xff00010100010001, 0xff00010100010100, 0xff00010101ff0000, 0xff0001010100ff00, 0xff00010101000001, 0xff00010101000101,
    0xff01ffffffffffff, 0xff01ffffffffff01, 0xff01ffffffff01ff, 0xff01ffffffff0101, 0xff01ffffff000000, 0xff01ffffff01ffff, 0xff01ffffff01ff01, 0xff01ffffff010000,
    0xff01ffffff0101ff, 0xff01ffffff010101, 0xff01ffff00ff0000, 0xff01ffff0000ff00, 0xff01ffff00000100, 0xff01ffff0001ff00, 0xff01ffff00010000, 0xff01ffff01ffffff,
    0xff01ffff01ffff01, 0xff01ffff01ff01ff, 0xff01ffff01ff0101, 0xff01ffff01000000, 0xff01ffff0101ffff, 0xff01ffff0101ff01, 0xff01ffff01010000, 0xff01ffff010101ff,
    0xff01ffff01010101, 0xff01ff00ffff0000, 0xff01ff00ff00ff00, 0xff01ff00ff0000ff, 0xff01ff00ff000100, 0xff01ff00ff010000, 0xff01ff0000ffff01, 0xff01ff0000ff00ff,
    0xff01ff0000ff0100, 0xff01ff0000000000, 0xff01ff00000001ff, 0xff01ff0000000101, 0xff01ff000001ff00, 0xff01ff00000100ff, 0xff01ff0000010000, 0xff01ff0000010001,
    0xff01ff0001ff0000, 0xff01ff000100ffff, 0xff01ff0001000001, 0xff01ff0001000100, 0xff01ff0001010000, 0xff01ff01ffffff00, 0xff01ff01ffff01ff, 0xff01ff01ffff0101,
    0xff01ff01ff00ff00, 0xff01ff01ff000000, 0xff01ff01ff01ffff, 0xff01ff01ff01ff01, 0xff01ff01ff0101ff, 0xff01ff01ff010101, 0xff01ff0100ff0000, 0xff01ff010000ff00,
    0xff01ff0100000001, 0xff01ff0100000100, 0xff01ff0100010000, 0xff01ff0101ffff00, 0xff01ff0101ff01ff, 0xff01ff0101ff0101, 0xff01ff010100ff00, 0xff01ff0101000000,
    0xff01ff010101ffff, 0xff01ff010101ff01, 0xff01ff01010101ff, 0xff01ff0101010101, 0xff0100ffffff0000, 0xff0100ffff0000ff, 0xff0100ffff000001, 0xff0100ffff000100,
    0xff0100ffff010000, 0xff0100ff00ff00ff, 0xff0100ff00ff0000, 0xff0100ff00ff0001, 0xff0100ff00ff0100, 0xff0100ff0000ff01, 0xff0100ff00000000, 0xff0100ff000001ff,
    0xff0100ff00000101, 0xff0100ff00010001, 0xff0100ff01ff0000, 0xff0100ff0100ff00, 0xff0100ff010000ff, 0xff0100ff01000100, 0xff0100ff0101ff00, 0xff0100ff01010000,
    0xff010000ffff0100, 0xff010000ff000000, 0xff010000ff01ff00, 0xff010000ff010100, 0xff01000000ffffff, 0xff01000000ff0000, 0xff01000000ff01ff, 0xff0100000000ff00,
    0xff010000000000ff, 0xff01000000000000, 0xff01000000000100, 0xff0100000001ff01, 0xff01000000010000, 0xff010000000101ff, 0xff01000001ff0100, 0xff0100000100ffff,
    0xff010000010000ff, 0xff01000001000000, 0xff010000010001ff, 0xff01000001000101, 0xff0100000101ff00, 0xff010000010100ff, 0xff01000001010001, 0xff01000001010100,
    0xff010001ffff0000, 0xff010001ff00ffff, 0xff010001ff00ff01, 0xff010001ff000100, 0xff010001ff010000, 0xff01000100ffff00, 0xff01000100ff0100, 0xff01000100000000,
    0xff0100010001ffff, 0xff0100010001ff00, 0xff01000100010100, 0xff01000101ff00ff, 0xff01000101ff0001, 0xff0100010100ffff, 0xff01000101000101, 0xff0101ffffffffff,
    0xff0101ffffffff01, 0xff0101ffffff01ff, 0xff0101ffffff0101, 0xff0101ffff000000, 0xff0101ffff01ffff, 0xff0101ffff01ff01, 0xff0101ffff0101ff, 0xff0101ffff010101,
    0xff0101ff00ff0000, 0xff0101ff0000ff00, 0xff0101ff000000ff, 0xff0101ff00010000, 0xff0101ff01ffffff, 0xff0101ff01ffff01, 0xff0101ff01ff01ff, 0xff0101ff01ff0101,
    0xff0101ff0101ffff, 0xff0101ff0101ff01, 0xff0101ff010101ff, 0xff0101ff01010101, 0xff010100ffff0100, 0xff010100ff00ff00, 0xff010100ff0000ff, 0xff010100ff000100,
    0xff010100ff010000, 0xff01010000ff0001, 0xff01010000ff0100, 0xff0101000000ff01, 0xff01010000000000, 0xff0101000001ff00, 0xff010100000100ff, 0xff01010000010001,
    0xff01010000010100, 0xff01010001ff0000, 0xff0101000100ffff, 0xff01010001000001, 0xff01010001000100, 0xff010100010100ff, 0xff01010001010000, 0xff010101ffffffff,
    0xff010101ffffff01, 0xff010101ffff01ff, 0xff010101ffff0101, 0xff010101ff01ffff, 0xff010101ff01ff01, 0xff010101ff0101ff, 0xff010101ff010101, 0xff01010100ff0000,
    0xff0101010000ff00, 0xff01010100000001, 0xff01010100000100, 0xff01010100010000, 0xff01010101ffffff, 0xff01010101ffff01, 0xff01010101ff01ff, 0xff01010101ff0101,
    0xff01010101000000, 0xff0101010101ffff, 0xff0101010101ff01, 0xff010101010101ff, 0xff01010101010101, 0x00ffffffffff0000, 0x00ffffffff00ff00, 0x00ffffffff000001,
    0x00ffffffff010000, 0x00ffffff00ff0100, 0x00ffffff0000ff01, 0x00ffffff00000000, 0x00ffffff000001ff, 0x00ffffff00000101, 0x00ffffff0001ff00, 0x00ffffff000100ff,
    0x00ffffff00010001, 0x00ffffff010000ff, 0x00ffffff01000100, 0x00ffffff0101ff00, 0x00ffffff01010001, 0x00ffff00ffffffff, 0x00ffff00ffffff00, 0x00ffff00ffff00ff,
    0x00ffff00ffff0001, 0x00ffff00ffff0100, 0x00ffff00ff00ff01, 0x00ffff00ff000000, 0x00ffff00ff000001, 0x00ffff00ff0001ff, 0x00ffff00ff000101, 0x00ffff00ff01ff00,
    0x00ffff00ff010001, 0x00ffff00ff010100, 0x00ffff0000ff0000, 0x00ffff0000ff01ff, 0x00ffff0000ff0101, 0x00ffff000000ff00, 0x00ffff00000000ff, 0x00ffff0000000000,
    0x00ffff0000000001, 0x00ffff0000000100, 0x00ffff0000000101, 0x00ffff0000010000, 0x00ffff00000101ff, 0x00ffff0000010101, 0x00ffff0001ffff00, 0x00ffff0001ff00ff,
    0x00ffff0001ff0001, 0x00ffff000100ffff, 0x00ffff000100ff01, 0x00ffff0001000000, 0x00ffff000101ffff, 0x00ffff000101ff00, 0x00ffff000101ff01, 0x00ffff01ffff0000,
    0x00ffff01ff00ff00, 0x00ffff01ff0000ff, 0x00ffff01ff000001, 0x00ffff01ff010000, 0x00ffff0100ffff00, 0x00ffff010000ff01, 0x00ffff0100000000, 0x00ffff0100000101,
    0x00ffff01000100ff, 0x00ffff0100010100, 0x00ffff0101ff0100, 0x00ffff01010000ff, 0x00ffff0101010000, 0x00ff00ffffffff00, 0x00ff00ffff000000, 0x00ff00ffff000100,
    0x00ff00ffff010100, 0x00ff00ff00ff0000, 0x00ff00ff00ff01ff, 0x00ff00ff00ff0101, 0x00ff00ff0000ff00, 0x00ff00ff000000ff, 0x00ff00ff00000000, 0x00ff00ff00000001,
    0x00ff00ff0001ff00, 0x00ff00ff0001ff01, 0x00ff00ff00010000, 0x00ff00ff000101ff, 0x00ff00ff00010101, 0x00ff00ff01ffff00, 0x00ff00ff01ff0001, 0x00ff00ff01ff0100,
    0x00ff00ff0100ffff, 0x00ff00ff0100ff01, 0x00ff00ff01000000, 0x00ff00ff0101ffff, 0x00ff00ff0101ff00, 0x00ff00ff01010100, 0x00ff0000ffffff00, 0x00ff0000ffffff01,
    0x00ff0000ffff0000, 0x00ff0000ffff0101, 0x00ff0000ff00ff00, 0x00ff0000ff0000ff, 0x00ff0000ff000000, 0x00ff0000ff000001, 0x00ff0000ff000100, 0x00ff0000ff01ffff,
    0x00ff0000ff010000, 0x00ff0000ff010101, 0x00ff000000ffff00, 0x00ff000000ff00ff, 0x00ff000000ff0000, 0x00ff000000ff0001, 0x00ff000000ff0100, 0x00ff00000000ffff,
    0x00ff00000000ff00, 0x00ff0000000000ff, 0x00ff000000000000, 0x00ff000000000001, 0x00ff0000000001ff, 0x00ff000000000100, 0x00ff00000001ff00, 0x00ff0000000100ff,
    0x00ff000000010000, 0x00ff000000010001, 0x00ff000000010100, 0x00ff000001ffff01, 0x00ff000001ff00ff, 0x00ff000001ff0000, 0x00ff000001ff01ff, 0x00ff00000100ff00,
    0x00ff0000010000ff, 0x00ff000001000000, 0x00ff000001000001, 0x00ff000001000100, 0x00ff000001000101, 0x00ff000001010000, 0x00ff0000010101ff, 0x00ff000001010101,
    0x00ff0001ffffff00, 0x00ff0001ffff0000, 0x00ff0001ffff0100, 0x00ff0001ff0000ff, 0x00ff0001ff000000, 0x00ff0001ff0001ff, 0x00ff0001ff000101, 0x00ff0001ff01ff00,
    0x00ff0001ff0100ff, 0x00ff0001ff010100, 0x00ff000100ffffff, 0x00ff000100ffff01, 0x00ff000100ff0000, 0x00ff000100ff01ff, 0x00ff00010000ffff, 0x00ff00010000ff00,
    0x00ff00010000ff01, 0x00ff000100000000, 0x00ff000100000001, 0x00ff000100000100, 0x00ff00010001ff01, 0x00ff000100010000, 0x00ff0001000101ff, 0x00ff000101ffff00,
    0x00ff000101ff0000, 0x00ff000101ff0101, 0x00ff0001010000ff, 0x00ff000101000000, 0x00ff00010101ff00, 0x00ff0001010100ff, 0x00ff000101010001, 0x00ff01ffffff0000,
    0x00ff01ffff00ff00, 0x00ff01ffff000000, 0x00ff01ffff000101, 0x00ff01ffff010000, 0x00ff01ff00ffff01, 0x00ff01ff00ff0100, 0x00ff01ff0000ffff, 0x00ff01ff00000000,
    0x00ff01ff000001ff, 0x00ff01ff0001ff00, 0x00ff01ff000100ff, 0x00ff01ff00010001, 0x00ff01ff00010100, 0x00ff01ff01ff0000, 0x00ff01ff0100ff00, 0x00ff01ff010000ff,
    0x00ff01ff01000001, 0x00ff01ff01000100, 0x00ff01ff01010000, 0x00ff0100ffffff00, 0x00ff0100ffff0000, 0x00ff0100ffff0001, 0x00ff0100ffff0101, 0x00ff0100ff00ffff,
    0x00ff0100ff0000ff, 0x00ff0100ff000000, 0x00ff0100ff0001ff, 0x00ff0100ff01ff00, 0x00ff0100ff0100ff, 0x00ff0100ff010001, 0x00ff010000ffffff, 0x00ff010000ff0000,
    0x00ff010000ff0101, 0x00ff01000000ff00, 0x00ff01000000ff01, 0x00ff0100000000ff, 0x00ff010000000000, 0x00ff010000000001, 0x00ff010000000100, 0x00ff01000001ffff,
    0x00ff01000001ff01, 0x00ff010000010000, 0x00ff010000010001, 0x00ff010000010101, 0x00ff010001ff0001, 0x00ff010001ff0100, 0x00ff01000100ff01, 0x00ff010001000000,
    0x00ff010001000001, 0x00ff0100010001ff, 0x00ff01000101ff00, 0x00ff0100010100ff, 0x00ff010001010001, 0x00ff010001010100, 0x00ff0101ff000001, 0x00ff010100ff00ff,
    0x00ff010100ff0001, 0x00ff010100ff0100, 0x00ff010100000000, 0x00ff0101000001ff, 0x00ff010100000101, 0x00ff0101000100ff, 0x00ff010100010100, 0x00ff0101010000ff,
    0x00ff010101010000, 0x0000ffffffffff00, 0x0000ffffffff00ff, 0x0000ffffffff0000, 0x0000ffffffff0001, 0x0000ffffffff0100, 0x0000ffffff00ff01, 0x0000ffffff000000,
    0x0000ffffff000101, 0x0000ffffff01ff00, 0x0000ffffff0100ff, 0x0000ffffff010100, 0x0000ffff00ffffff, 0x0000ffff00ff0000, 0x0000ffff00ff01ff, 0x0000ffff0000ff00,
    0x0000ffff000000ff, 0x0000ffff00000000, 0x0000ffff00000001, 0x0000ffff00000100, 0x0000ffff00010000, 0x0000ffff000101ff, 0x0000ffff01ff0001, 0x0000ffff01ff0100,
    0x0000ffff01000000, 0x0000ffff010001ff, 0x0000ffff0101ffff, 0x0000ffff0101ff00, 0x0000ffff01010001, 0x0000ffff01010100, 0x0000ff00ffff0000, 0x0000ff00ffff01ff,
    0x0000ff00ffff0100, 0x0000ff00ffff0101, 0x0000ff00ff00ff00, 0x0000ff00ff0000ff, 0x0000ff00ff000000, 0x0000ff00ff000001, 0x0000ff00ff0001ff, 0x0000ff00ff000100,
    0x0000ff00ff01ffff, 0x0000ff00ff010000, 0x0000ff00ff010001, 0x0000ff00ff0101ff, 0x0000ff00ff010101, 0x0000ff0000ffff00, 0x0000ff0000ff00ff, 0x0000ff0000ff0000,
    0x0000ff0000ff0001, 0x0000ff0000ff0100, 0x0000ff000000ffff, 0x0000ff000000ff00, 0x0000ff000000ff01, 0x0000ff00000000ff, 0x0000ff0000000000, 0x0000ff0000000001,
    0x0000ff00000001ff, 0x0000ff0000000100, 0x0000ff0000000101, 0x0000ff000001ff00, 0x0000ff00000100ff, 0x0000ff0000010000, 0x0000ff0000010001, 0x0000ff0000010100,
    0x0000ff0001ffff01, 0x0000ff0001ff0000, 0x0000ff000100ff00, 0x0000ff00010000ff, 0x0000ff0001000000, 0x0000ff0001000001, 0x0000ff0001000100, 0x0000ff000101ffff,
    0x0000ff0001010000, 0x0000ff0001010101, 0x0000ff01ffffff00, 0x0000ff01ffff0001, 0x0000ff01ff00ff01, 0x0000ff01ff000000, 0x0000ff01ff000101, 0x0000ff01ff01ff00,
    0x0000ff01ff0100ff, 0x0000ff0100ffff01, 0x0000ff0100ff0000, 0x0000ff0100ff0101, 0x0000ff010000ff00, 0x0000ff01000000ff, 0x0000ff0100000000, 0x0000ff0100000001,
    0x0000ff0100000100, 0x0000ff010001ff01, 0x0000ff0100010000, 0x0000ff0101ff0000, 0x0000ff010100ffff, 0x0000ff010100ff01, 0x0000ff0101000000, 0x0000ff0101000100,
    0x0000ff0101000101, 0x0000ff01010100ff, 0x000000ffffff00ff, 0x000000ffffff0000, 0x000000ffff00ff00, 0x000000ffff0000ff, 0x000000ffff000000, 0x000000ffff000001,
    0x000000ffff0001ff, 0x000000ffff000100, 0x000000ffff01ff00, 0x000000ffff010000, 0x000000ffff0101ff, 0x000000ffff010101, 0x000000ff00ffff00, 0x000000ff00ff00ff,
    0x000000ff00ff0000, 0x000000ff00ff0001, 0x000000ff00ff0100, 0x000000ff00ff0101, 0x000000ff0000ffff, 0x000000ff0000ff00, 0x000000ff000000ff, 0x000000ff00000000,
    0x000000ff00000001, 0x000000ff000001ff, 0x000000ff00000100, 0x000000ff00000101, 0x000000ff0001ff00, 0x000000ff0001ff01, 0x000000ff000100ff, 0x000000ff00010000,
    0x000000ff00010001, 0x000000ff00010100, 0x000000ff01ffffff, 0x000000ff01ff01ff, 0x000000ff01ff0101, 0x000000ff0100ff00, 0x000000ff010000ff, 0x000000ff01000000,
    0x000000ff01000001, 0x000000ff01000100, 0x000000ff0101ff00, 0x000000ff010100ff, 0x000000ff01010000, 0x000000ff01010101, 0x00000000ffffff00, 0x00000000ffffff01,
    0x00000000ffff00ff, 0x00000000ffff0000, 0x00000000ffff0001, 0x00000000ffff0100, 0x00000000ff00ffff, 0x00000000ff00ff00, 0x00000000ff00ff01, 0x00000000ff0000ff,
    0x00000000ff000000, 0x00000000ff000001, 0x00000000ff000100, 0x00000000ff000101, 0x00000000ff01ff00, 0x00000000ff0100ff, 0x00000000ff010000, 0x00000000ff010001,
    0x00000000ff010100, 0x0000000000ffffff, 0x0000000000ffff00, 0x0000000000ffff01, 0x0000000000ff00ff, 0x0000000000ff0000, 0x0000000000ff0001, 0x0000000000ff01ff,
    0x0000000000ff0100, 0x000000000000ffff, 0x000000000000ff00, 0x000000000000ff01, 0x00000000000000ff, 0x0000000000000000, 0x0000000000000001, 0x00000000000001ff,
    0x0000000000000100, 0x0000000000000101, 0x000000000001ffff, 0x000000000001ff00, 0x00000000000100ff, 0x0000000000010000, 0x0000000000010001, 0x00000000000101ff,
    0x0000000000010100, 0x0000000000010101, 0x0000000001ffff00, 0x0000000001ff00ff, 0x0000000001ff0000, 0x0000000001ff0100, 0x0000000001ff0101, 0x000000000100ffff,
    0x000000000100ff00, 0x00000000010000ff, 0x0000000001000000, 0x0000000001000001, 0x00000000010001ff, 0x0000000001000100, 0x000000000101ff00, 0x00000000010100ff,
    0x0000000001010000, 0x0000000001010001, 0x0000000001010100, 0x00000001ffffffff, 0x00000001ffffff00, 0x00000001ffffff01, 0x00000001ffff00ff, 0x00000001ffff0001,
    0x00000001ffff01ff, 0x00000001ffff0100, 0x00000001ff00ff00, 0x00000001ff0000ff, 0x00000001ff000000, 0x00000001ff0001ff, 0x00000001ff000100, 0x00000001ff01ffff,
    0x00000001ff01ff00, 0x00000001ff01ff01, 0x00000001ff0100ff, 0x00000001ff010000, 0x00000001ff010001, 0x00000001ff0101ff, 0x00000001ff010100, 0x0000000100ffff00,
    0x0000000100ff0000, 0x0000000100ff0001, 0x0000000100ff01ff, 0x0000000100ff0100, 0x0000000100ff0101, 0x000000010000ffff, 0x000000010000ff00, 0x000000010000ff01,
    0x00000001000000ff, 0x0000000100000000, 0x0000000100000001, 0x00000001000001ff, 0x0000000100000100, 0x0000000100000101, 0x000000010001ff00, 0x00000001000100ff,
    0x0000000100010000, 0x0000000100010100, 0x0000000101ffff01, 0x0000000101ff0000, 0x0000000101ff0001, 0x0000000101ff01ff, 0x0000000101ff0100, 0x0000000101ff0101,
    0x000000010100ff00, 0x0000000101000000, 0x0000000101000101, 0x000000010101ff01, 0x0000000101010000, 0x0000000101010001, 0x00000001010101ff, 0x0000000101010100,
    0x000001ffffff00ff, 0x000001ffffff0000, 0x000001ffffff0001, 0x000001ffffff0100, 0x000001ffff00ffff, 0x000001ffff000000, 0x000001ffff0001ff, 0x000001ffff01ff00,
    0x000001ffff010101, 0x000001ff00ff0000, 0x000001ff00ff01ff, 0x000001ff00ff0101, 0x000001ff0000ff00, 0x000001ff000000ff, 0x000001ff00000000, 0x000001ff00000001,
    0x000001ff000001ff, 0x000001ff00000100, 0x000001ff0001ffff, 0x000001ff0001ff01, 0x000001ff000100ff, 0x000001ff00010000, 0x000001ff01ffff01, 0x000001ff01ff0100,
    0x000001ff0100ffff, 0x000001ff0100ff01, 0x000001ff01000000, 0x000001ff010001ff, 0x000001ff0101ff00, 0x000001ff01010100, 0x00000100ffffff00, 0x00000100ffffff01,
    0x00000100ffff0000, 0x00000100ffff0101, 0x00000100ff00ff00, 0x00000100ff0000ff, 0x00000100ff000000, 0x00000100ff000001, 0x00000100ff000100, 0x00000100ff010000,
    0x0000010000ffff00, 0x0000010000ff00ff, 0x0000010000ff0000, 0x0000010000ff0001, 0x0000010000ff0100, 0x000001000000ffff, 0x000001000000ff00, 0x000001000000ff01,
    0x00000100000000ff, 0x0000010000000000, 0x0000010000000001, 0x00000100000001ff, 0x0000010000000100, 0x0000010000000101, 0x000001000001ff00, 0x00000100000100ff,
    0x0000010000010000, 0x0000010000010001, 0x0000010000010100, 0x0000010001ffff00, 0x0000010001ff0000, 0x0000010001ff0100, 0x000001000100ff00, 0x00000100010000ff,
    0x0000010001000000, 0x0000010001000001, 0x00000100010001ff, 0x0000010001000100, 0x0000010001010000, 0x00000101ffff00ff, 0x00000101ffff01ff, 0x00000101ff000000,
    0x00000101ff000101, 0x00000101ff01ffff, 0x00000101ff010000, 0x00000101ff010001, 0x00000101ff010100, 0x0000010100ff0000, 0x0000010100ff01ff, 0x0000010100ff0100,
    0x000001010000ff00, 0x0000010100000000, 0x0000010100000001, 0x00000101000001ff, 0x0000010100000100, 0x000001010001ff01, 0x0000010100010000, 0x00000101000101ff,
    0x0000010100010101, 0x0000010101ffff00, 0x0000010101ff0101, 0x000001010100ff01, 0x0000010101000000, 0x0000010101000001, 0x00000101010001ff, 0x0000010101000101,
    0x000001010101ff00, 0x0001ffffffff0000, 0x0001ffffff0000ff, 0x0001ffffff000001, 0x0001ffffff000100, 0x0001ffffff010000, 0x0001ffff00ff00ff, 0x0001ffff0000ffff,
    0x0001ffff00000000, 0x0001ffff00000001, 0x0001ffff000001ff, 0x0001ffff00000101, 0x0001ffff0001ff00, 0x0001ffff000100ff, 0x0001ffff00010001, 0x0001ffff00010100,
    0x0001ffff01ffff00, 0x0001ffff01000001, 0x0001ffff01010000, 0x0001ff00ffffff00, 0x0001ff00ffff00ff, 0x0001ff00ffff0001, 0x0001ff00ffff0100, 0x0001ff00ff00ff01,
    0x0001ff00ff000000, 0x0001ff00ff01ff00, 0x0001ff00ff01ff01, 0x0001ff00ff010001, 0x0001ff00ff010100, 0x0001ff0000ff0000, 0x0001ff0000ff0100, 0x0001ff000000ff00,
    0x0001ff0000000000, 0x0001ff0000000001, 0x0001ff0000000100, 0x0001ff0000010000, 0x0001ff0000010001, 0x0001ff0000010101, 0x0001ff0001ff00ff, 0x0001ff0001ff0101,
    0x0001ff000100ff01, 0x0001ff0001000000, 0x0001ff000101ff00, 0x0001ff0001010001, 0x0001ff0001010100, 0x0001ff01ff00ff00, 0x0001ff01ff000001, 0x0001ff01ff000100,
    0x0001ff0100ffffff, 0x0001ff0100ffff00, 0x0001ff0100ff0001, 0x0001ff0100000000, 0x0001ff0100000001, 0x0001ff01000001ff, 0x0001ff010001ffff, 0x0001ff0101ff0000,
    0x0001ff010100ff00, 0x0001ff0101000001, 0x0001ff0101010000, 0x000100ffff00ff00, 0x000100ffff00ff01, 0x000100ffff000000, 0x000100ffff000001, 0x000100ffff000101,
    0x000100ffff01ff00, 0x000100ffff010001, 0x000100ffff010100, 0x000100ff00ffffff, 0x000100ff00ffff01, 0x000100ff00ff0000, 0x000100ff00ff01ff, 0x000100ff00ff0101,
    0x000100ff0000ff00, 0x000100ff000000ff, 0x000100ff00000000, 0x000100ff00000001, 0x000100ff00000100, 0x000100ff00000101, 0x000100ff0001ffff, 0x000100ff0001ff01,
    0x000100ff00010000, 0x000100ff01ff00ff, 0x000100ff01ff0000, 0x000100ff01ff0100, 0x000100ff0100ffff, 0x000100ff0100ff01, 0x000100ff010000ff, 0x000100ff01000000,
    0x000100ff01000001, 0x000100ff010001ff, 0x000100ff01000101, 0x000100ff0101ff00, 0x000100ff010100ff, 0x000100ff01010100, 0x00010000ffff0000, 0x00010000ffff01ff,
    0x00010000ffff0101, 0x00010000ff00ff00, 0x00010000ff000000, 0x00010000ff000001, 0x00010000ff000100, 0x0001000000ff00ff, 0x0001000000ff0000, 0x0001000000ff0001,
    0x0001000000ff0100, 0x000100000000ffff, 0x000100000000ff00, 0x00010000000000ff, 0x0001000000000000, 0x0001000000000001, 0x0001000000000100, 0x000100000001ff00,
    0x00010000000100ff, 0x0001000000010000, 0x0001000000010001, 0x0001000000010100, 0x0001000001ff0001, 0x0001000001ff0100, 0x0001000001ff0101, 0x000100000100ff00,
    0x0001000001000000, 0x0001000001000001, 0x0001000001000100, 0x0001000001000101, 0x000100000101ff01, 0x0001000001010000, 0x0001000001010001, 0x00010000010101ff,
    0x00010001ffffff01, 0x00010001ffff0100, 0x00010001ff000000, 0x00010001ff01ffff, 0x00010001ff010001, 0x00010001ff0101ff, 0x00010001ff010100, 0x0001000100ffffff,
    0x0001000100ff0000, 0x0001000100ff01ff, 0x0001000100ff0101, 0x000100010000ff00, 0x00010001000000ff, 0x0001000100000000, 0x0001000100000001, 0x00010001000001ff,
    0x0001000100000101, 0x000100010001ffff, 0x0001000100010000, 0x00010001000101ff, 0x0001000101ffffff, 0x0001000101ffff01, 0x0001000101ff0000, 0x0001000101ff0101,
    0x00010001010000ff, 0x0001000101000001, 0x00010001010001ff, 0x0001000101000100, 0x000100010101ffff, 0x00010001010100ff, 0x0001000101010001, 0x0001000101010101,
    0x000101ffff000001, 0x000101ffff000100, 0x000101ffff010000, 0x000101ff00ffff00, 0x000101ff0000ff01, 0x000101ff00000000, 0x000101ff00000101, 0x000101ff0001ff00,
    0x000101ff00010100, 0x000101ff01ff0000, 0x000101ff0100ff00, 0x000101ff010001ff, 0x000101ff01010001, 0x00010100ffffff00, 0x00010100ffff00ff, 0x00010100ff00ffff,
    0x00010100ff000000, 0x00010100ff01ff00, 0x00010100ff0100ff, 0x00010100ff010001, 0x00010100ff010100, 0x0001010000ffffff, 0x0001010000ffff00, 0x0001010000ff0000,
    0x0001010000ff0001, 0x0001010000ff01ff, 0x000101000000ff00, 0x00010100000000ff, 0x0001010000000000, 0x0001010000000001, 0x0001010000000100, 0x000101000001ffff,
    0x0001010000010000, 0x0001010000010101, 0x0001010001ffff01, 0x0001010001ff00ff, 0x0001010001ff0101, 0x0001010001000000, 0x000101000101ff00, 0x00010100010100ff,
    0x0001010001010000, 0x0001010001010100, 0x00010101ff00ff00, 0x00010101ff000001, 0x00010101ff0001ff, 0x0001010100ffff00, 0x0001010100ff00ff, 0x0001010100ff0100,
    0x000101010000ffff, 0x0001010100000000, 0x00010101000001ff, 0x0001010100000101, 0x00010101000100ff, 0x0001010100010000, 0x0001010100010100, 0x0001010101ff0001,
    0x00010101010000ff, 0x00010101010001ff, 0x0001010101000101, 0x0001010101010001, 0x01ffffffffffffff, 0x01ffffffffffff01, 0x01ffffffffff01ff, 0x01ffffffffff0101,
    0x01ffffffff01ffff, 0x01ffffffff01ff01, 0x01ffffffff0101ff, 0x01ffffffff010101, 0x01ffffff00ff0000, 0x01ffffff0000ffff, 0x01ffffff0000ff00, 0x01ffffff000000ff,
    0x01ffffff00000001, 0x01ffffff00000100, 0x01ffffff00010000, 0x01ffffff01ffffff, 0x01ffffff01ffff01, 0x01ffffff01ff01ff, 0x01ffffff01ff0101, 0x01ffffff01000000,
    0x01ffffff0101ffff, 0x01ffffff0101ff01, 0x01ffffff010101ff, 0x01ffffff01010101, 0x01ffff00ffff0000, 0x01ffff00ff00ff00, 0x01ffff00ff0000ff, 0x01ffff00ff000001,
    0x01ffff00ff000100, 0x01ffff00ff010000, 0x01ffff0000ffff00, 0x01ffff0000ff00ff, 0x01ffff0000ff0100, 0x01ffff000000ffff, 0x01ffff000000ff01, 0x01ffff0000000000,
    0x01ffff0000000001, 0x01ffff00000001ff, 0x01ffff0000000100, 0x01ffff00000100ff, 0x01ffff0000010001, 0x01ffff0000010100, 0x01ffff0001ff0000, 0x01ffff0001ff0100,
    0x01ffff00010000ff, 0x01ffff0001000001, 0x01ffff0001000100, 0x01ffff0001010000, 0x01ffff01ffffffff, 0x01ffff01ffffff01, 0x01ffff01ffff01ff, 0x01ffff01ffff0101,
    0x01ffff01ff000000, 0x01ffff01ff01ffff, 0x01ffff01ff01ff01, 0x01ffff01ff0101ff, 0x01ffff01ff010101, 0x01ffff010000ff00, 0x01ffff01000000ff, 0x01ffff0100000100,
    0x01ffff0100010000, 0x01ffff0101ffffff, 0x01ffff0101ffff01, 0x01ffff0101ff01ff, 0x01ffff0101ff0101, 0x01ffff0101000000, 0x01ffff010101ffff, 0x01ffff010101ff01,
    0x01ffff01010101ff, 0x01ffff0101010101, 0x01ff00ffff0000ff, 0x01ff00ffff000100, 0x01ff00ff00ffff00, 0x01ff00ff00ff00ff, 0x01ff00ff0000ff00, 0x01ff00ff00000000,
    0x01ff00ff00000101, 0x01ff00ff0001ff00, 0x01ff00ff000100ff, 0x01ff00ff00010100, 0x01ff00ff010000ff, 0x01ff00ff01000100, 0x01ff0000ffffff00, 0x01ff0000ffff0100,
    0x01ff0000ff00ff01, 0x01ff0000ff000000, 0x01ff0000ff000101, 0x01ff0000ff010001, 0x01ff0000ff010100, 0x01ff000000ffffff, 0x01ff000000ffff00, 0x01ff000000ff0000,
    0x01ff000000ff01ff, 0x01ff00000000ff00, 0x01ff0000000000ff, 0x01ff000000000000, 0x01ff000000000001, 0x01ff000000000100, 0x01ff000000000101, 0x01ff000000010000,
    0x01ff000000010001, 0x01ff0000000101ff, 0x01ff000000010101, 0x01ff000001ffff00, 0x01ff000001ff00ff, 0x01ff000001ff0001, 0x01ff000001ff0100, 0x01ff00000100ffff,
    0x01ff00000100ff01, 0x01ff000001000000, 0x01ff0000010001ff, 0x01ff000001010001, 0x01ff0001ff00ff00, 0x01ff0001ff000001, 0x01ff0001ff000100, 0x01ff0001ff010000,
    0x01ff000100ffff00, 0x01ff000100ff00ff, 0x01ff000100ff0100, 0x01ff000100ff0101, 0x01ff00010000ffff, 0x01ff000100000000, 0x01ff000100000100, 0x01ff000100000101,
    0x01ff00010001ff00, 0x01ff000100010001, 0x01ff000100010101, 0x01ff000101ff0000, 0x01ff00010100ff00, 0x01ff000101000101, 0x01ff0001010100ff, 0x01ff01ffffffffff,
    0x01ff01ffffffff01, 0x01ff01ffffff01ff, 0x01ff01ffffff0101, 0x01ff01ffff000000, 0x01ff01ffff01ffff, 0x01ff01ffff01ff01, 0x01ff01ffff0101ff, 0x01ff01ffff010101,
    0x01ff01ff00ffff00, 0x01ff01ff00ff0000, 0x01ff01ff0000ff00, 0x01ff01ff000000ff, 0x01ff01ff00000100, 0x01ff01ff00010000, 0x01ff01ff00010100, 0x01ff01ff01ffffff,
    0x01ff01ff01ffff01, 0x01ff01ff01ff01ff, 0x01ff01ff01ff0101, 0x01ff01ff01000000, 0x01ff01ff0101ffff, 0x01ff01ff0101ff01, 0x01ff01ff010101ff, 0x01ff01ff01010101,
    0x01ff0100ffff0000, 0x01ff0100ffff0001, 0x01ff0100ff00ff00, 0x01ff0100ff0000ff, 0x01ff0100ff000001, 0x01ff0100ff010000, 0x01ff010000ffff00, 0x01ff010000ff00ff,
    0x01ff010000ff0001, 0x01ff010000ff0100, 0x01ff01000000ffff, 0x01ff01000000ff01, 0x01ff010000000000, 0x01ff010000000101, 0x01ff01000001ff00, 0x01ff0100000100ff,
    0x01ff010001ff0000, 0x01ff010001000001, 0x01ff010001000100, 0x01ff010001010000, 0x01ff0101ffffffff, 0x01ff0101ffffff01, 0x01ff0101ffff01ff, 0x01ff0101ffff0101,
    0x01ff0101ff000000, 0x01ff0101ff01ffff, 0x01ff0101ff01ff01, 0x01ff0101ff0101ff, 0x01ff0101ff010101, 0x01ff010100ff0000, 0x01ff01010000ff00, 0x01ff0101000000ff,
    0x01ff010100000001, 0x01ff010101ffffff, 0x01ff010101ffff01, 0x01ff010101ff01ff, 0x01ff010101ff0101, 0x01ff010101000000, 0x01ff01010101ffff, 0x01ff01010101ff01,
    0x01ff0101010101ff, 0x01ff010101010101, 0x0100ffffffff0000, 0x0100ffffff00ff00, 0x0100ffffff000001, 0x0100ffffff0001ff, 0x0100ffffff000100, 0x0100ffffff010000,
    0x0100ffff00ffff00, 0x0100ffff00ff0001, 0x0100ffff00ff0100, 0x0100ffff00000000, 0x0100ffff000001ff, 0x0100ffff00000101, 0x0100ffff00010100, 0x0100ffff00010101,
    0x0100ffff01ff0000, 0x0100ffff0100ff00, 0x0100ffff010000ff, 0x0100ffff01000001, 0x0100ffff01000100, 0x0100ffff01010000, 0x0100ff00ffffff00, 0x0100ff00ffff00ff,
    0x0100ff00ffff0001, 0x0100ff00ffff0100, 0x0100ff00ff00ffff, 0x0100ff00ff000000, 0x0100ff00ff0001ff, 0x0100ff00ff000101, 0x0100ff00ff01ff00, 0x0100ff00ff0100ff,
    0x0100ff00ff010001, 0x0100ff00ff010100, 0x0100ff0000ffffff, 0x0100ff0000ff0000, 0x0100ff000000ffff, 0x0100ff000000ff00, 0x0100ff00000000ff, 0x0100ff0000000000,
    0x0100ff0000000001, 0x0100ff0000000100, 0x0100ff000001ff01, 0x0100ff0000010000, 0x0100ff0001ff00ff, 0x0100ff0001ff0001, 0x0100ff000100ff01, 0x0100ff0001000000,
    0x0100ff00010001ff, 0x0100ff000101ff00, 0x0100ff00010100ff, 0x0100ff0001010001, 0x0100ff0001010100, 0x0100ff01ffff0000, 0x0100ff01ff00ff00, 0x0100ff01ff0000ff,
    0x0100ff01ff000100, 0x0100ff01ff010000, 0x0100ff0100ff00ff, 0x0100ff0100ff0001, 0x0100ff0100ff0100, 0x0100ff010000ffff, 0x0100ff010000ff01, 0x0100ff0100000000,
    0x0100ff01000001ff, 0x0100ff0100010001, 0x0100ff0100010100, 0x0100ff0101ff0000, 0x0100ff01010000ff, 0x0100ff0101000001, 0x0100ff0101010100, 0x010000ffffffff00,
    0x010000ffffff00ff, 0x010000ffffff0001, 0x010000ffff00ffff, 0x010000ffff000000, 0x010000ffff0001ff, 0x010000ffff010001, 0x010000ff00ffffff, 0x010000ff00ff0101,
    0x010000ff0000ff00, 0x010000ff000000ff, 0x010000ff00000000, 0x010000ff00000001, 0x010000ff000001ff, 0x010000ff00000100, 0x010000ff0001ffff, 0x010000ff0001ff00,
    0x010000ff0001ff01, 0x010000ff00010000, 0x010000ff01ff00ff, 0x010000ff01ff0001, 0x010000ff0100ff01, 0x010000ff010000ff, 0x010000ff01000000, 0x010000ff010001ff,
    0x010000ff0101ff00, 0x010000ff01010100, 0x01000000ffffffff, 0x01000000ffff0000, 0x01000000ffff01ff, 0x01000000ffff0101, 0x01000000ff00ffff, 0x01000000ff00ff00,
    0x01000000ff0000ff, 0x01000000ff000000, 0x01000000ff000001, 0x01000000ff000100, 0x01000000ff01ff00, 0x01000000ff010000, 0x01000000ff010100, 0x01000000ff010101,
    0x0100000000ffff00, 0x0100000000ff00ff, 0x0100000000ff0000, 0x0100000000ff0001, 0x0100000000ff0100, 0x010000000000ffff, 0x010000000000ff00, 0x010000000000ff01,
    0x01000000000000ff, 0x0100000000000000, 0x0100000000000001, 0x01000000000001ff, 0x0100000000000100, 0x0100000000000101, 0x010000000001ff00, 0x01000000000100ff,
    0x0100000000010000, 0x0100000000010001, 0x0100000000010100, 0x0100000001ffff00, 0x0100000001ff0000, 0x0100000001ff01ff, 0x010000000100ff00, 0x010000000100ff01,
    0x01000000010000ff, 0x0100000001000000, 0x0100000001000001, 0x0100000001000100, 0x0100000001000101, 0x010000000101ffff, 0x010000000101ff01, 0x0100000001010000,
    0x01000000010101ff, 0x0100000001010101, 0x01000001ffffff00, 0x01000001ffff00ff, 0x01000001ff00ffff, 0x01000001ff000000, 0x01000001ff000100, 0x01000001ff01ffff,
    0x01000001ff010001, 0x01000001ff010100, 0x0100000100ff0000, 0x0100000100ff01ff, 0x0100000100ff0100, 0x010000010000ff00, 0x010000010000ff01, 0x0100000100000000,
    0x0100000100000001, 0x0100000100000100, 0x0100000100010000, 0x01000001000101ff, 0x0100000101ffff01, 0x0100000101ff00ff, 0x0100000101ff0100, 0x0100000101ff0101,
    0x010000010100ff01, 0x01000001010000ff, 0x0100000101000000, 0x01000001010100ff, 0x0100000101010001, 0x0100000101010100, 0x010001ffffff0000, 0x010001ffff000001,
    0x010001ffff000100, 0x010001ffff010000, 0x010001ff00ffff00, 0x010001ff00ff0001, 0x010001ff0000ffff, 0x010001ff0000ff01, 0x010001ff00000000, 0x010001ff00000001,
    0x010001ff00000101, 0x010001ff000100ff, 0x010001ff00010000, 0x010001ff01ff0000, 0x010001ff0100ff00, 0x010001ff01000001, 0x010001ff01000100, 0x010001ff01010000,
    0x01000100ffff00ff, 0x01000100ffff0001, 0x01000100ffff0100, 0x01000100ff00ffff, 0x01000100ff00ff01, 0x01000100ff000000, 0x01000100ff0001ff, 0x01000100ff000101,
    0x01000100ff01ffff, 0x01000100ff01ff00, 0x01000100ff0100ff, 0x01000100ff010001, 0x0100010000ffffff, 0x0100010000ffff01, 0x0100010000ff0000, 0x0100010000ff01ff,
    0x0100010000ff0101, 0x010001000000ff00, 0x01000100000000ff, 0x0100010000000000, 0x0100010000000001, 0x0100010000000100, 0x010001000001ff01, 0x0100010000010000,
    0x0100010000010001, 0x0100010000010101, 0x0100010001ffff00, 0x0100010001ff00ff, 0x010001000100ffff, 0x010001000100ff01, 0x0100010001000000, 0x0100010001000101,
    0x010001000101ff00, 0x0100010001010001, 0x01000101ffff0000, 0x01000101ff000000, 0x01000101ff010000, 0x0100010100ff00ff, 0x0100010100ff0001, 0x0100010100ff0100,
    0x010001010000ffff, 0x0100010100000000, 0x01000101000001ff, 0x010001010001ff00, 0x0100010101ff0000, 0x010001010100ff00, 0x01000101010000ff, 0x0100010101000000,
    0x0100010101000001, 0x0101ffffffffffff, 0x0101ffffffffff01, 0x0101ffffffff01ff, 0x0101ffffffff0101, 0x0101ffffff000000, 0x0101ffffff01ffff, 0x0101ffffff01ff01,
    0x0101ffffff0101ff, 0x0101ffffff010101, 0x0101ffff00ff0000, 0x0101ffff0000ff00, 0x0101ffff000000ff, 0x0101ffff00000001, 0x0101ffff00000100, 0x0101ffff01ffffff,
    0x0101ffff01ffff01, 0x0101ffff01ff01ff, 0x0101ffff01ff0101, 0x0101ffff01000000, 0x0101ffff0101ffff, 0x0101ffff0101ff01, 0x0101ffff010101ff, 0x0101ffff01010101,
    0x0101ff00ffff0000, 0x0101ff00ffff0100, 0x0101ff00ff00ff00, 0x0101ff00ff0000ff, 0x0101ff00ff000001, 0x0101ff00ff000100, 0x0101ff00ff000101, 0x0101ff0000ff0001,
    0x0101ff0000ff0100, 0x0101ff000000ff00, 0x0101ff0000000000, 0x0101ff00000001ff, 0x0101ff0000000101, 0x0101ff000001ff00, 0x0101ff00000100ff, 0x0101ff0001ff0000,
    0x0101ff000100ffff, 0x0101ff000100ff01, 0x0101ff0001000001, 0x0101ff0001000100, 0x0101ff01ffffff01, 0x0101ff01ffff01ff, 0x0101ff01ffff0101, 0x0101ff01ff00ffff,
    0x0101ff01ff000100, 0x0101ff01ff01ff01, 0x0101ff01ff0101ff, 0x0101ff01ff010101, 0x0101ff0100ff0000, 0x0101ff010000ff00, 0x0101ff0100000001, 0x0101ff0100000100,
    0x0101ff0100010000, 0x0101ff0101ffffff, 0x0101ff0101ffff01, 0x0101ff0101ff01ff, 0x0101ff0101ff0101, 0x0101ff0101000000, 0x0101ff010101ffff, 0x0101ff010101ff01,
    0x0101ff01010101ff, 0x0101ff0101010101, 0x010100ffff000100, 0x010100ffff010000, 0x010100ff00ffff00, 0x010100ff00ff00ff, 0x010100ff0000ffff, 0x010100ff000000ff,
    0x010100ff00000000, 0x010100ff000001ff, 0x010100ff00000101, 0x010100ff0001ff00, 0x010100ff00010000, 0x010100ff00010001, 0x010100ff000101ff, 0x010100ff00010100,
    0x010100ff01ff0000, 0x01010000ffff0001, 0x01010000ffff0100, 0x01010000ff00ffff, 0x01010000ff00ff01, 0x01010000ff000000, 0x01010000ff0001ff, 0x01010000ff010001,
    0x01010000ff010100, 0x0101000000ffff01, 0x0101000000ff0000, 0x010100000000ff00, 0x01010000000000ff, 0x0101000000000000, 0x0101000000000001, 0x0101000000000100,
    0x0101000000010000, 0x0101000000010101, 0x0101000001ffff00, 0x0101000001ff00ff, 0x0101000001ff0000, 0x0101000001ff0001, 0x0101000001ff0100, 0x010100000100ff01,
    0x0101000001000000, 0x01010000010001ff, 0x01010001ffff0000, 0x01010001ff00ff00, 0x01010001ff000001, 0x01010001ff000101, 0x01010001ff01ff00, 0x01010001ff010000,
    0x0101000100ff00ff, 0x0101000100ff0001, 0x0101000100ff0101, 0x010100010000ff01, 0x0101000100000000, 0x0101000100000001, 0x01010001000001ff, 0x010100010001ffff,
    0x010100010001ff01, 0x0101000101ff0001, 0x010100010100ffff, 0x0101000101000000, 0x0101000101000001, 0x0101000101000100, 0x010100010101ff00, 0x01010001010100ff,
    0x0101000101010001, 0x010101ffffffffff, 0x010101ffffffff01, 0x010101ffffff01ff, 0x010101ffffff0101, 0x010101ffff01ffff, 0x010101ffff01ff01, 0x010101ffff0101ff,
    0x010101ffff010101, 0x010101ff0000ff00, 0x010101ff000000ff, 0x010101ff00000001, 0x010101ff00000100, 0x010101ff01ffffff, 0x010101ff01ffff01, 0x010101ff01ff01ff,
    0x010101ff01ff0101, 0x010101ff01000000, 0x010101ff0101ffff, 0x010101ff0101ff01, 0x010101ff010101ff, 0x010101ff01010101, 0x01010100ffff0000, 0x01010100ff0000ff,
    0x01010100ff000100, 0x01010100ff01ff00, 0x01010100ff010000, 0x0101010000ffff00, 0x010101000000ffff, 0x0101010000000000, 0x0101010000000101, 0x010101000001ff00,
    0x0101010000010001, 0x0101010000010100, 0x010101000100ffff, 0x0101010001000001, 0x01010101ffffffff, 0x01010101ffffff01, 0x01010101ffff01ff, 0x01010101ffff0101,
    0x01010101ff01ffff, 0x01010101ff01ff01, 0x01010101ff0101ff, 0x01010101ff010101, 0x010101010000ff00, 0x01010101000000ff, 0x0101010100000001, 0x0101010101ffffff,
    0x0101010101ffff01, 0x0101010101ff01ff, 0x0101010101ff0101, 0x0101010101000000, 0x010101010101ffff, 0x010101010101ff01, 0x01010101010101ff, 0x0101010101010101,
};

const kvalues_fp4 = [_]i8{
    0, 1, 2, 3, 4, 6, 8, 12,
    0, 1, 2, 3, 4, 6, 8, 12,
};


/// IQ4_XS: super-bloques de 256. Cada bloque (136 bytes):
///   d f16 (offset 0), scales_h u16 (offset 2), scales_l[4] (offset 4),
///   qs[128] (offset 8).
///   8 sub-bloques de 32: scale ls = (scales_l[ib/2] >> 4*(ib%2)) & 0xF
///                         | ((scales_h >> 2*ib) & 3) << 4
///   dl = d * (ls - 32); val = dl * kvalues_iq4nl[q]
///   (ref: ggml dequantize_row_iq4_xs)
pub fn dequantIq4_xs(bytes: []const u8, out: []f32) void {
    const qk = 256;
    const block_bytes = 136;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const scales_h = std.mem.readInt(u16, bytes[base + 2 ..][0..2], .little);
        const scales_l = bytes[base + 4 .. base + 8];
        const qs = bytes[base + 8 .. base + 136];
        for (0..qk / 32) |ib| {
            const ls: i32 = @as(i32, @intCast((scales_l[ib / 2] >> @intCast(4 * (ib % 2))) & 0xF)) |
                (@as(i32, @intCast((scales_h >> @intCast(2 * ib)) & 3)) << 4);
            const dl = d * @as(f32, @floatFromInt(ls - 32));
            const q = qs[ib * 16 ..];
            for (0..16) |j| {
                out[i + ib * 32 + j] = dl * @as(f32, @floatFromInt(kvalues_iq4nl[q[j] & 0xF]));
                out[i + ib * 32 + j + 16] = dl * @as(f32, @floatFromInt(kvalues_iq4nl[q[j] >> 4]));
            }
        }
        nb += 1;
    }
}

/// IQ3_S: super-bloques de 256. Cada bloque (110 bytes):
///   d f16 (offset 0), qs[64] (offset 2), qh[8] (offset 66),
///   signs[32] (offset 74), scales[4] (offset 106).
///   Índice de grid de 9 bits: qs[k] | ((qh[b] << (8-2*l)) & 256)
///   db = d * (1 + 2*sc); val = db * byte_j(iq3s_grid[idx]) * sign
///   (ref: ggml dequantize_row_iq3_s)
pub fn dequantIq3_s(bytes: []const u8, out: []f32) void {
    const qk = 256;
    const block_bytes = 110;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const qs = bytes[base + 2 .. base + 66];
        const qh = bytes[base + 66 .. base + 74];
        const signs = bytes[base + 74 .. base + 106];
        const scales = bytes[base + 106 .. base + 110];
        // it = ib32/2 (0..3); cada it consume qs[16], signs[8], qh[2]
        for (0..qk / 64) |it| {
            const sc = scales[it];
            const db1 = d * (1 + 2 * @as(f32, @floatFromInt(sc & 0xf)));
            const db2 = d * (1 + 2 * @as(f32, @floatFromInt(sc >> 4)));
            const q0 = qs[it * 16 ..][0..8];
            const q1 = qs[it * 16 + 8 ..][0..8];
            // Primera mitad de 32 elementos (db1), usa qh[2*it]
            for (0..4) |l| {
                const e1 = iq3s_grid[idx3s(q0[2 * l], qh[2 * it], l)];
                const e2 = iq3s_grid[idx3s2(q0[2 * l + 1], qh[2 * it], l)];
                const sm = signs[it * 8 + l];
                for (0..4) |j| {
                    const s1: f32 = if (sm & kmask_iq2xs[j] != 0) -1 else 1;
                    const s2: f32 = if (sm & kmask_iq2xs[j + 4] != 0) -1 else 1;
                    out[i + it * 64 + l * 8 + j] = db1 * gridByte(e1, j) * s1;
                    out[i + it * 64 + l * 8 + j + 4] = db1 * gridByte(e2, j) * s2;
                }
            }
            // Segunda mitad de 32 elementos (db2), usa qh[2*it+1]
            for (0..4) |l| {
                const e1 = iq3s_grid[idx3s(q1[2 * l], qh[2 * it + 1], l)];
                const e2 = iq3s_grid[idx3s2(q1[2 * l + 1], qh[2 * it + 1], l)];
                const sm = signs[it * 8 + 4 + l];
                for (0..4) |j| {
                    const s1: f32 = if (sm & kmask_iq2xs[j] != 0) -1 else 1;
                    const s2: f32 = if (sm & kmask_iq2xs[j + 4] != 0) -1 else 1;
                    out[i + it * 64 + 32 + l * 8 + j] = db2 * gridByte(e1, j) * s1;
                    out[i + it * 64 + 32 + l * 8 + j + 4] = db2 * gridByte(e2, j) * s2;
                }
            }
        }
        nb += 1;
    }
}

inline fn idx3s(q: u8, h: u8, l: usize) usize {
    return @as(usize, q) | ((@as(usize, h) << @as(u6, @intCast(8 - 2 * l))) & 256);
}

inline fn idx3s2(q: u8, h: u8, l: usize) usize {
    return @as(usize, q) | ((@as(usize, h) << @as(u6, @intCast(7 - 2 * l))) & 256);
}

inline fn gridByte(entry: u32, j: usize) f32 {
    return @as(f32, @floatFromInt(@as(u8, @intCast((entry >> @intCast(8 * j)) & 0xFF))));
}

/// IQ4_NL: bloques de 32. Cada bloque (18 bytes):
///   d f16 (offset 0), qs[16] (offset 2, 4 bits/elem).
///   val = d * kvalues_iq4nl[q]   (ref: ggml dequantize_row_iq4_nl)
pub fn dequantIq4_nl(bytes: []const u8, out: []f32) void {
    const qk = 32;
    const block_bytes = 18;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const qs = bytes[base + 2 .. base + 18];
        for (0..qk / 2) |j| {
            out[i + j] = d * @as(f32, @floatFromInt(kvalues_iq4nl[qs[j] & 0xF]));
            out[i + j + qk / 2] = d * @as(f32, @floatFromInt(kvalues_iq4nl[qs[j] >> 4]));
        }
        nb += 1;
    }
}

/// IQ2_XXS: super-bloques de 256. Cada bloque (66 bytes):
///   d f16 (offset 0), qs uint16[32] (offset 2). 8 sub-bloques de 32:
///   cada uno usa 8 bytes de qs (2 uint32): 4 índices de grid (bytes bajos
///   del primer u32) + scale de 4 bits y 28 bits de signos.
///   db = d * (0.5 + (aux32[1] >> 28)) * 0.25
///   val = db * byte_j(iq2xxs_grid[idx]) * sign   (ref: ggml dequantize_row_iq2_xxs)
pub fn dequantIq2_xxs(bytes: []const u8, out: []f32) void {
    const qk = 256;
    const block_bytes = 66;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const qs = bytes[base + 2 .. base + 66];
        for (0..qk / 32) |ib| {
            const aux32_0 = std.mem.readInt(u32, qs[ib * 8 ..][0..4], .little);
            const aux32_1 = std.mem.readInt(u32, qs[ib * 8 + 4 ..][0..4], .little);
            const db = d * (0.5 + @as(f32, @floatFromInt(aux32_1 >> 28))) * 0.25;
            for (0..4) |l| {
                const idx = @as(usize, @intCast((aux32_0 >> @intCast(8 * l)) & 0xFF));
                const signs = ksigns_iq2xs[@as(usize, @intCast((aux32_1 >> @intCast(7 * l)) & 127))];
                const g = iq2xxs_grid[idx];
                for (0..8) |j| {
                    const s: f32 = if (signs & kmask_iq2xs[j] != 0) -1 else 1;
                    const gv: f32 = @floatFromInt(@as(u8, @intCast((g >> @intCast(8 * j)) & 0xFF)));
                    out[i + ib * 32 + l * 8 + j] = db * gv * s;
                }
            }
        }
        nb += 1;
    }
}

/// IQ2_XS: super-bloques de 256. Cada bloque (74 bytes):
///   d f16 (offset 0), qs uint16[32] (offset 2), scales[8] (offset 66).
///   db[0] = d*(0.5+(scales[ib]&0xf))*0.25; db[1] = d*(0.5+(scales[ib]>>4))*0.25
///   idx = qs[4*ib+l] & 511 (9 bits); signs = ksigns_iq2xs[qs >> 9]
///   val = db[l/2] * byte_j(iq2xs_grid[idx]) * sign
///   (ref: ggml dequantize_row_iq2_xs)
pub fn dequantIq2_xs(bytes: []const u8, out: []f32) void {
    const qk = 256;
    const block_bytes = 74;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const qs = bytes[base + 2 .. base + 66];
        const scales = bytes[base + 66 .. base + 74];
        for (0..qk / 32) |ib| {
            const db0 = d * (0.5 + @as(f32, @floatFromInt(scales[ib] & 0xF))) * 0.25;
            const db1 = d * (0.5 + @as(f32, @floatFromInt(scales[ib] >> 4))) * 0.25;
            for (0..4) |l| {
                const v = std.mem.readInt(u16, qs[ib * 8 + l * 2 ..][0..2], .little);
                const idx: usize = v & 511;
                const signs = ksigns_iq2xs[v >> 9];
                const g = iq2xs_grid[idx];
                const db = if (l < 2) db0 else db1;
                for (0..8) |j| {
                    const s: f32 = if (signs & kmask_iq2xs[j] != 0) -1 else 1;
                    const gv: f32 = @floatFromInt(@as(u8, @intCast((g >> @intCast(8 * j)) & 0xFF)));
                    out[i + ib * 32 + l * 8 + j] = db * gv * s;
                }
            }
        }
        nb += 1;
    }
}

/// IQ3_XXS: super-bloques de 256. Cada bloque (98 bytes):
///   d f16 (offset 0), qs[96] (offset 2). scales+signs están al final de qs
///   (offset 66, 32 bytes). Cada sub-bloque de 32: aux32 (4 bytes) con
///   scale de 4 bits (bits 28-31) y 28 bits de signos.
///   db = d * (0.5 + (aux32 >> 28)) * 0.5
///   val = db * byte_j(iq3xxs_grid[qs[2l]]) * sign; y val = db * byte_j(iq3xxs_grid[qs[2l+1]]) * sign
///   (ref: ggml dequantize_row_iq3_xxs)
pub fn dequantIq3_xxs(bytes: []const u8, out: []f32) void {
    const qk = 256;
    const block_bytes = 98;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const qs = bytes[base + 2 .. base + 66];
        const ss = bytes[base + 66 .. base + 98];
        for (0..qk / 32) |ib| {
            const aux32 = std.mem.readInt(u32, ss[ib * 4 ..][0..4], .little);
            const db = d * (0.5 + @as(f32, @floatFromInt(aux32 >> 28))) * 0.5;
            for (0..4) |l| {
                const signs = ksigns_iq2xs[@as(usize, @intCast((aux32 >> @intCast(7 * l)) & 127))];
                const g1 = iq3xxs_grid[qs[ib * 8 + 2 * l]];
                const g2 = iq3xxs_grid[qs[ib * 8 + 2 * l + 1]];
                for (0..4) |j| {
                    const s1: f32 = if (signs & kmask_iq2xs[j] != 0) -1 else 1;
                    const s2: f32 = if (signs & kmask_iq2xs[j + 4] != 0) -1 else 1;
                    out[i + ib * 32 + l * 8 + j] = db * gridByte(g1, j) * s1;
                    out[i + ib * 32 + l * 8 + j + 4] = db * gridByte(g2, j) * s2;
                }
            }
        }
        nb += 1;
    }
}

/// IQ1_S: super-bloques de 256. Cada bloque (50 bytes):
///   d f16 (offset 0), qs[32] (offset 2), qh uint16[8] (offset 34).
///   dl = d * (2*((qh[ib] >> 12) & 7) + 1); delta = ±IQ1S_DELTA según bit 15.
///   idx = qs[l] | (((qh[ib] >> 3*l) & 7) << 8); val = dl * (byte_j(iq1s_grid[idx]) + delta)
///   (ref: ggml dequantize_row_iq1_s)
pub fn dequantIq1_s(bytes: []const u8, out: []f32) void {
    const qk = 256;
    const block_bytes = 50;
    const delta: f32 = 0.125;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const qs = bytes[base + 2 .. base + 34];
        for (0..qk / 32) |ib| {
            const qhb = std.mem.readInt(u16, bytes[base + 34 + ib * 2 ..][0..2], .little);
            const dl = d * (2 * @as(f32, @floatFromInt((qhb >> 12) & 7)) + 1);
            const dd: f32 = if (qhb & 0x8000 != 0) -delta else delta;
            for (0..4) |l| {
                const idx: usize = qs[ib * 4 + l] | (@as(usize, (qhb >> @intCast(3 * l)) & 7) << 8);
                const g = iq1s_grid[idx];
                for (0..8) |j| {
                    const gv: f32 = @floatFromInt(@as(i8, @intCast((g >> @intCast(8 * j)) & 0xFF)));
                    out[i + ib * 32 + l * 8 + j] = dl * (gv + dd);
                }
            }
        }
        nb += 1;
    }
}

/// IQ2_S: super-bloques de 256. Cada bloque (82 bytes):
///   d f16 (offset 0), qs[64] (offset 2), qh[8] (offset 66), scales[8] (offset 74).
///   signs = qs + 32 (offsets 34..66). idx = qs[l] | (qh[ib32] << (8-2*l) & 0x300)
///   db[0] = d*(0.5+(scales[ib]&0xf))*0.25; db[1] = d*(0.5+(scales[ib]>>4))*0.25
///   val = db[l/2] * byte_j(iq2s_grid[idx]) * sign   (ref: ggml dequantize_row_iq2_s)
pub fn dequantIq2_s(bytes: []const u8, out: []f32) void {
    const qk = 256;
    const block_bytes = 82;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const qs = bytes[base + 2 .. base + 34];
        const signs = bytes[base + 34 .. base + 66];
        const qh = bytes[base + 66 .. base + 74];
        const scales = bytes[base + 74 .. base + 82];
        for (0..qk / 32) |ib| {
            const db0 = d * (0.5 + @as(f32, @floatFromInt(scales[ib] & 0xF))) * 0.25;
            const db1 = d * (0.5 + @as(f32, @floatFromInt(scales[ib] >> 4))) * 0.25;
            for (0..4) |l| {
                const idx: usize = qs[ib * 4 + l] | ((@as(usize, qh[ib]) << @intCast(8 - 2 * l)) & 0x300);
                const g = iq2s_grid[idx];
                const db = if (l < 2) db0 else db1;
                for (0..8) |j| {
                    const s: f32 = if (signs[ib * 4 + l] & kmask_iq2xs[j] != 0) -1 else 1;
                    const gv: f32 = @floatFromInt(@as(u8, @intCast((g >> @intCast(8 * j)) & 0xFF)));
                    out[i + ib * 32 + l * 8 + j] = db * gv * s;
                }
            }
        }
        nb += 1;
    }
}

/// IQ1_M: super-bloques de 256. Cada bloque (56 bytes, sin escala fp16):
///   qs[32] (offset 0), qh[16] (offset 32), scales[8] (offset 48).
///   La escala global d es un f16 ensamblado desde bits dispersos de scales.
///   dl1 = d * (2*((sc[ib/2] >> (6*(ib%2)+0)) & 7) + 1); dl2 análogo con +3.
///   val = dl * (byte_j(iq1s_grid[idx]) + delta)   (ref: ggml dequantize_row_iq1_m)
pub fn dequantIq1_m(bytes: []const u8, out: []f32) void {
    const qk = 256;
    const block_bytes = 56;
    const delta: f32 = 0.125;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const sc = bytes[base .. base + 8];
        const scale_bits = (@as(u32, sc[0]) << 4) | (@as(u32, sc[1]) << 8) |
            (@as(u32, sc[2]) << 12) | (@as(u32, sc[3]) << 16);
        const d: f32 = @floatCast(@as(f16, @bitCast(@as(u16, @intCast(scale_bits)))));
        const qs = bytes[base + 0 .. base + 32];
        const qh = bytes[base + 32 .. base + 48];
        for (0..qk / 32) |ib| {
            const sc16 = std.mem.readInt(u16, sc[(@divFloor(ib, 2)) * 2 ..][0..2], .little);
            const dl1 = d * (2 * @as(f32, @floatFromInt((sc16 >> @intCast(6 * (@mod(ib, 2)) + 0)) & 0x7)) + 1);
            const dl2 = d * (2 * @as(f32, @floatFromInt((sc16 >> @intCast(6 * (@mod(ib, 2)) + 3)) & 0x7)) + 1);
            const q0 = qs[ib * 4 ..];
            const qh0 = qh[(@divFloor(ib, 2)) * 2 ..];
            var idx: [4]usize = undefined;
            var dd: [4]f32 = undefined;
            idx[0] = q0[0] | ((@as(usize, qh0[0]) << 8) & 0x700);
            idx[1] = q0[1] | ((@as(usize, qh0[0]) << 4) & 0x700);
            idx[2] = q0[2] | ((@as(usize, qh0[1]) << 8) & 0x700);
            idx[3] = q0[3] | ((@as(usize, qh0[1]) << 4) & 0x700);
            dd[0] = if (qh0[0] & 0x08 != 0) -delta else delta;
            dd[1] = if (qh0[0] & 0x80 != 0) -delta else delta;
            dd[2] = if (qh0[1] & 0x08 != 0) -delta else delta;
            dd[3] = if (qh0[1] & 0x80 != 0) -delta else delta;
            for (0..2) |l| {
                const g = iq1s_grid[idx[l]];
                for (0..8) |j| {
                    const gv: f32 = @floatFromInt(@as(i8, @intCast((g >> @intCast(8 * j)) & 0xFF)));
                    out[i + ib * 32 + l * 8 + j] = dl1 * (gv + dd[l]);
                }
            }
            for (2..4) |l| {
                const g = iq1s_grid[idx[l]];
                for (0..8) |j| {
                    const gv: f32 = @floatFromInt(@as(i8, @intCast((g >> @intCast(8 * j)) & 0xFF)));
                    out[i + ib * 32 + l * 8 + j] = dl2 * (gv + dd[l]);
                }
            }
        }
        nb += 1;
    }
}

/// TQ1_0: super-bloques de 256. Cada bloque (54 bytes):
///   qs[48] (offset 0), qh[4] (offset 48), d f16 (offset 52).
///   Base-3: 5 elementos/byte en qs (48 bytes -> 240), 4 elementos/byte en qh (16).
///   val = (((u16)(q * pow3[n]) * 3) >> 8 - 1) * d   (ref: ggml dequantize_row_tq1_0)
pub fn dequantTq1_0(bytes: []const u8, out: []f32) void {
    const qk = 256;
    const block_bytes = 54;
    const pow3 = [_]u16{ 1, 3, 9, 27, 81, 243 };
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base + 52 ..][0..2], .little))));
        const qs = bytes[base .. base + 48];
        const qh = bytes[base + 48 .. base + 52];
        var o: usize = 0;
        // qs: 32 bytes con 5 niveles, luego 16 bytes con 5 niveles
        const seg_lens = [_]usize{ 32, 16 };
        const seg_offs = [_]usize{ 0, 32 };
        for (0..2) |seg| {
            const len = seg_lens[seg];
            const off = seg_offs[seg];
            for (0..len) |j| {
                for (0..5) |n| {
                    const q: u8 = @truncate(@as(u16, qs[off + j]) * pow3[n]);
                    const xi: i32 = @intCast((@as(u16, q) * 3) >> 8);
                    out[i + o] = @as(f32, @floatFromInt(xi - 1)) * d;
                    o += 1;
                }
            }
        }
        // qh: 4 bytes con 4 niveles (128 elementos... 4 bytes * 4 = 16)
        for (0..4) |j| {
            for (0..4) |n| {
                const q: u8 = @truncate(@as(u16, qh[j]) * pow3[n]);
                const xi: i32 = @intCast((@as(u16, q) * 3) >> 8);
                out[i + o] = @as(f32, @floatFromInt(xi - 1)) * d;
                o += 1;
            }
        }
        nb += 1;
    }
}

/// TQ2_0: super-bloques de 256. Cada bloque (66 bytes):
///   qs[64] (offset 0, 2 bits/elem), d f16 (offset 64).
///   val = ((q - 1) * d); q = (qs[j] >> 2*l) & 3   (ref: ggml dequantize_row_tq2_0)
pub fn dequantTq2_0(bytes: []const u8, out: []f32) void {
    const qk = 256;
    const block_bytes = 66;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base + 64 ..][0..2], .little))));
        const qs = bytes[base .. base + 64];
        for (0..2) |seg| {
            for (0..4) |l| {
                for (0..32) |m| {
                    const q: i32 = @intCast((qs[seg * 32 + m] >> @as(u3, @intCast(2 * l))) & 3);
                    out[i + seg * 128 + l * 32 + m] = @as(f32, @floatFromInt(q - 1)) * d;
                }
            }
        }
        nb += 1;
    }
}

/// MXFP4: bloques de 32. Cada bloque (17 bytes):
///   e u8 E8M0 (offset 0), qs[16] (offset 1, 4 bits/elem).
///   d = e8m0_to_fp32_half(e); val = kvalues_mxfp4[q] * d
///   (ref: ggml dequantize_row_mxfp4)
pub fn dequantMxfp4(bytes: []const u8, out: []f32) void {
    const qk = 32;
    const block_bytes = 17;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const e = bytes[base];
        const d = e8m0Half(e);
        const qs = bytes[base + 1 .. base + 17];
        for (0..qk / 2) |j| {
            out[i + j] = @as(f32, @floatFromInt(kvalues_fp4[qs[j] & 0xF])) * d;
            out[i + j + qk / 2] = @as(f32, @floatFromInt(kvalues_fp4[qs[j] >> 4])) * d;
        }
        nb += 1;
    }
}

inline fn e8m0Half(x: u8) f32 {
    const bits: u32 = if (x < 2)
        (@as(u32, 0x00200000) << @as(u5, @intCast(x)))
    else
        (@as(u32, x - 1) << 23);
    return @as(f32, @bitCast(bits));
}

/// Q5_K: super-bloques de 256. Cada bloque (176 bytes):
///   d f16 (offset 0), dmin f16 (offset 2), scales[12] (offset 4),
///   qh[32] (offset 16), qs[128] (offset 48).
///   val = d*sc*(q4 + qh_bit*16) - min*m  (ref: ggml dequantize_row_q5_K)
pub fn dequantQ5_K(bytes: []const u8, out: []f32) void {
    const qk = 256;
    const block_bytes = 176;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const min: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base + 2 ..][0..2], .little))));
        const scales = bytes[base + 4 .. base + 16];
        const qh = bytes[base + 16 .. base + 48];
        const qs = bytes[base + 48 .. base + 176];
        var is: usize = 0;
        var bit1: u8 = 1;
        var bit2: u8 = 2;
        var j: usize = 0;
        while (j < qk) : (j += 64) {
            const s1 = getScaleMinK4(is + 0, scales);
            const d1 = d * @as(f32, @floatFromInt(s1.d));
            const m1 = min * @as(f32, @floatFromInt(s1.m));
            const s2 = getScaleMinK4(is + 1, scales);
            const d2 = d * @as(f32, @floatFromInt(s2.d));
            const m2 = min * @as(f32, @floatFromInt(s2.m));
            const ql = qs[(j / 64) * 32 ..];
            for (0..32) |l| {
                const v1: f32 = @floatFromInt((ql[l] & 0xF) + @as(u8, @intFromBool(qh[l] & bit1 != 0)) * 16);
                const v2: f32 = @floatFromInt((ql[l] >> 4) + @as(u8, @intFromBool(qh[l] & bit2 != 0)) * 16);
                out[i + j + l] = d1 * v1 - m1;
                out[i + j + 32 + l] = d2 * v2 - m2;
            }
            is += 2;
            bit1 <<= 2;
            bit2 <<= 2;
        }
        nb += 1;
    }
}

/// Decodifica escala (6 bits) y min (6 bits) de block_q4_K (ref: get_scale_min_k4)
fn getScaleMinK4(j: usize, q: []const u8) ScaleMin {
    if (j < 4) {
        return .{ .d = q[j] & 63, .m = q[j + 4] & 63 };
    }
    return .{
        .d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
        .m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
    };
}

/// Dequantiza UN bloque de un tipo cuantizado. `bytes` apunta a los
/// blockBytes() de ese bloque; `out` debe tener al menos blockSize() elementos
/// (los tipos de super-bloque de 256 escriben todo el bloque). `elems` limita
/// cuántos elementos se producen en los tipos float (para cuantizados se
/// ignora, el llamador sólo debe leer min(blockSize, restantes)).
pub fn dequantBlock(dtype: GgmlType, bytes: []const u8, out: []f32, elems: usize) void {
    const bs = dtype.blockSize();
    switch (dtype) {
        .f32 => @memcpy(out[0..elems], std.mem.bytesAsSlice(f32, bytes[0 .. elems * 4])),
        .f16 => for (0..elems) |i| {
            out[i] = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little))));
        },
        .bf16 => for (0..elems) |i| {
            const bits = @as(u32, @as(u32, std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little)) << 16);
            out[i] = @as(f32, @bitCast(bits));
        },
        .f64 => dequantF64(bytes, out[0..elems]),
        .i8 => dequantI8(bytes, out[0..elems]),
        .i16 => dequantI16(bytes, out[0..elems]),
        .i32 => dequantI32(bytes, out[0..elems]),
        .i64 => dequantI64(bytes, out[0..elems]),
        .q8_0 => dequantQ8_0(bytes, out[0..bs]),
        .q8_1 => dequantQ8_1(bytes, out[0..bs]),
        .q4_0 => dequantQ4_0(bytes, out[0..bs]),
        .q4_1 => dequantQ4_1(bytes, out[0..bs]),
        .q5_0 => dequantQ5_0(bytes, out[0..bs]),
        .q5_1 => dequantQ5_1(bytes, out[0..bs]),
        .q4_k => dequantQ4_K(bytes, out[0..bs]),
        .q5_k => dequantQ5_K(bytes, out[0..bs]),
        .q6_k => dequantQ6_K(bytes, out[0..bs]),
        .q2_k => dequantQ2_K(bytes, out[0..bs]),
        .q3_k => dequantQ3_K(bytes, out[0..bs]),
        .q8_k => dequantQ8_K(bytes, out[0..bs]),
        .iq4_xs => dequantIq4_xs(bytes, out[0..bs]),
        .iq3_s => dequantIq3_s(bytes, out[0..bs]),
        .iq4_nl => dequantIq4_nl(bytes, out[0..bs]),
        .iq2_xxs => dequantIq2_xxs(bytes, out[0..bs]),
        .iq2_xs => dequantIq2_xs(bytes, out[0..bs]),
        .iq3_xxs => dequantIq3_xxs(bytes, out[0..bs]),
        .iq1_s => dequantIq1_s(bytes, out[0..bs]),
        .iq2_s => dequantIq2_s(bytes, out[0..bs]),
        .iq1_m => dequantIq1_m(bytes, out[0..bs]),
        .tq1_0 => dequantTq1_0(bytes, out[0..bs]),
        .tq2_0 => dequantTq2_0(bytes, out[0..bs]),
        .mxfp4 => dequantMxfp4(bytes, out[0..bs]),
    }
}

/// Dequantizar un tensor GGUF completo a f32
///
/// Tipos soportados end-to-end (pesos cargables por el motor):
///   no cuantizados:  f32, f16, bf16, f64, i8, i16, i32, i64
///   GGML estándar:   q8_0, q8_1, q4_0, q4_1, q5_0, q5_1, q4_k, q5_k, q6_k
///   IQ (unsloth):    iq4_xs, iq3_s
/// Cualquier otro tipo (q2_k, q3_k, q8_k, iq4_nl, iq2_xxs, iq2_s, iq1_s,
/// iq3_xxs, iq1_m, mxfp4, tq1_0, tq2_0, ...) devuelve `UnsupportedDtype`.
pub fn dequantTensor(info: *const TensorInfo, bytes: []const u8, out: []f32) GgufError!void {
    if (out.len < info.numel()) return GgufError.InvalidData;
    switch (info.dtype) {
        .f32 => dequantF32(bytes, out),
        .f16 => dequantF16(bytes, out),
        .bf16 => dequantBF16(bytes, out),
        .f64 => dequantF64(bytes, out),
        .i8 => dequantI8(bytes, out),
        .i16 => dequantI16(bytes, out),
        .i32 => dequantI32(bytes, out),
        .i64 => dequantI64(bytes, out),
        .q8_0 => dequantQ8_0(bytes, out),
        .q8_1 => dequantQ8_1(bytes, out),
        .q4_0 => dequantQ4_0(bytes, out),
        .q4_1 => dequantQ4_1(bytes, out),
        .q5_0 => dequantQ5_0(bytes, out),
        .q5_1 => dequantQ5_1(bytes, out),
        .q4_k => dequantQ4_K(bytes, out),
        .q5_k => dequantQ5_K(bytes, out),
        .q6_k => dequantQ6_K(bytes, out),
        .q2_k => dequantQ2_K(bytes, out),
        .q3_k => dequantQ3_K(bytes, out),
        .q8_k => dequantQ8_K(bytes, out),
        .iq4_xs => dequantIq4_xs(bytes, out),
        .iq3_s => dequantIq3_s(bytes, out),
        else => return GgufError.UnsupportedDtype,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

fn writeString(buf: []u8, pos: *usize, s: []const u8) void {
    std.mem.writeInt(u64, buf[pos.*..][0..8], s.len, .little);
    pos.* += 8;
    @memcpy(buf[pos.*..][0..s.len], s);
    pos.* += s.len;
}

test "gguf parse header and metadata" {
    const allocator = std.testing.allocator;

    var buf = try allocator.alloc(u8, 4096);
    defer allocator.free(buf);
    var p: usize = 0;

    // Header
    std.mem.writeInt(u32, buf[p..][0..4], GGUF_MAGIC, .little); p += 4;
    std.mem.writeInt(u32, buf[p..][0..4], 3, .little); p += 4;
    std.mem.writeInt(u64, buf[p..][0..8], 2, .little); p += 8; // tensor_count
    std.mem.writeInt(u64, buf[p..][0..8], 2, .little); p += 8; // metadata_kv_count

    // Metadata 1: general.architecture = "llama" (string)
    writeString(buf, &p, "general.architecture");
    std.mem.writeInt(u32, buf[p..][0..4], @intFromEnum(MetaValueType.string), .little); p += 4;
    writeString(buf, &p, "llama");

    // Metadata 2: llama.embedding_length = 4096 (uint64)
    writeString(buf, &p, "llama.embedding_length");
    std.mem.writeInt(u32, buf[p..][0..4], @intFromEnum(MetaValueType.uint64), .little); p += 4;
    std.mem.writeInt(u64, buf[p..][0..8], 4096, .little); p += 8;

    // Tensor 1: token_embd.weight, [4096, 32000], F16
    writeString(buf, &p, "token_embd.weight");
    std.mem.writeInt(u32, buf[p..][0..4], 2, .little); p += 4;
    std.mem.writeInt(u64, buf[p..][0..8], 4096, .little); p += 8;
    std.mem.writeInt(u64, buf[p..][0..8], 32000, .little); p += 8;
    std.mem.writeInt(u32, buf[p..][0..4], @intFromEnum(GgmlType.f16), .little); p += 4;
    std.mem.writeInt(u64, buf[p..][0..8], 0, .little); p += 8;

    // Tensor 2: output.weight, [32000, 4096], Q8_0
    writeString(buf, &p, "output.weight");
    std.mem.writeInt(u32, buf[p..][0..4], 2, .little); p += 4;
    std.mem.writeInt(u64, buf[p..][0..8], 32000, .little); p += 8;
    std.mem.writeInt(u64, buf[p..][0..8], 4096, .little); p += 8;
    std.mem.writeInt(u32, buf[p..][0..4], @intFromEnum(GgmlType.q8_0), .little); p += 4;
    std.mem.writeInt(u64, buf[p..][0..8], 4096 * 32000 * 2, .little); p += 8;

    var gguf = try GgufFile.fromBytes(allocator, buf);
    defer gguf.deinit();

    try std.testing.expectEqual(@as(u32, 3), gguf.version);
    try std.testing.expectEqual(@as(u32, DEFAULT_ALIGNMENT), gguf.alignment);
    try std.testing.expectEqualStrings("llama", gguf.arch().?);

    const emb = gguf.getMeta("llama.embedding_length").?;
    try std.testing.expectEqual(@as(u64, 4096), emb.asU64().?);

    const embd = gguf.getTensor("token_embd.weight").?;
    try std.testing.expectEqual(@as(u32, 2), embd.n_dims);
    try std.testing.expectEqual(@as(u64, 4096), embd.dims[0]);
    try std.testing.expectEqual(@as(u64, 32000), embd.dims[1]);
    try std.testing.expectEqual(GgmlType.f16, embd.dtype);

    const out = gguf.getTensor("output.weight").?;
    try std.testing.expectEqual(GgmlType.q8_0, out.dtype);
}

test "gguf metadata array of strings" {
    const allocator = std.testing.allocator;

    var buf = try allocator.alloc(u8, 4096);
    defer allocator.free(buf);
    var p: usize = 0;

    std.mem.writeInt(u32, buf[p..][0..4], GGUF_MAGIC, .little); p += 4;
    std.mem.writeInt(u32, buf[p..][0..4], 3, .little); p += 4;
    std.mem.writeInt(u64, buf[p..][0..8], 0, .little); p += 8; // tensor_count
    std.mem.writeInt(u64, buf[p..][0..8], 1, .little); p += 8; // metadata_kv_count

    // tokenizer.ggml.tokens = ["<pad>", "hello", "world"]
    writeString(buf, &p, "tokenizer.ggml.tokens");
    std.mem.writeInt(u32, buf[p..][0..4], @intFromEnum(MetaValueType.array), .little); p += 4;
    std.mem.writeInt(u32, buf[p..][0..4], @intFromEnum(MetaValueType.string), .little); p += 4;
    std.mem.writeInt(u64, buf[p..][0..8], 3, .little); p += 8;
    writeString(buf, &p, "<pad>");
    writeString(buf, &p, "hello");
    writeString(buf, &p, "world");

    var gguf = try GgufFile.fromBytes(allocator, buf[0..p]);
    defer gguf.deinit();

    const tokens = gguf.getMeta("tokenizer.ggml.tokens").?;
    const arr = tokens.array;
    try std.testing.expectEqual(@as(usize, 3), arr.items.len);
    try std.testing.expectEqualStrings("hello", arr.items[1].string);
}

test "gguf dequant q8_0" {
    const allocator = std.testing.allocator;
    var bytes = try allocator.alloc(u8, 34);
    defer allocator.free(bytes);

    // d = 0.5 (f16), qs = [-8, 7, 1, -1, ...]
    std.mem.writeInt(u16, bytes[0..2], @bitCast(@as(f16, 0.5)), .little);
    bytes[2] = 0xF8; // -8
    bytes[3] = 7;
    bytes[4] = 1;
    bytes[5] = 0xFF; // -1

    var out: [32]f32 = undefined;
    dequantQ8_0(bytes, &out);

    try std.testing.expectApproxEqRel(@as(f32, -4.0), out[0], 1e-3);
    try std.testing.expectApproxEqRel(@as(f32, 3.5), out[1], 1e-3);
    try std.testing.expectApproxEqRel(@as(f32, 0.5), out[2], 1e-3);
    try std.testing.expectApproxEqRel(@as(f32, -0.5), out[3], 1e-3);
}

test "gguf dequant q5_0" {
    const allocator = std.testing.allocator;
    var bytes = try allocator.alloc(u8, 22);
    defer allocator.free(bytes);
    @memset(bytes, 0);

    // d = 2.0. qs[0]=0x12 -> low=2 (el 0), high=1 (el 16). qh=0 -> val = d*(n-16)
    std.mem.writeInt(u16, bytes[0..2], @bitCast(@as(f16, 2.0)), .little);
    bytes[6] = 0x12;
    bytes[7] = 0xF8; // low=8 (el 1), high=15 (el 17)

    var out: [32]f32 = undefined;
    dequantQ5_0(bytes, &out);
    try std.testing.expectApproxEqRel(@as(f32, 2.0 * (2 - 16)), out[0], 1e-3);
    try std.testing.expectApproxEqRel(@as(f32, 2.0 * (8 - 16)), out[1], 1e-3);
    try std.testing.expectApproxEqRel(@as(f32, 2.0 * (1 - 16)), out[16], 1e-3);
    try std.testing.expectApproxEqRel(@as(f32, 2.0 * (15 - 16)), out[17], 1e-3);

    // qh bit0 -> elemento 0 gana +16; qh bit16 -> elemento 16 gana +16
    std.mem.writeInt(u32, bytes[2..6], @bitCast(@as(u32, 0x00010001)), .little);
    dequantQ5_0(bytes, &out);
    try std.testing.expectApproxEqRel(@as(f32, 2.0 * (2 + 16 - 16)), out[0], 1e-3);
    try std.testing.expectApproxEqRel(@as(f32, 2.0 * (1 + 16 - 16)), out[16], 1e-3);
    try std.testing.expectApproxEqRel(@as(f32, 2.0 * (8 - 16)), out[1], 1e-3);
}

test "gguf dequant q5_1" {
    const allocator = std.testing.allocator;
    var bytes = try allocator.alloc(u8, 24);
    defer allocator.free(bytes);
    @memset(bytes, 0);

    // d = 2.0, m = 0.5. qs[0]=0x12 -> low=2 (el 0), high=1 (el 16)
    std.mem.writeInt(u16, bytes[0..2], @bitCast(@as(f16, 2.0)), .little);
    std.mem.writeInt(u16, bytes[2..4], @bitCast(@as(f16, 0.5)), .little);
    bytes[8] = 0x12;

    var out: [32]f32 = undefined;
    dequantQ5_1(bytes, &out);
    try std.testing.expectApproxEqRel(@as(f32, 2.0 * 2 + 0.5), out[0], 1e-3);
    try std.testing.expectApproxEqRel(@as(f32, 2.0 * 1 + 0.5), out[16], 1e-3);
    try std.testing.expectApproxEqRel(@as(f32, 0.5), out[1], 1e-3);

    // qh bit16 -> elemento 16 gana +16
    std.mem.writeInt(u32, bytes[4..8], @bitCast(@as(u32, 0x00010000)), .little);
    dequantQ5_1(bytes, &out);
    try std.testing.expectApproxEqRel(@as(f32, 2.0 * (1 + 16) + 0.5), out[16], 1e-3);
    try std.testing.expectApproxEqRel(@as(f32, 2.0 * 2 + 0.5), out[0], 1e-3);
}

test "gguf dequant q8_1" {
    const allocator = std.testing.allocator;
    var bytes = try allocator.alloc(u8, 36);
    defer allocator.free(bytes);
    @memset(bytes, 0);

    // d = 0.5, s = 1.0. qs[0] = -8, qs[1] = 7
    std.mem.writeInt(u16, bytes[0..2], @bitCast(@as(f16, 0.5)), .little);
    std.mem.writeInt(u16, bytes[2..4], @bitCast(@as(f16, 1.0)), .little);
    bytes[4] = 0xF8; // -8
    bytes[5] = 7;

    var out: [32]f32 = undefined;
    dequantQ8_1(bytes, &out);
    try std.testing.expectApproxEqRel(@as(f32, 0.5 * -8 + 1.0), out[0], 1e-3);
    try std.testing.expectApproxEqRel(@as(f32, 0.5 * 7 + 1.0), out[1], 1e-3);
    try std.testing.expectApproxEqRel(@as(f32, 1.0), out[2], 1e-3);
}

test "gguf dequant integer and f64 types" {
    const allocator = std.testing.allocator;

    var bytes8 = try allocator.alloc(u8, 8);
    defer allocator.free(bytes8);
    bytes8[0] = 0xF8; // i8 -8
    bytes8[1] = 0x7F; // 127
    var out: [2]f32 = undefined;
    dequantI8(bytes8, &out);
    try std.testing.expectApproxEqRel(@as(f32, -8.0), out[0], 1e-5);
    try std.testing.expectApproxEqRel(@as(f32, 127.0), out[1], 1e-5);

    var bytes16 = try allocator.alloc(u8, 8);
    defer allocator.free(bytes16);
    std.mem.writeInt(i16, bytes16[0..2], -1000, .little);
    std.mem.writeInt(i16, bytes16[2..4], 30000, .little);
    dequantI16(bytes16, &out);
    try std.testing.expectApproxEqRel(@as(f32, -1000.0), out[0], 1e-5);
    try std.testing.expectApproxEqRel(@as(f32, 30000.0), out[1], 1e-5);

    var bytes32 = try allocator.alloc(u8, 8);
    defer allocator.free(bytes32);
    std.mem.writeInt(i32, bytes32[0..4], -123456, .little);
    std.mem.writeInt(i32, bytes32[4..8], 2000000, .little);
    dequantI32(bytes32, &out);
    try std.testing.expectApproxEqRel(@as(f32, -123456.0), out[0], 1e-3);
    try std.testing.expectApproxEqRel(@as(f32, 2000000.0), out[1], 1e-3);

    var bytes64 = try allocator.alloc(u8, 16);
    defer allocator.free(bytes64);
    std.mem.writeInt(i64, bytes64[0..8], -1, .little);
    std.mem.writeInt(i64, bytes64[8..16], 2, .little);
    dequantI64(bytes64, &out);
    try std.testing.expectApproxEqRel(@as(f32, -1.0), out[0], 1e-5);
    try std.testing.expectApproxEqRel(@as(f32, 2.0), out[1], 1e-5);

    var bytesf = try allocator.alloc(u8, 16);
    defer allocator.free(bytesf);
    std.mem.writeInt(u64, bytesf[0..8], @bitCast(@as(f64, 3.5)), .little);
    std.mem.writeInt(u64, bytesf[8..16], @bitCast(@as(f64, -0.25)), .little);
    dequantF64(bytesf, &out);
    try std.testing.expectApproxEqRel(@as(f32, 3.5), out[0], 1e-5);
    try std.testing.expectApproxEqRel(@as(f32, -0.25), out[1], 1e-5);
}

 test "gguf dequant q4_0 split layout" {
     const allocator = std.testing.allocator;
     var bytes = try allocator.alloc(u8, 18);
     defer allocator.free(bytes);

     // d = 2.0 (f16). Layout "split": elemento j (0..15) = nibble bajo de
     // qs[j], elemento j+16 = nibble alto de qs[j]. val = d*(nibble-8).
     std.mem.writeInt(u16, bytes[0..2], @bitCast(@as(f16, 2.0)), .little);
     bytes[2] = 0x21; // low=1 (el 0), high=2 (el 16)
     bytes[3] = 0x43; // low=3 (el 1), high=4 (el 17)
     bytes[4] = 0xF8; // low=8 (el 2), high=15 (el 18)

     var out: [32]f32 = undefined;
     dequantQ4_0(bytes, &out);

     try std.testing.expectApproxEqRel(@as(f32, 2.0 * (1 - 8)), out[0], 1e-3);
     try std.testing.expectApproxEqRel(@as(f32, 2.0 * (3 - 8)), out[1], 1e-3);
     try std.testing.expectApproxEqRel(@as(f32, 2.0 * (8 - 8)), out[2], 1e-3);
     try std.testing.expectApproxEqRel(@as(f32, 2.0 * (2 - 8)), out[16], 1e-3);
     try std.testing.expectApproxEqRel(@as(f32, 2.0 * (4 - 8)), out[17], 1e-3);
     try std.testing.expectApproxEqRel(@as(f32, 2.0 * (15 - 8)), out[18], 1e-3);
     // interleaved (incorrecto) pondría el nibble alto de qs[0] en out[1]
     try std.testing.expect(out[1] != out[16]);
 }

 test "gguf dequant q4_k" {
    const allocator = std.testing.allocator;
    // d=1.0, dmin=0.0, escalas todas 1 (d-scale), nibbles todos 15
    var block = try allocator.alloc(u8, 144);
    defer allocator.free(block);
    @memset(block, 0);
    std.mem.writeInt(u16, block[0..2], @bitCast(@as(f16, 1.0)), .little);
    std.mem.writeInt(u16, block[2..4], @bitCast(@as(f16, 0.0)), .little);
    for (0..12) |i| block[4 + i] = 0x01;
    for (16..144) |i| block[i] = 0xFF; // cada nibble = 15

    var out: [256]f32 = undefined;
    dequantQ4_K(block, &out);
    for (out) |v| {
        try std.testing.expectApproxEqRel(@as(f32, 15.0), v, 1e-3);
    }

    // Segundo escenario: d=2, dmin=1, nibbles=0 -> val = -min*m
    // get_scale_min_k4(j<4): m = scales[j+4] & 63 = 1 -> val = -1.0
    // get_scale_min_k4(j>=4): m = (scales[j+4]>>4)|((scales[j]>>6)<<4) = 0 -> val = 0
    std.mem.writeInt(u16, block[0..2], @bitCast(@as(f16, 2.0)), .little);
    std.mem.writeInt(u16, block[2..4], @bitCast(@as(f16, 1.0)), .little);
    for (0..12) |i| block[4 + i] = 0x01;
    for (16..144) |i| block[i] = 0x00;
    dequantQ4_K(block, &out);
    for (out[0..128]) |v| {
        try std.testing.expectApproxEqRel(@as(f32, -1.0), v, 1e-3);
    }
    for (out[128..256]) |v| {
        try std.testing.expectApproxEqRel(@as(f32, 0.0), v, 1e-3);
    }
}

test "gguf dequant q6_k" {
    const allocator = std.testing.allocator;
    // d=0.5, scales todos 1, ql/qh=0 -> q=-32 -> val = 0.5*1*(-32) = -16
    var block = try allocator.alloc(u8, 210);
    defer allocator.free(block);
    @memset(block, 0);
    for (192..208) |i| block[i] = 1; // scales = 1
    std.mem.writeInt(u16, block[208..210], @bitCast(@as(f16, 0.5)), .little);

    var out: [256]f32 = undefined;
    dequantQ6_K(block, &out);
    for (out) |v| {
        try std.testing.expectApproxEqRel(@as(f32, -16.0), v, 1e-3);
    }

    // ql bajo = 0x0F (15) -> q1 = 15-32 = -17 -> val = -8.5
    for (0..64) |l| block[l] = 0x0F;
    dequantQ6_K(block, &out);
    try std.testing.expectApproxEqRel(@as(f32, -8.5), out[0], 1e-3);
    // ql[32..] bajo = 15 -> q2 = -17 en offset 32
    try std.testing.expectApproxEqRel(@as(f32, -8.5), out[32], 1e-3);
    // ql>>4 = 0 -> q3 = -32 en offset 64
    try std.testing.expectApproxEqRel(@as(f32, -16.0), out[64], 1e-3);
}

test "gguf dequant q2_k" {
    const allocator = std.testing.allocator;
    // d=1.0, dmin=1.0, escalas=0x01 (dl=1, ml=0), qs=0 -> val = 1*0 - 0 = 0
    var block = try allocator.alloc(u8, 84);
    defer allocator.free(block);
    @memset(block, 0);
    std.mem.writeInt(u16, block[80..82], @bitCast(@as(f16, 1.0)), .little);
    std.mem.writeInt(u16, block[82..84], @bitCast(@as(f16, 1.0)), .little);
    for (0..16) |i| block[i] = 0x01;

    var out: [256]f32 = undefined;
    dequantQ2_K(block, &out);
    for (out) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), v, 1e-3);
    }

    // escala 0x11 -> dl=1, ml=1 -> val = -1
    for (0..16) |i| block[i] = 0x11;
    dequantQ2_K(block, &out);
    for (out) |v| {
        try std.testing.expectApproxEqRel(@as(f32, -1.0), v, 1e-3);
    }

    // qs bits: q[0]=0x03 (2 bits bajos = 3), escala 0x01 -> val = 3 - 0 = 3
    for (0..16) |i| block[i] = 0x01;
    block[16] = 0x03;
    dequantQ2_K(block, &out);
    try std.testing.expectApproxEqRel(@as(f32, 3.0), out[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[16], 1e-3);
    // shift=2 del mismo byte -> q2 = 0 -> val = 0 en offset 32
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[32], 1e-3);
}

test "gguf dequant q3_k" {
    const allocator = std.testing.allocator;
    // d=1.0, escalas: aux0=0x01.. -> scal16[0]=1 -> dl = 1-32 = -31
    // qs=0, hmask=0 -> val = -31 * (0 - 4) = 124
    var block = try allocator.alloc(u8, 110);
    defer allocator.free(block);
    @memset(block, 0);
    std.mem.writeInt(u16, block[108..110], @bitCast(@as(f16, 1.0)), .little);
    for (0..12) |i| block[96 + i] = 0x01;

    var out: [256]f32 = undefined;
    dequantQ3_K(block, &out);
    // scal16[0] = (0x01 & 0x0f) | (((0x01>>0)&3)<<4) = 1 | 16 = 17 -> dl = 17-32 = -15
    // val = -15 * (0-4) = 60
    try std.testing.expectApproxEqRel(@as(f32, 60.0), out[0], 1e-3);
    try std.testing.expectApproxEqRel(@as(f32, 60.0), out[16], 1e-3);

    // hmask[0] bit0 = 1 -> val = -15 * (0-0) = 0
    block[0] = 0x01;
    dequantQ3_K(block, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 1e-3);
    try std.testing.expectApproxEqRel(@as(f32, 60.0), out[16], 1e-3);

    // qs bit bajo = 3, sin hmask -> val = -15 * (3-4) = 15
    block[0] = 0;
    block[32] = 0x03;
    dequantQ3_K(block, &out);
    try std.testing.expectApproxEqRel(@as(f32, 15.0), out[0], 1e-3);
}

test "gguf dequant q8_k" {
    const allocator = std.testing.allocator;
    // d=0.5 (f32), qs=[-8, 7, ...]
    var block = try allocator.alloc(u8, 292);
    defer allocator.free(block);
    @memset(block, 0);
    std.mem.writeInt(u32, block[0..4], @bitCast(@as(f32, 0.5)), .little);
    block[4] = 0xF8; // -8
    block[5] = 7;

    var out: [256]f32 = undefined;
    dequantQ8_K(block, &out);
    try std.testing.expectApproxEqRel(@as(f32, -4.0), out[0], 1e-3);
    try std.testing.expectApproxEqRel(@as(f32, 3.5), out[1], 1e-3);
    try std.testing.expectApproxEqRel(@as(f32, 0.0), out[2], 1e-3);
}

test "gguf dequant q5_k" {
    const allocator = std.testing.allocator;
    // d=1.0, dmin=0.0, escalas todas 1, ql=0x0F (15), qh=0 -> val = 15
    var block = try allocator.alloc(u8, 176);
    defer allocator.free(block);
    @memset(block, 0);
    std.mem.writeInt(u16, block[0..2], @bitCast(@as(f16, 1.0)), .little);
    std.mem.writeInt(u16, block[2..4], @bitCast(@as(f16, 0.0)), .little);
    for (0..12) |i| block[4 + i] = 0x01;
    for (48..176) |i| block[i] = 0xFF; // qs nibbles = 15

    var out: [256]f32 = undefined;
    dequantQ5_K(block, &out);
    for (out) |v| {
        try std.testing.expectApproxEqRel(@as(f32, 15.0), v, 1e-3);
    }

    // qh bit0 puesto en primer grupo (bits 1,4,16,64) -> +16
    // u1=1,u2=2 en el primer sub-bloque de 64.
    @memset(block[48..176], 0); // qs = 0
    block[16] = 0x01; // qh[0] bit0 -> q1 (offset 0..31) +16
    block[16] = 0x01 | 0x02; // qh[0] bits 0 y 1 -> q1 y q2 +16
    dequantQ5_K(block, &out);
    try std.testing.expectApproxEqRel(@as(f32, 16.0), out[0], 1e-3); // q1 +16
    try std.testing.expectApproxEqRel(@as(f32, 16.0), out[32], 1e-3); // q2 +16
    try std.testing.expectApproxEqRel(@as(f32, 0.0), out[64], 1e-3); // q3 +0

    // Segundo bloque de 64: bit1=4, bit2=8 -> bits en qh[0] (mismo byte 0)
    block[16] = 0x04 | 0x08;
    dequantQ5_K(block, &out);
    try std.testing.expectApproxEqRel(@as(f32, 16.0), out[64], 1e-3);
    try std.testing.expectApproxEqRel(@as(f32, 16.0), out[96], 1e-3);
    try std.testing.expectApproxEqRel(@as(f32, 0.0), out[0], 1e-3);
}

test "gguf dequant iq4_xs" {
    const allocator = std.testing.allocator;
    // d=1.0, scales_h=0, scales_l=0 -> ls=0 -> dl = -32
    // qs todos 0 -> kvalues_iq4nl[0] = -127 -> val = -32 * -127 = 4064
    var block = try allocator.alloc(u8, 136);
    defer allocator.free(block);
    @memset(block, 0);
    std.mem.writeInt(u16, block[0..2], @bitCast(@as(f16, 1.0)), .little);

    var out: [256]f32 = undefined;
    dequantIq4_xs(block, &out);
    for (out) |v| {
        try std.testing.expectApproxEqRel(@as(f32, 4064.0), v, 1e-3);
    }

    // ls=32 -> dl=0 -> todos 0
    // ls = (scales_l[0] & 0xF) | ((scales_h & 3) << 4); para ib=0
    block[4] = 0x0F; // scales_l[0] bajo = 15
    block[2] = 0x01; // scales_h bit0 = 1 -> 16
    block[3] = 0x00;
    // ls = 15 | (1<<4) = 31 -> dl = -1 -> val = 127
    dequantIq4_xs(block, &out);
    try std.testing.expectApproxEqRel(@as(f32, 127.0), out[0], 1e-3);
    try std.testing.expectApproxEqRel(@as(f32, 127.0), out[16], 1e-3);

    // sub-bloque ib=2 (offset 64): ls usa scales_l[1] y scales_h bits 4-5
    // scales_l[1] = 0x05, scales_h = 0x10 (bit4) -> ls = 5 | (1<<4) = 21 -> dl=-11
    @memset(block[0..136], 0);
    std.mem.writeInt(u16, block[0..2], @bitCast(@as(f16, 1.0)), .little);
    block[4] = 0x0F; // ib=0: ls = 15 -> dl=-17
    block[5] = 0x05; // scales_l[1] (ib=2: nibble bajo, ib=3: nibble alto)
    std.mem.writeInt(u16, block[2..4], @bitCast(@as(u16, 0x0010)), .little); // scales_h bit4
    dequantIq4_xs(block, &out);
    // qs=0 -> kvalues_iq4nl[0] = -127
    // ib=0: ls = (scales_l[0]&0xF) | ((scales_h>>0)&3)<<4 = 15|0 = 15 -> dl=-17
    try std.testing.expectApproxEqRel(@as(f32, 127.0 * 17.0), out[0], 1e-2);
    // ib=2: ls = (scales_l[1]&0xF) | ((scales_h>>4)&3)<<4 = 5 | (1<<4) = 21 -> dl=-11
    try std.testing.expectApproxEqRel(@as(f32, 127.0 * 11.0), out[64], 1e-2);
    // ib=3: scales_l[1]>>4 = 0, scales_h>>6 = 0 -> ls = 0 -> dl=-32
    try std.testing.expectApproxEqRel(@as(f32, 127.0 * 32.0), out[96], 1e-2);
}

// Verificado byte a byte contra la referencia C de ggml (dequantize_row_iq3_s)
// sobre 64 bloques aleatorios + los dos bloques deterministas de abajo.
test "gguf dequant iq3_s" {
    const allocator = std.testing.allocator;

    // Bloque A: d=1.0, scales[0]=0x01 (db1=3, db2=1), qs[0..15]={0..15},
    // qh=0, signs=0. Cubre los índices de grid 0..15 para ambos nibbles de
    // escala en la primera iteración (it=0); el resto queda en 1.0.
    const block_a = [_]u8{0} ** 110;
    var ba = block_a;
    std.mem.writeInt(u16, ba[0..2], @bitCast(@as(f16, 1.0)), .little);
    for (0..16) |i| ba[2 + i] = @intCast(i);
    ba[106] = 0x01;

    const expect_a = [_]f32{
        3, 3, 3, 3, 9, 3, 3, 3,
        15, 3, 3, 3, 33, 3, 3, 3,
        45, 3, 3, 3, 3, 9, 3, 3,
        9, 9, 3, 3, 15, 9, 3, 3,
        9, 3, 1, 1, 13, 3, 1, 1,
        1, 5, 1, 1, 3, 5, 1, 1,
        11, 5, 1, 1, 7, 7, 1, 1,
        1, 9, 1, 1, 5, 9, 1, 1,
    };

    // Bloque B: d=1.0, scales=0 (db=1), qs[0..15]={0..15}, qh[0]=0x01 y
    // qh[1]=0x02 (bit 9 del índice de grid), signs[0..7] variados. Cubre el
    // bit 256 del índice y todas las máscaras de signo.
    const block_b = [_]u8{0} ** 110;
    var bb = block_b;
    std.mem.writeInt(u16, bb[0..2], @bitCast(@as(f16, 1.0)), .little);
    for (0..16) |i| bb[2 + i] = @intCast(i);
    bb[66] = 0x01;
    bb[67] = 0x02;
    bb[74] = 0x01;
    bb[75] = 0x03;
    bb[76] = 0x0f;
    bb[77] = 0xff;
    bb[78] = 0x80;
    bb[79] = 0x55;
    bb[80] = 0xaa;
    bb[81] = 0x00;

    const expect_b = [_]f32{
        -7, 5, 9, 5, 3, 1, 1, 1,
        -5, -1, 1, 1, 11, 1, 1, 1,
        -15, -1, -1, -1, 1, 3, 1, 1,
        -3, -3, -1, -1, -5, -3, -1, -1,
        9, 3, 1, 1, 15, 7, 11, -5,
        -1, 5, -1, 1, -3, 5, -1, 1,
        11, -5, 1, -1, 7, -7, 1, -1,
        1, 9, 1, 1, 5, 9, 1, 1,
    };

    var out: [256]f32 = undefined;

    dequantIq3_s(&ba, &out);
    for (expect_a, 0..) |exp, i| {
        try std.testing.expectApproxEqRel(exp, out[i], 1e-5);
    }
    for (out[64..]) |v| {
        try std.testing.expectApproxEqRel(@as(f32, 1.0), v, 1e-5);
    }

    dequantIq3_s(&bb, &out);
    for (expect_b, 0..) |exp, i| {
        try std.testing.expectApproxEqRel(exp, out[i], 1e-5);
    }
    for (out[64..]) |v| {
        try std.testing.expectApproxEqRel(@as(f32, 1.0), v, 1e-5);
    }

    // 2 bloques contiguos (como un tensor real): verificar offsets de bloque
    var two = try allocator.alloc(u8, 220);
    defer allocator.free(two);
    @memcpy(two[0..110], ba[0..]);
    @memcpy(two[110..220], bb[0..]);
    var out2: [512]f32 = undefined;
    dequantIq3_s(two, &out2);
    for (expect_a, 0..) |exp, i| {
        try std.testing.expectApproxEqRel(exp, out2[i], 1e-5);
    }
    for (expect_b, 0..) |exp, i| {
        try std.testing.expectApproxEqRel(exp, out2[256 + i], 1e-5);
    }
}

test "gguf parse tensor names (C6)" {
    const t = std.testing;
    const expect = t.expectEqual;

    const embd = parseTensorName("token_embd.weight");
    try expect(TensorRole.token_embd, embd.role);
    try expect(@as(?usize, null), embd.layer);

    const out = parseTensorName("output.weight");
    try expect(TensorRole.output, out.role);

    const q = parseTensorName("blk.3.attn_q.weight");
    try expect(TensorRole.attn_q, q.role);
    try expect(@as(?usize, 3), q.layer);

    const v_bias = parseTensorName("blk.12.attn_v.bias");
    try expect(TensorRole.attn_v_bias, v_bias.role);
    try expect(@as(?usize, 12), v_bias.layer);

    const gate = parseTensorName("blk.0.ffn_gate.weight");
    try expect(TensorRole.ffn_gate, gate.role);
    try t.expect(gate.role.isFfnWeight());

    const w1 = parseTensorName("blk.5.feed_forward.w1.weight");
    try expect(TensorRole.ffn_gate, w1.role);

    const gate_proj = parseTensorName("blk.1.mlp.gate_proj.weight");
    try expect(TensorRole.ffn_gate, gate_proj.role);

    const attn_norm = parseTensorName("blk.0.attn_norm.weight");
    try expect(TensorRole.attn_norm, attn_norm.role);

    const other = parseTensorName("some_weird.tensor");
    try expect(TensorRole.other, other.role);
}
