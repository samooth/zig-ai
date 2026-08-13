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
            .iq2_xxs => 52,
            .iq2_xs => 84,
            .iq3_xxs => 74,
            .iq1_s => 50,
            .iq4_nl => 36,
            .iq3_s => 110, // d f16 + qs[64] + qh[8] + signs[32] + scales[4]
            .iq2_s => 84,
            .iq4_xs => 136, // f16 d + u16 scales_h + scales_l[4] + qs[128]
            .iq1_m => 52,
            .tq1_0 => 74,
            .tq2_0 => 36,
            .mxfp4 => 72,
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

/// Q4_0: bloques de 32. Cada bloque: f16 d, uint8 qs[16] (nibbles, low first).
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
        for (0..n) |j| {
            const q = if (j % 2 == 0) qs[j / 2] & 0x0F else qs[j / 2] >> 4;
            out[i + j] = d * @as(f32, @floatFromInt(@as(i8, @intCast(q)) - 8));
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
        .q8_0 => dequantQ8_0(bytes, out[0..bs]),
        .q4_0 => dequantQ4_0(bytes, out[0..bs]),
        .q4_k => dequantQ4_K(bytes, out[0..bs]),
        .q5_k => dequantQ5_K(bytes, out[0..bs]),
        .q6_k => dequantQ6_K(bytes, out[0..bs]),
        .iq4_xs => dequantIq4_xs(bytes, out[0..bs]),
        .iq3_s => dequantIq3_s(bytes, out[0..bs]),
        else => @panic("dequantBlock: dtype sin soporte"),
    }
}

/// Dequantizar un tensor GGUF completo a f32
pub fn dequantTensor(info: *const TensorInfo, bytes: []const u8, out: []f32) GgufError!void {
    if (out.len < info.numel()) return GgufError.InvalidData;
    switch (info.dtype) {
        .f32 => dequantF32(bytes, out),
        .f16 => dequantF16(bytes, out),
        .bf16 => dequantBF16(bytes, out),
        .q8_0 => dequantQ8_0(bytes, out),
        .q4_0 => dequantQ4_0(bytes, out),
        .q4_k => dequantQ4_K(bytes, out),
        .q5_k => dequantQ5_K(bytes, out),
        .q6_k => dequantQ6_K(bytes, out),
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
