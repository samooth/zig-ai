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
            .iq3_s => 98,
            .iq2_s => 84,
            .iq4_xs => 90,
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
    /// Copia completa del archivo (o vista mmap). El parser mantiene slices
    /// apuntando aquí, por lo que debe vivir tanto como este struct.
    data: []const u8,
    version: u32,
    alignment: u32,
    /// Offset en `data` donde empieza tensor_data
    tensor_data_offset: usize,
    tensors: std.StringHashMap(TensorInfo),
    metadata: std.StringHashMap(MetaValue),

    const Self = @This();

    pub fn deinit(self: *Self) void {
        destroyMetadata(self.allocator, &self.metadata);
        self.tensors.deinit();
        self.allocator.free(self.data);
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
        return parse(allocator, data);
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

/// Dequantizar un tensor GGUF completo a f32
pub fn dequantTensor(info: *const TensorInfo, bytes: []const u8, out: []f32) GgufError!void {
    if (out.len < info.numel()) return GgufError.InvalidData;
    switch (info.dtype) {
        .f32 => dequantF32(bytes, out),
        .f16 => dequantF16(bytes, out),
        .bf16 => dequantBF16(bytes, out),
        .q8_0 => dequantQ8_0(bytes, out),
        .q4_0 => dequantQ4_0(bytes, out),
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
