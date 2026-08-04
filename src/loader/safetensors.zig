const std = @import("std");
const Tensor = @import("core").Tensor;

/// Safetensors Loader — Parser del formato HuggingFace Safetensors
/// Formato: [header_len: u64 LE] [header_json] [tensor_data...]
/// 
/// Header JSON ejemplo:
/// {
///   "model.layers.0.self_attn.q_proj.weight": {
///     "dtype": "F16",
///     "shape": [4096, 4096],
///     "data_offsets": [0, 33554432]
///   },
///   "__metadata__": { "format": "pt" }
/// }

pub const SafetensorsError = error{
    InvalidHeader,
    InvalidJson,
    UnsupportedDtype,
    TensorNotFound,
    OffsetMismatch,
    OutOfMemory,
};

pub const Dtype = enum {
    F64, F32, F16, BF16, I64, I32, I16, I8, U8, BOOL,

    pub fn size(self: Dtype) usize {
        return switch (self) {
            .F64 => 8, .F32 => 4, .F16 => 2, .BF16 => 2,
            .I64 => 8, .I32 => 4, .I16 => 2, .I8 => 1, .U8 => 1,
            .BOOL => 1,
        };
    }

    pub fn fromString(str: []const u8) !Dtype {
        if (std.mem.eql(u8, str, "F64")) return .F64;
        if (std.mem.eql(u8, str, "F32")) return .F32;
        if (std.mem.eql(u8, str, "F16")) return .F16;
        if (std.mem.eql(u8, str, "BF16")) return .BF16;
        if (std.mem.eql(u8, str, "I64")) return .I64;
        if (std.mem.eql(u8, str, "I32")) return .I32;
        if (std.mem.eql(u8, str, "I16")) return .I16;
        if (std.mem.eql(u8, str, "I8")) return .I8;
        if (std.mem.eql(u8, str, "U8")) return .U8;
        if (std.mem.eql(u8, str, "BOOL")) return .BOOL;
        return SafetensorsError.UnsupportedDtype;
    }
};

pub const TensorInfo = struct {
    name: []const u8,  // owned
    dtype: Dtype,
    shape: []usize,    // owned
    data_offsets: [2]usize,
};

pub const SafetensorsFile = struct {
    allocator: std.mem.Allocator,
    file_data: []const u8,  // mmap o read completo
    tensors: std.StringHashMap(TensorInfo),
    metadata: std.StringHashMap([]const u8),

    const Self = @This();

    pub fn deinit(self: *Self) void {
        var tensor_iter = self.tensors.iterator();
        while (tensor_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.name);
            self.allocator.free(entry.value_ptr.shape);
        }
        self.tensors.deinit();

        var meta_iter = self.metadata.iterator();
        while (meta_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();

        self.allocator.free(self.file_data);
    }

    /// Cargar archivo safetensors desde bytes
    pub fn fromBytes(allocator: std.mem.Allocator, data: []const u8) !Self {
        if (data.len < 8) return SafetensorsError.InvalidHeader;

        const header_len = std.mem.readInt(u64, data[0..8], .little);
        if (8 + header_len > data.len) return SafetensorsError.InvalidHeader;

        const header_json = data[8..][0..header_len];

        var self = Self{
            .allocator = allocator,
            .file_data = try allocator.dupe(u8, data),
            .tensors = std.StringHashMap(TensorInfo).init(allocator),
            .metadata = std.StringHashMap([]const u8).init(allocator),
        };
        errdefer self.deinit();

        try self.parseHeader(header_json);
        return self;
    }

    /// Cargar desde archivo
    pub fn fromFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !Self {
        const dir = std.Io.Dir.cwd();
        const data = try dir.readFileAlloc(io, path, allocator, .unlimited);
        errdefer allocator.free(data);
        return try fromBytes(allocator, data);
    }

    fn parseHeader(self: *Self, json: []const u8) !void {
        // Parser JSON mínimo para safetensors
        // El formato es predecible: {"name": {"dtype": "F16", "shape": [a,b], "data_offsets": [x,y]}, ...}

        var idx: usize = 0;
        idx = skipWhitespace(json, idx);
        if (idx >= json.len or json[idx] != '{') return SafetensorsError.InvalidJson;
        idx += 1;

        while (idx < json.len) {
            idx = skipWhitespace(json, idx);
            if (idx < json.len and json[idx] == '}') break;
            if (idx >= json.len) return SafetensorsError.InvalidJson;

            // Parsear key
            const key = try parseJsonString(self.allocator, json, &idx);
            defer self.allocator.free(key);

            idx = skipWhitespace(json, idx);
            if (idx >= json.len or json[idx] != ':') return SafetensorsError.InvalidJson;
            idx += 1;
            idx = skipWhitespace(json, idx);

            if (std.mem.startsWith(u8, key, "__metadata__")) {
                // Saltar metadata por ahora
                idx = skipJsonValue(json, idx);
            } else {
                // Parsear tensor info
                const info = try self.parseTensorInfo(key, json, &idx);
                // key se mueve a info.name, no liberar
                const name_copy = try self.allocator.dupe(u8, info.name);
                errdefer self.allocator.free(name_copy);
                try self.tensors.put(name_copy, info);
            }

            idx = skipWhitespace(json, idx);
            if (idx < json.len and json[idx] == ',') {
                idx += 1;
            }
        }
    }

    fn parseTensorInfo(self: *Self, name: []const u8, json: []const u8, idx: *usize) !TensorInfo {
        var pos = idx.*;

        if (pos >= json.len or json[pos] != '{') return SafetensorsError.InvalidJson;
        pos += 1;

        var dtype: ?Dtype = null;
        var shape: ?[]usize = null;
        var offsets: ?[2]usize = null;

        while (pos < json.len) {
            pos = skipWhitespace(json, pos);
            if (pos < json.len and json[pos] == '}') break;

            const key = try parseJsonString(self.allocator, json, &pos);
            defer self.allocator.free(key);

            pos = skipWhitespace(json, pos);
            if (pos >= json.len or json[pos] != ':') return SafetensorsError.InvalidJson;
            pos += 1;
            pos = skipWhitespace(json, pos);

            if (std.mem.eql(u8, key, "dtype")) {
                const dtype_str = try parseJsonString(self.allocator, json, &pos);
                defer self.allocator.free(dtype_str);
                dtype = try Dtype.fromString(dtype_str);
            } else if (std.mem.eql(u8, key, "shape")) {
                shape = try parseShape(self.allocator, json, &pos);
            } else if (std.mem.eql(u8, key, "data_offsets")) {
                offsets = try parseOffsets(json, &pos);
            } else {
                pos = skipJsonValue(json, pos);
            }

            pos = skipWhitespace(json, pos);
            if (pos < json.len and json[pos] == ',') pos += 1;
        }

        if (pos >= json.len or json[pos] != '}') return SafetensorsError.InvalidJson;
        pos += 1;
        idx.* = pos;

        const name_owned = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_owned);

        return TensorInfo{
            .name = name_owned,
            .dtype = dtype orelse return SafetensorsError.InvalidJson,
            .shape = shape orelse return SafetensorsError.InvalidJson,
            .data_offsets = offsets orelse return SafetensorsError.InvalidJson,
        };
    }

    /// Obtener tensor como f16 (conversión si es necesario)
    pub fn getTensorF16(self: Self, allocator: std.mem.Allocator, name: []const u8) !Tensor(f16) {
        const info = self.tensors.get(name) orelse return SafetensorsError.TensorNotFound;
        if (info.dtype != .F16) return SafetensorsError.UnsupportedDtype;

        const data_start = 8 + self.file_data.len - self.file_data.len + info.data_offsets[0];
        const data_end = 8 + self.file_data.len - self.file_data.len + info.data_offsets[1];
        const tensor_bytes = self.file_data[data_start..data_end];

        const tensor = try Tensor(f16).initUninitialized(allocator, info.shape);
        @memcpy(std.mem.sliceAsBytes(tensor.data), tensor_bytes);
        return tensor;
    }

    /// Obtener raw bytes de un tensor
    pub fn getTensorRaw(self: Self, name: []const u8) ![]const u8 {
        const info = self.tensors.get(name) orelse return SafetensorsError.TensorNotFound;
        const data_start = 8 + info.data_offsets[0];
        const data_end = 8 + info.data_offsets[1];
        return self.file_data[data_start..data_end];
    }

    /// Listar todos los tensores disponibles
    pub fn listTensors(self: Self, allocator: std.mem.Allocator) ![][]const u8 {
        var names: std.ArrayList([]const u8) = .empty;
        errdefer {
            for (names.items) |n| allocator.free(n);
            names.deinit(allocator);
        }

        var iter = self.tensors.keyIterator();
        while (iter.next()) |key| {
            try names.append(allocator, try allocator.dupe(u8, key.*));
        }
        return names.toOwnedSlice(allocator);
    }
};

// ─── JSON Helpers mínimos ───

fn skipWhitespace(json: []const u8, idx: usize) usize {
    var i = idx;
    while (i < json.len and (json[i] == ' ' or json[i] == '\n' or json[i] == '\r' or json[i] == '\t')) i += 1;
    return i;
}

fn parseJsonString(allocator: std.mem.Allocator, json: []const u8, idx: *usize) ![]u8 {
    var pos = idx.*;
    if (pos >= json.len or json[pos] != '"') return SafetensorsError.InvalidJson;
    pos += 1;

    const start = pos;
    while (pos < json.len and json[pos] != '"') {
        if (json[pos] == '\\') pos += 1;
        pos += 1;
    }
    if (pos >= json.len) return SafetensorsError.InvalidJson;

    const result = try allocator.dupe(u8, json[start..pos]);
    pos += 1; // skip closing quote
    idx.* = pos;
    return result;
}

fn parseShape(allocator: std.mem.Allocator, json: []const u8, idx: *usize) ![]usize {
    var pos = idx.*;
    if (pos >= json.len or json[pos] != '[') return SafetensorsError.InvalidJson;
    pos += 1;

    var shape: std.ArrayList(usize) = .empty;
    errdefer shape.deinit(allocator);

    while (pos < json.len) {
        pos = skipWhitespace(json, pos);
        if (pos < json.len and json[pos] == ']') break;

        const start = pos;
        while (pos < json.len and std.ascii.isDigit(json[pos])) pos += 1;
        if (start == pos) return SafetensorsError.InvalidJson;

        const num = try std.fmt.parseInt(usize, json[start..pos], 10);
        try shape.append(allocator, num);

        pos = skipWhitespace(json, pos);
        if (pos < json.len and json[pos] == ',') pos += 1;
    }

    if (pos >= json.len or json[pos] != ']') return SafetensorsError.InvalidJson;
    pos += 1;
    idx.* = pos;
    return shape.toOwnedSlice(allocator);
}

fn parseOffsets(json: []const u8, idx: *usize) ![2]usize {
    var pos = idx.*;
    if (pos >= json.len or json[pos] != '[') return SafetensorsError.InvalidJson;
    pos += 1;
    pos = skipWhitespace(json, pos);

    const start1 = pos;
    while (pos < json.len and std.ascii.isDigit(json[pos])) pos += 1;
    const off1 = try std.fmt.parseInt(usize, json[start1..pos], 10);

    pos = skipWhitespace(json, pos);
    if (pos >= json.len or json[pos] != ',') return SafetensorsError.InvalidJson;
    pos += 1;
    pos = skipWhitespace(json, pos);

    const start2 = pos;
    while (pos < json.len and std.ascii.isDigit(json[pos])) pos += 1;
    const off2 = try std.fmt.parseInt(usize, json[start2..pos], 10);

    pos = skipWhitespace(json, pos);
    if (pos >= json.len or json[pos] != ']') return SafetensorsError.InvalidJson;
    pos += 1;
    idx.* = pos;
    return [2]usize{ off1, off2 };
}

fn skipJsonValue(json: []const u8, idx: usize) usize {
    var pos = skipWhitespace(json, idx);
    if (pos >= json.len) return pos;

    switch (json[pos]) {
        '"' => {
            pos += 1;
            while (pos < json.len and json[pos] != '"') {
                if (json[pos] == '\\') pos += 1;
                pos += 1;
            }
            if (pos < json.len) pos += 1;
        },
        '{', '[' => {
            const open = json[pos];
            const close: u8 = if (open == '{') '}' else ']';
            var depth: usize = 1;
            pos += 1;
            while (pos < json.len and depth > 0) {
                if (json[pos] == '"') {
                    pos += 1;
                    while (pos < json.len and json[pos] != '"') {
                        if (json[pos] == '\\') pos += 1;
                        pos += 1;
                    }
                } else if (json[pos] == open) {
                    depth += 1;
                } else if (json[pos] == close) {
                    depth -= 1;
                }
                pos += 1;
            }
        },
        else => {
            while (pos < json.len and json[pos] != ',' and json[pos] != '}' and json[pos] != ']') pos += 1;
        },
    }
    return pos;
}

// ─── Tests ───

test "safetensors parse header" {
    const allocator = std.testing.allocator;

    // Construir un safetensors mínimo en memoria
    const header = "{ \"weight\": { \"dtype\": \"F16\", \"shape\": [2,2], \"data_offsets\": [0,8] } }";
    const header_len = header.len;

    var data = try allocator.alloc(u8, 8 + header_len + 8);
    defer allocator.free(data);

    std.mem.writeInt(u64, data[0..8], header_len, .little);
    @memcpy(data[8..][0..header_len], header);

    var st = try SafetensorsFile.fromBytes(allocator, data);
    defer st.deinit();

    try std.testing.expect(st.tensors.contains("weight"));

    const info = st.tensors.get("weight").?;
    try std.testing.expectEqual(Dtype.F16, info.dtype);
    try std.testing.expectEqual(@as(usize, 2), info.shape.len);
    try std.testing.expectEqual(@as(usize, 2), info.shape[0]);
    try std.testing.expectEqual(@as(usize, 2), info.shape[1]);
}
