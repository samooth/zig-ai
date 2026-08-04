//! GgufTokenizer (D1) — extrae el tokenizer embebido en un GGUF:
//! modelo, vocab, merges, token types y special tokens.
//! Los strings son vistas prestadas a los datos del GgufFile; las arrays son
//! propiedad de este struct.
const std = @import("std");
const gguf = @import("gguf");

pub const GgufMerge = struct {
    left: []const u8,
    right: []const u8,
};

pub const GgufTokenizer = struct {
    allocator: std.mem.Allocator,
    /// "gpt2", "llama", "qwen2", "tiktoken", ...
    model: []const u8,
    /// "default", "llama", "deepseek", ...
    pre: []const u8,
    tokens: [][]const u8,
    merges: []GgufMerge,
    token_types: ?[]i32,
    bos_id: ?u32,
    eos_id: ?u32,
    unk_id: ?u32,
    pad_id: ?u32,
    add_bos: bool,
    add_eos: bool,

    const Self = @This();

    pub fn deinit(self: *Self) void {
        if (self.tokens.len > 0) self.allocator.free(self.tokens);
        if (self.merges.len > 0) self.allocator.free(self.merges);
        if (self.token_types) |tt| self.allocator.free(tt);
    }

    pub fn fromGguf(allocator: std.mem.Allocator, g: *const gguf.GgufFile) !Self {
        const model = (g.getMeta("tokenizer.ggml.model") orelse return error.NoTokenizerModel).asString() orelse return error.InvalidTokenizerModel;
        const pre = if (g.getMeta("tokenizer.ggml.pre")) |v|
            v.asString() orelse "default"
        else
            "default";

        const tokens_meta = g.getMeta("tokenizer.ggml.tokens") orelse return error.NoTokens;
        const tokens_arr = tokens_meta.array;
        const tokens = try allocator.alloc([]const u8, tokens_arr.items.len);
        errdefer allocator.free(tokens);
        for (tokens_arr.items, 0..) |item, i| {
            tokens[i] = item.string;
        }

        var merges: []GgufMerge = &.{};
        if (g.getMeta("tokenizer.ggml.merges")) |merges_meta| {
            const merges_arr = merges_meta.array;
            merges = try allocator.alloc(GgufMerge, merges_arr.items.len);
            errdefer allocator.free(merges);
            for (merges_arr.items, 0..) |item, i| {
                const s = item.string;
                const sep = std.mem.indexOfScalar(u8, s, ' ') orelse return error.InvalidMerge;
                merges[i] = .{ .left = s[0..sep], .right = s[sep + 1 ..] };
            }
        }

        var token_types: ?[]i32 = null;
        if (g.getMeta("tokenizer.ggml.token_type")) |tt_meta| {
            const tt_arr = tt_meta.array;
            const tt = try allocator.alloc(i32, tt_arr.items.len);
            errdefer allocator.free(tt);
            for (tt_arr.items, 0..) |item, i| {
                tt[i] = item.int32;
            }
            token_types = tt;
        }

        const id_u32 = struct {
            fn get(g2: *const gguf.GgufFile, key: []const u8) ?u32 {
                const v = g2.getMeta(key) orelse return null;
                if (v.asU32()) |x| return x;
                if (v.asU64()) |x| return @intCast(x);
                return null;
            }
        }.get;

        return .{
            .allocator = allocator,
            .model = model,
            .pre = pre,
            .tokens = tokens,
            .merges = merges,
            .token_types = token_types,
            .bos_id = id_u32(g, "tokenizer.ggml.bos_token_id"),
            .eos_id = id_u32(g, "tokenizer.ggml.eos_token_id"),
            .unk_id = id_u32(g, "tokenizer.ggml.unk_token_id"),
            .pad_id = id_u32(g, "tokenizer.ggml.padding_token_id"),
            .add_bos = if (g.getMeta("tokenizer.ggml.add_bos_token")) |v| (v.asBool() orelse false) else false,
            .add_eos = if (g.getMeta("tokenizer.ggml.add_eos_token")) |v| (v.asBool() orelse false) else false,
        };
    }

    /// Buscar el id de un token exacto (para tests)
    pub fn lookup(self: *const Self, token: []const u8) ?u32 {
        for (self.tokens, 0..) |t, i| {
            if (std.mem.eql(u8, t, token)) return @intCast(i);
        }
        return null;
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// Tests (con un GGUF sintético)
// ═══════════════════════════════════════════════════════════════════════════

const testing = std.testing;

fn writeStr(buf: []u8, pos: *usize, s: []const u8) void {
    std.mem.writeInt(u64, buf[pos.*..][0..8], s.len, .little);
    pos.* += 8;
    @memcpy(buf[pos.*..][0..s.len], s);
    pos.* += s.len;
}

fn buildTokenizerGguf(allocator: std.mem.Allocator) !gguf.GgufFile {
    var buf = try allocator.alloc(u8, 8192);
    defer allocator.free(buf);
    var p: usize = 0;

    std.mem.writeInt(u32, buf[p..][0..4], gguf.GGUF_MAGIC, .little);
    p += 4;
    std.mem.writeInt(u32, buf[p..][0..4], 3, .little);
    p += 4;
    std.mem.writeInt(u64, buf[p..][0..8], 0, .little);
    p += 8;
    std.mem.writeInt(u64, buf[p..][0..8], 5, .little);
    p += 8;

    const Meta = gguf.MetaValueType;

    // tokenizer.ggml.model = "gpt2"
    writeStr(buf, &p, "tokenizer.ggml.model");
    std.mem.writeInt(u32, buf[p..][0..4], @intFromEnum(Meta.string), .little);
    p += 4;
    writeStr(buf, &p, "gpt2");

    // tokenizer.ggml.tokens = ["hello", "world", "Ġhello", "Ġworld"]
    writeStr(buf, &p, "tokenizer.ggml.tokens");
    std.mem.writeInt(u32, buf[p..][0..4], @intFromEnum(Meta.array), .little);
    p += 4;
    std.mem.writeInt(u32, buf[p..][0..4], @intFromEnum(Meta.string), .little);
    p += 4;
    std.mem.writeInt(u64, buf[p..][0..8], 4, .little);
    p += 8;
    writeStr(buf, &p, "hello");
    writeStr(buf, &p, "world");
    writeStr(buf, &p, "\xc4\xa0hello");
    writeStr(buf, &p, "\xc4\xa0world");

    // tokenizer.ggml.merges = ["hello world", "Ġhello Ġworld"]
    writeStr(buf, &p, "tokenizer.ggml.merges");
    std.mem.writeInt(u32, buf[p..][0..4], @intFromEnum(Meta.array), .little);
    p += 4;
    std.mem.writeInt(u32, buf[p..][0..4], @intFromEnum(Meta.string), .little);
    p += 4;
    std.mem.writeInt(u64, buf[p..][0..8], 2, .little);
    p += 8;
    writeStr(buf, &p, "hello world");
    writeStr(buf, &p, "\xc4\xa0hello \xc4\xa0world");

    // tokenizer.ggml.bos_token_id = 1 (uint32)
    writeStr(buf, &p, "tokenizer.ggml.bos_token_id");
    std.mem.writeInt(u32, buf[p..][0..4], @intFromEnum(Meta.uint32), .little);
    p += 4;
    std.mem.writeInt(u32, buf[p..][0..4], 1, .little);
    p += 4;

    // tokenizer.ggml.eos_token_id = 2 (uint32)
    writeStr(buf, &p, "tokenizer.ggml.eos_token_id");
    std.mem.writeInt(u32, buf[p..][0..4], @intFromEnum(Meta.uint32), .little);
    p += 4;
    std.mem.writeInt(u32, buf[p..][0..4], 2, .little);
    p += 4;

    return gguf.GgufFile.fromBytes(allocator, buf);
}

test "gguf_tokenizer extraction" {
    var g = try buildTokenizerGguf(testing.allocator);
    defer g.deinit();

    var tok = try GgufTokenizer.fromGguf(testing.allocator, &g);
    defer tok.deinit();

    try testing.expectEqualStrings("gpt2", tok.model);
    try testing.expectEqual(@as(usize, 4), tok.tokens.len);
    try testing.expectEqualStrings("hello", tok.tokens[0]);
    try testing.expectEqual(@as(usize, 2), tok.merges.len);
    try testing.expectEqualStrings("hello", tok.merges[0].left);
    try testing.expectEqualStrings("world", tok.merges[0].right);
    try testing.expectEqual(@as(?u32, 1), tok.bos_id);
    try testing.expectEqual(@as(?u32, 2), tok.eos_id);
    try testing.expectEqual(@as(?u32, null), tok.unk_id);
    try testing.expectEqual(@as(u32, 1), tok.lookup("world").?);
    try testing.expectEqual(@as(u32, 2), tok.lookup("\xc4\xa0hello").?);
}
