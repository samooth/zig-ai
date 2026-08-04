//! ModelConfig — deriva la configuración del modelo a partir de la metadata
//! KV de un archivo GGUF (claves con prefijo `general.architecture`).
const std = @import("std");
const gguf = @import("gguf");

pub const ModelConfigError = error{
    MissingArchitecture,
    MissingRequiredMetadata,
    InvalidMetadata,
    UnsupportedArchitecture,
};

pub const ModelConfig = struct {
    architecture: []const u8, // "llama", "gemma", "mistral", ...
    context_length: usize,
    embedding_length: usize,
    block_count: usize,
    feed_forward_length: usize,
    head_count: usize,
    head_count_kv: usize,
    layer_norm_rms_epsilon: f32,
    rope_dimension_count: usize,
    rope_freq_base: f32,
    vocab_size: usize,

    pub const Self = @This();

    /// Construye la config desde metadata GGUF. `tokenizer.ggml.tokens`
    /// (si existe) se usa como fallback del vocab_size.
    pub fn fromGguf(g: *const gguf.GgufFile) ModelConfigError!Self {
        const arch = g.arch() orelse return ModelConfigError.MissingArchitecture;

        const embedding_length = try u64Meta(g, arch, "embedding_length", null);
        const block_count = try u64Meta(g, arch, "block_count", null);
        const head_count = try u64Meta(g, arch, "attention.head_count", null);

        var cfg: Self = .{
            .architecture = arch,
            .context_length = try u64Meta(g, arch, "context_length", 2048),
            .embedding_length = embedding_length,
            .block_count = block_count,
            .feed_forward_length = try u64Meta(g, arch, "feed_forward_length", null),
            .head_count = head_count,
            .head_count_kv = try u64Meta(g, arch, "attention.head_count_kv", head_count),
            .layer_norm_rms_epsilon = try f32Meta(g, arch, "attention.layer_norm_rms_epsilon", 1e-5),
            .rope_dimension_count = try u64Meta(g, arch, "rope.dimension_count", 0),
            .rope_freq_base = try f32Meta(g, arch, "rope.freq_base", 10000.0),
            .vocab_size = try u64Meta(g, arch, "vocab_size", 0),
        };

        // rope.dimension_count ausente → head_dim por cabeza
        if (cfg.rope_dimension_count == 0) {
            cfg.rope_dimension_count = cfg.embedding_length / cfg.head_count;
        }

        // vocab_size ausente → tamaño del array de tokens del tokenizer embebido
        if (cfg.vocab_size == 0) {
            if (g.getMeta("tokenizer.ggml.tokens")) |v| {
                cfg.vocab_size = v.array.items.len;
            }
        }
        return cfg;
    }

    /// True si la arquitectura es compatible con el pipeline actual (LLaMA-like)
    pub fn isSupportedArch(arch: []const u8) bool {
        const llama_like = [_][]const u8{
            "llama",    "mistral",  "mixtral", "gemma",
            "gemma2",   "falcon",   "gpt2",    "gptj",
            "phi2",     "phi3",     "qwen2",   "qwen2moe",
            "starcoder2", "deepseek2", "granite",
        };
        for (llama_like) |a| {
            if (std.mem.eql(u8, arch, a)) return true;
        }
        return false;
    }
};

/// Lee un u64 opcional con prefijo de arquitectura
fn u64Meta(g: *const gguf.GgufFile, arch: []const u8, key: []const u8, default: ?u64) ModelConfigError!u64 {
    var buf: [128]u8 = undefined;
    const full = std.fmt.bufPrint(&buf, "{s}.{s}", .{ arch, key }) catch unreachable;
    if (g.getMeta(full)) |v| {
        return v.asU64() orelse ModelConfigError.InvalidMetadata;
    }
    return default orelse ModelConfigError.MissingRequiredMetadata;
}

/// Lee un f32 opcional con prefijo de arquitectura
fn f32Meta(g: *const gguf.GgufFile, arch: []const u8, key: []const u8, default: ?f32) ModelConfigError!f32 {
    var buf: [128]u8 = undefined;
    const full = std.fmt.bufPrint(&buf, "{s}.{s}", .{ arch, key }) catch unreachable;
    if (g.getMeta(full)) |v| {
        return v.asF32() orelse ModelConfigError.InvalidMetadata;
    }
    return default orelse ModelConfigError.MissingRequiredMetadata;
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

const testing = std.testing;

fn buildFakeGguf(allocator: std.mem.Allocator) !gguf.GgufFile {
    var buf = try allocator.alloc(u8, 8192);
    defer allocator.free(buf);
    var p: usize = 0;

    std.mem.writeInt(u32, buf[p..][0..4], gguf.GGUF_MAGIC, .little);
    p += 4;
    std.mem.writeInt(u32, buf[p..][0..4], 3, .little);
    p += 4;
    std.mem.writeInt(u64, buf[p..][0..8], 0, .little); // tensor_count
    p += 8;
    std.mem.writeInt(u64, buf[p..][0..8], 6, .little); // metadata_kv_count
    p += 8;

    const Meta = gguf.MetaValueType;

    // general.architecture = "llama"
    writeStr(buf, &p, "general.architecture");
    std.mem.writeInt(u32, buf[p..][0..4], @intFromEnum(Meta.string), .little);
    p += 4;
    writeStr(buf, &p, "llama");

    // llama.block_count = 2 (uint64)
    writeStr(buf, &p, "llama.block_count");
    std.mem.writeInt(u32, buf[p..][0..4], @intFromEnum(Meta.uint64), .little);
    p += 4;
    std.mem.writeInt(u64, buf[p..][0..8], 2, .little);
    p += 8;

    // llama.embedding_length = 128 (uint64)
    writeStr(buf, &p, "llama.embedding_length");
    std.mem.writeInt(u32, buf[p..][0..4], @intFromEnum(Meta.uint64), .little);
    p += 4;
    std.mem.writeInt(u64, buf[p..][0..8], 128, .little);
    p += 8;

    // llama.attention.head_count = 4 (uint64)
    writeStr(buf, &p, "llama.attention.head_count");
    std.mem.writeInt(u32, buf[p..][0..4], @intFromEnum(Meta.uint64), .little);
    p += 4;
    std.mem.writeInt(u64, buf[p..][0..8], 4, .little);
    p += 8;

    // llama.feed_forward_length = 512 (uint64)
    writeStr(buf, &p, "llama.feed_forward_length");
    std.mem.writeInt(u32, buf[p..][0..4], @intFromEnum(Meta.uint64), .little);
    p += 4;
    std.mem.writeInt(u64, buf[p..][0..8], 512, .little);
    p += 8;

    // llama.vocab_size = 32000 (uint64)
    writeStr(buf, &p, "llama.vocab_size");
    std.mem.writeInt(u32, buf[p..][0..4], @intFromEnum(Meta.uint64), .little);
    p += 4;
    std.mem.writeInt(u64, buf[p..][0..8], 32000, .little);
    p += 8;

    return gguf.GgufFile.fromBytes(allocator, buf);
}

fn writeStr(buf: []u8, pos: *usize, s: []const u8) void {
    std.mem.writeInt(u64, buf[pos.*..][0..8], s.len, .little);
    pos.* += 8;
    @memcpy(buf[pos.*..][0..s.len], s);
    pos.* += s.len;
}

test "model_config from gguf metadata" {
    var g = try buildFakeGguf(testing.allocator);
    defer g.deinit();

    const cfg = try ModelConfig.fromGguf(&g);

    try testing.expectEqualStrings("llama", cfg.architecture);
    try testing.expectEqual(@as(usize, 2), cfg.block_count);
    try testing.expectEqual(@as(usize, 128), cfg.embedding_length);
    try testing.expectEqual(@as(usize, 512), cfg.feed_forward_length);
    try testing.expectEqual(@as(usize, 32000), cfg.vocab_size);
    try testing.expectEqual(@as(usize, 4), cfg.head_count);
    try testing.expectEqual(@as(usize, 4), cfg.head_count_kv); // default = head_count
    try testing.expectEqual(@as(usize, 2048), cfg.context_length); // default
    try testing.expectApproxEqRel(@as(f32, 10000.0), cfg.rope_freq_base, 1e-3);
}

test "model_config isSupportedArch" {
    try testing.expect(ModelConfig.isSupportedArch("llama"));
    try testing.expect(ModelConfig.isSupportedArch("mistral"));
    try testing.expect(!ModelConfig.isSupportedArch("xls_transformer"));
}
