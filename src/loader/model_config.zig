//! ModelConfig — deriva la configuración del modelo a partir de la metadata
//! KV de un archivo GGUF (claves con prefijo `general.architecture`).
const std = @import("std");
const gguf = @import("gguf");

pub const ModelConfigError = error{
    MissingArchitecture,
    MissingRequiredMetadata,
    InvalidMetadata,
    UnsupportedArchitecture,
    OutOfMemory,
};

pub const ModelConfig = struct {
    architecture: []const u8, // "llama", "gemma", "mistral", "qwen35", "lfm2", ...
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

    // Qwen3.5 / qwen35 hybrid (SSM + attention)
    is_hybrid: bool = false,
    head_dim: usize = 0, // dimensión de cabeza de atención (key_length)
    full_attention_interval: usize = 0, // capa i es atención si (i+1)%interval==0
    ssm_conv_kernel: usize = 0,
    ssm_inner_size: usize = 0,
    ssm_state_size: usize = 0,
    ssm_time_step_rank: usize = 0,
    ssm_group_count: usize = 0,
    rope_sections: [4]usize = [_]usize{ 0, 0, 0, 0 }, // IMROPE sections

    // LFM2 hybrid (ShortConv + Attention)
    per_layer_attn: ?[]bool = null, // per-layer: true=attention, false=shortconv
    shortconv_l_cache: usize = 0,   // conv kernel size - 1 (l_cache=3 -> kernel=4)

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
            .head_count_kv = head_count, // default, will be overridden for LFM2
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

        // ── Qwen3.5 hybrid (qwen35 / qwen35moe) ──
        if (std.mem.eql(u8, arch, "qwen35") or std.mem.eql(u8, arch, "qwen35moe")) {
            cfg.is_hybrid = true;
            cfg.head_dim = try u64Meta(g, arch, "attention.key_length", cfg.embedding_length / cfg.head_count);
            cfg.full_attention_interval = try u64Meta(g, arch, "attention.full_attention_interval", 4);
            cfg.ssm_conv_kernel = try u64Meta(g, arch, "ssm.conv_kernel", 0);
            cfg.ssm_inner_size = try u64Meta(g, arch, "ssm.inner_size", 0);
            cfg.ssm_state_size = try u64Meta(g, arch, "ssm.state_size", 0);
            cfg.ssm_time_step_rank = try u64Meta(g, arch, "ssm.time_step_rank", 0);
            cfg.ssm_group_count = try u64Meta(g, arch, "ssm.group_count", 0);

            // rope.dimension_sections (array de 4 enteros, IMROPE)
            var sections_buf: [4]u64 = undefined;
            const n_sections = try arrU64Meta(g, arch, "rope.dimension_sections", &sections_buf) orelse 0;
            for (0..@min(4, n_sections)) |i| cfg.rope_sections[i] = sections_buf[i];
        }

        // ── LFM2 hybrid (ShortConv + Attention) ──
        if (std.mem.eql(u8, arch, "lfm2")) {
            cfg.is_hybrid = true;
            cfg.head_dim = cfg.embedding_length / cfg.head_count; // 64

            // head_count_kv es un array de 30 int32: capas [2,5,9,13,17,21,24,27] tienen 8, resto 0
            var kv_heads_buf: [64]i32 = undefined;
            const n_kv = try arrI32Meta(g, arch, "attention.head_count_kv", &kv_heads_buf) orelse 0;
            if (n_kv > 0) {
                // First non-zero value is the attention layer's kv_heads (8)
                for (0..n_kv) |i| {
                    if (kv_heads_buf[i] > 0) {
                        cfg.head_count_kv = @as(usize, @intCast(kv_heads_buf[i]));
                        break;
                    }
                }
                // If all zero, fallback to head_count
                if (cfg.head_count_kv == 0) {
                    cfg.head_count_kv = cfg.head_count;
                }

                // Build per-layer attention flag array
                var per_layer = try g.allocator.alloc(bool, cfg.block_count);
                errdefer g.allocator.free(per_layer);
                for (0..cfg.block_count) |i| {
                    per_layer[i] = (i < n_kv and kv_heads_buf[i] > 0);
                }
                cfg.per_layer_attn = per_layer;
            }

            // shortconv.l_cache = 3 (kernel size = l_cache + 1 = 4)
            cfg.shortconv_l_cache = try u64Meta(g, arch, "shortconv.l_cache", 3);
        }

        return cfg;
    }

    /// True si la capa `il` usa atención densa (Qwen3.5 hybrid).
    /// Las capas recurrentes (SSM linear attention) se intercalan cada
    /// `full_attention_interval` capas: capa i es atención si (i+1)%interval == 0.
    /// Para LFM2: usa per_layer_attn array si está disponible.
    pub fn isFullAttentionLayer(self: Self, il: usize) bool {
        if (!self.is_hybrid) return true;
        // LFM2: per-layer attention flags
        if (std.mem.eql(u8, self.architecture, "lfm2")) {
            if (self.per_layer_attn) |arr| {
                if (il < arr.len) return arr[il];
            }
            return false;
        }
        // Qwen3.5: periodic pattern
        if (self.full_attention_interval == 0) return true;
        return (il + 1) % self.full_attention_interval == 0;
    }

    /// True si la arquitectura es compatible con el pipeline actual (LLaMA-like)
    pub fn isSupportedArch(arch: []const u8) bool {
        const llama_like = [_][]const u8{
            "llama",    "mistral",  "mixtral", "gemma",
            "gemma2",   "falcon",   "gpt2",    "gptj",
            "phi2",     "phi3",     "qwen2",   "qwen2moe",
            "starcoder2", "deepseek2", "granite",
            "qwen35",   "qwen35moe",
            "lfm2",
        };
        for (llama_like) |a| {
            if (std.mem.eql(u8, arch, a)) return true;
        }
        return false;
    }
};

/// Lee un array de u64 con prefijo de arquitectura en el buffer `out`,
/// devolviendo el número de elementos leídos (0 si la clave no existe).
fn arrU64Meta(
    g: *const gguf.GgufFile,
    arch: []const u8,
    key: []const u8,
    out: []u64,
) ModelConfigError!?usize {
    var buf: [128]u8 = undefined;
    const full = std.fmt.bufPrint(&buf, "{s}.{s}", .{ arch, key }) catch unreachable;
    const v = g.getMeta(full) orelse return null;
    const n = @min(out.len, v.array.items.len);
    for (v.array.items[0..n], 0..) |it, i| {
        out[i] = it.asU64() orelse return ModelConfigError.InvalidMetadata;
    }
    return n;
}

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

/// Lee un array de int32 (para lfm2.attention.head_count_kv que es array de 30 int32)
fn arrI32Meta(
    g: *const gguf.GgufFile,
    arch: []const u8,
    key: []const u8,
    out: []i32,
) ModelConfigError!?usize {
    var buf: [128]u8 = undefined;
    const full = std.fmt.bufPrint(&buf, "{s}.{s}", .{ arch, key }) catch unreachable;
    const v = g.getMeta(full) orelse return null;
    const n = @min(out.len, v.array.items.len);
    for (v.array.items[0..n], 0..) |it, i| {
        out[i] = it.asI32() orelse return ModelConfigError.InvalidMetadata;
    }
    return n;
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
