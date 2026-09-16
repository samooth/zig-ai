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

pub const RopeScalingType = enum { none, linear, yarn };

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
    shortconv_l_cache: usize = 0, // conv kernel size - 1 (l_cache=3 -> kernel=4)

    // K2-Horizon hybrid (Grouped RMSNorm + optional Q/K norms + optional softplus gate + MoE/MoVA)
    is_k2_horizon: bool = false,
    n_norm_groups: usize = 1, // number of groups for grouped RMSNorm
    has_attn_q_norm: bool = false, // whether Q has per-head grouped RMSNorm
    has_attn_k_norm: bool = false, // whether K has per-head grouped RMSNorm
    has_attn_gate: bool = false, // whether attention has softplus gate

    // K2-Horizon MoVA
    n_value_expert: usize = 0,
    n_value_expert_used: usize = 0,

    // RoPE scaling (YaRN / linear / none)
    rope_scaling_type: RopeScalingType = .none,
    rope_scaling_factor: f32 = 1.0,
    rope_scaling_orig_ctx: usize = 0,
    rope_attn_factor: f32 = 1.0,
    yarn_ext_factor: f32 = 1.0,
    yarn_attn_factor: f32 = 1.0,
    yarn_beta_fast: f32 = 32.0,
    yarn_beta_slow: f32 = 1.0,

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
            // 7.1e (lane-f): GQA real — el GGUF declara attention.head_count_kv
            // (llama-3.2: 8 con 24 query heads). Antes el default genérico era
            // head_count ⇒ modelos GQA (llama-3.x, qwen2moe...) construían
            // slabs k/v 3× sobredimensionados y shapes GEMM mal (assert en
            // linearProjection). Solo qwen35/lfm2 lo sobreescrbían después.
            // 2026-09-09 lane-e: kvHeadsMeta TAMBIÉN en la init genérica —
            // lfm2.head_count_kv es un ARRAY per-layer (30×i32): u64Meta
            // revienta con InvalidMetadata ANTES de llegar al bloque lfm2
            // que lo parsea bien (arrI32Meta). Escalar→u64, array→primer
            // no-cero (el bloque de arch puede refinarlo después).
            .head_count_kv = try kvHeadsMeta(g, arch, head_count),
            .layer_norm_rms_epsilon = try f32Meta(g, arch, "attention.layer_norm_rms_epsilon", 1e-5),
            .rope_dimension_count = try u64Meta(g, arch, "rope.dimension_count", 0),
            .rope_freq_base = try f32Meta(g, arch, "rope.freq_base", 10000.0),
            .vocab_size = try u64Meta(g, arch, "vocab_size", 0),
        };

        // rope.dimension_count ausente → head_dim por cabeza
        if (cfg.rope_dimension_count == 0) {
            cfg.rope_dimension_count = cfg.embedding_length / cfg.head_count;
        }

        // rope_sections: si la familia no define rope.dimension_sections
        // (IMROPE de qwen35), TODO el bloque rotatorio va en la sección 0 —
        // equivalente al RoPE estándar (assert de applyRoPEMultiSection:
        // sum(sections) == n_rot/2).
        if (cfg.rope_sections[0] + cfg.rope_sections[1] + cfg.rope_sections[2] + cfg.rope_sections[3] == 0) {
            cfg.rope_sections[0] = cfg.rope_dimension_count / 2;
        }

        // vocab_size ausente → tamaño del array de tokens del tokenizer embebido
        if (cfg.vocab_size == 0) {
            if (g.getMeta("tokenizer.ggml.tokens")) |v| {
                cfg.vocab_size = v.array.items.len;
            }
        }

        // ── Qwen3.5 hybrid (qwen35 / qwen35moe) ──
        // NOTA: `qwen3moe` NO pertenece aquí: es el MoE denso clásico
        // (attn_q/k/v separados, sin ssm_*/attn_qkv) — va por la rama
        // genérica llama-like (igual que qwen2moe).
        if (std.mem.eql(u8, arch, "qwen35") or
            std.mem.eql(u8, arch, "qwen35moe"))
        {
            cfg.is_hybrid = true;
            cfg.head_dim = try u64Meta(g, arch, "attention.key_length", cfg.embedding_length / cfg.head_count);
            cfg.head_count_kv = try kvHeadsMeta(g, arch, cfg.head_count);
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
            } else {
                // 2026-09-12 (lane-b): LFM2-350M trae head_count_kv como
                // array VACÍO ⇒ n_kv=0 ⇒ sin per_layer_attn, 0 capas attn
                // en un modelo híbrido real (16Q, RoPE 1e6) ⇒ panic
                // "non-equal lengths" en hybrid_layer (todas ShortConv con
                // shapes de atención). Fallback robusto espejo de
                // llama.cpp lfm2.cpp:10 (is_recr = n_head_kv==0; capa
                // attention tiene attn_q_norm, recurrente tiene
                // shortconv.conv): sondear el tensor por capa.
                var per_layer = try g.allocator.alloc(bool, cfg.block_count);
                errdefer g.allocator.free(per_layer);
                var any_attn = false;
                for (0..cfg.block_count) |i| {
                    var buf: [128]u8 = undefined;
                    const qn = std.fmt.bufPrint(&buf, "blk.{d}.attn_q_norm.weight", .{i}) catch unreachable;
                    const is_attn = g.getTensor(qn) != null;
                    per_layer[i] = is_attn;
                    any_attn = any_attn or is_attn;
                }
                if (any_attn) {
                    cfg.per_layer_attn = per_layer;
                    cfg.head_count_kv = cfg.head_count;
                    // MHA: LFM2 attention layers son MQA-less full MHA
                    // (n_head_kv == n_head cuando el array viene vacío).
                } else {
                    g.allocator.free(per_layer);
                }
            }

            // shortconv.l_cache = 3 (kernel size = l_cache + 1 = 4)
            cfg.shortconv_l_cache = try u64Meta(g, arch, "shortconv.l_cache", 3);
        }

        // ── DFlash sidecar (5.2 lane-b1): draft-model del sidecar DFlash/DSpark.
        // Geometría Qwen3-style clásica (GQA + q/k_norm + SwiGLU par), NO
        // híbrido. hparams bajo prefijo "dflash.*". Verificado contra
        // qwen35-9b-dflash-Q8_0.gguf real: block_count=6, embd=4096,
        // heads=32, kv=8, key_length=128 (head_dim explícito), ffn=12288.
        if (std.mem.eql(u8, arch, "dflash")) {
            cfg.is_hybrid = false;
            cfg.head_dim = try u64Meta(g, arch, "attention.key_length", cfg.embedding_length / cfg.head_count);
            cfg.head_count_kv = try kvHeadsMeta(g, arch, cfg.head_count);
        }

        // ── K2-Horizon hybrid (Grouped RMSNorm + optional Q/K norms + softplus gate + MoVA) ──
        if (std.mem.eql(u8, arch, "k2-horizon")) {
            cfg.is_hybrid = true;
            cfg.is_k2_horizon = true;
            cfg.head_dim = try u64Meta(g, arch, "attention.key_length", cfg.embedding_length / cfg.head_count);

            // Grouped RMSNorm: number of groups (default 1 = standard RMSNorm)
            cfg.n_norm_groups = try u64Meta(g, arch, "attention.group_norm_groups", 1);
            if (cfg.n_norm_groups == 0) cfg.n_norm_groups = 1;

            // Optional Q/K norms (per-head grouped RMSNorm) — detectados
            // por presencia de tensor en el GGUF, no por metadata KV.
            cfg.has_attn_q_norm = false;
            cfg.has_attn_k_norm = false;
            {
                var buf: [128]u8 = undefined;
                const q_norm_name = std.fmt.bufPrint(&buf, "blk.0.attn_q_norm.weight", .{}) catch unreachable;
                if (g.getTensor(q_norm_name)) |_| cfg.has_attn_q_norm = true;
            }
            {
                var buf: [128]u8 = undefined;
                const k_norm_name = std.fmt.bufPrint(&buf, "blk.0.attn_k_norm.weight", .{}) catch unreachable;
                if (g.getTensor(k_norm_name)) |_| cfg.has_attn_k_norm = true;
            }

            // Optional softplus gate on attention output — detectado por
            // presencia de tensor attn_gate.weight en el GGUF.
            cfg.has_attn_gate = false;
            {
                var buf: [128]u8 = undefined;
                const gate_name = std.fmt.bufPrint(&buf, "blk.0.attn_gate.weight", .{}) catch unreachable;
                if (g.getTensor(gate_name)) |_| cfg.has_attn_gate = true;
            }

            // MoVA
            cfg.n_value_expert = try u64Meta(g, arch, "attention.value_expert_count", 0);
            cfg.n_value_expert_used = try u64Meta(g, arch, "attention.value_expert_used_count", 0);
        }

        // ── RoPE scaling (YaRN / linear / none) ──
        if (strMeta(g, arch, "rope.scaling.type", null)) |scaling_type| {
            if (std.mem.eql(u8, scaling_type, "linear")) {
                cfg.rope_scaling_type = .linear;
            } else if (std.mem.eql(u8, scaling_type, "yarn")) {
                cfg.rope_scaling_type = .yarn;
            }
        }
        cfg.rope_scaling_factor = try f32Meta(g, arch, "rope.scaling.factor", 1.0);
        cfg.rope_scaling_orig_ctx = @as(usize, try u64Meta(g, arch, "rope.scaling.original_context_length", 0));
        cfg.rope_attn_factor = try f32Meta(g, arch, "rope.scaling.attn_factor", 1.0);
        cfg.yarn_ext_factor = try f32Meta(g, arch, "rope.scaling.yarn_ext_factor", 1.0);
        cfg.yarn_attn_factor = try f32Meta(g, arch, "rope.scaling.yarn_attn_factor", 1.0);
        cfg.yarn_beta_fast = try f32Meta(g, arch, "rope.scaling.yarn_beta_fast", 32.0);
        cfg.yarn_beta_slow = try f32Meta(g, arch, "rope.scaling.yarn_beta_slow", 1.0);

        return cfg;
    }

    /// C6.2 (5.2, lane-c): config sintética del SIDEcar DFlash/DSpark para
    /// instanciar sus capas blk.* como HybridLayers de atención densa.
    /// Los hparams viven con prefijo "dflash." (arch real del GGUF); el
    /// resto (rope/eps) sigue el mismo convenio. vocab_size: los sidecars
    /// NO traen token list usable — el caller (target) lo hereda.
    pub fn fromSidecarDflash(g: *const gguf.GgufFile, vocab_size: usize) ModelConfigError!Self {
        const arch = "dflash";
        var cfg: Self = .{
            .architecture = "qwen35", // las capas blk.* del sidecar son attn densa estándar
            .context_length = try u64Meta(g, arch, "context_length", 2048),
            .embedding_length = try u64Meta(g, arch, "embedding_length", null),
            .block_count = try u64Meta(g, arch, "block_count", null),
            .feed_forward_length = try u64Meta(g, arch, "feed_forward_length", null),
            .head_count = try u64Meta(g, arch, "attention.head_count", null),
            .head_count_kv = try u64Meta(g, arch, "attention.head_count_kv", null),
            .layer_norm_rms_epsilon = try f32Meta(g, arch, "attention.layer_norm_rms_epsilon", 1e-6),
            .rope_dimension_count = try u64Meta(g, arch, "attention.key_length", 128),
            .rope_freq_base = try f32Meta(g, arch, "rope.freq_base", 10000000.0),
            .vocab_size = vocab_size, // heredado del target (el sidecar no trae tokenizer)
        };
        cfg.head_dim = try u64Meta(g, arch, "attention.key_length", 128);
        // sidecar = SOLO capas de atención (sin SSM/shortconv): el patrón
        // full_attention_interval no aplica; hybrid_layer decide por tensor.
        cfg.is_hybrid = false;
        return cfg;
    }

    /// True si la capa `il` usa atención densa (Qwen3.5 hybrid).
    /// Las capas recurrentes (SSM linear attention) se intercalan cada
    /// `full_attention_interval` capas: capa i es atención si (i+1)%interval == 0.
    /// Para LFM2: usa per_layer_attn array si está disponible.
    /// Para K2-Horizon: todas las capas son atención (MoE en FFN, no en attn).
    pub fn isFullAttentionLayer(self: Self, il: usize) bool {
        if (!self.is_hybrid) return true;
        // K2-Horizon: todas las capas usan atención
        if (std.mem.eql(u8, self.architecture, "k2-horizon")) return true;
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
            "llama",        "mistral",    "mixtral",  "gemma",
            "gemma2",       "falcon",     "gpt2",     "gptj",
            "phi2",         "phi3",       "qwen2",    "qwen2moe",
            "starcoder2",   "deepseek2",  "granite",  "qwen35",
            "qwen35moe",    "qwen3moe",   "lfm2",     "bitnet",
            "bitnet-b1.58", "k2-horizon",
        };
        for (llama_like) |a| {
            if (std.mem.eql(u8, arch, a)) return true;
        }
        return false;
    }

    /// True si la arquitectura es K2-Horizon
    pub fn isK2Horizon(arch: []const u8) bool {
        return std.mem.eql(u8, arch, "k2-horizon");
    }

    /// BitNet b1.58 (lane-kvc P2): llama-like con activación squared_relu y
    /// sub-norms (attn_sub_norm tras attention pre-wo, ffn_sub_norm entre
    /// activación y down). El GGUF real declara "bitnet-b1.58"; llama.cpp
    /// mapea "bitnet" — aceptamos ambas.
    pub fn isBitnet(arch: []const u8) bool {
        return std.mem.eql(u8, arch, "bitnet") or
            std.mem.eql(u8, arch, "bitnet-b1.58");
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

/// Lee attention.head_count_kv para qwen35: acepta un escalar u64 o un array
/// (primer valor no cero, p.ej. per-layer como LFM2). Devuelve `default`
/// (head_count) si la clave no existe o no hay valor útil.
fn kvHeadsMeta(g: *const gguf.GgufFile, arch: []const u8, default: usize) ModelConfigError!usize {
    var buf: [128]u8 = undefined;
    const full = std.fmt.bufPrint(&buf, "{s}.{s}", .{ arch, "attention.head_count_kv" }) catch unreachable;
    const v = g.getMeta(full) orelse return default;
    if (v.asU64()) |x| {
        if (x > 0) return @intCast(x);
        return default;
    }
    // Array (int32 per-layer): primer valor no cero
    for (v.array.items) |it| {
        if (it.asU64()) |x| {
            if (x > 0) return @intCast(x);
        }
    }
    return default;
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

/// Lee un string opcional con prefijo de arquitectura
fn strMeta(
    g: *const gguf.GgufFile,
    arch: []const u8,
    key: []const u8,
    default: ?[]const u8,
) ?[]const u8 {
    var buf: [128]u8 = undefined;
    const full = std.fmt.bufPrint(&buf, "{s}.{s}", .{ arch, key }) catch return default;
    if (g.getMeta(full)) |v| {
        return v.asString();
    }
    return default;
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

fn writeMetaU64(buf: []u8, pos: *usize, key: []const u8, v: u64) void {
    writeStr(buf, pos, key);
    std.mem.writeInt(u32, buf[pos.*..][0..4], @intFromEnum(gguf.MetaValueType.uint64), .little);
    pos.* += 4;
    std.mem.writeInt(u64, buf[pos.*..][0..8], v, .little);
    pos.* += 8;
}

/// Escribe metadata array de int32: [item_type u32][len u64][items...]
fn writeMetaI32Array(buf: []u8, pos: *usize, key: []const u8, items: []const i32) void {
    writeStr(buf, pos, key);
    std.mem.writeInt(u32, buf[pos.*..][0..4], @intFromEnum(gguf.MetaValueType.array), .little);
    pos.* += 4;
    std.mem.writeInt(u32, buf[pos.*..][0..4], @intFromEnum(gguf.MetaValueType.int32), .little);
    pos.* += 4;
    std.mem.writeInt(u64, buf[pos.*..][0..8], items.len, .little);
    pos.* += 8;
    for (items) |it| {
        std.mem.writeInt(i32, buf[pos.*..][0..4], it, .little);
        pos.* += 4;
    }
}

/// GGUF fake de qwen35 (64 capas, 24 Q heads) con `head_count_kv` opcional.
/// `kv_heads`: null → sin clave (default), 0 → escalar, >0 → escalar.
fn buildFakeQwen35Gguf(
    allocator: std.mem.Allocator,
    kv_heads: ?u64,
    kv_heads_array: ?[]const i32,
) !gguf.GgufFile {
    var buf = try allocator.alloc(u8, 8192);
    defer allocator.free(buf);
    var p: usize = 0;

    std.mem.writeInt(u32, buf[p..][0..4], gguf.GGUF_MAGIC, .little);
    p += 4;
    std.mem.writeInt(u32, buf[p..][0..4], 3, .little);
    p += 4;
    std.mem.writeInt(u64, buf[p..][0..8], 0, .little); // tensor_count
    p += 8;
    var kv_count: usize = 7; // architecture string + 6 uint64
    if (kv_heads != null) kv_count += 1;
    if (kv_heads_array != null) kv_count += 1;
    std.mem.writeInt(u64, buf[p..][0..8], kv_count, .little);
    p += 8;

    writeStr(buf, &p, "general.architecture");
    std.mem.writeInt(u32, buf[p..][0..4], @intFromEnum(gguf.MetaValueType.string), .little);
    p += 4;
    writeStr(buf, &p, "qwen35");

    writeMetaU64(buf, &p, "qwen35.block_count", 64);
    writeMetaU64(buf, &p, "qwen35.embedding_length", 5120);
    writeMetaU64(buf, &p, "qwen35.attention.head_count", 24);
    writeMetaU64(buf, &p, "qwen35.attention.key_length", 256);
    writeMetaU64(buf, &p, "qwen35.feed_forward_length", 17408);
    writeMetaU64(buf, &p, "qwen35.vocab_size", 248320);
    if (kv_heads) |kv| writeMetaU64(buf, &p, "qwen35.attention.head_count_kv", kv);
    if (kv_heads_array) |arr| writeMetaI32Array(buf, &p, "qwen35.attention.head_count_kv", arr);

    return gguf.GgufFile.fromBytes(allocator, buf);
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
    // GGUF del loggenix-moe reporta arch=qwen3moe (no qwen35moe).
    try testing.expect(ModelConfig.isSupportedArch("qwen3moe"));
    // BitNet b1.58 (lane-kvc P2): el GGUF real declara "bitnet-b1.58";
    // llama.cpp mapea "bitnet" — ambas cadenas deben pasar.
    try testing.expect(ModelConfig.isSupportedArch("bitnet-b1.58"));
    try testing.expect(ModelConfig.isSupportedArch("bitnet"));
    try testing.expect(ModelConfig.isBitnet("bitnet-b1.58"));
    try testing.expect(ModelConfig.isBitnet("bitnet"));
    try testing.expect(!ModelConfig.isBitnet("llama"));
}

test "model_config qwen35 reads scalar head_count_kv" {
    var g = try buildFakeQwen35Gguf(testing.allocator, 4, null);
    defer g.deinit();

    const cfg = try ModelConfig.fromGguf(&g);

    try testing.expect(cfg.is_hybrid);
    try testing.expectEqualStrings("qwen35", cfg.architecture);
    try testing.expectEqual(@as(usize, 64), cfg.block_count);
    try testing.expectEqual(@as(usize, 24), cfg.head_count);
    try testing.expectEqual(@as(usize, 4), cfg.head_count_kv); // Qwen3.8-27B: 4 KV heads
    try testing.expectEqual(@as(usize, 256), cfg.head_dim);
    try testing.expectEqual(@as(usize, 4), cfg.full_attention_interval);
}

test "model_config qwen35 head_count_kv array (per-layer)" {
    var g = try buildFakeQwen35Gguf(testing.allocator, null, &.{ 0, 0, 0, 4, 0, 0, 0, 4 });
    defer g.deinit();

    const cfg = try ModelConfig.fromGguf(&g);

    try testing.expectEqual(@as(usize, 4), cfg.head_count_kv); // primer valor no cero
}

test "model_config qwen35 head_count_kv defaults to head_count" {
    var g = try buildFakeQwen35Gguf(testing.allocator, null, null);
    defer g.deinit();

    const cfg = try ModelConfig.fromGguf(&g);

    try testing.expectEqual(@as(usize, 24), cfg.head_count_kv);
}

test "model_config k2-horizon dense 0.9B" {
    var buf = try testing.allocator.alloc(u8, 4096);
    defer testing.allocator.free(buf);
    var p: usize = 0;

    std.mem.writeInt(u32, buf[p..][0..4], gguf.GGUF_MAGIC, .little);
    p += 4;
    std.mem.writeInt(u32, buf[p..][0..4], 3, .little);
    p += 4;
    std.mem.writeInt(u64, buf[p..][0..8], 0, .little);
    p += 8;
    std.mem.writeInt(u64, buf[p..][0..8], 12, .little);
    p += 8;

    const Meta = gguf.MetaValueType;

    writeStr(buf, &p, "general.architecture");
    std.mem.writeInt(u32, buf[p..][0..4], @intFromEnum(Meta.string), .little);
    p += 4;
    writeStr(buf, &p, "k2-horizon");

    writeMetaU64(buf, &p, "k2-horizon.block_count", 28);
    writeMetaU64(buf, &p, "k2-horizon.embedding_length", 1536);
    writeMetaU64(buf, &p, "k2-horizon.attention.head_count", 12);
    writeMetaU64(buf, &p, "k2-horizon.attention.head_count_kv", 4);
    writeMetaU64(buf, &p, "k2-horizon.feed_forward_length", 8192);
    writeMetaU64(buf, &p, "k2-horizon.vocab_size", 151936);
    writeMetaU64(buf, &p, "k2-horizon.context_length", 32768);
    writeMetaU64(buf, &p, "k2-horizon.attention.layer_norm_rms_epsilon", 1e-6);
    writeMetaU64(buf, &p, "k2-horizon.rope.dimension_count", 64);
    writeMetaU64(buf, &p, "k2-horizon.attention.group_norm_groups", 4);

    var g = try gguf.GgufFile.fromBytes(testing.allocator, buf);
    defer g.deinit();

    const cfg = try ModelConfig.fromGguf(&g);

    try testing.expectEqualStrings("k2-horizon", cfg.architecture);
    try testing.expect(cfg.is_hybrid);
    try testing.expect(cfg.is_k2_horizon);
    try testing.expectEqual(@as(usize, 28), cfg.block_count);
    try testing.expectEqual(@as(usize, 1536), cfg.embedding_length);
    try testing.expectEqual(@as(usize, 12), cfg.head_count);
    try testing.expectEqual(@as(usize, 4), cfg.head_count_kv);
    try testing.expectEqual(@as(usize, 8192), cfg.feed_forward_length);
    try testing.expectEqual(@as(usize, 4), cfg.n_norm_groups);
}
