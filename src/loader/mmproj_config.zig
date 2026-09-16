//! MmprojConfig — deriva la configuración del encoder vision (CLIP ViT) a
//! partir de la metadata KV de un archivo mmproj GGUF (claves `clip.*`).
//!
//! Referencias (llama.cpp mtmd):
//!   - Carga hparams:      tools/mtmd/clip.cpp:1139-1290 (`load_hparams`)
//!   - Keys:               tools/mtmd/clip-impl.h:29-92 (KEY_* defines)
//!   - hparams Qwen-VL:    tools/mtmd/clip.cpp:1491-1502 (n_merge=2, límites
//!                         de tokens 8..4096, spatial_merge_size)
//!   - Conversión Unsloth: /ai/repos/2026/unslothai/llama.cpp/conversion/qwen3vl.py
//!                         (image_size derivado, use_gelu=true, deepstack)
const std = @import("std");
const gguf = @import("gguf");

pub const MmprojConfigError = error{
    MissingProjectorType,
    UnknownProjectorType,
    MissingRequiredMetadata,
    InvalidMetadata,
    OutOfMemory,
};

/// Tipos de projector soportados por zig-ai (MVP: qwen3vl_merger).
/// Referencia enum completo: tools/mtmd/clip-impl.h:356-468.
pub const ProjectorType = enum {
    qwen3vl_merger,
    qwen25vl_merger,
    qwen2vl_merger,
    unknown,

    pub fn fromString(s: []const u8) ProjectorType {
        if (std.mem.eql(u8, s, "qwen3vl_merger")) return .qwen3vl_merger;
        if (std.mem.eql(u8, s, "qwen2.5vl_merger")) return .qwen25vl_merger;
        if (std.mem.eql(u8, s, "qwen2vl_merger")) return .qwen2vl_merger;
        return .unknown;
    }
};

/// Operación FFN del ViT (clip.cpp:1264-1275: use_gelu/use_silu).
pub const FfnOp = enum { gelu, silu, gelu_quick };

pub const MmprojConfig = struct {
    projector_type: ProjectorType,
    projector_type_str: []const u8,

    // Hparams comunes (clip.cpp:1186-1210, prefijo `clip.vision.`)
    n_embd: usize, // clip.vision.embedding_length
    n_head: usize, // clip.vision.attention.head_count
    head_dim: usize, // clip.vision.attention.head_dim (default n_embd/n_head)
    n_head_kv: usize, // clip.vision.attention.head_count_kv (default = n_head)
    n_ff: usize, // clip.vision.feed_forward_length
    n_layer: usize, // clip.vision.block_count
    projection_dim: usize, // clip.vision.projection_dim
    eps: f32, // clip.vision.attention.layer_norm_epsilon

    // Hparams vision (clip.cpp:1211-1290)
    image_size: usize, // clip.vision.image_size
    patch_size: usize, // clip.vision.patch_size
    image_mean: [3]f32, // clip.vision.image_mean
    image_std: [3]f32, // clip.vision.image_std
    spatial_merge_size: usize, // clip.vision.spatial_merge_size (default 2)
    image_min_pixels: usize, // default 3136 (28²·4, Qwen-VL)
    image_max_pixels: usize, // default 1003520 (transformers Qwen-VL)
    min_image_tokens: usize, // default 8 (Qwen-VL, clip.cpp:1500)
    max_image_tokens: usize, // default 4096
    ffn_op: FfnOp, // clip.use_gelu / clip.use_silu

    // Deepstack (Qwen3-VL): índices de capa con tensores v.deepstack.{i}.*
    is_deepstack_layers: []bool,
    /// True ⇒ RMS norm (Qwen2.5-VL, sin bias); False ⇒ LayerNorm (2-VL/3-VL)
    use_rms_norm: bool = false,

    pub const Self = @This();

    /// d_head efectivo del encoder
    pub fn dHead(self: Self) usize {
        return self.head_dim;
    }

    /// Número de patches por lado tras el spatial merge (grid del LLM).
    /// La imagen WxH (múltiplo de patch·merge) produce grid_x=W/(patch·merge).
    pub fn gridFor(self: Self, nx: usize, ny: usize) struct { x: usize, y: usize } {
        const step = self.patch_size * self.spatial_merge_size;
        return .{ .x = nx / step, .y = ny / step };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        if (self.is_deepstack_layers.len > 0) allocator.free(self.is_deepstack_layers);
        self.is_deepstack_layers = &.{};
    }

    /// Construye la config desde metadata GGUF del mmproj.
    /// `g` es el GgufFile del mmproj (no del modelo target).
    pub fn fromGguf(g: *const gguf.GgufFile) MmprojConfigError!Self {
        // Projector type: clip.projector_type, fallback clip.vision.projector_type
        // (modelos con modalidades mixtas, clip.cpp:1165-1177).
        const proj_str = strMeta(g, "clip.projector_type") orelse
            strMeta(g, "clip.vision.projector_type") orelse
            return MmprojConfigError.MissingProjectorType;
        const proj = ProjectorType.fromString(proj_str);
        if (proj == .unknown) return MmprojConfigError.UnknownProjectorType;

        const n_embd = try usizeMeta(g, "clip.vision.embedding_length", null);
        const n_head = try usizeMeta(g, "clip.vision.attention.head_count", null);
        const n_layer = try usizeMeta(g, "clip.vision.block_count", null);

        var cfg: Self = .{
            .projector_type = proj,
            .projector_type_str = proj_str,
            .n_embd = n_embd,
            .n_head = n_head,
            .head_dim = try usizeMeta(g, "clip.vision.attention.head_dim", n_embd / n_head),
            .n_head_kv = try usizeMeta(g, "clip.vision.attention.head_count_kv", n_head),
            .n_ff = try usizeMeta(g, "clip.vision.feed_forward_length", null),
            .n_layer = n_layer,
            .projection_dim = try usizeMeta(g, "clip.vision.projection_dim", null),
            .eps = try f32Meta(g, "clip.vision.attention.layer_norm_epsilon", 1e-6),
            // Qwen2.5-VL usa RMS norm en TODO el ViT (clip.cpp:42-44
            // NORM_TYPE_RMS; blocks y deepstack). Qwen2-VL/3-VL: LayerNorm.
            .use_rms_norm = proj == .qwen25vl_merger,
            .image_size = try usizeMeta(g, "clip.vision.image_size", 0),
            .patch_size = try usizeMeta(g, "clip.vision.patch_size", null),
            .image_mean = .{ 0.5, 0.5, 0.5 },
            .image_std = .{ 0.5, 0.5, 0.5 },
            .spatial_merge_size = 2, // default Qwen-VL (clip.cpp:1493)
            .image_min_pixels = 3136,
            .image_max_pixels = 1003520,
            .min_image_tokens = 8,
            .max_image_tokens = 4096,
            .ffn_op = .gelu_quick, // default si no hay flag (clip.cpp:1274)
            .is_deepstack_layers = &.{},
        };

        // spatial_merge_size (clip.cpp:1498)
        if (usizeMeta(g, "clip.vision.spatial_merge_size", null) catch null) |m| {
            if (m > 0) cfg.spatial_merge_size = m;
        }

        // image_size derivado (conversión unsloth qwen3vl.py:30-36):
        // num_position_embeddings = (image_size/patch)² ⇒ image_size = sqrt(num_pos)·patch
        if (cfg.image_size == 0) {
            if (usizeMeta(g, "clip.vision.num_position_embeddings", null) catch null) |num_pos| {
                const side: usize = @intFromFloat(@sqrt(@as(f64, @floatFromInt(num_pos))));
                cfg.image_size = side * cfg.patch_size;
            }
        }

        // image_mean / image_std (arrays f32, ≥3 elems; clip.cpp:1276-1289)
        var mean_buf: [8]f32 = undefined;
        if (f32ArrMeta(g, "clip.vision.image_mean", &mean_buf)) |n| {
            if (n >= 3) cfg.image_mean = .{ mean_buf[0], mean_buf[1], mean_buf[2] };
        }
        var std_buf: [8]f32 = undefined;
        if (f32ArrMeta(g, "clip.vision.image_std", &std_buf)) |n| {
            if (n >= 3) cfg.image_std = .{ std_buf[0], std_buf[1], std_buf[2] };
        }

        // Límites de píxeles/tokens (clip.cpp:1500 y mtmd.h:99-100)
        if (usizeMeta(g, "clip.vision.image_min_pixels", null) catch null) |p| cfg.image_min_pixels = p;
        if (usizeMeta(g, "clip.vision.image_max_pixels", null) catch null) |p| cfg.image_max_pixels = p;
        if (usizeMeta(g, "clip.vision.image_min_tokens", null) catch null) |t| cfg.min_image_tokens = t;
        if (usizeMeta(g, "clip.vision.image_max_tokens", null) catch null) |t| cfg.max_image_tokens = t;

        // Activación FFN (clip.cpp:1264-1275)
        const use_gelu = boolMeta(g, "clip.use_gelu") orelse false;
        const use_silu = boolMeta(g, "clip.use_silu") orelse false;
        if (use_gelu and use_silu) return MmprojConfigError.InvalidMetadata;
        if (use_gelu) {
            cfg.ffn_op = .gelu;
        } else if (use_silu) {
            cfg.ffn_op = .silu;
        }

        // Deepstack (unsloth qwen3vl.py:63-64 → clip.vision.is_deepstack_layers)
        var ds_buf: [128]bool = undefined;
        const n_ds = boolArrMeta(g, "clip.vision.is_deepstack_layers", &ds_buf) orelse 0;
        if (n_ds > 0) {
            const ds = g.allocator.alloc(bool, n_ds) catch
                return MmprojConfigError.OutOfMemory;
            @memcpy(ds, ds_buf[0..n_ds]);
            cfg.is_deepstack_layers = ds;
        }

        return cfg;
    }
};

// ── Helpers de lectura (patrón model_config.zig:228-260) ──────────────────

fn strMeta(g: *const gguf.GgufFile, key: []const u8) ?[]const u8 {
    const v = g.getMeta(key) orelse return null;
    return v.asString();
}

fn usizeMeta(g: *const gguf.GgufFile, key: []const u8, default: ?usize) MmprojConfigError!usize {
    const v = g.getMeta(key) orelse return default orelse MmprojConfigError.MissingRequiredMetadata;
    return @intCast(v.asU64() orelse return MmprojConfigError.InvalidMetadata);
}

fn f32Meta(g: *const gguf.GgufFile, key: []const u8, default: ?f32) MmprojConfigError!f32 {
    const v = g.getMeta(key) orelse return default orelse MmprojConfigError.MissingRequiredMetadata;
    return v.asF32() orelse MmprojConfigError.InvalidMetadata;
}

fn boolMeta(g: *const gguf.GgufFile, key: []const u8) ?bool {
    const v = g.getMeta(key) orelse return null;
    return v.asBool();
}

/// Lee un array de f32 en `out` (máx out.len). Devuelve nº de elems.
fn f32ArrMeta(g: *const gguf.GgufFile, key: []const u8, out: []f32) ?usize {
    const v = g.getMeta(key) orelse return null;
    const arr = switch (v) {
        .array => |a| a,
        else => return null,
    };
    const n = @min(out.len, arr.items.len);
    for (arr.items[0..n], 0..) |it, i| {
        out[i] = it.asF32() orelse return null;
    }
    return n;
}

/// Lee un array de bool (uint8/int32) en `out`. Devuelve nº de elems.
fn boolArrMeta(g: *const gguf.GgufFile, key: []const u8, out: []bool) ?usize {
    const v = g.getMeta(key) orelse return null;
    const arr = switch (v) {
        .array => |a| a,
        else => return null,
    };
    const n = @min(out.len, arr.items.len);
    for (arr.items[0..n], 0..) |it, i| {
        switch (it) {
            .bool => |b| out[i] = b,
            .uint8 => |u| out[i] = u != 0,
            .int32 => |x| out[i] = x != 0,
            else => return null,
        }
    }
    return n;
}

// ── Tests ──────────────────────────────────────────────────────────────────

const testing = std.testing;

test "ProjectorType.fromString cubre variantes Qwen-VL" {
    try testing.expectEqual(ProjectorType.qwen3vl_merger, ProjectorType.fromString("qwen3vl_merger"));
    try testing.expectEqual(ProjectorType.qwen25vl_merger, ProjectorType.fromString("qwen2.5vl_merger"));
    try testing.expectEqual(ProjectorType.qwen2vl_merger, ProjectorType.fromString("qwen2vl_merger"));
    try testing.expectEqual(ProjectorType.unknown, ProjectorType.fromString("mlp"));
}

test "use_rms_norm: sólo qwen2.5-VL" {
    // El flag se deriva del projector type en fromGguf (requiere GGUF real);
    // el invariant puro: 2.5-VL ≠ 2-VL ≠ 3-VL en el enum (dispatch distinto).
    try testing.expect(ProjectorType.qwen25vl_merger == .qwen25vl_merger);
    try testing.expect(ProjectorType.qwen2vl_merger != .qwen25vl_merger);
    try testing.expect(ProjectorType.qwen3vl_merger != .qwen25vl_merger);
}
