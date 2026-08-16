//! Hybrid Transformer Layer — unifica SSM (Gated DeltaNet) y Attention
//! para qwen35. Dispatch por capa usando ModelConfig.isFullAttentionLayer.
const std = @import("std");
const Tensor = @import("core").Tensor;
const matmul = @import("matmul");
const QuantWeight = @import("quant_weight").QuantWeight;
const gguf = @import("gguf");
const norm = @import("norm");
const ffn = @import("ffn");
const rope_mod = @import("rope");
const gqa_mod = @import("gqa");
const model_config = @import("model_config");
const AttentionLayer = @import("hybrid_attn").AttentionLayer;
const SsmLayer = @import("ssm").SsmLayer;
const paged = @import("paged_attention");

pub const HybridLayerError = error{
    WeightFileNotFound,
    ShapeMismatch,
    KvCacheNotSet,
};

/// Parámetros derivados de ModelConfig para una capa híbrida
pub const HybridLayerParams = struct {
    n_embd: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    n_rot: usize,
    rope_sections: [4]usize,
    rope_freq_base: f32,
    rms_eps: f32,
    max_seq_len: usize,

    // SSM params
    d_inner: usize,
    d_state: usize,
    dt_rank: usize,
    n_group: usize,
    d_conv: usize,

    // FFN
    intermediate_dim: usize,

    pub fn fromModelConfig(cfg: model_config.ModelConfig, max_seq_len: usize) HybridLayerParams {
        const ssm_d_inner = if (cfg.ssm_inner_size > 0) cfg.ssm_inner_size else cfg.embedding_length * 3;
        const ssm_d_state = if (cfg.ssm_state_size > 0) cfg.ssm_state_size else 128;
        const ssm_dt_rank = if (cfg.ssm_time_step_rank > 0) cfg.ssm_time_step_rank else 32;
        const ssm_n_group = if (cfg.ssm_group_count > 0) cfg.ssm_group_count else 16;
        const ssm_d_conv = if (cfg.ssm_conv_kernel > 0) cfg.ssm_conv_kernel else 4;

        return .{
            .n_embd = cfg.embedding_length,
            .n_head = cfg.head_count,
            .n_kv_head = cfg.head_count_kv,
            .head_dim = if (cfg.head_dim > 0) cfg.head_dim else cfg.embedding_length / cfg.head_count,
            .n_rot = cfg.rope_dimension_count,
            .rope_sections = cfg.rope_sections,
            .rope_freq_base = cfg.rope_freq_base,
            .rms_eps = cfg.layer_norm_rms_epsilon,
            .max_seq_len = max_seq_len,
            .d_inner = ssm_d_inner,
            .d_state = ssm_d_state,
            .dt_rank = ssm_dt_rank,
            .n_group = ssm_n_group,
            .d_conv = ssm_d_conv,
            .intermediate_dim = if (cfg.feed_forward_length > 0) cfg.feed_forward_length else cfg.embedding_length * 3,
        };
    }
};

pub const HybridLayer = struct {
    allocator: std.mem.Allocator,
    layer_idx: usize,
    params: HybridLayerParams,
    matmul_engine: matmul.MatmulEngine,
    is_attention: bool,

    // Pesos comunes (normalización)
    attn_norm: Tensor(f32),      // [n_embd]
    attn_post_norm: Tensor(f32), // [n_embd]

    // FFN weights (QuantWeight)
    w_gate: QuantWeight,
    w_up: QuantWeight,
    w_down: QuantWeight,

    // Scratch f16 para FFN
    scratch_gate: []f32,
    scratch_up: []f32,
    scratch_down: []f32,

    // Sub-layer específica
    attn_layer: ?AttentionLayer = null,
    ssm_layer: ?SsmLayer = null,

    // KV-Cache paginado compartido (solo usado por capas de atención)
    paged_kv: ?*paged.PagedKVCache = null,
    block_table: ?*paged.BlockTable = null,
    paged_gpu: ?*paged.PagedAttentionGpu = null,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        layer_idx: usize,
        params: HybridLayerParams,
        is_attention: bool,
        backend: matmul.Backend,
        paged_kv: ?*paged.PagedKVCache,
        block_table: ?*paged.BlockTable,
        paged_gpu: ?*paged.PagedAttentionGpu,
    ) !Self {
        var engine = try matmul.MatmulEngine.init(allocator, backend, .f32);
        errdefer engine.deinit();

        const scratch_gate = try allocator.alloc(f32, params.intermediate_dim * params.n_embd);
        errdefer allocator.free(scratch_gate);
        const scratch_up = try allocator.alloc(f32, params.intermediate_dim * params.n_embd);
        errdefer allocator.free(scratch_up);
        const scratch_down = try allocator.alloc(f32, params.n_embd * params.intermediate_dim);
        errdefer allocator.free(scratch_down);

        var attn_norm = try Tensor(f32).alloc(allocator, &.{params.n_embd});
        errdefer attn_norm.deinit();
        var attn_post_norm = try Tensor(f32).alloc(allocator, &.{params.n_embd});
        errdefer attn_post_norm.deinit();

        var self = Self{
            .allocator = allocator,
            .layer_idx = layer_idx,
            .params = params,
            .matmul_engine = engine,
            .is_attention = is_attention,
            .attn_norm = attn_norm,
            .attn_post_norm = attn_post_norm,
            .w_gate = undefined,
            .w_up = undefined,
            .w_down = undefined,
            .scratch_gate = scratch_gate,
            .scratch_up = scratch_up,
            .scratch_down = scratch_down,
            .attn_layer = null,
            .ssm_layer = null,
            .paged_kv = paged_kv,
            .block_table = block_table,
            .paged_gpu = paged_gpu,
        };

        if (is_attention) {
            const attn_params = @import("hybrid_attn").HybridAttnParams{
                .n_embd = params.n_embd,
                .n_head = params.n_head,
                .n_kv_head = params.n_kv_head,
                .head_dim = params.head_dim,
                .n_rot = params.n_rot,
                .rope_sections = params.rope_sections,
                .rope_freq_base = params.rope_freq_base,
                .rms_eps = params.rms_eps,
                .max_seq_len = params.max_seq_len,
            };
            self.attn_layer = try AttentionLayer.init(
                allocator,
                layer_idx,
                attn_params,
                backend,
                paged_kv orelse return HybridLayerError.KvCacheNotSet,
                block_table orelse return HybridLayerError.KvCacheNotSet,
                paged_gpu,
            );
            errdefer if (self.attn_layer) |l| l.deinit();
        } else {
            const ssm_params = @import("ssm").SsmParams{
                .n_embd = params.n_embd,
                .d_inner = params.d_inner,
                .d_state = params.d_state,
                .dt_rank = params.dt_rank,
                .n_group = params.n_group,
                .d_conv = params.d_conv,
                .rms_eps = params.rms_eps,
            };
            self.ssm_layer = try SsmLayer.init(allocator, layer_idx, ssm_params, backend);
            errdefer if (self.ssm_layer) |l| l.deinit();
        }

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.matmul_engine.deinit();
        self.allocator.free(self.scratch_gate);
        self.allocator.free(self.scratch_up);
        self.allocator.free(self.scratch_down);
        self.attn_norm.deinit();
        self.attn_post_norm.deinit();
        if (self.attn_layer) |*l| l.deinit();
        if (self.ssm_layer) |*l| l.deinit();
    }

    pub fn resetState(self: *Self) void {
        if (self.attn_layer) |l| l.resetState();
        if (self.ssm_layer) |l| l.resetState();
    }

    /// Carga pesos desde GGUF (nombres qwen35)
    pub fn loadWeightsFromGguf(self: *Self, g: *const gguf.GgufFile) !void {
        const prefix = try std.fmt.allocPrint(self.allocator, "blk.{d}.", .{self.layer_idx});
        defer self.allocator.free(prefix);

        // Norm weights
        self.attn_norm.deinit();
        self.attn_norm = try loadGgufF32(self.allocator, g, prefix, "attn_norm.weight");
        self.attn_post_norm.deinit();
        self.attn_post_norm = try loadGgufF32(self.allocator, g, prefix, "post_attention_norm.weight");

        // FFN weights
        self.w_gate = try loadQuantWeight(g, prefix, "ffn_gate.weight");
        self.w_up = try loadQuantWeight(g, prefix, "ffn_up.weight");
        self.w_down = try loadQuantWeight(g, prefix, "ffn_down.weight");

        if (self.is_attention) {
            if (self.attn_layer) |*l| try l.loadWeightsFromGguf(g);
        } else {
            if (self.ssm_layer) |*l| try l.loadWeightsFromGguf(g);
        }
    }

    /// Forward del bloque híbrido:
    /// x → attn_norm → (SSM | Attention) → +residual → attn_post_norm → FFN → +residual → out
    pub fn forward(self: *Self, x: Tensor(f32), out: *Tensor(f32), start_pos: usize, n: usize) !void {
        const p = self.params;
        const N = n;

        // === 1. Pre-Attention/SSM RMSNorm ===
        var norm_buf = try Tensor(f32).alloc(self.allocator, &.{ N, p.n_embd });
        defer norm_buf.deinit();
        norm.rmsNorm(f32, f32, x, self.attn_norm, p.rms_eps, &norm_buf);

        // === 2. SSM o Attention ===
        var mixer_out = try Tensor(f32).alloc(self.allocator, &.{ N, p.n_embd });
        defer mixer_out.deinit();

        if (self.is_attention) {
            if (self.attn_layer) |*l| {
                try l.forward(norm_buf, &mixer_out, start_pos, N);
            }
        } else {
            if (self.ssm_layer) |*l| {
                try l.forward(norm_buf, &mixer_out, N);
            }
        }

        // === 3. Residual connection (Mixer) ===
        for (out.data, mixer_out.data, x.data) |*o, m, xv| {
            o.* = m + xv;
        }

        // === 4. Post-Attention/SSM RMSNorm ===
        var post_norm_buf = try Tensor(f32).alloc(self.allocator, &.{ N, p.n_embd });
        defer post_norm_buf.deinit();
        norm.rmsNorm(f32, f32, out.*, self.attn_post_norm, p.rms_eps, &post_norm_buf);

        // === 5. FFN SwiGLU ===
        self.w_gate.dequantToF32Transposed(self.scratch_gate);
        var w_gate_shape = [_]usize{ p.intermediate_dim, p.n_embd };
        var w_gate_strides = [_]usize{ p.n_embd, 1 };
        const w_gate32 = Tensor(f32){
            .data = self.scratch_gate,
            .shape = &w_gate_shape,
            .strides = &w_gate_strides,
            .offset = 0,
            .allocator = null,
            .owns_data = false,
        };

        self.w_up.dequantToF32Transposed(self.scratch_up);
        var w_up_shape = [_]usize{ p.intermediate_dim, p.n_embd };
        var w_up_strides = [_]usize{ p.n_embd, 1 };
        const w_up32 = Tensor(f32){
            .data = self.scratch_up,
            .shape = &w_up_shape,
            .strides = &w_up_strides,
            .offset = 0,
            .allocator = null,
            .owns_data = false,
        };

        self.w_down.dequantToF32Transposed(self.scratch_down);
        var w_down_shape = [_]usize{ p.n_embd, p.intermediate_dim };
        var w_down_strides = [_]usize{ p.intermediate_dim, 1 };
        const w_down32 = Tensor(f32){
            .data = self.scratch_down,
            .shape = &w_down_shape,
            .strides = &w_down_strides,
            .offset = 0,
            .allocator = null,
            .owns_data = false,
        };

        var gate_buf = try Tensor(f32).alloc(self.allocator, &.{ N, p.intermediate_dim });
        defer gate_buf.deinit();
        var up_buf = try Tensor(f32).alloc(self.allocator, &.{ N, p.intermediate_dim });
        defer up_buf.deinit();
        var ffn_out = try Tensor(f32).alloc(self.allocator, &.{ N, p.n_embd });
        defer ffn_out.deinit();

        const post_norm_2d = try post_norm_buf.reshape(&[_]usize{ N, p.n_embd });
        defer { if (post_norm_2d.allocator) |a| { a.free(post_norm_2d.shape); a.free(post_norm_2d.strides); } }

        try ffn.swiGluForward(
            &self.matmul_engine, f32,
            post_norm_2d,
            w_gate32, w_up32, w_down32,
            &gate_buf, &up_buf, &ffn_out,
        );

        // === 6. Residual connection (FFN) ===
        for (out.data, ffn_out.data) |*o, f| {
            o.* += f;
        }
    }
};

fn loadQuantWeight(g: *const gguf.GgufFile, prefix: []const u8, name: []const u8) !QuantWeight {
    const full = try std.fmt.allocPrint(std.heap.page_allocator, "{s}{s}", .{ prefix, name });
    defer std.heap.page_allocator.free(full);
    const info = g.getTensor(full) orelse return HybridLayerError.WeightFileNotFound;
    return QuantWeight.init(info, g.tensorData(info));
}

fn loadGgufF32(
    allocator: std.mem.Allocator,
    g: *const gguf.GgufFile,
    prefix: []const u8,
    name: []const u8,
) !Tensor(f32) {
    const full = try std.fmt.allocPrint(allocator, "{s}{s}", .{ prefix, name });
    defer allocator.free(full);
    const info = g.getTensor(full) orelse return HybridLayerError.WeightFileNotFound;
    const numel: usize = @intCast(info.numel());

    const f32buf = try allocator.alloc(f32, numel);
    defer allocator.free(f32buf);
    try gguf.dequantTensor(info, g.tensorData(info), f32buf);

    var out_dim: usize = 1;
    var in_dim: usize = 1;
    var tensor: Tensor(f32) = undefined;
    if (info.n_dims >= 2) {
        // GGUF guarda [in, out]; la capa espera [out, in] → transponer.
        in_dim = @intCast(info.dims[0]);
        out_dim = @intCast(info.dims[1]);
        tensor = try Tensor(f32).initUninitialized(allocator, &.{ out_dim, in_dim });
        for (0..in_dim) |r| {
            for (0..out_dim) |c| {
                tensor.data[c * in_dim + r] = f32buf[r + c * in_dim];
            }
        }
    } else {
        tensor = try Tensor(f32).initUninitialized(allocator, &.{numel});
        @memcpy(tensor.data, f32buf);
    }
    return tensor;
}