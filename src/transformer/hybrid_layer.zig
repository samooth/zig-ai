//! Hybrid Transformer Layer — unifica SSM (Gated DeltaNet) y Attention
//! para qwen35. Dispatch por capa usando ModelConfig.isFullAttentionLayer.
//! También soporta LFM2 (ShortConv + Attention).
const std = @import("std");
const Tensor = @import("core").Tensor;
const matmul = @import("matmul");
const cublas = @import("cublas");
const cudaz = @import("cudaz");
const layer_kernels = @import("layer_kernels");
const QuantWeight = @import("quant_weight").QuantWeight;
const gguf = @import("gguf");
const norm = @import("norm");
const ffn = @import("ffn");
const rope_mod = @import("rope");
const gqa_mod = @import("gqa");
const model_config = @import("model_config");
const AttentionLayer = @import("hybrid_attn").AttentionLayer;
const SsmLayer = @import("ssm").SsmLayer;
const ShortConvLayer = @import("short_conv").ShortConvLayer;
const paged = @import("paged_attention");
const gpu_weight_pool = @import("gpu_weight_pool");
const moe_layer_mod = @import("moe_layer"); // lane-e E5 (sección FFN)
const debugz = @import("debug");
const activation_pool = @import("activation_pool");
const build_options = @import("build_options");

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

    // SSM params (Qwen3.5)
    d_inner: usize,
    d_state: usize,
    dt_rank: usize,
    n_group: usize,
    d_conv: usize,

    // FFN
    intermediate_dim: usize,

    // LFM2 params
    is_lfm2: bool = false,
    shortconv_l_cache: usize = 0,
    no_gate: bool = false, // LFM2 attention: separate Q (not fused Q+G)
    use_mrope: bool = true, // LFM2: use standard RoPE instead of MRoPE

    // K2-Horizon params (IFM): grouped RMSNorm + RoPE clásico (yarn) —
    // el resto de la geometría es GQA clásica.
    is_k2_horizon: bool = false,
    n_norm_groups: usize = 1,

    // RoPE scaling
    rope_scaling: rope_mod.RopeScaling = .{},

    // RLT SWA: sliding window attention per-layer (null = full context)
    swa_window: ?usize = null,

    pub fn fromModelConfig(cfg: model_config.ModelConfig, max_seq_len: usize) HybridLayerParams {
        const ssm_d_inner = if (cfg.ssm_inner_size > 0) cfg.ssm_inner_size else cfg.embedding_length * 3;
        const ssm_d_state = if (cfg.ssm_state_size > 0) cfg.ssm_state_size else 128;
        const ssm_dt_rank = if (cfg.ssm_time_step_rank > 0) cfg.ssm_time_step_rank else 32;
        const ssm_n_group = if (cfg.ssm_group_count > 0) cfg.ssm_group_count else 16;
        const ssm_d_conv = if (cfg.ssm_conv_kernel > 0) cfg.ssm_conv_kernel else 4;

        const is_lfm2 = std.mem.eql(u8, cfg.architecture, "lfm2");
        const is_k2 = cfg.is_k2_horizon;
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
            .is_lfm2 = is_lfm2,
            .shortconv_l_cache = cfg.shortconv_l_cache,
            .no_gate = is_lfm2,
            .use_mrope = !is_lfm2, // LFM2 uses standard RoPE, Qwen3.5 uses MRoPE
            .is_k2_horizon = is_k2,
            .n_norm_groups = if (cfg.n_norm_groups == 0) 1 else cfg.n_norm_groups,
            .rope_scaling = .{
                .scaling_type = switch (cfg.rope_scaling_type) {
                    .none => .none,
                    .linear => .linear,
                    .yarn => .yarn,
                },
                .factor = cfg.rope_scaling_factor,
                .orig_ctx = cfg.rope_scaling_orig_ctx,
                .attn_factor = cfg.rope_attn_factor,
                .yarn_ext_factor = cfg.yarn_ext_factor,
                .yarn_beta_fast = cfg.yarn_beta_fast,
                .yarn_beta_slow = cfg.yarn_beta_slow,
            },
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
    attn_norm: Tensor(f32), // [n_embd]
    attn_post_norm: ?Tensor(f32) = null, // [n_embd] - optional for LFM2 (uses ffn_norm)

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
    short_conv_layer: ?ShortConvLayer = null,

    // MoE offload (lane-e): null ⇒ FFN densa bit-idéntica al path original.
    moe: ?*moe_layer_mod.MoeLayer = null,

    // KV-Cache paginado compartido (solo usado por capas de atención)
    paged_kv: ?*paged.PagedKVCache = null,
    block_table: ?*paged.BlockTable = null,
    paged_gpu: ?*paged.PagedAttentionGpu = null,

    // Track weight load state for layer streaming (AirLLM)
    weights_loaded: bool = false,

    // Buffers GPU para el forward residente (Path B).
    gpu: ?HybridGpu = null,

    // Activation pool (CPU path): reutiliza buffers de activaciones intermedias
    act_pool: activation_pool.ActivationPool = undefined,

    // ─── RLT: Recurrent Looped Transformer feedback ────────────────────
    // Merge: u_t = Merge(e_t, s_{t-1}) — gated residual feedback from
    // previous token's final decoder output into current token's input.
    // Opt-in: alpha > 0 activates the merge; without GGUF weights → skip.
    rlt_alpha: f32 = 0.0,
    rlt_w_gate: ?Tensor(f32) = null, // [d, 2d] — projects [x; RMSNorm(prev)]
    rlt_w_state: ?Tensor(f32) = null, // [d, d] — projects RMSNorm(prev)
    rlt_prev_state: ?[]f32 = null, // [n_embd] — s_{t-1}, persists between tokens
    rlt_scratch_gate: ?[]f32 = null, // [n_embd] — scratch for gate computation
    rlt_scratch_state: ?[]f32 = null, // [n_embd] — scratch for state projection

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

        // Scratch FFN (pesos f32 dequantizados) LAZY: se allocan en
        // loadWeightsFromGguf la primera vez que se cargan pesos. Con eager
        // aquí, un modelo grande (27B: 3×5120×17408×4B ≈ 1GB por capa ×65)
        // agota la RAM del host antes de llegar al streaming de capas.
        // unloadWeights() los libera y esta misma ruta los re-alloca.

        var attn_norm = try Tensor(f32).alloc(allocator, &.{params.n_embd});
        errdefer attn_norm.deinit();

        // LFM2 uses ffn_norm instead of post_attention_norm for both attention and shortconv layers
        // For Qwen3.5, allocate post_attention_norm; for LFM2, we'll load ffn_norm into this slot
        var attn_post_norm: ?Tensor(f32) = null;
        if (!params.is_lfm2) {
            attn_post_norm = try Tensor(f32).alloc(allocator, &.{params.n_embd});
            errdefer if (attn_post_norm) |t| t.deinit();
        }

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
            .scratch_gate = &[_]f32{},
            .scratch_up = &[_]f32{},
            .scratch_down = &[_]f32{},
            .attn_layer = null,
            .ssm_layer = null,
            .short_conv_layer = null,
            .paged_kv = paged_kv,
            .block_table = block_table,
            .paged_gpu = paged_gpu,
            .act_pool = activation_pool.ActivationPool.init(allocator, 256 * 1024 * 1024),
        };

        if (params.is_lfm2) {
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
                    .no_gate = params.no_gate,
                    .use_mrope = params.use_mrope,
                    .rope_scaling = params.rope_scaling,
                    .swa_window = params.swa_window,
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
                // LFM2 ShortConv layer
                const sc_params = @import("short_conv").ShortConvParams{
                    .n_embd = params.n_embd,
                    .conv_dim = params.n_embd,
                    .l_cache = params.shortconv_l_cache,
                    .rms_eps = params.rms_eps,
                };
                self.short_conv_layer = try ShortConvLayer.init(allocator, layer_idx, sc_params, backend);
                errdefer if (self.short_conv_layer) |l| l.deinit();
            }
        } else {
            // Qwen3.5 hybrid (SSM + Attention) — también K2-Horizon (toda
            // capa es atención; RoPE clásico, sin MRoPE, sin gate Q+G).
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
                    .no_gate = params.is_k2_horizon, // K2: proyecciones Q/K/V separadas
                    .use_mrope = params.is_k2_horizon == false and params.use_mrope, // K2: RoPE clásico
                    .has_softplus_gate = params.is_k2_horizon, // detectado por tensor-presence en loadWeights
                    .rope_scaling = params.rope_scaling,
                    .swa_window = params.swa_window,
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
        }

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.matmul_engine.deinit();
        self.allocator.free(self.scratch_gate);
        self.allocator.free(self.scratch_up);
        self.allocator.free(self.scratch_down);
        self.attn_norm.deinit();
        if (self.attn_post_norm) |*t| t.deinit();
        self.act_pool.deinit();
        if (self.attn_layer) |*l| l.deinit();
        if (self.ssm_layer) |*l| l.deinit();
        if (self.short_conv_layer) |*l| l.deinit();
        // RLT feedback cleanup
        self.deinitRltFeedback();
    }

    fn deinitRltFeedback(self: *Self) void {
        if (self.rlt_w_gate) |*t| t.deinit();
        if (self.rlt_w_state) |*t| t.deinit();
        if (self.rlt_prev_state) |s| self.allocator.free(s);
        if (self.rlt_scratch_gate) |s| self.allocator.free(s);
        if (self.rlt_scratch_state) |s| self.allocator.free(s);
        self.rlt_w_gate = null;
        self.rlt_w_state = null;
        self.rlt_prev_state = null;
        self.rlt_scratch_gate = null;
        self.rlt_scratch_state = null;
    }

    pub fn resetState(self: *Self) void {
        // P0-1: |*l| — captura por puntero (a través de self *Self); la
        // captura por valor |l| es const y resetState sub-layers exige
        // mutabilidad (latente: fn nunca referenciada hasta resetStateGpu).
        if (self.attn_layer) |*l| l.resetState();
        if (self.ssm_layer) |*l| l.resetState();
        if (self.short_conv_layer) |*l| l.resetState();
        // RLT: zero feedback state on sequence reset
        if (self.rlt_prev_state) |state| {
            @memset(state, 0);
        }
    }

    /// P0-1 (dev RLT): reset COMPLETO host+device del estado recurrente para
    /// PPL sliding-window — cada chunk reconstruye SSM/conv state desde
    /// ctx_start (semántica llama.cpp). resetState() solo toca host.
    pub fn resetStateGpu(self: *Self) void {
        self.resetState();
        if (self.ssm_layer) |*l| l.zeroGpuState();
    }

    /// Carga pesos desde GGUF (nombres qwen35 o lfm2). Si weights_loaded es false,
    /// re-alloca los scratch buffers antes de dequantizar.
    /// T1 fix regresión CPU (bisect D 02:5x): consumidores f32 del scratch
    /// FFN garantizan alloc+dequant bajo demanda.
    fn ensureFfnScratchFilled(self: *HybridLayer) !void {
        const p = self.params;
        // 9.4.1 (lane-b) fix decode CPU 5×: el dequant corría en CADA
        // forward (~226 MB F16→f32 por token en el 350M ⇒ 76-82ms/capa,
        // medido [hybrid] ffn=76414us LFM2-350M ReleaseFast). El mixer
        // (attn/ssm/shortconv) SÍ tenía guard (ensureF32Scratch) — el FFN
        // no. Guard simétrico: dequant UNA vez; re-alloc si unloadWeights
        // liberó los scratch (len==0).
        if (self.scratch_gate.len == 0) {
            self.scratch_gate = try self.allocator.alloc(f32, p.intermediate_dim * p.n_embd);
            self.scratch_up = try self.allocator.alloc(f32, p.intermediate_dim * p.n_embd);
            self.scratch_down = try self.allocator.alloc(f32, p.n_embd * p.intermediate_dim);
            self.w_gate.dequantToF32Transposed(self.scratch_gate);
            self.w_up.dequantToF32Transposed(self.scratch_up);
            self.w_down.dequantToF32Transposed(self.scratch_down);
        }
    }

    pub fn loadWeightsFromGguf(self: *HybridLayer, g: *const gguf.GgufFile, sidecar: ?*const gguf.GgufFile) !void {
        // F/C fix 700 (known-issue v2, 2ª parte): la metadata puede mentir
        // sobre el FFN (loggenix qwen3moe: feed_forward_length=1536 pero los
        // bancos reales ffn_gate_exps son [512,768,E] — el FFN del experto
        // es 768). El qgemm denso experto-0 corría con N=1536 sobre un
        // banco de 768 filas ⇒ OOB ⇒ CUDA 700 en el primer forward del
        // FFN (recogido por el kernel siguiente — addInplace/gather).
        // La GEOMETRÍA del tensor manda: parcheamos intermediate_dim antes
        // de que scratch/shapes/forward lo consuman.
        {
            var nbuf: [96]u8 = undefined;
            const exps_name = std.fmt.bufPrint(&nbuf, "blk.{d}.ffn_gate_exps.weight", .{self.layer_idx}) catch unreachable;
            if (g.getTensor(exps_name)) |ti| {
                if (ti.n_dims >= 2) {
                    const ff_real: usize = @intCast(ti.dims[1]);
                    if (ff_real > 0 and ff_real != self.params.intermediate_dim) {
                        const ff_meta = self.params.intermediate_dim;
                        self.params.intermediate_dim = ff_real;
                        debugz.dbg.printLevel(.info, "[hybrid] capa {d}: ffn_gate_exps out={d} ⇒ intermediate_dim {d}→{d} (metadata decía {d})\n", .{ self.layer_idx, ff_real, ff_meta, ff_real, ff_meta });
                    }
                }
            }
        }
        const prefix = try std.fmt.allocPrint(self.allocator, "blk.{d}.", .{self.layer_idx});
        defer self.allocator.free(prefix);

        // Norm weights (FIX dangling: reset antes del try para que un fallo
        // deje un tensor sin ownership, no uno liberado).
        self.attn_norm.deinit();
        self.attn_norm = .{ .data = &.{}, .shape = &.{}, .strides = &.{}, .offset = 0, .allocator = null, .owns_data = false };
        self.attn_norm = try loadGgufF32(self.allocator, g, prefix, "attn_norm.weight");

        // Post-attn norm: el nombre depende de la familia. FIX dangling:
        // deinit la vieja y NULL el optional ANTES del try — si el load
        // falla, deinit() no ve un tensor liberado (double-free evitado).
        var post_norm_name: [64]u8 = undefined;
        const post_norm: []const u8 = blk: {
            if (self.params.is_lfm2 or self.params.is_k2_horizon) break :blk "ffn_norm.weight";
            // ¿Existe post_attention_norm (Qwen3.5 híbrido)? Si no (MoE
            // denso: qwen3moe/qwen2moe), la norm post-attn es ffn_norm.
            const full = std.fmt.bufPrint(&post_norm_name, "{s}post_attention_norm.weight", .{prefix}) catch break :blk "ffn_norm.weight";
            if (g.getTensor(full) != null) break :blk "post_attention_norm.weight";
            break :blk "ffn_norm.weight";
        };
        if (self.attn_post_norm) |*t| t.deinit();
        self.attn_post_norm = null;
        self.attn_post_norm = try loadGgufF32(self.allocator, g, prefix, post_norm);

        // FFN weights. MoE denso (qwen3moe/qwen2moe): no existe FFN densa
        // por capa — los pesos viven en tensores 3-D `ffn_*_exps` [in,out,E].
        // Cargamos el EXPERTO 0 como FFN de la capa (info sintetizada 2-D +
        // bytes del slice) para que el forward mecánico sea válido; el MoE
        // real lo enruta MoeLayer (lane-e) sobre los mismos bancos.
        self.w_gate = try loadQuantWeightOrExpert0(g, prefix, "ffn_gate");
        self.w_up = try loadQuantWeightOrExpert0(g, prefix, "ffn_up");
        self.w_down = try loadQuantWeightOrExpert0(g, prefix, "ffn_down");

        // T1 VRAM-spec: dequantizar FFN UNA vez SOLO si el camino cuantizado
        // no cubre los tres pesos. Con kernel qgemm los scratch f32 no se leen
        // nunca y con streaming el host acumulaba ~1GB × capas densas (24GB
        // anónimos medidos en Q8_K_XL): el streamer solo expulsa por presión
        // de VRAM, y el device ya no se presiona tras el routing qgemm.
        // La ruta CPU legacy (forward()) requiere NOQ4=1 con modelos
        // todo-cuantizados — misma condición que el GPU fallback.
        const need_ffn_f32 = !(layer_kernels.quantPath() and !debugz.dbg.no_q4_ffn and
            SsmLayer.qgemmTypeFor(self.w_gate.dtype()) != null and
            SsmLayer.qgemmTypeFor(self.w_up.dtype()) != null and
            SsmLayer.qgemmTypeFor(self.w_down.dtype()) != null);
        if (need_ffn_f32) {
            if (self.weights_loaded == false and self.scratch_gate.len == 0) {
                const p = self.params;
                self.scratch_gate = try self.allocator.alloc(f32, p.intermediate_dim * p.n_embd);
                self.scratch_up = try self.allocator.alloc(f32, p.intermediate_dim * p.n_embd);
                self.scratch_down = try self.allocator.alloc(f32, p.n_embd * p.intermediate_dim);
            }
            self.w_gate.dequantToF32Transposed(self.scratch_gate);
            self.w_up.dequantToF32Transposed(self.scratch_up);
            self.w_down.dequantToF32Transposed(self.scratch_down);
        }

        if (self.params.is_lfm2) {
            if (self.is_attention) {
                if (self.attn_layer) |*l| try l.loadWeightsFromGguf(g);
            } else {
                if (self.short_conv_layer) |*l| try l.loadWeightsFromGguf(g);
            }
        } else {
            // Qwen3.5 hybrid (SSM + Attention)
            if (self.is_attention) {
                if (self.attn_layer) |*l| try l.loadWeightsFromGguf(g);
            } else {
                if (self.ssm_layer) |*l| try l.loadWeightsFromGguf(g);
            }
        }
        self.weights_loaded = true;

        // ─── RLT: load feedback weights from GGUF (opt-in) ─────────────
        // Metadata: rlt.feedback_alpha (f32, default 0 = OFF)
        // Tensors: blk.{layer}.rlt.feedback_gate.weight [n_embd, 2*n_embd]
        //          blk.{layer}.rlt.feedback_state.weight [n_embd, n_embd]
        self.loadRltWeights(g, sidecar) catch |err| {
            // If weights not found → feedback stays disabled (alpha=0)
            if (err != HybridLayerError.WeightFileNotFound) return err;
        };

        // ─── RLT SWA: sliding window attention per-layer ──────────────
        // Metadata: rlt.swa_window (u32, default 0 = full context)
        // If present, each layer retains only W-1 historical KV pairs.
        if (g.getMeta("rlt.swa_window")) |v| {
            if (v.asU32()) |w| {
                if (w > 0) self.params.swa_window = w;
            }
        }
    }

    /// Libera los pesos dequantizados (scratch f32). Los QuantWeight (mmap
    /// references) permanecen — pueden recargarse vía loadWeightsFromGguf.
    /// El LayerStreamer llama esto en LRU eviction; reload es posible porque
    /// loadWeightsFromGguf re-alloca los scratch si están vacíos.
    pub fn unloadWeights(self: *HybridLayer) void {
        if (!self.weights_loaded) return;
        // Free scratch buffers (re-allocatables in loadWeightsFromGguf)
        // 7.1a-b (lane-c): evicción selectiva del weight_cache — el clear
        // global des-cacheaba las capas residentes válidas (thrash PCIe
        // por ciclo LRU del streamer). Ver nota completa en
        // hybrid_attn.unloadWeights. Las sublayers (attn/ssm/short_conv)
        // evictan las suyas en sus propios unloadWeights.
        _ = self.matmul_engine.evictWeightCachePtr(@intFromPtr(self.scratch_gate.ptr));
        _ = self.matmul_engine.evictWeightCachePtr(@intFromPtr(self.scratch_up.ptr));
        _ = self.matmul_engine.evictWeightCachePtr(@intFromPtr(self.scratch_down.ptr));
        self.allocator.free(self.scratch_gate);
        self.allocator.free(self.scratch_up);
        self.allocator.free(self.scratch_down);
        self.scratch_gate = &[_]f32{};
        self.scratch_up = &[_]f32{};
        self.scratch_down = &[_]f32{};
        // Unload sub-layer weights (attention/SSM/ShortConv) — frees their scratch too
        if (self.attn_layer) |*l| l.unloadWeights();
        if (self.ssm_layer) |*l| l.unloadWeights();
        if (self.short_conv_layer) |*l| l.unloadWeights();
        // Norm weights remain resident (small: [n_embd] each)
        self.weights_loaded = false;
    }

    /// Forward del bloque híbrido:
    /// x → attn_norm → (SSM | Attention | ShortConv) → +residual → attn_post_norm → FFN → +residual → out
    /// `pos_ids`: position-ids M-RoPE per-token (opcional; null ⇒ clásico).
    pub fn forward(self: *HybridLayer, x: Tensor(f32), out: *Tensor(f32), start_pos: usize, n: usize, pos_ids: ?[]const [4]i32) !void {
        const p = self.params;
        const N = n;
        // 9.4.1 diag: timing por fase gated .trace — localizar 80ms/capa
        const dbg_at = debugz.dbg.at(.trace);
        const hlc_now = struct {
            fn ns() i128 {
                var ts: std.posix.timespec = undefined;
                const rc = std.posix.system.clock_gettime(.MONOTONIC, &ts);
                if (rc != 0) return 0;
                return @as(i128, @intCast(ts.sec)) * std.time.ns_per_s + @as(i128, @intCast(ts.nsec));
            }
        }.ns;
        const t0: i128 = if (dbg_at) hlc_now() else 0;
        var t_prev: i128 = t0;

        // === RLT: Gated merge feedback (before first norm) ===
        // u_t = e_t + α * σ(W_g [e_t; RMSNorm(s_{t-1})]) ⊙ W_s RMSNorm(s_{t-1})
        // Only active for single-token decode (N=1) with feedback enabled.
        if (N == 1 and self.rlt_alpha > 0 and self.rlt_prev_state != null and self.rlt_w_gate != null) {
            const t_rlt_start: i128 = if (debugz.dbg.perf_rlt) hlc_now() else 0;
            const buf_numel = p.n_embd;
            const merge_data = try self.act_pool.alloc(buf_numel);
            defer self.act_pool.release(merge_data);
            // mergeFeedback writes into merge_data, then we copy back to x's slice
            self.mergeFeedback(x.data[0..buf_numel], self.rlt_prev_state.?, merge_data[0..buf_numel]);
            // Copy merged result back into x's data (overwrite first n_embd elements)
            @memcpy(x.data[0..buf_numel], merge_data[0..buf_numel]);
            if (debugz.dbg.perf_rlt) {
                const dt_ns = hlc_now() - t_rlt_start;
                debugz.dbg.printLevel(.info, "[rlt] L{d} merge: {d:.3}ms\n", .{ self.layer_idx, @as(f64, @floatFromInt(@divTrunc(dt_ns, std.time.ns_per_ms))) });
            }
            if (debugz.dbg.dump_rlt_state) {
                // Stats: mean |state|, mean |gate contribution|
                var sum_state: f32 = 0;
                for (self.rlt_prev_state.?) |v| sum_state += @abs(v);
                debugz.dbg.printLevel(.info, "[rlt] L{d} state_mean={d:.6}\n", .{ self.layer_idx, sum_state / @as(f32, @floatFromInt(p.n_embd)) });
            }
            // Update prev_state for next token (will be set at the end of forward)
        }

        // === 1. Pre-Attention/SSM RMSNorm (pool: reuse [N, n_embd] buffer) ===
        const buf_numel = N * p.n_embd;
        const norm_buf_data = try self.act_pool.alloc(buf_numel);
        defer self.act_pool.release(norm_buf_data);
        var norm_buf_shape = [_]usize{ N, p.n_embd };
        var norm_buf_strides = [_]usize{ p.n_embd, 1 };
        var norm_buf = Tensor(f32){ .data = norm_buf_data, .shape = &norm_buf_shape, .strides = &norm_buf_strides, .offset = 0, .allocator = null, .owns_data = false };
        if (p.is_k2_horizon) {
            // K2-Horizon: grouped RMSNorm (oráculo k2_horizon_group_rms_norm —
            // reshape(n_embd/g, g, tokens) → rms_norm → reshape → mul(w)).
            norm.groupedRmsNorm(f32, f32, x, self.attn_norm, p.n_norm_groups, p.rms_eps, &norm_buf);
        } else {
            norm.rmsNorm(f32, f32, x, self.attn_norm, p.rms_eps, &norm_buf);
        }

        // === 2. SSM o Attention o ShortConv (pool: reuse [N, n_embd]) ===
        if (dbg_at) {
            const now = hlc_now();
            debugz.dbg.printLevel(.trace, "[hybrid] L{d} norm={d}us\n", .{ self.layer_idx, @as(u64, @intCast(now - t_prev)) / 1000 });
            t_prev = now;
        }
        const mixer_data = try self.act_pool.alloc(buf_numel);
        defer self.act_pool.release(mixer_data);
        var mixer_shape = [_]usize{ N, p.n_embd };
        var mixer_strides = [_]usize{ p.n_embd, 1 };
        var mixer_out = Tensor(f32){ .data = mixer_data, .shape = &mixer_shape, .strides = &mixer_strides, .offset = 0, .allocator = null, .owns_data = false };

        if (self.params.is_lfm2) {
            if (self.is_attention) {
                if (self.attn_layer) |*l| {
                    try l.forward(norm_buf, &mixer_out, start_pos, N, pos_ids);
                }
            } else {
                if (self.short_conv_layer) |*l| {
                    try l.forward(norm_buf, &mixer_out, N);
                }
            }
        } else {
            // Qwen3.5 hybrid (SSM + Attention)
            if (self.is_attention) {
                if (self.attn_layer) |*l| {
                    try l.forward(norm_buf, &mixer_out, start_pos, N, pos_ids);
                }
            } else {
                if (self.ssm_layer) |*l| {
                    try l.forward(norm_buf, &mixer_out, N);
                }
            }
        }

        // === 3. Residual connection (Mixer) ===
        if (dbg_at) {
            const now = hlc_now();
            debugz.dbg.printLevel(.trace, "[hybrid] L{d} mixer={d}us\n", .{ self.layer_idx, @as(u64, @intCast(now - t_prev)) / 1000 });
            t_prev = now;
        }
        // OJO pool: mixer_out.data puede ser bloque reciclado mayor que la
        // shape lógica [N, n_embd] — iterar buf_numel (tamaño lógico del paso).
        const out_slice = out.data[0..buf_numel];
        const x_slice = x.data[0..buf_numel];
        for (out_slice, mixer_out.data[0..buf_numel], x_slice) |*o, m, xv| {
            o.* = m + xv;
        }

        // === 4. Post-Attention/SSM RMSNorm (pool: reuse [N, n_embd]) ===
        const post_norm_data = try self.act_pool.alloc(buf_numel);
        defer self.act_pool.release(post_norm_data);
        var post_norm_shape = [_]usize{ N, p.n_embd };
        var post_norm_strides = [_]usize{ p.n_embd, 1 };
        var post_norm_buf = Tensor(f32){ .data = post_norm_data, .shape = &post_norm_shape, .strides = &post_norm_strides, .offset = 0, .allocator = self.allocator, .owns_data = false };
        // attn_post_norm holds post_attention_norm (Qwen3.5) or ffn_norm (LFM2/K2-Horizon)
        const post_norm_weight = self.attn_post_norm.?;
        if (p.is_k2_horizon) {
            norm.groupedRmsNorm(f32, f32, out.*, post_norm_weight, p.n_norm_groups, p.rms_eps, &post_norm_buf);
        } else {
            norm.rmsNorm(f32, f32, out.*, post_norm_weight, p.rms_eps, &post_norm_buf);
        }

        // === 5. FFN SwiGLU ===
        if (dbg_at) {
            const now = hlc_now();
            debugz.dbg.printLevel(.trace, "[hybrid] L{d} postnorm={d}us\n", .{ self.layer_idx, @as(u64, @intCast(now - t_prev)) / 1000 });
            t_prev = now;
        }
        // T1 fix regresión CPU: consumidor f32 garantiza scratch lleno.
        try self.ensureFfnScratchFilled();
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

        // Pool: reuse [N, intermediate_dim] buffers for gate/up/ffn_out
        const ff_in_numel = N * p.intermediate_dim;
        const gate_data = try self.act_pool.alloc(ff_in_numel);
        defer self.act_pool.release(gate_data);
        var gate_shape = [_]usize{ N, p.intermediate_dim };
        var gate_strides = [_]usize{ p.intermediate_dim, 1 };
        var gate_buf = Tensor(f32){ .data = gate_data, .shape = &gate_shape, .strides = &gate_strides, .offset = 0, .allocator = null, .owns_data = false };

        const up_data = try self.act_pool.alloc(ff_in_numel);
        defer self.act_pool.release(up_data);
        var up_shape = [_]usize{ N, p.intermediate_dim };
        var up_strides = [_]usize{ p.intermediate_dim, 1 };
        var up_buf = Tensor(f32){ .data = up_data, .shape = &up_shape, .strides = &up_strides, .offset = 0, .allocator = null, .owns_data = false };

        const ffn_out_data = try self.act_pool.alloc(buf_numel);
        defer self.act_pool.release(ffn_out_data);
        var ffn_shape = [_]usize{ N, p.n_embd };
        var ffn_strides = [_]usize{ p.n_embd, 1 };
        var ffn_out = Tensor(f32){ .data = ffn_out_data, .shape = &ffn_shape, .strides = &ffn_strides, .offset = 0, .allocator = null, .owns_data = false };

        const post_norm_2d = try post_norm_buf.reshape(&[_]usize{ N, p.n_embd });
        defer {
            if (post_norm_2d.allocator) |a| {
                a.free(post_norm_2d.shape);
                a.free(post_norm_2d.strides);
            }
        }

        try ffn.swiGluForward(
            &self.matmul_engine,
            f32,
            post_norm_2d,
            w_gate32,
            w_up32,
            w_down32,
            &gate_buf,
            &up_buf,
            &ffn_out,
        );

        // === 6. Residual connection (FFN) ===
        if (dbg_at) {
            const now = hlc_now();
            debugz.dbg.printLevel(.trace, "[hybrid] L{d} ffn={d}us (total {d}us)\n", .{ self.layer_idx, @as(u64, @intCast(now - t_prev)) / 1000, @as(u64, @intCast(now - t0)) / 1000 });
        }
        // OJO pool: ffn_out.data puede ser bloque reciclado > shape lógica.
        const out_slice2 = out.data[0..buf_numel];
        for (out_slice2, ffn_out.data[0..buf_numel]) |*o, f| {
            o.* += f;
        }

        // === RLT: Update feedback state with final output (for next token) ===
        if (N == 1 and self.rlt_prev_state != null) {
            // out.data[0..n_embd] now contains the final residual stream output
            // which is s_t — the recurrent state for the next token.
            @memcpy(self.rlt_prev_state.?, out.data[0..p.n_embd]);
        }
    }

    // ─── Forward híbrido residente en GPU (Path B) ──────────────────────────────
    // Activa, mixer, residual, FFN y normas viven íntegramente en device; solo hay
    // UNA sincronización de stream por token (en el llamador). Para capas de
    // atención el mixer (incluido decode paginado, MRoPE y KV) también corre en
    // GPU (Phase 1b); el KV escrito se baja al host pool vía D2H async.
    pub const HybridGpu = struct {
        g_norm: cublas.GpuTensor(f32),
        g_mixer: cublas.GpuTensor(f32),
        g_post: cublas.GpuTensor(f32),
        g_gate: cublas.GpuTensor(f32),
        g_up: cublas.GpuTensor(f32),
        g_ffn: cublas.GpuTensor(f32),
        g_attn_norm: cublas.GpuBuffer(f32),
        g_attn_post_norm: cublas.GpuBuffer(f32),
        // RLT: persistent device buffer for recurrent feedback state [n_embd]
        g_rlt_prev_state: cublas.GpuBuffer(f32) = undefined,
        g_rlt_gate_dev: cudaz.CUdeviceptr = 0, // [d, 2d] gate projection on device
        g_rlt_state_dev: cudaz.CUdeviceptr = 0, // [d, d] state projection on device
        rlt_weights_uploaded: bool = false,
        cap_n: usize,
        params: HybridLayerParams,

        fn alloc(p: HybridLayerParams) !HybridGpu {
            const g_attn_norm = try cublas.GpuBuffer(f32).alloc(p.n_embd);
            const g_attn_post_norm = try cublas.GpuBuffer(f32).alloc(p.n_embd);
            return .{
                .g_norm = try cublas.GpuTensor(f32).alloc(p.n_embd),
                .g_mixer = try cublas.GpuTensor(f32).alloc(p.n_embd),
                .g_post = try cublas.GpuTensor(f32).alloc(p.n_embd),
                .g_gate = try cublas.GpuTensor(f32).alloc(p.intermediate_dim),
                .g_up = try cublas.GpuTensor(f32).alloc(p.intermediate_dim),
                .g_ffn = try cublas.GpuTensor(f32).alloc(p.n_embd),
                .g_attn_norm = g_attn_norm,
                .g_attn_post_norm = g_attn_post_norm,
                .cap_n = 1,
                .params = p,
            };
        }

        fn ensureN(self: *HybridGpu, n: usize) !void {
            if (self.cap_n >= n) return;
            const p = self.params;
            if (self.cap_n > 0) {
                self.g_norm.deinit();
                self.g_mixer.deinit();
                self.g_post.deinit();
                self.g_gate.deinit();
                self.g_up.deinit();
                self.g_ffn.deinit();
            }
            self.g_norm = try cublas.GpuTensor(f32).alloc(n * p.n_embd);
            self.g_mixer = try cublas.GpuTensor(f32).alloc(n * p.n_embd);
            self.g_post = try cublas.GpuTensor(f32).alloc(n * p.n_embd);
            self.g_gate = try cublas.GpuTensor(f32).alloc(n * p.intermediate_dim);
            self.g_up = try cublas.GpuTensor(f32).alloc(n * p.intermediate_dim);
            self.g_ffn = try cublas.GpuTensor(f32).alloc(n * p.n_embd);
            self.cap_n = n;
        }

        fn deinit(self: *HybridGpu) void {
            self.g_norm.deinit();
            self.g_mixer.deinit();
            self.g_post.deinit();
            self.g_gate.deinit();
            self.g_up.deinit();
            self.g_ffn.deinit();
            self.g_attn_norm.free();
            self.g_attn_post_norm.free();
            // RLT: free persistent device state + weight buffers
            if (self.rlt_weights_uploaded) {
                self.g_rlt_prev_state.free();
                if (self.g_rlt_gate_dev != 0) cudaz.cuMemFree(self.g_rlt_gate_dev) catch {};
                if (self.g_rlt_state_dev != 0) cudaz.cuMemFree(self.g_rlt_state_dev) catch {};
            }
        }
    };

    /// Copia el estado recurrente del prefill CPU al GPU para capas SSM.
    pub fn seedGpuFromHost(self: *HybridLayer) !void {
        if (self.ssm_layer) |*l| try SsmLayer.seedGpuFromHost(l);
    }

    /// RLT: snapshot GPU feedback state → host (for speculative decoding rollback).
    /// `dst` must have >= n_embd elements.
    pub fn snapshotRltGpuState(self: *const HybridLayer, dst: []f32) !void {
        const g = self.gpu orelse return;
        if (!g.rlt_weights_uploaded) return;
        if (dst.len < self.params.n_embd) return error.BufferTooSmall;
        try cudaz.cuMemcpyDtoH(@intFromPtr(dst.ptr), g.g_rlt_prev_state.dev_ptr, self.params.n_embd * @sizeOf(f32));
    }

    /// RLT: restore GPU feedback state from host snapshot.
    pub fn restoreRltGpuState(self: *HybridLayer, src: []const f32) !void {
        const g = self.gpu orelse return;
        if (!g.rlt_weights_uploaded) return;
        if (src.len < self.params.n_embd) return error.BufferTooSmall;
        try cudaz.cuMemcpyHtoD(g.g_rlt_prev_state.dev_ptr, @intFromPtr(src.ptr), self.params.n_embd * @sizeOf(f32));
    }

    pub fn ensureGpu(self: *HybridLayer) !void {
        if (self.gpu != null) return;
        var g = try HybridGpu.alloc(self.params);
        try g.g_attn_norm.upload(self.attn_norm.data);
        if (self.attn_post_norm) |post_norm| {
            try g.g_attn_post_norm.upload(post_norm.data);
        }
        // RLT: allocate + zero device state buffer, upload weights once
        if (self.rlt_alpha > 0) {
            g.g_rlt_prev_state = try cublas.GpuBuffer(f32).alloc(self.params.n_embd);
            // Zero-init via cuMemcpyHtoD (sync) — ensureGpu no está en graph
            // capture, así que sync es seguro y evita el problema de stream null
            // con opaque pointer en Zig.
            var zero_buf: [4096]u8 = undefined;
            var remaining = self.params.n_embd * @sizeOf(f32);
            var offset: usize = 0;
            while (remaining > 0) {
                const chunk = @min(remaining, zero_buf.len);
                @memset(zero_buf[0..chunk], 0);
                try cudaz.cuMemcpyHtoD(
                    @intFromPtr(g.g_rlt_prev_state.dev_ptr) + offset,
                    @intFromPtr(&zero_buf),
                    chunk,
                );
                offset += chunk;
                remaining -= chunk;
            }
            // Upload gate + state projection weights to device
            if (self.rlt_w_gate) |w_gate| {
                const d = self.params.n_embd;
                const gate_bytes = d * 2 * d * @sizeOf(f32);
                const d_gate_ptr = try cudaz.cuMemAlloc(gate_bytes);
                try cudaz.cuMemcpyHtoD(d_gate_ptr, @intFromPtr(w_gate.data.ptr), gate_bytes);
                g.g_rlt_gate_dev = d_gate_ptr;
            }
            if (self.rlt_w_state) |ws| {
                const d = self.params.n_embd;
                const state_bytes = d * d * @sizeOf(f32);
                const d_state_ptr = try cudaz.cuMemAlloc(state_bytes);
                try cudaz.cuMemcpyHtoD(d_state_ptr, @intFromPtr(ws.data.ptr), state_bytes);
                g.g_rlt_state_dev = d_state_ptr;
            }
            g.rlt_weights_uploaded = true;
        }
        self.gpu = g;
    }

    pub fn warmupGpuWeights(self: *HybridLayer) !void {
        const p = self.params;
        if (self.params.is_lfm2) {
            if (self.is_attention) {
                if (self.attn_layer) |*l| try l.warmupGpuWeights();
            } else {
                if (self.short_conv_layer) |*l| try l.warmupGpuWeights();
            }
        } else {
            if (self.is_attention) {
                if (self.attn_layer) |*l| try l.warmupGpuWeights();
            } else {
                if (self.ssm_layer) |*l| try SsmLayer.warmupGpuWeights(l);
            }
        }
        // T1 VRAM-spec: mismo criterio que forward — si hay kernel qgemm para
        // el dtype NO hace falta el W_T f32 en cache.
        const q4_ok = layer_kernels.quantPath() and !debugz.dbg.no_q4_ffn and
            SsmLayer.qgemmTypeFor(self.w_gate.dtype()) != null and
            SsmLayer.qgemmTypeFor(self.w_up.dtype()) != null;
        if (!q4_ok) {
            try self.ensureFfnScratchFilled();
            var w_gate_shape = [_]usize{ p.intermediate_dim, p.n_embd };
            var w_gate_strides = [_]usize{ p.n_embd, 1 };
            _ = try self.matmul_engine.projectionDevicePtr(Tensor(f32){ .data = self.scratch_gate, .shape = &w_gate_shape, .strides = &w_gate_strides, .offset = 0, .allocator = null, .owns_data = false });
            var w_up_shape = [_]usize{ p.intermediate_dim, p.n_embd };
            var w_up_strides = [_]usize{ p.n_embd, 1 };
            _ = try self.matmul_engine.projectionDevicePtr(Tensor(f32){ .data = self.scratch_up, .shape = &w_up_shape, .strides = &w_up_strides, .offset = 0, .allocator = null, .owns_data = false });
        }
        const w_down_q4 = layer_kernels.quantPath() and !debugz.dbg.no_q4_ffn and
            SsmLayer.qgemmTypeFor(self.w_down.dtype()) != null;
        if (!w_down_q4) {
            try self.ensureFfnScratchFilled();
            var w_down_shape = [_]usize{ p.n_embd, p.intermediate_dim };
            var w_down_strides = [_]usize{ p.intermediate_dim, 1 };
            _ = try self.matmul_engine.projectionDevicePtr(Tensor(f32){ .data = self.scratch_down, .shape = &w_down_shape, .strides = &w_down_strides, .offset = 0, .allocator = null, .owns_data = false });
        }
    }

    /// M3 slice 2 (Dev-B, P4): delega a `AttentionLayer.kvDevicePtrs` si
    /// esta capa es de atención. Retorna null si es SSM/ShortConv o si
    /// forwardGPU aún no se llamó sobre la capa attn interna.
    pub const KvDevicePtrs = AttentionLayer.KvDevicePtrs;
    pub fn kvDevicePtrs(self: *const HybridLayer) ?KvDevicePtrs {
        if (self.attn_layer) |*al| return al.kvDevicePtrs();
        return null;
    }

    pub fn forwardGPU(
        self: *HybridLayer,
        lk: *layer_kernels.LayerKernels,
        x: cublas.GpuTensor(f32),
        out: *cublas.GpuTensor(f32),
        start_pos: usize,
        n: usize,
        pos_ids: ?[]const [4]i32,
    ) !void {
        const p = self.params;
        try HybridLayer.ensureGpu(self);
        const g = &self.gpu.?;
        try g.ensureN(n);

        // === RLT: Gated merge feedback (decode-only, before first norm) ===
        if (n == 1 and self.rlt_alpha > 0 and g.rlt_weights_uploaded and g.g_rlt_gate_dev != 0) {
            // mergeFeedback kernel: in-place update of x.ptr()
            // Writes merged output to g.g_norm (reuse as scratch), then copy to x
            try lk.mergeFeedback(
                x.ptr(), // encoder_rep (input token embedding)
                @intFromPtr(g.g_rlt_prev_state.dev_ptr), // prev_state on device
                g.g_rlt_gate_dev, // w_gate on device
                g.g_rlt_state_dev, // w_state on device
                g.g_norm.ptr(), // output (temporary, overwrite g_norm)
                p.n_embd,
                self.rlt_alpha,
            );
            // Copy merged result back to x for the rest of the forward
            try lk.add(g.g_norm.ptr(), 0, x.ptr(), p.n_embd); // x = g_norm + 0 (copy)
        }

        try lk.rmsNorm(x.ptr(), @intFromPtr(g.g_attn_norm.dev_ptr), g.g_norm.ptr(), n, p.n_embd, p.rms_eps);

        if (self.params.is_lfm2) {
            if (self.is_attention) {
                if (self.attn_layer) |*l| try l.forwardGPU(lk, g.g_norm, &g.g_mixer, start_pos, n, pos_ids);
                try lk.add(x.ptr(), g.g_mixer.ptr(), out.ptr(), n * p.n_embd);
            } else {
                // 8.3: LFM2 residual — lfm2.cpp:259 `cur = ggml_add(prev_cur,
                // cur)` aplica SIEMPRE el residual tras el shortconv block.
                // El short_conv escribia directamente en `out` SIN el
                // `x +` — sin residual la señal degrada a blanks/PAD.
                if (self.short_conv_layer) |*l| try l.forwardGPU(lk, g.g_norm, &g.g_mixer, n);
                try lk.add(x.ptr(), g.g_mixer.ptr(), out.ptr(), n * p.n_embd);
            }
        } else {
            // Qwen3.5 hybrid (SSM + Attention)
            if (self.is_attention) {
                if (self.attn_layer) |*l| try l.forwardGPU(lk, g.g_norm, &g.g_mixer, start_pos, n, pos_ids);
                try lk.add(x.ptr(), g.g_mixer.ptr(), out.ptr(), n * p.n_embd);
            } else {
                if (self.ssm_layer) |*l| try SsmLayer.forwardGPU(l, lk, g.g_norm, &g.g_mixer, n);
                try lk.add(x.ptr(), g.g_mixer.ptr(), out.ptr(), n * p.n_embd);
            }
        }

        if (self.attn_post_norm) |_| {
            try lk.rmsNorm(out.ptr(), @intFromPtr(g.g_attn_post_norm.dev_ptr), g.g_post.ptr(), n, p.n_embd, p.rms_eps);
        }

        // ── FFN MoE offload (lane-e): ruta exclusiva cuando la capa tiene spec.
        // v1 solo decode bs=1; PREFILL (n>1) cae al FFN denso sintetizado
        // (experto 0, loadQuantWeightOrExpert0) — aproximación documentada
        // hasta que llegue el prefill batched por-experto (integración C).
        // El decode es donde el MoE domina el coste (routing cambia por token).
        if (self.moe) |ml| {
            if (n == 1) {
                try ml.forwardGPU(g.g_post.ptr(), g.g_ffn.ptr());
                try lk.addInplace(out.ptr(), g.g_ffn.ptr(), n * p.n_embd);
                return;
            }
            debugz.dbg.printLevel(.info, "[moe_layer] capa {d}: prefill n={d} por FFN denso (experto 0) — MoE batched llega con integración C\n", .{ self.layer_idx, n });
        }

        var w_gate_shape = [_]usize{ p.intermediate_dim, p.n_embd };
        var w_gate_strides = [_]usize{ p.n_embd, 1 };
        const w_gate32 = Tensor(f32){ .data = self.scratch_gate, .shape = &w_gate_shape, .strides = &w_gate_strides, .offset = 0, .allocator = null, .owns_data = false };
        var w_up_shape = [_]usize{ p.intermediate_dim, p.n_embd };
        var w_up_strides = [_]usize{ p.n_embd, 1 };
        const w_up32 = Tensor(f32){ .data = self.scratch_up, .shape = &w_up_shape, .strides = &w_up_strides, .offset = 0, .allocator = null, .owns_data = false };
        var w_down_shape = [_]usize{ p.n_embd, p.intermediate_dim };
        var w_down_strides = [_]usize{ p.intermediate_dim, 1 };
        const w_down32 = Tensor(f32){ .data = self.scratch_down, .shape = &w_down_shape, .strides = &w_down_strides, .offset = 0, .allocator = null, .owns_data = false };

        // FFN con pesos Q4_0/Q4_1 → GEMM cuantizado device (8× menos tráfico VRAM),
        // también batched para prefill (n > 1).
        // T1 VRAM-spec (ticket C 21:20): GEMM cuantizado para TODO dtype con
        // kernel qgemm (antes solo q4_0/q4_1; el resto dequantizaba ~1GB f32
        // por capa a weight_cache sin eviction). Fallback f32 solo para
        // dtypes sin kernel (f16/f32/q8_k).
        const qt_gate = if (layer_kernels.quantPath() and !debugz.dbg.no_q4_ffn) SsmLayer.qgemmTypeFor(self.w_gate.dtype()) else null;
        const qt_up = if (layer_kernels.quantPath() and !debugz.dbg.no_q4_ffn) SsmLayer.qgemmTypeFor(self.w_up.dtype()) else null;
        if (qt_gate != null and qt_up != null) {
            try lk.qgemmLinear(self.allocator, g.g_post.ptr(), self.w_gate.bytes, g.g_gate.ptr(), n, p.n_embd, p.intermediate_dim, qt_gate.?);
            try lk.qgemmLinear(self.allocator, g.g_post.ptr(), self.w_up.bytes, g.g_up.ptr(), n, p.n_embd, p.intermediate_dim, qt_up.?);
        } else {
            try self.ensureFfnScratchFilled();
            try self.matmul_engine.linearProjectionDevice(g.g_post, w_gate32, &g.g_gate, n, p.n_embd, p.intermediate_dim);
            try self.matmul_engine.linearProjectionDevice(g.g_post, w_up32, &g.g_up, n, p.n_embd, p.intermediate_dim);
        }
        try lk.swiglu(g.g_gate.ptr(), g.g_up.ptr(), n * p.intermediate_dim);
        const qt_down = if (layer_kernels.quantPath() and !debugz.dbg.no_q4_ffn) SsmLayer.qgemmTypeFor(self.w_down.dtype()) else null;
        if (qt_down) |qt| {
            try lk.qgemmLinear(self.allocator, g.g_gate.ptr(), self.w_down.bytes, g.g_ffn.ptr(), n, p.intermediate_dim, p.n_embd, qt);
        } else {
            try self.ensureFfnScratchFilled();
            try self.matmul_engine.linearProjectionDevice(g.g_gate, w_down32, &g.g_ffn, n, p.intermediate_dim, p.n_embd);
        }
        try lk.addInplace(out.ptr(), g.g_ffn.ptr(), n * p.n_embd);

        // === RLT: Update feedback state with final output (for next token) ===
        if (n == 1 and g.rlt_weights_uploaded) {
            // out.ptr() now contains s_t — copy to persistent device state
            try cudaz.cuMemcpyDtoD(@intFromPtr(g.g_rlt_prev_state.dev_ptr), out.ptr(), p.n_embd * @sizeOf(f32));
        }
    }

    /// Upload all layer weights to GPU via GpuWeightPool (async H2D).
    /// Returns CUevent to wait on before compute can use the weights.
    pub fn uploadWeightsToGpu(
        self: *HybridLayer,
        pool: *gpu_weight_pool.GpuWeightPool,
        tensors: *std.ArrayList(gpu_weight_pool.GpuWeightPool.GpuTensorHandle),
    ) !cudaz.CUevent {
        // Ensure GPU structures are allocated
        try HybridLayer.ensureGpu(self);

        // Upload all weight tensors for this layer
        var tensors_list: std.ArrayList(gpu_weight_pool.GpuWeightPool.GpuTensorHandle) = .empty;
        errdefer tensors_list.deinit(self.allocator);

        // Upload common FFN weights
        try uploadQuantWeight(self.w_gate, &self.matmul_engine, pool, &tensors_list, self.allocator);
        try uploadQuantWeight(self.w_up, &self.matmul_engine, pool, &tensors_list, self.allocator);
        try uploadQuantWeight(self.w_down, &self.matmul_engine, pool, &tensors_list, self.allocator);

        // Upload norm weights
        if (self.attn_post_norm) |*apn| try uploadNorm(apn, pool);
        try uploadNorm(&self.attn_norm, pool);

        // Upload sub-layer specific weights
        if (self.is_attention) {
            if (self.attn_layer) |*attn| {
                try uploadQuantWeight(attn.w_q, &self.matmul_engine, pool, &tensors_list, self.allocator);
                try uploadQuantWeight(attn.w_k, &self.matmul_engine, pool, &tensors_list, self.allocator);
                try uploadQuantWeight(attn.w_v, &self.matmul_engine, pool, &tensors_list, self.allocator);
                try uploadQuantWeight(attn.w_o, &self.matmul_engine, pool, &tensors_list, self.allocator);
                try uploadNorm(&attn.attn_q_norm, pool);
                try uploadNorm(&attn.attn_k_norm, pool);
            }
        } else if (self.short_conv_layer) |*sc| {
            try uploadQuantWeight(sc.w_in_proj, &self.matmul_engine, pool, &tensors_list, self.allocator);
            try uploadQuantWeight(sc.w_out_proj, &self.matmul_engine, pool, &tensors_list, self.allocator);
            try uploadNorm(&sc.attn_norm, pool);
            try uploadNorm(&sc.ffn_norm, pool);
        } else if (self.ssm_layer) |*ssm| {
            try uploadQuantWeight(ssm.w_qkv, &self.matmul_engine, pool, &tensors_list, self.allocator);
            try uploadQuantWeight(ssm.w_z, &self.matmul_engine, pool, &tensors_list, self.allocator);
            try uploadQuantWeight(ssm.w_out, &self.matmul_engine, pool, &tensors_list, self.allocator);
            try uploadNorm(&ssm.ssm_norm, pool);
        }

        // Move tensors to the output list
        try tensors.appendSlice(self.allocator, tensors_list.items);

        // Return an event for the last upload (approximate sync point)
        return try cudaz.cuEventCreate(0);
    }
    // ─── RLT: Recurrent Looped Transformer feedback ──────────────────────────────
    // Implements gated merge: u_t = e_t + α * σ(W_g [e_t; RMSNorm(s_{t-1})]) ⊙ W_s RMSNorm(s_{t-1})
    // All opt-in: without GGUF weights/alpha → merge is skipped.

    fn loadRltWeights(self: *HybridLayer, g: *const gguf.GgufFile, sidecar: ?*const gguf.GgufFile) !void {
        const p = self.params;
        const d: usize = p.n_embd;

        // Check CLI/env override first
        if (debugz.dbg.rlt_feedback) |override| {
            if (!override) {
                // Force OFF
                self.rlt_alpha = 0;
                return;
            }
            // Force ON — use default alpha if GGUF doesn't specify
            if (g.getMeta("rlt.feedback_alpha")) |v| {
                self.rlt_alpha = v.asF32() orelse 0.1;
            } else {
                self.rlt_alpha = 0.1;
            }
        } else {
            // Auto: read alpha from metadata (default 0 = OFF)
            if (g.getMeta("rlt.feedback_alpha")) |v| {
                self.rlt_alpha = v.asF32() orelse 0.0;
            }
        }
        if (self.rlt_alpha == 0) return; // OFF — skip weight loading

        // Load gate projection: [d, 2d] → transposed to [2d, d]
        const gate_name = try std.fmt.allocPrint(self.allocator, "blk.{d}.rlt.feedback_gate.weight", .{self.layer_idx});
        defer self.allocator.free(gate_name);
        const state_name = try std.fmt.allocPrint(self.allocator, "blk.{d}.rlt.feedback_state.weight", .{self.layer_idx});
        defer self.allocator.free(state_name);
        const src_g = if (sidecar) |sc| blk: {
            if (sc.getTensor(gate_name)) |_| break :blk sc;
            break :blk g;
        } else g;
        if (src_g.getTensor(gate_name)) |_| {
            self.rlt_w_gate = try loadGgufF32(self.allocator, src_g, "", gate_name);
        } else {
            // No gate weight → disable feedback
            self.rlt_alpha = 0;
            return;
        }
        if (src_g.getTensor(state_name)) |_| {
            self.rlt_w_state = try loadGgufF32(self.allocator, src_g, "", state_name);
        } else {
            self.rlt_alpha = 0;
            return;
        }

        // Allocate persistent state + scratch buffers
        self.rlt_prev_state = try self.allocator.alloc(f32, d);
        @memset(self.rlt_prev_state.?, 0);
        self.rlt_scratch_gate = try self.allocator.alloc(f32, d);
        self.rlt_scratch_state = try self.allocator.alloc(f32, d);

        debugz.dbg.printLevel(.info, "[rlt] capa {d}: feedback activo alpha={d:.3}\n", .{ self.layer_idx, self.rlt_alpha });
    }

    /// Gated merge: u_t = e_t + α * gate ⊙ W_s * RMSNorm(s_{t-1})
    /// Input: encoder_rep [n_embd], prev_state [n_embd]
    /// Output: merged [n_embd]
    /// Uses scratch_gate [n_embd] and scratch_state [n_embd] as temporaries.
    fn mergeFeedback(
        self: *HybridLayer,
        encoder_rep: []f32, // [n_embd] — e_t from encoder
        prev_state: []f32, // [n_embd] — s_{t-1}
        merged: []f32, // [n_embd] — output u_t
    ) void {
        const d = self.params.n_embd;
        const w_gate = self.rlt_w_gate.?.data; // [2d, d] transposed
        const w_state = self.rlt_w_state.?.data; // [d, d] transposed
        const scratch_gate = self.rlt_scratch_gate.?;
        const scratch_state = self.rlt_scratch_state.?;

        // 1. r = RMSNorm(prev_state)
        var sum_sq: f32 = 0;
        for (prev_state) |v| sum_sq += v * v;
        const rms = @sqrt(sum_sq / @as(f32, @floatFromInt(d)) + 1e-6);
        for (prev_state, 0..) |v, i| {
            // Store normalized prev_state back into prev_state (reuse as r)
            // But we need prev_state later? No — we only use r after this.
            // Actually let's use scratch_state as r to avoid overwriting prev_state
            scratch_state[i] = v / rms;
        }

        // 2. Concat [encoder_rep; r] → [2d] and project through W_g → [d]
        // W_gate is [2d, d] transposed (GGUF stores [in, out], loadGgufF32 transposes to [out, in])
        // So W_gate data layout is [2d rows × d cols] — we need to compute:
        // gate = W_gate @ concat(encoder_rep, r) = sum over j of W_gate[i][j] * concat[j]
        // But concat is [2d] and W_gate is [2d, d] → output is [d]
        for (0..d) |i| {
            var acc: f32 = 0;
            // First d elements of concat = encoder_rep
            for (0..d) |j| {
                acc += w_gate[i * (2 * d) + j] * encoder_rep[j];
            }
            // Next d elements of concat = r (normalized prev_state)
            for (0..d) |j| {
                acc += w_gate[i * (2 * d) + d + j] * scratch_state[j];
            }
            scratch_gate[i] = acc; // pre-activation gate
        }

        // 3. Sigmoid gate: gate = σ(scratch_gate)
        for (scratch_gate) |*v| {
            v.* = 1.0 / (1.0 + @exp(-v.*));
        }

        // 4. State projection: state_out = W_state @ r
        for (0..d) |i| {
            var acc: f32 = 0;
            for (0..d) |j| {
                acc += w_state[i * d + j] * scratch_state[j];
            }
            scratch_state[i] = acc;
        }

        // 5. merged = encoder_rep + α * gate ⊙ state_out
        for (merged, encoder_rep, scratch_gate, scratch_state) |*o, e, g, s| {
            o.* = e + self.rlt_alpha * g * s;
        }
    }

    /// Allocate RLT feedback state for a new sequence.
    /// Called from the pipeline when a new sequence is created.
    pub fn allocateRltState(self: *HybridLayer) !void {
        if (self.rlt_alpha == 0) return; // no feedback
        if (self.rlt_prev_state != null) return; // already allocated
        const d = self.params.n_embd;
        self.rlt_prev_state = try self.allocator.alloc(f32, d);
        @memset(self.rlt_prev_state.?, 0);
    }

    /// Free RLT feedback state (sequence removed).
    pub fn freeRltState(self: *HybridLayer) void {
        if (self.rlt_prev_state) |s| {
            self.allocator.free(s);
            self.rlt_prev_state = null;
        }
    }
};

fn uploadQuantWeight(
    qw: QuantWeight,
    engine: *matmul.MatmulEngine,
    pool: *gpu_weight_pool.GpuWeightPool,
    tensors: *std.ArrayList(gpu_weight_pool.GpuWeightPool.GpuTensorHandle),
    allocator: std.mem.Allocator,
) !void {
    _ = engine; // TODO: use for quantization-aware upload
    // Get or create GPU buffer for this weight
    const bytes = qw.bytes.len;
    var d_ptr: cudaz.CUdeviceptr = 0;

    if (pool.gpu_buffer_pool) |*gpu_pool| {
        const ptr = try gpu_pool.acquire(bytes);
        d_ptr = @intFromPtr(ptr);
    } else {
        d_ptr = try cudaz.cuMemAlloc(bytes);
    }

    // Use pinned buffer for staging
    const pinned_ptr: [*]u8 = @ptrFromInt(pool.pinned_buffer.?);
    @memcpy(pinned_ptr[0..bytes], qw.bytes);

    // Async H2D copy
    try cudaz.cuMemcpyHtoDAsync(d_ptr, @intFromPtr(pinned_ptr), bytes, pool.stream);

    // Record event for this upload
    const event = try cudaz.cuEventCreate(0);
    try cudaz.cuEventRecord(event, pool.stream);

    try tensors.append(allocator, .{ .d_ptr = d_ptr, .bytes = bytes, .host_ptr = pool.pinned_buffer });
}

fn uploadNorm(
    norm_weight: *Tensor(f32),
    pool: *gpu_weight_pool.GpuWeightPool,
) !void {
    // Upload norm weight (f32) - small, sync is fine
    const bytes = norm_weight.data.len * @sizeOf(f32);
    var d_ptr: cudaz.CUdeviceptr = 0;

    if (pool.gpu_buffer_pool) |*gpu_pool| {
        const ptr = try gpu_pool.acquire(bytes);
        d_ptr = @intFromPtr(ptr);
    } else {
        d_ptr = try cudaz.cuMemAlloc(bytes);
    }

    try cudaz.cuMemcpyHtoD(d_ptr, @intFromPtr(norm_weight.data.ptr), bytes);
}

fn loadQuantWeight(g: *const gguf.GgufFile, prefix: []const u8, name: []const u8) !QuantWeight {
    const full = try std.fmt.allocPrint(std.heap.page_allocator, "{s}{s}", .{ prefix, name });
    defer std.heap.page_allocator.free(full);
    const info = g.getTensor(full) orelse return HybridLayerError.WeightFileNotFound;
    return QuantWeight.init(info, g.tensorData(info));
}

/// FFN densa (`ffn_<k>.weight`) o, en modelos MoE (qwen3moe/qwen2moe), el
/// EXPERTO 0 del banco 3-D `ffn_<k>_exps.weight` [in,out,E]. El experto vive
/// contiguo en el tensor ⇒ slice de bytes + TensorInfo sintetizada 2-D.
// Patch por (capa, kind): el stub v1 usaba UN global [3] compartido — la
// capa 1 parcheaba sus dims y las capas 2..N reusaban ese patch con SUS
// bytes ⇒ geometría incoherente → puntero OOB en el primer H2D del qgemm
// (CudaError INVALID_VALUE, repro loggenix qwen3moe 12 capas). Cada capa
// MoE necesita SU info 2-D sintética porque la geometría expert_bytes
// difiere por capa.
var g_moe_expert_infos: [128][3]gguf.TensorInfo = undefined;
var g_moe_expert_used: [128][3]bool = .{.{ false, false, false }} ** 128;

fn loadQuantWeightOrExpert0(g: *const gguf.GgufFile, prefix: []const u8, kind: []const u8) !QuantWeight {
    const alloc = std.heap.page_allocator;
    const dense = try std.fmt.allocPrint(alloc, "{s}{s}.weight", .{ prefix, kind });
    defer alloc.free(dense);
    if (g.getTensor(dense)) |info| {
        return QuantWeight.init(info, g.tensorData(info));
    }

    const exps = try std.fmt.allocPrint(alloc, "{s}{s}_exps.weight", .{ prefix, kind });
    defer alloc.free(exps);
    const info = g.getTensor(exps) orelse return HybridLayerError.WeightFileNotFound;

    // 3-D [in, out, E]: experto e = bytes[e*expert_bytes..]. La dim de
    // experto es la ÚLTIMA (convención gguf_moe); info 2-D sintética con
    // las dos primeras dims (bytes de un experto = dataBytes/E).
    const E: usize = @intCast(info.dims[2]);
    if (E == 0) return HybridLayerError.WeightFileNotFound;
    const expert_bytes = info.dataBytes() / E;
    const all_bytes = g.tensorData(info);

    const idx: usize = switch (kind[4]) {
        'g' => 0, // ffn_gate
        'u' => 1, // ffn_up
        else => 2, // ffn_down
    };
    // Capa desde el prefix "blk.N." — clave del patch per-capa.
    const li: usize = blk: {
        if (prefix.len >= 5 and std.mem.startsWith(u8, prefix, "blk.")) {
            const rest = prefix[4..];
            var v: usize = 0;
            var i: usize = 0;
            while (i < rest.len and rest[i] >= '0' and rest[i] <= '9') : (i += 1) {
                v = v * 10 + (rest[i] - '0');
            }
            if (i > 0 and i < rest.len and rest[i] == '.') break :blk @min(v, g_moe_expert_infos.len - 1);
        }
        break :blk 0; // sin prefix de capa (draft/shared): usa el slot 0
    };
    if (!g_moe_expert_used[li][idx]) {
        var patch = info.*;
        patch.n_dims = 2;
        patch.dims[0] = info.dims[0];
        patch.dims[1] = info.dims[1];
        patch.dims[2] = 0;
        patch.dims[3] = 0;
        g_moe_expert_infos[li][idx] = patch;
        g_moe_expert_used[li][idx] = true;
    }
    const patch_ptr: *const gguf.TensorInfo = &g_moe_expert_infos[li][idx];
    return QuantWeight.init(patch_ptr, all_bytes[0..expert_bytes]);
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
