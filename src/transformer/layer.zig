const std = @import("std");
const Tensor = @import("core").Tensor;
const matmul = @import("matmul");
const fa = @import("fa");
const kvcache = @import("kv_cache");
const norm = @import("norm");
const ffn = @import("ffn");
const rope_mod = @import("rope");
const gqa_mod = @import("gqa");
const gguf = @import("gguf");
const quant_weight = @import("quant_weight");
const cudaz = @import("cudaz");
const debugz = @import("debug");

/// Re-export de la capa híbrida (SSM + atención) para poder usarla desde
/// otros módulos que importan `transformer`.
pub const HybridLayer = @import("hybrid_layer").HybridLayer;
pub const HybridLayerParams = @import("hybrid_layer").HybridLayerParams;
// lane-f F1: PERF_SSM desglose por-etapa (reporte desde main.zig)
pub const ssmStageReport = @import("ssm").ssmStageReport;
// STUDY §5.2 (1.4): oráculo WY-chunk del prefill ΔNet.
pub const prefill_wy = @import("prefill_wy.zig");

const FlashAttention = fa.FlashAttention;
const FlashAttentionCpu = fa.FlashAttentionCpu;
const FlashAttentionConfig = fa.fa_config.FlashAttentionConfig;
const MatmulEngine = matmul.MatmulEngine;
const PrecisionMode = matmul.PrecisionMode;
const KVCacheManager = kvcache.KVCacheManager;
const KVCacheConfig = kvcache.KVCacheConfig;

/// Motor de atención: GPU (CUDA) si está disponible, CPU en caso contrario
pub const AttentionEngine = union(enum) {
    gpu: FlashAttention,
    cpu: FlashAttentionCpu,

    pub fn init(
        allocator: std.mem.Allocator,
        cfg: FlashAttentionConfig,
        ptx_path: []const u8,
        /// F-3 (lane-f): backend explícito del usuario (null = auto).
        /// `--backend cpu` no debe llamar cuInit: (a) el cuInit del driver
        /// 580 BLOQUEA/spinea cuando la GPU está contendida por otro
        /// proceso (deadlock intermitente del gate legacy — repro con
        /// coredump: main thread dentro de libcuda.so pese a backend
        /// cpu); (b) semántica "CPU real" del AGENTS.md.
        explicit_backend: ?matmul.Backend,
    ) AttentionEngine {
        // Auto (null) o gpu ⇒ puede tocar CUDA; cualquier backend CPU
        // explícito (.parallel/.simd/.tiled/.naive/.openblas) NO.
        if (explicit_backend != null and explicit_backend.? != .cublas and explicit_backend.? != .fp8_block) {
            return .{ .cpu = FlashAttentionCpu.init(allocator, cfg) };
        }
        if (cudaz.isCudaAvailable()) {
            return .{ .gpu = FlashAttention.init(allocator, cfg, ptx_path) catch {
                return .{ .cpu = FlashAttentionCpu.init(allocator, cfg) };
            } };
        }
        return .{ .cpu = FlashAttentionCpu.init(allocator, cfg) };
    }

    pub fn deinit(self: *AttentionEngine) void {
        switch (self.*) {
            .gpu => |*eng| eng.deinit(),
            .cpu => {},
        }
    }

    pub fn forward(self: *AttentionEngine, Q: Tensor(f16), K: Tensor(f16), V: Tensor(f16), O: *Tensor(f16)) !void {
        switch (self.*) {
            .gpu => |*eng| try eng.forward(Q, K, V, O),
            .cpu => |eng| try eng.forward(Q, K, V, O),
        }
    }

    /// 7.1b-B: devuelve la config del FA engine compartido.
    pub fn config(self: *const AttentionEngine) FlashAttentionConfig {
        return switch (self.*) {
            .gpu => |eng| eng.config,
            .cpu => |eng| eng.config,
        };
    }
};

pub const TransformerError = error{
    WeightFileNotFound,
    MatmulNotImplemented,
    CacheOverflow,
    InvalidPrecision,
    BackendMismatch,
    KvCacheNotSet,
};

pub const LayerPrecision = struct {
    compute: PrecisionMode,
    weights_on_gpu: bool,
    use_quantized: bool,
};

/// Capa Transformer completa con:
/// - Pre-LayerNorm (RMSNorm)
/// - Self-Attention (QKV proj + RoPE + FA + GQA)
/// - Residual
/// - Post-Attention RMSNorm
/// - FFN SwiGLU
/// - Residual
/// - KV-Cache cuantizado
pub const TransformerLayer = struct {
    allocator: std.mem.Allocator,
    layer_idx: usize,

    // Pesos de atención (host, transpuestos)
    w_q_t: ?Tensor(f16) = null,
    w_k_t: ?Tensor(f16) = null,
    w_v_t: ?Tensor(f16) = null,
    w_o_t: ?Tensor(f16) = null,

    // Pesos de FFN (host, transpuestos)
    w_gate_t: ?Tensor(f16) = null,
    w_up_t: ?Tensor(f16) = null,
    w_down_t: ?Tensor(f16) = null,

    // Pesos cuant crudos (dequant on-the-fly: bytes mmap, RAM barata).
    // Se materializan a *_t f16 lazy (ensureWeightsF16).
    w_q_qw: ?quant_weight.QuantWeight = null,
    w_k_qw: ?quant_weight.QuantWeight = null,
    w_v_qw: ?quant_weight.QuantWeight = null,
    w_o_qw: ?quant_weight.QuantWeight = null,
    w_gate_qw: ?quant_weight.QuantWeight = null,
    w_up_qw: ?quant_weight.QuantWeight = null,
    w_down_qw: ?quant_weight.QuantWeight = null,

    // Pesos de normalización
    attn_norm: ?Tensor(f32) = null,
    ffn_norm: ?Tensor(f32) = null,

    // BitNet b1.58 (lane-kvc P4): sub-norms tras attention (pre-wo) y tras
    // la activación FFN (pre-down). Referencia: models/bitnet.cpp unsloth
    // (attn_sub_norm tras build_attn antes de wo; ffn_sub_norm entre
    // silu(gate)*up y ffn_down, con LLM_FFN_SILU + LLM_FFN_PAR).
    attn_sub_norm: ?Tensor(f32) = null,
    ffn_sub_norm: ?Tensor(f32) = null,
    /// FFN silu(gate)*up PAR en vez de SwiGLU (BitNet b1.58, ref unsloth
    /// bitnet.cpp L146: LLM_FFN_SILU + LLM_FFN_PAR).
    is_bitnet: bool = false,

    /// BitNet feedback: recurrente ligero (gate escalar + proyección [d]→[d]).
    /// Solo activo si is_bitnet && bitnet_feedback_scale > 0.
    bitnet_feedback_scale: f32 = 0.0,
    bitnet_prev_state: ?[]f32 = null, // [hidden_dim] — s_{t-1}
    bitnet_w_proj: ?Tensor(f32) = null, // [hidden_dim, hidden_dim] from GGUF

    /// lane-kvc P4: scratch f32 por fila para el FFN de BitNet.
    /// silu(gate)*up puede salirse del rango cómodo de f16 con pesos
    /// ternarios escala ~1.2 (ver comentario de activación en forward):
    /// mantener el producto en f32 hasta el ffn_sub_norm evita inf/NaN.
    bitnet_ffn_f32: []f32 = &.{},

    // Pesos cuantizados (opcional)
    w_q_t_q: ?matmul.QuantizedTensor = null,
    w_k_t_q: ?matmul.QuantizedTensor = null,
    w_v_t_q: ?matmul.QuantizedTensor = null,
    w_o_t_q: ?matmul.QuantizedTensor = null,

    // Motores
    matmul_engine: MatmulEngine,
    /// 7.1b-B: puntero al FA engine compartido entre todas las capas
    /// (una sola instancia en vez de 28 → ~648MB ahorrados en GPU pinned+device).
    fa_engine: *AttentionEngine,

    // Configuración
    hidden_dim: usize,
    head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    intermediate_dim: usize,
    precision: LayerPrecision,
    rms_eps: f32 = 1e-5,
    rope_freq_base: f32 = 10000.0,

    // Buffers intermedios (position-major: [batch, N, heads*d])
    /// F-4 (lane-f): salida f16 de la proyección O — el GEMM homogéneo
    /// escribe aquí y el residual lo suma al stream f32 (1 redondeo por
    /// capa, sin acumulación compuesta en f16).
    o_scratch: Tensor(f16),
    /// F-4 (lane-f): salida f16 del FFN — mismo patrón que o_scratch.
    ffn_out_scratch: Tensor(f16),
    q_pos: Tensor(f16),
    k_pos: Tensor(f16),
    v_pos: Tensor(f16),
    attn_pos: Tensor(f16),
    ffn_gate: Tensor(f16),
    ffn_up: Tensor(f16),
    ffn_out: Tensor(f16),
    norm_buf: Tensor(f16),

    // KV-Cache
    kv_manager: ?*KVCacheManager = null,
    seq_id: u64 = 0,
    /// 7.2 (lane-f): seq_len del chunk en curso (set en forward) — usado
    /// por retrieveKvCache para concatenar historia + chunk local.
    chunk_len: usize = 0,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        layer_idx: usize,
        /// 7.1b-B: puntero al FA engine compartido (creado en pipeline.init).
        fa_engine: *AttentionEngine,
        hidden_dim: usize,
        precision: LayerPrecision,
        num_kv_heads: usize,
        intermediate_dim: usize,
        /// 7.1b: capacidad de buffers de activación (ubatch), separada de
        /// N del FA config (ctx completa). Reducir de N a ubatch_size
        /// ahorra ~5.2GB para 28 capas Llama-3.2-3B (216MB→27MB/layer).
        act_capacity: usize,
    ) !Self {
        const backend = if (precision.weights_on_gpu) matmul.Backend.cublas else matmul.Backend.auto;
        var engine = try MatmulEngine.init(allocator, backend, precision.compute);
        errdefer engine.deinit();

        const fa_config_val = fa_engine.config();
        const num_heads = fa_config_val.num_heads;
        const d = fa_config_val.d;
        const batch_size = fa_config_val.batch_size;

        // 7.1b: buffers de activación dimensionados a act_capacity (ubatch),
        // NO a N (ctx completa). En forward() se crean vistas prefix2d con
        // seq_len runtime sobre estos slabs más compactos.
        const A = act_capacity;
        var q_pos = try Tensor(f16).alloc(allocator, &.{ batch_size, A, num_heads * d });
        errdefer q_pos.deinit();
        var k_pos = try Tensor(f16).alloc(allocator, &.{ batch_size, A, num_kv_heads * d });
        errdefer k_pos.deinit();
        var v_pos = try Tensor(f16).alloc(allocator, &.{ batch_size, A, num_kv_heads * d });
        errdefer v_pos.deinit();
        var attn_pos = try Tensor(f16).alloc(allocator, &.{ batch_size, A, num_heads * d });
        errdefer attn_pos.deinit();
        var ffn_gate = try Tensor(f16).alloc(allocator, &.{ batch_size, A, intermediate_dim });
        errdefer ffn_gate.deinit();
        var ffn_up = try Tensor(f16).alloc(allocator, &.{ batch_size, A, intermediate_dim });
        errdefer ffn_up.deinit();
        var ffn_out = try Tensor(f16).alloc(allocator, &.{ batch_size, A, hidden_dim });
        errdefer ffn_out.deinit();
        var norm_buf = try Tensor(f16).alloc(allocator, &.{ batch_size, A, hidden_dim });
        errdefer norm_buf.deinit();
        var o_scratch = try Tensor(f16).alloc(allocator, &.{ batch_size, A, hidden_dim });
        errdefer o_scratch.deinit();
        var ffn_out_scratch = try Tensor(f16).alloc(allocator, &.{ batch_size, A, hidden_dim });
        errdefer ffn_out_scratch.deinit();

        return .{
            .allocator = allocator,
            .layer_idx = layer_idx,
            .matmul_engine = engine,
            .fa_engine = fa_engine,
            .hidden_dim = hidden_dim,
            .head_dim = d,
            .num_heads = num_heads,
            .num_kv_heads = num_kv_heads,
            .intermediate_dim = intermediate_dim,
            .precision = precision,
            .q_pos = q_pos,
            .k_pos = k_pos,
            .v_pos = v_pos,
            .attn_pos = attn_pos,
            .ffn_gate = ffn_gate,
            .ffn_up = ffn_up,
            .ffn_out = ffn_out,
            .norm_buf = norm_buf,
            .o_scratch = o_scratch,
            .ffn_out_scratch = ffn_out_scratch,
        };
    }

    pub fn deinit(self: *Self) void {
        // 7.1b-B: NO liberar fa_engine — es compartido, lo libera pipeline.
        self.q_pos.deinit();
        self.k_pos.deinit();
        self.v_pos.deinit();
        self.attn_pos.deinit();
        self.ffn_gate.deinit();
        self.ffn_up.deinit();
        self.ffn_out.deinit();
        self.norm_buf.deinit();
        self.o_scratch.deinit();
        self.ffn_out_scratch.deinit();
        self.matmul_engine.deinit();

        if (self.w_q_t) |*w| w.deinit();
        if (self.w_k_t) |*w| w.deinit();
        if (self.w_v_t) |*w| w.deinit();
        if (self.w_o_t) |*w| w.deinit();
        if (self.w_gate_t) |*w| w.deinit();
        if (self.w_up_t) |*w| w.deinit();
        if (self.w_down_t) |*w| w.deinit();
        if (self.attn_norm) |*w| w.deinit();
        if (self.ffn_norm) |*w| w.deinit();
        if (self.attn_sub_norm) |*w| w.deinit(); // lane-kvc P4: BitNet
        if (self.ffn_sub_norm) |*w| w.deinit(); // lane-kvc P4: BitNet
        if (self.bitnet_ffn_f32.len > 0) self.allocator.free(self.bitnet_ffn_f32); // lane-kvc P4
        if (self.bitnet_prev_state) |s| self.allocator.free(s);
        if (self.bitnet_w_proj) |*w| w.deinit();

        if (self.w_q_t_q) |*w| w.deinit();
        if (self.w_k_t_q) |*w| w.deinit();
        if (self.w_v_t_q) |*w| w.deinit();
        if (self.w_o_t_q) |*w| w.deinit();
    }

    pub fn loadWeights(self: *Self, io: std.Io, checkpoint_dir: []const u8) !void {
        const base = try std.fmt.allocPrint(self.allocator, "{s}/layer.{d}.", .{ checkpoint_dir, self.layer_idx });
        defer self.allocator.free(base);

        self.w_q_t = try loadWeightFile(io, self.allocator, base, "self_attn.q_proj.weight_t");
        self.w_k_t = try loadWeightFile(io, self.allocator, base, "self_attn.k_proj.weight_t");
        self.w_v_t = try loadWeightFile(io, self.allocator, base, "self_attn.v_proj.weight_t");
        self.w_o_t = try loadWeightFile(io, self.allocator, base, "self_attn.o_proj.weight_t");
        self.w_gate_t = try loadWeightFile(io, self.allocator, base, "mlp.gate_proj.weight_t");
        self.w_up_t = try loadWeightFile(io, self.allocator, base, "mlp.up_proj.weight_t");
        self.w_down_t = try loadWeightFile(io, self.allocator, base, "mlp.down_proj.weight_t");

        // Cargar norm weights (f32)
        self.attn_norm = try loadWeightFileF32(io, self.allocator, base, "input_layernorm.weight");
        self.ffn_norm = try loadWeightFileF32(io, self.allocator, base, "post_attention_layernorm.weight");

        if (self.precision.use_quantized) {
            const qcfg = matmul.QuantConfig{ .bits = 8, .symmetric = true, .per_channel = true, .group_size = 0 };
            self.w_q_t_q = try matmul.quantizeInt8PerChannel(self.allocator, self.w_q_t.?, qcfg);
            self.w_k_t_q = try matmul.quantizeInt8PerChannel(self.allocator, self.w_k_t.?, qcfg);
            self.w_v_t_q = try matmul.quantizeInt8PerChannel(self.allocator, self.w_v_t.?, qcfg);
            self.w_o_t_q = try matmul.quantizeInt8PerChannel(self.allocator, self.w_o_t.?, qcfg);
        }
    }

    /// Carga los pesos de esta capa desde un GGUF (dequant a f16).
    /// Los pesos GGUF son matrices [out, in] row-major (dims[0]=in, dims[1]=out),
    /// que coincide con el layout que espera linearProjection (trans_b=true).
    /// Soporta los alias comunes: attn_output/attn_o, ffn_gate/mlp.gate_proj, etc.
    pub fn loadWeightsFromGguf(self: *Self, g: *const gguf.GgufFile) !void {
        const prefix = try std.fmt.allocPrint(self.allocator, "blk.{d}.", .{self.layer_idx});
        defer self.allocator.free(prefix);

        const q_names = [_][]const u8{ "attn_q.weight", "wq.weight" };
        const k_names = [_][]const u8{ "attn_k.weight", "wk.weight" };
        const v_names = [_][]const u8{ "attn_v.weight", "wv.weight" };
        const o_names = [_][]const u8{ "attn_output.weight", "attn_o.weight", "wo.weight" };
        const gate_names = [_][]const u8{ "ffn_gate.weight", "mlp.gate_proj.weight", "feed_forward.w1.weight" };
        const up_names = [_][]const u8{ "ffn_up.weight", "mlp.up_proj.weight", "feed_forward.w3.weight" };
        const down_names = [_][]const u8{ "ffn_down.weight", "mlp.down_proj.weight", "feed_forward.w2.weight" };
        const attn_norm_names = [_][]const u8{ "attn_norm.weight", "input_layernorm.weight" };
        const ffn_norm_names = [_][]const u8{ "ffn_norm.weight", "post_attention_layernorm.weight" };

        // Dequant on-the-fly (patrón llama.cpp / parity con hybrid_path):
        // en load solo guardamos los QuantWeight crudos (bytes mmap, RAM =
        // archivo cuant). Los Tensor(f16) se materializan LAZY en el primer
        // forward de la capa (o por ensureWeightsF16 del caller). Elimina el
        // pico f16+f32buf total del load de modelos densos legacy (Llama-3B
        // Q8_0: -8GB de pico; con --layer-stream solo N capas materializadas).
        self.w_q_qw = try loadGgufQuantWeight(self.allocator, g, prefix, &q_names);
        self.w_k_qw = try loadGgufQuantWeight(self.allocator, g, prefix, &k_names);
        self.w_v_qw = try loadGgufQuantWeight(self.allocator, g, prefix, &v_names);
        self.w_o_qw = try loadGgufQuantWeight(self.allocator, g, prefix, &o_names);
        self.w_gate_qw = try loadGgufQuantWeight(self.allocator, g, prefix, &gate_names);
        self.w_up_qw = try loadGgufQuantWeight(self.allocator, g, prefix, &up_names);
        self.w_down_qw = try loadGgufQuantWeight(self.allocator, g, prefix, &down_names);

        self.attn_norm = try loadGgufNormF32(self.allocator, g, prefix, &attn_norm_names);
        self.ffn_norm = try loadGgufNormF32(self.allocator, g, prefix, &ffn_norm_names);

        // BitNet b1.58 (lane-kvc P4): sub-norms opcionales — solo presentes
        // en modelos bitnet-b1.58 (121 tensores f32 en el GGUF real).
        // 1D (n_dims==1): el contract de loadGgufNormF32 exige 2D, así que
        // los cargamos directo con dequantF32 (son f32 planos).
        if (self.is_bitnet) {
            self.attn_sub_norm = try loadGgufNorm1DOptF32(self.allocator, g, prefix, "attn_sub_norm.weight");
            self.ffn_sub_norm = try loadGgufNorm1DOptF32(self.allocator, g, prefix, "ffn_sub_norm.weight");
            // BitNet feedback: lightweight recurrent gate (optional, GGUF rlt.* keys)
            if (g.getMeta("rlt.bitnet_feedback_scale")) |v| {
                if (v.asF32()) |scale| {
                    self.bitnet_feedback_scale = scale;
                    self.bitnet_w_proj = try loadBitNetFeedbackWeight(self.allocator, g, prefix);
                    self.bitnet_prev_state = try self.allocator.alloc(f32, self.hidden_dim);
                    @memset(self.bitnet_prev_state.?, 0);
                }
            }
        }
    }

    /// Forward completo: PreNorm -> Attn -> Residual -> PostNorm -> FFN -> Residual
    pub fn forward(
        self: *Self,
        /// F-4 (lane-f): residual stream en f32 — el acumulador capa a
        /// capa ya no redondea a f16 en cada suma (híbrido f32 = golden;
        /// legacy f16 = 1.8× PPL + overflow BitNet). Proyecciones/KV
        /// siguen f16: casts solo en los bordes de GEMM.
        hidden_state: Tensor(f32),
        output: *Tensor(f32),
        position: usize,
        _is_prefill: bool,
    ) !void {
        _ = _is_prefill;
        // Dequant on-the-fly: materializar f16 lazy (cuant-residente load).
        try ensureWeightsF16(self);
        const batch_size = hidden_state.shape[0];
        const seq_len = hidden_state.shape[1];

        // Reinterpretar como 2D para GEMM
        const X_2d = try hidden_state.reshape(&[_]usize{ batch_size * seq_len, self.hidden_dim });
        defer {
            if (X_2d.allocator) |a| {
                a.free(X_2d.shape);
                a.free(X_2d.strides);
            }
        }

        // === BitNet feedback: lightweight recurrent merge ===
        // Only for single-token decode (N=1) with feedback enabled.
        if (seq_len == 1 and self.is_bitnet and self.bitnet_feedback_scale > 0) {
            self.mergeBitNetFeedback(X_2d.data);
        }

        // === 1. Pre-Attention RMSNorm ===
        // 7.1c: rmsNorm exige output.shape == input.shape (seq runtime),
        // pero el slab norm_buf tiene capacidad N del FA config. Creamos
        // una vista 3D [batch, seq, hidden] del prefix — sin reshape
        // exacto (ese assert era el crash seq≠emb_len).
        const nb_shape = try self.allocator.alloc(usize, 3);
        const nb_strides = try self.allocator.alloc(usize, 3);
        defer {
            self.allocator.free(nb_shape);
            self.allocator.free(nb_strides);
        }
        nb_shape[0] = batch_size;
        nb_shape[1] = seq_len;
        nb_shape[2] = self.hidden_dim;
        nb_strides[0] = seq_len * self.hidden_dim;
        nb_strides[1] = self.hidden_dim;
        nb_strides[2] = 1;

        if (self.attn_norm) |gamma| {
            // 7.1c: rmsNorm exige output.shape == input.shape (3-D seq
            // runtime). Vista del prefix del slab norm_buf (capacidad N).
            // F-4: stream f32 → slab f16 (entrada de las proyecciones QKV)
            // con acumulador f32 — rmsNormIo, sin round-trip del stream.
            var norm_out = self.norm_buf.view(nb_shape, nb_strides, 0);
            norm.rmsNormIo(f32, f16, f32, hidden_state, gamma, self.rms_eps, &norm_out);
        } else {
            // Sin gamma (bitnet sub_norm path lo cubre aparte): cast plano
            // f32→f16 al slab — uniforma tipos para las proyecciones.
            const n_cast = batch_size * seq_len * self.hidden_dim;
            const nb0 = self.norm_buf.view(nb_shape, nb_strides, 0);
            for (hidden_state.data[0..n_cast], nb0.data[0..n_cast]) |s, *d| d.* = @floatCast(s);
        }
        const norm_prefix = self.norm_buf.view(nb_shape, nb_strides, 0);
        const norm_2d = try norm_prefix.reshape(&[_]usize{ batch_size * seq_len, self.hidden_dim });
        defer {
            if (norm_2d.allocator) |a| {
                a.free(norm_2d.shape);
                a.free(norm_2d.strides);
            }
        }

        // 7.2 (lane-f): breadcrumbs de stage — sonda de paridad numérica
        // por capa (gated DEBUG_LEVEL=3). Técnica 7.1b: la divergencia
        // se localiza por el salto de max/sum entre capas.
        const dbg_stage = debugz.dbg.at(.trace);
        if (dbg_stage) debugz.dbg.printLevel(.trace, "[layer] {d} in: max={d:.4} sum={d:.4}\n", .{ self.layer_idx, debugz.maxAbsF32(hidden_state.data), debugz.sumAbsF32(hidden_state.data) });
        if (dbg_stage) debugz.dbg.printLevel(.trace, "[layer] {d} norm: max={d:.4}\n", .{ self.layer_idx, debugz.maxAbsF16Real(norm_prefix.data[0..@min(norm_prefix.data.len, seq_len * self.hidden_dim)]) });

        // === 2. Proyecciones Q, K, V ===
        if (self.precision.use_quantized) {
            try self.projectQQuantized(norm_2d);
            try self.projectKQuantized(norm_2d);
            try self.projectVQuantized(norm_2d);
        } else {
            if (debugz.dbg.at(.trace)) debugz.dbg.printLevel(.trace, "[prefill_chunk] shapes: X=[{d},{d}] wq=[{d},{d}] wk=[{d},{d}] Yq_outdim={d} nh={d} nkv={d} hd={d} kpos=[{d},{d},{d}]\n", .{
                norm_2d.shape[0],                  norm_2d.shape[1],
                self.w_q_t.?.shape[0],             self.w_q_t.?.shape[1],
                self.w_k_t.?.shape[0],             self.w_k_t.?.shape[1],
                self.num_kv_heads * self.head_dim, self.num_heads,
                self.num_kv_heads,                 self.head_dim,
                self.k_pos.shape[0],               self.k_pos.shape[1],
                self.k_pos.shape[2],
            });
            try self.projectQ(norm_2d);
            try self.projectK(norm_2d);
            try self.projectV(norm_2d);
        }

        // === 3. RoPE (on head-major views) ===
        // Create head-major views first, then apply RoPE
        var q_shape = try self.allocator.alloc(usize, 4);
        q_shape[0] = batch_size;
        q_shape[1] = self.num_heads;
        q_shape[2] = seq_len;
        q_shape[3] = self.head_dim;
        var q_strides = try self.allocator.alloc(usize, 4);
        q_strides[0] = seq_len * self.num_heads * self.head_dim;
        q_strides[1] = self.head_dim;
        q_strides[2] = self.num_heads * self.head_dim;
        q_strides[3] = 1;
        var q_hm = self.q_pos.view(q_shape, q_strides, 0);

        var k_shape = try self.allocator.alloc(usize, 4);
        k_shape[0] = batch_size;
        k_shape[1] = self.num_kv_heads;
        k_shape[2] = seq_len;
        k_shape[3] = self.head_dim;
        var k_strides = try self.allocator.alloc(usize, 4);
        k_strides[0] = seq_len * self.num_kv_heads * self.head_dim;
        k_strides[1] = self.head_dim;
        k_strides[2] = self.num_kv_heads * self.head_dim;
        k_strides[3] = 1;
        var k_hm = self.k_pos.view(k_shape, k_strides, 0);

        rope_mod.applyRoPE(f16, &q_hm, &k_hm, position, self.head_dim, self.rope_freq_base, .auto);
        if (dbg_stage) debugz.dbg.printLevel(.trace, "[layer] {d} qkv+rope: q max={d:.4} k max={d:.4} v max={d:.4}\n", .{ self.layer_idx, debugz.maxAbsF16Real(q_hm.data), debugz.maxAbsF16Real(k_hm.data), debugz.maxAbsF16Real(self.v_pos.data) });

        // === 4. KV-Cache ===
        // 7.2 (lane-f): el chunk se guarda ANTES del retrieve — pero el
        // current_len NO avanza hasta que el pipeline lo haga tras el
        // loop de capas. retrieveKvCache concatena historia (manager,
        // [0, current_len)) + chunk local (slabs).
        self.chunk_len = seq_len;
        if (self.kv_manager) |mgr| {
            try self.storeKvCache(mgr, seq_len);
        }

        // === 5. Recuperar K/V full ===
        var k_full: Tensor(f16) = undefined;
        var v_full: Tensor(f16) = undefined;
        var k_full_owned = false;
        var v_full_owned = false;

        if (self.kv_manager) |mgr| {
            const hist_len = try mgr.getSequenceLen(self.seq_id);
            const full_len = hist_len + seq_len; // historia + chunk actual
            k_full = try Tensor(f16).alloc(self.allocator, &.{ batch_size, self.num_kv_heads, full_len, self.head_dim });
            v_full = try Tensor(f16).alloc(self.allocator, &.{ batch_size, self.num_kv_heads, full_len, self.head_dim });
            k_full_owned = true;
            v_full_owned = true;
            try self.retrieveKvCache(mgr, &k_full, &v_full);
        } else {
            // k_pos/v_pos are position-major [batch, N, kv_heads*d]; create head-major views
            var k_shape2 = try self.allocator.alloc(usize, 4);
            k_shape2[0] = batch_size;
            k_shape2[1] = self.num_kv_heads;
            k_shape2[2] = seq_len;
            k_shape2[3] = self.head_dim;
            var k_strides2 = try self.allocator.alloc(usize, 4);
            k_strides2[0] = seq_len * self.num_kv_heads * self.head_dim;
            k_strides2[1] = self.head_dim;
            k_strides2[2] = self.num_kv_heads * self.head_dim;
            k_strides2[3] = 1;
            const k_hm2 = self.k_pos.view(k_shape2, k_strides2, 0);
            defer {
                self.allocator.free(k_shape2);
                self.allocator.free(k_strides2);
            }

            var v_shape2 = try self.allocator.alloc(usize, 4);
            v_shape2[0] = batch_size;
            v_shape2[1] = self.num_kv_heads;
            v_shape2[2] = seq_len;
            v_shape2[3] = self.head_dim;
            var v_strides2 = try self.allocator.alloc(usize, 4);
            v_strides2[0] = seq_len * self.num_kv_heads * self.head_dim;
            v_strides2[1] = self.head_dim;
            v_strides2[2] = self.num_kv_heads * self.head_dim;
            v_strides2[3] = 1;
            const v_hm2 = self.v_pos.view(v_shape2, v_strides2, 0);
            defer {
                self.allocator.free(v_shape2);
                self.allocator.free(v_strides2);
            }

            k_full = try Tensor(f16).alloc(self.allocator, &.{ batch_size, self.num_kv_heads, seq_len, self.head_dim });
            v_full = try Tensor(f16).alloc(self.allocator, &.{ batch_size, self.num_kv_heads, seq_len, self.head_dim });
            k_full_owned = true;
            v_full_owned = true;
            // 7.1c: las views head-major son strided (k_hm2.data es el
            // backing completo del slab, N·kv·d ≠ seq·kv·d) — copiar por
            // strides de la view, no @memcpy del backing.
            for (0..batch_size) |b| {
                for (0..self.num_kv_heads) |h| {
                    for (0..seq_len) |p| {
                        const src_off = k_hm2.offset + b * k_hm2.strides[0] + h * k_hm2.strides[1] + p * k_hm2.strides[2];
                        const dst_off = ((b * self.num_kv_heads + h) * seq_len + p) * self.head_dim;
                        @memcpy(k_full.data[dst_off .. dst_off + self.head_dim], k_hm2.data[src_off .. src_off + self.head_dim]);
                    }
                }
            }
            for (0..batch_size) |b| {
                for (0..self.num_kv_heads) |h| {
                    for (0..seq_len) |p| {
                        const src_off = v_hm2.offset + b * v_hm2.strides[0] + h * v_hm2.strides[1] + p * v_hm2.strides[2];
                        const dst_off = ((b * self.num_kv_heads + h) * seq_len + p) * self.head_dim;
                        @memcpy(v_full.data[dst_off .. dst_off + self.head_dim], v_hm2.data[src_off .. src_off + self.head_dim]);
                    }
                }
            }
        }

        // === 6. Expandir GQA si es necesario ===
        var k_expanded: Tensor(f16) = undefined;
        var v_expanded: Tensor(f16) = undefined;
        var k_exp_owned = false;
        var v_exp_owned = false;

        if (self.num_kv_heads < self.num_heads) {
            k_expanded = try gqa_mod.expandGqaFallback(self.allocator, k_full, self.num_heads);
            v_expanded = try gqa_mod.expandGqaFallback(self.allocator, v_full, self.num_heads);
            k_exp_owned = true;
            v_exp_owned = true;
        } else {
            k_expanded = k_full;
            v_expanded = v_full;
        }

        // === 7. FlashAttention (head-major) ===
        // 7.2 (lane-f): el FA consume la HISTORIA KV completa recuperada del
        // manager (k_expanded/v_expanded, ya expandidas a num_heads via GQA)
        // — antes se le pasaban k_hm/v_hm (solo los tokens actuales del
        // slab), la atención no veía nada del pasado. Q sí es la vista del
        // slab (solo tokens actuales): FlashAttentionCpu resuelve el
        // solape causal con kv_offset = T_kv - T_q.
        defer {
            self.allocator.free(q_shape);
            self.allocator.free(q_strides);
            self.allocator.free(k_shape);
            self.allocator.free(k_strides);
        }

        // Temporary head-major output, then transpose to position-major
        var attn_hm = try Tensor(f16).alloc(self.allocator, &.{ batch_size, self.num_heads, seq_len, self.head_dim });
        defer attn_hm.deinit();

        try self.fa_engine.forward(q_hm, k_expanded, v_expanded, &attn_hm);
        if (dbg_stage) debugz.dbg.printLevel(.trace, "[layer] {d} attn: max={d:.4} sum={d:.4}\n", .{ self.layer_idx, debugz.maxAbsF16Real(attn_hm.data), debugz.sumAbsF16Real(attn_hm.data) });

        // Transpose attn_hm [batch, heads, seq, d] -> attn_pos [batch, seq, heads*d] (position-major)
        for (0..batch_size) |b| {
            for (0..seq_len) |p| {
                for (0..self.num_heads) |h| {
                    for (0..self.head_dim) |k| {
                        const src = attn_hm.data[((b * self.num_heads + h) * seq_len + p) * self.head_dim + k];
                        const dst = (b * seq_len + p) * self.num_heads * self.head_dim + h * self.head_dim + k;
                        self.attn_pos.data[dst] = src;
                    }
                }
            }
        }

        if (k_exp_owned) k_expanded.deinit();
        if (v_exp_owned) v_expanded.deinit();
        if (k_full_owned) k_full.deinit();
        if (v_full_owned) v_full.deinit();

        // === 7.5 BitNet attn_sub_norm (lane-kvc P4) ===
        // Referencia bitnet.cpp: build_attn → attn_sub_norm (RMSNorm) → wo.
        // En modelos no-bitnet el campo es null: coste cero.
        if (self.attn_sub_norm) |gamma| {
            // In-place sobre attn_pos [batch, seq_runtime, heads*d] — el
            // prefix escrito por la transposición FA (paso previo).
            var sub_out = self.attn_pos;
            norm.rmsNorm(f16, f32, sub_out, gamma, self.rms_eps, &sub_out);
        }
        // === 8. Proyección de salida O ===
        if (self.precision.use_quantized) {
            try self.projectOutQuantized(seq_len);
        } else {
            try self.projectOut(seq_len);
        }

        // === 9. Residual connection (Attn) ===
        // F-4 (lane-f): stream f32 — la proyección O (f16, slab) se suma
        // al stream SIN redondear el acumulador.
        {
            const n_out = batch_size * seq_len * self.hidden_dim;
            for (output.data[0..n_out], self.o_scratch.data[0..n_out], hidden_state.data[0..n_out]) |*o, w, h| {
                o.* = @as(f32, @floatCast(w)) + h;
            }
            // F-4: NOF32STREAM=1 — el legacy redondeaba el acumulador en
            // AMBOS residuales (attn + FFN); reproducir el primero aquí
            // (el FFN legacy leía x+attn ya redondeado a f16).
            if (debugz.dbg.no_f32_stream) {
                for (output.data[0..n_out]) |*o| o.* = @floatCast(@as(f16, @floatCast(o.*)));
            }
        }

        // === 10. Post-Attention RMSNorm ===
        // 7.1c: idem paso 1 — vista 3D runtime del prefix de norm_buf
        // (capacidad N) para que shape == input.shape (assert rmsNorm).
        // F-4: la norma escribe al slab f16 (norm_buf) — el GEMM del FFN
        // consume el slab; el stream f32 SOLO lo tocan los residuales.
        // Sin gamma: cast plano f32→f16 al mismo slab (uniforma tipos).
        var normffn_prefix: Tensor(f16) = undefined;
        if (self.ffn_norm) |gamma| {
            var norm_out = self.norm_buf.view(nb_shape, nb_strides, 0);
            norm.rmsNormIo(f32, f16, f32, output.*, gamma, self.rms_eps, &norm_out);
            normffn_prefix = self.norm_buf.view(nb_shape, nb_strides, 0);
        } else {
            const n_cast = batch_size * seq_len * self.hidden_dim;
            const nb = self.norm_buf.view(nb_shape, nb_strides, 0);
            for (output.data[0..n_cast], nb.data[0..n_cast]) |s, *d| d.* = @floatCast(s);
            normffn_prefix = nb;
        }

        // === 11. FFN SwiGLU ===
        const attn_res_2d = try normffn_prefix.reshape(&[_]usize{ batch_size * seq_len, self.hidden_dim });
        defer {
            if (attn_res_2d.allocator) |a| {
                a.free(attn_res_2d.shape);
                a.free(attn_res_2d.strides);
            }
        }
        if (dbg_stage) debugz.dbg.printLevel(.trace, "[layer] {d} attn+res: max={d:.4}\n", .{ self.layer_idx, debugz.maxAbsF16Real(attn_res_2d.data[0..@min(attn_res_2d.data.len, batch_size * seq_len * self.hidden_dim)]) });
        if (dbg_stage) debugz.dbg.printLevel(.trace, "[pipeline] {d} stream f32 post-attn-res: max={d:.4} sum={d:.4}\n", .{ self.layer_idx, debugz.maxAbsF32(output.data[0 .. batch_size * seq_len * self.hidden_dim]), debugz.sumAbsF32(output.data[0 .. batch_size * seq_len * self.hidden_dim]) });

        // 7.1c: prefix runtime de los slabs FFN (capacidad N) — 3 views 2D.
        const fo_shape = try self.allocator.alloc(usize, 2);
        const fo_strides = try self.allocator.alloc(usize, 2);
        const gt_shape = try self.allocator.alloc(usize, 2);
        const gt_strides = try self.allocator.alloc(usize, 2);
        const up_shape = try self.allocator.alloc(usize, 2);
        const up_strides = try self.allocator.alloc(usize, 2);
        defer {
            self.allocator.free(fo_shape);
            self.allocator.free(fo_strides);
            self.allocator.free(gt_shape);
            self.allocator.free(gt_strides);
            self.allocator.free(up_shape);
            self.allocator.free(up_strides);
        }
        // F-4 (lane-f): el FFN escribe al slab ffn_out_scratch — el
        // residual f32 lo suma al stream tras el down.
        var ffn_out_2d = prefix2d(self.ffn_out_scratch, batch_size * seq_len, self.hidden_dim, fo_shape, fo_strides);
        var gate_2d = prefix2d(self.ffn_gate, batch_size * seq_len, self.intermediate_dim, gt_shape, gt_strides);
        var up_2d = prefix2d(self.ffn_up, batch_size * seq_len, self.intermediate_dim, up_shape, up_strides);

        if (self.is_bitnet) {
            // BitNet b1.58 FFN (lane-kvc P4, ref bitnet.cpp unsloth L146:
            // LLM_FFN_SILU + LLM_FFN_PAR): inner = silu(gate) * up →
            // ffn_sub_norm (RMSNorm) → down. El sub_norm vive ENTRE la
            // activación y down.
            try self.matmul_engine.linearProjection(f16, attn_res_2d, self.w_gate_t.?, &gate_2d);
            try self.matmul_engine.linearProjection(f16, attn_res_2d, self.w_up_t.?, &up_2d);
            // El producto vive en f32 (ver comentario de bitnet_ffn_f32)
            // incluso tras cambiar la activación: magnitudes O(1e3) tras la
            // multiplicación siguen fuera del rango cómodo de f16.
            const inter = self.intermediate_dim;
            if (self.bitnet_ffn_f32.len < inter) {
                if (self.bitnet_ffn_f32.len > 0) self.allocator.free(self.bitnet_ffn_f32);
                self.bitnet_ffn_f32 = try self.allocator.alloc(f32, inter);
            }
            const scratch = self.bitnet_ffn_f32[0..inter];
            const rows = batch_size * seq_len;
            for (0..rows) |r| {
                const off = r * inter;
                for (0..inter) |j| {
                    const g: f32 = @floatCast(gate_2d.data[off + j]);
                    const u: f32 = @floatCast(up_2d.data[off + j]);
                    // BitNet b1.58 (ref bitnet.cpp unsloth L146):
                    // build_ffn(..., LLM_FFN_SILU, LLM_FFN_PAR) — la
                    // activación es SiLU (g·sigmoid(g)), NO squared_relu
                    // (relu² explotaba el residual: gate~40 ⇒ 1600·up).
                    const silu = g * (1.0 / (1.0 + @exp(-g)));
                    scratch[j] = silu * u;
                }
                if (self.ffn_sub_norm) |gamma| {
                    // RMSNorm f32 in-place (misma semántica que norm.rmsNorm:
                    // x * rsqrt(mean(x²) + eps) * gamma), pero sin el
                    // roundtrip f16 que desbordaba.
                    var mean_sq: f32 = 0.0;
                    for (scratch) |v| mean_sq += v * v;
                    mean_sq /= @as(f32, @floatFromInt(inter));
                    const scale = 1.0 / @sqrt(mean_sq + self.rms_eps);
                    for (scratch, 0..) |v, j| scratch[j] = v * scale * gamma.data[j];
                }
                for (0..inter) |j| gate_2d.data[off + j] = @floatCast(scratch[j]);
            }
            if (dbg_stage) {
                var mx: f32 = 0.0;
                for (gate_2d.data[0..@min(gate_2d.data.len, rows * inter)]) |v| mx = @max(mx, @abs(@as(f32, @floatCast(v))));
                debugz.dbg.printLevel(.trace, "[layer] {d} bitnet ffn post-subnorm: max={d:.4}\n", .{ self.layer_idx, mx });
            }
            try self.matmul_engine.linearProjection(f16, gate_2d, self.w_down_t.?, &ffn_out_2d);
        } else {
            try ffn.swiGluForward(
                &self.matmul_engine,
                f16,
                attn_res_2d,
                self.w_gate_t.?,
                self.w_up_t.?,
                self.w_down_t.?,
                &gate_2d,
                &up_2d,
                &ffn_out_2d,
            );
        }

        // === 12. Residual connection (FFN) ===
        // F-4 (lane-f): el residual acumula en f32 — la mitigación
        // rescale-pow2 de BitNet (lane-kvc P4, overflow f16 del stream:
        // 883→55680→inf en 8 capas) queda obsoleta: f32 max 3.4e38 no
        // satura con crecimiento geométrico de 30 capas. El path bitnet
        // f32 del FFN (bitnet_ffn_f32) se MANTIENE — el overflow del
        // producto silu·up era ANTES del residual (gate~44 ⇒ 1e5 > 65504).
        const n_out = batch_size * seq_len * self.hidden_dim;
        if (dbg_stage) debugz.dbg.printLevel(.trace, "[layer] {d} ffn_out: max={d:.4}\n", .{ self.layer_idx, debugz.maxAbsF16Real(self.ffn_out_scratch.data[0..n_out]) });

        for (output.data[0..n_out], self.ffn_out_scratch.data[0..n_out]) |*o, f| {
            o.* += f;
        }
        // F-4 (lane-f): NOF32STREAM=1 — A/B exacto contra el stream f16
        // clásico: el mismo redondeo post-residual que el código
        // pre-53a37ab (ver roundStreamF16 en pipeline.zig).
        if (debugz.dbg.no_f32_stream) {
            for (output.data[0..n_out]) |*o| o.* = @floatCast(@as(f16, @floatCast(o.*)));
        }
        // BitNet feedback: store final hidden state for next token
        if (seq_len == 1 and self.is_bitnet and self.bitnet_prev_state != null) {
            self.storeBitNetState(output.data[0..n_out]);
        }
        if (dbg_stage) debugz.dbg.printLevel(.trace, "[layer] {d} out: max={d:.4} sum={d:.4}\n", .{ self.layer_idx, debugz.maxAbsF32(output.data[0..n_out]), debugz.sumAbsF32(output.data[0..n_out]) });
    }

    // ─── KV-Cache helpers ───
    fn storeKvCache(self: *Self, mgr: *KVCacheManager, seq_len: usize) !void {
        // 7.2 (lane-f): head_idx per-KV-head (slots [layer][num_heads], con
        // num_heads >= num_kv_heads siempre). Antes se mapeaba
        // q_head_for_kv = kv_h * (num_heads/num_kv_heads) — con GQA real
        // (24q/8kv) solo se escribían los slots 0,3,6.. y el retrieve
        // per-q-head caía en SlotEmpty.
        //
        // 7.2: chunk multi-token — appendTokens escribe TODO el chunk en
        // offset = seq.current_len (compartido por todas las capas). El
        // SIN advanceSequence aquí: el pipeline avanza seq_len UNA vez
        // DESPUÉS del loop de capas (antes el advance por capa ×28 hacía
        // que la capa 1 escribiera en offset 6 tras un prefill de 6).
        for (0..self.num_kv_heads) |kv_h| {
            const k_chunk = try self.allocator.alloc(f16, seq_len * self.head_dim);
            defer self.allocator.free(k_chunk);
            const v_chunk = try self.allocator.alloc(f16, seq_len * self.head_dim);
            defer self.allocator.free(v_chunk);

            // Copiar el chunk del slab position-major [b, N, kv_heads*d]
            for (0..seq_len) |pos| {
                const src_off = (pos * self.num_kv_heads + kv_h) * self.head_dim;
                const dst_off = pos * self.head_dim;
                @memcpy(k_chunk[dst_off .. dst_off + self.head_dim], self.k_pos.data[src_off .. src_off + self.head_dim]);
                @memcpy(v_chunk[dst_off .. dst_off + self.head_dim], self.v_pos.data[src_off .. src_off + self.head_dim]);
            }

            try mgr.appendTokensF16(self.seq_id, @as(u32, @intCast(self.layer_idx)), @as(u32, @intCast(kv_h)), k_chunk, v_chunk);
        }
    }

    fn retrieveKvCache(self: *Self, mgr: *KVCacheManager, out_k: *Tensor(f16), out_v: *Tensor(f16)) !void {
        // 7.2 (lane-f): retrieve per-kv-head (simétrico al store).
        //
        // El manager contiene la historia PREVIA al chunk actual (el
        // current_len avanza en el pipeline tras el loop de capas): los
        // tokens del chunk en curso NO están en el manager. La historia
        // completa = historia_previa + chunk local (k_pos/v_pos).
        const hist_len = try mgr.getSequenceLen(self.seq_id);
        const full_len = hist_len + self.chunk_len;
        std.debug.assert(out_k.shape[2] == full_len);

        // 7.2: out_k/out_v son [batch, num_kv_heads, full_len, head_dim]
        // HEAD-MAJOR densos — los lee expandGqaFallback y el FA como
        // (h*full_len + t)*head_dim. El dst de antes era position-major
        // ((pos*nkv + kv_h)*d): heads y posiciones revueltas ⇒ atención
        // a datos aleatorios (firma "solo ve el último token").
        for (0..self.num_kv_heads) |kv_h| {
            var k_head = try self.allocator.alloc(f16, hist_len * self.head_dim);
            defer self.allocator.free(k_head);
            var v_head = try self.allocator.alloc(f16, hist_len * self.head_dim);
            defer self.allocator.free(v_head);

            if (hist_len > 0) {
                try mgr.retrieveForAttention(self.seq_id, @as(u32, @intCast(self.layer_idx)), @as(u32, @intCast(kv_h)), k_head, v_head);
            }

            const head_base = kv_h * full_len * self.head_dim;
            // historia (manager, head-contigua) + chunk local (slabs
            // position-major [b, N, kv_heads*d])
            for (0..hist_len) |pos| {
                const src_offset = pos * self.head_dim;
                const dst_offset = head_base + pos * self.head_dim;
                @memcpy(out_k.data[dst_offset .. dst_offset + self.head_dim], k_head[src_offset .. src_offset + self.head_dim]);
                @memcpy(out_v.data[dst_offset .. dst_offset + self.head_dim], v_head[src_offset .. src_offset + self.head_dim]);
            }
            for (0..self.chunk_len) |pos| {
                const src_offset = (pos * self.num_kv_heads + kv_h) * self.head_dim;
                const dst_offset = head_base + (hist_len + pos) * self.head_dim;
                @memcpy(out_k.data[dst_offset .. dst_offset + self.head_dim], self.k_pos.data[src_offset .. src_offset + self.head_dim]);
                @memcpy(out_v.data[dst_offset .. dst_offset + self.head_dim], self.v_pos.data[src_offset .. src_offset + self.head_dim]);
            }
        }
    }

    // ─── Proyecciones ───

    /// 7.1c: vista head-major [batch, n_heads_arg, seq, d] con seq RUNTIME
    /// sobre un slab position-major [batch, cap, n_heads_arg*d]. El buffer
    /// (cap = N del FA config) es la capacidad; la vista usa solo seq
    /// posiciones — sin assert de reshape exacto.
    fn headMajorView(self: *Self, slab: Tensor(f16), n_heads_arg: usize, seq: usize, shape_buf: []usize, strides_buf: []usize) Tensor(f16) {
        shape_buf[0] = 1; // batch_size siempre 1 en el path legacy
        shape_buf[1] = n_heads_arg;
        shape_buf[2] = seq;
        shape_buf[3] = self.head_dim;
        strides_buf[0] = seq * n_heads_arg * self.head_dim;
        strides_buf[1] = self.head_dim;
        strides_buf[2] = n_heads_arg * self.head_dim;
        strides_buf[3] = 1;
        return slab.view(shape_buf, strides_buf, 0);
    }

    /// 7.1c: vista 2D [seq*1, dims] del prefix (seq runtime) de un slab
    /// position-major — para proyecciones GEMM y FFN sin reshape exacto.
    fn prefix2d(slab: Tensor(f16), rows: usize, cols: usize, shape_buf: []usize, strides_buf: []usize) Tensor(f16) {
        shape_buf[0] = rows;
        shape_buf[1] = cols;
        strides_buf[0] = cols;
        strides_buf[1] = 1;
        return slab.view(shape_buf, strides_buf, 0);
    }

    fn projectQ(self: *Self, X: Tensor(f16)) !void {
        const p_shape = try self.allocator.alloc(usize, 2);
        const p_strides = try self.allocator.alloc(usize, 2);
        defer {
            self.allocator.free(p_shape);
            self.allocator.free(p_strides);
        }
        var Q_2d = prefix2d(self.q_pos, X.shape[0], self.num_heads * self.head_dim, p_shape, p_strides);
        try self.matmul_engine.linearProjection(f16, X, self.w_q_t.?, &Q_2d);
    }
    fn projectK(self: *Self, X: Tensor(f16)) !void {
        const p_shape = try self.allocator.alloc(usize, 2);
        const p_strides = try self.allocator.alloc(usize, 2);
        defer {
            self.allocator.free(p_shape);
            self.allocator.free(p_strides);
        }
        var K_2d = prefix2d(self.k_pos, X.shape[0], self.num_kv_heads * self.head_dim, p_shape, p_strides);
        try self.matmul_engine.linearProjection(f16, X, self.w_k_t.?, &K_2d);
    }
    fn projectV(self: *Self, X: Tensor(f16)) !void {
        const p_shape = try self.allocator.alloc(usize, 2);
        const p_strides = try self.allocator.alloc(usize, 2);
        defer {
            self.allocator.free(p_shape);
            self.allocator.free(p_strides);
        }
        var V_2d = prefix2d(self.v_pos, X.shape[0], self.num_kv_heads * self.head_dim, p_shape, p_strides);
        try self.matmul_engine.linearProjection(f16, X, self.w_v_t.?, &V_2d);
    }
    fn projectOut(self: *Self, seq_len: usize) !void {
        // F-4 (lane-f): escribe al slab o_scratch f16 (el stream f32 se
        // suma en el residual tras la proyección). El GEMM sigue
        // homogéneo f16 — pesos f16, activación f16.
        const a_shape = try self.allocator.alloc(usize, 2);
        const a_strides = try self.allocator.alloc(usize, 2);
        defer {
            self.allocator.free(a_shape);
            self.allocator.free(a_strides);
        }
        const attn_2d = prefix2d(self.attn_pos, seq_len, self.num_heads * self.head_dim, a_shape, a_strides);
        const o_shape = try self.allocator.alloc(usize, 2);
        const o_strides = try self.allocator.alloc(usize, 2);
        defer {
            self.allocator.free(o_shape);
            self.allocator.free(o_strides);
        }
        var out_2d = prefix2d(self.o_scratch, seq_len, self.hidden_dim, o_shape, o_strides);
        try self.matmul_engine.linearProjection(f16, attn_2d, self.w_o_t.?, &out_2d);
    }

    // ─── Proyecciones cuantizadas ───
    fn projectQQuantized(self: *Self, X: Tensor(f16)) !void {
        const p_shape = try self.allocator.alloc(usize, 2);
        const p_strides = try self.allocator.alloc(usize, 2);
        defer {
            self.allocator.free(p_shape);
            self.allocator.free(p_strides);
        }
        const Q_2d = prefix2d(self.q_pos, X.shape[0], self.num_heads * self.head_dim, p_shape, p_strides);
        const X_f32 = try Tensor(f32).alloc(self.allocator, X.shape);
        defer @constCast(&X_f32).deinit();
        for (X.data, X_f32.data) |s, *d| d.* = @as(f32, @floatCast(s));
        const Q_f32 = try Tensor(f32).alloc(self.allocator, Q_2d.shape);
        defer @constCast(&Q_f32).deinit();
        try self.matmul_engine.gemmQuantized(X_f32, self.w_q_t_q.?, &Q_f32, X.shape[0], Q_2d.shape[1], X.shape[1]);
        for (Q_f32.data, Q_2d.data) |s, *d| d.* = @floatCast(s);
    }
    fn projectKQuantized(self: *Self, X: Tensor(f16)) !void {
        const p_shape = try self.allocator.alloc(usize, 2);
        const p_strides = try self.allocator.alloc(usize, 2);
        defer {
            self.allocator.free(p_shape);
            self.allocator.free(p_strides);
        }
        const K_2d = prefix2d(self.k_pos, X.shape[0], self.num_kv_heads * self.head_dim, p_shape, p_strides);
        const X_f32 = try Tensor(f32).alloc(self.allocator, X.shape);
        defer @constCast(&X_f32).deinit();
        for (X.data, X_f32.data) |s, *d| d.* = @as(f32, @floatCast(s));
        const K_f32 = try Tensor(f32).alloc(self.allocator, K_2d.shape);
        defer @constCast(&K_f32).deinit();
        try self.matmul_engine.gemmQuantized(X_f32, self.w_k_t_q.?, &K_f32, X.shape[0], K_2d.shape[1], X.shape[1]);
        for (K_f32.data, K_2d.data) |s, *d| d.* = @floatCast(s);
    }
    fn projectVQuantized(self: *Self, X: Tensor(f16)) !void {
        const p_shape = try self.allocator.alloc(usize, 2);
        const p_strides = try self.allocator.alloc(usize, 2);
        defer {
            self.allocator.free(p_shape);
            self.allocator.free(p_strides);
        }
        const V_2d = prefix2d(self.v_pos, X.shape[0], self.num_kv_heads * self.head_dim, p_shape, p_strides);
        const X_f32 = try Tensor(f32).alloc(self.allocator, X.shape);
        defer @constCast(&X_f32).deinit();
        for (X.data, X_f32.data) |s, *d| d.* = @as(f32, @floatCast(s));
        const V_f32 = try Tensor(f32).alloc(self.allocator, V_2d.shape);
        defer @constCast(&V_f32).deinit();
        try self.matmul_engine.gemmQuantized(X_f32, self.w_v_t_q.?, &V_f32, X.shape[0], V_2d.shape[1], X.shape[1]);
        for (V_f32.data, V_2d.data) |s, *d| d.* = @floatCast(s);
    }
    fn projectOutQuantized(self: *Self, seq_len: usize) !void {
        // F-4 (lane-f): idem projectOut — salida al slab o_scratch f16.
        const a_shape = try self.allocator.alloc(usize, 2);
        const a_strides = try self.allocator.alloc(usize, 2);
        defer {
            self.allocator.free(a_shape);
            self.allocator.free(a_strides);
        }
        const attn_2d = prefix2d(self.attn_pos, seq_len, self.num_heads * self.head_dim, a_shape, a_strides);
        const o_shape = try self.allocator.alloc(usize, 2);
        const o_strides = try self.allocator.alloc(usize, 2);
        defer {
            self.allocator.free(o_shape);
            self.allocator.free(o_strides);
        }
        const out_2d = prefix2d(self.o_scratch, seq_len, self.hidden_dim, o_shape, o_strides);
        const attn_f32 = try Tensor(f32).alloc(self.allocator, attn_2d.shape);
        defer @constCast(&attn_f32).deinit();
        for (attn_2d.data, attn_f32.data) |s, *d| d.* = @as(f32, @floatCast(s));
        const out_f32 = try Tensor(f32).alloc(self.allocator, out_2d.shape);
        defer @constCast(&out_f32).deinit();
        try self.matmul_engine.gemmQuantized(attn_f32, self.w_o_t_q.?, &out_f32, attn_2d.shape[0], out_2d.shape[1], attn_2d.shape[1]);
        for (out_f32.data, out_2d.data) |s, *d| d.* = @floatCast(s);
    }

    /// BitNet feedback: lightweight recurrent merge.
    /// u = e + α * sigmoid(W_proj @ prev_state) ⊙ (W_proj @ prev_state)
    /// Lighter than full RLT: no concat, no 2× projection (single [d]→[d] linear).
    pub fn mergeBitNetFeedback(self: *Self, encoder_rep: []f32) void {
        if (self.bitnet_feedback_scale == 0) return;
        const prev = self.bitnet_prev_state orelse return;
        const proj = self.bitnet_w_proj orelse return;
        const d = self.hidden_dim;
        // 1. gate = sigmoid(W_proj @ prev_state)
        // 2. state_proj = W_proj @ prev_state (reuse same matrix)
        // 3. out = e + α * gate ⊙ state_proj
        for (0..d) |i| {
            var acc: f32 = 0;
            for (0..d) |j| acc += proj.data[i * d + j] * prev[j];
            const gate = 1.0 / (1.0 + @exp(-acc)); // sigmoid
            encoder_rep[i] += self.bitnet_feedback_scale * gate * acc;
        }
    }

    /// Store final hidden state as prev_state for next token.
    pub fn storeBitNetState(self: *Self, hidden: []const f32) void {
        if (self.bitnet_prev_state) |prev| {
            @memcpy(prev, hidden[0..self.hidden_dim]);
        }
    }
};

fn loadWeightFile(io: std.Io, allocator: std.mem.Allocator, base: []const u8, name: []const u8) !Tensor(f16) {
    const path = try std.fmt.allocPrint(allocator, "{s}{s}.bin", .{ base, name });
    defer allocator.free(path);
    const dir = std.Io.Dir.cwd();
    const bytes = dir.readFileAlloc(io, path, allocator, .unlimited) catch {
        std.log.err("FATAL: Weight file not found: {s}", .{path});
        return TransformerError.WeightFileNotFound;
    };
    defer allocator.free(bytes);
    const num_elements = bytes.len / 2;
    const tensor = try Tensor(f16).initUninitialized(allocator, &.{num_elements});
    @memcpy(std.mem.sliceAsBytes(tensor.data), bytes);
    return tensor;
}

fn loadWeightFileF32(io: std.Io, allocator: std.mem.Allocator, base: []const u8, name: []const u8) !Tensor(f32) {
    const path = try std.fmt.allocPrint(allocator, "{s}{s}.bin", .{ base, name });
    defer allocator.free(path);
    const dir = std.Io.Dir.cwd();
    const bytes = dir.readFileAlloc(io, path, allocator, .unlimited) catch {
        std.log.err("FATAL: Weight file not found: {s}", .{path});
        return TransformerError.WeightFileNotFound;
    };
    defer allocator.free(bytes);
    const num_elements = bytes.len / 4;
    const tensor = try Tensor(f32).initUninitialized(allocator, &.{num_elements});
    @memcpy(std.mem.sliceAsBytes(tensor.data), bytes);
    return tensor;
}

/// Carga un peso 2D del GGUF y lo dequantiza a f16 en layout [out, in] row-major.
/// Carga un peso 2-D como QuantWeight crudo (bytes mmap PRESTADOS del
/// GGUF — el caller debe mantener el GgufFile vivo mientras viva la capa).
/// RAM = 0 extra: el dequant va lazy (ensureWeightsF16) o al kernel.
/// Devuelve error si el tensor no es cuantizado (los f16/bf16 nativos
/// se materializan directo: no hay beneficio en diferirlos).
fn loadGgufQuantWeight(
    allocator: std.mem.Allocator,
    g: *const gguf.GgufFile,
    prefix: []const u8,
    names: []const []const u8,
) !quant_weight.QuantWeight {
    var found: ?*const gguf.TensorInfo = null;
    for (names) |n| {
        const full = try std.fmt.allocPrint(allocator, "{s}{s}", .{ prefix, n });
        defer allocator.free(full);
        if (g.getTensor(full)) |info| {
            found = info;
            break;
        }
    }
    const info = found orelse return TransformerError.WeightFileNotFound;
    if (info.n_dims != 2) return TransformerError.WeightFileNotFound;
    return quant_weight.QuantWeight.init(info, g.tensorData(info));
}

/// Materializa los Tensor(f16) desde los QuantWeight crudos (dequant
/// on-the-fly). Idempotente. Pico por peso: f32buf transitorio + f16 out.
///
/// 7.1a: usa F16Residency LRU — si todas las slots están ocupadas, evicta
/// la víctima LRU antes de materializar. Los QuantWeight se conservan
/// (re-materialización on-demand tras evicción).
pub fn ensureWeightsF16(self: *TransformerLayer) !void {
    if (self.w_q_t != null or self.w_q_qw == null) return;

    // 7.1a: si el cap exige evicción, evict la víctima LRU
    if (F16Residency.needsEviction()) {
        _ = F16Residency.evictVictim();
    }

    self.w_q_t = try materializeQwF16(self.allocator, self.w_q_qw.?);
    self.w_k_t = try materializeQwF16(self.allocator, self.w_k_qw.?);
    self.w_v_t = try materializeQwF16(self.allocator, self.w_v_qw.?);
    self.w_o_t = try materializeQwF16(self.allocator, self.w_o_qw.?);
    self.w_gate_t = try materializeQwF16(self.allocator, self.w_gate_qw.?);
    self.w_up_t = try materializeQwF16(self.allocator, self.w_up_qw.?);
    self.w_down_t = try materializeQwF16(self.allocator, self.w_down_qw.?);
    // NO nulificamos QuantWeight — se conservan para re-materialización
    // tras evicción LRU (los bytes son mmap-prestados, sin alloc propio).
    F16Residency.touch(self);
}

/// 7.1a: evict f16 tensors de esta capa (libera RAM, conserva QuantWeight).
pub fn evictWeightsF16(self: *TransformerLayer) void {
    if (self.w_q_t == null) return; // ya evicted
    if (self.w_q_t) |*w| w.deinit();
    if (self.w_k_t) |*w| w.deinit();
    if (self.w_v_t) |*w| w.deinit();
    if (self.w_o_t) |*w| w.deinit();
    if (self.w_gate_t) |*w| w.deinit();
    if (self.w_up_t) |*w| w.deinit();
    if (self.w_down_t) |*w| w.deinit();
    self.w_q_t = null;
    self.w_k_t = null;
    self.w_v_t = null;
    self.w_o_t = null;
    self.w_gate_t = null;
    self.w_up_t = null;
    self.w_down_t = null;
}

fn materializeQwF16(allocator: std.mem.Allocator, qw: quant_weight.QuantWeight) !Tensor(f16) {
    const shape = qw.shape(); // [in, out] GGUF
    const in_dim: usize = @intCast(shape[0]);
    const out_dim: usize = @intCast(shape[1]);
    const numel = in_dim * out_dim;
    const f32buf = try allocator.alloc(f32, numel);
    defer allocator.free(f32buf);
    // 7.2 (lane-f): transponer al materializar. El tensor de la capa es
    // [out, in] (orientación trans_b=true de linearProjection), pero
    // antes se copiaba el buffer lineal GGUF [in, out] SIN transponer —
    // pesos mezclados en TODAS las proyecciones del path legacy (el
    // híbrido ya usaba dequantToF32Transposed; este camino no).
    qw.dequantToF32Transposed(f32buf);
    const tensor = try Tensor(f16).initUninitialized(allocator, &.{ out_dim, in_dim });
    for (tensor.data, f32buf) |*d, s| d.* = @floatCast(s);
    return tensor;
}

fn loadGgufWeightF16(
    allocator: std.mem.Allocator,
    g: *const gguf.GgufFile,
    prefix: []const u8,
    names: []const []const u8,
) !Tensor(f16) {
    var found: ?*const gguf.TensorInfo = null;
    for (names) |n| {
        const full = try std.fmt.allocPrint(allocator, "{s}{s}", .{ prefix, n });
        defer allocator.free(full);
        if (g.getTensor(full)) |info| {
            found = info;
            break;
        }
    }
    const info = found orelse return TransformerError.WeightFileNotFound;

    if (info.n_dims != 2) return TransformerError.WeightFileNotFound;
    const in_dim: usize = @intCast(info.dims[0]);
    const out_dim: usize = @intCast(info.dims[1]);
    const numel = in_dim * out_dim;

    const f32buf = try allocator.alloc(f32, numel);
    defer allocator.free(f32buf);
    try gguf.dequantTensor(info, g.tensorData(info), f32buf);

    const tensor = try Tensor(f16).initUninitialized(allocator, &.{ out_dim, in_dim });
    for (tensor.data, f32buf) |*d, s| d.* = @floatCast(s);
    return tensor;
}

/// Carga un peso 1D de norma (RMSNorm gamma) en f32.
fn loadGgufNormF32(
    allocator: std.mem.Allocator,
    g: *const gguf.GgufFile,
    prefix: []const u8,
    names: []const []const u8,
) !Tensor(f32) {
    var found: ?*const gguf.TensorInfo = null;
    for (names) |n| {
        const full = try std.fmt.allocPrint(allocator, "{s}{s}", .{ prefix, n });
        defer allocator.free(full);
        if (g.getTensor(full)) |info| {
            found = info;
            break;
        }
    }
    const info = found orelse return TransformerError.WeightFileNotFound;
    const numel: usize = @intCast(info.numel());

    const f32buf = try allocator.alloc(f32, numel);
    defer allocator.free(f32buf);
    try gguf.dequantTensor(info, g.tensorData(info), f32buf);

    const tensor = try Tensor(f32).initUninitialized(allocator, &.{numel});
    @memcpy(tensor.data, f32buf);
    return tensor;
}

/// BitNet b1.58 (lane-kvc P4): norm 1D con nombre exacto (attn_sub_norm /
/// ffn_sub_norm). Falla suave: si el tensor no existe devuelve null (el
/// wiring lo trata como identidad).
fn loadGgufNorm1DOptF32(
    allocator: std.mem.Allocator,
    g: *const gguf.GgufFile,
    prefix: []const u8,
    name: []const u8,
) !?Tensor(f32) {
    const full = try std.fmt.allocPrint(allocator, "{s}{s}", .{ prefix, name });
    defer allocator.free(full);
    const info = g.getTensor(full) orelse return null;
    const numel: usize = @intCast(info.numel());
    const f32buf = try allocator.alloc(f32, numel);
    defer allocator.free(f32buf);
    try gguf.dequantTensor(info, g.tensorData(info), f32buf);
    const tensor = try Tensor(f32).initUninitialized(allocator, &.{numel});
    @memcpy(tensor.data, f32buf);
    return tensor;
}

/// BitNet feedback: load projection weight [d, d] from GGUF. Returns null if not present.
fn loadBitNetFeedbackWeight(
    allocator: std.mem.Allocator,
    g: *const gguf.GgufFile,
    prefix: []const u8,
) !?Tensor(f32) {
    const name = "rlt.bitnet_feedback.weight";
    const full = try std.fmt.allocPrint(allocator, "{s}{s}", .{ prefix, name });
    defer allocator.free(full);
    const info = g.getTensor(full) orelse return null;
    const numel: usize = @intCast(info.numel());
    const f32buf = try allocator.alloc(f32, numel);
    defer allocator.free(f32buf);
    try gguf.dequantTensor(info, g.tensorData(info), f32buf);
    const tensor = try Tensor(f32).initUninitialized(allocator, &.{numel});
    @memcpy(tensor.data, f32buf);
    return tensor;
}

/// Cache KV simple (legacy)
pub const KVCache = struct {
    allocator: std.mem.Allocator,
    k_cache: Tensor(f16),
    v_cache: Tensor(f16),
    max_seq_len: usize,
    current_len: usize = 0,

    pub fn init(allocator: std.mem.Allocator, config: FlashAttentionConfig) !KVCache {
        const k_cache = try Tensor(f16).alloc(allocator, &.{ config.batch_size, config.num_heads, config.N, config.d });
        const v_cache = try Tensor(f16).alloc(allocator, &.{ config.batch_size, config.num_heads, config.N, config.d });
        return .{ .allocator = allocator, .k_cache = k_cache, .v_cache = v_cache, .max_seq_len = config.N };
    }
    pub fn deinit(self: *KVCache) void {
        self.k_cache.deinit();
        self.v_cache.deinit();
    }
    pub fn append(self: *KVCache, k_new: *Tensor(f16), v_new: *Tensor(f16)) !void {
        const new_len = k_new.shape[2];
        if (self.current_len + new_len > self.max_seq_len) return TransformerError.CacheOverflow;
        const d = self.k_cache.shape[3];
        const tokens_per_bh = new_len * d;
        const cache_tokens_per_bh = self.max_seq_len * d;
        for (0..self.k_cache.shape[0]) |b| {
            for (0..self.k_cache.shape[1]) |h| {
                const bh = b * self.k_cache.shape[1] + h;
                const src_offset = bh * tokens_per_bh;
                const dst_offset = bh * cache_tokens_per_bh + self.current_len * d;
                @memcpy(self.k_cache.data[dst_offset..][0..tokens_per_bh], k_new.data[src_offset..][0..tokens_per_bh]);
                @memcpy(self.v_cache.data[dst_offset..][0..tokens_per_bh], v_new.data[src_offset..][0..tokens_per_bh]);
            }
        }
        self.current_len += new_len;
    }
    pub fn clear(self: *KVCache) void {
        self.current_len = 0;
        @memset(self.k_cache.data, 0);
        @memset(self.v_cache.data, 0);
    }
};

/// 7.1a: LRU manager para f16 tensors por capa. Evita materializar TODAS
/// las capas a la vez (~7GB para Llama-3.2-3B × 28). Con max_resident=2
/// solo 2 capas tienen f16 → ~500MB vs ~7GB.
pub const F16Residency = struct {
    const max_layers = 128;
    const Entry = struct {
        ptr: ?*TransformerLayer = null,
        last_used: u64 = 0,
    };

    var entries: [max_layers]Entry = undefined;
    var count: usize = 0;
    var tick: u64 = 0;
    var max_resident: usize = 2;

    pub fn setMaxResident(n: usize) void {
        max_resident = n;
    }

    pub fn getMaxResident() usize {
        return max_resident;
    }

    /// Registra una capa en el registry global. Llamar desde pipeline.init.
    pub fn register(layer: *TransformerLayer) void {
        for (entries[0..count]) |e| {
            if (e.ptr == layer) return;
        }
        if (count < max_layers) {
            entries[count] = .{ .ptr = layer };
            count += 1;
        }
    }

    /// ¿Necesita evicción? (materialized >= max_resident)
    pub fn needsEviction() bool {
        var materialized: usize = 0;
        for (entries[0..count]) |e| {
            if (e.ptr) |p| {
                if (p.w_q_t != null) materialized += 1;
            }
        }
        return materialized >= max_resident;
    }

    /// Evict la víctima LRU. Devuelve el layer_idx de la víctima o null.
    pub fn evictVictim() ?usize {
        var best_ptr: ?*TransformerLayer = null;
        var best_tick: u64 = std.math.maxInt(u64);
        for (entries[0..count]) |*e| {
            if (e.ptr) |p| {
                if (p.w_q_t != null and e.last_used < best_tick) {
                    best_tick = e.last_used;
                    best_ptr = p;
                }
            }
        }
        if (best_ptr) |p| {
            evictWeightsF16(p);
            return p.layer_idx;
        }
        return null;
    }

    /// Touch: actualizar tick de una capa.
    pub fn touch(layer: *TransformerLayer) void {
        tick +|= 1;
        for (entries[0..count]) |*e| {
            if (e.ptr == layer) {
                e.last_used = tick;
                return;
            }
        }
    }
};
