const std = @import("std");
const Tensor = @import("core").Tensor;
const matmul = @import("matmul");
const fa = @import("fa");
const kvcache = @import("kv_cache");
const norm = @import("norm.zig");
const ffn = @import("ffn.zig");
const rope_mod = @import("rope.zig");
const gqa_mod = @import("gqa.zig");

const FlashAttention = fa.FlashAttention;
const FlashAttentionConfig = fa.fa_config.FlashAttentionConfig;
const MatmulEngine = matmul.MatmulEngine;
const PrecisionMode = matmul.PrecisionMode;
const KVCacheManager = kvcache.KVCacheManager;
const KVCacheConfig = kvcache.KVCacheConfig;

pub const TransformerError = error{
    WeightFileNotFound, MatmulNotImplemented, CacheOverflow,
    InvalidPrecision, BackendMismatch, KvCacheNotSet,
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

    // Pesos de normalización
    attn_norm: ?Tensor(f32) = null,
    ffn_norm: ?Tensor(f32) = null,

    // Pesos cuantizados (opcional)
    w_q_t_q: ?matmul.QuantizedTensor = null,
    w_k_t_q: ?matmul.QuantizedTensor = null,
    w_v_t_q: ?matmul.QuantizedTensor = null,
    w_o_t_q: ?matmul.QuantizedTensor = null,

    // Motores
    matmul_engine: MatmulEngine,
    fa_engine: FlashAttention,

    // Configuración
    hidden_dim: usize,
    head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    intermediate_dim: usize,
    precision: LayerPrecision,
    rms_eps: f32 = 1e-5,

    // Buffers intermedios
    q_proj: Tensor(f16),
    k_proj: Tensor(f16),
    v_proj: Tensor(f16),
    attn_out: Tensor(f16),
    ffn_gate: Tensor(f16),
    ffn_up: Tensor(f16),
    ffn_out: Tensor(f16),
    norm_buf: Tensor(f16),

    // KV-Cache
    kv_manager: ?*KVCacheManager = null,
    seq_id: u64 = 0,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        layer_idx: usize,
        fa_config_val: FlashAttentionConfig,
        ptx_path: []const u8,
        hidden_dim: usize,
        precision: LayerPrecision,
        num_kv_heads: usize,
        intermediate_dim: usize,
    ) !Self {
        const backend = if (precision.weights_on_gpu) matmul.Backend.cublas else matmul.Backend.auto;
        var engine = try MatmulEngine.init(allocator, backend, precision.compute);
        errdefer engine.deinit();

        var fa_eng = try FlashAttention.init(allocator, fa_config_val, ptx_path);
        errdefer fa_eng.deinit();
        const N = fa_config_val.N;
        const num_heads = fa_config_val.num_heads;
        const d = fa_config_val.d;
        const batch_size = fa_config_val.batch_size;

        var q_proj = try Tensor(f16).alloc(allocator, &.{ batch_size, num_heads, N, d });
        errdefer q_proj.deinit();
        var k_proj = try Tensor(f16).alloc(allocator, &.{ batch_size, num_kv_heads, N, d });
        errdefer k_proj.deinit();
        var v_proj = try Tensor(f16).alloc(allocator, &.{ batch_size, num_kv_heads, N, d });
        errdefer v_proj.deinit();
        var attn_out = try Tensor(f16).alloc(allocator, &.{ batch_size, num_heads, N, d });
        errdefer attn_out.deinit();
        var ffn_gate = try Tensor(f16).alloc(allocator, &.{ batch_size, N, intermediate_dim });
        errdefer ffn_gate.deinit();
        var ffn_up = try Tensor(f16).alloc(allocator, &.{ batch_size, N, intermediate_dim });
        errdefer ffn_up.deinit();
        var ffn_out = try Tensor(f16).alloc(allocator, &.{ batch_size, N, hidden_dim });
        errdefer ffn_out.deinit();
        var norm_buf = try Tensor(f16).alloc(allocator, &.{ batch_size, N, hidden_dim });
        errdefer norm_buf.deinit();

        return .{
            .allocator = allocator,
            .layer_idx = layer_idx,
            .matmul_engine = engine,
            .fa_engine = fa_eng,
            .hidden_dim = hidden_dim,
            .head_dim = d,
            .num_heads = num_heads,
            .num_kv_heads = num_kv_heads,
            .intermediate_dim = intermediate_dim,
            .precision = precision,
            .q_proj = q_proj,
            .k_proj = k_proj,
            .v_proj = v_proj,
            .attn_out = attn_out,
            .ffn_gate = ffn_gate,
            .ffn_up = ffn_up,
            .ffn_out = ffn_out,
            .norm_buf = norm_buf,
        };
    }

    pub fn deinit(self: *Self) void {
        self.fa_engine.deinit();
        self.q_proj.deinit(); self.k_proj.deinit();
        self.v_proj.deinit(); self.attn_out.deinit();
        self.ffn_gate.deinit(); self.ffn_up.deinit();
        self.ffn_out.deinit(); self.norm_buf.deinit();
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

        if (self.w_q_t_q) |*w| w.deinit();
        if (self.w_k_t_q) |*w| w.deinit();
        if (self.w_v_t_q) |*w| w.deinit();
        if (self.w_o_t_q) |*w| w.deinit();
    }

    pub fn loadWeights(self: *Self, checkpoint_dir: []const u8) !void {
        const base = try std.fmt.allocPrint(self.allocator, "{s}/layer.{d}.", .{ checkpoint_dir, self.layer_idx });
        defer self.allocator.free(base);

        self.w_q_t = try loadWeightFile(self.allocator, base, "self_attn.q_proj.weight_t");
        self.w_k_t = try loadWeightFile(self.allocator, base, "self_attn.k_proj.weight_t");
        self.w_v_t = try loadWeightFile(self.allocator, base, "self_attn.v_proj.weight_t");
        self.w_o_t = try loadWeightFile(self.allocator, base, "self_attn.o_proj.weight_t");
        self.w_gate_t = try loadWeightFile(self.allocator, base, "mlp.gate_proj.weight_t");
        self.w_up_t = try loadWeightFile(self.allocator, base, "mlp.up_proj.weight_t");
        self.w_down_t = try loadWeightFile(self.allocator, base, "mlp.down_proj.weight_t");

        // Cargar norm weights (f32)
        self.attn_norm = try loadWeightFileF32(self.allocator, base, "input_layernorm.weight");
        self.ffn_norm = try loadWeightFileF32(self.allocator, base, "post_attention_layernorm.weight");

        if (self.precision.use_quantized) {
            const qcfg = matmul.QuantConfig{ .bits = 8, .symmetric = true, .per_channel = true, .group_size = 0 };
            self.w_q_t_q = try matmul.quantizeInt8PerChannel(self.allocator, self.w_q_t.?, qcfg);
            self.w_k_t_q = try matmul.quantizeInt8PerChannel(self.allocator, self.w_k_t.?, qcfg);
            self.w_v_t_q = try matmul.quantizeInt8PerChannel(self.allocator, self.w_v_t.?, qcfg);
            self.w_o_t_q = try matmul.quantizeInt8PerChannel(self.allocator, self.w_o_t.?, qcfg);
        }
    }

    /// Forward completo: PreNorm -> Attn -> Residual -> PostNorm -> FFN -> Residual
    pub fn forward(
        self: *Self,
        hidden_state: Tensor(f16),
        output: *Tensor(f16),
        position: usize,
        _is_prefill: bool,
    ) !void {
        _ = _is_prefill;
        const batch_size = hidden_state.shape[0];
        const seq_len = hidden_state.shape[1];

        // Reinterpretar como 2D para GEMM
        const X_2d = try hidden_state.reshape(&[_]usize{ batch_size * seq_len, self.hidden_dim });
        defer { if (X_2d.allocator) |a| { a.free(X_2d.shape); a.free(X_2d.strides); } }

        // === 1. Pre-Attention RMSNorm ===
        const norm_2d = try self.norm_buf.reshape(&[_]usize{ batch_size * seq_len, self.hidden_dim });
        defer { if (norm_2d.allocator) |a| { a.free(norm_2d.shape); a.free(norm_2d.strides); } }

        if (self.attn_norm) |gamma| {
            norm.rmsNorm(f16, f32, hidden_state, gamma, self.rms_eps, &self.norm_buf);
        }

        // === 2. Proyecciones Q, K, V ===
        if (self.precision.use_quantized) {
            try self.projectQQuantized(norm_2d);
            try self.projectKQuantized(norm_2d);
            try self.projectVQuantized(norm_2d);
        } else {
            try self.projectQ(norm_2d);
            try self.projectK(norm_2d);
            try self.projectV(norm_2d);
        }

        // === 3. RoPE ===
        rope_mod.applyRoPE(&self.q_proj, &self.k_proj, position, self.head_dim, 10000.0);

        // === 4. KV-Cache ===
        if (self.kv_manager) |mgr| {
            try self.storeKvCache(mgr, seq_len);
        }

        // === 5. Recuperar K/V full ===
        var k_full: Tensor(f16) = undefined;
        var v_full: Tensor(f16) = undefined;
        var k_full_owned = false;
        var v_full_owned = false;

        if (self.kv_manager) |mgr| {
            const total_len = try mgr.getSequenceLen(self.seq_id);
            k_full = try Tensor(f16).alloc(self.allocator, &.{ batch_size, self.num_kv_heads, total_len, self.head_dim });
            v_full = try Tensor(f16).alloc(self.allocator, &.{ batch_size, self.num_kv_heads, total_len, self.head_dim });
            k_full_owned = true;
            v_full_owned = true;
            try self.retrieveKvCache(mgr, &k_full, &v_full);
        } else {
            k_full = self.k_proj;
            v_full = self.v_proj;
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

        // === 7. FlashAttention ===
        try self.fa_engine.forward(self.q_proj, k_expanded, v_expanded, &self.attn_out);

        if (k_exp_owned) k_expanded.deinit();
        if (v_exp_owned) v_expanded.deinit();
        if (k_full_owned) k_full.deinit();
        if (v_full_owned) v_full.deinit();

        // === 8. Proyección de salida O ===
        if (self.precision.use_quantized) {
            try self.projectOutQuantized(output);
        } else {
            try self.projectOut(output);
        }

        // === 9. Residual connection (Attn) ===
        for (output.data, hidden_state.data) |*o, h| {
            const val = @as(f32, @floatCast(o.*)) + @as(f32, @floatCast(h));
            o.* = @floatCast(val);
        }

        // === 10. Post-Attention RMSNorm ===
        var attn_residual = output.*;
        if (self.ffn_norm) |gamma| {
            norm.rmsNorm(f16, f32, attn_residual, gamma, self.rms_eps, &self.norm_buf);
            attn_residual = self.norm_buf;
        }

        // === 11. FFN SwiGLU ===
        const attn_res_2d = try attn_residual.reshape(&[_]usize{ batch_size * seq_len, self.hidden_dim });
        defer { if (attn_res_2d.allocator) |a| { a.free(attn_res_2d.shape); a.free(attn_res_2d.strides); } }

        var ffn_out_2d = try self.ffn_out.reshape(&[_]usize{ batch_size * seq_len, self.hidden_dim });
        defer { if (ffn_out_2d.allocator) |a| { a.free(ffn_out_2d.shape); a.free(ffn_out_2d.strides); } }

        var gate_2d = try self.ffn_gate.reshape(&[_]usize{ batch_size * seq_len, self.intermediate_dim });
        defer { if (gate_2d.allocator) |a| { a.free(gate_2d.shape); a.free(gate_2d.strides); } }
        var up_2d = try self.ffn_up.reshape(&[_]usize{ batch_size * seq_len, self.intermediate_dim });
        defer { if (up_2d.allocator) |a| { a.free(up_2d.shape); a.free(up_2d.strides); } }

        try ffn.swiGluForward(
            &self.matmul_engine, f16,
            attn_res_2d,
            self.w_gate_t.?, self.w_up_t.?, self.w_down_t.?,
            &gate_2d, &up_2d, &ffn_out_2d,
        );

        // === 12. Residual connection (FFN) ===
        for (output.data, self.ffn_out.data) |*o, f| {
            const val = @as(f32, @floatCast(o.*)) + @as(f32, @floatCast(f));
            o.* = @floatCast(val);
        }
    }

    // ─── KV-Cache helpers ───
    fn storeKvCache(self: *Self, mgr: *KVCacheManager, seq_len: usize) !void {
        for (0..self.num_kv_heads) |kv_h| {
            for (0..seq_len) |pos| {
                const k_slice = try self.allocator.alloc(f16, self.head_dim);
                defer self.allocator.free(k_slice);
                const v_slice = try self.allocator.alloc(f16, self.head_dim);
                defer self.allocator.free(v_slice);

                const k_offset = ((0 * self.num_kv_heads + kv_h) * self.k_proj.shape[2] + pos) * self.head_dim;
                const v_offset = ((0 * self.num_kv_heads + kv_h) * self.v_proj.shape[2] + pos) * self.head_dim;
                @memcpy(k_slice, self.k_proj.data[k_offset..k_offset + self.head_dim]);
                @memcpy(v_slice, self.v_proj.data[v_offset..v_offset + self.head_dim]);

                const q_head_for_kv = kv_h * (self.num_heads / self.num_kv_heads);
                try mgr.appendTokensF16(self.seq_id, @as(u32, @intCast(self.layer_idx)), @as(u32, @intCast(q_head_for_kv)), k_slice, v_slice);
            }
        }
        for (0..seq_len) |_| try mgr.advanceSequence(self.seq_id);
    }

    fn retrieveKvCache(self: *Self, mgr: *KVCacheManager, out_k: *Tensor(f16), out_v: *Tensor(f16)) !void {
        const total_len = try mgr.getSequenceLen(self.seq_id);
        for (0..self.num_kv_heads) |kv_h| {
            var k_head = try self.allocator.alloc(f16, total_len * self.head_dim);
            defer self.allocator.free(k_head);
            var v_head = try self.allocator.alloc(f16, total_len * self.head_dim);
            defer self.allocator.free(v_head);

            const q_head_for_kv = kv_h * (self.num_heads / self.num_kv_heads);
            try mgr.retrieveForAttention(self.seq_id, @as(u32, @intCast(self.layer_idx)), @as(u32, @intCast(q_head_for_kv)), k_head, v_head);

            for (0..total_len) |pos| {
                const src_offset = pos * self.head_dim;
                const dst_offset = ((0 * self.num_kv_heads + kv_h) * total_len + pos) * self.head_dim;
                @memcpy(out_k.data[dst_offset..dst_offset + self.head_dim], k_head[src_offset..src_offset + self.head_dim]);
                @memcpy(out_v.data[dst_offset..dst_offset + self.head_dim], v_head[src_offset..src_offset + self.head_dim]);
            }
        }
    }

    // ─── Proyecciones ───
    fn projectQ(self: *Self, X: Tensor(f16)) !void {
        var Q_2d = try self.q_proj.reshape(&[_]usize{ X.shape[0], self.num_heads * self.head_dim });
        defer { if (Q_2d.allocator) |a| { a.free(Q_2d.shape); a.free(Q_2d.strides); } }
        try self.matmul_engine.linearProjection(f16, X, self.w_q_t.?, &Q_2d);
    }
    fn projectK(self: *Self, X: Tensor(f16)) !void {
        var K_2d = try self.k_proj.reshape(&[_]usize{ X.shape[0], self.num_kv_heads * self.head_dim });
        defer { if (K_2d.allocator) |a| { a.free(K_2d.shape); a.free(K_2d.strides); } }
        try self.matmul_engine.linearProjection(f16, X, self.w_k_t.?, &K_2d);
    }
    fn projectV(self: *Self, X: Tensor(f16)) !void {
        var V_2d = try self.v_proj.reshape(&[_]usize{ X.shape[0], self.num_kv_heads * self.head_dim });
        defer { if (V_2d.allocator) |a| { a.free(V_2d.shape); a.free(V_2d.strides); } }
        try self.matmul_engine.linearProjection(f16, X, self.w_v_t.?, &V_2d);
    }
    fn projectOut(self: *Self, output: *Tensor(f16)) !void {
        const attn_2d = try self.attn_out.reshape(&[_]usize{ output.shape[0] * output.shape[1], self.num_heads * self.head_dim });
        defer { if (attn_2d.allocator) |a| { a.free(attn_2d.shape); a.free(attn_2d.strides); } }
        var out_2d = try output.reshape(&[_]usize{ output.shape[0] * output.shape[1], self.hidden_dim });
        defer { if (out_2d.allocator) |a| { a.free(out_2d.shape); a.free(out_2d.strides); } }
        try self.matmul_engine.linearProjection(f16, attn_2d, self.w_o_t.?, &out_2d);
    }

    // ─── Proyecciones cuantizadas ───
    fn projectQQuantized(self: *Self, X: Tensor(f16)) !void {
        const Q_2d = try self.q_proj.reshape(&[_]usize{ X.shape[0], self.num_heads * self.head_dim });
        defer { if (Q_2d.allocator) |a| { a.free(Q_2d.shape); a.free(Q_2d.strides); } }
        var X_f32 = try Tensor(f32).alloc(self.allocator, X.shape);
        defer X_f32.deinit();
        for (X.data, X_f32.data) |s, *d| d.* = @as(f32, @floatCast(s));
        var Q_f32 = try Tensor(f32).alloc(self.allocator, Q_2d.shape);
        defer Q_f32.deinit();
        try self.matmul_engine.gemmQuantized(X_f32, self.w_q_t_q.?, &Q_f32, X.shape[0], Q_2d.shape[1], X.shape[1]);
        for (Q_f32.data, Q_2d.data) |s, *d| d.* = @floatCast(s);
    }
    fn projectKQuantized(self: *Self, X: Tensor(f16)) !void {
        const K_2d = try self.k_proj.reshape(&[_]usize{ X.shape[0], self.num_kv_heads * self.head_dim });
        defer { if (K_2d.allocator) |a| { a.free(K_2d.shape); a.free(K_2d.strides); } }
        var X_f32 = try Tensor(f32).alloc(self.allocator, X.shape);
        defer X_f32.deinit();
        for (X.data, X_f32.data) |s, *d| d.* = @as(f32, @floatCast(s));
        var K_f32 = try Tensor(f32).alloc(self.allocator, K_2d.shape);
        defer K_f32.deinit();
        try self.matmul_engine.gemmQuantized(X_f32, self.w_k_t_q.?, &K_f32, X.shape[0], K_2d.shape[1], X.shape[1]);
        for (K_f32.data, K_2d.data) |s, *d| d.* = @floatCast(s);
    }
    fn projectVQuantized(self: *Self, X: Tensor(f16)) !void {
        const V_2d = try self.v_proj.reshape(&[_]usize{ X.shape[0], self.num_kv_heads * self.head_dim });
        defer { if (V_2d.allocator) |a| { a.free(V_2d.shape); a.free(V_2d.strides); } }
        var X_f32 = try Tensor(f32).alloc(self.allocator, X.shape);
        defer X_f32.deinit();
        for (X.data, X_f32.data) |s, *d| d.* = @as(f32, @floatCast(s));
        var V_f32 = try Tensor(f32).alloc(self.allocator, V_2d.shape);
        defer V_f32.deinit();
        try self.matmul_engine.gemmQuantized(X_f32, self.w_v_t_q.?, &V_f32, X.shape[0], V_2d.shape[1], X.shape[1]);
        for (V_f32.data, V_2d.data) |s, *d| d.* = @floatCast(s);
    }
    fn projectOutQuantized(self: *Self, output: *Tensor(f16)) !void {
        const attn_2d = try self.attn_out.reshape(&[_]usize{ output.shape[0] * output.shape[1], self.num_heads * self.head_dim });
        defer { if (attn_2d.allocator) |a| { a.free(attn_2d.shape); a.free(attn_2d.strides); } }
        const out_2d = try output.reshape(&[_]usize{ output.shape[0] * output.shape[1], self.hidden_dim });
        defer { if (out_2d.allocator) |a| { a.free(out_2d.shape); a.free(out_2d.strides); } }
        var attn_f32 = try Tensor(f32).alloc(self.allocator, attn_2d.shape);
        defer attn_f32.deinit();
        for (attn_2d.data, attn_f32.data) |s, *d| d.* = @as(f32, @floatCast(s));
        var out_f32 = try Tensor(f32).alloc(self.allocator, out_2d.shape);
        defer out_f32.deinit();
        try self.matmul_engine.gemmQuantized(attn_f32, self.w_o_t_q.?, &out_f32, attn_2d.shape[0], out_2d.shape[1], attn_2d.shape[1]);
        for (out_f32.data, out_2d.data) |s, *d| d.* = @floatCast(s);
    }
};

fn loadWeightFile(allocator: std.mem.Allocator, base: []const u8, name: []const u8) !Tensor(f16) {
    const path = try std.fmt.allocPrint(allocator, "{s}{s}.bin", .{ base, name });
    defer allocator.free(path);
    const file = std.fs.cwd().openFile(path, .{}) catch {
        std.log.err("FATAL: Weight file not found: {s}", .{path});
        return TransformerError.WeightFileNotFound;
    };
    defer file.close();
    const size = try file.getEndPos();
    const num_elements = size / 2;
    const tensor = try Tensor(f16).initUninitialized(allocator, &.{num_elements});
    _ = try file.readAll(std.mem.sliceAsBytes(tensor.data));
    return tensor;
}

fn loadWeightFileF32(allocator: std.mem.Allocator, base: []const u8, name: []const u8) !Tensor(f32) {
    const path = try std.fmt.allocPrint(allocator, "{s}{s}.bin", .{ base, name });
    defer allocator.free(path);
    const file = std.fs.cwd().openFile(path, .{}) catch {
        std.log.err("FATAL: Weight file not found: {s}", .{path});
        return TransformerError.WeightFileNotFound;
    };
    defer file.close();
    const size = try file.getEndPos();
    const num_elements = size / 4;
    const tensor = try Tensor(f32).initUninitialized(allocator, &.{num_elements});
    _ = try file.readAll(std.mem.sliceAsBytes(tensor.data));
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
    pub fn deinit(self: *KVCache) void { self.k_cache.deinit(); self.v_cache.deinit(); }
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
