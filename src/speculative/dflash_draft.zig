//! 5.2 (lane-b1) — Draft-model DFlash/DSpark del sidecar: KV-inject + denoise.

//!

//! Oráculo: llama.cpp `src/models/dflash.cpp` (leído completo, semántica exacta):

//!

//! **Modo embd (KV-inject, prefill del target):** las features fusionadas del

//! encoder (fc + RMSNorm sobre las taps del target) entran a CADA capa blk.*

//! por `wk`/`wv` → k_norm + RoPE → append DIRECTO al KV-cache del draft en

//! las posiciones del target. NO se computa atención, NO se toca q/o/ffn.

//! → `AttentionLayer.appendKVOnly` (hybrid_attn.zig) es ese camino.

//!

//! **Modo token (denoise, drafting):** ids `[anchor, MASK×(bs-1)]` →

//! embeddings del TARGET (heredados) → forward de las capas blk.* con

//! atención NO-causal (causal=false, flag de AttentionLayer) →

//! output_norm + lm_head (heredados del target) → logits por posición →

//! top-k con truncado p_min.

//!

//! El sidecar NO trae tok_embd ni output: `SidecarDraft` (gguf_model.zig)

//! hereda ambos del target.

//!

//! C6.x (5.1, lane-a) sube encoder+denoise a kernels device; este módulo

//! es el camino CPU de referencia.



const std = @import("std");

const gguf = @import("gguf");

const gguf_model = @import("gguf_model");

const model_config = @import("model_config");

const Tensor = @import("core").Tensor;

const hybrid_layer = @import("hybrid_layer");

const paged_attn = @import("paged_attention");

const quant_weight = @import("quant_weight");

const QuantWeight = quant_weight.QuantWeight;

const debugz = @import("debug");



pub const DflashError = error{

    SidecarNotDflash,

    MissingTensor,

    NotImplemented,
};



/// ID del token MASK del denoiser (dflash.mask_token_id del GGUF).

pub const MASK_TOKEN_DEFAULT: u32 = 0xFFFF_FFFE;



pub const DflashDraftModel = struct {

    allocator: std.mem.Allocator,

    sidecar: *gguf_model.SidecarDraft,

    /// Capas blk.* del sidecar como HybridLayers (chasis U1: autodetecta

    /// no_gate/n_kv_head por geometría). Todas con causal=false (denoise).

    layers: []hybrid_layer.HybridLayer,

    /// KV paginado compartido con el target (el draft usa SU secuencia).

    draft_kv: *paged_attn.PagedKVCache,

    draft_bt: *paged_attn.BlockTable,

    /// fc del encoder: GGUF [in=n_extract*n_embd, out=n_embd].

    enc_fc: QuantWeight,

    /// fc dequant transposed [n_embd, n_in] (para linearProjection).

    enc_fc_w32: []f32,

    /// output_norm del encoder (tras fc), [n_embd] f32.

    enc_norm: Tensor(f32),

    /// output_norm del DECODER (final del sidecar; el oráculo lo trae).

    out_norm: Tensor(f32),

    cfg: model_config.ModelConfig,

    /// dflash.block_size (default 16).

    block_size: usize,

    /// Capas del target cuyas features alimenta el encoder (n_extract).

    target_layers: []i32,

    /// p_min de truncado (CLI --spec-p-min).

    p_min: f32,

    /// ID del token MASK.

    mask_token: u32,

    /// Embedding del target [vocab_ext, n_embd] f16 (heredado).

    tok_embd: Tensor(f16),

    /// lm_head del target (heredado) f16 [vocab, n_embd].

    lm_head: Tensor(f16),



    pub fn init(

        allocator: std.mem.Allocator,

        sidecar: *gguf_model.SidecarDraft,

        draft_kv: *paged_attn.PagedKVCache,

        backend: anytype,

        p_min: f32,

    ) !DflashDraftModel {

        const g = &sidecar.model.file;

        const cfg = sidecar.model.config;



        // dflash.block_size

        var block_size: usize = 16;

        if (g.getMeta("dflash.block_size")) |v| {

            switch (v) {

                .uint32 => |x| block_size = @intCast(x),

                .int32 => |x| block_size = @intCast(x),

                .uint64 => |x| block_size = @intCast(x),

                .int64 => |x| block_size = @intCast(x),

                .string => |s| block_size = std.fmt.parseInt(usize, s, 10) catch 16,

                else => {},

            }

        }

        if (block_size == 0) return DflashError.SidecarNotDflash;



        // target_layers es OBLIGATORIO (el oráculo lo exige en hparams).

        if (sidecar.target_layers.len == 0) return DflashError.SidecarNotDflash;



        var mask_token: u32 = MASK_TOKEN_DEFAULT;

        if (g.getMeta("dflash.mask_token_id")) |v| {

            switch (v) {

                .uint32 => |x| mask_token = x,

                .int32 => |x| mask_token = @intCast(x),

                .uint64 => |x| mask_token = @intCast(x),

                else => {},

            }

        }



        // Contar capas blk.* válidas (attn_q presente).

        var n_layers: usize = 0;

        for (0..cfg.block_count * 2) |i| {

            var buf: [64]u8 = undefined;

            const nm = std.fmt.bufPrint(&buf, "blk.{d}.attn_q.weight", .{i}) catch unreachable;

            if (g.getTensor(nm) != null) n_layers += 1 else break;

        }

        if (n_layers == 0) return DflashError.MissingTensor;



        const seq_draft = try draft_kv.createSequence();

        const bt = draft_kv.getBlockTableMut(seq_draft) orelse return DflashError.MissingTensor;



        const layers = try allocator.alloc(hybrid_layer.HybridLayer, n_layers);

        errdefer allocator.free(layers);

        for (layers, 0..) |*l, i| {

            l.* = try hybrid_layer.HybridLayer.init(

                allocator,

                i,

                hybrid_layer.HybridLayerParams.fromModelConfig(cfg, cfg.block_count * block_size * 4),

                true,

                backend,

                draft_kv,

                bt,

                null, // CPU: sin motor GPU (C6.x/5.1 lo sube)

            );

            errdefer l.deinit();

            try l.loadWeightsFromGguf(g, null);

            // 5.2: denoise — TODAS las capas del draft son NO-causales.

            if (l.attn_layer) |*at| at.causal = false;

        }



        // Encoder fc: GGUF [n_extract*n_embd, n_embd] → transposed f32.

        const fc_info = g.getTensor("fc.weight") orelse return DflashError.MissingTensor;

        const enc_fc = QuantWeight.init(fc_info, g.tensorData(fc_info));

        // GGUF: dims[0]=contigua (in), dims[1]=filas (out) — misma convención

        // que w_q/w_k en hybrid_attn (q_dim_real = dims[1]).

        const n_in: usize = @intCast(fc_info.dims[0]);

        if (n_in != sidecar.target_layers.len * cfg.embedding_length) {

            debugz.dbg.printLevel(.info, "[dflash] fc in={d} ≠ target_layers({d})×n_embd({d})\n", .{ n_in, sidecar.target_layers.len, cfg.embedding_length });

            return DflashError.SidecarNotDflash;

        }

        const enc_fc_w32 = try allocator.alloc(f32, fc_info.numel());

        errdefer allocator.free(enc_fc_w32);

        enc_fc.dequantToF32Transposed(enc_fc_w32);



        // Naming REAL del sidecar (verificado qwen35-9b-dflash): "enc.output_norm.weight"

        // (punto) — el conv de llama.cpp usa LLM_TENSOR_ENC_OUTPUT_NORM = enc.output_norm.

        var enc_norm = try loadF32OrOnes(allocator, g, "enc.output_norm.weight", cfg.embedding_length);

        errdefer enc_norm.deinit();

        var out_norm = try loadF32Tensor(allocator, g, "output_norm.weight");

        errdefer out_norm.deinit();



        var tok_embd = try sidecar.loadEmbedding();

        errdefer tok_embd.deinit();

        var lm_head = try sidecar.loadLmHead();

        errdefer lm_head.deinit();



        return .{

            .allocator = allocator,

            .sidecar = sidecar,

            .layers = layers,

            .draft_kv = draft_kv,

            .draft_bt = bt,

            .enc_fc = enc_fc,

            .enc_fc_w32 = enc_fc_w32,

            .enc_norm = enc_norm,

            .out_norm = out_norm,

            .cfg = cfg,

            .block_size = block_size,

            .target_layers = sidecar.target_layers,

            .p_min = p_min,

            .mask_token = mask_token,

            .tok_embd = tok_embd,

            .lm_head = lm_head,

        };

    }



    pub fn deinit(self: *DflashDraftModel) void {

        for (self.layers) |*l| l.deinit();

        self.allocator.free(self.layers);

        self.allocator.free(self.enc_fc_w32);

        self.enc_norm.deinit();

        self.out_norm.deinit();

        self.tok_embd.deinit();

        self.lm_head.deinit();

    }



    /// Posiciones inyectadas en el KV del draft.

    pub fn kvLen(self: *const DflashDraftModel) usize {

        return self.draft_bt.num_tokens;

    }



    /// Trunca el KV del draft a `n` (tras rechazo del verify — oráculo:

    /// "los drafts rechazados se revierten truncando el cache").

    pub fn rollbackTo(self: *DflashDraftModel, n: usize) !void {

        if (self.draft_bt.num_tokens > n) {

            try self.draft_bt.truncate(self.draft_kv.block_alloc, n);

        }

    }



    /// Encoder: taps del target [n_tok, n_extract*n_embd] → fc → RMSNorm →

    /// fused [n_tok, n_embd]. Gamma = enc_norm.

    pub fn encoderForward(self: *DflashDraftModel, matmul_engine: anytype, taps: []const f32, n_tok: usize, out_fused: []f32) !void {

        const n_embd = self.cfg.embedding_length;

        const n_in = self.target_layers.len * n_embd;

        const engine = &self.layers[0].attn_layer.?.matmul_engine;

        _ = engine;

        _ = matmul_engine;



        var w_shape = [_]usize{ n_embd, n_in };

        var w_strides = [_]usize{ n_in, 1 };

        const w32 = Tensor(f32){

            .data = self.enc_fc_w32,

            .shape = &w_shape,

            .strides = &w_strides,

            .offset = 0,

            .allocator = null,

            .owns_data = false,

        };

        var x_shape = [_]usize{ n_tok, n_in };

        var x_strides = [_]usize{ n_in, 1 };

        const x = Tensor(f32){

            .data = @constCast(taps),

            .shape = &x_shape,

            .strides = &x_strides,

            .offset = 0,

            .allocator = null,

            .owns_data = false,

        };

        var y_shape = [_]usize{ n_tok, n_embd };

        var y_strides = [_]usize{ n_embd, 1 };

        var y = Tensor(f32){

            .data = out_fused,

            .shape = &y_shape,

            .strides = &y_strides,

            .offset = 0,

            .allocator = null,

            .owns_data = false,

        };

        try self.layers[0].attn_layer.?.matmul_engine.linearProjection(f32, x, w32, &y);

        for (0..n_tok) |t| {

            rmsNormInPlace(out_fused[t * n_embd ..][0..n_embd], self.enc_norm.data, self.cfg.layer_norm_rms_epsilon);

        }

    }



    /// **Modo embd — KV-inject.** fused: [n_tok, n_embd] features del

    /// encoder (YA fusionadas). Por capa blk.*: appendKVOnly (wk/wv →

    /// k_norm → RoPE → pool). Sin atención.

    pub fn kvInject(self: *DflashDraftModel, fused: []const f32, n_tok: usize, start_pos: usize) !void {

        const n_embd = self.cfg.embedding_length;

        var x_shape = [_]usize{ n_tok, n_embd };

        var x_strides = [_]usize{ n_embd, 1 };

        const x = Tensor(f32){

            .data = @constCast(fused),

            .shape = &x_shape,

            .strides = &x_strides,

            .offset = 0,

            .allocator = null,

            .owns_data = false,

        };

        for (self.layers) |*l| {

            const at = &(l.attn_layer orelse continue);

            try at.appendKVOnly(x, start_pos);

        }

    }



    /// **Modo token — denoise.** [anchor, MASK×(bs-1)] → forward no-causal

    /// → out_logits [(bs-1)×vocab]. Devuelve filas escritas (bs-1).

    pub fn denoiseDraft(self: *DflashDraftModel, anchor: u32, out_logits: []f32) !usize {

        const n_embd = self.cfg.embedding_length;

        const bs = self.block_size;

        if (bs < 2 or bs > 256) return DflashError.NotImplemented;



        var ids_buf: [256]u32 = undefined;

        ids_buf[0] = anchor;

        for (1..bs) |i| ids_buf[i] = self.mask_token;



        // embeddings heredados: [bs, n_embd] f16 → f32

        const x = try self.allocator.alloc(f32, bs * n_embd);

        defer self.allocator.free(x);

        const vocab_rows = self.tok_embd.shape[0];

        for (0..bs) |i| {

            const row = ids_buf[i] % @as(u32, @intCast(vocab_rows));

            const src = self.tok_embd.data[@as(usize, row) * n_embd ..][0..n_embd];

            for (src, 0..) |s, j| x[i * n_embd + j] = @floatCast(s);

        }



        // forward no-causal (causal=false en init). start_pos = KV inyectado.

        // Contrato del forward CPU: los bloques del rango [start_pos, +bs) deben

        // existir en el BlockTable ANTES de llamar (el scheduler del pipeline lo

        // hace; denoise es autocontenido: asegura el mapping él mismo).

        const start_pos = self.draft_bt.num_tokens;

        try self.draft_bt.appendTokens(self.draft_kv.block_alloc, bs);

        var buf_a = try Tensor(f32).alloc(self.allocator, &.{ bs, n_embd });

        defer buf_a.deinit();

        var buf_b = try Tensor(f32).alloc(self.allocator, &.{ bs, n_embd });

        defer buf_b.deinit();

        @memcpy(buf_a.data, x);

        var cur = &buf_a;

        var nxt = &buf_b;

        for (self.layers) |*l| {

            try l.forward(cur.*, nxt, start_pos, bs, null);

            const t = cur;

            cur = nxt;

            nxt = t;

        }



        // output_norm + lm_head por posición 1..bs-1 → [(bs-1), vocab]

        if (out_logits.len % (bs - 1) != 0) return DflashError.NotImplemented;

        const vocab = out_logits.len / (bs - 1);

        const h = try self.allocator.alloc(f32, n_embd);

        defer self.allocator.free(h);

        for (1..bs) |i| {

            @memcpy(h, cur.data[i * n_embd ..][0..n_embd]);

            rmsNormInPlace(h, self.out_norm.data, self.cfg.layer_norm_rms_epsilon);

            cpuLmHeadRow(h, self.lm_head.data, out_logits[(i - 1) * vocab ..][0..vocab]);

        }

        return bs - 1;

    }

    /// **Modo token batched — denoise no-causal.** N anchors →

    /// forward no-causal por cada uno → logits [(N×(bs-1)), vocab].

    /// Usa el KV del draft creciente (cada forward añade bs tokens).

    /// Contrato: `out_logits.len == anchors.len * (bs - 1) * vocab`.

    pub fn denoiseDraftBatched(

        self: *DflashDraftModel,

        anchors: []const u32,

        out_logits: []f32,

    ) !usize {

        const n_embd = self.cfg.embedding_length;

        const bs = self.block_size;

        if (bs < 2 or bs > 256) return DflashError.NotImplemented;

        if (anchors.len == 0) return 0;

        const vocab_rows = self.tok_embd.shape[0];

        const vocab = self.cfg.vocab_size;

        const expected = anchors.len * (bs - 1) * vocab;

        if (out_logits.len != expected) return DflashError.NotImplemented;



        // Buffers reusables para embeddings y activaciones.

        var x_buf = try self.allocator.alloc(f32, bs * n_embd);

        defer self.allocator.free(x_buf);

        var buf_a = try Tensor(f32).alloc(self.allocator, &.{ bs, n_embd });

        defer buf_a.deinit();

        var buf_b = try Tensor(f32).alloc(self.allocator, &.{ bs, n_embd });

        defer buf_b.deinit();

        const h = try self.allocator.alloc(f32, n_embd);

        defer self.allocator.free(h);



        // start_pos inicial: KV actual antes de cualquier denoise.

        var start_pos = self.draft_bt.num_tokens;

        var out_offset: usize = 0;

        for (anchors) |anchor| {

            // ids [anchor, MASK×(bs-1)]

            var ids_buf: [256]u32 = undefined;

            ids_buf[0] = anchor;

            for (1..bs) |i| ids_buf[i] = self.mask_token;



            // embeddings [bs, n_embd] f16 → f32

            for (0..bs) |i| {

                const row = ids_buf[i] % @as(u32, @intCast(vocab_rows));

                const src = self.tok_embd.data[@as(usize, row) * n_embd ..][0..n_embd];

                for (src, 0..) |s, j| x_buf[i * n_embd + j] = @floatCast(s);

            }



            // Asegurar bloques en BlockTable para este denoise.

            if (self.draft_bt.num_tokens < start_pos + bs) {

                try self.draft_bt.appendTokens(self.draft_kv.block_alloc, start_pos + bs - self.draft_bt.num_tokens);

            }



            // Forward no-causal de las capas del sidecar.

            @memcpy(buf_a.data, x_buf);

            var cur = &buf_a;

            var nxt = &buf_b;

            for (self.layers) |*l| {

                try l.forward(cur.*, nxt, start_pos, bs, null);

                const t = cur;

                cur = nxt;

                nxt = t;

            }



            // output_norm + lm_head por posición 1..bs-1.

            for (1..bs) |i| {

                @memcpy(h, cur.data[i * n_embd ..][0..n_embd]);

                rmsNormInPlace(h, self.out_norm.data, self.cfg.layer_norm_rms_epsilon);

                cpuLmHeadRow(h, self.lm_head.data, out_logits[out_offset ..][0..vocab]);

                out_offset += vocab;

            }

            start_pos += bs;

        }

        return anchors.len * (bs - 1);

    }





fn rmsNormInPlace(x: []f32, gamma: []const f32, eps: f32) void {

    var s: f64 = 0;

    for (x) |v| s += @as(f64, v) * v;

    const inv = @as(f32, @floatCast(1.0 / @sqrt(s / @as(f64, @floatFromInt(x.len)) + eps)));

    for (x, 0..) |*v, i| v.* = v.* * inv * gamma[i];

}



/// GEMV fila: h [n_embd] · lm_head [vocab, n_embd] f16 → out [vocab].

fn cpuLmHeadRow(x: []const f32, head_f16: []const f16, out: []f32) void {

    const n_embd = x.len;

    for (0..out.len) |v| {

        const w = head_f16[v * n_embd ..][0..n_embd];

        var acc: f32 = 0;

        for (x, 0..) |xv, j| acc += xv * @as(f32, @floatCast(w[j]));

        out[v] = acc;

    }

}



fn onesTensor(allocator: std.mem.Allocator, n: usize) !Tensor(f32) {

    const t = try Tensor(f32).initUninitialized(allocator, &.{n});

    @memset(t.data, 1.0);

    return t;

}



fn loadF32Tensor(allocator: std.mem.Allocator, g: *const gguf.GgufFile, name: []const u8) !Tensor(f32) {

    const info = g.getTensor(name) orelse return DflashError.MissingTensor;

    const numel: usize = @intCast(info.numel());

    const f32buf = try allocator.alloc(f32, numel);

    defer allocator.free(f32buf);

    try gguf.dequantTensor(info, g.tensorData(info), f32buf);

    const t = try Tensor(f32).initUninitialized(allocator, &.{numel});

    @memcpy(t.data, f32buf);

    return t;

}



fn loadF32OrOnes(allocator: std.mem.Allocator, g: *const gguf.GgufFile, name: []const u8, numel: usize) !Tensor(f32) {

    if (g.getTensor(name) != null) return loadF32Tensor(allocator, g, name);

    return onesTensor(allocator, numel);

}



test "rmsNormInPlace: vector uniforme ⇒ valor constante" {

    const allocator = std.testing.allocator;

    const x = try allocator.alloc(f32, 4);

    defer allocator.free(x);

    @memcpy(x, &[_]f32{ 2.0, 2.0, 2.0, 2.0 });

    const gamma = try allocator.alloc(f32, 4);

    defer allocator.free(gamma);

    @memset(gamma, 1.0);

    rmsNormInPlace(x, gamma[0..], 1e-6);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), x[0], 1e-4);

}





test "cpuLmHeadRow: producto punto correcto" {

    const allocator = std.testing.allocator;

    const w = try allocator.alloc(f16, 6);

    defer allocator.free(w);

    // vocab=2, n_embd=3: fila0 = [1,0,2], fila1 = [0,1,0]

    w[0] = 1;

    w[1] = 0;

    w[2] = 2;

    w[3] = 0;

    w[4] = 1;

    w[5] = 0;

    const x = [_]f32{ 3, 4, 5 };

    const out = try allocator.alloc(f32, 2);

    defer allocator.free(out);

    cpuLmHeadRow(&x, w, out);

    try std.testing.expectApproxEqAbs(@as(f32, 13), out[0], 1e-5); // 3*1+0+5*2

    try std.testing.expectApproxEqAbs(@as(f32, 4), out[1], 1e-5); // 0+4*1+0

}



};
