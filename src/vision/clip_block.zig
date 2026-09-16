//! ClipBlock — bloque transformer del ViT: LN1 → Attn → +res → LN2 → FFN → +res
//! (+ deepstack opcional en Qwen3-VL).
//!
//! Referencias:
//!   - Grafo: llama.cpp models/qwen3vl.cpp:69-161
//!   - FFN sin gate: clip.cpp:587-660 (FFN_GELU: gelu(up(x)) → down)
//!   - Deepstack: qwen3vl.cpp:143-158 (reshape [n_embd·4, n_pos/4] → LN →
//!     fc1 gelu → fc2; features concatenadas al final)
const std = @import("std");
const Tensor = @import("core").Tensor;
const matmul = @import("matmul");
const norm = @import("norm");
const QuantWeight = @import("quant_weight").QuantWeight;
const attn_mod = @import("clip_attention");
const ClipAttention = attn_mod.ClipAttention;
const debugz = @import("debug");

pub const FfnOp = @import("mmproj_config").FfnOp;

pub const ClipBlockError = error{
    MissingWeights,
    ShapeMismatch,
    OutOfMemory,
};

pub const ClipBlock = struct {
    allocator: std.mem.Allocator,
    engine: *matmul.MatmulEngine,

    attn: ClipAttention,

    // LayerNorms (dequantizados a slices f32)
    ln1_w: []f32,
    ln1_b: ?[]f32,
    ln2_w: []f32,
    ln2_b: ?[]f32,

    // FFN: up → gelu → down (sin gate en el ViT Qwen3-VL)
    ff_up_t: Tensor(f32), // [n_ff, n_embd]
    ff_up_b: ?[]f32,
    ff_down_t: Tensor(f32), // [n_embd, n_ff]
    ff_down_b: ?[]f32,
    ffn_op: FfnOp,

    // Deepstack (Qwen3-VL, opcional)
    ds: ?Deepstack = null,

    n_embd: usize,
    n_ff: usize,
    layer_idx: usize,
    /// Qwen2.5-VL: RMS norm (sin bias) en ln1/ln2/deepstack; resto LayerNorm
    use_rms_norm: bool = false,
    /// eps de clip.vision.attention.layer_norm_epsilon (config del mmproj)
    eps: f32 = 1e-6,

    const Self = @This();

    pub const Deepstack = struct {
        norm_w: []f32,
        norm_b: ?[]f32,
        fc1_t: Tensor(f32), // [n_ff, n_embd·4]
        fc1_b: ?[]f32,
        fc2_t: Tensor(f32), // [n_embd·4, n_ff]
        fc2_b: ?[]f32,
        n_ff: usize,
    };

    /// Carga desde el mmproj. `m` = *const MmprojModel (anytype: sin ciclo import).
    pub fn init(
        allocator: std.mem.Allocator,
        engine: *matmul.MatmulEngine,
        m: anytype,
        il: usize,
        n_embd: usize,
        n_head: usize,
        head_dim: usize,
        n_head_kv: usize,
        n_ff: usize,
        ffn_op: FfnOp,
        is_deepstack: bool,
        use_rms_norm: bool,
        eps: f32,
    ) !Self {
        const ln1_w = try attn_mod.dequantF32Slice(allocator, m.blk(il, "ln1.weight") orelse return ClipBlockError.MissingWeights);
        errdefer allocator.free(ln1_w);
        var ln1_b: ?[]f32 = null;
        errdefer if (ln1_b) |b| allocator.free(b);
        if (m.blk(il, "ln1.bias")) |w| ln1_b = try attn_mod.dequantF32Slice(allocator, w);

        const ln2_w = try attn_mod.dequantF32Slice(allocator, m.blk(il, "ln2.weight") orelse return ClipBlockError.MissingWeights);
        errdefer allocator.free(ln2_w);
        var ln2_b: ?[]f32 = null;
        errdefer if (ln2_b) |b| allocator.free(b);
        if (m.blk(il, "ln2.bias")) |w| ln2_b = try attn_mod.dequantF32Slice(allocator, w);

        const up_w = m.blk(il, "ffn_up.weight") orelse return ClipBlockError.MissingWeights;
        var ff_up_t = try m.dequantToF32Transposed(up_w, n_ff, n_embd);
        errdefer ff_up_t.deinit();
        var ff_up_b: ?[]f32 = null;
        errdefer if (ff_up_b) |b| allocator.free(b);
        if (m.blk(il, "ffn_up.bias")) |w| ff_up_b = try attn_mod.dequantF32Slice(allocator, w);

        const down_w = m.blk(il, "ffn_down.weight") orelse return ClipBlockError.MissingWeights;
        var ff_down_t = try m.dequantToF32Transposed(down_w, n_embd, n_ff);
        errdefer ff_down_t.deinit();
        var ff_down_b: ?[]f32 = null;
        errdefer if (ff_down_b) |b| allocator.free(b);
        if (m.blk(il, "ffn_down.bias")) |w| ff_down_b = try attn_mod.dequantF32Slice(allocator, w);

        var attn = try ClipAttention.init(allocator, engine, m, il, n_embd, n_head, head_dim, n_head_kv);
        errdefer attn.deinit();

        // Deepstack (Qwen3-VL): norm + fc1 + fc2 sobre [n_embd·merge², n_pos/merge²]
        var ds: ?Deepstack = null;
        errdefer if (ds) |*d| deinitDeepstack(allocator, d);
        if (is_deepstack) {
            const norm_w = try attn_mod.dequantF32Slice(allocator, m.deepstack(il, "norm.weight") orelse return ClipBlockError.MissingWeights);
            errdefer allocator.free(norm_w);
            var norm_b: ?[]f32 = null;
            errdefer if (norm_b) |b| allocator.free(b);
            if (m.deepstack(il, "norm.bias")) |w| norm_b = try attn_mod.dequantF32Slice(allocator, w);

            // fc1: [n_ff, n_embd·4] — leer dims reales del tensor
            const fc1_w = m.deepstack(il, "fc1.weight") orelse return ClipBlockError.MissingWeights;
            const dsff_shape = fc1_w.info.shape();
            const ds_ff: usize = @intCast(dsff_shape[1]); // dim externa GGUF = out
            const ds_in: usize = @intCast(dsff_shape[0]); // dim contigua = in
            var fc1_t = try m.dequantToF32Transposed(fc1_w, ds_ff, ds_in);
            errdefer fc1_t.deinit();
            var fc1_b: ?[]f32 = null;
            errdefer if (fc1_b) |b| allocator.free(b);
            if (m.deepstack(il, "fc1.bias")) |w| fc1_b = try attn_mod.dequantF32Slice(allocator, w);

            const fc2_w = m.deepstack(il, "fc2.weight") orelse return ClipBlockError.MissingWeights;
            const fc2_shape = fc2_w.info.shape();
            var fc2_t = try m.dequantToF32Transposed(fc2_w, @intCast(fc2_shape[1]), @intCast(fc2_shape[0]));
            errdefer fc2_t.deinit();
            var fc2_b: ?[]f32 = null;
            errdefer if (fc2_b) |b| allocator.free(b);
            if (m.deepstack(il, "fc2.bias")) |w| fc2_b = try attn_mod.dequantF32Slice(allocator, w);

            ds = .{
                .norm_w = norm_w,
                .norm_b = norm_b,
                .fc1_t = fc1_t,
                .fc1_b = fc1_b,
                .fc2_t = fc2_t,
                .fc2_b = fc2_b,
                .n_ff = ds_ff,
            };
        }

        return .{
            .allocator = allocator,
            .engine = engine,
            .attn = attn,
            .ln1_w = ln1_w,
            .ln1_b = ln1_b,
            .ln2_w = ln2_w,
            .ln2_b = ln2_b,
            .ff_up_t = ff_up_t,
            .ff_up_b = ff_up_b,
            .ff_down_t = ff_down_t,
            .ff_down_b = ff_down_b,
            .ffn_op = ffn_op,
            .ds = ds,
            .n_embd = n_embd,
            .n_ff = n_ff,
            .layer_idx = il,
            .use_rms_norm = use_rms_norm,
            .eps = eps,
        };
    }

    pub fn deinit(self: *Self) void {
        self.attn.deinit();
        self.allocator.free(self.ln1_w);
        if (self.ln1_b) |b| self.allocator.free(b);
        self.allocator.free(self.ln2_w);
        if (self.ln2_b) |b| self.allocator.free(b);
        self.ff_up_t.deinit();
        if (self.ff_up_b) |b| self.allocator.free(b);
        self.ff_down_t.deinit();
        if (self.ff_down_b) |b| self.allocator.free(b);
        if (self.ds) |*d| deinitDeepstack(self.allocator, d);
    }

    fn deinitDeepstack(allocator: std.mem.Allocator, d: *Deepstack) void {
        allocator.free(d.norm_w);
        if (d.norm_b) |b| allocator.free(b);
        d.fc1_t.deinit();
        if (d.fc1_b) |b| allocator.free(b);
        d.fc2_t.deinit();
        if (d.fc2_b) |b| allocator.free(b);
    }

    /// Requisito total de scratch del bloque para forward(n_pos):
    /// 4·A (ln1/attn-out/inp_l + reuso) + B (ffn_mid) + attn.scratchNeed.
    pub fn scratchNeed(self: *const Self, n_pos: usize) usize {
        const A = n_pos * self.n_embd;
        const B = n_pos * self.n_ff;
        return 4 * A + B + self.attn.scratchNeed(n_pos);
    }

    /// Forward: `x` [n_pos, n_embd] (input/residual del bloque) →
    /// `out` [n_pos, n_embd]. Si el bloque tiene deepstack, escribe además
    /// `ds_out` [n_pos/merge², n_embd·merge²] (features para el concat final).
    ///
    /// Layout de scratch (sin solapes):
    ///   [0 .. A)                 ln1_out     (A = n_pos·n_embd)
    ///   [A .. 2A)                attn_out
    ///   [2A .. 3A)                inp_l (residual post-attn)
    ///   [3A .. 3A+B)             ffn_mid     (B = n_pos·n_ff)
    ///   [3A+B .. 4A+B)           ffn_out
    ///   [4A+B .. 4A+B+R)         attn_scratch (R: ver ClipAttention.forward)
    pub fn forward(
        self: *const Self,
        x: []const f32, // [n_pos, n_embd] input del bloque
        out: []f32, // [n_pos, n_embd] salida
        n_pos: usize,
        ids: []const [4]i32,
        ds_out: ?[]f32, // deepstack features out (o null si no hay)
        scratch: []f32,
    ) !void {
        const n_embd = self.n_embd;
        const eps = self.eps;

        const A = n_pos * n_embd;
        const B = n_pos * self.n_ff;
        const off_mid = 3 * A;
        const off_ffn_out = 3 * A + B;
        const off_attn_scr = 4 * A + B;

        // ── LN1 → attn → residual 1
        const ln1_out = scratch[0..A];
        if (self.use_rms_norm) {
            rmsNormSlice(x, ln1_out, n_pos, n_embd, self.ln1_w, eps);
        } else {
            layerNormSlice(x, ln1_out, n_pos, n_embd, self.ln1_w, self.ln1_b, eps);
        }
        if (debugz.dbg.dump_mm_input and debugz.dbg.at(.info)) {
            debugz.dbg.printLevel(.info, "[mmproj-ln1-{d}] tok0[3]={d:.4},{d:.4},{d:.4} tok1[3]={d:.4},{d:.4},{d:.4}\n", .{
                self.layer_idx,  ln1_out[0],          ln1_out[1],          ln1_out[2],
                ln1_out[n_embd], ln1_out[n_embd + 1], ln1_out[n_embd + 2],
            });
        }

        const attn_out = scratch[A .. 2 * A];
        try self.attn.forward(ln1_out, attn_out, n_pos, ids, scratch[off_attn_scr..]);
        if (debugz.dbg.dump_mm_input and debugz.dbg.at(.info)) {
            debugz.dbg.printLevel(.info, "[mmproj-aout-{d}] tok0[3]={d:.4},{d:.4},{d:.4}\n", .{
                self.layer_idx, attn_out[0], attn_out[1], attn_out[2],
            });
        }

        const inp_l = scratch[2 * A .. 3 * A];
        for (0..A) |i| inp_l[i] = x[i] + attn_out[i];
        if (debugz.dbg.dump_mm_input and debugz.dbg.at(.info)) {
            debugz.dbg.printLevel(.info, "[mmproj-res1-{d}] tok0[3]={d:.4},{d:.4},{d:.4} last[3]={d:.4},{d:.4},{d:.4}\n", .{
                self.layer_idx,    inp_l[0],          inp_l[1],          inp_l[2],
                inp_l[n_embd - 3], inp_l[n_embd - 2], inp_l[n_embd - 1],
            });
        }

        // ── LN2 → FFN → residual 2   (reusamos attn_out como ln2_out)
        const ln2_out = attn_out;
        if (self.use_rms_norm) {
            rmsNormSlice(inp_l, ln2_out, n_pos, n_embd, self.ln2_w, eps);
        } else {
            layerNormSlice(inp_l, ln2_out, n_pos, n_embd, self.ln2_w, self.ln2_b, eps);
        }

        // FFN: gelu(up(x)) → down (sin gate — clip.cpp:627-630 FFN_GELU else-branch)
        const mid = scratch[off_mid .. off_mid + B];
        const ln2_t_shape = [_]usize{ n_pos, n_embd };
        var ln2_t_strides = [_]usize{ n_embd, 1 };
        const ln2_t = Tensor(f32){
            .data = ln2_out,
            .shape = &ln2_t_shape,
            .strides = &ln2_t_strides,
            .offset = 0,
            .allocator = null,
            .owns_data = false,
        };
        const mid_t_shape = [_]usize{ n_pos, self.n_ff };
        var mid_t_strides = [_]usize{ self.n_ff, 1 };
        var mid_t = Tensor(f32){
            .data = mid,
            .shape = &mid_t_shape,
            .strides = &mid_t_strides,
            .offset = 0,
            .allocator = null,
            .owns_data = false,
        };
        try self.engine.linearProjection(f32, ln2_t, self.ff_up_t, &mid_t);
        if (self.ff_up_b) |b| {
            for (0..n_pos) |t| {
                const row = mid[t * self.n_ff ..][0..self.n_ff];
                for (b, 0..) |bv, i| row[i] += bv;
            }
        }
        // activación
        for (mid) |*v| {
            const f: f32 = v.*;
            v.* = switch (self.ffn_op) {
                .gelu => blk: {
                    const c = f * f * f;
                    const inner = @sqrt(2.0 / std.math.pi) * (f + 0.044715 * c);
                    break :blk f * 0.5 * (1.0 + std.math.tanh(inner));
                },
                .silu => f / (1.0 + @exp(-f)),
                .gelu_quick => f * (1.0 / (1.0 + @exp(-1.702 * f))),
            };
        }

        const ffn_out = scratch[off_ffn_out .. off_ffn_out + A];
        const ffn_out_t_shape = [_]usize{ n_pos, n_embd };
        var ffn_out_t_strides = [_]usize{ n_embd, 1 };
        var ffn_out_t = Tensor(f32){
            .data = ffn_out,
            .shape = &ffn_out_t_shape,
            .strides = &ffn_out_t_strides,
            .offset = 0,
            .allocator = null,
            .owns_data = false,
        };
        try self.engine.linearProjection(f32, mid_t, self.ff_down_t, &ffn_out_t);
        if (self.ff_down_b) |b| {
            for (0..n_pos) |t| {
                const row = ffn_out[t * n_embd ..][0..n_embd];
                for (b, 0..) |bv, i| row[i] += bv;
            }
        }

        // residual 2: out = inpL + ffn_out (qwen3vl.cpp:140)
        for (0..A) |i| out[i] = inp_l[i] + ffn_out[i];
        if (debugz.dbg.dump_mm_input and debugz.dbg.at(.info)) {
            debugz.dbg.printLevel(.info, "[mmproj-lout-{d}] tok0[3]={d:.4},{d:.4},{d:.4} tok1[3]={d:.4},{d:.4},{d:.4}\n", .{
                self.layer_idx, out[0],          out[1],          out[2],
                out[n_embd],    out[n_embd + 1], out[n_embd + 2],
            });
        }

        // ── Deepstack (Qwen3-VL, qwen3vl.cpp:143-158):
        // feat = reshape(out, [n_embd·merge², n_pos/merge²]) → LN →
        // fc1(gelu) → fc2. Los n_pos ya están en orden pixel-shuffle
        // (heredado del input), así que el reshape es un simple view.
        if (self.ds) |*d| {
            const dst = ds_out orelse return ClipBlockError.ShapeMismatch;
            const merge2: usize = 4; // merge² (2²) — hardcoded como el grafo (merge_factor=4)
            const n_rows = n_pos / merge2;
            if (dst.len != n_rows * d.norm_w.len)
                return ClipBlockError.ShapeMismatch;

            // view lineal: out [n_pos·n_embd] ≡ [n_rows, n_embd·merge²]
            // LN por fila (gamma de tamaño n_embd·merge²)
            if (self.use_rms_norm) {
                rmsNormSlice(out, dst, n_rows, d.norm_w.len, d.norm_w, eps);
            } else {
                layerNormSlice(out, dst, n_rows, d.norm_w.len, d.norm_w, d.norm_b, eps);
            }

            // fc1 → gelu → fc2 (n_ff del deepstack puede diferir del FFN)
            const ds_mid = scratch[off_ffn_out + A .. off_ffn_out + A + n_rows * d.n_ff];
            const dst_t_shape = [_]usize{ n_rows, d.norm_w.len };
            var dst_t_strides = [_]usize{ d.norm_w.len, 1 };
            const dst_t = Tensor(f32){
                .data = dst,
                .shape = &dst_t_shape,
                .strides = &dst_t_strides,
                .offset = 0,
                .allocator = null,
                .owns_data = false,
            };
            const ds_mid_t_shape = [_]usize{ n_rows, d.n_ff };
            var ds_mid_t_strides = [_]usize{ d.n_ff, 1 };
            var ds_mid_t = Tensor(f32){
                .data = ds_mid,
                .shape = &ds_mid_t_shape,
                .strides = &ds_mid_t_strides,
                .offset = 0,
                .allocator = null,
                .owns_data = false,
            };
            try self.engine.linearProjection(f32, dst_t, d.fc1_t, &ds_mid_t);
            if (d.fc1_b) |b| {
                for (0..n_rows) |t| {
                    const row = ds_mid[t * d.n_ff ..][0..d.n_ff];
                    for (b, 0..) |bv, i| row[i] += bv;
                }
            }
            for (ds_mid) |*v| {
                const f: f32 = v.*;
                const c = f * f * f;
                const inner = @sqrt(2.0 / std.math.pi) * (f + 0.044715 * c);
                v.* = f * 0.5 * (1.0 + std.math.tanh(inner)); // GELU fijo (qwen3vl.cpp:150)
            }
            const ds_out_t_shape = [_]usize{ n_rows, d.norm_w.len };
            var ds_out_t_strides = [_]usize{ d.norm_w.len, 1 };
            var ds_out_t = Tensor(f32){
                .data = dst,
                .shape = &ds_out_t_shape,
                .strides = &ds_out_t_strides,
                .offset = 0,
                .allocator = null,
                .owns_data = false,
            };
            try self.engine.linearProjection(f32, ds_mid_t, d.fc2_t, &ds_out_t);
            if (d.fc2_b) |b| {
                for (0..n_rows) |t| {
                    const row = dst[t * d.norm_w.len ..][0..d.norm_w.len];
                    for (b, 0..) |bv, i| row[i] += bv;
                }
            }
        }
    }
};

/// RMS norm (Qwen2.5-VL, ggml_rms_norm + mul weight — sin bias/mean):
/// x·w / sqrt(mean(x²) + eps). Sobre slices [n_pos, n_embd].
pub fn rmsNormSlice(
    x: []const f32,
    out: []f32,
    n_pos: usize,
    n_embd: usize,
    gamma: []const f32,
    eps: f32,
) void {
    for (0..n_pos) |t| {
        const row = x[t * n_embd ..][0..n_embd];
        const dst = out[t * n_embd ..][0..n_embd];
        var ssq: f64 = 0;
        for (row) |v| ssq += @as(f64, v) * v;
        const inv: f32 = @floatCast(1.0 / @sqrt(ssq / @as(f64, @floatFromInt(n_embd)) + eps));
        for (0..n_embd) |i| dst[i] = row[i] * inv * gamma[i];
    }
}

/// LayerNorm sobre slices [n_pos, n_embd] con gamma/beta f32.
/// (norm.zig:36 usa Tensor; aquí trabajamos con slices crudos.)
pub fn layerNormSlice(
    x: []const f32,
    out: []f32,
    n_pos: usize,
    n_embd: usize,
    gamma: []const f32,
    beta: ?[]const f32,
    eps: f32,
) void {
    for (0..n_pos) |t| {
        const row = x[t * n_embd ..][0..n_embd];
        const dst = out[t * n_embd ..][0..n_embd];
        var mean: f32 = 0;
        for (row) |v| mean += v;
        mean /= @floatFromInt(n_embd);
        var var_: f32 = 0;
        for (row) |v| var_ += (v - mean) * (v - mean);
        var_ /= @floatFromInt(n_embd);
        const inv_std = 1.0 / @sqrt(var_ + eps);
        for (0..n_embd) |i| {
            const b: f32 = if (beta) |bb| bb[i] else 0.0;
            dst[i] = (row[i] - mean) * inv_std * gamma[i] + b;
        }
    }
}
