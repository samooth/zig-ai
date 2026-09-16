//! Qwen3-VL merger — proyección final de features del ViT al espacio del
//! LLM. Port de qwen3vl.cpp:163-180 + weights clip.cpp:2199-2205.
//!
//!   embeddings = post_ln(vit_out)                    // v.post_ln (merger.norm)
//!   embeddings = reshape [n_embd·4, n_pos/4]          // merge 4→1 real
//!   embeddings = mm.2( gelu( mm.0(embeddings) ) )    // ⚠ fc1=mm.0, fc2=mm.2
//!   [si deepstack] concat沿 dim features               // [proj·(1+n_ds), T]
//!
//! Output final: [n_tokens, projection_dim·(1+n_deepstack)] donde
//! n_tokens = n_pos/4 (grid merge) — clip.cpp:4990-4997.
const std = @import("std");
const Tensor = @import("core").Tensor;
const matmul = @import("matmul");
const block_mod = @import("clip_block");
const attn_mod = @import("clip_attention");
const debugz = @import("debug");

pub const ProjectorError = error{
    MissingWeights,
    ShapeMismatch,
    OutOfMemory,
};

pub const Qwen3VlProjector = struct {
    allocator: std.mem.Allocator,
    engine: *matmul.MatmulEngine,

    post_ln_w: []f32,
    post_ln_b: ?[]f32,

    /// mm.0 (linear_fc1): [proj_hidden, n_embd·4] — dims reales del GGUF
    mm0_t: Tensor(f32),
    mm0_b: ?[]f32,
    /// mm.2 (linear_fc2): [projection_dim, proj_hidden]
    mm2_t: Tensor(f32),
    mm2_b: ?[]f32,

    /// Dim de features de salida por token (sin deepstack: projection_dim)
    out_dim: usize,
    /// Nº de capas deepstack reales (true-count)
    n_ds: usize,
    /// n_ff intermedio del merger
    hidden_dim: usize,
    in_dim: usize, // n_embd·4 (merge²·n_embd)

    const Self = @This();

    /// `n_ds` = nº de capas deepstack (para calcular out_dim total).
    /// `m` = *const MmprojModel.
    pub fn init(
        allocator: std.mem.Allocator,
        engine: *matmul.MatmulEngine,
        m: anytype,
        n_embd: usize,
        n_ds: usize,
    ) !Self {
        const ln_w = try attn_mod.dequantF32Slice(allocator, m.postLn("weight") orelse return ProjectorError.MissingWeights);
        errdefer allocator.free(ln_w);
        var ln_b: ?[]f32 = null;
        errdefer if (ln_b) |b| allocator.free(b);
        if (m.postLn("bias")) |w| ln_b = try attn_mod.dequantF32Slice(allocator, w);

        // ⚠ mm.0 = fc1, mm.2 = fc2 (NO mm.1) — clip.cpp:2201-2202
        const mm0_w = m.mm(0, "weight") orelse return ProjectorError.MissingWeights;
        const s0 = mm0_w.info.shape();
        const in_dim: usize = @intCast(s0[0]); // n_embd·4 (contigua)
        const hidden: usize = @intCast(s0[1]); // out (externa)
        var mm0_t = try m.dequantToF32Transposed(mm0_w, hidden, in_dim);
        errdefer mm0_t.deinit();
        var mm0_b: ?[]f32 = null;
        errdefer if (mm0_b) |b| allocator.free(b);
        if (m.mm(0, "bias")) |w| mm0_b = try attn_mod.dequantF32Slice(allocator, w);

        const mm2_w = m.mm(2, "weight") orelse return ProjectorError.MissingWeights;
        const s2 = mm2_w.info.shape();
        const proj_dim: usize = @intCast(s2[1]);
        if (s2[0] != hidden) return ProjectorError.ShapeMismatch; // fc2 in == fc1 out
        var mm2_t = try m.dequantToF32Transposed(mm2_w, proj_dim, hidden);
        errdefer mm2_t.deinit();
        var mm2_b: ?[]f32 = null;
        errdefer if (mm2_b) |b| allocator.free(b);
        if (m.mm(2, "bias")) |w| mm2_b = try attn_mod.dequantF32Slice(allocator, w);

        if (in_dim != n_embd * 4) return ProjectorError.ShapeMismatch;

        return .{
            .allocator = allocator,
            .engine = engine,
            .post_ln_w = ln_w,
            .post_ln_b = ln_b,
            .mm0_t = mm0_t,
            .mm0_b = mm0_b,
            .mm2_t = mm2_t,
            .mm2_b = mm2_b,
            .out_dim = proj_dim * (1 + n_ds),
            .n_ds = n_ds,
            .hidden_dim = hidden,
            .in_dim = in_dim,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.post_ln_w);
        if (self.post_ln_b) |b| self.allocator.free(b);
        self.mm0_t.deinit();
        if (self.mm0_b) |b| self.allocator.free(b);
        self.mm2_t.deinit();
        if (self.mm2_b) |b| self.allocator.free(b);
    }

    /// Project: `vit_out` [n_pos, n_embd] (salida del encoder, orden
    /// pixel-shuffle) + `ds_feats` (si hay: [n_ds][n_pos/4, n_embd·4·?])
    /// → `out` [n_tokens, out_dim] con n_tokens = n_pos/4.
    ///
    /// NOTA: el reshape [n_embd·4, n_pos/4] del grafo ggml sobre el layout
    /// [n_embd, n_pos] equivale a un simple view lineal porque los 4
    /// patches de cada merge-block son contiguos en n_pos (pixel-shuffle
    /// del input) — ggml reshape sobre [ne0=n_embd, ne1=n_pos]:
    ///   view [n_embd·4, n_pos/4] toma 4 columnas consecutivas por fila.
    /// En nuestro layout row-major [n_pos, n_embd] el equivalente es: por
    /// cada grupo de 4 tokens consecutivos, concatenar sus n_embd en UNA
    /// fila de n_embd·4. (Es decir: un memcpy por bloques de 4 filas.)
    pub fn project(
        self: *const Self,
        vit_out: []const f32, // [n_pos, n_embd] row-major pos-major
        ds_feats: []const []const f32, // [n_ds] cada una [n_tokens, ds_dim]
        out: []f32, // [n_tokens, out_dim]
        n_pos: usize,
        n_embd: usize,
        scratch: []f32,
    ) !void {
        const n_tokens = n_pos / 4;
        const eps = 1e-6;

        // 1. post_ln sobre vit_out (in-place sobre scratch)
        if (scratch.len < n_pos * n_embd + n_tokens * self.hidden_dim + n_tokens * self.in_dim)
            return ProjectorError.ShapeMismatch;
        const ln_out = scratch[0 .. n_pos * n_embd];
        block_mod.layerNormSlice(vit_out, ln_out, n_pos, n_embd, self.post_ln_w, self.post_ln_b, eps);

        // 2. merge 4→1: [n_pos, n_embd] → [n_tokens, n_embd·4]
        const merged = scratch[n_pos * n_embd .. n_pos * n_embd + n_tokens * self.in_dim];
        for (0..n_tokens) |t| {
            const dst = merged[t * self.in_dim ..][0..self.in_dim];
            for (0..4) |k| {
                const src = ln_out[(t * 4 + k) * n_embd ..][0..n_embd];
                @memcpy(dst[k * n_embd ..][0..n_embd], src);
            }
        }

        // 3. mm.0 → gelu → mm.2
        const mid = scratch[n_pos * n_embd + n_tokens * self.in_dim ..][0 .. n_tokens * self.hidden_dim];
        const merged_t_shape = [_]usize{ n_tokens, self.in_dim };
        var merged_t_strides = [_]usize{ self.in_dim, 1 };
        const merged_t = Tensor(f32){
            .data = merged,
            .shape = &merged_t_shape,
            .strides = &merged_t_strides,
            .offset = 0,
            .allocator = null,
            .owns_data = false,
        };
        const mid_t_shape = [_]usize{ n_tokens, self.hidden_dim };
        var mid_t_strides = [_]usize{ self.hidden_dim, 1 };
        var mid_t = Tensor(f32){
            .data = mid,
            .shape = &mid_t_shape,
            .strides = &mid_t_strides,
            .offset = 0,
            .allocator = null,
            .owns_data = false,
        };
        try self.engine.linearProjection(f32, merged_t, self.mm0_t, &mid_t);
        if (self.mm0_b) |b| {
            for (0..n_tokens) |t| {
                const row = mid[t * self.hidden_dim ..][0..self.hidden_dim];
                for (b, 0..) |bv, i| row[i] += bv;
            }
        }
        for (mid) |*v| {
            const f: f32 = v.*;
            const c = f * f * f;
            const inner = @sqrt(2.0 / std.math.pi) * (f + 0.044715 * c);
            v.* = f * 0.5 * (1.0 + std.math.tanh(inner)); // GELU (qwen3vl.cpp:176)
        }

        // out principal: [n_tokens, proj_dim]
        const proj_dim = self.mm2_t.shape[0];
        // Escribir la proyección en la PRIMERA sub-columna de out
        // (mm2 produce [n_tokens, proj_dim]; deepstack concat después).
        // Para evitar un buffer intermedio: proyectar a un view de out.
        // Zig 0.16 Tensor no soporta view de sub-columna fácilmente ⇒
        // usamos un buffer temporal si hay deepstack, si no directo.
        if (ds_feats.len > 0) {
            const tmp = scratch[n_pos * n_embd + n_tokens * self.in_dim + n_tokens * self.hidden_dim ..][0 .. n_tokens * proj_dim];
            const tmp_t_shape = [_]usize{ n_tokens, proj_dim };
            var tmp_t_strides = [_]usize{ proj_dim, 1 };
            var tmp_t = Tensor(f32){
                .data = tmp,
                .shape = &tmp_t_shape,
                .strides = &tmp_t_strides,
                .offset = 0,
                .allocator = null,
                .owns_data = false,
            };
            try self.engine.linearProjection(f32, mid_t, self.mm2_t, &tmp_t);
            if (self.mm2_b) |b| {
                for (0..n_tokens) |t| {
                    const row = tmp[t * proj_dim ..][0..proj_dim];
                    for (b, 0..) |bv, i| row[i] += bv;
                }
            }
            // out = [tmp | ds_feat_0 | ds_feat_1 | ...] por fila
            for (0..n_tokens) |t| {
                const dst = out[t * self.out_dim ..][0..self.out_dim];
                @memcpy(dst[0..proj_dim], tmp[t * proj_dim ..][0..proj_dim]);
                var off = proj_dim;
                for (ds_feats) |ds| {
                    const ds_row = ds[t * (ds.len / n_tokens) ..][0 .. ds.len / n_tokens];
                    @memcpy(dst[off .. off + ds_row.len], ds_row);
                    off += ds_row.len;
                }
            }
        } else {
            const direct_shape = [_]usize{ n_tokens, proj_dim };
            var direct_strides = [_]usize{ proj_dim, 1 };
            var direct = Tensor(f32){
                .data = out,
                .shape = &direct_shape,
                .strides = &direct_strides,
                .offset = 0,
                .allocator = null,
                .owns_data = false,
            };
            try self.engine.linearProjection(f32, mid_t, self.mm2_t, &direct);
            if (self.mm2_b) |b| {
                for (0..n_tokens) |t| {
                    const row = out[t * proj_dim ..][0..proj_dim];
                    for (b, 0..) |bv, i| row[i] += bv;
                }
            }
        }
    }
};
