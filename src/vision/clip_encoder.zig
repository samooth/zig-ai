//! ClipEncoder — orquestador del encoder vision Qwen2/3-VL (mmproj).
//!
//! Pipeline completo (port de models/qwen3vl.cpp:3-186):
//!   1. preprocess: RGB → smartResize → CHW f32 normalizado
//!   2. Conv2D patch embed (kernel 0; still image — n_batch==1,
//!      qwen2vl.cpp:12-16: sólo patch_embeddings_0)
//!   3. Pixel-shuffle spatial merge del input (qwen3vl.cpp:18-31):
//!      [w, h, c] → reordenar patches en bloques 2×2 consecutivos
//!   4. + patch_bias + position_embd (bilinear resize, ALIGN_CORNERS,
//!      qwen3vl.cpp:34-52)
//!   5. pre_ln (si existe)
//!   6. N× ClipBlock (LN→attn(MRoPE vision)→LN→FFN)
//!   7. post_ln + merger (projectors/qwen3vl.zig)
//!
//! Diferencia vs Qwen2-VL (además de deepstack): el pos-emb ABSOLUTO
//! aprendido con resize bilinear — exclusivo de Qwen3-VL
//! (resize_position_embeddings, clip-graph.h:81).
const std = @import("std");
const Tensor = @import("core").Tensor;
const matmul = @import("matmul");
const gguf = @import("gguf");
const mmproj_config = @import("mmproj_config");
const debugz = @import("debug");
const conv2d = @import("conv2d");
const block_mod = @import("clip_block");
const ClipBlock = block_mod.ClipBlock;
const attn_mod = @import("clip_attention");
const preprocess_mod = @import("preprocess");
const mrope = @import("mrope_vision");
const timez = @import("time");
const proj = @import("qwen3vl_projector");
const clip_gpu = @import("clip_gpu");
const cudaz = @import("cudaz");

pub const EncoderError = error{
    UnsupportedProjector,
    MissingWeights,
    ShapeMismatch,
    PreprocessFailed,
    OutOfMemory,
};

pub const EncodedImage = struct {
    /// Embeddings finales de imagen: [n_tokens, out_dim]
    /// n_tokens = (grid_x/merge)·(grid_y/merge); out_dim = proj·(1+n_ds)
    embeddings: []f32,
    n_tokens: usize,
    out_dim: usize,
    /// Grid final (tokens por eje) — para pos-ids 2D del LLM
    grid_x: usize, // nx (tras merge)
    grid_y: usize, // ny

    pub fn deinit(self: *EncodedImage, allocator: std.mem.Allocator) void {
        allocator.free(self.embeddings);
        self.embeddings = &.{};
    }
};

pub const ClipEncoder = struct {
    allocator: std.mem.Allocator,
    engine: *matmul.MatmulEngine,
    cfg: *const mmproj_config.MmprojConfig,

    // Pesos del patch embed (dequantizados; layout GGUF [KW,KH,IC,OC]).
    // VIDEO (TODO 10.7): el ViT trae el Conv3D temporal split en 2 kernels
    // (unsloth qwenvl.py:111-117): w0 = t(0,·,·), w1 = t(1,·,·). Para still
    // ambos convolucionan la MISMA imagen (qwen2vl.cpp:12-16) — encode()
    // usa el pre-suma (conv0(x)+conv1(x)==conv(w0+w1)(x)). Para un PAR de
    // frames van separados: out = conv0(f0)+conv1(f1) (qwen2vl.cpp:19-26).
    patch_w: []f32, // w0+w1 (still path)
    patch_w0: []f32, // kernel temporal t=0 (video path)
    patch_w1: ?[]f32, // kernel temporal t=1 (null ⇒ mmproj sin split)
    patch_b: ?[]f32,
    /// Position embedding absoluto: [n_embd, pos_grid*pos_grid] (o null
    /// para Qwen2-VL, que no lo usa). Row-major GGUF: [ne0=n_embd, ne1=n_pos]
    pos_emb: ?[]f32,
    pos_emb_side: usize, // lado del grid original (p.ej. 48 → 48x48 pos)
    pre_ln_w: ?[]f32,
    pre_ln_b: ?[]f32,
    blocks: []ClipBlock,
    projector: proj.Qwen3VlProjector,

    n_embd: usize,
    n_head: usize,
    head_dim: usize,
    n_ff: usize,
    patch_size: usize,
    merge: usize,
    eps: f32,
    /// Qwen2.5-VL: RMS norm (propagado a blocks y pre_ln)
    use_rms_norm: bool = false,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        engine: *matmul.MatmulEngine,
        m: anytype, // *const MmprojModel
    ) !Self {
        const cfg = &m.config;
        switch (cfg.projector_type) {
            .qwen3vl_merger, .qwen25vl_merger, .qwen2vl_merger => {},
            .unknown => return EncoderError.UnsupportedProjector,
        }

        // ── Patch embed: v.patch_embd.weight [KW, KH, IC, OC] f32 lineal.
        // TEMPORAL MERGE (golden vs llama.cpp node_0..node_14): el grafo
        // qwen2vl/qwen3vl para still (n_batch=1) SUMA los DOS kernels
        // conv sobre la MISMA imagen (ggml_add(conv_2d(w0), conv_2d(w1)),
        // qwen2vl.cpp:12-16). Como ambos operan sobre el mismo input:
        // conv0(x)+conv1(x) == conv(w0+w1)(x) ⇒ pre-sumamos los pesos UNA
        // VEZ en init (coste runtime cero; antes usábamos sólo w0 ⇒ todo
        // embedding desplazado — hallado con mtmd-debug golden test).
        const pw = m.patchEmb0() orelse return EncoderError.MissingWeights;
        // VIDEO 10.7: w0/w1 SEPARADOS para el path de pares; patch_w (still)
        // sigue siendo el pre-suma.
        const patch_w0 = try attn_mod.dequantF32Slice(allocator, pw);
        errdefer allocator.free(patch_w0);
        var patch_w1: ?[]f32 = null;
        errdefer if (patch_w1) |w| allocator.free(w);
        var patch_w = try allocator.dupe(f32, patch_w0);
        errdefer allocator.free(patch_w);
        if (m.patchEmb1()) |pw1| {
            const w1 = try attn_mod.dequantF32Slice(allocator, pw1);
            defer allocator.free(w1);
            if (w1.len != patch_w.len) return EncoderError.ShapeMismatch;
            patch_w1 = try allocator.dupe(f32, w1);
            for (patch_w[0..], w1) |*a, b| a.* += b;
        }
        var patch_b: ?[]f32 = null;
        errdefer if (patch_b) |b| allocator.free(b);
        if (m.patchBias()) |w| patch_b = try attn_mod.dequantF32Slice(allocator, w);

        // ── Position embedding (Qwen3-VL; ausente en 2/2.5-VL)
        var pos_emb: ?[]f32 = null;
        errdefer if (pos_emb) |p| allocator.free(p);
        var pos_side: usize = 0;
        if (m.posEmb()) |w| {
            pos_emb = try attn_mod.dequantF32Slice(allocator, w);
            // GGUF [ne0, ne1] = [n_embd, n_pos] → side = sqrt(n_pos)
            const n_pos_emb = pos_emb.?.len / cfg.n_embd;
            pos_side = @intFromFloat(@sqrt(@as(f64, @floatFromInt(n_pos_emb))));
        }

        // ── pre_ln
        var pre_ln_w: ?[]f32 = null;
        errdefer if (pre_ln_w) |b| allocator.free(b);
        var pre_ln_b: ?[]f32 = null;
        errdefer if (pre_ln_b) |b| allocator.free(b);
        if (m.preLn("weight")) |w| {
            pre_ln_w = try attn_mod.dequantF32Slice(allocator, w);
            if (m.preLn("bias")) |bw| pre_ln_b = try attn_mod.dequantF32Slice(allocator, bw);
        }

        // ── Bloques
        const blocks = try allocator.alloc(ClipBlock, cfg.n_layer);
        errdefer allocator.free(blocks);
        var n_init: usize = 0;
        errdefer for (blocks[0..n_init]) |*b| b.deinit();
        for (0..cfg.n_layer) |il| {
            const is_ds = if (il < cfg.is_deepstack_layers.len)
                cfg.is_deepstack_layers[il]
            else
                false;
            blocks[il] = try ClipBlock.init(
                allocator,
                engine,
                m,
                il,
                cfg.n_embd,
                cfg.n_head,
                cfg.head_dim,
                cfg.n_head_kv,
                cfg.n_ff,
                cfg.ffn_op,
                is_ds,
                cfg.use_rms_norm, // Qwen2.5-VL: RMS; resto LayerNorm
                cfg.eps,
            );
            n_init += 1;
        }

        // ── Projector (merger)
        var n_ds_true: usize = 0;
        for (cfg.is_deepstack_layers) |b| {
            if (b) n_ds_true += 1;
        }
        const n_ds = n_ds_true; // sólo capas deepstack reales (clip.cpp:2024)
        const projector = try proj.Qwen3VlProjector.init(allocator, engine, m, cfg.n_embd, n_ds);
        errdefer projector.deinit();

        return .{
            .allocator = allocator,
            .engine = engine,
            .cfg = cfg,
            .patch_w = patch_w,
            .patch_w0 = patch_w0,
            .patch_w1 = patch_w1,
            .patch_b = patch_b,
            .pos_emb = pos_emb,
            .pos_emb_side = pos_side,
            .pre_ln_w = pre_ln_w,
            .pre_ln_b = pre_ln_b,
            .blocks = blocks,
            .projector = projector,
            .n_embd = cfg.n_embd,
            .n_head = cfg.n_head,
            .head_dim = cfg.head_dim,
            .n_ff = cfg.n_ff,
            .patch_size = cfg.patch_size,
            .merge = cfg.spatial_merge_size,
            .eps = cfg.eps,
            .use_rms_norm = cfg.use_rms_norm,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.patch_w);
        self.allocator.free(self.patch_w0);
        if (self.patch_w1) |w| self.allocator.free(w);
        if (self.patch_b) |b| self.allocator.free(b);
        if (self.pos_emb) |p| self.allocator.free(p);
        if (self.pre_ln_w) |w| self.allocator.free(w);
        if (self.pre_ln_b) |b| self.allocator.free(b);
        for (self.blocks) |*b| b.deinit();
        self.allocator.free(self.blocks);
        self.projector.deinit();
    }

    /// Reordena [OC, OH, OW] (salida del conv, layout "w,h,c" del grafo
    /// ggml → nuestro CHW) al layout pixel-shuffle del input ViT:
    /// cada merge-block 2×2 de patches queda como 4 tokens consecutivos
    /// con features contiguas.
    ///
    /// Port de qwen3vl.cpp:18-31 (permute [w,h,c]→[c,w,h] + cont_4d +
    /// reshape + permute). Resultado: [n_pos, n_embd] con n_pos=OH·OW
    /// en orden merge-block-major (fila de bloques × bloque × dy×dx).
    pub fn spatialMergePatches(
        conv_out: []const f32, // [OC, OH, OW] (CHW)
        oc: usize,
        oh: usize,
        ow: usize,
        merge: usize,
        out: []f32, // [n_pos, OC]
    ) void {
        const mw = ow / merge; // merge-blocks por fila
        var ptr: usize = 0;
        var by: usize = 0;
        while (by < oh) : (by += merge) {
            var bx: usize = 0;
            while (bx < ow) : (bx += merge) {
                // 4 patches del bloque (dy, dx) en orden (0,0),(0,1),(1,0),(1,1)
                var dy: usize = 0;
                while (dy < merge) : (dy += 1) {
                    var dx: usize = 0;
                    while (dx < merge) : (dx += 1) {
                        const py = by + dy;
                        const px = bx + dx;
                        const dst = out[ptr * oc ..][0..oc];
                        for (0..oc) |c| {
                            dst[c] = conv_out[(c * oh + py) * ow + px];
                        }
                        ptr += 1;
                    }
                }
            }
        }
        _ = mw;
    }

    /// Resize bilinear del position embedding aprendido al grid real.
    /// Port de resize_position_embeddings (GGML_SCALE_MODE_BILINEAR |
    /// ALIGN_CORNERS, qwen3vl.cpp:40) sobre [n_embd, side, side] →
    /// [n_embd, oh, ow].
    pub fn resizePosEmb(
        self: *const Self,
        src: []const f32, // [n_embd, side·side]
        side: usize,
        oh: usize,
        ow: usize,
        out: []f32, // [n_embd, oh·ow]
    ) void {
        const n_embd = self.n_embd;
        // ALIGN_CORNERS: ratio = (src-1)/(dst-1) — como preprocess.zig
        const yr: f32 = if (oh > 1)
            @as(f32, @floatFromInt(side - 1)) / @as(f32, @floatFromInt(oh - 1))
        else
            0;
        const xr: f32 = if (ow > 1)
            @as(f32, @floatFromInt(side - 1)) / @as(f32, @floatFromInt(ow - 1))
        else
            0;

        for (0..n_embd) |c| {
            const d_plane = out[c * oh * ow ..][0 .. oh * ow];
            for (0..oh) |y| {
                const py = @as(f32, @floatFromInt(y)) * yr;
                const y0 = @min(@as(usize, @intFromFloat(py)), side - 1);
                const y1 = @min(y0 + 1, side - 1);
                const yf = py - @as(f32, @floatFromInt(y0));
                for (0..ow) |x| {
                    const px = @as(f32, @floatFromInt(x)) * xr;
                    const x0 = @min(@as(usize, @intFromFloat(px)), side - 1);
                    const x1 = @min(x0 + 1, side - 1);
                    const xf = px - @as(f32, @floatFromInt(x0));

                    // GGUF pe {ne0=n_embd, ne1=side²}: ggml guarda ne0 (c) MÁS
                    // RÁPIDO ⇒ lineal i = c + t·n_embd (verificado vs golden
                    // node_23: antes indexábamos c·side²+t = slice de t's
                    // consecutivos — embeddings desalineados).
                    // t = a·side + b con (a,b)=(y,x) tras permute(2,0,1).
                    const pe = struct {
                        fn at(buf: []const f32, cc: usize, tt: usize, ne: usize) f32 {
                            return buf[cc + tt * ne];
                        }
                    };
                    const p00 = pe.at(src, c, y0 * side + x0, n_embd);
                    const p10 = pe.at(src, c, y0 * side + x1, n_embd);
                    const p01 = pe.at(src, c, y1 * side + x0, n_embd);
                    const p11 = pe.at(src, c, y1 * side + x1, n_embd);
                    const top = p00 + (p10 - p00) * xf;
                    const bottom = p01 + (p11 - p01) * xf;
                    d_plane[y * ow + x] = top + (bottom - top) * yf;
                }
            }
        }
    }

    /// Requisito de scratch para encode(n_pos = grid_x·grid_y):
    /// 4·n_pos·n_embd (conv_out/patches/pos_emb/ln + buf_a/buf_b) +
    /// max(block.scratchNeed) + margen del projector.
    pub fn scratchNeed(self: *const Self, n_pos: usize) usize {
        const n_embd = self.n_embd;
        const base = 5 * n_pos * n_embd;
        var blk_need: usize = 0;
        for (self.blocks) |*b| {
            blk_need = @max(blk_need, b.scratchNeed(n_pos));
        }
        // projector: ln_out [n_pos·n_embd] + merged [n_tok·in_dim] +
        // mid [n_tok·hidden] + tmp [n_tok·proj] — acotado por n_pos·(n_embd+in_dim+hidden+proj)
        const proj_need = n_pos * (n_embd + self.projector.in_dim + self.projector.hidden_dim + self.projector.out_dim);
        return base + blk_need + proj_need;
    }

    /// encode: imagen RGB → embeddings finales listos para el LLM.
    /// `rgb`: [H, W, 3] HWC u8.
    pub fn encode(
        self: *const Self,
        allocator: std.mem.Allocator,
        rgb: []const u8,
        width: usize,
        height: usize,
        scratch: []f32,
    ) !EncodedImage {
        const cfg = self.cfg;
        const t0 = timez.Timer.start();
        debugz.dbg.printLevel(.detail, "[mmproj] encode: {d}x{d} → ", .{ width, height });

        // ── 1. Preprocess
        const pp = preprocess_mod.preprocess(
            allocator,
            rgb,
            width,
            height,
            self.patch_size,
            self.merge,
            cfg.image_min_pixels,
            cfg.image_max_pixels,
            cfg.image_mean,
            cfg.image_std,
        ) catch return EncoderError.PreprocessFailed;
        defer allocator.free(pp.data);

        const grid_y = pp.grid_y;
        const grid_x = pp.grid_x;
        const n_pos = grid_y * grid_x;
        const n_embd = self.n_embd;

        debugz.dbg.printLevel(.detail, "[mmproj] res {d}x{d}, grid {d}x{d} ({d} pos)\n", .{ pp.width, pp.height, grid_x, grid_y, n_pos });

        if (debugz.dbg.perf_mm) {
            debugz.dbg.printLevel(.info, "[mmproj-perf] preprocess: {d} ms (grid {d}x{d})\n", .{ @divTrunc(t0.read(), std.time.ns_per_ms), grid_x, grid_y });
        }
        // ── 2. Conv2D patch embed (stride=patch, sin pad)
        const conv_out = scratch[0 .. n_pos * n_embd];
        conv2d.conv2dDirect(
            pp.data,
            3,
            pp.height,
            pp.width,
            self.patch_w,
            self.patch_size,
            self.patch_size,
            n_embd,
            self.patch_b,
            conv_out,
            self.patch_size,
        ) catch return EncoderError.ShapeMismatch;

        // ── 3. Pixel-shuffle: [OC, OH, OW] → [n_pos, n_embd] merge-order
        const patches = scratch[n_pos * n_embd .. 2 * n_pos * n_embd];
        spatialMergePatches(conv_out, n_embd, grid_y, grid_x, self.merge, patches);
        if (debugz.dbg.dump_mm_input) {
            debugz.dbg.printLevel(.info, "[mmproj-conv] tok0[3]={d:.4},{d:.4},{d:.4} last[3]={d:.4},{d:.4},{d:.4} sum={d:.2}\n", .{
                patches[0],                patches[1],          patches[2],
                patches[n_embd - 3],       patches[n_embd - 2], patches[n_embd - 1],
                debugz.sumAbsF32(patches),
            });
        }

        // ── 4. patch_bias ya sumado en conv; position_embd (Qwen3-VL):
        // aprendido [n_embd, side·side] → UPSCALE bilinear → pixel-shuffle.
        // NOTA (golden vs llama.cpp node_23..inp_pos_emb): el pos_embd
        // upscaled pasa por el MISMO pixel-shuffle que los patches (raster
        // → merge-block order) ANTES del ADD — sumarlo en orden raster
        // desalineaba los embeddings (hallado con mtmd-debug golden test).
        if (self.pos_emb) |pe| {
            const pe_resized = scratch[2 * n_pos * n_embd .. 3 * n_pos * n_embd];
            self.resizePosEmb(pe, self.pos_emb_side, grid_y, grid_x, pe_resized);
            // pixel-shuffle del pos_embd (CHW raster → [n_pos, n_embd] merge)
            const pe_shuffled = scratch[3 * n_pos * n_embd .. 4 * n_pos * n_embd];
            spatialMergePatches(pe_resized, n_embd, grid_y, grid_x, self.merge, pe_shuffled);
            if (debugz.dbg.dump_mm_input) {
                debugz.dbg.printLevel(.info, "[mmproj-pe-raw] src[0..3]={d:.4},{d:.4},{d:.4} src[11520..11522]={d:.4},{d:.4},{d:.4} up01={d:.4}\n", .{
                    pe[0],         pe[1],     pe[2],
                    pe[11520],     pe[11521], pe[11522],
                    pe_resized[1],
                });
            }
            if (debugz.dbg.dump_mm_input) {
                debugz.dbg.printLevel(.info, "[mmproj-pe] resized(c=0) y0x0..3={d:.4},{d:.4},{d:.4},{d:.4} y1x0={d:.4} | shuffled tok0[4]={d:.4},{d:.4},{d:.4},{d:.4}\n", .{
                    pe_resized[0],              pe_resized[grid_x], pe_resized[2 * grid_x], pe_resized[3 * grid_x],
                    pe_resized[grid_x * 1 + 0], pe_shuffled[0],     pe_shuffled[1],         pe_shuffled[2],
                    pe_shuffled[3],
                });
            }
            // inp += pos_emb (ambos en orden pixel-shuffle)
            for (0..n_pos) |t| {
                const row = patches[t * n_embd ..][0..n_embd];
                const pe_row = pe_shuffled[t * n_embd ..][0..n_embd];
                for (0..n_embd) |c| row[c] += pe_row[c];
            }
            if (debugz.dbg.dump_mm_input) {
                debugz.dbg.printLevel(.info, "[mmproj-inp] t0={d:.4},{d:.4},{d:.4} t1={d:.4} t4={d:.4} t12={d:.4},{d:.4},{d:.4} t13={d:.4} t15={d:.4},{d:.4},{d:.4}\n", .{
                    patches[0],               patches[1],               patches[2],
                    patches[n_embd],          patches[4 * n_embd],      patches[12 * n_embd],
                    patches[12 * n_embd + 1], patches[12 * n_embd + 2], patches[13 * n_embd],
                    patches[15 * n_embd],     patches[15 * n_embd + 1], patches[15 * n_embd + 2],
                });
            }
        } else if (debugz.dbg.dump_mm_input) {
            debugz.dbg.printLevel(.info, "[mmproj-inp] tok0[3]={d:.4},{d:.4},{d:.4} tok1[3]={d:.4},{d:.4},{d:.4} tok4[3]={d:.4},{d:.4},{d:.4}\n", .{
                patches[0],          patches[1],              patches[2],
                patches[n_embd],     patches[n_embd + 1],     patches[n_embd + 2],
                patches[4 * n_embd], patches[4 * n_embd + 1], patches[4 * n_embd + 2],
            });
        }

        // ── 5. pre_ln
        var cur: []f32 = patches;
        if (self.pre_ln_w) |w| {
            const ln_out = scratch[2 * n_pos * n_embd .. 3 * n_pos * n_embd]; // reuso (pos_emb ya sumado)
            if (self.use_rms_norm) {
                block_mod.rmsNormSlice(patches, ln_out, n_pos, n_embd, w, self.eps);
            } else {
                block_mod.layerNormSlice(patches, ln_out, n_pos, n_embd, w, self.pre_ln_b, self.eps);
            }
            cur = ln_out;
        }

        // ── 6. N× blocks
        const pos_ids = allocator.alloc([4]i32, n_pos) catch
            return EncoderError.OutOfMemory;
        defer allocator.free(pos_ids);
        mrope.visionPosIds(pos_ids, grid_y, grid_x, self.merge);

        // bucle doble-buffer: buf_a → block → buf_b; swap
        const buf_a = cur; // [n_pos, n_embd] en scratch
        const buf_b = scratch[3 * n_pos * n_embd .. 4 * n_pos * n_embd];
        var src = buf_a;
        var dst = buf_b;

        // deepstack features acumuladas (concat final)
        const n_ds = cfg.is_deepstack_layers.len;
        var ds_feats: []([]f32) = &.{};
        defer if (ds_feats.len > 0) {
            for (ds_feats) |f| allocator.free(f);
            allocator.free(ds_feats);
        };
        if (n_ds > 0) {
            ds_feats = allocator.alloc([]f32, n_ds) catch
                return EncoderError.OutOfMemory;
            @memset(ds_feats, &.{});
        }

        const ds_rows = n_pos / (self.merge * self.merge);
        // scratch de block: tras 4·n_pos·n_embd usamos el resto
        const block_scratch = scratch[5 * n_pos * n_embd ..];

        var ds_count: usize = 0;
        const t_blocks = timez.Timer.start();
        for (self.blocks) |*blk| {
            // deepstack: el bloque escribe sus features directamente
            if (blk.ds) |*d| {
                const f = allocator.alloc(f32, ds_rows * d.norm_w.len) catch
                    return EncoderError.OutOfMemory;
                ds_feats[ds_count] = f;
                ds_count += 1;
                try blk.forward(src, dst, n_pos, pos_ids, f, block_scratch);
            } else {
                try blk.forward(src, dst, n_pos, pos_ids, null, block_scratch);
            }
            const tmp = src;
            src = dst;
            dst = tmp;
        }
        if (debugz.dbg.perf_mm) {
            debugz.dbg.printLevel(.info, "[mmproj-perf] {d} blocks: {d} ms (n_pos={d})\n", .{ self.blocks.len, @divTrunc(t_blocks.read(), std.time.ns_per_ms), n_pos });
        }

        // ── 7. post_ln + merger
        const n_tokens = ds_rows;
        const out_dim = self.projector.out_dim;
        const embeddings = allocator.alloc(f32, n_tokens * out_dim) catch
            return EncoderError.OutOfMemory;
        errdefer allocator.free(embeddings);

        // ds_feats como []const []const
        var ds_const: []([]const f32) = &.{};
        defer if (ds_const.len > 0) allocator.free(ds_const);
        if (n_ds > 0) {
            ds_const = allocator.alloc([]const f32, n_ds) catch
                return EncoderError.OutOfMemory;
            for (ds_feats, 0..) |f, i| ds_const[i] = f;
        }

        self.projector.project(
            src, // última salida del encoder (ya swapped)
            ds_const,
            embeddings,
            n_pos,
            n_embd,
            scratch[5 * n_pos * n_embd ..],
        ) catch return EncoderError.ShapeMismatch;
        if (debugz.dbg.perf_mm) {
            debugz.dbg.printLevel(.info, "[mmproj-perf] merger: {d} ms\n", .{@divTrunc(t0.read() - t_blocks.read(), std.time.ns_per_ms)});
        }

        const t1 = t0.read();
        debugz.dbg.printLevel(.info, "[mmproj] encode ok: {d} tokens, dim {d} ({d} ms)\n", .{ n_tokens, out_dim, @divTrunc(t1, std.time.ns_per_ms) });

        return .{
            .embeddings = embeddings,
            .n_tokens = n_tokens,
            .out_dim = out_dim,
            .grid_x = pp.merge_grid_x,
            .grid_y = pp.merge_grid_y,
        };
    }

    /// VIDEO (TODO 10.7): encode de un PAR de frames (temporal merge t=2).
    /// Réplica de la rama n_batch==2 del oráculo (qwen2vl.cpp:19-26):
    /// out = conv0(f0) + conv1(f1) con los kernels temporales SEPARADOS
    /// (v.patch_embd.weight = t0, .weight.1 = t1). El par colapsa a UN
    /// grid espacial — el resto del grafo es idéntico al still.
    /// REQUIERE: mmproj CON split temporal (patch_w1 != null); frames del
    /// MISMO tamaño (el pipeline de video lo garantiza).
    pub fn encodePair(
        self: *const Self,
        allocator: std.mem.Allocator,
        rgb0: []const u8, // frame 0 HWC
        rgb1: []const u8, // frame 1 HWC
        width: usize,
        height: usize,
        scratch: []f32,
    ) !EncodedImage {
        const cfg = self.cfg;
        const w1 = self.patch_w1 orelse return EncoderError.MissingWeights;
        const t0 = timez.Timer.start();

        // Preprocess AMBAS frames (mismo target por tamaño idéntico)
        const pp0 = preprocess_mod.preprocess(
            allocator,
            rgb0,
            width,
            height,
            self.patch_size,
            self.merge,
            cfg.image_min_pixels,
            cfg.image_max_pixels,
            cfg.image_mean,
            cfg.image_std,
        ) catch return EncoderError.PreprocessFailed;
        defer allocator.free(pp0.data);
        const pp1 = preprocess_mod.preprocess(
            allocator,
            rgb1,
            width,
            height,
            self.patch_size,
            self.merge,
            cfg.image_min_pixels,
            cfg.image_max_pixels,
            cfg.image_mean,
            cfg.image_std,
        ) catch return EncoderError.PreprocessFailed;
        defer allocator.free(pp1.data);

        const grid_y = pp0.grid_y;
        const grid_x = pp0.grid_x;
        const n_pos = grid_y * grid_x;
        const n_embd = self.n_embd;
        if (pp1.grid_y != grid_y or pp1.grid_x != grid_x) return EncoderError.ShapeMismatch;

        debugz.dbg.printLevel(.info, "[mmproj] encodePair: {d}x{d} grid {d}x{d} ({d} pos, 2 frames)\n", .{ width, height, grid_x, grid_y, n_pos });
        if (debugz.dbg.perf_mm) {
            debugz.dbg.printLevel(.info, "[mmproj-perf] preprocess par: {d} ms (grid {d}x{d})\n", .{ @divTrunc(t0.read(), std.time.ns_per_ms), grid_x, grid_y });
        }

        // Conv temporal: conv0(f0) → conv_tmp, conv1(f1) → conv_out, suma.
        // conv_out [n_embd, grid_y, grid_x] CHW — mismo layout que still.
        const conv_out = scratch[0 .. n_pos * n_embd];
        const conv_tmp = scratch[n_pos * n_embd .. 2 * n_pos * n_embd];
        conv2d.conv2dDirect(
            pp0.data,
            3,
            pp0.height,
            pp0.width,
            self.patch_w0,
            self.patch_size,
            self.patch_size,
            n_embd,
            self.patch_b,
            conv_out,
            self.patch_size,
        ) catch return EncoderError.ShapeMismatch;
        conv2d.conv2dDirect(
            pp1.data,
            3,
            pp1.height,
            pp1.width,
            w1,
            self.patch_size,
            self.patch_size,
            n_embd,
            null, // bias UNA vez (en la conv t=0): la identidad
            //  par(f,f)==still(f) exige bias una sola — igual que el grafo
            //  del oráculo (el bias vive en el conv params del loader,
            //  qwen2vl.cpp:14-16 lo suma en el add de convs con bias común).
            conv_tmp,
            self.patch_size,
        ) catch return EncoderError.ShapeMismatch;
        for (conv_out, conv_tmp) |*a, b| a.* += b;

        debugz.dbg.printLevel(.detail, "[mmproj] temporal merge: sum|v|={d:.2}\n", .{debugz.sumAbsF32(conv_out)});

        // ── cuerpo idéntico al still desde el pixel-shuffle
        const patches = scratch[2 * n_pos * n_embd .. 3 * n_pos * n_embd];
        spatialMergePatches(conv_out, n_embd, grid_y, grid_x, self.merge, patches);
        if (self.pos_emb) |pe| {
            const pe_resized = scratch[3 * n_pos * n_embd .. 4 * n_pos * n_embd];
            self.resizePosEmb(pe, self.pos_emb_side, grid_y, grid_x, pe_resized);
            const pe_shuffled = scratch[4 * n_pos * n_embd .. 5 * n_pos * n_embd];
            spatialMergePatches(pe_resized, n_embd, grid_y, grid_x, self.merge, pe_shuffled);
            for (0..n_pos) |t| {
                const row = patches[t * n_embd ..][0..n_embd];
                const pe_row = pe_shuffled[t * n_embd ..][0..n_embd];
                for (0..n_embd) |c| row[c] += pe_row[c];
            }
        }
        var cur: []f32 = patches;
        if (self.pre_ln_w) |w| {
            const ln_out = scratch[5 * n_pos * n_embd .. 6 * n_pos * n_embd];
            if (self.use_rms_norm) {
                block_mod.rmsNormSlice(patches, ln_out, n_pos, n_embd, w, self.eps);
            } else {
                block_mod.layerNormSlice(patches, ln_out, n_pos, n_embd, w, self.pre_ln_b, self.eps);
            }
            cur = ln_out;
        }

        const pos_ids = allocator.alloc([4]i32, n_pos) catch return EncoderError.OutOfMemory;
        defer allocator.free(pos_ids);
        mrope.visionPosIds(pos_ids, grid_y, grid_x, self.merge);

        const buf_a = cur;
        const buf_b = scratch[6 * n_pos * n_embd .. 7 * n_pos * n_embd];
        var src = buf_a;
        var dst = buf_b;

        const ds_rows = n_pos / (self.merge * self.merge);
        const block_scratch = scratch[7 * n_pos * n_embd ..];

        const t_blocks = timez.Timer.start();
        for (self.blocks) |*blk| {
            try blk.forward(src, dst, n_pos, pos_ids, null, block_scratch);
            const tmp = src;
            src = dst;
            dst = tmp;
        }
        if (debugz.dbg.perf_mm) {
            debugz.dbg.printLevel(.info, "[mmproj-perf] {d} blocks (par): {d} ms (n_pos={d})\n", .{ self.blocks.len, @divTrunc(t_blocks.read(), std.time.ns_per_ms), n_pos });
        }

        const n_tokens = ds_rows;
        const out_dim = self.projector.out_dim;
        const embeddings = allocator.alloc(f32, n_tokens * out_dim) catch
            return EncoderError.OutOfMemory;
        errdefer allocator.free(embeddings);
        self.projector.project(
            src,
            &.{}, // sin deepstack en el path par (igual que GPU path)
            embeddings,
            n_pos,
            n_embd,
            scratch[7 * n_pos * n_embd ..],
        ) catch return EncoderError.ShapeMismatch;

        const t1 = t0.read();
        debugz.dbg.printLevel(.info, "[mmproj] encodePair ok: {d} tokens, dim {d} ({d} ms)\n", .{ n_tokens, out_dim, @divTrunc(t1, std.time.ns_per_ms) });

        return .{
            .embeddings = embeddings,
            .n_tokens = n_tokens,
            .out_dim = out_dim,
            .grid_x = pp0.merge_grid_x,
            .grid_y = pp0.merge_grid_y,
        };
    }
    /// Encode 10.2 device-resident: pre-bloques en CPU (conv/shuffle/pe/
    /// pre_ln ~10ms), los N blocks en GPU (GpuClipEncoder), merger en CPU.
    /// `gpu` se construye UNA vez por encoder+resolución (pesos residentes).
    /// REQUIERE: mmproj sin deepstack (fallback a encode() CPU si hay).
    pub fn encodeGPU(
        self: *const Self,
        allocator: std.mem.Allocator,
        rgb: []const u8,
        width: usize,
        height: usize,
        gpu: *clip_gpu.GpuClipEncoder,
        scratch: []f32,
    ) !EncodedImage {
        const cfg = self.cfg;
        const has_ds = blk: {
            for (cfg.is_deepstack_layers) |d| {
                if (d) break :blk true;
            }
            break :blk false;
        };
        const t0 = timez.Timer.start();

        // ── 1. Preprocess + conv + shuffle + pe + pre_ln (host, barato)
        const pp = preprocess_mod.preprocess(
            allocator,
            rgb,
            width,
            height,
            self.patch_size,
            self.merge,
            cfg.image_min_pixels,
            cfg.image_max_pixels,
            cfg.image_mean,
            cfg.image_std,
        ) catch return EncoderError.PreprocessFailed;
        defer allocator.free(pp.data);

        const grid_y = pp.grid_y;
        const grid_x = pp.grid_x;
        const n_pos = grid_y * grid_x;
        const n_embd = self.n_embd;

        const conv_out = scratch[0 .. n_pos * n_embd];
        conv2d.conv2dDirect(
            pp.data,
            3,
            pp.height,
            pp.width,
            self.patch_w,
            self.patch_size,
            self.patch_size,
            n_embd,
            self.patch_b,
            conv_out,
            self.patch_size,
        ) catch return EncoderError.ShapeMismatch;

        const patches = scratch[n_pos * n_embd .. 2 * n_pos * n_embd];
        spatialMergePatches(conv_out, n_embd, grid_y, grid_x, self.merge, patches);

        if (self.pos_emb) |pe| {
            const pe_resized = scratch[2 * n_pos * n_embd .. 3 * n_pos * n_embd];
            self.resizePosEmb(pe, self.pos_emb_side, grid_y, grid_x, pe_resized);
            const pe_shuffled = scratch[3 * n_pos * n_embd .. 4 * n_pos * n_embd];
            spatialMergePatches(pe_resized, n_embd, grid_y, grid_x, self.merge, pe_shuffled);
            for (0..n_pos) |t| {
                const row = patches[t * n_embd ..][0..n_embd];
                const pe_row = pe_shuffled[t * n_embd ..][0..n_embd];
                for (0..n_embd) |c| row[c] += pe_row[c];
            }
        }

        var cur: []f32 = patches;
        if (self.pre_ln_w) |w| {
            const ln_out = scratch[4 * n_pos * n_embd .. 5 * n_pos * n_embd];
            if (self.use_rms_norm) {
                block_mod.rmsNormSlice(patches, ln_out, n_pos, n_embd, w, self.eps);
            } else {
                block_mod.layerNormSlice(patches, ln_out, n_pos, n_embd, w, self.pre_ln_b, self.eps);
            }
            cur = ln_out;
        }

        // ── 2. N blocks en GPU (device-resident)
        const pos_ids = allocator.alloc([4]i32, n_pos) catch return EncoderError.OutOfMemory;
        defer allocator.free(pos_ids);
        mrope.visionPosIds(pos_ids, grid_y, grid_x, self.merge);

        const t_blocks = timez.Timer.start();
        try gpu.setPosIds(pos_ids);
        try gpu.uploadInput(cur);
        try gpu.runBlocks();
        // sync implícito: downloadOutput ya es copia síncrona del d_x.
        // (cuMemcpyDtoH sincroniza el stream de los kernels previos).
        try gpu.downloadOutput(cur); // resultado vuelve al buffer host `cur`
        if (has_ds and gpu.has_deepstack) {
            const ds_rows = n_pos / (self.merge * self.merge);
            const ds_dim = gpu.ds[0].ds_out_dim;
            const ds_buf = scratch[5 * n_pos * n_embd .. 5 * n_pos * n_embd + ds_rows * ds_dim];
            try gpu.downloadDsOutput(ds_buf);
        }
        if (debugz.dbg.perf_mm) {
            debugz.dbg.printLevel(.info, "[mmproj-perf] {d} blocks GPU: {d} ms (n_pos={d})\n", .{ self.blocks.len, @divTrunc(t_blocks.read(), std.time.ns_per_ms), n_pos });
        }

        // ── 3. post_ln + merger (host)
        const n_tokens = n_pos / (self.merge * self.merge);
        const out_dim = self.projector.out_dim;
        const embeddings = allocator.alloc(f32, n_tokens * out_dim) catch
            return EncoderError.OutOfMemory;
        errdefer allocator.free(embeddings);

        const ds_feats = if (has_ds and gpu.has_deepstack) blk: {
            const ds_rows = n_pos / (self.merge * self.merge);
            const ds_dim = gpu.ds[0].ds_out_dim;
            const ds_buf = scratch[5 * n_pos * n_embd .. 5 * n_pos * n_embd + ds_rows * ds_dim];
            const f = allocator.alloc(f32, ds_rows * ds_dim) catch return EncoderError.OutOfMemory;
            @memcpy(f, ds_buf);
            const s = allocator.alloc([]f32, 1) catch return EncoderError.OutOfMemory;
            s[0] = f;
            break :blk s;
        } else &.{};
        defer if (ds_feats.len > 0) {
            allocator.free(ds_feats[0]);
            allocator.free(ds_feats);
        };

        self.projector.project(
            cur,
            ds_feats,
            embeddings,
            n_pos,
            n_embd,
            scratch[5 * n_pos * n_embd ..],
        ) catch return EncoderError.ShapeMismatch;

        const t1 = t0.read();
        debugz.dbg.printLevel(.info, "[mmproj] encodeGPU ok: {d} tokens, dim {d} ({d} ms)\n", .{ n_tokens, out_dim, @divTrunc(t1, std.time.ns_per_ms) });

        return .{
            .embeddings = embeddings,
            .n_tokens = n_tokens,
            .out_dim = out_dim,
            .grid_x = pp.merge_grid_x,
            .grid_y = pp.merge_grid_y,
        };
    }

    /// VIDEO (TODO 10.7): encodePair con blocks GPU — pre-bloques CPU (conv
    /// temporal doble), N blocks device-resident, merger CPU. Misma
    /// amortización de pesos que encodeGPU (GpuClipEncoder persistente).
    pub fn encodePairGPU(
        self: *const Self,
        allocator: std.mem.Allocator,
        rgb0: []const u8,
        rgb1: []const u8,
        width: usize,
        height: usize,
        gpu: *clip_gpu.GpuClipEncoder,
        scratch: []f32,
    ) !EncodedImage {
        const cfg = self.cfg;
        const has_ds = blk: {
            for (cfg.is_deepstack_layers) |d| {
                if (d) break :blk true;
            }
            break :blk false;
        };
        const w1 = self.patch_w1 orelse return EncoderError.MissingWeights;
        const t0 = timez.Timer.start();

        const pp0 = preprocess_mod.preprocess(
            allocator,
            rgb0,
            width,
            height,
            self.patch_size,
            self.merge,
            cfg.image_min_pixels,
            cfg.image_max_pixels,
            cfg.image_mean,
            cfg.image_std,
        ) catch return EncoderError.PreprocessFailed;
        defer allocator.free(pp0.data);
        const pp1 = preprocess_mod.preprocess(
            allocator,
            rgb1,
            width,
            height,
            self.patch_size,
            self.merge,
            cfg.image_min_pixels,
            cfg.image_max_pixels,
            cfg.image_mean,
            cfg.image_std,
        ) catch return EncoderError.PreprocessFailed;
        defer allocator.free(pp1.data);

        const grid_y = pp0.grid_y;
        const grid_x = pp0.grid_x;
        const n_pos = grid_y * grid_x;
        const n_embd = self.n_embd;
        if (pp1.grid_y != grid_y or pp1.grid_x != grid_x) return EncoderError.ShapeMismatch;

        // Conv temporal: conv0(f0)+conv1(f1), bias una vez
        const conv_out = scratch[0 .. n_pos * n_embd];
        const conv_tmp = scratch[n_pos * n_embd .. 2 * n_pos * n_embd];
        conv2d.conv2dDirect(pp0.data, 3, pp0.height, pp0.width, self.patch_w0, self.patch_size, self.patch_size, n_embd, self.patch_b, conv_out, self.patch_size) catch return EncoderError.ShapeMismatch;
        conv2d.conv2dDirect(pp1.data, 3, pp1.height, pp1.width, w1, self.patch_size, self.patch_size, n_embd, null, conv_tmp, self.patch_size) catch return EncoderError.ShapeMismatch;
        for (conv_out, conv_tmp) |*a, b| a.* += b;

        const patches = scratch[2 * n_pos * n_embd .. 3 * n_pos * n_embd];
        spatialMergePatches(conv_out, n_embd, grid_y, grid_x, self.merge, patches);

        if (self.pos_emb) |pe| {
            const pe_resized = scratch[3 * n_pos * n_embd .. 4 * n_pos * n_embd];
            self.resizePosEmb(pe, self.pos_emb_side, grid_y, grid_x, pe_resized);
            const pe_shuffled = scratch[4 * n_pos * n_embd .. 5 * n_pos * n_embd];
            spatialMergePatches(pe_resized, n_embd, grid_y, grid_x, self.merge, pe_shuffled);
            for (0..n_pos) |t| {
                const row = patches[t * n_embd ..][0..n_embd];
                const pe_row = pe_shuffled[t * n_embd ..][0..n_embd];
                for (0..n_embd) |c| row[c] += pe_row[c];
            }
        }

        var cur: []f32 = patches;
        if (self.pre_ln_w) |w| {
            const ln_out = scratch[5 * n_pos * n_embd .. 6 * n_pos * n_embd];
            if (self.use_rms_norm) {
                block_mod.rmsNormSlice(patches, ln_out, n_pos, n_embd, w, self.eps);
            } else {
                block_mod.layerNormSlice(patches, ln_out, n_pos, n_embd, w, self.pre_ln_b, self.eps);
            }
            cur = ln_out;
        }

        const pos_ids = allocator.alloc([4]i32, n_pos) catch
            return EncoderError.OutOfMemory;
        defer allocator.free(pos_ids);
        mrope.visionPosIds(pos_ids, grid_y, grid_x, self.merge);

        const t_blocks = timez.Timer.start();
        try gpu.setPosIds(pos_ids);
        try gpu.uploadInput(cur);
        try gpu.runBlocks();
        try gpu.downloadOutput(cur);
        if (has_ds and gpu.has_deepstack) {
            const ds_rows = n_pos / (self.merge * self.merge);
            const ds_dim = gpu.ds[0].ds_out_dim;
            const ds_buf = scratch[6 * n_pos * n_embd .. 6 * n_pos * n_embd + ds_rows * ds_dim];
            try gpu.downloadDsOutput(ds_buf);
        }
        if (debugz.dbg.perf_mm) {
            debugz.dbg.printLevel(.info, "[mmproj-perf] {d} blocks GPU (par): {d} ms (n_pos={d})\n", .{ self.blocks.len, @divTrunc(t_blocks.read(), std.time.ns_per_ms), n_pos });
        }

        const n_tokens = n_pos / (self.merge * self.merge);
        const out_dim = self.projector.out_dim;
        const embeddings = allocator.alloc(f32, n_tokens * out_dim) catch
            return EncoderError.OutOfMemory;
        errdefer allocator.free(embeddings);

        const ds_feats = if (has_ds and gpu.has_deepstack) blk: {
            const ds_rows = n_pos / (self.merge * self.merge);
            const ds_dim = gpu.ds[0].ds_out_dim;
            const ds_buf = scratch[6 * n_pos * n_embd .. 6 * n_pos * n_embd + ds_rows * ds_dim];
            const f = allocator.alloc(f32, ds_rows * ds_dim) catch return EncoderError.OutOfMemory;
            @memcpy(f, ds_buf);
            const s = allocator.alloc([]f32, 1) catch return EncoderError.OutOfMemory;
            s[0] = f;
            break :blk s;
        } else &.{};
        defer if (ds_feats.len > 0) {
            allocator.free(ds_feats[0]);
            allocator.free(ds_feats);
        };

        self.projector.project(
            cur,
            ds_feats,
            embeddings,
            n_pos,
            n_embd,
            scratch[6 * n_pos * n_embd ..],
        ) catch return EncoderError.ShapeMismatch;

        const t1 = t0.read();
        debugz.dbg.printLevel(.info, "[mmproj] encodePairGPU ok: {d} tokens, dim {d} ({d} ms)\n", .{ n_tokens, out_dim, @divTrunc(t1, std.time.ns_per_ms) });

        return .{
            .embeddings = embeddings,
            .n_tokens = n_tokens,
            .out_dim = out_dim,
            .grid_x = pp0.merge_grid_x,
            .grid_y = pp0.merge_grid_y,
        };
    }
};

fn countDsBefore(cfg: *const mmproj_config.MmprojConfig, il: usize) usize {
    var n: usize = 0;
    for (cfg.is_deepstack_layers[0..@min(il, cfg.is_deepstack_layers.len)]) |b| {
        if (b) n += 1;
    }
    return n;
}
