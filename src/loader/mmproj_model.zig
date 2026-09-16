//! MmprojModel — envoltura de alto nivel sobre un archivo mmproj GGUF
//! (encoder vision CLIP ViT + projector). Patrón dual-load idéntico al
//! sidecar draft (`gguf_model.zig:175`): el target y el mmproj son dos
//! GgufFile independientes mapeados en paralelo.
//!
//! Nombres de tensores (llama.cpp clip-impl.h:98-149):
//!   v.patch_embd.weight       — Conv2D kernel 0 (split temporal t=0)
//!   v.patch_embd.weight.1     — Conv2D kernel 1 (split temporal t=1)
//!   v.patch_embd.bias         — bias del patch embed (Qwen3-VL lo trae)
//!   v.position_embd.weight    — positional embedding absoluto (Qwen3-VL;
//!                               se resize bilinear a la resolución real)
//!   v.pre_ln.{weight,bias}    — pre-layernorm
//!   v.post_ln.{weight,bias}   — post-layernorm (merger norm de unsloth
//!                               visual.merger.norm → v.post_ln)
//!   v.blk.{i}.ln1/attn_qkv/attn_out/ln2/ffn_up/ffn_gate/ffn_down
//!   mm.0.{weight,bias}        — merger linear_fc1 (unsloth qwen3vl.py:107)
//!   mm.2.{weight,bias}        — merger linear_fc2 (⚠ NO mm.1; clip.cpp:2201)
//!   v.deepstack.{i}.norm/fc1/fc2.{weight,bias}
const std = @import("std");
const gguf = @import("gguf");
const mmproj_config = @import("mmproj_config");
const QuantWeight = @import("quant_weight").QuantWeight;
const Tensor = @import("core").Tensor;
const debugz = @import("debug");

pub const MmprojModelError = error{
    MissingTensor,
    NotAVisionMmproj,
};

pub const MmprojModel = struct {
    allocator: std.mem.Allocator,
    file: gguf.GgufFile,
    config: mmproj_config.MmprojConfig,

    const Self = @This();

    /// Carga el mmproj (mmap) y deriva MmprojConfig de las claves clip.*.
    pub fn load(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !Self {
        var file = try gguf.GgufFile.fromFileMmap(io, allocator, path);
        errdefer file.deinit();
        const config = try mmproj_config.MmprojConfig.fromGguf(&file);
        if (config.projector_type == .unknown) return MmprojModelError.NotAVisionMmproj;
        debugz.dbg.printLevel(.info, "[mmproj] cargado: type={s} n_embd={d} n_layer={d} head={d}x{d} patch={d} merge={d}\n", .{
            config.projector_type_str, config.n_embd,   config.n_layer,
            config.n_head,             config.head_dim, config.patch_size,
            config.spatial_merge_size,
        });
        return .{ .allocator = allocator, .file = file, .config = config };
    }

    pub fn deinit(self: *Self) void {
        self.config.deinit(self.allocator);
        self.file.deinit();
    }

    // ── Lookup de tensores (nombres clip-impl.h:98-149) ─────────────────────

    fn qw(self: *const Self, name: []const u8) ?QuantWeight {
        const info = self.file.getTensor(name) orelse return null;
        return QuantWeight.init(info, self.file.tensorData(info));
    }

    /// v.patch_embd.weight — kernel Conv2D 0 [hidden, 3, k, k] (t=0).
    pub fn patchEmb0(self: *const Self) ?QuantWeight {
        return self.qw("v.patch_embd.weight");
    }

    /// v.patch_embd.weight.1 — kernel Conv2D 1 (t=1, video; unused en still).
    pub fn patchEmb1(self: *const Self) ?QuantWeight {
        return self.qw("v.patch_embd.weight.1");
    }

    /// v.patch_embd.bias — [hidden] (Qwen3-VL lo incluye).
    pub fn patchBias(self: *const Self) ?QuantWeight {
        return self.qw("v.patch_embd.bias");
    }

    /// v.position_embd.weight — absolute pos-emb del ViT (Qwen3-VL).
    pub fn posEmb(self: *const Self) ?QuantWeight {
        return self.qw("v.position_embd.weight");
    }

    pub fn preLn(self: *const Self, wb: []const u8) ?QuantWeight {
        var buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&buf, "v.pre_ln.{s}", .{wb}) catch return null;
        return self.qw(name);
    }

    pub fn postLn(self: *const Self, wb: []const u8) ?QuantWeight {
        var buf: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&buf, "v.post_ln.{s}", .{wb}) catch return null;
        return self.qw(name);
    }

    /// v.blk.{i}.{name} — pesos por capa (attn_qkv, ln1, ffn_up, ...).
    pub fn blk(self: *const Self, il: usize, name: []const u8) ?QuantWeight {
        var buf: [96]u8 = undefined;
        const full = std.fmt.bufPrint(&buf, "v.blk.{d}.{s}", .{ il, name }) catch return null;
        return self.qw(full);
    }

    /// mm.{idx}.{weight,bias} — merger MLP. ⚠ fc1=mm.0, fc2=mm.2 (clip.cpp:2201).
    pub fn mm(self: *const Self, idx: usize, wb: []const u8) ?QuantWeight {
        var buf: [48]u8 = undefined;
        const name = std.fmt.bufPrint(&buf, "mm.{d}.{s}", .{ idx, wb }) catch return null;
        return self.qw(name);
    }

    /// v.deepstack.{i}.{norm|fc1|fc2}.{weight,bias}
    pub fn deepstack(self: *const Self, il: usize, name: []const u8) ?QuantWeight {
        var buf: [96]u8 = undefined;
        const full = std.fmt.bufPrint(&buf, "v.deepstack.{d}.{s}", .{ il, name }) catch return null;
        return self.qw(full);
    }

    // ── Helpers de dequant (patrón gguf_model.zig dequantToF16) ─────────────

    /// Dequantiza un QuantWeight a Tensor(f32) con la forma dada.
    pub fn dequantToF32(
        self: *const Self,
        w: QuantWeight,
        shape: []const usize,
    ) !Tensor(f32) {
        const n: usize = @intCast(w.numel());
        var t = try Tensor(f32).alloc(self.allocator, shape);
        errdefer t.deinit();
        const info = w.info;
        switch (info.dtype) {
            .f32 => {
                @memcpy(t.data, std.mem.bytesAsSlice(f32, w.bytes[0 .. n * 4]));
            },
            .f16 => {
                for (0..n) |i| {
                    const h = std.mem.readInt(u16, w.bytes[i * 2 ..][0..2], .little);
                    t.data[i] = @floatCast(@as(f16, @bitCast(h)));
                }
            },
            .bf16 => {
                for (0..n) |i| {
                    const b = std.mem.readInt(u16, w.bytes[i * 2 ..][0..2], .little);
                    const bits: u32 = @as(u32, b) << 16;
                    t.data[i] = @bitCast(bits);
                }
            },
            else => {
                // Cuantizado: dequant bloque a bloque vía dequantToF16+f32.
                const f16buf = try self.allocator.alloc(f16, n);
                defer self.allocator.free(f16buf);
                w.dequantToF16(f16buf);
                for (0..n) |i| t.data[i] = @floatCast(f16buf[i]);
            },
        }
        return t;
    }

    /// Dequantiza un QuantWeight 2D [in, out] GGUF → Tensor(f32) [out, in]
    /// (transpuesto, orientación linearProjection trans_b=true).
    /// Reutiliza dequantToF16Transposed + conversión a f32.
    pub fn dequantToF32Transposed(
        self: *const Self,
        w: QuantWeight,
        out_rows: usize,
        in_cols: usize,
    ) !Tensor(f32) {
        const n: usize = @intCast(w.numel());
        var t = try Tensor(f32).alloc(self.allocator, &.{ out_rows, in_cols });
        errdefer t.deinit();
        const info = w.info;
        switch (info.dtype) {
            // GGUF ggml layout {ne0=in, ne1=out}: ne0 MÁS RÁPIDO ⇒
            // lineal s = i + o·ne0 (i=in, o=out). W_T[o][i] = src[i + o·in_cols].
            // (ANTES: src[c·out_rows + r] leía [in][out] row-major — desalineado,
            // hallado con golden test mtmd-debug: QKV proj divergía con input
            // idéntico.)
            .f32 => {
                const src = std.mem.bytesAsSlice(f32, w.bytes[0 .. n * 4]);
                for (0..out_rows) |r| {
                    for (0..in_cols) |c| {
                        t.data[r * in_cols + c] = src[c + r * in_cols];
                    }
                }
            },
            .bf16 => {
                for (0..out_rows) |r| {
                    for (0..in_cols) |c| {
                        const s = c + r * in_cols;
                        const b = std.mem.readInt(u16, w.bytes[s * 2 ..][0..2], .little);
                        t.data[r * in_cols + c] = @bitCast(@as(u32, b) << 16);
                    }
                }
            },
            .f16 => {
                for (0..out_rows) |r| {
                    for (0..in_cols) |c| {
                        const s = c + r * in_cols;
                        const h = std.mem.readInt(u16, w.bytes[s * 2 ..][0..2], .little);
                        t.data[r * in_cols + c] = @floatCast(@as(f16, @bitCast(h)));
                    }
                }
            },
            else => {
                const f16buf = try self.allocator.alloc(f16, n);
                defer self.allocator.free(f16buf);
                w.dequantToF16Transposed(f16buf);
                for (0..n) |i| t.data[i] = @floatCast(f16buf[i]);
            },
        }
        return t;
    }
};
