//! KT-B (lane-f): runtime opt-in KV-transfer — prefila en un modelo SOURCE
//! y transfiere su KV-cache al TARGET mediante mappers lineales por capa.
//!
//! Pipeline por capa (fila KT-B, TODO.md):
//!   1. retrieveForAttention del cache SOURCE (f16 dequantizado, por kv-head)
//!   2. strip RoPE source: rotación INVERSA con los ángulos del source
//!      (applyRoPEInverseOnSlice, rope.zig) — el cache legacy guarda K
//!      post-RoPE
//!   3. mapper por capa: identity (copia) o dense (y = x·W^T + b sobre el
//!      bloque [kv_heads·head_dim], gemm f32 del MatmulEngine: cuBLAS o CPU)
//!   4. aplicar RoPE target (rotación forward)
//!   5. appendTokensF16 al cache TARGET (cuantiza al -ctk/-ctv del target)
//!
//! Formato de pesos .ktb (little-endian):
//!   magic "ZKTB" u32 · version u32=1 · n_layers u32 · n_kv_heads_source u32
//!   · n_kv_heads_target u32 · head_dim u32  → header 24 bytes
//!   por capa: kind u32 (0=identity, 1=dense)
//!     dense: W_k f32 [out,in] · b_k f32 [out] · W_v f32 [out,in] · b_v [out]
//!     (out = n_kv_t·hd, in = n_kv_s·hd; hd compartido — validado on-load)
//!
//! El modo source==target (secuencia dual en el mismo manager, patrón MTP
//! draft de cli.zig:2752) es el gate-0: identity roundtrip debe reproducir
//! el prefill normal. Cross-model dual-load sigue el patrón SidecarDraft.
//!
//! Breadcrumbs: tag [kt_transfer], gated DEBUG_LEVEL.

const std = @import("std");
const Tensor = @import("core").Tensor;
const kvcache = @import("kv_cache_manager.zig");
const rope_mod = @import("rope");
const matmul = @import("matmul");
const debugz = @import("debug");

/// Re-export para callers del runtime (main.zig no importa rope directo).
pub const RopePairing = rope_mod.RopePairing;

pub const KtError = error{
    BadMagic,
    BadVersion,
    BadHeader,
    TruncatedFile,
    HeadDimMismatch,
    LayerCountMismatch,
    SequenceEmpty,
};

pub const KT_MAGIC: u32 = 0x42544B5A; // "ZKTB" LE
pub const KT_VERSION: u32 = 1;

pub const LayerMapperKind = enum(u32) { identity = 0, dense = 1 };

pub const DenseMapper = struct {
    /// Row-major [out_dim, in_dim]; out = n_kv_t·hd, in = n_kv_s·hd.
    w_k: []f32,
    b_k: []f32,
    w_v: []f32,
    b_v: []f32,
};

pub const LayerMapper = union(LayerMapperKind) {
    identity: void,
    dense: DenseMapper,
};

pub const KtWeights = struct {
    allocator: std.mem.Allocator,
    n_layers: u32,
    n_kv_heads_source: u32,
    n_kv_heads_target: u32,
    head_dim: u32,
    layers: []LayerMapper,

    const Self = @This();

    pub fn deinit(self: *Self) void {
        for (self.layers) |*m| switch (m.*) {
            .dense => |*d| {
                self.allocator.free(d.w_k);
                self.allocator.free(d.b_k);
                self.allocator.free(d.w_v);
                self.allocator.free(d.b_v);
            },
            .identity => {},
        };
        self.allocator.free(self.layers);
        self.layers = &[_]LayerMapper{};
    }

    pub fn load(
        allocator: std.mem.Allocator,
        io: std.Io,
        path: []const u8,
        n_layers_expect: u32,
        n_kv_source_expect: u32,
        n_kv_target_expect: u32,
        head_dim_expect: u32,
    ) !Self {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
        defer allocator.free(bytes);

        if (bytes.len < 24) return KtError.TruncatedFile;
        if (std.mem.readInt(u32, bytes[0..4], .little) != KT_MAGIC) return KtError.BadMagic;
        if (std.mem.readInt(u32, bytes[4..8], .little) != KT_VERSION) return KtError.BadVersion;
        const n_layers = std.mem.readInt(u32, bytes[8..12], .little);
        const n_kv_s = std.mem.readInt(u32, bytes[12..16], .little);
        const n_kv_t = std.mem.readInt(u32, bytes[16..20], .little);
        const head_dim = std.mem.readInt(u32, bytes[20..24], .little);

        if (n_layers != n_layers_expect) return KtError.LayerCountMismatch;
        if (n_kv_s != n_kv_source_expect or n_kv_t != n_kv_target_expect) return KtError.BadHeader;
        if (head_dim != head_dim_expect) return KtError.HeadDimMismatch;

        var self = Self{
            .allocator = allocator,
            .n_layers = n_layers,
            .n_kv_heads_source = n_kv_s,
            .n_kv_heads_target = n_kv_t,
            .head_dim = head_dim,
            .layers = try allocator.alloc(LayerMapper, n_layers),
        };
        errdefer self.deinit();

        var off: usize = 24;
        const in_dim = @as(usize, n_kv_s) * head_dim;
        const out_dim = @as(usize, n_kv_t) * head_dim;
        const w_len = in_dim * out_dim;
        for (self.layers) |*m| {
            if (off + 4 > bytes.len) return KtError.TruncatedFile;
            const kind_raw = std.mem.readInt(u32, bytes[off..][0..4], .little);
            off += 4;
            if (kind_raw == @intFromEnum(LayerMapperKind.identity)) {
                m.* = .{ .identity = {} };
                continue;
            }
            if (kind_raw != @intFromEnum(LayerMapperKind.dense)) return KtError.BadHeader;
            const need = (w_len + out_dim) * 2 * @sizeOf(f32);
            if (off + need > bytes.len) return KtError.TruncatedFile;
            const rdW = struct {
                fn go(buf: []const u8, o: *usize, a: std.mem.Allocator, n: usize) ![]f32 {
                    const out = try a.alloc(f32, n);
                    @memcpy(out, std.mem.bytesAsSlice(f32, buf[o.* .. o.* + n * @sizeOf(f32)]));
                    o.* += n * @sizeOf(f32);
                    return out;
                }
            }.go;
            const w_k = try rdW(bytes, &off, allocator, w_len);
            errdefer allocator.free(w_k);
            const b_k = try rdW(bytes, &off, allocator, out_dim);
            errdefer allocator.free(b_k);
            const w_v = try rdW(bytes, &off, allocator, w_len);
            errdefer allocator.free(w_v);
            const b_v = try rdW(bytes, &off, allocator, out_dim);
            m.* = .{ .dense = .{ .w_k = w_k, .b_k = b_k, .w_v = w_v, .b_v = b_v } };
        }
        debugz.dbg.printLevel(.info, "[kt_transfer] pesos {s}: {d} capas (kv {d}→{d}, hd {d})\n", .{ path, n_layers, n_kv_s, n_kv_t, head_dim });
        return self;
    }
};

/// Fixture del gate-0 y semilla de calibración KT-A: .ktb identity.
/// Los fixtures viven en tests/corpora/ (repo) — /tmp es barrido y mató
/// runs en silencio antes (lección corpora PPL).
pub fn writeIdentityKtb(io: std.Io, path: []const u8, n_layers: u32, n_kv_heads: u32, head_dim: u32) !void {
    var out = std.ArrayList(u8).empty;
    defer out.deinit(std.heap.page_allocator);
    var hdr: [24]u8 = undefined;
    std.mem.writeInt(u32, hdr[0..4], KT_MAGIC, .little);
    std.mem.writeInt(u32, hdr[4..8], KT_VERSION, .little);
    std.mem.writeInt(u32, hdr[8..12], n_layers, .little);
    std.mem.writeInt(u32, hdr[12..16], n_kv_heads, .little);
    std.mem.writeInt(u32, hdr[16..20], n_kv_heads, .little);
    std.mem.writeInt(u32, hdr[20..24], head_dim, .little);
    try out.appendSlice(std.heap.page_allocator, &hdr);
    var kbuf: [4]u8 = undefined;
    std.mem.writeInt(u32, kbuf[0..4], @intFromEnum(LayerMapperKind.identity), .little);
    for (0..n_layers) |_| try out.appendSlice(std.heap.page_allocator, &kbuf);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = out.items });
}

pub const KtModelSpec = struct {
    manager: *kvcache.KVCacheManager,
    rope_base: f32 = 10000.0,
    pairing: rope_mod.RopePairing = .auto,
};

pub const KtRuntime = struct {
    allocator: std.mem.Allocator,
    weights: *const KtWeights,
    seq_source: u64,
    seq_target: u64,

    const Self = @This();

    /// Transfiere `n_tokens` del cache source (len == n_tokens) al target
    /// (arranca en 0). Por capa: retrieve → strip RoPE → mapper → re-rope
    /// → append. El advance del target lo hace el caller (contrato 7.2:
    /// UNA vez por chunk — lo hace este transfer al final).
    pub fn transfer(
        self: *Self,
        source: KtModelSpec,
        target: KtModelSpec,
        engine: *matmul.MatmulEngine,
        n_tokens: usize,
    ) !void {
        const hd: usize = self.weights.head_dim;
        const n_kv_s: usize = self.weights.n_kv_heads_source;
        const n_kv_t: usize = self.weights.n_kv_heads_target;
        const in_dim = n_kv_s * hd;
        const out_dim = n_kv_t * hd;

        const src_len = try source.manager.getSequenceLen(self.seq_source);
        if (n_tokens == 0 or src_len < n_tokens) return KtError.SequenceEmpty;
        const tgt_len = try target.manager.getSequenceLen(self.seq_target);
        if (tgt_len != 0) return KtError.BadHeader; // target debe arrancar vacío

        // Scratch del transfer (una sola allocación por transfer, reusada
        // por TODAS las capas — el append por head copia a chunks locales).
        const k_head_src = try self.allocator.alloc(f16, n_tokens * hd); // [pos·hd] un head
        defer self.allocator.free(k_head_src);
        const v_head_src = try self.allocator.alloc(f16, n_tokens * hd);
        defer self.allocator.free(v_head_src);
        // x_k/x_v: [tokens, in_dim] row-major (entrada del mapper).
        const x_k = try self.allocator.alloc(f32, n_tokens * in_dim);
        defer self.allocator.free(x_k);
        const x_v = try self.allocator.alloc(f32, n_tokens * in_dim);
        defer self.allocator.free(x_v);
        // y_k/y_v: [tokens, out_dim] (salida del mapper dense).
        const y_k = try self.allocator.alloc(f32, n_tokens * out_dim);
        defer self.allocator.free(y_k);
        const y_v = try self.allocator.alloc(f32, n_tokens * out_dim);
        defer self.allocator.free(y_v);

        for (self.weights.layers, 0..) |*m, li_usize| {
            const li: u32 = @intCast(li_usize);

            // (1)+(2) retrieve per-kv-head + strip RoPE → x en pre-RoPE.
            for (0..n_kv_s) |kv_h| {
                try source.manager.retrieveForAttention(self.seq_source, li, @intCast(kv_h), k_head_src, v_head_src);
                for (0..n_tokens) |pos| {
                    const src = pos * hd;
                    const row = pos * in_dim + kv_h * hd;
                    rope_mod.applyRoPEInverseOnSlice(
                        f16,
                        k_head_src[src .. src + hd],
                        x_k[row .. row + hd],
                        pos,
                        hd,
                        source.rope_base,
                        source.pairing,
                    );
                    for (0..hd) |c| x_v[row + c] = @as(f32, @floatCast(v_head_src[src + c]));
                }
            }

            // (3) mapper K/V.
            switch (m.*) {
                .identity => {
                    // x ya está en pre-RoPE: solo re-empaquetar a y (f32→f16 va
                    // en el append).
                    for (0..n_tokens * out_dim) |i| {
                        y_k[i] = x_k[i];
                        y_v[i] = x_v[i];
                    }
                },
                .dense => |*d| {
                    try self.projDense(engine, x_k, d.w_k, d.b_k, y_k, n_tokens, in_dim, out_dim);
                    try self.projDense(engine, x_v, d.w_v, d.b_v, y_v, n_tokens, in_dim, out_dim);
                },
            }

            // (4)+(5) re-rope target sobre K + append per-kv-head.
            for (0..n_kv_t) |kv_h| {
                // Re-rope K in-place por posición (slice del pack row-major).
                for (0..n_tokens) |pos| {
                    const row = pos * out_dim + kv_h * hd;
                    rope_mod.applyRoPEForwardOnSlice(
                        f32,
                        y_k[row .. row + hd],
                        pos,
                        hd,
                        target.rope_base,
                        target.pairing,
                    );
                }
            }
            // Chunks [pos·hd] por head (contrato appendTokensF16).
            const k_chunk = try self.allocator.alloc(f16, n_tokens * hd);
            defer self.allocator.free(k_chunk);
            const v_chunk = try self.allocator.alloc(f16, n_tokens * hd);
            defer self.allocator.free(v_chunk);
            for (0..n_kv_t) |kv_h| {
                for (0..n_tokens) |pos| {
                    const src = pos * out_dim + kv_h * hd;
                    const dst = pos * hd;
                    for (0..hd) |c| {
                        k_chunk[dst + c] = @floatCast(y_k[src + c]);
                        v_chunk[dst + c] = @floatCast(y_v[src + c]);
                    }
                }
                try target.manager.appendTokensF16(self.seq_target, li, @intCast(kv_h), k_chunk, v_chunk);
            }

            if (debugz.dbg.at(.info)) {
                debugz.dbg.printLevel(.info, "[kt_transfer] capa {d}: {d} tok ({s})\n", .{ li, n_tokens, @tagName(m.*) });
            }
        }

        // Advance del target: n_tokens UNA vez (contrato 7.2 del pipeline —
        // todas las capas escribieron en offset current_len compartido).
        for (0..n_tokens) |_| try target.manager.advanceSequence(self.seq_target);
    }

    /// y = x·W^T + b con el dispatcher gemm del MatmulEngine (cuBLAS si el
    /// backend lo es; tiled/parallel/openblas en CPU).
    fn projDense(
        self: *Self,
        engine: *matmul.MatmulEngine,
        x: []const f32,
        w: []const f32,
        b: []const f32,
        y: []f32,
        m: usize,
        kk: usize,
        n: usize,
    ) !void {
        // OJO Tensor.fromSlice DUPEa los datos: el gemm escribe en la copia
        // C, no en y — hay que devolver el resultado (lección del dense-I
        // gate: gemm correcto + y intacto ⇒ garbage post-first-token).
        var A = try Tensor(f32).fromSlice(self.allocator, x, &.{ m, kk });
        defer A.deinit();
        var B = try Tensor(f32).fromSlice(self.allocator, w, &.{ n, kk });
        defer B.deinit();
        var C = try Tensor(f32).fromSlice(self.allocator, y, &.{ m, n });
        defer C.deinit();
        try engine.gemm(f32, A, B, &C, false, true);
        @memcpy(y[0 .. m * n], C.data[0 .. m * n]);
        for (0..m) |r| {
            for (0..n) |c| y[r * n + c] += b[c];
        }
    }
};

test "ktb identity: write + load roundtrip header" {
    const allocator = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const tmp = "tests/corpora/ktb_identity_test.ktb";
    try writeIdentityKtb(io, tmp, 4, 8, 128);
    var w = try KtWeights.load(allocator, io, tmp, 4, 8, 8, 128);
    defer w.deinit();
    try std.testing.expectEqual(@as(u32, 4), w.n_layers);
    try std.testing.expectEqual(@as(u32, 8), w.n_kv_heads_source);
    try std.testing.expectEqual(@as(u32, 8), w.n_kv_heads_target);
    try std.testing.expectEqual(@as(u32, 128), w.head_dim);
    for (w.layers) |m| try std.testing.expect(m == .identity);
    // Mismatch de geometría → error claro.
    try std.testing.expectError(KtError.LayerCountMismatch, KtWeights.load(allocator, io, tmp, 7, 8, 8, 128));
    try std.testing.expectError(KtError.BadHeader, KtWeights.load(allocator, io, tmp, 4, 9, 8, 128));
    try std.testing.expectError(KtError.HeadDimMismatch, KtWeights.load(allocator, io, tmp, 4, 8, 8, 64));
}

test "ktb dense: write + load + valores" {
    const allocator = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var body = std.ArrayList(u8).empty;
    defer body.deinit(std.heap.page_allocator);
    var hdr: [24]u8 = undefined;
    std.mem.writeInt(u32, hdr[0..4], KT_MAGIC, .little);
    std.mem.writeInt(u32, hdr[4..8], KT_VERSION, .little);
    std.mem.writeInt(u32, hdr[8..12], 1, .little); // 1 capa
    std.mem.writeInt(u32, hdr[12..16], 2, .little); // kv_s
    std.mem.writeInt(u32, hdr[16..20], 3, .little); // kv_t
    std.mem.writeInt(u32, hdr[20..24], 4, .little); // hd
    try body.appendSlice(std.heap.page_allocator, &hdr);
    var kbuf: [4]u8 = undefined;
    std.mem.writeInt(u32, kbuf[0..4], @intFromEnum(LayerMapperKind.dense), .little);
    try body.appendSlice(std.heap.page_allocator, &kbuf);
    var w_k: [12]f32 = undefined;
    for (&w_k, 0..) |*v, i| v.* = @floatFromInt(i);
    try body.appendSlice(std.heap.page_allocator, std.mem.sliceAsBytes(&w_k));
    const b_k: [12]f32 = @splat(0.5);
    try body.appendSlice(std.heap.page_allocator, std.mem.sliceAsBytes(&b_k));
    var w_v: [12]f32 = undefined;
    for (&w_v, 0..) |*v, i| v.* = -@as(f32, @floatFromInt(i));
    try body.appendSlice(std.heap.page_allocator, std.mem.sliceAsBytes(&w_v));
    const b_v: [12]f32 = @splat(-0.5);
    try body.appendSlice(std.heap.page_allocator, std.mem.sliceAsBytes(&b_v));
    const path = "tests/corpora/ktb_dense_test.ktb";
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = body.items });

    var w = try KtWeights.load(allocator, io, path, 1, 2, 3, 4);
    defer w.deinit();
    const d = w.layers[0].dense;
    try std.testing.expect(d.w_k.len == 12 and d.w_k[7] == 7.0);
    try std.testing.expect(d.b_k[0] == 0.5);
    try std.testing.expect(d.w_v[3] == -3.0);
    try std.testing.expect(d.b_v[11] == -0.5);
}
