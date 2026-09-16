//! KvTrace — captura de trazas K/V del forward CPU para el paso 1 del
//! KV-Codec (docs/PLAN_KVCODEC_STEP1.md §A1).
//!
//! Gate: `DUMP_KV_TRACE=<dir>` (env, null = coste cero absoluto).
//! Dump SIEMPRE f16 (presupuesto disco, ~3GB por 3 modelos × 2 corpus).
//!
//! Layout de captura [T, n_kv_head, head_dim] (v y k) / [64, n_head, head_dim]
//! (q_tail: SOLO últimas 64 posiciones del chunk actual).
//!
//! Naming: `<dir>/<model>/<corpus>/L<layer>/{k_pre,k_post,v,q_tail}.bin`
//! + `<dir>/<model>/<corpus>/manifest.json`.
//!
//! La captura es APPEND por chunk de prefill: kv_trace.appendChunk acumula
//! en buffers f16 crecientes por capa; el pipeline llama a kv_trace.finish()
//! al final de la generación para volcar a disco. Un solo modelo+corpus por
//! proceso (manifest único, sin ambigüedad de estado).

const std = @import("std");

// ═══════════════════════════════════════════════════════════════════
// Hook global de captura (lane-kvc paso 1): hybrid_attn.forward escribe
// aquí los 3 snapshots por chunk. main setea el tracer al arrancar y
// apaga al terminar. Coste cero absoluto cuando active == null.
// ═══════════════════════════════════════════════════════════════════

pub const hook = struct {
    /// Tracer activo (main lo setea; null = captura off).
    pub var active: ?*KvTrace = null;
    /// Staging por forward (f32); UNA capa a la vez (forward secuencial).
    var k_pre_buf: std.ArrayList(f32) = .empty;
    var v_buf: std.ArrayList(f32) = .empty;
    var gpa: ?std.mem.Allocator = null;

    /// main: activar/desactivar captura con su tracer y allocator.
    pub fn setTracer(tr: ?*KvTrace, allocator: std.mem.Allocator) void {
        active = tr;
        gpa = allocator;
        k_pre_buf.clearRetainingCapacity();
        v_buf.clearRetainingCapacity();
    }

    /// hybrid_attn: stage de k_pre (tras RMSNorm, pre-RoPE).
    pub fn stageKPre(data: []const f32) !void {
        const a = gpa orelse return;
        k_pre_buf.clearRetainingCapacity();
        try k_pre_buf.ensureTotalCapacity(a, data.len);
        for (data) |x| k_pre_buf.appendAssumeCapacity(x);
    }

    /// hybrid_attn: stage de v (post-reshape, pre-caché — V no lleva RoPE).
    pub fn stageV(data: []const f32) !void {
        const a = gpa orelse return;
        v_buf.clearRetainingCapacity();
        try v_buf.ensureTotalCapacity(a, data.len);
        for (data) |x| v_buf.appendAssumeCapacity(x);
    }

    /// hybrid_attn: cierre de chunk — k_post + q van directos del forward.
    pub fn appendChunk(layer_idx: usize, n: usize, kv_dim: usize, k_post: []const f32, q: []const f32) !void {
        const tr = active orelse return;
        const k_pre: []const f32 = k_pre_buf.items;
        const v: []const f32 = v_buf.items;
        try tr.appendChunk(layer_idx, n, k_pre, k_post[0 .. n * kv_dim], v, q);
    }
};

pub const max_layers = 128;
pub const q_tail_len = 64;

/// Buffer de una capa: acumula [T, elems] en f16 little-endian.
const LayerBuf = struct {
    k_pre: std.ArrayList(u16) = .empty,
    k_post: std.ArrayList(u16) = .empty,
    v: std.ArrayList(u16) = .empty,
    q_tail: std.ArrayList(u16) = .empty,
    tokens: usize = 0,
};

pub const KvTrace = struct {
    allocator: std.mem.Allocator,
    io: *std.Io,
    /// Directorio raíz (env DUMP_KV_TRACE).
    root_dir: []const u8,
    /// Etiqueta de modelo (basename del .gguf sin extensión).
    model: []const u8,
    /// Etiqueta de corpus ("prose" | "code").
    corpus: []const u8,
    layers: []LayerBuf,
    layer_count: usize = 0,
    n_kv_head: usize = 0,
    head_dim: usize = 0,
    n_head: usize = 0,
    /// Metadatos opcionales inyectados por el caller antes de finish().
    gguf_sha: []const u8 = "",
    corpus_file: []const u8 = "",
    corpus_sha: []const u8 = "",
    lane_base_sha: []const u8 = "",
    tokenizer: []const u8 = "",
    enabled: bool = true,

    pub fn init(
        allocator: std.mem.Allocator,
        io: *std.Io,
        root_dir: []const u8,
        model: []const u8,
        corpus: []const u8,
        num_layers: usize,
        n_kv_head: usize,
        head_dim: usize,
        n_head: usize,
    ) !KvTrace {
        const layers = try allocator.alloc(LayerBuf, @min(num_layers, max_layers));
        for (layers) |*l| l.* = .{};
        return .{
            .allocator = allocator,
            .io = io,
            .root_dir = root_dir,
            .model = try allocator.dupe(u8, model),
            .corpus = try allocator.dupe(u8, corpus),
            .layers = layers,
            .layer_count = @min(num_layers, max_layers),
            .n_kv_head = n_kv_head,
            .head_dim = head_dim,
            .n_head = n_head,
        };
    }

    pub fn deinit(self: *KvTrace) void {
        for (self.layers) |*l| {
            l.k_pre.deinit(self.allocator);
            l.k_post.deinit(self.allocator);
            l.v.deinit(self.allocator);
            l.q_tail.deinit(self.allocator);
        }
        self.allocator.free(self.layers);
        self.allocator.free(self.model);
        self.allocator.free(self.corpus);
    }

    fn f16ToBits(x: f32) u16 {
        return @bitCast(@as(f16, @floatCast(x)));
    }

    /// Añade un chunk de prefill de la capa `layer_idx`.
    /// Buffers: k_pre/k_post/v [n, n_kv_head, head_dim] f32; q [n, n_head, head_dim]
    /// f32 POST-RoPE (se trunca a las últimas q_tail_len posiciones).
    pub fn appendChunk(
        self: *KvTrace,
        layer_idx: usize,
        n: usize,
        k_pre: []const f32,
        k_post: []const f32,
        v: []const f32,
        q: []const f32,
    ) !void {
        if (!self.enabled) return;
        if (layer_idx >= self.layer_count) return;
        const l = &self.layers[layer_idx];
        const kv_elems = self.n_kv_head * self.head_dim;

        const k_pre_total = n * kv_elems;
        try l.k_pre.ensureUnusedCapacity(self.allocator, k_pre_total);
        try l.k_post.ensureUnusedCapacity(self.allocator, k_pre_total);
        try l.v.ensureUnusedCapacity(self.allocator, k_pre_total);
        for (k_pre[0..k_pre_total]) |x| l.k_pre.appendAssumeCapacity(f16ToBits(x));
        for (k_post[0..k_pre_total]) |x| l.k_post.appendAssumeCapacity(f16ToBits(x));
        for (v[0..k_pre_total]) |x| l.v.appendAssumeCapacity(f16ToBits(x));
        l.tokens += n;

        // q_tail: últimas q_tail_len posiciones de ESTE chunk (prefill por
        // chunks: el tail del último chunk es el tail de la secuencia).
        const q_elems = self.n_head * self.head_dim;
        const tail_n = @min(q_tail_len, n);
        const tail_from = n - tail_n;
        // Reemplaza el tail anterior: el último chunk manda.
        l.q_tail.clearRetainingCapacity();
        try l.q_tail.ensureTotalCapacity(self.allocator, tail_n * q_elems);
        for (0..tail_n) |t| {
            for (0..q_elems) |i| {
                l.q_tail.appendAssumeCapacity(f16ToBits(q[(tail_from + t) * q_elems + i]));
            }
        }
    }

    /// Volcar todo a disco + manifest. Borra los buffers al terminar.
    pub fn finish(self: *KvTrace) !void {
        if (!self.enabled) return;
        // mkdir -p root/model/corpus vía libc (Io.Dir.createDirPath necesita
        // recorrer; patrón hf_download mkdirP).
        var pbuf: [512]u8 = undefined;
        const base = try std.fmt.bufPrint(&pbuf, "{s}/{s}/{s}", .{ self.root_dir, self.model, self.corpus });
        mkdirP(base);
        var cbuf: [512]u8 = undefined;

        for (self.layers, 0..) |*l, li| {
            if (l.tokens == 0) continue;
            const ldir = try std.fmt.bufPrint(&cbuf, "{s}/L{d}", .{ base, li });
            mkdirP(ldir);

            const kv_total = l.tokens * self.n_kv_head * self.head_dim;
            try self.writeBin(ldir, "k_pre.bin", l.k_pre.items[0..kv_total]);
            try self.writeBin(ldir, "k_post.bin", l.k_post.items[0..kv_total]);
            try self.writeBin(ldir, "v.bin", l.v.items[0..kv_total]);
            if (l.q_tail.items.len > 0) {
                try self.writeBin(ldir, "q_tail.bin", l.q_tail.items);
            }
        }

        // manifest.json: append si existe (multi-corpus mismo modelo).
        var mbuf: [2048]u8 = undefined;
        const mpath = try std.fmt.bufPrint(&mbuf, "{s}/manifest.json", .{base});
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.REALTIME, &ts);
        const entry = try std.fmt.bufPrint(&pbuf, // reusa pbuf (base ya usado)
            "{{\"model\":\"{s}\",\"corpus\":\"{s}\",\"gguf_sha\":\"{s}\",\"T\":{d}," ++
                "\"n_kv_head\":{d},\"head_dim\":{d},\"n_head\":{d},\"layers_captured\":{d}," ++
                "\"corpus_file\":\"{s}\",\"corpus_sha\":\"{s}\",\"tokenizer\":\"{s}\"," ++
                "\"lane_base_sha\":\"{s}\",\"timestamp\":{d}}}\n", .{
                self.model,       self.corpus,
                self.gguf_sha,
                blk: {
                    // primera capa con tokens (la 0 puede ser SSM sin captura)
                    for (self.layers) |*l| {
                        if (l.tokens > 0) break :blk l.tokens;
                    }
                    break :blk 0;
                },
                self.n_kv_head,   self.head_dim,
                self.n_head,
                blk: {
                    var c: usize = 0;
                    for (self.layers) |*l| {
                        if (l.tokens > 0) c += 1;
                    }
                    break :blk c;
                },
                self.corpus_file, self.corpus_sha,
                self.tokenizer,   self.lane_base_sha,
                @as(u64, @intCast(ts.sec)),
            });
        appendFileAbsolute(self.io, mpath, entry);
        // (breadcrumb de volcado lo emite main vía debug.dbg tras finish)
        return;
    }

    fn writeBin(self: *KvTrace, dir: []const u8, name: []const u8, data: []const u16) !void {
        var pbuf: [512]u8 = undefined;
        const path = try std.fmt.bufPrint(&pbuf, "{s}/{s}", .{ dir, name });
        const bytes = std.mem.sliceAsBytes(data);
        var file = try std.Io.Dir.cwd().createFile(self.io.*, path, .{});
        defer file.close(self.io.*);
        try file.writeStreamingAll(self.io.*, bytes);
    }
};

/// mkdir -p vía libc (patrón hf_download.zig:87 — Io.Dir no expone makePath
/// sin Io de recorridos; mkdir es idempotente).
pub fn mkdirP(path: []const u8) void {
    var buf: [512]u8 = undefined;
    if (path.len >= buf.len) return;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    var i: usize = 1;
    while (i <= path.len) : (i += 1) {
        if (buf[i] == '/' or buf[i] == 0) {
            buf[i] = 0;
            _ = std.c.mkdir(@ptrCast(buf[0..].ptr), 0o755);
            if (i < path.len) buf[i] = '/';
        }
    }
}

fn appendFileAbsolute(io: *std.Io, path: []const u8, data: []const u8) void {
    _ = io; // fopen libc no necesita instancia Io
    // openFile con .mode append no está en CreateFlags; usar fopen libc.
    var cpath_buf: [512]u8 = undefined;
    if (path.len >= cpath_buf.len) return;
    @memcpy(cpath_buf[0..path.len], path);
    cpath_buf[path.len] = 0;
    const cpath: [*c]const u8 = @ptrCast(&cpath_buf);
    const f = std.c.fopen(cpath, "ab") orelse return;
    defer _ = std.c.fclose(f);
    _ = std.c.fwrite(data.ptr, 1, data.len, f);
}
