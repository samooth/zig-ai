//! Captura de hidden states del engine para entrenamiento RLT (fase 2).
//!
//! El pipeline (α=0, feedback OFF) corre el corpus por prefill y vuelca:
//!   tokens.bin      u32[T]        — ids de tokens (labels next-token)
//!   layerN.e.bin    f32[T*d]      — entrada de la capa N (hidden stream)
//!   layerN.s0.bin   f32[d]        — estado inicial (zeros, informativo)
//!   meta.txt        T, d, n_layers (validación al cargar)
//!
//! Semántica del engine (hybrid_layer.zig:737): s_prev[t] = OUT de la capa en
//! t-1. Para el merge de la capa N en el token t el estado real es
//! out_{N}[t-1] — que el trainer aproxima con e_N[t-1] (misma d, stream
//! adyacente) en BPTT depth-1 con "proxy state". La captura guarda SOLO las
//! entradas (e); los outs se derivan como e_{N+1} (excepto última capa).
//!
//! Formato .rltcap (un único fichero, endian nativo, simple de parsear):
//!   [magic "RLTC" u32][version u32=1][T u32][d u32][n_layers u32]
//!   [tokens: T×u32]
//!   [e por capa contigua: n_layers × T × d × f32]  (capa 0 primero)
const builtin = @import("builtin");
const std = @import("std");
const mem = std.mem;

pub const MAGIC: u32 = 0x43544C52; // "RLTC" LE
pub const VERSION: u32 = 1;

pub const CaptureHeader = struct {
    T: u32,
    d: u32,
    n_layers: u32,
};

pub const Capture = struct {
    header: CaptureHeader,
    tokens: []u32, // [T]
    /// Entradas por capa: e[layer][t*d .. (t+1)*d]
    e: []f32, // [n_layers][T][d] contiguo

    pub fn tokensSlice(self: *const Capture) []u32 {
        return self.tokens;
    }

    /// e de una capa: [T*d]
    pub fn layerE(self: *const Capture, layer: usize) []const f32 {
        const T = self.header.T;
        const d = self.header.d;
        const off = layer * T * d;
        return self.e[off .. off + T * d];
    }

    /// e de un token en una capa: [d]
    pub fn tokenE(self: *const Capture, layer: usize, t: usize) []const f32 {
        const d = self.header.d;
        const base = self.layerE(layer);
        return base[t * d ..][0..d];
    }

    pub fn deinit(self: *Capture, allocator: std.mem.Allocator) void {
        allocator.free(self.tokens);
        allocator.free(self.e);
    }
};

// ─── Writer ─────────────────────────────────────────────────────────────────

/// mkdir -p vía libc (patrón hf_download.zig:87 — Io.Dir no expone makePath).
fn mkdirP(path: []const u8) void {
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

pub const CaptureWriter = struct {
    io: std.Io,
    allocator: std.mem.Allocator,
    dir: std.Io.Dir,
    dir_path: []const u8,
    T: usize,
    d: usize,
    n_layers: usize,
    layers_to_capture: []const usize,
    /// Buffer por capa: [T*d] f32 — el pipeline copia e[t] tras cada forward.
    e_bufs: [][]f32,
    tokens: []u32,
    next_layer: usize = 0,

    /// `layers_to_capture`: índices de capa a capturar (del env). Deben estar
    /// ordenados ascendente — el pipeline llama addToken por capa en orden.
    pub fn init(
        io: std.Io,
        allocator: std.mem.Allocator,
        dir_path: []const u8,
        T: usize,
        d: usize,
        layers_to_capture: []const usize,
        tokens: []const u32,
    ) !CaptureWriter {
        std.debug.assert(layers_to_capture.len > 0);

        // mkdir -p vía libc (patrón hf_download.zig:87 — Io.Dir no expone makePath)
        mkdirP(dir_path);
        const dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{});

        const n = layers_to_capture.len;
        const e_bufs = try allocator.alloc([]f32, n);
        errdefer allocator.free(e_bufs);
        for (e_bufs) |*b| {
            b.* = try allocator.alloc(f32, T * d);
            @memset(b.*, 0);
        }

        const toks = try allocator.alloc(u32, T);
        @memcpy(toks, tokens);

        return .{
            .io = io,
            .allocator = allocator,
            .dir = dir,
            .dir_path = dir_path,
            .T = T,
            .d = d,
            .n_layers = n,
            .layers_to_capture = layers_to_capture,
            .e_bufs = e_bufs,
            .tokens = toks,
        };
    }

    pub fn deinit(self: *CaptureWriter) void {
        for (self.e_bufs) |b| self.allocator.free(b);
        self.allocator.free(self.e_bufs);
        self.allocator.free(self.tokens);
        self.dir.close(self.io);
    }

    /// El pipeline llama esto tras el forward de la capa `layer_idx` con el
    /// hidden stream COMPLETO [T*d] (entrada de la capa = e para todos los
    /// tokens). Solo captura si layer_idx es el siguiente de la lista.
    pub fn addLayer(self: *CaptureWriter, layer_idx: usize, e_all: []const f32) void {
        if (self.next_layer >= self.layers_to_capture.len) return;
        if (self.layers_to_capture[self.next_layer] != layer_idx) return;
        std.debug.assert(e_all.len >= self.T * self.d);
        @memcpy(self.e_bufs[self.next_layer], e_all[0 .. self.T * self.d]);
        self.next_layer += 1;
    }

    /// Vuelca todo a <dir>/<name>.rltcap
    pub fn flush(self: *CaptureWriter, name: []const u8) !void {
        var path_buf: [256]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ self.dir_path, name });

        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(self.allocator);

        // Header: magic + version + T + d + n_layers
        var hdr: [20]u8 = undefined;
        std.mem.writeInt(u32, hdr[0..4], MAGIC, .little);
        std.mem.writeInt(u32, hdr[4..8], VERSION, .little);
        std.mem.writeInt(u32, hdr[8..12], @intCast(self.T), .little);
        std.mem.writeInt(u32, hdr[12..16], @intCast(self.d), .little);
        std.mem.writeInt(u32, hdr[16..20], @intCast(self.n_layers), .little);
        try buf.appendSlice(self.allocator, &hdr);

        // Tokens
        try buf.appendSlice(self.allocator, mem.sliceAsBytes(self.tokens));

        // e por capa
        for (self.e_bufs) |b| {
            try buf.appendSlice(self.allocator, mem.sliceAsBytes(b));
        }

        const file = try self.dir.createFile(self.io, path, .{});
        defer file.close(self.io);
        var wbuf: [4096]u8 = undefined;
        var fw = file.writer(self.io, &wbuf);
        const w = &fw.interface;
        try w.writeAll(buf.items);
        try w.flush();

        std.debug.print("[rlt_capture] {s}: T={d} d={d} layers={d} ({d} bytes)\n", .{
            path, self.T, self.d, self.n_layers, buf.items.len,
        });
    }
};

// ─── Reader (trainer) ──────────────────────────────────────────────────────

/// Lee un .rltcap completo. El trainer posee tokens+e hasta deinit.
pub fn loadCapture(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
) !Capture {
    const dir = std.Io.Dir.cwd();
    const data = try dir.readFileAlloc(io, path, allocator, .unlimited);
    defer allocator.free(data);

    if (data.len < 20) return error.TruncatedCapture;
    if (std.mem.readInt(u32, data[0..4], .little) != MAGIC) return error.BadMagic;
    if (std.mem.readInt(u32, data[4..8], .little) != VERSION) return error.BadVersion;

    const T = std.mem.readInt(u32, data[8..12], .little);
    const d = std.mem.readInt(u32, data[12..16], .little);
    const n_layers = std.mem.readInt(u32, data[16..20], .little);
    if (T == 0 or d == 0 or n_layers == 0) return error.EmptyCapture;

    const need = 20 + T * 4 + n_layers * T * d * 4;
    if (data.len < need) return error.TruncatedCapture;

    const tokens = try allocator.alloc(u32, T);
    errdefer allocator.free(tokens);
    @memcpy(tokens, mem.bytesAsSlice(u32, data[20 .. 20 + T * 4]));

    const e = try allocator.alloc(f32, n_layers * T * d);
    errdefer allocator.free(e);
    @memcpy(e, mem.bytesAsSlice(f32, data[20 + T * 4 ..][0 .. n_layers * T * d * 4]));

    return .{
        .header = .{ .T = T, .d = d, .n_layers = n_layers },
        .tokens = tokens,
        .e = e,
    };
}

// ─── Tests ─────────────────────────────────────────────────────────────────

test "capture roundtrip writer→reader" {
    const allocator = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();

    const T: usize = 4;
    const d: usize = 3;
    const layers = [_]usize{ 0, 2 };

    var ts: std.posix.timespec = undefined;
    if (builtin.target.os.tag != .windows) _ = std.posix.system.clock_gettime(.REALTIME, &ts);
    const nsec32: u32 = @truncate(@as(u64, @bitCast(ts.nsec)));
    var name_buf: [64]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "/tmp/rltcap_{d}.rltcap", .{nsec32});

    // Fichero directo (no dir+name) — el test usa flush con dir_path="/tmp"
    // y name relativo. Ajuste: escribimos en /tmp con nombre plano.
    const tokens = [_]u32{ 5, 10, 15, 20 };

    const layers_slice: []const usize = &layers;
    const tokens_slice: []const u32 = &tokens;
    var cw = try CaptureWriter.init(io, allocator, "/tmp", T, d, layers_slice, tokens_slice);
    defer cw.deinit();

    // T tokens × 2 capas: e[layer][t] = valores distinguibles. addLayer recibe
    // el stream completo [T*d] de la capa.
    for (layers, 0..) |li, li_idx| {
        const e_all = try allocator.alloc(f32, T * d);
        defer allocator.free(e_all);
        for (0..T) |t| {
            for (0..d) |i| {
                e_all[t * d + i] = @floatFromInt(li * 100 + t * 10 + i);
            }
        }
        cw.addLayer(li, e_all);
        _ = li_idx;
    }

    // flush espera name RELATIVO al dir; construimos el path final
    var rel_buf: [64]u8 = undefined;
    const rel = try std.fmt.bufPrint(&rel_buf, "rltcap_{d}.rltcap", .{nsec32});
    try cw.flush(rel);
    const rm = struct {
        extern "c" fn unlink(path: [*:0]const u8) c_int;
        fn rmTmp(path: []const u8) void {
            var zbuf: [128]u8 = undefined;
            if (path.len >= zbuf.len) return;
            @memcpy(zbuf[0..path.len], path);
            zbuf[path.len] = 0;
            _ = unlink(zbuf[0..path.len :0].ptr);
        }
    }.rmTmp;
    defer rm(name);

    var cap = try loadCapture(io, allocator, name);
    defer cap.deinit(allocator);

    try std.testing.expectEqual(@as(u32, T), cap.header.T);
    try std.testing.expectEqual(@as(u32, d), cap.header.d);
    try std.testing.expectEqual(@as(u32, 2), cap.header.n_layers);
    try std.testing.expectEqualSlices(u32, &tokens, cap.tokens);

    // e de capa 0, token 2 = 0*100 + 2*10 + i
    const e02 = cap.tokenE(0, 2);
    try std.testing.expectApproxEqAbs(@as(f32, 20), e02[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 21), e02[1], 1e-6);
    // e de capa 2 (idx 1 en captura), token 1
    const e21 = cap.tokenE(1, 1);
    try std.testing.expectApproxEqAbs(@as(f32, 210), e21[0], 1e-6);
}
