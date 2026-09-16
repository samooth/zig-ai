//! gguf_moe — detección y stacking MoE como funciones puras sobre GgufFile.
//!
//! Lane E (P2a del ESTUDIO_MEJORAS_FREETOKEN.md §7.2). Este módulo NO carga
//! pesos ni toca GPU: expone referencias crudas zero-copy sobre el mmap del
//! GGUF para que el offload cache (src/moe/offload_cache.zig) y la capa MoE
//! (src/transformer/moe_layer.zig) las consuman directamente.
//!
//! Patrones de nombres soportados (convención llama.cpp sobre GGUF):
//!   Router  : `blk.{N}.ffn_gate_inp.weight`            [n_embd, n_expert]
//!   Fusión  : `blk.{N}.ffn_{gate,up,down}_exps.weight`  ne=[in, out, E]
//!   Split   : `blk.{N}.ffn_{gate,up,down}.experts.{i}.weight` (gpt-oss /
//!             qwen3-moe modernos), apilado válido sólo si los E tensores son
//!             contiguos en el mmap (orden ascendente, back-to-back).
//!
//! Layout GGUF: dims[0] es el eje de variación más rápida; un peso [out,in]
//! se guarda ne=[in,out]. En `_exps` la dimensión de experto es la última ⇒
//! cada experto ocupa un bloque contiguo `expert_bytes` dentro del banco.
//!
//! El stacking split exige contiguidad (error.ExpertsNotContiguous si hay
//! huecos): FreeToken resuelve lo mismo apilando banks contiguos y copiando
//! los dispersos; aquí el convertidor GGUF escribe los expertos consecutivos,
//! así que la vista directa cubre el caso real sin una sola copia.
//!
//! Breadcrumbs: gated por DEBUG_LEVEL (debug.dbg.printLevel) + flag propio
//! MOE_DEBUG=1 (routing/detección por capa). debug.zig queda intocado.

const std = @import("std");
const debug = @import("debug");
const gguf = @import("gguf");

pub const MoeError = error{
    RouterNotFound,
    ExpertsNotFound,
    ExpertsNotContiguous,
    ExpertDimMismatch,
    InvalidRouterDims,
    UnsupportedRouterDtype,
    MissingMetadata,
    NameTooLong,
};

pub const MoeFamily = enum {
    gpt_oss,
    qwen2_moe,
    qwen35_moe,
    gemma_moe,
    generic,

    pub fn fromArch(arch: []const u8) MoeFamily {
        if (std.mem.eql(u8, arch, "gpt-oss")) return .gpt_oss;
        if (std.mem.eql(u8, arch, "qwen2moe")) return .qwen2_moe;
        if (std.mem.eql(u8, arch, "qwen35moe") or std.mem.eql(u8, arch, "qwen3moe"))
            return .qwen35_moe;
        if (std.mem.startsWith(u8, arch, "gemma") and
            (std.mem.endsWith(u8, arch, "moe") or std.mem.indexOf(u8, arch, "moe") != null))
            return .gemma_moe;
        return .generic;
    }
};

pub const BankKind = enum { gate, up, down };

/// Banco de expertos apilado [n_expert × out_dim filas], vista cruda contigua
/// sobre el mmap (jamás copia). `row_bytes` es el stride de UNA fila de salida
/// (lo que consume el expert-GEMM/GEMV); los bytes son canónicos GGUF con
/// escalas inline según Contrato 1 del PLAN_MAESTRO.
pub const BankRef = struct {
    bytes: []const u8,
    dtype: gguf.GgmlType,
    /// Dimensión de entrada (ne[0] del tensor per-experto).
    in_dim: usize,
    /// Dimensión de salida (filas; ne[1]).
    out_dim: usize,
    n_expert: usize,
    /// Escalas EXTERNAS por experto (ticket 4.4): presets UD-XL (p.ej.
    /// gemma-4-26B-A4B-Q6_K_XL) escriben `ffn_{k}_exps.scale` [n_expert] f32
    /// SEPARADO del peso; el peso q8_0 lleva d embebidas SIN la escala global
    /// ⇒ el GEMV canónico lee mal. `composeCanonical` produce el banco
    /// canónico multiplicando cada d de bloque por scale[e]. Null = inline
    /// canónico (sin tensor .scale).
    external_scale: ?[]const f32 = null,
    /// Ventana por bloque (para bancos combinados tipo `ffn_gate_up_exps`,
    /// donde cada experto empaqueta [gate_rows | up_rows] contiguos):
    ///   block_stride = bytes del bloque COMPLETO del experto (default 0 ⇒
    ///   derivado = bytes.len/n_expert), block_offset = offset de ESTA vista
    ///   dentro del bloque (default 0). region_len = bytes útiles de la
    ///   ventana por experto (default 0 ⇒ block_stride - block_offset).
    block_stride: usize = 0,
    block_offset: usize = 0,
    region_len: usize = 0,

    /// Bytes del BLOQUE completo de un experto (incluye otras ventanas).
    pub fn blockBytes(self: BankRef) usize {
        if (self.block_stride != 0) return self.block_stride;
        return self.bytes.len / self.n_expert;
    }

    /// Bytes de la VENTANA de esta vista dentro de cada bloque.
    pub fn expertBytes(self: BankRef) usize {
        if (self.region_len != 0) return self.region_len;
        return self.blockBytes() - self.block_offset;
    }

    /// Bytes de una fila de salida (stride del GEMV por experto).
    pub fn rowBytes(self: BankRef) usize {
        return self.expertBytes() / self.out_dim;
    }

    /// Slice crudo de la ventana del experto `e` (fila base para el gather E4).
    pub fn expertSlice(self: BankRef, e: usize) []const u8 {
        std.debug.assert(e < self.n_expert);
        const bs = self.blockBytes();
        const off = self.block_offset + e * bs;
        const len = self.expertBytes();
        // Con ventana, los bloques viven DENTRO de bytes (banco combinado
        // compartido con otra vista); sin ventana, bytes.len == n*bs.
        return self.bytes[off..][0..len];
    }

    /// Ticket 4.4: produce una COPIA canónica del banco aplicando las
    /// escalas externas por experto. Para cada bloque cuantizado del experto
    /// `e`, la escala embebida d queda d' = d · external_scale[e] (clamp f16).
    /// Soporta dtypes con d f16 @ offset 0 del bloque (q8_0/q4_0/q5_0/
    /// q4_1/q5_1); los K-quants/I-quants con layouts de escala distintos
    /// devuelven error (ningún preset XL real los usa como down hoy).
    /// El caller es dueño del buffer devuelto (alloc normal del host).
    pub fn composeCanonical(self: BankRef, allocator: std.mem.Allocator) ![]u8 {
        const scales = self.external_scale orelse return error.NoExternalScales;
        std.debug.assert(scales.len >= self.n_expert);
        if (self.block_offset != 0) return error.CombinedBankUnsupported; // ventana no soportada aún
        const bs = self.blockBytes();
        const row_bytes = self.rowBytes();
        // dtype con d f16@0 por bloque de `bs_elems` elementos:
        const blk: struct { elems: usize, bytes: usize } = switch (self.dtype) {
            .q8_0 => .{ .elems = 32, .bytes = 34 },
            .q4_0 => .{ .elems = 32, .bytes = 18 },
            .q4_1 => .{ .elems = 32, .bytes = 20 }, // d@0, m@2 — solo d escala
            .q5_0 => .{ .elems = 32, .bytes = 22 },
            .q5_1 => .{ .elems = 32, .bytes = 24 },
            // iq4_nl: d f16@0 + qs[16] nibbles (18B/bloque de 32) — la LUT
            // kvalues se aplica en dequant, la escala embebida es f16@0
            // como los legacy (verificado contra dequantIq4_nl).
            .iq4_nl => .{ .elems = 32, .bytes = 18 },
            else => return error.UnsupportedExternalScaleDtype,
        };
        if (row_bytes % blk.bytes != 0) return error.RowBytesNotMultiple;
        const out = try allocator.alloc(u8, self.bytes.len);
        errdefer allocator.free(out);
        @memcpy(out, self.bytes);
        for (0..self.n_expert) |e| {
            const s: f32 = scales[e];
            if (s == 1.0) continue; // fast-path: nada que componer
            const bank = out[e * bs ..][0 .. bs - self.block_offset];
            const n_blocks = bank.len / blk.bytes;
            for (0..n_blocks) |b| {
                const base = b * blk.bytes;
                const d_bits = std.mem.readInt(u16, bank[base..][0..2], .little);
                const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
                const d2: f16 = @floatCast(d * s);
                std.mem.writeInt(u16, out[e * bs + base ..][0..2], @bitCast(d2), .little);
            }
        }
        return out;
    }
};

/// Router `ffn_gate_inp` [n_embd, n_expert]; dtype flotante pequeño residente.
pub const RouterRef = struct {
    bytes: []const u8,
    dtype: gguf.GgmlType,
    n_embd: usize,
    n_expert: usize,
};

/// Spec completa de una capa MoE: referencias prestadas al GgufFile (deben
/// vivir mientras el archivo esté mapeado). Cero copias, cero dequant.
pub const MoeLayerSpec = struct {
    layer_id: u32,
    family: MoeFamily,
    router: RouterRef,
    gate: BankRef,
    up: BankRef,
    down: BankRef,
    n_expert: u32,
    top_k: u32,
};

/// Info a nivel modelo (derivada de blk.0 + metadata {arch}.expert_*).
pub const MoeModelInfo = struct {
    family: MoeFamily,
    n_expert: u32,
    top_k: u32,
};

// ─── Gate de breadcrumbs propio (MOE_DEBUG=1) ───────────────────────────────

var g_moe_debug: ?bool = null;

fn moeDebug() bool {
    if (g_moe_debug == null) g_moe_debug = std.c.getenv("MOE_DEBUG") != null;
    return g_moe_debug.?;
}

// ─── Nombres de tensores ────────────────────────────────────────────────────

fn fmtName(buf: []u8, comptime fmt: []const u8, args: anytype) MoeError![]const u8 {
    return std.fmt.bufPrint(buf, fmt, args) catch MoeError.NameTooLong;
}

fn routerName(buf: []u8, il: usize) MoeError![]const u8 {
    return fmtName(buf, "blk.{d}.ffn_gate_inp.weight", .{il});
}

fn fusedName(buf: []u8, il: usize, comptime kind: BankKind) MoeError![]const u8 {
    return fmtName(buf, "blk.{d}.ffn_" ++ @tagName(kind) ++ "_exps.weight", .{il});
}

fn combinedName(buf: []u8, il: usize) MoeError![]const u8 {
    // Gemma-MoE (llama.cpp): gate+up empaquetados en UN banco por experto
    // [in, 2*out_ff, E]; cada bloque de experto = [filas gate | filas up].
    return fmtName(buf, "blk.{d}.ffn_gate_up_exps.weight", .{il});
}

fn splitName(buf: []u8, il: usize, comptime kind: BankKind, e: usize) MoeError![]const u8 {
    return fmtName(buf, "blk.{d}.ffn_" ++ @tagName(kind) ++ ".experts.{d}.weight", .{ il, e });
}

// ─── Metadata ───────────────────────────────────────────────────────────────

fn metaU64Prefixed(g: *const gguf.GgufFile, arch: []const u8, comptime key: []const u8) ?u64 {
    var buf: [160]u8 = undefined;
    if (std.fmt.bufPrint(&buf, "{s}." ++ key, .{arch})) |full| {
        if (g.getMeta(full)) |v| {
            if (v.asU64()) |n| return n;
        }
    } else |_| {}
    if (g.getMeta(key)) |v| {
        if (v.asU64()) |n| return n;
    }
    return null;
}

/// {arch}.expert_count (fallback: clave sin prefijo).
pub fn expertCountMeta(g: *const gguf.GgufFile, arch: []const u8) ?u64 {
    return metaU64Prefixed(g, arch, "expert_count");
}

/// {arch}.expert_used_count (fallback: clave sin prefijo).
pub fn topKMeta(g: *const gguf.GgufFile, arch: []const u8) ?u64 {
    return metaU64Prefixed(g, arch, "expert_used_count");
}

/// {arch}.block_count (fallback: clave sin prefijo).
pub fn blockCountMeta(g: *const gguf.GgufFile, arch: []const u8) ?u64 {
    return metaU64Prefixed(g, arch, "block_count");
}

// ─── Detección ──────────────────────────────────────────────────────────────

/// ¿Existe router MoE en el bloque `il`?
pub fn isMoeLayer(g: *const gguf.GgufFile, il: usize) bool {
    var buf: [96]u8 = undefined;
    const rn = routerName(&buf, il) catch return false;
    return g.getTensor(rn) != null;
}

/// ¿El modelo tiene capas MoE? Chequeo barato de presencia (blk.0).
pub fn isMoeModel(g: *const gguf.GgufFile) bool {
    return isMoeLayer(g, 0);
}

/// Info estricta del modelo MoE: familia, n_expert y top_k. Deriva n_expert
/// de metadata → forma fusionada → conteo split; top_k EXIGE metadata
/// ({arch}.expert_used_count) — nunca se adivina.
pub fn moeInfo(g: *const gguf.GgufFile) MoeError!MoeModelInfo {
    const arch = g.arch() orelse return MoeError.MissingMetadata;
    const family = MoeFamily.fromArch(arch);

    var buf: [96]u8 = undefined;
    const rn = try routerName(&buf, 0);
    const router_info = g.getTensor(rn) orelse return MoeError.RouterNotFound;

    var n_expert: usize = 0;
    if (expertCountMeta(g, arch)) |n| {
        n_expert = @intCast(n);
    } else if (g.getTensor(fusedName(&buf, 0, .gate) catch unreachable)) |fi| {
        n_expert = @intCast(fi.shape()[2]);
    } else {
        n_expert = countSplitExperts(g, 0, .gate);
    }
    if (n_expert == 0) return MoeError.ExpertsNotFound;
    if (router_info.shape()[1] != n_expert) return MoeError.InvalidRouterDims;

    const top_k_raw = topKMeta(g, arch) orelse return MoeError.MissingMetadata;
    const top_k: usize = @intCast(top_k_raw);
    if (top_k == 0 or top_k > n_expert) return MoeError.InvalidRouterDims;

    if (moeDebug()) debug.dbg.printLevel(.info, "[gguf_moe] detectado arch={s} family={s} n_expert={d} top_k={d}\n", .{ arch, @tagName(family), n_expert, top_k });

    return .{
        .family = family,
        .n_expert = @intCast(n_expert),
        .top_k = @intCast(top_k),
    };
}

// ─── Stacking por capa ──────────────────────────────────────────────────────

/// Cuenta expertos split contiguos-presentes empezando en 0 (no valida
/// contiguidad de datos; eso lo hace resolveSplitBank).
fn countSplitExperts(g: *const gguf.GgufFile, il: usize, comptime kind: BankKind) usize {
    var buf: [128]u8 = undefined;
    var n: usize = 0;
    while (n < 1024) : (n += 1) {
        const nm = splitName(&buf, il, kind, n) catch break;
        if (g.getTensor(nm) == null) break;
    }
    return n;
}

/// Resuelve un banco fusionado O apilado para la capa `il`. Fusionado tiene
/// prioridad (un solo tensor _exps); si no existe, apila `.experts.{i}`
/// exigiendo contiguidad back-to-back en el mmap.
fn resolveBank(g: *const gguf.GgufFile, il: usize, comptime kind: BankKind) MoeError!BankRef {
    var buf: [128]u8 = undefined;

    // ── Caso fusionado: ne=[in, out, E], experto mayor ⇒ bloques contiguos.
    const fname = try fusedName(&buf, il, kind);
    if (g.getTensor(fname)) |fi| {
        if (fi.n_dims != 3) return MoeError.ExpertDimMismatch;
        const sh = fi.shape();
        return .{
            .bytes = g.tensorData(fi),
            .dtype = fi.dtype,
            .in_dim = @intCast(sh[0]),
            .out_dim = @intCast(sh[1]),
            .n_expert = @intCast(sh[2]),
            .external_scale = resolveExternalScale(g, il, kind, @intCast(sh[2])),
        };
    }

    // ── Caso split: E tensores per-experto idénticos y contiguos.
    const n = countSplitExperts(g, il, kind);
    if (n == 0) return MoeError.ExpertsNotFound;
    const first_name = try splitName(&buf, il, kind, 0);
    const first = g.getTensor(first_name) orelse return MoeError.ExpertsNotFound;
    const first_bytes = g.tensorData(first);

    var total: usize = first_bytes.len;
    var prev_ptr: usize = @intFromPtr(first_bytes.ptr);
    var prev_len: usize = first_bytes.len;
    var e: usize = 1;
    while (e < n) : (e += 1) {
        const nm = try splitName(&buf, il, kind, e);
        const info = g.getTensor(nm) orelse return MoeError.ExpertsNotFound;
        if (info.dtype != first.dtype or
            info.n_dims != first.n_dims or
            !std.mem.eql(u64, info.shape(), first.shape()))
            return MoeError.ExpertDimMismatch;
        const cur = g.tensorData(info);
        const cur_ptr: usize = @intFromPtr(cur.ptr);
        if (cur_ptr != prev_ptr + prev_len) return MoeError.ExpertsNotContiguous;
        prev_ptr = cur_ptr;
        prev_len = cur.len;
        total += cur.len;
    }

    const sh = first.shape();
    return .{
        .bytes = first_bytes.ptr[0..total],
        .dtype = first.dtype,
        .in_dim = @intCast(sh[0]),
        .out_dim = @intCast(sh[1]),
        .n_expert = n,
        .external_scale = resolveExternalScale(g, il, kind, n),
    };
}

/// 4.3' copy-once: compone las escalas EXTERNAS de TODOS los bancos con
/// .scale IN-PLACE sobre el buffer pinned del caller (una única pasada
/// global — muta cada d f16 de bloque por scale[e]; el buffer queda
/// totalmente canónico para el gather/kernels). `buf` es el copy-once
/// (mismo layout que g.data). Devuelve el nº de bancos compuestos.
pub fn composeExternalScalesInPlace(g: *const gguf.GgufFile, buf: []u8) usize {
    const arch = g.arch() orelse return 0;
    const bc = blockCountMeta(g, arch) orelse return 0;
    var n_composed: usize = 0;
    for (0..bc) |il| {
        if (!isMoeLayer(g, il)) continue;
        const spec = layerSpec(g, il) catch continue;
        const banks = [_]BankRef{ spec.gate, spec.up, spec.down };
        for (banks) |bank| {
            const scales = bank.external_scale orelse continue;
            const blk: struct { elems: usize, bytes: usize } = switch (bank.dtype) {
                .q8_0 => .{ .elems = 32, .bytes = 34 },
                .q4_0, .iq4_nl => .{ .elems = 32, .bytes = 18 },
                .q4_1 => .{ .elems = 32, .bytes = 20 },
                .q5_0 => .{ .elems = 32, .bytes = 22 },
                .q5_1 => .{ .elems = 32, .bytes = 24 },
                else => continue, // dtype sin compose: el banco sigue NO canónico
            };
            const off_in_buf = @intFromPtr(bank.bytes.ptr) - @intFromPtr(g.data.ptr);
            if (off_in_buf + bank.bytes.len > buf.len) continue; // banco fuera del buffer (¿modelo distinto?)
            const bank_buf = buf[off_in_buf..][0..bank.bytes.len];
            const bs = bank.blockBytes();
            for (0..bank.n_expert) |e| {
                const sc: f32 = scales[e];
                if (sc == 1.0) continue;
                const eb = bank_buf[e * bs ..];
                const n_blocks = eb.len / blk.bytes;
                for (0..n_blocks) |b| {
                    const base = b * blk.bytes;
                    const d_bits = std.mem.readInt(u16, eb[base..][0..2], .little);
                    const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
                    const d2: f16 = @floatCast(d * sc);
                    std.mem.writeInt(u16, eb[base..][0..2], @bitCast(d2), .little);
                }
            }
            n_composed += 1;
        }
    }
    return n_composed;
}

/// Ticket 4.4: busca `blk.{il}.ffn_{k}_exps.scale` [n_expert] f32 y lo
/// devuelve como slice f32 prestado del mmap. Null si no existe (banco
/// canónico inline). Solo f32 [n_expert] es válido — cualquier otra shape
/// se ignora con breadcrumb (preset corrupto no debe tumbar la carga).
fn resolveExternalScale(g: *const gguf.GgufFile, il: usize, comptime kind: BankKind, n_expert: usize) ?[]const f32 {
    var buf: [128]u8 = undefined;
    const sname = std.fmt.bufPrint(&buf, "blk.{d}.ffn_{s}_exps.scale", .{ il, @tagName(kind) }) catch return null;
    const si = g.getTensor(sname) orelse return null;
    if (si.dtype != .f32 or si.n_dims != 1) {
        if (moeDebug()) debug.dbg.printLevel(.info, "[gguf_moe] {s}: dtype/dims inesperado (f32 [n] esperado) — ignorado\n", .{sname});
        return null;
    }
    if (si.shape()[0] != n_expert) {
        if (moeDebug()) debug.dbg.printLevel(.info, "[gguf_moe] {s}: len={d} != n_expert={d} — ignorado\n", .{ sname, si.shape()[0], n_expert });
        return null;
    }
    const raw = g.tensorData(si);
    const n_f32 = raw.len / @sizeOf(f32);
    if (n_f32 != n_expert) return null;
    return @as([*]const f32, @ptrCast(@alignCast(raw.ptr)))[0..n_f32];
}

/// Resuelve el par gate/up: (a) banco combinado `ffn_gate_up_exps`
/// (ventanas por mitad del bloque de cada experto — convención llama.cpp
/// [gate_rows | up_rows]), o (b) bancos separados vía resolveBank.
fn resolveGateUp(g: *const gguf.GgufFile, il: usize) MoeError![2]BankRef {
    var buf: [128]u8 = undefined;
    const cname = try combinedName(&buf, il);
    if (g.getTensor(cname)) |ci| {
        if (ci.n_dims != 3) return MoeError.ExpertDimMismatch;
        const sh = ci.shape();
        const out_total: usize = @intCast(sh[1]);
        if (out_total % 2 != 0) return MoeError.ExpertDimMismatch;
        const half: usize = out_total / 2;
        const base_bytes = g.tensorData(ci);
        const mk = struct {
            fn view(full: []const u8, in_d: usize, out_d: usize, e_count: usize, stride: usize, off: usize, dtype: gguf.GgmlType) BankRef {
                return .{
                    .bytes = full,
                    .dtype = dtype,
                    .in_dim = in_d,
                    .out_dim = out_d,
                    .n_expert = e_count,
                    .block_stride = stride,
                    .block_offset = off,
                    .region_len = stride / 2,
                };
            }
        };
        const per_expert: usize = @intCast(base_bytes.len / sh[2]);
        return .{
            mk.view(base_bytes, @intCast(sh[0]), half, @intCast(sh[2]), per_expert, 0, ci.dtype),
            mk.view(base_bytes, @intCast(sh[0]), half, @intCast(sh[2]), per_expert, per_expert / 2, ci.dtype),
        };
    }
    return .{ try resolveBank(g, il, .gate), try resolveBank(g, il, .up) };
}

/// Spec completa de la capa MoE `il`. Función pura: todas las referencias
/// apuntan al mmap del GgufFile pasado.
pub fn layerSpec(g: *const gguf.GgufFile, il: usize) MoeError!MoeLayerSpec {
    var buf: [96]u8 = undefined;
    const rn = try routerName(&buf, il);
    const rinfo = g.getTensor(rn) orelse return MoeError.RouterNotFound;
    if (rinfo.n_dims != 2) return MoeError.InvalidRouterDims;
    const rsh = rinfo.shape();
    switch (rinfo.dtype) {
        .f32, .f16, .bf16 => {},
        else => return MoeError.UnsupportedRouterDtype,
    }

    const gu = try resolveGateUp(g, il);
    const gate = gu[0];
    const up = gu[1];
    const down = try resolveBank(g, il, .down);

    if (gate.n_expert != up.n_expert or gate.n_expert != down.n_expert)
        return MoeError.ExpertDimMismatch;
    // gate/up: [n_embd → ffn]; down: [ffn → n_embd]
    if (gate.in_dim != up.in_dim or gate.out_dim != up.out_dim)
        return MoeError.ExpertDimMismatch;
    if (down.in_dim != gate.out_dim) return MoeError.ExpertDimMismatch;

    const info = try moeInfo(g);
    if (rsh[1] != gate.n_expert or rsh[1] != info.n_expert)
        return MoeError.InvalidRouterDims;

    if (moeDebug()) debug.dbg.printLevel(.detail, "[gguf_moe] capa {d}: E={d} top_k={d} gate[{d},{d}] {s} row_bytes(gate/up/down)={d}/{d}/{d}\n", .{
        il,                gate.n_expert,   info.top_k,    gate.out_dim,    gate.in_dim,
        gate.dtype.name(), gate.rowBytes(), up.rowBytes(), down.rowBytes(),
    });

    return .{
        .layer_id = @intCast(il),
        .family = info.family,
        .router = .{
            .bytes = g.tensorData(rinfo),
            .dtype = rinfo.dtype,
            .n_embd = @intCast(rsh[0]),
            .n_expert = @intCast(rsh[1]),
        },
        .gate = gate,
        .up = up,
        .down = down,
        .n_expert = @intCast(gate.n_expert),
        .top_k = info.top_k,
    };
}
