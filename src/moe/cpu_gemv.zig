//! GEMV cuantizado CPU sobre bytes GGUF crudos (Lane F, tarea F1).
//!
//! Memory-bound: lee W cuantizado (Q4_0/Q8_0) y lo multiplica por activaciones
//! f32 sin materializar la fila dequantizada. Tablas/semántica de dequant PROPIAS
//! (fiel a ggml/llama.cpp; verificado contra el oráculo `gguf.dequantBlock`):
//!   Q4_0: bloque 32 elems / 18 bytes — d:f16 LE, qs[16]; elem j = nibble bajo
//!         de qs[j], elem j+16 = nibble alto; val = d * (nibble − 8).
//!   Q8_0: bloque 32 elems / 34 bytes — d:f16 LE, qs[32] i8; val = d * qs[i].
//!
//! Productos por elemento BIT-IDÉNTICOS al oráculo ((d*q)*x); solo difiere el
//! ORDEN de reducción (acumulación por carriles SIMD + árbol fijo) → paridad
//! exacta en las rutas escalares, epsilon documentado en la ruta vectorizada.
//! Determinismo híbrido: el paralelismo (F2) reparte por FILAS completas,
//! jamás dentro de una fila ⇒ misma salida con cualquier nº de workers.
//!
//! Dispatch ISA runtime: ZIG_AI_CPU_GEMV_ISA=scalar|auto (default auto; scalar
//! solo para depuración/comparativa).
const std = @import("std");
const debugz = @import("debug");
const iq_grids = @import("kv_cache").iq_grids;

pub const block_elems: usize = 32;
pub const q4_0_bytes: usize = 18;
pub const q8_0_bytes: usize = 34;
pub const q4_1_bytes: usize = 20; // d:f16 @0, m:f16 @2, qs[16] @4
pub const q4_k_bytes: usize = 144; // d:f16 @0, min:f16 @2, scales[12] @4, qs[128] @16
pub const q5_k_bytes: usize = 176; // igual a q4_K + qh[32] @16 (bits altos), qs[128] @48
pub const q3_k_bytes: usize = 110; // hmask[32] @0, qs[64] @32, scales[12] @96, d:f16 @108
pub const q2_k_bytes: usize = 84; // scales[16] @0 (4+4 bits), qs[64] @16, d:f16 @80, dmin:f16 @82
pub const iq3_s_bytes: usize = 110; // d@0, qs[64] @2, qh[8] @66, signs[32] @74, scales[4] @106
pub const iq4_nl_bytes: usize = 18; // d:f16 @0, qs[16] nibbles @2 (32 elems)
pub const iq2_xxs_bytes: usize = 66; // d:f16 @0, qs[32] @2 (8 sub-bloques × 8B)

// Q6_K: súper-bloques de 256 elems / 210 bytes.
pub const q6_k_block_elems: usize = 256;
pub const q6_k_block_bytes: usize = 210;

/// Formatos soportados por el GEMV CPU (enum propio: cero dependencias).
pub const Format = enum(u4) {
    q4_0,
    q8_0,
    q6_k,
    q4_1,
    q4_k,
    q5_k,
    q3_k,
    q2_k,
    iq3_s,
    iq2_s,
    iq4_nl,
    iq2_xxs,

    pub fn rowBytes(self: Format, n_elems: usize) usize {
        return switch (self) {
            .q4_0 => (n_elems / block_elems) * q4_0_bytes,
            .q8_0 => (n_elems / block_elems) * q8_0_bytes,
            .q6_k => (n_elems / q6_k_block_elems) * q6_k_block_bytes,
            .q4_1 => (n_elems / block_elems) * q4_1_bytes,
            .q4_k => (n_elems / q6_k_block_elems) * q4_k_bytes,
            .q5_k => (n_elems / q6_k_block_elems) * q5_k_bytes,
            .q3_k => (n_elems / q6_k_block_elems) * q3_k_bytes,
            .q2_k => (n_elems / q6_k_block_elems) * q2_k_bytes,
            .iq3_s => (n_elems / q6_k_block_elems) * iq3_s_bytes,
            .iq2_s => (n_elems / q6_k_block_elems) * 82, // IQ2_S super-block 256 = 82 bytes
            .iq4_nl => (n_elems / block_elems) * iq4_nl_bytes,
            .iq2_xxs => (n_elems / q6_k_block_elems) * iq2_xxs_bytes,
        };
    }

    /// Granularidad de elems exigida a las filas para este formato.
    pub fn rowAlign(self: Format) usize {
        return switch (self) {
            .q4_0, .q8_0, .q4_1, .iq4_nl => block_elems,
            .q6_k, .q4_k, .q5_k, .q3_k, .q2_k, .iq3_s, .iq2_s, .iq2_xxs => q6_k_block_elems,
        };
    }
};

pub const IsaMode = enum { auto, scalar };

fn parseIsa(v: ?[*:0]const u8) IsaMode {
    const s = v orelse return .auto;
    if (std.mem.eql(u8, std.mem.span(s), "scalar")) return .scalar;
    return .auto;
}

var isa_cached: ?IsaMode = null;

/// Resuelve ZIG_AI_CPU_GEMV_ISA una sola vez (breadcrumb info gated).
pub fn resolveIsa() IsaMode {
    if (isa_cached) |m| return m;
    const m = parseIsa(std.c.getenv("ZIG_AI_CPU_GEMV_ISA"));
    isa_cached = m;
    debugz.dbg.printLevel(.info, "[cpu_gemv] isa={s}\n", .{@tagName(m)});
    return m;
}

// ============================================================================
// Dequant vectorizado (paridad bit-exacta vs gguf.dequantBlock; soporta
// bloque final parcial como el oráculo).
// ============================================================================

pub fn dequantQ4_0(bytes: []const u8, out: []f32) void {
    var i: usize = 0;
    var b: usize = 0;
    while (i + block_elems <= out.len) : ({
        i += block_elems;
        b += q4_0_bytes;
    }) {
        const d_bits = std.mem.readInt(u16, bytes[b..][0..2], .little);
        const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
        const qs: *const [16]u8 = bytes[b + 2 ..][0..16];
        const qv: @Vector(16, u8) = @bitCast(qs.*);
        const lo8: @Vector(16, u8) = qv & @as(@Vector(16, u8), @splat(0x0F));
        const hi8: @Vector(16, u8) = qv >> @as(@Vector(16, u3), @splat(4));
        const e: @Vector(16, i16) = @splat(8);
        const lo_q: @Vector(16, f32) = @floatFromInt(@as(@Vector(16, i16), @intCast(lo8)) - e);
        const hi_q: @Vector(16, f32) = @floatFromInt(@as(@Vector(16, i16), @intCast(hi8)) - e);
        const ds: @Vector(16, f32) = @splat(d);
        (out[i..][0..16].*) = @bitCast(ds * lo_q);
        (out[i + 16 ..][0..16].*) = @bitCast(ds * hi_q);
    }
    // Bloque final parcial (mismo orden condicional que el oráculo).
    if (i < out.len) {
        const n = out.len - i;
        const d_bits = std.mem.readInt(u16, bytes[b..][0..2], .little);
        const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
        const qs = bytes[b + 2 ..];
        const half = @min(block_elems / 2, n);
        for (0..half) |j| {
            const lo: i8 = @as(i8, @intCast(qs[j] & 0x0F)) - 8;
            const hi: i8 = @as(i8, @intCast(qs[j] >> 4)) - 8;
            out[i + j] = d * @as(f32, @floatFromInt(lo));
            if (j + 16 < n) out[i + j + 16] = d * @as(f32, @floatFromInt(hi));
        }
    }
}

pub fn dequantQ8_0(bytes: []const u8, out: []f32) void {
    var i: usize = 0;
    var b: usize = 0;
    while (i + block_elems <= out.len) : ({
        i += block_elems;
        b += q8_0_bytes;
    }) {
        const d_bits = std.mem.readInt(u16, bytes[b..][0..2], .little);
        const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
        const ds: @Vector(16, f32) = @splat(d);
        inline for (0..2) |h| {
            const qs: *const [16]u8 = bytes[b + 2 + h * 16 ..][0..16];
            const qi: @Vector(16, i16) = @intCast(@as(@Vector(16, i8), @bitCast(qs.*)));
            (out[i + h * 16 ..][0..16].*) = @bitCast(ds * @as(@Vector(16, f32), @floatFromInt(qi)));
        }
    }
    if (i < out.len) {
        const n = out.len - i;
        const d_bits = std.mem.readInt(u16, bytes[b..][0..2], .little);
        const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
        const qs = bytes[b + 2 ..];
        for (0..n) |j| {
            out[i + j] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(qs[j]))));
        }
    }
}

/// Dequant de un bloque completo (n elems, puede ser parcial) según formato.
pub fn dequantRow(fmt: Format, bytes: []const u8, out: []f32) void {
    switch (fmt) {
        .q4_0 => dequantQ4_0(bytes, out),
        .q8_0 => dequantQ8_0(bytes, out),
        .q6_k => dequantQ6_K(bytes, out),
        .q4_1 => dequantQ4_1(bytes, out),
        .q4_k => dequantQ4_K(bytes, out),
        .q5_k => dequantQ5_K(bytes, out),
        .q3_k => dequantQ3_K(bytes, out),
        .q2_k => dequantQ2_K(bytes, out),
        .iq3_s => dequantIq3_s(bytes, out),
        .iq2_s => dequantIq2_S(bytes, out),
        .iq4_nl => dequantIq4_nl(bytes, out),
        .iq2_xxs => dequantIq2_xxs(bytes, out),
    }
}

// ============================================================================
// Q6_K — bloque 256 elems / 210 bytes: ql[128] @0 (nibbles), qh[64] @128
// (bits altos ×2), sc[16] i8 @192, d f16 LE @208. Escalas sc2[is+{0,2,4,6}]
// con is=l/16 por grupos de 32 (convención verificada vs upstream master,
// HANDOFFS lane-a 16:40). Producto con asociación del ORÁCULO: (d·sc)·q.
// ============================================================================

/// Procesa un chunk de 16 l's de un medio-bloque: produce los 4 vectores de
/// productos parciales (sin x; solo dequant) hacia out en +0,+32,+64,+96.
inline fn q6kChunk16(
    comptime l0: usize,
    ql2: *const [64]u8,
    qh2: *const [32]u8,
    sc2: *const [8]u8,
    ds: @Vector(16, f32),
    out: []f32,
    off: usize,
) void {
    const V = @Vector(16, f32);
    const Vi = @Vector(16, i16);
    const e32: Vi = @splat(32);
    const nib_lo: @Vector(16, u8) = ql2[l0..][0..16].*;
    const nib_hi: @Vector(16, u8) = ql2[l0 + 32 ..][0..16].*;
    const hb: @Vector(16, u8) = qh2[l0..][0..16].*;

    const m0: Vi = @intCast((nib_lo & @as(@Vector(16, u8), @splat(0x0F))) | ((hb >> @as(@Vector(16, u3), @splat(0))) & @as(@Vector(16, u8), @splat(3))) << @as(@Vector(16, u3), @splat(4)));
    const m1: Vi = @intCast((nib_hi & @as(@Vector(16, u8), @splat(0x0F))) | ((hb >> @as(@Vector(16, u3), @splat(2))) & @as(@Vector(16, u8), @splat(3))) << @as(@Vector(16, u3), @splat(4)));
    const m2: Vi = @intCast((nib_lo >> @as(@Vector(16, u3), @splat(4))) | ((hb >> @as(@Vector(16, u3), @splat(4))) & @as(@Vector(16, u8), @splat(3))) << @as(@Vector(16, u3), @splat(4)));
    const m3: Vi = @intCast((nib_hi >> @as(@Vector(16, u3), @splat(4))) | ((hb >> @as(@Vector(16, u3), @splat(6))) & @as(@Vector(16, u8), @splat(3))) << @as(@Vector(16, u3), @splat(4)));

    const is = l0 / 16;
    const s0: V = @splat(ds[0] * @as(f32, @floatFromInt(@as(i8, @bitCast(sc2[is + 0])))));
    const s1: V = @splat(ds[0] * @as(f32, @floatFromInt(@as(i8, @bitCast(sc2[is + 2])))));
    const s2: V = @splat(ds[0] * @as(f32, @floatFromInt(@as(i8, @bitCast(sc2[is + 4])))));
    const s3: V = @splat(ds[0] * @as(f32, @floatFromInt(@as(i8, @bitCast(sc2[is + 6])))));

    (out[off..][0..16].*) = @bitCast(s0 * @as(V, @floatFromInt(m0 - e32)));
    (out[off + 32 ..][0..16].*) = @bitCast(s1 * @as(V, @floatFromInt(m1 - e32)));
    (out[off + 64 ..][0..16].*) = @bitCast(s2 * @as(V, @floatFromInt(m2 - e32)));
    (out[off + 96 ..][0..16].*) = @bitCast(s3 * @as(V, @floatFromInt(m3 - e32)));
}

pub fn dequantQ6_K(bytes: []const u8, out: []f32) void {
    var i: usize = 0;
    var nb: usize = 0;
    while (i + q6_k_block_elems <= out.len) : ({
        i += q6_k_block_elems;
        nb += 1;
    }) {
        const base = nb * q6_k_block_bytes;
        const d_bits = std.mem.readInt(u16, bytes[base + 208 ..][0..2], .little);
        const ds: @Vector(16, f32) = @splat(@floatCast(@as(f16, @bitCast(d_bits))));
        inline for (0..2) |h| {
            const ql2: *const [64]u8 = bytes[base + h * 64 ..][0..64];
            const qh2: *const [32]u8 = bytes[base + 128 + h * 32 ..][0..32];
            const sc2: *const [8]u8 = bytes[base + 192 + h * 8 ..][0..8];
            const off = i + h * 128;
            inline for (0..2) |lc| {
                q6kChunk16(lc * 16, ql2, qh2, sc2, ds, out, off + lc * 16);
            }
        }
    }
    // Cola parcial defensiva (el oráculo gguf.zig ESCRIBE FUERA DE RANGO en
    // este caso; aquí se recorta a n elems con las mismas fórmulas).
    if (i < out.len) {
        const remaining = out.len - i;
        const base = nb * q6_k_block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base + 208 ..][0..2], .little))));
        const ql = bytes[base .. base + 128];
        const qh = bytes[base + 128 .. base + 192];
        const sc = bytes[base + 192 .. base + 208];
        var n: usize = 0;
        while (n < @min(q6_k_block_elems, remaining)) : (n += 128) {
            const ql2 = ql[(n / 128) * 64 ..];
            const qh2 = qh[(n / 128) * 32 ..];
            const sc2 = sc[(n / 128) * 8 ..];
            for (0..32) |l| {
                const is = l / 16;
                if (n + l < remaining) {
                    const q1: f32 = @floatFromInt(@as(i8, @bitCast((ql2[l] & 0xF) | (((qh2[l] >> 0) & 3) << 4))) - 32);
                    out[i + n + l] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc2[is + 0])))) * q1;
                }
                if (n + l + 32 < remaining) {
                    const q2: f32 = @floatFromInt(@as(i8, @bitCast((ql2[l + 32] & 0xF) | (((qh2[l] >> 2) & 3) << 4))) - 32);
                    out[i + n + l + 32] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc2[is + 2])))) * q2;
                }
                if (n + l + 64 < remaining) {
                    const q3: f32 = @floatFromInt(@as(i8, @bitCast((ql2[l] >> 4) | (((qh2[l] >> 4) & 3) << 4))) - 32);
                    out[i + n + l + 64] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc2[is + 4])))) * q3;
                }
                if (n + l + 96 < remaining) {
                    const q4: f32 = @floatFromInt(@as(i8, @bitCast((ql2[l + 32] >> 4) | (((qh2[l] >> 6) & 3) << 4))) - 32);
                    out[i + n + l + 96] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc2[is + 6])))) * q4;
                }
            }
        }
    }
}

// ============================================================================
// Rutas escalares de referencia (bit-exactas vs dequant-oracle + dot secuencial).
// Requieren n % 32 == 0 (filas GGUF reales siempre lo cumplen).
// ============================================================================

pub fn dotQ4_0Scalar(w: []const u8, x: []const f32) f32 {
    std.debug.assert(x.len % block_elems == 0);
    var acc: f32 = 0;
    var i: usize = 0;
    var b: usize = 0;
    while (i < x.len) : ({
        i += block_elems;
        b += q4_0_bytes;
    }) {
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w[b..][0..2], .little))));
        const qs = w[b + 2 ..];
        // Orden de MEMORIA (lo0..lo15, hi0..hi15) = orden de la referencia
        // secuencial sobre la fila dequantizada ⇒ bit-exacto.
        for (0..16) |j| {
            const lo: i8 = @as(i8, @intCast(qs[j] & 0x0F)) - 8;
            const p = (d * @as(f32, @floatFromInt(lo))) * x[i + j];
            acc += p;
        }
        for (0..16) |j| {
            const hi: i8 = @as(i8, @intCast(qs[j] >> 4)) - 8;
            const p = (d * @as(f32, @floatFromInt(hi))) * x[i + j + 16];
            acc += p;
        }
    }
    return acc;
}

// ============================================================================
// Q4_1 — bloque 32 elems / 20 bytes: d:f16 LE @0, m:f16 LE @2, qs[16] @4.
// Layout split como Q4_0 (bajo→elem j, alto→j+16); val = d·q + m.
// ============================================================================

pub fn dequantQ4_1(bytes: []const u8, out: []f32) void {
    var i: usize = 0;
    var b: usize = 0;
    while (i + block_elems <= out.len) : ({
        i += block_elems;
        b += q4_1_bytes;
    }) {
        const d_bits = std.mem.readInt(u16, bytes[b..][0..2], .little);
        const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
        const m_bits = std.mem.readInt(u16, bytes[b + 2 ..][0..2], .little);
        const m: f32 = @floatCast(@as(f16, @bitCast(m_bits)));
        const qs: *const [16]u8 = bytes[b + 4 ..][0..16];
        const qv: @Vector(16, u8) = qs.*;
        const lo8: @Vector(16, u8) = qv & @as(@Vector(16, u8), @splat(0x0F));
        const hi8: @Vector(16, u8) = qv >> @as(@Vector(16, u3), @splat(4));
        const lo_f: @Vector(16, f32) = @floatFromInt(@as(@Vector(16, i16), @intCast(lo8)));
        const hi_f: @Vector(16, f32) = @floatFromInt(@as(@Vector(16, i16), @intCast(hi8)));
        const dv: @Vector(16, f32) = @splat(d);
        const mv: @Vector(16, f32) = @splat(m);
        (out[i..][0..16].*) = @bitCast(dv * lo_f + mv);
        (out[i + 16 ..][0..16].*) = @bitCast(dv * hi_f + mv);
    }
    if (i < out.len) {
        const n = out.len - i;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[b..][0..2], .little))));
        const m: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[b + 2 ..][0..2], .little))));
        const qs = bytes[b + 4 ..];
        const half = @min(block_elems / 2, n);
        for (0..half) |j| {
            out[i + j] = d * @as(f32, @floatFromInt(@as(i32, qs[j] & 0x0F))) + m;
            if (j + 16 < n) out[i + j + 16] = d * @as(f32, @floatFromInt(@as(i32, qs[j] >> 4))) + m;
        }
    }
}

/// Escalar de referencia bit-exacto: orden MEMORIA (lo0..15, hi0..16+) y
/// asociación del oráculo ((d·q)+m).
pub fn dotQ4_1Scalar(w: []const u8, x: []const f32) f32 {
    std.debug.assert(x.len % block_elems == 0);
    var acc: f32 = 0;
    var i: usize = 0;
    var b: usize = 0;
    while (i < x.len) : ({
        i += block_elems;
        b += q4_1_bytes;
    }) {
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w[b..][0..2], .little))));
        const m: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w[b + 2 ..][0..2], .little))));
        const qs = w[b + 4 ..];
        for (0..16) |j| {
            const vlo = (d * @as(f32, @floatFromInt(@as(i32, qs[j] & 0x0F)))) + m;
            const p1 = vlo * x[i + j];
            acc += p1;
        }
        for (0..16) |j| {
            const vhi = (d * @as(f32, @floatFromInt(@as(i32, qs[j] >> 4)))) + m;
            const p2 = vhi * x[i + j + 16];
            acc += p2;
        }
    }
    return acc;
}

/// Vectorizado: val = d·q + m por carriles; 4 cadenas independientes.
pub fn dotQ4_1Simd(w: []const u8, x: []const f32) f32 {
    std.debug.assert(x.len % block_elems == 0);
    const V = @Vector(16, f32);
    var a0: V = @splat(0.0);
    var a1: V = @splat(0.0);
    var a2: V = @splat(0.0);
    var a3: V = @splat(0.0);
    var i: usize = 0;
    var b: usize = 0;
    while (i + 64 <= x.len) : ({
        i += 64;
        b += 2 * q4_1_bytes;
    }) {
        inline for (0..2) |p| {
            const blk = b + p * q4_1_bytes;
            const base = i + p * 32;
            const d_bits = std.mem.readInt(u16, w[blk..][0..2], .little);
            const m_bits = std.mem.readInt(u16, w[blk + 2 ..][0..2], .little);
            const dv: V = @splat(@floatCast(@as(f16, @bitCast(d_bits))));
            const mv: V = @splat(@floatCast(@as(f16, @bitCast(m_bits))));
            const qv: @Vector(16, u8) = w[blk + 4 ..][0..16].*;
            const lo_f: V = @floatFromInt(@as(@Vector(16, i16), @intCast(qv & @as(@Vector(16, u8), @splat(0x0F)))));
            const hi_f: V = @floatFromInt(@as(@Vector(16, i16), @intCast(qv >> @as(@Vector(16, u3), @splat(4)))));
            if (p == 0) {
                a0 += (dv * lo_f + mv) * @as(V, @bitCast(x[base..][0..16].*));
                a1 += (dv * hi_f + mv) * @as(V, @bitCast(x[base + 16 ..][0..16].*));
            } else {
                a2 += (dv * lo_f + mv) * @as(V, @bitCast(x[base..][0..16].*));
                a3 += (dv * hi_f + mv) * @as(V, @bitCast(x[base + 16 ..][0..16].*));
            }
        }
    }
    if (i < x.len) {
        // Bloque final impar con el mismo orden por carriles.
        const d_bits = std.mem.readInt(u16, w[b..][0..2], .little);
        const m_bits = std.mem.readInt(u16, w[b + 2 ..][0..2], .little);
        const dv: V = @splat(@floatCast(@as(f16, @bitCast(d_bits))));
        const mv: V = @splat(@floatCast(@as(f16, @bitCast(m_bits))));
        const qv: @Vector(16, u8) = w[b + 4 ..][0..16].*;
        const lo_f: V = @floatFromInt(@as(@Vector(16, i16), @intCast(qv & @as(@Vector(16, u8), @splat(0x0F)))));
        const hi_f: V = @floatFromInt(@as(@Vector(16, i16), @intCast(qv >> @as(@Vector(16, u3), @splat(4)))));
        a0 += (dv * lo_f + mv) * @as(V, @bitCast(x[i..][0..16].*));
        a1 += (dv * hi_f + mv) * @as(V, @bitCast(x[i + 16 ..][0..16].*));
    }
    return (@reduce(.Add, a0) + @reduce(.Add, a1)) + (@reduce(.Add, a2) + @reduce(.Add, a3));
}

/// Par de filas Q4_1 fusionado (comparte cargas de x entre ambas filas).
pub fn dotQ4_1Pair(w0: []const u8, w1: []const u8, x: []const f32, o0: *f32, o1: *f32) void {
    std.debug.assert(x.len % block_elems == 0);
    const V = @Vector(16, f32);
    var r0: V = @splat(0.0);
    var r1: V = @splat(0.0);
    var s0: V = @splat(0.0);
    var s1: V = @splat(0.0);
    var i: usize = 0;
    var b: usize = 0;
    while (i < x.len) : ({
        i += block_elems;
        b += q4_1_bytes; // CADA fila avanza SU propio bloque k por iteración
    }) {
        inline for (0..2) |row| {
            const wb = if (row == 0) w0 else w1;
            const blk = b; // índice de bloque DENTRO del slice de esa fila
            const d_bits = std.mem.readInt(u16, wb[blk..][0..2], .little);
            const m_bits = std.mem.readInt(u16, wb[blk + 2 ..][0..2], .little);
            const dv: V = @splat(@floatCast(@as(f16, @bitCast(d_bits))));
            const mv: V = @splat(@floatCast(@as(f16, @bitCast(m_bits))));
            const qv: @Vector(16, u8) = wb[blk + 4 ..][0..16].*;
            const lo_f: V = @floatFromInt(@as(@Vector(16, i16), @intCast(qv & @as(@Vector(16, u8), @splat(0x0F)))));
            const hi_f: V = @floatFromInt(@as(@Vector(16, i16), @intCast(qv >> @as(@Vector(16, u3), @splat(4)))));
            const x0: V = @bitCast(x[i..][0..16].*);
            const x1: V = @bitCast(x[i + 16 ..][0..16].*);
            if (row == 0) {
                r0 += (dv * lo_f + mv) * x0;
                r1 += (dv * hi_f + mv) * x1;
            } else {
                s0 += (dv * lo_f + mv) * x0;
                s1 += (dv * hi_f + mv) * x1;
            }
        }
    }
    o0.* = @reduce(.Add, r0) + @reduce(.Add, r1);
    o1.* = @reduce(.Add, s0) + @reduce(.Add, s1);
}

// ============================================================================
// Q4_K — súper-bloque 256 elems / 144 bytes: d:f16 @0, min:f16 @2,
// scales[12] @4 (escalas y mins de 6 bits para 8 subgrupos de 32),
// qs[128] @16. val = (d·sc_d)·q − (m·sc_m); q nibble bajo/alto por mitad
// del subgrupo. Escalas vía getScaleMinK4 (idéntico al oráculo).
// ============================================================================

const ScaleMin = struct { d: i32, m: i32 };

inline fn getScaleMinK4(j: usize, q: *const [12]u8) ScaleMin {
    if (j < 4) {
        return .{ .d = @intCast(q[j] & 63), .m = @intCast(q[j + 4] & 63) };
    }
    return .{
        .d = @as(i32, @intCast(q[j + 4] & 0xF)) | (@as(i32, @intCast(q[j - 4] >> 6)) << 4),
        .m = @as(i32, @intCast(q[j + 4] >> 4)) | (@as(i32, @intCast(q[j] >> 6)) << 4),
    };
}

/// Dequant vectorizada por subgrupos de 32 (nibbles lo/hi); colas parciales
/// recortadas igual que el oráculo (putTail).
pub fn dequantQ4_K(bytes: []const u8, out: []f32) void {
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : ({
        i += q6_k_block_elems;
        nb += 1;
    }) {
        const nv = @min(q6_k_block_elems, out.len - i);
        const base = nb * q4_k_bytes;
        const d_bits = std.mem.readInt(u16, bytes[base..][0..2], .little);
        const mn_bits = std.mem.readInt(u16, bytes[base + 2 ..][0..2], .little);
        const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
        const mn: f32 = @floatCast(@as(f16, @bitCast(mn_bits)));
        const sc: *const [12]u8 = bytes[base + 4 ..][0..12];
        const qs: *const [128]u8 = bytes[base + 16 ..][0..128];

        inline for (0..4) |g| {
            const joff = g * 64;
            const s_lo = getScaleMinK4(g * 2, sc);
            const s_hi = getScaleMinK4(g * 2 + 1, sc);
            const d1: @Vector(16, f32) = @splat(d * @as(f32, @floatFromInt(s_lo.d)));
            const m1: @Vector(16, f32) = @splat(mn * @as(f32, @floatFromInt(s_lo.m)));
            const d2: @Vector(16, f32) = @splat(d * @as(f32, @floatFromInt(s_hi.d)));
            const m2: @Vector(16, f32) = @splat(mn * @as(f32, @floatFromInt(s_hi.m)));
            const qb: *const [32]u8 = qs[g * 32 ..][0..32];
            inline for (0..2) |hc| {
                const hv: @Vector(16, u8) = qb[hc * 16 ..][0..16].*;
                const lo_f: @Vector(16, f32) = @floatFromInt(@as(@Vector(16, i16), @intCast(hv & @as(@Vector(16, u8), @splat(0x0F)))));
                const hi_f: @Vector(16, f32) = @floatFromInt(@as(@Vector(16, i16), @intCast(hv >> @as(@Vector(16, u3), @splat(4)))));
                const o0 = i + joff + hc * 16;
                const o1 = i + joff + 32 + hc * 16;
                const v_lo: [16]f32 = @bitCast(d1 * lo_f - m1);
                const v_hi: [16]f32 = @bitCast(d2 * hi_f - m2);
                if (o0 + 16 <= i + nv) {
                    (out[o0..][0..16].*) = v_lo;
                } else if (o0 < i + nv) {
                    for (0..i + nv - o0) |k| out[o0 + k] = v_lo[k];
                }
                if (o1 + 16 <= i + nv) {
                    (out[o1..][0..16].*) = v_hi;
                } else if (o1 < i + nv) {
                    for (0..i + nv - o1) |k| out[o1 + k] = v_hi[k];
                }
            }
        }
    }
}

/// Escalar de referencia bit-exacto: orden MEMORIA (los 32 lo del grupo,
/// luego los 32 hi) y asociación del oráculo ((d·sc)·q − (m·sc)).
pub fn dotQ4_KScalar(w: []const u8, x: []const f32) f32 {
    std.debug.assert(x.len % q6_k_block_elems == 0);
    var acc: f32 = 0;
    var i: usize = 0;
    var b: usize = 0;
    while (i < x.len) : ({
        i += q6_k_block_elems;
        b += q4_k_bytes;
    }) {
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w[b..][0..2], .little))));
        const mn: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w[b + 2 ..][0..2], .little))));
        const sc: *const [12]u8 = w[b + 4 ..][0..12];
        const qs: *const [128]u8 = w[b + 16 ..][0..128];
        inline for (0..4) |g| {
            const joff = g * 64;
            const s_lo = getScaleMinK4(g * 2, sc);
            const s_hi = getScaleMinK4(g * 2 + 1, sc);
            const d1 = d * @as(f32, @floatFromInt(s_lo.d));
            const m1 = mn * @as(f32, @floatFromInt(s_lo.m));
            const d2 = d * @as(f32, @floatFromInt(s_hi.d));
            const m2 = mn * @as(f32, @floatFromInt(s_hi.m));
            const qb: *const [32]u8 = qs[g * 32 ..][0..32];
            for (0..32) |l| {
                const vlo = d1 * @as(f32, @floatFromInt(qb[l] & 0xF)) - m1;
                acc += vlo * x[i + joff + l];
            }
            for (0..32) |l| {
                const vhi = d2 * @as(f32, @floatFromInt(qb[l] >> 4)) - m2;
                acc += vhi * x[i + joff + 32 + l];
            }
        }
    }
    return acc;
}

/// Vectorizado 4 cadenas; colas por bloque completo únicamente.
pub fn dotQ4_KSimd(w: []const u8, x: []const f32) f32 {
    std.debug.assert(x.len % q6_k_block_elems == 0);
    const V = @Vector(16, f32);
    var acc: [4]V = .{ @splat(0.0), @splat(0.0), @splat(0.0), @splat(0.0) };
    var i: usize = 0;
    var b: usize = 0;
    while (i < x.len) : ({
        i += q6_k_block_elems;
        b += q4_k_bytes;
    }) {
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w[b..][0..2], .little))));
        const mn: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w[b + 2 ..][0..2], .little))));
        const sc: *const [12]u8 = w[b + 4 ..][0..12];
        const qs: *const [128]u8 = w[b + 16 ..][0..128];
        inline for (0..4) |g| {
            const joff = i + g * 64;
            const s_lo = getScaleMinK4(g * 2, sc);
            const s_hi = getScaleMinK4(g * 2 + 1, sc);
            const d1: V = @splat(d * @as(f32, @floatFromInt(s_lo.d)));
            const m1: V = @splat(mn * @as(f32, @floatFromInt(s_lo.m)));
            const d2: V = @splat(d * @as(f32, @floatFromInt(s_hi.d)));
            const m2: V = @splat(mn * @as(f32, @floatFromInt(s_hi.m)));
            const qb: *const [32]u8 = qs[g * 32 ..][0..32];
            inline for (0..2) |hc| {
                const hv: @Vector(16, u8) = qb[hc * 16 ..][0..16].*;
                const lo_f: V = @floatFromInt(@as(@Vector(16, i16), @intCast(hv & @as(@Vector(16, u8), @splat(0x0F)))));
                const hi_f: V = @floatFromInt(@as(@Vector(16, i16), @intCast(hv >> @as(@Vector(16, u3), @splat(4)))));
                const vlo = d1 * lo_f - m1;
                acc[(g * 2 + hc) % 4] += vlo * @as(V, @bitCast(x[joff + hc * 16 ..][0..16].*));
                const vhi = d2 * hi_f - m2;
                acc[(g * 2 + hc + 1) % 4] += vhi * @as(V, @bitCast(x[joff + 32 + hc * 16 ..][0..16].*));
            }
        }
    }
    return (@reduce(.Add, acc[0]) + @reduce(.Add, acc[1])) + (@reduce(.Add, acc[2]) + @reduce(.Add, acc[3]));
}

pub fn dotQ8_0Scalar(w: []const u8, x: []const f32) f32 {
    std.debug.assert(x.len % block_elems == 0);
    var acc: f32 = 0;
    var i: usize = 0;
    var b: usize = 0;
    while (i < x.len) : ({
        i += block_elems;
        b += q8_0_bytes;
    }) {
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w[b..][0..2], .little))));
        const qs = w[b + 2 ..];
        for (0..block_elems) |j| {
            const q: f32 = @floatFromInt(@as(i8, @bitCast(qs[j])));
            acc += (d * q) * x[i + j];
        }
    }
    return acc;
}

pub fn dotScalar(fmt: Format, w: []const u8, x: []const f32) f32 {
    return switch (fmt) {
        .q4_0 => dotQ4_0Scalar(w, x),
        .q8_0 => dotQ8_0Scalar(w, x),
        .q6_k => dotQ6_KScalar(w, x),
        .q4_1 => dotQ4_1Scalar(w, x),
        .q4_k => dotQ4_KScalar(w, x),
        .q5_k => dotQ5_KScalar(w, x),
        .q3_k => dotQ3_KScalar(w, x),
        .q2_k => dotQ2_KScalar(w, x),
        .iq3_s => dotIq3_sScalar(w, x),
        .iq2_s => dotIq2_SScalar(w, x),
        .iq4_nl => dotIq4_nlScalar(w, x),
        .iq2_xxs => dotIq2_xxsScalar(w, x),
    };
}

/// Q6_K escalar de referencia: orden de MEMORIA (l, +32, +64, +96 por l
/// ascendente) = orden de la referencia secuencial sobre la fila dequantizada.
pub fn dotQ6_KScalar(w: []const u8, x: []const f32) f32 {
    std.debug.assert(x.len % q6_k_block_elems == 0);
    var acc: f32 = 0;
    var i: usize = 0;
    var b: usize = 0;
    while (i < x.len) : ({
        i += q6_k_block_elems;
        b += q6_k_block_bytes;
    }) {
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w[b + 208 ..][0..2], .little))));
        const ql = w[b .. b + 128];
        const qh = w[b + 128 .. b + 192];
        const sc = w[b + 192 .. b + 208];
        var n: usize = 0;
        while (n < 256) : (n += 128) {
            const ql2 = ql[(n / 128) * 64 ..];
            const qh2 = qh[(n / 128) * 32 ..];
            const sc2 = sc[(n / 128) * 8 ..];
            // Orden MEMORIA: subgrupo externo (elems contiguos +0,+32,+64,+96),
            // l interno ascendente ⇒ idéntico al scan secuencial de la fila
            // dequantizada (bit-exacto).
            inline for (.{ 0, 1, 2, 3 }) |sg| {
                const shift: u3 = @intCast(sg * 2);
                for (0..32) |l| {
                    const is = l / 16;
                    const nib = if (sg % 2 == 0)
                        if (sg == 0) ql2[l] & 0xF else ql2[l] >> 4
                    else if (sg == 1) ql2[l + 32] & 0xF else ql2[l + 32] >> 4;
                    const q: f32 = @floatFromInt(@as(i8, @bitCast(nib | (((qh2[l] >> shift) & 3) << 4))) - 32);
                    const scv: f32 = @floatFromInt(@as(i8, @bitCast(sc2[is + sg * 2])));
                    const dq = (d * scv) * q; // asociación del oráculo
                    acc += dq * x[i + n + l + sg * 32];
                }
            }
        }
    }
    return acc;
}

// ============================================================================
// Rutas vectorizadas (@Vector(16)) — producción.
// Acumulación por carriles consistente + reduce árbol fijo ⇒ determinista.
// ============================================================================

pub fn dotQ4_0Simd(w: []const u8, x: []const f32) f32 {
    std.debug.assert(x.len % block_elems == 0);
    const V = @Vector(16, f32);
    var a0: V = @splat(0.0);
    var a1: V = @splat(0.0);
    var a2: V = @splat(0.0);
    var a3: V = @splat(0.0);
    var i: usize = 0;
    var b: usize = 0;
    // Dos bloques por iteración ⇒ 4 cadenas de acumulación independientes
    // (rompe la dependencia del fadd; carril j suma elems j, j+32, …).
    while (i + 64 <= x.len) : ({
        i += 64;
        b += 36;
    }) {
        const dA: V = @splat(@floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w[b..][0..2], .little)))));
        const qvA: @Vector(16, u8) = w[b + 2 ..][0..16].*;
        const dBlk = b + q4_0_bytes;
        const dB: V = @splat(@floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w[dBlk..][0..2], .little)))));
        const qvB: @Vector(16, u8) = w[dBlk + 2 ..][0..16].*;
        const e: @Vector(16, i16) = @splat(8);
        const loA: V = @floatFromInt(@as(@Vector(16, i16), @intCast(qvA & @as(@Vector(16, u8), @splat(0x0F)))) - e);
        const hiA: V = @floatFromInt(@as(@Vector(16, i16), @intCast(qvA >> @as(@Vector(16, u3), @splat(4)))) - e);
        const loB: V = @floatFromInt(@as(@Vector(16, i16), @intCast(qvB & @as(@Vector(16, u8), @splat(0x0F)))) - e);
        const hiB: V = @floatFromInt(@as(@Vector(16, i16), @intCast(qvB >> @as(@Vector(16, u3), @splat(4)))) - e);
        a0 += (dA * loA) * @as(V, @bitCast(x[i..][0..16].*));
        a1 += (dA * hiA) * @as(V, @bitCast(x[i + 16 ..][0..16].*));
        a2 += (dB * loB) * @as(V, @bitCast(x[i + 32 ..][0..16].*));
        a3 += (dB * hiB) * @as(V, @bitCast(x[i + 48 ..][0..16].*));
    }
    // Bloque final impar si lo hay.
    if (i < x.len) {
        const d: V = @splat(@floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w[b..][0..2], .little)))));
        const qv: @Vector(16, u8) = w[b + 2 ..][0..16].*;
        const e: @Vector(16, i16) = @splat(8);
        const loq: V = @floatFromInt(@as(@Vector(16, i16), @intCast(qv & @as(@Vector(16, u8), @splat(0x0F)))) - e);
        const hiq: V = @floatFromInt(@as(@Vector(16, i16), @intCast(qv >> @as(@Vector(16, u3), @splat(4)))) - e);
        a0 += (d * loq) * @as(V, @bitCast(x[i..][0..16].*));
        a1 += (d * hiq) * @as(V, @bitCast(x[i + 16 ..][0..16].*));
    }
    return (@reduce(.Add, a0) + @reduce(.Add, a1)) + (@reduce(.Add, a2) + @reduce(.Add, a3));
}

pub fn dotQ8_0Simd(w: []const u8, x: []const f32) f32 {
    std.debug.assert(x.len % block_elems == 0);
    const V = @Vector(16, f32);
    var a0: V = @splat(0.0);
    var a1: V = @splat(0.0);
    var a2: V = @splat(0.0);
    var a3: V = @splat(0.0);
    var i: usize = 0;
    var b: usize = 0;
    // Dos bloques por iteración ⇒ 4 cadenas independientes (ver dotQ4_0Simd).
    while (i + 64 <= x.len) : ({
        i += 64;
        b += 68;
    }) {
        const dA: V = @splat(@floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w[b..][0..2], .little)))));
        const qA0: @Vector(16, i16) = @intCast(@as(@Vector(16, i8), @bitCast(w[b + 2 ..][0..16].*)));
        const qA1: @Vector(16, i16) = @intCast(@as(@Vector(16, i8), @bitCast(w[b + 18 ..][0..16].*)));
        const dBlk = b + q8_0_bytes;
        const dB: V = @splat(@floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w[dBlk..][0..2], .little)))));
        const qB0: @Vector(16, i16) = @intCast(@as(@Vector(16, i8), @bitCast(w[dBlk + 2 ..][0..16].*)));
        const qB1: @Vector(16, i16) = @intCast(@as(@Vector(16, i8), @bitCast(w[dBlk + 18 ..][0..16].*)));
        a0 += (dA * @as(V, @floatFromInt(qA0))) * @as(V, @bitCast(x[i..][0..16].*));
        a1 += (dA * @as(V, @floatFromInt(qA1))) * @as(V, @bitCast(x[i + 16 ..][0..16].*));
        a2 += (dB * @as(V, @floatFromInt(qB0))) * @as(V, @bitCast(x[i + 32 ..][0..16].*));
        a3 += (dB * @as(V, @floatFromInt(qB1))) * @as(V, @bitCast(x[i + 48 ..][0..16].*));
    }
    if (i < x.len) {
        const d: V = @splat(@floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w[b..][0..2], .little)))));
        const q0: @Vector(16, i16) = @intCast(@as(@Vector(16, i8), @bitCast(w[b + 2 ..][0..16].*)));
        const q1: @Vector(16, i16) = @intCast(@as(@Vector(16, i8), @bitCast(w[b + 18 ..][0..16].*)));
        a0 += (d * @as(V, @floatFromInt(q0))) * @as(V, @bitCast(x[i..][0..16].*));
        a1 += (d * @as(V, @floatFromInt(q1))) * @as(V, @bitCast(x[i + 16 ..][0..16].*));
    }
    return (@reduce(.Add, a0) + @reduce(.Add, a1)) + (@reduce(.Add, a2) + @reduce(.Add, a3));
}

/// Par de filas Q8_0 fusionado: comparte las cargas de x entre ambas filas
/// (x domina el tráfico: 256B vs 68B de W por par de bloques ⇒ ~2× aritmética
/// intensiva). Determinismo intacto: cada fila conserva sus propias cadenas.
pub fn dotQ8_0Pair(w0: []const u8, w1: []const u8, x: []const f32, o0: *f32, o1: *f32) void {
    std.debug.assert(x.len % block_elems == 0);
    const V = @Vector(16, f32);
    var ra0: V = @splat(0.0);
    var ra1: V = @splat(0.0);
    var rb0: V = @splat(0.0);
    var rb1: V = @splat(0.0);
    var sa0: V = @splat(0.0);
    var sa1: V = @splat(0.0);
    var sb0: V = @splat(0.0);
    var sb1: V = @splat(0.0);
    var i: usize = 0;
    var b: usize = 0;
    while (i + 64 <= x.len) : ({
        i += 64;
        b += 68;
    }) {
        const dA: V = @splat(@floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w0[b..][0..2], .little)))));
        const dA2: V = @splat(@floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w1[b..][0..2], .little)))));
        const qA0: @Vector(16, i16) = @intCast(@as(@Vector(16, i8), @bitCast(w0[b + 2 ..][0..16].*)));
        const qA1: @Vector(16, i16) = @intCast(@as(@Vector(16, i8), @bitCast(w0[b + 18 ..][0..16].*)));
        const pA0: @Vector(16, i16) = @intCast(@as(@Vector(16, i8), @bitCast(w1[b + 2 ..][0..16].*)));
        const pA1: @Vector(16, i16) = @intCast(@as(@Vector(16, i8), @bitCast(w1[b + 18 ..][0..16].*)));
        const dBlk = b + q8_0_bytes;
        const dB: V = @splat(@floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w0[dBlk..][0..2], .little)))));
        const dB2: V = @splat(@floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w1[dBlk..][0..2], .little)))));
        const qB0: @Vector(16, i16) = @intCast(@as(@Vector(16, i8), @bitCast(w0[dBlk + 2 ..][0..16].*)));
        const qB1: @Vector(16, i16) = @intCast(@as(@Vector(16, i8), @bitCast(w0[dBlk + 18 ..][0..16].*)));
        const pB0: @Vector(16, i16) = @intCast(@as(@Vector(16, i8), @bitCast(w1[dBlk + 2 ..][0..16].*)));
        const pB1: @Vector(16, i16) = @intCast(@as(@Vector(16, i8), @bitCast(w1[dBlk + 18 ..][0..16].*)));
        const x0: V = @bitCast(x[i..][0..16].*);
        const x1: V = @bitCast(x[i + 16 ..][0..16].*);
        const x2: V = @bitCast(x[i + 32 ..][0..16].*);
        const x3: V = @bitCast(x[i + 48 ..][0..16].*);
        ra0 += (dA * @as(V, @floatFromInt(qA0))) * x0;
        ra1 += (dA * @as(V, @floatFromInt(qA1))) * x1;
        rb0 += (dB * @as(V, @floatFromInt(qB0))) * x2;
        rb1 += (dB * @as(V, @floatFromInt(qB1))) * x3;
        sa0 += (dA2 * @as(V, @floatFromInt(pA0))) * x0;
        sa1 += (dA2 * @as(V, @floatFromInt(pA1))) * x1;
        sb0 += (dB2 * @as(V, @floatFromInt(pB0))) * x2;
        sb1 += (dB2 * @as(V, @floatFromInt(pB1))) * x3;
    }
    if (i < x.len) {
        // Bloque final impar por fila (raro: K múltiplo de 64 en la práctica).
        const tail = dotQ8_0SimdTail(w0[b..], x[i..]) + @reduce(.Add, ra0) + @reduce(.Add, ra1) + @reduce(.Add, rb0) + @reduce(.Add, rb1);
        o0.* = tail;
        o1.* = dotQ8_0Simd(w1, x); // fila par sin pareja de cola: camino simple
        return;
    }
    o0.* = (@reduce(.Add, ra0) + @reduce(.Add, ra1)) + (@reduce(.Add, rb0) + @reduce(.Add, rb1));
    o1.* = (@reduce(.Add, sa0) + @reduce(.Add, sa1)) + (@reduce(.Add, sb0) + @reduce(.Add, sb1));
}

fn dotQ8_0SimdTail(wb: []const u8, xb: []const f32) f32 {
    const V = @Vector(16, f32);
    const d: V = @splat(@floatCast(@as(f16, @bitCast(std.mem.readInt(u16, wb[0..2], .little)))));
    const q0: @Vector(16, i16) = @intCast(@as(@Vector(16, i8), @bitCast(wb[2..18].*)));
    const q1: @Vector(16, i16) = @intCast(@as(@Vector(16, i8), @bitCast(wb[18..34].*)));
    return (@reduce(.Add, (d * @as(V, @floatFromInt(q0))) * @as(V, @bitCast(xb[0..16].*))) +
        @reduce(.Add, (d * @as(V, @floatFromInt(q1))) * @as(V, @bitCast(xb[16..32].*))));
}

/// Par de filas Q4_K fusionado (comparte cargas de x entre ambas filas).
pub fn dotQ4_KPair(w0: []const u8, w1: []const u8, x: []const f32, o0: *f32, o1: *f32) void {
    std.debug.assert(x.len % q6_k_block_elems == 0);
    const V = @Vector(16, f32);
    var ra: [4]V = .{ @splat(0.0), @splat(0.0), @splat(0.0), @splat(0.0) };
    var sa: [4]V = .{ @splat(0.0), @splat(0.0), @splat(0.0), @splat(0.0) };
    var i: usize = 0;
    var b: usize = 0;
    while (i < x.len) : ({
        i += q6_k_block_elems;
        b += q4_k_bytes;
    }) {
        const dA: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w0[b..][0..2], .little))));
        const mnA: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w0[b + 2 ..][0..2], .little))));
        const dB: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w1[b..][0..2], .little))));
        const mnB: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w1[b + 2 ..][0..2], .little))));
        const scA: *const [12]u8 = w0[b + 4 ..][0..12];
        const scB: *const [12]u8 = w1[b + 4 ..][0..12];
        inline for (0..4) |g| {
            const joff = i + g * 64;
            const sAl = getScaleMinK4(g * 2, scA);
            const sAh = getScaleMinK4(g * 2 + 1, scA);
            const sBl = getScaleMinK4(g * 2, scB);
            const sBh = getScaleMinK4(g * 2 + 1, scB);
            const qbA: *const [32]u8 = w0[b + 16 + g * 32 ..][0..32];
            const qbB: *const [32]u8 = w1[b + 16 + g * 32 ..][0..32];
            inline for (0..2) |hc| {
                const hvA: @Vector(16, u8) = qbA[hc * 16 ..][0..16].*;
                const hvB: @Vector(16, u8) = qbB[hc * 16 ..][0..16].*;
                const loA: V = @floatFromInt(@as(@Vector(16, i16), @intCast(hvA & @as(@Vector(16, u8), @splat(0x0F)))));
                const hiA: V = @floatFromInt(@as(@Vector(16, i16), @intCast(hvA >> @as(@Vector(16, u3), @splat(4)))));
                const loB: V = @floatFromInt(@as(@Vector(16, i16), @intCast(hvB & @as(@Vector(16, u8), @splat(0x0F)))));
                const hiB: V = @floatFromInt(@as(@Vector(16, i16), @intCast(hvB >> @as(@Vector(16, u3), @splat(4)))));
                const dAl: V = @splat(dA * @as(f32, @floatFromInt(sAl.d)));
                const mA: V = @splat(mnA * @as(f32, @floatFromInt(sAl.m)));
                const dAh: V = @splat(dA * @as(f32, @floatFromInt(sAh.d)));
                const mAh: V = @splat(mnA * @as(f32, @floatFromInt(sAh.m)));
                const dBl: V = @splat(dB * @as(f32, @floatFromInt(sBl.d)));
                const mB: V = @splat(mnB * @as(f32, @floatFromInt(sBl.m)));
                const dBh: V = @splat(dB * @as(f32, @floatFromInt(sBh.d)));
                const mBh: V = @splat(mnB * @as(f32, @floatFromInt(sBh.m)));
                const x0: V = @bitCast(x[joff + hc * 16 ..][0..16].*);
                const x1: V = @bitCast(x[joff + 32 + hc * 16 ..][0..16].*);
                ra[(g * 2 + hc) % 4] += (dAl * loA - mA) * x0;
                ra[(g * 2 + hc + 1) % 4] += (dAh * hiA - mAh) * x1;
                sa[(g * 2 + hc) % 4] += (dBl * loB - mB) * x0;
                sa[(g * 2 + hc + 1) % 4] += (dBh * hiB - mBh) * x1;
            }
        }
    }
    o0.* = (@reduce(.Add, ra[0]) + @reduce(.Add, ra[1])) + (@reduce(.Add, ra[2]) + @reduce(.Add, ra[3]));
    o1.* = (@reduce(.Add, sa[0]) + @reduce(.Add, sa[1])) + (@reduce(.Add, sa[2]) + @reduce(.Add, sa[3]));
}

// ============================================================================
// Q5_K — súper-bloque 256 elems / 176 bytes: como Q4_K + qh[32] @16 con los
// bits altos (16·bit) por grupo de 64; qs[128] @48. bit1=1<<(2g), bit2=2<<2g.
// ============================================================================

pub fn dequantQ5_K(bytes: []const u8, out: []f32) void {
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : ({
        i += q6_k_block_elems;
        nb += 1;
    }) {
        const nv = @min(q6_k_block_elems, out.len - i);
        const base = nb * q5_k_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const mn: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base + 2 ..][0..2], .little))));
        const sc: *const [12]u8 = bytes[base + 4 ..][0..12];
        const qh: *const [32]u8 = bytes[base + 16 ..][0..32];
        const qs: *const [128]u8 = bytes[base + 48 ..][0..128];

        inline for (0..4) |g| {
            const joff = g * 64;
            const s_lo = getScaleMinK4(g * 2, sc);
            const s_hi = getScaleMinK4(g * 2 + 1, sc);
            const d1: @Vector(16, f32) = @splat(d * @as(f32, @floatFromInt(s_lo.d)));
            const m1: @Vector(16, f32) = @splat(mn * @as(f32, @floatFromInt(s_lo.m)));
            const d2: @Vector(16, f32) = @splat(d * @as(f32, @floatFromInt(s_hi.d)));
            const m2: @Vector(16, f32) = @splat(mn * @as(f32, @floatFromInt(s_hi.m)));
            const qb: *const [32]u8 = qs[g * 32 ..][0..32];
            const sh1: u3 = @intCast(2 * g);
            const sh2: u3 = @intCast(2 * g + 1);
            inline for (0..2) |hc| {
                const hv: @Vector(16, u8) = qb[hc * 16 ..][0..16].*;
                const hbv: @Vector(16, u8) = qh[hc * 16 ..][0..16].*;
                const hi1: @Vector(16, u8) = ((hbv >> @as(@Vector(16, u3), @splat(sh1))) & @as(@Vector(16, u8), @splat(1))) * @as(@Vector(16, u8), @splat(16));
                const hi2: @Vector(16, u8) = ((hbv >> @as(@Vector(16, u3), @splat(sh2))) & @as(@Vector(16, u8), @splat(1))) * @as(@Vector(16, u8), @splat(16));
                const v1: @Vector(16, f32) = @floatFromInt(@as(@Vector(16, i16), @intCast((hv & @as(@Vector(16, u8), @splat(0x0F))) + hi1)));
                const v2: @Vector(16, f32) = @floatFromInt(@as(@Vector(16, i16), @intCast((hv >> @as(@Vector(16, u3), @splat(4))) + hi2)));
                const o0 = i + joff + hc * 16;
                const o1 = i + joff + 32 + hc * 16;
                const vals1: [16]f32 = @bitCast(d1 * v1 - m1);
                const vals2: [16]f32 = @bitCast(d2 * v2 - m2);
                if (o0 + 16 <= i + nv) {
                    (out[o0..][0..16].*) = vals1;
                } else if (o0 < i + nv) {
                    for (0..i + nv - o0) |k| out[o0 + k] = vals1[k];
                }
                if (o1 + 16 <= i + nv) {
                    (out[o1..][0..16].*) = vals2;
                } else if (o1 < i + nv) {
                    for (0..i + nv - o1) |k| out[o1 + k] = vals2[k];
                }
            }
        }
    }
}

/// Escalar de referencia bit-exacto: orden MEMORIA por grupos y asociación
/// del oráculo ((d·sc)·q − (m·sc)).
pub fn dotQ5_KScalar(w: []const u8, x: []const f32) f32 {
    std.debug.assert(x.len % q6_k_block_elems == 0);
    var acc: f32 = 0;
    var i: usize = 0;
    var b: usize = 0;
    while (i < x.len) : ({
        i += q6_k_block_elems;
        b += q5_k_bytes;
    }) {
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w[b..][0..2], .little))));
        const mn: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w[b + 2 ..][0..2], .little))));
        const sc: *const [12]u8 = w[b + 4 ..][0..12];
        const qh: *const [32]u8 = w[b + 16 ..][0..32];
        const qs: *const [128]u8 = w[b + 48 ..][0..128];
        inline for (0..4) |g| {
            const joff = g * 64;
            const s_lo = getScaleMinK4(g * 2, sc);
            const s_hi = getScaleMinK4(g * 2 + 1, sc);
            const d1 = d * @as(f32, @floatFromInt(s_lo.d));
            const m1 = mn * @as(f32, @floatFromInt(s_lo.m));
            const d2 = d * @as(f32, @floatFromInt(s_hi.d));
            const m2 = mn * @as(f32, @floatFromInt(s_hi.m));
            const qb: *const [32]u8 = qs[g * 32 ..][0..32];
            const b1: u8 = @as(u8, 1) << @intCast(2 * g);
            const b2: u8 = @as(u8, 1) << @intCast(2 * g + 1);
            for (0..32) |l| {
                const v1: f32 = @floatFromInt((qb[l] & 0xF) + @as(u8, @intFromBool(qh[l] & b1 != 0)) * 16);
                acc += (d1 * v1 - m1) * x[i + joff + l];
            }
            for (0..32) |l| {
                const v2: f32 = @floatFromInt((qb[l] >> 4) + @as(u8, @intFromBool(qh[l] & b2 != 0)) * 16);
                acc += (d2 * v2 - m2) * x[i + joff + 32 + l];
            }
        }
    }
    return acc;
}

/// Vectorizado 4 cadenas por grupos.
pub fn dotQ5_KSimd(w: []const u8, x: []const f32) f32 {
    std.debug.assert(x.len % q6_k_block_elems == 0);
    const V = @Vector(16, f32);
    var acc: [4]V = .{ @splat(0.0), @splat(0.0), @splat(0.0), @splat(0.0) };
    var i: usize = 0;
    var b: usize = 0;
    while (i < x.len) : ({
        i += q6_k_block_elems;
        b += q5_k_bytes;
    }) {
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w[b..][0..2], .little))));
        const mn: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w[b + 2 ..][0..2], .little))));
        const sc: *const [12]u8 = w[b + 4 ..][0..12];
        const qh: *const [32]u8 = w[b + 16 ..][0..32];
        const qs: *const [128]u8 = w[b + 48 ..][0..128];
        inline for (0..4) |g| {
            const joff = i + g * 64;
            const s_lo = getScaleMinK4(g * 2, sc);
            const s_hi = getScaleMinK4(g * 2 + 1, sc);
            const d1: V = @splat(d * @as(f32, @floatFromInt(s_lo.d)));
            const m1: V = @splat(mn * @as(f32, @floatFromInt(s_lo.m)));
            const d2: V = @splat(d * @as(f32, @floatFromInt(s_hi.d)));
            const m2: V = @splat(mn * @as(f32, @floatFromInt(s_hi.m)));
            const qb: *const [32]u8 = qs[g * 32 ..][0..32];
            const sh1: u3 = @intCast(2 * g);
            const sh2: u3 = @intCast(2 * g + 1);
            inline for (0..2) |hc| {
                const hv: @Vector(16, u8) = qb[hc * 16 ..][0..16].*;
                const hbv: @Vector(16, u8) = qh[hc * 16 ..][0..16].*;
                const hi1: @Vector(16, u8) = ((hbv >> @as(@Vector(16, u3), @splat(sh1))) & @as(@Vector(16, u8), @splat(1))) * @as(@Vector(16, u8), @splat(16));
                const hi2: @Vector(16, u8) = ((hbv >> @as(@Vector(16, u3), @splat(sh2))) & @as(@Vector(16, u8), @splat(1))) * @as(@Vector(16, u8), @splat(16));
                const v1: V = @floatFromInt(@as(@Vector(16, i16), @intCast((hv & @as(@Vector(16, u8), @splat(0x0F))) + hi1)));
                const v2: V = @floatFromInt(@as(@Vector(16, i16), @intCast((hv >> @as(@Vector(16, u3), @splat(4))) + hi2)));
                const x0: V = @bitCast(x[joff + hc * 16 ..][0..16].*);
                const x1: V = @bitCast(x[joff + 32 + hc * 16 ..][0..16].*);
                acc[(g * 2 + hc) % 4] += (d1 * v1 - m1) * x0;
                acc[(g * 2 + hc + 1) % 4] += (d2 * v2 - m2) * x1;
            }
        }
    }
    return (@reduce(.Add, acc[0]) + @reduce(.Add, acc[1])) + (@reduce(.Add, acc[2]) + @reduce(.Add, acc[3]));
}

/// Par de filas Q5_K fusionado (comparte cargas de x).
pub fn dotQ5_KPair(w0: []const u8, w1: []const u8, x: []const f32, o0: *f32, o1: *f32) void {
    std.debug.assert(x.len % q6_k_block_elems == 0);
    const V = @Vector(16, f32);
    var ra: [4]V = .{ @splat(0.0), @splat(0.0), @splat(0.0), @splat(0.0) };
    var sa: [4]V = .{ @splat(0.0), @splat(0.0), @splat(0.0), @splat(0.0) };
    var i: usize = 0;
    var b: usize = 0;
    while (i < x.len) : ({
        i += q6_k_block_elems;
        b += q5_k_bytes;
    }) {
        const dA: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w0[b..][0..2], .little))));
        const mnA: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w0[b + 2 ..][0..2], .little))));
        const dB: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w1[b..][0..2], .little))));
        const mnB: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w1[b + 2 ..][0..2], .little))));
        const scA: *const [12]u8 = w0[b + 4 ..][0..12];
        const scB: *const [12]u8 = w1[b + 4 ..][0..12];
        inline for (0..4) |g| {
            const joff = i + g * 64;
            const sAl = getScaleMinK4(g * 2, scA);
            const sAh = getScaleMinK4(g * 2 + 1, scA);
            const sBl = getScaleMinK4(g * 2, scB);
            const sBh = getScaleMinK4(g * 2 + 1, scB);
            const qbA: *const [32]u8 = w0[b + 48 + g * 32 ..][0..32];
            const qbB: *const [32]u8 = w1[b + 48 + g * 32 ..][0..32];
            const hbA: *const [32]u8 = w0[b + 16 ..][0..32];
            const hbB: *const [32]u8 = w1[b + 16 ..][0..32];
            const sh1: u3 = @intCast(2 * g);
            const sh2: u3 = @intCast(2 * g + 1);
            inline for (0..2) |hc| {
                const hvA: @Vector(16, u8) = qbA[hc * 16 ..][0..16].*;
                const hvB: @Vector(16, u8) = qbB[hc * 16 ..][0..16].*;
                const hbAv: @Vector(16, u8) = hbA[hc * 16 ..][0..16].*;
                const hbBv: @Vector(16, u8) = hbB[hc * 16 ..][0..16].*;
                const hiA1: @Vector(16, u8) = ((hbAv >> @as(@Vector(16, u3), @splat(sh1))) & @as(@Vector(16, u8), @splat(1))) * @as(@Vector(16, u8), @splat(16));
                const hiA2: @Vector(16, u8) = ((hbAv >> @as(@Vector(16, u3), @splat(sh2))) & @as(@Vector(16, u8), @splat(1))) * @as(@Vector(16, u8), @splat(16));
                const hiB1: @Vector(16, u8) = ((hbBv >> @as(@Vector(16, u3), @splat(sh1))) & @as(@Vector(16, u8), @splat(1))) * @as(@Vector(16, u8), @splat(16));
                const hiB2: @Vector(16, u8) = ((hbBv >> @as(@Vector(16, u3), @splat(sh2))) & @as(@Vector(16, u8), @splat(1))) * @as(@Vector(16, u8), @splat(16));
                const vAl: V = @floatFromInt(@as(@Vector(16, i16), @intCast((hvA & @as(@Vector(16, u8), @splat(0x0F))) + hiA1)));
                const vAh: V = @floatFromInt(@as(@Vector(16, i16), @intCast((hvA >> @as(@Vector(16, u3), @splat(4))) + hiA2)));
                const vBl: V = @floatFromInt(@as(@Vector(16, i16), @intCast((hvB & @as(@Vector(16, u8), @splat(0x0F))) + hiB1)));
                const vBh: V = @floatFromInt(@as(@Vector(16, i16), @intCast((hvB >> @as(@Vector(16, u3), @splat(4))) + hiB2)));
                const dAl: V = @splat(dA * @as(f32, @floatFromInt(sAl.d)));
                const mA: V = @splat(mnA * @as(f32, @floatFromInt(sAl.m)));
                const dAh: V = @splat(dA * @as(f32, @floatFromInt(sAh.d)));
                const mAh: V = @splat(mnA * @as(f32, @floatFromInt(sAh.m)));
                const dBl: V = @splat(dB * @as(f32, @floatFromInt(sBl.d)));
                const mB: V = @splat(mnB * @as(f32, @floatFromInt(sBl.m)));
                const dBh: V = @splat(dB * @as(f32, @floatFromInt(sBh.d)));
                const mBh: V = @splat(mnB * @as(f32, @floatFromInt(sBh.m)));
                const x0: V = @bitCast(x[joff + hc * 16 ..][0..16].*);
                const x1: V = @bitCast(x[joff + 32 + hc * 16 ..][0..16].*);
                ra[(g * 2 + hc) % 4] += (dAl * vAl - mA) * x0;
                ra[(g * 2 + hc + 1) % 4] += (dAh * vAh - mAh) * x1;
                sa[(g * 2 + hc) % 4] += (dBl * vBl - mB) * x0;
                sa[(g * 2 + hc + 1) % 4] += (dBh * vBh - mBh) * x1;
            }
        }
    }
    o0.* = (@reduce(.Add, ra[0]) + @reduce(.Add, ra[1])) + (@reduce(.Add, ra[2]) + @reduce(.Add, ra[3]));
    o1.* = (@reduce(.Add, sa[0]) + @reduce(.Add, sa[1])) + (@reduce(.Add, sa[2]) + @reduce(.Add, sa[3]));
}

// ============================================================================
// Q3_K — súper-bloque 256 elems / 110 bytes: hmask[32] @0 (bit de signo por
// j), qs[64] @32 (2 bits/elem), scales[12] @96 con REORDEN kmask a 16 i8,
// d:f16 @108. val = dl·(q − h) con dl=d·(sc−32), h=0 si signo-set si no +4.
// ============================================================================

/// Reordenamiento de escalas fiel al C de ggml (aux shuffle kmask).
fn reorderScalesQ3K(scales: *const [12]u8) [16]i8 {
    const kmask1: u32 = 0x03030303;
    const kmask2: u32 = 0x0f0f0f0f;
    var aux: [4]u32 = undefined;
    @memcpy(std.mem.sliceAsBytes(aux[0..4])[0..12], scales);
    const tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    var out16: [16]i8 = undefined;
    @memcpy(std.mem.sliceAsBytes(out16[0..16]), std.mem.sliceAsBytes(aux[0..4]));
    return out16;
}

pub fn dequantQ3_K(bytes: []const u8, out: []f32) void {
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : ({
        i += q6_k_block_elems;
        nb += 1;
    }) {
        const nv = @min(q6_k_block_elems, out.len - i);
        const base = nb * q3_k_bytes;
        const d_bits = std.mem.readInt(u16, bytes[base + 108 ..][0..2], .little);
        const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
        const hm: *const [32]u8 = bytes[base..][0..32];
        const qs: *const [64]u8 = bytes[base + 32 ..][0..64];
        const sc16 = reorderScalesQ3K(bytes[base + 96 ..][0..12].ptr[0..12]);

        inline for (0..2) |n| {
            const is_base = n * 8;
            const qb: *const [32]u8 = qs[n * 32 ..][0..32];
            // shift y m CONTINÚAN entre mitades (declarados fuera en el oráculo).
            inline for (0..4) |j| {
                const shift: u3 = @intCast(2 * j); // reinicia por mitad (oráculo)
                const mb: u8 = @as(u8, 1) << @intCast(n * 4 + j); // m continúa
                const dl: @Vector(16, f32) = @splat(d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc16[is_base + j * 2])) - 32)));
                const dl2: @Vector(16, f32) = @splat(d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc16[is_base + j * 2 + 1])) - 32)));
                var q1: @Vector(16, i16) = @splat(0);
                var q2: @Vector(16, i16) = @splat(0);
                if (shift < 8) {
                    const s3: u3 = @intCast(shift);
                    const q1v: @Vector(16, u8) = qb[0..16].*;
                    const q2v: @Vector(16, u8) = qb[16..32].*;
                    q1 = @intCast((q1v >> @as(@Vector(16, u3), @splat(s3))) & @as(@Vector(16, u8), @splat(3)));
                    q2 = @intCast((q2v >> @as(@Vector(16, u3), @splat(s3))) & @as(@Vector(16, u8), @splat(3)));
                }
                const hm1: @Vector(16, u8) = hm[0..16].*;
                const hm2: @Vector(16, u8) = hm[16..32].*;
                const z: @Vector(16, u8) = @splat(0);
                const c1: @Vector(16, i16) = @intCast(@as(@Vector(16, u8), @select(u8, hm1 & @as(@Vector(16, u8), @splat(mb)) == @as(@Vector(16, u8), @splat(mb)), z, @as(@Vector(16, u8), @splat(4)))));
                const c2: @Vector(16, i16) = @intCast(@as(@Vector(16, u8), @select(u8, hm2 & @as(@Vector(16, u8), @splat(mb)) == @as(@Vector(16, u8), @splat(mb)), z, @as(@Vector(16, u8), @splat(4)))));
                const v1: [16]f32 = @bitCast(dl * @as(@Vector(16, f32), @floatFromInt(q1 - c1)));
                const v2: [16]f32 = @bitCast(dl2 * @as(@Vector(16, f32), @floatFromInt(q2 - c2)));
                const o0 = i + n * 128 + j * 32;
                const o1 = o0 + 16;
                if (o0 + 16 <= i + nv) {
                    (out[o0..][0..16].*) = v1;
                } else if (o0 < i + nv) {
                    for (0..i + nv - o0) |k| out[o0 + k] = v1[k];
                }
                if (o1 + 16 <= i + nv) {
                    (out[o1..][0..16].*) = v2;
                } else if (o1 < i + nv) {
                    for (0..i + nv - o1) |k| out[o1 + k] = v2[k];
                }
            }
        }
    }
}

/// Escalar de referencia bit-exacto: acumulación en ORDEN MEMORIA
/// (posición ascendente dentro del SB) con asociación del oráculo.
pub fn dotQ3_KScalar(w: []const u8, x: []const f32) f32 {
    std.debug.assert(x.len % q6_k_block_elems == 0);
    var acc: f32 = 0;
    var i: usize = 0;
    var b: usize = 0;
    while (i < x.len) : ({
        i += q6_k_block_elems;
        b += q3_k_bytes;
    }) {
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w[b + 108 ..][0..2], .little))));
        const hm: *const [32]u8 = w[b..][0..32];
        const qs: *const [64]u8 = w[b + 32 ..][0..64];
        const sc16 = reorderScalesQ3K(w[b + 96 ..][0..12].ptr[0..12]);
        inline for (0..2) |n| {
            const is_base = n * 8;
            // Posición ascendente: p ⇒ j=p/32, rem=p%32, l=rem%16, side=rem/16.
            for (0..128) |p| {
                const j = p / 32;
                const l = p % 16;
                const side = p % 32 / 16;
                // shift reinicia por mitad; m continúa vía n (ver oráculo)
                const shift: u3 = @intCast(2 * j);
                const mb: u8 = @as(u8, 1) << @intCast(n * 4 + j);
                const qb: u8 = if (side == 0) qs[n * 32 + l] else qs[n * 32 + 16 + l];
                const hb: u8 = if (side == 0) hm[l] else hm[l + 16];
                const sc_i8: i8 = @bitCast(sc16[is_base + j * 2 + side]);
                const dl = d * @as(f32, @floatFromInt(sc_i8 - 32));
                const h: i32 = if (hb & mb != 0) 0 else 4;
                const q: i32 = (@as(i32, qb) >> shift) & 3;
                const val = dl * @as(f32, @floatFromInt(q - h));
                acc += val * x[i + n * 128 + p];
            }
        }
    }
    return acc;
}

/// Vectorizado 4 cadenas por (mitad, j).
pub fn dotQ3_KSimd(w: []const u8, x: []const f32) f32 {
    std.debug.assert(x.len % q6_k_block_elems == 0);
    const V = @Vector(16, f32);
    var acc: [4]V = .{ @splat(0.0), @splat(0.0), @splat(0.0), @splat(0.0) };
    var i: usize = 0;
    var b: usize = 0;
    while (i < x.len) : ({
        i += q6_k_block_elems;
        b += q3_k_bytes;
    }) {
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w[b + 108 ..][0..2], .little))));
        const hm: *const [32]u8 = w[b..][0..32];
        const qs: *const [64]u8 = w[b + 32 ..][0..64];
        const sc16 = reorderScalesQ3K(w[b + 96 ..][0..12].ptr[0..12]);
        inline for (0..2) |n| {
            const is_base = n * 8;
            const qb: *const [32]u8 = qs[n * 32 ..][0..32];
            // shift reinicia por mitad; m continúa vía n (ver oráculo).
            inline for (0..4) |j| {
                const shift: u3 = @intCast(2 * j);
                const mb: u8 = @as(u8, 1) << @intCast(n * 4 + j);
                const dl: V = @splat(d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc16[is_base + j * 2])) - 32)));
                const dl2: V = @splat(d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc16[is_base + j * 2 + 1])) - 32)));
                var q1: @Vector(16, i16) = @splat(0);
                var q2: @Vector(16, i16) = @splat(0);
                if (shift < 8) {
                    const s3: u3 = @intCast(shift);
                    const q1v: @Vector(16, u8) = qb[0..16].*;
                    const q2v: @Vector(16, u8) = qb[16..32].*;
                    q1 = @intCast((q1v >> @as(@Vector(16, u3), @splat(s3))) & @as(@Vector(16, u8), @splat(3)));
                    q2 = @intCast((q2v >> @as(@Vector(16, u3), @splat(s3))) & @as(@Vector(16, u8), @splat(3)));
                }
                const hm1: @Vector(16, u8) = hm[0..16].*;
                const hm2: @Vector(16, u8) = hm[16..32].*;
                const z: @Vector(16, u8) = @splat(0);
                const c1: @Vector(16, i16) = @intCast(@as(@Vector(16, u8), @select(u8, hm1 & @as(@Vector(16, u8), @splat(mb)) == @as(@Vector(16, u8), @splat(mb)), z, @as(@Vector(16, u8), @splat(4)))));
                const c2: @Vector(16, i16) = @intCast(@as(@Vector(16, u8), @select(u8, hm2 & @as(@Vector(16, u8), @splat(mb)) == @as(@Vector(16, u8), @splat(mb)), z, @as(@Vector(16, u8), @splat(4)))));
                const x0: V = @bitCast(x[i + n * 128 + j * 32 ..][0..16].*);
                const x1: V = @bitCast(x[i + n * 128 + j * 32 + 16 ..][0..16].*);
                acc[(n * 4 + j * 2) % 4] += dl * @as(V, @floatFromInt(q1 - c1)) * x0;
                acc[(n * 4 + j * 2 + 1) % 4] += dl2 * @as(V, @floatFromInt(q2 - c2)) * x1;
            }
        }
    }
    return (@reduce(.Add, acc[0]) + @reduce(.Add, acc[1])) + (@reduce(.Add, acc[2]) + @reduce(.Add, acc[3]));
}

/// Par de filas Q3_K fusionado (comparte cargas de x).
pub fn dotQ3_KPair(w0: []const u8, w1: []const u8, x: []const f32, o0: *f32, o1: *f32) void {
    std.debug.assert(x.len % q6_k_block_elems == 0);
    const V = @Vector(16, f32);
    var ra: [4]V = .{ @splat(0.0), @splat(0.0), @splat(0.0), @splat(0.0) };
    var sa: [4]V = .{ @splat(0.0), @splat(0.0), @splat(0.0), @splat(0.0) };
    var i: usize = 0;
    var b: usize = 0;
    while (i < x.len) : ({
        i += q6_k_block_elems;
        b += q3_k_bytes;
    }) {
        const dA: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w0[b + 108 ..][0..2], .little))));
        const dB: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w1[b + 108 ..][0..2], .little))));
        const hmA: *const [32]u8 = w0[b..][0..32];
        const hmB: *const [32]u8 = w1[b..][0..32];
        const qsA: *const [64]u8 = w0[b + 32 ..][0..64];
        const qsB: *const [64]u8 = w1[b + 32 ..][0..64];
        const scA = reorderScalesQ3K(w0[b + 96 ..][0..12].ptr[0..12]);
        const scB = reorderScalesQ3K(w1[b + 96 ..][0..12].ptr[0..12]);
        inline for (0..2) |n| {
            const qa: *const [32]u8 = qsA[n * 32 ..][0..32];
            const qbb: *const [32]u8 = qsB[n * 32 ..][0..32];
            inline for (0..4) |j| {
                const shift: u3 = @intCast(2 * j); // reinicia por mitad
                const mb: u8 = @as(u8, 1) << @intCast(n * 4 + j); // m continúa
                const dlA: V = @splat(dA * @as(f32, @floatFromInt(@as(i8, @bitCast(scA[n * 8 + j * 2])) - 32)));
                const dlA2: V = @splat(dA * @as(f32, @floatFromInt(@as(i8, @bitCast(scA[n * 8 + j * 2 + 1])) - 32)));
                const dlB: V = @splat(dB * @as(f32, @floatFromInt(@as(i8, @bitCast(scB[n * 8 + j * 2])) - 32)));
                const dlB2: V = @splat(dB * @as(f32, @floatFromInt(@as(i8, @bitCast(scB[n * 8 + j * 2 + 1])) - 32)));
                const qa1: @Vector(16, u8) = qa[0..16].*;
                const qa2: @Vector(16, u8) = qa[16..32].*;
                const qb1: @Vector(16, u8) = qbb[0..16].*;
                const qb2: @Vector(16, u8) = qbb[16..32].*;
                var qA1: @Vector(16, i16) = @splat(0);
                var qA2: @Vector(16, i16) = @splat(0);
                var qB1: @Vector(16, i16) = @splat(0);
                var qB2: @Vector(16, i16) = @splat(0);
                if (shift < 8) {
                    const s3: u3 = @intCast(shift);
                    qA1 = @intCast((qa1 >> @as(@Vector(16, u3), @splat(s3))) & @as(@Vector(16, u8), @splat(3)));
                    qA2 = @intCast((qa2 >> @as(@Vector(16, u3), @splat(s3))) & @as(@Vector(16, u8), @splat(3)));
                    qB1 = @intCast((qb1 >> @as(@Vector(16, u3), @splat(s3))) & @as(@Vector(16, u8), @splat(3)));
                    qB2 = @intCast((qb2 >> @as(@Vector(16, u3), @splat(s3))) & @as(@Vector(16, u8), @splat(3)));
                }
                const ha1: @Vector(16, u8) = hmA[0..16].*;
                const ha2: @Vector(16, u8) = hmA[16..32].*;
                const hb1: @Vector(16, u8) = hmB[0..16].*;
                const hb2: @Vector(16, u8) = hmB[16..32].*;
                const z: @Vector(16, u8) = @splat(0);
                const cA1: @Vector(16, i16) = @intCast(@as(@Vector(16, u8), @select(u8, ha1 & @as(@Vector(16, u8), @splat(mb)) == @as(@Vector(16, u8), @splat(mb)), z, @as(@Vector(16, u8), @splat(4)))));
                const cA2: @Vector(16, i16) = @intCast(@as(@Vector(16, u8), @select(u8, ha2 & @as(@Vector(16, u8), @splat(mb)) == @as(@Vector(16, u8), @splat(mb)), z, @as(@Vector(16, u8), @splat(4)))));
                const cB1: @Vector(16, i16) = @intCast(@as(@Vector(16, u8), @select(u8, hb1 & @as(@Vector(16, u8), @splat(mb)) == @as(@Vector(16, u8), @splat(mb)), z, @as(@Vector(16, u8), @splat(4)))));
                const cB2: @Vector(16, i16) = @intCast(@as(@Vector(16, u8), @select(u8, hb2 & @as(@Vector(16, u8), @splat(mb)) == @as(@Vector(16, u8), @splat(mb)), z, @as(@Vector(16, u8), @splat(4)))));
                const x0: V = @bitCast(x[i + n * 128 + j * 32 ..][0..16].*);
                const x1: V = @bitCast(x[i + n * 128 + j * 32 + 16 ..][0..16].*);
                ra[(n * 4 + j * 2) % 4] += dlA * @as(V, @floatFromInt(qA1 - cA1)) * x0;
                ra[(n * 4 + j * 2 + 1) % 4] += dlA2 * @as(V, @floatFromInt(qA2 - cA2)) * x1;
                sa[(n * 4 + j * 2) % 4] += dlB * @as(V, @floatFromInt(qB1 - cB1)) * x0;
                sa[(n * 4 + j * 2 + 1) % 4] += dlB2 * @as(V, @floatFromInt(qB2 - cB2)) * x1;
            }
        }
    }
    o0.* = (@reduce(.Add, ra[0]) + @reduce(.Add, ra[1])) + (@reduce(.Add, ra[2]) + @reduce(.Add, ra[3]));
    o1.* = (@reduce(.Add, sa[0]) + @reduce(.Add, sa[1])) + (@reduce(.Add, sa[2]) + @reduce(.Add, sa[3]));
}

// ============================================================================
// Q2_K — súper-bloque 256 elems / 84 bytes: scales[16] @0 (low nibble=esc,
// high=min), qs[64] @16 (2 bits/elem), d:f16 @80, dmin:f16 @82.
// val = dl·q − ml con dl=d·(sc&15), ml=mín·(sc>>4). is continúa entre
// mitades; shift reinicia (mismo patrón que Q3_K).
// ============================================================================

pub fn dequantQ2_K(bytes: []const u8, out: []f32) void {
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : ({
        i += q6_k_block_elems;
        nb += 1;
    }) {
        const nv = @min(q6_k_block_elems, out.len - i);
        const base = nb * q2_k_bytes;
        const d_bits = std.mem.readInt(u16, bytes[base + 80 ..][0..2], .little);
        const mn_bits = std.mem.readInt(u16, bytes[base + 82 ..][0..2], .little);
        const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
        const mn: f32 = @floatCast(@as(f16, @bitCast(mn_bits)));
        const sc: *const [16]u8 = bytes[base..][0..16];
        const qs: *const [64]u8 = bytes[base + 16 ..][0..64];

        var is: usize = 0;
        inline for (0..2) |n| {
            const qb: *const [32]u8 = qs[n * 32 ..][0..32];
            inline for (0..4) |j| {
                const shift: u3 = @intCast(2 * j); // reinicia por mitad
                const sc1 = sc[is];
                is += 1;
                const sc2 = sc[is];
                is += 1;
                const dl: @Vector(16, f32) = @splat(d * @as(f32, @floatFromInt(sc1 & 0xF)));
                const ml: @Vector(16, f32) = @splat(mn * @as(f32, @floatFromInt(sc1 >> 4)));
                const dl2: @Vector(16, f32) = @splat(d * @as(f32, @floatFromInt(sc2 & 0xF)));
                const ml2: @Vector(16, f32) = @splat(mn * @as(f32, @floatFromInt(sc2 >> 4)));
                const q1v: @Vector(16, u8) = qb[0..16].*;
                const q2v: @Vector(16, u8) = qb[16..32].*;
                const q1: @Vector(16, i16) = @intCast((q1v >> @as(@Vector(16, u3), @splat(shift))) & @as(@Vector(16, u8), @splat(3)));
                const q2: @Vector(16, i16) = @intCast((q2v >> @as(@Vector(16, u3), @splat(shift))) & @as(@Vector(16, u8), @splat(3)));
                const o0 = i + n + j * 32;
                const o1 = o0 + 16;
                const v1: [16]f32 = @bitCast(dl * @as(@Vector(16, f32), @floatFromInt(q1)) - ml);
                const v2: [16]f32 = @bitCast(dl2 * @as(@Vector(16, f32), @floatFromInt(q2)) - ml2);
                if (o0 + 16 <= i + nv) {
                    (out[o0..][0..16].*) = v1;
                } else if (o0 < i + nv) {
                    for (0..i + nv - o0) |k| out[o0 + k] = v1[k];
                }
                if (o1 + 16 <= i + nv) {
                    (out[o1..][0..16].*) = v2;
                } else if (o1 < i + nv) {
                    for (0..i + nv - o1) |k| out[o1 + k] = v2[k];
                }
            }
        }
    }
}

/// Escalar de referencia bit-exacto (orden memoria, asociación oráculo).
pub fn dotQ2_KScalar(w: []const u8, x: []const f32) f32 {
    std.debug.assert(x.len % q6_k_block_elems == 0);
    var acc: f32 = 0;
    var i: usize = 0;
    var b: usize = 0;
    while (i < x.len) : ({
        i += q6_k_block_elems;
        b += q2_k_bytes;
    }) {
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w[b + 80 ..][0..2], .little))));
        const mn: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w[b + 82 ..][0..2], .little))));
        const sc: *const [16]u8 = w[b..][0..16];
        const qs: *const [64]u8 = w[b + 16 ..][0..64];
        var is: usize = 0;
        inline for (0..2) |n| {
            const qb: *const [32]u8 = qs[n * 32 ..][0..32];
            inline for (0..4) |j| {
                const shift: u3 = @intCast(2 * j);
                const sc1 = sc[is];
                is += 1;
                const sc2 = sc[is];
                is += 1;
                const dl = d * @as(f32, @floatFromInt(sc1 & 0xF));
                const ml = mn * @as(f32, @floatFromInt(sc1 >> 4));
                const dl2 = d * @as(f32, @floatFromInt(sc2 & 0xF));
                const ml2 = mn * @as(f32, @floatFromInt(sc2 >> 4));
                for (0..16) |l| {
                    const q1: i32 = @intCast((qb[l] >> shift) & 3);
                    const p1 = dl * @as(f32, @floatFromInt(q1)) - ml;
                    acc += p1 * x[i + n + j * 32 + l];
                }
                for (0..16) |l| {
                    const q2: i32 = @intCast((qb[l + 16] >> shift) & 3);
                    const p2 = dl2 * @as(f32, @floatFromInt(q2)) - ml2;
                    acc += p2 * x[i + n + j * 32 + 16 + l];
                }
            }
        }
    }
    return acc;
}

/// Vectorizado 4 cadenas por (mitad, j).
pub fn dotQ2_KSimd(w: []const u8, x: []const f32) f32 {
    std.debug.assert(x.len % q6_k_block_elems == 0);
    const V = @Vector(16, f32);
    var acc: [4]V = .{ @splat(0.0), @splat(0.0), @splat(0.0), @splat(0.0) };
    var i: usize = 0;
    var b: usize = 0;
    while (i < x.len) : ({
        i += q6_k_block_elems;
        b += q2_k_bytes;
    }) {
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w[b + 80 ..][0..2], .little))));
        const mn: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w[b + 82 ..][0..2], .little))));
        const sc: *const [16]u8 = w[b..][0..16];
        const qs: *const [64]u8 = w[b + 16 ..][0..64];
        var is: usize = 0;
        inline for (0..2) |n| {
            const qb: *const [32]u8 = qs[n * 32 ..][0..32];
            inline for (0..4) |j| {
                const shift: u3 = @intCast(2 * j);
                const sc1 = sc[is];
                is += 1;
                const sc2 = sc[is];
                is += 1;
                const dl: V = @splat(d * @as(f32, @floatFromInt(sc1 & 0xF)));
                const ml: V = @splat(mn * @as(f32, @floatFromInt(sc1 >> 4)));
                const dl2: V = @splat(d * @as(f32, @floatFromInt(sc2 & 0xF)));
                const ml2: V = @splat(mn * @as(f32, @floatFromInt(sc2 >> 4)));
                const q1v: @Vector(16, u8) = qb[0..16].*;
                const q2v: @Vector(16, u8) = qb[16..32].*;
                const q1: @Vector(16, i16) = @intCast((q1v >> @as(@Vector(16, u3), @splat(shift))) & @as(@Vector(16, u8), @splat(3)));
                const q2: @Vector(16, i16) = @intCast((q2v >> @as(@Vector(16, u3), @splat(shift))) & @as(@Vector(16, u8), @splat(3)));
                const x0: V = @bitCast(x[i + n + j * 32 ..][0..16].*);
                const x1: V = @bitCast(x[i + n + j * 32 + 16 ..][0..16].*);
                acc[(n * 4 + j * 2) % 4] += dl * @as(V, @floatFromInt(q1)) * x0 - ml * x0;
                acc[(n * 4 + j * 2 + 1) % 4] += dl2 * @as(V, @floatFromInt(q2)) * x1 - ml2 * x1;
            }
        }
    }
    return (@reduce(.Add, acc[0]) + @reduce(.Add, acc[1])) + (@reduce(.Add, acc[2]) + @reduce(.Add, acc[3]));
}

/// Par de filas Q2_K fusionado (comparte cargas de x).
pub fn dotQ2_KPair(w0: []const u8, w1: []const u8, x: []const f32, o0: *f32, o1: *f32) void {
    std.debug.assert(x.len % q6_k_block_elems == 0);
    const V = @Vector(16, f32);
    var ra: [4]V = .{ @splat(0.0), @splat(0.0), @splat(0.0), @splat(0.0) };
    var sa: [4]V = .{ @splat(0.0), @splat(0.0), @splat(0.0), @splat(0.0) };
    var i: usize = 0;
    var b: usize = 0;
    while (i < x.len) : ({
        i += q6_k_block_elems;
        b += q2_k_bytes;
    }) {
        const dA: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w0[b + 80 ..][0..2], .little))));
        const mnA: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w0[b + 82 ..][0..2], .little))));
        const dB: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w1[b + 80 ..][0..2], .little))));
        const mnB: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w1[b + 82 ..][0..2], .little))));
        const scA: *const [16]u8 = w0[b..][0..16];
        const scB: *const [16]u8 = w1[b..][0..16];
        const qsA: *const [64]u8 = w0[b + 16 ..][0..64];
        const qsB: *const [64]u8 = w1[b + 16 ..][0..64];
        var is: usize = 0;
        inline for (0..2) |n| {
            const qa: *const [32]u8 = qsA[n * 32 ..][0..32];
            const qbb: *const [32]u8 = qsB[n * 32 ..][0..32];
            inline for (0..4) |j| {
                const shift: u3 = @intCast(2 * j);
                const sA1 = scA[is];
                const sA2 = scA[is + 1];
                const sB1 = scB[is];
                const sB2 = scB[is + 1];
                is += 2;
                const qa1: @Vector(16, u8) = qa[0..16].*;
                const qa2: @Vector(16, u8) = qa[16..32].*;
                const qb1: @Vector(16, u8) = qbb[0..16].*;
                const qb2: @Vector(16, u8) = qbb[16..32].*;
                const qA1: @Vector(16, i16) = @intCast((qa1 >> @as(@Vector(16, u3), @splat(shift))) & @as(@Vector(16, u8), @splat(3)));
                const qA2: @Vector(16, i16) = @intCast((qa2 >> @as(@Vector(16, u3), @splat(shift))) & @as(@Vector(16, u8), @splat(3)));
                const qB1: @Vector(16, i16) = @intCast((qb1 >> @as(@Vector(16, u3), @splat(shift))) & @as(@Vector(16, u8), @splat(3)));
                const qB2: @Vector(16, i16) = @intCast((qb2 >> @as(@Vector(16, u3), @splat(shift))) & @as(@Vector(16, u8), @splat(3)));
                const x0: V = @bitCast(x[i + n + j * 32 ..][0..16].*);
                const x1: V = @bitCast(x[i + n + j * 32 + 16 ..][0..16].*);
                const dAl: V = @splat(dA * @as(f32, @floatFromInt(sA1 & 0xF)));
                const mAl: V = @splat(mnA * @as(f32, @floatFromInt(sA1 >> 4)));
                const dAh: V = @splat(dA * @as(f32, @floatFromInt(sA2 & 0xF)));
                const mAh: V = @splat(mnA * @as(f32, @floatFromInt(sA2 >> 4)));
                const dBl: V = @splat(dB * @as(f32, @floatFromInt(sB1 & 0xF)));
                const mBl: V = @splat(mnB * @as(f32, @floatFromInt(sB1 >> 4)));
                const dBh: V = @splat(dB * @as(f32, @floatFromInt(sB2 & 0xF)));
                const mBh: V = @splat(mnB * @as(f32, @floatFromInt(sB2 >> 4)));
                ra[(n * 4 + j * 2) % 4] += dAl * @as(V, @floatFromInt(qA1)) * x0 - mAl * x0;
                ra[(n * 4 + j * 2 + 1) % 4] += dAh * @as(V, @floatFromInt(qA2)) * x1 - mAh * x1;
                sa[(n * 4 + j * 2) % 4] += dBl * @as(V, @floatFromInt(qB1)) * x0 - mBl * x0;
                sa[(n * 4 + j * 2 + 1) % 4] += dBh * @as(V, @floatFromInt(qB2)) * x1 - mBh * x1;
            }
        }
    }
    o0.* = (@reduce(.Add, ra[0]) + @reduce(.Add, ra[1])) + (@reduce(.Add, ra[2]) + @reduce(.Add, ra[3]));
    o1.* = (@reduce(.Add, sa[0]) + @reduce(.Add, sa[1])) + (@reduce(.Add, sa[2]) + @reduce(.Add, sa[3]));
}

// ============================================================================
// IQ3_S — súper-bloque 256 elems / 110 bytes, TABLE-DRIVEN: índice de 9 bits
// a iq3s_grid[512]u32 (byte j = entrada>>8j, valores impares 1..15), signos
// por kmask, escalas δ: db = d·(1+2·sc). Implementación SCALAR honesta:
// gather-bound (sin ganancia SIMD real; mismo criterio que llama.cpp IQ).
// Tablas copiadas VERBATIM de src/kv_cache/iq_grids.zig (generadas de
// tables.cuh — no editar a mano).
// ============================================================================

const kmask_iq2xs = [8]u8{ 1, 2, 4, 8, 16, 32, 64, 128 };

const iq3s_grid = [512]u32{
    16843009,  16843011,  16843013,  16843019,  16843023,  16843521,  16843523,  16843525,
    16843529,  16843533,  16844033,  16844035,  16844043,  16844551,  16845057,  16845061,
    16845067,  16845071,  16845571,  16845575,  16846081,  16846085,  16846595,  16846601,
    16846607,  16974081,  16974083,  16974085,  16974089,  16974593,  16974595,  16974603,
    16975105,  16975111,  16975119,  16975619,  16975627,  16976137,  16977155,  16977163,
    16977669,  17105153,  17105155,  17105163,  17105167,  17105665,  17105671,  17105677,
    17106179,  17106187,  17106689,  17106697,  17107205,  17107211,  17107215,  17107715,
    17107719,  17108737,  17108743,  17236231,  17236739,  17236747,  17237249,  17237253,
    17237763,  17237767,  17237773,  17238281,  17238785,  17238789,  17239311,  17239811,
    17239819,  17367297,  17367815,  17367823,  17368323,  17368329,  17368837,  17369345,
    17369351,  17369859,  17370881,  17498373,  17498377,  17499393,  17499397,  17499405,
    17499911,  17500419,  17500427,  17500431,  17501453,  17501959,  17629453,  17629955,
    17629959,  17630979,  17632005,  17633027,  17760513,  17760517,  17760521,  17761537,
    17761541,  17761549,  17762055,  17763073,  17763081,  50397441,  50397443,  50397445,
    50397449,  50397953,  50397955,  50397959,  50397963,  50397967,  50398465,  50398469,
    50398979,  50398985,  50398989,  50400009,  50400013,  50400515,  50401029,  50528513,
    50528515,  50528519,  50528525,  50529025,  50529033,  50529539,  50530049,  50530055,
    50530563,  50531073,  50531077,  50532097,  50532109,  50659585,  50660101,  50660107,
    50660111,  50660609,  50660617,  50661125,  50661633,  50661639,  50662155,  50662657,
    50663173,  50790659,  50790665,  50790671,  50791169,  50791175,  50791683,  50791695,
    50792193,  50792201,  50792707,  50793733,  50794241,  50921735,  50921739,  50922245,
    50922249,  50923267,  50923271,  50923781,  50923789,  50924289,  50924297,  51052803,
    51053313,  51053319,  51053827,  51054337,  51054341,  51055363,  51184897,  51184905,
    51184911,  51185929,  51185933,  51314947,  51314951,  51315457,  51315461,  51315971,
    51316491,  51316995,  51318021,  51318529,  83951873,  83951875,  83951879,  83951883,
    83951887,  83952385,  83952389,  83952393,  83952397,  83952899,  83952903,  83952911,
    83953409,  83953413,  83953923,  83953927,  83953931,  83954433,  83954437,  83954959,
    83955457,  83955463,  83955467,  84082945,  84082949,  84083457,  84083463,  84083471,
    84083973,  84083979,  84084483,  84084489,  84084997,  84085507,  84214019,  84214025,
    84214031,  84215043,  84215047,  84215553,  84215567,  84216067,  84216583,  84216591,
    84217603,  84217609,  84345089,  84345093,  84345099,  84345603,  84346117,  84346121,
    84346627,  84346631,  84347141,  84347649,  84348173,  84476163,  84476175,  84477185,
    84477191,  84477701,  84477707,  84478211,  84479749,  84479755,  84607241,  84607747,
    84608261,  84608783,  84609281,  84609799,  84610817,  84738305,  84738309,  84738319,
    84739331,  84740875,  84741379,  84869387,  84869891,  84870413,  84870913,  84871431,
    84871937,  117506309, 117506819, 117506823, 117506827, 117506831, 117507333, 117507843,
    117507847, 117507851, 117508357, 117508361, 117508367, 117508867, 117509383, 117509891,
    117637379, 117637383, 117637387, 117637897, 117638403, 117638407, 117639425, 117640449,
    117640965, 117640973, 117768449, 117768965, 117769473, 117769989, 117769993, 117771009,
    117899523, 117900033, 117900041, 117900547, 117900551, 117900559, 117901057, 117901571,
    117901575, 117901583, 117902091, 117903111, 118030599, 118031107, 118031117, 118031621,
    118032131, 118033157, 118033665, 118033673, 118161667, 118162177, 118162181, 118162699,
    118163205, 118163721, 118164237, 118165255, 118293261, 118294787, 118423811, 118423815,
    118424833, 118424837, 118425355, 151060737, 151060745, 151061253, 151061761, 151061769,
    151061775, 151062277, 151062787, 151063297, 151064321, 151191813, 151191823, 151192323,
    151192327, 151192837, 151193345, 151193355, 151193863, 151194371, 151194379, 151322883,
    151322887, 151323393, 151323403, 151323907, 151324423, 151324929, 151325455, 151325957,
    151326465, 151453961, 151454467, 151454471, 151454977, 151454981, 151455491, 151455499,
    151585025, 151585029, 151586057, 151586575, 151587073, 151588611, 151716107, 151716111,
    151717123, 151719173, 151847687, 151848713, 151850241, 151978753, 151978763, 151979777,
    151980295, 151980803, 184615173, 184615681, 184615689, 184616197, 184617217, 184617225,
    184617231, 184617733, 184618253, 184618761, 184746243, 184746247, 184746251, 184746757,
    184747267, 184747781, 184749829, 184877313, 184877827, 184878343, 184878849, 184878861,
    184879879, 185008389, 185008399, 185008897, 185009423, 185010441, 185010947, 185011467,
    185011975, 185139459, 185139465, 185140481, 185140997, 185141517, 185271045, 185271565,
    185273091, 185273095, 185403653, 185532677, 185532681, 185533701, 218170115, 218170119,
    218170123, 218171139, 218171143, 218172673, 218300673, 218301697, 218301711, 218303753,
    218432261, 218433289, 218433797, 218434315, 218434821, 218435329, 218562817, 218563337,
    218563843, 218564865, 218694923, 218695943, 218696965, 218824961, 218824967, 218826505,
    218828033, 218956043, 218958081, 219087619, 219087623, 251724033, 251724041, 251724047,
    251725057, 251725061, 251725581, 251726081, 251726601, 251727109, 251855109, 251855619,
    251856137, 251857159, 251857163, 251986179, 251986185, 251986689, 251986701, 251987203,
    251987713, 251988739, 252117253, 252118789, 252118795, 252119815, 252248323, 252248331,
    252248839, 252249345, 252250881, 252380421, 252381445, 252510469, 252512003, 252641537,
};

inline fn idx3s(q: u8, h: u8, l: usize) usize {
    return @as(usize, q) | ((@as(usize, h) << @as(u6, @intCast(8 - 2 * l))) & 256);
}

inline fn idx3s2(q: u8, h: u8, l: usize) usize {
    return @as(usize, q) | ((@as(usize, h) << @as(u6, @intCast(7 - 2 * l))) & 256);
}

inline fn gridByte(entry: u32, j: usize) f32 {
    return @floatFromInt(@as(u8, @intCast((entry >> @intCast(8 * j)) & 0xFF)));
}

pub fn dequantIq3_s(bytes: []const u8, out: []f32) void {
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : ({
        i += q6_k_block_elems;
        nb += 1;
    }) {
        const nv = @min(q6_k_block_elems, out.len - i);
        const base = nb * iq3_s_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const qs = bytes[base + 2 .. base + 66];
        const qh = bytes[base + 66 .. base + 74];
        const signs = bytes[base + 74 .. base + 106];
        const scales = bytes[base + 106 .. base + 110];
        for (0..q6_k_block_elems / 64) |it| {
            const sc = scales[it];
            const db1 = d * (1 + 2 * @as(f32, @floatFromInt(sc & 0xf)));
            const db2 = d * (1 + 2 * @as(f32, @floatFromInt(sc >> 4)));
            const q0 = qs[it * 16 ..][0..8];
            const q1 = qs[it * 16 + 8 ..][0..8];
            for (0..4) |l| {
                const e1 = iq3s_grid[idx3s(q0[2 * l], qh[2 * it], l)];
                const e2 = iq3s_grid[idx3s2(q0[2 * l + 1], qh[2 * it], l)];
                const sm = signs[it * 8 + l];
                for (0..4) |j| {
                    const s1: f32 = if (sm & kmask_iq2xs[j] != 0) -1 else 1;
                    const s2: f32 = if (sm & kmask_iq2xs[j + 4] != 0) -1 else 1;
                    const p0 = i + it * 64 + l * 8 + j;
                    const p1 = p0 + 4;
                    if (p0 < i + nv) out[p0] = db1 * gridByte(e1, j) * s1;
                    if (p1 < i + nv) out[p1] = db1 * gridByte(e2, j) * s2;
                }
            }
            for (0..4) |l| {
                const e1 = iq3s_grid[idx3s(q1[2 * l], qh[2 * it + 1], l)];
                const e2 = iq3s_grid[idx3s2(q1[2 * l + 1], qh[2 * it + 1], l)];
                const sm = signs[it * 8 + 4 + l];
                for (0..4) |j| {
                    const s1: f32 = if (sm & kmask_iq2xs[j] != 0) -1 else 1;
                    const s2: f32 = if (sm & kmask_iq2xs[j + 4] != 0) -1 else 1;
                    const p0 = i + it * 64 + 32 + l * 8 + j;
                    const p1 = p0 + 4;
                    if (p0 < i + nv) out[p0] = db2 * gridByte(e1, j) * s1;
                    if (p1 < i + nv) out[p1] = db2 * gridByte(e2, j) * s2;
                }
            }
        }
    }
}

/// Dot IQ3_S: recorrido en orden memoria (it → l → mitad → j) acumulando
/// val·x — bit-exacto contra dequant-oráculo + producto secuencial.
pub fn dotIq3_sScalar(w: []const u8, x: []const f32) f32 {
    std.debug.assert(x.len % q6_k_block_elems == 0);
    var acc: f32 = 0;
    var i: usize = 0;
    var b: usize = 0;
    while (i < x.len) : ({
        i += q6_k_block_elems;
        b += iq3_s_bytes;
    }) {
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w[b..][0..2], .little))));
        const qs = w[b + 2 .. b + 66];
        const qh = w[b + 66 .. b + 74];
        const signs = w[b + 74 .. b + 106];
        const scales = w[b + 106 .. b + 110];
        for (0..q6_k_block_elems / 64) |it| {
            const sc = scales[it];
            const db1 = d * (1 + 2 * @as(f32, @floatFromInt(sc & 0xf)));
            const db2 = d * (1 + 2 * @as(f32, @floatFromInt(sc >> 4)));
            const q0 = qs[it * 16 ..][0..8];
            const q1 = qs[it * 16 + 8 ..][0..8];
            inline for (0..2) |half| {
                const qq = if (half == 0) q0 else q1;
                const db = if (half == 0) db1 else db2;
                const hh = qh[2 * it + half];
                const sbase = it * 64 + half * 32;
                for (0..4) |l| {
                    const e1 = iq3s_grid[idx3s(qq[2 * l], hh, l)];
                    const e2 = iq3s_grid[idx3s2(qq[2 * l + 1], hh, l)];
                    const sm = signs[it * 8 + half * 4 + l];
                    for (0..4) |j| {
                        const s1: f32 = if (sm & kmask_iq2xs[j] != 0) -1 else 1;
                        const s2: f32 = if (sm & kmask_iq2xs[j + 4] != 0) -1 else 1;
                        acc += db * gridByte(e1, j) * s1 * x[i + sbase + l * 8 + j];
                        acc += db * gridByte(e2, j) * s2 * x[i + sbase + l * 8 + j + 4];
                    }
                }
            }
        }
    }
    return acc;
}

/// Dot IQ2_S: similar a IQ3_S pero con layout diferente (82B/SB).
/// IQ2_S: d@0, qs[64]@2, signs[32]@34, qh[8]@66, scales[8]@74. (82B/SB 256)
pub fn dotIq2_SScalar(w: []const u8, x: []const f32) f32 {
    std.debug.assert(x.len % q6_k_block_elems == 0);
    var acc: f32 = 0;
    var i: usize = 0;
    var b: usize = 0;
    while (i < x.len) : ({
        i += q6_k_block_elems;
        b += 82;
    }) {
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w[b..][0..2], .little))));
        const qs = w[b + 2 .. b + 34];
        const signs = w[b + 34 .. b + 66];
        const qh = w[b + 66 .. b + 74];
        const scales = w[b + 74 .. b + 82];
        for (0..q6_k_block_elems / 32) |ib| {
            const db0 = d * (0.5 + @as(f32, @floatFromInt(scales[ib] & 0xF))) * 0.25;
            const db1 = d * (0.5 + @as(f32, @floatFromInt(scales[ib] >> 4))) * 0.25;
            for (0..4) |l| {
                const idx: usize = qs[ib * 4 + l] | ((@as(usize, qh[ib]) << @intCast(8 - 2 * l)) & 0x300);
                const g = iq_grids.iq2s_grid[idx];
                const db = if (l < 2) db0 else db1;
                for (0..8) |j| {
                    const s: f32 = if (signs[ib * 4 + l] & kmask_iq2xs[j] != 0) -1 else 1;
                    const gv: f32 = @floatFromInt(@as(i8, @bitCast(@as(u8, @intCast((g >> @intCast(8 * j)) & 0xFF)))));
                    acc += db * gv * s * x[i + ib * 32 + l * 8 + j];
                }
            }
        }
    }
    return acc;
}

/// Dispatch ISA (auto = vectorizado; scalar = referencia lenta para debug).
pub fn dot(fmt: Format, w: []const u8, x: []const f32) f32 {
    if (resolveIsa() == .scalar) return dotScalar(fmt, w, x);
    return switch (fmt) {
        .q4_0 => dotQ4_0Simd(w, x),
        .q8_0 => dotQ8_0Simd(w, x),
        .q6_k => dotQ6_KSimd(w, x),
        .q4_1 => dotQ4_1Simd(w, x),
        .q4_k => dotQ4_KSimd(w, x),
        .q5_k => dotQ5_KSimd(w, x),
        .q3_k => dotQ3_KSimd(w, x),
        .q2_k => dotQ2_KSimd(w, x),
        .iq3_s => dotIq3_sScalar(w, x), // gather-bound: sin versión SIMD
        .iq2_s => dotIq2_SScalar(w, x), // gather-bound: sin versión SIMD
        .iq4_nl => dotIq4_nlScalar(w, x), // LUT-bound: sin versión SIMD
        .iq2_xxs => dotIq2_xxsScalar(w, x), // gather-bound: sin versión SIMD
    };
}

/// Par de filas Q4_0 fusionado (misma idea que dotQ8_0Pair).
pub fn dotQ4_0Pair(w0: []const u8, w1: []const u8, x: []const f32, o0: *f32, o1: *f32) void {
    std.debug.assert(x.len % block_elems == 0);
    const V = @Vector(16, f32);
    var ra0: V = @splat(0.0);
    var ra1: V = @splat(0.0);
    var rb0: V = @splat(0.0);
    var rb1: V = @splat(0.0);
    var sa0: V = @splat(0.0);
    var sa1: V = @splat(0.0);
    var sb0: V = @splat(0.0);
    var sb1: V = @splat(0.0);
    const e: @Vector(16, i16) = @splat(8);
    var i: usize = 0;
    var b: usize = 0;
    while (i + 64 <= x.len) : ({
        i += 64;
        b += 36;
    }) {
        const dA: V = @splat(@floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w0[b..][0..2], .little)))));
        const dA2: V = @splat(@floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w1[b..][0..2], .little)))));
        const qvA0: @Vector(16, u8) = w0[b + 2 ..][0..16].*;
        const qvA1: @Vector(16, u8) = w1[b + 2 ..][0..16].*;
        const dBlk = b + q4_0_bytes;
        const dB: V = @splat(@floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w0[dBlk..][0..2], .little)))));
        const dB2: V = @splat(@floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w1[dBlk..][0..2], .little)))));
        const qvB0: @Vector(16, u8) = w0[dBlk + 2 ..][0..16].*;
        const qvB1: @Vector(16, u8) = w1[dBlk + 2 ..][0..16].*;
        const loR0: V = @floatFromInt(@as(@Vector(16, i16), @intCast(qvA0 & @as(@Vector(16, u8), @splat(0x0F)))) - e);
        const hiR0: V = @floatFromInt(@as(@Vector(16, i16), @intCast(qvA0 >> @as(@Vector(16, u3), @splat(4)))) - e);
        const loR1: V = @floatFromInt(@as(@Vector(16, i16), @intCast(qvA1 & @as(@Vector(16, u8), @splat(0x0F)))) - e);
        const hiR1: V = @floatFromInt(@as(@Vector(16, i16), @intCast(qvA1 >> @as(@Vector(16, u3), @splat(4)))) - e);
        const loS0: V = @floatFromInt(@as(@Vector(16, i16), @intCast(qvB0 & @as(@Vector(16, u8), @splat(0x0F)))) - e);
        const hiS0: V = @floatFromInt(@as(@Vector(16, i16), @intCast(qvB0 >> @as(@Vector(16, u3), @splat(4)))) - e);
        const loS1: V = @floatFromInt(@as(@Vector(16, i16), @intCast(qvB1 & @as(@Vector(16, u8), @splat(0x0F)))) - e);
        const hiS1: V = @floatFromInt(@as(@Vector(16, i16), @intCast(qvB1 >> @as(@Vector(16, u3), @splat(4)))) - e);
        const x0: V = @bitCast(x[i..][0..16].*);
        const x1: V = @bitCast(x[i + 16 ..][0..16].*);
        const x2: V = @bitCast(x[i + 32 ..][0..16].*);
        const x3: V = @bitCast(x[i + 48 ..][0..16].*);
        ra0 += (dA * loR0) * x0;
        ra1 += (dA * hiR0) * x1;
        rb0 += (dB * loS0) * x2;
        rb1 += (dB * hiS0) * x3;
        sa0 += (dA2 * loR1) * x0;
        sa1 += (dA2 * hiR1) * x1;
        sb0 += (dB2 * loS1) * x2;
        sb1 += (dB2 * hiS1) * x3;
    }
    if (i < x.len) {
        o0.* = (@reduce(.Add, ra0) + @reduce(.Add, ra1)) + (@reduce(.Add, rb0) + @reduce(.Add, rb1)) + dotQ4_0Tail(w0[b..], x[i..]);
        o1.* = dotQ4_0Simd(w1, x);
        return;
    }
    o0.* = (@reduce(.Add, ra0) + @reduce(.Add, ra1)) + (@reduce(.Add, rb0) + @reduce(.Add, rb1));
    o1.* = (@reduce(.Add, sa0) + @reduce(.Add, sa1)) + (@reduce(.Add, sb0) + @reduce(.Add, sb1));
}

fn dotQ4_0Tail(wb: []const u8, xb: []const f32) f32 {
    const V = @Vector(16, f32);
    const d: V = @splat(@floatCast(@as(f16, @bitCast(std.mem.readInt(u16, wb[0..2], .little)))));
    const qv: @Vector(16, u8) = wb[2..18].*;
    const e: @Vector(16, i16) = @splat(8);
    const loq: V = @floatFromInt(@as(@Vector(16, i16), @intCast(qv & @as(@Vector(16, u8), @splat(0x0F)))) - e);
    const hiq: V = @floatFromInt(@as(@Vector(16, i16), @intCast(qv >> @as(@Vector(16, u3), @splat(4)))) - e);
    return (@reduce(.Add, (d * loq) * @as(V, @bitCast(xb[0..16].*))) +
        @reduce(.Add, (d * hiq) * @as(V, @bitCast(xb[16..32].*))));
}

/// GEMV fila-major empaquetada: out[r] = dot(W[r], x).
/// `w` contiene `out.len` filas contiguas de `fmt.rowBytes(n)` bytes cada una.
/// Recorre en PARES fusionados (comparten cargas de x); determinista porque
/// cada fila se acumula en sus propias cadenas.
pub fn gemv(fmt: Format, w: []const u8, n: usize, x: []const f32, out: []f32) void {
    const rb = fmt.rowBytes(n);
    std.debug.assert(w.len >= out.len * rb);
    if (resolveIsa() == .scalar) {
        for (out, 0..) |*o, r| o.* = dotScalar(fmt, w[r * rb ..][0..rb], x);
        return;
    }
    var r: usize = 0;
    while (r + 2 <= out.len) : (r += 2) {
        switch (fmt) {
            .q4_0 => dotQ4_0Pair(w[r * rb ..][0..rb], w[(r + 1) * rb ..][0..rb], x, &out[r], &out[r + 1]),
            .q8_0 => dotQ8_0Pair(w[r * rb ..][0..rb], w[(r + 1) * rb ..][0..rb], x, &out[r], &out[r + 1]),
            .q6_k => dotQ6_KPair(w[r * rb ..][0..rb], w[(r + 1) * rb ..][0..rb], x, &out[r], &out[r + 1]),
            .q4_1 => dotQ4_1Pair(w[r * rb ..][0..rb], w[(r + 1) * rb ..][0..rb], x, &out[r], &out[r + 1]),
            .q4_k => dotQ4_KPair(w[r * rb ..][0..rb], w[(r + 1) * rb ..][0..rb], x, &out[r], &out[r + 1]),
            .q5_k => dotQ5_KPair(w[r * rb ..][0..rb], w[(r + 1) * rb ..][0..rb], x, &out[r], &out[r + 1]),
            .q3_k => dotQ3_KPair(w[r * rb ..][0..rb], w[(r + 1) * rb ..][0..rb], x, &out[r], &out[r + 1]),
            .q2_k => dotQ2_KPair(w[r * rb ..][0..rb], w[(r + 1) * rb ..][0..rb], x, &out[r], &out[r + 1]),
            .iq3_s => {
                out[r] = dotIq3_sScalar(w[r * rb ..][0..rb], x);
                out[r + 1] = dotIq3_sScalar(w[(r + 1) * rb ..][0..rb], x);
            },
            .iq2_s => {
                out[r] = dotIq2_SScalar(w[r * rb ..][0..rb], x);
                out[r + 1] = dotIq2_SScalar(w[(r + 1) * rb ..][0..rb], x);
            },
            .iq4_nl => {
                out[r] = dotIq4_nlScalar(w[r * rb ..][0..rb], x);
                out[r + 1] = dotIq4_nlScalar(w[(r + 1) * rb ..][0..rb], x);
            },
            .iq2_xxs => {
                out[r] = dotIq2_xxsScalar(w[r * rb ..][0..rb], x);
                out[r + 1] = dotIq2_xxsScalar(w[(r + 1) * rb ..][0..rb], x);
            },
        }
    }
    if (r < out.len) out[r] = switch (fmt) {
        .q4_0 => dotQ4_0Simd(w[r * rb ..][0..rb], x),
        .q8_0 => dotQ8_0Simd(w[r * rb ..][0..rb], x),
        .q6_k => dotQ6_KSimd(w[r * rb ..][0..rb], x),
        .q4_1 => dotQ4_1Simd(w[r * rb ..][0..rb], x),
        .q4_k => dotQ4_KSimd(w[r * rb ..][0..rb], x),
        .q5_k => dotQ5_KSimd(w[r * rb ..][0..rb], x),
        .q3_k => dotQ3_KSimd(w[r * rb ..][0..rb], x),
        .q2_k => dotQ2_KSimd(w[r * rb ..][0..rb], x),
        .iq3_s => dotIq3_sScalar(w[r * rb ..][0..rb], x),
        .iq2_s => dotIq2_SScalar(w[r * rb ..][0..rb], x),
        .iq4_nl => dotIq4_nlScalar(w[r * rb ..][0..rb], x),
        .iq2_xxs => dotIq2_xxsScalar(w[r * rb ..][0..rb], x),
    };
}

// ============================================================================
// 4.6 — IQ4_NL + IQ2_XXS (gemma-4 IQ2_XXS: down=iq4_nl, gate/up=iq2_xxs).
// Espejos EXACTOS de gguf.dequantIq4_nl / dequantIq2_xxs; mismo orden de
// memoria en los dot para paridad bit-exacta con el oráculo.
// ============================================================================

const kv_quant = @import("kv_cache").kv_quant;

/// IQ4_NL: bloques de 32 elems / 18 bytes. d:f16@0, qs[16] nibbles@2.
/// val = d * kvalues_iq4nl[q] (LUT de 16 valores no lineales).
pub fn dequantIq4_nl(bytes: []const u8, out: []f32) void {
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : ({
        i += block_elems;
        nb += 1;
    }) {
        const nv = @min(block_elems, out.len - i);
        const base = nb * iq4_nl_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const qs = bytes[base + 2 .. base + 18];
        for (0..block_elems / 2) |j| {
            const kv_lo = kv_quant.kvalues_iq4nl[qs[j] & 0xF];
            const kv_hi = kv_quant.kvalues_iq4nl[qs[j] >> 4];
            if (i + j < i + nv) out[i + j] = d * @as(f32, @floatFromInt(kv_lo));
            if (i + j + 16 < i + nv) out[i + j + 16] = d * @as(f32, @floatFromInt(kv_hi));
        }
    }
}

/// IQ2_XXS: súper-bloques de 256 elems / 66 bytes. d:f16@0, qs[32]@2.
/// 8 sub-bloques de 32: cada uno 8B de qs = [aux32_0 (4 índices grid)]
/// + [aux32_1 (scale 4 bits @28 + 28 bits de signos)].
/// db = d * (0.5 + (aux32_1 >> 28)) * 0.25
/// val = db * byte_j(iq2xxs_grid[idx]) * sign   (signs = ksigns_iq2xs[7-bit idx]).
pub fn dequantIq2_xxs(bytes: []const u8, out: []f32) void {
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : ({
        i += q6_k_block_elems;
        nb += 1;
    }) {
        const nv = @min(q6_k_block_elems, out.len - i);
        const base = nb * iq2_xxs_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const qs = bytes[base + 2 .. base + 66];
        for (0..q6_k_block_elems / 32) |ib| {
            const aux32_0 = std.mem.readInt(u32, qs[ib * 8 ..][0..4], .little);
            const aux32_1 = std.mem.readInt(u32, qs[ib * 8 + 4 ..][0..4], .little);
            const db = d * (0.5 + @as(f32, @floatFromInt(aux32_1 >> 28))) * 0.25;
            for (0..4) |l| {
                const idx: usize = @intCast((aux32_0 >> @intCast(8 * l)) & 0xFF);
                const signs = iq_grids.ksigns_iq2xs[@intCast((aux32_1 >> @intCast(7 * l)) & 127)];
                const g = iq_grids.iq2xxs_grid[idx];
                for (0..8) |j| {
                    const sv: f32 = if (signs & kmask_iq2xs[j] != 0) -1 else 1;
                    const gv: f32 = @floatFromInt(@as(u8, @intCast((g >> @intCast(8 * j)) & 0xFF)));
                    const pos = i + ib * 32 + l * 8 + j;
                    if (pos < i + nv) out[pos] = db * gv * sv;
                }
            }
        }
    }
}

/// IQ2_S dequant (espejo de gguf.dequantIq2_s; el dot ya existía).
pub fn dequantIq2_S(bytes: []const u8, out: []f32) void {
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : ({
        i += q6_k_block_elems;
        nb += 1;
    }) {
        const nv = @min(q6_k_block_elems, out.len - i);
        const base = nb * 82;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const qs = bytes[base + 2 .. base + 34];
        const signs = bytes[base + 34 .. base + 66];
        const qh = bytes[base + 66 .. base + 74];
        const scales = bytes[base + 74 .. base + 82];
        for (0..q6_k_block_elems / 32) |ib| {
            const db0 = d * (0.5 + @as(f32, @floatFromInt(scales[ib] & 0xF))) * 0.25;
            const db1 = d * (0.5 + @as(f32, @floatFromInt(scales[ib] >> 4))) * 0.25;
            for (0..4) |l| {
                const idx: usize = qs[ib * 4 + l] | ((@as(usize, qh[ib]) << @intCast(8 - 2 * l)) & 0x300);
                const g = iq_grids.iq2s_grid[idx];
                const db = if (l < 2) db0 else db1;
                for (0..8) |j| {
                    const sv: f32 = if (signs[ib * 4 + l] & kmask_iq2xs[j] != 0) -1 else 1;
                    const gv: f32 = @floatFromInt(@as(u8, @intCast((g >> @intCast(8 * j)) & 0xFF)));
                    const pos = i + ib * 32 + l * 8 + j;
                    if (pos < i + nv) out[pos] = db * gv * sv;
                }
            }
        }
    }
}

/// Dot IQ4_NL escalar (orden de memoria = oráculo secuencial).
pub fn dotIq4_nlScalar(w: []const u8, x: []const f32) f32 {
    std.debug.assert(x.len % block_elems == 0);
    var acc: f32 = 0;
    var i: usize = 0;
    var b: usize = 0;
    while (i < x.len) : ({
        i += block_elems;
        b += iq4_nl_bytes;
    }) {
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w[b..][0..2], .little))));
        const qs = w[b + 2 .. b + 18];
        for (0..block_elems / 2) |j| {
            const kv_lo = kv_quant.kvalues_iq4nl[qs[j] & 0xF];
            const kv_hi = kv_quant.kvalues_iq4nl[qs[j] >> 4];
            acc += d * @as(f32, @floatFromInt(kv_lo)) * x[i + j];
            acc += d * @as(f32, @floatFromInt(kv_hi)) * x[i + j + 16];
        }
    }
    return acc;
}

/// Dot IQ2_XXS escalar (orden de memoria = oráculo secuencial).
pub fn dotIq2_xxsScalar(w: []const u8, x: []const f32) f32 {
    std.debug.assert(x.len % q6_k_block_elems == 0);
    var acc: f32 = 0;
    var i: usize = 0;
    var b: usize = 0;
    while (i < x.len) : ({
        i += q6_k_block_elems;
        b += iq2_xxs_bytes;
    }) {
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, w[b..][0..2], .little))));
        const qs = w[b + 2 .. b + 66];
        for (0..q6_k_block_elems / 32) |ib| {
            const aux32_0 = std.mem.readInt(u32, qs[ib * 8 ..][0..4], .little);
            const aux32_1 = std.mem.readInt(u32, qs[ib * 8 + 4 ..][0..4], .little);
            const db = d * (0.5 + @as(f32, @floatFromInt(aux32_1 >> 28))) * 0.25;
            for (0..4) |l| {
                const idx: usize = @intCast((aux32_0 >> @intCast(8 * l)) & 0xFF);
                const signs = iq_grids.ksigns_iq2xs[@intCast((aux32_1 >> @intCast(7 * l)) & 127)];
                const g = iq_grids.iq2xxs_grid[idx];
                for (0..8) |j| {
                    const sv: f32 = if (signs & kmask_iq2xs[j] != 0) -1 else 1;
                    const gv: f32 = @floatFromInt(@as(u8, @intCast((g >> @intCast(8 * j)) & 0xFF)));
                    acc += db * gv * sv * x[i + ib * 32 + l * 8 + j];
                }
            }
        }
    }
    return acc;
}

test "parseIsa acepta scalar y defaultea auto" {
    try std.testing.expectEqual(IsaMode.scalar, parseIsa("scalar"));
    try std.testing.expectEqual(IsaMode.auto, parseIsa("auto"));
    try std.testing.expectEqual(IsaMode.auto, parseIsa("basura"));
    try std.testing.expectEqual(IsaMode.auto, parseIsa(null));
}

test "rowBytes formatos" {
    try std.testing.expectEqual(@as(usize, 2304), Format.q4_0.rowBytes(4096));
    try std.testing.expectEqual(@as(usize, 4352), Format.q8_0.rowBytes(4096));
}

/// Q6_K vectorizado: por chunk de 16 l's produce 4 vectores (subgrupos +0,
/// +32, +64, +96); 4 cadenas de acumulación independientes por par de chunks.
pub fn dotQ6_KSimd(w: []const u8, x: []const f32) f32 {
    std.debug.assert(x.len % q6_k_block_elems == 0);
    const V = @Vector(16, f32);
    var a0: V = @splat(0.0);
    var a1: V = @splat(0.0);
    var a2: V = @splat(0.0);
    var a3: V = @splat(0.0);
    var i: usize = 0;
    var b: usize = 0;
    while (i < x.len) : ({
        i += q6_k_block_elems;
        b += q6_k_block_bytes;
    }) {
        const d_bits = std.mem.readInt(u16, w[b + 208 ..][0..2], .little);
        const ds: V = @splat(@floatCast(@as(f16, @bitCast(d_bits))));
        inline for (0..2) |h| {
            const ql2: *const [64]u8 = w[b + h * 64 ..][0..64];
            const qh2: *const [32]u8 = w[b + 128 + h * 32 ..][0..32];
            const sc2: *const [8]u8 = w[b + 192 + h * 8 ..][0..8];
            const off = i + h * 128;
            inline for (0..2) |lc| {
                const l0 = lc * 16;
                const is = l0 / 16;
                const e32: @Vector(16, i16) = @splat(32);
                const nib_lo: @Vector(16, u8) = ql2[l0..][0..16].*;
                const nib_hi: @Vector(16, u8) = ql2[l0 + 32 ..][0..16].*;
                const hb: @Vector(16, u8) = qh2[l0..][0..16].*;
                const F = @Vector(16, u8);
                const S3 = @Vector(16, u3);
                const m0: @Vector(16, i16) = @intCast((nib_lo & @as(F, @splat(0x0F))) | ((hb >> @as(S3, @splat(0))) & @as(F, @splat(3))) << @as(S3, @splat(4)));
                const m1: @Vector(16, i16) = @intCast((nib_hi & @as(F, @splat(0x0F))) | ((hb >> @as(S3, @splat(2))) & @as(F, @splat(3))) << @as(S3, @splat(4)));
                const m2: @Vector(16, i16) = @intCast((nib_lo >> @as(S3, @splat(4))) | ((hb >> @as(S3, @splat(4))) & @as(F, @splat(3))) << @as(S3, @splat(4)));
                const m3: @Vector(16, i16) = @intCast((nib_hi >> @as(S3, @splat(4))) | ((hb >> @as(S3, @splat(6))) & @as(F, @splat(3))) << @as(S3, @splat(4)));
                const s0: V = @splat(ds[0] * @as(f32, @floatFromInt(@as(i8, @bitCast(sc2[is + 0])))));
                const s1: V = @splat(ds[0] * @as(f32, @floatFromInt(@as(i8, @bitCast(sc2[is + 2])))));
                const s2: V = @splat(ds[0] * @as(f32, @floatFromInt(@as(i8, @bitCast(sc2[is + 4])))));
                const s3: V = @splat(ds[0] * @as(f32, @floatFromInt(@as(i8, @bitCast(sc2[is + 6])))));
                a0 += s0 * @as(V, @floatFromInt(m0 - e32)) * @as(V, @bitCast(x[off + l0 ..][0..16].*));
                a1 += s1 * @as(V, @floatFromInt(m1 - e32)) * @as(V, @bitCast(x[off + l0 + 32 ..][0..16].*));
                a2 += s2 * @as(V, @floatFromInt(m2 - e32)) * @as(V, @bitCast(x[off + l0 + 64 ..][0..16].*));
                a3 += s3 * @as(V, @floatFromInt(m3 - e32)) * @as(V, @bitCast(x[off + l0 + 96 ..][0..16].*));
            }
        }
    }
    return (@reduce(.Add, a0) + @reduce(.Add, a1)) + (@reduce(.Add, a2) + @reduce(.Add, a3));
}

/// Par de filas Q6_K fusionado (comparte cargas de x entre ambas filas).
pub fn dotQ6_KPair(w0: []const u8, w1: []const u8, x: []const f32, o0: *f32, o1: *f32) void {
    std.debug.assert(x.len % q6_k_block_elems == 0);
    const V = @Vector(16, f32);
    var ra0: V = @splat(0.0);
    var ra1: V = @splat(0.0);
    var rb0: V = @splat(0.0);
    var rb1: V = @splat(0.0);
    var sa0: V = @splat(0.0);
    var sa1: V = @splat(0.0);
    var sb0: V = @splat(0.0);
    var sb1: V = @splat(0.0);
    var i: usize = 0;
    var b: usize = 0;
    while (i < x.len) : ({
        i += q6_k_block_elems;
        b += q6_k_block_bytes;
    }) {
        const dA_bits = std.mem.readInt(u16, w0[b + 208 ..][0..2], .little);
        const dB_bits = std.mem.readInt(u16, w1[b + 208 ..][0..2], .little);
        const dA: V = @splat(@floatCast(@as(f16, @bitCast(dA_bits))));
        const dB: V = @splat(@floatCast(@as(f16, @bitCast(dB_bits))));
        inline for (0..2) |h| {
            const off = i + h * 128;
            inline for (0..2) |lc| {
                const l0 = lc * 16;
                const is = l0 / 16;
                const e32: @Vector(16, i16) = @splat(32);
                const F = @Vector(16, u8);
                const S3 = @Vector(16, u3);
                // fila A
                const qlA: *const [64]u8 = w0[b + h * 64 ..][0..64];
                const qhA: *const [32]u8 = w0[b + 128 + h * 32 ..][0..32];
                const scA: *const [8]u8 = w0[b + 192 + h * 8 ..][0..8];
                // fila B
                const qlB: *const [64]u8 = w1[b + h * 64 ..][0..64];
                const qhB: *const [32]u8 = w1[b + 128 + h * 32 ..][0..32];
                const scB: *const [8]u8 = w1[b + 192 + h * 8 ..][0..8];

                const nloA: F = qlA[l0..][0..16].*;
                const nhiA: F = qlA[l0 + 32 ..][0..16].*;
                const hbA: F = qhA[l0..][0..16].*;
                const nloB: F = qlB[l0..][0..16].*;
                const nhiB: F = qlB[l0 + 32 ..][0..16].*;
                const hbB: F = qhB[l0..][0..16].*;

                const rA0: V = @floatFromInt(@as(@Vector(16, i16), @intCast((nloA & @as(F, @splat(0x0F))) | ((hbA >> @as(S3, @splat(0))) & @as(F, @splat(3))) << @as(S3, @splat(4)))) - e32);
                const rA1: V = @floatFromInt(@as(@Vector(16, i16), @intCast((nhiA & @as(F, @splat(0x0F))) | ((hbA >> @as(S3, @splat(2))) & @as(F, @splat(3))) << @as(S3, @splat(4)))) - e32);
                const rA2: V = @floatFromInt(@as(@Vector(16, i16), @intCast((nloA >> @as(S3, @splat(4))) | ((hbA >> @as(S3, @splat(4))) & @as(F, @splat(3))) << @as(S3, @splat(4)))) - e32);
                const rA3: V = @floatFromInt(@as(@Vector(16, i16), @intCast((nhiA >> @as(S3, @splat(4))) | ((hbA >> @as(S3, @splat(6))) & @as(F, @splat(3))) << @as(S3, @splat(4)))) - e32);
                const rB0: V = @floatFromInt(@as(@Vector(16, i16), @intCast((nloB & @as(F, @splat(0x0F))) | ((hbB >> @as(S3, @splat(0))) & @as(F, @splat(3))) << @as(S3, @splat(4)))) - e32);
                const rB1: V = @floatFromInt(@as(@Vector(16, i16), @intCast((nhiB & @as(F, @splat(0x0F))) | ((hbB >> @as(S3, @splat(2))) & @as(F, @splat(3))) << @as(S3, @splat(4)))) - e32);
                const rB2: V = @floatFromInt(@as(@Vector(16, i16), @intCast((nloB >> @as(S3, @splat(4))) | ((hbB >> @as(S3, @splat(4))) & @as(F, @splat(3))) << @as(S3, @splat(4)))) - e32);
                const rB3: V = @floatFromInt(@as(@Vector(16, i16), @intCast((nhiB >> @as(S3, @splat(4))) | ((hbB >> @as(S3, @splat(6))) & @as(F, @splat(3))) << @as(S3, @splat(4)))) - e32);

                const sA0: V = @splat(dA[0] * @as(f32, @floatFromInt(@as(i8, @bitCast(scA[is + 0])))));
                const sA1: V = @splat(dA[0] * @as(f32, @floatFromInt(@as(i8, @bitCast(scA[is + 2])))));
                const sA2: V = @splat(dA[0] * @as(f32, @floatFromInt(@as(i8, @bitCast(scA[is + 4])))));
                const sA3: V = @splat(dA[0] * @as(f32, @floatFromInt(@as(i8, @bitCast(scA[is + 6])))));
                const sB0: V = @splat(dB[0] * @as(f32, @floatFromInt(@as(i8, @bitCast(scB[is + 0])))));
                const sB1: V = @splat(dB[0] * @as(f32, @floatFromInt(@as(i8, @bitCast(scB[is + 2])))));
                const sB2: V = @splat(dB[0] * @as(f32, @floatFromInt(@as(i8, @bitCast(scB[is + 4])))));
                const sB3: V = @splat(dB[0] * @as(f32, @floatFromInt(@as(i8, @bitCast(scB[is + 6])))));

                const x0: V = @bitCast(x[off + l0 ..][0..16].*);
                const x1: V = @bitCast(x[off + l0 + 32 ..][0..16].*);
                const x2: V = @bitCast(x[off + l0 + 64 ..][0..16].*);
                const x3: V = @bitCast(x[off + l0 + 96 ..][0..16].*);

                ra0 += sA0 * rA0 * x0;
                ra1 += sA1 * rA1 * x1;
                rb0 += sA2 * rA2 * x2;
                rb1 += sA3 * rA3 * x3;
                sa0 += sB0 * rB0 * x0;
                sa1 += sB1 * rB1 * x1;
                sb0 += sB2 * rB2 * x2;
                sb1 += sB3 * rB3 * x3;
            }
        }
    }
    o0.* = (@reduce(.Add, ra0) + @reduce(.Add, ra1)) + (@reduce(.Add, rb0) + @reduce(.Add, rb1));
    o1.* = (@reduce(.Add, sa0) + @reduce(.Add, sa1)) + (@reduce(.Add, sb0) + @reduce(.Add, sb1));
}
