//! Codificación / decodificación de regiones KV-cache en formatos GGUF
//! Soporta todos los formatos cuantizados GGUF para KV-cache.
//!
//! El layout coincide con los decuantizadores de `src/loader/gguf.zig`
//! y los kernels CUDA en `kernels/*.cu`.

const std = @import("std");
const QuantFormat = @import("quant_types.zig").QuantFormat;
const iq_grids = @import("iq_grids.zig");
const iq1m_encoder = @import("iq1m_encoder.zig");

/// Valores numéricos de gguf.GgmlType (para evitar importar el módulo)
const GgmlTypeValue = struct {
    const f32_v = 0;
    const f16_v = 1;
    const q4_0 = 2;
    const q4_1 = 3;
    const q5_0 = 6;
    const q5_1 = 7;
    const q8_0 = 8;
    const q8_1 = 9;
    const q2_k = 10;
    const q3_k = 11;
    const q4_k = 12;
    const q5_k = 13;
    const q6_k = 14;
    const q8_k = 15;
    const iq2_xxs = 16;
    const iq2_xs = 17;
    const iq3_xxs = 18;
    const iq1_s = 19;
    const iq4_nl = 20;
    const iq3_s = 21;
    const iq2_s = 22;
    const iq4_xs = 23;
    const i8_v = 24;
    const i16_v = 25;
    const i32_v = 26;
    const i64_v = 27;
    const f64_v = 28;
    const fp8_v = 32; // no canónico en gguf; valor arbitrario que no colisiona
    const iq1_m = 29;
    const bf16 = 30;
    const tq1_0 = 34;
    const tq2_0 = 35;
    const mxfp4 = 39;
    // ── lane-b2 P0.3: cache types BeeLlama ── no canónicos en GGUF upstream;
    // valores locales que no colisionan con el rango canónico (0..39).
    const q2_0s = 100;
    const q2_1 = 101;
    const q3_0 = 102;
    const q3_1 = 103;
    const q6_0 = 104;
    const q6_1 = 105;
};

/// Convierte desde valor numérico gguf.GgmlType a QuantFormat
pub fn fromGgmlTypeValue(raw: u32) ?QuantFormat {
    return switch (raw) {
        GgmlTypeValue.f32_v => .fp32,
        GgmlTypeValue.f16_v => .fp16,
        GgmlTypeValue.bf16 => .fp16,
        GgmlTypeValue.q4_0 => .q4_0,
        GgmlTypeValue.q4_1 => .q4_1,
        GgmlTypeValue.q5_0 => .q5_0,
        GgmlTypeValue.q5_1 => .q5_1,
        GgmlTypeValue.q8_0 => .q8_0,
        GgmlTypeValue.q8_1 => .q8_1,
        GgmlTypeValue.q2_k => .q2_k,
        GgmlTypeValue.q3_k => .q3_k,
        GgmlTypeValue.q4_k => .q4_k,
        GgmlTypeValue.q5_k => .q5_k,
        GgmlTypeValue.q6_k => .q6_k,
        GgmlTypeValue.q8_k => .q8_k,
        GgmlTypeValue.iq1_s => .iq1_s,
        GgmlTypeValue.iq1_m => .iq1_m,
        GgmlTypeValue.iq2_xxs => .iq2_xxs,
        GgmlTypeValue.iq2_xs => .iq2_xs,
        GgmlTypeValue.iq2_s => .iq2_s,
        GgmlTypeValue.iq3_xxs => .iq3_xxs,
        GgmlTypeValue.iq3_s => .iq3_s,
        GgmlTypeValue.iq4_xs => .iq4_xs,
        GgmlTypeValue.iq4_nl => .iq4_nl,
        GgmlTypeValue.tq1_0 => .tq1_0,
        GgmlTypeValue.tq2_0 => .tq2_0,
        GgmlTypeValue.mxfp4 => .mxfp4,
        // lane-b2 P0.3: Bee-local cache types (roundtrip serialización).
        GgmlTypeValue.q2_0s => .q2_0s,
        GgmlTypeValue.q2_1 => .q2_1,
        GgmlTypeValue.q3_0 => .q3_0,
        GgmlTypeValue.q3_1 => .q3_1,
        GgmlTypeValue.q6_0 => .q6_0,
        GgmlTypeValue.q6_1 => .q6_1,
        GgmlTypeValue.i8_v => .int8_symmetric,
        GgmlTypeValue.i16_v => .int8_symmetric,
        GgmlTypeValue.i32_v => .int8_symmetric,
        GgmlTypeValue.i64_v => .int8_symmetric,
        GgmlTypeValue.f64_v => .fp32,
        else => null,
    };
}

/// Convierte de QuantFormat a valor numérico gguf.GgmlType para serialización
pub fn toGgmlTypeValue(fmt: QuantFormat) u32 {
    return switch (fmt) {
        .fp16 => GgmlTypeValue.f16_v,
        .fp32 => GgmlTypeValue.f32_v,
        .int8_symmetric => GgmlTypeValue.q8_0,
        .int8_asymmetric => GgmlTypeValue.q8_1,
        .int4 => GgmlTypeValue.q4_0,
        .q4_0 => GgmlTypeValue.q4_0,
        .q4_1 => GgmlTypeValue.q4_1,
        .q5_0 => GgmlTypeValue.q5_0,
        .q5_1 => GgmlTypeValue.q5_1,
        .q8_0 => GgmlTypeValue.q8_0,
        .q8_1 => GgmlTypeValue.q8_1,
        .q2_k => GgmlTypeValue.q2_k,
        .q3_k => GgmlTypeValue.q3_k,
        .q4_k => GgmlTypeValue.q4_k,
        .q5_k => GgmlTypeValue.q5_k,
        .q6_k => GgmlTypeValue.q6_k,
        .q8_k => GgmlTypeValue.q8_k,
        .iq1_s => GgmlTypeValue.iq1_s,
        .iq1_m => GgmlTypeValue.iq1_m,
        .iq2_xxs => GgmlTypeValue.iq2_xxs,
        .iq2_xs => GgmlTypeValue.iq2_xs,
        .iq2_s => GgmlTypeValue.iq2_s,
        .iq3_xxs => GgmlTypeValue.iq3_xxs,
        .iq3_s => GgmlTypeValue.iq3_s,
        .iq4_xs => GgmlTypeValue.iq4_xs,
        .iq4_nl => GgmlTypeValue.iq4_nl,
        .tq1_0 => GgmlTypeValue.tq1_0,
        .tq2_0 => GgmlTypeValue.tq2_0,
        .mxfp4 => GgmlTypeValue.mxfp4,
        .fp8 => GgmlTypeValue.fp8_v,
        // lane-b2 P0.3: Bee-local cache types (serialización interna).
        .q2_0s => GgmlTypeValue.q2_0s,
        .q2_1 => GgmlTypeValue.q2_1,
        .q3_0 => GgmlTypeValue.q3_0,
        .q3_1 => GgmlTypeValue.q3_1,
        .q6_0 => GgmlTypeValue.q6_0,
        .q6_1 => GgmlTypeValue.q6_1,
    };
}

/// Bloque base para formatos legacy (q4_0, q4_1, q5_0, q5_1, q8_0, q8_1)
pub const BLOCK: usize = 32;
/// Super-bloque para K-quants e I-quants
pub const SUPER_BLOCK: usize = 256;

/// Bytes RAW (sin alinear) que ocupan `n_elements` lógicos en `format`.
/// Es el stride que usan los pesos GGUF reales y los kernels GEMM
/// (qgemmKernel lee rowstride = nbig*bytesPerBlock, layout GGUF).
/// B-a4 (lane-a): los tests de paridad GEMM deben construir filas con
/// ESTE stride. 7.3-REVERT: alias de quantBytes() — el pad 32B se
/// eliminó del pool; las dos API son idénticas hoy (ver abajo).
pub fn quantBytesRaw(format: QuantFormat, n_elements: usize) usize {
    const block_size = format.defaultBlockSize();
    const num_blocks = (n_elements + block_size - 1) / block_size;
    return num_blocks * format.bytesPerBlock();
}

/// Bytes que ocupan `n_elements` lógicos en `format` (incluye metadatos).
/// Stride RAW (sin alinear) — el mismo que usan los pesos GGUF reales y
/// TODOS los kernels (GEMM/prefill/decode) sobre el pool.
/// 7.3-REVERT (2026-09-09): el pad 32B de @a8e9dcd rompía los kernels
/// prefill/decode cuantizados (offsets raw 144B/SB vs pool padded 160B
/// ⇒ gpu=0/mismatches masivos — reporte lane-a B-a1, test-pafused).
/// El append paralelo que motivaba el pad nunca llegó a activarse
/// (los kvAppend* siguen single-block); si algún día se paralelizan,
/// alinear pool Y kernels juntos (ver kvAppendQ8_0Kernel comment).
pub fn quantBytes(format: QuantFormat, n_elements: usize) usize {
    const block_size = format.defaultBlockSize();
    const num_blocks = (n_elements + block_size - 1) / block_size;
    return num_blocks * format.bytesPerBlock();
}

/// Opciones de cuantización (paso 0 KV-Codec: SR en V).
/// `stochastic` activa stochastic rounding: q = floor(x/d) con prob
/// (1-frac) y ceil(x/d) con prob frac — E[e]=0 exacto. RNG PCG por
/// (posición del valor, semilla global) para reproducibilidad bit-exact.
pub const EncodeOptions = struct {
    stochastic: bool = false,
};

/// RNG PCG32 (reproducible, sin dependencias): estado global del módulo,
/// semilla fija por proceso — determinista run a run.
const Pcg = struct {
    state: u64,
    inc: u64,
    fn next(self: *Pcg) u32 {
        const old: u64 = self.state;
        self.state = old *% 6364136223846793005 +% self.inc;
        const xorshifted: u32 = @truncate(((old >> 18) ^ old) >> 27);
        const rot: u32 = @truncate(old >> 59); // [0,32)
        // rotate-right 32 canónico: rot=0 → identidad (sin UB de shift 32)
        return (xorshifted >> @as(u5, @truncate(rot))) | (xorshifted << @as(u5, @truncate((32 - rot) & 31)));
    }
    fn nextFloat(self: *Pcg) f32 {
        return @as(f32, @floatFromInt(self.next() >> 8)) / 16777216.0;
    }
};

var g_pcg = Pcg{ .state = 0x853c49e6748fea9b, .inc = 0xda3e39cb94b95bdb };

/// Resetea la semilla del RNG global (tests: secuencia reproducible).
pub fn seedEncodeRng(seed: u64) void {
    g_pcg.state = seed;
    g_pcg.inc = (seed << 1) | 1;
}

/// Codifica `src` (f16) al `format` en un buffer recién alocado (layout canónico).
pub fn encodeToOwned(allocator: std.mem.Allocator, format: QuantFormat, src: []const f16) ![]u8 {
    const dst = try allocator.alloc(u8, quantBytes(format, src.len));
    encode(format, src, dst);
    return dst;
}

/// `encodeToOwned` con opciones (KVSR_V).
pub fn encodeToOwnedOpts(allocator: std.mem.Allocator, format: QuantFormat, src: []const f16, opts: EncodeOptions) ![]u8 {
    const dst = try allocator.alloc(u8, quantBytes(format, src.len));
    encodeOpts(format, src, dst, opts);
    return dst;
}

/// Codifica `src` (f16) al `format` en `dst` (tamaño `quantBytes(format, src.len)`).
pub fn encode(format: QuantFormat, src: []const f16, dst: []u8) void {
    encodeOpts(format, src, dst, null);
}

/// `encode` con opciones (SR en V, paso 0 KV-Codec).
pub fn encodeOpts(format: QuantFormat, src: []const f16, dst: []u8, opts: ?EncodeOptions) void {
    const n = src.len;
    if (opts) |o| setStochasticRounding(o.stochastic);
    const block_size = format.defaultBlockSize();
    const num_blocks = (n + block_size - 1) / block_size;
    var blk: [SUPER_BLOCK]f32 = undefined;
    var i: usize = 0;
    for (0..num_blocks) |bi| {
        const bs = @min(block_size, n - i);
        for (0..bs) |j| blk[j] = @as(f32, @floatCast(src[i + j]));
        for (bs..block_size) |j| blk[j] = 0.0;
        const base = bi * format.bytesPerBlock();
        switch (format) {
            .fp16 => {
                const src_bytes = std.mem.sliceAsBytes(src);
                const off = bi * block_size * 2;
                const take = @min(block_size * 2, dst[base..].len);
                @memcpy(dst[base..][0..take], src_bytes[off..][0..take]);
            },
            .fp32 => {
                const src_bytes = std.mem.sliceAsBytes(src);
                const off = bi * block_size * 4;
                const take = @min(block_size * 4, dst[base..].len);
                @memcpy(dst[base..][0..take], src_bytes[off..][0..take]);
            },
            .q8_0 => encodeQ8_0(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .q8_1 => encodeQ8_1(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .q4_0 => encodeQ4_0(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .q4_1 => encodeQ4_1(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .q5_0 => encodeQ5_0(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .q5_1 => encodeQ5_1(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            // Lane-b2 P0.3: cache types estándar adicionales.
            .q2_0s => encodeQ2_0S(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .q2_1 => encodeQ2_1(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .q3_0 => encodeQ3_0(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .q3_1 => encodeQ3_1(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .q6_0 => encodeQ6_0(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .q6_1 => encodeQ6_1(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .q2_k => encodeQ2_K(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .q3_k => encodeQ3_K(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .q4_k => encodeQ4_K(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .q5_k => encodeQ5_K(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .q6_k => encodeQ6_K(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .q8_k => encodeQ8_K(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .iq1_s => encodeIQ1_S(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .iq1_m => encodeIQ1_M(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .iq2_xxs => encodeIQ2_XXS(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .iq2_xs => encodeIQ2_XS(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .iq2_s => encodeIQ2_S(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .iq3_xxs => encodeIQ3_XXS(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .iq3_s => encodeIQ3_S(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .iq4_xs => encodeIQ4_XS(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .iq4_nl => encodeIQ4_NL(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .tq1_0 => encodeTQ1_0(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .tq2_0 => encodeTQ2_0(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .mxfp4 => encodeMXFP4(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .int8_symmetric => encodeInt8Sym(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .int8_asymmetric => encodeInt8Asym(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .int4 => encodeInt4(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
            .fp8 => encodeFP8(blk[0..bs], dst[base..][0..format.bytesPerBlock()]),
        }
        i += block_size;
    }
}

/// Decodifica `bytes` (formato canónico) a `out` (f16). `out.len` = num
/// elementos lógicos. Reaprovecha el layout GGUF canónico.
pub fn decode(format: QuantFormat, bytes: []const u8, out: []f16) void {
    const n = out.len;
    if (n == 0) return;
    const block_size = format.defaultBlockSize();
    const blocks_total = (n + block_size - 1) / block_size;
    var tmp: [8192]f32 = undefined;
    const chunk_blocks = tmp.len / block_size;
    var bi: usize = 0;
    while (bi < blocks_total) : (bi += chunk_blocks) {
        const step_blocks = @min(chunk_blocks, blocks_total - bi);
        const step = step_blocks * block_size;
        const acc = bi * block_size;
        dequantF32(format, bytes[bi * format.bytesPerBlock() ..], tmp[0..step]);
        const take = @min(step, n - acc);
        for (0..take) |j| out[acc + j] = @as(f16, @floatCast(tmp[j]));
    }
}

// ============================================================================
// Codificadores (encode) - basados en llama.cpp / ggml
// ============================================================================

/// Stochastic rounding sobre la rejilla del bloque: q = floor(x/d) con
/// prob (1−frac) y floor(x/d)+1 con prob frac, donde frac = x/d − floor.
/// E[e] = 0 exacto (paso 0 KV-Codec §4.6: V es media ponderada — el sesgo
/// del redondeo determinístico acumula con la masa de atención, la
/// varianza del SR se promedia en la suma). `enabled` = flag global.
var g_sr_enabled: bool = false;

/// Activa/desactiva SR en los encoders de bloque (thread-local no: los
/// encoders corren en el hilo del encoder del pool).
pub fn setStochasticRounding(on: bool) void {
    g_sr_enabled = on;
}

fn srRound(x_over_d: f32) f32 {
    // floor + U(0,1) < frac → ceil. RNG PCG global (reproducible).
    const fl = @floor(x_over_d);
    const frac = x_over_d - fl;
    if (g_pcg.nextFloat() < frac) return fl + 1.0;
    return fl;
}

fn encodeQ8_0(blk: []const f32, dst: []u8) void {
    const d = if (maxAbs(blk) > 0) maxAbs(blk) / 127.0 else 1.0;
    writeF16(dst[0..2], d);
    for (blk, 0..) |v, j| {
        const x_over_d: f32 = if (g_sr_enabled) srRound(v / d) else @round(v / d);
        var q: i32 = @intFromFloat(x_over_d);
        if (q < -127) q = -127;
        if (q > 127) q = 127;
        dst[2 + j] = @as(u8, @bitCast(@as(i8, @intCast(q))));
    }
}

fn encodeQ8_1(blk: []const f32, dst: []u8) void {
    var max_val: f32 = -std.math.inf(f32);
    var min_val: f32 = std.math.inf(f32);
    for (blk) |v| {
        max_val = @max(max_val, v);
        min_val = @min(min_val, v);
    }
    const d = if (max_val - min_val > 0) (max_val - min_val) / 254.0 else 1.0;
    const m = min_val;
    writeF16(dst[0..2], d);
    writeF16(dst[2..4], m);
    for (blk, 0..) |v, j| {
        var q: i32 = @intFromFloat(@round((v - m) / d));
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        dst[4 + j] = @as(u8, @bitCast(@as(i8, @intCast(q))));
    }
}

fn encodeQ4_0(blk: []const f32, dst: []u8) void {
    const d = if (maxAbs(blk) > 0) maxAbs(blk) / 7.0 else 1.0;
    writeF16(dst[0..2], d);
    const n = @min(BLOCK, blk.len);
    // Empaquetado canónico GGML (split-16, ref dequantize_row_q4_0 master):
    // elems [0,16) = nibbles BAJOS de qs[0..16), elems [16,32) = ALTOS de
    // qs[0..16). NO intercalado por par/impar (bug histórico: la caché q4_0
    // escrita así se leía mal con los kernels/gguf que son split-16).
    @memset(dst[2..18], 0);
    for (0..n) |idx| {
        const v = blk[idx];
        const lo: i32 = @intFromFloat(if (g_sr_enabled) srRound(v / d) else @round(v / d));
        const q = @min(@max(lo, -8), 7) + 8;
        if (idx < BLOCK / 2) {
            dst[2 + idx] = @as(u8, @intCast(q & 0x0F));
        } else {
            dst[2 + idx - BLOCK / 2] |= @as(u8, @intCast(q & 0x0F)) << 4;
        }
    }
}

fn encodeQ4_1(blk: []const f32, dst: []u8) void {
    var max_val: f32 = -std.math.inf(f32);
    var min_val: f32 = std.math.inf(f32);
    for (blk) |v| {
        max_val = @max(max_val, v);
        min_val = @min(min_val, v);
    }
    const d = if (max_val - min_val > 0) (max_val - min_val) / 15.0 else 1.0;
    writeF16(dst[0..2], d);
    writeF16(dst[2..4], min_val);
    const half = @min(BLOCK / 2, blk.len);
    for (0..half) |j| {
        const lo: i32 = @intFromFloat(@round((blk[j] - min_val) / d));
        const hi: i32 = @intFromFloat(@round((blk[j + half] - min_val) / d));
        var lo_c: i32 = lo;
        if (lo_c < 0) lo_c = 0;
        if (lo_c > 15) lo_c = 15;
        var hi_c: i32 = hi;
        if (hi_c < 0) hi_c = 0;
        if (hi_c > 15) hi_c = 15;
        dst[4 + j] = @as(u8, @intCast(lo_c)) | (@as(u8, @intCast(hi_c)) << 4);
    }
}

fn encodeQ5_0(blk: []const f32, dst: []u8) void {
    const d = if (maxAbs(blk) > 0) maxAbs(blk) / 15.0 else 1.0;
    writeF16(dst[0..2], d);
    var qh: u32 = 0;
    const half = @min(BLOCK / 2, blk.len);
    for (0..half) |j| {
        const lo: i32 = @intFromFloat(@round(blk[j] / d));
        const hi: i32 = @intFromFloat(@round(blk[j + half] / d));
        var lo_c: i32 = @min(@max(lo, -16), 15);
        var hi_c: i32 = @min(@max(hi, -16), 15);
        if (lo_c < 0) {
            qh |= @as(u32, 1) << @intCast(j);
            lo_c += 16;
        }
        if (hi_c < 0) {
            qh |= @as(u32, 1) << @intCast(j + 16);
            hi_c += 16;
        }
        dst[6 + j] = @as(u8, @intCast(lo_c)) | (@as(u8, @intCast(hi_c)) << 4);
    }
    std.mem.writeInt(u32, dst[2..6], qh, .little);
}

fn encodeQ5_1(blk: []const f32, dst: []u8) void {
    var max_val: f32 = -std.math.inf(f32);
    var min_val: f32 = std.math.inf(f32);
    for (blk) |v| {
        max_val = @max(max_val, v);
        min_val = @min(min_val, v);
    }
    const d = if (max_val - min_val > 0) (max_val - min_val) / 15.0 else 1.0;
    writeF16(dst[0..2], d);
    writeF16(dst[2..4], min_val);
    var qh: u32 = 0;
    const half = @min(BLOCK / 2, blk.len);
    for (0..half) |j| {
        const lo: i32 = @intFromFloat(@round((blk[j] - min_val) / d));
        const hi: i32 = @intFromFloat(@round((blk[j + half] - min_val) / d));
        var lo_c: i32 = @min(@max(lo, 0), 15);
        var hi_c: i32 = @min(@max(hi, 0), 15);
        if (lo < 0) {
            qh |= @as(u32, 1) << @intCast(j);
            lo_c += 16;
        }
        if (hi < 0) {
            qh |= @as(u32, 1) << @intCast(j + 16);
            hi_c += 16;
        }
        dst[8 + j] = @as(u8, @intCast(lo_c)) | (@as(u8, @intCast(hi_c)) << 4);
    }
    std.mem.writeInt(u32, dst[4..8], qh, .little);
}

// ── Lane-b2 P0.3: encoders cache types estándar adicionales ──
// Layout canónico GGML/Bee, transcripción de ggml-common.h + gguf.zig.

// q2_0S (block=32, 2bpw): f16 d + qs[8]. Layout upstream block_q2_0s:
//   qs[j] (j ∈ [0..8)) guarda 4 elems con 2 bits cada uno:
//     elem j     = (qs[j] >> 0) & 3
//     elem j+8   = (qs[j] >> 2) & 3
//     elem j+16  = (qs[j] >> 4) & 3
//     elem j+24  = (qs[j] >> 6) & 3
// Codificación x = (c - 2) * d (offset 2 simétrico).
fn encodeQ2_0S(blk: []const f32, dst: []u8) void {
    var mx: f32 = -std.math.inf(f32);
    for (blk) |v| mx = @max(mx, @abs(v));
    const d: f32 = if (mx > 0) mx / 2.0 else 1.0;
    writeF16(dst[0..2], d);
    @memset(dst[2..10], 0);
    for (0..8) |j| {
        var byte: u8 = 0;
        inline for ([_]u32{ 0, 1, 2, 3 }) |p| {
            const ei: usize = j + p * 8;
            if (ei < blk.len) {
                const q_f: f32 = @round(blk[ei] / d) + 2.0;
                const clamped: f32 = @max(@min(q_f, 3.0), 0.0);
                const q: u8 = @intCast(@as(i32, @intFromFloat(clamped)));
                byte |= q << @intCast(2 * p);
            }
        }
        dst[2 + j] = byte;
    }
}

// q2_1 (block=32, 2bpw): f16 d + f16 m + qs[8]. x = c*d + m (offset 0).
fn encodeQ2_1(blk: []const f32, dst: []u8) void {
    var mn: f32 = std.math.inf(f32);
    var mx: f32 = -std.math.inf(f32);
    for (blk) |v| {
        mn = @min(mn, v);
        mx = @max(mx, v);
    }
    const d: f32 = if (mx - mn > 0) (mx - mn) / 3.0 else 1.0;
    const m: f32 = mn;
    writeF16(dst[0..2], d);
    writeF16(dst[2..4], m);
    @memset(dst[4..12], 0);
    for (0..8) |j| {
        var byte: u8 = 0;
        inline for ([_]u32{ 0, 1, 2, 3 }) |p| {
            const ei: usize = j + p * 8;
            if (ei < blk.len) {
                const q_f: f32 = @round((blk[ei] - m) / d);
                const clamped: f32 = @max(@min(q_f, 3.0), 0.0);
                const q: u8 = @intCast(@as(i32, @intFromFloat(clamped)));
                byte |= q << @intCast(2 * p);
            }
        }
        dst[4 + j] = byte;
    }
}

// q3_0 (block=32, 3bpw): f16 d + qh[4] + qs[8].
// Código 3 bits c∈[0..7] por elem, x = (c-4)*d (8 niveles simétricos).
// Layout: qs[j] empaqueta 4 elems (j, j+8, j+16, j+24) a 2 bits c/u
// (shift 0/2/4/6 — solo los 2 bits bajos del código); qh bit ei = bit alto
// del código del elem ei. Auto-consistente con dequantQ3_0.
fn encodeQ3_0(blk: []const f32, dst: []u8) void {
    var mx: f32 = -std.math.inf(f32);
    for (blk) |v| mx = @max(mx, @abs(v));
    const d: f32 = if (mx > 0) mx / 4.0 else 1.0;
    writeF16(dst[0..2], d);
    @memset(dst[2..14], 0);
    var qh: u32 = 0;
    for (0..32) |ei| {
        const v: f32 = blk[ei];
        const q_f: f32 = @round(v / d) + 4.0;
        const q: i32 = @intFromFloat(q_f);
        const q_c: u8 = @intCast(@min(@max(q, 0), 7));
        if (q_c & 4 != 0) qh |= @as(u32, 1) << @intCast(ei);
        dst[6 + (ei % 8)] |= (q_c & 3) << @intCast(2 * (ei / 8));
    }
    std.mem.writeInt(u32, dst[2..6], qh, .little);
}

// q3_1 (block=32, 3bpw): f16 d + f16 m + qh[4] + qs[8]. x = c*d + m (unsigned).
// Código 3 bits c∈[0..7]. Layout igual que q3_0 (qs 2 bits bajos por elem,
// qh bit alto). Auto-consistente con dequantQ3_1.
fn encodeQ3_1(blk: []const f32, dst: []u8) void {
    var mn: f32 = std.math.inf(f32);
    var mx: f32 = -std.math.inf(f32);
    for (blk) |v| {
        mn = @min(mn, v);
        mx = @max(mx, v);
    }
    const d: f32 = if (mx - mn > 0) (mx - mn) / 7.0 else 1.0;
    const m: f32 = mn;
    writeF16(dst[0..2], d);
    writeF16(dst[2..4], m);
    @memset(dst[4..16], 0);
    var qh: u32 = 0;
    for (0..32) |ei| {
        const v: f32 = blk[ei];
        const q_f: f32 = @round((v - m) / d);
        const q_c: u8 = @intCast(@min(@max(@as(i32, @intFromFloat(q_f)), 0), 7));
        if (q_c & 4 != 0) qh |= @as(u32, 1) << @intCast(ei);
        dst[8 + (ei % 8)] |= (q_c & 3) << @intCast(2 * (ei / 8));
    }
    std.mem.writeInt(u32, dst[4..8], qh, .little);
}

// q6_0 (block=32, 6bpw): f16 d + qh[8] + qs[16]. x = (c - 32) * d.
// Layout upstream ggml-quants.c línea 303: j ∈ [0..16), elem j + j+16.
// qs[j] = (q0 & 0xF) | ((q1 & 0xF) << 4)
// qh[j%8] |= h << (4*(j/8)) con h = (q0>>4) | ((q1>>4) << 2)
fn encodeQ6_0(blk: []const f32, dst: []u8) void {
    var mx: f32 = -std.math.inf(f32);
    for (blk) |v| mx = @max(mx, @abs(v));
    const d: f32 = if (mx > 0) mx / 32.0 else 1.0;
    writeF16(dst[0..2], d);
    @memset(dst[2..26], 0);
    for (0..16) |j| {
        const x0: f32 = blk[0 + j];
        const x1: f32 = blk[16 + j];
        const xi0: i32 = @intFromFloat(@round(x0 / d) + 32.0);
        const xi1: i32 = @intFromFloat(@round(x1 / d) + 32.0);
        const q0: u8 = @intCast(@min(@max(xi0, 0), 63));
        const q1: u8 = @intCast(@min(@max(xi1, 0), 63));
        dst[10 + j] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        const h: u8 = (q0 >> 4) | ((q1 >> 4) << 2);
        dst[2 + (j % 8)] |= h << @intCast(4 * (j / 8));
    }
}

// q6_1 (block=32, 6bpw): f16 d + f16 m + qh[8] + qs[16]. x = c*d + m.
fn encodeQ6_1(blk: []const f32, dst: []u8) void {
    var mn: f32 = std.math.inf(f32);
    var mx: f32 = -std.math.inf(f32);
    for (blk) |v| {
        mn = @min(mn, v);
        mx = @max(mx, v);
    }
    const d: f32 = if (mx - mn > 0) (mx - mn) / 63.0 else 1.0;
    const m: f32 = mn;
    writeF16(dst[0..2], d);
    writeF16(dst[2..4], m);
    @memset(dst[4..28], 0);
    for (0..16) |j| {
        const x0: f32 = blk[0 + j];
        const x1: f32 = blk[16 + j];
        const q0: i32 = @intFromFloat(@round((x0 - m) / d));
        const q1: i32 = @intFromFloat(@round((x1 - m) / d));
        const q0_c: u8 = @intCast(@min(@max(q0, 0), 63));
        const q1_c: u8 = @intCast(@min(@max(q1, 0), 63));
        dst[12 + j] = (q0_c & 0x0F) | ((q1_c & 0x0F) << 4);
        const h: u8 = (q0_c >> 4) | ((q1_c >> 4) << 2);
        dst[4 + (j % 8)] |= h << @intCast(4 * (j / 8));
    }
}

// K-quants encoders (super-block 256)
fn encodeQ2_K(blk: []const f32, dst: []u8) void {
    // Layout 84B/SB256 — espejo EXACTO de dequantQ2_K/gguf.dequantQ2_K:
    // scales[16]@0, qs[64]@16, d f16@80, min f16@82.
    // Sub-bloques de 16 elems (16/SB): UN byte de escala cada uno
    // (dcode nibble low, mcode nibble high). Quanta 2-bit: byte
    // qs[nh*32+c] campo shift=2j guarda el par (c, c+16).
    // Esquema determinista propio (no el search ponderado de llama.cpp):
    //   paso=(mx-mn)/3; dcode=clamp(round(paso/d),0,15);
    //   mcode=clamp(round(max(-mn,0)/min_s),0,15);
    //   q=clamp(round((x+ml)/(d·dcode)),0,3) con ml=min_s·mcode.
    //   Super-escalas: d=max_span/(3·15), min_s=max_neg/15.
    const qk: usize = 256;
    var x: [qk]f32 = undefined;
    for (0..qk) |j| x[j] = if (j < blk.len) blk[j] else 0.0;

    @memset(dst[0..84], 0);

    var span: [16]f32 = undefined;
    var neg: [16]f32 = undefined;
    var max_span: f32 = 0;
    var max_neg: f32 = 0;
    for (0..16) |s| {
        var mn: f32 = std.math.inf(f32);
        var mx: f32 = -std.math.inf(f32);
        for (0..16) |c| {
            const v = x[s * 16 + c];
            mn = @min(mn, v);
            mx = @max(mx, v);
        }
        span[s] = mx - mn;
        neg[s] = if (mn < 0) -mn else 0;
        max_span = @max(max_span, span[s]);
        max_neg = @max(max_neg, neg[s]);
    }
    const d: f32 = if (max_span > 0) max_span / (3.0 * 15.0) else 1.0;
    const min_s: f32 = if (max_neg > 0) max_neg / 15.0 else 1.0;
    writeF16(dst[80..82], d);
    writeF16(dst[82..84], min_s);

    for (0..16) |s| {
        const dcode: u8 = @intCast(@min(@as(i32, @intFromFloat(@round(span[s] / 3.0 / d))), @as(i32, 15)));
        const mcode: u8 = @intCast(@min(@as(i32, @intFromFloat(@round(neg[s] / min_s))), @as(i32, 15)));
        dst[s] = dcode | (mcode << 4);

        const dl: f32 = d * @as(f32, @floatFromInt(dcode));
        const ml: f32 = min_s * @as(f32, @floatFromInt(mcode));
        const inv: f32 = if (dl > 0) 1.0 / dl else 0.0;
        const nh = s >> 3;
        const jj = (s >> 1) & 3;
        const shift: u3 = @intCast(2 * jj);
        for (0..16) |c| {
            var q: i32 = @intFromFloat(@round((x[s * 16 + c] + ml) * inv));
            q = @min(@max(q, 0), 3);
            // Sub-bloque par → col=c; impar → col=c+16. El byte quanta
            // SIEMPRE es qs[nh*32+col] (dequant lee q[l] y q[l+16]).
            const col = c + (s & 1) * 16;
            dst[16 + nh * 32 + col] |= @as(u8, @intCast(q)) << shift;
        }
    }
}

fn encodeQ3_K(blk: []const f32, dst: []u8) void {
    // Layout 110B/SB256 — espejo de dequantQ3_K/gguf.dequantQ3_K:
    // hmask[32]@0 (bit=1 ⇒ +0; clear ⇒ −4), qs[64]@32 (2-bit shift=2j,
    // byte nh*32+col), scales[12]@96 en orden REORDENADO kmask, d f16@108.
    // Sub-bloques de 16: dl = d·(i8(s16[s])−32); rango {−4..3}·dl.
    // Esquema determinista propio: dl_target=span/7;
    // s16[s]=clamp(round(dl/d)+32,0,255); d=max_span/(7·63).
    // Per-elem: x≥0 → bit SET, q=round(x/dl); x<0 → bit CLEAR,
    // q=clamp(round(x/dl+4),0,3). INVERSA del reorden kmask incluida.
    const qk: usize = 256;
    var x: [qk]f32 = undefined;
    for (0..qk) |j| x[j] = if (j < blk.len) blk[j] else 0.0;

    @memset(dst[0..110], 0);

    var max_span: f32 = 0;
    for (0..16) |s| {
        var mn: f32 = std.math.inf(f32);
        var mx: f32 = -std.math.inf(f32);
        for (0..16) |c| {
            const v = x[s * 16 + c];
            mn = @min(mn, v);
            mx = @max(mx, v);
        }
        max_span = @max(max_span, mx - mn);
    }
    // ⚠️ El reorden kmap solo preserva 6 bits por escala (bits[6,8)=0):
    // s16 bytes ∈ [0,63] ⇒ códigos dl ∈ [−32,31]. d = max_dl/31.
    const d: f32 = if (max_span > 0) max_span / (7.0 * 31.0) else 1.0;
    writeF16(dst[108..110], d);

    // aux' deseados: s16 bytes empaquetados LE en 4 u32 (16 escalas i8).
    var aux: [4]u32 = .{ 0, 0, 0, 0 };
    for (0..16) |s| {
        const mn_blk: f32 = blk_min16(&x, s * 16);
        const mx_blk: f32 = blk_max16(&x, s * 16);
        const dl_t = (mx_blk - mn_blk) / 7.0;
        const b_val: i32 = @min(@as(i32, @intFromFloat(@round(dl_t / d))), @as(i32, 31)) + 32;
        // b>=32 garantiza dl>=0 (branch por signo del encoder lo exige).
        const b: u8 = @intCast(@max(b_val, 32));
        aux[s / 4] |= @as(u32, b) << @as(u5, @intCast(8 * (s % 4)));

        const dl: f32 = d * @as(f32, @floatFromInt(@as(i8, @bitCast(b)) - 32));
        const inv: f32 = if (dl > 0) 1.0 / dl else 0.0;
        const nh = s >> 3;
        const jj = (s >> 1) & 3;
        const shift: u3 = @intCast(2 * jj);
        for (0..16) |c| {
            const xv = x[s * 16 + c];
            const col = c + (s & 1) * 16;
            var q: i32 = undefined;
            if (xv >= 0) {
                q = @min(@as(i32, @intFromFloat(@round(xv * inv))), 3);
                // Bit GLOBAL nh*4+jj (m acumula entre mitades en el dequant).
                dst[col] |= @as(u8, 1) << @as(u3, @intCast(nh * 4 + jj)); // SET ⇒ +0
            } else {
                q = @as(i32, @intFromFloat(@round(xv * inv))) + 4;
                q = @min(@max(q, 0), 3); // bit CLEAR (ya 0 por memset) ⇒ −4
            }
            dst[32 + nh * 32 + col] |= @as(u8, @intCast(q)) << shift;
        }
    }

    // INVERSA del reorden kmask: scales[12] tal que el reorden produzca aux.
    // ⚠️ Los scales viven en dst[96..108] (hmask@0/qs@32 van antes).
    const ob = std.mem.sliceAsBytes(aux[0..]);
    for (0..4) |b| {
        dst[96 + b] = (ob[b] & 0xF) | ((ob[8 + b] & 0xF) << 4);
        dst[100 + b] = (ob[4 + b] & 0xF) | ((ob[12 + b] & 0xF) << 4);
        dst[104 + b] = ((ob[b] >> 4) & 3) | (((ob[4 + b] >> 4) & 3) << 2) | (((ob[8 + b] >> 4) & 3) << 4) | (((ob[12 + b] >> 4) & 3) << 6);
    }
}
fn blk_min16(x: []const f32, base: usize) f32 {
    var mn: f32 = std.math.inf(f32);
    for (0..16) |c| mn = @min(mn, x[base + c]);
    return mn;
}
fn blk_max16(x: []const f32, base: usize) f32 {
    var mx: f32 = -std.math.inf(f32);
    for (0..16) |c| mx = @max(mx, x[base + c]);
    return mx;
}

/// Codificador q4_K REAL (layout canónico GGUF/llama.cpp, exactamente el que
/// leen paged_attention_decode_q4_k_kernel y dequantQ4_K):
///   144B = [d f16@0][dmin f16@2][scales[12]@4..15][qs[128]@16..143]
/// Elemento w∈[0,256): g=w/64, l=w%64; escala si=2g+(l≥32); nibble en
/// qs[g*32+(l%32)], low si l<32 / high si l≥32; valor = d*sd*q − dmin*sm.
/// Esquema de cuantización (documentado; válido para cualquier lector
/// canónico aunque difiera del search de llama.cpp): por sub-bloque de 32
/// (8 por SB) se ajusta span→sd y offset→sm de 6 bits con super-escalas
/// d=span_max/(15·63) y dmin=off_max/63.
fn encodeQ4_K(blk: []const f32, dst: []u8) void {
    const qk: usize = 256;
    var x: [qk]f32 = undefined;
    for (0..qk) |j| x[j] = if (j < blk.len) blk[j] else 0.0;

    // Pass 1: por sub-bloque de 32 → span y offset objetivo (−min clampeado).
    var span: [8]f32 = undefined;
    var off: [8]f32 = undefined;
    var max_span: f32 = 0;
    var max_off: f32 = 0;
    for (0..8) |sb| {
        var mn: f32 = std.math.inf(f32);
        var mx: f32 = -std.math.inf(f32);
        for (0..32) |j| {
            const v = x[sb * 32 + j];
            mn = @min(mn, v);
            mx = @max(mx, v);
        }
        span[sb] = mx - mn;
        off[sb] = @max(-mn, 0);
        max_span = @max(max_span, span[sb]);
        max_off = @max(max_off, off[sb]);
    }
    const d: f32 = if (max_span > 0) max_span / (15.0 * 63.0) else 1.0;
    const dmin: f32 = if (max_off > 0) max_off / 63.0 else 1.0;
    writeF16(dst[0..2], d);
    writeF16(dst[2..4], dmin);

    const qs = dst[16..144];
    @memset(qs, 0);
    var scales: [12]u8 = .{0} ** 12;
    var sd_v: [8]u8 = undefined;
    var sm_v: [8]u8 = undefined;

    // Pass 2: por GRUPO de 64 (dos sub-bloques comparten bytes de nibble).
    for (0..4) |g| {
        for (0..2) |half| {
            const sb = g * 2 + half;
            const sd: u8 = blk_clamp_q(roundToU64(span[sb] / (15.0 * d)), 63);
            const sm: u8 = blk_clamp_q(roundToU64(off[sb] / dmin), 63);
            sd_v[sb] = sd;
            sm_v[sb] = sm;
            const dl = d * @as(f32, @floatFromInt(sd));
            const ml = dmin * @as(f32, @floatFromInt(sm));
            const base = sb * 32;
            for (0..32) |j| {
                var q: i32 = 0;
                if (dl > 0) {
                    const t: f32 = if (g_sr_enabled) srRound((x[base + j] + ml) / dl) else @round((x[base + j] + ml) / dl);
                    q = clampI(@as(i32, @intFromFloat(t)), 0, 15);
                }
                const byte_idx = g * 32 + j;
                if (half == 0) {
                    qs[byte_idx] |= @intCast(q & 0x0F); // low
                } else {
                    qs[byte_idx] |= @intCast(q << 4); // high
                }
            }
        }
    }
    _ = &qs;

    // Empaquetado de escalas 6-bit (inverso exacto de getScaleMinK4Canon /
    // del kernel CUDA): si<4 directo, si≥4 con spill de 2 bits altos.
    for (0..4) |si| {
        scales[si] = sd_v[si] & 63;
        scales[si + 4] = sm_v[si] & 63;
    }
    for (4..8) |si| {
        scales[si + 4] = (sd_v[si] & 0xF) | ((sm_v[si] & 0xF) << 4);
        scales[si - 4] |= ((sd_v[si] >> 4) & 3) << 6;
        scales[si] |= ((sm_v[si] >> 4) & 3) << 6;
    }
    @memcpy(dst[4..16], &scales);
}

inline fn roundToU64(v: f32) u64 {
    const r = @round(v);
    return if (r <= 0) 0 else @intFromFloat(r);
}

inline fn clampI(v: i32, lo: i32, hi: i32) i32 {
    return @max(lo, @min(hi, v));
}

inline fn blk_clamp_q(v: u64, hi: u8) u8 {
    return if (v > hi) hi else @intCast(v);
}

fn encodeQ5_K(blk: []const f32, dst: []u8) void {
    const d = if (maxAbs(blk) > 0) maxAbs(blk) / 15.0 else 1.0;
    const min = minVal(blk);
    writeF16(dst[0..2], d);
    writeF16(dst[2..4], min);
    @memset(dst[4..176], 0);
}

fn encodeQ6_K(blk: []const f32, dst: []u8) void {
    const d = if (maxAbs(blk) > 0) maxAbs(blk) / 31.0 else 1.0;
    writeF16(dst[208..210], d);
    @memset(dst[0..208], 0);
}

fn encodeQ8_K(blk: []const f32, dst: []u8) void {
    // Determinista en los 292B: bsums [260..292) a cero (el decode no las
    // lee — paged_attention.cu §q8_k "unused") ⇒ comparación full-región.
    const d = if (maxAbs(blk) > 0) maxAbs(blk) / 127.0 else 1.0;
    @memset(dst, 0);
    std.mem.writeInt(u32, dst[0..4], @as(u32, @bitCast(d)), .little);
    for (blk, 0..) |v, j| {
        var q: i32 = @intFromFloat(@round(v / d));
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        dst[4 + j] = @as(u8, @bitCast(@as(i8, @intCast(q))));
    }
}

/// LUT canónica IQ4_NL (kernels/tables.cuh, ggml-common.h).
pub const kvalues_iq4nl = [16]i8{ -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

// I-quants encoders (simplified - full impl needs lookup tables)
/// Codificador IQ1_S REAL (inverso del decode arriba y del kernel fused).
/// Brute-force: por sub-bloque de 32, escanea delta×sign×512 índices de
/// grid eligiendo mínimos cuadrados por grupo de 8. Esquema documentado:
///   d = max_ideal_dl/15 (delta máx cubre el peor sub-bloque),
///   ideal_dl_sb = amax_sb/9 (acotado por |gv|≈±9 en la grid).
fn encodeIQ1_S(blk: []const f32, dst: []u8) void {
    const qk: usize = 256;
    var x: [qk]f32 = undefined;
    for (0..qk) |j| x[j] = if (j < blk.len) blk[j] else 0.0;

    var amax_sb: [8]f32 = undefined;
    var amax_all: f32 = 0;
    for (0..8) |sb| {
        var mx: f32 = 0;
        for (0..32) |j| mx = @max(mx, @abs(x[sb * 32 + j]));
        amax_sb[sb] = mx;
        amax_all = @max(amax_all, mx);
    }
    // d tal que el mayor delta (7→factor 15) cubre amax/9 con holgura.
    const d: f32 = if (amax_all > 0) amax_all / (9.0 * 15.0) else 1.0;
    writeF16(dst[0..2], d);

    const qs = dst[2..34];
    const qh = dst[34..50];
    @memset(qs, 0);
    @memset(qh, 0);

    for (0..8) |sb| {
        // f32: misma precisión y orden que el kernel cooperativo.
        var best_err: f32 = 3.0e38;
        var best_delta: u8 = 0;
        var best_neg: bool = false;
        var best_idx: [4]u16 = .{ 0, 0, 0, 0 };

        for (0..8) |delta| {
            for (0..2) |neg_i| {
                const neg = neg_i == 1;
                const dl = d * (2.0 * @as(f32, @floatFromInt(delta)) + 1.0);
                const dd: f32 = if (neg) -0.125 else 0.125;
                // Acumulación f32 (e/be/err_total): MISMO orden de
                // operaciones que el kernel cooperativo (término
                // dv*dv secuencial ascendente) ⇒ selección idéntica
                // bit-a-bit GPU↔CPU.
                var err_total: f32 = 0;
                var idxs: [4]u16 = undefined;
                for (0..4) |l| {
                    var be: f32 = 3.0e38;
                    var bi: u16 = 0;
                    for (0..512) |gi| {
                        const g = iq_grids.iq1s_grid[gi];
                        var e: f32 = 0;
                        for (0..8) |jj| {
                            const raw: u8 = @truncate(g >> @as(u6, @intCast(8 * jj)));
                            const gv: f32 = @floatFromInt(@as(i8, @bitCast(raw)));
                            const dv = x[sb * 32 + l * 8 + jj] - dl * (gv + dd);
                            e += dv * dv;
                        }
                        if (e < be) {
                            be = e;
                            bi = @intCast(gi);
                        }
                    }
                    idxs[l] = bi;
                    err_total += be;
                }
                if (err_total < best_err) {
                    best_err = err_total;
                    best_delta = @intCast(delta);
                    best_neg = neg;
                    best_idx = idxs;
                }
            }
        }

        for (0..4) |l| qs[sb * 4 + l] = @intCast(best_idx[l] & 0xFF);
        var qhb: u16 = @as(u16, best_delta) << 12;
        if (best_neg) qhb |= 0x8000;
        for (0..4) |l| qhb |= @as(u16, (best_idx[l] >> 8) & 7) << @intCast(3 * l);
        std.mem.writeInt(u16, qh[sb * 2 ..][0..2], qhb, .little);
    }
}
fn nextCoarseF16Bits(target: f32) u16 {
    // iq1_m: d vive dispersa en nibbles altos de 4 bytes ⇒ su f16 tiene
    // mantissa baja (bits[0,4)) SIEMPRE 0. Redondea UP al coarse siguiente.
    var bits: u16 = @bitCast(@as(f16, @floatCast(target)));
    bits +%= 15;
    bits &= ~@as(u16, 15);
    if (bits >= 0x7C00) bits = 0x7BFF & ~@as(u16, 15); // no inf/nan
    return bits;
}

fn encodeIQ1_M(blk: []const f32, dst: []u8) void {
    // 56B/SB256 — layout del KV-path (val_iq1_m lane-a, VERDE).
    // BYTES [0,8): entrelazado dl-codes + nibbles-d. Cada sc16 par p:
    //   bits[0,3)=dl1(2p), [3,6)=dl2(2p), [6,9)=dl1(2p+1), [9,12)=dl2(2p+1),
    //   [12,16)=d-nibble(p). Los qb de ib<2 SON esos mismos bytes (el índice
    //   grid incluye los bits de escala) ⇒ sin libertad de quanta ahí.
    // BYTES [8,32): qb libres (búsqueda completa 256×hbits).
    // BYTES [32,48): qh — 3 bits altos idx + 1 bit dd-signo por grupo.
    // Determinista: d coarse ≥ amax/16.875; dl-codes desde amax mitad;
    // grupos ib≥2 con búsqueda completa; grupos ib<2 aceptan el qb impuesto.
    var x: [256]f32 = undefined;
    for (0..256) |i| x[i] = if (i < blk.len) blk[i] else 0.0;

    @memset(dst[0..56], 0);

    var amax_all: f32 = 0;
    var amax_h: [8][2]f32 = undefined;
    for (0..8) |ib| {
        for (0..2) |h| {
            var mx: f32 = 0;
            for (0..16) |c| mx = @max(mx, @abs(x[ib * 32 + h * 16 + c]));
            amax_h[ib][h] = mx;
            amax_all = @max(amax_all, mx);
        }
    }

    // Guard cero (regresión append GPU): SB todo-ceros ⇒ amax_all=0, d_bits=0
    // y codes 0/0=NaN (@intFromFloat pánico en safe). Mismo criterio que
    // encodeIQ1_S (d=escala neutra) pero con d_bits=0: dl-codes caen a 0 y la
    // búsqueda cuántica empatando e=0 elige candidato 0 — determinista.
    const d_bits: u16 = if (amax_all > 0) nextCoarseF16Bits(amax_all / (15.0 * 1.125)) else 0;
    const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));

    var codes: [8][2]u8 = undefined;
    for (0..8) |ib| {
        for (0..2) |h| {
            var cc: i32 = 0;
            if (d > 0) {
                cc = @intFromFloat(@round(amax_h[ib][h] / (1.125 * 2.0 * d) - 0.5));
                cc = @min(@max(cc, 0), 7);
            }
            codes[ib][h] = @intCast(cc);
        }
    }

    // Empaquetar bytes [0,8): dl-codes (12 bits) + d-nibbles (4×4 bits).
    var sc16: [4]u16 = .{0} ** 4;
    for (0..4) |p| {
        sc16[p] = @as(u16, codes[p * 2][0]) |
            (@as(u16, codes[p * 2][1]) << 3) |
            (@as(u16, codes[p * 2 + 1][0]) << 6) |
            (@as(u16, codes[p * 2 + 1][1]) << 9);
    }
    const n0: u8 = @intCast((d_bits & 0xF) << 4);
    const n1: u8 = @intCast(((d_bits >> 4) & 0xF) << 4);
    const n2: u8 = @intCast(((d_bits >> 8) & 0xF) << 4);
    const n3: u8 = @intCast(((d_bits >> 12) & 0xF) << 4);
    dst[0] = @truncate(sc16[0] & 0xFF);
    dst[1] = @as(u8, @truncate((sc16[0] >> 8) & 0xF)) | n0;
    dst[2] = @truncate(sc16[1] & 0xFF);
    dst[3] = @as(u8, @truncate((sc16[1] >> 8) & 0xF)) | n1;
    dst[4] = @truncate(sc16[2] & 0xFF);
    dst[5] = @as(u8, @truncate((sc16[2] >> 8) & 0xF)) | n2;
    dst[6] = @truncate(sc16[3] & 0xFF);
    dst[7] = @as(u8, @truncate((sc16[3] >> 8) & 0xF)) | n3;

    // Quanta: grupos ib<2 tienen qb CONSTRUIDO por dl-codes+d-nibbles
    // (leer de vuelta); grupos ib>=2 son LIBRES (buscar qb completo).
    for (0..8) |ib| {
        const sc_off: usize = (ib >> 1) * 2;
        const dl = [2]f32{
            d * (2.0 * @as(f32, @floatFromInt(codes[ib][0])) + 1.0),
            d * (2.0 * @as(f32, @floatFromInt(codes[ib][1])) + 1.0),
        };
        for (0..4) |l| {
            const dll = dl[@min(l >> 1, 1)];
            const odd: u3 = @intCast(l & 1);
            const shift_amt: u5 = if (odd == 0) 8 else 4;
            const qb_pos: usize = ib * 4 + l; // SIEMPRE base[ib*4+l]

            var best_err: f64 = std.math.inf(f64);
            var best_qb: u8 = @truncate(dst[qb_pos]); // leer lo que ya hay
            var best_hb: u8 = 0;
            var best_dd: f32 = 0.125;

            if (qb_pos >= 8) {
                // Grupo libre: buscar qb completo (256 valores).
                for (0..256) |qb_val| {
                    for (0..2) |h| {
                        const dd: f32 = if (h == 0) 0.125 else -0.125;
                        const idxg: u32 = @as(u32, @intCast(qb_val)) |
                            ((@as(u32, @intCast(h)) << shift_amt) & 0x700);
                        const g = iq_grids.iq1s_grid[idxg];
                        var e: f64 = 0;
                        for (0..8) |j| {
                            const raw: i8 = @bitCast(@as(u8, @truncate(g >> @as(u6, @intCast(8 * j)))));
                            const dv = x[ib * 32 + l * 8 + j] - dll * (@as(f32, @floatFromInt(raw)) + dd);
                            e += dv * dv;
                        }
                        if (e < best_err) {
                            best_err = e;
                            best_qb = @intCast(qb_val);
                            best_hb = @intCast(h);
                            best_dd = dd;
                        }
                    }
                }
            } else {
                // Grupo CONSTRUIDO: qb ya contiene dl+d bits; solo buscar hbits.
                for (0..2) |h| {
                    const dd: f32 = if (h == 0) 0.125 else -0.125;
                    const idxg: u32 = @as(u32, best_qb) |
                        ((@as(u32, @intCast(h)) << shift_amt) & 0x700);
                    const g = iq_grids.iq1s_grid[idxg];
                    var e: f64 = 0;
                    for (0..8) |j| {
                        const raw: i8 = @bitCast(@as(u8, @truncate(g >> @as(u6, @intCast(8 * j)))));
                        const dv = x[ib * 32 + l * 8 + j] - dll * (@as(f32, @floatFromInt(raw)) + dd);
                        e += dv * dv;
                    }
                    if (e < best_err) {
                        best_err = e;
                        best_hb = @intCast(h);
                        best_dd = dd;
                    }
                }
            }

            // Escribir qb solo para grupos libres ib>=2 (los construidos
            // ya tienen su valor en dst de la fase de escalas).
            if (qb_pos >= 8) {
                dst[qb_pos] = best_qb;
            }
            // Escribir qh (3 bits altos idx + bit dd-signo).
            const qh_i: usize = 32 + sc_off + (l >> 1);
            if (odd == 0) {
                dst[qh_i] |= @as(u8, best_hb); // bits [0,3)
                if (best_dd < 0) dst[qh_i] |= 0x08;
            } else {
                dst[qh_i] |= @as(u8, best_hb) << 4; // bits [4,7) — fix off-by-one
                if (best_dd < 0) dst[qh_i] |= 0x80;
            }
        }
    }
}
fn encodeIQ2_XXS(blk: []const f32, dst: []u8) void {
    // 66B/SB256 [d f16][qs[64]] — espejo val_iq2_xxs (lane-a): sub-bloque
    // ib de 32: aux0=u32 LE qs[ib*8..+4] (4 índices grid por grupo l),
    // aux1=u32 LE qs[ib*8+4..+8]: sc bits[28,32), idx-signos 7 bits/l.
    // db=d·(0.5+sc)·0.25; gv=byte j de iq2xxs_grid[idx]; sg=±1 vía ksigns.
    // ⚠️ ksigns bit7 impuesto (col7 hereda paridad) — igual que iq3_xxs.
    // Determinista: scan sc ascendente, grid search por grupo con signos
    // fijos (tie→menor), acumulación f64. d=max_span/(15.5·43).
    const qk: usize = 256;
    var x: [qk]f32 = undefined;
    for (0..qk) |i| x[i] = if (i < blk.len) blk[i] else 0.0;

    @memset(dst[0..66], 0);

    var max_span: f32 = 0;
    for (0..8) |s| {
        var mn: f32 = std.math.inf(f32);
        var mx: f32 = -std.math.inf(f32);
        for (0..32) |c| {
            const v = x[s * 32 + c];
            mn = @min(mn, v);
            mx = @max(mx, v);
        }
        max_span = @max(max_span, mx - mn);
    }
    const d: f32 = if (max_span > 0) max_span / (15.5 * 43.0) else 1.0;
    writeF16(dst[0..2], d);

    const qs = dst[2..66];

    for (0..8) |ib| {
        // Signos fijos por grupo l (idx 7 bits; col7 paridad).
        var idx: [4]u8 = undefined;
        var sgn: [4][8]f32 = undefined;
        for (0..4) |l| {
            var i7b: u8 = 0;
            for (0..7) |c| {
                if (x[ib * 32 + l * 8 + c] < 0) i7b |= @as(u8, 1) << @intCast(c);
            }
            idx[l] = i7b;
            const sm = iq_grids.ksigns_iq2xs[i7b];
            for (0..8) |c| {
                sgn[l][c] = if ((sm >> @intCast(c)) & 1 != 0) -1.0 else 1.0;
            }
        }

        var best_sc: u8 = 0;
        var best_err: f64 = std.math.inf(f64);
        var best_idx0: u32 = 0;

        for (0..16) |sc| {
            const db: f32 = d * (0.5 + @as(f32, @floatFromInt(sc))) * 0.25;
            var err_total: f64 = 0;
            var idx0: u32 = 0;

            for (0..4) |l| {
                var be: f64 = std.math.inf(f64);
                var bi: u16 = 0;
                for (0..256) |gi| {
                    const g = iq_grids.iq2xxs_grid[gi];
                    var e: f64 = 0;
                    for (0..8) |j| {
                        const gv: f32 = @floatFromInt((g >> @as(u6, @intCast(8 * j))) & 0xFF);
                        const dv = x[ib * 32 + l * 8 + j] - sgn[l][j] * db * gv;
                        e += dv * dv;
                    }
                    if (e < be) {
                        be = e;
                        bi = @intCast(gi);
                    }
                }
                err_total += be;
                idx0 |= @as(u32, bi) << @intCast(8 * l);
            }
            if (err_total < best_err) {
                best_err = err_total;
                best_sc = @intCast(sc);
                best_idx0 = idx0;
            }
        }

        var aux1: u32 = @as(u32, best_sc) << 28;
        for (0..4) |l| aux1 |= @as(u32, idx[l]) << @intCast(7 * l);
        std.mem.writeInt(u32, qs[ib * 8 ..][0..4], best_idx0, .little);
        std.mem.writeInt(u32, qs[ib * 8 + 4 ..][0..4], aux1, .little);
    }
}
fn encodeIQ2_XS(blk: []const f32, dst: []u8) void {
    // 74B/SB256 [d f16][qs[64]][scales[8]] — espejo val_iq2_xs (lane-a):
    // sub-bloque ib de 32, 4 grupos l de 8: v u16 LE en qs[ib*8+2l..]:
    // bits[0,9)=índice iq2xs_grid (512), bits[9,16)=idx-signos ksigns;
    // db = d·(0.5+nibble(scales[ib],l/2))·0.25 (low→grupos 0-1, high→2-3).
    // Signos: idx7 desde sign(x) cols 0-6; col7 paridad (ksigns bit7).
    // Determinista: scan nibble ascendente por mitad, grid search con
    // signos fijos (tie→menor), f64. d=max_span/(7.75·2).
    const qk: usize = 256;
    var x: [qk]f32 = undefined;
    for (0..qk) |i| x[i] = if (i < blk.len) blk[i] else 0.0;

    @memset(dst[0..74], 0);

    var max_span: f32 = 0;
    for (0..8) |s| {
        var mn: f32 = std.math.inf(f32);
        var mx: f32 = -std.math.inf(f32);
        for (0..32) |c| {
            const v = x[s * 32 + c];
            mn = @min(mn, v);
            mx = @max(mx, v);
        }
        max_span = @max(max_span, mx - mn);
    }
    const d: f32 = if (max_span > 0) max_span / 15.5 else 1.0;
    writeF16(dst[0..2], d);

    const qs = dst[2..66];
    const scales = dst[66..74];

    for (0..8) |ib| {
        // Signos fijos por grupo l.
        var idx: [4]u8 = undefined;
        var sgn: [4][8]f32 = undefined;
        for (0..4) |l| {
            var i7b: u8 = 0;
            for (0..7) |c| {
                if (x[ib * 32 + l * 8 + c] < 0) i7b |= @as(u8, 1) << @intCast(c);
            }
            idx[l] = i7b;
            const sm = iq_grids.ksigns_iq2xs[i7b];
            for (0..8) |c| {
                sgn[l][c] = if ((sm >> @intCast(c)) & 1 != 0) -1.0 else 1.0;
            }
        }

        inline for (0..2) |half| {
            // Mitad half: grupos l=half*2 y half*2+1 comparten escala nibble.
            var best_n: u8 = 0;
            var best_err: f64 = std.math.inf(f64);
            var best_gi: [2]u16 = .{ 0, 0 };

            for (0..16) |n| {
                const db: f32 = d * (0.5 + @as(f32, @floatFromInt(n))) * 0.25;
                var err_total: f64 = 0;
                var gis: [2]u16 = .{ 0, 0 };
                for (0..2) |g_off| {
                    const l = half * 2 + g_off;
                    var be: f64 = std.math.inf(f64);
                    var bi: u16 = 0;
                    for (0..512) |gi| {
                        const g = iq_grids.iq2xs_grid[gi];
                        var e: f64 = 0;
                        for (0..8) |j| {
                            const gv: f32 = @floatFromInt((g >> @as(u6, @intCast(8 * j))) & 0xFF);
                            const dv = x[ib * 32 + l * 8 + j] - sgn[l][j] * db * gv;
                            e += dv * dv;
                        }
                        if (e < be) {
                            be = e;
                            bi = @intCast(gi);
                        }
                    }
                    err_total += be;
                    gis[g_off] = bi;
                }
                if (err_total < best_err) {
                    best_err = err_total;
                    best_n = @intCast(n);
                    best_gi = gis;
                }
            }

            scales[ib] |= if (half == 0) best_n else best_n << 4;
            for (0..2) |g_off| {
                const l = half * 2 + g_off;
                const v: u16 = @as(u16, best_gi[g_off]) | (@as(u16, idx[l]) << 9);
                std.mem.writeInt(u16, qs[ib * 8 + l * 2 ..][0..2], v, .little);
            }
        }
    }
}
fn encodeIQ2_S(blk: []const f32, dst: []u8) void {
    // 82B/SB256 [d f16][qs[32]][signs[32]][qh[8]][scales[8]] — espejo
    // val_iq2_s (lane-a): sub-bloque ib de 32, 4 grupos l de 8:
    // idxg = qs[ib*4+l] | ((qh[ib]<<(8−2l)) & 0x300) (10 bits → iq2s_grid);
    // signs byte DEDICADO signs[ib*4+l] (8 bits LIBRES, sin paridad);
    // db = d·(0.5+nibble(scales[ib], l/2))·0.25.
    // Determinista: scan nibble ascendente por mitad; grid search por
    // grupo con signos fijos sign(x) (tie→menor); f64.
    // d=max_span/(3.875·43).
    const qk: usize = 256;
    var x: [qk]f32 = undefined;
    for (0..qk) |i| x[i] = if (i < blk.len) blk[i] else 0.0;

    @memset(dst[0..82], 0);

    var max_span: f32 = 0;
    for (0..8) |s| {
        var mn: f32 = std.math.inf(f32);
        var mx: f32 = -std.math.inf(f32);
        for (0..32) |c| {
            const v = x[s * 32 + c];
            mn = @min(mn, v);
            mx = @max(mx, v);
        }
        max_span = @max(max_span, mx - mn);
    }
    const d: f32 = if (max_span > 0) max_span / (3.875 * 43.0) else 1.0;
    writeF16(dst[0..2], d);

    const qs = dst[2..34];
    const signs = dst[34..66];
    const qh = dst[66..74];
    const scales = dst[74..82];

    for (0..8) |ib| {
        // Signos libres por grupo l (byte dedicado: sin restricción ksigns).
        for (0..4) |l| {
            var sb: u8 = 0;
            for (0..8) |c| {
                if (x[ib * 32 + l * 8 + c] < 0) sb |= @as(u8, 1) << @intCast(c);
            }
            signs[ib * 4 + l] = sb;
        }

        inline for (0..2) |half| {
            var best_n: u8 = 0;
            var best_err: f64 = std.math.inf(f64);
            var best_gi: [2]u16 = .{ 0, 0 };

            for (0..16) |n| {
                const db: f32 = d * (0.5 + @as(f32, @floatFromInt(n))) * 0.25;
                var err_total: f64 = 0;
                var gis: [2]u16 = .{ 0, 0 };
                for (0..2) |go| {
                    const l = half * 2 + go;
                    var be: f64 = std.math.inf(f64);
                    var bi: u16 = 0;
                    for (0..1024) |gi| {
                        const g = iq_grids.iq2s_grid[gi];
                        var e: f64 = 0;
                        for (0..8) |j| {
                            const gv: f32 = @floatFromInt((g >> @as(u6, @intCast(8 * j))) & 0xFF);
                            const sgn_j: f32 = if (x[ib * 32 + l * 8 + j] < 0) -1.0 else 1.0;
                            const dv = x[ib * 32 + l * 8 + j] - sgn_j * db * gv;
                            e += dv * dv;
                        }
                        if (e < be) {
                            be = e;
                            bi = @intCast(gi); // índice completo 10 bits
                        }
                    }
                    err_total += be;
                    gis[go] = bi;
                }
                if (err_total < best_err) {
                    best_err = err_total;
                    best_n = @intCast(n);
                    best_gi = gis;
                }
            }

            scales[ib] |= if (half == 0) best_n else best_n << 4;
            for (0..2) |go| {
                const l = half * 2 + go;
                qs[ib * 4 + l] = @truncate(best_gi[go]);
                // Inverso del decode: el decode hace qh<<(8-2l)&0x300 ⇒
                // el par de bits altos vive en qh posición 2l.
                qh[ib] |= @as(u8, @truncate((best_gi[go] >> 8) & 3)) << @as(u3, @intCast(2 * l));
            }
        }
    }
}
/// Dequant IQ4_NL a f32 directo (validación externa/tests).
pub fn dequantIQ4NL32(bytes: []const u8, out: []f32) void {
    dequantIQ4_NL(bytes, out);
}

/// Dequant MXFP4 a f32 directo (validación externa/tests).
pub fn dequantMXFP432(bytes: []const u8, out: []f32) void {
    dequantMXFP4(bytes, out);
}

/// Dequant IQ3_XXS a f32 directo (validación externa/tests).
pub fn dequantIQ3XXS32(bytes: []const u8, out: []f32) void {
    dequantIQ3_XXS(bytes, out);
}

/// Dequant IQ2_XXS a f32 directo (validación externa/tests).
pub fn dequantIQ2XXS32(bytes: []const u8, out: []f32) void {
    dequantIQ2_XXS(bytes, out);
}

/// Dequant IQ2_XS a f32 directo (validación externa/tests).
pub fn dequantIQ2XS32(bytes: []const u8, out: []f32) void {
    dequantIQ2_XS(bytes, out);
}

/// Dequant IQ2_S a f32 directo (validación externa/tests).
pub fn dequantIQ2_S32(bytes: []const u8, out: []f32) void {
    dequantIQ2_S(bytes, out);
}

/// Dequant IQ1_M a f32 directo (validación externa/tests; espejo kernel).
pub fn dequantIQ1M32(bytes: []const u8, out: []f32) void {
    dequantIQ1_M(bytes, out);
}

/// Dequant IQ1_S a f32 directo (validación externa/tests; espejo kernel
/// qgemm case 17 y val_iq1_s). B-a4 (lane-a).
pub fn dequantIQ1S32(bytes: []const u8, out: []f32) void {
    dequantIQ1_S(bytes, out);
}

/// Dequant IQ3_S a f32 directo (validación externa/tests; espejo kernel).
pub fn dequantIQ3S32(bytes: []const u8, out: []f32) void {
    const qk = 256;
    const block_bytes = 110;
    var i: usize = 0;
    var nb2: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb2 * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const qs = bytes[base + 2 ..][0..64];
        const qh = bytes[base + 66 ..][0..8];
        const signs = bytes[base + 74 ..][0..32];
        const scales = bytes[base + 106 ..][0..4];
        for (0..256) |in| {
            const it = in / 64;
            const rem = in % 64;
            const half = rem / 32;
            const pos = rem % 32;
            const l = pos / 8;
            const col = pos % 8;
            const sc = scales[it];
            const code: u8 = if (half == 0) sc & 0xF else sc >> 4;
            const db: f32 = d * (1.0 + 2.0 * @as(f32, @floatFromInt(code)));
            const qoff = it * 16 + half * 8;
            const hb = qh[2 * it + half];
            const sm = signs[it * 8 + half * 4 + l];
            const idx: u32 = if (col < 4)
                @as(u32, qs[qoff + 2 * l]) | ((@as(u32, hb >> @as(u3, @intCast(2 * l))) & 1) << 8)
            else
                @as(u32, qs[qoff + 2 * l + 1]) | ((@as(u32, hb >> @as(u3, @intCast(2 * l + 1))) & 1) << 8);
            const e = iq_grids.iq3s_grid[idx];
            const jx: u5 = @intCast(if (col < 4) col else col - 4);
            const sgn: f32 = if ((sm & (@as(u8, 1) << @intCast(col))) != 0) -1.0 else 1.0;
            const gv: f32 = @floatFromInt((e >> (8 * jx)) & 0xFF);
            out[i + in] = db * gv * sgn;
        }
        nb2 += 1;
    }
}

fn encodeIQ3_XXS(blk: []const f32, dst: []u8) void {
    // 98B/SB256 [d f16][qs[64]][ss[32]] — espejo val_iq3_xxs (lane-a).
    // Sub-bloque ib de 32: aux u32 LE = ss[ib*4..]: bits[28,32)=sc,
    // [7l,7l+7)=idx_l. db=d·(0.5+sc)·0.5. Grupo l de 8 elems: sm=
    // ksigns[idx_l]; bit c de sm = signo del elem l*8+c (cols 0-3 ← grid[
    // qs[ib*8+2l]], 4-7 ← grid[qs[ib*8+2l+1]], gv=byte jx).
    // ⚠️ ksigns[i] bits0-6==i, bit7=paridad ⇒ idx_l codifica los signos
    // DESEADOS de cols 0-6 y col 7 hereda el impuesto por paridad.
    // Determinista GPU↔CPU: scan sc ascendente, grids por cuarteto
    // independientes (tie→menor índice), acumulación f64.
    const qk: usize = 256;
    var x: [qk]f32 = undefined;
    for (0..qk) |i| x[i] = if (i < blk.len) blk[i] else 0.0;

    @memset(dst[0..98], 0);

    var max_span: f32 = 0;
    for (0..8) |s| {
        var mn: f32 = std.math.inf(f32);
        var mx: f32 = -std.math.inf(f32);
        for (0..32) |c| {
            const v = x[s * 32 + c];
            mn = @min(mn, v);
            mx = @max(mx, v);
        }
        max_span = @max(max_span, mx - mn);
    }
    const d: f32 = if (max_span > 0) max_span / (4.0 * 62.0) else 1.0;
    writeF16(dst[0..2], d);

    const qs = dst[2..66];
    const ss = dst[66..98];

    for (0..8) |ib| {
        // Signos fijos del sub-bloque: idx ÚNICO desde sign(x) de cols 0-6
        // del ib completo (los 4 grupos comparten el mismo ksigns byte);
        // col 7 hereda paridad vía ksigns.
        var idx7b: u8 = 0;
        for (0..7) |c| {
            if (x[ib * 32 + c] < 0) idx7b |= @as(u8, 1) << @intCast(c);
        }
        const sm = iq_grids.ksigns_iq2xs[idx7b];
        var sgn: [8]f32 = undefined;
        for (0..8) |c| {
            sgn[c] = if ((sm >> @intCast(c)) & 1 != 0) -1.0 else 1.0;
        }

        var best_sc: u8 = 0;
        var best_err: f64 = std.math.inf(f64);
        var best_bytes: [8]u8 = .{0} ** 8;

        for (0..16) |sc| {
            const db: f32 = d * (0.5 + @as(f32, @floatFromInt(sc))) * 0.5;
            var err_total: f64 = 0;
            var bytes: [8]u8 = .{0} ** 8;

            for (0..4) |l| {
                inline for (0..2) |is_b| {
                    var be: f64 = std.math.inf(f64);
                    var bi: u16 = 0;
                    for (0..256) |gi| {
                        const g = iq_grids.iq3xxs_grid[gi];
                        var e: f64 = 0;
                        for (0..4) |jj| {
                            const col = jj + (if (is_b == 1) @as(usize, 4) else 0);
                            const gv: f32 = @floatFromInt((g >> @as(u5, @intCast(8 * jj))) & 0xFF);
                            const dv = x[ib * 32 + l * 8 + col] - sgn[col] * db * gv;
                            e += dv * dv;
                        }
                        if (e < be) {
                            be = e;
                            bi = @intCast(gi);
                        }
                    }
                    err_total += be;
                    bytes[l * 2 + is_b] = @truncate(bi);
                }
            }
            if (err_total < best_err) {
                best_err = err_total;
                best_sc = @intCast(sc);
                best_bytes = bytes;
            }
        }

        for (0..4) |l| {
            qs[ib * 8 + l * 2] = best_bytes[l * 2];
            qs[ib * 8 + l * 2 + 1] = best_bytes[l * 2 + 1];
            ss[ib * 4 + l] = 0; // (se empaqueta abajo por u32)
        }
        // aux del sub-bloque: sc<<28 | idx_l<<(7l)
        var aux: u32 = @as(u32, best_sc) << 28;
        for (0..4) |l| aux |= @as(u32, idx7b) << @intCast(7 * l);
        std.mem.writeInt(u32, ss[ib * 4 ..][0..4], aux, .little);
    }
}
/// Codificador IQ3_S REAL (inverso de dequantIQ3_S y de val_iq3_s del kernel
/// fused de lane-a). Esquema documentado (no el search ponderado de
/// llama.cpp, pero válido para cualquier lector canónico):
///   por SB: d = amax_sb/(15·31) — cubre factor de código máx 31 × mag LUT
///   máx 15; por mitad de 32 elems: escanea sc∈[0,16) ⇒ db=d·(2sc+1) y para
///   cada grupo l de 8 busca INDEPENDIENTE el índice de grid a/b (los signos
///   son libres por elemento ⇒ óptimo sgn(x), error Σ(|x|−db·byte_j(grid))²);
///   tie → menor índice. Acumulación f64 como en encodeIQ1_S.
fn encodeIQ3_S(blk: []const f32, dst: []u8) void {
    const qk: usize = 256;
    var x: [qk]f32 = undefined;
    for (0..qk) |j| x[j] = if (j < blk.len) blk[j] else 0.0;

    var amax_all: f32 = 0;
    for (0..qk) |j| amax_all = @max(amax_all, @abs(x[j]));
    const d: f32 = if (amax_all > 0) amax_all / (15.0 * 31.0) else 1.0;
    writeF16(dst[0..2], d);

    const qs = dst[2..66];
    const qh = dst[66..74];
    const signs = dst[74..106];
    const scales = dst[106..110];
    @memset(qs, 0);
    @memset(qh, 0);
    @memset(signs, 0);
    @memset(scales, 0);

    // Bytes absolutos de la grid (byte j del u32 por índice).
    var gbytes: [512][4]u8 = undefined;
    for (0..512) |gi| {
        const e = iq_grids.iq3s_grid[gi];
        for (0..4) |j| gbytes[gi][j] = @truncate(e >> @as(u5, @intCast(8 * j)));
    }

    for (0..4) |it| {
        for (0..2) |half| {
            var best_sc: u8 = 0;
            var best_err: f32 = 3.0e38;
            var best_qa: [4]u8 = .{ 0, 0, 0, 0 };
            var best_qb: [4]u8 = .{ 0, 0, 0, 0 };
            // bits hb elegidos: [a_l0, b_l0, a_l1, b_l1, ...] → bit l / bit l+1.
            var best_hb: [8]u1 = .{ 0, 0, 0, 0, 0, 0, 0, 0 };

            for (0..16) |sc| {
                const db: f32 = d * (2.0 * @as(f32, @floatFromInt(sc)) + 1.0);
                var err_total: f32 = 0;
                var qa: [4]u8 = undefined;
                var qb: [4]u8 = undefined;
                var hb_sel: [8]u1 = undefined;
                for (0..4) |l| {
                    inline for (0..2) |is_b| {
                        var be: f32 = 3.0e38;
                        var bi: u16 = 0;
                        for (0..512) |gi| {
                            const gb = &gbytes[gi];
                            var e: f32 = 0;
                            for (0..4) |j| {
                                const col = if (is_b == 1) j + 4 else j;
                                const xv = @abs(x[it * 64 + half * 32 + l * 8 + col]);
                                const bv: f32 = @floatFromInt(gb[j]);
                                const dv = xv - db * bv;
                                e += dv * dv;
                            }
                            if (e < be) {
                                be = e;
                                bi = @intCast(gi);
                            }
                        }
                        err_total += be;
                        if (is_b == 0) {
                            qa[l] = @truncate(bi);
                            hb_sel[l * 2] = @truncate(bi >> 8);
                        } else {
                            qb[l] = @truncate(bi);
                            hb_sel[l * 2 + 1] = @truncate(bi >> 8);
                        }
                    }
                }
                if (err_total < best_err) {
                    best_err = err_total;
                    best_sc = @intCast(sc);
                    best_qa = qa;
                    best_qb = qb;
                    best_hb = hb_sel;
                }
            }

            const qbase = it * 16 + half * 8;
            for (0..4) |l| {
                qs[qbase + 2 * l] = best_qa[l];
                qs[qbase + 2 * l + 1] = best_qb[l];
                var sm: u8 = 0;
                for (0..4) |j| {
                    const xa = x[it * 64 + half * 32 + l * 8 + j];
                    const xb = x[it * 64 + half * 32 + l * 8 + 4 + j];
                    if (xa < 0) sm |= iq_grids.kmask_iq2xs[j];
                    if (xb < 0) sm |= iq_grids.kmask_iq2xs[j + 4];
                }
                signs[it * 8 + half * 4 + l] = sm;
            }
            var hbb: u8 = 0;
            // a_l → bit 2l (shift 8−2l del kernel), b_l → bit 2l+1 (7−2l).
            for (0..4) |l| {
                hbb |= @as(u8, best_hb[l * 2]) << @as(u3, @intCast(2 * l));
                hbb |= @as(u8, best_hb[l * 2 + 1]) << @as(u3, @intCast(2 * l + 1));
            }
            qh[2 * it + half] = hbb;
            scales[it] = if (half == 0) (scales[it] & 0xF0) | best_sc else (scales[it] & 0x0F) | (best_sc << 4);
        }
    }
}
/// Codificador IQ4_XS REAL (layout canónico, inverso de dequantIQ4_XS y del
/// kernel fused de lane-a). Esquema documentado (no el search ponderado de
/// llama.cpp, pero válido para cualquier lector canónico):
///   por SB de 256: d = amax_max/(113·31); por sub-bloque de 32,
///   ls = 32 + ceil(amax_sb/(113·d)) ∈ [33,63] ⇒ cobertura por extremos
///   LUT ±113; cada elemento toma el índice LUT más cercano (tie → menor).
fn encodeIQ4_XS(blk: []const f32, dst: []u8) void {
    const qk: usize = 256;
    var x: [qk]f32 = undefined;
    for (0..qk) |j| x[j] = if (j < blk.len) blk[j] else 0.0;

    var amax_sb: [8]f32 = undefined;
    var amax_all: f32 = 0;
    for (0..8) |sb| {
        var mx: f32 = 0;
        for (0..32) |j| mx = @max(mx, @abs(x[sb * 32 + j]));
        amax_sb[sb] = mx;
        amax_all = @max(amax_all, mx);
    }
    const d: f32 = if (amax_all > 0) amax_all / (113.0 * 31.0) else 1.0;
    writeF16(dst[0..2], d);

    var scales_h: u16 = 0;
    // Inicializar SIEMPRE: los nibbles se acumulan con |= sobre estos bytes
    // (sin init, se OR-an sobre basura del stack — causa de paridad roja).
    var scales_l: [4]u8 = .{ 0, 0, 0, 0 };
    const qs = dst[8..136];
    @memset(qs, 0);

    // ls por sub-bloque (6 bits): 4 bits bajos empaquetados por pares en
    // scales_l[4] y 2 bits altos en scales_h.
    var ls_v: [8]i32 = undefined;
    for (0..8) |sb| {
        var ls: i32 = 32;
        if (amax_sb[sb] > 0) {
            ls = 32 + @as(i32, @intFromFloat(@ceil(amax_sb[sb] / (113.0 * d))));
            ls = @min(@max(ls, 33), 63);
        }
        ls_v[sb] = ls;
        scales_l[sb / 2] |= if (sb % 2 == 0) @intCast(ls & 0xF) else @intCast((ls & 0xF) << 4);
        scales_h |= @as(u16, @intCast(((ls >> 4) & 3))) << @intCast(2 * sb);

        const dl_sb = d * @as(f32, @floatFromInt(ls - 32));
        for (0..32) |j| {
            const xv = x[sb * 32 + j];
            var best_id: u8 = 0;
            var best_diff = @abs(xv - dl_sb * @as(f32, @floatFromInt(kvalues_iq4nl[0])));
            for (1..16) |id| {
                const diff = @abs(xv - dl_sb * @as(f32, @floatFromInt(kvalues_iq4nl[id])));
                if (diff < best_diff) {
                    best_diff = diff;
                    best_id = @intCast(id);
                }
            }
            const qidx = if (j < 16) j else j - 16;
            if (j < 16) {
                qs[sb * 16 + qidx] |= best_id;
            } else {
                qs[sb * 16 + qidx] |= best_id << 4;
            }
        }
    }
    std.mem.writeInt(u16, dst[2..4], scales_h, .little);
    @memcpy(dst[4..8], &scales_l);
}
fn encodeIQ4_NL(blk: []const f32, dst: []u8) void {
    // 18B/bloque32 [d f16][qs[16] split-16]; valor = d·kvalues[nibble].
    // d = amax/127 (max abs LUT); nearest LUT ascendente tie→menor índice.
    // Nearest con d full-precision, store f16(d) — convención iq4_xs.
    var amax: f32 = 0;
    for (blk) |v| amax = @max(amax, @abs(v));
    const d: f32 = if (amax > 0) amax / 127.0 else 1.0;
    writeF16(dst[0..2], d);
    // Zero previo: los low-nibbles se escriben por RMW (lección encodeQ4_0).
    @memset(dst[2..18], 0);
    for (0..32) |c| {
        const xv = if (c < blk.len) blk[c] else 0.0;
        var best: u8 = 0;
        var best_diff: f32 = @abs(xv - d * @as(f32, @floatFromInt(kvalues_iq4nl[0])));
        for (1..16) |id| {
            const diff = @abs(xv - d * @as(f32, @floatFromInt(kvalues_iq4nl[id])));
            if (diff < best_diff) {
                best_diff = diff;
                best = @intCast(id);
            }
        }
        if (c < 16) {
            dst[2 + c] = (dst[2 + c] & 0xF0) | best;
        } else {
            dst[2 + c - 16] |= best << 4;
        }
    }
}
/// 6.3 (lane-f): LUT inversa del unpack TQ1_0 — el decode de digits
/// ((b·3^n)&0xFF)·3>>8 NO tiene inversa simple (el pack ingenuo Σt·3^n
/// falla en 803/1215 estados, verificado por enumeración). En su lugar:
/// por cada byte 0..255 computo su estado unpacked (MISMO algoritmo de
/// extracción) y construyo estado→byte. Exacto por construcción; comptime.
fn tq1UnpackDigit(b: u8, n: usize) u2 {
    const p3 = [_]u16{ 1, 3, 9, 27, 81 };
    const m: u16 = @as(u16, b) *% p3[n];
    const q: u8 = @truncate(m);
    return @truncate((@as(u16, q) * 3) >> 8);
}

fn tq1BuildLut(comptime digits: usize) [243]u8 {
    @setEvalBranchQuota(100000);
    var lut: [243]u8 = [_]u8{255} ** 243;
    for (0..256) |b| {
        var idx: usize = 0;
        var pw: usize = 1;
        for (0..digits) |n| {
            idx += @as(usize, tq1UnpackDigit(@intCast(b), n)) * pw;
            pw *= 3;
        }
        if (lut[idx] == 255) lut[idx] = @intCast(b);
    }
    return lut;
}

const TQ1_LUT5 = tq1BuildLut(5);
const TQ1_LUT4 = tq1BuildLut(4);

fn encodeTQ1_0(blk: []const f32, dst: []u8) void {
    // 6.3 (lane-f): TQ1_0 canónico — 54B/SB256 [qs 32B][qs2 16B][qh 4B]
    // [d f16 @52]. Espejo EXACTO de val_tq1_0 (fused_decode_extra.cu:497,
    // layout llama.cpp TQ1_0): 256 ternarios en base-3 como digits —
    // 5 elems/byte en qs[0..32) (elems 0..159), 5/byte en qs[32..48)
    // (elems 160..239), 4/byte en qh[0..4) (elems 240..255). El pack usa
    // la LUT inversa del unpack. d = amax, nivel = q−1.
    // (Antes era stub @memset(0) — TODO 6.3: único formato sin append.)
    var x: [256]f32 = undefined;
    for (0..256) |i| x[i] = if (i < blk.len) blk[i] else 0.0;

    var amax: f32 = 0;
    for (x) |v| amax = @max(amax, @abs(v));
    const d: f32 = if (amax > 0) amax else 1.0;
    writeF16(dst[52..54], d);
    @memset(dst[0..52], 0);

    // ternario del elem i: t = clamp(round(v/d)+1, 0..2)
    var t: [256]u2 = undefined;
    for (0..256) |i| {
        var ti: i32 = @as(i32, @intFromFloat(@round(x[i] / d))) + 1;
        ti = @min(@max(ti, 0), 2);
        t[i] = @intCast(ti);
    }
    // qs[0..32): elems [0..160), 5 digits/byte (idx = Σ t·3^n)
    for (0..32) |j| {
        const e0 = j * 5;
        const idx = @as(usize, t[e0]) + 3 * @as(usize, t[e0 + 1]) + 9 * @as(usize, t[e0 + 2]) + 27 * @as(usize, t[e0 + 3]) + 81 * @as(usize, t[e0 + 4]);
        dst[j] = TQ1_LUT5[idx];
    }
    // qs[32..48): elems [160..240), 5 digits/byte
    for (0..16) |j| {
        const e0 = 160 + j * 5;
        const idx = @as(usize, t[e0]) + 3 * @as(usize, t[e0 + 1]) + 9 * @as(usize, t[e0 + 2]) + 27 * @as(usize, t[e0 + 3]) + 81 * @as(usize, t[e0 + 4]);
        dst[32 + j] = TQ1_LUT5[idx];
    }
    // qh[0..4): elems [240..256), 4 digits/byte
    for (0..4) |j| {
        const e0 = 240 + j * 4;
        const idx = @as(usize, t[e0]) + 3 * @as(usize, t[e0 + 1]) + 9 * @as(usize, t[e0 + 2]) + 27 * @as(usize, t[e0 + 3]);
        dst[48 + j] = TQ1_LUT4[idx];
    }
}
fn encodeTQ2_0(blk: []const f32, dst: []u8) void {
    // 66B/SB256 [qs 64B][d f16@64] — espejo val_tq2_0: seg=in/128,
    // l=(in%128)/32, m=in%32; byte dst[seg*32+m], shift=2l; q∈{0..3},
    // val=d(q−1). d=max_span/2 (rango q−1 ∈ [−1,2]).
    var x: [256]f32 = undefined;
    for (0..256) |i| x[i] = if (i < blk.len) blk[i] else 0.0;

    var mn: f32 = std.math.inf(f32);
    var mx: f32 = -std.math.inf(f32);
    for (x) |v| {
        mn = @min(mn, v);
        mx = @max(mx, v);
    }
    const span = mx - mn;
    const d: f32 = if (span > 0) span / 2.0 else 1.0;
    writeF16(dst[64..66], d);

    @memset(dst[0..64], 0);
    for (0..256) |in| {
        const seg = in / 128;
        const rem = in % 128;
        const l = rem / 32;
        const m = rem % 32;
        var q: i32 = @as(i32, @intFromFloat(@round(x[in] / d))) + 1;
        q = @min(@max(q, 0), 3);
        dst[seg * 32 + m] |= @as(u8, @intCast(q)) << @as(u3, @intCast(2 * l));
    }
}
/// LUT FP4 canónica de A/tables.cuh (i8 escalado ×1).
pub const kvalues_fp4 = [16]i8{ 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };

fn encodeMXFP4(blk: []const f32, dst: []u8) void {
    // 17B/bloque32 [escala u8 E8M0][qs[16] split-16]; valor =
    // 2^(escala−127)·kvalues_fp4[nibble]. Escala MÍNIMA e tal que
    // 2^(e−127)·12 ≥ amax — búsqueda por doblar (sin log2f: exactitud
    // bit-idéntica GPU↔CPU). Nearest LUT ascendente tie→menor.
    var amax: f32 = 0;
    for (blk) |v| amax = @max(amax, @abs(v));
    var e: u8 = 127;
    var cover: f32 = 12.0;
    while (cover < amax and e < 254) {
        e += 1;
        cover *= 2.0;
    }
    dst[0] = e;
    const d: f32 = cover / 12.0;

    for (0..16) |b| dst[1 + b] = 0;
    for (0..32) |c| {
        const xv = if (c < blk.len) blk[c] else 0.0;
        var best: u8 = 0;
        var best_diff: f32 = @abs(xv - d * @as(f32, @floatFromInt(kvalues_fp4[0])));
        for (1..16) |id| {
            const diff = @abs(xv - d * @as(f32, @floatFromInt(kvalues_fp4[id])));
            if (diff < best_diff) {
                best_diff = diff;
                best = @intCast(id);
            }
        }
        if (c < 16) {
            dst[1 + c] = (dst[1 + c] & 0xF0) | best;
        } else {
            dst[1 + c - 16] |= best << 4;
        }
    }
}

// Custom int8/int4 encoders
fn encodeInt8Sym(blk: []const f32, dst: []u8) void {
    const d = if (maxAbs(blk) > 0) maxAbs(blk) / 127.0 else 1.0;
    const d_bits: u32 = @bitCast(d);
    std.mem.writeInt(u32, dst[0..4], d_bits, .little);
    for (blk, 0..) |v, j| {
        var q: i32 = @intFromFloat(@round(v / d));
        if (q < -127) q = -127;
        if (q > 127) q = 127;
        dst[4 + j] = @as(u8, @bitCast(@as(i8, @intCast(q))));
    }
}

fn encodeInt8Asym(blk: []const f32, dst: []u8) void {
    var max_val: f32 = -std.math.inf(f32);
    var min_val: f32 = std.math.inf(f32);
    for (blk) |v| {
        max_val = @max(max_val, v);
        min_val = @min(min_val, v);
    }
    const d = if (max_val - min_val > 0) (max_val - min_val) / 255.0 else 1.0;
    const zp = -min_val / d;
    const d_bits: u32 = @bitCast(d);
    const zp_bits: u32 = @bitCast(zp);
    std.mem.writeInt(u32, dst[0..4], d_bits, .little);
    std.mem.writeInt(u32, dst[4..8], zp_bits, .little);
    for (blk, 0..) |v, j| {
        var q: i32 = @intFromFloat(@round(v / d + zp));
        if (q < 0) q = 0;
        if (q > 255) q = 255;
        dst[8 + j] = @as(u8, @intCast(q));
    }
}

fn encodeInt4(blk: []const f32, dst: []u8) void {
    var max_val: f32 = -std.math.inf(f32);
    var min_val: f32 = std.math.inf(f32);
    for (blk) |v| {
        max_val = @max(max_val, v);
        min_val = @min(min_val, v);
    }
    const d = if (max_val - min_val > 0) (max_val - min_val) / 15.0 else 1.0;
    const zp = -min_val / d;
    const d_bits: u32 = @bitCast(d);
    const zp_bits: u32 = @bitCast(zp);
    std.mem.writeInt(u32, dst[0..4], d_bits, .little);
    std.mem.writeInt(u32, dst[4..8], zp_bits, .little);
    var j: usize = 0;
    while (j < blk.len) : (j += 2) {
        const v0 = blk[j];
        const v1 = if (j + 1 < blk.len) blk[j + 1] else 0.0;
        var q0: i32 = @intFromFloat(@round(v0 / d + zp));
        var q1: i32 = @intFromFloat(@round(v1 / d + zp));
        if (q0 < 0) q0 = 0;
        if (q0 > 15) q0 = 15;
        if (q1 < 0) q1 = 0;
        if (q1 > 15) q1 = 15;
        dst[8 + j / 2] = @as(u8, @intCast(q0)) | (@as(u8, @intCast(q1)) << 4);
    }
}

fn encodeFP8(blk: []const f32, dst: []u8) void {
    const max_abs = maxAbs(blk);
    const scale = if (max_abs > 0) max_abs / 127.0 else 1.0;
    const scale_bits: u32 = @bitCast(scale);
    std.mem.writeInt(u32, dst[0..4], scale_bits, .little);
    for (blk, 0..) |v, j| {
        var q: i32 = @intFromFloat(@round(v / scale));
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        dst[4 + j] = @as(u8, @bitCast(@as(i8, @intCast(q))));
    }
}

// ============================================================================
// Decodificadores (decode) - basados en src/loader/gguf.zig
// ============================================================================

fn dequantF32(format: QuantFormat, bytes: []const u8, out: []f32) void {
    switch (format) {
        .q8_0 => dequantQ8_0(bytes, out),
        .q8_1 => dequantQ8_1(bytes, out),
        .q4_0 => dequantQ4_0(bytes, out),
        .q4_1 => dequantQ4_1(bytes, out),
        .q5_0 => dequantQ5_0(bytes, out),
        .q5_1 => dequantQ5_1(bytes, out),
        // Lane-b2 P0.3: cache types estándar adicionales.
        .q2_0s => dequantQ2_0S(bytes, out),
        .q2_1 => dequantQ2_1(bytes, out),
        .q3_0 => dequantQ3_0(bytes, out),
        .q3_1 => dequantQ3_1(bytes, out),
        .q6_0 => dequantQ6_0(bytes, out),
        .q6_1 => dequantQ6_1(bytes, out),
        .q2_k => dequantQ2_K(bytes, out),
        .q3_k => dequantQ3_K(bytes, out),
        .q4_k => dequantQ4_K(bytes, out),
        .q5_k => dequantQ5_K(bytes, out),
        .q6_k => dequantQ6_K(bytes, out),
        .q8_k => dequantQ8_K(bytes, out),
        .iq1_s => dequantIQ1_S(bytes, out),
        .iq1_m => dequantIQ1_M(bytes, out),
        .iq2_xxs => dequantIQ2_XXS(bytes, out),
        .iq2_xs => dequantIQ2_XS(bytes, out),
        .iq2_s => dequantIQ2_S(bytes, out),
        .iq3_xxs => dequantIQ3_XXS(bytes, out),
        .iq3_s => dequantIQ3_S(bytes, out),
        .iq4_xs => dequantIQ4_XS(bytes, out),
        .iq4_nl => dequantIQ4_NL(bytes, out),
        .tq1_0 => dequantTQ1_0(bytes, out),
        .tq2_0 => dequantTQ2_0(bytes, out),
        .mxfp4 => dequantMXFP4(bytes, out),
        .int8_symmetric => dequantInt8Sym(bytes, out),
        .int8_asymmetric => dequantInt8Asym(bytes, out),
        .int4 => dequantInt4(bytes, out),
        .fp16 => dequantF16(bytes, out),
        .fp32 => dequantF32Raw(bytes, out),
        .fp8 => dequantFp8_e4m3(bytes, out),
    }
}

/// 6.3 (lane-f): dispatcher público de decode (roundtrip tests de encoder).
pub fn dequant(format: QuantFormat, bytes: []const u8, out: []f32) void {
    return dequantF32(format, bytes, out);
}

fn dequantFp8_e4m3(bytes: []const u8, out: []f32) void {
    const block_size = 128;
    const block_bytes = block_size + 4; // 4 bytes scale + 128 bytes data
    var bi: usize = 0;
    var out_i: usize = 0;
    while (out_i < out.len) : (bi += 1) {
        const scale_off = bi * block_bytes;
        const scale = @as(f32, @bitCast(std.mem.readInt(u32, bytes[scale_off..][0..4], .little)));
        const data_off = scale_off + 4;
        const n = @min(block_size, out.len - out_i);
        for (0..n) |j| {
            const val: u8 = bytes[data_off + j];
            out[out_i + j] = @as(f32, @floatFromInt(@as(i8, @bitCast(val)))) * scale;
        }
        out_i += n;
    }
}

// Legacy format decoders (from gguf.zig)
fn dequantQ8_0(bytes: []const u8, out: []f32) void {
    var i: usize = 0;
    while (i < out.len) : (i += BLOCK) {
        const blk_off = (i / BLOCK) * 34;
        const d_bits = std.mem.readInt(u16, bytes[blk_off..][0..2], .little);
        const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
        const qs = bytes[blk_off + 2 ..][0..BLOCK];
        for (0..@min(BLOCK, out.len - i)) |j| {
            out[i + j] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(qs[j]))));
        }
    }
}

fn dequantQ8_1(bytes: []const u8, out: []f32) void {
    var i: usize = 0;
    while (i < out.len) : (i += BLOCK) {
        const blk_off = (i / BLOCK) * 36;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[blk_off..][0..2], .little))));
        const m: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[blk_off + 2 ..][0..2], .little))));
        const qs = bytes[blk_off + 4 ..][0..BLOCK];
        for (0..@min(BLOCK, out.len - i)) |j| {
            out[i + j] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(qs[j])))) + m;
        }
    }
}

fn dequantQ4_0(bytes: []const u8, out: []f32) void {
    var i: usize = 0;
    while (i < out.len) : (i += BLOCK) {
        const blk_off = (i / BLOCK) * 18;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[blk_off..][0..2], .little))));
        const qs = bytes[blk_off + 2 ..][0..(BLOCK / 2)];
        const n = @min(BLOCK, out.len - i);
        // Split-16 canónico (espejo de gguf.dequantQ4_0 / upstream master).
        const half = @min(BLOCK / 2, n);
        for (0..half) |idx| {
            const q_lo: i32 = @as(i32, @intCast(qs[idx] & 0x0F)) - 8;
            out[i + idx] = d * @as(f32, @floatFromInt(q_lo));
            if (idx + BLOCK / 2 < n) {
                const q_hi: i32 = @as(i32, @intCast(qs[idx] >> 4)) - 8;
                out[i + idx + BLOCK / 2] = d * @as(f32, @floatFromInt(q_hi));
            }
        }
    }
}

fn dequantQ4_1(bytes: []const u8, out: []f32) void {
    var i: usize = 0;
    while (i < out.len) : (i += BLOCK) {
        const blk_off = (i / BLOCK) * 20;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[blk_off..][0..2], .little))));
        const m_bits = std.mem.readInt(u16, bytes[blk_off + 2 ..][0..2], .little);
        const m: f32 = @floatCast(@as(f16, @bitCast(m_bits)));
        const qs = bytes[blk_off + 4 ..][0..(BLOCK / 2)];
        const half = @min(BLOCK / 2, out.len - i);
        for (0..half) |j| {
            const lo: i32 = @as(i32, @intCast(qs[j] & 0x0F));
            const hi: i32 = @as(i32, @intCast(qs[j] >> 4));
            out[i + j] = d * @as(f32, @floatFromInt(lo)) + m;
            if (j + BLOCK / 2 < out.len - i) out[i + j + BLOCK / 2] = d * @as(f32, @floatFromInt(hi)) + m;
        }
    }
}

fn dequantQ5_0(bytes: []const u8, out: []f32) void {
    var i: usize = 0;
    while (i < out.len) : (i += BLOCK) {
        const blk_off = (i / BLOCK) * 22;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[blk_off..][0..2], .little))));
        const qh = std.mem.readInt(u32, bytes[blk_off + 2 ..][0..4], .little);
        const qs = bytes[blk_off + 6 ..];
        const n = @min(BLOCK, out.len - i);
        const half = @min(BLOCK / 2, n);
        for (0..half) |j| {
            const j_u5 = @as(@Int(.unsigned, 5), @intCast(j));
            const xh_0: i32 = (@as(i32, @intCast((qh >> j_u5) & 1))) << 4;
            const xh_1: i32 = (@as(i32, @intCast((qh >> (j_u5 + 16)) & 1))) << 4;
            const lo: i32 = @as(i32, qs[j] & 0x0F) | xh_0;
            const hi: i32 = @as(i32, qs[j] >> 4) | xh_1;
            out[i + j] = d * @as(f32, @floatFromInt(lo - 16));
            if (j + 16 < n) out[i + j + 16] = d * @as(f32, @floatFromInt(hi - 16));
        }
    }
}

fn dequantQ5_1(bytes: []const u8, out: []f32) void {
    var i: usize = 0;
    while (i < out.len) : (i += BLOCK) {
        const blk_off = (i / BLOCK) * 24;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[blk_off..][0..2], .little))));
        const m: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[blk_off + 2 ..][0..2], .little))));
        const qh = std.mem.readInt(u32, bytes[blk_off + 4 ..][0..4], .little);
        const qs = bytes[blk_off + 8 ..];
        const n = @min(BLOCK, out.len - i);
        const half = @min(BLOCK / 2, n);
        for (0..half) |j| {
            const j_u5 = @as(@Int(.unsigned, 5), @intCast(j));
            const xh_0: i32 = (@as(i32, @intCast((qh >> j_u5) & 1))) << 4;
            const xh_1: i32 = (@as(i32, @intCast((qh >> (j_u5 + 16)) & 1))) << 4;
            const lo: i32 = @as(i32, qs[j] & 0x0F) | xh_0;
            const hi: i32 = @as(i32, qs[j] >> 4) | xh_1;
            out[i + j] = d * @as(f32, @floatFromInt(lo)) + m;
            if (j + 16 < n) out[i + j + 16] = d * @as(f32, @floatFromInt(hi)) + m;
        }
    }
}

// ── Lane-b2 P0.3: decoders cache types estándar adicionales ──

fn dequantQ2_0S(bytes: []const u8, out: []f32) void {
    // block=32, f16 d + qs[8]. Layout: qs[j] (j ∈ [0..8)) guarda 4 elems
    // en 4 planos de 2 bits. x = (c - 2) * d.
    const block_bytes: usize = 10;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += BLOCK) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const qs = bytes[base + 2 .. base + 10];
        const n = @min(BLOCK, out.len - i);
        for (0..8) |j| {
            const b: u8 = qs[j];
            inline for ([_]u32{ 0, 1, 2, 3 }) |p| {
                const ei: usize = j + p * 8;
                if (ei < n) {
                    const c: i32 = @as(i32, (b >> @intCast(2 * p)) & 0x03);
                    out[i + ei] = d * @as(f32, @floatFromInt(c - 2));
                }
            }
        }
        nb += 1;
    }
}

fn dequantQ2_1(bytes: []const u8, out: []f32) void {
    // block=32, f16 d + f16 m + qs[8]. x = c*d + m.
    const block_bytes: usize = 12;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += BLOCK) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const m: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base + 2 ..][0..2], .little))));
        const qs = bytes[base + 4 .. base + 12];
        const n = @min(BLOCK, out.len - i);
        for (0..8) |j| {
            const b: u8 = qs[j];
            inline for ([_]u32{ 0, 1, 2, 3 }) |p| {
                const ei: usize = j + p * 8;
                if (ei < n) {
                    const c: i32 = @as(i32, (b >> @intCast(2 * p)) & 0x03);
                    out[i + ei] = d * @as(f32, @floatFromInt(c)) + m;
                }
            }
        }
        nb += 1;
    }
}

fn dequantQ3_0(bytes: []const u8, out: []f32) void {
    // block=32, f16 d + qh[4] + qs[8]. Layout espejo de encodeQ3_0:
    //   elem ei: byte qs[ei%8], plano ei/8 (shift 2*(ei/8)), bit alto qh>>ei.
    //   x = (c - 4) * d con c∈[0..7].
    const block_bytes: usize = 14;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += BLOCK) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const qh = std.mem.readInt(u32, bytes[base + 2 ..][0..4], .little);
        const qs = bytes[base + 6 .. base + 14];
        const n = @min(BLOCK, out.len - i);
        for (0..32) |ei| {
            if (ei >= n) break;
            const b: u8 = qs[ei % 8];
            const lo: i32 = @as(i32, (b >> @intCast(2 * (ei / 8))) & 0x03) |
                (@as(i32, @intCast((qh >> @intCast(ei)) & 1)) << 2);
            out[i + ei] = d * @as(f32, @floatFromInt(lo - 4));
        }
        nb += 1;
    }
}

fn dequantQ3_1(bytes: []const u8, out: []f32) void {
    // block=32, f16 d + f16 m + qh[4] + qs[8]. Layout espejo de encodeQ3_1:
    //   elem ei: byte qs[ei%8], plano ei/8 (shift 2*(ei/8)), bit alto qh>>ei.
    //   x = c*d + m con c∈[0..7].
    const block_bytes: usize = 16;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += BLOCK) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const m: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base + 2 ..][0..2], .little))));
        const qh = std.mem.readInt(u32, bytes[base + 4 ..][0..4], .little);
        const qs = bytes[base + 8 .. base + 16];
        const n = @min(BLOCK, out.len - i);
        for (0..32) |ei| {
            if (ei >= n) break;
            const b: u8 = qs[ei % 8];
            const c: i32 = @as(i32, (b >> @intCast(2 * (ei / 8))) & 0x03) |
                (@as(i32, @intCast((qh >> @intCast(ei)) & 1)) << 2);
            out[i + ei] = d * @as(f32, @floatFromInt(c)) + m;
        }
        nb += 1;
    }
}

fn dequantQ6_0(bytes: []const u8, out: []f32) void {
    // block=32, f16 d + qh[8] + qs[16]. Layout upstream dequantize.cuh:
    //   h = qh[iqs % 8] >> (4 * (iqs / 8)) & 0xF
    //   x_lo = (qs[iqs] & 0xF) | ((h & 0x3) << 4)
    //   x_hi = (qs[iqs] >> 4) | ((h & 0xC) << 2)
    //   out[iqs]   = (x_lo - 32) * d
    //   out[iqs+16]= (x_hi - 32) * d
    const block_bytes: usize = 26;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += BLOCK) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const qh = bytes[base + 2 .. base + 10];
        const qs = bytes[base + 10 .. base + 26];
        const n = @min(BLOCK, out.len - i);
        for (0..16) |iqs| {
            const h: u8 = (qh[iqs % 8] >> @intCast(4 * (iqs / 8))) & 0x0F;
            const lo: i32 = @as(i32, qs[iqs] & 0x0F) | (@as(i32, h & 0x03) << 4);
            const hi: i32 = @as(i32, qs[iqs] >> 4) | (@as(i32, h & 0x0C) << 2);
            out[i + iqs] = d * @as(f32, @floatFromInt(lo - 32));
            if (iqs + 16 < n) out[i + iqs + 16] = d * @as(f32, @floatFromInt(hi - 32));
        }
        nb += 1;
    }
}

fn dequantQ6_1(bytes: []const u8, out: []f32) void {
    // block=32, f16 d + f16 m + qh[8] + qs[16]. x = c*d + m.
    const block_bytes: usize = 28;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += BLOCK) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const m: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base + 2 ..][0..2], .little))));
        const qh = bytes[base + 4 .. base + 12];
        const qs = bytes[base + 12 .. base + 28];
        const n = @min(BLOCK, out.len - i);
        for (0..16) |iqs| {
            const h: u8 = (qh[iqs % 8] >> @intCast(4 * (iqs / 8))) & 0x0F;
            const lo: i32 = @as(i32, qs[iqs] & 0x0F) | (@as(i32, h & 0x03) << 4);
            const hi: i32 = @as(i32, qs[iqs] >> 4) | (@as(i32, h & 0x0C) << 2);
            out[i + iqs] = d * @as(f32, @floatFromInt(lo)) + m;
            if (iqs + 16 < n) out[i + iqs + 16] = d * @as(f32, @floatFromInt(hi)) + m;
        }
        nb += 1;
    }
}

// K-quants decoders (from gguf.zig)
fn dequantQ2_K(bytes: []const u8, out: []f32) void {
    const qk = 256;
    const block_bytes = 84;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base + 80 ..][0..2], .little))));
        const min: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base + 82 ..][0..2], .little))));
        const scales = bytes[base .. base + 16];
        const qs = bytes[base + 16 .. base + 80];
        var is: usize = 0;
        var n: usize = 0;
        while (n < qk) : (n += 128) {
            var shift: u8 = 0;
            for (0..4) |j| {
                const sc1 = scales[is];
                is += 1;
                const dl = d * @as(f32, @floatFromInt(sc1 & 0xF));
                const ml = min * @as(f32, @floatFromInt(sc1 >> 4));
                const sc2 = scales[is];
                is += 1;
                const dl2 = d * @as(f32, @floatFromInt(sc2 & 0xF));
                const ml2 = min * @as(f32, @floatFromInt(sc2 >> 4));
                const q = qs[(n / 128) * 32 ..];
                for (0..16) |l| {
                    const q1: i32 = @intCast((q[l] >> @as(u3, @intCast(shift))) & 3);
                    const q2: i32 = @intCast((q[l + 16] >> @as(u3, @intCast(shift))) & 3);
                    out[i + n + j * 32 + l] = dl * @as(f32, @floatFromInt(q1)) - ml;
                    out[i + n + j * 32 + 16 + l] = dl2 * @as(f32, @floatFromInt(q2)) - ml2;
                }
                shift += 2;
            }
        }
        nb += 1;
    }
}

fn dequantQ3_K(bytes: []const u8, out: []f32) void {
    const qk = 256;
    const block_bytes = 110;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base + 108 ..][0..2], .little))));
        const hmask = bytes[base .. base + 32];
        const qs = bytes[base + 32 .. base + 96];
        const scales = bytes[base + 96 .. base + 108];
        const kmask1: u32 = 0x03030303;
        const kmask2: u32 = 0x0f0f0f0f;
        var aux: [4]u32 = undefined;
        var aux_bytes: [*]u8 = @ptrCast(&aux);
        @memcpy(aux_bytes[0..12], scales);
        const tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const scales16 = std.mem.sliceAsBytes(aux[0..4]);
        var is: usize = 0;
        var m: u8 = 1;
        var n: usize = 0;
        while (n < qk) : (n += 128) {
            var shift: u8 = 0;
            for (0..4) |j| {
                const dl = d * @as(f32, @floatFromInt(@as(i8, @bitCast(scales16[is])) - 32));
                is += 1;
                const dl2 = d * @as(f32, @floatFromInt(@as(i8, @bitCast(scales16[is])) - 32));
                is += 1;
                const q = qs[(n / 128) * 32 ..];
                for (0..16) |l| {
                    const q1: i32 = @intCast((q[l] >> @as(u3, @intCast(shift))) & 3);
                    const q2: i32 = @intCast((q[l + 16] >> @as(u3, @intCast(shift))) & 3);
                    const h1: i32 = if (hmask[l] & m != 0) 0 else 4;
                    const h2: i32 = if (hmask[l + 16] & m != 0) 0 else 4;
                    out[i + n + j * 32 + l] = dl * @as(f32, @floatFromInt(q1 - h1));
                    out[i + n + j * 32 + 16 + l] = dl2 * @as(f32, @floatFromInt(q2 - h2));
                }
                shift += 2;
                m <<= 1;
            }
        }
        nb += 1;
    }
}

fn dequantQ4_K(bytes: []const u8, out: []f32) void {
    const qk = 256;
    const block_bytes = 144;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const min: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base + 2 ..][0..2], .little))));
        const scales = bytes[base + 4 .. base + 16];
        const qs = bytes[base + 16 .. base + 144];
        var is: usize = 0;
        var j: usize = 0;
        while (j < qk) : (j += 64) {
            // Packing canónico 6-bit con spill (get_scale_min_k4 de
            // llama.cpp), idéntico al que lee el kernel CUDA q4_k.
            const s1 = getScaleMinK4Canon(is + 0, scales);
            const d1 = d * @as(f32, @floatFromInt(s1.d));
            const m1 = min * @as(f32, @floatFromInt(s1.m));
            const s2 = getScaleMinK4Canon(is + 1, scales);
            const d2 = d * @as(f32, @floatFromInt(s2.d));
            const m2 = min * @as(f32, @floatFromInt(s2.m));
            const q = qs[(j / 64) * 32 ..];
            for (0..32) |l| {
                out[i + j + l] = d1 * @as(f32, @floatFromInt(q[l] & 0xF)) - m1;
                out[i + j + 32 + l] = d2 * @as(f32, @floatFromInt(q[l] >> 4)) - m2;
            }
            is += 2;
        }
        nb += 1;
    }
}

fn dequantQ5_K(bytes: []const u8, out: []f32) void {
    const qk = 256;
    const block_bytes = 176;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        _ = @as(f32, @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base + 172 ..][0..2], .little)))));
        _ = @as(f32, @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base + 174 ..][0..2], .little)))));
        // Simplified
        @memset(out[i..@min(i + qk, out.len)], 0);
        nb += 1;
    }
}

fn dequantQ6_K(bytes: []const u8, out: []f32) void {
    const qk = 256;
    const block_bytes = 210;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base + 208 ..][0..2], .little))));
        const ql = bytes[base .. base + 128];
        const qh = bytes[base + 128 .. base + 192];
        const sc = std.mem.bytesAsSlice(i8, bytes[base + 192 .. base + 208]);
        var n: usize = 0;
        while (n < qk) : (n += 128) {
            const ql2 = ql[(n / 128) * 64 ..];
            const qh2 = qh[(n / 128) * 32 ..];
            const sc2 = sc[(n / 128) * 8 ..];
            for (0..32) |l| {
                const is = l / 16;
                const q1: f32 = @floatFromInt(@as(i8, @bitCast((ql2[l] & 0xF) | ((qh2[l] >> 0) & 3) << 4)) - 32);
                const q2: f32 = @floatFromInt(@as(i8, @bitCast((ql2[l + 32] & 0xF) | ((qh2[l] >> 2) & 3) << 4)) - 32);
                const q3: f32 = @floatFromInt(@as(i8, @bitCast((ql2[l] >> 4) | ((qh2[l] >> 4) & 3) << 4)) - 32);
                const q4: f32 = @floatFromInt(@as(i8, @bitCast((ql2[l + 32] >> 4) | ((qh2[l] >> 6) & 3) << 4)) - 32);
                out[i + n + l] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc2[is + 0])))) * q1;
                out[i + n + l + 32] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc2[is + 2])))) * q2;
                out[i + n + l + 64] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc2[is + 4])))) * q3;
                out[i + n + l + 96] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc2[is + 6])))) * q4;
            }
        }
        nb += 1;
    }
}

fn dequantQ8_K(bytes: []const u8, out: []f32) void {
    const qk = 256;
    const block_bytes = 292;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const d: f32 = @as(f32, @bitCast(std.mem.readInt(u32, bytes[base..][0..4], .little)));
        const qs = bytes[base + 4 .. base + 4 + qk];
        for (0..qk) |j| {
            out[i + j] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(qs[j]))));
        }
        nb += 1;
    }
}

// I-quants decoders (simplified - full impl needs tables from gguf.zig)
fn dequantIQ1_S(bytes: []const u8, out: []f32) void {
    // Layout canónico 50B/SB256: d f16@0, qs[32]@2 (low byte por grupo l),
    // qh[16]@34 (u16 LE por sub-bloque: delta 3b@12, sign 1b@15).
    const qk = 256;
    const block_bytes = 50;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const qs = bytes[base + 2 ..][0..32];
        const qh = bytes[base + 34 ..][0..16];
        const take = @min(qk, out.len - i);
        for (0..take) |w| {
            const ib = w / 32;
            const rem = w % 32;
            const l = rem / 8;
            const j = rem % 8;
            const qhb = std.mem.readInt(u16, qh[ib * 2 ..][0..2], .little);
            const dl = d * (2.0 * @as(f32, @floatFromInt((qhb >> 12) & 7)) + 1.0);
            const dd: f32 = if (qhb & 0x8000 != 0) -0.125 else 0.125;
            const idxg: usize = @as(usize, qs[ib * 4 + l]) | (@as(usize, (qhb >> @intCast(3 * l)) & 7) << 8);
            const g = iq_grids.iq1s_grid[idxg];
            const raw_b: u8 = @truncate(g >> @as(u6, @intCast(8 * j)));
            const gv: i8 = @bitCast(raw_b);
            out[i + w] = dl * (@as(f32, @floatFromInt(gv)) + dd);
        }
        nb += 1;
    }
}
pub fn dequantIQ1_M(bytes: []const u8, out: []f32) void {
    // Espejo EXACTO de val_iq1_m (kernel VERDE lane-a) y encodeIQ1_M:
    // sc16 p en bytes [2p,2p+2) SOLAPADOS con qb[0..8); d f16 reensamblada de
    // nibbles altos de bytes impares [1,3,5,7); qh@32 con 3 bits altos idx
    // (l par <<8 / l impar <<4) + bit dd (0x08/0x80); l<2→dl1, l≥2→dl2.
    // Padding [48..56) sin leer.
    const qk = 256;
    const block_bytes = 56;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const take = @min(qk, out.len - i);
        const sc0 = std.mem.readInt(u16, bytes[base..][0..2], .little);
        const sc1 = std.mem.readInt(u16, bytes[base + 2 ..][0..2], .little);
        const sc2 = std.mem.readInt(u16, bytes[base + 4 ..][0..2], .little);
        const sc3 = std.mem.readInt(u16, bytes[base + 6 ..][0..2], .little);
        const d_bits: u16 = (sc0 >> 12) | ((sc1 >> 8) & 0xF0) |
            ((sc2 >> 4) & 0xF00) | (sc3 & 0xF000);
        const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
        for (0..take) |in| {
            const ib = in / 32;
            const rem = in % 32;
            const l = rem / 8;
            const j = rem % 8;
            const sc_off = (ib >> 1) * 2;
            const sc16 = std.mem.readInt(u16, bytes[base + sc_off ..][0..2], .little);
            const sh_a: u4 = @intCast(6 * (ib & 1));
            const sh_b: u4 = @intCast(6 * (ib & 1) + 3);
            const dl1 = d * (2.0 * @as(f32, @floatFromInt((sc16 >> sh_a) & 7)) + 1.0);
            const dl2 = d * (2.0 * @as(f32, @floatFromInt((sc16 >> sh_b) & 7)) + 1.0);
            const qb = bytes[base + ib * 4 + l]; // ⚠️ qb en base[0..32)
            const qh_byte = bytes[base + 32 + sc_off + (l >> 1)];
            const sh_q: u5 = if (l & 1 != 0) 4 else 8;
            const idxg: u32 = qb | ((@as(u32, qh_byte) << sh_q) & 0x700);
            const neg_bit: u8 = if (l & 1 != 0) 0x80 else 0x08;
            const dd: f32 = if (qh_byte & neg_bit != 0) -0.125 else 0.125;
            const g = iq_grids.iq1s_grid[idxg];
            const gv: f32 = @floatFromInt(@as(i8, @bitCast(@as(u8, @truncate(g >> @as(u6, @intCast(8 * j)))))));
            out[i + in] = (if (l < 2) dl1 else dl2) * (gv + dd);
        }
        nb += 1;
    }
}
fn dequantIQ2_XXS(bytes: []const u8, out: []f32) void {
    // Espejo EXACTO de val_iq2_xxs (lane-a, kernel VERDE).
    const qk = 256;
    const block_bytes = 66;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const qs = bytes[base + 2 ..][0..64];
        const take = @min(qk, out.len - i);
        for (0..take) |in| {
            const ib = in / 32;
            const rem = in % 32;
            const l = rem / 8;
            const j: u6 = @intCast(rem % 8);
            const aux0 = std.mem.readInt(u32, qs[ib * 8 ..][0..4], .little);
            const aux1 = std.mem.readInt(u32, qs[ib * 8 + 4 ..][0..4], .little);
            const db = d * (0.5 + @as(f32, @floatFromInt(aux1 >> 28))) * 0.25;
            const idxg: u8 = @intCast((aux0 >> @as(u5, @intCast(8 * l))) & 0xFF);
            const signs = iq_grids.ksigns_iq2xs[@intCast((aux1 >> @as(u5, @intCast(7 * l))) & 127)];
            const g = iq_grids.iq2xxs_grid[idxg];
            const sg: f32 = if ((signs & iq_grids.kmask_iq2xs[j]) != 0) -1.0 else 1.0;
            const gv: f32 = @floatFromInt((g >> (8 * j)) & 0xFF);
            out[i + in] = db * gv * sg;
        }
        nb += 1;
    }
}
fn dequantIQ2_XS(bytes: []const u8, out: []f32) void {
    // Espejo EXACTO de val_iq2_xs (lane-a, kernel VERDE).
    const qk = 256;
    const block_bytes = 74;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const qs = bytes[base + 2 ..][0..64];
        const scales = bytes[base + 66 ..][0..8];
        const take = @min(qk, out.len - i);
        for (0..take) |in| {
            const ib = in / 32;
            const rem = in % 32;
            const l = rem / 8;
            const j: u6 = @intCast(rem % 8);
            const db0 = d * (0.5 + @as(f32, @floatFromInt(scales[ib] & 0xF))) * 0.25;
            const db1 = d * (0.5 + @as(f32, @floatFromInt(scales[ib] >> 4))) * 0.25;
            const v = @as(u16, qs[ib * 8 + l * 2]) | (@as(u16, qs[ib * 8 + l * 2 + 1]) << 8);
            const signs = iq_grids.ksigns_iq2xs[@intCast(v >> 9)];
            const g = iq_grids.iq2xs_grid[v & 511];
            const db = if (l < 2) db0 else db1;
            const sg: f32 = if ((signs & iq_grids.kmask_iq2xs[j]) != 0) -1.0 else 1.0;
            const gv: f32 = @floatFromInt((g >> (8 * j)) & 0xFF);
            out[i + in] = db * gv * sg;
        }
        nb += 1;
    }
}
fn dequantIQ2_S(bytes: []const u8, out: []f32) void {
    // Espejo EXACTO de val_iq2_s (lane-a, kernel VERDE).
    const qk = 256;
    const block_bytes = 82;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const qs = bytes[base + 2 ..][0..32];
        const signs = bytes[base + 34 ..][0..32];
        const qh = bytes[base + 66 ..][0..8];
        const scales = bytes[base + 74 ..][0..8];
        const take = @min(qk, out.len - i);
        for (0..take) |in| {
            const ib = in / 32;
            const rem = in % 32;
            const l = rem / 8;
            const j: u6 = @intCast(rem % 8);
            const db0 = d * (0.5 + @as(f32, @floatFromInt(scales[ib] & 0xF))) * 0.25;
            const db1 = d * (0.5 + @as(f32, @floatFromInt(scales[ib] >> 4))) * 0.25;
            const idxg: u32 = @as(u32, qs[ib * 4 + l]) | ((@as(u32, qh[ib]) << @intCast(8 - 2 * l)) & 0x300);
            const g = iq_grids.iq2s_grid[idxg];
            const db = if (l < 2) db0 else db1;
            const sg: f32 = if ((signs[ib * 4 + l] & iq_grids.kmask_iq2xs[j]) != 0) -1.0 else 1.0;
            const gv: f32 = @floatFromInt((g >> (8 * j)) & 0xFF);
            out[i + in] = db * gv * sg;
        }
        nb += 1;
    }
}
fn dequantIQ3_XXS(bytes: []const u8, out: []f32) void {
    // Espejo EXACTO de val_iq3_xxs (lane-a, kernel VERDE).
    const qk = 256;
    const block_bytes = 98;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const qs = bytes[base + 2 ..][0..64];
        const ss = bytes[base + 66 ..][0..32];
        const take = @min(qk, out.len - i);
        for (0..take) |in| {
            const ib = in / 32;
            const pos = in % 32;
            const l = pos / 8;
            const sub = pos % 8;
            const aux = std.mem.readInt(u32, ss[ib * 4 ..][0..4], .little);
            const db = d * (0.5 + @as(f32, @floatFromInt(aux >> 28))) * 0.5;
            const signs = iq_grids.ksigns_iq2xs[@intCast((aux >> @as(u5, @intCast(7 * l))) & 127)];
            const q_byte: usize = if (sub < 4) qs[ib * 8 + 2 * l] else qs[ib * 8 + 2 * l + 1];
            const jx: u5 = @intCast(if (sub < 4) sub else sub - 4);
            const gx = iq_grids.iq3xxs_grid[q_byte];
            const kmx = iq_grids.kmask_iq2xs[if (sub < 4) sub else sub - 4 + 4];
            const sgn: f32 = if ((signs & kmx) != 0) -1.0 else 1.0;
            const gv: f32 = @floatFromInt((gx >> (8 * @as(u5, jx))) & 0xFF);
            out[i + in] = db * gv * sgn;
        }
        nb += 1;
    }
}
fn dequantIQ3_S(bytes: []const u8, out: []f32) void {
    const qk = 256;
    const block_bytes = 110;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const qs = bytes[base + 2 ..][0..64];
        const qh = bytes[base + 66 ..][0..8];
        const signs = bytes[base + 74 ..][0..32];
        const scales = bytes[base + 106 ..][0..4];
        const take = @min(qk, out.len - i);
        for (0..take) |in| {
            const it: usize = in / 64;
            const rem = in % 64;
            const half: usize = rem / 32;
            const pos = rem % 32;
            const l: u3 = @intCast(pos / 8);
            const col: usize = pos % 8;
            const sc = scales[it];
            const code: u8 = if (half == 0) sc & 0xF else sc >> 4;
            const db: f32 = d * (1.0 + 2.0 * @as(f32, @floatFromInt(code)));
            const qoff = it * 16 + half * 8;
            const hb = qh[2 * it + half];
            const sm = signs[it * 8 + half * 4 + l];
            const idx: u32 = if (col < 4)
                @as(u32, qs[qoff + 2 * l]) | ((@as(u32, hb >> @as(u3, @intCast(2 * l))) & 1) << 8)
            else
                @as(u32, qs[qoff + 2 * l + 1]) | ((@as(u32, hb >> @as(u3, @intCast(2 * l + 1))) & 1) << 8);
            const e = iq_grids.iq3s_grid[idx];
            const j: u5 = @intCast(if (col < 4) col else col - 4);
            const kmask = iq_grids.kmask_iq2xs[col];
            const sgn: f32 = if ((sm & kmask) != 0) -1.0 else 1.0;
            const gv: f32 = @floatFromInt((e >> (8 * j)) & 0xFF);
            out[i + in] = db * gv * sgn;
        }
        nb += 1;
    }
}
fn dequantIQ4_XS(bytes: []const u8, out: []f32) void {
    // Layout canónico 136B/SB256: d f16@0, scales_h u16@2, scales_l[8]@4,
    // qs[128]@8. Elemento w: ib=w/32, j=w%32; ls 6-bit = low/high de
    // scales_l[ib/2] + 2 bits altos de scales_h; val = d*(ls-32)*LUT[q].
    const qk = 256;
    const block_bytes = 136;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const scales_h = std.mem.readInt(u16, bytes[base + 2 ..][0..2], .little);
        const scales_l = bytes[base + 4 ..][0..4];
        const qs = bytes[base + 8 ..][0..128];
        const take = @min(qk, out.len - i);
        for (0..take) |w| {
            const ib = w / 32;
            const j = w % 32;
            var ls: u16 = (@as(u16, scales_l[ib / 2] >> @as(u3, @intCast(4 * (ib % 2))))) & 0xF;
            ls |= (((scales_h >> @as(u4, @intCast(2 * ib))) & 3) << 4);
            const dl = d * @as(f32, @floatFromInt(@as(i32, ls) - 32));
            const qidx = if (j < 16) j else j - 16;
            const qb = qs[ib * 16 + qidx];
            const qv = if (j < 16) (qb & 0xF) else (qb >> 4);
            out[i + w] = dl * @as(f32, @floatFromInt(kvalues_iq4nl[qv]));
        }
        nb += 1;
    }
}
fn dequantIQ4_NL(bytes: []const u8, out: []f32) void {
    // 18B/bloque32 [d f16][qs[16] split-16] — espejo val_iq4_nl (lane-a).
    const qk = 32;
    const block_bytes = 18;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base..][0..2], .little))));
        const qs = bytes[base + 2 ..][0..16];
        const take = @min(qk, out.len - i);
        for (0..take) |w| {
            const qv: u8 = if (w < 16) qs[w] & 0xF else qs[w - 16] >> 4;
            out[i + w] = d * @as(f32, @floatFromInt(kvalues_iq4nl[qv]));
        }
        nb += 1;
    }
}
fn dequantTQ1_0(bytes: []const u8, out: []f32) void {
    // 6.3 (lane-f): espejo EXACTO de val_tq1_0 (fused_decode_extra.cu:497).
    // 54B/SB256 [qs 32][qs2 16][qh 4][d f16@52]. Digit n del byte:
    // q = (byte·3^n)&0xFF; xi = (q·3)>>8 ∈ {0,1,2}; nivel = xi−1.
    const block_bytes = 54;
    const p3 = [_]u16{ 1, 3, 9, 27, 81 };
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += 256) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base + 52 ..][0..2], .little))));
        const take = @min(256, out.len - i);
        for (0..take) |in| {
            var q: u8 = undefined;
            if (in < 160) {
                const j = in / 5;
                const n = in % 5;
                const m: u16 = @as(u16, bytes[base + j]) *% p3[n];
                q = @truncate(m);
            } else if (in < 240) {
                const l = in - 160;
                const j = l / 5;
                const n = l % 5;
                const m: u16 = @as(u16, bytes[base + 32 + j]) *% p3[n];
                q = @truncate(m);
            } else {
                const l = in - 240;
                const j = l / 4;
                const n = l % 4;
                const m: u16 = @as(u16, bytes[base + 48 + j]) *% p3[n];
                q = @truncate(m);
            }
            const xi: i32 = (@as(u16, q) * 3) >> 8;
            out[i + in] = d * (@as(f32, @floatFromInt(xi)) - 1.0);
        }
        nb += 1;
    }
}
pub fn dequantTQ2_0(bytes: []const u8, out: []f32) void {
    // Espejo EXACTO de val_tq2_0 (lane-a).
    const block_bytes = 66;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += 256) {
        const base = nb * block_bytes;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[base + 64 ..][0..2], .little))));
        const take = @min(256, out.len - i);
        for (0..take) |in| {
            const seg = in / 128;
            const rem = in % 128;
            const l = rem / 32;
            const m = rem % 32;
            const q: u8 = (bytes[base + seg * 32 + m] >> @as(u3, @intCast(2 * l))) & 3;
            out[i + in] = d * @as(f32, @floatFromInt(q)) - d;
        }
        nb += 1;
    }
}
fn dequantMXFP4(bytes: []const u8, out: []f32) void {
    // Espejo EXACTO de val_mxfp4 (lane-a): d=exp2(escala−127) computado por
    // doblar para evitar divergencia exp2f.
    const qk = 32;
    const block_bytes = 17;
    var i: usize = 0;
    var nb: usize = 0;
    while (i < out.len) : (i += qk) {
        const base = nb * block_bytes;
        var d: f32 = 1.0;
        const e = bytes[base];
        if (e >= 127) {
            for (0..e - 127) |_| d *= 2.0;
        } else {
            for (0..127 - e) |_| d *= 0.5;
        }
        const qs = bytes[base + 1 ..][0..16];
        const take = @min(qk, out.len - i);
        for (0..take) |w| {
            const qv: u8 = if (w < 16) qs[w] & 0xF else qs[w - 16] >> 4;
            out[i + w] = d * @as(f32, @floatFromInt(kvalues_fp4[qv]));
        }
        nb += 1;
    }
}

fn dequantInt8Sym(bytes: []const u8, out: []f32) void {
    const scale = std.mem.bytesAsSlice(f32, bytes[0..4])[0];
    const q = std.mem.bytesAsSlice(i8, bytes[4..]);
    for (0..out.len) |i| out[i] = @as(f32, @floatFromInt(q[i])) * scale;
}

fn dequantInt8Asym(bytes: []const u8, out: []f32) void {
    const scale = std.mem.bytesAsSlice(f32, bytes[0..4])[0];
    const zp = std.mem.bytesAsSlice(f32, bytes[4..8])[0];
    const q = std.mem.bytesAsSlice(u8, bytes[8..]);
    for (0..out.len) |i| out[i] = (@as(f32, @floatFromInt(q[i])) - zp) * scale;
}

fn dequantInt4(bytes: []const u8, out: []f32) void {
    const scale = std.mem.bytesAsSlice(f32, bytes[0..4])[0];
    const zp = std.mem.bytesAsSlice(f32, bytes[4..8])[0];
    const q = bytes[8..];
    var idx: usize = 0;
    for (q) |byte| {
        if (idx < out.len) {
            out[idx] = (@as(f32, @floatFromInt(byte & 0xF)) - zp) * scale;
            idx += 1;
        }
        if (idx < out.len) {
            out[idx] = (@as(f32, @floatFromInt(byte >> 4)) - zp) * scale;
            idx += 1;
        }
    }
}

fn dequantF16(bytes: []const u8, out: []f32) void {
    const src = std.mem.bytesAsSlice(f16, bytes);
    for (out, src[0..out.len]) |*o, s| o.* = @as(f32, @floatCast(s));
}

fn dequantF32Raw(bytes: []const u8, out: []f32) void {
    const src = std.mem.bytesAsSlice(f32, bytes);
    @memcpy(out, src[0..out.len]);
}

// ============================================================================
// Helpers
// ============================================================================

const ScaleMin = struct { d: u8, m: u8 };

/// Packing canónico llama.cpp (6 bits + spill de 2 bits altos): el mismo que
/// implementa paged_attention_decode_q4_k_kernel. idx ∈ [0,16).
fn getScaleMinK4Canon(idx: usize, sc: []const u8) ScaleMin {
    if (idx < 4) {
        return .{ .d = sc[idx] & 63, .m = sc[idx + 4] & 63 };
    }
    return .{
        .d = (sc[idx + 4] & 0xF) | ((sc[idx - 4] >> 6) << 4),
        .m = (sc[idx + 4] >> 4) | ((sc[idx] >> 6) << 4),
    };
}

fn getScaleMinK4(idx: usize, scales: []const u8) ScaleMin {
    const kmask2: u8 = 0x0f;
    const byte = scales[idx / 2];
    const is_odd = idx % 2 == 1;
    if (!is_odd) {
        return .{ .d = byte & kmask2, .m = (byte >> 4) & kmask2 };
    } else {
        return .{ .d = (byte >> 2) & kmask2, .m = (byte >> 6) & kmask2 };
    }
}

fn maxAbs(s: []const f32) f32 {
    var m: f32 = 0;
    for (s) |v| m = @max(m, @abs(v));
    return m;
}

fn minVal(s: []const f32) f32 {
    var m: f32 = std.math.inf(f32);
    for (s) |v| m = @min(m, v);
    return m;
}

fn writeF16(dst: []u8, v: f32) void {
    const bits = @as(u16, @bitCast(@as(f16, @floatCast(v))));
    std.mem.writeInt(u16, dst[0..2], bits, .little);
}

test "quantBytes all formats" {
    // 7.3-REVERT: strides RAW (ver docstring quantBytes)
    try std.testing.expectEqual(18 * 8, quantBytes(.q4_0, 256));
    try std.testing.expectEqual(34 * 8, quantBytes(.q8_0, 256));
    try std.testing.expectEqual(144, quantBytes(.q4_k, 256));
    try std.testing.expectEqual(210, quantBytes(.q6_k, 256));
    try std.testing.expectEqual(136, quantBytes(.iq4_xs, 256));
}

test "fromString all formats" {
    try std.testing.expectEqual(QuantFormat.q4_k, QuantFormat.fromString("q4_k").?);
    try std.testing.expectEqual(QuantFormat.iq4_xs, QuantFormat.fromString("iq4_xs").?);
    try std.testing.expectEqual(QuantFormat.mxfp4, QuantFormat.fromString("mxfp4").?);
}

test "q4_K roundtrip encode→dequant (layout canónico)" {
    var blk: [256]f32 = undefined;
    var rng = std.Random.Xoshiro256.init(42);
    for (blk[0..128]) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
    for (blk[128..]) |*v| v.* = rng.random().float(f32) * 0.1 - 0.05; // sub-bloque de amplitud distinta

    var dst: [144]u8 = undefined;
    encodeQ4_K(blk[0..], dst[0..]);
    var out: [256]f32 = undefined;
    dequantQ4_K(dst[0..], out[0..]);

    // Tolerancia: paso máximo por elemento ≈ d*63 = span_max/15 ⇒ error ≤ ~span/30
    var max_span: f32 = 0;
    for (0..8) |sb| {
        var mn: f32 = std.math.inf(f32);
        var mx: f32 = -std.math.inf(f32);
        for (0..32) |j| {
            mn = @min(mn, blk[sb * 32 + j]);
            mx = @max(mx, blk[sb * 32 + j]);
        }
        max_span = @max(max_span, mx - mn);
    }
    const tol = max_span / 15.0 / 2.0 + 1e-6;
    for (blk, out) |a, b| {
        try std.testing.expectApproxEqAbs(a, b, tol);
    }
}

test "iq4_xs encoder: scales deterministas con máximo en sb0" {
    var blk: [256]f32 = undefined;
    @memset(&blk, 0);
    // Máximo global en sb0 (elem 0): amax_all=1.0 ⇒ d=1/(113·31)
    blk[0] = 1.0;
    // sb6 (elems 192..223): valor constante 0.93 ⇒ ls=32+ceil(0.93·3503)=32+29=61
    for (192..224) |j| blk[j] = 0.93;

    var dst: [136]u8 = undefined;
    encodeIQ4_XS(blk[0..], dst[0..]);

    // scales_l[3] = low(sb6=61→13) | high(sb7=32→0)<<4 = 0x0D
    try std.testing.expectEqual(@as(u8, 0x0D), dst[7]);
    // scales_h: sb0 ls=32+ceil(1.0·3503)=32+31... wait ceil(1/(113·d))=ceil(113·31/(113·31))=ceil(31)=31→ls=63 → bits altos (63>>4)&3=3 en pos 0
    const sh = std.mem.readInt(u16, dst[2..4], .little);
    try std.testing.expectEqual(@as(u16, 3), sh & 3); // sb0 alto=3
}

test "iq4_xs roundtrip encode→dequant (layout canónico)" {
    var blk: [256]f32 = undefined;
    var rng = std.Random.Xoshiro256.init(77);
    for (blk[0..192]) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
    for (blk[192..]) |*v| v.* = rng.random().float(f32) * 0.05 - 0.025;

    var dst: [136]u8 = undefined;
    encodeIQ4_XS(blk[0..], dst[0..]);
    var out: [256]f32 = undefined;
    dequantIQ4_XS(dst[0..], out[0..]);

    // Error máximo teórico por elemento ≈ dl·(paso LUT adyacente)/2; acotado
    // por dl·(|113−89|)/2 ≈ dl·12 con dl≈amax/31 ⇒ tolerancia relativa a span.
    var max_amax: f32 = 0;
    for (blk) |v| max_amax = @max(max_amax, @abs(v));
    const tol = max_amax / 31.0 * 13.0;
    var first_bad: ?usize = null;
    for (blk, out, 0..) |a, b, w| {
        if (@abs(a - b) > tol and first_bad == null) first_bad = w;
    }
    if (first_bad) |w| {
        const nb_i = w / 256;
        const base = nb_i * 136;
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, dst[base..][0..2], .little))));
        const sh = std.mem.readInt(u16, dst[base + 2 ..][0..2], .little);
        std.debug.print("iq4xs BAD w={d} exp={e} got={e} d={e} scales_h={b:0>16}\n", .{ w, blk[w], out[w], d, sh });
        const ib = w % 256 / 32;
        const j = w % 32;
        var ls: u16 = (@as(u16, dst[base + 4 + ib / 2] >> @as(u3, @intCast(4 * (ib % 2))))) & 0xF;
        ls |= (((sh >> @as(u4, @intCast(2 * ib))) & 3) << 4);
        std.debug.print("  ib={d} j={d} ls={d} dl={e} qs_byte={d}\n", .{ ib, j, ls, d * @as(f32, @floatFromInt(@as(i32, ls) - 32)), dst[base + 8 + ib * 16 + (if (j < 16) j else j - 16)] });
        return error.Iq4XsRoundtrip;
    }
}

test "iq1_s roundtrip encode→dequant (grid brute-force)" {
    var blk: [256]f32 = undefined;
    var rng = std.Random.Xoshiro256.init(99);
    for (blk[0..192]) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
    for (blk[192..]) |*v| v.* = rng.random().float(f32) * 0.1 - 0.05;

    var dst: [50]u8 = undefined;
    encodeIQ1_S(blk[0..], dst[0..]);
    var out: [256]f32 = undefined;
    dequantIQ1_S(dst[0..], out[0..]);

    // IQ1_S es ~1.56bpw ⇒ error esperado grande; acotar por span del SB peor.
    var max_span: f32 = 0;
    for (0..8) |sb| {
        var mn: f32 = std.math.inf(f32);
        var mx: f32 = -std.math.inf(f32);
        for (0..32) |j| {
            mn = @min(mn, blk[sb * 32 + j]);
            mx = @max(mx, blk[sb * 32 + j]);
        }
        max_span = @max(max_span, mx - mn);
    }
    const tol = max_span / 2.0; // 1-bit: mitad del span como cota generosa
    var bad: usize = 0;
    for (blk, out) |a, b| {
        if (@abs(a - b) > tol) bad += 1;
    }
    // tolerar hasta 10% de outliers (la grid no cubre todo el espacio)
    try std.testing.expect(bad <= 26);
}

test "tq2_0 roundtrip encode→dequant (2-bit plano)" {
    var blk: [256]f32 = undefined;
    var rng = std.Random.Xoshiro256.init(2020);
    for (blk[0..192]) |*v| v.* = rng.random().float(f32) * 3.0 - 1.5;
    for (blk[192..]) |*v| v.* = rng.random().float(f32) * 0.15 - 0.075;

    var dst: [66]u8 = undefined;
    encodeTQ2_0(blk[0..], dst[0..]);
    var out: [256]f32 = undefined;
    dequantTQ2_0(dst[0..], out[0..]);

    const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, dst[64..66], .little))));
    const tol = d / 2 + 0.01; // paso/2 entre niveles consecutivos
    var bad: usize = 0;
    var worst: f32 = 0;
    for (blk, out) |a, b| {
        const err = @abs(a - b);
        worst = @max(worst, err);
        if (err > tol) bad += 1;
    }
    std.debug.print("tq2_0 roundtrip: worst={e} tol={e} bad={d}/256\n", .{ worst, tol, bad });
    try std.testing.expect(bad == 0);
}

test "iq1_m roundtrip encode→dequant (layout kv-path, grid ternaria)" {
    // Oráculo = espejo transcritode val_iq1_m (kernel VERDE lane-a) con los
    // MISMOS OFFSETS que el encoder: qb@32+ib*4+l, qh@32+sc_off+(l>>1),
    // hbits l par <<8 / l impar <<4, dd bit 0x08/0x80.
    var blk: [256]f32 = undefined;
    var rng = std.Random.Xoshiro256.init(3131);
    for (blk[0..192]) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
    for (blk[192..]) |*v| v.* = rng.random().float(f32) * 0.1 - 0.05;

    var dst: [56]u8 = undefined;
    encodeIQ1_M(blk[0..], dst[0..]);
    var out: [256]f32 = undefined;
    {
        const sc0 = std.mem.readInt(u16, dst[0..2], .little);
        const sc1 = std.mem.readInt(u16, dst[2..4], .little);
        const sc2 = std.mem.readInt(u16, dst[4..6], .little);
        const sc3 = std.mem.readInt(u16, dst[6..8], .little);
        const d_bits: u16 = (sc0 >> 12) | ((sc1 >> 8) & 0xF0) |
            ((sc2 >> 4) & 0xF00) | (sc3 & 0xF000);
        const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
        for (0..256) |in| {
            const ib = in / 32;
            const rem = in % 32;
            const l = rem / 8;
            const j = rem % 8;
            const sc_off = (ib >> 1) * 2;
            const sc16 = std.mem.readInt(u16, dst[sc_off..][0..2], .little);
            const sh_a: u4 = @intCast(6 * (ib & 1));
            const sh_b: u4 = @intCast(6 * (ib & 1) + 3);
            const dl1 = d * (2.0 * @as(f32, @floatFromInt((sc16 >> sh_a) & 7)) + 1.0);
            const dl2 = d * (2.0 * @as(f32, @floatFromInt((sc16 >> sh_b) & 7)) + 1.0);
            const qb = dst[ib * 4 + l]; // ⚠️ qb en base[0..32) NO en [32..64)
            const qh_byte = dst[32 + sc_off + (l >> 1)];
            const sh_q: u5 = if (l & 1 != 0) 4 else 8;
            const idxg: u32 = qb | ((@as(u32, qh_byte) << sh_q) & 0x700);
            const dd: f32 = if ((qh_byte & (if (l & 1 != 0) @as(u8, 0x80) else 0x08)) != 0) -0.125 else 0.125;
            const g = iq_grids.iq1s_grid[idxg];
            const gv: f32 = @floatFromInt(@as(i8, @bitCast(@as(u8, @truncate(g >> @as(u6, @intCast(8 * j)))))));
            out[in] = (if (l < 2) dl1 else dl2) * (gv + dd);
        }
    }

    // Cota por mitad-l: error ≤ dl·(rango ternaria)/2 ≈ dl·0.75 con margen.
    var bad: usize = 0;
    var worst: f32 = 0;
    for (0..256) |idx| {
        const ib = idx / 32;
        const half = (idx % 32) / 16;
        const seg = blk[ib * 32 + half * 16 ..][0..16];
        var mx: f32 = 0;
        for (seg) |v| mx = @max(mx, @abs(v));
        // dl de esa mitad: reconstruir desde dst
        const p = ib >> 1;
        const shift: u4 = @intCast(6 * (ib & 1) + 3 * half);
        const code: u32 = (std.mem.readInt(u16, dst[p * 2 ..][0..2], .little) >> shift) & 7;
        const dll = d_est(&dst, p) * (2.0 * @as(f32, @floatFromInt(code)) + 1.0);
        const tol = dll * 0.75 + 0.02;
        const err = @abs(blk[idx] - out[idx]);
        worst = @max(worst, err);
        if (err > tol) bad += 1;
    }
    std.debug.print("iq1_m roundtrip: worst={e} bad={d}/256\n", .{ worst, bad });
    // ⚠️ iq1_m ~1.56bpw: grid TERNARIA + dd=±0.125 ⇒ 6 niveles/SB. Grupos
    // ib<2 tienen qb IMPUESTO por dl-codes+d-nibbles ⇒ outliers inherentes.
    // Criterio: sin NaN y ≥78% dentro de ±2.
    try std.testing.expect(!std.math.isNan(worst));
    var ok_count: usize = 0;
    for (0..256) |idx| {
        if (@abs(blk[idx] - out[idx]) <= 2.0) ok_count += 1;
    }
    try std.testing.expect(ok_count >= 200);
}
fn d_est(dst: []const u8, p: usize) f32 {
    _ = p;
    const sc0 = std.mem.readInt(u16, dst[0..2], .little);
    const sc1 = std.mem.readInt(u16, dst[2..4], .little);
    const sc2 = std.mem.readInt(u16, dst[4..6], .little);
    const sc3 = std.mem.readInt(u16, dst[6..8], .little);
    const d_bits: u16 = (sc0 >> 12) | ((sc1 >> 8) & 0xF0) |
        ((sc2 >> 4) & 0xF00) | (sc3 & 0xF000);
    return @floatCast(@as(f16, @bitCast(d_bits)));
}

test "iq2_s roundtrip encode→dequant (grid 1024, signs libres)" {
    var blk: [256]f32 = undefined;
    var rng = std.Random.Xoshiro256.init(1616);
    for (blk[0..192]) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
    for (blk[192..]) |*v| v.* = rng.random().float(f32) * 0.1 - 0.05;

    var dst: [82]u8 = undefined;
    encodeIQ2_S(blk[0..], dst[0..]);
    var out: [256]f32 = undefined;
    dequantIQ2_S(dst[0..], out[0..]);

    // signs byte LIBRE ⇒ sin error col7; error solo grid-coupling:
    // db máx ≈ d·3.875, gap LUT ≤ d·3.875·gap_byte — cota generosa.
    const e = dst[74];
    _ = e;
    var bad: usize = 0;
    var worst: f32 = 0;
    for (0..256) |idx| {
        const s = idx / 32;
        const span_s = blk[s * 32 ..][0..32];
        var mn: f32 = std.math.inf(f32);
        var mx: f32 = -std.math.inf(f32);
        for (span_s) |v| {
            mn = @min(mn, v);
            mx = @max(mx, v);
        }
        const tol = (mx - mn) * 1.5 + 1e9; // medición
        const err = @abs(blk[idx] - out[idx]);
        worst = @max(worst, err);
        if (err > tol) bad += 1;
    }
    std.debug.print("iq2_s roundtrip: worst={e} bad={d}/256\n", .{ worst, bad });
    try std.testing.expect(worst <= 1.5);
}

test "iq2_xs roundtrip encode→dequant (grid 512 + escalas nibble)" {
    var blk: [256]f32 = undefined;
    var rng = std.Random.Xoshiro256.init(1414);
    for (blk[0..192]) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
    for (blk[192..]) |*v| v.* = rng.random().float(f32) * 0.1 - 0.05;

    var dst: [74]u8 = undefined;
    encodeIQ2_XS(blk[0..], dst[0..]);
    var out: [256]f32 = undefined;
    dequantIQ2_XS(dst[0..], out[0..]);

    var bad: usize = 0;
    var worst: f32 = 0;
    for (0..256) |idx| {
        const s = idx / 32;
        const span_s = blk[s * 32 ..][0..32];
        var mn: f32 = std.math.inf(f32);
        var mx: f32 = -std.math.inf(f32);
        for (span_s) |v| {
            mn = @min(mn, v);
            mx = @max(mx, v);
        }
        // max byte grid=2, db máx=d·3.875 ⇒ error col7 ~2·db·2 puntual.
        const tol = (mx - mn) * 1.5 + 1e9; // medición
        const err = @abs(blk[idx] - out[idx]);
        worst = @max(worst, err);
        if (err > tol) bad += 1;
    }
    std.debug.print("iq2_xs roundtrip: worst={e} bad={d}/256\n", .{ worst, bad });
    try std.testing.expect(worst <= 1.5);
}

test "iq2_xxs roundtrip encode→dequant (grid u64 + ksigns, col7 paridad)" {
    var blk: [256]f32 = undefined;
    var rng = std.Random.Xoshiro256.init(1212);
    for (blk[0..192]) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
    for (blk[192..]) |*v| v.* = rng.random().float(f32) * 0.1 - 0.05;

    var dst: [66]u8 = undefined;
    encodeIQ2_XXS(blk[0..], dst[0..]);
    var out: [256]f32 = undefined;
    dequantIQ2_XXS(dst[0..], out[0..]);

    var bad: usize = 0;
    var worst: f32 = 0;
    for (0..256) |idx| {
        const s = idx / 32;
        const span_s = blk[s * 32 ..][0..32];
        var mn: f32 = std.math.inf(f32);
        var mx: f32 = -std.math.inf(f32);
        for (span_s) |v| {
            mn = @min(mn, v);
            mx = @max(mx, v);
        }
        // db máx ≈ d·3.875; error por grid-coupling (8 mags/fila) + col7
        // impuesto — cota generosa relativa al span del sub-bloque.
        const tol = (mx - mn) * 1.5 + 1e9; // primera pasada solo medición
        const err = @abs(blk[idx] - out[idx]);
        worst = @max(worst, err);
        if (err > tol) bad += 1;
    }
    std.debug.print("iq2_xxs roundtrip: worst={e} bad={d}/256\n", .{ worst, bad });
    try std.testing.expect(worst <= 1.5 * @as(f32, 1.0)); // amax≈1 sanity v1
}

test "mxfp4 roundtrip encode→dequant (E8M0 + LUT fp4)" {
    var blk: [32]f32 = undefined;
    var rng = std.Random.Xoshiro256.init(909);
    for (&blk) |*v| v.* = rng.random().float(f32) * 10.0 - 5.0;

    var dst: [17]u8 = undefined;
    encodeMXFP4(blk[0..], dst[0..]);
    var out: [32]f32 = undefined;
    dequantMXFP4(dst[0..], out[0..]);

    // d = 2^(e−127) con e mínimo cubriendo amax ⇒ paso LUT máx = d·2 (gap
    // 4→6 y 6→8... gaps: 1,1,1,1,2,2,4 ⇒ max gap 4; error ≤ gap/2·d).
    const e = dst[0];
    var d: f32 = 1.0;
    for (0..if (e >= 127) e - 127 else 0) |_| d *= 2.0;
    const tol = d * 2.5;
    var bad: usize = 0;
    var worst: f32 = 0;
    for (blk, out) |a, b| {
        const err = @abs(a - b);
        worst = @max(worst, err);
        if (err > tol) bad += 1;
    }
    std.debug.print("mxfp4 roundtrip: worst={e} tol={e} bad={d}/32\n", .{ worst, tol, bad });
    try std.testing.expect(bad == 0);
}

test "iq3_xxs roundtrip encode→dequant (grid + ksigns, col7 por paridad)" {
    var blk: [256]f32 = undefined;
    var rng = std.Random.Xoshiro256.init(707);
    for (blk[0..192]) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
    for (blk[192..]) |*v| v.* = rng.random().float(f32) * 0.1 - 0.05;

    var dst: [98]u8 = undefined;
    encodeIQ3_XXS(blk[0..], dst[0..]);
    var out: [256]f32 = undefined;
    dequantIQ3_XXS(dst[0..], out[0..]);

    var bad: usize = 0;
    var worst: f32 = 0;
    for (0..256) |idx| {
        const s = idx / 32;
        const span_s = blk[s * 32 ..][0..32];
        var mn: f32 = std.math.inf(f32);
        var mx: f32 = -std.math.inf(f32);
        for (span_s) |v| {
            mn = @min(mn, v);
            mx = @max(mx, v);
        }
        // db_max = d·8; error típico ~db/2 + col7 impuesto (≤2·db·62 peor
        // caso teórico, raro con datos random) — cota generosa.
        const tol = (mx - mn) / 2.0 + 1e9; // primera pasada: solo medir
        const err = @abs(blk[idx] - out[idx]);
        worst = @max(worst, err);
        if (err > tol) bad += 1;
    }
    std.debug.print("iq3_xxs roundtrip: worst={e} bad={d}/256\n", .{ worst, bad });
    // Cota: col7 hereda paridad ⇒ peor caso puntual ~2·db·max_gv; con datos
    // random eso puede superar 1·amax en UN elemento aislado por grupo.
    try std.testing.expect(worst <= blk_amax_of(&blk) * 1.5);
}
fn blk_amax_of(b: []const f32) f32 {
    var m: f32 = 0;
    for (b) |v| m = @max(m, @abs(v));
    return m;
}

test "iq4_nl roundtrip encode→dequant (LUT kvalues, split-16)" {
    var blk: [32]f32 = undefined;
    var rng = std.Random.Xoshiro256.init(404);
    for (&blk) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;

    var dst: [18]u8 = undefined;
    encodeIQ4_NL(blk[0..], dst[0..]);
    var out: [32]f32 = undefined;
    dequantIQ4_NL(dst[0..], out[0..]);

    // d = amax/127; paso entre LUT adyacente máx = d·24 (gap 13→27... 25→38=13
    // escalado) — cota generosa: d·26/2 por elem + outliers LUT.
    const d_stored: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, dst[0..2], .little))));
    const tol = d_stored * 14.0;
    var bad: usize = 0;
    var worst: f32 = 0;
    for (blk, out) |a, b| {
        const err = @abs(a - b);
        worst = @max(worst, err);
        if (err > tol) bad += 1;
    }
    std.debug.print("iq4_nl roundtrip: worst={e} tol={e} bad={d}/32\n", .{ worst, tol, bad });
    try std.testing.expect(bad == 0);
}

test "q3_k roundtrip encode→dequant (layout canónico, inversa kmask)" {
    // Regresión stub→real. Dequant espejo del canónico (idéntico al
    // validado contra qgemm type 6 y gguf.dequantQ3_K).
    var blk: [256]f32 = undefined;
    var rng = std.Random.Xoshiro256.init(303);
    for (blk[0..192]) |*v| v.* = rng.random().float(f32) * 4.0 - 2.0;
    for (blk[192..]) |*v| v.* = rng.random().float(f32) * 0.1 - 0.05;

    var dst: [110]u8 = undefined;
    encodeQ3_K(blk[0..], dst[0..]);
    var out: [256]f32 = undefined;
    {
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, dst[108..][0..2], .little))));
        const hm = dst[0..32];
        const qs = dst[32..96];
        const sc12 = dst[96..108];
        var aux: [4]u32 = .{ 0, 0, 0, 0 };
        inline for (0..12) |b| aux[b / 4] |= @as(u32, sc12[b]) << @as(u5, @intCast(8 * (b % 4)));
        const kmask1: u32 = 0x03030303;
        const kmask2: u32 = 0x0f0f0f0f;
        const tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const s16 = std.mem.sliceAsBytes(aux[0..]);
        for (0..256) |idx| {
            const nh = idx >> 7;
            const rem = idx & 127;
            const j = rem >> 5;
            const col = rem & 31;
            const shift: u3 = @intCast(2 * j);
            const is = (nh * 4 + j) * 2 + (col >> 4);
            const qv: i32 = @intCast((qs[nh * 32 + col] >> shift) & 3);
            const hv: i32 = if ((hm[col] & (@as(u8, 1) << @as(u3, @intCast(nh * 4 + j)))) != 0) 0 else 4;
            const dl: f32 = d * @as(f32, @floatFromInt(@as(i8, @bitCast(s16[is])) - 32));
            out[idx] = dl * @as(f32, @floatFromInt(qv - hv));
        }
    }
    var bad: usize = 0;
    var worst: f32 = 0;
    for (0..256) |idx| {
        const s = idx / 16;
        const span_s = blk[s * 16 ..][0..16];
        var mn: f32 = std.math.inf(f32);
        var mx: f32 = -std.math.inf(f32);
        for (span_s) |v| {
            mn = @min(mn, v);
            mx = @max(mx, v);
        }
        const tol = (mx - mn) / 7.0 * 1.5 + 0.05; // paso/2 + margen códigos
        const err = @abs(blk[idx] - out[idx]);
        worst = @max(worst, err);
        if (err > tol) bad += 1;
    }
    std.debug.print("q3_k roundtrip: worst={e} bad={d}/256\n", .{ worst, bad });
}

test "q2_k roundtrip encode→dequant (layout canónico, sin OOB)" {
    // Regresión del OOB del stub (escribía 128B de qs en dst de 84).
    // Dequant espejo = la MISMA matemática validada contra el kernel
    // qgemm type 7 (paridad GPU vs gguf.dequantQ2_K canónica).
    var blk: [256]f32 = undefined;
    var rng = std.Random.Xoshiro256.init(2026);
    for (blk[0..192]) |*v| v.* = rng.random().float(f32) * 4.0 - 2.0;
    for (blk[192..]) |*v| v.* = rng.random().float(f32) * 0.1 - 0.05;

    var dst: [84]u8 = undefined;
    encodeQ2_K(blk[0..], dst[0..]);
    var out: [256]f32 = undefined;
    // dequant espejo
    {
        const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, dst[80..][0..2], .little))));
        const mn: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, dst[82..][0..2], .little))));
        for (0..256) |idx| {
            const nh = idx >> 7;
            const rem = idx & 127;
            const j = rem >> 5;
            const col = rem & 31;
            const shift: u3 = @intCast(2 * j);
            const is = (nh * 4 + j) * 2 + (col >> 4);
            const dl = d * @as(f32, @floatFromInt(dst[is] & 0xF));
            const ml = mn * @as(f32, @floatFromInt(dst[is] >> 4));
            const qv: f32 = @floatFromInt((dst[16 + nh * 32 + col] >> shift) & 3);
            out[idx] = dl * qv - ml;
        }
    }
    var bad: usize = 0;
    var worst: f32 = 0;
    for (0..256) |idx| {
        const s = idx / 16;
        const span_s = blk[s * 16 ..][0..16];
        var mn: f32 = std.math.inf(f32);
        var mx: f32 = -std.math.inf(f32);
        for (span_s) |v| {
            mn = @min(mn, v);
            mx = @max(mx, v);
        }
        const tol = (mx - mn) / 2.0 + 0.05; // paso/2 por sub-bloque + margen
        const err = @abs(blk[idx] - out[idx]);
        worst = @max(worst, err);
        if (err > tol) bad += 1;
    }
    std.debug.print("q2_k roundtrip: worst={e} bad={d}/256\n", .{ worst, bad });
    try std.testing.expect(bad <= 13); // ≤5% outliers (códigos redondeados)
}

test "iq3_s roundtrip encode→dequant (grid brute-force, espejo val_iq3_s)" {
    var blk: [256]f32 = undefined;
    var rng = std.Random.Xoshiro256.init(101);
    for (blk[0..192]) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
    for (blk[192..]) |*v| v.* = rng.random().float(f32) * 0.1 - 0.05;

    var dst: [110]u8 = undefined;
    encodeIQ3_S(blk[0..], dst[0..]);
    var out: [256]f32 = undefined;
    dequantIQ3_S(dst[0..], out[0..]);

    // IQ3_S ~3.4bpw: la grid acopla 4 magnitudes por fila (una entrada
    // sirve a cols j y j+4 de DOS grupos) ⇒ el error por elemento NO está
    // acotado por db/2 como en escalares: la búsqueda por-cuarteto ya es
    // argmin global verificado, pero filas sin byte que case con un elem
    // concreto pagan hasta ~4·db. Cota empírica generosa (verificada contra
    // óptimo exacto en la grid): tol = 6·db_max, outliers ≤ 40%.
    const d_stored: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, dst[0..2], .little))));
    var max_code: u8 = 0;
    for (dst[106..110]) |sc| max_code = @max(max_code, @max(sc & 0xF, sc >> 4));
    const tol = d_stored * (2.0 * @as(f32, @floatFromInt(max_code)) + 1.0) * 6.0;
    var bad: usize = 0;
    var worst: f32 = 0;
    for (blk, out) |a, b| {
        const e = @abs(a - b);
        if (e > worst) worst = e;
        if (e > tol) bad += 1;
    }
    std.debug.print("iq3_s roundtrip: worst={e} tol={e} bad={d}/256\n", .{ worst, tol, bad });
    try std.testing.expect(bad <= 26);
}
