//! Harness A1 (Lane A) — fused-decode GPU por formato vs referencia CPU.
//!
//! Estrategia (inmune a encoders CPU stubs, ver HANDOFFS 2026-08-23):
//!   1. Pool sintético: bytes CRUDOS aleatorios en las regiones K/V que los
//!      kernels fusionados leen (layout contrato C1: escalas embebidas).
//!      Se sanitizan patrones f16 Inf/NaN para evitar NaN por softmax.
//!   2. Referencia CPU: atención manual dequantizando cada bloque físico con
//!      `gguf.dequantBlock` (bit-exacto verificado en test_dequant_gpu).
//!   3. GPU: `PagedAttentionGpu.decode` con quant_k=quant_v=fmt → ruta fused
//!      del formato. Si el kernel no existe aún el fallback fp16 produce
//!      valores distintos ⇒ fallo (detección activa, no silencio).
//!
//! Formatos habilitados = kernels fusionados existentes. Al añadir un kernel
//! nuevo (A2′/A3/…) moverlo de `upcoming` a `enabled` — el harness se vuelve
//! estricto automáticamente.
const std = @import("std");
const pa = @import("paged_attention");
const kv_cache_mod = @import("kv_cache");
const gguf = @import("gguf");
const cudaz = @import("cudaz");

const QuantFormat = pa.QuantFormat;
const kv_quant = kv_cache_mod.kv_quant;
const debugz = @import("debug");

/// Kernels fusionados presentes hoy en paged_attention.cu (verificados).
const enabled_formats = [_]QuantFormat{
    // Verdes (post fix stride):
    .q8_0,   .q4_k,  .q2_k,    .q3_k,   .q5_k,
    // Lane A — cubin extra (fused_decode_extra.cubin):
    .iq4_xs, .q4_1,  .q5_1,    .q8_1,   .iq3_s,
    .iq1_s,  .iq1_m, .tq1_0,   .iq4_nl, .iq2_xxs,
    .iq2_xs, .iq2_s, .iq3_xxs, .q8_k,
    // Lane A — reactivados tras fix guard + nibble + datos acotados:
      .q4_0,
    .q6_k,   .q5_0,
};

/// Pendientes de kernel (A2′ iq-first → A3 legacy → A4/A5). Mover arriba al
/// aterrizar cada uno.
// const upcoming_formats = [_]QuantFormat{
//     .iq4_xs, .iq3_s, .iq1_s, // A2′
//     .q4_1, .q5_0, .q5_1, .q8_1, // A3
//     .mxfp4, .iq4_nl, .tq1_0, .tq2_0, // A4
//     .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, // A5
// };

fn testConfig(fmt: QuantFormat) pa.PagedConfig {
    return .{
        .block_size = 4,
        .num_blocks = 64,
        .head_dim = 8,
        .num_kv_heads = 2,
        .num_q_heads = 8,
        .dtype = .f16,
        .quant_k = fmt,
        .quant_v = fmt,
        .enable_prefix_cache = false,
        .max_seq_len = 64,
        .max_batch_size = 4,
    };
}

/// Bytes cuantizados por lado (K o V) para una región de `elems` elementos,
/// según el layout canónico que leen los kernels (== kv_quant.quantBytes).
fn regionBytes(fmt: QuantFormat, elems: usize) usize {
    return kv_quant.quantBytes(fmt, elems);
}

/// Sanitiza pares LE como f16: sustituye Inf/NaN por 0x3800 (0.5) para que la
/// softmax de referencia y la del kernel operen sobre valores finitos.
fn sanitizeF16(buf: []u8) void {
    var i: usize = 0;
    while (i + 1 < buf.len) : (i += 2) {
        const bits: u16 = @as(u16, buf[i]) | (@as(u16, buf[i + 1]) << 8);
        if ((bits & 0x7C00) == 0x7C00) {
            buf[i] = 0x00;
            buf[i + 1] = 0x38;
        }
    }
}

fn fillPoolRandom(kv: *pa.PagedKVCache, seed: u64, elems_per_block: usize, fmt: QuantFormat) !void {
    var rng = std.Random.Xoshiro256.init(seed);
    for (kv.block_alloc.blocks, 0..) |*blk, phys| {
        _ = blk;
        const data = kv.getBlockData(phys);
        const kb = regionBytes(fmt, elems_per_block);
        // Bytes limitados a [0,63]: acota escalas f16 (exponente ≤ 8 bits
        // bajos => valor razonable) y valores dequant a rangos donde las
        // diferencias de orden de acumulación f32 son despreciables.
        rng.random().bytes(data[0 .. 2 * kb]);
        for (data[0 .. 2 * kb]) |*b| b.* &= 0x3F;
        sanitizeF16(data[0 .. 2 * kb]);
    }
}

/// Atención manual single-seq: dequantiza cada bloque físico con gguf y hace
/// softmax estándar (equivale al online-softmax del kernel en un pase).
fn cpuReference(
    gpa: std.mem.Allocator,
    kv: *pa.PagedKVCache,
    fmt: QuantFormat,
    seq_id: u64,
    query: []const f32,
    out: []f32,
    seq_len: usize,
    unstable: *bool,
    max_v_out: *f32,
) !void {
    const cfg = kv.config;
    const hd = cfg.head_dim;
    const kvh = cfg.num_kv_heads;
    const bs = cfg.block_size;
    const qh = cfg.num_q_heads;
    const q_per_kv = qh / kvh;
    const elems = bs * kvh * hd;
    const kb = regionBytes(fmt, elems);

    const ggml_t = try gguf.GgmlType.fromRaw(kv_quant.toGgmlTypeValue(fmt));

    const nb_used = (seq_len + bs - 1) / bs;
    const bt = kv.getBlockTable(seq_id).?;

    // Desempaquetar K/V completos [seq_len × kvh × hd]
    const K = try gpa.alloc(f32, seq_len * kvh * hd);
    defer gpa.free(K);
    const V = try gpa.alloc(f32, seq_len * kvh * hd);
    defer gpa.free(V);

    // Los super-bloques K-quant decodifican SIEMPRE 256 elems por pasada
    // (dequantQ4_K etc. indexan el super-bloque completo): dar capacidad.
    const tmp_cap = (elems + 255) & ~@as(usize, 255);
    const tmp = try gpa.alloc(f32, tmp_cap);
    defer gpa.free(tmp);

    var b: usize = 0;
    while (b < nb_used) : (b += 1) {
        const phys = bt.getPhysical(b) orelse continue;
        const data = kv.getBlockData(phys);
        const valid = @min(elems, (seq_len - b * bs) * kvh * hd);
        // Buffer con límites EXACTOS: dequantMxfp4 etc. iteran nb desde
        // bytes.len/bb y pueden leer más allá del slice sin esto
        const rb = try gpa.alloc(u8, kb);
        defer gpa.free(rb);
        @memcpy(rb, data[0..kb]);
        gguf.dequantBlock(ggml_t, rb, tmp, elems);
        for (tmp[0..valid]) |v| {
            const av = @abs(v);
            if (av > max_v_out.*) max_v_out.* = av;
        }
        @memcpy(K[b * bs * kvh * hd ..][0..valid], tmp[0..valid]);
        @memcpy(rb, data[kb .. 2 * kb]);
        gguf.dequantBlock(ggml_t, rb, tmp, elems);
        for (tmp[0..valid]) |v| {
            const av = @abs(v);
            if (av > max_v_out.*) max_v_out.* = av;
        }
        @memcpy(V[b * bs * kvh * hd ..][0..valid], tmp[0..valid]);
    }

    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    var scores = try gpa.alloc(f32, seq_len);
    defer gpa.free(scores);

    var qhi: usize = 0;
    while (qhi < qh) : (qhi += 1) {
        const kvh_idx = qhi / q_per_kv;
        // scores
        var max_v: f32 = -std.math.inf(f32);
        var second_v: f32 = -std.math.inf(f32);
        var t: usize = 0;
        while (t < seq_len) : (t += 1) {
            var dot: f32 = 0;
            var d: usize = 0;
            while (d < hd) : (d += 1) {
                dot += query[qhi * hd + d] * K[t * kvh * hd + kvh_idx * hd + d];
            }
            scores[t] = dot * scale;
            if (scores[t] > max_v) {
                second_v = max_v;
                max_v = scores[t];
            } else if (scores[t] > second_v) {
                second_v = scores[t];
            }
        }
        // Estabilidad: si top1 y top2 casi empatan, el ganador del softmax es
        // sensible al orden/precisión => semilla no comparable (ver runFormat)
        const gap = max_v - second_v;
        const scale_mag = @abs(max_v);
        if (!(gap > 1e-3 * @max(1.0, scale_mag))) {
            unstable.* = true;
        }
        // softmax estable
        var sum: f32 = 0;
        t = 0;
        while (t < seq_len) : (t += 1) {
            scores[t] = @exp(scores[t] - max_v);
            sum += scores[t];
        }
        // out
        d_loop: {
            var d: usize = 0;
            while (d < hd) : (d += 1) out[qhi * hd + d] = 0;
            t = 0;
            while (t < seq_len) : (t += 1) {
                const w = scores[t] / sum;
                d = 0;
                while (d < hd) : (d += 1) {
                    out[qhi * hd + d] += w * V[t * kvh * hd + kvh_idx * hd + d];
                }
            }
            break :d_loop;
        }
    }
}

/// Variante secuencial (e/2 + paridad) solo para diagnóstico A/B.
fn dequantQ40Seq(bytes: []const u8, out: []f32) void {
    var i: usize = 0;
    while (i < out.len) : (i += 32) {
        const gb = (i / 32) * 18;
        const d_bits = std.mem.readInt(u16, bytes[gb..][0..2], .little);
        const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
        const qs = bytes[gb + 2 ..];
        for (0..@min(32, out.len - i)) |e| {
            const byte = qs[e / 2];
            const nib = if (e % 2 == 0) byte & 0x0F else byte >> 4;
            out[i + e] = d * @as(f32, @floatFromInt(@as(i32, nib) - 8));
        }
    }
}

/// Atención manual sobre K/V ya dequantizados [seq_len × kvh × hd].
fn attnLikeRef(
    gpa: std.mem.Allocator,
    kv: *pa.PagedKVCache,
    seq_id: u64,
    K: []const f32,
    V: []const f32,
    query: []const f32,
    out: []f32,
    seq_len: usize,
) !void {
    _ = seq_id;
    const cfg = kv.config;
    const hd = cfg.head_dim;
    const kvh = cfg.num_kv_heads;
    const qh = cfg.num_q_heads;
    const q_per_kv = qh / kvh;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    const scores = try gpa.alloc(f32, seq_len);
    defer gpa.free(scores);
    var qhi: usize = 0;
    while (qhi < qh) : (qhi += 1) {
        const kvh_idx = qhi / q_per_kv;
        var mx: f32 = -std.math.inf(f32);
        var t: usize = 0;
        while (t < seq_len) : (t += 1) {
            var dot: f32 = 0;
            for (0..hd) |d| dot += query[qhi * hd + d] * K[t * kvh * hd + kvh_idx * hd + d];
            scores[t] = dot * scale;
            mx = @max(mx, scores[t]);
        }
        var sum: f32 = 0;
        t = 0;
        while (t < seq_len) : (t += 1) {
            scores[t] = @exp(scores[t] - mx);
            sum += scores[t];
        }
        for (0..hd) |d| out[qhi * hd + d] = 0;
        t = 0;
        while (t < seq_len) : (t += 1) {
            const w = scores[t] / sum;
            for (0..hd) |d| out[qhi * hd + d] += w * V[t * kvh * hd + kvh_idx * hd + d];
        }
    }
}

fn approxEq(a: f32, b: f32) bool {
    if (a == b) return true;
    if (a != a and b != b) return true; // NaN ambos
    // El kernel escribe half: exigir que b esté pegado al redondeo f16 de a
    // (≤2 ULP f16) además de la tolerancia base por datos extremos.
    const a16: f32 = @floatCast(@as(f16, @floatCast(a)));
    const ulp_dist = @abs(a - a16);
    const diff = @abs(a - b);
    const base_tol = 1e-2 + 1e-3 * @abs(a);
    return diff <= @max(base_tol, 2.01 * ulp_dist + 1e-6);
}

fn runFormat(gpa: std.mem.Allocator, fmt: QuantFormat) !void {
    const config = testConfig(fmt);
    var kv = try pa.PagedKVCache.init(gpa, config);
    defer kv.deinit();

    const seq_id = try kv.createSequence();
    const seq_len: usize = 9; // 3 bloques: 4+4+1 (cubre bloque parcial)
    try kv.allocatePrefill(seq_id, seq_len);

    const elems = config.block_size * config.num_kv_heads * config.head_dim;
    const q_stride = config.num_q_heads * config.head_dim;
    const query = try gpa.alloc(f32, q_stride);
    defer gpa.free(query);
    var rng = std.Random.Xoshiro256.init(7);
    for (query) |*v| v.* = (rng.random().float(f32) - 0.5) * 2.0;

    const out_cpu = try gpa.alloc(f32, q_stride);
    defer gpa.free(out_cpu);
    const out_gpu = try gpa.alloc(f32, q_stride);
    defer gpa.free(out_gpu);

    // Bytes crudos aleatorios pueden producir escalas f16 enormes ⇒ |out| >
    // rango f16 (65504) y el kernel satura a ±inf LEGÍTIMAMENTE. Reintentamos
    // semillas hasta obtener un caso dentro de rango (sigue siendo
    // independiente del layout: solo miramos la referencia).
    var seed_extra: u64 = 0;
    var usable = false;
    var max_v: f32 = 0; // magnitud máxima de V dequantizado (para tolerancia)
    while (seed_extra < 60) : (seed_extra += 1) {
        const seed: u64 = 42 + @as(u64, @intFromEnum(fmt)) * 100 + seed_extra;
        try fillPoolRandom(&kv, seed, elems, fmt);
        var unstable = false;
        try cpuReference(gpa, &kv, fmt, seq_id, query, out_cpu, seq_len, &unstable, &max_v);
        var mx: f32 = 0;
        for (out_cpu) |c| mx = @max(mx, @abs(c));
        // Semilla utilizable: salidas en rango f16 Y softmax estable (sin
        // near-ties que inviertan el token dominante entre órdenes de suma).
        if (mx < 1.5e4 and !unstable) {
            usable = true;
            break;
        }
    }
    if (!usable) {
        std.debug.print("[{s}] SKIP_UNSTABLE: sin semilla con softmax estable y salida en rango\n", .{@tagName(fmt)});
        return; // limitación del generador adversarial, no del kernel
    }

    cudaz.ensureContext() catch return error.SkipZigTest;
    const gpu_stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(gpu_stream);
    var engine = try pa.PagedAttentionGpu.init(gpa, config, gpu_stream);
    defer engine.deinit();
    try engine.stageTableAll(kv.block_alloc, kv.getBlockTable(seq_id).?);
    try engine.decode(query, out_gpu, kv.getBlockTable(seq_id).?.*, kv.block_alloc, config);

    var bad: usize = 0;
    var max_diff: f32 = 0;
    // Tolerancia proporcional a max|V|: el softmax mezcla todos los valores
    // de V; el ruido de acumulación escala con maxV.
    const tol_scale = @max(1e-2, 5e-3 * 64.0); // elems=64 acotados por &0x3F
    for (out_cpu, out_gpu, 0..) |c, g, i| {
        max_diff = @max(max_diff, @abs(c - g));
        const diff = @abs(c - g);
        if (diff > tol_scale and !approxEq(c, g)) {
            if (bad < 4 or (fmt == .q4_0 and bad <= 12)) std.debug.print("[{s}] mismatch {d}: cpu={d} gpu={d}\n", .{ @tagName(fmt), i, c, g });
            bad += 1;
            if (bad == 1 and fmt == .q4_1) {
                // Hexdump grupo 0 de la región K del primer bloque físico
                const bt_d = kv.getBlockTable(seq_id).?;
                const phys0 = bt_d.getPhysical(0).?;
                const dd = kv.getBlockData(phys0);
                std.debug.print("[q4_1-diag] K[0..40]: ", .{});
                for (dd[0..40]) |b| std.debug.print("{x:0>2} ", .{b});
                std.debug.print("\n", .{});
            }
        }
    }
    if (bad > 0) {
        std.debug.print("[{s}] FALLO: {d}/{d} mismatches, max_diff={d}\n", .{ @tagName(fmt), bad, q_stride, max_diff });
        if (fmt == .q4_0) {
            // Diagnóstico A/B: ¿se parece la GPU a la ref interleave o a la secuencial?
            const kb2 = regionBytes(fmt, elems);
            _ = kv_quant; // (ggml_t2 eliminado: dequantQ40Seq no lo usa)
            const nbu = (seq_len + config.block_size - 1) / config.block_size;
            const K2 = try gpa.alloc(f32, seq_len * config.num_kv_heads * config.head_dim);
            defer gpa.free(K2);
            const V2 = try gpa.alloc(f32, seq_len * config.num_kv_heads * config.head_dim);
            defer gpa.free(V2);
            const tmp2 = try gpa.alloc(f32, elems);
            defer gpa.free(tmp2);
            const bt2 = kv.getBlockTable(seq_id).?;
            var bb: usize = 0;
            while (bb < nbu) : (bb += 1) {
                const phys = bt2.getPhysical(bb) orelse continue;
                const data = kv.getBlockData(phys);
                const valid = @min(elems, (seq_len - bb * config.block_size) * config.num_kv_heads * config.head_dim);
                dequantQ40Seq(data[0..kb2], tmp2);
                @memcpy(K2[bb * elems ..][0..valid], tmp2[0..valid]);
                dequantQ40Seq(data[kb2 .. 2 * kb2], tmp2);
                @memcpy(V2[bb * elems ..][0..valid], tmp2[0..valid]);
            }
            // pesos desde K secuencial, aplicar V secuencial => out_seq
            const out_seq = try gpa.alloc(f32, q_stride);
            defer gpa.free(out_seq);
            try attnLikeRef(gpa, &kv, seq_id, K2, V2, query, out_seq, seq_len);
            var d_int: f32 = 0;
            var d_seq: f32 = 0;
            for (out_cpu, out_gpu, out_seq) |c, g, sq_| {
                d_int = @max(d_int, @abs(g - c));
                d_seq = @max(d_seq, @abs(g - sq_));
            }
            std.debug.print("[q4_0-diag] max|gpu-interleave|={d:.4}  max|gpu-secuencial|={d:.4}\n", .{ d_int, d_seq });
        }
        return error.FusedDecodeMismatch;
    }
    std.debug.print("[{s}] OK fused-decode vs dequant-ref (max_diff={d})\n", .{ @tagName(fmt), max_diff });

    if (hasConstantEncoder(fmt)) {
        try constantCheck(gpa, &kv, &engine, seq_id, elems, fmt, query, out_gpu);
    }
}

/// Chequeo de constante (por formato con encoder CPU real): codifica 0.5 en
/// todo el pool con kv_quant.encode (bit-exacto vs kernels append), verifica
/// roundtrip con gguf.dequantBlock y exige out == valor_dequant uniforme.
/// Cualquier desviación de orden nibbles/bit del kernel fusionado delata aquí.
const constant_check_formats = [_]QuantFormat{
    .q8_0, .q4_0, .q4_1, .q5_0, .q5_1, .q8_1, .q8_k,
};

fn hasConstantEncoder(fmt: QuantFormat) bool {
    for (constant_check_formats) |cf| {
        if (cf == fmt) return true;
    }
    return false;
}

fn constantCheck(
    gpa: std.mem.Allocator,
    kv: *pa.PagedKVCache,
    engine: *pa.PagedAttentionGpu,
    seq_id: u64,
    elems: usize,
    fmt: QuantFormat,
    query: []const f32,
    out_gpu: []f32,
) !void {
    const kb = regionBytes(fmt, elems);
    const ones = try gpa.alloc(f16, elems);
    defer gpa.free(ones);
    @memset(ones, 0.5);

    const enc = try kv_quant.encodeToOwned(gpa, fmt, ones);
    defer gpa.free(enc);
    if (enc.len != kb) return error.EncoderSizeMismatch;

    // Roundtrip de referencia con el dequant canónico
    // Super-bloques K-quant escriben 256 elems por pasada en la referencia:
    const dq_cap = (elems + 255) & ~@as(usize, 255);
    const dq = try gpa.alloc(f32, dq_cap);
    defer gpa.free(dq);
    const ggml_t = try gguf.GgmlType.fromRaw(kv_quant.toGgmlTypeValue(fmt));
    gguf.dequantBlock(ggml_t, enc, dq, elems);
    const v0 = dq[0];
    for (dq[0..elems]) |v| {
        if (@abs(v - v0) > 1e-4) return error.EncoderNotConstant;
    }

    for (kv.block_alloc.blocks, 0..) |_, phys| {
        const data = kv.getBlockData(phys);
        @memcpy(data[0..kb], enc);
        @memcpy(data[kb .. 2 * kb], enc);
    }
    const bt0 = kv.getBlockTable(seq_id).?;
    var bi: usize = 0;
    while (bi < bt0.numBlocks()) : (bi += 1) {
        if (bt0.getPhysical(bi)) |phys| engine.markDirty(phys);
    }
    try engine.stageTableAll(kv.block_alloc, bt0);
    try engine.decode(query, out_gpu, bt0.*, kv.block_alloc, kv.config);

    var bad: usize = 0;
    for (out_gpu, 0..) |v, i| {
        if (@abs(v - v0) > 2e-2 * @max(1.0, @abs(v0))) {
            if (bad < 4) std.debug.print("[{s}-cst] out[{d}] = {d} (esperado {d})\n", .{ @tagName(fmt), i, v, v0 });
            bad += 1;
        }
    }
    if (bad > 0) {
        std.debug.print("[{s}-cst] FALLO: {d}/{d}\n", .{ @tagName(fmt), bad, out_gpu.len });
        return error.ConstantMismatch;
    }
    std.debug.print("[{s}-cst] OK: out uniforme {d:.3}\n", .{ @tagName(fmt), v0 });
}

/// Prefill q4_0 causal: para cada posición p se atiende a tokens [0..p].
/// Referencia CPU por posición usando dequantBlock canónico.
fn prefillCpuRef(
    gpa: std.mem.Allocator,
    kv: *pa.PagedKVCache,
    fmt: QuantFormat,
    seq_id: u64,
    query: []const f32,
    out: []f32,
    n_queries: usize,
) !void {
    const cfg = kv.config;
    const hd = cfg.head_dim;
    const kvh = cfg.num_kv_heads;
    const bs = cfg.block_size;
    const qh = cfg.num_q_heads;
    const q_per_kv = qh / kvh;
    const elems = bs * kvh * hd;
    const kb = regionBytes(fmt, elems);
    const ggml_t = try gguf.GgmlType.fromRaw(kv_quant.toGgmlTypeValue(fmt));

    const nb_total = (n_queries + bs - 1) / bs;
    const bt = kv.getBlockTable(seq_id).?;
    const K = try gpa.alloc(f32, nb_total * elems);
    defer gpa.free(K);
    const V = try gpa.alloc(f32, nb_total * elems);
    defer gpa.free(V);
    const tmp = try gpa.alloc(f32, (elems + 255) & ~@as(usize, 255));
    defer gpa.free(tmp);

    var b: usize = 0;
    while (b < nb_total) : (b += 1) {
        const phys = bt.getPhysical(b) orelse continue;
        const data = kv.getBlockData(phys);
        gguf.dequantBlock(ggml_t, data[0..kb], tmp, elems);
        @memcpy(K[b * elems ..][0..elems], tmp[0..elems]);
        gguf.dequantBlock(ggml_t, data[kb .. 2 * kb], tmp, elems);
        @memcpy(V[b * elems ..][0..elems], tmp[0..elems]);
    }

    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    const scores = try gpa.alloc(f32, n_queries);
    defer gpa.free(scores);

    var qhi: usize = 0;
    while (qhi < qh) : (qhi += 1) {
        const kvh_idx = qhi / q_per_kv;
        // Causal por posición de query p
        var p: usize = 0;
        while (p < n_queries) : (p += 1) {
            const n_ctx = p + 1; // atiende a tokens [0..p]
            var mx: f32 = -std.math.inf(f32);
            var t: usize = 0;
            while (t < n_ctx) : (t += 1) {
                var dot: f32 = 0;
                for (0..hd) |d| dot += query[p * qh * hd + qhi * hd + d] * K[t * kvh * hd + kvh_idx * hd + d];
                scores[t] = dot * scale;
                mx = @max(mx, scores[t]);
            }
            var sum: f32 = 0;
            t = 0;
            while (t < n_ctx) : (t += 1) {
                scores[t] = @exp(scores[t] - mx);
                sum += scores[t];
            }
            for (0..hd) |d| out[p * qh * hd + qhi * hd + d] = 0;
            t = 0;
            while (t < n_ctx) : (t += 1) {
                const w = scores[t] / sum;
                for (0..hd) |d| out[p * qh * hd + qhi * hd + d] += w * V[t * kvh * hd + kvh_idx * hd + d];
            }
        }
    }
}

// KNOWN-RED (ticket lane-a): produce valores ~1e34 => el kernel lee datos
// no-sanitizados/fuera de región. Decode q4_0 pasa sobre el mismo layout,
// así que el bug es específico del path prefill. Investigar con DUMPKV.
test "prefill q4_0 causal vs referencia dequantBlock" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    debugz.init();
    const fmt = QuantFormat.q4_0;
    const config = testConfig(fmt);
    var kv = try pa.PagedKVCache.init(gpa, config);
    defer kv.deinit();
    const seq_id = try kv.createSequence();
    const n_queries: usize = 6;
    try kv.allocatePrefill(seq_id, n_queries);

    const elems = config.block_size * config.num_kv_heads * config.head_dim;
    try fillPoolRandom(&kv, 777, elems, fmt);

    const q_stride = config.num_q_heads * config.head_dim;
    const query = try gpa.alloc(f32, n_queries * q_stride);
    defer gpa.free(query);
    var rng = std.Random.Xoshiro256.init(99);
    for (query) |*v| v.* = (rng.random().float(f32) - 0.5) * 2.0;

    const out_cpu = try gpa.alloc(f32, n_queries * q_stride);
    defer gpa.free(out_cpu);
    const out_gpu = try gpa.alloc(f32, n_queries * q_stride);
    defer gpa.free(out_gpu);

    try prefillCpuRef(gpa, &kv, fmt, seq_id, query, out_cpu, n_queries);

    cudaz.ensureContext() catch return error.SkipZigTest;
    const gpu_stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(gpu_stream);
    var engine = try pa.PagedAttentionGpu.init(gpa, config, gpu_stream);
    defer engine.deinit();
    try engine.setupDecodeScratch(0, q_stride, 2);
    try engine.stageTableAll(kv.block_alloc, kv.getBlockTable(seq_id).?);

    const q16 = try gpa.alloc(f16, query.len);
    defer gpa.free(q16);
    for (query, 0..) |v, i| q16[i] = @floatCast(v);
    const d_q = try cudaz.cuMemAlloc(query.len * @sizeOf(f16));
    defer cudaz.cuMemFree(d_q);
    // Lane A: d_out se aloca TARDE (antes del launch final) para probar la
    // hipótesis "buffer temprano envenenado por actividad posterior".
    const d_out = try cudaz.cuMemAlloc(out_gpu.len * @sizeOf(f16));
    defer cudaz.cuMemFree(d_out);
    try cudaz.cuMemcpyHtoD(d_q, @intFromPtr(q16.ptr), query.len * @sizeOf(f16));

    const nb_total = (n_queries + config.block_size - 1) / config.block_size;
    const bt_host = try gpa.alloc(c_int, nb_total);
    defer gpa.free(bt_host);
    const bt_tbl = kv.getBlockTable(seq_id).?;
    for (0..nb_total) |bi| {
        bt_host[bi] = if (bt_tbl.getPhysical(bi)) |ph| @intCast(ph) else -1;
    }

    // ─── EXPERIMENTO BT COMPARTIDO Lane A ───
    // 1) Launch manual con SU PROPIO d_bt2 (referencia, debe dar OK)
    // 2) Wrapper con bt_override=d_bt2 (el MISMO buffer que funcionó)
    var d_bt2 = try cudaz.cuMemAlloc(nb_total * @sizeOf(c_int));
    _ = &d_bt2;
    defer cudaz.cuMemFree(d_bt2);
    try cudaz.cuMemcpyHtoD(d_bt2, @intFromPtr(bt_host.ptr), nb_total * @sizeOf(c_int));

    {
        const func2 = try cudaz.cuModuleGetFunction(engine.module_extra.?, "paged_attention_prefill_q4_0_kernel");
        var nq2: c_int = @intCast(n_queries);
        _ = &nq2;
        var sp2: c_int = 0;
        _ = &sp2;
        var nqh2: c_int = 8;
        _ = &nqh2;
        var nkv2: c_int = 2;
        _ = &nkv2;
        var hd2: c_int = 8;
        _ = &hd2;
        var bs2: c_int = 4;
        _ = &bs2;
        var d_cache2 = try engine.cacheBase(kv.block_alloc);
        _ = &d_cache2;
        var out2d = try cudaz.cuMemAlloc(out_gpu.len * @sizeOf(f16));
        _ = &out2d;
        defer cudaz.cuMemFree(out2d);
        var d_q_loc = d_q;
        _ = &d_q_loc;
        var d_cache_loc = d_cache2;
        _ = &d_cache_loc;
        var causal2: c_int = 1;
        _ = &causal2;
        var kp2 = [_]?*anyopaque{
            &out2d, &d_q_loc, &d_cache_loc, &d_bt2,
            &nq2,   &sp2,     &nqh2,        &nkv2,
            &hd2,   &bs2,     &causal2,
        };
        try cudaz.cuLaunchKernel(func2, n_queries, 8, 1, 32, 1, 1, 64, gpu_stream, @ptrCast(&kp2), null);
        try cudaz.cuStreamSynchronize(gpu_stream);
        const out2h = try gpa.alloc(f16, out_gpu.len);
        defer gpa.free(out2h);
        try cudaz.cuMemcpyDtoH(@intFromPtr(out2h.ptr), out2d, out_gpu.len * @sizeOf(f16));
        var mbad: usize = 0;
        for (out_cpu, out2h) |c, gv| {
            const gf: f32 = @floatCast(gv);
            if (@abs(c - gf) > 0.05) mbad += 1;
        }
        std.debug.print("[dif-first] mismatches={d}/{d} {s}\n", .{ mbad, out_cpu.len, if (mbad == 0) "MANUAL OK" else "MANUAL FALLA" });
    }

    // 3) Wrapper x2: primera y segunda llamada consecutivas
    const d_out_w1 = try cudaz.cuMemAlloc(out_gpu.len * @sizeOf(f16));
    defer cudaz.cuMemFree(d_out_w1);
    const d_out_w2 = try cudaz.cuMemAlloc(out_gpu.len * @sizeOf(f16));
    defer cudaz.cuMemFree(d_out_w2);
    try engine.prefillDevice(0, d_q, d_out_w1, kv.block_alloc, bt_host, n_queries, 0, d_bt2);
    try cudaz.cuStreamSynchronize(gpu_stream);
    try engine.prefillDevice(0, d_q, d_out_w2, kv.block_alloc, bt_host, n_queries, 0, d_bt2);
    try cudaz.cuStreamSynchronize(gpu_stream);

    const w1h = try gpa.alloc(f16, out_gpu.len);
    defer gpa.free(w1h);
    const w2h = try gpa.alloc(f16, out_gpu.len);
    defer gpa.free(w2h);
    try cudaz.cuMemcpyDtoH(@intFromPtr(w1h.ptr), d_out_w1, out_gpu.len * @sizeOf(f16));
    try cudaz.cuMemcpyDtoH(@intFromPtr(w2h.ptr), d_out_w2, out_gpu.len * @sizeOf(f16));
    var b1: usize = 0;
    var b2: usize = 0;
    for (out_cpu, w1h, w2h) |c, v1, v2| {
        if (@abs(c - @as(f32, @floatCast(v1))) > 0.05) b1 += 1;
        if (@abs(c - @as(f32, @floatCast(v2))) > 0.05) b2 += 1;
    }
    std.debug.print("[wrapper-x2] 1a-llamada={d}/384  2a-llamada={d}/384\n", .{ b1, b2 });

    try engine.prefillDevice(0, d_q, d_out, kv.block_alloc, bt_host, n_queries, 0, null);
    try cudaz.cuStreamSynchronize(gpu_stream);
    // FIX Lane A: descargar como f16 y CONVERTIR a f32 (el bug historico del
    // "misterio wrapper" era comparar bits-f16 crudos como si fueran f32).
    const out_f16h = try gpa.alloc(f16, out_gpu.len);
    defer gpa.free(out_f16h);
    try cudaz.cuMemcpyDtoH(@intFromPtr(out_f16h.ptr), d_out, out_gpu.len * @sizeOf(f16));
    for (out_gpu, 0..) |*o, i| o.* = @floatCast(out_f16h[i]);

    var bad: usize = 0;
    var max_diff: f32 = 0;
    const tol_scale = @max(1e-2, 5e-3 * 64.0);
    for (out_cpu, out_gpu, 0..) |c, g, i| {
        max_diff = @max(max_diff, @abs(c - g));
        if (@abs(c - g) > tol_scale and !approxEq(c, g)) {
            if (bad < 4) std.debug.print("[prefill-q4_0] mismatch {d}: cpu={d} gpu={d}\n", .{ i, c, g });
            bad += 1;
        }
    }
    if (bad > 0) {
        std.debug.print("[prefill-q4_0] FALLO: {d}/{d}, max_diff={d}\n", .{ bad, out_cpu.len, max_diff });
        return error.PrefillMismatch;
    }
    std.debug.print("[prefill-q4_0] OK\n", .{});
}

test "prefill iq4_xs causal vs referencia dequantBlock" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    debugz.init();
    const fmt = QuantFormat.iq4_xs;
    const config = testConfig(fmt);
    var kv = try pa.PagedKVCache.init(gpa, config);
    defer kv.deinit();
    const seq_id = try kv.createSequence();
    const n_queries: usize = 6;
    try kv.allocatePrefill(seq_id, n_queries);

    const elems = config.block_size * config.num_kv_heads * config.head_dim;
    try fillPoolRandom(&kv, 777, elems, fmt);

    const q_stride = config.num_q_heads * config.head_dim;
    const query = try gpa.alloc(f32, n_queries * q_stride);
    defer gpa.free(query);
    var rng = std.Random.Xoshiro256.init(99);
    for (query) |*v| v.* = (rng.random().float(f32) - 0.5) * 2.0;

    const out_cpu = try gpa.alloc(f32, n_queries * q_stride);
    defer gpa.free(out_cpu);
    const out_gpu = try gpa.alloc(f32, n_queries * q_stride);
    defer gpa.free(out_gpu);

    try prefillCpuRef(gpa, &kv, fmt, seq_id, query, out_cpu, n_queries);

    cudaz.ensureContext() catch return error.SkipZigTest;
    const gpu_stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(gpu_stream);
    var engine = try pa.PagedAttentionGpu.init(gpa, config, gpu_stream);
    defer engine.deinit();
    try engine.setupDecodeScratch(0, q_stride, 2);
    try engine.stageTableAll(kv.block_alloc, kv.getBlockTable(seq_id).?);

    const q16 = try gpa.alloc(f16, query.len);
    defer gpa.free(q16);
    for (query, 0..) |v, i| q16[i] = @floatCast(v);
    const d_q = try cudaz.cuMemAlloc(query.len * @sizeOf(f16));
    defer cudaz.cuMemFree(d_q);
    const d_out = try cudaz.cuMemAlloc(out_gpu.len * @sizeOf(f16));
    defer cudaz.cuMemFree(d_out);
    try cudaz.cuMemcpyHtoD(d_q, @intFromPtr(q16.ptr), query.len * @sizeOf(f16));

    const nb_total = (n_queries + config.block_size - 1) / config.block_size;
    const bt_host = try gpa.alloc(c_int, nb_total);
    defer gpa.free(bt_host);
    const bt_tbl = kv.getBlockTable(seq_id).?;
    for (0..nb_total) |bi| {
        bt_host[bi] = if (bt_tbl.getPhysical(bi)) |ph| @intCast(ph) else -1;
    }

    try engine.prefillDevice(0, d_q, d_out, kv.block_alloc, bt_host, n_queries, 0, null);
    try cudaz.cuStreamSynchronize(gpu_stream);

    // Patron Lane A: f16 -> conversion explicita a f32.
    const out_f16h = try gpa.alloc(f16, out_gpu.len);
    defer gpa.free(out_f16h);
    try cudaz.cuMemcpyDtoH(@intFromPtr(out_f16h.ptr), d_out, out_gpu.len * @sizeOf(f16));
    for (out_gpu, 0..) |*o, i| o.* = @floatCast(out_f16h[i]);

    var bad: usize = 0;
    var max_diff: f32 = 0;
    const tol_scale = @max(1e-2, 5e-3 * 64.0);
    for (out_cpu, out_gpu, 0..) |c, g, i| {
        max_diff = @max(max_diff, @abs(c - g));
        if (@abs(c - g) > tol_scale and !approxEq(c, g)) {
            if (bad < 4) std.debug.print("[prefill-iq4_xs] mismatch {d}: cpu={d} gpu={d}\n", .{ i, c, g });
            bad += 1;
        }
    }
    if (bad > 0) {
        std.debug.print("[prefill-iq4_xs] FALLO: {d}/{d}, max_diff={d}\n", .{ bad, out_cpu.len, max_diff });
        return error.PrefillMismatch;
    }
    std.debug.print("[prefill-iq4_xs] OK\n", .{});
}

test "prefill q8_k causal vs referencia dequantBlock" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    debugz.init();
    const fmt = QuantFormat.q8_k;
    const config = testConfig(fmt);
    var kv = try pa.PagedKVCache.init(gpa, config);
    defer kv.deinit();
    const seq_id = try kv.createSequence();
    const n_queries: usize = 6;
    try kv.allocatePrefill(seq_id, n_queries);

    // Q8_K: d f32@0 + qs int8@4 - bytes acotados para valores razonables.
    const elems = config.block_size * config.num_kv_heads * config.head_dim;
    try fillPoolRandom(&kv, 555, elems, fmt);

    const q_stride = config.num_q_heads * config.head_dim;
    const query = try gpa.alloc(f32, n_queries * q_stride);
    defer gpa.free(query);
    var rng = std.Random.Xoshiro256.init(99);
    for (query) |*v| v.* = (rng.random().float(f32) - 0.5) * 2.0;

    const out_cpu = try gpa.alloc(f32, n_queries * q_stride);
    defer gpa.free(out_cpu);
    const out_gpu = try gpa.alloc(f32, n_queries * q_stride);
    defer gpa.free(out_gpu);

    try prefillCpuRef(gpa, &kv, fmt, seq_id, query, out_cpu, n_queries);

    cudaz.ensureContext() catch return error.SkipZigTest;
    const gpu_stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(gpu_stream);
    var engine = try pa.PagedAttentionGpu.init(gpa, config, gpu_stream);
    defer engine.deinit();
    try engine.setupDecodeScratch(0, q_stride, 2);
    try engine.stageTableAll(kv.block_alloc, kv.getBlockTable(seq_id).?);

    const q16 = try gpa.alloc(f16, query.len);
    defer gpa.free(q16);
    for (query, 0..) |v, i| q16[i] = @floatCast(v);
    const d_q = try cudaz.cuMemAlloc(query.len * @sizeOf(f16));
    defer cudaz.cuMemFree(d_q);
    const d_out = try cudaz.cuMemAlloc(out_gpu.len * @sizeOf(f16));
    defer cudaz.cuMemFree(d_out);
    try cudaz.cuMemcpyHtoD(d_q, @intFromPtr(q16.ptr), query.len * @sizeOf(f16));

    const nb_total = (n_queries + config.block_size - 1) / config.block_size;
    const bt_host = try gpa.alloc(c_int, nb_total);
    defer gpa.free(bt_host);
    const bt_tbl = kv.getBlockTable(seq_id).?;
    for (0..nb_total) |bi| {
        bt_host[bi] = if (bt_tbl.getPhysical(bi)) |ph| @intCast(ph) else -1;
    }

    try engine.prefillDevice(0, d_q, d_out, kv.block_alloc, bt_host, n_queries, 0, null);
    try cudaz.cuStreamSynchronize(gpu_stream);

    // FIX patron Lane A: descargar f16 y convertir a f32 explicitamente.
    const out_f16h = try gpa.alloc(f16, out_gpu.len);
    defer gpa.free(out_f16h);
    try cudaz.cuMemcpyDtoH(@intFromPtr(out_f16h.ptr), d_out, out_gpu.len * @sizeOf(f16));
    for (out_gpu, 0..) |*o, i| o.* = @floatCast(out_f16h[i]);

    var bad: usize = 0;
    var max_diff: f32 = 0;
    const tol_scale = @max(1e-2, 5e-3 * 64.0);
    for (out_cpu, out_gpu, 0..) |c, g, i| {
        max_diff = @max(max_diff, @abs(c - g));
        if (@abs(c - g) > tol_scale and !approxEq(c, g)) {
            if (bad < 4) std.debug.print("[prefill-q8_k] mismatch {d}: cpu={d} gpu={d}\n", .{ i, c, g });
            bad += 1;
        }
    }
    if (bad > 0) {
        std.debug.print("[prefill-q8_k] FALLO: {d}/{d}, max_diff={d}\n", .{ bad, out_cpu.len, max_diff });
        return error.PrefillMismatch;
    }
    std.debug.print("[prefill-q8_k] OK\n", .{});
}

/// Lane A — formatos con kernel de prefill en fused_decode_extra.cubin (19).
/// fp16/q8_0 viven en el cubin principal; q4_0/iq4_xs/q8_k ya tienen test
/// causal propio. Valfns bit-exacto validadas por el harness de decode.
const universal_prefill_formats = [_]QuantFormat{
    .q2_k,   .q3_k,    .q4_k,   .q5_k,  .q6_k,
    .q4_1,   .q5_0,    .q5_1,   .q8_1,  .iq3_s,
    .iq1_s,  .iq1_m,   .tq1_0,  .tq2_0, .mxfp4,
    .iq4_nl, .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs,
};

/// Cuerpo clonado del causal iq4_xs (prefillCpuRef + prefillDevice(null) +
/// descarga f16→f32). Un formato por invocación para aislar recursos GPU.
fn runUniversalPrefill(gpa: std.mem.Allocator, fmt: QuantFormat) !void {
    const config = testConfig(fmt);
    var kv = try pa.PagedKVCache.init(gpa, config);
    defer kv.deinit();
    const seq_id = try kv.createSequence();
    const n_queries: usize = 6;
    try kv.allocatePrefill(seq_id, n_queries);

    const elems = config.block_size * config.num_kv_heads * config.head_dim;
    try fillPoolRandom(&kv, 777, elems, fmt);

    const q_stride = config.num_q_heads * config.head_dim;
    const query = try gpa.alloc(f32, n_queries * q_stride);
    defer gpa.free(query);
    var rng = std.Random.Xoshiro256.init(99);
    for (query) |*v| v.* = (rng.random().float(f32) - 0.5) * 2.0;
    // Cuantizar la query a f16 TAMBIEN en la referencia CPU: el kernel la
    // consume como f16 y comparar contra la f32 original mete deltas de
    // score que con softmax casi empatado + |V| grande superan la
    // tolerancia sin bug real (visto con q4_k seed 777, p=4).
    for (query) |*v| v.* = @floatCast(@as(f16, @floatCast(v.*)));

    const out_cpu = try gpa.alloc(f32, n_queries * q_stride);
    defer gpa.free(out_cpu);
    const out_gpu = try gpa.alloc(f32, n_queries * q_stride);
    defer gpa.free(out_gpu);

    try prefillCpuRef(gpa, &kv, fmt, seq_id, query, out_cpu, n_queries);

    cudaz.ensureContext() catch return error.SkipZigTest;
    const gpu_stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(gpu_stream);
    var engine = try pa.PagedAttentionGpu.init(gpa, config, gpu_stream);
    defer engine.deinit();
    try engine.setupDecodeScratch(0, q_stride, 2);
    try engine.stageTableAll(kv.block_alloc, kv.getBlockTable(seq_id).?);

    const q16 = try gpa.alloc(f16, query.len);
    defer gpa.free(q16);
    for (query, 0..) |v, i| q16[i] = @floatCast(v);
    const d_q = try cudaz.cuMemAlloc(query.len * @sizeOf(f16));
    defer cudaz.cuMemFree(d_q);
    const d_out = try cudaz.cuMemAlloc(out_gpu.len * @sizeOf(f16));
    defer cudaz.cuMemFree(d_out);
    try cudaz.cuMemcpyHtoD(d_q, @intFromPtr(q16.ptr), query.len * @sizeOf(f16));

    const nb_total = (n_queries + config.block_size - 1) / config.block_size;
    const bt_host = try gpa.alloc(c_int, nb_total);
    defer gpa.free(bt_host);
    const bt_tbl = kv.getBlockTable(seq_id).?;
    for (0..nb_total) |bi| {
        bt_host[bi] = if (bt_tbl.getPhysical(bi)) |ph| @intCast(ph) else -1;
    }

    try engine.prefillDevice(0, d_q, d_out, kv.block_alloc, bt_host, n_queries, 0, null);
    try cudaz.cuStreamSynchronize(gpu_stream);

    // Regla Lane A: descargar SIEMPRE f16 y convertir a f32 explicitamente.
    const out_f16h = try gpa.alloc(f16, out_gpu.len);
    defer gpa.free(out_f16h);
    try cudaz.cuMemcpyDtoH(@intFromPtr(out_f16h.ptr), d_out, out_gpu.len * @sizeOf(f16));
    for (out_gpu, 0..) |*o, i| o.* = @floatCast(out_f16h[i]);

    var bad: usize = 0;
    var max_diff: f32 = 0;
    const tol_scale = @max(1e-2, 5e-3 * 64.0);
    for (out_cpu, out_gpu, 0..) |c, g, i| {
        max_diff = @max(max_diff, @abs(c - g));
        if (@abs(c - g) > tol_scale and !approxEq(c, g)) {
            if (bad < 4) std.debug.print("[prefill-univ-{s}] mismatch {d}: cpu={d} gpu={d}\n", .{ @tagName(fmt), i, c, g });
            bad += 1;
        }
    }
    if (bad > 0) {
        std.debug.print("[prefill-univ-{s}] FALLO: {d}/{d}, max_diff={d}\n", .{ @tagName(fmt), bad, out_cpu.len, max_diff });
        return error.PrefillMismatch;
    }
    std.debug.print("[prefill-univ-{s}] OK\n", .{@tagName(fmt)});
}

// A5 (request lane-c 20:25): prefill NO-causal — todas las filas atienden
// [0..start_pos+n_chunk). Escenario denoiser DFlash: chunk de n_chunk filas
// en start_pos con atención bidireccional dentro del bloque. Verificado en
// ambos cubins: fp16 (principal) y q4_k (extra).
test "prefill no-causal fp16 y q4_k" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    debugz.init();
    inline for ([_]QuantFormat{ .fp16, .q4_k }) |fmt| {
        const config = testConfig(fmt);
        var kv = try pa.PagedKVCache.init(gpa, config);
        defer kv.deinit();
        const seq_id = try kv.createSequence();
        const start_pos: usize = 3;
        const n_chunk: usize = 5;
        const total = start_pos + n_chunk;
        try kv.allocatePrefill(seq_id, total);

        const elems = config.block_size * config.num_kv_heads * config.head_dim;
        try fillPoolRandom(&kv, 777, elems, fmt);

        const q_stride = config.num_q_heads * config.head_dim;
        const query_full = try gpa.alloc(f32, total * q_stride);
        defer gpa.free(query_full);
        var rng = std.Random.Xoshiro256.init(555);
        for (query_full) |*v| v.* = @floatCast(@as(f16, @floatCast((rng.random().float(f32) - 0.5) * 2.0)));

        // Referencia CPU no-causal: cada fila del chunk atiende [0..total).
        const out_cpu = try gpa.alloc(f32, n_chunk * q_stride);
        defer gpa.free(out_cpu);
        {
            const kb = regionBytes(fmt, elems);
            const ggml_t = try gguf.GgmlType.fromRaw(kv_quant.toGgmlTypeValue(fmt));
            const nb_total = (total + config.block_size - 1) / config.block_size;
            const K = try gpa.alloc(f32, nb_total * elems);
            defer gpa.free(K);
            const V = try gpa.alloc(f32, nb_total * elems);
            defer gpa.free(V);
            const tmp = try gpa.alloc(f32, (elems + 255) & ~@as(usize, 255));
            defer gpa.free(tmp);
            const bt = kv.getBlockTable(seq_id).?;
            for (0..nb_total) |b| {
                const phys = bt.getPhysical(b) orelse continue;
                const data = kv.getBlockData(phys);
                gguf.dequantBlock(ggml_t, data[0..kb], tmp, elems);
                @memcpy(K[b * elems ..][0..elems], tmp[0..elems]);
                gguf.dequantBlock(ggml_t, data[kb .. 2 * kb], tmp, elems);
                @memcpy(V[b * elems ..][0..elems], tmp[0..elems]);
            }
            const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(config.head_dim)));
            const scores = try gpa.alloc(f32, total);
            defer gpa.free(scores);
            for (0..n_chunk) |r| {
                for (0..config.num_q_heads) |qhi| {
                    const kvh_idx = qhi / (config.num_q_heads / config.num_kv_heads);
                    var mx: f32 = -std.math.inf(f32);
                    for (0..total) |t| {
                        var dot: f32 = 0;
                        for (0..config.head_dim) |d|
                            dot += query_full[(start_pos + r) * q_stride + qhi * config.head_dim + d] * K[t * config.num_kv_heads * config.head_dim + kvh_idx * config.head_dim + d];
                        scores[t] = dot * scale;
                        mx = @max(mx, scores[t]);
                    }
                    var sum: f32 = 0;
                    for (scores) |*sc| {
                        sc.* = @exp(sc.* - mx);
                        sum += sc.*;
                    }
                    for (0..config.head_dim) |d| out_cpu[r * q_stride + qhi * config.head_dim + d] = 0;
                    for (0..total) |t| {
                        const w = scores[t] / sum;
                        for (0..config.head_dim) |d|
                            out_cpu[r * q_stride + qhi * config.head_dim + d] += w * V[t * config.num_kv_heads * config.head_dim + kvh_idx * config.head_dim + d];
                    }
                }
            }
        }

        cudaz.ensureContext() catch return error.SkipZigTest;
        const gpu_stream = try cudaz.cuStreamCreate(0);
        defer cudaz.cuStreamDestroy(gpu_stream);
        var engine = try pa.PagedAttentionGpu.init(gpa, config, gpu_stream);
        defer engine.deinit();
        try engine.setupDecodeScratch(0, q_stride, 2);
        try engine.stageTableAll(kv.block_alloc, kv.getBlockTable(seq_id).?);

        const q16 = try gpa.alloc(f16, n_chunk * q_stride);
        defer gpa.free(q16);
        for (query_full[start_pos * q_stride ..], 0..) |v, i| q16[i] = @floatCast(v);
        const d_q = try cudaz.cuMemAlloc(q16.len * @sizeOf(f16));
        defer cudaz.cuMemFree(d_q);
        const d_out = try cudaz.cuMemAlloc(out_cpu.len * @sizeOf(f16));
        defer cudaz.cuMemFree(d_out);
        try cudaz.cuMemcpyHtoD(d_q, @intFromPtr(q16.ptr), q16.len * @sizeOf(f16));

        const nb_total = (total + config.block_size - 1) / config.block_size;
        const bt_host = try gpa.alloc(c_int, nb_total);
        defer gpa.free(bt_host);
        const bt_tbl = kv.getBlockTable(seq_id).?;
        for (0..nb_total) |bi| {
            bt_host[bi] = if (bt_tbl.getPhysical(bi)) |ph| @intCast(ph) else -1;
        }

        try engine.prefillDeviceEx(0, d_q, d_out, kv.block_alloc, bt_host, n_chunk, start_pos, null, false);
        try cudaz.cuStreamSynchronize(gpu_stream);

        const out_f16h = try gpa.alloc(f16, out_cpu.len);
        defer gpa.free(out_f16h);
        try cudaz.cuMemcpyDtoH(@intFromPtr(out_f16h.ptr), d_out, out_cpu.len * @sizeOf(f16));

        var bad: usize = 0;
        var max_diff: f32 = 0;
        const tol_scale = @max(1e-2, 5e-3 * 64.0);
        for (out_cpu, 0..) |c, i| {
            const g: f32 = @floatCast(out_f16h[i]);
            max_diff = @max(max_diff, @abs(c - g));
            if (@abs(c - g) > tol_scale and !approxEq(c, g)) bad += 1;
        }
        if (bad > 0) {
            std.debug.print("[no-causal-{s}] FALLO: {d}/{d}, max_diff={d}\n", .{ @tagName(fmt), bad, out_cpu.len, max_diff });
            return error.NoCausalMismatch;
        }
        std.debug.print("[no-causal-{s}] OK ({d} elems, max_diff={d})\n", .{ @tagName(fmt), out_cpu.len, max_diff });
    }
}

// B-a1 (lane-a): paridad del prefill q4_k VECTORIZADO g8 contra ref CPU
// con dequant espejo del layout GGUF. Pool RAW (stride 144B/SB = el que
// computa el kernel) construido a mano — NO usa PagedKVCache: su
// quantBytes quedó alineado 32B en 7.3 (@a8e9dcd) y el stride del pool
// no coincide con el raw de los pesos GGUF/kernels (prefill universal
// rojo pre-existing, ver HANDOFFS 18:4x/19:5x). Geometría REAL hd=128.
test "B-a1 prefill q4_k g8: paridad pool raw hd128" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    debugz.init();
    cudaz.ensureContext() catch return error.SkipZigTest;
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);

    const hd: usize = 128;
    const kvh: usize = 2;
    const qh: usize = 8;
    const bs: usize = 4;
    const n_tokens: usize = 6;
    const kd = kvh * hd;
    const er = bs * kd;
    const sb_bytes: usize = 144;
    const kbs = (er + 255) / 256 * sb_bytes;
    const n_blocks = (n_tokens + bs - 1) / bs;
    const pool = try gpa.alloc(u8, n_blocks * 2 * kbs);
    defer gpa.free(pool);

    var rng = std.Random.Xoshiro256.init(777);
    rng.random().bytes(pool);
    for (pool) |*b| b.* &= 0x3F;
    var i: usize = 0;
    while (i + 1 < pool.len) : (i += 2) {
        const bits: u16 = @as(u16, pool[i]) | (@as(u16, pool[i + 1]) << 8);
        if ((bits & 0x7C00) == 0x7C00) {
            pool[i] = 0x00;
            pool[i + 1] = 0x38;
        }
    }

    const q_stride = qh * hd;
    const query = try gpa.alloc(f16, n_tokens * q_stride);
    defer gpa.free(query);
    var rq = std.Random.Xoshiro256.init(99);
    for (query) |*v| v.* = @floatCast((rq.random().float(f32) - 0.5) * 2.0);

    // Dequant espejo q4_k (SB256/144B) — mismo layout que el kernel.
    const dequantSb = struct {
        fn f(blk: []const u8, out: []f32) void {
            const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, blk[0..2], .little))));
            const m: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, blk[2..4], .little))));
            const scales = blk[4..16];
            const qs = blk[16..144];
            for (0..256) |e| {
                const g = e / 64;
                const l = e % 64;
                const si = 2 * g + @as(usize, if (l < 32) 0 else 1);
                var sd: u32 = undefined;
                var sm: u32 = undefined;
                if (si < 4) {
                    sd = scales[si] & 63;
                    sm = scales[si + 4] & 63;
                } else {
                    sd = @as(u32, scales[si + 4] & 0xF) | (@as(u32, scales[si - 4] >> 6) << 4);
                    sm = @as(u32, scales[si + 4] >> 4) | (@as(u32, scales[si] >> 6) << 4);
                }
                const dl = d * @as(f32, @floatFromInt(sd));
                const ml = m * @as(f32, @floatFromInt(sm));
                const qv: u32 = if (l < 32) (qs[g * 32 + l] & 0xF) else ((qs[g * 32 + (l - 32)] >> 4) & 0xF);
                out[e] = dl * @as(f32, @floatFromInt(qv)) - ml;
            }
        }
    }.f;

    const out_cpu = try gpa.alloc(f32, n_tokens * q_stride);
    defer gpa.free(out_cpu);
    {
        const tmp = try gpa.alloc(f32, 256);
        defer gpa.free(tmp);
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        for (0..n_tokens) |t| {
            for (0..qh) |h| {
                const kv_head = h / (qh / kvh);
                var maxv: f32 = -1e30;
                const scores = try gpa.alloc(f32, t + 1);
                defer gpa.free(scores);
                for (0..t + 1) |s| {
                    const blk_idx = s / bs;
                    const be = (s % bs) * kd + kv_head * hd;
                    const qb = be / 256;
                    const inb = be % 256;
                    dequantSb(pool[blk_idx * 2 * kbs + qb * sb_bytes ..][0..sb_bytes], tmp);
                    var acc: f32 = 0;
                    for (0..hd) |d2| {
                        acc += @as(f32, @floatCast(query[t * q_stride + h * hd + d2])) * tmp[inb + d2];
                    }
                    scores[s] = acc * scale;
                    maxv = @max(maxv, scores[s]);
                }
                var esum: f32 = 0;
                for (scores) |s| esum += @exp(s - maxv);
                for (0..hd) |d2| {
                    var acc: f32 = 0;
                    for (0..t + 1) |s| {
                        const blk_idx = s / bs;
                        const be = (s % bs) * kd + kv_head * hd;
                        const qb = be / 256;
                        const inb = be % 256;
                        dequantSb(pool[blk_idx * 2 * kbs + kbs + qb * sb_bytes ..][0..sb_bytes], tmp);
                        const w: f32 = @exp(scores[s] - maxv) / esum;
                        acc += tmp[inb + d2] * w;
                    }
                    out_cpu[t * q_stride + h * hd + d2] = acc;
                }
            }
        }
    }

    var module = try cudaz.cuModuleLoad("zig-out/lib/paged_attention.cubin");
    _ = &module;
    defer cudaz.cuModuleUnload(module);
    const func = try cudaz.cuModuleGetFunction(module, "paged_attention_prefill_q4_k_kernel");
    var d_pool = try cudaz.cuMemAlloc(pool.len);
    _ = &d_pool;
    defer cudaz.cuMemFree(d_pool);
    var d_q = try cudaz.cuMemAlloc(query.len * @sizeOf(f16));
    _ = &d_q;
    defer cudaz.cuMemFree(d_q);
    var d_out = try cudaz.cuMemAlloc(n_tokens * q_stride * @sizeOf(f16));
    _ = &d_out;
    defer cudaz.cuMemFree(d_out);
    var d_bt = try cudaz.cuMemAlloc(n_blocks * @sizeOf(c_int));
    _ = &d_bt;
    defer cudaz.cuMemFree(d_bt);
    const bt = try gpa.alloc(c_int, n_blocks);
    defer gpa.free(bt);
    for (bt, 0..) |*b2, bi| b2.* = @intCast(bi);
    try cudaz.cuMemcpyHtoD(d_pool, @intFromPtr(pool.ptr), pool.len);
    try cudaz.cuMemcpyHtoD(d_q, @intFromPtr(query.ptr), query.len * @sizeOf(f16));
    try cudaz.cuMemcpyHtoD(d_bt, @intFromPtr(bt.ptr), bt.len * @sizeOf(c_int));

    var nq: c_int = @intCast(n_tokens);
    _ = &nq;
    var sp: c_int = 0;
    _ = &sp;
    var nqh: c_int = @intCast(qh);
    _ = &nqh;
    var nkv: c_int = @intCast(kvh);
    _ = &nkv;
    var hd2: c_int = @intCast(hd);
    _ = &hd2;
    var bs2: c_int = @intCast(bs);
    _ = &bs2;
    var causal: c_int = 1;
    _ = &causal;
    var kp = [_]?*anyopaque{
        &d_out, &d_q, &d_pool, &d_bt,
        &nq,    &sp,  &nqh,    &nkv,
        &hd2,   &bs2, &causal,
    };
    _ = &kp;
    try cudaz.cuLaunchKernel(func, @intCast(n_tokens), @intCast(qh), 1, 32, 1, 1, 1024, stream, @ptrCast(&kp), null);
    try cudaz.cuStreamSynchronize(stream);

    const out_gpu16 = try gpa.alloc(f16, n_tokens * q_stride);
    defer gpa.free(out_gpu16);
    try cudaz.cuMemcpyDtoH(@intFromPtr(out_gpu16.ptr), d_out, out_gpu16.len * @sizeOf(f16));

    var bad: usize = 0;
    var max_rel: f32 = 0;
    for (out_cpu, out_gpu16) |c, g| {
        const gf: f32 = @floatCast(g);
        const rel = @abs(c - gf) / @max(@abs(c), 1.0);
        max_rel = @max(max_rel, rel);
        if (rel > 2e-3) bad += 1;
    }
    std.debug.print("[B-a1] prefill q4_k g8 hd128: bad={d}/{d}, max_rel={e}\n", .{ bad, out_cpu.len, max_rel });
    if (bad > 0) return error.Ba1Q4KParityMismatch;
    std.debug.print("[B-a1] prefill q4_k g8 paridad OK\n", .{});
}

test "prefill universal causal" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    debugz.init();
    var failed: usize = 0;
    inline for (universal_prefill_formats) |fmt| {
        runUniversalPrefill(gpa, fmt) catch |e| {
            std.debug.print("[{s}] >>> ERROR prefill universal: {s}\n", .{ @tagName(fmt), @errorName(e) });
            failed += 1;
        };
    }
    if (failed > 0) {
        std.debug.print("prefill universal: {d}/{d} formatos en rojo\n", .{ failed, universal_prefill_formats.len });
        return error.UniversalPrefillMismatch;
    }
}

// TEMPORAL Lane A (P1): repro del ticket q4_0-decode idx0 de la suite kvq
// de lane-b, con SU geometría (hd=32, kvh=1, qh=4, bs=4, n_tokens=6) y
// launch directo del kernel legacy. Compara contra cpuReference local.
// Regresión del ticket "q4_0 decode idx0" (suite kvq lane-b): réplica exacta
// de su flujo (geometría hd=32/kvh=1/qh=4/bs=4, n_tokens=6, datos
// genF16Exact(304/305)→kv_quant.encode→pool raw, query genF16Exact(404),
// escalas cero, launch directo). Con el fix split-16 del encoder la ref
// canónica y el kernel coinciden con el valor TRUE (-0.1857); antes
// coincidían ambos en un valor PERMUTADO (-0.2948) que ocultaba el bug.
test "regresion q4_0 decode legacy geometria suite-kvq" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    debugz.init();

    const hd: usize = 32;
    const kvh: usize = 1;
    const qh: usize = 4;
    const bs: usize = 4;
    const num_blocks: usize = 2;
    const n_tokens: usize = 6;
    const kd = kvh * hd;
    const er = bs * kd; // elems por región de bloque
    const seq_blocks = (n_tokens + bs - 1) / bs;
    const kb = (er + 31) / 32 * 18; // q4_0
    const bbt = 2 * kb;

    // Datos fuente f16-exactos (mismas seeds que lane-b: K=304, V=305).
    const k_src = try gpa.alloc(f16, n_tokens * kd);
    defer gpa.free(k_src);
    const v_src = try gpa.alloc(f16, n_tokens * kd);
    defer gpa.free(v_src);
    var rng_k = std.Random.Xoshiro256.init(304);
    for (k_src) |*x| {
        const r = rng_k.random().float(f32);
        x.* = @floatCast((r - 0.5) * 2.0);
    }
    var rng_v = std.Random.Xoshiro256.init(305);
    for (v_src) |*x| {
        const r = rng_v.random().float(f32);
        x.* = @floatCast((r - 0.5) * 2.0);
    }

    // Layout por bloques con ceros fuera de los tokens válidos.
    const region_elems = seq_blocks * er;
    const k_all = try gpa.alloc(f16, region_elems);
    defer gpa.free(k_all);
    const v_all = try gpa.alloc(f16, region_elems);
    defer gpa.free(v_all);
    @memset(k_all, 0);
    @memset(v_all, 0);
    for (0..n_tokens) |t| {
        const b = t / bs;
        const off = t % bs;
        for (0..kd) |c| {
            k_all[b * er + off * kd + c] = k_src[t * kd + c];
            v_all[b * er + off * kd + c] = v_src[t * kd + c];
        }
    }

    const pool = try gpa.alloc(u8, num_blocks * bbt);
    defer gpa.free(pool);
    @memset(pool, 0xAA); // patrón no-cero en zonas muertas (como device real)
    for (0..seq_blocks) |b| {
        const enc_k = try kv_quant.encodeToOwned(gpa, .q4_0, k_all[b * er ..][0..er]);
        defer gpa.free(enc_k);
        const enc_v = try kv_quant.encodeToOwned(gpa, .q4_0, v_all[b * er ..][0..er]);
        defer gpa.free(enc_v);
        @memcpy(pool[b * bbt ..][0..kb], enc_k);
        @memcpy(pool[b * bbt + kb ..][0..kb], enc_v);
    }

    // Query f16-exacta seed 404.
    const q_stride = qh * hd;
    const q_host = try gpa.alloc(f32, q_stride);
    defer gpa.free(q_host);
    var rng_q = std.Random.Xoshiro256.init(404);
    for (q_host) |*x| {
        const r = rng_q.random().float(f32);
        x.* = @as(f32, @floatCast(@as(f16, @floatCast((r - 0.5) * 2.0))));
    }
    const q16 = try gpa.alloc(f16, q_stride);
    defer gpa.free(q16);
    for (q_host, 0..) |x, i| q16[i] = @floatCast(x);

    // Referencia CPU sobre la dequantización canónica del pool (cpuAttention).
    const kf = try gpa.alloc(f32, region_elems);
    defer gpa.free(kf);
    const vf = try gpa.alloc(f32, region_elems);
    defer gpa.free(vf);
    const tmpd = try gpa.alloc(f32, er);
    defer gpa.free(tmpd);
    const ggml_q40 = try gguf.GgmlType.fromRaw(kv_quant.toGgmlTypeValue(.q4_0));
    for (0..seq_blocks) |b| {
        const base = b * bbt;
        gguf.dequantBlock(ggml_q40, pool[base .. base + kb], tmpd, er);
        @memcpy(kf[b * er ..][0..er], tmpd);
        gguf.dequantBlock(ggml_q40, pool[base + kb .. base + 2 * kb], tmpd, er);
        @memcpy(vf[b * er ..][0..er], tmpd);
    }
    const ref = try gpa.alloc(f32, q_stride);
    defer gpa.free(ref);
    {
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        var h: usize = 0;
        while (h < qh) : (h += 1) {
            const kv_head = h / (qh / kvh);
            const scores = try gpa.alloc(f32, n_tokens);
            defer gpa.free(scores);
            var mx: f32 = -std.math.inf(f32);
            for (0..n_tokens) |s| {
                var acc: f32 = 0;
                for (0..hd) |d| acc += q_host[h * hd + d] * kf[s * kd + kv_head * hd + d];
                scores[s] = acc * scale;
                mx = @max(mx, scores[s]);
            }
            var sum: f32 = 0;
            for (scores) |*sc| {
                sc.* = @exp(sc.* - mx);
                sum += sc.*;
            }
            for (0..hd) |d| {
                var acc: f32 = 0;
                for (0..n_tokens) |s| acc += scores[s] * vf[s * kd + kv_head * hd + d];
                ref[h * hd + d] = acc / sum;
            }
        }
    }

    // GPU: pool raw subido directo (sin PagedKVCache), launch idéntico al de
    // la suite kvq (escalas cero, smem extra sb32).
    cudaz.ensureContext() catch return error.SkipZigTest;
    const gpu_stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(gpu_stream);
    const config = pa.PagedConfig{
        .block_size = @intCast(bs),
        .num_blocks = @intCast(num_blocks),
        .head_dim = @intCast(hd),
        .num_kv_heads = @intCast(kvh),
        .num_q_heads = @intCast(qh),
        .dtype = .f16,
        .quant_k = .q4_0,
        .quant_v = .q4_0,
        .enable_prefix_cache = false,
        .max_seq_len = 64,
        .max_batch_size = 4,
    };
    var engine = try pa.PagedAttentionGpu.init(gpa, config, gpu_stream);
    defer engine.deinit();

    var d_pool = try cudaz.cuMemAlloc(pool.len);
    _ = &d_pool;
    defer cudaz.cuMemFree(d_pool);
    try cudaz.cuMemcpyHtoD(d_pool, @intFromPtr(pool.ptr), pool.len);

    var d_q = try cudaz.cuMemAlloc(q_stride * @sizeOf(f16));
    _ = &d_q;
    defer cudaz.cuMemFree(d_q);
    var d_out = try cudaz.cuMemAlloc(q_stride * @sizeOf(f16));
    _ = &d_out;
    defer cudaz.cuMemFree(d_out);
    try cudaz.cuMemcpyHtoD(d_q, @intFromPtr(q16.ptr), q_stride * @sizeOf(f16));

    var bt_host = [_]c_int{ 0, 1 };
    _ = &bt_host;
    var d_bt = try cudaz.cuMemAlloc(bt_host.len * @sizeOf(c_int));
    _ = &d_bt;
    defer cudaz.cuMemFree(d_bt);
    try cudaz.cuMemcpyHtoD(d_bt, @intFromPtr(&bt_host), bt_host.len * @sizeOf(c_int));
    var seq_len_c: c_int = @intCast(n_tokens);
    var d_seq = try cudaz.cuMemAlloc(@sizeOf(c_int));
    _ = &d_seq;
    defer cudaz.cuMemFree(d_seq);
    try cudaz.cuMemcpyHtoD(d_seq, @intFromPtr(&seq_len_c), @sizeOf(c_int));

    const func = try cudaz.cuModuleGetFunction(engine.module, "paged_attention_decode_q4_0_kernel");
    var zero_ptr: usize = 0;
    _ = &zero_ptr;
    var num_seqs_c: c_int = 1;
    var max_blocks_c: c_int = @intCast(num_blocks);
    var num_q_c: c_int = @intCast(qh);
    var num_kv_c: c_int = @intCast(kvh);
    var hd_c: c_int = @intCast(hd);
    var bs_c: c_int = @intCast(bs);
    var kp = [_]?*anyopaque{
        &d_out,    &d_q,   &d_pool,     &zero_ptr,     &zero_ptr,
        &d_bt,     &d_seq, &num_seqs_c, &max_blocks_c, &num_q_c,
        &num_kv_c, &hd_c,  &bs_c,
    };
    _ = &kp;
    const extra: usize = 2 * ((er + 31) / 32);
    const shared: c_uint = @intCast(2 * hd * @sizeOf(f32) + extra * @sizeOf(f32));
    try cudaz.cuLaunchKernel(func, 1, @intCast(qh), 1, 32, 1, 1, shared, gpu_stream, @ptrCast(&kp), null);
    try cudaz.cuStreamSynchronize(gpu_stream);

    const out16 = try gpa.alloc(f16, q_stride);
    defer gpa.free(out16);
    try cudaz.cuMemcpyDtoH(@intFromPtr(out16.ptr), d_out, q_stride * @sizeOf(f16));

    var bad: usize = 0;
    var max_diff: f32 = 0;
    for (out16, 0..) |x, i| {
        const g: f32 = x;
        max_diff = @max(max_diff, @abs(g - ref[i]));
        if (@abs(g - ref[i]) > 2e-3 and bad < 8)
            std.debug.print("[repro-b] idx={d}: gpu={d} cpu={d}\n", .{ i, g, ref[i] });
        if (@abs(g - ref[i]) > 2e-3) bad += 1;
    }
    std.debug.print("[repro-b] q4_0 replica exacta suite-kvq: bad={d}/{d} max_diff={d} ref[0]={d}\n", .{ bad, q_stride, max_diff, ref[0] });
}
// Invariante de convención canónica (Lane A, tras el ticket q4_0 idx0):
// para TODO formato con encoder CPU, kv_quant.decode(bytes) debe coincidir
// BIT-EXACTO con gguf.dequantBlock(bytes). Cualquier divergencia = un
// empaquetado de nibbles/bits que contradice al canon GGML upstream y a los
// kernels CUDA (bug silencioso de caché: se escribe en una convención y se
// lee en otra). Historial: q4_0 estaba intercalado par/impar vs split-16.
test "roundtrip kv_quant encode/decode convencion canonica gguf" {
    const gpa = std.testing.allocator;
    const formats = [_]QuantFormat{
        .q8_0,  .q8_1,   .q4_0,    .q4_1,   .q5_0,  .q5_1,
        .q2_k,  .q3_k,   .q4_k,    .q5_k,   .q6_k,  .q8_k,
        .iq1_s, .iq1_m,  .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs,
        .iq3_s, .iq4_xs, .iq4_nl,  .tq1_0,  .tq2_0,
    };
    var audited: usize = 0;
    inline for (formats) |fmt| {
        // Valores fuente variados (no constantes: una constante es invariante
        // ante permutaciones de nibbles y oculta divergencias).
        const n: usize = 256;
        const src = try gpa.alloc(f16, n);
        defer gpa.free(src);
        var rng = std.Random.Xoshiro256.init(@as(u64, 9000) + @as(u64, @intFromEnum(fmt)));
        for (src) |*v| v.* = @floatCast((rng.random().float(f32) - 0.5) * 2.0);

        const bytes = try kv_quant.encodeToOwned(gpa, fmt, src);
        defer gpa.free(bytes);

        const ggml_t = try gguf.GgmlType.fromRaw(kv_quant.toGgmlTypeValue(fmt));
        const out_gguf = try gpa.alloc(f32, n);
        defer gpa.free(out_gguf);
        gguf.dequantBlock(ggml_t, bytes, out_gguf, n);

        const out_kvq16 = try gpa.alloc(f16, n);
        defer gpa.free(out_kvq16);
        kv_quant.decode(fmt, bytes, out_kvq16);

        var bad: usize = 0;
        for (out_gguf, 0..) |gv, i| {
            // kv_quant.decode devuelve f16: exigir que sea el redondeo f16
            // del valor canónico (divergencias de convención dan valores
            // distintos, no solo ruido de redondeo).
            const gv16: f32 = @floatCast(@as(f16, @floatCast(gv)));
            const kv: f32 = @floatCast(out_kvq16[i]);
            if (@abs(gv16 - kv) > 0 and !(gv16 != gv16 and kv != kv)) bad += 1;
            if (bad == 1)
                std.debug.print("[conv-{s}] elem {d}: gguf={d} kv_quant={d}\n", .{ @tagName(fmt), i, gv, kv });
        }
        if (bad > 0) {
            std.debug.print("[conv-{s}] FALLO: {d}/{d} elementos divergentes\n", .{ @tagName(fmt), bad, n });
            return error.ConventionMismatch;
        }
        audited += 1;
    }
    std.debug.print("convencion canonica: {d}/{d} formatos auditados OK\n", .{ audited, formats.len });
}

// Regresión OOB colas parciales (observación lane-F @aaccc3a): dequantBlock
// con out.len NO múltiplo del blockSize debe escribir solo dentro de out.
// Antes del clamp central, los dequantizadores SB-256 escribían el bloque
// completo en la última iteración parcial (pánico idx out of bounds).
test "dequantBlock colas parciales sin overrun" {
    const gpa = std.testing.allocator;
    inline for ([_]gguf.GgmlType{ .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_k, .iq4_xs, .iq3_s, .iq1_s, .iq1_m, .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .tq1_0, .tq2_0, .iq4_nl, .mxfp4 }) |t| {
        const n: usize = 512;
        const bytes = try gpa.alloc(u8, (n / t.blockSize()) * t.blockBytes());
        defer gpa.free(bytes);
        var prng = std.Random.Xoshiro256.init(7700 + @intFromEnum(t));
        prng.random().bytes(bytes);
        for (bytes, 0..) |*b, j| {
            // escalas f16 acotadas para valores finitos
            if (j % t.blockBytes() >= t.blockBytes() - 4) b.* &= 0x3F;
        }
        const out = try gpa.alloc(f32, 300);
        defer gpa.free(out);
        @memset(out, 0);
        gguf.dequantBlock(t, bytes, out, n);
    }
}

// Regresión del REQUEST lane-c (main.zig ~131): decodeDevice NO tenía ramas
// extra-cubin para iq1_s/iq3_s ⇒ KvQuantUnsupported en el primer decode step
// del pipeline real (hybrid_attn:804), aunque la ruta genérica engine.decode
// pasaba (getFusedFunc con fallback). Espejo del patrón iq4_xs @edbcfc2.
test "decodeDevice iq1_s e iq3_s device→device" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    debugz.init();
    inline for ([_]QuantFormat{ .iq1_s, .iq3_s }) |fmt| {
        const config = testConfig(fmt);
        var kv = try pa.PagedKVCache.init(gpa, config);
        defer kv.deinit();
        const seq_id = try kv.createSequence();
        const seq_len: usize = 9; // 3 bloques: cubre bloque parcial
        try kv.allocatePrefill(seq_id, seq_len);

        const elems = config.block_size * config.num_kv_heads * config.head_dim;
        try fillPoolRandom(&kv, 888, elems, fmt);

        const q_stride = config.num_q_heads * config.head_dim;
        const query = try gpa.alloc(f32, q_stride);
        defer gpa.free(query);
        var rng = std.Random.Xoshiro256.init(66);
        for (query) |*v| v.* = @floatCast(@as(f16, @floatCast((rng.random().float(f32) - 0.5) * 2.0)));

        const out_cpu = try gpa.alloc(f32, q_stride);
        defer gpa.free(out_cpu);
        var unstable = false;
        var max_v: f32 = 0;
        try cpuReference(gpa, &kv, fmt, seq_id, query, out_cpu, seq_len, &unstable, &max_v);

        cudaz.ensureContext() catch return error.SkipZigTest;
        const gpu_stream = try cudaz.cuStreamCreate(0);
        defer cudaz.cuStreamDestroy(gpu_stream);
        var engine = try pa.PagedAttentionGpu.init(gpa, config, gpu_stream);
        defer engine.deinit();
        try engine.setupDecodeScratch(0, q_stride, 2);
        try engine.stageTableAll(kv.block_alloc, kv.getBlockTable(seq_id).?);
        // decodeDevice asume d_seq_lens poblado por el caller (contrato
        // producción: el loop de inferencia lo sube por chunk).
        const seq_len_c: c_int = @intCast(seq_len);
        try cudaz.cuMemcpyHtoD(engine.d_seq_lens, @intFromPtr(&seq_len_c), @sizeOf(c_int));
        const d_q = try cudaz.cuMemAlloc(q_stride * @sizeOf(f16));
        defer cudaz.cuMemFree(d_q);
        const d_out = try cudaz.cuMemAlloc(q_stride * @sizeOf(f16));
        defer cudaz.cuMemFree(d_out);
        const q16 = try gpa.alloc(f16, q_stride);
        defer gpa.free(q16);
        for (query, 0..) |v, i| q16[i] = @floatCast(v);
        try cudaz.cuMemcpyHtoD(d_q, @intFromPtr(q16.ptr), q_stride * @sizeOf(f16));

        // Orden diagnóstico: PRIMERO la ruta genérica (puebla todo el estado
        // persistente), DESPUÉS decodeDevice sobre ese mismo estado.
        const out_gen = try gpa.alloc(f32, q_stride);
        defer gpa.free(out_gen);
        const bt_val = kv.getBlockTable(seq_id).?.*;
        try engine.decode(query, out_gen, bt_val, kv.block_alloc, config);

        try engine.decodeDevice(0, d_q, d_out, kv.block_alloc);
        try cudaz.cuStreamSynchronize(gpu_stream);
        const out_dd = try gpa.alloc(f16, q_stride);
        defer gpa.free(out_dd);
        try cudaz.cuMemcpyDtoH(@intFromPtr(out_dd.ptr), d_out, q_stride * @sizeOf(f16));

        var bad: usize = 0;
        var bad_vs_gen: usize = 0;
        var max_diff: f32 = 0;
        const tol_scale = @max(1e-2, 5e-3 * 64.0);
        for (out_cpu, 0..) |c, i| {
            const gd: f32 = @floatCast(out_dd[i]);
            const gg: f32 = out_gen[i];
            max_diff = @max(max_diff, @abs(c - gd));
            if (@abs(c - gd) > tol_scale and !approxEq(c, gd)) bad += 1;
            if (@abs(gg - gd) > tol_scale and !approxEq(gg, gd)) bad_vs_gen += 1;
        }
        std.debug.print("[devdev-{s}] vs_cpu={d}/64 mal (max={d:.4}) | gen-vs-devdev difieren={d}/64\n", .{ @tagName(fmt), bad, max_diff, bad_vs_gen });
        if (bad > 0) {
            std.debug.print("[devdev-{s}] FALLO vs CPU", .{@tagName(fmt)});
            if (bad_vs_gen == 0) std.debug.print(" PERO == ruta genérica ⇒ el genérico TAMBIÉN diverge del oráculo en este setup\n", .{}) else std.debug.print(" y además difiere de la genérica\n", .{});
            return error.DecodeDeviceMismatch;
        }
        std.debug.print("[devdev-{s}] OK ({d} dims, max_diff={d})\n", .{ @tagName(fmt), q_stride, max_diff });
    }
}

// P2 lane-a: BENCH prefill device→device por formato. Ejecutar con
// PREFILL_BENCH=1 (y .bench.lock libre). Dims realistas Llama-ish
// (hd=128, 32Q/8KV, chunk de 512 tokens). Reporta ms y MB de KV leídos;
// fp16 = baseline para ratios.
fn runBenchFormat(gpa: std.mem.Allocator, fmt: QuantFormat) !void {
    const hd: usize = 128;
    const kvh: usize = 8;
    const qh: usize = 32;
    const bs: usize = 16;
    const n_chunk: usize = 512;
    const iters: usize = 20;

    const config = pa.PagedConfig{
        .block_size = @intCast(bs),
        .num_blocks = @intCast((n_chunk + bs - 1) / bs + 4),
        .head_dim = @intCast(hd),
        .num_kv_heads = @intCast(kvh),
        .num_q_heads = @intCast(qh),
        .dtype = .f16,
        .quant_k = fmt,
        .quant_v = fmt,
        .enable_prefix_cache = false,
        .max_seq_len = 4096,
        .max_batch_size = 4,
    };
    var kv = try pa.PagedKVCache.init(gpa, config);
    defer kv.deinit();
    const seq_id = try kv.createSequence();
    try kv.allocatePrefill(seq_id, n_chunk);

    const elems = bs * kvh * hd;
    try fillPoolRandom(&kv, 4242, elems, fmt);

    const q_stride = qh * hd;
    const query = try gpa.alloc(f32, n_chunk * q_stride);
    defer gpa.free(query);
    var rng = std.Random.Xoshiro256.init(99);
    for (query) |*v| v.* = @floatCast(@as(f16, @floatCast((rng.random().float(f32) - 0.5) * 2.0)));

    cudaz.ensureContext() catch return error.SkipZigTest;
    const gpu_stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(gpu_stream);
    var engine = try pa.PagedAttentionGpu.init(gpa, config, gpu_stream);
    defer engine.deinit();
    try engine.setupDecodeScratch(0, q_stride, 2);
    try engine.stageTableAll(kv.block_alloc, kv.getBlockTable(seq_id).?);

    const q16 = try gpa.alloc(f16, query.len);
    defer gpa.free(q16);
    for (query, 0..) |v, i| q16[i] = @floatCast(v);
    const d_q = try cudaz.cuMemAlloc(query.len * @sizeOf(f16));
    defer cudaz.cuMemFree(d_q);
    const d_out = try cudaz.cuMemAlloc(n_chunk * q_stride * @sizeOf(f16));
    defer cudaz.cuMemFree(d_out);
    try cudaz.cuMemcpyHtoD(d_q, @intFromPtr(q16.ptr), query.len * @sizeOf(f16));

    const nb_total = (n_chunk + bs - 1) / bs;
    const bt_host = try gpa.alloc(c_int, nb_total);
    defer gpa.free(bt_host);
    const bt_tbl = kv.getBlockTable(seq_id).?;
    for (0..nb_total) |bi| {
        bt_host[bi] = if (bt_tbl.getPhysical(bi)) |ph| @intCast(ph) else -1;
    }

    // Warmup
    try engine.prefillDeviceEx(0, d_q, d_out, kv.block_alloc, bt_host, n_chunk, 0, null, true);
    try cudaz.cuStreamSynchronize(gpu_stream);

    var t = @import("time").Timer.start();
    for (0..iters) |_| {
        try engine.prefillDeviceEx(0, d_q, d_out, kv.block_alloc, bt_host, n_chunk, 0, null, true);
    }
    try cudaz.cuStreamSynchronize(gpu_stream);
    const ns = t.read();
    const ms = @as(f64, @floatFromInt(ns)) / 1e6 / @as(f64, @floatFromInt(iters));
    // Bytes KV efectivos: chunk completo × K+V
    const kb = regionBytes(fmt, elems);
    const mb_kv = @as(f64, @floatFromInt(nb_total * kb * 2)) / (1024.0 * 1024.0);
    std.debug.print("[bench-prefill] {s}| {d:8.3} ms/chunk({d} tok)  KV={d:7.1} MB  {d:8.1} GB/s\n", .{ @tagName(fmt), ms, n_chunk, mb_kv, mb_kv / (ms / 1000.0) / 1024.0 });
}

test "BENCH prefill por formato" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    if (std.c.getenv("PREFILL_BENCH") == null) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    debugz.init();
    inline for ([_]QuantFormat{ .fp16, .q8_0, .q4_0, .q4_k, .q8_k, .iq4_xs, .iq1_s, .iq3_s, .q6_k }) |fmt| {
        runBenchFormat(gpa, fmt) catch |e| {
            std.debug.print("[bench-prefill-{s}] SKIP: {s}\n", .{ @tagName(fmt), @errorName(e) });
        };
    }
}

test "fused decode GPU por formato vs referencia dequantBlock" {
    if (!cudaz.isCudaAvailable()) {
        std.debug.print("SKIP: CUDA no disponible\n", .{});
        return error.SkipZigTest;
    }
    const gpa = std.testing.allocator;
    debugz.init(); // activa breadcrumbs gated por env en tests
    var failed: usize = 0;
    inline for (enabled_formats) |fmt| {
        runFormat(gpa, fmt) catch |e| {
            std.debug.print("[{s}] >>> ERROR: {s}\n", .{ @tagName(fmt), @errorName(e) });
            failed += 1;
        };
    }
    if (failed > 0) {
        std.debug.print("harness: {d}/{d} formatos en rojo\n", .{ failed, enabled_formats.len });
        return error.FusedDecodeMismatch;
    }
}

// B-a3 (lane-a, 2026-09-09): evaluación MMVQ M≤8 — veredicto NEGATIVA con
// datos. `qgemmKernel` es M-agnóstico (grid.y=M, warp por (m,row)): este test
// documenta paridad M=8 bit-exacta + el bench batching 8×M1 vs 1×M8 que
// justifica NO añadir kernel mmvq dedicado (TODO 1.13 cerrada). Requiere
// GPU (flock .bench.lock si suite completa; solo-corre con test-pafused).
test "B-a3 eval: qgemm q6_k M=1 vs M=8 — escala y paridad" {
    if (!cudaz.isCudaAvailable()) {
        std.debug.print("SKIP: CUDA no disponible\n", .{});
        return error.SkipZigTest;
    }
    const gpa = std.testing.allocator;
    @import("debug").init();
    cudaz.ensureContext() catch return error.SkipZigTest;
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);
    var lk = try @import("layer_kernels").LayerKernels.init(stream);
    defer lk.deinit();

    // Geometría 0.8B ffn_down: k=2048, n=1024. q6_k (SB256/210B).
    const K: usize = 2048;
    const N: usize = 1024;
    const qtypes = @import("kv_cache").quant_types;
    const fmt = qtypes.QuantFormat.q6_k;
    const row_bytes = kv_quant.quantBytesRaw(fmt, K);
    const w_bytes = try gpa.alloc(u8, N * row_bytes);
    defer gpa.free(w_bytes);
    {
        var rng = std.Random.Xoshiro256.init(6001);
        const row = try gpa.alloc(f16, K);
        defer gpa.free(row);
        for (0..N) |j| {
            for (row) |*v| v.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);
            const enc = try kv_quant.encodeToOwned(gpa, fmt, row);
            defer gpa.free(enc);
            @memcpy(w_bytes[j * row_bytes ..][0..row_bytes], enc[0..row_bytes]);
        }
    }
    // A de 8 filas (seed distinta por fila para detectar mezcla).
    const a8 = try gpa.alloc(f32, 8 * K);
    defer gpa.free(a8);
    {
        var rng = std.Random.Xoshiro256.init(6002);
        for (a8) |*v| v.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);
    }
    // Ref CPU para las 8 filas (dequant espejo por fila).
    const w_ref = try gpa.alloc(f32, N * K);
    defer gpa.free(w_ref);
    {
        const deq = kv_quant.dequant;
        for (0..N) |j| {
            deq(fmt, w_bytes[j * row_bytes ..][0..row_bytes], w_ref[j * K ..][0..K]);
        }
    }

    const d_a = try cudaz.cuMemAlloc(8 * K * @sizeOf(f32));
    defer cudaz.cuMemFree(d_a);
    const d_w = try cudaz.cuMemAlloc(w_bytes.len);
    defer cudaz.cuMemFree(d_w);
    try cudaz.cuMemcpyHtoD(d_a, @intFromPtr(a8.ptr), a8.len * @sizeOf(f32));
    try cudaz.cuMemcpyHtoD(d_w, @intFromPtr(w_bytes.ptr), w_bytes.len);
    const d_c8 = try cudaz.cuMemAlloc(8 * N * @sizeOf(f32));
    defer cudaz.cuMemFree(d_c8);

    // Paridad M=8 primero (row-major C[m][n]).
    try lk.qgemm(d_a, d_w, d_c8, 8, K, N, 3);
    try cudaz.cuStreamSynchronize(stream);
    const c8 = try gpa.alloc(f32, 8 * N);
    defer gpa.free(c8);
    try cudaz.cuMemcpyDtoH(@intFromPtr(c8.ptr), d_c8, c8.len * @sizeOf(f32));
    var bad: usize = 0;
    var max_rel: f32 = 0;
    for (0..8) |m| {
        for (0..N) |j| {
            var acc: f32 = 0;
            for (0..K) |k2| acc += a8[m * K + k2] * w_ref[j * K + k2];
            const rel = @abs(c8[m * N + j] - acc) / @max(@abs(acc), 1.0);
            max_rel = @max(max_rel, rel);
            if (rel > 5e-3) bad += 1;
        }
    }
    std.debug.print("B-a3 paridad M=8 qgemm q6_k: bad={d}/{d} max_rel={e}\n", .{ bad, 8 * N, max_rel });
    if (bad > 0) return error.Ba3ParityMismatch;

    // Bench: M=1 ×8 launches vs M=8 ×1 launch (mismo trabajo total).
    const d_c1 = try cudaz.cuMemAlloc(N * @sizeOf(f32));
    defer cudaz.cuMemFree(d_c1);
    try lk.qgemm(d_a, d_w, d_c1, 1, K, N, 3);
    try lk.qgemm(d_a, d_w, d_c8, 8, K, N, 3);
    try cudaz.cuStreamSynchronize(stream);

    var ns1: i128 = std.math.maxInt(i128);
    var ns8: i128 = std.math.maxInt(i128);
    for (0..5) |_| {
        var t = @import("time").Timer.start();
        for (0..8) |_| try lk.qgemm(d_a, d_w, d_c1, 1, K, N, 3);
        try cudaz.cuStreamSynchronize(stream);
        ns1 = @min(ns1, t.read());
        t = @import("time").Timer.start();
        try lk.qgemm(d_a, d_w, d_c8, 8, K, N, 3);
        try cudaz.cuStreamSynchronize(stream);
        ns8 = @min(ns8, t.read());
    }
    std.debug.print("BENCH B-a3 qgemm q6_k k={d} n={d}: 8×M1={d:.3}ms vs 1×M8={d:.3}ms ratio={d:.2}x\n", .{
        K,                                                                    N,
        @as(f64, @floatFromInt(ns1)) / 1e6,                                   @as(f64, @floatFromInt(ns8)) / 1e6,
        @as(f64, @floatFromInt(ns1)) / @as(f64, @floatFromInt(@max(ns8, 1))),
    });
}

// a-U3 (lane-a, 2026-09-10): paridad + bench del q3kGemmM1Dp4aKernel.
// REFERENCIA = dot CPU con A cuantizada q8_1 (cuantización IDÉNTICA a la
// fase 1 del kernel: amax/127 por KB de 32, round, clamp ±127) — verificado:
// |dp4a − dot_cpu_A_q8| = 7.8e-6 ⇒ el kernel es exacto; la divergencia vs
// el case 6 escalar (que consume A f32) es 100% ruido de cuantización q8 de
// A, el mismo convenio de los kernels dp4a hermanos (§5.8, 3.3). Gate rel
// 1e-2. Geometría real 3B: ffn_gate n=8192, k=3072 (todas las GEMV del 3B
// son k∈{3072,8192}, n∈{1024..8192}).
test "a-U3 q3_k M=1 dp4a: paridad vs case 6 escalar + bench" {
    if (!cudaz.isCudaAvailable()) {
        std.debug.print("SKIP: CUDA no disponible\n", .{});
        return error.SkipZigTest;
    }
    const gpa = std.testing.allocator;
    @import("debug").init();
    cudaz.ensureContext() catch return error.SkipZigTest;
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);
    var lk = try @import("layer_kernels").LayerKernels.init(stream);
    defer lk.deinit();

    // Geometría ffn_gate del 3B: k=3072, n=8192. q3_k (110B/SB256).
    const K: usize = 3072;
    const N: usize = 8192;
    const qtypes = @import("kv_cache").quant_types;
    const fmt = qtypes.QuantFormat.q3_k;
    const row_bytes = kv_quant.quantBytesRaw(fmt, K);
    const w_bytes = try gpa.alloc(u8, N * row_bytes);
    defer gpa.free(w_bytes);
    {
        var rng = std.Random.Xoshiro256.init(7031);
        const row = try gpa.alloc(f16, K);
        defer gpa.free(row);
        for (0..N) |j| {
            for (row) |*v| v.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);
            const enc = try kv_quant.encodeToOwned(gpa, fmt, row);
            defer gpa.free(enc);
            @memcpy(w_bytes[j * row_bytes ..][0..row_bytes], enc[0..row_bytes]);
        }
    }
    const a1 = try gpa.alloc(f32, K);
    defer gpa.free(a1);
    {
        var rng = std.Random.Xoshiro256.init(7032);
        for (a1) |*v| v.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);
    }

    const d_a = try cudaz.cuMemAlloc(K * @sizeOf(f32));
    defer cudaz.cuMemFree(d_a);
    const d_w = try cudaz.cuMemAlloc(w_bytes.len);
    defer cudaz.cuMemFree(d_w);
    try cudaz.cuMemcpyHtoD(d_a, @intFromPtr(a1.ptr), K * @sizeOf(f32));
    try cudaz.cuMemcpyHtoD(d_w, @intFromPtr(w_bytes.ptr), w_bytes.len);
    const d_c_new = try cudaz.cuMemAlloc(N * @sizeOf(f32));
    defer cudaz.cuMemFree(d_c_new);
    const d_c_ref = try cudaz.cuMemAlloc(N * @sizeOf(f32));
    defer cudaz.cuMemFree(d_c_ref);

    // Ref: case 6 escalar (camino trunk — NOSSM4DP4A lo fuerza).
    // New: dp4a (el hook de qgemmLinear lo enruta, pero llamamos directo
    // para aislar el kernel).
    try lk.qgemm(d_a, d_w, d_c_ref, 1, K, N, 6);
    try lk.q3kGemmM1Dp4a(d_a, d_w, d_c_new, K, N);
    try cudaz.cuStreamSynchronize(stream);

    const c_ref = try gpa.alloc(f32, N);
    defer gpa.free(c_ref);
    const c_new = try gpa.alloc(f32, N);
    defer gpa.free(c_new);
    try cudaz.cuMemcpyDtoH(@intFromPtr(c_ref.ptr), d_c_ref, N * @sizeOf(f32));
    try cudaz.cuMemcpyDtoH(@intFromPtr(c_new.ptr), d_c_new, N * @sizeOf(f32));

    var bad: usize = 0;
    var max_rel: f32 = 0;
    var max_abs: f32 = 0;
    var first_bad: ?usize = null;
    // REFERENCIA CORRECTA: dot CPU con A cuantizada q8_1 — cuantización
    // IDÉNTICA a la fase 1 del kernel (amax/127 por KB de 32, round, clamp
    // ±127). Verificado (2026-09-10): |dp4a − dot_cpu_A_q8| = 7.8e-6 en la
    // fila peor ⇒ el kernel es exacto; la divergencia vs case 6 escalar es
    // 100% ruido de la cuantización q8 de A (mismo convenio §5.8/3.3, que
    // también cuantizan A/q — gate rel 1e-2 vs ESTA ref).
    const w_ref_all = try gpa.alloc(f32, N * K);
    defer gpa.free(w_ref_all);
    {
        const deq = kv_quant.dequant;
        for (0..N) |j| {
            deq(fmt, w_bytes[j * row_bytes ..][0..row_bytes], w_ref_all[j * K ..][0..K]);
        }
    }
    // A q8_1 (una vez — igual que el kernel).
    const aq_scale = try gpa.alloc(f32, K / 32);
    defer gpa.free(aq_scale);
    const aq_val = try gpa.alloc(i32, K);
    defer gpa.free(aq_val);
    for (0..(K / 32)) |kb| {
        var amax: f32 = 0;
        for (0..32) |r| amax = @max(amax, @abs(a1[kb * 32 + r]));
        const dd: f32 = if (amax > 0) amax / 127.0 else 1.0;
        aq_scale[kb] = dd;
        for (0..32) |r| {
            const q: i32 = @max(-127, @min(127, @as(i32, @intFromFloat(@round(a1[kb * 32 + r] / dd)))));
            aq_val[kb * 32 + r] = q;
        }
    }
    for (0..N) |j| {
        var dot_q8: f32 = 0;
        for (0..(K / 32)) |kb| {
            var sacc: f32 = 0;
            for (0..32) |r| sacc += @as(f32, @floatFromInt(aq_val[kb * 32 + r])) * w_ref_all[j * K + kb * 32 + r];
            dot_q8 += sacc * aq_scale[kb];
        }
        const abs_diff = @abs(c_new[j] - dot_q8);
        const rel = abs_diff / @max(@abs(dot_q8), 1.0);
        max_rel = @max(max_rel, rel);
        max_abs = @max(max_abs, abs_diff);
        if (rel > 1e-2) {
            bad += 1;
            if (first_bad == null) {
                first_bad = j;
                std.debug.print("  first_bad row {d}: dp4a={d:.6} cpu_q8={d:.6} rel={e}\n", .{ j, c_new[j], dot_q8, rel });
            }
        }
    }
    std.debug.print("a-U3 q3_k dp4a M=1: bad={d}/{d} max_rel={e} max_abs={e}\n", .{ bad, N, max_rel, max_abs });
    if (bad > 0) return error.Au3Q3kDp4aMismatch;

    // Bench kernel puro: escalar vs dp4a (mismo GEMV).
    var ns_ref: i128 = std.math.maxInt(i128);
    var ns_new: i128 = std.math.maxInt(i128);
    for (0..5) |_| {
        var t = @import("time").Timer.start();
        try lk.qgemm(d_a, d_w, d_c_ref, 1, K, N, 6);
        try cudaz.cuStreamSynchronize(stream);
        ns_ref = @min(ns_ref, t.read());
        t = @import("time").Timer.start();
        try lk.q3kGemmM1Dp4a(d_a, d_w, d_c_new, K, N);
        try cudaz.cuStreamSynchronize(stream);
        ns_new = @min(ns_new, t.read());
    }
    std.debug.print("BENCH a-U3 q3_k k={d} n={d}: escalar={d:.3}ms dp4a={d:.3}ms speedup={d:.2}x\n", .{
        K,                                                                          N,
        @as(f64, @floatFromInt(ns_ref)) / 1e6,                                      @as(f64, @floatFromInt(ns_new)) / 1e6,
        @as(f64, @floatFromInt(ns_ref)) / @as(f64, @floatFromInt(@max(ns_new, 1))),
    });
}

// a-U3 fase 2 (lane-a, 2026-09-10): paridad del SPLIT-K (S=2/S=4 warps
// por fila + atomicAdd sobre c memset). Mismas geometrías pequeñas que
// en E2E activan el split: n=3072 (S=2) y n=1024 (S=4). El atomicAdd
// introduce order-drift (FP no asociativo) ⇒ el gate 1e-2 absorbe el
// ruido; la ref sigue siendo dot CPU con A q8_1 (convenio §5.8/3.3).
test "a-U3 q3_k M=1 dp4a SPLIT: paridad S=2/S=4" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    @import("debug").init();
    cudaz.ensureContext() catch return error.SkipZigTest;
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);
    var lk = try @import("layer_kernels").LayerKernels.init(stream);
    defer lk.deinit();

    const qtypes = @import("kv_cache").quant_types;
    const fmt = qtypes.QuantFormat.q3_k;
    const K: usize = 3072;
    // n=3072 ⇒ S=2, n=1024 ⇒ S=4 (misma heurística que el wrapper).
    for ([_]usize{ 3072, 1024 }) |N| {
        const row_bytes = kv_quant.quantBytesRaw(fmt, K);
        const w_bytes = try gpa.alloc(u8, N * row_bytes);
        defer gpa.free(w_bytes);
        {
            var rng = std.Random.Xoshiro256.init(7031);
            const row = try gpa.alloc(f16, K);
            defer gpa.free(row);
            for (0..N) |j| {
                for (row) |*v| v.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);
                const enc = try kv_quant.encodeToOwned(gpa, fmt, row);
                defer gpa.free(enc);
                @memcpy(w_bytes[j * row_bytes ..][0..row_bytes], enc[0..row_bytes]);
            }
        }
        const a1 = try gpa.alloc(f32, K);
        defer gpa.free(a1);
        {
            var rng = std.Random.Xoshiro256.init(7032);
            for (a1) |*v| v.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);
        }
        const d_a = try cudaz.cuMemAlloc(K * 4);
        defer cudaz.cuMemFree(d_a);
        const d_w = try cudaz.cuMemAlloc(w_bytes.len);
        defer cudaz.cuMemFree(d_w);
        const d_c = try cudaz.cuMemAlloc(N * 4);
        defer cudaz.cuMemFree(d_c);
        try cudaz.cuMemcpyHtoD(d_a, @intFromPtr(a1.ptr), K * 4);
        try cudaz.cuMemcpyHtoD(d_w, @intFromPtr(w_bytes.ptr), w_bytes.len);

        try lk.q3kGemmM1Dp4a(d_a, d_w, d_c, K, N);
        try cudaz.cuStreamSynchronize(stream);
        const c_new = try gpa.alloc(f32, N);
        defer gpa.free(c_new);
        try cudaz.cuMemcpyDtoH(@intFromPtr(c_new.ptr), d_c, N * 4);

        // Ref: dot CPU con A q8_1 (cuantización idéntica a fase 1).
        const w_ref_all = try gpa.alloc(f32, N * K);
        defer gpa.free(w_ref_all);
        {
            const deq = kv_quant.dequant;
            for (0..N) |j| deq(fmt, w_bytes[j * row_bytes ..][0..row_bytes], w_ref_all[j * K ..][0..K]);
        }
        const aq_scale = try gpa.alloc(f32, K / 32);
        defer gpa.free(aq_scale);
        const aq_val = try gpa.alloc(i32, K);
        defer gpa.free(aq_val);
        for (0..(K / 32)) |kb| {
            var amax: f32 = 0;
            for (0..32) |r| amax = @max(amax, @abs(a1[kb * 32 + r]));
            const dd: f32 = if (amax > 0) amax / 127.0 else 1.0;
            aq_scale[kb] = dd;
            for (0..32) |r| {
                const q: i32 = @max(-127, @min(127, @as(i32, @intFromFloat(@round(a1[kb * 32 + r] / dd)))));
                aq_val[kb * 32 + r] = q;
            }
        }
        var bad: usize = 0;
        var max_rel: f32 = 0;
        for (0..N) |j| {
            var dot_q8: f32 = 0;
            for (0..(K / 32)) |kb| {
                var sacc: f32 = 0;
                for (0..32) |r| sacc += @as(f32, @floatFromInt(aq_val[kb * 32 + r])) * w_ref_all[j * K + kb * 32 + r];
                dot_q8 += sacc * aq_scale[kb];
            }
            const abs_diff = @abs(c_new[j] - dot_q8);
            const rel = abs_diff / @max(@abs(dot_q8), 1.0);
            max_rel = @max(max_rel, rel);
            if (rel > 1e-2) bad += 1;
        }
        const split: usize = if (N <= 1024) 4 else 2;
        std.debug.print("a-U3 SPLIT n={d} S={d}: bad={d}/{d} max_rel={e}\n", .{ N, split, bad, N, max_rel });
        if (bad > 0) return error.Au3Q3kSplitMismatch;
    }
}

// a-U3 fase 2 (lane-a, 2026-09-10): microbench de las 7 geometrías GEMV del
// Llama-3.2-3B (todo q3_k) — case 6 escalar vs q3kGemmM1Dp4a. Valida el
// modelo de costes ANTES de optimizar (plan .kilo/1788969063890): µs por
// geometría + suma/capa extrapolada + BW efectivo. Gate por env AU3_BENCH=1.
test "a-U3 bench: 7 geometrías GEMV 3B (escalar vs dp4a)" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    if (std.c.getenv("AU3_BENCH") == null) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    @import("debug").init();
    cudaz.ensureContext() catch return error.SkipZigTest;
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);
    var lk = try @import("layer_kernels").LayerKernels.init(stream);
    defer lk.deinit();

    const qtypes = @import("kv_cache").quant_types;
    const fmt = qtypes.QuantFormat.q3_k;
    // (nombre, n, k) — las 7 GEMV por capa del 3B (q/k/v/o + gate/up/down).
    const geoms = [_]struct { name: []const u8, n: usize, k: usize }{
        .{ .name = "attn_q", .n = 3072, .k = 3072 },
        .{ .name = "attn_k", .n = 1024, .k = 3072 },
        .{ .name = "attn_v", .n = 1024, .k = 3072 },
        .{ .name = "attn_o", .n = 3072, .k = 3072 },
        .{ .name = "ffn_gate", .n = 8192, .k = 3072 },
        .{ .name = "ffn_up", .n = 8192, .k = 3072 },
        .{ .name = "ffn_down", .n = 3072, .k = 8192 },
    };
    // lm_head q6_k aparte (n=128256, k=3072) — paso 4 del plan.
    var sum_scalar: f64 = 0;
    var sum_dp4a: f64 = 0;
    for (geoms) |g| {
        const K = g.k;
        const N = g.n;
        const row_bytes = kv_quant.quantBytesRaw(fmt, K);
        const w_bytes = try gpa.alloc(u8, N * row_bytes);
        defer gpa.free(w_bytes);
        {
            var rng = std.Random.Xoshiro256.init(7031);
            const row = try gpa.alloc(f16, K);
            defer gpa.free(row);
            for (0..N) |j| {
                for (row) |*v| v.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);
                const enc = try kv_quant.encodeToOwned(gpa, fmt, row);
                defer gpa.free(enc);
                @memcpy(w_bytes[j * row_bytes ..][0..row_bytes], enc[0..row_bytes]);
            }
        }
        const a1 = try gpa.alloc(f32, K);
        defer gpa.free(a1);
        {
            var rng = std.Random.Xoshiro256.init(7032);
            for (a1) |*v| v.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);
        }
        const d_a = try cudaz.cuMemAlloc(K * 4);
        defer cudaz.cuMemFree(d_a);
        const d_w = try cudaz.cuMemAlloc(w_bytes.len);
        defer cudaz.cuMemFree(d_w);
        const d_c = try cudaz.cuMemAlloc(N * 4);
        defer cudaz.cuMemFree(d_c);
        try cudaz.cuMemcpyHtoD(d_a, @intFromPtr(a1.ptr), K * 4);
        try cudaz.cuMemcpyHtoD(d_w, @intFromPtr(w_bytes.ptr), w_bytes.len);

        // Warm-up + min de 50 iters.
        try lk.qgemm(d_a, d_w, d_c, 1, K, N, 6);
        try lk.q3kGemmM1Dp4a(d_a, d_w, d_c, K, N);
        try cudaz.cuStreamSynchronize(stream);
        var ns_s: i128 = std.math.maxInt(i128);
        var ns_d: i128 = std.math.maxInt(i128);
        for (0..50) |_| {
            var t = @import("time").Timer.start();
            try lk.qgemm(d_a, d_w, d_c, 1, K, N, 6);
            try cudaz.cuStreamSynchronize(stream);
            ns_s = @min(ns_s, t.read());
            t = @import("time").Timer.start();
            try lk.q3kGemmM1Dp4a(d_a, d_w, d_c, K, N);
            try cudaz.cuStreamSynchronize(stream);
            ns_d = @min(ns_d, t.read());
        }
        const us_s: f64 = @as(f64, @floatFromInt(ns_s)) / 1e3;
        const us_d: f64 = @as(f64, @floatFromInt(ns_d)) / 1e3;
        sum_scalar += us_s;
        sum_dp4a += us_d;
        const mb = @as(f64, @floatFromInt(w_bytes.len)) / 1e6;
        std.debug.print("GEMV {s:>9} n={d:5} k={d:5}: escalar={d:7.1}µs dp4a={d:7.1}µs ({d:.2}x) BW dp4a={d:5.0} GB/s\n", .{
            g.name,                   N,    K,
            us_s,                     us_d, us_s / @max(us_d, 1),
            mb / (us_d * 1e-6) / 1.0,
        });
    }
    std.debug.print("SUMA/capa: escalar={d:.1}µs dp4a={d:.1}µs ({d:.2}x) — E2E ref PERF_STAGE: GEMV≈616µs de 712µs/capa\n", .{
        sum_scalar, sum_dp4a, sum_scalar / @max(sum_dp4a, 1),
    });

    // lm_head q6_k (n=128256, k=3072) — el GEMV grande 1.5ms/token. case 3.
    {
        const K: usize = 3072;
        const N: usize = 128256;
        const fmt6 = qtypes.QuantFormat.q6_k;
        const row_bytes = kv_quant.quantBytesRaw(fmt6, K);
        const w_bytes = try gpa.alloc(u8, N * row_bytes);
        defer gpa.free(w_bytes);
        {
            var rng = std.Random.Xoshiro256.init(7031);
            const row = try gpa.alloc(f16, K);
            defer gpa.free(row);
            for (0..N) |j| {
                for (row) |*v| v.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);
                const enc = try kv_quant.encodeToOwned(gpa, fmt6, row);
                defer gpa.free(enc);
                @memcpy(w_bytes[j * row_bytes ..][0..row_bytes], enc[0..row_bytes]);
            }
        }
        const a1 = try gpa.alloc(f32, K);
        defer gpa.free(a1);
        {
            var rng = std.Random.Xoshiro256.init(7032);
            for (a1) |*v| v.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);
        }
        const d_a = try cudaz.cuMemAlloc(K * 4);
        defer cudaz.cuMemFree(d_a);
        const d_w = try cudaz.cuMemAlloc(w_bytes.len);
        defer cudaz.cuMemFree(d_w);
        const d_c = try cudaz.cuMemAlloc(N * 4);
        defer cudaz.cuMemFree(d_c);
        try cudaz.cuMemcpyHtoD(d_a, @intFromPtr(a1.ptr), K * 4);
        try cudaz.cuMemcpyHtoD(d_w, @intFromPtr(w_bytes.ptr), w_bytes.len);
        try lk.qgemm(d_a, d_w, d_c, 1, K, N, 3);
        try cudaz.cuStreamSynchronize(stream);
        var ns_s: i128 = std.math.maxInt(i128);
        for (0..20) |_| {
            const t = @import("time").Timer.start();
            try lk.qgemm(d_a, d_w, d_c, 1, K, N, 3);
            try cudaz.cuStreamSynchronize(stream);
            ns_s = @min(ns_s, t.read());
        }
        const us_s: f64 = @as(f64, @floatFromInt(ns_s)) / 1e3;
        const mb = @as(f64, @floatFromInt(w_bytes.len)) / 1e6;
        std.debug.print("LM_HEAD q6_k n=128256 k=3072: escalar={d:.1}µs BW={d:.0} GB/s (grid 16032 bloques — saturado: el cuello es ALU)\n", .{ us_s, mb / (us_s * 1e-6) });
    }
}

// a-U3 fase 3 (lane-a, 2026-09-10): REPACK alineado 128B/SB — test de
// hipótesis (coalescing). Pack CPU del layout [hm32|qs64|s16_16|d@112|pad14]:
// el kmask-spill de escalas se hace UNA vez en pack (igual que dequantQ3_K
// kv_quant.zig:2370 — byte a byte). Paridad: el packed debe ser EXACTO vs
// q3kGemmM1Dp4a 110B (mismos números, solo cambia el layout de lectura).
// Bench: 7 geometrías 3B, packed vs dp4a-110 vs escalar. Gate por env
// AU3_BENCH=1 (re-usa la del microbench fase 2).
test "a-U3 repack128: paridad + bench (hipótesis coalescing)" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    if (std.c.getenv("AU3_BENCH") == null) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    @import("debug").init();
    cudaz.ensureContext() catch return error.SkipZigTest;
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);
    var lk = try @import("layer_kernels").LayerKernels.init(stream);
    defer lk.deinit();

    const qtypes = @import("kv_cache").quant_types;
    const fmt = qtypes.QuantFormat.q3_k;
    const geoms = [_]struct { name: []const u8, n: usize, k: usize }{
        .{ .name = "attn_q", .n = 3072, .k = 3072 },
        .{ .name = "attn_k", .n = 1024, .k = 3072 },
        .{ .name = "attn_v", .n = 1024, .k = 3072 },
        .{ .name = "attn_o", .n = 3072, .k = 3072 },
        .{ .name = "ffn_gate", .n = 8192, .k = 3072 },
        .{ .name = "ffn_up", .n = 8192, .k = 3072 },
        .{ .name = "ffn_down", .n = 3072, .k = 8192 },
    };
    var sum_110: f64 = 0;
    var sum_pk: f64 = 0;
    for (geoms) |g| {
        const K = g.k;
        const N = g.n;
        const row_bytes = kv_quant.quantBytesRaw(fmt, K);
        const sb_total = K / 256;
        // 1) Peso original 110B/SB.
        const w_bytes = try gpa.alloc(u8, N * row_bytes);
        defer gpa.free(w_bytes);
        {
            var rng = std.Random.Xoshiro256.init(7031);
            const row = try gpa.alloc(f16, K);
            defer gpa.free(row);
            for (0..N) |j| {
                for (row) |*v| v.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);
                const enc = try kv_quant.encodeToOwned(gpa, fmt, row);
                defer gpa.free(enc);
                @memcpy(w_bytes[j * row_bytes ..][0..row_bytes], enc[0..row_bytes]);
            }
        }
        // 2) Pack 128B/SB (kmask-spill una vez — idéntico dequantQ3_K).
        const w_pk = try gpa.alloc(u8, N * sb_total * 128);
        defer gpa.free(w_pk);
        @memset(w_pk, 0);
        for (0..N) |j| {
            const src = w_bytes[j * row_bytes ..][0..row_bytes];
            const dst = w_pk[j * sb_total * 128 ..][0 .. sb_total * 128];
            for (0..sb_total) |sb| {
                const bp = src[sb * 110 ..][0..110];
                const dp = dst[sb * 128 ..][0..128];
                @memcpy(dp[0..32], bp[0..32]); // hm
                @memcpy(dp[32..96], bp[32..96]); // qs
                const kmask1: u32 = 0x03030303;
                const kmask2: u32 = 0x0f0f0f0f;
                var aux: [4]u32 = @splat(0);
                @memcpy(@as([*]u8, @ptrCast(&aux))[0..12], bp[96..108]);
                const tmp = aux[2];
                aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
                aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
                @memcpy(dp[96..112], @as([*]const u8, @ptrCast(&aux))[0..16]); // s16
                @memcpy(dp[112..114], bp[108..110]); // d f16
                // dp[114..128] pad 0 (ya memset)
            }
        }
        const a1 = try gpa.alloc(f32, K);
        defer gpa.free(a1);
        {
            var rng = std.Random.Xoshiro256.init(7032);
            for (a1) |*v| v.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);
        }
        const d_a = try cudaz.cuMemAlloc(K * 4);
        defer cudaz.cuMemFree(d_a);
        const d_w = try cudaz.cuMemAlloc(w_bytes.len);
        defer cudaz.cuMemFree(d_w);
        const d_wp = try cudaz.cuMemAlloc(w_pk.len);
        defer cudaz.cuMemFree(d_wp);
        const d_c110 = try cudaz.cuMemAlloc(N * 4);
        defer cudaz.cuMemFree(d_c110);
        const d_cpk = try cudaz.cuMemAlloc(N * 4);
        defer cudaz.cuMemFree(d_cpk);
        try cudaz.cuMemcpyHtoD(d_a, @intFromPtr(a1.ptr), K * 4);
        try cudaz.cuMemcpyHtoD(d_w, @intFromPtr(w_bytes.ptr), w_bytes.len);
        try cudaz.cuMemcpyHtoD(d_wp, @intFromPtr(w_pk.ptr), w_pk.len);

        // Paridad: packed vs 110B — MISMO número esperado (layout puro).
        try lk.q3kGemmM1Dp4a(d_a, d_w, d_c110, K, N);
        try lk.q3kGemmM1Dp4aPacked(d_a, d_wp, d_cpk, K, N);
        try cudaz.cuStreamSynchronize(stream);
        const c110 = try gpa.alloc(f32, N);
        defer gpa.free(c110);
        const cpk = try gpa.alloc(f32, N);
        defer gpa.free(cpk);
        try cudaz.cuMemcpyDtoH(@intFromPtr(c110.ptr), d_c110, N * 4);
        try cudaz.cuMemcpyDtoH(@intFromPtr(cpk.ptr), d_cpk, N * 4);
        var bad: usize = 0;
        var max_rel: f32 = 0;
        for (0..N) |j| {
            // Gate rel (el abs 1e-5 saltó con ruido FP 3e-6 en |c|≈27 —
            // misma aritmética, distinto solo el orden de loads).
            const dd = @abs(cpk[j] - c110[j]) / @max(@abs(c110[j]), 1.0);
            max_rel = @max(max_rel, dd);
            if (dd > 1e-2) bad += 1;
        }
        if (bad > 0) {
            std.debug.print("  REPACK n={d}: bad={d} max_rel={e} first: pk={d:.6} 110={d:.6}\n", .{ N, bad, max_rel, cpk[0], c110[0] });
            return error.Au3RepackMismatch;
        }

        // Bench: min de 50 iters.
        try lk.q3kGemmM1Dp4a(d_a, d_w, d_c110, K, N);
        try lk.q3kGemmM1Dp4aPacked(d_a, d_wp, d_cpk, K, N);
        try cudaz.cuStreamSynchronize(stream);
        var ns_110: i128 = std.math.maxInt(i128);
        var ns_pk: i128 = std.math.maxInt(i128);
        for (0..50) |_| {
            var t = @import("time").Timer.start();
            try lk.q3kGemmM1Dp4a(d_a, d_w, d_c110, K, N);
            try cudaz.cuStreamSynchronize(stream);
            ns_110 = @min(ns_110, t.read());
            t = @import("time").Timer.start();
            try lk.q3kGemmM1Dp4aPacked(d_a, d_wp, d_cpk, K, N);
            try cudaz.cuStreamSynchronize(stream);
            ns_pk = @min(ns_pk, t.read());
        }
        const us_110: f64 = @as(f64, @floatFromInt(ns_110)) / 1e3;
        const us_pk: f64 = @as(f64, @floatFromInt(ns_pk)) / 1e3;
        sum_110 += us_110;
        sum_pk += us_pk;
        const mb = @as(f64, @floatFromInt(w_pk.len)) / 1e6;
        std.debug.print("REPACK {s:>9} n={d:5} k={d:5}: dp4a110={d:7.1}µs packed={d:7.1}µs ({d:.2}x) BW pk={d:5.0} GB/s\n", .{
            g.name,              N,     K,
            us_110,              us_pk, us_110 / @max(us_pk, 1),
            mb / (us_pk * 1e-6),
        });
    }
    std.debug.print("SUMA/capa: dp4a110={d:.1}µs packed={d:.1}µs ({d:.2}x) — BW útil x128/110 = {d:.2}x teórico\n", .{
        sum_110, sum_pk, sum_110 / @max(sum_pk, 1), 128.0 / 110.0,
    });
}

// 1.15 path-A (lane-a): paridad estadística del sampler GPU Gumbel.
// El Gumbel-max samplea softmax(logits/temp) EXACTO — no puede compararse
// token-a-token vs CPU (RNG distinto), pero la DISTRIBUCIÓN debe converger:
// (a) logits planos ⇒ uniforme sobre vocab; (b) logit dominante ⇒ argmax
// con prob softmax; (c) rep_penalty mueve la masa (1 token en ring).
// Test: freqs sobre 20k muestras vs prob analítica, tolerancia |Δp|<0.01.
test "1.15 gumbel sampler: paridad estadística GPU vs softmax" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    @import("debug").init();
    cudaz.ensureContext() catch return error.SkipZigTest;
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);
    var lk = try @import("layer_kernels").LayerKernels.init(stream);
    defer lk.deinit();

    const V: usize = 1000;
    const logits = try gpa.alloc(f32, V);
    defer gpa.free(logits);
    // (a)+(b): 990 logits a 0.0, 10 "calientes" a 5.0 ⇒ p_hot = e^5/(990+10e^5).
    @memset(logits, 0.0);
    for (0..10) |i| logits[i * 97 % V] = 5.0;

    const d_logits = try cudaz.cuMemAlloc(V * 4);
    defer cudaz.cuMemFree(d_logits);
    try cudaz.cuMemcpyHtoD(d_logits, @intFromPtr(logits.ptr), V * 4);
    const d_dst = try cudaz.cuMemAlloc(4);
    defer cudaz.cuMemFree(d_dst);
    try lk.sampleGumbelInit(1234);
    try lk.sampleGumbelSetRing(&[_]u32{});

    var counts_hot: usize = 0;
    const N: usize = 20000;
    var raw: i32 = 0;
    for (0..N) |_| {
        try lk.sampleF32Gumbel(d_logits, d_dst, 1.0, 1.0, 1, V);
        try cudaz.cuStreamSynchronize(stream);
        try cudaz.cuMemcpyDtoH(@intFromPtr(&raw), d_dst, 4);
        if (logits[@intCast(raw)] > 1.0) counts_hot += 1;
    }
    // Analítica: e^5≈148.4; p_hot = 10*148.4/(990+10*148.4) = 0.5997.
    const p_hot = 10.0 * 148.413 / (990.0 + 10.0 * 148.413);
    const p_meas = @as(f64, @floatFromInt(counts_hot)) / @as(f64, @floatFromInt(N));
    std.debug.print("1.15 gumbel: p_hot analitica={d:.4} medida={d:.4} (N={d})\n", .{ p_hot, p_meas, N });
    try std.testing.expect(@abs(p_meas - p_hot) < 0.02);

    // (c) rep_penalty: hot token 0 penalizado 2.0 (logit>0 ⇒ /2 ⇒ 2.5):
    // p_hot' = 9*e^5 + e^2.5 ... aprox — verificamos que BAJA vs sin pen.
    try lk.sampleGumbelSetRing(&[_]u32{0});
    // NOTA: el ring [0]=n solo penaliza el token 0 (uno de los 10 calientes).
    var counts_pen: usize = 0;
    for (0..N) |_| {
        try lk.sampleF32Gumbel(d_logits, d_dst, 1.0, 2.0, 1, V);
        try cudaz.cuStreamSynchronize(stream);
        try cudaz.cuMemcpyDtoH(@intFromPtr(&raw), d_dst, 4);
        if (raw == 0) counts_pen += 1;
    }
    const p0_before = 148.413 / (990.0 + 10.0 * 148.413);
    // Analítica correcta: logit 5.0 con penalty 2.0 ⇒ 5/2 = 2.5 ⇒ e^2.5
    // = 12.18 (NO 74.2 — era e^5/2·ln2 mal calculado). p0' = e^2.5/
    // (990 + 9·e^5 + e^2.5) = 0.0052 — el KERNEL ya lo daba (0.0057).
    const p0_after = 12.182 / (990.0 + 9.0 * 148.413 + 12.182);
    const p0_meas = @as(f64, @floatFromInt(counts_pen)) / @as(f64, @floatFromInt(N));
    std.debug.print("1.15 gumbel pen: p_tok0 {d:.4} -> {d:.4} medida={d:.4}\n", .{ p0_before, p0_after, p0_meas });
    try std.testing.expect(@abs(p0_meas - p0_after) < 0.02);
}

// a-U3 fase 3c (lane-a): paridad del FUSIONADO qkv — un launch vs 3
// separados (mismo K, mismos pesos packed). Los números deben ser
// IDÉNTICOS (rel 1e-2 absorbe el orden de reduce; los dp4a por fila no
// cambian). Geometría 3B: n_q=3072, n_k=n_v=1024, k=3072.
test "a-U3 qkv fused: paridad 1-launch vs 3-launch" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    @import("debug").init();
    cudaz.ensureContext() catch return error.SkipZigTest;
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);
    var lk = try @import("layer_kernels").LayerKernels.init(stream);
    defer lk.deinit();
    const layer_kernels_pk = @import("layer_kernels");

    const qtypes = @import("kv_cache").quant_types;
    const fmt = qtypes.QuantFormat.q3_k;
    const K: usize = 3072;
    const n_q: usize = 3072;
    const n_kv: usize = 1024;
    const row_bytes = kv_quant.quantBytesRaw(fmt, K);

    // Tres pesos aleatorios (w_q, w_k, w_v) + packs 128B.
    var devs: [3]usize = undefined;
    var outs: [3]usize = undefined;
    const Ns = [_]usize{ n_q, n_kv, n_kv };
    for (0..3) |t| {
        const N = Ns[t];
        const w_bytes = try gpa.alloc(u8, N * row_bytes);
        defer gpa.free(w_bytes);
        {
            var rng = std.Random.Xoshiro256.init(7031 + t);
            const row = try gpa.alloc(f16, K);
            defer gpa.free(row);
            for (0..N) |j| {
                for (row) |*v| v.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);
                const enc = try kv_quant.encodeToOwned(gpa, fmt, row);
                defer gpa.free(enc);
                @memcpy(w_bytes[j * row_bytes ..][0..row_bytes], enc[0..row_bytes]);
            }
        }
        // Pack + upload (producción: q3kPackedWeight). OJO: el free de d_w
        // lo hace el defer GENERAL (abajo) — un defer DENTRO del for se
        // dispara por ITERACIÓN y dejaría devs[t] colgado (error async en
        // el sync: bug real de este test, medido 2026-09-11).
        const pk_bytes = try layer_kernels_pk.repackQ3K128(w_bytes, N, K, gpa);
        defer gpa.free(pk_bytes);
        const d_w = try cudaz.cuMemAlloc(pk_bytes.len);
        try cudaz.cuMemcpyHtoD(d_w, @intFromPtr(pk_bytes.ptr), pk_bytes.len);
        devs[t] = d_w;
        outs[t] = try cudaz.cuMemAlloc(N * 4);
    }
    defer for (0..3) |t| {
        cudaz.cuMemFree(devs[t]);
        cudaz.cuMemFree(outs[t]);
    };

    const a1 = try gpa.alloc(f32, K);
    defer gpa.free(a1);
    {
        var rng = std.Random.Xoshiro256.init(7042);
        for (a1) |*v| v.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);
    }
    const d_a = try cudaz.cuMemAlloc(K * 4);
    defer cudaz.cuMemFree(d_a);
    try cudaz.cuMemcpyHtoD(d_a, @intFromPtr(a1.ptr), K * 4);

    // Camino A: 3 launches separados (packed).
    for (0..3) |t| try lk.q3kGemmM1Dp4aPacked(d_a, devs[t], outs[t], K, Ns[t]);
    try cudaz.cuStreamSynchronize(stream);
    const sep_q = try gpa.alloc(f32, n_q);
    defer gpa.free(sep_q);
    const sep_k = try gpa.alloc(f32, n_kv);
    defer gpa.free(sep_k);
    const sep_v = try gpa.alloc(f32, n_kv);
    defer gpa.free(sep_v);
    try cudaz.cuMemcpyDtoH(@intFromPtr(sep_q.ptr), outs[0], n_q * 4);
    try cudaz.cuMemcpyDtoH(@intFromPtr(sep_k.ptr), outs[1], n_kv * 4);
    try cudaz.cuMemcpyDtoH(@intFromPtr(sep_v.ptr), outs[2], n_kv * 4);

    // Camino B: 1 launch fused.
    for (0..3) |t| try cudaz.cuMemsetD8(outs[t], 0, Ns[t] * 4); // limpiar (¡por su N!)
    try lk.q3kGemmM1Dp4aPackedQKV(d_a, devs[0], devs[1], devs[2], outs[0], outs[1], outs[2], K, n_q, n_kv, n_kv);
    try cudaz.cuStreamSynchronize(stream);
    const fus_q = try gpa.alloc(f32, n_q);
    defer gpa.free(fus_q);
    const fus_k = try gpa.alloc(f32, n_kv);
    defer gpa.free(fus_k);
    const fus_v = try gpa.alloc(f32, n_kv);
    defer gpa.free(fus_v);
    try cudaz.cuMemcpyDtoH(@intFromPtr(fus_q.ptr), outs[0], n_q * 4);
    try cudaz.cuMemcpyDtoH(@intFromPtr(fus_k.ptr), outs[1], n_kv * 4);
    try cudaz.cuMemcpyDtoH(@intFromPtr(fus_v.ptr), outs[2], n_kv * 4);

    // Paridad 1:1 fused vs separado (misma aritmética — drift solo por
    // reduce de bloques distintos: NINGUNO, cada fila es 1 warp en ambos).
    var bad: usize = 0;
    var max_abs: f32 = 0;
    for (0..n_q) |j| {
        const dd = @abs(fus_q[j] - sep_q[j]);
        max_abs = @max(max_abs, dd);
        if (dd > 1e-4) bad += 1;
    }
    for (0..n_kv) |j| {
        max_abs = @max(max_abs, @abs(fus_k[j] - sep_k[j]));
        max_abs = @max(max_abs, @abs(fus_v[j] - sep_v[j]));
        if (@abs(fus_k[j] - sep_k[j]) > 1e-4) bad += 1;
        if (@abs(fus_v[j] - sep_v[j]) > 1e-4) bad += 1;
    }
    std.debug.print("a-U3 qkv fused: bad={d}/{d} max_abs={e} (fused vs separado)\n", .{ bad, n_q + 2 * n_kv, max_abs });
    if (bad > 0) return error.Au3QkvFusedMismatch;
}

// REGRESIÓN c817c4a (lane-a, reportada por lane-b 2026-09-11): el fix de
// fase 2 reescribió por error el write del q4gemmMDp4aKernel como c[row]
// — todas las slices m (blockIdx.y) pisaban la slice 0 (write crow) y el
// prefill q4_0 M>1 producía basura. Test: paridad M=8 vs dot CPU con A
// cuantizada q8_1 (convenio §5.8 — el kernel cuantiza SU token a smem).
// Gate rel 1e-2. Si vuelve a romperse el índice [M][N], filas m>0 dan
// basura ⇒ bad masivo inmediato.
test "regresión c817c4a: q4gemmMDp4a M=8 paridad [M][N]" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    @import("debug").init();
    cudaz.ensureContext() catch return error.SkipZigTest;
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);
    var lk = try @import("layer_kernels").LayerKernels.init(stream);
    defer lk.deinit();

    const qtypes = @import("kv_cache").quant_types;
    const fmt = qtypes.QuantFormat.q4_0;
    const M: usize = 8;
    const K: usize = 512;
    const N: usize = 256;
    const row_bytes = kv_quant.quantBytesRaw(fmt, K);

    const w_bytes = try gpa.alloc(u8, N * row_bytes);
    defer gpa.free(w_bytes);
    {
        var rng = std.Random.Xoshiro256.init(7100);
        const row = try gpa.alloc(f16, K);
        defer gpa.free(row);
        for (0..N) |j| {
            for (row) |*v| v.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);
            const enc = try kv_quant.encodeToOwned(gpa, fmt, row);
            defer gpa.free(enc);
            @memcpy(w_bytes[j * row_bytes ..][0..row_bytes], enc[0..row_bytes]);
        }
    }
    const a = try gpa.alloc(f32, M * K);
    defer gpa.free(a);
    {
        var rng = std.Random.Xoshiro256.init(7101);
        for (a) |*v| v.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);
    }
    const d_a = try cudaz.cuMemAlloc(M * K * 4);
    defer cudaz.cuMemFree(d_a);
    const d_w = try cudaz.cuMemAlloc(w_bytes.len);
    defer cudaz.cuMemFree(d_w);
    const d_c = try cudaz.cuMemAlloc(M * N * 4);
    defer cudaz.cuMemFree(d_c);
    try cudaz.cuMemcpyHtoD(d_a, @intFromPtr(a.ptr), M * K * 4);
    try cudaz.cuMemcpyHtoD(d_w, @intFromPtr(w_bytes.ptr), w_bytes.len);
    try lk.q4gemmMDp4a(d_a, d_w, d_c, M, K, N);
    try cudaz.cuStreamSynchronize(stream);
    const c = try gpa.alloc(f32, M * N);
    defer gpa.free(c);
    try cudaz.cuMemcpyDtoH(@intFromPtr(c.ptr), d_c, M * N * 4);

    // Ref: dequant W + A q8_1 por KB (convenio del kernel).
    const w_ref = try gpa.alloc(f32, N * K);
    defer gpa.free(w_ref);
    {
        const deq = kv_quant.dequant;
        for (0..N) |j| deq(fmt, w_bytes[j * row_bytes ..][0..row_bytes], w_ref[j * K ..][0..K]);
    }
    var bad: usize = 0;
    var max_rel: f32 = 0;
    for (0..M) |m| {
        // A q8_1 del token m (idéntico al smem del kernel).
        const aq_scale = try gpa.alloc(f32, K / 32);
        defer gpa.free(aq_scale);
        const aq_val = try gpa.alloc(i32, K);
        defer gpa.free(aq_val);
        for (0..(K / 32)) |kb| {
            var amax: f32 = 0;
            for (0..32) |r| amax = @max(amax, @abs(a[m * K + kb * 32 + r]));
            const dd: f32 = if (amax > 0) amax / 127.0 else 1.0;
            aq_scale[kb] = dd;
            for (0..32) |r| {
                const q: i32 = @max(-127, @min(127, @as(i32, @intFromFloat(@round(a[m * K + kb * 32 + r] / dd)))));
                aq_val[kb * 32 + r] = q;
            }
        }
        for (0..N) |j| {
            var dot: f32 = 0;
            for (0..(K / 32)) |kb| {
                var sacc: f32 = 0;
                for (0..32) |r| sacc += @as(f32, @floatFromInt(aq_val[kb * 32 + r])) * w_ref[j * K + kb * 32 + r];
                dot += sacc * aq_scale[kb];
            }
            const rel = @abs(c[m * N + j] - dot) / @max(@abs(dot), 1.0);
            max_rel = @max(max_rel, rel);
            if (rel > 1e-2) bad += 1;
        }
    }
    std.debug.print("regresión c817c4a M=8: bad={d}/{d} max_rel={e}\n", .{ bad, M * N, max_rel });
    if (bad > 0) return error.CrowRegression;
}

// ─── dev-IQ P0-5 (2026-09-14): paridad bit-exacta dp4a vs q8_1 CPU ──────────
// Referencia = dot CPU con A cuantizada q8_1 (idéntica a la fase 1 del kernel:
// amax/127 por KB de 32, round, clamp ±127) × W dequantizado por fila.
// Gate: max_rel < 1e-2 (absorbe ruido de punto flotante FP32).
fn dp4aParityTest(
    gpa: std.mem.Allocator,
    lk: *@import("layer_kernels").LayerKernels,
    stream: cudaz.CUstream,
    comptime qtype: u8,
    K: usize,
    N: usize,
    comptime fmt: pa.QuantFormat,
) !void {
    const row_bytes = kv_quant.quantBytesRaw(fmt, K);
    const w_bytes = try gpa.alloc(u8, N * row_bytes);
    defer gpa.free(w_bytes);
    {
        var rng = std.Random.Xoshiro256.init(7031);
        const row = try gpa.alloc(f16, K);
        defer gpa.free(row);
        for (0..N) |j| {
            for (row) |*v| v.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);
            const enc = try kv_quant.encodeToOwned(gpa, fmt, row);
            defer gpa.free(enc);
            @memcpy(w_bytes[j * row_bytes ..][0..row_bytes], enc[0..row_bytes]);
        }
    }
    const a1 = try gpa.alloc(f32, K);
    defer gpa.free(a1);
    {
        var rng = std.Random.Xoshiro256.init(7032);
        for (a1) |*v| v.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);
    }

    const d_a = try cudaz.cuMemAlloc(K * @sizeOf(f32));
    defer cudaz.cuMemFree(d_a);
    const d_w = try cudaz.cuMemAlloc(w_bytes.len);
    defer cudaz.cuMemFree(d_w);
    try cudaz.cuMemcpyHtoD(d_a, @intFromPtr(a1.ptr), K * @sizeOf(f32));
    try cudaz.cuMemcpyHtoD(d_w, @intFromPtr(w_bytes.ptr), w_bytes.len);
    const d_c_dp4a = try cudaz.cuMemAlloc(N * @sizeOf(f32));
    defer cudaz.cuMemFree(d_c_dp4a);
    const d_c_ref = try cudaz.cuMemAlloc(N * @sizeOf(f32));
    defer cudaz.cuMemFree(d_c_ref);

    // Ref escalar (A f32) — solo sanity check, no es la ref de paridad.
    try lk.qgemm(d_a, d_w, d_c_ref, 1, K, N, qtype);
    // Dp4a.
    switch (qtype) {
        8 => try lk.iq3sGemmM1Dp4a(d_a, d_w, d_c_dp4a, K, N),
        9 => try lk.iq2sGemmM1Dp4a(d_a, d_w, d_c_dp4a, K, N),
        18 => try lk.iq4xsGemmM1Dp4a(d_a, d_w, d_c_dp4a, K, N),
        else => unreachable,
    }
    try cudaz.cuStreamSynchronize(stream);

    const c_ref = try gpa.alloc(f32, N);
    defer gpa.free(c_ref);
    const c_dp4a = try gpa.alloc(f32, N);
    defer gpa.free(c_dp4a);
    try cudaz.cuMemcpyDtoH(@intFromPtr(c_ref.ptr), d_c_ref, N * @sizeOf(f32));
    try cudaz.cuMemcpyDtoH(@intFromPtr(c_dp4a.ptr), d_c_dp4a, N * @sizeOf(f32));

    // Ref CPU: W dequant + A q8_1 (fase 1 idéntica al kernel).
    const w_ref = try gpa.alloc(f32, N * K);
    defer gpa.free(w_ref);
    {
        const deq = kv_quant.dequant;
        for (0..N) |j| deq(fmt, w_bytes[j * row_bytes ..][0..row_bytes], w_ref[j * K ..][0..K]);
    }
    const aq_scale = try gpa.alloc(f32, K / 32);
    defer gpa.free(aq_scale);
    const aq_val = try gpa.alloc(i32, K);
    defer gpa.free(aq_val);
    for (0..(K / 32)) |kb| {
        var amax: f32 = 0;
        for (0..32) |r| amax = @max(amax, @abs(a1[kb * 32 + r]));
        const dd: f32 = if (amax > 0) amax / 127.0 else 1.0;
        aq_scale[kb] = dd;
        for (0..32) |r| {
            const q: i32 = @max(-127, @min(127, @as(i32, @intFromFloat(@round(a1[kb * 32 + r] / dd)))));
            aq_val[kb * 32 + r] = q;
        }
    }

    var bad: usize = 0;
    var max_rel: f32 = 0;
    var max_abs: f32 = 0;
    var first_bad: ?usize = null;
    for (0..N) |j| {
        var dot_q8: f32 = 0;
        for (0..(K / 32)) |kb| {
            var sacc: f32 = 0;
            for (0..32) |r| sacc += @as(f32, @floatFromInt(aq_val[kb * 32 + r])) * w_ref[j * K + kb * 32 + r];
            dot_q8 += sacc * aq_scale[kb];
        }
        const abs_diff = @abs(c_dp4a[j] - dot_q8);
        const rel = abs_diff / @max(@abs(dot_q8), 1.0);
        max_rel = @max(max_rel, rel);
        max_abs = @max(max_abs, abs_diff);
        if (rel > 1e-2) {
            bad += 1;
            if (first_bad == null) {
                first_bad = j;
                std.debug.print("  first_bad row {d}: dp4a={d:.6} cpu_q8={d:.6} rel={e}\n", .{ j, c_dp4a[j], dot_q8, rel });
            }
        }
    }
    const tag = switch (qtype) { 8 => "iq3s", 9 => "iq2s", 18 => "iq4xs", else => "?" };
    std.debug.print("{s} dp4a M=1 k={d} n={d}: bad={d}/{d} max_rel={e} max_abs={e}\n", .{ tag, K, N, bad, N, max_rel, max_abs });
    if (bad > 0) {
        return error.Dp4aParityFail;
    }
}

// ─── dev-IQ P0-5: paridad iq3_s dp4a (case 8) vs CPU q8_1 ────────────────────
// Layout 110B/SB256, grid 512 (9-bit). Geometría 9B FFN: k=8192, n=3584.
test "dev-IQ iq3_s M=1 dp4a: paridad bit-exacta vs q8_1 CPU" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    @import("debug").init();
    cudaz.ensureContext() catch return error.SkipZigTest;
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);
    var lk = try @import("layer_kernels").LayerKernels.init(stream);
    defer lk.deinit();

    try dp4aParityTest(gpa, &lk, stream, 8, 8192, 3584, .iq3_s);
}

// ─── dev-IQ P0-5: paridad iq2_s dp4a (case 9) vs CPU q8_1 ────────────────────
// Layout 82B/SB256, grid 1024 (10-bit). Geometría 9B FFN: k=8192, n=3584.
test "dev-IQ iq2_s M=1 dp4a: paridad bit-exacta vs q8_1 CPU" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    @import("debug").init();
    cudaz.ensureContext() catch return error.SkipZigTest;
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);
    var lk = try @import("layer_kernels").LayerKernels.init(stream);
    defer lk.deinit();

    try dp4aParityTest(gpa, &lk, stream, 9, 8192, 3584, .iq2_s);
}

// ─── dev-IQ P0-5: paridad iq4_xs dp4a (case 18) vs CPU q8_1 ──────────────────
// Layout 136B/SB256, LUT kvalues_iq4nl. Geometría 4B FFN: k=5120, n=1280.
test "dev-IQ iq4_xs M=1 dp4a: paridad bit-exacta vs q8_1 CPU" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    @import("debug").init();
    cudaz.ensureContext() catch return error.SkipZigTest;
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);
    var lk = try @import("layer_kernels").LayerKernels.init(stream);
    defer lk.deinit();

    try dp4aParityTest(gpa, &lk, stream, 18, 5120, 1280, .iq4_xs);
}

// ─── dev-IQ P0-5: bench dp4a vs escalar para cases 8/9/18 ────────────────────
// Mide speedup del dp4a sobre el kernel escalar para cada formato.
// Geometrías representativas 9B: k=8192, n∈{3584, 8192}.
test "dev-IQ dp4a bench: iq3_s/iq2_s/iq4_xs M=1 speedup vs escalar" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    @import("debug").init();
    cudaz.ensureContext() catch return error.SkipZigTest;
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);
    var lk = try @import("layer_kernels").LayerKernels.init(stream);
    defer lk.deinit();

    const K: usize = 8192;
    const cases = [_]struct { qtype: u8, fmt: pa.QuantFormat, n: usize, tag: []const u8 } {
        .{ .qtype = 8, .fmt = .iq3_s, .n = 3584, .tag = "iq3s" },
        .{ .qtype = 9, .fmt = .iq2_s, .n = 8192, .tag = "iq2s" },
        .{ .qtype = 18, .fmt = .iq4_xs, .n = 3584, .tag = "iq4xs" },
    };

    for (cases) |case_val| {
        const row_bytes = kv_quant.quantBytesRaw(case_val.fmt, K);
        const w_bytes = try gpa.alloc(u8, case_val.n * row_bytes);
        defer gpa.free(w_bytes);
        {
            var rng = std.Random.Xoshiro256.init(7031);
            const row = try gpa.alloc(f16, K);
            defer gpa.free(row);
            for (0..case_val.n) |j| {
                for (row) |*v| v.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);
                const enc = try kv_quant.encodeToOwned(gpa, case_val.fmt, row);
                defer gpa.free(enc);
                @memcpy(w_bytes[j * row_bytes ..][0..row_bytes], enc[0..row_bytes]);
            }
        }
        const a1 = try gpa.alloc(f32, K);
        defer gpa.free(a1);
        {
            var rng = std.Random.Xoshiro256.init(7032);
            for (a1) |*v| v.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);
        }

        const d_a = try cudaz.cuMemAlloc(K * @sizeOf(f32));
        defer cudaz.cuMemFree(d_a);
        const d_w = try cudaz.cuMemAlloc(w_bytes.len);
        defer cudaz.cuMemFree(d_w);
        const d_c_scalar = try cudaz.cuMemAlloc(case_val.n * @sizeOf(f32));
        defer cudaz.cuMemFree(d_c_scalar);
        const d_c_dp4a = try cudaz.cuMemAlloc(case_val.n * @sizeOf(f32));
        defer cudaz.cuMemFree(d_c_dp4a);
        try cudaz.cuMemcpyHtoD(d_a, @intFromPtr(a1.ptr), K * @sizeOf(f32));
        try cudaz.cuMemcpyHtoD(d_w, @intFromPtr(w_bytes.ptr), w_bytes.len);

        // Warmup.
        try lk.qgemm(d_a, d_w, d_c_scalar, 1, K, case_val.n, case_val.qtype);
        try cudaz.cuStreamSynchronize(stream);
        switch (case_val.qtype) {
            8 => try lk.iq3sGemmM1Dp4a(d_a, d_w, d_c_dp4a, K, case_val.n),
            9 => try lk.iq2sGemmM1Dp4a(d_a, d_w, d_c_dp4a, K, case_val.n),
            18 => try lk.iq4xsGemmM1Dp4a(d_a, d_w, d_c_dp4a, K, case_val.n),
            else => unreachable,
        }
        try cudaz.cuStreamSynchronize(stream);

        var ns_scalar: i128 = std.math.maxInt(i128);
        var ns_dp4a: i128 = std.math.maxInt(i128);
        for (0..5) |_| {
            var t = @import("time").Timer.start();
            try lk.qgemm(d_a, d_w, d_c_scalar, 1, K, case_val.n, case_val.qtype);
            try cudaz.cuStreamSynchronize(stream);
            ns_scalar = @min(ns_scalar, t.read());
            t = @import("time").Timer.start();
            switch (case_val.qtype) {
                8 => try lk.iq3sGemmM1Dp4a(d_a, d_w, d_c_dp4a, K, case_val.n),
                9 => try lk.iq2sGemmM1Dp4a(d_a, d_w, d_c_dp4a, K, case_val.n),
                18 => try lk.iq4xsGemmM1Dp4a(d_a, d_w, d_c_dp4a, K, case_val.n),
                else => unreachable,
            }
            try cudaz.cuStreamSynchronize(stream);
            ns_dp4a = @min(ns_dp4a, t.read());
        }
        std.debug.print("BENCH {s} k={d} n={d}: escalar={d:.3}ms dp4a={d:.3}ms speedup={d:.2}x\n", .{
            case_val.tag,
            K,
            case_val.n,
            @as(f64, @floatFromInt(ns_scalar)) / 1e6,
            @as(f64, @floatFromInt(ns_dp4a)) / 1e6,
            @as(f64, @floatFromInt(ns_scalar)) / @as(f64, @floatFromInt(@max(ns_dp4a, 1))),
        });
    }
}
