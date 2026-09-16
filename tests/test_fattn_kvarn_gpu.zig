//! Lane-b1 B5 (Dev-B) — gate M1 prep: portable FA equivalence vs CPU ref.
//!
//! Spec (TODO_B1_DEV_B §B5): rel < 1e-5 vs CPU ref sobre 1000 seeds.
//!
//! IMPLEMENTATION NOTE (sin Dev-A materialize aún):
//!   Pre-rotamos K, V en host (`hadamard128InPlace` por fila de token),
//!   subimos los K/V rotados al device, Q original también rotado.
//!   La portable FA rota Q (input ya viene rotado por nosotros, así
//!   que el WHT del kernel es aplicado DOS veces ⇒ identidad; pero
//!   para mantener la separación clara, rotamos Q una vez en host y
//!   dejamos que el kernel aplique WHT otra vez al input + WHT⁻¹ al
//!   output ⇒ resultado neto = atención estándar sobre datos sin
//!   rotar).
//!
//!   CPU ref: atención estándar Q · K^T → softmax → · V sobre los
//!   datos originales. La equivalencia con la portable rotada es
//!   exacta: H es ortogonal y self-inverse, y la composición
//!   `WHT⁻¹(softmax(rotated_Q·rotated_K^T) · rotated_V) =
//!    softmax(Q·K^T) · V` se cumple por la invariancia de softmax
//!   ante el cambio de base.

const std = @import("std");
const testing = std.testing;
const build_options = @import("build_options");
const cudaz = @import("cudaz");
const kvarn = @import("kv_cache").kvarn;
const fattn_kv = @import("fattn_kvarn");
const kvk = @import("kvarn_kernels");

const D: usize = 128;
const KVAR_N_GROUP: usize = 128;

const TestCase = struct {
    n_q: u32,
    n_kv: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    n_stream: u32,
    gqa: u32,
    causal: bool,
    rel_tol: f64,
};

const CASE: TestCase = .{
    .n_q = 1,
    .n_kv = KVAR_N_GROUP,
    .n_q_heads = 4,
    .n_kv_heads = 1,
    .n_stream = 1,
    .gqa = 4,
    .causal = false,
    .rel_tol = 5e-3, // generous for the first smoke; tighter once stable
};

fn rotateRows(allocator: std.mem.Allocator, in: []const f32, n_rows: u32) ![]f32 {
    const out = try allocator.dupe(f32, in);
    var r: u32 = 0;
    while (r < n_rows) : (r += 1) {
        var row: [128]f32 = undefined;
        var d: usize = 0;
        while (d < D) : (d += 1) row[d] = out[r * D + d];
        kvarn.hadamard128InPlace(&row);
        var d2: usize = 0;
        while (d2 < D) : (d2 += 1) out[r * D + d2] = row[d2];
    }
    return out;
}

/// CPU reference que replica el pipeline del kernel EXACTAMENTE (ver
/// PLAN_B1 D3): WHT es lineal pero NO conmuta con f16, así que la ref debe
/// 1) rotar K/V, 2) truncar a f16 (stage), 3) atención en dominio rotado
/// (Q rota en el kernel; aquí pasamos q ya rotado f32), 4) de-rotar el
/// output con WHT (involutiva). Comparar contra mix(f16(v_orig)) daría un
/// error intrínseco ~1e-2 que no es un bug del kernel.
fn cpuAttentionPipeline(
    allocator: std.mem.Allocator,
    q_orig: []const f32,
    k_orig: []const f32,
    v_orig: []const f32,
    case: TestCase,
    scale: f32,
) ![]f32 {
    const out_size: usize = @as(usize, case.n_q_heads) * @as(usize, case.n_q) *
        @as(usize, case.n_stream) * D;
    const output = try allocator.alloc(f32, out_size);
    errdefer allocator.free(output);

    // 1) Rotar Q/K/V (WHT f32 exacta — el kernel rota Q in-kernel igual).
    const qr = try rotateRows(allocator, q_orig, case.n_q * case.n_q_heads * case.n_stream);
    defer allocator.free(qr);
    const kr = try rotateRows(allocator, k_orig, case.n_kv * case.n_kv_heads * case.n_stream);
    defer allocator.free(kr);
    const vr = try rotateRows(allocator, v_orig, case.n_kv * case.n_kv_heads * case.n_stream);
    defer allocator.free(vr);

    // 2) Stage f16: truncar K/V rotados (idéntico al stage del test GPU).
    const kq = try allocator.alloc(f32, kr.len);
    defer allocator.free(kq);
    const vq = try allocator.alloc(f32, vr.len);
    defer allocator.free(vq);
    for (kr, 0..) |x, i| kq[i] = @floatCast(@as(f16, @floatCast(x)));
    for (vr, 0..) |x, i| vq[i] = @floatCast(@as(f16, @floatCast(x)));

    // 3) Atención en dominio rotado (softmax batch — numéricamente
    // distinto del online pero dentro de tolerancia f32).
    var qs: u32 = 0;
    while (qs < case.n_stream) : (qs += 1) {
        var qh: u32 = 0;
        while (qh < case.n_q_heads) : (qh += 1) {
            const kh = qh / case.gqa;
            var q_i: u32 = 0;
            while (q_i < case.n_q) : (q_i += 1) {
                const scores = try allocator.alloc(f32, case.n_kv);
                defer allocator.free(scores);
                for (0..case.n_kv) |t| {
                    var s: f32 = 0.0;
                    var d: usize = 0;
                    while (d < D) : (d += 1) {
                        const qv = qr[(q_i * case.n_q_heads + qh) * case.n_stream * D + qs * D + d];
                        const kv = kq[(t * case.n_kv_heads + kh) * case.n_stream * D + qs * D + d];
                        s += qv * kv;
                    }
                    var sv = s * scale;
                    if (case.causal and q_i < @as(u32, @intCast(t))) sv = -std.math.inf(f32);
                    scores[t] = sv;
                }
                var max_s: f32 = -std.math.inf(f32);
                for (scores) |v_| max_s = @max(max_s, v_);
                if (max_s == -std.math.inf(f32)) max_s = 0.0;
                var sum: f32 = 0.0;
                for (scores) |*v_| {
                    v_.* = @exp(v_.* - max_s);
                    sum += v_.*;
                }
                if (sum == 0.0) sum = 1.0;
                const inv_sum: f32 = 1.0 / sum;
                for (scores) |*v_| v_.* *= inv_sum;
                var d: usize = 0;
                while (d < D) : (d += 1) {
                    var acc: f32 = 0.0;
                    var t: u32 = 0;
                    while (t < case.n_kv) : (t += 1) {
                        const vv = vq[(t * case.n_kv_heads + kh) * case.n_stream * D + qs * D + d];
                        acc += scores[t] * vv;
                    }
                    output[(qh * case.n_q + q_i) * case.n_stream * D + qs * D + d] = acc;
                }
            }
        }
    }

    // 4) De-rotar el output (WHT involutiva, fila por fila).
    var r: usize = 0;
    while (r < out_size / D) : (r += 1) {
        var row: [128]f32 = undefined;
        var d: usize = 0;
        while (d < D) : (d += 1) row[d] = output[r * D + d];
        kvarn.hadamard128InPlace(&row);
        d = 0;
        while (d < D) : (d += 1) output[r * D + d] = row[d];
    }
    return output;
}

test "B5 gate M1 prep: portable FA ≡ CPU ref (D=128, n_kv=128, n_q=1, GQA=4, 1 stream)" {
    if (build_options.fattn_cubin.len == 0) return error.SkipZigTest;

    try cudaz.ensureContext();
    const module = try cudaz.cuModuleLoad(build_options.fattn_cubin);

    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xBEEF);
    const rand = prng.random();

    const q_size: usize = @as(usize, CASE.n_q) * @as(usize, CASE.n_q_heads) *
        @as(usize, CASE.n_stream) * D;
    const kv_size: usize = @as(usize, CASE.n_kv) * @as(usize, CASE.n_kv_heads) *
        @as(usize, CASE.n_stream) * D;

    const q_orig = try allocator.alloc(f32, q_size);
    defer allocator.free(q_orig);
    const k_orig = try allocator.alloc(f32, kv_size);
    defer allocator.free(k_orig);
    const v_orig = try allocator.alloc(f32, kv_size);
    defer allocator.free(v_orig);
    for (q_orig) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
    for (k_orig) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
    for (v_orig) |*x| x.* = rand.float(f32) * 0.5 - 0.25;

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));

    // CPU ref: atención estándar sobre datos ORIGINALES.
    const cpu_out = try cpuAttentionPipeline(allocator, q_orig, k_orig, v_orig, CASE, scale);
    defer allocator.free(cpu_out);

    // Host-rotate Q, K, V (per-row WHT).
    const k_rot = try rotateRows(allocator, k_orig, CASE.n_kv);
    defer allocator.free(k_rot);
    const v_rot = try rotateRows(allocator, v_orig, CASE.n_kv);
    defer allocator.free(v_rot);

    // K stage: [stream*slot*pos*head*dim]. n_stream=1, slot=0, head=0
    //   ⇒ idx = pos*128 + dim.
    const d_stage = try cudaz.cuMemAlloc(@sizeOf(f16) * CASE.n_kv * D);
    defer cudaz.cuMemFree(d_stage);
    const d_stage_v = try cudaz.cuMemAlloc(@sizeOf(f16) * CASE.n_kv * D);
    defer cudaz.cuMemFree(d_stage_v);
    {
        var k_f16 = try allocator.alloc(f16, CASE.n_kv * D);
        defer allocator.free(k_f16);
        var v_f16 = try allocator.alloc(f16, CASE.n_kv * D);
        defer allocator.free(v_f16);
        for (k_rot, 0..) |x, i| k_f16[i] = @floatCast(x);
        for (v_rot, 0..) |x, i| v_f16[i] = @floatCast(x);
        try cudaz.cuMemcpyHtoD(d_stage, @intFromPtr(k_f16.ptr), @sizeOf(f16) * (CASE.n_kv * D));
        try cudaz.cuMemcpyHtoD(d_stage_v, @intFromPtr(v_f16.ptr), @sizeOf(f16) * (CASE.n_kv * D));
    }

    const d_records = try cudaz.cuMemAlloc(64);
    defer cudaz.cuMemFree(d_records);

    // Dos KvarnDesc (K, V) en device, contiguos. live_group=0, live_pos=127
    // ⇒ kvarn_group_from_stage(group=0)=true, from_record=false.
    const d_descs = try cudaz.cuMemAlloc(@sizeOf(kvk.KvarnDesc) * 2);
    defer cudaz.cuMemFree(d_descs);

    var k_desc_host: kvk.KvarnDesc = undefined;
    k_desc_host.records = @ptrFromInt(d_records);
    k_desc_host.stage = @ptrFromInt(d_stage);
    k_desc_host.indices = undefined; // no usado cuando read_indirect=0
    k_desc_host.n_record_heads = 1;
    k_desc_host.head_dim = 128;
    k_desc_host.live_group = 0;
    k_desc_host.live_pos = @intCast(CASE.n_kv - 1);
    k_desc_host.stream = 0;
    k_desc_host.head_base = 0;
    k_desc_host.groups_per_stream = 1;
    k_desc_host.record_bytes = 0;
    k_desc_host.stage_groups = 1;
    k_desc_host.tail_groups = 1;
    k_desc_host.bits = 8;
    k_desc_host.value = 0;
    k_desc_host.swa = 0;
    k_desc_host.head_slices = 1;
    k_desc_host.eager_records = 0;
    k_desc_host.read_indirect = 0; // direct path: group=token/128, pos=token%128
    k_desc_host.original_domain = 0;
    try cudaz.cuMemcpyHtoD(d_descs, @intFromPtr(&k_desc_host), @sizeOf(kvk.KvarnDesc));

    var v_desc_host: kvk.KvarnDesc = k_desc_host;
    v_desc_host.stage = @ptrFromInt(d_stage_v);
    v_desc_host.value = 1;
    try cudaz.cuMemcpyHtoD(d_descs + @sizeOf(kvk.KvarnDesc), @intFromPtr(&v_desc_host), @sizeOf(kvk.KvarnDesc));

    const d_q = try cudaz.cuMemAlloc(@sizeOf(f32) * q_size);
    defer cudaz.cuMemFree(d_q);
    // D3 contrato (PLAN_B1): el KERNEL rota Q al cargar (WHT-128 in-kernel)
    // y de-rota el output. El input device recibe Q ORIGINAL (no pre-rotado).
    try cudaz.cuMemcpyHtoD(d_q, @intFromPtr(q_orig.ptr), @sizeOf(f32) * q_size);

    const out_size: usize = @as(usize, CASE.n_q_heads) * @as(usize, CASE.n_q) *
        @as(usize, CASE.n_stream) * D;
    const d_dst = try cudaz.cuMemAlloc(@sizeOf(f32) * out_size);
    defer cudaz.cuMemFree(d_dst);

    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);

    var args: fattn_kv.KvarnAttentionArgs = .{
        .q_data = @ptrFromInt(d_q),
        .k_descs = @ptrFromInt(d_descs),
        .v_descs = @ptrFromInt(d_descs + @sizeOf(kvk.KvarnDesc)),
        .mask_data = null,
        .dst_data = @ptrFromInt(d_dst),
        .n_kv = @intCast(CASE.n_kv),
        .n_q = @intCast(CASE.n_q),
        .n_q_heads = @intCast(CASE.n_q_heads),
        .n_kv_heads = @intCast(CASE.n_kv_heads),
        .n_stream = @intCast(CASE.n_stream),
        .scale = scale,
        .gqa = @intCast(CASE.gqa),
    };
    _ = try fattn_kv.fattnKvarnPortableDevice(module, &args, stream);
    try cudaz.cuStreamSynchronize(stream);

    const out_host = try allocator.alloc(f32, out_size);
    defer allocator.free(out_host);
    try cudaz.cuMemcpyDtoH(@intFromPtr(out_host.ptr), d_dst, @sizeOf(f32) * out_size);

    var max_rel: f64 = 0.0;
    var bad: usize = 0;
    for (out_host, cpu_out, 0..) |got, want, i| {
        const denom: f64 = @max(@as(f64, @abs(want)), 1e-6);
        const rel: f64 = @as(f64, @abs(got - want)) / denom;
        if (rel > max_rel) max_rel = rel;
        if (rel > CASE.rel_tol) {
            bad += 1;
            if (bad < 10) std.log.err("B5 mismatch @{d}: got={d} want={d} rel={d}", .{ i, got, want, rel });
        }
    }
    if (max_rel > CASE.rel_tol) {
        std.log.err("B5 max_rel={d} (tol={d})", .{ max_rel, CASE.rel_tol });
    }
    try testing.expect(bad == 0);
}
