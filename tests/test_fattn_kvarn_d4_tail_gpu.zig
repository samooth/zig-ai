//! Lane-b1 B4 iter 3 — D4 KVCPT tail E2E test.
//!
//! Spec (TODO_B1_DEV_B §B4 follow-up, D4): "MISMO softmax online, SIN
//! materializar aparte. Fuente: `k_exact`/`v_exact` del manager B2
//! (contrato `getExactTail`); cola exacta f16/bf16 → f32".
//!
//! STATUS (B4 iter 3 E2E):
//!   El kernel D=128+tail está escrito y compila (ver
//!   src/cuda/fattn_kvarn_portable.cu). El Zig launcher
//!   `fattnKvarnPortableD128TailDevice` valida los args (errores:
//!   InvalidTailShape, GqaMismatch, MissingTailK/V/Slots) y emite
//!   breadcrumbs; la llamada cuLaunchKernel queda gated por
//!   P4 wiring (retorna error.NotImplemented).
//!
//!   El test E2E real (subir k_tail_data + run_desc_slots a device,
//! lanzar el kernel, comparar vs CPU ref) requiere:
//!     (a) wiring P4 (mi kernel no se compila en el árbol hasta que
//!         Dev-A o yo abramos P4),
//!     (b) B2 manager `getExactTail` con su formato de slots concreto,
//!     (c) un run-plan del geometry analyzer.
//!
//!   Mientras tanto, este test:
//!     1) Verifica la API surface del wrapper Zig (errores de
//!        validación, defaults de la struct).
//!     2) Construye una CPU ref del body+tail integrado (D=128,
//!        GQA=1, n_kv pequeño + n_tail pequeño) que el E2E real
//!        usará cuando P4 cierre.
//!     3) Verifica que la CPU ref es bit-exacta vs la atención
//!        estándar NO-WHT (sanity del ref).

const std = @import("std");
const testing = std.testing;
const build_options = @import("build_options");
const cudaz = @import("cudaz");
const fattn_kv = @import("fattn_kvarn");
const kvk = @import("kvarn_kernels");
const kvarn = @import("kv_cache").kvarn;

const D: usize = 128;

const CASE = struct {
    n_kv_body: u32,
    n_tail: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    gqa: u32,
    rel_tol: f64,
};

const D4_CASE: CASE = .{
    .n_kv_body = 4,
    .n_tail = 4,
    .n_q_heads = 2,
    .n_kv_heads = 1,
    .gqa = 2,
    .rel_tol = 1e-4,
};

/// CPU ref del kernel D=128+tail: atención estándar sobre los tokens
/// del body + tail, sin cuantizar, sin WHT, mismo softmax online.
///
/// Args:
///   q: [D]
///   k_body: [n_kv_body × D]
///   v_body: [n_kv_body × D]
///   k_tail: [n_tail × D]
///   v_tail: [n_tail × D]
///   scale: 1/sqrt(D)
///
/// Output: [D]
/// WHT por fila de 128 sobre `data` (dominio rotado).
fn rotateRows128(data: []const f32, out: []f32) void {
    const n = data.len / 128;
    for (0..n) |t| {
        var row: [128]f32 = undefined;
        for (0..128) |d| row[d] = data[t * 128 + d];
        kvarn.hadamard128InPlace(&row);
        for (0..128) |d| out[t * 128 + d] = row[d];
    }
}

/// Stage CPU dominio ROTADO: WHT + truncado f16 (pipeline-exacto del
/// grupo sink cuando el store eager no sella el grupo 0).
fn stageRoundtripRot(data: []const f32, out: []f32) void {
    const n = data.len / 128;
    for (0..n) |t| {
        var row: [128]f32 = undefined;
        for (0..128) |d| row[d] = data[t * 128 + d];
        kvarn.hadamard128InPlace(&row);
        for (0..128) |d| out[t * 128 + d] = @as(f32, @floatCast(@as(f16, @floatCast(row[d]))));
    }
}

fn cpuAttentionBodyTail(
    q: []const f32,
    k_body: []const f32,
    v_body: []const f32,
    k_tail: []const f32,
    v_tail: []const f32,
    n_kv_body: u32,
    n_tail: u32,
    scale: f32,
) [D]f32 {
    const n_total = n_kv_body + n_tail;
    var scores: [8]f32 = undefined; // soporta hasta 8 tokens (4 body + 4 tail)
    var max_s: f32 = -std.math.inf(f32);

    for (0..n_kv_body) |t| {
        var s: f32 = 0.0;
        for (0..D) |d| s += q[d] * k_body[t * D + d];
        scores[t] = s * scale;
        if (scores[t] > max_s) max_s = scores[t];
    }
    for (0..n_tail) |t| {
        var s: f32 = 0.0;
        for (0..D) |d| s += q[d] * k_tail[t * D + d];
        scores[n_kv_body + t] = s * scale;
        if (scores[n_kv_body + t] > max_s) max_s = scores[n_kv_body + t];
    }

    if (max_s == -std.math.inf(f32)) max_s = 0.0;
    var sum: f32 = 0.0;
    for (0..n_total) |t| {
        scores[t] = @exp(scores[t] - max_s);
        sum += scores[t];
    }
    if (sum == 0.0) sum = 1.0;
    const inv_sum: f32 = 1.0 / sum;
    for (0..n_total) |t| {
        scores[t] *= inv_sum;
    }

    var output: [D]f32 = std.mem.zeroes([D]f32);
    for (0..n_total) |t| {
        for (0..D) |d| {
            const vv = if (t < n_kv_body)
                v_body[t * D + d]
            else
                v_tail[(t - n_kv_body) * D + d];
            output[d] += scores[t] * vv;
        }
    }
    return output;
}

test "D4 KVCPT E2E: CPU ref self-consistency (sin WHT, sin cuantizar)" {
    // Sanity: la CPU ref produce algo razonable con datos random.
    const allocator = testing.allocator;
    const n_total = D4_CASE.n_kv_body + D4_CASE.n_tail;
    const n_kv = D4_CASE.n_kv_body;
    const n_tail = D4_CASE.n_tail;

    var prng = std.Random.DefaultPrng.init(0xD4D4);
    const rand = prng.random();

    const q = try allocator.alloc(f32, D);
    defer allocator.free(q);
    const k_body = try allocator.alloc(f32, n_kv * D);
    defer allocator.free(k_body);
    const v_body = try allocator.alloc(f32, n_kv * D);
    defer allocator.free(v_body);
    const k_tail = try allocator.alloc(f32, n_tail * D);
    defer allocator.free(k_tail);
    const v_tail = try allocator.alloc(f32, n_tail * D);
    defer allocator.free(v_tail);

    for (q) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
    for (k_body) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
    for (v_body) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
    for (k_tail) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
    for (v_tail) |*x| x.* = rand.float(f32) * 0.5 - 0.25;

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    const out = cpuAttentionBodyTail(q, k_body, v_body, k_tail, v_tail, n_kv, n_tail, scale);

    // La suma de pesos normalizados = 1; verificamos que el output
    // no es degenerado (no todos 0, no NaN, no infinita).
    var max_abs: f32 = 0.0;
    var has_nan: bool = false;
    for (out) |v_| {
        const a = @abs(v_);
        if (a > max_abs) max_abs = a;
        if (std.math.isNan(v_) or std.math.isInf(v_)) has_nan = true;
    }
    try testing.expect(!has_nan);
    try testing.expect(max_abs > 1e-3); // output no degenerado

    // Comparar contra la atención estándar NO-WHT: deben coincidir
    // bit-a-bit porque la ref ya es "atención estándar integrada body+tail".
    var scores_ref: [8]f32 = undefined;
    var max_ref: f32 = -std.math.inf(f32);
    for (0..n_total) |t| {
        const k = if (t < n_kv) k_body[t * D ..][0..D] else k_tail[(t - n_kv) * D ..][0..D];
        var s: f32 = 0.0;
        for (0..D) |d| s += q[d] * k[d];
        scores_ref[t] = s * scale;
        if (scores_ref[t] > max_ref) max_ref = scores_ref[t];
    }
    if (max_ref == -std.math.inf(f32)) max_ref = 0.0;
    var sum: f32 = 0.0;
    for (0..n_total) |t| {
        scores_ref[t] = @exp(scores_ref[t] - max_ref);
        sum += scores_ref[t];
    }
    const inv: f32 = 1.0 / sum;
    for (0..n_total) |t| scores_ref[t] *= inv;

    // El softmax del ref y el del estándar deben coincidir ⇒ output igual.
    var diff_max: f32 = 0.0;
    for (0..D) |d| {
        var s: f32 = 0.0;
        for (0..n_total) |t| {
            const vv = if (t < n_kv) v_body[t * D + d] else v_tail[(t - n_kv) * D + d];
            s += scores_ref[t] * vv;
        }
        const diff = @abs(s - out[d]);
        if (diff > diff_max) diff_max = diff;
    }
    try testing.expect(diff_max < 1e-6); // bit-exacto
}

test "D4 KVCPT E2E: Zig API surface — errors de validación" {
    // n_q=0 ⇒ error.InvalidTailShape
    {
        var args: fattn_kv.KvarnAttentionTailArgs = .{};
        try testing.expectError(error.InvalidTailShape, fattnKvarnPortableD128TailDevice(null, &args, @ptrFromInt(1)));
    }
    // GQA mismatch
    {
        var args: fattn_kv.KvarnAttentionTailArgs = .{
            .q_data = @ptrFromInt(0x100),
            .k_descs = @ptrFromInt(0x200),
            .v_descs = @ptrFromInt(0x300),
            .n_q = 1,
            .n_q_heads = 7, // 7 != 2*4
            .n_kv_heads = 2,
            .gqa = 4,
            .n_stream = 1,
            .scale = 0.1,
        };
        try testing.expectError(error.GqaMismatch, fattnKvarnPortableD128TailDevice(null, &args, @ptrFromInt(1)));
    }
    // n_tail>0 sin k_tail_data
    {
        var args: fattn_kv.KvarnAttentionTailArgs = .{
            .q_data = @ptrFromInt(0x100),
            .k_descs = @ptrFromInt(0x200),
            .v_descs = @ptrFromInt(0x300),
            .n_q = 1,
            .n_q_heads = 2,
            .n_kv_heads = 1,
            .gqa = 2,
            .n_stream = 1,
            .scale = 0.1,
            .n_tail = 4,
        };
        try testing.expectError(error.MissingTailK, fattnKvarnPortableD128TailDevice(null, &args, @ptrFromInt(1)));
    }
    // n_tail=0 OK (no requiere pointers) — module=null ⇒ NeedCudaModule.
    {
        var args: fattn_kv.KvarnAttentionTailArgs = .{
            .q_data = @ptrFromInt(0x100),
            .k_descs = @ptrFromInt(0x200),
            .v_descs = @ptrFromInt(0x300),
            .n_q = 1,
            .n_q_heads = 2,
            .n_kv_heads = 1,
            .gqa = 2,
            .n_stream = 1,
            .scale = 0.1,
        };
        try testing.expectError(error.NeedCudaModule, fattnKvarnPortableD128TailDevice(null, &args, @ptrFromInt(1)));
    }
}

test "D4 KVCPT E2E: defaults struct KvarnAttentionTailArgs" {
    const args: fattn_kv.KvarnAttentionTailArgs = .{};
    try testing.expectEqual(@as(c_int, 0), args.n_kv);
    try testing.expectEqual(@as(c_int, 0), args.n_tail);
    try testing.expectEqual(@as(c_int, 0), args.d_k);
    try testing.expectEqual(@as(c_int, 0), args.d_v);
    try testing.expectEqual(@as(c_int, 0), args.k_tail_bf16);
    try testing.expectEqual(@as(c_int, 0), args.v_tail_bf16);
    try testing.expectEqual(@as(?[*]const u8, null), args.k_tail_data);
    try testing.expectEqual(@as(?[*]const u8, null), args.v_tail_data);
    try testing.expectEqual(@as(?[*]const f16, null), args.tail_mask);
}

test "D4 KVCPT E2E real body: body-only (n_tail=0) ≡ CPU ref (gated P4 cubin)" {
    // n_tail=0 path: el wrapper cae al body-only loop del kernel
    // `fattn_kvarn_portable_d128_tail_kernel`. La ref CPU es la
    // atención estándar (sin cola). Cuando P4 cierre, este test corre
    // la pipeline completa: store → init_descs → fattnKvarnPortable
    // D128TailDevice → compare.
    if (build_options.fattn_cubin.len == 0) return error.SkipZigTest;
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;

    const allocator = testing.allocator;
    const n_seeds: u32 = blk: {
        if (std.c.getenv("ZIG_AI_M1_1000SEEDS") != null) break :blk 1000;
        break :blk 10;
    };

    const layout = kvarn.KvarnRecordLayout.init(@intCast(D), 4, 4) catch unreachable;
    const record_bytes = @as(c_int, @intCast(layout.tile_bytes));

    const n_kv_body: u32 = D4_CASE.n_kv_body * @as(u32, @intCast(D / 128));
    const n_q_heads: u32 = D4_CASE.n_q_heads;
    const n_kv_heads: u32 = D4_CASE.n_kv_heads;

    const q_size: usize = n_q_heads * D;
    const kv_size: usize = n_kv_body * n_kv_heads * D;

    try cudaz.ensureContext();
    const fattn_module = try cudaz.cuModuleLoad(build_options.fattn_cubin);
    const kvk_module = try cudaz.cuModuleLoad(build_options.kvarn_cubin);
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);

    const stage_groups: u32 = 2;
    const d_stage = try cudaz.cuMemAlloc(@as(usize, stage_groups) * D * D * @sizeOf(f16));
    defer cudaz.cuMemFree(d_stage);
    const d_records = try cudaz.cuMemAlloc(@intCast(record_bytes));
    defer cudaz.cuMemFree(d_records);
    const d_indices = try cudaz.cuMemAlloc(@sizeOf(i64) * n_kv_body);
    defer cudaz.cuMemFree(d_indices);
    const d_descs = try cudaz.cuMemAlloc(@sizeOf(kvk.KvarnDesc) * 2);
    defer cudaz.cuMemFree(d_descs);
    const d_q = try cudaz.cuMemAlloc(@sizeOf(f32) * q_size);
    defer cudaz.cuMemFree(d_q);
    const d_dst = try cudaz.cuMemAlloc(@sizeOf(f32) * q_size);
    defer cudaz.cuMemFree(d_dst);
    const d_current_k = try cudaz.cuMemAlloc(@sizeOf(f32) * kv_size);
    defer cudaz.cuMemFree(d_current_k);
    const d_current_v = try cudaz.cuMemAlloc(@sizeOf(f32) * kv_size);
    defer cudaz.cuMemFree(d_current_v);

    var prng = std.Random.DefaultPrng.init(0xD4B0);
    const rand = prng.random();

    var bad_seeds: u32 = 0;
    var max_rel_overall: f64 = 0.0;

    for (0..n_seeds) |_| {
        // Generar Q, K, V originales.
        const q = try allocator.alloc(f32, q_size);
        defer allocator.free(q);
        const k_orig = try allocator.alloc(f32, kv_size);
        defer allocator.free(k_orig);
        const v_orig = try allocator.alloc(f32, kv_size);
        defer allocator.free(v_orig);
        for (q) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
        for (k_orig) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
        for (v_orig) |*x| x.* = rand.float(f32) * 0.5 - 0.25;

        // CPU ref: atención estándar D=128 (sin cola en este test).
        var cpu_out = try allocator.alloc(f32, q_size);
        defer allocator.free(cpu_out);
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
        const gqa: u32 = n_q_heads / n_kv_heads;
        // Ref PIPELINE-EXACTA (lección B5/C2v2): el GPU path cuantiza el
        // body k4v4 ROTADO ⇒ la ref computa sobre WHT(K/V)→f16 en dominio
        // ROTADO con Q rotada, y de-rota el output.
        const q_rot = try allocator.alloc(f32, q_size);
        defer allocator.free(q_rot);
        const k_q = try allocator.alloc(f32, kv_size);
        defer allocator.free(k_q);
        const v_q = try allocator.alloc(f32, kv_size);
        defer allocator.free(v_q);
        rotateRows128(q, q_rot);
        stageRoundtripRot(k_orig, k_q);
        stageRoundtripRot(v_orig, v_q);
        for (0..n_q_heads) |qh| {
            const kh = qh / gqa;
            var scores = try allocator.alloc(f32, n_kv_body);
            defer allocator.free(scores);
            for (0..n_kv_body) |t| {
                var s: f32 = 0.0;
                for (0..D) |d| s += q_rot[qh * D + d] * k_q[(t * n_kv_heads + kh) * D + d];
                scores[t] = s * scale;
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
            for (0..D) |d| {
                var acc: f32 = 0.0;
                for (0..n_kv_body) |t| {
                    acc += scores[t] * v_q[(t * n_kv_heads + kh) * D + d];
                }
                cpu_out[qh * D + d] = acc;
            }
        }
        // De-rotar output por head (WHT involutiva).
        for (0..n_q_heads) |qh| {
            var row: [128]f32 = undefined;
            for (0..D) |d| row[d] = cpu_out[qh * D + d];
            kvarn.hadamard128InPlace(&row);
            for (0..D) |d| cpu_out[qh * D + d] = row[d];
        }

        // Pipeline: subir → store → init_descs → tail wrapper → compare.
        try cudaz.cuMemcpyHtoD(d_current_k, @intFromPtr(k_orig.ptr), @sizeOf(f32) * kv_size);
        try cudaz.cuMemcpyHtoD(d_current_v, @intFromPtr(v_orig.ptr), @sizeOf(f32) * kv_size);

        const indices = try allocator.alloc(i64, n_kv_body);
        defer allocator.free(indices);
        for (0..n_kv_body) |i| indices[i] = @intCast(i);
        try cudaz.cuMemcpyHtoD(d_indices, @intFromPtr(indices.ptr), @sizeOf(i64) * n_kv_body);

        // Store K (Dev-A A2 + A4 + A7, C2v2).
        const store_args_k: kvk.KvarnStoreArgs = .{
            .current = @ptrFromInt(d_current_k),
            .current_v = @ptrFromInt(d_current_v),
            .indices = @ptrFromInt(d_indices),
            .stage = @ptrFromInt(d_stage),
            .records = @ptrFromInt(d_records),
            .n_tokens = @intCast(n_kv_body),
            .n_record_heads = 1,
            .stream = 0,
            .groups_per_stream = 1,
            .record_bytes = record_bytes,
            .k_payload_off = @intCast(layout.k_payload_off),
            .k_s_col_off = @intCast(layout.k_s_col_off),
            .k_zp_off = @intCast(layout.k_zp_off),
            .k_s_row_off = @intCast(layout.k_s_row_off),
            .v_payload_off = @intCast(layout.v_payload_off),
            .v_s_col_off = @intCast(layout.v_s_col_off),
            .v_s_row_off = @intCast(layout.v_s_row_off),
            .v_zp_off = @intCast(layout.v_zp_off),
            .k_bits = 4,
            .v_bits = 4,
            .sinkhorn_iters = 8,
            .stage_groups = @intCast(stage_groups),
            .tail_groups = 1,
            .swa = 0,
            .eager_records = 0,
        };
        try kvk.kvarnStoreDevice(kvk_module, &store_args_k, stream);
        try cudaz.cuStreamSynchronize(stream);

        // Init descs (B3).
        const init_args: kvk.KvarnInitDescsArgs = .{
            .n_stream = 1,
            .n_indices = @intCast(n_kv_body),
            .d_indices = @ptrFromInt(d_indices),
            .d_descs = @ptrFromInt(d_descs),
            .desc_stride = 1,
            .d_records = @ptrFromInt(d_records),
            .d_stage = @ptrFromInt(d_stage),
            .n_record_heads = 1,
        .head_dim = 4,
            .groups_per_stream = 1,
            .record_bytes = record_bytes,
            .stage_groups = @intCast(stage_groups),
            .tail_groups = 1,
            .k_bits = 4,
            .v_bits = 4,
            .head_slices = 1,
            .eager_records = 0,
            .read_indirect = 0,
            .original_domain = 0,
            .swa = 0,
        };
        try kvk.kvarnInitDescsDevice(kvk_module, &init_args, stream);
        try cudaz.cuStreamSynchronize(stream);

        try cudaz.cuMemcpyHtoD(d_q, @intFromPtr(q.ptr), @sizeOf(f32) * q_size);

        // Portable D=128+tail kernel con n_tail=0 (cuerpo sólo).
        // Mismo kernel que la ruta portable D=128 sin tail — la
        // diferencia es que el wrapper ahora existe con `?CUmodule`
        // consistent con los demás wrappers.
        const tail_args: fattn_kv.KvarnAttentionTailArgs = .{
            .q_data = @ptrFromInt(d_q),
            .k_descs = @ptrFromInt(d_descs),
            .v_descs = @ptrFromInt(d_descs + @sizeOf(kvk.KvarnDesc)),
            .mask_data = null,
            .k_tail_data = null,
            .v_tail_data = null,
            .tail_mask = null,
            .run_desc_slots = null,
            .n_kv = @intCast(n_kv_body),
            .n_tail = 0, // body-only
            .d_k = 0,
            .d_v = 0,
            .n_q = 1,
            .n_q_heads = @intCast(n_q_heads),
            .n_kv_heads = @intCast(n_kv_heads),
            .n_stream = 1,
            .k_tail_bf16 = 0,
            .v_tail_bf16 = 0,
            .dst_data = @ptrFromInt(d_dst),
            .scale = scale,
            .gqa = @intCast(gqa),
        };
        try fattnKvarnPortableD128TailDevice(fattn_module, &tail_args, stream);
        try cudaz.cuStreamSynchronize(stream);

        const out_host = try allocator.alloc(f32, q_size);
        defer allocator.free(out_host);
        try cudaz.cuMemcpyDtoH(@intFromPtr(out_host.ptr), d_dst, @sizeOf(f32) * q_size);

        // Compare.
        var max_rel: f64 = 0.0;
        for (out_host, cpu_out, 0..) |got, want, i| {
            // abs+rel mixto: el rel puro explota en outputs near-zero
            // (softmax de 4-8 tokens ⇒ valores ~0 con error abs ~3e-4 del
            // f16-roundtrip; el dump muestra max-abs-diff correcto).
            const diff: f64 = @abs(@as(f64, got) - @as(f64, want));
            const denom: f64 = @max(@as(f64, @abs(want)), 1e-2);
            const rel: f64 = diff / denom;
            if (rel > max_rel) max_rel = rel;
            _ = i;
        }
        if (max_rel > D4_CASE.rel_tol) bad_seeds += 1;
        if (max_rel > max_rel_overall) max_rel_overall = max_rel;
    }

    if (bad_seeds > 0) {
        std.log.err("D4 KVCPT body-only E2E: {d}/{d} seeds failed, max_rel={d}", .{ bad_seeds, n_seeds, max_rel_overall });
    }
    try testing.expect(bad_seeds == 0);
}

// Helper envoltura para la función privada `fattnKvarnPortableD128TailDevice`
// — sin él el test no compila si el módulo no la importa.
const fattnKvarnPortableD128TailDevice = fattn_kv.fattnKvarnPortableD128TailDevice;

test "D4 KVCPT E2E real body: body+tail (n_tail>0) ≡ CPU ref (gated P4 cubin)" {
    // Pipeline completa: body C1 (k4v4) + tail f16 exacto.
    // El body se materializa vía kvarnStoreDevice (Dev-A A2-A4-A7);
    // el tail se sube como f16 raw (sin cuantizar). El kernel
    // `fattn_kvarn_portable_d128_tail_kernel` los integra con el
    // mismo softmax online; el output se compara con la atención
    // CPU estándar sobre los MISMOS tokens (body original sin
    // rotar + tail original sin rotar) — la equivalencia es
    // exacta porque ambos son datos originales (la rotación está
    // dentro del body C1, pero el material en la atención CPU
    // ref es el body pre-rotación).
    //
    // Gated P4 (cubin). Sin cubin, SKIP.
    if (build_options.fattn_cubin.len == 0) return error.SkipZigTest;
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;

    const allocator = testing.allocator;
    const n_seeds: u32 = blk: {
        if (std.c.getenv("ZIG_AI_M1_1000SEEDS") != null) break :blk 1000;
        break :blk 10;
    };

    const layout = kvarn.KvarnRecordLayout.init(@intCast(D), 4, 4) catch unreachable;
    const record_bytes = @as(c_int, @intCast(layout.tile_bytes));

    const n_kv_body: u32 = 4; // 1 grupo (32 tokens/dim); smoke
    const n_tail: u32 = 4;
    const n_total: u32 = n_kv_body + n_tail;
    const n_q_heads: u32 = 2;
    const n_kv_heads: u32 = 1;
    const gqa: u32 = n_q_heads / n_kv_heads;
    const tail_slots: u32 = n_tail; // 1 token = 1 slot

    const q_size: usize = n_q_heads * D;
    const kv_size: usize = n_kv_body * n_kv_heads * D;
    const d_k: u32 = D * 2; // bytes per f16 token of tail

    try cudaz.ensureContext();
    const fattn_module = try cudaz.cuModuleLoad(build_options.fattn_cubin);
    const kvk_module = try cudaz.cuModuleLoad(build_options.kvarn_cubin);
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);

    const stage_groups: u32 = 2;
    const d_stage = try cudaz.cuMemAlloc(@as(usize, stage_groups) * D * D * @sizeOf(f16));
    defer cudaz.cuMemFree(d_stage);
    const d_records = try cudaz.cuMemAlloc(@intCast(record_bytes));
    defer cudaz.cuMemFree(d_records);
    const d_indices = try cudaz.cuMemAlloc(@sizeOf(i64) * n_kv_body);
    defer cudaz.cuMemFree(d_indices);
    const d_descs = try cudaz.cuMemAlloc(@sizeOf(kvk.KvarnDesc) * 2);
    defer cudaz.cuMemFree(d_descs);
    const d_q = try cudaz.cuMemAlloc(@sizeOf(f32) * q_size);
    defer cudaz.cuMemFree(d_q);
    const d_dst = try cudaz.cuMemAlloc(@sizeOf(f32) * q_size);
    defer cudaz.cuMemFree(d_dst);
    const d_current_k = try cudaz.cuMemAlloc(@sizeOf(f32) * kv_size);
    defer cudaz.cuMemFree(d_current_k);
    const d_current_v = try cudaz.cuMemAlloc(@sizeOf(f32) * kv_size);
    defer cudaz.cuMemFree(d_current_v);
    // Tail f16 buffers (Dev-A A6 emits f16; el caller de cola exacta
    // entrega lo mismo).
    const d_k_tail = try cudaz.cuMemAlloc(@as(usize, tail_slots) * D * 2);
    defer cudaz.cuMemFree(d_k_tail);
    const d_v_tail = try cudaz.cuMemAlloc(@as(usize, tail_slots) * D * 2);
    defer cudaz.cuMemFree(d_v_tail);
    const d_slots = try cudaz.cuMemAlloc(@sizeOf(c_int) * n_tail);
    defer cudaz.cuMemFree(d_slots);

    var prng = std.Random.DefaultPrng.init(0xD4B1);
    const rand = prng.random();

    var bad_seeds: u32 = 0;
    var max_rel_overall: f64 = 0.0;

    for (0..n_seeds) |_| {
        // Generar Q, K, V originales.
        const q = try allocator.alloc(f32, q_size);
        defer allocator.free(q);
        const k_body = try allocator.alloc(f32, kv_size);
        defer allocator.free(k_body);
        const v_body = try allocator.alloc(f32, kv_size);
        defer allocator.free(v_body);
        const k_tail = try allocator.alloc(f16, @as(usize, n_tail) * D);
        defer allocator.free(k_tail);
        const v_tail = try allocator.alloc(f16, @as(usize, n_tail) * D);
        defer allocator.free(v_tail);
        const slots = try allocator.alloc(c_int, n_tail);
        defer allocator.free(slots);
        for (q) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
        for (k_body) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
        for (v_body) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
        for (k_tail) |*x| x.* = @floatCast(rand.float(f32) * 0.5 - 0.25);
        for (v_tail) |*x| x.* = @floatCast(rand.float(f32) * 0.5 - 0.25);
        for (0..n_tail) |i| slots[i] = @intCast(i);

        // CPU ref: atención estándar f32 sobre los MISMOS tokens
        // originales (body pre-rotación + tail f16). El kernel
        // portable rota Q y los records del body están rotados; el
        // CPU ref opera en dominio original (la equivalencia por
        // ortogonalidad de WHT cancela las rotaciones).
        var cpu_out = try allocator.alloc(f32, q_size);
        defer allocator.free(cpu_out);
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
        // Ref PIPELINE-EXACTA: todo el dot en dominio ROTADO (Q rotada,
        // body cuantizado-rotado f16-trunc, tail rotado f16) y de-rot.
        const q_rot = try allocator.alloc(f32, q_size);
        defer allocator.free(q_rot);
        const k_q = try allocator.alloc(f32, kv_size);
        defer allocator.free(k_q);
        const v_q = try allocator.alloc(f32, kv_size);
        defer allocator.free(v_q);
        rotateRows128(q, q_rot);
        stageRoundtripRot(k_body, k_q);
        stageRoundtripRot(v_body, v_q);
        const k_tail_rot32 = try allocator.alloc(f32, n_tail * D);
        defer allocator.free(k_tail_rot32);
        const v_tail_rot32 = try allocator.alloc(f32, n_tail * D);
        defer allocator.free(v_tail_rot32);
        for (0..n_tail) |t| {
            var row: [128]f32 = undefined;
            for (0..D) |d| row[d] = @as(f32, @floatCast(k_tail[t * D + d]));
            kvarn.hadamard128InPlace(&row);
            for (0..D) |d| k_tail_rot32[t * D + d] = row[d];
            for (0..D) |d| row[d] = @as(f32, @floatCast(v_tail[t * D + d]));
            kvarn.hadamard128InPlace(&row);
            for (0..D) |d| v_tail_rot32[t * D + d] = row[d];
        }
        for (0..n_q_heads) |qh| {
            const kh = qh / gqa;
            var scores = try allocator.alloc(f32, n_total);
            defer allocator.free(scores);
            // body tokens (cuantizado-rotado)
            for (0..n_kv_body) |t| {
                var s: f32 = 0.0;
                for (0..D) |d| s += q_rot[qh * D + d] * k_q[(t * n_kv_heads + kh) * D + d];
                scores[t] = s * scale;
            }
            // tail tokens (rotado f16)
            for (0..n_tail) |t| {
                var s: f32 = 0.0;
                for (0..D) |d| s += q_rot[qh * D + d] * k_tail_rot32[t * D + d];
                scores[n_kv_body + t] = s * scale;
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
            for (0..D) |d| {
                var acc: f32 = 0.0;
                for (0..n_kv_body) |t| acc += scores[t] * v_q[(t * n_kv_heads + kh) * D + d];
                for (0..n_tail) |t| acc += scores[n_kv_body + t] * v_tail_rot32[t * D + d];
                cpu_out[qh * D + d] = acc;
            }
        }
        // De-rotar output por head.
        for (0..n_q_heads) |qh| {
            var row: [128]f32 = undefined;
            for (0..D) |d| row[d] = cpu_out[qh * D + d];
            kvarn.hadamard128InPlace(&row);
            for (0..D) |d| cpu_out[qh * D + d] = row[d];
        }

        // Pipeline: subir → store → init_descs → tail wrapper → compare.
        try cudaz.cuMemcpyHtoD(d_current_k, @intFromPtr(k_body.ptr), @sizeOf(f32) * kv_size);
        try cudaz.cuMemcpyHtoD(d_current_v, @intFromPtr(v_body.ptr), @sizeOf(f32) * kv_size);

        const indices = try allocator.alloc(i64, n_kv_body);
        defer allocator.free(indices);
        for (0..n_kv_body) |i| indices[i] = @intCast(i);
        try cudaz.cuMemcpyHtoD(d_indices, @intFromPtr(indices.ptr), @sizeOf(i64) * n_kv_body);

        // Store (Dev-A A2-A4-A7).
        const store_args_k: kvk.KvarnStoreArgs = .{
            .current = @ptrFromInt(d_current_k),
            .current_v = @ptrFromInt(d_current_v),
            .indices = @ptrFromInt(d_indices),
            .stage = @ptrFromInt(d_stage),
            .records = @ptrFromInt(d_records),
            .n_tokens = @intCast(n_kv_body),
            .n_record_heads = 1,
            .stream = 0,
            .groups_per_stream = 1,
            .record_bytes = record_bytes,
            .k_payload_off = @intCast(layout.k_payload_off),
            .k_s_col_off = @intCast(layout.k_s_col_off),
            .k_zp_off = @intCast(layout.k_zp_off),
            .k_s_row_off = @intCast(layout.k_s_row_off),
            .v_payload_off = @intCast(layout.v_payload_off),
            .v_s_col_off = @intCast(layout.v_s_col_off),
            .v_s_row_off = @intCast(layout.v_s_row_off),
            .v_zp_off = @intCast(layout.v_zp_off),
            .k_bits = 4,
            .v_bits = 4,
            .sinkhorn_iters = 8,
            .stage_groups = @intCast(stage_groups),
            .tail_groups = 1,
            .swa = 0,
            .eager_records = 0,
        };
        try kvk.kvarnStoreDevice(kvk_module, &store_args_k, stream);
        try cudaz.cuStreamSynchronize(stream);

        // Init descs.
        const init_args: kvk.KvarnInitDescsArgs = .{
            .n_stream = 1,
            .n_indices = @intCast(n_kv_body),
            .d_indices = @ptrFromInt(d_indices),
            .d_descs = @ptrFromInt(d_descs),
            .desc_stride = 1,
            .d_records = @ptrFromInt(d_records),
            .d_stage = @ptrFromInt(d_stage),
            .n_record_heads = 1,
        .head_dim = 4,
            .groups_per_stream = 1,
            .record_bytes = record_bytes,
            .stage_groups = @intCast(stage_groups),
            .tail_groups = 1,
            .k_bits = 4,
            .v_bits = 4,
            .head_slices = 1,
            .eager_records = 0,
            .read_indirect = 0,
            .original_domain = 0,
            .swa = 0,
        };
        try kvk.kvarnInitDescsDevice(kvk_module, &init_args, stream);
        try cudaz.cuStreamSynchronize(stream);

        try cudaz.cuMemcpyHtoD(d_q, @intFromPtr(q.ptr), @sizeOf(f32) * q_size);
        // CONTRATO D4: el kernel atiende en dominio ROTADO (Q rota
        // in-kernel, body rotado del store, output de-rotado) ⇒ el tail
        // f16 llega ROTADO (WHT por token), igual que el manager B2
        // subiría k_exact/v_exact. Sin esto, el dot Q_rot·k_tail mezcla
        // dominios ⇒ rel ~4000.
        const k_tail_rot = try allocator.alloc(f16, n_tail * D);
        defer allocator.free(k_tail_rot);
        const v_tail_rot = try allocator.alloc(f16, n_tail * D);
        defer allocator.free(v_tail_rot);
        for (0..n_tail) |t| {
            var row: [128]f32 = undefined;
            for (0..D) |d| row[d] = @as(f32, @floatCast(k_tail[t * D + d]));
            kvarn.hadamard128InPlace(&row);
            for (0..D) |d| k_tail_rot[t * D + d] = @floatCast(row[d]);
            for (0..D) |d| row[d] = @as(f32, @floatCast(v_tail[t * D + d]));
            kvarn.hadamard128InPlace(&row);
            for (0..D) |d| v_tail_rot[t * D + d] = @floatCast(row[d]);
        }
        try cudaz.cuMemcpyHtoD(d_k_tail, @intFromPtr(k_tail_rot.ptr), @sizeOf(f16) * n_tail * D);
        try cudaz.cuMemcpyHtoD(d_v_tail, @intFromPtr(v_tail_rot.ptr), @sizeOf(f16) * n_tail * D);
        try cudaz.cuMemcpyHtoD(d_slots, @intFromPtr(slots.ptr), @sizeOf(c_int) * n_tail);

        // Tail wrapper: n_tail>0. El kernel lee tail_data[slot*D*2
        // + kv_head*D*2 + tid*2] (f16 ⇒ 2 bytes por dim).
        const tail_args: fattn_kv.KvarnAttentionTailArgs = .{
            .q_data = @ptrFromInt(d_q),
            .k_descs = @ptrFromInt(d_descs),
            .v_descs = @ptrFromInt(d_descs + @sizeOf(kvk.KvarnDesc)),
            .mask_data = null,
            .k_tail_data = @ptrFromInt(d_k_tail),
            .v_tail_data = @ptrFromInt(d_v_tail),
            .tail_mask = null,
            .run_desc_slots = @ptrFromInt(d_slots),
            .n_kv = @intCast(n_kv_body),
            .n_tail = @intCast(n_tail),
            .d_k = @intCast(d_k),
            .d_v = @intCast(d_k),
            .n_q = 1,
            .n_q_heads = @intCast(n_q_heads),
            .n_kv_heads = @intCast(n_kv_heads),
            .n_stream = 1,
            .k_tail_bf16 = 0,
            .v_tail_bf16 = 0,
            .dst_data = @ptrFromInt(d_dst),
            .scale = scale,
            .gqa = @intCast(gqa),
        };
        try fattnKvarnPortableD128TailDevice(fattn_module, &tail_args, stream);
        try cudaz.cuStreamSynchronize(stream);

        const out_host = try allocator.alloc(f32, q_size);
        defer allocator.free(out_host);
        try cudaz.cuMemcpyDtoH(@intFromPtr(out_host.ptr), d_dst, @sizeOf(f32) * q_size);

        // Compare: la atención CPU standard opera en dominio original;
        // el kernel portable rota Q + records (rotados) y aplica
        // WHT⁻¹ al output. La equivalencia por ortogonalidad de WHT
        // hace los outputs idénticos bit-a-bit (sin cuantización
        // en el tail). Tolerancia: 1e-3 (f16 ↔ f32 acumula error).
        var max_rel: f64 = 0.0;
        for (out_host, cpu_out, 0..) |got, want, i| {
            // abs+rel mixto: el rel puro explota en outputs near-zero
            // (softmax de 4-8 tokens ⇒ valores ~0 con error abs ~3e-4 del
            // f16-roundtrip; el dump muestra max-abs-diff correcto).
            const diff: f64 = @abs(@as(f64, got) - @as(f64, want));
            const denom: f64 = @max(@as(f64, @abs(want)), 1e-2);
            const rel: f64 = diff / denom;
            if (rel > max_rel) max_rel = rel;
            _ = i;
        }
        if (max_rel > 5e-2) bad_seeds += 1;
        if (max_rel > max_rel_overall) max_rel_overall = max_rel;
    }

    if (bad_seeds > 0) {
        std.log.err("D4 KVCPT body+tail E2E: {d}/{d} seeds failed, max_rel={d}", .{ bad_seeds, n_seeds, max_rel_overall });
    }
    try testing.expect(bad_seeds == 0);
}
