//! Lane-b1 B4 iter 5+ — portable D=512 portable FA E2E test.
//!
//! Spec (TODO_B1_DEV_B §B4 + D2): portable FA también en D=512
//! (4 slices de 128) con cross-slice WHT-4 (closed-form). Gated D2
//! (lane-b2 CPU ref del cross-slice ratifica el patrón).
//!
//! STATUS (B4 iter 5+): el kernel `fattn_kvarn_portable_d512_kernel`
//! está escrito (en mi worktree) y compila. El wrapper Zig
//! `fattnKvarnPortableD512Device` está añadido (iter 11). El test
//! E2E con materialized K/V (Dev-A A2/A7/A6) gated por cubin: usa
//! el mismo flujo que B5 iter 2 (store + init_descs + portable FA
//! + CPU ref) con D=512 y head_slices=4.
//!
//! CPU ref: atención estándar f32 sobre D=512 (sin WHT — la
//! equivalencia por ortogonalidad de WHT cancela las rotaciones).
//! El WHT-128 de Q lo aplica el kernel in-place; el WHT⁻¹ al
//! output vuelve al dominio original. La cruz de slices (×1/4)
//! también es self-inverse por la simetría del Hadamard-4
//! closed-form.
//!
//! Gating: SKIP si no hay cubin (`fattn_cubin` + `kvarn_cubin`) y
//! si ZIG_AI_KVARN_D2_UNLOCK=1. N_SEEDS=10 default; 1000 con
//! ZIG_AI_M1_1000SEEDS=1.

const std = @import("std");
const testing = std.testing;
const build_options = @import("build_options");
const cudaz = @import("cudaz");
const kvk = @import("kvarn_kernels");
const fattn_kv = @import("fattn_kvarn");
const kvarn = @import("kv_cache").kvarn;

const D: usize = 512;
const D_U32: u32 = 512;

const Case = struct {
    n_q: u32,
    n_kv: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    n_stream: u32,
    gqa: u32,
    rel_tol: f64,
};

const CASE: Case = .{
    .n_q = 1,
    .n_kv = 128, // 1 grupo
    .n_q_heads = 4,
    .n_kv_heads = 1,
    .n_stream = 1,
    .gqa = 4,
    .rel_tol = 1e-3, // D=512 cross-slice es reciente; tightening posterior
};

const N_SEEDS_DEFAULT: u32 = 10;
const N_SEEDS_FULL: u32 = 1000;

fn cpuAttention(
    q: []const f32,
    k: []const f32,
    v: []const f32,
    n_kv: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    gqa: u32,
    scale: f32,
) ![]f32 {
    const out_size: usize = @as(usize, n_q_heads) * D;
    const output = try testing.allocator.alloc(f32, out_size);
    errdefer testing.allocator.free(output);

    var qh: u32 = 0;
    while (qh < n_q_heads) : (qh += 1) {
        const kh = qh / gqa;
        var scores = try testing.allocator.alloc(f32, n_kv);
        defer testing.allocator.free(scores);
        for (0..n_kv) |t| {
            var s: f32 = 0.0;
            for (0..D) |d| {
                s += q[qh * D + d] * k[(t * n_kv_heads + kh) * D + d];
            }
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
            for (0..n_kv) |t| {
                acc += scores[t] * v[(t * n_kv_heads + kh) * D + d];
            }
            output[qh * D + d] = acc;
        }
    }
    return output;
}

test "B4 iter 5+: portable D=512 ≡ CPU ref con materialized K/V (gated P4 cubin + D2)" {
    if (build_options.fattn_cubin.len == 0) return error.SkipZigTest;
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;
    if (std.c.getenv("ZIG_AI_KVARN_D2_UNLOCK") == null) return error.SkipZigTest;
    // Gate D2 (PLAN_B1 §D2): el store (A2) aún NO aplica cross-slice WHT
    // (solo intra-128) mientras el portable rota Q intra+cross ⇒ dominios
    // incompatibles hasta que B2 ratifique cross-slice en el CPU ref y el
    // store lo implemente. El E2E real queda gated tras D2.
    if (std.c.getenv("ZIG_AI_KVARN_D2_UNLOCK") == null) return error.SkipZigTest;

    const allocator = testing.allocator;
    const n_seeds = blk: {
        if (std.c.getenv("ZIG_AI_M1_1000SEEDS") != null) break :blk N_SEEDS_FULL;
        break :blk N_SEEDS_DEFAULT;
    };

    const layout = kvarn.KvarnRecordLayout.init(D_U32, 4, 4) catch unreachable;
    const record_bytes = @as(c_int, @intCast(layout.tile_bytes));

    const q_size: usize = @as(usize, CASE.n_q) * @as(usize, CASE.n_q_heads) * D;
    const kv_size: usize = @as(usize, CASE.n_kv) * @as(usize, CASE.n_kv_heads) * D;

    try cudaz.ensureContext();
    const fattn_module = try cudaz.cuModuleLoad(build_options.fattn_cubin);
    const kvk_module = try cudaz.cuModuleLoad(build_options.kvarn_cubin);
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);

    const stage_groups: u32 = 2;
    // D=512: stage es 2 slices × 128 × 1 head × 128 × 2B (el stage no
    // cambia con head_slices; las slices viven en los records).
    const stage_size: usize = @as(usize, stage_groups) * 128 * 1 * 128 * @sizeOf(f16);
    const d_stage = try cudaz.cuMemAlloc(stage_size);
    defer cudaz.cuMemFree(d_stage);
    const d_records = try cudaz.cuMemAlloc(@intCast(record_bytes));
    defer cudaz.cuMemFree(d_records);
    const d_indices = try cudaz.cuMemAlloc(@sizeOf(i64) * CASE.n_kv);
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

    var prng = std.Random.DefaultPrng.init(0xD512);
    const rand = prng.random();

    var max_rel_overall: f64 = 0.0;
    var bad_seeds: usize = 0;
    var seed_idx: u32 = 0;
    while (seed_idx < n_seeds) : (seed_idx += 1) {
        const q = try allocator.alloc(f32, q_size);
        defer allocator.free(q);
        const k_orig = try allocator.alloc(f32, kv_size);
        defer allocator.free(k_orig);
        const v_orig = try allocator.alloc(f32, kv_size);
        defer allocator.free(v_orig);
        for (q) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
        for (k_orig) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
        for (v_orig) |*x| x.* = rand.float(f32) * 0.5 - 0.25;

        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
        const cpu_out = try cpuAttention(q, k_orig, v_orig, CASE.n_kv, CASE.n_q_heads, CASE.n_kv_heads, CASE.gqa, scale);
        defer allocator.free(cpu_out);

        try cudaz.cuMemcpyHtoD(d_current_k, @intFromPtr(k_orig.ptr), @sizeOf(f32) * kv_size);
        try cudaz.cuMemcpyHtoD(d_current_v, @intFromPtr(v_orig.ptr), @sizeOf(f32) * kv_size);

        const indices = try allocator.alloc(i64, CASE.n_kv);
        defer allocator.free(indices);
        for (0..CASE.n_kv) |i| indices[i] = @intCast(i);
        try cudaz.cuMemcpyHtoD(d_indices, @intFromPtr(indices.ptr), @sizeOf(i64) * CASE.n_kv);

        // Store K (A2 + A4 + A7 + C2v2).
        const store_args_k: kvk.KvarnStoreArgs = .{
            .current = @ptrFromInt(d_current_k),
            .current_v = @ptrFromInt(d_current_v),
            .indices = @ptrFromInt(d_indices),
            .stage = @ptrFromInt(d_stage),
            .records = @ptrFromInt(d_records),
            .n_tokens = @intCast(CASE.n_kv),
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

        // Init descs (B3) — D=512 ⇒ head_slices=4.
        const init_args: kvk.KvarnInitDescsArgs = .{
            .n_stream = 1,
            .n_indices = @intCast(CASE.n_kv),
            .d_indices = @ptrFromInt(d_indices),
            .d_descs = @ptrFromInt(d_descs),
            .desc_stride = 1,
            .d_records = @ptrFromInt(d_records),
            .d_stage = @ptrFromInt(d_stage),
            .n_record_heads = 1,
        .head_dim = 512,
            .groups_per_stream = 1,
            .record_bytes = record_bytes,
            .stage_groups = @intCast(stage_groups),
            .tail_groups = 1,
            .k_bits = 4,
            .v_bits = 4,
            .head_slices = 4, // D=512 = 4 slices
            .eager_records = 0,
            .read_indirect = 0,
            .original_domain = 0,
            .swa = 0,
        };
        try kvk.kvarnInitDescsDevice(kvk_module, &init_args, stream);
        try cudaz.cuStreamSynchronize(stream);

        try cudaz.cuMemcpyHtoD(d_q, @intFromPtr(q.ptr), @sizeOf(f32) * q_size);

        // Portable D=512 — usa extern `fattn_kvarn_portable_d512_kernel`
        // vía `fattnKvarnPortableD512Device` (iter 11).
        var attn_args: fattn_kv.KvarnAttentionArgs = .{
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
        _ = try fattn_kv.fattnKvarnPortableD512Device(fattn_module, &attn_args, stream);
        try cudaz.cuStreamSynchronize(stream);

        const out_host = try allocator.alloc(f32, q_size);
        defer allocator.free(out_host);
        try cudaz.cuMemcpyDtoH(@intFromPtr(out_host.ptr), d_dst, @sizeOf(f32) * q_size);

        var max_rel: f64 = 0.0;
        var bad: usize = 0;
        for (out_host, cpu_out, 0..) |got, want, i| {
            const denom: f64 = @max(@as(f64, @abs(want)), 1e-6);
            const rel: f64 = @as(f64, @abs(got - want)) / denom;
            if (rel > max_rel) max_rel = rel;
            if (rel > CASE.rel_tol) {
                bad += 1;
                if (bad < 10) std.log.err("B4 D=512 mismatch @{d}: got={d} want={d} rel={d}", .{ i, got, want, rel });
            }
        }
        if (max_rel > max_rel_overall) max_rel_overall = max_rel;
        if (bad > 0) bad_seeds += 1;
    }

    if (bad_seeds > 0) {
        std.log.err("B4 D=512: {d}/{d} seeds failed, max_rel={d}", .{ bad_seeds, n_seeds, max_rel_overall });
    }
    try testing.expect(bad_seeds == 0);
}

test "B4 iter 5+ CPU ref sanity: D=512, Q=K=V=0.1 ⇒ output=0.1 (sin GPU)" {
    const allocator = testing.allocator;
    const n_q_heads: u32 = 2;
    const n_kv: u32 = 8;
    const n_kv_heads: u32 = 1;
    const gqa: u32 = 2;
    const q = try allocator.alloc(f32, n_q_heads * D);
    defer allocator.free(q);
    const k = try allocator.alloc(f32, n_kv * n_kv_heads * D);
    defer allocator.free(k);
    const v = try allocator.alloc(f32, n_kv * n_kv_heads * D);
    defer allocator.free(v);
    for (q) |*x| x.* = 0.1;
    for (k) |*x| x.* = 0.1;
    for (v) |*x| x.* = 0.1;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    const out = try cpuAttention(q, k, v, n_kv, n_q_heads, n_kv_heads, gqa, scale);
    defer allocator.free(out);
    for (out) |v_| {
        try testing.expectApproxEqAbs(@as(f32, 0.1), v_, 1e-5);
    }
}
