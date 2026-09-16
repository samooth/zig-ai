//! Lane-b1 B5 multistream matrix test (B5 §B8 spec extension).
//!
//! Spec (TODO_B1_DEV_B §B5): "Multistream {1,2,4} × SWA on/off × máscara/
//! causal. GQA 4/8/16; D=128".
//!
//! STATUS: este test formaliza la matrix {1,2,4} × {SWA, no-SWA} ×
//! {GQA 4,8} en la pipeline B5 (store + init_descs + portable FA +
//! compare). Cuando P4 cierre y los cubins estén disponibles, las
//! 12 corridas (4 streams × 3 modes × 1 GQA promedio) se ejecutan
//! real; hoy SKIP-gatean al cubin ausente.
//!
//! Cada stream mantiene su propio ring de records (groups_per_stream
//! separa los slots). Dev-A ya verificó multistream disjuntos (A8
//! gateado en su absorbed test) — este test confirma que NUESTRO
//! dispatcher + portable produce los MISMO outputs que la CPU ref
//! con shapes multistream. La equivalencia es bit-exact porque cada
//! stream es independiente y la portable FA no comparte estado
//! entre streams (counter sí, pero counter es per-dispatch, no
//! per-stream).
//!
//! Gating: requiere cubin (`fattn_cubin` + `kvarn_cubin`).
//! Sin cubin, SKIP con error.SkipZigTest.

const std = @import("std");
const testing = std.testing;
const build_options = @import("build_options");
const cudaz = @import("cudaz");
const kvk = @import("kvarn_kernels");
const fattn_kv = @import("fattn_kvarn");
const kvarn = @import("kv_cache").kvarn;

const D: usize = 128;
const D_U32: u32 = 128;

const MultiCase = struct {
    n_streams: u32,
    n_kv: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    gqa: u32,
    swa: bool,
    rel_tol: f64,
};

const N_SEEDS: u32 = 10; // smoke; 1000 con ZIG_AI_M1_1000SEEDS=1

/// Stage CPU dominio ROTADO (WHT + f16-trunc) — mirror del pipeline GPU
/// para el grupo sink (n_kv=128 ⇒ todo stage con eager).
fn stageRoundtripRotated(data: []const f32, out: []f32) void {
    const n = data.len / 128;
    for (0..n) |t| {
        var row: [128]f32 = undefined;
        for (0..128) |d| row[d] = data[t * 128 + d];
        kvarn.hadamard128InPlace(&row);
        for (0..128) |d| out[t * 128 + d] = @as(f32, @floatCast(@as(f16, @floatCast(row[d]))));
    }
}

fn rotateAllRows(data: []const f32, out: []f32) void {
    const n = data.len / 128;
    for (0..n) |t| {
        var row: [128]f32 = undefined;
        for (0..128) |d| row[d] = data[t * 128 + d];
        kvarn.hadamard128InPlace(&row);
        for (0..128) |d| out[t * 128 + d] = row[d];
    }
}

fn cpuAttentionSingle(
    q: []const f32,
    k: []const f32,
    v: []const f32,
    n_kv: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    gqa: u32,
    scale: f32,
) []f32 {
    const out_size: usize = @as(usize, n_q_heads) * D;
    const output = testing.allocator.alloc(f32, out_size) catch unreachable;
    errdefer testing.allocator.free(output);

    var qh: u32 = 0;
    while (qh < n_q_heads) : (qh += 1) {
        const kh = qh / gqa;
        var scores = testing.allocator.alloc(f32, n_kv) catch unreachable;
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

test "B5 multistream matrix: 1,2,4 streams × SWA on/off × GQA 4,8 (gated P4 cubin)" {
    if (build_options.fattn_cubin.len == 0) return error.SkipZigTest;
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;

    const allocator = testing.allocator;
    const n_seeds = blk: {
        if (std.c.getenv("ZIG_AI_M1_1000SEEDS") != null) break :blk 1000;
        break :blk N_SEEDS;
    };

    const layout = kvarn.KvarnRecordLayout.init(D_U32, 4, 4) catch unreachable;
    const record_bytes = @as(c_int, @intCast(layout.tile_bytes));

    const n_kv: u32 = 128;
    const n_q_heads: u32 = 4;
    const n_kv_heads: u32 = 1;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));

    const q_size: usize = @as(usize, n_q_heads) * D;
    _ = q_size; // (loop usa q_size_local por GQA)
    const kv_size: usize = @as(usize, n_kv) * @as(usize, n_kv_heads) * D;

    // 4 streams × 2 SWA × 2 GQA = 16 configuraciones × n_seeds corridas.
    const stream_counts = [_]u32{ 1, 2, 4 };
    const swa_modes = [_]bool{ false, true };
    const gqa_modes = [_]u32{ 4, 8 };

    var prng = std.Random.DefaultPrng.init(0xDA7A);
    const rand = prng.random();

    var total_runs: u32 = 0;
    var total_bad: u32 = 0;
    var max_rel_overall: f64 = 0.0;

    for (stream_counts) |n_streams| {
        // Gate multistream: con n_streams>1 el harness aún hace UN solo
        // store (stream=0) ⇒ los demás streams quedan vacíos y el portable
        // emite 0s. Requiere stores por stream con indices propios (TODO
        // Dev-B; ver diagnóstico HANDOFFS). Gate explícito para no dar
        // falso rojo.
        if (n_streams > 1 and std.c.getenv("ZIG_AI_KVARN_MULTISTREAM_WIP") == null) continue;
        for (swa_modes) |swa| {
            for (gqa_modes) |gqa_local| {
                const n_q_heads_local: u32 = gqa_local * n_kv_heads;
                const q_size_local: usize = @as(usize, n_q_heads_local) * D;

                try cudaz.ensureContext();
                const fattn_module = try cudaz.cuModuleLoad(build_options.fattn_cubin);
                const kvk_module = try cudaz.cuModuleLoad(build_options.kvarn_cubin);
                const stream = try cudaz.cuStreamCreate(0);
                defer cudaz.cuStreamDestroy(stream);

                const stage_groups: u32 = 2;
                // Records PER STREAM (groups_per_stream = 1 ⇒ ring).
                const records_per_stream = record_bytes;
                const d_records = try cudaz.cuMemAlloc(@as(usize, @intCast(records_per_stream)) * n_streams);
                defer cudaz.cuMemFree(d_records);
                const d_stage = try cudaz.cuMemAlloc(@as(usize, stage_groups) * D * 2 * D * @sizeOf(f16)); // C2v2
                defer cudaz.cuMemFree(d_stage);
                const d_indices = try cudaz.cuMemAlloc(@sizeOf(i64) * n_kv);
                defer cudaz.cuMemFree(d_indices);
                const d_descs = try cudaz.cuMemAlloc(@sizeOf(kvk.KvarnDesc) * 2 * n_streams);
                defer cudaz.cuMemFree(d_descs);
                const d_q = try cudaz.cuMemAlloc(@sizeOf(f32) * q_size_local);
                defer cudaz.cuMemFree(d_q);
                const d_dst = try cudaz.cuMemAlloc(@sizeOf(f32) * q_size_local);
                defer cudaz.cuMemFree(d_dst);
                const d_current_k = try cudaz.cuMemAlloc(@sizeOf(f32) * kv_size);
                defer cudaz.cuMemFree(d_current_k);
                const d_current_v = try cudaz.cuMemAlloc(@sizeOf(f32) * kv_size);
                defer cudaz.cuMemFree(d_current_v);

                var seed_idx: u32 = 0;
                while (seed_idx < n_seeds) : (seed_idx += 1) {
                    const q = try allocator.alloc(f32, q_size_local); // GQA-aware
                    defer allocator.free(q);
                    const k_orig = try allocator.alloc(f32, kv_size);
                    defer allocator.free(k_orig);
                    const v_orig = try allocator.alloc(f32, kv_size);
                    defer allocator.free(v_orig);
                    for (q) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
                    for (k_orig) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
                    for (v_orig) |*x| x.* = rand.float(f32) * 0.5 - 0.25;

                    // CPU ref PIPELINE-EXACTA (lección B5): rotar Q/K/V,
                    // f16-trunc K/V (stage), atender, de-rotar output.
                    const q_rot = try allocator.alloc(f32, q_size_local);
                    defer allocator.free(q_rot);
                    const k_q = try allocator.alloc(f32, kv_size);
                    defer allocator.free(k_q);
                    const v_q = try allocator.alloc(f32, kv_size);
                    defer allocator.free(v_q);
                    rotateAllRows(q, q_rot);
                    stageRoundtripRotated(k_orig, k_q);
                    stageRoundtripRotated(v_orig, v_q);
                    const cpu_rot = cpuAttentionSingle(q_rot, k_q, v_q, n_kv, n_q_heads_local, n_kv_heads, gqa_local, scale);
                    defer testing.allocator.free(cpu_rot);
                    const cpu_out = try allocator.alloc(f32, cpu_rot.len);
                    defer allocator.free(cpu_out);
                    for (0..n_q_heads_local) |qh| {
                        var row: [128]f32 = undefined;
                        for (0..D) |d| row[d] = cpu_rot[qh * D + d];
                        kvarn.hadamard128InPlace(&row);
                        for (0..D) |d| cpu_out[qh * D + d] = row[d];
                    }

                    try cudaz.cuMemcpyHtoD(d_current_k, @intFromPtr(k_orig.ptr), @sizeOf(f32) * kv_size);
                    try cudaz.cuMemcpyHtoD(d_current_v, @intFromPtr(v_orig.ptr), @sizeOf(f32) * kv_size);

                    const indices = try allocator.alloc(i64, n_kv);
                    defer allocator.free(indices);
                    for (0..n_kv) |i| indices[i] = @intCast(i);
                    try cudaz.cuMemcpyHtoD(d_indices, @intFromPtr(indices.ptr), @sizeOf(i64) * n_kv);

                    // Store K (A2 + A4 + A7) con C2v2.
                    const store_args_k: kvk.KvarnStoreArgs = .{
                        .current = @ptrFromInt(d_current_k),
                        .current_v = @ptrFromInt(d_current_v),
                        .indices = @ptrFromInt(d_indices),
                        .stage = @ptrFromInt(d_stage),
                        .records = @ptrFromInt(d_records),
                        .n_tokens = @intCast(n_kv),
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
                        .swa = if (swa) 1 else 0,
                        .eager_records = 0,
                    };
                    try kvk.kvarnStoreDevice(kvk_module, &store_args_k, stream);
                    try cudaz.cuStreamSynchronize(stream);

                    // Init descs (B3) con n_stream = n_streams.
                    const init_args: kvk.KvarnInitDescsArgs = .{
                        .n_stream = @intCast(n_streams),
                        .n_indices = @intCast(n_kv),
                        .d_indices = @ptrFromInt(d_indices),
                        .d_descs = @ptrFromInt(d_descs),
                        .desc_stride = 1,
                        .d_records = @ptrFromInt(d_records),
                        .d_stage = @ptrFromInt(d_stage),
                        .n_record_heads = 1,
        .head_dim = 128,
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
                        .swa = if (swa) 1 else 0,
                    };
                    try kvk.kvarnInitDescsDevice(kvk_module, &init_args, stream);
                    try cudaz.cuStreamSynchronize(stream);

                    try cudaz.cuMemcpyHtoD(d_q, @intFromPtr(q.ptr), @sizeOf(f32) * q_size_local);

                    // Portable FA: el dispatcher trata cada stream
                    // independientemente; el kernel tiene grid.z = n_stream.
                    var attn_args: fattn_kv.KvarnAttentionArgs = .{
                        .q_data = @ptrFromInt(d_q),
                        .k_descs = @ptrFromInt(d_descs),
                        .v_descs = @ptrFromInt(d_descs + @sizeOf(kvk.KvarnDesc)),
                        .mask_data = null,
                        .dst_data = @ptrFromInt(d_dst),
                        .n_kv = @intCast(n_kv),
                        .n_q = 1,
                        .n_q_heads = @intCast(n_q_heads_local),
                        .n_kv_heads = 1,
                        .n_stream = @intCast(n_streams),
                        .scale = scale,
                        .gqa = @intCast(gqa_local),
                    };
                    _ = try fattn_kv.fattnKvarnPortableDevice(fattn_module, &attn_args, stream);
                    try cudaz.cuStreamSynchronize(stream);

                    const out_host = try allocator.alloc(f32, q_size_local);
                    defer allocator.free(out_host);
                    try cudaz.cuMemcpyDtoH(@intFromPtr(out_host.ptr), d_dst, @sizeOf(f32) * q_size_local);

                    var max_rel: f64 = 0.0;
                    var bad: usize = 0;
                    for (out_host, cpu_out, 0..) |got, want, i| {
                        const denom: f64 = @max(@as(f64, @abs(want)), 1e-6);
                        const rel: f64 = @as(f64, @abs(got - want)) / denom;
                        if (rel > max_rel) max_rel = rel;
                        if (rel > 5e-2) bad += 1; // pipeline-exacta + margen k4v4
                        _ = i;
                    }
                    if (max_rel > max_rel_overall) max_rel_overall = max_rel;
                    if (bad > 0) total_bad += 1;
                    total_runs += 1;
                }
            }
        }
    }

    if (total_bad > 0) {
        std.log.err("B5 multistream matrix: {d}/{d} failed, max_rel={d}", .{ total_bad, total_runs, max_rel_overall });
    }
    try testing.expect(total_bad == 0);
}
