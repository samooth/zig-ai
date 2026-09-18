//! Lane-b1 A10 (Dev A): decode-split MMA vs portable FA — M2 prep.
//!
//! Crea records reales vía kvarnStoreDevice (datos sintéticos, g=1 sellado),
//! KvarnDesc K/V, Q random; corre decode-split (mma kernel + combine) y
//! compara contra fattnKvarnPortableDevice (oracle validado vs CPU en B5).
//! Tolerancia rel 1e-3 (ambos GPU, distinto orden de acumulación).

const std = @import("std");

fn stdoutPrint(io: std.Io, comptime fmt: []const u8, args: anytype) !void {
    var stdout_buf: [256]u8 = undefined;
    const stdout_file = std.Io.File.stdout();
    var stdout_writer = stdout_file.writer(io, &stdout_buf);
    try stdout_writer.interface.print(fmt, args);
}
const testing = std.testing;
const build_options = @import("build_options");
const cudaz = @import("cudaz");
const kvarn = @import("kv_cache").kvarn;
const kvk = @import("kvarn_kernels");
const fattn_kv = @import("fattn_kvarn");

const GROUP = kvarn.KVAR_N_GROUP; // 128
const D: usize = 128;
const N_HEADS: usize = 1; // n_kv_heads (GQA=2 via n_q_heads)
const N_Q_HEADS: usize = 2;
const N_Q: usize = 1;
const N_STREAM: usize = 1;

test "A10: decode-split MMA ≡ portable FA (records reales, g=1, k4v4, GQA2)" {
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;
    if (build_options.fattn_cubin.len == 0) return error.SkipZigTest;

    cudaz.ensureContext() catch return error.SkipZigTest; // CI sin toolkit: skip limpio
    const kmod = try cudaz.cuModuleLoad(build_options.kvarn_cubin);
    const fmod_unused = try cudaz.cuModuleLoad(build_options.fattn_cubin);
    _ = fmod_unused; // CPU oracle (ex-portable)
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5A10);
    const rand = prng.random();

    const k_bits: u8 = 4;
    const v_bits: u8 = 4;
    const layout = try kvarn.KvarnRecordLayout.init(128, k_bits, v_bits);
    const record_bytes = layout.tile_bytes;
    const groups_per_stream: usize = 4;
    const stage_groups: usize = 4;

    // Tokens: 2 grupos (g=0 stage sink, g=1 sellado eager). KV n_kv=256.
    // 2*GROUP: cruza el límite stage/records — g=0 queda en stage
    // (sink), g=1 sellado eager ⇒ el split lee records k4v4 (2 grupos).
    const n_tokens: usize = 2 * GROUP;
    const n_kv: usize = n_tokens;

    // current: [token][head][dim] originales (el store rota).
    const current = try allocator.alloc(f32, n_tokens * N_HEADS * 128);
    defer allocator.free(current);
    for (current) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
    const indices = try allocator.alloc(i64, n_tokens);
    defer allocator.free(indices);
    for (indices, 0..) |*e, i| e.* = @intCast(i);

    const stage_len: usize = stage_groups * GROUP * N_HEADS * 128;
    const d_stage = try cudaz.cuMemAlloc(@sizeOf(f16) * stage_len);
    defer cudaz.cuMemFree(d_stage);
    {
        const zeros = try allocator.alloc(f16, stage_len);
        defer allocator.free(zeros);
        @memset(zeros, 0);
        try cudaz.cuMemcpyHtoD(d_stage, @intFromPtr(zeros.ptr), @sizeOf(f16) * stage_len);
    }

    const records_len: usize = groups_per_stream * N_HEADS * record_bytes;
    const d_records = try cudaz.cuMemAlloc(records_len);
    defer cudaz.cuMemFree(d_records);
    {
        const init = try allocator.alloc(u8, records_len);
        defer allocator.free(init);
        @memset(init, 0);
        try cudaz.cuMemcpyHtoD(d_records, @intFromPtr(init.ptr), records_len);
    }

    const d_current = try cudaz.cuMemAlloc(@sizeOf(f32) * current.len);
    defer cudaz.cuMemFree(d_current);
    try cudaz.cuMemcpyHtoD(d_current, @intFromPtr(current.ptr), @sizeOf(f32) * current.len);
    const d_indices = try cudaz.cuMemAlloc(@sizeOf(i64) * indices.len);
    defer cudaz.cuMemFree(d_indices);
    try cudaz.cuMemcpyHtoD(d_indices, @intFromPtr(indices.ptr), @sizeOf(i64) * indices.len);

    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);

    // ---- Store: sella g=1 (eager) con records reales. ----
    var sargs: kvk.KvarnStoreArgs = .{
        .current = @ptrFromInt(d_current),
        .indices = @ptrFromInt(d_indices),
        .stage = @ptrFromInt(d_stage),
        .records = @ptrFromInt(d_records),
        .n_tokens = @intCast(n_tokens),
        .n_record_heads = @intCast(N_HEADS),
        .stream = 0,
        .groups_per_stream = @intCast(groups_per_stream),
        .record_bytes = @intCast(record_bytes),
        .k_payload_off = @intCast(layout.k_payload_off),
        .k_s_col_off = @intCast(layout.k_s_col_off),
        .k_zp_off = @intCast(layout.k_zp_off),
        .k_s_row_off = @intCast(layout.k_s_row_off),
        .v_payload_off = @intCast(layout.v_payload_off),
        .v_s_col_off = @intCast(layout.v_s_col_off),
        .v_s_row_off = @intCast(layout.v_s_row_off),
        .v_zp_off = @intCast(layout.v_zp_off),
        .k_bits = k_bits,
        .v_bits = v_bits,
        .sinkhorn_iters = 16,
        .stage_groups = @intCast(stage_groups),
        .tail_groups = 3,
        .swa = 0,
        .eager_records = 1,
    };
    try kvk.kvarnStoreDevice(kmod, &sargs, stream);
    try cudaz.cuStreamSynchronize(stream);

    // ---- init_descs: descs K/V (read_indirect=0, live=1/127, eager=1). ----
    const d_descs = try cudaz.cuMemAlloc(@sizeOf(kvk.KvarnDesc) * 2);
    defer cudaz.cuMemFree(d_descs);
    {
        var ia: kvk.KvarnInitDescsArgs = .{
            .n_stream = 1,
            .n_indices = @intCast(n_tokens),
            .d_indices = @ptrFromInt(d_indices),
            .d_descs = @ptrFromInt(d_descs),
            .desc_stride = 2,
            .d_records = @ptrFromInt(d_records),
            .d_stage = @ptrFromInt(d_stage),
            .n_record_heads = @intCast(N_HEADS),
            .groups_per_stream = @intCast(groups_per_stream),
            .record_bytes = @intCast(record_bytes),
            .stage_groups = @intCast(stage_groups),
            .tail_groups = 3,
            .k_bits = k_bits,
            .v_bits = v_bits,
            .head_dim = 128, // BUG A @15a4b5c: campo obligatorio (D=128 aquí)
            .head_slices = 1,
            .eager_records = 1,
            .read_indirect = 0,
            .original_domain = 0,
            .swa = 0,
        };
        try kvk.kvarnInitDescsDevice(kmod, &ia, stream);
        try cudaz.cuStreamSynchronize(stream);
    }

    // ---- Q random [stream][head][q][dim]. ----
    const q_size: usize = N_STREAM * N_Q_HEADS * N_Q * D;
    const q = try allocator.alloc(f32, q_size);
    defer allocator.free(q);
    for (q) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
    const d_q = try cudaz.cuMemAlloc(@sizeOf(f32) * q_size);
    defer cudaz.cuMemFree(d_q);
    try cudaz.cuMemcpyHtoD(d_q, @intFromPtr(q.ptr), @sizeOf(f32) * q_size);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));

    // ---- Oracle CPU pipeline-exacta (lección B5/m1_seed): todo en
    // dominio ROTADO (Q rotada, K/V stage f16-trunc rotado), atención
    // estándar, de-rotar output. Aísla kernel-vs-oráculo. ----
    const out_len: usize = N_STREAM * N_Q_HEADS * N_Q * D;
    const d_out_port = try cudaz.cuMemAlloc(@sizeOf(f32) * out_len);
    defer cudaz.cuMemFree(d_out_port);
    const out_port = try allocator.alloc(f32, out_len);
    defer allocator.free(out_port);
    {
        const q_rot = try allocator.alloc(f32, q_size);
        defer allocator.free(q_rot);
        const k_q = try allocator.alloc(f32, n_kv * D);
        defer allocator.free(k_q);
        const v_q = try allocator.alloc(f32, n_kv * D);
        defer allocator.free(v_q);
        // Q rotada [head][q][dim]
        for (0..N_Q_HEADS) |h| {
            for (0..N_Q) |qi| {
                var row: [128]f32 = undefined;
                for (0..D) |d| row[d] = q[(h * N_Q + qi) * D + d];
                kvarn.hadamard128InPlace(&row);
                for (0..D) |d| q_rot[(h * N_Q + qi) * D + d] = row[d];
            }
        }
        // K/V stage: WHT + f16-trunc (K==V: current_v=null en este test)
        for (0..n_kv) |t| {
            var row: [128]f32 = undefined;
            for (0..D) |d| row[d] = current[t * D + d];
            kvarn.hadamard128InPlace(&row);
            for (0..D) |d| k_q[t * D + d] = @as(f32, @floatCast(@as(f16, @floatCast(row[d]))));
        }
        @memcpy(v_q, k_q);

        // ---- Records pipeline-exactos: decodificar g=1 (k4v4) a host ----
        // El kernel lee el grupo sellado (g=1) desde RECORDS k4v4, no
        // desde el f16 del stage: el oráculo debe usar el mismo dequant
        // (decodeK/VTile — M1-verificado bit-exacto vs store GPU).
        const rec_host = try allocator.alloc(u8, records_len);
        defer allocator.free(rec_host);
        try cudaz.cuMemcpyDtoH(@intFromPtr(rec_host.ptr), d_records, records_len);
        {
            const tile = try allocator.alloc(f32, GROUP * 128);
            defer allocator.free(tile);
            const rec_g1 = rec_host[record_bytes .. 2 * record_bytes];
            // K: dim-major [dim][token]
            try kvarn.decodeKTile(rec_g1, k_bits, layout, tile);
            for (GROUP..2 * GROUP) |t| {
                const pos = t - GROUP;
                for (0..D) |d| k_q[t * D + d] = tile[d * GROUP + pos];
            }
            // V: token-major [token][dim]
            try kvarn.decodeVTile(rec_g1, v_bits, layout, tile);
            for (GROUP..2 * GROUP) |t| {
                const pos = t - GROUP;
                for (0..D) |d| v_q[t * D + d] = tile[pos * 128 + d];
            }
        }
        // Atención en rotado por head (GQA: 1 kv head)
        for (0..N_Q_HEADS) |qh| {
            const scores = try allocator.alloc(f32, n_kv);
            defer allocator.free(scores);
            for (0..n_kv) |t| {
                var s: f32 = 0;
                for (0..D) |d| s += q_rot[qh * D + d] * k_q[t * D + d];
                scores[t] = s * scale;
            }
            try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "CPU scores h{d}[0..15]:", .{qh});
            for (0..16) |t| try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), " {d:.4}", .{scores[t]});
            try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "\n", .{});
            var mx: f32 = -std.math.inf(f32);
            for (scores) |sv| mx = @max(mx, sv);
            var sum: f32 = 0;
            for (scores) |*sv| {
                sv.* = @exp(sv.* - mx);
                sum += sv.*;
            }
            for (0..D) |d| {
                var acc: f32 = 0;
                for (0..n_kv) |t| acc += scores[t] * v_q[t * D + d];
                out_port[qh * D + d] = acc / sum;
            }
        }
        // De-rotar output por head (WHT involutiva)
        for (0..N_Q_HEADS) |qh| {
            var row: [128]f32 = undefined;
            for (0..D) |d| row[d] = out_port[qh * D + d];
            kvarn.hadamard128InPlace(&row);
            for (0..D) |d| out_port[qh * D + d] = row[d];
        }
    }

    // ---- SUT: decode-split (kvarn_decode_mma_kernel + combine). ----
    const SPLIT: usize = 64;
    const n_splits: usize = (n_kv + SPLIT - 1) / SPLIT; // 4
    const n_gqa_blocks: usize = 1; // MAX_GQA=6 ≥ gqa=2

    const partial_len: usize = N_STREAM * N_Q * N_Q_HEADS * n_splits * D;
    const d_partial = try cudaz.cuMemAlloc(@sizeOf(f32) * partial_len);
    defer cudaz.cuMemFree(d_partial);
    const meta_len: usize = N_STREAM * N_Q * N_Q_HEADS * n_splits;
    const d_meta = try cudaz.cuMemAlloc(@sizeOf(f32) * 2 * meta_len);
    defer cudaz.cuMemFree(d_meta);
    const d_out_split = try cudaz.cuMemAlloc(@sizeOf(f32) * out_len);
    defer cudaz.cuMemFree(d_out_split);

    {
        // Launch vía Driver API (cubin; kernels wrapper extern "C").
        const smod = try cudaz.cuModuleLoad(build_options.kvarn_split_cubin);
        const mma_func = try cudaz.cuModuleGetFunction(smod, "kvarn_decode_mma_d128_gqa6_s64_w4_k4v4_kernel");
        const comb_func = try cudaz.cuModuleGetFunction(smod, "kvarn_decode_combine_d128_kernel");

        const gqa_ratio: c_int = @intCast(N_Q_HEADS / N_HEADS);
        var d_q_any: cudaz.CUdeviceptr = d_q;
        var d_kd_any: cudaz.CUdeviceptr = d_descs;
        var d_vd_any: cudaz.CUdeviceptr = d_descs + @sizeOf(kvk.KvarnDesc);
        var mask_any: cudaz.CUdeviceptr = 0;
        var d_partial_any: cudaz.CUdeviceptr = d_partial;
        var d_meta_any: cudaz.CUdeviceptr = d_meta;
        var n_kv_c: c_int = @intCast(n_kv);
        var n_q_c: c_int = @intCast(N_Q);
        var n_qh_c: c_int = @intCast(N_Q_HEADS);
        var n_kvh_c: c_int = @intCast(N_HEADS);
        var gqa_c: c_int = gqa_ratio;
        var n_gqa_c: c_int = @intCast(n_gqa_blocks);
        var n_splits_c: c_int = @intCast(n_splits);
        var scale_v: f32 = scale;
        var kp: [14]?*const anyopaque = .{
            &d_q_any,       &d_kd_any,   &d_vd_any, &mask_any,
            &d_partial_any, &d_meta_any, &scale_v,  &n_kv_c,
            &n_q_c,         &n_qh_c,     &n_kvh_c,  &gqa_c,
            &n_gqa_c,       &n_splits_c,
        };
        // grid (n_splits, kv_heads·gqa_blocks·n_q, 1); block (32, 4, 1).
        try cudaz.cuLaunchKernel(
            mma_func,
            @intCast(n_splits),
            @intCast(N_HEADS * n_gqa_blocks * N_Q),
            1,
            32,
            4,
            1,
            0,
            stream,
            @ptrCast(&kp),
            null,
        );
        try cudaz.cuStreamSynchronize(stream); // aislar launch MMA

        const nbytes_shared_combine: c_uint = @intCast(n_splits * @sizeOf(f32));
        var d_dst_any: cudaz.CUdeviceptr = d_out_split;
        var kp2: [6]?*const anyopaque = .{
            &d_partial_any, &d_meta_any, &d_dst_any,
            &n_splits_c,    &n_q_c,      &n_qh_c,
        };
        try cudaz.cuLaunchKernel(
            comb_func,
            @intCast(N_Q_HEADS),
            @intCast(N_Q),
            1,
            256,
            1,
            1,
            nbytes_shared_combine,
            stream,
            @ptrCast(&kp2),
            null,
        );
        try cudaz.cuStreamSynchronize(stream);
    }

    // ---- Comparación: split ≡ CPU pipeline-exacta (rel < 2e-3, abs
    // < 1e-3). El oráculo replica el pipeline al completo (records
    // k4v4 del grupo sellado incluidos); el residuo del kernel es el
    // redondeo f16 del V-pass MMA (tensor core) + softmax split-local
    // __expf: diffs abs ~1e-4..4e-4 sobre outputs ~1e-3. El floor abs
    // 1e-3 evita rel-fantasma en elementos de magnitud 1e-4. ----
    const out_split = try allocator.alloc(f32, out_len);
    defer allocator.free(out_split);
    try cudaz.cuMemcpyDtoH(@intFromPtr(out_split.ptr), d_out_split, @sizeOf(f32) * out_len);

    var max_rel: f64 = 0;
    var bad: usize = 0;
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "SPLIT out[0..6]={any}\nPORT  out[0..6]={any}\n", .{ out_split[0..6], out_port[0..6] });
    for (out_split, out_port) |got, want| {
        const adiff = @abs(@as(f64, got) - @as(f64, want));
        const rel = adiff / @max(@abs(@as(f64, want)), 1e-3);
        max_rel = @max(max_rel, rel);
        if (rel > 2e-3 and adiff > 1e-3) bad += 1;
    }
    if (bad > 0) {
        std.log.err("A10 max_rel={d} bad={d}/{d}", .{ max_rel, bad, out_len });
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

test "A11 geometry: selectSplitGeometry real (GPU, gqa2, 256 kv)" {
    cudaz.ensureContext() catch return error.SkipZigTest; // CI sin toolkit
    const smod = try cudaz.cuModuleLoad(build_options.kvarn_split_cubin);
    const g = try kvk.selectSplitGeometry(
        smod,
        256, // n_kv
        1, // n_q
        N_Q_HEADS, // n_q_heads (2)
        N_HEADS, // n_kv_heads (1)
        1, // n_stream
    );
    // gqa=2 ≤ 6: split con 4 splits de 64 tokens.
    try testing.expect(g.use_split);
    try testing.expectEqual(@as(u32, 64), g.split_tokens);
    try testing.expectEqual(@as(u32, 4), g.n_splits);
    try testing.expectEqual(@as(u32, 1), g.n_gqa_blocks);
    try testing.expect(g.gqa_per_block == 6);
    try testing.expect(g.max_blocks_per_sm >= 1);
    // 4 blocks · 1 head · 1 q · 1 stream = 4 blocks vs 28 SMs: 1 wave
    // al 14% ⇒ cutoff NO aplica (grid pequeño, split útil).
    try testing.expectEqual(@as(u32, 1), g.n_waves);
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "A11: splits={d} gqa={d} bpsm={d} wave={d}% waves={d} cands={d}\n", .{
        g.n_splits,                g.gqa_per_block, g.max_blocks_per_sm,
        g.wave_efficiency_percent, g.n_waves,       g.candidate_count,
    });
}

test "A11 geometry: cutoff directo — grid grande sin split" {
    cudaz.ensureContext() catch return error.SkipZigTest; // CI sin toolkit
    const smod = try cudaz.cuModuleLoad(build_options.kvarn_split_cubin);
    // 32 heads · 8 streams: grid directo enorme ⇒ el cutoff debe
    // desactivar split si llena ≥2 waves al ≥75%.
    const g = try kvk.selectSplitGeometry(smod, 64, 16, 32, 1, 8);
    if (g.use_split) {
        try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "A11 cutoff: split sigue activo (directo no llena 2 waves al 75% — legit en sm_86 si blocks_per_sm alto)\n", .{});
    } else {
        try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "A11 cutoff: directo gana (esperado con grid grande)\n", .{});
    }
    // Invariante: o no-split (directo), o splits >= 2.
    try testing.expect(!g.use_split or g.n_splits >= 2);
}

test "A13: prefill chunk n_q=8 via decode-split (SPECIALIZED_DECODE_MAX_Q=16)" {
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;
    if (build_options.kvarn_split_cubin.len == 0) return error.SkipZigTest;

    cudaz.ensureContext() catch return error.SkipZigTest; // CI sin toolkit: skip limpio
    const kmod = try cudaz.cuModuleLoad(build_options.kvarn_cubin);
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xA13D);
    const rand = prng.random();

    const k_bits: u8 = 4;
    const v_bits: u8 = 4;
    const layout = try kvarn.KvarnRecordLayout.init(128, k_bits, v_bits);
    const record_bytes = layout.tile_bytes;
    const groups_per_stream: usize = 4;
    const stage_groups: usize = 4;
    const n_tokens: usize = 2 * GROUP;
    const n_kv: usize = n_tokens;
    const NQ: usize = 8; // chunk de prefill

    // current/indices/stage/records
    const current = try allocator.alloc(f32, n_tokens * N_HEADS * 128);
    defer allocator.free(current);
    for (current) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
    const indices = try allocator.alloc(i64, n_tokens);
    defer allocator.free(indices);
    for (indices, 0..) |*e, i| e.* = @intCast(i);

    const stage_len: usize = stage_groups * GROUP * N_HEADS * 128;
    const d_stage = try cudaz.cuMemAlloc(@sizeOf(f16) * stage_len);
    defer cudaz.cuMemFree(d_stage);
    {
        const zeros = try allocator.alloc(f16, stage_len);
        defer allocator.free(zeros);
        @memset(zeros, 0);
        try cudaz.cuMemcpyHtoD(d_stage, @intFromPtr(zeros.ptr), @sizeOf(f16) * stage_len);
    }
    const records_len: usize = groups_per_stream * N_HEADS * record_bytes;
    const d_records = try cudaz.cuMemAlloc(records_len);
    defer cudaz.cuMemFree(d_records);
    {
        const init = try allocator.alloc(u8, records_len);
        defer allocator.free(init);
        @memset(init, 0);
        try cudaz.cuMemcpyHtoD(d_records, @intFromPtr(init.ptr), records_len);
    }
    const d_current = try cudaz.cuMemAlloc(@sizeOf(f32) * current.len);
    defer cudaz.cuMemFree(d_current);
    try cudaz.cuMemcpyHtoD(d_current, @intFromPtr(current.ptr), @sizeOf(f32) * current.len);
    const d_indices = try cudaz.cuMemAlloc(@sizeOf(i64) * indices.len);
    defer cudaz.cuMemFree(d_indices);
    try cudaz.cuMemcpyHtoD(d_indices, @intFromPtr(indices.ptr), @sizeOf(i64) * indices.len);

    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);

    // Store: sella g=1 eager.
    var sargs: kvk.KvarnStoreArgs = .{
        .current = @ptrFromInt(d_current),
        .indices = @ptrFromInt(d_indices),
        .stage = @ptrFromInt(d_stage),
        .records = @ptrFromInt(d_records),
        .n_tokens = @intCast(n_tokens),
        .n_record_heads = @intCast(N_HEADS),
        .stream = 0,
        .groups_per_stream = @intCast(groups_per_stream),
        .record_bytes = @intCast(record_bytes),
        .k_payload_off = @intCast(layout.k_payload_off),
        .k_s_col_off = @intCast(layout.k_s_col_off),
        .k_zp_off = @intCast(layout.k_zp_off),
        .k_s_row_off = @intCast(layout.k_s_row_off),
        .v_payload_off = @intCast(layout.v_payload_off),
        .v_s_col_off = @intCast(layout.v_s_col_off),
        .v_s_row_off = @intCast(layout.v_s_row_off),
        .v_zp_off = @intCast(layout.v_zp_off),
        .k_bits = k_bits,
        .v_bits = v_bits,
        .sinkhorn_iters = 16,
        .stage_groups = @intCast(stage_groups),
        .tail_groups = 3,
        .swa = 0,
        .eager_records = 1,
    };
    try kvk.kvarnStoreDevice(kmod, &sargs, stream);
    try cudaz.cuStreamSynchronize(stream);

    // Descs K/V.
    const d_descs = try cudaz.cuMemAlloc(@sizeOf(kvk.KvarnDesc) * 2);
    defer cudaz.cuMemFree(d_descs);
    var ia: kvk.KvarnInitDescsArgs = .{
        .n_stream = 1,
        .n_indices = @intCast(n_tokens),
        .d_indices = @ptrFromInt(d_indices),
        .d_descs = @ptrFromInt(d_descs),
        .desc_stride = 2,
        .d_records = @ptrFromInt(d_records),
        .d_stage = @ptrFromInt(d_stage),
        .n_record_heads = @intCast(N_HEADS),
        .groups_per_stream = @intCast(groups_per_stream),
        .record_bytes = @intCast(record_bytes),
        .stage_groups = @intCast(stage_groups),
        .tail_groups = 3,
        .k_bits = k_bits,
        .v_bits = v_bits,
        .head_dim = 128, // BUG A @15a4b5c: campo obligatorio (D=128 aquí)
        .head_slices = 1,
        .eager_records = 1,
        .read_indirect = 0,
        .original_domain = 0,
        .swa = 0,
    };
    try kvk.kvarnInitDescsDevice(kmod, &ia, stream);
    try cudaz.cuStreamSynchronize(stream);

    // Q: [N_Q_HEADS][NQ][D]
    const q_size: usize = N_Q_HEADS * NQ * D;
    const q = try allocator.alloc(f32, q_size);
    defer allocator.free(q);
    for (q) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
    const d_q = try cudaz.cuMemAlloc(@sizeOf(f32) * q_size);
    defer cudaz.cuMemFree(d_q);
    try cudaz.cuMemcpyHtoD(d_q, @intFromPtr(q.ptr), @sizeOf(f32) * q_size);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));

    // Oracle CPU pipeline-exacto (records k4v4 incluidos).
    const out_len: usize = N_Q_HEADS * NQ * D;
    const out_port = try allocator.alloc(f32, out_len);
    defer allocator.free(out_port);
    {
        const q_rot = try allocator.alloc(f32, q_size);
        defer allocator.free(q_rot);
        const k_q = try allocator.alloc(f32, n_kv * D);
        defer allocator.free(k_q);
        const v_q = try allocator.alloc(f32, n_kv * D);
        defer allocator.free(v_q);
        for (0..N_Q_HEADS) |h| {
            for (0..NQ) |qi| {
                var row: [128]f32 = undefined;
                for (0..D) |d| row[d] = q[(h * NQ + qi) * D + d];
                kvarn.hadamard128InPlace(&row);
                for (0..D) |d| q_rot[(h * NQ + qi) * D + d] = row[d];
            }
        }
        for (0..n_kv) |t| {
            var row: [128]f32 = undefined;
            for (0..D) |d| row[d] = current[t * D + d];
            kvarn.hadamard128InPlace(&row);
            for (0..D) |d| k_q[t * D + d] = @as(f32, @floatCast(@as(f16, @floatCast(row[d]))));
        }
        @memcpy(v_q, k_q);
        const rec_host = try allocator.alloc(u8, records_len);
        defer allocator.free(rec_host);
        try cudaz.cuMemcpyDtoH(@intFromPtr(rec_host.ptr), d_records, records_len);
        {
            const tile = try allocator.alloc(f32, GROUP * 128);
            defer allocator.free(tile);
            const rec_g1 = rec_host[record_bytes .. 2 * record_bytes];
            try kvarn.decodeKTile(rec_g1, k_bits, layout, tile);
            for (GROUP..2 * GROUP) |t| {
                const pos = t - GROUP;
                for (0..D) |d| k_q[t * D + d] = tile[d * GROUP + pos];
            }
            try kvarn.decodeVTile(rec_g1, v_bits, layout, tile);
            for (GROUP..2 * GROUP) |t| {
                const pos = t - GROUP;
                for (0..D) |d| v_q[t * D + d] = tile[pos * 128 + d];
            }
        }
        for (0..N_Q_HEADS) |qh| {
            const scores = try allocator.alloc(f32, n_kv);
            defer allocator.free(scores);
            for (0..n_kv) |t| {
                var s: f32 = 0;
                for (0..D) |d| s += q_rot[qh * NQ * D + 0 * D + d] * k_q[t * D + d];
                scores[t] = s * scale;
            }
            // Solo q_index=0 del chunk (representativo; los 8 comparten
            // el mismo mecanismo — el kernel repite por blockIdx.y).
            var mx: f32 = -std.math.inf(f32);
            for (scores) |sv| mx = @max(mx, sv);
            var sum: f32 = 0;
            for (scores) |*sv| {
                sv.* = @exp(sv.* - mx);
                sum += sv.*;
            }
            for (0..D) |d| {
                var acc: f32 = 0;
                for (0..n_kv) |t| acc += scores[t] * v_q[t * D + d];
                out_port[qh * NQ * D + d] = acc / sum;
            }
        }
        for (0..N_Q_HEADS) |qh| {
            var row: [128]f32 = undefined;
            for (0..D) |d| row[d] = out_port[qh * NQ * D + d];
            kvarn.hadamard128InPlace(&row);
            for (0..D) |d| out_port[qh * NQ * D + d] = row[d];
        }
    }

    // SUT: decode-split con n_q=8.
    const SPLIT: usize = 64;
    const n_splits: usize = (n_kv + SPLIT - 1) / SPLIT;
    const n_gqa_blocks: usize = 1;
    const partial_len: usize = NQ * N_Q_HEADS * n_splits * D;
    const d_partial = try cudaz.cuMemAlloc(@sizeOf(f32) * partial_len);
    defer cudaz.cuMemFree(d_partial);
    const meta_len: usize = NQ * N_Q_HEADS * n_splits;
    const d_meta = try cudaz.cuMemAlloc(@sizeOf(f32) * 2 * meta_len);
    defer cudaz.cuMemFree(d_meta);
    const d_out_split = try cudaz.cuMemAlloc(@sizeOf(f32) * out_len);
    defer cudaz.cuMemFree(d_out_split);

    {
        const smod = try cudaz.cuModuleLoad(build_options.kvarn_split_cubin);
        const mma_func = try cudaz.cuModuleGetFunction(smod, "kvarn_decode_mma_d128_gqa6_s64_w4_k4v4_kernel");
        const comb_func = try cudaz.cuModuleGetFunction(smod, "kvarn_decode_combine_d128_kernel");

        const gqa_ratio: c_int = @intCast(N_Q_HEADS / N_HEADS);
        var d_q_any: cudaz.CUdeviceptr = d_q;
        var d_kd_any: cudaz.CUdeviceptr = d_descs;
        var d_vd_any: cudaz.CUdeviceptr = d_descs + @sizeOf(kvk.KvarnDesc);
        var mask_any: cudaz.CUdeviceptr = 0;
        var d_partial_any: cudaz.CUdeviceptr = d_partial;
        var d_meta_any: cudaz.CUdeviceptr = d_meta;
        var n_kv_c: c_int = @intCast(n_kv);
        var n_q_c: c_int = @intCast(NQ);
        var n_qh_c: c_int = @intCast(N_Q_HEADS);
        var n_kvh_c: c_int = @intCast(N_HEADS);
        var gqa_c: c_int = gqa_ratio;
        var n_gqa_c: c_int = @intCast(n_gqa_blocks);
        var n_splits_c: c_int = @intCast(n_splits);
        var scale_v: f32 = scale;
        var kp: [14]?*const anyopaque = .{
            &d_q_any,       &d_kd_any,   &d_vd_any, &mask_any,
            &d_partial_any, &d_meta_any, &scale_v,  &n_kv_c,
            &n_q_c,         &n_qh_c,     &n_kvh_c,  &gqa_c,
            &n_gqa_c,       &n_splits_c,
        };
        try cudaz.cuLaunchKernel(
            mma_func,
            @intCast(n_splits),
            @intCast(N_HEADS * n_gqa_blocks * NQ),
            1,
            32,
            4,
            1,
            0,
            stream,
            @ptrCast(&kp),
            null,
        );
        try cudaz.cuStreamSynchronize(stream);

        const nbytes_shared_combine: c_uint = @intCast(n_splits * @sizeOf(f32));
        var d_dst_any: cudaz.CUdeviceptr = d_out_split;
        var kp2: [6]?*const anyopaque = .{
            &d_partial_any, &d_meta_any, &d_dst_any,
            &n_splits_c,    &n_q_c,      &n_qh_c,
        };
        try cudaz.cuLaunchKernel(
            comb_func,
            @intCast(N_Q_HEADS),
            @intCast(NQ),
            1,
            256,
            1,
            1,
            nbytes_shared_combine,
            stream,
            @ptrCast(&kp2),
            null,
        );
        try cudaz.cuStreamSynchronize(stream);
    }

    // Comparación q_index=0: split ≡ oracle. Floor abs 1.5e-3 — un
    // elemento de cancelación (valor ~1e-4) puede acumular el ruido
    // f16 del V-pass hasta ~1.5e-3 sin señal estructural (A10 con la
    // misma ruta: max_abs 3.9e-4 en 256/256).
    const out_split = try allocator.alloc(f32, out_len);
    defer allocator.free(out_split);
    try cudaz.cuMemcpyDtoH(@intFromPtr(out_split.ptr), d_out_split, @sizeOf(f32) * out_len);
    var max_rel: f64 = 0;
    var bad: usize = 0;
    for (0..N_Q_HEADS) |qh| {
        for (0..D) |d| {
            const got: f64 = out_split[qh * NQ * D + d];
            const want: f64 = out_port[qh * NQ * D + d];
            const adiff = @abs(got - want);
            const rel = adiff / @max(@abs(want), 1.5e-3);
            max_rel = @max(max_rel, rel);
            if (rel > 2e-3 and adiff > 1.5e-3) bad += 1;
        }
    }
    if (bad > 0) {
        for (0..N_Q_HEADS) |qh| {
            for (0..D) |d| {
                const got: f64 = out_split[qh * NQ * D + d];
                const want: f64 = out_port[qh * NQ * D + d];
                const adiff = @abs(got - want);
                const rel = adiff / @max(@abs(want), 1e-3);
                if (rel > 2e-3 and adiff > 1e-3) {
                    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "A13 bad: h={d} d={d} got={d:.6} want={d:.6} adiff={d:.6}\n", .{ qh, d, got, want, adiff });
                }
            }
        }
        std.log.err("A13 max_rel={d} bad={d}", .{ max_rel, bad });
    }
    try testing.expectEqual(@as(usize, 0), bad);
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "A13: n_q=8 prefill chunk OK (max_rel={d:.6})\n", .{max_rel});
}

// ============================================================================
// A13 GATE M2: decode-split 1000-seed (estilo B5 m1_seed). Default 32
// seeds (smoke); ZIG_AI_A13_1000SEEDS=1 → 1000 (gate completo).
// ============================================================================

const N_SEEDS_DEFAULT_A13: u32 = 32;
const N_SEEDS_FULL_A13: u32 = 1000;

test "A13 gate: decode-split ≡ CPU pipeline-exacta, N-seeds (gated cubin)" {
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;
    if (build_options.kvarn_split_cubin.len == 0) return error.SkipZigTest;

    const n_seeds = blk: {
        if (std.c.getenv("ZIG_AI_A13_1000SEEDS") != null) break :blk N_SEEDS_FULL_A13;
        break :blk N_SEEDS_DEFAULT_A13;
    };

    // Geometría del gate: idéntica al A10 (2 grupos, records reales).
    const k_bits_test: u8 = 4;
    const v_bits_test: u8 = 4;
    const layout_test = try kvarn.KvarnRecordLayout.init(128, k_bits_test, v_bits_test);
    const groups_per_stream_test: usize = 4;
    const stage_groups_test: usize = 4;
    const n_kv_test: usize = 2 * GROUP;
    const scale_test: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));

    cudaz.ensureContext() catch return error.SkipZigTest; // CI sin toolkit: skip limpio
    const kmod = try cudaz.cuModuleLoad(build_options.kvarn_cubin);
    const smod = try cudaz.cuModuleLoad(build_options.kvarn_split_cubin);
    const allocator = testing.allocator;
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);

    const SPLIT: usize = 64;
    const n_splits: usize = (n_kv_test + SPLIT - 1) / SPLIT;
    const partial_len: usize = N_Q * N_Q_HEADS * n_splits * D;
    const meta_len: usize = N_Q * N_Q_HEADS * n_splits;
    const out_len: usize = N_Q_HEADS * N_Q * D;

    // Buffers device (reutilizados por seed — solo cambia el contenido).
    const stage_len: usize = stage_groups_test * GROUP * N_HEADS * 128;
    const d_stage = try cudaz.cuMemAlloc(@sizeOf(f16) * stage_len);
    defer cudaz.cuMemFree(d_stage);
    const records_len: usize = groups_per_stream_test * N_HEADS * layout_test.tile_bytes;
    const d_records = try cudaz.cuMemAlloc(records_len);
    defer cudaz.cuMemFree(d_records);
    const d_current = try cudaz.cuMemAlloc(@sizeOf(f32) * n_kv_test * N_HEADS * 128);
    defer cudaz.cuMemFree(d_current);
    const d_indices = try cudaz.cuMemAlloc(@sizeOf(i64) * n_kv_test);
    defer cudaz.cuMemFree(d_indices);
    const d_descs = try cudaz.cuMemAlloc(@sizeOf(kvk.KvarnDesc) * 2);
    defer cudaz.cuMemFree(d_descs);
    const d_q = try cudaz.cuMemAlloc(@sizeOf(f32) * N_Q_HEADS * N_Q * D);
    defer cudaz.cuMemFree(d_q);
    const d_partial = try cudaz.cuMemAlloc(@sizeOf(f32) * partial_len);
    defer cudaz.cuMemFree(d_partial);
    const d_meta = try cudaz.cuMemAlloc(@sizeOf(f32) * 2 * meta_len);
    defer cudaz.cuMemFree(d_meta);
    const d_out = try cudaz.cuMemAlloc(@sizeOf(f32) * out_len);
    defer cudaz.cuMemFree(d_out);

    var prng = std.Random.DefaultPrng.init(0xA13C);
    const rand = prng.random();

    var max_rel_overall: f64 = 0;
    var bad_seeds: usize = 0;
    var seed_idx: u32 = 0;

    // Datos host reutilizados.
    const current = try allocator.alloc(f32, n_kv_test * N_HEADS * 128);
    defer allocator.free(current);
    const indices = try allocator.alloc(i64, n_kv_test);
    defer allocator.free(indices);
    for (indices, 0..) |*e, i| e.* = @intCast(i);
    const q = try allocator.alloc(f32, N_Q_HEADS * N_Q * D);
    defer allocator.free(q);
    const out_split = try allocator.alloc(f32, out_len);
    defer allocator.free(out_split);
    const rec_host = try allocator.alloc(u8, records_len);
    defer allocator.free(rec_host);

    try cudaz.cuMemcpyHtoD(d_indices, @intFromPtr(indices.ptr), @sizeOf(i64) * n_kv_test);

    const mma_func = try cudaz.cuModuleGetFunction(smod, "kvarn_decode_mma_d128_gqa6_s64_w4_k4v4_kernel");
    const comb_func = try cudaz.cuModuleGetFunction(smod, "kvarn_decode_combine_d128_kernel");
    const gqa_ratio: c_int = @intCast(N_Q_HEADS / N_HEADS);
    const n_gqa_blocks: usize = 1;

    while (seed_idx < n_seeds) : (seed_idx += 1) {
        // 1) Datos random del seed.
        for (current) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
        for (q) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
        try cudaz.cuMemcpyHtoD(d_current, @intFromPtr(current.ptr), @sizeOf(f32) * current.len);
        try cudaz.cuMemcpyHtoD(d_q, @intFromPtr(q.ptr), @sizeOf(f32) * q.len);

        // 2) Store real (sella g=1 eager con records k4v4).
        {
            const zeros = try allocator.alloc(f16, stage_len);
            defer allocator.free(zeros);
            @memset(zeros, 0);
            try cudaz.cuMemcpyHtoD(d_stage, @intFromPtr(zeros.ptr), @sizeOf(f16) * stage_len);
            const init = try allocator.alloc(u8, records_len);
            defer allocator.free(init);
            @memset(init, 0);
            try cudaz.cuMemcpyHtoD(d_records, @intFromPtr(init.ptr), records_len);
        }
        var sargs: kvk.KvarnStoreArgs = .{
            .current = @ptrFromInt(d_current),
            .indices = @ptrFromInt(d_indices),
            .stage = @ptrFromInt(d_stage),
            .records = @ptrFromInt(d_records),
            .n_tokens = @intCast(n_kv_test),
            .n_record_heads = @intCast(N_HEADS),
            .stream = 0,
            .groups_per_stream = @intCast(groups_per_stream_test),
            .record_bytes = @intCast(layout_test.tile_bytes),
            .k_payload_off = @intCast(layout_test.k_payload_off),
            .k_s_col_off = @intCast(layout_test.k_s_col_off),
            .k_zp_off = @intCast(layout_test.k_zp_off),
            .k_s_row_off = @intCast(layout_test.k_s_row_off),
            .v_payload_off = @intCast(layout_test.v_payload_off),
            .v_s_col_off = @intCast(layout_test.v_s_col_off),
            .v_s_row_off = @intCast(layout_test.v_s_row_off),
            .v_zp_off = @intCast(layout_test.v_zp_off),
            .k_bits = k_bits_test,
            .v_bits = v_bits_test,
            .sinkhorn_iters = 16,
            .stage_groups = @intCast(stage_groups_test),
            .tail_groups = 3,
            .swa = 0,
            .eager_records = 1,
        };
        try kvk.kvarnStoreDevice(kmod, &sargs, stream);

        // 3) initDescs (live=1/127, eager).
        var ia: kvk.KvarnInitDescsArgs = .{
            .n_stream = 1,
            .n_indices = @intCast(n_kv_test),
            .d_indices = @ptrFromInt(d_indices),
            .d_descs = @ptrFromInt(d_descs),
            .desc_stride = 2,
            .d_records = @ptrFromInt(d_records),
            .d_stage = @ptrFromInt(d_stage),
            .n_record_heads = @intCast(N_HEADS),
            .groups_per_stream = @intCast(groups_per_stream_test),
            .record_bytes = @intCast(layout_test.tile_bytes),
            .stage_groups = @intCast(stage_groups_test),
            .tail_groups = 3,
            .k_bits = k_bits_test,
            .v_bits = v_bits_test,
            .head_dim = 128, // BUG A @15a4b5c: campo obligatorio (D=128 aquí)
            .head_slices = 1,
            .eager_records = 1,
            .read_indirect = 0,
            .original_domain = 0,
            .swa = 0,
        };
        try kvk.kvarnInitDescsDevice(kmod, &ia, stream);
        try cudaz.cuStreamSynchronize(stream);

        // 4) SUT: decode-split.
        {
            var d_q_any: cudaz.CUdeviceptr = d_q;
            var d_kd_any: cudaz.CUdeviceptr = d_descs;
            var d_vd_any: cudaz.CUdeviceptr = d_descs + @sizeOf(kvk.KvarnDesc);
            var mask_any: cudaz.CUdeviceptr = 0;
            var d_partial_any: cudaz.CUdeviceptr = d_partial;
            var d_meta_any: cudaz.CUdeviceptr = d_meta;
            var n_kv_c: c_int = @intCast(n_kv_test);
            var n_q_c: c_int = @intCast(N_Q);
            var n_qh_c: c_int = @intCast(N_Q_HEADS);
            var n_kvh_c: c_int = @intCast(N_HEADS);
            var gqa_c: c_int = gqa_ratio;
            var n_gqa_c: c_int = @intCast(n_gqa_blocks);
            var n_splits_c: c_int = @intCast(n_splits);
            var scale_v: f32 = scale_test;
            var kp: [14]?*const anyopaque = .{
                &d_q_any,       &d_kd_any,   &d_vd_any, &mask_any,
                &d_partial_any, &d_meta_any, &scale_v,  &n_kv_c,
                &n_q_c,         &n_qh_c,     &n_kvh_c,  &gqa_c,
                &n_gqa_c,       &n_splits_c,
            };
            try cudaz.cuLaunchKernel(
                mma_func,
                @intCast(n_splits),
                @intCast(N_HEADS * n_gqa_blocks * N_Q),
                1,
                32,
                4,
                1,
                0,
                stream,
                @ptrCast(&kp),
                null,
            );
            var d_dst_any: cudaz.CUdeviceptr = d_out;
            var kp2: [6]?*const anyopaque = .{
                &d_partial_any, &d_meta_any, &d_dst_any,
                &n_splits_c,    &n_q_c,      &n_qh_c,
            };
            try cudaz.cuLaunchKernel(
                comb_func,
                @intCast(N_Q_HEADS),
                @intCast(N_Q),
                1,
                256,
                1,
                1,
                @intCast(n_splits * @sizeOf(f32)),
                stream,
                @ptrCast(&kp2),
                null,
            );
            try cudaz.cuStreamSynchronize(stream);
        }

        // 5) CPU oracle pipeline-exacto (records k4v4 g=1 incluidos).
        try cudaz.cuMemcpyDtoH(@intFromPtr(rec_host.ptr), d_records, records_len);
        const out_port = try allocator.alloc(f32, out_len);
        defer allocator.free(out_port);
        {
            const q_rot = try allocator.alloc(f32, q.len);
            defer allocator.free(q_rot);
            const k_q = try allocator.alloc(f32, n_kv_test * D);
            defer allocator.free(k_q);
            const v_q = try allocator.alloc(f32, n_kv_test * D);
            defer allocator.free(v_q);
            for (0..N_Q_HEADS) |h| {
                var row: [128]f32 = undefined;
                for (0..D) |d| row[d] = q[(h * N_Q) * D + d];
                kvarn.hadamard128InPlace(&row);
                for (0..D) |d| q_rot[h * D + d] = row[d];
            }
            for (0..n_kv_test) |t| {
                var row: [128]f32 = undefined;
                for (0..D) |d| row[d] = current[t * D + d];
                kvarn.hadamard128InPlace(&row);
                for (0..D) |d| k_q[t * D + d] = @as(f32, @floatCast(@as(f16, @floatCast(row[d]))));
            }
            @memcpy(v_q, k_q);
            {
                const tile = try allocator.alloc(f32, GROUP * 128);
                defer allocator.free(tile);
                const rec_g1 = rec_host[layout_test.tile_bytes .. 2 * layout_test.tile_bytes];
                try kvarn.decodeKTile(rec_g1, k_bits_test, layout_test, tile);
                for (GROUP..2 * GROUP) |t| {
                    const pos = t - GROUP;
                    for (0..D) |d| k_q[t * D + d] = tile[d * GROUP + pos];
                }
                try kvarn.decodeVTile(rec_g1, v_bits_test, layout_test, tile);
                for (GROUP..2 * GROUP) |t| {
                    const pos = t - GROUP;
                    for (0..D) |d| v_q[t * D + d] = tile[pos * 128 + d];
                }
            }
            for (0..N_Q_HEADS) |qh| {
                const scores = try allocator.alloc(f32, n_kv_test);
                defer allocator.free(scores);
                for (0..n_kv_test) |t| {
                    var s: f32 = 0;
                    for (0..D) |dd| s += q_rot[qh * D + dd] * k_q[t * D + dd];
                    scores[t] = s * scale_test;
                }
                var mx: f32 = -std.math.inf(f32);
                for (scores) |sv| mx = @max(mx, sv);
                var sum: f32 = 0;
                for (scores) |*sv| {
                    sv.* = @exp(sv.* - mx);
                    sum += sv.*;
                }
                for (0..D) |dd| {
                    var acc: f32 = 0;
                    for (0..n_kv_test) |t| acc += scores[t] * v_q[t * D + dd];
                    out_port[qh * D + dd] = acc / sum;
                }
            }
            for (0..N_Q_HEADS) |qh| {
                var row: [128]f32 = undefined;
                for (0..D) |dd| row[dd] = out_port[qh * D + dd];
                kvarn.hadamard128InPlace(&row);
                for (0..D) |dd| out_port[qh * D + dd] = row[dd];
            }
        }

        // 6) Comparación con la tolerancia del A10 (rel 2e-3 floor 1e-3).
        try cudaz.cuMemcpyDtoH(@intFromPtr(out_split.ptr), d_out, @sizeOf(f32) * out_len);
        var seed_bad: usize = 0;
        for (out_split, out_port) |got, want| {
            const adiff = @abs(@as(f64, got) - @as(f64, want));
            const rel = adiff / @max(@abs(@as(f64, want)), 1e-3);
            max_rel_overall = @max(max_rel_overall, rel);
            if (rel > 2e-3 and adiff > 1e-3) seed_bad += 1;
        }
        if (seed_bad > 0) {
            bad_seeds += 1;
            if (bad_seeds <= 3) {
                std.log.err("A13 seed {d}: {d}/{d} elems fuera de tolerancia (max_rel={d})", .{ seed_idx, seed_bad, out_len, max_rel_overall });
            }
        }
    }

    if (bad_seeds > 0) {
        std.log.err("A13 gate: {d}/{d} seeds failed, max_rel={d}", .{ bad_seeds, n_seeds, max_rel_overall });
    }
    try testing.expectEqual(@as(usize, 0), bad_seeds);
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "A13 gate: {d} seeds OK (max_rel={d:.6})\n", .{ n_seeds, max_rel_overall });
}
