//! 9.4 (lane-b) REPRO E2E: divergence debugger — replica la secuencia
//! EXACTA del engine con K/V/Q reales volcados por ZIG_AI_ATT_DUMP:
//!   append(n=5, base=0) → init_descs → append(n=1, base=5) → init_descs
//!   → fattn D256 vs CPU ref exacta.
//!
//! Uso:
//!   1) Generar dumps: run engine ATT_DBG_ALL=1 ZIG_AI_ATT_DUMP=<dir>
//!      (brazo paged, -n 1) → s0_L3_{k,v}_rope.f32 + s5_L3_{k,v}_rope.f32
//!      + s5_L3_q_rope.f32
//!   2) ZIG_AI_ATT_DUMP_DIR=<dir> zig build test-fattn-kvarn-repro --summary all
//!
//! Gating: skip sin CUDA/cubin/dumps.

const std = @import("std");
const testing = std.testing;
const build_options = @import("build_options");
const cudaz = @import("cudaz");
const kvk = @import("kvarn_kernels");
const fattn_kv = @import("fattn_kvarn");
const kvarn = @import("kv_cache").kvarn;

const D: usize = 256;
const KV_DIM: usize = 512; // 2 kv_heads × 256
const Q_DIM: usize = 2048; // 8 q_heads × 256

fn loadF32(allocator: std.mem.Allocator, path: []const u8) ![]f32 {
    const io = std.Io.Threaded.global_single_threaded.io();
    const f = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only });
    defer f.close(io);
    const n = (try f.stat(io)).size;
    const buf = try allocator.alloc(u8, n);
    defer allocator.free(buf);
    _ = try f.readPositionalAll(io, buf, 0);
    const f32s: []align(4) const f32 = @alignCast(std.mem.bytesAsSlice(f32, buf));
    const out = try allocator.alloc(f32, f32s.len);
    @memcpy(out, f32s);
    return out;
}

fn randBuf(allocator: std.mem.Allocator, n: usize) ![]f32 {
    const out = try allocator.alloc(f32, n);
    var prng = std.Random.DefaultPrng.init(0x940A6A);
    const rand = prng.random();
    for (out) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
    return out;
}

/// Probe: [n_tokens][KV_DIM]. token_global_base = índice del primer token
/// de este buffer (0 para prefill, n_pre para decode). Valor del token
/// global g = g+1 en dim `dim` de CADA kv_head. WEIGHTS: V[g] = e_g.
/// value_side: 0=K (dim k_dim), 1=V (dim k_dim+v_off, escala v_scale).
fn probeK(allocator: std.mem.Allocator, n_tokens: usize, token_global_base: usize, value_side: u32) ![]f32 {
    const out = try allocator.alloc(f32, n_tokens * KV_DIM);
    @memset(out, 0);
    const probe_dim: usize = blk: {
        const pv = std.c.getenv("ZIG_AI_REPRO_PROBE_SLICE") orelse break :blk 0;
        break :blk (std.fmt.parseInt(usize, pv[0..std.mem.len(pv)], 10) catch 0) * 128;
    };
    const v_off: usize = blk: {
        const pv = std.c.getenv("ZIG_AI_REPRO_PROBE_VOFF") orelse break :blk 0;
        break :blk std.fmt.parseInt(usize, pv[0..std.mem.len(pv)], 10) catch 0;
    };
    const v_scale: f32 = blk: {
        const pv = std.c.getenv("ZIG_AI_REPRO_PROBE_VSCALE") orelse break :blk 1.0;
        break :blk std.fmt.parseFloat(f32, pv[0..std.mem.len(pv)]) catch 1.0;
    };
    const is_v = value_side == 1;
    const weights_mode = std.c.getenv("ZIG_AI_REPRO_PROBE_WEIGHTS") != null and is_v;
    const n_pre_env: usize = blk: {
        const pv = std.c.getenv("ZIG_AI_REPRO_PROBE_NPRE") orelse break :blk 5;
        break :blk std.fmt.parseInt(usize, pv[0..std.mem.len(pv)], 10) catch 5;
    };
    const dim = probe_dim + (if (is_v) v_off else 0);
    const scale_factor: f32 = if (is_v) v_scale else 1.0;
    for (0..n_tokens) |t| {
        const g = token_global_base + t;
        for (0..2) |kvh| {
            if (weights_mode) {
                out[t * KV_DIM + kvh * D + (g % 128)] = 1.0;
            } else {
                out[t * KV_DIM + kvh * D + dim] = @as(f32, @floatFromInt(g + 1)) * scale_factor;
            }
        }
    }
    _ = n_pre_env;
    return out;
}

test "9.4 repro E2E: append incremental 5+1 con K/V/Q reales" {
    if (build_options.fattn_cubin.len == 0) return error.SkipZigTest;
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;
    const dump_dir_raw = std.c.getenv("ZIG_AI_ATT_DUMP_DIR");
    const use_random = std.c.getenv("ZIG_AI_REPRO_RANDOM") != null;
    const use_probe = std.c.getenv("ZIG_AI_REPRO_PROBE") != null;
    if (dump_dir_raw == null and !use_random and !use_probe) return error.SkipZigTest;
    var n_pre: usize = 5;
    if (std.c.getenv("ZIG_AI_REPRO_PROBE_NPRE")) |nv| n_pre = std.fmt.parseInt(usize, nv[0..std.mem.len(nv)], 10) catch 5;

    var pbuf: [512]u8 = undefined;
    // Probe mode: K[t] = s_t·e_0 (score_t = s_t·(1/16)), V[t] = (t+1)·e_0.
    // Q = e_0 por head. El output por head = softmax(s/16)·[1,2,..,6]·e_0
    // — revela los pesos/identidad de tokens que el kernel usó.
    const k_pre = if (use_probe) try probeK(testing.allocator, n_pre, 0, 0) else if (use_random) try randBuf(testing.allocator, 5 * KV_DIM) else try loadF32(testing.allocator, try std.fmt.bufPrint(&pbuf, "{s}/s0_L3_k_rope.f32", .{dump_dir_raw.?}));
    defer testing.allocator.free(k_pre);
    const v_pre = if (use_probe) try probeK(testing.allocator, n_pre, 0, 1) else if (use_random) try randBuf(testing.allocator, 5 * KV_DIM) else try loadF32(testing.allocator, try std.fmt.bufPrint(&pbuf, "{s}/s0_L3_v_rope.f32", .{dump_dir_raw.?}));
    defer testing.allocator.free(v_pre);
    const k_dec = if (use_probe) try probeK(testing.allocator, 1, n_pre, 0) else if (use_random) try randBuf(testing.allocator, KV_DIM) else try loadF32(testing.allocator, try std.fmt.bufPrint(&pbuf, "{s}/s5_L3_k_rope.f32", .{dump_dir_raw.?}));
    defer testing.allocator.free(k_dec);
    const v_dec = if (use_probe) try probeK(testing.allocator, 1, n_pre, 1) else if (use_random) try randBuf(testing.allocator, KV_DIM) else try loadF32(testing.allocator, try std.fmt.bufPrint(&pbuf, "{s}/s5_L3_v_rope.f32", .{dump_dir_raw.?}));
    defer testing.allocator.free(v_dec);
    const q_dec = try testing.allocator.alloc(f32, Q_DIM);
    defer testing.allocator.free(q_dec);
    @memset(q_dec, 0);
    if (use_probe) {
        // SLICE-PROBE: ZIG_AI_REPRO_PROBE_SLICE=<n> (0|1) ⇒ el valor vive
        // SOLO en dim slice·128 del head. Default: ambos slices (dim 0).
        const probe_slice_raw = std.c.getenv("ZIG_AI_REPRO_PROBE_SLICE");
        const probe_dim: usize = if (probe_slice_raw) |pv| (std.fmt.parseInt(usize, pv[0..std.mem.len(pv)], 10) catch 0) * 128 else 0;
        const probe_qh: usize = blk: {
            const qv = std.c.getenv("ZIG_AI_REPRO_PROBE_QH") orelse break :blk 8;
            break :blk std.fmt.parseInt(usize, qv[0..std.mem.len(qv)], 10) catch 8;
        };
        for (0..probe_qh) |qh| q_dec[qh * D + probe_dim] = 16.0; // score_t = K_t·Q·(1/16) = s_t
    } else {
        const ql = try loadF32(testing.allocator, try std.fmt.bufPrint(&pbuf, "{s}/s5_L3_q_rope.f32", .{dump_dir_raw.?}));
        defer testing.allocator.free(ql);
        @memcpy(q_dec, ql);
    }

    // ZIG_AI_REPRO_SCALE=<f>: escala todos los datos (K/V/Q) para A/B
    // magnitud (ej. 0.025 lleva ±10 → ±0.25).
    var repro_scale: f32 = 1.0;
    if (std.c.getenv("ZIG_AI_REPRO_SCALE")) |sv| {
        repro_scale = std.fmt.parseFloat(f32, sv[0..std.mem.len(sv)]) catch 1.0;
    }
    if (repro_scale != 1.0) {
        for (k_pre) |*x| x.* *= repro_scale;
        for (v_pre) |*x| x.* *= repro_scale;
        for (k_dec) |*x| x.* *= repro_scale;
        for (v_dec) |*x| x.* *= repro_scale;
        for (q_dec) |*x| x.* *= repro_scale;
    }

    try testing.expectEqual(@as(usize, 5 * KV_DIM), k_pre.len);
    try testing.expectEqual(@as(usize, KV_DIM), k_dec.len);
    try testing.expectEqual(@as(usize, Q_DIM), q_dec.len);

    cudaz.ensureContext() catch return error.SkipZigTest;
    const module_kv = try cudaz.cuModuleLoad(build_options.kvarn_cubin);
    const module_fa = try cudaz.cuModuleLoad(build_options.fattn_cubin);

    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);

    // Geometría E2E 0.8B: 2 kv_heads lógicas D=256 ⇒ 4 físicas.
    // ZIG_AI_REPRO_PROBE_KVH=<n> override (1 ⇒ sin GQA, discriminador
    // posicional vs kv-mapping).
    var n_kv_heads: usize = 2;
    if (std.c.getenv("ZIG_AI_REPRO_PROBE_KVH")) |kv| n_kv_heads = std.fmt.parseInt(usize, kv[0..std.mem.len(kv)], 10) catch 2;
    const n_physical: usize = n_kv_heads * 2;
    var n_q_heads: usize = 8;
    if (std.c.getenv("ZIG_AI_REPRO_PROBE_QH")) |qv| n_q_heads = std.fmt.parseInt(usize, qv[0..std.mem.len(qv)], 10) catch 8;
    const gqa: usize = n_q_heads / n_kv_heads;
    const n_tot: usize = n_pre + 1;

    // Layout C1 (kvarn.zig) + config E2E: k5 v4, sg=4, tg=3.
    const layout128 = try kvarn.KvarnRecordLayout.init(128, 5, 4);
    const record_bytes: usize = layout128.tile_bytes;
    const stage_groups: usize = 4;
    const tail_groups: usize = 3;
    const groups_per_stream: usize = 512; // ctx 65536/128
    const stage_len: usize = stage_groups * 128 * (2 * n_physical) * 128;
    const indices_len: usize = 65536;

    const h_indices = try testing.allocator.alloc(i64, indices_len);
    defer testing.allocator.free(h_indices);
    @memset(h_indices, -1);
    for (0..n_pre) |i| h_indices[i] = @intCast(i);
    h_indices[n_pre] = @intCast(n_pre);

    const d_stage = try cudaz.cuMemAlloc(stage_len * 2);
    defer cudaz.cuMemFree(d_stage);
    const d_records = try cudaz.cuMemAlloc(record_bytes * groups_per_stream * n_physical);
    defer cudaz.cuMemFree(d_records);
    const d_indices = try cudaz.cuMemAlloc(indices_len * 8);
    defer cudaz.cuMemFree(d_indices);
    const d_descs = try cudaz.cuMemAlloc(2 * n_kv_heads * @sizeOf(kvk.KvarnDesc) * 2);
    defer cudaz.cuMemFree(d_descs);
    try cudaz.cuMemsetD8(d_stage, 0, stage_len * 2);
    try cudaz.cuMemsetD8(d_descs, 0, 2 * n_kv_heads * @sizeOf(kvk.KvarnDesc) * 2);

    const d_k = try cudaz.cuMemAlloc(KV_DIM * n_pre * 4 + KV_DIM * 4);
    defer cudaz.cuMemFree(d_k);
    const d_v = try cudaz.cuMemAlloc(KV_DIM * n_pre * 4 + KV_DIM * 4);
    defer cudaz.cuMemFree(d_v);
    const d_q = try cudaz.cuMemAlloc(Q_DIM * 4);
    defer cudaz.cuMemFree(d_q);
    const d_dst = try cudaz.cuMemAlloc(Q_DIM * 4);
    defer cudaz.cuMemFree(d_dst);
    try cudaz.cuMemcpyHtoD(d_q, @intFromPtr(q_dec.ptr), Q_DIM * 4);

    const makeStore = struct {
        fn go(n_tokens: usize, d_k_p: cudaz.CUdeviceptr, d_v_p: cudaz.CUdeviceptr, lay: kvarn.KvarnRecordLayout, gps: usize, sg: usize, tg: usize, d_indices_p: cudaz.CUdeviceptr, d_stage_p: cudaz.CUdeviceptr, d_records_p: cudaz.CUdeviceptr) kvk.KvarnStoreD256Args {
            return .{
                .current = @ptrFromInt(d_k_p),
                .current_v = @ptrFromInt(d_v_p),
                .indices = @ptrFromInt(d_indices_p),
                .stage = @ptrFromInt(d_stage_p),
                .records = @ptrFromInt(d_records_p),
                .n_tokens = @intCast(n_tokens),
                .n_logical_heads = 2,
                .n_record_heads = 4,
                .stream = 0,
                .groups_per_stream = @intCast(gps),
                .record_bytes = @intCast(lay.tile_bytes),
                .k_payload_off = @intCast(lay.k_payload_off),
                .k_s_col_off = @intCast(lay.k_s_col_off),
                .k_zp_off = @intCast(lay.k_zp_off),
                .k_s_row_off = @intCast(lay.k_s_row_off),
                .v_payload_off = @intCast(lay.v_payload_off),
                .v_s_col_off = @intCast(lay.v_s_col_off),
                .v_s_row_off = @intCast(lay.v_s_row_off),
                .v_zp_off = @intCast(lay.v_zp_off),
                .k_bits = 5,
                .v_bits = 4,
                .sinkhorn_iters = 16,
                .stage_groups = @intCast(sg),
                .tail_groups = @intCast(tg),
                .swa = 0,
                .eager_records = 1,
            };
        }
    }.go;

    // === Modo A/B: ZIG_AI_REPRO_ONE_SHOT=1 ⇒ store único 6 tokens ===
    // ZIG_AI_REPRO_CFG=test ⇒ geometría del test original (gps=1, sg=2, tg=1)
    // vs default E2E (gps=512, sg=4, tg=3).
    const cfg_raw = std.c.getenv("ZIG_AI_REPRO_CFG");
    const cfg_test = if (cfg_raw) |v| std.mem.eql(u8, v[0..std.mem.len(v)], "test") else false;
    const cfg_gps: usize = if (cfg_test) 1 else groups_per_stream;
    const cfg_sg: usize = if (cfg_test) 2 else stage_groups;
    const cfg_tg: usize = if (cfg_test) 1 else tail_groups;
    const one_shot = std.c.getenv("ZIG_AI_REPRO_ONE_SHOT") != null;
    if (one_shot) {
        try cudaz.cuMemcpyHtoD(d_indices, @intFromPtr(h_indices.ptr), indices_len * 8);
        try cudaz.cuMemcpyHtoD(d_k, @intFromPtr(k_pre.ptr), KV_DIM * n_pre * 4);
        try cudaz.cuMemcpyHtoD(d_v, @intFromPtr(v_pre.ptr), KV_DIM * n_pre * 4);
        // token 5 tras los 5 del prefill en el MISMO buffer current.
        try cudaz.cuMemcpyHtoD(d_k + KV_DIM * n_pre * 4, @intFromPtr(k_dec.ptr), KV_DIM * 4);
        try cudaz.cuMemcpyHtoD(d_v + KV_DIM * n_pre * 4, @intFromPtr(v_dec.ptr), KV_DIM * 4);
        h_indices[n_pre] = @intCast(n_pre);
        var sa6 = makeStore(6, d_k, d_v, layout128, cfg_gps, cfg_sg, cfg_tg, d_indices, d_stage, d_records);
        try kvk.kvarnStoreD256Device(module_kv, &sa6, stream);
        var ia6: kvk.KvarnInitDescsArgs = .{
            .n_stream = 1,
            .n_indices = 6,
            .d_indices = @ptrFromInt(d_indices),
            .d_descs = @ptrFromInt(d_descs),
            .desc_stride = @intCast(2 * n_kv_heads),
            .d_records = @ptrFromInt(d_records),
            .d_stage = @ptrFromInt(d_stage),
            .n_record_heads = @intCast(n_physical),
        .head_dim = 128,
            .groups_per_stream = @intCast(cfg_gps),
            .record_bytes = @intCast(record_bytes),
            .stage_groups = @intCast(cfg_sg),
            .tail_groups = @intCast(cfg_tg),
            .k_bits = 5,
            .v_bits = 4,
            .head_slices = 2,
            .eager_records = 1,
            .read_indirect = 0,
            .original_domain = 0,
            .swa = 0,
        };
        try kvk.kvarnInitDescsDevice(module_kv, &ia6, stream);
    } else {
        // === Secuencia E2E exacta: append(5,0) + descs; append(1,5) + descs ===
        try cudaz.cuMemcpyHtoD(d_indices, @intFromPtr(h_indices.ptr), indices_len * 8);
        try cudaz.cuMemcpyHtoD(d_k, @intFromPtr(k_pre.ptr), KV_DIM * n_pre * 4);
        try cudaz.cuMemcpyHtoD(d_v, @intFromPtr(v_pre.ptr), KV_DIM * n_pre * 4);
        {
            var sa = makeStore(n_pre, d_k, d_v, layout128, cfg_gps, cfg_sg, cfg_tg, d_indices, d_stage, d_records);
            try kvk.kvarnStoreD256Device(module_kv, &sa, stream);
        }
        {
            var ia: kvk.KvarnInitDescsArgs = .{
                .n_stream = 1,
                .n_indices = @intCast(n_pre),
                .d_indices = @ptrFromInt(d_indices),
                .d_descs = @ptrFromInt(d_descs),
                .desc_stride = @intCast(2 * n_kv_heads),
                .d_records = @ptrFromInt(d_records),
                .d_stage = @ptrFromInt(d_stage),
                .n_record_heads = @intCast(n_physical),
        .head_dim = 128,
                .groups_per_stream = @intCast(cfg_gps),
                .record_bytes = @intCast(record_bytes),
                .stage_groups = @intCast(cfg_sg),
                .tail_groups = @intCast(cfg_tg),
                .k_bits = 5,
                .v_bits = 4,
                .head_slices = 2,
                .eager_records = 1,
                .read_indirect = 0,
                .original_domain = 0,
                .swa = 0,
            };
            try kvk.kvarnInitDescsDevice(module_kv, &ia, stream);
        }
        // Decode: solo el token 5 en current; indices[5]=5.
        try cudaz.cuMemcpyHtoD(d_k, @intFromPtr(k_dec.ptr), KV_DIM * 4);
        try cudaz.cuMemcpyHtoD(d_v, @intFromPtr(v_dec.ptr), KV_DIM * 4);
        {
            var sa = makeStore(1, d_k, d_v, layout128, cfg_gps, cfg_sg, cfg_tg, d_indices, d_stage, d_records);
            try kvk.kvarnStoreD256Device(module_kv, &sa, stream);
        }
        {
            var ia: kvk.KvarnInitDescsArgs = .{
                .n_stream = 1,
                .n_indices = @intCast(n_tot),
                .d_indices = @ptrFromInt(d_indices),
                .d_descs = @ptrFromInt(d_descs),
                .desc_stride = @intCast(2 * n_kv_heads),
                .d_records = @ptrFromInt(d_records),
                .d_stage = @ptrFromInt(d_stage),
                .n_record_heads = @intCast(n_physical),
        .head_dim = 128,
                .groups_per_stream = @intCast(cfg_gps),
                .record_bytes = @intCast(record_bytes),
                .stage_groups = @intCast(cfg_sg),
                .tail_groups = @intCast(cfg_tg),
                .k_bits = 5,
                .v_bits = 4,
                .head_slices = 2,
                .eager_records = 1,
                .read_indirect = 0,
                .original_domain = 0,
                .swa = 0,
            };
            try kvk.kvarnInitDescsDevice(module_kv, &ia, stream);
        }
    }
    try cudaz.cuStreamSynchronize(stream);

    // Fattn D256 con la config del engine.
    var attn_args: fattn_kv.KvarnAttentionArgs = .{
        .q_data = @ptrFromInt(d_q),
        .k_descs = @ptrFromInt(d_descs),
        .v_descs = @ptrFromInt(d_descs + @as(usize, @sizeOf(kvk.KvarnDesc)) * n_kv_heads),
        .mask_data = null,
        .dst_data = @ptrFromInt(d_dst),
        .n_kv = @intCast(n_tot),
        .n_q = 1,
        .n_q_heads = @intCast(n_q_heads),
        .n_kv_heads = @intCast(n_kv_heads),
        .n_stream = 1,
        .scale = 1.0 / @sqrt(@as(f32, 256.0)),
        .gqa = @intCast(gqa),
    };
    _ = try fattn_kv.fattnKvarnPortableD256Device(module_fa, &attn_args, stream);
    try cudaz.cuStreamSynchronize(stream);

    const out_host = try testing.allocator.alloc(f32, Q_DIM);
    defer testing.allocator.free(out_host);
    try cudaz.cuMemcpyDtoH(@intFromPtr(out_host.ptr), d_dst, Q_DIM * 4);

    // CPU ref exacta con K/V/Q reales.
    const K_all = try testing.allocator.alloc(f32, n_tot * KV_DIM);
    defer testing.allocator.free(K_all);
    @memcpy(K_all[0 .. 5 * KV_DIM], k_pre);
    @memcpy(K_all[5 * KV_DIM ..], k_dec);
    const V_all = try testing.allocator.alloc(f32, n_tot * KV_DIM);
    defer testing.allocator.free(V_all);
    @memcpy(V_all[0 .. 5 * KV_DIM], v_pre);
    @memcpy(V_all[5 * KV_DIM ..], v_dec);

    const cpu_out = try testing.allocator.alloc(f32, Q_DIM);
    defer testing.allocator.free(cpu_out);
    @memset(cpu_out, 0);
    const scale: f32 = 1.0 / @sqrt(@as(f32, 256.0));

    // ── Verificación intermedia: atención CPU sobre el STAGE real ──
    // Separa store/stage (si falla aquí ⇒ store) de fattn kernel.
    // Stage layout C2v2: [pos][2·n_physical][128] f16; K física f fila 2f,
    // V física f fila 2f+1. Head lógico kvh: físicas 2kvh (K slice 0/1 =
    // físicas 2kvh, 2kvh+1 tras cross).
    if (std.c.getenv("ZIG_AI_REPRO_STAGE_CHECK") != null) {
        const stage_f16 = try testing.allocator.alloc(f16, stage_len);
        defer testing.allocator.free(stage_f16);
        try cudaz.cuMemcpyDtoH(@intFromPtr(stage_f16.ptr), d_stage, stage_len * 2);
        // WHT CPU helpers
        const wht128 = struct {
            fn go(a: []f32, b: []const f32) void {
                @memcpy(a, b);
                var stride: usize = 1;
                while (stride < 128) : (stride *= 2) {
                    var j: usize = 0;
                    while (j < 128) : (j += 2 * stride) {
                        for (0..stride) |k| {
                            const a1 = a[j + k];
                            const b1 = a[j + k + stride];
                            a[j + k] = a1 + b1;
                            a[j + k + stride] = a1 - b1;
                        }
                    }
                }
                for (a) |*x| x.* *= 0.08838834764831845;
            }
        }.go;
        var st_out = try testing.allocator.alloc(f32, Q_DIM);
        defer testing.allocator.free(st_out);
        @memset(st_out, 0);
        const q_rot = try testing.allocator.alloc(f32, D);
        defer testing.allocator.free(q_rot);
        var k_sl: [2][]f32 = undefined;
        const k_sl0 = try testing.allocator.alloc(f32, 128);
        defer testing.allocator.free(k_sl0);
        const k_sl1 = try testing.allocator.alloc(f32, 128);
        defer testing.allocator.free(k_sl1);
        k_sl[0] = k_sl0;
        k_sl[1] = k_sl1;
        for (0..n_q_heads) |qh| {
            const kvh = qh / gqa;
            // Q rot: intra-128 + cross-slice
            for (0..2) |sl| wht128(q_rot[sl * 128 .. (sl + 1) * 128], q_dec[qh * D + sl * 128 .. qh * D + sl * 128 + 128]);
            for (0..128) |i| {
                const a = q_rot[i];
                const b = q_rot[128 + i];
                q_rot[i] = (a + b) * 0.70710678118654752440;
                q_rot[128 + i] = (a - b) * 0.70710678118654752440;
            }
            var scores: [6]f32 = undefined;
            var v_rot_acc: [2][128]f32 = .{ .{0} ** 128, .{0} ** 128 };
            // v_acc en dominio ROTADO
            var v_acc_rot: [2][128]f32 = .{ .{0} ** 128, .{0} ** 128 };
            for (0..n_tot) |t| {
                // K del stage: físicas 2kvh (slice0), 2kvh+1 (slice1) —
                // filas 2·f y 2·f+1? NO: K física f fila 2f; la física de K
                // slice s del lógico kvh es f = 2kvh+s ⇒ fila 2·(2kvh+s).
                for (0..2) |sl| {
                    const f = 2 * kvh + sl;
                    const row = 2 * f; // K
                    const pos_base = (t) * (2 * n_physical) * 128;
                    for (0..128) |i| k_sl[sl][i] = @floatCast(stage_f16[pos_base + row * 128 + i]);
                }
                var s: f32 = 0;
                for (0..128) |i| {
                    s += q_rot[i] * k_sl[0][i] + q_rot[128 + i] * k_sl[1][i];
                }
                scores[t] = s * scale;
            }
            var mx = scores[0];
            for (scores[1..]) |s| mx = @max(mx, s);
            var z: f32 = 0;
            for (scores) |s| z += @exp(s - mx);
            for (0..n_tot) |t| {
                const w = @exp(scores[t] - mx) / z;
                // V stage: física f = 2kvh+sl ⇒ fila 2f+1.
                for (0..2) |sl| {
                    const f = 2 * kvh + sl;
                    const row = 2 * f + 1; // V
                    const pos_base = (t) * (2 * n_physical) * 128;
                    for (0..128) |i| v_acc_rot[sl][i] += w * @as(f32, @floatCast(stage_f16[pos_base + row * 128 + i]));
                }
            }
            // De-rot V acc: cross inverso + intra inverso (WHT self-inverse)
            for (0..128) |i| {
                const a = v_acc_rot[0][i];
                const b = v_acc_rot[1][i];
                v_rot_acc[0][i] = (a + b) * 0.70710678118654752440;
                v_rot_acc[1][i] = (a - b) * 0.70710678118654752440;
            }
            for (0..2) |sl| {
                wht128(st_out[qh * D + sl * 128 .. qh * D + sl * 128 + 128], v_rot_acc[sl][0..]);
            }
        }
        var st_max_diff: f64 = 0;
        for (st_out, cpu_out) |g, w| st_max_diff = @max(st_max_diff, @abs(@as(f64, g) - @as(f64, w)));
        std.debug.print("9.4 repro STAGE-CHECK: max_diff(stage-CPU vs ref)={d:.6}\n", .{st_max_diff});
        // Dump del stage crudo para análisis externo (layout C2v2).
        if (std.c.getenv("ZIG_AI_REPRO_STAGE_DUMP")) |sd| {
            const io = std.Io.Threaded.global_single_threaded.io();
            var pb2: [512]u8 = undefined;
            const path = try std.fmt.bufPrintZ(&pb2, "{s}/stage.f16", .{sd[0..std.mem.len(sd)]});
            const sf = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
            defer sf.close(io);
            var wb: [8192]u8 = undefined;
            var w = sf.writer(io, &wb);
            const wr = &w.interface;
            try wr.writeAll(std.mem.sliceAsBytes(stage_f16));
            try wr.flush();
        }
    }

    for (0..n_q_heads) |qh| {
        const kvh = qh / gqa;
        var scores: [6]f32 = undefined;
        for (0..n_tot) |t| {
            var s: f32 = 0;
            for (0..D) |i| s += q_dec[qh * D + i] * K_all[t * KV_DIM + kvh * D + i];
            scores[t] = s * scale;
        }
        var mx = scores[0];
        for (scores[1..]) |s| mx = @max(mx, s);
        var z: f32 = 0;
        for (scores) |s| z += @exp(s - mx);
        for (0..n_tot) |t| {
            const w = @exp(scores[t] - mx) / z;
            for (0..D) |i| cpu_out[qh * D + i] += w * V_all[t * KV_DIM + kvh * D + i];
        }
    }

    var max_rel: f64 = 0;
    var max_diff: f64 = 0;
    var bad: usize = 0;
    for (out_host, cpu_out) |got, want| {
        const diff: f64 = @abs(@as(f64, got) - @as(f64, want));
        const rel = diff / @max(@abs(@as(f64, want)), 1e-6);
        max_rel = @max(max_rel, rel);
        max_diff = @max(max_diff, diff);
        // Criterio estándar kvarn (d256/m1seed): rel alto solo cuenta si el
        // diff abs también es significativo — elems |want|~1e-6 (softmax
        // de slices vacíos) dan rel enorme sin error real.
        if (rel > 5e-2 and diff > 1e-4) bad += 1;
    }
    std.debug.print("9.4 repro E2E: max_rel={d:.6} max_diff={d:.6} bad={d}\n", .{ max_rel, max_diff, bad });
    // Dump out_host + cpu_out para análisis de patrón.
    if (std.c.getenv("ZIG_AI_REPRO_OUT_DUMP")) |od| {
        const io = std.Io.Threaded.global_single_threaded.io();
        var pb3: [512]u8 = undefined;
        {
            const path = try std.fmt.bufPrintZ(&pb3, "{s}/fa_out.f32", .{od[0..std.mem.len(od)]});
            const sf = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
            defer sf.close(io);
            var wb: [8192]u8 = undefined;
            var w = sf.writer(io, &wb);
            const wr = &w.interface;
            try wr.writeAll(std.mem.sliceAsBytes(out_host));
            try wr.flush();
        }
        {
            const path = try std.fmt.bufPrintZ(&pb3, "{s}/cpu_ref.f32", .{od[0..std.mem.len(od)]});
            const sf = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
            defer sf.close(io);
            var wb: [8192]u8 = undefined;
            var w = sf.writer(io, &wb);
            const wr = &w.interface;
            try wr.writeAll(std.mem.sliceAsBytes(cpu_out));
            try wr.flush();
        }
    }
    try testing.expect(bad == 0);
}
