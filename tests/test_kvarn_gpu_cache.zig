//! KvarnGpuCache (lane-b1 Dev-A M3): superficie de alloc por capa +
//! validación de config. GPU-append e2e gated cubin (igual que los
//! otros tests kvarn).

const std = @import("std");
const testing = std.testing;
const build_options = @import("build_options");
const cudaz = @import("cudaz");
const kvk = @import("kvarn_kernels");
const kvg = @import("kvarn_gpu_cache");
const fattn_kv = @import("fattn_kvarn");
const bc = @import("backend_capabilities");

test "KvarnGpuCache: config validation (CPU)" {
    const allocator = testing.allocator;
    const c_d2 = kvg.KvarnGpuConfig{
        .num_layers = 4,
        .n_kv_heads = 2,
        .head_dim = 256, // D2 soportado desde 9.4
        .k_bits = 5,
        .v_bits = 4,
        .max_ctx_tokens = 2048,
    };
    // D256 válido desde 9.4: init exitoso ⇒ deinit OBLIGATORIO (el alloc
    // de `layers` es host-side; sin CUDA — module null, capas lazy).
    var cache_d2 = try kvg.KvarnGpuCache.init(allocator, c_d2, null, undefined); // D256 now valid
    cache_d2.deinit();
    const c_bad = kvg.KvarnGpuConfig{
        .num_layers = 4,
        .n_kv_heads = 2,
        .head_dim = 512, // 9.13: D64 soportado; 512 sigue gated (store D-slice)
        .k_bits = 5,
        .v_bits = 4,
        .max_ctx_tokens = 2048,
    };
    _ = try std.testing.expectError(error.UnsupportedHeadDim, kvg.KvarnGpuCache.init(allocator, c_bad, null, undefined));
    const c_zero = kvg.KvarnGpuConfig{
        .num_layers = 0,
        .n_kv_heads = 2,
        .head_dim = 128,
        .k_bits = 5,
        .v_bits = 4,
        .max_ctx_tokens = 2048,
    };
    _ = try std.testing.expectError(error.InvalidConfig, kvg.KvarnGpuCache.init(allocator, c_zero, null, undefined));
}

test "KvarnGpuCache: D64 geometry (CPU, no GPU) — 9.13" {
    const allocator = testing.allocator;
    const cfg = kvg.KvarnGpuConfig{
        .num_layers = 2,
        .n_kv_heads = 2,
        .head_dim = 64,
        .k_bits = 5,
        .v_bits = 4,
        .max_ctx_tokens = 512,
    };
    var cache = try kvg.KvarnGpuCache.init(allocator, cfg, null, undefined);
    defer cache.deinit();
    // D64: 1 slice físico por lógica — heads físicas == lógicas.
    try testing.expectEqual(@as(u32, 1), cache.headSlices());
    try testing.expectEqual(@as(u32, 2), cache.physicalHeads());
    // Stage: inner dim SIGUE siendo 128 (el kernel D64 escribe rows de
    // 128 con first64 usado — OOB si se reduce a 64).
    const expect_stage = @as(usize, 4) * 128 * (2 * 2) * 128;
    try testing.expectEqual(expect_stage, cache.stageLen());
    // Records: layout rect 64 (NO 128) por cabeza física.
    const lay64 = kvg.kvarn_types.KvarnRecordLayout.init(64, 5, 4) catch unreachable;
    const expect_records = @as(usize, 4) * 2 * lay64.tile_bytes;
    try testing.expectEqual(expect_records, cache.recordsLen());
}

test "KvarnGpuCache: geometry sanity (CPU, no GPU)" {
    const allocator = testing.allocator;
    // init NO toca CUDA si la config es válida — solo alloc del array
    // de capas. Verificamos deinit limpio con config mínima.
    const cfg = kvg.KvarnGpuConfig{
        .num_layers = 2,
        .n_kv_heads = 2,
        .head_dim = 128,
        .k_bits = 5,
        .v_bits = 4,
        .max_ctx_tokens = 512,
    };
    var cache = try kvg.KvarnGpuCache.init(allocator, cfg, null, undefined);
    defer cache.deinit();
    // Sin ensureLayer: tokenCount=0 y kDescs da LayerNotAllocated.
    try testing.expectEqual(@as(u32, 0), cache.tokenCount(0));
    _ = try std.testing.expectError(error.LayerNotAllocated, cache.kDescs(0));
    // groupsPerStream: 512 tokens ⇒ 4 grupos.
    try testing.expectEqual(@as(u32, 4), cache.groupsPerStream());
}

test "KvarnGpuCache: appendTokens e2e + descs + split attention (gated cubin)" {
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;
    if (build_options.kvarn_split_cubin.len == 0) return error.SkipZigTest;

    try cudaz.ensureContext();
    const kmod = try cudaz.cuModuleLoad(build_options.kvarn_cubin);
    const smod = try cudaz.cuModuleLoad(build_options.kvarn_split_cubin);

    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);
    const allocator = testing.allocator;

    // 2 capas, 1 kv head, 128 dims, ctx 256 (2 grupos), k5v4.
    const cfg = kvg.KvarnGpuConfig{
        .num_layers = 2,
        .n_kv_heads = 1,
        .head_dim = 128,
        .k_bits = 5,
        .v_bits = 4,
        .max_ctx_tokens = 256,
    };
    var cache = try kvg.KvarnGpuCache.init(allocator, cfg, kmod, stream);
    defer cache.deinit();

    // current_k: [256 tok × 1 head × 128] f32 dominio ORIGINAL.
    var prng = std.Random.DefaultPrng.init(0x6C1E);
    const rand = prng.random();
    const n_tokens: usize = 256;
    const current = try allocator.alloc(f32, n_tokens * 128);
    defer allocator.free(current);
    for (current) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
    const d_current = try cudaz.cuMemAlloc(@sizeOf(f32) * current.len);
    defer cudaz.cuMemFree(d_current);
    try cudaz.cuMemcpyHtoD(d_current, @intFromPtr(current.ptr), @sizeOf(f32) * current.len);

    // Append en DOS tandas (prefill 128 + decode 128) a la capa 0.
    try cache.appendTokens(0, d_current, null, 128, 0, null, stream);
    try cache.appendTokens(0, d_current + @sizeOf(f32) * 128 * 128, null, 128, 128, null, stream);
    try cudaz.cuStreamSynchronize(stream);
    try testing.expectEqual(@as(u32, 256), cache.tokenCount(0));

    // La capa 1 NO se tocó: sigue sin alloc.
    try testing.expectEqual(@as(u32, 0), cache.tokenCount(1));

    // Descs válidos: sparse-check del contenido (live_group=1 tras
    // 256 tokens, eager ⇒ live_pos=127).
    {
        const descs = try allocator.alloc(kvk.KvarnDesc, 2);
        defer allocator.free(descs);
        try cudaz.cuMemcpyDtoH(@intFromPtr(descs.ptr), try cache.kDescs(0), @sizeOf(kvk.KvarnDesc) * 2);
        try testing.expectEqual(@as(c_int, 1), descs[0].live_group);
        try testing.expectEqual(@as(c_int, 127), descs[0].live_pos);
        try testing.expectEqual(@as(c_int, 1), descs[1].value); // V side
    }

    // E2E: split attention sobre la capa 0 con Q random.
    const q_size: usize = 2 * 1 * 128; // 2 q_heads × 1 q × 128
    const q = try allocator.alloc(f32, q_size);
    defer allocator.free(q);
    for (q) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
    const d_q = try cudaz.cuMemAlloc(@sizeOf(f32) * q_size);
    defer cudaz.cuMemFree(d_q);
    try cudaz.cuMemcpyHtoD(d_q, @intFromPtr(q.ptr), @sizeOf(f32) * q_size);
    const d_dst = try cudaz.cuMemAlloc(@sizeOf(f32) * q_size);
    defer cudaz.cuMemFree(d_dst);

    const n_splits: usize = (256 + 63) / 64;
    const partial_len: usize = 1 * 2 * n_splits * 128;
    const meta_len: usize = 1 * 2 * n_splits;
    const d_partial = try cudaz.cuMemAlloc(@sizeOf(f32) * partial_len);
    defer cudaz.cuMemFree(d_partial);
    const d_meta = try cudaz.cuMemAlloc(@sizeOf(f32) * 2 * meta_len);
    defer cudaz.cuMemFree(d_meta);

    var sargs: kvk.KvarnSplitLaunchArgs = .{
        .q_data = d_q,
        .k_descs = try cache.kDescs(0),
        .v_descs = try cache.vDescs(0),
        .partial_data = d_partial,
        .meta_data = d_meta,
        .dst_data = d_dst,
        .n_kv = 256,
        .n_q = 1,
        .n_q_heads = 2,
        .n_kv_heads = 1,
        .n_splits = @intCast(n_splits),
        .scale = 1.0 / @sqrt(@as(f32, 128.0)),
    };
    _ = try kvk.kvarnDecodeSplitDevice(smod, &sargs, stream);
    try cudaz.cuStreamSynchronize(stream);

    // Salida finita (validación e2e de integridad; la exactitud del
    // split ya está gate-ada en test_kvarn_split con 1000 seeds).
    const out = try allocator.alloc(f32, q_size);
    defer allocator.free(out);
    try cudaz.cuMemcpyDtoH(@intFromPtr(out.ptr), d_dst, @sizeOf(f32) * q_size);
    for (out) |v| try testing.expect(std.math.isFinite(v));
}

// ============================================================================
// LADDER VRAM M3 (gate: −37% vs q8_0 en ctx largo, k2v2) — el −37%
// del PLAN_B1 es el régimen asintótico donde el stage fijo se amortiza.
// ============================================================================

test "M3 ladder VRAM: KVarN vs q8_0 (recordBytes reales, amortizado)" {
    // q8_0: 8 bits + 2B escala por bloque de 32 ⇒ 8.5 bit/elem.
    const q8_0_bpt: f64 = 128.0 * 8.5 / 8.0; // 136 B/token/head
    const fp16_bpt: f64 = 128.0 * 2.0; // 256 B/token/head

    // Tabla por config usando recordBytes REAL (C1 layout B2):
    const cases = [_]struct { kb: u8, vb: u8, name: []const u8 }{
        .{ .kb = 5, .vb = 4, .name = "k5v4" },
        .{ .kb = 4, .vb = 4, .name = "k4v4" },
        .{ .kb = 3, .vb = 3, .name = "k3v3" },
        .{ .kb = 2, .vb = 2, .name = "k2v2" },
    };
    for (cases) |c| {
        const rb = kvg.kvarn_types.KvarnRecordLayout.init(128, c.kb, c.vb) catch unreachable;
        const per_token: f64 = @as(f64, @floatFromInt(rb.tile_bytes)) / 128.0;
        std.debug.print("ladder {s}: {d:.1} B/token/head (payload) vs q8_0 {d:.1} = {d:.1}%\n", .{
            c.name, per_token, q8_0_bpt, (per_token / q8_0_bpt - 1.0) * 100.0,
        });
        // Invariante: TODO config debe ser < fp16 (el punto del ladder).
        try testing.expect(per_token < fp16_bpt);
    }

    // Régimen asintótico k2v2 (el −37% del gate): VRAM total con
    // stage fijo amortizado, modelo 8 kv_heads, ctx 32k, 1 capa.
    const ctx_tokens: usize = 32768;
    const n_kv_heads: usize = 8;
    const rb2 = kvg.kvarn_types.KvarnRecordLayout.init(128, 2, 2) catch unreachable;
    const groups: usize = ctx_tokens / 128;
    const records: f64 = @as(f64, @floatFromInt(groups * n_kv_heads * rb2.tile_bytes));
    // Stage fijo (C2v2): UNA vez por (layer, stream): 4 grupos × 128 pos
    // × (2·kvh) filas × 128 dims × 2B.
    const stage: f64 = @as(f64, @floatFromInt(4 * 128 * (2 * n_kv_heads) * 128 * 2));
    const descs: f64 = @as(f64, @floatFromInt(2 * n_kv_heads * @sizeOf(kvk.KvarnDesc)));
    const total_kvarn = records + stage + descs;
    const total_q8_0: f64 = @as(f64, @floatFromInt(ctx_tokens * n_kv_heads)) * q8_0_bpt;
    const rel = (total_kvarn / total_q8_0 - 1.0) * 100.0;
    std.debug.print("ladder k2v2 ctx=32k kvh=8: kvarn={d:.1}MB q8_0={d:.1}MB ⇒ {d:.1}% (gate: ≤ −37%)\n", .{
        total_kvarn / 1e6, total_q8_0 / 1e6, rel,
    });
    try testing.expect(rel <= -37.0);

    // Sanity GQA-heavy: el ladder también debe ganar en payload puro
    // (records sin stage) para TODA config ≤ k4v4 vs q8_0:
    for ([_]u8{ 2, 3 }) |kb| {
        const rbl = kvg.kvarn_types.KvarnRecordLayout.init(128, kb, kb) catch unreachable;
        const pt: f64 = @as(f64, @floatFromInt(rbl.tile_bytes)) / 128.0;
        try testing.expect(pt < q8_0_bpt);
    }
}

// ============================================================================
// A5-adaptativo: appendTokens con smem_optin forzado a 64KB (sm_75-like)
// debe elegir LOWSHMEM y producir descs/live idénticos al camino
// hishmem (records bit-idénticos — test A5 19968/19968).
// ============================================================================

test "M3 appendTokens adaptive: smem_optin 64KB (lowshmem) vs null (hishmem) — descs idénticos" {
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;
    if (build_options.kvarn_split_cubin.len == 0) return error.SkipZigTest;

    try cudaz.ensureContext();
    const kmod = try cudaz.cuModuleLoad(build_options.kvarn_cubin);

    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);
    const allocator = testing.allocator;

    const cfg_base = kvg.KvarnGpuConfig{
        .num_layers = 1,
        .n_kv_heads = 1,
        .head_dim = 128,
        .k_bits = 5,
        .v_bits = 4,
        .max_ctx_tokens = 256,
    };

    // Datos: 128 tokens aleatorios K/V.
    var prng = std.Random.DefaultPrng.init(0xA5AD);
    const rand = prng.random();
    const n: usize = 128;
    const cur_k = try allocator.alloc(f32, n * 128);
    defer allocator.free(cur_k);
    const cur_v = try allocator.alloc(f32, n * 128);
    defer allocator.free(cur_v);
    for (cur_k) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
    for (cur_v) |*x| x.* = rand.float(f32) * 0.5 - 0.25;

    const d_k = try cudaz.cuMemAlloc(@sizeOf(f32) * cur_k.len);
    defer cudaz.cuMemFree(d_k);
    const d_v = try cudaz.cuMemAlloc(@sizeOf(f32) * cur_v.len);
    defer cudaz.cuMemFree(d_v);
    try cudaz.cuMemcpyHtoD(d_k, @intFromPtr(cur_k.ptr), @sizeOf(f32) * cur_k.len);
    try cudaz.cuMemcpyHtoD(d_v, @intFromPtr(cur_v.ptr), @sizeOf(f32) * cur_v.len);

    // Pasada A: adaptive null (sm_86 ⇒ hishmem).
    var cache_a = try kvg.KvarnGpuCache.init(allocator, cfg_base, kmod, stream);
    defer cache_a.deinit();
    try cache_a.appendTokens(0, d_k, d_v, n, 0, null, stream);

    // Pasada B: smem_optin = 64KB (sm_75-like ⇒ lowshmem).
    var cfg_b = cfg_base;
    cfg_b.smem_optin = 64 * 1024;
    var cache_b = try kvg.KvarnGpuCache.init(allocator, cfg_b, kmod, stream);
    defer cache_b.deinit();
    try cache_b.appendTokens(0, d_k, d_v, n, 0, null, stream);

    // Descs par K/V: tras 128 tokens, live_group=1, live_pos=127.
    const KvarnDescSz = @sizeOf(kvk.KvarnDesc);
    const descs_a = try allocator.alloc(u8, 2 * KvarnDescSz);
    defer allocator.free(descs_a);
    const descs_b = try allocator.alloc(u8, 2 * KvarnDescSz);
    defer allocator.free(descs_b);
    const da_ptr = try cache_a.kDescs(0);
    const db_ptr = try cache_b.kDescs(0);
    try cudaz.cuMemcpyDtoH(@intFromPtr(descs_a.ptr), da_ptr, 2 * KvarnDescSz);
    try cudaz.cuMemcpyDtoH(@intFromPtr(descs_b.ptr), db_ptr, 2 * KvarnDescSz);

    // hishmem vs lowshmem: descs de caches distintas ⇒ PTRS difieren
    // (allocs separados). Oráculo A5 = CONTENIDO de records (bit-exacto).
    const da0 = @as([*]align(1) const kvk.KvarnDesc, @ptrCast(descs_a.ptr))[0];
    const db0 = @as([*]align(1) const kvk.KvarnDesc, @ptrCast(descs_b.ptr))[0];
    try testing.expectEqual(da0.live_group, db0.live_group);
    try testing.expectEqual(da0.live_pos, db0.live_pos);
    try testing.expectEqual(da0.groups_per_stream, db0.groups_per_stream);
    try testing.expectEqual(da0.record_bytes, db0.record_bytes);
    try testing.expectEqual(da0.bits, db0.bits);
    try testing.expectEqual(da0.tail_groups, db0.tail_groups);

    // Records CONTENT: volcar desde cada cache y comparar bit a bit.
    const rec_len: usize = @as(usize, @intCast(da0.groups_per_stream)) * @as(usize, @intCast(da0.record_bytes));
    const rec_a = try allocator.alloc(u8, rec_len);
    defer allocator.free(rec_a);
    const rec_b = try allocator.alloc(u8, rec_len);
    defer allocator.free(rec_b);
    try cudaz.cuMemcpyDtoH(@intFromPtr(rec_a.ptr), @intFromPtr(da0.records), rec_len);
    try cudaz.cuMemcpyDtoH(@intFromPtr(rec_b.ptr), @intFromPtr(db0.records), rec_len);
    var eq: usize = 0;
    for (rec_a, rec_b) |a, b| {
        if (a == b) eq += 1;
    }
    std.debug.print("adaptive: records hishmem==lowshmem {d}/{d} B, live=({d},{d})\n", .{ eq, rec_len, da0.live_group, da0.live_pos });
    try testing.expectEqualSlices(u8, rec_a, rec_b);
    try testing.expect(da0.live_pos == 127);
}
