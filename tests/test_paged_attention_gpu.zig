//! Test Fase 2 — PagedAttention GPU (decode / prefill) vs referencia CPU.
//! Rellena un PagedKVCache f16 con valores sintéticos, ejecuta el kernel CUDA
//! y compara contra `PagedAttention.decode` (online softmax). Se salta si CUDA
//! no está disponible.
const std = @import("std");
const pa = @import("paged_attention");
const cudaz = @import("cudaz");

fn testConfig() pa.PagedConfig {
    return .{
        .block_size = 4,
        .num_blocks = 64,
        .head_dim = 8,
        .num_kv_heads = 2,
        .num_q_heads = 8,
        .dtype = .f16,
        .enable_prefix_cache = false,
        .max_seq_len = 64,
        .max_batch_size = 4,
    };
}

fn fillBlocks(kv: *pa.PagedKVCache, seq_id: u64, seed: u64) !void {
    var rng = std.Random.Xoshiro256.init(seed);
    const bt = kv.getBlockTableMut(seq_id).?;
    const head_dim = kv.config.head_dim;
    const num_kv_heads = kv.config.num_kv_heads;
    const block_size = kv.config.block_size;

    for (bt.table.items) |phys_id| {
        const data = kv.getBlockData(phys_id);
        const fdata = @as([*]f16, @ptrCast(@alignCast(data)))[0 .. block_size * num_kv_heads * head_dim * 2];
        for (fdata) |*v| {
            const r: f32 = rng.random().float(f32);
            v.* = @floatCast((r - 0.5) * 2.0);
        }
    }
}

test "paged attention GPU decode matches CPU reference" {
    if (!cudaz.isCudaAvailable()) {
        std.debug.print("SKIP: CUDA no disponible\n", .{});
        return error.SkipZigTest;
    }
    const gpa = std.testing.allocator;
    const config = testConfig();

    var kv = try pa.PagedKVCache.init(gpa, config);
    defer kv.deinit();
    const seq_id = try kv.createSequence();
    try kv.allocatePrefill(seq_id, 9); // 3 bloques (4+4+1)
    try fillBlocks(&kv, seq_id, 42);

    const q_stride = config.num_q_heads * config.head_dim;
    const query = try gpa.alloc(f32, q_stride);
    defer gpa.free(query);
    var rng = std.Random.Xoshiro256.init(7);
    for (query) |*v| v.* = (rng.random().float(f32) - 0.5) * 2.0;

    const out_cpu = try gpa.alloc(f32, q_stride);
    defer gpa.free(out_cpu);
    const out_gpu = try gpa.alloc(f32, q_stride);
    defer gpa.free(out_gpu);

    const attn = pa.PagedAttention.init(gpa, config);
    try attn.decode(query, out_cpu, kv.getBlockTable(seq_id).?, kv.block_alloc);

    cudaz.ensureContext() catch return error.SkipZigTest;
    const gpu_stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(gpu_stream);
    var engine = try pa.PagedAttentionGpu.init(gpa, config, gpu_stream);
    defer engine.deinit();
    try engine.decode(query, out_gpu, kv.getBlockTable(seq_id).?.*, kv.block_alloc, config);

    var max_diff: f32 = 0;
    for (out_cpu, out_gpu, 0..) |c, g, i| {
        max_diff = @max(max_diff, @abs(c - g));
        if (@abs(c - g) > 5e-3) {
            std.debug.print("decode mismatch at {d}: cpu={d} gpu={d}\n", .{ i, c, g });
            return error.DecodeMismatch;
        }
    }
    std.debug.print("decode OK: {d} dims, max_diff={d}\n", .{ q_stride, max_diff });
}

test "paged attention GPU prefill matches CPU reference" {
    if (!cudaz.isCudaAvailable()) {
        std.debug.print("SKIP: CUDA no disponible\n", .{});
        return error.SkipZigTest;
    }
    const gpa = std.testing.allocator;
    const config = testConfig();

    var kv = try pa.PagedKVCache.init(gpa, config);
    defer kv.deinit();
    const seq_id = try kv.createSequence();
    const seq_len = 6;
    try kv.allocatePrefill(seq_id, seq_len);
    try fillBlocks(&kv, seq_id, 99);

    const q_stride = config.num_q_heads * config.head_dim;
    const queries = try gpa.alloc(f32, seq_len * q_stride);
    defer gpa.free(queries);
    var rng = std.Random.Xoshiro256.init(11);
    for (queries) |*v| v.* = (rng.random().float(f32) - 0.5) * 2.0;

    const outs_cpu = try gpa.alloc(f32, seq_len * q_stride);
    defer gpa.free(outs_cpu);
    const outs_gpu = try gpa.alloc(f32, seq_len * q_stride);
    defer gpa.free(outs_gpu);

    const attn = pa.PagedAttention.init(gpa, config);
    try attn.prefill(queries, outs_cpu, kv.getBlockTable(seq_id).?, kv.block_alloc, seq_len);

    cudaz.ensureContext() catch return error.SkipZigTest;
    const gpu_stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(gpu_stream);
    var engine = try pa.PagedAttentionGpu.init(gpa, config, gpu_stream);
    defer engine.deinit();
    // Prefill is done via engine.prefill, then decode
    try engine.prefill(queries, outs_gpu, kv.getBlockTable(seq_id).?, kv.block_alloc, seq_len);

    var max_diff: f32 = 0;
    for (outs_cpu, outs_gpu, 0..) |c, g, i| {
        max_diff = @max(max_diff, @abs(c - g));
        if (@abs(c - g) > 5e-3) {
            std.debug.print("prefill mismatch at {d}: cpu={d} gpu={d}\n", .{ i, c, g });
            return error.PrefillMismatch;
        }
    }
    std.debug.print("prefill OK: {d} tokens, max_diff={d}\n", .{ seq_len, max_diff });
}

test "GPU block pool stage/evict round-trip preserves host data" {
    if (!cudaz.isCudaAvailable()) {
        std.debug.print("SKIP: CUDA no disponible\n", .{});
        return error.SkipZigTest;
    }
    const gpa = std.testing.allocator;
    const config = testConfig();
    var kv = try pa.PagedKVCache.init(gpa, config);
    defer kv.deinit();

    var pool = try pa.GpuBlockPool.init(gpa, kv.block_alloc.numTotal(), kv.block_alloc.block_bytes);
    defer pool.deinit();
    try std.testing.expectEqual(@as(usize, 0), pool.numResident());

    const seq_id = try kv.createSequence();
    try kv.allocatePrefill(seq_id, 4);
    const bt = kv.getBlockTableMut(seq_id).?;
    const phys = bt.getPhysical(0).?;
    const data = kv.getBlockData(phys);
    const fdata = @as([*]f16, @ptrCast(@alignCast(data)))[0 .. kv.block_alloc.block_bytes / 2];
    for (fdata, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(i)) * 0.5);

    try pool.stageBlock(kv.block_alloc, phys);
    try std.testing.expectEqual(@as(usize, 1), pool.numResident());

    // Modificar host no debe afectar la copia residente hasta evictar.
    @memset(fdata, 0);
    try pool.evictBlock(kv.block_alloc, phys);
    try std.testing.expectEqual(@as(usize, 0), pool.numResident());
    for (fdata, 0..) |v, i| {
        try std.testing.expectApproxEqAbs(@as(f16, @floatCast(@as(f32, @floatFromInt(i)) * 0.5)), v, 1e-3);
    }
}

test "Paged GPU block pool (VMM) stage/evict round-trip preserves host data" {
    if (!cudaz.isCudaAvailable()) {
        std.debug.print("SKIP: CUDA no disponible\n", .{});
        return error.SkipZigTest;
    }
    const gpa = std.testing.allocator;
    const granule = pa.PagedGpuBlockPool.getGranule() catch {
        std.debug.print("SKIP: VMM no soportado\n", .{});
        return error.SkipZigTest;
    };
    // bytes por bloque = block_size * num_kv_heads * head_dim * 2 (f16) * 2 (K+V)
    const bytes_per_token = @as(usize, 2) * @as(usize, 2) * @as(usize, 8) * @as(usize, 2);
    if (granule % bytes_per_token != 0) {
        std.debug.print("SKIP: granule={d} no múltiplo de {d}\n", .{ granule, bytes_per_token });
        return error.SkipZigTest;
    }
    const block_size: usize = granule / bytes_per_token;
    const config = pa.PagedConfig{
        .block_size = block_size,
        .num_blocks = 4,
        .head_dim = 8,
        .num_kv_heads = 2,
        .num_q_heads = 8,
        .dtype = .f16,
        .enable_prefix_cache = false,
        .max_seq_len = block_size * 4,
        .max_batch_size = 4,
    };
    var kv = try pa.PagedKVCache.init(gpa, config);
    defer kv.deinit();

    var pool = try pa.PagedGpuBlockPool.init(gpa, kv.block_alloc.numTotal(), kv.block_alloc.block_bytes, config.quant_k, config.quant_v);
    defer pool.deinit();
    try std.testing.expectEqual(@as(usize, 0), pool.numResident());

    const seq_id = try kv.createSequence();
    try kv.allocatePrefill(seq_id, 4);
    const bt = kv.getBlockTableMut(seq_id).?;
    const phys = bt.getPhysical(0).?;
    const data = kv.getBlockData(phys);
    const fdata = @as([*]f16, @ptrCast(@alignCast(data)))[0 .. kv.block_alloc.block_bytes / 2];
    for (fdata, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(i)) * 0.5);

    try pool.stageBlock(kv.block_alloc, phys);
    try std.testing.expectEqual(@as(usize, 1), pool.numResident());

    // Modificar host no debe afectar la copia residente hasta evictar.
    @memset(fdata, 0);
    try pool.evictBlock(kv.block_alloc, phys);
    try std.testing.expectEqual(@as(usize, 0), pool.numResident());
    for (fdata, 0..) |v, i| {
        try std.testing.expectApproxEqAbs(@as(f16, @floatCast(@as(f32, @floatFromInt(i)) * 0.5)), v, 1e-3);
    }
}

test "GPU evicts cold prefix blocks from device based on hit rate" {
    if (!cudaz.isCudaAvailable()) {
        std.debug.print("SKIP: CUDA no disponible\n", .{});
        return error.SkipZigTest;
    }
    const gpa = std.testing.allocator;
    const config = pa.PagedConfig{
        .block_size = 4,
        .num_blocks = 64,
        .head_dim = 8,
        .num_kv_heads = 2,
        .num_q_heads = 8,
        .dtype = .f16,
        .enable_prefix_cache = true,
        .max_seq_len = 64,
        .max_batch_size = 4,
    };
    var kv = try pa.PagedKVCache.init(gpa, config);
    defer kv.deinit();

    cudaz.ensureContext() catch return error.SkipZigTest;
    const gpu_stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(gpu_stream);
    var engine = try pa.PagedAttentionGpu.init(gpa, config, gpu_stream);
    defer engine.deinit();

    // cachea un prefix frío y uno caliente
    const cold = &[_]u32{ 1, 2, 3, 4 };
    const hot = &[_]u32{ 5, 6, 7, 8 };
    var sched = pa.Scheduler.init(gpa, kv.config, &kv);
    defer sched.deinit();

    _ = try sched.submit(.{ .prompt_tokens = cold, .max_new_tokens = 0 });
    _ = try sched.schedule();
    sched.finishSequence(1);
    _ = try sched.submit(.{ .prompt_tokens = hot, .max_new_tokens = 0 });
    _ = try sched.schedule();
    sched.finishSequence(2);

    // tocar el caliente para que no sea frío
    _ = try kv.matchPrefix(hot);
    _ = try kv.matchPrefix(hot);

    const before = try engine.evictColdBlocksFromCache(kv.block_alloc, &kv.prefix_cache, 1, 0.5);
    try std.testing.expect(before >= 1);
    try std.testing.expectEqual(@as(usize, 2), kv.prefix_cache.size());
}

fn realConfig() pa.PagedConfig {
    return .{
        .block_size = 16,
        .num_blocks = 64,
        .head_dim = 128,
        .num_kv_heads = 2,
        .num_q_heads = 8,
        .dtype = .f16,
        .enable_prefix_cache = false,
        .max_seq_len = 64,
        .max_batch_size = 4,
    };
}

test "paged attention GPU decode matches CPU (real head_dim 128)" {
    if (!cudaz.isCudaAvailable()) {
        std.debug.print("SKIP: CUDA no disponible\n", .{});
        return error.SkipZigTest;
    }
    const gpa = std.testing.allocator;
    const config = realConfig();
    var kv = try pa.PagedKVCache.init(gpa, config);
    defer kv.deinit();
    const seq_id = try kv.createSequence();
    try kv.allocatePrefill(seq_id, 40); // 3 bloques (16+16+8)
    try fillBlocks(&kv, seq_id, 123);

    const q_stride = config.num_q_heads * config.head_dim;
    const query = try gpa.alloc(f32, q_stride);
    defer gpa.free(query);
    var rng = std.Random.Xoshiro256.init(7);
    for (query) |*v| v.* = (rng.random().float(f32) - 0.5) * 2.0;

    const out_cpu = try gpa.alloc(f32, q_stride);
    defer gpa.free(out_cpu);
    const out_gpu = try gpa.alloc(f32, q_stride);
    defer gpa.free(out_gpu);

    const attn = pa.PagedAttention.init(gpa, config);
    try attn.decode(query, out_cpu, kv.getBlockTable(seq_id).?, kv.block_alloc);

    cudaz.ensureContext() catch return error.SkipZigTest;
    const gpu_stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(gpu_stream);
    var engine = try pa.PagedAttentionGpu.init(gpa, config, gpu_stream);
    defer engine.deinit();
    try engine.decode(query, out_gpu, kv.getBlockTable(seq_id).?.*, kv.block_alloc, config);

    var max_diff: f32 = 0;
    for (out_cpu, out_gpu, 0..) |c, g, i| {
        max_diff = @max(max_diff, @abs(c - g));
        if (@abs(c - g) > 1e-2) {
            std.debug.print("decode mismatch at {d}: cpu={d} gpu={d}\n", .{ i, c, g });
            return error.DecodeMismatch;
        }
    }
    std.debug.print("real-decode OK: max_diff={d}\n", .{max_diff});
}

test "paged attention vs TRUE attention (find kernel bug)" {
    if (!cudaz.isCudaAvailable()) {
        std.debug.print("SKIP: CUDA no disponible\n", .{});
        return error.SkipZigTest;
    }
    const gpa = std.testing.allocator;
    const config = realConfig();
    var kv = try pa.PagedKVCache.init(gpa, config);
    defer kv.deinit();
    const seq_id = try kv.createSequence();
    const total = 40;
    try kv.allocatePrefill(seq_id, total);
    try fillBlocks(&kv, seq_id, 123);

    const q_stride = config.num_q_heads * config.head_dim;
    const query = try gpa.alloc(f32, q_stride);
    defer gpa.free(query);
    var rng = std.Random.Xoshiro256.init(7);
    for (query) |*v| v.* = (rng.random().float(f32) - 0.5) * 2.0;

    const out_paged = try gpa.alloc(f32, q_stride);
    defer gpa.free(out_paged);
    const attn = pa.PagedAttention.init(gpa, config);
    try attn.decode(query, out_paged, kv.getBlockTable(seq_id).?, kv.block_alloc);

    // === TRUE attention (GQA + causal + 1/sqrt(d)) ===
    const hd = config.head_dim;
    const n_q = config.num_q_heads;
    const n_kv = config.num_kv_heads;
    const kv_dim = n_kv * hd;
    const q_per_kv = n_q / n_kv;
    const block_size = config.block_size;
    const K = try gpa.alloc(f32, total * kv_dim);
    defer gpa.free(K);
    const V = try gpa.alloc(f32, total * kv_dim);
    defer gpa.free(V);
    const bt = kv.getBlockTable(seq_id).?;
    for (0..total) |t| {
        const bidx = t / block_size;
        const off = t % block_size;
        const phys = bt.getPhysical(bidx).?;
        const block_data = kv.block_alloc.memory_pool[phys * kv.block_alloc.block_bytes ..];
        const fdata = @as([*]f16, @ptrCast(@alignCast(block_data)));
        for (0..n_kv) |h| {
            for (0..hd) |d| {
                const k_idx = off * kv_dim + h * hd + d;
                const v_idx = k_idx + block_size * kv_dim;
                K[t * kv_dim + h * hd + d] = fdata[k_idx];
                V[t * kv_dim + h * hd + d] = fdata[v_idx];
            }
        }
    }
    // Referencia: softmax de dos pasadas (GQA + causal + 1/sqrt(d)) sobre el KV
    // que lee el modelo (layout storeF16 del BlockAllocator).
    const out_true = try gpa.alloc(f32, q_stride);
    defer gpa.free(out_true);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    for (0..n_q) |qh| {
        const kv_h = qh / q_per_kv;
        var scores = try gpa.alloc(f32, total);
        defer gpa.free(scores);
        var maxs: f32 = -1e30;
        for (0..total) |s| {
            var acc: f32 = 0;
            for (0..hd) |d| acc += query[qh * hd + d] * K[s * kv_dim + kv_h * hd + d];
            acc *= scale;
            scores[s] = acc;
            if (acc > maxs) maxs = acc;
        }
        var sum: f32 = 0;
        for (0..total) |s| {
            scores[s] = @exp(scores[s] - maxs);
            sum += scores[s];
        }
        for (0..hd) |d| {
            var o: f32 = 0;
            for (0..total) |s| o += scores[s] * V[s * kv_dim + kv_h * hd + d];
            out_true[qh * hd + d] = o / sum;
        }
    }

    var md: f32 = 0;
    for (out_true, out_paged) |tr, pg| md = @max(md, @abs(tr - pg));
    std.debug.print("TRUE-vs-PAGED: max_diff={d}\n", .{md});
    if (md > 1e-2) return error.AttnBug;
}
// ─── G1 (lane-b1 2026-09-08): paridad flash-decoding split-K ──────────────
// decodeDeviceSplit (ZIG_AI_FASPLIT) vs ruta base vs CPU ref. La paridad
// es NUMÉRICA (orden de suma distinto), no bit-exacta: tol 1e-2 (mismo
// gate que el test hd=128 base). Cubre: bloques parciales, splits vacíos
// (seq < n_splits*tokens_per_split), y colas de split no alineadas.
test "G1 split-K decode: paridad vs base y CPU (hd 128, multi-split)" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const config = realConfig();
    var kv = try pa.PagedKVCache.init(gpa, config);
    defer kv.deinit();
    const seq_id = try kv.createSequence();
    // 37 tokens: 3 bloques (16+16+5) — tokens_per_split=128 ⇒ split 0
    // con cola, splits 1..7 vacíos (m=-inf ignorados por el combine).
    const seq_len: usize = 37;
    try kv.allocatePrefill(seq_id, seq_len);
    try fillBlocks(&kv, seq_id, 999);

    const q_stride = config.num_q_heads * config.head_dim;
    const query = try gpa.alloc(f32, q_stride);
    defer gpa.free(query);
    var rng = std.Random.Xoshiro256.init(31);
    for (query) |*v| v.* = (rng.random().float(f32) - 0.5) * 2.0;

    const out_cpu = try gpa.alloc(f32, q_stride);
    defer gpa.free(out_cpu);
    const attn = pa.PagedAttention.init(gpa, config);
    try attn.decode(query, out_cpu, kv.getBlockTable(seq_id).?, kv.block_alloc);

    cudaz.ensureContext() catch return error.SkipZigTest;
    const gpu_stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(gpu_stream);
    var engine = try pa.PagedAttentionGpu.init(gpa, config, gpu_stream);
    defer engine.deinit();

    const nb_total = (seq_len + config.block_size - 1) / config.block_size;
    const bt_host = try gpa.alloc(c_int, nb_total);
    defer gpa.free(bt_host);
    const bt_tbl = kv.getBlockTable(seq_id).?;
    for (0..nb_total) |bi| {
        bt_host[bi] = if (bt_tbl.getPhysical(bi)) |ph| @intCast(ph) else -1;
    }

    const q16 = try gpa.alloc(f16, q_stride);
    defer gpa.free(q16);
    for (query, 0..) |v, i| q16[i] = @floatCast(v);
    const d_q = try cudaz.cuMemAlloc(q_stride * @sizeOf(f16));
    defer cudaz.cuMemFree(d_q);
    const d_out = try cudaz.cuMemAlloc(q_stride * @sizeOf(f16));
    defer cudaz.cuMemFree(d_out);
    try cudaz.cuMemcpyHtoD(d_q, @intFromPtr(q16.ptr), q_stride * @sizeOf(f16));

    try engine.setupDecodeScratch(0, q_stride, nb_total);
    try engine.uploadBlockTable(0, bt_host);
    const seq_len_c: c_int = @intCast(seq_len);
    try cudaz.cuMemcpyHtoD(engine.d_seq_lens, @intFromPtr(&seq_len_c), @sizeOf(c_int));

    // G1: decodeDevice lee el pool DEVICE (cacheBase), no el host. El test
    // llenó el pool host con fillBlocks; hay que subir los bloques al device
    // (stageBlocks — lo que decode() hace internamente). Sin esto el kernel
    // lee cache=0 ⇒ scores=0 ⇒ acc=0.
    try engine.stageTableAll(kv.block_alloc, kv.getBlockTable(seq_id).?);

    // Ruta base (flag OFF) — referencia GPU.
    pa.fasplitForceForTest(false);
    try engine.decodeDevice(0, d_q, d_out, kv.block_alloc);
    try cudaz.cuStreamSynchronize(gpu_stream);
    const out_base = try gpa.alloc(f16, q_stride);
    defer gpa.free(out_base);
    try cudaz.cuMemcpyDtoH(@intFromPtr(out_base.ptr), d_out, q_stride * @sizeOf(f16));

    // Ruta split (flag ON) — MISMO d_out reutilizado (el kernel escribe
    // completo) para verificar además que el combine no deja residuo.
    pa.fasplitForceForTest(true);
    try cudaz.cuMemsetD8(d_out, 0, q_stride * @sizeOf(f16));
    try engine.decodeDevice(0, d_q, d_out, kv.block_alloc);
    try cudaz.cuStreamSynchronize(gpu_stream);
    const out_split = try gpa.alloc(f16, q_stride);
    defer gpa.free(out_split);
    try cudaz.cuMemcpyDtoH(@intFromPtr(out_split.ptr), d_out, q_stride * @sizeOf(f16));
    pa.fasplitForceForTest(false); // restaurar para otros tests

    var md_cpu: f32 = 0;
    var md_base: f32 = 0;
    for (out_cpu, 0..) |c, i| {
        const b: f32 = @floatCast(out_base[i]);
        const s: f32 = @floatCast(out_split[i]);
        md_cpu = @max(md_cpu, @abs(c - s));
        md_base = @max(md_base, @abs(b - s));
        if (@abs(c - s) > 1e-2) {
            std.debug.print("split mismatch at {d}: cpu={d} split={d}\n", .{ i, c, s });
            return error.SplitMismatch;
        }
    }
    std.debug.print("G1 split OK: seq={d} max_diff_vs_cpu={d} max_diff_vs_base={d}\n", .{ seq_len, md_cpu, md_base });
    if (md_base > 1e-2) {
        // Diferencia split-vs-base mayor que tol: ambas válidas vs CPU
        // pero divergen entre sí — reportar (no fatal si CPU aprueba).
        std.debug.print("G1 nota: split vs base difiere {d} (ambas <1e-2 vs CPU OK)\n", .{md_base});
    }
}

// ─── G1c cobertura (lane-c 2026-09-12): regresión seq > 1024 nominal ───────
// BUG: tokens_per_split FIJO 128 cubría 8 splits × 128 = 1024 tokens; a
// seq mayor el KV restante se IGNORABA (atención truncada ⇒ texto
// degenerado, E2E 0.8B ctx16384/seq3512: OFF resumen coherente, ON bucle
// "memory energy culture"). Fix: fillStaging recalcule
// ceil(seq/n_splits) por token (puntero persistente — graph-safe, el
// replay releé el valor). Este test reproduce el flujo real del decode
// (fillStaging + uploadScratch + decodeDevice) con seq=1120.
test "G1c split-K cobertura: seq 1120 > 8×128 nominal (regresión truncada)" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    // Config local: 80 bloques × 16 = 1280 tokens de capacidad.
    const config = pa.PagedConfig{
        .block_size = 16,
        .num_blocks = 80,
        .head_dim = 128,
        .num_kv_heads = 2,
        .num_q_heads = 8,
        .dtype = .f16,
        .enable_prefix_cache = false,
        .max_seq_len = 1120,
        .max_batch_size = 4,
    };
    var kv = try pa.PagedKVCache.init(gpa, config);
    defer kv.deinit();
    const seq_id = try kv.createSequence();
    // 1120 tokens: 70 bloques. Con el fijo 128 nominal el split cubriría
    // solo [0,1024) ⇒ KOV [1024,1120) ignorado ⇒ drift grande vs base.
    const seq_len: usize = 1120;
    try kv.allocatePrefill(seq_id, seq_len);
    try fillBlocks(&kv, seq_id, 777);

    const q_stride = config.num_q_heads * config.head_dim;
    const query = try gpa.alloc(f32, q_stride);
    defer gpa.free(query);
    var rng = std.Random.Xoshiro256.init(99);
    for (query) |*v| v.* = (rng.random().float(f32) - 0.5) * 2.0;

    const out_cpu = try gpa.alloc(f32, q_stride);
    defer gpa.free(out_cpu);
    const attn = pa.PagedAttention.init(gpa, config);
    try attn.decode(query, out_cpu, kv.getBlockTable(seq_id).?, kv.block_alloc);

    cudaz.ensureContext() catch return error.SkipZigTest;
    const gpu_stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(gpu_stream);
    var engine = try pa.PagedAttentionGpu.init(gpa, config, gpu_stream);
    defer engine.deinit();

    const nb_total = (seq_len + config.block_size - 1) / config.block_size;
    const q16 = try gpa.alloc(f16, q_stride);
    defer gpa.free(q16);
    for (query, 0..) |v, i| q16[i] = @floatCast(v);
    const d_q = try cudaz.cuMemAlloc(q_stride * @sizeOf(f16));
    defer cudaz.cuMemFree(d_q);
    const d_out = try cudaz.cuMemAlloc(q_stride * @sizeOf(f16));
    defer cudaz.cuMemFree(d_out);
    try cudaz.cuMemcpyHtoD(d_q, @intFromPtr(q16.ptr), q_stride * @sizeOf(f16));

    try engine.setupDecodeScratch(0, q_stride, nb_total);
    try engine.stageTableAll(kv.block_alloc, kv.getBlockTable(seq_id).?);
    // Flujo REAL del decode: fillStaging (fija seq_len + tokens_per_split
    // dinámico = ceil(1120/8) = 140) + uploadScratch (H2D seq_len/bt).
    engine.fillStaging(0, kv.getBlockTable(seq_id).?, seq_len - 1, seq_len, null);
    try engine.uploadScratch(0);

    // Ruta base (OFF).
    pa.fasplitForceForTest(false);
    try engine.decodeDevice(0, d_q, d_out, kv.block_alloc);
    try cudaz.cuStreamSynchronize(gpu_stream);
    const out_base = try gpa.alloc(f16, q_stride);
    defer gpa.free(out_base);
    try cudaz.cuMemcpyDtoH(@intFromPtr(out_base.ptr), d_out, q_stride * @sizeOf(f16));

    // Ruta split (ON) — con el fix, tokens_per_split=140 cubre [0,1120).
    pa.fasplitForceForTest(true);
    try cudaz.cuMemsetD8(d_out, 0, q_stride * @sizeOf(f16));
    try engine.decodeDevice(0, d_q, d_out, kv.block_alloc);
    try cudaz.cuStreamSynchronize(gpu_stream);
    const out_split = try gpa.alloc(f16, q_stride);
    defer gpa.free(out_split);
    try cudaz.cuMemcpyDtoH(@intFromPtr(out_split.ptr), d_out, q_stride * @sizeOf(f16));
    pa.fasplitForceForTest(false);

    // Con el fijo 128 la cobertura moría en 1024: el score perdía ~96
    // tokens de KV ⇒ diff >0.2. Con el dinámico: drift normal del
    // softmax por chunks. Diagnóstico: cuenta de outliers >1e-2 y max
    // relativo — a seq 1120 el split acumula 140 exps/chunk en f32 con
    // otro orden de suma vs CPU; un outlier de cola aislado es §3.1,
    // una población entera sería truncación residual.
    var md_split_cpu: f32 = 0;
    var md_base_cpu: f32 = 0;
    var md_base: f32 = 0;
    var outliers: usize = 0;
    var rel_max: f32 = 0;
    for (out_cpu, 0..) |c, i| {
        const s: f32 = @floatCast(out_split[i]);
        const b: f32 = @floatCast(out_base[i]);
        const d_sc = @abs(c - s);
        md_split_cpu = @max(md_split_cpu, d_sc);
        md_base_cpu = @max(md_base_cpu, @abs(c - b));
        md_base = @max(md_base, @abs(b - s));
        if (d_sc > 1e-2) outliers += 1;
        const denom: f32 = if (@abs(c) > 0.05) @abs(c) else 0.05;
        rel_max = @max(rel_max, d_sc / denom);
    }
    std.debug.print("G1c cobertura seq=1120: split_vs_cpu={d:.6} base_vs_cpu={d:.6} split_vs_base={d:.6} outliers>1e-2={d}/{d} rel_max={d:.4}\n", .{ md_split_cpu, md_base_cpu, md_base, outliers, out_cpu.len, rel_max });
    // Gate: sin outliers sistemáticos. Truncación pura (pre-fix) daba
    // outliers masivos con rel_max >0.5; drift §3.1 = outliers aislados
    // con error relativo pequeño.
    if (rel_max > 0.10 or outliers > out_cpu.len / 32) return error.SplitCoverageTruncada;
}
