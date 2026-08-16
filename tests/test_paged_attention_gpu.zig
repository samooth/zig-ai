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

    var engine = try pa.PagedAttentionGpu.init(gpa, config);
    defer engine.deinit();
    try engine.decode(query, out_gpu, kv.getBlockTable(seq_id).?, kv.block_alloc);

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

    var engine = try pa.PagedAttentionGpu.init(gpa, config);
    defer engine.deinit();
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