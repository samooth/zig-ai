const std = @import("std");
const pa = @import("paged_attention");

fn testConfig() pa.PagedConfig {
    return .{
        .block_size = 4,
        .num_blocks = 32,
        .head_dim = 8,
        .num_kv_heads = 2,
        .num_q_heads = 8,
        .dtype = .f32,
        .enable_prefix_cache = true,
        .max_seq_len = 64,
        .max_batch_size = 4,
    };
}

test "BlockAllocator alloc/free" {
    const gpa = std.testing.allocator;
    var alloc = try pa.BlockAllocator.init(gpa, 16, 4, 2, 8, .f32, false);
    defer alloc.deinit();
    try std.testing.expectEqual(@as(usize, 16), alloc.numTotal());
    try std.testing.expectEqual(@as(usize, 16), alloc.numFree());
    const b1 = alloc.alloc().?;
    try std.testing.expectEqual(@as(usize, 15), alloc.numFree());
    alloc.acquire(b1);
    try std.testing.expectEqual(@as(u32, 1), alloc.blocks[b1].ref_count);
    alloc.release(b1);
    try std.testing.expectEqual(@as(usize, 16), alloc.numFree());
}

test "BlockTable append and COW" {
    const gpa = std.testing.allocator;
    var alloc = try pa.BlockAllocator.init(gpa, 16, 4, 2, 8, .f32, false);
    defer alloc.deinit();
    var bt = pa.BlockTable.init(gpa, 4);
    defer bt.deinit(&alloc);
    try bt.appendTokens(&alloc, 5);
    try std.testing.expectEqual(@as(usize, 2), bt.numBlocks());
    try std.testing.expectEqual(@as(usize, 5), bt.num_tokens);
    var bt2 = try bt.fork(&alloc);
    defer bt2.deinit(&alloc);
    try std.testing.expectEqual(@as(u32, 2), alloc.blocks[bt.getPhysical(0).?].ref_count);
    try bt2.prepareWrite(&alloc);
}

test "PagedKVCache sequence lifecycle" {
    const gpa = std.testing.allocator;
    var cache = try pa.PagedKVCache.init(gpa, testConfig());
    defer cache.deinit();
    const seq1 = try cache.createSequence();
    try cache.allocatePrefill(seq1, 7);
    try std.testing.expectEqual(@as(usize, 2), cache.getBlockTable(seq1).?.numBlocks());
    const seq2 = try cache.forkSequence(seq1);
    try std.testing.expectEqual(@as(usize, 2), cache.getBlockTable(seq2).?.numBlocks());
    cache.removeSequence(seq1);
    cache.removeSequence(seq2);
}

test "PrefixCache basic operations" {
    const gpa = std.testing.allocator;
    var pc = try pa.PrefixCache.init(gpa, 100);
    defer pc.deinit();
    try pc.insert(0x1234, 5, 16);
    try std.testing.expectEqual(@as(?usize, 5), pc.lookup(0x1234));
    try std.testing.expectEqual(@as(?usize, null), pc.lookup(0x9999));
}

test "Scheduler admit and batch" {
    const gpa = std.testing.allocator;
    var kv = try pa.PagedKVCache.init(gpa, testConfig());
    defer kv.deinit();
    var sched = pa.Scheduler.init(gpa, testConfig(), &kv);
    defer sched.deinit();
    const tokens = &[_]u32{ 1, 2, 3, 4, 5, 6, 7 };
    const req_id = try sched.submit(.{
        .prompt_tokens = tokens,
        .max_new_tokens = 4,
        .num_samples = 1,
    });
    _ = req_id;
    const batch = try sched.schedule();
    try std.testing.expect(batch.len > 0);
}

test "PagedAttention decode correctness" {
    const gpa = std.testing.allocator;
    var kv = try pa.PagedKVCache.init(gpa, testConfig());
    defer kv.deinit();
    const seq_id = try kv.createSequence();
    try kv.allocatePrefill(seq_id, 4);
    const bt = kv.getBlockTableMut(seq_id).?;
    const head_dim = 8;
    const num_kv_heads = 2;
    const block_size = 4;
    for (bt.table.items) |phys_id| {
        const data = kv.getBlockData(phys_id);
        const fdata = @as([*]f32, @ptrCast(@alignCast(data)))[0 .. block_size * num_kv_heads * head_dim * 2];
        for (fdata, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(i % 7)) * 0.1;
        }
    }
    const attn = pa.PagedAttention.init(gpa, testConfig());
    const query = try gpa.alloc(f32, 8 * head_dim);
    defer gpa.free(query);
    @memset(query, 0.5);
    const out = try gpa.alloc(f32, 8 * head_dim);
    defer gpa.free(out);
    try attn.decode(query, out, bt, &kv.block_alloc);
    for (out) |v| {
        try std.testing.expect(std.math.isFinite(v));
    }
}
