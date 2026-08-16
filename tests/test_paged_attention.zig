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
    var alloc = try pa.BlockAllocator.init(gpa, 16, 4, 2, 8, .f32, false);
    defer alloc.deinit();
    var pc = try pa.PrefixCache.init(gpa, &alloc, 100);
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
    try attn.decode(query, out, bt, kv.block_alloc);
    for (out) |v| {
        try std.testing.expect(std.math.isFinite(v));
    }
}

test "PrefixCache blocks stay alive across sequence removal" {
    const gpa = std.testing.allocator;
    var kv = try pa.PagedKVCache.init(gpa, testConfig());
    defer kv.deinit();
    const tokens = &[_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const seq_id = try kv.createSequence();
    try kv.allocatePrefill(seq_id, 8);
    try kv.cachePrefix(seq_id, tokens);
    const cached_phys = kv.getBlockTable(seq_id).?.getPhysical(0).?;
    kv.removeSequence(seq_id);

    const matched = try kv.matchPrefix(tokens);
    try std.testing.expectEqual(@as(usize, 8), matched);
    const matched_phys = kv.prefix_cache.lookup(pa.hashTokens(tokens[0..4]));
    try std.testing.expectEqual(@as(?usize, cached_phys), matched_phys);
}

test "Scheduler reuses prefix blocks via cache" {
    const gpa = std.testing.allocator;
    var kv = try pa.PagedKVCache.init(gpa, testConfig());
    defer kv.deinit();
    var sched = pa.Scheduler.init(gpa, testConfig(), &kv);
    defer sched.deinit();

    const full_a = &[_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const full_b = &[_]u32{ 1, 2, 3, 4, 5, 6, 9, 9 };

    _ = try sched.submit(.{ .prompt_tokens = full_a, .max_new_tokens = 2 });
    _ = try sched.schedule();
    const a_phys = kv.getBlockTable(1).?.getPhysical(0).?;
    sched.finishSequence(1);

    _ = try sched.submit(.{ .prompt_tokens = full_b, .max_new_tokens = 2 });
    _ = try sched.schedule();
    const b_phys = kv.getBlockTable(2).?.getPhysical(0).?;
    try std.testing.expectEqual(a_phys, b_phys);
    try std.testing.expect(kv.getBlockTable(2).?.getPhysical(1) != kv.getBlockTable(2).?.getPhysical(0));
}

test "Scheduler preempts and restores a low-priority sequence" {
    const gpa = std.testing.allocator;
    var kv = try pa.PagedKVCache.init(gpa, .{
        .block_size = 4,
        .num_blocks = 4,
        .head_dim = 8,
        .num_kv_heads = 2,
        .num_q_heads = 8,
        .dtype = .f32,
        .enable_prefix_cache = true,
        .max_seq_len = 64,
        .max_batch_size = 4,
    });
    defer kv.deinit();
    var sched = pa.Scheduler.init(gpa, kv.config, &kv);
    defer sched.deinit();

    const tokens_a = &[_]u32{ 1, 2, 3, 4 };
    const tokens_b = &[_]u32{ 5, 6, 7, 8 };
    _ = try sched.submit(.{ .prompt_tokens = tokens_a, .max_new_tokens = 2, .priority = 10 });
    _ = try sched.schedule();
    _ = try sched.submit(.{ .prompt_tokens = tokens_b, .max_new_tokens = 2, .priority = 5 });
    _ = try sched.schedule();

    // decode growth fills the pool -> low-priority seq 2 must be preempted
    _ = try sched.schedule();
    try std.testing.expect(sched.numPreempted() > 0);
    try std.testing.expect(sched.kv_cache.freeBlocks() > 0);

    // finish the high-priority sequence, next schedule restores seq 2
    sched.finishSequence(sched.running.items[0]);
    _ = try sched.schedule();
    try std.testing.expect(sched.numPreempted() == 0);
    try std.testing.expect(sched.numRunning() > 0);
}
