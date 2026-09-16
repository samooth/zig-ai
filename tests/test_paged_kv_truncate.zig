//! Tests truncate/rollback de PagedKVCache (lane-c C3).
const std = @import("std");
const pa = @import("paged_attention");

fn testConfig() pa.PagedConfig {
    return .{
        .block_size = 4,
        .num_blocks = 16,
        .head_dim = 8,
        .num_kv_heads = 2,
        .num_q_heads = 8,
        .dtype = .f32,
        .quant_k = .fp32,
        .quant_v = .fp32,
        .enable_prefix_cache = false,
        .max_seq_len = 64,
        .max_batch_size = 4,
    };
}

test "PagedKVCache truncate → append → coherente" {
    const gpa = std.testing.allocator;
    var cache = try pa.PagedKVCache.init(gpa, testConfig());
    defer cache.deinit();

    const seq = try cache.createSequence();
    // 9 tokens con block_size=4 → 3 bloques (4+4+1).
    try cache.allocatePrefill(seq, 9);
    const bt = cache.getBlockTable(seq).?;
    try std.testing.expectEqual(@as(usize, 3), bt.numBlocks());
    try std.testing.expectEqual(@as(usize, 9), bt.num_tokens);
    try std.testing.expectEqual(@as(usize, 13), cache.freeBlocks()); // 16-3

    // Truncar a 5 → quedan 2 bloques; el último conserva 1 token.
    const released_phys = bt.getPhysical(2).?;
    try cache.truncate(seq, 5);
    try std.testing.expectEqual(@as(usize, 2), bt.numBlocks());
    try std.testing.expectEqual(@as(usize, 5), bt.num_tokens);
    const last_phys = bt.getPhysical(1).?;
    try std.testing.expectEqual(@as(u32, 1), cache.block_alloc.blocks[last_phys].num_tokens);
    try std.testing.expectEqual(@as(usize, 14), cache.freeBlocks());

    // Re-append 4 tokens decode → vuelve a 3 bloques / 9 tokens.
    for (0..4) |_| try cache.appendDecode(seq);
    try std.testing.expectEqual(@as(usize, 3), bt.numBlocks());
    try std.testing.expectEqual(@as(usize, 9), bt.num_tokens);
    try std.testing.expectEqual(@as(usize, 13), cache.freeBlocks());

    // El bloque re-allocado es exactamente el que se liberó (freelist LIFO).
    try std.testing.expectEqual(released_phys, bt.getPhysical(2).?);
}

test "truncate a 0 y no-op más allá del largo" {
    const gpa = std.testing.allocator;
    var cache = try pa.PagedKVCache.init(gpa, testConfig());
    defer cache.deinit();
    const seq = try cache.createSequence();
    try cache.allocatePrefill(seq, 6);

    // No-op: new_len >= num_tokens
    try cache.truncate(seq, 6);
    try cache.truncate(seq, 100);
    try std.testing.expectEqual(@as(usize, 6), cache.getBlockTable(seq).?.num_tokens);

    // A 0 libera todo
    try cache.truncate(seq, 0);
    try std.testing.expectEqual(@as(usize, 0), cache.getBlockTable(seq).?.numBlocks());
    try std.testing.expectEqual(@as(usize, 0), cache.getBlockTable(seq).?.num_tokens);
    try std.testing.expectEqual(@as(usize, 16), cache.freeBlocks());

    // Secuencia inexistente → error
    try std.testing.expectError(error.SequenceNotFound, cache.truncate(999, 0));
}

test "truncate con fork: bloque compartido intacto para el padre" {
    const gpa = std.testing.allocator;
    var cache = try pa.PagedKVCache.init(gpa, testConfig());
    defer cache.deinit();

    const parent = try cache.createSequence();
    try cache.allocatePrefill(parent, 8); // 2 bloques llenos
    const child = try cache.forkSequence(parent);
    try std.testing.expectEqual(@as(usize, 2), cache.getBlockTable(child).?.numBlocks());

    // Ambos comparten los bloques (ref_count=2)
    for (0..2) |b| {
        const pp = cache.getBlockTable(parent).?.getPhysical(b).?;
        try std.testing.expectEqual(@as(u32, 2), cache.block_alloc.blocks[pp].ref_count);
    }

    // El hijo trunca a 3 (corte DENTRO del bloque 0 compartido) → COW: el
    // hijo pasa a una copia privada; el padre conserva sus 2 bloques con
    // ref_count=1 cada uno.
    try cache.truncate(child, 3);
    try std.testing.expectEqual(@as(usize, 3), cache.getBlockTable(child).?.num_tokens);
    const child_bt = cache.getBlockTable(child).?;
    const parent_bt = cache.getBlockTable(parent).?;
    // Copia privada: físicos distintos para el bloque 0
    try std.testing.expect(child_bt.getPhysical(0).? != parent_bt.getPhysical(0).?);
    for (0..2) |b| {
        const pp = cache.block_alloc.blocks[parent_bt.getPhysical(b).?];
        try std.testing.expectEqual(@as(u32, 1), pp.ref_count); // sólo padre
    }

    // Append del hijo: cae en su copia privada (offset 3 del bloque 0)
    try cache.appendDecode(child);
    try std.testing.expectEqual(@as(usize, 4), child_bt.num_tokens);
}
