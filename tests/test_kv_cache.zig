const std = @import("std");
const kvc = @import("kv_cache");
const KVCacheManager = kvc.KVCacheManager;
const KVCacheConfig = kvc.KVCacheConfig;
const QuantFormat = kvc.QuantFormat;

test "kv_cache manager init/deinit" {
    const allocator = std.testing.allocator;
    const config = KVCacheConfig.default(4, 32, 128, 1024);
    var mgr = try KVCacheManager.init(allocator, config, 64);
    defer mgr.deinit();
}

test "kv_cache create sequence" {
    const allocator = std.testing.allocator;
    const config = KVCacheConfig.default(2, 16, 64, 512);
    var mgr = try KVCacheManager.init(allocator, config, 32);
    defer mgr.deinit();
    try mgr.createSequence(1);
    try std.testing.expectEqual(@as(usize, 0), try mgr.getSequenceLen(1));
    try mgr.advanceSequence(1);
    try std.testing.expectEqual(@as(usize, 1), try mgr.getSequenceLen(1));
}

test "kv_cache append and retrieve" {
    const allocator = std.testing.allocator;
    var config = KVCacheConfig.default(2, 16, 64, 512);
    const layer_cfgs = [_]kvc.LayerQuantConfig{
        .{ .k_format = .fp16, .v_format = .fp16, .k_block_size = 32, .v_block_size = 32, .quant_threshold = null },
        .{ .k_format = .fp16, .v_format = .fp16, .k_block_size = 32, .v_block_size = 32, .quant_threshold = null },
    };
    config.layer_configs = &layer_cfgs;
    config.use_gpu_dequant = false;
    var mgr = try KVCacheManager.init(allocator, config, 32);
    defer mgr.deinit();
    try mgr.createSequence(42);
    const k_f16 = try allocator.alloc(f16, 64);
    defer allocator.free(k_f16);
    const v_f16 = try allocator.alloc(f16, 64);
    defer allocator.free(v_f16);
    @memset(k_f16, 0.5);
    @memset(v_f16, 0.3);
    try mgr.appendTokensF16(42, 0, 0, k_f16, v_f16);
    try mgr.advanceSequence(42);
    const out_k = try allocator.alloc(f16, 64);
    defer allocator.free(out_k);
    const out_v = try allocator.alloc(f16, 64);
    defer allocator.free(out_v);
    try mgr.retrieveForAttention(42, 0, 0, out_k, out_v);
}

test "kv_cache append and retrieve q8_0" {
    const allocator = std.testing.allocator;
    var config = KVCacheConfig.default(2, 16, 64, 512);
    const layer_cfgs = [_]kvc.LayerQuantConfig{
        .{ .k_format = .q8_0, .v_format = .q8_0, .k_block_size = 32, .v_block_size = 32, .quant_threshold = null },
        .{ .k_format = .q8_0, .v_format = .q8_0, .k_block_size = 32, .v_block_size = 32, .quant_threshold = null },
    };
    config.layer_configs = &layer_cfgs;
    config.use_gpu_dequant = false;
    var mgr = try KVCacheManager.init(allocator, config, 32);
    defer mgr.deinit();
    try mgr.createSequence(7);
    const k_f16 = try allocator.alloc(f16, 64);
    defer allocator.free(k_f16);
    const v_f16 = try allocator.alloc(f16, 64);
    defer allocator.free(v_f16);
    for (0..64) |i| {
        k_f16[i] = @as(f16, @floatCast(@sin(@as(f32, @floatFromInt(i)) * 0.1) * 3.5));
        v_f16[i] = @as(f16, @floatCast(@cos(@as(f32, @floatFromInt(i)) * 0.13) * 2.0));
    }
    try mgr.appendTokensF16(7, 0, 0, k_f16, v_f16);
    try mgr.advanceSequence(7);
    const out_k = try allocator.alloc(f16, 64);
    defer allocator.free(out_k);
    const out_v = try allocator.alloc(f16, 64);
    defer allocator.free(out_v);
    try mgr.retrieveForAttention(7, 0, 0, out_k, out_v);
    var max_err: f32 = 0;
    for (0..64) |i| {
        max_err = @max(max_err, @abs(@as(f32, @floatCast(k_f16[i])) - @as(f32, @floatCast(out_k[i]))));
        max_err = @max(max_err, @abs(@as(f32, @floatCast(v_f16[i])) - @as(f32, @floatCast(out_v[i]))));
    }
    // q8_0 dequant error <= d/2 where d = maxAbs/127 over each vector; K max≈3.5
    // hence d/2≈0.014. Assert a tight but safe bound that still verifies correctness.
    try std.testing.expect(max_err < 0.05);
}

test "gqa mapping" {
    var config = KVCacheConfig.default(32, 32, 128, 4096);
    config.num_kv_heads = 8;
    try std.testing.expectEqual(@as(usize, 0), config.qHeadToKvHead(0));
    try std.testing.expectEqual(@as(usize, 0), config.qHeadToKvHead(3));
    try std.testing.expectEqual(@as(usize, 1), config.qHeadToKvHead(4));
    try std.testing.expectEqual(@as(usize, 7), config.qHeadToKvHead(31));
}
