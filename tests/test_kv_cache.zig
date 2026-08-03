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

test "gqa mapping" {
    var config = KVCacheConfig.default(32, 32, 128, 4096);
    config.num_kv_heads = 8;
    try std.testing.expectEqual(@as(usize, 0), config.qHeadToKvHead(0));
    try std.testing.expectEqual(@as(usize, 0), config.qHeadToKvHead(3));
    try std.testing.expectEqual(@as(usize, 1), config.qHeadToKvHead(4));
    try std.testing.expectEqual(@as(usize, 7), config.qHeadToKvHead(31));
}
