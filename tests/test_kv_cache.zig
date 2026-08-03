const std = @import("std");
const kvc = @import("kv_cache");
const KVCacheManager = kvc.KVCacheManager;
const KVCacheConfig = kvc.KVCacheConfig;
const QuantFormat = kvc.QuantFormat;
test "kv_cache manager init/deinit" {
    const allocator = std.testing.allocator;
    const config = KVCacheConfig.default(4, 32, 8, 128, 1024);
    var mgr = try KVCacheManager.init(allocator, config, 64);
    defer mgr.deinit();
}
test "kv_cache create sequence" {
    const allocator = std.testing.allocator;
    const config = KVCacheConfig.default(2, 16, 4, 64, 512);
    var mgr = try KVCacheManager.init(allocator, config, 32);
    defer mgr.deinit();
    try mgr.createSequence(1);
    try std.testing.expectEqual(@as(u32, 0), try mgr.getSequenceLen(1));
}
test "kv_cache append and retrieve" {
    const allocator = std.testing.allocator;
    const config = KVCacheConfig.default(2, 16, 4, 64, 512);
    var mgr = try KVCacheManager.init(allocator, config, 32);
    defer mgr.deinit();
    try mgr.createSequence(42);
    var k_f16 = try allocator.alloc(f16, 64);
    defer allocator.free(k_f16);
    var v_f16 = try allocator.alloc(f16, 64);
    defer allocator.free(v_f16);
    @memset(k_f16, 0.5);
    @memset(v_f16, 0.3);
    try mgr.appendTokensF16(42, 0, 0, k_f16, v_f16);
    try mgr.advanceSequence(42);
    var out_k = try allocator.alloc(f16, 64);
    defer allocator.free(out_k);
    var out_v = try allocator.alloc(f16, 64);
    defer allocator.free(out_v);
    try mgr.retrieveForAttention(42, 0, 0, out_k, out_v);
}
test "gqa mapping" {
    const config = KVCacheConfig.default(32, 32, 8, 128, 4096);
    try std.testing.expectEqual(@as(u32, 4), config.gqaGroupSize());
    try std.testing.expectEqual(@as(u32, 0), config.qHeadToKvHead(0));
    try std.testing.expectEqual(@as(u32, 0), config.qHeadToKvHead(3));
    try std.testing.expectEqual(@as(u32, 1), config.qHeadToKvHead(4));
    try std.testing.expectEqual(@as(u32, 7), config.qHeadToKvHead(31));
}
test "quant_ops q4_0" {
    const allocator = std.testing.allocator;
    const num_elements: usize = 64;
    var src = try allocator.alloc(f16, num_elements);
    defer allocator.free(src);
    for (0..num_elements) |i| {
        src[i] = @floatCast(@sin(@as(f32, @floatFromInt(i)) * 0.1));
    }
    const dst_size = kvc.quantizedSize(.q4_0, num_elements, 32);
    var dst = try allocator.alloc(u8, dst_size);
    defer allocator.free(dst);
    const written = try kvc.quantizeQ4_0(src, dst, 32);
    try std.testing.expectEqual(dst_size, written);
}
test "quant_ops q8_0" {
    const allocator = std.testing.allocator;
    const num_elements: usize = 64;
    var src = try allocator.alloc(f16, num_elements);
    defer allocator.free(src);
    for (0..num_elements) |i| {
        src[i] = @floatCast(@sin(@as(f32, @floatFromInt(i)) * 0.1));
    }
    const dst_size = kvc.quantizedSize(.q8_0, num_elements, 32);
    var dst = try allocator.alloc(u8, dst_size);
    defer allocator.free(dst);
    const written = try kvc.quantizeQ8_0(src, dst, 32);
    try std.testing.expectEqual(dst_size, written);
}
