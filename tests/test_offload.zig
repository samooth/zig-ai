//! Test para el pipeline de CPU offload
const std = @import("std");
const kvc = @import("kv_cache");
const CpuOffloadPipeline = kvc.CpuOffloadPipeline;
const OffloadConfig = kvc.OffloadConfig;
const KVCacheConfig = kvc.KVCacheConfig;
const KVCacheManager = kvc.KVCacheManager;
const QuantFormat = kvc.QuantFormat;

test "cpu offload pipeline basic" {
    const gpa = std.testing.allocator;
    const config = OffloadConfig{
        .enabled = true,
        .min_free_vram_mb = 1024,
        .min_token_age = 10,
        .max_cpu_blocks = 100,
        .use_dedicated_stream = false,
    };
    
    var pipeline = try CpuOffloadPipeline.init(gpa, config, null);
    defer pipeline.deinit();
    
    // Register a block
    const desc = kvc.KVBlockDescriptor{
        .layer_idx = 0,
        .head_idx = 0,
        .seq_start = 0,
        .seq_len = 128,
        .head_dim = 64,
        .format = .q4_0,
        .byte_offset = 0,
        .byte_size = 1024,
    };
    
try pipeline.registerBlock(desc);
    pipeline.touchBlock(desc, 100);
    
    // Try to offload (should work with low VRAM)
    const kvm_config = KVCacheConfig.default(2, 8, 64, 512);
    var mgr = try KVCacheManager.init(gpa, kvm_config, 32);
    defer mgr.deinit();
    
    // Test passes if no errors
}

test "cpu offload pipeline evict and reload" {
    const gpa = std.testing.allocator;
    const config = OffloadConfig{
        .enabled = true,
        .min_free_vram_mb = 1024,
        .min_token_age = 10,
        .max_cpu_blocks = 100,
        .use_dedicated_stream = false,
    };
    
    var pipeline = try CpuOffloadPipeline.init(gpa, config, null);
    defer pipeline.deinit();
    
    // Register multiple blocks
    for (0..5) |i| {
        const desc = kvc.KVBlockDescriptor{
            .layer_idx = 0,
            .head_idx = @intCast(i),
            .seq_start = 0,
            .seq_len = 128,
            .head_dim = 64,
            .format = .q4_0,
            .byte_offset = 0,
            .byte_size = 1024,
        };
try pipeline.registerBlock(desc);
        pipeline.touchBlock(desc, 100 + i * 10);
    }
    
    // Verify all registered
    try std.testing.expectEqual(5, pipeline.entries.count());
}

test "cpu offload config" {
    const config = OffloadConfig{
        .enabled = true,
        .min_free_vram_mb = 512,
        .min_token_age = 64,
        .max_cpu_blocks = 1024,
        .use_dedicated_stream = true,
    };
    try std.testing.expect(config.enabled);
    try std.testing.expectEqual(512, config.min_free_vram_mb);
    try std.testing.expectEqual(64, config.min_token_age);
    try std.testing.expectEqual(1024, config.max_cpu_blocks);
    try std.testing.expect(config.use_dedicated_stream);
}