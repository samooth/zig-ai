const std = @import("std");
const Tensor = @import("core").Tensor;
const fa = @import("fa");
const transformer = @import("transformer");
const cudaz = @import("cudaz");

const TransformerLayer = transformer.TransformerLayer;
const LayerPrecision = transformer.LayerPrecision;
const FlashAttentionConfig = fa.fa_config.FlashAttentionConfig;

test "transformer layer init/deinit" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const config = FlashAttentionConfig{
        .N = 32, .d = 64, .num_heads = 4, .batch_size = 1,
        .dtype = .f16, .causal = true,
    };
    const precision = LayerPrecision{
        .compute = .f16,
        .weights_on_gpu = false,
        .use_quantized = false,
    };

    var layer = try TransformerLayer.init(allocator, 0, config, "cuda/flash_attention.ptx", 256, precision, config.num_heads, 1024);
    defer layer.deinit();

    try std.testing.expectEqual(@as(usize, 0), layer.layer_idx);
    try std.testing.expectEqual(@as(usize, 256), layer.hidden_dim);
    try std.testing.expectEqual(@as(usize, 64), layer.head_dim);
    try std.testing.expectEqual(@as(usize, 4), layer.num_heads);
}

test "kv cache init and append" {
    const allocator = std.testing.allocator;
    const config = FlashAttentionConfig{
        .N = 16, .d = 64, .num_heads = 2, .batch_size = 1,
        .dtype = .f16, .causal = true,
    };

    var cache = try transformer.KVCache.init(allocator, config);
    defer cache.deinit();

    try std.testing.expectEqual(@as(usize, 0), cache.current_len);

    var k_new = try Tensor(f16).alloc(allocator, &[_]usize{ 1, 2, 4, 64 });
    defer k_new.deinit();
    var v_new = try Tensor(f16).alloc(allocator, &[_]usize{ 1, 2, 4, 64 });
    defer v_new.deinit();

    try cache.append(&k_new, &v_new);
    try std.testing.expectEqual(@as(usize, 4), cache.current_len);

    try cache.append(&k_new, &v_new);
    try std.testing.expectEqual(@as(usize, 8), cache.current_len);

    cache.clear();
    try std.testing.expectEqual(@as(usize, 0), cache.current_len);
}

test "kv cache overflow" {
    const allocator = std.testing.allocator;
    const config = FlashAttentionConfig{
        .N = 8, .d = 64, .num_heads = 2, .batch_size = 1,
        .dtype = .f16, .causal = true,
    };

    var cache = try transformer.KVCache.init(allocator, config);
    defer cache.deinit();

    var k_new = try Tensor(f16).alloc(allocator, &[_]usize{ 1, 2, 8, 64 });
    defer k_new.deinit();
    var v_new = try Tensor(f16).alloc(allocator, &[_]usize{ 1, 2, 8, 64 });
    defer v_new.deinit();

    try cache.append(&k_new, &v_new);
    // Segundo append deberia fallar por overflow
    try std.testing.expectError(transformer.TransformerError.CacheOverflow, cache.append(&k_new, &v_new));
}
