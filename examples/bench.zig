//! Benchmark de KV-cache cuantizado
//! Compara latencia y throughput de diferentes formatos

const std = @import("std");
const kvc = @import("kv_cache");

const KVCacheManager = kvc.KVCacheManager;
const KVCacheConfig = kvc.KVCacheConfig;
const QuantFormat = kvc.QuantFormat;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("=== KV-Cache GPU Benchmark ===", .{});

    const configs = &[_]struct {
        name: []const u8,
        num_layers: u32,
        num_heads: u32,
        head_dim: u32,
        max_seq_len: u32,
        pool_mb: usize,
    }{
        .{ .name = "Llama-7B", .num_layers = 32, .num_heads = 32, .head_dim = 128, .max_seq_len = 4096, .pool_mb = 512 },
        .{ .name = "Llama-70B", .num_layers = 80, .num_heads = 64, .head_dim = 128, .max_seq_len = 8192, .pool_mb = 2048 },
    };

    for (configs) |cfg| {
        std.log.info("\n--- Modelo: {s} ---", .{cfg.name});

        const cache_config = KVCacheConfig.default(
            cfg.num_layers,
            cfg.num_heads,
            cfg.head_dim,
            cfg.max_seq_len,
        );

        var manager = try KVCacheManager.init(allocator, cache_config, cfg.pool_mb);
        defer manager.deinit();

        try manager.createSequence(1);

        const formats = &[_]QuantFormat{ .fp16, .q8_0, .q4_0, .int8_symmetric };
        for (formats) |fmt| {
            const bytes_per_token = cfg.num_layers * cfg.num_heads * cfg.head_dim * fmt.bitsPerElement() / 8;
            const total_mb = @as(f64, @floatFromInt(bytes_per_token * cfg.max_seq_len)) / (1024.0 * 1024.0);
            std.log.info("  {s}: {d:.1} MB total, {d} bytes/token", .{
                @tagName(fmt), total_mb, bytes_per_token,
            });
        }

        manager.reportMetrics();
    }

    std.log.info("\nBenchmark completado.", .{});
}
