const std = @import("std");

pub const Block = @import("block.zig").Block;
pub const hashTokens = @import("block.zig").hashTokens;
pub const BlockAllocator = @import("allocator.zig").BlockAllocator;
pub const BlockTable = @import("block_table.zig").BlockTable;
pub const PagedKVCache = @import("paged_kv_cache.zig").PagedKVCache;
pub const PagedAttention = @import("attention.zig").PagedAttention;
pub const Scheduler = @import("scheduler.zig").Scheduler;
pub const PrefixCache = @import("prefix_cache.zig").PrefixCache;
pub const Sequence = @import("scheduler.zig").Sequence;
pub const Request = @import("scheduler.zig").Request;
pub const PagedAttentionGpu = @import("gpu_kernels.zig").PagedAttentionGpu;

pub const DType = enum { f32, f16, bf16 };

pub const PagedConfig = struct {
    block_size: usize = 16,
    num_blocks: usize = 1024,
    head_dim: usize = 128,
    num_kv_heads: usize = 8,
    num_q_heads: usize = 32,
    dtype: DType = .f16,
    enable_prefix_cache: bool = true,
    enable_cpu_offload: bool = false,
    max_seq_len: usize = 8192,
    max_batch_size: usize = 64,
};

pub const Stats = struct {
    blocks_allocated: usize = 0,
    blocks_free: usize = 0,
    blocks_shared: usize = 0,
    sequences_active: usize = 0,
    prefix_hits: usize = 0,
    prefix_misses: usize = 0,
    prefix_evictions: usize = 0,
    prefix_hit_rate: f64 = 0.0,
};
