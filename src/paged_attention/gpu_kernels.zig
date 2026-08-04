const std = @import("std");

pub const LaunchConfig = struct {
    num_seqs: usize,
    max_seq_len: usize,
    num_blocks: usize,
    block_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
};

pub const CudaStream = *anyopaque;

pub extern "c" fn paged_attention_decode_f16(
    out: *anyopaque,
    query: *const anyopaque,
    key_cache: *const anyopaque,
    value_cache: *const anyopaque,
    block_tables: *const anyopaque,
    seq_lens: *const anyopaque,
    config: LaunchConfig,
    stream: CudaStream,
) void;

pub extern "c" fn paged_attention_prefill_f16(
    out: *anyopaque,
    query: *const anyopaque,
    key_cache: *const anyopaque,
    value_cache: *const anyopaque,
    block_tables: *const anyopaque,
    seq_lens: *const anyopaque,
    config: LaunchConfig,
    stream: CudaStream,
) void;

pub extern "c" fn reshape_and_block_write_f16(
    key_cache: *anyopaque,
    value_cache: *anyopaque,
    new_keys: *const anyopaque,
    new_values: *const anyopaque,
    block_tables: *const anyopaque,
    seq_lens: *const anyopaque,
    config: LaunchConfig,
    stream: CudaStream,
) void;

pub extern "c" fn block_copy_f16(
    dst_cache: *anyopaque,
    src_cache: *const anyopaque,
    copy_map: *const anyopaque,
    num_copies: usize,
    block_bytes: usize,
    stream: CudaStream,
) void;

pub const CpuReference = struct {
    pub const decode = @import("attention.zig").PagedAttention.decode;
    pub const prefill = @import("attention.zig").PagedAttention.prefill;
};
