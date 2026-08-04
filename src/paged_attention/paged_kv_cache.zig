const std = @import("std");
const BlockAllocator = @import("allocator.zig").BlockAllocator;
const BlockTable = @import("block_table.zig").BlockTable;
const PrefixCache = @import("prefix_cache.zig").PrefixCache;
const PagedConfig = @import("root.zig").PagedConfig;

pub const PagedKVCache = struct {
    allocator: std.mem.Allocator,
    config: PagedConfig,
    block_alloc: BlockAllocator,
    prefix_cache: PrefixCache,
    sequences: std.AutoHashMap(u64, BlockTable),
    next_seq_id: u64 = 1,

    const Self = @This();

    pub fn init(gpa: std.mem.Allocator, config: PagedConfig) !Self {
        const block_alloc = try BlockAllocator.init(
            gpa,
            config.num_blocks,
            config.block_size,
            config.num_kv_heads,
            config.head_dim,
            config.dtype,
            config.enable_cpu_offload,
        );
        const prefix_cache = try PrefixCache.init(gpa, 4096);
        return .{
            .allocator = gpa,
            .config = config,
            .block_alloc = block_alloc,
            .prefix_cache = prefix_cache,
            .sequences = std.AutoHashMap(u64, BlockTable).init(gpa),
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.sequences.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.deinit(&self.block_alloc);
        }
        self.sequences.deinit();
        self.prefix_cache.deinit();
        self.block_alloc.deinit();
    }

    pub fn createSequence(self: *Self) !u64 {
        const seq_id = self.next_seq_id;
        self.next_seq_id += 1;
        const bt = BlockTable.init(self.allocator, self.config.block_size);
        try self.sequences.put(seq_id, bt);
        return seq_id;
    }

    pub fn forkSequence(self: *Self, parent_seq_id: u64) !u64 {
        const parent = self.sequences.getPtr(parent_seq_id) orelse return error.SequenceNotFound;
        const seq_id = self.next_seq_id;
        self.next_seq_id += 1;
        const child = try parent.fork(&self.block_alloc);
        try self.sequences.put(seq_id, child);
        return seq_id;
    }

    pub fn removeSequence(self: *Self, seq_id: u64) void {
        var bt = self.sequences.fetchRemove(seq_id) orelse return;
        bt.value.deinit(&self.block_alloc);
    }

    pub fn allocatePrefill(self: *Self, seq_id: u64, num_tokens: usize) !void {
        var bt = self.sequences.getPtr(seq_id) orelse return error.SequenceNotFound;
        try bt.appendTokens(&self.block_alloc, num_tokens);
    }

    pub fn appendDecode(self: *Self, seq_id: u64) !void {
        var bt = self.sequences.getPtr(seq_id) orelse return error.SequenceNotFound;
        try bt.appendToken(&self.block_alloc);
    }

    pub fn prepareWrite(self: *Self, seq_id: u64) !void {
        var bt = self.sequences.getPtr(seq_id) orelse return error.SequenceNotFound;
        try bt.prepareWrite(&self.block_alloc);
    }

    pub fn getBlockTable(self: *const Self, seq_id: u64) ?*const BlockTable {
        return self.sequences.getPtr(seq_id);
    }

    pub fn getBlockTableMut(self: *Self, seq_id: u64) ?*BlockTable {
        return self.sequences.getPtr(seq_id);
    }

    pub fn canAllocate(self: *const Self, num_blocks: usize) bool {
        return self.block_alloc.numFree() >= num_blocks;
    }

    pub fn freeBlocks(self: *const Self) usize {
        return self.block_alloc.numFree();
    }

    pub fn matchPrefix(self: *Self, tokens: []const u32) !usize {
        if (!self.config.enable_prefix_cache or tokens.len == 0) return 0;
        return self.prefix_cache.longestPrefixMatch(tokens);
    }

    pub fn cachePrefix(self: *Self, seq_id: u64, tokens: []const u32) !void {
        if (!self.config.enable_prefix_cache or tokens.len < self.config.block_size) return;
        const bt = self.sequences.getPtr(seq_id) orelse return;
        var token_idx: usize = 0;
        while (token_idx + self.config.block_size <= tokens.len) : (token_idx += self.config.block_size) {
            const block_idx = token_idx / self.config.block_size;
            if (block_idx >= bt.numBlocks()) break;
            const phys_id = bt.getPhysical(block_idx) orelse break;
            const hash = @import("block.zig").hashTokens(tokens[token_idx..][0..self.config.block_size]);
            self.block_alloc.blocks[phys_id].block_hash = hash;
            try self.prefix_cache.insert(hash, phys_id, self.config.block_size);
        }
    }

    pub fn getBlockData(self: *Self, phys_id: usize) [*]u8 {
        return self.block_alloc.blocks[phys_id].data.?;
    }

    pub fn getBlock(self: *Self, phys_id: usize) *@import("block.zig").Block {
        return &self.block_alloc.blocks[phys_id];
    }
};
