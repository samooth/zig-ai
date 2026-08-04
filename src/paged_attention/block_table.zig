const std = @import("std");
const BlockAllocator = @import("allocator.zig").BlockAllocator;

pub const BlockTable = struct {
    allocator: std.mem.Allocator,
    table: std.ArrayList(usize),
    num_tokens: usize = 0,
    block_size: usize,

    const Self = @This();

    pub fn init(gpa: std.mem.Allocator, block_size: usize) Self {
        return .{
            .allocator = gpa,
            .table = std.ArrayList(usize).init(gpa),
            .block_size = block_size,
        };
    }

    pub fn deinit(self: *Self, block_alloc: *BlockAllocator) void {
        for (self.table.items) |phys_id| {
            block_alloc.release(phys_id);
        }
        self.table.deinit();
    }

    pub fn numBlocks(self: *const Self) usize {
        return self.table.items.len;
    }

    pub fn appendToken(self: *Self, block_alloc: *BlockAllocator) !void {
        const token_idx = self.num_tokens;
        const block_idx = token_idx / self.block_size;
        const offset = token_idx % self.block_size;

        if (offset == 0) {
            const phys_id = block_alloc.alloc() orelse return error.OutOfMemory;
            block_alloc.acquire(phys_id);
            try self.table.append(phys_id);
        }
        self.num_tokens += 1;

        const phys_id = self.table.items[block_idx];
        block_alloc.blocks[phys_id].num_tokens = @intCast(offset + 1);
    }

    pub fn appendTokens(self: *Self, block_alloc: *BlockAllocator, count: usize) !void {
        var remaining = count;
        while (remaining > 0) {
            const token_idx = self.num_tokens;
            const block_idx = token_idx / self.block_size;
            const offset = token_idx % self.block_size;

            if (offset == 0) {
                const phys_id = block_alloc.alloc() orelse return error.OutOfMemory;
                block_alloc.acquire(phys_id);
                try self.table.append(phys_id);
            }

            const fill = @min(remaining, self.block_size - offset);
            self.num_tokens += fill;
            remaining -= fill;

            const phys_id = self.table.items[block_idx];
            block_alloc.blocks[phys_id].num_tokens = @intCast(@min(self.num_tokens - block_idx * self.block_size, self.block_size));
        }
    }

    pub fn fork(self: *const Self, block_alloc: *BlockAllocator) !BlockTable {
        var new_table = BlockTable.init(self.allocator, self.block_size);
        try new_table.table.ensureTotalCapacityPrecise(self.table.items.len);
        for (self.table.items) |phys_id| {
            block_alloc.acquire(phys_id);
            new_table.table.appendAssumeCapacity(phys_id);
        }
        new_table.num_tokens = self.num_tokens;
        return new_table;
    }

    pub fn prepareWrite(self: *Self, block_alloc: *BlockAllocator) !void {
        if (self.table.items.len == 0) return;
        const last_idx = self.table.items.len - 1;
        const phys_id = self.table.items[last_idx];
        if (block_alloc.blocks[phys_id].isShared()) {
            const new_id = try block_alloc.copyOnWrite(phys_id);
            self.table.items[last_idx] = new_id;
        }
    }

    pub fn getPhysical(self: *const Self, logical_idx: usize) ?usize {
        if (logical_idx >= self.table.items.len) return null;
        return self.table.items[logical_idx];
    }

    pub fn computePrefixHash(self: *const Self, block_alloc: *const BlockAllocator, num_tokens: usize) ?u64 {
        if (num_tokens == 0) return null;
        const block_idx = (num_tokens - 1) / self.block_size;
        if (block_idx >= self.table.items.len) return null;
        return block_alloc.blocks[self.table.items[block_idx]].block_hash;
    }

    pub fn setBlockHash(self: *Self, block_alloc: *BlockAllocator, logical_idx: usize, hash: u64) void {
        if (logical_idx >= self.table.items.len) return;
        block_alloc.blocks[self.table.items[logical_idx]].block_hash = hash;
    }
};
