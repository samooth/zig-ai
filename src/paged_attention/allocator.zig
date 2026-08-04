const std = @import("std");
const Block = @import("block.zig").Block;
const DType = @import("root.zig").DType;

pub const BlockAllocator = struct {
    allocator: std.mem.Allocator,
    blocks: []Block,
    free_list: std.ArrayList(usize),
    block_size: usize,
    bytes_per_elem: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_bytes: usize,
    memory_pool: []u8,
    cpu_pool: ?[]u8 = null,

    const Self = @This();

    pub fn init(
        gpa: std.mem.Allocator,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        enable_cpu_offload: bool,
    ) !Self {
        const bytes_per_elem: usize = switch (dtype) {
            .f32 => 4,
            .f16, .bf16 => 2,
        };
        const block_bytes = block_size * num_kv_heads * head_dim * 2 * bytes_per_elem;

        var blocks = try gpa.alloc(Block, num_blocks);
        var free_list = try std.ArrayList(usize).initCapacity(gpa, num_blocks);
        const memory_pool = try gpa.alloc(u8, num_blocks * block_bytes);
        @memset(memory_pool, 0);

        var cpu_pool: ?[]u8 = null;
        if (enable_cpu_offload) {
            cpu_pool = try gpa.alloc(u8, num_blocks * block_bytes);
            @memset(cpu_pool.?, 0);
        }

        for (0..num_blocks) |i| {
            blocks[i] = Block.init(i);
            blocks[i].data = memory_pool.ptr + i * block_bytes;
            try free_list.append(i);
        }

        return .{
            .allocator = gpa,
            .blocks = blocks,
            .free_list = free_list,
            .block_size = block_size,
            .bytes_per_elem = bytes_per_elem,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .block_bytes = block_bytes,
            .memory_pool = memory_pool,
            .cpu_pool = cpu_pool,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.blocks);
        self.allocator.free(self.memory_pool);
        if (self.cpu_pool) |cpu| self.allocator.free(cpu);
        self.free_list.deinit();
    }

    pub fn alloc(self: *Self) ?usize {
        if (self.free_list.items.len == 0) return null;
        return self.free_list.pop();
    }

    pub fn free(self: *Self, block_id: usize) void {
        std.debug.assert(block_id < self.blocks.len);
        var block = &self.blocks[block_id];
        std.debug.assert(block.ref_count == 0);
        block.num_tokens = 0;
        block.block_hash = null;
        block.is_cpu = false;
        @memset(self.memory_pool[block_id * self.block_bytes ..][0..self.block_bytes], 0);
        self.free_list.append(block_id) catch unreachable;
    }

    pub fn acquire(self: *Self, block_id: usize) void {
        self.blocks[block_id].acquire();
    }

    pub fn release(self: *Self, block_id: usize) void {
        self.blocks[block_id].release();
        if (self.blocks[block_id].isFree()) {
            self.free(block_id);
        }
    }

    pub fn copyOnWrite(self: *Self, block_id: usize) !usize {
        const block = &self.blocks[block_id];
        if (!block.isShared()) return block_id;

        const new_id = self.alloc() orelse return error.OutOfMemory;
        const new_block = &self.blocks[new_id];
        new_block.acquire();

        const src = self.memory_pool[block_id * self.block_bytes ..][0..self.block_bytes];
        const dst = self.memory_pool[new_id * self.block_bytes ..][0..self.block_bytes];
        @memcpy(dst, src);

        new_block.num_tokens = block.num_tokens;
        new_block.block_hash = block.block_hash;

        block.release();
        if (block.isFree()) self.free(block_id);

        return new_id;
    }

    pub fn swapToCpu(self: *Self, block_id: usize) !void {
        if (self.cpu_pool == null) return error.CpuOffloadDisabled;
        const block = &self.blocks[block_id];
        if (block.is_cpu) return;
        const src = self.memory_pool[block_id * self.block_bytes ..][0..self.block_bytes];
        const dst = self.cpu_pool.?[block_id * self.block_bytes ..][0..self.block_bytes];
        @memcpy(dst, src);
        block.is_cpu = true;
    }

    pub fn swapFromCpu(self: *Self, block_id: usize) !void {
        if (self.cpu_pool == null) return error.CpuOffloadDisabled;
        const block = &self.blocks[block_id];
        if (!block.is_cpu) return;
        const src = self.cpu_pool.?[block_id * self.block_bytes ..][0..self.block_bytes];
        const dst = self.memory_pool[block_id * self.block_bytes ..][0..self.block_bytes];
        @memcpy(dst, src);
        block.is_cpu = false;
    }

    pub fn numFree(self: *const Self) usize {
        return self.free_list.items.len;
    }

    pub fn numTotal(self: *const Self) usize {
        return self.blocks.len;
    }
};
