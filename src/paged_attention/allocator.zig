const std = @import("std");
const Block = @import("block.zig").Block;
const DType = @import("root.zig").DType;
const PagedConfig = @import("root.zig").PagedConfig;
const QuantFormat = @import("root.zig").QuantFormat;
const kv_quant = @import("root.zig").kv_quant;
const debugz = @import("debug");

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
    /// Quantized KV cache support
    quant_k: QuantFormat = .fp16,
    quant_v: QuantFormat = .fp16,
    k_scales: []const f16,
    v_scales: []const f16,
    k_scale_block_stride: usize,
    v_scale_block_stride: usize,

    const Self = @This();

    pub fn init(
        gpa: std.mem.Allocator,
        config: PagedConfig,
    ) !Self {
        const bytes_per_elem: usize = switch (config.dtype) {
            .f32 => 4,
            .f16, .bf16 => 2,
        };
        // 7.3-REVERT: stride RAW del bloque físico (el pad 32B de @a8e9dcd
        // rompía los kernels prefill/decode cuantizados — offsets raw GGUF)
        const block_bytes = config.totalBlockBytes();

        var blocks = try gpa.alloc(Block, config.num_blocks);
        var free_list = try std.ArrayList(usize).initCapacity(gpa, config.num_blocks);
        const memory_pool = try gpa.alloc(u8, config.num_blocks * block_bytes);
        @memset(memory_pool, 0);

        var cpu_pool: ?[]u8 = null;
        if (config.enable_cpu_offload) {
            cpu_pool = try gpa.alloc(u8, config.num_blocks * block_bytes);
            @memset(cpu_pool.?, 0);
        }

        // Scale arrays for quantized KV
        const k_scale_elems = config.num_blocks * config.quantBlocksPerBlock();
        const v_scale_elems = k_scale_elems;
        const k_scales = if (config.quant_k.hasScales()) try gpa.alloc(f16, k_scale_elems) else &.{};
        const v_scales = if (config.quant_v.hasScales()) try gpa.alloc(f16, v_scale_elems) else &.{};

        for (0..config.num_blocks) |i| {
            blocks[i] = Block.init(i);
            blocks[i].data = memory_pool.ptr + i * block_bytes;
            try free_list.append(gpa, i);
        }

        return .{
            .allocator = gpa,
            .blocks = blocks,
            .free_list = free_list,
            .block_size = config.block_size,
            .bytes_per_elem = bytes_per_elem,
            .num_kv_heads = config.num_kv_heads,
            .head_dim = config.head_dim,
            .block_bytes = block_bytes,
            .memory_pool = memory_pool,
            .cpu_pool = cpu_pool,
            .quant_k = config.quant_k,
            .quant_v = config.quant_v,
            .k_scales = k_scales,
            .v_scales = v_scales,
            .k_scale_block_stride = config.quantBlocksPerBlock(),
            .v_scale_block_stride = config.quantBlocksPerBlock(),
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.blocks);
        self.allocator.free(self.memory_pool);
        if (self.cpu_pool) |cpu| self.allocator.free(cpu);
        if (self.quant_k.hasScales()) self.allocator.free(self.k_scales);
        if (self.quant_v.hasScales()) self.allocator.free(self.v_scales);
        self.free_list.deinit(self.allocator);
    }

    pub fn alloc(self: *Self) !?usize {
        if (self.free_list.items.len == 0) return null;
        const id = self.free_list.pop() orelse return null;
        if (self.blocks[id].is_cpu) {
            try self.swapFromCpu(id);
        }
        self.blocks[id].last_access = timestampNow();
        return id;
    }

    pub fn free(self: *Self, block_id: usize) void {
        std.debug.assert(block_id < self.blocks.len);
        var block = &self.blocks[block_id];
        std.debug.assert(block.ref_count == 0);
        block.num_tokens = 0;
        block.block_hash = null;
        block.is_cpu = false;
        @memset(self.memory_pool[block_id * self.block_bytes ..][0..self.block_bytes], 0);
        self.free_list.append(self.allocator, block_id) catch unreachable;
    }

    pub fn acquire(self: *Self, block_id: usize) void {
        self.blocks[block_id].acquire();
        self.blocks[block_id].last_access = timestampNow();
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

        const new_id = try self.alloc() orelse return error.OutOfMemory;
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
        @memset(self.memory_pool[block_id * self.block_bytes ..][0..self.block_bytes], 0);
        block.is_cpu = true;
        debugz.dbg.printLevel(.detail, "[kv_offload] swapToCpu block={d} bytes={d}\n", .{ block_id, self.block_bytes });
    }

    pub fn swapFromCpu(self: *Self, block_id: usize) !void {
        if (self.cpu_pool == null) return error.CpuOffloadDisabled;
        const block = &self.blocks[block_id];
        if (!block.is_cpu) return;
        const src = self.cpu_pool.?[block_id * self.block_bytes ..][0..self.block_bytes];
        const dst = self.memory_pool[block_id * self.block_bytes ..][0..self.block_bytes];
        @memcpy(dst, src);
        block.is_cpu = false;
        debugz.dbg.printLevel(.detail, "[kv_offload] swapFromCpu block={d} bytes={d}\n", .{ block_id, self.block_bytes });
    }

    /// Update LRU timestamp for a block (call on every access).
    pub fn touchBlock(self: *Self, block_id: usize, timestamp: u64) void {
        self.blocks[block_id].last_access = timestamp;
    }

    /// Spill cold blocks to CPU until at least `target_free` blocks are free
    /// in the GPU pool, or no more spillable blocks exist.
    /// Returns number of blocks spilled.
    pub fn maybeSpillToCpu(self: *Self, target_free: usize) !usize {
        if (self.cpu_pool == null) return 0;
        if (self.numFree() >= target_free) return 0;

        var spilled: usize = 0;
        // Scan all blocks, spill the coldest ones first (simple sort-free scan).
        // We only spill blocks that are resident in GPU (not already in CPU)
        // and have ref_count == 0 (not in use by active sequences).
        const needed = target_free - self.numFree();
        var candidates: [1024]usize = undefined;
        var n_candidates: usize = 0;

        for (self.blocks) |*block| {
            if (n_candidates >= candidates.len) break;
            if (block.is_cpu) continue;
            if (block.ref_count > 0) continue;
            if (block.block_hash == null) continue; // free blocks: skip, already reusable
            candidates[n_candidates] = block.id;
            n_candidates += 1;
        }

        // Sort candidates by last_access ascending (coldest first)
        // Simple selection sort for small arrays (no allocs).
        for (0..n_candidates) |i| {
            var min_idx = i;
            for (i + 1..n_candidates) |j| {
                if (self.blocks[candidates[j]].last_access < self.blocks[candidates[min_idx]].last_access) {
                    min_idx = j;
                }
            }
            if (min_idx != i) {
                const tmp = candidates[i];
                candidates[i] = candidates[min_idx];
                candidates[min_idx] = tmp;
            }
        }

        for (0..@min(needed, n_candidates)) |i| {
            const block_id = candidates[i];
            try self.swapToCpu(block_id);
            spilled += 1;
        }

        if (spilled == 0 and needed > 0 and debugz.dbg.at(.detail)) {
            debugz.dbg.printLevel(.detail, "[kv_offload] no spillable blocks (needed={d}, candidates={d})\n", .{ needed, n_candidates });
        }

        return spilled;
    }

    /// Ensure a block is resident in GPU memory. If it's in CPU, swap it back.
    /// Returns error if CPU offload is disabled or swap fails.
    pub fn ensureBlockInGpu(self: *Self, block_id: usize) !void {
        if (block_id >= self.blocks.len) return;
        const block = &self.blocks[block_id];
        if (block.is_cpu) {
            try self.swapFromCpu(block_id);
            debugz.dbg.printLevel(.detail, "[kv_offload] ensureBlockInGpu reloaded block={d}\n", .{block_id});
        }
    }

    pub fn numFree(self: *const Self) usize {
        return self.free_list.items.len;
    }

    pub fn numTotal(self: *const Self) usize {
        return self.blocks.len;
    }

    /// Get the K and V data pointers for a block (quantized layout)
    pub fn blockPtrs(self: *Self, block_id: usize) struct { k: []u8, v: []u8 } {
        const base = block_id * self.block_bytes;
        const k_bytes = kv_quant.quantBytes(self.quant_k, self.block_size * self.num_kv_heads * self.head_dim);
        return .{
            .k = self.memory_pool[base..][0..k_bytes],
            .v = self.memory_pool[base + k_bytes ..][0 .. self.block_bytes - k_bytes],
        };
    }

    /// Get scale pointers for a block
    pub fn blockScalePtrs(self: *Self, block_id: usize) struct { k: ?[]f16, v: ?[]f16 } {
        const k_offset = block_id * self.k_scale_block_stride;
        const v_offset = block_id * self.v_scale_block_stride;
        return .{
            .k = if (self.quant_k.hasScales()) self.k_scales[k_offset..][0..self.k_scale_block_stride] else null,
            .v = if (self.quant_v.hasScales()) self.v_scales[v_offset..][0..self.v_scale_block_stride] else null,
        };
    }

    /// Get K data for a specific head within a block (for CPU access)
    pub fn getKData(self: *Self, block_id: usize, head: usize) []f16 {
        if (self.quant_k != .fp16) return &.{};
        const ptrs = self.blockPtrs(block_id);
        const elems_per_head = self.block_size * self.head_dim;
        const head_offset = head * elems_per_head * 2;
        return std.mem.sliceAsBytes(ptrs.k[head_offset..][0 .. elems_per_head * 2]);
    }

    /// Get V data for a specific head within a block
    pub fn getVData(self: *Self, block_id: usize, head: usize) []f16 {
        if (self.quant_v != .fp16) return &.{};
        const ptrs = self.blockPtrs(block_id);
        const elems_per_head = self.block_size * self.head_dim;
        const k_bytes = kv_quant.quantBytes(self.quant_k, self.block_size * self.num_kv_heads * self.head_dim);
        const v_head_offset = head * elems_per_head * 2;
        // V data starts after K data
        return std.mem.sliceAsBytes(ptrs.v[k_bytes + v_head_offset ..][0 .. elems_per_head * 2]);
    }
};

fn timestampNow() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1000000000 + @as(u64, @intCast(ts.nsec));
}

/// Simple CPU offload manager for KV blocks.
/// Wraps BlockAllocator's native CPU swap capabilities.
pub const CpuOffloadManager = struct {
    block_alloc: *BlockAllocator,
    enabled: bool,
    check_interval: u32,
    tokens_since_check: u32 = 0,
    min_free_blocks: usize,
    spilled_total: usize = 0,
    reloaded_total: usize = 0,

    const Self = @This();

    pub fn init(block_alloc: *BlockAllocator, check_interval: u32, min_free_blocks: usize) Self {
        return .{
            .block_alloc = block_alloc,
            .enabled = block_alloc.cpu_pool != null,
            .check_interval = check_interval,
            .min_free_blocks = min_free_blocks,
        };
    }

    /// Call after each token is generated. Only does work every `check_interval` tokens.
    pub fn maybeSpill(self: *Self) !void {
        if (!self.enabled) return;
        self.tokens_since_check += 1;
        if (self.tokens_since_check < self.check_interval) return;
        self.tokens_since_check = 0;

        const free_before = self.block_alloc.numFree();
        if (free_before >= self.min_free_blocks) return;

        const spilled = try self.block_alloc.maybeSpillToCpu(self.min_free_blocks);
        self.spilled_total += spilled;
        if (spilled > 0 and debugz.dbg.at(.detail)) {
            debugz.dbg.printLevel(.detail, "[kv_offload] spilled {d} blocks to CPU (free={d}→{d})\n", .{
                spilled, free_before, self.block_alloc.numFree(),
            });
        }
    }

    /// Ensure a block is resident in GPU. Call before using a block that might be in CPU.
    pub fn ensureInGpu(self: *Self, block_id: usize) !void {
        if (!self.enabled) return;
        if (block_id >= self.block_alloc.blocks.len) return;
        if (self.block_alloc.blocks[block_id].is_cpu) {
            try self.block_alloc.swapFromCpu(block_id);
            self.reloaded_total += 1;
            debugz.dbg.printLevel(.detail, "[kv_offload] reloaded block={d} from CPU (reloaded_total={d})\n", .{ block_id, self.reloaded_total });
        }
    }

    pub fn report(self: *Self) void {
        if (!self.enabled) return;
        debugz.dbg.printLevel(.info, "[kv_offload] spilled={d} reloaded={d} free={d}/{d}\n", .{
            self.spilled_total,
            self.reloaded_total,
            self.block_alloc.numFree(),
            self.block_alloc.numTotal(),
        });
    }
};

