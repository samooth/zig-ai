const std = @import("std");
const PagedKVCache = @import("paged_kv_cache.zig").PagedKVCache;
const BlockTable = @import("block_table.zig").BlockTable;
const PagedConfig = @import("root.zig").PagedConfig;

pub const Phase = enum {
    waiting,
    prefill,
    decode,
    preempted,
    done,
};

pub const Sequence = struct {
    seq_id: u64,
    tokens: std.ArrayList(u32),
    phase: Phase = .waiting,
    num_processed: usize = 0,
    max_new_tokens: usize = 1024,
    generated: usize = 0,
    priority: u32 = 0,
    arrival_time: u64,
    cpu_backup: ?[]u8 = null,
    preempted_step: ?u64 = null,
};

pub const Request = struct {
    req_id: u64 = 0,
    prompt_tokens: []const u32,
    max_new_tokens: usize,
    num_samples: usize = 1,
    priority: u32 = 0,
};

pub const Scheduler = struct {
    allocator: std.mem.Allocator,
    config: PagedConfig,
    kv_cache: *PagedKVCache,
    sequences: std.AutoHashMap(u64, Sequence),
    waiting_queue: std.ArrayList(Request),
    running: std.ArrayList(u64),
    preempted: std.ArrayList(u64),
    next_req_id: u64 = 1,
    step_count: u64 = 0,

    const Self = @This();

    pub fn init(gpa: std.mem.Allocator, config: PagedConfig, kv_cache: *PagedKVCache) Self {
        return .{
            .allocator = gpa,
            .config = config,
            .kv_cache = kv_cache,
            .sequences = std.AutoHashMap(u64, Sequence).init(gpa),
            .waiting_queue = .empty,
            .running = .empty,
            .preempted = .empty,
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.sequences.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.cpu_backup) |backup| self.allocator.free(backup);
            entry.value_ptr.tokens.deinit(self.allocator);
        }
        self.sequences.deinit();
        self.waiting_queue.deinit(self.allocator);
        self.running.deinit(self.allocator);
        self.preempted.deinit(self.allocator);
    }

    pub fn submit(self: *Self, req: Request) !u64 {
        const req_id = self.next_req_id;
        self.next_req_id += 1;
        var req_copy = req;
        req_copy.req_id = req_id;
        try self.waiting_queue.append(self.allocator, req_copy);
        return req_id;
    }

    pub fn schedule(self: *Self) ![]const u64 {
        self.step_count += 1;
        try self.admitRequests();
        try self.allocateBlocks();
        try self.preemptIfNeeded();
        try self.restorePreempted();
        return self.running.items;
    }

    fn admitRequests(self: *Self) !void {
        const i: usize = 0;
        while (i < self.waiting_queue.items.len) {
            const req = &self.waiting_queue.items[i];
            const prefix_len = try self.kv_cache.matchPrefix(req.prompt_tokens);
            const new_tokens = req.prompt_tokens.len - prefix_len;
            const blocks_needed = (new_tokens + self.config.block_size - 1) / self.config.block_size;

            if (!self.kv_cache.canAllocate(blocks_needed + 2)) {
                break;
            }

            for (0..req.num_samples) |_| {
                const seq_id = if (prefix_len > 0)
                    try self.kv_cache.createSequenceWithPrefix(req.prompt_tokens, prefix_len)
                else blk: {
                    const id = try self.kv_cache.createSequence();
                    try self.kv_cache.allocatePrefill(id, req.prompt_tokens.len);
                    break :blk id;
                };
                var seq = Sequence{
                    .seq_id = seq_id,
                    .tokens = .empty,
                    .phase = .prefill,
                    .max_new_tokens = req.max_new_tokens,
                    .arrival_time = self.step_count,
                    .priority = req.priority,
                };
                try seq.tokens.appendSlice(self.allocator, req.prompt_tokens);
                try self.sequences.put(seq_id, seq);
                try self.running.append(self.allocator, seq_id);
                if (prefix_len > 0) {
                    self.sequences.getPtr(seq_id).?.num_processed = prefix_len;
                }
            }
            _ = self.waiting_queue.orderedRemove(i);
        }
    }

    fn allocateBlocks(self: *Self) !void {
        for (self.running.items) |seq_id| {
            var seq = self.sequences.getPtr(seq_id) orelse continue;
            const bt = self.kv_cache.getBlockTableMut(seq_id) orelse continue;

            if (seq.phase == .prefill) {
                seq.phase = .decode;
                seq.num_processed = seq.tokens.items.len;
                try self.kv_cache.cachePrefix(seq_id, seq.tokens.items);
            } else if (seq.phase == .decode) {
                const needed = bt.num_tokens + 1;
                const blocks_needed = (needed + self.config.block_size - 1) / self.config.block_size;
                while (bt.numBlocks() < blocks_needed) {
                    try bt.appendToken(self.kv_cache.block_alloc);
                }
                try bt.prepareWrite(self.kv_cache.block_alloc);
            }
        }
    }

    fn preemptIfNeeded(self: *Self) !void {
        while (self.kv_cache.freeBlocks() == 0 and self.running.items.len > 0) {
            var victim_idx: usize = 0;
            var victim_prio: u32 = std.math.maxInt(u32);
            var victim_arrival: u64 = 0;
            for (self.running.items, 0..) |seq_id, idx| {
                const seq = self.sequences.get(seq_id) orelse continue;
                const worse = seq.priority < victim_prio or
                    (seq.priority == victim_prio and seq.arrival_time > victim_arrival);
                if (idx == 0 or worse) {
                    victim_idx = idx;
                    victim_prio = seq.priority;
                    victim_arrival = seq.arrival_time;
                }
            }

            const victim_id = self.running.items[victim_idx];
            _ = self.running.orderedRemove(victim_idx);
            try self.preempted.append(self.allocator, victim_id);

            var seq = self.sequences.getPtr(victim_id).?;
            seq.phase = .preempted;
            seq.preempted_step = self.step_count;

            const bt = self.kv_cache.getBlockTableMut(victim_id).?;
            const backup_bytes = bt.table.items.len * self.kv_cache.block_alloc.block_bytes;
            const backup = try self.allocator.alloc(u8, backup_bytes);
            var off: usize = 0;
            for (bt.table.items) |phys_id| {
                const src = self.kv_cache.block_alloc.memory_pool[phys_id * self.kv_cache.block_alloc.block_bytes ..][0..self.kv_cache.block_alloc.block_bytes];
                @memcpy(backup[off .. off + self.kv_cache.block_alloc.block_bytes], src);
                off += self.kv_cache.block_alloc.block_bytes;
            }
            seq.cpu_backup = backup;
            self.kv_cache.removeSequence(victim_id);
        }
    }

    fn restorePreempted(self: *Self) !void {
        if (self.kv_cache.freeBlocks() == 0 or self.preempted.items.len == 0) return;
        for (self.preempted.items, 0..) |seq_id, idx| {
            const seq = self.sequences.getPtr(seq_id) orelse continue;
            if (seq.preempted_step != null and seq.preempted_step.? >= self.step_count) continue;
            const needed = seq.num_processed;
            const blocks_needed = (needed + self.config.block_size - 1) / self.config.block_size;
            if (!self.kv_cache.canAllocate(blocks_needed)) continue;

            try self.kv_cache.restoreSequence(seq_id, needed);
            const bt = self.kv_cache.getBlockTableMut(seq_id).?;
            const backup = seq.cpu_backup orelse return;
            var off: usize = 0;
            for (bt.table.items) |phys_id| {
                const dst = self.kv_cache.block_alloc.memory_pool[phys_id * self.kv_cache.block_alloc.block_bytes ..][0..self.kv_cache.block_alloc.block_bytes];
                @memcpy(dst, backup[off .. off + self.kv_cache.block_alloc.block_bytes]);
                off += self.kv_cache.block_alloc.block_bytes;
            }
            self.allocator.free(backup);
            seq.cpu_backup = null;

            self.sequences.getPtr(seq_id).?.phase = .decode;
            _ = self.preempted.orderedRemove(idx);
            try self.running.append(self.allocator, seq_id);
        }
    }

    pub fn finishSequence(self: *Self, seq_id: u64) void {
        if (self.sequences.fetchRemove(seq_id)) |entry| {
            var seq = entry.value;
            if (seq.cpu_backup) |backup| self.allocator.free(backup);
            seq.tokens.deinit(self.allocator);
        }
        for (self.running.items, 0..) |id, i| {
            if (id == seq_id) { _ = self.running.orderedRemove(i); break; }
        }
        for (self.preempted.items, 0..) |id, i| {
            if (id == seq_id) { _ = self.preempted.orderedRemove(i); break; }
        }
        self.kv_cache.removeSequence(seq_id);
    }

    pub fn appendToken(self: *Self, seq_id: u64, token: u32) !void {
        var seq = self.sequences.getPtr(seq_id) orelse return;
        try seq.tokens.append(self.allocator, token);
        seq.generated += 1;
        seq.num_processed += 1;
        try self.kv_cache.appendDecode(seq_id);
        if (seq.generated >= seq.max_new_tokens) {
            seq.phase = .done;
        }
    }

    pub fn numRunning(self: *const Self) usize {
        return self.running.items.len;
    }

    pub fn numWaiting(self: *const Self) usize {
        return self.waiting_queue.items.len;
    }

    pub fn numPreempted(self: *const Self) usize {
        return self.preempted.items.len;
    }
};
