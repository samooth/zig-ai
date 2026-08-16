const std = @import("std");
const BlockAllocator = @import("allocator.zig").BlockAllocator;

pub const CacheEntry = struct {
    block_hash: u64,
    phys_id: usize,
    num_tokens: usize,
    last_access: u64,
};

pub const PrefixCache = struct {
    allocator: std.mem.Allocator,
    block_alloc: *BlockAllocator,
    map: std.AutoHashMap(u64, CacheEntry),
    max_entries: usize,
    access_counter: u64 = 0,
    hits: usize = 0,
    misses: usize = 0,
    evictions: usize = 0,
    proactive_evictions: usize = 0,

    const Self = @This();

    pub fn init(gpa: std.mem.Allocator, block_alloc: *BlockAllocator, max_entries: usize) !Self {
        return .{
            .allocator = gpa,
            .block_alloc = block_alloc,
            .map = std.AutoHashMap(u64, CacheEntry).init(gpa),
            .max_entries = max_entries,
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.map.valueIterator();
        while (iter.next()) |entry| {
            self.block_alloc.release(entry.phys_id);
        }
        self.map.deinit();
    }

    pub fn insert(self: *Self, block_hash: u64, phys_id: usize, num_tokens: usize) !void {
        self.access_counter += 1;
        if (self.map.contains(block_hash)) return;
        if (self.map.count() >= self.max_entries) {
            try self.evictLRU();
        }
        self.block_alloc.acquire(phys_id);
        errdefer self.block_alloc.release(phys_id);
        try self.map.put(block_hash, .{
            .block_hash = block_hash,
            .phys_id = phys_id,
            .num_tokens = num_tokens,
            .last_access = self.access_counter,
        });
    }

    pub fn lookup(self: *Self, block_hash: u64) ?usize {
        self.access_counter += 1;
        if (self.map.getPtr(block_hash)) |entry| {
            self.hits += 1;
            entry.last_access = self.access_counter;
            return entry.phys_id;
        }
        self.misses += 1;
        return null;
    }

    pub fn longestPrefixMatch(self: *Self, tokens: []const u32, block_size: usize) usize {
        if (tokens.len < block_size) return 0;

        var matched: usize = 0;
        var i: usize = 0;
        while (i + block_size <= tokens.len) : (i += block_size) {
            const hash = @import("block.zig").hashTokens(tokens[i..][0..block_size]);
            if (self.lookup(hash) == null) break;
            matched += block_size;
        }
        return matched;
    }

    fn evictLRU(self: *Self) !void {
        var iter = self.map.iterator();
        var oldest_hash: u64 = 0;
        var oldest_access: u64 = std.math.maxInt(u64);
        var found = false;
        while (iter.next()) |entry| {
            if (entry.value_ptr.last_access < oldest_access) {
                oldest_access = entry.value_ptr.last_access;
                oldest_hash = entry.key_ptr.*;
                found = true;
            }
        }
        if (found) {
            const entry = self.map.get(oldest_hash).?;
            self.block_alloc.release(entry.phys_id);
            _ = self.map.remove(oldest_hash);
            self.evictions += 1;
        }
    }

    pub fn hitRate(self: *const Self) f64 {
        const total = self.hits + self.misses;
        if (total == 0) return 0.0;
        return @as(f64, @floatFromInt(self.hits)) / @as(f64, @floatFromInt(total));
    }

    pub fn evictStale(self: *Self, max_age: u64) usize {
        if (max_age == 0 or self.map.count() == 0) return 0;
        const keys = self.allocator.alloc(u64, self.map.count()) catch return 0;
        defer self.allocator.free(keys);

        var n: usize = 0;
        var iter = self.map.iterator();
        while (iter.next()) |entry| {
            const age = self.access_counter - entry.value_ptr.last_access;
            if (age >= max_age) {
                keys[n] = entry.key_ptr.*;
                n += 1;
            }
        }

        var evicted: usize = 0;
        for (keys[0..n]) |hash| {
            if (self.map.fetchRemove(hash)) |removed| {
                self.block_alloc.release(removed.value.phys_id);
                self.proactive_evictions += 1;
                evicted += 1;
            }
        }
        return evicted;
    }

    pub fn size(self: *const Self) usize {
        return self.map.count();
    }

    pub fn clear(self: *Self) void {
        var iter = self.map.valueIterator();
        while (iter.next()) |entry| {
            self.block_alloc.release(entry.phys_id);
        }
        self.map.clearRetainingCapacity();
    }
};