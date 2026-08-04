const std = @import("std");

pub const CacheEntry = struct {
    block_hash: u64,
    phys_id: usize,
    num_tokens: usize,
    last_access: u64,
};

pub const PrefixCache = struct {
    allocator: std.mem.Allocator,
    map: std.AutoHashMap(u64, CacheEntry),
    max_entries: usize,
    access_counter: u64 = 0,

    const Self = @This();

    pub fn init(gpa: std.mem.Allocator, max_entries: usize) !Self {
        return .{
            .allocator = gpa,
            .map = std.AutoHashMap(u64, CacheEntry).init(gpa),
            .max_entries = max_entries,
        };
    }

    pub fn deinit(self: *Self) void {
        self.map.deinit();
    }

    pub fn insert(self: *Self, block_hash: u64, phys_id: usize, num_tokens: usize) !void {
        self.access_counter += 1;
        if (self.map.count() >= self.max_entries) {
            try self.evictLRU();
        }
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
            entry.last_access = self.access_counter;
            return entry.phys_id;
        }
        return null;
    }

    pub fn longestPrefixMatch(self: *Self, tokens: []const u32) usize {
        const block_size = 16;
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
            _ = self.map.remove(oldest_hash);
        }
    }

    pub fn size(self: *const Self) usize {
        return self.map.count();
    }

    pub fn clear(self: *Self) void {
        self.map.clearRetainingCapacity();
    }
};
