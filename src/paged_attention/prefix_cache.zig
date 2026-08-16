const std = @import("std");
const BlockAllocator = @import("allocator.zig").BlockAllocator;

pub const CacheEntry = struct {
    block_hash: u64,
    phys_id: usize,
    num_tokens: usize,
    last_access: u64,
    hits: usize = 0,
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
            entry.hits += 1;
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
        return self.evictWhere(max_age, 0.0);
    }

    /// Evicta entradas frías: no accedidas en `max_age` accesos **o** con muy
    /// pocos hits respecto a la edad (`min_hit_rate`), p. ej. bloque subido a
    /// GPU pero nunca reutilizado. Devuelve los phys_ids desalojados para que
    /// el llamador los baje también del pool GPU.
    pub fn evictCold(self: *Self, max_age: u64, min_hit_rate: f64) []usize {
        return self.evictWhereReturning(max_age, min_hit_rate);
    }

    /// Igual que `evictCold` pero sin tocar la caché: devuelve los phys_ids
    /// de bloques GPU fríos que podrían bajarse del dispositivo.
    pub fn evictGpuCold(self: *Self, max_age: u64, min_hit_rate: f64) []usize {
        if (self.map.count() == 0) return &.{};
        const result = self.allocator.alloc(usize, self.map.count()) catch return &.{};
        var n: usize = 0;
        var iter = self.map.iterator();
        while (iter.next()) |entry| {
            if (self.isCold(entry.value_ptr, max_age, min_hit_rate)) {
                result[n] = entry.value_ptr.phys_id;
                n += 1;
            }
        }
        if (n < result.len) {
            return self.allocator.realloc(result, n) catch result;
        }
        return result;
    }

    fn isCold(self: *const Self, entry: *const CacheEntry, max_age: u64, min_hit_rate: f64) bool {
        const age = self.access_counter - entry.last_access;
        const hit_rate: f64 = if (age == 0) 1.0 else
            @as(f64, @floatFromInt(entry.hits)) / @as(f64, @floatFromInt(age));
        return age >= max_age or (min_hit_rate > 0 and hit_rate < min_hit_rate);
    }

    fn evictWhere(self: *Self, max_age: u64, min_hit_rate: f64) usize {
        const evicted = self.evictWhereReturning(max_age, min_hit_rate);
        self.allocator.free(evicted);
        return evicted.len;
    }

    fn evictWhereReturning(self: *Self, max_age: u64, min_hit_rate: f64) []usize {
        if (self.map.count() == 0) return &.{};
        const keys = self.allocator.alloc(u64, self.map.count()) catch return &.{};
        defer self.allocator.free(keys);

        var n: usize = 0;
        var iter = self.map.iterator();
        while (iter.next()) |entry| {
            if (self.isCold(entry.value_ptr, max_age, min_hit_rate)) {
                keys[n] = entry.key_ptr.*;
                n += 1;
            }
        }

        const result = self.allocator.alloc(usize, n) catch return &.{};
        var m: usize = 0;
        for (keys[0..n]) |hash| {
            if (self.map.fetchRemove(hash)) |removed| {
                result[m] = removed.value.phys_id;
                m += 1;
                self.block_alloc.release(removed.value.phys_id);
                self.proactive_evictions += 1;
            }
        }
        if (m < result.len) {
            return self.allocator.realloc(result, m) catch result;
        }
        return result;
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