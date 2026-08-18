//! ActivationPool — pool de memoria para tensores intermedios.
//!
//! En lugar de alloc/free por cada forward pass, mantiene un free-list
//! por (numel, dtype) y reutiliza buffers. LRU eviction cuando el pool
//! supera `max_bytes`.
const std = @import("std");
const Tensor = @import("core").Tensor;
const debug = @import("debug");

const PoolEntry = struct {
    numel: usize,
    freed: bool,
    buffer: []f32,
};

pub const ActivationPool = struct {
    allocator: std.mem.Allocator,
    entries: std.ArrayList(PoolEntry),
    max_bytes: usize,
    used_bytes: usize,
    hits: u64,
    misses: u64,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, max_bytes: usize) Self {
        return .{
            .allocator = allocator,
            .entries = .empty,
            .max_bytes = max_bytes,
            .used_bytes = 0,
            .hits = 0,
            .misses = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.entries.items) |e| {
            self.allocator.free(e.buffer);
        }
        self.entries.deinit(self.allocator);
    }

    /// Alloca (o reutiliza) un buffer de `numel` elementos f32.
    /// El buffer se libera con `release()`.
    pub fn alloc(self: *Self, numel: usize) ![]f32 {
        // Buscar un buffer libre lo suficientemente grande (best-fit)
        var best_idx: ?usize = null;
        var best_size: usize = std.math.maxInt(usize);
        for (self.entries.items, 0..) |e, i| {
            if (e.freed and e.numel >= numel and e.numel < best_size) {
                best_idx = i;
                best_size = e.numel;
            }
        }

        if (best_idx) |idx| {
            self.entries.items[idx].freed = false;
            self.hits += 1;
            return self.entries.items[idx].buffer;
        }

        self.misses += 1;
        const buf = try self.allocator.alloc(f32, numel);
        try self.entries.append(self.allocator, .{
            .numel = numel,
            .freed = false,
            .buffer = buf,
        });
        self.used_bytes += numel * @sizeOf(f32);
        try self.maybeEvict();
        return buf;
    }

    /// Marca un buffer como libre para reutilización.
    pub fn release(self: *Self, buffer: []const f32) void {
        for (self.entries.items) |*e| {
            if (e.buffer.ptr == buffer.ptr and !e.freed) {
                e.freed = true;
                return;
            }
        }
    }

    /// Libera buffers LRU hasta estar bajo `max_bytes`.
    fn maybeEvict(self: *Self) !void {
        if (self.used_bytes <= self.max_bytes) return;
        // Simple: evict oldest freed entries
        var i: usize = 0;
        while (i < self.entries.items.len and self.used_bytes > self.max_bytes) {
            const e = &self.entries.items[i];
            if (e.freed) {
                self.allocator.free(e.buffer);
                self.used_bytes -= e.numel * @sizeOf(f32);
                _ = self.entries.orderedRemove(i);
            } else {
                i += 1;
            }
        }
    }

    pub fn reportMetrics(self: *Self) void {
        if (!debug.dbg.at(.detail)) return;
        const ratio = if (self.hits + self.misses > 0)
            @as(f64, @floatFromInt(self.hits)) / @as(f64, @floatFromInt(self.hits + self.misses)) * 100.0
        else 0.0;
        debug.dbg.printLevel(.detail,
            "ActivationPool: entries={d} used={d}MB hits={d} misses={d} hit_rate={d:.1}%\n",
            .{ self.entries.items.len, self.used_bytes / (1024*1024), self.hits, self.misses, ratio });
    }
};
