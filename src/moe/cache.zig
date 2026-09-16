//! OffloadCache: limited-slot LRU cache for MoE expert weights.
//!
//! Lane E (P2 of the FreeToken improvements). The cache stores a configurable
//! number of expert weight blobs (slots) and evicts the least-recently-used
//! one when a new expert needs to be loaded. A `step` counter drives the LRU
//! timestamps; callers bump it via `advanceStep` after each decode step.
//!
//! The public surface mirrors the existing `ExpertCache` tests so that the
//! rest of the engine can adopt the new type incrementally. The richer
//! mirror API expected by `tests/test_moe_cache.zig` is also provided.
//!
//! `Slot.ptr` is `?[*]u8` (null = empty) and is freed via the supplied
//! allocator in `deinit`. The `ensureExperts*` paths own the copy and
//! free the transient buffer returned by the load callback.
const std = @import("std");
const debugz = @import("debug");
const budget = @import("budget");

/// Configuration for an OffloadCache.
pub const Config = struct {
    /// Number of transformer layers that own experts.
    num_layers: u32,
    /// Number of experts per layer.
    num_experts: u32,
    /// Number of cache slots (capped to num_layers * num_experts).
    cache_size: u32,
    /// Maximum number of experts fetched in a single `ensureExperts*` call.
    max_fetch: u32,
};

/// A single cache slot. `ptr == null` means the slot is empty.
pub const Slot = struct {
    ptr: ?[*]u8 = null,
    len: usize = 0,
    layer: u16 = 0,
    expert: u16 = 0,
    /// Step counter when this slot was last touched (hit or load).
    last_used: u64 = 0,
};

/// Pack (layer, expert) into a u32 key.
fn keyOf(layer: u16, expert: u16) u32 {
    return (@as(u32, layer) << 16) | @as(u32, expert);
}

/// Limited-slot LRU cache for MoE expert weights.
pub const OffloadCache = struct {
    num_layers: u32,
    num_experts: u32,
    cache_size: u32,
    max_fetch: u32,

    /// Cache slots (length == cache_size).
    slots: []Slot,
    /// Reverse map from (layer, expert) to slot index.
    key_to_slot: std.AutoHashMapUnmanaged(u32, usize),
    /// Forward map: flat_id = layer*num_experts + expert -> slot index (-1 = not resident).
    slot_for_id: []i32,
    /// Reverse map: slot index -> flat_id (-1 = empty slot).
    id_of_slot: []i32,
    /// Output buffer of indices that the caller should fetch (length = max_fetch).
    src_indices: []i32,
    /// Output buffer of slot indices that were evicted (length = max_fetch).
    evict_slots: []i32,
    /// Number of valid entries in `src_indices` after the last mirror call.
    num_indices: i64,
    /// Number of missing experts encountered in the last mirror call.
    num_missing_full: i64,
    /// Current step counter (LRU timestamp).
    step: u64,

    /// Initialize a cache with the given configuration.
    pub fn init(allocator: std.mem.Allocator, cfg: Config) !OffloadCache {
        const total = cfg.num_layers * cfg.num_experts;
        const cache_size = @min(cfg.cache_size, if (total == 0) 0 else total);
        const max_fetch = if (cfg.max_fetch == 0) cache_size else cfg.max_fetch;

        const slots = try allocator.alloc(Slot, cache_size);
        errdefer allocator.free(slots);
        const slot_for_id = try allocator.alloc(i32, total);
        errdefer allocator.free(slot_for_id);
        const id_of_slot = try allocator.alloc(i32, cache_size);
        errdefer allocator.free(id_of_slot);
        const src_indices = try allocator.alloc(i32, max_fetch);
        errdefer allocator.free(src_indices);
        const evict_slots = try allocator.alloc(i32, max_fetch);
        errdefer allocator.free(evict_slots);

        @memset(slot_for_id, -1);
        @memset(id_of_slot, -1);
        @memset(src_indices, -1);
        @memset(evict_slots, -1);

        var key_to_slot: std.AutoHashMapUnmanaged(u32, usize) = .{};
        errdefer key_to_slot.deinit(allocator);

        return .{
            .num_layers = cfg.num_layers,
            .num_experts = cfg.num_experts,
            .cache_size = cache_size,
            .max_fetch = max_fetch,
            .slots = slots,
            .key_to_slot = key_to_slot,
            .slot_for_id = slot_for_id,
            .id_of_slot = id_of_slot,
            .src_indices = src_indices,
            .evict_slots = evict_slots,
            .num_indices = 0,
            .num_missing_full = 0,
            .step = 0,
        };
    }

    /// Free all resources owned by the cache.
    pub fn deinit(self: *OffloadCache, allocator: std.mem.Allocator) void {
        for (self.slots) |*slot| {
            if (slot.ptr) |p| allocator.free(p[0..slot.len]);
        }
        self.key_to_slot.deinit(allocator);
        allocator.free(self.slots);
        allocator.free(self.slot_for_id);
        allocator.free(self.id_of_slot);
        allocator.free(self.src_indices);
        allocator.free(self.evict_slots);
    }

    /// Reset per-call counters (does not free memory).
    pub fn reset(self: *OffloadCache) void {
        self.num_indices = 0;
        self.num_missing_full = 0;
        @memset(self.src_indices, -1);
        @memset(self.evict_slots, -1);
    }

    /// Advance the LRU step counter.
    pub fn advanceStep(self: *OffloadCache) void {
        self.step += 1;
    }

    /// Ensure the experts for `layer` identified by `expert_ids` are resident.
    /// The load callback is invoked for each miss; the returned buffer is
    /// freed by this function and a copy is retained.
    pub fn ensureExperts(
        self: *OffloadCache,
        layer: u16,
        expert_ids: []const u16,
        allocator: std.mem.Allocator,
        loadFn: *const fn (layer: u16, expert: u16) anyerror![]u8,
    ) !void {
        if (layer >= self.num_layers) return error.InvalidLayer;

        for (expert_ids) |expert_id| {
            if (expert_id >= self.num_experts) return error.InvalidExpert;
            const key = keyOf(layer, expert_id);

            if (self.key_to_slot.get(key)) |slot_idx| {
                // Hit: update LRU timestamp.
                self.slots[slot_idx].last_used = self.step;
                continue;
            }

            // Miss: need a free slot or evict one.
            const slot_idx = try self.acquireSlot(allocator);
            const slot = &self.slots[slot_idx];

            const expert_data = try loadFn(layer, expert_id);
            defer allocator.free(expert_data);
            const copy = try allocator.alloc(u8, expert_data.len);
            @memcpy(copy, expert_data);

            slot.ptr = copy.ptr;
            slot.len = expert_data.len;
            slot.layer = layer;
            slot.expert = expert_id;
            slot.last_used = self.step;

            try self.key_to_slot.put(allocator, key, slot_idx);
            const flat_id: usize = @as(usize, layer) * @as(usize, self.num_experts) + @as(usize, expert_id);
            self.slot_for_id[flat_id] = @intCast(slot_idx);
            self.id_of_slot[slot_idx] = @intCast(flat_id);
        }
    }

    /// Mirror variant used by the test suite. `ids` contains flat expert
    /// identifiers (layer * num_experts + expert). Entries with value < 0
    /// are treated as CPU‑served and skipped. The `frac_q16` parameter is
    /// accepted for API compatibility but currently unused; the
    /// implementation fetches every missing expert up to `max_fetch`.
    pub fn ensureExpertsMirror(self: *OffloadCache, layer: u32, ids: []const i32, frac_q16: u32) void {
        _ = frac_q16;
        self.reset();

        if (layer >= self.num_layers) return;

        var n_fetch: usize = 0;
        var n_missing: i64 = 0;
        for (ids) |raw| {
            if (raw < 0) continue; // CPU-served, not a cache fetch.
            const flat_id: usize = @intCast(raw);
            const exp: u16 = @intCast(@as(u32, @intCast(flat_id % @as(usize, self.num_experts))));
            const lay: u16 = @intCast(layer);
            const key = keyOf(lay, exp);

            if (self.key_to_slot.get(key)) |slot_idx| {
                // Hit: update recency and record in src_indices (id stays).
                self.slots[slot_idx].last_used = self.step;
                if (n_fetch < self.src_indices.len) {
                    self.src_indices[n_fetch] = raw;
                    n_fetch += 1;
                }
                continue;
            }

            // Miss.
            n_missing += 1;
            if (n_fetch >= self.max_fetch) continue; // budget exhausted.
            if (self.acquireSlotNoFail()) |slot_idx| {
                // Mark as will-be-loaded by the caller; we don't have data
                // here so we just record the id and the slot.
                self.slots[slot_idx].last_used = self.step;
                self.id_of_slot[slot_idx] = @intCast(flat_id);
                if (n_fetch < self.src_indices.len) {
                    self.src_indices[n_fetch] = raw;
                    n_fetch += 1;
                }
            } else {
                // No free slot and eviction not yet implemented in mirror
                // path: record as still missing.
            }
        }

        self.num_indices = @intCast(n_fetch);
        self.num_missing_full = n_missing;
    }

    /// Find a free slot or evict the LRU one. Returns the slot index.
    /// On allocation failure returns an error.
    fn acquireSlot(self: *OffloadCache, allocator: std.mem.Allocator) !usize {
        if (self.findFreeSlot()) |idx| return idx;
        const victim = self.findLruVictim() orelse return error.OutOfMemory;
        self.evictSlot(victim, allocator);
        return victim;
    }

    /// Same as `acquireSlot` but returns null on failure instead of erroring.
    fn acquireSlotNoFail(self: *OffloadCache) ?usize {
        if (self.findFreeSlot()) |idx| return idx;
        return self.findLruVictim();
    }

    fn findFreeSlot(self: *const OffloadCache) ?usize {
        for (self.slots, 0..) |slot, i| {
            if (slot.ptr == null) return i;
        }
        return null;
    }

    fn findLruVictim(self: *const OffloadCache) ?usize {
        var victim: ?usize = null;
        var best: u64 = std.math.maxInt(u64);
        for (self.slots, 0..) |slot, i| {
            if (slot.last_used < best) {
                best = slot.last_used;
                victim = i;
            }
        }
        return victim;
    }

    fn evictSlot(self: *OffloadCache, slot_idx: usize, allocator: std.mem.Allocator) void {
        const slot = &self.slots[slot_idx];
        if (slot.ptr) |p| allocator.free(p[0..slot.len]);
        const key = keyOf(slot.layer, slot.expert);
        _ = self.key_to_slot.remove(key);
        const flat_id: usize = @as(usize, slot.layer) * @as(usize, self.num_experts) + @as(usize, slot.expert);
        if (flat_id < self.slot_for_id.len) self.slot_for_id[flat_id] = -1;
        self.id_of_slot[slot_idx] = -1;
        slot.* = .{};
    }

    /// Total bytes currently resident (sum of filled slot lengths).
    pub fn residentBytes(self: *const OffloadCache) usize {
        var total: usize = 0;
        for (self.slots) |slot| {
            if (slot.ptr != null) total += slot.len;
        }
        return total;
    }

    /// Bytes that must stay resident (in‑flight layers). Currently 0.
    pub fn shrinkableFloor(self: *const OffloadCache) usize {
        _ = self;
        return 0;
    }

    /// Expose this cache as a `budget.Consumer` for the rebuild protocol.
    /// `desired` is set equal to `current` (no proactive growth); callers
    /// can adjust before calling `budget.plan()`.
    pub fn asConsumer(self: *OffloadCache, name: []const u8) budget.Consumer {
        return .{
            .name = name,
            .current = self.residentBytes(),
            .desired = self.residentBytes(),
            .shrinkable_floor = self.shrinkableFloor(),
        };
    }

    /// Evict slots until resident bytes are at or below `budget_bytes`.
    /// Uses LRU to pick victims.
    pub fn evictToBudget(self: *OffloadCache, budget_bytes: usize, allocator: std.mem.Allocator) void {
        while (self.residentBytes() > budget_bytes) {
            const victim = self.findLruVictim() orelse break;
            const slot = &self.slots[victim];
            if (moeDebug()) {
                debugz.dbg.print("[moe_layer] [moe_cache] evict slot={d} layer={d} expert={d} bytes={d} resident_after={d}\n", .{ victim, slot.layer, slot.expert, slot.len, self.residentBytes() - slot.len });
            }
            self.evictSlot(victim, allocator);
        }
    }
};

fn moeDebug() bool {
    return std.c.getenv("MOE_DEBUG") != null;
}

// ── Backwards-compat alias ──────────────────────────────────────────────────
//
// The pre‑rewrite code exposed `ExpertCache`. Keep that name as a thin alias
// to `OffloadCache` so callers compiled against the old surface still link.
// New code should use `OffloadCache` directly.

/// Legacy alias for `OffloadCache` (kept for ABI compatibility).
pub const ExpertCache = OffloadCache;

// ── Tests ───────────────────────────────────────────────────────────────────

const testing = std.testing;

fn testLoad(layer: u16, expert: u16) anyerror![]u8 {
    const data = try testing.allocator.alloc(u8, 4);
    data[0] = @truncate(layer);
    data[1] = @truncate(expert);
    data[2] = @truncate(layer +% expert);
    data[3] = @truncate(layer -% expert);
    return data;
}

test "OffloadCache init creates correct sized arrays" {
    const allocator = testing.allocator;
    var c = try OffloadCache.init(allocator, .{
        .num_layers = 32,
        .num_experts = 32,
        .cache_size = 32 * 32,
        .max_fetch = 64,
    });
    defer c.deinit(allocator);
    try testing.expectEqual(@as(usize, 32 * 32), c.slots.len);
    try testing.expectEqual(@as(u32, 32), c.num_layers);
    try testing.expectEqual(@as(u32, 32), c.num_experts);
    try testing.expectEqual(@as(u64, 0), c.step);
}

test "advanceStep increments counter" {
    const allocator = testing.allocator;
    var c = try OffloadCache.init(allocator, .{
        .num_layers = 10,
        .num_experts = 2,
        .cache_size = 20,
        .max_fetch = 20,
    });
    defer c.deinit(allocator);
    c.advanceStep();
    try testing.expectEqual(@as(u64, 1), c.step);
    c.advanceStep();
    try testing.expectEqual(@as(u64, 2), c.step);
}

test "ensureExperts loads missing experts and hits on cached" {
    const allocator = testing.allocator;
    var c = try OffloadCache.init(allocator, .{
        .num_layers = 2,
        .num_experts = 4,
        .cache_size = 8,
        .max_fetch = 8,
    });
    defer c.deinit(allocator);

    try c.ensureExperts(0, &[_]u16{ 0, 1 }, allocator, testLoad);
    try testing.expect(c.slots[0].ptr != null);
    try testing.expect(c.slots[1].ptr != null);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 0, 0, 0 }, c.slots[0].ptr.?[0..4]);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 1, 1, 255 }, c.slots[1].ptr.?[0..4]);

    // Hit: same call again should not allocate.
    try c.ensureExperts(0, &[_]u16{ 0, 1 }, allocator, testLoad);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 0, 0, 0 }, c.slots[0].ptr.?[0..4]);

    // Load more experts in different layers.
    try c.ensureExperts(1, &[_]u16{ 2, 3 }, allocator, testLoad);
    try testing.expect(c.slots[2].ptr != null);
    try testing.expect(c.slots[3].ptr != null);
}

test "eviction: LRU evicts oldest slot when full" {
    const allocator = testing.allocator;
    var c = try OffloadCache.init(allocator, .{
        .num_layers = 1,
        .num_experts = 4,
        .cache_size = 2,
        .max_fetch = 4,
    });
    defer c.deinit(allocator);

    // Fill cache with experts 0 and 1.
    try c.ensureExperts(0, &[_]u16{0}, allocator, testLoad);
    c.advanceStep();
    try c.ensureExperts(0, &[_]u16{1}, allocator, testLoad);
    c.advanceStep();

    // Load expert 2: should evict expert 0 (oldest).
    try c.ensureExperts(0, &[_]u16{2}, allocator, testLoad);
    try testing.expect(c.key_to_slot.get(keyOf(0, 0)) == null);
    try testing.expect(c.key_to_slot.get(keyOf(0, 1)) != null);
    try testing.expect(c.key_to_slot.get(keyOf(0, 2)) != null);
}
