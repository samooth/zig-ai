//! Elastic memory budget with runtime rebuild (FreeToken Technique 6).
//!
//! Single source of truth for the inference memory budget
//! (FreeToken cache_budget.py:31):
//!
//!   net = round(memory_ratio * baseline_free) − weights_bytes − fixed_cache
//!
//! The budget is *elastic*: when the VRAM/RAM landscape changes (weights
//! re-resident, KV cache resized, OS pressure), the runtime rebuild protocol
//! re-derives the budget and, if needed, re-fits the caches:
//!
//!   pre-validate → fit-check → snapshot → apply → rollback on failure
//!
//! All arithmetic is pure and testable (no GPU calls); the caller supplies
//! `baseline_free` from the device prober and applies the resulting plan.
const std = @import("std");
const debugz = @import("debug");
const tier = @import("tier");

/// Fraction of baseline free memory usable for caches (FreeToken default).
pub const default_memory_ratio: f32 = 0.90;

/// Round-trip safety margin applied to any fit-check (bytes).
pub const safety_margin_bytes: usize = 512 * 1024 * 1024;

/// Pure budget arithmetic (FreeToken cache_budget.py:31, saturating).
/// Returns 0 when the fixed costs alone exceed the pool — never wraps.
pub fn netCacheBudgetBytes(
    memory_ratio: f32,
    baseline_free: usize,
    weights_bytes: usize,
    fixed_cache: usize,
) usize {
    if (memory_ratio <= 0 or baseline_free == 0) return 0;
    const scaled: f64 = @as(f64, memory_ratio) * @as(f64, @floatFromInt(baseline_free));
    if (scaled >= @as(f64, @floatFromInt(std.math.maxInt(u53))) or !std.math.isFinite(scaled)) {
        // Absurd inputs → clamp to the raw pool (fixed costs subtracted below).
        return baseline_free -| weights_bytes -| fixed_cache;
    }
    const total: usize = @intFromFloat(@round(scaled));
    return total -| weights_bytes -| fixed_cache;
}

/// Component of the budget snapshot (what the caches must fit into).
pub const BudgetSnapshot = struct {
    /// Baseline free bytes at budget-derivation time.
    baseline_free: usize,
    /// Bytes of resident weights (device-side).
    weights_bytes: usize,
    /// Fixed cache bytes that cannot be rebuilt (KV blocks already in use,
    /// activation scratch, CUDA context overhead...).
    fixed_cache: usize,
    /// Ratio of baseline_free usable for caches.
    memory_ratio: f32,

    pub fn net(self: BudgetSnapshot) usize {
        return netCacheBudgetBytes(self.memory_ratio, self.baseline_free, self.weights_bytes, self.fixed_cache);
    }
};

/// One resizable consumer of the cache budget (e.g. PagedAttention pool,
/// expert cache, quantized-weight cache). `current` is what it holds now,
/// `desired` is what it would grow to; `shrinkable_floor` is how far it can
/// be trimmed without data loss.
///
/// MH-14: `heat`/`last_access` feed the LFRU tiebreaker from `tier.zig` when
/// the budget is insufficient and `planLfru()` is used.
pub const Consumer = struct {
    name: []const u8,
    current: usize,
    desired: usize,
    shrinkable_floor: usize,
    /// Bytes actually granted by the last plan (set by plan()).
    granted: usize = 0,
    /// LFRU metadata: higher heat + recent access = less likely to be cut.
    heat: u32 = 0,
    last_access: u64 = 0,

    fn slack(self: Consumer) usize {
        return self.current -| self.shrinkable_floor;
    }
};

/// Result of a rebuild plan.
pub const PlanResult = enum {
    /// All consumers fit at their desired sizes.
    fit,
    /// Budget was insufficient; consumers were capped proportionally.
    capped,
    /// Even the floors don't fit; caller must evict/rollback.
    overcommitted,
};

/// Computes a feasible allocation plan over `consumers` within `budget`.
/// Pure: mutates only `granted` on each consumer. Uses proportional
/// water-filling: first guarantees every floor, then distributes the
/// remaining budget proportionally to (desired − floor).
pub fn plan(budget: usize, consumers: []Consumer) PlanResult {
    var floors: usize = 0;
    var wants: usize = 0;
    for (consumers) |*c| {
        floors += c.shrinkable_floor;
        wants += c.desired -| c.shrinkable_floor;
    }
    if (floors > budget) return .overcommitted;

    const disposable = budget - floors;
    if (wants <= disposable) {
        for (consumers) |*c| c.granted = c.desired;
        return .fit;
    }

    // Proportional capping of the growth region.
    for (consumers) |*c| {
        const want_i = c.desired -| c.shrinkable_floor;
        if (wants == 0) {
            c.granted = c.shrinkable_floor;
            continue;
        }
        // floor + want_i * disposable / wants, computed without overflow:
        const scaled_num = @as(u128, want_i) * @as(u128, disposable);
        c.granted = c.shrinkable_floor + @as(usize, @intCast(scaled_num / @as(u128, wants)));
    }
    return .capped;
}

/// LFRU-aware plan variant (MH-14): same floor guarantee as `plan()`, but
/// when the disposable budget is insufficient, growth is not distributed
/// proportionally. Instead, `tier.pick_lfru()` selects the coldest consumer
/// first, and we iteratively cap the coldest until everything fits or we
/// reach `overcommitted`.
///
/// Preconditions:
/// - `consumers[i].heat` and `consumers[i].last_access` must be populated
///   by the caller before invoking this function.
/// - `clock` is the current logical timestamp (e.g. global step counter).
pub fn planLfru(budget: usize, consumers: []Consumer, clock: u64) PlanResult {
    var floors: usize = 0;
    var wants: usize = 0;
    for (consumers) |*c| {
        floors += c.shrinkable_floor;
        wants += c.desired -| c.shrinkable_floor;
    }
    if (floors > budget) return .overcommitted;

    const disposable = budget - floors;
    if (wants <= disposable) {
        for (consumers) |*c| c.granted = c.desired;
        return .fit;
    }

    // Initialize granted to floor.
    for (consumers) |*c| {
        c.granted = c.shrinkable_floor;
    }

    // Build temporary LFRU metadata arrays.
    var heats: [256]u32 = undefined;
    var lasts: [256]u64 = undefined;
    var pinned: [256]bool = undefined;
    const n = @min(consumers.len, heats.len);
    for (consumers[0..n], 0..) |c, i| {
        heats[i] = c.heat;
        lasts[i] = c.last_access;
        pinned[i] = false;
    }

    var remaining_disposable = disposable;
    var remaining_wants = wants;
    var remaining: usize = n;

    while (remaining_disposable > 0 and remaining_wants > 0 and remaining > 0) {
        const pick = tier.pick_lfru(
            heats[0..n],
            lasts[0..n],
            clock,
            pinned[0..n],
        ) orelse break;

        const want_i = consumers[pick.slot].desired -| consumers[pick.slot].shrinkable_floor;
        if (want_i == 0) {
            heats[pick.slot] = std.math.maxInt(u32);
            lasts[pick.slot] = 0;
            remaining -= 1;
            continue;
        }

        const share = @min(
            want_i,
            (remaining_disposable * want_i + (remaining_wants - 1)) / remaining_wants,
        );
        consumers[pick.slot].granted += share;
        remaining_disposable -= share;
        remaining_wants -= share;

        // Mark as processed by pushing score to max so pick_lfru skips it.
        heats[pick.slot] = std.math.maxInt(u32);
        lasts[pick.slot] = 0;
        remaining -= 1;
    }

    return .capped;
}

/// State machine of the runtime rebuild protocol:
///   pre-validate → fit-check → snapshot → apply → rollback on failure.
pub const Rebuilder = struct {
    /// Last accepted snapshot (rollback target).
    applied: BudgetSnapshot,
    /// Consumers participating in the rebuild (owned by the caller).
    consumers: []Consumer,
    /// Generation counter, bumped on each successful apply.
    generation: u64 = 0,

    const Self = @This();

    pub fn init(snapshot: BudgetSnapshot, consumers: []Consumer) Self {
        return .{ .applied = snapshot, .consumers = consumers };
    }

    /// Pre-validation: a candidate snapshot must be sane before any
    /// allocation happens (non-zero pool, ratio in (0,1], weights fit).
    pub fn preValidate(candidate: BudgetSnapshot) bool {
        if (!(candidate.memory_ratio > 0.0 and candidate.memory_ratio <= 1.0)) return false;
        if (candidate.baseline_free == 0) return false;
        if (candidate.weights_bytes > candidate.baseline_free) return false;
        return true;
    }

    /// Fit-check + snapshot: computes the plan for `candidate` WITHOUT
    /// applying it. Returns null if pre-validation fails; consumers' granted
    /// fields hold the prospective sizes on return.
    pub fn fitCheck(self: *Self, candidate: BudgetSnapshot) ?PlanResult {
        if (!preValidate(candidate)) return null;
        // Snapshot current grants so a failed apply can restore them.
        for (self.consumers) |*c| {
            c.granted = c.current;
        }
        const budget = candidate.net();
        const result = plan(budget, self.consumers);
        if (result == .overcommitted) return null;
        return result;
    }

    /// Applies the last fit-check: bumps generation, records the snapshot.
    /// The caller performs the actual allocations using `granted`; if any
    /// allocation fails, `rollback()` restores the previous state.
    pub fn apply(self: *Self, candidate: BudgetSnapshot) void {
        self.applied = candidate;
        self.generation += 1;
        for (self.consumers) |*c| {
            c.current = c.granted;
        }
        if (debugz.dbg.at(.info)) {
            debugz.dbg.printLevel(.info, "[budget] rebuild gen={d} net={d} free={d} weights={d} fixed={d}\n", .{
                self.generation, candidate.net(), candidate.baseline_free, candidate.weights_bytes, candidate.fixed_cache,
            });
        }
    }

    /// Restores the last applied snapshot's plan (post-failure path).
    pub fn rollback(self: *Self) void {
        const budget = self.applied.net();
        _ = plan(budget, self.consumers);
        for (self.consumers) |*c| {
            c.current = c.granted;
        }
        if (debugz.dbg.at(.info)) {
            debugz.dbg.printLevel(.info, "[budget] rollback a gen={d} (net={d})\n", .{ self.generation, budget });
        }
    }
};

const testing = std.testing;

test "netCacheBudgetBytes basic arithmetic" {
    // 90% of 10 GiB (f32 ratio carries its own precision), minus 6 GiB
    // weights minus 1 GiB fixed.
    const net = netCacheBudgetBytes(0.90, 10 * 1024 * 1024 * 1024, 6 * 1024 * 1024 * 1024, 1 * 1024 * 1024 * 1024);
    const expected_scaled: usize = @intFromFloat(@round(@as(f64, @as(f32, 0.90)) * @as(f64, @floatFromInt(10 * 1024 * 1024 * 1024))));
    try testing.expectEqual(expected_scaled - 7 * 1024 * 1024 * 1024, net);
}

test "netCacheBudgetBytes saturates instead of wrapping" {
    // Costs exceed the pool → 0, never underflow.
    try testing.expectEqual(@as(usize, 0), netCacheBudgetBytes(0.9, 1000, 2000, 500));
    try testing.expectEqual(@as(usize, 0), netCacheBudgetBytes(0.9, 0, 0, 0));
    try testing.expectEqual(@as(usize, 0), netCacheBudgetBytes(-1.0, 1000, 0, 0));
}

test "netCacheBudgetBytes rounding" {
    // ratio*free = 0.5*3 = 1.5 → rounds to 2; minus 1 = 1.
    try testing.expectEqual(@as(usize, 1), netCacheBudgetBytes(0.5, 3, 1, 0));
}

test "plan fits when everything fits" {
    var consumers = [_]Consumer{
        .{ .name = "kv", .current = 100, .desired = 300, .shrinkable_floor = 100 },
        .{ .name = "experts", .current = 100, .desired = 200, .shrinkable_floor = 50 },
    };
    try testing.expectEqual(PlanResult.fit, plan(600, &consumers));
    try testing.expectEqual(@as(usize, 300), consumers[0].granted);
    try testing.expectEqual(@as(usize, 200), consumers[1].granted);
}

test "plan caps proportionally" {
    var consumers = [_]Consumer{
        .{ .name = "kv", .current = 100, .desired = 300, .shrinkable_floor = 100 }, // wants 200
        .{ .name = "experts", .current = 100, .desired = 300, .shrinkable_floor = 100 }, // wants 200
    };
    // Budget 450: floors=200, disposable=250 of 400 wanted → 62.5% each.
    try testing.expectEqual(PlanResult.capped, plan(450, &consumers));
    try testing.expectEqual(@as(usize, 100 + 125), consumers[0].granted);
    try testing.expectEqual(@as(usize, 100 + 125), consumers[1].granted);
}

test "plan detects overcommit" {
    var consumers = [_]Consumer{
        .{ .name = "kv", .current = 500, .desired = 500, .shrinkable_floor = 500 },
    };
    try testing.expectEqual(PlanResult.overcommitted, plan(400, &consumers));
}

test "plan proportional capping with unequal wants" {
    var consumers = [_]Consumer{
        .{ .name = "kv", .current = 0, .desired = 900, .shrinkable_floor = 0 }, // wants 900
        .{ .name = "exp", .current = 0, .desired = 300, .shrinkable_floor = 0 }, // wants 300
    };
    // Budget 600 of 1200 wanted → 50% each: 450 + 150.
    try testing.expectEqual(PlanResult.capped, plan(600, &consumers));
    try testing.expectEqual(@as(usize, 450), consumers[0].granted);
    try testing.expectEqual(@as(usize, 150), consumers[1].granted);
}

test "Rebuilder protocol: pre-validate → fit-check → apply → rollback" {
    var consumers = [_]Consumer{
        .{ .name = "kv", .current = 100, .desired = 400, .shrinkable_floor = 100 },
    };
    var rb = Rebuilder.init(.{
        .baseline_free = 1000,
        .weights_bytes = 300,
        .fixed_cache = 0,
        .memory_ratio = 1.0,
    }, &consumers);

    // Invalid ratios are rejected before any allocation.
    try testing.expectEqual(false, Rebuilder.preValidate(.{ .baseline_free = 1000, .weights_bytes = 0, .fixed_cache = 0, .memory_ratio = 0.0 }));
    try testing.expectEqual(false, Rebuilder.preValidate(.{ .baseline_free = 1000, .weights_bytes = 0, .fixed_cache = 0, .memory_ratio = 1.5 }));
    try testing.expectEqual(false, Rebuilder.preValidate(.{ .baseline_free = 0, .weights_bytes = 0, .fixed_cache = 0, .memory_ratio = 1.0 }));

    // Valid candidate: net = 1000−300 = 700 ≥ floor → fits.
    const candidate = BudgetSnapshot{ .baseline_free = 1000, .weights_bytes = 300, .fixed_cache = 0, .memory_ratio = 1.0 };
    const result = rb.fitCheck(candidate);
    try testing.expect(result != null);
    try testing.expectEqual(PlanResult.fit, result.?);

    rb.apply(candidate);
    try testing.expectEqual(@as(u64, 1), rb.generation);
    try testing.expectEqual(@as(usize, 400), consumers[0].current);

    // A later, tighter snapshot fails the fit-check (floor over the net).
    const tight = BudgetSnapshot{ .baseline_free = 350, .weights_bytes = 300, .fixed_cache = 0, .memory_ratio = 1.0 };
    try testing.expect(rb.fitCheck(tight) == null);

    // Simulated allocation failure after a partial apply → rollback.
    rb.rollback();
    try testing.expectEqual(@as(usize, 400), consumers[0].current);
}

test "planLfru fits when everything fits" {
    var consumers = [_]Consumer{
        .{ .name = "kv", .current = 100, .desired = 300, .shrinkable_floor = 100, .heat = 10, .last_access = 100 },
        .{ .name = "experts", .current = 100, .desired = 200, .shrinkable_floor = 50, .heat = 20, .last_access = 200 },
    };
    try testing.expectEqual(PlanResult.fit, planLfru(600, &consumers, 300));
    try testing.expectEqual(@as(usize, 300), consumers[0].granted);
    try testing.expectEqual(@as(usize, 200), consumers[1].granted);
}

test "planLfru caps coldest first" {
    var consumers = [_]Consumer{
        .{ .name = "kv", .current = 100, .desired = 300, .shrinkable_floor = 100, .heat = 5, .last_access = 50 },
        .{ .name = "experts", .current = 100, .desired = 300, .shrinkable_floor = 100, .heat = 50, .last_access = 500 },
    };
    // Budget 350: floors=200, disposable=150 of 400 wanted.
    // Coldest (kv, heat=5) should be capped harder than experts (heat=50).
    try testing.expectEqual(PlanResult.capped, planLfru(350, &consumers, 600));
    try testing.expect(consumers[0].granted < consumers[1].granted);
}
