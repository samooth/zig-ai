const std = @import("std");

pub fn lfru_score(heat: u32, last: u64, clock: u64) u64 {
    const recent = if (clock >= last) clock -| last else 0;
    return (@as(u64, heat) << 8) | (recent & 0xFF);
}

pub fn should_promote(hot: u32, cold: u32) bool {
    return cold + (cold >> 2) + 4 > hot;
}

pub const LfruPick = struct {
    slot: usize,
    eid: u32,
    gain: u32,
};

pub fn pick_lfru(
    heat: []const u32,
    last: []const u64,
    clock: u64,
    pinned: []const bool,
) ?LfruPick {
    std.debug.assert(heat.len == last.len);
    std.debug.assert(heat.len == pinned.len);
    if (heat.len == 0) return null;

    var best: ?LfruPick = null;
    var best_age: u64 = 0;

    for (heat, last, pinned, 0..) |h, l, p, slot| {
        if (p) continue;
        const age = if (clock >= l) clock -| l else 0;
        if (best == null or h < best.?.eid or (h == best.?.eid and age > best_age)) {
            best_age = age;
            best = LfruPick{
                .slot = slot,
                .eid = h,
                .gain = h,
            };
        }
    }

    return best;
}

pub fn decay(heat: []u32) void {
    for (heat) |*h| {
        h.* = @max(0, @as(i32, @intCast(h.*)) - (h.* >> 3));
    }
}
