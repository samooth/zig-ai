const std = @import("std");
const tier_manager = @import("tier_manager");

test "admit fits vram when capacity available" {
    const ctx = tier_manager.AdmissionContext{
        .vram_capacity = 100,
        .ram_capacity = 200,
        .disk_capacity = 300,
        .vram_used = 50,
        .ram_used = 0,
        .disk_used = 0,
        .heat = &.{},
        .last = &.{},
        .pinned = &.{},
        .clock = 0,
        .item_bytes = 30,
    };
    const decision = tier_manager.admit(ctx);
    try std.testing.expect(decision != null);
    try std.testing.expectEqual(tier_manager.Tier.vram, decision.?.tier);
    try std.testing.expectEqualStrings("fits_vram", decision.?.reason);
    try std.testing.expect(!decision.?.evicted);
}

test "admit falls back to ram when vram full and item fits ram" {
    const ctx = tier_manager.AdmissionContext{
        .vram_capacity = 100,
        .ram_capacity = 200,
        .disk_capacity = 300,
        .vram_used = 100,
        .ram_used = 50,
        .disk_used = 0,
        .heat = &.{100},
        .last = &.{0},
        .pinned = &.{false},
        .clock = 10,
        .item_bytes = 80,
    };
    const decision = tier_manager.admit(ctx);
    try std.testing.expect(decision != null);
    try std.testing.expectEqual(tier_manager.Tier.ram, decision.?.tier);
    try std.testing.expectEqualStrings("fits_ram_after_vram_evict", decision.?.reason);
    try std.testing.expect(!decision.?.evicted);
}

test "admit falls back to disk when vram and ram full" {
    const ctx = tier_manager.AdmissionContext{
        .vram_capacity = 100,
        .ram_capacity = 100,
        .disk_capacity = 300,
        .vram_used = 100,
        .ram_used = 100,
        .disk_used = 50,
        .heat = &.{},
        .last = &.{},
        .pinned = &.{},
        .clock = 0,
        .item_bytes = 80,
    };
    const decision = tier_manager.admit(ctx);
    try std.testing.expect(decision != null);
    try std.testing.expectEqual(tier_manager.Tier.disk, decision.?.tier);
    try std.testing.expectEqualStrings("fits_disk_after_ram_evict", decision.?.reason);
    try std.testing.expect(!decision.?.evicted);
}

test "admit returns null when all tiers full" {
    const ctx = tier_manager.AdmissionContext{
        .vram_capacity = 100,
        .ram_capacity = 100,
        .disk_capacity = 100,
        .vram_used = 100,
        .ram_used = 100,
        .disk_used = 100,
        .heat = &.{100},
        .last = &.{0},
        .pinned = &.{false},
        .clock = 10,
        .item_bytes = 10,
    };
    try std.testing.expect(tier_manager.admit(ctx) == null);
}
