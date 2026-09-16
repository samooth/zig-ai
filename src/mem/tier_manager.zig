const std = @import("std");
const tier = @import("tier.zig");

/// Tier de residencia para admission/eviction.
pub const Tier = enum {
    vram,
    ram,
    disk,

    pub fn label(self: Tier) []const u8 {
        return switch (self) {
            .vram => "VRAM",
            .ram => "RAM",
            .disk => "Disk",
        };
    }
};

/// Decisión de admisión para un elemento candidato.
pub const AdmissionDecision = struct {
    tier: Tier,
    slot: usize,
    evicted: bool,
    reason: []const u8,
};

/// Contexto read-only del manager para una decisión de admisión.
pub const AdmissionContext = struct {
    /// Capacidad en bytes por tier.
    vram_capacity: usize,
    ram_capacity: usize,
    disk_capacity: usize,
    /// Uso actual en bytes por tier.
    vram_used: usize,
    ram_used: usize,
    disk_used: usize,
    /// Metadata LFRU del pool objetivo.
    heat: []const u32,
    last: []const u64,
    pinned: []const bool,
    clock: u64,
    /// Tamaño del candidato en bytes.
    item_bytes: usize,
};

/// 3-tier admission with LFRU eviction:
///   1. Try VRAM. If fit, admit.
///   2. Else try RAM after evicting coldest VRAM victim (if reclaimable).
///   3. Else try Disk after evicting coldest RAM victim (if reclaimable).
/// Returns null if even Disk cannot fit.
pub fn admit(ctx: AdmissionContext) ?AdmissionDecision {
    if (ctx.item_bytes == 0) return null;

    if (ctx.vram_used + ctx.item_bytes <= ctx.vram_capacity) {
        return .{
            .tier = .vram,
            .slot = 0,
            .evicted = false,
            .reason = "fits_vram",
        };
    }

    const space_after_vram_admit = ctx.vram_capacity -| ctx.item_bytes;
    const vram_reclaimable = ctx.vram_used -| space_after_vram_admit;
    if (vram_reclaimable > 0) {
        if (tier.pick_lfru(ctx.heat, ctx.last, ctx.clock, ctx.pinned)) |pick| {
            if (pick.gain <= vram_reclaimable) {
                return .{
                    .tier = .ram,
                    .slot = pick.slot,
                    .evicted = true,
                    .reason = "evict_vram_to_ram",
                };
            }
        }
    }

    const ram_available = ctx.ram_capacity -| ctx.ram_used;
    if (ram_available >= ctx.item_bytes) {
        return .{
            .tier = .ram,
            .slot = 0,
            .evicted = false,
            .reason = "fits_ram_after_vram_evict",
        };
    }

    const space_after_ram_admit = ctx.ram_capacity -| ctx.item_bytes;
    const ram_reclaimable = ctx.ram_used -| space_after_ram_admit;
    if (ram_reclaimable > 0) {
        if (tier.pick_lfru(ctx.heat, ctx.last, ctx.clock, ctx.pinned)) |pick| {
            if (pick.gain <= ram_reclaimable) {
                return .{
                    .tier = .disk,
                    .slot = pick.slot,
                    .evicted = true,
                    .reason = "evict_ram_to_disk",
                };
            }
        }
    }

    const disk_available = ctx.disk_capacity -| ctx.disk_used;
    if (disk_available >= ctx.item_bytes) {
        return .{
            .tier = .disk,
            .slot = 0,
            .evicted = false,
            .reason = "fits_disk_after_ram_evict",
        };
    }

    return null;
}
