//! MH-9 (RLT): RSS guard — limitación reactiva de memoria residente.
//!
//! RssGuard dispara un trim cuando el RSS supera el límite. No es reactivo
//! por token: el check se ejecuta cada N tokens (default 16) para evitar
//! overhead en el hot-path.
//!
//! Uso:
//!   var guard = RssGuard.init(.{ .limit_gb = 12.0 });
//!   guard.check(current_rss_gb, ecache); // trim si excede
const std = @import("std");
const debugz = @import("debug");

pub const RssGuard = struct {
    limit_gb: f64,
    check_interval: u32,
    tokens_since_check: u32 = 0,
    last_check_ns: i128 = 0,
    last_rss_gb: f64 = 0.0,

    pub fn init(config: struct { limit_gb: f64, check_interval: u32 }) RssGuard {
        return .{
            .limit_gb = config.limit_gb,
            .check_interval = config.check_interval,
        };
    }

    /// Llamar por token. Solo hace trabajo cada `check_interval` tokens.
    /// `current_rss_gb` viene del caller (platform_get_rss_gb o similar).
    /// `disk_used_gb` es el disco usado por offload (MH-8: 3-tier).
    /// `ecache` es el evacuator del pool de activaciones (offload si existe).
    pub fn check(self: *RssGuard, current_rss_gb: f64, disk_used_gb: f64, ecache: ?*anyopaque) void {
        self.tokens_since_check += 1;
        if (self.tokens_since_check < self.check_interval) return;
        self.tokens_since_check = 0;

        self.last_rss_gb = current_rss_gb;
        self.last_check_ns = nowNs();

        const total_gb = current_rss_gb + disk_used_gb;
        if (total_gb > self.limit_gb) {
            debugz.dbg.print("[rss_guard] RSS {d:.2} + disk {d:.2} = {d:.2} GB > limit {d:.2} GB — trim\n", .{
                current_rss_gb, disk_used_gb, total_gb, self.limit_gb
            });
            if (ecache) |cache| {
                cache.*.evictIfRssOver(self.limit_gb);
            }
        }
    }

    pub fn report(self: RssGuard) void {
        debugz.dbg.print("[rss_guard] limit={d:.2} GB last_rss={d:.2} GB tokens_since={d}\n", .{
            self.limit_gb, self.last_rss_gb, self.tokens_since_check,
        });
    }
};

fn nowNs() i128 {
    return @import("time").Timer.now();
}

test "RssGuard: no trim bajo el límite" {
    var g = RssGuard.init(.{ .limit_gb = 10.0, .check_interval = 4 });
    g.check(5.0, 0.0, null);
    g.check(6.0, 0.0, null);
    g.check(7.0, 0.0, null);
    // Solo el 4to token dispara check
    g.check(8.0, 0.0, null);
}

test "RssGuard: trim al exceder límite (RSS + disk)" {
    var g = RssGuard.init(.{ .limit_gb = 10.0, .check_interval = 1 });
    g.check(8.0, 3.5, null); // 8 + 3.5 = 11.5 > 10
    try std.testing.expect(g.last_rss_gb == 8.0);
}
