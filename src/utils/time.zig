const std = @import("std");

pub const Timer = struct {
    start_ns: i128,

    pub fn start() Timer {
        return .{ .start_ns = now() };
    }

    pub fn read(self: Timer) i128 {
        return now() - self.start_ns;
    }

    pub fn now() i128 {
        var ts: std.posix.timespec = undefined;
        const rc = std.posix.system.clock_gettime(.MONOTONIC, &ts);
        if (rc != 0) return 0;
        return @as(i128, @intCast(ts.sec)) * std.time.ns_per_s + @as(i128, @intCast(ts.nsec));
    }
};
