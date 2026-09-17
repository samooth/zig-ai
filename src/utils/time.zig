const builtin = @import("builtin");
const std = @import("std");

// Windows QPC for portable high-res timing
extern "kernel32" fn QueryPerformanceCounter(lpPerformanceCount: *i64) i32;
extern "kernel32" fn QueryPerformanceFrequency(lpFrequency: *i64) i32;
extern "kernel32" fn GetSystemTimeAsFileTime(lpFileTime: *FileTime) void;
const FileTime = extern struct { dwLowDateTime: u32, dwHighDateTime: u32 };

var win_qpc_freq: ?i64 = null;
fn winFreq() i64 {
    if (win_qpc_freq) |f| return f;
    var f: i64 = undefined;
    _ = QueryPerformanceFrequency(&f);
    win_qpc_freq = f;
    return f;
}
fn winNowNs() i64 {
    var counter: i64 = undefined;
    _ = QueryPerformanceCounter(&counter);
    return @divTrunc(counter * std.time.ns_per_s, winFreq());
}
fn winWallSec() i64 {
    var ft: FileTime = undefined;
    GetSystemTimeAsFileTime(&ft);
    const ft100ns: u64 = (@as(u64, ft.dwHighDateTime) << 32) | ft.dwLowDateTime;
    return @intCast(@divTrunc(ft100ns - 116444736000000000, 10000000));
}

pub const Timer = struct {
    start_ns: i128,

    pub fn start() Timer {
        return .{ .start_ns = now() };
    }

    pub fn read(self: Timer) i128 {
        return now() - self.start_ns;
    }

    pub fn now() i128 {
        if (builtin.target.os.tag == .windows) {
            return winNowNs();
        }
        var ts: std.posix.timespec = undefined;
        const rc = std.posix.system.clock_gettime(.MONOTONIC, &ts);
        if (rc != 0) return 0;
        return @as(i128, @intCast(ts.sec)) * std.time.ns_per_s + @as(i128, @intCast(ts.nsec));
    }
};

/// Wall-clock timestamp en segundos (reemplazo de std.time.timestamp() que
/// se eliminó en Zig 0.16). Usa CLOCK_REALTIME POSIX.
pub fn wallClockSec() i64 {
    if (builtin.target.os.tag == .windows) {
        return winWallSec();
    }
    var ts: std.posix.timespec = undefined;
    const rc = std.posix.system.clock_gettime(.REALTIME, &ts);
    if (rc != 0) return 0;
    return @intCast(ts.sec);
}
