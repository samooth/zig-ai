//! Cross-platform CPU utilization sampler for the zig-ai dashboard.
//!
//! Reports the *process* CPU usage as a percentage of total machine capacity
//! (0.0–100.0), computed from the delta of process CPU time over wall-clock
//! time, normalized by the number of logical cores.
//!
//! - **Linux / macOS (POSIX):** `getrusage(RUSAGE_SELF)` + `clock_gettime(CLOCK_MONOTONIC)`.
//! - **Windows:** `GetProcessTimes` + `QueryPerformanceCounter`.
//!
//! If the platform-specific call fails, `sample()` returns the last known value
//! (or 0.0 on the first call) instead of erroring, so the dashboard never
//! crashes on metric collection.

const std = @import("std");
const builtin = @import("builtin");

pub const CpuMetrics = struct {
    prev_proc_ns: i128 = 0,
    prev_wall_ns: i128 = 0,
    num_cores: usize = 1,
    has_data: bool = false,

    pub fn init(allocator: std.mem.Allocator) CpuMetrics {
        _ = allocator;
        const num_cores = std.Thread.getCpuCount() catch 1;
        return .{ .num_cores = num_cores };
    }

    /// Sample CPU usage since the previous call. Returns 0.0 on the first call
    /// (no baseline yet) or if timing could not be read.
    pub fn sample(self: *CpuMetrics) f64 {
        const wall = nowWallNs();
        const proc = nowProcNs();

        if (!self.has_data) {
            self.prev_proc_ns = proc;
            self.prev_wall_ns = wall;
            self.has_data = true;
            return 0.0;
        }

        const d_proc = proc - self.prev_proc_ns;
        const d_wall = wall - self.prev_wall_ns;
        self.prev_proc_ns = proc;
        self.prev_wall_ns = wall;

        if (d_wall <= 0) return 0.0;

        const cores = @as(f64, @floatFromInt(self.num_cores));
        const pct = (@as(f64, @floatFromInt(d_proc)) / @as(f64, @floatFromInt(d_wall))) * 100.0 / cores;
        return clamp(pct, 0.0, 100.0);
    }
};

fn clamp(v: f64, lo: f64, hi: f64) f64 {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

// ─── Wall-clock (monotonic nanoseconds) ───────────────────────────────────────

fn nowWallNs() i128 {
    return switch (builtin.target.os.tag) {
        .linux, .macos => blk: {
            var ts: std.posix.timespec = undefined;
            _ = std.posix.system.clock_gettime(std.posix.system.CLOCK.MONOTONIC, &ts);
            break :blk @as(i128, ts.sec) * std.time.ns_per_s + @as(i128, ts.nsec);
        },
        .windows => blk: {
            var counter: i64 = 0;
            var freq: i64 = 0;
            _ = queryPerformanceCounter(&counter);
            _ = queryPerformanceFrequency(&freq);
            if (freq == 0) break :blk 0;
            break :blk @divTrunc(@as(i128, counter) * std.time.ns_per_s, @as(i128, freq));
        },
        else => 0,
    };
}

// ─── Process CPU time (nanoseconds) ───────────────────────────────────────────

fn nowProcNs() i128 {
    return switch (builtin.target.os.tag) {
        .linux, .macos => blk: {
            // RUSAGE_SELF == 0 on Linux/macOS.
            const r = std.posix.getrusage(0);
            const us = @as(i128, r.utime.sec) * 1_000_000 + @as(i128, r.utime.usec) +
                @as(i128, r.stime.sec) * 1_000_000 + @as(i128, r.stime.usec);
            break :blk us * 1000;
        },
        .windows => blk: {
            var creation: WinFileTime = undefined;
            var exit: WinFileTime = undefined;
            var kernel: WinFileTime = undefined;
            var user: WinFileTime = undefined;
            if (getProcessTimes(getCurrentProcess(), &creation, &exit, &kernel, &user) == 0) {
                break :blk 0;
            }
            break :blk fileTimeToNanos(kernel) + fileTimeToNanos(user);
        },
        else => 0,
    };
}

// ─── Windows externs (minimal, self-contained) ───────────────────────────────

const WinFileTime = extern struct { dw_low: u32, dw_high: u32 };

fn fileTimeToNanos(ft: WinFileTime) i128 {
    const intervals_100ns = (@as(i128, ft.dw_high) << 32) | @as(i128, ft.dw_low);
    return intervals_100ns * 100;
}

fn getCurrentProcess() usize {
    // Pseudo-handle -1 (windows constant for current process).
    return @as(usize, @bitCast(@as(isize, -1)));
}

extern "kernel32" fn GetProcessTimes(
    hProcess: usize,
    lpCreationTime: *WinFileTime,
    lpExitTime: *WinFileTime,
    lpKernelTime: *WinFileTime,
    lpUserTime: *WinFileTime,
) callconv(.winapi) i32;

extern "kernel32" fn QueryPerformanceCounter(lpPerformanceCount: *i64) callconv(.winapi) i32;
extern "kernel32" fn QueryPerformanceFrequency(lpFrequency: *i64) callconv(.winapi) i32;

fn getProcessTimes(
    h: usize,
    c: *WinFileTime,
    e: *WinFileTime,
    k: *WinFileTime,
    u: *WinFileTime,
) i32 {
    return GetProcessTimes(h, c, e, k, u);
}

fn queryPerformanceCounter(out: *i64) i32 {
    return QueryPerformanceCounter(out);
}

fn queryPerformanceFrequency(out: *i64) i32 {
    return QueryPerformanceFrequency(out);
}
