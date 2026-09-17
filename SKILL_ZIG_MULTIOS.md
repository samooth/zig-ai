# Zig 0.16 Cross-Platform Patterns (Linux/macOS/Windows)

Reference for porting Zig 0.16 codebases across all 3 CI targets.

## Zig 0.16 std library: what does NOT exist

| Symbol | Status |
|---|---|
| `std.time.nanoTimestamp` | **Eliminado** |
| `std.time.milliTimestamp` | **Eliminado** |
| `std.time.timestamp` | **Eliminado** |
| `std.time.Timer` | **No existe en 0.16** (era 0.13-0.15) |
| `std.crypto.random` | **Eliminado** |
| `std.c.time` | **Eliminado** |
| `std.c.clock` | **Eliminado** |
| `std.posix.open` | **Eliminado** |
| `std.c.open` | **Eliminado** |

## What DOES work

| API | Notes |
|---|---|
| `std.Io.Clock.now(io, .awake)` | Portable, needs `io: std.Io` |
| `std.Io.File.readPositionalAll(io, buf, offset)` | Cross-platform file read |
| `std.Thread.sleep(ns)` | Cross-platform sleep (replaces `std.c.nanosleep`) |
| `std.process.Args.Iterator.initAllocator(args, alloc)` | Windows-only path (replaces `.init(args)`) |
| `std.os.linux.*` | Linux-only, comptime-guard required |
| `extern "kernel32" fn QueryPerformanceCounter/Frequency` | Windows monotonic clock |
| `extern "kernel32" fn GetSystemTimeAsFileTime` | Windows wall clock |

## Pattern 1: `std.c.clock_gettime` fails on Windows

`std.c.clockid_t` is `void` on Windows → `std.c.clock_gettime` has `void` parameter → compile error even with libc.

**All** callers of `std.c.clock_gettime` or `std.posix.system.clock_gettime` must be guarded.

### Portable monotonic clock (use `time.Timer.now()`)

Create `src/utils/time.zig`:
```zig
const builtin = @import("builtin");
const std = @import("std");

extern "kernel32" fn QueryPerformanceCounter(lpPerformanceCount: *i64) i32;
extern "kernel32" fn QueryPerformanceFrequency(lpFrequency: *i64) i32;

const FileTime = extern struct { dwLowDateTime: u32, dwHighDateTime: u32 };
extern "kernel32" fn GetSystemTimeAsFileTime(lpFileTime: *FileTime) void;

var win_qpc_freq: ?i64 = null;
fn winFreq() i64 { ... }
fn winNowNs() i64 { ... }
fn winWallSec() i64 { ... }

pub const Timer = struct {
    pub fn now() i128 {
        if (builtin.target.os.tag == .windows) return winNowNs();
        var ts: std.posix.timespec = undefined;
        _ = std.posix.system.clock_gettime(.MONOTONIC, &ts);
        return @as(i128, ts.sec) * std.time.ns_per_s + @as(i128, ts.nsec);
    }
};

pub fn wallClockSec() i64 { ... }
```

Add to build.zig for every module that needs it:
```zig
mymod.addImport("time", time_mod);
```

### Inline replacement (when module import isn't feasible)

```zig
fn nowNs() i128 {
    return @import("time").Timer.now();
}
```

## Pattern 2: `extern "c"` glibc declarations fail on cross-compile

`extern "c" fn open64(...)`, `extern "c" fn pread(...)`, `extern "c" fn sched_getaffinity(...)` etc. resolve to glibc symbols that don't exist in musl (Zig's bundled libc).

**Replace with `std.os.linux.*` + comptime guards:**

```zig
fn linuxOpen(path: [*:0]const u8, flags: u32) i32 {
    if (comptime builtin.target.os.tag != .linux) return -1;
    return @intCast(std.os.linux.open(path, @bitCast(flags), 0));
}

fn linuxPread(fd: i32, buf: [*]u8, count: usize, offset: u64) isize {
    if (comptime builtin.target.os.tag != .linux) return -1;
    return std.os.linux.pread(fd, buf, count, offset);
}
```

## Pattern 3: `std.c.timespec` fields are `void` on Windows

`std.c.timespec.sec` and `.nsec` are `void` on Windows → any arithmetic on them fails.

**Guard the entire block:**
```zig
if (comptime builtin.target.os.tag != .windows) {
    const ts: std.c.timespec = .{ .sec = ..., .nsec = ... };
    _ = nanosleep(&ts, null);
} else {
    std.Thread.sleep(us * 1000);
}
```

## Pattern 4: `std.process.Args.Iterator.init` compileError on Windows

```zig
// WRONG on Windows:
var it = std.process.Args.Iterator.init(args);

// RIGHT:
var it = if (comptime builtin.target.os.tag == .windows)
    try std.process.Args.Iterator.initAllocator(args, allocator)
else
    std.process.Args.Iterator.init(args);
```

Note: `initAllocator` returns `!Iterator` (error union), `init` returns `Iterator` directly.

## Pattern 5: `std.DynLib` unsupported on Windows

`std.DynLib.open` and `.lookup` are stubs on Windows → compile error.

```zig
fn ensureLoaded() bool {
    if (g_lib != null) return true;
    if (comptime builtin.target.os.tag != .linux) return false;  // guard
    g_lib = std.DynLib.open("libcuda.so.1") catch return false;
    return true;
}

fn lookup(comptime T: type, name: [:0]const u8) ?T {
    if (comptime builtin.target.os.tag != .linux) return null;  // guard
    return g_lib.?.lookup(T, name);
}
```

## Pattern 6: `f.handle` is `*anyopaque` on Windows

`std.fs.File.handle` on Windows is `os.windows.HANDLE` (`*anyopaque`), not an integer fd.

```zig
// WRONG:
const fd: c_int = @intCast(f.handle);

// RIGHT: guard the entire Linux-specific block
if (comptime builtin.target.os.tag == .linux) {
    const fd: c_int = @intCast(f.handle);
    // io_uring, pread, /proc/self/fd/...
} else {
    _ = f.readPositionalAll(io, buf, 0) catch |e| return e;
}
```

## Pattern 7: extern struct type mismatch (Windows)

Anonymous `extern struct` types are unique per declaration. Two `extern struct { a: u32, b: u32 }` are different types.

```zig
// WRONG:
extern "kernel32" fn GetSystemTimeAsFileTime(
    lpFileTime: *extern struct { dwLowDateTime: u32, dwHighDateTime: u32 }
) void;
fn winWallSec() i64 {
    var ft: extern struct { dwLowDateTime: u32, dwHighDateTime: u32 } = undefined;  // DIFFERENT TYPE!
    GetSystemTimeAsFileTime(&ft);
}

// RIGHT: use a named type
const FileTime = extern struct { dwLowDateTime: u32, dwHighDateTime: u32 };
extern "kernel32" fn GetSystemTimeAsFileTime(lpFileTime: *FileTime) void;
fn winWallSec() i64 {
    var ft: FileTime = undefined;  // SAME TYPE
    GetSystemTimeAsFileTime(&ft);
}
```

## Checklist for cross-platform PR

1. **Grep** for `std.c.clock_gettime`, `std.posix.system.clock_gettime` — all must be guarded
2. **Grep** for `extern "c" fn` — all must use `std.os.linux.*` with comptime guards
3. **Grep** for `f.handle` — must be guarded for non-Linux
4. **Grep** for `std.c.nanosleep`, `std.c.flock` — guard or replace with `std.Thread.sleep`
5. **Grep** for `std.process.Args.Iterator.init` — use `initAllocator` on Windows
6. **Grep** for `std.DynLib` — guard for non-Linux
7. **Verify** `extern struct` types used with `extern "kernel32"` are named, not anonymous
8. **Verify** `build.zig` has `addImport("time", time_mod)` for all modules using the time module
9. **Cross-compile** for all 3 targets: `-Dtarget=x86_64-linux`, `-Dtarget=aarch64-macos`, `-Dtarget=x86_64-windows`
10. **Run** `zig build test` on native target — pre-existing failures are OK, new failures are not
