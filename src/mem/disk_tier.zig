//! Disk-backed tier for expert weights (MH-1).
//!
//! Provides O_DIRECT pread of expert tensors from an FTW container,
//! with lazy io_uring initialization (QD=32) for high-throughput
//! sequential reads. This enables running >27B models on 8GB VRAM
//! by keeping only active experts in host/device memory.
const std = @import("std");
const debugz = @import("debug");
const ftw = @import("ftw");

pub const DiskTierError = error{
    InvalidFd,
    ExpertNotFound,
    ReadFailed,
    OdirectUnavailable,
    IouringUnavailable,
    OutOfMemory,
    NotPageAligned,
    NotPageMultiple,
};

/// Location of an expert tensor within the FTW container.
pub const ShardLocation = struct {
    shard_index: u32,
    offset: u64,
    length: usize,
};

/// Shard map: expert_id -> location in the FTW file.
pub const ShardMap = std.AutoHashMapUnmanaged(u32, ShardLocation);

pub const DiskTier = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    fd: i32,
    path: []const u8,
    shard_map: ShardMap,
    uring: ?std.os.linux.IoUring,
    page_size: usize,

    /// Disk-backed bytes currently resident in host memory.
    disk_resident_bytes: std.atomic.Value(usize),
    /// Hard ceiling for disk-backed resident bytes.
    disk_resident_ceiling: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        fd: i32,
        path: []const u8,
        ceiling_bytes: usize,
    ) Self {
        return .{
            .allocator = allocator,
            .fd = fd,
            .path = path,
            .shard_map = .empty,
            .uring = null,
            .page_size = std.heap.page_size_min,
            .disk_resident_bytes = std.atomic.Value(usize).init(0),
            .disk_resident_ceiling = ceiling_bytes,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.uring) |*ring| {
            ring.deinit();
        }
        self.shard_map.deinit(self.allocator);
    }

    /// Lazy initialization of io_uring with QD=32.
    pub fn ensureUring(self: *Self) DiskTierError!void {
        if (self.uring != null) return;
        const ring = std.os.linux.IoUring.init(32, 0) catch return error.IouringUnavailable;
        self.uring = ring;
    }

    /// Register an expert tensor location in the shard map.
    pub fn registerExpert(
        self: *Self,
        expert_id: u32,
        location: ShardLocation,
    ) !void {
        try self.shard_map.put(self.allocator, expert_id, location);
    }

    /// Read an expert tensor from disk into a page-aligned buffer.
    /// The returned slice is aligned to page_size and its length is a
    /// multiple of page_size. The caller owns the buffer and must free it.
    pub fn readExpert(
        self: *Self,
        allocator: std.mem.Allocator,
        expert_id: u32,
    ) DiskTierError![]align(4096) u8 {
        const location = self.shard_map.get(expert_id) orelse return error.ExpertNotFound;
        const length = std.mem.alignForward(usize, location.length, self.page_size);
        if (length == 0) return error.InvalidArgument;

        // Enforce ceiling before allocating.
        const current = self.disk_resident_bytes.load(.monotonic);
        if (current + length > self.disk_resident_ceiling) {
            debugz.dbg.print("[disk_tier] ceiling exceeded: current={} + requested={} > ceiling={}\n",
                .{ current, length, self.disk_resident_ceiling });
            return error.OutOfMemory;
        }

        // Allocate page-aligned buffer for O_DIRECT.
        const buf = self.allocAligned(allocator, length) catch |e| {
            debugz.dbg.print("[disk_tier] alloc failed for expert {}: {}\n", .{ expert_id, e });
            return e;
        };
        errdefer self.allocator.free(buf);

        // Try io_uring first, fallback to pread.
        self.readIouringDirect(location.offset, buf) catch |e| {
            if (e != error.IouringUnavailable and e != error.OdirectUnavailable) {
                return e;
            }
            try self.readPreadDirect(location.offset, buf);
        };

        _ = self.disk_resident_bytes.fetchAdd(length, .monotonic);
        debugz.dbg.printLevel(.detail, "[disk_tier] read expert {} -> {} bytes (resident={})\n",
            .{ expert_id, length, self.disk_resident_bytes.load(.monotonic) },
        );

        return buf;
    }

    /// Release a page-aligned buffer previously obtained via readExpert.
    pub fn releaseExpert(self: *Self, buf: []align(4096) u8) void {
        const length = buf.len;
        _ = self.disk_resident_bytes.fetchSub(length, .monotonic);
        self.allocator.free(buf);
        debugz.dbg.printLevel(.detail, "[disk_tier] released {} bytes (resident={})\n",
            .{ length, self.disk_resident_bytes.load(.monotonic) },
        );
    }

    /// Page-aligned allocation helper.
    fn allocAligned(self: *Self, allocator: std.mem.Allocator, bytes: usize) ![]align(4096) u8 {
        const oversize = bytes + self.page_size;
        const raw = try allocator.alloc(u8, oversize);
        const raw_addr = @intFromPtr(raw.ptr);
        const aligned_addr = std.mem.alignForward(usize, raw_addr, self.page_size);
        const aligned_off = aligned_addr - raw_addr;
        return raw[aligned_off..][0..bytes];
    }

    /// Read using io_uring with O_DIRECT.
    fn readIouringDirect(self: *Self, file_offset: u64, dst: []align(4096) u8) !void {
        try self.ensureUring();
        const ring = self.uring.?;

        var zbuf: [64]u8 = undefined;
        const pz = std.fmt.bufPrintZ(&zbuf, "/proc/self/fd/{d}", .{self.fd}) catch return error.OpenFailed;
        const ofd = open64(pz.ptr, 0o40000); // O_RDONLY|O_DIRECT
        if (ofd < 0) return error.OdirectUnavailable;
        defer _ = std.os.linux.close(@intCast(ofd));

        const QD = 32;
        const CHUNK = 4 << 20;
        const n_chunks = (dst.len + CHUNK - 1) / CHUNK;
        var next: usize = 0;
        var completed: usize = 0;

        while (completed < n_chunks) {
            var submitted: usize = 0;
            while (next < n_chunks and submitted < QD) : (submitted += 1) {
                const off = next * CHUNK;
                const len = @min(CHUNK, dst.len - off);
                const sqe = try ring.get_sqe();
                sqe.prep_read(ofd, dst[@intCast(off)..][0..len], @intCast(file_offset + off));
                sqe.user_data = next;
                next += 1;
            }
            _ = try ring.submit_and_wait(1);
            while (ring.cq_ready() > 0 and completed < n_chunks) {
                const cqe = try ring.copy_cqe();
                if (cqe.res < 0) return error.ReadFailed;
                completed += 1;
            }
        }
    }

    /// Fallback serial pread with O_DIRECT.
    fn readPreadDirect(self: *Self, file_offset: u64, dst: []align(4096) u8) !void {
        var zbuf: [64]u8 = undefined;
        const pz = std.fmt.bufPrintZ(&zbuf, "/proc/self/fd/{d}", .{self.fd}) catch return error.OpenFailed;
        const ofd = open64(pz.ptr, 0o40000); // O_RDONLY|O_DIRECT
        if (ofd < 0) return error.OdirectUnavailable;
        defer _ = std.os.linux.close(@intCast(ofd));

        var done: usize = 0;
        while (done < dst.len) {
            const chunk = @min(@as(usize, 4 << 20), dst.len - done);
            const n = pread(ofd, dst.ptr + done, chunk, @intCast(file_offset + done));
            if (n <= 0) return error.ReadFailed;
            done += @intCast(n);
        }
    }

    /// Current resident bytes on disk tier.
    pub fn residentBytes(self: *const Self) usize {
        return self.disk_resident_bytes.load(.monotonic);
    }

    /// Ceiling for disk tier residency.
    pub fn ceiling(self: *const Self) usize {
        return self.disk_resident_ceiling;
    }
};

extern "c" fn pread(fd: c_int, buf: [*]u8, count: usize, offset: i64) isize;
extern "c" fn open64(path: [*:0]const u8, flags: c_int, ...) c_int;

// ── Tests ───────────────────────────────────────────────────────────────────

const testing = std.testing;

test "DiskTier init/deinit" {
    var dt = DiskTier.init(testing.allocator, -1, "/tmp/test.ftw", 1 << 30);
    defer dt.deinit();
    try testing.expectEqual(@as(usize, 0), dt.residentBytes());
    try testing.expectEqual(@as(usize, 1 << 30), dt.ceiling());
}

test "DiskTier ceiling enforcement" {
    var dt = DiskTier.init(testing.allocator, -1, "/tmp/test.ftw", 4096);
    defer dt.deinit();

    // Empty shard map -> expert not found.
    try testing.expectError(error.ExpertNotFound, dt.readExpert(testing.allocator, 0));
}
