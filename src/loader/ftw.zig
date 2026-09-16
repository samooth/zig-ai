//! FTW (Fast Tensor Weights) format loader — FreeToken Technique 5.
//!
//! A minimal container for *fast weight loading*: a single contiguous
//! logical region split into shards of ≤ 8 GiB, with every tensor placed
//! at a 4096-aligned offset so shards can be read with O_DIRECT directly
//! into page-aligned (pinnable) HostBanks. Load time then becomes
//! max(disk BW, CUDA register) with no per-tensor copies.
//!
//! Layout (little-endian):
//!
//!   [FTWHeader]
//!   [tensor directory: FTWEntry × tensor_count]
//!   [tensor data: tensor_count blobs, each 4096-aligned within its shard]
//!
//! The header/directory are tiny (KBs); the data section is addressed by
//! (shard_index, offset_in_shard) so a reader can mmap or O_DIRECT-read
//! exactly the tensors it needs.
//!
//! GGUF→FTW conversion is provided (`convertFromGguf`): it re-lays-out the
//! tensors of a GgufFile into the aligned FTW layout, preserving raw
//! (possibly quantized) bytes — never dequantizing to BF16 on load.
const std = @import("std");
const debugz = @import("debug");
const gguf = @import("gguf");

pub const FTW_MAGIC: u32 = 0x575446; // "FTW" (LE: 'F','T','W',0)
pub const FTW_VERSION: u32 = 1;

/// Max shard size (FreeToken: 8 GiB shards).
pub const max_shard_bytes: usize = 8 * 1024 * 1024 * 1024;

/// Every tensor starts at a multiple of this (O_DIRECT / page requirement).
pub const tensor_align: usize = 4096;

pub const FtwError = error{
    BadMagic,
    BadVersion,
    Truncated,
    EntryNotFound,
    MisalignedEntry,
    ShardTooLarge,
    InvalidArgument,
    OutOfMemory,
};

pub const FTWHeader = extern struct {
    magic: u32,
    version: u32,
    /// Total tensors in the directory.
    tensor_count: u32,
    /// Number of shards the data section spans.
    shard_count: u32,
    /// Bytes of the tensor data section (sum of shard data regions).
    data_bytes: u64,
    /// Byte offset (from file start) where shard 0's data begins; all
    /// shards are contiguous from here.
    data_start: u64,
};

pub const FTWEntry = extern struct {
    /// Offset of the tensor NAME inside the string table (header region).
    name_off: u32,
    name_len: u32,
    /// GGML dtype code (raw bytes are preserved as-is).
    dtype: u32,
    n_dims: u32,
    dims: [4]u64,
    /// Offset of the tensor data relative to `data_start` (always aligned
    /// to `tensor_align`).
    data_off: u64,
    data_len: u64,
};

/// Directory of one converted/loaded FTW container (parsed header +
/// entries; data stays in the backing buffer/shards).
pub const FtwFile = struct {
    allocator: std.mem.Allocator,
    /// Whole container bytes (header + directory + data), mmap-friendly.
    /// The caller owns the backing memory (e.g. an mmap or a HostBank).
    data: []const u8,
    header: FTWHeader,
    /// Borrowed slices into `data`.
    entries: []const FTWEntry,
    /// Borrowed string-table region for tensor names.
    names_blob: []const u8,

    const Self = @This();

    /// Parses and validates the FTW container in `data`.
    pub fn fromBytes(allocator: std.mem.Allocator, data: []const u8) FtwError!Self {
        if (data.len < @sizeOf(FTWHeader)) return error.Truncated;
        var header: FTWHeader = undefined;
        @memcpy(std.mem.asBytes(&header), data[0..@sizeOf(FTWHeader)]);
        if (header.magic != FTW_MAGIC) return error.BadMagic;
        if (header.version != FTW_VERSION) return error.BadVersion;

        const dir_bytes: usize = @as(usize, header.tensor_count) * @sizeOf(FTWEntry);
        const dir_start = @sizeOf(FTWHeader);
        const names_start = dir_start + dir_bytes;
        if (data.len < names_start) return error.Truncated;
        if (header.data_start < names_start) return error.Truncated;
        if (data.len < header.data_start + header.data_bytes) return error.Truncated;

        const entries_bytes = data[dir_start..][0..dir_bytes];
        const entries: []const FTWEntry = @as([*]const FTWEntry, @ptrCast(@alignCast(entries_bytes.ptr)))[0..header.tensor_count];

        const names_blob = data[names_start..@intCast(header.data_start)];
        return .{
            .allocator = allocator,
            .data = data,
            .header = header,
            .entries = entries,
            .names_blob = names_blob,
        };
    }

    pub fn deinit(self: *Self) void {
        // Nothing owned: entries/names borrow from `data` (caller-owned).
        self.data = &.{};
    }

    pub fn tensorCount(self: *const Self) usize {
        return self.entries.len;
    }

    pub fn find(self: *const Self, name: []const u8) ?*const FTWEntry {
        for (self.entries) |*e| {
            const en = self.entryName(e);
            if (std.mem.eql(u8, en, name)) return e;
        }
        return null;
    }

    pub fn entryName(self: *const Self, e: *const FTWEntry) []const u8 {
        return self.names_blob[e.name_off..][0..e.name_len];
    }

    /// Raw tensor bytes (no copy; borrowed from `data`).
    pub fn tensorData(self: *const Self, e: *const FTWEntry) FtwError![]const u8 {
        if (e.data_off % tensor_align != 0) return error.MisalignedEntry;
        const start: usize = @intCast(self.header.data_start + e.data_off);
        if (start + e.data_len > self.data.len) return error.Truncated;
        return self.data[start..][0..@intCast(e.data_len)];
    }

    /// O_DIRECT-friendly read plan: byte ranges within the container for
    /// the given tensors, coalesced and sorted. The reader can issue one
    /// pread per range into a page-aligned HostBank at the same offsets.
    pub fn readPlan(
        self: *const Self,
        allocator: std.mem.Allocator,
        names: []const []const u8,
    ) FtwError![]Range {
        var ranges = try allocator.alloc(Range, names.len);
        errdefer allocator.free(ranges);
        var n: usize = 0;
        for (names) |name| {
            const e = self.find(name) orelse return error.EntryNotFound;
            ranges[n] = .{
                .start = self.header.data_start + e.data_off,
                .len = @intCast(e.data_len),
            };
            n += 1;
        }
        std.mem.sort(Range, ranges[0..n], {}, struct {
            fn lt(_: void, a: Range, b: Range) bool {
                return a.start < b.start;
            }
        }.lt);
        // Coalesce adjacent/overlapping ranges (4096-aligned tensors often
        // sit back-to-back).
        var w: usize = 0;
        for (ranges[0..n]) |r| {
            if (w > 0 and ranges[w - 1].start + ranges[w - 1].len >= r.start) {
                const end = @max(ranges[w - 1].start + ranges[w - 1].len, r.start + r.len);
                ranges[w - 1].len = end - ranges[w - 1].start;
            } else {
                ranges[w] = r;
                w += 1;
            }
        }
        return ranges[0..w];
    }

    pub const Range = struct {
        start: u64,
        len: usize,
    };
};

/// Builder/converter: re-lays-out a GgufFile's tensors into an FTW
/// container (single data region, 4096-aligned offsets). Raw quantized
/// bytes are copied verbatim — never dequantized.
pub const FtwBuilder = struct {
    allocator: std.mem.Allocator,
    entries: std.ArrayList(FTWEntry),
    names: std.ArrayList(u8),
    /// (data_off, bytes) pairs appended in visit order.
    blobs: std.ArrayList(struct { off: u64, bytes: []const u8 }),
    data_len: u64 = 0,
    shard_count: u32 = 1,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .entries = .empty,
            .names = .empty,
            .blobs = .empty,
        };
    }

    pub fn deinit(self: *Self) void {
        self.entries.deinit(self.allocator);
        self.names.deinit(self.allocator);
        self.blobs.deinit(self.allocator);
    }

    /// Adds one tensor (raw bytes preserved).
    pub fn addTensor(
        self: *Self,
        name: []const u8,
        dtype_raw: u32,
        dims: []const u64,
        raw_bytes: []const u8,
    ) FtwError!void {
        if (name.len > 1024 or dims.len > 4) return error.InvalidArgument;
        const off = std.mem.alignForward(u64, self.data_len, tensor_align);
        var e = FTWEntry{
            .name_off = @intCast(self.names.items.len),
            .name_len = @intCast(name.len),
            .dtype = dtype_raw,
            .n_dims = @intCast(dims.len),
            .dims = .{ 0, 0, 0, 0 },
            .data_off = off,
            .data_len = @intCast(raw_bytes.len),
        };
        @memcpy(e.dims[0..dims.len], dims);
        try self.names.appendSlice(self.allocator, name);
        try self.entries.append(self.allocator, e);
        try self.blobs.append(self.allocator, .{ .off = off, .bytes = raw_bytes });
        self.data_len = off + raw_bytes.len;
        self.shard_count = @intCast(@divTrunc(self.data_len, max_shard_bytes) + @intFromBool(self.data_len % max_shard_bytes != 0));
        if (self.shard_count == 0) self.shard_count = 1;
    }

    /// Serializes the container into a freshly allocated buffer.
    pub fn finish(self: *Self) FtwError![]u8 {
        const dir_bytes = self.entries.items.len * @sizeOf(FTWEntry);
        const data_start = std.mem.alignForward(usize, @sizeOf(FTWHeader) + dir_bytes + self.names.items.len, tensor_align);
        const total: usize = data_start + @as(usize, @intCast(self.data_len));
        const buf = self.allocator.alloc(u8, total) catch return error.Truncated;
        errdefer self.allocator.free(buf);
        @memset(buf, 0);

        const header = FTWHeader{
            .magic = FTW_MAGIC,
            .version = FTW_VERSION,
            .tensor_count = @intCast(self.entries.items.len),
            .shard_count = self.shard_count,
            .data_bytes = self.data_len,
            .data_start = @intCast(data_start),
        };
        @memcpy(buf[0..@sizeOf(FTWHeader)], std.mem.asBytes(&header));

        // Directory + names (data_start covers alignment padding).
        if (dir_bytes > 0) {
            const dir_dst = buf[@sizeOf(FTWHeader)..][0..dir_bytes];
            const src: [*]const u8 = @ptrCast(self.entries.items.ptr);
            @memcpy(dir_dst, src[0..dir_bytes]);
        }
        if (self.names.items.len > 0) {
            @memcpy(buf[@sizeOf(FTWHeader) + dir_bytes ..][0..self.names.items.len], self.names.items);
        }

        // Tensor blobs at their aligned offsets.
        for (self.blobs.items) |b| {
            const dst = buf[data_start + @as(usize, @intCast(b.off)) ..][0..b.bytes.len];
            @memcpy(dst, b.bytes);
        }
        return buf;
    }

    /// Converts every tensor of a GgufFile into a new FTW buffer.
    pub fn convertFromGguf(allocator: std.mem.Allocator, g: *const gguf.GgufFile) FtwError![]u8 {
        var b = FtwBuilder.init(allocator);
        defer b.deinit();

        var it = g.tensors.iterator();
        while (it.next()) |kv| {
            const info = kv.value_ptr;
            const raw = g.tensorData(info);
            try b.addTensor(kv.key_ptr.*, @intFromEnum(info.dtype), info.shape(), raw);
        }
        return b.finish();
    }
};

// ── Tests ───────────────────────────────────────────────────────────────────

const testing = std.testing;

test "FtwBuilder round-trip and alignment" {
    const allocator = testing.allocator;

    const a_bytes = [_]u8{ 1, 2, 3, 4 };
    const b_bytes = [_]u8{5} ** 100;

    var b = FtwBuilder.init(allocator);
    defer b.deinit();
    try b.addTensor("attn.q", 0, &[_]u64{4}, &a_bytes);
    try b.addTensor("ffn.down", 0, &[_]u64{ 4, 25 }, &b_bytes);

    const buf = try b.finish();
    defer allocator.free(buf);

    var f = try FtwFile.fromBytes(allocator, buf);
    defer f.deinit();
    try testing.expectEqual(@as(usize, 2), f.tensorCount());
    try testing.expectEqual(@as(u32, 1), f.header.shard_count);

    // Names and raw bytes survive.
    const e_a = f.find("attn.q") orelse return error.TestUnexpectedResult;
    try testing.expectEqualSlices(u8, "attn.q", f.entryName(e_a));
    try testing.expectEqualSlices(u8, &a_bytes, try f.tensorData(e_a));

    const e_b = f.find("ffn.down") orelse return error.TestUnexpectedResult;
    try testing.expectEqualSlices(u8, &b_bytes, try f.tensorData(e_b));

    // Every entry offset is 4096-aligned (O_DIRECT contract).
    for (f.entries) |*e| {
        try testing.expectEqual(@as(u64, 0), e.data_off % tensor_align);
    }

    // The second tensor must not overlap the first: aligned up.
    try testing.expect(e_b.data_off >= e_a.data_off + a_bytes.len);
}

test "FtwFile rejects bad magic and truncation" {
    const allocator = testing.allocator;
    var junk = [_]u8{0} ** 64;
    try testing.expectError(error.BadMagic, FtwFile.fromBytes(allocator, &junk));
    junk[0] = 'F';
    junk[1] = 'T';
    junk[2] = 'W';
    junk[3] = 0;
    // Right magic, wrong version.
    try testing.expectError(error.BadVersion, FtwFile.fromBytes(allocator, &junk));
    // Truncated header.
    try testing.expectError(error.Truncated, FtwFile.fromBytes(allocator, junk[0..8]));
}

test "readPlan coalesces sorted ranges" {
    const allocator = testing.allocator;

    var b = FtwBuilder.init(allocator);
    defer b.deinit();
    const t0 = [_]u8{1} ** 10;
    const t1 = [_]u8{2} ** 10;
    const t2 = [_]u8{3} ** 10;
    try b.addTensor("t0", 0, &.{1}, &t0);
    try b.addTensor("t1", 0, &.{1}, &t1);
    try b.addTensor("t2", 0, &.{1}, &t2);
    const buf = try b.finish();
    defer allocator.free(buf);

    var f = try FtwFile.fromBytes(allocator, buf);
    defer f.deinit();

    const ranges = try f.readPlan(allocator, &[_][]const u8{ "t2", "t0", "t1" });
    defer allocator.free(ranges);
    // Small tensors sit at distinct 4096-aligned offsets → 3 disjoint ranges,
    // sorted by start (t0 < t1 < t2).
    try testing.expectEqual(@as(usize, 3), ranges.len);
    try testing.expect(ranges[0].start < ranges[1].start and ranges[1].start < ranges[2].start);
    for (ranges) |r| try testing.expectEqual(@as(usize, 10), r.len);
}

test "readPlan error on missing tensor" {
    const allocator = testing.allocator;
    var b = FtwBuilder.init(allocator);
    defer b.deinit();
    const t0 = [_]u8{1} ** 10;
    try b.addTensor("t0", 0, &.{1}, &t0);
    const buf = try b.finish();
    defer allocator.free(buf);
    var f = try FtwFile.fromBytes(allocator, buf);
    defer f.deinit();
    try testing.expectError(error.EntryNotFound, f.readPlan(allocator, &[_][]const u8{"missing"}));
}

test "shard count grows with data size" {
    const allocator = testing.allocator;
    var b = FtwBuilder.init(allocator);
    defer b.deinit();
    const blob = [_]u8{7} ** 4096;
    // Two aligned 4-KiB tensors: fits in one shard.
    try b.addTensor("a", 0, &.{4096}, &blob);
    try testing.expectEqual(@as(u32, 1), b.shard_count);
    try testing.expectEqual(@as(u64, 1), @as(u64, @divTrunc(b.data_len, max_shard_bytes) + @intFromBool(b.data_len % max_shard_bytes != 0)));
    const buf = try b.finish();
    defer allocator.free(buf);
    var f = try FtwFile.fromBytes(allocator, buf);
    defer f.deinit();
    try testing.expectEqual(@as(u32, 1), f.header.shard_count);
}
