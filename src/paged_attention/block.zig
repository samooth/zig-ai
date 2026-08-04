const std = @import("std");

pub const Block = struct {
    id: usize,
    ref_count: u32 = 0,
    block_hash: ?u64 = null,
    is_cpu: bool = false,
    data: ?[*]u8 = null,
    num_tokens: u32 = 0,

    const Self = @This();

    pub fn init(id: usize) Self {
        return .{ .id = id };
    }

    pub fn acquire(self: *Self) void {
        self.ref_count += 1;
    }

    pub fn release(self: *Self) void {
        std.debug.assert(self.ref_count > 0);
        self.ref_count -= 1;
    }

    pub fn isFree(self: *const Self) bool {
        return self.ref_count == 0;
    }

    pub fn isShared(self: *const Self) bool {
        return self.ref_count > 1;
    }
};

pub fn hashTokens(tokens: []const u32) u64 {
    var h: u64 = 0xcbf29ce484222325;
    for (tokens) |t| {
        h ^= t;
        h *%= 0x100000001b3;
    }
    return h;
}
