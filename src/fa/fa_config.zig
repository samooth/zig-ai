const std = @import("std");

pub const DType = enum {
    f16, bf16, f32,
    pub fn size(self: DType) usize {
        return switch (self) { .f16, .bf16 => 2, .f32 => 4 };
    }
    pub fn name(self: DType) []const u8 {
        return switch (self) { .f16 => "f16", .bf16 => "bf16", .f32 => "f32" };
    }
};

pub const FlashAttentionConfig = struct {
    N: usize, d: usize, num_heads: usize, batch_size: usize,
    dtype: DType = .f16,
    causal: bool = true,
    bq: usize = 64, bkv: usize = 64,

    pub fn scale(self: FlashAttentionConfig) f32 {
        return 1.0 / @sqrt(@as(f32, @floatFromInt(self.d)));
    }
    pub fn qkv_bytes_per_head(self: FlashAttentionConfig) usize {
        return self.N * self.d * self.dtype.size();
    }
    pub fn total_qkv_bytes(self: FlashAttentionConfig) usize {
        return self.batch_size * self.num_heads * self.qkv_bytes_per_head();
    }
    pub fn total_elements(self: FlashAttentionConfig) usize {
        return self.batch_size * self.num_heads * self.N * self.d;
    }
    pub fn validate(self: FlashAttentionConfig) !void {
        if (self.d != 64 and self.d != 128) return error.InvalidConfig;
        if (self.N == 0 or self.batch_size == 0 or self.num_heads == 0) return error.InvalidConfig;
        if (self.bq == 0 or self.bkv == 0) return error.InvalidConfig;
    }
    pub fn validateComptime(comptime self: FlashAttentionConfig) void {
        if (self.d != 64 and self.d != 128) @compileError("Invalid head dimension. Must be 64 or 128.");
        if (self.N == 0) @compileError("N must be > 0");
        if (self.batch_size == 0) @compileError("batch_size must be > 0");
        if (self.num_heads == 0) @compileError("num_heads must be > 0");
    }
};

pub const TileConfig = struct {
    bq: usize, bkv: usize, block_size: usize,
    pub fn forArchitecture(sm: c_int) TileConfig {
        return switch (sm) {
            90 => .{ .bq = 128, .bkv = 128, .block_size = 256 },
            80, 86 => .{ .bq = 64, .bkv = 64, .block_size = 256 },
            75 => .{ .bq = 64, .bkv = 32, .block_size = 128 },
            else => .{ .bq = 64, .bkv = 64, .block_size = 256 },
        };
    }
};

pub const ComputePrecision = enum { f32, f16 };

pub fn formatBytes(allocator: std.mem.Allocator, bytes: usize) ![]u8 {
    const units = [_][]const u8{ "B", "KB", "MB", "GB", "TB" };
    var value = @as(f64, @floatFromInt(bytes));
    var unit_idx: usize = 0;
    while (value >= 1024.0 and unit_idx < units.len - 1) : (unit_idx += 1) value /= 1024.0;
    return try std.fmt.allocPrint(allocator, "{d:.2} {s}", .{ value, units[unit_idx] });
}
