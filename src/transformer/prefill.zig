//! Double-buffered prefill for FreeToken's MoE runtime.
//!
//! Overlaps computation and data transfer during the prefill phase by using two
//! buffers: while one buffer is being filled with new tokens, the other buffer is
//! being processed by the transformer layers. This hides the latency of memory
//! operations and improves throughput.
const std = @import("std");
const debugz = @import("debug");

/// Manages double-buffered prefill operations.
pub const DoubleBufferedPrefill = struct {
    /// First buffer for prefill data (e.g., activations or KV cache entries).
    buffer_a: []u8,
    /// Second buffer for prefill data.
    buffer_b: []u8,
    /// Size of each buffer in bytes.
    buffer_size: usize,
    /// Index of the currently active buffer (0 for A, 1 for B).
    active_buffer: u1 = 0,
    /// Whether the manager is initialized.
    initialized: bool = false,

    /// Initializes the double-buffered prefill manager with two buffers of the given size.
    pub fn init(buffer_size: usize, allocator: std.mem.Allocator) !DoubleBufferedPrefill {
        const buffer_a = try allocator.alloc(u8, buffer_size);
        errdefer allocator.free(buffer_a);
        const buffer_b = try allocator.alloc(u8, buffer_size);
        return DoubleBufferedPrefill{
            .buffer_a = buffer_a,
            .buffer_b = buffer_b,
            .buffer_size = buffer_size,
            .active_buffer = 0,
            .initialized = true,
        };
    }

    /// Deinitializes the manager and frees the buffers.
    pub fn deinit(self: *DoubleBufferedPrefill, allocator: std.mem.Allocator) void {
        allocator.free(self.buffer_a);
        allocator.free(self.buffer_b);
        self.initialized = false;
    }

    /// Returns the currently inactive buffer for filling.
    pub fn getInactiveBuffer(self: *DoubleBufferedPrefill) []u8 {
        if (self.active_buffer == 0) {
            return self.buffer_b;
        } else {
            return self.buffer_a;
        }
    }

    /// Returns the currently active buffer for processing.
    pub fn getActiveBuffer(self: *const DoubleBufferedPrefill) []const u8 {
        if (self.active_buffer == 0) {
            return self.buffer_a;
        } else {
            return self.buffer_b;
        }
    }

    /// Swaps the active and inactive buffers.
    pub fn swapBuffers(self: *DoubleBufferedPrefill) void {
        self.active_buffer = if (self.active_buffer == 0) 1 else 0;
    }

    /// Fills the inactive buffer with data from the given source slice.
    /// Returns the destination buffer for further processing if needed.
    pub fn fillInactiveBuffer(self: *DoubleBufferedPrefill, data: []const u8) ![]u8 {
        const inactive = self.getInactiveBuffer();
        if (data.len > self.buffer_size) {
            return error.DataExceedsBufferSize;
        }
        @memcpy(inactive[0..data.len], data);
        return inactive[0..data.len];
    }
};

const testing = std.testing;

test "DoubleBufferedPrefill init and deinit" {
    const allocator = std.testing.allocator;
    var manager = try DoubleBufferedPrefill.init(1024, allocator);
    defer manager.deinit(allocator);
    try testing.expect(manager.initialized);
    try testing.expectEqual(manager.buffer_a.len, 1024);
    try testing.expectEqual(manager.buffer_b.len, 1024);
    try testing.expectEqual(manager.active_buffer, 0);
}

test "DoubleBufferedPrefill buffer access" {
    const allocator = std.testing.allocator;
    var manager = try DoubleBufferedPrefill.init(1024, allocator);
    defer manager.deinit(allocator);
    
    // Initially, active buffer is A, inactive is B
    const active = manager.getActiveBuffer();
    const inactive = manager.getInactiveBuffer();
    try testing.expectEqual(active.ptr, manager.buffer_a.ptr);
    try testing.expectEqual(inactive.ptr, manager.buffer_b.ptr);
    
    // Swap buffers
    manager.swapBuffers();
    try testing.expectEqual(manager.active_buffer, 1);
    const new_active = manager.getActiveBuffer();
    const new_inactive = manager.getInactiveBuffer();
    try testing.expectEqual(new_active.ptr, manager.buffer_b.ptr);
    try testing.expectEqual(new_inactive.ptr, manager.buffer_a.ptr);
}

test "DoubleBufferedPrefill fill inactive buffer" {
    const allocator = std.testing.allocator;
    var manager = try DoubleBufferedPrefill.init(1024, allocator);
    defer manager.deinit(allocator);
    
    const data = [_]u8{ 1, 2, 3, 4, 5 };
    const filled = try manager.fillInactiveBuffer(&data);
    try testing.expectEqual(filled.len, data.len);
    try testing.expectEqual(filled[0], 1);
    try testing.expectEqual(filled[4], 5);
    
    // Check that the data was copied to buffer B (inactive initially)
    for (manager.buffer_b[0..data.len], 0..) |byte, i| {
        try testing.expectEqual(byte, data[i]);
    }
}

test "DoubleBufferedPrefill fill exceeds buffer size" {
    const allocator = std.testing.allocator;
    var manager = try DoubleBufferedPrefill.init(4, allocator);
    defer manager.deinit(allocator);
    
    const data = [_]u8{ 1, 2, 3, 4, 5 };
    const result = manager.fillInactiveBuffer(&data);
    try testing.expectError(error.DataExceedsBufferSize, result);
}