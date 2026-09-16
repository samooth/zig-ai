const std = @import("std");
const gguf = @import("gguf");
const cudaz = @import("cudaz");
const cublas = @import("cublas");
const hybrid_layer = @import("hybrid_layer");
const vram_budget = @import("vram_budget");
const debug = @import("debug");

/// GPU Weight Pool — dedicated pool for layer weights with async H2D/D2H,
/// VRAM budget enforcement, and LRU eviction.
pub const GpuWeightPool = struct {
    allocator: std.mem.Allocator,
    stream: cudaz.CUstream, // Dedicated stream for async H2D/D2H
    pinned_buffer: ?cudaz.CUdeviceptr = null, // Pinned host staging buffer
    pinned_buffer_bytes: usize = 0,
    /// layer_idx -> LayerGpuWeights
    resident: std.AutoHashMap(usize, LayerGpuWeights),
    bytes_used: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    max_bytes: usize,
    vram_budget: ?*vram_budget.VramBudget = null,
    gpu_buffer_pool: ?cublas.GpuMemoryPool = null,

    const Self = @This();

    const LayerGpuWeights = struct {
        tensors: std.ArrayList(GpuTensorHandle),
        bytes: usize,
        last_used: u64 = 0,
        upload_event: ?cudaz.CUevent = null,
    };

    pub const GpuTensorHandle = struct {
        d_ptr: cudaz.CUdeviceptr,
        bytes: usize,
        host_ptr: ?cudaz.CUdeviceptr = null, // pinned host mirror for fast D2H
    };

    pub fn init(
        allocator: std.mem.Allocator,
        stream: cudaz.CUstream,
        max_bytes: usize,
        gpu_buffer_pool: ?cublas.GpuMemoryPool,
        vram_budget_ptr: ?*vram_budget.VramBudget,
    ) !Self {
        var pool = Self{
            .allocator = allocator,
            .stream = stream,
            .max_bytes = max_bytes,
            .resident = std.AutoHashMap(usize, LayerGpuWeights).init(allocator),
            .vram_budget = vram_budget_ptr,
            .gpu_buffer_pool = gpu_buffer_pool,
        };

        // Pre-allocate pinned host buffer for staging (largest layer ~50MB)
        try pool.ensurePinnedBuffer(100 * 1024 * 1024);

        return pool;
    }

    pub fn deinit(self: *Self) void {
        // Evict all resident layers
        var iterator = self.resident.iterator();
        while (iterator.next()) |kv| {
            const layer_idx = kv.key_ptr.*;
            self.evictLayerSync(layer_idx) catch {};
        }
        self.resident.deinit();

        if (self.pinned_buffer) |ptr| {
            cudaz.cuMemFreeHost(@ptrFromInt(ptr));
        }

        if (self.resident.count() > 0) {
            debug.dbg.printLevel(.info, "[gpu_weight_pool] WARNING - {} layers still resident on deinit\n", .{self.resident.count()});
        }
    }

    /// Ensure pinned host staging buffer is large enough
    fn ensurePinnedBuffer(self: *Self, min_bytes: usize) !void {
        if (self.pinned_buffer) |ptr| {
            if (self.pinned_buffer_bytes >= min_bytes) return;
            cudaz.cuMemFreeHost(@ptrFromInt(ptr));
        }
        const ptr = try cudaz.cuMemAllocHost(min_bytes);
        self.pinned_buffer = @intFromPtr(ptr);
        self.pinned_buffer_bytes = min_bytes;
    }

    /// Upload all weights for a layer asynchronously.
    /// Returns CUevent to wait on before compute.
    pub fn uploadLayer(self: *Self, layer: *hybrid_layer.HybridLayer, layer_idx: usize) !cudaz.CUevent {
        // Check VRAM budget first
        const estimated = gpuWeightEstimate(layer);
        if (self.vram_budget) |vram| {
            if (!vram.canAlloc(vram_budget.Category.weights, estimated)) {
                try self.makeRoom(estimated);
            }
            try vram.reserve(vram_budget.Category.weights, estimated);
        }

        // Ensure pinned buffer is large enough
        try self.ensurePinnedBuffer(estimated);

        // Upload all weight tensors for this layer
        var tensors = try std.ArrayList(GpuTensorHandle).initCapacity(self.allocator, 0);
        errdefer tensors.deinit(self.allocator);

        const event = try layer.uploadWeightsToGpu(self, &tensors);

        var layer_weights = LayerGpuWeights{
            .tensors = tensors,
            .bytes = estimated,
            .last_used = blk: {
                var ts: std.c.timespec = undefined;
                _ = std.c.clock_gettime(.MONOTONIC, &ts);
                break :blk @intCast(@divTrunc(@as(i64, ts.sec) * 1000 + @divTrunc(@as(i64, ts.nsec), 1_000_000), 1000));
            },
            .upload_event = event,
        };

        self.resident.put(layer_idx, layer_weights) catch {
            // If insert fails, free uploaded tensors
            self.freeLayerTensors(&layer_weights);
            return error.OutOfMemory;
        };
        _ = self.bytes_used.fetchAdd(estimated, .acq_rel);

        debug.dbg.printLevel(.detail, "[gpu_weight_pool] uploaded layer {} ({:.1} MB), total used: {:.1} MB\n", .{ layer_idx, @as(f64, @floatFromInt(estimated)) / (1024.0 * 1024.0), @as(f64, @floatFromInt(self.bytes_used.load(.acquire))) / (1024.0 * 1024.0) });

        return event;
    }

    /// Synchronous upload (for prefill or when no async overlap)
    pub fn uploadLayerSync(self: *Self, layer: *hybrid_layer.HybridLayer, layer_idx: usize) !void {
        const event = try self.uploadLayer(layer, layer_idx);
        try cudaz.cuEventSynchronize(event);
    }

    /// Wait for layer upload to complete before compute
    pub fn waitForLayer(self: *Self, layer_idx: usize) !void {
        if (self.resident.get(layer_idx)) |weights| {
            if (weights.upload_event) |event| try cudaz.cuEventSynchronize(event);
            var ts: std.c.timespec = undefined;
            _ = std.c.clock_gettime(.MONOTONIC, &ts);
            self.resident.getPtr(layer_idx).?.last_used = @intCast(@divTrunc(@as(i64, ts.sec) * 1000 + @divTrunc(@as(i64, ts.nsec), 1_000_000), 1000));
        }
    }

    /// Check if layer weights are on GPU
    pub fn isResident(self: *Self, layer_idx: usize) bool {
        return self.resident.get(layer_idx) != null;
    }

    /// Evict layer weights from GPU (async D2H or free)
    pub fn evictLayer(self: *Self, layer_idx: usize) !void {
        if (self.resident.fetchRemove(layer_idx)) |kv| {
            var weights = kv.value;
            // Launch async D2H copy for each tensor
            for (weights.tensors.items) |tensor| {
                if (tensor.host_ptr) |host_ptr| {
                    cudaz.cuMemcpyDtoHAsync(host_ptr, tensor.d_ptr, tensor.bytes, self.stream) catch {};
                }
            }
            self.freeLayerTensors(&weights);
            _ = self.bytes_used.fetchSub(weights.bytes, .acq_rel);
            if (self.vram_budget) |vram| {
                vram.release(vram_budget.Category.weights, weights.bytes);
            }
            debug.dbg.printLevel(.detail, "[gpu_weight_pool] evicted layer {} ({:.1} MB freed)\n", .{ layer_idx, @as(f64, @floatFromInt(weights.bytes)) / (1024.0 * 1024.0) });
        }
    }

    /// Synchronous eviction (for shutdown or urgent OOM)
    pub fn evictLayerSync(self: *Self, layer_idx: usize) !void {
        if (self.resident.fetchRemove(layer_idx)) |kv| {
            var weights = kv.value;
            for (weights.tensors.items) |tensor| {
                if (tensor.host_ptr) |host_ptr| {
                    cudaz.cuMemcpyDtoH(host_ptr, tensor.d_ptr, tensor.bytes) catch {};
                }
            }
            self.freeLayerTensors(&weights);
            _ = self.bytes_used.fetchSub(weights.bytes, .acq_rel);
            if (self.vram_budget) |vram| {
                vram.release(vram_budget.Category.weights, weights.bytes);
            }
        }
    }

    /// Force eviction of LRU layers to make room for need_bytes
    pub fn makeRoom(self: *Self, need_bytes: usize) !void {
        if (self.bytes_used.load(.acquire) + need_bytes <= self.max_bytes) return;

        debug.dbg.printLevel(.info, "[gpu_weight_pool] making room for {} MB (used: {:.1} MB, max: {:.1} MB)\n", .{ @as(f64, @floatFromInt(need_bytes)) / (1024.0 * 1024.0), @as(f64, @floatFromInt(self.bytes_used.load(.acquire))) / (1024.0 * 1024.0), @as(f64, @floatFromInt(self.max_bytes)) / (1024.0 * 1024.0) });

        // Find LRU resident layers - use fixed array since max layers is small
        var lru_layers: [128]usize = undefined;
        var lru_count: usize = 0;
        var iterator = self.resident.iterator();
        while (iterator.next()) |kv| {
            lru_layers[lru_count] = kv.key_ptr.*;
            lru_count += 1;
        }

        // Sort by last_used (oldest first) - simple insertion sort for small N
        for (1..lru_count) |i| {
            var j = i;
            while (j > 0) : (j -= 1) {
                const a = lru_layers[j];
                const b = lru_layers[j - 1];
                const ua = self.resident.get(a) orelse break;
                const ub = self.resident.get(b) orelse break;
                if (ua.last_used >= ub.last_used) break;
                lru_layers[j] = b;
                lru_layers[j - 1] = a;
            }
        }

        // Evict until enough room
        for (0..lru_count) |i| {
            const layer_idx = lru_layers[i];
            if (self.bytes_used.load(.acquire) + need_bytes <= self.max_bytes) break;
            self.evictLayerSync(layer_idx) catch {};
        }

        // If still not enough, we have a problem
        if (self.bytes_used.load(.acquire) + need_bytes > self.max_bytes) {
            debug.dbg.printLevel(.info, "[gpu_weight_pool] WARNING - could not make enough room (need {} MB, freed {} MB)\n", .{ @as(f64, @floatFromInt(need_bytes)) / (1024.0 * 1024.0), @as(f64, @floatFromInt(self.max_bytes - self.bytes_used.load(.acquire))) / (1024.0 * 1024.0) });
        }
    }

    fn freeLayerTensors(self: *Self, weights: *LayerGpuWeights) void {
        for (weights.tensors.items) |tensor| {
            // Free GPU memory via pool or directly
            if (self.gpu_buffer_pool) |*pool| {
                const d_ptr: *anyopaque = @ptrFromInt(tensor.d_ptr);
                pool.release(d_ptr);
            } else {
                _ = cudaz.cuMemFree(tensor.d_ptr);
            }
            if (tensor.host_ptr) |_| {
                // Host ptr is in our pinned buffer, don't free individually
            }
        }
        weights.tensors.deinit(self.allocator);
    }

    pub fn bytesUsed(self: *Self) usize {
        return self.bytes_used.load(.acquire);
    }

    pub fn residentCount(self: *Self) usize {
        return self.resident.count();
    }
};

/// Estimate GPU weight bytes for a layer (for VRAM budget)
pub fn gpuWeightEstimate(layer: *hybrid_layer.HybridLayer) usize {
    var total: usize = 0;

    // FFN weights (shared by all layer types)
    total += layer.w_gate.bytes.len;
    total += layer.w_up.bytes.len;
    total += layer.w_down.bytes.len;

    // Attention weights (only for attention layers)
    if (layer.is_attention) {
        if (layer.attn_layer) |*attn| {
            total += attn.w_q.bytes.len;
            total += attn.w_k.bytes.len;
            total += attn.w_v.bytes.len;
            total += attn.w_o.bytes.len;
            total += attn.attn_q_norm.data.len * @sizeOf(f32);
            total += attn.attn_k_norm.data.len * @sizeOf(f32);
        }
    } else if (layer.short_conv_layer) |*sc| {
        // ShortConv weights
        total += sc.w_in_proj.bytes.len;
        total += sc.w_out_proj.bytes.len;
        total += sc.attn_norm.data.len * @sizeOf(f32);
        total += sc.ffn_norm.data.len * @sizeOf(f32);
    } else if (layer.ssm_layer) |*ssm| {
        // SSM weights
        total += ssm.w_qkv.bytes.len;
        total += ssm.w_z.bytes.len;
        total += ssm.w_out.bytes.len;
        total += ssm.w_beta.data.len * @sizeOf(f32);
        total += ssm.w_alpha.data.len * @sizeOf(f32);
        total += ssm.dt_bias.data.len * @sizeOf(f32);
        total += ssm.ssm_a.data.len * @sizeOf(f32);
        total += ssm.conv1d.data.len * @sizeOf(f32);
        total += ssm.ssm_norm.data.len * @sizeOf(f32);
    }

    // Norms
    total += layer.attn_norm.data.len * @sizeOf(f32);
    if (layer.attn_post_norm) |n| total += n.data.len * @sizeOf(f32);

    // Add 20% overhead for dequantized f32 copies and alignment
    return (total * 12) / 10;
}
