const std = @import("std");
const cudaz = @import("cudaz");
pub const fa_config = @import("fa_config.zig");
pub const fa_utils = @import("fa_utils.zig");
const fa_kernels = @import("fa_kernels.zig");
const Tensor = @import("core").Tensor;

const FlashAttentionConfig = fa_config.FlashAttentionConfig;
const DType = fa_config.DType;
const CudaKernel = fa_kernels.CudaKernel;
const FlashAttentionBuffers = fa_kernels.FlashAttentionBuffers;

pub const FlashAttentionError = error{
    CudaError, CudaOutOfMemory, CudaInvalidDevice, CudaInvalidValue,
    CudaLaunchFailed, CudaUnknown, InvalidConfig, UnsupportedDtype,
    KernelLaunchFailed, MemoryAllocationFailed, PtxNotFound, CpuNTooLarge,
};

/// Motor principal de FlashAttention
pub const FlashAttention = struct {
    allocator: std.mem.Allocator,
    config: FlashAttentionConfig,
    kernel: CudaKernel,
    buffers: FlashAttentionBuffers,
    stream: cudaz.CUstream,
    h_q: []u8, h_k: []u8, h_v: []u8, h_o: []u8,
    h_q_raw: *anyopaque, h_k_raw: *anyopaque, h_v_raw: *anyopaque, h_o_raw: *anyopaque,
    cuda_context: cudaz.CUcontext,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: FlashAttentionConfig, ptx_path: []const u8) !Self {
        try config.validate();
        try cudaz.cuInit(0);
        const device = try cudaz.cuDeviceGet(0);
        const ctx = try cudaz.cuCtxCreate(0, device);
        var kernel = try CudaKernel.load(ptx_path, "launch_flash_attention");
        errdefer kernel.unload();
        const stream = try cudaz.cuStreamCreate(0);
        var buffers = try FlashAttentionBuffers.alloc(config);
        errdefer buffers.free();
        const total_bytes = config.total_qkv_bytes();
        const h_q_raw = try cudaz.cuMemAllocHost(total_bytes);
        const h_k_raw = try cudaz.cuMemAllocHost(total_bytes);
        const h_v_raw = try cudaz.cuMemAllocHost(total_bytes);
        const h_o_raw = try cudaz.cuMemAllocHost(total_bytes);
        return .{
            .allocator = allocator, .config = config, .kernel = kernel,
            .buffers = buffers, .stream = stream, .cuda_context = ctx,
            .h_q_raw = h_q_raw, .h_k_raw = h_k_raw, .h_v_raw = h_v_raw, .h_o_raw = h_o_raw,
            .h_q = @as([*]u8, @ptrCast(h_q_raw))[0..total_bytes],
            .h_k = @as([*]u8, @ptrCast(h_k_raw))[0..total_bytes],
            .h_v = @as([*]u8, @ptrCast(h_v_raw))[0..total_bytes],
            .h_o = @as([*]u8, @ptrCast(h_o_raw))[0..total_bytes],
        };
    }

    pub fn deinit(self: *Self) void {
        cudaz.cuMemFreeHost(self.h_q_raw); cudaz.cuMemFreeHost(self.h_k_raw);
        cudaz.cuMemFreeHost(self.h_v_raw); cudaz.cuMemFreeHost(self.h_o_raw);
        self.buffers.free(); cudaz.cuStreamDestroy(self.stream);
        self.kernel.unload(); cudaz.cuCtxDestroy(self.cuda_context);
    }

    /// Forward: Q, K, V host -> GPU -> FA -> host
    pub fn forward(self: *Self, Q: Tensor(f16), K: Tensor(f16), V: Tensor(f16), O: *Tensor(f16)) !void {
        const cfg = self.config;
        const total_bytes = cfg.total_qkv_bytes();
        if (Q.shape.len != 4 or K.shape.len != 4 or V.shape.len != 4 or O.shape.len != 4)
            return FlashAttentionError.InvalidConfig;
        if (Q.shape[0] != cfg.batch_size or Q.shape[1] != cfg.num_heads or Q.shape[2] != cfg.N or Q.shape[3] != cfg.d)
            return FlashAttentionError.InvalidConfig;

        @memcpy(self.h_q[0..total_bytes], std.mem.sliceAsBytes(Q.data));
        @memcpy(self.h_k[0..total_bytes], std.mem.sliceAsBytes(K.data));
        @memcpy(self.h_v[0..total_bytes], std.mem.sliceAsBytes(V.data));

        try cudaz.cuMemcpyHtoDAsync(self.buffers.d_q, @intFromPtr(self.h_q.ptr), total_bytes, self.stream);
        try cudaz.cuMemcpyHtoDAsync(self.buffers.d_k, @intFromPtr(self.h_k.ptr), total_bytes, self.stream);
        try cudaz.cuMemcpyHtoDAsync(self.buffers.d_v, @intFromPtr(self.h_v.ptr), total_bytes, self.stream);

        try fa_kernels.launchFlashAttentionV1(self.kernel.function, cfg, self.buffers, self.stream);

        try cudaz.cuMemcpyDtoHAsync(@intFromPtr(self.h_o.ptr), self.buffers.d_o, total_bytes, self.stream);
        try cudaz.cuStreamSynchronize(self.stream);
        @memcpy(std.mem.sliceAsBytes(O.data), self.h_o[0..total_bytes]);
    }

    /// Forward device-to-device (todo en GPU)
    pub fn forwardDevice(self: *Self, d_q: cudaz.CUdeviceptr, d_k: cudaz.CUdeviceptr,
        d_v: cudaz.CUdeviceptr, d_o: cudaz.CUdeviceptr) !void {
        const temp = FlashAttentionBuffers{
            .d_q = d_q, .d_k = d_k, .d_v = d_v, .d_o = d_o,
            .bytes = self.config.total_qkv_bytes(),
        };
        try fa_kernels.launchFlashAttentionV1(self.kernel.function, self.config, temp, self.stream);
        try cudaz.cuStreamSynchronize(self.stream);
    }

    /// Forward batch
    pub fn forwardBatch(self: *Self, Q: []const Tensor(f16), K: []const Tensor(f16),
        V: []const Tensor(f16), O: []*Tensor(f16)) !void {
        if (Q.len != K.len or Q.len != V.len or Q.len != O.len) return FlashAttentionError.InvalidConfig;
        for (Q, K, V, O) |q, k, v, *o| try self.forward(q, k, v, o);
    }

    pub fn printDeviceInfo() !void {
        try cudaz.cuInit(0);
        const device = try cudaz.cuDeviceGet(0);
        var allocator = std.heap.page_allocator;
        const name = try cudaz.getDeviceName(device, allocator);
        defer allocator.free(name);
        const mem = try cudaz.getDeviceTotalMem(device);
        const mem_str = try fa_utils.formatBytes(allocator, mem);
        defer allocator.free(mem_str);
        std.debug.print("CUDA Device: {s}\n", .{name});
        std.debug.print("Total Memory: {s}\n", .{mem_str});
    }
};

/// Version CPU para validacion
pub const FlashAttentionCpu = struct {
    allocator: std.mem.Allocator,
    config: FlashAttentionConfig,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: FlashAttentionConfig) Self {
        return .{ .allocator = allocator, .config = config };
    }

    pub fn forward(self: Self, Q: Tensor(f16), K: Tensor(f16), V: Tensor(f16), O: *Tensor(f16)) !void {
        const cfg = self.config;
        const N = cfg.N; const d = cfg.d; const scale = cfg.scale(); const causal = cfg.causal;
        if (N > 4096) {
            const mem_mb = N * N * 4 / 1024 / 1024;
            std.log.warn("FlashAttentionCpu usa O(N^2) memoria. N={d} requiere {d}MB scores.", .{ N, mem_mb });
        }
        const num_heads = cfg.num_heads; const batch_size = cfg.batch_size;
        const scores = try self.allocator.alloc(f32, N * N);
        defer self.allocator.free(scores);
        const softmax_out = try self.allocator.alloc(f32, N * N);
        defer self.allocator.free(softmax_out);

        for (0..batch_size) |b| {
            for (0..num_heads) |h| {
                const head_offset = ((b * num_heads) + h) * N * d;
                for (0..N) |i| {
                    for (0..N) |j| {
                        var dot: f32 = 0;
                        for (0..d) |k| {
                            dot += fa_utils.f16ToF32(Q.data[head_offset + i * d + k]) *
                                   fa_utils.f16ToF32(K.data[head_offset + j * d + k]);
                        }
                        var score = dot * scale;
                        if (causal and j > i) score = -std.math.inf(f32);
                        scores[i * N + j] = score;
                    }
                }
                for (0..N) |i| {
                    var max_val: f32 = -std.math.inf(f32);
                    for (0..N) |j| {
                        if (scores[i * N + j] > max_val) max_val = scores[i * N + j];
                    }
                    var sum: f32 = 0;
                    for (0..N) |j| { softmax_out[i * N + j] = @exp(scores[i * N + j] - max_val); sum += softmax_out[i * N + j]; }
                    for (0..N) |j| softmax_out[i * N + j] /= sum;
                }
                for (0..N) |i| {
                    for (0..d) |k| {
                        var sum: f32 = 0;
                        for (0..N) |j| sum += softmax_out[i * N + j] * fa_utils.f16ToF32(V.data[head_offset + j * d + k]);
                        O.data[head_offset + i * d + k] = fa_utils.f32ToF16(sum);
                    }
                }
            }
        }
    }
};
