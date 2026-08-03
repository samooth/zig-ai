//! Wrappers Zig para kernels de de-cuantizacion CUDA
//! Lanza de-cuantizacion async en stream, sin round-trip a host

const std = @import("std");
const cudaz = @import("cudaz");
const qt = @import("quant_types.zig");
const QuantFormat = qt.QuantFormat;
const QuantizedTensor = qt.QuantizedTensor;

/// Buffers GPU persistentes para K/V de-cuantizados
pub const GpuDequantBuffers = struct {
    d_k_fp16: cudaz.CUdeviceptr,
    d_v_fp16: cudaz.CUdeviceptr,
    max_elements: usize,
    bytes: usize,

    pub fn alloc(max_elements: usize) !GpuDequantBuffers {
        const bytes = max_elements * 2; // f16 = 2 bytes
        const d_k = try cudaz.cuMemAlloc(bytes);
        const d_v = try cudaz.cuMemAlloc(bytes);
        return .{
            .d_k_fp16 = d_k,
            .d_v_fp16 = d_v,
            .max_elements = max_elements,
            .bytes = bytes,
        };
    }

    pub fn free(self: *GpuDequantBuffers) void {
        cudaz.cuMemFree(self.d_k_fp16);
        cudaz.cuMemFree(self.d_v_fp16);
    }

    pub fn ensureSize(self: *GpuDequantBuffers, elements: usize) !void {
        if (elements > self.max_elements) {
            const new_bytes = elements * 2;
            cudaz.cuMemFree(self.d_k_fp16);
            cudaz.cuMemFree(self.d_v_fp16);
            self.d_k_fp16 = try cudaz.cuMemAlloc(new_bytes);
            self.d_v_fp16 = try cudaz.cuMemAlloc(new_bytes);
            self.max_elements = elements;
            self.bytes = new_bytes;
        }
    }
};

/// Motor de de-cuantizacion GPU
pub const GpuDequantEngine = struct {
    allocator: std.mem.Allocator,
    module: cudaz.CUmodule,
    kernels: std.StringHashMap(cudaz.CUfunction),
    stream: cudaz.CUstream,
    buffers: GpuDequantBuffers,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, ptx_path: []const u8, max_elements: usize) !Self {
        try cudaz.cuInit(0);
        const device = try cudaz.cuDeviceGet(0);
        const ctx = try cudaz.cuCtxCreate(0, device);
        _ = ctx;

        const module = try cudaz.cuModuleLoad(ptx_path);
        errdefer cudaz.cuModuleUnload(module);

        var kernels = std.StringHashMap(cudaz.CUfunction).init(allocator);
        errdefer kernels.deinit();

        // Cargar funciones del PTX
        const kernel_names = &.{
            "launch_dequant_int8_sym",
            "launch_dequant_int8_asym",
            "launch_dequant_int4",
            "launch_dequant_q4_0",
            "launch_dequant_q8_0",
        };

        for (kernel_names) |name| {
            const func = try cudaz.cuModuleGetFunction(module, name);
            try kernels.put(name, func);
        }

        const stream = try cudaz.cuStreamCreate(0);
        const buffers = try GpuDequantBuffers.alloc(max_elements);

        return .{
            .allocator = allocator,
            .module = module,
            .kernels = kernels,
            .stream = stream,
            .buffers = buffers,
        };
    }

    pub fn deinit(self: *Self) void {
        self.buffers.free();
        cudaz.cuStreamDestroy(self.stream);
        for (self.kernels.valueIterator()) |func| {
            _ = func;
        }
        self.kernels.deinit();
        cudaz.cuModuleUnload(self.module);
    }

    /// De-cuantiza un QuantizedTensor en device
    /// d_raw: device ptr a datos cuantizados
    /// d_scales, d_zeros: device ptr a metadatos (pueden ser null)
    /// num_elements: numero de valores logicos
    pub fn dequantize(
        self: *Self,
        format: QuantFormat,
        d_raw: cudaz.CUdeviceptr,
        d_scales: ?cudaz.CUdeviceptr,
        d_zeros: ?cudaz.CUdeviceptr,
        num_elements: usize,
        block_size: usize,
    ) !cudaz.CUdeviceptr {
        try self.buffers.ensureSize(num_elements);

        const kernel_name = switch (format) {
            .int8_symmetric => "launch_dequant_int8_sym",
            .int8_asymmetric => "launch_dequant_int8_asym",
            .int4 => "launch_dequant_int4",
            .q4_0 => "launch_dequant_q4_0",
            .q8_0 => "launch_dequant_q8_0",
            else => return error.UnsupportedGpuFormat,
        };

        const func = self.kernels.get(kernel_name) orelse return error.KernelNotFound;

        switch (format) {
            .int8_symmetric => {
                const args = .{
                    &d_raw, &d_scales.?, &self.buffers.d_k_fp16,
                    @as(c_int, @intCast(num_elements)),
                    @as(c_int, @intCast(block_size)),
                    self.stream,
                };
                try cudaz.cuLaunchKernel(func, @intCast((num_elements + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, &args, null);
            },
            .int8_asymmetric => {
                const args = .{
                    &d_raw, &d_scales.?, &d_zeros.?, &self.buffers.d_k_fp16,
                    @as(c_int, @intCast(num_elements)),
                    @as(c_int, @intCast(block_size)),
                    self.stream,
                };
                try cudaz.cuLaunchKernel(func, @intCast((num_elements + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, &args, null);
            },
            .int4 => {
                const args = .{
                    &d_raw, &d_scales.?, &d_zeros.?, &self.buffers.d_k_fp16,
                    @as(c_int, @intCast(num_elements)),
                    @as(c_int, @intCast(block_size)),
                    self.stream,
                };
                try cudaz.cuLaunchKernel(func, @intCast((num_elements / 2 + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, &args, null);
            },
            .q4_0, .q8_0 => {
                const args = .{
                    &d_raw, &self.buffers.d_k_fp16,
                    @as(c_int, @intCast(num_elements)),
                    self.stream,
                };
                try cudaz.cuLaunchKernel(func, @intCast((num_elements + 255) / 256), 1, 1, 256, 1, 1, 0, self.stream, &args, null);
            },
            else => unreachable,
        }

        try cudaz.cuStreamSynchronize(self.stream);
        return self.buffers.d_k_fp16;
    }

    /// De-cuantiza K y V de un bloque del cache
    pub fn dequantizeBlock(
        self: *Self,
        k_format: QuantFormat,
        d_k_raw: cudaz.CUdeviceptr,
        d_k_scales: ?cudaz.CUdeviceptr,
        d_k_zeros: ?cudaz.CUdeviceptr,
        v_format: QuantFormat,
        d_v_raw: cudaz.CUdeviceptr,
        d_v_scales: ?cudaz.CUdeviceptr,
        d_v_zeros: ?cudaz.CUdeviceptr,
        num_elements: usize,
        block_size: usize,
    ) !struct { d_k: cudaz.CUdeviceptr, d_v: cudaz.CUdeviceptr } {
        try self.buffers.ensureSize(num_elements * 2); // K + V

        const d_k = try self.dequantizeInternal(k_format, d_k_raw, d_k_scales, d_k_zeros, num_elements, block_size, true);
        const d_v = try self.dequantizeInternal(v_format, d_v_raw, d_v_scales, d_v_zeros, num_elements, block_size, false);

        return .{ .d_k = d_k, .d_v = d_v };
    }

    fn dequantizeInternal(
        self: *Self,
        format: QuantFormat,
        d_raw: cudaz.CUdeviceptr,
        d_scales: ?cudaz.CUdeviceptr,
        d_zeros: ?cudaz.CUdeviceptr,
        num_elements: usize,
        block_size: usize,
        is_k: bool,
    ) !cudaz.CUdeviceptr {
        const out_ptr = if (is_k) self.buffers.d_k_fp16 else self.buffers.d_v_fp16;

        const kernel_name = switch (format) {
            .int8_symmetric => "launch_dequant_int8_sym",
            .int8_asymmetric => "launch_dequant_int8_asym",
            .int4 => "launch_dequant_int4",
            .q4_0 => "launch_dequant_q4_0",
            .q8_0 => "launch_dequant_q8_0",
            else => return error.UnsupportedGpuFormat,
        };

        const func = self.kernels.get(kernel_name) orelse return error.KernelNotFound;
        const blocks = @as(c_uint, @intCast((num_elements + 255) / 256));

        switch (format) {
            .int8_symmetric => {
                const args = .{ &d_raw, &d_scales.?, &out_ptr, @as(c_int, @intCast(num_elements)), @as(c_int, @intCast(block_size)), self.stream };
                try cudaz.cuLaunchKernel(func, blocks, 1, 1, 256, 1, 1, 0, self.stream, &args, null);
            },
            .int8_asymmetric => {
                const args = .{ &d_raw, &d_scales.?, &d_zeros.?, &out_ptr, @as(c_int, @intCast(num_elements)), @as(c_int, @intCast(block_size)), self.stream };
                try cudaz.cuLaunchKernel(func, blocks, 1, 1, 256, 1, 1, 0, self.stream, &args, null);
            },
            .int4 => {
                const args = .{ &d_raw, &d_scales.?, &d_zeros.?, &out_ptr, @as(c_int, @intCast(num_elements)), @as(c_int, @intCast(block_size)), self.stream };
                try cudaz.cuLaunchKernel(func, blocks, 1, 1, 256, 1, 1, 0, self.stream, &args, null);
            },
            .q4_0, .q8_0 => {
                const args = .{ &d_raw, &out_ptr, @as(c_int, @intCast(num_elements)), self.stream };
                try cudaz.cuLaunchKernel(func, blocks, 1, 1, 256, 1, 1, 0, self.stream, &args, null);
            },
            else => unreachable,
        }

        return out_ptr;
    }
};
