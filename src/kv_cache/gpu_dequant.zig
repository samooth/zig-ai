//! Motor de de-cuantización GPU unificado usando CUDA Runtime API
//! Reemplaza el Driver API (PTX) por kernels compilados a objetos (kernels/*.cu)
//! Soporta TODOS los formatos QuantFormat (22+ tipos GGUF + custom int8/int4)

const std = @import("std");
const crt = @import("cuda_runtime");
const qt = @import("quant_types.zig");
const QuantFormat = qt.QuantFormat;
const QuantizedTensor = qt.QuantizedTensor;
const build_options = @import("build_options");

pub const GpuDequantError = error{
    CudaUnavailable,
    KernelNotFound,
    CudaError,
    UnsupportedFormat,
};

// Launchers compilados por nvcc en kernels/*.cu (firma Runtime API)
// Legacy formats
extern "c" fn dequant_q4_0_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_q4_1_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_q5_0_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_q5_1_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_q8_0_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_q8_1_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
// K-quants
extern "c" fn dequant_q2_k_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_q3_k_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_q4_k_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_q5_k_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_q6_k_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_q8_k_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
// I-quants
extern "c" fn dequant_iq1_s_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_iq1_m_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_iq2_xxs_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_iq2_xs_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_iq2_s_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_iq3_xxs_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_iq3_s_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_iq4_xs_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_iq4_nl_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
// T-quants & MXFP4
extern "c" fn dequant_tq1_0_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_tq2_0_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_mxfp4_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
// Custom int8/int4 (5 params: out, in, num_elements, block_size, stream)
extern "c" fn dequant_int8_sym_launcher([*c]f32, [*c]const u8, c_int, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_int8_asym_launcher([*c]f32, [*c]const u8, c_int, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_int4_launcher([*c]f32, [*c]const u8, c_int, c_int, crt.cudaStream_t) void;

// Wrapper functions to match 4-parameter signature
fn dequant_int8_sym_wrapper(out: [*c]f32, in: [*c]const u8, n: c_int, stream: crt.cudaStream_t) callconv(.c) void {
    dequant_int8_sym_launcher(out, in, n, 64, stream);
}

fn dequant_int8_asym_wrapper(out: [*c]f32, in: [*c]const u8, n: c_int, stream: crt.cudaStream_t) callconv(.c) void {
    dequant_int8_asym_launcher(out, in, n, 64, stream);
}

fn dequant_int4_wrapper(out: [*c]f32, in: [*c]const u8, n: c_int, stream: crt.cudaStream_t) callconv(.c) void {
    dequant_int4_launcher(out, in, n, 64, stream);
}

/// Selector de launcher por formato
fn launcherFor(fmt: QuantFormat) ?*const fn ([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) callconv(.c) void {
    return switch (fmt) {
        .q4_0 => dequant_q4_0_launcher,
        .q4_1 => dequant_q4_1_launcher,
        .q5_0 => dequant_q5_0_launcher,
        .q5_1 => dequant_q5_1_launcher,
        .q8_0 => dequant_q8_0_launcher,
        .q8_1 => dequant_q8_1_launcher,
        .q2_k => dequant_q2_k_launcher,
        .q3_k => dequant_q3_k_launcher,
        .q4_k => dequant_q4_k_launcher,
        .q5_k => dequant_q5_k_launcher,
        .q6_k => dequant_q6_k_launcher,
        .q8_k => dequant_q8_k_launcher,
        .iq1_s => dequant_iq1_s_launcher,
        .iq1_m => dequant_iq1_m_launcher,
        .iq2_xxs => dequant_iq2_xxs_launcher,
        .iq2_xs => dequant_iq2_xs_launcher,
        .iq2_s => dequant_iq2_s_launcher,
        .iq3_xxs => dequant_iq3_xxs_launcher,
        .iq3_s => dequant_iq3_s_launcher,
        .iq4_xs => dequant_iq4_xs_launcher,
        .iq4_nl => dequant_iq4_nl_launcher,
        .tq1_0 => dequant_tq1_0_launcher,
        .tq2_0 => dequant_tq2_0_launcher,
        .mxfp4 => dequant_mxfp4_launcher,
        .int8_symmetric => dequant_int8_sym_wrapper,
        .int8_asymmetric => dequant_int8_asym_wrapper,
        .int4 => dequant_int4_wrapper,
        else => null,
    };
}

/// Buffers GPU persistentes para K/V de-cuantizados (fp16)
pub const GpuDequantBuffers = struct {
    d_k_fp16: ?*anyopaque,
    d_v_fp16: ?*anyopaque,
    max_elements: usize,

    pub fn alloc(max_elements: usize) !GpuDequantBuffers {
        const bytes = max_elements * 2; // f16 = 2 bytes
        const d_k = try crt.malloc(bytes);
        const d_v = try crt.malloc(bytes);
        return .{
            .d_k_fp16 = d_k,
            .d_v_fp16 = d_v,
            .max_elements = max_elements,
        };
    }

    pub fn free(self: *GpuDequantBuffers) void {
        if (self.d_k_fp16) |ptr| crt.free(ptr);
        if (self.d_v_fp16) |ptr| crt.free(ptr);
    }

    pub fn ensureSize(self: *GpuDequantBuffers, elements: usize) !void {
        if (elements > self.max_elements) {
            const new_bytes = elements * 2;
            if (self.d_k_fp16) |ptr| crt.free(ptr);
            if (self.d_v_fp16) |ptr| crt.free(ptr);
            self.d_k_fp16 = try crt.malloc(new_bytes);
            self.d_v_fp16 = try crt.malloc(new_bytes);
            self.max_elements = elements;
        }
    }
};

/// Motor unificado de de-cuantización GPU
pub const GpuDequantEngine = struct {
    allocator: std.mem.Allocator,
    stream: crt.cudaStream_t,
    buffers: GpuDequantBuffers,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, max_elements: usize) !Self {
        if (!build_options.has_cuda) return error.CudaUnavailable;
        try crt.init(0);
        const stream = try crt.streamCreate();
        const buffers = try GpuDequantBuffers.alloc(max_elements);
        return .{ .allocator = allocator, .stream = stream, .buffers = buffers };
    }

    pub fn deinit(self: *Self) void {
        self.buffers.free();
        crt.streamDestroy(self.stream);
    }

    /// De-cuantiza datos cuantizados en device a f32
    /// d_raw: device ptr a datos cuantizados
    /// d_scales: device ptr a escalas (puede ser null)
    /// d_zeros: device ptr a zero points (puede ser null)
    /// num_elements: número de valores lógicos
    /// block_size: tamaño de bloque usado (para int8/int4 custom)
    pub fn dequantize(
        self: *Self,
        format: QuantFormat,
        d_raw: crt.cudaDeviceptr,
        _d_scales: ?crt.cudaDeviceptr,
        _d_zeros: ?crt.cudaDeviceptr,
        num_elements: usize,
        block_size: usize,
    ) GpuDequantError!crt.cudaDeviceptr {
        _ = _d_scales;
        _ = _d_zeros;
        if (num_elements == 0) return self.buffers.d_k_fp16.?;

        try self.buffers.ensureSize(num_elements);

        const d_out = self.buffers.d_k_fp16.?;

        // Launch with appropriate arguments based on format
        if (format.hasZeroPoints()) {
            // int8_asymmetric, int4 need scale + zero_point + block_size
            self.launchDequantWithZeros(format, d_raw, d_out, num_elements, block_size);
        } else if (format.hasScales()) {
            // formats with scales but no zero_point
            try self.launchDequantWithScales(format, d_raw, d_out, num_elements);
        } else {
            // fp16, fp32 - just copy
            try crt.memcpyAsync(d_out, @ptrCast(d_raw), num_elements * 2, crt.cudaMemcpyKind.device_to_device, self.stream);
        }

        try crt.streamSync(self.stream);
        return d_out;
    }

    /// De-cuantiza K y V de un bloque del cache
    pub fn dequantizeBlock(
        self: *Self,
        k_format: QuantFormat,
        d_k_raw: crt.cudaDeviceptr,
        d_k_scales: ?crt.cudaDeviceptr,
        d_k_zeros: ?crt.cudaDeviceptr,
        v_format: QuantFormat,
        d_v_raw: crt.cudaDeviceptr,
        d_v_scales: ?crt.cudaDeviceptr,
        d_v_zeros: ?crt.cudaDeviceptr,
        num_elements: usize,
        k_block_size: usize,
        v_block_size: usize,
    ) GpuDequantError!struct { d_k: crt.cudaDeviceptr, d_v: crt.cudaDeviceptr } {
        try self.buffers.ensureSize(num_elements * 2); // K + V

        const d_k = try self.dequantizeInternal(k_format, d_k_raw, d_k_scales, d_k_zeros, num_elements, k_block_size);
        const d_v = try self.dequantizeInternal(v_format, d_v_raw, d_v_scales, d_v_zeros, num_elements, v_block_size);

        return .{ .d_k = d_k, .d_v = d_v };
    }

    fn dequantizeInternal(
        self: *Self,
        format: QuantFormat,
        d_raw: crt.cudaDeviceptr,
        d_scales: ?crt.cudaDeviceptr,
        d_zeros: ?crt.cudaDeviceptr,
        num_elements: usize,
        block_size: usize,
    ) GpuDequantError!crt.cudaDeviceptr {
        if (num_elements == 0) return crt.cudaDeviceptr(0);

        const d_out = self.buffers.d_k_fp16.?; // reuse buffer (caller must sequence)

        if (format.hasZeroPoints()) {
            self.launchDequantWithZeros(format, d_raw, d_scales.?, d_zeros.?, d_out, num_elements, block_size);
        } else if (format.hasScales()) {
            self.launchDequantWithScales(format, d_raw, d_scales.?, d_out, num_elements);
        } else {
            try crt.memcpyAsync(d_out, d_raw, num_elements * 2, crt.cudaMemcpyKind.DeviceToDevice, self.stream);
        }

        try crt.streamSync(self.stream);
        return d_out;
    }

    /// Launch dequant kernel with zero points (int8_asymmetric, int4)
    fn launchDequantWithZeros(
        self: *Self,
        format: QuantFormat,
        d_raw: crt.cudaDeviceptr,
        d_out: crt.cudaDeviceptr,
        num_elements: usize,
        _: usize,
    ) void {
        switch (format) {
            .int8_asymmetric => {
                dequant_int8_asym_launcher(
                    @ptrCast(@alignCast(d_out)),
                    @ptrCast(@alignCast(@constCast(d_raw))),
                    @intCast(num_elements),
                    @intCast(64), // block_size is always 64 for int8_asymmetric
                    self.stream,
                );
            },
            .int4 => {
                dequant_int4_launcher(
                    @ptrCast(@alignCast(d_out)),
                    @ptrCast(@alignCast(@constCast(d_raw))),
                    @intCast(num_elements),
                    @intCast(64), // block_size is always 64 for int4
                    self.stream,
                );
            },
            else => unreachable,
        }
    }

    /// Launch dequant kernel with scales (most formats)
    fn launchDequantWithScales(
        self: *Self,
        format: QuantFormat,
        d_raw: crt.cudaDeviceptr,
        d_out: crt.cudaDeviceptr,
        num_elements: usize,
    ) !void {
        switch (format) {
            .q4_0 => dequant_q4_0_launcher(@ptrCast(@alignCast(d_out)), @ptrCast(@alignCast(@constCast(d_raw))), @intCast(num_elements), self.stream),
            .q4_1 => dequant_q4_1_launcher(@ptrCast(@alignCast(d_out)), @ptrCast(@alignCast(@constCast(d_raw))), @intCast(num_elements), self.stream),
            .q5_0 => dequant_q5_0_launcher(@ptrCast(@alignCast(d_out)), @ptrCast(@alignCast(@constCast(d_raw))), @intCast(num_elements), self.stream),
            .q5_1 => dequant_q5_1_launcher(@ptrCast(@alignCast(d_out)), @ptrCast(@alignCast(@constCast(d_raw))), @intCast(num_elements), self.stream),
            .q8_0 => dequant_q8_0_launcher(@ptrCast(@alignCast(d_out)), @ptrCast(@alignCast(@constCast(d_raw))), @intCast(num_elements), self.stream),
            .q8_1 => dequant_q8_1_launcher(@ptrCast(@alignCast(d_out)), @ptrCast(@alignCast(@constCast(d_raw))), @intCast(num_elements), self.stream),
            .q2_k => dequant_q2_k_launcher(@ptrCast(@alignCast(d_out)), @ptrCast(@alignCast(@constCast(d_raw))), @intCast(num_elements), self.stream),
            .q3_k => dequant_q3_k_launcher(@ptrCast(@alignCast(d_out)), @ptrCast(@alignCast(@constCast(d_raw))), @intCast(num_elements), self.stream),
            .q4_k => dequant_q4_k_launcher(@ptrCast(@alignCast(d_out)), @ptrCast(@alignCast(@constCast(d_raw))), @intCast(num_elements), self.stream),
            .q5_k => dequant_q5_k_launcher(@ptrCast(@alignCast(d_out)), @ptrCast(@alignCast(@constCast(d_raw))), @intCast(num_elements), self.stream),
            .q6_k => dequant_q6_k_launcher(@ptrCast(@alignCast(d_out)), @ptrCast(@alignCast(@constCast(d_raw))), @intCast(num_elements), self.stream),
            .q8_k => dequant_q8_k_launcher(@ptrCast(@alignCast(d_out)), @ptrCast(@alignCast(@constCast(d_raw))), @intCast(num_elements), self.stream),
            .iq1_s => dequant_iq1_s_launcher(@ptrCast(@alignCast(d_out)), @ptrCast(@alignCast(@constCast(d_raw))), @intCast(num_elements), self.stream),
            .iq1_m => dequant_iq1_m_launcher(@ptrCast(@alignCast(d_out)), @ptrCast(@alignCast(@constCast(d_raw))), @intCast(num_elements), self.stream),
            .iq2_xxs => dequant_iq2_xxs_launcher(@ptrCast(@alignCast(d_out)), @ptrCast(@alignCast(@constCast(d_raw))), @intCast(num_elements), self.stream),
            .iq2_xs => dequant_iq2_xs_launcher(@ptrCast(@alignCast(d_out)), @ptrCast(@alignCast(@constCast(d_raw))), @intCast(num_elements), self.stream),
            .iq2_s => dequant_iq2_s_launcher(@ptrCast(@alignCast(d_out)), @ptrCast(@alignCast(@constCast(d_raw))), @intCast(num_elements), self.stream),
            .iq3_xxs => dequant_iq3_xxs_launcher(@ptrCast(@alignCast(d_out)), @ptrCast(@alignCast(@constCast(d_raw))), @intCast(num_elements), self.stream),
            .iq3_s => dequant_iq3_s_launcher(@ptrCast(@alignCast(d_out)), @ptrCast(@alignCast(@constCast(d_raw))), @intCast(num_elements), self.stream),
            .iq4_xs => dequant_iq4_xs_launcher(@ptrCast(@alignCast(d_out)), @ptrCast(@alignCast(@constCast(d_raw))), @intCast(num_elements), self.stream),
            .iq4_nl => dequant_iq4_nl_launcher(@ptrCast(@alignCast(d_out)), @ptrCast(@alignCast(@constCast(d_raw))), @intCast(num_elements), self.stream),
            .tq1_0 => dequant_tq1_0_launcher(@ptrCast(@alignCast(d_out)), @ptrCast(@alignCast(@constCast(d_raw))), @intCast(num_elements), self.stream),
            .tq2_0 => dequant_tq2_0_launcher(@ptrCast(@alignCast(d_out)), @ptrCast(@alignCast(@constCast(d_raw))), @intCast(num_elements), self.stream),
            .mxfp4 => dequant_mxfp4_launcher(@ptrCast(@alignCast(d_out)), @ptrCast(@alignCast(@constCast(d_raw))), @intCast(num_elements), self.stream),
            .int8_symmetric => dequant_int8_sym_launcher(@ptrCast(@alignCast(d_out)), @ptrCast(@alignCast(@constCast(d_raw))), @intCast(num_elements), @intCast(64), self.stream),
            .fp16, .fp32 => {
                try crt.memcpyAsync(d_out, d_raw, @intCast(num_elements * 2), crt.cudaMemcpyKind.device_to_device, self.stream);
            },
            else => unreachable,
        }
    }
};

test {
    std.testing.refAllDecls(@This());
}
