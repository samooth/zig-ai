//! Dequantización en GPU de tensores GGUF (Q4_K, Q6_K, IQ4_XS, IQ3_S).
//! Carga el módulo compilado por el build y lanza los kernels bit-exact vs la
//! referencia CPU en `gguf.zig` (dequantQ4_K / dequantQ6_K / dequantIq4_xs /
//! dequantIq3_s). Salida en f32 para comparación exacta con la CPU.
const std = @import("std");
const cudaz = @import("cudaz");
const build_options = @import("build_options");

pub const GgufDtype = enum {
    q4_k,
    q6_k,
    iq4_xs,
    iq3_s,

    fn kernelName(self: GgufDtype) []const u8 {
        return switch (self) {
            .q4_k => "dequant_q4_k_kernel",
            .q6_k => "dequant_q6_k_kernel",
            .iq4_xs => "dequant_iq4_xs_kernel",
            .iq3_s => "dequant_iq3_s_kernel",
        };
    }

    pub fn blockBytes(self: GgufDtype) usize {
        return switch (self) {
            .q4_k => 144,
            .q6_k => 210,
            .iq4_xs => 136,
            .iq3_s => 110,
        };
    }

    pub fn blockSize(_: GgufDtype) usize {
        return 256;
    }
};

pub const GgufDequantError = error{
    CudaUnavailable,
    KernelNotFound,
};

pub const GgufDequantEngine = struct {
    module: cudaz.CUmodule,
    stream: cudaz.CUstream,

    const Self = @This();

    pub fn init() !Self {
        const ptx_path = build_options.dequant_ptx;
        if (ptx_path.len == 0) return error.CudaUnavailable;
        try cudaz.ensureContext();
        const module = try cudaz.cuModuleLoad(ptx_path);
        errdefer cudaz.cuModuleUnload(module);
        const stream = try cudaz.cuStreamCreate(0);
        errdefer cudaz.cuStreamDestroy(stream);
        return .{ .module = module, .stream = stream };
    }

    pub fn deinit(self: *Self) void {
        cudaz.cuStreamDestroy(self.stream);
        cudaz.cuModuleUnload(self.module);
    }

    /// Dequantiza `bytes` (bloques GGUF del dtype dado) a f32.
    /// `out.len` debe ser `(bytes.len / blockBytes) * blockSize`.
    pub fn dequant(self: *Self, dtype: GgufDtype, bytes: []const u8, out: []f32) !void {
        const num_elements = out.len;
        const bytes_len = bytes.len;

        try cudaz.ensureCurrent();
        var d_raw = try cudaz.cuMemAlloc(bytes_len);
        defer cudaz.cuMemFree(d_raw);
        var d_out = try cudaz.cuMemAlloc(num_elements * @sizeOf(f32));
        defer cudaz.cuMemFree(d_out);

        try cudaz.cuMemcpyHtoD(d_raw, @intFromPtr(bytes.ptr), bytes_len);

        const func = cudaz.cuModuleGetFunction(self.module, dtype.kernelName()) catch return error.KernelNotFound;

        const blocks: c_uint = @intCast((num_elements + 255) / 256);
        var nel: c_int = @intCast(num_elements);
        var kp = [_]?*anyopaque{ &d_raw, &d_out, &nel };
        try cudaz.cuLaunchKernel(func, blocks, 1, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
        try cudaz.cuStreamSynchronize(self.stream);

        try cudaz.cuMemcpyDtoH(@intFromPtr(out.ptr), d_out, num_elements * @sizeOf(f32));
        try cudaz.cuCtxSynchronize();
    }
};
