//! Dequantización en GPU de tensores GGUF (todos los dtypes con kernel CUDA).
//! Los kernels se compilan con nvcc a objetos (kernels/*.cu) y se enlazan al
//! ejecutable vía build.zig (addObjectFile). Se usa la CUDA Runtime API
//! (cuda_runtime.zig) en vez de cargar PTX con la Driver API.
const std = @import("std");
const crt = @import("cuda_runtime");
const gguf = @import("gguf");
const build_options = @import("build_options");

pub const GgufDequantError = error{
    CudaUnavailable,
    KernelNotFound,
    CudaError,
};

// Launchers compilados por nvcc en kernels/*.cu (firma Runtime API).
extern "c" fn dequant_q4_k_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_q6_k_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_iq4_xs_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_iq3_s_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_iq4_nl_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_iq2_xxs_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_iq2_xs_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_iq3_xxs_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_iq1_s_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_iq2_s_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_iq1_m_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_tq1_0_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_tq2_0_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;
extern "c" fn dequant_mxfp4_launcher([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) void;

fn launcherFor(dtype: gguf.GgmlType) ?*const fn ([*c]f32, [*c]const u8, c_int, crt.cudaStream_t) callconv(.c) void {
    return switch (dtype) {
        .q4_k => dequant_q4_k_launcher,
        .q6_k => dequant_q6_k_launcher,
        .iq4_xs => dequant_iq4_xs_launcher,
        .iq3_s => dequant_iq3_s_launcher,
        .iq4_nl => dequant_iq4_nl_launcher,
        .iq2_xxs => dequant_iq2_xxs_launcher,
        .iq2_xs => dequant_iq2_xs_launcher,
        .iq3_xxs => dequant_iq3_xxs_launcher,
        .iq1_s => dequant_iq1_s_launcher,
        .iq2_s => dequant_iq2_s_launcher,
        .iq1_m => dequant_iq1_m_launcher,
        .tq1_0 => dequant_tq1_0_launcher,
        .tq2_0 => dequant_tq2_0_launcher,
        .mxfp4 => dequant_mxfp4_launcher,
        else => null,
    };
}

pub const GgufDequantEngine = struct {
    stream: crt.cudaStream_t,

    const Self = @This();

    pub fn init() !Self {
        if (!build_options.has_cuda) return error.CudaUnavailable;
        try crt.init(0);
        const stream = try crt.streamCreate();
        return .{ .stream = stream };
    }

    pub fn deinit(self: *const Self) void {
        crt.streamDestroy(self.stream);
    }

    /// Dequantiza `bytes` (bloques GGUF del dtype dado) a f32.
    /// `out.len` debe ser `(bytes.len / blockBytes(dtype)) * blockSize(dtype)`.
    pub fn dequant(self: *const Self, dtype: gguf.GgmlType, bytes: []const u8, out: []f32) GgufDequantError!void {
        const launcher = launcherFor(dtype) orelse return error.KernelNotFound;
        if (out.len == 0) return;
        const d_raw = try crt.GpuBuffer.alloc(bytes.len);
        defer d_raw.free();
        const d_out = try crt.GpuBuffer.alloc(out.len * @sizeOf(f32));
        defer d_out.free();

        try d_raw.upload(bytes);
        launcher(@ptrCast(@alignCast(d_out.ptr)), @ptrCast(@alignCast(d_raw.ptr)), @intCast(out.len), self.stream);
        try crt.checkLaunch();
        try crt.streamSync(self.stream);
        try d_out.download(out);
    }
};
