// FP8 Block-Scaled Linear Zig Wrapper
// Launches FP8 kernels via CUDA driver API (same pattern as layer_kernels.zig)

const std = @import("std");
const cudaz = @import("cudaz");
const build_options = @import("build_options");
const debugz = @import("debug");

var g_module: ?cudaz.CUmodule = null;

const kernel_names = [_][:0]const u8{
    "per_token_group_quant_fp8_kernel",
    "per_token_group_quant_fp8_f32_kernel",
    "block_fp8_gemm_kernel",
    "block_fp8_gemv_splitk_kernel",
};
var g_funcs: [kernel_names.len]?cudaz.CUfunction = .{null} ** kernel_names.len;

fn loadModule() !cudaz.CUmodule {
    if (g_module) |m| return m;
    const cubin_path = build_options.fp8_cubin;
    if (cubin_path.len == 0) return error.CudaUnavailable;
    try cudaz.ensureContext();
    g_module = try cudaz.cuModuleLoad(cubin_path);
    if (debugz.dbg.dump_graph) dumpFuncs();
    return g_module.?;
}

fn dumpFuncs() void {
    const mod = g_module.?;
    for (kernel_names) |kn| {
        const f = cudaz.cuModuleGetFunction(mod, kn) catch continue;
        debugz.dbg.print("[graph] DUMP_GRAPH func {x} = {s}\n", .{ @intFromPtr(f), kn });
    }
}

fn getFuncByName(name: [:0]const u8) !cudaz.CUfunction {
    _ = try loadModule();
    for (kernel_names, 0..) |kn, i| {
        if (std.mem.eql(u8, kn, name)) {
            if (g_funcs[i]) |f| return f;
            g_funcs[i] = try cudaz.cuModuleGetFunction(g_module.?, name);
            return g_funcs[i].?;
        }
    }
    return error.InvalidArgument;
}

fn n_u(v: usize) c_uint {
    return @as(c_uint, @intCast(v));
}

/// FP8 Block-Scaled Linear engine
pub const Fp8BlockLinear = struct {
    stream: cudaz.CUstream,

    pub fn init(stream: cudaz.CUstream) !Fp8BlockLinear {
        _ = try loadModule();
        return .{ .stream = stream };
    }

    /// Quantize activations to FP8 with per-token, per-128-group scales
    /// input: [M, K] bf16 (device)
    /// output_fp8: [M, K] fp8 (device, preallocated)
    /// output_scales: [M, K/128] fp32 (device, preallocated)
    pub fn quantizeActivations(
        self: *Fp8BlockLinear,
        input: cudaz.CUdeviceptr,
        output_fp8: cudaz.CUdeviceptr,
        output_scales: cudaz.CUdeviceptr,
        M: i32,
        K: i32,
    ) !void {
        const group_size = 128;
        const num_groups = @divTrunc(K, group_size);
        const func = try getFuncByName("per_token_group_quant_fp8_kernel");
        const grid_x = n_u(@as(usize, @intCast(M)));
        const grid_y = n_u(@as(usize, @intCast(num_groups)));
        const block_x = 256;
        // FIX 2.1 (lane-f): el kernel firma (input, output, scales, M, K) —
        // el wrapper pasaba SOLO 4 args (K quedaba garbage en el kernel).
        var in_v = input;
        var ofp_v = output_fp8;
        var osc_v = output_scales;
        var m1: i32 = M;
        var k1: i32 = K;
        var kp = [_]?*anyopaque{ &in_v, &ofp_v, &osc_v, &m1, &k1 };
        try cudaz.cuLaunchKernel(func, grid_x, grid_y, 1, block_x, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    /// 2.1 (lane-f): variante f32 — las activaciones del engine son f32
    /// (el original lee half*). Misma matemática (amax/448, clamp ±448).
    /// Con M=N cuantiza PESOS [N,K] (fila="token") — el missing piece del
    /// gemmFp8Block: nadie producía W_fp8/W_scales.
    pub fn quantizeActivationsF32(
        self: *Fp8BlockLinear,
        input: cudaz.CUdeviceptr,
        output_fp8: cudaz.CUdeviceptr,
        output_scales: cudaz.CUdeviceptr,
        M: i32,
        K: i32,
    ) !void {
        const group_size = 128;
        const num_groups = @divTrunc(K, group_size);
        const func = try getFuncByName("per_token_group_quant_fp8_f32_kernel");
        const grid_x = n_u(@as(usize, @intCast(M)));
        const grid_y = n_u(@as(usize, @intCast(num_groups)));
        var in_v = input;
        var ofp_v = output_fp8;
        var osc_v = output_scales;
        var m1: i32 = M;
        var k1: i32 = K;
        var kp = [_]?*anyopaque{ &in_v, &ofp_v, &osc_v, &m1, &k1 };
        try cudaz.cuLaunchKernel(func, grid_x, grid_y, 1, 256, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    /// Block-scaled FP8 GEMM: C = A @ B^T
    /// A: [M, K] fp8, a_scales: [M, K/128] fp32
    /// B: [N, K] fp8, b_scales: [N, K/128] fp32
    /// C: [M, N] fp32 (accumulated)
    pub fn gemm(
        self: *Fp8BlockLinear,
        A: cudaz.CUdeviceptr,
        a_scales: cudaz.CUdeviceptr,
        B: cudaz.CUdeviceptr,
        b_scales: cudaz.CUdeviceptr,
        C: cudaz.CUdeviceptr,
        M: i32,
        N: i32,
        K: i32,
    ) !void {
        const func = try getFuncByName("block_fp8_gemm_kernel");
        const grid_x = n_u(@as(usize, @intCast(@divTrunc(N + 63, 64))));
        const grid_y = n_u(@as(usize, @intCast(@divTrunc(M + 63, 64))));
        const block_x = 16;
        const block_y = 16;
        var a_v = A;
        var asc_v = a_scales;
        var b_v = B;
        var bsc_v = b_scales;
        var c_v = C;
        var m1: i32 = M;
        var n1: i32 = N;
        var k1: i32 = K;
        var kp = [_]?*anyopaque{ &a_v, &asc_v, &b_v, &bsc_v, &c_v, &m1, &n1, &k1 };
        try cudaz.cuLaunchKernel(func, grid_x, grid_y, 1, block_x, block_y, 1, 0, self.stream, @ptrCast(&kp), null);
    }

    /// Split-K FP8 GEMV for decode (M=1)
    /// x: [K] fp8, x_scale: [K/128] fp32
    /// w: [N, K] fp8, w_scales: [N, K/128] fp32
    /// out: [N] fp32
    pub fn gemvSplitK(
        self: *Fp8BlockLinear,
        x: cudaz.CUdeviceptr,
        x_scale: cudaz.CUdeviceptr,
        w: cudaz.CUdeviceptr,
        w_scales: cudaz.CUdeviceptr,
        out: cudaz.CUdeviceptr,
        N: i32,
        K: i32,
        num_splits: i32,
    ) !void {
        const func = try getFuncByName("block_fp8_gemv_splitk_kernel");
        const grid_x = n_u(@as(usize, @intCast(@divTrunc(N + 255, 256))));
        const grid_y = n_u(@as(usize, @intCast(num_splits)));
        const block_x = 256;
        var x_v = x;
        var xs_v = x_scale;
        var w_v = w;
        var ws_v = w_scales;
        var o_v = out;
        var n1: i32 = N;
        var k1: i32 = K;
        var ns1: i32 = num_splits;
        var kp = [_]?*anyopaque{ &x_v, &xs_v, &w_v, &ws_v, &o_v, &n1, &k1, &ns1 };
        try cudaz.cuLaunchKernel(func, grid_x, grid_y, 1, block_x, 1, 1, 0, self.stream, @ptrCast(&kp), null);
    }
};