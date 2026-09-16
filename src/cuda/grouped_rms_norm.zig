//! Wrapper Zig para el kernel CUDA GroupedRMSNorm (K2-Horizon).
//!
//! Oráculo CPU: `cpuGroupedRmsNormFlat` en `src/inference/cli.zig`.
//!
//! Firma: input[N, hidden_dim], gamma[hidden_dim] → output[N, hidden_dim]
//! n_groups particiona hidden_dim; cada grupo se normaliza independientemente.

const std = @import("std");
const cudaz = @import("cudaz");

pub const GroupedRmsNormError = error{
    CudaError,
    InvalidShape,
    OutOfMemory,
};

/// Contexto persistente para GroupedRMSNorm GPU.
/// Cachea device buffers y el módulo/cubín para evitar re-cuMemAlloc por llamada.
pub const Context = struct {
    stream: cudaz.CUstream,
    module: ?cudaz.CUmodule = null,
    func: ?cudaz.CUfunction = null,
    d_input: cudaz.CUdeviceptr = 0,
    d_gamma: cudaz.CUdeviceptr = 0,
    d_output: cudaz.CUdeviceptr = 0,
    cap_n: usize = 0,
    cap_hidden: usize = 0,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, stream: cudaz.CUstream) !Self {
        return .{ .allocator = allocator, .stream = stream };
    }

    pub fn deinit(self: *Self) void {
        if (self.d_input != 0) cudaz.cuMemFree(self.d_input);
        if (self.d_gamma != 0) cudaz.cuMemFree(self.d_gamma);
        if (self.d_output != 0) cudaz.cuMemFree(self.d_output);
        if (self.module) |m| cudaz.cuModuleUnload(m);
        self.* = .{ .allocator = self.allocator, .stream = self.stream };
    }

    fn ensureBuffers(self: *Self, N: usize, hidden_dim: usize) !void {
        const bytes = N * hidden_dim * @sizeOf(f32);
        const gamma_bytes = hidden_dim * @sizeOf(f32);
        if (self.cap_n < N or self.cap_hidden < hidden_dim) {
            if (self.d_input != 0) cudaz.cuMemFree(self.d_input);
            if (self.d_gamma != 0) cudaz.cuMemFree(self.d_gamma);
            if (self.d_output != 0) cudaz.cuMemFree(self.d_output);
            self.d_input = try cudaz.cuMemAlloc(bytes);
            self.d_gamma = try cudaz.cuMemAlloc(gamma_bytes);
            self.d_output = try cudaz.cuMemAlloc(bytes);
            self.cap_n = N;
            self.cap_hidden = hidden_dim;
        }
    }

    fn ensureModule(self: *Self) !void {
        if (self.module != null) return;
        const mod_path = @import("build_options").grouped_rms_norm_cubin;
        if (mod_path.len == 0) return error.CudaError;
        self.module = try cudaz.cuModuleLoad(mod_path);
        self.func = try cudaz.cuModuleGetFunction(self.module.?, "groupedRmsNormKernel");
    }

    /// Ejecuta grouped RMSNorm con datos host (H2D → kernel → D2H).
    pub fn runHost(
        self: *Self,
        input: []const f32,
        gamma: []const f32,
        output: []f32,
        n_groups: usize,
        eps: f32,
    ) GroupedRmsNormError!void {
        const N = input.len / gamma.len;
        const hidden_dim = gamma.len;
        if (hidden_dim % n_groups != 0) return error.InvalidShape;
        if (input.len != N * hidden_dim) return error.InvalidShape;
        if (output.len != N * hidden_dim) return error.InvalidShape;

        try self.ensureBuffers(N, hidden_dim);
        try self.ensureModule();

        try cudaz.cuMemcpyHtoD(self.d_input, @intFromPtr(input.ptr), input.len * @sizeOf(f32));
        try cudaz.cuMemcpyHtoD(self.d_gamma, @intFromPtr(gamma.ptr), gamma.len * @sizeOf(f32));

        const group_dim = hidden_dim / n_groups;
        if (group_dim > 1024) return error.InvalidShape;

        var N_c: c_int = @intCast(N);
        var hd_c: c_int = @intCast(hidden_dim);
        var ng_c: c_int = @intCast(n_groups);
        var eps_c: f32 = eps;

        var kp = [_]?*anyopaque{
            @ptrFromInt(self.d_input),
            @ptrFromInt(self.d_gamma),
            @ptrFromInt(self.d_output),
            &N_c,
            &hd_c,
            &ng_c,
            &eps_c,
        };

        try cudaz.cuLaunchKernel(self.func.?, @intCast(N), @intCast(n_groups), 1, @intCast(group_dim), 1, 1, 0, self.stream, @ptrCast(&kp), null);

        try cudaz.cuMemcpyDtoH(@intFromPtr(output.ptr), self.d_output, output.len * @sizeOf(f32));
    }
};
