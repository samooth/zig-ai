const std = @import("std");
const cudaz = @import("cudaz");
const fa_config = @import("fa_config.zig");

const FlashAttentionConfig = fa_config.FlashAttentionConfig;

pub const CudaKernel = struct {
    module: cudaz.CUmodule,
    function: cudaz.CUfunction,
    name: []const u8,
    pub fn load(ptx_path: []const u8, kernel_name: []const u8) !CudaKernel {
        const module = try cudaz.cuModuleLoad(ptx_path);
        const function = try cudaz.cuModuleGetFunction(module, kernel_name);
        return .{ .module = module, .function = function, .name = kernel_name };
    }
    pub fn unload(self: *CudaKernel) void { cudaz.cuModuleUnload(self.module); }
};

pub const FlashAttentionBuffers = struct {
    d_q: cudaz.CUdeviceptr, d_k: cudaz.CUdeviceptr,
    d_v: cudaz.CUdeviceptr, d_o: cudaz.CUdeviceptr,
    bytes: usize,
    pub fn alloc(config: FlashAttentionConfig) !FlashAttentionBuffers {
        const bytes = config.total_qkv_bytes();
        return .{
            .d_q = try cudaz.cuMemAlloc(bytes),
            .d_k = try cudaz.cuMemAlloc(bytes),
            .d_v = try cudaz.cuMemAlloc(bytes),
            .d_o = try cudaz.cuMemAlloc(bytes),
            .bytes = bytes,
        };
    }
    pub fn free(self: *FlashAttentionBuffers) void {
        cudaz.cuMemFree(self.d_q); cudaz.cuMemFree(self.d_k);
        cudaz.cuMemFree(self.d_v); cudaz.cuMemFree(self.d_o);
    }
};

pub fn computeSharedMemSize(config: FlashAttentionConfig) usize {
    const bq = config.bq; const bkv = config.bkv; const d = config.d;
    return (bq * d + 2 * bkv * d + bq * bkv) * @sizeOf(f32);
}

pub const GridBlockConfig = struct {
    grid_x: c_uint, grid_y: c_uint, grid_z: c_uint,
    block_x: c_uint, block_y: c_uint, block_z: c_uint,
};

pub fn computeLaunchConfig(config: FlashAttentionConfig) GridBlockConfig {
    return .{
        .grid_x = @intCast(config.batch_size * config.num_heads),
        .grid_y = @intCast((config.N + config.bq - 1) / config.bq),
        .grid_z = 1, .block_x = 256, .block_y = 1, .block_z = 1,
    };
}

pub fn ptxExists(path: []const u8) bool {
    const file = std.fs.cwd().openFile(path, .{}) catch return false;
    file.close(); return true;
}

pub fn getPtxPath(allocator: std.mem.Allocator) ![]const u8 {
    const paths = [_][]const u8{ "cuda/flash_attention.ptx", "flash_attention.ptx", "../cuda/flash_attention.ptx" };
    for (paths) |p| if (ptxExists(p)) return try allocator.dupe(u8, p);
    return error.PtxNotFound;
}

pub fn launchFlashAttentionV1(kernel: cudaz.CUfunction, config: FlashAttentionConfig,
    buffers: FlashAttentionBuffers, stream: cudaz.CUstream) !void {
    const launch_config = computeLaunchConfig(config);
    const smem_size = computeSharedMemSize(config);
    const args = .{
        &buffers.d_q, &buffers.d_k, &buffers.d_v, &buffers.d_o,
        @as(c_int, @intCast(config.N)),
        @as(c_int, @intCast(config.num_heads)),
        config.scale(),
        @as(c_int, @intFromBool(config.causal)),
        @as(c_int, @intCast(config.bq)),
        @as(c_int, @intCast(config.bkv)),
        @as(c_int, @intCast(config.d)),
        stream,
    };
    try cudaz.cuLaunchKernel(kernel, launch_config.grid_x, launch_config.grid_y, launch_config.grid_z,
        launch_config.block_x, launch_config.block_y, launch_config.block_z,
        @intCast(smem_size), stream, &args, null);
}

pub fn launchFlashAttentionV2(kernel: cudaz.CUfunction, config: FlashAttentionConfig,
    buffers: FlashAttentionBuffers, stream: cudaz.CUstream) !void {
    const launch_config = computeLaunchConfig(config);
    const smem_size = computeSharedMemSize(config);
    const args = .{
        &buffers.d_q, &buffers.d_k, &buffers.d_v, &buffers.d_o,
        @as(c_int, @intCast(config.N)),
        @as(c_int, @intCast(config.d)),
        config.scale(),
        @as(c_int, @intFromBool(config.causal)),
    };
    try cudaz.cuLaunchKernel(kernel, launch_config.grid_x, launch_config.grid_y, launch_config.grid_z,
        launch_config.block_x, launch_config.block_y, launch_config.block_z,
        @intCast(smem_size), stream, &args, null);
}
