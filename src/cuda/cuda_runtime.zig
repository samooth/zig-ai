//! Mínimas vinculaciones de la CUDA Runtime API (patrón zig-cuda-agent).
//! Usa `cudaMalloc`/`cudaMemcpy`/`cudaStream_t` y enlaza `cudart` (no Driver API).
//! Los kernels se compilan con nvcc a objetos y se enlazan vía addObjectFile.
const std = @import("std");

pub const cudaStream_t = ?*anyopaque;
pub const cudaError_t = c_int;

pub const cudaMemcpyKind = enum(c_int) {
    host_to_host = 0,
    host_to_device = 1,
    device_to_host = 2,
    device_to_device = 3,
};

extern "c" fn cudaMalloc(dptr: *?*anyopaque, bytesize: usize) cudaError_t;
extern "c" fn cudaFree(dptr: ?*anyopaque) cudaError_t;
extern "c" fn cudaMemcpy(dst: ?*anyopaque, src: ?*anyopaque, count: usize, kind: cudaMemcpyKind) cudaError_t;
extern "c" fn cudaStreamCreate(pstream: *cudaStream_t) cudaError_t;
extern "c" fn cudaStreamDestroy(stream: cudaStream_t) cudaError_t;
extern "c" fn cudaStreamSynchronize(stream: cudaStream_t) cudaError_t;
extern "c" fn cudaGetErrorString(err: cudaError_t) [*:0]const u8;
extern "c" fn cudaGetLastError() cudaError_t;
extern "c" fn cudaSetDevice(device: c_int) cudaError_t;
extern "c" fn cudaDeviceSynchronize() cudaError_t;

pub const CudaRuntimeError = error{CudaError};

fn check(err: cudaError_t) CudaRuntimeError!void {
    if (err == 0) return;
    const msg = cudaGetErrorString(err);
    std.log.err("CUDA error: {s}", .{msg});
    return error.CudaError;
}

pub const GpuBuffer = struct {
    ptr: ?*anyopaque,
    len: usize,

    pub fn alloc(nbytes: usize) CudaRuntimeError!GpuBuffer {
        var p: ?*anyopaque = null;
        try check(cudaMalloc(&p, nbytes));
        return .{ .ptr = p, .len = nbytes };
    }

    pub fn fromSlice(comptime T: type, data: []const T) CudaRuntimeError!GpuBuffer {
        const buf = try GpuBuffer.alloc(data.len * @sizeOf(T));
        try buf.upload(data);
        return buf;
    }

    pub fn upload(self: GpuBuffer, data: anytype) CudaRuntimeError!void {
        const src: [*]const u8 = @ptrCast(@constCast(data.ptr));
        try check(cudaMemcpy(self.ptr, @ptrCast(@constCast(src)), data.len * @sizeOf(@TypeOf(data[0])), .host_to_device));
    }

    pub fn download(self: GpuBuffer, out: anytype) CudaRuntimeError!void {
        const dst: [*]u8 = @ptrCast(out.ptr);
        try check(cudaMemcpy(@ptrCast(dst), self.ptr, out.len * @sizeOf(@TypeOf(out[0])), .device_to_host));
    }

    pub fn free(self: GpuBuffer) void {
        _ = cudaFree(self.ptr);
    }
};

pub fn streamCreate() CudaRuntimeError!cudaStream_t {
    var s: cudaStream_t = null;
    try check(cudaStreamCreate(&s));
    return s;
}

pub fn streamDestroy(s: cudaStream_t) void {
    _ = cudaStreamDestroy(s);
}

pub fn streamSync(s: cudaStream_t) CudaRuntimeError!void {
    try check(cudaStreamSynchronize(s));
}

pub fn init(device: c_int) CudaRuntimeError!void {
    try check(cudaSetDevice(device));
}

pub fn deviceSync() CudaRuntimeError!void {
    try check(cudaDeviceSynchronize());
}

pub fn checkLaunch() CudaRuntimeError!void {
    try check(cudaGetLastError());
}
