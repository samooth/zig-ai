const std = @import("std");

pub const CUresult = enum(c_int) {
    SUCCESS = 0, ERROR_INVALID_VALUE = 1, ERROR_OUT_OF_MEMORY = 2,
    ERROR_NOT_INITIALIZED = 3, ERROR_DEINITIALIZED = 4,
    ERROR_NO_DEVICE = 100, ERROR_INVALID_DEVICE = 101,
    ERROR_INVALID_IMAGE = 200, ERROR_INVALID_CONTEXT = 201,
    ERROR_CONTEXT_ALREADY_CURRENT = 202, ERROR_MAP_FAILED = 205,
    ERROR_UNMAP_FAILED = 206, ERROR_ARRAY_IS_MAPPED = 207,
    ERROR_ALREADY_MAPPED = 208, ERROR_NO_BINARY_FOR_GPU = 209,
    ERROR_ALREADY_ACQUIRED = 210, ERROR_NOT_MAPPED = 211,
    ERROR_NOT_MAPPED_AS_ARRAY = 212, ERROR_NOT_MAPPED_AS_POINTER = 213,
    ERROR_ECC_UNCORRECTABLE = 214, ERROR_UNSUPPORTED_LIMIT = 215,
    ERROR_CONTEXT_ALREADY_IN_USE = 216, ERROR_PEER_ACCESS_UNSUPPORTED = 217,
    ERROR_INVALID_SOURCE = 300, ERROR_FILE_NOT_FOUND = 301,
    ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302, ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    ERROR_OPERATING_SYSTEM = 304, ERROR_INVALID_HANDLE = 400,
    ERROR_NOT_FOUND = 500, ERROR_NOT_READY = 600,
    ERROR_LAUNCH_FAILED = 700, ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    ERROR_LAUNCH_TIMEOUT = 702, ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
    ERROR_UNKNOWN = 999,
};

pub const CUdevice = c_int;
pub const CUcontext = *opaque {};
pub const CUmodule = *opaque {};
pub const CUfunction = *opaque {};
pub const CUstream = *opaque {};
pub const CUdeviceptr = usize;
pub const CUevent = *opaque {};

pub fn cuInit(flags: c_uint) !void {
    const res = cudalib.cuInit(flags);
    if (res != .SUCCESS) return error.CudaError;
}

pub fn cuDeviceGet(ordinal: c_int) !CUdevice {
    var device: CUdevice = undefined;
    const res = cudalib.cuDeviceGet(&device, ordinal);
    if (res != .SUCCESS) return error.CudaError;
    return device;
}

pub fn cuCtxCreate(flags: c_uint, dev: CUdevice) !CUcontext {
    var ctx: CUcontext = undefined;
    const res = cudalib.cuCtxCreate(&ctx, flags, dev);
    if (res != .SUCCESS) return error.CudaError;
    return ctx;
}

pub fn cuCtxDestroy(ctx: CUcontext) void { _ = cudalib.cuCtxDestroy(ctx); }

pub fn cuModuleLoad(path: []const u8) !CUmodule {
    var module: CUmodule = undefined;
    const c_path = try std.heap.c_allocator.dupeZ(u8, path);
    defer std.heap.c_allocator.free(c_path);
    const res = cudalib.cuModuleLoad(&module, c_path.ptr);
    if (res != .SUCCESS) return error.CudaError;
    return module;
}

pub fn cuModuleUnload(module: CUmodule) void { _ = cudalib.cuModuleUnload(module); }

pub fn cuModuleGetFunction(module: CUmodule, name: []const u8) !CUfunction {
    var func: CUfunction = undefined;
    const c_name = try std.heap.c_allocator.dupeZ(u8, name);
    defer std.heap.c_allocator.free(c_name);
    const res = cudalib.cuModuleGetFunction(&func, module, c_name.ptr);
    if (res != .SUCCESS) return error.CudaError;
    return func;
}

pub fn cuMemAlloc(bytes: usize) !CUdeviceptr {
    var dptr: CUdeviceptr = 0;
    const res = cudalib.cuMemAlloc(&dptr, bytes);
    if (res != .SUCCESS) return error.CudaError;
    return dptr;
}

pub fn cuMemFree(dptr: CUdeviceptr) void { _ = cudalib.cuMemFree(dptr); }

pub fn cuMemAllocHost(bytes: usize) !*anyopaque {
    var ptr: ?*anyopaque = null;
    const res = cudalib.cuMemAllocHost(&ptr, bytes);
    if (res != .SUCCESS) return error.CudaError;
    return ptr orelse return error.CudaError;
}

pub fn cuMemFreeHost(ptr: *anyopaque) void { _ = cudalib.cuMemFreeHost(ptr); }

pub fn cuMemcpyHtoDAsync(dst: CUdeviceptr, src: usize, bytes: usize, stream: CUstream) !void {
    const res = cudalib.cuMemcpyHtoDAsync(dst, @ptrFromInt(src), bytes, stream);
    if (res != .SUCCESS) return error.CudaError;
}

pub fn cuMemcpyDtoHAsync(dst: usize, src: CUdeviceptr, bytes: usize, stream: CUstream) !void {
    const res = cudalib.cuMemcpyDtoHAsync(@ptrFromInt(dst), src, bytes, stream);
    if (res != .SUCCESS) return error.CudaError;
}

/// Copias síncronas (sin stream) para evitar carreras async.
pub fn cuMemcpyHtoD(dst: CUdeviceptr, src: usize, bytes: usize) !void {
    const res = cudalib.cuMemcpyHtoD(dst, @ptrFromInt(src), bytes);
    if (res != .SUCCESS) return error.CudaError;
}

pub fn cuMemcpyDtoH(dst: usize, src: CUdeviceptr, bytes: usize) !void {
    const res = cudalib.cuMemcpyDtoH(@ptrFromInt(dst), src, bytes);
    if (res != .SUCCESS) return error.CudaError;
}

/// Sincroniza el device (contexto actual).
pub fn cuCtxSynchronize() !void {
    const res = cudalib.cuCtxSynchronize();
    if (res != .SUCCESS) return error.CudaError;
}

pub fn cuStreamCreate(flags: c_uint) !CUstream {
    var stream: CUstream = undefined;
    const res = cudalib.cuStreamCreate(&stream, flags);
    if (res != .SUCCESS) return error.CudaError;
    return stream;
}

pub fn cuStreamDestroy(stream: CUstream) void { _ = cudalib.cuStreamDestroy(stream); }

pub fn cuStreamSynchronize(stream: CUstream) !void {
    const res = cudalib.cuStreamSynchronize(stream);
    if (res != .SUCCESS) return error.CudaError;
}

pub fn cuLaunchKernel(
    f: CUfunction, gridDimX: c_uint, gridDimY: c_uint, gridDimZ: c_uint,
    blockDimX: c_uint, blockDimY: c_uint, blockDimZ: c_uint,
    sharedMemBytes: c_uint, hStream: CUstream, kernelParams: ?*anyopaque, extra: ?*anyopaque,
) !void {
    const res = cudalib.cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
    if (res != .SUCCESS) return error.CudaError;
}

/// Bindings reales a la CUDA Driver API (enlazados vía -lcuda).
/// Las funciones son los símbolos estándar exportados por libcuda.so.1
/// (los nombres sin sufijo `_v2` siguen exportados por el driver).
const cudalib = struct {
    extern "c" fn cuInit(flags: c_uint) CUresult;
    extern "c" fn cuDeviceGet(device: *CUdevice, ordinal: c_int) CUresult;
    extern "c" fn cuCtxCreate(ctx: *CUcontext, flags: c_uint, dev: CUdevice) CUresult;
    extern "c" fn cuCtxDestroy(ctx: CUcontext) CUresult;
    extern "c" fn cuDevicePrimaryCtxRetain(ctx: *CUcontext, dev: CUdevice) CUresult;
    extern "c" fn cuCtxSetCurrent(ctx: CUcontext) CUresult;
    extern "c" fn cuCtxGetCurrent(ctx: *CUcontext) CUresult;
    extern "c" fn cuModuleLoad(module: *CUmodule, fname: [*:0]const u8) CUresult;
    extern "c" fn cuModuleUnload(module: CUmodule) CUresult;
    extern "c" fn cuModuleGetFunction(hfunc: *CUfunction, hmod: CUmodule, name: [*:0]const u8) CUresult;
    extern "c" fn cuMemAlloc(dptr: *CUdeviceptr, bytesize: usize) CUresult;
    extern "c" fn cuMemFree(dptr: CUdeviceptr) CUresult;
    extern "c" fn cuMemAllocHost(pp: *?*anyopaque, bytesize: usize) CUresult;
    extern "c" fn cuMemFreeHost(p: *anyopaque) CUresult;
    extern "c" fn cuMemcpyHtoDAsync(dst: CUdeviceptr, src: *const anyopaque, bytes: usize, stream: CUstream) CUresult;
    extern "c" fn cuMemcpyDtoHAsync(dst: *anyopaque, src: CUdeviceptr, bytes: usize, stream: CUstream) CUresult;
    extern "c" fn cuMemcpyHtoD(dst: CUdeviceptr, src: *const anyopaque, bytes: usize) CUresult;
    extern "c" fn cuMemcpyDtoH(dst: *anyopaque, src: CUdeviceptr, bytes: usize) CUresult;
    extern "c" fn cuCtxSynchronize() CUresult;
    extern "c" fn cuStreamCreate(phStream: *CUstream, flags: c_uint) CUresult;
    extern "c" fn cuStreamDestroy(hStream: CUstream) CUresult;
    extern "c" fn cuStreamSynchronize(hStream: CUstream) CUresult;
    extern "c" fn cuLaunchKernel(f: CUfunction, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, sm: c_uint, stream: CUstream, params: ?*anyopaque, extra: ?*anyopaque) CUresult;
    extern "c" fn cuDeviceGetName(name: [*]u8, len: c_int, dev: CUdevice) CUresult;
    extern "c" fn cuDeviceTotalMem(mem: *usize, dev: CUdevice) CUresult;
};

/// Contexto CUDA creado por `ensureContext()`.
var g_cuda_ctx: ?CUcontext = null;

/// Inicializa CUDA (cuInit + device 0) y asegura un contexto actual
/// (primary context retenido y seteado como current).
/// Necesario antes de usar cuBLAS o cualquier kernel.
pub fn ensureContext() !void {
    if (g_cuda_ctx != null) return;
    try cuInit(0);
    const device = try cuDeviceGet(0);
    var ctx: CUcontext = undefined;
    if (cudalib.cuCtxCreate(&ctx, 0, device) != .SUCCESS) return error.CudaError;
    if (cudalib.cuCtxSetCurrent(ctx) != .SUCCESS) return error.CudaError;
    g_cuda_ctx = ctx;
}

/// Re-asegura que el contexto CUDA está current en este thread.
/// Algunas llamadas (p. ej. cuModuleLoad) pueden alterar el contexto actual.
pub fn ensureCurrent() !void {
    try ensureContext();
    if (cudalib.cuCtxSetCurrent(g_cuda_ctx.?) != .SUCCESS) return error.CudaError;
}

/// Contexto CUDA actual (null si no hay).
pub fn currentContext() ?CUcontext {
    var ctx: CUcontext = undefined;
    if (cudalib.cuCtxGetCurrent(&ctx) != .SUCCESS) return null;
    return if (ctx) |c| c else null;
}

pub const Dim3 = extern struct { x: c_uint = 1, y: c_uint = 1, z: c_uint = 1 };

pub fn isCudaAvailable() bool {
    _ = cuInit(0) catch return false;
    _ = cuDeviceGet(0) catch return false;
    return true;
}

pub fn getDeviceName(dev: CUdevice, allocator: std.mem.Allocator) ![]u8 {
    var name_buf: [256]u8 = undefined;
    const res = cudalib.cuDeviceGetName(&name_buf, name_buf.len, dev);
    if (res != .SUCCESS) return error.CudaError;
    const len = std.mem.indexOfScalar(u8, &name_buf, 0) orelse name_buf.len;
    return try allocator.dupe(u8, name_buf[0..len]);
}

pub fn getDeviceTotalMem(dev: CUdevice) !usize {
    var mem: usize = undefined;
    const res = cudalib.cuDeviceTotalMem(&mem, dev);
    if (res != .SUCCESS) return error.CudaError;
    return mem;
}
