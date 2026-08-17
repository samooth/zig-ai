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
pub const CUmemGenericAllocationHandle = u64;

pub const CUmemAllocationType = enum(c_int) {
    INVALID = 0,
    PINNED = 1,
};

pub const CUmemAllocationHandleType = enum(c_int) {
    NONE = 0,
    POSIX_FILE_DESCRIPTOR = 1,
    WIN32 = 2,
    WIN32_KMT = 4,
};

pub const CUmemLocationType = enum(c_int) {
    INVALID = 0,
    DEVICE = 1,
    HOST = 2,
    HOST_NUMA = 3,
    HOST_NUMA_CURRENT = 4,
};

pub const CUmemLocation = extern struct {
    type: CUmemLocationType,
    id: c_int,
};

pub const CUmemAllocationProp = extern struct {
    type: CUmemAllocationType,
    requestedHandleTypes: CUmemAllocationHandleType,
    location: CUmemLocation,
    win32HandleMetaData: ?*anyopaque = null,
    allocFlags: extern struct {
        compressionType: u8 = 0,
        gpuDirectRDMACapable: u8 = 0,
        usage: u16 = 0,
        reserved: [4]u8 = [_]u8{ 0, 0, 0, 0 },
    } = .{},
};

pub const CUmemAccess_flags = enum(c_int) {
    PROT_NONE = 0,
    PROT_READ = 1,
    PROT_READWRITE = 3,
    PROT_MAX = 0x7FFFFFFF,
};

pub const CUmemAccessDesc = extern struct {
    location: CUmemLocation,
    flags: CUmemAccess_flags,
};

pub const CU_MEM_ALLOCATION_GRANULARITY_MINIMUM: u64 = 0;
pub const CU_MEM_ALLOCATION_GRANULARITY_RECOMMENDED: u64 = 1;

pub const CUpointer_attribute = enum(c_int) {
    CONTEXT = 1,
    MEMORY_TYPE = 2,
    DEVICE_POINTER = 3,
    HOST_POINTER = 4,
    P2P_TOKENS = 5,
    SYNC_MEMOPS = 6,
    BUFFER_ID = 7,
    IS_MANAGED = 8,
    DEVICE_ORDINAL = 9,
    IS_LEGACY_CUDA_IPC_CAPABLE = 10,
    IS_SYSMEM_REMOTE = 11,
    ALLOCATION_START = 12,
    ALLOCATION_SIZE = 13,
    ALLOCATION_TYPE = 14,
    BUFFER_SIZE = 15,
    BUFFER_START = 16,
};

pub fn cuPointerGetAttribute(data: *anyopaque, attribute: CUpointer_attribute, ptr: CUdeviceptr) !void {
    const res = cudalib.cuPointerGetAttribute(data, attribute, ptr);
    if (res != .SUCCESS) return error.CudaError;
}

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
    const res = cudalib.cuCtxCreate_v2(&ctx, flags, dev);
    if (res != .SUCCESS) return error.CudaError;
    return ctx;
}

pub fn cuCtxDestroy(ctx: CUcontext) void { _ = cudalib.cuCtxDestroy_v2(ctx); }

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
    const res = cudalib.cuMemAlloc_v2(&dptr, bytes);
    if (res != .SUCCESS) return error.CudaError;
    return dptr;
}

pub fn cuMemFree(dptr: CUdeviceptr) void { _ = cudalib.cuMemFree_v2(dptr); }

pub fn cuMemAllocHost(bytes: usize) !*anyopaque {
    var ptr: ?*anyopaque = null;
    const res = cudalib.cuMemAllocHost_v2(&ptr, bytes);
    if (res != .SUCCESS) return error.CudaError;
    return ptr orelse return error.CudaError;
}

pub fn cuMemFreeHost(ptr: *anyopaque) void { _ = cudalib.cuMemFreeHost(ptr); }

pub fn cuMemAddressReserve(ptr: *CUdeviceptr, size: usize, alignment: usize, addr: CUdeviceptr, flags: u64) !void {
    const res = cudalib.cuMemAddressReserve(ptr, size, alignment, addr, flags);
    if (res != .SUCCESS) return error.CudaError;
}

pub fn cuMemAddressFree(ptr: CUdeviceptr, size: usize) void { _ = cudalib.cuMemAddressFree(ptr, size); }

pub fn cuMemCreate(handle: *CUmemGenericAllocationHandle, size: usize, prop: *CUmemAllocationProp, flags: u64) !void {
    const res = cudalib.cuMemCreate(handle, size, prop, flags);
    if (res != .SUCCESS) return error.CudaError;
}

pub fn cuMemRelease(handle: CUmemGenericAllocationHandle) void { _ = cudalib.cuMemRelease(handle); }

pub fn cuMemMap(ptr: CUdeviceptr, size: usize, offset: usize, handle: CUmemGenericAllocationHandle, flags: u64) !void {
    const res = cudalib.cuMemMap(ptr, size, offset, handle, flags);
    if (res != .SUCCESS) return error.CudaError;
}

pub fn cuMemUnmap(ptr: CUdeviceptr, size: usize) void { _ = cudalib.cuMemUnmap(ptr, size); }

pub fn cuMemSetAccess(ptr: CUdeviceptr, size: usize, desc: *const CUmemAccessDesc, count: usize) !void {
    const res = cudalib.cuMemSetAccess(ptr, size, desc, count);
    if (res != .SUCCESS) return error.CudaError;
}

pub fn cuMemGetAllocationGranularity(granularity: *usize, prop: *CUmemAllocationProp, option: u64) !void {
    const res = cudalib.cuMemGetAllocationGranularity(granularity, prop, option);
    if (res != .SUCCESS) return error.CudaError;
}

pub fn cuMemcpyHtoDAsync(dst: CUdeviceptr, src: usize, bytes: usize, stream: CUstream) !void {
    const res = cudalib.cuMemcpyHtoDAsync_v2(dst, @ptrFromInt(src), bytes, stream);
    if (res != .SUCCESS) return error.CudaError;
}

pub fn cuMemcpyDtoHAsync(dst: usize, src: CUdeviceptr, bytes: usize, stream: CUstream) !void {
    const res = cudalib.cuMemcpyDtoHAsync_v2(@ptrFromInt(dst), src, bytes, stream);
    if (res != .SUCCESS) return error.CudaError;
}

pub fn cuMemsetD8(dst: CUdeviceptr, value: u8, count: usize) !void {
    const res = cudalib.cuMemsetD8_v2(dst, value, count);
    if (res != .SUCCESS) return error.CudaError;
}

pub fn cuMemcpyDtoD(dst: CUdeviceptr, src: CUdeviceptr, bytes: usize) !void {
    const res = cudalib.cuMemcpyDtoD_v2(dst, src, bytes);
    if (res != .SUCCESS) return error.CudaError;
}

/// Copias síncronas (sin stream) para evitar carreras async.
pub fn cuMemcpyHtoD(dst: CUdeviceptr, src: usize, bytes: usize) !void {
    const res = cudalib.cuMemcpyHtoD_v2(dst, @ptrFromInt(src), bytes);
    if (res != .SUCCESS) return error.CudaError;
}

pub fn cuMemcpyDtoH(dst: usize, src: CUdeviceptr, bytes: usize) !void {
    const res = cudalib.cuMemcpyDtoH_v2(@ptrFromInt(dst), src, bytes);
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

pub fn cuStreamDestroy(stream: CUstream) void { _ = cudalib.cuStreamDestroy_v2(stream); }

pub fn cuStreamSynchronize(stream: CUstream) !void {
    const res = cudalib.cuStreamSynchronize(stream);
    if (res != .SUCCESS) return error.CudaError;
}

pub fn cuEventCreate(flags: c_uint) !CUevent {
    var ev: CUevent = undefined;
    const res = cudalib.cuEventCreate(&ev, flags);
    if (res != .SUCCESS) return error.CudaError;
    return ev;
}

pub fn cuEventDestroy(ev: CUevent) void { _ = cudalib.cuEventDestroy(ev); }

pub fn cuEventRecord(ev: CUevent, stream: CUstream) !void {
    const res = cudalib.cuEventRecord(ev, stream);
    if (res != .SUCCESS) return error.CudaError;
}

pub fn cuEventSynchronize(ev: CUevent) !void {
    const res = cudalib.cuEventSynchronize(ev);
    if (res != .SUCCESS) return error.CudaError;
}

pub fn cuEventElapsedTime(ms: *f32, start: CUevent, end: CUevent) !void {
    const res = cudalib.cuEventElapsedTime(ms, start, end);
    if (res != .SUCCESS) return error.CudaError;
}

pub fn cuLaunchKernel(
    f: CUfunction, gridDimX: c_uint, gridDimY: c_uint, gridDimZ: c_uint,
    blockDimX: c_uint, blockDimY: c_uint, blockDimZ: c_uint,
    sharedMemBytes: c_uint, hStream: CUstream, kernelParams: ?*anyopaque, extra: ?*anyopaque,
) !void {
    const res = cudalib.cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
    if (res != .SUCCESS) {
        var errbuf: [*:0]const u8 = "?";
        _ = cudalib.cuGetErrorString(res, &errbuf);
        std.debug.print("cuLaunchKernel FAILED: {s} (0x{x})\n", .{ errbuf, @intFromEnum(res) });
        return error.CudaError;
    }
}

pub fn cuFuncSetAttribute(hfunc: CUfunction, attrib: c_int, value: i64) !void {
    const res = cudalib.cuFuncSetAttribute(hfunc, attrib, value);
    if (res != .SUCCESS) return error.CudaError;
}

/// Bindings reales a la CUDA Driver API (enlazados vía -lcuda).
/// Las funciones son los símbolos estándar exportados por libcuda.so.1
/// (los nombres sin sufijo `_v2` siguen exportados por el driver).
const cudalib = struct {
    extern "c" fn cuInit(flags: c_uint) CUresult;
    extern "c" fn cuDeviceGet(device: *CUdevice, ordinal: c_int) CUresult;
    extern "c" fn cuCtxCreate_v2(ctx: *CUcontext, flags: c_uint, dev: CUdevice) CUresult;
    extern "c" fn cuCtxDestroy_v2(ctx: CUcontext) CUresult;
    extern "c" fn cuDevicePrimaryCtxRetain(ctx: *CUcontext, dev: CUdevice) CUresult;
    extern "c" fn cuCtxSetCurrent(ctx: CUcontext) CUresult;
    extern "c" fn cuCtxGetCurrent(ctx: *CUcontext) CUresult;
    extern "c" fn cuModuleLoad(module: *CUmodule, fname: [*:0]const u8) CUresult;
    extern "c" fn cuModuleUnload(module: CUmodule) CUresult;
    extern "c" fn cuModuleGetFunction(hfunc: *CUfunction, hmod: CUmodule, name: [*:0]const u8) CUresult;
    extern "c" fn cuGetErrorString(err: CUresult, pStr: *[*:0]const u8) CUresult;
    extern "c" fn cuFuncSetAttribute(hfunc: CUfunction, attrib: c_int, value: i64) CUresult;
    extern "c" fn cuMemAlloc_v2(dptr: *CUdeviceptr, bytesize: usize) CUresult;
    extern "c" fn cuMemFree_v2(dptr: CUdeviceptr) CUresult;
    extern "c" fn cuMemAllocHost_v2(pp: *?*anyopaque, bytesize: usize) CUresult;
    extern "c" fn cuMemFreeHost(p: *anyopaque) CUresult;
    extern "c" fn cuMemAddressReserve(ptr: *CUdeviceptr, size: usize, alignment: usize, addr: CUdeviceptr, flags: u64) CUresult;
    extern "c" fn cuMemAddressFree(ptr: CUdeviceptr, size: usize) CUresult;
    extern "c" fn cuMemCreate(handle: *CUmemGenericAllocationHandle, size: usize, prop: *CUmemAllocationProp, flags: u64) CUresult;
    extern "c" fn cuMemRelease(handle: CUmemGenericAllocationHandle) CUresult;
    extern "c" fn cuMemMap(ptr: CUdeviceptr, size: usize, offset: usize, handle: CUmemGenericAllocationHandle, flags: u64) CUresult;
    extern "c" fn cuMemUnmap(ptr: CUdeviceptr, size: usize) CUresult;
    extern "c" fn cuMemSetAccess(ptr: CUdeviceptr, size: usize, desc: *const CUmemAccessDesc, count: usize) CUresult;
    extern "c" fn cuMemGetAllocationGranularity(granularity: *usize, prop: *CUmemAllocationProp, option: u64) CUresult;
    extern "c" fn cuPointerGetAttribute(data: *anyopaque, attribute: CUpointer_attribute, ptr: CUdeviceptr) CUresult;
    extern "c" fn cuMemcpyHtoDAsync_v2(dst: CUdeviceptr, src: *const anyopaque, bytes: usize, stream: CUstream) CUresult;
    extern "c" fn cuMemcpyDtoHAsync_v2(dst: *anyopaque, src: CUdeviceptr, bytes: usize, stream: CUstream) CUresult;
    extern "c" fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const anyopaque, bytes: usize) CUresult;
    extern "c" fn cuMemsetD8_v2(dst: CUdeviceptr, value: u8, count: usize) CUresult;
    extern "c" fn cuMemcpyDtoD_v2(dst: CUdeviceptr, src: CUdeviceptr, bytes: usize) CUresult;
    extern "c" fn cuMemcpyDtoH_v2(dst: *anyopaque, src: CUdeviceptr, bytes: usize) CUresult;
    extern "c" fn cuCtxSynchronize() CUresult;
    extern "c" fn cuStreamCreate(phStream: *CUstream, flags: c_uint) CUresult;
    extern "c" fn cuStreamDestroy_v2(hStream: CUstream) CUresult;
    extern "c" fn cuStreamSynchronize(hStream: CUstream) CUresult;
    extern "c" fn cuEventCreate(event: *CUevent, flags: c_uint) CUresult;
    extern "c" fn cuEventDestroy(event: CUevent) CUresult;
    extern "c" fn cuEventRecord(event: CUevent, stream: CUstream) CUresult;
    extern "c" fn cuEventSynchronize(event: CUevent) CUresult;
    extern "c" fn cuEventElapsedTime(ms: *f32, start: CUevent, end: CUevent) CUresult;
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
    // Contexto PRIMARIO: cuBLAS y la CUDA Runtime API operan sobre el contexto
    // primario; cuCtxCreate crearía un contexto no-primario incompatible.
    // (Los símbolos driver usados son las variantes _v2, ABI actual.)
    if (cudalib.cuDevicePrimaryCtxRetain(&ctx, device) != .SUCCESS) return error.CudaError;
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
    if (@intFromPtr(ctx) == 0) return null;
    return ctx;
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
