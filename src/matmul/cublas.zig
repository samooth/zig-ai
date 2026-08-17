
//! FFI a NVIDIA cuBLAS — Avanzado
//! Incluye: streams async, batch GEMM, strided GEMM, mem pool persistente,
//! cublasGemmEx para precisión mixta FP16/BF16.
//! Requiere compilar con: zig build -Dcublas=true

const std = @import("std");
const Tensor = @import("core").Tensor;
const time = @import("time");

// ═══════════════════════════════════════════════════════════════════════════════
// Declaraciones externas CUDA / cuBLAS
// ═══════════════════════════════════════════════════════════════════════════════

extern "c" fn cublasCreate_v2(handle: **anyopaque) i32;
extern "c" fn cublasDestroy_v2(handle: *anyopaque) void;
extern "c" fn cublasSetStream_v2(handle: *anyopaque, stream: *anyopaque) i32;
extern "c" fn cublasGetStream_v2(handle: *anyopaque, stream: **anyopaque) i32;

extern "c" fn cublasSgemm_v2(
    handle: *anyopaque, transa: i32, transb: i32,
    m: i32, n: i32, k: i32,
    alpha: *const f32, A: *const anyopaque, lda: i32,
    B: *const anyopaque, ldb: i32,
    beta: *const f32, C: *anyopaque, ldc: i32,
) i32;

extern "c" fn cublasHgemm(
    handle: *anyopaque, transa: i32, transb: i32,
    m: i32, n: i32, k: i32,
    alpha: *const anyopaque, A: *const anyopaque, lda: i32,
    B: *const anyopaque, ldb: i32,
    beta: *const anyopaque, C: *anyopaque, ldc: i32,
) i32;

extern "c" fn cublasGemmEx(
    handle: *anyopaque,
    transa: i32, transb: i32,
    m: i32, n: i32, k: i32,
    alpha: *const anyopaque,
    A: *const anyopaque, cuda_type_A: i32, lda: i32,
    B: *const anyopaque, cuda_type_B: i32, ldb: i32,
    beta: *const anyopaque,
    C: *anyopaque, cuda_type_C: i32, ldc: i32,
    compute_type: i32, algo: i32,
) i32;

extern "c" fn cublasSgemmStridedBatched(
    handle: *anyopaque,
    transa: i32, transb: i32,
    m: i32, n: i32, k: i32,
    alpha: *const f32,
    A: *const anyopaque, lda: i32, strideA: i64,
    B: *const anyopaque, ldb: i32, strideB: i64,
    beta: *const f32,
    C: *anyopaque, ldc: i32, strideC: i64,
    batchCount: i32,
) i32;

extern "c" fn cublasGemmStridedBatchedEx(
    handle: *anyopaque,
    transa: i32, transb: i32,
    m: i32, n: i32, k: i32,
    alpha: *const anyopaque,
    A: *const anyopaque, cuda_type_A: i32, lda: i32, strideA: i64,
    B: *const anyopaque, cuda_type_B: i32, ldb: i32, strideB: i64,
    beta: *const anyopaque,
    C: *anyopaque, cuda_type_C: i32, ldc: i32, strideC: i64,
    batchCount: i32,
    compute_type: i32, algo: i32,
) i32;

// CUDA runtime
extern "c" fn cudaMalloc(devPtr: **anyopaque, size: usize) i32;
extern "c" fn cudaFree(devPtr: *anyopaque) i32;
extern "c" fn cudaMallocAsync(devPtr: **anyopaque, size: usize, stream: *anyopaque) i32;
extern "c" fn cudaFreeAsync(devPtr: *anyopaque, stream: *anyopaque) i32;
extern "c" fn cudaMemcpyAsync(dst: *anyopaque, src: *const anyopaque, size: usize, kind: i32, stream: *anyopaque) i32;
extern "c" fn cudaMemcpy(dst: *anyopaque, src: *const anyopaque, size: usize, kind: i32) i32;
extern "c" fn cudaStreamCreate(stream: **anyopaque) i32;
extern "c" fn cudaStreamDestroy(stream: *anyopaque) i32;
extern "c" fn cudaStreamSynchronize(stream: *anyopaque) i32;
extern "c" fn cudaDeviceSynchronize() i32;
extern "c" fn cudaGetDeviceCount(count: *i32) i32;

const cudaMemcpyHostToDevice = 1;
const cudaMemcpyDeviceToHost = 2;
const cudaMemcpyDeviceToDevice = 3;

const CUBLAS_OP_N: c_int = 0;
const CUBLAS_OP_T: c_int = 1;
const CUBLAS_STATUS_SUCCESS = 0;

// CUDA data types para GemmEx
const CUDA_R_32F = 0;
const CUDA_R_16F = 2;
const CUDA_R_8I = 3;
const CUDA_R_8U = 8;
const CUDA_R_64F = 1;
const CUDA_R_32I = 10;
const CUDA_R_32U = 12;
const CUDA_R_16BF = 14;

// Compute types
const CUBLAS_COMPUTE_16F = 64;
const CUBLAS_COMPUTE_32F = 68;
const CUBLAS_COMPUTE_32F_FAST_16F = 74;
const CUBLAS_COMPUTE_32F_FAST_16BF = 75;
const CUBLAS_COMPUTE_32F_FAST_TF32 = 77;

const CUBLAS_GEMM_DEFAULT = -1;

/// Convierte en host la salida de cuBLAS (column-major, ldc=M) a row-major
/// [rows, cols]: cuBLAS escribe `buf[i + rows*j] = C[i][j]`; aquí lo movemos
/// a `C[i*cols + j]`. (No es un transpose: el shape [rows, cols] se conserva.)
fn colMajorToRowMajor(data: []f32, rows: usize, cols: usize) !void {
    const tmp = try std.heap.c_allocator.alloc(f32, data.len);
    defer std.heap.c_allocator.free(tmp);
    for (0..rows) |i| {
        for (0..cols) |j| {
            tmp[i * cols + j] = data[i + rows * j];
        }
    }
    @memcpy(data, tmp);
}

// ═══════════════════════════════════════════════════════════════════════════════
// CuBlasHandle — con soporte de streams
// ═══════════════════════════════════════════════════════════════════════════════

pub const CuBlasHandle = struct {
    raw: *anyopaque,
    stream: ?*anyopaque,

    pub fn init() !CuBlasHandle {
        var raw: *anyopaque = undefined;
        const status = cublasCreate_v2(&raw);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std.log.err("cublasCreate failed: {}", .{status});
            return error.CuBlasInitFailed;
        }
        return CuBlasHandle{ .raw = raw, .stream = null };
    }

    pub fn deinit(self: CuBlasHandle) void {
        _ = cublasDestroy_v2(self.raw);
    }

    /// Asociar un stream CUDA para operaciones async
    pub fn setStream(self: *CuBlasHandle, stream: *anyopaque) !void {
        const status = cublasSetStream_v2(self.raw, stream);
        if (status != CUBLAS_STATUS_SUCCESS) return error.CuBlasSetStreamFailed;
        self.stream = stream;
    }

    /// Sincronizar el stream asociado (o device entero si no hay stream)
    pub fn sync(self: CuBlasHandle) void {
        if (self.stream) |s| {
            _ = cudaStreamSynchronize(s);
        } else {
            _ = cudaDeviceSynchronize();
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// CudaStream — wrapper para streams async
// ═══════════════════════════════════════════════════════════════════════════════

pub const CudaStream = struct {
    raw: *anyopaque,

    pub fn create() !CudaStream {
        var raw: *anyopaque = undefined;
        const status = cudaStreamCreate(&raw);
        if (status != 0) return error.CudaStreamCreateFailed;
        return CudaStream{ .raw = raw };
    }

    pub fn destroy(self: CudaStream) void {
        _ = cudaStreamDestroy(self.raw);
    }

    pub fn synchronize(self: CudaStream) void {
        _ = cudaStreamSynchronize(self.raw);
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// GpuBuffer — con soporte de streams async
// ═══════════════════════════════════════════════════════════════════════════════

pub fn GpuBuffer(comptime T: type) type {
    return struct {
        const Self = @This();
        dev_ptr: *anyopaque,
        len: usize,
        stream: ?*anyopaque,

        pub fn alloc(len: usize) !Self {
            var dev_ptr: *anyopaque = undefined;
            const size = len * @sizeOf(T);
            const status = cudaMalloc(&dev_ptr, size);
            if (status != 0) return error.CudaMallocFailed;
            return Self{ .dev_ptr = dev_ptr, .len = len, .stream = null };
        }

        pub fn allocAsync(len: usize, stream: *anyopaque) !Self {
            var dev_ptr: *anyopaque = undefined;
            const size = len * @sizeOf(T);
            const status = cudaMallocAsync(&dev_ptr, size, stream);
            if (status != 0) return error.CudaMallocAsyncFailed;
            return Self{ .dev_ptr = dev_ptr, .len = len, .stream = stream };
        }

        pub fn free(self: Self) void {
            _ = cudaFree(self.dev_ptr);
        }

        pub fn freeAsync(self: Self) void {
            if (self.stream) |s| {
                _ = cudaFreeAsync(self.dev_ptr, s);
            } else {
                _ = cudaFree(self.dev_ptr);
            }
        }

        pub fn upload(self: Self, host_data: []const T) !void {
            const status = cudaMemcpy(self.dev_ptr, host_data.ptr, host_data.len * @sizeOf(T), cudaMemcpyHostToDevice);
            if (status != 0) return error.CudaMemcpyFailed;
        }

        pub fn uploadAsync(self: Self, host_data: []const T) !void {
            const s = self.stream orelse return error.NoStreamSet;
            const status = cudaMemcpyAsync(self.dev_ptr, host_data.ptr, host_data.len * @sizeOf(T), cudaMemcpyHostToDevice, s);
            if (status != 0) return error.CudaMemcpyAsyncFailed;
        }

        pub fn download(self: Self, host_data: []T) !void {
            const status = cudaMemcpy(host_data.ptr, self.dev_ptr, host_data.len * @sizeOf(T), cudaMemcpyDeviceToHost);
            if (status != 0) return error.CudaMemcpyFailed;
        }

        pub fn downloadAsync(self: Self, host_data: []T) !void {
            const s = self.stream orelse return error.NoStreamSet;
            const status = cudaMemcpyAsync(host_data.ptr, self.dev_ptr, host_data.len * @sizeOf(T), cudaMemcpyDeviceToHost, s);
            if (status != 0) return error.CudaMemcpyAsyncFailed;
        }

        pub fn copyFrom(self: Self, other: Self) !void {
            const size = @min(self.len, other.len) * @sizeOf(T);
            const status = cudaMemcpy(self.dev_ptr, other.dev_ptr, size, cudaMemcpyDeviceToDevice);
            if (status != 0) return error.CudaMemcpyFailed;
        }
    };
}

/// Tensor que vive íntegramente en GPU (sin ida/vuelta a host por matmul).
/// Usado por la ruta fused GPU-resident de la capa híbrida.
pub fn GpuTensor(comptime T: type) type {
    return struct {
        const Self = @This();
        buf: GpuBuffer(T),

        pub fn alloc(n: usize) !Self {
            return .{ .buf = try GpuBuffer(T).alloc(n) };
        }
        pub fn ptr(self: Self) usize {
            return @intFromPtr(self.buf.dev_ptr);
        }
        pub fn deinit(self: Self) void {
            self.buf.free();
        }
    };
}

// ═══════════════════════════════════════════════════════════════════════════════
// GpuMemoryPool — gestión persistente de memoria GPU
// ═══════════════════════════════════════════════════════════════════════════════

pub const GpuMemoryPool = struct {
    const Block = struct {
        ptr: *anyopaque,
        size: usize,
        in_use: bool,
    };

    allocator: std.mem.Allocator,
    blocks: std.ArrayList(Block),
    stream: ?*anyopaque,

    pub fn init(allocator: std.mem.Allocator) GpuMemoryPool {
        return GpuMemoryPool{
            .allocator = allocator,
            .blocks = .empty,
            .stream = null,
        };
    }

    pub fn deinit(self: *GpuMemoryPool) void {
        for (self.blocks.items) |block| {
            _ = cudaFree(block.ptr);
        }
        self.blocks.deinit(self.allocator);
    }

    pub fn setStream(self: *GpuMemoryPool, stream: *anyopaque) void {
        self.stream = stream;
    }

    /// Obtener un bloque de al menos `size` bytes. Reutiliza si hay libres.
    pub fn acquire(self: *GpuMemoryPool, size: usize) !*anyopaque {
        // Buscar bloque libre del tamaño adecuado
        for (self.blocks.items) |*block| {
            if (!block.in_use and block.size >= size) {
                block.in_use = true;
                return block.ptr;
            }
        }

        // No hay bloque libre, allocar nuevo
        var ptr: *anyopaque = undefined;
        const status = if (self.stream) |s|
            cudaMallocAsync(&ptr, size, s)
        else
            cudaMalloc(&ptr, size);

        if (status != 0) return error.CudaMallocFailed;

        try self.blocks.append(self.allocator, Block{ .ptr = ptr, .size = size, .in_use = true });
        return ptr;
    }

    /// Liberar un bloque para reutilización (no hace cudaFree real)
    pub fn release(self: *GpuMemoryPool, ptr: *anyopaque) void {
        for (self.blocks.items) |*block| {
            if (block.ptr == ptr) {
                block.in_use = false;
                return;
            }
        }
    }

    /// Liberar todos los bloques no en uso
    pub fn defragment(self: *GpuMemoryPool) void {
        var i: usize = self.blocks.items.len;
        while (i > 0) : (i -= 1) {
            const block = &self.blocks.items[i - 1];
            if (!block.in_use) {
                _ = cudaFree(block.ptr);
                _ = self.blocks.orderedRemove(i - 1);
            }
        }
    }

    pub const PoolStats = struct { total: usize, used: usize, free: usize };

    pub fn stats(self: GpuMemoryPool) PoolStats {
        var total: usize = 0;
        var used: usize = 0;
        for (self.blocks.items) |block| {
            total += block.size;
            if (block.in_use) used += block.size;
        }
        return .{ .total = total, .used = used, .free = total - used };
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// GEMM FP32
// ═══════════════════════════════════════════════════════════════════════════════

pub fn gemmCuBlasF32(
    handle: CuBlasHandle,
    A: Tensor(f32),
    B: Tensor(f32),
    C: *Tensor(f32),
    M: usize,
    N: usize,
    K: usize,
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
    beta: f32,
) !void {
    std.debug.assert(A.isContiguous() and B.isContiguous() and C.isContiguous());

    var d_A = try GpuBuffer(f32).alloc(A.data.len);
    defer d_A.free();
    try d_A.upload(A.data);

    var d_B = try GpuBuffer(f32).alloc(B.data.len);
    defer d_B.free();
    try d_B.upload(B.data);

    var d_C = try GpuBuffer(f32).alloc(C.data.len);
    defer d_C.free();
    try d_C.upload(C.data);

    // Tensores ROW-MAJOR; cuBLAS es COLUMN-MAJOR. Un tensor row-major [R, C]
    // se pasa como su transpuesto column-major con op='T' y leading dim = C.
    const op_a = if (trans_a) CUBLAS_OP_N else CUBLAS_OP_T;
    const op_b = if (trans_b) CUBLAS_OP_N else CUBLAS_OP_T;
    const lda: i32 = if (trans_a) @intCast(M) else @intCast(K);
    const ldb: i32 = if (trans_b) @intCast(K) else @intCast(N);
    const ldc: i32 = @intCast(M);

    const status = cublasSgemm_v2(
        handle.raw,
        op_a, op_b,
        @intCast(M), @intCast(N), @intCast(K),
        &alpha,
        d_A.dev_ptr, lda,
        d_B.dev_ptr, ldb,
        &beta,
        d_C.dev_ptr, ldc,
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std.log.err("cublasSgemm failed: {}", .{status});
        return error.CuBlasGemmFailed;
    }

    try d_C.download(C.data);
    // cuBLAS escribe C column-major → transponer a row-major
    try colMajorToRowMajor(C.data, M, N);
    handle.sync();
}

/// GEMM f32 donde B (los pesos) YA está residente en el device (d_B). Solo se
/// sube A (activación, pequeña) y se descarga C. Evita re-subir los pesos en
/// cada token de generación (el cuello de botella original de este engine).
// ─── Scratch GPU pool compartido (evita cudaMalloc/cudaFree por cada GEMM) ───
var g_scratch_pool: GpuMemoryPool = undefined;
var g_scratch_pool_init = false;

fn scratchPool() !*GpuMemoryPool {
    if (!g_scratch_pool_init) {
        g_scratch_pool = GpuMemoryPool.init(std.heap.c_allocator);
        g_scratch_pool_init = true;
    }
    return &g_scratch_pool;
}

// ─── Instrumentación de rendimiento (DBG) ───

pub fn gemmCuBlasF32Resident(
    handle: CuBlasHandle,
    A: Tensor(f32),
    d_B: GpuBuffer(f32),
    C: *Tensor(f32),
    M: usize,
    N: usize,
    K: usize,
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
    beta: f32,
) !void {
    std.debug.assert(A.isContiguous() and C.isContiguous());

    const pool = try scratchPool();
    const bytes_a = A.data.len * @sizeOf(f32);
    const bytes_c = C.data.len * @sizeOf(f32);

    const d_A_ptr = try pool.acquire(bytes_a);
    defer pool.release(d_A_ptr);
    const d_C_ptr = try pool.acquire(bytes_c);
    defer pool.release(d_C_ptr);

    // Subir A (H2D). C solo se sube si beta != 0 (los callers pasan 0, así que
    // el resultado es puramente alpha*A*B y la subida de C es desperdicio).
    if (cudaMemcpy(d_A_ptr, A.data.ptr, bytes_a, cudaMemcpyHostToDevice) != 0) {
        return error.CudaMemcpyFailed;
    }
    if (beta != 0) {
        if (cudaMemcpy(d_C_ptr, C.data.ptr, bytes_c, cudaMemcpyHostToDevice) != 0) {
            return error.CudaMemcpyFailed;
        }
    }

    const op_a = if (trans_a) CUBLAS_OP_N else CUBLAS_OP_T;
    const op_b = if (trans_b) CUBLAS_OP_N else CUBLAS_OP_T;
    const lda: i32 = if (trans_a) @intCast(M) else @intCast(K);
    const ldb: i32 = if (trans_b) @intCast(K) else @intCast(N);
    const ldc: i32 = @intCast(M);

    const status = cublasSgemm_v2(
        handle.raw,
        op_a, op_b,
        @intCast(M), @intCast(N), @intCast(K),
        &alpha,
        d_A_ptr, lda,
        d_B.dev_ptr, ldb,
        &beta,
        d_C_ptr, ldc,
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std.log.err("cublasSgemm (resident) failed: {}", .{status});
        return error.CuBlasGemmFailed;
    }

    if (cudaMemcpy(C.data.ptr, d_C_ptr, bytes_c, cudaMemcpyDeviceToHost) != 0) {
        return error.CudaMemcpyFailed;
    }
    // Para M==1 (decode) column-major == row-major: no hace falta transponer.
    if (M != 1) try colMajorToRowMajor(C.data, M, N);
    handle.sync();
}

/// GEMM device→device: A, B y C ya viven en GPU. NO hace H2D/D2H ni sincroniza;
/// el llamador sincroniza el stream una sola vez por token. Devuelve al
/// retornar la GEMM está encolada en el stream por defecto de cuBLAS.
pub fn gemmCuBlasF32Device(
    handle: CuBlasHandle,
    d_A: GpuBuffer(f32),
    d_B: GpuBuffer(f32),
    d_C: GpuBuffer(f32),
    M: usize,
    N: usize,
    K: usize,
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
    beta: f32,
) !void {
    const op_a = if (trans_a) CUBLAS_OP_N else CUBLAS_OP_T;
    const op_b = if (trans_b) CUBLAS_OP_N else CUBLAS_OP_T;
    const lda: i32 = if (trans_a) @intCast(M) else @intCast(K);
    const ldb: i32 = if (trans_b) @intCast(K) else @intCast(N);
    const ldc: i32 = @intCast(M);

    const status = cublasSgemm_v2(
        handle.raw,
        op_a, op_b,
        @intCast(M), @intCast(N), @intCast(K),
        &alpha,
        d_A.dev_ptr, lda,
        d_B.dev_ptr, ldb,
        &beta,
        d_C.dev_ptr, ldc,
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        std.log.err("cublasSgemm (device) failed: {}", .{status});
        return error.CuBlasGemmFailed;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GEMM Async con streams — overlap compute/memcpy
// ═══════════════════════════════════════════════════════════════════════════════

pub fn gemmCuBlasF32Async(
    handle: CuBlasHandle,
    stream: CudaStream,
    pool: *GpuMemoryPool,
    A: Tensor(f32),
    B: Tensor(f32),
    C: *Tensor(f32),
    M: usize,
    N: usize,
    K: usize,
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
    beta: f32,
) !void {
    std.debug.assert(A.isContiguous() and B.isContiguous() and C.isContiguous());

    var h = handle;
    try h.setStream(stream.raw);

    const size_A = A.data.len * @sizeOf(f32);
    const size_B = B.data.len * @sizeOf(f32);
    const size_C = C.data.len * @sizeOf(f32);

    // Adquirir buffers del pool (o allocar nuevos)
    const d_A_ptr = try pool.acquire(size_A);
    defer pool.release(d_A_ptr);
    const d_B_ptr = try pool.acquire(size_B);
    defer pool.release(d_B_ptr);
    const d_C_ptr = try pool.acquire(size_C);
    defer pool.release(d_C_ptr);

    // Upload async
    const status_h2d_a = cudaMemcpyAsync(d_A_ptr, A.data.ptr, size_A, cudaMemcpyHostToDevice, stream.raw);
    if (status_h2d_a != 0) return error.CudaMemcpyAsyncFailed;

    const status_h2d_b = cudaMemcpyAsync(d_B_ptr, B.data.ptr, size_B, cudaMemcpyHostToDevice, stream.raw);
    if (status_h2d_b != 0) return error.CudaMemcpyAsyncFailed;

    // GEMM async en stream (tensores row-major → op='T', leading dim = cols)
    const op_a = if (trans_a) CUBLAS_OP_N else CUBLAS_OP_T;
    const op_b = if (trans_b) CUBLAS_OP_N else CUBLAS_OP_T;
    const lda: i32 = if (trans_a) @intCast(M) else @intCast(K);
    const ldb: i32 = if (trans_b) @intCast(K) else @intCast(N);
    const ldc: i32 = @intCast(M);

    const status = cublasSgemm_v2(
        h.raw,
        op_a, op_b,
        @intCast(M), @intCast(N), @intCast(K),
        &alpha,
        d_A_ptr, lda,
        d_B_ptr, ldb,
        &beta,
        d_C_ptr, ldc,
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std.log.err("cublasSgemm async failed: {}", .{status});
        return error.CuBlasGemmFailed;
    }
    // Download async
    const status_d2h = cudaMemcpyAsync(C.data.ptr, d_C_ptr, size_C, cudaMemcpyDeviceToHost, stream.raw);
    if (status_d2h != 0) return error.CudaMemcpyAsyncFailed;
    stream.synchronize();
    // cuBLAS escribe C column-major → transponer a row-major
    try colMajorToRowMajor(C.data, M, N);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Batch GEMM — múltiples secuencias
// ═══════════════════════════════════════════════════════════════════════════════

/// Batch GEMM FP32: C[b] = alpha * A[b] * B[b] + beta * C[b]
/// A, B, C son arrays de batchCount tensores, todos [M,K], [K,N], [M,N]
pub fn gemmBatchF32(
    handle: CuBlasHandle,
    A_batch: []const Tensor(f32),
    B_batch: []const Tensor(f32),
    C_batch: []const *Tensor(f32),
    M: usize,
    N: usize,
    K: usize,
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
    beta: f32,
) !void {
    const batchCount = A_batch.len;
    std.debug.assert(B_batch.len == batchCount and C_batch.len == batchCount);

    // Contiguar todos los datos en buffers flat
    const total_A = batchCount * M * K;
    const total_B = batchCount * K * N;
    const total_C = batchCount * M * N;

    var d_A = try GpuBuffer(f32).alloc(total_A);
    defer d_A.free();
    var d_B = try GpuBuffer(f32).alloc(total_B);
    defer d_B.free();
    var d_C = try GpuBuffer(f32).alloc(total_C);
    defer d_C.free();

    // Upload batch
    for (A_batch, 0..) |A, b| {
        const offset = b * M * K;
        const status = cudaMemcpy(
            @ptrCast(@alignCast(@as([*]u8, @ptrCast(d_A.dev_ptr)) + offset * @sizeOf(f32))),
            A.data.ptr, A.data.len * @sizeOf(f32), cudaMemcpyHostToDevice,
        );
        if (status != 0) return error.CudaMemcpyFailed;
    }

    for (B_batch, 0..) |B, b| {
        const offset = b * K * N;
        const status = cudaMemcpy(
            @ptrCast(@alignCast(@as([*]u8, @ptrCast(d_B.dev_ptr)) + offset * @sizeOf(f32))),
            B.data.ptr, B.data.len * @sizeOf(f32), cudaMemcpyHostToDevice,
        );
        if (status != 0) return error.CudaMemcpyFailed;
    }

    for (C_batch, 0..) |C, b| {
        const offset = b * M * N;
        const status = cudaMemcpy(
            @ptrCast(@alignCast(@as([*]u8, @ptrCast(d_C.dev_ptr)) + offset * @sizeOf(f32))),
            C.data.ptr, C.data.len * @sizeOf(f32), cudaMemcpyHostToDevice,
        );
        if (status != 0) return error.CudaMemcpyFailed;
    }

    const op_a = if (trans_a) CUBLAS_OP_N else CUBLAS_OP_T;
    const op_b = if (trans_b) CUBLAS_OP_N else CUBLAS_OP_T;
    const lda: i32 = if (trans_a) @intCast(M) else @intCast(K);
    const ldb: i32 = if (trans_b) @intCast(K) else @intCast(N);
    const ldc: i32 = @intCast(M);

    const strideA: i64 = @intCast(M * K);
    const strideB: i64 = @intCast(K * N);
    const strideC: i64 = @intCast(M * N);

    const status = cublasSgemmStridedBatched(
        handle.raw,
        op_a, op_b,
        @intCast(M), @intCast(N), @intCast(K),
        &alpha,
        d_A.dev_ptr, lda, strideA,
        d_B.dev_ptr, ldb, strideB,
        &beta,
        d_C.dev_ptr, ldc, strideC,
        @intCast(batchCount),
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std.log.err("cublasSgemmStridedBatched failed: {}", .{status});
        return error.CuBlasBatchGemmFailed;
    }

    // Download batch
    for (C_batch, 0..) |C, b| {
        const offset = b * M * N;
        const copy_status = cudaMemcpy(
            C.data.ptr,
            @ptrCast(@alignCast(@as([*]u8, @ptrCast(d_C.dev_ptr)) + offset * @sizeOf(f32))),
            C.data.len * @sizeOf(f32), cudaMemcpyDeviceToHost,
        );
        if (copy_status != 0) return error.CudaMemcpyFailed;
    }

    handle.sync();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Strided GEMM — para GQA/MQA (múltiples heads con strides fijos)
// ═══════════════════════════════════════════════════════════════════════════════

/// Strided GEMM para GQA: cada "head" comparte los mismos K/V pero tiene Q distinto.
/// O para MQA: un solo K/V, múltiples Q.
/// A_flat: [batchCount, M, K] contiguo
/// B_flat: [batchCount, K, N] contiguo (o compartido con strideB=0 para MQA)
/// C_flat: [batchCount, M, N] contiguo
pub fn gemmStridedF32(
    handle: CuBlasHandle,
    A_flat: []const f32,
    B_flat: []const f32,
    C_flat: []f32,
    M: usize,
    N: usize,
    K: usize,
    batchCount: usize,
    strideA: i64,
    strideB: i64,
    strideC: i64,
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
    beta: f32,
) !void {
    const total_A = if (strideA == 0) M * K else batchCount * M * K;
    const total_B = if (strideB == 0) K * N else batchCount * K * N;
    const total_C = batchCount * M * N;

    var d_A = try GpuBuffer(f32).alloc(total_A);
    defer d_A.free();
    try d_A.upload(A_flat[0..total_A]);

    var d_B = try GpuBuffer(f32).alloc(total_B);
    defer d_B.free();
    try d_B.upload(B_flat[0..total_B]);

    var d_C = try GpuBuffer(f32).alloc(total_C);
    defer d_C.free();
    try d_C.upload(C_flat[0..total_C]);

    const op_a = if (trans_a) CUBLAS_OP_N else CUBLAS_OP_T;
    const op_b = if (trans_b) CUBLAS_OP_N else CUBLAS_OP_T;
    const lda: i32 = if (trans_a) @intCast(M) else @intCast(K);
    const ldb: i32 = if (trans_b) @intCast(K) else @intCast(N);
    const ldc: i32 = @intCast(M);

    const status = cublasSgemmStridedBatched(
        handle.raw,
        op_a, op_b,
        @intCast(M), @intCast(N), @intCast(K),
        &alpha,
        d_A.dev_ptr, lda, strideA,
        d_B.dev_ptr, ldb, strideB,
        &beta,
        d_C.dev_ptr, ldc, strideC,
        @intCast(batchCount),
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std.log.err("cublasSgemmStridedBatched (strided) failed: {}", .{status});
        return error.CuBlasStridedGemmFailed;
    }

    try d_C.download(C_flat[0..total_C]);
    handle.sync();
}

// ═══════════════════════════════════════════════════════════════════════════════
// cublasGemmEx — Precisión mixta FP16/BF16 entrada, FP32 acumulación
// ═══════════════════════════════════════════════════════════════════════════════

/// GemmEx con FP16 entrada y FP32 acumulación (Tensor Cores)
/// A, B: f16 (zig f16), C: f32
pub fn gemmExF16F32(
    handle: CuBlasHandle,
    A: Tensor(f16),
    B: Tensor(f16),
    C: *Tensor(f32),
    M: usize,
    N: usize,
    K: usize,
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
    beta: f32,
) !void {
    std.debug.assert(A.isContiguous() and B.isContiguous() and C.isContiguous());

    var d_A = try GpuBuffer(f16).alloc(A.data.len);
    defer d_A.free();
    try d_A.upload(A.data);

    var d_B = try GpuBuffer(f16).alloc(B.data.len);
    defer d_B.free();
    try d_B.upload(B.data);

    var d_C = try GpuBuffer(f32).alloc(C.data.len);
    defer d_C.free();
    try d_C.upload(C.data);

    const op_a = if (trans_a) CUBLAS_OP_N else CUBLAS_OP_T;
    const op_b = if (trans_b) CUBLAS_OP_N else CUBLAS_OP_T;
    const lda: i32 = if (trans_a) @intCast(M) else @intCast(K);
    const ldb: i32 = if (trans_b) @intCast(K) else @intCast(N);
    const ldc: i32 = @intCast(M);

    const status = cublasGemmEx(
        handle.raw,
        op_a, op_b,
        @intCast(M), @intCast(N), @intCast(K),
        &alpha,
        d_A.dev_ptr, CUDA_R_16F, lda,
        d_B.dev_ptr, CUDA_R_16F, ldb,
        &beta,
        d_C.dev_ptr, CUDA_R_32F, ldc,
        CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT,
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std.log.err("cublasGemmEx (FP16->FP32) failed: {}", .{status});
        return error.CuBlasGemmExFailed;
    }

    try d_C.download(C.data);
    handle.sync();
}

/// GemmEx con BF16 entrada y FP32 acumulación
/// BF16 en Zig: usamos u16 como storage, reinterpretamos en GPU
pub fn gemmExBF16F32(
    handle: CuBlasHandle,
    A: Tensor(u16), // BF16 almacenado como u16
    B: Tensor(u16),
    C: *Tensor(f32),
    M: usize,
    N: usize,
    K: usize,
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
    beta: f32,
) !void {
    std.debug.assert(A.isContiguous() and B.isContiguous() and C.isContiguous());

    var d_A = try GpuBuffer(u16).alloc(A.data.len);
    defer d_A.free();
    try d_A.upload(A.data);

    var d_B = try GpuBuffer(u16).alloc(B.data.len);
    defer d_B.free();
    try d_B.upload(B.data);

    var d_C = try GpuBuffer(f32).alloc(C.data.len);
    defer d_C.free();
    try d_C.upload(C.data);

    const op_a = if (trans_a) CUBLAS_OP_N else CUBLAS_OP_T;
    const op_b = if (trans_b) CUBLAS_OP_N else CUBLAS_OP_T;
    const lda: i32 = if (trans_a) @intCast(M) else @intCast(K);
    const ldb: i32 = if (trans_b) @intCast(K) else @intCast(N);
    const ldc: i32 = @intCast(M);

    const status = cublasGemmEx(
        handle.raw,
        op_a, op_b,
        @intCast(M), @intCast(N), @intCast(K),
        &alpha,
        d_A.dev_ptr, CUDA_R_16BF, lda,
        d_B.dev_ptr, CUDA_R_16BF, ldb,
        &beta,
        d_C.dev_ptr, CUDA_R_32F, ldc,
        CUBLAS_COMPUTE_32F_FAST_16BF,
        CUBLAS_GEMM_DEFAULT,
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std.log.err("cublasGemmEx (BF16->FP32) failed: {}", .{status});
        return error.CuBlasGemmExFailed;
    }

    try d_C.download(C.data);
    handle.sync();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Batch GemmEx — precisión mixta para múltiples secuencias
// ═══════════════════════════════════════════════════════════════════════════════

pub fn gemmBatchExF16F32(
    handle: CuBlasHandle,
    A_batch: []const Tensor(f16),
    B_batch: []const Tensor(f16),
    C_batch: []const *Tensor(f32),
    M: usize,
    N: usize,
    K: usize,
    batchCount: usize,
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
    beta: f32,
) !void {
    const total_A = batchCount * M * K;
    const total_B = batchCount * K * N;
    const total_C = batchCount * M * N;

    var d_A = try GpuBuffer(f16).alloc(total_A);
    defer d_A.free();
    var d_B = try GpuBuffer(f16).alloc(total_B);
    defer d_B.free();
    var d_C = try GpuBuffer(f32).alloc(total_C);
    defer d_C.free();

    // Upload batches
    for (A_batch, 0..) |A, b| {
        const offset = b * M * K;
        const status = cudaMemcpy(
            @ptrCast(@alignCast(@as([*]u8, @ptrCast(d_A.dev_ptr)) + offset * @sizeOf(f16))),
            A.data.ptr, A.data.len * @sizeOf(f16), cudaMemcpyHostToDevice,
        );
        if (status != 0) return error.CudaMemcpyFailed;
    }

    for (B_batch, 0..) |B, b| {
        const offset = b * K * N;
        const status = cudaMemcpy(
            @ptrCast(@alignCast(@as([*]u8, @ptrCast(d_B.dev_ptr)) + offset * @sizeOf(f16))),
            B.data.ptr, B.data.len * @sizeOf(f16), cudaMemcpyHostToDevice,
        );
        if (status != 0) return error.CudaMemcpyFailed;
    }

    for (C_batch, 0..) |C, b| {
        const offset = b * M * N;
        const status = cudaMemcpy(
            @ptrCast(@alignCast(@as([*]u8, @ptrCast(d_C.dev_ptr)) + offset * @sizeOf(f32))),
            C.data.ptr, C.data.len * @sizeOf(f32), cudaMemcpyHostToDevice,
        );
        if (status != 0) return error.CudaMemcpyFailed;
    }

    const op_a = if (trans_a) CUBLAS_OP_N else CUBLAS_OP_T;
    const op_b = if (trans_b) CUBLAS_OP_N else CUBLAS_OP_T;
    const lda: i32 = if (trans_a) @intCast(M) else @intCast(K);
    const ldb: i32 = if (trans_b) @intCast(K) else @intCast(N);
    const ldc: i32 = @intCast(M);

    const strideA: i64 = @intCast(M * K);
    const strideB: i64 = @intCast(K * N);
    const strideC: i64 = @intCast(M * N);

    const status = cublasGemmStridedBatchedEx(
        handle.raw,
        op_a, op_b,
        @intCast(M), @intCast(N), @intCast(K),
        &alpha,
        d_A.dev_ptr, CUDA_R_16F, lda, strideA,
        d_B.dev_ptr, CUDA_R_16F, ldb, strideB,
        &beta,
        d_C.dev_ptr, CUDA_R_32F, ldc, strideC,
        @intCast(batchCount),
        CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT,
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std.log.err("cublasGemmStridedBatchedEx failed: {}", .{status});
        return error.CuBlasBatchGemmExFailed;
    }

    // Download
    for (C_batch, 0..) |C, b| {
        const offset = b * M * N;
        const copy_status = cudaMemcpy(
            C.data.ptr,
            @ptrCast(@alignCast(@as([*]u8, @ptrCast(d_C.dev_ptr)) + offset * @sizeOf(f32))),
            C.data.len * @sizeOf(f32), cudaMemcpyDeviceToHost,
        );
        if (copy_status != 0) return error.CudaMemcpyFailed;
    }

    handle.sync();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Wrapper simplificado
// ═══════════════════════════════════════════════════════════════════════════════

pub fn gemmCuBlasSimple(
    handle: CuBlasHandle,
    A: Tensor(f32),
    B: Tensor(f32),
    C: *Tensor(f32),
    M: usize,
    N: usize,
    K: usize,
) !void {
    try gemmCuBlasF32(handle, A, B, C, M, N, K, false, false, 1.0, 0.0);
}


