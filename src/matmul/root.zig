
//! Motor Matmul en Zig — Interfaz pública (v2)
//!
//! Backends disponibles:
//!   .naive      — Correctitud, lento
//!   .simd       — SIMD nativo (@Vector), sin dependencias
//!   .tiled      — Tiling en caché + SIMD
//!   .parallel   — Multihilo + SIMD
//!   .openblas   — FFI a OpenBLAS (requiere -Dopenblas)
//!   .cublas     — FFI a cuBLAS (requiere -Dcublas)
//!
//! Nuevas funcionalidades v2:
//!   - FP16/BF16 conversiones y kernels CPU
//!   - Cuantización INT8/INT4 (simétrica, asimétrica, per-channel)
//!   - cuBLAS: streams async, batch GEMM, strided GEMM, mem pool persistente
//!   - cublasGemmEx: precisión mixta FP16/BF16 -> FP32 (Tensor Cores)

const std = @import("std");
const Tensor = @import("core").Tensor;
const build_options = @import("build_options");

const naive = @import("naive.zig");
const simd = @import("simd.zig");
const tiled = @import("tiled.zig");
const parallel = @import("parallel.zig");
const openblas = @import("openblas.zig");
const cublas = @import("cublas");

const types = @import("types.zig");
const f16bf16 = @import("f16bf16.zig");
const quant = @import("quant.zig");

pub const Backend = enum {
    naive,
    simd,
    tiled,
    parallel,
    openblas,
    cublas,
    auto,
};

pub const PrecisionMode = enum {
    f32,
    f16,
    bf16,
    int8,
    int4,
};

pub const MatmulEngine = struct {
    const Self = @This();

    // Stream CUDA compartido por TODOS los engines (y los kernels elementwise
    // de la capa híbrida residente). Así todas las GEMM y todos los kernels
    // corren en un mismo stream y quedan ordenados; se sincroniza una vez por
    // token. Se crea una sola vez y no se destruye (vive hasta el exit).
    var g_shared_stream: ?cublas.CudaStream = null;

    pub fn sharedCudaStream() !cublas.CudaStream {
        if (g_shared_stream) |s| return s;
        g_shared_stream = try cublas.CudaStream.create();
        return g_shared_stream.?;
    }

    allocator: std.mem.Allocator,
    backend: Backend,
    precision: PrecisionMode,
    num_threads: usize,
    cublas_handle: ?cublas.CuBlasHandle,
    cuda_stream: ?cublas.CudaStream,
    gpu_pool: ?cublas.GpuMemoryPool,
    /// Caché de pesos residentes en GPU: host_ptr(W) -> buffer device.
    /// Sube cada matriz de pesos UNA vez y la reusa en todos los tokens.
    weight_cache: ?std.AutoHashMap(usize, cublas.GpuBuffer(f32)),
    tile_config: types.TileConfig,

    pub fn init(allocator: std.mem.Allocator, preferred: Backend, precision: PrecisionMode) !Self {
        var engine = Self{
            .allocator = allocator,
            .backend = preferred,
            .precision = precision,
            .num_threads = 1,
            .cublas_handle = null,
            .cuda_stream = null,
            .gpu_pool = null,
            .weight_cache = null,
            .tile_config = types.TileConfig.default(),
        };

        if (preferred == .auto) {
            engine.backend = detectBestBackend();
        }

        switch (engine.backend) {
            .parallel => {
                const n_cpus = try std.Thread.getCpuCount();
                engine.num_threads = n_cpus;
            },
            .cublas => {
                if (!build_options.has_cuda) return error.CuBlasNotLinked;
                engine.cublas_handle = try cublas.CuBlasHandle.init();
                engine.cuda_stream = try sharedCudaStream();
                var gpu_pool = cublas.GpuMemoryPool.init(allocator);
                gpu_pool.setStream(engine.cuda_stream.?.raw);
                engine.gpu_pool = gpu_pool;
                if (engine.cublas_handle) |*h| try h.setStream(engine.cuda_stream.?.raw);
                engine.weight_cache = std.AutoHashMap(usize, cublas.GpuBuffer(f32)).init(allocator);
            },
            else => {},
        }

        return engine;
    }

    pub fn deinit(self: *Self) void {
        if (build_options.has_cuda) {
            if (self.weight_cache) |*cache| {
                var it = cache.valueIterator();
                while (it.next()) |entry| entry.*.free();
                cache.deinit();
            }
            // cuda_stream es compartido (g_shared_stream): no se destruye aquí.
            if (self.gpu_pool) |*pool| pool.deinit();
            if (self.cublas_handle) |handle| handle.deinit();
        }
    }

    // ─── GEMM general ───

    pub fn gemm(
        self: *Self,
        comptime T: type,
        A: Tensor(T),
        B: Tensor(T),
        C: *Tensor(T),
        trans_a: bool,
        trans_b: bool,
    ) !void {
        std.debug.assert(A.shape.len == 2 and B.shape.len == 2);
        const M = if (trans_a) A.shape[1] else A.shape[0];
        const KA = if (trans_a) A.shape[0] else A.shape[1];
        const KB = if (trans_b) B.shape[1] else B.shape[0];
        const N = if (trans_b) B.shape[0] else B.shape[1];
        std.debug.assert(KA == KB);
        std.debug.assert(C.shape[0] == M and C.shape[1] == N);
        const K = KA;

        switch (self.backend) {
            .naive => naive.gemmNaive(T, A, B, C, M, N, K, trans_a, trans_b, 1.0, 0.0),
            .simd => {
                if (comptime T != f32 and T != f64) {
                    naive.gemmNaive(T, A, B, C, M, N, K, trans_a, trans_b, 1.0, 0.0);
                } else if (trans_b) {
                    simd.gemmSimd(T, A, B, C, M, N, K);
                } else {
                    const Bt = try B.transpose();
                    defer { if (Bt.allocator) |a| { a.free(Bt.shape); a.free(Bt.strides); } }
                    simd.gemmSimd(T, A, Bt, C, M, N, K);
                }
            },
            .tiled => {
                if (comptime T != f32) {
                    naive.gemmNaive(T, A, B, C, M, N, K, trans_a, trans_b, 1.0, 0.0);
                } else if (trans_b) {
                    tiled.gemmTiled(T, A, B, C, M, N, K, self.tile_config);
                } else {
                    const Bt = try B.transpose();
                    defer { if (Bt.allocator) |a| { a.free(Bt.shape); a.free(Bt.strides); } }
                    tiled.gemmTiled(T, A, Bt, C, M, N, K, self.tile_config);
                }
            },
            .parallel => {
                if (self.num_threads == 0) return error.ThreadPoolNotInitialized;
                if (trans_b) {
                    try parallel.gemmParallel(T, self.allocator, A, B, C, M, N, K, .{
                        .num_threads = self.num_threads, .use_simd = true,
                    });
                } else {
                    const Bt = try B.transpose();
                    defer { if (Bt.allocator) |a| { a.free(Bt.shape); a.free(Bt.strides); } }
                    try parallel.gemmParallel(T, self.allocator, A, Bt, C, M, N, K, .{
                        .num_threads = self.num_threads, .use_simd = true,
                    });
                }
            },
            .openblas => if (build_options.has_openblas) openblas.gemmOpenBlas(T, A, B, C, M, N, K, trans_a, trans_b, 1.0, 0.0) else return error.OpenBlasNotLinked,
            .cublas => {
                if (!build_options.has_cuda) return error.CuBlasNotLinked;
                if (self.cublas_handle) |handle| {
                    if (T == f32) {
                        // Ruta síncrona simple (cudaMalloc + cudaMemcpy + cublasSgemm),
                        // validada contra CPU. El path async (cudaMallocAsync) aún no
                        // es fiable en este entorno.
                        try cublas.gemmCuBlasF32(handle, A, B, C, M, N, K, trans_a, trans_b, 1.0, 0.0);
                    } else if (T == f16) {
                        // Conversión a f32 + cublasSgemm (ruta bien soportada).
                        var a_f32 = try Tensor(f32).alloc(self.allocator, &.{ A.shape[0], A.shape[1] });
                        defer a_f32.deinit();
                        for (A.data, a_f32.data) |s, *d| d.* = @floatCast(s);
                        var b_f32 = try Tensor(f32).alloc(self.allocator, &.{ B.shape[0], B.shape[1] });
                        defer b_f32.deinit();
                        for (B.data, b_f32.data) |s, *d| d.* = @floatCast(s);
                        var c_f32 = try Tensor(f32).alloc(self.allocator, C.shape);
                        defer c_f32.deinit();
                        try cublas.gemmCuBlasF32(handle, a_f32, b_f32, &c_f32, M, N, K, trans_a, trans_b, 1.0, 0.0);
                        for (C.data, c_f32.data) |*d, s| d.* = @floatCast(s);
                    } else {
                        return error.CuBlasTypeNotSupported;
                    }
                } else return error.CuBlasNotInitialized;
            },
            .auto => unreachable,
        }
    }

    pub fn gemmNoTrans(self: *Self, comptime T: type, A: Tensor(T), B: Tensor(T), C: *Tensor(T)) !void {
        try self.gemm(T, A, B, C, false, false);
    }

    // ─── Proyección lineal ───

    pub fn linearProjection(self: *Self, comptime T: type, X: Tensor(T), W_T: Tensor(T), Y: *Tensor(T)) !void {
        std.debug.assert(X.shape.len == 2 and W_T.shape.len == 2);
        std.debug.assert(X.shape[1] == W_T.shape[1]);
        std.debug.assert(Y.shape[0] == X.shape[0]);
        std.debug.assert(Y.shape[1] == W_T.shape[0]);

        // Caché de pesos residentes en GPU: subir W_T una vez y reusarlo en
        // todos los tokens (el cuello de botella original era re-subir ~3GB de
        // pesos por token). Solo aplica a proyecciones lineales (B = peso
        // constante); la atención (Q@K^T) usa gemm() directo sin caché.
        if (self.backend == .cublas and self.weight_cache != null and self.cublas_handle != null) {
            const handle = self.cublas_handle.?;
            const M = X.shape[0];
            const K = X.shape[1];
            const N = W_T.shape[0];
            const key = @intFromPtr(W_T.data.ptr);
            if (T == f32) {
                if (self.weight_cache.?.get(key)) |d_B| {
                    try cublas.gemmCuBlasF32Resident(handle, X, d_B, Y, M, N, K, false, true, 1.0, 0.0);
                    return;
                } else {
                    var d_B = try cublas.GpuBuffer(f32).alloc(W_T.data.len);
                    try d_B.upload(W_T.data);
                    try self.weight_cache.?.put(key, d_B);
                    try cublas.gemmCuBlasF32Resident(handle, X, d_B, Y, M, N, K, false, true, 1.0, 0.0);
                    return;
                }
            } else if (T == f16) {
                var a_f32 = try Tensor(f32).alloc(self.allocator, &.{ X.shape[0], X.shape[1] });
                defer a_f32.deinit();
                for (X.data, a_f32.data) |s, *d| d.* = @floatCast(s);
                var c_f32 = try Tensor(f32).alloc(self.allocator, Y.shape);
                defer c_f32.deinit();
                if (self.weight_cache.?.get(key)) |d_B| {
                    try cublas.gemmCuBlasF32Resident(handle, a_f32, d_B, &c_f32, M, N, K, false, true, 1.0, 0.0);
                } else {
                    var b_f32 = try Tensor(f32).alloc(self.allocator, &.{ W_T.shape[0], W_T.shape[1] });
                    defer b_f32.deinit();
                    for (W_T.data, b_f32.data) |s, *d| d.* = @floatCast(s);
                    var d_B = try cublas.GpuBuffer(f32).alloc(b_f32.data.len);
                    try d_B.upload(b_f32.data);
                    try self.weight_cache.?.put(key, d_B);
                    try cublas.gemmCuBlasF32Resident(handle, a_f32, d_B, &c_f32, M, N, K, false, true, 1.0, 0.0);
                }
                for (Y.data, c_f32.data) |*d, s| d.* = @floatCast(s);
                return;
            }
        }

        try self.gemm(T, X, W_T, Y, false, true);
    }

    // ─── Proyección lineal GPU-resident (A y C ya en device) ───
    // X y Y son GpuTensor(f32); el peso W_T (host f32 dequantizado) se sube una
    // vez y se cachea. NO hay H2D de X ni D2H de Y (la salida queda en GPU).
    pub fn linearProjectionDevice(
        self: *Self,
        X: cublas.GpuTensor(f32),
        W_T: Tensor(f32),
        Y: *cublas.GpuTensor(f32),
        M: usize,
        K: usize,
        N: usize,
    ) !void {
        if (self.backend != .cublas or self.weight_cache == null or self.cublas_handle == null) {
            @panic("linearProjectionDevice requiere backend cublas");
        }
        const handle = self.cublas_handle.?;
        const key = @intFromPtr(W_T.data.ptr);
        const d_B = if (self.weight_cache.?.get(key)) |b| b else blk: {
            var buf = try cublas.GpuBuffer(f32).alloc(W_T.data.len);
            try buf.upload(W_T.data);
            try self.weight_cache.?.put(key, buf);
            break :blk buf;
        };
        try cublas.gemmCuBlasF32Device(handle, X.buf, d_B, Y.buf, M, N, K, false, true, 1.0, 0.0);
    }

    /// Proyección lineal device→device con peso f16 (p.ej. lm_head): X ya vive en
    /// GPU (f32), el peso se convierte a f32 y se cachea como en linearProjection;
    /// Y (f32) se escribe en GPU sin pasar por host.
    pub fn linearProjectionDeviceF16(
        self: *Self,
        X32: cublas.GpuTensor(f32),
        W_T16: Tensor(f16),
        Y32: *cublas.GpuTensor(f32),
        M: usize,
        K: usize,
        N: usize,
    ) !void {
        const handle = self.cublas_handle.?;
        const key = @intFromPtr(W_T16.data.ptr);
        const d_B = if (self.weight_cache.?.get(key)) |b| b else blk: {
            var b_f32 = try Tensor(f32).alloc(self.allocator, &.{ W_T16.shape[0], W_T16.shape[1] });
            defer b_f32.deinit();
            for (W_T16.data, b_f32.data) |s, *d| d.* = @floatCast(s);
            var buf = try cublas.GpuBuffer(f32).alloc(b_f32.data.len);
            try buf.upload(b_f32.data);
            try self.weight_cache.?.put(key, buf);
            break :blk buf;
        };
        try cublas.gemmCuBlasF32Device(handle, X32.buf, d_B, Y32.buf, M, N, K, false, true, 1.0, 0.0);
    }

    // ─── FFN SwiGLU ───

    pub fn ffnProjections(self: *Self, comptime T: type, X: Tensor(T), W_gate_T: Tensor(T), W_up_T: Tensor(T), gate_out: *Tensor(T), up_out: *Tensor(T)) !void {
        try self.linearProjection(T, X, W_gate_T, gate_out);
        try self.linearProjection(T, X, W_up_T, up_out);
    }

    // ─── GEMM cuantizado (INT8) ───

    pub fn gemmQuantized(self: *Self, A: Tensor(f32), B_q: QuantizedTensor, C: *Tensor(f32), M: usize, N: usize, K: usize) !void {
        _ = self;
        quant.gemmWithQuantizedB(A, B_q, C, M, N, K);
    }

    // ─── Batch GEMM (cuBLAS) ───
    pub fn gemmBatch(self: *Self, A_batch: []const Tensor(f32), B_batch: []const Tensor(f32), C_batch: []const *Tensor(f32), M: usize, N: usize, K: usize) !void {
        if (self.backend != .cublas) return error.BatchGemmRequiresCuBlas;
        if (self.cublas_handle) |handle| {
            try cublas.gemmBatchF32(handle, A_batch, B_batch, C_batch, M, N, K, false, false, 1.0, 0.0);
        } else return error.CuBlasNotInitialized;
    }

    // ─── Strided GEMM para GQA/MQA ───

    pub fn gemmStrided(self: *Self, A_flat: []const f32, B_flat: []const f32, C_flat: []f32, M: usize, N: usize, K: usize, batchCount: usize, strideA: i64, strideB: i64, strideC: i64) !void {
        if (self.backend != .cublas) return error.StridedGemmRequiresCuBlas;
        if (self.cublas_handle) |handle| {
            try cublas.gemmStridedF32(handle, A_flat, B_flat, C_flat, M, N, K, batchCount, strideA, strideB, strideC, false, false, 1.0, 0.0);
        } else return error.CuBlasNotInitialized;
    }

    // ─── GEMM con precisión mixta (cuBLAS) ───

    pub fn gemmExF16(self: *Self, A: Tensor(f16), B: Tensor(f16), C: *Tensor(f32), M: usize, N: usize, K: usize) !void {
        if (self.backend != .cublas) return error.GemmExRequiresCuBlas;
        if (self.cublas_handle) |handle| {
            try cublas.gemmExF16F32(handle, A, B, C, M, N, K, false, false, 1.0, 0.0);
        } else return error.CuBlasNotInitialized;
    }

    pub fn gemmExBF16(self: *Self, A: Tensor(u16), B: Tensor(u16), C: *Tensor(f32), M: usize, N: usize, K: usize) !void {
        if (self.backend != .cublas) return error.GemmExRequiresCuBlas;
        if (self.cublas_handle) |handle| {
            try cublas.gemmExBF16F32(handle, A, B, C, M, N, K, false, false, 1.0, 0.0);
        } else return error.CuBlasNotInitialized;
    }

    pub fn gemmBatchExF16(self: *Self, A_batch: []const Tensor(f16), B_batch: []const Tensor(f16), C_batch: []const *Tensor(f32), M: usize, N: usize, K: usize, batchCount: usize) !void {
        if (self.backend != .cublas) return error.BatchGemmExRequiresCuBlas;
        if (self.cublas_handle) |handle| {
            try cublas.gemmBatchExF16F32(handle, A_batch, B_batch, C_batch, M, N, K, batchCount, false, false, 1.0, 0.0);
        } else return error.CuBlasNotInitialized;
    }

    // ─── Configuración ───

    pub fn setTileConfig(self: *Self, config: types.TileConfig) void {
        self.tile_config = config;
    }

    pub fn backendName(self: Self) []const u8 {
        return switch (self.backend) {
            .naive => "naive", .simd => "simd", .tiled => "tiled",
            .parallel => "parallel", .openblas => "openblas",
            .cublas => "cublas", .auto => "auto",
        };
    }

    pub const PoolStats = cublas.GpuMemoryPool.PoolStats;

    pub fn gpuPoolStats(self: Self) ?PoolStats {
        if (self.gpu_pool) |pool| return pool.stats();
        return null;
    }
};

fn detectBestBackend() Backend {
    const n_cpus = std.Thread.getCpuCount() catch 1;
    if (n_cpus > 1) return .parallel;
    if (types.SimdInfo.has_simd) return .simd;
    return .naive;
}

// Re-exports
pub const TileConfig = types.TileConfig;
pub const Timer = types.Timer;
pub const tensorsApproxEq = types.tensorsApproxEq;
pub const BF16 = f16bf16.BF16;
pub const F16 = f16bf16.F16;
pub const tensorF32ToF16 = f16bf16.tensorF32ToF16;
pub const tensorF16ToF32 = f16bf16.tensorF16ToF32;
pub const tensorF32ToBF16 = f16bf16.tensorF32ToBF16;
pub const tensorBF16ToF32 = f16bf16.tensorBF16ToF32;
pub const QuantConfig = quant.QuantConfig;
pub const QuantizedTensor = quant.QuantizedTensor;
pub const quantizeInt8Symmetric = quant.quantizeInt8Symmetric;
pub const quantizeInt8Asymmetric = quant.quantizeInt8Asymmetric;
pub const quantizeInt8PerChannel = quant.quantizeInt8PerChannel;
pub const quantizeInt4Symmetric = quant.quantizeInt4Symmetric;
pub const dequantizeToF32 = quant.dequantizeToF32;
pub const gemmWithQuantizedB = quant.gemmWithQuantizedB;


