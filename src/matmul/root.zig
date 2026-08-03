
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

const naive = @import("naive.zig");
const simd = @import("simd.zig");
const tiled = @import("tiled.zig");
const parallel = @import("parallel.zig");
const openblas = @import("openblas.zig");
const cublas = @import("cublas.zig");
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

pub const MatmulEngine = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    backend: Backend,
    thread_pool: ?std.Thread.Pool,
    cublas_handle: ?cublas.CuBlasHandle,
    cuda_stream: ?cublas.CudaStream,
    gpu_pool: ?cublas.GpuMemoryPool,
    tile_config: types.TileConfig,

    pub fn init(allocator: std.mem.Allocator, preferred: Backend) !Self {
        var engine = Self{
            .allocator = allocator,
            .backend = preferred,
            .thread_pool = null,
            .cublas_handle = null,
            .cuda_stream = null,
            .gpu_pool = null,
            .tile_config = types.TileConfig.default(),
        };

        if (preferred == .auto) {
            engine.backend = detectBestBackend();
        }

        switch (engine.backend) {
            .parallel => {
                var pool: std.Thread.Pool = undefined;
                const n_cpus = try std.Thread.getCpuCount();
                try pool.init(.{ .allocator = allocator, .n_jobs = n_cpus });
                engine.thread_pool = pool;
            },
            .cublas => {
                engine.cublas_handle = try cublas.CuBlasHandle.init();
                engine.cuda_stream = try cublas.CudaStream.create();
                var gpu_pool = cublas.GpuMemoryPool.init(allocator);
                gpu_pool.setStream(engine.cuda_stream.?.raw);
                engine.gpu_pool = gpu_pool;
                try engine.cublas_handle.?.setStream(engine.cuda_stream.?.raw);
            },
            else => {},
        }

        return engine;
    }

    pub fn deinit(self: *Self) void {
        if (self.thread_pool) |*pool| pool.deinit();
        if (self.cuda_stream) |stream| stream.destroy();
        if (self.gpu_pool) |*pool| pool.deinit();
        if (self.cublas_handle) |handle| handle.deinit();
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
                if (trans_b) {
                    simd.gemmSimd(T, A, B, C, M, N, K);
                } else {
                    var Bt = try B.transpose();
                    defer { if (Bt.allocator) |a| { a.free(Bt.shape); a.free(Bt.strides); } }
                    simd.gemmSimd(T, A, Bt, C, M, N, K);
                }
            },
            .tiled => {
                if (trans_b) {
                    tiled.gemmTiled(T, A, B, C, M, N, K, self.tile_config);
                } else {
                    var Bt = try B.transpose();
                    defer { if (Bt.allocator) |a| { a.free(Bt.shape); a.free(Bt.strides); } }
                    tiled.gemmTiled(T, A, Bt, C, M, N, K, self.tile_config);
                }
            },
            .parallel => {
                if (self.thread_pool) |*pool| {
                    if (trans_b) {
                        try parallel.gemmParallel(T, pool, A, B, C, M, N, K, .{
                            .num_threads = pool.threads.len, .use_simd = true,
                        });
                    } else {
                        var Bt = try B.transpose();
                        defer { if (Bt.allocator) |a| { a.free(Bt.shape); a.free(Bt.strides); } }
                        try parallel.gemmParallel(T, pool, A, Bt, C, M, N, K, .{
                            .num_threads = pool.threads.len, .use_simd = true,
                        });
                    }
                } else return error.ThreadPoolNotInitialized;
            },
            .openblas => openblas.gemmOpenBlas(T, A, B, C, M, N, K, trans_a, trans_b, 1.0, 0.0),
            .cublas => {
                if (self.cublas_handle) |handle| {
                    if (T == f32) {
                        if (self.cuda_stream != null and self.gpu_pool != null) {
                            try cublas.gemmCuBlasF32Async(handle, self.cuda_stream.?, &self.gpu_pool.?, A, B, C, M, N, K, trans_a, trans_b, 1.0, 0.0);
                            self.cuda_stream.?.synchronize();
                        } else {
                            try cublas.gemmCuBlasF32(handle, A, B, C, M, N, K, trans_a, trans_b, 1.0, 0.0);
                        }
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
        try self.gemm(T, X, W_T, Y, false, false);
    }

    // ─── FFN SwiGLU ───

    pub fn ffnProjections(self: *Self, comptime T: type, X: Tensor(T), W_gate_T: Tensor(T), W_up_T: Tensor(T), gate_out: *Tensor(T), up_out: *Tensor(T)) !void {
        try self.linearProjection(T, X, W_gate_T, gate_out);
        try self.linearProjection(T, X, W_up_T, up_out);
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

    pub fn gpuPoolStats(self: Self) ?struct { total: usize, used: usize, free: usize } {
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


