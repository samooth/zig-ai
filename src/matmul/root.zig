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
const fp8_kernels = @import("fp8_kernels");
const cudaz = @import("cudaz");
const ext_mem = @import("cudaz_ext_mem");
const graph_capture = @import("graph_capture");
const debugz = @import("debug");
// Post-freeze 2026-09-13: sizing del pool parallel vía presupuesto central
// (físicos SMT-collapsed + loadavg + ZIG_AI_CPU_WORKERS global), no
// getCpuCount()=lógicos — varios lanes con pools de 16 congelaban el host.
const resources = @import("resources");

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
    fp8_block,
    auto,
};

pub const PrecisionMode = enum {
    f32,
    f16,
    bf16,
    int8,
    int4,
    fp8,
};

/// 2.2 (lane-f): routing FP8 end-to-end (CLI --quant fp8). true ⇒
/// linearProjectionDevice enruta las proyecciones al camino block-scaled
/// E4M3 con engine fp8 LAZY (el backend del MatmulEngine sigue siendo
/// cuBLAS para el resto: KV append, FA, gemm host). v1 con backend
/// .fp8_block global rompía las capas (cada AttentionLayer crea su
/// engine y perdía el KV path — KvCacheNotSet).
pub var fp8_route_enabled: bool = false;

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

    // ─── Lane D (D3): stream de COPIA dedicado + eventos ───
    // Distinto de sharedCudaStream(): permite H2D async solapado con el
    // cómputo (doble búfer copy↔compute, Contrato 5). Lazy, uno para todo
    // el proceso; vive hasta el exit igual que el stream principal.

    /// Tick monotónico para LRU del weight_cache (D3): cada touch/get lo
    /// incrementa; la víctima LRU es la de menor last_used.
    var g_cache_tick = std.atomic.Value(u64).init(0);

    var g_copy_stream: ?cublas.CudaStream = null;

    /// Stream de copia dedicado (≠ sharedCudaStream). REGLA ADITIVA: no
    /// sustituye al stream compartido; los consumidores existentes siguen
    /// viendo exactamente el mismo stream de siempre.
    pub fn copyStream() !cublas.CudaStream {
        if (g_copy_stream) |s| return s;
        if (!build_options.has_cuda) return error.CuBlasNotLinked;
        g_copy_stream = try cublas.CudaStream.create();
        return g_copy_stream.?;
    }

    /// Mismo stream como handle Driver API (CUstream) para consumidores que
    /// lanzan con cudaz.cuMemcpyHtoDAsync / kernels propios (E/F).
    pub fn copyStreamRaw() !cudaz.CUstream {
        const s = try copyStream();
        return @ptrCast(s.raw);
    }

    /// Evento listo para record/wait entre streams (ready/release del doble
    /// búfer). El caller lo destruye con destroyCopyEvent.
    pub fn createCopyEvent() !cudaz.CUevent {
        return cudaz.cuEventCreate(0);
    }

    pub fn destroyCopyEvent(ev: cudaz.CUevent) void {
        cudaz.cuEventDestroy(ev);
    }

    /// Record en el stream de copia (ready tras un H2D async).
    pub fn recordOnCopyStream(ev: cudaz.CUevent) !void {
        try cudaz.cuEventRecord(ev, try copyStreamRaw());
    }

    /// Wait desde el stream de compute (el compute espera la copia).
    pub fn computeWaitsEvent(ev: cudaz.CUevent) !void {
        try ext_mem.streamWaitEvent(@ptrCast(try sharedCudaStreamRaw()), @ptrCast(ev), 0);
    }

    pub fn sharedCudaStreamRaw() !cudaz.CUstream {
        const s = try sharedCudaStream();
        return @ptrCast(s.raw);
    }

    allocator: std.mem.Allocator,
    backend: Backend,
    precision: PrecisionMode,
    num_threads: usize,
    cublas_handle: ?cublas.CuBlasHandle,
    cuda_stream: ?cublas.CudaStream,
    gpu_pool: ?cublas.GpuMemoryPool,
    /// Caché de pesos residentes en GPU: host_ptr(W) -> entrada {buffer, tick}.
    /// Sube cada matriz de pesos UNA vez y la reusa en todos los tokens.
    /// D3: entrada con last_used para LRU — sin eviction, dtypes sin kernel
    /// cuantizado acumulaban f32 en VRAM sin techo (causa raíz 27B, ticket A).
    weight_cache: ?std.AutoHashMap(usize, WeightCacheEntry),
    tile_config: types.TileConfig,
    fp8_engine: ?fp8_kernels.Fp8BlockLinear,
    /// 2.1 (lane-f): scratch FP8 del gemmFp8Block — activaciones cuantizadas
    /// [M,K] u8 + escalas [M,K/128] f32, allocados lazy por geometría y
    /// REUTILIZADOS (el decode repite M=1 con la misma K constantemente).
    fp8_act_buf: ?cublas.GpuBuffer(u8) = null,
    fp8_act_scale_buf: ?cublas.GpuBuffer(f32) = null,
    fp8_act_m: usize = 0,
    fp8_act_k: usize = 0,
    /// Cache de pesos FP8: host_ptr(W f32) → {fp8 [N,K] u8, scales [N,K/128]
    /// f32} en device. Cuantiza UNA vez on-first-use (2.1: el missing piece —
    /// gemmFp8Block recibía W_fp8/W_scales que NADIE producía).
    fp8_weight_cache: ?std.AutoHashMap(usize, Self.Fp8WeightEntry) = null,

    /// 2.1 (lane-f): entrada del cache de pesos FP8 (device buffers).
    pub const Fp8WeightEntry = struct {
        fp8: cublas.GpuBuffer(u8),
        scales: cublas.GpuBuffer(f32),
        n: usize,
        k: usize,
    };

    /// 2.2 (lane-f): engine FP8 LAZY para el routing de proyecciones.
    /// Inicializa fp8_engine + fp8_weight_cache + scratch una sola vez
    /// (primera proyección con fp8_route_enabled). Usa el MISMO shared
    /// cuda_stream del engine cublas — sin streams paralelos.
    fn fp8_engine_opt(self: *Self) ?*fp8_kernels.Fp8BlockLinear {
        if (!fp8_route_enabled) return null;
        if (!build_options.has_cuda) return null;
        if (self.cuda_stream == null) return null; // engine CPU: sin routing
        if (self.fp8_engine == null) {
            self.fp8_engine = fp8_kernels.Fp8BlockLinear.init(@ptrCast(self.cuda_stream.?.raw)) catch return null;
            self.fp8_weight_cache = std.AutoHashMap(usize, Self.Fp8WeightEntry).init(self.allocator);
        }
        return &self.fp8_engine.?;
    }

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
            .fp8_engine = null,
        };

        if (preferred == .auto) {
            engine.backend = detectBestBackend();
        }

        switch (engine.backend) {
            .parallel => {
                // Presupuesto central (2026-09-13): físicos−1 modulado por
                // load externo / ZIG_AI_CPU_WORKERS / cap tests (≤4). Antes:
                // getCpuCount() = TODOS los lógicos — origen del freeze.
                engine.num_threads = resources.computeThreadBudget(engine.allocator, "matmul_parallel");
            },
            .cublas => {
                if (!build_options.has_cuda) return error.CuBlasNotLinked;
                engine.cublas_handle = try cublas.CuBlasHandle.init();
                engine.cuda_stream = try sharedCudaStream();
                var gpu_pool = cublas.GpuMemoryPool.init(allocator);
                gpu_pool.setStream(engine.cuda_stream.?.raw);
                engine.gpu_pool = gpu_pool;
                if (engine.cublas_handle) |*h| try h.setStream(engine.cuda_stream.?.raw);
                engine.weight_cache = std.AutoHashMap(usize, WeightCacheEntry).init(allocator);
            },
            .fp8_block => {
                if (!build_options.has_cuda) return error.CuBlasNotLinked;
                engine.cuda_stream = try sharedCudaStream();
                engine.fp8_engine = try fp8_kernels.Fp8BlockLinear.init(@ptrCast(engine.cuda_stream.?.raw));
                engine.fp8_weight_cache = std.AutoHashMap(usize, Self.Fp8WeightEntry).init(allocator);
            },
            else => {},
        }

        return engine;
    }

    pub fn deinit(self: *Self) void {
        if (build_options.has_cuda) {
            if (self.weight_cache) |*cache| {
                var it = cache.valueIterator();
                while (it.next()) |entry| entry.*.buf.free();
                cache.deinit();
            }
            // 2.1 (lane-f): frees del camino FP8.
            if (self.fp8_weight_cache) |*cache| {
                var it = cache.valueIterator();
                while (it.next()) |entry| {
                    entry.*.fp8.free();
                    entry.*.scales.free();
                }
                cache.deinit();
            }
            if (self.fp8_act_buf) |*b| b.free();
            if (self.fp8_act_scale_buf) |*b| b.free();
            // cuda_stream es compartido (g_shared_stream): no se destruye aquí.
            if (self.gpu_pool) |*pool| pool.deinit();
            if (self.cublas_handle) |handle| handle.deinit();
        }
    }

    /// Libera el caché de pesos GPU (host_ptr(W) -> buffer device). Debe llamarse
    /// al descargar pesos (LRU eviction): los scratch f32 host se liberan y el
    /// allocator puede reutilizar la misma dirección para otro peso, lo que haría
    /// que el caché (keyed por host_ptr) devuelva el buffer equivocado.
    pub fn clearWeightCache(self: *Self) void {
        if (build_options.has_cuda) {
            if (self.weight_cache) |*cache| {
                var it = cache.valueIterator();
                while (it.next()) |entry| entry.*.buf.free();
                cache.clearRetainingCapacity();
            }
        }
    }

    // ─── Lane D (D3): hooks de evicción LRU del weight_cache ───
    // Ticket de lane-a (T1): sin eviction, cualquier dtype sin kernel
    // cuantizado acumulaba f32 en VRAM sin techo. API ADITIVA: las firmas
    // que consume hybrid_attn (projectionDevicePtr/linearProjectionDevice*)
    // NO cambian; el touch LRU es transparente en los get.

    pub const WeightCacheEntry = struct {
        buf: cublas.GpuBuffer(f32),
        last_used: u64,
    };

    fn cacheTick() u64 {
        return g_cache_tick.fetchAdd(1, .monotonic);
    }

    /// Renueva la frescura LRU de una entrada tras un uso (get de cualquier
    /// proyección). Sin efecto si la entrada no existe.
    pub fn touchWeightCache(self: *Self, host_ptr: usize) void {
        if (self.weight_cache) |*cache| {
            if (cache.getPtr(host_ptr)) |e| e.last_used = cacheTick();
        }
    }

    /// Bytes device ocupados por el cache.
    pub fn weightCacheBytes(self: *Self) usize {
        var total: usize = 0;
        if (self.weight_cache) |*cache| {
            var it = cache.valueIterator();
            while (it.next()) |e| total += e.buf.len * @sizeOf(f32);
        }
        return total;
    }

    /// Entradas residentes.
    pub fn weightCacheCount(self: *Self) usize {
        if (self.weight_cache) |*c| return c.count();
        return 0;
    }

    /// Expulsa UNA entrada por host_ptr (p.ej. al unload de una capa).
    /// Idempotente: ptr ausente ⇒ 0 bytes liberados.
    /// Devuelve bytes device liberados.
    pub fn evictWeightCachePtr(self: *Self, host_ptr: usize) usize {
        if (self.weight_cache) |*cache| {
            if (cache.fetchRemove(host_ptr)) |kv| {
                const bytes = kv.value.buf.len * @sizeOf(f32);
                kv.value.buf.free();
                debugz.dbg.printLevel(.detail, "[matmul] weight_cache evict ptr={x} ({d:.1} MB)\n", .{ host_ptr, @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0) });
                return bytes;
            }
        }
        return 0;
    }

    // ─── Lane D (D4): caché de pesos CUANTIZADOS residentes (bytes crudos) ───
    pub const QuantResidentEntry = struct {
        dev: cudaz.CUdeviceptr,
        bytes: usize,
    };

    var g_quant_cache: ?std.AutoHashMap(usize, QuantResidentEntry) = null;
    var g_quant_bytes_total: std.atomic.Value(usize) = std.atomic.Value(usize).init(0);
    /// Bytes H2D REALES (DMA ocurrido); los hits de cache no suman.
    var g_quant_dma_bytes: std.atomic.Value(usize) = std.atomic.Value(usize).init(0);

    /// Bytes H2D reales acumulados por el cache cuantizado desde el inicio.
    pub fn quantDmaBytes() usize {
        return g_quant_dma_bytes.load(.monotonic);
    }

    /// Bytes device ocupados por pesos cuantizados residentes.
    pub fn quantCacheBytes() usize {
        return g_quant_bytes_total.load(.monotonic);
    }

    /// ¿Ya está residente este bloque (por dirección host)? Para contabilidad
    /// de bytes H2D real: hit ⇒ no hay DMA.
    pub fn quantIsResident(bytes: []const u8) bool {
        const c = &(g_quant_cache orelse return false);
        return c.contains(@intFromPtr(bytes.ptr));
    }

    /// Sube (solo la primera vez) los bytes crudos y devuelve el puntero device.
    pub fn quantResidentPtr(allocator: std.mem.Allocator, bytes: []const u8) !usize {
        if (g_quant_cache == null) g_quant_cache = std.AutoHashMap(usize, QuantResidentEntry).init(allocator);
        if (g_quant_cache.?.get(@intFromPtr(bytes.ptr))) |e| return e.dev;
        const dev = try cudaz.cuMemAlloc(bytes.len);
        errdefer cudaz.cuMemFree(dev);
        try cudaz.cuMemcpyHtoD(dev, @intFromPtr(bytes.ptr), bytes.len);
        try g_quant_cache.?.put(@intFromPtr(bytes.ptr), .{ .dev = dev, .bytes = bytes.len });
        _ = g_quant_bytes_total.fetchAdd(bytes.len, .monotonic);
        _ = g_quant_dma_bytes.fetchAdd(bytes.len, .monotonic);
        debugz.dbg.printLevel(.detail, "[matmul] quant_resident subido {d:.2} MB (total {d:.1} MB)\n", .{
            @as(f64, @floatFromInt(bytes.len)) / (1024.0 * 1024.0),
            @as(f64, @floatFromInt(quantCacheBytes())) / (1024.0 * 1024.0),
        });
        return dev;
    }

    /// Expulsa UNA entrada por CLAVE HOST (@intFromPtr(w.bytes.ptr), la misma
    /// que se pasó como fuente a quantResidentPtr). Idempotente; devuelve
    /// bytes liberados.
    ///
    /// ¡¡OJO!! NO pasar aquí el puntero DEVICE que devuelve quantResidentPtr:
    /// son espacios de dirección distintos y fetchRemove fallaría en silencio
    /// (bug del harness stream_bench root-caused 2026-08-27).
    pub fn evictQuantCachePtr(host_ptr: usize) usize {
        if (g_quant_cache) |*cache| {
            if (cache.fetchRemove(host_ptr)) |kv| {
                cudaz.cuMemFree(kv.value.dev);
                _ = g_quant_bytes_total.fetchSub(kv.value.bytes, .monotonic);
                return kv.value.bytes;
            }
        }
        return 0;
    }

    /// Expulsa TODAS las entradas cuantizadas (transición de capa del
    /// streamer: lo simple-correcto según ticket T1; re-upload por switch
    /// ≈100-300 MB ≈ 15-45 ms por PCIe — aceptable v1).
    pub fn evictQuantCacheAll() void {
        if (g_quant_cache) |*cache| {
            var it = cache.iterator();
            while (it.next()) |kv| {
                cudaz.cuMemFree(kv.value_ptr.dev);
                _ = g_quant_bytes_total.fetchSub(kv.value_ptr.bytes, .monotonic);
            }
            cache.clearRetainingCapacity();
        }
    }

    /// Reset del contador DMA (para delimitar ventanas de medición).
    pub fn quantDmaBytesReset() void {
        g_quant_dma_bytes.store(0, .monotonic);
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
                    defer {
                        if (Bt.allocator) |a| {
                            a.free(Bt.shape);
                            a.free(Bt.strides);
                        }
                    }
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
                    defer {
                        if (Bt.allocator) |a| {
                            a.free(Bt.shape);
                            a.free(Bt.strides);
                        }
                    }
                    tiled.gemmTiled(T, A, Bt, C, M, N, K, self.tile_config);
                }
            },
            .parallel => {
                if (self.num_threads == 0) return error.ThreadPoolNotInitialized;
                if (trans_b) {
                    try parallel.gemmParallel(T, self.allocator, A, B, C, M, N, K, .{
                        .num_threads = self.num_threads,
                        .use_simd = true,
                    });
                } else {
                    const Bt = try B.transpose();
                    defer {
                        if (Bt.allocator) |a| {
                            a.free(Bt.shape);
                            a.free(Bt.strides);
                        }
                    }
                    try parallel.gemmParallel(T, self.allocator, A, Bt, C, M, N, K, .{
                        .num_threads = self.num_threads,
                        .use_simd = true,
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
            .fp8_block => {
                // 2.2 (lane-f): gemm() genérico host-side NO es el camino
                // FP8 (las proyecciones van por linearProjectionDevice →
                // gemmFp8Block). Los pocos gemm host que quedan (pipeline
                // legacy CPU, tests) caen al NAIVE — antes esto devolvía
                // InvalidParameter y mataba el engine al primer prefill.
                naive.gemmNaive(T, A, B, C, M, N, K, trans_a, trans_b, 1.0, 0.0);
            },
            .auto => unreachable,
        }
    }

    // ─── FP8 Block-Scaled GEMM ───
    // Requiere backend .fp8_block y precision .fp8
    // ─── FP8 Block-Scaled GEMM (2.1, lane-f) ─────────────────────────────
    // FIRMA REPARADA: el original pedía W_fp8/W_scales "pre-cuantizados" que
    // NADIE producía (código muerto con 2 bugs latentes: llamada
    // quantizeActivations(A) incompatible con el wrapper y asserts de layout
    // de escalas [N/128,K/128] ≠ [N,K/128] del kernel). Ahora: toma el peso
    // W_T f32 [N,K] directo, lo cuantiza ON-DEMAND con cache
    // (fp8_weight_cache, una vez por peso) y las activaciones con scratch
    // reutilizado por geometría. Requiere backend .fp8_block y K%128==0.
    pub fn gemmFp8Block(
        self: *Self,
        A: Tensor(f32), // activations [M, K] f32 host
        W_T: Tensor(f32), // weights [N, K] f32 host (filas = salida)
        C: *Tensor(f32), // output [M, N] f32 host
    ) !void {
        if (self.backend != .fp8_block) return error.BackendMismatch;
        const engine = &(self.fp8_engine orelse return error.Fp8EngineNotInitialized);
        const M = A.shape[0];
        const K = A.shape[1];
        const N = W_T.shape[0];
        if (K % 128 != 0) return error.InvalidParameter; // grupos de 128
        if (W_T.shape[1] != K) return error.InvalidParameter;
        if (C.shape[0] != M or C.shape[1] != N) return error.InvalidParameter;

        // ── Pesos: cuantizar UNA vez (cache por host_ptr) ──
        const w_key = @intFromPtr(W_T.data.ptr);
        const w_entry: *Self.Fp8WeightEntry = blk: {
            if (self.fp8_weight_cache.?.getPtr(w_key)) |hit| break :blk hit;
            // Subida f32 temporal → cuant kernel → buffers FP8 residentes.
            const w_f32_dev = try cublas.GpuBuffer(f32).alloc(N * K);
            defer w_f32_dev.free();
            try w_f32_dev.upload(W_T.data);
            const fp8_buf = try cublas.GpuBuffer(u8).alloc(N * K);
            const scales_buf = try cublas.GpuBuffer(f32).alloc(N * (K / 128));
            try engine.quantizeActivationsF32( // mismo kernel: fila="token"
                @intFromPtr(w_f32_dev.dev_ptr),
                @intFromPtr(fp8_buf.dev_ptr),
                @intFromPtr(scales_buf.dev_ptr),
                @intCast(N),
                @intCast(K),
            );
            // Sincronizar ANTES de liberar w_f32_dev (el kernel lo lee async).
            try cudaz.cuStreamSynchronize(@ptrCast(self.cuda_stream.?.raw));
            try self.fp8_weight_cache.?.put(w_key, .{
                .fp8 = fp8_buf,
                .scales = scales_buf,
                .n = N,
                .k = K,
            });
            break :blk self.fp8_weight_cache.?.getPtr(w_key).?;
        };

        // ── Activaciones: scratch reutilizado por geometría (M,K) ──
        if (self.fp8_act_buf == null or self.fp8_act_m != M or self.fp8_act_k != K) {
            if (self.fp8_act_buf) |*b| b.free();
            if (self.fp8_act_scale_buf) |*b| b.free();
            self.fp8_act_buf = try cublas.GpuBuffer(u8).alloc(M * K);
            self.fp8_act_scale_buf = try cublas.GpuBuffer(f32).alloc(M * (K / 128));
            self.fp8_act_m = M;
            self.fp8_act_k = K;
        }
        const a_f32_dev = try cublas.GpuBuffer(f32).alloc(M * K);
        defer a_f32_dev.free();
        try a_f32_dev.upload(A.data);
        try engine.quantizeActivationsF32(
            @intFromPtr(a_f32_dev.dev_ptr),
            @intFromPtr(self.fp8_act_buf.?.dev_ptr),
            @intFromPtr(self.fp8_act_scale_buf.?.dev_ptr),
            @intCast(M),
            @intCast(K),
        );

        // ── Salida device + memset (el gemm acumula con atomicAdd) ──
        const c_dev = try cublas.GpuBuffer(u8).alloc(M * N * @sizeOf(f32));
        defer c_dev.free();
        try cudaz.cuMemsetD8(@intFromPtr(c_dev.dev_ptr), 0, M * N * @sizeOf(f32));
        try engine.gemm(
            @intFromPtr(self.fp8_act_buf.?.dev_ptr),
            @intFromPtr(self.fp8_act_scale_buf.?.dev_ptr),
            @intFromPtr(w_entry.fp8.dev_ptr),
            @intFromPtr(w_entry.scales.dev_ptr),
            @intFromPtr(c_dev.dev_ptr),
            @intCast(M),
            @intCast(N),
            @intCast(K),
        );
        try cudaz.cuStreamSynchronize(@ptrCast(self.cuda_stream.?.raw));
        try cudaz.cuMemcpyDtoH(@intFromPtr(C.data.ptr), @intFromPtr(c_dev.dev_ptr), M * N * @sizeOf(f32));
    }

    /// 2.1 (lane-f): GEMV FP8 (M=1 decode) — split-K sobre K, mismo cache de
    /// pesos que gemmFp8Block. num_splits recomendado = min(K/128, 4-8).
    pub fn gemvFp8Block(
        self: *Self,
        x: Tensor(f32), // [K] f32 host
        W_T: Tensor(f32), // [N, K] f32 host
        out: Tensor(f32), // [N] f32 host
        num_splits: usize,
    ) !void {
        if (self.backend != .fp8_block) return error.BackendMismatch;
        const engine = &(self.fp8_engine orelse return error.Fp8EngineNotInitialized);
        const K = x.shape[x.shape.len - 1];
        const N = W_T.shape[0];
        if (K % 128 != 0) return error.InvalidParameter;
        if (W_T.shape[1] != K or out.data.len != N) return error.InvalidParameter;

        // Pesos: mismo cache que gemmFp8Block (comparte fp8_weight_cache).
        const w_key = @intFromPtr(W_T.data.ptr);
        const w_entry: *Self.Fp8WeightEntry = blk: {
            if (self.fp8_weight_cache.?.getPtr(w_key)) |hit| break :blk hit;
            const w_f32_dev = try cublas.GpuBuffer(f32).alloc(N * K);
            defer w_f32_dev.free();
            try w_f32_dev.upload(W_T.data);
            const fp8_buf = try cublas.GpuBuffer(u8).alloc(N * K);
            const scales_buf = try cublas.GpuBuffer(f32).alloc(N * (K / 128));
            try engine.quantizeActivationsF32(
                @intFromPtr(w_f32_dev.dev_ptr),
                @intFromPtr(fp8_buf.dev_ptr),
                @intFromPtr(scales_buf.dev_ptr),
                @intCast(N),
                @intCast(K),
            );
            try cudaz.cuStreamSynchronize(@ptrCast(self.cuda_stream.?.raw));
            try self.fp8_weight_cache.?.put(w_key, .{
                .fp8 = fp8_buf,
                .scales = scales_buf,
                .n = N,
                .k = K,
            });
            break :blk self.fp8_weight_cache.?.getPtr(w_key).?;
        };

        // Activación [1, K] + salida memset (atomicAdd en split-K).
        if (self.fp8_act_buf == null or self.fp8_act_m != 1 or self.fp8_act_k != K) {
            if (self.fp8_act_buf) |*b| b.free();
            if (self.fp8_act_scale_buf) |*b| b.free();
            self.fp8_act_buf = try cublas.GpuBuffer(u8).alloc(K);
            self.fp8_act_scale_buf = try cublas.GpuBuffer(f32).alloc(K / 128);
            self.fp8_act_m = 1;
            self.fp8_act_k = K;
        }
        const x_f32_dev = try cublas.GpuBuffer(f32).alloc(K);
        defer x_f32_dev.free();
        try x_f32_dev.upload(x.data);
        try engine.quantizeActivationsF32(
            @intFromPtr(x_f32_dev.dev_ptr),
            @intFromPtr(self.fp8_act_buf.?.dev_ptr),
            @intFromPtr(self.fp8_act_scale_buf.?.dev_ptr),
            1,
            @intCast(K),
        );
        const out_dev = try cublas.GpuBuffer(u8).alloc(N * @sizeOf(f32));
        defer out_dev.free();
        try cudaz.cuMemsetD8(@intFromPtr(out_dev.dev_ptr), 0, N * @sizeOf(f32));
        try engine.gemvSplitK(
            @intFromPtr(self.fp8_act_buf.?.dev_ptr),
            @intFromPtr(self.fp8_act_scale_buf.?.dev_ptr),
            @intFromPtr(w_entry.fp8.dev_ptr),
            @intFromPtr(w_entry.scales.dev_ptr),
            @intFromPtr(out_dev.dev_ptr),
            @intCast(N),
            @intCast(K),
            @intCast(num_splits),
        );
        try cudaz.cuStreamSynchronize(@ptrCast(self.cuda_stream.?.raw));
        try cudaz.cuMemcpyDtoH(@intFromPtr(out.data.ptr), @intFromPtr(out_dev.dev_ptr), N * @sizeOf(f32));
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
                if (self.weight_cache.?.get(key)) |hit| {
                    const d_B = hit.buf;
                    self.touchWeightCache(key);
                    try cublas.gemmCuBlasF32Resident(handle, X, d_B, Y, M, N, K, false, true, 1.0, 0.0);
                    return;
                } else {
                    var d_B = try cublas.GpuBuffer(f32).alloc(W_T.data.len);
                    try d_B.upload(W_T.data);
                    try self.weight_cache.?.put(key, .{ .buf = d_B, .last_used = cacheTick() });
                    try cublas.gemmCuBlasF32Resident(handle, X, d_B, Y, M, N, K, false, true, 1.0, 0.0);
                    return;
                }
            } else if (T == f16) {
                var a_f32 = try Tensor(f32).alloc(self.allocator, &.{ X.shape[0], X.shape[1] });
                defer a_f32.deinit();
                for (X.data, a_f32.data) |s, *d| d.* = @floatCast(s);
                var c_f32 = try Tensor(f32).alloc(self.allocator, Y.shape);
                defer c_f32.deinit();
                if (self.weight_cache.?.get(key)) |hit| {
                    const d_B = hit.buf;
                    self.touchWeightCache(key);
                    try cublas.gemmCuBlasF32Resident(handle, a_f32, d_B, &c_f32, M, N, K, false, true, 1.0, 0.0);
                } else {
                    var b_f32 = try Tensor(f32).alloc(self.allocator, &.{ W_T.shape[0], W_T.shape[1] });
                    defer b_f32.deinit();
                    for (W_T.data, b_f32.data) |s, *d| d.* = @floatCast(s);
                    var d_B = try cublas.GpuBuffer(f32).alloc(b_f32.data.len);
                    try d_B.upload(b_f32.data);
                    try self.weight_cache.?.put(key, .{ .buf = d_B, .last_used = cacheTick() });
                    try cublas.gemmCuBlasF32Resident(handle, a_f32, d_B, &c_f32, M, N, K, false, true, 1.0, 0.0);
                }
                for (Y.data, c_f32.data) |*d, s| d.* = @floatCast(s);
                return;
            }
        }

        // 9.4.1 (lane-b) fix decode híbrido CPU 5×: fast-path GEMV M==1.
        // El fallthrough gemm/parallel hace spawn+join de num_threads y un
        // transpose alloc POR PROYECCIÓN — con M=1 (decode) el cómputo es
        // µs y el overhead domina (medido LFM2-350M: ~2s/capa, 16 capas
        // ⇒ ~32s/token; hallazgo lane-d U1: brazo B unified 0.06 t/s vs
        // legacy 3×+). gemvSimd single-thread: 0 spawn, 0 transpose (W_T
        // ya transposed), SIMD f32. Bit-exactitud: suma por fila f32 SIMD
        // — mismo orden de acumulación que tiled para K divisible (el
        // gate U1 md5 estricto ya es inalcanzable clase §3.1, no aplica).
        // 9.4.1 (lane-b) fix decode híbrido CPU 5×: fast-path GEMV M==1.
        // El fallthrough gemm/parallel hace spawn+join de num_threads y un
        // transpose alloc POR PROYECCIÓN — con M=1 (decode) el cómputo es
        // µs y el overhead domina (medido LFM2-350M: ~2s/capa, 16 capas
        // ⇒ ~32s/token; hallazgo lane-d U1: brazo B unified 0.06 t/s vs
        // legacy). GEMV inline single-thread SIMD: 0 spawn, 0 transpose
        // (W_T ya transposed [N,K], X fila 0 ⇒ y[n] = Σ_k W[n,k]·x[k]).
        // Escritura directa Y.data[n] (rank-agnostic, sin vistas).
        if (X.shape[0] == 1 and T == f32) {
            const N_rows = W_T.shape[0];
            const K_cols = W_T.shape[1];
            const VecLen = @import("types.zig").SimdInfo.vec_len_f32;
            const Vec = @Vector(VecLen, f32);
            const x_row = X.data[0..K_cols];
            const w = W_T.data;
            const k_vec_end = K_cols - (K_cols % VecLen);
            var n: usize = 0;
            while (n < N_rows) : (n += 1) {
                var sum_vec: Vec = @splat(0.0);
                var k: usize = 0;
                const row_off = n * K_cols;
                while (k < k_vec_end) : (k += VecLen) {
                    const a_vec: Vec = w[row_off + k ..][0..VecLen].*;
                    const x_vec: Vec = x_row[k..][0..VecLen].*;
                    sum_vec += a_vec * x_vec;
                }
                var sum: f32 = @reduce(.Add, sum_vec);
                while (k < K_cols) : (k += 1) {
                    sum += w[row_off + k] * x_row[k];
                }
                Y.data[n] = sum;
            }
            return;
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
        // 2.2 (lane-f): routing FP8 end-to-end. Con fp8_route_enabled (CLI
        // --quant fp8) TODAS las proyecciones lineales (attn qkv/out, ssm
        // qkv/z/out, ffn) van por el camino block-scaled E4M3: pesos
        // cuantizados on-demand con cache y activaciones con scratch
        // reutilizado. El backend SIGUE siendo cublas — el resto del engine
        // (KV append, FA, gemm host) funciona normal; sólo las proyecciones
        // device enrutan FP8. (Diseño v2: el v1 con backend .fp8_block
        // global rompía las capas de atención — cada AttentionLayer crea su
        // PROPIO engine y perdía el KV path con KvCacheNotSet.)
        // K%128==0 requerido (grupos E4M3) — fail-fast con breadcrumb.
        if (fp8_route_enabled and self.fp8_engine_opt() != null) {
            if (K % 128 != 0) {
                debugz.dbg.printLevel(.info, "[fp8-route] K={d} no es múltiplo de 128 — geometría sin soporte FP8 block\n", .{K});
                return error.Fp8UnsupportedGeometry;
            }
            if (debugz.dbg.at(.detail))
                debugz.dbg.printLevel(.detail, "[fp8-route] proj M={d} K={d} N={d}\n", .{ M, K, N });
            return self.linearProjectionFp8Device(X, W_T, Y, M, K, N);
        }
        if (self.backend != .cublas or self.weight_cache == null or self.cublas_handle == null) {
            @panic("linearProjectionDevice requiere backend cublas");
        }
        const handle = self.cublas_handle.?;
        const key = @intFromPtr(W_T.data.ptr);
        const d_B = if (self.weight_cache.?.get(key)) |hit| blk: {
            self.touchWeightCache(key);
            break :blk hit.buf;
        } else blk: {
            var buf = try cublas.GpuBuffer(f32).alloc(W_T.data.len);
            try buf.upload(W_T.data);
            try self.weight_cache.?.put(key, .{ .buf = buf, .last_used = cacheTick() });
            break :blk buf;
        };
        try cublas.gemmCuBlasF32Device(handle, d_B, X.buf, Y.buf, N, M, K, false, true, 1.0, 0.0);
    }

    /// Devuelve el puntero device del peso f32 (subido/cacheado como en
    /// linearProjectionDevice) sin hacer el matmul, para kernels custom que
    /// leen el peso directamente.
    pub fn projectionDevicePtr(self: *Self, W_T: Tensor(f32)) !usize {
        if (self.backend != .cublas or self.weight_cache == null) {
            @panic("projectionDevicePtr requiere backend cublas");
        }
        const key = @intFromPtr(W_T.data.ptr);
        const d_B = if (self.weight_cache.?.get(key)) |hit| blk: {
            self.touchWeightCache(key);
            break :blk hit.buf;
        } else blk: {
            var buf = try cublas.GpuBuffer(f32).alloc(W_T.data.len);
            try buf.upload(W_T.data);
            try self.weight_cache.?.put(key, .{ .buf = buf, .last_used = cacheTick() });
            break :blk buf;
        };
        return @intFromPtr(d_B.dev_ptr);
    }

    /// Proyección lineal device→device con peso f16 (p.ej. lm_head): X ya vive en
    /// GPU (f32), el peso se convierte a f32 y se cachea como en linearProjection;
    /// Y (f32) se escribe en GPU sin pasar por host.
    /// 2.2 (lane-f): proyección lineal FP8 block-scaled device-to-device.
    /// X [M,K] f32 ya en GPU → Y [M,N] f32 en GPU. Pesos W_T f32 [N,K] host
    /// cuantizados on-demand con fp8_weight_cache; activaciones con scratch
    /// reutilizado por geometría. M=1 usa gemvSplitK (decode), M>1 el gemm.
    fn linearProjectionFp8Device(
        self: *Self,
        X: cublas.GpuTensor(f32),
        W_T: Tensor(f32),
        Y: *cublas.GpuTensor(f32),
        M: usize,
        K: usize,
        N: usize,
    ) !void {
        const engine = self.fp8_engine_opt() orelse return error.Fp8EngineNotInitialized;
        if (W_T.shape[0] != N or W_T.shape[1] != K) return error.InvalidParameter;

        // ── Pesos: cuantizar UNA vez (cache por host_ptr) ──
        const w_key = @intFromPtr(W_T.data.ptr);
        const w_entry: *Self.Fp8WeightEntry = blk: {
            if (self.fp8_weight_cache.?.getPtr(w_key)) |hit| break :blk hit;
            const w_f32_dev = try cublas.GpuBuffer(f32).alloc(N * K);
            defer w_f32_dev.free();
            try w_f32_dev.upload(W_T.data);
            const fp8_buf = try cublas.GpuBuffer(u8).alloc(N * K);
            const scales_buf = try cublas.GpuBuffer(f32).alloc(N * (K / 128));
            try engine.quantizeActivationsF32( // mismo kernel: fila="token"
                @intFromPtr(w_f32_dev.dev_ptr),
                @intFromPtr(fp8_buf.dev_ptr),
                @intFromPtr(scales_buf.dev_ptr),
                @intCast(N),
                @intCast(K),
            );
            // Sync ANTES de liberar w_f32_dev (el kernel lo lee async).
            try cudaz.cuStreamSynchronize(@ptrCast(self.cuda_stream.?.raw));
            try self.fp8_weight_cache.?.put(w_key, .{
                .fp8 = fp8_buf,
                .scales = scales_buf,
                .n = N,
                .k = K,
            });
            break :blk self.fp8_weight_cache.?.getPtr(w_key).?;
        };

        // ── Activaciones: scratch por K (M-agnóstico) ──
        // 2.2 FIX: dimensionar por K y CAPACIDAD-M (max visto), no por M
        // exacto — con scratch por (M,K) el decode (M=1) re-allocaba tras
        // el prefill (M=5) DENTRO del graph capture ⇒ 901 en cada launch
        // (lección 1.3: nada de alloc en camino capturable). El buffer del
        // M máximo sirve para todos los M menores (colas contiguas).
        if (self.fp8_act_buf == null or self.fp8_act_k != K or self.fp8_act_m < M) {
            if (self.fp8_act_buf) |*b| b.free();
            if (self.fp8_act_scale_buf) |*b| b.free();
            self.fp8_act_buf = try cublas.GpuBuffer(u8).alloc(M * K);
            self.fp8_act_scale_buf = try cublas.GpuBuffer(f32).alloc(M * (K / 128));
            self.fp8_act_m = M;
            self.fp8_act_k = K;
        }
        try engine.quantizeActivationsF32(
            X.ptr(),
            @intFromPtr(self.fp8_act_buf.?.dev_ptr),
            @intFromPtr(self.fp8_act_scale_buf.?.dev_ptr),
            @intCast(M),
            @intCast(K),
        );

        // ── GEMM/GEMV + descarga a Y ──
        // El gemm acumula con atomicAdd ⇒ Y necesita memset PREVIO. La
        // cuMemsetD8 es síncrona vs el stream: encola ANTES del launch,
        // orden correcto.
        // Async (stream-ordered): el memset síncrono es ILEGAL en graph
        // capture (901 en cada launch del replay — lección 1.3).
        try cudaz.cuMemsetD8Async(Y.ptr(), 0, M * N * @sizeOf(f32), @ptrCast(self.cuda_stream.?.raw));
        if (M == 1) {
            const num_splits: i32 = @intCast(@min(K / 128, 8));
            try engine.gemvSplitK(
                @intFromPtr(self.fp8_act_buf.?.dev_ptr),
                @intFromPtr(self.fp8_act_scale_buf.?.dev_ptr),
                @intFromPtr(w_entry.fp8.dev_ptr),
                @intFromPtr(w_entry.scales.dev_ptr),
                Y.ptr(),
                @intCast(N),
                @intCast(K),
                num_splits,
            );
        } else {
            try engine.gemm(
                @intFromPtr(self.fp8_act_buf.?.dev_ptr),
                @intFromPtr(self.fp8_act_scale_buf.?.dev_ptr),
                @intFromPtr(w_entry.fp8.dev_ptr),
                @intFromPtr(w_entry.scales.dev_ptr),
                Y.ptr(),
                @intCast(M),
                @intCast(N),
                @intCast(K),
            );
        }
    }

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
        const d_B = if (self.weight_cache.?.get(key)) |hit| blk: {
            self.touchWeightCache(key);
            break :blk hit.buf;
        } else blk: {
            var b_f32 = try Tensor(f32).alloc(self.allocator, &.{ W_T16.shape[0], W_T16.shape[1] });
            defer b_f32.deinit();
            for (W_T16.data, b_f32.data) |s, *d| d.* = @floatCast(s);
            var buf = try cublas.GpuBuffer(f32).alloc(b_f32.data.len);
            try buf.upload(b_f32.data);
            try self.weight_cache.?.put(key, .{ .buf = buf, .last_used = cacheTick() });
            break :blk buf;
        };
        try cublas.gemmCuBlasF32Device(handle, d_B, X32.buf, Y32.buf, N, M, K, false, true, 1.0, 0.0);
    }

    // ─── FFN SwiGLU ───

    pub fn ffnProjections(self: *Self, comptime T: type, X: Tensor(T), W_gate_T: Tensor(T), W_up_T: Tensor(T), gate_out: *Tensor(T), up_out: *Tensor(T)) !void {
        try self.linearProjection(T, X, W_gate_T, gate_out);
        try self.linearProjection(T, X, W_up_T, up_out);
    }

    // ─── GEMM cuantizado (INT8) ───

    pub fn gemmQuantized(self: *Self, A: Tensor(f32), B_q: QuantizedTensor, C: *const Tensor(f32), M: usize, N: usize, K: usize) !void {
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
            .naive => "naive",
            .simd => "simd",
            .tiled => "tiled",
            .parallel => "parallel",
            .openblas => "openblas",
            .cublas => "cublas",
            .fp8_block => "fp8_block",
            .auto => "auto",
        };
    }

    pub const PoolStats = cublas.GpuMemoryPool.PoolStats;

    pub fn gpuPoolStats(self: Self) ?PoolStats {
        if (self.gpu_pool) |pool| return pool.stats();
        return null;
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// GraphCapture — CUDA Graph capture refactor (Phase 3, FreeToken Technique 9)
// ═══════════════════════════════════════════════════════════════════════════════
// La implementación vive en src/cuda/graph_capture.zig (módulo mínimo: solo
// cudaz + debug) para que decode_graph pueda componerla sin depender de
// matmul. Aquí se re-exporta para mantener matmul.GraphCapture estable.
pub const GraphCapture = graph_capture.GraphCapture;

fn detectBestBackend() Backend {
    // Presupuesto central (2026-09-13): si tras load/knob quedan ≥2 cores
    // computantes, paralelo; si no, SIMD/naive. Antes: getCpuCount().
    var fba_buf: [256]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&fba_buf);
    if (resources.computeThreadBudget(fba.allocator(), "detect_backend") > 1) return .parallel;
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
// lane-f Phase 3: encode GPU (MXFP4/Q8_0) — espejo bit-exact de kv_quant.
pub const quant_encode = @import("quant/encode.zig");

// ── Tests GraphCapture (máquina de estados pura, sin GPU) ────────────────────

const testing = std.testing;

test "GraphCapture: estado inicial no replayable, exec nulo tras deinit" {
    // Sin GPU: solo verificamos el estado puro del struct.
    var gc = GraphCapture.init(@as(cudaz.CUstream, @ptrFromInt(0xdead)));
    try testing.expectEqual(false, gc.isReplayable());
    try testing.expectEqual(@as(usize, 0), gc.last_node_count);
    try testing.expectEqual(@as(u64, 0), gc.generation);
    gc.deinit(); // exec == null: no-op, sin crash
    try testing.expectEqual(false, gc.isReplayable());
}
