//! Motor GPU de PagedAttention (decode / prefill / reshape / copy).
//! Carga el cubin `paged_attention_sm86.cubin` compilado por el build y lanza
//! los kernels equivalentes a la referencia CPU (`attention.zig`).
//! Layout de bloques: memory-pool de `BlockAllocator` (K region + V region por
//! bloque físico), el mismo que lee `PagedAttention.decode`.
const std = @import("std");
const cudaz = @import("cudaz");
const build_options = @import("build_options");
const BlockTable = @import("block_table.zig").BlockTable;
const BlockAllocator = @import("allocator.zig").BlockAllocator;
const PagedGpuBlockPool = @import("paged_gpu_pool.zig").PagedGpuBlockPool;
const PagedConfig = @import("root.zig").PagedConfig;

pub const PagedAttentionGpuError = error{
    CudaUnavailable,
    KernelNotFound,
};

/// Pool de bloques persistente en el dispositivo. Aloca el buffer `d_cache`
/// una sola vez (`num_blocks * block_bytes`) y mantiene un bitmap de qué
/// bloques físicos están residentes en GPU. `stageBlock` sube bloques
/// individuales H2D; `evictBlock` los baja D2H de vuelta al host. Sustituye
/// las copias completas del memory-pool en cada llamada a `decode`.
pub const GpuBlockPool = struct {
    allocator: std.mem.Allocator,
    num_blocks: usize,
    block_bytes: usize,
    d_cache: cudaz.CUdeviceptr,
    resident: []bool,
    dirty: []bool,

    const Self = @This();

    pub fn init(gpa: std.mem.Allocator, num_blocks: usize, block_bytes: usize) !Self {
        const resident = try gpa.alloc(bool, num_blocks);
        errdefer gpa.free(resident);
        @memset(resident, false);
        const dirty = try gpa.alloc(bool, num_blocks);
        errdefer gpa.free(dirty);
        @memset(dirty, false);
        const d_cache = try cudaz.cuMemAlloc(num_blocks * block_bytes);
        return .{
            .allocator = gpa,
            .num_blocks = num_blocks,
            .block_bytes = block_bytes,
            .d_cache = d_cache,
            .resident = resident,
            .dirty = dirty,
        };
    }

    pub fn deinit(self: *Self) void {
        cudaz.cuMemFree(self.d_cache);
        self.allocator.free(self.resident);
        self.allocator.free(self.dirty);
    }

    pub fn markDirty(self: *Self, phys_id: usize) void {
        if (phys_id >= self.num_blocks) return;
        self.dirty[phys_id] = true;
    }

    /// Sube el bloque a GPU solo si no está residente o fue modificado en host
    /// (dirty). La copia del KV del token actual la marca dirty el forward de
    /// atención; los bloques pasados ya residentes se saltan (evita el H2D
    /// creciente con el contexto).
    pub fn stageBlock(self: *Self, block_alloc: *BlockAllocator, phys_id: usize) !void {
        if (phys_id >= self.num_blocks) return;
        if (self.resident[phys_id] and !self.dirty[phys_id]) return;
        const src = @intFromPtr(block_alloc.memory_pool.ptr) + phys_id * self.block_bytes;
        try cudaz.cuMemcpyHtoD(self.d_cache + phys_id * self.block_bytes, src, self.block_bytes);
        self.resident[phys_id] = true;
        self.dirty[phys_id] = false;
    }

    pub fn evictBlock(self: *Self, block_alloc: *BlockAllocator, phys_id: usize) !void {
        if (phys_id >= self.num_blocks or !self.resident[phys_id]) return;
        const dst = @intFromPtr(block_alloc.memory_pool.ptr) + phys_id * self.block_bytes;
        try cudaz.cuMemcpyDtoH(dst, self.d_cache + phys_id * self.block_bytes, self.block_bytes);
        self.resident[phys_id] = false;
    }

    /// Marca el bloque residente en device sin copiar H2D: el KV se escribe por
    /// GPU (kvAppendF16) directamente sobre `d_cache`. Para el pool contiguo el
    /// espacio ya está alocado, solo se actualiza el estado.
    pub fn ensureCommitted(self: *Self, phys_id: usize) void {
        if (phys_id >= self.num_blocks) return;
        self.resident[phys_id] = true;
        self.dirty[phys_id] = false;
    }

    /// Baja (async, stream-ordered) el bloque escrito por GPU al host pool.
    pub fn syncBlockToHost(self: *Self, block_alloc: *BlockAllocator, phys_id: usize, stream: cudaz.CUstream) !void {
        if (phys_id >= self.num_blocks) return;
        const dst = @intFromPtr(block_alloc.memory_pool.ptr) + phys_id * self.block_bytes;
        try cudaz.cuMemcpyDtoHAsync(dst, self.d_cache + phys_id * self.block_bytes, self.block_bytes, stream);
    }

    pub fn stageTable(self: *Self, block_alloc: *BlockAllocator, block_table: *const BlockTable) !void {
        for (0..block_table.numBlocks()) |i| {
            if (block_table.getPhysical(i)) |phys| try self.stageBlock(block_alloc, phys);
        }
    }

    pub fn evictAll(self: *Self, block_alloc: *BlockAllocator) !void {
        for (0..self.num_blocks) |phys| {
            try self.evictBlock(block_alloc, phys);
        }
    }

    /// Baja del dispositivo una lista de bloques fríos (D2H) y los marca no
    /// residentes. Se usa cuando la prefix cache desaloja entradas frías.
    pub fn evictBlocks(self: *Self, block_alloc: *BlockAllocator, phys_ids: []const usize) !void {
        for (phys_ids) |phys| {
            try self.evictBlock(block_alloc, phys);
        }
    }

    pub fn numResident(self: *const Self) usize {
        var n: usize = 0;
        for (self.resident) |r| {
            if (r) n += 1;
        }
        return n;
    }
};

pub const PagedAttentionGpu = struct {
    allocator: std.mem.Allocator,
    config: PagedConfig,
    module: cudaz.CUmodule,
    // Stream CUDA compartido con el resto de la capa híbrida (un solo stream,
    // una sola sincronización por token). NO se destruye en deinit.
    stream: cudaz.CUstream,
    pool: ?GpuBlockPool = null,
    paged_pool: ?PagedGpuBlockPool = null,
    // Buffers de decode persistentes (evitan cuMemAlloc/free por token).
    d_q16: cudaz.CUdeviceptr = 0,
    d_out16: cudaz.CUdeviceptr = 0,
    d_bt: cudaz.CUdeviceptr = 0,
    d_seq_lens: cudaz.CUdeviceptr = 0,
    q_buf_cap: usize = 0,
    bt_cap: usize = 0,

    const Self = @This();

    pub fn init(gpa: std.mem.Allocator, config: PagedConfig, stream: cudaz.CUstream) !Self {
        const cubin_path = build_options.paged_cubin;
        if (cubin_path.len == 0) return error.CudaUnavailable;
        try cudaz.ensureContext();
        const module = try cudaz.cuModuleLoad(cubin_path);
        errdefer cudaz.cuModuleUnload(module);
        return .{ .allocator = gpa, .config = config, .module = module, .stream = stream };
    }

    pub fn deinit(self: *Self) void {
        if (self.pool) |*p| p.deinit();
        if (self.paged_pool) |*p| p.deinit();
        if (self.d_q16 != 0) cudaz.cuMemFree(self.d_q16);
        if (self.d_out16 != 0) cudaz.cuMemFree(self.d_out16);
        if (self.d_bt != 0) cudaz.cuMemFree(self.d_bt);
        if (self.d_seq_lens != 0) cudaz.cuMemFree(self.d_seq_lens);
        cudaz.cuModuleUnload(self.module);
    }

    /// Aloca (o reutiliza) el pool de bloques del dispositivo.
    /// Prefiere el pool VMM paginado (commit físico por bloque); si el driver
    /// no soporta VMM o `block_bytes` no está alineado a granularidad, cae al
    /// pool contiguo.
    pub fn ensurePool(self: *Self, block_alloc: *BlockAllocator) !void {
        if (self.pool != null or self.paged_pool != null) return;
        if (PagedGpuBlockPool.init(
            self.allocator,
            block_alloc.numTotal(),
            block_alloc.block_bytes,
        )) |pp| {
            self.paged_pool = pp;
        } else |_| {
            self.pool = try GpuBlockPool.init(
                self.allocator,
                block_alloc.numTotal(),
                block_alloc.block_bytes,
            );
        }
    }

    pub fn cacheBase(self: *Self, block_alloc: *BlockAllocator) !cudaz.CUdeviceptr {
        try self.ensurePool(block_alloc);
        if (self.paged_pool) |*pp| return pp.vaddr;
        return self.pool.?.d_cache;
    }

    /// Marca un bloque físico como modificado en host (nuevo KV escrito). El
    /// próximo stage lo re-subirá aunque esté residente.
    pub fn markDirty(self: *Self, phys_id: usize) void {
        if (self.paged_pool) |*pp| {
            pp.markDirty(phys_id);
        } else if (self.pool) |*p| {
            p.markDirty(phys_id);
        }
    }

    fn ensureDecodeBuffers(self: *Self, q_stride: usize, max_num_blocks: usize) !void {
        try cudaz.ensureCurrent();
        if (self.q_buf_cap < q_stride) {
            if (self.d_q16 != 0) cudaz.cuMemFree(self.d_q16);
            if (self.d_out16 != 0) cudaz.cuMemFree(self.d_out16);
            self.d_q16 = try cudaz.cuMemAlloc(q_stride * @sizeOf(f16));
            self.d_out16 = try cudaz.cuMemAlloc(q_stride * @sizeOf(f16));
            self.q_buf_cap = q_stride;
        }
        if (self.bt_cap < max_num_blocks) {
            if (self.d_bt != 0) cudaz.cuMemFree(self.d_bt);
            self.d_bt = try cudaz.cuMemAlloc(max_num_blocks * @sizeOf(c_int));
            self.bt_cap = max_num_blocks;
        }
        if (self.d_seq_lens == 0) self.d_seq_lens = try cudaz.cuMemAlloc(@sizeOf(c_int));
    }

    fn stageBlocks(self: *Self, block_alloc: *BlockAllocator, block_table: *const BlockTable) !void {
        try self.ensurePool(block_alloc);
        if (self.paged_pool) |*pp| {
            try pp.stageTable(block_alloc, block_table);
        } else {
            try self.pool.?.stageTable(block_alloc, block_table);
        }
    }

    /// Sube todos los bloques de una block table al device (uso puntual tras el
    /// prefill CPU, antes del decode GPU-residente).
    pub fn stageTableAll(self: *Self, block_alloc: *BlockAllocator, block_table: *const BlockTable) !void {
        try self.stageBlocks(block_alloc, block_table);
    }

    fn stageBlock(self: *Self, block_alloc: *BlockAllocator, phys_id: usize) !void {
        try self.ensurePool(block_alloc);
        if (self.paged_pool) |*pp| {
            try pp.stageBlock(block_alloc, phys_id);
        } else {
            try self.pool.?.stageBlock(block_alloc, phys_id);
        }
    }

    fn evictBlocks(self: *Self, block_alloc: *BlockAllocator, phys_ids: []const usize) !void {
        if (self.paged_pool) |*pp| {
            try pp.evictBlocks(block_alloc, phys_ids);
        } else if (self.pool) |*p| {
            try p.evictBlocks(block_alloc, phys_ids);
        }
    }

    fn evictBlock(self: *Self, block_alloc: *BlockAllocator, phys_id: usize) !void {
        if (self.paged_pool) |*pp| {
            try pp.evictBlock(block_alloc, phys_id);
        } else if (self.pool) |*p| {
            try p.evictBlock(block_alloc, phys_id);
        }
    }

    /// Decode de un token (1 secuencia). `query`/`out` en f32; K/V se leen del
    /// memory-pool de `block_alloc` como f16. Equivale a `PagedAttention.decode`.
    pub fn decode(
        self: *Self,
        query: []const f32,
        out: []f32,
        block_table: *const BlockTable,
        block_alloc: *BlockAllocator,
    ) !void {
        const config = self.config;
        const num_q_heads = config.num_q_heads;
        const num_kv_heads = config.num_kv_heads;
        const head_dim = config.head_dim;
        const block_size = config.block_size;
        const q_stride = num_q_heads * head_dim;
        const seq_len = block_table.num_tokens;
        const max_num_blocks = block_table.numBlocks();

        std.debug.assert(query.len == q_stride);
        std.debug.assert(out.len == q_stride);

        try cudaz.ensureCurrent();

        const q_f16 = try self.allocator.alloc(f16, q_stride);
        defer self.allocator.free(q_f16);
        for (query, 0..) |v, i| q_f16[i] = @floatCast(v);

        const bt_host = try self.allocator.alloc(c_int, max_num_blocks);
        defer self.allocator.free(bt_host);
        for (0..max_num_blocks) |i| {
            bt_host[i] = if (block_table.getPhysical(i)) |phys| @intCast(phys) else -1;
        }

        // Subir solo los bloques referenciados por la block table.
        try self.stageBlocks(block_alloc, block_table);

        try self.ensureDecodeBuffers(q_stride, max_num_blocks);

        var seq_len_c: c_int = @intCast(seq_len);

        try cudaz.cuMemcpyHtoD(self.d_q16, @intFromPtr(q_f16.ptr), q_stride * @sizeOf(f16));
        try cudaz.cuMemcpyHtoD(self.d_bt, @intFromPtr(bt_host.ptr), max_num_blocks * @sizeOf(c_int));
        try cudaz.cuMemcpyHtoD(self.d_seq_lens, @intFromPtr(&seq_len_c), @sizeOf(c_int));

        const func = cudaz.cuModuleGetFunction(self.module, "paged_attention_decode_f16_kernel") catch return error.KernelNotFound;

        // Scalar params: kernel recibe int por valor; el driver los lee del host.
        var num_seqs_c: c_int = 1;
        var max_blocks_c: c_int = @intCast(max_num_blocks);
        var num_q_c: c_int = @intCast(num_q_heads);
        var num_kv_c: c_int = @intCast(num_kv_heads);
        var head_dim_c: c_int = @intCast(head_dim);
        var block_size_c: c_int = @intCast(block_size);

        var d_cache_v = try self.cacheBase(block_alloc);

        var kp = [_]?*anyopaque{
            &self.d_out16, &self.d_q16,   &d_cache_v, &self.d_bt,
            &self.d_seq_lens, &num_seqs_c, &max_blocks_c, &num_q_c,
            &num_kv_c,   &head_dim_c, &block_size_c,
        };
        const shared_bytes: c_uint = @intCast(2 * head_dim * @sizeOf(f32));
        try cudaz.cuLaunchKernel(func, 1, @intCast(num_q_heads), 1, 32, 1, 1, shared_bytes, self.stream, @ptrCast(&kp), null);
        try cudaz.cuStreamSynchronize(self.stream);

        const out_f16 = try self.allocator.alloc(f16, q_stride);
        defer self.allocator.free(out_f16);
        try cudaz.cuMemcpyDtoH(@intFromPtr(out_f16.ptr), self.d_out16, q_stride * @sizeOf(f16));

        for (out_f16, 0..) |v, i| out[i] = @floatCast(v);
    }

    // ─── Ruta decode 100% device (Phase 1b) ─────────────────────────────────
    // q16/out16 ya viven en GPU (AttentionGpu); el KV se escribe con
    // kvAppendF16 y el decode lee el pool sin staging ni sync. El llamador
    // sincroniza el stream una vez por token.
    pub fn uploadBlockTable(self: *Self, bt_host: []const c_int) !void {
        try cudaz.ensureCurrent();
        try self.ensureDecodeBuffers(self.config.num_q_heads * self.config.head_dim, bt_host.len);
        try cudaz.cuMemcpyHtoDAsync(self.d_bt, @intFromPtr(bt_host.ptr), bt_host.len * @sizeOf(c_int), self.stream);
    }

    pub fn getDbt(self: *Self) cudaz.CUdeviceptr {
        return self.d_bt;
    }

    /// El bloque será escrito por GPU (kvAppendF16) en lugar de H2D: lo marca
    /// residente/committed sin copiar para que el decode device pueda leerlo.
    pub fn ensureBlockCommitted(self: *Self, block_alloc: *BlockAllocator, phys_id: usize) !void {
        try self.ensurePool(block_alloc);
        if (self.paged_pool) |*pp| {
            try pp.ensureCommitted(phys_id);
        } else if (self.pool) |*p| {
            p.ensureCommitted(phys_id);
        }
    }

    /// D2H async (stream-ordered) del bloque escrito por GPU → host pool
    /// (mantiene el pool host autoritativo para COW/scheduler).
    pub fn syncBlockToHost(self: *Self, block_alloc: *BlockAllocator, phys_id: usize) !void {
        if (self.paged_pool) |*pp| {
            try pp.syncBlockToHost(block_alloc, phys_id, self.stream);
        } else if (self.pool) |*p| {
            try p.syncBlockToHost(block_alloc, phys_id, self.stream);
        }
    }

    /// Decode device→device: `q16`/`out16` son buffers f16 de GPU y `bt_host`
    /// el block table host (se sube a d_bt). Sin staging de bloques y sin sync.
    pub fn decodeDevice(
        self: *Self,
        q16: cudaz.CUdeviceptr,
        out16: cudaz.CUdeviceptr,
        block_alloc: *BlockAllocator,
        bt_host: []const c_int,
        seq_len: usize,
    ) !void {
        const config = self.config;
        const num_q_heads = config.num_q_heads;
        const num_kv_heads = config.num_kv_heads;
        const head_dim = config.head_dim;
        const block_size = config.block_size;
        const q_stride = num_q_heads * head_dim;
        const max_num_blocks = bt_host.len;

        try cudaz.ensureCurrent();
        try self.ensureDecodeBuffers(q_stride, max_num_blocks);
        try cudaz.cuMemcpyHtoDAsync(self.d_bt, @intFromPtr(bt_host.ptr), max_num_blocks * @sizeOf(c_int), self.stream);

        var seq_len_c: c_int = @intCast(seq_len);
        try cudaz.cuMemcpyHtoDAsync(self.d_seq_lens, @intFromPtr(&seq_len_c), @sizeOf(c_int), self.stream);

        const func = cudaz.cuModuleGetFunction(self.module, "paged_attention_decode_f16_kernel") catch return error.KernelNotFound;

        var num_seqs_c: c_int = 1;
        var max_blocks_c: c_int = @intCast(max_num_blocks);
        var num_q_c: c_int = @intCast(num_q_heads);
        var num_kv_c: c_int = @intCast(num_kv_heads);
        var head_dim_c: c_int = @intCast(head_dim);
        var block_size_c: c_int = @intCast(block_size);

        var d_cache_v = try self.cacheBase(block_alloc);
        var q16v = q16;
        var out16v = out16;

        var kp = [_]?*anyopaque{
            &out16v, &q16v, &d_cache_v, &self.d_bt,
            &self.d_seq_lens, &num_seqs_c, &max_blocks_c, &num_q_c,
            &num_kv_c, &head_dim_c, &block_size_c,
        };
const shared_bytes: c_uint = @intCast(2 * head_dim * @sizeOf(f32));
        try cudaz.cuLaunchKernel(func, 1, @intCast(num_q_heads), 1, 32, 1, 1, shared_bytes, self.stream, @ptrCast(&kp), null);
        try cudaz.cuStreamSynchronize(self.stream);
    }

    /// Prefill device→device causal (chunks): `q16`/`out16` son buffers f16 de
    /// GPU con `n_queries` tokens; atienden causalmente sobre los tokens ya
    /// escritos en el pool (posiciones [0..start_pos+n_queries)). `start_pos`
    /// es la posición absoluta del primer query. Sin staging de bloques ni sync.
    pub fn prefillDevice(
        self: *Self,
        q16: cudaz.CUdeviceptr,
        out16: cudaz.CUdeviceptr,
        block_alloc: *BlockAllocator,
        bt_host: []const c_int,
        n_queries: usize,
        start_pos: usize,
    ) !void {
        const config = self.config;
        const num_q_heads = config.num_q_heads;
        const num_kv_heads = config.num_kv_heads;
        const head_dim = config.head_dim;
        const block_size = config.block_size;
        const q_stride = num_q_heads * head_dim;
        const max_num_blocks = bt_host.len;

        try cudaz.ensureCurrent();
        try self.ensureDecodeBuffers(q_stride, max_num_blocks);
        try cudaz.cuMemcpyHtoDAsync(self.d_bt, @intFromPtr(bt_host.ptr), max_num_blocks * @sizeOf(c_int), self.stream);

        const func = cudaz.cuModuleGetFunction(self.module, "paged_attention_prefill_f16_kernel") catch return error.KernelNotFound;

        var n_queries_c: c_int = @intCast(n_queries);
        var start_pos_c: c_int = @intCast(start_pos);
        var num_q_c: c_int = @intCast(num_q_heads);
        var num_kv_c: c_int = @intCast(num_kv_heads);
        var head_dim_c: c_int = @intCast(head_dim);
        var block_size_c: c_int = @intCast(block_size);

        var d_cache_v = try self.cacheBase(block_alloc);
        var q16v = q16;
        var out16v = out16;

        var kp = [_]?*anyopaque{
            &out16v,    &q16v, &d_cache_v, &self.d_bt,
            &n_queries_c, &start_pos_c, &num_q_c, &num_kv_c, &head_dim_c,
            &block_size_c,
        };
        const shared_bytes: c_uint = @intCast(2 * head_dim * @sizeOf(f32));
        try cudaz.cuLaunchKernel(func, @intCast(n_queries), @intCast(num_q_heads), 1, 32, 1, 1, shared_bytes, self.stream, @ptrCast(&kp), null);
    }

    /// Prefill batch: un solo lanzamiento del kernel `paged_attention_prefill_f16_kernel`
    /// (bloque por (token, q_head)) con máscara causal, en vez de iterar decode
    /// por posición. Equivale a `PagedAttention.prefill`.
    pub fn prefill(
        self: *Self,
        queries: []const f32,
        outs: []f32,
        block_table: *const BlockTable,
        block_alloc: *BlockAllocator,
        seq_len: usize,
    ) !void {
        const config = self.config;
        const num_q_heads = config.num_q_heads;
        const num_kv_heads = config.num_kv_heads;
        const head_dim = config.head_dim;
        const block_size = config.block_size;
        const q_stride = num_q_heads * head_dim;

        std.debug.assert(queries.len == seq_len * q_stride);
        std.debug.assert(outs.len == seq_len * q_stride);
        if (seq_len == 0) return;

        try cudaz.ensureCurrent();

        const total = seq_len * q_stride;
        const q_f16 = try self.allocator.alloc(f16, total);
        defer self.allocator.free(q_f16);
        for (queries, 0..) |v, i| q_f16[i] = @floatCast(v);

        const max_num_blocks = block_table.numBlocks();
        const bt_host = try self.allocator.alloc(c_int, max_num_blocks);
        defer self.allocator.free(bt_host);
        for (0..max_num_blocks) |i| {
            bt_host[i] = if (block_table.getPhysical(i)) |phys| @intCast(phys) else -1;
        }

        try self.stageBlocks(block_alloc, block_table);

        var d_outs = try cudaz.cuMemAlloc(total * @sizeOf(f16));
        defer cudaz.cuMemFree(d_outs);
        var d_queries = try cudaz.cuMemAlloc(total * @sizeOf(f16));
        defer cudaz.cuMemFree(d_queries);
        var d_bt = try cudaz.cuMemAlloc(max_num_blocks * @sizeOf(c_int));
        defer cudaz.cuMemFree(d_bt);

        try cudaz.cuMemcpyHtoD(d_queries, @intFromPtr(q_f16.ptr), total * @sizeOf(f16));
        try cudaz.cuMemcpyHtoD(d_bt, @intFromPtr(bt_host.ptr), max_num_blocks * @sizeOf(c_int));

        const func = cudaz.cuModuleGetFunction(self.module, "paged_attention_prefill_f16_kernel") catch return error.KernelNotFound;

        var n_queries_c: c_int = @intCast(seq_len);
        var start_pos_c: c_int = 0;
        var num_q_c: c_int = @intCast(num_q_heads);
        var num_kv_c: c_int = @intCast(num_kv_heads);
        var head_dim_c: c_int = @intCast(head_dim);
        var block_size_c: c_int = @intCast(block_size);

        var d_cache_v = try self.cacheBase(block_alloc);

        var kp = [_]?*anyopaque{
            &d_outs,    &d_queries, &d_cache_v, &d_bt,
            &n_queries_c, &start_pos_c, &num_q_c, &num_kv_c, &head_dim_c,
            &block_size_c,
        };
        const shared_bytes: c_uint = @intCast(2 * head_dim * @sizeOf(f32));
        try cudaz.cuLaunchKernel(func, @intCast(seq_len), @intCast(num_q_heads), 1, 32, 1, 1, shared_bytes, self.stream, @ptrCast(&kp), null);
        try cudaz.cuStreamSynchronize(self.stream);

        const out_f16 = try self.allocator.alloc(f16, total);
        defer self.allocator.free(out_f16);
        try cudaz.cuMemcpyDtoH(@intFromPtr(out_f16.ptr), d_outs, total * @sizeOf(f16));

        for (out_f16, 0..) |v, i| outs[i] = @floatCast(v);
    }

    /// Copia bloques físicos (COW / fork) vía `block_copy_f16_kernel`.
    pub fn blockCopy(
        self: *Self,
        block_alloc: *BlockAllocator,
        copy_map: []const [2]c_int,
    ) !void {
        try cudaz.ensureCurrent();
        if (copy_map.len == 0) return;

        // Subir bloques fuente que no estén residentes.
        for (copy_map) |m| {
            const src: usize = @intCast(m[1]);
            try self.stageBlock(block_alloc, src);
        }

        const map_host = try self.allocator.alloc([2]c_int, copy_map.len);
        defer self.allocator.free(map_host);
        @memcpy(map_host, copy_map);
        var d_map = try cudaz.cuMemAlloc(map_host.len * @sizeOf([2]c_int));
        defer cudaz.cuMemFree(d_map);
        try cudaz.cuMemcpyHtoD(d_map, @intFromPtr(map_host.ptr), map_host.len * @sizeOf([2]c_int));

        var num_copies_c: c_int = @intCast(copy_map.len);
        var block_bytes_c: c_int = @intCast(block_alloc.block_bytes);
        const d_cache = try self.cacheBase(block_alloc);

        const func = cudaz.cuModuleGetFunction(self.module, "block_copy_f16_kernel") catch return error.KernelNotFound;
        var kp = [_]?*anyopaque{ &d_cache, &d_cache, &d_map, &num_copies_c, &block_bytes_c };
        const blocks: c_uint = @intCast((copy_map.len + 127) / 128);
        try cudaz.cuLaunchKernel(func, blocks, 1, 1, 128, 1, 1, 0, self.stream, @ptrCast(&kp), null);
        try cudaz.cuStreamSynchronize(self.stream);

        // Bajar solo los bloques destino modificados.
        for (copy_map) |m| {
            const dst: usize = @intCast(m[0]);
            try self.evictBlock(block_alloc, dst);
        }
        try cudaz.cuCtxSynchronize();
    }

    /// Baja del dispositivo los bloques fríos de la prefix cache (según su
    /// hit rate) usando `PrefixCache.evictGpuCold`, liberando memoria GPU para
    /// bloques calientes. Devuelve cuántos bloques se bajaron.
    pub fn evictColdBlocksFromCache(
        self: *Self,
        block_alloc: *BlockAllocator,
        prefix_cache: *@import("prefix_cache.zig").PrefixCache,
        max_age: u64,
        min_hit_rate: f64,
    ) !usize {
        try cudaz.ensureCurrent();
        const cold = prefix_cache.evictGpuCold(max_age, min_hit_rate);
        defer self.allocator.free(cold);
        try self.evictBlocks(block_alloc, cold);
        return cold.len;
    }
};

pub const CpuReference = struct {
    pub const decode = @import("attention.zig").PagedAttention.decode;
    pub const prefill = @import("attention.zig").PagedAttention.prefill;
};