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
const PagedConfig = @import("root.zig").PagedConfig;

pub const PagedAttentionGpuError = error{
    CudaUnavailable,
    KernelNotFound,
};

pub const PagedAttentionGpu = struct {
    allocator: std.mem.Allocator,
    config: PagedConfig,
    module: cudaz.CUmodule,
    stream: cudaz.CUstream,

    const Self = @This();

    pub fn init(gpa: std.mem.Allocator, config: PagedConfig) !Self {
        const cubin_path = build_options.paged_cubin;
        if (cubin_path.len == 0) return error.CudaUnavailable;
        try cudaz.ensureContext();
        const module = try cudaz.cuModuleLoad(cubin_path);
        errdefer cudaz.cuModuleUnload(module);
        const stream = try cudaz.cuStreamCreate(0);
        errdefer cudaz.cuStreamDestroy(stream);
        return .{ .allocator = gpa, .config = config, .module = module, .stream = stream };
    }

    pub fn deinit(self: *Self) void {
        cudaz.cuStreamDestroy(self.stream);
        cudaz.cuModuleUnload(self.module);
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

        var d_out = try cudaz.cuMemAlloc(q_stride * @sizeOf(f16));
        defer cudaz.cuMemFree(d_out);
        var d_query = try cudaz.cuMemAlloc(q_stride * @sizeOf(f16));
        defer cudaz.cuMemFree(d_query);
        var d_cache = try cudaz.cuMemAlloc(block_alloc.memory_pool.len);
        defer cudaz.cuMemFree(d_cache);
        var d_bt = try cudaz.cuMemAlloc(max_num_blocks * @sizeOf(c_int));
        defer cudaz.cuMemFree(d_bt);
        var seq_len_c: c_int = @intCast(seq_len);
        var d_seq_lens = try cudaz.cuMemAlloc(@sizeOf(c_int));
        defer cudaz.cuMemFree(d_seq_lens);

        try cudaz.cuMemcpyHtoD(d_query, @intFromPtr(q_f16.ptr), q_stride * @sizeOf(f16));
        try cudaz.cuMemcpyHtoD(d_cache, @intFromPtr(block_alloc.memory_pool.ptr), block_alloc.memory_pool.len);
        try cudaz.cuMemcpyHtoD(d_bt, @intFromPtr(bt_host.ptr), max_num_blocks * @sizeOf(c_int));
        try cudaz.cuMemcpyHtoD(d_seq_lens, @intFromPtr(&seq_len_c), @sizeOf(c_int));

        const func = cudaz.cuModuleGetFunction(self.module, "paged_attention_decode_f16_kernel") catch return error.KernelNotFound;

        // Scalar params: kernel recibe int por valor; el driver los lee del host.
        var num_seqs_c: c_int = 1;
        var max_blocks_c: c_int = @intCast(max_num_blocks);
        var num_q_c: c_int = @intCast(num_q_heads);
        var num_kv_c: c_int = @intCast(num_kv_heads);
        var head_dim_c: c_int = @intCast(head_dim);
        var block_size_c: c_int = @intCast(block_size);

        var kp = [_]?*anyopaque{
            &d_out,      &d_query,   &d_cache,     &d_bt,
            &d_seq_lens, &num_seqs_c, &max_blocks_c, &num_q_c,
            &num_kv_c,   &head_dim_c, &block_size_c,
        };
        const shared_bytes: c_uint = @intCast(2 * head_dim * @sizeOf(f32));
        try cudaz.cuLaunchKernel(func, 1, @intCast(num_q_heads), 1, 32, 1, 1, shared_bytes, self.stream, @ptrCast(&kp), null);
        try cudaz.cuStreamSynchronize(self.stream);

        const out_f16 = try self.allocator.alloc(f16, q_stride);
        defer self.allocator.free(out_f16);
        try cudaz.cuMemcpyDtoH(@intFromPtr(out_f16.ptr), d_out, q_stride * @sizeOf(f16));
        try cudaz.cuCtxSynchronize();

        for (out_f16, 0..) |v, i| out[i] = @floatCast(v);
    }

    /// Prefill: decodifica cada posición de la secuencia (igual que la
    /// referencia CPU `PagedAttention.prefill`, que itera decode por posición).
    pub fn prefill(
        self: *Self,
        queries: []const f32,
        outs: []f32,
        block_table: *const BlockTable,
        block_alloc: *BlockAllocator,
        seq_len: usize,
    ) !void {
        const q_stride = self.config.num_q_heads * self.config.head_dim;
        for (0..seq_len) |i| {
            try self.decode(
                queries[i * q_stride ..][0..q_stride],
                outs[i * q_stride ..][0..q_stride],
                block_table,
                block_alloc,
            );
        }
    }

    /// Copia bloques físicos (COW / fork) vía `block_copy_f16_kernel`.
    pub fn blockCopy(
        self: *Self,
        block_alloc: *BlockAllocator,
        copy_map: []const [2]c_int,
    ) !void {
        try cudaz.ensureCurrent();
        if (copy_map.len == 0) return;

        var d_cache = try cudaz.cuMemAlloc(block_alloc.memory_pool.len);
        defer cudaz.cuMemFree(d_cache);
        try cudaz.cuMemcpyHtoD(d_cache, @intFromPtr(block_alloc.memory_pool.ptr), block_alloc.memory_pool.len);

        const map_host = try self.allocator.alloc([2]c_int, copy_map.len);
        defer self.allocator.free(map_host);
        @memcpy(map_host, copy_map);
        var d_map = try cudaz.cuMemAlloc(map_host.len * @sizeOf([2]c_int));
        defer cudaz.cuMemFree(d_map);
        try cudaz.cuMemcpyHtoD(d_map, @intFromPtr(map_host.ptr), map_host.len * @sizeOf([2]c_int));

        var num_copies_c: c_int = @intCast(copy_map.len);
        var block_bytes_c: c_int = @intCast(block_alloc.block_bytes);

        const func = cudaz.cuModuleGetFunction(self.module, "block_copy_f16_kernel") catch return error.KernelNotFound;
        var kp = [_]?*anyopaque{ &d_cache, &d_cache, &d_map, &num_copies_c, &block_bytes_c };
        const blocks: c_uint = @intCast((copy_map.len + 127) / 128);
        try cudaz.cuLaunchKernel(func, blocks, 1, 1, 128, 1, 1, 0, self.stream, @ptrCast(&kp), null);
        try cudaz.cuStreamSynchronize(self.stream);
        try cudaz.cuMemcpyDtoH(@intFromPtr(block_alloc.memory_pool.ptr), d_cache, block_alloc.memory_pool.len);
        try cudaz.cuCtxSynchronize();
    }
};

pub const CpuReference = struct {
    pub const decode = @import("attention.zig").PagedAttention.decode;
    pub const prefill = @import("attention.zig").PagedAttention.prefill;
};