//! CPU Offload Pipeline for KV-Cache
//! Mueve bloques fríos de VRAM a RAM del host (LRU) y los recarga bajo demanda.
//! Usa punteros opacos para evitar dependencias circulares con paged_attention.

const std = @import("std");
const disk_tier = @import("disk_tier"); // MH-8: 3-tier offload
const qt = @import("quant_types.zig");
const QuantFormat = qt.QuantFormat;
const OffloadConfig = qt.OffloadConfig;
const KVBlockDescriptor = qt.KVBlockDescriptor;
const alloc = @import("allocator.zig");
const KVPoolAllocator = alloc.KVPoolAllocator;
const crt = @import("cuda_runtime");

/// Estado de un bloque en el sistema de offload
pub const OffloadBlockState = enum {
    /// Residente en GPU
    GpuResident,
    /// Residente en CPU (host RAM)
    CpuResident,
    /// Offloaded a disco via disk_tier (MH-8)
    DiskResident,
    /// Evicted (liberado completamente)
    Evicted,
};

/// Puntero opaco a pool de bloques GPU (evita importar paged_attention)
pub const GpuBlockPoolOpaque = *anyopaque;
/// Puntero opaco a pool paginado GPU
pub const PagedGpuBlockPoolOpaque = *anyopaque;

/// Entrada de un bloque offloaded
pub const OffloadEntry = struct {
    /// Descriptor del bloque
    descriptor: KVBlockDescriptor,
    /// Estado actual
    state: OffloadBlockState,
    /// Timestamp de último acceso (para LRU)
    last_access: u64,
    /// Datos en host (si CpuResident)
    k_data_host: []u8,
    v_data_host: []u8,
    /// Scales en host (si aplica)
    k_scales_host: ?[]u8,
    v_scales_host: ?[]u8,
    /// ID del bloque físico en GPU (si GpuResident)
    gpu_phys_id: ?usize,
    /// Clave única para el mapa y la cola LRU
    key: []const u8,
    /// Ruta de disco para DiskResident (MH-8)
    disk_path: ?[]const u8 = null,
    /// Bytes en disco (si DiskResident)
    disk_length: usize = 0,

    pub fn init(allocator: std.mem.Allocator, desc: KVBlockDescriptor) OffloadEntry {
        const key = std.fmt.allocPrint(allocator, "{}:{}:{}:{}:{}:{s}", .{ desc.layer_idx, desc.head_idx, desc.seq_start, desc.seq_len, desc.head_dim, desc.format.toString() }) catch &.{};
        return .{
            .key = key,
            .descriptor = desc,
            .state = .GpuResident,
            .last_access = 0,
            .k_data_host = &.{},
            .v_data_host = &.{},
            .k_scales_host = null,
            .v_scales_host = null,
            .gpu_phys_id = null,
        };
    }

    pub fn deinit(self: *OffloadEntry, allocator: std.mem.Allocator) void {
        if (self.key.len > 0) {
            allocator.free(self.key);
        }
    }
};

/// Pipeline de offload CPU
pub const CpuOffloadPipeline = struct {
    allocator: std.mem.Allocator,
    config: OffloadConfig,
    /// Mapa de descriptor -> entrada offload
    entries: std.StringHashMap(OffloadEntry),
    /// Cola LRU (IDs de descriptores ordenados por último acceso)
    lru_queue: std.ArrayList([]const u8),
    /// Buffer de staging para H2D/D2H async
    staging_buffer: ?[]u8,
    /// Stream dedicado para offload
    offload_stream: ?crt.cudaStream_t,
    /// Métricas
    metrics: Metrics,

    const Self = @This();

    /// Tipo de función para evict GPU block
    const EvictGpuBlockFn = fn (GpuBlockPoolOpaque, *KVPoolAllocator, usize) anyerror!void;
    /// Tipo de función para evict paged GPU block
    const EvictPagedGpuBlockFn = fn (PagedGpuBlockPoolOpaque, *KVPoolAllocator, usize) anyerror!void;
    /// Tipo de función para alloc GPU block
    const AllocGpuBlockFn = fn (GpuBlockPoolOpaque, *KVPoolAllocator) anyerror!usize;
    /// Tipo de función para alloc paged GPU block
    const AllocPagedGpuBlockFn = fn (PagedGpuBlockPoolOpaque, *KVPoolAllocator) anyerror!usize;
    /// Tipo de función para stage GPU block
    const StageGpuBlockFn = fn (GpuBlockPoolOpaque, *KVPoolAllocator, usize) anyerror!void;
    /// Tipo de función para stage paged GPU block
    const StagePagedGpuBlockFn = fn (PagedGpuBlockPoolOpaque, *KVPoolAllocator, usize) anyerror!void;

    pub const Metrics = struct {
        offloads_to_cpu: u64 = 0,
        reloads_from_cpu: u64 = 0,
        evictions: u64 = 0,
        bytes_offloaded: u64 = 0,
        bytes_reloaded: u64 = 0,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        config: OffloadConfig,
        stream: ?crt.cudaStream_t,
    ) !Self {
        var entries = std.StringHashMap(OffloadEntry).init(allocator);
        errdefer entries.deinit();
        var lru_queue: std.ArrayList([]const u8) = .empty;
        errdefer lru_queue.deinit(allocator);

        var staging: ?[]u8 = null;
        if (config.enabled) {
            staging = try allocator.alloc(u8, 16 * 1024 * 1024); // 16MB staging buffer
        }

        return .{
            .allocator = allocator,
            .config = config,
            .entries = entries,
            .lru_queue = lru_queue,
            .staging_buffer = staging,
            .offload_stream = stream,
            .metrics = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        // Clear LRU queue (does not free key buffers)
        self.lru_queue.items.len = 0;
        // Deinitialize all entries (will free key buffers) and free host data
        var iter = self.entries.valueIterator();
        while (iter.next()) |entry| {
            // Free host data and scales
            self.allocator.free(entry.k_data_host);
            self.allocator.free(entry.v_data_host);
            if (entry.k_scales_host) |s| self.allocator.free(s);
            if (entry.v_scales_host) |s| self.allocator.free(s);
            // Deinitialize OffloadEntry (frees key)
            entry.*.deinit(self.allocator);
        }
        self.entries.deinit();
        self.lru_queue.deinit(self.allocator);
        if (self.staging_buffer) |buf| self.allocator.free(buf);
        if (self.offload_stream) |s| crt.streamDestroy(s);
    }
    pub fn registerBlock(self: *Self, desc: KVBlockDescriptor) !void {
        if (!self.config.enabled) return;
        const key_tmp = self.descriptorKey(desc);
        if (!self.entries.contains(key_tmp)) {
            const entry = OffloadEntry.init(self.allocator, desc);
            try self.entries.put(entry.key, entry);
            try self.lru_queue.append(self.allocator, entry.key);
        }
        if (key_tmp.len > 0) {
            self.allocator.free(key_tmp);
        }
    }

    /// Marca un bloque como accedido (actualiza LRU)
    pub fn touchBlock(self: *Self, desc: KVBlockDescriptor, timestamp: u64) void {
        if (!self.config.enabled) return;
        const key_tmp = self.descriptorKey(desc);
        if (self.entries.getPtr(key_tmp)) |entry| {
            entry.last_access = timestamp;
            // Move to front of LRU queue using the entry's key
            self.updateLru(entry.key);
        }
        if (key_tmp.len > 0) {
            self.allocator.free(key_tmp);
        }
    }

    /// Verifica si necesita hacer offload y lo ejecuta
    /// El caller debe proporcionar funciones de callback para interactuar con los pools GPU
    pub fn maybeOffload(
        self: *Self,
        block_alloc: *KVPoolAllocator,
        gpu_pool: GpuBlockPoolOpaque,
        paged_gpu_pool: PagedGpuBlockPoolOpaque,
        current_vram_free_mb: usize,
        current_timestamp: u64,
        disk_tier_opt: ?*disk_tier.DiskTier,
        // Callbacks para operaciones GPU (evitan importar paged_attention)
        evict_gpu_block: EvictGpuBlockFn,
        evict_paged_gpu_block: EvictPagedGpuBlockFn,
    ) !void {
        if (!self.config.enabled) return;
        if (current_vram_free_mb >= self.config.min_free_vram_mb) return;

        // Find candidate blocks to offload (oldest first)
        var candidates: std.ArrayList([]const u8) = .{};
        defer candidates.deinit(self.allocator);

        for (self.lru_queue.items) |key| {
            if (self.entries.getPtr(key.?)) |entry| {
                if (entry.state == .GpuResident) {
                    const age = current_timestamp - entry.last_access;
                    if (age >= self.config.min_token_age) {
                        try candidates.append(self.allocator, key);
                    }
                }
            }
        }

        // Offload oldest candidates until VRAM is sufficient
        for (candidates.items) |key| {
            if (current_vram_free_mb >= self.config.min_free_vram_mb) break;
            if (self.entries.getPtr(key.?)) |entry| {
                try self.offloadToCpu(entry, block_alloc, gpu_pool, paged_gpu_pool, evict_gpu_block, evict_paged_gpu_block);
                // Update VRAM estimate (simplified)
                const block_mb = (entry.k_data_host.len + entry.v_data_host.len) / (1024 * 1024);
                current_vram_free_mb += block_mb;
            }
        }

        // MH-8: Tier 3 — offload a disco si VRAM sigue baja
        if (disk_tier_opt) |dt| {
            var disk_candidates: std.ArrayList([]const u8) = .{};
            defer disk_candidates.deinit(self.allocator);
            for (self.lru_queue.items) |key| {
                if (self.entries.getPtr(key.?)) |entry| {
                    if (entry.state == .CpuResident and entry.k_data_host.len > 0) {
                        const age = current_timestamp - entry.last_access;
                        if (age >= self.config.min_token_age) {
                            try disk_candidates.append(self.allocator, key);
                        }
                    }
                }
            }
            for (disk_candidates.items) |key| {
                if (current_vram_free_mb >= self.config.min_free_vram_mb) break;
                if (self.entries.getPtr(key.?)) |entry| {
                    try self.offloadToDisk(entry, dt);
                }
            }
        }
    }

    // MH-8: Offload un bloque a disco via disk_tier
    pub fn offloadToDisk(self: *Self, entry: *OffloadEntry, dt: *disk_tier.DiskTier) !void {
        if (!self.config.enabled) return;
        if (entry.state != .CpuResident) return;
        if (entry.k_data_host.len == 0) return;

        const key = self.descriptorKey(entry.descriptor);
        defer if (key.len > 0) self.allocator.free(key);

        // Serializar K+V a buffer plano (MH-8: layout KV contiguo)
        const k_len = entry.k_data_host.len;
        const v_len = entry.v_data_host.len;
        const total_len = k_len + v_len;
        var buf = try self.allocator.alloc(u8, total_len);
        errdefer self.allocator.free(buf);
        @memcpy(buf[0..k_len], entry.k_data_host);
        @memcpy(buf[k_len..][0..v_len], entry.v_data_host);

        // Registrar en disk_tier (expert_id = hash del key para KV)
        const expert_id = blk: {
            var h: u32 = 0;
            for (key) |c| h = h * 31 + c;
            break :blk h;
        };
        try dt.registerExpert(expert_id, buf);

        // Liberar buffer CPU (ya está en disco)
        self.allocator.free(entry.k_data_host);
        self.allocator.free(entry.v_data_host);
        if (entry.k_scales_host) |s| self.allocator.free(s);
        if (entry.v_scales_host) |s| self.allocator.free(s);
        entry.k_data_host = &.{};
        entry.v_data_host = &.{};
        entry.k_scales_host = null;
        entry.v_scales_host = null;
        entry.disk_path = key;
        entry.disk_length = total_len;
        entry.state = .DiskResident;
        self.metrics.offloads_to_disk += 1;
    }

    // MH-8: Reload bloque desde disco a CPU
    pub fn reloadFromDisk(self: *Self, entry: *OffloadEntry, dt: *disk_tier.DiskTier) !void {
        if (!self.config.enabled) return;
        if (entry.state != .DiskResident) return;
        const expert_id = blk: {
            var h: u32 = 0;
            for (entry.key) |c| h = h * 31 + c;
            break :blk h;
        };

        const buf = try dt.readExpert(self.allocator, expert_id);
        defer dt.releaseExpert(buf);
        const k_len = entry.disk_length / 2;
        const v_len = entry.disk_length - k_len;

        entry.k_data_host = try self.allocator.alloc(u8, k_len);
        entry.v_data_host = try self.allocator.alloc(u8, v_len);
        @memcpy(entry.k_data_host, buf[0..k_len]);
        @memcpy(entry.v_data_host, buf[k_len..][0..v_len]);
        entry.state = .CpuResident;
        entry.disk_path = null;
        entry.disk_length = 0;
        self.metrics.reloads_from_disk += 1;
    }

    /// Offload un bloque específico a CPU
    fn offloadToCpu(
        self: *Self,
        entry: *OffloadEntry,
        block_alloc: *KVPoolAllocator,
        gpu_pool: GpuBlockPoolOpaque,
        paged_gpu_pool: PagedGpuBlockPoolOpaque,
        evict_gpu_block: EvictGpuBlockFn,
        evict_paged_gpu_block: EvictPagedGpuBlockFn,
    ) !void {
        // Copy K/V data from host pool to offload buffers
        const k_buf = block_alloc.getBuffer(entry.descriptor.k_slot.?) orelse return;
        const v_buf = block_alloc.getBuffer(entry.descriptor.v_slot.?) orelse return;

        entry.k_data_host = try self.allocator.dup(u8, k_buf);
        entry.v_data_host = try self.allocator.dup(u8, v_buf);

        // If quantized, also copy scales (stored separately in paged pool)
        // For simplicity, we assume scales are part of the buffer or handled by caller

        // Free GPU memory
        if (entry.gpu_phys_id) |phys_id| {
            if (paged_gpu_pool != 0) {
                try evict_paged_gpu_block(@ptrCast(@alignCast(paged_gpu_pool)), block_alloc, phys_id);
            } else if (gpu_pool != 0) {
                try evict_gpu_block(@ptrCast(@alignCast(gpu_pool)), block_alloc, phys_id);
            }
            entry.gpu_phys_id = null;
        }

        entry.state = .CpuResident;
        self.metrics.offloads_to_cpu += 1;
        self.metrics.bytes_offloaded += entry.k_data_host.len + entry.v_data_host.len;
    }

    /// Recarga un bloque desde CPU a GPU
    /// Retorna el nuevo ID físico en GPU
    pub fn reloadFromCpu(
        self: *Self,
        desc: KVBlockDescriptor,
        block_alloc: *KVPoolAllocator,
        gpu_pool: GpuBlockPoolOpaque,
        paged_gpu_pool: PagedGpuBlockPoolOpaque,
        // Callbacks para operaciones GPU
        alloc_gpu_block: AllocGpuBlockFn,
        alloc_paged_gpu_block: AllocPagedGpuBlockFn,
        stage_gpu_block: StageGpuBlockFn,
        stage_paged_gpu_block: StagePagedGpuBlockFn,
    ) !usize {
        if (!self.config.enabled) return 0;

        const key = self.descriptorKey(desc);
        defer if (key.len > 0) self.allocator.free(key);
        const entry = self.entries.getPtr(key) orelse return 0;

        if (entry.state != .CpuResident) return 0;

        // Allocate new GPU block via callback
        const new_phys = if (paged_gpu_pool != 0) {
            try alloc_paged_gpu_block(@ptrCast(@alignCast(paged_gpu_pool)), block_alloc);
        } else if (gpu_pool != 0) {
            try alloc_gpu_block(@ptrCast(@alignCast(gpu_pool)), block_alloc);
        } else {
            return 0;
        };

        // Copy data back to GPU via callback
        if (paged_gpu_pool != 0) {
            try stage_paged_gpu_block(@ptrCast(@alignCast(paged_gpu_pool)), block_alloc, new_phys);
        } else if (gpu_pool != 0) {
            try stage_gpu_block(@ptrCast(@alignCast(gpu_pool)), block_alloc, new_phys);
        }

        entry.gpu_phys_id = new_phys;
        entry.state = .GpuResident;
        entry.last_access = std.time.timestamp();
        self.updateLru(key);

        self.metrics.reloads_from_cpu += 1;
        self.metrics.bytes_reloaded += entry.k_data_host.len + entry.v_data_host.len;

        // Free host memory
        self.allocator.free(entry.k_data_host);
        self.allocator.free(entry.v_data_host);
        entry.k_data_host = &.{};
        entry.v_data_host = &.{};
        if (entry.k_scales_host) |s| self.allocator.free(s);
        if (entry.v_scales_host) |s| self.allocator.free(s);
        entry.k_scales_host = null;
        entry.v_scales_host = null;

        return new_phys;
    }

    /// Evict un bloque completamente (libera tanto GPU como CPU)
    pub fn evictBlock(self: *Self, desc: KVBlockDescriptor, _block_alloc: *KVPoolAllocator) !void {
        _ = _block_alloc;
        if (!self.config.enabled) return;

        const key = self.descriptorKey(desc);
        defer if (key.len > 0) self.allocator.free(key);
        const entry = self.entries.getPtr(key) orelse return;

        // Free host memory and scales
        self.allocator.free(entry.k_data_host);
        self.allocator.free(entry.v_data_host);
        if (entry.k_scales_host) |s| self.allocator.free(s);
        if (entry.v_scales_host) |s| self.allocator.free(s);
        // Set to empty/null to avoid double-free if something else tries to free them
        entry.k_data_host = &.{};
        entry.v_data_host = &.{};
        entry.k_scales_host = null;
        entry.v_scales_host = null;

        if (entry.state == .GpuResident) {
            if (entry.gpu_phys_id) |_| {
                // Note: caller should handle GPU eviction via callbacks if needed
                entry.gpu_phys_id = null;
            }
        }

        // Remove key from LRU queue
        self.removeKeyFromLru(entry.key);
        // Deinitialize the OffloadEntry (this frees the key buffer)
        entry.*.deinit(self.allocator);
        // Remove entry from map
        self.entries.remove(key);

        entry.state = .Evicted;
        self.metrics.evictions += 1;
    }

    /// Genera clave única para un descriptor
    fn descriptorKey(self: *Self, desc: KVBlockDescriptor) []const u8 {
        // layer_idx:head_idx:seq_start:seq_len:head_dim:format
        return std.fmt.allocPrint(self.allocator, "{}:{}:{}:{}:{}:{s}", .{ desc.layer_idx, desc.head_idx, desc.seq_start, desc.seq_len, desc.head_dim, desc.format.toString() }) catch &.{};
    }

    /// Actualiza posición en cola LRU
    fn updateLru(self: *Self, key: []const u8) void {
        // Remove from current position
        var idx: usize = 0;
        var found = false;
        for (self.lru_queue.items, 0..) |item, i| {
            if (std.mem.eql(u8, item, key)) {
                idx = i;
                found = true;
                break;
            }
        }
        if (found) {
            // Shift elements left to remove the item at idx
            for (idx..self.lru_queue.items.len - 1) |i| {
                self.lru_queue.items[i] = self.lru_queue.items[i + 1];
            }
            // Move the removed item to the end
            self.lru_queue.items[self.lru_queue.items.len - 1] = key;
        } else {
            // Key not found, add to end (should only happen for new blocks)
            self.lru_queue.append(self.allocator, key) catch {};
        }
    }
    /// Removes a key from the LRU queue, if present. Does not free the buffer.
    fn removeKeyFromLru(self: *Self, key: []const u8) bool {
        // Remove from current position
        var idx: usize = 0;
        var found = false;
        for (self.lru_queue.items, 0..) |item, i| {
            if (std.mem.eql(u8, item, key)) {
                idx = i;
                found = true;
                break;
            }
        }
        if (found) {
            // Shift elements left to remove the item at idx
            for (idx..self.lru_queue.items.len - 1) |i| {
                self.lru_queue.items[i] = self.lru_queue.items[i + 1];
            }
            self.lru_queue.items.len -= 1;
            return true;
        }
        return false;
    }

    /// Reporta métricas
    pub fn reportMetrics(self: *Self) void {
        std.log.info("==== CPU Offload Metrics ====", .{});
        std.log.info("  Offloads to CPU: {}", .{self.metrics.offloads_to_cpu});
        std.log.info("  Reloads from CPU: {}", .{self.metrics.reloads_from_cpu});
        std.log.info("  Evictions: {}", .{self.metrics.evictions});
        std.log.info("  Bytes offloaded: {} MB", .{self.metrics.bytes_offloaded / (1024 * 1024)});
        std.log.info("  Bytes reloaded: {} MB", .{self.metrics.bytes_reloaded / (1024 * 1024)});
        std.log.info("  Tracked blocks: {}", .{self.entries.count()});
    }
};

test {
    std.testing.refAllDecls(@This());
}
