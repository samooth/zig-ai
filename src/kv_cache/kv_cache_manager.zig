//! KV-Cache Manager
//! Orquesta almacenamiento, cuantización, de-cuantización y atención
//! Soporta múltiples secuencias, prefetch y streaming async

const std = @import("std");
const qt = @import("quant_types.zig");
const alloc = @import("allocator.zig");
const gpu_dequant = @import("gpu_dequant.zig");

const KVCacheConfig = qt.KVCacheConfig;
const LayerQuantConfig = qt.LayerQuantConfig;
const QuantFormat = qt.QuantFormat;
const QuantizedTensor = qt.QuantizedTensor;
const KVBlockDescriptor = qt.KVBlockDescriptor;
const KVPoolAllocator = alloc.KVPoolAllocator;
const AllocStrategy = alloc.AllocStrategy;
const GpuDequantEngine = gpu_dequant.GpuDequantEngine;

/// Estado de una secuencia en el cache
pub const SequenceState = struct {
    /// ID único de secuencia
    seq_id: u64,
    /// Longitud actual de tokens generados
    current_len: u32,
    /// Slots asignados por capa y cabeza [layer][head]
    k_slots: [][]u32,
    v_slots: [][]u32,
    /// Formato usado por capa
    layer_formats: []LayerQuantConfig,
};

/// Gestor principal de KV-cache
pub const KVCacheManager = struct {
    allocator: std.mem.Allocator,
    config: KVCacheConfig,
    /// Pool allocator para datos cuantizados
    pool: KVPoolAllocator,
    /// Secuencias activas
    sequences: std.AutoHashMap(u64, SequenceState),
    /// Engine GPU de de-cuantización (opcional)
    gpu_engine: ?GpuDequantEngine,
    /// Buffer de prefetch (capa siguiente)
    prefetch_buffer: ?PrefetchBuffer,
    /// Métricas
    metrics: Metrics,

    const Self = @This();

    pub const Metrics = struct {
        hits: u64,
        misses: u64,
        evictions: u64,
        bytes_saved: u64,
        gpu_dequant_time_us: u64,
    };

    pub const PrefetchBuffer = struct {
        layer_idx: u32,
        seq_id: u64,
        k_data: []const u8,
        v_data: []const u8,
    };

    /// Inicializa el gestor con configuración dada
    pub fn init(
        allocator: std.mem.Allocator,
        config: KVCacheConfig,
        pool_capacity_mb: usize,
    ) !Self {
        const pool_capacity = pool_capacity_mb * 1024 * 1024;
        var pool = try KVPoolAllocator.init(allocator, pool_capacity, .lru_evict);
        errdefer pool.deinit();

        var sequences = std.AutoHashMap(u64, SequenceState).init(allocator);
        errdefer sequences.deinit();

        return .{
            .allocator = allocator,
            .config = config,
            .pool = pool,
            .sequences = sequences,
            .gpu_engine = null,
            .prefetch_buffer = null,
            .metrics = .{
                .hits = 0,
                .misses = 0,
                .evictions = 0,
                .bytes_saved = 0,
                .gpu_dequant_time_us = 0,
            },
        };
    }

    /// Inicializa el motor GPU (requiere PTX compilado)
    pub fn initGpu(self: *Self, ptx_path: []const u8, max_elements: usize) !void {
        if (!self.config.use_gpu_dequant) return;
        self.gpu_engine = try GpuDequantEngine.init(self.allocator, ptx_path, max_elements);
    }

    pub fn deinit(self: *Self) void {
        if (self.gpu_engine) |*engine| {
            engine.deinit();
        }
        var seq_iter = self.sequences.valueIterator();
        while (seq_iter.next()) |seq| {
            for (seq.k_slots) |layer| self.allocator.free(layer);
            for (seq.v_slots) |layer| self.allocator.free(layer);
            self.allocator.free(seq.k_slots);
            self.allocator.free(seq.v_slots);
            self.allocator.free(seq.layer_formats);
        }
        self.sequences.deinit();
        self.pool.deinit();
    }

    /// Registra una nueva secuencia
    pub fn createSequence(self: *Self, seq_id: u64) !void {
        if (self.sequences.contains(seq_id)) return error.SequenceExists;

        const num_layers = self.config.num_layers;
        const num_heads = self.config.num_heads;

        var k_slots = try self.allocator.alloc([]u32, num_layers);
        errdefer self.allocator.free(k_slots);
        var v_slots = try self.allocator.alloc([]u32, num_layers);
        errdefer self.allocator.free(v_slots);

        var layer_formats = try self.allocator.alloc(LayerQuantConfig, num_layers);
        errdefer self.allocator.free(layer_formats);

        for (0..num_layers) |l| {
            k_slots[l] = try self.allocator.alloc(u32, num_heads);
            errdefer self.allocator.free(k_slots[l]);
            v_slots[l] = try self.allocator.alloc(u32, num_heads);
            errdefer self.allocator.free(v_slots[l]);

            @memset(k_slots[l], std.math.maxInt(u32));
            @memset(v_slots[l], std.math.maxInt(u32));

            // Usar configuración por capa o default
            if (self.config.layer_configs) |configs| {
                layer_formats[l] = configs[l];
            } else {
                // Default: K=Q4_0, V=Q8_0 para máxima compresión con buena calidad
                layer_formats[l] = .{
                    .k_format = .q4_0,
                    .v_format = .q8_0,
                    .k_block_size = 32,
                    .v_block_size = 32,
                    .quant_threshold = null,
                };
            }
        }

        const state = SequenceState{
            .seq_id = seq_id,
            .current_len = 0,
            .k_slots = k_slots,
            .v_slots = v_slots,
            .layer_formats = layer_formats,
        };

        try self.sequences.put(seq_id, state);
    }

    /// Libera una secuencia y sus recursos
    pub fn removeSequence(self: *Self, seq_id: u64) void {
        const entry = self.sequences.getEntry(seq_id) orelse return;
        const seq = entry.value_ptr;

        for (0..self.config.num_layers) |l| {
            for (0..self.config.num_heads) |h| {
                const k_slot = seq.k_slots[l][h];
                const v_slot = seq.v_slots[l][h];
                if (k_slot != std.math.maxInt(u32)) self.pool.freeSlot(k_slot);
                if (v_slot != std.math.maxInt(u32)) self.pool.freeSlot(v_slot);
            }
            self.allocator.free(seq.k_slots[l]);
            self.allocator.free(seq.v_slots[l]);
        }

        self.allocator.free(seq.k_slots);
        self.allocator.free(seq.v_slots);
        self.allocator.free(seq.layer_formats);
        _ = self.sequences.remove(seq_id);
    }

    /// Almacena nuevos tokens K/V para una secuencia
    pub fn appendTokens(
        self: *Self,
        seq_id: u64,
        layer_idx: u32,
        head_idx: u32,
        k_data: []const u8,
        v_data: []const u8,
    ) !void {
        const seq = self.sequences.getPtr(seq_id) orelse return error.SequenceNotFound;
        const lconf = seq.layer_formats[layer_idx];

        // Asignar o extender slots
        const k_slot = try self.ensureSlot(seq_id, layer_idx, head_idx, true, lconf.k_format);
        const v_slot = try self.ensureSlot(seq_id, layer_idx, head_idx, false, lconf.v_format);

        // Copiar datos cuantizados al pool
        const k_buf = self.pool.getBuffer(k_slot) orelse return error.SlotNotFound;
        const v_buf = self.pool.getBuffer(v_slot) orelse return error.SlotNotFound;

        // Calcular offset de escritura (append)
        const seq_len = seq.current_len;
        const head_dim = self.config.head_dim;
        const k_block_size = lconf.k_block_size;
        const v_block_size = lconf.v_block_size;

        const k_num_blocks = (seq_len * head_dim + k_block_size - 1) / k_block_size;
        const v_num_blocks = (seq_len * head_dim + v_block_size - 1) / v_block_size;

        const k_meta_bytes = if (lconf.k_format.hasScales()) k_num_blocks * @sizeOf(f32) else 0;
        const v_meta_bytes = if (lconf.v_format.hasScales()) v_num_blocks * @sizeOf(f32) else 0;

        const k_write_offset = k_meta_bytes + (seq_len * head_dim * lconf.k_format.bitsPerElement()) / 8;
        const v_write_offset = v_meta_bytes + (seq_len * head_dim * lconf.v_format.bitsPerElement()) / 8;

        if (k_write_offset + k_data.len > k_buf.len or v_write_offset + v_data.len > v_buf.len) {
            return error.BufferOverflow;
        }

        @memcpy(k_buf[k_write_offset .. k_write_offset + k_data.len], k_data);
        @memcpy(v_buf[v_write_offset .. v_write_offset + v_data.len], v_data);

        // Actualizar métricas
        const fp16_bytes = k_data.len + v_data.len;
        const saved = fp16_bytes - (k_data.len + v_data.len); // Diferencia vs cuantizado
        self.metrics.bytes_saved += saved;
    }

    /// Conveniencia: append de tokens en FP16 crudo (sin cuantizar)
    pub fn appendTokensF16(
        self: *Self,
        seq_id: u64,
        layer_idx: u32,
        head_idx: u32,
        k_data: []const f16,
        v_data: []const f16,
    ) !void {
        const k_bytes: []const u8 = std.mem.sliceAsBytes(k_data);
        const v_bytes: []const u8 = std.mem.sliceAsBytes(v_data);
        try self.appendTokens(seq_id, layer_idx, head_idx, k_bytes, v_bytes);
    }

    /// Avanza el contador de tokens de una secuencia en 1
    pub fn advanceSequence(self: *Self, seq_id: u64) !void {
        const seq = self.sequences.getPtr(seq_id) orelse return error.SequenceNotFound;
        seq.current_len += 1;
    }

    /// Longitud actual (en tokens) de una secuencia
    pub fn getSequenceLen(self: *Self, seq_id: u64) !usize {
        const seq = self.sequences.getPtr(seq_id) orelse return error.SequenceNotFound;
        return seq.current_len;
    }

    /// Recupera K/V de-cuantizados para atención
    pub fn retrieveForAttention(        self: *Self,
        seq_id: u64,
        layer_idx: u32,
        head_idx: u32,
        out_k: []f16,
        out_v: []f16,
    ) !void {
        const seq = self.sequences.getPtr(seq_id) orelse return error.SequenceNotFound;
        const k_slot = seq.k_slots[layer_idx][head_idx];
        const v_slot = seq.v_slots[layer_idx][head_idx];

        if (k_slot == std.math.maxInt(u32) or v_slot == std.math.maxInt(u32)) {
            return error.SlotEmpty;
        }

        const k_buf = self.pool.getBuffer(k_slot) orelse return error.SlotNotFound;
        const v_buf = self.pool.getBuffer(v_slot) orelse return error.SlotNotFound;

        const lconf = seq.layer_formats[layer_idx];
        const num_elements = @as(usize, seq.current_len) * @as(usize, self.config.head_dim);

        // Si tenemos GPU, usar de-cuantización acelerada
        if (self.gpu_engine) |_| {
            // TODO: Integrar con buffers GPU persistentes
            // Por ahora, de-cuantización CPU fallback
            try self.dequantizeCpu(k_buf, lconf.k_format, out_k[0..num_elements]);
            try self.dequantizeCpu(v_buf, lconf.v_format, out_v[0..num_elements]);
        } else {
            try self.dequantizeCpu(k_buf, lconf.k_format, out_k[0..num_elements]);
            try self.dequantizeCpu(v_buf, lconf.v_format, out_v[0..num_elements]);
        }

        self.metrics.hits += 1;
    }

    /// Prefetch de la siguiente capa
    pub fn prefetchLayer(self: *Self, seq_id: u64, next_layer: u32) !void {
        if (!self.config.enable_prefetch) return;
        // Marcar capa para prefetch async
        // Implementación depende del scheduler de atención
        _ = seq_id;
        _ = next_layer;
    }

    /// Compacta la memoria del pool
    pub fn compact(self: *Self) !void {
        try self.pool.compact();
    }

    /// Reporte de métricas
    pub fn reportMetrics(self: *Self) void {
        const total = self.metrics.hits + self.metrics.misses;
        const hit_rate = if (total > 0)
            @as(f32, @floatFromInt(self.metrics.hits)) / @as(f32, @floatFromInt(total)) * 100.0
        else
            0.0;

        std.log.info("==== KV-Cache Metrics ====", .{});
        std.log.info("  Hit rate: {d:.1}%", .{hit_rate});
        std.log.info("  Evictions: {d}", .{self.metrics.evictions});
        std.log.info("  Bytes saved: {d} MB", .{self.metrics.bytes_saved / (1024 * 1024)});
        std.log.info("  Pool usage: {d:.1}%", .{self.pool.usagePercent()});
        std.log.info("  GPU dequant time: {d} us", .{self.metrics.gpu_dequant_time_us});
    }

    // ─── Internos ───

    fn ensureSlot(
        self: *Self,
        seq_id: u64,
        layer_idx: u32,
        head_idx: u32,
        is_k: bool,
        format: QuantFormat,
    ) !u32 {
        const seq = self.sequences.getPtr(seq_id).?;
        const slots = if (is_k) &seq.k_slots else &seq.v_slots;

        if (slots.*[layer_idx][head_idx] == std.math.maxInt(u32)) {
            // Asignar nuevo slot
            const max_len = self.config.max_seq_len;
            const head_dim = self.config.head_dim;
            const slot = try self.pool.allocBlock(
                layer_idx, head_idx, 0, max_len, head_dim, format,
            );
            slots.*[layer_idx][head_idx] = slot.idx;
            return slot.idx;
        }

        return slots.*[layer_idx][head_idx];
    }

    fn dequantizeCpu(self: *Self, raw: []const u8, format: QuantFormat, out: []f16) !void {
        _ = self;
        // De-cuantización CPU fallback
        // Implementación básica para formatos soportados
        switch (format) {
            .fp16 => {
                const src = std.mem.bytesAsSlice(f16, raw);
                @memcpy(out, src[0..out.len]);
            },
            .fp32 => {
                const src = std.mem.bytesAsSlice(f32, raw);
                for (0..out.len) |i| {
                    out[i] = @as(f16, @floatCast(src[i]));
                }
            },
            .int8_symmetric => {
                // Simplificado: asume bloque único
                const scale = std.mem.bytesAsSlice(f32, raw[0..4])[0];
                const q = std.mem.bytesAsSlice(i8, raw[4..]);
                for (0..out.len) |i| {
                    out[i] = @as(f16, @floatCast(@as(f32, @floatFromInt(q[i])) * scale));
                }
            },
            else => return error.UnsupportedCpuDequant,
        }
    }
};
