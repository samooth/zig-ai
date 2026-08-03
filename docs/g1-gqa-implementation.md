# G1: Implementación GQA / MQA en zig-kv-cache-gpu

> **Paso 1 del plan de integración**  
> **Fecha:** 2026-08-02

---

## 1. Cambios en `src/kv_cache/quant_types.zig`

Añadimos `num_kv_heads` a `KVCacheConfig` y actualizamos constructores.

```zig
//! Tipos de datos para KV-cache cuantizado

const std = @import("std");

/// Formatos de cuantización soportados
pub const QuantFormat = enum {
    fp16, fp32, int8_symmetric, int8_asymmetric, int4, q4_0, q8_0,

    pub fn bitsPerElement(self: QuantFormat) u8 {
        return switch (self) {
            .fp16 => 16, .fp32 => 32,
            .int8_symmetric, .int8_asymmetric, .q8_0 => 8,
            .int4, .q4_0 => 4,
        };
    }

    pub fn defaultBlockSize(self: QuantFormat) usize {
        return switch (self) {
            .fp16, .fp32 => 1,
            .int8_symmetric, .int8_asymmetric => 64,
            .int4 => 64,
            .q4_0, .q8_0 => 32,
        };
    }

    pub fn hasScales(self: QuantFormat) bool {
        return switch (self) { .fp16, .fp32 => false, else => true };
    }

    pub fn hasZeroPoints(self: QuantFormat) bool {
        return switch (self) { .int8_asymmetric, .int4 => true, else => false };
    }

    pub fn bytesPerBlock(self: QuantFormat) usize {
        return switch (self) {
            .fp16 => 2, .fp32 => 4,
            .int8_symmetric => 64 + 4,
            .int8_asymmetric => 64 + 8,
            .int4 => 32 + 8,
            .q4_0 => 18,
            .q8_0 => 34,
        };
    }
};

pub const QuantizedTensor = struct {
    format: QuantFormat,
    raw: []const u8,
    scales: ?[]const f32,
    zero_points: ?[]const f32,
    num_elements: usize,
    block_size: usize,
    num_blocks: usize,

    pub fn init(format: QuantFormat, raw: []const u8, scales: ?[]const f32,
                zero_points: ?[]const f32, num_elements: usize, block_size: usize) QuantizedTensor {
        const num_blocks = (num_elements + block_size - 1) / block_size;
        return .{ .format = format, .raw = raw, .scales = scales, .zero_points = zero_points,
                  .num_elements = num_elements, .block_size = block_size, .num_blocks = num_blocks };
    }

    pub fn totalBytes(self: QuantizedTensor) usize {
        var total = self.raw.len;
        if (self.scales) |s| total += s.len * @sizeOf(f32);
        if (self.zero_points) |z| total += z.len * @sizeOf(f32);
        return total;
    }

    pub fn compressionRatio(self: QuantizedTensor) f32 {
        const fp16_bytes = self.num_elements * 2;
        return @as(f32, @floatFromInt(fp16_bytes)) / @as(f32, @floatFromInt(self.totalBytes()));
    }
};

pub const KVBlockDescriptor = struct {
    layer_idx: u32, head_idx: u32, seq_start: u32, seq_len: u32,
    head_dim: u32, format: QuantFormat, byte_offset: usize, byte_size: usize,
};

pub const LayerQuantConfig = struct {
    k_format: QuantFormat, v_format: QuantFormat,
    k_block_size: usize, v_block_size: usize, quant_threshold: ?usize,
};

/// Configuración global de KV-cache
/// Ahora incluye num_kv_heads para GQA/MQA
pub const KVCacheConfig = struct {
    num_layers: u32,
    /// Número de cabezas de Query (atención)
    num_heads: u32,
    /// Número de cabezas de Key/Value (GQA: num_kv_heads <= num_heads)
    num_kv_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
    layer_configs: ?[]const LayerQuantConfig,
    use_gpu_dequant: bool,
    enable_prefetch: bool,
    enable_streaming: bool,

    /// Ratio de agrupación: cuántos Q-heads comparten un KV-head
    pub fn gqaGroupSize(self: KVCacheConfig) u32 {
        std.debug.assert(self.num_heads >= self.num_kv_heads);
        std.debug.assert(self.num_heads % self.num_kv_heads == 0);
        return self.num_heads / self.num_kv_heads;
    }

    /// Mapea un índice de Q-head a su KV-head correspondiente
    pub fn qHeadToKvHead(self: KVCacheConfig, q_head_idx: u32) u32 {
        return q_head_idx * self.num_kv_heads / self.num_heads;
    }

    pub fn default(num_layers: u32, num_heads: u32, num_kv_heads: u32,
                   head_dim: u32, max_seq_len: u32) KVCacheConfig {
        return .{
            .num_layers = num_layers, .num_heads = num_heads,
            .num_kv_heads = num_kv_heads, .head_dim = head_dim,
            .max_seq_len = max_seq_len, .layer_configs = null,
            .use_gpu_dequant = true, .enable_prefetch = true,
            .enable_streaming = true,
        };
    }

    pub fn aggressive(num_layers: u32, num_heads: u32, num_kv_heads: u32,
                      head_dim: u32, max_seq_len: u32) KVCacheConfig {
        return .{
            .num_layers = num_layers, .num_heads = num_heads,
            .num_kv_heads = num_kv_heads, .head_dim = head_dim,
            .max_seq_len = max_seq_len, .layer_configs = null,
            .use_gpu_dequant = true, .enable_prefetch = true,
            .enable_streaming = true,
        };
    }

    /// Tamaño estimado en bytes para una secuencia completa
    /// Ahora usa num_kv_heads para K+V (no num_heads)
    pub fn estimatedSize(self: KVCacheConfig, format: QuantFormat) usize {
        const elements_per_layer = @as(usize, self.num_kv_heads)
                                 * @as(usize, self.max_seq_len)
                                 * @as(usize, self.head_dim);
        const elements_total = elements_per_layer * self.num_layers * 2; // K + V
        const bits = format.bitsPerElement();
        return (elements_total * bits) / 8;
    }

    /// Bytes por token (K+V) usando num_kv_heads
    pub fn bytesPerToken(self: KVCacheConfig, format: QuantFormat) usize {
        const elements = @as(usize, self.num_layers)
                       * @as(usize, self.num_kv_heads)
                       * @as(usize, self.head_dim);
        return (elements * format.bitsPerElement()) / 8 * 2; // K + V
    }
};

pub const CacheSlot = struct {
    idx: u32, occupied: bool, ref_count: u32,
    last_access: u64, descriptor: KVBlockDescriptor,
};
```

---

## 2. Cambios en `src/kv_cache/kv_cache_manager.zig`

El manager ahora opera sobre `num_kv_heads` para K/V. Las funciones públicas reciben `q_head_idx` y mapean internamente.

```zig
//! KV-Cache Manager con soporte GQA/MQA

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
/// k_slots y v_slots se indexan por [layer][kv_head] (no q_head)
pub const SequenceState = struct {
    seq_id: u64,
    current_len: u32,
    /// Slots por [layer][kv_head]
    k_slots: [][]u32,
    v_slots: [][]u32,
    layer_formats: []LayerQuantConfig,
};

pub const KVCacheManager = struct {
    allocator: std.mem.Allocator,
    config: KVCacheConfig,
    pool: KVPoolAllocator,
    sequences: std.AutoHashMap(u64, SequenceState),
    gpu_engine: ?GpuDequantEngine,
    prefetch_buffer: ?PrefetchBuffer,
    metrics: Metrics,

    const Self = @This();

    pub const Metrics = struct {
        hits: u64, misses: u64, evictions: u64,
        bytes_saved: u64, gpu_dequant_time_us: u64,
    };

    pub const PrefetchBuffer = struct {
        layer_idx: u32, seq_id: u64,
        k_data: []const u8, v_data: []const u8,
    };

    pub fn init(allocator: std.mem.Allocator, config: KVCacheConfig,
                pool_capacity_mb: usize) !Self {
        const pool_capacity = pool_capacity_mb * 1024 * 1024;
        var pool = try KVPoolAllocator.init(allocator, pool_capacity, .lru_evict);
        errdefer pool.deinit();
        var sequences = std.AutoHashMap(u64, SequenceState).init(allocator);
        errdefer sequences.deinit();
        return .{
            .allocator = allocator, .config = config, .pool = pool,
            .sequences = sequences, .gpu_engine = null,
            .prefetch_buffer = null,
            .metrics = .{ .hits = 0, .misses = 0, .evictions = 0,
                          .bytes_saved = 0, .gpu_dequant_time_us = 0 },
        };
    }

    pub fn initGpu(self: *Self, ptx_path: []const u8, max_elements: usize) !void {
        if (!self.config.use_gpu_dequant) return;
        self.gpu_engine = try GpuDequantEngine.init(self.allocator, ptx_path, max_elements);
    }

    pub fn deinit(self: *Self) void {
        if (self.gpu_engine) |*engine| engine.deinit();
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

    /// Crea una nueva secuencia.
    /// Reserva slots para num_kv_heads (no num_heads).
    pub fn createSequence(self: *Self, seq_id: u64) !void {
        if (self.sequences.contains(seq_id)) return error.SequenceExists;

        const num_layers = self.config.num_layers;
        const num_kv_heads = self.config.num_kv_heads;

        var k_slots = try self.allocator.alloc([]u32, num_layers);
        errdefer self.allocator.free(k_slots);
        var v_slots = try self.allocator.alloc([]u32, num_layers);
        errdefer self.allocator.free(v_slots);
        var layer_formats = try self.allocator.alloc(LayerQuantConfig, num_layers);
        errdefer self.allocator.free(layer_formats);

        for (0..num_layers) |l| {
            k_slots[l] = try self.allocator.alloc(u32, num_kv_heads);
            errdefer self.allocator.free(k_slots[l]);
            v_slots[l] = try self.allocator.alloc(u32, num_kv_heads);
            errdefer self.allocator.free(v_slots[l]);

            @memset(k_slots[l], std.math.maxInt(u32));
            @memset(v_slots[l], std.math.maxInt(u32));

            if (self.config.layer_configs) |configs| {
                layer_formats[l] = configs[l];
            } else {
                // Default: K=Q4_0, V=Q8_0
                layer_formats[l] = .{
                    .k_format = .q4_0, .v_format = .q8_0,
                    .k_block_size = 32, .v_block_size = 32,
                    .quant_threshold = null,
                };
            }
        }

        try self.sequences.put(seq_id, .{
            .seq_id = seq_id, .current_len = 0,
            .k_slots = k_slots, .v_slots = v_slots,
            .layer_formats = layer_formats,
        });
    }

    pub fn removeSequence(self: *Self, seq_id: u64) void {
        const entry = self.sequences.getEntry(seq_id) orelse return;
        const seq = entry.value_ptr;
        for (0..self.config.num_layers) |l| {
            for (0..self.config.num_kv_heads) |h| {
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

    /// Almacena K/V para un token nuevo.
    /// `q_head_idx` es el índice de Q-head; se mapea internamente a kv_head.
    pub fn appendTokens(
        self: *Self,
        seq_id: u64,
        layer_idx: u32,
        q_head_idx: u32,  // ← NUEVO: recibe q_head_idx, mapea a kv_head
        k_data: []const u8,
        v_data: []const u8,
    ) !void {
        const seq = self.sequences.getPtr(seq_id) orelse return error.SequenceNotFound;
        const lconf = seq.layer_formats[layer_idx];

        // GQA: mapear q_head_idx → kv_head_idx
        const kv_head_idx = self.config.qHeadToKvHead(q_head_idx);

        const k_slot = try self.ensureSlot(seq_id, layer_idx, kv_head_idx, true, lconf.k_format);
        const v_slot = try self.ensureSlot(seq_id, layer_idx, kv_head_idx, false, lconf.v_format);

        const k_buf = self.pool.getBuffer(k_slot) orelse return error.SlotNotFound;
        const v_buf = self.pool.getBuffer(v_slot) orelse return error.SlotNotFound;

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

        if (k_write_offset + k_data.len > k_buf.len or
            v_write_offset + v_data.len > v_buf.len) {
            return error.BufferOverflow;
        }

        @memcpy(k_buf[k_write_offset .. k_write_offset + k_data.len], k_data);
        @memcpy(v_buf[v_write_offset .. v_write_offset + v_data.len], v_data);

        const fp16_bytes = k_data.len + v_data.len;
        const saved = fp16_bytes - (k_data.len + v_data.len);
        self.metrics.bytes_saved += saved;
    }

    /// Recupera K/V de-cuantizados para atención.
    /// `q_head_idx` es el índice de Q-head; se mapea internamente a kv_head.
    /// El output debe tener espacio para `seq.current_len * head_dim` elementos.
    pub fn retrieveForAttention(
        self: *Self,
        seq_id: u64,
        layer_idx: u32,
        q_head_idx: u32,  // ← NUEVO: recibe q_head_idx
        out_k: []f16,
        out_v: []f16,
    ) !void {
        const seq = self.sequences.getPtr(seq_id) orelse return error.SequenceNotFound;

        // GQA: mapear q_head_idx → kv_head_idx
        const kv_head_idx = self.config.qHeadToKvHead(q_head_idx);

        const k_slot = seq.k_slots[layer_idx][kv_head_idx];
        const v_slot = seq.v_slots[layer_idx][kv_head_idx];

        if (k_slot == std.math.maxInt(u32) or v_slot == std.math.maxInt(u32)) {
            return error.SlotEmpty;
        }

        const k_buf = self.pool.getBuffer(k_slot) orelse return error.SlotNotFound;
        const v_buf = self.pool.getBuffer(v_slot) orelse return error.SlotNotFound;

        const lconf = seq.layer_formats[layer_idx];
        const num_elements = @as(usize, seq.current_len) * @as(usize, self.config.head_dim);

        // TODO: GPU dequant cuando esté integrado
        try self.dequantizeCpu(k_buf, lconf.k_format, out_k[0..num_elements]);
        try self.dequantizeCpu(v_buf, lconf.v_format, out_v[0..num_elements]);

        self.metrics.hits += 1;
    }

    /// Obtiene el buffer raw cuantizado para un KV-head específico.
    /// Útil para de-cuantización externa (ej. en kernel FA).
    pub fn retrieveRaw(
        self: *Self,
        seq_id: u64,
        layer_idx: u32,
        q_head_idx: u32,
    ) !struct { k_buf: []u8, v_buf: []u8, format_k: QuantFormat, format_v: QuantFormat, len: u32 } {
        const seq = self.sequences.getPtr(seq_id) orelse return error.SequenceNotFound;
        const kv_head_idx = self.config.qHeadToKvHead(q_head_idx);

        const k_slot = seq.k_slots[layer_idx][kv_head_idx];
        const v_slot = seq.v_slots[layer_idx][kv_head_idx];
        if (k_slot == std.math.maxInt(u32) or v_slot == std.math.maxInt(u32))
            return error.SlotEmpty;

        const k_buf = self.pool.getBuffer(k_slot) orelse return error.SlotNotFound;
        const v_buf = self.pool.getBuffer(v_slot) orelse return error.SlotNotFound;

        return .{
            .k_buf = k_buf, .v_buf = v_buf,
            .format_k = seq.layer_formats[layer_idx].k_format,
            .format_v = seq.layer_formats[layer_idx].v_format,
            .len = seq.current_len,
        };
    }

    /// Incrementa la longitud de secuencia tras un append global.
    /// Llamar UNA VEZ por token (no por head).
    pub fn advanceSequence(self: *Self, seq_id: u64) !void {
        const seq = self.sequences.getPtr(seq_id) orelse return error.SequenceNotFound;
        if (seq.current_len >= self.config.max_seq_len) return error.SequenceMaxLength;
        seq.current_len += 1;
    }

    pub fn prefetchLayer(self: *Self, seq_id: u64, next_layer: u32) !void {
        if (!self.config.enable_prefetch) return;
        _ = seq_id; _ = next_layer;
    }

    pub fn compact(self: *Self) !void {
        try self.pool.compact();
    }

    pub fn reportMetrics(self: *Self) void {
        const total = self.metrics.hits + self.metrics.misses;
        const hit_rate = if (total > 0)
            @as(f32, @floatFromInt(self.metrics.hits)) / @as(f32, @floatFromInt(total)) * 100.0
        else 0.0;
        std.log.info("==== KV-Cache Metrics ====", .{});
        std.log.info("  Hit rate: {d:.1}%", .{hit_rate});
        std.log.info("  Evictions: {d}", .{self.metrics.evictions});
        std.log.info("  Bytes saved: {d} MB", .{self.metrics.bytes_saved / (1024 * 1024)});
        std.log.info("  Pool usage: {d:.1}%", .{self.pool.usagePercent()});
    }

    // ─── Internos ───

    fn ensureSlot(self: *Self, seq_id: u64, layer_idx: u32,
                  kv_head_idx: u32, is_k: bool, format: QuantFormat) !u32 {
        const seq = self.sequences.getPtr(seq_id).?;
        const slots = if (is_k) &seq.k_slots else &seq.v_slots;

        if (slots.*[layer_idx][kv_head_idx] == std.math.maxInt(u32)) {
            const max_len = self.config.max_seq_len;
            const head_dim = self.config.head_dim;
            const slot = try self.pool.allocBlock(
                layer_idx, @as(u32, @intCast(kv_head_idx)),
                0, max_len, head_dim, format,
            );
            slots.*[layer_idx][kv_head_idx] = slot.idx;
            return slot.idx;
        }
        return slots.*[layer_idx][kv_head_idx];
    }

    fn dequantizeCpu(self: *Self, raw: []const u8, format: QuantFormat, out: []f16) !void {
        _ = self;
        switch (format) {
            .fp16 => {
                const src = std.mem.bytesAsSlice(f16, raw);
                @memcpy(out, src[0..out.len]);
            },
            .fp32 => {
                const src = std.mem.bytesAsSlice(f32, raw);
                for (0..out.len) |i| out[i] = @as(f16, @floatCast(src[i]));
            },
            .int8_symmetric => {
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
```

---

## 3. Test de GQA

```zig
// tests/test_gqa.zig

const std = @import("std");
const kvc = @import("kv_cache");

const KVCacheManager = kvc.KVCacheManager;
const KVCacheConfig = kvc.KVCacheConfig;
const QuantFormat = kvc.QuantFormat;

test "gqa config mapping" {
    // Llama-3-8B: 32 Q-heads, 8 KV-heads
    const config = KVCacheConfig.default(32, 32, 8, 128, 4096);

    try std.testing.expectEqual(@as(u32, 4), config.gqaGroupSize());

    // q_head 0,1,2,3 → kv_head 0
    try std.testing.expectEqual(@as(u32, 0), config.qHeadToKvHead(0));
    try std.testing.expectEqual(@as(u32, 0), config.qHeadToKvHead(1));
    try std.testing.expectEqual(@as(u32, 0), config.qHeadToKvHead(2));
    try std.testing.expectEqual(@as(u32, 0), config.qHeadToKvHead(3));

    // q_head 4,5,6,7 → kv_head 1
    try std.testing.expectEqual(@as(u32, 1), config.qHeadToKvHead(4));
    try std.testing.expectEqual(@as(u32, 1), config.qHeadToKvHead(5));
    try std.testing.expectEqual(@as(u32, 1), config.qHeadToKvHead(6));
    try std.testing.expectEqual(@as(u32, 1), config.qHeadToKvHead(7));

    // q_head 31 → kv_head 7
    try std.testing.expectEqual(@as(u32, 7), config.qHeadToKvHead(31));
}

test "gqa memory savings" {
    const allocator = std.testing.allocator;

    // Config MHA (num_kv_heads = num_heads)
    const mha = KVCacheConfig.default(32, 32, 32, 128, 4096);
    const mha_bytes = mha.estimatedSize(.q4_0);

    // Config GQA (num_kv_heads = 8)
    const gqa = KVCacheConfig.default(32, 32, 8, 128, 4096);
    const gqa_bytes = gqa.estimatedSize(.q4_0);

    // GQA debe usar 4x menos memoria que MHA
    try std.testing.expectEqual(@as(usize, mha_bytes / 4), gqa_bytes);

    // Bytes por token
    const mha_per_token = mha.bytesPerToken(.q4_0);
    const gqa_per_token = gqa.bytesPerToken(.q4_0);
    try std.testing.expectEqual(@as(usize, mha_per_token / 4), gqa_per_token);
}

test "gqa sequence slots" {
    const allocator = std.testing.allocator;

    // GQA: 32 heads Q, 8 heads KV
    const config = KVCacheConfig.default(4, 32, 8, 128, 1024);
    var mgr = try KVCacheManager.init(allocator, config, 64);
    defer mgr.deinit();

    try mgr.createSequence(1);

    // Simular append para múltiples Q-heads del mismo grupo
    // Todos los q_heads 0-3 deben mapear al mismo kv_head 0
    const dummy_k = &[_]u8{0x00} ** 32;
    const dummy_v = &[_]u8{0x00} ** 64;

    try mgr.appendTokens(1, 0, 0, dummy_k, dummy_v); // q_head 0 → kv_head 0
    try mgr.appendTokens(1, 0, 1, dummy_k, dummy_v); // q_head 1 → kv_head 0
    try mgr.appendTokens(1, 0, 2, dummy_k, dummy_v); // q_head 2 → kv_head 0
    try mgr.appendTokens(1, 0, 3, dummy_k, dummy_v); // q_head 3 → kv_head 0

    // Avanzar secuencia 1 token
    try mgr.advanceSequence(1);

    // Retrieve para q_head 2 → debe devolver datos del kv_head 0
    var out_k = try allocator.alloc(f16, 128);
    defer allocator.free(out_k);
    var out_v = try allocator.alloc(f16, 128);
    defer allocator.free(out_v);

    try mgr.retrieveForAttention(1, 0, 2, out_k, out_v);

    // Retrieve para q_head 6 → kv_head 1, debe fallar (no hay datos)
    var out_k2 = try allocator.alloc(f16, 128);
    defer allocator.free(out_k2);
    var out_v2 = try allocator.alloc(f16, 128);
    defer allocator.free(out_v2);

    try std.testing.expectError(error.SlotEmpty, mgr.retrieveForAttention(1, 0, 6, out_k2, out_v2));
}

test "gqa raw retrieve" {
    const allocator = std.testing.allocator;
    const config = KVCacheConfig.default(2, 16, 4, 64, 512);
    var mgr = try KVCacheManager.init(allocator, config, 32);
    defer mgr.deinit();

    try mgr.createSequence(42);

    const dummy = &[_]u8{0xAB} ** 16;
    try mgr.appendTokens(42, 0, 0, dummy, dummy);
    try mgr.appendTokens(42, 0, 4, dummy, dummy); // q_head 4 → kv_head 1
    try mgr.advanceSequence(42);

    // q_head 0 → kv_head 0
    const raw0 = try mgr.retrieveRaw(42, 0, 0);
    try std.testing.expectEqual(@as(u32, 1), raw0.len);

    // q_head 1 → también kv_head 0 (mismo grupo)
    const raw1 = try mgr.retrieveRaw(42, 0, 1);
    try std.testing.expectEqual(raw0.k_buf.ptr, raw1.k_buf.ptr); // Mismo slot

    // q_head 4 → kv_head 1 (distinto grupo)
    const raw4 = try mgr.retrieveRaw(42, 0, 4);
    try std.testing.expect(raw0.k_buf.ptr != raw4.k_buf.ptr); // Slot distinto
}
```

---

## 4. Resumen de Cambios

| Archivo | Cambio |
|---------|--------|
| `quant_types.zig` | Añade `num_kv_heads`, `gqaGroupSize()`, `qHeadToKvHead()`, `bytesPerToken()` |
| `kv_cache_manager.zig` | Slots por `num_kv_heads`, `appendTokens`/`retrieveForAttention` reciben `q_head_idx` y mapean, nuevo `retrieveRaw()`, `advanceSequence()` |
| `test_gqa.zig` | 5 tests: mapping, memoria, slots, retrieve, raw retrieve |

## 5. Próximo paso

Con GQA implementado, el siguiente paso es **G2 — Cuantización CPU** (`quant_ops.zig`):
- `quantizeQ4_0(src: []f16, dst: []u8)`
- `quantizeQ8_0(src: []f16, dst: []u8)`
- `quantizeInt8Symmetric(src: []f16, dst: []u8)`
- Modificar `appendTokens` para recibir `[]f16` en lugar de `[]u8`
