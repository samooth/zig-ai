//! KV-Cache Manager
//! Orquesta almacenamiento, cuantización, de-cuantización y atención
//! Soporta múltiples secuencias, prefetch y streaming async.
//!
//! ## KVCPT (lane-b2 P0.2 integration)
//!
//! Cuando `config.tail_tokens > 0`, cada `appendTokensF16` escribe en dos
//! destinos:
//!   - **Body** (comprimido, formato de la capa): el cache histórico
//!     cuantizado estándar.
//!   - **Exact ring** (f16/bf16 según `config.tail_type`): los últimos
//!     `tail_tokens` tokens en un ring buffer f16 (128 tokens por slot),
//!     mantenidos por el `KVPoolAllocator`.
//!
//! El rollback especulativo (`rollbackN`) decrementa `current_len` y
//! recompone el ring desde la cabeza — el body se queda intacto para los
//! tokens ya commiteados (no se reescriben records).

const std = @import("std");
const qt = @import("quant_types.zig");
const alloc = @import("allocator.zig");
const gpu_dequant = @import("gpu_dequant.zig");
const kv_quant = @import("kv_quant.zig");

/// KVSR_V=1 (paso 0 KV-Codec §4.6): SR en el encoder V. Leído directo del
/// env (kv_cache no depende del módulo debug — orden de creación en build).
fn kvSrV() bool {
    const v = std.c.getenv("KVSR_V");
    return v != null and std.mem.eql(u8, std.mem.span(v.?), "1");
}
const kvarn = @import("kvarn.zig");
const tail_request = @import("tail_request.zig");

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
    /// Slots asignados por capa y cabeza [layer][head] — body comprimido
    k_slots: [][]u32,
    v_slots: [][]u32,
    /// ── Lane-b2 P0.2 (KVCPT): cola exacta F16/BF16 ──
    /// Slabs f16 para el ring exacto: `[layer][head][exact_groups × head_dim]`.
    /// `null` si KVCPT desactivado. Cada append escribe el token en
    /// posición `[layer][head][tail_idx % exact_groups, :head_dim]`.
    k_exact: ?[][][]f16 = null,
    v_exact: ?[][][]f16 = null,
    /// Número de grupos exactos (= ceil(tail_tokens / KVCPT_GROUP)).
    exact_groups: u32 = 0,
    /// Tokens exactos efectivos (= exact_groups × KVCPT_GROUP).
    exact_tokens: u32 = 0,
    /// Tipo del slot exacto (resolved).
    tail_type: qt.ExactType = .default,
    /// Slots asignados por capa y cabeza [layer][head].
    /// `null` si KVCPT está desactivado (tail_tokens == 0).
    k_exact_slots: ?[][]u32 = null,
    v_exact_slots: ?[][]u32 = null,
    /// Formato usado por capa
    layer_formats: []LayerQuantConfig,
    /// 9.1 (lane-c C-1): store KVarN — records por capa/head/grupo cuando
    /// la config activa bits kvarn (LayerQuantConfig.kvarn_k_bits/v_bits).
    /// Estructura: k_records[layer][head][group] = record bytes del tile
    /// K de KVAR_N_GROUP tokens; v_records idem para V. `null` si KVarN
    /// desactivado. Los grupos NO completos (staging < 128 tokens) NO
    /// tienen record — el retrieve los sirve desde `k_exact`/fp16.
    k_records: ?[][][][]u8 = null,
    v_records: ?[][][][]u8 = null,
    /// 9.1 (lane-c C-1): staging de grupo — tokens f16 acumulados hasta
    /// llenar un tile de KVAR_N_GROUP antes del encodeKTile/encodeVTile
    /// (KVarN cuantiza el GRUPO entero, no tokens sueltos).
    /// [layer][head] = buffer [KVAR_N_GROUP * head_dim]f16 + fill count.
    k_stage: ?[][]KvarnStage = null,
    v_stage: ?[][]KvarnStage = null,
    /// Layout de record resuelto de la config de capa 0 (head_dim + bits
    /// de TODAS las capas KVarN — el manager exige una sola geometría).
    kvarn_k_layout: ?kvarn.KvarnRecordLayout = null,
    kvarn_v_layout: ?kvarn.KvarnRecordLayout = null,
};

/// 9.1 (lane-c C-1): staging de un grupo KVarN por (layer, head).
pub const KvarnStage = struct {
    buf: []f16,
    fill: u32,
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
    gpu_engine: ?*GpuDequantEngine,
    /// Buffer de prefetch (capa siguiente)
    prefetch_buffer: ?PrefetchBuffer,
    /// Métricas
    metrics: Metrics,
    /// 9.1 (lane-c C-1): scratch del tile kvarn [KVAR_N_GROUP × head_dim]f32
    /// — reusado por cada encodeKvarnGroup (encodeV/encodeK lo exigen f32).
    /// Lazy: aloca en el primer encode, libera en deinit.
    kvarn_tile_scratch: ?[]f32 = null,

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

    /// Inicializa el motor GPU (requiere kernels CUDA enlazados)
    pub fn initGpu(self: *Self, max_elements: usize) !void {
        if (!self.config.use_gpu_dequant) return;
        const engine = try self.allocator.create(GpuDequantEngine);
        engine.* = try GpuDequantEngine.init(self.allocator, max_elements);
        self.gpu_engine = engine;
    }

    pub fn deinit(self: *Self) void {
        if (self.gpu_engine) |engine| {
            engine.deinit();
        }
        var seq_iter = self.sequences.valueIterator();
        while (seq_iter.next()) |seq| {
            for (seq.k_slots) |layer| self.allocator.free(layer);
            for (seq.v_slots) |layer| self.allocator.free(layer);
            self.allocator.free(seq.k_slots);
            self.allocator.free(seq.v_slots);
            self.allocator.free(seq.layer_formats);
            // 9.1 (lane-c C-1): store kvarn.
            self.freeKvarnStore(seq);
            // Lane-b2 P0.2: libera slabs de cola exacta si existen.
            if (seq.k_exact_slots) |kes| {
                for (kes) |layer| self.allocator.free(layer);
                self.allocator.free(kes);
            }
            if (seq.v_exact_slots) |ves| {
                for (ves) |layer| self.allocator.free(layer);
                self.allocator.free(ves);
            }
            if (seq.k_exact) |ke| {
                for (ke) |layer| {
                    for (layer) |head| self.allocator.free(head);
                    self.allocator.free(layer);
                }
                self.allocator.free(ke);
            }
            if (seq.v_exact) |ve| {
                for (ve) |layer| {
                    for (layer) |head| self.allocator.free(head);
                    self.allocator.free(layer);
                }
                self.allocator.free(ve);
            }
        }
        self.sequences.deinit();
        self.pool.deinit();
        // 9.1 (lane-c C-1): scratch del tile kvarn.
        if (self.kvarn_tile_scratch) |t| self.allocator.free(t);
    }

    /// Registra una nueva secuencia
    pub fn createSequence(self: *Self, seq_id: u64) !void {
        if (self.sequences.contains(seq_id)) return error.SequenceExists;

        const num_layers = self.config.num_layers;
        const num_heads = self.config.num_heads;
        const head_dim = self.config.head_dim;

        var k_slots = try self.allocator.alloc([]u32, num_layers);
        errdefer self.allocator.free(k_slots);
        var v_slots = try self.allocator.alloc([]u32, num_layers);
        errdefer self.allocator.free(v_slots);

        // KVCPT (lane-b2 P0.2): calcular cola exacta.
        const kv_tail_enabled = self.config.tail_tokens > 0;
        const exact_groups: u32 = if (kv_tail_enabled)
            @as(u32, @intCast((self.config.tail_tokens + kvarn.KVAR_N_GROUP - 1) / kvarn.KVAR_N_GROUP))
        else
            0;
        const exact_tokens: u32 = exact_groups * kvarn.KVAR_N_GROUP;
        const tail_type: qt.ExactType = self.config.tail_type;

        var k_exact: ?[][][]f16 = null;
        var v_exact: ?[][][]f16 = null;
        var k_exact_slots: ?[][]u32 = null;
        var v_exact_slots: ?[][]u32 = null;
        if (kv_tail_enabled) {
            k_exact = try self.allocator.alloc([][]f16, num_layers);
            v_exact = try self.allocator.alloc([][]f16, num_layers);
            k_exact_slots = try self.allocator.alloc([]u32, num_layers);
            v_exact_slots = try self.allocator.alloc([]u32, num_layers);
        }

        var layer_formats = try self.allocator.alloc(LayerQuantConfig, num_layers);
        errdefer self.allocator.free(layer_formats);

        for (0..num_layers) |l| {
            k_slots[l] = try self.allocator.alloc(u32, num_heads);
            errdefer self.allocator.free(k_slots[l]);
            v_slots[l] = try self.allocator.alloc(u32, num_heads);
            errdefer self.allocator.free(v_slots[l]);

            @memset(k_slots[l], std.math.maxInt(u32));
            @memset(v_slots[l], std.math.maxInt(u32));

            if (k_exact) |ke| {
                ke[l] = try self.allocator.alloc([]f16, num_heads);
                errdefer self.allocator.free(ke[l]);
            }
            if (v_exact) |ve| {
                ve[l] = try self.allocator.alloc([]f16, num_heads);
                errdefer self.allocator.free(ve[l]);
            }
            if (k_exact_slots) |kes| {
                kes[l] = try self.allocator.alloc(u32, num_heads);
                errdefer self.allocator.free(kes[l]);
                @memset(kes[l], std.math.maxInt(u32));
            }
            if (v_exact_slots) |ves| {
                ves[l] = try self.allocator.alloc(u32, num_heads);
                errdefer self.allocator.free(ves[l]);
                @memset(ves[l], std.math.maxInt(u32));
            }
            if (k_exact) |ke| {
                for (0..num_heads) |h| {
                    ke[l][h] = try self.allocator.alloc(f16, exact_groups * head_dim);
                    @memset(ke[l][h], 0.0);
                }
            }
            if (v_exact) |ve| {
                for (0..num_heads) |h| {
                    ve[l][h] = try self.allocator.alloc(f16, exact_groups * head_dim);
                    @memset(ve[l][h], 0.0);
                }
            }

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

        // 9.1 (lane-c C-1): store KVarN — sólo si la config de capa activa
        // bits kvarn. Una única geometría por manager (head_dim de config;
        // bits de la PRIMERA capa kvarn — el CLI los aplica iguales a
        // todas). Records: [layer][head][group]; staging: [layer][head].
        const kvarn_active = blk: {
            if (self.config.layer_configs) |cfgs| {
                for (cfgs) |c| {
                    if (c.kvarn_k_bits > 0 or c.kvarn_v_bits > 0) break :blk true;
                }
            }
            break :blk false;
        };
        var k_records: ?[][][][]u8 = null;
        var v_records: ?[][][][]u8 = null;
        var k_stage: ?[][]KvarnStage = null;
        var v_stage: ?[][]KvarnStage = null;
        var k_layout: ?kvarn.KvarnRecordLayout = null;
        if (kvarn_active) {
            const k_bits: u8 = if (self.config.layer_configs) |cfgs| cfgs[0].kvarn_k_bits else 0;
            const v_bits: u8 = if (self.config.layer_configs) |cfgs| cfgs[0].kvarn_v_bits else 0;
            k_layout = try kvarn.KvarnRecordLayout.init(self.config.head_dim, if (k_bits > 0) k_bits else 4, if (v_bits > 0) v_bits else 4);
            const num_groups = (self.config.max_seq_len + kvarn.KVAR_N_GROUP - 1) / kvarn.KVAR_N_GROUP;
            k_records = try self.allocKvarnRecords(num_groups, k_bits > 0, k_layout.?);
            v_records = try self.allocKvarnRecords(num_groups, v_bits > 0, k_layout.?);
            if (k_bits > 0) k_stage = try self.allocKvarnStages();
            if (v_bits > 0) v_stage = try self.allocKvarnStages();
        }

        const state = SequenceState{
            .seq_id = seq_id,
            .current_len = 0,
            .k_slots = k_slots,
            .v_slots = v_slots,
            .k_exact = k_exact,
            .v_exact = v_exact,
            .exact_groups = exact_groups,
            .exact_tokens = exact_tokens,
            .tail_type = tail_type,
            .k_exact_slots = k_exact_slots,
            .v_exact_slots = v_exact_slots,
            .layer_formats = layer_formats,
            .k_records = k_records,
            .v_records = v_records,
            .k_stage = k_stage,
            .v_stage = v_stage,
            .kvarn_k_layout = k_layout,
            .kvarn_v_layout = k_layout,
        };

        try self.sequences.put(seq_id, state);
    }

    /// 9.1 (lane-c C-1): aloca el tensor 4D de records kvarn
    /// [layer][head][group][record_bytes]. `with_payload` = false deja
    /// records vacíos (len 0 — formato desactivado para ese lado).
    fn allocKvarnRecords(self: *Self, num_groups: usize, with_payload: bool, layout: kvarn.KvarnRecordLayout) ![][][][]u8 {
        const L = self.config.num_layers;
        const H = self.config.num_heads;
        const layers = try self.allocator.alloc([][][]u8, L);
        errdefer self.allocator.free(layers);
        for (layers) |*heads| {
            heads.* = try self.allocator.alloc([][]u8, H);
            @memset(heads.*, &.{});
        }
        errdefer for (layers) |*heads| self.allocator.free(heads.*);
        if (!with_payload) return layers;
        for (layers) |*heads| {
            for (heads.*) |*groups| {
                groups.* = try self.allocator.alloc([]u8, num_groups);
                for (groups.*) |*rec| {
                    rec.* = try self.allocator.alloc(u8, layout.tile_bytes);
                    @memset(rec.*, 0);
                }
            }
        }
        return layers;
    }

    /// 9.1 (lane-c C-1): aloca staging [layer][head] de un grupo f16.
    fn allocKvarnStages(self: *Self) ![][]KvarnStage {
        const L = self.config.num_layers;
        const H = self.config.num_heads;
        const layers = try self.allocator.alloc([]KvarnStage, L);
        errdefer self.allocator.free(layers);
        for (layers) |*heads| {
            heads.* = try self.allocator.alloc(KvarnStage, H);
            for (heads.*) |*st| {
                st.* = .{
                    .buf = try self.allocator.alloc(f16, kvarn.KVAR_N_GROUP * self.config.head_dim),
                    .fill = 0,
                };
                @memset(st.buf, 0);
            }
        }
        errdefer for (layers) |*heads| {
            for (heads.*) |*st| self.allocator.free(st.buf);
            self.allocator.free(heads.*);
        };
        return layers;
    }

    /// 9.1 (lane-c C-1): libera el store kvarn de una secuencia.
    fn freeKvarnStore(self: *Self, seq: *SequenceState) void {
        inline for (.{ seq.k_records, seq.v_records }) |maybe| {
            if (maybe) |records| {
                for (records) |heads| {
                    for (heads) |groups| {
                        for (groups) |rec| {
                            if (rec.len > 0) self.allocator.free(rec);
                        }
                        self.allocator.free(groups);
                    }
                    self.allocator.free(heads);
                }
                self.allocator.free(records);
            }
        }
        inline for (.{ seq.k_stage, seq.v_stage }) |maybe| {
            if (maybe) |stages| {
                for (stages) |heads| {
                    for (heads) |*st| self.allocator.free(st.buf);
                    self.allocator.free(heads);
                }
                self.allocator.free(stages);
            }
        }
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
                if (seq.k_exact_slots) |kes| {
                    const k_exact = kes[l][h];
                    if (k_exact != std.math.maxInt(u32)) self.pool.freeSlot(k_exact);
                }
                if (seq.v_exact_slots) |ves| {
                    const v_exact = ves[l][h];
                    if (v_exact != std.math.maxInt(u32)) self.pool.freeSlot(v_exact);
                }
            }
            self.allocator.free(seq.k_slots[l]);
            self.allocator.free(seq.v_slots[l]);
            if (seq.k_exact_slots) |kes| self.allocator.free(kes[l]);
            if (seq.v_exact_slots) |ves| self.allocator.free(ves[l]);
            if (seq.k_exact) |ke| {
                for (0..self.config.num_heads) |h| self.allocator.free(ke[l][h]);
                self.allocator.free(ke[l]);
            }
            if (seq.v_exact) |ve| {
                for (0..self.config.num_heads) |h| self.allocator.free(ve[l][h]);
                self.allocator.free(ve[l]);
            }
        }

        self.allocator.free(seq.k_slots);
        self.allocator.free(seq.v_slots);
        if (seq.k_exact_slots) |kes| self.allocator.free(kes);
        if (seq.v_exact_slots) |ves| self.allocator.free(ves);
        if (seq.k_exact) |ke| self.allocator.free(ke);
        if (seq.v_exact) |ve| self.allocator.free(ve);
        self.allocator.free(seq.layer_formats);
        // 9.1 (lane-c C-1): store kvarn.
        self.freeKvarnStore(seq);
        _ = self.sequences.remove(seq_id);
    }

    /// Almacena nuevos tokens K/V ya cuantizados (bytes canónicos por bloque)
    /// para una secuencia, en un run contiguo.
    pub fn appendTokens(
        self: *Self,
        seq_id: u64,
        layer_idx: u32,
        head_idx: u32,
        k_data: []const u8,
        v_data: []const u8,
    ) !void {
        const seq = self.sequences.getPtr(seq_id) orelse return error.SequenceNotFound;
        const lconf = &seq.layer_formats[layer_idx];

        const k_slot = try self.ensureSlot(seq_id, layer_idx, head_idx, true, lconf.k_format);
        const v_slot = try self.ensureSlot(seq_id, layer_idx, head_idx, false, lconf.v_format);

        const k_buf = self.pool.getBuffer(k_slot) orelse return error.SlotNotFound;
        const v_buf = self.pool.getBuffer(v_slot) orelse return error.SlotNotFound;

        const seq_len = seq.current_len;
        const head_dim = self.config.head_dim;

        const k_stride = kv_quant.quantBytes(lconf.k_format, @as(usize, head_dim));
        const v_stride = kv_quant.quantBytes(lconf.v_format, @as(usize, head_dim));

        const k_write_offset = seq_len * k_stride;
        const v_write_offset = seq_len * v_stride;

        if (k_write_offset + k_data.len > k_buf.len or v_write_offset + v_data.len > v_buf.len) {
            return error.BufferOverflow;
        }

        @memcpy(k_buf[k_write_offset .. k_write_offset + k_data.len], k_data);
        @memcpy(v_buf[v_write_offset .. v_write_offset + v_data.len], v_data);

        const fp16_bytes = (k_data.len + v_data.len) * 2;
        self.metrics.bytes_saved += fp16_bytes - (k_data.len + v_data.len);
    }

    /// Conveniencia: append de tokens en FP16 crudo.
    /// Cuantiza internamente según el formato de capa configurado.
    ///
    /// Si `config.tail_tokens > 0`, también escribe los f16 al ring exacto
    /// KVCPT (cola exacta) — el llamante usa `getExactTail` para recuperar.
    pub fn appendTokensF16(
        self: *Self,
        seq_id: u64,
        layer_idx: u32,
        head_idx: u32,
        k_data: []const f16,
        v_data: []const f16,
    ) !void {
        const seq = self.sequences.getPtr(seq_id) orelse return error.SequenceNotFound;
        const lconf = &seq.layer_formats[layer_idx];
        const k_fmt: QuantFormat = lconf.k_format;
        const v_fmt: QuantFormat = lconf.v_format;

        const k_bytes = try kv_quant.encodeToOwned(self.allocator, k_fmt, k_data);
        defer self.allocator.free(k_bytes);
        var v_bytes: []u8 = undefined;
        if (kvSrV() and v_fmt != .fp16) {
            v_bytes = try kv_quant.encodeToOwnedOpts(self.allocator, v_fmt, v_data, .{ .stochastic = true });
        } else {
            v_bytes = try kv_quant.encodeToOwned(self.allocator, v_fmt, v_data);
        }
        defer self.allocator.free(v_bytes);

        try self.appendTokens(seq_id, layer_idx, head_idx, k_bytes, v_bytes);

        // 9.1 (lane-c C-1): staging KVarN — acumula el token en el buffer
        // del grupo; al llenar KVAR_N_GROUP, encode del tile al record.
        // El body cuantizado (k_bytes arriba) SIGUE escribiéndose: el
        // retrieve prefiere records kvarn de grupos COMPLETOS y sirve el
        // resto desde el body — mismo dual-store que la cola exacta.
        if (seq.k_stage) |stages| {
            const st = &stages[layer_idx][head_idx];
            if (k_data.len == self.config.head_dim) {
                @memcpy(st.buf[st.fill * self.config.head_dim ..][0..self.config.head_dim], k_data);
                st.fill += 1;
                if (st.fill == kvarn.KVAR_N_GROUP) {
                    try self.encodeKvarnGroup(seq, layer_idx, head_idx, .k);
                    st.fill = 0;
                }
            }
        }
        if (seq.v_stage) |stages| {
            const st = &stages[layer_idx][head_idx];
            if (v_data.len == self.config.head_dim) {
                @memcpy(st.buf[st.fill * self.config.head_dim ..][0..self.config.head_dim], v_data);
                st.fill += 1;
                if (st.fill == kvarn.KVAR_N_GROUP) {
                    try self.encodeKvarnGroup(seq, layer_idx, head_idx, .v);
                    st.fill = 0;
                }
            }
        }

        // Dual-write a la cola exacta (KVCPT lane-b2 P0.2).
        // Solo si los slices K/V son del tamaño head_dim (1 token). Si el
        // caller pasa más tokens (varios a la vez), el contrato es
        // 1-token-per-call para el ring exacto (válido para decode).
        if (seq.k_exact) |ke| {
            if (k_data.len == self.config.head_dim) {
                const head_dim = self.config.head_dim;
                const tail_pos = seq.current_len;
                const group_idx = @as(u32, @intCast(tail_pos % seq.exact_groups));
                const dst_off = group_idx * head_dim;
                const dst = ke[layer_idx][head_idx][dst_off..][0..head_dim];
                @memcpy(dst, k_data);
            }
        }
        if (seq.v_exact) |ve| {
            if (v_data.len == self.config.head_dim) {
                const head_dim = self.config.head_dim;
                const group_idx = @as(u32, @intCast(seq.current_len % seq.exact_groups));
                const dst_off = group_idx * head_dim;
                const dst = ve[layer_idx][head_idx][dst_off..][0..head_dim];
                @memcpy(dst, v_data);
            }
        }
    }

    /// Avanza el contador de tokens de una secuencia en 1
    /// 9.1 (lane-c C-1): encode de un grupo KVarN completo (staging f16 →
    /// tile f32 → Hadamard por filas → record). Espejo del pipeline CPU de
    /// beellama (llama_kvarn_quantize_tile): K pre-rota filas, V se pasa
    /// tal cual (encodeVTile aplica su normalización). El record escrito
    /// es el del grupo lógico `current_len/KVAR_N_GROUP - 1`.
    fn encodeKvarnGroup(self: *Self, seq: *SequenceState, layer_idx: u32, head_idx: u32, side: enum { k, v }) !void {
        const layout = seq.kvarn_k_layout orelse return;
        const group = if (side == .k) seq.k_stage orelse return else seq.v_stage orelse return;
        const hd = self.config.head_dim;
        const n = kvarn.KVAR_N_GROUP * hd;
        if (self.kvarn_tile_scratch == null) {
            self.kvarn_tile_scratch = try self.allocator.alloc(f32, n);
        }
        const tile = self.kvarn_tile_scratch.?;

        const src = group[layer_idx][head_idx].buf;
        for (0..n) |i| tile[i] = @as(f32, @floatCast(src[i]));
        if (side == .k) kvarn.hadamard128Rows(tile, hd);

        const records = if (side == .k) seq.k_records orelse return else seq.v_records orelse return;
        const bits = if (side == .k) seq.layer_formats[layer_idx].kvarn_k_bits else seq.layer_formats[layer_idx].kvarn_v_bits;
        if (bits == 0) return;
        const group_idx = (seq.current_len + kvarn.KVAR_N_GROUP - 1) / kvarn.KVAR_N_GROUP - 1;
        const rec = records[layer_idx][head_idx][group_idx];

        // Reset del record (packBit ORs bits — necesita base limpia).
        @memset(rec, 0);
        if (side == .k) {
            try kvarn.encodeKTile(tile, 3, bits, layout, rec);
        } else {
            try kvarn.encodeVTile(tile, 3, bits, layout, rec);
        }
    }

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
    pub fn retrieveForAttention(
        self: *Self,
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

        // 9.1 (lane-c C-1): path KVarN — sirve los GRUPOS COMPLETOS desde
        // los records kvarn (decode tile → Hadamard inverso K) y el resto
        // (grupo parcial en curso) desde el body/cola exacta. El body
        // cuantizado clásico sigue escribiéndose por appendTokensF16, así
        // el camino no-kvarn queda como fallback exacto del grupo parcial.
        const kvarn_k = seq.k_records != null and lconf.kvarn_k_bits > 0;
        const kvarn_v = seq.v_records != null and lconf.kvarn_v_bits > 0;
        if (kvarn_k or kvarn_v) {
            const hd = self.config.head_dim;
            const full_groups = seq.current_len / kvarn.KVAR_N_GROUP;
            const partial_elems = (seq.current_len % kvarn.KVAR_N_GROUP) * hd;
            if (kvarn_k) {
                const records = seq.k_records.?;
                var tile: ?[]f32 = null;
                if (full_groups > 0) {
                    if (self.kvarn_tile_scratch == null) self.kvarn_tile_scratch = try self.allocator.alloc(f32, kvarn.KVAR_N_GROUP * hd);
                    tile = self.kvarn_tile_scratch.?;
                }
                for (0..full_groups) |g| {
                    try kvarn.decodeKTile(records[layer_idx][head_idx][g], lconf.kvarn_k_bits, seq.kvarn_k_layout.?, tile.?);
                    kvarn.hadamard128Rows(tile.?, hd); // K encode pre-rota → inverso al leer
                    for (0..kvarn.KVAR_N_GROUP) |t| {
                        for (0..hd) |c| out_k[(g * kvarn.KVAR_N_GROUP + t) * hd + c] = @floatCast(tile.?[t * hd + c]);
                    }
                }
                if (partial_elems > 0) {
                    // Grupo parcial: f16 exacto del staging (mismo dato que se
                    // cuantizará cuando cierre el grupo — cero pérdida de coherencia).
                    const st = seq.k_stage.?[layer_idx][head_idx];
                    for (0..partial_elems) |i| out_k[full_groups * kvarn.KVAR_N_GROUP * hd + i] = st.buf[i];
                }
            }
            if (kvarn_v) {
                const records = seq.v_records.?;
                var tile: ?[]f32 = null;
                if (full_groups > 0) {
                    if (self.kvarn_tile_scratch == null) self.kvarn_tile_scratch = try self.allocator.alloc(f32, kvarn.KVAR_N_GROUP * hd);
                    tile = self.kvarn_tile_scratch.?;
                }
                for (0..full_groups) |g| {
                    try kvarn.decodeVTile(records[layer_idx][head_idx][g], lconf.kvarn_v_bits, seq.kvarn_v_layout.?, tile.?);
                    for (0..kvarn.KVAR_N_GROUP) |t| {
                        for (0..hd) |c| out_v[(g * kvarn.KVAR_N_GROUP + t) * hd + c] = @floatCast(tile.?[t * hd + c]);
                    }
                }
                if (partial_elems > 0) {
                    const st = seq.v_stage.?[layer_idx][head_idx];
                    for (0..partial_elems) |i| out_v[full_groups * kvarn.KVAR_N_GROUP * hd + i] = st.buf[i];
                }
            }
            self.metrics.hits += 1;
            return;
        }

        // Si tenemos GPU, usar de-cuantización acelerada
        if (self.gpu_engine) |engine| {
            try self.dequantizeGpu(engine, k_buf, lconf.k_format, lconf.k_block_size, out_k[0..num_elements]);
            try self.dequantizeGpu(engine, v_buf, lconf.v_format, lconf.v_block_size, out_v[0..num_elements]);
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

    /// Rollback de N tokens en una secuencia (lane-b2 P0.2 KVCPT).
    ///
    /// Decrementa `current_len` en N tokens. NO reescribe el body
    /// comprimido (los tokens "rollbackeados" se sobrescribirán cuando
    /// el caller haga nuevos appends en esa posición). El ring exacto
    /// mantiene sus slots intactos — los últimos `exact_tokens` tokens
    /// desde el nuevo `current_len` siguen siendo accesibles vía
    /// `getExactTail`.
    ///
    /// Usado por el speculative decoding driver (B3) tras un rechazo
    /// del verifier. El contrato: rollback es seguro en cualquier
    /// secuencia cuyo `current_len >= N`.
    pub fn rollbackN(self: *Self, seq_id: u64, n: u32) !void {
        if (n == 0) return;
        const seq = self.sequences.getPtr(seq_id) orelse return error.SequenceNotFound;
        if (n > seq.current_len) return error.RollbackBeyondStart;
        seq.current_len -= n;
    }

    /// Lee el ring exacto KVCPT: devuelve el bloque f16 (o bf16) que
    /// contiene los últimos `tail_tokens` tokens de la secuencia.
    ///
    /// `out` debe tener tamaño >= `seq.exact_groups * head_dim`. Para
    /// `tail_tokens = 0` o KVCPT desactivado, devuelve `error.NoExactTail`.
    ///
    /// Layout del ring (decisión C1 lane-b2): FIFO con slots `exact_groups
    /// × head_dim`. El slot del token `current_len - 1 - i` (0 = más
    /// reciente) está en `slot_idx = (current_len - 1 - i) % exact_groups`.
    pub fn getExactTail(
        self: *Self,
        seq_id: u64,
        layer_idx: u32,
        head_idx: u32,
        k_out: []f16,
        v_out: []f16,
    ) !void {
        const seq = self.sequences.getPtr(seq_id) orelse return error.SequenceNotFound;
        const ke = seq.k_exact orelse return error.NoExactTail;
        const ve = seq.v_exact orelse return error.NoExactTail;
        const exact_groups = seq.exact_groups;
        const head_dim = self.config.head_dim;
        const need = exact_groups * head_dim;
        if (k_out.len < need or v_out.len < need) return error.BufferTooSmall;

        const k_buf = ke[layer_idx][head_idx];
        const v_buf = ve[layer_idx][head_idx];
        @memcpy(k_out[0..need], k_buf[0..need]);
        @memcpy(v_out[0..need], v_buf[0..need]);
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
                layer_idx,
                head_idx,
                0,
                max_len,
                head_dim,
                format,
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
            .q8_0, .q4_0, .q4_1, .q2_0s, .q2_1, .q3_0, .q3_1, .q6_0, .q6_1 => kv_quant.decode(format, raw, out),
            .int8_asymmetric, .int4 => return error.UnsupportedCpuDequant,
            // lane-b2 P0.3: los 6 tipos nuevos caen a UnsupportedCpuDequant
            // hasta que el path CPU-GPU los cablee (roundtrips en kv_quant ✓).
            .q8_1, .q5_0, .q5_1, .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_k, .iq1_s, .iq1_m, .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s, .iq4_xs, .iq4_nl, .tq1_0, .tq2_0, .mxfp4, .fp8 => return error.UnsupportedCpuDequant,
        }
    }

    /// De-cuantización usando el motor GPU (Runtime API)
    fn dequantizeGpu(
        self: *Self,
        engine: *gpu_dequant.GpuDequantEngine,
        raw: []const u8,
        format: QuantFormat,
        block_size: usize,
        out: []f16,
    ) !void {
        // Allocate device memory for raw data
        const crt = @import("cuda_runtime");
        const d_raw_buf = try crt.GpuBuffer.alloc(raw.len);
        errdefer d_raw_buf.free();
        try d_raw_buf.upload(raw);
        const d_raw = d_raw_buf.ptr.?;

        // For formats with scales embedded in raw data (q8_0, q4_0, q4_1, fp16)
        // Use GPU dequant kernel
        if (format == .q8_0 or format == .q4_0 or format == .q4_1 or format == .fp16 or format == .fp8) {
            // These formats have scales embedded in the raw data
            // Use GPU dequant kernel
            _ = engine.dequantize(format, d_raw, null, null, out.len, block_size) catch {
                // Fallback to CPU
                try self.dequantizeCpu(raw, format, out);
                return;
            };

            // Copy result back to host
            const f32_out = try self.allocator.alloc(f32, out.len);
            defer self.allocator.free(f32_out);
            const out_buf = crt.GpuBuffer{ .ptr = @ptrCast(f32_out.ptr), .len = out.len * @sizeOf(f32) };
            try out_buf.download(f32_out);
            for (f32_out, 0..) |f, i| out[i] = @as(f16, @floatCast(f));
        } else {
            // Fallback to CPU for other formats
            try self.dequantizeCpu(raw, format, out);
        }
    }
};
