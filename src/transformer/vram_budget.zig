//! VramBudget — presupuesto dinámico de memoria GPU.
//!
//! Categoria: weights | activations | kv_cache
//! Cada categoria tiene un presupuesto; canAlloc() verifica antes de allocar.
//! LayerStreamer y ActivationPool consultan este presupuesto antes de cargar
//! pesos o allocar buffers, forzando LRU eviction si se excede.
const std = @import("std");
const debug = @import("debug");
// lane-f P3: presupuesto elástico (Phase 3) como fuente del techo del rebuild.
const budget = @import("budget");

pub const Category = enum { weights, activations, kv_cache };

pub const KVQuantFormat = enum {
    fp16,
    q8_0,
    q4_0,
    q4_1,
    q5_0,
    q5_1,
    q8_1,
    q2_k,
    q3_k,
    q4_k,
    q5_k,
    q6_k,
    q8_k,
    iq1_s,
    iq1_m,
    iq2_xxs,
    iq2_xs,
    iq2_s,
    iq3_xxs,
    iq3_s,
    iq4_xs,
    iq4_nl,
    tq1_0,
    tq2_0,
    mxfp4,

    /// (block_size elems, bytes/block) — layout GGUF canónico, escalas embebidas.
    pub fn spec(self: KVQuantFormat) struct { bs: usize, bb: usize } {
        return switch (self) {
            .fp16 => .{ .bs = 1, .bb = 2 },
            .q4_0 => .{ .bs = 32, .bb = 18 },
            .q4_1 => .{ .bs = 32, .bb = 20 },
            .q5_0 => .{ .bs = 32, .bb = 22 },
            .q5_1 => .{ .bs = 32, .bb = 24 },
            .q8_0 => .{ .bs = 32, .bb = 34 },
            .q8_1 => .{ .bs = 32, .bb = 36 },
            .mxfp4 => .{ .bs = 32, .bb = 17 },
            .iq4_nl => .{ .bs = 32, .bb = 18 },
            .q2_k => .{ .bs = 256, .bb = 84 },
            .q3_k => .{ .bs = 256, .bb = 110 },
            .q4_k => .{ .bs = 256, .bb = 144 },
            .q5_k => .{ .bs = 256, .bb = 176 },
            .q6_k => .{ .bs = 256, .bb = 210 },
            .q8_k => .{ .bs = 256, .bb = 292 },
            .iq1_s => .{ .bs = 256, .bb = 50 },
            .iq1_m => .{ .bs = 256, .bb = 56 },
            .iq2_xxs => .{ .bs = 256, .bb = 66 },
            .iq2_xs => .{ .bs = 256, .bb = 74 },
            .iq2_s => .{ .bs = 256, .bb = 82 },
            .iq3_xxs => .{ .bs = 256, .bb = 98 },
            .iq3_s => .{ .bs = 256, .bb = 110 },
            .iq4_xs => .{ .bs = 256, .bb = 136 },
            .tq1_0 => .{ .bs = 256, .bb = 54 },
            .tq2_0 => .{ .bs = 256, .bb = 66 },
        };
    }
};

pub const VramBudgetConfig = struct {
    total_vram: usize,
    num_layers: usize,
    num_attn_layers: usize,
    hidden_dim: usize,
    head_dim: usize,
    num_kv_heads: usize,
    max_seq_len: usize,
    kv_quant: KVQuantFormat,
    feed_forward_dim: usize,
};

pub const VramBudget = struct {
    total_vram: usize,
    weights_budget: usize,
    activations_budget: usize,
    kv_budget: usize,
    safety_margin: usize,
    weights_used: std.atomic.Value(usize),
    activations_used: std.atomic.Value(usize),
    kv_used: std.atomic.Value(usize),

    const Self = @This();

    pub fn init(config: VramBudgetConfig) Self {
        // Layout: 60% weights, 20% activations, 20% KV-cache, 5% safety
        const safe_total = config.total_vram * 95 / 100;
        return .{
            .total_vram = config.total_vram,
            .weights_budget = safe_total * 60 / 100,
            .activations_budget = safe_total * 20 / 100,
            .kv_budget = safe_total * 20 / 100,
            .safety_margin = config.total_vram * 5 / 100,
            .weights_used = std.atomic.Value(usize).init(0),
            .activations_used = std.atomic.Value(usize).init(0),
            .kv_used = std.atomic.Value(usize).init(0),
        };
    }

    pub fn canAlloc(self: *Self, cat: Category, bytes: usize) bool {
        const cat_budget = switch (cat) {
            .weights => self.weights_budget,
            .activations => self.activations_budget,
            .kv_cache => self.kv_budget,
        };
        const used = switch (cat) {
            .weights => self.weights_used.load(.acquire),
            .activations => self.activations_used.load(.acquire),
            .kv_cache => self.kv_used.load(.acquire),
        };
        return used + bytes <= cat_budget;
    }

    pub fn reserve(self: *Self, cat: Category, bytes: usize) !void {
        if (!self.canAlloc(cat, bytes)) {
            debug.dbg.printLevel(.info, "[vram_budget] reserve {s} {d} bytes EXCEEDS budget, triggering eviction\n", .{ @tagName(cat), bytes });
            try self.maybeEvict(cat, bytes);
        }
        switch (cat) {
            .weights => _ = self.weights_used.fetchAdd(bytes, .acq_rel),
            .activations => _ = self.activations_used.fetchAdd(bytes, .acq_rel),
            .kv_cache => _ = self.kv_used.fetchAdd(bytes, .acq_rel),
        }
    }

    pub fn release(self: *Self, cat: Category, bytes: usize) void {
        switch (cat) {
            .weights => _ = self.weights_used.fetchSub(bytes, .acq_rel),
            .activations => _ = self.activations_used.fetchSub(bytes, .acq_rel),
            .kv_cache => _ = self.kv_used.fetchSub(bytes, .acq_rel),
        }
    }

    /// Evict entries from a category to make room. Caller must also
    /// free the actual buffers after this returns.
    pub fn maybeEvict(self: *Self, cat: Category, need_bytes: usize) !void {
        const used = switch (cat) {
            .weights => self.weights_used.load(.acquire),
            .activations => self.activations_used.load(.acquire),
            .kv_cache => self.kv_used.load(.acquire),
        };
        if (used + need_bytes <= self.budgetFor(cat)) return;

        // Signal eviction needed — caller (LayerStreamer/ActivationPool)
        // should respond by calling unloadLayer / releasing buffers.
        debug.dbg.printLevel(.info, "[vram_budget] {s} needs {d} bytes (used={d}, budget={d}) — eviction required\n", .{ @tagName(cat), need_bytes, used, self.budgetFor(cat) });
    }

    fn budgetFor(self: *Self, cat: Category) usize {
        return switch (cat) {
            .weights => self.weights_budget,
            .activations => self.activations_budget,
            .kv_cache => self.kv_budget,
        };
    }

    pub fn reportMetrics(self: *Self) void {
        if (!debug.dbg.at(.info)) return;
        const w = self.weights_used.load(.acquire);
        const a = self.activations_used.load(.acquire);
        const k = self.kv_used.load(.acquire);
        const total = w + a + k;
        debug.dbg.printLevel(.info, "[vram_budget] total={d}MB weights={d}/{d}MB activations={d}/{d}MB kv={d}/{d}MB ({{total={d:.1}%}})\n", .{
            self.total_vram / (1024 * 1024),
            w / (1024 * 1024),
            self.weights_budget / (1024 * 1024),
            a / (1024 * 1024),
            self.activations_budget / (1024 * 1024),
            k / (1024 * 1024),
            self.kv_budget / (1024 * 1024),
            @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(self.total_vram)) * 100.0,
        });
    }
};

/// Estimation parameters for VRAM calculation
pub const VramEstimate = struct {
    /// Compressed model weights on disk/mmap (bytes per layer, avg)
    compressed_weight_per_layer: usize,
    /// Number of layers in model
    num_layers: usize,
    /// Number of attention layers
    num_attn_layers: usize,
    /// Hidden dimension
    hidden_dim: usize,
    /// Head dimension
    head_dim: usize,
    /// Number of KV heads
    num_kv_heads: usize,
    /// Feed-forward dimension
    feed_forward_dim: usize,
    /// Max sequence length (context)
    max_seq_len: usize,
    /// KV cache quantization format
    kv_quant: KVQuantFormat,
    /// Block size for PagedAttention
    block_size: usize = 16,
    /// Max resident layers (for layer streaming)
    max_resident: usize = 2,
    /// CUDA context overhead (bytes)
    cuda_overhead: usize = 150 * 1024 * 1024,
    // --- Lane F5: términos que el presupuesto estático NO modelaba ---
    /// Residentes fijos no-por-capa: embedding/lm_head/q80-device siempre
    /// en VRAM (ticket C: q80 27B ≈1.29GB comía el margen invisible).
    resident_fixed_bytes: usize = 0,
    /// Pico TRANSITORIO por forward (p.ej. W_T f32 d_inner×n_embd de
    /// ssm linearProjectionDevice ≈250MB en 27B): muere el PRIMER forward
    /// post-prefill aunque el prefill pase.
    transient_peak_bytes: usize = 0,
    /// Cap del pool CONTIGUO de KV cuando VMM no aplica (anti-OOM host
    /// de lane-c lo fija en 512MB); el excedente KV debe ir offload.
    contiguous_pool_cap: usize = 0,
};

/// Detailed VRAM breakdown
pub const VramBreakdown = struct {
    weights_vram: usize,
    kv_vram: usize,
    activations_vram: usize,
    cuda_overhead: usize,
    total_vram: usize,
    weights_host: usize,
    kv_host: usize,
    // --- Lane F5 ---
    /// Residentes fijos (emb/lm_head/q80-device).
    resident_fixed: usize = 0,
    /// Pico transitorio por forward (W_T etc.).
    transient_peak: usize = 0,
    /// true si el pool contiguo capó el KV efectivo (excedente → offload).
    pool_capped: bool = false,

    pub fn totalHostRam(self: VramBreakdown) usize {
        return self.weights_host + self.kv_host;
    }
};

/// Calculate KV cache block bytes (K + V regions) for the given format.
/// GGUF K/I-quant layouts embed their scales inside the block bytes, so the
/// total is simply ceil(elems/bs) * bb per region.
fn kvBlockBytes(config: VramEstimate) usize {
    const s = config.kv_quant.spec();
    const elems = config.block_size * config.num_kv_heads * config.head_dim;
    const per_region = (elems + s.bs - 1) / s.bs * s.bb;
    return per_region * 2; // K + V
}

/// Calculate activations VRAM per resident layer (decode, N=1)
fn activationBytesPerLayer(config: VramEstimate) usize {
    // Per layer decode buffers (f32):
    // norm_buf[1,hidden], mixer_out[1,hidden], gate_buf[1,ffn], up_buf[1,ffn], ffn_out[1,hidden], post_norm[1,hidden]
    return 4 * config.hidden_dim + 2 * config.feed_forward_dim + 2 * config.hidden_dim;
}

/// Estimate total VRAM requirement for a model configuration
pub fn estimateTotalVram(config: VramEstimate) VramBreakdown {
    const block_bytes = kvBlockBytes(config);
    const num_blocks = ((config.max_seq_len + config.block_size - 1) / config.block_size) * config.num_attn_layers;

    // Weights on GPU: max_resident layers * f32 scratch (8x compressed)
    const f32_per_layer = config.compressed_weight_per_layer * 8;
    const weights_vram = config.max_resident * f32_per_layer;

    // Weights on host: all layers compressed
    const weights_host = config.num_layers * config.compressed_weight_per_layer;

    // KV cache on GPU (device pool) — con cap contiguo si VMM no aplica.
    const kv_raw = num_blocks * block_bytes;
    var pool_capped = false;
    var kv_vram = kv_raw;
    if (config.contiguous_pool_cap > 0 and kv_raw > config.contiguous_pool_cap) {
        kv_vram = config.contiguous_pool_cap;
        pool_capped = true;
    }

    // KV cache on host (always allocated)
    const kv_host = num_blocks * block_bytes;

    // Activations on GPU: max_resident layers
    const activation_per_layer = activationBytesPerLayer(config);
    const activations_vram = config.max_resident * activation_per_layer;

    const total_vram = weights_vram + kv_vram + activations_vram +
        config.resident_fixed_bytes + config.transient_peak_bytes + config.cuda_overhead;

    return .{
        .weights_vram = weights_vram,
        .kv_vram = kv_vram,
        .activations_vram = activations_vram,
        .cuda_overhead = config.cuda_overhead,
        .total_vram = total_vram,
        .weights_host = weights_host,
        .kv_host = kv_host,
        .resident_fixed = config.resident_fixed_bytes,
        .transient_peak = config.transient_peak_bytes,
        .pool_capped = pool_capped,
    };
}

/// Bytes del transitorio W_T típico de SSM (linearProjectionDevice):
/// proyección f32 [d_inner × n_embd] materializada por forward.
pub fn ssmTransientBytes(d_inner: usize, n_embd: usize) usize {
    return d_inner * n_embd * 4;
}

/// Check if model fits in VRAM with given config, returns suggested max_resident if not
pub fn suggestLayerStreamConfig(
    total_vram: usize,
    config: VramEstimate,
) ?usize {
    // Try with max_resident = 2 first
    var test_config = config;
    test_config.max_resident = 2;
    var breakdown = estimateTotalVram(test_config);

    if (breakdown.total_vram <= total_vram * 85 / 100) {
        return 2;
    }

    // Try with max_resident = 1
    test_config.max_resident = 1;
    breakdown = estimateTotalVram(test_config);

    if (breakdown.total_vram <= total_vram * 85 / 100) {
        return 1;
    }

    // Even 1 layer doesn't fit - need CPU offload
    return null;
}

// ============================================================================
// Lane F5 — Rebuild elástico slots-MoE ↔ páginas KV sin reinicio.
//
// Protocolo FreeToken (engine.py:765-909) como MÁQUINA DE ESTADOS de
// aritmética pura (testable sin GPU):
//   0a. prevalidate()  — geometría inválida → rechazo RECOVERABLE, estado intacto
//   0b. fitCheck()     — cuenta de memoria contra el account; no cabe → rechazo limpio
//   1.  pointOfNoReturn() — snapshot de la geometría actual; desde aquí un fallo
//       destructivo exige rollback (reconstrucción a la geometría snapshot-eada)
//   2.  applyResize(.ok|.oom por pool) — .oom → rollback() restaura snapshot y
//       devuelve error (el llamante dueño de los pools re-ejecuta su teardown real)
//   3.  complete()
// El wiring físico (teardown de graphs, resize in-place, re-captura) lo hacen
// los dueños de cada pool vía tickets; aquí vive la DISCIPLINA y la aritmética.
// ============================================================================

pub const RebuildError = error{
    /// Geometría inválida o mínimos no alcanzados (rechazo recoverable).
    RejectedInvalidGeometry,
    /// El objetivo no cabe en la cuenta de memoria (rechazo recoverable).
    RejectedDoesNotFit,
    /// OOM durante resize destructivo: snapshot restaurado.
    OutOfMemoryDuringResize,
};

/// Reparto del presupuesto entre familias de pools (en bytes).
pub const BudgetSplit = struct {
    total_vram: usize,
    fixed_bytes: usize, // pesos residentes + overhead CUDA + emb/lm_head
    activation_bytes: usize,
    transient_reserve: usize, // pico W_T reservado SIEMPRE como headroom
    kv_bytes: usize,
    moe_slots_bytes: usize,

    pub fn committed(self: BudgetSplit) usize {
        return self.fixed_bytes + self.activation_bytes + self.kv_bytes + self.moe_slots_bytes;
    }
    /// Headroom libre = total − comprometido − reserva transitoria.
    pub fn freeHeadroom(self: BudgetSplit) usize {
        const used = self.committed() + self.transient_reserve;
        return if (self.total_vram > used) self.total_vram - used else 0;
    }
};

/// Precios por unidad de cada familia (el llamante los deriva del modelo).
pub const PoolPricing = struct {
    kv_page_bytes: usize,
    moe_slot_bytes: usize,
    min_kv_pages: usize = 1,
    min_moe_slots: usize = 0, // 0 = modelo sin MoE ⇒ slots siempre rechazados
};

pub const RebuildRequest = struct {
    kv_pages: ?usize = null,
    moe_slots: ?usize = null,
};

pub const ResizeOutcome = enum { ok, oom };

pub const RebuildPhase = enum { idle, prevalidated, fit_checked, torn_down, resized, rolled_back, done };

pub const RebuildTx = struct {
    phase: RebuildPhase = .idle,
    current: BudgetSplit,
    snapshot: ?BudgetSplit = null,
    pricing: PoolPricing,
    account_free: usize, // VRAM libre MEDIDA al iniciar (baseline)

    const Self = @This();

    fn priceSplit(pricing: PoolPricing, req: RebuildRequest, base: BudgetSplit) BudgetSplit {
        var out = base;
        if (req.kv_pages) |p| out.kv_bytes = p * pricing.kv_page_bytes;
        if (req.moe_slots) |s| out.moe_slots_bytes = s * pricing.moe_slot_bytes;
        return out;
    }

    /// 0a. Validación geométrica ANTES de liberar nada. Falla → estado intacto.
    pub fn prevalidate(self: *Self, req: RebuildRequest) !void {
        std.debug.assert(self.phase == .idle);
        if (req.kv_pages == null and req.moe_slots == null) return error.RejectedInvalidGeometry;
        if (req.kv_pages) |p| {
            if (p < self.pricing.min_kv_pages) return error.RejectedInvalidGeometry;
            if (self.pricing.kv_page_bytes == 0) return error.RejectedInvalidGeometry;
        }
        if (req.moe_slots) |s| {
            if (self.pricing.min_moe_slots == 0 and s > 0) return error.RejectedInvalidGeometry; // sin MoE
            if (self.pricing.moe_slot_bytes == 0) return error.RejectedInvalidGeometry;
            if (s < self.pricing.min_moe_slots or s == 0) return error.RejectedInvalidGeometry;
        }
        self.phase = .prevalidated;
    }

    /// 0b. Fit-check aritmético contra presupuesto neto. Falla → rechazo limpio.
    pub fn fitCheck(self: *Self, req: RebuildRequest) !BudgetSplit {
        std.debug.assert(self.phase == .prevalidated);
        const target = priceSplit(self.pricing, req, self.current);
        // Presupuesto neto: todo_vram − fijo − activaciones − reserva transitoria.
        const ceiling = self.current.total_vram -|
            (self.current.fixed_bytes + self.current.activation_bytes + self.current.transient_reserve);
        if (target.kv_bytes + target.moe_slots_bytes > ceiling) return error.RejectedDoesNotFit;
        self.phase = .fit_checked;
        return target;
    }

    /// 0b'. Variante elástica (Phase 3): además del techo aritmético,
    /// consulta el presupuesto elástico (`budget.Rebuilder`) con los
    /// consumers del caller (p.ej. OffloadCache.asConsumer) y reparte el
    /// headroom libre proporcionalmente entre ellos. Los grants resultantes
    /// (consumer.granted) dictan cuántos slots/páginas puede tocar cada
    /// pool en su teardown físico; el Split devuelto sigue siendo la
    /// geometría OBJETIVO. Rechazo limpio en overcommit (estado intacto:
    /// la fase queda en .prevalidated para reintentar otra geometría).
    pub fn fitCheckElastic(
        self: *Self,
        req: RebuildRequest,
        consumers: []budget.Consumer,
    ) !BudgetSplit {
        // Validación elástica PRIMERO (sin tocar la fase): snapshot derivado
        // del split actual — el pool libre medido es el baseline; los bytes
        // fijos+activaciones son los pesos del presupuesto y la reserva
        // transitoria el fixed_cache.
        const snap = budget.BudgetSnapshot{
            .baseline_free = self.account_free,
            .weights_bytes = self.current.fixed_bytes + self.current.activation_bytes,
            .fixed_cache = self.current.transient_reserve,
            .memory_ratio = 1.0,
        };
        if (!budget.Rebuilder.preValidate(snap)) return error.RejectedDoesNotFit;
        var rb = budget.Rebuilder.init(snap, consumers);
        // fitCheck del Rebuilder restaura grants a current y planifica sin
        // mutar nada del RebuildTx: overcommit → null ⇒ rechazo limpio.
        if (rb.fitCheck(snap) == null) return error.RejectedDoesNotFit;

        // Elástico OK: ahora sí el fit-check aritmético (avanza la fase).
        return self.fitCheck(req);
    }

    /// Punto de no retorno: snapshot para rollback. Desde aquí un fallo
    /// destructivo deja el servicio SOLO recuperable vía rebuild.
    pub fn pointOfNoReturn(self: *Self) void {
        std.debug.assert(self.phase == .fit_checked);
        self.snapshot = self.current;
        self.phase = .torn_down;
    }

    /// Aplica el resize destino; outcome .oom en CUALQUIER pool → rollback
    /// a la geometría snapshot-eada + error (aritmética; el teardown físico
    /// lo repite el dueño del pool con sus buffers reales).
    pub fn applyResize(self: *Self, target: BudgetSplit, outcomes: []const ResizeOutcome) !void {
        std.debug.assert(self.phase == .torn_down);
        for (outcomes) |oc| {
            if (oc == .oom) {
                try self.rollback();
                return error.OutOfMemoryDuringResize;
            }
        }
        self.current = target;
        self.phase = .resized;
    }

    /// Rollback destructivo: restaurar geometría snapshot-eada.
    pub fn rollback(self: *Self) !void {
        const snap = self.snapshot orelse return error.RejectedInvalidGeometry;
        self.current = snap;
        self.snapshot = null;
        self.phase = .rolled_back;
    }

    pub fn complete(self: *Self) void {
        std.debug.assert(self.phase == .resized);
        self.phase = .done;
    }
};
