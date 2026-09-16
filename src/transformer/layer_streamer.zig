//! LayerStreamer — carga asíncrona y prefetch de capas para inferencia AirLLM-style.
//! Mientras la GPU computa la capa i, el thread pool descuenta/prefija la capa i+1
//! a RAM/GPU. LRU eviction mantiene max_resident capas cargadas.
//! Ahora con soporte para GPU weight pool (async H2D/D2H) y VRAM budget enforcement.
const std = @import("std");
const hybrid_layer = @import("hybrid_layer");
const HybridLayer = hybrid_layer.HybridLayer;
const gguf = @import("gguf");
const model_config = @import("model_config");
const debug = @import("debug");
const vram_budget = @import("vram_budget");
const gpu_weight_pool = @import("gpu_weight_pool");
const cudaz = @import("cudaz");

const LayerState = enum(u8) {
    unloaded = 0,
    loading = 1,
    loaded = 2,
};

/// Estado de residencia GPU por capa
const GpuResidentState = enum(u8) {
    not_resident = 0,
    uploading = 1,
    resident = 2,
};

pub const LayerStreamer = struct {
    allocator: std.mem.Allocator,
    g: *const gguf.GgufFile,
    layers: []HybridLayer,
    cfg: model_config.ModelConfig,
    states: []std.atomic.Value(LayerState),
    gpu_states: []std.atomic.Value(GpuResidentState),
    last_used: []std.atomic.Value(u64),
    max_resident: usize,
    resident_count: std.atomic.Value(usize),
    mutex: std.atomic.Mutex,
    spawned_threads: []?std.Thread,
    tick: std.atomic.Value(u64),
    debug_enabled: bool,
    vram_budget: ?*vram_budget.VramBudget = null,
    gpu_weight_pool: ?*gpu_weight_pool.GpuWeightPool = null,
    stream: ?cudaz.CUstream = null,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        layers: []HybridLayer,
        g: *const gguf.GgufFile,
        cfg: model_config.ModelConfig,
        max_resident: usize,
        num_workers: usize,
        vram_budget_ptr: ?*vram_budget.VramBudget,
        gpu_weight_pool_ptr: ?*gpu_weight_pool.GpuWeightPool,
        stream: ?cudaz.CUstream,
    ) !Self {
        _ = num_workers;
        const num_layers = cfg.block_count;
        if (layers.len != num_layers) {
            debug.dbg.printLevel(.info, "[layer_streamer] MISMATCH layers.len={d} block_count={d}\n", .{ layers.len, num_layers });
        }

        const states = try allocator.alloc(std.atomic.Value(LayerState), num_layers);
        errdefer allocator.free(states);
        for (states) |*s| s.* = std.atomic.Value(LayerState).init(.unloaded);

        const gpu_states = try allocator.alloc(std.atomic.Value(GpuResidentState), num_layers);
        errdefer allocator.free(gpu_states);
        for (gpu_states) |*s| s.* = std.atomic.Value(GpuResidentState).init(.not_resident);

        const last_used = try allocator.alloc(std.atomic.Value(u64), num_layers);
        errdefer allocator.free(last_used);
        for (last_used) |*t| t.* = std.atomic.Value(u64).init(0);

        const spawned = try allocator.alloc(?std.Thread, num_layers);
        errdefer allocator.free(spawned);
        @memset(spawned, null);

        return .{
            .allocator = allocator,
            .g = g,
            .layers = layers,
            .cfg = cfg,
            .states = states,
            .gpu_states = gpu_states,
            .last_used = last_used,
            .max_resident = max_resident,
            .resident_count = std.atomic.Value(usize).init(0),
            .mutex = .unlocked,
            .spawned_threads = spawned,
            .tick = std.atomic.Value(u64).init(0),
            .debug_enabled = false,
            .vram_budget = vram_budget_ptr,
            .gpu_weight_pool = gpu_weight_pool_ptr,
            .stream = stream,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.spawned_threads) |t| {
            if (t) |thread| thread.join();
        }
        self.allocator.free(self.states);
        self.allocator.free(self.gpu_states);
        self.allocator.free(self.last_used);
        self.allocator.free(self.spawned_threads);
    }

    pub fn enableDebug(self: *Self) void {
        self.debug_enabled = true;
    }

    /// 4.12: resize elástico de capas residentes (Gap 4, fase C2).
    ///
    /// Hot-resize SIN restart: shrink ⇒ eviction LRU inmediata (las capas
    /// menos recientes se descargan de RAM/GPU); grow ⇒ sólo sube el cap
    /// (las cargas futuras lo aprovechan; no fuerza prefetch). Al ser
    /// puramente host-side (contadores + unloadWeights), es seguro en
    /// cualquier punto del decode — PERO si hay un CUDA graph capturado
    /// que referencie pesos de capa, el caller debe invalidarlo (los
    /// unloadWeights liberan device memory). Protocolo FreeToken
    /// engine.py:766-909 fase 2 "resize caches in-place".
    pub fn setMaxResidentLayers(self: *Self, n: usize) void {
        const old = self.max_resident;
        self.max_resident = n;
        if (n < old) {
            // Shrink: evict hasta cumplir el nuevo cap (síncrono).
            self.maybeEvict();
            self.maybeEvictGpu() catch {};
        }
        if (self.debug_enabled or n < old) {
            debug.dbg.printLevel(.info, "[layer_streamer] resize max_resident {d} → {d} (resident={d})\n", .{ old, n, self.residentCount() });
        }
    }

    /// Alias semántico del plan GPUCPU (fase C2: `LayerStreamer.resize`).
    pub fn resize(self: *Self, max_resident: usize) void {
        self.setMaxResidentLayers(max_resident);
    }

    fn lock(self: *Self) void {
        while (!self.mutex.tryLock()) {
            std.Thread.yield() catch {};
        }
    }

    fn unlock(self: *Self) void {
        self.mutex.unlock();
    }

    pub fn prefetchLayer(self: *Self, layer_idx: usize) !void {
        self.lock();
        defer self.unlock();

        const state = self.states[layer_idx].load(.acquire);
        if (state != .unloaded) return;

        self.states[layer_idx].store(.loading, .release);
        const thread = try std.Thread.spawn(.{}, runLoad, .{ self, layer_idx });
        self.spawned_threads[layer_idx] = thread;
    }

    pub fn ensureLayerLoaded(self: *Self, layer_idx: usize) !void {
        self.lock();
        var state = self.states[layer_idx].load(.acquire);
        while (state == .loading) {
            self.unlock();
            std.Thread.yield() catch {};
            self.lock();
            state = self.states[layer_idx].load(.acquire);
        }
        self.unlock();

        if (state == .unloaded) {
            try self.loadLayerSync(layer_idx);
        }
        // El tick ANTES del evict: la capa recién asegurada debe ser la más
        // reciente para el LRU — si no, maybeEvict la expulsa a ella misma
        // y el warmup/forward posterior operaría sobre scratch liberado.
        self.last_used[layer_idx].store(self.tick.fetchAdd(1, .acq_rel), .release);

        // 7.1: este es el camino del warmup global (main.zig) y del forward
        // bajo grafo — ninguno pasa por prefetchNext, así que el resident
        // crecía a max_resident+1 (in-flight) y NUNCA se expulsaba aquí.
        // Con dtypes sin kernel qgemm (p.ej. iq1_s en el 27B) cada capa
        // residente extra sostiene ~1GB de W_T f32 en weight_cache ⇒
        // CudaMallocFailed a pocas capas del arranque.
        self.maybeEvict();

        if (self.debug_enabled) {
            debug.dbg.printLevel(.info, "[layer_streamer] layer {d} ensured loaded\n", .{layer_idx});
        }
    }

    fn loadLayerSync(self: *Self, layer_idx: usize) !void {
        self.lock();
        self.states[layer_idx].store(.loading, .release);
        self.unlock();

        try self.layers[layer_idx].loadWeightsFromGguf(self.g, null);

        self.lock();
        self.states[layer_idx].store(.loaded, .release);
        _ = self.resident_count.fetchAdd(1, .acq_rel);
        self.unlock();
    }

    fn runLoad(streamer: *LayerStreamer, layer_idx: usize) void {
        if (streamer.debug_enabled) {
            debug.dbg.printLevel(.info, "[layer_streamer] runLoad START li={d} resident={d}\n", .{ layer_idx, streamer.resident_count.load(.acquire) });
        }
        const result = streamer.layers[layer_idx].loadWeightsFromGguf(streamer.g, null);
        if (streamer.debug_enabled) {
            debug.dbg.printLevel(.info, "[layer_streamer] runLoad DONE li={d} ok={any}\n", .{ layer_idx, result });
        }

        streamer.lock();
        if (result) {
            streamer.states[layer_idx].store(.loaded, .release);
            _ = streamer.resident_count.fetchAdd(1, .acq_rel);
            if (streamer.debug_enabled) {
                debug.dbg.printLevel(.info, "[layer_streamer] async load layer {d} OK (resident={d})\n", .{ layer_idx, streamer.resident_count.load(.acquire) });
            }
        } else |e| {
            streamer.states[layer_idx].store(.unloaded, .release);
            if (streamer.debug_enabled) {
                debug.dbg.printLevel(.info, "[layer_streamer] async load layer {d} FAILED: {}\n", .{ layer_idx, e });
            }
        }
        streamer.unlock();
    }

    pub fn unloadLayer(self: *Self, layer_idx: usize) void {
        self.lock();
        const state = self.states[layer_idx].load(.acquire);
        if (state != .loaded) {
            self.unlock();
            return;
        }
        self.unlock();

        self.layers[layer_idx].unloadWeights();

        self.lock();
        self.states[layer_idx].store(.unloaded, .release);
        _ = self.resident_count.fetchSub(1, .acq_rel);
        self.unlock();

        if (self.debug_enabled) {
            debug.dbg.printLevel(.info, "[layer_streamer] layer {d} unloaded\n", .{layer_idx});
        }
    }

    fn maybeEvict(self: *Self) void {
        const count = self.resident_count.load(.acquire);
        if (count <= self.max_resident) return;

        self.lock();
        defer self.unlock();

        var evicted: usize = 0;
        while (self.resident_count.load(.acquire) > self.max_resident and evicted < self.layers.len) : (evicted += 1) {
            var lru_idx: usize = 0;
            var lru_time: u64 = std.math.maxInt(u64);
            for (self.states, 0..) |s, i| {
                if (s.load(.acquire) == .loaded) {
                    const t = self.last_used[i].load(.acquire);
                    if (t < lru_time) {
                        lru_time = t;
                        lru_idx = i;
                    }
                }
            }

            self.states[lru_idx].store(.unloaded, .release);
            _ = self.resident_count.fetchSub(1, .acq_rel);

            self.unlock();
            self.layers[lru_idx].unloadWeights();
            self.lock();

            if (self.debug_enabled) {
                debug.dbg.printLevel(.info, "[layer_streamer] evicted layer {d} (LRU, resident={d})\n", .{ lru_idx, self.resident_count.load(.acquire) });
            }
        }
        // Notify VramBudget if callback is set
        if (self.vram_budget) |vram| {
            vram.maybeEvict(vram_budget.Category.weights, 0) catch {};
        }
    }

    /// Ensure layer weights are on GPU (upload if needed, wait for completion)
    pub fn ensureLayerOnGpu(self: *Self, layer_idx: usize) !void {
        if (self.gpu_weight_pool) |pool| {
            const gpu_state = self.gpu_states[layer_idx].load(.acquire);
            if (gpu_state == .resident) {
                try pool.waitForLayer(layer_idx);
                self.last_used[layer_idx].store(self.tick.fetchAdd(1, .acq_rel), .release);
                return;
            }
            if (gpu_state == .uploading) {
                // Wait for ongoing upload
                if (self.gpu_weight_pool) |p| {
                    try p.waitForLayer(layer_idx);
                }
                self.gpu_states[layer_idx].store(.resident, .release);
                self.last_used[layer_idx].store(self.tick.fetchAdd(1, .acq_rel), .release);
                return;
            }
            // Start upload
            self.gpu_states[layer_idx].store(.uploading, .release);
            _ = try self.gpu_weight_pool.?.uploadLayer(&self.layers[layer_idx], layer_idx);
            self.gpu_states[layer_idx].store(.resident, .release);
            try self.gpu_weight_pool.?.waitForLayer(layer_idx);
        }
    }

    /// Upload layer weights asynchronously (called during prefetchNext)
    pub fn uploadLayerAsync(self: *Self, layer_idx: usize) !void {
        if (self.gpu_weight_pool) |p| {
            const gpu_state = self.gpu_states[layer_idx].load(.acquire);
            if (gpu_state != .not_resident) return;
            self.gpu_states[layer_idx].store(.uploading, .release);
            _ = try p.uploadLayer(&self.layers[layer_idx], layer_idx);
            self.gpu_states[layer_idx].store(.resident, .release);
        }
    }

    /// Evict layer from GPU (called by tryEvictAfterForward / maybeEvict)
    pub fn evictLayerFromGpu(self: *Self, layer_idx: usize) !void {
        if (self.gpu_weight_pool) |pool| {
            if (self.gpu_states[layer_idx].load(.acquire) == .resident) {
                try pool.evictLayer(layer_idx);
                self.gpu_states[layer_idx].store(.not_resident, .release);
            }
        }
    }

    /// Called after forwardGPU to trigger LRU/VRAM eviction
    pub fn tryEvictAfterForward(self: *Self, layer_idx: usize) !void {
        self.last_used[layer_idx].store(self.tick.fetchAdd(1, .acq_rel), .release);
        self.maybeEvictGpu();
    }

    /// LRU eviction with VRAM budget awareness for GPU weights
    fn maybeEvictGpu(self: *Self) !void {
        if (self.gpu_weight_pool) |pool| {
            // Check if we need to evict based on VRAM budget
            if (self.vram_budget) |_| {
                const used = pool.bytesUsed();
                const budget = self.vram_budget.?.weights_budget;
                if (used > budget) {
                    const need = used - budget + 1024 * 1024; // 1MB buffer
                    try pool.makeRoom(need);
                }
            }
            // Also enforce max_resident
            var resident_count: usize = 0;
            for (self.gpu_states) |s| {
                if (s.load(.acquire) == .resident) resident_count += 1;
            }
            if (resident_count > self.max_resident) {
                // Find LRU resident layer
                var lru_idx: usize = 0;
                var lru_time: u64 = std.math.maxInt(u64);
                for (self.gpu_states, 0..) |s, i| {
                    if (s.load(.acquire) == .resident) {
                        const t = self.last_used[i].load(.acquire);
                        if (t < lru_time) {
                            lru_time = t;
                            lru_idx = i;
                        }
                    }
                }
                self.evictLayerFromGpu(lru_idx) catch {};
            }
        }
    }

    pub fn prefetchNext(self: *Self, layer_idx: usize) !void {
        const next = layer_idx + 1;
        if (next >= self.layers.len) return;
        try self.prefetchLayer(next);
        // Async upload next layer to GPU
        if (self.gpu_weight_pool) |_| {
            try self.uploadLayerAsync(next);
        }
        self.maybeEvict();
    }

    pub fn residentCount(self: *Self) usize {
        return self.resident_count.load(.acquire);
    }

    pub fn reportMetrics(self: *Self) void {
        if (!self.debug_enabled) return;
        var loaded_count: usize = 0;
        var loading_count: usize = 0;
        for (self.states) |s| {
            const state = s.load(.acquire);
            if (state == .loaded) loaded_count += 1;
            if (state == .loading) loading_count += 1;
        }
        debug.dbg.printLevel(.info, "[layer_streamer] resident={d} loaded={d} loading={d} max={d}\n", .{
            self.residentCount(), loaded_count, loading_count, self.max_resident,
        });
    }
};

const testing = std.testing;

test "[layer_streamer] state transitions and LRU tick" {
    var states: [3]std.atomic.Value(LayerState) = undefined;
    var i: usize = 0;
    while (i < 3) : (i += 1) {
        states[i] = std.atomic.Value(LayerState).init(.unloaded);
    }

    states[0].store(.loaded, .release);
    states[1].store(.loaded, .release);
    states[2].store(.unloaded, .release);

    var loaded: usize = 0;
    for (states) |s| {
        if (s.load(.acquire) == .loaded) loaded += 1;
    }
    try testing.expectEqual(loaded, 2);
    try testing.expect(states[2].load(.acquire) == .unloaded);

    states[0].store(.unloaded, .release);
    try testing.expect(states[0].load(.acquire) == .unloaded);
}

test "[layer_streamer] prefetchNext bounds — next >= len is skipped" {
    const num_layers = 3;
    const last_idx = num_layers - 1;
    const next = last_idx + 1;
    try testing.expect(next >= num_layers);
}

test "[layer_streamer] lock/unlock spin via tryLock" {
    var mutex = std.atomic.Mutex.unlocked;
    while (!mutex.tryLock()) {
        std.Thread.yield() catch {};
    }
    mutex.unlock();
}
