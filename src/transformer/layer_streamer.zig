//! LayerStreamer — carga asíncrona y prefetch de capas para inferencia AirLLM-style.
//! Mientras la GPU computa la capa i, el thread pool descuenta/prefija la capa i+1
//! a RAM/GPU. LRU eviction mantiene max_resident capas cargadas.
const std = @import("std");
const hybrid_layer = @import("hybrid_layer");
const HybridLayer = hybrid_layer.HybridLayer;
const gguf = @import("gguf");
const model_config = @import("model_config");
const debug = @import("debug");

const LayerState = enum(u8) {
    unloaded = 0,
    loading = 1,
    loaded = 2,
};

pub const LayerStreamer = struct {
    allocator: std.mem.Allocator,
    g: *const gguf.GgufFile,
    layers: []HybridLayer,
    cfg: model_config.ModelConfig,
    states: []std.atomic.Value(LayerState),
    last_used: []std.atomic.Value(u64),
    max_resident: usize,
    resident_count: std.atomic.Value(usize),
    mutex: std.atomic.Mutex,
    spawned_threads: []?std.Thread,
    tick: std.atomic.Value(u64),
    debug_enabled: bool,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        layers: []HybridLayer,
        g: *const gguf.GgufFile,
        cfg: model_config.ModelConfig,
        max_resident: usize,
        num_workers: usize,
    ) !Self {
        _ = num_workers;
        const num_layers = cfg.block_count;

        const states = try allocator.alloc(std.atomic.Value(LayerState), num_layers);
        errdefer allocator.free(states);
        for (states) |*s| s.* = std.atomic.Value(LayerState).init(.unloaded);

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
            .last_used = last_used,
            .max_resident = max_resident,
            .resident_count = std.atomic.Value(usize).init(0),
            .mutex = .unlocked,
            .spawned_threads = spawned,
            .tick = std.atomic.Value(u64).init(0),
            .debug_enabled = false,
        };
    }

    pub fn deinit(self: *Self) void {
        // Join all spawned threads
        for (self.spawned_threads) |t| {
            if (t) |thread| thread.join();
        }
        self.allocator.free(self.states);
        self.allocator.free(self.last_used);
        self.allocator.free(self.spawned_threads);
    }

    pub fn enableDebug(self: *Self) void {
        self.debug_enabled = true;
    }

    pub fn setMaxResidentLayers(self: *Self, n: usize) void {
        self.max_resident = n;
        self.maybeEvict();
    }

    /// Spin-lock helper using std.atomic.Mutex (tryLock + yield).
    fn lock(self: *Self) void {
        while (!self.mutex.tryLock()) {
            std.Thread.yield() catch {};
        }
    }

    fn unlock(self: *Self) void {
        self.mutex.unlock();
    }

    /// Dispara carga async de los pesos de `layer_idx` (no bloquea).
    /// Si ya está cargado o en progreso, retorna inmediatamente.
    pub fn prefetchLayer(self: *Self, layer_idx: usize) !void {
        self.lock();
        defer self.unlock();

        const state = self.states[layer_idx].load(.acquire);
        if (state != .unloaded) return;

        self.states[layer_idx].store(.loading, .release);
        const thread = try std.Thread.spawn(.{}, runLoad, .{ self, layer_idx });
        self.spawned_threads[layer_idx] = thread;
    }

    /// Bloquea hasta que `layer_idx` esté loaded, luego marca como usado.
    pub fn ensureLayerLoaded(self: *Self, layer_idx: usize) !void {
        self.lock();
        // Wait while loading is in progress
        var state = self.states[layer_idx].load(.acquire);
        while (state == .loading) {
            self.unlock();
            std.Thread.yield() catch {};
            self.lock();
            state = self.states[layer_idx].load(.acquire);
        }
        self.states[layer_idx].store(.loading, .release);
        self.unlock();

        // If still unloaded, load synchronously
        if (state == .unloaded) {
            try self.loadLayerSync(layer_idx);
        }

        self.last_used[layer_idx].store(self.tick.fetchAdd(1, .acq_rel), .release);

        if (self.debug_enabled) {
            debug.dbg.printLevel(.info, "LayerStreamer: layer {d} ensured loaded\n", .{layer_idx});
        }
    }

    /// Carga sincronamente los pesos de una capa.
    fn loadLayerSync(self: *Self, layer_idx: usize) !void {
        self.lock();
        self.states[layer_idx].store(.loading, .release);
        self.unlock();

        try self.layers[layer_idx].loadWeightsFromGguf(self.g);

        self.lock();
        self.states[layer_idx].store(.loaded, .release);
        _ = self.resident_count.fetchAdd(1, .acq_rel);
        self.unlock();
    }

    /// Worker thread: carga pesos y notifica.
    fn runLoad(streamer: *LayerStreamer, layer_idx: usize) void {
        const result = streamer.layers[layer_idx].loadWeightsFromGguf(streamer.g);

        streamer.lock();
        if (result) {
            streamer.states[layer_idx].store(.loaded, .release);
            _ = streamer.resident_count.fetchAdd(1, .acq_rel);
            if (streamer.debug_enabled) {
                debug.dbg.printLevel(.info, "LayerStreamer: async load layer {d} OK\n", .{layer_idx});
            }
        } else |e| {
            streamer.states[layer_idx].store(.unloaded, .release);
            if (streamer.debug_enabled) {
                debug.dbg.printLevel(.info, "LayerStreamer: async load layer {d} FAILED: {}\n", .{layer_idx, e});
            }
        }
        streamer.unlock();
    }

    /// Libera pesos de una capa específica.
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
            debug.dbg.printLevel(.info, "LayerStreamer: layer {d} unloaded\n", .{layer_idx});
        }
    }

    /// Evict LRU hasta estar bajo el límite de residentes.
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
                debug.dbg.printLevel(.info, "LayerStreamer: evicted layer {d} (LRU)\n", .{lru_idx});
            }
        }
    }

    /// Prefetch de la capa siguiente (async). Llamar después de ensureLayerLoaded(i).
    pub fn prefetchNext(self: *Self, layer_idx: usize) !void {
        const next = layer_idx + 1;
        if (next >= self.layers.len) return;
        try self.prefetchLayer(next);
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
        debug.dbg.printLevel(.info, "LayerStreamer: resident={d} loaded={d} loading={d} max={d}\n", .{
            self.residentCount(), loaded_count, loading_count, self.max_resident,
        });
    }
};
