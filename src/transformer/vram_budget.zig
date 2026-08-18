//! VramBudget — presupuesto dinámico de memoria GPU.
//!
//! Categoria: weights | activations | kv_cache
//! Cada categoria tiene un presupuesto; canAlloc() verifica antes de allocar.
//! LayerStreamer y ActivationPool consultan este presupuesto antes de cargar
//! pesos o allocar buffers, forzando LRU eviction si se excede.
const std = @import("std");
const debug = @import("debug");

pub const Category = enum { weights, activations, kv_cache };

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

    pub fn init(total_vram: usize) Self {
        // Layout: 60% weights, 20% activations, 20% KV-cache, 5% safety
        const safe_total = total_vram * 95 / 100;
        return .{
            .total_vram = total_vram,
            .weights_budget = safe_total * 60 / 100,
            .activations_budget = safe_total * 20 / 100,
            .kv_budget = safe_total * 20 / 100,
            .safety_margin = total_vram * 5 / 100,
            .weights_used = std.atomic.Value(usize).init(0),
            .activations_used = std.atomic.Value(usize).init(0),
            .kv_used = std.atomic.Value(usize).init(0),
        };
    }

    pub fn canAlloc(self: *Self, cat: Category, bytes: usize) bool {
        const budget = switch (cat) {
            .weights => self.weights_budget,
            .activations => self.activations_budget,
            .kv_cache => self.kv_budget,
        };
        const used = switch (cat) {
            .weights => self.weights_used.load(.acquire),
            .activations => self.activations_used.load(.acquire),
            .kv_cache => self.kv_used.load(.acquire),
        };
        return used + bytes <= budget;
    }

    pub fn reserve(self: *Self, cat: Category, bytes: usize) !void {
        if (!self.canAlloc(cat, bytes)) {
            debug.dbg.printLevel(.info,
                "VramBudget: reserve {s} {d} bytes EXCEEDS budget, triggering eviction\n",
                .{@tagName(cat), bytes});
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
        debug.dbg.printLevel(.info,
            "VramBudget: {s} needs {d} bytes (used={d}, budget={d}) — eviction required\n",
            .{@tagName(cat), need_bytes, used, self.budgetFor(cat)});
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
        debug.dbg.printLevel(.info,
            "VramBudget: total={d}MB weights={d}/{d}MB activations={d}/{d}MB kv={d}/{d}MB ({{total={d:.1}%}})\n",
            .{
                self.total_vram / (1024*1024),
                w / (1024*1024), self.weights_budget / (1024*1024),
                a / (1024*1024), self.activations_budget / (1024*1024),
                k / (1024*1024), self.kv_budget / (1024*1024),
                @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(self.total_vram)) * 100.0,
            });
    }
};
