//! Integration contract between the zig-ai inference engine and the
//! zig-ai dashboard (TUI monitor).
//!
//! The engine's inference loop (`runHybridInference` / `runInference` in
//! `main.zig`) is instrumented with an optional set of *metric hooks*. When the
//! dashboard is active it passes a populated `MetricHooks` struct; when it is not
//! (plain CLI run) it passes `null` and the inference code is unchanged (zero
//! runtime overhead — every hook site is guarded by a null check).
//!
//! This module lives on the *engine* side and references no dashboard types, so
//! it can be imported by both `main.zig` (which emits the hooks) and the
//! dashboard's bridge (`engine.zig`, which consumes them) without creating an
//! import cycle.

const std = @import("std");

/// Per-layer timing/profiling data, emitted once per transformer layer.
pub const LayerMetrics = struct {
    layer_idx: usize,
    /// Total forward pass time for the layer (ms).
    forward_ms: f64,
    /// Time spent in the attention/SSM mixer (ms).
    attention_ms: f64,
    /// Time spent in the FFN (ms).
    ffn_ms: f64,
    /// Activations / weights resident memory for the layer (KB).
    memory_kb: f64,
    /// Whether this layer is an attention layer (vs SSM/ShortConv).
    is_attention: bool,
};

/// Kind of token for colorized streaming in the dashboard.
pub const TokenKind = enum {
    prompt,
    completion,
    special,
};

/// A point-in-time snapshot of scalar metrics. Plain numeric fields only
/// (no dashboard enums) to keep this module dependency-free.
pub const MetricsSnapshot = struct {
    tokens_per_sec: f64 = 0,
    /// GPU pool / device memory currently in use (MB). 0 when not applicable
    /// (CPU backend or before model load).
    vram_mb: f64 = 0,
    /// Free GPU memory (MB). 0 when not applicable.
    free_vram_mb: f64 = 0,
    latency_ms: f64 = 0,
    total_tokens: u64 = 0,
    is_generating: bool = false,
};

/// Optional callbacks from the inference loop to the dashboard. Every field is
/// an optional function pointer; a `null` field means "no-op".
pub const MetricHooks = struct {
    /// Called per transformer layer (after the layer forward completes).
    on_layer: ?*const fn (*const LayerMetrics) void = null,
    /// Called for each generated/streamed token (text + kind).
    on_token: ?*const fn ([]const u8, TokenKind) void = null,
    /// Called for status / info log lines.
    on_log: ?*const fn ([]const u8) void = null,
    /// Called periodically (per generated token) with scalar metrics.
    on_metrics: ?*const fn (*const MetricsSnapshot) void = null,

    pub fn layer(self: MetricHooks, m: LayerMetrics) void {
        if (self.on_layer) |cb| cb(&m);
    }

    pub fn token(self: MetricHooks, text: []const u8, kind: TokenKind) void {
        if (self.on_token) |cb| cb(text, kind);
    }

    pub fn log(self: MetricHooks, text: []const u8) void {
        if (self.on_log) |cb| cb(text);
    }

    pub fn metrics(self: MetricHooks, s: MetricsSnapshot) void {
        if (self.on_metrics) |cb| cb(&s);
    }
};
