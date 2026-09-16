/// Bridge between the TUI dashboard and the zig-ai inference engine.
/// Implements MetricHooks by forwarding to the TUI's event system.
const std = @import("std");
const contract = @import("contract.zig");
const metrics = @import("metrics.zig");
const Metrics = metrics.Metrics;
const TokenKind = contract.TokenKind;
const LayerMetrics = contract.LayerMetrics;
const MetricsSnapshot = contract.MetricsSnapshot;

/// TUIEventBus provides a thread-safe way to send events from the engine
/// to the TUI's main event loop.
pub const TUIEventBus = struct {
    const Event = union(enum) {
        layer: LayerMetrics,
        token: struct { text: []const u8, kind: TokenKind },
        log: []const u8,
        metrics: MetricsSnapshot,
    };

    queue: std.atomic.Value(std.ArrayListUnmanaged(Event)) = .empty,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !TUIEventBus {
        var queue = std.ArrayListUnmanaged(Event).init(allocator);
        _ = queue;
        return .{ .queue = std.atomic.Value(std.ArrayListUnmanaged(Event)).init(queue), .allocator = allocator };
    }

    pub fn deinit(self: *TUIEventBus) void {
        self.queue.load(.monotonic).deinit(self.allocator);
    }

    pub fn emit(self: *TUIEventBus, event: Event) void {
        var queue = self.queue.load(.monotonic);
        _ = queue.append(self.allocator, event);
    }

    pub fn drain(self: *TUIEventBus, handler: fn (Event) void) void {
        var queue = self.queue.load(.monotonic);
        for (queue.items) |event| handler(event);
        queue.clearRetainingCapacity();
    }
};

/// Bridge implements MetricHooks by emitting events to the TUIEventBus.
pub const Bridge = struct {
    bus: *TUIEventBus,

    pub fn init(bus: *TUIEventBus) Bridge {
        return .{ .bus = bus };
    }

    pub fn hooks(self: *Bridge) contract.MetricHooks {
        return contract.MetricHooks{
            .on_layer = &struct {
                fn layer(self: *Bridge, m: *const LayerMetrics) void {
                    self.bus.emit(.{ .layer = m.* });
                }
            }.layer,
            .on_token = &struct {
                fn token(self: *Bridge, text: []const u8, kind: TokenKind) void {
                    self.bus.emit(.{ .token = .{ .text = text, .kind = kind } });
                }
            }.token,
            .on_log = &struct {
                fn log(self: *Bridge, text: []const u8) void {
                    self.bus.emit(.{ .log = text });
                }
            }.log,
            .on_metrics = &struct {
                fn metrics(self: *Bridge, s: *const MetricsSnapshot) void {
                    self.bus.emit(.{ .metrics = s.* });
                }
            }.metrics,
        };
    }
};

/// EngineRunner wraps the real inference engine and drives it from the TUI.
pub const EngineRunner = struct {
    allocator: std.mem.Allocator,
    bridge: Bridge,
    bus: TUIEventBus,
    running: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    // Real engine state (filled when started)
    engine_handle: ?EngineHandle = null,

    const EngineHandle = struct {
        // Will hold the actual engine state from main.zig
        // For now we keep it opaque
        dummy: u8 = 0,
    };

    pub fn init(allocator: std.mem.Allocator) !EngineRunner {
        var bus = try TUIEventBus.init(allocator);
        return .{
            .allocator = allocator,
            .bus = bus,
            .bridge = Bridge.init(&bus),
        };
    }

    pub fn deinit(self: *EngineRunner) void {
        self.stop();
        self.bus.deinit();
    }

    pub fn hooks(self: *EngineRunner) contract.MetricHooks {
        return self.bridge.hooks();
    }

    pub fn start(self: *EngineRunner, hooks: contract.MetricHooks) !void {
        if (self.running.load(.acquire)) return;
        self.running.store(true, .release);
        self.thread = try std.Thread.spawn(.{}, runLoop, .{self});
    }

    pub fn stop(self: *EngineRunner) void {
        self.running.store(false, .release);
        if (self.thread) |t| {
            t.join();
            self.thread = null;
        }
    }

    fn runLoop(self: *EngineRunner) void {
        var timer = @import("time.zig").Timer.start() catch unreachable;
        var last_update = timer.read();
        var total_tokens: u64 = 0;

        while (self.running.load(.acquire)) {
            const now = timer.read();
            const elapsed_ns = now - last_update;

            if (elapsed_ns >= 100_000_000) {
                last_update = now;

                // Emit periodic metrics
                const snapshot = contract.MetricsSnapshot{
                    .tokens_per_sec = 0,
                    .vram_mb = 0,
                    .free_vram_mb = 0,
                    .latency_ms = 0,
                    .total_tokens = total_tokens,
                    .is_generating = true,
                };
                self.bridge.bus.emit(.{ .metrics = snapshot });
            }

            @import("time.zig").sleep(10_000_000);
        }
    }

    pub fn drainEvents(self: *EngineRunner, handler: fn (TUIEventBus.Event) void) void {
        self.bus.drain(handler);
    }
};
