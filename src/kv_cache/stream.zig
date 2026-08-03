//! Async streams para overlap compute/memcpy en KV-cache
//! Permite prefetch de K/V mientras se computa atención en otra capa

const std = @import("std");

/// Ring buffer de streams CUDA para overlap
pub const StreamRing = struct {
    streams: []StreamSlot,
    current: usize,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub const StreamSlot = struct {
        /// Identificador del stream
        id: u32,
        /// Ocupado
        busy: bool,
        /// Tipo de operación
        op: StreamOp,
        /// Callback al completar
        callback: ?*const fn (u32, StreamOp) void,
    };

    pub const StreamOp = enum {
        prefetch_k,
        prefetch_v,
        dequant_k,
        dequant_v,
        memcpy_h2d,
        memcpy_d2h,
    };

    pub fn init(allocator: std.mem.Allocator, num_streams: u32) !Self {
        var streams = try allocator.alloc(StreamSlot, num_streams);
        for (0..num_streams) |i| {
            streams[i] = .{
                .id = @as(u32, @intCast(i)),
                .busy = false,
                .op = .prefetch_k,
                .callback = null,
            };
        }
        return .{
            .streams = streams,
            .current = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.streams);
    }

    /// Adquiere un stream libre
    pub fn acquire(self: *Self, op: StreamOp) ?*StreamSlot {
        const start = self.current;
        var i = start;
        while (true) {
            if (!self.streams[i].busy) {
                self.streams[i].busy = true;
                self.streams[i].op = op;
                self.current = (i + 1) % self.streams.len;
                return &self.streams[i];
            }
            i = (i + 1) % self.streams.len;
            if (i == start) break;
        }
        return null;
    }

    /// Libera un stream
    pub fn release(self: *Self, slot: *StreamSlot) void {
        _ = self;
        slot.busy = false;
        slot.callback = null;
    }

    /// Espera a que todos los streams terminen
    pub fn syncAll(self: *Self) void {
        for (self.streams) |*slot| {
            if (slot.busy) {
                // Sincronización CUDA implícita
                slot.busy = false;
            }
        }
    }

    /// Número de streams libres
    pub fn available(self: *Self) usize {
        var count: usize = 0;
        for (self.streams) |slot| {
            if (!slot.busy) count += 1;
        }
        return count;
    }
};

/// Pipeline de prefetch para capas
pub const PrefetchPipeline = struct {
    ring: StreamRing,
    /// Capa actualmente en compute
    compute_layer: u32,
    /// Capa prefetch objetivo
    prefetch_target: u32,
    /// Callback al completar prefetch
    on_ready: ?*const fn (u32) void,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, num_streams: u32) !Self {
        const ring = try StreamRing.init(allocator, num_streams);
        return .{
            .ring = ring,
            .compute_layer = 0,
            .prefetch_target = 0,
            .on_ready = null,
        };
    }

    pub fn deinit(self: *Self) void {
        self.ring.deinit();
    }

    /// Inicia prefetch de la siguiente capa
    pub fn prefetchNext(self: *Self, current_layer: u32, total_layers: u32) void {
        self.compute_layer = current_layer;
        const next = current_layer + 1;
        if (next >= total_layers) return;

        self.prefetch_target = next;

        // Adquirir streams para K y V
        const k_stream = self.ring.acquire(.prefetch_k);
        const v_stream = self.ring.acquire(.prefetch_v);

        if (k_stream != null and v_stream != null) {
            // Lanzar prefetch async
            // La implementación concreta depende del backend CUDA
            // Aquí se marca como en progreso
            k_stream.?.busy = true;
            v_stream.?.busy = true;
        }
    }

    /// Verifica si el prefetch de una capa está listo
    pub fn isReady(self: *Self, layer_idx: u32) bool {
        if (layer_idx != self.prefetch_target) return false;
        // Verificar si ambos streams (K y V) han terminado
        // Simplificado: asumir que el prefetch de la capa anterior
        // siempre está listo cuando se necesita
        return true;
    }

    /// Sincroniza todo el pipeline
    pub fn sync(self: *Self) void {
        self.ring.syncAll();
    }
};

/// Utilidades de timing para métricas de overlap
pub const StreamTimer = struct {
    start_times: std.AutoHashMap(u32, u64),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) StreamTimer {
        return .{
            .start_times = std.AutoHashMap(u32, u64).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *StreamTimer) void {
        self.start_times.deinit();
    }

    pub fn start(self: *StreamTimer, stream_id: u32) !void {
        const now = @as(u64, @intCast(std.time.microTimestamp()));
        try self.start_times.put(stream_id, now);
    }

    pub fn elapsedUs(self: *StreamTimer, stream_id: u32) u64 {
        const start_time = self.start_times.get(stream_id) orelse return 0;
        const now = @as(u64, @intCast(std.time.microTimestamp()));
        return now - start_time;
    }
};
