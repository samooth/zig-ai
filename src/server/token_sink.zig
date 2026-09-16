//! TokenSink — interfaz de streaming token-por-token del engine.
//!
//! Un TokenSink recibe tokens a medida que el engine los genera (dentro
//! del BatchingLoop, en el hilo del engine) y decide qué hacer con ellos:
//! emitir un chunk SSE, escribir NDJSON, acumular para una response JSON,
//! o descartar.
//!
//! Contrato: `emit` NUNCA bloquea. Si el cliente TCP va lento, el
//! transporte aplica backpressure por su cuenta; el loop del engine no
//! puede esperar al cliente o bloquearía a las demás secuencias del batch.
//! `emit` devuelve error sólo para señales de control (stop-sequence).

const std = @import("std");

/// Motivo de finalización de una secuencia.
pub const FinishReason = enum {
    stop, // EOS natural o stop-sequence
    length, // max_tokens alcanzado
    cancelled, // timeout / disconnect del cliente
    err, // error del engine
};

/// Métricas de una secuencia completada.
pub const Usage = struct {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_ms: u64,
};

/// Señales de control que un sink puede devolver al engine.
pub const SinkSignal = error{
    /// Stop-sequence detectada por el sink: terminar la secuencia
    /// (el texto hasta ese punto ya fue emitido/acumulado).
    StopSequenceHit,
};

/// vtable de un consumidor de tokens.
pub const TokenSink = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        emit: *const fn (ctx: *anyopaque, token: u32, text: []const u8) SinkSignal!void,
        finish: *const fn (ctx: *anyopaque, reason: FinishReason, usage: Usage) void,
    };

    pub fn emit(self: TokenSink, token: u32, text: []const u8) SinkSignal!void {
        return self.vtable.emit(self.ctx, token, text);
    }

    pub fn finish(self: TokenSink, reason: FinishReason, usage: Usage) void {
        self.vtable.finish(self.ctx, reason, usage);
    }
};

/// Sink colector: acumula tokens/texto para responses no-streaming.
/// Sólo lo toca el hilo del engine (sin locks).
pub const CollectingSink = struct {
    allocator: std.mem.Allocator,
    tokens: std.ArrayList(u32) = .empty,
    text: std.ArrayList(u8) = .empty,
    finish_reason: ?FinishReason = null,
    usage: ?Usage = null,
    /// Stop-sequences: recortadas del texto final (el cliente no debe verlas).
    stop_seqs: []const []const u8 = &.{},

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Self) void {
        self.tokens.deinit(self.allocator);
        self.text.deinit(self.allocator);
    }

    pub fn sink(self: *Self) TokenSink {
        return .{ .ctx = self, .vtable = &vtable };
    }

    const vtable = TokenSink.VTable{
        .emit = emitImpl,
        .finish = finishImpl,
    };

    fn emitImpl(ctx: *anyopaque, token: u32, text: []const u8) SinkSignal!void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.tokens.append(self.allocator, token) catch return; // OOM: drop, no matar el batch
        self.text.appendSlice(self.allocator, text) catch {};
    }

    fn finishImpl(ctx: *anyopaque, reason: FinishReason, usage: Usage) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.finish_reason = reason;
        self.usage = usage;
        for (self.stop_seqs) |ss| {
            if (ss.len > 0 and std.mem.endsWith(u8, self.text.items, ss)) {
                self.text.shrinkRetainingCapacity(self.text.items.len - ss.len);
            }
        }
    }
};

test "collecting sink accumulates and strips stop-seq" {
    const allocator = std.testing.allocator;
    var cs = CollectingSink.init(allocator);
    defer cs.deinit();
    var stops = [_][]const u8{"\nUser:"};
    cs.stop_seqs = &stops;

    const s = cs.sink();
    try s.emit(1, "Hello");
    try s.emit(2, " world");
    try s.emit(3, "\nUser:");
    s.finish(.stop, .{ .prompt_tokens = 4, .completion_tokens = 3, .total_ms = 10 });

    try std.testing.expectEqualStrings("Hello world", cs.text.items);
    try std.testing.expectEqual(@as(usize, 3), cs.tokens.items.len);
    try std.testing.expect(cs.usage.?.completion_tokens == 3);
}
