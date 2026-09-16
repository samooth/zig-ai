//! Server-Sent Events (SSE) helpers sobre httpx.zig.
//!
//! httpx.zig tiene `Context.startSSEStreaming()` que devuelve un `StreamWriter`.
//! Aquí envolvemos eso en un API más idiomática para los handlers:
//!   - `SseWriter` con métodos `event(name)`, `data(text)`, `id(...)`, `send()`
//!   - Helpers de formato específicos para OpenAI / Anthropic / Ollama.

const std = @import("std");
const httpx = @import("httpx");

/// Writer incremental de eventos SSE. Mantiene estado entre `event()`,
/// `data()` y `send()` para emitir líneas `event:`, `data:` y el terminador
/// correctamente.
pub const SseWriter = struct {
    allocator: std.mem.Allocator,
    /// StreamWriter de httpx (no owned, vive en el Context).
    raw: httpx.server.StreamWriter,
    /// Buffer del evento en construcción.
    pending_event: ?[]const u8 = null,
    pending_id: ?[]const u8 = null,
    pending_data: std.ArrayList(u8) = .empty,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, raw: httpx.server.StreamWriter) Self {
        return .{
            .allocator = allocator,
            .raw = raw,
        };
    }

    pub fn deinit(self: *Self) void {
        self.pending_data.deinit(self.allocator);
    }

    /// Set del campo `event:` (cliente lo lee en `evt.event`).
    pub fn event(self: *Self, name: []const u8) !void {
        self.pending_event = name;
    }

    /// Set del campo `id:` (cliente lo lee en `evt.lastEventId`).
    pub fn id(self: *Self, val: []const u8) !void {
        self.pending_id = val;
    }

    /// Append a `data:` con un fragmento. NO flushea. Usar `send()` para emitir.
    pub fn data(self: *Self, chunk: []const u8) !void {
        try self.pending_data.appendSlice(self.allocator, chunk);
    }

    /// Emite el evento pendiente (event:, id:, data:, línea vacía) al socket.
    /// Limpia el buffer para el siguiente.
    pub fn send(self: *Self) !void {
        if (self.pending_id) |idv| {
            try self.raw.writeAll("id: ");
            try self.raw.writeAll(idv);
            try self.raw.writeAll("\n");
        }
        if (self.pending_event) |name| {
            try self.raw.writeAll("event: ");
            try self.raw.writeAll(name);
            try self.raw.writeAll("\n");
        }
        // data: con multilínea: cada línea del contenido lleva su prefijo
        var lines = std.mem.splitScalar(u8, self.pending_data.items, '\n');
        while (lines.next()) |line| {
            try self.raw.writeAll("data: ");
            try self.raw.writeAll(line);
            try self.raw.writeAll("\n");
        }
        try self.raw.writeAll("\n");
        self.pending_event = null;
        self.pending_id = null;
        self.pending_data.clearRetainingCapacity();
    }

    /// Escribe un comentario (`: ...\n\n`) — útil como heartbeat keep-alive.
    /// NO cuenta como evento (los clientes lo ignoran).
    pub fn writeComment(self: *Self, text: []const u8) !void {
        try self.raw.writeAll(": ");
        try self.raw.writeAll(text);
        try self.raw.writeAll("\n\n");
    }
};
