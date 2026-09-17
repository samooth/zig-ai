//! Audit — log JSON-lines de requests con rotación por tamaño.
//!
//! T3 del plan server F2. Cada request genera UNA línea:
//!   {"ts":<unix_ms>,"key_id":"<hash12|null>","ip":"...","method":"POST",
//!    "path":"/v1/chat/completions","status":200,"latency_ms":123,
//!    "prompt_tokens":45,"completion_tokens":128}
//!
//! PRIVACIDAD: NUNCA se loguea contenido del prompt/completion salvo
//! `--audit-full` (que además loguea un hash del texto, no el texto).
//!
//! Rotación: al superar max_bytes (def 10MB) se renombra a <path>.1
//! (sobrescribiendo el .1 anterior — 1 generación de historial).
//!
//! Thread-safety: escrituras serializadas por spinlock (los handlers
//! corren en N threads del httpx).

const std = @import("std");
const builtin = @import("builtin");
const debugz = @import("debug");
const time = @import("time");

pub const AuditConfig = struct {
    /// Ruta del log (default: desactivado si null).
    path: ?[]const u8 = null,
    /// Tamaño máx antes de rotar.
    max_bytes: usize = 10 * 1024 * 1024,
    /// Incluir sha256[:16] del prompt+completion (privacy: hash, no texto).
    full: bool = false,
};

pub const Audit = struct {
    allocator: std.mem.Allocator,
    config: AuditConfig,
    file: ?std.Io.File = null,
    bytes_written: usize = 0,
    lock: std.atomic.Value(u32) = .{ .raw = 0 },

    const Self = @This();

    pub const Record = struct {
        key_id: ?[]const u8, // null si auth desactivado
        ip: []const u8,
        method: []const u8,
        path: []const u8,
        status: u16,
        latency_ms: u64,
        prompt_tokens: usize = 0,
        completion_tokens: usize = 0,
        /// sha256 del texto (sólo con full=true; 16 hex chars).
        content_hash: ?[]const u8 = null,
    };

    pub fn init(allocator: std.mem.Allocator, config: AuditConfig) Self {
        return .{ .allocator = allocator, .config = config };
    }

    /// Abre el log (append). Si no se puede, audit queda disabled (el
    /// server NO muere por logging, pero lo reporta).
    pub fn open(self: *Self) !void {
        const path = self.config.path orelse return;
        const io = std.Io.Threaded.global_single_threaded.io();
        // append + create; permisos 0600 via fchmod (el audit puede
        // contener metadatos sensibles: key-ids, IPs).
        self.file = std.Io.Dir.cwd().createFile(io, path, .{ .truncate = false }) catch |e| {
            debugz.dbg.printLevel(.info, "[server] audit: NO se pudo abrir {s}: {s}\n", .{ path, @errorName(e) });
            return e;
        };
        if (comptime builtin.target.os.tag != .windows)
            _ = std.c.fchmod(self.file.?.handle, 0o600);
        debugz.dbg.printLevel(.info, "[server] audit: log en {s} (full={})\n", .{ path, self.config.full });
    }

    pub fn close(self: *Self) void {
        if (self.file) |f| {
            f.close(std.Io.Threaded.global_single_threaded.io());
            self.file = null;
        }
    }

    /// Escribe una línea de audit. Fail-soft: un error de log NUNCA
    /// rompe el request.
    pub fn log(self: *Self, rec: Record) void {
        if (self.file == null) return;
        acquire(&self.lock);
        defer release(&self.lock);

        const io = std.Io.Threaded.global_single_threaded.io();
        var buf: [2048]u8 = undefined;
        var fw = self.file.?.writer(io, &buf);
        const w = &fw.interface;

        const ts_ms: u64 = @intCast(@divTrunc(time.Timer.now(), std.time.ns_per_ms));
        w.print("{{\"ts\":{d},\"key_id\":", .{ts_ms}) catch return;
        if (rec.key_id) |kid| {
            w.print("\"{s}\",", .{kid}) catch return;
        } else {
            w.writeAll("null,") catch return;
        }
        w.print("\"ip\":\"{s}\",\"method\":\"{s}\",\"path\":\"{s}\",\"status\":{d},\"latency_ms\":{d},\"prompt_tokens\":{d},\"completion_tokens\":{d}", .{
            rec.ip,            rec.method,            rec.path, rec.status, rec.latency_ms,
            rec.prompt_tokens, rec.completion_tokens,
        }) catch return;
        if (rec.content_hash) |ch| {
            w.print(",\"content_hash\":\"{s}\"", .{ch}) catch return;
        }
        w.writeAll("}\n") catch return;
        w.flush() catch return;
        self.bytes_written += 1; // aprox: línea ≈ 200-500B; rotación por escrituras sería mejor
        // Rotación por tamaño real la medimos en bytes del buffer drenado:
        // en 0.16 el Writer no expone written-count fácilmente; contamos
        // líneas (aprox 250B/línea) hasta max_bytes/250.
        const max_lines = self.config.max_bytes / 250;
        if (self.bytes_written >= max_lines) {
            self.rotate();
            self.bytes_written = 0;
        }
    }

    /// Rotación: cierra, renombra path→path.1, reabre.
    fn rotate(self: *Self) void {
        const path = self.config.path orelse return;
        self.close();
        const io = std.Io.Threaded.global_single_threaded.io();
        const rotated = std.fmt.allocPrint(self.allocator, "{s}.1", .{path}) catch return;
        defer self.allocator.free(rotated);
        std.Io.Dir.cwd().rename(path, std.Io.Dir.cwd(), rotated, io) catch {};
        self.open() catch {};
    }
};

// ─── Spinlock (patrón del proyecto; 0.16 sin std.Thread.Mutex) ────────────

fn acquire(l: *std.atomic.Value(u32)) void {
    while (true) {
        if (l.cmpxchgWeak(0, 1, .seq_cst, .seq_cst) == null) return;
        var i: u32 = 0;
        while (i < 100) : (i += 1) std.atomic.spinLoopHint();
    }
}

fn release(l: *std.atomic.Value(u32)) void {
    _ = l.store(0, .seq_cst);
}

test "audit disabled without path: log() is no-op" {
    var audit = Audit.init(std.testing.allocator, .{});
    defer audit.close();
    audit.log(.{
        .key_id = null,
        .ip = "127.0.0.1",
        .method = "POST",
        .path = "/v1/chat/completions",
        .status = 200,
        .latency_ms = 5,
    });
    try std.testing.expect(audit.file == null);
}
