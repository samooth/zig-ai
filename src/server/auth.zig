//! Auth — tabla de API keys con comparación timing-safe.
//!
//! T3 del plan server F2. Fuentes de keys:
//!   1. `--api-key-file <path>`: una key por línea, comentarios `#`,
//!      permisos 0600 verificados al cargar (rechazo si group/other).
//!   2. `ZIG_AI_API_KEY` env var: single-key fallback.
//!
//! Seguridad:
//!   - Comparación timing-safe (constante en la longitud de la key).
//!   - Key-id = sha256(key)[0..12] para logs/audit SIN exponer la key.
//!   - Sin keys Y bind no-loopback => el server se niega a arrancar
//!     (InsecureConfiguration) — impide exponer generación sin auth.
//!
//! Thread-safety: la tabla es inmutable tras init (sólo lectura en los
//! handlers); no requiere locks.

const std = @import("std");
const debugz = @import("debug");

pub const AuthError = error{
    /// El key-file existe pero sus permisos son > 0600.
    KeyFilePermissions,
    /// El key-file no se puede leer/parsear.
    KeyFileUnreadable,
    /// Sin keys configuradas Y bind no-loopback.
    InsecureConfiguration,
    OutOfMemory,
};

/// Una key registrada (pre-hashed para lookup O(1) y key-id para audit).
pub const Key = struct {
    /// sha256(key) completo — no reversible, no logueable como key.
    hash: [32]u8,
    /// Prefijo del hash para identificar en audit (12 hex chars).
    id: [12]u8,
};

pub const Auth = struct {
    allocator: std.mem.Allocator,
    keys: []Key = &.{},
    /// True si la fuente fue env var (para mensajes de arranque).
    from_env: bool = false,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Self) void {
        if (self.keys.len > 0) self.allocator.free(self.keys);
        self.keys = &.{};
    }

    /// Carga keys desde file (0600, una por línea, # comentarios) y/o env.
    /// `loopback` indica si el server se bindea sólo a 127.0.0.1/::1.
    /// Sin keys: OK sólo si loopback (documentado); error si público.
    pub fn load(
        self: *Self,
        key_file: ?[]const u8,
        env_key: ?[]const u8,
        loopback: bool,
    ) AuthError!void {
        var list: std.ArrayList(Key) = .empty;
        errdefer list.deinit(self.allocator);

        if (key_file) |path| {
            try self.loadFile(self.allocator, path, &list);
        }
        if (env_key) |ek| {
            if (ek.len > 0) {
                try self.appendKey(self.allocator, &list, ek);
                self.from_env = true;
            }
        }

        if (list.items.len == 0 and !loopback) {
            // errdefer ya libera `list` — no deinit manual (double-free).
            return AuthError.InsecureConfiguration;
        }
        self.keys = try list.toOwnedSlice(self.allocator);
        debugz.dbg.printLevel(.info, "[server] auth: {d} keys cargadas ({s})\n", .{
            self.keys.len,
            if (key_file != null) "file" else if (self.from_env) "env" else "sin auth (loopback)",
        });
    }

    fn loadFile(self: *Self, allocator: std.mem.Allocator, path: []const u8, list: *std.ArrayList(Key)) AuthError!void {
        const io = std.Io.Threaded.global_single_threaded.io();

        // 1) Verificar permisos ANTES de leer: 0600 exacto (ni group ni other).
        const st = std.Io.Dir.cwd().statFile(io, path, .{}) catch
            return AuthError.KeyFileUnreadable;
        const perm: u32 = @intFromEnum(st.permissions) & 0o777;
        if (perm != 0o600) {
            debugz.dbg.printLevel(.info, "[server] auth: key-file permisos {o} != 600 — rechazado\n", .{perm});
            return AuthError.KeyFilePermissions;
        }

        // 2) Leer el file entero.
        const f = std.Io.Dir.cwd().openFile(io, path, .{}) catch return AuthError.KeyFileUnreadable;
        defer f.close(io);
        var read_buf: [64 * 1024]u8 = undefined;
        var reader = f.reader(io, &read_buf);
        var contents: std.ArrayList(u8) = .empty;
        defer contents.deinit(allocator);
        while (true) {
            const chunk = reader.interface.readAlloc(allocator, 4096) catch break;
            defer allocator.free(chunk);
            contents.appendSlice(allocator, chunk) catch return AuthError.OutOfMemory;
        }

        // 3) Parsear líneas: trim, skip vacías y #comentarios.
        var lines = std.mem.splitScalar(u8, contents.items, '\n');
        while (lines.next()) |raw_line| {
            const line = std.mem.trim(u8, raw_line, " \t\r");
            if (line.len == 0 or line[0] == '#') continue;
            try self.appendKey(allocator, list, line);
        }
    }

    fn appendKey(self: *Self, allocator: std.mem.Allocator, list: *std.ArrayList(Key), raw: []const u8) AuthError!void {
        _ = self;
        var k: Key = undefined;
        std.crypto.hash.sha2.Sha256.hash(raw, &k.hash, .{});
        _ = hexEncode12(k.hash[0..6], &k.id);
        list.append(allocator, k) catch return AuthError.OutOfMemory;
    }

    /// Valida un Bearer token. Devuelve el key-id si es válido, null si no.
    /// Timing-safe: compara el sha256 del token contra cada key (el hash
    /// elimina length-leak del token crudo).
    pub fn validate(self: *const Self, token: []const u8) ?[]const u8 {
        if (self.keys.len == 0) return null;
        var tok_hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(token, &tok_hash, .{});
        for (self.keys) |k| {
            if (std.crypto.timing_safe.eql([32]u8, tok_hash, k.hash)) {
                return &k.id;
            }
        }
        return null;
    }

    /// ¿Auth activo? (false = loopback sin keys, permitir todo).
    pub fn enabled(self: *const Self) bool {
        return self.keys.len > 0;
    }

    /// Número de keys cargadas (para logs de arranque).
    pub fn keyCount(self: *const Self) usize {
        return self.keys.len;
    }
};

fn hexEncode12(bytes: []const u8, out: *[12]u8) []const u8 {
    const hex = "0123456789abcdef";
    var i: usize = 0;
    while (i < 6) : (i += 1) {
        out[i * 2] = hex[bytes[i] >> 4];
        out[i * 2 + 1] = hex[bytes[i] & 0xF];
    }
    return out;
}

/// Extrae el token del header Authorization (Bearer <token>) o
/// x-api-key (Anthropic). Devuelve null si ausente/malformado.
pub fn extractToken(auth_header: ?[]const u8, api_key_header: ?[]const u8) ?[]const u8 {
    if (auth_header) |h| {
        if (std.mem.startsWith(u8, h, "Bearer ")) return h["Bearer ".len..];
        if (std.mem.startsWith(u8, h, "bearer ")) return h["bearer ".len..];
    }
    if (api_key_header) |h| {
        if (h.len > 0) return h;
    }
    return null;
}

// ─── Tests ───────────────────────────────────────────────────────────────

test "validate accepts registered key, rejects others (timing-safe)" {
    const allocator = std.testing.allocator;
    var auth = Auth.init(allocator);
    defer auth.deinit();

    var list: std.ArrayList(Key) = .empty;
    defer list.deinit(allocator);
    try auth.appendKey(allocator, &list, "secret-key-1");
    auth.keys = try list.toOwnedSlice(allocator);
    // NOTA: toOwnedSlice consume list; el defer deinit de list no corre.

    const id = auth.validate("secret-key-1");
    try std.testing.expect(id != null);
    try std.testing.expectEqual(@as(?[]const u8, null), auth.validate("wrong"));
    try std.testing.expectEqual(@as(?[]const u8, null), auth.validate(""));
}

test "extractToken parses Bearer and x-api-key" {
    try std.testing.expectEqualStrings("tok123", extractToken("Bearer tok123", null).?);
    try std.testing.expectEqualStrings("tok456", extractToken(null, "tok456").?);
    try std.testing.expectEqual(@as(?[]const u8, null), extractToken("Basic abc", null));
    try std.testing.expectEqual(@as(?[]const u8, null), extractToken("Bearer", null));
    try std.testing.expectEqual(@as(?[]const u8, null), extractToken(null, ""));
}
