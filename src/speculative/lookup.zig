//! C7 — Lookup-fill: tabla n-grama del contexto (prompt + generados) que
//! propone drafts para rellenar slots libres del bloque de verificación.
//! Truco "lookup-fill" del estudio (combinación con DFlash/MTP, no variante):
//! hasta `max_drafts` tokens por ronda; los rechazados se truncan como en MTP.
//! CPU puro, sin dependencias nuevas: Wyhash sobre secuencias u32.
const std = @import("std");

pub const Config = struct {
    /// Longitud mínima de n-grama indexada (matches más cortos = más ruido).
    min_ngram: usize = 2,
    /// Longitud máxima intentada al redactar (match largo = más confianza).
    max_ngram: usize = 8,
    /// Tope de drafts por consulta (verify-window configurable hasta 16).
    max_drafts: usize = 16,
    /// Máximo de candidatos retenidos por n-grama (dedup por inserción).
    max_cands_per_key: usize = 4,
};

pub const NGramTable = struct {
    allocator: std.mem.Allocator,
    cfg: Config,
    /// clave = hash de la secuencia [L]u32 (L ∈ [min_ngram..max_ngram]);
    /// valor = siguientes tokens observados tras esa secuencia (orden de
    /// primera aparición, sin duplicados).
    map: std.AutoHashMapUnmanaged(u64, std.ArrayListUnmanaged(u32)) = .empty,
    total_entries: usize = 0,

    pub fn init(allocator: std.mem.Allocator, cfg: Config) NGramTable {
        return .{ .allocator = allocator, .cfg = cfg };
    }

    pub fn deinit(self: *NGramTable) void {
        var it = self.map.iterator();
        while (it.next()) |e| e.value_ptr.deinit(self.allocator);
        self.map.deinit(self.allocator);
    }

    fn hashSeq(seq: []const u32) u64 {
        var h = std.hash.Wyhash.init(0x6c6f6f6b); // "look"
        for (seq) |t| h.update(std.mem.asBytes(&t));
        return h.final();
    }

    /// Indexa TODOS los sufijos de longitud [min_ngram..max_ngram] que
    /// terminan en cada posición con siguiente token conocido.
    pub fn feed(self: *NGramTable, tokens: []const u32) !void {
        if (tokens.len <= self.cfg.min_ngram) return;
        const Lmax = @min(self.cfg.max_ngram, tokens.len - 1);
        var len = self.cfg.min_ngram;
        while (len <= Lmax) : (len += 1) {
            // ventana [i-len .. i) → siguiente tokens[i]
            var i: usize = len;
            while (i < tokens.len) : (i += 1) {
                const key = hashSeq(tokens[i - len .. i]);
                const gop = try self.map.getOrPut(self.allocator, key);
                if (!gop.found_existing) gop.value_ptr.* = .empty;
                const lst = gop.value_ptr;
                var dup = false;
                for (lst.items) |c| {
                    if (c == tokens[i]) {
                        dup = true;
                        break;
                    }
                }
                if (!dup and lst.items.len < self.cfg.max_cands_per_key) {
                    try lst.append(self.allocator, tokens[i]);
                    self.total_entries += 1;
                }
            }
        }
    }

    /// Propone hasta out.len tokens: match MÁS LARGO primero contra la cola
    /// `last_tokens`; avanza la ventana con cada propuesta (autorregresivo).
    /// Devuelve cuántos escribió en out.
    pub fn draft(self: *NGramTable, last_tokens: []const u32, out: []u32) usize {
        var n: usize = 0;
        var window_buf: [64]u32 = undefined;
        const wcap = @min(window_buf.len, self.cfg.max_ngram + out.len + 1);
        // semilla: cola del contexto hasta max_ngram
        var wlen: usize = @min(@min(last_tokens.len, self.cfg.max_ngram), wcap);
        @memcpy(window_buf[0..wlen], last_tokens[last_tokens.len - wlen ..][0..wlen]);

        while (n < out.len and wlen >= self.cfg.min_ngram) {
            const L = @min(self.cfg.max_ngram, wlen);
            var cand: ?u32 = null;
            // intenta L decreciente hasta min_ngram
            var l = L;
            while (l >= self.cfg.min_ngram) : (l -= 1) {
                if (wlen < l) continue;
                if (self.map.get(hashSeq(window_buf[wlen - l .. wlen]))) |lst| {
                    if (lst.items.len > 0) {
                        cand = lst.items[0];
                        break;
                    }
                }
                if (l == self.cfg.min_ngram) break;
            }
            const c = cand orelse break;
            out[n] = c;
            n += 1;
            if (wlen == wcap) {
                std.mem.copyForwards(u32, window_buf[0 .. wlen - 1], window_buf[1..wlen]);
                window_buf[wlen - 1] = c;
            } else {
                window_buf[wlen] = c;
                wlen += 1;
            }
        }
        return n;
    }

    pub fn reset(self: *NGramTable) void {
        var it = self.map.iterator();
        while (it.next()) |e| e.value_ptr.deinit(self.allocator);
        self.map.clearRetainingCapacity();
        self.total_entries = 0;
    }
};

test "draft reproduce continuación conocida" {
    var t = NGramTable.init(std.testing.allocator, .{});
    defer t.deinit();
    const seq = [_]u32{ 10, 20, 30, 40, 50, 60 };
    try t.feed(&seq);
    var out: [4]u32 = undefined;
    // cola [40,50] → debe proponer 60 y encadenar si hay datos
    const cola = [_]u32{ 30, 40, 50 };
    const n = t.draft(&cola, &out);
    try std.testing.expectEqual(@as(usize, 1), n);
    try std.testing.expectEqual(@as(u32, 60), out[0]);
}

test "match más largo gana sobre prefijos" {
    var t = NGramTable.init(std.testing.allocator, .{});
    defer t.deinit();
    try t.feed(&[_]u32{ 1, 2, 3, 100 });
    try t.feed(&[_]u32{ 9, 2, 3, 200 });
    var out: [2]u32 = undefined;
    // cola completa [1,2,3]: match de longitud 3 → 100 (no el 200 de [2,3])
    const n = t.draft(&[_]u32{ 1, 2, 3 }, &out);
    try std.testing.expectEqual(@as(usize, 1), n);
    try std.testing.expectEqual(@as(u32, 100), out[0]);
}

test "tabla vacía → 0 drafts; reset funciona" {
    var t = NGramTable.init(std.testing.allocator, .{});
    defer t.deinit();
    var out: [3]u32 = undefined;
    try std.testing.expectEqual(@as(usize, 0), t.draft(&[_]u32{ 5, 6 }, &out));
    try t.feed(&[_]u32{ 5, 6, 7 });
    t.reset();
    try std.testing.expectEqual(@as(usize, 0), t.draft(&[_]u32{ 5, 6 }, &out));
}

test "encadena propuestas autorregresivas" {
    var t = NGramTable.init(std.testing.allocator, .{ .min_ngram = 2 });
    defer t.deinit();
    // patrón periódico: después de X viene X+1
    var seq: [40]u32 = undefined;
    for (&seq, 0..) |*v, i| v.* = @intCast(i % 10);
    try t.feed(&seq);
    var out: [3]u32 = undefined;
    _ = t.draft(&[_]u32{ 8, 9 }, &out);
    try std.testing.expectEqual(@as(u32, 0), out[0]);
    try std.testing.expectEqual(@as(u32, 1), out[1]);
    try std.testing.expectEqual(@as(u32, 2), out[2]);
}

test "sin match corto suficiente → corta" {
    var t = NGramTable.init(std.testing.allocator, .{});
    defer t.deinit();
    try t.feed(&[_]u32{ 1, 2, 3, 4 });
    var out: [4]u32 = undefined;
    // cola [98,99]: no existe → 0
    try std.testing.expectEqual(@as(usize, 0), t.draft(&[_]u32{ 98, 99 }, &out));
}
