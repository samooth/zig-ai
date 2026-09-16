//! Replicación exacta de la clasificación/pre-tokenización Unicode de
//! llama.cpp (src/unicode.cpp + src/unicode-data.cpp) para el tokenizer
//! byte-level BPE tipo GPT-2 / Qwen3.5.
const std = @import("std");
const unicode_data = @import("unicode_data");

pub const FLAG_UNDEFINED: u16 = 0x0001;
pub const FLAG_NUMBER: u16 = 0x0002;
pub const FLAG_LETTER: u16 = 0x0004;
pub const FLAG_ACCENT_MARK: u16 = 0x0010;
pub const FLAG_PUNCTUATION: u16 = 0x0020;
pub const FLAG_SYMBOL: u16 = 0x0040;
pub const FLAG_WHITESPACE: u16 = 0x0100;

pub const Flags = struct {
    raw: u16,

    pub inline fn isNumber(self: Flags) bool {
        return self.raw & FLAG_NUMBER != 0;
    }
    pub inline fn isLetter(self: Flags) bool {
        return self.raw & FLAG_LETTER != 0;
    }
    pub inline fn isAccentMark(self: Flags) bool {
        return self.raw & FLAG_ACCENT_MARK != 0;
    }
    pub inline fn isWhitespace(self: Flags) bool {
        return self.raw & FLAG_WHITESPACE != 0;
    }
    pub inline fn asUint(self: Flags) u16 {
        return self.raw;
    }
};

pub const OUT_OF_RANGE: u32 = 0xFFFFFFFF;
const MAX_CODEPOINTS: u32 = 0x110000;

/// Flag de un codepoint: binsearch en las tablas de rangos generadas.
pub fn flagsFromCpt(cpt: u32) Flags {
    if (cpt >= MAX_CODEPOINTS) return .{ .raw = FLAG_UNDEFINED };
    var lo: usize = 0;
    var hi: usize = unicode_data.ranges_flags.len / 2;
    while (lo + 1 < hi) {
        const mid = lo + (hi - lo) / 2;
        if (unicode_data.ranges_flags[2 * mid] <= cpt) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    var raw: u16 = @intCast(unicode_data.ranges_flags[2 * lo + 1]);
    if (isWhitespaceCpt(cpt)) raw |= FLAG_WHITESPACE;
    return .{ .raw = raw };
}

pub fn isWhitespaceCpt(cpt: u32) bool {
    var lo: usize = 0;
    var hi: usize = unicode_data.whitespace.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const v = unicode_data.whitespace[mid];
        if (v == cpt) return true;
        if (v < cpt) lo = mid + 1 else hi = mid;
    }
    return false;
}

/// tolower de un codepoint (tabla de llama.cpp); identidad si no está.
pub fn toLower(cpt: u32) u32 {
    var lo: usize = 0;
    var hi: usize = unicode_data.lowercase.len / 2;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const from = unicode_data.lowercase[2 * mid];
        if (from == cpt) return unicode_data.lowercase[2 * mid + 1];
        if (from < cpt) lo = mid + 1 else hi = mid;
    }
    return cpt;
}

/// Decodificar todo el texto a codepoints. Bytes inválidos → U+FFFD.
pub fn cptsFromUtf8(text: []const u8, allocator: std.mem.Allocator) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(allocator);
    var i: usize = 0;
    while (i < text.len) {
        const n = std.unicode.utf8ByteSequenceLength(text[i]) catch {
            try out.append(allocator, 0xFFFD);
            i += 1;
            continue;
        };
        if (i + n > text.len) {
            try out.append(allocator, 0xFFFD);
            i += 1;
            continue;
        }
        const cpt = std.unicode.utf8Decode(text[i .. i + n]) catch {
            try out.append(allocator, 0xFFFD);
            i += 1;
            continue;
        };
        try out.append(allocator, cpt);
        i += n;
    }
    return out.toOwnedSlice(allocator);
}

/// Codificar un codepoint a UTF-8.
pub fn cptToUtf8(cpt: u32, buf: []u8) []u8 {
    const n = std.unicode.utf8Encode(@intCast(cpt), buf) catch return &.{};
    return buf[0..n];
}

/// bytes_to_unicode de GPT-2: mapea cada byte a un codepoint.
/// Devuelve el codepoint (no la cadena) para el byte dado.
pub fn byteToUnicodeCpt(byte: u8) u32 {
    const c: u32 = byte;
    if ((c >= 0x21 and c <= 0x7E) or (c >= 0xA1 and c <= 0xAC) or (c >= 0xAE and c <= 0xFF)) return c;
    if (c <= 0x20) return 0x100 + c; // 0x00..0x20 → 0x100..0x120
    if (c >= 0x7F and c <= 0xA0) return 0x100 + (c - 0x5E); // 0x7F..0xA0 → 0x121..0x142
    // c == 0xAD
    return 0x143;
}

/// Devuelve el token (single-char) correspondiente al byte dado en la forma
/// byte-encodificada (id de vocabulario si el byte está en el vocab).
pub fn byteEncodedToken(byte: u8, buf: []u8) []u8 {
    const cpt = byteToUnicodeCpt(byte);
    return cptToUtf8(cpt, buf);
}

/// Inversa de bytes_to_unicode: codepoint → byte original. null si no aplica.
pub fn unicodeCptToByte(cpt: u32) ?u8 {
    if (cpt < 0x100) {
        if ((cpt >= 0x21 and cpt <= 0x7E) or (cpt >= 0xA1 and cpt <= 0xAC) or (cpt >= 0xAE and cpt <= 0xFF)) {
            return @intCast(cpt);
        }
        return null;
    }
    const n = cpt - 0x100;
    if (n <= 32) return @intCast(n); // 0x00..0x20
    if (n <= 66) return @intCast(0x7F + (n - 33)); // 0x7F..0xA0
    if (n == 67) return 0xAD;
    return null;
}

/// Pre-tokenización Qwen3.5 (unicode_regex_split_custom_qwen35 de llama.cpp).
/// Devuelve las palabras como slices de codepoints del texto original.
pub fn splitQwen35(cpts: []const u32, allocator: std.mem.Allocator) ![]const []const u32 {
    var words: std.ArrayList([]const u32) = .empty;
    errdefer words.deinit(allocator);

    const n = cpts.len;
    var prev_end: usize = 0;
    var pos: usize = 0;

    const addToken = struct {
        fn call(gpa: std.mem.Allocator, word_list: *std.ArrayList([]const u32), cpts2: []const u32, prev: *usize, end: usize) !void {
            const start = prev.*;
            if (start <= end and end <= cpts2.len) {
                const len = end - start;
                if (len > 0) try word_list.append(gpa, cpts2[start..end]);
                prev.* = end;
            }
        }
    }.call;

    while (pos < n) {
        const cpt = cpts[pos];
        const flags = flagsFromCpt(cpt);

        // regex: (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
        if (cpt == '\'' and pos + 1 < n) {
            const next = toLower(cpts[pos + 1]);
            if (next == 's' or next == 't' or next == 'm' or next == 'd') {
                try addToken(allocator, &words, cpts, &prev_end, pos + 2);
                pos = prev_end;
                continue;
            }
            if (pos + 2 < n) {
                const nnext = toLower(cpts[pos + 2]);
                if ((next == 'r' and nnext == 'e') or (next == 'v' and nnext == 'e') or (next == 'l' and nnext == 'l')) {
                    try addToken(allocator, &words, cpts, &prev_end, pos + 3);
                    pos = prev_end;
                    continue;
                }
            }
        }

        // regex: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if (!(cpt == '\r' or cpt == '\n' or flags.isNumber())) {
            const nxt_flags = if (pos + 1 < n) flagsFromCpt(cpts[pos + 1]) else Flags{ .raw = 0 };
            if (flags.isLetter() or flags.isAccentMark() or nxt_flags.isAccentMark() or nxt_flags.isLetter()) {
                pos += 1;
                while (pos < n) {
                    const f = flagsFromCpt(cpts[pos]);
                    if (!(f.isLetter() or f.isAccentMark())) break;
                    pos += 1;
                }
                try addToken(allocator, &words, cpts, &prev_end, pos);
                continue;
            }
        }

        // regex: \p{N}
        if (flags.isNumber()) {
            pos += 1;
            try addToken(allocator, &words, cpts, &prev_end, pos);
            continue;
        }

        // regex: <espacio>?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        var flags2 = if (cpt == ' ') (if (pos + 1 < n) flagsFromCpt(cpts[pos + 1]) else Flags{ .raw = 0 }) else flags;
        if (!(flags2.isWhitespace() or flags2.isLetter() or flags2.isAccentMark() or flags2.isNumber()) and flags.asUint() != 0) {
            pos += if (cpt == ' ') @as(usize, 1) else 0;
            while (pos < n) {
                flags2 = flagsFromCpt(cpts[pos]);
                if (flags2.isWhitespace() or flags2.isLetter() or flags2.isAccentMark() or flags2.isNumber() or flags2.asUint() == 0) break;
                pos += 1;
            }
            var cpt2: u32 = if (pos < n) cpts[pos] else OUT_OF_RANGE;
            while (cpt2 == '\r' or cpt2 == '\n') {
                pos += 1;
                cpt2 = if (pos < n) cpts[pos] else OUT_OF_RANGE;
            }
            try addToken(allocator, &words, cpts, &prev_end, pos);
            continue;
        }

        // \s*
        var num_ws: usize = 0;
        var last_rn: usize = 0;
        while (pos + num_ws < n) {
            const f = flagsFromCpt(cpts[pos + num_ws]);
            if (!f.isWhitespace()) break;
            const c2 = cpts[pos + num_ws];
            if (c2 == '\r' or c2 == '\n') last_rn = pos + num_ws + 1;
            num_ws += 1;
        }

        // regex: \s*[\r\n]+
        if (last_rn > 0) {
            pos = last_rn;
            try addToken(allocator, &words, cpts, &prev_end, pos);
            continue;
        }

        // regex: \s+(?!\S)
        if (num_ws > 1 and pos + num_ws < n) {
            pos += num_ws - 1;
            try addToken(allocator, &words, cpts, &prev_end, pos);
            continue;
        }

        // regex: \s+
        if (num_ws > 0) {
            pos += num_ws;
            try addToken(allocator, &words, cpts, &prev_end, pos);
            continue;
        }

        // no matches
        try addToken(allocator, &words, cpts, &prev_end, pos + 1);
        pos += 1;
    }

    return words.toOwnedSlice(allocator);
}

test "bytes_to_unicode ida y vuelta" {
    try std.testing.expectEqual(@as(u32, 0x61), byteToUnicodeCpt('a'));
    try std.testing.expectEqual(@as(u32, 0xA9), byteToUnicodeCpt(0xA9));
    try std.testing.expectEqual(@as(u32, 0xC3), byteToUnicodeCpt(0xC3));
    try std.testing.expectEqual(@as(u32, 0x120), byteToUnicodeCpt(0x20));
    try std.testing.expectEqual(@as(u32, 0x121), byteToUnicodeCpt(0x7F));
    try std.testing.expectEqual(@as(u32, 0x143), byteToUnicodeCpt(0xAD));
    try std.testing.expectEqual(@as(?u8, 0x61), unicodeCptToByte(0x61));
    try std.testing.expectEqual(@as(?u8, 0x20), unicodeCptToByte(0x120));
    try std.testing.expectEqual(@as(?u8, 0xAD), unicodeCptToByte(0x143));
    try std.testing.expectEqual(@as(?u8, null), unicodeCptToByte(0x0000));
}

test "splitQwen35 agrupa espacio+palabra" {
    const allocator = std.testing.allocator;
    const text = "qué tal";
    const cpts = try cptsFromUtf8(text, allocator);
    defer allocator.free(cpts);
    const words = try splitQwen35(cpts, allocator);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(usize, 2), words.len);
    try std.testing.expectEqual(@as(usize, 3), words[0].len);
    try std.testing.expectEqual(@as(u32, ' '), words[1][0]);
    try std.testing.expectEqual(@as(usize, 4), words[1].len);
}
