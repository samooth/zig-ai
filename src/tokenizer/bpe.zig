const std = @import("std");
const gguf_tokenizer = @import("gguf_tokenizer");
const unicode = @import("unicode");

/// BPE Tokenizer — Implementación limpia del algoritmo Byte-Pair Encoding
/// Compatible con formatos tipo GPT-2 / Llama tokenizer.json simplificado
///
/// No depende de librerías externas. Soporta:
/// - Vocabulario con merges BPE
/// - Pre-tokenización por regex (GPT-2 style)
/// - Post-procesado: añadir BOS/EOS tokens
/// - Encoding/decoding con manejo de unk
pub const BPETokenizer = struct {
    allocator: std.mem.Allocator,
    /// "gpt2", "llama", "qwen2", ... (para pre-tokenización específica)
    model: []const u8,
    vocab: std.StringHashMap(u32), // token_str -> id
    vocab_inv: std.AutoHashMap(u32, []const u8), // id -> token_str
    merges: std.ArrayList(MergePair),
    unk_token: u32,
    bos_token: ?u32,
    eos_token: ?u32,
    pad_token: ?u32,
    add_bos: bool,
    add_eos: bool,

    const Self = @This();

    pub const MergePair = struct {
        left: []const u8,
        right: []const u8,
        priority: u32, // índice en el archivo de merges (menor = prioridad alta)
    };

    pub const EncodeOptions = struct {
        /// null (default) = usar la política del tokenizer (self.add_bos,
        /// espejo de llama-vocab.cpp: pre llama3-like ⇒ true). true/false
        /// explícitos la sobreescriben.
        add_bos: ?bool = null,
        add_eos: bool = false,
    };

    /// Inicializar tokenizer vacío
    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .model = "gpt2",
            .vocab = std.StringHashMap(u32).init(allocator),
            .vocab_inv = std.AutoHashMap(u32, []const u8).init(allocator),
            .merges = .empty,
            .unk_token = 0,
            .bos_token = null,
            .eos_token = null,
            .pad_token = null,
            .add_bos = false,
            .add_eos = false,
        };
    }

    /// Construir un tokenizer a partir de un tokenizer GGUF embebido (D2).
    /// Los tokens/merges se duplican (ownership propio).
    pub fn fromTokenizer(allocator: std.mem.Allocator, t: *const gguf_tokenizer.GgufTokenizer) !Self {
        var tok = Self.init(allocator);
        errdefer tok.deinit();
        tok.model = t.model;
        for (t.tokens, 0..) |token_str, i| {
            try tok.addToken(token_str, @intCast(i));
        }
        for (t.merges, 0..) |m, i| {
            try tok.addMerge(m.left, m.right, @intCast(i));
        }
        tok.bos_token = t.bos_id;
        tok.eos_token = t.eos_id;
        tok.unk_token = t.unk_id orelse 0;
        tok.pad_token = t.pad_id;
        tok.add_bos = t.add_bos;
        tok.add_eos = t.add_eos;
        return tok;
    }

    pub fn deinit(self: *Self) void {
        var vocab_iter = self.vocab.iterator();
        while (vocab_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.vocab.deinit();

        var inv_iter = self.vocab_inv.iterator();
        while (inv_iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.vocab_inv.deinit();

        for (self.merges.items) |merge| {
            self.allocator.free(merge.left);
            self.allocator.free(merge.right);
        }
        self.merges.deinit(self.allocator);
    }

    /// Añadir token al vocabulario
    pub fn addToken(self: *Self, token_str: []const u8, id: u32) !void {
        const owned_str = try self.allocator.dupe(u8, token_str);
        try self.vocab.put(owned_str, id);

        const owned_inv = try self.allocator.dupe(u8, token_str);
        try self.vocab_inv.put(id, owned_inv);
    }

    /// Añadir merge al ranking
    pub fn addMerge(self: *Self, left: []const u8, right: []const u8, priority: u32) !void {
        const owned_left = try self.allocator.dupe(u8, left);
        const owned_right = try self.allocator.dupe(u8, right);
        try self.merges.append(self.allocator, .{
            .left = owned_left,
            .right = owned_right,
            .priority = priority,
        });
    }

    /// Cargar vocabulario desde un archivo de texto (formato: token
    /// )
    pub fn loadVocabFromText(self: *Self, text: []const u8) !void {
        var lines = std.mem.splitScalar(u8, text, '\n');
        var id: u32 = 0;
        while (lines.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \r\t");
            if (trimmed.len == 0) continue;
            try self.addToken(trimmed, id);
            id += 1;
        }
    }

    /// Cargar merges desde archivo de texto (formato: left right
    /// )
    pub fn loadMergesFromText(self: *Self, text: []const u8) !void {
        var lines = std.mem.splitScalar(u8, text, '\n');
        var priority: u32 = 0;
        while (lines.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \r\t");
            if (trimmed.len == 0 or trimmed[0] == '#') continue;

            var parts = std.mem.splitScalar(u8, trimmed, ' ');
            const left = parts.next() orelse continue;
            const right = parts.next() orelse continue;
            try self.addMerge(left, right, priority);
            priority += 1;
        }
    }

    /// Encode: texto -> tokens (byte-level BPE estilo GPT-2/Qwen3.5).
    /// Replica la pre-tokenización Unicode y el byte-encoding de llama.cpp.
    pub fn encode(self: *Self, text: []const u8, options: EncodeOptions) ![]u32 {
        var tokens: std.ArrayList(u32) = .empty;
        errdefer tokens.deinit(self.allocator);

        // 8.3 BOS-parity: default = política del tokenizer (llama3-like ⇒
        // true; llama.cpp inserta BOS en tokenize() vía vocab.add_bos).
        if (options.add_bos orelse self.add_bos) {
            if (self.bos_token) |bos| try tokens.append(self.allocator, bos);
        }

        const cpts = try unicode.cptsFromUtf8(text, self.allocator);
        defer self.allocator.free(cpts);
        const words = try unicode.splitQwen35(cpts, self.allocator);
        defer self.allocator.free(words);

        for (words) |word| {
            try self.encodeWord(word, &tokens);
        }

        if (options.add_eos) {
            if (self.eos_token) |eos| try tokens.append(self.allocator, eos);
        }

        return tokens.toOwnedSlice(self.allocator);
    }

    fn encodeWord(self: *Self, word: []const u32, tokens: *std.ArrayList(u32)) !void {
        // 1) Byte-encoding GPT-2: el texto (codepoints) → bytes → unicode
        //    de cada byte (bytes_to_unicode), formando la palabra codificada.
        var encoded: std.ArrayList(u8) = .empty;
        defer encoded.deinit(self.allocator);
        var raw_buf: [4]u8 = undefined;
        var enc_buf: [4]u8 = undefined;
        for (word) |cpt| {
            const raw = unicode.cptToUtf8(cpt, &raw_buf);
            for (raw) |byte| {
                const enc = unicode.byteEncodedToken(byte, &enc_buf);
                try encoded.appendSlice(self.allocator, enc);
            }
        }

        // 2) Símbolos iniciales: cada char UTF-8 de la palabra codificada.
        var symbols: std.ArrayList([]const u8) = .empty;
        defer {
            for (symbols.items) |s| self.allocator.free(s);
            symbols.deinit(self.allocator);
        }

        var i: usize = 0;
        while (i < encoded.items.len) {
            const n = try std.unicode.utf8ByteSequenceLength(encoded.items[i]);
            const sym = try self.allocator.dupe(u8, encoded.items[i .. i + n]);
            try symbols.append(self.allocator, sym);
            i += n;
        }

        if (symbols.items.len == 0) return;

        // Aplicar merges BPE
        while (true) {
            var best_merge: ?usize = null;
            var best_priority: u32 = std.math.maxInt(u32);

            for (0..symbols.items.len - 1) |idx| {
                const left = symbols.items[idx];
                const right = symbols.items[idx + 1];

                for (self.merges.items, 0..) |merge, mi| {
                    if (std.mem.eql(u8, merge.left, left) and std.mem.eql(u8, merge.right, right)) {
                        if (merge.priority < best_priority) {
                            best_priority = merge.priority;
                            best_merge = mi;
                        }
                        break;
                    }
                }
            }

            if (best_merge == null) break;

            // Aplicar el mejor merge
            const merge = self.merges.items[best_merge.?];
            var new_symbols: std.ArrayList([]const u8) = .empty;
            errdefer {
                for (new_symbols.items) |s| self.allocator.free(s);
                new_symbols.deinit(self.allocator);
            }

            var ii: usize = 0;
            while (ii < symbols.items.len) {
                if (ii < symbols.items.len - 1 and
                    std.mem.eql(u8, symbols.items[ii], merge.left) and
                    std.mem.eql(u8, symbols.items[ii + 1], merge.right))
                {
                    const merged = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ symbols.items[ii], symbols.items[ii + 1] });
                    try new_symbols.append(self.allocator, merged);
                    ii += 2;
                } else {
                    const copied = try self.allocator.dupe(u8, symbols.items[ii]);
                    try new_symbols.append(self.allocator, copied);
                    ii += 1;
                }
            }

            // Liberar symbols antiguos
            for (symbols.items) |s| self.allocator.free(s);
            symbols.deinit(self.allocator);
            symbols = new_symbols;
        }

        // Mapear símbolos finales a IDs (con fallback por byte, como llama.cpp)
        for (symbols.items) |sym| {
            if (self.vocab.get(sym)) |id| {
                try tokens.append(self.allocator, id);
            } else {
                var tmp: [4]u8 = undefined;
                for (sym) |byte| {
                    const tok_str = unicode.byteEncodedToken(byte, &tmp);
                    if (self.vocab.get(tok_str)) |bid| try tokens.append(self.allocator, bid);
                }
            }
        }
    }

    /// Decode: tokens -> texto (invierte bytes_to_unicode)
    pub fn decode(self: Self, tokens: []const u32, allocator: std.mem.Allocator) ![]u8 {
        var result: std.ArrayList(u8) = .empty;
        errdefer result.deinit(allocator);

        for (tokens) |token| {
            if (self.vocab_inv.get(token)) |str| {
                var i: usize = 0;
                while (i < str.len) {
                    const n = std.unicode.utf8ByteSequenceLength(str[i]) catch {
                        i += 1;
                        continue;
                    };
                    if (i + n > str.len) break;
                    const cpt = std.unicode.utf8Decode(str[i .. i + n]) catch {
                        i += n;
                        continue;
                    };
                    if (unicode.unicodeCptToByte(cpt)) |byte| {
                        try result.append(allocator, byte);
                    }
                    i += n;
                }
            } else {
                try result.appendSlice(allocator, "<unk>");
            }
        }

        return result.toOwnedSlice(allocator);
    }

    /// Decode un solo token a un buffer stack (sin allocator).
    /// Escribe los bytes decodificados en `buf` y devuelve la longitud escrita.
    /// Devuelve `error.BufferTooSmall` si el token no cabe en `buf`.
    pub fn decodeOne(self: Self, token: u32, buf: []u8) !usize {
        const str = self.vocab_inv.get(token) orelse {
            if (buf.len < 5) return error.BufferTooSmall;
            @memcpy(buf[0..5], "<unk>");
            return 5;
        };

        var out_i: usize = 0;
        var i: usize = 0;
        while (i < str.len) {
            const n = std.unicode.utf8ByteSequenceLength(str[i]) catch {
                i += 1;
                continue;
            };
            if (i + n > str.len) break;
            const cpt = std.unicode.utf8Decode(str[i .. i + n]) catch {
                i += n;
                continue;
            };
            if (unicode.unicodeCptToByte(cpt)) |byte| {
                if (out_i >= buf.len) return error.BufferTooSmall;
                buf[out_i] = byte;
                out_i += 1;
            }
            i += n;
        }
        return out_i;
    }

    /// Crear un tokenizer dummy para tests (vocab de bytes 0-255 + algunos merges)
    pub fn initDummy(allocator: std.mem.Allocator) !Self {
        var tok = Self.init(allocator);

        // Vocabulario: bytes 0-255 como tokens individuales byte-encodificados
        for (0..256) |b| {
            var buf: [4]u8 = undefined;
            const str = unicode.byteEncodedToken(@as(u8, @intCast(b)), &buf);
            try tok.addToken(str, @as(u32, @intCast(b)));
        }

        tok.unk_token = 0;
        tok.bos_token = 256;
        tok.eos_token = 257;
        try tok.addToken("<bos>", 256);
        try tok.addToken("<eos>", 257);

        return tok;
    }
};

// ─── Tests ───

test "bpe init and encode" {
    const allocator = std.testing.allocator;
    var tok = try BPETokenizer.initDummy(allocator);
    defer tok.deinit();

    const text = "ab";
    const tokens = try tok.encode(text, .{});
    defer allocator.free(tokens);

    try std.testing.expect(tokens.len > 0);
}

// 8.3 BOS-parity: encode() default usa la política del tokenizer
// (self.add_bos) — espejo de llama.cpp vocab.add_bos (pre llama3-like).
test "bpe encode bos policy" {
    const allocator = std.testing.allocator;
    var tok = try BPETokenizer.initDummy(allocator);
    defer tok.deinit();

    const text = "ab";

    // Política ON con bos_token: primera posición = BOS.
    tok.add_bos = true;
    tok.bos_token = 7;
    {
        const tokens = try tok.encode(text, .{});
        defer allocator.free(tokens);
        try std.testing.expectEqual(@as(u32, 7), tokens[0]);
    }

    // Override explícito false gana sobre la política.
    {
        const tokens = try tok.encode(text, .{ .add_bos = false });
        defer allocator.free(tokens);
        try std.testing.expect(tokens[0] != 7);
    }
}

test "bpe decode" {
    const allocator = std.testing.allocator;
    var tok = try BPETokenizer.initDummy(allocator);
    defer tok.deinit();

    const tokens = &[_]u32{ 0, 1, 2 };
    const decoded = try tok.decode(tokens, allocator);
    defer allocator.free(decoded);

    try std.testing.expect(decoded.len > 0);
}

test "bpe decodeOne parity vs decode (single token)" {
    const allocator = std.testing.allocator;
    var tok = try BPETokenizer.initDummy(allocator);
    defer tok.deinit();

    // Probar cada token del vocab dummy (bytes 0-255 + BOS/EOS + unk)
    const test_ids = &[_]u32{0, 1, 42, 127, 128, 255, 256, 257, 999};
    for (test_ids) |id| {
        var buf: [256]u8 = undefined;
        const len = tok.decodeOne(id, &buf) catch 0;
        const piece = buf[0..len];

        // Comparar contra decode() de un solo token
        const full = tok.decode(&[_]u32{id}, allocator) catch "";
        defer allocator.free(full);

        try std.testing.expectEqualSlices(u8, full, piece);
    }
}
