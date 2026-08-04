const std = @import("std");
const gguf_tokenizer = @import("gguf_tokenizer");

fn isPunct(c: u8) bool {
    return (c >= 0x21 and c <= 0x2F) or
        (c >= 0x3A and c <= 0x40) or
        (c >= 0x5B and c <= 0x60) or
        (c >= 0x7B and c <= 0x7E);
}

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
    vocab: std.StringHashMap(u32),           // token_str -> id
    vocab_inv: std.AutoHashMap(u32, []const u8),  // id -> token_str
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
        priority: u32,  // índice en el archivo de merges (menor = prioridad alta)
    };

    pub const EncodeOptions = struct {
        add_bos: bool = false,
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

    /// Pre-tokenización simple: divide por espacios y puntuación básica
    /// Para GPT-2 real se necesita regex más complejo, esto es suficiente para demo
    pub fn preTokenize(self: Self, text: []const u8, words: *std.ArrayList([]const u8)) !void {
        var start: usize = 0;
        var i: usize = 0;
        while (i < text.len) : (i += 1) {
            const c = text[i];
            if (std.ascii.isWhitespace(c) or isPunct(c)) {
                if (i > start) {
                    try words.append(self.allocator, text[start..i]);
                }
                if (isPunct(c)) {
                    try words.append(self.allocator, text[i..i+1]);
                }
                start = i + 1;
            }
        }
        if (start < text.len) {
            try words.append(self.allocator, text[start..]);
        }
    }

    /// Encode: texto -> tokens
    pub fn encode(self: *Self, text: []const u8, options: EncodeOptions) ![]u32 {
        var tokens: std.ArrayList(u32) = .empty;
        errdefer tokens.deinit(self.allocator);

        if (options.add_bos) {
            if (self.bos_token) |bos| try tokens.append(self.allocator, bos);
        }

        var words: std.ArrayList([]const u8) = .empty;
        defer words.deinit(self.allocator);
        try self.preTokenize(text, &words);

        for (words.items) |word| {
            try self.encodeWord(word, &tokens);
        }

        if (options.add_eos) {
            if (self.eos_token) |eos| try tokens.append(self.allocator, eos);
        }

        return tokens.toOwnedSlice(self.allocator);
    }

    fn encodeWord(self: *Self, word: []const u8, tokens: *std.ArrayList(u32)) !void {
        // Inicializar word como secuencia de bytes individuales
        var symbols: std.ArrayList([]const u8) = .empty;
        defer {
            for (symbols.items) |s| self.allocator.free(s);
            symbols.deinit(self.allocator);
        }

        for (word) |byte| {
            const sym = try self.allocator.dupe(u8, &[_]u8{byte});
            try symbols.append(self.allocator, sym);
        }

        if (symbols.items.len == 0) return;

        // Aplicar merges BPE
        while (true) {
            var best_merge: ?usize = null;
            var best_priority: u32 = std.math.maxInt(u32);

            for (0..symbols.items.len - 1) |i| {
                const left = symbols.items[i];
                const right = symbols.items[i + 1];

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

            var i: usize = 0;
            while (i < symbols.items.len) {
                if (i < symbols.items.len - 1 and 
                    std.mem.eql(u8, symbols.items[i], merge.left) and
                    std.mem.eql(u8, symbols.items[i + 1], merge.right)) {
                    const merged = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ symbols.items[i], symbols.items[i + 1] });
                    try new_symbols.append(self.allocator, merged);
                    i += 2;
                } else {
                    const copied = try self.allocator.dupe(u8, symbols.items[i]);
                    try new_symbols.append(self.allocator, copied);
                    i += 1;
                }
            }

            // Liberar symbols antiguos
            for (symbols.items) |s| self.allocator.free(s);
            symbols.deinit(self.allocator);
            symbols = new_symbols;
        }

        // Mapear símbolos finales a IDs
        for (symbols.items) |sym| {
            const id = self.vocab.get(sym) orelse self.unk_token;
            try tokens.append(self.allocator, id);
        }
    }

    /// Decode: tokens -> texto
    pub fn decode(self: Self, tokens: []const u32, allocator: std.mem.Allocator) ![]u8 {
        var result: std.ArrayList(u8) = .empty;
        errdefer result.deinit(allocator);

        for (tokens) |token| {
            if (self.vocab_inv.get(token)) |str| {
                try result.appendSlice(allocator, str);
            } else {
                try result.appendSlice(allocator, "<unk>");
            }
        }

        return result.toOwnedSlice(allocator);
    }

    /// Crear un tokenizer dummy para tests (vocab de bytes 0-255 + algunos merges)
    pub fn initDummy(allocator: std.mem.Allocator) !Self {
        var tok = Self.init(allocator);

        // Vocabulario: bytes 0-255 como tokens individuales
        for (0..256) |b| {
            const str = try std.fmt.allocPrint(allocator, "<0x{X:0>2}>", .{b});
            defer allocator.free(str);
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

test "bpe decode" {
    const allocator = std.testing.allocator;
    var tok = try BPETokenizer.initDummy(allocator);
    defer tok.deinit();

    const tokens = &[_]u32{ 0, 1, 2 };
    const decoded = try tok.decode(tokens, allocator);
    defer allocator.free(decoded);

    try std.testing.expect(decoded.len > 0);
}
