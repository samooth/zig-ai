//! Corpus loader para entrenamiento RLT — lee ficheros de texto, tokeniza BPE
//! y empaqueta en secuencias de longitud fija.
//!
//! v1 (synthetic): la clase Corpus NO se usa aún — el train CLI v1 entrena con
//! hidden states sintéticos. Este módulo es la pieza de la fase 2 (captura de
//! hidden states del modelo real), donde load() consumirá el BPE del GGUF
//! (GgufTokenizer.fromGguf → BPETokenizer.fromTokenizer, patrón de main.zig).
const std = @import("std");

pub const Corpus = struct {
    /// Tokens empaquetados: floor(total/seq_len) secuencias completas.
    tokens: []u32,
    /// Total de tokens ANTES de truncar al múltiplo de seq_len (diagnóstico).
    total_len: usize,
    seq_len: usize,
    allocator: std.mem.Allocator,

    /// Carga corpus desde ficheros de texto y tokeniza.
    /// `io`: instancia Io para I/O de ficheros (0.16: sin std.fs.cwd()).
    /// `tokenizer`: BPE inicializado desde el GGUF del modelo base.
    pub fn load(
        io: std.Io,
        allocator: std.mem.Allocator,
        files: []const []const u8,
        tokenizer: anytype, // *BPETokenizer — anytype evita dependencia cíclica de módulos
        seq_len: usize,
    ) !Corpus {
        var all_tokens: std.ArrayList(u32) = .empty;
        defer all_tokens.deinit(allocator);

        const dir = std.Io.Dir.cwd();
        for (files) |path| {
            const content = try dir.readFileAlloc(io, path, allocator, .limited(16 * 1024 * 1024));
            defer allocator.free(content);
            const toks = try tokenizer.encode(content, .{});
            defer allocator.free(toks);
            try all_tokens.appendSlice(allocator, toks);
        }

        // Empaquetar: solo secuencias completas de seq_len
        const n_seqs = all_tokens.items.len / seq_len;
        const packed_len = n_seqs * seq_len;
        const packed_tokens = try allocator.alloc(u32, packed_len);
        @memcpy(packed_tokens, all_tokens.items[0..packed_len]);

        return .{
            .tokens = packed_tokens,
            .total_len = all_tokens.items.len,
            .seq_len = seq_len,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Corpus) void {
        self.allocator.free(self.tokens);
    }

    /// Batch step: slice de [batch_size*seq_len] tokens desde el offset del step.
    pub fn getBatch(self: *const Corpus, step: usize, batch_size: usize) []const u32 {
        const offset = step * batch_size * self.seq_len;
        const end = @min(offset + batch_size * self.seq_len, self.tokens.len);
        if (offset >= end) return &.{};
        return self.tokens[offset..end];
    }

    pub fn numBatches(self: *const Corpus, batch_size: usize) usize {
        return self.tokens.len / (batch_size * self.seq_len);
    }
};
