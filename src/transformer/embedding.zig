const std = @import("std");
const Tensor = @import("core").Tensor;
const matmul = @import("matmul");

/// Embedding lookup: convierte tokens [batch, seq] -> embeddings [batch, seq, hidden_dim]
pub fn embeddingLookup(
    embedding_table: Tensor(f16),  // [vocab_size, hidden_dim]
    tokens: []const u32,             // tokens planos
    batch_size: usize,
    seq_len: usize,
    output: *Tensor(f16),          // [batch, seq, hidden_dim]
) void {
    std.debug.assert(tokens.len == batch_size * seq_len);
    std.debug.assert(output.shape.len == 3);
    std.debug.assert(output.shape[0] == batch_size);
    std.debug.assert(output.shape[1] == seq_len);

    const hidden_dim = embedding_table.shape[1];
    std.debug.assert(output.shape[2] == hidden_dim);

    for (0..batch_size) |b| {
        for (0..seq_len) |s| {
            const token = tokens[b * seq_len + s];
            const token_idx = @min(token, @as(u32, @intCast(embedding_table.shape[0] - 1)));
            const out_offset = (b * seq_len + s) * hidden_dim;
            const emb_offset = token_idx * hidden_dim;
            @memcpy(output.data[out_offset..out_offset + hidden_dim], 
                    embedding_table.data[emb_offset..emb_offset + hidden_dim]);
        }
    }
}

/// LM Head: proyecta hidden [batch, seq, hidden_dim] -> logits [batch, seq, vocab_size]
pub fn lmHeadForward(
    engine: *matmul.MatmulEngine,
    hidden: Tensor(f16),       // [batch*seq, hidden_dim]
    head_weight_t: Tensor(f16), // [vocab_size, hidden_dim] transpuesto
    logits: *Tensor(f16),      // [batch*seq, vocab_size]
) !void {
    try engine.linearProjection(f16, hidden, head_weight_t, logits);
}

/// Aplica scaling a embeddings (sqrt(hidden_dim) para algunos modelos)
pub fn scaleEmbeddings(comptime T: type, embeddings: *Tensor(T), hidden_dim: usize) void {
    const scale = @sqrt(@as(f32, @floatFromInt(hidden_dim)));
    for (embeddings.data) |*p| {
        p.* = @as(T, @floatCast(@as(f32, @floatCast(p.*)) * scale));
    }
}

// ─── Tests ───

test "embedding lookup" {
    const allocator = std.testing.allocator;
    const vocab_size: usize = 100;
    const hidden_dim: usize = 16;
    const batch_size: usize = 2;
    const seq_len: usize = 3;

    var emb_table = try Tensor(f16).alloc(allocator, &[_]usize{ vocab_size, hidden_dim });
    defer emb_table.deinit();
    for (emb_table.data, 0..) |*p, i| {
        p.* = @floatCast(@as(f32, @floatFromInt(i)) * 0.01);
    }

    var tokens = [_]u32{ 0, 5, 10, 20, 30, 40 };
    var output = try Tensor(f16).alloc(allocator, &[_]usize{ batch_size, seq_len, hidden_dim });
    defer output.deinit();

    embeddingLookup(emb_table, &tokens, batch_size, seq_len, &output);

    // Verificar que el token 0 mapea a la fila 0 de emb_table
    for (0..hidden_dim) |d| {
        const expected = emb_table.data[d];
        try std.testing.expectApproxEqAbs(expected, output.data[d], 1e-4);
    }
}
