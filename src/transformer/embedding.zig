const std = @import("std");
const Tensor = @import("core").Tensor;
const matmul = @import("matmul");
const QuantWeight = @import("quant_weight").QuantWeight;
const gguf = @import("gguf");

/// Embedding lookup: convierte tokens [batch, seq] -> embeddings [batch, seq, hidden_dim]
pub fn embeddingLookup(
    embedding_table: Tensor(f16), // [vocab_size, hidden_dim]
    tokens: []const u32, // tokens planos
    batch_size: usize,
    seq_len: usize,
    output: *Tensor(f16), // [batch, seq, hidden_dim]
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
            @memcpy(output.data[out_offset .. out_offset + hidden_dim], embedding_table.data[emb_offset .. emb_offset + hidden_dim]);
        }
    }
}

/// Embedding lookup f16→f32: mismo índice que embeddingLookup pero vuelca
/// al stream f32 del pipeline (F-4 lane-f: el residual stream legacy pasa
/// a f32 — la tabla sigue f16, el cast es solo en el borde de entrada).
pub fn embeddingLookupF32(
    embedding_table: Tensor(f16), // [vocab_size, hidden_dim]
    tokens: []const u32,
    batch_size: usize,
    seq_len: usize,
    output: *Tensor(f32), // [batch, seq, hidden_dim]
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
            for (0..hidden_dim) |d| {
                output.data[out_offset + d] = @as(f32, @floatCast(embedding_table.data[emb_offset + d]));
            }
        }
    }
}

/// LM Head: proyecta hidden [batch, seq, hidden_dim] -> logits [batch, seq, vocab_size]
pub fn lmHeadForward(
    engine: *matmul.MatmulEngine,
    hidden: Tensor(f16), // [batch*seq, hidden_dim]
    head_weight_t: Tensor(f16), // [vocab_size, hidden_dim] transpuesto
    logits: *Tensor(f16), // [batch*seq, vocab_size]
) !void {
    try engine.linearProjection(f16, hidden, head_weight_t, logits);
}

/// Fuente de embeddings para el pipeline (7.1d wiring): tabla f16
/// materializada clásica O QuantWeight cuant-residente (dequant on-demand
/// solo de las filas pedidas — 0 bytes de tabla residente). Ambos caminos
/// son paridad bit-exacta (tests/test_embedding_quant.zig).
pub const EmbSource = union(enum) {
    table: Tensor(f16),
    quant: *const QuantWeight,

    pub fn lookup(self: EmbSource, tokens: []const u32, batch_size: usize, seq_len: usize, output: *Tensor(f16)) void {
        switch (self) {
            .table => |t| embeddingLookup(t, tokens, batch_size, seq_len, output),
            .quant => |qw| embeddingLookupQuant(qw, tokens, batch_size, seq_len, output),
        }
    }

    /// F-4 (lane-f): variante f32 del stream — mismo enrutado, salida al
    /// residual stream f32 (tabla/quant siguen f16; cast en el borde).
    pub fn lookupF32(self: EmbSource, tokens: []const u32, batch_size: usize, seq_len: usize, output: *Tensor(f32)) void {
        switch (self) {
            .table => |t| embeddingLookupF32(t, tokens, batch_size, seq_len, output),
            .quant => |qw| embeddingLookupQuantF32(qw, tokens, batch_size, seq_len, output),
        }
    }
};

/// Fuente del lm_head para el pipeline (7.1d, lane-c): tabla f16 densa
/// clásica (round-trip f16→f32→gemm→f16 vía linearProjection) O bytes
/// q8_0 on-load (loadLmHeadQ80 — layout GGUF [d f16][i8×32], 34B/bloque,
/// consumo directo sin materializar el f16 de 0.79GB) O QuantWeight mmap
/// nativo (lm_head ya cuantizado en el GGUF — q4_0/q6_k/...).
/// `lmHeadForwardSource` enruta; M=1 va por GEMV directo (sin round-trip
/// f32 del peso completo), M>1 por GEMM de filas dequantizadas.
pub const LmHeadSource = union(enum) {
    table: Tensor(f16),
    /// bytes q8_0 on-load [vocab][kb*34], kb = hidden/32.
    q80: []const u8,
    /// QuantWeight mmap del tensor output.weight nativo del GGUF.
    quant: *const QuantWeight,

    pub fn vocabSize(self: LmHeadSource) usize {
        return switch (self) {
            .table => |t| t.shape[0],
            .q80 => |b| b.len / 34, // sólo válido si hidden==32; usar con dims explícitas
            .quant => |qw| @intCast(qw.info.dims[1]),
        };
    }
};

/// GEMV q8_0 directo (7.1d, lane-c): logits[j] = dot(row_j_q80, x) con
/// dequant de bloque inline — CERO materialización f16 del peso y cero
/// round-trip f32 del tensor completo (el camino f16 convertía el peso
/// entero a f32 para el gemm host). `bytes` layout GGUF por fila:
/// kb bloques de [d f16 @0][i8×32 @2]. Paridad con cpuLmHeadLogits de la
/// tabla f16: delta ≤ cuantización q8 del peso (rel ~1e-3 por término).
pub fn lmHeadGemvQ80(x: []const f16, bytes: []const u8, vocab: usize, hidden: usize, out: []f32) void {
    const kb = hidden / 32;
    const row_out: usize = kb * 34;
    for (0..vocab) |j| {
        const row = bytes[j * row_out ..][0..row_out];
        var acc: f32 = 0;
        for (0..kb) |bi| {
            const blk = row[bi * 34 ..][0..34];
            const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, blk[0..2], .little))));
            const qs = blk[2..34];
            const xb = x[bi * 32 ..][0..32];
            for (0..32) |c| {
                const q: i8 = @bitCast(qs[c]);
                acc += d * @as(f32, @floatFromInt(q)) * @as(f32, @floatCast(xb[c]));
            }
        }
        out[j] = acc;
    }
}

/// GEMV desde QuantWeight nativo (lm_head ya cuantizado en el GGUF —
/// q4_0/q6_k/q8_0/...): dequant de bloque inline por fila, mismo contrato
/// que lmHeadGemvQ80 pero leyendo el formato del tensor.
pub fn lmHeadGemvQuant(x: []const f16, qw: *const QuantWeight, out: []f32) void {
    const info = qw.info;
    const hidden: usize = @intCast(info.dims[0]);
    const vocab: usize = @intCast(info.dims[1]);
    const bs = info.dtype.blockSize();
    const bb = info.dtype.blockBytes();
    const kb = (hidden + bs - 1) / bs;
    var tmp: [256]f32 = undefined;
    for (0..vocab) |j| {
        const row = qw.bytes[j * kb * bb ..][0 .. kb * bb];
        var acc: f32 = 0;
        var dst: usize = 0;
        for (0..kb) |bi| {
            const n = @min(bs, hidden - bi * bs);
            gguf.dequantBlock(info.dtype, row[bi * bb ..][0..bb], &tmp, n);
            for (0..n) |c| {
                acc += tmp[c] * @as(f32, @floatCast(x[dst + c]));
            }
            dst += n;
        }
        out[j] = acc;
    }
}

/// LM Head vía LmHeadSource (7.1d, lane-c): enruta M=1 a GEMV directo
/// (bytes q8_0/QuantWeight, logits f32) o M>1 a GEMM de la tabla f16
/// (camino clásico — el batch del prefill amortiza el round-trip).
/// `hidden_2d` [M, hidden] f16; `logits` [M, vocab] f32 de salida.
pub fn lmHeadForwardSource(
    engine: *matmul.MatmulEngine,
    hidden_2d: Tensor(f16),
    source: LmHeadSource,
    hidden_dim: usize,
    vocab_size: usize,
    logits_f32: *Tensor(f32),
) !void {
    const M = hidden_2d.shape[0];
    if (M == 1) {
        std.debug.assert(logits_f32.shape.len == 2 and logits_f32.shape[0] == 1 and logits_f32.shape[1] == vocab_size);
        switch (source) {
            .q80 => |bytes| lmHeadGemvQ80(hidden_2d.data[0..hidden_dim], bytes, vocab_size, hidden_dim, logits_f32.data[0..vocab_size]),
            .quant => |qw| lmHeadGemvQuant(hidden_2d.data[0..hidden_dim], qw, logits_f32.data[0..vocab_size]),
            .table => |t| {
                var logits16 = try Tensor(f16).alloc(engine.allocator, &.{ 1, vocab_size });
                defer logits16.deinit();
                try lmHeadForward(engine, hidden_2d, t, &logits16);
                for (logits16.data, logits_f32.data) |s, *d| d.* = @floatCast(s);
            },
        }
        return;
    }
    // M>1: camino clásico por filas sobre la tabla f16 (o dequant por filas
    // de la fuente cuant — 1 fila a la vez, M pequeñas del prefill).
    var row = try Tensor(f16).alloc(engine.allocator, &.{ 1, hidden_dim });
    defer row.deinit();
    var row_logits = try Tensor(f16).alloc(engine.allocator, &.{ 1, vocab_size });
    defer row_logits.deinit();
    for (0..M) |m| {
        @memcpy(row.data, hidden_2d.data[m * hidden_dim ..][0..hidden_dim]);
        switch (source) {
            .q80 => |bytes| lmHeadGemvQ80(row.data, bytes, vocab_size, hidden_dim, logits_f32.data[m * vocab_size ..][0..vocab_size]),
            .quant => |qw| lmHeadGemvQuant(row.data, qw, logits_f32.data[m * vocab_size ..][0..vocab_size]),
            .table => |t| {
                try lmHeadForward(engine, row, t, &row_logits);
                for (row_logits.data, 0..) |s, i| logits_f32.data[m * vocab_size + i] = @floatCast(s);
            },
        }
    }
}

/// F-4 (lane-f): variante f32 del lm_head sobre el stream f32 — cast
/// f32→f16 SOLO de la(s) fila(s) de entrada (1 redondeo final, los GEMV
/// q80/quant y el GEMM table siguen bit-idénticos al camino f16).
pub fn lmHeadForwardSourceF32(
    engine: *matmul.MatmulEngine,
    hidden_2d: Tensor(f32),
    source: LmHeadSource,
    hidden_dim: usize,
    vocab_size: usize,
    logits_f32: *Tensor(f32),
) !void {
    var hidden16 = try Tensor(f16).alloc(engine.allocator, hidden_2d.shape);
    defer hidden16.deinit();
    for (hidden_2d.data, hidden16.data) |s, *d| d.* = @floatCast(s);
    try lmHeadForwardSource(engine, hidden16, source, hidden_dim, vocab_size, logits_f32);
}

/// Embedding lookup cuant-residente (7.1d): dequantiza SOLO las filas de
/// los tokens pedidos desde el QuantWeight mmap (bytes del GGUF, cero
/// materialización de la tabla completa — la tabla f16 [vocab, hidden]
/// costaba p.ej. 0.79GB en Llama-3.2-3B, este camino 0 bytes residuales).
///
/// Layout GGUF del token_embd.weight: dims[0]=hidden (contiguo), dims[1]=vocab.
/// La fila del token v son los elementos planos [v*hidden, (v+1)*hidden) —
/// contiguos, por lo que toca ceil(hidden/bs) bloques cuant consecutivos
/// empezando en el bloque v*hidden/bs (hidden es múltiplo de bs en GGUF).
/// Equivalente bit-exacto a `embeddingLookup(loadEmbedding(), ...)`.
pub fn embeddingLookupQuant(
    qw: *const QuantWeight,
    tokens: []const u32,
    batch_size: usize,
    seq_len: usize,
    output: *Tensor(f16), // [batch, seq, hidden_dim]
) void {
    const info = qw.info;
    const hidden: usize = @intCast(info.dims[0]);
    const vocab: usize = @intCast(info.dims[1]);
    const bs = info.dtype.blockSize();
    const bb = info.dtype.blockBytes();

    std.debug.assert(tokens.len == batch_size * seq_len);
    std.debug.assert(output.shape.len == 3);
    std.debug.assert(output.shape[0] == batch_size);
    std.debug.assert(output.shape[1] == seq_len);
    std.debug.assert(output.shape[2] == hidden);

    var tmp: [256]f32 = undefined;
    for (0..batch_size) |b| {
        for (0..seq_len) |s| {
            const token = tokens[b * seq_len + s];
            const token_idx = @min(token, @as(u32, @intCast(vocab - 1)));
            const out_offset = (b * seq_len + s) * hidden;
            // Bloques [first, last] que cubren la fila (contiguos en plano).
            const first_elem = @as(usize, token_idx) * hidden;
            const first_block = first_elem / bs;
            const last_block = (first_elem + hidden - 1) / bs;
            var dst: usize = 0;
            for (first_block..last_block + 1) |blk| {
                const block_start = blk * bs;
                const n = @min(bs, @as(usize, @intCast(info.numel())) - block_start);
                gguf.dequantBlock(info.dtype, qw.bytes[blk * bb .. blk * bb + bb], &tmp, n);
                for (0..n) |j| {
                    const elem = block_start + j;
                    if (elem >= first_elem and elem < first_elem + hidden) {
                        output.data[out_offset + dst] = @floatCast(tmp[j]);
                        dst += 1;
                    }
                }
            }
        }
    }
}

/// F-4 (lane-f): variante f32 del lookup cuant-residente — dequant a f32
/// directo (sin el round-trip f16): misma geometría, cast final al stream.
pub fn embeddingLookupQuantF32(
    qw: *const QuantWeight,
    tokens: []const u32,
    batch_size: usize,
    seq_len: usize,
    output: *Tensor(f32), // [batch, seq, hidden_dim]
) void {
    const info = qw.info;
    const hidden: usize = @intCast(info.dims[0]);
    const vocab: usize = @intCast(info.dims[1]);
    const bs = info.dtype.blockSize();
    const bb = info.dtype.blockBytes();

    std.debug.assert(tokens.len == batch_size * seq_len);
    std.debug.assert(output.shape.len == 3);
    std.debug.assert(output.shape[0] == batch_size);
    std.debug.assert(output.shape[1] == seq_len);
    std.debug.assert(output.shape[2] == hidden);

    var tmp: [256]f32 = undefined;
    for (0..batch_size) |b| {
        for (0..seq_len) |s| {
            const token = tokens[b * seq_len + s];
            const token_idx = @min(token, @as(u32, @intCast(vocab - 1)));
            const out_offset = (b * seq_len + s) * hidden;
            // Bloques [first, last] que cubren la fila (contiguos en plano).
            const first_elem = @as(usize, token_idx) * hidden;
            const first_block = first_elem / bs;
            const last_block = (first_elem + hidden - 1) / bs;
            var dst: usize = 0;
            for (first_block..last_block + 1) |blk| {
                const block_start = blk * bs;
                const n = @min(bs, @as(usize, @intCast(info.numel())) - block_start);
                gguf.dequantBlock(info.dtype, qw.bytes[blk * bb .. blk * bb + bb], &tmp, n);
                for (0..n) |j| {
                    const elem = block_start + j;
                    if (elem >= first_elem and elem < first_elem + hidden) {
                        output.data[out_offset + dst] = tmp[j];
                        dst += 1;
                    }
                }
            }
        }
    }
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
