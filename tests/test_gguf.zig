//! Test C8 — cargar un GGUF real (vía mmap, C4) y verificar metadata,
//! ModelConfig y shapes de tensores.
//! Requiere la variable de entorno `GGUF_MODEL_PATH` apuntando a un .gguf;
//! si no está definida, el test se salta (error.SkipZigTest).
const std = @import("std");
const gguf = @import("gguf");
const gguf_tokenizer = @import("gguf_tokenizer");
const model_config = @import("model_config");
const bpe = @import("tokenizer");

test "load real gguf and verify config + tensor shapes" {
    const gpa = std.testing.allocator;

    const env_path = std.c.getenv("GGUF_MODEL_PATH") orelse {
        std.debug.print("SKIP: GGUF_MODEL_PATH no está definida\n", .{});
        return error.SkipZigTest;
    };
    const path = std.mem.span(env_path);

    const io = std.Io.Threaded.global_single_threaded.io();

    var g = try gguf.GgufFile.fromFileMmap(io, gpa, path);
    defer g.deinit();

    // Header
    try std.testing.expectEqual(@as(u32, 3), g.version);
    const arch = g.arch() orelse return error.MissingArchitecture;
    std.debug.print("arch={s} alignment={d} tensors={d}\n", .{
        arch, g.alignment, g.tensors.count(),
    });

    // Config
    const cfg = try model_config.ModelConfig.fromGguf(&g);
    std.debug.print("embedding={d} layers={d} heads={d} kv_heads={d} ffn={d} vocab={d} ctx={d} rope_theta={d:.0}\n", .{
        cfg.embedding_length,
        cfg.block_count,
        cfg.head_count,
        cfg.head_count_kv,
        cfg.feed_forward_length,
        cfg.vocab_size,
        cfg.context_length,
        cfg.rope_freq_base,
    });
    try std.testing.expect(cfg.embedding_length >= 256);
    try std.testing.expect(cfg.block_count >= 1);
    try std.testing.expect(cfg.head_count >= 1);
    try std.testing.expect(cfg.head_count_kv >= 1 and cfg.head_count_kv <= cfg.head_count);
    try std.testing.expect(cfg.feed_forward_length >= cfg.embedding_length);
    try std.testing.expect(cfg.vocab_size >= 1000);
    try std.testing.expect(cfg.context_length >= 64);
    try std.testing.expect(model_config.ModelConfig.isSupportedArch(arch));

    // Offset de tensor_data alineado
    try std.testing.expect(g.tensor_data_offset % g.alignment == 0);

    // Tensores clave deben existir (naming arch-dependiente: llama usa
    // attn_o, qwen usa attn_output; la norm final puede llamarse
    // output_norm o token_embd_norm, o no existir).
    const attn_out = if (g.getTensor("blk.0.attn_o.weight") != null)
        "blk.0.attn_o.weight"
    else
        "blk.0.attn_output.weight";
    const required = [_][]const u8{
        "token_embd.weight",
        attn_out,
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "blk.0.attn_norm.weight",
        "blk.0.ffn_norm.weight",
    };
    for (required) |name| {
        const t = g.getTensor(name) orelse {
            std.debug.print("FALTA tensor: {s}\n", .{name});
            return error.MissingTensor;
        };
        // shape debe coincidir con la config
        if (gguf.parseTensorName(name).role == .attn_q) {
            try std.testing.expectEqual(@as(u64, cfg.embedding_length), t.dims[1]);
        }
        if (gguf.parseTensorName(name).role == .token_embd) {
            try std.testing.expectEqual(@as(u64, cfg.vocab_size), t.dims[1]);
        }
    }

    // Norm final opcional
    const final_norm = g.getTensor("output_norm.weight") orelse
        g.getTensor("token_embd_norm.weight") orelse null;
    if (final_norm) |t| {
        try std.testing.expectEqual(@as(u64, cfg.embedding_length), t.dims[0]);
    }

    // Última capa
    const last = cfg.block_count - 1;
    var buf: [64]u8 = undefined;
    const last_ffn = std.fmt.bufPrint(&buf, "blk.{d}.ffn_down.weight", .{last}) catch unreachable;
    try std.testing.expect(g.getTensor(last_ffn) != null);

    // Datos de tensor accesibles (no vacíos) para token_embd
    const embd = g.getTensor("token_embd.weight").?;
    const embd_bytes = g.tensorData(embd);
    try std.testing.expect(embd_bytes.len > 0);
    std.debug.print("token_embd dtype={s} shape=[{d} {d}] bytes={d}\n", .{
        embd.dtype.name(), embd.dims[0], embd.dims[1], embd_bytes.len,
    });
}

test "load real gguf tokenizer and build bpe (D1/D2/D4)" {
    const gpa = std.testing.allocator;

    const env_path = std.c.getenv("GGUF_MODEL_PATH") orelse {
        std.debug.print("SKIP: GGUF_MODEL_PATH no está definida\n", .{});
        return error.SkipZigTest;
    };
    const path = std.mem.span(env_path);

    const io = std.Io.Threaded.global_single_threaded.io();

    var g = try gguf.GgufFile.fromFileMmap(io, gpa, path);
    defer g.deinit();

    var gt = try gguf_tokenizer.GgufTokenizer.fromGguf(gpa, &g);
    defer gt.deinit();

    std.debug.print("tokenizer model={s} pre={s} tokens={d} merges={d}\n", .{
        gt.model, gt.pre, gt.tokens.len, gt.merges.len,
    });
    try std.testing.expect(gt.tokens.len >= 1000);
    try std.testing.expect(gt.merges.len > 0);
    try std.testing.expect(gt.bos_id != null);
    try std.testing.expect(gt.eos_id != null);

    // Construir el tokenizer BPE a partir del GGUF
    var tok = try bpe.BPETokenizer.fromTokenizer(gpa, &gt);
    defer tok.deinit();

    try std.testing.expectEqualStrings(gt.model, tok.model);
    try std.testing.expectEqual(gt.tokens.len, tok.vocab.count());

    // Encoder produce ids válidos
    const ids = try tok.encode("Hello, world! This is a Zig test.", .{});
    defer gpa.free(ids);
    std.debug.print("encode -> {d} tokens: ", .{ids.len});
    for (ids) |id| {
        const s = tok.vocab_inv.get(id);
        std.debug.print("[{d}:'{s}'] ", .{ id, if (s) |x| x else "<unk>" });
    }
    std.debug.print("\n", .{});
    try std.testing.expect(ids.len > 0);
    for (ids) |id| {
        try std.testing.expect(tok.vocab_inv.get(id) != null);
    }

    // Special tokens
    try std.testing.expectEqual(gt.bos_id, tok.bos_token);
    try std.testing.expectEqual(gt.eos_id, tok.eos_token);
}
