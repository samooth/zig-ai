//! Test C8 — cargar un GGUF real (vía mmap, C4) y verificar metadata,
//! ModelConfig y shapes de tensores.
//! Requiere la variable de entorno `GGUF_MODEL_PATH` apuntando a un .gguf;
//! si no está definida, el test se salta (error.SkipZigTest).
const std = @import("std");
const gguf = @import("gguf");
const gguf_tokenizer = @import("gguf_tokenizer");
const model_config = @import("model_config");
const gguf_model = @import("gguf_model");
const bpe = @import("tokenizer");
const transformer = @import("transformer");
const Tensor = @import("core").Tensor;
const fa = @import("fa");

const TransformerLayer = transformer.TransformerLayer;
const LayerPrecision = transformer.LayerPrecision;
const FlashAttentionConfig = fa.fa_config.FlashAttentionConfig;

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

    var dt_counts: [32]usize = [_]usize{0} ** 32;
    var it = g.tensors.iterator();
    while (it.next()) |e| {
        const idx: usize = @intFromEnum(e.value_ptr.dtype);
        if (idx < 32) dt_counts[idx] += 1;
    }
    for (dt_counts, 0..) |c, i| {
        if (c > 0) {
            const t_enum = std.enums.fromInt(gguf.GgmlType, @as(u32, @intCast(i)));
            std.debug.print("dtype {s}: {d}\n", .{ t_enum.?.name(), c });
        }
    }
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

test "load real gguf model: embedding, layer 0 weights, forward pass (E1/E2)" {
    const gpa = std.testing.allocator;

    const env_path = std.c.getenv("GGUF_MODEL_PATH") orelse {
        std.debug.print("SKIP: GGUF_MODEL_PATH no está definida\n", .{});
        return error.SkipZigTest;
    };
    const path = std.mem.span(env_path);

    const io = std.Io.Threaded.global_single_threaded.io();

    var model = try gguf_model.GgufModel.load(io, gpa, path);
    defer model.deinit();
    const cfg = model.config;

    std.debug.print("\n=== E: forward capa 0 (CPU) ===\n", .{});
    std.debug.print("arch={s} emb={d} layers={d} heads={d} kv={d} ffn={d} head_dim={d} rope={d:.1}\n", .{
        cfg.architecture, cfg.embedding_length, cfg.block_count, cfg.head_count,
        cfg.head_count_kv, cfg.feed_forward_length,
        cfg.rope_dimension_count, cfg.rope_freq_base,
    });

    // Embedding table dequantizada a f16
    var emb = try model.loadEmbedding();
    defer emb.deinit();
    try std.testing.expectEqual(@as(usize, cfg.vocab_size), emb.shape[0]);
    try std.testing.expectEqual(@as(usize, cfg.embedding_length), emb.shape[1]);
    std.debug.print("token_embd dequant -> [{d}, {d}] f16\n", .{ emb.shape[0], emb.shape[1] });

    // Config de atención: N = tamaño de secuencia del test
    const head_dim = cfg.rope_dimension_count;
    const seq_n: usize = 6;
    const fa_config = FlashAttentionConfig{
        .N = seq_n, .d = head_dim, .num_heads = cfg.head_count, .batch_size = 1,
        .dtype = .f16, .causal = true,
    };
    const precision = LayerPrecision{ .compute = .f16, .weights_on_gpu = false, .use_quantized = false };

    var layer = try TransformerLayer.init(
        gpa, 0, fa_config, "cuda/flash_attention.ptx",
        cfg.embedding_length, precision, cfg.head_count_kv, cfg.feed_forward_length,
    );
    defer layer.deinit();
    layer.rms_eps = cfg.layer_norm_rms_epsilon;
    layer.rope_freq_base = cfg.rope_freq_base;

    try layer.loadWeightsFromGguf(&model.file);
    std.debug.print("layer 0: wq=[{d},{d}] wk=[{d},{d}] wo=[{d},{d}] gate=[{d},{d}] down=[{d},{d}]\n", .{
        layer.w_q_t.?.shape[0], layer.w_q_t.?.shape[1],
        layer.w_k_t.?.shape[0], layer.w_k_t.?.shape[1],
        layer.w_o_t.?.shape[0], layer.w_o_t.?.shape[1],
        layer.w_gate_t.?.shape[0], layer.w_gate_t.?.shape[1],
        layer.w_down_t.?.shape[0], layer.w_down_t.?.shape[1],
    });
    try std.testing.expectEqual(@as(usize, cfg.head_count * head_dim), layer.w_q_t.?.shape[0]);
    try std.testing.expectEqual(@as(usize, cfg.embedding_length), layer.w_q_t.?.shape[1]);
    try std.testing.expectEqual(@as(usize, cfg.head_count_kv * head_dim), layer.w_k_t.?.shape[0]);
    try std.testing.expectEqual(@as(usize, cfg.feed_forward_length), layer.w_gate_t.?.shape[0]);

    // Embedding lookup de tokens de prueba
    const test_tokens = [_]u32{ 9707, 11, 30, 1484, 13, 905 }; // "Hello, world!..."
    const tokens = test_tokens[0..seq_n];
    var hidden = try Tensor(f16).alloc(gpa, &.{ 1, seq_n, cfg.embedding_length });
    defer hidden.deinit();
    const embedding_mod = @import("embedding");
    embedding_mod.embeddingLookup(emb, tokens, 1, seq_n, &hidden);

    var output = try Tensor(f16).alloc(gpa, hidden.shape);
    defer output.deinit();

    try layer.forward(hidden, &output, 0, true);

    // Salida finita y con magnitud razonable
    var max_abs: f32 = 0;
    var any_nan = false;
    for (output.data) |v| {
        const f = @as(f32, @floatCast(v));
        if (std.math.isNan(f) or std.math.isInf(f)) any_nan = true;
        max_abs = @max(max_abs, @abs(f));
    }
    try std.testing.expect(!any_nan);
    try std.testing.expect(max_abs > 0);
    std.debug.print("forward layer 0 OK: max_abs={d:.3}\n", .{max_abs});
}
