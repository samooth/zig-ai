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
const Tensor = @import("core").Tensor;

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
    // Para qwen35 híbrido, blk.0 es una capa SSM (attn_qkv fusionado +
    // ssm_*) y las capas de atención están en i con (i+1)%interval==0.
    const is_hybrid = cfg.is_hybrid;
    const is_attn0 = cfg.isFullAttentionLayer(0);

    var attn_idx: usize = 0;
    if (is_hybrid and !is_attn0) attn_idx = cfg.full_attention_interval - 1;

    var attn_pre: [64]u8 = undefined;
    const attn_out = if (is_hybrid) std.fmt.bufPrint(&attn_pre, "blk.{d}.attn_output.weight", .{attn_idx}) catch unreachable else
        if (g.getTensor("blk.0.attn_o.weight") != null)
        "blk.0.attn_o.weight"
    else
        "blk.0.attn_output.weight";

    var required: [12][]const u8 = undefined;
    var req_len: usize = 0;
    required[req_len] = "token_embd.weight";
    req_len += 1;
    if (is_hybrid and !is_attn0) {
        // Capa SSM: attn_qkv fusionado + pesos ssm_*
        required[req_len] = "blk.0.attn_qkv.weight";
        req_len += 1;
        required[req_len] = "blk.0.attn_gate.weight";
        req_len += 1;
        required[req_len] = "blk.0.ssm_out.weight";
        req_len += 1;
        required[req_len] = "blk.0.attn_norm.weight";
        req_len += 1;
    } else {
        required[req_len] = attn_out;
        req_len += 1;
        required[req_len] = "blk.0.attn_q.weight";
        req_len += 1;
        required[req_len] = "blk.0.attn_k.weight";
        req_len += 1;
        required[req_len] = "blk.0.attn_v.weight";
        req_len += 1;
        required[req_len] = "blk.0.attn_norm.weight";
        req_len += 1;
    }
    required[req_len] = "blk.0.ffn_gate.weight";
    req_len += 1;
    required[req_len] = "blk.0.ffn_up.weight";
    req_len += 1;
    required[req_len] = "blk.0.ffn_down.weight";
    req_len += 1;
    required[req_len] = if (is_hybrid) "blk.0.post_attention_norm.weight" else "blk.0.ffn_norm.weight";
    req_len += 1;

    for (required[0..req_len]) |name| {
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

    // Capa de atención híbrida (qwen35): blk.3 debe tener q/k/v/output
    if (is_hybrid) {
        const h_attn_q = try std.fmt.allocPrint(gpa, "blk.{d}.attn_q.weight", .{attn_idx});
        defer gpa.free(h_attn_q);
        const h_attn_k = try std.fmt.allocPrint(gpa, "blk.{d}.attn_k.weight", .{attn_idx});
        defer gpa.free(h_attn_k);
        const h_attn_v = try std.fmt.allocPrint(gpa, "blk.{d}.attn_v.weight", .{attn_idx});
        defer gpa.free(h_attn_v);
        const h_attn_o = try std.fmt.allocPrint(gpa, "blk.{d}.attn_output.weight", .{attn_idx});
        defer gpa.free(h_attn_o);
        const h_q_norm = try std.fmt.allocPrint(gpa, "blk.{d}.attn_q_norm.weight", .{attn_idx});
        defer gpa.free(h_q_norm);
        const h_k_norm = try std.fmt.allocPrint(gpa, "blk.{d}.attn_k_norm.weight", .{attn_idx});
        defer gpa.free(h_k_norm);
        for ([_][]const u8{ h_attn_q, h_attn_k, h_attn_v, h_attn_o, h_q_norm, h_k_norm }) |name| {
            try std.testing.expect(g.getTensor(name) != null);
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
    // qwen35 no define bos_token en metadata; solo eos es obligatorio
    try std.testing.expect(gt.eos_id != null);

    // Construir el tokenizer BPE a partir del GGUF
    var tok = try bpe.BPETokenizer.fromTokenizer(gpa, &gt);
    defer tok.deinit();

    if (gt.bos_id != null) {
        try std.testing.expectEqual(gt.bos_id, tok.bos_token);
    } else {
        try std.testing.expect(tok.bos_token == null);
    }

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

test "load real gguf model: embedding, hybrid layer weights, forward pass (E1/E2)" {
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

    std.debug.print("\n=== E: forward capa híbrida (CPU) ===\n", .{});
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

    // Primer capa de atención del modelo híbrido (qwen35: blk.3)
    const layer_idx: usize = if (cfg.is_hybrid) cfg.full_attention_interval - 1 else 0;

    const hybrid_layer = @import("hybrid_layer");
    const hparams = hybrid_layer.HybridLayerParams.fromModelConfig(cfg, 128);

    var layer = try hybrid_layer.HybridLayer.init(gpa, layer_idx, hparams, true);
    defer layer.deinit();

    try layer.loadWeightsFromGguf(&model.file);
    std.debug.print("hybrid attn layer {d} loaded OK\n", .{layer_idx});

    // Embedding lookup de tokens de prueba
    const test_tokens = [_]u32{ 9707, 11, 30, 1484, 13, 905 }; // "Hello, world!..."
    const tokens = test_tokens[0..6];
    var hidden3d = try Tensor(f16).alloc(gpa, &.{ 1, 6, cfg.embedding_length });
    defer hidden3d.deinit();
    const embedding_mod = @import("embedding");
    embedding_mod.embeddingLookup(emb, tokens, 1, 6, &hidden3d);

    var hidden = try hidden3d.reshape(&[_]usize{ 6, cfg.embedding_length });
    defer { if (hidden.allocator) |a| { a.free(hidden.shape); a.free(hidden.strides); } }

    var output = try Tensor(f16).alloc(gpa, hidden.shape);
    defer output.deinit();

    try layer.forward(hidden, &output, 0, 6);

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
    std.debug.print("forward hybrid layer {d} OK: max_abs={d:.3}\n", .{layer_idx, max_abs});
}
