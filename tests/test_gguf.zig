//! Test C8 — cargar un GGUF real (vía mmap, C4) y verificar metadata,
//! ModelConfig y shapes de tensores.
//! Requiere la variable de entorno `GGUF_MODEL_PATH` apuntando a un .gguf;
//! si no está definida, el test se salta (error.SkipZigTest).
const std = @import("std");

fn stdoutPrint(io: std.Io, comptime fmt: []const u8, args: anytype) !void {
    var stdout_buf: [256]u8 = undefined;
    const stdout_file = std.Io.File.stdout();
    var stdout_writer = stdout_file.writer(io, &stdout_buf);
    try stdout_writer.interface.print(fmt, args);
}
const gguf = @import("gguf");
const gguf_tokenizer = @import("gguf_tokenizer");
const model_config = @import("model_config");
const gguf_model = @import("gguf_model");
const bpe = @import("tokenizer");
const Tensor = @import("core").Tensor;

test "load real gguf and verify config + tensor shapes" {
    const gpa = std.testing.allocator;

    const env_path = std.c.getenv("GGUF_MODEL_PATH") orelse {
        try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "SKIP: GGUF_MODEL_PATH no está definida\n", .{});
        return error.SkipZigTest;
    };
    const path = std.mem.span(env_path);

    const io = std.Io.Threaded.global_single_threaded.io();

    var g = try gguf.GgufFile.fromFileMmap(io, gpa, path);
    defer g.deinit();

    // Header
    try std.testing.expectEqual(@as(u32, 3), g.version);
    const arch = g.arch() orelse return error.MissingArchitecture;
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "arch={s} alignment={d} tensors={d}\n", .{
        arch, g.alignment, g.tensors.count(),
    });

    // Config
    const cfg = try model_config.ModelConfig.fromGguf(&g);
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "embedding={d} layers={d} heads={d} kv_heads={d} ffn={d} vocab={d} ctx={d} rope_theta={d:.0}\n", .{
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
    const attn_out = if (is_hybrid) std.fmt.bufPrint(&attn_pre, "blk.{d}.attn_output.weight", .{attn_idx}) catch unreachable else if (g.getTensor("blk.0.attn_o.weight") != null)
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
    // MoE (qwen2moe/qwen3moe): expertos fusionados `*_exps` + router
    // `ffn_gate_inp` en vez de FFN densa por capa.
    const is_moe = g.getTensor("blk.0.ffn_gate_exps.weight") != null;
    if (is_moe) {
        required[req_len] = "blk.0.ffn_gate_exps.weight";
        req_len += 1;
        required[req_len] = "blk.0.ffn_up_exps.weight";
        req_len += 1;
        required[req_len] = "blk.0.ffn_down_exps.weight";
        req_len += 1;
        required[req_len] = "blk.0.ffn_gate_inp.weight";
        req_len += 1;
    } else {
        required[req_len] = "blk.0.ffn_gate.weight";
        req_len += 1;
        required[req_len] = "blk.0.ffn_up.weight";
        req_len += 1;
        required[req_len] = "blk.0.ffn_down.weight";
        req_len += 1;
    }
    required[req_len] = if (is_hybrid) "blk.0.post_attention_norm.weight" else "blk.0.ffn_norm.weight";
    req_len += 1;

    for (required[0..req_len]) |name| {
        const t = g.getTensor(name) orelse {
            try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "FALTA tensor: {s}\n", .{name});
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

    // Última capa (FFN densa o expertos MoE)
    const last = cfg.block_count - 1;
    var buf: [64]u8 = undefined;
    const last_ffn = if (is_moe)
        std.fmt.bufPrint(&buf, "blk.{d}.ffn_down_exps.weight", .{last}) catch unreachable
    else
        std.fmt.bufPrint(&buf, "blk.{d}.ffn_down.weight", .{last}) catch unreachable;
    try std.testing.expect(g.getTensor(last_ffn) != null);

    // Datos de tensor accesibles (no vacíos) para token_embd
    const embd = g.getTensor("token_embd.weight").?;
    const embd_bytes = g.tensorData(embd);
    try std.testing.expect(embd_bytes.len > 0);
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "token_embd dtype={s} shape=[{d} {d}] bytes={d}\n", .{
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
            try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "dtype {s}: {d}\n", .{ t_enum.?.name(), c });
        }
    }
}

test "load real gguf tokenizer and build bpe (D1/D2/D4)" {
    const gpa = std.testing.allocator;

    const env_path = std.c.getenv("GGUF_MODEL_PATH") orelse {
        try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "SKIP: GGUF_MODEL_PATH no está definida\n", .{});
        return error.SkipZigTest;
    };
    const path = std.mem.span(env_path);

    const io = std.Io.Threaded.global_single_threaded.io();

    var g = try gguf.GgufFile.fromFileMmap(io, gpa, path);
    defer g.deinit();

    var gt = try gguf_tokenizer.GgufTokenizer.fromGguf(gpa, &g);
    defer gt.deinit();

    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "tokenizer model={s} pre={s} tokens={d} merges={d}\n", .{
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
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "encode -> {d} tokens: ", .{ids.len});
    for (ids) |id| {
        const s = tok.vocab_inv.get(id);
        try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "[{d}:'{s}'] ", .{ id, if (s) |x| x else "<unk>" });
    }
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "\n", .{});
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
        try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "SKIP: GGUF_MODEL_PATH no está definida\n", .{});
        return error.SkipZigTest;
    };
    const path = std.mem.span(env_path);

    const io = std.Io.Threaded.global_single_threaded.io();

    var model = try gguf_model.GgufModel.load(io, gpa, path);
    defer model.deinit();
    const cfg = model.config;

    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "\n=== E: forward capa híbrida (CPU) ===\n", .{});
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "arch={s} emb={d} layers={d} heads={d} kv={d} ffn={d} head_dim={d} rope={d:.1}\n", .{
        cfg.architecture,  cfg.embedding_length,    cfg.block_count,          cfg.head_count,
        cfg.head_count_kv, cfg.feed_forward_length, cfg.rope_dimension_count, cfg.rope_freq_base,
    });

    // Embedding table dequantizada a f16
    var emb = try model.loadEmbedding();
    defer emb.deinit();
    try std.testing.expectEqual(@as(usize, cfg.vocab_size), emb.shape[0]);
    try std.testing.expectEqual(@as(usize, cfg.embedding_length), emb.shape[1]);
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "token_embd dequant -> [{d}, {d}] f16\n", .{ emb.shape[0], emb.shape[1] });

    // Primer capa de atención del modelo híbrido (qwen35: blk.3)
    const layer_idx: usize = if (cfg.is_hybrid) cfg.full_attention_interval - 1 else 0;

    const hybrid_layer = @import("hybrid_layer");
    const hparams = hybrid_layer.HybridLayerParams.fromModelConfig(cfg, 128);
    const paged_attn = @import("paged_attention");

    const head_dim = if (cfg.head_dim > 0) cfg.head_dim else cfg.embedding_length / cfg.head_count;
    var paged_kv = try paged_attn.PagedKVCache.init(gpa, .{
        .block_size = 16,
        .num_blocks = 64,
        .head_dim = head_dim,
        .num_kv_heads = cfg.head_count_kv,
        .num_q_heads = cfg.head_count,
        .dtype = .f16,
        .enable_prefix_cache = false,
        .enable_cpu_offload = false,
        .max_seq_len = 128,
        .max_batch_size = 1,
    });
    defer paged_kv.deinit();
    var block_table = paged_attn.BlockTable.init(gpa, 16);
    defer block_table.deinit(paged_kv.block_alloc);

    var layer = try hybrid_layer.HybridLayer.init(gpa, layer_idx, hparams, true, .auto, &paged_kv, &block_table, null);
    defer layer.deinit();

    try layer.loadWeightsFromGguf(&model.file, null);
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "hybrid attn layer {d} loaded OK\n", .{layer_idx});

    // Embedding lookup de tokens de prueba
    const test_tokens = [_]u32{ 9707, 11, 30, 1484, 13, 905 }; // "Hello, world!..."
    const tokens = test_tokens[0..6];
    var hidden3d = try Tensor(f16).alloc(gpa, &.{ 1, 6, cfg.embedding_length });
    defer hidden3d.deinit();
    const embedding_mod = @import("embedding");
    embedding_mod.embeddingLookup(emb, tokens, 1, 6, &hidden3d);

    const hidden16 = try hidden3d.reshape(&[_]usize{ 6, cfg.embedding_length });
    defer {
        if (hidden16.allocator) |a| {
            a.free(hidden16.shape);
            a.free(hidden16.strides);
        }
    }

    var hidden = try Tensor(f32).alloc(gpa, &.{ 6, cfg.embedding_length });
    defer hidden.deinit();
    for (hidden.data, hidden16.data) |*d, s| d.* = @as(f32, @floatCast(s));

    var output = try Tensor(f32).alloc(gpa, hidden.shape);
    defer output.deinit();

    // Pre-allocate blocks for the 6 prompt tokens (scheduler normally does this)
    try block_table.appendTokens(paged_kv.block_alloc, 6);

    try layer.forward(hidden, &output, 0, 6, null);

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
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "forward hybrid layer {d} OK: max_abs={d:.3}\n", .{ layer_idx, max_abs });
}

test "kv offload 4k context fits in 8GB VRAM (F3)" {
    const gpa = std.testing.allocator;
    const env_path = std.c.getenv("GGUF_MODEL_PATH") orelse {
        try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "SKIP: GGUF_MODEL_PATH no está definida\n", .{});
        return error.SkipZigTest;
    };
    const path = std.mem.span(env_path);
    const io = std.Io.Threaded.global_single_threaded.io();

    var model = try gguf_model.GgufModel.load(io, gpa, path);
    defer model.deinit();
    const cfg = model.config;

    const paged_attn = @import("paged_attention");

    const head_dim = if (cfg.head_dim > 0) cfg.head_dim else cfg.embedding_length / cfg.head_count;

    // 4096 tokens = 256 blocks of 16, need ~256 blocks + margin
    var paged_kv = try paged_attn.PagedKVCache.init(gpa, .{
        .block_size = 16,
        .num_blocks = 300,
        .head_dim = head_dim,
        .num_kv_heads = cfg.head_count_kv,
        .num_q_heads = cfg.head_count,
        .dtype = .f16,
        .enable_prefix_cache = false,
        .enable_cpu_offload = false,
        .max_seq_len = 4096,
        .max_batch_size = 1,
    });
    defer paged_kv.deinit();

    // Test block allocation for 4096 tokens (256 blocks of 16 tokens each)
    var block_table = paged_attn.BlockTable.init(gpa, 16);
    defer block_table.deinit(paged_kv.block_alloc);

    // Allocate blocks for 4096 tokens
    try block_table.appendTokens(paged_kv.block_alloc, 4096);

    const num_blocks = block_table.numBlocks();
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "Allocated {} blocks (expected 256)\n", .{num_blocks});
    if (num_blocks != 256) {
        try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "ERROR: Expected 256 blocks, got {}\n", .{num_blocks});
    }
    try std.testing.expect(num_blocks == 256);

    // Verify we can read/write block mappings
    for (0..256) |i| {
        const physical = block_table.getPhysical(i) orelse return error.Unexpected;
        try std.testing.expect(physical < 300);
    }

    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "kv offload 4k context test passed: 256 blocks allocated\n", .{});
}

test "dequant Q1_0: sign bit per weight, values ±d" {
    const gpa = std.testing.allocator;
    const n = 128; // exactamente 1 bloque
    const bytes = try gpa.alloc(u8, 18);
    defer gpa.free(bytes);
    // d = 2.0
    std.mem.writeInt(u16, bytes[0..2], @bitCast(@as(f16, @floatCast(2.0))), .little);
    // qs: patrón de signos 0x00..0xFF
    for (0..16) |i| { bytes[2 + i] = @intCast(i); }
    const out = try gpa.alloc(f32, n);
    defer gpa.free(out);
    gguf.dequantQ1_0(bytes, out);
    for (out, 0..) |v, j| {
        const byte_idx = j / 8;
        const bit_idx = j % 8;
        const sign = (bytes[2 + byte_idx] >> @intCast(bit_idx)) & 1;
        const expected: f32 = if (sign != 0) 2.0 else -2.0;
        try std.testing.expectApproxEqAbs(expected, v, 1e-5);
    }
}

test "dequant Q2_0: 2-bit codes {-1,0,+1,+2} * d" {
    const gpa = std.testing.allocator;
    const n = 64; // exactamente 1 bloque
    const bytes = try gpa.alloc(u8, 18);
    defer gpa.free(bytes);
    // d = 0.5
    std.mem.writeInt(u16, bytes[0..2], @bitCast(@as(f16, @floatCast(0.5))), .little);
    // qs: recorrer códigos 0..3
    for (0..16) |i| {
        const code: u8 = @intCast(i % 4);
        const packed_val: u8 = @intCast(code | ((i / 4) << 6)); // 4 códigos por byte
        bytes[2 + i] = packed_val;
    }
    const out = try gpa.alloc(f32, n);
    defer gpa.free(out);
    gguf.dequantQ2_0(bytes, out);
    for (out, 0..) |v, j| {
        const byte_idx = j / 4;
        const bit_shift = (j % 4) * 2;
        const q = (bytes[2 + byte_idx] >> @intCast(bit_shift)) & 0x3;
        const expected = @as(f32, @floatFromInt(@as(i32, q) - 1)) * 0.5;
        try std.testing.expectApproxEqAbs(expected, v, 1e-5);
    }
}
