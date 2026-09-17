//! U2-llama (eje §12 Unificación GPU, lane-b1 Dev A): gate de CAPA 512tok
//! con pesos Llama REALES — de-riesga el gate final U2 sin esperar a U1
//! (el route llama→híbrido de lane-d): este arnés construye HybridLayer
//! DIRECTAMENTE desde los tensores del GGUF, el mismo chasis que U1
//! configurará en main.zig (is_attention=true, use_mrope=false, RoPE NEOX
//! clásico vía mropeKernel con sections {n_rot/2,0,0,0} ≡ applyRoPE).
//!
//! CPU golden: HybridLayer.forward (dequant f32 + applyRoPEMultiSection
//! con sections default = applyRoPE NEOX). GPU: forwardGPU n>1 batched
//! (proyecciones n, splitQG/no_gate, rmsNorm n*heads, mrope, kvAppend n,
//! prefillDevice n, gate → FFN qgemm).
//!
//! Chasis probado aquí que U1 reutiliza:
//!   - attn_q_norm/attn_k_norm AUSENTES en llama ⇒ ones (identidad)
//!   - post_attention_norm ausente ⇒ ffn_norm (fallback ya existente)
//!   - no_gate autodetect por geometría de attn_q (out == q_dim)
//!   - n_kv_head del tensor real de attn_k
//!
//! Requiere GGUF_MODEL_PATH apuntando a un GGUF LLAMA-LIKE denso (p.ej.
//! /ai/models/Llama-3.2-3B-Instruct-Q3_K_S.gguf) — arch "llama".
//! Gate: rel < 1e-2 para Q3_K_S (ruido de aritmética f32-CPU vs qgemm
//! cuantizado GPU, misma clase §3.1); escala coherente [0.9,1.1]; 0 NaN.

const std = @import("std");
const gpa = std.testing.allocator;
const Tensor = @import("core").Tensor;
const gguf_model = @import("gguf_model");
const gguf = @import("gguf");
const paged_attn = @import("paged_attention");
const layer_kernels = @import("layer_kernels");
const matmul = @import("matmul");
const cudaz = @import("cudaz");
const cublas = @import("cublas");
const hybrid_layer = @import("hybrid_layer");
const embedding_mod = @import("embedding");

test "U2-llama: prefill GPU dense batched n=512 (capa real Llama-3.2) ≈ CPU golden" {
    const env_path = std.c.getenv("GGUF_MODEL_PATH") orelse {
        std.debug.print("SKIP: GGUF_MODEL_PATH no está definida\n", .{});
        return error.SkipZigTest;
    };
    if (!cudaz.isCudaAvailable()) {
        std.debug.print("SKIP: CUDA no disponible\n", .{});
        return error.SkipZigTest;
    }
    const path = std.mem.span(env_path);
    const io = std.Io.Threaded.global_single_threaded.io();

    var model = try gguf_model.GgufModel.load(io, gpa, path);
    defer model.deinit();
    const cfg = model.config;

    // Chasis U1: SOLO llama-like denso (este test no aplica a híbridos).
    const is_llama_like = !cfg.is_hybrid and std.mem.eql(u8, cfg.architecture, "llama");
    if (!is_llama_like) {
        std.debug.print("SKIP: arch={s} no es llama denso (chasis U1 llama)\n", .{cfg.architecture});
        return error.SkipZigTest;
    }

    std.debug.print("=== U2-llama: prefill batched n=512 CPU vs GPU ===\n", .{});
    const head_dim = if (cfg.head_dim > 0) cfg.head_dim else cfg.embedding_length / cfg.head_count;
    std.debug.print("arch={s} n_embd={d} heads={d} kv={d} hd={d} n_rot={d} base={d} blk={d}\n", .{
        cfg.architecture,   cfg.embedding_length,
        cfg.head_count,     cfg.head_count_kv,
        head_dim,           cfg.rope_dimension_count,
        cfg.rope_freq_base, cfg.block_count,
    });

    // Embeddings (tabla f16 — lookup clásico para el input).
    var emb = try model.loadEmbedding();
    defer emb.deinit();

    // Chasis U1: params con use_mrope=false (llama NEOX clásico) — el
    // fromModelConfig default daría use_mrope=true (qwen35), que con
    // sections {n_rot/2,0,0,0} es MATEMÁTICAMENTE equivalente (todos los
    // sectores usan theta_t) — lo fijamos a false para reflejar el chasis
    // exacto de U1 y ejercer el camino CPU applyRoPE clásico del golden.
    var hparams = hybrid_layer.HybridLayerParams.fromModelConfig(cfg, 640);
    hparams.use_mrope = false; // llama: RoPE estándar NEOX (chasis U1)
    // is_attention=true TODAS (modelo denso — U1 construye HybridLayer[]
    // con is_attention=true); probamos la capa 0.
    const layer_idx: usize = 0;
    const n_embd = cfg.embedding_length;
    const block_size: usize = 16;

    // ── Dos instancias INDEPENDIENTES de paged KV + block table.
    var kv_cpu = try paged_attn.PagedKVCache.init(gpa, .{
        .block_size = block_size,
        .num_blocks = 128,
        .head_dim = head_dim,
        .num_kv_heads = cfg.head_count_kv,
        .num_q_heads = cfg.head_count,
        .dtype = .f16,
        .enable_prefix_cache = false,
        .enable_cpu_offload = false,
        .max_seq_len = 640,
        .max_batch_size = 1,
    });
    defer kv_cpu.deinit();
    var bt_cpu = paged_attn.BlockTable.init(gpa, block_size);
    defer bt_cpu.deinit(kv_cpu.block_alloc);

    var kv_gpu = try paged_attn.PagedKVCache.init(gpa, .{
        .block_size = block_size,
        .num_blocks = 128,
        .head_dim = head_dim,
        .num_kv_heads = cfg.head_count_kv,
        .num_q_heads = cfg.head_count,
        .dtype = .f16,
        .enable_prefix_cache = false,
        .enable_cpu_offload = false,
        .max_seq_len = 640,
        .max_batch_size = 1,
    });
    defer kv_gpu.deinit();
    var bt_gpu = paged_attn.BlockTable.init(gpa, block_size);
    defer bt_gpu.deinit(kv_gpu.block_alloc);

    try cudaz.ensureContext();
    var paged_gpu = try paged_attn.PagedAttentionGpu.init(
        gpa,
        .{
            .block_size = block_size,
            .num_blocks = 0,
            .head_dim = head_dim,
            .num_kv_heads = cfg.head_count_kv,
            .num_q_heads = cfg.head_count,
            .dtype = .f16,
            .quant_k = .fp16,
            .quant_v = .fp16,
        },
        @ptrCast((try matmul.MatmulEngine.sharedCudaStream()).raw),
    );
    defer paged_gpu.deinit();

    var lcpu = try hybrid_layer.HybridLayer.init(gpa, layer_idx, hparams, true, .parallel, &kv_cpu, &bt_cpu, null);
    defer lcpu.deinit();
    try lcpu.loadWeightsFromGguf(&model.file, null);

    var lgpu = try hybrid_layer.HybridLayer.init(gpa, layer_idx, hparams, true, .auto, &kv_gpu, &bt_gpu, &paged_gpu);
    defer lgpu.deinit();
    try lgpu.loadWeightsFromGguf(&model.file, null);

    var lk = try layer_kernels.LayerKernels.init(@ptrCast((try matmul.MatmulEngine.sharedCudaStream()).raw));
    defer lk.deinit();

    // ── Input: 512 tokens (texto France del gate) con embedding lookup.
    const n: usize = 512;
    // "The capital of France is the city of Paris" repetido — tokens
    // Llama-3 reales (vocab 128256; ids verificados con el tokenizer del
    // modelo en el arnés 0.8B de U2; aquí basta determinismo).
    const test_tokens = [_]u32{ 791, 6821, 311, 278, 9060, 310, 29871, 1576, 322, 3384 };
    var hidden3d = try Tensor(f16).alloc(gpa, &.{ 1, n, n_embd });
    defer hidden3d.deinit();
    var toks = try gpa.alloc(u32, n);
    defer gpa.free(toks);
    var i: usize = 0;
    while (i < n) : (i += 1) toks[i] = test_tokens[i % test_tokens.len];
    embedding_mod.embeddingLookup(emb, toks, 1, n, &hidden3d);

    var hidden = try Tensor(f32).alloc(gpa, &.{ n, n_embd });
    defer hidden.deinit();
    for (hidden.data, hidden3d.data) |*d, s| d.* = @as(f32, @floatCast(s));

    // Pre-allocate blocks (512/16 = 32 bloques por instancia).
    try bt_cpu.appendTokens(kv_cpu.block_alloc, n);
    try bt_gpu.appendTokens(kv_gpu.block_alloc, n);

    // ── Golden CPU: forward con n=512, start_pos=0.
    var out_cpu = try Tensor(f32).alloc(gpa, hidden.shape);
    defer out_cpu.deinit();
    try lcpu.forward(hidden, &out_cpu, 0, n, null);

    // ── GPU: forwardGPU con n=512, start_pos=0.
    var x_dev = try cublas.GpuTensor(f32).alloc(n * n_embd);
    defer x_dev.deinit();
    var out_dev = try cublas.GpuTensor(f32).alloc(n * n_embd);
    defer out_dev.deinit();
    try cudaz.cuMemcpyHtoD(x_dev.ptr(), @intFromPtr(hidden.data.ptr), n * n_embd * @sizeOf(f32));
    try cudaz.cuMemsetD8(out_dev.ptr(), 0, n * n_embd * @sizeOf(f32));

    try lgpu.forwardGPU(&lk, x_dev, &out_dev, 0, n, null);
    try cudaz.cuStreamSynchronize(lk.stream);

    const out_gpu = try gpa.alloc(f32, n * n_embd);
    defer gpa.free(out_gpu);
    try cudaz.cuMemcpyDtoH(@intFromPtr(out_gpu.ptr), out_dev.ptr(), n * n_embd * @sizeOf(f32));

    // ── Gate: rel < 1e-2 (Q3_K_S: CPU dequant f32 vs GPU qgemm cuantizado
    // = ruido de aritmética, clase §3.1 — 1e-3 exige f16 puro).
    var diff: f64 = 0;
    var norm: f64 = 0;
    var norm_gpu: f64 = 0;
    var max_abs: f32 = 0;
    var nan_count: usize = 0;
    for (out_cpu.data, out_gpu) |c, g| {
        const d = @as(f64, c) - @as(f64, g);
        diff += d * d;
        norm += @as(f64, c) * @as(f64, c);
        norm_gpu += @as(f64, g) * @as(f64, g);
        max_abs = @max(max_abs, @abs(c - g));
        if (std.math.isNan(g) or std.math.isNan(c)) nan_count += 1;
    }
    const rel: f64 = if (norm > 0) @sqrt(diff / norm) else 0;
    const scale_ratio: f64 = if (norm > 0) @sqrt(norm_gpu / norm) else 1;
    std.debug.print("U2-llama prefill n={d}: rel={d:.6} max_abs={d:.6} scale_ratio={d:.4} nan={d} (gate 1e-2 cuantizado)\n", .{ n, rel, max_abs, scale_ratio, nan_count });
    if (nan_count > 0) return error.PrefillDenseNaN;
    if (scale_ratio < 0.9 or scale_ratio > 1.1) {
        std.debug.print("U2-llama FALLO coherencia: scale_ratio={d:.4} fuera de [0.9,1.1]\n", .{scale_ratio});
        return error.PrefillDenseIncoherent;
    }
    if (rel > 1e-2) {
        std.debug.print("U2-llama FALLO: rel={d:.6} > 1e-2\n", .{rel});
        return error.PrefillDenseMismatch;
    }
    std.debug.print("U2-llama OK: prefill GPU batched 512tok capa real Llama ≈ CPU golden (rel={d:.6})\n", .{rel});

    // Limpieza del caché global q4 (pattern main.zig:870).
    layer_kernels.deinitQ4Cache();
}
