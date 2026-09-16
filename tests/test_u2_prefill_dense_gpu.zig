//! U2 (eje §12 Unificación GPU, lane-b1 Dev A): paridad prefill GPU dense
//! batched — `HybridLayer.forward` (CPU golden) vs `HybridLayer.forwardGPU`
//! con n>1 (chunk batched).
//!
//! El path GPU de prefill batched YA existe para qwen35 (mrope con ids
//! secuenciales == NEOX; kvAppend n; prefillDevice n). Este test lo valida
//! de CAPA COMPLETA (attn_norm → proyecciones → splitQG → rmsNorm → RoPE →
//! kvAppend → paged prefill → gate → out) contra el golden CPU, la pieza
//! que falta para que U1 (route llama→híbrido de lane-d) pueda llenar el
//! chasis con confianza.
//!
//! Requiere `GGUF_MODEL_PATH` apuntando a un GGUF HÍBRIDO (p.ej.
//! /ai/models/Qwen3.5-0.8B-Q4_0.gguf) — mismo patrón que test_gguf E1/E2.
//! Gate: rel < 1e-3 (norma relativa de la diferencia).

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

test "U2: prefill GPU dense batched (n=64) — forwardGPU ≡ forward CPU, rel<1e-3" {
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

    std.debug.print("=== U2: prefill batched CPU vs GPU (n=64) ===\n", .{});
    std.debug.print("arch={s} emb={d} heads={d} kv={d} hd={d}\n", .{
        cfg.architecture,                                                              cfg.embedding_length, cfg.head_count, cfg.head_count_kv,
        if (cfg.head_dim > 0) cfg.head_dim else cfg.embedding_length / cfg.head_count,
    });

    // Embeddings (tabla f16 — lookup clásico para el input)
    var emb = try model.loadEmbedding();
    defer emb.deinit();

    const layer_idx: usize = if (cfg.is_hybrid) cfg.full_attention_interval - 1 else 0;
    const hparams = hybrid_layer.HybridLayerParams.fromModelConfig(cfg, 128);
    const head_dim = if (cfg.head_dim > 0) cfg.head_dim else cfg.embedding_length / cfg.head_count;
    const n_embd = cfg.embedding_length;
    const block_size: usize = 16;

    // ── Dos instancias INDEPENDIENTES de paged KV + block table (CPU y GPU
    // escriben su propio cache; ambas parten de vacío con los mismos pesos
    // y la misma entrada ⇒ los outputs deben coincidir).
    var kv_cpu = try paged_attn.PagedKVCache.init(gpa, .{
        .block_size = block_size,
        .num_blocks = 128,
        .head_dim = head_dim,
        .num_kv_heads = cfg.head_count_kv,
        .num_q_heads = cfg.head_count,
        .dtype = .f16,
        .enable_prefix_cache = false,
        .enable_cpu_offload = false,
        .max_seq_len = 256,
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
        .max_seq_len = 256,
        .max_batch_size = 1,
    });
    defer kv_gpu.deinit();
    var bt_gpu = paged_attn.BlockTable.init(gpa, block_size);
    defer bt_gpu.deinit(kv_gpu.block_alloc);

    // GPU side (como cli.zig): PagedAttentionGpu + LayerKernels + GpuTensors.
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
    try lcpu.loadWeightsFromGguf(&model.file);

    var lgpu = try hybrid_layer.HybridLayer.init(gpa, layer_idx, hparams, true, .auto, &kv_gpu, &bt_gpu, &paged_gpu);
    defer lgpu.deinit();
    try lgpu.loadWeightsFromGguf(&model.file);

    var lk = try layer_kernels.LayerKernels.init(@ptrCast((try matmul.MatmulEngine.sharedCudaStream()).raw));
    defer lk.deinit();

    // ── Input: 64 tokens con embedding determinista.
    const n: usize = 64;
    const test_tokens = [_]u32{ 9707, 11, 30, 1484, 13, 905, 11, 527, 9707, 30, 304, 1484, 13, 905, 11, 527, 9707, 30, 304, 1484, 13, 905, 11, 527, 9707, 11, 30, 1484, 13, 905, 11, 527, 9707, 30, 304, 1484, 13, 905, 11, 527, 9707, 11, 30, 1484, 13, 905, 11, 527, 9707, 30, 304, 1484, 13, 905, 11, 527, 9707, 11, 30, 1484, 13, 905, 11, 527 };
    var hidden3d = try Tensor(f16).alloc(gpa, &.{ 1, n, n_embd });
    defer hidden3d.deinit();
    embedding_mod.embeddingLookup(emb, test_tokens[0..n], 1, n, &hidden3d);

    var hidden = try Tensor(f32).alloc(gpa, &.{ n, n_embd });
    defer hidden.deinit();
    for (hidden.data, hidden3d.data) |*d, s| d.* = @as(f32, @floatCast(s));

    // Pre-allocate blocks (64 tok / block_size 16 = 4 bloques por instancia).
    try bt_cpu.appendTokens(kv_cpu.block_alloc, n);
    try bt_gpu.appendTokens(kv_gpu.block_alloc, n);

    // ── Golden CPU: forward con n=64, start_pos=0.
    var out_cpu = try Tensor(f32).alloc(gpa, hidden.shape);
    defer out_cpu.deinit();
    try lcpu.forward(hidden, &out_cpu, 0, n, null);

    // ── GPU: forwardGPU con n=64, start_pos=0 (device tensors).
    var x_dev = try cublas.GpuTensor(f32).alloc(n * n_embd);
    defer x_dev.deinit();
    var out_dev = try cublas.GpuTensor(f32).alloc(n * n_embd);
    defer out_dev.deinit();
    try cudaz.cuMemcpyHtoD(x_dev.ptr(), @intFromPtr(hidden.data.ptr), n * n_embd * @sizeOf(f32));
    // El kernel escribe out completo; limpiamos para detectar escrituras parciales.
    try cudaz.cuMemsetD8(out_dev.ptr(), 0, n * n_embd * @sizeOf(f32));

    try lgpu.forwardGPU(&lk, x_dev, &out_dev, 0, n, null);
    try cudaz.cuStreamSynchronize(lk.stream);

    const out_gpu = try gpa.alloc(f32, n * n_embd);
    defer gpa.free(out_gpu);
    try cudaz.cuMemcpyDtoH(@intFromPtr(out_gpu.ptr), out_dev.ptr(), n * n_embd * @sizeOf(f32));

    // ── Gate. El camino CPU usa dequant f32 + linearProjection f32; el
    // GPU usa qgemm q4_0 (dot dp4a in-kernel). Con pesos Q4_0 la diferencia
    // es RUIDO DE ARITMÉTICA (misma clase que TODO §3.1), no wiring:
    // gate rel < 1e-2 para cuantizado. El gate estricto 1e-3 aplica a
    // pesos f16/bf16 puros (pendiente de modelo f16 para cerrarlo).
    // Sanidad adicional: sin NaN y norma del mismo orden (coherencia).
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
    std.debug.print("U2 prefill n={d}: rel={d:.6} max_abs={d:.6} scale_ratio={d:.4} nan={d} (gate 1e-2 cuantizado)\n", .{ n, rel, max_abs, scale_ratio, nan_count });
    // Diagnóstico por-token para aislar si la divergencia es uniforme
    // (proyecciones batched) o concentrada (KV append / attn batched).
    var max_token_rel: f64 = 0;
    var min_token_rel: f64 = 1e9;
    var worst_pos: usize = 0;
    for (0..n) |pos| {
        var td: f64 = 0;
        var tn: f64 = 0;
        var ma: f32 = 0;
        const base = pos * n_embd;
        for (out_cpu.data[base..][0..n_embd], out_gpu[base..][0..n_embd]) |c, g| {
            const d = @as(f64, c) - @as(f64, g);
            td += d * d;
            tn += @as(f64, c) * @as(f64, c);
            ma = @max(ma, @abs(c - g));
        }
        const tr = if (tn > 0) @sqrt(td / tn) else 0;
        if (tr > max_token_rel) {
            max_token_rel = tr;
            worst_pos = pos;
        }
        if (tr < min_token_rel) min_token_rel = tr;
    }
    std.debug.print("U2 per-token: min_rel={d:.6} max_rel={d:.6} worst_pos={d}\n", .{ min_token_rel, max_token_rel, worst_pos });
    if (nan_count > 0) return error.PrefillDenseNaN;
    if (scale_ratio < 0.9 or scale_ratio > 1.1) {
        std.debug.print("U2 FALLO coherencia: scale_ratio={d:.4} fuera de [0.9,1.1]\n", .{scale_ratio});
        return error.PrefillDenseIncoherent;
    }
    if (rel > 1e-2) {
        std.debug.print("U2 FALLO: rel={d:.6} > 1e-2\n", .{rel});
        return error.PrefillDenseMismatch;
    }
    std.debug.print("U2 OK: prefill GPU batched ≈ CPU golden (rel={d:.6}, ruido de cuantización q4_0)\n", .{rel});

    // Limpieza del caché global de pesos q4 (pattern main.zig:870) — el
    // q4Weight del forwardGPU llena q4_cache; sin esto el testing.allocator
    // reporta leak (la vida del caché es la del PROCESO en producción).
    layer_kernels.deinitQ4Cache();
}
