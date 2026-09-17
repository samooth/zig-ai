//! U2-llama-stack (eje §12, lane-b1 Dev A): paridad E2E-casi — PILA COMPLETA
//! de HybridLayers (28 capas reales del Llama-3.2-3B) + output_norm +
//! lm_head cuantizado (CPU-GEMV lmHeadGemvQuant, 7.1d), CPU vs GPU.
//!
//! Esto valida TODO lo que U1 (route llama→híbrido de lane-d) pondrá en
//! marcha en main.zig, EXCEPTO el route mismo: mis capas se construyen
//! directo del GGUF con el chasis ya gateado (test_u2_prefill_llama_gpu:
//! q/k_norm ones, ffn_norm fallback, mrope≡NEOX, qgemm 0 fallbacks).
//!
//! Métrica E2E real: greedy top-1 del lm_head en posiciones del prompt —
//! si el stack GPU diverge del golden CPU en la top-1, el E2E Paris de
//! U1 no puede ser coherente; si coinciden, el gate md5 Paris de U1 queda
//! de-riesgado a nivel red completa.
//!
//! Protocolo de memoria (espejo de cli.zig:1500-1650):
//!   - PagedKVCache ÚNICO compartido por TODAS las capas (por brazo)
//!   - BlockTable por capa (create por capa de atención)
//!   - PagedAttentionGpu ÚNICO compartido (pool por phys_id global)
//!
//! Requiere GGUF_MODEL_PATH = GGUF llama denso (arch "llama").
//! Gates: (a) pila: rel < 1e-2; (b) greedy: top-1 idéntico CPU vs GPU en
//! 8 posiciones sampleadas; (c) 0 NaN.

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
const norm = @import("norm");
const debugz = @import("debug");
const QuantWeight = @import("quant_weight").QuantWeight;

test "U2-llama-stack: pila 28 capas + output_norm + lm_head — CPU ≡ GPU (greedy top-1)" {
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

    const is_llama_like = !cfg.is_hybrid and std.mem.eql(u8, cfg.architecture, "llama");
    if (!is_llama_like) {
        std.debug.print("SKIP: arch={s} no es llama denso\n", .{cfg.architecture});
        return error.SkipZigTest;
    }

    const head_dim = if (cfg.head_dim > 0) cfg.head_dim else cfg.embedding_length / cfg.head_count;
    const n_embd = cfg.embedding_length;
    const n_layers = cfg.block_count;
    std.debug.print("=== U2-llama-stack: {d} capas n=512 CPU vs GPU ===\n", .{n_layers});
    std.debug.print("n_embd={d} heads={d} kv={d} hd={d} n_rot={d} base={d}\n", .{
        n_embd,                   cfg.head_count,
        cfg.head_count_kv,        head_dim,
        cfg.rope_dimension_count, cfg.rope_freq_base,
    });

    var emb = try model.loadEmbedding();
    defer emb.deinit();
    var out_norm = try model.loadOutputNorm();
    defer out_norm.deinit();
    // lm_head cuantizado: GEMV CPU idéntico para AMBOS brazos (sólo la
    // PILA es lo que se compara — el lm_head no es superficie U1/U2).
    const lm_qw = try model.loadLmHeadQuant();
    const vocab: usize = @intCast(lm_qw.info.dims[1]);

    // ── KV paginado: UNA instancia por brazo, compartida por todas las
    // capas del brazo (espejo cli.zig:1500). 512tok/16 = 32 bloques por
    // capa × 28 capas ⇒ 896 bloques por brazo + margen.
    const block_size: usize = 16;
    const n: usize = 512;
    const max_seq: usize = 640;

    var kv_cpu = try paged_attn.PagedKVCache.init(gpa, .{
        .block_size = block_size,
        .num_blocks = 64 * n_layers,
        .head_dim = head_dim,
        .num_kv_heads = cfg.head_count_kv,
        .num_q_heads = cfg.head_count,
        .dtype = .f16,
        .enable_prefix_cache = false,
        .enable_cpu_offload = false,
        .max_seq_len = max_seq,
        .max_batch_size = 1,
    });
    defer kv_cpu.deinit();
    var kv_gpu = try paged_attn.PagedKVCache.init(gpa, .{
        .block_size = block_size,
        .num_blocks = 64 * n_layers,
        .head_dim = head_dim,
        .num_kv_heads = cfg.head_count_kv,
        .num_q_heads = cfg.head_count,
        .dtype = .f16,
        .enable_prefix_cache = false,
        .enable_cpu_offload = false,
        .max_seq_len = max_seq,
        .max_batch_size = 1,
    });
    defer kv_gpu.deinit();

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

    // ── Capas: TODAS is_attention=true (denso — chasis U1). BlockTable por
    // capa (create per layer, espejo cli.zig:1640).
    var hparams = hybrid_layer.HybridLayerParams.fromModelConfig(cfg, max_seq);
    hparams.use_mrope = false; // llama: NEOX clásico (chasis U1)

    var layers_cpu = try gpa.alloc(hybrid_layer.HybridLayer, n_layers);
    defer gpa.free(layers_cpu);
    var bts_cpu = try gpa.alloc(*paged_attn.BlockTable, n_layers);
    defer gpa.free(bts_cpu);
    var layers_gpu = try gpa.alloc(hybrid_layer.HybridLayer, n_layers);
    defer gpa.free(layers_gpu);
    var bts_gpu = try gpa.alloc(*paged_attn.BlockTable, n_layers);
    defer gpa.free(bts_gpu);

    for (0..n_layers) |i| {
        bts_cpu[i] = try gpa.create(paged_attn.BlockTable);
        bts_cpu[i].* = paged_attn.BlockTable.init(gpa, block_size);
        layers_cpu[i] = try hybrid_layer.HybridLayer.init(gpa, i, hparams, true, .parallel, &kv_cpu, bts_cpu[i], null);
        try layers_cpu[i].loadWeightsFromGguf(&model.file, null);

        bts_gpu[i] = try gpa.create(paged_attn.BlockTable);
        bts_gpu[i].* = paged_attn.BlockTable.init(gpa, block_size);
        layers_gpu[i] = try hybrid_layer.HybridLayer.init(gpa, i, hparams, true, .auto, &kv_gpu, bts_gpu[i], &paged_gpu);
        try layers_gpu[i].loadWeightsFromGguf(&model.file, null);
        debugz.dbg.printLevel(.info, "[milestone] capa {d}/{d} cargada (cpu+gpu)\n", .{ i + 1, n_layers });
    }
    defer for (bts_cpu) |bt| {
        bt.deinit(kv_cpu.block_alloc);
        gpa.destroy(bt);
    };
    defer for (bts_gpu) |bt| {
        bt.deinit(kv_gpu.block_alloc);
        gpa.destroy(bt);
    };
    defer for (layers_cpu) |*l| l.deinit();
    defer for (layers_gpu) |*l| l.deinit();

    var lk = try layer_kernels.LayerKernels.init(@ptrCast((try matmul.MatmulEngine.sharedCudaStream()).raw));
    defer lk.deinit();

    // ── Input: 512 tokens (prompt determinista, mismo del test de capa).
    const test_tokens = [_]u32{ 791, 6821, 311, 278, 9060, 310, 29871, 1576, 322, 3384 };
    var hidden3d = try Tensor(f16).alloc(gpa, &.{ 1, n, n_embd });
    defer hidden3d.deinit();
    var toks = try gpa.alloc(u32, n);
    defer gpa.free(toks);
    for (0..n) |i| toks[i] = test_tokens[i % test_tokens.len];
    embedding_mod.embeddingLookup(emb, toks, 1, n, &hidden3d);

    var hidden = try Tensor(f32).alloc(gpa, &.{ n, n_embd });
    defer hidden.deinit();
    for (hidden.data, hidden3d.data) |*d, s| d.* = @as(f32, @floatCast(s));

    // ── Pre-alloc de bloques por capa.
    for (bts_cpu) |bt| try bt.appendTokens(kv_cpu.block_alloc, n);
    for (bts_gpu) |bt| try bt.appendTokens(kv_gpu.block_alloc, n);

    // ── Golden CPU: pila completa forward n=512 (ping-pong buffers).
    var buf_a_cpu = try Tensor(f32).alloc(gpa, hidden.shape);
    defer buf_a_cpu.deinit();
    var buf_b_cpu = try Tensor(f32).alloc(gpa, hidden.shape);
    defer buf_b_cpu.deinit();
    @memcpy(buf_a_cpu.data, hidden.data);
    var last_cpu: *Tensor(f32) = &buf_a_cpu;
    for (0..n_layers) |i| {
        const src = last_cpu;
        const dst: *Tensor(f32) = if (src == &buf_a_cpu) &buf_b_cpu else &buf_a_cpu;
        try layers_cpu[i].forward(src.*, dst, 0, n, null);
        last_cpu = dst;
    }

    // ── GPU: pila completa forwardGPU n=512 (ping-pong device buffers).
    var x_dev = try cublas.GpuTensor(f32).alloc(n * n_embd);
    defer x_dev.deinit();
    var y_dev = try cublas.GpuTensor(f32).alloc(n * n_embd);
    defer y_dev.deinit();
    try cudaz.cuMemcpyHtoD(x_dev.ptr(), @intFromPtr(hidden.data.ptr), n * n_embd * @sizeOf(f32));
    var last_dev: *cublas.GpuTensor(f32) = &x_dev;
    for (0..n_layers) |i| {
        const src = last_dev;
        const dst: *cublas.GpuTensor(f32) = if (src == &x_dev) &y_dev else &x_dev;
        try layers_gpu[i].forwardGPU(&lk, src.*, dst, 0, n, null);
        try cudaz.cuStreamSynchronize(lk.stream);
        last_dev = dst;
    }
    const out_gpu = try gpa.alloc(f32, n * n_embd);
    defer gpa.free(out_gpu);
    try cudaz.cuMemcpyDtoH(@intFromPtr(out_gpu.ptr), last_dev.ptr(), n * n_embd * @sizeOf(f32));

    // ── Gate (a): paridad de pila, rel < 1e-2.
    var diff: f64 = 0;
    var nrm: f64 = 0;
    var nrm_gpu: f64 = 0;
    var nan_count: usize = 0;
    var max_abs: f32 = 0;
    for (last_cpu.data, out_gpu) |c, g| {
        const d = @as(f64, c) - @as(f64, g);
        diff += d * d;
        nrm += @as(f64, c) * @as(f64, c);
        nrm_gpu += @as(f64, g) * @as(f64, g);
        max_abs = @max(max_abs, @abs(c - g));
        if (std.math.isNan(g) or std.math.isNan(c)) nan_count += 1;
    }
    const rel: f64 = if (nrm > 0) @sqrt(diff / nrm) else 0;
    const scale_ratio: f64 = if (nrm > 0) @sqrt(nrm_gpu / nrm) else 1;
    std.debug.print("U2-llama-stack: rel={d:.6} max_abs={d:.6} scale_ratio={d:.4} nan={d}\n", .{ rel, max_abs, scale_ratio, nan_count });
    if (nan_count > 0) return error.StackNaN;
    if (scale_ratio < 0.9 or scale_ratio > 1.1) return error.StackIncoherent;
    if (rel > 1e-2) return error.StackMismatch;

    // ── Gate (b): greedy top-1 CPU vs GPU en posiciones sampleadas.
    // output_norm (RMSNorm final) + lm_head GEMV cuantizado — MISMO camino
    // en ambos brazos (sólo la pila difiere). El argmax con desempate por
    // índice mínimo = Sampler.greedyArgmax.
    const sample_positions = [_]usize{ n - 1, n - 33, n - 65, n - 129, n - 257, 256, 128, 64 };
    var hbuf = try Tensor(f32).alloc(gpa, &.{ 1, n_embd });
    defer hbuf.deinit();
    const h16 = try gpa.alloc(f16, n_embd);
    defer gpa.free(h16);
    const logits_buf = try gpa.alloc(f32, vocab);
    defer gpa.free(logits_buf);
    var mismatches: usize = 0;
    for (sample_positions) |pos| {
        const top_cpu = greedyTop1(last_cpu.data[pos * n_embd ..][0..n_embd], &out_norm, &lm_qw, &hbuf, h16, logits_buf, cfg.layer_norm_rms_epsilon);
        const top_gpu = greedyTop1(out_gpu[pos * n_embd ..][0..n_embd], &out_norm, &lm_qw, &hbuf, h16, logits_buf, cfg.layer_norm_rms_epsilon);
        std.debug.print("pos={d}: cpu top-1={d} gpu top-1={d} {s}\n", .{ pos, top_cpu, top_gpu, if (top_cpu == top_gpu) "OK" else "DIVERGE" });
        if (top_cpu != top_gpu) mismatches += 1;
    }
    if (mismatches > 0) {
        std.debug.print("U2-llama-stack FALLO: {d}/{d} top-1 divergentes\n", .{ mismatches, sample_positions.len });
        return error.StackGreedyDivergence;
    }
    std.debug.print("U2-llama-stack OK: {d} capas reales + head, greedy top-1 idéntico en {d} posiciones (rel={d:.6})\n", .{ n_layers, sample_positions.len, rel });

    layer_kernels.deinitQ4Cache();
}

/// RMSNorm final + lm_head cuantizado GEMV + argmax greedy (índice mínimo
/// entre empates — misma semántica que Sampler.greedyArgmax y que el
/// argmaxF32Kernel de 1.7).
fn greedyTop1(
    row: []const f32,
    out_norm: *Tensor(f32),
    lm_qw: *const QuantWeight,
    hbuf: *Tensor(f32),
    h16: []f16,
    logits_buf: []f32,
    eps: f32,
) u32 {
    // RMSNorm final sobre la fila.
    var row_shape = [_]usize{ 1, row.len };
    var row_strides = [_]usize{ row.len, 1 };
    const row_view = Tensor(f32){
        .data = @constCast(row),
        .shape = &row_shape,
        .strides = &row_strides,
        .offset = 0,
        .allocator = null,
        .owns_data = false,
    };
    const w_view = Tensor(f32){
        .data = out_norm.data,
        .shape = out_norm.shape,
        .strides = out_norm.strides,
        .offset = 0,
        .allocator = null,
        .owns_data = false,
    };
    norm.rmsNorm(f32, f32, row_view, w_view, eps, hbuf);
    // f16 bridge (lmHeadGemvQuant firma x: []const f16).
    for (h16, hbuf.data) |*d, s| d.* = @floatCast(s);
    // lm_head GEMV cuantizado + argmax con desempate por índice mínimo.
    embedding_mod.lmHeadGemvQuant(h16, lm_qw, logits_buf);
    var best: u32 = 0;
    var best_v: f32 = logits_buf[0];
    for (logits_buf[1..], 1..) |v, j| {
        if (v > best_v) {
            best_v = v;
            best = @intCast(j);
        }
    }
    return best;
}
