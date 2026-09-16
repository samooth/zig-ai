//! Módulo de inferencia CLI — extracción mecánica de runHybridInference
//! y sus helpers desde main.zig (T1 del plan server F2, lane-server-f2).
//!
//! ESTE FICHERO ES UN MOVIMIENTO EXACTO: los cuerpos de función son
//! byte-idénticos a los originales de main.zig. El gate de aceptación es
//! que el CLI produce la misma salida que antes de la extracción.
//!
//! La pub-ificación de los símbolos es lo único añadido (para que main.zig
//! delegue). T2+ (server) extraerá el estado a EngineState; este fichero
//! seguirá siendo el camino CLI.

const std = @import("std");
const Tensor = @import("core").Tensor;
const matmul = @import("matmul");
const fa = @import("fa");
const transformer = @import("transformer");
const kvcache = @import("kv_cache");

const FlashAttentionConfig = fa.fa_config.FlashAttentionConfig;
const pipeline = @import("pipeline");
const QuantFormat = kvcache.QuantFormat;
const gguf_model = @import("gguf_model");
const QuantWeight = @import("quant_weight").QuantWeight; // lane-f 7.1d wiring
const gguf_tokenizer = @import("gguf_tokenizer");
const bpe = @import("tokenizer");
const cudaz = @import("cudaz");
const cublas = @import("cublas");
const layer_kernels = @import("layer_kernels");
const ssm_mod = @import("ssm");
const gguf = @import("gguf");
const embedding = @import("embedding");
const hybrid_layer = @import("transformer");
const norm = @import("norm");
const paged_attn = @import("paged_attention");
const decode_graph = @import("decode_graph");
const layer_streamer = @import("layer_streamer");
const vram_budget = @import("vram_budget");
const model_config = @import("model_config");
const debugz = @import("debug");
// Post-freeze 2026-09-13: presupuesto CPU central para pools del pipeline
// (cpuLmHeadLogits y futuros) — ver utils/resources.zig.
const resources = @import("resources");
const specdrv = @import("speculative");
const gguf_moe = @import("gguf_moe");
const moe_layer = @import("moe_layer");
const moe_cuda = @import("moe_cuda");
const offload_cache = @import("offload_cache");
const kvarn_gpu_cache = @import("kvarn_gpu_cache"); // lane-f P3: kvarn M3 (trunk)
const mmproj_model = @import("mmproj_model"); // lane-f P3: vision multi-imagen (trunk)
const mmproj_config = @import("mmproj_config"); // lane-f P3
const vision_preprocess = @import("vision_preprocess"); // lane-f P3
const vision_encoder = @import("vision_clip_encoder"); // lane-f P3
const vision_clip_gpu = @import("vision_clip_gpu"); // lane-f P3
const vision_inject = @import("vision_token_inject"); // lane-f P3
const vision_video = @import("vision_video"); // lane-mmproj 10.7: --video ffmpeg decode
const host_bank = @import("host_bank"); // lane-f P3: copy-once 4.3'
const expert_bundle = @import("expert_bundle"); // lane-e 11.2: bundle contiguo
const cpu_executor = @import("cpu_executor"); // lane-f P3: Contrato 8
const moe_cpu_gemv = @import("moe_cpu_gemv"); // lane-f P3
const build_options = @import("build_options"); // lane-f P3: kvarn M3

/// Server F2 T2c: sink de streaming. Módulo `token_sink` (registrado en
/// build.zig; compartido por inference y server — sin ciclos).
pub const TokenSink = @import("token_sink").TokenSink;
const SinkFinishReason = @import("token_sink").FinishReason;

/// Parámetros de runtime compartidos CLI/engine (movidos de main.zig).
pub const CliParams = struct {
    model_path: ?[]const u8 = null,
    prompt: ?[]const u8 = null,
    rlt_sidecar_path: []const u8 = "",
    /// lane-kvc tANS C-a: archivo de texto para --ppl (perplexity, semántica
    /// llama.cpp: sliding window 2048 / stride 1024, PPL = exp(Σ log p / N)).
    ppl_file: ?[]const u8 = null,
    /// KT-B (lane-f): ruta a pesos .ktb del KV-transfer (flag --kv-transfer,
    /// env espejo ZIG_AI_KV_TRANSFER). Null = transfer desactivado.
    kv_transfer_path: ?[]const u8 = null,
    max_new_tokens: usize = 128,
    seed: u64 = 42,
    backend: []const u8 = "auto",
    sampler: pipeline.Sampler = .{},
    // Server Fase 2 (activación por main.zig; aquí no se consume).
    serve: bool = false,
    serve_host: []const u8 = "127.0.0.1",
    serve_port: u16 = 8080,
    /// API keys: file una-key-por-línea (permisos 0600 verificados).
    serve_api_key_file: ?[]const u8 = null,
    /// Audit log JSON-lines (fail-soft si no se puede abrir).
    serve_audit_log: ?[]const u8 = null,
    /// Rate limit requests/min por IP (default 60; comptime buckets T8).
    serve_rate_limit_rpm: u32 = 60,
    /// TLS: cert+key PEM. Bind no-loopback sin TLS es rechazado.
    serve_tls_cert: ?[]const u8 = null,
    serve_tls_key: ?[]const u8 = null,
    /// Warmup al arranque: 1 inferencia corta (default true en --serve).
    serve_warmup: bool = true,
    cache_type_k: kvcache.QuantFormat = .fp16,
    cache_type_v: kvcache.QuantFormat = .fp16,
    /// 9.1 (lane-c C-1): bits KVarN del KV body K (0 = off). Seteado por
    /// `-ctk kvarn<b>[v<b>]` — p.ej. kvarn4, kvarn4v6. El store KVarN
    /// (records por grupo de 128 tokens) suplanta al cuant por-bloque;
    /// cache_type_k queda como formato del BODY dual (fp16 recomendado).
    kvarn_k_bits: u8 = 0,
    /// 9.1 (lane-c C-1): bits KVarN de V (0 = off).
    kvarn_v_bits: u8 = 0,
    spec_draft_type_k: kvcache.QuantFormat = .fp16,
    spec_draft_type_v: kvcache.QuantFormat = .fp16,
    num_parallel: usize = 1,
    spec_type: SpecType = .none,
    spec_draft_n_max: usize = 16,
    spec_p_min: f32 = 0.1,
    /// 9.5: ProfitController (BeeLlama P1.1). `off` usa `spec_draft_n_max`
    /// estático; `profit` decide `n_max` dinámicamente por EWMA. Sin efecto
    /// cuando `spec_type == .none`.
    spec_dm_controller: specdrv.adaptive_dm.Controller = .profit,
    /// Re-baseline interval for adaptive controller (default 1024).
    spec_dm_baseline_interval: ?u32 = null,
    /// 9.6: LoopGuard (BeeLlama P1.2). `off` desactiva; `force-close`
    /// corta la generación; `warn` imprime advertencia y continúa.
    spec_loop_guard_mode: specdrv.loop_guard.Mode = .off,
    spec_loop_guard_window: u32 = 64,
    spec_loop_guard_max_period: u32 = 16,
    /// Loop guard channel (hidden|visible|both). Default hidden.
    spec_loop_guard_channel: ?[]const u8 = null,
    /// Preset INI path (--preset).
    preset_path: ?[]const u8 = null,
    /// Models directory (--models-dir).
    models_dir: ?[]const u8 = null,
    /// RLT (Recurrent Looped Transformer): force feedback merge ON/OFF.
    ///   null = auto (ON if GGUF has rlt.feedback_alpha > 0 + weights),
    ///   true = force ON, false = force OFF.
    ///   CLI: --rlt-feedback (ON), --no-rlt-feedback (OFF).
    rlt_feedback: ?bool = null,
    /// RLT: use recurrent (sequential) prefill instead of parallel batched.
    /// Slower but produces states identical to decode path.
    recurrent_prefill: bool = false,
    /// RLT: exact replay for speculative decoding — recompute all states from
    /// scratch instead of using snapshot/restore. Guarantees consistency after
    /// weight updates but is 5-20× slower. Default: OFF.
    exact_replay: bool = false,
    /// R-4 (dev RLT): captura hidden states reales para entrenamiento.
    ///   vacío = OFF; ruta = escribir `.rltcap` durante runHybridPpl (α=0).
    ///   CLI: --capture-rlt <path>.
    capture_rlt_path: []const u8 = "",
    /// R-4 (dev RLT): vuelca lm_head f32 (d×vocab) a disco para entrenamiento.
    ///   vacío = OFF; ruta = escribir .bin con f32 LE row-major.
    ///   CLI: --dump-lm-head <path>.
    dump_lm_head_path: []const u8 = "",
    /// R-4 (dev RLT): vuelca logits target (T×vocab f32 LE) a disco para
    /// entrenamiento RLT con MSE loss.
    ///   vacío = OFF; ruta = escribir .bin con f32 LE row-major.
    ///   CLI: --dump-logits-target <path>.
    dump_logits_target_path: []const u8 = "",
    /// P0-RPERF: bounded sliding-window attention per decoder layer.
    ///   0 = full context (default); >0 = cap window to this value.
    ///   CLI: --swa <n>.
    swa: usize = 0,
    /// Models preset path (--models-preset).
    models_preset_path: ?[]const u8 = null,
    use_jinja: bool = false,
    quant: QuantMode = .auto,
    batch_size: usize = 2048,
    ubatch_size: usize = 512,
    n_gpu_layers: ?usize = null,
    layer_stream: bool = false,
    layer_stream_max: usize = 2,
    /// 7.1a: máximo de capas con pesos f16 materializados (LRU eviction).
    f16_max_resident: usize = 2,
    context_length: usize = 65536,
    model_draft: []const u8 = "",
    spec_draft_block_size: usize = 0,
    spec_lookup_n: usize = 0,
    spec_selector_top_k: usize = 10,
    spec_selector_rank: usize = 128,
    download_dflash: bool = false,
    download_dspark: bool = false,
    download_dflash2: bool = false,
    // --- Vision (mmproj): encoder CLIP ViT dual-load (PLAN_MMPROJ Fase 0.4) ---
    /// mmproj GGUF path (llama.cpp --mmproj)
    mmproj_path: ?[]const u8 = null,
    /// Imagen de entrada para el encoder vision (llama.cpp --image)
    image_paths: std.ArrayList([]const u8) = .empty, // --image repetible (multi-imagen)
    /// Video de entrada para el encoder vision con merge temporal Conv3D
    /// (llama.cpp --video; lane-mmproj 10.7, Qwen3-VL temporal merge)
    video_paths: std.ArrayList([]const u8) = .empty, // --video repetible
};

pub const SpecType = specdrv.SpecType;
pub const QuantMode = enum { auto, off, fp8 };

/// Estimar tamaño comprimido por capa (aproximado desde tensores GGUF)
pub fn estimateCompressedWeightPerLayer(allocator: std.mem.Allocator, g: *gguf.GgufFile, cfg: model_config.ModelConfig) !usize {
    if (cfg.block_count == 0) return 0;
    // Bytes comprimidos por capa: UNA pasada por el índice de TENSORES
    // (los nombres de tensores NO viven en metadata — la versión anterior
    // escaneaba g.metadata y devolvía siempre 0, desactivando de facto el
    // auto-layer-stream y mandando los modelos grandes a carga eager → OOM).
    const per_layer = try allocator.alloc(usize, cfg.block_count);
    defer allocator.free(per_layer);
    @memset(per_layer, 0);
    var it = g.tensors.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        if (!std.mem.startsWith(u8, name, "blk.")) continue;
        const dot = std.mem.indexOfScalarPos(u8, name, 4, '.') orelse continue;
        const idx = std.fmt.parseInt(usize, name[4..dot], 10) catch continue;
        if (idx >= cfg.block_count) continue;
        per_layer[idx] += entry.value_ptr.*.dataBytes();
    }
    var total_bytes: usize = 0;
    for (per_layer) |b| total_bytes += b;
    return total_bytes / cfg.block_count;
}

/// B6 (lane-b): contexto del lm_head cuantizado on-load a q8_0.
/// Proyecta filas ya normalizadas: cuantiza la activación en host,
/// H2D de aq/ad, zero de C y GEMV device. El kernel B6 soporta M≤8
/// (accf[8]) para el verify batched de C4.3.
pub const LmQ80Ctx = struct {
    /// Techo M del kernel mmqQ8_0WGEMVKernel.
    const MAX_M: usize = 8;

    w_dev: usize,
    aq_dev: usize,
    ad_dev: usize,
    blk_host: []u8,
    x16_host: []f16,
    aq_host: []i8,
    ad_host: []f16,
    hidden: usize,
    vocab: usize,
    lk: *layer_kernels.LayerKernels,

    /// Batch máximo seguro según smem del kernel (aq alineado + escalas
    /// f32 por fila contra el límite práctico de 48KB por bloque).
    fn maxBatchFor(self: LmQ80Ctx) usize {
        const kb = self.hidden / 32;
        const per_row = self.hidden + kb * 4;
        return @max(1, @min(MAX_M, (48 * 1024) / per_row));
    }

    fn project(self: LmQ80Ctx, x_f32: []const f32, out_dev: usize) !void {
        return self.projectBatched(x_f32, 1, out_dev);
    }

    /// M filas contiguas [M][hidden] → C device [M][vocab] (el launcher
    /// hace el zero + atomics). El llamador trocea a maxBatchFor().
    fn projectBatched(self: LmQ80Ctx, x_rows: []const f32, m_in: usize, out_dev: usize) !void {
        std.debug.assert(m_in >= 1 and m_in <= MAX_M);
        const E = self.hidden;
        const kb = E / 32;
        for (0..m_in) |mi| {
            const row = x_rows[mi * E ..][0..E];
            for (row, self.x16_host) |v, *d| d.* = @floatCast(v);
            kvcache.kv_quant.encode(.q8_0, self.x16_host, self.blk_host);
            const aq_row = self.aq_host[mi * E ..][0..E];
            const ad_row = self.ad_host[mi * kb ..][0..kb];
            for (0..kb) |bi| {
                const src = self.blk_host[bi * 34 ..][0..34];
                ad_row[bi] = @bitCast(std.mem.readInt(u16, src[0..2], .little));
                for (0..32) |cc| aq_row[bi * 32 + cc] = @as(i8, @bitCast(src[2 + cc]));
            }
        }
        try cudaz.cuMemcpyHtoD(self.aq_dev, @intFromPtr(self.aq_host.ptr), m_in * E);
        try cudaz.cuMemcpyHtoD(self.ad_dev, @intFromPtr(self.ad_host.ptr), m_in * kb * @sizeOf(f16));
        try self.lk.mmqQ8_0WGEMV(self.aq_dev, self.ad_dev, self.w_dev, out_dev, m_in, E, self.vocab);
    }
};

/// RMSNorm CPU (espejo de norm.rmsNorm para buffers planos).
pub fn cpuRmsNormFlat(x: []const f32, gamma: []const f32, eps: f32, out: []f32) void {
    var ssq: f64 = 0;
    for (x) |v| ssq += @as(f64, v) * v;
    const inv: f32 = @floatCast(1.0 / @sqrt(ssq / @as(f64, @floatFromInt(x.len)) + eps));
    for (out, x, gamma) |*d, v, g| d.* = v * inv * g;
}

/// K2-Horizon: grouped RMSNorm flat (oráculo k2_horizon_group_rms_norm —
/// n_groups particiones del vector, rmsNorm independiente cada una, luego
/// mul(w); SIN el (1+w) Gemma-style). n_groups=1 ⇒ cpuRmsNormFlat.
pub fn cpuGroupedRmsNormFlat(x: []const f32, gamma: []const f32, eps: f32, n_groups: usize, out: []f32) void {
    if (n_groups <= 1) return cpuRmsNormFlat(x, gamma, eps, out);
    const dim = x.len;
    std.debug.assert(dim % n_groups == 0);
    const group_dim = dim / n_groups;
    var g_off: usize = 0;
    while (g_off < dim) : (g_off += group_dim) {
        const xs = x[g_off..][0..group_dim];
        const gs = gamma[g_off..][0..group_dim];
        const os = out[g_off..][0..group_dim];
        var ssq: f64 = 0;
        for (xs) |v| ssq += @as(f64, v) * v;
        const inv: f32 = @floatCast(1.0 / @sqrt(ssq / @as(f64, @floatFromInt(group_dim)) + eps));
        for (os, xs, gs) |*d, v, g| d.* = v * inv * g;
    }
}

/// RAM disponible en el host leyendo /proc/meminfo (0 si no se puede leer).
/// Evita morir por swap: la carga eager de un modelo grande dequantiza TODO
/// a f32 en el host y el OOM killer congela la máquina entera.
/// (2026-09-13) Implementación MOVIDA a utils/resources.zig — presupuesto
/// central post-freeze; esto es re-export para no romper los callers de
/// inference (engine_api, server...).
pub const hostMemAvailableBytes = resources.hostMemAvailableBytes;

/// Numel total (elems) de los tensores blk.{0..max_blk-1} — footprint f32
/// de la carga eager es numel×4.
pub fn totalBlkNumel(g: *gguf.GgufFile, max_blk: usize) usize {
    var total: usize = 0;
    var it = g.tensors.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        if (!std.mem.startsWith(u8, name, "blk.")) continue;
        const dot = std.mem.indexOfScalarPos(u8, name, 4, '.') orelse continue;
        const idx = std.fmt.parseInt(usize, name[4..dot], 10) catch continue;
        if (idx >= max_blk) continue;
        total += @intCast(entry.value_ptr.*.numel());
    }
    return total;
}

/// 7.1-fix (lane-f): estimador de footprint de RAM host en load.
/// Refleja los caminos REALES post-c2f63ef (patrón llama.cpp, GPU-first):
///
/// · Pesos: siempre bytes COMPRIMIDOS (QuantWeight mmap) — nunca
///   numel×f32 global. El f32/f16 dequant de un peso solo ocurre donde el
///   kernel qgemm no cubre su dtype.
/// · Atención (híbrido, backend cublas): si TODOS los pesos attn de una
///   capa tienen dtype qgemm (attnQuantResident) → 0 scratch host. Si no
///   (legacy/CPU), la capa materializa scratch f16 de atención.
/// · FFN (T1): scratch f32 solo si algún dtype FFN no cubre qgemm.
/// · Streaming activo (estimator se llama ANTES de decidirlo — para el
///   umbral se evalúa SIN streaming, conservador: suma total): el pico
///   real con streamer es max_resident capas materializadas.
/// · emb/lm_head: emb f16 + lm_head f32/f16 (según LMQ80) — trato
///   conservador: f32+f16 como el estimador viejo.
///
/// El resultado en bytes; se compara con host_avail×70% en el caller.
pub fn estimateHostLoadFootprint(g: *gguf.GgufFile, eff_blocks: usize, cfg: model_config.ModelConfig, backend: matmul.Backend, max_resident_hint: usize) usize {
    _ = max_resident_hint; // pico = suma (conservador); el streamer rebaja el real
    var weight_bytes: usize = 0;
    var ffn_f32_bytes: usize = 0;
    var attn_f16_bytes: usize = 0;
    const quant_ok = backend == .cublas and layer_kernels.quantPath();

    var it = g.tensors.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        const ti = entry.value_ptr.*;
        if (!std.mem.startsWith(u8, name, "blk.")) continue;
        const dot = std.mem.indexOfScalarPos(u8, name, 4, '.') orelse continue;
        const idx = std.fmt.parseInt(usize, name[4..dot], 10) catch continue;
        if (idx >= eff_blocks) continue;
        const nbytes: usize = ti.dataBytes();
        const numel: usize = @intCast(ti.numel());
        weight_bytes += nbytes;
        if (!quant_ok) continue;
        const dt = ti.dtype;
        // FFN: dtype no cubierto por qgemm ⇒ scratch f32 (T1 need_ffn_f32)
        if (std.mem.endsWith(u8, name, "ffn_gate.weight") or
            std.mem.endsWith(u8, name, "ffn_up.weight") or
            std.mem.endsWith(u8, name, "ffn_down.weight") or
            std.mem.endsWith(u8, name, "feed_forward.w1.weight") or
            std.mem.endsWith(u8, name, "feed_forward.w2.weight") or
            std.mem.endsWith(u8, name, "feed_forward.w3.weight"))
        {
            if (ssm_mod.SsmLayer.qgemmTypeFor(dt) == null) ffn_f32_bytes += numel * @sizeOf(f32);
            continue;
        }
        // Atención (híbrido): dtype cubierto ⇒ quant-residente (0 scratch);
        // no cubierto ⇒ scratch f16 (dequant on-the-fly del primer forward)
        if (std.mem.endsWith(u8, name, "attn_q.weight") or
            std.mem.endsWith(u8, name, "attn_k.weight") or
            std.mem.endsWith(u8, name, "attn_v.weight") or
            std.mem.endsWith(u8, name, "attn_output.weight"))
        {
            if (ssm_mod.SsmLayer.qgemmTypeFor(dt) == null) attn_f16_bytes += numel * @sizeOf(f16);
        }
    }
    // · emb/lm_head: loadEmbedding/loadLmHead dequantizan a f16 HOST
    //   (2× vocab×hidden×2B — lo que realmente se materializa; el término
    //   viejo f32+f16 sobreestimaba 3×). Con LMQ80 el lm_head cuantizado
    //   sustituye al f16 pero dejamos el f16 como conservador (se cachea).
    var emb_lm_head: usize = 2 * cfg.vocab_size * cfg.embedding_length * @sizeOf(f16);
    // output.weight en el GGUF puede ser más grande que vocab efectivo; usar
    // el tensor REAL si existe (bytes comprimidos ya contados arriba NO —
    // el loop solo cubre blk.*, así que emb/lm_head no está en weight_bytes).
    _ = &emb_lm_head;
    return weight_bytes + ffn_f32_bytes + attn_f16_bytes + emb_lm_head;
}

/// GEMV de un rango de filas [begin,end): out[j] = dot(row_j, x).
pub fn gemvRange(x: []const f32, head_data: []const f16, out: []f32, begin: usize, end: usize) void {
    const E = x.len;
    for (out[begin..end], begin..) |*oj, j| {
        const row = head_data[j * E ..][0..E];
        var acc: f32 = 0;
        for (row, 0..) |h16, i| acc += @as(f32, h16) * x[i];
        oj.* = acc;
    }
}

/// GEMV CPU de lm_head sobre la tabla f16 host: out[j] = dot(row_j, x).
/// Fallback para lm_head grandes no cuantizados (bf16/f16 ≥512MB) cuya
/// subida íntegra a device provoca OOM con capas residentes.
/// Paralelizado por rangos de filas (C4-tune T5): la fase draft del driver
/// especulativo llama esto k veces por ronda.
pub fn cpuLmHeadLogits(x: []const f32, head_data: []const f16, out: []f32) void {
    const V = out.len;
    // Presupuesto central (2026-09-13): antes getCpuCount() = lógicos;
    // el spec driver llama esto k veces/ronda y multiplicaba el freeze.
    var fba_buf: [256]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&fba_buf);
    const nth = resources.computeThreadBudget(fba.allocator(), "lm_head_gemv");
    if (nth <= 1 or V < 16384) {
        gemvRange(x, head_data, out, 0, V);
        return;
    }
    const chunk = (V + nth - 1) / nth;
    var handles: [128]std.Thread = undefined;
    const use = @min(@min(nth, (V + chunk - 1) / chunk), handles.len);
    var spawned: usize = 0;
    while (spawned < use - 1) : (spawned += 1) {
        const b = spawned * chunk;
        const e = @min(V, b + chunk);
        handles[spawned] = std.Thread.spawn(.{}, gemvRange, .{ x, head_data, out, b, e }) catch break;
    }
    gemvRange(x, head_data, out, spawned * chunk, V);
    for (handles[0..spawned]) |h| h.join();
}

pub fn formatPrompt(
    allocator: std.mem.Allocator,
    raw_prompt: []const u8,
    use_jinja: bool,
    gt: *const gguf_tokenizer.GgufTokenizer,
) ![]const u8 {
    if (!use_jinja) return try allocator.dupe(u8, raw_prompt);
    if (gt.chat_template == null) return try allocator.dupe(u8, raw_prompt);
    return gt.applyChatTemplate(allocator, raw_prompt, "You are a helpful assistant.") catch |err| {
        debugz.dbg.printLevel(.info, "[cli] chat template falló: {s}, usando prompt raw\n", .{@errorName(err)});
        return try allocator.dupe(u8, raw_prompt);
    };
}

/// Captura la secuencia GPU completa de un token de decode (embed H2D + capas
/// híbridas + rmsNorm final + lm_head) en un CUDA graph. Devuelve `false` en
/// cualquier error (auto-fallback al camino normal); en `true` el grafo queda
/// instanciado en `g.exec` y el estado ssm restaurado. La captura SIEMPRE se
/// termina (o aborta) aunque un nodo falle, para que el stream vuelva a modo
/// normal y el fallback pueda lanzar async ops.
pub fn captureDecodeGraph(
    g: *decode_graph.DecodeGraph,
    lk: *layer_kernels.LayerKernels,
    layers: []hybrid_layer.HybridLayer,
    g_cur: *cublas.GpuTensor(f32),
    g_nxt: *cublas.GpuTensor(f32),
    g_normed: *cublas.GpuTensor(f32),
    g_logits: *cublas.GpuTensor(f32),
    g_out_norm: *cublas.GpuBuffer(f32),
    engine: *matmul.MatmulEngine,
    allocator: std.mem.Allocator,
    n_embd: usize,
    vocab: usize,
    rms_eps: f32,
    current_pos: usize,
    state_parts: []const decode_graph.StatePart,
    lm_head_q4: bool,
    lm_head_q6k: bool,
    // lane-c (7.1d-regresión, 2026-09-12): con lm_head_cpu_fb/lmq80 el peso
    // f16 NO se captura — un cuMemAlloc de ~2GB dentro del capture window
    // = 901 en cascada + decode 10× más lento (repro 9B-UD-IQ2_M greedy:
    // 73s vs 12s con LMQ80_FORCE=1). Con head_off el grafo termina tras el
    // rmsNorm final; el replay hace D2H de g_normed + project (lmq80/CPU)
    // fuera del grafo (mismo camino que el decode no-graph, líneas ~3762).
    lm_head_off: bool,
    // lane-f F2A (LMQ40): bytes efectivos del lm_head cuantizado (q4_0
    // re-cuant on-load cuando LMQ40=1; q4_0/q6_k del GGUF si no).
    lm_head_bytes: []const u8,
    lm_head: anytype,
    // G2 (TODO 1.7): puntero device [1]i32 donde el argmax escribe el token
    // (0 = argmax desactivado ⇒ el grafo termina en los logits, como antes).
    // Debe venir YA allocado (lk.argmaxOut) — un cuMemAlloc dentro del
    // capture desactiva el grafo en silencio (lección TODO 1.3).
    argmax_out: usize,
    // 1.15 path-A (lane-a): true ⇒ el último nodo es sampleF32Gumbel
    // (temp>0) en vez de argmaxF32 (greedy). temp/penalty con los
    // flags del run; el ring de penalty vive en device ([0]=n) y lo
    // refresca el host por replay (ver cli decode loop).
    gumbel: bool,
    temp: f32,
    penalty: f32,
) bool {
    g.backupState(state_parts) catch return false;
    g.beginCapture() catch return false;

    var ok = true;
    cudaz.cuMemcpyHtoDAsync(g_cur.*.ptr(), @intFromPtr(g.embed_staging.ptr), n_embd * @sizeOf(f32), lk.stream) catch {
        debugz.dbg.printLevel(.info, "[graph_capture] falló embed H2D: {s}\n", .{@errorName(error.CudaError)});
        ok = false;
    };
    var c2g = g_cur.*;
    var n2g = g_nxt.*;
    if (ok) {
        for (layers, 0..) |*layer, i| {
            hybrid_layer.HybridLayer.forwardGPU(layer, lk, c2g, &n2g, current_pos, 1, null) catch |e| {
                debugz.dbg.printLevel(.info, "[graph_capture] falló capa {d}: {s}\n", .{ i, @errorName(e) });
                ok = false;
            };
            const tmp = c2g;
            c2g = n2g;
            n2g = tmp;
        }
    }
    if (ok) {
        lk.rmsNorm(c2g.ptr(), @intFromPtr(g_out_norm.*.dev_ptr), g_normed.*.ptr(), 1, n_embd, rms_eps) catch |e| {
            debugz.dbg.printLevel(.info, "[graph_capture] falló rmsNorm: {s}\n", .{@errorName(e)});
            ok = false;
        };
        if (ok and !lm_head_off) {
            if (lm_head_q4) {
                lk.q4gemmLinear(allocator, g_normed.*.ptr(), lm_head_bytes, g_logits.*.ptr(), n_embd, vocab) catch |e| {
                    debugz.dbg.printLevel(.info, "[graph_capture] falló lm_head q4: {s}\n", .{@errorName(e)});
                    ok = false;
                };
            } else if (lm_head_q6k) {
                lk.qgemmLinear(allocator, g_normed.*.ptr(), lm_head_bytes, g_logits.*.ptr(), 1, n_embd, vocab, 3) catch |e| {
                    debugz.dbg.printLevel(.info, "[graph_capture] falló lm_head q6k: {s}\n", .{@errorName(e)});
                    ok = false;
                };
            } else {
                engine.linearProjectionDeviceF16(g_normed.*, lm_head, g_logits, 1, n_embd, vocab) catch |e| {
                    debugz.dbg.printLevel(.info, "[graph_capture] falló lm_head f16: {s}\n", .{@errorName(e)});
                    ok = false;
                };
            }
        }
    }
    // G2 (TODO 1.7): argmax EN DEVICE como último nodo del grafo — así el
    // grafo termina produciendo el id del token y el único D2H del token es
    // de 4 bytes (en vez de vocab·4B ≈ 993 KB).
    // 1.15 path-A (lane-a): con temp>0 el último nodo es el sampleF32Gumbel
    // (misma salida [1]i32): sampleo softmax(temp) exacto. `gumbel` true ⇒
    // temp/penalty/ring con los flags del run; el counter philox avanza EN
    // DEVICE tras cada replay (replay-safe). El ring de penalty lo sube el
    // host ANTES de cada replay (memcpy capturable, contenido por-paso).
    if (ok and argmax_out != 0 and !lm_head_off) {
        if (gumbel) {
            lk.sampleF32Gumbel(g_logits.*.ptr(), argmax_out, temp, penalty, 1, vocab) catch |e| {
                debugz.dbg.printLevel(.info, "[graph_capture] falló gumbel: {s}\n", .{@errorName(e)});
                ok = false;
            };
        } else {
            lk.argmaxF32(g_logits.*.ptr(), argmax_out, 1, vocab) catch |e| {
                debugz.dbg.printLevel(.info, "[graph_capture] falló argmax: {s}\n", .{@errorName(e)});
                ok = false;
            };
        }
    }

    if (!ok) {
        // Nodo fallido dentro de la captura: terminarla/abortarla para volver
        // el stream a modo normal (auto-fallback).
        _ = g.endCapture();
        return false;
    }
    g.endCaptureAndInstantiate() catch return false;
    g.restoreState(state_parts) catch return false;
    return true;
}

/// 9.4 (lane-b) FA-native KVarN: trampoline de appendTokens que el
/// AttentionLayer llama tras proyectar K/V y antes de la atención native.
/// Devuelve la base de los descs K de la capa (V = base + sizeof(KvarnDesc))
/// para el fattn; el append garantiza que la capa esté alocada.
fn kvarnFaAppend(
    ctx: ?*anyopaque,
    layer: u32,
    k: cudaz.CUdeviceptr,
    v: ?cudaz.CUdeviceptr,
    n: u32,
    base: u32,
    stream: cudaz.CUstream,
) anyerror!cudaz.CUdeviceptr {
    const cache: *kvarn_gpu_cache.KvarnGpuCache = @ptrCast(@alignCast(ctx orelse return error.KvarnNoCtx));
    try cache.appendTokens(layer, k, v, n, base, null, stream);
    return cache.kDescs(layer);
}

/// Lane-KVC: basename sin extensión de un path (etiqueta de modelo en trazas).
fn basenameNoExt(buf: []u8, path: []const u8) []const u8 {
    const base = std.fs.path.basename(path);
    const stem = if (std.mem.lastIndexOfScalar(u8, base, '.')) |dot| base[0..dot] else base;
    if (stem.len >= buf.len) return stem;
    @memcpy(buf[0..stem.len], stem);
    return buf[0..stem.len];
}

/// Inferencia híbrida (SSM + atención, p.ej. Qwen3.5) usando HybridLayer.
pub fn runHybridInference(
    io: std.Io,
    allocator: std.mem.Allocator,
    model: *gguf_model.GgufModel,
    model_path: []const u8,
    params: CliParams,
    backend: matmul.Backend,
    stdout: anytype,
) !void {
    return runHybridInferenceSink(io, allocator, model, model_path, params, backend, stdout, null);
}

/// Igual que runHybridInference pero con TokenSink opcional (server F2 T2c).
/// sink=null ⇒ comportamiento byte-idéntico al CLI legacy (gate T1).
/// El sink recibe cada token generado (camino no-spec) con su detokenizado
/// incremental. NOTA: el camino especulativo (spec_active) emite los tokens
/// aceptados por ronda en batch al final de cada ronda.
const VisionInput = struct {
    /// Embeddings [n_tokens, proj_dim] (f32; proj_dim == n_embd del target)
    embeddings: []f32,
    n_tokens: usize,
    out_dim: usize,
    /// Grid final (tokens por eje) para pos-ids 2D del LLM
    grid_x: usize,
    grid_y: usize,
    /// TODO 10.7 (lane-mmproj): índice del PAR temporal de vídeo al que
    /// pertenece este input (null = imagen estática)
    video_chunk: ?usize = null,
    /// TODO 10.7: eje temporal vLLM-style — t = video_pos_0 + chunk·f
    /// segundos/par (0 = t constante, paridad llama.cpp)
    t_factor: f32 = 0,

    fn deinit(self: *VisionInput, allocator: std.mem.Allocator) void {
        allocator.free(self.embeddings);
        self.embeddings = &.{};
    }
};

/// Convierte VisionInput[] (embeddings + metadata) al shape ligero de
/// token_inject.ExpandInput (sin embeddings) para expandVisionTokens y
/// buildVisionPrompt. Caller libera. (lane-mmproj 10.7)
fn buildExpandInputs(allocator: std.mem.Allocator, imgs: []const VisionInput) ![]vision_inject.ExpandInput {
    const out = try allocator.alloc(vision_inject.ExpandInput, imgs.len);
    for (imgs, 0..) |*v, i| out[i] = .{
        .n_tokens = v.n_tokens,
        .grid_x = v.grid_x,
        .grid_y = v.grid_y,
        .video_chunk = v.video_chunk,
        .t_factor = v.t_factor,
    };
    return out;
}

/// Codifica las imágenes de --image (repetible) con --mmproj → lista de
/// VisionInput (embeddings listos para el prefill del target). El caller
/// posee cada `embeddings` (deinit). mmproj/encoder/pesos GPU se cargan UNA
/// vez y se reutilizan entre imágenes (amortización: el GpuClipEncoder
/// reusa los pesos residentes subidos en la primera imagen).
/// `stdout`: para trazabilidad. `n_embd`: dim del target (validación).
fn encodeVision(
    io: std.Io,
    allocator: std.mem.Allocator,
    params: CliParams,
    n_embd: usize,
    stdout: anytype,
) !?[]VisionInput {
    const mm_path = params.mmproj_path orelse return null;
    const img_paths = params.image_paths.items;
    const vid_paths = params.video_paths.items;
    if (img_paths.len == 0 and vid_paths.len == 0) return null;

    var mm = try mmproj_model.MmprojModel.load(io, allocator, mm_path);
    defer mm.deinit();

    // 10.2 (EXPLORADO 2026-09-02): backend cublas aquí sólo dio +3%
    // (83.2→80.4s encode 27-capas) — gemmCuBlasF32Resident sube/baja X/Y
    // por GEMM y los GEMMs del ViT son skinny (1200×1152): la latencia
    // PCIe+sync domina. La vía real es un encoder device-resident
    // (activaciones en GPU, tipo forwardGPU del LLM) — ver PLAN_MMPROJ 10.2.
    var vis_engine = try matmul.MatmulEngine.init(allocator, .parallel, .f32);
    defer vis_engine.deinit();

    var enc = try vision_encoder.ClipEncoder.init(allocator, &vis_engine, &mm);
    defer enc.deinit();

    // 10.2 device-resident (lane-mmproj): encode de los N blocks del ViT en
    // GPU (activaciones residentes, pesos subidos una vez, cuBLAS D2D — 14×
    // en 27 capas). Default: GPU si CUDA disponible. Overrides: MMPROJ_GPU=0
    // fuerza CPU; MMPROJ_GPU=1 fuerza GPU (fallback CPU si CUDA falla).
    // Multi-imagen: el GpuClipEncoder vive FUERA del loop — los pesos
    // subidos en la 1ª imagen quedan residentes para las siguientes (la
    // subida de 456MB sólo se paga una vez).
    const use_gpu_encode = blk: {
        const env = std.c.getenv("MMPROJ_GPU");
        if (env) |e| {
            break :blk e[0] != '0' and @import("cudaz").isCudaAvailable();
        }
        break :blk @import("cudaz").isCudaAvailable();
    };

    var gpu_enc: ?vision_clip_gpu.GpuClipEncoder = null;
    defer if (gpu_enc) |*g| g.deinit();
    var gpu_lk: ?layer_kernels.LayerKernels = null;
    defer if (gpu_lk) |*l| l.deinit();
    if (use_gpu_encode) {
        try stdout.print("[+] encode: GPU device-resident (default; MMPROJ_GPU=0 para CPU)\n", .{});
        try stdout.flush();
        gpu_lk = try layer_kernels.LayerKernels.init(@ptrCast((try matmul.MatmulEngine.sharedCudaStream()).raw));
    }

    var inputs: std.ArrayList(VisionInput) = .empty;
    errdefer {
        for (inputs.items) |*vin| vin.deinit(allocator);
        inputs.deinit(allocator);
    }

    // n_pos máx entre imágenes: el GpuClipEncoder se crea lazy con el n_pos
    // de la primera imagen (fromEncoder sube pesos + aloca activaciones) y
    // ensureN lo redimensiona si una imagen posterior es más grande.
    for (img_paths, 0..) |img_path, img_i| {
        try stdout.print("[*] Codificando imagen {d}/{d}: {s}\n", .{ img_i + 1, img_paths.len, img_path });
        try stdout.flush();

        var rgb = vision_preprocess.loadImage(allocator, img_path) catch |err| {
            try stdout.print("[!] mmproj: fallo al cargar imagen {s}: {s}\n", .{ img_path, @errorName(err) });
            try stdout.flush();
            return err;
        };
        defer rgb.deinit(allocator);
        try stdout.print("[+] imagen: {d}x{d} RGB\n", .{ rgb.width, rgb.height });
        try stdout.flush();

        const cfg_v = enc.cfg;
        const align_px = cfg_v.patch_size * cfg_v.spatial_merge_size;
        const target = vision_preprocess.smartResizeTarget(rgb.width, rgb.height, align_px, cfg_v.image_min_pixels, cfg_v.image_max_pixels);
        const n_pos = (target.h / cfg_v.patch_size) * (target.w / cfg_v.patch_size);
        const scratch = try allocator.alloc(f32, enc.scratchNeed(n_pos));
        defer allocator.free(scratch);

        if (use_gpu_encode) {
            if (gpu_enc == null) {
                gpu_enc = try vision_clip_gpu.GpuClipEncoder.fromEncoder(allocator, &gpu_lk.?, &enc, n_pos);
            } else {
                try gpu_enc.?.ensureN(n_pos);
            }
        }

        const encoded = if (gpu_enc) |*g|
            try enc.encodeGPU(allocator, rgb.data, rgb.width, rgb.height, g, scratch)
        else
            try enc.encode(allocator, rgb.data, rgb.width, rgb.height, scratch);

        try stdout.print("[+] encode ok: {d} tokens de imagen (grid {d}x{d}), dim={d}\n", .{
            encoded.n_tokens, encoded.grid_x, encoded.grid_y, encoded.out_dim,
        });
        try stdout.flush();

        if (encoded.out_dim != n_embd) {
            try stdout.print("[!] mmproj: dim de salida {d} != n_embd del target {d} (deepstack o projector no compatible)\n", .{ encoded.out_dim, n_embd });
            try stdout.flush();
            var e = encoded;
            e.deinit(allocator);
            if (inputs.items.len == 0) return null;
            return try inputs.toOwnedSlice(allocator);
        }

        // Transferencia de ownership de los embeddings al VisionInput
        try inputs.append(allocator, .{
            .embeddings = encoded.embeddings,
            .n_tokens = encoded.n_tokens,
            .out_dim = encoded.out_dim,
            .grid_x = encoded.grid_x,
            .grid_y = encoded.grid_y,
        });
    }

    // ── VIDEO (TODO 10.7, lane-mmproj): cada --video → frames ffmpeg →
    // PARES temporales → un VisionInput POR PAR (grid still-sized).
    // t_factor estilo vLLM (second_per_grid_ts·tokens_per_second,
    // qwen2_5_vl.py:1310-1313): default = frames-per-pair / fps =
    // segundos que cubre cada par. Override ZIG_AI_VIDEO_TFACTOR
    // (0 = t constante estilo llama.cpp).
    for (params.video_paths.items, 0..) |vid_path, vid_i| {
        try stdout.print("[*] Decodificando video {d}/{d}: {s}\n", .{ vid_i + 1, params.video_paths.items.len, vid_path });
        try stdout.flush();

        // fps de remuestreo: default 4.0 (oráculo mtmd-helper.h:127)
        var fps_target: f32 = 4.0;
        if (std.c.getenv("ZIG_AI_VIDEO_FPS")) |raw| {
            const parsed = std.fmt.parseFloat(f32, std.mem.span(raw)) catch 0;
            if (parsed > 0) fps_target = parsed;
        }

        var vf = vision_video.decode(io, allocator, vid_path, fps_target) catch |err| {
            try stdout.print("[!] video: fallo al decodificar {s}: {s}\n", .{ vid_path, @errorName(err) });
            try stdout.flush();
            return err;
        };
        defer vf.deinit(allocator);
        try stdout.print("[+] video: {d} frames {d}x{d} @ {d:.2}fps → {d} pares\n", .{ vf.frames.len, vf.info.width, vf.info.height, vf.info.fps, vf.nPairs() });
        try stdout.flush();

        const t_factor: f32 = blk: {
            if (std.c.getenv("ZIG_AI_VIDEO_TFACTOR")) |raw| {
                const parsed = std.fmt.parseFloat(f32, std.mem.span(raw)) catch 0;
                if (parsed > 0) break :blk parsed;
                break :blk 0; // 0 explícito = t constante
            }
            // default vLLM: cada par cubre 2 frames ⇒ segundos por par =
            // 2/fps_decode
            const fps_dec: f32 = if (fps_target > 0) fps_target else vf.info.fps;
            if (fps_dec > 0) break :blk 2.0 / fps_dec;
            break :blk 0;
        };

        for (0..vf.nPairs()) |pi| {
            const pr = vf.pair(pi).?;
            const n_pos_pair = blk2: {
                // n_pos del par = grid del resize (align patch·merge)
                const align_px = enc.cfg.patch_size * enc.cfg.spatial_merge_size;
                const tgt = vision_preprocess.smartResizeTarget(vf.info.width, vf.info.height, align_px, enc.cfg.image_min_pixels, enc.cfg.image_max_pixels);
                break :blk2 (tgt.h / enc.cfg.patch_size) * (tgt.w / enc.cfg.patch_size);
            };
            const scratch = try allocator.alloc(f32, enc.scratchNeed(n_pos_pair) + 4 * n_pos_pair * enc.n_embd);
            defer allocator.free(scratch);

            const encoded = if (gpu_enc) |*g|
                try enc.encodePairGPU(allocator, pr.f0, pr.f1, vf.info.width, vf.info.height, g, scratch)
            else
                try enc.encodePair(allocator, pr.f0, pr.f1, vf.info.width, vf.info.height, scratch);

            try stdout.print("[+] video: par {d}/{d} → {d} tokens (grid {d}x{d})\n", .{ pi + 1, vf.nPairs(), encoded.n_tokens, encoded.grid_x, encoded.grid_y });
            try stdout.flush();

            if (encoded.out_dim != n_embd) {
                try stdout.print("[!] video: dim {d} != n_embd {d}; video ignorado\n", .{ encoded.out_dim, n_embd });
                try stdout.flush();
                var e = encoded;
                e.deinit(allocator);
                break;
            }

            try inputs.append(allocator, .{
                .embeddings = encoded.embeddings,
                .n_tokens = encoded.n_tokens,
                .out_dim = encoded.out_dim,
                .grid_x = encoded.grid_x,
                .grid_y = encoded.grid_y,
                .video_chunk = pi,
                .t_factor = t_factor,
            });
        }
    }

    if (inputs.items.len == 0) return null;
    return try inputs.toOwnedSlice(allocator);
}

/// Marcadores vision reconocidos en el prompt raw (A.2). El BPE del
/// proyecto NO tiene manejo de special tokens (byte-level GPT-2 style) ⇒
/// `<|image_pad|>` se rompería en bytes. Aquí pre-split-eamos el prompt por
/// los marcadores y emitimos sus ids DIRECTOS (resueltos vía vocab lookup,
/// sin pasar por el BPE). Fiel al contracto mtmd: UNA posición de
/// <|image_pad|> por imagen (el caller la expande a n_tokens).
pub const vision_markers = [_][]const u8{
    "<|vision_start|>",
    "<|image_pad|>",
    "<|vision_end|>",
    "<|video_pad|>",
};

/// Encode del prompt con marcadores vision como tokens únicos.
/// Devuelve tokens asignados (caller libera) e `image_pad_idxs`: índices de
/// CADA <|image_pad|> en orden de aparición (multi-imagen; el caller empareja
/// con las --image por orden). Los otros marcadores se emiten como tokens
/// normales del vocab.
fn encodePromptWithVision(
    allocator: std.mem.Allocator,
    prompt: []const u8,
    tok: *bpe.BPETokenizer,
    gt: *const gguf_tokenizer.GgufTokenizer,
    stdout: anytype,
) !struct { ids: []u32, image_pad_idxs: []usize, video_pad_idxs: []usize } {
    // Buscar el primer marcador presente en el prompt
    var first: usize = prompt.len;
    var found = false;
    for (vision_markers) |m| {
        if (std.mem.indexOf(u8, prompt, m)) |i| {
            if (i < first) {
                first = i;
                found = true;
            }
        }
    }
    if (!found) {
        const ids = try tok.encode(prompt, .{});
        return .{ .ids = ids, .image_pad_idxs = &[_]usize{}, .video_pad_idxs = &[_]usize{} };
    }

    // Resolver ids de los marcadores vía vocab (lookup exacto)
    var m_ids: [vision_markers.len]?u32 = undefined;
    for (vision_markers, 0..) |m, i| {
        m_ids[i] = gt.lookup(m);
    }

    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(allocator);
    var pad_idxs: std.ArrayList(usize) = .empty;
    errdefer pad_idxs.deinit(allocator);
    var vpad_idxs: std.ArrayList(usize) = .empty;
    errdefer vpad_idxs.deinit(allocator);

    var rest = prompt;
    while (true) {
        // Buscar el marcador más cercano en `rest`
        var best: ?usize = null;
        var best_m: usize = 0;
        for (vision_markers, 0..) |m, i| {
            if (std.mem.indexOf(u8, rest, m)) |idx| {
                if (best == null or idx < best.?) {
                    best = idx;
                    best_m = i;
                }
            }
        }
        if (best == null) break;
        const idx = best.?;

        // Texto antes del marcador → BPE normal
        if (idx > 0) {
            const pre_ids = try tok.encode(rest[0..idx], .{});
            defer allocator.free(pre_ids);
            try out.appendSlice(allocator, pre_ids);
        }

        // Marcador → id directo
        if (m_ids[best_m]) |mid| {
            if (best_m == 1) { // <|image_pad|>
                try pad_idxs.append(allocator, out.items.len);
            }
            if (best_m == 3) { // <|video_pad|> (TODO 10.7, lane-mmproj)
                try vpad_idxs.append(allocator, out.items.len);
            }
            try out.append(allocator, mid);
        } else {
            try stdout.print("[!] vision: marcador {s} no está en el vocab del target; ignorado\n", .{vision_markers[best_m]});
            try stdout.flush();
        }

        rest = rest[idx + vision_markers[best_m].len ..];
    }
    // Cola tras el último marcador
    if (rest.len > 0) {
        const tail_ids = try tok.encode(rest, .{});
        defer allocator.free(tail_ids);
        try out.appendSlice(allocator, tail_ids);
    }

    return .{
        .ids = try out.toOwnedSlice(allocator),
        .image_pad_idxs = try pad_idxs.toOwnedSlice(allocator),
        .video_pad_idxs = try vpad_idxs.toOwnedSlice(allocator),
    };
}

fn printKvOomDiag(
    paged_kv: anytype,
    num_attn_layers: usize,
    block_size: usize,
    max_seq_len: usize,
    pos: usize,
    stdout: anytype,
) void {
    const ba = paged_kv.block_alloc;
    const free = ba.numFree();
    const total = ba.numTotal();
    const blocks_per_layer = if (num_attn_layers > 0) total / num_attn_layers else total;
    const real_ctx = blocks_per_layer * block_size;
    const phase = if (pos < 8) "prefill" else "decode";
    stdout.print(
        \\
        \\[OOM-KV] pool de bloques KV agotado en {s} (pos {d}/{d})
        \\[OOM-KV] pool: {d} bloques × {d}B — {d} libres, {d} en uso
        \\[OOM-KV] consumo: {d} capas attn × ~{d} bloques/seq ⇒ contexto REAL ≈ {d} tok
        \\[OOM-KV] causa: cap del pool aplicado en carga (max(512MB, 30% VRAM) — ver banner de arranque)
        \\[OOM-KV] sugerencias: --ctx-size {d} | -ctk q8_0 -ctv q8_0 (KV ÷2) | -ctk q4_0 -ctv q4_0 (KV ÷4)
        \\
    , .{
        phase,
        pos,
        max_seq_len,
        total,
        ba.block_bytes,
        free,
        total - free,
        num_attn_layers,
        blocks_per_layer,
        real_ctx,
        real_ctx,
    }) catch {};
}

/// Construye el VramEstimate para el presupuesto dinámico de VRAM.
fn buildVramEstimate(
    c: model_config.ModelConfig,
    compressed_wpl: usize,
    seq_len: usize,
    kv_fmt: QuantFormat,
    kv_block_size: usize,
    max_resident: usize,
    resident_fixed: usize,
    total_vram: usize,
) vram_budget.VramEstimate {
    // F5 (lane-f): términos que el presupuesto estático no modelaba.
    // (a) resident_fixed lo calcula el caller según el camino PLANNEADO del
    //     lm_head (q8_0 on-load / cuantizado / W_T f32 cacheado);
    // (b) pico transitorio por forward: mayor proyección densa materializada
    //     a f32 (FFN d_ff×d_model domina sobre SSM inner);
    // (c) cap del pool KV contiguo (anti-OOM host, 512MB).
    const E = c.embedding_length;
    _ = E;
    return .{
        .compressed_weight_per_layer = compressed_wpl,
        .num_layers = c.block_count,
        .num_attn_layers = blk: {
            var n: usize = 0;
            for (0..c.block_count) |i| {
                if (c.isFullAttentionLayer(i)) n += 1;
            }
            break :blk n;
        },
        .hidden_dim = c.embedding_length,
        .head_dim = if (c.head_dim > 0) c.head_dim else c.embedding_length / c.head_count,
        .num_kv_heads = c.head_count_kv,
        .feed_forward_dim = c.feed_forward_length,
        .max_seq_len = seq_len,
        .kv_quant = switch (kv_fmt) {
            .q8_0 => vram_budget.KVQuantFormat.q8_0,
            .q4_0 => vram_budget.KVQuantFormat.q4_0,
            .q4_1 => vram_budget.KVQuantFormat.q4_1,
            .q5_0 => vram_budget.KVQuantFormat.q5_0,
            .q5_1 => vram_budget.KVQuantFormat.q5_1,
            .q8_1 => vram_budget.KVQuantFormat.q8_1,
            .q2_k => vram_budget.KVQuantFormat.q2_k,
            .q3_k => vram_budget.KVQuantFormat.q3_k,
            .q4_k => vram_budget.KVQuantFormat.q4_k,
            .q5_k => vram_budget.KVQuantFormat.q5_k,
            .q6_k => vram_budget.KVQuantFormat.q6_k,
            .q8_k => vram_budget.KVQuantFormat.q8_k,
            .iq1_s => vram_budget.KVQuantFormat.iq1_s,
            .iq1_m => vram_budget.KVQuantFormat.iq1_m,
            .iq2_xxs => vram_budget.KVQuantFormat.iq2_xxs,
            .iq2_xs => vram_budget.KVQuantFormat.iq2_xs,
            .iq2_s => vram_budget.KVQuantFormat.iq2_s,
            .iq3_xxs => vram_budget.KVQuantFormat.iq3_xxs,
            .iq3_s => vram_budget.KVQuantFormat.iq3_s,
            .iq4_xs => vram_budget.KVQuantFormat.iq4_xs,
            .iq4_nl => vram_budget.KVQuantFormat.iq4_nl,
            .tq1_0 => vram_budget.KVQuantFormat.tq1_0,
            .tq2_0 => vram_budget.KVQuantFormat.tq2_0,
            .mxfp4 => vram_budget.KVQuantFormat.mxfp4,
            else => vram_budget.KVQuantFormat.fp16,
        },
        .block_size = kv_block_size,
        .max_resident = max_resident,
        // F5: residente fijo planneado (caller) + pico transitorio denso +
        // cap del pool contiguo. 4.11-a: espejo del cap de runHybridInference
        // (max(512MB, 30% VRAM total)) — si difieren, el auto-reduce aprueba
        // ctx que el pool luego trunca. 512MB floor sin CUDA.
        .resident_fixed_bytes = resident_fixed,
        .transient_peak_bytes = if (@max(c.feed_forward_length, c.ssm_inner_size) > 0)
            @max(c.feed_forward_length, c.ssm_inner_size) * c.embedding_length * 4
        else
            0,
        .contiguous_pool_cap = if (total_vram > 0)
            @max(512 * 1024 * 1024, total_vram * 30 / 100)
        else
            512 * 1024 * 1024,
    };
}

pub fn runHybridInferenceSink(
    io: std.Io,
    allocator: std.mem.Allocator,
    model: *gguf_model.GgufModel,
    model_path: []const u8,
    params: CliParams,
    backend: matmul.Backend,
    stdout: anytype,
    sink: ?TokenSink,
) !void {
    const cfg = model.config;
    const n_embd = cfg.embedding_length;
    const vocab = cfg.vocab_size;
    // 5.2 (lane-b1): path del sidecar DFlash cuando --spec-type draft-dflash*
    // (se resuelve en el bloque de descarga dual-load; la construcción del
    // modelo llega tras paged_kv — necesita el pool para su secuencia draft).
    var dflash_sidecar_path: []const u8 = "";
    var max_seq_len = if (params.context_length == 0) cfg.context_length else params.context_length;

    // 7.1d (lane-b1 @025461d + wiring lane-f): embedding cuant-residente.
    // La tabla f16 [vocab, hidden] (0.79GB en Llama-3.2-3B) NO se
    // materializa — embeddingLookupQuant dequantiza solo las filas de
    // los tokens on-demand desde el QuantWeight mmap (paridad bit-exacta,
    // tests/test_embedding_quant.zig 2/2). A/B con ZIG_AI_EMB_F16=1.
    var emb_f16: ?Tensor(f16) = null;
    defer if (emb_f16) |*t| t.deinit();
    const emb_quant = if (std.c.getenv("ZIG_AI_EMB_F16") == null)
        try model.loadEmbeddingQuant()
    else blk: {
        emb_f16 = try model.loadEmbedding();
        break :blk null;
    };
    debugz.dbg.printLevel(.info, "[milestone] emb cuant-residente (7.1d): {d} MB ahorrados vs f16\n", .{cfg.vocab_size * n_embd * @sizeOf(f16) / (1024 * 1024)});
    const Emb = struct {
        fn lookup(eq_: ?QuantWeight, ef_: ?Tensor(f16), tokens_: []const u32, bs_: usize, sl_: usize, out_: *Tensor(f16)) void {
            if (eq_) |*qw| {
                embedding.embeddingLookupQuant(qw, tokens_, bs_, sl_, out_);
            } else {
                embedding.embeddingLookup(ef_.?, tokens_, bs_, sl_, out_);
            }
        }
    };
    var lm_head = try model.loadLmHead();
    defer lm_head.deinit();
    debugz.dbg.printLevel(.info, "[milestone] lm_head cargado ({d} MB)\n", .{lm_head.data.len * @sizeOf(f16) / (1024 * 1024)});
    const lm_head_q = try model.loadLmHeadQuant();
    // 2.2 (lane-f): con --quant fp8 el engine va por fp8_block y los kernels
    // qgemm (q4_0/q6_k/...) deben APAGARSE — de lo contrario el path SSM/attn
    // cuantizaría los pesos GGUF de nuevo (doble cuantización).
    layer_kernels.quant_enabled = params.quant != .off and params.quant != .fp8;
    const lm_head_q4 = lm_head_q.dtype() == gguf.GgmlType.q4_0 and layer_kernels.quantPath();
    const lm_head_q6k = lm_head_q.dtype() == gguf.GgmlType.q6_k and layer_kernels.quantPath();
    // LMQ40 (lane-f F2A, TODO 1.5): re-cuant on-load q6_k→q4_0 del lm_head.
    // PERF_SSM 2026-09-03: lm_head = 14.6% del token (834µs, 250 GB/s efectivos
    // en q6_k canónico); q4_0 = 143MB @ mmq split-K (LMSPLIT) ⇒ objetivo
    // ~500-600µs. Gate OPT-IN LMQ40=1 (A/B; default intacto). La calidad de
    // logits se valida con el gate de texto greedy + battery de prompts.
    var lmq40: ?struct {
        bytes: []u8,
        vocab: usize,
        hidden: usize,
    } = null;
    defer if (lmq40) |*q| allocator.free(q.bytes);
    if (std.c.getenv("LMQ40") != null and lm_head_q6k) {
        const r40 = model.loadLmHeadQ40(allocator) catch |e| blk: {
            debugz.dbg.printLevel(.info, "[milestone] LMQ40 on-load falló ({any}) → camino q6_k intacto\n", .{e});
            break :blk null;
        };
        if (r40) |r| {
            lmq40 = .{ .bytes = r.bytes, .vocab = r.vocab, .hidden = r.hidden };
            try stdout.print("[+] LMQ40 activo: lm_head q6_k→q4_0 on-load ({d} MB vs {d} MB)\n", .{ r.bytes.len / (1024 * 1024), lm_head_q.bytes.len / (1024 * 1024) });
        }
    }
    // lm_head GRANDE no-cuantizado (p.ej. bf16 en Q8_K_XL): subirlo entero a
    // device son ~2.4GB ⇒ OOM con capa residente. Fallback: GEMV en CPU sobre
    // la tabla f16 host (≈1.3 GFLOP, sub-segundo) — lane-c, ver HANDOFFS
    // 11:16. Se puede forzar device con SPEC_CPU_LMHEAD=0.
    const spec_cpu_lmhead_env = std.c.getenv("SPEC_CPU_LMHEAD");
    // LMQ80_FORCE: 1 = activa el camino q8_0 on-load aunque la cabeza quepa
    // en device (testing A/B del GEMV B6 y del batched de verify; equivale a
    // un --lm-head-q80 de facto hasta CLI). 0 = desactiva q80 y fuerza el
    // fallback CPU-GEMV original (para medir el coste del camino device).
    const lmq80_env = std.c.getenv("LMQ80_FORCE");
    const lmq80_force_on = lmq80_env != null and lmq80_env.?[0] == '1';
    const lmq80_force_off = lmq80_env != null and lmq80_env.?[0] == '0';
    const lm_head_cpu_fb = (!lm_head_q4 and !lm_head_q6k) and
        ((lm_head.data.len * @sizeOf(f16) >= 512 * 1024 * 1024 and
            (spec_cpu_lmhead_env == null or spec_cpu_lmhead_env.?[0] != '0')) or lmq80_force_on);
    if (lm_head_cpu_fb and debugz.dbg.at(.info)) {
        debugz.dbg.printLevel(.info, "[milestone] lm_head CPU-fallback activo ({d} MB f16)\n", .{lm_head.data.len * @sizeOf(f16) / (1024 * 1024)});
    }
    // F5 (lane-f): residente fijo planneado según el camino del lm_head —
    // q8_0 on-load (~34B/row-block), cuantizado q4/q6k, o el W_T f32 que
    // weight_cache deja residente en el camino f16-device. Lo consume
    // buildVramEstimate (auto-presupuesto y auto-layer-stream).
    const lm_head_resident_plan: usize = blk: {
        if (lm_head_cpu_fb and !lmq80_force_off)
            break :blk cfg.vocab_size * ((cfg.embedding_length + 31) / 32) * 34;
        // lane-f F2A: con LMQ40 el residente es el buffer q4_0 on-load.
        if (lmq40) |*q| break :blk q.bytes.len;
        if (lm_head_q4 or lm_head_q6k) break :blk lm_head_q.bytes.len;
        break :blk cfg.vocab_size * cfg.embedding_length * 4;
    };
    var out_norm = try model.loadOutputNorm();
    defer out_norm.deinit();
    const rms_eps = cfg.layer_norm_rms_epsilon;

    // ── Vision (PLAN_MMPROJ 3.2): encode de --image (repetible) con el mmproj ──
    const vision_in: ?[]VisionInput = encodeVision(io, allocator, params, n_embd, stdout) catch |err| {
        try stdout.print("[!] mmproj: fallo en el encoder vision: {s}\n", .{@errorName(err)});
        try stdout.flush();
        return err;
    };
    defer if (vision_in) |list| {
        for (list) |*vin| vin.deinit(allocator);
        allocator.free(list);
    };
    const vision: ?[]VisionInput = vision_in;

    // Capas híbridas: cada una enruta SSM vs atención según isFullAttentionLayer.
    // KV-cache paginado compartido: cada capa de atención obtiene su propio block_table.
    const head_dim = if (cfg.head_dim > 0) cfg.head_dim else n_embd / cfg.head_count;
    const block_size: usize = 16;

    // ─── Cabeza MTP: excluir del camino de inferencia normal ────────────────
    // qwen35.block_count INCLUYE la(s) capa(s) nextn (blk.{N}.nextn.*, N ==
    // block_count - predict_layers .. block_count-1). Esas capas son el draft
    // head de la especulativa (C4): no participan en el forward del target y
    // su patrón attn/SSM no sigue el intervalo periódico — intentar
    // cargarlas como capa normal muere con WeightFileNotFound (27B NEO-MTP,
    // corrida larga 2026-08-24). Hasta integrar C4 se excluyen.
    const mtp_info = specdrv.detectMtp(&model.file, cfg.block_count);
    const eff_blocks = if (mtp_info) |m| m.layer_idx else cfg.block_count;
    if (mtp_info != null) {
        debugz.dbg.printLevel(.info, "[milestone] MTP: {d} capas target + nexten en blk.{d}..{d}\n", .{ eff_blocks, eff_blocks, cfg.block_count - 1 });
        try stdout.print("[+] Cabeza MTP detectada en blk.{d}..{d}: excluida del forward target ({d} capas efectivas)\n", .{ eff_blocks, cfg.block_count - 1, eff_blocks });
        try stdout.flush();
    }

    var num_attn_layers: usize = 0;
    for (0..eff_blocks) |i| {
        if (cfg.isFullAttentionLayer(i)) num_attn_layers += 1;
    }

    // ─── Lane-KVC (paso 1 KV-Codec §A1): tracer de K/V ─────────────────────
    // DUMP_KV_TRACE=<dir> + DUMP_KV_TRACE_CORPUS=<prose|code|...>: captura
    // k_pre/k_post/v/q_tail del forward CPU (f16). Coste cero si unset.
    var kv_tracer: ?kvcache.kv_trace.KvTrace = null;
    var kv_tracer_corpus_buf: [16]u8 = undefined;
    var io_mut = io;
    if (debugz.dbg.kv_trace_dir) |trace_dir| {
        const corpus_label = blk: {
            const v = std.c.getenv("DUMP_KV_TRACE_CORPUS");
            break :blk if (v) |cv| std.fmt.bufPrint(&kv_tracer_corpus_buf, "{s}", .{std.mem.span(cv)}) catch "prompt" else "prompt";
        };
        var model_label_buf: [128]u8 = undefined;
        const model_label = basenameNoExt(&model_label_buf, model_path);
        kv_tracer = try kvcache.kv_trace.KvTrace.init(
            allocator,
            &io_mut,
            trace_dir,
            model_label,
            corpus_label,
            cfg.block_count,
            cfg.head_count_kv,
            head_dim,
            cfg.head_count,
        );
        if (kv_tracer) |*tr| {
            tr.lane_base_sha = "1e46c31";
            kvcache.kv_trace.hook.setTracer(tr, allocator);
            try stdout.print("[+] KV-TRACE: captura activa → {s} (corpus={s}, {d} capas attn)\n", .{ trace_dir, corpus_label, num_attn_layers });
            try stdout.flush();
        }
    }
    defer if (kv_tracer) |*tr| {
        tr.finish() catch |err| {
            debugz.dbg.print("[kv_trace] [KV-TRACE] finish fallo: {s}\n", .{@errorName(err)});
        };
        kvcache.kv_trace.hook.setTracer(null, allocator);
    };

    // ─── Pipeline KV en GPU: fuente única de verdad ──────────────────────────
    // gpuKvPipelineReady() (ámbito de archivo): un formato está listo sólo con
    // TODO su pipeline device (decode fusionado + append/prefill cuantizado).
    // Hoy sólo fp16 cumple: kvAppendF16 y prefill device siguen siendo f16.
    const effective_cache_type_k: QuantFormat =
        if (gpuKvPipelineReady(params.cache_type_k)) params.cache_type_k else .fp16;
    const effective_cache_type_v: QuantFormat =
        if (gpuKvPipelineReady(params.cache_type_v)) params.cache_type_v else .fp16;
    if (effective_cache_type_k != params.cache_type_k or effective_cache_type_v != params.cache_type_v) {
        try stdout.print("[!] KV -ctk {s} / -ctv {s}: decode GPU no validado E2E (cuarentena) o f16-only → usando fp16 (KVFORCE=1 para forzar)\n", .{
            @tagName(params.cache_type_k), @tagName(params.cache_type_v),
        });
    }

    // ─── Auto-presupuesto VRAM: reduce contexto y decide offload KV ──────────
    var total_vram_cached: usize = 0;
    var enable_cpu_offload_kv = false;
    if (backend == .cublas) {
        total_vram_cached = cudaz.getDeviceTotalMem(cudaz.cuDeviceGet(0) catch 0) catch 0;
    }
    if (total_vram_cached > 0) {
        const wpl = estimateCompressedWeightPerLayer(allocator, &model.file, cfg) catch 50 * 1024 * 1024;
        var ctx = max_seq_len;
        var breakdown: vram_budget.VramBreakdown = undefined;
        while (true) {
            breakdown = vram_budget.estimateTotalVram(buildVramEstimate(cfg, wpl, ctx, effective_cache_type_k, block_size, params.layer_stream_max, lm_head_resident_plan, total_vram_cached));
            if (breakdown.total_vram <= total_vram_cached * 85 / 100 or ctx <= 1024) break;
            ctx = @max(@as(usize, 1024), ctx * 3 / 4);
        }
        if (ctx != max_seq_len) {
            try stdout.print("[+] Contexto auto-reducido: {d} → {d} tokens (presupuesto VRAM)\n", .{ max_seq_len, ctx });
        }
        max_seq_len = ctx;

        enable_cpu_offload_kv = breakdown.kv_vram > total_vram_cached * 40 / 100;
        if (enable_cpu_offload_kv and debugz.dbg.at(.info)) {
            try stdout.print("[+] CPU offload KV candidato (kv={d}MB > 40% de VRAM)\n", .{breakdown.kv_vram / (1024 * 1024)});
        }
    }

    if (max_seq_len != cfg.context_length) {
        try stdout.print("[+] Contexto efectivo: {d} (modelo entrenado: {d})\n", .{ max_seq_len, cfg.context_length });
        if (max_seq_len > cfg.context_length) {
            try stdout.print("[!] Contexto mayor que entrenado ({d} > {d}) — extrapolando RoPE\n", .{ max_seq_len, cfg.context_length });
        }
    }
    try stdout.print("[+] arch={s} capas={d} heads={d} kv={d} emb={d} ffn={d} vocab={d} ctx={d} (path paged)\n", .{
        cfg.architecture,     eff_blocks,              cfg.head_count, cfg.head_count_kv,
        cfg.embedding_length, cfg.feed_forward_length, cfg.vocab_size, max_seq_len,
    });

    const blocks_per_seq: usize = (max_seq_len + block_size - 1) / block_size;
    const num_blocks_raw = @max(64, num_attn_layers * blocks_per_seq);
    // ─── Cap del pool KV device (lane-c) ─────────────────────────────────────
    // Sin VMM el pool es UN solo cudaMalloc contiguo de num_blocks×block_bytes:
    // con ctx grande eso son GBs y revienta. Presupuesto del pool device;
    // el contexto real lo limita el auto-reduce de arriba.
    // fp16: block_bytes = bs × kv_dim × 2B × 2(K,V).
    //
    // 4.11-a (lane-f, @2026-09-05): el cap 512MB FIJO contradecía al
    // presupuesto F5: el auto-reduce valida contra kv_vram YA capped
    // (512MB) ⇒ aprobaba ctx 65536 y luego el pool truncaba los bloques
    // a 16384 ⇒ contexto REAL ~2730 tok/seq silencioso (Qwen3.5-0.8B:
    // 24576→16384). Fix: cap = max(512MB floor, 30% VRAM total) — 3080
    // 7.7GB ⇒ ~2.3GB ⇒ ctx 65536 (1.5GB KV) entra COMPLETO. El
    // auto-reduce sigue siendo el guardián (85% VRAM). Aviso P5 a lane-c
    // (dueño del cap original): mismo valor en buildVramEstimate.
    const kv_dim_est = cfg.head_count_kv * head_dim;
    const block_bytes_est = block_size * kv_dim_est * 2 * 2;
    const kv_pool_cap_bytes = if (total_vram_cached > 0)
        @max(512 * 1024 * 1024, total_vram_cached * 30 / 100)
    else
        512 * 1024 * 1024;
    const kv_pool_cap_blocks = @max(64, kv_pool_cap_bytes / @max(1, block_bytes_est));
    const num_blocks = @min(num_blocks_raw, kv_pool_cap_blocks);
    debugz.dbg.printLevel(.info, "[ppl] KV pool cap: raw={d} vram_cap={d} final={d} bpb={d}\n", .{ num_blocks_raw, kv_pool_cap_blocks, num_blocks, block_bytes_est });
    if (num_blocks != num_blocks_raw) {
        debugz.dbg.printLevel(.info, "[milestone] pool KV limitado: {d}→{d} bloques (cap {d}MB device)\n", .{ num_blocks_raw, num_blocks, kv_pool_cap_bytes / (1024 * 1024) });
        // OOM-diag (lane-f, 4.11-a follow-up): elevar SIEMPRE a stdout —
        // antes este truncado era silencioso y el error.OutOfMemory llegaba
        // a mitad de generación sin causa raíz. Contexto real estimado:
        // bloques/capa × block_size (pool compartido por las capas attn).
        if (num_attn_layers > 0) {
            const real_ctx = (num_blocks / num_attn_layers) * block_size;
            try stdout.print("[!] Pool KV limitado: {d}→{d} bloques ({d}MB) ⇒ contexto REAL ≈ {d} tok (pediste {d})\n", .{ num_blocks_raw, num_blocks, kv_pool_cap_bytes / (1024 * 1024), real_ctx, max_seq_len });
            if (real_ctx < max_seq_len) {
                try stdout.print("    Sugerencias: --ctx-size {d} | -ctk q8_0 -ctv q8_0 (KV ÷2) | -ctk q4_0 -ctv q4_0 (KV ÷4)\n", .{real_ctx});
            }
        }
    }

    // R-2 (dev RLT): --exact-replay activa replay determinista sin draft
    // cache (consistencia tras weight update).
    if (params.exact_replay) {
        debugz.dbg.no_graph = true;
        try stdout.print("[+] exact-replay: replay determinista (NOGRAPH, sin draft)\n", .{});
    }

    // --- Spec-decoding: validar --spec-type draft-mtp contra el GGUF ---
    // La detección real (blk.{N}.nextn) ya se hizo arriba vía specdrv.detectMtp
    // para excluir las capas nextn del forward; aquí sólo validamos la petición.
    if (params.spec_type == .draft_mtp and mtp_info == null) {
        try stdout.print(
            \\
            \\[!] --spec-type draft-mtp solicitado pero el modelo NO contiene cabeza MTP
            \\    (`nextn.*` no encontrado en {s}). La decodificación especulativa
            \\    requiere un modelo con cabeza MTP; no puede ejecutarse con este modelo.
            \\    Continue sin --spec-type o use un modelo con cabeza MTP.
            \\
        , .{model.config.architecture});
        try stdout.flush();
        return;
    }

    // ─── C6-infra: sidecar DFlash/DSpark/DFlash2 (downloader + dual-load) ────
    // Resuelve el GGUF del drafter (--model-draft o --download-* con la
    // convención sibling de llama.cpp); el dual-load y la construcción del
    // DflashDraftModel (encoder + kvInject + denoise) corren tras el
    // paged_kv (5.2 lane-b1). [Fase A GPU device-resident preservada @b7cd73e.]
    if (params.spec_type == .draft_dflash or params.spec_type == .draft_dspark or params.spec_type == .draft_dflash2) {
        const variant: specdrv.hf_download.Variant = switch (params.spec_type) {
            .draft_dflash => .dflash,
            .draft_dspark => .dspark,
            .draft_dflash2 => .dflash2,
            else => unreachable,
        };
        var sc_path: []const u8 = params.model_draft;
        var sc_downloaded = false;
        // 5.2: si el path pasa a dflash_sidecar_path, la ownership del alloc
        // descargado se transfiere (el consumidor lo libera con params-derived
        // lifetime — model_draft de params vive todo el run; el descargado se
        // marca sc_path_keep para evitar el double-free del defer).
        var sc_path_keep = false;
        defer if (sc_downloaded and !sc_path_keep) allocator.free(sc_path);
        const want_dl = switch (variant) {
            .dflash => params.download_dflash,
            .dspark => params.download_dspark,
            .dflash2 => params.download_dflash2,
        };
        if (want_dl and sc_path.len == 0) {
            const ref = params.model_path orelse "";
            if (ref.len == 0) {
                try stdout.print("[!] --download-* requiere -m <modelo> (local con sibling o ref HF)\n", .{});
                return;
            }
            sc_path = specdrv.hf_download.ensureSidecar(io, allocator, ref, variant) catch |e| {
                try stdout.print("[!] descarga de sidecar {s} falló: {s} (¿repo/sibling correcto?)\n", .{ @tagName(variant), @errorName(e) });
                return;
            };
            sc_downloaded = true;
            try stdout.print("[+] sidecar {s} listo: {s}\n", .{ @tagName(variant), sc_path });
        }
        if (sc_path.len == 0) {
            try stdout.print("[!] --spec-type {s} requiere --model-draft <sidecar.gguf> (o --download-{s})\n", .{ @tagName(params.spec_type), @tagName(variant) });
            return;
        }
        // 5.2 (lane-b1, arbitraje coordinador 01:4x): el dual-load es DIFERIDO
        // al punto con paged_kv — aquí registramos path y variante; la
        // construcción del DflashDraftModel (encoder + kvInject + denoise)
        // va tras el pool. [El camino device-resident alternativo (Fase A
        // GPU: DflashEncoder + capas sidecar como HybridLayers con seq
        // propia + KV-inject tras prefill, tap-capture DtoD por
        // target_layer) está preservado en @b7cd73e — reutilizable para el
        // denoise GPU de la Fase B.]
        dflash_sidecar_path = sc_path;
        sc_path_keep = sc_downloaded;
    }

    // Driver especulativo + cabeza MTP (C4.1): carga de pesos nextn vía mmap.
    // El bucle draft→verify→accept se enchufa en el decode loop (C4.2).
    var spec_driver = specdrv.SpecDriver.init(allocator, .{
        .spec_type = params.spec_type,
        .n_max = params.spec_draft_n_max,
        .p_min = params.spec_p_min,
        .seed = params.seed,
    });
    defer spec_driver.deinit();
    var mtp_head: ?specdrv.MtpHead = null;
    defer if (mtp_head) |*h| h.deinit(allocator);
    if (params.spec_type == .draft_mtp) {
        mtp_head = specdrv.MtpHead.load(allocator, &model.file, mtp_info.?, n_embd) catch |e| {
            try stdout.print("[!] fallo cargando cabeza MTP: {s}\n", .{@errorName(e)});
            return e;
        };
        debugz.dbg.printLevel(.info, "[milestone] MtpHead lista: eh_proj dtype={s}\n", .{@tagName(mtp_head.?.eh_proj.dtype())});
        try stdout.print("[*] draft-mtp activo: cabeza blk.{d} cargada, n_max={d}, p_min={d}\n", .{ mtp_info.?.layer_idx, params.spec_draft_n_max, params.spec_p_min });
        try stdout.flush();
    }

    // Auto-enable layer streaming if model doesn't fit in VRAM
    // (reusa total_vram_cached / buildVramEstimate del presupuesto inicial)
    var auto_layer_stream = params.layer_stream;
    var auto_max_resident = params.layer_stream_max;
    // 7.1b: señal de OOM-host sobrevive al scope de detección — el streamer
    // la necesita para NO rescatar el grafo (ver comentario en max_resident).
    var force_stream_host = false;
    if (backend == .cublas) {
        const total_vram = total_vram_cached;
        if (total_vram > 0) {
            const estimated_weight_per_layer = estimateCompressedWeightPerLayer(allocator, &model.file, cfg) catch 50 * 1024 * 1024;
            debugz.dbg.printLevel(.info, "[milestone] wpl estimado={d} MB ctx={d}\n", .{ estimated_weight_per_layer / (1024 * 1024), max_seq_len });
            const vram_estimate = buildVramEstimate(cfg, estimated_weight_per_layer, max_seq_len, effective_cache_type_k, block_size, params.layer_stream_max, lm_head_resident_plan, total_vram);
            const breakdown = vram_budget.estimateTotalVram(vram_estimate);
            debugz.dbg.printLevel(.info, "[milestone] estimate total={} MB vs umbral={} MB\n", .{ breakdown.total_vram / (1024 * 1024), total_vram * 85 / 100 / (1024 * 1024) });

            // ── Guardia anti-OOM HOST ─────────────────────────────────────
            // 7.1-fix (lane-f, 2026-09-08): el estimador anterior usaba
            // eager_f32 = totalBlkNumel×4B (peor caso del camino LEGACY):
            // 9B Q4_K_M → 38GB "necesarios" vs 20GB RAM → streaming forzado
            // AUNQUE el modelo (5.7GB cuantizado) cabe entero en VRAM.
            // Patrón llama.cpp (load_tensors: GPU-first, fallback CPU
            // buffer): los pesos QUANTIZADOS van a VRAM sin dequant f32
            // global — el f32 solo se materializa donde el qgemm no cubre
            // el dtype. Post-c2f63ef el híbrido ES quant-residente para
            // atención (attnQuantResident) y FFN (T1: need_ffn_f32 por
            // capa). Estimador nuevo = suma REAL de lo que va a host:
            //   · scratch f32 solo de capas con FFN dtype no-qgemm
            //   · scratch f16 de atención solo si el camino híbrido NO es
            //     quant-residente (fallback legacy)
            //   · pesos siempre como bytes comprimidos (mmap, = archivo)
            // Bajo streaming, además, solo max_resident capas viven
            // materializadas a la vez — el pico es el techo, no la suma.
            const host_avail = hostMemAvailableBytes();
            const host_need = estimateHostLoadFootprint(&model.file, eff_blocks, cfg, backend, params.layer_stream_max);
            if (host_avail > 0) {
                const need_mb = host_need / (1024 * 1024);
                const avail_mb = host_avail / (1024 * 1024);
                debugz.dbg.printLevel(.info, "[milestone] host_need={d} MB vs disponible={d} MB\n", .{ need_mb, avail_mb });
                if (host_need * 100 > host_avail * 70) {
                    force_stream_host = true;
                    auto_layer_stream = true;
                    auto_max_resident = @max(1, params.layer_stream_max);
                    try stdout.print("[!] Prevención OOM host: carga estimada ~{d} MB vs {d} MB disponibles → forzando layer-streaming\n", .{ need_mb, avail_mb });
                    try stdout.flush();
                }
            }

            if (force_stream_host or breakdown.total_vram > total_vram * 85 / 100) {
                if (!force_stream_host) {
                    auto_layer_stream = true;
                    auto_max_resident = @max(1, params.layer_stream_max);
                    if (vram_budget.suggestLayerStreamConfig(total_vram, vram_estimate)) |suggested| {
                        auto_max_resident = suggested;
                    }
                    try stdout.print("[+] Auto-enabled layer streaming: estimated VRAM {} MB > available {} MB (max_resident={d})\n", .{
                        breakdown.total_vram / (1024 * 1024), total_vram / (1024 * 1024), auto_max_resident,
                    });
                    try stdout.flush();
                }
            } else if (debugz.dbg.at(.info)) {
                try stdout.print("[+] VRAM estimate: {} MB / {} MB available (fits)\n", .{
                    breakdown.total_vram / (1024 * 1024), total_vram / (1024 * 1024),
                });
            }
        }
    }

    var paged_kv = try paged_attn.PagedKVCache.init(allocator, .{
        .block_size = block_size,
        .num_blocks = num_blocks,
        .head_dim = head_dim,
        .num_kv_heads = cfg.head_count_kv,
        .num_q_heads = cfg.head_count,
        .dtype = .f16,
        .quant_k = effective_cache_type_k,
        .quant_v = effective_cache_type_v,
        .enable_prefix_cache = false,
        .enable_cpu_offload = enable_cpu_offload_kv,
        .max_seq_len = max_seq_len,
    });
    defer paged_kv.deinit();
    debugz.dbg.printLevel(.info, "[milestone] paged_kv listo (blocks={d} x {d}B)\n", .{ num_blocks, paged_kv.block_alloc.block_bytes });

    var kv_offload = paged_attn.CpuOffloadManager.init(paged_kv.block_alloc, 64, 64);
    defer kv_offload.report();

    // ── 5.2 (lane-b1): draft-model DFlash — construcción tras el pool ────
    // El sidecar se resolvió arriba (dflash_sidecar_path). El draft usa un
    // POOL PROPIO (oráculo llama.cpp: KV del draft separado del target) con
    // la geometría del sidecar; las capas blk.* viven como HybridLayers
    // no-causales (causal=false en DflashDraftModel.init).
    var dflash_model: ?specdrv.dflash_draft.DflashDraftModel = null;
    defer if (dflash_model) |*m| m.deinit();
    var dflash_kv: ?paged_attn.PagedKVCache = null;
    defer if (dflash_kv) |*kv| kv.deinit();
    // Taps del target: entrada PRE-norm de cada capa target_layers, buffer
    // [n_taps_pend, n_extract*n_embd] (layout interleave del oráculo).
    var dflash_taps: []f32 = &[_]f32{};
    defer if (dflash_taps.len > 0) allocator.free(dflash_taps);
    if (dflash_sidecar_path.len > 0) {
        var sidecar = gguf_model.GgufModel.loadSidecarDraft(model, io, allocator, dflash_sidecar_path) catch |e| {
            try stdout.print("[!] dual-load del sidecar dflash falló: {s}\n", .{@errorName(e)});
            return;
        };
        defer sidecar.deinit();
        dflash_kv = try paged_attn.PagedKVCache.init(allocator, .{
            .block_size = block_size,
            .num_blocks = num_blocks,
            .head_dim = sidecar.model.config.head_dim,
            .num_kv_heads = sidecar.model.config.head_count_kv,
            .num_q_heads = sidecar.model.config.head_count,
            .dtype = .f16,
            .enable_prefix_cache = false,
        });
        dflash_model = try specdrv.dflash_draft.DflashDraftModel.init(allocator, &sidecar, &dflash_kv.?, backend, params.spec_p_min);
        dflash_taps = try allocator.alloc(f32, dflash_model.?.target_layers.len * n_embd);
        try stdout.print("[+] 5.2 dflash: draft-model listo (capas={d} bs={d} taps={d} kv_len={d})\n", .{
            dflash_model.?.layers.len, dflash_model.?.block_size, dflash_model.?.target_layers.len, dflash_model.?.kvLen(),
        });
    }

    // ── M3 KVarN cache (lane-b1, opt-in ZIG_AI_KVARN_CACHE=1) ──────────
    // El cache vive por capa y se aloca lazy (sólo capas attn tocadas).
    // Slice 2 (P4): cubin kvarn_kernels se carga si build_options.kvarn_cubin
    // está disponible (bbuild con GPU); el cache se hidrata vía attachCubin
    // y los hooks prefill+decode llaman appendTokens con los device ptrs
    // que AttentionLayer.kvDevicePtrs() expone.
    // Slice 3 (P4): A5-adaptativo (Dev-A 1e5b1ca) — smem_optin se llena
    // con el max shared memory per block real de la GPU, vía cuDeviceInfo.
    // Mapeo compute capability → smem per block (estática, conservadora):
    //   7.0/7.5 (Volta/Turing) 64KB
    //   8.0 (A100) 163KB
    //   8.6/8.9 (RTX/Ada) 99KB
    //   9.0 (Hopper) 228KB
    //   default 48KB (conservador pre-Pascal).
    const kvarn_cache_env = std.c.getenv("ZIG_AI_KVARN_CACHE");
    const kvarn_cache_on = kvarn_cache_env != null and kvarn_cache_env.?[0] == '1';
    var kvarn_cache_opt: ?kvarn_gpu_cache.KvarnGpuCache = null;
    var kvarn_module_opt: ?cudaz.CUmodule = null;
    // 9.4 (lane-b) D2: D=256 ratificado (store d256 + portable fixes
    // @63a4421); Qwen3.5 attention es D=256 (key_length 256).
    // 9.13 (rlt): D=64 ratificado (store d64 rect + fattn portable D64).
    const kvarn_dim_ok = head_dim == 64 or head_dim == 128 or head_dim == 256;
    if (kvarn_cache_on and kvarn_dim_ok) {
        // smem_optin via probe device — A5 adaptativo
        const smem_optin: ?u32 = blk: {
            const dev = cudaz.cuDeviceGet(0) catch break :blk null;
            const info = cudaz.cuDeviceInfo(allocator, dev) catch break :blk null;
            defer allocator.free(info.name);
            const smem: u32 = switch (info.major) {
                7 => 64 * 1024,
                8 => if (info.minor == 0) 163 * 1024 else 99 * 1024,
                9 => 228 * 1024,
                else => 48 * 1024,
            };
            break :blk smem;
        };
        // Stream = el mismo sharedCudaStream que consume todo el motor.
        const shared_stream = matmul.MatmulEngine.sharedCudaStream() catch null;
        if (shared_stream) |ss| {
            // init(module=null) = cache LATENTE hasta slice 2 (Dev-A
            // 7dc6ab7): attachCubin tras cuModuleLoad(build_options.
            // kvarn_cubin). null es válido — NoCubin solo al usarlo.
            if (kvarn_gpu_cache.KvarnGpuCache.init(allocator, .{
                .num_layers = @intCast(eff_blocks),
                .n_kv_heads = @intCast(cfg.head_count_kv),
                .head_dim = @intCast(head_dim),
                .k_bits = 5,
                .v_bits = 4,
                .max_ctx_tokens = @intCast(max_seq_len),
                .smem_optin = smem_optin,
            }, null, @ptrCast(ss.raw))) |kvgc| {
                kvarn_cache_opt = kvgc;
                debugz.dbg.printLevel(.info, "[kvarn] cache ACTIVA: {d} capas × {d} kv_heads × {d}D, k5v4, ctx≤{d}, smem_optin={?d}B\n", .{ eff_blocks, cfg.head_count_kv, head_dim, max_seq_len, smem_optin });
                // Cubin opcional: si build lo proveyó, lo cargamos y
                // lo unimos al cache. Si no, appendTokens devolverá
                // error.NoCubin y el breadcrumb seguirá ahí (D=128
                // sin GPU ⇒ no hay test E2E posible).
                if (build_options.kvarn_cubin.len > 0) {
                    if (cudaz.cuModuleLoad(build_options.kvarn_cubin)) |m| {
                        kvarn_module_opt = m;
                        kvarn_cache_opt.?.attachCubin(m);
                        debugz.dbg.printLevel(.info, "[kvarn] cubin cargado: {s}\n", .{build_options.kvarn_cubin});
                    } else |e| {
                        debugz.dbg.printLevel(.info, "[kvarn] cubin load FALLÓ ({s}); cache latente sin módulo\n", .{@errorName(e)});
                    }
                } else {
                    debugz.dbg.printLevel(.info, "[kvarn] cubin NO provisto por build (build_options.kvarn_cubin vacío); cache latente sin módulo\n", .{});
                }
            } else |e| {
                debugz.dbg.printLevel(.info, "[kvarn] cache init FALLÓ ({s}); sigo sin KVarN\n", .{@errorName(e)});
            }
        } else {
            debugz.dbg.printLevel(.info, "[kvarn] cache SKIP: sharedCudaStream no disponible\n", .{});
        }
    } else if (kvarn_cache_on and !kvarn_dim_ok) {
        debugz.dbg.printLevel(.info, "[kvarn] cache SKIP: head_dim={d} (soportado 64/128/256; 512 pendiente store D-slice)\n", .{head_dim});
    }
    defer if (kvarn_cache_opt) |*c| c.deinit();
    defer if (kvarn_module_opt) |m| cudaz.cuModuleUnload(m);

    // Scheduler for request admission + sequence lifecycle management
    var scheduler = paged_attn.Scheduler.init(allocator, paged_kv.config, &paged_kv);
    defer scheduler.deinit();

    // Pool GPU único compartido entre todas las capas de atención: el pool se
    // indexa por phys_id global del BlockAllocator compartido, así que una sola
    // instancia (num_blocks * block_bytes) sirve a todas las capas. (Antes cada
    // capa alocaba su propio pool -> num_attn_layers * num_blocks * block_bytes,
    // OOM en modelos con context_length grande.)
    // Sólo usar el motor GPU de PagedAttention cuando el backend matmul lo pide
    // (cublas). Con --backend cpu se fuerza paged_gpu=null para que la ruta
    // legacy de-deshacer cuantizado en host se ejercite.
    // El motor GPU (decode + prefill) usa online-softmax con reescalado correcto
    // del acumulador al cambiar el máximo corriente, por lo que produce la misma
    // salida que la referencia CPU.
    const gpu_attention_enabled = true;
    // GPU KV soportado para fp16 y q8_0 (pipeline completo). El resto cae a
    // fp16 con mensaje vía gpuKvPipelineReady.
    const gpu_kv_supported_k = gpuKvPipelineReady(effective_cache_type_k);
    const gpu_kv_supported_v = gpuKvPipelineReady(effective_cache_type_v);
    const use_gpu_kv = (backend == .cublas) and gpu_kv_supported_k and gpu_kv_supported_v and gpu_attention_enabled;
    var shared_paged_gpu: ?paged_attn.PagedAttentionGpu = if (use_gpu_kv)
        paged_attn.PagedAttentionGpu.init(
            allocator,
            .{
                .block_size = block_size,
                .num_blocks = 0,
                .head_dim = head_dim,
                .num_kv_heads = cfg.head_count_kv,
                .num_q_heads = cfg.head_count,
                .dtype = .f16,
                .quant_k = effective_cache_type_k,
                .quant_v = effective_cache_type_v,
            },
            @ptrCast((try matmul.MatmulEngine.sharedCudaStream()).raw),
        ) catch null
    else
        null;
    defer if (shared_paged_gpu) |*g| g.deinit();
    const shared_gpu_ptr: ?*paged_attn.PagedAttentionGpu = if (shared_paged_gpu) |*g| g else null;

    var layers = try allocator.alloc(hybrid_layer.HybridLayer, eff_blocks);
    var layer_block_tables = try allocator.alloc(?*paged_attn.BlockTable, eff_blocks);
    defer allocator.free(layer_block_tables);
    @memset(layer_block_tables, null);
    for (0..eff_blocks) |i| {
        if (cfg.isFullAttentionLayer(i)) {
            const bt = try allocator.create(paged_attn.BlockTable);
            bt.* = paged_attn.BlockTable.init(allocator, block_size);
            layer_block_tables[i] = bt;
        }
        layers[i] = try hybrid_layer.HybridLayer.init(
            allocator,
            i,
            hybrid_layer.HybridLayerParams.fromModelConfig(cfg, max_seq_len),
            cfg.isFullAttentionLayer(i),
            backend,
            &paged_kv,
            if (layer_block_tables[i]) |bt| bt else null,
            shared_gpu_ptr,
        );
        debugz.dbg.printLevel(.info, "[milestone] capa {d}/{d} inicializada\n", .{ i + 1, eff_blocks });
    }

    // P0-RPERF: bounded SWA per decoder layer
    if (params.swa > 0) {
        for (layers) |*layer| {
            if (layer.params.swa_window) |w| {
                if (w > params.swa) {
                    layer.params.swa_window = params.swa;
                    debugz.dbg.printLevel(.detail, "[swa] L{d} clamp {d}→{d}\n", .{ layer.layer_idx, w, params.swa });
                }
            }
        }
    }
    defer allocator.free(layers);
    defer for (layers) |*l| l.deinit();

    // 9.4 (lane-b) FA-native KVarN: opt-in ZIG_AI_KVARN_FA=1 (+cache
    // ZIG_AI_KVARN_CACHE=1). Carga el portable cubin e inyecta en cada capa
    // de atención: descs por capa (via kDescs(layer)), módulo fattn y el
    // trampoline de appendTokens. El camino decae solo a paged en cualquier
    // fallo (flip seguro por paso).
    var kvarn_fa_module: ?cudaz.CUmodule = null;
    const kvarn_fa_env = std.c.getenv("ZIG_AI_KVARN_FA");
    const kvarn_fa_on = kvarn_fa_env != null and kvarn_fa_env.?[0] == '1';
    // Debug: ZIG_AI_KVARN_FA_L=<n> activa SOLO la capa n (aislar por capa).
    const kvarn_fa_only: ?usize = blk: {
        const raw = std.c.getenv("ZIG_AI_KVARN_FA_L") orelse break :blk null;
        var v: usize = 0;
        var i: usize = 0;
        while (raw[i] != 0) : (i += 1) {
            const digit = raw[i] - '0';
            if (digit > 9) break :blk null;
            v = v * 10 + digit;
        }
        break :blk v;
    };
    if (kvarn_fa_on and kvarn_cache_opt != null and build_options.fattn_cubin.len > 0) {
        if (cudaz.cuModuleLoad(build_options.fattn_cubin)) |fm| {
            kvarn_fa_module = fm;
            var n_attn_fa: usize = 0;
            for (layers, 0..) |*layer, li| {
                if (layer.attn_layer) |*attn| {
                    if (kvarn_fa_only) |only| {
                        if (li != only) continue;
                    }
                    attn.kvarn_native = true;
                    attn.kvarn_fattn_module = fm;
                    attn.kvarn_append_fn = kvarnFaAppend;
                    attn.kvarn_append_ctx = @ptrCast(&kvarn_cache_opt.?);
                    n_attn_fa += 1;
                }
            }
            debugz.dbg.printLevel(.info, "[kvarn-fa] FA-native ACTIVA en {d} capas attn (D={d}){?s}\n", .{ n_attn_fa, head_dim, if (kvarn_fa_only) |_| " [solo-L]" else null });
        } else |e| {
            debugz.dbg.printLevel(.info, "[kvarn-fa] fattn cubin load FALLÓ ({s}); atención paged clásica\n", .{@errorName(e)});
        }
    }
    defer if (kvarn_fa_module) |m| cudaz.cuModuleUnload(m);

    // 4.3' copy-once: el bank pinned debe vivir MÁS que las capas — sus VAs
    // son las fuentes del gather de cada MoeLayer (src_gate/up/down). Se
    // declara a ESTE scope para que el defer (LIFO) corra tras el deinit
    // de capas: bank liberado antes ⇒ VAs colgantes ⇒ primer kernel lee el
    // pinned freed → LAUNCH_FAILED(700) sticky (fc4). unreg() con
    // pinned_via_hostalloc=true es no-op seguro.
    var co_bank: ?host_bank.HostBank = null;
    var co_buf: ?[]const u8 = null;
    defer if (co_bank) |*b| {
        b.unreg();
        b.deinit();
    };
    // Re-parse de los metadatos DESDE el buffer pinned: los specs MoE deben
    // apuntar a las VAs del co_buf (pinned, UVA) — si se computan desde el
    // mmap original (model.file), las fuentes del gather son VAs file-backed
    // NO registradas y el kernel faulea → 700 sticky / PAD-only (fc-lb2).
    // deinit (solo metadatos: data es borrowed) antes del bank (LIFO).
    var co_file: ?gguf.GgufFile = null;
    defer if (co_file) |*g| g.deinit();

    // ── 11.2 p2: bundle contiguo como fuente de expertos (lane-e) ──
    // ZIG_AI_BUNDLE=<path>: abre el bundle ZBND (moe-make-bundle) y mapea
    // sus slots contiguos como fuente del gather — 1 lectura secuencial
    // por fetch (PowerInfer §3.1) vs page-faults dispersos del GGUF.
    // El mmap del bundle se REGISTRA completo (pin-after-fill, Contrato 5):
    // expertSlice() del bundle = VAs UVA device-accesibles, mismo contrato
    // que el co_buf del copy-once. Vive hasta tras el deinit de las capas.
    var bundle_src: ?*expert_bundle.BundleSource = null;
    var bundle_bank: ?host_bank.HostBank = null;
    defer {
        if (bundle_src) |bs| {
            bs.deinit();
            allocator.destroy(bs);
        }
        if (bundle_bank) |*b| {
            b.unreg();
            b.deinit();
        }
    }

    // ── Wiring MoE (tickets F/C): attach de MoeLayer por capa MoE del GGUF ──
    // Path híbrido + modelo MoE denso (qwen3moe/qwen2moe/gemma-moe): el FFN
    // de cada capa MoE lo enruta moe_layer (lane-e E5) sobre slot cache GPU
    // compartido. Decode bs=1 nativo; prefill por FFN denso (experto 0,
    // aproximación documentada en hybrid_layer). Con ZIG_AI_MOE_COPYONCE=1
    // el GgufFile se parsea DESDE el buffer pinned (4.3' — RAM 1× modelo).
    var moe_cache_gpu: ?*moe_cuda.MoeCacheGpu = null;
    var moe_gatherer: ?*moe_cuda.ExpertGatherer = null;
    var moe_lk: ?*layer_kernels.LayerKernels = null;
    var moe_layers_attached: usize = 0;
    // 4.5-a (Contrato 8 wiring): executors CPU del híbrido — viven hasta el
    // deinit de las capas (los submits en vuelo completan en el sync del
    // último forward; el watchdog hace join en deinit).
    var moe_exec_gu: ?*cpu_executor.Executor = null;
    var moe_exec_dn: ?*cpu_executor.Executor = null;
    defer {
        // Executors ANTES de las capas (las capas los referencian).
        if (moe_exec_gu) |e| e.deinit();
        if (moe_exec_dn) |e| e.deinit();
        // MoeLayer antes que lk/cache/gatherer (orden inverso del attach):
        // ml.deinit llama kernels/axpy de mcg/gatherer aún vivos.
        for (layers) |*l| {
            if (l.moe) |ml| {
                ml.deinit(allocator);
                allocator.destroy(ml);
                l.moe = null;
            }
        }
        if (moe_gatherer) |g| {
            g.deinit();
            allocator.destroy(g);
        }
        if (moe_cache_gpu) |c| {
            c.deinit();
            allocator.destroy(c);
        }
        if (moe_lk) |k| {
            k.deinit();
            allocator.destroy(k);
        }
        moe_cuda.fetchStreamDeinit();
    }
    if (gguf_moe.isMoeModel(&model.file)) {
        if (gguf_moe.moeInfo(&model.file)) |minfo| {
            const moe_stream: cudaz.CUstream = @ptrCast((try matmul.MatmulEngine.sharedCudaStream()).raw);
            const n_moe_layers = blk: {
                var nm: usize = 0;
                for (0..eff_blocks) |il| {
                    if (gguf_moe.isMoeLayer(&model.file, il)) nm += 1;
                }
                break :blk nm;
            };
            if (n_moe_layers > 0) {
                // Pool que cubre TODOS los expertos de UNA capa (offload puro
                // en steady state; MOE_CACHE=N constreñido para churn medible).
                const mcache_cfg = offload_cache.Config{
                    .num_layers = @intCast(n_moe_layers),
                    .num_experts = minfo.n_expert,
                    .cache_size = blk2: {
                        if (std.c.getenv("MOE_CACHE")) |a| {
                            if (std.fmt.parseInt(u32, std.mem.span(a), 10)) |n| break :blk2 @max(1, n) else |_| {}
                        }
                        break :blk2 minfo.n_expert;
                    },
                    .max_fetch = minfo.top_k * 2,
                };
                const mcg = try allocator.create(moe_cuda.MoeCacheGpu);
                mcg.* = try moe_cuda.MoeCacheGpu.init(mcache_cfg);
                moe_cache_gpu = mcg;
                const mg = try allocator.create(moe_cuda.ExpertGatherer);
                mg.* = try moe_cuda.ExpertGatherer.init(3);
                moe_gatherer = mg;
                const mlk = try allocator.create(layer_kernels.LayerKernels);
                mlk.* = try layer_kernels.LayerKernels.init(moe_stream);
                moe_lk = mlk;

                // Copy-once (4.3'): con ZIG_AI_MOE_COPYONCE=1 el modelo se
                // recarga desde buffer pinned — REGISTRO DEL BANK con el
                // g_pinned_preregistered del moe_bench (mismo contrato 5).
                // co_bank/co_buf son del scope de runHybridInference: viven
                // hasta tras el deinit de las capas (ver declaración arriba).
                if (std.c.getenv("ZIG_AI_MOE_COPYONCE") != null) {
                    const co = try host_bank.HostBank.fromFileCopyOnce(allocator, io, model_path);
                    co_bank = co.bank;
                    co_buf = co.buf;
                    // Re-parse borrowed: las capas MoE leen del buffer pinned.
                    co_file = try gguf.GgufFile.fromBytesBorrowed(allocator, co_buf.?);
                    const n_composed = gguf_moe.composeExternalScalesInPlace(&co_file.?, @constCast(co_buf.?));
                    moe_layer.g_pinned_preregistered = co.buf;
                    debugz.dbg.printLevel(.info, "[milestone] copy-once: {d} MB pinned + {d} escalas compuestas\n", .{ co.buf.len / 1_000_000, n_composed });
                }
                // La región de fuentes del gather: copy-once si activo, si no
                // el mmap original (g.data del fromFileMmap de GgufModel).
                const moe_region: []const u8 = co_buf orelse model.file.data;
                // Specs desde el re-parse pinned (copy-once) o el mmap (default):
                // los .bytes del spec DEBEN ser VAs de moe_region.
                const spec_file: *gguf.GgufFile = if (co_file) |*g| g else &model.file;

                // ── 11.2 p2: bundle contiguo como fuente (lane-e) ──────────
                // ZIG_AI_BUNDLE=<path>: mmap del bundle registrado completo
                // (pin-after-fill) ⇒ expertSlice del bundle = VAs UVA. Los
                // src_* de cada MoeLayer se REPROGRAMAN a la ventana del
                // bundle (stride slot exacto v3 = feat_bytes homogéneo). El
                // GGUF sigue siendo la fuente de METADATOS (specs, router);
                // el bundle solo transporta los BANCOS de expertos.
                var bundle_homog: bool = true;
                if (std.c.getenv("ZIG_AI_BUNDLE")) |bpath| blk: {
                    if (co_buf != null) {
                        debugz.dbg.printLevel(.info, "[cli] ZIG_AI_BUNDLE ignorado: ZIG_AI_MOE_COPYONCE activo (regiones excluyentes)\n", .{});
                        break :blk;
                    }
                    const bs = try allocator.create(expert_bundle.BundleSource);
                    errdefer allocator.destroy(bs);
                    bs.* = expert_bundle.BundleSource.open(allocator, std.mem.span(bpath)) catch |e| {
                        allocator.destroy(bs);
                        debugz.dbg.printLevel(.info, "[cli] bundle open falló ({s}) — fuentes: mmap GGUF\n", .{@errorName(e)});
                        break :blk;
                    };
                    // Validación de layout contra el GGUF (L/E/slot_bytes).
                    expert_bundle.validateAgainstGguf(bs, &model.file, allocator) catch |e| {
                        bs.deinit();
                        allocator.destroy(bs);
                        debugz.dbg.printLevel(.info, "[cli] bundle layout ≠ GGUF ({s}) — fuentes: mmap GGUF\n", .{@errorName(e)});
                        break :blk;
                    };
                    // El gather itera src + id·feat_bytes: el bundle exige
                    // stride del banco == feat_bytes en TODAS las capas
                    // (homogéneo). Heterogéneo (4B SmallThinker) queda
                    // documentado fuera del v3 path (fallback mmap).
                    var first: ?gguf_moe.MoeLayerSpec = null;
                    for (0..eff_blocks) |ili| {
                        if (!gguf_moe.isMoeLayer(&model.file, ili)) continue;
                        const s_chk = gguf_moe.layerSpec(&model.file, ili) catch continue;
                        if (first == null) {
                            first = s_chk;
                        } else {
                            const f = first.?;
                            if (s_chk.gate.expertBytes() != f.gate.expertBytes() or
                                s_chk.up.expertBytes() != f.up.expertBytes() or
                                s_chk.down.expertBytes() != f.down.expertBytes())
                            {
                                bundle_homog = false;
                            }
                        }
                    }
                    if (!bundle_homog) {
                        bs.deinit();
                        allocator.destroy(bs);
                        debugz.dbg.printLevel(.info, "[cli] bundle: modelo heterogéneo (FFN por tramo) — stride del gather ≠ slot; fallback mmap GGUF\n", .{});
                        break :blk;
                    }
                    // p2-v2: SIN registro — el mmap file-backed del bundle NO es
                    // registrable (driver 580: refutado 4.3 + A/B 21B
                    // InvalidValue). Ventanas pageable → staging CLÁSICO
                    // via MoeLayer.init con g_bundle_source (CERO pinned).
                    try bs.ensureMmap(io);
                    bundle_src = bs;
                    moe_layer.g_bundle_source = .{
                        .base = bs.mmap.?.memory,
                        .slot_bytes = .{ bs.hdr.slot_bytes[0], bs.hdr.slot_bytes[1], bs.hdr.slot_bytes[2] },
                        .n_experts = bs.hdr.n_experts,
                        .n_layers = bs.hdr.n_layers,
                        .payload_offset = bs.hdr.payload_offset,
                    };
                    moe_layer.g_bundle_layer_seq = 0;
                    debugz.dbg.printLevel(.info, "[milestone] bundle 11.2 p2-v2: {d} MB mapeados (pageable, staging clásico, CERO pinned), L={d} E={d} slot={d}B\n", .{ (@as(usize, bs.hdr.slot_bytes[0]) + bs.hdr.slot_bytes[1] + bs.hdr.slot_bytes[2]) * bs.hdr.n_experts * bs.hdr.n_layers / 1_000_000, bs.hdr.n_layers, bs.hdr.n_experts, bs.hdr.slot_bytes[0] });
                }

                // ATTACH por capa MoE: el experto-0 sintetizado del FFN denso
                // queda como fallback del prefill; el decode usa MoeLayer.
                var moe_layer_idx: usize = 0;
                for (0..eff_blocks) |il| {
                    if (!gguf_moe.isMoeLayer(&model.file, il)) continue;
                    const spec_i = gguf_moe.layerSpec(spec_file, il) catch |e| {
                        debugz.dbg.printLevel(.info, "[cli] capa {d}: spec MoE inválida ({s}) — FFN denso\n", .{ il, @errorName(e) });
                        continue;
                    };
                    const ml = try allocator.create(moe_layer.MoeLayer);
                    ml.* = moe_layer.MoeLayer.init(allocator, spec_i, mcache_cfg, moe_stream, mlk, mcg, mg, moe_region) catch |e| {
                        allocator.destroy(ml);
                        debugz.dbg.printLevel(.info, "[cli] capa {d}: MoeLayer.init falló ({s}) — FFN denso\n", .{ il, @errorName(e) });
                        continue;
                    };
                    layers[il].moe = ml;
                    moe_layers_attached += 1;
                    moe_layer_idx += 1;
                    // 11.2 p2: reprogramar las fuentes del gather a las
                    // ventanas del bundle (VAs UVA del mmap registrado).
                    // Ventana = slotFileOffset(L,k,0) .. +E·slot_bytes[k]:
                    // stride exacto v3 ⇒ el pseudo-banco es drop-in del
                    // gather (src + id·feat_bytes == slot del experto id).
                    // (p2-v2: fuentes resueltas DENTRO de init via g_bundle_source)
                }
                // El global bundle solo aplica al attach de ARRIBA: reset
                // (MoeLayer.init consumió las ventanas; nada posterior hereda).
                moe_layer.g_bundle_source = null;
                // Executors CPU (Contrato 8, 4.5-a wiring lane-e): con
                // ZIG_AI_HYBRID=1 se crean y attachan los Executors por capa
                // MoE (patrón moe_bench:270-290). sin el env (o dtype de banco
                // sin GEMV CPU): offload puro — moe_layer.hybridEnabled()
                // queda salvaguardado por executorAttached() y el frac cae a
                // 0. LA GEOMETRÍA: los Executors se dimensionan sobre las
                // ventanas del spec (gate/up ya son la MITAD del bloque
                // fusionado cuando el banco es `ffn_gate_up_exps` — lo
                // resuelve resolveGateUp; aquí NO se divide por 2 otra vez).
                if (std.c.getenv("ZIG_AI_HYBRID") != null) blk: {
                    if (moe_layers_attached == 0) break :blk;
                    var gu_any: ?moe_cpu_gemv.Format = null;
                    var dn_any: ?moe_cpu_gemv.Format = null;
                    var gu_in: usize = 0;
                    var gu_out: usize = 0;
                    var dn_in: usize = 0;
                    var dn_out: usize = 0;
                    var e_geom: usize = 0;
                    // La geometría es homogénea entre capas del mismo modelo:
                    // se toma de la PRIMERA capa attachada (fallback: specs
                    // distintas ⇒ geometría de la última vista gana).
                    for (0..eff_blocks) |il| {
                        const ml = layers[il].moe orelse continue;
                        const s = &ml.spec;
                        gu_any = ggmlToCpuFormatCli(s.gate.dtype);
                        dn_any = ggmlToCpuFormatCli(s.down.dtype);
                        gu_in = s.gate.in_dim;
                        gu_out = s.gate.out_dim;
                        dn_in = s.down.in_dim;
                        dn_out = s.down.out_dim;
                        e_geom = s.gate.n_expert;
                        break;
                    }
                    if (gu_any == null or dn_any == null) {
                        debugz.dbg.printLevel(.info, "[cli] ZIG_AI_HYBRID=1 ignorado: dtype de bancos sin GEMV CPU → offload puro\n", .{});
                        break :blk;
                    }
                    const eg = cpu_executor.Executor.initFull(allocator, .{
                        .fmt = gu_any.?,
                        .k_dim = gu_in,
                        .out_dim = gu_out,
                        .n_experts = @intCast(e_geom),
                    }) catch null;
                    const ed = cpu_executor.Executor.initFull(allocator, .{
                        .fmt = dn_any.?,
                        .k_dim = dn_in,
                        .out_dim = dn_out,
                        .n_experts = @intCast(e_geom),
                    }) catch null;
                    if (eg == null or ed == null) {
                        if (eg) |e| e.deinit();
                        if (ed) |e| e.deinit();
                        debugz.dbg.printLevel(.info, "[cli] ZIG_AI_HYBRID=1: fallo init executor → offload puro\n", .{});
                        break :blk;
                    }
                    moe_exec_gu = eg;
                    moe_exec_dn = ed;
                    _ = eg.?.attachStream(@intFromPtr(moe_stream));
                    _ = ed.?.attachStream(@intFromPtr(moe_stream));
                    var attached: usize = 0;
                    for (0..eff_blocks) |il| {
                        const ml = layers[il].moe orelse continue;
                        ml.attachExecutors(eg.?, ed.?);
                        attached += 1;
                    }
                    debugz.dbg.printLevel(.info, "[milestone] ZIG_AI_HYBRID=1: executors CPU attachados ({d} capas) — split q* activo en decode\n", .{attached});
                }
                debugz.dbg.printLevel(.info, "[milestone] MoE attach: {d}/{d} capas (E={d} top_k={d} cache={d})\n", .{ moe_layers_attached, n_moe_layers, minfo.n_expert, minfo.top_k, mcache_cfg.cache_size });
            }
        } else |_| {}
    }
    defer {
        for (layer_block_tables) |bt_opt| {
            if (bt_opt) |bt| {
                bt.deinit(paged_kv.block_alloc);
                allocator.destroy(bt);
            }
        }
    }

    // LayerStreamer: async prefetch + LRU eviction (AirLLM-style)
    var streamer: ?layer_streamer.LayerStreamer = null;
    var vram: ?vram_budget.VramBudget = null;
    // Use auto-detected values if layer streaming was auto-enabled
    const use_layer_stream = params.layer_stream or auto_layer_stream;
    const effective_max_resident = if (auto_layer_stream) auto_max_resident else params.layer_stream_max;
    if (use_layer_stream) {
        const total_vram = if (backend == .cublas) cudaz.getDeviceTotalMem(cudaz.cuDeviceGet(0) catch return) catch 0 else 0;

        const vb_config = vram_budget.VramBudgetConfig{
            .total_vram = total_vram,
            .num_layers = eff_blocks,
            .num_attn_layers = num_attn_layers,
            .hidden_dim = cfg.embedding_length,
            .head_dim = head_dim,
            .num_kv_heads = cfg.head_count_kv,
            .max_seq_len = max_seq_len,
            .kv_quant = switch (effective_cache_type_k) {
                .fp16 => vram_budget.KVQuantFormat.fp16,
                .q8_0 => vram_budget.KVQuantFormat.q8_0,
                .q4_0 => vram_budget.KVQuantFormat.q4_0,
                .q4_1 => vram_budget.KVQuantFormat.q4_1,
                else => vram_budget.KVQuantFormat.fp16,
            },
            .feed_forward_dim = cfg.feed_forward_length,
        };
        var vb = vram_budget.VramBudget.init(vb_config);
        if (debugz.dbg.at(.info)) vb.reportMetrics();
        vram = vb;

        var estimated_weight_per_layer = estimateCompressedWeightPerLayer(allocator, &model.file, cfg) catch 50 * 1024 * 1024;
        // Ensure minimum weight estimate to avoid division by zero
        if (estimated_weight_per_layer == 0) estimated_weight_per_layer = 50 * 1024 * 1024;
        const estimated_f32_per_layer = estimated_weight_per_layer * 8;
        const calc_auto_max_resident = @max(2, vb.weights_budget / (estimated_weight_per_layer + estimated_f32_per_layer));
        var max_resident = @min(effective_max_resident, calc_auto_max_resident);
        // 8.3 FIX (replay+streaming): el decode graph captura PUNTEROS device de
        // TODAS las capas (warmup las carga una a una). Si el LRU expulsa capas
        // tras la captura (max_resident < eff_blocks), el replay opera sobre
        // scratch liberado / q4_cache con host_ptrs reciclados ⇒ logits basura
        // (LFM2.5 generaba blanks: sólo funcionaba con --layer-stream-max 30 o
        // NOGRAPH=1). El replay exige punteros estables: sin expulsión.
        // spec_active (mtp_head+temp<=0) desactiva el grafo ⇒ streaming libre.
        // BUG 7.1b (repro: Ornith 9B, -cl 8, text-only → basura): si el host
        // OOM forzó streaming (la RAM no cabe el eager f32), NO podemos
        // "rescatar" el grafo subiendo max_resident a eff_blocks — eso
        // desactiva el streaming de facto (carga eager 38GB → swap → basura).
        // En ese caso el grafo NO se captura (decode_g=null → path no-graph)
        // y el streaming real (max_resident=2) es el que corre.
        const graph_may_capture = !debugz.dbg.no_graph and !(mtp_head != null and params.sampler.temperature <= 0.0) and !force_stream_host;
        if (graph_may_capture) {
            max_resident = @max(max_resident, eff_blocks);
        } else if (force_stream_host) {
            debugz.dbg.printLevel(.info, "[layer_streamer] OOM host: graph captura desactivada, streaming real (max_resident={d})\n", .{max_resident});
        }
        if (debugz.dbg.at(.info)) {
            try stdout.print("[+] Auto max_resident: {d}, user limit: {d}, using: {d} (weights budget: {} MB, weight/layer: {} MB compressed, {} MB f32)\n", .{
                calc_auto_max_resident,                     params.layer_stream_max,                 max_resident, vb.weights_budget / (1024 * 1024),
                estimated_weight_per_layer / (1024 * 1024), estimated_f32_per_layer / (1024 * 1024),
            });
        }

        var s = try layer_streamer.LayerStreamer.init(
            allocator,
            layers,
            &model.file,
            cfg,
            max_resident,
            2,
            &vb, // Pass VramBudget for VRAM-aware eviction
            null, // gpu_weight_pool (not yet initialized)
            null, // stream (not yet initialized)
        );
        if (debugz.dbg.at(.info)) s.enableDebug();
        streamer = s;
        const total_vram_print = if (backend == .cublas) cudaz.getDeviceTotalMem(cudaz.cuDeviceGet(0) catch return) catch 0 else 0;
        try stdout.print("[+] LayerStreamer activado: max_resident={d} vram={d}MB\n", .{ max_resident, total_vram_print / (1024 * 1024) });
        try stdout.flush();
    } else {
        // Eager load: load all weights up front (original behavior)
        for (0..eff_blocks) |i| {
            try layers[i].loadWeightsFromGguf(&model.file, null);
        }
    }
    if (streamer) |*s| {
        try s.prefetchLayer(0);
    }
    defer if (streamer) |*s| s.deinit();

    // Tokenizer
    var gt = try gguf_tokenizer.GgufTokenizer.fromGguf(allocator, &model.file);
    defer gt.deinit();
    var tok = try bpe.BPETokenizer.fromTokenizer(allocator, &gt);
    defer tok.deinit();

    const raw_prompt = params.prompt orelse "Hola";
    // 10.7 UX vision (lane-mmproj): sin marcadores en el prompt pero con
    // --image/--video, anteponer un bloque por input (patrón mtmd-cli.cpp:441-447)
    const vision_prompt = blk_v: {
        const vp_inputs = if (vision) |vi| try buildExpandInputs(allocator, vi) else null;
        defer if (vp_inputs) |l| allocator.free(l);
        break :blk_v try vision_inject.buildVisionPrompt(allocator, raw_prompt, vp_inputs);
    };
    defer allocator.free(vision_prompt);
    const prompt = try formatPrompt(allocator, vision_prompt, params.use_jinja, &gt);
    defer allocator.free(prompt);
    // A.2: marcadores vision (<|image_pad|> etc.) como tokens únicos — el
    // BPE byte-level los rompería; encodePromptWithVision pre-splitea.
    const enc_result = try encodePromptWithVision(allocator, prompt, &tok, &gt, stdout);
    var prompt_ids = enc_result.ids;
    defer allocator.free(prompt_ids);
    const image_pad_idx_hints = enc_result.image_pad_idxs;
    defer if (image_pad_idx_hints.len > 0) allocator.free(image_pad_idx_hints);
    const video_pad_idx_hints = enc_result.video_pad_idxs;
    defer if (video_pad_idx_hints.len > 0) allocator.free(video_pad_idx_hints);

    // ── Vision (PLAN_MMPROJ 3.2): expandir prompt con tokens de imagen ──
    // Los embeddings del ViT sustituyen a los marcadores <|image_pad|>
    // (id 248056 en Qwen3.8; fallback: prepend). pos_ids per-token para el
    // M-RoPE 2D del target (mtmd-helper.cpp:142 set_position_mrope_2d).
    // Multi-imagen: cada <|image_pad|> consume la imagen k-ésima (por orden
    // de aparición). Los spans se computan tras la expansión total.
    var vision_pos_ids: ?[][4]i32 = null;
    defer if (vision_pos_ids) |p| allocator.free(p);
    // spans por imagen: [start, n) en el prompt EXPANDIDO
    var vision_spans: std.ArrayList(struct { start: usize, n: usize, img: usize }) = .empty;
    defer vision_spans.deinit(allocator);
    var vision_pos_consumed: usize = 0; // posiciones LLM que consumen TODAS las imágenes
    if (vision) |imgs| {
        // Validar dims (todas las imágenes deben coincidir con el target)
        var all_ok = true;
        for (imgs) |v| {
            if (v.out_dim != n_embd) all_ok = false;
        }
        if (!all_ok) {
            try stdout.print("[!] mmproj: dim de imagen != n_embd del target {d} (deepstack?); visión desactivada\n", .{n_embd});
            try stdout.flush();
        } else if (prompt_ids.len > 0) {
            // Markers: <|image_pad|> resueltos por encodePromptWithVision (A.2),
            // con fallback a búsqueda por id (ZIG_AI_IMG_MARKER override para
            // modelos text-only de test con marker sintético).
            const img_marker: u32 = blk: {
                const env = std.c.getenv("ZIG_AI_IMG_MARKER");
                if (env) |e| break :blk std.fmt.parseInt(u32, std.mem.span(e), 10) catch 248056;
                break :blk 248056;
            };
            // Índices de TODOS los markers: hints del encode (multi) + scan
            // fallback por id para los no resueltos por vocab.
            var marker_idxs: std.ArrayList(usize) = .empty;
            defer marker_idxs.deinit(allocator);
            for (image_pad_idx_hints) |h| try marker_idxs.append(allocator, h);
            for (video_pad_idx_hints) |h| try marker_idxs.append(allocator, h);
            if (image_pad_idx_hints.len + video_pad_idx_hints.len > 0) {
                // orden de aparición en el prompt
                std.mem.sort(usize, marker_idxs.items, {}, std.sort.asc(usize));
            }
            if (marker_idxs.items.len == 0) {
                var i: usize = 0;
                while (std.mem.indexOfScalarPos(u32, prompt_ids, i, img_marker)) |mi| {
                    try marker_idxs.append(allocator, mi);
                    i = mi + 1;
                }
            }

            // VIDEO 10.7 (lane-mmproj): expansión de markers por tipo +
            // pos-ids M-RoPE (vision_inject.expandVisionTokens — F3
            // extraída del inline para tests de regresión). Un
            // <|video_pad|> expande TODOS los chunks de su vídeo; un
            // <|image_pad|> una imagen 1:1.
            const exp_inputs = try buildExpandInputs(allocator, imgs);
            defer allocator.free(exp_inputs);
            var expanded = try vision_inject.expandVisionTokens(
                allocator,
                prompt_ids,
                exp_inputs,
                image_pad_idx_hints,
                video_pad_idx_hints,
                img_marker,
            );
            defer expanded.deinit(allocator);
            if (expanded.spans.len == 0) {
                try stdout.print("[i] vision: sin marker (id {d}) en el prompt; embeddings no inyectados\n", .{img_marker});
                try stdout.flush();
            } else {
                allocator.free(prompt_ids);
                prompt_ids = expanded.ids;
                // ownership movida a prompt_ids: desvincular de expanded
                expanded.ids = &.{};
                vision_pos_ids = expanded.pos_ids;
                expanded.pos_ids = &.{};
                for (expanded.spans) |sp| {
                    try vision_spans.append(allocator, .{ .start = sp.start, .n = sp.n, .img = sp.img });
                }
                vision_pos_consumed = expanded.pos_consumed;

                for (vision_spans.items) |sp| {
                    const v = imgs[sp.img];
                    try stdout.print("[+] vision img {d}: {d} embeds en pos {d}..{d} (grid {d}x{d})\n", .{
                        sp.img + 1, sp.n, sp.start, sp.start + sp.n - 1, v.grid_x, v.grid_y,
                    });
                }
                try stdout.print("[+] vision: {d} chunks inyectados, consumen {d} posiciones de contexto\n", .{ vision_spans.items.len, vision_pos_consumed });
                try stdout.flush();

                // Decode GPU: habilitar d_rope_pos (posición de contexto separada
                // del slot KV) y fijar el delta por capa de atención. El delta es
                // negativo: los slots de imagen sólo consumen la suma de
                // max(nx,ny) posiciones de contexto.
                const n_img_total = blk: {
                    var t: usize = 0;
                    for (vision_spans.items) |sp| t += sp.n;
                    break :blk t;
                };
                if (shared_gpu_ptr) |gpu| {
                    gpu.enableRopePos() catch |err| {
                        try stdout.print("[!] vision: enableRopePos falló ({s}); decode usará posiciones de slot\n", .{@errorName(err)});
                        try stdout.flush();
                    };
                    // Fase B: d_pos_ids para el mropePosIdsKernel del prefill
                    // GPU (chunk máx = ubatch). El último chunk puede ser
                    // menor — uploadPosIds sube sólo n ids.
                    gpu.enablePosIds(params.ubatch_size) catch |err| {
                        try stdout.print("[!] vision: enablePosIds falló ({s}); prefill GPU sin pos-ids\n", .{@errorName(err)});
                        try stdout.flush();
                    };
                    const delta: i64 = @as(i64, @intCast(vision_pos_consumed)) - @as(i64, @intCast(n_img_total));
                    for (layers) |*layer| {
                        if (layer.attn_layer) |*attn| attn.rope_pos_delta = delta;
                    }
                    debugz.dbg.printLevel(.info, "[mmproj] decode rope_pos_delta={d} (slot → ctx)\n", .{delta});
                }
            }
        }
    }

    var engine = try matmul.MatmulEngine.init(allocator, backend, .f32);
    defer engine.deinit();

    var rng = std.Random.Xoshiro256.init(params.seed);

    // === Submit request to scheduler (admission + block allocation) ===
    const seq_id: u64 = try scheduler.submit(.{
        .req_id = 0,
        .prompt_tokens = prompt_ids,
        .max_new_tokens = params.max_new_tokens,
        .num_samples = params.num_parallel,
        .priority = 0,
    });
    _ = try scheduler.schedule();

    // === Prefill ===
    const seq_len = prompt_ids.len;
    var hidden = try Tensor(f16).alloc(allocator, &.{ 1, seq_len, n_embd });
    defer hidden.deinit();
    Emb.lookup(emb_quant, emb_f16, prompt_ids, 1, seq_len, &hidden);

    // Vision: sobrescribir las filas del marcador con los embeddings del ViT
    // (los tokens 248056 no existen en la tabla del target ⇒ lookup ~0).
    // Multi-imagen: un span por imagen k.
    if (vision_spans.items.len > 0) {
        const imgs = vision.?;
        for (vision_spans.items) |sp| {
            const v = imgs[sp.img];
            for (0..sp.n) |k| {
                const row = hidden.data[(sp.start + k) * n_embd ..][0..n_embd];
                const src = v.embeddings[k * v.out_dim ..][0..v.out_dim];
                for (src, 0..) |s, i| row[i] = @floatCast(s);
            }
            debugz.dbg.printLevel(.info, "[mmproj] img {d}: {d} embeddings inyectados en [{d}..{d})\n", .{ sp.img + 1, sp.n, sp.start, sp.start + sp.n });
        }
    }

    const hidden_2d = try hidden.reshape(&[_]usize{ seq_len, n_embd });
    defer {
        if (hidden_2d.allocator) |a| {
            a.free(hidden_2d.shape);
            a.free(hidden_2d.strides);
        }
    }

    // Ensure all attention layers have blocks for the prefill sequence
    for (layer_block_tables, 0..) |bt_opt, i| {
        if (bt_opt) |bt| {
            if (bt.num_tokens < seq_len) {
                bt.appendTokens(paged_kv.block_alloc, seq_len - bt.num_tokens) catch |e| {
                    if (e == error.OutOfMemory) printKvOomDiag(paged_kv, num_attn_layers, block_size, max_seq_len, seq_len, stdout);
                    return e;
                };
            }
        }
        _ = i;
    }
    // Vision: el prefill GPU aún no soporta pos-ids per-token (el kernel
    // mropeKernel usa start_pos escalar) ⇒ forzar el path CPU, que sí
    // enruta a applyRoPEMultiSectionPosIds. TODO(GPU): kernel pos-ids.
    // Vision (Fase B): prefill GPU con pos-ids per-token vía mropePosIdsKernel.
    // A/B con NOGPU_VISION_PREFILL=1 fuerza el path CPU (paridad/debug).
    const no_gpu_vision_prefill = blk: {
        const env = std.c.getenv("NOGPU_VISION_PREFILL");
        break :blk env != null and env.?[0] == '1';
    };
    const use_gpu_prefill = !debugz.dbg.no_gpu_prefill and backend == .cublas and
        (vision_pos_ids == null or !no_gpu_vision_prefill);
    if (vision_pos_ids != null and backend == .cublas and no_gpu_vision_prefill) {
        try stdout.print("[i] vision: prefill CPU forzado por NOGPU_VISION_PREFILL=1\n", .{});
        try stdout.flush();
    }
    // Hidden pre-norma del último token del prompt (ancla de la 1ª ronda draft).
    var spec_hprev: ?[]f32 = null;
    defer if (spec_hprev) |b| allocator.free(b);
    var logits_f32 = try allocator.alloc(f32, vocab);
    defer allocator.free(logits_f32);

    // [5.2 Fase A GPU device-resident (lane-c) retirada en merge según
    // arbitraje coordinador 01:4x — camino b1 CPU/DflashDraftModel manda.
    // Preservada íntegra en @b7cd73e: DflashEncoder 5.1 + capas sidecar
    // como HybridLayers con seq propia + tap-capture DtoD prefill +
    // KV-inject tras prefill + graph_ok !dflash_active. Reutilizable
    // para el denoise GPU de la Fase B.]
    const t_prefill = @import("time").Timer.start();
    if (use_gpu_prefill) {
        // Prefill 100% GPU en chunks de `ubatch_size` (llama.cpp -ub): embeddings
        // H2D por chunk, capas forwardGPU con n = chunk. El estado recurrente SSM
        // y el KV de atención quedan en el pool GPU (sin siembra host→device).
        const ub = params.ubatch_size;
        var lk = try layer_kernels.LayerKernels.init(@ptrCast((try matmul.MatmulEngine.sharedCudaStream()).raw));
        defer lk.deinit();
        var g_cur = try cublas.GpuTensor(f32).alloc(ub * n_embd);
        defer g_cur.deinit();
        var g_nxt = try cublas.GpuTensor(f32).alloc(ub * n_embd);
        defer g_nxt.deinit();
        var g_normed = try cublas.GpuTensor(f32).alloc(n_embd);
        defer g_normed.deinit();
        var g_logits = try cublas.GpuTensor(f32).alloc(vocab);
        defer g_logits.deinit();
        var g_out_norm = try cublas.GpuBuffer(f32).alloc(n_embd);
        defer g_out_norm.free();
        try g_out_norm.upload(out_norm.data);

        const stage = try allocator.alloc(f32, ub * n_embd);
        defer allocator.free(stage);

        const perf_t = @import("time").Timer.start();
        const perf_prefill = debugz.dbg.perf_stage;
        var p_ev: []cudaz.CUevent = undefined;
        var p_layer_ns: []i128 = undefined;
        var p_t_embed_ns: i128 = 0;
        var p_t_enq_ns: i128 = 0;
        var p_gpu_total_ns: i128 = 0;
        var p_gpu_head_ns: i128 = 0;
        if (perf_prefill) {
            p_ev = try allocator.alloc(cudaz.CUevent, layers.len + 3);
            p_layer_ns = try allocator.alloc(i128, layers.len);
            @memset(p_layer_ns, 0);
            for (p_ev) |*e| e.* = try cudaz.cuEventCreate(0);
        }

        var pos: usize = 0;
        var last_n: usize = 0;
        var cur2gpu = g_cur;
        var nxt2gpu = g_nxt;
        while (pos < seq_len) {
            const n = @min(ub, seq_len - pos);
            const t_emb0 = perf_t.read();
            for (0..n * n_embd) |i| {
                stage[i] = @as(f32, @floatCast(hidden_2d.data[pos * n_embd + i]));
            }
            try cudaz.cuMemcpyHtoD(cur2gpu.ptr(), @intFromPtr(stage.ptr), n * n_embd * @sizeOf(f32));
            p_t_embed_ns += perf_t.read() - t_emb0;
            if (perf_prefill) try cudaz.cuEventRecord(p_ev[0], lk.stream);
            const t_enq0 = perf_t.read();
            for (layers, 0..) |*layer, li| {
                if (streamer) |*s| try s.ensureLayerLoaded(li);
                try hybrid_layer.HybridLayer.forwardGPU(layer, &lk, cur2gpu, &nxt2gpu, pos, n, vision_pos_ids);
                // [tap-capture DtoD por target_layer (Fase A) — retirada, @b7cd73e]
                // M3 slice 2: appendTokens enqueue al mismo lk.stream
                // (stream-order ⇒ consume las K/V projections que forwardGPU
                // dejó en device). Usa el accessor AttentionLayer.kvDevicePtrs
                // vía HybridLayer.kvDevicePtrs. Sólo capas attn.
                if (kvarn_cache_opt != null) {
                    if (layer.kvDevicePtrs()) |kvp| {
                        kvarn_cache_opt.?.appendTokens(@intCast(li), kvp.k, kvp.v, @intCast(n), @intCast(pos), null, lk.stream) catch |e| {
                            debugz.dbg.printLevel(.info, "[kvarn] prefill li={d} pos={d} n={d} appendTokens FALLÓ ({s})\n", .{ li, pos, n, @errorName(e) });
                        };
                    }
                }
                if (streamer) |*s| try s.prefetchNext(li);
                if (perf_prefill) try cudaz.cuEventRecord(p_ev[li + 1], lk.stream);
                const t2 = cur2gpu;
                cur2gpu = nxt2gpu;
                nxt2gpu = t2;
                if (debugz.dbg.dump_prefill_layers) {
                    try cudaz.cuStreamSynchronize(lk.stream);
                    const chk = try allocator.alloc(f32, n * n_embd);
                    defer allocator.free(chk);
                    try cudaz.cuMemcpyDtoH(@intFromPtr(chk.ptr), cur2gpu.ptr(), n * n_embd * @sizeOf(f32));
                    debugz.dbg.print("[pipeline] PREFILL_LAYER li={d} n={d} sum|v|={d:.6} max={d:.6} f0={d:.5} f1={d:.5} f2={d:.5}\n", .{ li, n, debugz.sumAbsF32(chk), debugz.maxAbsF32(chk), chk[0], chk[1], chk[2] });
                }
            }
            p_t_enq_ns += perf_t.read() - t_enq0;
            pos += n;
            last_n = n;
        }

        // LM head sobre el último token del prompt (última fila del buffer final).
        try cudaz.cuStreamSynchronize(lk.stream);
        if (perf_prefill) try cudaz.cuEventRecord(p_ev[layers.len + 2], lk.stream);
        const last_row = cur2gpu.ptr() + (last_n - 1) * n_embd * @sizeOf(f32);
        if (mtp_head != null and spec_hprev == null) {
            const b = try allocator.alloc(f32, n_embd);
            try cudaz.cuMemcpyDtoH(@intFromPtr(b.ptr), last_row, n_embd * @sizeOf(f32));
            spec_hprev = b;
        }
        try lk.rmsNorm(last_row, @intFromPtr(g_out_norm.dev_ptr), g_normed.ptr(), 1, n_embd, rms_eps);
        // [KV-inject tras prefill (Fase A) — retirada, @b7cd73e]
        var prefill_host_logits = false;
        // NOTA C: el prefill-tail NO usa lmq80 — la subida del peso q8_0 va
        // DESPUÉS del prefill: con él residente antes, el bootstrap OOMeó en
        // linearProjectionDevice (el prefill sí completó ⇒ apunta a historia
        // de allocs del sub-allocator, ticket lane-a). Coste CPU-GEMV aquí:
        // 1 proyección por corrida, despreciable.
        if (lm_head_cpu_fb) {
            const nb = try allocator.alloc(f32, n_embd);
            defer allocator.free(nb);
            try cudaz.cuStreamSynchronize(lk.stream);
            try cudaz.cuMemcpyDtoH(@intFromPtr(nb.ptr), g_normed.ptr(), n_embd * @sizeOf(f32));
            cpuLmHeadLogits(nb, lm_head.data, logits_f32);
            prefill_host_logits = true;
        } else if (lm_head_q4 or lmq40 != null) {
            const lmq40_b: []const u8 = if (lmq40) |*q| q.bytes else lm_head_q.bytes;
            try lk.q4gemmLinear(allocator, g_normed.ptr(), lmq40_b, g_logits.ptr(), n_embd, vocab);
        } else if (lm_head_q6k) {
            try lk.qgemmLinear(allocator, g_normed.ptr(), lm_head_q.bytes, g_logits.ptr(), 1, n_embd, vocab, 3);
        } else {
            try engine.linearProjectionDeviceF16(g_normed, lm_head, &g_logits, 1, n_embd, vocab);
        }
        if (perf_prefill) try cudaz.cuEventRecord(p_ev[layers.len + 1], lk.stream);
        try cudaz.cuStreamSynchronize(lk.stream);
        if (!prefill_host_logits) {
            try cudaz.cuMemcpyDtoH(@intFromPtr(logits_f32.ptr), g_logits.ptr(), vocab * @sizeOf(f32));
        }
        if (perf_prefill) {
            var ms: f32 = 0;
            for (layers, 0..) |_, li| {
                try cudaz.cuEventElapsedTime(&ms, p_ev[li], p_ev[li + 1]);
                const ns = @as(i128, @intFromFloat(@as(f64, ms) * std.time.ns_per_ms));
                p_layer_ns[li] += ns;
            }
            try cudaz.cuEventElapsedTime(&ms, p_ev[layers.len + 2], p_ev[layers.len + 1]);
            p_gpu_head_ns += @as(i128, @intFromFloat(@as(f64, ms) * std.time.ns_per_ms));
            try cudaz.cuEventElapsedTime(&ms, p_ev[0], p_ev[layers.len + 1]);
            p_gpu_total_ns += @as(i128, @intFromFloat(@as(f64, ms) * std.time.ns_per_ms));
            const us = std.time.ns_per_us;
            var ssm_ns: i128 = 0;
            var attn_ns: i128 = 0;
            for (layers, 0..) |layer, li| {
                if (layer.is_attention) attn_ns += p_layer_ns[li] else ssm_ns += p_layer_ns[li];
            }
            try stdout.print("[+] PERF prefill (total {d:.1} ms):\n", .{@as(f64, @floatFromInt(p_gpu_total_ns)) / @as(f64, @floatFromInt(std.time.ns_per_ms))});
            try stdout.print("  host  embed {d:.1} us  enqueue {d:.1} us\n", .{
                @as(f64, @floatFromInt(p_t_embed_ns)) / @as(f64, @floatFromInt(us)),
                @as(f64, @floatFromInt(p_t_enq_ns)) / @as(f64, @floatFromInt(us)),
            });
            try stdout.print("  gpu   ssm {d:.1} us  attn {d:.1} us  head {d:.1} us  total {d:.1} us\n", .{
                @as(f64, @floatFromInt(ssm_ns)) / @as(f64, @floatFromInt(us)),
                @as(f64, @floatFromInt(attn_ns)) / @as(f64, @floatFromInt(us)),
                @as(f64, @floatFromInt(p_gpu_head_ns)) / @as(f64, @floatFromInt(us)),
                @as(f64, @floatFromInt(p_gpu_total_ns)) / @as(f64, @floatFromInt(us)),
            });
            var top: [5]usize = undefined;
            for (0..5) |k| top[k] = k;
            for (layers, 0..) |_, li| {
                if (p_layer_ns[li] <= p_layer_ns[top[4]]) {
                    top[4] = li;
                    for (0..3) |k| {
                        if (p_layer_ns[top[k + 1]] > p_layer_ns[top[k]]) {
                            const tmp = top[k];
                            top[k] = top[k + 1];
                            top[k + 1] = tmp;
                        }
                    }
                }
            }
            try stdout.print("  top layers:\n", .{});
            for (top) |li| {
                try stdout.print("    L{d:<2} {s} {d:.1} us\n", .{
                    li,
                    if (layers[li].is_attention) "attn" else "ssm",
                    @as(f64, @floatFromInt(p_layer_ns[li])) / @as(f64, @floatFromInt(us)),
                });
            }
        }
    } else {
        var buf_a = try Tensor(f32).alloc(allocator, &.{ seq_len, n_embd });
        defer buf_a.deinit();
        var buf_b = try Tensor(f32).alloc(allocator, &.{ seq_len, n_embd });
        defer buf_b.deinit();
        // La primera capa recibe el embedding del prompt (f16 → f32)
        for (buf_a.data, hidden_2d.data) |*d, s| d.* = @as(f32, @floatCast(s));

        var cur = &buf_a;
        var nxt = &buf_b;
        for (layers, 0..) |*layer, li| {
            // 5.2 (lane-b1): taps DFlash del ÚLTIMO token del prefill (fila
            // seq_len-1 de la entrada de cada capa target_layers) — el primer
            // draft denoise consume estas taps.
            if (dflash_model != null and dflash_taps.len >= n_embd) {
                const want = blk_tl: {
                    if (dflash_model) |*df| {
                        for (df.target_layers) |tl| {
                            if (tl == @as(i32, @intCast(li))) break :blk_tl true;
                        }
                    }
                    break :blk_tl false;
                };
                if (want) {
                    var tap_slot: usize = 0;
                    if (dflash_model) |*df| {
                        for (df.target_layers, 0..) |tl, ti| {
                            if (tl == @as(i32, @intCast(li))) {
                                tap_slot = ti;
                                break;
                            }
                        }
                    }
                    @memcpy(
                        dflash_taps[tap_slot * n_embd ..][0..n_embd],
                        cur.data[(seq_len - 1) * n_embd ..][0..n_embd],
                    );
                }
            }
            // Con layer-streaming (Ornith 9B: eager 38GB > RAM ⇒ streamer
            // forzado), las capas llegan UNLOADED al prefill CPU — sin
            // ensureLayerLoaded el QuantWeight apunta a préstamo vacío y
            // ensureScratchFilled segfaulta. Simétrico al path GPU (que sí
            // hace ensure+prefetch por capa).
            if (streamer) |*s| try s.ensureLayerLoaded(li);
            try layer.forward(cur.*, nxt, 0, seq_len, vision_pos_ids);
            if (streamer) |*s| try s.prefetchNext(li);
            const t = cur;
            cur = nxt;
            nxt = t;
        }

        // Sembrar el estado recurrente SSM en GPU a partir del prefill CPU.
        for (layers) |*layer| try hybrid_layer.HybridLayer.seedGpuFromHost(layer);

        // Subir el KV de atención del prefill (host pool) al device pool antes
        // del decode GPU-residente: el decode device lee bloques residentes.
        if (shared_gpu_ptr) |gpu| {
            for (layer_block_tables) |bt_opt| {
                if (bt_opt) |bt| try gpu.stageTableAll(paged_kv.block_alloc, bt);
            }
        }

        // LM head sobre el último token del prefill (con output_norm final)
        var last_shape = [_]usize{ 1, n_embd };
        var last_strides = [_]usize{ n_embd, 1 };
        const last_2d = Tensor(f32){
            .data = cur.data[(seq_len - 1) * n_embd ..][0..n_embd],
            .shape = &last_shape,
            .strides = &last_strides,
            .offset = 0,
            .allocator = null,
            .owns_data = false,
        };
        if (mtp_head != null and spec_hprev == null) {
            const b = try allocator.alloc(f32, n_embd);
            @memcpy(b, cur.data[(seq_len - 1) * n_embd ..][0..n_embd]);
            spec_hprev = b;
        }

        var normed = try Tensor(f32).alloc(allocator, &.{ 1, n_embd });
        defer normed.deinit();
        norm.rmsNorm(f32, f32, last_2d, out_norm, rms_eps, &normed);

        var normed16 = try Tensor(f16).alloc(allocator, &.{ 1, n_embd });
        defer normed16.deinit();
        for (normed.data, normed16.data) |s, *d| d.* = @floatCast(s);

        var logits = try Tensor(f16).alloc(allocator, &.{ 1, vocab });
        defer logits.deinit();
        try embedding.lmHeadForward(&engine, normed16, lm_head, &logits);

        for (logits.data, 0..) |v, i| logits_f32[i] = @as(f32, @floatCast(v));
    }
    const prefill_ns = t_prefill.read();

    // 6.4 (coordinador): paridad greedy del prefill — top-5 logits tras
    // output_norm+lm_head, mismo formato que el pipeline clásico (main.zig).
    // Reintegrado post-merge (2add840 lo perdió): golden K2-Horizon llama.cpp
    // en main.zig:1156. Gated DEBUG_LEVEL=2, tag [pipeline].
    if (debugz.dbg.at(.detail)) {
        // Init -inf (NO memset-0: con ceros los puestos 2..5 quedaban
        // colgados del sort con val=0.0 y mentían "top-5 todos cero").
        var top: [5]struct { idx: u32, val: f32 } = undefined;
        for (&top) |*t| t.* = .{ .idx = 0, .val = -std.math.inf(f32) };
        for (logits_f32, 0..) |v, i| {
            const vi: u32 = @intCast(i);
            if (v > top[4].val) {
                top[4] = .{ .idx = vi, .val = v };
                std.mem.sort(@TypeOf(top[0]), &top, {}, struct {
                    fn lt(_: void, a: @TypeOf(top[0]), b: @TypeOf(top[0])) bool {
                        return a.val > b.val;
                    }
                }.lt);
            }
        }
        debugz.dbg.printLevel(.detail, "[pipeline] prefill top-5 (first_token={d})\n", .{top[0].idx});
        for (top) |t| debugz.dbg.printLevel(.detail, "[pipeline]   tok={d} logit={d:.3}\n", .{ t.idx, t.val });
        // Golden K2-Horizon 1B (IFM fork llama-server, prompt "The capital
        // of France is", temp 0, ctx 512, 2026-09-13): top-1 ' a'=265
        // lp -2.755; secuencia ' a common misconception' [265,4613,6466,
        // 723,10026]. Imprimimos el logit CRUDO de cada golden-tok para el
        // diff numérico vs oráculo.
        var max_l: f32 = -std.math.inf(f32);
        for (logits_f32) |v| {
            if (v > max_l) max_l = v;
        }
        var sum_exp: f64 = 0;
        for (logits_f32) |v| {
            sum_exp += @exp(@as(f64, v - max_l));
        }
        const lse = max_l + @as(f32, @floatCast(@log(sum_exp)));
        for ([_]u32{ 265, 222, 294, 848, 667, 4613 }) |g| {
            debugz.dbg.printLevel(.detail, "[pipeline] golden tok={d} logit={d:.3} logprob={d:.3}\n", .{ g, logits_f32[g], logits_f32[g] - lse });
        }
    }

    var gen_tokens: std.ArrayList(u32) = .empty;
    defer gen_tokens.deinit(allocator);
    const first_token = params.sampler.sample(logits_f32, &rng, &[_]u32{});
    try gen_tokens.append(allocator, first_token);
    var current_pos: usize = seq_len;

    // Server F2 T2c: el primer token (salido del prefill) también se emite.
    if (sink) |s| {
        const piece = tok.decode(&[_]u32{first_token}, allocator) catch "";
        defer if (piece.len > 0) allocator.free(piece);
        _ = s.emit(first_token, piece) catch {};
    }
    const t_gen = @import("time").Timer.start();

    try stdout.print("[*] Generando... 0 tokens\n", .{});

    // Activaciones residentes en GPU (Path B): un solo H2D (embedding) y un
    // solo D2H (norma final) por token; todo lo demás queda en device.
    var lk = try layer_kernels.LayerKernels.init(@ptrCast((try matmul.MatmulEngine.sharedCudaStream()).raw));

    defer lk.deinit();

    // ─── G2 (TODO 1.7, lane-a): argmax device + fin de grafo en device ─────
    // Hoy cada token de decode copia TODO el vector de logits a host
    // (vocab·4B ≈ 993 KB ≈ 257 µs) y hace el argmax en CPU (~84 µs). Con el
    // kernel `argmaxF32Kernel` el argmax corre en GPU y el D2H es de 4 bytes.
    //
    // Sólo es válido para GREEDY PURO: el resto de muestreadores (temp>0,
    // top-k/p, repetition penalty) necesita los logits completos en host.
    // Además exige que los logits se produzcan en GPU (no lm_head CPU-fallback).
    // OPT-IN `ZIG_AI_GPUARGMAX=1` (A/B); flip a default tras paridad + bench.
    // Nota: con DUMP_LOGITS el vector completo tiene que llegar a host sí o
    // sí (el breadcrumb vuelca logits_f32) ⇒ se desactiva el atajo.
    var gpu_argmax = debugz.dbg.gpu_argmax and
        !debugz.dbg.dump_logits and
        params.sampler.temperature <= 0 and
        params.sampler.repetition_penalty == 1.0 and
        !lm_head_cpu_fb;
    // 1.15 path-A (lane-a): Gumbel-trick con temp>0 — el sampleo EXACTO de
    // softmax(temp) (mismo camino que el fast-path CPU pipeline.zig:39) va
    // EN DEVICE: argmax(logits/temp + gumbel_i). Cubre temp>0 SIN top_k y
    // SIN top_p (los defaults del CLI) con o sin rep_penalty (scatter del
    // ring en device). Elimina softmax CPU ~1.3ms + D2H 501KB/token.
    // top_k/top_p≠defaults ⇒ CPU (path-B futuro).
    const gpu_gumbel = debugz.dbg.gpu_argmax and
        !debugz.dbg.dump_logits and
        params.sampler.temperature > 0 and
        params.sampler.top_k == 0 and
        params.sampler.top_p >= 1.0 and
        !lm_head_cpu_fb and
        mtp_head == null; // spec/rejection sampling necesita logits host (path-B)
    // Puntero device [1]i32 donde el argmax escribe el token (0 = off). Se
    // PRE-alloca AQUÍ, antes de cualquier captura de grafo: un cuMemAlloc
    // dentro del capture desactiva el grafo en silencio (lección TODO 1.3).
    var gpu_argmax_out: usize = 0;
    if (gpu_argmax) {
        gpu_argmax_out = @as(usize, @intCast(lk.argmaxOut(1) catch 0));
        if (gpu_argmax_out == 0) gpu_argmax = false;
    }
    // 1.15: el Gumbel reutiliza el MISMO buffer de salida (i32 por fila) y
    // su estado philox/ring se pre-aloca aquí igual (capture-safe). El
    // counter se siembra con la seed del sampler — misma seed ⇒ mismo
    // stream de gumbels (reproducible).
    if (gpu_gumbel) {
        gpu_argmax_out = @as(usize, @intCast(lk.argmaxOut(1) catch 0));
        if (gpu_argmax_out == 0) {
            // sin buffer no hay camino device
        } else {
            lk.sampleGumbelInit(params.seed) catch {
                debugz.dbg.printLevel(.info, "[1.15] sampleGumbelInit falló — sampling CPU\n", .{});
            };
        }
    }
    const gpu_sample = gpu_argmax or (gpu_gumbel and gpu_argmax_out != 0);
    if (gpu_gumbel and gpu_argmax_out != 0) {
        debugz.dbg.printLevel(.info, "[1.15-gumbel] activo: temp={d:.2} pen={d:.2} — sampleo softmax(temp) exacto en device (Gumbel-max), D2H 4B\n", .{ params.sampler.temperature, params.sampler.repetition_penalty });
    }
    if (gpu_argmax) {
        debugz.dbg.printLevel(.info, "[gpu-argmax] activo: argmax device + D2H 4B (ahorro {d:.0} KB/token)\n", .{@as(f64, @floatFromInt(vocab * @sizeOf(f32))) / 1024.0});
    }

    var g_cur = try cublas.GpuTensor(f32).alloc(n_embd);
    defer g_cur.deinit();
    var g_nxt = try cublas.GpuTensor(f32).alloc(n_embd);
    defer g_nxt.deinit();
    var g_normed = try cublas.GpuTensor(f32).alloc(n_embd);
    defer g_normed.deinit();
    var g_logits = try cublas.GpuTensor(f32).alloc(vocab);
    defer g_logits.deinit();
    var g_out_norm = try cublas.GpuBuffer(f32).alloc(n_embd);
    defer g_out_norm.free();
    try g_out_norm.upload(out_norm.data);

    // ─── B6 (lane-b) + C: lm_head q8_0 on-load para cabezas grandes ─────────
    // Sustituye al fallback CPU-GEMV en TODOS los sitios de proyección
    // (prefill-tail, draft, verify batched, bootstrap/bonus, decode):
    // proyección device sin subir los GB del peso bf16/f16. Va DESPUÉS del
    // prefill (orden probado E2E; antes OOMeaba el bootstrap — ver nota en
    // el tail del prefill). Instancia LayerKernels propia: la del prefill
    // es local a su bloque y la del decode nace después (el tipo es
    // stateless: módulo/funcs globales). Si no hay VRAM para el peso q8_0
    // se degrada a CPU-GEMV sin morir.
    var lmq80_lk: ?layer_kernels.LayerKernels = null;
    var lmq80_w: ?cublas.GpuBuffer(u8) = null;
    defer if (lmq80_w) |*b| b.free();
    var lmq80_aq: ?cublas.GpuTensor(i8) = null;
    defer if (lmq80_aq) |*b| b.deinit();
    var lmq80_ad: ?cublas.GpuBuffer(f16) = null;
    defer if (lmq80_ad) |*b| b.free();
    var lmq80: ?LmQ80Ctx = null;
    if (lm_head_cpu_fb and !lmq80_force_off) {
        lmq80_init: {
            const r = model.loadLmHeadQ80(allocator) catch break :lmq80_init;
            // Cubre TODAS las salidas anticipadas del bloque (OOM de
            // subida/alloc): sin esto filtraríamos hasta ~1.3GB host.
            defer allocator.free(r.bytes);
            const kb80 = r.hidden / 32;
            const blk_host = allocator.alloc(u8, kb80 * 34) catch break :lmq80_init;
            const x16_host = allocator.alloc(f16, r.hidden) catch break :lmq80_init;
            const aq_host = allocator.alloc(i8, LmQ80Ctx.MAX_M * r.hidden) catch break :lmq80_init;
            const ad_host = allocator.alloc(f16, LmQ80Ctx.MAX_M * kb80) catch break :lmq80_init;
            lmq80_w = cublas.GpuBuffer(u8).alloc(r.bytes.len) catch break :lmq80_init;
            lmq80_w.?.upload(r.bytes) catch break :lmq80_init;
            lmq80_aq = cublas.GpuTensor(i8).alloc(LmQ80Ctx.MAX_M * r.hidden) catch break :lmq80_init;
            lmq80_ad = cublas.GpuBuffer(f16).alloc(LmQ80Ctx.MAX_M * kb80) catch break :lmq80_init;
            lmq80_lk = layer_kernels.LayerKernels.init(@ptrCast((try matmul.MatmulEngine.sharedCudaStream()).raw)) catch break :lmq80_init;
            lmq80 = .{
                .w_dev = @intFromPtr(lmq80_w.?.dev_ptr),
                .aq_dev = lmq80_aq.?.ptr(),
                .ad_dev = @intFromPtr(lmq80_ad.?.dev_ptr),
                .blk_host = blk_host,
                .x16_host = x16_host,
                .aq_host = aq_host,
                .ad_host = ad_host,
                .hidden = r.hidden,
                .vocab = r.vocab,
                .lk = &lmq80_lk.?,
            };
            try stdout.print("[+] lm_head q8_0 on-load activo ({d} MB device, antes {d} MB f16)\n", .{ r.bytes.len / (1024 * 1024), r.vocab * r.hidden * 2 / (1024 * 1024) });
        }
        if (lmq80 == null and debugz.dbg.at(.info)) {
            debugz.dbg.printLevel(.info, "[milestone] lm_head q8_0 no disponible → CPU-GEMV queda como fallback\n", .{});
        }
    }

    // ─── Spec v1 secuencial (C4.2, lane-c) ───────────────────────────────────
    // Cola de drafts verificada por el decode normal comparando argmax.
    // Sin batch aún (C4.3), pero camino completo: aceptación/rechazo,
    // rollback del KV del draft vía truncate y métricas DUMP_SPEC/PERF_SPEC.
    // Requiere temperature<=0 (comparación greedy; con temp>0 se desactiva).
    // 5.5 (lane-f): con temp>0 el spec YA NO se desactiva — entra al modo
    // REJECTION SAMPLING (semántica llama.cpp): el drafter sigue eligiendo
    // greedy, el verify compara distribuciones softmax p/q con
    // sampler.rejectionStep (aceptación min(1,p/q), rechazo = resample del
    // residual max(0,p−q)). Camino greedy (temp<=0) intacto.
    // 5.2 (lane-b1): dflash activa el camino especulativo (rondas batched)
    // sin cabeza MTP — el draft sale del DflashDraftModel del sidecar.
    const spec_active = mtp_head != null or dflash_model != null;
    const spec_rejection = mtp_head != null and params.sampler.temperature > 0.0;
    var spec_round = specdrv.SpecDriver.Round.init();
    defer spec_round.deinit(allocator);
    var draft_layer: ?hybrid_layer.HybridLayer = null;
    defer if (draft_layer) |*dl| dl.deinit();
    var draft_bt: ?*paged_attn.BlockTable = null;
    var h_prev: []f32 = &[_]f32{};
    defer allocator.free(h_prev);
    var eh_w: []f32 = &[_]f32{};
    defer allocator.free(eh_w);
    var head_concat: []f32 = &[_]f32{};
    defer allocator.free(head_concat);
    var head_x0_host: []f32 = &[_]f32{};
    defer allocator.free(head_x0_host);
    var spec_logits_host: []f32 = &[_]f32{};
    // 5.5 (lane-f): logits del drafter por posición (kq×vocab) — sólo en
    // modo rejection (temp>0); greedy no los necesita.
    var draft_logits_rows: []f32 = &[_]f32{};
    defer if (draft_logits_rows.len > 0) allocator.free(draft_logits_rows);
    defer allocator.free(spec_logits_host);
    var emb_row_f32: []f32 = &[_]f32{};
    defer allocator.free(emb_row_f32);
    var g_head_in: ?cublas.GpuTensor(f32) = null;
    defer if (g_head_in) |*b| b.deinit();
    var g_head_out: ?cublas.GpuTensor(f32) = null;
    defer if (g_head_out) |*b| b.deinit();
    var g_spec_shnorm: ?cublas.GpuBuffer(f32) = null;
    defer if (g_spec_shnorm) |*b| b.free();
    var g_head_logits: ?cublas.GpuTensor(f32) = null;
    defer if (g_head_logits) |*b| b.deinit();

    if (spec_active) {
        if (spec_hprev) |hp| {
            h_prev = hp;
            spec_hprev = null; // transferencia de ownership
        } else if (dflash_model == null) {
            try stdout.print("[!] draft-mtp sin hidden del prefill; desactivando especulación\n", .{});
        }
    }
    // 5.2 (lane-b1): con DFlash, draft_bt = BlockTable del draft-model (su
    // propia secuencia en dflash_kv) — el preámbulo del loop de rondas y el
    // verify batched lo usan para alinear posiciones; el KV real del draft
    // lo gestiona el modelo (kvInject/rollbackTo).
    if (dflash_model) |*df| {
        draft_bt = df.draft_bt;
    }
    if (h_prev.len == n_embd and mtp_head != null) {
        const E = n_embd;
        eh_w = try allocator.alloc(f32, E * 2 * E);
        mtp_head.?.eh_proj.dequantToF32Transposed(eh_w);
        head_concat = try allocator.alloc(f32, 2 * E);
        head_x0_host = try allocator.alloc(f32, E);
        spec_logits_host = try allocator.alloc(f32, vocab);
        // 5.5 (lane-f): filas de logits del drafter por posición — el verify
        // con temp>0 (rejection) necesita q(x) en cada paso draft; greedy
        // (temp<=0) no las usa y las deja en &.{} (cero alloc).
        if (spec_rejection) draft_logits_rows = try allocator.alloc(f32, params.spec_draft_n_max * vocab);
        emb_row_f32 = try allocator.alloc(f32, E);
        g_head_in = try cublas.GpuTensor(f32).alloc(E);
        g_head_out = try cublas.GpuTensor(f32).alloc(E);
        g_spec_shnorm = try cublas.GpuBuffer(f32).alloc(E);
        try g_spec_shnorm.?.upload(mtp_head.?.shared_head_norm);
        g_head_logits = try cublas.GpuTensor(f32).alloc(vocab);

        // Capa nextn como capa de atención estándar con BlockTable propia
        // (el GGUF trae blk.{N}.attn_* completos; patrón periódico no aplica).
        const seq_draft = try paged_kv.createSequence();
        draft_bt = paged_kv.getBlockTableMut(seq_draft);
        draft_layer = try hybrid_layer.HybridLayer.init(
            allocator,
            eff_blocks,
            hybrid_layer.HybridLayerParams.fromModelConfig(cfg, max_seq_len),
            true,
            backend,
            &paged_kv,
            draft_bt,
            shared_gpu_ptr,
        );
        try draft_layer.?.loadWeightsFromGguf(&model.file, null);
        try stdout.print("[*] spec v1 activa: n_max={d}, cola secuencial sobre decode normal\n", .{params.spec_draft_n_max});
        try stdout.flush();
    }

    // Staging persistente del embedding (f32, PINNED): fuente del H2D normal y
    // de los nodos HtoDAsync capturados por el grafo de decode. Pinneado porque
    // los HtoDAsync con fuente pageable no son capturables por CUDA graphs.
    const embed_staging = try cudaz.pinnedAlloc(f32, n_embd);
    defer cudaz.pinnedFree(f32, embed_staging);

    // U1 (lane-b1): decode híbrido CPU — con --backend cpu el decode no tiene
    // camino (forwardGPU exige paged_gpu; el híbrido sólo tenía decode GPU).
    // Buffers ping-pong host [1, n_embd] para el layer.forward CPU por token.
    const decode_cpu = backend != .cublas;
    var dec_buf_a: ?Tensor(f32) = if (decode_cpu) try Tensor(f32).alloc(allocator, &.{ 1, n_embd }) else null;
    defer if (dec_buf_a) |*t| t.deinit();
    var dec_buf_b: ?Tensor(f32) = if (decode_cpu) try Tensor(f32).alloc(allocator, &.{ 1, n_embd }) else null;
    defer if (dec_buf_b) |*t| t.deinit();

    // ─── CUDA Graphs para el decode (una captura, replay por token) ───────
    // El grafo cubre: embed H2D + 24 capas híbridas + rmsNorm final + lm_head
    // (~290 kernels). Los valores que cambian por token (embedding, block
    // table, start_pos, seq_len) se pintan en staging host persistente y los
    // nodos capturados (HtoDAsync) los copian a buffers device fijos. NOGRAPH=1
    // desactiva; ante error de captura/instanciación se cae al camino normal.
    var decode_g: ?decode_graph.DecodeGraph = null;
    // lane-c (7.1d-regresión): true ⇒ el grafo termina tras rmsNorm (sin
    // lm_head/argmax); el replay hace D2H de g_normed + project fuera.
    var graph_head_off = false;
    // Con especulación activa el replay de grafo single-token no aplica
    // (los drafts se verifican por el camino normal token a token).
    // 8.3 (lane-f): LFM2.5/ShortConv YA es graph-safe — state_parts respalda
    // g_conv_state (doble-buffer @6541b3a: punteros fijos, copy-back DtoD
    // capturable) y forwardGPU/norm/rmsNorm son nodos normales. El gate
    // temporal !lfm2 (mitigación del panic :1883 reportado por coordinador)
    // se retira con el fix de fondo.
    // PERF_STAGE instrumenta con eventos CUDA por-launch — ilegal dentro de
    // graph capture (CudaError en el primer cuEventRecord). Los graphs y el
    // perf-detail son excluyentes; NOGRAPH ya da el perfil kernel-equivalente.
    // F/C v1: el forward MoE v1 hace cuStreamSynchronize + dtoh de ids
    // (moe_layer.zig:19 "v1 fuera de grafo") — ilegales en capture. Con
    // attach activo el decode MoE corre eager (la captura del path denso
    // no lo representa). Se habilita cuando el forward sea graph-safe.
    const moe_attached = moe_layers_attached > 0;
    // U1/C-1 fix: graph capture REQUIERE CUDA — con --backend cpu el
    // paged_gpu de cada capa es null y presizeDecodeScratch devolvía
    // KvCacheNotSet (repro: ZIG_AI_UNIFIED=1 Llama-3.2-3B --backend cpu,
    // stack hybrid_attn.zig:1145). El path CPU no captura graphs.
    // 5.2 (lane-b1): spec_active cubre dflash (dflash_model != null) — el
    // drafter corre forwards fuera del grafo (q4Weight lazy + cuMemAlloc
    // dentro del capture window = 901 en cascada, lección TODO 1.3).
    const graph_ok = backend == .cublas and !debugz.dbg.no_graph and !spec_active and !debugz.dbg.perf_stage and !moe_attached;
    if (graph_ok) {
        // Bloques para el token de la captura (el decode real los re-usa).
        for (layer_block_tables) |bt_opt| {
            if (bt_opt) |bt| {
                if (bt.num_tokens < current_pos + 1) {
                    bt.appendToken(paged_kv.block_alloc) catch |e| {
                        if (e == error.OutOfMemory) printKvOomDiag(paged_kv, num_attn_layers, block_size, max_seq_len, current_pos + 1, stdout);
                        return e;
                    };
                }
            }
        }
        // Commit de TODOS los bloques de la tabla ANTES de capturar: el run de
        // captura llama `ensureBlockCommitted` (cuMemMap/cuMemCreate), que NO
        // es capturable; si encuentra el bloque ya residente no toca el device.
        for (layers) |*layer| {
            if (layer.is_attention) {
                const attn = layer.attn_layer.?;
                if (attn.paged_gpu) |gpu| {
                    for (0..attn.block_table.numBlocks()) |bi| {
                        if (attn.block_table.getPhysical(bi)) |phys| {
                            try gpu.ensureBlockCommitted(paged_kv.block_alloc, phys);
                        }
                    }
                }
            }
        }
        // Staging pre-dimensionado al presupuesto completo de generación: los
        // punteros host y d_bt quedan fijos antes de capturar.
        const budget_seq = seq_len + params.max_new_tokens;
        const budget_blocks = (budget_seq + block_size - 1) / block_size;
        for (layers) |*layer| {
            if (layer.is_attention) {
                try layer.attn_layer.?.presizeDecodeScratch(budget_blocks);
            }
        }
        // Respaldo del estado recurrente ssm (d_s_state + d_conv_state): el
        // run de captura lo corrompe; se restaura tras instanciar el grafo.
        // 8.3 FIX (coordinador report): LFM2.5 tiene ShortConv (no SSM) en
        // las capas no-attention — `ssm_layer.?` panic en null (main.zig:1883,
        // repro: stack trace runHybridInference). El análogo recurrente del
        // ShortConv es g_conv_state (doble-buffer graph-safe @6541b3a: el
        // estado vivo SIEMPRE queda en g_conv_state tras el copy-back DtoD;
        // g_conv_state_next es scratch puro y NO se respalda).
        var state_parts: std.ArrayList(decode_graph.StatePart) = .empty;
        defer state_parts.deinit(allocator);
        for (layers) |*layer| {
            if (!layer.is_attention) {
                if (layer.ssm_layer) |ssm| {
                    if (ssm.gpu) |gpu| {
                        try state_parts.append(allocator, .{ .dev = @intFromPtr(gpu.d_s_state.dev_ptr), .bytes = gpu.d_s_state.len * @sizeOf(f32) });
                        try state_parts.append(allocator, .{ .dev = @intFromPtr(gpu.d_conv_state.dev_ptr), .bytes = gpu.d_conv_state.len * @sizeOf(f32) });
                    }
                } else if (layer.short_conv_layer) |sc| {
                    if (sc.gpu) |sgpu| {
                        try state_parts.append(allocator, .{ .dev = @intFromPtr(sgpu.g_conv_state.dev_ptr), .bytes = sgpu.g_conv_state.len * @sizeOf(f32) });
                    }
                }
            }
        }
        // 6c043a3 añadió el campo `capture` (GraphCapture compuesto): el struct
        // literal dejaba ese campo en `undefined` ⇒ beginCapture fallaba
        // silenciosamente ⇒ 3.3× más lento (sin replay de grafo). RESCATE:
        // usar init() que inicializa capture con el stream correcto.
        var dg: ?decode_graph.DecodeGraph = decode_graph.DecodeGraph.init(allocator, lk.stream, n_embd) catch null;
        if (debugz.dbg.chk_state) {
            try cudaz.cuCtxSynchronize();
            for (state_parts.items) |p| {
                const nf32 = p.bytes / @sizeOf(f32);
                const buf = try allocator.alloc(f32, nf32);
                defer allocator.free(buf);
                try cudaz.cuMemcpyDtoH(@intFromPtr(buf.ptr), p.dev, p.bytes);
                var s: f64 = 0;
                var mx: f32 = 0;
                for (buf) |v| {
                    s += @abs(@as(f64, v));
                    if (@abs(v) > mx) mx = @abs(v);
                }
                debugz.dbg.print("[pipeline] CHKSTATE_BEFORE dev={x} bytes={d} sum|v|={d:.6} max={d:.6}\n", .{ p.dev, p.bytes, s, mx });
            }
        }
        if (dg) |*g| {
            g.setEmbedStaging(embed_staging);
            // Always warmup GPU weights for cublas backend (needed for CUDA graph capture)
            for (layers, 0..) |*layer, li| {
                // T1 (root-cause repro li=59 simbolizado): la cabeza MTP
                // (li >= eff_blocks) NO participa en el forward — calentar
                // sus pesos (bf16, sin kernel qgemm ⇒ subida f32 adicional)
                // agotaba la VRAM al final del pase. Su forward es por el
                // draft driver, que gestiona sus propios buffers.
                if (li >= eff_blocks) break;
                if (streamer) |*s| try s.ensureLayerLoaded(li);
                try layer.warmupGpuWeights();
            }
            // lane-f F2A (LMQ40): bytes efectivos + flag q4 si el re-cuant
            // on-load está activo. El peso q4_0 sube lazy por q4Weight DENTRO
            // del capture (lección 1.3) — pre-warm ANTES del beginCapture.
            const lmq40_bytes: ?[]const u8 = if (lmq40) |*q| q.bytes else null;
            const lm_head_eff_bytes = lmq40_bytes orelse lm_head_q.bytes;
            const lm_head_q4_eff = lm_head_q4 or lmq40_bytes != null;
            if (lmq40_bytes != null) {
                // pre-warm del cache q4Weight fuera del capture
                _ = layer_kernels.q4Weight(allocator, @intFromPtr(lm_head_eff_bytes.ptr), lm_head_eff_bytes) catch {};
                debugz.dbg.printLevel(.detail, "[lmq40] pre-warm q4Weight {d} MB (fuera de capture)\n", .{lm_head_eff_bytes.len / (1024 * 1024)});
            }
            // G2: `gpu_argmax_out` viene YA allocado de más arriba (antes del
            // capture — ver declaración junto a `var lk`).
            // 1.15: gpu_sample cubre greedy (argmax) y temp>0 (gumbel) — el
            // grafo termina en token_id en ambos modos.
            if (captureDecodeGraph(g, &lk, layers, &g_cur, &g_nxt, &g_normed, &g_logits, &g_out_norm, &engine, allocator, n_embd, vocab, rms_eps, current_pos, state_parts.items, lm_head_q4_eff, lm_head_q6k, (lm_head_cpu_fb and !lmq80_force_off) or lmq80_force_on or lmq80 != null, lm_head_eff_bytes, lm_head, gpu_argmax_out, gpu_gumbel, params.sampler.temperature, params.sampler.repetition_penalty)) {
                decode_g = dg;
                graph_head_off = (lm_head_cpu_fb and !lmq80_force_off) or lmq80_force_on or lmq80 != null;
                if (debugz.dbg.chk_state) {
                    try cudaz.cuCtxSynchronize();
                    for (state_parts.items) |p| {
                        const nf32 = p.bytes / @sizeOf(f32);
                        const buf = try allocator.alloc(f32, nf32);
                        defer allocator.free(buf);
                        try cudaz.cuMemcpyDtoH(@intFromPtr(buf.ptr), p.dev, p.bytes);
                        var s: f64 = 0;
                        var mx: f32 = 0;
                        for (buf) |v| {
                            s += @abs(@as(f64, v));
                            if (@abs(v) > mx) mx = @abs(v);
                        }
                        debugz.dbg.print("[pipeline] CHKSTATE dev={x} bytes={d} sum|v|={d:.6} max={d:.6}\n", .{ p.dev, p.bytes, s, mx });
                    }
                }
                try stdout.print("[+] decode: CUDA graph capturado (modo replay)\n", .{});
            } else {
                g.deinit();
            }
        }
    }

    const perf_stage = debugz.dbg.perf_stage;
    var ev: []cudaz.CUevent = undefined;
    var layer_gpu_ns: []i128 = undefined;
    var t_blocks_ns: i128 = 0;
    var t_embed_ns: i128 = 0;
    var t_enqueue_ns: i128 = 0;
    var t_d2h_ns: i128 = 0;
    var t_sample_ns: i128 = 0;
    var gpu_total_ns: i128 = 0;
    var gpu_head_ns: i128 = 0;
    if (perf_stage) {
        ev = try allocator.alloc(cudaz.CUevent, layers.len + 2);
        layer_gpu_ns = try allocator.alloc(i128, layers.len);
        @memset(layer_gpu_ns, 0);
        for (ev) |*e| e.* = try cudaz.cuEventCreate(0);
    }
    const perf_t = @import("time").Timer.start();

    // ════════════════════════════════════════════════════════════════════
    // C4.3 — Generación ESPECULATIVA por rondas batched (lane-c)
    //
    // Ronda: (1) draft de k tokens con la cabeza MTP; (2) UN pase batched
    // [d₁..d_k] por el target (n=k) produciendo a₁..a_k; (3) aceptación del
    // prefijo más largo — d₁ se juzga contra el argmax del ancla guardado;
    // bonus = primer a_m no usado; (4) rollback: truncate de KV + restore
    // DeltaNet si m<k; (5) pase single del bonus que provee hidden+argmax
    // de la ronda siguiente. Techo ≈(k+1)/2× tokens por pase de capas.
    // ════════════════════════════════════════════════════════════════════
    if (spec_active and (h_prev.len == n_embd or dflash_model != null)) {
        // 5.2 (lane-b1): dflash no transfiere spec_hprev (no hay MTP) — el
        // bootstrap necesita un buffer host para el D2H del anchor.
        if (h_prev.len == 0 and dflash_model != null) {
            h_prev = try allocator.alloc(f32, n_embd);
        }
        // 9.5 C4 (lane-f): wiring del ProfitController — `profit` decide
        // n_max dinámicamente por ronda (EWMA aceptación vs costo de
        // ciclo); `off` mantiene el K estático de --spec-draft-n-max.
        var pc: ?specdrv.adaptive_dm.ProfitController = if (params.spec_dm_controller == .profit)
            specdrv.adaptive_dm.ProfitController.init(.{ .base_n_max = @intCast(@min(params.spec_draft_n_max, @as(usize, 16))) })
        else
            null;
        var K_dyn: usize = @min(params.spec_draft_n_max, @as(usize, 16));
        // 9.6: LoopGuard (BeeLlama P1.2) — detector de reasoning loops.
        var loop_guard: specdrv.loop_guard.LoopGuard = if (params.spec_loop_guard_mode != .off)
            specdrv.loop_guard.LoopGuard.init(.{
                .mode = params.spec_loop_guard_mode,
                .max_period = params.spec_loop_guard_max_period,
            })
        else
            specdrv.loop_guard.LoopGuard.init(.{ .mode = .off });
        // Snapshots DeltaNet por capa SSM (rollback de rondas parciales).
        var snaps = try allocator.alloc([]f32, layers.len);
        defer {
            for (snaps) |b| if (b.len > 0) allocator.free(b);
            allocator.free(snaps);
        }
        for (layers, 0..) |*l, li| {
            if (!l.is_attention) {
                // 8.3: LFM2 ShortConv — mismo fix que :1883 (ssm_layer null).
                const n: usize = if (l.ssm_layer) |ssm| ssm.gpuStateLen() else if (l.short_conv_layer) |sc| sc.gpuStateLen() else 0;
                snaps[li] = if (n > 0) try allocator.alloc(f32, n) else &[_]f32{};
            } else snaps[li] = &[_]f32{};
        }
        const K_max = @min(params.spec_draft_n_max, @as(usize, 16));
        var g_vcur = try cublas.GpuTensor(f32).alloc(K_max * n_embd);
        defer g_vcur.deinit();
        var g_vnxt = try cublas.GpuTensor(f32).alloc(K_max * n_embd);
        defer g_vnxt.deinit();
        var vstage = try allocator.alloc(f32, K_max * n_embd);
        defer allocator.free(vstage);
        const vnb = try allocator.alloc(f32, n_embd);
        defer allocator.free(vnb);
        // Logits device/host para el verify batched (K filas × vocab); el
        // device también recibe las proyecciones single-row de
        // bootstrap/bonus con lmq80 (fila 0).
        var g_spec_lmlogits = try cublas.GpuTensor(f32).alloc(K_max * vocab);
        defer g_spec_lmlogits.deinit();
        var spec_lmlogits_host = try allocator.alloc(f32, K_max * vocab);
        defer allocator.free(spec_lmlogits_host);

        var anchor_tok = gen_tokens.items[gen_tokens.items.len - 1];
        var saved_argmax = anchor_tok;
        // 5.5 (lane-f): fila de logits del target en la posición ancla —
        // el paso rejection j=0 la usa como p(x) (hoy sólo se guardaba el
        // argmax). Sólo se asigna en modo rejection.
        var base_logits_row: []f32 = &[_]f32{};
        defer if (base_logits_row.len > 0) allocator.free(base_logits_row);
        if (spec_rejection) base_logits_row = try allocator.alloc(f32, vocab);

        // ─── Bootstrap: ingestir el ancla inicial (first_token), n=1 ────────
        {
            for (layer_block_tables) |bt_opt| {
                if (bt_opt) |bt| {
                    if (bt.num_tokens < current_pos) bt.appendTokens(paged_kv.block_alloc, current_pos - bt.num_tokens) catch |e| {
                        if (e == error.OutOfMemory) printKvOomDiag(paged_kv, num_attn_layers, block_size, max_seq_len, current_pos, stdout);
                        return e;
                    };
                }
            }
            var h1 = try Tensor(f16).alloc(allocator, &.{ 1, 1, n_embd });
            defer h1.deinit();
            Emb.lookup(emb_quant, emb_f16, &[_]u32{anchor_tok}, 1, 1, &h1);
            for (h1.data, 0..) |sv, i| vstage[i] = @floatCast(sv);
            try cudaz.cuMemcpyHtoD(g_vcur.ptr(), @intFromPtr(vstage.ptr), n_embd * @sizeOf(f32));
            var cc = g_vcur;
            var nn = g_vnxt;
            for (layers, 0..) |*layer, li| {
                if (streamer) |*st| try st.ensureLayerLoaded(li);
                try hybrid_layer.HybridLayer.forwardGPU(layer, &lk, cc, &nn, current_pos - 1, 1, null);
                if (kvarn_cache_opt != null) {
                    // M3 slice 2: igual al prefill pero n=1 (decode bs=1).
                    if (layer.kvDevicePtrs()) |kvp| {
                        kvarn_cache_opt.?.appendTokens(@intCast(li), kvp.k, kvp.v, 1, @intCast(current_pos - 1), null, lk.stream) catch |e| {
                            debugz.dbg.printLevel(.info, "[kvarn] decode li={d} pos={d} n=1 appendTokens FALLÓ ({s})\n", .{ li, current_pos - 1, @errorName(e) });
                        };
                    }
                }
                if (streamer) |*st| try st.prefetchNext(li);
                const tmp = cc;
                cc = nn;
                nn = tmp;
            }
            try cudaz.cuStreamSynchronize(lk.stream);
            try cudaz.cuMemcpyDtoH(@intFromPtr(h_prev.ptr), cc.ptr(), n_embd * @sizeOf(f32));
            if (lmq80 != null) {
                cpuGroupedRmsNormFlat(h_prev, out_norm.data, rms_eps, cfg.n_norm_groups, vnb);
                try lmq80.?.project(vnb, g_spec_lmlogits.ptr());
                try cudaz.cuStreamSynchronize(lk.stream);
                try cudaz.cuMemcpyDtoH(@intFromPtr(spec_logits_host.ptr), g_spec_lmlogits.ptr(), vocab * @sizeOf(f32));
                saved_argmax = specdrv.sampler.greedy(spec_logits_host);
                if (spec_rejection) @memcpy(base_logits_row, spec_logits_host);
            } else {
                cpuGroupedRmsNormFlat(h_prev, out_norm.data, rms_eps, cfg.n_norm_groups, vnb);
                cpuLmHeadLogits(vnb, lm_head.data, logits_f32);
                saved_argmax = specdrv.sampler.greedy(logits_f32);
                if (spec_rejection) @memcpy(base_logits_row, logits_f32);
            }
        }

        try stdout.print("[*] Generando (spec batched k={d}{s})... 0 tokens\n", .{ K_dyn, if (params.spec_dm_controller == .profit) " (dm=profit)" else "" });
        try stdout.flush();

        // ─── Rondas especulativas ────────────────────────────────────────────
        while (gen_tokens.items.len < params.max_new_tokens) {
            // ═══ 1) DRAFT: k pasos autorregresivos de la cabeza MTP ═══
            spec_round.draft_base_len = if (draft_bt) |bt| bt.num_tokens else 0;
            spec_round.target_base_pos = current_pos;
            // 5.2: con DFlash el draft_bt es del POOL del draft (dflash_kv)
            // — el kvInject del draft-model gestiona su alineación; el
            // alineado-agresivo + ensureBlockCommitted del pool TARGET es
            // sólo del camino MTP.
            const draft_bt_is_dflash = dflash_model != null;
            if (!draft_bt_is_dflash and draft_bt.?.num_tokens < current_pos) {
                draft_bt.?.appendTokens(paged_kv.block_alloc, current_pos - draft_bt.?.num_tokens) catch |e| {
                    if (e == error.OutOfMemory) printKvOomDiag(paged_kv, num_attn_layers, block_size, max_seq_len, current_pos, stdout);
                    return e;
                };
                if (shared_gpu_ptr) |gpu| {
                    if (draft_bt.?.getPhysical(draft_bt.?.numBlocks() - 1)) |phys| {
                        try gpu.ensureBlockCommitted(paged_kv.block_alloc, phys);
                    }
                }
            }
            spec_round.queue.clearRetainingCapacity();
            var draft_tok = anchor_tok;
            var j: usize = 0;
            const t_round0 = perf_t.read();
            const t_draft0 = t_round0;
            const K = K_dyn; // 9.5 C4: n_max de ESTA ronda (decidido al cierre de la anterior)
            // NOTA 9.5 (remanente, no bloqueante): K==0 (shutdown del
            // ProfitController) sigue corriendo la ronda degenerada draft
            // 0 + verify n=0 + bonus — el coste es el machinery batched por
            // token (§ C4 overhead medido en controller_overhead.zig).
            // Skip al decode directo requiere factorizar el camino no-spec
            // (emit/loop-guard/EOS/anchor) fuera de la ronda: refactor que
            // se hará con GPU valida (A/B verificable). El ProfitController
            // puede reactivar vía probe (re-baseline → recordRound).
            // ═══ 1b) DRAFT DFlash (5.2 lane-b1): denoise batched no-causal ═══
            // KV del target inyectado (kvInject vía taps del último token) +
            // denoise batched: N anchors → logits [N×(bs-1), vocab] en una
            // sola pasada de capas sidecar. Greedy con truncado p_min.
            if (dflash_model) |*df| {
                const fused = try allocator.alloc(f32, n_embd);
                defer allocator.free(fused);
                try df.encoderForward(undefined, dflash_taps, 1, fused);
                try df.kvInject(fused, 1, df.kvLen());
                const bs = df.block_size;
                const df_rows_per = bs - 1;
                const max_draft = @min(df_rows_per, @min(params.spec_draft_n_max, @as(usize, 64)));
                // anchors: mínimo 1 (anchor_tok) + los drafts aceptados en
                // rondas anteriores de ESTA ronda (cola actual). Como DFlash
                // no tiene draft-loop iterativo (una pasada fija), usamos
                // solo anchor_tok + 0 drafts previos por ahora.
                const anchors = &[_]u32{anchor_tok};
                const df_logits = try allocator.alloc(f32, anchors.len * df_rows_per * vocab);
                defer allocator.free(df_logits);
                const total_rows = try df.denoiseDraftBatched(anchors, df_logits);
                const df_n = @min(total_rows, max_draft);
                var df_tokens: [64]u32 = undefined;
                const n_drafted = spec_driver.draftSidecarGreedy(df_logits, vocab, df_tokens[0..df_n]) catch 0;
                for (df_tokens[0..n_drafted]) |t| {
                    try spec_round.queue.append(allocator, t);
                }
                if (debugz.dbg.dump_spec or params.spec_p_min > 0) {
                    debugz.dbg.printLevel(.info, "[spec] dflash draft batched: {d} tokens ({d:.1} ms), kv_len={d}\n", .{ n_drafted, @as(f64, @floatFromInt(perf_t.read() - t_draft0)) / 1e6, df.kvLen() });
                }
                draft_tok = if (n_drafted > 0) df_tokens[n_drafted - 1] else anchor_tok;
                j = n_drafted;
                // j >= K salta el draft MTP; el VERIFY batched de abajo es común
            }
            while (j < K) : (j += 1) {
                if (emb_quant != null) {
                    var erow = try Tensor(f16).alloc(allocator, &.{ 1, 1, n_embd });
                    defer erow.deinit();
                    Emb.lookup(emb_quant, null, &[_]u32{draft_tok}, 1, 1, &erow);
                    for (erow.data, emb_row_f32) |s16, *d32| d32.* = @floatCast(s16);
                } else {
                    const emb_src = emb_f16.?.data[@as(usize, draft_tok) * n_embd ..][0..n_embd];
                    for (emb_src, emb_row_f32) |s16, *d32| d32.* = @floatCast(s16);
                }
                mtp_head.?.fuseAndProject(eh_w, emb_row_f32, h_prev, head_x0_host, head_concat);
                try cudaz.cuMemcpyHtoD(g_head_in.?.ptr(), @intFromPtr(head_x0_host.ptr), n_embd * @sizeOf(f32));
                const pos_j = current_pos + j;
                try draft_bt.?.prepareWrite(paged_kv.block_alloc);
                draft_bt.?.appendToken(paged_kv.block_alloc) catch |e| {
                    if (e == error.OutOfMemory) printKvOomDiag(paged_kv, num_attn_layers, block_size, max_seq_len, current_pos + 1, stdout);
                    return e;
                };
                if (shared_gpu_ptr) |gpu| {
                    if (draft_bt.?.getPhysical(draft_bt.?.numBlocks() - 1)) |phys| {
                        try gpu.ensureBlockCommitted(paged_kv.block_alloc, phys);
                    }
                }
                try hybrid_layer.HybridLayer.forwardGPU(&draft_layer.?, &lk, g_head_in.?, &g_head_out.?, pos_j, 1, null);
                try lk.rmsNorm(g_head_out.?.ptr(), @intFromPtr(g_spec_shnorm.?.dev_ptr), g_normed.ptr(), 1, n_embd, rms_eps);
                try cudaz.cuStreamSynchronize(lk.stream);
                if (lmq80 != null) {
                    const nb = try allocator.alloc(f32, n_embd);
                    defer allocator.free(nb);
                    try cudaz.cuMemcpyDtoH(@intFromPtr(nb.ptr), g_normed.ptr(), n_embd * @sizeOf(f32));
                    try lmq80.?.project(nb, g_head_logits.?.ptr());
                    try cudaz.cuStreamSynchronize(lk.stream);
                    try cudaz.cuMemcpyDtoH(@intFromPtr(spec_logits_host.ptr), g_head_logits.?.ptr(), vocab * @sizeOf(f32));
                } else if (lm_head_cpu_fb) {
                    try cudaz.cuMemcpyDtoH(@intFromPtr(vnb.ptr), g_normed.ptr(), n_embd * @sizeOf(f32));
                    cpuLmHeadLogits(vnb, lm_head.data, spec_logits_host);
                } else if (lm_head_q4) {
                    try lk.q4gemmLinear(allocator, g_normed.ptr(), lm_head_q.bytes, g_head_logits.?.ptr(), n_embd, vocab);
                    try cudaz.cuStreamSynchronize(lk.stream);
                    try cudaz.cuMemcpyDtoH(@intFromPtr(spec_logits_host.ptr), g_head_logits.?.ptr(), vocab * @sizeOf(f32));
                } else if (lm_head_q6k) {
                    try lk.qgemmLinear(allocator, g_normed.ptr(), lm_head_q.bytes, g_head_logits.?.ptr(), 1, n_embd, vocab, 3);
                    try cudaz.cuStreamSynchronize(lk.stream);
                    try cudaz.cuMemcpyDtoH(@intFromPtr(spec_logits_host.ptr), g_head_logits.?.ptr(), vocab * @sizeOf(f32));
                } else {
                    try engine.linearProjectionDeviceF16(g_normed, lm_head, &g_head_logits.?, 1, n_embd, vocab);
                    try cudaz.cuStreamSynchronize(lk.stream);
                    try cudaz.cuMemcpyDtoH(@intFromPtr(spec_logits_host.ptr), g_head_logits.?.ptr(), vocab * @sizeOf(f32));
                }
                draft_tok = specdrv.sampler.greedy(spec_logits_host);
                // 5.5 (lane-f): snapshot de la fila q(x) para el verify
                // rejection (spec_logits_host se reusa en el siguiente paso).
                if (spec_rejection and j < draft_logits_rows.len / vocab) {
                    @memcpy(draft_logits_rows[j * vocab ..][0..vocab], spec_logits_host);
                }
                // Confianza del drafter + top-3 (T1) y truncado p_min (T4):
                // un paso con confianza baja contamina el resto de la cola.
                const conf = specdrv.sampler.softmaxConfidence(spec_logits_host, draft_tok);
                if (debugz.dbg.dump_spec or params.spec_p_min > 0) {
                    var t3i: [3]u32 = undefined;
                    var t3v: [3]f32 = undefined;
                    specdrv.sampler.topK(spec_logits_host, &t3i, &t3v);
                    if (debugz.dbg.dump_spec) {
                        debugz.dbg.printLevel(.detail, "[spec] paso {d}: tok={d} conf={d:.3} top3={d:.3}/{d:.3}/{d:.3} @{d},{d},{d}\n", .{ j, draft_tok, conf, t3v[0], t3v[1], t3v[2], t3i[0], t3i[1], t3i[2] });
                    }
                    if (j > 0 and params.spec_p_min > 0 and conf < params.spec_p_min) {
                        if (debugz.dbg.dump_spec) {
                            debugz.dbg.printLevel(.detail, "[spec] confianza {d:.3} < p_min {d:.3}: corto la cola en {d}\n", .{ conf, params.spec_p_min, j });
                        }
                        break;
                    }
                }
                try spec_round.queue.append(allocator, draft_tok);
                try cudaz.cuMemcpyDtoH(@intFromPtr(h_prev.ptr), g_head_out.?.ptr(), n_embd * @sizeOf(f32));
            }
            const kq = spec_round.pending();
            if (debugz.dbg.perf_spec) {
                debugz.dbg.printLevel(.info, "[spec] PERF draft: {d} tokens en {d:.1} ms\n", .{ kq, @as(f64, @floatFromInt(perf_t.read() - t_draft0)) / 1e6 });
            }
            if (debugz.dbg.dump_spec) {
                debugz.dbg.print("[spec] cola draft: ", .{});
                for (spec_round.queue.items) |t| debugz.dbg.print("{d} ", .{t});
                debugz.dbg.print("\n", .{});
            }
            if (kq == 0) break;

            // ═══ 2) VERIFY batched: un pase del target con n=kq ═══
            for (layers, 0..) |*l, li| {
                if (!l.is_attention) {
                    // 8.3: ShortConv (LFM2) — misma interfaz snapshot que SSM.
                    if (snaps[li].len > 0) {
                        if (l.ssm_layer) |ssm| {
                            try ssm.snapshotGpuState(snaps[li]);
                        } else if (l.short_conv_layer) |*sc| {
                            try sc.snapshotGpuState(snaps[li]);
                        }
                    }
                }
            }
            for (layer_block_tables) |bt_opt| {
                if (bt_opt) |bt| {
                    if (bt.num_tokens < current_pos + kq) bt.appendTokens(paged_kv.block_alloc, current_pos + kq - bt.num_tokens) catch |e| {
                        if (e == error.OutOfMemory) printKvOomDiag(paged_kv, num_attn_layers, block_size, max_seq_len, current_pos + kq, stdout);
                        return e;
                    };
                }
            }
            {
                var erows = try Tensor(f16).alloc(allocator, &.{ 1, spec_round.queue.items.len, n_embd });
                defer erows.deinit();
                Emb.lookup(emb_quant, emb_f16, spec_round.queue.items, 1, spec_round.queue.items.len, &erows);
                for (erows.data, vstage[0 .. spec_round.queue.items.len * n_embd]) |s16, *d32| d32.* = @floatCast(s16);
            }
            // 5.2 (lane-b1): verify batched CPU — mismo forward host que el
            // prefill CPU (n=kq tokens de una vez); logits por fila en host.
            var verify_logits_cpu: []f32 = &[_]f32{};
            defer if (verify_logits_cpu.len > 0) allocator.free(verify_logits_cpu);
            var cc = g_vcur;
            var nn = g_vnxt;
            if (decode_cpu) {
                var vbuf_a = try Tensor(f32).alloc(allocator, &.{ kq, n_embd });
                defer vbuf_a.deinit();
                var vbuf_b = try Tensor(f32).alloc(allocator, &.{ kq, n_embd });
                defer vbuf_b.deinit();
                @memcpy(vbuf_a.data, vstage[0 .. kq * n_embd]);
                var vcur = &vbuf_a;
                var vnxt = &vbuf_b;
                for (layers, 0..) |*layer, li| {
                    // Taps del ÚLTIMO token del verify (para la próxima ronda
                    // dflash): entrada pre-norm de cada capa target_layers.
                    if (dflash_model != null and dflash_taps.len >= n_embd) {
                        const want = blk_tl: {
                            if (dflash_model) |*df| {
                                for (df.target_layers, 0..) |tl, ti| {
                                    if (tl == @as(i32, @intCast(li))) {
                                        @memcpy(dflash_taps[ti * n_embd ..][0..n_embd], vcur.data[(kq - 1) * n_embd ..][0..n_embd]);
                                        break :blk_tl true;
                                    }
                                }
                            }
                            break :blk_tl false;
                        };
                        _ = want;
                    }
                    if (streamer) |*st| try st.ensureLayerLoaded(li);
                    try layer.forward(vcur.*, vnxt, current_pos, kq, null);
                    if (streamer) |*st| try st.prefetchNext(li);
                    const tmp = vcur;
                    vcur = vnxt;
                    vnxt = tmp;
                }
                // Logits por fila: output_norm + lm_head CPU → argmax por posición
                verify_logits_cpu = try allocator.alloc(f32, kq * vocab);
                {
                    const h = try allocator.alloc(f32, n_embd);
                    defer allocator.free(h);
                    for (0..kq) |r| {
                        @memcpy(h, vcur.data[r * n_embd ..][0..n_embd]);
                        cpuGroupedRmsNormFlat(h, out_norm.data, rms_eps, cfg.n_norm_groups, h);
                        cpuLmHeadLogits(h, lm_head.data, verify_logits_cpu[r * vocab ..][0..vocab]);
                    }
                }
                // Los argmax por fila los computa el bloque común de abajo
                // (a_rows) leyendo verify_logits_cpu cuando decode_cpu.
            } else {
                try cudaz.cuMemcpyHtoD(g_vcur.ptr(), @intFromPtr(vstage.ptr), kq * n_embd * @sizeOf(f32));
                for (layers, 0..) |*layer, li| {
                    if (streamer) |*st| try st.ensureLayerLoaded(li);
                    try hybrid_layer.HybridLayer.forwardGPU(layer, &lk, cc, &nn, current_pos, kq, null);
                    if (streamer) |*st| try st.prefetchNext(li);
                    const tmp = cc;
                    cc = nn;
                    nn = tmp;
                }
                try cudaz.cuStreamSynchronize(lk.stream);
            } // else GPU verify

            // a[j] = argmax de la fila j (predice la posición de d_{j+1})
            const a_rows = try allocator.alloc(u32, kq);
            defer allocator.free(a_rows);
            if (decode_cpu) {
                // 5.2 (lane-b1): CPU — el forward batched host ya dejó los
                // logits por fila en verify_logits_cpu.
                for (0..kq) |jj| {
                    a_rows[jj] = specdrv.sampler.greedy(verify_logits_cpu[jj * vocab ..][0..vocab]);
                    if (spec_rejection) @memcpy(spec_lmlogits_host[jj * vocab ..][0..vocab], verify_logits_cpu[jj * vocab ..][0..vocab]);
                }
            } else if (lmq80 != null) {
                // C4-tune: norma host por fila (in-place sobre el staging,
                // ya consumido por el H2D de embeddings) + UNA proyección
                // device batched M≤8 por trozo (kernel B6 accf[8]) en vez
                // de kq CPU-GEMVs.
                try cudaz.cuMemcpyDtoH(@intFromPtr(vstage.ptr), cc.ptr(), kq * n_embd * @sizeOf(f32));
                const mb_max = lmq80.?.maxBatchFor();
                var off: usize = 0;
                while (off < kq) : (off += mb_max) {
                    const mb = @min(kq - off, mb_max);
                    for (0..mb) |r| {
                        const row = vstage[(off + r) * n_embd ..][0..n_embd];
                        cpuGroupedRmsNormFlat(row, out_norm.data, rms_eps, cfg.n_norm_groups, row);
                    }
                    const dst = g_spec_lmlogits.ptr() + off * vocab * @sizeOf(f32);
                    try lmq80.?.projectBatched(vstage[off * n_embd ..], mb, dst);
                }
                try cudaz.cuStreamSynchronize(lk.stream);
                try cudaz.cuMemcpyDtoH(@intFromPtr(spec_lmlogits_host.ptr), g_spec_lmlogits.ptr(), kq * vocab * @sizeOf(f32));
                for (0..kq) |jj| {
                    a_rows[jj] = specdrv.sampler.greedy(spec_lmlogits_host[jj * vocab ..][0..vocab]);
                }
            } else {
                for (0..kq) |jj| {
                    const row_ptr = cc.ptr() + jj * n_embd * @sizeOf(f32);
                    try cudaz.cuMemcpyDtoH(@intFromPtr(vnb.ptr), row_ptr, n_embd * @sizeOf(f32));
                    cpuGroupedRmsNormFlat(vnb, out_norm.data, rms_eps, cfg.n_norm_groups, vnb);
                    cpuLmHeadLogits(vnb, lm_head.data, logits_f32);
                    a_rows[jj] = specdrv.sampler.greedy(logits_f32);
                    // 5.5 (lane-f): el camino CPU-GEMV del verify escribía
                    // logits_f32 y dejaba spec_lmlogits_host SIN rellenar —
                    // el modo rejection leía filas a cero ⇒ softmax uniforme
                    // ⇒ bonus degenerado (basura multilingüe, t_max=-0.000
                    // en DBG). Espejo la fila para que el rejection tenga
                    // p(x) real en TODOS los caminos de lm_head.
                    if (spec_rejection) @memcpy(spec_lmlogits_host[jj * vocab ..][0..vocab], logits_f32);
                }
            }

            // ═══ 3) ACEPTACIÓN: prefijo más largo + bonus ═══
            // 5.5 (lane-f): dos modos.
            // · greedy (temp<=0, original C4.2): prefijo más largo con
            //   argmax exacto — judge(m)=saved_argmax/a_rows[m−1].
            // · rejection (temp>0): por posición, accept = min(1, p/q)
            //   con u~U(0,1); rechazo = resample del residual max(0,p−q)
            //   (sampler.rejectionStep, semántica llama.cpp). El token
            //   final de cada paso viene del StepResult.
            var m: usize = 0;
            var bonus: u32 = if (kq > 0) spec_round.queue.items[0] else anchor_tok;
            if (spec_rejection) {
                // Scratches softmax p/q (rejectionStep los reusa internamente;
                // q se sobreescribe por paso — tamaño vocab cada uno).
                const p_scr = try allocator.alloc(f32, vocab);
                defer allocator.free(p_scr);
                const q_scr = try allocator.alloc(f32, vocab);
                defer allocator.free(q_scr);
                while (m < kq) : (m += 1) {
                    const t_row: []const f32 = if (m == 0)
                        base_logits_row
                    else
                        spec_lmlogits_host[(m - 1) * vocab ..][0..vocab];
                    const d_row = draft_logits_rows[m * vocab ..][0..vocab];
                    const stp = try specdrv.sampler.rejectionStep(t_row, d_row, p_scr, q_scr, rng.random());
                    if (debugz.dbg.dump_spec) {
                        debugz.dbg.print("[spec] pos={d} cand={d} rejection -> {s} tok={d}\n", .{ current_pos + m, spec_round.queue.items[m], if (stp.accepted) "ACEPTA" else "rechaza", stp.token });
                        if (debugz.dbg.at(.trace)) {
                            // DBG 5.5: sanity p/q — máximos y valor del candidato.
                            var pm: f32 = -1e30;
                            var qm: f32 = -1e30;
                            for (t_row) |v| pm = @max(pm, v);
                            for (d_row) |v| qm = @max(qm, v);
                            debugz.dbg.print("[spec]   DBG logits: t_max={d:.3} d_max={d:.3} t_cand={d:.3} d_cand={d:.3}\n", .{ pm, qm, t_row[spec_round.queue.items[m]], d_row[spec_round.queue.items[m]] });
                        }
                    }
                    if (!stp.accepted) {
                        // El StepResult trae el token resampleado del
                        // residual: es el bonus de esta ronda.
                        bonus = stp.token;
                        break;
                    }
                }
                if (m == kq) {
                    // Todos aceptados: el bonus sale de la última fila del
                    // target (distribución de la posición siguiente).
                    const last_row = spec_lmlogits_host[(kq - 1) * vocab ..][0..vocab];
                    bonus = params.sampler.sample(last_row, &rng, gen_tokens.items);
                }
            } else {
                while (m < kq) : (m += 1) {
                    const judge = if (m == 0) saved_argmax else a_rows[m - 1];
                    if (spec_round.queue.items[m] != judge) break;
                }
                bonus = if (m == 0) saved_argmax else a_rows[m - 1];
            }

            if (debugz.dbg.dump_spec) {
                for (spec_round.queue.items, 0..) |cand, jj| {
                    const judge = if (jj == 0) saved_argmax else a_rows[jj - 1];
                    const ok = jj < m;
                    debugz.dbg.print("[spec] pos={d} cand={d} argmax={d} -> {s}\n", .{ current_pos + jj, cand, judge, if (ok) "ACEPTA" else if (jj == m) "rechaza" else "descarta" });
                }
                debugz.dbg.print("[spec] bonus={d}\n", .{bonus});
            }
            spec_driver.metrics.total_drafted += kq;
            spec_driver.metrics.total_accepted += m;
            spec_driver.metrics.total_bonus += 1;
            spec_driver.metrics.source_drafter += m;
            spec_driver.metrics.total_rejected_step += kq - m;
            spec_driver.metrics.total_rounds += 1;

            // 9.5 C4: alimentar el ProfitController con la observación de
            // esta ronda y decidir el n_max de la siguiente.
            if (pc) |*p| {
                const round_ns = perf_t.read() - t_round0;
                p.recordRound(.{
                    .requested_n_max = @intCast(kq),
                    .n_draft = @intCast(kq),
                    .n_accepted = @intCast(m + 1), // bonus cuenta como aceptado del verify
                    .draft_ms = @as(f32, @floatFromInt(perf_t.read() - t_draft0)) / 1e6,
                    .verify_ms = 0, // verify integrado en el ciclo (medición por etapas: PERF_SPEC)
                    .accept_ms = 0,
                    .cycle_ms = @as(f32, @floatFromInt(round_ns)) / 1e6,
                });
                const dr = p.decide();
                if (dr.recommended_n_max >= 0) K_dyn = @intCast(dr.recommended_n_max);
                if (debugz.dbg.dump_spec) {
                    debugz.dbg.printLevel(.info, "[spec] dm: n={d} accepted={d}/{d} cyc={d:.1}ms → next n_max={d} ({s})\n", .{ dr.recommended_n_max, m + 1, kq, @as(f32, @floatFromInt(round_ns)) / 1e6, K_dyn, dr.reason });
                }
            }

            // Commit: drafts aceptados + bonus
            var spec_stop_hit = false;
            for (spec_round.queue.items[0..m]) |tok_c| {
                try gen_tokens.append(allocator, tok_c);
                try scheduler.appendToken(seq_id, tok_c);
                // Server F2 T2c: emit de drafts aceptados por ronda.
                if (sink) |s| {
                    const piece = tok.decode(&[_]u32{tok_c}, allocator) catch "";
                    defer if (piece.len > 0) allocator.free(piece);
                    s.emit(tok_c, piece) catch {
                        spec_stop_hit = true;
                        break;
                    };
                }
            }
            if (!spec_stop_hit) {
                try gen_tokens.append(allocator, bonus);
                try scheduler.appendToken(seq_id, bonus);
                if (sink) |s| {
                    const piece = tok.decode(&[_]u32{bonus}, allocator) catch "";
                    defer if (piece.len > 0) allocator.free(piece);
                    s.emit(bonus, piece) catch {
                        spec_stop_hit = true;
                    };
                }
            }
            current_pos += m + 1;

            // 9.6: LoopGuard — feed tokens commitidos y check.
            if (params.spec_loop_guard_mode != .off) {
                for (spec_round.queue.items[0..m]) |t| loop_guard.feed(t);
                loop_guard.feed(bonus);
                const det = loop_guard.check();
                if (det.triggered) {
                    if (params.spec_loop_guard_mode == .force_close) {
                        debugz.dbg.printLevel(.info, "[loop-guard] trigger period={d} score={d:.3} — force-close\n", .{ det.period, det.score });
                        break;
                    } else {
                        try stdout.print("[!] loop detectado: periodo {d}, score {d:.3} — continuando (warn mode)\n", .{ det.period, det.score });
                    }
                }
            }

            // P0-RPERF: KV spill to CPU — periodic offload of cold blocks
            kv_offload.maybeSpill() catch {};

            // EOS sobre lo commitido esta ronda
            var hit_eos = false;
            if (gt.eos_id) |eos| {
                for (gen_tokens.items[gen_tokens.items.len - (m + 1) ..]) |tc| {
                    if (tc == eos) hit_eos = true;
                }
            }

            // ═══ 4) ROLLBACK al timeline commitido ═══
            for (layer_block_tables) |bt_opt| {
                if (bt_opt) |bt| {
                    if (bt.num_tokens > current_pos) try bt.truncate(paged_kv.block_alloc, current_pos);
                }
            }
            if (!draft_bt_is_dflash and draft_bt.?.num_tokens > current_pos) try draft_bt.?.truncate(paged_kv.block_alloc, current_pos);
            // 5.2 (lane-b1): DFlash — el KV del draft (pool propio dflash_kv)
            // también trunca al timeline commitido (oráculo: "los drafts
            // rechazados se revierten truncando el cache draft").
            if (dflash_model) |*df| {
                try df.rollbackTo(current_pos);
            }
            if (m < kq) {
                for (layers, 0..) |*l, li| {
                    if (!l.is_attention) {
                        // 8.3: ShortConv (LFM2) — rollback equivalente al SSM.
                        if (snaps[li].len > 0) {
                            if (l.ssm_layer != null) {
                                try l.ssm_layer.?.restoreGpuState(snaps[li]);
                            } else if (l.short_conv_layer != null) {
                                try l.short_conv_layer.?.restoreGpuState(snaps[li]);
                            }
                        }
                    }
                }
            }
            spec_round.reset();

            if (hit_eos) break;

            // ═══ 5) Pase single del bonus: provee ancla (hidden+argmax) ═══
            for (layer_block_tables) |bt_opt| {
                if (bt_opt) |bt| {
                    if (bt.num_tokens < current_pos) bt.appendTokens(paged_kv.block_alloc, current_pos - bt.num_tokens) catch |e| {
                        if (e == error.OutOfMemory) printKvOomDiag(paged_kv, num_attn_layers, block_size, max_seq_len, current_pos, stdout);
                        return e;
                    };
                }
            }
            anchor_tok = bonus;
            var hb = try Tensor(f16).alloc(allocator, &.{ 1, 1, n_embd });
            defer hb.deinit();
            Emb.lookup(emb_quant, emb_f16, &[_]u32{anchor_tok}, 1, 1, &hb);
            for (hb.data, 0..) |sv, i| vstage[i] = @floatCast(sv);
            try cudaz.cuMemcpyHtoD(g_vcur.ptr(), @intFromPtr(vstage.ptr), n_embd * @sizeOf(f32));
            cc = g_vcur;
            nn = g_vnxt;
            for (layers, 0..) |*layer, li| {
                if (streamer) |*st| try st.ensureLayerLoaded(li);
                try hybrid_layer.HybridLayer.forwardGPU(layer, &lk, cc, &nn, current_pos - 1, 1, null);
                if (streamer) |*st| try st.prefetchNext(li);
                const tmp = cc;
                cc = nn;
                nn = tmp;
            }
            try cudaz.cuStreamSynchronize(lk.stream);
            try cudaz.cuMemcpyDtoH(@intFromPtr(h_prev.ptr), cc.ptr(), n_embd * @sizeOf(f32));
            if (lmq80 != null) {
                cpuGroupedRmsNormFlat(h_prev, out_norm.data, rms_eps, cfg.n_norm_groups, vnb);
                try lmq80.?.project(vnb, g_spec_lmlogits.ptr());
                try cudaz.cuStreamSynchronize(lk.stream);
                try cudaz.cuMemcpyDtoH(@intFromPtr(spec_logits_host.ptr), g_spec_lmlogits.ptr(), vocab * @sizeOf(f32));
                saved_argmax = specdrv.sampler.greedy(spec_logits_host);
                if (spec_rejection) @memcpy(base_logits_row, spec_logits_host);
            } else {
                cpuGroupedRmsNormFlat(h_prev, out_norm.data, rms_eps, cfg.n_norm_groups, vnb);
                cpuLmHeadLogits(vnb, lm_head.data, logits_f32);
                saved_argmax = specdrv.sampler.greedy(logits_f32);
                if (spec_rejection) @memcpy(base_logits_row, logits_f32);
            }
        }

        try stdout.print("[spec] aceptación: ", .{});
        try spec_driver.reportMetrics(stdout);
    }
    var host_logits_ready = false;
    // 9.6: LoopGuard para path no-spec.
    var loop_guard_ns: specdrv.loop_guard.LoopGuard = if (params.spec_loop_guard_mode != .off)
        specdrv.loop_guard.LoopGuard.init(.{
            .mode = params.spec_loop_guard_mode,
            .max_period = params.spec_loop_guard_max_period,
        })
    else
        specdrv.loop_guard.LoopGuard.init(.{ .mode = .off });
    if (!spec_active) {
        for (0..params.max_new_tokens) |_| {
            const last = gen_tokens.items[gen_tokens.items.len - 1];

            // E3 (lane-e, 4.12 runtime): protocolo destroy-graphs→resize→
            // recapture bajo presión de VRAM (patrón FreeToken). El check
            // es barato (cuMemGetInfo) y el rebuild solo dispara bajo el
            // floor 512MB. Con MoE attachado el grafo ya está desactivado
            // (graph_ok gatea !moe_attached) ⇒ aquí solo redimensiona el
            // cache; el invalidate defensivo cubre el futuro MoE graph-safe.
            if (moe_cache_gpu != null) {
                const rb = moe_cuda.moeMaybeRebuildUnderPressure(moe_cache_gpu, if (decode_g) |*g| g else null, lk.stream);
                if (rb.resized) {
                    // El replay pasaría por punteros muertos: forzar el
                    // camino eager este token; la re-captura la hace el
                    // bloque pre-decode si vuelve a haber VRAM (hoy null).
                    if (decode_g) |*g| {
                        g.deinit();
                        decode_g = null;
                    }
                    try stdout.print("[!] MoE cache redimensionado a {d} slots bajo presión VRAM (decode eager este token)\n", .{rb.new_cache_size});
                }
            }

            // Ensure blocks for the new decode token before forward
            const t_block0 = perf_t.read();
            for (layer_block_tables) |bt_opt| {
                if (bt_opt) |bt| {
                    if (bt.num_tokens < current_pos + 1) {
                        try bt.appendToken(paged_kv.block_alloc);
                    }
                }
            }
            t_blocks_ns += perf_t.read() - t_block0;

            // P0-RPERF: KV spill to CPU — periodic offload of cold blocks
            kv_offload.maybeSpill() catch {};

            const t_embed0 = perf_t.read();
            var h1 = try Tensor(f16).alloc(allocator, &.{ 1, 1, n_embd });
            defer h1.deinit();
            Emb.lookup(emb_quant, emb_f16, &[_]u32{last}, 1, 1, &h1);
            const h2d = try h1.reshape(&[_]usize{ 1, n_embd });
            defer {
                if (h2d.allocator) |a| {
                    a.free(h2d.shape);
                    a.free(h2d.strides);
                }
            }

            // La primera capa recibe el embedding del token actual (f16 → f32).
            // Staging persistente: fuente del H2D normal y de los nodos capturados.
            for (embed_staging, h2d.data) |*d, s| d.* = @as(f32, @floatCast(s));
            t_embed_ns += perf_t.read() - t_embed0;

            const t_enq0 = perf_t.read();
            if (decode_g) |*g| {
                // Modo replay: staging host del decode (block table/start_pos/seq_len)
                // + commit de bloques, luego un único cuGraphLaunch con todo el token
                // (embed H2D + capas + rmsNorm + lm_head) en nodos capturados.
                for (layers) |*layer| {
                    if (layer.is_attention) try layer.attn_layer.?.stageDecodeHost(current_pos, 1);
                }
                // §5.7 STUDY: el cuEventRecord alrededor de un cuGraphLaunch
                // seguido de sync explícito devuelve CudaError — solo tiene
                // sentido instrumentar lanzamientos por-kernel (NOGRAPH=1).
                // PERF_STAGE + replay ⇒ medir sólo el extremo del grafo entero.
                try g.launch();
                // BUG-C (STATE.md): en modo replay, el D2H de syncDecodeBlocks veía
                // el pool host desactualizado pese al orden de stream. Sincronizar
                // explícitamente tras el grafo garantiza visibilidad de los writes
                // KV antes de emitir las copias. Coste ≈ 0: ya se sincroniza por token.
                try cudaz.cuStreamSynchronize(lk.stream);
                if (perf_stage and debugz.dbg.no_graph) try cudaz.cuEventRecord(ev[layers.len + 1], lk.stream);
                // lane-c (7.1d-regresión): grafo head-off — el lm_head no está
                // capturado (cpu_fb/lmq80). D2H del normed + project fuera del
                // grafo, mismo contrato que el camino no-graph de abajo.
                if (graph_head_off) {
                    const nb = try allocator.alloc(f32, n_embd);
                    defer allocator.free(nb);
                    try cudaz.cuMemcpyDtoH(@intFromPtr(nb.ptr), g_normed.ptr(), n_embd * @sizeOf(f32));
                    if (lmq80 != null) {
                        try lmq80.?.project(nb, g_logits.ptr());
                        // El project lmq80 escribe logits en device (GEMV q8_0).
                        if (gpu_argmax) {
                            try lk.argmaxF32(g_logits.ptr(), gpu_argmax_out, 1, vocab);
                        } else if (gpu_gumbel) {
                            try lk.sampleF32Gumbel(g_logits.ptr(), gpu_argmax_out, params.sampler.temperature, params.sampler.repetition_penalty, 1, vocab);
                        } else {
                            try cudaz.cuMemcpyDtoH(@intFromPtr(logits_f32.ptr), g_logits.ptr(), vocab * @sizeOf(f32));
                            host_logits_ready = true;
                        }
                    } else {
                        cpuRmsNormFlat(nb, out_norm.data, rms_eps, nb);
                        cpuLmHeadLogits(nb, lm_head.data, logits_f32);
                        host_logits_ready = true;
                    }
                }
            } else {
                // Camino normal: subir embedding (única H2D por token) y lanzar la
                // secuencia capa a capa.
                // U1 (lane-b1): decode CPU — mismo forward host que el prefill CPU
                // (layer.forward soporta start_pos + KV), norma final + lm_head en
                // host (patrón lm_head_cpu_fb). Escribe logits_f32 y sale ANTES de
                // cualquier llamada CUDA (presize/syncDecode/argmax device).
                if (decode_cpu) {
                    var cur_h = &(dec_buf_a.?);
                    var nxt_h = &(dec_buf_b.?);
                    @memcpy(cur_h.data, embed_staging);
                    // Índice de la próxima tap a llenar (ronda-robin simple:
                    // el encoder del oráculo espera las taps de las capas
                    // target_layers INTERLEAVEADAS por token — aquí el decode
                    // es de UN token: una tap por capa target).
                    var tap_slot: usize = 0;
                    for (layers, 0..) |*layer, li| {
                        // 5.2 (lane-b1): tap = ENTRADA pre-norm de la capa li
                        // (cur_h en este punto) si li ∈ target_layers del
                        // sidecar y queda slot en dflash_taps.
                        if (dflash_model != null and tap_slot < dflash_taps.len / n_embd) {
                            const want = blk_tl: {
                                if (dflash_model) |*df| {
                                    for (df.target_layers) |tl| {
                                        if (tl == @as(i32, @intCast(li))) break :blk_tl true;
                                    }
                                }
                                break :blk_tl false;
                            };
                            if (want) {
                                @memcpy(dflash_taps[tap_slot * n_embd ..][0..n_embd], cur_h.data[0..n_embd]);
                                tap_slot += 1;
                            }
                        }
                        if (streamer) |*s| try s.ensureLayerLoaded(li);
                        // 9.4.1 (lane-b) diagnóstico decode híbrido CPU 5×:
                        // timing por capa gated DEBUG_LEVEL=2 — localizar la
                        // capa/etapa que domina el decode (hallazgo lane-d
                        // U1 + LFM2-350M 0.02 t/s GPU del smoke 17:0x).
                        const d_t0 = if (debugz.dbg.at(.detail)) perf_t.read() else 0;
                        try layer.forward(cur_h.*, nxt_h, current_pos, 1, null);
                        if (debugz.dbg.at(.detail)) {
                            const d_ns = perf_t.read() - d_t0;
                            debugz.dbg.printLevel(.detail, "[decode-cpu] tok={d} L{d} {s} {d}us\n", .{ current_pos, li, if (layer.is_attention) "attn" else "ssm", @as(u64, @intCast(d_ns)) / 1000 });
                        }
                        if (streamer) |*s| try s.prefetchNext(li);
                        const t2 = cur_h;
                        cur_h = nxt_h;
                        nxt_h = t2;
                    }
                    cpuRmsNormFlat(cur_h.data[0..n_embd], out_norm.data, rms_eps, cur_h.data[0..n_embd]);
                    cpuLmHeadLogits(cur_h.data[0..n_embd], lm_head.data, logits_f32);
                    host_logits_ready = true;
                    t_enqueue_ns += perf_t.read() - t_enq0;
                } else {
                    if (perf_stage) try cudaz.cuEventRecord(ev[0], lk.stream);
                    try cudaz.cuMemcpyHtoDAsync(g_cur.ptr(), @intFromPtr(embed_staging.ptr), n_embd * @sizeOf(f32), lk.stream);
                    if (debugz.dbg.dump_prefill_layers) {
                        try cudaz.cuStreamSynchronize(lk.stream);
                        const chk = try allocator.alloc(f32, n_embd);
                        defer allocator.free(chk);
                        try cudaz.cuMemcpyDtoH(@intFromPtr(chk.ptr), g_cur.ptr(), n_embd * @sizeOf(f32));
                        debugz.dbg.print("[pipeline] EMBED_IN tok={d} sum|v|={d:.6} max={d:.6} f0={d:.5} f1={d:.5} f2={d:.5}\n", .{ current_pos, debugz.sumAbsF32(chk), debugz.maxAbsF32(chk), chk[0], chk[1], chk[2] });
                    }
                    var cur2gpu = g_cur;
                    var nxt2gpu = g_nxt;
                    for (layers, 0..) |*layer, li| {
                        if (streamer) |*s| try s.ensureLayerLoaded(li);
                        try hybrid_layer.HybridLayer.forwardGPU(layer, &lk, cur2gpu, &nxt2gpu, current_pos, 1, null);
                        if (streamer) |*s| try s.prefetchNext(li);
                        if (perf_stage) try cudaz.cuEventRecord(ev[li + 1], lk.stream);
                        const t2 = cur2gpu;
                        cur2gpu = nxt2gpu;
                        nxt2gpu = t2;
                        if (debugz.dbg.dump_prefill_layers) {
                            try cudaz.cuStreamSynchronize(lk.stream);
                            const chk = try allocator.alloc(f32, n_embd);
                            defer allocator.free(chk);
                            try cudaz.cuMemcpyDtoH(@intFromPtr(chk.ptr), cur2gpu.ptr(), n_embd * @sizeOf(f32));
                            debugz.dbg.print("[pipeline] DECODE_LAYER tok={d} li={d} sum|v|={d:.6} max={d:.6} f0={d:.5} f1={d:.5} f2={d:.5}\n", .{ current_pos, li, debugz.sumAbsF32(chk), debugz.maxAbsF32(chk), chk[0], chk[1], chk[2] });
                            if (li == 0) {
                                debugz.dbg.print("[pipeline] DECODE_INPUT tok={d} f0={d:.5} f1={d:.5} f2={d:.5}\n", .{ current_pos, chk[0], chk[1], chk[2] });
                            }
                        }
                    }
                    // Norma final en GPU, lm_head device→device (peso cacheado en GPU).
                    try lk.rmsNorm(cur2gpu.ptr(), @intFromPtr(g_out_norm.dev_ptr), g_normed.ptr(), 1, n_embd, rms_eps);
                    // lm_head M=1: Q4_0/Q6_K device→device si el peso está cuantizado.
                    if (lmq80 != null) {
                        const nb = try allocator.alloc(f32, n_embd);
                        defer allocator.free(nb);
                        try cudaz.cuStreamSynchronize(lk.stream);
                        try cudaz.cuMemcpyDtoH(@intFromPtr(nb.ptr), g_normed.ptr(), n_embd * @sizeOf(f32));
                        try lmq80.?.project(nb, g_logits.ptr());
                        host_logits_ready = true;
                    } else if (lm_head_cpu_fb) {
                        const nb = try allocator.alloc(f32, n_embd);
                        defer allocator.free(nb);
                        try cudaz.cuStreamSynchronize(lk.stream);
                        try cudaz.cuMemcpyDtoH(@intFromPtr(nb.ptr), g_normed.ptr(), n_embd * @sizeOf(f32));
                        cpuRmsNormFlat(nb, out_norm.data, rms_eps, nb);
                        cpuLmHeadLogits(nb, lm_head.data, logits_f32);
                        host_logits_ready = true;
                    } else if (lm_head_q4 or lmq40 != null) {
                        const lmq40_b: []const u8 = if (lmq40) |*q| q.bytes else lm_head_q.bytes;
                        try lk.q4gemmLinear(allocator, g_normed.ptr(), lmq40_b, g_logits.ptr(), n_embd, vocab);
                    } else if (lm_head_q6k) {
                        try lk.qgemmLinear(allocator, g_normed.ptr(), lm_head_q.bytes, g_logits.ptr(), 1, n_embd, vocab, 3);
                    } else {
                        try engine.linearProjectionDeviceF16(g_normed, lm_head, &g_logits, 1, n_embd, vocab);
                    }
                    // G2 (TODO 1.7): camino NO-grafo — el argmax se lanza aquí de
                    // forma explícita (en el camino de grafo va como último nodo
                    // capturado). Sólo si los logits quedaron en device.
                    // 1.15: con temp>0 el último nodo es el Gumbel (mismo
                    // buffer de salida [1]i32, mismo D2H 4B).
                    if (gpu_argmax and !host_logits_ready) {
                        try lk.argmaxF32(g_logits.ptr(), gpu_argmax_out, 1, vocab);
                    } else if (gpu_gumbel and !host_logits_ready) {
                        try lk.sampleF32Gumbel(g_logits.ptr(), gpu_argmax_out, params.sampler.temperature, params.sampler.repetition_penalty, 1, vocab);
                    }
                    if (perf_stage) try cudaz.cuEventRecord(ev[layers.len + 1], lk.stream);
                    // Ancla para la ronda draft: hidden pre-norma de ESTE forward.
                    // Incondicional: si la cola se vació por RECHAZO en la sección
                    // de muestreo (más abajo), esta misma iteración debe proveer el
                    // ancla correcto — con la captura condicionada a pending==0 el
                    // rechazo reutilizaba el ancla viejo y los drafts degeneraban.
                    if (spec_active and h_prev.len == n_embd) {
                        try cudaz.cuMemcpyDtoH(@intFromPtr(h_prev.ptr), cur2gpu.ptr(), n_embd * @sizeOf(f32));
                    }
                }
            }
            if (!decode_cpu or !host_logits_ready) t_enqueue_ns += perf_t.read() - t_enq0;

            if (debugz.dbg.dump_norm) {
                try cudaz.cuStreamSynchronize(lk.stream);
                const nb = try allocator.alloc(f32, n_embd);
                defer allocator.free(nb);
                try cudaz.cuMemcpyDtoH(@intFromPtr(nb.ptr), g_normed.ptr(), n_embd * @sizeOf(f32));
                var s: f64 = 0;
                var mx: f32 = 0;
                for (nb) |v| {
                    s += @abs(@as(f64, v));
                    if (@abs(v) > mx) mx = @abs(v);
                }
                debugz.dbg.print("[pipeline] DUMPNORM pos={d} sum|v|={d:.6} max={d:.6} first3={d:.4},{d:.4},{d:.4}\n", .{ current_pos, s, mx, nb[0], nb[1], nb[2] });
            }

            // D2H async (stream-ordered tras los kernels/grafo que los escribieron)
            // de los bloques KV del token; mantiene el pool host autoritativo.
            // (CPU: el KV ya vive en host — nada que sincronizar.)
            if (!decode_cpu) {
                for (layers) |*layer| {
                    if (layer.is_attention) try layer.attn_layer.?.syncDecodeBlocks(current_pos, 1);
                }
            }

            try cudaz.cuStreamSynchronize(lk.stream);
            if (debugz.dbg.dump_prefill_layers) {
                for (layers, 0..) |layer, li| {
                    if (!layer.is_attention) {
                        // 8.3: ShortConv (LFM2) — dump del g_conv_state vivo.
                        if (layer.ssm_layer) |l| {
                            if (l.gpu) |gpu| {
                                const nf32 = gpu.d_s_state.len;
                                const buf = try allocator.alloc(f32, nf32);
                                defer allocator.free(buf);
                                try cudaz.cuMemcpyDtoH(@intFromPtr(buf.ptr), @intFromPtr(gpu.d_s_state.dev_ptr), nf32 * @sizeOf(f32));
                                debugz.dbg.print("[pipeline] SSTATE tok={d} li={d} sum|v|={d:.6} max={d:.6}\n", .{ current_pos, li, debugz.sumAbsF32(buf), debugz.maxAbsF32(buf) });
                                if (gpu.d_conv_state.len > 0) {
                                    const nf2 = gpu.d_conv_state.len;
                                    const buf2 = try allocator.alloc(f32, nf2);
                                    defer allocator.free(buf2);
                                    try cudaz.cuMemcpyDtoH(@intFromPtr(buf2.ptr), @intFromPtr(gpu.d_conv_state.dev_ptr), nf2 * @sizeOf(f32));
                                    debugz.dbg.print("[pipeline] CONVSTATE tok={d} li={d} sum|v|={d:.6} max={d:.6}\n", .{ current_pos, li, debugz.sumAbsF32(buf2), debugz.maxAbsF32(buf2) });
                                }
                            }
                        } else if (layer.short_conv_layer) |*sc| {
                            if (sc.gpu) |sgpu| {
                                const nf = sgpu.g_conv_state.len;
                                const buf = try allocator.alloc(f32, nf);
                                defer allocator.free(buf);
                                try cudaz.cuMemcpyDtoH(@intFromPtr(buf.ptr), @intFromPtr(sgpu.g_conv_state.dev_ptr), nf * @sizeOf(f32));
                                debugz.dbg.print("[pipeline] SCSTATE tok={d} li={d} sum|v|={d:.6} max={d:.6}\n", .{ current_pos, li, debugz.sumAbsF32(buf), debugz.maxAbsF32(buf) });
                            }
                        }
                    }
                }
            }
            if (perf_stage) {
                // Los eventos de la ronda aún pueden no estar signaled (el
                // stream sigue drenando) — cuEventElapsedTime devuelve
                // ERROR_NOT_READY y el try lo convertía en CudaError fatal
                // (repro: NOGRAPH=1 PERF_STAGE=1 prompt 1-tok). Sync dura
                // antes de leer tiempos: el decode ya terminó su work.
                try cudaz.cuStreamSynchronize(lk.stream);
                var ms: f32 = 0;
                for (layers, 0..) |_, li| {
                    try cudaz.cuEventElapsedTime(&ms, ev[li], ev[li + 1]);
                    const ns = @as(i128, @intFromFloat(@as(f64, ms) * std.time.ns_per_ms));
                    layer_gpu_ns[li] += ns;
                }
                try cudaz.cuEventElapsedTime(&ms, ev[layers.len], ev[layers.len + 1]);
                gpu_head_ns += @as(i128, @intFromFloat(@as(f64, ms) * std.time.ns_per_ms));
                try cudaz.cuEventElapsedTime(&ms, ev[0], ev[layers.len + 1]);
                gpu_total_ns += @as(i128, @intFromFloat(@as(f64, ms) * std.time.ns_per_ms));
            }
            const t_d2h0 = perf_t.read();
            // G2 (TODO 1.7): con el argmax en device el D2H del token son 4
            // bytes (el id del token) en vez de vocab·4B (~993 KB). `next_tok_i32`
            // se rellena aquí y el sampler se salta más abajo.
            // 1.15: idéntico para el Gumbel — el graph/launch terminó en el
            // token_id en ambos modos; `gpu_sample` cubre argmax+gumbel.
            var gpu_argmax_tok: ?u32 = null;
            if (gpu_sample and !host_logits_ready) {
                var raw: i32 = 0;
                try cudaz.cuMemcpyDtoH(@intFromPtr(&raw), gpu_argmax_out, @sizeOf(i32));
                gpu_argmax_tok = @as(u32, @bitCast(raw));
                // 1.15: rep_penalty — el ring device se refresca ANTES del
                // siguiente replay (los tokens generados crecen por paso).
                // 1.15: rep_penalty — refrescar el ring device por paso
                // (el graph capturó el PUNTERO; contenido por-paso). [0]=n
                // en device ⇒ el kernel ve el tamaño actual sin re-capture.
                if (gpu_gumbel and params.sampler.repetition_penalty != 1.0) {
                    try lk.sampleGumbelSetRing(gen_tokens.items);
                }
            } else if (!host_logits_ready) {
                try cudaz.cuMemcpyDtoH(@intFromPtr(logits_f32.ptr), g_logits.ptr(), vocab * @sizeOf(f32));
            }
            host_logits_ready = false;

            if (debugz.dbg.dump_kv) {
                for (layers, 0..) |*layer, li| {
                    if (!layer.is_attention) continue;
                    const bt_opt = layer_block_tables[li] orelse continue;
                    const phys = bt_opt.getPhysical(0) orelse continue;
                    const ba = paged_kv.block_alloc;
                    if (debugz.dbg.dump_kv) {
                        for (layer_block_tables, 0..) |bt_opt2, li2| {
                            if (bt_opt2) |bt2| {
                                const hb2: ?usize = bt2.getPhysical(0);
                                debugz.dbg.print("[kv_quant] DUMPKV pos={d} LAYERTAB li={d} bt[0]={?d} blocks={d}\n", .{ current_pos, li2, hb2, bt2.numBlocks() });
                            }
                        }
                    }
                    const blk = ba.memory_pool[phys * ba.block_bytes ..][0..ba.block_bytes];
                    const hb = std.mem.bytesAsSlice(u16, blk);
                    const nkv = ba.num_kv_heads;
                    const hd = ba.head_dim;
                    const stride = nkv * hd;
                    var all_h: [16]f64 = [_]f64{0} ** 16;
                    for (0..@min(16, ba.block_size)) |tpos| {
                        const base = tpos * stride;
                        var sp: f64 = 0;
                        for (0..stride) |i| sp += @abs(@as(f64, @as(f32, @floatFromInt(hb[base + i]))));
                        all_h[tpos] = sp;
                    }
                    debugz.dbg.print("[kv_quant] DUMPKV pos={d} layer={d} HOST phys={d} all=", .{ current_pos, li, phys });
                    for (all_h) |sp| debugz.dbg.print("{d:.1},", .{sp});
                    debugz.dbg.print("\n", .{});
                    if (layer.attn_layer.?.paged_gpu) |gpu| {
                        const st0: c_int = if (li < gpu.bt_stagings.items.len and gpu.bt_stagings.items[li].len > 0) gpu.bt_stagings.items[li][0] else -999;
                        const hbt0: ?usize = layer.attn_layer.?.block_table.getPhysical(0);
                        var dbt0: c_int = 0;
                        try cudaz.cuMemcpyDtoH(@intFromPtr(&dbt0), gpu.getDbt(li), @sizeOf(c_int));
                        const stlen: usize = if (li < gpu.bt_stagings.items.len) gpu.bt_stagings.items[li].len else 0;
                        debugz.dbg.print("[kv_quant] DUMPKV pos={d} layer={d} host_bt[0]={?d} staging[0]={d} staging_len={d} d_bt[0]={d}\n", .{ current_pos, li, hbt0, st0, stlen, dbt0 });
                        const dv = try gpu.cacheBase(paged_kv.block_alloc);
                        const dbuf = try allocator.alloc(u16, ba.block_bytes / 2);
                        defer allocator.free(dbuf);
                        try cudaz.cuMemcpyDtoH(@intFromPtr(dbuf.ptr), dv + phys * ba.block_bytes, ba.block_bytes);
                        var all_d: [16]f64 = [_]f64{0} ** 16;
                        for (0..@min(16, ba.block_size)) |tpos| {
                            const base = tpos * stride;
                            var sp: f64 = 0;
                            for (0..stride) |i| sp += @abs(@as(f64, @as(f32, @floatFromInt(dbuf[base + i]))));
                            all_d[tpos] = sp;
                        }
                        debugz.dbg.print("[kv_quant] DUMPKV pos={d} layer={d} DEV  phys={d} all=", .{ current_pos, li, phys });
                        for (all_d) |sp| debugz.dbg.print("{d:.1},", .{sp});
                        debugz.dbg.print("\n", .{});
                        var dsp: c_int = 0;
                        try cudaz.cuMemcpyDtoH(@intFromPtr(&dsp), gpu.getDStartPos(), @sizeOf(c_int));
                        var dsq: c_int = 0;
                        try cudaz.cuMemcpyDtoH(@intFromPtr(&dsq), gpu.getDSeqLens(), @sizeOf(c_int));
                        var dbt: c_int = 0;
                        try cudaz.cuMemcpyDtoH(@intFromPtr(&dbt), gpu.getDbt(li), @sizeOf(c_int));
                        debugz.dbg.print("[kv_quant] DUMPKV pos={d} layer={d} d_start_pos={d} d_seq_len={d} d_bt[0]={d}\n", .{ current_pos, li, dsp, dsq, dbt });
                    }
                    break;
                }
            }
            if (debugz.dbg.dump_logits) {
                for (logits_f32, 0..) |v, i| debugz.dbg.print("[pipeline] LG {d} {d}\n", .{ i, v });
            }
            const t_samp0 = perf_t.read();
            // (C4.3: la especulación batched vive en su propia rama más abajo;
            // este bucle queda para el camino sin especulación.)
            const next_token = if (gpu_argmax_tok) |t| t else params.sampler.sample(logits_f32, &rng, gen_tokens.items);
            try gen_tokens.append(allocator, next_token);
            try scheduler.appendToken(seq_id, next_token);
            // Server F2 T2c: emisión token-por-token.
            if (sink) |s| {
                const piece = tok.decode(&[_]u32{next_token}, allocator) catch "";
                defer if (piece.len > 0) allocator.free(piece);
                s.emit(next_token, piece) catch |sig| switch (sig) {
                    error.StopSequenceHit => break,
                };
            }
            current_pos += 1;
            t_d2h_ns += t_samp0 - t_d2h0;
            t_sample_ns += perf_t.read() - t_samp0;

            // 9.6: LoopGuard — feed y check en path no-spec.
            if (params.spec_loop_guard_mode != .off) {
                loop_guard_ns.feed(next_token);
                const det = loop_guard_ns.check();
                if (det.triggered) {
                    if (params.spec_loop_guard_mode == .force_close) {
                        debugz.dbg.printLevel(.info, "[loop-guard] trigger period={d} score={d:.3} — force-close\n", .{ det.period, det.score });
                        break;
                    } else {
                        try stdout.print("[!] loop detectado: periodo {d}, score {d:.3} — continuando (warn mode)\n", .{ det.period, det.score });
                    }
                }
            }

            if (gt.eos_id != null and next_token == gt.eos_id.?) break;
        }
    }

    // Finish sequence: release blocks
    scheduler.finishSequence(seq_id);

    // Server F2 T2c: señal de finish al sink con usage real.
    if (sink) |s| {
        const gen_ms_u: u64 = @intCast(@divTrunc(t_gen.read(), std.time.ns_per_ms));
        const reason: SinkFinishReason = blk: {
            const hit_max = gen_tokens.items.len >= params.max_new_tokens + 1;
            break :blk if (hit_max) .length else .stop;
        };
        s.finish(reason, .{
            .prompt_tokens = seq_len,
            .completion_tokens = gen_tokens.items.len,
            .total_ms = gen_ms_u,
        });
    }

    if (spec_active) {
        try stdout.print("[spec] aceptación: ", .{});
        try spec_driver.reportMetrics(stdout);
    }

    const gen_ns = t_gen.read();
    const gen_ms = @as(f64, @floatFromInt(@divTrunc(gen_ns, std.time.ns_per_ms)));
    const prefill_ms = @as(f64, @floatFromInt(@divTrunc(prefill_ns, std.time.ns_per_ms)));
    const num_gen = if (gen_tokens.items.len > 0) gen_tokens.items.len - 1 else 0;
    const tok_s: f64 = if (gen_ms > 0) @as(f64, @floatFromInt(num_gen)) / (gen_ms / 1000.0) else 0;

    try stdout.print("✓ Listo! Generados {d} tokens en {d:.1}s ({d:.1} tok/s)\n\n", .{
        gen_tokens.items.len,
        @divTrunc(gen_ns, @as(i128, std.time.ns_per_s)),
        tok_s,
    });

    try stdout.print("\n[+] Generación ({d} tokens):\n", .{gen_tokens.items.len});
    const decoded = try tok.decode(gen_tokens.items, allocator);
    defer allocator.free(decoded);
    try stdout.print("{s}\n", .{decoded});
    try stdout.print("\n[+] Métricas detalladas:\n", .{});
    try stdout.print("  prefill   {d:.1} ms ({d} tok, {d:.1} ms/tok)\n", .{ prefill_ms, seq_len, if (seq_len > 0) prefill_ms / @as(f64, @floatFromInt(seq_len)) else 0 });
    try stdout.print("  decode    {d:.1} ms ({d} tok, {d:.2} ms/tok)\n", .{ gen_ms, gen_tokens.items.len, if (gen_tokens.items.len > 0) gen_ms / @as(f64, @floatFromInt(gen_tokens.items.len)) else 0 });
    try stdout.print("  throughput {d:.1} tok/s\n", .{tok_s});
    try stdout.print("  total     {d:.1} ms\n", .{prefill_ms + gen_ms});
    try stdout.print("\n=================================================\n", .{});
    try stdout.print("              Ejecucion completada               \n", .{});
    try stdout.flush();
    if (perf_stage) {
        const nt: f64 = @floatFromInt(@max(@as(usize, 1), gen_tokens.items.len));
        var ssm_ns: i128 = 0;
        var attn_ns: i128 = 0;
        for (layers, 0..) |layer, li| {
            if (layer.is_attention) attn_ns += layer_gpu_ns[li] else ssm_ns += layer_gpu_ns[li];
        }
        const us = std.time.ns_per_us;
        try stdout.print("[+] PERF (avg/token):\n", .{});
        try stdout.print("  host  blocks {d:.1} us  embed {d:.1} us  enqueue {d:.1} us  d2h {d:.1} us  sample {d:.1} us\n", .{
            @as(f64, @floatFromInt(t_blocks_ns)) / nt / @as(f64, @floatFromInt(us)),
            @as(f64, @floatFromInt(t_embed_ns)) / nt / @as(f64, @floatFromInt(us)),
            @as(f64, @floatFromInt(t_enqueue_ns)) / nt / @as(f64, @floatFromInt(us)),
            @as(f64, @floatFromInt(t_d2h_ns)) / nt / @as(f64, @floatFromInt(us)),
            @as(f64, @floatFromInt(t_sample_ns)) / nt / @as(f64, @floatFromInt(us)),
        });
        try stdout.print("  gpu   ssm {d:.1} us  attn {d:.1} us  head {d:.1} us  total {d:.1} us\n", .{
            @as(f64, @floatFromInt(ssm_ns)) / nt / @as(f64, @floatFromInt(us)),
            @as(f64, @floatFromInt(attn_ns)) / nt / @as(f64, @floatFromInt(us)),
            @as(f64, @floatFromInt(gpu_head_ns)) / nt / @as(f64, @floatFromInt(us)),
            @as(f64, @floatFromInt(gpu_total_ns)) / nt / @as(f64, @floatFromInt(us)),
        });
        // PERF_SSM (lane-f F1): desglose fino por-etapa de las capas SSM
        // (requiere NOGRAPH — con replay los eventos por-launch no son fiables).
        transformer.ssmStageReport();
        var top: [5]usize = undefined;
        for (0..5) |k| top[k] = k;
        for (layers, 0..) |_, li| {
            if (layer_gpu_ns[li] <= layer_gpu_ns[top[4]]) {
                top[4] = li;
                for (0..3) |k| {
                    if (layer_gpu_ns[top[k + 1]] > layer_gpu_ns[top[k]]) {
                        const tmp = top[k];
                        top[k] = top[k + 1];
                        top[k + 1] = tmp;
                    }
                }
            }
        }
        try stdout.print("  top layers:\n", .{});
        for (top) |li| {
            try stdout.print("    L{d:<2} {s} {d:.1} us\n", .{
                li,                                                                           if (layers[li].is_attention) "attn" else "ssm",
                @as(f64, @floatFromInt(layer_gpu_ns[li])) / nt / @as(f64, @floatFromInt(us)),
            });
        }
    }

    // Cleanup global Q4 weight cache (GPU allocations)
    layer_kernels.deinitQ4Cache();
}

pub fn gpuKvPipelineReady(f: QuantFormat) bool {
    // 7.5-repro (lane-f): KVFORCE=1 habilita los 4 formatos iq2/iq3_xxs en
    // cuarentena para repro del pool-degeneracy (prefill 22-75s). Sin el
    // env el gate se mantiene conservador (producción nunca decode-unsafe).
    // 6.3 (lane-f): tq1_0 entra en cuarentena E2E con datos — encoder/
    // append/decode bit-exact por tests (max_diff=0) y append GPU
    // implementado (kvAppendTQ1_0Kernel), PERO greedy 0.8B diverge
    // ("The capital of France is" → "located in the region of:" vs
    // fp16 " Paris.") — pérdida intrínseca del 1.7-bit, mismo caso que
    // iq1_s (3.1). KVFORCE=1 habilita para repro/eval.
    const force_iq2 = std.c.getenv("KVFORCE") != null;
    return switch (f) {
        .fp16, .q8_0, .q4_0, .q4_k, .q8_k, .iq4_xs, .q2_k, .q3_k, .q5_k, .q6_k, .iq1_s, .iq1_m, .iq3_s, .iq4_nl,
        .iq2_s, .iq2_xs, .iq2_xxs, .iq3_xxs => true,
        .tq1_0 => force_iq2,
        else => false,
    };
}

/// 4.5-a (lane-e wiring Contrato 8): mapeo GgmlType → Format del GEMV CPU.
/// Espejo de moe_bench.ggmlToCpuFormat (la copia privada de moe_layer no es
/// visible desde aquí). Null ⇒ dtype sin soporte CPU (offload puro).
fn ggmlToCpuFormatCli(t: gguf.GgmlType) ?moe_cpu_gemv.Format {
    return switch (t) {
        .q4_0 => .q4_0,
        .q4_1 => .q4_1,
        .q5_k => .q5_k,
        .q6_k => .q6_k,
        .q8_0 => .q8_0,
        .q4_k => .q4_k,
        .q3_k => .q3_k,
        .q2_k => .q2_k,
        .iq3_s => .iq3_s,
        .iq2_s => .iq2_s,
        .iq4_nl => .iq4_nl,
        .iq2_xxs => .iq2_xxs,
        else => null,
    };
}

// ═════════════════════════════════════════════════════════════════════════════
// P0-1 (dev RLT): PPL híbrido GPU — `--ppl` para Qwen3.5/LFM2.5 vía forwardGPU
// batcheado. Desbloquea G6 (PPL golden Qwen3.5-0.8B wiki12k).
//
// Arquitectura (espejo del loop legacy B1-B4 corregidos de main.zig runPpl):
//  - sliding window llama.cpp: window=min(2048,ctx), stride=window/2
//  - cada chunk = KV fresh (BTs de layers truncate(0)+appendTokens) + reset
//    del estado recurrente device (resetStateGpu: SSM s/conv state a zeros)
//    — el estado se reconstruye desde ctx_start, semántica llama.cpp
//  - prefill 100% GPU en sub-chunks de ubatch (embeddings H2D + forwardGPU)
//  - lm_head SOLO sobre las filas scored del chunk (M=scored ≤ window/2):
//    rmsNorm(rows) device → linearProjectionDeviceF16 batcheado (el peso W_T
//    f32 se cachea en device UNA vez vía weight_cache del engine)
//  - scoring NLL idéntico al legacy (invariantes B1-B4)
// ═════════════════════════════════════════════════════════════════════════════

pub fn runHybridPpl(
    io: std.Io,
    allocator: std.mem.Allocator,
    model: *gguf_model.GgufModel,
    params: CliParams,
    backend: matmul.Backend,
    stdout: anytype,
) !void {
    const cfg = model.config;
    const n_embd = cfg.embedding_length;
    const vocab = cfg.vocab_size;
    if (backend != .cublas) {
        try stdout.print("[!] --ppl híbrido requiere backend cublas (prefill GPU batcheado)\n", .{});
        return error.GpuRequired;
    }
    try cudaz.ensureCurrent();
    var engine = try matmul.MatmulEngine.init(allocator, backend, .f32);
    defer engine.deinit();

    const ppl_path = params.ppl_file orelse return error.NoPplFile;
    var max_seq_len = if (params.context_length == 0) cfg.context_length else params.context_length;

    // Embedding (cuant-residente opt-in igual que inference) + lm_head + norm
    var emb_f16: ?Tensor(f16) = null;
    defer if (emb_f16) |*t| t.deinit();
    const emb_quant = if (std.c.getenv("ZIG_AI_EMB_F16") == null)
        try model.loadEmbeddingQuant()
    else blk: {
        emb_f16 = try model.loadEmbedding();
        break :blk null;
    };
    // lm_head: camino cuant-residente cuando el GGUF trae output.weight
    // cuantizado (p.ej. q4_0 nativo en qat) — consume bytes RAW vía
    // qgemmLinear batched (m=lm_tile, ~143MB device en 4B) en vez de
    // dequantizar a f16 host + cache f32 device (2.54GB en 4B — OOM en
    // 8GB). Idem inference path. Fallback f16 clásico para dtypes sin
    // kernel (bf16/f16/f32).
    var lm_head_f16: ?Tensor(f16) = null;
    defer if (lm_head_f16) |*t| t.deinit();
    // QuantWeight es una VIEW sobre el mmap del GGUF — sin deinit.
    var lm_head_q: ?QuantWeight = null;
    var lm_head_qtype: u32 = std.math.maxInt(u32);
    blk: {
        const lq = model.loadLmHeadQuant() catch break :blk;
        const qt: ?u32 = switch (lq.dtype()) {
            .q4_0 => 0,
            .q4_1 => 1,
            .q5_k => 2,
            .q6_k => 3,
            .q4_k => 4,
            .q8_0 => 5,
            .q3_k => 6,
            .q2_k => 7,
            else => null,
        };
        if (qt != null) {
            lm_head_q = lq;
            lm_head_qtype = qt.?;
        }
    }
    const lm_head_bytes: ?[]const u8 = if (lm_head_q) |*q| q.bytes else null;
    var out_norm = try model.loadOutputNorm();
    defer out_norm.deinit();

    // MTP: excluir capas nextn del forward target (idem inference)
    const mtp_info = specdrv.detectMtp(&model.file, cfg.block_count);
    const eff_blocks = if (mtp_info) |m| m.layer_idx else cfg.block_count;

    const head_dim = if (cfg.head_dim > 0) cfg.head_dim else n_embd / cfg.head_count;
    const block_size: usize = 16;
    var num_attn_layers: usize = 0;
    for (0..eff_blocks) |i| {
        if (cfg.isFullAttentionLayer(i)) num_attn_layers += 1;
    }

    // Auto-presupuesto VRAM con cuantización KV efectiva (no hardcode fp16).
    var total_vram: usize = 0;
    total_vram = cudaz.getDeviceTotalMem(cudaz.cuDeviceGet(0) catch 0) catch 0;
    const effective_cache_type_k: QuantFormat =
        if (gpuKvPipelineReady(params.cache_type_k)) params.cache_type_k else .fp16;
    const effective_cache_type_v: QuantFormat =
        if (gpuKvPipelineReady(params.cache_type_v)) params.cache_type_v else .fp16;
    if (total_vram > 0) {
        const wpl = estimateCompressedWeightPerLayer(allocator, &model.file, cfg) catch 50 * 1024 * 1024;
        var ctx = max_seq_len;
        while (true) {
            const breakdown = vram_budget.estimateTotalVram(buildVramEstimate(cfg, wpl, ctx, effective_cache_type_k, block_size, params.layer_stream_max, 0, total_vram));
            if (breakdown.total_vram <= total_vram * 85 / 100 or ctx <= 1024) break;
            ctx = @max(@as(usize, 1024), ctx * 3 / 4);
        }
        if (ctx != max_seq_len) {
            try stdout.print("[+] Contexto auto-reducido: {d} → {d} tokens (presupuesto VRAM)\n", .{ max_seq_len, ctx });
        }
        max_seq_len = ctx;
    }

    // PPL sliding-window: contexto efectivo = window_max (no max_seq_len completo)
    const window_max: usize = @min(@as(usize, 2048), max_seq_len);
    const ppl_ctx: usize = window_max;
    if (effective_cache_type_k != params.cache_type_k or effective_cache_type_v != params.cache_type_v) {
        try stdout.print("[!] PPL KV -ctk {s} / -ctv {s}: decode GPU no validado → usando fp16 (KVFORCE=1 para forzar)\n", .{
            @tagName(params.cache_type_k), @tagName(params.cache_type_v),
        });
        try stdout.flush();
    }
    const blocks_per_seq: usize = (ppl_ctx + block_size - 1) / block_size;
    const kv_dim_est = cfg.head_count_kv * head_dim;
    const block_bytes_est = block_size * kv_dim_est * 2 * 2;
    const kv_pool_cap_bytes = if (total_vram > 0)
        @max(512 * 1024 * 1024, total_vram * 30 / 100)
    else
        512 * 1024 * 1024;
    const kv_pool_cap_blocks = @max(64, kv_pool_cap_bytes / @max(1, block_bytes_est));
    const num_blocks_raw = @max(64, num_attn_layers * blocks_per_seq);
    const num_blocks = @min(num_blocks_raw, kv_pool_cap_blocks);
    var paged_kv = try paged_attn.PagedKVCache.init(allocator, .{
        .block_size = block_size,
        .num_blocks = num_blocks,
        .head_dim = head_dim,
        .num_kv_heads = cfg.head_count_kv,
        .num_q_heads = cfg.head_count,
        .dtype = .f16,
        .quant_k = effective_cache_type_k,
        .quant_v = effective_cache_type_v,
        .enable_prefix_cache = false,
        .enable_cpu_offload = false,
        .max_seq_len = ppl_ctx,
    });
    defer paged_kv.deinit();
    debugz.dbg.printLevel(.info, "[ppl] kv pool: blocks={d} quant_k={s} quant_v={s} ppl_ctx={d}\n", .{ num_blocks, @tagName(effective_cache_type_k), @tagName(effective_cache_type_v), ppl_ctx });

    // Pool GPU paged para las capas attn (forwardGPU de attn lo REQUIERE:
    // prefillDevice/decodeDevice — sin él KvCacheNotSet). num_blocks=0: usa
    // el block_alloc del pool KV compartido.
    var shared_paged_gpu = paged_attn.PagedAttentionGpu.init(
        allocator,
        .{
            .block_size = block_size,
            .num_blocks = 0,
            .head_dim = head_dim,
            .num_kv_heads = cfg.head_count_kv,
            .num_q_heads = cfg.head_count,
            .dtype = .f16,
            .quant_k = effective_cache_type_k,
            .quant_v = effective_cache_type_v,
        },
        @ptrCast((try matmul.MatmulEngine.sharedCudaStream()).raw),
    ) catch null;
    defer if (shared_paged_gpu) |*g| g.deinit();
    const shared_gpu_ptr: ?*paged_attn.PagedAttentionGpu = if (shared_paged_gpu) |*g| g else null;

    // Capas híbridas
    var layers = try allocator.alloc(hybrid_layer.HybridLayer, eff_blocks);
    defer allocator.free(layers);
    defer for (layers) |*l| l.deinit();
    var layer_block_tables = try allocator.alloc(?*paged_attn.BlockTable, eff_blocks);
    defer allocator.free(layer_block_tables);
    @memset(layer_block_tables, null);
    defer for (layer_block_tables) |bt_opt| {
        if (bt_opt) |bt| {
            bt.deinit(paged_kv.block_alloc);
            allocator.destroy(bt);
        }
    };
    for (0..eff_blocks) |i| {
        if (cfg.isFullAttentionLayer(i)) {
            const bt = try allocator.create(paged_attn.BlockTable);
            bt.* = paged_attn.BlockTable.init(allocator, block_size);
            layer_block_tables[i] = bt;
        }
        layers[i] = try hybrid_layer.HybridLayer.init(
            allocator,
            i,
            hybrid_layer.HybridLayerParams.fromModelConfig(cfg, max_seq_len),
            cfg.isFullAttentionLayer(i),
            backend,
            &paged_kv,
            if (layer_block_tables[i]) |bt| bt else null,
            shared_gpu_ptr,
        );
        try layers[i].loadWeightsFromGguf(&model.file, null);
    }
    try stdout.print("[+] ppl híbrido: arch={s} capas={d} (attn {d}) emb={d} vocab={d} ctx={d} blocks={d}\n", .{ cfg.architecture, eff_blocks, num_attn_layers, n_embd, vocab, max_seq_len, num_blocks });
    try stdout.flush();

    // Tokenizar corpus
    var gt = try gguf_tokenizer.GgufTokenizer.fromGguf(allocator, &model.file);
    defer gt.deinit();
    var tok = try bpe.BPETokenizer.fromTokenizer(allocator, &gt);
    defer tok.deinit();
    var ppl_file = try std.Io.Dir.cwd().openFile(io, ppl_path, .{ .mode = .read_only });
    defer ppl_file.close(io);
    const fstat = try ppl_file.stat(io);
    const fdata = try allocator.alloc(u8, @intCast(fstat.size));
    defer allocator.free(fdata);
    _ = try ppl_file.readPositionalAll(io, fdata, 0);
    const token_ids = try tok.encode(fdata, .{});
    defer allocator.free(token_ids);
    try stdout.print("[+] ppl: {s} = {d} bytes, {d} tokens (kv {s}/{s})\n", .{ ppl_path, fdata.len, token_ids.len, @tagName(effective_cache_type_k), @tagName(effective_cache_type_v) });
    try stdout.flush();
    if (token_ids.len < 2) return error.NoTokensToScore;

    // Buffers GPU del prefill chunked
    const ub = params.ubatch_size;
    var lk = try layer_kernels.LayerKernels.init(@ptrCast((try matmul.MatmulEngine.sharedCudaStream()).raw));
    defer lk.deinit();
    var g_cur = try cublas.GpuTensor(f32).alloc(ub * n_embd);
    defer g_cur.deinit();
    var g_nxt = try cublas.GpuTensor(f32).alloc(ub * n_embd);
    defer g_nxt.deinit();
    // g_full: hidden COMPLETO del chunk [window × n_embd] — cada ubatch copia
    // su resultado DtoD aquí (cur2gpu rota y solo conserva el último ubatch;
    // el lm_head necesita filas de TODOS los ubatches).
    var g_full = try cublas.GpuTensor(f32).alloc(window_max * n_embd);
    defer g_full.deinit();
    // Tile del lm_head: los buffers full-window×vocab serían ~2GB (device+host).
    // Con tiles de lm_tile filas (~127MB c/u) el pico baja 16×.
    const lm_tile: usize = 128;
    var g_normed = try cublas.GpuTensor(f32).alloc(lm_tile * n_embd);
    defer g_normed.deinit();
    var g_logits = try cublas.GpuTensor(f32).alloc(lm_tile * vocab);
    defer g_logits.deinit();
    var g_out_norm = try cublas.GpuBuffer(f32).alloc(n_embd);
    defer g_out_norm.free();
    try g_out_norm.upload(out_norm.data);
    const stage = try allocator.alloc(f32, ub * n_embd);
    defer allocator.free(stage);
    const logit_rows = try allocator.alloc(f32, lm_tile * vocab);
    defer allocator.free(logit_rows);
    // R-4: buffer de logits target (T × vocab) para entrenamiento MSE
    var logits_target_host: ?[]f32 = null;
    defer if (logits_target_host) |lt| allocator.free(lt);
    if (params.dump_logits_target_path.len > 0) {
        logits_target_host = try allocator.alloc(f32, token_ids.len * vocab);
        @memset(logits_target_host.?, 0);
    }

    // R-3: modo recurrente (1 token a la vez, activa RLT feedback en forwardGPU).
    const recurrent_mode = params.recurrent_prefill;
    debugz.dbg.printLevel(.info, "[ppl] recurrent_mode={} token_ids.len={}\n", .{ recurrent_mode, token_ids.len });
    var hidden16_single: ?Tensor(f16) = null;
    // defer if (hidden16_single) |*t| t.deinit();  // TEMP: skip to debug comptime
    if (recurrent_mode) {
        hidden16_single = try Tensor(f16).alloc(allocator, &.{ 1, 1, n_embd });
    }
    var rlt_capture_buf: ?[]f32 = null;
    // defer replaced with explicit cleanup to avoid comptime eval
    if (recurrent_mode and params.capture_rlt_path.len > 0) {
        rlt_capture_buf = try allocator.alloc(f32, token_ids.len * n_embd);
        @memset(rlt_capture_buf.?, 0);
    }

    // ── Scoring loop (invariantes B1-B4 del legacy) ─────────────────────
    const window: usize = window_max;
    const stride: usize = @max(@as(usize, 1), window / 2);
    var nll_sum: f64 = 0;
    var n_scored: usize = 0;
    var pos: usize = 1; // chunk 0: el token 0 no tiene contexto — se salta
    var chunk_idx: usize = 0;
    const timer = @import("time").Timer.now();

    while (pos < token_ids.len) {
        debugz.dbg.printLevel(.info, "[ppl] chunk: pos={d} token_ids.len={d}\n", .{ pos, token_ids.len });
        const ctx_start = @min(pos - 1, pos -| (window - stride));
        const chunk_end = @min(ctx_start + window, token_ids.len);
        const chunk_len = chunk_end - ctx_start;
        const scored_in_chunk = chunk_end - pos;
        const first_row: usize = pos - ctx_start - 1;

        // Chunk nuevo: reset estado recurrente (SSM/conv device) + KV fresh.
        // Las BTs de las capas son la fuente de verdad del KV paged (el
        // scheduler del inference lleva contabilidad aparte; aquí 1 seq).
        for (layers) |*l| l.resetStateGpu();
        for (layer_block_tables) |bt_opt| {
            if (bt_opt) |bt| {
                if (bt.num_tokens > 0) try bt.truncate(paged_kv.block_alloc, 0);
                try bt.appendTokens(paged_kv.block_alloc, chunk_len);
            }
        }

        // Prefill GPU del chunk en ubatches
        var hidden16 = try Tensor(f16).alloc(allocator, &.{ 1, chunk_len, n_embd });
        defer hidden16.deinit();
        if (emb_quant) |*qw| {
            embedding.embeddingLookupQuant(qw, token_ids[ctx_start..chunk_end], 1, chunk_len, &hidden16);
        } else {
            embedding.embeddingLookup(emb_f16.?, token_ids[ctx_start..chunk_end], 1, chunk_len, &hidden16);
        }

        var cur2gpu = g_cur;
        var nxt2gpu = g_nxt;
        var p_off: usize = 0;
        while (p_off < chunk_len) {
            const n = @min(ub, chunk_len - p_off);
            for (0..n * n_embd) |i| {
                stage[i] = @as(f32, hidden16.data[p_off * n_embd + i]);
            }
            try cudaz.cuMemcpyHtoD(cur2gpu.ptr(), @intFromPtr(stage.ptr), n * n_embd * @sizeOf(f32));
            for (layers) |*layer| {
                try hybrid_layer.HybridLayer.forwardGPU(layer, &lk, cur2gpu, &nxt2gpu, p_off, n, null);
                const t2 = cur2gpu;
                cur2gpu = nxt2gpu;
                nxt2gpu = t2;
            }
            // Acumular el ubatch en g_full (cur2gpu rota — el lm_head necesita
            // filas de TODOS los ubatches del chunk)
            try cudaz.cuMemcpyDtoD(g_full.ptr() + p_off * n_embd * @sizeOf(f32), cur2gpu.ptr(), n * n_embd * @sizeOf(f32));
            p_off += n;
        }
        try cudaz.cuStreamSynchronize(lk.stream);

        // Diagnóstico P0-1: hidden del chunk (g_full) — gated info
        if (debugz.dbg.at(.info)) {
            const chk = try allocator.alloc(f32, chunk_len * n_embd);
            defer allocator.free(chk);
            try cudaz.cuMemcpyDtoH(@intFromPtr(chk.ptr), g_full.ptr(), chunk_len * n_embd * @sizeOf(f32));
            debugz.dbg.printLevel(.info, "[ppl] chunk {d}: hidden sum|v|={d:.3} max={d:.3} f0={d:.4}\n", .{ chunk_idx, debugz.sumAbsF32(chk), debugz.maxAbsF32(chk), chk[0] });
        }

        // rmsNorm + lm_head batcheado por tiles de lm_tile filas (pico VRAM
        // 16× menor que full-window; rmsNorm/projection soportan M arbitrario)
        var t_off: usize = 0;
        while (t_off < scored_in_chunk) {
            const m = @min(lm_tile, scored_in_chunk - t_off);
            const rows_dev = g_full.ptr() + (first_row + t_off) * n_embd * @sizeOf(f32);
            try lk.rmsNorm(rows_dev, @intFromPtr(g_out_norm.dev_ptr), g_normed.ptr(), m, n_embd, cfg.layer_norm_rms_epsilon);
            if (lm_head_bytes) |lb| {
                // Camino cuant-residente: bytes RAW del GGUF (qtype mapeado
                // on-load) → kernel dequantiza dentro. Batch m=lm_tile.
                try lk.qgemmLinear(allocator, g_normed.ptr(), lb, g_logits.ptr(), m, n_embd, vocab, lm_head_qtype);
            } else {
                if (lm_head_f16 == null) lm_head_f16 = try model.loadLmHead();
                try engine.linearProjectionDeviceF16(g_normed, lm_head_f16.?, &g_logits, m, n_embd, vocab);
            }
            try cudaz.cuStreamSynchronize(lk.stream);
            try cudaz.cuMemcpyDtoH(@intFromPtr(logit_rows.ptr), g_logits.ptr(), m * vocab * @sizeOf(f32));
            if (logits_target_host) |lt| {
                @memcpy(lt[pos + t_off ..][0 .. m * vocab], logit_rows[0 .. m * vocab]);
            }

            // NLL scoring (idéntico legacy, f32 logits)
            const per_tok_dbg = debugz.dbg.at(.trace);
            for (0..m) |i| {
                const row = logit_rows[i * vocab ..][0..vocab];
                const target = token_ids[pos + t_off + i];
                var max_l: f32 = -std.math.inf(f32);
                for (row) |v| {
                    if (v > max_l) max_l = v;
                }
                var sum_exp: f64 = 0;
                for (row) |v| sum_exp += @exp(@as(f64, v) - max_l);
                const lse: f64 = @as(f64, max_l) + @log(sum_exp);
                const tok_logit: f32 = row[target];
                const tok_nll = lse - @as(f64, tok_logit);
                nll_sum += tok_nll;
                n_scored += 1;
                if (per_tok_dbg) {
                    debugz.dbg.printLevel(.trace, "[ppl] tok pos={d} nll={d:.4} target={d}\n", .{ pos + t_off + i, tok_nll, target });
                }
            }
            t_off += m;
        }

        chunk_idx += 1;
        pos += scored_in_chunk;
        if (debugz.dbg.at(.info)) {
            debugz.dbg.printLevel(.info, "[ppl] chunk {d}: ctx_start={d} chunk_end={d} len={d} first_row={d} scored={d} nll_acc={d:.4}\n", .{ chunk_idx - 1, ctx_start, chunk_end, chunk_len, first_row, scored_in_chunk, nll_sum });
        }
    }

    if (recurrent_mode) {
        for (layers) |*l| l.resetStateGpu();
        const chunk_size: usize = 64;
        var rpos: usize = 1;
        while (rpos < token_ids.len) {
            const chunk_end = @min(rpos + chunk_size, token_ids.len);
            const chunk_len = chunk_end - rpos;
            for (layer_block_tables) |bt_opt| {
                if (bt_opt) |bt| {
                    if (bt.num_tokens > 0) try bt.truncate(paged_kv.block_alloc, 0);
                    try bt.appendTokens(paged_kv.block_alloc, chunk_len);
                }
            }
            var tpos = rpos;
            debugz.dbg.printLevel(.info, "[ppl] recurrent chunk: rpos={d} chunk_end={d} chunk_len={d}\n", .{ rpos, chunk_end, chunk_len });
            while (tpos < chunk_end) {
                const tok_slice = token_ids[tpos..tpos+1];
                if (emb_quant) |*qw| {
                    embedding.embeddingLookupQuant(qw, tok_slice, 1, 1, &hidden16_single.?);
                } else {
                    embedding.embeddingLookup(emb_f16.?, tok_slice, 1, 1, &hidden16_single.?);
                }
                for (0..n_embd) |i| {
                    stage[i] = @as(f32, hidden16_single.?.data[i]);
                }
                try cudaz.cuMemcpyHtoD(g_cur.ptr(), @intFromPtr(stage.ptr), n_embd * @sizeOf(f32));
                var cur = g_cur;
                var nxt = g_nxt;
                debugz.dbg.printLevel(.trace, "[ppl] about to forwardGPU tpos={d}\n", .{tpos});
                for (layers) |*layer| {
                    try hybrid_layer.HybridLayer.forwardGPU(layer, &lk, cur, &nxt, tpos - 1, 1, null);
                    const t2 = cur;
                    cur = nxt;
                    nxt = t2;
                }
                if (rlt_capture_buf) |cb| {
                    try cudaz.cuMemcpyDtoH(@intFromPtr(cb[tpos * n_embd..][0..n_embd].ptr), cur.ptr(), n_embd * @sizeOf(f32));
                }
                if (debugz.dbg.at(.info)) {
                    const chk = try allocator.alloc(f32, n_embd);
                    defer allocator.free(chk);
                    try cudaz.cuMemcpyDtoH(@intFromPtr(chk.ptr), cur.ptr(), n_embd * @sizeOf(f32));
                    debugz.dbg.printLevel(.info, "[ppl] tok {d}: hidden sum|v|={d:.3} max={d:.3} f0={d:.4}\n", .{ tpos, debugz.sumAbsF32(chk), debugz.maxAbsF32(chk), chk[0] });
                }
                try lk.rmsNorm(cur.ptr(), @intFromPtr(g_out_norm.dev_ptr), g_normed.ptr(), 1, n_embd, cfg.layer_norm_rms_epsilon);
                if (lm_head_bytes) |lb| {
                    try lk.qgemmLinear(allocator, g_normed.ptr(), lb, g_logits.ptr(), 1, n_embd, vocab, lm_head_qtype);
                } else {
                    if (lm_head_f16 == null) lm_head_f16 = try model.loadLmHead();
                    try engine.linearProjectionDeviceF16(g_normed, lm_head_f16.?, &g_logits, 1, n_embd, vocab);
                }
                try cudaz.cuStreamSynchronize(lk.stream);
                try cudaz.cuMemcpyDtoH(@intFromPtr(logit_rows.ptr), g_logits.ptr(), 1 * vocab * @sizeOf(f32));
                if (logits_target_host) |lt| {
                    @memcpy(lt[tpos * vocab..][0..vocab], logit_rows[0..vocab]);
                }
                const row = logit_rows[0..vocab];
                const target = token_ids[tpos];
                var max_l: f32 = -std.math.inf(f32);
                for (row) |v| { if (v > max_l) max_l = v; }
                var sum_exp: f64 = 0;
                for (row) |v| sum_exp += @exp(@as(f64, v) - max_l);
                const lse: f64 = @as(f64, max_l) + @log(sum_exp);
                const tok_logit: f32 = row[target];
                nll_sum += lse - @as(f64, tok_logit);
                n_scored += 1;
                if (debugz.dbg.at(.trace)) {
                    debugz.dbg.printLevel(.trace, "[ppl] tok pos={d} nll={d:.4} target={d}\n", .{ tpos, lse - @as(f64, tok_logit), target });
                }
                tpos += 1;
            }
            // No truncate entre chunks: KV queda committed para los tokens siguientes.
            rpos = chunk_end;
        }
        chunk_idx = if (n_scored > 0) n_scored else 0;
    }

    if (n_scored == 0) return error.NoTokensToScore;
    const ppl = @exp(nll_sum / @as(f64, @floatFromInt(n_scored)));
    const elapsed_s = @as(f64, @floatFromInt(@import("time").Timer.now() - timer)) / 1e9;
    try stdout.print("[+] ppl: {d} chunks, {d} tokens scored en {d:.1}s\n", .{ chunk_idx, n_scored, elapsed_s });
    try stdout.print("[+] PPL híbrido GPU ({s}/{s} KV): {d:.4}\n", .{ @tagName(effective_cache_type_k), @tagName(effective_cache_type_v), ppl });
    try stdout.flush();

    // R-4 (dev RLT): captura de hidden states reales para entrenamiento.
    if (params.capture_rlt_path.len > 0) {
        const T = token_ids.len;
        const last_attn = blk: {
            var idx: usize = 0;
            while (idx < eff_blocks) : (idx += 1) {
                if (cfg.isFullAttentionLayer(idx)) {
                    if (idx + 1 < eff_blocks and cfg.isFullAttentionLayer(idx + 1)) {
                        idx += 1;
                    } else {
                        break :blk idx;
                    }
                }
            }
            break :blk idx;
        };
        try stdout.print("[+] capture-rlt: última capa attn={d}, T={d} tokens, escribiendo {s}\n", .{ last_attn, T, params.capture_rlt_path });
        try stdout.flush();
        const e_host = try allocator.alloc(f32, T * n_embd);
        defer allocator.free(e_host);
        const tmp_gpu = try cublas.GpuTensor(f32).alloc(n_embd);
        defer tmp_gpu.deinit();
        const tok_buf = try allocator.alloc(u32, T);
        defer allocator.free(tok_buf);
        @memcpy(tok_buf, token_ids);
        for (0..T) |t| {
            const rows_dev = g_full.ptr() + t * n_embd * @sizeOf(f32);
            try cudaz.cuMemcpyDtoD(tmp_gpu.ptr(), rows_dev, n_embd * @sizeOf(f32));
            try cudaz.cuStreamSynchronize(lk.stream);
            try cudaz.cuMemcpyDtoH(@intFromPtr(e_host[t * n_embd ..].ptr), tmp_gpu.ptr(), n_embd * @sizeOf(f32));
        }
        const cwd = std.Io.Dir.cwd();
        var out_file = try cwd.createFile(io, params.capture_rlt_path, .{});
        defer out_file.close(io);
        var cap_buf2: [4096]u8 = undefined;
        var cw = out_file.writer(io, &cap_buf2);
        const writer = &cw.interface;
        try writer.writeInt(u32, 0x43544C52, .little);
        try writer.writeInt(u32, 1, .little);
        try writer.writeInt(u32, @intCast(T), .little);
        try writer.writeInt(u32, @intCast(n_embd), .little);
        try writer.writeInt(u32, 1, .little);
        for (tok_buf) |token_id| try writer.writeInt(u32, token_id, .little);
        for (e_host) |v| try writer.writeInt(u32, @bitCast(v), .little);
        try cw.flush();
        try stdout.print("[+] capture-rlt: escrito {s} ({d} bytes)\n", .{ params.capture_rlt_path, T * (4 + n_embd * 4) + 20 });
        try stdout.flush();
    }

    // R-4: volcado de lm_head f32 para entrenamiento
    if (params.dump_lm_head_path.len > 0) {
        const total = n_embd * vocab;
        const lh_f32 = try allocator.alloc(f32, total);
        defer allocator.free(lh_f32);
        if (lm_head_f16) |lh| {
            for (lh.data, 0..) |v, i| lh_f32[i] = @as(f32, v);
        } else if (lm_head_bytes) |lb| {
            const qt = lm_head_qtype;
            if (qt == 0) {
                const q4_block_bytes = 18;
                const q4_block_size = 32;
                for (0..total) |i| {
                    const block_idx = i / q4_block_size;
                    const in_block = i % q4_block_size;
                    const off = block_idx * q4_block_bytes;
                    const scale_raw = std.mem.readInt(u16, lb[off + 8..][0..2], .little);
                    const scale: f32 = @floatCast(@as(f16, @bitCast(scale_raw)));
                    const wb = lb[off + (in_block / 2)];
                    const q_val: f32 = @floatFromInt(if (in_block % 2 == 0) wb & 0xF else (wb >> 4) & 0xF);
                    lh_f32[i] = (q_val - 8.0) * scale;
                }
            } else {
                try stdout.print("[!] dump-lm-head: qtype {d} no soportada aún (solo q4_0=0)\n", .{qt});
                try stdout.flush();
            }
        } else {
            try stdout.print("[!] dump-lm-head: lm_head no disponible\n", .{});
            try stdout.flush();
        }
        const cwd = std.Io.Dir.cwd();
        var lh_file = try cwd.createFile(io, params.dump_lm_head_path, .{});
        defer lh_file.close(io);
        var lh_buf: [4096]u8 = undefined;
        var lhw = lh_file.writer(io, &lh_buf);
        const lh_writer = &lhw.interface;
        for (lh_f32) |v| try lh_writer.writeInt(u32, @bitCast(v), .little);
        try lhw.flush();
        try stdout.print("[+] dump-lm-head: escrito {s} ({d} bytes, d={d} vocab={d})\n", .{ params.dump_lm_head_path, total * 4, n_embd, vocab });
        try stdout.flush();
    }

    // R-4: volcado de logits target sampleados (1024 tokens) para entrenamiento MSE
    if (params.dump_logits_target_path.len > 0 and logits_target_host != null) {
        const lt = logits_target_host.?;
        const T = token_ids.len;
        const sample_n = @min(@as(usize, 1024), T);
        var rng2 = std.Random.DefaultPrng.init(42);
        const rand2 = rng2.random();
        var indices = try allocator.alloc(u32, sample_n);
        defer allocator.free(indices);
        var used = std.AutoHashMap(u32, void).init(allocator);
        defer used.deinit();
        var i: usize = 0;
        while (i < sample_n) {
            const idx: u32 = rand2.uintAtMost(u32, @as(u32, @intCast(T - 2))) + 1;
            if (used.contains(idx)) continue;
            used.put(idx, {}) catch {};
            indices[i] = idx;
            i += 1;
        }
        const cwd = std.Io.Dir.cwd();
        var lt_file = try cwd.createFile(io, params.dump_logits_target_path, .{});
        defer lt_file.close(io);
        var lt_buf: [4096]u8 = undefined;
        var ltw = lt_file.writer(io, &lt_buf);
        const lt_writer = &ltw.interface;
        try lt_writer.writeInt(u32, @intCast(sample_n), .little);
        for (indices) |idx| {
            try lt_writer.writeInt(u32, idx, .little);
            try lt_writer.writeInt(u32, token_ids[idx + 1], .little);
            for (0..vocab) |v| try lt_writer.writeInt(u32, @bitCast(lt[idx * vocab + v]), .little);
        }
        try ltw.flush();
        try stdout.print("[+] dump-logits-target: escrito {s} ({d} bytes, {d} tokens sampleados, vocab={d})\n", .{ params.dump_logits_target_path, sample_n * (4 + 4 + vocab * 4), sample_n, vocab });
        try stdout.flush();
    }
    if (rlt_capture_buf) |cb| {
        allocator.free(cb);
    }
}
