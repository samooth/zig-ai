const std = @import("std");
const builtin = @import("builtin");
const Tensor = @import("core").Tensor;
const matmul = @import("matmul");
const fa = @import("fa");
const transformer = @import("transformer");
const kvcache = @import("kv_cache");

const FlashAttention = fa.FlashAttention;
const FlashAttentionCpu = fa.FlashAttentionCpu;
const FlashAttentionConfig = fa.fa_config.FlashAttentionConfig;
const TransformerLayer = transformer.TransformerLayer;
const AttentionEngine = transformer.AttentionEngine;
const LayerPrecision = transformer.LayerPrecision;
const KVCacheManager = kvcache.KVCacheManager;
const KVCacheConfig = kvcache.KVCacheConfig;
const QuantFormat = kvcache.QuantFormat;
const pipeline = @import("pipeline");
const gguf_model = @import("gguf_model");
const gguf_tokenizer = @import("gguf_tokenizer");
const bpe = @import("tokenizer");
const cudaz = @import("cudaz");
const cublas = @import("cublas");
const layer_kernels = @import("layer_kernels");
const kvarn_gpu_cache = @import("kvarn_gpu_cache");
const build_options = @import("build_options");
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
const specdrv = @import("speculative");
// mmproj/vision (PLAN_MMPROJ Fase 0.4/3)
const mmproj_model = @import("mmproj_model");
const mmproj_config = @import("mmproj_config");
const vision_preprocess = @import("vision_preprocess");
const vision_encoder = @import("vision_clip_encoder");
const vision_clip_gpu = @import("vision_clip_gpu");
const vision_video = @import("vision_video");
const vision_inject = @import("vision_token_inject");
const gguf_moe = @import("gguf_moe"); // lane-f F/C: wiring MoE-aware
const moe_layer = @import("moe_layer"); // lane-f F/C
const moe_cuda = @import("moe_cuda"); // lane-f F/C
const offload_cache = @import("offload_cache"); // lane-f F/C
const host_bank = @import("host_bank"); // lane-f F/C: copy-once 4.3'
const cpu_executor = @import("cpu_executor"); // lane-f F/C: Contrato 8
const moe_cpu_gemv = @import("moe_cpu_gemv"); // lane-f F/C
/// Lane-B3 P3.5: KLD tool (subcomando `kld`).
const bench_kld = @import("kld");
/// Lane-B3 P3.4: presets INI (--preset, --models-dir).
const presets = @import("presets");

const SpecType = specdrv.SpecType; // fuente única (driver especulativo)

const QuantMode = inference.QuantMode;
const CliParams = inference.CliParams;

// ─── Delegación al módulo de inferencia (T1 extracción server F2) ─────────
// runHybridInference, captureDecodeGraph, formatPrompt, LmQ80Ctx,
// estimateCompressedWeightPerLayer, gpuKvPipelineReady, buildVramEstimate,
// cpuRmsNormFlat, cpuLmHeadLogits y helpers viven ahora en src/inference/cli.zig
// (movimiento mecánico byte-idéntico). Los alias mantienen los call-sites.
const inference = @import("inference");
const runHybridInference = inference.runHybridInference;
const captureDecodeGraph = inference.captureDecodeGraph;
const formatPrompt = inference.formatPrompt;
const gpuKvPipelineReady = inference.gpuKvPipelineReady;
const buildVramEstimate = inference.buildVramEstimate;
const estimateCompressedWeightPerLayer = inference.estimateCompressedWeightPerLayer;
const cpuRmsNormFlat = inference.cpuRmsNormFlat;
const cpuLmHeadLogits = inference.cpuLmHeadLogits;

/// Resuelve el backend matmul a usar según la opción `--backend`.
fn resolveBackend(backend: []const u8) matmul.Backend {
    if (std.mem.eql(u8, backend, "gpu")) return .cublas;
    if (std.mem.eql(u8, backend, "cpu")) return .parallel;
    // auto: GPU si CUDA está disponible, si no CPU
    if (@import("cudaz").isCudaAvailable()) return .cublas;
    return .parallel;
}

fn backendName(b: matmul.Backend) []const u8 {
    return switch (b) {
        .cublas => "gpu (cublas)",
        .openblas => "openblas",
        .parallel => "cpu (parallel)",
        .simd => "cpu (simd)",
        .tiled => "cpu (tiled)",
        .naive => "cpu (naive)",
        .fp8_block => "gpu (fp8_block)",
        .auto => "auto",
    };
}

/// Fuente única de verdad: un formato KV está listo en GPU sólo con TODO su
/// pipeline device (decode fusionado + append cuantizado + prefill cuantizado).
/// fp16/q8_0/iq4_xs/q8_k cumplen:
///   - q8_0: kvAppendQ8_0 + prefill q8_0 + decode reescrito (@02d1a7e).
///   - iq4_xs: decodeDevice extra-cubin (A) + causal prefill + B2.4-v2.
///   - q8_k: decode fix desalineación (A) + prefill_q8_k causal verde +
///     kvAppendQ8_K bit-exacto (B2.3) + guard preservación SBs (B2.6) —
///     flip consensuado A(05:00 ack)/B(17:55), validación E2E lane-c.
///   - q4_0: causa raíz histórica de los mismatch era la convención
///     INTERCALADA vs split-16 canónico — fix conjunto B @8e5561f
///     (encoder+append) con repro de B en verde y cierre de ticket A;
///     prefill causal verde (A). Validación E2E lane-c pendiente del
///     commit.
///   - iq1_s/iq3_s: prefill universal OK + append OK + decode extra-cubin.
///   - q4_k, q4_1, q5_0, q5_1, q8_1, tq1_0, tq2_0, mxfp4, iq4_nl: prefill
///     universal OK + append OK + decode extra-cubin. EN RUTA el
///     decodeDevice extendido (paged_attention/gpu_kernels.zig).
///   - iq2_s, iq2_xs, iq2_xxs, iq3_xxs: kernels de prefill cuantizado
///     DEGENERADOS (22-52s/prefill vs ~130ms esperados — grid o page-fault
///     en el super-block 256). Bug abierto; excluidos del gate hasta que
///     se corrija.
/// Pipeline KV en GPU: fuente única de verdad. Un formato está listo sólo con
/// TODO su pipeline device COMPLETO (append + prefill + decode device→device).
/// 13 formatos: los verificados E2E en 35896ae (fp16, q8_0, q4_0, q4_k,
/// q8_k, iq4_xs, q2_k, q3_k, q5_k, q6_k, iq1_s, iq3_s, iq4_nl).
/// iq2_s/iq2_xs/iq2_xxs/iq3_xxs: FUERA — el "degenerado" del ticket 7.5 SÍ
/// reproduce (verificado coordinador 2026-08-31 con -ctk/-ctv iq2_s en el
/// UD-IQ2_M: prefill 75.9s/15tok, decode 0.21 t/s; iq2_xs/iq3_xxs HANGAN).
/// Imprime nombre, compute capability y memoria de la GPU activa, y avisa si
/// la arquitectura detectada no coincide con la esperada por los cubins.
fn printGpuInfo(allocator: std.mem.Allocator, stdout: anytype) !void {
    const device = cudaz.cuDeviceGet(0) catch return;
    const info = cudaz.cuDeviceInfo(allocator, device) catch return;
    defer allocator.free(info.name);
    var arch_buf: [16]u8 = undefined;
    const gpu_arch = cudaz.gpuArchString(info, &arch_buf);
    try stdout.print(
        "[+] GPU: {s} (compute {d}.{d}, {s}, {d:.1} GiB)\n",
        .{ info.name, info.major, info.minor, gpu_arch, @as(f64, @floatFromInt(info.total_mem_bytes)) / (1024 * 1024 * 1024) },
    );
    try stdout.flush();
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.gpa;

    debugz.init();

    var stdout_buffer: [0x2000]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    // G5 fix: flush GARANTIZADO al salir — el leak-report del DebugAllocator
    // corre en el exit ANTES de que el writer vacíe su buffer ⇒ la salida
    // de generación del path legacy se perdía entera (log de 932B con solo
    // el banner). defer vacía SIEMPRE, exit()-independiente (si exit() se
    // llama antes, los handlers hacen flush explícito).
    defer stdout.flush() catch {};

    // Lane-B3 P4: detección temprana del subcomando `kld` (no requiere
    // --model). El subcomando se invoca como `zig-ai-engine kld ...` y
    // delega a `kldMain` sin pasar por el flujo de inferencia.
    const subcommand = subcommandFromArgs(init.minimal.args, allocator);
    if (subcommand) |sc| {
        if (std.mem.eql(u8, sc, "kld")) {
            const rc = kldMain(io, allocator, init.minimal.args, stdout);
            try stdout.flush();
            if (rc != 0) std.process.exit(rc);
            return;
        }
        // Subcomando desconocido: error claro y exit 1.
        try stdout.print("[!] Subcomando desconocido: {s}\n", .{sc});
        try stdout.print("    Subcomandos válidos: kld\n", .{});
        try stdout.flush();
        std.process.exit(1);
    }

    var params = try parseArgs(allocator, init.minimal.args, stdout);

    // Lane-B3 P3.4: precedencia CLI > preset > default. Si --preset está
    // presente, leer + parsear. Los errores de validación se reportan
    // aquí (InvalidCombination, etc.) antes de llegar a runInference.
    if (params.preset_path) |path| {
        var preset = presets.Preset{};
        presets.parseFromFile(io, allocator, path, &preset) catch |err| {
            try stdout.print("[!] preset: {s} (path={s})\n", .{ @errorName(err), path });
            try stdout.flush();
            std.process.exit(1);
        };
        // Aplicar precedencia. `null` en CLI cede al preset.
        if (params.context_length == 65536) {
            if (preset.ctx) |v| params.context_length = v;
        }
        if (params.batch_size == 2048) {
            if (preset.batch_size) |v| params.batch_size = v;
        }
        if (params.ubatch_size == 512) {
            if (preset.ubatch_size) |v| params.ubatch_size = v;
        }
        if (params.cache_type_k == .fp16) {
            if (preset.cache_type_k) |v| {
                if (QuantFormat.fromString(v)) |q| params.cache_type_k = q;
            }
        }
        if (params.cache_type_v == .fp16) {
            if (preset.cache_type_v) |v| {
                if (QuantFormat.fromString(v)) |q| params.cache_type_v = q;
            }
        }
        if (preset.dm_controller) |ctrl| {
            if (std.mem.eql(u8, ctrl, "off")) {
                params.spec_dm_controller = .off;
            } else if (std.mem.eql(u8, ctrl, "profit")) {
                params.spec_dm_controller = .profit;
            }
        }
        if (params.spec_dm_baseline_interval == null) {
            if (preset.dm_profit_baseline_interval) |v| {
                params.spec_dm_baseline_interval = std.math.cast(u32, v) orelse std.math.maxInt(u32);
            }
        }
        if (preset.reasoning_loop_mode) |mode_str| {
            if (specdrv.loop_guard.Mode.fromCli(mode_str)) |mode| {
                params.spec_loop_guard_mode = mode;
            }
        }
        if (preset.reasoning_loop_window) |v| {
            params.spec_loop_guard_window = std.math.cast(u32, v) orelse std.math.maxInt(u32);
        }
        if (preset.reasoning_loop_max_period) |v| {
            params.spec_loop_guard_max_period = std.math.cast(u32, v) orelse std.math.maxInt(u32);
        }
        if (preset.reasoning_loop_channel) |ch| {
            params.spec_loop_guard_channel = ch;
        }
    }

    try stdout.print("\n", .{});
    try stdout.print("=================================================\n", .{});
    try stdout.print("     Zig AI Engine — FlashAttention + Matmul     \n", .{});
    try stdout.print("=================================================\n", .{});
    try stdout.print("\n", .{});

    try stdout.print(
        "Sampling: temp={d:.3} top_k={d} top_p={d:.3} rep_penalty={d:.3} seed={d}\n",
        .{ params.sampler.temperature, params.sampler.top_k, params.sampler.top_p, params.sampler.repetition_penalty, params.seed },
    );
    try stdout.flush();

    const backend = resolveBackend(params.backend);
    try stdout.print("Backend matmul: {s}\n", .{backendName(backend)});
    try stdout.flush();

    // RLT: CLI flag overrides env-based default
    if (params.rlt_feedback) |override| {
        debugz.dbg.rlt_feedback = override;
    }

    if (params.model_path) |path| {
        if (backend == .cublas) {
            try @import("cudaz").ensureContext();
            try printGpuInfo(allocator, stdout);
        }

        // --serve: arranca el servidor HTTP (OpenAI/Anthropic/Ollama) en
        // lugar de hacer una inferencia única.
        if (params.serve) {
            const srv = @import("server");
            const cfg = srv.ServerConfig{
                .host = params.serve_host,
                .port = params.serve_port,
                .model_path = path,
                .cli_params = params,
                .api_key_file = params.serve_api_key_file,
                .audit_log = params.serve_audit_log,
                .rate_limit_rpm = params.serve_rate_limit_rpm,
                .tls_cert = params.serve_tls_cert,
                .tls_key = params.serve_tls_key,
                .warmup = params.serve_warmup,
            };
            try srv.runServer(allocator, cfg);
            return;
        }

        // ── mmproj/vision (PLAN_MMPROJ Fase 3) ──
        // Aquí sólo validamos el mmproj (banner). El encode de --image se
        // hace DENTRO de runHybridInference (necesita n_embd del target y
        // alimenta el prefill con pos-ids 2D — ver VisionInput).
        if (params.mmproj_path) |mm_path| {
            var mm = mmproj_model.MmprojModel.load(io, allocator, mm_path) catch |err| {
                try stdout.print("[!] mmproj: fallo al cargar {s}: {s}\n", .{ mm_path, @errorName(err) });
                try stdout.flush();
                return err;
            };
            defer mm.deinit();
            const mm_cfg = mm.config;
            try stdout.print("[+] mmproj vision: {s} (n_embd={d}, {d} capas, head {d}x{d}, patch={d}, merge={d})\n", .{
                mm_cfg.projector_type_str, mm_cfg.n_embd,   mm_cfg.n_layer,
                mm_cfg.n_head,             mm_cfg.head_dim, mm_cfg.patch_size,
                mm_cfg.spatial_merge_size,
            });
            if (params.image_paths.items.len == 0) {
                try stdout.print("[i] mmproj cargado sin --image: encode al usar --image\n", .{});
            }
            try stdout.flush();
        }

        if (params.image_paths.items.len > 0 and params.mmproj_path == null) {
            try stdout.print("[!] --image sin --mmproj: ignorando imagen (se requiere el encoder vision)\n", .{});
            try stdout.flush();
        }

        // lane-kvc tANS C-a: modo perplexity (antes que el inference normal
        // — no sampling, no generate; solo PPL del archivo).
        if (params.ppl_file) |ppl_path| {
            runPpl(io, allocator, path, ppl_path, params, backend, stdout) catch |err| {
                if (@errorReturnTrace()) |trace| {
                    _ = trace;
                    debugz.dbg.print("[pipeline] TRACE: {s}\n", .{@errorName(err)});
                    std.debug.dumpCurrentStackTrace(.{});
                }
                return err;
            };
            return;
        }

        runInference(io, allocator, path, params, backend, stdout) catch |err| {
            if (@errorReturnTrace()) |trace| {
                _ = trace; // 0.16.0 stable: builtin.StackTrace vs debug.StackTrace difieren
                debugz.dbg.print("[pipeline] TRACE: {s}\n", .{@errorName(err)});
                std.debug.dumpCurrentStackTrace(.{});
            }
            return err;
        };
        return;
    }

    // Sin --model: mostrar ayuda
    try printHelp(stdout);
}

/// Parsea los argumentos de línea de comandos.
fn parseArgs(allocator: std.mem.Allocator, args: std.process.Args, stdout: anytype) !CliParams {
    var params: CliParams = .{};
    var it = if (comptime builtin.target.os.tag == .windows)
        try std.process.Args.Iterator.initAllocator(args, allocator)
    else
        std.process.Args.Iterator.init(args);
    _ = it.next();

    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--model") or std.mem.eql(u8, arg, "-m")) {
            params.model_path = try nextValue(&it, arg);
        } else if (std.mem.eql(u8, arg, "--prompt") or std.mem.eql(u8, arg, "-p")) {
            // FIX métrica-prefill: -p era un alias NO registrado — cada bench
            // e2e anterior corrió con el default "Hola" (1 tok, prefill
            // constante ~115ms) y los flags -p se engullían en silencio.
            params.prompt = try nextValue(&it, "--prompt");
        } else if (std.mem.eql(u8, arg, "--ppl")) {
            // lane-kvc tANS C-a: perplexity de un archivo (semántica
            // llama.cpp: sliding window 2048 / stride 1024). Infra
            // permanente de medición (gates de cuantización KV).
            params.ppl_file = try nextValue(&it, "--ppl");
        } else if (std.mem.eql(u8, arg, "--kv-transfer")) {
            // KT-B (lane-f): pesos del mapper KV-transfer (formato .ktb).
            // Prefila el SOURCE y transfiere su KV al TARGET vía mappers
            // lineales por capa (identity = gate-0; dense = calibrado KT-A).
            // Env espejo: ZIG_AI_KV_TRANSFER.
            params.kv_transfer_path = try nextValue(&it, "--kv-transfer");
        } else if (std.mem.eql(u8, arg, "-n") or std.mem.eql(u8, arg, "--max-tokens")) {
            const v = try nextValue(&it, "--max-tokens");
            params.max_new_tokens = std.fmt.parseInt(usize, v, 10) catch 128;
        } else if (std.mem.eql(u8, arg, "--temperature")) {
            const v = try nextValue(&it, "--temperature");
            params.sampler.temperature = std.fmt.parseFloat(f32, v) catch 1.0;
        } else if (std.mem.eql(u8, arg, "--top-k")) {
            const v = try nextValue(&it, "--top-k");
            params.sampler.top_k = std.fmt.parseInt(usize, v, 10) catch 0;
        } else if (std.mem.eql(u8, arg, "--top-p")) {
            const v = try nextValue(&it, "--top-p");
            params.sampler.top_p = std.fmt.parseFloat(f32, v) catch 1.0;
        } else if (std.mem.eql(u8, arg, "--repetition-penalty")) {
            const v = try nextValue(&it, "--repetition-penalty");
            params.sampler.repetition_penalty = std.fmt.parseFloat(f32, v) catch 1.0;
        } else if (std.mem.eql(u8, arg, "--seed")) {
            const v = try nextValue(&it, "--seed");
            params.seed = std.fmt.parseInt(u64, v, 10) catch 42;
        } else if (std.mem.eql(u8, arg, "--backend")) {
            const v = try nextValue(&it, "--backend");
            if (std.mem.eql(u8, v, "auto") or std.mem.eql(u8, v, "cpu") or std.mem.eql(u8, v, "gpu")) {
                params.backend = v;
            } else {
                // Fail-fast: antes solo avisaba y seguía con el default —
                // runs de horas se perdieron por un typo (sesión G5/G6
                // 2026-09-14: '--backend cublas' → fallback gpu sin error).
                try stdout.print("[!] Backend inválido: {s} (auto|cpu|gpu)\n", .{v});
                return error.InvalidBackend;
            }
        } else if (std.mem.eql(u8, arg, "-ctk") or std.mem.eql(u8, arg, "--cache-type-k")) {
            // 9.1 (lane-c C-1): interceptar kvarn* ANTES del parseQuant —
            // setea bits kvarn (store por-grupo) y deja cache_type_k para
            // el body dual.
            const v = try nextValue(&it, arg);
            if (parseKvarnBits(v)) |kb| {
                if (kb.k > 0) {
                    params.kvarn_k_bits = kb.k;
                    params.kvarn_v_bits = kb.v;
                    debugz.dbg.printLevel(.info, "[milestone] KV cache KVarN K={d}b V={d}b (records grupo-128, lane-c 9.1)\n", .{ kb.k, kb.v });
                } else {
                    try stdout.print("[!] Formato kvarn inválido: {s} (kvarn<b>[v<b>], b=2|3|4|5|6|8)\n", .{v});
                }
            } else {
                if (QuantFormat.fromString(v)) |q| {
                    params.cache_type_k = q;
                } else {
                    debugz.dbg.printLevel(.info, "[cli] formato de cuantización inválido para {s}: {s}\n", .{ arg, v });
                    return error.InvalidQuantFormat;
                }
            }
        } else if (std.mem.eql(u8, arg, "-ctv") or std.mem.eql(u8, arg, "--cache-type-v")) {
            // 9.1: -ctv kvarn* solo ajusta V (K queda lo que -ctk dijo).
            const v = try nextValue(&it, arg);
            if (parseKvarnBits(v)) |kb| {
                if (kb.k > 0) {
                    params.kvarn_v_bits = kb.v;
                    debugz.dbg.printLevel(.info, "[milestone] KV cache KVarN V={d}b (sólo V)\n", .{kb.v});
                } else {
                    try stdout.print("[!] Formato kvarn inválido: {s} (kvarn<b>[v<b>], b=2|3|4|5|6|8)\n", .{v});
                }
            } else {
                if (QuantFormat.fromString(v)) |q| {
                    params.cache_type_v = q;
                } else {
                    debugz.dbg.printLevel(.info, "[cli] formato de cuantización inválido para {s}: {s}\n", .{ arg, v });
                    return error.InvalidQuantFormat;
                }
            }
        } else if (std.mem.eql(u8, arg, "--spec-draft-type-k")) {
            params.spec_draft_type_k = try parseQuant(&it, arg);
        } else if (std.mem.eql(u8, arg, "--spec-draft-type-v")) {
            params.spec_draft_type_v = try parseQuant(&it, arg);
        } else if (std.mem.eql(u8, arg, "-np")) {
            const v = try nextValue(&it, "-np");
            params.num_parallel = std.fmt.parseInt(usize, v, 10) catch 1;
            if (params.num_parallel < 1) params.num_parallel = 1;
        } else if (std.mem.eql(u8, arg, "-b") or std.mem.eql(u8, arg, "--batch-size")) {
            const v = try nextValue(&it, "--batch-size");
            params.batch_size = std.fmt.parseInt(usize, v, 10) catch 2048;
            if (params.batch_size < 1) params.batch_size = 1;
        } else if (std.mem.eql(u8, arg, "-ub") or std.mem.eql(u8, arg, "--ubatch-size")) {
            const v = try nextValue(&it, "--ubatch-size");
            params.ubatch_size = std.fmt.parseInt(usize, v, 10) catch 512;
            if (params.ubatch_size < 1) params.ubatch_size = 1;
        } else if (std.mem.eql(u8, arg, "--spec-type")) {
            const v = try nextValue(&it, "--spec-type");
            if (std.mem.eql(u8, v, "none")) {
                params.spec_type = .none;
            } else if (std.mem.eql(u8, v, "draft-mtp") or std.mem.eql(u8, v, "mtp")) {
                params.spec_type = .draft_mtp;
            } else if (std.mem.eql(u8, v, "draft-dflash") or std.mem.eql(u8, v, "dflash")) {
                params.spec_type = .draft_dflash;
            } else if (std.mem.eql(u8, v, "draft-dspark") or std.mem.eql(u8, v, "dspark")) {
                params.spec_type = .draft_dspark;
            } else if (std.mem.eql(u8, v, "draft-dflash2") or std.mem.eql(u8, v, "dflash2")) {
                params.spec_type = .draft_dflash2;
            } else {
                try stdout.print("[!] --spec-type inválido: {s} (none|draft-mtp|draft-dflash|draft-dspark|draft-dflash2)\n", .{v});
            }
        } else if (std.mem.eql(u8, arg, "--model-draft")) {
            params.model_draft = try nextValue(&it, "--model-draft");
        } else if (std.mem.eql(u8, arg, "--rlt-sidecar")) {
            params.rlt_sidecar_path = try nextValue(&it, "--rlt-sidecar");
        } else if (std.mem.eql(u8, arg, "--spec-draft-block-size")) {
            const v = try nextValue(&it, "--spec-draft-block-size");
            params.spec_draft_block_size = std.fmt.parseInt(usize, v, 10) catch 0;
        } else if (std.mem.eql(u8, arg, "--spec-p-min")) {
            const v = try nextValue(&it, "--spec-p-min");
            params.spec_p_min = std.fmt.parseFloat(f32, v) catch 0.1;
        } else if (std.mem.eql(u8, arg, "--spec-dm-controller")) {
            // 9.5: ProfitController (BeeLlama P1.1). off|profit.
            const v = try nextValue(&it, "--spec-dm-controller");
            if (std.mem.eql(u8, v, "off")) {
                params.spec_dm_controller = .off;
            } else if (std.mem.eql(u8, v, "profit")) {
                params.spec_dm_controller = .profit;
            } else {
                try stdout.print("[!] --spec-dm-controller inválido: {s} (off|profit)\n", .{v});
            }
        } else if (std.mem.eql(u8, arg, "--reasoning-loop-mode")) {
            // 9.6: LoopGuard (BeeLlama P1.2). off|force-close|warn.
            const v = try nextValue(&it, "--reasoning-loop-mode");
            if (std.mem.eql(u8, v, "off")) {
                params.spec_loop_guard_mode = .off;
            } else if (std.mem.eql(u8, v, "force-close")) {
                params.spec_loop_guard_mode = .force_close;
            } else if (std.mem.eql(u8, v, "warn")) {
                params.spec_loop_guard_mode = .warn;
            } else {
                try stdout.print("[!] --reasoning-loop-mode inválido: {s} (off|force-close|warn)\n", .{v});
            }
        } else if (std.mem.eql(u8, arg, "--reasoning-loop-window")) {
            const v = try nextValue(&it, "--reasoning-loop-window");
            params.spec_loop_guard_window = std.fmt.parseInt(u32, v, 10) catch 64;
        } else if (std.mem.eql(u8, arg, "--reasoning-loop-max-period")) {
            const v = try nextValue(&it, "--reasoning-loop-max-period");
            params.spec_loop_guard_max_period = std.fmt.parseInt(u32, v, 10) catch 16;
        } else if (std.mem.eql(u8, arg, "--spec-lookup-n")) {
            const v = try nextValue(&it, "--spec-lookup-n");
            params.spec_lookup_n = std.fmt.parseInt(usize, v, 10) catch 0;
        } else if (std.mem.eql(u8, arg, "--spec-selector-top-k")) {
            const v = try nextValue(&it, "--spec-selector-top-k");
            params.spec_selector_top_k = std.fmt.parseInt(usize, v, 10) catch 10;
        } else if (std.mem.eql(u8, arg, "--spec-selector-rank")) {
            const v = try nextValue(&it, "--spec-selector-rank");
            params.spec_selector_rank = std.fmt.parseInt(usize, v, 10) catch 128;
        } else if (std.mem.eql(u8, arg, "--download-dflash")) {
            params.download_dflash = true;
        } else if (std.mem.eql(u8, arg, "--download-dspark")) {
            params.download_dspark = true;
        } else if (std.mem.eql(u8, arg, "--download-dflash2")) {
            params.download_dflash2 = true;
        } else if (std.mem.eql(u8, arg, "--mmproj")) {
            params.mmproj_path = try nextValue(&it, "--mmproj");
        } else if (std.mem.eql(u8, arg, "--image")) {
            try params.image_paths.append(allocator, try nextValue(&it, "--image"));
        } else if (std.mem.eql(u8, arg, "--video")) {
            try params.video_paths.append(allocator, try nextValue(&it, "--video"));
        } else if (std.mem.eql(u8, arg, "--spec-draft-n-max")) {
            const v = try nextValue(&it, "--spec-draft-n-max");
            params.spec_draft_n_max = std.fmt.parseInt(usize, v, 10) catch 16;
        } else if (std.mem.eql(u8, arg, "--quant")) {
            const v = try nextValue(&it, "--quant");
            if (std.mem.eql(u8, v, "off")) {
                params.quant = .off;
            } else if (std.mem.eql(u8, v, "auto")) {
                params.quant = .auto;
            } else if (std.mem.eql(u8, v, "fp8")) {
                params.quant = .fp8;
            } else {
                try stdout.print("[!] --quant inválido: {s} (auto|off|fp8)\n", .{v});
            }
        } else if (std.mem.eql(u8, arg, "--layer-stream")) {
            params.layer_stream = true;
        } else if (std.mem.eql(u8, arg, "--layer-stream-max")) {
            const v = try nextValue(&it, "--layer-stream-max");
            params.layer_stream_max = std.fmt.parseInt(usize, v, 10) catch 2;
        } else if (std.mem.eql(u8, arg, "--f16-max-resident")) {
            const v = try nextValue(&it, "--f16-max-resident");
            params.f16_max_resident = std.fmt.parseInt(usize, v, 10) catch 2;
        } else if (std.mem.eql(u8, arg, "-ngl") or std.mem.eql(u8, arg, "--n-gpu-layers") or std.mem.eql(u8, arg, "--gpu-layers")) {
            const v = try nextValue(&it, "--n-gpu-layers");
            params.n_gpu_layers = std.fmt.parseInt(usize, v, 10) catch 0;
        } else if (std.mem.eql(u8, arg, "-c") or std.mem.eql(u8, arg, "--ctx-size") or
            std.mem.eql(u8, arg, "--ctx") or std.mem.eql(u8, arg, "--context-size") or
            std.mem.eql(u8, arg, "--context") or std.mem.eql(u8, arg, "--ctx_size"))
        {
            // Alias llama.cpp (--ctx es el más común en repros rotos, ver
            // TODO 7.1b: "repros usaban flags inexistentes --ctx/-cl").
            const v = try nextValue(&it, "--ctx-size");
            params.context_length = std.fmt.parseInt(usize, v, 10) catch 65536;
        } else if (std.mem.eql(u8, arg, "-jinja") or std.mem.eql(u8, arg, "--jinja")) {
            params.use_jinja = true;
        } else if (std.mem.eql(u8, arg, "--serve")) {
            params.serve = true;
        } else if (std.mem.eql(u8, arg, "--host")) {
            params.serve_host = try nextValue(&it, "--host");
        } else if (std.mem.eql(u8, arg, "--port")) {
            const v = try nextValue(&it, "--port");
            params.serve_port = std.fmt.parseInt(u16, v, 10) catch 8080;
        } else if (std.mem.eql(u8, arg, "--api-key-file")) {
            params.serve_api_key_file = try nextValue(&it, "--api-key-file");
        } else if (std.mem.eql(u8, arg, "--audit-log")) {
            params.serve_audit_log = try nextValue(&it, "--audit-log");
        } else if (std.mem.eql(u8, arg, "--rate-limit")) {
            const v = try nextValue(&it, "--rate-limit");
            const rpm = std.fmt.parseInt(u32, v, 10) catch 60;
            params.serve_rate_limit_rpm = rpm;
        } else if (std.mem.eql(u8, arg, "--tls-cert")) {
            params.serve_tls_cert = try nextValue(&it, "--tls-cert");
        } else if (std.mem.eql(u8, arg, "--tls-key")) {
            params.serve_tls_key = try nextValue(&it, "--tls-key");
        } else if (std.mem.eql(u8, arg, "--no-warmup")) {
            params.serve_warmup = false;
            // ─── RLT (Recurrent Looped Transformer) ─────────────────────────
        } else if (std.mem.eql(u8, arg, "--rlt-feedback")) {
            params.rlt_feedback = true;
        } else if (std.mem.eql(u8, arg, "--no-rlt-feedback")) {
            params.rlt_feedback = false;
        } else if (std.mem.eql(u8, arg, "--recurrent-prefill")) {
            params.recurrent_prefill = true;
        } else if (std.mem.eql(u8, arg, "--exact-replay")) {
            params.exact_replay = true;
        } else if (std.mem.eql(u8, arg, "--capture-rlt")) {
            params.capture_rlt_path = try nextValue(&it, "--capture-rlt");
        } else if (std.mem.eql(u8, arg, "--dump-lm-head")) {
            params.dump_lm_head_path = try nextValue(&it, "--dump-lm-head");
        } else if (std.mem.eql(u8, arg, "--dump-logits-target")) {
            params.dump_logits_target_path = try nextValue(&it, "--dump-logits-target");
        } else if (std.mem.eql(u8, arg, "--swa")) {
            params.swa = try nextInt(&it, "--swa");
        } else if (std.mem.eql(u8, arg, "--kv-offload")) {
            params.kv_offload = true;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            try printHelp(stdout);
            try stdout.flush();
            std.process.exit(0);
        } else if (std.mem.eql(u8, arg, "--version") or std.mem.eql(u8, arg, "-v")) {
            // R0 release v0.1.0: stamping con git sha a compile-time.
            const version_info = @import("version_info");
            try stdout.print("zig-ai-engine {s} ({s})\n", .{ version_info.version, version_info.git_sha });
            try stdout.flush();
            std.process.exit(0);
            // ─── Lane-B3 (BeeLlama P1.1/P1.2/P3.4) ────────────────────────
        } else if (std.mem.eql(u8, arg, "--spec-dm-controller")) {
            const v = try nextValue(&it, "--spec-dm-controller");
            // Acepta null-safe: "" = explícito off, "profit"/"off" = valor.
            if (std.mem.eql(u8, v, "off")) {
                params.spec_dm_controller = .off;
            } else if (std.mem.eql(u8, v, "profit")) {
                params.spec_dm_controller = .profit;
            } else {
                try stdout.print("[!] --spec-dm-controller inválido: {s} (off|profit)\n", .{v});
            }
        } else if (std.mem.eql(u8, arg, "--spec-dm-profit-baseline-interval")) {
            const v = try nextValue(&it, "--spec-dm-profit-baseline-interval");
            params.spec_dm_baseline_interval = std.fmt.parseInt(u32, v, 10) catch 1024;
        } else if (std.mem.eql(u8, arg, "--reasoning-loop-mode")) {
            const v = try nextValue(&it, "--reasoning-loop-mode");
            if (specdrv.loop_guard.Mode.fromCli(v)) |mode| {
                params.spec_loop_guard_mode = mode;
            } else {
                try stdout.print("[!] --reasoning-loop-mode inválido: {s} (force-close|warn|off)\n", .{v});
            }
        } else if (std.mem.eql(u8, arg, "--reasoning-loop-window")) {
            const v = try nextValue(&it, "--reasoning-loop-window");
            params.spec_loop_guard_window = std.fmt.parseInt(u32, v, 10) catch 512;
        } else if (std.mem.eql(u8, arg, "--reasoning-loop-max-period")) {
            const v = try nextValue(&it, "--reasoning-loop-max-period");
            params.spec_loop_guard_max_period = std.fmt.parseInt(u32, v, 10) catch 3;
        } else if (std.mem.eql(u8, arg, "--reasoning-loop-channel")) {
            const v = try nextValue(&it, "--reasoning-loop-channel");
            if (specdrv.loop_guard.ChannelCfg.fromCli(v) != null) {
                params.spec_loop_guard_channel = v;
            } else {
                try stdout.print("[!] --reasoning-loop-channel inválido: {s} (hidden|visible|both)\n", .{v});
            }
        } else if (std.mem.eql(u8, arg, "--preset")) {
            params.preset_path = try nextValue(&it, "--preset");
        } else if (std.mem.eql(u8, arg, "--models-dir")) {
            params.models_dir = try nextValue(&it, "--models-dir");
        } else if (std.mem.eql(u8, arg, "--models-preset")) {
            params.models_preset_path = try nextValue(&it, "--models-preset");
        } else {
            // UX (2026-09-13, TODO 7.1b): el error críptico "Argumento
            // desconocido" hizo que repros enteros corrieran con defaults
            // silenciosos (--ctx ignorado ⇒ ctx 65536). Sugerir el flag
            // canónico para los prefijos confundibles más comunes.
            const hint: ?[]const u8 = blk: {
                const pairs = [_]struct { wrong: []const u8, right: []const u8 }{
                    .{ .wrong = "--ctx", .right = "--ctx-size (-c)" },
                    .{ .wrong = "--context", .right = "--ctx-size (-c)" },
                    .{ .wrong = "--ngl", .right = "-ngl" },
                    .{ .wrong = "--gpu-layer", .right = "-ngl" },
                    .{ .wrong = "--layers", .right = "-ngl" },
                    .{ .wrong = "--temp", .right = "--temperature" },
                    .{ .wrong = "--n-predict", .right = "-n" },
                    .{ .wrong = "--threads", .right = "--cpu-workers (ZIG_AI_CPU_WORKERS)" },
                    .{ .wrong = "--batch", .right = "--ubatch-size" },
                    .{ .wrong = "-cl", .right = "-ngl (capas GPU)" },
                    .{ .wrong = "--gpu-layers", .right = "-ngl" },
                };
                for (pairs) |pr| {
                    if (std.mem.startsWith(u8, arg, pr.wrong)) break :blk pr.right;
                }
                break :blk null;
            };
            if (hint) |h| {
                try stdout.print("[!] Argumento desconocido: {s} — ¿quieres decir '{s}'?\n", .{ arg, h });
            } else {
                try stdout.print("[!] Argumento desconocido: {s}\n", .{arg});
            }
        }
    }
    return params;
}

 fn nextValue(it: *std.process.Args.Iterator, name: []const u8) ![]const u8 {
     return it.next() orelse {
         debugz.dbg.printLevel(.info, "[cli] falta valor para {s}\n", .{name});
         return error.MissingArgumentValue;
     };
 }

 fn nextInt(it: *std.process.Args.Iterator, name: []const u8) !usize {
     const raw = try nextValue(it, name);
     return std.fmt.parseInt(usize, raw, 10) catch |err| {
         debugz.dbg.printLevel(.info, "[cli] valor inválido para {s}: {s}\n", .{ name, raw });
         return err;
     };
 }

/// Devuelve la ruta al ejecutable actual. En Linux usa /proc/self/exe.
fn readSelfExePath(allocator: std.mem.Allocator) ![]u8 {
    if (@import("builtin").os.tag == .linux) {
        var buf: [4096]u8 = undefined;
        const n = std.os.linux.readlink("/proc/self/exe", &buf, buf.len);
        if (n == 0) return error.ReadlinkFailed;
        return try allocator.dupe(u8, buf[0..n]);
    }
    return try allocator.dupe(u8, "zig-ai-engine");
}

fn parseQuant(it: *std.process.Args.Iterator, flag: []const u8) !QuantFormat {
    const v = try nextValue(it, flag);
    if (QuantFormat.fromString(v)) |q| return q;
    debugz.dbg.printLevel(.info, "[cli] formato de cuantización inválido para {s}: {s}\n", .{ flag, v });
    return error.InvalidQuantFormat;
}

/// 9.1 (lane-c C-1): parse de `-ctk/-ctv kvarn<b>[v<b>]` (b ∈ 2|3|4|5|6|8).
/// Formas: `kvarn4` (K y V a 4 bits), `kvarn4v6` (K=4, V=6), `kvarn8v2`.
/// Devuelve (k_bits, v_bits) — 0 en ambos si `v` NO es kvarn* (el caller
/// usa parseQuant clásico). KvarnType.parse de kvarn.zig es el oráculo de
/// bits válidos.
fn parseKvarnBits(v: []const u8) ?struct { k: u8, v: u8 } {
    if (!std.mem.startsWith(u8, v, "kvarn")) return null;
    const rest = v["kvarn".len..];
    // Formato: <kb>[v<vb>] — ambos dígitos simples (bits ≤ 8).
    if (rest.len == 0 or rest.len > 3) return .{ .k = 0, .v = 0 };
    const k_bits: u8 = rest[0] - '0';
    var v_bits: u8 = k_bits; // simétrico por defecto
    if (rest.len >= 3 and rest[1] == 'v') {
        v_bits = rest[2] - '0';
    } else if (rest.len != 1) {
        return .{ .k = 0, .v = 0 };
    }
    const kvarn_mod = @import("kv_cache").kvarn;
    if (!kvarn_mod.isValidBits(k_bits) or !kvarn_mod.isValidBits(v_bits)) {
        debugz.dbg.printLevel(.info, "[cli] bits kvarn inválidos: {s} (válidos 2|3|4|5|6|8)\n", .{v});
        return .{ .k = 0, .v = 0 };
    }
    return .{ .k = k_bits, .v = v_bits };
}

fn printHelp(stdout: anytype) !void {
    try stdout.print(
        \\Uso: zig-ai-engine [opciones]
        \\                                  INFERENCIA
        \\  -m, --model <ruta>          Ruta a un modelo GGUF (activa inferencia)
        \\  -p, --prompt <texto>        Prompt de entrada
        \\  -n, --max-tokens <n>        Máximo de tokens a generar (def: 128)
        \\
        \\Servidor HTTP (--serve; OpenAI/Anthropic/Ollama compat):
        \\  --serve                     Arranca el servidor HTTP en vez de inferir
        \\  --host <ip>                 Bind del server (def: 127.0.0.1; no-loopback
        \\                              requiere --tls-cert/--tls-key)
        \\  --port <n>                  Puerto del server (def: 8080)
        \\  --api-key-file <ruta>       API keys, una por línea, permisos 0600
        \\  --audit-log <ruta>          Audit JSON-lines (fail-soft)
        \\  --rate-limit <n>            Rate limit req/min por IP (def: 60)
        \\  --tls-cert <ruta>           Certificado TLS PEM (bind público)
        \\  --tls-key <ruta>            Clave privada TLS PEM (bind público)
        \\  --no-warmup                 Salta la inferencia de warmup inicial
        \\
        \\  -ngl, --n-gpu-layers <n>    Capas a offload a GPU (def: auto; 0 = CPU,
        \\                              >0 = todas las capas a GPU)
        \\  -c, --ctx-size <n>          Contexto de inferencia (def: 65536; 0 = contexto entrenado)
        \\  --temperature <f>           Temperatura (def: 1.0; <=0 = greedy)
        \\  --top-k <n>                 Top-k (def: 0 = desactivado)
        \\  --top-p <f>                 Top-p / nucleus (def: 1.0 = desactivado)
        \\  --repetition-penalty <f>    Repetition penalty (def: 1.0 = desactivado)
        \\  --seed <n>                  Semilla del RNG (def: 42)
        \\  --backend <auto|cpu|gpu>    Backend matmul (def: auto → GPU si disponible)
        \\  -ctk, --cache-type-k <fmt>  Cuantización cache K (q8_0|q4_0|q4_1|fp16|...)
        \\  -ctv, --cache-type-v <fmt>  Cuantización cache V (q8_0|q4_0|q4_1|fp16|...)
        \\  --spec-draft-type-k <fmt>   Cuantización cache K del draft (def: fp16)
        \\  --spec-draft-type-v <fmt>   Cuantización cache V del draft (def: fp16)
        \\  -np <n>                     Secuencias paralelas / prefillo (def: 1)
        \\  -b, --batch-size <n>        Batch lógico de prefill en tokens (def: 2048)
        \\  -ub, --ubatch-size <n>      Batch físico por llamada GPU (def: 512)
        \\  --spec-type <type>          Decodificación especulativa (def: none)
        \\                             Types: none, draft-mtp, draft-dflash, draft-dspark, draft-dflash2
        \\  --model-draft <path>         Modelo GGUF draft sidecar (DFlash/DSpark)
        \\  --rlt-sidecar <path>         Sidecar GGUF con pesos RLT entrenados (blk.N.rlt.feedback_{{gate,state}})
        \\  --spec-draft-n-max <n>      Tokens draft por round (def: 16)
        \\  --spec-draft-n-min <n>      Mínimo tokens draft aceptados (def: 4)
        \\  --spec-draft-block-size <n> Block size dflash (0 = auto from GGUF)
        \\  --spec-p-min <f>            Probabilidad mínima draft (def: 0.1)
        \\  --spec-dm-controller <c>  ProfitController (9.5): off|profit (def: profit)
        \\  --reasoning-loop-mode <m> LoopGuard (9.6): off|force-close|warn (def: off)
        \\  --reasoning-loop-window <n> LoopGuard ventana (def: 64)
        \\  --reasoning-loop-max-period <n> LoopGuard periodo max (def: 16)
        \\  --spec-lookup-n <n>         Prompt-lookup n-gram fill (0 = disabled, def: 5)
        \\  --spec-selector-top-k <n>   DFlash2 selector top-k (def: 10)
        \\  --spec-selector-rank <n>    DFlash2 selector rank (def: 128)
        \\  --download-dflash           Auto-descargar sidecar DFlash de HF
        \\  --download-dspark           Auto-descargar sidecar DSpark de HF
        \\  --download-dflash2          Auto-descargar sidecar DFlash2 de HF
        \\  --mmproj <path>             mmproj GGUF vision (encoder CLIP ViT)
        \\  --image <path>              Imagen de entrada para el encoder vision
        \\  --spec-draft-n-max <n>      Tokens draft por round (def: 16)
        \\  --quant <auto|off|fp8>      GEMM de pesos cuantizados Q4_0/FP8 (def: auto)
        \\  --layer-stream             Activar layer streaming (prefetch async + LRU)
        \\  --layer-stream-max <n>      Max capas residentes en VRAM (def: 2)
        \\  --f16-max-resident <n>      Max capas con pesos f16 materializados (def: 2, 7.1a)
        \\  --rlt-feedback              Forzar merge RLT recurrente ON (incluso sin pesos GGUF)
        \\  --no-rlt-feedback           Forzar merge RLT OFF (cero overhead)
        \\  --recurrent-prefill         Prefill secuencial recurrente (mismo path que decode)
        \\  --exact-replay              Replay completo para spec decode (consistencia tras weight update)
        \\  --capture-rlt <path>        Captura hidden states reales a .rltcap para entrenamiento RLT (fase 2)
        \\  --dump-lm-head <path>      Vuelca lm_head f32 (d×vocab) a .bin para entrenamiento RLT
         \\  --dump-logits-target <path> Vuelca logits target (T×vocab f32) a .bin para entrenamiento RLT MSE
         \\  --swa <n>                   Cap ventana sliding-window attention por capa (0 = full context)
         \\  --kv-offload                Forzar offload KV a CPU/RAM host (reduce VRAM pico)
         \\  -jinja                     Usar plantilla chat jinja del tokenizer
        \\  -h, --help                  Muestra esta ayuda
        \\
        \\                                  LANE-B3 (BeeLlama P1.1/P1.2/P3.4)
        \\  --spec-dm-controller <off|profit>
        \\                              Adaptive draft-max controller (P1.1; def: off)
        \\  --spec-dm-profit-baseline-interval <n>
        \\                              Ciclos entre re-baselines del controller (def: 1024)
        \\  --reasoning-loop-mode <force-close|warn|off>
        \\                              Loop guard (P1.2; def: force-close)
        \\  --reasoning-loop-window <n>  Ventana del loop guard (def: 512)
        \\  --reasoning-loop-max-period <n>
        \\                              Repeticiones para confirmar loop (def: 3)
        \\  --reasoning-loop-channel <hidden|visible|both>
        \\                              Canal vigilado (def: hidden)
        \\  --preset <path.ini>          Preset INI (P3.4 C5; CLI>preset>default)
        \\  --models-dir <dir>          Directorio de modelos (placeholder Fase 3)
        \\  --models-preset <path.ini>  Preset dentro de --models-dir (placeholder)
        \\
        \\                                  SUBCOMANDOS
        \\  kld                          Mide KL-divergence del pipeline KV vs BF16
        \\                              (ver `zig-ai-engine kld --help`)
        \\
    , .{});
}

/// Lane-B3 P4: detecta el primer argumento no-flag como subcomando
/// (`zig-ai-engine kld ...`). Si el primer arg tras argv[0] no empieza
/// por `-` y matchea un subcomando conocido, lo devuelve. Si no, null
/// (caller sigue con el flujo de inferencia normal).
fn subcommandFromArgs(args: std.process.Args, allocator: std.mem.Allocator) ?[]const u8 {
    var it = if (comptime builtin.target.os.tag == .windows)
        std.process.Args.Iterator.initAllocator(args, allocator) catch return null
    else
        std.process.Args.Iterator.init(args);
    _ = it.next(); // argv[0]
    const first = it.next() orelse return null;
    if (first.len > 0 and first[0] == '-') return null; // flag
    if (std.mem.eql(u8, first, "kld")) return first;
    return null;
}

/// Lane-B3 P3.5 / P4: handler del subcomando `kld`. Stub: parsea flags
/// propias, lee el corpus, emite el report header y delega en
/// `bench.kld.run()` (que actualmente retorna NotImplemented hasta
/// que B2 cierre P0.1). Devuelve exit code.
fn kldMain(
    _: std.Io,
    allocator: std.mem.Allocator,
    args: std.process.Args,
    stdout: anytype,
) u8 {
    var cfg = bench_kld.KldConfig{};
    var it = if (comptime builtin.target.os.tag == .windows)
        std.process.Args.Iterator.initAllocator(args, allocator) catch return 1
    else
        std.process.Args.Iterator.init(args);
    _ = it.next(); // argv[0]
    _ = it.next(); // "kld"

    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
            cfg.model_path = nextValue(&it, "--model") catch return printCliErr(stdout, "falta valor para --model");
        } else if (std.mem.eql(u8, arg, "--kld-corpus")) {
            cfg.corpus_path = nextValue(&it, "--kld-corpus") catch return printCliErr(stdout, "falta valor para --kld-corpus");
        } else if (std.mem.eql(u8, arg, "-ctk") or std.mem.eql(u8, arg, "--cache-type-k")) {
            cfg.cache_type_k = nextValue(&it, "--cache-type-k") catch return printCliErr(stdout, "falta valor para --cache-type-k");
        } else if (std.mem.eql(u8, arg, "-ctv") or std.mem.eql(u8, arg, "--cache-type-v")) {
            cfg.cache_type_v = nextValue(&it, "--cache-type-v") catch return printCliErr(stdout, "falta valor para --cache-type-v");
        } else if (std.mem.eql(u8, arg, "--kv-tail-tokens")) {
            const v = nextValue(&it, "--kv-tail-tokens") catch return printCliErr(stdout, "falta valor para --kv-tail-tokens");
            cfg.kv_tail_tokens = std.fmt.parseInt(usize, v, 10) catch 0;
        } else if (std.mem.eql(u8, arg, "-b") or std.mem.eql(u8, arg, "--batch-size")) {
            const v = nextValue(&it, "--batch-size") catch return printCliErr(stdout, "falta valor para --batch-size");
            cfg.batch_size = std.fmt.parseInt(usize, v, 10) catch 0;
        } else if (std.mem.eql(u8, arg, "-ub") or std.mem.eql(u8, arg, "--ubatch-size")) {
            const v = nextValue(&it, "--ubatch-size") catch return printCliErr(stdout, "falta valor para --ubatch-size");
            cfg.ubatch_size = std.fmt.parseInt(usize, v, 10) catch 0;
        } else if (std.mem.eql(u8, arg, "--seed")) {
            const v = nextValue(&it, "--seed") catch return printCliErr(stdout, "falta valor para --seed");
            cfg.seed = std.fmt.parseInt(u64, v, 10) catch 42;
        } else if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            printKldHelp(stdout) catch {};
            return 0;
        } else if (arg.len > 0 and arg[0] == '-') {
            stdout.print("[!] kld: flag desconocida: {s}\n", .{arg}) catch {};
            return 2;
        } else {
            stdout.print("[!] kld: argumento posicional inesperado: {s}\n", .{arg}) catch {};
            return 2;
        }
    }

    if (cfg.model_path.len == 0) {
        stdout.print("[!] kld: -m/--model es obligatorio\n", .{}) catch {};
        return 2;
    }
    if (cfg.corpus_path.len == 0) {
        stdout.print("[!] kld: --kld-corpus es obligatorio\n", .{}) catch {};
        return 2;
    }

    cfg.validateBatches() catch {
        stdout.print("[!] kld: -b/-ub inválidos (ubatch > batch o cero)\n", .{}) catch {};
        return 2;
    };

    // Header del report siempre (incluso si run() falla después) — el
    // usuario ve que el parseo CLI pasó. Las muestras reales las
    // produciría el run() con KVarN real (post-B2 P0.1); por ahora
    // KldResult vacío ⇒ tabla con median=0 p99.9=0 n=0.
    const empty = bench_kld.KldResult{ .samples = &.{} };
    bench_kld.writeReport(stdout, cfg, empty) catch |e| {
        stdout.print("[!] kld: report falló: {s}\n", .{@errorName(e)}) catch {};
        return 3;
    };

    bench_kld.run(allocator, cfg) catch |err| {
        stdout.print("[!] kld.run falló: {s}\n", .{@errorName(err)}) catch {};
        stdout.print("    (esperando B2 P0.1: KVarN real; la tool ya está lista)\n", .{}) catch {};
        return 4;
    };
    return 0;
}

fn printCliErr(stdout: anytype, msg: []const u8) u8 {
    stdout.print("[!] {s}\n", .{msg}) catch {};
    return 2;
}

fn printKldHelp(stdout: anytype) !void {
    try stdout.print(
        \\Uso: zig-ai-engine kld -m <model.gguf> --kld-corpus <path> [opciones]
        \\  Mide KL-divergence del pipeline KV (candidato cuantizado) contra
        \\  un baseline BF16. Regla Bee: -b y -ub idénticos en baseline y
        \\  candidato; -b >= -ub; ambos > 0.
        \\
        \\Obligatorios:
        \\  -m, --model <ruta>           Modelo GGUF (baseline+candidato)
        \\  --kld-corpus <path>         Corpus de texto (prefill del bench)
        \\
        \\Opcionales:
        \\  -ctk, --cache-type-k <fmt>  Formato K candidato (def: bf16)
        \\  -ctv, --cache-type-v <fmt>  Formato V candidato (def: bf16)
        \\  --kv-tail-tokens <n>        Cola exacta f16 (def: 0)
        \\  -b, --batch-size <n>        Batch lógico de prefill (def: 2048)
        \\  -ub, --ubatch-size <n>      Batch físico por llamada GPU (def: 512)
        \\  --seed <n>                  Semilla determinista (def: 42)
        \\
    , .{});
}

/// Format prompt using GGUF chat template when -jinja is active.
/// Returns allocated string that caller must free. If jinja is off or no
/// template exists, returns the raw prompt duplicated for the caller to free.
/// Inferencia autoregresiva end-to-end con un modelo GGUF.
fn runInference(
    io: std.Io,
    allocator: std.mem.Allocator,
    model_path: []const u8,
    params: CliParams,
    backend: matmul.Backend,
    stdout: anytype,
) !void {
    const t_total = @import("time").Timer.start();
    try stdout.print("[+] Cargando modelo GGUF: {s}\n", .{model_path});
    try stdout.flush();
    var model = try gguf_model.GgufModel.load(io, allocator, model_path);
    const t_model_loaded = t_total.read();
    defer model.deinit();
    const cfg = model.config;

    // --n-gpu-layers/-ngl/--gpu-layers (semántica llama.cpp): este motor no
    // soporta offload parcial de capas — el path híbrido offloadea todas las
    // capas a GPU (requerido) y el legacy corre en CPU. Por tanto:
    //   -ngl 0    → CPU (legacy); en híbrido se advierte y se usa GPU.
    //   -ngl > 0  → GPU con todas las capas offloadadas (si hay CUDA).
    //   sin -ngl  → auto (GPU si disponible).
    var eff_backend = backend;
    if (params.n_gpu_layers) |ngl| {
        if (ngl == 0) {
            if (cfg.is_hybrid) {
                try stdout.print("[!] Modelo híbrido requiere offload GPU: ignorando -ngl 0 (todas las capas a GPU)\n", .{});
                eff_backend = .cublas;
            } else {
                try stdout.print("[+] -ngl 0: offload desactivado, inferencia CPU\n", .{});
                eff_backend = .parallel;
            }
        } else {
            if (@import("cudaz").isCudaAvailable()) {
                try stdout.print("[+] -ngl {d}: offload completo de {d} capas a GPU\n", .{ ngl, cfg.block_count });
                eff_backend = .cublas;
            } else {
                try stdout.print("[!] -ngl {d}: CUDA no disponible, usando CPU\n", .{ngl});
                eff_backend = .parallel;
            }
        }
    }
    if (eff_backend != backend) {
        try stdout.print("Backend efectivo: {s}\n", .{backendName(eff_backend)});
    }
    // 2.2 (lane-f): routing FP8 end-to-end en proyecciones. `--quant fp8`
    // activa matmul.fp8_route_enabled: TODAS las linearProjection* device
    // (attn qkv/attn_out, ssm qkv/z/out, ffn) van por block-scaled E4M3 con
    // engine FP8 lazy + pesos cuantizados on-demand. El backend SIGUE
    // cublas (v1 con backend .fp8_block global rompía las capas de
    // atención — cada una crea su engine y perdía el KV path).
    // Los kernels qgemm (q4_0/q6_k...) se APAGAN: doble cuantización.
    if (params.quant == .fp8) {
        if (cudaz.isCudaAvailable()) {
            matmul.fp8_route_enabled = true;
            try stdout.print("[+] --quant fp8: routing FP8 block-scaled E4M3 en proyecciones (attn/ssm/ffn)\n", .{});
        } else {
            try stdout.print("[!] --quant fp8 sin CUDA: ignorado\n", .{});
        }
    }

    // For quantized KV cache with hybrid models, GPU PagedAttention doesn't support
    // quantized KV cache yet, but we handle this in runHybridInference by forcing fp16 KV.

    // Effective context length: 0 = model's trained context, otherwise user override
    const max_seq_len = if (params.context_length == 0) cfg.context_length else params.context_length;
    if (max_seq_len != cfg.context_length) {
        try stdout.print("[+] Contexto override: {d} (modelo entrenado: {d})\n", .{ max_seq_len, cfg.context_length });
        if (max_seq_len > cfg.context_length) {
            try stdout.print("[!] Contexto mayor que entrenado ({d} > {d}) — extrapolando RoPE\n", .{ max_seq_len, cfg.context_length });
        }
    }

    // Detección automática del path según la arquitectura GGUF:
    //   - híbrido (qwen35/qwen35moe, cfg.is_hybrid) → path paged/híbrido
    //     (PagedKVCache + Scheduler + HybridLayer).
    //   - MoE denso (qwen3moe/qwen2moe/gemma4/mixtral, isMoeModel) → path
    //     híbrido TAMBIÉN (tickets F/C): el TransformerLayer legacy no sabe
    //     de FFN 3-D ffn_*_exps (WeightFileNotFound); el HybridLayer enruta
    //     capas full-attention estándar + MoeLayer attach en el FFN.
    //   - clásico (llama/gemma/mistral, ...) → path legacy contiguo
    //     (TransformerLayer + KVCacheManager).
    // F/C v1 (opt-in MOE_WIRING=1): path híbrido para MoE denso con attach
    // de MoeLayer. KNOWN-ISSUE documentado: el FFN denso sintético experto-0
    // (q4_k/q6_k del loggenix) crashea en q4Weight H2D dentro del path
    // híbrido (repro con MOE_WIRING_OFF=1 SIN attach — pre-existente del
    // HybridLayer con arch no-qwen35, no del wiring). Investigación en
    // curso; TODO F/C row.
    debugz.dbg.printLevel(.info, "[loader] pre-dispatch: is_hybrid={} backend={s} moe_dense_check...\n", .{ cfg.is_hybrid, params.backend });
    const moe_dense = gguf_moe.isMoeModel(&model.file) and std.c.getenv("MOE_WIRING") != null;
    // U1 (eje §12, lane-d): route llama-like → path híbrido unificado (opt-in
    // ZIG_AI_UNIFIED=1). Con el flag, TODAS las capas van como HybridLayer
    // is_attention=true (isFullAttentionLayer ya devuelve true para
    // !is_hybrid): PagedKVCache + PagedAttention GPU + qgemm cuant-residente
    // + prefill batched n>1 (U2) — el mismo chasis validado por los gates
    // U2-llama capa (rel=2.1e-4) y U2-llama-stack 28 capas (top-1 8/8).
    // Requisitos del chasis (ya en hybrid_attn.zig): attn_q/k_norm ausentes
    // ⇒ ONES identidad (llama), post_attention_norm ⇒ fallback ffn_norm,
    // no_gate autodetect por geometría attn_q (llama ⇒ clásica), mrope
    // sections default {n_rot/2,0,0,0} ≡ applyRoPE NEOX (verificado
    // 5fb64e2). n_kv_head real del tensor w_k (autodetect).
    const unified_route = std.c.getenv("ZIG_AI_UNIFIED") != null and
        !cfg.is_hybrid and !moe_dense and model_config.ModelConfig.isSupportedArch(cfg.architecture);
    if (cfg.is_hybrid or moe_dense or unified_route) {
        if (unified_route)
            try stdout.print("[+] U1: route unificado ZIG_AI_UNIFIED — {s} por path híbrido (paged+qgemm, {d} capas atención)\n", .{ cfg.architecture, cfg.block_count });
        if (moe_dense and !cfg.is_hybrid)
            try stdout.print("[+] MoE denso ({s}): path híbrido con MoeLayer attach (decode, v1)\n", .{cfg.architecture});
        try runHybridInference(io, allocator, &model, model_path, params, eff_backend, stdout);
        return;
    }
    debugz.dbg.printLevel(.info, "[loader] path legacy confirmado — construyendo capas\n", .{});

    try stdout.print("[+] arch={s} capas={d} heads={d} kv={d} emb={d} ffn={d} vocab={d} ctx={d} (path legacy)\n", .{
        cfg.architecture,     cfg.block_count,         cfg.head_count, cfg.head_count_kv,
        cfg.embedding_length, cfg.feed_forward_length, cfg.vocab_size, max_seq_len,
    });

    const head_dim: usize = cfg.embedding_length / cfg.head_count;
    const fa_config = FlashAttentionConfig{
        .N = max_seq_len,
        .d = head_dim,
        .num_heads = cfg.head_count,
        .batch_size = 1,
        .dtype = .f16,
        .causal = true,
    };

    // Pesos raíz
    // 7.1d (wiring lane-f): embedding cuant-residente — la tabla f16
    // [vocab, hidden] (0.79GB en Llama-3.2-3B) NO se materializa;
    // EmbSource.quant dequantiza solo las filas de los tokens
    // on-demand desde el QuantWeight mmap. A/B: ZIG_AI_EMB_F16=1.
    var emb_tbl: ?Tensor(f16) = null;
    defer if (emb_tbl) |*t| t.deinit();
    // QuantWeight zero-copy (apunta al mmap del GgufModel): necesita
    // estabilidad de puntero → variable con storage estable.
    const emb_qw = try model.loadEmbeddingQuant();
    const emb_source: embedding.EmbSource = if (std.c.getenv("ZIG_AI_EMB_F16") == null)
        .{ .quant = &emb_qw }
    else blk: {
        emb_tbl = try model.loadEmbedding();
        break :blk .{ .table = emb_tbl.? };
    };
    // 7.1d lm_head cuant-residente (lane-c): sustituye la tabla f16 densa
    // (0.79GB en Llama-3.2-3B) por bytes q8_0 on-load (loadLmHeadQ80 —
    // −58% del peso, GEMV directo M=1 sin round-trip f32) o el QuantWeight
    // mmap nativo cuando el GGUF ya trae el lm_head cuantizado. A/B del
    // camino clásico: ZIG_AI_LMF16=1 (tabla f16 completa).
    // Auto (espejo del híbrido, LMQ80 ≥512MB): f16/bf16/f32 grande ⇒ q80.
    var lm_head_tbl: ?Tensor(f16) = null;
    defer if (lm_head_tbl) |*t| t.deinit();
    var lm_head_q80_bytes: []u8 = &[_]u8{};
    defer if (lm_head_q80_bytes.len > 0) allocator.free(lm_head_q80_bytes);
    const lm_head_qw = model.loadLmHeadQuant() catch null;
    const lm_head_source: embedding.LmHeadSource = blk: {
        if (std.c.getenv("ZIG_AI_LMF16") != null) {
            lm_head_tbl = try model.loadLmHead();
            break :blk .{ .table = lm_head_tbl.? };
        }
        if (lm_head_qw) |*qw| {
            const is_dense: bool = switch (qw.dtype()) {
                .f16, .bf16, .f32 => true,
                else => false,
            };
            const head_mb = qw.bytes.len / (1024 * 1024);
            if (!is_dense) {
                // GGUF ya cuantizado (q4_0/q6_k/...): bytes mmap directos.
                debugz.dbg.printLevel(.info, "[milestone] lm_head cuant-residente nativo {s} ({d} MB, cero f16)\n", .{ @tagName(qw.dtype()), head_mb });
                break :blk .{ .quant = &lm_head_qw.? };
            }
            if (head_mb >= 512) {
                // Denso grande: re-cuant on-load a q8_0 (camino B6/lane-c,
                // paridad con LMQ80 del path híbrido — auto ≥512MB).
                if (model.loadLmHeadQ80(allocator)) |r| {
                    lm_head_q80_bytes = r.bytes;
                    debugz.dbg.printLevel(.info, "[milestone] lm_head q8_0 on-load ({d} MB vs {d} MB f16)\n", .{ lm_head_q80_bytes.len / (1024 * 1024), head_mb });
                    break :blk .{ .q80 = lm_head_q80_bytes };
                } else |e| {
                    debugz.dbg.printLevel(.info, "[milestone] LMQ80 on-load falló ({any}) → tabla f16\n", .{e});
                }
            }
        }
        lm_head_tbl = try model.loadLmHead();
        break :blk .{ .table = lm_head_tbl.? };
    };
    // 7.2 (lane-f): RMSNorm final del modelo — el path legacy NUNCA lo
    // aplicaba antes del lm_head (el híbrido sí, cli.zig:1979): logits
    // sobre hidden sin normalizar ⇒ argmax garbage ("320 ' ('").
    var output_norm = model.loadOutputNorm() catch null;
    defer if (output_norm) |*on| on.deinit();

    // Capas del transformer
    // 7.1b-B: crear UN solo FA engine compartido entre todas las capas.
    // Antes cada capa creaba el suyo (28 × 8 buffers ≈ 672MB);
    // ahora 1 × 8 buffers ≈ 24MB → ahorro ~648MB.
    var fa_engine = AttentionEngine.init(allocator, fa_config, "cuda/flash_attention.ptx", eff_backend);
    defer fa_engine.deinit();

    var layers = try allocator.alloc(TransformerLayer, cfg.block_count);
    defer allocator.free(layers); // MEJORAS A1: slice nunca liberado (leak-report en cada run)
    for (0..cfg.block_count) |i| {
        layers[i] = try TransformerLayer.init(
            allocator,
            i,
            &fa_engine,
            cfg.embedding_length,
            LayerPrecision{ .compute = .f32, .weights_on_gpu = false, .use_quantized = false },
            cfg.head_count_kv,
            cfg.feed_forward_length,
            params.ubatch_size,
        );
        layers[i].rope_freq_base = cfg.rope_freq_base;
        // lane-kvc P4: BitNet b1.58 usa squared_relu(gate)*up + sub-norms
        // (attn_sub_norm pre-wo, ffn_sub_norm pre-down) en vez de SwiGLU.
        layers[i].is_bitnet = model_config.ModelConfig.isBitnet(cfg.architecture);
        try layers[i].loadWeightsFromGguf(&model.file);
    }
    // Liberar capas ANTES del FA engine (las capas tienen punteros a él).
    defer for (layers) |*l| l.deinit();

    // KV cache manager
    const kv_config = KVCacheConfig.default(
        @intCast(cfg.block_count),
        @intCast(cfg.head_count),
        @intCast(head_dim),
        @intCast(max_seq_len),
    );

    // Apply --cache-type-{k,v} (llama.cpp semantics) as per-layer defaults.
    // fp16 (default) keeps the original store-as-f16 fast path; q8_0/q4_0/q4_1
    // route through quantized encode/decode in KVCacheManager.
    const k_fmt: QuantFormat = params.cache_type_k;
    const v_fmt: QuantFormat = if (params.cache_type_v != params.cache_type_k) params.cache_type_v else k_fmt;
    const layer_cfgs = try allocator.alloc(kvcache.LayerQuantConfig, cfg.block_count);
    // NOTA: sin errdefer — la liberación va SOLO en el defer del scope (856):
    // este errdefer + el defer post-kv_manager provocaban double-free en
    // cualquier error posterior al init (el errdefer dispara y el defer
    // del bloque también libera layer_cfgs vía kv_config_full).
    for (0..cfg.block_count) |l| {
        const kf: QuantFormat = if (k_fmt == .fp16) k_fmt else k_fmt;
        const vf: QuantFormat = if (v_fmt == .fp16) v_fmt else v_fmt;
        layer_cfgs[l] = .{
            .k_format = kf,
            .v_format = vf,
            .k_block_size = if (kf == .fp16) 32 else kf.defaultBlockSize(),
            .v_block_size = if (vf == .fp16) 32 else vf.defaultBlockSize(),
            .quant_threshold = null,
            // 9.1 (lane-c C-1): bits kvarn → store de records por grupo.
            .kvarn_k_bits = params.kvarn_k_bits,
            .kvarn_v_bits = params.kvarn_v_bits,
        };
    }
    const kv_config_full = blk: {
        var c = kv_config;
        c.layer_configs = layer_cfgs;
        c.use_gpu_dequant = params.cache_type_k != .fp16;
        break :blk c;
    };
    var kv_manager = try KVCacheManager.init(allocator, kv_config_full, 256);
    defer {
        if (kv_config_full.layer_configs) |cfgs| allocator.free(cfgs);
        kv_manager.deinit();
    }

    // Pipeline de inferencia
    // 7.1a: configurar LRU de f16 tensors antes de init (las capas se
    // registran en pipeline.init via F16Residency.register).
    transformer.F16Residency.setMaxResident(params.f16_max_resident);
    var pl = pipeline.InferencePipeline.init(io, allocator, layers, &kv_manager, cfg.embedding_length, cfg.vocab_size, fa_config, &fa_engine);
    pl.rms_eps = cfg.layer_norm_rms_epsilon; // 7.2: eps del RMSNorm final

    // Tokenizer
    var gt = try gguf_tokenizer.GgufTokenizer.fromGguf(allocator, &model.file);
    defer gt.deinit();
    var tok = try bpe.BPETokenizer.fromTokenizer(allocator, &gt);
    defer tok.deinit();

    const raw_prompt = params.prompt orelse "Hola";
    const prompt = try formatPrompt(allocator, raw_prompt, params.use_jinja, &gt);
    defer allocator.free(prompt);
    const prompt_ids = try tok.encode(prompt, .{});
    defer allocator.free(prompt_ids);
    try stdout.print("[+] prompt ({d} tokens): {s}\n", .{ prompt_ids.len, prompt });

    // Motor matmul
    if (eff_backend == .cublas) cudaz.ensureCurrent() catch {};
    var engine = try matmul.MatmulEngine.init(allocator, eff_backend, .f32);
    defer engine.deinit();

    const seq_id: u64 = 1;

    const t_prefill_init = t_total.read();
    const prefill_res = blk: {
        if (params.recurrent_prefill)
            break :blk try pl.recurrentPrefill(seq_id, prompt_ids, emb_source, lm_head_source, &engine, output_norm)
        else
            break :blk try pl.prefill(seq_id, prompt_ids, emb_source, lm_head_source, &engine, output_norm);
    };
    const first_token = prefill_res.last_token;
    // 7.2 (lane-f): paridad greedy del prefill — top-5 logits + logprobs
    // vs golden llama.cpp (/v1/completions, ctx 512, temp 0, mismo GGUF:
    // " Paris"=12366 lp=-0.209, " the"=279 -4.02, "..."=1131 -4.11,
    // " not"=539 -4.18, " a"=264 -4.19). Gated DEBUG_LEVEL=2, tag [pipeline].
    if (debugz.dbg.at(.detail)) {
        const piece = tok.decode(&[_]u32{first_token}, allocator) catch "";
        defer if (piece.len > 0) allocator.free(piece);
        var top: [5]struct { idx: u32, val: f32 } = undefined;
        for (&top) |*t| t.* = .{ .idx = 0, .val = -std.math.inf(f32) };
        for (prefill_res.logits.data, 0..) |v, i| {
            const fv = @as(f32, @floatCast(v));
            for (&top, 0..) |*t, ti| {
                if (fv > t.val) {
                    var k: usize = 4;
                    while (k > ti) : (k -= 1) top[k] = top[k - 1];
                    t.* = .{ .idx = @intCast(i), .val = fv };
                    break;
                }
            }
        }
        debugz.dbg.printLevel(.detail, "[pipeline] first_token={d} '{s}'\n", .{ first_token, piece });
        for (top) |t| debugz.dbg.printLevel(.detail, "[pipeline]   tok={d} logit={d:.3}\n", .{ t.idx, t.val });
        var max_l: f32 = -std.math.inf(f32);
        for (prefill_res.logits.data) |v| {
            const fv = @as(f32, @floatCast(v));
            if (fv > max_l) max_l = fv;
        }
        var sum_exp: f64 = 0;
        for (prefill_res.logits.data) |v| {
            sum_exp += @exp(@as(f64, @as(f32, @floatCast(v)) - max_l));
        }
        const lse = max_l + @as(f32, @floatCast(@log(sum_exp)));
        for ([_]u32{ 12366, 279, 1131, 539, 264, 320 }) |g| {
            debugz.dbg.printLevel(.detail, "[pipeline] golden tok={d} logit={d:.3} logprob={d:.3}\n", .{ g, @as(f32, @floatCast(prefill_res.logits.data[g])), @as(f32, @floatCast(prefill_res.logits.data[g])) - lse });
        }
    }
    var prefill_logits = prefill_res.logits;
    prefill_logits.deinit(); // 7.2: ownership devuelto por prefill

    const gen_config = pipeline.GenerationConfig{
        .max_new_tokens = params.max_new_tokens,
        .sampler = params.sampler,
        .eos_token = gt.eos_id,
        .pad_token = null,
        .stop_on_eos = true,
        .seed = params.seed,
    };

    // KT-B (lane-f): KV-transfer opt-in. Con --kv-transfer <pesos.ktb> el
    // prefill recién hecho (seq SOURCE=1) se transfiere a la seq TARGET=2
    // (retrieve → strip RoPE → mapper por capa → re-rope → append
    // cuantizado), y la generación continúa sobre el TARGET. Modo
    // source==target (mismo manager, dos secuencias — patrón MTP-draft):
    // con mapper identity reproduce el prefill (gate-0 CLI). El dual-load
    // cross-model sigue el patrón SidecarDraft cuando KT-A calibre pesos.
    var kt_weights: ?kvcache.kt_transfer.KtWeights = null;
    defer if (kt_weights) |*w| w.deinit();
    const kt_path = params.kv_transfer_path orelse blk: {
        const env = std.c.getenv("ZIG_AI_KV_TRANSFER");
        if (env) |e| break :blk std.mem.span(e);
        break :blk null;
    };
    var generate_seq: u64 = seq_id;
    if (kt_path) |path| {
        try stdout.print("[+] KT-B: kv-transfer desde {s}\n", .{path});
        kt_weights = kvcache.kt_transfer.KtWeights.load(
            allocator,
            io,
            path,
            @intCast(cfg.block_count),
            @intCast(cfg.head_count_kv),
            @intCast(cfg.head_count_kv),
            @intCast(head_dim),
        ) catch |e| {
            try stdout.print("[!] KT-B: fallo cargando .ktb: {s}\n", .{@errorName(e)});
            return e;
        };
        const seq_target: u64 = 2;
        try kv_manager.createSequence(seq_target);
        var kt_rt = kvcache.kt_transfer.KtRuntime{
            .allocator = allocator,
            .weights = &kt_weights.?,
            .seq_source = seq_id,
            .seq_target = seq_target,
        };
        const pairing: kvcache.kt_transfer.RopePairing = .norm;
        try kt_rt.transfer(
            .{ .manager = &kv_manager, .rope_base = cfg.rope_freq_base, .pairing = pairing },
            .{ .manager = &kv_manager, .rope_base = cfg.rope_freq_base, .pairing = pairing },
            &engine,
            prompt_ids.len,
        );
        try stdout.print("[+] KT-B: {d} tokens transferidos seq {d}→{d} (mapper {s})\n", .{ prompt_ids.len, seq_id, seq_target, "ok" });
        generate_seq = seq_target;
    }

    const result = try pl.generate(generate_seq, first_token, emb_source, lm_head_source, &engine, gen_config, output_norm);
    defer allocator.free(result.tokens);

    const t_end = t_total.read();
    const total_ms = @as(f64, @floatFromInt(@divTrunc(t_end, std.time.ns_per_ms)));
    const model_ms = @as(f64, @floatFromInt(@divTrunc(t_model_loaded, std.time.ns_per_ms)));
    const init_ms = @as(f64, @floatFromInt(@divTrunc(t_prefill_init - t_model_loaded, std.time.ns_per_ms)));
    const prefill_ms = result.prefill_time_ms;
    const gen_ms = result.generation_time_ms;
    const per_token_ms = if (result.num_tokens_generated > 0) gen_ms / @as(f64, @floatFromInt(result.num_tokens_generated)) else 0;

    try stdout.print("\n[+] Generación ({d} tokens, {d:.1} tok/s):\n", .{ result.num_tokens_generated, result.tokens_per_second });
    const decoded = try tok.decode(result.tokens, allocator);
    defer allocator.free(decoded);
    try stdout.print("{s}\n", .{decoded});
    try stdout.print("\n[+] Métricas detalladas:\n", .{});
    try stdout.print("  modelo    {d:.0} ms\n", .{model_ms});
    try stdout.print("  init      {d:.0} ms (capas+kv+weights)\n", .{init_ms});
    try stdout.print("  prefill   {d:.1} ms ({d} tok, {d:.1} ms/tok)\n", .{ prefill_ms, prompt_ids.len, if (prompt_ids.len > 0) prefill_ms / @as(f64, @floatFromInt(prompt_ids.len)) else 0 });
    try stdout.print("  decode    {d:.1} ms ({d} tok, {d:.2} ms/tok)\n", .{ gen_ms, result.num_tokens_generated, per_token_ms });
    try stdout.print("  total     {d:.0} ms\n", .{total_ms});
    try stdout.print("  throughput {d:.1} tok/s\n", .{result.tokens_per_second});
    try stdout.print("\n=================================================\n", .{});
    try stdout.print("              Ejecucion completada               \n", .{});
    try stdout.flush();

    layer_kernels.deinitQ4Cache();
}

/// lane-kvc tANS C-a — modo perplexity (semántica llama.cpp --perplexity):
/// tokeniza el archivo, prefill por chunks con sliding window (2048/1024),
/// PPL = exp(Σ log p(t_i | t_<i) / N). Reusa el pipeline de prefill legacy
/// (prefillPPL: logits por posición) sin sampling. El BOS policy sigue el
/// path legacy (llama3-like añade BOS).
fn runPpl(
    io: std.Io,
    allocator: std.mem.Allocator,
    model_path: []const u8,
    ppl_path: []const u8,
    params: CliParams,
    backend: matmul.Backend,
    stdout: anytype,
) !void {
    try stdout.print("[+] Cargando modelo GGUF: {s}\n", .{model_path});
    try stdout.flush();
    var model = try gguf_model.GgufModel.load(io, allocator, model_path);
    defer model.deinit();
    const cfg = model.config;

    var eff_backend = backend;
    if (params.n_gpu_layers) |ngl| {
        if (ngl == 0 and !cfg.is_hybrid) eff_backend = .parallel;
    }

    const max_seq_len = if (params.context_length == 0) cfg.context_length else params.context_length;

    // P0-1 (dev RLT): híbrido (Qwen3.5/LFM2.5) → PPL GPU batcheado vía
    // forwardGPU + paged KV. El path legacy (CPU) cubre llama/bitnet/mistral.
    const moe_dense = gguf_moe.isMoeModel(&model.file) and std.c.getenv("MOE_WIRING") != null;
    if (cfg.is_hybrid or moe_dense) {
        return inference.runHybridPpl(io, allocator, &model, params, backend, stdout);
    }

    try stdout.print("[+] ppl: arch={s} capas={d} ctx={d} (path legacy)\n", .{ cfg.architecture, cfg.block_count, max_seq_len });

    const head_dim: usize = cfg.embedding_length / cfg.head_count;
    const fa_config = FlashAttentionConfig{
        .N = max_seq_len,
        .d = head_dim,
        .num_heads = cfg.head_count,
        .batch_size = 1,
        .dtype = .f16,
        .causal = true,
    };

    // lane-kvc tANS C-a: ventana/stride del sliding window se calculan AQUÍ
    // (antes de init de capas) porque act_capacity (7.1b lane-d) debe cubrir
    // el chunk más grande que vamos a reenviar: si fuera ubatch_size (512)
    // los buffers de activación se quedarían cortos al forwardear chunks de
    // `window` tokens ⇒ overflow. Semántica llama.cpp: window/2.
    const window: usize = @min(@as(usize, 2048), max_seq_len);
    const stride: usize = @max(@as(usize, 1), window / 2);

    var emb = try model.loadEmbedding();
    defer emb.deinit();
    var lm_head = try model.loadLmHead();
    defer lm_head.deinit();
    var output_norm = model.loadOutputNorm() catch null;
    defer if (output_norm) |*on| on.deinit();

    var layers = try allocator.alloc(TransformerLayer, cfg.block_count);
    defer allocator.free(layers); // MEJORAS A1: idem path PPL
    defer for (layers) |*l| l.deinit();

    // 7.1b-B: FA engine compartido para PPL (idéntico al path inference).
    var fa_engine = AttentionEngine.init(allocator, fa_config, "cuda/flash_attention.ptx", eff_backend);
    defer fa_engine.deinit();

    // R-4 sidecar: solo para path híbrido (runHybridPpl); runPpl legacy no lo usa

    for (0..cfg.block_count) |i| {
        layers[i] = try TransformerLayer.init(
            allocator,
            i,
            &fa_engine,
            cfg.embedding_length,
            LayerPrecision{ .compute = .f32, .weights_on_gpu = false, .use_quantized = false },
            cfg.head_count_kv,
            cfg.feed_forward_length,
            window,
        );
        layers[i].rope_freq_base = cfg.rope_freq_base;
        layers[i].is_bitnet = model_config.ModelConfig.isBitnet(cfg.architecture);
        try layers[i].loadWeightsFromGguf(&model.file);
    }

    const k_fmt: QuantFormat = params.cache_type_k;
    const v_fmt: QuantFormat = if (params.cache_type_v != params.cache_type_k) params.cache_type_v else k_fmt;
    const layer_cfgs = try allocator.alloc(kvcache.LayerQuantConfig, cfg.block_count);
    defer allocator.free(layer_cfgs);
    for (layer_cfgs) |*lc| {
        lc.* = .{
            .k_format = k_fmt,
            .v_format = v_fmt,
            .k_block_size = if (k_fmt == .fp16) 32 else k_fmt.defaultBlockSize(),
            .v_block_size = if (v_fmt == .fp16) 32 else v_fmt.defaultBlockSize(),
            .quant_threshold = null,
        };
    }
    const kv_config = blk: {
        // 7.1-remanente (lane-c): el pool KV legacy dimensionaba cada slot
        // a max_seq_len COMPLETO (default 65536 ⇒ ~2GB host con 16 capas
        // fp16 1B — OOM en --ppl sin --ctx-size, hallazgo 11.6 del lane 1x).
        // El PPL usa sliding window (retención ≤ window, chunk ≤ window):
        // capacitar el KV a window+2 (BOS margen) basta y reduce el KV de
        // ctx-completo a ~2k en cualquier modelo. `--ctx-size` sigue
        // mandando si el usuario lo pasa (max_seq_len ya reducido arriba).
        const kv_len: usize = @min(@as(usize, max_seq_len), window + 2);
        var c = KVCacheConfig.default(
            @intCast(cfg.block_count),
            @intCast(cfg.head_count),
            @intCast(head_dim),
            @intCast(kv_len),
        );
        c.layer_configs = layer_cfgs;
        c.use_gpu_dequant = k_fmt != .fp16;
        break :blk c;
    };
    var kv_manager = try KVCacheManager.init(allocator, kv_config, 256);
    defer kv_manager.deinit();

    var pl = pipeline.InferencePipeline.init(io, allocator, layers, &kv_manager, cfg.embedding_length, cfg.vocab_size, fa_config, &fa_engine);
    pl.rms_eps = cfg.layer_norm_rms_epsilon;

    var gt = try gguf_tokenizer.GgufTokenizer.fromGguf(allocator, &model.file);
    defer gt.deinit();
    var tok = try bpe.BPETokenizer.fromTokenizer(allocator, &gt);
    defer tok.deinit();

    if (eff_backend == .cublas) cudaz.ensureCurrent() catch {};
    var engine = try matmul.MatmulEngine.init(allocator, eff_backend, .f32);
    defer engine.deinit();

    // Leer el corpus completo (stat + readPositionalAll, patrón del repo)
    var ppl_file = try std.Io.Dir.cwd().openFile(io, ppl_path, .{ .mode = .read_only });
    defer ppl_file.close(io);
    const fstat = try ppl_file.stat(io);
    const fdata = try allocator.alloc(u8, @intCast(fstat.size));
    defer allocator.free(fdata);
    _ = try ppl_file.readPositionalAll(io, fdata, 0);

    const token_ids = try tok.encode(fdata, .{});
    defer allocator.free(token_ids);
    try stdout.print("[+] ppl: {s} = {d} bytes, {d} tokens (kv {s}/{s})\n", .{
        ppl_path,        fdata.len,       token_ids.len,
        @tagName(k_fmt), @tagName(v_fmt),
    });
    try stdout.flush();
    if (token_ids.len < 2) return error.NoTokensToScore;

    // Sliding window estilo llama.cpp: cada token se puntúa en EXACTAMENTE un
    // chunk, condicionado a hasta (window - stride) tokens previos de contexto.
    // (window/stride ya calculados arriba junto a act_capacity.)
    //
    // Invariantes del bucle (bugs B1-B4 corregidos en lane-kvc):
    //  - ctx_start ≤ pos-1  ⇒ first_row = pos-ctx_start-1 siempre ≥ 0
    //  - chunk_end ≤ ctx_start+window ⇒ el chunk nunca excede la ventana
    //  - scored_in_chunk = chunk_end-pos ⇒ el último token SÍ se puntúa
    //  - fila que predice el token t es (t - ctx_start - 1) ⇒ first_row + i
    var nll_sum: f64 = 0;
    var n_scored: usize = 0;
    var pos: usize = 1; // chunk 0: el token 0 no tiene contexto — se salta
    var chunk_idx: usize = 0;
    const timer = @import("time").Timer.now();

    while (pos < token_ids.len) {
        const ctx_start = @min(pos - 1, pos -| (window - stride));
        const chunk_end = @min(ctx_start + window, token_ids.len);
        const chunk_len = chunk_end - ctx_start;
        const scored_in_chunk = chunk_end - pos;

        const seq_id: u64 = 100 + chunk_idx;
        const logits = try pl.prefillPPL(
            seq_id,
            token_ids[ctx_start..chunk_end],
            emb,
            lm_head,
            &engine,
            output_norm,
        );
        var logits_mut = logits;
        defer logits_mut.deinit();

        // La fila r del chunk predice el token (ctx_start + r + 1) ⇒ el token
        // t lo predice la fila (t - ctx_start - 1). first_row ya es la fila
        // que predice el token `pos` (B1: NO restar 1 otra vez — eso leía la
        // fila del token anterior y en el chunk 0 daba índice −1).
        const first_row: usize = pos - ctx_start - 1;
        const vocab = cfg.vocab_size;
        // lane-kvc diagnóstico lane-f: curva NLL-vs-posición
        const per_token_dbg = debugz.dbg.at(.trace);
        var i: usize = 0;
        while (i < scored_in_chunk) : (i += 1) {
            const row = logits.data[(first_row + i) * vocab ..][0..vocab];
            const target = token_ids[pos + i];
            var max_l: f32 = -std.math.inf(f32);
            for (row) |v| {
                const fv: f32 = @floatCast(v);
                if (fv > max_l) max_l = fv;
            }
            var sum_exp: f64 = 0;
            for (row) |v| {
                sum_exp += @exp(@as(f64, @as(f32, @floatCast(v))) - max_l);
            }
            const lse: f64 = @as(f64, max_l) + @log(sum_exp);
            const tok_logit: f32 = @floatCast(row[target]);
            const tok_nll = lse - @as(f64, tok_logit);
            nll_sum += tok_nll;
            n_scored += 1;
            if (per_token_dbg) {
                debugz.dbg.printLevel(.trace, "[ppl] tok pos={d} nll={d:.4} target={d}\n", .{ pos + i, tok_nll, target });
            }
        }

        kv_manager.removeSequence(seq_id);
        chunk_idx += 1;
        pos += scored_in_chunk;
        if (debugz.dbg.at(.info)) {
            debugz.dbg.printLevel(.info, "[ppl] chunk {d}: ctx_start={d} chunk_end={d} len={d} first_row={d} scored={d} nll_acc={d:.4}\n", .{ chunk_idx - 1, ctx_start, chunk_end, chunk_len, first_row, scored_in_chunk, nll_sum });
        }
    }

    if (n_scored == 0) return error.NoTokensToScore;
    const ppl = @exp(nll_sum / @as(f64, @floatFromInt(n_scored)));
    const elapsed_s = @as(f64, @floatFromInt(@import("time").Timer.now() - timer)) / 1e9;
    try stdout.print("[+] ppl: {d} chunks, {d} tokens scored en {d:.1}s\n", .{ chunk_idx, n_scored, elapsed_s });
    try stdout.print("[+] PPL ({s}/{s} KV): {d:.4}\n", .{ @tagName(k_fmt), @tagName(v_fmt), ppl });
    try stdout.flush();
}

fn runGpuMode(allocator: std.mem.Allocator, config: FlashAttentionConfig, stdout: anytype) !void {
    try stdout.print("\n[+] Inicializando TransformerLayer con GPU...\n", .{});

    const precision = LayerPrecision{
        .compute = .f16,
        .weights_on_gpu = false,
        .use_quantized = false,
    };

    var test_fa_engine = AttentionEngine.init(allocator, config, "cuda/flash_attention.ptx", null);
    defer test_fa_engine.deinit();
    var layer = try TransformerLayer.init(allocator, 0, &test_fa_engine, 1024, precision, config.num_heads, 4096, 512);
    defer layer.deinit();

    try stdout.print("[+] Capa transformer inicializada\n", .{});
    try stdout.print("[+] Backend matmul: {s}\n", .{layer.matmul_engine.backendName()});

    var hidden_state = try Tensor(f32).alloc(allocator, &.{ config.batch_size, config.N, 1024 });
    defer hidden_state.deinit();
    var rng = std.Random.Xoshiro256.init(42);
    hidden_state.randUniform(&rng, -0.1, 0.1);

    var output = try Tensor(f32).alloc(allocator, &.{ config.batch_size, config.N, 1024 });
    defer output.deinit();

    try stdout.print("[*] Warmup...\n", .{});
    try layer.forward(hidden_state, &output, 0, true);

    const iterations: usize = 10;
    const timer = @import("time").Timer.start();
    for (0..iterations) |_| {
        try layer.forward(hidden_state, &output, 0, true);
    }
    const total_ns = timer.read();
    const avg_ms = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(iterations)) / 1_000_000.0;

    try stdout.print("\n[GPU] {d} iteraciones promedio: {d:.3} ms\n", .{ iterations, avg_ms });

    if (layer.matmul_engine.gpuPoolStats()) |stats| {
        try stdout.print("[GPU] Pool: {d} total, {d} usado, {d} libre\n", .{ stats.total, stats.used, stats.free });
    }

    try stdout.print("\n=================================================\n", .{});
    try stdout.print("              Ejecucion completada               \n", .{});
    try stdout.print("=================================================\n", .{});
    try stdout.flush();
}

fn runCpuMode(allocator: std.mem.Allocator, config: FlashAttentionConfig, stdout: anytype) !void {
    var fa_cpu = FlashAttentionCpu.init(allocator, config);

    var Q = try Tensor(f16).alloc(allocator, &.{ 1, 8, 512, 128 });
    defer Q.deinit();
    var K = try Tensor(f16).alloc(allocator, &.{ 1, 8, 512, 128 });
    defer K.deinit();
    var V = try Tensor(f16).alloc(allocator, &.{ 1, 8, 512, 128 });
    defer V.deinit();
    var O = try Tensor(f16).alloc(allocator, &.{ 1, 8, 512, 128 });
    defer O.deinit();

    fa.fa_utils.initUniform(&Q, -0.1, 0.1, 42);
    fa.fa_utils.initUniform(&K, -0.1, 0.1, 43);
    fa.fa_utils.initUniform(&V, -0.1, 0.1, 44);

    const timer = @import("time").Timer.start();
    try fa_cpu.forward(Q, K, V, &O);
    const elapsed = timer.read();

    try stdout.print("[CPU] Forward completado en {d:.2} ms\n", .{@as(f64, @floatFromInt(elapsed)) / 1_000_000.0});
    try stdout.print("[CPU] Output sample: ", .{});
    O.printHead(10);
    try stdout.flush();
}
