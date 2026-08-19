const std = @import("std");
const Tensor = @import("core").Tensor;
const matmul = @import("matmul");
const fa = @import("fa");
const transformer = @import("transformer");
const kvcache = @import("kv_cache");

const FlashAttention = fa.FlashAttention;
const FlashAttentionCpu = fa.FlashAttentionCpu;
const FlashAttentionConfig = fa.fa_config.FlashAttentionConfig;
const TransformerLayer = transformer.TransformerLayer;
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
const gguf = @import("gguf");
const embedding = @import("embedding");
const hybrid_layer = @import("transformer");
const norm = @import("norm");
const paged_attn = @import("paged_attention");
const decode_graph = @import("decode_graph");
const layer_streamer = @import("layer_streamer");
const vram_budget = @import("vram_budget");
const debugz = @import("debug");

/// Parámetros de runtime parseados de la línea de comandos.
const CliParams = struct {
    model_path: ?[]const u8 = null,
    prompt: ?[]const u8 = null,
    max_new_tokens: usize = 128,
    seed: u64 = 42,
    backend: []const u8 = "auto", // auto | cpu | gpu
    sampler: pipeline.Sampler = .{},
    // --- KV cache quantization (llama.cpp --cache-type-{k,v}) ---
    cache_type_k: QuantFormat = .fp16,
    cache_type_v: QuantFormat = .fp16,
    // --- Draft KV cache quantization (llama.cpp --spec-draft-type-{k,v}) ---
    spec_draft_type_k: QuantFormat = .fp16,
    spec_draft_type_v: QuantFormat = .fp16,
    // --- Parallel sequences (llama.cpp -np) ---
    num_parallel: usize = 1,
    // --- Spec-decoding mode (llama.cpp --spec-type) ---
    spec_type: SpecType = .none,
    /// Máximo número de tokens a generar en el draft por round (llama.cpp --spec-draft-n-max; def 16)
    spec_draft_n_max: usize = 16,
    // --- Chat template (llama.cpp -jinja) ---
    use_jinja: bool = false,
    // --- Cuantización de pesos (--quant auto|off; activa el GEMM Q4_0 device) ---
    quant: QuantMode = .auto,
    // --- Batch de prefill (llama.cpp -b/--batch-size lógico, -ub/--ubatch-size
    // físico). El prompt se procesa en lotes lógicos de `batch_size` y cada uno
    // en sub-lotes físicos de `ubatch_size` tokens por llamada GPU.
    batch_size: usize = 2048,
    ubatch_size: usize = 512,
    // --- Offload de capas a GPU (llama.cpp -ngl/--n-gpu-layers/--gpu-layers).
    // null = auto (GPU si disponible); 0 = CPU; >0 = offload completo a GPU
    // (este motor no soporta offload parcial).
    n_gpu_layers: ?usize = null,
    /// AirLLM layer streaming: async prefetch + LRU eviction
    layer_stream: bool = false,
    layer_stream_max: usize = 2,
};

const SpecType = enum { none, draft_mtp };

const QuantMode = enum { auto, off };

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
        .auto => "auto",
    };
}

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

    var stdout_buffer: [0x200]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;

    const params = try parseArgs(allocator, init.minimal.args, stdout);

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

    if (params.model_path) |path| {
        if (backend == .cublas) {
            try @import("cudaz").ensureContext();
            try printGpuInfo(allocator, stdout);
        }
        runInference(io, allocator, path, params, backend, stdout) catch |err| {
            if (@errorReturnTrace()) |trace| {
                std.debug.print("TRACE: {s}\n", .{@errorName(err)});
                std.debug.dumpStackTrace(trace);
            }
            return err;
        };
        return;
    }

    // Sin --model: demo de benchmark (CPU o GPU)
    const config = FlashAttentionConfig{
        .N = 512, .d = 128, .num_heads = 8, .batch_size = 1,
        .dtype = .f16, .causal = true,
    };

    try stdout.print("Config: N={d}, d={d}, heads={d}, batch={d}, causal={}\n", .{
        config.N, config.d, config.num_heads, config.batch_size, config.causal,
    });

    const cuda_available = @import("cudaz").isCudaAvailable();
    try stdout.print("CUDA disponible: {}\n", .{cuda_available});

    if (!cuda_available) {
        try stdout.print("\n[!] CUDA no disponible. Ejecutando modo CPU...\n", .{});
        try runCpuMode(allocator, config, stdout);
        return;
    }

    try runGpuMode(allocator, config, stdout);
}

/// Parsea los argumentos de línea de comandos.
fn parseArgs(allocator: std.mem.Allocator, args: std.process.Args, stdout: anytype) !CliParams {
    var params: CliParams = .{};
    var it = std.process.Args.Iterator.init(args);
    // Saltar argv[0]
    _ = it.next();

    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--model") or std.mem.eql(u8, arg, "-m")) {
            params.model_path = try nextValue(&it, arg);
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            params.prompt = try nextValue(&it, "--prompt");
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
                try stdout.print("[!] Backend inválido: {s} (auto|cpu|gpu)\n", .{v});
            }
        } else if (std.mem.eql(u8, arg, "-ctk") or std.mem.eql(u8, arg, "--cache-type-k")) {
            params.cache_type_k = try parseQuant(&it, arg);
        } else if (std.mem.eql(u8, arg, "-ctv") or std.mem.eql(u8, arg, "--cache-type-v")) {
            params.cache_type_v = try parseQuant(&it, arg);
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
            } else {
                try stdout.print("[!] --spec-type inválido: {s} (none|draft-mtp)\n", .{v});
            }
        } else if (std.mem.eql(u8, arg, "--spec-draft-n-max")) {
            const v = try nextValue(&it, "--spec-draft-n-max");
            params.spec_draft_n_max = std.fmt.parseInt(usize, v, 10) catch 16;
        } else if (std.mem.eql(u8, arg, "--quant")) {
            const v = try nextValue(&it, "--quant");
            if (std.mem.eql(u8, v, "off")) {
                params.quant = .off;
            } else if (std.mem.eql(u8, v, "auto")) {
                params.quant = .auto;
            } else {
                try stdout.print("[!] --quant inválido: {s} (auto|off)\n", .{v});
            }
        } else if (std.mem.eql(u8, arg, "--layer-stream")) {
            params.layer_stream = true;
        } else if (std.mem.eql(u8, arg, "--layer-stream-max")) {
            const v = try nextValue(&it, "--layer-stream-max");
            params.layer_stream_max = std.fmt.parseInt(usize, v, 10) catch 2;
        } else if (std.mem.eql(u8, arg, "-ngl") or std.mem.eql(u8, arg, "--n-gpu-layers") or std.mem.eql(u8, arg, "--gpu-layers")) {
            const v = try nextValue(&it, "--n-gpu-layers");
            params.n_gpu_layers = std.fmt.parseInt(usize, v, 10) catch 0;
        } else if (std.mem.eql(u8, arg, "-jinja") or std.mem.eql(u8, arg, "--jinja")) {
            params.use_jinja = true;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            try printHelp(stdout);
            std.process.exit(0);
        } else {
            try stdout.print("[!] Argumento desconocido: {s}\n", .{arg});
        }
    }
    _ = allocator;
    return params;
}

fn nextValue(it: *std.process.Args.Iterator, name: []const u8) ![]const u8 {
    return it.next() orelse {
        std.debug.print("Falta valor para {s}\n", .{name});
        return error.MissingArgumentValue;
    };
}

fn parseQuant(it: *std.process.Args.Iterator, flag: []const u8) !QuantFormat {
    const v = try nextValue(it, flag);
    if (QuantFormat.fromString(v)) |q| return q;
    std.debug.print("[!] Formato de cuantización inválido para {s}: {s}\n", .{ flag, v });
    return error.InvalidQuantFormat;
}

fn printHelp(stdout: anytype) !void {
    try stdout.print(
        \\Uso: zig-ai-engine [opciones]
        \\  -m, --model <ruta>          Ruta a un modelo GGUF (activa inferencia)
        \\  --prompt <texto>            Prompt de entrada
        \\  -n, --max-tokens <n>        Máximo de tokens a generar (def: 128)
        \\  -ngl, --n-gpu-layers <n>    Capas a offload a GPU (def: auto; 0 = CPU,
        \\                              >0 = todas las capas a GPU)
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
        \\  --spec-type <draft-mtp>     Decodificación especulativa (def: none)
        \\  --spec-draft-n-max <n>      Tokens draft por round (def: 16)
        \\  --quant <auto|off>          GEMM de pesos cuantizados Q4_0 (def: auto)
        \\  --layer-stream             Activar layer streaming (prefetch async + LRU)
        \\  --layer-stream-max <n>      Max capas residentes en VRAM (def: 2)
        \\  -jinja                     Usar plantilla chat jinja del tokenizer
        \\  -h, --help                  Muestra esta ayuda
        \\
    , .{});
}

/// Captura la secuencia GPU completa de un token de decode (embed H2D + capas
/// híbridas + rmsNorm final + lm_head) en un CUDA graph. Devuelve `false` en
/// cualquier error (auto-fallback al camino normal); en `true` el grafo queda
/// instanciado en `g.exec` y el estado ssm restaurado. La captura SIEMPRE se
/// termina (o aborta) aunque un nodo falle, para que el stream vuelva a modo
/// normal y el fallback pueda lanzar async ops.
fn captureDecodeGraph(
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
    lm_head_q: anytype,
    lm_head: anytype,
) bool {
    g.backupState(state_parts) catch return false;
    g.beginCapture() catch return false;

    var ok = true;
    cudaz.cuMemcpyHtoDAsync(g_cur.*.ptr(), @intFromPtr(g.embed_staging.ptr), n_embd * @sizeOf(f32), lk.stream) catch {
        std.debug.print("CAPTURE FAIL: embed H2D: {s}\n", .{@errorName(error.CudaError)});
        ok = false;
    };
    var c2g = g_cur.*;
    var n2g = g_nxt.*;
    if (ok) {
        for (layers, 0..) |*layer, i| {
            hybrid_layer.HybridLayer.forwardGPU(layer, lk, c2g, &n2g, current_pos, 1) catch |e| {
                std.debug.print("CAPTURE FAIL: layer {d}: {s}\n", .{ i, @errorName(e) });
                ok = false;
            };
            const tmp = c2g;
            c2g = n2g;
            n2g = tmp;
        }
    }
    if (ok) {
        lk.rmsNorm(c2g.ptr(), @intFromPtr(g_out_norm.*.dev_ptr), g_normed.*.ptr(), 1, n_embd, rms_eps) catch |e| {
            std.debug.print("CAPTURE FAIL: rmsNorm: {s}\n", .{@errorName(e)});
            ok = false;
        };
        if (ok) {
            if (lm_head_q4) {
                lk.q4gemmLinear(allocator, g_normed.*.ptr(), lm_head_q.bytes, g_logits.*.ptr(), n_embd, vocab) catch |e| {
                    std.debug.print("CAPTURE FAIL: lm_head q4: {s}\n", .{@errorName(e)});
                    ok = false;
                };
            } else if (lm_head_q6k) {
                lk.qgemmLinear(allocator, g_normed.*.ptr(), lm_head_q.bytes, g_logits.*.ptr(), 1, n_embd, vocab, 3) catch |e| {
                    std.debug.print("CAPTURE FAIL: lm_head q6k: {s}\n", .{@errorName(e)});
                    ok = false;
                };
            } else {
                engine.linearProjectionDeviceF16(g_normed.*, lm_head, g_logits, 1, n_embd, vocab) catch |e| {
                    std.debug.print("CAPTURE FAIL: lm_head f16: {s}\n", .{@errorName(e)});
                    ok = false;
                };
            }
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

/// Inferencia autoregresiva end-to-end con un modelo GGUF.
fn runInference(
    io: std.Io,
    allocator: std.mem.Allocator,
    model_path: []const u8,
    params: CliParams,
    backend: matmul.Backend,
    stdout: anytype,
) !void {
    try stdout.print("[+] Cargando modelo GGUF: {s}\n", .{model_path});
    try stdout.flush();
    var model = try gguf_model.GgufModel.load(io, allocator, model_path);
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

    // Detección automática del path según la arquitectura GGUF:
    //   - híbrido (qwen35/qwen35moe, cfg.is_hybrid) → path paged/híbrido
    //     (PagedKVCache + Scheduler + HybridLayer).
    //   - clásico (llama/gemma/mistral, ...) → path legacy contiguo
    //     (TransformerLayer + KVCacheManager).
    if (cfg.is_hybrid) {
        try runHybridInference(allocator, &model, params, eff_backend, stdout);
        return;
    }

    try stdout.print("[+] arch={s} capas={d} heads={d} kv={d} emb={d} ffn={d} vocab={d} ctx={d} (path legacy)\n", .{
        cfg.architecture, cfg.block_count, cfg.head_count, cfg.head_count_kv,
        cfg.embedding_length, cfg.feed_forward_length, cfg.vocab_size, cfg.context_length,
    });

    const head_dim: usize = cfg.embedding_length / cfg.head_count;
    const fa_config = FlashAttentionConfig{
        .N = cfg.context_length, .d = head_dim, .num_heads = cfg.head_count,
        .batch_size = 1, .dtype = .f16, .causal = true,
    };

    // Pesos raíz
    var emb = try model.loadEmbedding();
    defer emb.deinit();
    var lm_head = try model.loadLmHead();
    defer lm_head.deinit();

    // Capas del transformer
    var layers = try allocator.alloc(TransformerLayer, cfg.block_count);
    for (0..cfg.block_count) |i| {
        layers[i] = try TransformerLayer.init(
            allocator, i, fa_config, "", cfg.embedding_length,
            LayerPrecision{ .compute = .f32, .weights_on_gpu = false, .use_quantized = false },
            cfg.head_count_kv, cfg.feed_forward_length,
        );
        layers[i].rope_freq_base = cfg.rope_freq_base;
        try layers[i].loadWeightsFromGguf(&model.file);
    }
    defer for (layers) |*l| l.deinit();

    // KV cache manager
    const kv_config = KVCacheConfig.default(
        @intCast(cfg.block_count), @intCast(cfg.head_count),
        @intCast(head_dim), @intCast(cfg.context_length),
    );

    // Apply --cache-type-{k,v} (llama.cpp semantics) as per-layer defaults.
    // fp16 (default) keeps the original store-as-f16 fast path; q8_0/q4_0/q4_1
    // route through quantized encode/decode in KVCacheManager.
    const k_fmt: QuantFormat = params.cache_type_k;
    const v_fmt: QuantFormat = if (params.cache_type_v != params.cache_type_k) params.cache_type_v else k_fmt;
    const layer_cfgs = try allocator.alloc(kvcache.LayerQuantConfig, cfg.block_count);
    errdefer allocator.free(layer_cfgs);
    for (0..cfg.block_count) |l| {
        const kf: QuantFormat = if (k_fmt == .fp16) k_fmt else k_fmt;
        const vf: QuantFormat = if (v_fmt == .fp16) v_fmt else v_fmt;
        layer_cfgs[l] = .{
            .k_format = kf,
            .v_format = vf,
            .k_block_size = if (kf == .fp16) 32 else kf.defaultBlockSize(),
            .v_block_size = if (vf == .fp16) 32 else vf.defaultBlockSize(),
            .quant_threshold = null,
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
    var pl = pipeline.InferencePipeline.init(allocator, layers, &kv_manager, cfg.embedding_length, cfg.vocab_size, fa_config);

    // Tokenizer
    var gt = try gguf_tokenizer.GgufTokenizer.fromGguf(allocator, &model.file);
    defer gt.deinit();
    var tok = try bpe.BPETokenizer.fromTokenizer(allocator, &gt);
    defer tok.deinit();

    const prompt = params.prompt orelse "Hola";
    const prompt_ids = try tok.encode(prompt, .{});
    defer allocator.free(prompt_ids);
    try stdout.print("[+] prompt ({d} tokens): {s}\n", .{ prompt_ids.len, prompt });

    // Motor matmul
    if (eff_backend == .cublas) cudaz.ensureCurrent() catch {};
    var engine = try matmul.MatmulEngine.init(allocator, eff_backend, .f32);
    defer engine.deinit();

    const seq_id: u64 = 1;

    const prefill_res = try pl.prefill(seq_id, prompt_ids, emb, lm_head, &engine);
    const first_token = prefill_res.last_token;

    const gen_config = pipeline.GenerationConfig{
        .max_new_tokens = params.max_new_tokens,
        .sampler = params.sampler,
        .eos_token = gt.eos_id,
        .pad_token = null,
        .stop_on_eos = true,
        .seed = params.seed,
    };

    const result = try pl.generate(seq_id, first_token, emb, lm_head, &engine, gen_config);
    defer allocator.free(result.tokens);

    try stdout.print("\n[+] Generación ({d} tokens, {d:.1} tok/s):\n", .{ result.num_tokens_generated, result.tokens_per_second });
    const decoded = try tok.decode(result.tokens, allocator);
    defer allocator.free(decoded);
    try stdout.print("{s}\n", .{decoded});
    try stdout.print("\n=================================================\n", .{});
    try stdout.print("              Ejecucion completada               \n", .{});
    try stdout.print("=================================================\n", .{});
    try stdout.flush();
}

/// Inferencia híbrida (SSM + atención, p.ej. Qwen3.5) usando HybridLayer.
fn runHybridInference(
    allocator: std.mem.Allocator,
    model: *gguf_model.GgufModel,
    params: CliParams,
    backend: matmul.Backend,
    stdout: anytype,
) !void {
    const cfg = model.config;
    const n_embd = cfg.embedding_length;
    const vocab = cfg.vocab_size;
    const max_seq_len = cfg.context_length;
    try stdout.print("[+] arch={s} capas={d} heads={d} kv={d} emb={d} ffn={d} vocab={d} ctx={d} (path paged)\n", .{
        cfg.architecture, cfg.block_count, cfg.head_count, cfg.head_count_kv,
        cfg.embedding_length, cfg.feed_forward_length, cfg.vocab_size, cfg.context_length,
    });

    var emb = try model.loadEmbedding();
    defer emb.deinit();
    var lm_head = try model.loadLmHead();
    defer lm_head.deinit();
    const lm_head_q = try model.loadLmHeadQuant();
    layer_kernels.quant_enabled = params.quant == .auto;
    const lm_head_q4 = lm_head_q.dtype() == gguf.GgmlType.q4_0 and layer_kernels.quantPath();
    const lm_head_q6k = lm_head_q.dtype() == gguf.GgmlType.q6_k and layer_kernels.quantPath();
    var out_norm = try model.loadOutputNorm();
    defer out_norm.deinit();
    const rms_eps = cfg.layer_norm_rms_epsilon;

    // Capas híbridas: cada una enruta SSM vs atención según isFullAttentionLayer.
    // KV-cache paginado compartido: cada capa de atención obtiene su propio block_table.
    const head_dim = if (cfg.head_dim > 0) cfg.head_dim else n_embd / cfg.head_count;
    const block_size: usize = 16;
    var num_attn_layers: usize = 0;
    for (0..cfg.block_count) |i| {
        if (cfg.isFullAttentionLayer(i)) num_attn_layers += 1;
    }
    const blocks_per_seq: usize = (max_seq_len + block_size - 1) / block_size;
    const num_blocks = @max(64, num_attn_layers * blocks_per_seq);

    // --- Spec-decoding: detectar cabeza MTP (tensores `nextn.*`) en el GGUF ---
    // Solo los modelos con cabeza MTP (p.ej. la familia Qwen3.5 con nextn)
    // pueden usar draft-mtp.
    if (params.spec_type == .draft_mtp) {
        const mtp_tensor = try std.fmt.allocPrint(allocator, "blk.0.nextn.eh_proj.weight", .{});
        defer allocator.free(mtp_tensor);
        if (model.file.getTensor(mtp_tensor) == null) {
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
        // MTP head presente: el driver especulativo se engancharía aquí (futuro).
        try stdout.print("[*] Draft MTP detectado: habilitando draft-mtp (driver completo pendiente). n_max={d}\n", .{params.spec_draft_n_max});
    }

    var paged_kv = try paged_attn.PagedKVCache.init(allocator, .{
        .block_size = block_size,
        .num_blocks = num_blocks,
        .head_dim = head_dim,
        .num_kv_heads = cfg.head_count_kv,
        .num_q_heads = cfg.head_count,
        .dtype = .f16,
        .quant_k = params.cache_type_k,
        .quant_v = params.cache_type_v,
        .enable_prefix_cache = false,
        .enable_cpu_offload = false,
        .max_seq_len = max_seq_len,
        .max_batch_size = 1,
    });
    defer paged_kv.deinit();

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
    const quant_on = params.cache_type_k != .fp16 or params.cache_type_v != .fp16;
    const gpu_attention_enabled = true;
    const use_gpu_kv = (backend == .cublas) and !quant_on and gpu_attention_enabled;
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
                .quant_k = params.cache_type_k,
                .quant_v = params.cache_type_v,
            },
            @ptrCast((try matmul.MatmulEngine.sharedCudaStream()).raw),
        ) catch null
    else null;
    defer if (shared_paged_gpu) |*g| g.deinit();
    const shared_gpu_ptr: ?*paged_attn.PagedAttentionGpu = if (shared_paged_gpu) |*g| g else null;

    var layers = try allocator.alloc(hybrid_layer.HybridLayer, cfg.block_count);
    var layer_block_tables = try allocator.alloc(?*paged_attn.BlockTable, cfg.block_count);
    defer allocator.free(layer_block_tables);
    @memset(layer_block_tables, null);
    for (0..cfg.block_count) |i| {
        if (cfg.isFullAttentionLayer(i)) {
            const bt = try allocator.create(paged_attn.BlockTable);
            bt.* = paged_attn.BlockTable.init(allocator, block_size);
            layer_block_tables[i] = bt;
        }
        layers[i] = try hybrid_layer.HybridLayer.init(
            allocator, i,
            hybrid_layer.HybridLayerParams.fromModelConfig(cfg, max_seq_len),
            cfg.isFullAttentionLayer(i),
            backend,
            &paged_kv,
            if (layer_block_tables[i]) |bt| bt else null,
            shared_gpu_ptr,
        );
    }
    defer for (layers) |*l| l.deinit();
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
    if (params.layer_stream) {
        const total_vram = if (backend == .cublas) cudaz.getDeviceTotalMem(cudaz.cuDeviceGet(0) catch return) catch 0 else 0;
        var vb = vram_budget.VramBudget.init(total_vram);
        if (debugz.dbg.at(.info)) vb.reportMetrics();
        vram = vb;

        // Adjust max_resident based on VRAM budget
        const max_res = params.layer_stream_max;
        var s = try layer_streamer.LayerStreamer.init(
            allocator, layers, &model.file, cfg,
            max_res, 2,
        );
        if (debugz.dbg.at(.info)) s.enableDebug();
        streamer = s;
        try stdout.print("[+] LayerStreamer activado: max_resident={d} vram={d}MB\n", .{max_res, total_vram / (1024*1024)});
        try stdout.flush();
    } else {
        // Eager load: load all weights up front (original behavior)
        for (0..cfg.block_count) |i| {
            try layers[i].loadWeightsFromGguf(&model.file);
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

    const prompt = params.prompt orelse "Hola";
    const prompt_ids = try tok.encode(prompt, .{});
    defer allocator.free(prompt_ids);

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
    embedding.embeddingLookup(emb, prompt_ids, 1, seq_len, &hidden);

    const hidden_2d = try hidden.reshape(&[_]usize{ seq_len, n_embd });
    defer { if (hidden_2d.allocator) |a| { a.free(hidden_2d.shape); a.free(hidden_2d.strides); } }

    // Ensure all attention layers have blocks for the prefill sequence
    for (layer_block_tables, 0..) |bt_opt, i| {
        if (bt_opt) |bt| {
            if (bt.num_tokens < seq_len) {
                try bt.appendTokens(paged_kv.block_alloc, seq_len - bt.num_tokens);
            }
        }
        _ = i;
    }
    const use_gpu_prefill = !debugz.dbg.no_gpu_prefill;
    var logits_f32 = try allocator.alloc(f32, vocab);
    defer allocator.free(logits_f32);

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
                try hybrid_layer.HybridLayer.forwardGPU(layer, &lk, cur2gpu, &nxt2gpu, pos, n);
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
                    std.debug.print("PREFILL_LAYER li={d} n={d} sum|v|={d:.6} max={d:.6} f0={d:.5} f1={d:.5} f2={d:.5}\n", .{ li, n, debugz.sumAbsF32(chk), debugz.maxAbsF32(chk), chk[0], chk[1], chk[2] });
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
        try lk.rmsNorm(last_row, @intFromPtr(g_out_norm.dev_ptr), g_normed.ptr(), 1, n_embd, rms_eps);
        if (lm_head_q4) {
            try lk.q4gemmLinear(allocator, g_normed.ptr(), lm_head_q.bytes, g_logits.ptr(), n_embd, vocab);
        } else if (lm_head_q6k) {
            try lk.qgemmLinear(allocator, g_normed.ptr(), lm_head_q.bytes, g_logits.ptr(), 1, n_embd, vocab, 3);
        } else {
            try engine.linearProjectionDeviceF16(g_normed, lm_head, &g_logits, 1, n_embd, vocab);
        }
        if (perf_prefill) try cudaz.cuEventRecord(p_ev[layers.len + 1], lk.stream);
        try cudaz.cuStreamSynchronize(lk.stream);
        try cudaz.cuMemcpyDtoH(@intFromPtr(logits_f32.ptr), g_logits.ptr(), vocab * @sizeOf(f32));
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
            try stdout.print("[+] PERF prefill (total {d:.1} ms):\n", .{ @as(f64, @floatFromInt(p_gpu_total_ns)) / @as(f64, @floatFromInt(std.time.ns_per_ms)) });
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
                    li, if (layers[li].is_attention) "attn" else "ssm",
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
        for (layers) |*layer| {
            try layer.forward(cur.*, nxt, 0, seq_len);
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

    var gen_tokens: std.ArrayList(u32) = .empty;
    defer gen_tokens.deinit(allocator);
    const first_token = params.sampler.sample(logits_f32, &rng, &[_]u32{});
    try gen_tokens.append(allocator, first_token);
     var current_pos: usize = seq_len;
     const t_gen = @import("time").Timer.start();

     try stdout.print("[*] Generando... 0 tokens\n", .{});

     // Activaciones residentes en GPU (Path B): un solo H2D (embedding) y un
     // solo D2H (norma final) por token; todo lo demás queda en device.
     var lk = try layer_kernels.LayerKernels.init(@ptrCast((try matmul.MatmulEngine.sharedCudaStream()).raw));
     defer lk.deinit();
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

// Staging persistente del embedding (f32, PINNED): fuente del H2D normal y
    // de los nodos HtoDAsync capturados por el grafo de decode. Pinneado porque
    // los HtoDAsync con fuente pageable no son capturables por CUDA graphs.
    var embed_staging = try cudaz.pinnedAlloc(f32, n_embd);
    defer cudaz.pinnedFree(f32, embed_staging);

     // ─── CUDA Graphs para el decode (una captura, replay por token) ───────
     // El grafo cubre: embed H2D + 24 capas híbridas + rmsNorm final + lm_head
     // (~290 kernels). Los valores que cambian por token (embedding, block
     // table, start_pos, seq_len) se pintan en staging host persistente y los
     // nodos capturados (HtoDAsync) los copian a buffers device fijos. NOGRAPH=1
     // desactiva; ante error de captura/instanciación se cae al camino normal.
     var decode_g: ?decode_graph.DecodeGraph = null;
     const graph_ok = !debugz.dbg.no_graph;
     if (graph_ok) {
         // Bloques para el token de la captura (el decode real los re-usa).
         for (layer_block_tables) |bt_opt| {
             if (bt_opt) |bt| {
                 if (bt.num_tokens < current_pos + 1) {
                     try bt.appendToken(paged_kv.block_alloc);
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
         var state_parts: std.ArrayList(decode_graph.StatePart) = .empty;
         defer state_parts.deinit(allocator);
         for (layers) |*layer| {
             if (!layer.is_attention) {
                 const ssm = layer.ssm_layer.?;
                 if (ssm.gpu) |gpu| {
                     try state_parts.append(allocator, .{ .dev = @intFromPtr(gpu.d_s_state.dev_ptr), .bytes = gpu.d_s_state.len * @sizeOf(f32) });
                     try state_parts.append(allocator, .{ .dev = @intFromPtr(gpu.d_conv_state.dev_ptr), .bytes = gpu.d_conv_state.len * @sizeOf(f32) });
                 }
             }
         }
         var dg: ?decode_graph.DecodeGraph = decode_graph.DecodeGraph{
             .allocator = allocator,
             .stream = lk.stream,
             .n_embd = n_embd,
         };
         if (debugz.dbg.chk_state) {
             try cudaz.cuCtxSynchronize();
             for (state_parts.items) |p| {
                 const nf32 = p.bytes / @sizeOf(f32);
                 const buf = try allocator.alloc(f32, nf32);
                 defer allocator.free(buf);
                 try cudaz.cuMemcpyDtoH(@intFromPtr(buf.ptr), p.dev, p.bytes);
                 var s: f64 = 0;
                 var mx: f32 = 0;
                 for (buf) |v| { s += @abs(@as(f64, v)); if (@abs(v) > mx) mx = @abs(v); }
                 std.debug.print("CHKSTATE_BEFORE dev={x} bytes={d} sum|v|={d:.6} max={d:.6}\n", .{ p.dev, p.bytes, s, mx });
             }
         }
         if (dg) |*g| {
             g.setEmbedStaging(embed_staging);
             if (captureDecodeGraph(g, &lk, layers, &g_cur, &g_nxt, &g_normed, &g_logits, &g_out_norm, &engine, allocator, n_embd, vocab, rms_eps, current_pos, state_parts.items, lm_head_q4, lm_head_q6k, lm_head_q, lm_head)) {
                 decode_g = dg;
if (debugz.dbg.chk_state) {
                     try cudaz.cuCtxSynchronize();
                     for (state_parts.items) |p| {
                         const nf32 = p.bytes / @sizeOf(f32);
                         const buf = try allocator.alloc(f32, nf32);
                         defer allocator.free(buf);
                         try cudaz.cuMemcpyDtoH(@intFromPtr(buf.ptr), p.dev, p.bytes);
                         var s: f64 = 0;
                         var mx: f32 = 0;
                         for (buf) |v| { s += @abs(@as(f64, v)); if (@abs(v) > mx) mx = @abs(v); }
                         std.debug.print("CHKSTATE dev={x} bytes={d} sum|v|={d:.6} max={d:.6}\n", .{ p.dev, p.bytes, s, mx });
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
      for (0..params.max_new_tokens) |_| {
          const last = gen_tokens.items[gen_tokens.items.len - 1];

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

         const t_embed0 = perf_t.read();
         var h1 = try Tensor(f16).alloc(allocator, &.{ 1, 1, n_embd });
        defer h1.deinit();
        embedding.embeddingLookup(emb, &[_]u32{last}, 1, 1, &h1);
        const h2d = try h1.reshape(&[_]usize{ 1, n_embd });
        defer { if (h2d.allocator) |a| { a.free(h2d.shape); a.free(h2d.strides); } }

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
            if (perf_stage) try cudaz.cuEventRecord(ev[0], lk.stream);
            try g.launch();
            if (perf_stage) try cudaz.cuEventRecord(ev[layers.len + 1], lk.stream);
        } else {
            // Camino normal: subir embedding (única H2D por token) y lanzar la
            // secuencia capa a capa.
            if (perf_stage) try cudaz.cuEventRecord(ev[0], lk.stream);
            try cudaz.cuMemcpyHtoDAsync(g_cur.ptr(), @intFromPtr(embed_staging.ptr), n_embd * @sizeOf(f32), lk.stream);
            if (debugz.dbg.dump_prefill_layers) {
                try cudaz.cuStreamSynchronize(lk.stream);
                const chk = try allocator.alloc(f32, n_embd);
                defer allocator.free(chk);
                try cudaz.cuMemcpyDtoH(@intFromPtr(chk.ptr), g_cur.ptr(), n_embd * @sizeOf(f32));
                std.debug.print("EMBED_IN tok={d} sum|v|={d:.6} max={d:.6} f0={d:.5} f1={d:.5} f2={d:.5}\n", .{ current_pos, debugz.sumAbsF32(chk), debugz.maxAbsF32(chk), chk[0], chk[1], chk[2] });
            }
            var cur2gpu = g_cur;
            var nxt2gpu = g_nxt;
            for (layers, 0..) |*layer, li| {
                if (streamer) |*s| try s.ensureLayerLoaded(li);
                try hybrid_layer.HybridLayer.forwardGPU(layer, &lk, cur2gpu, &nxt2gpu, current_pos, 1);
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
                    std.debug.print("DECODE_LAYER tok={d} li={d} sum|v|={d:.6} max={d:.6} f0={d:.5} f1={d:.5} f2={d:.5}\n", .{ current_pos, li, debugz.sumAbsF32(chk), debugz.maxAbsF32(chk), chk[0], chk[1], chk[2] });
                    if (li == 0) {
                        std.debug.print("DECODE_INPUT tok={d} f0={d:.5} f1={d:.5} f2={d:.5}\n", .{ current_pos, chk[0], chk[1], chk[2] });
                    }
                }
            }
            // Norma final en GPU, lm_head device→device (peso cacheado en GPU).
            try lk.rmsNorm(cur2gpu.ptr(), @intFromPtr(g_out_norm.dev_ptr), g_normed.ptr(), 1, n_embd, rms_eps);
            // lm_head M=1: Q4_0/Q6_K device→device si el peso está cuantizado.
            if (lm_head_q4) {
                try lk.q4gemmLinear(allocator, g_normed.ptr(), lm_head_q.bytes, g_logits.ptr(), n_embd, vocab);
            } else if (lm_head_q6k) {
                try lk.qgemmLinear(allocator, g_normed.ptr(), lm_head_q.bytes, g_logits.ptr(), 1, n_embd, vocab, 3);
            } else {
                try engine.linearProjectionDeviceF16(g_normed, lm_head, &g_logits, 1, n_embd, vocab);
            }
            if (perf_stage) try cudaz.cuEventRecord(ev[layers.len + 1], lk.stream);
        }
        t_enqueue_ns += perf_t.read() - t_enq0;

        if (debugz.dbg.dump_norm and current_pos == seq_len) {
            try cudaz.cuStreamSynchronize(lk.stream);
            const nb = try allocator.alloc(f32, n_embd);
            defer allocator.free(nb);
            try cudaz.cuMemcpyDtoH(@intFromPtr(nb.ptr), g_normed.ptr(), n_embd * @sizeOf(f32));
            var s: f64 = 0;
            var mx: f32 = 0;
            for (nb) |v| { s += @abs(@as(f64, v)); if (@abs(v) > mx) mx = @abs(v); }
            std.debug.print("DUMPNORM pos={d} sum|v|={d:.6} max={d:.6} first3={d:.4},{d:.4},{d:.4}\n", .{ current_pos, s, mx, nb[0], nb[1], nb[2] });
        }

        // D2H async (stream-ordered tras los kernels/grafo que los escribieron)
        // de los bloques KV del token; mantiene el pool host autoritativo.
        for (layers) |*layer| {
            if (layer.is_attention) try layer.attn_layer.?.syncDecodeBlocks(current_pos, 1);
        }

        try cudaz.cuStreamSynchronize(lk.stream);
        if (debugz.dbg.dump_prefill_layers) {
            for (layers, 0..) |layer, li| {
                if (!layer.is_attention) {
                    if (layer.ssm_layer.?.gpu) |gpu| {
                        const nf32 = gpu.d_s_state.len;
                        const buf = try allocator.alloc(f32, nf32);
                        defer allocator.free(buf);
                        try cudaz.cuMemcpyDtoH(@intFromPtr(buf.ptr), @intFromPtr(gpu.d_s_state.dev_ptr), nf32 * @sizeOf(f32));
                        std.debug.print("SSTATE tok={d} li={d} sum|v|={d:.6} max={d:.6}\n", .{ current_pos, li, debugz.sumAbsF32(buf), debugz.maxAbsF32(buf) });
                        if (gpu.d_conv_state.len > 0) {
                            const nf2 = gpu.d_conv_state.len;
                            const buf2 = try allocator.alloc(f32, nf2);
                            defer allocator.free(buf2);
                            try cudaz.cuMemcpyDtoH(@intFromPtr(buf2.ptr), @intFromPtr(gpu.d_conv_state.dev_ptr), nf2 * @sizeOf(f32));
                            std.debug.print("CONVSTATE tok={d} li={d} sum|v|={d:.6} max={d:.6}\n", .{ current_pos, li, debugz.sumAbsF32(buf2), debugz.maxAbsF32(buf2) });
                        }
                    }
                }
            }
        }
        if (perf_stage) {
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
        try cudaz.cuMemcpyDtoH(@intFromPtr(logits_f32.ptr), g_logits.ptr(), vocab * @sizeOf(f32));

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
                            std.debug.print("DUMPKV pos={d} LAYERTAB li={d} bt[0]={?d} blocks={d}\n", .{ current_pos, li2, hb2, bt2.numBlocks() });
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
                std.debug.print("DUMPKV pos={d} layer={d} HOST phys={d} all=", .{ current_pos, li, phys });
                for (all_h) |sp| std.debug.print("{d:.1},", .{sp});
                std.debug.print("\n", .{});
                if (layer.attn_layer.?.paged_gpu) |gpu| {
                    const st0: c_int = if (li < gpu.bt_stagings.items.len and gpu.bt_stagings.items[li].len > 0) gpu.bt_stagings.items[li][0] else -999;
                    const hbt0: ?usize = layer.attn_layer.?.block_table.getPhysical(0);
                    var dbt0: c_int = 0;
                    try cudaz.cuMemcpyDtoH(@intFromPtr(&dbt0), gpu.getDbt(li), @sizeOf(c_int));
                    const stlen: usize = if (li < gpu.bt_stagings.items.len) gpu.bt_stagings.items[li].len else 0;
                    std.debug.print("DUMPKV pos={d} layer={d} host_bt[0]={?d} staging[0]={d} staging_len={d} d_bt[0]={d}\n", .{ current_pos, li, hbt0, st0, stlen, dbt0 });
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
                    std.debug.print("DUMPKV pos={d} layer={d} DEV  phys={d} all=", .{ current_pos, li, phys });
                    for (all_d) |sp| std.debug.print("{d:.1},", .{sp});
                    std.debug.print("\n", .{});
                    var dsp: c_int = 0;
                    try cudaz.cuMemcpyDtoH(@intFromPtr(&dsp), gpu.getDStartPos(), @sizeOf(c_int));
                    var dsq: c_int = 0;
                    try cudaz.cuMemcpyDtoH(@intFromPtr(&dsq), gpu.getDSeqLens(), @sizeOf(c_int));
                    var dbt: c_int = 0;
                    try cudaz.cuMemcpyDtoH(@intFromPtr(&dbt), gpu.getDbt(li), @sizeOf(c_int));
                    std.debug.print("DUMPKV pos={d} layer={d} d_start_pos={d} d_seq_len={d} d_bt[0]={d}\n", .{ current_pos, li, dsp, dsq, dbt });
                }
                break;
            }
        }
        if (debugz.dbg.dump_logits) {
            for (logits_f32, 0..) |v, i| std.debug.print("LG {d} {d}\n", .{ i, v });
        }
         const t_samp0 = perf_t.read();
         const next_token = params.sampler.sample(logits_f32, &rng, gen_tokens.items);
         try gen_tokens.append(allocator, next_token);
         try scheduler.appendToken(seq_id, next_token);
         current_pos += 1;
         t_d2h_ns += t_samp0 - t_d2h0;
         t_sample_ns += perf_t.read() - t_samp0;

      if (gt.eos_id != null and next_token == gt.eos_id.?) break;
      }

      // Finish sequence: release blocks
      scheduler.finishSequence(seq_id);

     const gen_ns = t_gen.read();
     const gen_ms = @as(f64, @floatFromInt(@divTrunc(gen_ns, std.time.ns_per_ms)));
     const prefill_ms = @as(f64, @floatFromInt(@divTrunc(prefill_ns, std.time.ns_per_ms)));
     const tok_s: f64 = if (gen_ms > 0) @as(f64, @floatFromInt(gen_tokens.items.len)) / (gen_ms / 1000.0) else 0;

     try stdout.print("✓ Listo! Generados {d} tokens en {d:.1}s ({d:.1} tok/s)\n\n", .{
         gen_tokens.items.len,
         @divTrunc(gen_ns, @as(i128, std.time.ns_per_s)),
         tok_s,
     });

    try stdout.print("\n[+] Generación ({d} tokens):\n", .{ gen_tokens.items.len });
    const decoded = try tok.decode(gen_tokens.items, allocator);
    defer allocator.free(decoded);
    try stdout.print("{s}\n", .{decoded});
try stdout.print("[+] Métricas: prefill {d:.1} ms ({d} tok), generación {d:.1} ms ({d:.2} tok/s)\n", .{ prefill_ms, seq_len, gen_ms, tok_s });
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
                 li, if (layers[li].is_attention) "attn" else "ssm",
                 @as(f64, @floatFromInt(layer_gpu_ns[li])) / nt / @as(f64, @floatFromInt(us)),
             });
         }
     }
     try stdout.print("\n=================================================\n", .{});
    try stdout.print("              Ejecucion completada               \n", .{});
    try stdout.print("=================================================\n", .{});
    try stdout.flush();
}

fn runGpuMode(allocator: std.mem.Allocator, config: FlashAttentionConfig, stdout: anytype) !void {
    try stdout.print("\n[+] Inicializando TransformerLayer con GPU...\n", .{});

    const precision = LayerPrecision{
        .compute = .f16,
        .weights_on_gpu = false,
        .use_quantized = false,
    };

    var layer = try TransformerLayer.init(allocator, 0, config, "cuda/flash_attention.ptx", 1024, precision, config.num_heads, 4096);
    defer layer.deinit();

    try stdout.print("[+] Capa transformer inicializada\n", .{});
    try stdout.print("[+] Backend matmul: {s}\n", .{layer.matmul_engine.backendName()});

    var hidden_state = try Tensor(f16).alloc(allocator, &.{ config.batch_size, config.N, 1024 });
    defer hidden_state.deinit();
    var rng = std.Random.Xoshiro256.init(42);
    hidden_state.randUniform(&rng, -0.1, 0.1);

    var output = try Tensor(f16).alloc(allocator, &.{ config.batch_size, config.N, 1024 });
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
