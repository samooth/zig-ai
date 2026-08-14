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
const pipeline = @import("pipeline");
const gguf_model = @import("gguf_model");
const gguf_tokenizer = @import("gguf_tokenizer");
const bpe = @import("tokenizer");
const embedding = @import("embedding");
const hybrid_layer = @import("transformer");
const norm = @import("norm");
const gguf = @import("gguf");

/// Parámetros de runtime parseados de la línea de comandos.
const CliParams = struct {
    model_path: ?[]const u8 = null,
    prompt: ?[]const u8 = null,
    max_new_tokens: usize = 128,
    seed: u64 = 42,
    backend: []const u8 = "auto", // auto | cpu | gpu
    sampler: pipeline.Sampler = .{},
};

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

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.gpa;

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
        }
        try runInference(io, allocator, path, params, backend, stdout);
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
        if (std.mem.eql(u8, arg, "--model")) {
            params.model_path = try nextValue(&it, "--model");
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

fn printHelp(stdout: anytype) !void {
    try stdout.print(
        \\Uso: zig-ai-engine [opciones]
        \\  --model <ruta>            Ruta a un modelo GGUF (activa inferencia)
        \\  --prompt <texto>          Prompt de entrada
        \\  -n, --max-tokens <n>      Máximo de tokens a generar (def: 128)
        \\  --temperature <f>         Temperatura (def: 1.0; <=0 = greedy)
        \\  --top-k <n>               Top-k (def: 0 = desactivado)
        \\  --top-p <f>               Top-p / nucleus (def: 1.0 = desactivado)
        \\  --repetition-penalty <f>  Repetition penalty (def: 1.0 = desactivado)
        \\  --seed <n>                Semilla del RNG (def: 42)
        \\  --backend <auto|cpu|gpu>  Backend matmul (def: auto → GPU si disponible)
        \\  -h, --help                Muestra esta ayuda
        \\
    , .{});
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
    var model = try gguf_model.GgufModel.load(io, allocator, model_path);
    defer model.deinit();
    const cfg = model.config;

    if (cfg.is_hybrid) {
        try runHybridInference(allocator, &model, params, backend, stdout);
        return;
    }

    try stdout.print("[+] arch={s} capas={d} heads={d} kv={d} emb={d} ffn={d} vocab={d} ctx={d}\n", .{
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
        try layers[i].loadWeightsFromGguf(&model.file);
    }
    defer for (layers) |*l| l.deinit();

    // KV cache manager
    const kv_config = KVCacheConfig.default(
        @intCast(cfg.block_count), @intCast(cfg.head_count),
        @intCast(head_dim), @intCast(cfg.context_length),
    );
    var kv_manager = try KVCacheManager.init(allocator, kv_config, 256);
    defer kv_manager.deinit();

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
    var engine = try matmul.MatmulEngine.init(allocator, backend, .f32);
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
    for (result.tokens) |t| {
        if (tok.vocab_inv.get(t)) |s| try stdout.print("{s}", .{s})
        else try stdout.print("[{}]", .{t});
    }
    try stdout.print("\n", .{});
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

    var emb = try model.loadEmbedding();
    defer emb.deinit();
    var lm_head = try model.loadLmHead();
    defer lm_head.deinit();
    var out_norm = try model.loadOutputNorm();
    defer out_norm.deinit();
    const rms_eps = cfg.layer_norm_rms_epsilon;

    // Capas híbridas: cada una enruta SSM vs atención según isFullAttentionLayer.
    var layers = try allocator.alloc(hybrid_layer.HybridLayer, cfg.block_count);
    for (0..cfg.block_count) |i| {
        layers[i] = try hybrid_layer.HybridLayer.init(
            allocator, i,
            hybrid_layer.HybridLayerParams.fromModelConfig(cfg, max_seq_len),
            cfg.isFullAttentionLayer(i),
            backend,
        );
        try layers[i].loadWeightsFromGguf(&model.file);
    }
    defer for (layers) |*l| l.deinit();

    // Tokenizer
    var gt = try gguf_tokenizer.GgufTokenizer.fromGguf(allocator, &model.file);
    defer gt.deinit();
    var tok = try bpe.BPETokenizer.fromTokenizer(allocator, &gt);
    defer tok.deinit();

    const prompt = params.prompt orelse "Hola";
    const prompt_ids = try tok.encode(prompt, .{});
    defer allocator.free(prompt_ids);
    try stdout.print("[+] prompt ({d} tokens): {s}\n", .{ prompt_ids.len, prompt });

    var engine = try matmul.MatmulEngine.init(allocator, backend, .f32);
    defer engine.deinit();

    var rng = std.Random.Xoshiro256.init(params.seed);

    // === Prefill ===
    const seq_len = prompt_ids.len;
    var hidden = try Tensor(f16).alloc(allocator, &.{ 1, seq_len, n_embd });
    defer hidden.deinit();
    embedding.embeddingLookup(emb, prompt_ids, 1, seq_len, &hidden);

    const hidden_2d = try hidden.reshape(&[_]usize{ seq_len, n_embd });
    defer { if (hidden_2d.allocator) |a| { a.free(hidden_2d.shape); a.free(hidden_2d.strides); } }

    var buf_a = try Tensor(f16).alloc(allocator, &.{ seq_len, n_embd });
    defer buf_a.deinit();
    var buf_b = try Tensor(f16).alloc(allocator, &.{ seq_len, n_embd });
    defer buf_b.deinit();
    // La primera capa recibe el embedding del prompt
    @memcpy(buf_a.data, hidden_2d.data);

    var cur = &buf_a;
    var nxt = &buf_b;

    for (layers, 0..) |*layer, li| {
        try layer.forward(cur.*, nxt, 0, seq_len);
        const t = cur;
        cur = nxt;
        nxt = t;
        if (li == 0 or li == 1 or li == 2 or li == 3 or li == 23) {
            var shp = [_]usize{ 1, n_embd };
            var strd = [_]usize{ n_embd, 1 };
            const lastrow = Tensor(f16){ .data = cur.data[(seq_len - 1) * n_embd ..][0..n_embd], .shape = &shp, .strides = &strd, .offset = 0, .allocator = null, .owns_data = false };
            var nr = try Tensor(f16).alloc(allocator, &.{ 1, n_embd });
            defer nr.deinit();
            norm.rmsNorm(f16, f32, lastrow, out_norm, rms_eps, &nr);
            var lg = try Tensor(f16).alloc(allocator, &.{ 1, vocab });
            defer lg.deinit();
            try embedding.lmHeadForward(&engine, nr, lm_head, &lg);
            var best: u32 = 0;
            var bv: f32 = -std.math.inf(f32);
            for (lg.data, 0..) |v, i| {
                const f = @as(f32, @floatCast(v));
                if (f > bv) {
                    bv = f;
                    best = @intCast(i);
                }
            }
            try stdout.print("[dbg-l{d}] top1 id={d} ({d:.2})\n", .{ li, best, bv });
        }
    }

    // LM head sobre el último token del prefill (con output_norm final)
    var last_shape = [_]usize{ 1, n_embd };
    var last_strides = [_]usize{ n_embd, 1 };
    const last_2d = Tensor(f16){
        .data = cur.data[(seq_len - 1) * n_embd ..][0..n_embd],
        .shape = &last_shape,
        .strides = &last_strides,
        .offset = 0,
        .allocator = null,
        .owns_data = false,
    };

    var normed = try Tensor(f16).alloc(allocator, &.{ 1, n_embd });
    defer normed.deinit();
    norm.rmsNorm(f16, f32, last_2d, out_norm, rms_eps, &normed);

    var logits = try Tensor(f16).alloc(allocator, &.{ 1, vocab });
    defer logits.deinit();
    try embedding.lmHeadForward(&engine, normed, lm_head, &logits);

    var logits_f32 = try allocator.alloc(f32, vocab);
    defer allocator.free(logits_f32);
    for (logits.data, 0..) |v, i| logits_f32[i] = @as(f32, @floatCast(v));

    if (std.c.getenv("DUMP_LOGITS") != null) {
        for (logits_f32, 0..) |v, i| std.debug.print("LG {d} {d}\n", .{ i, v });
    }

    var gen_tokens: std.ArrayList(u32) = .empty;
    defer gen_tokens.deinit(allocator);
    const first_token = params.sampler.sample(logits_f32, &rng, &[_]u32{});
    try gen_tokens.append(allocator, first_token);

    var current_pos: usize = seq_len;
    for (0..params.max_new_tokens) |_| {
        const last = gen_tokens.items[gen_tokens.items.len - 1];

        var h1 = try Tensor(f16).alloc(allocator, &.{ 1, 1, n_embd });
        defer h1.deinit();
        embedding.embeddingLookup(emb, &[_]u32{last}, 1, 1, &h1);
        const h2d = try h1.reshape(&[_]usize{ 1, n_embd });
        defer { if (h2d.allocator) |a| { a.free(h2d.shape); a.free(h2d.strides); } }

        var ca = try Tensor(f16).alloc(allocator, &.{ 1, n_embd });
        defer ca.deinit();
        var cb = try Tensor(f16).alloc(allocator, &.{ 1, n_embd });
        defer cb.deinit();
        // La primera capa recibe el embedding del token actual
        @memcpy(ca.data, h2d.data);
        var cur2 = &ca;
        var nxt2 = &cb;
        for (layers) |*layer| {
            try layer.forward(cur2.*, nxt2, current_pos, 1);
            const t2 = cur2;
            cur2 = nxt2;
            nxt2 = t2;
        }

        var normed2 = try Tensor(f16).alloc(allocator, &.{ 1, n_embd });
        defer normed2.deinit();
        norm.rmsNorm(f16, f32, cur2.*, out_norm, rms_eps, &normed2);

        var logits2 = try Tensor(f16).alloc(allocator, &.{ 1, vocab });
        defer logits2.deinit();
        try embedding.lmHeadForward(&engine, normed2, lm_head, &logits2);

        for (logits2.data, 0..) |v, i| logits_f32[i] = @as(f32, @floatCast(v));
        const next_token = params.sampler.sample(logits_f32, &rng, gen_tokens.items);
        try gen_tokens.append(allocator, next_token);
        current_pos += 1;

        if (gt.eos_id != null and next_token == gt.eos_id.?) break;
    }

    try stdout.print("\n[+] Generación ({d} tokens):\n", .{ gen_tokens.items.len });
    for (gen_tokens.items) |tk| {
        if (tok.vocab_inv.get(tk)) |s| try stdout.print("{s}", .{s})
        else try stdout.print("[{}]", .{tk});
    }
    try stdout.print("\n", .{});
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
