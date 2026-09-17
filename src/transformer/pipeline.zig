const embedding = @import("embedding"); // 7.1d wiring
const std = @import("std");
const Tensor = @import("core").Tensor;
const matmul = @import("matmul");
const fa = @import("fa");
const transformer = @import("transformer");
const kvcache = @import("kv_cache");
const norm = @import("norm");
const nvtx = @import("nvtx"); // lane-cuda UC-4.2: rangos nsys legibles
const debugz = @import("debug");
const capture_mod = @import("capture");

/// Handle de captura RLT dentro de prefill (RLT_CAPTURE_DIR).
const CaptureHandle = struct {
    w: capture_mod.CaptureWriter,
    n_caps: usize = 0,

    fn deinit(self: *CaptureHandle) void {
        self.w.deinit();
    }
};

fn initCaptureHandle(
    allocator: std.mem.Allocator,
    cap_dir: []const u8,
    seq_len: usize,
    hidden_dim: usize,
    num_layers: usize,
    prompt_tokens: []const u32,
) !CaptureHandle {
    // RLT_CAPTURE_LAYERS=0,4,8 → índices; vacío/ausente = todas.
    var layer_buf: [256]usize = undefined;
    var n_sel: usize = 0;
    if (debugz.dbg.rlt_capture_layers) |spec| {
        var it = std.mem.tokenizeScalar(u8, spec, ',');
        while (it.next()) |tok| {
            const n = std.fmt.parseInt(usize, std.mem.trim(u8, tok, " "), 10) catch continue;
            if (n < num_layers and n_sel < layer_buf.len) {
                layer_buf[n_sel] = n;
                n_sel += 1;
            }
        }
    }
    if (n_sel == 0) {
        for (0..@min(num_layers, layer_buf.len)) |i| {
            layer_buf[i] = i;
        }
        n_sel = @min(num_layers, layer_buf.len);
    }
    // ordenar ascendente (addLayer espera la secuencia del loop)
    std.mem.sort(usize, layer_buf[0..n_sel], {}, std.sort.asc(usize));

    const io = std.Io.Threaded.global_single_threaded.io();
    const w = try capture_mod.CaptureWriter.init(io, allocator, cap_dir, seq_len, hidden_dim, layer_buf[0..n_sel], prompt_tokens);
    return .{ .w = w };
}

const FlashAttentionConfig = fa.fa_config.FlashAttentionConfig;
const AttentionEngine = transformer.AttentionEngine;
const TransformerLayer = transformer.TransformerLayer;
const LayerPrecision = transformer.LayerPrecision;
const KVCacheManager = kvcache.KVCacheManager;
const KVCacheConfig = kvcache.KVCacheConfig;

/// Sampler combinable: temperature + top-k + top-p + repetition penalty.
/// Cualquier parámetro en su valor por defecto (0 / 1.0) queda desactivado.
/// temperature <= 0 fuerza greedy (argmax).
pub const Sampler = struct {
    temperature: f32 = 1.0,
    top_k: usize = 0,
    top_p: f32 = 1.0,
    repetition_penalty: f32 = 1.0,

    work_buf: []f32 = &.{},
    vals_buf: []f32 = &.{},
    idx_buf: []usize = &.{},
    probs_buf: []f32 = &.{},

    pub fn initScratch(self: *Sampler, allocator: std.mem.Allocator, vocab_size: usize) !void {
        self.work_buf = try allocator.alloc(f32, vocab_size);
        self.vals_buf = try allocator.alloc(f32, vocab_size);
        self.idx_buf = try allocator.alloc(usize, vocab_size);
        self.probs_buf = try allocator.alloc(f32, vocab_size);
    }

    fn deinitScratch(self: *Sampler, allocator: std.mem.Allocator) void {
        if (self.work_buf.len > 0) allocator.free(self.work_buf);
        if (self.vals_buf.len > 0) allocator.free(self.vals_buf);
        if (self.idx_buf.len > 0) allocator.free(self.idx_buf);
        if (self.probs_buf.len > 0) allocator.free(self.probs_buf);
        self.* = .{};
    }

    pub fn sample(self: Sampler, logits: []const f32, rng: *std.Random.Xoshiro256, history: []const u32) u32 {
        const vocab = logits.len;

        // 0. Greedy directo: temperatura <= 0 sin repetition penalty → escaneo
        // vectorizado de logits (sin copia intermedia).
        if (self.temperature <= 0 and self.repetition_penalty == 1.0) {
            return greedyArgmax(logits);
        }

        // 0b. Fast path: top_p = 1 (sin recorte de núcleo), sin top-k y sin
        // repetition penalty. El muestreo ponderado NO necesita ordenar: la
        // probabilidad de cada índice es la misma en cualquier orden. Evita el
        // sort completo (248K) y las 3 asignaciones grandes por token, y produce
        // exactamente la misma distribución que el camino genérico.
        if (self.temperature > 0 and self.top_k == 0 and self.top_p >= 1.0 and self.repetition_penalty == 1.0) {
            const inv_t = 1.0 / self.temperature;
            var max_val: f32 = -std.math.inf(f32);
            for (logits) |v| max_val = @max(max_val, v);
            var sum: f32 = 0;
            for (logits) |v| sum += @exp((v - max_val) * inv_t);
            const r = rng.random().float(f32) * sum;
            var acc: f32 = 0;
            for (logits, 0..) |v, i| {
                acc += @exp((v - max_val) * inv_t);
                if (r <= acc) return @as(u32, @intCast(i));
            }
            return @as(u32, @intCast(logits.len - 1));
        }

        var work = self.work_buf;
        @memcpy(work, logits);

        // 1. Repetition penalty: penaliza tokens ya generados
        if (self.repetition_penalty != 1.0) {
            for (history) |tok| {
                if (tok < vocab) {
                    if (work[tok] > 0) work[tok] /= self.repetition_penalty else if (work[tok] < 0) work[tok] *= self.repetition_penalty;
                }
            }
        }

        // 2. Greedy si temperature <= 0
        if (self.temperature <= 0) {
            return greedyArgmax(work);
        }

        // 3. Temperature scaling
        const inv_t = 1.0 / self.temperature;
        for (work) |*v| v.* *= inv_t;

        // 4. Top-k: descarta lo que está por debajo del k-ésimo valor
        if (self.top_k > 0 and self.top_k < vocab) {
            var vals = self.vals_buf;
            @memcpy(vals, work);
            const k = self.top_k;
            for (0..k) |i| {
                var max_idx = i;
                var max_val = vals[i];
                for (i..vocab) |j| {
                    if (vals[j] > max_val) {
                        max_val = vals[j];
                        max_idx = j;
                    }
                }
                const tmp = vals[i];
                vals[i] = vals[max_idx];
                vals[max_idx] = tmp;
            }
            const kth = vals[k - 1];
            for (work) |*v| {
                if (v.* < kth) v.* = -std.math.inf(f32);
            }
        }

        // 5. Top-p (nucleus): ordenar descendente y acumular hasta p
        var idx = self.idx_buf;
        for (0..vocab) |i| idx[i] = i;

        // Candidatos (work != -inf)
        var cand_count: usize = 0;
        for (0..vocab) |i| {
            if (work[i] != -std.math.inf(f32)) {
                idx[cand_count] = i;
                cand_count += 1;
            }
        }

        // Ordenar candidatos por work descendente
        const SortCtx = struct {
            work: []const f32,
            fn lessThan(ctx: @This(), a: usize, b: usize) bool {
                return ctx.work[b] < ctx.work[a];
            }
        };
        std.sort.pdq(usize, idx[0..cand_count], SortCtx{ .work = work }, SortCtx.lessThan);

        // Softmax estable sobre candidatos
        var max_val: f32 = -std.math.inf(f32);
        for (0..cand_count) |i| max_val = @max(max_val, work[idx[i]]);
        var probs = self.probs_buf;
        var sum: f32 = 0;
        for (0..cand_count) |i| {
            probs[i] = @exp(work[idx[i]] - max_val);
            sum += probs[i];
        }
        for (probs) |*p| p.* /= sum;

        // Recortar cola hasta acumular top_p
        var cumsum: f32 = 0;
        var cutoff: usize = cand_count;
        for (0..cand_count) |i| {
            cumsum += probs[i];
            if (cumsum >= self.top_p) {
                cutoff = i + 1;
                break;
            }
        }

        // Renormalizar sobre el núcleo y samplear
        var sub_sum: f32 = 0;
        for (0..cutoff) |i| sub_sum += probs[i];
        const r = rng.random().float(f32) * sub_sum;
        cumsum = 0;
        for (0..cutoff) |i| {
            cumsum += probs[i];
            if (r <= cumsum) return @as(u32, @intCast(idx[i]));
        }
        return @as(u32, @intCast(idx[cutoff - 1]));
    }

    /// Argmax vectorizado de 16 en 16. Mantiene la semántica del escaneo
    /// escalar: el índice de la PRIMERA aparición del máximo.
    fn greedyArgmax(logits: []const f32) u32 {
        const V = @Vector(16, f32);
        var max_idx: usize = 0;
        var max_val: f32 = -std.math.inf(f32);
        var i: usize = 0;
        const n = logits.len;
        while (i + 16 <= n) : (i += 16) {
            const vf: V = logits[i..][0..16].*;
            const cm = @reduce(.Max, vf);
            if (cm > max_val) {
                max_val = cm;
                inline for (0..16) |k| {
                    if (vf[k] >= cm) {
                        max_idx = i + k;
                        break;
                    }
                }
            }
        }
        while (i < n) : (i += 1) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }
        return @as(u32, @intCast(max_idx));
    }
};

/// Configuración de generación
pub const GenerationConfig = struct {
    max_new_tokens: usize,
    sampler: Sampler,
    eos_token: ?u32,
    pad_token: ?u32,
    stop_on_eos: bool = true,
    seed: u64 = 42,
};

/// Resultado de generación
pub const GenerationResult = struct {
    tokens: []u32,
    num_tokens_generated: usize,
    prefill_time_ms: f64,
    generation_time_ms: f64,
    tokens_per_second: f64,
};

/// Pipeline de inferencia autoregresiva
pub const InferencePipeline = struct {
    allocator: std.mem.Allocator,
    layers: []TransformerLayer,
    kv_manager: *KVCacheManager,
    hidden_dim: usize,
    vocab_size: usize,
    num_layers: usize,
    fa_config: FlashAttentionConfig,
    /// 7.1b-B: puntero al FA engine compartido entre todas las capas
    /// (una sola instancia en vez de 28 → ~648MB ahorrados en GPU pinned+device).
    /// Ownership: main.zig crea y libera; pipeline solo referencia.
    fa_engine: *AttentionEngine,
    /// 7.2 (lane-f): eps del RMSNorm final (model_config).
    rms_eps: f32 = 1e-5,
    /// IO para timestamps (std.Io.Clock.now).
    io: std.Io,

    const Self = @This();

    pub fn init(
        io: std.Io,
        allocator: std.mem.Allocator,
        layers: []TransformerLayer,
        kv_manager: *KVCacheManager,
        hidden_dim: usize,
        vocab_size: usize,
        fa_config: FlashAttentionConfig,
        /// 7.1b-B: puntero al FA engine compartido (ownership en main.zig).
        fa_engine: *AttentionEngine,
    ) Self {
        // 7.1e (lane-f): conectar el KV manager a CADA capa — antes
        // layers[i].kv_manager quedaba null (pool 256MB muerto; el decode
        // moría en getSequenceLen SequenceNotFound porque el prefijo
        // almacenado por el prefill nunca se recuperaba).
        for (layers) |*layer| {
            layer.kv_manager = kv_manager;
        }
        // 7.1a: registrar todas las capas en el F16Residency global
        // para habilitar evicción LRU de pesos f16.
        for (layers) |*layer| transformer.F16Residency.register(layer);
        return .{
            .allocator = allocator,
            .layers = layers,
            .kv_manager = kv_manager,
            .hidden_dim = hidden_dim,
            .vocab_size = vocab_size,
            .num_layers = @intCast(layers.len),
            .fa_config = fa_config,
            .fa_engine = fa_engine,
            .io = io,
        };
    }

    /// Prefill: procesar prompt completo
    pub const PrefillResult = struct { logits: Tensor(f32), last_token: u32 };
    pub fn prefill(
        self: Self,
        seq_id: u64,
        prompt_tokens: []const u32,
        emb_source: embedding.EmbSource,
        lm_head_source: embedding.LmHeadSource,
        matmul_engine: *matmul.MatmulEngine,
        output_norm: ?Tensor(f32),
    ) !PrefillResult {
        const scope = debugz.BreadcrumbScope.init(self.io, "pipeline", "prefill");
        defer scope.exit(self.io);

        const batch_size: usize = 1;
        const seq_len = prompt_tokens.len;

        // Embedding
        // 7.1c: ownership lineal — TODOS los tensors intermedios van a
        // owned_inputs y se liberan en el defer final; el `defer
        // hidden.deinit()` clásico reasignaba la variable y liberaba el
        // último layer_output dos veces (segfault en DebugAllocator).
        // F-4 (lane-f): stream f32 — el residual no redondea a f16 por
        // capa (evidencia: híbrido f32 = golden −0.257 lp; legacy f16 =
        // 1.8× PPL + overflow BitNet). Pesos/proyecciones siguen f16.
        var hidden = try Tensor(f32).alloc(self.allocator, &.{ batch_size, seq_len, self.hidden_dim });
        var owned_inputs: std.ArrayList(Tensor(f32)) = .empty;
        defer {
            for (owned_inputs.items) |*t| t.deinit();
            owned_inputs.deinit(self.allocator);
        }
        try owned_inputs.append(self.allocator, hidden);

        // 7.1e: la secuencia debe existir ANTES del primer append del
        // forward (storeKvCache hace appendTokensF16 por capa; sin esto
        // el prefill moría en SequenceNotFound silencioso).
        try self.kv_manager.createSequence(seq_id);

        // ─── RLT train (fase 2): captura de hidden states ─────────────
        // RLT_CAPTURE_DIR=<dir> activa el volcado de e[t] por capa para el
        // entrenamiento del adapter. RLT_CAPTURE_LAYERS=0,4,8 filtra capas.
        // α=0 implícito: el merge RLT no corre durante la captura (el trainer
        // necesita los hidden states del BASE). El flush va tras el loop.
        var cap: ?CaptureHandle = null;
        defer if (cap) |*c| c.deinit();
        if (debugz.dbg.rlt_capture_dir) |cap_dir| {
            cap = initCaptureHandle(self.allocator, cap_dir, seq_len, self.hidden_dim, self.num_layers, prompt_tokens) catch |err| blk: {
                debugz.dbg.printLevel(.info, "[rlt_capture] init falló: {s} — captura OFF\n", .{@errorName(err)});
                break :blk null;
            };
        }

        emb_source.lookupF32(prompt_tokens, batch_size, seq_len, &hidden);

        // Forward por capas
        // 7.1c: ownership lineal — cada layer_output entra a owned_inputs
        // (el embedding ya está); hidden SOLO reasigna el puntero lógico.
        for (self.layers, 0..) |*layer, layer_idx| {
            var layer_output = try Tensor(f32).alloc(self.allocator, hidden.shape);
            try owned_inputs.append(self.allocator, layer_output);

            layer.seq_id = seq_id;
            nvtx.rangePush("prefill:layer"); // lane-cuda UC-4.2: por-capa en nsys
            defer nvtx.rangePop();
            try layer.forward(hidden, &layer_output, 0, true);
            if (cap) |*c| {
                // e de esta capa = hidden stream [T*d] ANTES del forward
                // (la entrada). Copia directa del buffer contiguo.
                c.w.addLayer(layer_idx, hidden.data[0 .. seq_len * self.hidden_dim]);
            }
            hidden = layer_output;
        }

        // ─── RLT train: flush de la captura (un .rltcap por prefill) ───
        if (cap) |*c| {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "cap_{d}_{d}.rltcap", .{ seq_id, seq_len }) catch "cap.rltcap";
            c.w.flush(name) catch |err| {
                debugz.dbg.printLevel(.info, "[rlt_capture] flush falló: {s}\n", .{@errorName(err)});
            };
        }

        // 7.2 (lane-f): avanzar el contador de secuencia UNA vez tras el
        // loop de capas — todas las capas escribieron su chunk en el
        // offset current_len (compartido); el advance por-capa de antes
        // desplazaba el offset de la capa siguiente.
        for (0..seq_len) |_| try self.kv_manager.advanceSequence(seq_id);

        // LM Head sobre el último token
        // F-4 (lane-f): el slice del stream es f32 — el lm_head consume
        // la ÚLTIMA fila ya normalizada en f32 (rmsNorm genérico).
        var last_hidden = try Tensor(f32).alloc(self.allocator, &.{ batch_size, 1, self.hidden_dim });
        defer last_hidden.deinit();
        const last_offset = (seq_len - 1) * self.hidden_dim;
        @memcpy(last_hidden.data, hidden.data[last_offset..][0..self.hidden_dim]);

        // 7.2 (lane-f): RMSNorm final del modelo antes del lm_head — el
        // path legacy nunca lo aplicaba (logits sobre hidden crudo).
        if (output_norm) |gamma| {
            norm.rmsNorm(f32, f32, last_hidden, gamma, self.rms_eps, &last_hidden);
        }

        const last_hidden_2d = try last_hidden.reshape(&[_]usize{ batch_size, self.hidden_dim });
        defer {
            if (last_hidden_2d.allocator) |a| {
                a.free(last_hidden_2d.shape);
                a.free(last_hidden_2d.strides);
            }
        }

        // 7.2 (lane-f): el struct de retorno ADQUIERE el ownership de
        // logits — antes había un `defer logits.deinit()` que liberaba
        // el tensor devuelto (UAF: el caller leía memoria liberada).
        // 7.1d (lane-c): logits f32 directos del GEMV cuant-residente (sin
        // el paso intermedio f16 del round-trip clásico).
        var logits = try Tensor(f32).alloc(self.allocator, &.{ batch_size, self.vocab_size });

        try embedding.lmHeadForwardSourceF32(matmul_engine, last_hidden_2d, lm_head_source, self.hidden_dim, self.vocab_size, &logits);

        // Greedy sample del último token
        var last_token: u32 = 0;
        var max_val: f32 = -std.math.inf(f32);
        for (logits.data, 0..) |v, i| {
            if (v > max_val) {
                max_val = v;
                last_token = @as(u32, @intCast(i));
            }
        }

        return .{ .logits = logits, .last_token = last_token };
    }

    /// RLT (Recurrent Looped Transformer): prefill secuencial recurrente.
    /// Procesa tokens uno a uno usando la misma transición que decode,
    /// garantizando consistencia prompt-response. Más lento que el prefill
    /// paralelo (secuencial) pero produce estados intermedios idénticos.
    /// CLI: --recurrent-prefill. Default: OFF (prefill paralelo actual).
    pub fn recurrentPrefill(
        self: Self,
        seq_id: u64,
        prompt_tokens: []const u32,
        emb_source: embedding.EmbSource,
        lm_head_source: embedding.LmHeadSource,
        matmul_engine: *matmul.MatmulEngine,
        output_norm: ?Tensor(f32),
    ) !PrefillResult {
        const scope = debugz.BreadcrumbScope.init(self.io, "pipeline", "recurrentPrefill");
        defer scope.exit(self.io);

        const batch_size: usize = 1;
        const seq_len = prompt_tokens.len;

        try self.kv_manager.createSequence(seq_id);

        // Allocate single-token buffers
        var hidden = try Tensor(f32).alloc(self.allocator, &.{ batch_size, 1, self.hidden_dim });
        var layer_output = try Tensor(f32).alloc(self.allocator, &.{ batch_size, 1, self.hidden_dim });
        defer hidden.deinit();
        defer layer_output.deinit();

        // Process each token sequentially
        for (0..seq_len) |pos| {
            // Embed single token
            const tok_slice = prompt_tokens[pos .. pos + 1];
            emb_source.lookupF32(tok_slice, batch_size, 1, &hidden);

            // Forward through all layers (same path as decode)
            for (self.layers) |*layer| {
                layer.seq_id = seq_id;
                try layer.forward(hidden, &layer_output, pos, false);
                // Swap hidden ↔ layer_output
                const tmp_data = hidden.data;
                hidden.data = layer_output.data;
                layer_output.data = tmp_data;
            }

            try self.kv_manager.advanceSequence(seq_id);
        }

        // hidden.data now points to the last layer's output for the last token
        var last_hidden = try Tensor(f32).alloc(self.allocator, &.{ batch_size, 1, self.hidden_dim });
        defer last_hidden.deinit();
        @memcpy(last_hidden.data, hidden.data[0..self.hidden_dim]);

        if (output_norm) |gamma| {
            norm.rmsNorm(f32, f32, last_hidden, gamma, self.rms_eps, &last_hidden);
        }

        const last_hidden_2d = try last_hidden.reshape(&[_]usize{ batch_size, self.hidden_dim });
        defer {
            if (last_hidden_2d.allocator) |a| {
                a.free(last_hidden_2d.shape);
                a.free(last_hidden_2d.strides);
            }
        }

        var logits = try Tensor(f32).alloc(self.allocator, &.{ batch_size, self.vocab_size });
        try embedding.lmHeadForwardSourceF32(matmul_engine, last_hidden_2d, lm_head_source, self.hidden_dim, self.vocab_size, &logits);

        var last_token: u32 = 0;
        var max_val: f32 = -std.math.inf(f32);
        for (logits.data, 0..) |v, i| {
            if (v > max_val) {
                max_val = v;
                last_token = @as(u32, @intCast(i));
            }
        }

        return .{ .logits = logits, .last_token = last_token };
    }

    /// lane-kvc tANS C-a: prefill con logits por POSICIÓN (perplexity).
    /// Idéntico a `prefill` pero el lm_head corre sobre TODO el chunk
    /// [seq, hidden] → [seq, vocab] (1 GEMM batcheado) y el caller
    /// adquiere el ownership del tensor de logits devuelto.
    ///
    /// El logprob del token t_i (posición i del chunk) se lee de la fila
    /// i-1: logits[i-1] predice el token de la posición i. La fila seq-1
    /// predice el token SIGUIENTE (fuera del chunk — la usa el caller si
    /// encadena chunks).
    pub fn prefillPPL(
        self: Self,
        seq_id: u64,
        prompt_tokens: []const u32,
        embedding_table: Tensor(f16),
        lm_head_weight_t: Tensor(f16),
        matmul_engine: *matmul.MatmulEngine,
        output_norm: ?Tensor(f32),
    ) !Tensor(f16) {
        const scope = debugz.BreadcrumbScope.init(self.io, "pipeline", "prefillPPL");
        defer scope.exit(self.io);

        const batch_size: usize = 1;
        const seq_len = prompt_tokens.len;

        // F-4 (lane-f): stream f32 (idem prefill) — el residual no
        // redondea por capa; el ppl legado era el mayor afectado (1.8×).
        var hidden = try Tensor(f32).alloc(self.allocator, &.{ batch_size, seq_len, self.hidden_dim });
        var owned_inputs: std.ArrayList(Tensor(f32)) = .empty;
        defer {
            for (owned_inputs.items) |*t| t.deinit();
            owned_inputs.deinit(self.allocator);
        }
        try owned_inputs.append(self.allocator, hidden);

        try self.kv_manager.createSequence(seq_id);

        const emb = @import("embedding");
        emb.embeddingLookupF32(embedding_table, prompt_tokens, batch_size, seq_len, &hidden);

        for (self.layers) |*layer| {
            var layer_output = try Tensor(f32).alloc(self.allocator, hidden.shape);
            try owned_inputs.append(self.allocator, layer_output);

            layer.seq_id = seq_id;
            try layer.forward(hidden, &layer_output, 0, true);
            hidden = layer_output;
        }

        for (0..seq_len) |_| try self.kv_manager.advanceSequence(seq_id);

        // RMSNorm final sobre TODAS las posiciones (fila a fila — la
        // normalización es por-token, mismo gamma/eps que generate).
        // F-4: f32 directo sobre el stream.
        if (output_norm) |gamma| {
            var hn = hidden;
            norm.rmsNorm(f32, f32, hn, gamma, self.rms_eps, &hn);
        }

        // [1, seq, hidden] → [seq, hidden] y lm_head batcheado
        // F-4 (lane-f): stream f32 → cast a f16 del input del GEMM (el
        // batched lmHeadForward sigue homogéneo f16; logits f16 idénticos).
        var logits = try Tensor(f16).alloc(self.allocator, &.{ seq_len, self.vocab_size });
        const hidden_2d = try hidden.reshape(&[_]usize{ seq_len, self.hidden_dim });
        defer {
            if (hidden_2d.allocator) |a| {
                a.free(hidden_2d.shape);
                a.free(hidden_2d.strides);
            }
        }
        var hidden16_2d = try Tensor(f16).alloc(self.allocator, hidden_2d.shape);
        defer hidden16_2d.deinit();
        for (hidden_2d.data, hidden16_2d.data) |s, *d| d.* = @floatCast(s);
        try emb.lmHeadForward(matmul_engine, hidden16_2d, lm_head_weight_t, &logits);
        return logits;
    }

    /// Generar tokens autoregresivamente
    pub fn generate(
        self: Self,
        seq_id: u64,
        first_token: u32,
        emb_source: embedding.EmbSource,
        lm_head_source: embedding.LmHeadSource,
        matmul_engine: *matmul.MatmulEngine,
        config: GenerationConfig,
        output_norm: ?Tensor(f32),
    ) !GenerationResult {
        const scope = debugz.BreadcrumbScope.init(self.io, "pipeline", "generate");
        defer scope.exit(self.io);

        var tokens: std.ArrayList(u32) = .empty;
        errdefer tokens.deinit(self.allocator);
        try tokens.append(self.allocator, first_token);

        var rng = std.Random.Xoshiro256.init(config.seed);

        const start_time = @import("time").Timer.now();
        var current_pos = try self.kv_manager.getSequenceLen(seq_id);

        // Persistent decode buffers — preasignados una sola vez para evitar allocs
        // por token en el hot path.
        var hidden = try Tensor(f32).alloc(self.allocator, &.{ 1, 1, self.hidden_dim });
        defer hidden.deinit();
        var layer_output = try Tensor(f32).alloc(self.allocator, &.{ 1, 1, self.hidden_dim });
        defer layer_output.deinit();
        var logits = try Tensor(f32).alloc(self.allocator, &.{ 1, self.vocab_size });
        defer logits.deinit();

        var hidden_2d_shape = try self.allocator.alloc(usize, 2);
        defer self.allocator.free(hidden_2d_shape);
        var hidden_2d_strides = try self.allocator.alloc(usize, 2);
        defer self.allocator.free(hidden_2d_strides);
        hidden_2d_shape[0] = 1;
        hidden_2d_shape[1] = self.hidden_dim;
        hidden_2d_strides[0] = self.hidden_dim;
        hidden_2d_strides[1] = 1;

        var sampler = config.sampler;
        try sampler.initScratch(self.allocator, self.vocab_size);
        defer sampler.deinitScratch(self.allocator);

        for (0..config.max_new_tokens) |_| {
            const last_token = tokens.items[tokens.items.len - 1];

            nvtx.rangePush("decode:step"); // lane-cuda UC-4.2: 1 step = 1 token en nsys
            defer nvtx.rangePop();

            // Reset persistent buffers for this token.
            hidden.data.len = 1 * 1 * self.hidden_dim;
            logits.data.len = self.vocab_size;

            // Embedding de 1 token
            // F-4 (lane-f): stream f32 (idem prefill).
            const single_token = &[_]u32{last_token};
            emb_source.lookupF32(single_token, 1, 1, &hidden);

            // Forward por capas
            for (self.layers) |*layer| {
                layer_output.data.len = hidden.data.len;
                layer.seq_id = seq_id;
                nvtx.rangePush("decode:layer"); // lane-cuda UC-4.2: por-capa en nsys
                defer nvtx.rangePop();
                try layer.forward(hidden, &layer_output, current_pos, false);
                hidden = layer_output;
            }

            // 7.2 (lane-f): avanzar el contador UNA vez tras el loop de
            // capas (simétrico al prefill; antes el advance dentro del
            // storeKvCache por capa desplazaba los offsets).
            try self.kv_manager.advanceSequence(seq_id);

            // LM Head
            // 7.2 (lane-f): RMSNorm final antes del lm_head (idem prefill).
            nvtx.rangePush("decode:lm_head"); // lane-cuda UC-4.2
            defer nvtx.rangePop();
            if (output_norm) |gamma| {
                norm.rmsNorm(f32, f32, hidden, gamma, self.rms_eps, &hidden);
            }

            const hidden_2d = Tensor(f32){
                .data = hidden.data[0..self.hidden_dim],
                .shape = hidden_2d_shape,
                .strides = hidden_2d_strides,
                .offset = 0,
                .allocator = null,
                .owns_data = false,
            };

            try embedding.lmHeadForwardSourceF32(matmul_engine, hidden_2d, lm_head_source, self.hidden_dim, self.vocab_size, &logits);

            // Samplear
            const next_token = sampler.sample(logits.data, &rng, tokens.items);
            try tokens.append(self.allocator, next_token);
            current_pos += 1;

            if (config.stop_on_eos and config.eos_token != null and next_token == config.eos_token.?) {
                break;
            }
        }

        const end_time = @import("time").Timer.now();
        const gen_time_ms = @as(f64, @floatFromInt(@divTrunc(end_time - start_time, std.time.ns_per_ms)));
        const num_gen = tokens.items.len - 1; // excluir first_token

        return GenerationResult{
            .tokens = try tokens.toOwnedSlice(self.allocator),
            .num_tokens_generated = num_gen,
            .prefill_time_ms = 0, // Calculado externamente
            .generation_time_ms = gen_time_ms,
            .tokens_per_second = if (gen_time_ms > 0) @as(f64, @floatFromInt(num_gen)) / (gen_time_ms / 1000.0) else 0,
        };
    }

    pub fn deinitResult(self: Self, result: *GenerationResult) void {
        self.allocator.free(result.tokens);
        result.* = undefined;
    }
};
