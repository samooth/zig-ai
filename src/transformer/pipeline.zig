const std = @import("std");
const Tensor = @import("core").Tensor;
const matmul = @import("matmul");
const fa = @import("fa");
const transformer = @import("transformer");
const kvcache = @import("kv_cache");

const FlashAttentionConfig = fa.fa_config.FlashAttentionConfig;
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

    /// Muestrea un token dado los logits y el historial ya generado
    /// (usado para repetition penalty).
    pub fn sample(self: Sampler, logits: []const f32, rng: *std.Random.Xoshiro256, history: []const u32) u32 {
        const vocab = logits.len;

        // 0. Fast path: top_p = 1 (sin recorte de núcleo), sin top-k y sin
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

        const alloc = std.heap.page_allocator;

        var work = alloc.alloc(f32, vocab) catch @panic("sampler OOM");
        defer alloc.free(work);
        @memcpy(work, logits);

        // 1. Repetition penalty: penaliza tokens ya generados
        if (self.repetition_penalty != 1.0) {
            for (history) |tok| {
                if (tok < vocab) {
                    if (work[tok] > 0) work[tok] /= self.repetition_penalty
                    else if (work[tok] < 0) work[tok] *= self.repetition_penalty;
                }
            }
        }

        // 2. Greedy si temperature <= 0
        if (self.temperature <= 0) {
            var max_idx: usize = 0;
            var max_val: f32 = -std.math.inf(f32);
            for (work, 0..) |v, i| {
                if (v > max_val) { max_val = v; max_idx = i; }
            }
            return @as(u32, @intCast(max_idx));
        }

        // 3. Temperature scaling
        const inv_t = 1.0 / self.temperature;
        for (work) |*v| v.* *= inv_t;

        // 4. Top-k: descarta lo que está por debajo del k-ésimo valor
        if (self.top_k > 0 and self.top_k < vocab) {
            var vals = alloc.alloc(f32, vocab) catch @panic("sampler OOM");
            defer alloc.free(vals);
            @memcpy(vals, work);
            const k = self.top_k;
            for (0..k) |i| {
                var max_idx = i;
                var max_val = vals[i];
                for (i..vocab) |j| {
                    if (vals[j] > max_val) { max_val = vals[j]; max_idx = j; }
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
        var idx = alloc.alloc(usize, vocab) catch @panic("sampler OOM");
        defer alloc.free(idx);
        for (0..vocab) |i| idx[i] = i;

        // Candidatos (work != -inf)
        var cand_count: usize = 0;
        for (0..vocab) |i| {
            if (work[i] != -std.math.inf(f32)) { idx[cand_count] = i; cand_count += 1; }
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
        var probs = alloc.alloc(f32, cand_count) catch @panic("sampler OOM");
        defer alloc.free(probs);
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
            if (cumsum >= self.top_p) { cutoff = i + 1; break; }
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

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        layers: []TransformerLayer,
        kv_manager: *KVCacheManager,
        hidden_dim: usize,
        vocab_size: usize,
        fa_config: FlashAttentionConfig,
    ) Self {
        return .{
            .allocator = allocator,
            .layers = layers,
            .kv_manager = kv_manager,
            .hidden_dim = hidden_dim,
            .vocab_size = vocab_size,
            .num_layers = @intCast(layers.len),
            .fa_config = fa_config,
        };
    }

    /// Prefill: procesar prompt completo
    pub fn prefill(
        self: Self,
        seq_id: u64,
        prompt_tokens: []const u32,
        embedding_table: Tensor(f16),
        lm_head_weight_t: Tensor(f16),
        matmul_engine: *matmul.MatmulEngine,
    ) !struct { logits: Tensor(f16), last_token: u32 } {
        const batch_size: usize = 1;
        const seq_len = prompt_tokens.len;

        // Embedding
        var hidden = try Tensor(f16).alloc(self.allocator, &.{ batch_size, seq_len, self.hidden_dim });
        defer hidden.deinit();

            const emb = @import("embedding");
        emb.embeddingLookup(embedding_table, prompt_tokens, batch_size, seq_len, &hidden);

        // Forward por capas
        for (self.layers, 0..) |*layer, l| {
            var layer_output = try Tensor(f16).alloc(self.allocator, hidden.shape);
            defer if (l < self.layers.len - 1) layer_output.deinit();

            layer.seq_id = seq_id;
            try layer.forward(hidden, &layer_output, 0, true);
            hidden = layer_output;
        }

        // LM Head sobre el último token
        var last_hidden = try Tensor(f16).alloc(self.allocator, &.{ batch_size, 1, self.hidden_dim });
        defer last_hidden.deinit();
        const last_offset = (seq_len - 1) * self.hidden_dim;
        @memcpy(last_hidden.data, hidden.data[last_offset..][0..self.hidden_dim]);

        const last_hidden_2d = try last_hidden.reshape(&[_]usize{ batch_size, self.hidden_dim });
        defer { if (last_hidden_2d.allocator) |a| { a.free(last_hidden_2d.shape); a.free(last_hidden_2d.strides); } }

        var logits = try Tensor(f16).alloc(self.allocator, &.{ batch_size, self.vocab_size });
        defer logits.deinit();

        try emb.lmHeadForward(matmul_engine, last_hidden_2d, lm_head_weight_t, &logits);

        // Greedy sample del último token
        var logits_f32 = try self.allocator.alloc(f32, self.vocab_size);
        defer self.allocator.free(logits_f32);
        for (logits.data, 0..) |v, i| logits_f32[i] = @as(f32, @floatCast(v));

        var last_token: u32 = 0;
        var max_val: f32 = -std.math.inf(f32);
        for (logits_f32, 0..) |v, i| {
            if (v > max_val) { max_val = v; last_token = @as(u32, @intCast(i)); }
        }

        return .{ .logits = logits, .last_token = last_token };
    }

    /// Generar tokens autoregresivamente
    pub fn generate(
        self: Self,
        seq_id: u64,
        first_token: u32,
        embedding_table: Tensor(f16),
        lm_head_weight_t: Tensor(f16),
        matmul_engine: *matmul.MatmulEngine,
        config: GenerationConfig,
    ) !GenerationResult {
        var tokens: std.ArrayList(u32) = .empty;
        errdefer tokens.deinit(self.allocator);
        try tokens.append(self.allocator, first_token);

        var rng = std.Random.Xoshiro256.init(config.seed);

        const start_time = @import("time").Timer.now();
        var current_pos = try self.kv_manager.getSequenceLen(seq_id);

        for (0..config.max_new_tokens) |_| {
            const last_token = tokens.items[tokens.items.len - 1];

            // Embedding de 1 token
            var hidden = try Tensor(f16).alloc(self.allocator, &.{ 1, 1, self.hidden_dim });
            defer hidden.deinit();

        const emb = @import("embedding");
            const single_token = &[_]u32{last_token};
            emb.embeddingLookup(embedding_table, single_token, 1, 1, &hidden);

            // Forward por capas
            for (self.layers, 0..) |*layer, l| {
                var layer_output = try Tensor(f16).alloc(self.allocator, hidden.shape);
                defer if (l < self.layers.len - 1) layer_output.deinit();

                layer.seq_id = seq_id;
                try layer.forward(hidden, &layer_output, current_pos, false);
                hidden = layer_output;
            }

            // LM Head
            const hidden_2d = try hidden.reshape(&[_]usize{ 1, self.hidden_dim });
            defer { if (hidden_2d.allocator) |a| { a.free(hidden_2d.shape); a.free(hidden_2d.strides); } }

            var logits = try Tensor(f16).alloc(self.allocator, &.{ 1, self.vocab_size });
            defer logits.deinit();

            try emb.lmHeadForward(matmul_engine, hidden_2d, lm_head_weight_t, &logits);

            // Samplear
            var logits_f32 = try self.allocator.alloc(f32, self.vocab_size);
            defer self.allocator.free(logits_f32);
            for (logits.data, 0..) |v, i| logits_f32[i] = @as(f32, @floatCast(v));

            const next_token = config.sampler.sample(logits_f32, &rng, tokens.items);
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
