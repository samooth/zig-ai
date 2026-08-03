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

/// Estrategias de sampling
pub const Sampler = union(enum) {
    greedy: GreedySampler,
    top_k: TopKSampler,
    top_p: TopPSampler,
    temperature: TemperatureSampler,

    pub fn sample(self: Sampler, logits: []const f32, rng: ?*std.Random.Xoshiro256) u32 {
        return switch (self) {
            .greedy => |s| s.sample(logits),
            .top_k => |*s| s.sample(logits, rng.?),
            .top_p => |*s| s.sample(logits, rng.?),
            .temperature => |*s| s.sample(logits, rng.?),
        };
    }
};

pub const GreedySampler = struct {
    pub fn sample(_: GreedySampler, logits: []const f32) u32 {
        var max_idx: usize = 0;
        var max_val: f32 = -std.math.inf(f32);
        for (logits, 0..) |v, i| {
            if (v > max_val) { max_val = v; max_idx = i; }
        }
        return @as(u32, @intCast(max_idx));
    }
};

pub const TopKSampler = struct {
    k: usize,
    rng: std.Random.Xoshiro256,

    pub fn init(k: usize, seed: u64) TopKSampler {
        return .{ .k = k, .rng = std.Random.Xoshiro256.init(seed) };
    }

    pub fn sample(self: *TopKSampler, logits: []const f32, rng: *std.Random.Xoshiro256) u32 {
        const vocab_size = logits.len;
        const k = @min(self.k, vocab_size);

        // Encontrar top-k con heap simple (O(vocab * k))
        var top_indices: [100]usize = undefined;
        var top_values: [100]f32 = undefined;
        const actual_k = @min(k, 100);

        for (0..actual_k) |i| {
            var max_idx = i;
            var max_val = logits[i];
            for (i..vocab_size) |j| {
                if (logits[j] > max_val) { max_val = logits[j]; max_idx = j; }
            }
            top_indices[i] = max_idx;
            top_values[i] = max_val;
        }

        // Softmax sobre top-k
        var max_val: f32 = -std.math.inf(f32);
        for (0..actual_k) |i| max_val = @max(max_val, top_values[i]);

        var sum: f32 = 0;
        for (0..actual_k) |i| {
            top_values[i] = @exp(top_values[i] - max_val);
            sum += top_values[i];
        }

        var r = rng.random().float(f32);
        var cumsum: f32 = 0;
        for (0..actual_k) |i| {
            cumsum += top_values[i] / sum;
            if (r <= cumsum) return @as(u32, @intCast(top_indices[i]));
        }
        return @as(u32, @intCast(top_indices[actual_k - 1]));
    }
};

pub const TopPSampler = struct {
    p: f32,
    rng: std.Random.Xoshiro256,

    pub fn init(p: f32, seed: u64) TopPSampler {
        return .{ .p = p, .rng = std.Random.Xoshiro256.init(seed) };
    }

    pub fn sample(self: *TopPSampler, logits: []const f32, rng: *std.Random.Xoshiro256) u32 {
        const vocab_size = logits.len;

        // Crear índices ordenados por valor descendente (bubble sort simple para vocab pequeño)
        var indices = std.heap.page_allocator.alloc(usize, vocab_size) catch unreachable;
        defer std.heap.page_allocator.free(indices);
        for (0..vocab_size) |i| indices[i] = i;

        // Ordenar por valor descendente
        for (0..vocab_size) |i| {
            for (0..vocab_size - i - 1) |j| {
                if (logits[indices[j]] < logits[indices[j + 1]]) {
                    const tmp = indices[j];
                    indices[j] = indices[j + 1];
                    indices[j + 1] = tmp;
                }
            }
        }

        // Calcular softmax
        var max_val = logits[indices[0]];
        var probs = std.heap.page_allocator.alloc(f32, vocab_size) catch unreachable;
        defer std.heap.page_allocator.free(probs);

        var sum: f32 = 0;
        for (0..vocab_size) |i| {
            probs[i] = @exp(logits[indices[i]] - max_val);
            sum += probs[i];
        }
        for (probs) |*p| p.* /= sum;

        // Acumular hasta p
        var cumsum: f32 = 0;
        var cutoff: usize = vocab_size;
        for (0..vocab_size) |i| {
            cumsum += probs[i];
            if (cumsum >= self.p) { cutoff = i + 1; break; }
        }

        // Samplear del top-p
        var sub_sum: f32 = 0;
        for (0..cutoff) |i| sub_sum += probs[i];

        var r = rng.random().float(f32) * sub_sum;
        cumsum = 0;
        for (0..cutoff) |i| {
            cumsum += probs[i];
            if (r <= cumsum) return @as(u32, @intCast(indices[i]));
        }
        return @as(u32, @intCast(indices[cutoff - 1]));
    }
};

pub const TemperatureSampler = struct {
    temperature: f32,
    rng: std.Random.Xoshiro256,

    pub fn init(temperature: f32, seed: u64) TemperatureSampler {
        return .{ .temperature = temperature, .rng = std.Random.Xoshiro256.init(seed) };
    }

    pub fn sample(self: *TemperatureSampler, logits: []const f32, rng: *std.Random.Xoshiro256) u32 {
        const vocab_size = logits.len;

        var max_val: f32 = -std.math.inf(f32);
        for (logits) |v| max_val = @max(max_val, v);

        var probs = std.heap.page_allocator.alloc(f32, vocab_size) catch unreachable;
        defer std.heap.page_allocator.free(probs);

        var sum: f32 = 0;
        for (logits, 0..) |v, i| {
            probs[i] = @exp((v - max_val) / self.temperature);
            sum += probs[i];
        }
        for (probs) |*p| p.* /= sum;

        var r = rng.random().float(f32);
        var cumsum: f32 = 0;
        for (probs, 0..) |p, i| {
            cumsum += p;
            if (r <= cumsum) return @as(u32, @intCast(i));
        }
        return @as(u32, @intCast(vocab_size - 1));
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

        const emb = @import("transformer/embedding.zig");
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

        var last_hidden_2d = try last_hidden.reshape(&[_]usize{ batch_size, self.hidden_dim });
        defer { if (last_hidden_2d.allocator) |a| { a.free(last_hidden_2d.shape); a.free(last_hidden_2d.strides); } }

        var logits = try Tensor(f16).alloc(self.allocator, &.{ batch_size, self.vocab_size });
        defer logits.deinit();

        emb.lmHeadForward(matmul_engine, last_hidden_2d, lm_head_weight_t, &logits);

        // Greedy sample del último token
        var logits_f32 = try self.allocator.alloc(f32, self.vocab_size);
        defer self.allocator.free(logits_f32);
        for (logits.data, 0..) |v, i| logits_f32[i] = @as(f32, @floatCast(v));

        const sampler = GreedySampler{};
        const last_token = sampler.sample(logits_f32);

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
        var tokens = std.ArrayList(u32).init(self.allocator);
        errdefer tokens.deinit();
        try tokens.append(first_token);

        var rng = std.Random.Xoshiro256.init(config.seed);

        const start_time = std.time.milliTimestamp();
        var current_pos = try self.kv_manager.getSequenceLen(seq_id);

        for (0..config.max_new_tokens) |_| {
            const last_token = tokens.items[tokens.items.len - 1];

            // Embedding de 1 token
            var hidden = try Tensor(f16).alloc(self.allocator, &.{ 1, 1, self.hidden_dim });
            defer hidden.deinit();

            const emb = @import("transformer/embedding.zig");
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
            var hidden_2d = try hidden.reshape(&[_]usize{ 1, self.hidden_dim });
            defer { if (hidden_2d.allocator) |a| { a.free(hidden_2d.shape); a.free(hidden_2d.strides); } }

            var logits = try Tensor(f16).alloc(self.allocator, &.{ 1, self.vocab_size });
            defer logits.deinit();

            emb.lmHeadForward(matmul_engine, hidden_2d, lm_head_weight_t, &logits);

            // Samplear
            var logits_f32 = try self.allocator.alloc(f32, self.vocab_size);
            defer self.allocator.free(logits_f32);
            for (logits.data, 0..) |v, i| logits_f32[i] = @as(f32, @floatCast(v));

            const next_token = config.sampler.sample(logits_f32, &rng);
            try tokens.append(next_token);
            current_pos += 1;

            if (config.stop_on_eos and config.eos_token != null and next_token == config.eos_token.?) {
                break;
            }
        }

        const end_time = std.time.milliTimestamp();
        const gen_time_ms = @as(f64, @floatFromInt(end_time - start_time));
        const num_gen = tokens.items.len - 1; // excluir first_token

        return GenerationResult{
            .tokens = try tokens.toOwnedSlice(),
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
