const std = @import("std");
const Tensor = @import("core").Tensor;
/// Sampling strategies for language model generation
pub const Sampler = union(enum) {
    greedy: GreedySampler,
    top_k: TopKSampler,
    top_p: TopPSampler,
    temperature: TemperatureSampler,
    pub fn sample(self: Sampler, logits: Tensor(f16)) u32 {
        return switch (self) {
            .greedy => |s| s.sample(logits),
            .top_k => |s| s.sample(logits),
            .top_p => |s| s.sample(logits),
            .temperature => |s| s.sample(logits),
        };
    }
};
pub const GreedySampler = struct {
    pub fn sample(_: GreedySampler, logits: Tensor(f16)) u32 {
        var max_idx: usize = 0;
        var max_val: f32 = -std.math.inf(f32);
        for (logits.data, 0..) |v, i| {
            const f = @as(f32, @floatCast(v));
            if (f > max_val) { max_val = f; max_idx = i; }
        }
        return @as(u32, @intCast(max_idx % logits.data.len));
    }
};
pub const TopKSampler = struct {
    k: usize,
    rng: std.Random.Xoshiro256,
    pub fn init(k: usize, seed: u64) TopKSampler {
        return .{ .k = k, .rng = std.Random.Xoshiro256.init(seed) };
    }
    pub fn sample(self: *TopKSampler, logits: Tensor(f16)) u32 {
        const vocab_size = logits.data.len;
        const k = @min(self.k, vocab_size);
        // Find top-k indices
        var indices = std.ArrayList(usize).init(std.heap.page_allocator);
        defer indices.deinit();
        for (0..vocab_size) |i| try indices.append(i);
        // Simple bubble sort for top-k
        var top_indices: [100]usize = undefined;
        var top_values: [100]f32 = undefined;
        const actual_k = @min(k, 100);
        for (0..actual_k) |i| {
            var max_idx = i;
            var max_val = @as(f32, @floatCast(logits.data[i]));
            for (i..vocab_size) |j| {
                const val = @as(f32, @floatCast(logits.data[j]));
                if (val > max_val) { max_val = val; max_idx = j; }
            }
            top_indices[i] = max_idx;
            top_values[i] = max_val;
        }
        // Softmax over top-k
        var sum: f32 = 0;
        for (0..actual_k) |i| sum += @exp(top_values[i]);
        var r = self.rng.random().float(f32);
        var cumsum: f32 = 0;
        for (0..actual_k) |i| {
            cumsum += @exp(top_values[i]) / sum;
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
    pub fn sample(self: *TopPSampler, logits: Tensor(f16)) u32 {
        // Simplified: sort and accumulate until p threshold
        const vocab_size = logits.data.len;
        var indices = std.ArrayList(usize).init(std.heap.page_allocator);
        defer indices.deinit();
        for (0..vocab_size) |i| try indices.append(i);
        // Sort by value descending
        // (simplified - just use greedy for now)
        _ = self;
        var max_idx: usize = 0;
        var max_val: f32 = -std.math.inf(f32);
        for (logits.data, 0..) |v, i| {
            const f = @as(f32, @floatCast(v));
            if (f > max_val) { max_val = f; max_idx = i; }
        }
        return @as(u32, @intCast(max_idx));
    }
};
pub const TemperatureSampler = struct {
    temperature: f32,
    rng: std.Random.Xoshiro256,
    pub fn init(temperature: f32, seed: u64) TemperatureSampler {
        return .{ .temperature = temperature, .rng = std.Random.Xoshiro256.init(seed) };
    }
    pub fn sample(self: *TemperatureSampler, logits: Tensor(f16)) u32 {
        const vocab_size = logits.data.len;
        // Apply temperature
        var probs = std.ArrayList(f32).init(std.heap.page_allocator);
        defer probs.deinit();
        var max_val: f32 = -std.math.inf(f32);
        for (logits.data) |v| {
            const f = @as(f32, @floatCast(v));
            if (f > max_val) max_val = f;
        }
        var sum: f32 = 0;
        for (logits.data) |v| {
            const p = @exp((@as(f32, @floatCast(v)) - max_val) / self.temperature);
            try probs.append(p);
            sum += p;
        }
        // Normalize
        for (probs.items) |*p| p.* /= sum;
        // Sample
        var r = self.rng.random().float(f32);
        var cumsum: f32 = 0;
        for (probs.items, 0..) |p, i| {
            cumsum += p;
            if (r <= cumsum) return @as(u32, @intCast(i));
        }
        return @as(u32, @intCast(vocab_size - 1));
    }
};
