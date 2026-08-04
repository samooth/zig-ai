const std = @import("std");
const Tensor = @import("core").Tensor;

/// Aplica RoPE a Q y K
pub fn applyRoPE(Q: *Tensor(f16), K: *Tensor(f16), pos: usize, d: usize) void {
    const N = Q.shape[2];
    const num_heads = Q.shape[1];
    const batch_size = Q.shape[0];
    for (0..batch_size) |b| {
        for (0..num_heads) |h| {
            for (0..N) |m| {
                const global_pos = pos + m;
                var i: usize = 0;
                while (i < d / 2) : (i += 1) {
                    const theta = @as(f32, @floatFromInt(global_pos)) *
                        std.math.pow(f32, 10000.0, -2.0 * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(d)));
                    const cos_val = @cos(theta);
                    const sin_val = @sin(theta);
                    const idx_even = ((b * num_heads + h) * N + m) * d + 2 * i;
                    const idx_odd = idx_even + 1;
                    const q_even = @as(f32, @floatCast(Q.data[idx_even]));
                    const q_odd = @as(f32, @floatCast(Q.data[idx_odd]));
                    Q.data[idx_even] = @floatCast(q_even * cos_val - q_odd * sin_val);
                    Q.data[idx_odd] = @floatCast(q_even * sin_val + q_odd * cos_val);
                    const k_even = @as(f32, @floatCast(K.data[idx_even]));
                    const k_odd = @as(f32, @floatCast(K.data[idx_odd]));
                    K.data[idx_even] = @floatCast(k_even * cos_val - k_odd * sin_val);
                    K.data[idx_odd] = @floatCast(k_even * sin_val + k_odd * cos_val);
                }
            }
        }
    }
}

pub fn isCausalAllowed(q_pos: usize, k_pos: usize) bool { return k_pos <= q_pos; }

pub fn applyCausalMask(scores: []f32, N: usize) void {
    for (0..N) |i| {
        for (0..N) |j| {
            if (j > i) scores[i * N + j] = -std.math.inf(f32);
        }
    }
}

pub fn softmax(input: []const f32, output: []f32) void {
    std.debug.assert(input.len == output.len);
    var max_val: f32 = -std.math.inf(f32);
    for (input) |v| {
        if (v > max_val) max_val = v;
    }
    var sum: f32 = 0;
    for (input, 0..) |v, i| { output[i] = @exp(v - max_val); sum += output[i]; }
    for (output) |*v| v.* /= sum;
}

pub const OnlineSoftmax = struct {
    m: f32 = -std.math.inf(f32), l: f32 = 0,
    pub fn update(self: *OnlineSoftmax, scores: []const f32) void {
        var m_local: f32 = -std.math.inf(f32);
        for (scores) |s| {
            if (s > m_local) m_local = s;
        }
        const m_new = if (m_local > self.m) m_local else self.m;
        const alpha = @exp(self.m - m_new);
        var l_local: f32 = 0;
        for (scores) |s| l_local += @exp(s - m_new);
        self.l = self.l * alpha + l_local;
        self.m = m_new;
    }
    pub fn normalize(self: OnlineSoftmax, accum: []f32) void {
        for (accum) |*v| v.* = @exp(v.* - self.m) / self.l;
    }
};

pub fn initNormal(tensor: *Tensor(f16), mean: f32, stddev: f32, seed: u64) void {
    var rng = std.Random.DefaultPrng.init(seed);
    for (tensor.data) |*v| {
        const r1 = rng.random().float(f32);
        const r2 = rng.random().float(f32);
        const z = @sqrt(-2.0 * @log(r1)) * @cos(2.0 * std.math.pi * r2);
        v.* = @floatCast(mean + stddev * z);
    }
}

pub fn initUniform(tensor: *Tensor(f16), min: f32, max: f32, seed: u64) void {
    var rng = std.Random.DefaultPrng.init(seed);
    for (tensor.data) |*v| {
        const u = rng.random().float(f32);
        v.* = @floatCast(min + u * (max - min));
    }
}

pub inline fn f16ToF32(v: f16) f32 { return @as(f32, @floatCast(v)); }
pub inline fn f32ToF16(v: f32) f16 { return @floatCast(v); }

pub fn benchmark(comptime func: anytype, args: anytype, iterations: usize) u64 {
    const timer = @import("time").Timer.start();
    for (0..iterations) |_| { @call(.auto, func, args); }
    return @intCast(timer.read() / @as(i128, @intCast(iterations)));
}
