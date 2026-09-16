//! RLT (Recurrent Looped Transformer) training layer — forward + backward manual.
//!
//! Forward (bit-exact con el engine, ver hybrid_layer.zig mergeFeedback y el
//! kernel CUDA mergeFeedbackKernel):
//!   n = RMSNorm(s_prev)                     // eps 1e-6 (paridad engine)
//!   r = [e ‖ n]                             // [2d]
//!   gate = sigmoid(W_gate @ r)              // [d], W_gate [d,2d] row-major
//!   mixed = W_state @ n                     // [d], W_state [d,d] row-major
//!   u = e + alpha * gate ⊙ mixed            // [d]
//!
//! Backward (truncated BPTT depth-1): el CALLER pasa dL/du. Se computan
//! dL/dW_gate, dL/dW_state, dL/dalpha para el paso actual. s_prev se trata
//! como constante (no propagamos dL/ds_prev a través del tiempo — v1);
//! `buf.d_rmsnorm_s` queda como scratch para un futuro BPTT profundo.
//!
//! NOTA: como el forward debe reproducir EXACTAMENTE el merge del engine en
//! inferencia (layout, eps, orden de ops), cualquier cambio aquí debe
//! espejarse en src/transformer/hybrid_layer.zig y src/cuda/layer_kernels.cu.
const std = @import("std");
const math = std.math;
const mem = std.mem;
const assert = std.debug.assert;

/// eps del RMSNorm — 1e-6 para paridad bit-exacta con mergeFeedback
/// (hybrid_layer.zig: `@sqrt(sum_sq / d + 1e-6)`). NO cambiar sin actualizar
/// el engine y el kernel CUDA.
pub const RMS_EPS: f32 = 1e-6;

pub const RltParams = struct {
    d: usize, // embedding dim
    alpha: f32 = 0.15, // merge coefficient
    learn_alpha: bool = false, // whether alpha is also trainable
};

/// Buffers for a single RLT layer during training.
pub const RltBuffers = struct {
    params: RltParams,

    // Forward scratch (all [d])
    rmsnorm_s: []f32, // RMSNorm(s_prev)
    r: []f32, // [e ‖ rmsnorm_s], shape [2*d]
    gate_logits: []f32, // pre-sigmoid
    gate: []f32, // sigmoid output
    mixed: []f32, // W_state @ rmsnorm_s
    u: []f32, // output: e + alpha * gate * mixed

    // Backward scratch
    d_gate_logits: []f32, // dL/d(gate_logits)
    d_r: []f32, // dL/d(r), shape [2*d]
    d_rmsnorm_s: []f32, // accumulated dL/d(rmsnorm_s)

    // Gradient accumulators for weights (zeroed each step)
    dw_gate: []f32, // [d, 2*d]  outer product accumulator
    dw_state: []f32, // [d, d]
    d_alpha: f32,

    // AdamW state (for each param: m and v running averages)
    m_w_gate: []f32,
    v_w_gate: []f32,
    m_w_state: []f32,
    v_w_state: []f32,
    m_alpha: f32,
    v_alpha: f32,

    pub fn alloc(allocator: std.mem.Allocator, p: RltParams) !RltBuffers {
        const d = p.d;
        const dd = d * d;
        const d2d = d * 2 * d;
        // Inicialización explícita para evitar "local variable is never
        // mutated" con `try` dentro de struct literal en Zig 0.16.
        var ret = RltBuffers{
            .params = p,
            .rmsnorm_s = undefined,
            .r = undefined,
            .gate_logits = undefined,
            .gate = undefined,
            .mixed = undefined,
            .u = undefined,
            .d_gate_logits = undefined,
            .d_r = undefined,
            .d_rmsnorm_s = undefined,
            .dw_gate = undefined,
            .dw_state = undefined,
            .d_alpha = 0,
            .m_w_gate = undefined,
            .v_w_gate = undefined,
            .m_w_state = undefined,
            .v_w_state = undefined,
            .m_alpha = 0,
            .v_alpha = 0,
        };
        ret.rmsnorm_s = try allocator.alloc(f32, d);
        ret.r = try allocator.alloc(f32, 2 * d);
        ret.gate_logits = try allocator.alloc(f32, d);
        ret.gate = try allocator.alloc(f32, d);
        ret.mixed = try allocator.alloc(f32, d);
        ret.u = try allocator.alloc(f32, d);
        ret.d_gate_logits = try allocator.alloc(f32, d);
        ret.d_r = try allocator.alloc(f32, 2 * d);
        ret.d_rmsnorm_s = try allocator.alloc(f32, d);
        ret.dw_gate = try allocator.alloc(f32, d2d);
        ret.dw_state = try allocator.alloc(f32, dd);
        ret.m_w_gate = try allocator.alloc(f32, d2d);
        ret.v_w_gate = try allocator.alloc(f32, d2d);
        ret.m_w_state = try allocator.alloc(f32, dd);
        ret.v_w_state = try allocator.alloc(f32, dd);
        // Momentos de AdamW DEBEN empezar en 0: malloc garbage (NaN/inf
        // residual) envenena m/v y el primer update produce NaN en pesos
        // con gradiente cero (repro: tests/scratch.zig 2026-09-14).
        @memset(ret.m_w_gate, 0);
        @memset(ret.v_w_gate, 0);
        @memset(ret.m_w_state, 0);
        @memset(ret.v_w_state, 0);
        return ret;
    }

    pub fn deinit(self: *RltBuffers, allocator: std.mem.Allocator) void {
        allocator.free(self.rmsnorm_s);
        allocator.free(self.r);
        allocator.free(self.gate_logits);
        allocator.free(self.gate);
        allocator.free(self.mixed);
        allocator.free(self.u);
        allocator.free(self.d_gate_logits);
        allocator.free(self.d_r);
        allocator.free(self.d_rmsnorm_s);
        allocator.free(self.dw_gate);
        allocator.free(self.dw_state);
        allocator.free(self.m_w_gate);
        allocator.free(self.v_w_gate);
        allocator.free(self.m_w_state);
        allocator.free(self.v_w_state);
    }
};

// ─── Forward ────────────────────────────────────────────────────────────────

pub const RltWeights = struct {
    w_gate: []f32, // [d, 2*d] — row-major: w_gate[i * 2d + j]
    w_state: []f32, // [d, d]
    alpha: f32,
};

/// Forward pass: u = e + alpha * sigmoid(W_gate @ [e ‖ rmsnorm(s)]) * (W_state @ rmsnorm(s))
/// All dims: e[d], s_prev[d], out_u[d], out_rmsnorm[d] = [2d], etc.
pub fn forward(
    e: []const f32, // [d] encoder input (frozen)
    s_prev: []const f32, // [d] recurrent state from previous token (frozen for this step)
    w: *const RltWeights, // trainable weights
    buf: *RltBuffers,
) []f32 {
    const d = buf.params.d;

    // 1. RMSNorm(s_prev)
    rmsNormForward(s_prev, buf.rmsnorm_s, d);

    // 2. r = [e ‖ rmsnorm_s]
    @memcpy(buf.r[0..d], e);
    @memcpy(buf.r[d..][0..d], buf.rmsnorm_s);

    // 3. gate_logits = W_gate @ r
    gemvForward(w.w_gate, buf.r, buf.gate_logits, d, 2 * d);

    // 4. gate = sigmoid(gate_logits)
    for (buf.gate_logits, 0..) |gl, i| {
        buf.gate[i] = sigmoid(gl);
    }

    // 5. mixed = W_state @ rmsnorm_s
    gemvForward(w.w_state, buf.rmsnorm_s, buf.mixed, d, d);

    // 6. u = e + alpha * gate * mixed
    for (0..d) |i| {
        buf.u[i] = e[i] + w.alpha * buf.gate[i] * buf.mixed[i];
    }

    return buf.u;
}

/// Backward: given upstream gradient dL/du, compute gradients for W_gate, W_state.
/// Accumulates into buf.dw_gate, buf.dw_state, buf.d_alpha.
pub fn backward(
    d_out: []const f32, // [d] dL/du (upstream gradient)
    w: *const RltWeights, // current weights (for computing local grads)
    buf: *RltBuffers,
) void {
    const d = buf.params.d;
    const in_dim = 2 * d; // stride de fila de W_gate [d, 2d]

    // Zero gradient accumulators
    @memset(buf.dw_gate, 0);
    @memset(buf.dw_state, 0);
    buf.d_alpha = 0;

    // ─── 6. u = e + alpha * gate ⊙ mixed ───
    // dL/d(gate[i]*mixed[i]) = alpha * d_out[i], luego reparto:
    //   dL/d_mixed[i] = alpha * gate[i] * d_out[i]
    //   dL/d_gate[i]  = alpha * mixed[i] * d_out[i]
    // (dL/de = d_out — camino directo; el caller lo usa si necesita BPTT)
    for (0..d) |i| {
        const d_mixed_i = w.alpha * buf.gate[i] * d_out[i];

        // ─── 5. mixed = W_state @ rmsnorm_s ───
        // dL/dW_state[i,j] += d_mixed[i] * rmsnorm_s[j]
        for (0..d) |j| {
            buf.dw_state[i * d + j] += d_mixed_i * buf.rmsnorm_s[j];
        }
    }

    // dL/d_rmsnorm_s (camino mixed): W_state^T @ d_mixed
    for (0..d) |j| {
        var acc: f32 = 0;
        for (0..d) |i| {
            acc += w.w_state[i * d + j] * (w.alpha * buf.gate[i] * d_out[i]);
        }
        buf.d_rmsnorm_s[j] = acc;
    }

    // ─── 4. gate = sigmoid(gate_logits) ───
    // dL/d_gate_logits[i] = d_gate[i] * gate[i] * (1 - gate[i])
    for (0..d) |i| {
        const d_gate_i = w.alpha * buf.mixed[i] * d_out[i];
        buf.d_gate_logits[i] = d_gate_i * buf.gate[i] * (1.0 - buf.gate[i]);
    }

    // ─── 3. gate_logits = W_gate @ r ───
    // dL/dW_gate[i,j] += d_gate_logits[i] * r[j]  (outer product)
    for (0..d) |i| {
        for (0..in_dim) |j| {
            buf.dw_gate[i * in_dim + j] += buf.d_gate_logits[i] * buf.r[j];
        }
    }
    // dL/d_r (camino gate): W_gate^T @ d_gate_logits — solo la mitad [d..2d]
    // alimenta a rmsnorm_s; la mitad [0..d] sería dL/de (no la usamos: e es
    // la salida congelada del encoder).
    for (0..in_dim) |j| {
        var acc: f32 = 0;
        for (0..d) |i| {
            acc += w.w_gate[i * in_dim + j] * buf.d_gate_logits[i];
        }
        buf.d_r[j] = acc;
    }

    // ─── 2. r = [e ‖ rmsnorm_s] ───
    // dL/d_rmsnorm_s += d_r[d..2d] (acumula sobre el camino mixed)
    for (0..d) |i| {
        buf.d_rmsnorm_s[i] += buf.d_r[d + i];
    }

    // ─── 1. rmsnorm_s = RMSNorm(s_prev) ───
    // Truncado: s_prev es constante w.r.t. {W_gate, W_state, alpha} EN ESTE
    // PASO, así que dL/ds_prev no afecta a los gradientes de los pesos. Para
    // BPTT depth-k habría que computar:
    //   dL/ds = (d_rmsnorm_s - mean(d_rmsnorm_s * n) * n) / rms
    // y propagarlo al paso t-1. Queda documentado; buf.d_rmsnorm_s ya lo tiene.

    // ─── d_alpha ───
    // u = e + alpha * gate ⊙ mixed => dL/dalpha = Σ gate[i]*mixed[i]*d_out[i]
    var d_alpha: f32 = 0;
    for (0..d) |i| {
        d_alpha += buf.gate[i] * buf.mixed[i] * d_out[i];
    }
    buf.d_alpha += d_alpha;
}

// ─── Helpers ────────────────────────────────────────────────────────────────

pub inline fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

pub inline fn sigmoidDeriv(gate: f32) f32 {
    return gate * (1.0 - gate);
}

/// RMSNorm forward: y[i] = x[i] / sqrt(mean(x²) + RMS_EPS)
/// Idéntico al merge del engine (eps 1e-6) para paridad train/inferencia.
pub fn rmsNormForward(x: []const f32, y: []f32, d: usize) void {
    var sum_sq: f32 = 0;
    for (0..d) |i| sum_sq += x[i] * x[i];
    const rms = @sqrt(sum_sq / @as(f32, @floatFromInt(d)) + RMS_EPS);
    const inv = 1.0 / rms;
    for (0..d) |i| y[i] = x[i] * inv;
}

/// Simple GEMV: out[i] = sum_j W[i*k + j] * v[j], for i in [0..m], j in [0..k]
pub fn gemvForward(W: []const f32, v: []const f32, out: []f32, m: usize, k: usize) void {
    for (0..m) |i| {
        var acc: f32 = 0;
        for (0..k) |j| acc += W[i * k + j] * v[j];
        out[i] = acc;
    }
}

/// Outer product: C[i,j] += a[i] * b[j]
pub fn outerAccum(a: []const f32, b: []const f32, C: []f32, m: usize, n: usize) void {
    for (0..m) |i| {
        for (0..n) |j| {
            C[i * n + j] += a[i] * b[j];
        }
    }
}

// ─── AdamW ──────────────────────────────────────────────────────────────────

pub const AdamWConfig = struct {
    lr: f32 = 1e-3,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps: f32 = 1e-8,
    weight_decay: f32 = 0.01,
    grad_clip: f32 = 1.0,
};

pub fn adamwStep(
    param: []f32, // [n] parameters (in-place update)
    grad: []const f32, // [n] gradient
    m: []f32, // [n] first moment
    v: []f32, // [n] second moment
    config: AdamWConfig,
    t: u32, // timestep (1-indexed)
    n: usize,
) void {
    const beta1 = config.beta1;
    const beta2 = config.beta2;
    const bc1 = 1.0 - std.math.pow(f32, beta1, @floatFromInt(t));
    const bc2 = 1.0 - std.math.pow(f32, beta2, @floatFromInt(t));

    // Gradient clipping (global norm)
    var grad_norm: f32 = 0;
    for (0..n) |i| grad_norm += grad[i] * grad[i];
    grad_norm = @sqrt(grad_norm);
    const clip_scale = if (grad_norm > config.grad_clip) config.grad_clip / grad_norm else 1.0;

    for (0..n) |i| {
        const g = grad[i] * clip_scale;

        // AdamW (Loshchilov & Hutter, decoupled): m/v track SOLO el gradiente;
        // el decay se aplica directo al parámetro, fuera de los momentos.
        m[i] = beta1 * m[i] + (1.0 - beta1) * g;
        v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;

        const m_hat = m[i] / bc1;
        const v_hat = v[i] / bc2;

        param[i] -= config.lr * (m_hat / (@sqrt(v_hat) + config.eps) + config.weight_decay * param[i]);
    }
}

pub fn adamwStepScalar(
    param: *f32,
    grad: f32,
    m: *f32,
    v: *f32,
    config: AdamWConfig,
    t: u32,
) void {
    const beta1 = config.beta1;
    const beta2 = config.beta2;
    const bc1 = 1.0 - std.math.pow(f32, beta1, @floatFromInt(t));
    const bc2 = 1.0 - std.math.pow(f32, beta2, @floatFromInt(t));

    m.* = beta1 * m.* + (1.0 - beta1) * grad;
    v.* = beta2 * v.* + (1.0 - beta2) * grad * grad;

    const m_hat = m.* / bc1;
    const v_hat = v.* / bc2;

    param.* -= config.lr * (m_hat / (@sqrt(v_hat) + config.eps) + config.weight_decay * param.*);
}
