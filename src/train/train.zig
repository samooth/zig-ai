//! Loop de entrenamiento RLT — forward/backward sobre hidden states + adapter.
//!
//! v1 (synthetic): e[t] aleatorios uniformes; objetivo L = ‖u‖²/d.
//! Solo valida el plumbing (gradiente fluye, AdamW actualiza, loss baja).
//! NO produce pesos útiles — la fase 2 sustituye la generación de e[t] por
//! la captura de hidden states del modelo real (forward del engine con α=0)
//! y el objetivo por CE vía lm_head congelado (ver TODO_RLT.md).
//!
//! Backward truncado: dL/du llega del objetivo; el gradiente de los pesos
//! es exacto para el paso t. s_prev se trata como constante (BPTT depth-1).
const std = @import("std");
const rlt = @import("rlt_layer");

fn nowNs() i128 {
    var ts: std.posix.timespec = undefined;
    _ = std.posix.system.clock_gettime(.MONOTONIC, &ts);
    return @as(i128, @intCast(ts.sec)) * std.time.ns_per_s + @as(i128, @intCast(ts.nsec));
}

pub const TrainConfig = struct {
    /// Ruta al GGUF base (reservado para fase 3: carga lm_head congelado).
    model_path: []const u8 = "",
    /// Ruta al .rltcap (tokens + hidden states capturados, última capa attn).
    rltcap_path: []const u8 = "",
    /// Ruta a logits target pre-computados (T × vocab f32 LE).
    /// Si se provee, se usa MSE loss (sin necesidad de lm_head).
    logits_target_path: []const u8 = "",
    vocab_size: usize = 0,
    out_path: []const u8 = "rlt_trained.gguf",
    steps: u32 = 200,
    seq_len: usize = 128,
    batch_size: usize = 1,
    lr: f32 = 3e-4,
    d: usize = 0,
    alpha_init: f32 = 0.15,
    learn_alpha: bool = true,
    alpha_clamp: f32 = 1.0,
    print_every: u32 = 10,
    seed: u64 = 42,
};

pub const TrainMetrics = struct {
    step: u32,
    loss: f32,
    avg_gate_mean: f32,
    alpha: f32,
    lr: f32,
    time_ms: f64,
};

/// Ejecuta el loop de entrenamiento (v1 synthetic). Ver docstring del módulo.
pub fn train(allocator: std.mem.Allocator, config: TrainConfig) !TrainMetrics {
    const d = config.d;
    if (d == 0) return error.DimensionRequired;

    const n_gate = d * 2 * d;
    const n_state = d * d;

    const w_gate = try allocator.alloc(f32, n_gate);
    defer allocator.free(w_gate);
    const w_state = try allocator.alloc(f32, n_state);
    defer allocator.free(w_state);

    // Xavier init
    const rng = std.Random.DefaultPrng.init(config.seed);
    const rand = rng.random();
    const scale_gate = @sqrt(2.0 / @as(f32, @floatFromInt(3 * d)));
    const scale_state = @sqrt(2.0 / @as(f32, @floatFromInt(2 * d)));
    for (w_gate) |*v| v.* = rand.float(f32) * scale_gate * 2.0 - scale_gate;
    for (w_state) |*v| v.* = rand.float(f32) * scale_state * 2.0 - scale_state;

    var weights = rlt.RltWeights{
        .w_gate = w_gate,
        .w_state = w_state,
        .alpha = config.alpha_init,
    };

    var buf = try rlt.RltBuffers.alloc(allocator, .{ .d = d });
    defer buf.deinit();

    const adam = rlt.AdamWConfig{
        .lr = config.lr,
        .weight_decay = 0.01,
        .grad_clip = 1.0,
    };

    const e_buf = try allocator.alloc(f32, d);
    defer allocator.free(e_buf);
    const s_buf = try allocator.alloc(f32, d);
    defer allocator.free(s_buf);
    const d_out = try allocator.alloc(f32, d);
    defer allocator.free(d_out);
    @memset(s_buf, 0);

    var loss_ema: f32 = 0;
    var gate_mean_ema: f32 = 0;

    var step: u32 = 0;
    while (step < config.steps) : (step += 1) {
        const t0_ns = nowNs();

        // Fase 2: aquí irá la captura de e[t] del forward del modelo base.
        for (e_buf) |*v| v.* = rand.float(f32) * 2.0 - 1.0;

        // ─── Forward ───
        const u = rlt.forward(e_buf, s_buf, &weights, &buf);

        // ─── Loss: L = ‖u‖² / d (energía media) ───
        var loss: f32 = 0;
        for (u) |v| loss += v * v;
        loss /= @as(f32, @floatFromInt(d));

        // ─── dL/du = 2u/d (derivada exacta del loss de arriba) ───
        const two_over_d = 2.0 / @as(f32, @floatFromInt(d));
        for (0..d) |i| d_out[i] = two_over_d * u[i];

        rlt.backward(d_out, &weights, &buf);

        // ─── AdamW step ───
        rlt.adamwStep(w_gate, buf.dw_gate, buf.m_w_gate, buf.v_w_gate, adam, step + 1, n_gate);
        rlt.adamwStep(w_state, buf.dw_state, buf.m_w_state, buf.v_w_state, adam, step + 1, n_state);
        if (config.learn_alpha) {
            rlt.adamwStepScalar(&weights.alpha, buf.d_alpha, &buf.m_alpha, &buf.v_alpha, adam, step + 1);
            // Clamp: alpha fuera de [0, alpha_clamp] rompe el contrato del
            // engine (negativo = substracción; grande = merge explosivo).
            weights.alpha = std.math.clamp(weights.alpha, 0.0, config.alpha_clamp);
        }

        // ─── EMA tracking ───
        const ema_beta: f32 = 0.95;
        loss_ema = ema_beta * loss_ema + (1.0 - ema_beta) * loss;

        var gate_sum: f32 = 0;
        for (buf.gate) |g| gate_sum += g;
        const gate_mean = gate_sum / @as(f32, @floatFromInt(d));
        gate_mean_ema = ema_beta * gate_mean_ema + (1.0 - ema_beta) * gate_mean;

        const t1_ns = nowNs();
        const step_ms = @as(f64, @floatFromInt(t1_ns - t0_ns)) / 1e6;

        // Estado recurrente para el siguiente token: s ← u (misma semántica
        // que el engine, que feed-forwarda el estado merged).
        @memcpy(s_buf, u);

        if (step % config.print_every == 0 or step == config.steps - 1) {
            std.debug.print(
                "[step {d:4}] loss={d:.4}  gate_mean={d:.4}  alpha={d:.4}  ({d:.1}ms)\n",
                .{ step, loss_ema, gate_mean_ema, weights.alpha, step_ms },
            );
        }

        // Gradient check periódico (assert suave: print en mismatch)
        if (step > 0 and step % 50 == 0) {
            _ = gradientCheck(allocator, &weights, e_buf, s_buf, d);
        }
    }

    return .{
        .step = step,
        .loss = loss_ema,
        .avg_gate_mean = gate_mean_ema,
        .alpha = weights.alpha,
        .lr = config.lr,
        .time_ms = 0,
    };
}

/// Finite-difference gradient check del gradiente analítico de alpha.
/// USA EL MISMO loss que el loop (L = ‖u‖²/d) — el gradiente analítico y el
/// numérico deben medir la misma función o la comparación no significa nada.
/// Returns max |analytic − numerical| (esperado < 1e-3 con eps central).
pub fn gradientCheck(
    allocator: std.mem.Allocator,
    weights: *rlt.RltWeights,
    e: []const f32,
    s: []const f32,
    d: usize,
) f32 {
    const eps: f32 = 1e-4;

    var buf_ref = rlt.RltBuffers.alloc(allocator, .{ .d = d }) catch return std.math.nan(f32);
    defer buf_ref.deinit(allocator);
    const u_ref = rlt.forward(e, s, weights, &buf_ref);

    // dL/du del MISMO loss L = ‖u‖²/d
    const d_out = allocator.alloc(f32, d) catch return std.math.nan(f32);
    defer allocator.free(d_out);
    const two_over_d = 2.0 / @as(f32, @floatFromInt(d));
    for (0..d) |i| d_out[i] = two_over_d * u_ref[i];
    rlt.backward(d_out, weights, &buf_ref);

    const analytic = buf_ref.d_alpha;
    const orig_alpha = weights.alpha;

    // L(α) numérico con la MISMA fórmula ‖u‖²/d
    weights.alpha = orig_alpha + eps;
    const u_plus = rlt.forward(e, s, weights, &buf_ref);
    var loss_plus: f32 = 0;
    for (u_plus) |v| loss_plus += v * v;
    loss_plus /= @as(f32, @floatFromInt(d));

    weights.alpha = orig_alpha - eps;
    const u_minus = rlt.forward(e, s, weights, &buf_ref);
    var loss_minus: f32 = 0;
    for (u_minus) |v| loss_minus += v * v;
    loss_minus /= @as(f32, @floatFromInt(d));

    weights.alpha = orig_alpha;
    const numerical = (loss_plus - loss_minus) / (2.0 * eps);

    const diff = @abs(analytic - numerical);
    if (diff > 1e-3) {
        std.debug.print("[grad_check] ALPHA MISMATCH: analytic={d:.6} numerical={d:.6} diff={d:.6}\n", .{ analytic, numerical, diff });
    }
    return diff;
}

/// Entrenamiento real (fase 2): lee .rltcap + lm_head congelado, aplica CE
/// loss via cross-entropy softmax, entrena w_gate/w_state/alpha.
/// Diseño: BPTT depth-1, s_prev como constante, batch_size=1 (token a token).
pub fn trainReal(allocator: std.mem.Allocator, config: TrainConfig) !TrainMetrics {
    const d = config.d;
    if (d == 0) return error.DimensionRequired;
    if (config.logits_target_path.len == 0 or config.vocab_size == 0) return error.LogitsTargetRequired;
    if (config.rltcap_path.len == 0) return error.CapturePathRequired;

    const n_gate = d * 2 * d;
    const n_state = d * d;

    var cap = try readRltcap(allocator, config.rltcap_path);
    defer cap.deinit(allocator);
    if (cap.header.d != d) return error.DimensionMismatch;
    const T = cap.header.T;
    if (T < 2) return error.NotEnoughTokens;

    const logits_target = try readLogitsTarget(allocator, config.logits_target_path, T, config.vocab_size);
    defer allocator.free(logits_target);

    const w_gate = try allocator.alloc(f32, n_gate);
    defer allocator.free(w_gate);
    const w_state = try allocator.alloc(f32, n_state);
    defer allocator.free(w_state);
    var rng = std.Random.DefaultPrng.init(config.seed);
    const rand = rng.random();
    const scale_gate = @sqrt(2.0 / @as(f32, @floatFromInt(3 * d)));
    const scale_state = @sqrt(2.0 / @as(f32, @floatFromInt(2 * d)));
    for (w_gate) |*v| v.* = rand.float(f32) * scale_gate * 2.0 - scale_gate;
    for (w_state) |*v| v.* = rand.float(f32) * scale_state * 2.0 - scale_state;
    var weights = rlt.RltWeights{
        .w_gate = w_gate,
        .w_state = w_state,
        .alpha = config.alpha_init,
    };
    var buf = try rlt.RltBuffers.alloc(allocator, .{ .d = d });
    defer buf.deinit(allocator);
    const adam = rlt.AdamWConfig{
        .lr = config.lr,
        .weight_decay = 0.01,
        .grad_clip = 1.0,
    };
    const e_buf = try allocator.alloc(f32, d);
    defer allocator.free(e_buf);
    const s_buf = try allocator.alloc(f32, d);
    defer allocator.free(s_buf);
    const d_out = try allocator.alloc(f32, d);
    defer allocator.free(d_out);
    @memset(s_buf, 0);

    const last_layer_e = cap.layerE(cap.header.n_layers - 1);
    const vocab = config.vocab_size;
    const two_over_d = 2.0 / @as(f32, @floatFromInt(d));

    var loss_ema: f32 = 0;
    var gate_mean_ema: f32 = 0;
    const ema_beta: f32 = 0.95;

    var step: u32 = 0;
    while (step < config.steps) : (step += 1) {
        const t0_ns = nowNs();
        const t: u32 = @intCast(rand.intRangeAtMost(u32, 0, @as(u32, @intCast(T - 2))));
        @memcpy(e_buf, last_layer_e[t * d ..][0..d]);

        const u = rlt.forward(e_buf, s_buf, &weights, &buf);

        // MSE loss: L = ||u - logits_target[t]||² / d
        var loss_val: f32 = 0;
        const target_off = t * vocab;
        for (0..d) |i| {
            const diff = u[i] - logits_target[target_off + i];
            loss_val += diff * diff;
        }
        loss_val /= @as(f32, @floatFromInt(d));

        // dL/du = 2(u - target) / d
        for (0..d) |i| {
            d_out[i] = two_over_d * (u[i] - logits_target[target_off + i]);
        }

        rlt.backward(d_out, &weights, &buf);

        rlt.adamwStep(w_gate, buf.dw_gate, buf.m_w_gate, buf.v_w_gate, adam, step + 1, n_gate);
        rlt.adamwStep(w_state, buf.dw_state, buf.m_w_state, buf.v_w_state, adam, step + 1, n_state);
        if (config.learn_alpha) {
            rlt.adamwStepScalar(&weights.alpha, buf.d_alpha, &buf.m_alpha, &buf.v_alpha, adam, step + 1);
            weights.alpha = std.math.clamp(weights.alpha, 0.0, config.alpha_clamp);
        }

        @memcpy(s_buf, u);

        loss_ema = ema_beta * loss_ema + (1.0 - ema_beta) * loss_val;
        var gate_sum: f32 = 0;
        for (buf.gate) |g| gate_sum += g;
        const gate_mean = gate_sum / @as(f32, @floatFromInt(d));
        gate_mean_ema = ema_beta * gate_mean_ema + (1.0 - ema_beta) * gate_mean;

        const t1_ns = nowNs();
        const step_ms = @as(f64, @floatFromInt(t1_ns - t0_ns)) / 1e6;
        if (step % config.print_every == 0 or step == config.steps - 1) {
            std.debug.print(
                "[step {d:4}] loss={d:.4}  gate_mean={d:.4}  alpha={d:.4}  ({d:.1}ms)\n",
                .{ step, loss_ema, gate_mean_ema, weights.alpha, step_ms },
            );
        }
    }

    try writeRltSidecar(allocator, config.out_path, weights, d, config.alpha_init);
    return .{
        .step = step,
        .loss = loss_ema,
        .avg_gate_mean = gate_mean_ema,
        .alpha = weights.alpha,
        .lr = config.lr,
        .time_ms = 0,
    };
}

// ─── .rltcap reader ───────────────────────────────────────────────────────────

const RltcapHeader = struct {
    T: usize,
    d: usize,
    n_layers: usize,
};

pub const Rltcap = struct {
    header: RltcapHeader,
    tokens: []u32,
    e: []f32,

    pub fn layerE(self: *const Rltcap, layer: usize) []const f32 {
        const off = layer * self.header.T * self.header.d;
        return self.e[off .. off + self.header.T * self.header.d];
    }

    pub fn deinit(self: *Rltcap, allocator: std.mem.Allocator) void {
        allocator.free(self.tokens);
        allocator.free(self.e);
    }
};

pub fn readRltcap(allocator: std.mem.Allocator, path: []const u8) !Rltcap {
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = std.Io.Dir.cwd();
    var file = try dir.openFile(io, path, .{ .mode = .read_only });
    defer file.close(io);
    const size = (try file.stat(io)).size;
    if (size < 20) return error.BadMagic;
    var mm = try std.Io.File.MemoryMap.create(io, file, .{
        .len = @intCast(size),
        .protection = .{ .read = true },
        .populate = false,
    });
    defer mm.destroy(io);
    const data = mm.memory;
    const magic = std.mem.readInt(u32, data[0..4], .little);
    if (magic != 0x43544C52) return error.BadMagic;
    _ = std.mem.readInt(u32, data[4..8], .little);
    const T = std.mem.readInt(u32, data[8..12], .little);
    const d = std.mem.readInt(u32, data[12..16], .little);
    const n_layers = std.mem.readInt(u32, data[16..20], .little);
    const tokens = try allocator.alloc(u32, T);
    for (tokens, 0..) |*tok, i| tok.* = std.mem.readInt(u32, data[20 + i * 4 ..][0..4], .little);
    const e_len = T * d;
    const e = try allocator.alloc(f32, e_len);
    const e_base = 20 + T * 4;
    for (e, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, data[e_base + i * 4 ..][0..4], .little));
    _ = n_layers;
    return .{ .header = .{ .T = T, .d = d, .n_layers = 1 }, .tokens = tokens, .e = e };
}

/// Escribe un sidecar GGUF con pesos RLT entrenados (fase 2).
fn writeRltSidecar(
    allocator: std.mem.Allocator,
    out_path: []const u8,
    weights: rlt.RltWeights,
    d: usize,
    alpha: f32,
) !void {
    var buf = std.array_list.Managed(u8).init(allocator);
    defer buf.deinit();

    var hdr: [24]u8 = undefined;
    std.mem.writeInt(u32, hdr[0..4], 0x46554747, .little);
    std.mem.writeInt(u32, hdr[4..8], 3, .little);
    std.mem.writeInt(u64, hdr[8..16], 0, .little);
    std.mem.writeInt(u64, hdr[16..24], 0, .little);
    try buf.appendSlice(hdr[0..]);

    var kv_off: u64 = 28;
    const n_layers = 1;
    const n_tensors = @as(u64, @intCast(n_layers * 2));

    try writeKvString(&buf, "general.name", "rlt-trained");
    kv_off += 8 + 8 + 10;
    try writeKvFloat(&buf, "rlt.feedback_alpha", alpha);
    kv_off += 8 + 4 + 4;

    const tensor_data_start = kv_off + n_tensors * 24;
    var tensor_off = tensor_data_start;
    const align_pad = (32 - (tensor_data_start % 32)) % 32;
    var pad_buf: [32]u8 = .{0} ** 32;
    try buf.appendSlice(pad_buf[0..align_pad]);

    for (0..n_layers) |layer| {
        const name_buf = try std.fmt.allocPrint(allocator, "blk.{d}.rlt.feedback_gate.weight", .{layer});
        defer allocator.free(name_buf);
        try writeTensorInfo(&buf, name_buf, &[_]u64{ @intCast(d * 2), @intCast(d) }, tensor_off);
        tensor_off += d * 2 * d * 4;
    }
    for (0..n_layers) |layer| {
        const name_buf = try std.fmt.allocPrint(allocator, "blk.{d}.rlt.feedback_state.weight", .{layer});
        defer allocator.free(name_buf);
        try writeTensorInfo(&buf, name_buf, &[_]u64{ @intCast(d), @intCast(d) }, tensor_off);
        tensor_off += d * d * 4;
    }

    for (weights.w_gate) |v| {
        var b4: [4]u8 = undefined;
        std.mem.writeInt(u32, &b4, @bitCast(v), .little);
        try buf.appendSlice(&b4);
    }
    for (weights.w_state) |v| {
        var b4: [4]u8 = undefined;
        std.mem.writeInt(u32, &b4, @bitCast(v), .little);
        try buf.appendSlice(&b4);
    }

    std.mem.writeInt(u64, buf.items[12..20], n_tensors, .little);
    std.mem.writeInt(u64, buf.items[20..28], kv_off, .little);

    const cwd = std.Io.Dir.cwd();
    var f = try cwd.createFile(std.Io.Threaded.global_single_threaded.io(), out_path, .{});
    defer f.close(std.Io.Threaded.global_single_threaded.io());
    var fbuf: [4096]u8 = undefined;
    var fw = f.writer(std.Io.Threaded.global_single_threaded.io(), &fbuf);
    const writer = &fw.interface;
    try writer.writeAll(buf.items);
    try writer.flush();
}

fn writeKvString(buf: *std.array_list.Managed(u8), key: []const u8, value: []const u8) !void {
    var hdr: [8]u8 = undefined;
    std.mem.writeInt(u64, &hdr, key.len, .little);
    try buf.appendSlice(hdr[0..]);
    try buf.appendSlice(key);
    std.mem.writeInt(u64, &hdr, value.len, .little);
    try buf.appendSlice(hdr[0..]);
    try buf.appendSlice(value);
}

fn writeKvFloat(buf: *std.array_list.Managed(u8), key: []const u8, value: f32) !void {
    var hdr: [8]u8 = undefined;
    std.mem.writeInt(u64, &hdr, key.len, .little);
    try buf.appendSlice(hdr[0..]);
    try buf.appendSlice(key);
    // GGUF value type for float32 = 0
    var type_buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &type_buf, 0, .little);
    try buf.appendSlice(&type_buf);
    // Float value
    var b4: [4]u8 = undefined;
    std.mem.writeInt(u32, &b4, @bitCast(value), .little);
    try buf.appendSlice(&b4);
}

fn writeTensorInfo(buf: *std.array_list.Managed(u8), name: []const u8, dims: []const u64, offset: u64) !void {
    var hdr: [8]u8 = undefined;
    std.mem.writeInt(u64, &hdr, name.len, .little);
    try buf.appendSlice(hdr[0..]);
    try buf.appendSlice(name);
    var b4: [4]u8 = undefined;
    std.mem.writeInt(u32, &b4, @intCast(dims.len), .little);
    try buf.appendSlice(b4[0..]);
    for (dims) |dim| {
        std.mem.writeInt(u64, &hdr, dim, .little);
        try buf.appendSlice(hdr[0..]);
    }
    std.mem.writeInt(u32, &b4, 0, .little);
    try buf.appendSlice(b4[0..]);
    std.mem.writeInt(u64, &hdr, offset, .little);
    try buf.appendSlice(hdr[0..]);
}

fn readLogitsTarget(allocator: std.mem.Allocator, path: []const u8, T: usize, vocab: usize) ![]f32 {
    const cwd = std.Io.Dir.cwd();
    var f = try cwd.openFile(std.Io.Threaded.global_single_threaded.io(), path, .{ .mode = .read_only });
    defer f.close(std.Io.Threaded.global_single_threaded.io());
    const expected = T * vocab * 4;
    const stat = try f.stat(std.Io.Threaded.global_single_threaded.io());
    if (stat.size != expected) return error.BadLogitsSize;
    const data = try allocator.alloc(f32, T * vocab);
    const buf = try allocator.alloc(u8, expected);
    defer allocator.free(buf);
    _ = try f.readPositionalAll(std.Io.Threaded.global_single_threaded.io(), buf, 0);
    for (data, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, buf[i * 4 ..][0..4], .little));
    return data;
}
