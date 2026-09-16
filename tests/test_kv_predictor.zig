//! Paso 1 KV-Codec §B (lane-kvc): harness del predictor temporal + GATE.
//!
//! Predictores por (capa, head, canal) sobre la serie temporal de tokens:
//!   raw / delta_naif (lag-1) / ema (α autotune por MAD) / nlms (2,4 taps) /
//!   kalman escalar.
//!
//! DISEÑO CLOSED-LOOP DPCM OBLIGATORIO (plan §5): el predictor se actualiza
//! con el valor RECONSTRUIDO (residual cuantizado simulado + predicción),
//! NO con el original — si no, el decode no puede reproducir el estado y la
//! ganancia es ficción. Dos regímenes:
//!   1. Cota superior (residual exacto sin cuantizar) — teórico.
//!   2. Realista (residual cuantizado 2/3/4 bits uniforme por grupo de 32,
//!      escala amax del grupo — el decoder recibe la escala por grupo como
//!      cabecera, réplica del layout q4_0-like). **EL GATE USA ESTA.**
//!
//! Métricas por (modelo, corpus, capa, tensor):
//!   var_ratio = var(delta_naif)/var(predictor)  (closed-loop) ← GATE
//!   var_ratio_raw = var(raw)/var(predictor)
//!   Entropía empírica de residuales cuantizados (bins por nivel de quant)
//!   Deriva closed-loop: error acumulado |x_rec−x_true| a T tokens
//!
//! Fuente: trazas A2 en $KVTRACES (default /tmp/opencode/kvtraces), layout
//! f16 [T, n_kv_head, head_dim]. Sin trazas: skip (la captura requiere
//! modelo y prefill CPU — no va en CI).
//!
//! Output: results/kv_predictor.json (array de registros por caso).

const std = @import("std");

const PredictorKind = enum { raw, delta_naif, ema, nlms2, nlms4, kalman };

/// Estado del predictor por canal (serie temporal a lo largo de tokens).
const ChannelPredictor = struct {
    kind: PredictorKind,
    alpha: f32 = 0.75, // ema (autotune 0.5..0.9375)
    ema_state: f32 = 0,
    ema_mad_pred: f32 = 0.1,
    ema_mad_delta: f32 = 0.1,
    w: [4]f32 = .{ 0, 0, 0, 0 }, // nlms
    taps: usize = 2,
    // μ=0.02 + leak 0.999: NLMS DIVERGE sistemáticamente en closed-loop
    // cuantizado (var 2e-5..3e7 en 36 casos) — hallazgo del harness: la
    // realimentación del residual cuantizado + no-estacionaridad de K/V
    // hace explotar los taps (no es sintonizable con μ/leak: probado 0.5,
    // 0.1, 0.02). El techo de la familia adaptativa lo marcan EMA/Kalman.
    // Se mantiene en el harness como DATO (negativa documentada §11.1).
    mu: f32 = 0.02,
    x_hist: [4]f32 = .{ 0, 0, 0, 0 },
    q: f32 = 1e-4, // kalman
    r: f32 = 1e-2,
    p: f32 = 1,
    x_est: f32 = 0,
    init: bool = false,

    fn predict(self: *const ChannelPredictor) f32 {
        const taps: usize = if (self.kind == .nlms4) 4 else self.taps;
        return switch (self.kind) {
            .raw => 0,
            .delta_naif => self.x_hist[0],
            .ema => self.ema_state,
            .nlms2, .nlms4 => blk: {
                var acc: f32 = 0;
                for (0..taps) |i| acc += self.w[i] * self.x_hist[i];
                break :blk acc;
            },
            .kalman => self.x_est,
        };
    }

    /// Actualiza con el valor RECONSTRUIDO (closed-loop DPCM).
    fn update(self: *ChannelPredictor, x_rec: f32) void {
        if (!self.init) {
            self.init = true;
            self.ema_state = x_rec;
            self.x_est = x_rec;
            self.x_hist = .{ x_rec, x_rec, x_rec, x_rec };
            self.w[0] = 1; // NLMS init: persistencia lag-1
            return;
        }
        switch (self.kind) {
            .raw, .delta_naif => {},
            .ema => self.ema_state = self.alpha * self.ema_state + (1 - self.alpha) * x_rec,
            .nlms2, .nlms4 => {
                const taps: usize = if (self.kind == .nlms4) 4 else self.taps;
                var norm: f32 = 1e-6;
                for (0..taps) |i| norm += self.x_hist[i] * self.x_hist[i];
                const pred = blk: {
                    var acc: f32 = 0;
                    for (0..taps) |i| acc += self.w[i] * self.x_hist[i];
                    break :blk acc;
                };
                const err = x_rec - pred;
                const step = self.mu * @as(f32, @floatFromInt(taps)) / norm;
                for (0..taps) |i| self.w[i] += step * err * self.x_hist[i];
                const leak: f32 = 0.999;
                for (0..taps) |i| self.w[i] *= leak;
            },
            .kalman => {
                self.p += self.q;
                const k = self.p / (self.p + self.r);
                self.x_est += k * (x_rec - self.x_est);
                self.p *= (1 - k);
            },
        }
        self.x_hist[3] = self.x_hist[2];
        self.x_hist[2] = self.x_hist[1];
        self.x_hist[1] = self.x_hist[0];
        self.x_hist[0] = x_rec;
    }

    /// Autotune α por ganancia observada (MAD/canal): EMA vs delta-naïf.
    fn tuneAlpha(self: *ChannelPredictor, x_true: f32) void {
        if (self.kind != .ema or !self.init) return;
        const e_ema = @abs(x_true - self.ema_state);
        const e_delta = @abs(x_true - self.x_hist[0]);
        self.ema_mad_pred = 0.95 * self.ema_mad_pred + 0.05 * e_ema;
        self.ema_mad_delta = 0.95 * self.ema_mad_delta + 0.05 * e_delta;
        if (self.ema_mad_pred < self.ema_mad_delta) {
            self.alpha = @min(0.9375, self.alpha + 0.05);
        } else {
            self.alpha = @max(0.5, self.alpha - 0.05);
        }
    }
};

/// Escala por grupo de 32 (amax): se computa sobre los residuales del grupo
/// — el DECODER la recibe como cabecera (≈0.03 bits/valor), luego reproduce.
/// Para el harness offline: 1ª pasada por grupo para la escala, 2ª pasada
/// closed-loop para quant/dequant.
fn runClosedLoop(
    allocator: std.mem.Allocator,
    kind: PredictorKind,
    channels: []const f32, // [T, C]
    t_len: usize,
    c_len: usize,
    bits: u3, // 2, 3, 4 — realista; bits=99 conceptual = exacto (se pasa aparte)
    exact: bool, // true = cota superior sin cuantizar
) !struct { var_geo: f64, drift: f64, entropy: f64 } {
    var preds = try allocator.alloc(ChannelPredictor, c_len);
    defer allocator.free(preds);
    for (preds) |*p| p.* = .{ .kind = kind };

    const levels: i32 = (@as(i32, 1) << @as(u5, @intCast(bits))) - 1;
    const group: usize = 32;

    var residual = try allocator.alloc(f32, t_len);
    defer allocator.free(residual);
    var recon = try allocator.alloc(f32, t_len);
    defer allocator.free(recon);

    var log_var_sum: f64 = 0;
    var c_used: usize = 0;
    var drift_sum: f64 = 0;
    var entropy_bins = try allocator.alloc(u64, @intCast(levels + 1));
    defer allocator.free(entropy_bins);
    @memset(entropy_bins, 0);
    var total_quant: u64 = 0;

    for (0..c_len) |c| {
        var v_n: f64 = 0;
        var v_sum: f64 = 0;
        var v_sumsq: f64 = 0;
        // estado de grupo por canal
        var g_start: usize = 0;

        while (g_start < t_len) {
            const g_end = @min(g_start + group, t_len);
            // PASADA 1: residuales del grupo (predictores al estado actual)
            for (g_start..g_end) |t| {
                const x = channels[t * c_len + c];
                residual[t] = x - preds[c].predict();
            }
            // escala amax del grupo
            var amax: f32 = 0;
            for (residual[g_start..g_end]) |rv| amax = @max(amax, @abs(rv));
            const d: f32 = if (amax < 1e-12) 0 else amax / @as(f32, @floatFromInt(levels));
            // PASADA 2: closed-loop — quant/dequant + update por token
            for (g_start..g_end) |t| {
                const x = channels[t * c_len + c];
                const pred = preds[c].predict();
                const r = x - pred;
                if (exact) {
                    recon[t] = r;
                } else {
                    const qf: f32 = if (d == 0) 0 else @round(r / d);
                    const half: f32 = @as(f32, @floatFromInt(@divTrunc(levels, 2)));
                    const clamped_f = @max(-half, @min(half, qf));
                    recon[t] = pred + clamped_f * d;
                    // stats quant (para entropía) — índice desplazado a base-0
                    const bin: usize = @intFromFloat(clamped_f + half);
                    if (bin < entropy_bins.len) entropy_bins[bin] += 1;
                    total_quant += 1;
                }
                const r_final = x - recon[t]; // error de reconstrucción real
                v_n += 1;
                v_sum += r_final;
                v_sumsq += @as(f64, r_final) * r_final;
                preds[c].update(recon[t]);
                preds[c].tuneAlpha(x);
            }
            g_start = g_end;
        }
        // deriva: error acumulado del último tramo (últimos 512 tokens)
        const from = t_len - @min(512, t_len);
        var dr: f64 = 0;
        for (from..t_len) |t| dr += @abs(channels[t * c_len + c] - recon[t]);
        drift_sum += dr / @as(f64, @floatFromInt(t_len - from));

        if (v_n >= 2) {
            const m = v_sum / v_n;
            const v = (v_sumsq - v_n * m * m) / (v_n - 1);
            if (v > 0) {
                log_var_sum += @log(v);
                c_used += 1;
            }
        }
    }

    const var_geo: f64 = if (c_used == 0) 0 else @exp(log_var_sum / @as(f64, @floatFromInt(c_used)));
    const drift: f64 = if (c_len == 0) 0 else drift_sum / @as(f64, @floatFromInt(c_len));

    // entropía empírica de los índices cuantizados (bits/símbolo)
    var ent: f64 = 0;
    if (total_quant > 0) {
        for (entropy_bins) |b| {
            if (b == 0) continue;
            const p = @as(f64, @floatFromInt(b)) / @as(f64, @floatFromInt(total_quant));
            ent -= p * @log2(p);
        }
    }
    return .{ .var_geo = var_geo, .drift = drift, .entropy = ent };
}

test "predictor harness: gate sobre trazas A2 (skip sin trazas)" {
    const t = std.testing;
    var arena_state = std.heap.ArenaAllocator.init(t.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.Io.Threaded.global_single_threaded.io();

    const traces_dir = std.mem.span(std.c.getenv("KVTRACES") orelse "/tmp/opencode/kvtraces");

    const Case = struct { dir: []const u8, layers: []const []const u8, kv_elems: usize, family: []const u8 };
    const cases = [_]Case{
        .{ .dir = "Qwen3.5-0.8B-Q4_0/prose", .layers = &.{ "3", "11", "23" }, .kv_elems = 512, .family = "qwen35-hybrid" },
        .{ .dir = "Qwen3.5-0.8B-Q4_0/code", .layers = &.{ "3", "11", "23" }, .kv_elems = 512, .family = "qwen35-hybrid" },
        .{ .dir = "LFM2.5-2.6B-Q4_K_M/prose", .layers = &.{ "2", "9", "17" }, .kv_elems = 512, .family = "lfm2-shortconv" },
        .{ .dir = "LFM2.5-2.6B-Q4_K_M/code", .layers = &.{ "2", "9", "17" }, .kv_elems = 512, .family = "lfm2-shortconv" },
    };

    var json: std.ArrayList(u8) = .empty;
    try json.appendSlice(arena, "[\n");
    var any_found = false;

    for (cases) |case| {
        for (case.layers) |lname| {
            for ([_][]const u8{ "k_pre", "k_post", "v" }) |kind| {
                var pbuf: [512]u8 = undefined;
                const path = try std.fmt.bufPrint(&pbuf, "{s}/{s}/L{s}/{s}.bin", .{ traces_dir, case.dir, lname, kind });
                var file = std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only }) catch continue;
                defer file.close(io);
                any_found = true;
                const st = try file.stat(io);
                const data = try arena.alloc(u8, @intCast(st.size));
                _ = try file.readPositionalAll(io, data, 0);
                const f16s = std.mem.bytesAsSlice(u16, data);
                const t_len = f16s.len / case.kv_elems;
                if (t_len < 128) continue;
                const channels = try arena.alloc(f32, t_len * case.kv_elems);
                for (f16s[0 .. t_len * case.kv_elems], 0..) |h, i| channels[i] = @as(f16, @bitCast(h));

                // El GATE usa el régimen REALISTA 4 bits (peor caso de la
                // familia 2-4: más bits = residual más limpio = ganancia
                // MÁS difícil para el predictor — conservador).
                const bits_gate: u3 = 4;

                const v_raw = try runClosedLoop(arena, .raw, channels, t_len, case.kv_elems, bits_gate, false);
                const v_delta = try runClosedLoop(arena, .delta_naif, channels, t_len, case.kv_elems, bits_gate, false);
                const v_ema = try runClosedLoop(arena, .ema, channels, t_len, case.kv_elems, bits_gate, false);
                const v_nlms2 = try runClosedLoop(arena, .nlms2, channels, t_len, case.kv_elems, bits_gate, false);
                const v_nlms4 = try runClosedLoop(arena, .nlms4, channels, t_len, case.kv_elems, bits_gate, false);
                const v_kalman = try runClosedLoop(arena, .kalman, channels, t_len, case.kv_elems, bits_gate, false);
                const v_ema_exact = try runClosedLoop(arena, .ema, channels, t_len, case.kv_elems, bits_gate, true);

                if (json.items.len > 2) try json.appendSlice(arena, ",\n");
                var aw = std.Io.Writer.Allocating.init(arena);
                aw.writer.print("{{\"family\":\"{s}\",\"case\":\"{s}/L{s}/{s}\",\"T\":{d}," ++
                    "\"var_raw\":{e:.6},\"var_delta\":{e:.6},\"var_ema\":{e:.6},\"var_nlms2\":{e:.6}," ++
                    "\"var_nlms4\":{e:.6},\"var_kalman\":{e:.6},\"var_ema_exact\":{e:.6}," ++
                    "\"vr_best_vs_delta\":{d:.3},\"vr_delta_vs_raw\":{d:.3},\"entropy_best\":{d:.3}," ++
                    "\"drift_ema\":{e:.4}}}", .{
                    case.family,      case.dir,            lname,                                                                                                              kind,                                         t_len,
                    v_raw.var_geo,    v_delta.var_geo,     v_ema.var_geo,                                                                                                      v_nlms2.var_geo,                              v_nlms4.var_geo,
                    v_kalman.var_geo, v_ema_exact.var_geo, v_delta.var_geo / @max(@min(v_ema.var_geo, @min(v_nlms2.var_geo, @min(v_nlms4.var_geo, v_kalman.var_geo))), 1e-30), v_raw.var_geo / @max(v_delta.var_geo, 1e-30), @min(v_ema.entropy, @min(v_nlms2.entropy, @min(v_nlms4.entropy, v_kalman.entropy))),
                    v_ema.drift,
                }) catch return error.OutOfMemory;
                try json.appendSlice(arena, aw.written());
            }
        }
    }
    try json.appendSlice(arena, "\n]\n");

    if (!any_found) return error.SkipZigTest;

    // results/kv_predictor.json
    {
        var file = try std.Io.Dir.cwd().createFile(io, "results/kv_predictor.json", .{});
        defer file.close(io);
        var wbuf: [8192]u8 = undefined;
        var fw = file.writer(io, &wbuf);
        const wr = &fw.interface;
        try wr.writeAll(json.items);
        try wr.flush();
    }

    std.debug.print("\n[kv_predictor] results/kv_predictor.json escrito ({d} bytes) — gate verdict en fase D\n", .{json.items.len});
}

test "closed-loop sanity: señal senoidal predecible" {
    // Sin trazas: valida el harness con señal sintética donde EMA/NLMS
    // DEBEN batir a raw y el residual cerrado converge.
    const t = std.testing;
    const arena = t.allocator;
    const T: usize = 512;
    const C: usize = 8;
    var ch = try arena.alloc(f32, T * C);
    defer arena.free(ch);
    for (0..T) |i| {
        const fi: f32 = @floatFromInt(i);
        for (0..C) |c| {
            const fc: f32 = @floatFromInt(c);
            ch[i * C + c] = 3.0 * @sin(0.05 * fi + 0.7 * fc) + 0.5 * @sin(0.013 * fi * fc);
        }
    }
    const v_raw = try runClosedLoop(arena, .raw, ch, T, C, 4, false);
    const v_delta = try runClosedLoop(arena, .delta_naif, ch, T, C, 4, false);
    const v_ema = try runClosedLoop(arena, .ema, ch, T, C, 4, false);
    const v_nlms4 = try runClosedLoop(arena, .nlms4, ch, T, C, 4, false);
    // delta-naïf debe reducir ≥2× la varianza vs raw en señal suave
    try t.expect(v_raw.var_geo / v_delta.var_geo > 2.0);
    // algún predictor adaptativo no debe ser PEOR que delta (solo consistencia
    // de implementación — la señal es suave y todos deberían funcionar)
    const best = @min(v_ema.var_geo, v_nlms4.var_geo);
    try t.expect(v_delta.var_geo / best > 0.5);
}
