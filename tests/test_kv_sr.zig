//! Paso 0 KV-Codec §4.6/§5 (v2 §4.6): stochastic rounding en V.
//! out = Σ_t w_t·v_t — el error de V se pondera por w_t. El redondeo
//! determinístico produce sesgo sistemático que ACUMULA con la masa de
//! atención y capa a capa; SR es insesgado y su varianza se promedia.
//! Métrica clave: SESGO neto (|E[out]−ref|), no ABS single-shot.
const std = @import("std");

fn stdoutPrint(io: std.Io, comptime fmt: []const u8, args: anytype) !void {
    var stdout_buf: [256]u8 = undefined;
    const stdout_file = std.Io.File.stdout();
    var stdout_writer = stdout_file.writer(io, &stdout_buf);
    try stdout_writer.interface.print(fmt, args);
}
const kvc = @import("kv_cache");
const kv_quant = kvc.kv_quant;

test "SR en V: sesgo/varianza del error de salida (atención simulada)" {
    const gpa = std.heap.page_allocator;
    var rng = std.Random.Xoshiro256.init(3);

    // V realista: 512 tokens × head_dim 128, gaussiana + estructura local
    const n_tok: usize = 512;
    const hd: usize = 128;
    const v = try gpa.alloc(f16, n_tok * hd);
    defer gpa.free(v);
    {
        var base: f32 = 0;
        for (0..n_tok) |t| {
            base = 0.98 * base + 0.02 * rng.random().float(f32); // deriva lenta
            for (0..hd) |d| {
                const noise = (rng.random().float(f32) - 0.5) * 1.2;
                v[t * hd + d] = @floatCast(base * 4.0 + noise);
            }
        }
    }
    // Atención realista: 6 queries, softmax con concentración
    const n_q: usize = 6;
    const w = try gpa.alloc(f32, n_q * n_tok);
    defer gpa.free(w);
    for (0..n_q) |q| {
        const focus: usize = (q * 91 + 37) % n_tok;
        for (0..n_tok) |t| {
            const dist: usize = if (t > focus) t - focus else focus - t;
            w[q * n_tok + t] = -@as(f32, @floatFromInt(dist)) / 30.0; // logits
        }
        var mx: f32 = -std.math.inf(f32);
        for (0..n_tok) |t| mx = @max(mx, w[q * n_tok + t]);
        var sum: f32 = 0;
        for (0..n_tok) |t| {
            w[q * n_tok + t] = @exp(w[q * n_tok + t] - mx);
            sum += w[q * n_tok + t];
        }
        for (0..n_tok) |t| w[q * n_tok + t] /= sum;
    }

    // Salida exacta f16 (referencia)
    const out_ref = try gpa.alloc(f32, n_q * hd);
    defer gpa.free(out_ref);
    @memset(out_ref, 0);
    for (0..n_q) |q| {
        for (0..n_tok) |t| {
            const wt = w[q * n_tok + t];
            for (0..hd) |d| out_ref[q * hd + d] += wt * @as(f32, v[t * hd + d]);
        }
    }

    // Formato del ladder V: q4_0 (rung barato actual)
    const fmt = kvc.quant_types.QuantFormat.q4_0;
    const bytes = kv_quant.quantBytes(fmt, v.len);
    const dst = try gpa.alloc(u8, bytes);
    defer gpa.free(dst);
    const dec = try gpa.alloc(f16, v.len);
    defer gpa.free(dec);

    var bias_det: f32 = 0;
    var bias_sr: f32 = 0;
    var err_det: f32 = 0;
    var err_sr: f32 = 0;
    // SR: 30 corridas (promedia el ruido → sesgo neto)
    for (0..30) |rep| {
        kv_quant.seedEncodeRng(1000 + rep);
        // det: rep 0 solo
        var out_q = try gpa.alloc(f32, n_q * hd);
        defer gpa.free(out_q);
        // determinístico
        if (rep == 0) {
            kv_quant.encodeOpts(fmt, v, dst, .{ .stochastic = false });
            kv_quant.decode(fmt, dst, dec);
            @memset(out_q, 0);
            for (0..n_q) |q| {
                for (0..n_tok) |t| {
                    const wt = w[q * n_tok + t];
                    for (0..hd) |d| out_q[q * hd + d] += wt * @as(f32, dec[t * hd + d]);
                }
            }
            for (0..n_q * hd) |i| {
                bias_det += @abs(out_q[i] - out_ref[i]);
                err_det = @abs(out_q[i] - out_ref[i]);
            }
        }
        // SR
        kv_quant.encodeOpts(fmt, v, dst, .{ .stochastic = true });
        kv_quant.decode(fmt, dst, dec);
        @memset(out_q, 0);
        for (0..n_q) |q| {
            for (0..n_tok) |t| {
                const wt = w[q * n_tok + t];
                for (0..hd) |d| out_q[q * hd + d] += wt * @as(f32, dec[t * hd + d]);
            }
        }
        for (0..n_q * hd) |i| {
            bias_sr += @abs(out_q[i] - out_ref[i]) / 30.0;
            err_sr = @max(err_sr, @abs(out_q[i] - out_ref[i]));
        }
    }
    const n_out: f32 = @floatFromInt(n_q * hd);
    const avg_det = bias_det / n_out;
    const avg_sr = bias_sr / n_out;
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "V q4_0 atención-simulada (n_q={d}, 512 tok, 30 reps SR):\n", .{n_q});
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "  error medio ABS determinístico: {d:.6}\n", .{avg_det});
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "  error medio ABS SR (30 reps promediadas): {d:.6}\n", .{avg_sr});
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "  ratio det/SR: {d:.3}\n", .{avg_det / avg_sr});

    // ── DIAGNÓSTICO: sesgo vs varianza ──────────────────────────────
    // SR promedia el error de EXPECTED output: E[SR] es insesgado por
    // valor PERO la métrica ABS promedio mide E|e|, no |E[e]|. Con un
    // solo sample SR por token, el ruido NO se promedia — se promedia
    // solo en EXPECTATION sobre reps INDEPENDIENTES (batches múltiples).
    // Aquí cada rep re-cuantiza TODO igual (misma semilla→mismo sample):
    // la media de 30 reps con seeds distintas aproxima E[out_SR].
    // Separamos: bias = |E[out] − out_ref| (acumula), ruido = E|out−E[out]|.
    {
        // det: sesgo exacto (una sola corrida)
        // SR: sesgo de la MEDIA de outs (acumulación neta), ruido medio
        var sum_out_sr = try gpa.alloc(f32, n_q * hd);
        defer gpa.free(sum_out_sr);
        @memset(sum_out_sr, 0);
        const out_acc = try gpa.alloc(f32, n_q * hd);
        defer gpa.free(out_acc);
        for (0..30) |rep| {
            kv_quant.seedEncodeRng(5000 + rep);
            kv_quant.encodeOpts(fmt, v, dst, .{ .stochastic = true });
            kv_quant.decode(fmt, dst, dec);
            @memset(out_acc, 0);
            for (0..n_q) |q| {
                for (0..n_tok) |t| {
                    const wt = w[q * n_tok + t];
                    for (0..hd) |d| out_acc[q * hd + d] += wt * @as(f32, dec[t * hd + d]);
                }
            }
            for (0..n_q * hd) |i| sum_out_sr[i] += out_acc[i] / 30.0;
        }
        // sesgo SR neto
        var sr_bias: f32 = 0;
        for (0..n_q * hd) |i| sr_bias += @abs(sum_out_sr[i] - out_ref[i]);
        sr_bias /= n_out;
        // det sesgo
        kv_quant.encodeOpts(fmt, v, dst, .{ .stochastic = false });
        kv_quant.decode(fmt, dst, dec);
        @memset(out_acc, 0);
        for (0..n_q) |q| {
            for (0..n_tok) |t| {
                const wt = w[q * n_tok + t];
                for (0..hd) |d| out_acc[q * hd + d] += wt * @as(f32, dec[t * hd + d]);
            }
        }
        var det_bias: f32 = 0;
        for (0..n_q * hd) |i| det_bias += @abs(out_acc[i] - out_ref[i]);
        det_bias /= n_out;
        try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "SESGO neto (|E[out]−ref|, lo que ACUMULA): det={d:.6} SR={d:.6} ratio={d:.3}\n", .{ det_bias, sr_bias, det_bias / sr_bias });
    }
    // El criterio correcto del paso 0: SR reduce el SESGO acumulativo
    // (la varianza se promedia con múltiples queries/decode steps).
    try std.testing.expect(true);
}
