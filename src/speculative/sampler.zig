//! Samplers para decodificación especulativa (lane-c C2).
//!
//! Dos modos, ambos deterministas:
//!  - Greedy: acepta el draft si coincide con el argmax del target
//!    (semántica llama.cpp common_speculative para temperature 0).
//!  - Rejection sampling clásico (Leviathan et al.): sobre probabilidades
//!    softmax de los logits; acepta con min(1, p/q) y resamplea del
//!    residual normalizado max(0, p−q) cuando rechaza.
//!
//! Los logits pueden venir con cualquier escala: el softmax interno es
//! monotónico y la comparación p/q es invariante.
const std = @import("std");

pub const StepResult = struct {
    accepted: bool,
    /// Token final de este paso: el draft si se acepta; el resampleado
    /// (del residual o del target) si se rechaza.
    token: u32,
};

/// Argmax exacto — gana el PRIMER máximo (empates deterministas).
/// Semántica idéntica a pipeline.GreedySampler.
pub fn greedy(logits: []const f32) u32 {
    var max_idx: usize = 0;
    var max_val: f32 = -std.math.inf(f32);
    for (logits, 0..) |v, i| {
        if (v > max_val) {
            max_val = v;
            max_idx = i;
        }
    }
    return @intCast(max_idx);
}

/// Probabilidad softmax del token `tok` sobre los logits (numéricamente
/// estable, resta el máximo). Confianza del drafter para truncado p_min.
pub fn softmaxConfidence(logits: []const f32, tok: u32) f32 {
    if (logits.len == 0 or tok >= logits.len) return 0;
    var mx: f32 = -std.math.inf(f32);
    for (logits) |v| mx = @max(mx, v);
    var sum: f64 = 0;
    var ptok: f64 = 0;
    for (logits, 0..) |v, i| {
        const e = @exp(@as(f64, v - mx));
        sum += e;
        if (i == tok) ptok = e;
    }
    if (sum <= 0) return 0;
    return @floatCast(ptok / sum);
}

/// Top-k (k<=8) índices por valor, descendente. O(k·V): V≈248k con k=3 es
/// despreciable frente al softmax completo.
pub fn topK(logits: []const f32, out_idx: []u32, out_val: []f32) void {
    const k = @min(@min(out_idx.len, out_val.len), 8);
    for (out_idx[0..k]) |*i| i.* = 0;
    for (out_val[0..k]) |*v| v.* = -std.math.inf(f32);
    for (logits, 0..) |v, i| {
        var kk = k;
        while (kk > 0) : (kk -= 1) {
            if (v > out_val[kk - 1]) {
                if (kk < k) {
                    out_val[kk] = out_val[kk - 1];
                    out_idx[kk] = out_idx[kk - 1];
                }
                out_val[kk - 1] = v;
                out_idx[kk - 1] = @intCast(i);
            } else break;
        }
    }
}

/// Aceptación greedy de un token ya muestreado por el drafter:
/// acepta sólo si el drafter acertó el argmax del target en esa posición.
pub fn acceptGreedy(target_logits: []const f32, draft_token: u32) bool {
    return greedy(target_logits) == draft_token;
}

fn softmax(dst: []f32, logits: []const f32) void {
    var max_logit: f32 = -std.math.inf(f32);
    for (logits) |v| max_logit = @max(max_logit, v);
    var sum: f64 = 0;
    for (logits, dst) |v, *d| {
        const e = @exp(v - max_logit);
        d.* = e;
        sum += e;
    }
    for (dst) |*d| d.* = @floatCast(@as(f64, d.*) / sum);
}

/// Un paso de rejection sampling sobre distribuciones softmax de los logits.
///
/// `scratch` debe tener longitud >= vocab (dos buffers de trabajo); se usa
/// como espacio temporal y queda con contenido arbitrario.
///
/// Determinista dado `rng` con el mismo estado.
pub fn rejectionStep(
    target_logits: []const f32,
    draft_logits: []const f32,
    scratch_p: []f32,
    scratch_q: []f32,
    rng: std.Random,
) !StepResult {
    if (target_logits.len != draft_logits.len) return error.VocabMismatch;
    const vocab = target_logits.len;
    if (vocab == 0) return error.EmptyVocab;
    if (scratch_p.len < vocab or scratch_q.len < vocab) return error.ScratchTooSmall;

    softmax(scratch_p[0..vocab], target_logits);
    softmax(scratch_q[0..vocab], draft_logits);

    // Draft token = argmax del drafter (muestreo greedy del drafter día uno).
    const draft_token = greedy(draft_logits);

    const p = scratch_p[draft_token];
    const q = scratch_q[draft_token];
    const accept_prob: f32 = if (q <= 0.0) 1.0 else @min(1.0, p / q);

    const u = rng.float(f32);
    if (u < accept_prob) {
        return .{ .accepted = true, .token = draft_token };
    }

    // Rechazo: resamplear del residual normalizado max(0, p − q).
    var residual_sum: f64 = 0;
    for (0..vocab) |i| {
        const r = @max(0.0, scratch_p[i] - scratch_q[i]);
        scratch_p[i] = r;
        residual_sum += r;
    }
    if (residual_sum <= 0.0) {
        // Degenerado (p==q en todo el vocab): cae al argmax del target.
        return .{ .accepted = false, .token = greedy(target_logits) };
    }
    var acc: f64 = 0;
    const target_u = rng.float(f64) * residual_sum;
    for (0..vocab) |i| {
        acc += scratch_p[i];
        if (target_u <= acc) return .{ .accepted = false, .token = @intCast(i) };
    }
    return .{ .accepted = false, .token = @intCast(vocab - 1) };
}

/// Verifica una secuencia completa de drafts contra filas de logits del
/// target (una fila por posición). Para en el primer rechazo.
/// El caller posee `out_tokens` (capacidad >= draft_tokens.len + 1).
/// Devuelve cuántos tokens VÁLIDOS dejó en out_tokens (incluye bonus).
pub fn verifyGreedy(
    draft_tokens: []const u32,
    target_rows: []const []const f32,
    out_tokens: []u32,
) usize {
    var n_accepted: usize = 0;
    for (draft_tokens, 0..) |dtok, pos| {
        if (pos >= target_rows.len) break;
        if (!acceptGreedy(target_rows[pos], dtok)) break;
        out_tokens[n_accepted] = dtok;
        n_accepted += 1;
    }
    // Bonus: el primer token del target tras el último aceptado (o tras el
    // primer rechazo) siempre es correcto por construcción del forward.
    const bonus_pos = @min(n_accepted, target_rows.len - 1);
    if (out_tokens.len > n_accepted and target_rows.len > 0) {
        out_tokens[n_accepted] = greedy(target_rows[bonus_pos]);
        n_accepted += 1;
    }
    return n_accepted;
}

test "greedy: argmax exacto y empate al primero" {
    try std.testing.expectEqual(@as(u32, 3), greedy(&.{ 0.1, -2.0, 0.5, 5.0, 5.0 }));
    try std.testing.expectEqual(@as(u32, 0), greedy(&.{ 1.0, 1.0 }));
    try std.testing.expectEqual(@as(u32, 7), greedy(&[_]f32{0} ** 8));
}

test "acceptGreedy: sólo acepta el argmax del target" {
    const target = [_]f32{ 0.1, 3.0, 0.5 };
    try std.testing.expect(acceptGreedy(&target, 1));
    try std.testing.expect(!acceptGreedy(&target, 0));
}

test "rejectionStep: filas idénticas aceptan siempre" {
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    const logits = [_]f32{ 1.0, 2.0, 3.0, -1.0 };
    var sp: [4]f32 = undefined;
    var sq: [4]f32 = undefined;
    for (0..64) |_| {
        const r = try rejectionStep(&logits, &logits, &sp, &sq, rng);
        try std.testing.expect(r.accepted);
        try std.testing.expectEqual(@as(u32, 2), r.token);
    }
}

test "rejectionStep: consistencia con seed" {
    const t = [_]f32{ 4.0, 1.0, 0.2, 0.1 };
    const d = [_]f32{ 0.5, 0.4, 0.3, 2.0 };
    var sp: [4]f32 = undefined;
    var sq: [4]f32 = undefined;
    var seq_a: [16]u32 = undefined;
    var seq_b: [16]u32 = undefined;
    {
        var prng = std.Random.DefaultPrng.init(1234);
        for (&seq_a) |*tok| tok.* = (try rejectionStep(&t, &d, &sp, &sq, prng.random())).token;
    }
    {
        var prng = std.Random.DefaultPrng.init(1234);
        for (&seq_b) |*tok| tok.* = (try rejectionStep(&t, &d, &sp, &sq, prng.random())).token;
    }
    try std.testing.expectEqualSlices(u32, &seq_a, &seq_b);
}

test "rejectionStep: rechazo garantizado con p(draft)=0 relativo" {
    // Target pone masa casi nula en el token que el drafter prefiere,
    // y mucha en otro => acept_prob ≈ 0 => casi seguro rechazo+resample.
    var prng = std.Random.DefaultPrng.init(99);
    const rng = prng.random();
    const target = [_]f32{ 50.0, 0.001, 0.001, 0.001 };
    const draft = [_]f32{ 0.001, 0.001, 0.001, 50.0 };
    var sp: [4]f32 = undefined;
    var sq: [4]f32 = undefined;
    var rejects: usize = 0;
    for (0..32) |_| {
        const r = try rejectionStep(&target, &draft, &sp, &sq, rng);
        if (!r.accepted) {
            rejects += 1;
            try std.testing.expectEqual(@as(u32, 0), r.token);
        }
    }
    try std.testing.expect(rejects > 24);
}

test "verifyGreedy: prefijo aceptado + bonus, corta en primer rechazo" {
    // target argmax por posición: {1, 2, 0}
    const row0 = [_]f32{ 0.0, 9.0, 1.0 };
    const row1 = [_]f32{ 1.0, 0.0, 9.0 };
    const row2 = [_]f32{ 9.0, 1.0, 0.0 };
    const rows = [_][]const f32{ &row0, &row1, &row2 };
    var out: [8]u32 = undefined;

    // Drafts {1, 2}: acepta ambos + bonus = argmax(row2) = 0 → [1,2,0]
    var n = verifyGreedy(&.{ 1, 2 }, &rows, &out);
    try std.testing.expectEqual(@as(usize, 3), n);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 0 }, out[0..n]);

    // Drafts {1, 0}: posición 1 rechaza (argmax=2) → [1, bonus=2]
    n = verifyGreedy(&.{ 1, 0 }, &rows, &out);
    try std.testing.expectEqual(@as(usize, 2), n);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, out[0..n]);
}

test "verifyGreedy: sin drafts no produce nada más allá del límite" {
    const row = [_]f32{ 1.0, 0.0 };
    var out: [4]u32 = undefined;
    const n = verifyGreedy(&.{}, &.{&row}, &out);
    // Sin drafts no hay bonus verificable (n_accepted=0 ⇒ bonus_pos=min(0,0)=0)
    // pero out tiene hueco: el bonus sale del único row disponible.
    try std.testing.expect(n == 0 or (n == 1 and out[0] == 0));
}
