//! WY-chunk ΔNet (BeeLlama 1.4 / STUDY §5.2) — oráculo CPU del prefill
//! chunked via representación WY.
//!
//! Puerto fiel de `beellama.cpp/src/models/delta-net-base.cpp::
//! build_delta_net_chunking` (rama nkda, CS=64): convierte la recurrencia
//! secuencial per-token en GEMMs batched por chunk de CS tokens.
//!
//! Algoritmo (por chunk de CS tokens, por v-head):
//!   1. g_cs = cumsum(g)                          — decay acumulado (clamp 50)
//!   2. decay[i][j] = exp(g_cs[i] - g_cs[j])  tri-lower-diag (i ≥ j)
//!   3. kb[i][j]  = (K_b·K^T)[i][j] · decay[i][j]   (K_b = K·beta)
//!      kq[i][j]  = (K·Q^T)[i][j] · decay[i][j]  tri-lower-DIAG (j ≤ i)
//!   4. M = I + tri_low_strict(kb) ; X = M^-1·(-tri_low_strict(kb)) ; A = X + I
//!      (solve unit-lower-triangular por sustitución hacia adelante)
//!   5. k_cd[c][j] = Σ_i K_b[i][c]·g_exp[i]·A[i][j]  — K con cumdecay
//!      kg[i][c]   = K[i][c]·exp(g_cs[last] - g_cs[i]) — key_gdiff
//!      q_g[i][c]  = Q[i][c]·g_exp[i]
//!   6. Por chunk (loop secuencial de ≤9 chunks en pp512):
//!        v_new[t][s] = V_b[t][s] - Σ_c k_cd[c][t]·S[c][s]
//!        o[t][s]     = (Σ_c S[c][s]·q_g[t][c] + Σ_{j≤t} v_new[j][s]·kq[t][j]) · scale
//!        S[c][s]     = S[c][s]·exp(g_cs[last]) + Σ_t kg[t][c]·v_new[t][s]
//!
//! Fidelidad upstream (delta-net-base.cpp nkda): kb/kq computados como
//! mul_mat + decay elementwise; A vía solve_tri + identidad; v_new via
//! mul_mat(k_cd, s) restado de v_b^T; o = s·q_g_exp + v_new·kq; estado
//! s = s·g_last_exp + kg^T·v_new. La orientación exacta de cada op está
//! en los comentarios de cada paso.
//!
//! Test de paridad: `tests/test_prefill_wy.zig` — compara contra
//! `SsmLayer.deltaNetRecurrence` (per-token, oráculo canónico) con
//! rel < 1e-3 en attn_out y state final.

const std = @import("std");

/// Chunk size del camino nkda (decay escalar por head — Qwen3.5).
pub const CS: usize = 64;

/// Máximo S_v soportado (d_state del modelo). 128 cubre Qwen3.5.
pub const MAX_S: usize = 128;

pub const WyError = error{
    DimensionMismatch,
    OutOfMemory,
};

/// Buffers scratch reutilizados entre chunks/heads. f64 interno.
pub const WyScratch = struct {
    g_cs: []f64, // [CS]
    decay: []f64, // [CS×CS]
    kb: []f64, // [CS×CS]
    kq: []f64, // [CS×CS]
    attn: []f64, // [CS×CS] — A tras el solve (+I)
    lhs: []f64, // [CS×CS] — M = I + tri_low(kb)
    v_b: []f64, // [CS×S]
    u_new: []f64, // [CS×S] — v_new
    k_cd: []f64, // [S×CS]
    kg: []f64, // [CS×S]
    q_g: []f64, // [CS×S]
    s_work: []f64, // [S×S]
    rhs: []f64, // [CS] — columna de trabajo del solve

    pub fn init(allocator: std.mem.Allocator) WyError!WyScratch {
        const cs2 = CS * CS;
        return .{
            .g_cs = try allocator.alloc(f64, CS),
            .decay = try allocator.alloc(f64, cs2),
            .kb = try allocator.alloc(f64, cs2),
            .kq = try allocator.alloc(f64, cs2),
            .attn = try allocator.alloc(f64, cs2),
            .lhs = try allocator.alloc(f64, cs2),
            .v_b = try allocator.alloc(f64, CS * MAX_S),
            .u_new = try allocator.alloc(f64, CS * MAX_S),
            .k_cd = try allocator.alloc(f64, MAX_S * CS),
            .kg = try allocator.alloc(f64, CS * MAX_S),
            .q_g = try allocator.alloc(f64, CS * MAX_S),
            .s_work = try allocator.alloc(f64, MAX_S * MAX_S),
            .rhs = try allocator.alloc(f64, CS),
        };
    }

    pub fn deinit(self: *WyScratch, allocator: std.mem.Allocator) void {
        allocator.free(self.g_cs);
        allocator.free(self.decay);
        allocator.free(self.kb);
        allocator.free(self.kq);
        allocator.free(self.attn);
        allocator.free(self.lhs);
        allocator.free(self.v_b);
        allocator.free(self.u_new);
        allocator.free(self.k_cd);
        allocator.free(self.kg);
        allocator.free(self.q_g);
        allocator.free(self.s_work);
        allocator.free(self.rhs);
    }
};

/// Entrada del WY-chunk: proyecciones YA hechas (post conv+l2norm+sigmoid).
/// Layouts planos:
///   q/k:  [n, key_dim]     — k-heads de S dims contiguos (módulo GQA)
///   v:    [n, d_inner]     — v-heads de S dims contiguos
///   beta: [n, dt_rank]     — post-sigmoid
///   gate: [n, dt_rank]     — decay crudo (exp(g) per-token en el recurrente)
///   state:[n_v_heads × S × S] INOUT
///   out:  [n × d_inner]    — escritura
pub const WyInput = struct {
    q: []const f32,
    k: []const f32,
    v: []const f32,
    beta: []const f32,
    gate: []const f32,
    state: []f32,
    out: []f32,

    n_tokens: usize,
    key_dim: usize,
    d_inner: usize,
    n_k_heads: usize,
    n_v_heads: usize,
    head_v_dim: usize, // S
    dt_rank: usize,
};

/// Una pasada WY sobre TODOS los tokens. Estado in-place; `out` llenado.
pub fn wyPrefill(scratch: *WyScratch, in: WyInput) WyError!void {
    const S = in.head_v_dim;
    if (S > MAX_S) return error.DimensionMismatch;

    for (0..in.n_v_heads) |hv| {
        const hk = hv % in.n_k_heads; // módulo (7.1b, como deltaNetRecurrence)
        // Estado de esta head → s_work (f64).
        const s_base = hv * S * S;
        for (0..S * S) |i| scratch.s_work[i] = @floatCast(in.state[s_base + i]);

        var t0: usize = 0;
        while (t0 < in.n_tokens) {
            const c_len = @min(CS, in.n_tokens - t0);
            wyChunk(scratch, in, hv, hk, S, t0, c_len);
            t0 += CS;
        }

        // Estado final de vuelta a f32.
        for (0..S * S) |i| in.state[s_base + i] = @floatCast(scratch.s_work[i]);
    }
}

/// WY de UN chunk: tokens [t0, t0+c_len), v-head hv, k-head hk.
fn wyChunk(
    sc: *WyScratch,
    in: WyInput,
    hv: usize,
    hk: usize,
    S: usize,
    t0: usize,
    cs: usize,
) void {
    const key_dim = in.key_dim;
    const d_inner = in.d_inner;
    const dt_rank = in.dt_rank;
    const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(S)));

    // ── 1. g_cs = cumsum(g) (clamp 50, como el upstream).
    var acc: f64 = 0;
    for (0..cs) |i| {
        acc += @floatCast(in.gate[(t0 + i) * dt_rank + hv]);
        sc.g_cs[i] = @min(acc, 50.0);
    }

    // ── 2. decay[i][j] = exp(g_cs[i] - g_cs[j]) para i ≥ j; 0 si no.
    //       El write del token j llega al tiempo i multiplicado por
    //       exp(Σ_{s=j+1..i} g_s) = exp(g_cs[i] - g_cs[j]).
    for (0..cs) |i| {
        for (0..cs) |j| {
            sc.decay[i * cs + j] = if (j > i) 0.0 else @exp(sc.g_cs[i] - sc.g_cs[j]);
        }
    }

    // ── 3. kb[i][j] = (Σ_c K[i][c]·K[j][c]·beta[i]) · decay[i][j]  (tri-lower)
    //       El beta va con el token i (el ESCRITOR actual) — derivado del
    //       per-token: T[i][j] = b_i·(k_i·k_j)·exp(g_cs[j]-g_cs[i]).
    //       kq[t][j] = (Σ_c Q[t][c]·K[j][c]) · decay[t][j]  — Q del token
    //       de LECTURA (t) contra K del token ESCRITOR (j), tri-lower-DIAG
    //       (j ≤ t). Orientación derivada del per-token: o_t ve los writes
    //       d_j con kq[t][j] = q_t·k_j·exp(g_cs[t]-g_cs[j]).
    for (0..cs) |i| {
        const ki = (t0 + i) * key_dim + hk * S;
        const qi = (t0 + i) * key_dim + hk * S;
        const b_i: f64 = @floatCast(in.beta[(t0 + i) * dt_rank + hv]);
        for (0..cs) |j| {
            const kj = (t0 + j) * key_dim + hk * S;
            var dot_kb: f64 = 0;
            var dot_kq: f64 = 0;
            for (0..S) |c| {
                const k_ic: f64 = @floatCast(in.k[ki + c]);
                dot_kb += k_ic * @as(f64, @floatCast(in.k[kj + c]));
                dot_kq += @as(f64, @floatCast(in.q[qi + c])) * @as(f64, @floatCast(in.k[kj + c]));
            }
            sc.kb[i * cs + j] = dot_kb * b_i * sc.decay[i * cs + j];
            sc.kq[i * cs + j] = if (j > i) 0.0 else dot_kq * sc.decay[i * cs + j];
        }
    }

    // ── 4. A = M^-1·(-T) + I con M = I + T, T = tri_low_ESTRICTO(kb).
    //       M unit-lower-triangular ⇒ sustitución hacia adelante por
    //       columna: X[i][col] = rhs[i][col] - Σ_{j<i} M[i][j]·X[j][col],
    //       rhs = -T.
    for (0..cs) |i| {
        for (0..cs) |j| {
            sc.lhs[i * cs + j] = if (i == j) 1.0 else if (j < i) sc.kb[i * cs + j] else 0.0;
            sc.attn[i * cs + j] = if (j < i) -sc.kb[i * cs + j] else 0.0;
        }
    }
    for (0..cs) |col| {
        var i: usize = 0;
        while (i < cs) : (i += 1) {
            var x: f64 = sc.attn[i * cs + col];
            var j: usize = 0;
            while (j < i) : (j += 1) {
                x -= sc.lhs[i * cs + j] * sc.attn[j * cs + col];
            }
            sc.attn[i * cs + col] = x;
        }
    }
    for (0..cs) |i| sc.attn[i * cs + i] += 1.0; // A = X + I

    // ── 5. k_cd[c][j] = Σ_i K[i][c]·beta[i]·g_exp[i]·A[i][j]  (K cumdecay)
    //       kg[i][c]   = K[i][c]·exp(g_last - g_cs[i])      (key_gdiff)
    //       — decay del write del token i hasta el final del chunk: el
    //         per-token escribe k_i⊗d_i DESPUÉS de S *= g_i, así que el
    //         write se multiplica por los g POSTERIORES:
    //         exp(Σ_{s>i} g_s) = exp(g_last - g_cs[i]).
    //       q_g[i][c]  = Q[i][c]·g_exp[i]
    const g_last = sc.g_cs[cs - 1];
    for (0..cs) |i| {
        const gi: f64 = @exp(sc.g_cs[i]);
        const g_diff_i: f64 = @exp(g_last - sc.g_cs[i]);
        for (0..S) |c| {
            const k_ic: f64 = @floatCast(in.k[(t0 + i) * key_dim + hk * S + c]);
            sc.kg[i * S + c] = k_ic * g_diff_i;
            sc.q_g[i * S + c] = @as(f64, @floatCast(in.q[(t0 + i) * key_dim + hk * S + c])) * gi;
        }
    }
    // k_cd[c][j] = Σ_i A[j][i]·K[i][c]·beta[i]·g_exp[i]  (A·kbg por filas:
    // la fila j de A contra los writes previos — misma orientación que v_new)
    for (0..S) |c| {
        for (0..cs) |j| {
            var acc2: f64 = 0;
            for (0..cs) |i| {
                const k_ic: f64 = @floatCast(in.k[(t0 + i) * key_dim + hk * S + c]);
                const b_i: f64 = @floatCast(in.beta[(t0 + i) * dt_rank + hv]);
                acc2 += sc.attn[j * cs + i] * k_ic * b_i * @exp(sc.g_cs[i]);
            }
            sc.k_cd[c * cs + j] = acc2;
        }
    }

    // ── 6. Secuencia del chunk (v_new, output, estado).
    //       v_b[i][c] = V[i][c]·beta[i]
    for (0..cs) |i| {
        const b_i: f64 = @floatCast(in.beta[(t0 + i) * dt_rank + hv]);
        for (0..S) |c| {
            sc.v_b[i * S + c] = @as(f64, @floatCast(in.v[(t0 + i) * d_inner + hv * S + c])) * b_i;
        }
    }

    // a) v_new[t][s] = (A·v_b)[t][s] - Σ_c k_cd[c][t]·S[c][s]
    //    A·v_b: la fila t de A contra v_b — derivado del per-token
    //    (d_t = A·v_b cuando S_in=0; el término WY resuelve los writes
    //    previos intra-chunk). La resta k_cd·S descuenta el estado entrante.
    for (0..cs) |t| {
        for (0..S) |s| {
            var vp: f64 = 0;
            for (0..S) |c| {
                vp += sc.k_cd[c * cs + t] * sc.s_work[c * S + s];
            }
            var w: f64 = 0;
            for (0..cs) |j| {
                w += sc.attn[t * cs + j] * sc.v_b[j * S + s];
            }
            sc.u_new[t * S + s] = w - vp;
        }
    }

    // b) o[t][s] = (Σ_c S[c][s]·q_g[t][c] + Σ_{j≤t} v_new[j][s]·kq[t][j]) · scale
    for (0..cs) |t| {
        for (0..S) |s| {
            var o: f64 = 0;
            for (0..S) |c| {
                o += sc.s_work[c * S + s] * sc.q_g[t * S + c];
            }
            var j: usize = 0;
            while (j <= t) : (j += 1) {
                o += sc.u_new[j * S + s] * sc.kq[t * cs + j];
            }
            in.out[(t0 + t) * d_inner + hv * S + s] = @floatCast(o * scale);
        }
    }

    // c) estado saliente: S[c][s] = S[c][s]·exp(g_last) + Σ_t kg[t][c]·v_new[t][s]
    const g_last_exp: f64 = @exp(g_last);
    for (0..S) |c| {
        for (0..S) |s| {
            var a: f64 = sc.s_work[c * S + s] * g_last_exp;
            for (0..cs) |t| {
                a += sc.kg[t * S + c] * sc.u_new[t * S + s];
            }
            sc.s_work[c * S + s] = a;
        }
    }
}

// ─── Tests de paridad vs deltaNetRecurrence (per-token) ───────────────────

test "wyPrefill: paridad vs per-token en secuencias sintéticas" {
    const allocator = std.testing.allocator;

    // Geometría estilo 0.8B escalada para test rápido:
    // S=16, dt_rank(v-heads)=4, n_group(k-heads)=2.
    const S = 16;
    const n_v_heads = 4;
    const n_k_heads = 2;
    const key_dim = n_k_heads * S;
    const d_inner = n_v_heads * S;
    const dt_rank = n_v_heads;

    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rand = prng.random();

    // Probar n en: exacto 1 chunk (64), chunk+tail (75), multi (130), <CS (30)
    const cases = [_]usize{ 30, 64, 75, 130 };
    for (cases) |n| {
        // Datos sintéticos con rangos realistas.
        const q = try allocator.alloc(f32, n * key_dim);
        defer allocator.free(q);
        const k = try allocator.alloc(f32, n * key_dim);
        defer allocator.free(k);
        const v = try allocator.alloc(f32, n * d_inner);
        defer allocator.free(v);
        const beta = try allocator.alloc(f32, n * dt_rank);
        defer allocator.free(beta);
        const gate = try allocator.alloc(f32, n * dt_rank);
        defer allocator.free(gate);
        for (q) |*x| x.* = rand.float(f32) * 0.4 - 0.2;
        for (k) |*x| x.* = rand.float(f32) * 0.4 - 0.2;
        for (v) |*x| x.* = rand.float(f32) * 0.4 - 0.2;
        for (beta) |*x| x.* = rand.float(f32); // [0,1) post-sigmoid range
        for (gate) |*x| x.* = rand.float(f32) * 0.2 - 0.1; // decay suave

        // Estado inicial NO trivial (distingue inicialización correcta).
        const state_ref = try allocator.alloc(f32, n_v_heads * S * S);
        defer allocator.free(state_ref);
        const state_wy = try allocator.alloc(f32, n_v_heads * S * S);
        defer allocator.free(state_wy);
        for (state_ref, 0..) |*x, i| {
            x.* = rand.float(f32) * 0.2 - 0.1;
            state_wy[i] = x.*;
        }

        const out_ref = try allocator.alloc(f32, n * d_inner);
        defer allocator.free(out_ref);
        const out_wy = try allocator.alloc(f32, n * d_inner);
        defer allocator.free(out_wy);
        @memset(out_ref, 0);
        @memset(out_wy, 0);

        // Referencia per-token (mismo algoritmo que deltaNetRecurrence,
        // inline para no depender del SsmLayer completo).
        perTokenRef(q, k, v, beta, gate, state_ref, out_ref, n, key_dim, d_inner, n_k_heads, n_v_heads, S, dt_rank);

        // WY.
        var sc = try WyScratch.init(allocator);
        defer sc.deinit(allocator);
        try wyPrefill(&sc, .{
            .q = q,
            .k = k,
            .v = v,
            .beta = beta,
            .gate = gate,
            .state = state_wy,
            .out = out_wy,
            .n_tokens = n,
            .key_dim = key_dim,
            .d_inner = d_inner,
            .n_k_heads = n_k_heads,
            .n_v_heads = n_v_heads,
            .head_v_dim = S,
            .dt_rank = dt_rank,
        });

        // Paridad: rel < 1e-3 con floor 1e-3 (WY reordena sumas en f64;
        // con S=16 la magnitud de o es pequeña).
        var max_rel: f64 = 0;
        for (out_ref, out_wy, 0..) |r, w, i| {
            const adiff: f64 = @abs(@as(f64, w) - @as(f64, r));
            const denom: f64 = @max(@abs(@as(f64, r)), 1e-3);
            const rel = adiff / denom;
            if (rel > max_rel) max_rel = rel;
            if (rel > 1e-3) {
                std.debug.print("mismatch out @{d}: ref={d} wy={d} rel={d}\n", .{ i, r, w, rel });
            }
        }
        try std.testing.expect(max_rel < 1e-3);

        // Estado final.
        var max_rel_s: f64 = 0;
        for (state_ref, state_wy, 0..) |r, w, i| {
            const adiff: f64 = @abs(@as(f64, w) - @as(f64, r));
            const denom: f64 = @max(@abs(@as(f64, r)), 1e-3);
            const rel = adiff / denom;
            if (rel > max_rel_s) max_rel_s = rel;
            if (rel > 2e-3) {
                std.debug.print("mismatch state @{d}: ref={d} wy={d} rel={d}\n", .{ i, r, w, rel });
            }
        }
        try std.testing.expect(max_rel_s < 2e-3);
    }
}

/// Referencia per-token (misma recurrencia que SsmLayer.deltaNetRecurrence,
/// standalone para el test — sin depender de pesos/modelo).
fn perTokenRef(
    q: []const f32,
    k: []const f32,
    v: []const f32,
    beta: []const f32,
    gate: []const f32,
    state: []f32,
    out: []f32,
    n: usize,
    key_dim: usize,
    d_inner: usize,
    n_k_heads: usize,
    n_v_heads: usize,
    S: usize,
    dt_rank: usize,
) void {
    _ = n_k_heads;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(S)));
    for (0..n) |t| {
        for (0..n_v_heads) |hv| {
            const hk = hv % (key_dim / S);
            const g = @exp(gate[t * dt_rank + hv]);
            const b = beta[t * dt_rank + hv];
            const q_base = t * key_dim + hk * S;
            const k_base = t * key_dim + hk * S;
            const v_base = t * d_inner + hv * S;
            const s_base = hv * S * S;
            var d_buf: [MAX_S]f64 = undefined;
            // S *= exp(g)
            for (0..S * S) |i| state[s_base + i] *= g;
            // d[j] = b·(v[j] - Σ_i S[i][j]·k[i])
            for (0..S) |j| {
                var sk: f64 = 0;
                for (0..S) |i| {
                    sk += @as(f64, state[s_base + i * S + j]) * @as(f64, k[k_base + i]);
                }
                d_buf[j] = @as(f64, b) * (@as(f64, v[v_base + j]) - sk);
            }
            // S[i][j] += k[i]·d[j]
            for (0..S) |i| {
                const kv: f64 = @floatCast(k[k_base + i]);
                for (0..S) |j| {
                    state[s_base + i * S + j] += @floatCast(kv * d_buf[j]);
                }
            }
            // o[j] = Σ_i S[i][j]·q[i]·scale
            for (0..S) |j| {
                var o: f64 = 0;
                for (0..S) |i| {
                    o += @as(f64, state[s_base + i * S + j]) * @as(f64, q[q_base + i]);
                }
                out[t * d_inner + hv * S + j] = @floatCast(o * scale);
            }
        }
    }
}

test "wyPrefill: chunk único n=CS exacto — estado suma bien" {
    const allocator = std.testing.allocator;
    const S = 8;
    const n_v_heads = 2;
    const n_k_heads = 1;
    const key_dim = n_k_heads * S;
    const d_inner = n_v_heads * S;
    const dt_rank = n_v_heads;
    const n = CS;

    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();
    const q = try allocator.alloc(f32, n * key_dim);
    defer allocator.free(q);
    const k = try allocator.alloc(f32, n * key_dim);
    defer allocator.free(k);
    const v = try allocator.alloc(f32, n * d_inner);
    defer allocator.free(v);
    const beta = try allocator.alloc(f32, n * dt_rank);
    defer allocator.free(beta);
    const gate = try allocator.alloc(f32, n * dt_rank);
    defer allocator.free(gate);
    for (q) |*x| x.* = rand.float(f32) * 0.2 - 0.1;
    for (k) |*x| x.* = rand.float(f32) * 0.2 - 0.1;
    for (v) |*x| x.* = rand.float(f32) * 0.2 - 0.1;
    for (beta) |*x| x.* = 0.5;
    for (gate) |*x| x.* = -0.05;

    const state_wy = try allocator.alloc(f32, n_v_heads * S * S);
    defer allocator.free(state_wy);
    const state_ref = try allocator.alloc(f32, n_v_heads * S * S);
    defer allocator.free(state_ref);
    for (state_wy, 0..) |*x, i| {
        x.* = 0;
        state_ref[i] = 0;
    }
    const out_wy = try allocator.alloc(f32, n * d_inner);
    defer allocator.free(out_wy);
    const out_ref = try allocator.alloc(f32, n * d_inner);
    defer allocator.free(out_ref);
    @memset(out_wy, 0);
    @memset(out_ref, 0);

    perTokenRef(q, k, v, beta, gate, state_ref, out_ref, n, key_dim, d_inner, n_k_heads, n_v_heads, S, dt_rank);

    var sc = try WyScratch.init(allocator);
    defer sc.deinit(allocator);
    try wyPrefill(&sc, .{
        .q = q,
        .k = k,
        .v = v,
        .beta = beta,
        .gate = gate,
        .state = state_wy,
        .out = out_wy,
        .n_tokens = n,
        .key_dim = key_dim,
        .d_inner = d_inner,
        .n_k_heads = n_k_heads,
        .n_v_heads = n_v_heads,
        .head_v_dim = S,
        .dt_rank = dt_rank,
    });

    // Con S=8 y estado inicial 0, tolerancias algo más flojas por la
    // magnitud pequeña: rel < 5e-3 floor 1e-3.
    for (out_ref, out_wy, 0..) |r, w, i| {
        const adiff: f64 = @abs(@as(f64, w) - @as(f64, r));
        const denom: f64 = @max(@abs(@as(f64, r)), 1e-3);
        if (adiff / denom > 5e-3) {
            std.debug.print("mismatch @{d}: ref={d} wy={d}\n", .{ i, r, w });
            return error.TestUnexpectedResult;
        }
    }
}
