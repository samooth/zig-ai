//! STUDY §5.2 WY (1.4): paridad del oráculo WY-chunk vs per-token.
//! El oráculo CPU (src/transformer/prefill_wy.zig) factoriza la recurrencia
//! DeltaNet en chunks de CS=64 tokens via representación WY; este test
//! valida rel < 1e-3 (out) y < 2e-3 (state) contra la recurrencia
//! secuencial canónica en 4 geometrías de chunk (parcial, exacto,
//! chunk+tail, multi-chunk) con gate NO-cero, GQA real (hk = hv mod n_k)
//! y estado inicial no trivial.

const std = @import("std");
const wy = @import("transformer").prefill_wy;

test "wyPrefill: paridad vs per-token (4 geometrías, gate≠0, GQA, estado≠0)" {
    const allocator = std.testing.allocator;

    const S = 16;
    const n_v_heads = 4;
    const n_k_heads = 2;
    const key_dim = n_k_heads * S;
    const d_inner = n_v_heads * S;
    const dt_rank = n_v_heads;

    const cases = [_]usize{ 30, 64, 75, 130 };
    for (cases) |n| {
        var prng = std.Random.DefaultPrng.init(0xC0FFEE);
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
        for (q) |*x| x.* = rand.float(f32) * 0.4 - 0.2;
        for (k) |*x| x.* = rand.float(f32) * 0.4 - 0.2;
        for (v) |*x| x.* = rand.float(f32) * 0.4 - 0.2;
        for (beta) |*x| x.* = rand.float(f32);
        for (gate) |*x| x.* = rand.float(f32) * 0.2 - 0.1;

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

        // Referencia per-token (misma recurrencia que
        // SsmLayer.deltaNetRecurrence, standalone sin pesos).
        {
            const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(S)));
            for (0..n) |t| {
                for (0..n_v_heads) |hv| {
                    const hk = hv % n_k_heads;
                    const g = @exp(gate[t * dt_rank + hv]);
                    const b = beta[t * dt_rank + hv];
                    const qb = t * key_dim + hk * S;
                    const vb = t * d_inner + hv * S;
                    const sb = hv * S * S;
                    var d_buf: [128]f64 = undefined;
                    for (0..S * S) |i| state_ref[sb + i] *= g;
                    for (0..S) |j| {
                        var sk: f64 = 0;
                        for (0..S) |i| {
                            sk += @as(f64, state_ref[sb + i * S + j]) * @as(f64, k[qb + i]);
                        }
                        d_buf[j] = @as(f64, b) * (@as(f64, v[vb + j]) - sk);
                    }
                    for (0..S) |i| {
                        const kv: f64 = @floatCast(k[qb + i]);
                        for (0..S) |j| {
                            state_ref[sb + i * S + j] += @floatCast(kv * d_buf[j]);
                        }
                    }
                    for (0..S) |j| {
                        var o: f64 = 0;
                        for (0..S) |i| {
                            o += @as(f64, state_ref[sb + i * S + j]) * @as(f64, q[qb + i]);
                        }
                        out_ref[t * d_inner + hv * S + j] = @floatCast(o * scale);
                    }
                }
            }
        }

        var sc = try wy.WyScratch.init(allocator);
        defer sc.deinit(allocator);
        try wy.wyPrefill(&sc, .{
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
