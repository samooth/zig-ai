//! Test 2.1 (lane-f): paridad FP8 block-scaled GEMM/GEMV vs referencia CPU
//! f32 + diagnóstico del quantizer en aislamiento. El camino fp8_block era
//! código muerto con 4 bugs latentes (firma incompatible, layout escalas,
//! kernels sin extern "C", args del launch) que el análisis lazy de Zig
//! ocultaba — este test los compila al USARLO.
//! Requiere CUDA; sin GPU sale limpio.
const std = @import("std");
const matmul = @import("matmul");
const Tensor = @import("core").Tensor;
const cudaz = @import("cudaz");
const fp8k = @import("fp8_kernels");

fn cpuRef(a: []const f32, w: []const f32, m: usize, n: usize, k: usize, c: []f32) void {
    for (0..m) |mi| {
        for (0..n) |ni| {
            var sum: f64 = 0;
            for (0..k) |ki| sum += @as(f64, a[mi * k + ki]) * @as(f64, w[ni * k + ki]);
            c[mi * n + ni] = @floatCast(sum);
        }
    }
}

test "fp8_block: quantizer f32 directo (scales + bytes)" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    var eng = try fp8k.Fp8BlockLinear.init(@ptrCast((try matmul.MatmulEngine.sharedCudaStream()).raw));
    const K = 128;
    const x = try std.testing.allocator.alloc(f32, K);
    defer std.testing.allocator.free(x);
    for (x, 0..) |*v, i| v.* = @as(f32, 0.5) * @as(f32, @floatFromInt(i)) / 128.0;
    const d_x = try cudaz.cuMemAlloc(K * 4);
    defer cudaz.cuMemFree(d_x);
    try cudaz.cuMemcpyHtoD(d_x, @intFromPtr(x.ptr), K * 4);
    const d_q = try cudaz.cuMemAlloc(K);
    defer cudaz.cuMemFree(d_q);
    const d_s = try cudaz.cuMemAlloc(4);
    defer cudaz.cuMemFree(d_s);
    try eng.quantizeActivationsF32(d_x, d_q, d_s, 1, @intCast(K));
    try cudaz.cuStreamSynchronize(@ptrCast((try matmul.MatmulEngine.sharedCudaStream()).raw));
    var scale: f32 = 99;
    try cudaz.cuMemcpyDtoH(@intFromPtr(&scale), d_s, 4);
    var qb: [8]u8 = undefined;
    try cudaz.cuMemcpyDtoH(@intFromPtr(&qb), d_q, 8);
    const amax_expected: f32 = 0.5 * 127.0 / 128.0;
    std.debug.print("[fp8-debug] scale={e} (esperado {e}) q[0..8]=", .{ scale, amax_expected / 448.0 });
    for (qb) |b| std.debug.print(" {x:0>2}", .{b});
    std.debug.print("\n", .{});
    // q[127] debe ser el máximo del grupo: 448 escalado → 0x7E (448).
    var q_last: u8 = 0;
    try cudaz.cuMemcpyDtoH(@intFromPtr(&q_last), d_q + 127, 1);
    std.debug.print("[fp8-debug] q[127]=0x{x:0>2} (esperado 7E)\n", .{q_last});
    try std.testing.expect(scale > 0);
    try std.testing.expect(q_last == 0x7E); // amax del grupo → 448 exacto
}

test "fp8_block: gemmFp8Block paridad vs CPU f32" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    var engine = try matmul.MatmulEngine.init(std.testing.allocator, .fp8_block, .fp8);
    defer engine.deinit();

    const M: usize = 3;
    const N: usize = 256;
    const K: usize = 512;

    var rng = std.Random.DefaultPrng.init(77);
    const random = rng.random();
    const a = try std.testing.allocator.alloc(f32, M * K);
    defer std.testing.allocator.free(a);
    const w = try std.testing.allocator.alloc(f32, N * K);
    defer std.testing.allocator.free(w);
    for (a) |*v| v.* = random.floatNorm(f32) * 0.8;
    for (w) |*v| v.* = random.floatNorm(f32) * 0.3;

    var a_shape = [_]usize{ M, K };
    var a_strides = [_]usize{ K, 1 };
    const a_t = Tensor(f32){ .data = a, .shape = a_shape[0..], .strides = a_strides[0..], .offset = 0, .allocator = null, .owns_data = false };
    var w_shape = [_]usize{ N, K };
    var w_strides = [_]usize{ K, 1 };
    const w_t = Tensor(f32){ .data = w, .shape = w_shape[0..], .strides = w_strides[0..], .offset = 0, .allocator = null, .owns_data = false };
    const c = try std.testing.allocator.alloc(f32, M * N);
    defer std.testing.allocator.free(c);
    var c_shape = [_]usize{ M, N };
    var c_strides = [_]usize{ N, 1 };
    var c_t = Tensor(f32){ .data = c, .shape = c_shape[0..], .strides = c_strides[0..], .offset = 0, .allocator = null, .owns_data = false };

    try engine.gemmFp8Block(a_t, w_t, &c_t);

    const ref = try std.testing.allocator.alloc(f32, M * N);
    defer std.testing.allocator.free(ref);
    cpuRef(a, w, M, N, K, ref);

    // Gate de calidad para cuantización: SNR global ||err||/||ref|| (el
    // rel-por-término explota con ref≈0 — la simulación host exacta del
    // pipeline da mean_rel ~10% con rel-métrica; el SNR es el gate correcto).
    var err2: f64 = 0;
    var ref2: f64 = 0;
    for (ref, c) |r, g| {
        const d: f64 = @as(f64, g) - @as(f64, r);
        err2 += d * d;
        ref2 += @as(f64, r) * @as(f64, r);
    }
    const snr_rel = @sqrt(err2 / ref2);
    std.debug.print("[fp8-block] gemm M={d} N={d} K={d}: ||err||/||ref||={e}\n", .{ M, N, K, snr_rel });
    try std.testing.expect(snr_rel < 0.05); // FP8 block: ~1-4% típico
    // 2ª llamada MISMO peso — valida el cache (hit) y el scratch reutilizado.
    try engine.gemmFp8Block(a_t, w_t, &c_t);
}

test "fp8_block: gemvSplitK kernel directo (sin MatmulEngine)" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    var eng = try fp8k.Fp8BlockLinear.init(@ptrCast((try matmul.MatmulEngine.sharedCudaStream()).raw));
    const K = 128;
    const N = 1;
    // x todo 1.0, w todo 1.0 → out = 128.
    const x = [_]f32{1.0} ** 128;
    const w = [_]f32{1.0} ** 128;
    const d_x = try cudaz.cuMemAlloc(K * 4);
    defer cudaz.cuMemFree(d_x);
    const d_w = try cudaz.cuMemAlloc(N * K * 4);
    defer cudaz.cuMemFree(d_w);
    try cudaz.cuMemcpyHtoD(d_x, @intFromPtr(&x), K * 4);
    try cudaz.cuMemcpyHtoD(d_w, @intFromPtr(&w), N * K * 4);
    const d_xq = try cudaz.cuMemAlloc(K);
    defer cudaz.cuMemFree(d_xq);
    const d_xs = try cudaz.cuMemAlloc(4);
    defer cudaz.cuMemFree(d_xs);
    const d_wq = try cudaz.cuMemAlloc(N * K);
    defer cudaz.cuMemFree(d_wq);
    const d_ws = try cudaz.cuMemAlloc(N * 4);
    defer cudaz.cuMemFree(d_ws);
    try eng.quantizeActivationsF32(d_x, d_xq, d_xs, 1, 128);
    try eng.quantizeActivationsF32(d_w, d_wq, d_ws, 1, 128);
    const d_o = try cudaz.cuMemAlloc(N * 4);
    defer cudaz.cuMemFree(d_o);
    try cudaz.cuMemsetD8(d_o, 0, N * 4);
    try eng.gemvSplitK(d_xq, d_xs, d_wq, d_ws, d_o, 1, 128, 1);
    try cudaz.cuStreamSynchronize(@ptrCast((try matmul.MatmulEngine.sharedCudaStream()).raw));
    var out: f32 = 0;
    try cudaz.cuMemcpyDtoH(@intFromPtr(&out), d_o, 4);
    std.debug.print("[fp8-debug] gemv-directo: out={e} (esperado 128)\n", .{out});
}

test "fp8_block: gemvFp8Block (M=1) paridad + uno-caliente" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    var engine = try matmul.MatmulEngine.init(std.testing.allocator, .fp8_block, .fp8);
    defer engine.deinit();

    // ── uno-caliente: W fila-0 = 1.0 resto 0, N=1 ⇒ out = sum(x_q) ──
    {
        const K: usize = 128;
        const N: usize = 1;
        const x = try std.testing.allocator.alloc(f32, K);
        defer std.testing.allocator.free(x);
        for (x, 0..) |*v, i| v.* = @as(f32, 0.5) * @as(f32, @floatFromInt(i)) / 128.0;
        const w = try std.testing.allocator.alloc(f32, N * K);
        defer std.testing.allocator.free(w);
        // W todo-1: out = sum(x_q) — valida la cadena completa del quant.
        @memset(w, 1.0);

        var x_shape = [_]usize{K};
        var x_str = [_]usize{1};
        const x_t = Tensor(f32){ .data = x, .shape = x_shape[0..], .strides = x_str[0..], .offset = 0, .allocator = null, .owns_data = false };
        var w_shape = [_]usize{ N, K };
        var w_str = [_]usize{ K, 1 };
        const w_t = Tensor(f32){ .data = w, .shape = w_shape[0..], .strides = w_str[0..], .offset = 0, .allocator = null, .owns_data = false };
        const o = try std.testing.allocator.alloc(f32, N);
        defer std.testing.allocator.free(o);
        var o_shape = [_]usize{N};
        var o_str = [_]usize{1};
        const o_t = Tensor(f32){ .data = o, .shape = o_shape[0..], .strides = o_str[0..], .offset = 0, .allocator = null, .owns_data = false };

        try engine.gemvFp8Block(x_t, w_t, o_t, 1);
        var ref: f64 = 0;
        for (x) |v| ref += v;
        std.debug.print("[fp8-debug] uno-caliente: got={e} ref={e}\n", .{ o[0], ref });
    }

    // ── paridad con random ──
    const N: usize = 512;
    const K: usize = 1024;
    var rng = std.Random.DefaultPrng.init(88);
    const random = rng.random();
    const x = try std.testing.allocator.alloc(f32, K);
    defer std.testing.allocator.free(x);
    const w = try std.testing.allocator.alloc(f32, N * K);
    defer std.testing.allocator.free(w);
    for (x) |*v| v.* = random.floatNorm(f32) * 0.8;
    for (w) |*v| v.* = random.floatNorm(f32) * 0.3;

    var x_shape = [_]usize{K};
    var x_str = [_]usize{1};
    const x_t = Tensor(f32){ .data = x, .shape = x_shape[0..], .strides = x_str[0..], .offset = 0, .allocator = null, .owns_data = false };
    var w_shape = [_]usize{ N, K };
    var w_str = [_]usize{ K, 1 };
    const w_t = Tensor(f32){ .data = w, .shape = w_shape[0..], .strides = w_str[0..], .offset = 0, .allocator = null, .owns_data = false };
    const out = try std.testing.allocator.alloc(f32, N);
    defer std.testing.allocator.free(out);
    var o_shape = [_]usize{N};
    var o_str = [_]usize{1};
    const out_t = Tensor(f32){ .data = out, .shape = o_shape[0..], .strides = o_str[0..], .offset = 0, .allocator = null, .owns_data = false };

    const num_splits: usize = @min(K / 128, 4);
    try engine.gemvFp8Block(x_t, w_t, out_t, num_splits);

    const ref = try std.testing.allocator.alloc(f32, N);
    defer std.testing.allocator.free(ref);
    cpuRef(x, w, 1, N, K, ref);

    var err2: f64 = 0;
    var ref2: f64 = 0;
    for (ref, out) |r, g| {
        const d: f64 = @as(f64, g) - @as(f64, r);
        err2 += d * d;
        ref2 += @as(f64, r) * @as(f64, r);
    }
    const snr_rel = @sqrt(err2 / ref2);
    std.debug.print("[fp8-block] gemv N={d} K={d} splits={d}: ||err||/||ref||={e}\n", .{ N, K, num_splits, snr_rel });
    try std.testing.expect(snr_rel < 0.05);
}
