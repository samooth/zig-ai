//! Lane E (E5) — MoeLayer.forwardGPU vs referencia CPU con expertos q6_k
//! sintéticos. Ejercita router top-k → ensure → gather → expert-GEMM →
//! reduce ponderado end-to-end en dims pequeñas.
//!
//! Disciplina GPU: .bench.lock no-bloqueante (60 s → SKIP). Sin CUDA: SKIP.
const std = @import("std");

fn stdoutPrint(io: std.Io, comptime fmt: []const u8, args: anytype) !void {
    var stdout_buf: [256]u8 = undefined;
    const stdout_file = std.Io.File.stdout();
    var stdout_writer = stdout_file.writer(io, &stdout_buf);
    try stdout_writer.interface.print(fmt, args);
}
const builtin = @import("builtin");
const gguf = @import("gguf");
const cudaz = @import("cudaz");
const moe_cuda = @import("moe_cuda");
const cache_mod = @import("offload_cache");
const gguf_moe = @import("gguf_moe");
const layer_kernels = @import("layer_kernels");
const moe_layer = @import("moe_layer");
const cpu_executor = @import("moe_cpu_executor");

const testing = std.testing;

// Geometría (dims múltiplos de 128 ⇒ feat%16==0; q4_1: bs=32, bb=20)
const N_EMBD: usize = 256;
const FF: usize = 512;
const E: usize = 4;
const TOPK: usize = 2;
const Q41_BS: usize = 32;
const Q41_BB: usize = 20;

fn bankBytes(in_dim: usize, out_dim: usize) usize {
    return in_dim * out_dim / Q41_BS * Q41_BB;
}

/// Codifica valores a un banco q4_1 bloque a bloque (d/m f16 + nibbles).
fn encodeQ41Bank(vals: []const f32, bank: []u8) void {
    var vb: usize = 0;
    var bb: usize = 0;
    const nblocks = vals.len / Q41_BS;
    for (0..nblocks) |_| {
        var mn: f32 = vals[vb];
        var mx: f32 = vals[vb];
        for (vals[vb .. vb + Q41_BS]) |v| {
            mn = @min(mn, v);
            mx = @max(mx, v);
        }
        if (mx == mn) mx = mn + 1e-4;
        const d: f32 = (mx - mn) / 15.0;
        const d16: f16 = @floatCast(d);
        const m16: f16 = @floatCast(mn);
        std.mem.writeInt(u16, bank[bb..][0..2], @bitCast(d16), .little);
        std.mem.writeInt(u16, bank[bb + 2 ..][0..2], @bitCast(m16), .little);
        @memset(bank[bb + 4 ..][0..16], 0);
        for (0..Q41_BS) |j| {
            const q: u8 = @intFromFloat(std.math.clamp((vals[vb + j] - mn) / d, 0, 15));
            if (j % 2 == 0) {
                bank[bb + 4 + j / 2] |= q & 0xF;
            } else {
                bank[bb + 4 + j / 2] |= q << 4;
            }
        }
        vb += Q41_BS;
        bb += Q41_BB;
    }
}

const BenchLock = struct {
    file: ?std.Io.File = null,

    fn acquire(io: std.Io) !BenchLock {
        if (comptime builtin.target.os.tag == .windows) return .{};
        const dir = std.Io.Dir.cwd();
        var waited: u32 = 0;
        while (true) {
            const f = try dir.createFile(io, ".bench.lock", .{ .truncate = false });
            if (std.c.flock(f.handle, 6) == 0) return .{ .file = f };
            f.close(io);
            waited += 5;
            if (waited >= 60) return error.BenchLockBusy;
            var ts: std.c.timespec = .{ .sec = 5, .nsec = 0 };
            _ = std.c.nanosleep(&ts, null);
        }
    }

    fn release(self: BenchLock, io: std.Io) void {
        if (comptime builtin.target.os.tag == .windows) return;
        if (self.file) |f| {
            _ = std.c.flock(f.handle, 8);
            f.close(io);
        }
    }
};

test "E5 capa MoE completa: paridad GPU↔CPU con expertos q4_1" {
    const gpa = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const lock = BenchLock.acquire(io) catch |e| {
        if (e == error.BenchLockBusy) {
            try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "SKIP: .bench.lock ocupada\n", .{});
            return error.SkipZigTest;
        }
        return e;
    };
    defer lock.release(io);

    // ── Datos sintéticos ──
    var rng = std.Random.Xoshiro256.init(0xE5E501);
    const rand = rng.random();

    const eb_gu = bankBytes(N_EMBD, FF);
    const eb_down = bankBytes(FF, N_EMBD);

    // Bancos con pesos reales cuantizados a q4_1 (evita escalas basura).
    const wtmp = try gpa.alloc(f32, N_EMBD * FF);
    defer gpa.free(wtmp);
    const gate_bank = try gpa.alloc(u8, E * eb_gu);
    defer gpa.free(gate_bank);
    const up_bank = try gpa.alloc(u8, E * eb_gu);
    defer gpa.free(up_bank);
    const down_bank = try gpa.alloc(u8, E * eb_down);
    defer gpa.free(down_bank);
    for (0..E) |e| {
        for (wtmp) |*v| v.* = (rand.float(f32) - 0.5) * 0.5;
        encodeQ41Bank(wtmp, gate_bank[e * eb_gu ..][0..eb_gu]);
        encodeQ41Bank(wtmp, up_bank[e * eb_gu ..][0..eb_gu]);
        for (wtmp) |*v| v.* = (rand.float(f32) - 0.5) * 0.5;
        encodeQ41Bank(wtmp, down_bank[e * eb_down ..][0..eb_down]);
    }

    const router_bytes = try gpa.alloc(u8, N_EMBD * E * @sizeOf(f32));
    defer gpa.free(router_bytes);
    {
        const f32s: [*]f32 = @ptrCast(@alignCast(router_bytes.ptr));
        for (0..N_EMBD * E) |i| f32s[i] = rand.float(f32) - 0.5;
    }

    const spec = gguf_moe.MoeLayerSpec{
        .layer_id = 0,
        .family = .generic,
        .router = .{ .bytes = router_bytes, .dtype = .f32, .n_embd = N_EMBD, .n_expert = E },
        .gate = .{ .bytes = gate_bank, .dtype = .q4_1, .in_dim = N_EMBD, .out_dim = FF, .n_expert = E },
        .up = .{ .bytes = up_bank, .dtype = .q4_1, .in_dim = N_EMBD, .out_dim = FF, .n_expert = E },
        .down = .{ .bytes = down_bank, .dtype = .q4_1, .in_dim = FF, .out_dim = N_EMBD, .n_expert = E },
        .n_expert = @intCast(E),
        .top_k = @intCast(TOPK),
    };

    // ── Setup GPU ──
    cudaz.ensureContext() catch return error.SkipZigTest; // CI sin toolkit: skip limpio
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);

    var lk = try layer_kernels.LayerKernels.init(stream);

    const cfg = cache_mod.Config{ .num_layers = 1, .num_experts = @intCast(E), .cache_size = @intCast(E), .max_fetch = 8 };
    var cache_gpu = try moe_cuda.MoeCacheGpu.init(cfg);
    defer cache_gpu.deinit();
    var gatherer = try moe_cuda.ExpertGatherer.init(3);
    defer gatherer.deinit();

    // Ruta FALLBACK (sin mmap_region): fuentes pinned por copia.
    var layer = try moe_layer.MoeLayer.init(gpa, spec, cfg, stream, &lk, &cache_gpu, &gatherer, null);
    defer layer.deinit(gpa);

    // ── Entrada + salida device ──
    const x_host = try gpa.alloc(f32, N_EMBD);
    defer gpa.free(x_host);
    for (x_host) |*v| v.* = (rand.float(f32) - 0.5) * 2.0;
    const x_dev = try cudaz.cuMemAlloc(N_EMBD * @sizeOf(f32));
    defer cudaz.cuMemFree(x_dev);
    try moe_cuda.htod(f32, x_dev, x_host);

    const out_dev = try cudaz.cuMemAlloc(N_EMBD * @sizeOf(f32));
    defer cudaz.cuMemFree(out_dev);

    try layer.forwardGPU(x_dev, out_dev);
    try cudaz.cuStreamSynchronize(stream);

    { // diagnóstico temporal
        var w2: [TOPK]f32 = undefined;
        var s2: [TOPK]i32 = undefined;
        try moe_cuda.dtoh(f32, &w2, layer.dev_weights);
        try moe_cuda.dtoh(i32, &s2, layer.dev_ids);
        try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "[dbg] weights={any} slots={any}\n", .{ w2, s2 });
        var acc_host: [N_EMBD]f32 = undefined;
        try moe_cuda.dtoh(f32, &acc_host, layer.dev_acc);
        var na: f32 = 0;
        for (acc_host) |v| na += v * v;
        try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "[dbg] ||acc||={d}\n", .{na});
    }

    const out_gpu = try gpa.alloc(f32, N_EMBD);
    defer gpa.free(out_gpu);
    try moe_cuda.dtoh(f32, out_gpu, out_dev);

    // ── Referencia CPU ──
    // Router: logits f32 exactos, softmax estable, topk por prob con máscara,
    // renormalización entre seleccionados (idéntico al kernel).
    var logits: [E]f32 = undefined;
    {
        const rf: [*]const f32 = @ptrCast(@alignCast(router_bytes.ptr));
        for (0..E) |e| {
            var acc: f32 = 0;
            for (0..N_EMBD) |i| acc += rf[e * N_EMBD + i] * x_host[i];
            logits[e] = acc;
        }
    }
    var mx: f32 = -std.math.floatMax(f32);
    for (logits) |l| mx = @max(mx, l);
    var denom: f32 = 0;
    for (logits) |l| denom += @exp(l - mx);
    var probs: [E]f32 = undefined;
    for (logits, 0..) |l, e| probs[e] = @exp(l - mx) / denom;

    var chosen: [TOPK]usize = undefined;
    var wsel: [TOPK]f32 = undefined;
    var masked = probs;
    for (0..TOPK) |k| {
        var best_p: f32 = -1;
        var best_e: usize = 0;
        for (masked, 0..) |p, e| {
            if (p > best_p) {
                best_p = p;
                best_e = e;
            }
        }
        chosen[k] = best_e;
        wsel[k] = best_p;
        masked[best_e] = -1;
    }
    {
        var s: f32 = 0;
        for (wsel) |w| s += w;
        for (&wsel) |*w| w.* /= s;
    }

    // Expertos CPU con dequant q6_k real.
    const expected = try gpa.alloc(f32, N_EMBD);
    defer gpa.free(expected);
    @memset(expected, 0);

    const gate_dq = try gpa.alloc(f32, N_EMBD * FF);
    defer gpa.free(gate_dq);
    const up_dq = try gpa.alloc(f32, N_EMBD * FF);
    defer gpa.free(up_dq);
    const down_dq = try gpa.alloc(f32, FF * N_EMBD);
    defer gpa.free(down_dq);
    for (0..TOPK) |k| {
        dequantBankQ41(gate_bank[chosen[k] * eb_gu ..][0..eb_gu], gate_dq);
        dequantBankQ41(up_bank[chosen[k] * eb_gu ..][0..eb_gu], up_dq);
        dequantBankQ41(down_bank[chosen[k] * eb_down ..][0..eb_down], down_dq);
        // GGUF layout [in,out] fila-major por out: gate_dq[o*N+i]
        var h: [FF]f32 = undefined;
        for (0..FF) |o| {
            var g_acc: f32 = 0;
            var u_acc: f32 = 0;
            for (0..N_EMBD) |i| {
                g_acc += gate_dq[o * N_EMBD + i] * x_host[i];
                u_acc += up_dq[o * N_EMBD + i] * x_host[i];
            }
            const sil = g_acc / (1.0 + @exp(-g_acc)); // SiLU
            h[o] = sil * u_acc;
        }
        for (0..N_EMBD) |o| {
            var acc: f32 = 0;
            for (0..FF) |i| acc += down_dq[o * FF + i] * h[i];
            expected[o] += wsel[k] * acc;
        }
    }

    var max_diff: f32 = 0;
    var scale: f32 = 1e-9;
    for (expected) |v| scale = @max(scale, @abs(v));
    var norm_out: f32 = 0;
    var norm_exp: f32 = 0;
    for (out_gpu) |v| norm_out += v * v;
    for (expected) |v| norm_exp += v * v;
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "[E5] ||out||={d:.4} ||exp||={d:.4}\n", .{ norm_out, norm_exp });
    try testing.expect(norm_out > 1e-8); // la salida NO puede ser cero
    try testing.expect(norm_exp > 1e-8);
    for (out_gpu, expected) |gv, ev_| {
        max_diff = @max(max_diff, @abs(gv - ev_));
    }
    const rel = max_diff / scale;
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "[E5] max_diff={d:.6} rel={d:.6} (tolerancia 0.05)\n", .{ max_diff, rel });
    try testing.expect(rel < 0.05);
}

test "E5 4.10 streaming: paridad forwardGPUStreaming == batch (mismo routing)" {
    // Misma capa MoE, DOS ejecuciones sobre la MISMA entrada:
    //   A) camino batch clásico (default).
    //   B) MOE_EXPERT_STREAM=1 → ping-pong 2-slot + GEMMs por experto.
    // Ambas computan los mismos expertos/pesos ⇒ salidas DEBEN coincidir.
    // El env se setea DESPUÉS de que forwardGPU ya corrió en A (gated por
    // proceso — el modo B se activa antes del 2º forward).
    const gpa = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const lock = BenchLock.acquire(io) catch |e| {
        if (e == error.BenchLockBusy) {
            try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "SKIP: .bench.lock ocupada\n", .{});
            return error.SkipZigTest;
        }
        return e;
    };
    defer lock.release(io);

    var rng = std.Random.Xoshiro256.init(0xE51041);
    const rand = rng.random();

    const eb_gu = bankBytes(N_EMBD, FF);
    const eb_down = bankBytes(FF, N_EMBD);

    const wtmp = try gpa.alloc(f32, N_EMBD * FF);
    defer gpa.free(wtmp);
    const gate_bank = try gpa.alloc(u8, E * eb_gu);
    defer gpa.free(gate_bank);
    const up_bank = try gpa.alloc(u8, E * eb_gu);
    defer gpa.free(up_bank);
    const down_bank = try gpa.alloc(u8, E * eb_down);
    defer gpa.free(down_bank);
    for (0..E) |e| {
        for (wtmp) |*v| v.* = (rand.float(f32) - 0.5) * 0.5;
        encodeQ41Bank(wtmp, gate_bank[e * eb_gu ..][0..eb_gu]);
        encodeQ41Bank(wtmp, up_bank[e * eb_gu ..][0..eb_gu]);
        for (wtmp) |*v| v.* = (rand.float(f32) - 0.5) * 0.5;
        encodeQ41Bank(wtmp, down_bank[e * eb_down ..][0..eb_down]);
    }

    const router_bytes = try gpa.alloc(u8, N_EMBD * E * @sizeOf(f32));
    defer gpa.free(router_bytes);
    {
        const f32s: [*]f32 = @ptrCast(@alignCast(router_bytes.ptr));
        for (0..N_EMBD * E) |i| f32s[i] = rand.float(f32) - 0.5;
    }

    const spec = gguf_moe.MoeLayerSpec{
        .layer_id = 0,
        .family = .generic,
        .router = .{ .bytes = router_bytes, .dtype = .f32, .n_embd = N_EMBD, .n_expert = E },
        .gate = .{ .bytes = gate_bank, .dtype = .q4_1, .in_dim = N_EMBD, .out_dim = FF, .n_expert = E },
        .up = .{ .bytes = up_bank, .dtype = .q4_1, .in_dim = N_EMBD, .out_dim = FF, .n_expert = E },
        .down = .{ .bytes = down_bank, .dtype = .q4_1, .in_dim = FF, .out_dim = N_EMBD, .n_expert = E },
        .n_expert = @intCast(E),
        .top_k = @intCast(TOPK),
    };

    cudaz.ensureContext() catch return error.SkipZigTest; // CI sin toolkit: skip limpio
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);
    var lk = try layer_kernels.LayerKernels.init(stream);

    const cfg = cache_mod.Config{ .num_layers = 1, .num_experts = @intCast(E), .cache_size = @intCast(E), .max_fetch = 8 };
    var cache_gpu = try moe_cuda.MoeCacheGpu.init(cfg);
    defer cache_gpu.deinit();
    var gatherer = try moe_cuda.ExpertGatherer.init(3);
    defer gatherer.deinit();

    var layer = try moe_layer.MoeLayer.init(gpa, spec, cfg, stream, &lk, &cache_gpu, &gatherer, null);
    defer layer.deinit(gpa);

    const x_host = try gpa.alloc(f32, N_EMBD);
    defer gpa.free(x_host);
    for (x_host) |*v| v.* = (rand.float(f32) - 0.5) * 2.0;
    const x_dev = try cudaz.cuMemAlloc(N_EMBD * @sizeOf(f32));
    defer cudaz.cuMemFree(x_dev);
    try moe_cuda.htod(f32, x_dev, x_host);
    const out_dev = try cudaz.cuMemAlloc(N_EMBD * @sizeOf(f32));
    defer cudaz.cuMemFree(out_dev);

    // A) batch (default, env OFF — el gate se lee POR FORWARD en 4.10).
    try layer.forwardGPU(x_dev, out_dev);
    try cudaz.cuStreamSynchronize(stream);
    const out_batch = try gpa.alloc(f32, N_EMBD);
    defer gpa.free(out_batch);
    try moe_cuda.dtoh(f32, out_batch, out_dev);

    // B) streaming: activar el gate ANTES del 2º forward (cache LRU del
    // pool ya tiene los expertos ⇒ la reserva ve hits y los slots del
    // ping-pong son los ya cargados — el stageHtoD re-escribe los mismos
    // bytes, coherente).
    const expert_streamer = @import("expert_streamer");
    expert_streamer.forceForTest(true);
    defer expert_streamer.forceForTest(false);
    try layer.forwardGPU(x_dev, out_dev);
    try cudaz.cuStreamSynchronize(stream);
    const out_stream = try gpa.alloc(f32, N_EMBD);
    defer gpa.free(out_stream);
    try moe_cuda.dtoh(f32, out_stream, out_dev);

    var norm_b: f32 = 0;
    for (out_batch) |v| norm_b += v * v;
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "[E5-4.10] ||batch||={d:.4} ||stream||={d:.4}\n", .{ norm_b, blk: {
        var n: f32 = 0;
        for (out_stream) |v| n += v * v;
        break :blk n;
    } });
    try testing.expect(norm_b > 1e-8); // salida no trivial

    // Paridad EXACTA: mismos GEMMs sobre los mismos bytes de expertos ⇒
    // bitwise-identical (el streaming no cambia el orden de reducción:
    // axpyMul por experto en el MISMO orden del router).
    for (out_batch, out_stream, 0..) |bv, sv, i| {
        if (bv != sv) {
            try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "[E5-4.10] mismatch @{d}: batch={d:.6} stream={d:.6}\n", .{ i, bv, sv });
            return error.StreamingParityMismatch;
        }
    }
}

test "E5 4.10 streaming con nw==1: camino de un solo experto sin ping-pong real" {
    // top_k=1 ⇒ nw==1 ⇒ take==1 ⇒ guard de reserva dispara fallback batch
    // (take!=2) — el resultado debe seguir siendo correcto vía el camino
    // clásico. Valida el GUARD añadido (sin el fix: slots_ok=false con
    // take==1 y @intCast(reserved[0]) podía ser −1 ⇒ panic).
    const gpa = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const lock = BenchLock.acquire(io) catch |e| {
        if (e == error.BenchLockBusy) {
            try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "SKIP: .bench.lock ocupada\n", .{});
            return error.SkipZigTest;
        }
        return e;
    };
    defer lock.release(io);

    var rng = std.Random.Xoshiro256.init(0xE51042);
    const rand = rng.random();

    const eb_gu = bankBytes(N_EMBD, FF);
    const eb_down = bankBytes(FF, N_EMBD);
    const wtmp = try gpa.alloc(f32, N_EMBD * FF);
    defer gpa.free(wtmp);
    const gate_bank = try gpa.alloc(u8, E * eb_gu);
    defer gpa.free(gate_bank);
    const up_bank = try gpa.alloc(u8, E * eb_gu);
    defer gpa.free(up_bank);
    const down_bank = try gpa.alloc(u8, E * eb_down);
    defer gpa.free(down_bank);
    for (0..E) |e| {
        for (wtmp) |*v| v.* = (rand.float(f32) - 0.5) * 0.5;
        encodeQ41Bank(wtmp, gate_bank[e * eb_gu ..][0..eb_gu]);
        encodeQ41Bank(wtmp, up_bank[e * eb_gu ..][0..eb_gu]);
        for (wtmp) |*v| v.* = (rand.float(f32) - 0.5) * 0.5;
        encodeQ41Bank(wtmp, down_bank[e * eb_down ..][0..eb_down]);
    }
    const router_bytes = try gpa.alloc(u8, N_EMBD * E * @sizeOf(f32));
    defer gpa.free(router_bytes);
    {
        const f32s: [*]f32 = @ptrCast(@alignCast(router_bytes.ptr));
        for (0..N_EMBD * E) |i| f32s[i] = rand.float(f32) - 0.5;
    }

    const spec = gguf_moe.MoeLayerSpec{
        .layer_id = 0,
        .family = .generic,
        .router = .{ .bytes = router_bytes, .dtype = .f32, .n_embd = N_EMBD, .n_expert = E },
        .gate = .{ .bytes = gate_bank, .dtype = .q4_1, .in_dim = N_EMBD, .out_dim = FF, .n_expert = E },
        .up = .{ .bytes = up_bank, .dtype = .q4_1, .in_dim = N_EMBD, .out_dim = FF, .n_expert = E },
        .down = .{ .bytes = down_bank, .dtype = .q4_1, .in_dim = FF, .out_dim = N_EMBD, .n_expert = E },
        .n_expert = @intCast(E),
        .top_k = 1,
    };

    cudaz.ensureContext() catch return error.SkipZigTest; // CI sin toolkit: skip limpio
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);
    var lk = try layer_kernels.LayerKernels.init(stream);
    const cfg = cache_mod.Config{ .num_layers = 1, .num_experts = @intCast(E), .cache_size = @intCast(E), .max_fetch = 8 };
    var cache_gpu = try moe_cuda.MoeCacheGpu.init(cfg);
    defer cache_gpu.deinit();
    var gatherer = try moe_cuda.ExpertGatherer.init(3);
    defer gatherer.deinit();

    var layer = try moe_layer.MoeLayer.init(gpa, spec, cfg, stream, &lk, &cache_gpu, &gatherer, null);
    defer layer.deinit(gpa);

    const x_host = try gpa.alloc(f32, N_EMBD);
    defer gpa.free(x_host);
    for (x_host) |*v| v.* = (rand.float(f32) - 0.5) * 2.0;
    const x_dev = try cudaz.cuMemAlloc(N_EMBD * @sizeOf(f32));
    defer cudaz.cuMemFree(x_dev);
    try moe_cuda.htod(f32, x_dev, x_host);
    const out_dev = try cudaz.cuMemAlloc(N_EMBD * @sizeOf(f32));
    defer cudaz.cuMemFree(out_dev);

    // Streaming activo con top_k=1 → nw=1 → guard → fallback batch correcto.
    const expert_streamer = @import("expert_streamer");
    expert_streamer.forceForTest(true);
    defer expert_streamer.forceForTest(false);
    try layer.forwardGPU(x_dev, out_dev);
    try cudaz.cuStreamSynchronize(stream);
    const out = try gpa.alloc(f32, N_EMBD);
    defer gpa.free(out);
    try moe_cuda.dtoh(f32, out, out_dev);

    var norm: f32 = 0;
    for (out) |v| norm += v * v;
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "[E5-4.10-nw1] ||out||={d:.4} (finito, no-cero vía fallback)\n", .{norm});
    try testing.expect(norm > 1e-8 and norm == norm);
}

/// Dequantiza un banco q4_1 lineal a `out` vía gguf.dequantBlock.
fn dequantBankQ41(bytes: []const u8, out: []f32) void {
    const num_blocks = out.len / Q41_BS;
    var off_b: usize = 0;
    var off_e: usize = 0;
    for (0..num_blocks) |_| {
        gguf.dequantBlock(.q4_1, bytes[off_b .. off_b + Q41_BB], out[off_e .. off_e + Q41_BS], Q41_BS);
        off_b += Q41_BB;
        off_e += Q41_BS;
    }
}

test "E5 variante ZERO-COPY: fuentes = VAs del mmap registrado (Contrato 5)" {
    // Requiere la fixture MoE sintética en disco (generable con
    // `zig build moe-make-fixture -- /tmp/opencode/moe_fixture.gguf`).
    // Ejercita la ruta del swap E8: HostBank.fromFileMmapWhole registra el
    // mmap COMPLETO y las fuentes del gather son VAs directas del archivo.
    const env_path = std.c.getenv("MOE_FIXTURE_PATH") orelse
        "/tmp/opencode/moe_fixture.gguf";
    const path = std.mem.span(env_path);

    const gpa = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var lock = BenchLock.acquire(io) catch return error.SkipZigTest;
    defer lock.release(io);

    const fixture_exists = blk: {
        var f = std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only }) catch break :blk false;
        const st = f.stat(io) catch {
            f.close(io);
            break :blk false;
        };
        f.close(io);
        break :blk st.size > 0;
    };
    if (!fixture_exists) {
        try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "SKIP: fixture no encontrada ({s}) — genera con zig build moe-make-fixture\n", .{path});
        return error.SkipZigTest;
    }

    var g = try gguf.GgufFile.fromFileMmap(io, gpa, path);
    defer g.deinit();

    if (!gguf_moe.isMoeModel(&g)) {
        try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "SKIP: {s} no es MoE\n", .{path});
        return error.SkipZigTest;
    }
    const info = try gguf_moe.moeInfo(&g);
    const spec = try gguf_moe.layerSpec(&g, 0);

    cudaz.ensureContext() catch return error.SkipZigTest; // CI sin toolkit: skip limpio
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);
    var lk = try layer_kernels.LayerKernels.init(stream);

    const cfg = cache_mod.Config{
        .num_layers = @intCast(gguf_moe.blockCountMeta(&g, g.arch().?) orelse 1),
        .num_experts = info.n_expert,
        .cache_size = info.n_expert,
        .max_fetch = info.top_k * 2,
    };
    var cache_gpu = try moe_cuda.MoeCacheGpu.init(cfg);
    defer cache_gpu.deinit();
    var gatherer = try moe_cuda.ExpertGatherer.init(3);
    defer gatherer.deinit();

    // Ruta ZERO-COPY: pasamos el mmap completo ⇒ HostBank registra y el
    // gather lee VAs del archivo directamente.
    var layer = try moe_layer.MoeLayer.init(gpa, spec, cfg, stream, &lk, &cache_gpu, &gatherer, g.data);
    defer layer.deinit(gpa);
    // KNOWN-ISSUE driver 580.173: cuMemHostRegister devuelve INVALID_VALUE
    // sobre mmaps file-backed MAP_PRIVATE ⇒ la ruta zero-copy degrada sola a
    // copia pinned (diseño E8). Aceptamos ambos modos: lo que se valida es el
    // pipeline completo leyendo de las fuentes que sean.
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "[E5-zero-copy] bank_registered={any} (false = limitación driver/file-mmap, ver LANE-E/HANDOFFS)\n", .{layer.bank_registered});

    const n_embd: usize = @intCast(spec.router.n_embd);
    const x_host = try gpa.alloc(f32, n_embd);
    defer gpa.free(x_host);
    var rng = std.Random.Xoshiro256.init(0xE5C077);
    for (x_host) |*v| v.* = (rng.random().float(f32) - 0.5) * 2.0;

    const x_dev = try cudaz.cuMemAlloc(n_embd * @sizeOf(f32));
    defer cudaz.cuMemFree(x_dev);
    try moe_cuda.htod(f32, x_dev, x_host);
    const out_dev = try cudaz.cuMemAlloc(n_embd * @sizeOf(f32));
    defer cudaz.cuMemFree(out_dev);

    // Dos pasos: el primero fetchea todo (cold), el segundo es puro hit —
    // ambas rutas leen fuentes zero-copy sin copias intermedias.
    try layer.forwardGPU(x_dev, out_dev);
    try cudaz.cuStreamSynchronize(stream);
    try layer.forwardGPU(x_dev, out_dev);
    try cudaz.cuStreamSynchronize(stream);

    const out = try gpa.alloc(f32, n_embd);
    defer gpa.free(out);
    try moe_cuda.dtoh(f32, out, out_dev);
    var norm: f32 = 0;
    for (out) |v| norm += v * v;
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "[E5-zero-copy] ||out||={d:.4} (no-NaN, no-cero)\n", .{norm});
    try testing.expect(norm > 1e-8 and norm == norm); // finito y no vacío
}

test "E5 híbrido: overflow a CPU executor == offload puro (paridad)" {
    // Misma fixture, DOS ejecuciones:
    //   A) offload puro (sin executors): todos los expertos en GPU.
    //   B) híbrido con cap=1 ⇒ overflow garantizado → CPU via patrón F.
    // Las sumas finales deben coincidir (mismos pesos, mismo routing).
    const env_path = std.c.getenv("MOE_FIXTURE_PATH") orelse
        "/tmp/opencode/moe_fixture.gguf";
    const path = std.mem.span(env_path);

    const gpa = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var lock = BenchLock.acquire(io) catch return error.SkipZigTest;
    defer lock.release(io);

    const fixture_exists = blk: {
        var f = std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only }) catch break :blk false;
        const st = f.stat(io) catch {
            f.close(io);
            break :blk false;
        };
        f.close(io);
        break :blk st.size > 0;
    };
    if (!fixture_exists) {
        try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "SKIP: fixture ausente ({s})\n", .{path});
        return error.SkipZigTest;
    }

    var g = try gguf.GgufFile.fromFileMmap(io, gpa, path);
    defer g.deinit();
    if (!gguf_moe.isMoeModel(&g)) return error.SkipZigTest;

    const info = try gguf_moe.moeInfo(&g);
    const spec = try gguf_moe.layerSpec(&g, 0);

    cudaz.ensureContext() catch return error.SkipZigTest; // CI sin toolkit: skip limpio
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);
    var lk = try layer_kernels.LayerKernels.init(stream);

    // Executors F (host_staging; geometrías gate/up y down).
    const exec_gu = try cpu_executor.Executor.init(gpa, 2);
    defer exec_gu.deinit();
    const exec_dn = try cpu_executor.Executor.init(gpa, 2);
    defer exec_dn.deinit();

    const n_embd: usize = @intCast(spec.router.n_embd);

    // ── A) offload puro: cache completa, cap generoso ──
    const cfg_off = cache_mod.Config{
        .num_layers = 1,
        .num_experts = info.n_expert,
        .cache_size = info.n_expert,
        .max_fetch = info.top_k,
    };
    var cache_a = try moe_cuda.MoeCacheGpu.init(cfg_off);
    defer cache_a.deinit();
    var gatherer_a = try moe_cuda.ExpertGatherer.init(3);
    defer gatherer_a.deinit();
    var layer_a = try moe_layer.MoeLayer.init(gpa, spec, cfg_off, stream, &lk, &cache_a, &gatherer_a, g.data);
    defer layer_a.deinit(gpa);

    // ── B) híbrido: cap=1 < top_k ⇒ overflow seguro a CPU ──
    const cfg_hyb = cfg_off;
    var cache_b = try moe_cuda.MoeCacheGpu.init(cfg_hyb);
    defer cache_b.deinit();
    var gatherer_b = try moe_cuda.ExpertGatherer.init(3);
    defer gatherer_b.deinit();
    var layer_b = try moe_layer.MoeLayer.init(gpa, spec, cfg_hyb, stream, &lk, &cache_b, &gatherer_b, g.data);
    defer layer_b.deinit(gpa);
    layer_b.exec_gu = exec_gu;
    layer_b.exec_down = exec_dn;
    // fmt de los Executors = dtype real de los bancos (fixture q4_1).
    // (Config se fija por el caller vía initFull; para el test usamos la
    // variante simple init + fijación manual de geometría/fmt.)
    exec_gu.cfg = .{
        .fmt = switch (spec.gate.dtype) {
            .q4_1 => .q4_1,
            .q4_0 => .q4_0,
            .q8_0 => .q8_0,
            .q6_k => .q6_k,
            .q4_k => .q4_k,
            .q5_k => .q5_k,
            else => return error.SkipZigTest,
        },
        .k_dim = n_embd,
        .out_dim = @intCast(spec.gate.out_dim),
        .n_experts = info.n_expert,
    };
    exec_dn.cfg = .{
        .fmt = switch (spec.down.dtype) {
            .q4_1 => .q4_1,
            .q4_0 => .q4_0,
            .q8_0 => .q8_0,
            .q6_k => .q6_k,
            .q4_k => .q4_k,
            .q5_k => .q5_k,
            else => return error.SkipZigTest,
        },
        .k_dim = @intCast(spec.gate.out_dim),
        .out_dim = n_embd,
        .n_experts = info.n_expert,
    };
    try testing.expect(layer_b.executorAttached());

    // Entrada común.
    var rng = std.Random.Xoshiro256.init(0xE5A17);
    const x_host = try gpa.alloc(f32, n_embd);
    defer gpa.free(x_host);
    for (x_host) |*v| v.* = (rng.random().float(f32) - 0.5) * 2.0;
    const x_dev = try cudaz.cuMemAlloc(n_embd * @sizeOf(f32));
    defer cudaz.cuMemFree(x_dev);
    try moe_cuda.htod(f32, x_dev, x_host);
    const out_dev = try cudaz.cuMemAlloc(n_embd * @sizeOf(f32));
    defer cudaz.cuMemFree(out_dev);

    // A) puro.
    try layer_a.forwardGPU(x_dev, out_dev);
    try cudaz.cuStreamSynchronize(stream);
    const out_a = try gpa.alloc(f32, n_embd);
    defer gpa.free(out_a);
    try moe_cuda.dtoh(f32, out_a, out_dev);

    // B) híbrido (max_fetch=1 ⇒ al menos un −1 con top_k=2).
    layer_b.cfg.max_fetch = 1;
    try layer_b.forwardGPU(x_dev, out_dev);
    try cudaz.cuStreamSynchronize(stream);
    const out_b = try gpa.alloc(f32, n_embd);
    defer gpa.free(out_b);
    try moe_cuda.dtoh(f32, out_b, out_dev);

    var norm_b: f32 = 0;
    for (out_b) |v| norm_b += v * v;
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "[E5-hybrid] ||A(offload)||={d:.4} ||B(hybrid)||={d:.4} overflow={d}\n", .{
        blk3: {
            var na: f32 = 0;
            for (out_a) |v| na += v * v;
            break :blk3 na;
        },
        norm_b,
        layer_b.last_overflow_n,
    });
    try testing.expect(layer_b.last_overflow_n > 0); // el escenario debe dar overflow

    // Paridad: mismos expertos seleccionados ⇒ mismas sumas (tolerancia gemv).
    for (out_a, out_b, 0..) |a_v, b_v, i| {
        const diff = @abs(a_v - b_v);
        if (diff > 0.05 * (1.0 + @abs(a_v))) {
            try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "[E5-hybrid] mismatch @{d}: offload={d:.4} hybrid={d:.4}\n", .{ i, a_v, b_v });
            return error.HybridParityMismatch;
        }
    }
}
