//! 5.1 (lane-c): paridad del DflashEncoder device-resident vs ref CPU.
//! TODO 5.1 "C6.x kernels: encoder fc+norm device-resident, taps sin D2H".
//!
//! Camino GPU: dflashTapsGatherKernel (concat taps device→device, sin D2H)
//! + dflashFcGemmM1Kernel (GEMV q8_0 split-K por tiles — K=32768 del fc del
//! 9B excede el smem del M1 genérico) + rmsNormKernel.
//! Ref CPU: dequant q8_0 + dot f64 + rmsNorm f32.
//!
//! Env (skip limpio si faltan):
//!   GGUF_MODEL_PATH     — target (n_embd del encoder; el sidecar hereda)
//!   DFLASH_SIDECAR_PATH — sidecar dflash GGUF (fc q8_0 + enc.output_norm
//!                          + dflash.target_layers)
//! ⚠ Test con kernels CUDA — REQUIERE flock .bench.lock (protocolo GPU).

const std = @import("std");
const testing = std.testing;
const gguf_model = @import("gguf_model");
const speculative = @import("speculative");
const cudaz = @import("cudaz");
const layer_kernels = @import("layer_kernels");
const kv_cache = @import("kv_cache");

test "dflash encoder paridad GPU vs CPU (fc+norm device-resident, taps sin D2H)" {
    const io = std.Io.Threaded.global_single_threaded.io();
    const allocator = std.heap.page_allocator;
    const target_path = std.c.getenv("GGUF_MODEL_PATH") orelse return error.SkipZigTest;
    const sc_path = std.c.getenv("DFLASH_SIDECAR_PATH") orelse return error.SkipZigTest;

    cudaz.ensureContext() catch return error.SkipZigTest;
    _ = cudaz.cuDeviceGet(0) catch return error.SkipZigTest;
    const stream: cudaz.CUstream = try cudaz.cuStreamCreate(0);
    var lk = try layer_kernels.LayerKernels.init(stream);

    var target = try gguf_model.GgufModel.load(io, allocator, std.mem.span(target_path));
    defer target.deinit();
    var sc = try gguf_model.GgufModel.loadSidecarDraft(&target, io, allocator, std.mem.span(sc_path));
    defer sc.deinit();
    if (sc.target_layers.len == 0) return error.SkipZigTest; // sidecar sin target_layers

    const n_embd: usize = target.config.embedding_length;
    const n_taps = sc.target_layers.len;
    const fc_k = n_taps * n_embd;

    var enc = try speculative.dflash_encoder.DflashEncoder.init(allocator, &lk, &sc, n_embd, 1e-6);
    defer enc.deinit();

    // Taps sintéticos deterministas (2 patrones distintos para ejercitar
    // el gather multi-ptr).
    const taps_host = try allocator.alloc(f32, n_taps * n_embd);
    defer allocator.free(taps_host);
    for (taps_host, 0..) |*v, i| {
        const tap = i / n_embd;
        const c = i % n_embd;
        v.* = @sin(@as(f32, @floatFromInt(tap * 7 + c))) * 0.5;
    }
    var tap_devs: [16]usize = undefined;
    for (0..n_taps) |t| {
        const d = try cudaz.cuMemAlloc(n_embd * 4);
        tap_devs[t] = d;
        try cudaz.cuMemcpyHtoD(d, @intFromPtr(taps_host[t * n_embd ..].ptr), n_embd * 4);
    }
    defer for (0..n_taps) |t| {
        cudaz.cuMemFree(tap_devs[t]);
    };

    // ── Camino GPU (gather + GEMV split-K + rmsNorm, 0 D2H de taps) ──
    const h_dev = try enc.forward(tap_devs[0..n_taps]);
    try cudaz.cuStreamSynchronize(stream);
    const h_gpu = try allocator.alloc(f32, n_embd);
    defer allocator.free(h_gpu);
    try cudaz.cuMemcpyDtoH(@intFromPtr(h_gpu.ptr), h_dev, n_embd * 4);

    // ── Ref CPU ──
    const fc = sc.model.file.getTensor("fc.weight").?;
    const fc_data = sc.model.file.tensorData(fc);
    const norm = sc.model.file.getTensor("enc.output_norm.weight").?;
    const norm_data = sc.model.file.tensorData(norm);
    const norm_f32 = std.mem.bytesAsSlice(f32, @as([*]const u8, @ptrCast(norm_data.ptr))[0..norm_data.len]);

    const h_ref = try allocator.alloc(f32, n_embd);
    defer allocator.free(h_ref);
    const fc_f32 = try allocator.alloc(f32, fc_k * n_embd);
    defer allocator.free(fc_f32);
    kv_cache.kv_quant.dequant(.q8_0, fc_data, fc_f32);
    for (0..n_embd) |o| {
        var acc: f64 = 0;
        for (0..fc_k) |k| acc += @as(f64, fc_f32[o * fc_k + k]) * @as(f64, taps_host[k]);
        h_ref[o] = @floatCast(acc);
    }
    var ss: f64 = 0;
    for (h_ref) |v| ss += @as(f64, v) * @as(f64, v);
    const inv: f32 = @floatCast(1.0 / @sqrt(ss / @as(f64, @floatFromInt(n_embd)) + 1e-6));
    for (h_ref, 0..) |*v, i| v.* = v.* * inv * norm_f32[i];

    // ── Diff (tolerancia: q8_0 + accum f32 GPU vs f64 CPU) ──
    var worst: f32 = 0;
    var rel_max: f32 = 0;
    for (h_gpu, h_ref) |g, r| {
        const d = @abs(g - r);
        worst = @max(worst, d);
        const denom = @max(@abs(r), 0.05);
        rel_max = @max(rel_max, d / denom);
    }
    debugPrint(worst, rel_max, n_taps, fc_k);
    try testing.expect(worst < 5e-2);
    try testing.expect(rel_max < 2e-2);
    for (h_gpu) |v| try testing.expect(!std.math.isNan(v) and !std.math.isInf(v));
}

fn debugPrint(worst: f32, rel: f32, n_taps: usize, fc_k: usize) void {
    std.debug.print("dflash enc paridad: worst_at={e} rel(≥0.05 floor)={e} n_taps={d} fc_k={d}\n", .{ worst, rel, n_taps, fc_k });
}
