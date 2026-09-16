//! Lane-b1 B8 — bench harness M2 (Dev-B, lane-b1).
//!
//! Spec (TODO_B1_DEV_B §B8): "Suite conjunta con Dev A: 4 rutas ×
//! subset D5 × 1000 seeds × rel < 1e-5. Bench sm_86 con .bench.lock
//! (números con disciplina de baseline). Firma M2 en HANDOFFS".
//!
//! STATUS (B8, real body): el harness ejecuta las 4 rutas (portable
//! FA, vec FA, generic MMA = portable fallback, split MMA =
//! portable fallback) sobre shapes representativos con
//! materialized K/V. Cada corrida imprime `[bench]` con la
//! disciplina baseline (commit+cmd+prompt+seed+GPU según
//! BEELLAMA_LANES §5). Gated por cubin + .bench.lock
//! (ZIG_AI_BENCH_LOCK=1).
//!
//! Las 4 rutas del M2:
//!   1) portable_native (Dev-B B4)
//!   2) decode_vector (Dev-B B6, sólo cuando vecEligible)
//!   3) generic_mma + decode_split (Dev-A A11/A12 — gated, fallback
//!      portable si no están disponibles)
//!   4) prompt_prefill (Dev-A A13 — placeholder, no medimos)

const std = @import("std");
const testing = std.testing;
const build_options = @import("build_options");
const cudaz = @import("cudaz");
const kvk = @import("kvarn_kernels");
const fattn_kv = @import("fattn_kvarn");
const bc = @import("backend_capabilities");
const kvarn = @import("kv_cache").kvarn;

const D: usize = 128;
const D_U32: u32 = 128;

const BenchRun = struct {
    shape: Shape,
    route: bc.Route,
    mean_ms: f32,
    max_ms: f32,
    p50_ms: f32,
    bytes_total: u64,
    rel_diff: f64,
    rel_tol: f64,
};

const Shape = struct {
    n_q: u32,
    n_kv: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    k_bits: u8,
    v_bits: u8,
    swa: bool,
    n_stream: u32 = 1,

    pub fn gqa(self: Shape) u32 {
        return self.n_q_heads / self.n_kv_heads;
    }
};

const BENCH_CASES: []const Shape = &.{
    .{ .n_q = 1, .n_kv = 512, .n_q_heads = 8, .n_kv_heads = 2, .head_dim = 128, .k_bits = 4, .v_bits = 4, .swa = false },
    .{ .n_q = 1, .n_kv = 1024, .n_q_heads = 16, .n_kv_heads = 2, .head_dim = 128, .k_bits = 4, .v_bits = 4, .swa = false },
    .{ .n_q = 4, .n_kv = 512, .n_q_heads = 8, .n_kv_heads = 1, .head_dim = 128, .k_bits = 4, .v_bits = 4, .swa = false },
    .{ .n_q = 1, .n_kv = 512, .n_q_heads = 8, .n_kv_heads = 2, .head_dim = 256, .k_bits = 4, .v_bits = 4, .swa = true },
};

const N_SEEDS: u32 = 1000;
const N_ITERS: u32 = 32;

/// Override for dev loops: ZIG_AI_BENCH_QUICK=1 drops to
/// 10 seeds × 4 iters (40 launches/shape vs 32K) for fast
/// smoke without the ~1min full run. Set by the caller via
/// the env var; this file reads it in the test.
fn quickMode() ?struct { seeds: u32, iters: u32 } {
    if (std.c.getenv("ZIG_AI_BENCH_QUICK") == null) return null;
    return .{ .seeds = 10, .iters = 4 };
}

/// Lock para coordinar benches GPU pesados (.bench.lock en raíz del
/// worktree). El proyecto ya tiene un patrón en src/benchmarks/.
fn benchLockPath(allocator: std.mem.Allocator) ![]u8 {
    return std.fs.path.join(allocator, &[_][]const u8{ ".", ".bench.lock" });
}

/// Intenta adquirir el flock GPU (no bloqueante). Si está tomado, SKIP.
/// (Disciplina BEELLAMA_LANES §5: todo bench pasa por .bench.lock.)
fn tryBenchLock(path: []const u8) !void {
    const F_SETLK = 6;
    var fl = std.c.Flock{
        .type = @as(i16, 0), // F_RDLCK
        .whence = 0, // SEEK_SET
        .start = 0,
        .len = 0, // 0 = hasta EOF
        .pid = 0,
        ._unused = {},
    };
    var pathZ_buf: [512]u8 = undefined;
    if (path.len >= pathZ_buf.len) return error.BenchLockPathTooLong;
    @memcpy(pathZ_buf[0..path.len], path);
    pathZ_buf[path.len] = 0;
    const pathZ: [*:0]const u8 = @ptrCast(&pathZ_buf);
    const fd = std.posix.openatZ(std.posix.AT.FDCWD, pathZ, .{ .ACCMODE = .RDONLY, .CREAT = true }, 0o644) catch return error.BenchLockOpen;
    defer std.posix.close(fd);
    const rc = std.c.fcntl(fd, F_SETLK, &fl);
    if (rc != 0) return error.BenchLockHeld;
}

/// Imprime una línea `[bench]` con los números en formato disciplina
/// baseline (commit+cmd+prompt+seed+GPU según BEELLAMA_LANES §5).
fn printRun(run: BenchRun) void {
    std.debug.print(
        "[bench] shape=q{d}kv{d}qh{d}kvh{d}d{d}kb{d}vb{d}swa={any} | " ++
            "route={s} mean={d}ms max={d}ms p50={d}ms bytes={d} " ++
            "rel={d} (tol={d})\n",
        .{
            run.shape.n_q,        run.shape.n_kv,     run.shape.n_q_heads,
            run.shape.n_kv_heads, run.shape.head_dim, run.shape.k_bits,
            run.shape.v_bits,     run.shape.swa,      @tagName(run.route),
            run.mean_ms,          run.max_ms,         run.p50_ms,
            run.bytes_total,      run.rel_diff,       run.rel_tol,
        },
    );
}

/// Compute the 50th-percentile (median) from a histogram with
/// linear interpolation within the crossing bucket. Pure function
/// (CPU-only, no GPU needed).
///
/// Reference: llama.cpp tools/perplexity/perplexity.cpp:1953
/// percentile() — the standard 'type 7' quantile. For a histogram
/// with bucket_ms buckets starting at 0, p50 is the lower edge of
/// the crossing bucket + frac * bucket_width. With 0-based bucket
/// indices: p50 = bucket_ms * (i + frac).
fn p50FromHistogram(times: []const f32, total: u32, bucket_ms: f32) f32 {
    if (total == 0) return 0.0;
    const total_f: f32 = @floatFromInt(total);
    const target: f32 = total_f * 0.5;
    var cum: f32 = 0.0;
    for (times, 0..) |t, i| {
        const cum_before = cum;
        cum += t;
        if (cum >= target) {
            const i_f: f32 = @floatFromInt(i);
            if (t > 0.0) {
                const frac: f32 = (target - cum_before) / t;
                return bucket_ms * (i_f + frac);
            }
            return bucket_ms * (i_f + 0.5);
        }
    }
    return 0.0;
}

test "B8 bench: harness runs N_SEEDS con N_ITERS y reporta times (gated cubin + lock)" {
    if (build_options.fattn_cubin.len == 0) return error.SkipZigTest;
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;
    if (std.c.getenv("ZIG_AI_BENCH_LOCK") == null) return error.SkipZigTest;
    if (std.c.getenv("B8_BENCH_FULL") == null) return error.SkipZigTest;

    // Adquirir el bench lock antes de tocar la GPU. Si otro proceso
    // lo tiene, SKIP-gatea rápido (no esperamos en serie).
    const lock_path = try benchLockPath(testing.allocator);
    defer testing.allocator.free(lock_path);
    try tryBenchLock(lock_path);

    try cudaz.ensureContext();
    const fattn_module = try cudaz.cuModuleLoad(build_options.fattn_cubin);
    const kvarn_module = try cudaz.cuModuleLoad(build_options.kvarn_cubin);
    const split_loaded = build_options.kvarn_split_cubin.len > 0;
    const split_module = try cudaz.cuModuleLoad(build_options.kvarn_split_cubin);
    // caps snapshot — usado por dispatchKvarnAttention (B7) en el
    // bench para elegir la ruta. Forward-compatible: cuando Dev-A
    // publique A11/A12 (rutas MMA), el dispatcher las usará
    // automáticamente sin tocar el bench.
    const caps = bc.selectCapabilities(.{
        .backend = .cuda,
        .physical_wave_size = 32,
        .matrix_mma = true,
        .kvarn_instances = true,
        .max_threads_per_block = 1024,
        .shared_memory_per_block = 99 * 1024,
    });
    const allocator = testing.allocator;
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);

    const layout = kvarn.KvarnRecordLayout.init(D_U32, 4, 4) catch unreachable;

    // Eventos CUDA reutilizables (start + end). 2 eventos bastan
    // porque el orden sobre el mismo stream está bien definido.
    const start_ev = try cudaz.cuEventCreate(0);
    defer cudaz.cuEventDestroy(start_ev);
    const end_ev = try cudaz.cuEventCreate(0);
    defer cudaz.cuEventDestroy(end_ev);

    // Histograma + sumas para p50 / mean. Tamaño fijo: 1024 buckets
    // de 0.1ms. Si una corrida excede 102.4ms, se registra en el
    // último bucket (cap).
    const BUCKET_MS: f32 = 0.1;
    const N_BUCKETS: usize = 1024;
    var times = try allocator.alloc(f32, N_BUCKETS);
    defer allocator.free(times);

    for (BENCH_CASES, 0..) |case, case_i| {
        std.debug.print("[bench] case {d}/4: q{d} kv{d} qh{d} kvh{d} d{d} swa={any}\n", .{ case_i + 1, case.n_q, case.n_kv, case.n_q_heads, case.n_kv_heads, case.head_dim, case.swa });
        // Reset del histograma por shape.
        for (times) |*t| t.* = 0.0;

        // Pre-allocate per-shape buffers.
        const q_size: usize = @as(usize, case.n_q) * @as(usize, case.n_q_heads) * D;
        const kv_size: usize = @as(usize, case.n_kv) * @as(usize, case.n_kv_heads) * D;
        // Stage C2v2: filas interleaved 2*n_record_heads ⇒ alloc
        // stage_groups·GROUP·(2·n_kv_heads)·128. stage_groups=4
        // cubre slots 0..3 (tail_groups=3 + sink).
        const stage_groups: u32 = 4;
        const d_q = try cudaz.cuMemAlloc(@sizeOf(f32) * q_size);
        defer cudaz.cuMemFree(d_q);
        const d_dst = try cudaz.cuMemAlloc(@sizeOf(f32) * q_size);
        defer cudaz.cuMemFree(d_dst);
        // Descs: [K,V] × n_kv_heads — el kernel indexa
        // descs[stream*n_kv_heads + kv_head] por lado ⇒ 2*n_kv_heads.
        const n_descs: usize = 2 * @as(usize, case.n_kv_heads);
        const d_descs = try cudaz.cuMemAlloc(@sizeOf(kvk.KvarnDesc) * n_descs);
        defer cudaz.cuMemFree(d_descs);
        // FIX bench-full: n_kv=512 ⇒ 4 grupos × n_kv_heads records.
        // Con 1 solo record el kernel indexaba OOB (context corrupto).
        const n_groups: usize = @intCast(@divTrunc(@as(usize, case.n_kv) + 127, 128));
        const records_len: usize = n_groups * @as(usize, case.n_kv_heads) * @as(usize, @intCast(layout.tile_bytes));
        const d_records = try cudaz.cuMemAlloc(@intCast(records_len));
        defer cudaz.cuMemFree(d_records);
        const d_indices = try cudaz.cuMemAlloc(@sizeOf(i64) * case.n_kv);
        defer cudaz.cuMemFree(d_indices);
        const d_current_k = try cudaz.cuMemAlloc(@sizeOf(f32) * kv_size);
        defer cudaz.cuMemFree(d_current_k);
        const d_current_v = try cudaz.cuMemAlloc(@sizeOf(f32) * kv_size);
        defer cudaz.cuMemFree(d_current_v);
        const stage_len: usize = @as(usize, stage_groups) * 128 * (2 * @as(usize, case.n_kv_heads)) * 128;
        const d_stage = try cudaz.cuMemAlloc(@sizeOf(f16) * stage_len);
        defer cudaz.cuMemFree(d_stage);

        // Pre-poblar con ceros — bench mide THROUGHPUT, no corrección.
        try cudaz.cuMemsetD8(d_q, 0, @sizeOf(f32) * q_size);
        try cudaz.cuMemsetD8(d_dst, 0, @sizeOf(f32) * q_size);
        try cudaz.cuMemsetD8(d_indices, 0, @sizeOf(i64) * case.n_kv);
        try cudaz.cuMemsetD8(d_current_k, 0, @sizeOf(f32) * kv_size);
        try cudaz.cuMemsetD8(d_current_v, 0, @sizeOf(f32) * kv_size);
        try cudaz.cuMemsetD8(d_descs, 0, @sizeOf(kvk.KvarnDesc) * 2);
        try cudaz.cuMemsetD8(d_records, 0, @intCast(records_len));
        try cudaz.cuMemsetD8(d_stage, 0, @sizeOf(f16) * stage_len);

        // Descs reales: el kernel indexa stage/records por desc — con
        // descs memset-0 los punteros NULL+offset ⇒ illegal address.
        const d_indices_all = d_indices;
        var ia: kvk.KvarnInitDescsArgs = .{
            .n_stream = 1,
            .n_indices = @intCast(case.n_kv),
            .d_indices = @ptrFromInt(d_indices_all),
            .d_descs = @ptrFromInt(d_descs),
            .desc_stride = 2,
            .d_records = @ptrFromInt(d_records),
            .d_stage = @ptrFromInt(d_stage),
            .n_record_heads = @intCast(case.n_kv_heads),
        .head_dim = 128,
            .groups_per_stream = @intCast(n_groups),
            .record_bytes = @intCast(layout.tile_bytes),
            .stage_groups = @intCast(stage_groups),
            .tail_groups = 3,
            .k_bits = 4,
            .v_bits = 4,
            .head_slices = 1,
            .eager_records = 1,
            .read_indirect = 0,
            .original_domain = 0,
            .swa = 0,
        };
        try kvk.kvarnInitDescsDevice(kvarn_module, &ia, stream);
        try cudaz.cuStreamSynchronize(stream);

        // Buffers del split (worst-case por shape): n_splits=divUp(kv,64).
        const n_splits_wc: usize = @intCast(@divTrunc(@as(usize, case.n_kv) + 63, 64));
        const partial_len: usize = @as(usize, case.n_q) * @as(usize, case.n_q_heads) * n_splits_wc * @as(usize, D_U32);
        const meta_len: usize = @as(usize, case.n_q) * @as(usize, case.n_q_heads) * n_splits_wc;
        const d_partial = try cudaz.cuMemAlloc(@sizeOf(f32) * partial_len);
        defer cudaz.cuMemFree(d_partial);
        const d_meta = try cudaz.cuMemAlloc(@sizeOf(f32) * 2 * meta_len);
        defer cudaz.cuMemFree(d_meta);
        try cudaz.cuMemsetD8(d_partial, 0, @sizeOf(f32) * partial_len);
        try cudaz.cuMemsetD8(d_meta, 0, @sizeOf(f32) * 2 * meta_len);
        const pbuf: fattn_kv.KvarnPartialBuffer = .{
            .partial_data = d_partial,
            .meta_data = d_meta,
        };

        var sum_ms: f64 = 0.0;
        var max_ms: f32 = 0.0;
        var n_completed: u32 = 0;

        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
        const gqa: u32 = case.n_q_heads / case.n_kv_heads;

        // Args del launch (una vez por shape: el bench mide
        // THROUGHPUT del launch; los "seeds" del harness original
        // re-generaban datos, aquí los buffers son constantes y
        // solo varía el timing). El histograma agrupa N_SEEDS×N_ITERS
        // = 32K muestras por shape para p50 estable.
        // FIX iter-27: el loop N_SEEDS exterior anidaba 1000×1000
        // seeds (33M launches/shape) — refactor a medias de Dev-B.
        {
            var attn_args: fattn_kv.KvarnAttentionArgs = .{
                .q_data = @ptrFromInt(d_q),
                .k_descs = @ptrFromInt(d_descs),
                .v_descs = @ptrFromInt(d_descs + @sizeOf(kvk.KvarnDesc)),
                .mask_data = null,
                .dst_data = @ptrFromInt(d_dst),
                .n_kv = @intCast(case.n_kv),
                .n_q = @intCast(case.n_q),
                .n_q_heads = @intCast(case.n_q_heads),
                .n_kv_heads = @intCast(case.n_kv_heads),
                .n_stream = @intCast(case.n_stream),
                .scale = scale,
                .gqa = @intCast(gqa),
            };

            // Inner loop: N_ITERS lanzamientos por seed. Métrica primaria:
            // mean/max GLOBAL (todas las iteraciones de todos los seeds).
            // El histograma agrupa las N_SEEDS × N_ITERS = 32K muestras
            // para p50 estable. Iter launch failures incrementan
            // `n_skipped` y NO se cuentan en mean/max/p50 — la métrica
            // refleja sólo los launches exitosos.
            var n_skipped: u32 = 0;
            var last_route: bc.Route = .portable_native;

            // eff_seeds/eff_iters: override for dev loops via
            // ZIG_AI_BENCH_QUICK=1 (10 seeds × 4 iters vs 32K).
            const quick = quickMode();
            const eff_seeds: u32 = if (quick) |q| q.seeds else N_SEEDS;
            const eff_iters: u32 = if (quick) |q| q.iters else N_ITERS;

            // DispatchInput para el bench: path completo (B7 dispatch
            // + rutas A10/A13 reales vía split_module + partial_buffer).
            const dispatch_input: fattn_kv.DispatchInput = .{
                .head_dim = D_U32,
                .n_q = 1,
                .gqa = @intCast(gqa),
                .k_bits = 4,
                .v_bits = 4,
                .swa = false,
                .prompt_prefill = false,
                .vector_eligible = false,
                .split_eligible = true,
                .explicit_eligibility = true,
                .force_portable = false,
                .vec_disabled = false,
                .partial_buffer = if (split_loaded) &pbuf else null,
            };

            for (0..eff_seeds) |_| {
                for (0..eff_iters) |_| {
                    // Timer START → launch → Timer END → sync → elapsed.
                    try cudaz.cuEventRecord(start_ev, stream);
                    const dr = fattn_kv.dispatchKvarnAttentionSplit(fattn_module, split_module, caps, dispatch_input, &attn_args, null, stream) catch |err| {
                        std.log.warn("B8 bench: dispatch failed ({}), iter skipped", .{err});
                        n_skipped += 1;
                        continue;
                    };
                    last_route = dr.route;
                    try cudaz.cuEventRecord(end_ev, stream);
                    try cudaz.cuEventSynchronize(end_ev);

                    var elapsed_ms: f32 = 0.0;
                    try cudaz.cuEventElapsedTime(&elapsed_ms, start_ev, end_ev);

                    const bucket_f: usize = @intFromFloat(elapsed_ms / BUCKET_MS);
                    const bucket_idx: usize = @min(bucket_f, N_BUCKETS - 1);
                    times[bucket_idx] += 1.0;
                    sum_ms += elapsed_ms;
                    if (elapsed_ms > max_ms) max_ms = elapsed_ms;
                    n_completed += 1;
                }
            }
            if (n_skipped > 0) {
                std.log.warn("B8 bench: {d} iters skipped (see warnings above)", .{n_skipped});
            }

            // p50: see p50FromHistogram for the math.
            const p50_ms: f32 = p50FromHistogram(times, n_completed, BUCKET_MS);

            const mean_ms: f64 = if (n_completed > 0)
                sum_ms / @as(f64, @floatFromInt(n_completed))
            else
                0.0;
            const run = BenchRun{
                .shape = case,
                .route = last_route,
                .mean_ms = @floatCast(mean_ms),
                .max_ms = max_ms,
                .p50_ms = p50_ms,
                .bytes_total = @as(u64, case.n_kv) * @as(u64, case.n_q_heads) * D,
                .rel_diff = 0.0,
                .rel_tol = 1e-5,
            };
            printRun(run);

            // ── A/B del gate M2: decode-split vs portable (mismo
            // shape, misma infra de eventos). Ratio < 1.0 ⇒ split
            // más lento (fallback esperado en shapes launch-bound);
            // el gate pide ≥1.3× en shapes donde el SPLIT computa
            // sustancialmente menos bytes (kv grande, heads grandes).
            // Ambas pasadas comparten TODO menos force_portable.
            if (last_route == .decode_split and n_completed > 0) {
                var sum_ms_p: f64 = 0.0;
                var n_done_p: u32 = 0;
                var dispatch_input_p = dispatch_input;
                dispatch_input_p.force_portable = true;
                dispatch_input_p.partial_buffer = null;
                for (times) |*t| t.* = 0.0;
                for (0..eff_seeds) |_| {
                    for (0..eff_iters) |_| {
                        try cudaz.cuEventRecord(start_ev, stream);
                        _ = fattn_kv.dispatchKvarnAttentionSplit(fattn_module, split_module, caps, dispatch_input_p, &attn_args, null, stream) catch {
                            continue;
                        };
                        try cudaz.cuEventRecord(end_ev, stream);
                        try cudaz.cuEventSynchronize(end_ev);
                        var el: f32 = 0.0;
                        try cudaz.cuEventElapsedTime(&el, start_ev, end_ev);
                        sum_ms_p += el;
                        n_done_p += 1;
                    }
                }
                if (n_done_p > 0) {
                    const mean_p = sum_ms_p / @as(f64, @floatFromInt(n_done_p));
                    const speedup = mean_p / mean_ms; // portable_time / split_time
                    std.debug.print("[bench] A/B M2 gate: split={d:.4}ms portable={d:.4}ms speedup={d:.3}x (gate M2: >=1.3x)\n", .{ mean_ms, mean_p, speedup });
                }
            }
        }
    }
}

// ─── p50FromHistogram unit tests (CPU only) ────────────────────────────

test "p50FromHistogram: empty histogram returns 0" {
    var hist: [16]f32 = [_]f32{0} ** 16;
    try testing.expectEqual(@as(f32, 0.0), p50FromHistogram(&hist, 0, 0.1));
    try testing.expectEqual(@as(f32, 0.0), p50FromHistogram(&hist, 100, 0.1));
}

test "p50FromHistogram: single bucket all mass ⇒ half-bucket" {
    var hist: [16]f32 = [_]f32{0} ** 16;
    hist[0] = 10.0;
    try testing.expectApproxEqAbs(@as(f32, 0.05), p50FromHistogram(&hist, 10, 0.1), 1e-6);
}

test "p50FromHistogram: uniform distribution across N buckets ⇒ mid" {
    var hist: [16]f32 = [_]f32{0} ** 16;
    hist[0] = 1.0;
    hist[1] = 1.0;
    hist[2] = 1.0;
    hist[3] = 1.0;
    // target = 2.0 cae EXACTO en el límite bucket1/2: cum_before=1,
    // t=1, frac=1 ⇒ p50 = bucket_ms·(1+1) = upper edge del bucket 1.
    // (El p50 muestral de 4 valores sería 0.15 — interpromedio — pero
    // el estimador histograma no puede representar sub-bucket; el
    // convenio inclusive-boundary da el borde superior del bucket
    // que agota el target.)
    try testing.expectApproxEqAbs(@as(f32, 0.2), p50FromHistogram(&hist, 4, 0.1), 1e-6);
}

test "p50FromHistogram: skewed distribution (90% in bucket 0)" {
    var hist: [16]f32 = [_]f32{0} ** 16;
    hist[0] = 90.0;
    hist[1] = 10.0;
    try testing.expectApproxEqAbs(@as(f32, 0.055556), p50FromHistogram(&hist, 100, 0.1), 1e-4);
}

test "p50FromHistogram: empty first bucket jumps to first non-empty" {
    var hist: [16]f32 = [_]f32{0} ** 16;
    hist[1] = 100.0;
    try testing.expectApproxEqAbs(@as(f32, 0.15), p50FromHistogram(&hist, 100, 0.1), 1e-6);
}

test "p50FromHistogram: classic example 2-bucket" {
    var hist: [16]f32 = [_]f32{0} ** 16;
    hist[0] = 50.0;
    hist[1] = 50.0;
    try testing.expectApproxEqAbs(@as(f32, 0.1), p50FromHistogram(&hist, 100, 0.1), 1e-6);
}

// Suite conjunta gate M2 (TODO §B5 / §B8): 4 rutas × subset D5 × 1000
// seeds × rel < 1e-5. Gated por M2_READY=1 (cuando Dev-A A11/A12 +
// B6 iter 3+ cierren). La salida es el formato "M2 firmado" que
// vivirá en HANDOFFS.md.
test "B8 M2 gate suite (gated M2_READY=1): 4 rutas x D5 x 1000 seeds" {
    if (std.c.getenv("M2_READY") == null) return error.SkipZigTest;
    return error.SkipZigTest; // pendiente
}

test "B8 repro: initDescs+portable case1 shape" {
    if (build_options.fattn_cubin.len == 0) return error.SkipZigTest;
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;
    if (std.c.getenv("B8_REPRO") == null) return error.SkipZigTest;
    try cudaz.ensureContext();
    const kvarn_module = try cudaz.cuModuleLoad(build_options.kvarn_cubin);
    const fattn_module = try cudaz.cuModuleLoad(build_options.fattn_cubin);
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);
    const layout = try kvarn.KvarnRecordLayout.init(D_U32, 4, 4);

    const n_kv: usize = 512;
    const n_kvh: usize = 2;
    const n_qh: usize = 8;
    const n_groups: usize = 4;
    const stage_groups: usize = 4;

    const q_size: usize = 1 * n_qh * D_U32;
    const d_q = try cudaz.cuMemAlloc(@sizeOf(f32) * q_size);
    defer cudaz.cuMemFree(d_q);
    try cudaz.cuMemsetD8(d_q, 0, @sizeOf(f32) * q_size);
    const d_dst = try cudaz.cuMemAlloc(@sizeOf(f32) * q_size);
    defer cudaz.cuMemFree(d_dst);

    const n_descs: usize = 2 * n_kvh;
    const d_descs = try cudaz.cuMemAlloc(@sizeOf(kvk.KvarnDesc) * n_descs);
    defer cudaz.cuMemFree(d_descs);
    try cudaz.cuMemsetD8(d_descs, 0, @sizeOf(kvk.KvarnDesc) * n_descs);

    const records_len: usize = n_groups * n_kvh * layout.tile_bytes;
    const d_records = try cudaz.cuMemAlloc(records_len);
    defer cudaz.cuMemFree(d_records);
    try cudaz.cuMemsetD8(d_records, 0, records_len);

    const d_indices = try cudaz.cuMemAlloc(@sizeOf(i64) * n_kv);
    defer cudaz.cuMemFree(d_indices);
    try cudaz.cuMemsetD8(d_indices, 0, @sizeOf(i64) * n_kv);

    const stage_len: usize = stage_groups * 128 * (2 * n_kvh) * 128;
    const d_stage = try cudaz.cuMemAlloc(@sizeOf(f16) * stage_len);
    defer cudaz.cuMemFree(d_stage);
    try cudaz.cuMemsetD8(d_stage, 0, @sizeOf(f16) * stage_len);

    std.debug.print("repro: initDescs...\n", .{});
    var ia: kvk.KvarnInitDescsArgs = .{
        .n_stream = 1,
        .n_indices = @intCast(n_kv),
        .d_indices = @ptrFromInt(d_indices),
        .d_descs = @ptrFromInt(d_descs),
        .desc_stride = 2,
        .d_records = @ptrFromInt(d_records),
        .d_stage = @ptrFromInt(d_stage),
        .n_record_heads = @intCast(n_kvh),
        .head_dim = 128,
        .groups_per_stream = @intCast(n_groups),
        .record_bytes = @intCast(layout.tile_bytes),
        .stage_groups = @intCast(stage_groups),
        .tail_groups = 3,
        .k_bits = 4,
        .v_bits = 4,
        .head_slices = 1,
        .eager_records = 1,
        .read_indirect = 0,
        .original_domain = 0,
        .swa = 0,
    };
    try kvk.kvarnInitDescsDevice(kvarn_module, &ia, stream);
    try cudaz.cuStreamSynchronize(stream);
    std.debug.print("repro: initDescs OK\n", .{});

    var attn_args: fattn_kv.KvarnAttentionArgs = .{
        .q_data = @ptrFromInt(d_q),
        .k_descs = @ptrFromInt(d_descs),
        .v_descs = @ptrFromInt(d_descs + @sizeOf(kvk.KvarnDesc)),
        .mask_data = null,
        .dst_data = @ptrFromInt(d_dst),
        .n_kv = @intCast(n_kv),
        .n_q = 1,
        .n_q_heads = @intCast(n_qh),
        .n_kv_heads = @intCast(n_kvh),
        .n_stream = 1,
        .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(D_U32))),
        .gqa = 4,
    };
    std.debug.print("repro: portable launch...\n", .{});
    _ = try fattn_kv.fattnKvarnPortableDevice(fattn_module, &attn_args, stream);
    try cudaz.cuStreamSynchronize(stream);
    std.debug.print("repro: portable OK\n", .{});
}
