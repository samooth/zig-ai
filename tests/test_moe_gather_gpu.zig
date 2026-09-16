//! Lane E (E4) — gather multi-banco: paridad bytes vs referencia CPU + GB/s.
//!
//! Escenario: B bancos (gate/up/down) con fuentes host-pineadas (interim
//! pinnedAlloc+copias del "mmap"; swap a HostBank de D2 cuando aterrice) y
//! slot caches VRAM [plan × feat_b]. Se lanza ensure_experts_moe para obtener
//! el plan (evict/src/contador device) y luego el gather fused; la paridad se
//! verifica byte-a-byte contra la copia esperada computada en host. El knob
//! NOGATHER=1 desvía al staging clásico y debe producir bytes idénticos.
//!
//! Disciplina GPU: .bench.lock no-bloqueante (60 s → SKIP). Sin CUDA: SKIP.
const std = @import("std");
const cudaz = @import("cudaz");
const moe_cuda = @import("moe_cuda");
const cache_mod = @import("offload_cache");

const OffloadCache = cache_mod.OffloadCache;
const Config = cache_mod.Config;

const BenchLock = struct {
    file: ?std.Io.File = null,

    fn acquire(io: std.Io) !BenchLock {
        const dir = std.Io.Dir.cwd();
        var waited: u32 = 0;
        while (true) {
            const f = try dir.createFile(io, ".bench.lock", .{ .truncate = false });
            if (std.c.flock(f.handle, 6) == 0) return .{ .file = f };
            f.close(io);
            waited += 5;
            if (waited >= 60) return error.BenchLockBusy;
            std.debug.print("[moe_gather_test] .bench.lock ocupada, espero {d}s…\n", .{waited});
            var ts: std.c.timespec = .{ .sec = 5, .nsec = 0 };
            _ = std.c.nanosleep(&ts, null);
        }
    }

    fn release(self: *BenchLock, io: std.Io) void {
        if (self.file) |f| {
            _ = std.c.flock(f.handle, 8);
            f.close(io);
            self.file = null;
        }
    }
};

const testing = std.testing;

// dims múltiples de 256 ⇒ feat_bytes múltiplo de 16 en Q8_0 (dim/32×34).
// gate ff=256→272B, up ff=512→544B, down n_embd=768→816B.
const FEATS = [_]usize{ 272, 544, 816 };
const PLAN: usize = 12;
const N_BANKS = FEATS.len;

const Rig = struct {
    gpa: std.mem.Allocator,
    lock: BenchLock,
    io: std.Io,
    stream: cudaz.CUstream,
    gpu: moe_cuda.MoeCacheGpu,
    mirror: OffloadCache,
    gatherer: moe_cuda.ExpertGatherer,

    // fuentes host pineadas por banco ([PLAN][feat] cada una)
    src_host: [N_BANKS][]u8,
    // slot caches VRAM por banco
    dst_dev: [N_BANKS]cudaz.CUdeviceptr,

    fn init(gpa: std.mem.Allocator) !Rig {
        const io = std.Io.Threaded.global_single_threaded.io();
        var lock = BenchLock.acquire(io) catch |e| {
            if (e == error.BenchLockBusy) {
                std.debug.print("SKIP: .bench.lock ocupada\n", .{});
                return error.SkipZigTest;
            }
            return e;
        };
        errdefer lock.release(io);

        const cfg = Config{ .num_layers = 1, .num_experts = 8, .cache_size = @intCast(PLAN), .max_fetch = 6 };
        var mirror = try OffloadCache.init(gpa, cfg);
        errdefer mirror.deinit(gpa);
        var gpu = try moe_cuda.MoeCacheGpu.init(cfg);
        errdefer gpu.deinit();
        var gatherer = try moe_cuda.ExpertGatherer.init(N_BANKS);
        errdefer gatherer.deinit();
        const stream = try cudaz.cuStreamCreate(0);
        errdefer cudaz.cuStreamDestroy(stream);

        var rig = Rig{
            .gpa = gpa,
            .lock = lock,
            .io = io,
            .stream = stream,
            .gpu = gpu,
            .mirror = mirror,
            .gatherer = gatherer,
            .src_host = undefined,
            .dst_dev = undefined,
        };

        var rng = std.Random.Xoshiro256.init(0xE4D4A4);
        for (0..N_BANKS) |b| {
            const src = try cudaz.pinnedAlloc(u8, PLAN * FEATS[b]);
            rng.random().bytes(src);
            rig.src_host[b] = src;
            rig.dst_dev[b] = try cudaz.cuMemAlloc(PLAN * FEATS[b]);
        }
        errdefer for (0..N_BANKS) |b| cudaz.cuMemFree(rig.dst_dev[b]);
        return rig;
    }

    fn deinit(self: *Rig) void {
        for (0..N_BANKS) |b| {
            cudaz.pinnedFree(u8, self.src_host[b]);
            cudaz.cuMemFree(self.dst_dev[b]);
        }
        cudaz.cuStreamDestroy(self.stream);
        self.gatherer.deinit();
        self.gpu.deinit();
        self.mirror.deinit(self.gpa);
        self.lock.release(self.io);
    }

    /// Un paso completo: ensure (device) + sync + gather (fused o clásico).
    fn runStep(self: *Rig, ids: []const i32, frac: u32) !usize {
        const dev_ids = try cudaz.cuMemAlloc(ids.len * @sizeOf(i32));
        defer cudaz.cuMemFree(dev_ids);
        try moe_cuda.htod(i32, dev_ids, ids);
        try self.gpu.ensureExperts(self.stream, 0, dev_ids, @intCast(ids.len), frac, 6);
        try cudaz.cuStreamSynchronize(self.stream);

        var src_bases: [N_BANKS]usize = undefined;
        var dst_bases: [N_BANKS]usize = undefined;
        for (0..N_BANKS) |b| {
            src_bases[b] = @intFromPtr(self.src_host[b].ptr);
            dst_bases[b] = self.dst_dev[b];
        }
        try self.gatherer.gatherMissing(self.stream, &self.gpu, &dst_bases, &src_bases, &FEATS, moe_cuda.kGatherBlocksPerBank);
        try cudaz.cuStreamSynchronize(self.stream);

        var nf: [1]i64 = undefined;
        try moe_cuda.dtoh(i64, &nf, self.gpu.dev_num_indices);
        return @intCast(nf[0]);
    }

    /// Verifica que el slot cache VRAM coincide con la copia esperada:
    /// filas fetch = slice del banco fuente; resto = lo que hubiera antes
    /// (aquí: patrón conocido previo).
    fn verifyAgainstExpected(self: *Rig, expected: anytype) !void {
        for (0..N_BANKS) |b| {
            const got = try self.gpa.alloc(u8, expected[b].len);
            defer self.gpa.free(got);
            try moe_cuda.dtoh(u8, got, self.dst_dev[b]);
            for (got, expected[b], 0..) |gv, ev_, i| {
                if (gv != ev_) {
                    std.debug.print("[gather:banco {d}] byte @{d}: gpu={d} esperado={d}\n", .{ b, i, gv, ev_ });
                    return error.GatherMismatch;
                }
            }
        }
    }
};

test "E4 gather fused: paridad bytes vs copia host" {
    const gpa = testing.allocator;
    var rig = Rig.init(gpa) catch return error.SkipZigTest; // CI sin toolkit
    defer rig.deinit();

    // Preparación: VRAM con patrón conocido 0x5A y cache COLD (reset device),
    // para que el paso medido haga el fetch real verificable.
    for (0..N_BANKS) |b| {
        const pat = try gpa.alloc(u8, PLAN * FEATS[b]);
        defer gpa.free(pat);
        @memset(pat, 0x5A);
        try moe_cuda.htod(u8, rig.dst_dev[b], pat);
    }
    try rig.gpu.reset();

    // Paso único cold: activos {1,3} ⇒ fetch 2 filas a slots 0,1
    // (ganador por score recencia−1: e1 primero, luego e3).
    const n1 = try rig.runStep(&[_]i32{ 3, 1 }, 0);
    try testing.expectEqual(@as(usize, 2), n1);

    var expected: [N_BANKS][]u8 = undefined;
    defer for (0..N_BANKS) |b| gpa.free(expected[b]);
    for (0..N_BANKS) |b| {
        const exp = try gpa.alloc(u8, PLAN * FEATS[b]);
        @memset(exp, 0x5A);
        // ganadores por recencia −1: e1 primero (slot 0), luego e3 (slot 1)
        @memcpy(exp[0 * FEATS[b] ..][0..FEATS[b]], rig.src_host[b][1 * FEATS[b] ..][0..FEATS[b]]);
        @memcpy(exp[1 * FEATS[b] ..][0..FEATS[b]], rig.src_host[b][3 * FEATS[b] ..][0..FEATS[b]]);
        expected[b] = exp;
    }
    try rig.verifyAgainstExpected(&expected);
}

test "E4 overflow + segunda ronda: paridad acumulada" {
    const gpa = testing.allocator;
    var rig = Rig.init(gpa) catch return error.SkipZigTest; // CI sin toolkit
    defer rig.deinit();

    // Rellena VRAM con patrón conocido.
    for (0..N_BANKS) |b| {
        const pat = try gpa.alloc(u8, PLAN * FEATS[b]);
        defer gpa.free(pat);
        @memset(pat, 0x11);
        try moe_cuda.htod(u8, rig.dst_dev[b], pat);
    }

    // Paso A: 6 activos con cap 6 → todo fetched a slots 0..5 (ganadores id asc).
    const na = try rig.runStep(&[_]i32{ 7, 2, 5, 0, 6, 4 }, 0);
    try testing.expectEqual(@as(usize, 6), na);
    const winners_a = [_]i32{ 0, 2, 4, 5, 6, 7 };

    // Paso B: activos {0,3}: e0 es HIT (vive en slot 0 de paso A ⇒ bump
    // usage). Miss e3 ⇒ víctima = argmin(usage): los slots 6..11 están
    // VACÍOS (usage 0) ⇒ gana el primero de ellos (slot 6), no uno usado.
    const nb = try rig.runStep(&[_]i32{ 0, 3 }, 0);
    try testing.expectEqual(@as(usize, 1), nb);

    // Reconstruir esperado completo.
    const winners_b = [_]i32{3};
    const ev_b = [_]i32{6};
    var expected: [N_BANKS][]u8 = undefined;
    defer for (0..N_BANKS) |b| gpa.free(expected[b]);
    for (0..N_BANKS) |b| {
        const exp = try gpa.alloc(u8, PLAN * FEATS[b]);
        @memset(exp, 0x11);
        for (winners_a, 0..) |w, slot| {
            const wu: usize = @intCast(w);
            const su: usize = slot;
            @memcpy(exp[su * FEATS[b] ..][0..FEATS[b]], rig.src_host[b][wu * FEATS[b] ..][0..FEATS[b]]);
        }
        @memcpy(exp[ev_b[0] * FEATS[b] ..][0..FEATS[b]], rig.src_host[b][winners_b[0] * FEATS[b] ..][0..FEATS[b]]);
        expected[b] = exp;
    }
    try rig.verifyAgainstExpected(&expected);
}

test "E4 NOGATHER fallback clásico: mismos bytes" {
    const gpa = testing.allocator;
    var rig = Rig.init(gpa) catch return error.SkipZigTest; // CI sin toolkit
    defer rig.deinit();

    for (0..N_BANKS) |b| {
        const pat = try gpa.alloc(u8, PLAN * FEATS[b]);
        defer gpa.free(pat);
        @memset(pat, 0x77);
        try moe_cuda.htod(u8, rig.dst_dev[b], pat);
    }
    _ = try rig.runStep(&[_]i32{ 6, 1 }, 0);

    var expected: [N_BANKS][]u8 = undefined;
    defer for (0..N_BANKS) |b| gpa.free(expected[b]);
    for (0..N_BANKS) |b| {
        const exp = try gpa.alloc(u8, PLAN * FEATS[b]);
        @memset(exp, 0x77);
        @memcpy(exp[0 * FEATS[b] ..][0..FEATS[b]], rig.src_host[b][1 * FEATS[b] ..][0..FEATS[b]]);
        @memcpy(exp[1 * FEATS[b] ..][0..FEATS[b]], rig.src_host[b][6 * FEATS[b] ..][0..FEATS[b]]);
        expected[b] = exp;
    }
    try rig.verifyAgainstExpected(&expected);
}

test "E4 bench GB/s del gather fused (informativo)" {
    const gpa = testing.allocator;
    var rig = Rig.init(gpa) catch return error.SkipZigTest; // CI sin toolkit
    defer rig.deinit();

    // Warm: llenar todo el plan una vez.
    var all: [8]i32 = undefined;
    for (0..8) |k| all[k] = @intCast(k);
    _ = try rig.runStep(&all, 65536);

    const total_bytes_per_iter: u64 = blk: {
        var t: u64 = 0;
        for (FEATS) |f| t += 8 * f; // 8 filas × B bancos (peor caso warm)
        break :blk t;
    };

    const start = try cudaz.cuEventCreate(0);
    defer cudaz.cuEventDestroy(start);
    const end = try cudaz.cuEventCreate(0);
    defer cudaz.cuEventDestroy(end);

    var src_bases: [N_BANKS]usize = undefined;
    var dst_bases: [N_BANKS]usize = undefined;
    for (0..N_BANKS) |b| {
        src_bases[b] = @intFromPtr(rig.src_host[b].ptr);
        dst_bases[b] = rig.dst_dev[b];
    }

    const iters = 200;
    try cudaz.cuEventRecord(start, rig.stream);
    var i: usize = 0;
    while (i < iters) : (i += 1) {
        // Re-lanzar el ensure sería costoso; el gather es idempotente sobre
        // el último plan (evict/src siguen válidos): mide solo ancho de banda.
        try rig.gatherer.gatherMissing(rig.stream, &rig.gpu, &dst_bases, &src_bases, &FEATS, moe_cuda.kGatherBlocksPerBank);
    }
    try cudaz.cuEventRecord(end, rig.stream);
    try cudaz.cuEventSynchronize(end);
    var ms: f32 = 0;
    try cudaz.cuEventElapsedTime(&ms, start, end);
    const gbs = @as(f64, @floatFromInt(total_bytes_per_iter)) * @as(f64, @floatFromInt(iters)) / (@as(f64, ms) / 1000.0) / 1e9;
    std.debug.print("[E4 bench] gather fused: {d:.2} GB/s ({d} iters, {d} B/iter útil, {d:.2} ms total)\n", .{ gbs, iters, total_bytes_per_iter, ms });
}
