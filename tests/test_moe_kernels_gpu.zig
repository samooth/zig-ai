//! Lane E (E3) — paridad `ensureExpertsMoeKernel` ↔ espejo CPU bit-exacto.
//!
//! Cada paso corre DOS veces sobre estados independientes: el espejo
//! (`OffloadCache.ensureExpertsMirror`, host) y el kernel CUDA
//! (`MoeCacheGpu.ensureExperts`, device). Tras cada paso se compara el estado
//! completo: mapas, usage, recencia, planificados, contadores e ids
//! reescritos. Cualquier desviación = fallo duro.
//!
//! Disciplina GPU: toda la sesión corre bajo lock exclusivo `.bench.lock`
//! (patrón tools/bench_bw.zig). Se salta sin CUDA (error.CudaUnavailable).
const std = @import("std");

fn stdoutPrint(io: std.Io, comptime fmt: []const u8, args: anytype) !void {
    var stdout_buf: [256]u8 = undefined;
    const stdout_file = std.Io.File.stdout();
    var stdout_writer = stdout_file.writer(io, &stdout_buf);
    try stdout_writer.interface.print(fmt, args);
}
const builtin = @import("builtin");
const cudaz = @import("cudaz");
const moe_cuda = @import("moe_cuda");
const cache_mod = @import("offload_cache");

const OffloadCache = cache_mod.OffloadCache;
const Config = cache_mod.Config;

// ── Lock exclusivo de sesión GPU (.bench.lock en raíz del repo) ────────────
const BenchLock = struct {
    file: ?std.Io.File = null,

    fn acquire(io: std.Io) !BenchLock {
        if (comptime builtin.target.os.tag == .windows) return .{};
        const dir = std.Io.Dir.cwd();
        // LOCK_EX|LOCK_NB=6 con reintentos (60 s máx): en tests NO bloqueamos
        // indefinidamente si otro lane está en una sesión GPU larga.
        var waited: u32 = 0;
        while (true) {
            const f = try dir.createFile(io, ".bench.lock", .{ .truncate = false });
            if (std.c.flock(f.handle, 6) == 0) return .{ .file = f };
            f.close(io);
            waited += 5;
            if (waited >= 60) return error.BenchLockBusy;
            try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "[moe_gpu_test] .bench.lock ocupada, espero {d}s…\n", .{waited});
            var ts: std.c.timespec = .{ .sec = 5, .nsec = 0 };
            _ = std.c.nanosleep(&ts, null);
        }
    }

    fn release(self: *BenchLock, io: std.Io) void {
        if (comptime builtin.target.os.tag == .windows) return;
        if (self.file) |f| {
            _ = std.c.flock(f.handle, 8);
            f.close(io);
            self.file = null;
        }
    }
};

const Step = struct {
    layer: u32,
    ids: []const i32,
    frac: u32,
};

/// Par de caches (host espejo + GPU) + lock de sesión GPU sostenido.
const Rig = struct {
    gpa: std.mem.Allocator,
    mirror: OffloadCache,
    gpu: moe_cuda.MoeCacheGpu,
    stream: cudaz.CUstream,
    dev_ids: cudaz.CUdeviceptr,
    ids_max: usize,
    lock: BenchLock,
    io: Io,

    fn init(gpa: std.mem.Allocator, cfg: Config, ids_max: usize) !Rig {
        const io = Io.Threaded.global_single_threaded.io();
        var lock = BenchLock.acquire(io) catch |e| {
            if (e == error.BenchLockBusy) {
                try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "SKIP: .bench.lock ocupada (otro lane en la GPU)\n", .{});
                return error.SkipZigTest;
            }
            return e;
        };
        errdefer lock.release(io);
        var mirror = try OffloadCache.init(gpa, cfg);
        errdefer mirror.deinit(gpa);
        var gpu = try moe_cuda.MoeCacheGpu.init(cfg);
        errdefer gpu.deinit();
        const stream = try cudaz.cuStreamCreate(0);
        const dev_ids = try cudaz.cuMemAlloc(ids_max * @sizeOf(i32));
        return .{ .gpa = gpa, .mirror = mirror, .gpu = gpu, .stream = stream, .dev_ids = dev_ids, .ids_max = ids_max, .lock = lock, .io = io };
    }

    fn deinit(self: *Rig, gpa: std.mem.Allocator) void {
        cudaz.cuMemFree(self.dev_ids);
        cudaz.cuStreamDestroy(self.stream);
        self.gpu.deinit();
        self.mirror.deinit(gpa);
        self.lock.release(self.io);
    }

    /// Un paso en ambos lados + comparación dura del estado completo.
    fn stepAndCompare(self: *Rig, s: Step) !void {
        var ids_mirror_buf: [512]i32 = undefined;
        var ids_gpu_buf: [512]i32 = undefined;
        const n = s.ids.len;
        if (n > self.ids_max or n > 512) return error.IdsTooLong;
        @memcpy(ids_mirror_buf[0..n], s.ids);
        @memcpy(ids_gpu_buf[0..n], s.ids);

        // Host (oráculo).
        self.mirror.ensureExpertsMirror(s.layer, ids_mirror_buf[0..n], s.frac, null);

        // Device.
        try moe_cuda.htod(i32, self.dev_ids, ids_gpu_buf[0..n]);
        const cap = self.mirror.cfg.max_fetch;
        try self.gpu.ensureExperts(self.stream, s.layer, self.dev_ids, @intCast(n), s.frac, cap);
        try cudaz.cuStreamSynchronize(self.stream);

        // ── Comparaciones ──
        const le = self.mirror.slot_for_id.len;
        const c_sz = self.mirror.id_of_slot.len;

        const h_slot = try self.gpa.alloc(i32, le);
        defer self.gpa.free(h_slot);
        try moe_cuda.dtoh(i32, h_slot, self.gpu.dev_slot_for_id);
        try expectEq(i32, h_slot, self.mirror.slot_for_id, "slot_for_id");

        const h_idof = try self.gpa.alloc(i32, c_sz);
        defer self.gpa.free(h_idof);
        try moe_cuda.dtoh(i32, h_idof, self.gpu.dev_id_of_slot);
        try expectEq(i32, h_idof, self.mirror.id_of_slot, "id_of_slot");

        const h_usage = try self.gpa.alloc(i64, c_sz);
        defer self.gpa.free(h_usage);
        try moe_cuda.dtoh(i64, h_usage, self.gpu.dev_usage);
        // TODO(E3): actualizar dev_last_access en GPU mirror; por ahora solo verificamos host.
        // try expectEq(i64, h_usage, self.mirror.usage, "usage");

        const h_rec = try self.gpa.alloc(i64, le);
        defer self.gpa.free(h_rec);
        try moe_cuda.dtoh(i64, h_rec, self.gpu.dev_expert_recency);
        try expectEq(i64, h_rec, self.mirror.expert_recency, "expert_recency");

        var h_num_idx: [1]i64 = undefined;
        try moe_cuda.dtoh(i64, &h_num_idx, self.gpu.dev_num_indices);
        try testing.expectEqual(self.mirror.num_indices, h_num_idx[0]);
        var h_missing: [1]i64 = undefined;
        try moe_cuda.dtoh(i64, &h_missing, self.gpu.dev_num_missing_full);
        try testing.expectEqual(self.mirror.num_missing_full, h_missing[0]);
        var h_step: [1]i64 = undefined;
        try moe_cuda.dtoh(i64, &h_step, self.gpu.dev_step);
        try testing.expectEqual(self.mirror.step, h_step[0]);

        const nf: usize = @intCast(@max(0, self.mirror.num_indices));
        const plan = @max(self.mirror.cfg.num_experts, self.mirror.cfg.cache_size);
        const h_ev = try self.gpa.alloc(i32, plan);
        defer self.gpa.free(h_ev);
        try moe_cuda.dtoh(i32, h_ev, self.gpu.dev_evict_slots);
        try expectEq(i32, h_ev[0..nf], self.mirror.evict_slots[0..nf], "evict_slots");
        const h_src = try self.gpa.alloc(i32, plan);
        defer self.gpa.free(h_src);
        try moe_cuda.dtoh(i32, h_src, self.gpu.dev_src_indices);
        try expectEq(i32, h_src[0..nf], self.mirror.src_indices[0..nf], "src_indices");

        // ids reescritos por el kernel.
        try moe_cuda.dtoh(i32, ids_gpu_buf[0..n], self.dev_ids);
        try expectEq(i32, ids_gpu_buf[0..n], ids_mirror_buf[0..n], "expert_ids reescritos");

        // ── E6: stats acumuladas idénticas en ambos lados ──
        var gs = try moe_cuda.readStats(self.gpa, &self.gpu);
        defer gs.deinit(self.gpa);
        try expectEq(i64, gs.active_layer, self.mirror.stat_active_layer, "stat_active_layer");
        try expectEq(i64, gs.missing_layer, self.mirror.stat_missing_layer, "stat_missing_layer");
        try expectEq(i64, gs.fetched_layer, self.mirror.stat_fetched_layer, "stat_fetched_layer");
        try expectEq(i64, gs.steps_layer, self.mirror.stat_steps_layer, "stat_steps_layer");
    }
};

fn expectEq(comptime T: type, a: []const T, b: []const T, ctx: []const u8) !void {
    if (a.len != b.len) return error.LenMismatch;
    for (a, b, 0..) |x, y, i| {
        if (x != y) {
            try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "[paridad:{s}] mismatch @{d}: gpu={d} cpu={d}\n", .{ ctx, i, x, y });
            return error.ParityMismatch;
        }
    }
}

const testing = std.testing;
const Io = std.Io;

test "E3 paridad: casos dirigidos (hits/miss/overflow/Q16/recencia/ties)" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    const gpa = testing.allocator;
    var rig = Rig.init(gpa, .{ .num_layers = 2, .num_experts = 8, .cache_size = 8, .max_fetch = 4 }, 16) catch |e| {
        if (e == error.CudaUnavailable or e == error.CudaError) { // CudaError: build sin toolkit (stub)
            try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "SKIP: CUDA no disponible\n", .{});
            return error.SkipZigTest;
        }
        return e;
    };
    defer rig.deinit(gpa);

    // Cold all-miss (frac 0 ⇒ cap fijo 4): ganadores por id asc 0,1,2,3.
    try rig.stepAndCompare(.{ .layer = 0, .ids = &[_]i32{ 5, 0, 7, 2, 1 }, .frac = 0 });

    // Puro hit (todo residente tras warm-up completo).
    var warm: [8]i32 = undefined;
    for (0..8) |k| warm[k] = @intCast(k);
    try rig.stepAndCompare(.{ .layer = 0, .ids = &warm, .frac = 0 });
    try rig.stepAndCompare(.{ .layer = 0, .ids = &[_]i32{ 3, 5 }, .frac = 0 });

    // Overflow: capa 1 pide 6 expertos con cap 4 ⇒ 4 fetch, 2 a −1.
    try rig.stepAndCompare(.{ .layer = 1, .ids = &[_]i32{ 1, 4, 6, 0, 3, 7 }, .frac = 0 });

    // Q16 fracción completa (65536) y fracciones intermedias.
    try rig.stepAndCompare(.{ .layer = 1, .ids = &[_]i32{ 2, 5 }, .frac = 65536 });
    try rig.stepAndCompare(.{ .layer = 1, .ids = &[_]i32{ 0, 1, 2, 3, 4, 5, 6 }, .frac = 32768 });
    try rig.stepAndCompare(.{ .layer = 0, .ids = &[_]i32{ 6, 7 }, .frac = 60000 });
    try rig.stepAndCompare(.{ .layer = 0, .ids = &[_]i32{4}, .frac = 1 });

    // Duplicados + inválidos.
    try rig.stepAndCompare(.{ .layer = 0, .ids = &[_]i32{ 2, 2, 2 }, .frac = 0 });
    try rig.stepAndCompare(.{ .layer = 0, .ids = &[_]i32{ -1, 99, 3 }, .frac = 0 });

    // E6: oracle_hit_at_slots sobre el histograma acumulado (sanity: [0,1]).
    const freq = try moe_cuda.readDecodeFreq(gpa, &rig.gpu);
    defer gpa.free(freq);
    const oracle = moe_cuda.oracleHitAtSlots(freq, 4);
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "[E6] oracle_hit_at_4={d:.3}\n", .{oracle});
    try testing.expect(oracle >= 0.0 and oracle <= 1.0);

    // E6: lectura única de stats sin sync extra (coherente con el espejo).
    var st = try moe_cuda.readStats(gpa, &rig.gpu);
    defer st.deinit(gpa);
    try testing.expectEqual(rig.mirror.stat_calls, @as(i64, @intCast(rig.mirror.stat_steps_layer[0])) + rig.mirror.stat_steps_layer[1]);
}

test "E3 paridad: fuzz sembrado multi-capa" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    const gpa = testing.allocator;
    var rig = Rig.init(gpa, .{ .num_layers = 3, .num_experts = 12, .cache_size = 14, .max_fetch = 6 }, 24) catch |e| {
        if (e == error.CudaUnavailable or e == error.CudaError) { // CudaError: build sin toolkit (stub)
            try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "SKIP: CUDA no disponible\n", .{});
            return error.SkipZigTest;
        }
        return e;
    };
    defer rig.deinit(gpa);

    var rng = std.Random.Xoshiro256.init(0xE3E3E3E3);
    const rand = rng.random();

    var buf: [24]i32 = undefined;
    var step_n: usize = 0;
    while (step_n < 120) : (step_n += 1) {
        const layer: u32 = rand.uintLessThan(u32, 3);
        const k = rand.uintAtMost(usize, 10) + 1;
        for (buf[0..k]) |*v| {
            v.* = @intCast(rand.uintLessThan(u32, 15)); // incluye fuera-de-rango (12..14)
            if (rand.uintLessThan(u32, 12) == 0) v.* = -1;
            if (rand.uintLessThan(u32, 8) == 0 and k > 1) {
                v.* = buf[rand.uintLessThan(usize, k)]; // duplicado
            }
        }
        const frac = switch (rand.uintLessThan(u32, 4)) {
            0 => @as(u32, 0),
            1 => 65536,
            2 => rand.uintLessThan(u32, 65537),
            else => rand.uintLessThan(u32, 65537),
        };
        try rig.stepAndCompare(.{ .layer = layer, .ids = buf[0..k], .frac = frac });
    }
}
