//! Tests Lane F3: handshake submit/sync del cpu_executor (Contrato 8).
//!
//! Camino HOST (sin GPU): staging síncrono + coordinador + workers — valida
//! la maquinaria completa de slots/seq/watchdog con datos sintéticos. El
//! camino cuda_memops se ejercita en integración (E) y verificación manual
//! con nvidia-smi dmon (la ventana CPU debe dejar SMs libres).
const std = @import("std");
const builtin = @import("builtin");
const gemv_mod = @import("moe_cpu_gemv");
const exec_mod = @import("moe_cpu_executor");
const ext_sync = @import("cudaz_ext_sync");
const gguf_mod = @import("gguf");

// Geometría sintética: E expertos, proyección [out_dim × k_dim] por experto,
// W empaquetada q8_0 fila-major [n_experts][out_dim][rb].
const E: usize = 6;
const OUT_DIM: usize = 3;
const K: usize = 128;
const T: usize = 4; // tokens en el paso

fn fillW(w: []u8) void {
    const rb = gemv_mod.Format.q8_0.rowBytes(K);
    var scales: [1]f16 = .{1.0};
    for (0..E * OUT_DIM) |row| {
        scales[0] = @floatFromInt(1 + (row % 5));
        const base = row * rb;
        std.mem.writeInt(u16, w[base..][0..2], @bitCast(scales[0]), .little);
        for (0..32) |j| {
            w[base + 2 + j] = @truncate((row * 31 + j * 17 + 3));
        }
    }
}

test "submit→sync host: parcial correcto vs escalar" {
    const a = std.testing.allocator;
    const rb = gemv_mod.Format.q8_0.rowBytes(K);

    const w_q = try a.alloc(u8, E * OUT_DIM * rb);
    defer a.free(w_q);
    fillW(w_q);
    // Convención Contrato 8: bytes reinterpretados como []const f32.
    const w_as_f32: []const f32 = @as([*]const f32, @ptrCast(@alignCast(w_q.ptr)))[0 .. w_q.len / 4];

    var x: [T * K]f32 = undefined;
    var prng = std.Random.Xoshiro256.init(0xF3C);
    const rnd = prng.random();
    for (&x) |*v| v.* = rnd.float(f32) * 2.0 - 1.0;

    const ex = try exec_mod.Executor.initFull(a, .{
        .workers_requested = 3,
        .watchdog_ms = 5000,
        .fmt = .q8_0,
        .k_dim = K,
        .out_dim = OUT_DIM,
        .n_experts = E,
    });
    defer ex.deinit();

    // ids[t] = experto asignado al token t (≥0 ⇒ ruta CPU con peso real).
    const ids = [_]i32{ 2, 5, 0, 2 };
    const hidden_host_ptr = @intFromPtr(&x);

    const p = try ex.submit(7, hidden_host_ptr, w_as_f32, @constCast(&ids));
    try std.testing.expectEqual(@as(u32, 7), p.layer_id);
    const out = try ex.sync(p);
    defer ex.freePartial(out); // Contrato 8: sync() transfiere ownership (SIEMPRE freePartial)

    // Referencia secuencial bit-a-bit (misma primitiva dotScalar).
    for (0..T) |t| {
        for (0..OUT_DIM) |d| {
            const e: usize = @intCast(ids[t]);
            const row = w_q[(e * OUT_DIM + d) * rb ..][0..rb];
            const want = gemv_mod.dotScalar(.q8_0, row, x[t * K ..][0..K]);
            const got = out[t * OUT_DIM + d];
            const diff = @abs(want - got);
            if (!(diff / @max(@abs(want), 1e-9) < 1e-4)) {
                std.debug.print("t={d} d={d}: want={d} got={d}\n", .{ t, d, want, got });
                return error.SyncMismatch;
            }
        }
    }
}

test "dos capas en vuelo simultáneas" {
    const a = std.testing.allocator;
    const w_q = try a.alloc(u8, E * OUT_DIM * gemv_mod.Format.q8_0.rowBytes(K));
    defer a.free(w_q);
    fillW(w_q);
    const w_as_f32: []const f32 = @as([*]const f32, @ptrCast(@alignCast(w_q.ptr)))[0 .. w_q.len / 4];

    var x: [T * K]f32 = undefined;
    var prng = std.Random.Xoshiro256.init(0xD0);
    const rnd = prng.random();
    for (&x) |*v| v.* = rnd.float(f32) - 0.5;

    const ex = try exec_mod.Executor.initFull(a, .{
        .workers_requested = 2,
        .fmt = .q8_0,
        .k_dim = K,
        .out_dim = OUT_DIM,
        .n_experts = E,
    });
    defer ex.deinit();

    var ids_a = [_]i32{ 1, 3 };
    var ids_b = [_]i32{4};
    const pa = try ex.submit(10, @intFromPtr(&x), w_as_f32, &ids_a);
    const pb = try ex.submit(11, @intFromPtr(&x), w_as_f32, &ids_b);

    const out_a = try ex.sync(pa);
    ex.freePartial(out_a); // ownership transfer (Contrato 8: SIEMPRE freePartial)
    const out_b = try ex.sync(pb);
    ex.freePartial(out_b);
}

test "4.1 reciclaje: >max_slots submits en la vida del executor" {
    // Regresión del ticket 4.1: antes los slots NUNCA volvían a empty ⇒ tras
    // max_slots(16) submits, submit() devolvía SlotBusy y moe_layer saltaba
    // el overflow en silencio (tokens CPU desaparecidos del merge).
    const a = std.testing.allocator;
    const w_q = try a.alloc(u8, E * OUT_DIM * gemv_mod.Format.q8_0.rowBytes(K));
    defer a.free(w_q);
    fillW(w_q);
    const w_as_f32: []const f32 = @as([*]const f32, @ptrCast(@alignCast(w_q.ptr)))[0 .. w_q.len / 4];

    var x: [T * K]f32 = undefined;
    var prng = std.Random.Xoshiro256.init(0x51C);
    const rnd = prng.random();
    for (&x) |*v| v.* = rnd.float(f32) - 0.5;

    const ex = try exec_mod.Executor.initFull(a, .{
        .workers_requested = 2,
        .fmt = .q8_0,
        .k_dim = K,
        .out_dim = OUT_DIM,
        .n_experts = E,
    });
    defer ex.deinit();

    const ids = [_]i32{ 0, 1 };
    // 4×max_slots ciclos submit→sync (mucho más que los 16 slots del pool).
    const rounds: usize = 4 * exec_mod.max_slots;
    for (0..rounds) |i| {
        const p = try ex.submit(@intCast(i), @intFromPtr(&x), w_as_f32, &ids);
        const out = try ex.sync(p);
        ex.freePartial(out);
    }
    // Todos los slots deben estar empty y sin claim tras el reciclaje.
    for (&ex.slots) |*s| {
        try std.testing.expectEqual(@intFromEnum(exec_mod.StState.empty), s.state.load(.monotonic));
        try std.testing.expect(!s.claim.load(.monotonic));
    }
}

test "4.1 drainSlot: pending sin consumir se recicla (par submit fallido)" {
    const a = std.testing.allocator;
    const rb = gemv_mod.Format.q8_0.rowBytes(K);
    const w_q = try a.alloc(u8, E * OUT_DIM * rb);
    defer a.free(w_q);
    fillW(w_q);
    const w_as_f32: []const f32 = @as([*]const f32, @ptrCast(@alignCast(w_q.ptr)))[0 .. w_q.len / 4];

    var x: [K]f32 = undefined;
    @memset(&x, 0.5);

    const ex = try exec_mod.Executor.initFull(a, .{
        .workers_requested = 1,
        .fmt = .q8_0,
        .k_dim = K,
        .out_dim = OUT_DIM,
        .n_experts = E,
    });
    defer ex.deinit();

    // Simula el patrón moe_layer: pg OK, pu "falla" (aquí lo hacemos OK y
    // lo drenamos sin sync) — el drain debe devolver el slot a empty.
    const ids = [_]i32{1};
    const pg = try ex.submit(3, @intFromPtr(&x), w_as_f32, &ids);
    const pu = try ex.submit(3, @intFromPtr(&x), w_as_f32, &ids);

    ex.drainSlot(pu); // descartado sin consumir
    const out_g = try ex.sync(pg);
    ex.freePartial(out_g);

    // Drain idempotente tras sync (StalePending interno = no-op).
    ex.drainSlot(pu);

    for (&ex.slots) |*s| {
        try std.testing.expectEqual(@intFromEnum(exec_mod.StState.empty), s.state.load(.monotonic));
        try std.testing.expect(!s.claim.load(.monotonic));
    }
    // Y el pool sigue operativo: un submit posterior debe funcionar.
    const p2 = try ex.submit(4, @intFromPtr(&x), w_as_f32, &ids);
    const out2 = try ex.sync(p2);
    ex.freePartial(out2);
}

test "4.2 sync doble sobre el mismo Pending → StalePending (sin aliasing)" {
    const a = std.testing.allocator;
    const rb = gemv_mod.Format.q8_0.rowBytes(K);
    const w_q = try a.alloc(u8, E * OUT_DIM * rb);
    defer a.free(w_q);
    fillW(w_q);
    const w_as_f32: []const f32 = @as([*]const f32, @ptrCast(@alignCast(w_q.ptr)))[0 .. w_q.len / 4];
    var x: [K]f32 = undefined;
    @memset(&x, 0.5);

    const ex = try exec_mod.Executor.initFull(a, .{
        .workers_requested = 1,
        .fmt = .q8_0,
        .k_dim = K,
        .out_dim = OUT_DIM,
        .n_experts = E,
    });
    defer ex.deinit();

    const ids = [_]i32{2};
    const p = try ex.submit(5, @intFromPtr(&x), w_as_f32, &ids);
    const out = try ex.sync(p);
    ex.freePartial(out);
    // El slot ya se recicló ⇒ el segundo sync es StalePending, NO devuelve
    // un aliasing del buffer ya liberado (double-use del caller imposible).
    try std.testing.expectError(error.StalePending, ex.sync(p));
}

test "watchdog: slot wedged manual → WatchdogTimeout + flag" {
    const a = std.testing.allocator;
    var x: [K]f32 = undefined;
    @memset(&x, 0.25);

    const ex = try exec_mod.Executor.initFull(a, .{
        .workers_requested = 1,
        .watchdog_ms = 100,
        .fmt = .q8_0,
        .k_dim = K,
        .out_dim = OUT_DIM,
        .n_experts = E,
    });
    defer ex.deinit();

    // Wedge manual determinista del slot 5: staged con edad > watchdog.
    const s = &ex.slots[5];
    s.claim.store(true, .release);
    s.layer_id = 9;
    s.seq = 1;
    s.posted_ms = exec_mod.nowMsPublic() - 5000;
    s.state.store(@intFromEnum(exec_mod.StState.staged), .release);

    const r = ex.sync(.{ .slot = 5, .layer_id = 9, .seq = 1 });
    try std.testing.expectError(error.WatchdogTimeout, r);
    try std.testing.expect(ex.wedged.load(.monotonic));
    std.debug.print("[test] watchdog ok: wedged flag activo\n", .{});
}

test "shutdown en vuelo → error limpio, hilos salen" {
    const a = std.testing.allocator;
    const w_q = try a.alloc(u8, E * OUT_DIM * gemv_mod.Format.q8_0.rowBytes(K));
    defer a.free(w_q);
    fillW(w_q);
    const w_as_f32: []const f32 = @as([*]const f32, @ptrCast(@alignCast(w_q.ptr)))[0 .. w_q.len / 4];
    var x: [K]f32 = undefined;
    @memset(&x, 0.5);

    const ex = try exec_mod.Executor.initFull(a, .{
        .workers_requested = 1,
        .fmt = .q8_0,
        .k_dim = K,
        .out_dim = OUT_DIM,
        .n_experts = E,
    });
    const ids = [_]i32{0};
    const p = try ex.submit(4, @intFromPtr(&x), w_as_f32, @constCast(&ids));
    // Coordinador "muerto": shutdown sin drain (sin join aún).
    ex.shutdown.store(true, .release);
    try std.testing.expectError(error.ExecutorShutdown, ex.sync(p));
    ex.deinit(); // joins limpios
}

test "determinismo: dos corridas idénticas bit a bit" {
    const a = std.testing.allocator;
    const rb = gemv_mod.Format.q8_0.rowBytes(K);
    const w_q = try a.alloc(u8, E * OUT_DIM * rb);
    defer a.free(w_q);
    fillW(w_q);
    const w_as_f32: []const f32 = @as([*]const f32, @ptrCast(@alignCast(w_q.ptr)))[0 .. w_q.len / 4];

    var x: [T * K]f32 = undefined;
    var prng = std.Random.Xoshiro256.init(0xDE7);
    const rnd = prng.random();
    for (&x) |*v| v.* = rnd.float(f32) * 2 - 1;

    const ids = [_]i32{ 3, 1, 4, 1 };

    var run1: [T * OUT_DIM]f32 = undefined;
    var run2: [T * OUT_DIM]f32 = undefined;

    {
        const ex = try exec_mod.Executor.initFull(a, .{ .workers_requested = 4, .fmt = .q8_0, .k_dim = K, .out_dim = OUT_DIM, .n_experts = E });
        defer ex.deinit();
        const p = try ex.submit(1, @intFromPtr(&x), w_as_f32, @constCast(&ids));
        const o = try ex.sync(p);
        @memcpy(&run1, o);
        a.free(o); // ownership transfer (Contrato 8)
    }
    {
        const ex = try exec_mod.Executor.initFull(a, .{ .workers_requested = 2, .fmt = .q8_0, .k_dim = K, .out_dim = OUT_DIM, .n_experts = E });
        defer ex.deinit();
        const p = try ex.submit(1, @intFromPtr(&x), w_as_f32, @constCast(&ids));
        const o = try ex.sync(p);
        @memcpy(&run2, o);
        a.free(o);
    }
    for (run1, run2) |r1, r2| {
        try std.testing.expectEqual(@as(u32, @bitCast(r1)), @as(u32, @bitCast(r2)));
    }
}

test "probe memops reporta estado (informativo)" {
    std.debug.print("[test] modo esperado sin GPU adjunta: host_staging; memops={}\n", .{ext_sync.memopsAvailable()});
}

// ============================================================================
// Contrato 6: benchbw.json — derivación q* y lectura con fallbacks seguros.
// ============================================================================

test "fetchFracQ16: derivación, clamps e inválidos" {
    // q = pcie_ov/(cpu_ov+pcie_ov); caso del ejemplo del contrato:
    const ex = exec_mod.fetchFracQ16FromGbs(27.1, 11.8).?;
    const want: u32 = @intFromFloat(@round(11.8 / (27.1 + 11.8) * 65536.0));
    try std.testing.expectEqual(want, @as(u32, ex));
    // Clamps: pcie dominante → 65536; cpu dominante → ~0.
    try std.testing.expectEqual(@as(u32, 65536), exec_mod.fetchFracQ16FromGbs(0.001, 1000).?);
    try std.testing.expectEqual(@as(u32, 0), exec_mod.fetchFracQ16FromGbs(1000, 0.0001).?);
    // Inválidos → null (llamante aplicará cap 1).
    try std.testing.expectEqual(@as(?u32, null), exec_mod.fetchFracQ16FromGbs(-1, 5));
    try std.testing.expectEqual(@as(?u32, null), exec_mod.fetchFracQ16FromGbs(0, 0));
}

test "parseBenchBwText: válido, corrupto y campos ausentes" {
    const a = std.testing.allocator;
    const good =
        \\{ "version": 1,
        \\  "cpu_gbs": 38.2, "pcie_gbs": 13.9,
        \\  "cpu_overlap_gbs": 27.1, "pcie_overlap_gbs": 11.8,
        \\  "verdict": "hybrid", "fetch_frac_q16": 12190 }
    ;
    const bw = exec_mod.parseBenchBwText(a, good) orelse return error.ShouldParse;
    try std.testing.expectEqual(@as(u32, 1), bw.version);
    try std.testing.expectEqualStrings("hybrid", bw.verdict);
    try std.testing.expectEqual(@as(u32, 12190), bw.reported_fetch_frac_q16.?);
    const frac = exec_mod.fetchFracQ16FromGbs(bw.cpu_overlap_gbs, bw.pcie_overlap_gbs).?;
    try std.testing.expect(frac > 15000 and frac < 25000);

    try std.testing.expectEqual(@as(?exec_mod.BenchBw, null), exec_mod.parseBenchBwText(a, "{ no json ]"));
    try std.testing.expectEqual(@as(?exec_mod.BenchBw, null), exec_mod.parseBenchBwText(a, "{\"version\":2}")); // versión desconocida
    // Sin overlaps → null (cap 1 aguas abajo).
    try std.testing.expectEqual(@as(?exec_mod.BenchBw, null), exec_mod.parseBenchBwText(a, "{\"version\":1}"));
}

test "resolveFetchFracQ16: fichero real vs ausente (cap seguro)" {
    if (comptime builtin.target.os.tag == .windows) return error.SkipZigTest;
    const a = std.testing.allocator;
    // Ausente → cap 1.
    try std.testing.expectEqual(@as(u32, 1), exec_mod.resolveFetchFracQ16(a, "/tmp/opencode/no-existe-benchbw.json"));

    // Fichero válido en /tmp/opencode (área de trabajo aprobada).
    const path = "/tmp/opencode/benchbw_fixture.json";
    writeFixture(path);
    const frac = exec_mod.resolveFetchFracQ16(a, path);
    try std.testing.expectEqual(@as(u32, 32768), frac); // 50% exacto
    deleteFixture(path);
}

// helpers de fixture vía libc (std.fs.cwd no disponible en este build)
extern "c" fn open(path: [*:0]const u8, flags: c_int) c_int;
// Variante con modo (ABI-compat con open variadic de glibc): sin el 3er arg
// el modo sale con basura del registro y el fichero nace ilegible.
extern "c" fn open64(path: [*:0]const u8, flags: c_int, mode: c_uint) c_int;
extern "c" fn write(fd: c_int, buf: [*]const u8, n: usize) isize;
extern "c" fn read(fd: c_int, buf: [*]u8, count: usize) isize;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn unlink(path: [*:0]const u8) c_int;

fn writeFixture(path: [:0]const u8) void {
    if (comptime builtin.target.os.tag == .windows) return;
    const fd = open64(path, 0o1101, 0o644); // O_WRONLY|O_CREAT|O_TRUNC, rw-r--r--
    if (fd < 0) return;
    defer _ = close(fd);
    const body =
        \\{"version":1,"cpu_gbs":30,"pcie_gbs":12,
        \\"cpu_overlap_gbs":20,"pcie_overlap_gbs":20,"verdict":"hybrid"}
    ;
    _ = write(fd, body.ptr, body.len);
}
fn deleteFixture(path: [:0]const u8) void {
    if (comptime builtin.target.os.tag == .windows) return;
    _ = unlink(path);
}

test "perfil REAL de D (smoke oportunista): deriva q16 si benchbw existe" {
    // No aserta contenido (D puede regenerarlo); valida que mi lector
    // procesa el fichero REAL del Contrato 6 end-to-end cuando está presente
    // y emite el valor para el ticket a E.
    const a = std.testing.allocator;
    const frac = exec_mod.resolveFetchFracQ16(a, null);
    std.debug.print("[test] fetch_frac_q16 efectivo (perfil real o cap): {d}\n", .{frac});
}

// ============================================================================
// Semántica híbrida INVIOLABLE (Contrato 8 / LANE_F):
//   ids −1 → CPU con peso REAL · ruta GPU pone pesos CPU a 0 ·
//   merge = SUMA de parciales · cada token computado EXACTAMENTE una vez ⇒
//   salida idéntica a la referencia monolítica (misma seed).
// ============================================================================

test "merge gpu+cpu == referencia monolítica (split por ids)" {
    const a = std.testing.allocator;
    const rb = gemv_mod.Format.q8_0.rowBytes(K);
    const w_q = try a.alloc(u8, E * OUT_DIM * rb);
    defer a.free(w_q);
    fillW(w_q);
    var x: [T * K]f32 = undefined;
    var prng = std.Random.Xoshiro256.init(0x3EF);
    const rnd = prng.random();
    for (&x) |*v| v.* = rnd.float(f32) * 2.0 - 1.0;

    // Referencia monolítica: TODOS los tokens con pesos reales.
    var ref: [T * OUT_DIM]f32 = undefined;
    const ids_full = [_]i32{ 0, 1, 2, 3 };
    for (0..T) |t| {
        for (0..OUT_DIM) |d| {
            ref[t * OUT_DIM + d] = gemv_mod.dotScalar(
                .q8_0,
                w_q[(@as(usize, @intCast(ids_full[t])) * OUT_DIM + d) * rb ..][0..rb],
                x[t * K ..][0..K],
            );
        }
    }

    // Split: tokens {1,3} → CPU (peso real); GPU procesa todos pero con los
    // pesos de los expertos CPU-routed PUESTOS A CERO (como hace el kernel).
    const cpu_mask = [_]bool{ false, true, false, true };
    const ex = try exec_mod.Executor.initFull(a, .{
        .workers_requested = 2,
        .fmt = .q8_0,
        .k_dim = K,
        .out_dim = OUT_DIM,
        .n_experts = E,
    });
    defer ex.deinit();

    // parcial CPU: sólo tokens enmascarados (los otros NO se envían: −1).
    var ids_cpu_buf: [T]i32 = undefined;
    for (0..T) |t| {
        ids_cpu_buf[t] = if (cpu_mask[t]) ids_full[t] else -1;
    }
    const p = try ex.submit(5, @intFromPtr(&x), @as([*]const f32, @ptrCast(@alignCast(w_q.ptr)))[0 .. w_q.len / 4], &ids_cpu_buf);
    // El executor salta ids −1 (defensivo); el out queda para tokens CPU.
    const cpu_out = try ex.sync(p);
    defer a.free(cpu_out); // ownership transfer (Contrato 8)

    // parcial GPU simulado: mismos tokens, W con filas CPU-routed a CERO.
    const w_gpu = try a.alloc(u8, w_q.len);
    defer a.free(w_gpu);
    @memcpy(w_gpu, w_q);
    for (0..T) |t| {
        if (!cpu_mask[t]) continue;
        const e: usize = @intCast(ids_full[t]);
        @memset(w_gpu[e * OUT_DIM * rb ..][0 .. OUT_DIM * rb], 0);
    }
    var gpu_out: [T * OUT_DIM]f32 = undefined;
    for (0..T) |t| {
        for (0..OUT_DIM) |d| {
            gpu_out[t * OUT_DIM + d] = gemv_mod.dotScalar(
                .q8_0,
                w_gpu[(@as(usize, @intCast(ids_full[t])) * OUT_DIM + d) * rb ..][0..rb],
                x[t * K ..][0..K],
            );
        }
    }

    // MERGE = suma; cada token exactamente una vez. NOTA: el executor usa
    // índices POSICIONALES (ids −1 dejan hueco sin tocar en cpu_out).
    var merged: [T * OUT_DIM]f32 = undefined;
    for (0..T) |t| {
        for (0..OUT_DIM) |d| {
            merged[t * OUT_DIM + d] = gpu_out[t * OUT_DIM + d];
        }
        if (cpu_mask[t]) {
            for (0..OUT_DIM) |d| {
                merged[t * OUT_DIM + d] += cpu_out[t * OUT_DIM + d];
            }
        }
    }

    // Paridad vs monolítico: productos bit-idénticos, suma en distinto orden
    // ⇒ epsilon estrecho (misma política que el resto de suites lane-f).
    for (0..T * OUT_DIM) |ix| {
        const diff = @abs(ref[ix] - merged[ix]);
        if (!(diff / @max(@abs(ref[ix]), 1e-9) < 1e-4)) {
            std.debug.print("ix={d}: ref={d} merged={d}\n", .{ ix, ref[ix], merged[ix] });
            return error.MergeMismatch;
        }
    }
}

test "E8: overhead medido de cuLaunchHostFunc (informativo si hay driver)" {
    if (!ext_sync.hostFuncAvailable()) {
        std.debug.print("[test] E8: sin driver/hostfunc — skip\n", .{});
        return;
    }
    if (ext_sync.benchHostFuncOverhead(200)) |r| {
        std.debug.print("[test] E8 hostfunc: enqueue={d:.1}ns roundtrip={d:.1}ns por llamada\n", .{ r.enqueue_ns, r.roundtrip_ns });
        // Referencia FreeToken ~30-50µs round-trip en host libre. Techo
        // generoso anti-flake (loadavg alto del árbol compartido puede
        // multiplicar el round-trip ×10; informativo, no gate de HW).
        try std.testing.expect(r.roundtrip_ns < 5_000_000.0); // <5ms
    } else {
        std.debug.print("[test] E8: bench no disponible\n", .{});
    }
}

// ============================================================================
// Validación contra la fixture MoE REAL de lane-e (GGUF con bancos q4_1
// fusionados): geometría de mis rowBytes contra tensores reales + dot de
// filas reales vs oráculo. Skip elegante si el fixture no existe.
// ============================================================================
const fixture_paths = [_][]const u8{
    "/tmp/opencode/moe_fixture.gguf", // bancos 3-D apilados [in,out,E]
    "/tmp/opencode/moe_fixture_split.gguf", // expertos individuales 2-D
};

test "fixtures MoE reales: geometría + dot filas reales q4_1 (ambas variantes)" {
    if (comptime builtin.target.os.tag == .windows) return error.SkipZigTest;
    const a = std.testing.allocator;
    var validated_any = false;

    for (fixture_paths) |path| {
        const bytes = readFileAlloc(a, path) catch continue; // ausente → skip
        defer a.free(bytes);

        var model = try gguf_mod.GgufFile.fromBytes(a, bytes);
        defer model.deinit();

        var it = model.tensors.iterator();
        while (it.next()) |entry| {
            const ti = entry.value_ptr;
            if (ti.dtype != .q4_1) continue;
            if (std.mem.indexOf(u8, ti.name, "blk.0.") == null) continue;
            if (std.mem.indexOf(u8, ti.name, "ffn_") == null) continue;
            // Solo matrices de pesos (descarta routers f32 y norms).
            if (std.mem.indexOf(u8, ti.name, "inp") != null) continue;
            if (std.mem.indexOf(u8, ti.name, "norm") != null) continue;

            const shape = ti.shape();
            const n_in: usize = @intCast(shape[0]);
            try std.testing.expect(n_in % 32 == 0);
            const rb = gemv_mod.Format.q4_1.rowBytes(n_in);
            const data = model.tensorData(ti);
            try std.testing.expectEqual(@as(usize, 0), data.len % rb);

            // Dot fila real vs oráculo+secuencial en cada variante.
            if (n_in >= 64 and n_in <= 1024 and shape.len >= 2) {
                var x: [1024]f32 = undefined;
                for (x[0..n_in]) |*v| v.* = 0.25;
                const got = gemv_mod.dotScalar(.q4_1, data[0..rb], x[0..n_in]);
                var dq: [1024]f32 = undefined;
                gguf_mod.dequantBlock(.q4_1, data[0..rb], dq[0..n_in], n_in);
                var want: f32 = 0;
                for (dq[0..n_in], x[0..n_in]) |dv, xv| want += dv * xv;
                const diff = @abs(want - got);
                try std.testing.expect(diff / @max(@abs(want), 1e-9) < 1e-6);
                validated_any = true;
                if (checked_print < 3) {
                    checked_print += 1;
                    std.debug.print("[test] fixture {s}: {any} rb={d} dot OK\n", .{ ti.name, shape, rb });
                }
            }
        }
    }
    if (validated_any) {
        checked_print = 0;
        std.debug.print("[test] fixtures MoE validadas contra GEMV CPU\n", .{});
    }
}
var checked_print: usize = 0;

// Lectura completa de fichero vía libc (reusa externs open/read/close de
// los helpers writeFixture/deleteFixture declarados arriba).
fn readFileAlloc(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    if (comptime builtin.target.os.tag == .windows) return error.FileNotFound;
    var pathz: [512]u8 = undefined;
    const pz = try std.fmt.bufPrintZ(&pathz, "{s}", .{path});
    const fd = open(pz.ptr, 0);
    if (fd < 0) return error.FileNotFound;
    defer _ = close(fd);
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    var chunk: [65536]u8 = undefined;
    while (true) {
        const n = read(fd, &chunk, chunk.len);
        if (n <= 0) break;
        try buf.appendSlice(allocator, chunk[0..@intCast(n)]);
    }
    return try buf.toOwnedSlice(allocator);
}

// ============================================================================
// Orquestación FFN COMPLETA en CPU (patrón para el wiring de E):
//   gate/up comparten geometría ⇒ un Executor; down es otra geometría ⇒ otro.
//   Secuencia: submit(gate)+submit(up) en vuelo → sync ambos → swiglu HOST
//   (misma fórmula del kernel: g/(1+e^-g)·u) → submit(down sobre act) →
//   sync → parcial listo para merge. Determinista y sin carreras entre slots.
// ============================================================================
const FF: usize = 96;
const DOWN_OUT: usize = 64;

fn fillBankQ80(w: []u8, e_count: usize, out_dim: usize, k_dim: usize, salt: u8) void {
    const rb = gemv_mod.Format.q8_0.rowBytes(k_dim);
    var b: usize = 0;
    while (b < e_count * out_dim) : (b += 1) {
        const base = b * rb;
        std.mem.writeInt(u16, w[base..][0..2], @bitCast(@as(f16, @floatFromInt(1 + (b % 4)))), .little);
        for (0..32) |j| {
            w[base + 2 + j] = salt +% @as(u8, @truncate(b *% 13 +% j *% 7));
        }
    }
}

test "FFN completo en CPU: gate→up→swiglu→down orquestado == monolítico" {
    const a = std.testing.allocator;
    const rb_k = gemv_mod.Format.q8_0.rowBytes(K);
    const rb_ff = gemv_mod.Format.q8_0.rowBytes(FF);

    const w_gate = try a.alloc(u8, E * FF * rb_k);
    defer a.free(w_gate);
    const w_up = try a.alloc(u8, E * FF * rb_k);
    defer a.free(w_up);
    const w_down = try a.alloc(u8, E * DOWN_OUT * rb_ff);
    defer a.free(w_down);
    fillBankQ80(w_gate, E, FF, K, 0x10);
    fillBankQ80(w_up, E, FF, K, 0xA0);
    fillBankQ80(w_down, E, DOWN_OUT, FF, 0x55);
    const wq = struct {
        fn cast(bytes: []const u8) []const f32 {
            return @as([*]const f32, @ptrCast(@alignCast(bytes.ptr)))[0 .. bytes.len / 4];
        }
    };

    const NTOK = 3;
    var x: [NTOK * K]f32 = undefined;
    var prng = std.Random.Xoshiro256.init(0xF0F);
    const rnd = prng.random();
    for (&x) |*v| v.* = rnd.float(f32) * 2.0 - 1.0;

    // Referencia monolítica (mismas fases en escalar).
    // MISMA asignación que el split de abajo: token0→e2 · token1→e1(GPU) · token2→e3
    const ids_full = [_]i32{ 2, 1, 3 };
    var ref: [NTOK * DOWN_OUT]f32 = undefined;
    for (0..NTOK) |t| {
        const e: usize = @intCast(ids_full[t]);
        var g: [FF]f32 = undefined;
        var u: [FF]f32 = undefined;
        for (0..FF) |d| {
            g[d] = gemv_mod.dotScalar(.q8_0, w_gate[(e * FF + d) * rb_k ..][0..rb_k], x[t * K ..][0..K]);
            u[d] = gemv_mod.dotScalar(.q8_0, w_up[(e * FF + d) * rb_k ..][0..rb_k], x[t * K ..][0..K]);
            g[d] = g[d] / (1.0 + @exp(-g[d])) * u[d]; // swigluKernel exacto
        }
        for (0..DOWN_OUT) |d| {
            ref[t * DOWN_OUT + d] = gemv_mod.dotScalar(.q8_0, w_down[(e * DOWN_OUT + d) * rb_ff ..][0..rb_ff], &g);
        }
    }

    // Ejecutores: uno para gate/up (ff×k) y otro para down (down_out×ff).
    const ex_gu = try exec_mod.Executor.initFull(a, .{ .workers_requested = 2, .fmt = .q8_0, .k_dim = K, .out_dim = FF, .n_experts = E });
    defer ex_gu.deinit();
    const ex_dn = try exec_mod.Executor.initFull(a, .{ .workers_requested = 2, .fmt = .q8_0, .k_dim = FF, .out_dim = DOWN_OUT, .n_experts = E });
    defer ex_dn.deinit();

    // Split: token 1 va a GPU (−1 aquí); 0 y 2 a CPU.
    var ids_cpu = [_]i32{ 2, -1, 3 };
    const hp = @intFromPtr(&x);

    // Dos pendings EN VUELO simultáneos (slots independientes del mismo pool).
    const pg = try ex_gu.submit(1, hp, wq.cast(w_gate), &ids_cpu);
    const pu = try ex_gu.submit(1, hp, wq.cast(w_up), &ids_cpu);
    const og = try ex_gu.sync(pg);
    const ou = try ex_gu.sync(pu);
    defer ex_gu.freePartial(og); // ownership transfer (Contrato 8: SIEMPRE freePartial)
    defer ex_gu.freePartial(ou);

    // swiglu HOST sólo en tokens CPU (índices posicionales; −1 intacto).
    var act: [NTOK * FF]f32 = undefined;
    for (0..NTOK) |t| {
        if (ids_cpu[t] < 0) continue;
        for (0..FF) |d| {
            const gv = og[t * FF + d];
            act[t * FF + d] = gv / (1.0 + @exp(-gv)) * ou[t * FF + d];
        }
    }

    const pd = try ex_dn.submit(1, @intFromPtr(&act), wq.cast(w_down), &ids_cpu);
    const od = try ex_dn.sync(pd);
    defer ex_dn.freePartial(od); // ownership transfer (Contrato 8: SIEMPRE freePartial)

    // Merge posicional contra referencia.
    for (0..NTOK) |t| {
        if (ids_cpu[t] < 0) continue;
        for (0..DOWN_OUT) |d| {
            const got = od[t * DOWN_OUT + d];
            const want = ref[t * DOWN_OUT + d];
            const diff = @abs(want - got);
            if (!(diff / @max(@abs(want), 1e-9) < 1e-4)) {
                std.debug.print("t={d} d={d}: want={d} got={d}\n", .{ t, d, want, got });
                return error.FfnMismatch;
            }
        }
    }
}

// ============================================================================
// Smoke contra MoE REAL descargada (gemma-4-26B-A4B UD-Q6_K_XL):
// lectura DIRIGIDA por pread (el archivo es grande; nunca se carga entero).
// Valida: geometría del banco gate_up_exps (Q6_K estándar apilado
// [n_embd, n_ff, E]) contra mis rowBytes + dot de fila real vs oráculo.
// NOTA documentada: los tensores `down` del preset XL llevan escalas
// EXTERNAS (.scale separado) ⇒ fuera del alcance del GEMV estándar.
// ============================================================================
const gemma4_path = "/ai/models/gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf";

extern "c" fn pread(fd: c_int, buf: [*]u8, count: usize, offset: i64) isize;

test "gemma-4 real: banco gate_up Q6_K apilado cuadra rowBytes + dot fila real" {
    if (comptime builtin.target.os.tag == .windows) return error.SkipZigTest;
    const a = std.testing.allocator;
    var pathz: [512]u8 = undefined;
    const pz = std.fmt.bufPrintZ(&pathz, "{s}", .{gemma4_path}) catch return error.NameTooLong;
    const fd = open(pz.ptr, 0);
    if (fd < 0) {
        std.debug.print("[test] gemma-4 no presente — skip\n", .{});
        return;
    }
    defer _ = close(fd);

    // --- mini-parser GGUF v3 dirigido (header completo, tipos todos) ---
    const SIZES = [_]usize{ 1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8 };
    var buf: [4096]u8 = undefined;
    var magic: [4]u8 = undefined;
    _ = try preadAll(fd, &magic, 0);
    try std.testing.expectEqualStrings("GGUF", &magic);
    var pos: u64 = 4;

    const rdU64 = struct {
        fn go(fd_: c_int, p: *u64) !u64 {
            var b: [8]u8 = undefined;
            _ = try preadAll(fd_, &b, p.*);
            p.* += 8;
            return std.mem.readInt(u64, &b, .little);
        }
    }.go;

    pos += 4; // version
    const n_tensors = try rdU64(fd, &pos);
    const n_kv = try rdU64(fd, &pos);

    var kv_ix: u64 = 0;
    while (kv_ix < n_kv) : (kv_ix += 1) {
        const kl = try rdU64(fd, &pos);
        _ = try preadAll(fd, buf[0..kl], pos);
        pos += kl;
        var b4kv: [4]u8 = undefined;
        _ = try preadAll(fd, &b4kv, pos);
        const vt_raw = std.mem.readInt(u32, &b4kv, .little);
        pos += 4;
        if (vt_raw == 8) {
            const sl = try rdU64(fd, &pos);
            pos += sl;
        } else if (vt_raw == 9) {
            const et = blk: {
                var b: [4]u8 = undefined;
                _ = try preadAll(fd, &b, pos);
                break :blk std.mem.readInt(u32, &b, .little);
            };
            pos += 4;
            const cnt = try rdU64(fd, &pos);
            if (et == 8) {
                for (0..cnt) |_| {
                    const sl = try rdU64(fd, &pos);
                    pos += sl;
                }
            } else pos += SIZES[et] * cnt;
        } else pos += SIZES[vt_raw];
    }

    // Tensores: buscar blk.N.ffn_gate_up_exps.weight (primero).
    var found_dims: [3]u64 = .{ 0, 0, 0 };
    var found_off: u64 = 0;
    var found_dtype: u32 = 0;
    var ti: u64 = 0;
    while (ti < n_tensors) : (ti += 1) {
        const nl = try rdU64(fd, &pos);
        var name_len: usize = @intCast(@min(nl, buf.len));
        _ = try preadAll(fd, buf[0..name_len], pos);
        pos += nl;
        const name = buf[0..name_len];

        var b4: [4]u8 = undefined;
        _ = try preadAll(fd, &b4, pos);
        const nd_raw = std.mem.readInt(u32, &b4, .little);
        pos += 4;

        const ndims: usize = @min(nd_raw, 4);
        var dims: [3]u64 = .{ 0, 0, 0 };
        for (0..ndims) |d| {
            var b8: [8]u8 = undefined;
            _ = try preadAll(fd, &b8, pos);
            if (d < 3) dims[d] = std.mem.readInt(u64, &b8, .little);
            pos += 8;
        }
        _ = try preadAll(fd, &b4, pos);
        const dt = std.mem.readInt(u32, &b4, .little);
        pos += 4;
        var b8t: [8]u8 = undefined;
        _ = try preadAll(fd, &b8t, pos);
        const toff = std.mem.readInt(u64, &b8t, .little);
        pos += 8;

        if (found_dims[0] == 0 and dt == 14 and
            std.mem.indexOf(u8, name, "ffn_gate_up_exps") != null)
        {
            found_dims = dims;
            found_off = toff;
            found_dtype = dt;
        }
        _ = &name_len;
    }

    try std.testing.expect(found_dims[0] > 0);
    try std.testing.expectEqual(@as(u32, 14), found_dtype); // Q6_K
    const n_in: usize = @intCast(found_dims[0]);
    const n_out: usize = @intCast(found_dims[1]);
    const n_e: usize = @intCast(found_dims[2]);
    const rb = gemv_mod.Format.q6_k.rowBytes(n_in);
    std.debug.print("[test] gemma-4 gate_up: [{d},{d},{d}] rb={d}\n", .{ n_in, n_out, n_e, rb });

    // Leer UNA fila real (expert 0, row 0) por pread dirigido.
    const row = try a.alloc(u8, rb);
    defer a.free(row);
    _ = try preadAll(fd, row, found_off);

    var x: [512]f32 = undefined;
    const k_use = @min(n_in, 512);
    var prng = std.Random.Xoshiro256.init(0xE4A);
    const rnd = prng.random();
    for (x[0..k_use]) |*v| v.* = rnd.float(f32) * 2 - 1;

    const got = gemv_mod.dotScalar(.q6_k, row, x[0..k_use]);
    var dq: [512]f32 = undefined;
    gguf_mod.dequantBlock(.q6_k, row, dq[0..k_use], k_use);
    var want: f32 = 0;
    for (dq[0..k_use], x[0..k_use]) |dv, xv| want += dv * xv;
    const diff = @abs(want - got);
    try std.testing.expect(diff / @max(@abs(want), 1e-9) < 1e-4);
    std.debug.print("[test] gemma-4: dot fila REAL Q6_K OK ({d:.2})\n", .{got});
}

fn preadAll(fd: c_int, buf: []u8, offset: u64) !usize {
    if (comptime builtin.target.os.tag == .windows) return error.ShortRead;
    var total: usize = 0;
    while (total < buf.len) {
        const r = pread(fd, buf.ptr + total, buf.len - total, @intCast(offset + total));
        if (r <= 0) return error.ShortRead;
        total += @intCast(r);
    }
    return total;
}
