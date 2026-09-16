//! Presupuesto de recursos del host — fuente única de verdad (post-freeze
//! 2026-09-13: varios lanes lanzaron pools de hilos a la vez y el host se
//! congeló por oversubscription; cada módulo hacia su propio sizing con
//! `std.Thread.getCpuCount()` = lógicos, sin mirar SMT ni load ni memoria).
//!
//! Módulo puro de lectura (sysfs/procfs + env knobs), sin estado compartido
//! ni threads: seguro de llamar desde cualquier sitio, incluidos tests.
//!
//! API central:
//!   - `physicalCoreCpus(allocator)` — un representante por núcleo físico
//!     (colapso SMT por thread_siblings_list, restringido a la afinidad
//!     del proceso). Fuente: cpu_executor.zig (lane-f F2); MOVIDA aquí
//!     para que TODOS los pools la usen. cpu_executor re-exporta.
//!   - `resolveThreadsAndAffinity(allocator, requested)` — pedido N hilos
//!     repartidos primero físicos, luego SMT siblings (patrón FreeToken).
//!   - `ambientLoadAvg1m()` — load medio 1-min; −1 si no legible.
//!   - `hostMemAvailableBytes()` — MemAvailable de /proc/meminfo; 0 si
//!     no legible. Fuente: inference/cli.zig (eager-load guard).
//!   - `availableCores(allocator)` — presupuesto COMPUTE: min(físicos de
//!     la afinidad, ajustado por load externo y por ZIG_AI_CPU_WORKERS).
//!   - `cpuBudgetFor(tag)` — helpers de sizing por consumidor con
//!     breadcrumb `[cpu_budget]` único (DEBUG_LEVEL≥1): hace visible la
//!     oversubscription en una línea por pool creado.
//!
//! Knobs (jerarquía: env explícito > condiciones del host):
//!   - `ZIG_AI_CPU_WORKERS=N` — cap GLOBAL de threads computantes (antes
//!     solo lo veía el executor MoE; ahora matmul/vision/engine también).
//!     N=0 = automático. Default automático: n_físicos−1 (reserva 1 core
//!     para el coordinador/interactivo), nunca <1.
//!   - En tests (`builtin.is_test`): cap extra ≤4 workers — la suite lanza
//!     varios binarios a la vez y cada uno creaba un pool de 16.
const std = @import("std");
const builtin = @import("builtin");
const debugz = @import("debug");

// ============================================================================
// Syscalls libc ya enlazada en todo el árbol (sin dependencias nuevas).
// ============================================================================

extern "c" fn open(path: [*:0]const u8, flags: i32) i32;
extern "c" fn read(fd: i32, buf: [*]u8, count: usize) isize;
extern "c" fn close(fd: i32) i32;
extern "c" fn sched_getaffinity(pid: c_int, cpusetsize: usize, mask: *CpuSet) c_int;
extern "c" fn sched_setaffinity(pid: c_int, cpusetsize: usize, mask: *const CpuSet) c_int;

const O_RDONLY: c_int = 0;

/// Máscara de 1024 CPUs (cpu_set_t de glibc).
pub const CpuSet = extern struct {
    bits: [16]u64,

    pub fn contains(self: *const CpuSet, cpu: u32) bool {
        if (cpu >= 1024) return false;
        return (self.bits[cpu / 64] >> @intCast(cpu % 64)) & 1 != 0;
    }
    pub fn add(self: *CpuSet, cpu: u32) void {
        if (cpu >= 1024) return;
        self.bits[cpu / 64] |= @as(u64, 1) << @intCast(cpu % 64);
    }
};

/// Lee un fichero pequeño completo (sysfs/procfs); null si no existe/vacío.
fn readSmallFile(path: [:0]const u8, buf: []u8) ?[]const u8 {
    const fd = open(path.ptr, O_RDONLY);
    if (fd < 0) return null;
    defer _ = close(fd);
    var total: usize = 0;
    while (total < buf.len) {
        const n = read(fd, buf.ptr + total, buf.len - total);
        if (n <= 0) break;
        total += @intCast(n);
    }
    if (total == 0) return null;
    return std.mem.trim(u8, buf[0..total], " \n\t\r");
}

/// Máscara de afinidad del proceso actual; null si el SO no la reporta.
pub fn currentAffinity() ?CpuSet {
    var set: CpuSet = std.mem.zeroes(CpuSet);
    const rc = sched_getaffinity(0, @sizeOf(CpuSet), &set);
    if (rc < 0) return null;
    return set;
}

/// Pin del hilo actual a una CPU lógica (syscall sobre el hilo llamante).
pub fn pinToCpu(cpu: u32) bool {
    if (cpu >= 1024) return false;
    var set: CpuSet = std.mem.zeroes(CpuSet);
    set.add(cpu);
    return sched_setaffinity(0, @sizeOf(CpuSet), &set) == 0;
}

// ============================================================================
// Topología: un representante por núcleo físico (colapso SMT).
// ============================================================================

/// Un CPU lógico por núcleo físico, restringido a la afinidad del proceso.
/// Dedup por contenido EXACTO de thread_siblings_list (clave canónica del
/// kernel; cubre formatos "0-1", "0", "0,2"). Fallback: lista completa si
/// sysfs no está disponible (contenedor/host sin topología).
pub fn physicalCoreCpus(allocator: std.mem.Allocator) ![]u32 {
    var reps: std.ArrayList(u32) = .empty;
    defer reps.deinit(allocator);
    var keys: std.ArrayList([]const u8) = .empty;
    defer {
        for (keys.items) |k| allocator.free(k);
        keys.deinit(allocator);
    }

    const logical = std.Thread.getCpuCount() catch 1;
    const upper: u32 = @intCast(@min(@max(logical, 1), 1024));
    const allowed = currentAffinity();

    var path_buf: [96]u8 = undefined;
    var key_buf: [128]u8 = undefined;

    var cpu: u32 = 0;
    while (cpu < upper) : (cpu += 1) {
        if (allowed) |a| {
            if (!a.contains(cpu)) continue;
        }
        const path = std.fmt.bufPrintZ(&path_buf, "/sys/devices/system/cpu/cpu{d}/topology/thread_siblings_list", .{cpu}) catch unreachable;
        const key = readSmallFile(path, &key_buf) orelse {
            // Sin sysfs para este cpu: representante de grupo propio.
            try reps.append(allocator, cpu);
            continue;
        };
        var dup = false;
        for (keys.items) |k| {
            if (std.mem.eql(u8, k, key)) {
                dup = true;
                break;
            }
        }
        if (dup) continue;
        try keys.append(allocator, try allocator.dupe(u8, key));
        try reps.append(allocator, cpu);
    }

    if (reps.items.len == 0) {
        // Última línea de defensa: todos los lógicos permitidos.
        var cpu2: u32 = 0;
        while (cpu2 < upper) : (cpu2 += 1) {
            if (allowed) |a| {
                if (a.contains(cpu2)) try reps.append(allocator, cpu2);
            } else try reps.append(allocator, cpu2);
        }
    }
    return allocator.dupe(u32, reps.items);
}

/// FreeToken resolve_threads_and_affinity: requested==0 → un hilo por núcleo
/// físico; N>0 → honra N CPUs repartiendo primero físicos y luego lógicos
/// restantes (nunca dos hilos en el mismo hermano SMT antes de agotar la
/// lista). Devuelve los core_ids (uno por hilo del pool TOTAL incluido
/// coordinador).
pub fn resolveThreadsAndAffinity(allocator: std.mem.Allocator, requested: usize) ![]u32 {
    const reps = try physicalCoreCpus(allocator);
    defer allocator.free(reps);
    if (requested == 0) return allocator.dupe(u32, reps);

    const allowed = currentAffinity();
    var order: std.ArrayList(u32) = .empty;
    defer order.deinit(allocator);
    try order.appendSlice(allocator, reps);
    const upper: u32 = @intCast(@min(std.Thread.getCpuCount() catch 1, 1024));
    var cpu: u32 = 0;
    while (cpu < upper) : (cpu += 1) {
        if (allowed) |a| {
            if (!a.contains(cpu)) continue;
        }
        var is_rep = false;
        for (reps) |r| {
            if (r == cpu) {
                is_rep = true;
                break;
            }
        }
        if (!is_rep) try order.append(allocator, cpu);
    }
    if (order.items.len == 0) try order.append(allocator, 0);

    const ids = try allocator.alloc(u32, requested);
    for (ids, 0..) |*id, i| id.* = order.items[i % order.items.len];
    return ids;
}

// ============================================================================
// Load y memoria del host.
// ============================================================================

/// Load medio 1-min del host (para gates de asserts en benchmarks); −1 si
/// no disponible.
pub fn ambientLoadAvg1m() f32 {
    var buf: [64]u8 = undefined;
    const s = readSmallFile("/proc/loadavg", &buf) orelse return -1.0;
    const sp1 = std.mem.indexOfAny(u8, s, " ") orelse return -1.0;
    return std.fmt.parseFloat(f32, s[0..sp1]) catch -1.0;
}

/// RAM disponible en el host leyendo /proc/meminfo (0 si no se puede leer).
/// Evita morir por swap: la carga eager de un modelo grande dequantiza TODO
/// a f32 en el host y el OOM killer congela la máquina entera.
pub fn hostMemAvailableBytes() usize {
    var buf: [4096]u8 = undefined;
    const s = readSmallFile("/proc/meminfo", &buf) orelse return 0;
    var it = std.mem.splitScalar(u8, s, '\n');
    while (it.next()) |line| {
        if (std.mem.startsWith(u8, line, "MemAvailable:")) {
            const rest = std.mem.trim(u8, line["MemAvailable:".len..], " \t");
            const sp = std.mem.indexOfScalar(u8, rest, ' ') orelse return 0;
            const kb = std.fmt.parseInt(usize, rest[0..sp], 10) catch return 0;
            return kb * 1024;
        }
    }
    return 0;
}

// ============================================================================
// Presupuesto COMPUTE — sizing adaptativo para cualquier pool de threads.
// ============================================================================

/// Cap de threads computantes que consume el host sin saturarlo.
///
/// Composición (de más a menos autoridad):
///   1. `ZIG_AI_CPU_WORKERS=N` (N>0): explícito del usuario/dev — se honra.
///   2. Tests (`builtin.is_test`): ≤4 — la suite corre VARIOS binarios en
///      paralelo y cada uno es un proceso completo del engine.
///   3. Automático: núcleos FÍSICOS de la afinidad − 1 (coordinador/
///      interactivo), con descuento proporcional si el load externo
///      (loadavg − lo nuestro) ya consume el host: cada core ocupado por
///      terceros resta 1 del budget (floor 1).
///
/// Nunca devuelve 0: un pool sin hilos rompe contratos existentes.
/// Emite el breadcrumb `[cpu_budget]` UNA vez por proceso (el primer
/// consumidor) — ver `logCpuBudget`.
pub fn computeThreadBudget(allocator: std.mem.Allocator, consumer: []const u8) usize {
    const phys = physicalCoreCpus(allocator) catch {
        // Sin topología legible: lógicos − 1, floor 1 (mejor que fallar).
        const logical = std.Thread.getCpuCount() catch 1;
        const b = @max(1, logical - 1);
        logCpuBudget(consumer, b, null, logical);
        return b;
    };
    defer allocator.free(phys);
    const n_phys: usize = @max(1, phys.len);

    var budget: usize = undefined;
    blk: {
        // 1. Knob global explícito.
        if (cpuWorkersEnv()) |n| {
            if (n > 0) {
                budget = n;
                break :blk;
            }
        }

        // 2. Tests: cap duro — paralelismo de la SUITE, no del binario.
        if (builtin.is_test) {
            budget = @min(n_phys, 4);
            break :blk;
        }

        // 3. Auto: físicos − 1, menos lo que ya consumen terceros.
        var b = n_phys - 1;
        const load = ambientLoadAvg1m();
        if (load >= 0) {
            // Load externo ≈ cores ocupados por otros. Descuento 1:1, floor 1.
            // (loadavg incluye NUESTROS threads en marcha; el pool aún no
            // existe al llamar esto, así que en práctica mide ruido externo.)
            const external = @as(usize, @intFromFloat(@max(0, load)));
            b = b -| @min(external, b - 1);
        }
        budget = @max(1, b);
    }
    logCpuBudget(consumer, budget, phys, n_phys);
    return budget;
}

/// Presupuesto específico para el test runner / jobs de test.
///
/// Diferencias con `computeThreadBudget`:
/// - NUNCA usa el knob `ZIG_AI_CPU_WORKERS` (los tests deben ser
///   deterministas independientes del entorno).
/// - Aplica un cap duro de 4 workers o el número de núcleos físicos,
///   lo que sea menor.
/// - Emite el breadcrumb `[cpu_budget]` con tag `test_runner`.
pub fn testRunnerBudget(allocator: std.mem.Allocator) usize {
    const phys = physicalCoreCpus(allocator) catch {
        const logical = std.Thread.getCpuCount() catch 1;
        const b = @min(logical, 4);
        logCpuBudget("test_runner", b, null, logical);
        return b;
    };
    defer allocator.free(phys);
    const n_phys: usize = @max(1, phys.len);
    const budget = @min(n_phys, 4);
    logCpuBudget("test_runner", budget, phys, n_phys);
    return budget;
}

/// `ZIG_AI_CPU_WORKERS` parseado una vez por llamada (barato: getenv).
/// N>0 = cap explícito; 0/ausente = automático. Era knob solo-MoE; ahora
/// es GLOBAL (matmul parallel, vision, engine, moe comparten el budget).
pub fn cpuWorkersEnv() ?usize {
    const v = std.c.getenv("ZIG_AI_CPU_WORKERS") orelse return null;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch null;
}

/// Breadcrumb de presupuesto — UNA vez por PROCESO (el engine crea un
/// MatmulEngine por capa y el camino híbrido crea pools en varios puntos;
/// el primer log basta, el resto sería spam idéntico). Tag [cpu_budget],
/// DEBUG_LEVEL≥1. Hace la oversubscription visible: si dos lanes ven
/// "workers=15" en el mismo loadavg, el bug salta en el log compartido.
var budget_logged_once: bool = false;
fn logCpuBudget(consumer: []const u8, workers: usize, phys: ?[]const u32, n_phys_hint: usize) void {
    if (budget_logged_once and !builtin.is_test) return;
    budget_logged_once = true;
    const n_phys: usize = if (phys) |p| p.len else n_phys_hint;
    const mem_mb = hostMemAvailableBytes() / (1024 * 1024);
    debugz.dbg.printLevel(.info, "[cpu_budget] consumer={s} workers={d} phys={d} load={d:.1} mem_free={d}MB test={}\n", .{
        consumer, workers, n_phys, ambientLoadAvg1m(), mem_mb, builtin.is_test,
    });
}

/// Log explícito para tests/diagnóstico manual (sin dedup de proceso).
pub fn logCpuBudgetForced(consumer: []const u8, workers: usize, allocator: std.mem.Allocator) void {
    const phys = physicalCoreCpus(allocator) catch null;
    defer if (phys) |p| allocator.free(p);
    const n_phys: usize = if (phys) |p| p.len else 0;
    const mem_mb = hostMemAvailableBytes() / (1024 * 1024);
    debugz.dbg.printLevel(.info, "[cpu_budget] consumer={s} workers={d} phys={d} load={d:.1} mem_free={d}MB test={}\n", .{
        consumer, workers, n_phys, ambientLoadAvg1m(), mem_mb, builtin.is_test,
    });
}

test "physicalCoreCpus devuelve representantes sin duplicar grupos" {
    const a = std.testing.allocator;
    const reps = try physicalCoreCpus(a);
    defer a.free(reps);
    try std.testing.expect(reps.len >= 1);
    try std.testing.expect(reps.len <= 1024);
    for (reps, 0..) |r, i| {
        if (i > 0) try std.testing.expect(r > reps[i - 1]); // orden ascendente
    }
    // Si sysfs expone topología, el colapso SMT debe haber agrupado:
    // representantes ≤ mitades redondeadas de lógicos.
    const logical = std.Thread.getCpuCount() catch 1;
    var probe: [8]u8 = undefined;
    const sysfs_ok = readSmallFile("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list", &probe) != null;
    if (sysfs_ok and logical >= 2) {
        try std.testing.expect(reps.len * 2 >= logical); // sanidad inversa
        try std.testing.expect(reps.len <= (logical + 1) / 2);
        std.debug.print("resources topology: {d} físicos de {d} lógicos\n", .{ reps.len, logical });
    }
}

test "computeThreadBudget nunca es 0 ni excede físicos (sin knob)" {
    const a = std.testing.allocator;
    const budget = computeThreadBudget(a);
    try std.testing.expect(budget >= 1);
    const phys = try physicalCoreCpus(a);
    defer a.free(phys);
    // En test: cap ≤4. Sin test pero sin knob: ≤ n_físicos.
    const ceiling: usize = if (builtin.is_test) @min(phys.len, 4) else phys.len;
    try std.testing.expect(budget <= @max(1, ceiling));
}

test "hostMemAvailableBytes plausible (>0 en Linux con /proc)" {
    const avail = hostMemAvailableBytes();
    // Host vivo: al menos 64MB disponibles (0 solo sin /proc/meminfo).
    if (avail == 0) return; // entorno sin /proc — no fail
    try std.testing.expect(avail >= 64 * 1024 * 1024);
}

test "ambientLoadAvg1m plausible" {
    const load = ambientLoadAvg1m();
    if (load < 0) return; // entorno sin /proc/loadavg
    try std.testing.expect(load >= 0);
    try std.testing.expect(load < 4096);
}
