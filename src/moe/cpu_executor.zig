//! Executor CPU para expertos MoE desbordados (Lane F).
//!
//! F2 — pool de workers: un hilo por NÚCLEO FÍSICO (colapso SMT leyendo
//! /sys/devices/system/cpu/cpu*/topology/thread_siblings_list, patrón
//! FreeToken `physical_core_cpus`), pin por worker vía sched_setaffinity
//! (extern libc ya enlazada), barrera por contadores atómicos con backoff,
//! y UN hilo reservado como coordinador (en F3 poll-ea los flags CUDA del
//! handshake; aquí solo despacha el lote GEMV). Los hermanos SMT sólo
//! disputan puertos de carga sin ancho de banda ⇒ se excluyen por
//! construcción, no por tuning.
//!
//! Determinismo híbrido: reparto estático contiguo por FILAS completas
//! (cada fila la computa un único worker con sus cadenas propias) ⇒ salida
//! independiente del scheduling y del nº de workers.
//!
//! Knobs: ZIG_AI_CPU_WORKERS=N (N workers computantes; default 0 =
//! automático: n_físicos − 1 workers + coordinador).
const builtin = @import("builtin");
const std = @import("std");
const debugz = @import("debug");
const gemv_mod = @import("moe_cpu_gemv");
pub const gemv_pub = gemv_mod;
const ext_sync = @import("cudaz_ext_sync");
const ext_mem = @import("cudaz_ext_mem");
// Post-freeze 2026-09-13: la topología/afinidad/loadavg VIVEN en
// `utils/resources.zig` (presupuesto CPU central); re-export para no
// romper la API histórica de este módulo (tests vía moe_cpu_executor).
const resources = @import("resources");
pub const CpuSet = resources.CpuSet;
pub const currentAffinity = resources.currentAffinity;
pub const pinToCpu = resources.pinToCpu;
pub const physicalCoreCpus = resources.physicalCoreCpus;
pub const resolveThreadsAndAffinity = resources.resolveThreadsAndAffinity;
pub const ambientLoadAvg1m = resources.ambientLoadAvg1m;
pub const hostMemAvailableBytes = resources.hostMemAvailableBytes;

// ============================================================================
// Syscalls libc ya enlazada (sin dependencias nuevas): sysfs + affinity.
// ============================================================================

extern "c" fn open(path: [*:0]const u8, flags: i32) i32;
extern "c" fn read(fd: i32, buf: [*]u8, count: usize) isize;
extern "c" fn write(fd: i32, buf: [*]const u8, count: usize) isize;
extern "c" fn close(fd: i32) i32;
extern "c" fn mmap(
    addr: ?*const anyopaque,
    length: usize,
    prot: i32,
    flags: i32,
    fd: i32,
    offset: i64,
) ?*anyopaque;
extern "c" fn munmap(addr: ?*const anyopaque, length: usize) i32;
extern "c" fn unlink(path: [*:0]const u8) i32;
extern "c" fn nanosleep(rqtp: *const std.c.timespec, rmtp: ?*std.c.timespec) c_int;
extern "kernel32" fn Sleep(dwMilliseconds: u32) callconv(.c) void;

const O_RDONLY: c_int = 0;

fn threadSleepUs(us: u64) void {
    if (comptime builtin.target.os.tag == .windows) {
        Sleep(@intCast(@max(1, us / 1000)));
        return;
    }
    const ts: std.c.timespec = .{
        .sec = @intCast(us / 1_000_000),
        .nsec = @intCast((us % 1_000_000) * 1000),
    };
    _ = nanosleep(&ts, null);
}

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

/// Lee un fichero completo a memoria asignada (sysfs/procfs/json pequeños).
fn readSmallFileAlloc(allocator: std.mem.Allocator, path: [:0]const u8, max_bytes: usize) ?[]u8 {
    const fd = open(path.ptr, O_RDONLY);
    if (fd < 0) return null;
    defer _ = close(fd);
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    var chunk: [4096]u8 = undefined;
    while (buf.items.len < max_bytes) {
        const n = read(fd, &chunk, @min(chunk.len, max_bytes - buf.items.len));
        if (n <= 0) break;
        buf.appendSlice(allocator, chunk[0..@intCast(n)]) catch return null;
    }
    if (buf.items.len == 0) return null;
    return buf.toOwnedSlice(allocator) catch null;
}

// Post-freeze 2026-09-13: CpuSet/currentAffinity/pinToCpu/physicalCoreCpus/
// resolveThreadsAndAffinity/ambientLoadAvg1m MOVIDOS a utils/resources.zig
// (re-exportados arriba). Los extern sched_* quedaban aquí para el pin de
// los workers; resources.zig declara los suyos propios.

// ============================================================================
// Executor: coordinador reservado + workers GEMV con barreras atómicas.
// ============================================================================

pub const Job = struct {
    fmt: gemv_mod.Format,
    w: []const u8,
    x: []const f32,
    out: []f32,
    n: usize, // columnas (elems por fila / k_dim)
    ids: []const i32 = &.{}, // vacío = filas contiguas (F2); con ids[t]=experto por token (Contrato 8)
    out_dim: usize = 0, // con ids: pares planos t*out_dim+d
};

// ============================================================================
// Contrato 8: submit/sync asíncrono con handshake de flags CUDA.
// ============================================================================

pub const Mode = enum {
    /// Camino lento documentado: staging host síncrono, sin CUDA.
    host_staging,
    /// Handshake por memops del front-end (sin SMs residentes) — solape pleno.
    cuda_memops,
    /// Sin memops: D2H asíncrono + notificación por cuLaunchHostFunc; el
    /// sync sigue siendo bloqueante en host ⇒ solape PARCIAL (solo copia),
    /// pero submit no stalla el hilo de decode.
    cuda_hostfunc,
};

// --- Breadcrumbs HYBRID_DEBUG (env leída AQUÍ: debug.zig es de otro lane) ---
var hybrid_debug_cached: ?bool = null;

/// Gate ZIG_AI_HYBRID_DEBUG=1: M misses, split elegido y latencias por
/// capa/paso. F fetched (device) llega vía readStats del lane-E.
pub fn hybridDebugEnabled() bool {
    if (hybrid_debug_cached) |v| return v;
    const v = if (std.c.getenv("ZIG_AI_HYBRID_DEBUG")) |p|
        std.mem.eql(u8, std.mem.span(p), "1")
    else
        false;
    hybrid_debug_cached = v;
    return v;
}

fn hdLog(comptime fmt: []const u8, args: anytype) void {
    if (hybridDebugEnabled()) debugz.dbg.print("[hybrid] " ++ fmt, args);
}

// Canary: probe both ext_sync.hostAlloc(64) and ext_mem.hostRegister at
// startup. Sets a process-global flag (one probe ever) so attachStream
// and the rest of the executor know whether to try the cuda_memops path
// or fall back to host_staging. Covers both ERROR_NOT_INITIALIZED (3)
// and ERROR_INVALID_VALUE (1) — any canary error ⇒ fallback. Gated
// breadcrumb on ZIG_AI_HYBRID_DEBUG=1.
var g_memops_canary: ?bool = null;

/// Envenena el canary tras un fallo del hostRegister de PRODUCCIÓN (p.ej.
/// el mmap real del GGUF en MoeLayer.init): las capas/layers posteriores
/// saltan directo al fallback sin reintentar. Un probe sintético puede
/// pasar mientras la operación real falla — la fuente de verdad es la
/// llamada real, y esto la propaga como estado cacheado.
pub fn poisonMemopsCanary() void {
    g_memops_canary = false;
}

pub fn memopsCanary() bool {
    if (g_memops_canary) |v| return v;
    // Probe 1: hostAlloc (pinned staging).
    const a = ext_sync.hostAlloc(64);
    const alloc_ok = a != null;
    if (a) |buf| ext_sync.hostFree(buf);
    // Probe 2: hostRegister on a small FILE-BACKED mmap. The production
    // path (MoeLayer.init / fromFileMmapWhole) registers the mmap of the
    // GGUF file, which is file-backed. On this host anonymous buffers
    // register fine but file-backed mappings fail with ERROR_INVALID_VALUE.
    var mmap_ok = false;
    var reg_ok = false;
    const tmp_path = "/tmp/cpu_executor_canary_probe.bin";
    const fd = open(@ptrCast(tmp_path), 0x242); // O_RDWR|O_CREAT|O_TRUNC
    if (fd >= 0) {
        // Probe at the production size class: 256 MiB sparse file.
        // The driver fails on large file-backed mappings (ERROR_INVALID_VALUE)
        // even though small ones register fine. Sparse alloc = virtual memory
        // only (no physical pages committed until touched).
        const size: usize = 256 * 1024 * 1024;
        var page_buf: [4096]u8 = undefined;
        @memset(&page_buf, 0);
        var written: usize = 0;
        while (written < size) : (written += page_buf.len) {
            _ = write(fd, &page_buf, page_buf.len);
        }
        const ptr = mmap(null, size, 0x1 | 0x2, 0x02, fd, 0); // PROT_READ|PROT_WRITE, MAP_PRIVATE
        if (ptr) |p| {
            if (@intFromPtr(p) != @as(usize, @bitCast(@as(isize, -1)))) {
                mmap_ok = true;
                if (ext_mem.hostRegister(@ptrCast(@alignCast(p)), size, ext_mem.CU_MEMHOSTREGISTER_DEVICEMAP)) |_| {
                    reg_ok = true;
                    ext_mem.hostUnregister(@ptrCast(@alignCast(p))) catch {};
                } else |_| {}
                _ = munmap(p, size);
            }
        }
        _ = close(fd);
        _ = unlink(@ptrCast(tmp_path));
    }
    const ok = alloc_ok and mmap_ok and reg_ok;
    g_memops_canary = ok;
    if (hybridDebugEnabled()) {
        debugz.dbg.printLevel(
            .info,
            "[hybrid] canary: hostAlloc={s} file_mmap={s} hostRegister={s} → overall={s}\n",
            .{
                if (alloc_ok) "ok" else "err",
                if (mmap_ok) "ok" else "err",
                if (reg_ok) "ok" else "err",
                if (ok) "ok" else "err",
            },
        );
    }
    return ok;
}

pub const Pending = struct {
    slot: u8,
    layer_id: u32,
    seq: u64,
};

pub const StState = enum(u8) { empty = 0, staged = 1, computing = 2, done = 3 };

/// Slots acotados estilo FreeToken (_FLAG_SLOTS_PER_LAYER): estado por
/// (capa en vuelo). El staging es válido hasta el próximo submit del MISMO
/// slot (el merge de E debe consumirlo antes).
pub const max_slots: usize = 16;

const Slot = struct {
    state: std.atomic.Value(u8) = .init(@intFromEnum(StState.empty)),
    claim: std.atomic.Value(bool) = .init(false), // reserva caller↔reclaim
    layer_id: u32 = 0,
    seq: u64 = 0,
    posted_ms: i64 = 0,
    pinned: bool = false,

    fmt: gemv_mod.Format = .q8_0,
    w_ref: []const u8 = &.{},
    x: []f32 = &.{},
    ids: []i32 = &.{},
    out: []f32 = &.{},

    ready_flag: usize = 0, // dirección del flag READY (cuda; host-pineada)
    done_flag: usize = 0, // dirección del flag DONE (cuda)
    flags_buf: ?[]align(64) u8 = null, // buffer pineado que contiene ambos flags
    /// Señal READY para modo hostfunc (el callback del driver escribe aquí).
    host_ready_seq: std.atomic.Value(u64) = .init(0),
};

/// Callback C-ABI del driver (hilo del driver: SOLO store atómico, jamás
/// bloquear). Marca la secuencia como lista para el coordinador.
fn hostReadyCallback(raw: ?*anyopaque) callconv(.c) void {
    const slot: *Slot = @ptrCast(@alignCast(raw.?));
    slot.host_ready_seq.store(slot.seq, .release);
}

pub const Config = struct {
    workers_requested: usize = 0,
    watchdog_ms: i64 = 5000,
    fmt: gemv_mod.Format = .q8_0,
    /// Geometría de la primitiva expert-proyección (obligatoria para submit):
    /// W empaquetada [n_experts][out_dim][rb] fila-major; x staging [T][k_dim].
    k_dim: usize = 0,
    out_dim: usize = 0,
    n_experts: usize = 0,
};

fn nowMs() i64 {
    return @intCast(@divTrunc(@import("time").Timer.now(), std.time.ns_per_s * 1000));
}

/// Wrappers de staging pinned para diagnósticos externos (tests/bench).
pub fn hostAllocPub(bytes: usize) ?[]align(64) u8 {
    return ext_sync.hostAlloc(bytes);
}
pub fn hostFreePub(buf: []align(64) u8) void {
    ext_sync.hostFree(buf);
}

/// Reloj monotónico ms expuesto para tests (inyección de edades).
pub const nowMsPublic = nowMs;

pub const Executor = struct {
    allocator: std.mem.Allocator,
    core_ids_total: []u32, // [0]=coordinador, [1..]=workers (pin 1:1)
    n_workers: u32,
    threads: []std.Thread, // incluye coordinador en [0]

    job: Job = undefined,
    pending_seq: std.atomic.Value(u64) = .init(0), // caller → coord
    dispatch_seq: std.atomic.Value(u64) = .init(0), // coord → workers
    done_workers: std.atomic.Value(u32) = .init(0),
    completed_seq: std.atomic.Value(u64) = .init(0), // coord → caller
    shutdown: std.atomic.Value(bool) = .init(false),

    // --- Contrato 8 ---
    cfg: Config = .{},
    mode: Mode = .host_staging,
    stream_handle: usize = 0,
    slots: [max_slots]Slot = [_]Slot{.{}} ** max_slots,
    wedged: std.atomic.Value(bool) = .init(false),
    wd_thread: ?std.Thread = null,

    /// requested==0 → auto (físicos−1 workers + coordinador).
    pub fn init(allocator: std.mem.Allocator, requested: usize) !*Executor {
        const env_requested: usize = blk: {
            const v = std.c.getenv("ZIG_AI_CPU_WORKERS") orelse break :blk requested;
            const n = std.fmt.parseInt(usize, std.mem.span(v), 10) catch break :blk requested;
            break :blk n;
        };

        // Pool total = requested + coordinador (auto: n_phys total).
        const total_ids = try resolveThreadsAndAffinity(allocator, if (env_requested == 0) 0 else env_requested + 1);
        errdefer allocator.free(total_ids);

        const self = try allocator.create(Executor);
        errdefer allocator.destroy(self);
        const n_workers: u32 = @intCast(total_ids.len - 1);

        self.* = .{
            .allocator = allocator,
            .core_ids_total = total_ids,
            .n_workers = n_workers,
            .threads = try allocator.alloc(std.Thread, total_ids.len),
        };
        errdefer allocator.free(self.threads);

        // Coordinador (hilo reservado, NO computa).
        self.threads[0] = try std.Thread.spawn(.{}, coordinatorMain, .{self});

        // Workers computantes (cada uno se pinea a sí mismo al arrancar).
        for (0..n_workers) |i| {
            self.threads[1 + i] = try std.Thread.spawn(.{}, workerMain, .{ self, @as(u32, @intCast(i)) });
        }

        debugz.dbg.printLevel(.info, "[cpu_executor] nucleos_fisicos={d} workers={d} cores={any}\n", .{
            total_ids.len, n_workers, total_ids,
        });
        return self;
    }

    /// Init completo (Contrato 8): pool + watchdog thread + geometría.
    pub fn initFull(allocator: std.mem.Allocator, cfg: Config) !*Executor {
        const self = try Executor.init(allocator, cfg.workers_requested);
        self.cfg = cfg;
        self.wd_thread = std.Thread.spawn(.{}, watchdogMain, .{self}) catch null;
        return self;
    }

    /// Activa el modo cuda_memops si el driver soporta las memops (canary
    /// probe at startup); si no, queda host_staging (camino lento documentado)
    /// y lo reporta. El canary reemplaza la bandera `ext_sync.memopsAvailable()`
    /// (que informaba incorrectamente en hosts con driver cargado pero sin
    /// memops funcionales). ZIG_AI_HYBRID_DEBUG=1 muestra el resultado.
    pub fn attachStream(self: *Executor, stream_handle: usize) Mode {
        self.stream_handle = stream_handle;
        // Contrato: handle NULL (0) no puede servir memops/hostfunc — la
        // construcción @ptrFromInt(0) en submit/sync sería UB. El canary es
        // un fast-path cache; la fuente de verdad es la operación real
        // (poisonMemopsCanary la propaga tras un fallo de producción).
        if (stream_handle == 0 or !memopsCanary()) {
            self.mode = .host_staging;
            debugz.dbg.printLevel(.info, "[cpu_executor] modo=host_staging (handle={s}, canary={s})\n", .{
                if (stream_handle == 0) "null" else "válido",
                if (memopsCanary()) "ok" else "falló",
            });
            return self.mode;
        }
        self.mode = .cuda_memops;
        // Flags estables por slot (64B pineados c/u).
        for (&self.slots) |*s| {
            if (ext_sync.hostAlloc(64)) |buf| {
                @memset(buf, 0);
                s.ready_flag = @intFromPtr(buf.ptr);
                s.done_flag = @intFromPtr(buf.ptr + 8);
                s.pinned = false; // staging aún no asignado
                s.flags_buf = buf;
            } else break; // sin pinned → host path
        }
        debugz.dbg.printLevel(.info, "[cpu_executor] modo=cuda_memops stream=0x{x}\n", .{stream_handle});
        return self.mode;
    }

    pub fn deinit(self: *Executor) void {
        self.shutdown.store(true, .release);
        // Un bump despierta a coord y workers; ambos re-chequean shutdown.
        _ = self.dispatch_seq.fetchAdd(1, .release);
        for (self.threads) |t| t.join();
        if (self.wd_thread) |t| t.join();
        for (&self.slots) |*s| freeSlotStaging(self.allocator, s);
        if (self.mode == .cuda_memops) {
            for (&self.slots) |*s| {
                if (s.flags_buf) |b| ext_sync.hostFree(b);
                s.flags_buf = null;
            }
        }
        self.allocator.free(self.threads);
        self.allocator.free(self.core_ids_total);
        self.allocator.destroy(self);
    }

    /// GEMV paralelo síncrono (F2). Single-flight: un submit pendiente a la vez.
    pub fn gemvBlocking(self: *Executor, fmt: gemv_mod.Format, w: []const u8, n: usize, x: []const f32, out: []f32) void {
        std.debug.assert(self.completed_seq.load(.monotonic) == self.pending_seq.load(.monotonic)); // no re-entrancia
        self.job = .{ .fmt = fmt, .w = w, .x = x, .out = out, .n = n };
        const target = self.pending_seq.load(.monotonic) + 1;
        self.pending_seq.store(target, .release);
        while (self.completed_seq.load(.acquire) != target) {
            backoffCaller();
        }
    }

    // ------------------------------------------------------------------
    // Contrato 8 (F3): submit asíncrono + sync con handshake de flags.
    // Semántica de rutas: ids −1 → CPU con peso real; cada token se computa
    // EXACTAMENTE una vez; merge = suma de parciales (lado GPU de E).
    // ------------------------------------------------------------------

    /// Registra trabajo CPU para esta capa.
    /// ⚠️ Semántica de hidden_dev según modo:
    ///   • cuda_memops  → puntero DE device (D2H asíncrono).
    ///   • cuda_hostfunc / host_staging → puntero DE HOST (copia síncrona;
    ///     pasar un device ptr aquí es SEGFAULT por desreferencia).
    /// Los pesos `w` son bytes GGUF empaquetados reinterpretados como
    /// []const f32 (convención Contrato 8; longitud mínima validada).
    pub fn submit(self: *Executor, layer_id: u32, hidden_dev: usize, w: []const f32, ids_cpu: []const i32) !Pending {
        const k = self.cfg.k_dim;
        const od = self.cfg.out_dim;
        if (k == 0 or od == 0 or self.cfg.n_experts == 0) return error.InvalidGeometry;
        if (ids_cpu.len == 0) return error.EmptyIds;

        const fmt = self.cfg.fmt;
        const rb = fmt.rowBytes(k);
        const w_bytes_needed = self.cfg.n_experts * od * rb;
        if (w.len < @divTrunc(w_bytes_needed + 3, 4)) return error.WeightsTooSmall;

        // Reclamar slot libre (CAS de claim para evitar carrera con reclaim).
        // 4.1: los slots sólo llegan a empty vía sync()/drain() (reciclaje),
        // que liberan el staging ANTES de publicar empty ⇒ un slot claimable
        // está limpio por construcción (nada del job previo que pisar).
        var slot_ix: ?usize = null;
        for (&self.slots, 0..) |*s, i| {
            if (s.state.load(.monotonic) != @intFromEnum(StState.empty)) continue;
            if (!s.claim.swap(true, .acq_rel)) {
                slot_ix = i;
                break;
            }
        }
        const si = slot_ix orelse return error.SlotBusy;
        errdefer self.slots[si].claim.store(false, .release);
        const s = &self.slots[si];

        s.* = .{
            .state = s.state,
            .claim = s.claim,
            .layer_id = layer_id,
            .seq = s.seq + 1,
            .posted_ms = nowMs(),
            .fmt = fmt,
            .flags_buf = s.flags_buf,
            .ready_flag = s.ready_flag,
            .done_flag = s.done_flag,
        };

        const t_n = ids_cpu.len;
        // Staging: pinned en cuda-mode; heap normal en host-mode.
        if (self.mode == .cuda_memops) {
            s.pinned = true;
            s.x = blk: {
                const b = ext_sync.hostAlloc(t_n * k * 4) orelse break :blk null;
                break :blk @as([*]f32, @ptrCast(@alignCast(b.ptr)))[0 .. t_n * k];
            } orelse return error.MemopsUnavailable;
            errdefer ext_free(self.allocator, s.x, true);
            s.out = blk: {
                const b = ext_sync.hostAlloc(t_n * od * 4) orelse break :blk null;
                break :blk @as([*]f32, @ptrCast(@alignCast(b.ptr)))[0 .. t_n * od];
            } orelse return error.MemopsUnavailable;
            errdefer ext_free(self.allocator, s.out, true);
            s.ids = try self.allocator.dupe(i32, ids_cpu);
            s.w_ref = @as([*]const u8, @ptrCast(w.ptr))[0..w_bytes_needed];

            std.debug.assert(self.stream_handle != 0); // attachStream(null) fuerza host_staging
            const stream: ext_sync.Stream = @ptrFromInt(self.stream_handle);
            try ext_sync.dtohAsync(@intFromPtr(s.x.ptr), hidden_dev, t_n * k * 4, stream);
            if (self.mode == .cuda_memops) {
                try ext_sync.writeReady(stream, s.ready_flag, s.seq); // front-end, sin SMs
            } else {
                // hostfunc: el callback del driver marca host_ready_seq al
                // alcanzar su posición (tras la copia, mismo orden de stream).
                try ext_sync.launchHostFunc(stream, hostReadyCallback, s);
            }
        } else {
            s.pinned = false;
            s.x = try self.allocator.alloc(f32, t_n * k);
            errdefer self.allocator.free(s.x);
            s.out = try self.allocator.alloc(f32, t_n * od);
            errdefer self.allocator.free(s.out);
            s.ids = try self.allocator.dupe(i32, ids_cpu);
            s.w_ref = @as([*]const u8, @ptrCast(w.ptr))[0..w_bytes_needed];
            // Camino lento: copia host inmediata desde hidden_dev (host ptr).
            const src: [*]const f32 = @ptrFromInt(hidden_dev);
            @memcpy(s.x, src[0 .. t_n * k]);
            @memcpy(s.ids, ids_cpu);
        }
        debugz.dbg.printLevel(.detail, "[cpu_executor] submit capa={d} tokens={d} slot={d} seq={d} modo={s}\n", .{ layer_id, t_n, si, s.seq, @tagName(self.mode) });

        s.state.store(@intFromEnum(StState.staged), .release);
        return .{ .slot = @intCast(si), .layer_id = layer_id, .seq = s.seq };
    }

    /// Espera el parcial y lo devuelve listo para merge. En cuda-mode encola
    /// además el WaitValue64 DONE en el stream ⇒ los nodos capturados
    /// posteriores leen el staging ya publicado (orden garantizado).
    ///
    /// **Ownership contract (Contrato 8):** the returned `[]f32` slice is
    /// **transferred to the caller**, who MUST free it via
    /// `Executor.freePartial(self, part)` (NEVER with their own allocator —
    /// in cuda_memops mode the staging is pinned, and freeing with a foreign
    /// allocator would corrupt the heap). The slot is RECYCLED on handover
    /// (4.1): staging freed here, state→empty, claim released — the next
    /// `submit` gets a clean slot and cannot stomp a prior partial. A second
    /// `sync` on the same `Pending` returns `error.StalePending` (seq
    /// mismatched) — there is no aliasing of the returned buffer.
    pub fn sync(self: *Executor, p: Pending) ![]f32 {
        const s = &self.slots[p.slot];
        if (s.layer_id != p.layer_id or s.seq != p.seq) return error.StalePending;
        while (s.state.load(.acquire) != @intFromEnum(StState.done)) {
            if (self.shutdown.load(.acquire)) return error.ExecutorShutdown;
            if (nowMs() - s.posted_ms > self.cfg.watchdog_ms) {
                self.wedged.store(true, .release);
                debugz.dbg.printLevel(.info, "[cpu_executor] !! WATCHDOG: capa {d} wedged >{d}ms (slot {d})\n", .{ p.layer_id, self.cfg.watchdog_ms, p.slot });
                return error.WatchdogTimeout;
            }
            threadSleepUs(100);
        }
        // cuda-mode: ordenar la espera DONE en el stream ANTES de que E
        // capture nodos consumidores (idempotente respecto al coordinador).
        if (self.mode == .cuda_memops) {
            std.debug.assert(self.stream_handle != 0); // attachStream(null) fuerza host_staging
            const stream: ext_sync.Stream = @ptrFromInt(self.stream_handle);
            try ext_sync.waitDone(stream, s.done_flag, p.seq);
        }
        // Transfer ownership: caller frees the returned slice.
        const out = s.out;
        s.out = &.{}; // ya no es propiedad del slot (evita double-free en recycle)
        debugz.dbg.printLevel(.detail, "[cpu_executor] sync capa={d} ok ({d} tokens) — ownership transferred to caller\n", .{ p.layer_id, s.ids.len });
        self.recycleSlot(s);
        return out;
    }

    /// Recicla el slot a `empty` (4.1): libera el staging NO transferido y
    /// publica el estado ANTES de soltar el claim ⇒ quien gane el claim ve
    /// empty+limpio (sin job previo pisable). `x`/`ids`/`w_ref` son del job
    /// muerto; `out` ya fue entregado (o se libera aquí si nadie lo tomó).
    /// `layer_id` se invalida a sentinel ANTES del publish: un `sync`/`drain`
    /// tardío sobre el Pending viejo falla el guard de staleness al instante
    /// (StalePending) en vez de esperar un `done` que ya no existirá.
    const recycled_layer_sentinel: u32 = std.math.maxInt(u32);

    fn recycleSlot(self: *Executor, s: *Slot) void {
        freeSlotStaging(self.allocator, s);
        s.layer_id = recycled_layer_sentinel;
        s.state.store(@intFromEnum(StState.empty), .release);
        s.claim.store(false, .release);
    }

    /// Descarta un pending SIN consumir su resultado (4.1: par submit fallido
    /// — p.ej. `submit(up)` error tras `submit(gate)` OK en moe_layer). Bloquea
    /// hasta done (el coordinador ya computa el staged) y recicla; el staging
    /// `out` se libera aquí. Idempotente con un `sync` ya consumido
    /// (StalePending = nada que hacer).
    pub fn drainSlot(self: *Executor, p: Pending) void {
        const s = &self.slots[p.slot];
        if (s.layer_id != p.layer_id or s.seq != p.seq) return; // ya reciclado
        while (s.state.load(.acquire) != @intFromEnum(StState.done)) {
            if (self.shutdown.load(.acquire)) return;
            if (nowMs() - s.posted_ms > self.cfg.watchdog_ms) {
                self.wedged.store(true, .release);
                debugz.dbg.printLevel(.info, "[cpu_executor] !! WATCHDOG: drain capa={d} wedged >{d}ms (slot {d}) — recycle forzado\n", .{ p.layer_id, self.cfg.watchdog_ms, p.slot });
                break; // wedge: reciclar igual (staging no puede quedar eterno)
            }
            threadSleepUs(100);
        }
        debugz.dbg.printLevel(.detail, "[cpu_executor] drain capa={d} (slot {d}) descartado\n", .{ p.layer_id, p.slot });
        self.recycleSlot(s);
    }

    /// Free a partial returned by `sync()`. Uses the executor's internal
    /// allocator (and handles pinned staging if in cuda_memops mode).
    /// Callers (e.g. `moe_layer.zig` overflow path) MUST use this instead of
    /// their own allocator, because the partial was allocated with `self.allocator`.
    /// No-op on empty slices (defers releasing a failed handover).
    pub fn freePartial(self: *Executor, part: []f32) void {
        if (part.len == 0) return;
        if (self.mode == .cuda_memops) {
            ext_free(self.allocator, part, true);
        } else {
            self.allocator.free(part);
        }
    }

    fn publishSlotJob(self: *Executor, s: *Slot) void {
        s.state.store(@intFromEnum(StState.computing), .release);
        self.job = .{
            .fmt = s.fmt,
            .w = s.w_ref,
            .x = s.x,
            .out = s.out,
            .n = self.cfg.k_dim,
            .ids = s.ids,
            .out_dim = self.cfg.out_dim,
        };
        self.done_workers.store(0, .release);
        _ = self.dispatch_seq.fetchAdd(1, .release);
        var spin: usize = 0;
        while (self.done_workers.load(.acquire) != self.n_workers) {
            if (self.shutdown.load(.acquire)) return;
            backoffCoord(&spin);
        }
        if (self.mode == .cuda_memops and s.done_flag != 0) {
            ext_sync.hostPublishDone(s.done_flag, s.seq);
        }
        s.state.store(@intFromEnum(StState.done), .release);
    }
};

fn ext_free(allocator: std.mem.Allocator, buf: []f32, pinned: bool) void {
    if (!pinned) {
        allocator.free(buf);
        return;
    }
    const bytes = std.mem.sliceAsBytes(buf);
    ext_sync.hostFree(@alignCast(bytes));
}

/// Libera el staging de un slot (según cómo se asignó).
fn freeSlotStaging(allocator: std.mem.Allocator, s: *Slot) void {
    if (s.pinned) {
        if (s.x.len > 0) ext_free(allocator, s.x, true);
        if (s.out.len > 0) ext_free(allocator, s.out, true);
    } else {
        if (s.x.len > 0) allocator.free(s.x);
        if (s.out.len > 0) allocator.free(s.out);
    }
    if (s.ids.len > 0) allocator.free(s.ids);
    s.x = &.{};
    s.out = &.{};
    s.ids = &.{};
    s.w_ref = &.{};
}

/// Watchdog: pendings .staged más viejos que watchdog_ms → ruido + flag
/// (sync() devuelve WatchdogTimeout por su propio deadline; esto añade la
/// señal audible en background estilo FreeToken).
fn watchdogMain(self: *Executor) void {
    while (!self.shutdown.load(.acquire)) {
        threadSleepUs(50_000);
        const now = nowMs();
        for (&self.slots) |*s| {
            if (s.state.load(.acquire) != @intFromEnum(StState.staged)) continue;
            const age = now - s.posted_ms;
            if (age > self.cfg.watchdog_ms) {
                self.wedged.store(true, .release);
                debugz.dbg.printLevel(.info, "[cpu_executor] !! WATCHDOG bg: capa {d} staged hace {d}ms (>{d}ms)\n", .{ s.layer_id, age, self.cfg.watchdog_ms });
            }
        }
    }
}

fn pinInThread(core: u32) void {
    if (!pinToCpu(core)) {
        debugz.dbg.printLevel(.info, "[cpu_executor] pin cpu{d} rechazado\n", .{core});
    }
}

fn backoffCoord(iter: *usize) void {
    iter.* += 1;
    if (iter.* % 2048 == 0) threadSleepUs(20) else std.atomic.spinLoopHint();
}
fn backoffWorker(iter: *usize) void {
    iter.* += 1;
    if (iter.* % 4096 == 0) threadSleepUs(100) else std.atomic.spinLoopHint();
}
fn backoffCaller() void {
    var i: usize = 0;
    _ = &i;
    std.atomic.spinLoopHint();
}

fn coordinatorMain(self: *Executor) void {
    pinInThread(self.core_ids_total[0]);
    var seen: u64 = 0;
    var spin: usize = 0;
    while (true) {
        if (self.shutdown.load(.acquire)) return;

        // 1) Canal legacy F2 (gemvBlocking).
        const p = self.pending_seq.load(.acquire);
        if (p != seen) {
            spin = 0;
            seen = p;
            self.done_workers.store(0, .release);
            _ = self.dispatch_seq.fetchAdd(1, .release);
            while (self.done_workers.load(.acquire) != self.n_workers) {
                if (self.shutdown.load(.acquire)) return;
                backoffCoord(&spin);
            }
            self.completed_seq.store(seen, .release);
            continue;
        }

        // 2) Slots Contrato 8: staged → computar → done (+ flag GPU si cuda).
        var serviced = false;
        for (&self.slots) |*s| {
            if (s.state.load(.acquire) != @intFromEnum(StState.staged)) continue;
            if (self.mode == .cuda_memops and s.ready_flag != 0) {
                // Poll-ea del READY publicado por la memop del front-end.
                const rv: u64 = @as(*volatile u64, @ptrFromInt(s.ready_flag)).*;
                if (rv < s.seq) continue; // aún en vuelo en el stream
            }
            serviced = true;
            self.publishSlotJob(s);
        }
        if (!serviced) backoffCoord(&spin) else spin = 0;
    }
}

fn workerMain(self: *Executor, id: u32) void {
    pinInThread(self.core_ids_total[1 + id]);
    const my_core = self.core_ids_total[1 + id];
    debugz.dbg.printLevel(.detail, "[cpu_executor] worker{d} pinned cpu{d}\n", .{ id, my_core });
    var seen: u64 = 0;
    var spin: usize = 0;
    while (true) {
        const disp = self.dispatch_seq.load(.acquire);
        if (disp == seen) {
            if (self.shutdown.load(.acquire)) return;
            backoffWorker(&spin);
            continue;
        }
        spin = 0;
        seen = disp;
        if (self.shutdown.load(.acquire)) return;
        const jb = self.job;
        if (jb.ids.len == 0) {
            // F2: filas contiguas (chunk por worker).
            const rb = jb.fmt.rowBytes(jb.n);
            const rows = jb.out.len;
            const chunk = (rows + self.n_workers - 1) / self.n_workers;
            const base = id * chunk;
            if (base < rows) {
                const cnt = @min(chunk, rows - base);
                gemv_mod.gemv(jb.fmt, jb.w[base * rb ..][0 .. cnt * rb], jb.n, jb.x, jb.out[base..][0..cnt]);
            }
        } else {
            // Contrato 8: pares planos (token, out_dim): out[t*od+d] =
            // dot(W[ids[t]][d], x[t]) — cada par lo computa UN solo worker.
            const od = jb.out_dim;
            const pairs = jb.out.len;
            const chunk = (pairs + self.n_workers - 1) / self.n_workers;
            const base = id * chunk;
            if (base < pairs) {
                const cnt = @min(chunk, pairs - base);
                const rb = jb.fmt.rowBytes(jb.n);
                for (base..base + cnt) |pi| {
                    const t = pi / od;
                    const d = pi % od;
                    const e = jb.ids[t];
                    if (e < 0) continue; // defensivo: token no-CPU no se toca
                    const eu: usize = @intCast(e);
                    const row = jb.w[eu * od * rb + d * rb ..][0..rb];
                    jb.out[pi] = gemv_mod.dot(jb.fmt, row, jb.x[t * jb.n ..][0..jb.n]);
                }
            }
        }
        _ = self.done_workers.fetchAdd(1, .acq_rel);
    }
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
        debugz.dbg.print("[cpu_executor] topology: {d} físicos de {d} lógicos\n", .{ reps.len, logical });
    }
}

test "resolveThreadsAndAffinity honra conteo explicito" {
    const a = std.testing.allocator;
    const ids = try resolveThreadsAndAffinity(a, 3);
    defer a.free(ids);
    try std.testing.expectEqual(@as(usize, 3), ids.len);
    // Sin repetir CPU antes de agotar la lista (asumiendo ≥3 lógicas).
    const logical = std.Thread.getCpuCount() catch 1;
    if (logical >= 3) {
        for (0..ids.len) |i| {
            for (0..i) |j| try std.testing.expect(ids[i] != ids[j]);
        }
    }
}

// ============================================================================
// Contrato 6 — lector benchbw.json (D escribe, F consume). F4-prep.
// fetch_frac_q16 = round(pcie_ov/(pcie_ov+cpu_ov)·65536) clamp [0,65536];
// sin perfil válido → cap fijo 1 (default seguro).
// ============================================================================

pub const BenchBw = struct {
    version: u32,
    cpu_gbs: f64,
    pcie_gbs: f64,
    cpu_overlap_gbs: f64,
    pcie_overlap_gbs: f64,
    verdict: []const u8,
    /// El que D escribió (informativo); la AUTORIDAD es la derivación F.
    reported_fetch_frac_q16: ?u32,
};

/// Derivación pura (testable): null si entradas inválidas.
/// Rango del contrato [0,65536] INCLUYE 65536 ⇒ u32 (u16 no alcanza).
pub fn fetchFracQ16FromGbs(cpu_overlap_gbs: f64, pcie_overlap_gbs: f64) ?u32 {
    if (cpu_overlap_gbs < 0 or pcie_overlap_gbs < 0) return null;
    const denom = cpu_overlap_gbs + pcie_overlap_gbs;
    if (!(denom > 0)) return null;
    const q = pcie_overlap_gbs / denom;
    const scaled = @round(q * 65536.0);
    if (scaled <= 0) return 0;
    if (scaled >= 65536.0) return 65536;
    return @intFromFloat(scaled);
}

const benchbw_raw = struct {
    const Self = @This();
    version: u32 = 0,
    cpu_gbs: f64 = -1,
    pcie_gbs: f64 = -1,
    cpu_overlap_gbs: f64 = -1,
    pcie_overlap_gbs: f64 = -1,
    verdict: []const u8 = "",
    fetch_frac_q16: ?u32 = null,
};

/// Parse puro de texto JSON → perfil; null si corrupto/incompleto.
pub fn parseBenchBwText(allocator: std.mem.Allocator, text: []const u8) ?BenchBw {
    const parsed = std.json.parseFromSlice(benchbw_raw.Self, allocator, text, .{
        .ignore_unknown_fields = true,
    }) catch return null;
    defer parsed.deinit();
    const r = parsed.value;
    if (r.version != 1) return null;
    if (r.cpu_overlap_gbs < 0 or r.pcie_overlap_gbs < 0) return null;
    return .{
        .version = r.version,
        .cpu_gbs = r.cpu_gbs,
        .pcie_gbs = r.pcie_gbs,
        .cpu_overlap_gbs = r.cpu_overlap_gbs,
        .pcie_overlap_gbs = r.pcie_overlap_gbs,
        .verdict = r.verdict,
        .reported_fetch_frac_q16 = r.fetch_frac_q16,
    };
}

fn benchBwPath(buf: []u8, override: ?[]const u8) ?[]const u8 {
    if (override) |p| return p;
    if (std.c.getenv("ZIG_AI_BENCHBW_PATH")) |p| return std.mem.span(p);
    const xdg = std.c.getenv("XDG_CACHE_HOME") orelse {
        const home = std.c.getenv("HOME") orelse return null;
        return std.fmt.bufPrint(buf, "{s}/.cache/zig-ai/benchbw.json", .{std.mem.span(home)}) catch null;
    };
    return std.fmt.bufPrint(buf, "{s}/zig-ai/benchbw.json", .{std.mem.span(xdg)}) catch null;
}

/// Resuelve el fetch_frac_q16 efectivo: perfil válido → derivado del Contrato
/// 6; ausente/corrupto → cap fijo 1. `override_path` solo para tests.
pub fn resolveFetchFracQ16(allocator: std.mem.Allocator, override_path: ?[]const u8) u32 {
    var buf: [512]u8 = undefined;
    const path = benchBwPath(&buf, override_path) orelse {
        debugz.dbg.printLevel(.info, "[cpu_executor] benchbw: sin ruta → cap q16=1\n", .{});
        return 1;
    };
    var pathz: [512]u8 = undefined;
    const pz = std.fmt.bufPrintZ(&pathz, "{s}", .{path}) catch return 1;
    const text = readSmallFileAlloc(allocator, pz, 64 * 1024) orelse {
        debugz.dbg.printLevel(.info, "[cpu_executor] benchbw: {s} no legible → cap q16=1\n", .{path});
        return 1;
    };
    defer allocator.free(text);
    const bw = parseBenchBwText(allocator, text) orelse {
        debugz.dbg.printLevel(.info, "[cpu_executor] benchbw: corrupto → cap q16=1\n", .{});
        return 1;
    };
    const frac = fetchFracQ16FromGbs(bw.cpu_overlap_gbs, bw.pcie_overlap_gbs) orelse 1;
    debugz.dbg.printLevel(.info, "[cpu_executor] benchbw verdict={s} cpu_ov={d:.1} pcie_ov={d:.1} → q16={d}\n", .{ bw.verdict, bw.cpu_overlap_gbs, bw.pcie_overlap_gbs, frac });
    if (bw.reported_fetch_frac_q16) |rep| {
        const rep_i: i64 = @intCast(rep);
        const frac_i: i64 = @intCast(frac);
        const diff: u64 = @intCast(@abs(rep_i - frac_i));
        if (diff * 100 > 2 * @max(rep, frac)) {
            debugz.dbg.printLevel(.detail, "[cpu_executor] benchbw: reportado {d} difiere >2% del derivado {d}\n", .{ rep, frac });
        }
    }
    return frac;
}

/// Nº de expertos únicos en un conjunto de ids (para el breadcrumb split).
fn uniqueExperts(ids: []const i32) usize {
    var seen: [64]bool = [_]bool{false} ** 64;
    var count: usize = 0;
    for (ids) |e| {
        const u: usize = @intCast(@max(e, 0));
        const bucket = u % 64;
        if (!seen[bucket]) {
            seen[bucket] = true;
            count += 1;
        }
    }
    return count;
}
