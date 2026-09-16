//!
//! Proporciona un allocador de memoria host que puede registrar regiones como pinned
//! (para acceso zero-copy desde GPU), lockear en RAM (para acceso CPU sin paginación),
//! o dejar como página normal. Adaptado de la semántica FreeToken host_banks.py.
//! Incluye soporte para pin-after-fill (allocate PAGEABLE, fill, then pin as PINNED).
const builtin = @import("builtin");
const std = @import("std");
const debugz = @import("debug");
const cudaz = @import("cudaz");
const ext_mem = @import("cudaz_ext_mem");
const ext_sync = @import("cudaz_ext_sync");
extern "c" fn pread(fd: c_int, buf: [*]u8, count: usize, offset: i64) isize;
extern "c" fn open64(path: [*:0]const u8, flags: c_int, ...) c_int;
const ftw = @import("ftw");
const disk_tier = @import("disk_tier");
const os = std.posix;

/// Niveles de residenciá de memoria host.
pub const HostResidency = enum(u8) {
    PAGEABLE, // malloc normal, sin garantías de permanencia en RAM ni acceso directo GPU
    LOCKED, // mlock + RLIMIT_MEMLOCK: permanece en RAM, sin dirección de dispositivo
    PINNED, // cudaHostAlloc (o cudaHostRegister): accesible directamente por GPU (zero-copy)
    DISK_BACKED, // No residente en RAM; respaldado por DiskTier, se materializa bajo demanda.
};

pub const HostBank = struct {
    const Self = @This();

    /// Puntero a la región de memoria.
    ptr: ?*u8,
    /// Longitud de la región en bytes.
    len: usize,
    /// Tipo de residenciá de la región.
    residency: HostResidency = .PAGEABLE,
    /// Verdadero si nosotros poseemos la memoria y debemos liberarla en deinit.
    owned: bool = false,
    /// Verdadero si la memoria ha sido registrada como pinned (cudaHostRegister con DEVICEMAP).
    /// Solo relevante para residenciá PINNED.
    registered: bool = false,
    /// Verdadero si la memoria PINNED fue allocada con ext_sync.hostAlloc (no requiere unreg).
    pinned_via_hostalloc: bool = false,
    /// Allocator usado para alloc/allocAligned (PAGEABLE/LOCKED o pin-after-fill). Indefinido para no-alloc.
    allocator: std.mem.Allocator = undefined,
    /// Slice original devuelto por el allocator (para free correcto en deinit).
    orig_slice: ?[]u8 = null,
    /// Backend disk_tier para residency DISK_BACKED (MH-15).
    disk_tier: ?*disk_tier.DiskTier = null,
    /// Identificador de experto/tensor para DiskTier.
    disk_key: u32 = 0,

    /// Registra la región como pinned (pin-after-fill) con DEVICEMAP para que
    /// además sea desreferenciable desde kernels (gather P2). La región debe:
    ///   - estar ALINEADA a página y con longitud múltiplo de página,
    ///   - estar ya tocada (el loader GGUF leyó esos rangos),
    ///   - tener contexto CUDA current (caller: cudaz.ensureCurrent).
    pub fn fromMmapBytes(bytes: []align(4096) const u8) !Self {
        if (bytes.len == 0) return error.EmptyBank;
        if (!std.mem.isAligned(@intFromPtr(bytes.ptr), std.heap.page_size_min))
            return error.NotPageAligned;
        if (bytes.len % std.heap.page_size_min != 0)
            return error.NotPageMultiple;
        var bank = Self{
            .ptr = @ptrCast(@alignCast(@constCast(bytes.ptr))),
            .len = bytes.len,
            .residency = .PINNED,
            .owned = false,
            .registered = false,
            .pinned_via_hostalloc = false,
        };
        try ext_mem.hostRegister(@constCast(bytes.ptr), bytes.len, ext_mem.CU_MEMHOSTREGISTER_DEVICEMAP);
        bank.registered = true;
        _ = g_registered_bytes.fetchAdd(bank.len, .monotonic);
        return bank;
    }

    /// VA device-visible del host pineado. En Linux UVA coincide con el puntero
    /// host: sirve como fuente de cuMemcpyHtoDAsync y como dirección global en
    /// kernels (DEVICEMAP).
    pub fn devPtr(self: *const Self) usize {
        return @intFromPtr(self.ptr);
    }

    /// Bytes del banco (para presupuesto/stats).
    pub fn length(self: *const Self) usize {
        return self.len;
    }

    /// Des-registra. Idempotente (segundo unreg es no-op). NO libera los bytes:
    /// son préstamo del mmap/buffer original (o memoria propia si owned=true).
    /// No llama unreg si la memoria fue allocada con hostAlloc (ya no está registrada vía hostRegister).
    pub fn unreg(self: *Self) void {
        if (!self.registered) return;
        // No des-registrar memoria cuMemHostAlloc: ya NO está registrada vía
        // hostRegister (deinit() aplica el mismo gate). El unregister sobre
        // ella falla NOT_REGISTERED y su breadcrumb ruidoso confunde el
        // diagnóstico de fallos REALES del driver.
        if (self.pinned_via_hostalloc) {
            self.registered = false;
            return;
        }
        if (self.ptr) |ptr| {
            ext_mem.hostUnregister(@constCast(ptr)) catch {};
        }
        self.registered = false;
        _ = g_registered_bytes.fetchSub(self.len, .monotonic);
        // Ensure no pending GPU work references the region before the caller
        // frees the underlying allocation. Synchronous wait is required because
        // hostUnregister is fire-and-forget on some drivers and freeing while a
        // DMA is in flight corrupts the heap (the original allocator’s free
        // then faults in its poison-write).
        cudaz.cuCtxSynchronize() catch {};
    }

    /// Registra el mmap COMPLETO de un GgufFile cargado con fromFileMmap.
    ///
    /// Los tensores GGUF van alineados a 32 B (no a página), así que registrar
    /// slices individuales falla por `NotPageAligned`; en cambio el INICIO del
    /// archivo sí es page-aligned y la longitud se recorta a múltiplo de
    /// pagina. Un solo registro deja TODOS los tensores del archivo
    /// device-accesibles (UVA + DEVICEMAP): H2D async directo y gather P2 sin
    /// slabs intermedios. El caller conserva el HostBank vivo mientras use el
    /// archivo y llama unreg al soltarlo.
    pub fn fromFileMmapWhole(memory: []const u8) !Self {
        if (memory.len == 0) return error.EmptyBank;
        if (!std.mem.isAligned(@intFromPtr(memory.ptr), std.heap.page_size_min))
            return error.NotPageAligned;
        const usable = memory.len - (memory.len % std.heap.page_size_min);
        if (usable == 0) return error.EmptyBank;
        // memory.ptr is page-aligned (checked above) and usable is a multiple of page_size,
        // so we can create a slice with alignment 4096.
        const aligned_slice = @as([]align(4096) const u8, @ptrCast(@alignCast(memory[0..usable])));
        return fromMmapBytes(aligned_slice);
    }

    /// Alloca una nueva región de memoria con la residenciá especificada.
    /// La memoria es propiedad del HostBank y sera liberada en deinit.
    pub fn alloc(allocator: std.mem.Allocator, bytes: usize, residency: HostResidency) !Self {
        if (bytes == 0) return error.InvalidArgument;
        const page_size = std.heap.page_size_min;
        switch (residency) {
            .PAGEABLE => {
                const ptr = try allocator.alloc(u8, bytes);
                return Self{
                    .ptr = @ptrCast(&ptr[0]),
                    .len = bytes,
                    .residency = residency,
                    .owned = true,
                    .registered = false,
                    .pinned_via_hostalloc = false,
                    .allocator = allocator,
                    .orig_slice = ptr,
                };
            },
            .LOCKED => {
                const aligned_bytes = ((bytes + page_size - 1) / page_size) * page_size;
                const ptr = try allocator.alloc(u8, aligned_bytes);
                return Self{
                    .ptr = @as(?*u8, @ptrCast(ptr.ptr)),
                    .len = aligned_bytes,
                    .residency = residency,
                    .owned = true,
                    .registered = false,
                    .pinned_via_hostalloc = false,
                    .allocator = allocator,
                    .orig_slice = ptr,
                };
            },
            .PINNED => {
                const buf = ext_sync.hostAlloc(bytes) orelse return error.OutOfMemory;
                const ptr: ?*u8 = @ptrCast(buf.ptr);
                const bank = Self{
                    .ptr = ptr,
                    .len = bytes,
                    .residency = residency,
                    .owned = true,
                    .registered = true,
                    .pinned_via_hostalloc = true,
                    .allocator = allocator,
                };
                _ = g_registered_bytes.fetchAdd(bank.len, .monotonic);
                return bank;
            },
            .DISK_BACKED => {
                return error.InvalidArgument;
            },
        }
    }

    /// Alloca una nueva región de memoria con la residenciá especificada, asegurando
    /// que la memoria esté alineada a página y su longitud sea múltiplo de página.
    /// Esto es necesario para poder pinar la memoria posteriormente (después de llenarla).
    /// La memoria es propiedad del HostBank y sera liberada en deinit.
    pub fn allocAligned(allocator: std.mem.Allocator, bytes: usize, residency: HostResidency) !Self {
        if (bytes == 0) return error.InvalidArgument;
        const page_size = std.heap.page_size_min;
        const aligned_bytes = ((bytes + page_size - 1) / page_size) * page_size;
        var bank = Self{
            .ptr = null,
            .len = 0,
            .residency = residency,
            .owned = true,
            .registered = false,
            .pinned_via_hostalloc = false,
            .allocator = allocator,
        };
        switch (residency) {
            .PAGEABLE, .LOCKED => {
                // Over-allocate to guarantee a page-aligned address without
                // leaving the allocator (the testing DebugAllocator would not
                // own memory obtained via posix_memalign). deinit() frees
                // the ORIGINAL whole slice (bank.orig_slice) — never a
                // derived alias — so the allocator's free list stays sane.
                const oversize = aligned_bytes + page_size;
                const raw = try allocator.alloc(u8, oversize);
                const raw_addr = @intFromPtr(raw.ptr);
                const aligned_addr = std.mem.alignForward(usize, raw_addr, page_size);
                const aligned_off = aligned_addr - raw_addr;
                const aligned = raw[aligned_off..][0..aligned_bytes];
                bank.ptr = @ptrCast(aligned.ptr);
                bank.len = aligned_bytes;
                bank.orig_slice = raw;
            },
            .PINNED => {
                // Usar cudaHostAlloc (vía cudaz_ext_sync) para memoria page-locked accesible por GPU
                const buf = ext_sync.hostAlloc(aligned_bytes) orelse return error.OutOfMemory;
                bank.ptr = @ptrCast(buf.ptr);
                bank.len = aligned_bytes;
                bank.registered = true;
                bank.pinned_via_hostalloc = true;
                _ = g_registered_bytes.fetchAdd(bank.len, .monotonic);
            },
            .DISK_BACKED => {
                return error.InvalidArgument;
            },
        }
        return bank;
    }

    /// Load the byte ranges planned by `ftw.readPlan(names)` into a
    /// page‑aligned HostBank. The resulting bank spans from the first
    /// range start to the end of the last range; every range lands at
    /// its original FTW offset (relative to the bank’s start), so
    /// 4096‑aligned tensor blobs stay aligned for O_DIRECT.
    pub fn loadFromFtw(
        allocator: std.mem.Allocator,
        f: *const ftw.FtwFile,
        names: []const []const u8,
    ) !Self {
        const ranges = try f.readPlan(allocator, names);
        defer allocator.free(ranges);
        if (ranges.len == 0) return error.InvalidArgument;
        const min_start = ranges[0].start;
        const max_end = ranges[ranges.len - 1].start + ranges[ranges.len - 1].len;
        const span: usize = @intCast(max_end - min_start);
        const bank = try allocAligned(allocator, span, .PAGEABLE);
        const slice = bank.orig_slice.?;
        for (ranges) |r| {
            const off: usize = @intCast(r.start - min_start);
            @memcpy(slice[off..][0..r.len], f.data[@intCast(r.start)..][0..r.len]);
        }
        return bank;
    }

    /// Pinna la region de memoria como pinned (cudaHostRegister con DEVICEMAP) para
    /// permitir acceso zero-copy desde GPU y desreferenciacion desde kernels.
    /// La region debe estar ya llena con los datos deseados (pin-after-fill).
    /// Solo se puede llamar en regiones que esten en residency PAGEABLE y owned=true.
    /// Después de llamar, la residencia cambia a PINNED y registered=true.
    pub fn pin(self: *Self) !void {
        if (!self.owned) {
            return error.NotOwned;
        }
        // Idempotente: si ya está pinneado, no hacer nada (incluye la
        // segunda llamada después de un pin exitoso).
        if (self.registered) {
            return;
        }
        if (self.residency != .PAGEABLE) {
            return error.NotPageable;
        }
        // Verificar alineación y longitud múltiplo de pagina (debería ser cierta si se usó allocAligned)
        const page_size = std.heap.page_size_min;
        if (!std.mem.isAligned(@intFromPtr(self.ptr), page_size)) {
            return error.NotPageAligned;
        }
        if (self.len % page_size != 0) {
            return error.NotPageMultiple;
        }
        // Honest GPU guard: bail out if there is no usable device even if the
        // driver is loaded. cuInit may succeed with 0 devices, in which case
        // hostRegister would either no-op or fault inside the driver.
        _ = cudaz.cuDeviceGet(0) catch return error.NoDevice;
        const ptr = self.ptr orelse return error.NoMemory;
        try ext_mem.hostRegister(@constCast(ptr), self.len, ext_mem.CU_MEMHOSTREGISTER_DEVICEMAP);
        self.registered = true;
        self.pinned_via_hostalloc = false; // usamos hostRegister
        self.residency = .PINNED;
        _ = g_registered_bytes.fetchAdd(self.len, .monotonic);
    }

    /// Libera los recursos asociados al HostBank.
    /// Si owned=true, libera la memoria según la residenciá.
    /// Si registered=true y no es pinned_via_hostalloc, primero desregistra.
    pub fn deinit(self: *Self) void {
        if (hybridDebug()) {
            const orig_ptr: usize = if (self.orig_slice) |s| @intFromPtr(s.ptr) else 0;
            const bank_ptr: usize = if (self.ptr) |p| @intFromPtr(p) else 0;
            debugz.dbg.print(
                "[pool] deinit: residency={s} registered={} pinned_via_hostalloc={} orig_slice={x} bank.ptr={x}\n",
                .{ @tagName(self.residency), self.registered, self.pinned_via_hostalloc, orig_ptr, bank_ptr },
            );
        }
        if (self.registered and !self.pinned_via_hostalloc) {
            self.unreg();
        }
        if (self.owned) {
            switch (self.residency) {
                .PAGEABLE, .LOCKED => {
                    if (self.orig_slice) |slice| {
                        self.allocator.free(slice);
                    }
                },
                .PINNED => {
                    if (self.ptr) |ptr| {
                        if (self.pinned_via_hostalloc) {
                            // Memoria allocada con ext_sync.hostAlloc; usar ext_sync.hostFree.
                            const ptr64: [*]align(64) u8 = @ptrCast(@alignCast(ptr));
                            const slice: []align(64) u8 = ptr64[0..self.len];
                            ext_sync.hostFree(slice);
                            _ = g_registered_bytes.fetchSub(self.len, .monotonic);
                        } else {
                            // Memoria pineada con hostRegister (pin-after-fill); liberar el slice original.
                            if (self.orig_slice) |slice| {
                                self.allocator.free(slice);
                            }
                        }
                    }
                },
                .DISK_BACKED => {
                    if (self.orig_slice) |slice| {
                        self.allocator.free(slice);
                    }
                },
            }
        }
        // El ptr no se establece en null porque después de deinit el objeto no debería usarse.
    }

    /// 4.3' copy-once: UNA copia física del fichero a un buffer PINNED anónimo
    /// (cuMemHostAlloc — registro DEVICEMAP implícito que el driver 580 SÍ acepta
    /// para memoria no-file-backed; ver 4.3 refutado con control). El buffer
    /// resultante es válido como `mmap_region` de MoeLayer.init: hostRegister
    /// sobre él funciona (anon) y el gather lee sus VAs directos — cero copias
    /// por capa. RAM = tamaño del fichero (una sola vez).
    /// El caller debe `unreg()` + liberar el buffer original tras deinit de las capas.
    pub fn fromFileCopyOnce(allocator: std.mem.Allocator, io: anytype, path: []const u8) !struct { bank: Self, buf: []u8 } {
        const dir = std.Io.Dir.cwd();
        var f = try dir.openFile(io, path, .{ .mode = .read_only });
        defer f.close(io);
        const size = (try f.stat(io)).size;
        if (size == 0) return error.EmptyBank;

        const page = std.heap.page_size_min;
        const aligned_bytes = ((size + page - 1) / page) * page;

        // PINNED anónimo: registro DEVICEMAP del driver OK (control 4.3).
        var bank = try Self.allocAligned(allocator, aligned_bytes, .PINNED);
        errdefer bank.deinit();

        // Lectura única del fichero al buffer pineado. 4.13 (bench-copyonce):
        // io_uring QD32 chunk 4MiB = 7.4× cold vs pread serial en NVMe
        // (12.6s→1.7s para 9.9GB); fallback pread serial si el ring falla
        // (kernel sin io_uring o QD saturado). El buffer YA es pinned
        // CUDA/UVA — no register_buffers (RLIMIT_MEMLOCK aparte, ver 4.13).
        const fd: c_int = @intCast(f.handle);
        const dst: [*]u8 = @ptrCast(bank.ptr.?);
        // 4.13-bis: O_DIRECT (offset+len 4096-aligned; tail <4KB vía fd
        // buffered /proc/self/fd/N) — el read buffered de 9.9GB contamina
        // ~9.7GB de page-cache y con RAM presionada (devs compilando)
        // thrashea; O_DIRECT deja delta ~7MB (bench-copyonce medido).
        const aligned_end: usize = (size / 4096) * 4096;
        readIouringDirect(fd, dst[0..aligned_end]) catch {
            // Fallback: read buffered completo (ring/O_DIRECT no disponibles).
            readSerial(fd, dst[0..size]) catch |e| return e;
        };
        if (aligned_end < size) {
            var pbuf: [64]u8 = undefined;
            const pz = std.fmt.bufPrintZ(&pbuf, "/proc/self/fd/{d}", .{fd}) catch return error.OpenFailed;
            const bfd = open64(pz.ptr, 0);
            if (bfd < 0) return error.OpenFailed;
            defer _ = std.os.linux.close(@intCast(bfd));
            var t: usize = 0;
            const tl = size - aligned_end;
            while (t < tl) {
                const piece = @min(@as(usize, 1 << 20), tl - t);
                const n = pread(bfd, dst + aligned_end + t, piece, @intCast(aligned_end + t));
                if (n <= 0) return error.ReadFailed;
                t += @intCast(n);
            }
        }
        _ = &bank;
        return .{ .bank = bank, .buf = dst[0..@intCast(size)] };
    }

    /// MH-15: Crea un HostBank respaldado por DiskTier.
    /// Materializa el experto/tensor `key` desde disco a un buffer PINNED
    /// page-aligned. El caller debe liberar el bank con `deinit()`.
    pub fn fromDisk(allocator: std.mem.Allocator, dt: *disk_tier.DiskTier, key: u32) !Self {
        const buf = try dt.readExpert(allocator, key);
        errdefer allocator.free(buf);
        const ptr: ?*u8 = @ptrCast(@alignCast(buf.ptr));
        return Self{
            .ptr = ptr,
            .len = buf.len,
            .residency = .DISK_BACKED,
            .owned = true,
            .registered = false,
            .pinned_via_hostalloc = false,
            .allocator = allocator,
            .orig_slice = buf,
            .disk_tier = dt,
            .disk_key = key,
        };
    }

    /// MH-15: Offloada el contenido del bank a DiskTier.
    /// Registra la ubicación en el shard map y marca el bank como DISK_BACKED.
    /// No libera la memoria host; el caller decide si la mantiene o la libera.
    pub fn offloadToDisk(self: *Self, location: disk_tier.ShardLocation) !void {
        if (self.disk_tier) |dt| {
            try dt.registerExpert(self.disk_key, location);
        }
        self.residency = .DISK_BACKED;
    }

    /// MH-15: Recarga el contenido desde DiskTier.
    /// Solo válido para bancos DISK_BACKED con disk_tier asignado.
    pub fn reloadFromDisk(self: *Self) !void {
        if (self.disk_tier) |dt| {
            if (self.owned and self.orig_slice) |slice| {
                self.allocator.free(slice);
            }
            const buf = try dt.readExpert(self.allocator, self.disk_key);
            const ptr: ?*u8 = @ptrCast(@alignCast(buf.ptr));
            self.ptr = ptr;
            self.len = buf.len;
            self.orig_slice = buf;
            self.residency = .PINNED;
        }
    }
};

/// 4.13: lectura io_uring QD alto sobre el buffer pinned. Depth 32 con
/// chunks de 4 MiB satura la NVMe (5.79 GB/s medidos vs 0.79 serial).
/// 4.13: lectura io_uring QD alto con O_DIRECT sobre el buffer pinned.
/// Depth 32 chunks 4MiB satura la NVMe (5.79 GB/s vs 0.79 serial) y
/// O_DIRECT evita contaminar page-cache (delta 7MB vs 9702MB en 9.9GB).
/// `dst.len` debe ser múltiplo de 4096 (contrato ftw.zig); el tail corto
/// lo gestiona el caller vía fd buffered.
fn readIouringDirect(fd: c_int, dst: []u8) !void {
    if (builtin.target.os.tag != .linux) return error.IouringUnavailable;
    var zbuf: [64]u8 = undefined;
    const pz = std.fmt.bufPrintZ(&zbuf, "/proc/self/fd/{d}", .{fd}) catch return error.OpenFailed;
    const ofd = open64(pz.ptr, 0o40000); // O_RDONLY|O_DIRECT
    if (ofd < 0) return error.OdirectUnavailable;
    defer _ = std.os.linux.close(@intCast(ofd));

    const Ring = std.os.linux.IoUring;
    const QD = 32;
    const CHUNK = 4 << 20;
    var ring = Ring.init(QD, 0) catch return error.IouringUnavailable;
    defer ring.deinit();
    const n_chunks = (dst.len + CHUNK - 1) / CHUNK;
    var next: usize = 0;
    var completed: usize = 0;
    while (completed < n_chunks) {
        var submitted: usize = 0;
        while (next < n_chunks and submitted < QD) : (submitted += 1) {
            const off = next * CHUNK;
            const len = @min(CHUNK, dst.len - off);
            const sqe = try ring.get_sqe();
            sqe.prep_read(ofd, dst[off .. off + len], @intCast(off));
            sqe.user_data = next;
            next += 1;
        }
        _ = try ring.submit_and_wait(1);
        while (ring.cq_ready() > 0 and completed < n_chunks) {
            const cqe = try ring.copy_cqe();
            if (cqe.res < 0) return error.ReadFailed;
            completed += 1;
        }
    }
}

/// Fallback serial (espejo del loop pread original).
fn readSerial(fd: c_int, dst: []u8) !void {
    var done: usize = 0;
    while (done < dst.len) {
        const chunk = @min(@as(usize, 1 << 20), dst.len - done);
        const n = pread(fd, dst.ptr + done, chunk, @intCast(done));
        if (n <= 0) return error.ReadFailed;
        done += @intCast(n);
    }
}

fn hybridDebug() bool {
    return std.c.getenv("ZIG_AI_HYBRID_DEBUG") != null;
}

/// Presupuesto global de pin anotado (Linux nativo no tiene cupo duro; se
/// registra para WSL futuro y para stats, estudio §5.2).
var g_registered_bytes = std.atomic.Value(usize).init(0);

pub fn registeredBytes() usize {
    return g_registered_bytes.load(.monotonic);
}

const testing = std.testing;

test "HostBank: rechaza no alineado y vacío sin tocar driver" {
    try testing.expectError(error.EmptyBank, HostBank.fromMmapBytes(@as([]align(4096) const u8, &[_]u8{})));
}

test "HostBank: contabilidad registeredBytes sube/baja con par register/unr" {
    // Solo la aritmética del contador; el registro real necesita GPU y va en
    // tests/test_host_bank.zig (gated by CUDA).
    const before = registeredBytes();
    const delta: usize = 3 * std.heap.page_size_min;
    _ = g_registered_bytes.fetchAdd(delta, .monotonic);
    try testing.expectEqual(before + delta, registeredBytes());
    _ = g_registered_bytes.fetchSub(delta, .monotonic);
    try testing.expectEqual(before, registeredBytes());
}

test "HostBank: alloc PAGEABLE devuelve memoria válida" {
    const a = std.testing.allocator;
    var bank = try HostBank.alloc(a, 1024, .PAGEABLE);
    defer bank.deinit();
    try testing.expectEqual(bank.len, 1024);
    try testing.expectEqual(bank.residency, .PAGEABLE);
    try testing.expectEqual(bank.owned, true);
    try testing.expectEqual(bank.registered, false);
}

test "HostBank: alloc LOCKED devuelve memoria válida y está bloqueada" {
    const a = std.testing.allocator;
    var bank = try HostBank.alloc(a, 1024, .LOCKED);
    defer bank.deinit();
    try testing.expectEqual(bank.len, ((1024 + std.heap.page_size_min - 1) / std.heap.page_size_min) * std.heap.page_size_min);
    try testing.expectEqual(bank.residency, .LOCKED);
    try testing.expectEqual(bank.owned, true);
    try testing.expectEqual(bank.registered, false);
    // Nota: no podemos probar fácilmente mlock en unit test sin privilegios, pero al menos verificamos que no falle.
}

test "HostBank: alloc PINNED devuelve memoria válida y está registrada" {
    const a = std.testing.allocator;
    var bank = try HostBank.alloc(a, 1024, .PINNED);
    defer bank.deinit();
    try testing.expectEqual(bank.len, 1024);
    try testing.expectEqual(bank.residency, .PINNED);
    try testing.expectEqual(bank.owned, true);
    try testing.expectEqual(bank.registered, true);
    // Nota: el registro global de bytes debería haber aumentado
    try testing.expectEqual(registeredBytes(), 1024);
}

test "HostBank: allocAligned PAGEABLE devuelve memoria alineada y múltiplo de pagina" {
    const a = std.testing.allocator;
    var bank = try HostBank.allocAligned(a, 1024, .PAGEABLE);
    defer bank.deinit();
    const page_size = std.heap.page_size_min;
    try testing.expectEqual(bank.len % page_size, 0);
    try testing.expectEqual(std.mem.isAligned(@intFromPtr(bank.ptr), page_size), true);
    try testing.expectEqual(bank.residency, .PAGEABLE);
    try testing.expectEqual(bank.owned, true);
    try testing.expectEqual(bank.registered, false);
}

test "HostBank: allocAligned LOCKED devuelve memoria alineada, multiples de pagina y está bloqueada" {
    const a = std.testing.allocator;
    var bank = try HostBank.allocAligned(a, 1024, .LOCKED);
    defer bank.deinit();
    const page_size = std.heap.page_size_min;
    try testing.expectEqual(bank.len % page_size, 0);
    try testing.expectEqual(std.mem.isAligned(@intFromPtr(bank.ptr), page_size), true);
    try testing.expectEqual(bank.residency, .LOCKED);
    try testing.expectEqual(bank.owned, true);
    try testing.expectEqual(bank.registered, false);
    // Nota: no podemos probar fácilmente mlock en unit test sin privilegios, pero al menos verificamos que no falle.
}

test "HostBank: allocAligned PINNED devuelve memoria alineada, multiples de pagina y está registrada" {
    const a = std.testing.allocator;
    var bank = try HostBank.allocAligned(a, 1024, .PINNED);
    defer bank.deinit();
    const page_size = std.heap.page_size_min;
    try testing.expectEqual(bank.len % page_size, 0);
    try testing.expectEqual(std.mem.isAligned(@intFromPtr(bank.ptr), page_size), true);
    try testing.expectEqual(bank.residency, .PINNED);
    try testing.expectEqual(bank.owned, true);
    try testing.expectEqual(bank.registered, true);
    // Nota: el registro global de bytes debería haber aumentado
    try testing.expectEqual(registeredBytes(), bank.len);
}

test "HostBank: pin después de allocAligned PAGEABLE funciona" {
    cudaz.ensureContext() catch return error.SkipZigTest;
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var bank = try HostBank.allocAligned(a, 1024, .PAGEABLE);
    defer bank.deinit();
    try testing.expectEqual(bank.residency, .PAGEABLE);
    try testing.expectEqual(bank.registered, false);
    try bank.pin();
    try testing.expectEqual(bank.residency, .PINNED);
    try testing.expectEqual(bank.registered, true);
    // Nota: el registro global de bytes debería haber aumentado
    try testing.expectEqual(registeredBytes(), bank.len);
}

test "HostBank: pin falla si no es owned" {
    const a = std.testing.allocator;
    const bytes = try a.alloc(u8, 1024);
    defer a.free(bytes);
    var bank = HostBank{ .ptr = @ptrCast(@alignCast(@constCast(bytes.ptr))), .len = bytes.len, .residency = .PAGEABLE, .owned = false, .registered = false };
    try testing.expectError(error.NotOwned, bank.pin());
}

test "HostBank: pin falla si no es PAGEABLE" {
    const a = std.testing.allocator;
    var bank = try HostBank.alloc(a, 1024, .LOCKED);
    defer bank.deinit();
    try testing.expectError(error.NotPageable, bank.pin());
}

test "HostBank: pin es idempotente" {
    cudaz.ensureContext() catch return error.SkipZigTest;
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var bank = try HostBank.allocAligned(a, 1024, .PAGEABLE);
    defer bank.deinit();
    try bank.pin();
    try bank.pin(); // segundo pin no debe fallar
    try testing.expectEqual(bank.residency, .PINNED);
    try testing.expectEqual(bank.registered, true);
}

test "HostBank: alloc→pin→unreg→deinit (single-slice identity)" {
    cudaz.ensureContext() catch return error.SkipZigTest;
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var bank = try HostBank.allocAligned(a, 4096, .PAGEABLE);
    bank.pin() catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    try testing.expect(bank.registered);
    bank.unreg();
    try testing.expect(!bank.registered);
    bank.deinit();
}

test "HostBank: loadFromFtw copies planned ranges into page-aligned bank" {
    const a = std.testing.allocator;
    // Build a small FTW with two tensors, each its own 4096‑aligned page.
    var b = ftw.FtwBuilder.init(a);
    defer b.deinit();
    const t0 = [_]u8{0xAA} ** 16;
    const t1 = [_]u8{0xBB} ** 32;
    try b.addTensor("alpha", 0, &[_]u64{16}, &t0);
    try b.addTensor("beta", 0, &[_]u64{32}, &t1);
    const ftw_buf = try b.finish();
    defer a.free(ftw_buf);
    var f = try ftw.FtwFile.fromBytes(a, ftw_buf);
    defer f.deinit();

    // Load the planned ranges for both tensors.
    var bank = try HostBank.loadFromFtw(a, &f, &[_][]const u8{ "alpha", "beta" });
    defer bank.deinit();

    // The bank must be page‑aligned and span both ranges.
    const page_size = std.heap.page_size_min;
    try testing.expectEqual(@as(usize, 0), bank.len % page_size);
    try testing.expect(std.mem.isAligned(@intFromPtr(bank.ptr), page_size));

    // Every tensor lands at its original FTW offset (relative to min_start).
    const e_alpha = f.find("alpha") orelse return error.TestUnexpectedResult;
    const e_beta = f.find("beta") orelse return error.TestUnexpectedResult;
    const min_start = e_alpha.data_off + f.header.data_start;
    const alpha_off: usize = @intCast((e_alpha.data_off + f.header.data_start) - min_start);
    const beta_off: usize = @intCast((e_beta.data_off + f.header.data_start) - min_start);
    const slice = bank.orig_slice.?;
    try testing.expectEqualSlices(u8, &t0, slice[alpha_off..][0..t0.len]);
    try testing.expectEqualSlices(u8, &t1, slice[beta_off..][0..t1.len]);
}
