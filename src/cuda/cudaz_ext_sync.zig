//! Bindings Lane F: memops del FRONT-END CUDA para el handshake CPU↔GPU
//! (cuStreamWriteValue64 / cuStreamWaitValue64).
//!
//! ⚠️ Lección anti-trampa FreeToken (cpu_executor.py:36-43): NUNCA esperar con
//! un spin-kernel SM-residente — clampa la frecuencia CPU del laptop y miente
//! la utilización de GPU. Las memops ejecutan en el front-end del stream sin
//! ocupar SMs ⇒ la ventana CPU deja la GPU verazmente libre.
//!
//! Resolución DINÁMICA (dlopen de libcuda.so.1 al primer uso, handle retenido):
//! si el driver no expone memops (WDDM/vGPU/antiguo) o no hay GPU, `probe()`
//! lo reporta y el executor cae al camino lento documentado (staging host
//! síncrono). Sin dependencia dura de enlace en modo host-only.
const std = @import("std");
const debugz = @import("debug");

/// Handle de stream opaco (bit-compatible con cudaz.CUstream vía usize).
pub const Stream = *opaque {};

/// CU_STREAM_LEGACY (default stream del driver).
pub const legacy_stream: Stream = @ptrFromInt(1);

/// Flags estándar: WAIT_VALUE_GTE (espera hasta valor >= objetivo; secuencias
/// monotónicas por slot). WRITE usa DEFAULT (0).
pub const WAIT_VALUE_GTE: u32 = 1;

const FnWriteValue64 = *const fn (stream: Stream, ptr: usize, value: u64, flags: u32) callconv(.c) c_int;
const FnWaitValue64 = *const fn (stream: Stream, ptr: usize, value: u64, flags: u32) callconv(.c) c_int;
const FnSync = *const fn (stream: Stream) callconv(.c) c_int;
const FnHostAlloc = *const fn (pp: *?*anyopaque, bytes: usize) callconv(.c) c_int;
const FnHostFree = *const fn (p: *anyopaque) callconv(.c) c_int;
const FnDtoHAsync = *const fn (dst: *anyopaque, src: *const anyopaque, bytes: usize, stream: Stream) callconv(.c) c_int;
const FnInit = *const fn (flags: c_uint) callconv(.c) c_int;
const FnDeviceGet = *const fn (dev: *c_int, ordinal: c_int) callconv(.c) c_int;
const FnPrimaryRetain = *const fn (ctx: *?*anyopaque, dev: c_int) callconv(.c) c_int;
const FnCtxSetCurrent = *const fn (ctx: ?*anyopaque) callconv(.c) c_int;
const FnStreamCreate = *const fn (stream: *?*anyopaque, flags: c_uint) callconv(.c) c_int;
const FnStreamDestroy = *const fn (stream: ?*anyopaque) callconv(.c) c_int;
/// CUhostFn = void(*)(void*) — corre en un hilo del driver al alcanzar su
/// posición EN EL STREAM (no bloquea nodos posteriores; solo notifica).
const HostFn = *const fn (?*anyopaque) callconv(.c) void;
const FnLaunchHostFunc = *const fn (stream: Stream, callback: HostFn, user_data: ?*anyopaque) callconv(.c) c_int;

const Ops = struct {
    init: FnInit,
    device_get: FnDeviceGet,
    primary_retain: FnPrimaryRetain,
    ctx_set_current: FnCtxSetCurrent,
    stream_create: FnStreamCreate,
    stream_destroy: FnStreamDestroy,
    write_value64: FnWriteValue64,
    wait_value64: FnWaitValue64,
    sync_stream: FnSync,
    host_alloc: FnHostAlloc,
    host_free: FnHostFree,
    dtoh_async: FnDtoHAsync,
    launch_host_func: FnLaunchHostFunc,
};

// Handle dlopen RETENIDO para siempre (los punteros deben vivir todo el
// proceso); es un leak intencional de una sola vez.
var g_lib: ?std.DynLib = null;
var g_ops: ?Ops = null;
var probe_done: bool = false;
/// Capacidades INDEPENDIENTES: en esta GeForce las memops no existen (801)
/// pero cuLaunchHostFunc/hostAlloc/dtoh sí funcionan.
pub var caps_memops: bool = false;
pub var caps_hostfunc: bool = false;

fn ensureLoaded() bool {
    if (g_lib != null) return true;
    g_lib = std.DynLib.open("libcuda.so.1") catch |err| {
        debugz.dbg.printLevel(.info, "[cudaz_ext_sync] libcuda.so.1 no cargable: {s}\n", .{@errorName(err)});
        return false;
    };
    return true;
}

fn lookup(comptime T: type, name: [:0]const u8) ?T {
    return g_lib.?.lookup(T, name);
}

/// Sonda funcional idempotente con CAPACIDADES INDEPENDIENTES: en esta
/// GeForce las memops no existen (801) pero cuLaunchHostFunc/hostAlloc/dtoh
/// sí funcionan ⇒ no se acoplan.
fn probeOnce() void {
    if (probe_done) return;
    probe_done = true;

    if (!ensureLoaded()) return;
    const ops: Ops = .{
        .init = lookup(FnInit, "cuInit") orelse return,
        .device_get = lookup(FnDeviceGet, "cuDeviceGet") orelse return,
        .primary_retain = lookup(FnPrimaryRetain, "cuDevicePrimaryCtxRetain") orelse return,
        .ctx_set_current = lookup(FnCtxSetCurrent, "cuCtxSetCurrent") orelse return,
        .stream_create = lookup(FnStreamCreate, "cuStreamCreate") orelse return,
        .stream_destroy = lookup(FnStreamDestroy, "cuStreamDestroy") orelse return,
        .write_value64 = lookup(FnWriteValue64, "cuStreamWriteValue64") orelse return,
        .wait_value64 = lookup(FnWaitValue64, "cuStreamWaitValue64") orelse return,
        .sync_stream = lookup(FnSync, "cuStreamSynchronize") orelse return,
        .host_alloc = lookup(FnHostAlloc, "cuMemAllocHost_v2") orelse return,
        .host_free = lookup(FnHostFree, "cuMemFreeHost") orelse return,
        .dtoh_async = lookup(FnDtoHAsync, "cuMemcpyDtoHAsync_v2") orelse return,
        .launch_host_func = lookup(FnLaunchHostFunc, "cuLaunchHostFunc") orelse return,
    };

    // Driver inicializado + contexto primario actual.
    if (ops.init(0) != 0) return;
    var dev: c_int = 0;
    var ctx: ?*anyopaque = null;
    if (ops.device_get(&dev, 0) != 0) return;
    if (ops.primary_retain(&ctx, dev) != 0 or ctx == null) return;
    _ = ops.ctx_set_current(ctx);

    var p: ?*anyopaque = null;
    if (ops.host_alloc(&p, 64) != 0 or p == null) return;
    defer _ = ops.host_free(p.?);

    // Stream REAL: las memops no están soportadas en el legacy/NULL stream.
    var ps: ?*anyopaque = null;
    if (ops.stream_create(&ps, 0) != 0 or ps == null) return;
    const probe_stream: Stream = @ptrCast(ps.?);
    defer _ = ops.stream_destroy(ps);

    const uptr = @intFromPtr(p.?);
    const mark: u64 = 0xC0FFEE_5EED_0001;

    // CAP 1: host-func (independiente de memops).
    const ProbeCb = struct {
        fired: std.atomic.Value(bool) = .init(false),
        fn cb(raw: ?*anyopaque) callconv(.c) void {
            const c: *@This() = @ptrCast(@alignCast(raw.?));
            c.fired.store(true, .release);
        }
    };
    var pc: ProbeCb = .{};
    if (ops.launch_host_func(probe_stream, ProbeCb.cb, &pc) == 0 and
        ops.sync_stream(probe_stream) == 0 and pc.fired.load(.acquire))
    {
        caps_hostfunc = true;
    }

    // CAP 2: memops — write64 verificado desde host + wait64 sin colgarse.
    var memops = ops.write_value64(probe_stream, uptr, mark, 0) == 0;
    memops = memops and ops.sync_stream(probe_stream) == 0;
    if (memops) {
        const got: u64 = @as(*align(8) const volatile u64, @ptrCast(@alignCast(p.?))).*;
        memops = got == mark;
    }
    memops = memops and ops.wait_value64(probe_stream, uptr, mark, WAIT_VALUE_GTE) == 0;
    memops = memops and ops.sync_stream(probe_stream) == 0;
    caps_memops = memops;

    if (caps_memops or caps_hostfunc) {
        g_ops = ops; // base común útil (hostAlloc/dtoh/sync) aunque memops falles
        debugz.dbg.printLevel(.info, "[cudaz_ext_sync] caps: memops={} hostfunc={}\n", .{ caps_memops, caps_hostfunc });
    }
}

/// Memops del front-end disponibles (en esta GeForce: NO).
pub fn memopsAvailable() bool {
    probeOnce();
    return caps_memops;
}

fn requireOps() ?Ops {
    probeOnce();
    return g_ops;
}

/// Memoria host page-locked del driver (device-visible vía UVA).
pub fn hostAlloc(bytes: usize) ?[]align(64) u8 {
    const m = requireOps() orelse return null;
    var p: ?*anyopaque = null;
    if (m.host_alloc(&p, bytes) != 0 or p == null) return null;
    return @as([*]align(64) u8, @ptrCast(@alignCast(p.?)))[0..bytes];
}

pub fn hostFree(buf: []align(64) u8) void {
    const m = requireOps() orelse return;
    _ = m.host_free(@ptrCast(buf.ptr));
}

/// Encola en `stream` la escritura del flag READY (lado GPU, front-end).
pub fn writeReady(stream: Stream, flag_ptr: usize, seq: u64) !void {
    const m = requireOps() orelse return error.MemopsUnavailable;
    if (m.write_value64(stream, flag_ptr, seq, 0) != 0) return error.CudaError;
}

/// Encola en `stream` la espera del DONE (libera nodos posteriores del grafo
/// cuando el coordinador host publique la secuencia).
pub fn waitDone(stream: Stream, flag_ptr: usize, seq: u64) !void {
    const m = requireOps() orelse return error.MemopsUnavailable;
    if (m.wait_value64(stream, flag_ptr, seq, WAIT_VALUE_GTE) != 0) return error.CudaError;
}

/// cuLaunchHostFunc disponible (capacidad independiente de memops).
pub fn hostFuncAvailable() bool {
    probeOnce();
    return caps_hostfunc;
}

/// Encola un callback de driver en `stream` (no bloquea nodos posteriores).
pub fn launchHostFunc(stream: Stream, callback: HostFn, user_data: ?*anyopaque) !void {
    const m = requireOps() orelse return error.MemopsUnavailable;
    if (m.launch_host_func(stream, callback, user_data) != 0) return error.CudaError;
}

/// D2H asíncrono (device → host pineado) encolado en `stream`.
pub fn dtohAsync(dst_host: usize, src_dev: usize, bytes: usize, stream: Stream) !void {
    const m = requireOps() orelse return error.MemopsUnavailable;
    if (m.dtoh_async(@ptrFromInt(dst_host), @ptrFromInt(src_dev), bytes, stream) != 0) return error.CudaError;
}

/// Publicación DONE desde el host: store volátil release sobre memoria
/// host-pineada mapeada (visible al waitValue64 del dispositivo).
pub fn hostPublishDone(flag_ptr: usize, seq: u64) void {
    const p: *std.atomic.Value(u64) = @ptrFromInt(flag_ptr);
    p.store(seq, .release);
}

test "memops probe no revienta sin GPU" {
    // Debe devolver true/false sin colgarse ni paniquear; en CI sin driver
    // simplemente habrá quedado unavailable.
    const avail = memopsAvailable();
    std.debug.print("[test] cudaz_ext_sync memops disponibles={}\n", .{avail});
}

/// Microbench Lane F/E8: coste por llamada de cuLaunchHostFunc en un stream
/// efímero. Devuelve (ns/launch solo-encolar, ns/launch+sincronizar) o null
/// sin GPU/driver. Referencia FreeToken: ~30-50µs el round-trip.
pub fn benchHostFuncOverhead(n: usize) ?struct { enqueue_ns: f64, roundtrip_ns: f64 } {
    const m = requireOps() orelse return null;
    var ps: ?*anyopaque = null;
    if (m.stream_create(&ps, 0) != 0 or ps == null) return null;
    defer _ = m.stream_destroy(ps);
    const st: Stream = @ptrCast(ps.?);

    const Noop = struct {
        fn cb(_: ?*anyopaque) callconv(.c) void {}
    };
    var dummy: u8 = 0;
    _ = &dummy;

    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    var t0: i128 = @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;

    for (0..n) |_| {
        _ = m.launch_host_func(st, Noop.cb, &dummy);
    }
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    var t1: i128 = @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
    if (m.sync_stream(st) != 0) return null;
    const enqueue_ns: f64 = @floatFromInt(t1 - t0);

    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    t0 = @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
    for (0..n) |_| {
        _ = m.launch_host_func(st, Noop.cb, &dummy);
        _ = m.sync_stream(st);
    }
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    t1 = @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;

    const nf: f64 = @floatFromInt(n);
    return .{
        .enqueue_ns = enqueue_ns / nf,
        .roundtrip_ns = @as(f64, @floatFromInt(t1 - t0)) / nf,
    };
}
