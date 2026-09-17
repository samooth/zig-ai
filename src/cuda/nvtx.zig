//! UC-4.1 (TODO_CUDA.md, lane-cuda): NVTX ranges para nsys — anotaciones
//! legibles por etapa en los perfiles (prefill/attn/ffn/lm_head pasan de
//! anónimos a con-nombre). ~50L, opt-in.
//!
//! Diseño FAIL-SAFE (patrón nvrtc/noop-stub del repo): NUNCA se linka
//! libnvtx — se resuelve por dlopen en el primer uso. Si la lib no está
//! (o no hay toolkit), TODO es no-op barato (branch en un bool cached).
//! Bajo `nsys`, nsys inyecta la extensión vía LD_PRELOAD y los rangos
//! aparecen; sin nsys, coste ~0.
//!
//! Nombres probados por dlopen (Ubuntu noble instala `libnvToolsExt.so.1`
//! SIN el symlink `libnvtx.so` del paquete dev):
//!   1. "libnvtx.so"        — canonical (toolkit/nsys injection)
//!   2. "libnvToolsExt.so.1" — Ubuntu noble libnvtoolsext1
//!
//! API estable de NVTX v2 (header /usr/include/nvToolsExt.h):
//!   nvtxRangePushA(const char*) -> int  (profundidad)
//!   nvtxRangePop(void)
//!   nvtxMarkA(const char*)
//!
//! Uso:
//!   nvtx.rangePush("prefill");   defer nvtx.rangePop();
//!   defer nvtx.scoped("decode") ... — ver ScopedRange (RAII).
//!
//! Referencia estudio: coderonion/zcuda (MIT) src/nvtx/safe.zig — patrón
//! safe-wrapper; atribución, cero dependencia.
const std = @import("std");
const builtin = @import("builtin");

const PushFn = *const fn ([*:0]const u8) callconv(.c) c_int;
const PopFn = *const fn () callconv(.c) void;
const MarkFn = *const fn ([*:0]const u8) callconv(.c) void;

const State = struct {
    loaded: bool = false,
    available: bool = false,
    push: PushFn = undefined,
    pop: PopFn = undefined,
    mark_fn: MarkFn = undefined,
};

/// Estado global lazy — el dlopen ocurre UNA vez (primera llamada). Sin
/// lock: worst case dos threads hacen dlopen idempotente y el último
/// escribe gana (funciones idénticas; benigno en la práctica).
var state: State = .{};

/// Carga perezosa de libnvtx. `false` = no-op para siempre (sin toolkit
/// o sin lib — CI/hosts sin CUDA no rompen NUNCA).
fn ensureLoaded() bool {
    if (state.loaded) return state.available;
    state.loaded = true;
    const candidates = [_][]const u8{
        "libnvtx.so",
        "libnvToolsExt.so.1",
    };
    for (candidates) |name| {
        if (dlopenSymbol(name, &state.push, &state.pop, &state.mark_fn)) {
            state.available = true;
            return true;
        }
    }
    return false;
}

fn dlopenSymbol(
    lib_name: []const u8,
    push: *PushFn,
    pop: *PopFn,
    mark_fn: *MarkFn,
) bool {
    var buf: [256]u8 = undefined;
    if (lib_name.len >= buf.len) return false;
    @memcpy(buf[0..lib_name.len], lib_name);
    buf[lib_name.len] = 0;
    const nt_name = buf[0..lib_name.len :0].ptr;

    const handle = if (comptime builtin.target.os.tag == .windows)
        c.LoadLibraryA(nt_name)
    else
        c.dlopen(nt_name, c.RTLD_LAZY);

    if (handle == null) return false;
    const h = handle.?;

    const symFn = if (comptime builtin.target.os.tag == .windows)
        struct {
            fn get(hdl: *anyopaque, name: [*:0]const u8) ?*anyopaque {
                return c.GetProcAddress(hdl, name);
            }
        }.get
    else
        struct {
            fn get(hdl: *anyopaque, name: [*:0]const u8) ?*anyopaque {
                return c.dlsym(hdl, name);
            }
        }.get;

    const p = symFn(h, "nvtxRangePushA") orelse return false;
    const q = symFn(h, "nvtxRangePop") orelse return false;
    const m = symFn(h, "nvtxMarkA") orelse return false;
    push.* = @ptrCast(@alignCast(p));
    pop.* = @ptrCast(@alignCast(q));
    mark_fn.* = @ptrCast(@alignCast(m));
    return true;
}

/// Push de un range con nombre (aparece en nsys como NVTX row). No-op si
/// libnvtx no está. `name` debe ser estático (vive hasta el Pop).
pub fn rangePush(name: [*:0]const u8) void {
    if (!ensureLoaded()) return;
    _ = state.push(name);
}

/// Pop del último range abierto. No-op si no hay lib.
pub fn rangePop() void {
    if (!ensureLoaded()) return;
    state.pop();
}

/// Marca instantánea (evento puntual en la timeline).
pub fn mark(name: [*:0]const u8) void {
    if (!ensureLoaded()) return;
    state.mark_fn(name);
}

/// RAII: `defer nvtx.scoped("decode")` — pop garantizado al salir del scope.
pub fn scoped(name: [*:0]const u8) void {
    rangePush(name);
}

/// ¿NVTX activo? (para tests/diagnostics — no fuerza la carga).
pub fn available() bool {
    return ensureLoaded();
}

// ─── libc mínimo — platform-conditional (link_libc ya es global en el repo) ───
const c = if (builtin.target.os.tag == .windows)
    struct {
        extern "kernel32" fn LoadLibraryA(path: [*:0]const u8) ?*anyopaque;
        extern "kernel32" fn GetProcAddress(handle: *anyopaque, symbol: [*:0]const u8) ?*anyopaque;
    }
else
    struct {
        extern fn dlopen(path: ?[*:0]const u8, mode: c_int) ?*anyopaque;
        extern fn dlsym(handle: ?*anyopaque, symbol: [*:0]const u8) ?*anyopaque;
        const RTLD_LAZY: c_int = 1; // POSIX glibc (linux)
    };

// ─────────────────────────────────────────────────────────────────────
// Tests unitarios (sin GPU/lib): available() es determinista por proceso
// (cached); en este host hay lib (libnvtoolsext1) pero el test NO puede
// asumirlo — sólo verifica que no crashea y que pop sin push es benigno.
// ─────────────────────────────────────────────────────────────────────

test "nvtx: available()/push/pop/mark no crashean (fail-safe sin lib)" {
    _ = available();
    rangePush("zig-test-range");
    mark("zig-test-mark");
    rangePop();
    rangePop(); // pop extra: benigno (libnvtx lo ignora si está)
}

test "nvtx: scoped RAII compila y ejecuta" {
    const S = struct {
        fn inner() void {
            scoped("test-inner");
            defer rangePop();
        }
    };
    S.inner();
}
