//! UC-1 (TODO_CUDA.md, lane-cuda): NVRTC JIT runtime — compilación de
//! kernels `.cu` EN RUNTIME (2-5s) como alternativa al cubin build-time de
//! nvcc (~3 min/rebuild medido). Objetivo: loop de iteración kernel
//! edit→run sin rebuild del binario.
//!
//! Patrón del repo (cudaz_stub.zig): bindings extern "c" directos a
//! libnvrtc, SIN @cImport. Estilo de opciones/log adaptado del estudio del
//! repo externo coderonion/zcuda (MIT): src/nvrtc/{sys,result,safe}.zig
//! — atribución aquí, cero dependencia.
//!
//! Gate de adopción: env `ZIG_AI_NVRTC=1` (default OFF, fallback cubin).
//!
//! Trampa documentada (UC-1.4): NVRTC NO trae los headers del SDK — no
//! resuelve `#include <cuda_runtime.h>` etc. Los fuentes que se compilen
//! JIT deben ser self-contained o recibir include-dirs explícitos
//! (`CompileOptions.include_dirs`); ver `src/cuda/nvrtc_kernels.cu`
//! (ficheros JIT dedicados, sin includes del SDK).
const std = @import("std");
const cudaz = @import("cudaz");
const debugz = @import("debug");
const timez = @import("time");

pub const Error = error{
    OutOfMemory,
    ProgramCreationFailure,
    InvalidInput,
    InvalidProgram,
    InvalidOption,
    Compilation,
    BuiltinOperationFailure,
    TheProgramCountMismatch,
    InvalidLtoFile,
    InternalError,
    NvrtcUnavailable,
};

/// nvrtcResult (nvrtc.h) — códigos que usa este módulo.
const Result = enum(c_int) {
    SUCCESS = 0,
    OUT_OF_MEMORY = 1,
    PROGRAM_CREATION_FAILURE = 2,
    INVALID_INPUT = 3,
    INVALID_PROGRAM = 4,
    INVALID_OPTION = 5,
    COMPILATION = 6,
    BUILTIN_OPERATION_FAILURE = 7,
    NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
    NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
    NAME_EXPRESSION_NOT_VALID = 10,
    INTERNAL_ERROR = 11,
    THE_PROGRAM_COUNT_MISMATCH = 12,
    INVALID_ARCH = 13,
    INVALID_LTO_FILE = 14,
};

fn toError(res: Result) Error {
    return switch (res) {
        .OUT_OF_MEMORY => error.OutOfMemory,
        .PROGRAM_CREATION_FAILURE => error.ProgramCreationFailure,
        .INVALID_INPUT => error.InvalidInput,
        .INVALID_PROGRAM => error.InvalidProgram,
        .INVALID_OPTION => error.InvalidOption,
        .COMPILATION => error.Compilation,
        .BUILTIN_OPERATION_FAILURE => error.BuiltinOperationFailure,
        .THE_PROGRAM_COUNT_MISMATCH => error.TheProgramCountMismatch,
        .INVALID_LTO_FILE => error.InvalidLtoFile,
        .SUCCESS, .NO_NAME_EXPRESSIONS_AFTER_COMPILATION, .NO_LOWERED_NAMES_BEFORE_COMPILATION, .NAME_EXPRESSION_NOT_VALID, .INVALID_ARCH, .INTERNAL_ERROR => error.InternalError,
    };
}

fn resName(res: Result) []const u8 {
    return switch (res) {
        .SUCCESS => "SUCCESS",
        .OUT_OF_MEMORY => "OUT_OF_MEMORY",
        .PROGRAM_CREATION_FAILURE => "PROGRAM_CREATION_FAILURE",
        .INVALID_INPUT => "INVALID_INPUT",
        .INVALID_PROGRAM => "INVALID_PROGRAM",
        .INVALID_OPTION => "INVALID_OPTION",
        .COMPILATION => "COMPILATION",
        .BUILTIN_OPERATION_FAILURE => "BUILTIN_OPERATION_FAILURE",
        .NO_NAME_EXPRESSIONS_AFTER_COMPILATION => "NO_NAME_EXPRESSIONS_AFTER_COMPILATION",
        .NO_LOWERED_NAMES_BEFORE_COMPILATION => "NO_LOWERED_NAMES_BEFORE_COMPILATION",
        .NAME_EXPRESSION_NOT_VALID => "NAME_EXPRESSION_NOT_VALID",
        .INTERNAL_ERROR => "INTERNAL_ERROR",
        .THE_PROGRAM_COUNT_MISMATCH => "THE_PROGRAM_COUNT_MISMATCH",
        .INVALID_ARCH => "INVALID_ARCH",
        .INVALID_LTO_FILE => "INVALID_LTO_FILE",
    };
}

/// Bindings extern "c" a libnvrtc.so (CUDA 12: símbolos sin sufijo).
const nvrtclib = struct {
    extern "c" fn nvrtcVersion(major: *c_int, minor: *c_int) Result;
    extern "c" fn nvrtcCreateProgram(prog: *?*anyopaque, src: [*:0]const u8, name: ?[*:0]const u8, num_headers: c_int, headers: ?*const ?[*:0]const u8, include_names: ?*const ?[*:0]const u8) Result;
    extern "c" fn nvrtcDestroyProgram(prog: *?*anyopaque) Result;
    extern "c" fn nvrtcCompileProgram(prog: ?*anyopaque, num_options: c_int, options: ?[*]const ?[*:0]const u8) Result;
    extern "c" fn nvrtcGetErrorString(res: Result) ?[*:0]const u8;
    extern "c" fn nvrtcGetProgramLogSize(prog: ?*anyopaque, size: *usize) Result;
    extern "c" fn nvrtcGetProgramLog(prog: ?*anyopaque, log: [*]u8) Result;
    extern "c" fn nvrtcGetPTXSize(prog: ?*anyopaque, size: *usize) Result;
    extern "c" fn nvrtcGetPTX(prog: ?*anyopaque, ptx: [*]u8) Result;
    extern "c" fn nvrtcGetCUBINSize(prog: ?*anyopaque, size: *usize) Result;
    extern "c" fn nvrtcGetCUBIN(prog: ?*anyopaque, cubin: [*]u8) Result;
};

/// ¿Está libnvrtc disponible en este proceso? El primer call cachea.
/// CI sin toolkit (noop-stub): devuelve false y el módulo degrada a error
/// claro — el binario no debe romper el link (UC-1.2: link opcional).
pub fn available() bool {
    const S = struct {
        var probe: ?bool = null;
    };
    if (S.probe) |p| return p;
    var major: c_int = 0;
    var minor: c_int = 0;
    const ok = nvrtclib.nvrtcVersion(&major, &minor) == .SUCCESS;
    S.probe = ok;
    if (ok) debugz.dbg.printLevel(.info, "[nvrtc] libnvrtc {d}.{d} detectada\n", .{ major, minor });
    return ok;
}

pub const Version = struct { major: i32, minor: i32 };

pub fn getVersion() !Version {
    var major: c_int = 0;
    var minor: c_int = 0;
    const res = nvrtclib.nvrtcVersion(&major, &minor);
    if (res != .SUCCESS) return toError(res);
    return .{ .major = major, .minor = minor };
}

/// Opciones de compilación NVRTC (espejo de las flags nvcc que usamos en
/// build.zig para cubins: -arch=sm_XX -O3). `include_dirs` inyecta
/// `-I` por entrada (UC-1.4: sin esto, `#include` local falla). CUDA 12.0
/// NVRTC usa `-I`; `--include-dir=` llegó después (verificado InvalidOption).
pub const CompileOptions = struct {
    /// Target: "sm_XX" para CUBIN, "compute_XX" para PTX. null = default
    /// del toolkit (NO recomendado: puede no matchear la GPU del host).
    /// La arch del host se detecta en runtime (cuDeviceComputeCapability) —
    /// ver tryJitOrFallback; nunca hardcodear una arch concreta.
    arch: ?[]const u8 = null,
    /// Nivel de optimización 0-3 (default nvrtc: 3).
    opt_level: u8 = 3,
    /// --device-debug (kernels de diagnóstico).
    debug: bool = false,
    /// --generate-line-info (nsys/cuda-gdb source correlation).
    lineinfo: bool = false,
    /// --use_fast_math (espejo de los dequant .o de build.zig).
    fast_math: bool = false,
    /// --maxrregcount (spill tradeoff en kernels saturados).
    maxrregcount: ?u32 = null,
    /// Include-dirs extra para `#include` locales ("kernels", "src/cuda").
    include_dirs: []const []const u8 = &.{},
    /// Flags extra crudos (--std=c++17, -DFOO=1, ...).
    extra_flags: []const []const u8 = &.{},

    fn fmtZ(allocator: std.mem.Allocator, comptime fmt: []const u8, args: anytype) ![:0]u8 {
        const str = try std.fmt.allocPrint(allocator, fmt, args);
        defer allocator.free(str);
        return try allocator.dupeZ(u8, str);
    }

    /// Lista de opciones dupeZ-eadas para nvrtcCompileProgram. El caller
    /// libera cada string + la lista (ver buildOptions decompile fns).
    fn buildOptions(self: CompileOptions, allocator: std.mem.Allocator) !std.ArrayListUnmanaged([:0]u8) {
        var options = std.ArrayListUnmanaged([:0]u8).empty;
        errdefer {
            for (options.items) |opt| allocator.free(opt);
            options.deinit(allocator);
        }
        if (self.arch) |arch| {
            try options.append(allocator, try fmtZ(allocator, "--gpu-architecture={s}", .{arch}));
        }
        if (self.opt_level > 0 and self.opt_level != 3) {
            try options.append(allocator, try fmtZ(allocator, "--opt-level={d}", .{self.opt_level}));
        }
        if (self.debug) try options.append(allocator, try allocator.dupeZ(u8, "--device-debug"));
        if (self.lineinfo) try options.append(allocator, try allocator.dupeZ(u8, "--generate-line-info"));
        if (self.fast_math) try options.append(allocator, try allocator.dupeZ(u8, "--use_fast_math"));
        if (self.maxrregcount) |count| {
            try options.append(allocator, try fmtZ(allocator, "--maxrregcount={d}", .{count}));
        }
        for (self.include_dirs) |dir| {
            try options.append(allocator, try fmtZ(allocator, "-I{s}", .{dir}));
        }
        for (self.extra_flags) |flag| {
            try options.append(allocator, try allocator.dupeZ(u8, flag));
        }
        return options;
    }
};

fn freeOptions(allocator: std.mem.Allocator, options: *std.ArrayListUnmanaged([:0]u8)) void {
    for (options.items) |opt| allocator.free(opt);
    options.deinit(allocator);
}

const Program = struct {
    handle: ?*anyopaque,

    fn create(allocator: std.mem.Allocator, src: []const u8, name: ?[]const u8) !Program {
        const src_z = try allocator.dupeZ(u8, src);
        defer allocator.free(src_z);
        const name_z: ?[*:0]const u8 = if (name) |n| (try allocator.dupeZ(u8, n)).ptr else null;
        defer if (name_z) |p| allocator.free(p[0..strlen(p) :0]);
        var prog: ?*anyopaque = null;
        const res = nvrtclib.nvrtcCreateProgram(&prog, src_z.ptr, name_z, 0, null, null);
        if (res != .SUCCESS) return toError(res);
        return .{ .handle = prog };
    }

    fn destroy(self: *Program) void {
        _ = nvrtclib.nvrtcDestroyProgram(&self.handle);
    }

    fn compile(self: *const Program, allocator: std.mem.Allocator, options: CompileOptions) !void {
        var opts = try options.buildOptions(allocator);
        defer freeOptions(allocator, &opts);

        const opts_ptrs = try allocator.alloc([*:0]const u8, opts.items.len);
        defer allocator.free(opts_ptrs);
        for (opts.items, 0..) |opt, i| opts_ptrs[i] = opt.ptr;

        const res = nvrtclib.nvrtcCompileProgram(self.handle, @intCast(opts_ptrs.len), opts_ptrs.ptr);
        if (res != .SUCCESS) {
            // Log legible (requisito UC-1.1): el program-log de nvrtc trae
            // las líneas del fuente + columna del error, como nvcc.
            dumpLog(self.handle, allocator);
            return toError(res);
        }
    }

    fn dumpLog(handle: ?*anyopaque, allocator: std.mem.Allocator) void {
        var log_size: usize = 0;
        if (nvrtclib.nvrtcGetProgramLogSize(handle, &log_size) != .SUCCESS) return;
        if (log_size <= 1) return;
        const log = allocator.alloc(u8, log_size) catch return;
        defer allocator.free(log);
        if (nvrtclib.nvrtcGetProgramLog(handle, log.ptr) != .SUCCESS) return;
        debugz.dbg.print("[gpu_kernels] compile error:\n{s}", .{log[0 .. log_size - 1]});
    }

    fn getBlob(self: *const Program, allocator: std.mem.Allocator, comptime kind: enum { ptx, cubin }) ![]u8 {
        var size: usize = 0;
        const size_res = switch (kind) {
            .ptx => nvrtclib.nvrtcGetPTXSize(self.handle, &size),
            .cubin => nvrtclib.nvrtcGetCUBINSize(self.handle, &size),
        };
        if (size_res != .SUCCESS) return toError(size_res);
        const blob = try allocator.alloc(u8, size);
        errdefer allocator.free(blob);
        const get_res = switch (kind) {
            .ptx => nvrtclib.nvrtcGetPTX(self.handle, blob.ptr),
            .cubin => nvrtclib.nvrtcGetCUBIN(self.handle, blob.ptr),
        };
        if (get_res != .SUCCESS) return toError(get_res);
        return blob;
    }
};

fn strlen(p: [*:0]const u8) usize {
    return std.mem.len(p);
}

/// Compila fuente CUDA C++ a PTX (compute_XX). Cargable con
/// cuModuleLoadData — el driver hace JIT final al contexto actual.
pub fn compilePtx(allocator: std.mem.Allocator, src: []const u8, options: CompileOptions) ![]u8 {
    if (!available()) return error.NvrtcUnavailable;
    var prog = try Program.create(allocator, src, null);
    defer prog.destroy();
    try prog.compile(allocator, options);
    return prog.getBlob(allocator, .ptx);
}

/// Compila fuente CUDA C++ a CUBIN nativo (sm_XX — arch OBLIGATORIA y debe
/// ser la GPU del host para que el driver lo cargue sin JIT extra).
pub fn compileCubin(allocator: std.mem.Allocator, src: []const u8, options: CompileOptions) ![]u8 {
    if (!available()) return error.NvrtcUnavailable;
    var prog = try Program.create(allocator, src, null);
    defer prog.destroy();
    try prog.compile(allocator, options);
    return prog.getBlob(allocator, .cubin);
}

// ─────────────────────────────────────────────────────────────────────────────
// UC-1.3: integración con el camino de carga existente (loadModule).
// nvrtcLoadKernel compila+carga el fuente .cu por JIT y devuelve el CUmodule
// — MISMO tipo que cuModuleLoad(cubin_path): los launchers (cuModuleGetFunction
// + cuLaunchKernel) NO cambian.
// ─────────────────────────────────────────────────────────────────────────────

/// Fuente y opciones del módulo JIT activo (gate ZIG_AI_NVRTC). Único por
/// proceso; los módulos JIT dedicados viven en nvrtc_kernels.cu.
pub const JitModule = struct {
    /// Fuente CUDA C++ self-contained (comptime embed del .cu dedicado).
    src: []const u8,
    /// Nombre lógico (breadcrumb).
    name: []const u8,
    /// Arch target "sm_XX" (null = auto-detect del device actual; cubin
    /// JIT la requiere — nunca hardcodear).
    arch: ?[]const u8 = null,
    /// Include-dirs (para .cu que usan headers locales; ver UC-1.4).
    include_dirs: []const []const u8 = &.{},
    /// Resto de opciones.
    options: CompileOptions = .{},
};

/// Cache de módulos JIT: (nombre → CUmodule). El proceso compila una vez.
var g_modules: std.StringHashMapUnmanaged(cudaz.CUmodule) = .empty;
var g_modules_gpa: ?std.mem.Allocator = null;

/// Inicializa la cache (llamar una vez, idealmente desde el init del engine).
pub fn initCache(allocator: std.mem.Allocator) void {
    if (g_modules_gpa == null) g_modules_gpa = allocator;
}

/// Compila y carga un módulo JIT (cubin nativo para la arch actual). La
/// cache evita recompilar en el mismo proceso. Devuelve el CUmodule — el
/// mismo camino que loadModule() de cubins (launchers sin cambios).
pub fn loadJitModule(allocator: std.mem.Allocator, spec: JitModule) !cudaz.CUmodule {
    if (g_modules.get(spec.name)) |m| return m;
    if (!available()) return error.NvrtcUnavailable;

    try cudaz.ensureContext();

    var opts = spec.options;
    if (spec.arch) |a| opts.arch = a;
    opts.include_dirs = spec.include_dirs;

    const cubin = try compileCubin(allocator, spec.src, opts);
    defer allocator.free(cubin);

    const module = try cudaz.cuModuleLoadData(cubin);

    // Cache bajo el allocator del proceso (gpa del engine), no el del
    // caller transitorio — la CUmodule vive hasta el exit.
    const gpa = g_modules_gpa orelse allocator;
    const key = try gpa.dupe(u8, spec.name);
    try g_modules.put(gpa, key, module);
    if (debugz.dbg.at(.info)) {
        debugz.dbg.printLevel(.info, "[nvrtc] módulo '{s}' JIT: {d} bytes cubin, arch={s}\n", .{ spec.name, cubin.len, spec.arch orelse "?" });
    }

    return module;
}

/// ¿El gate ZIG_AI_NVRTC está activo? (default OFF — fallback cubin).
pub fn gateEnabled() bool {
    const S = struct {
        var cached: ?bool = null;
    };
    if (S.cached) |c| return c;
    const v = std.c.getenv("ZIG_AI_NVRTC");
    const on = if (v) |p| std.mem.eql(u8, std.mem.span(p), "1") else false;
    S.cached = on;
    if (on) debugz.dbg.printLevel(.info, "[nvrtc] gate ZIG_AI_NVRTC=1 activo — módulos JIT en vez de cubins\n", .{});
    return on;
}

/// UC-1.4: lee un fuente .cu del disco para JIT. Patrón del repo
/// (hybrid_attn.zig:1369): Io global single-threaded, sin acoplar el
/// allocator del engine. Buffer a liberar por el caller.
pub fn readSrcFile(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = std.Io.Dir.cwd();
    return dir.readFileAlloc(io, path, allocator, .unlimited);
}

/// UC-1.6 (2026-09-14): cache DISCO de cubins JIT — convierte el JIT en
/// herramienta de iteración práctica: 8.4s compile SOLO la primera vez
/// tras editar el .cu; runs consecutivos con fuente sin cambios cargan el
/// cubin cacheado (~0s). Ruta: $XDG_CACHE_HOME/zig-ai/nvrtc/<name>-<hash8>.cubin
/// (hash = fuente + arch + flags — cambiar CUALQUIERA invalida solo esa
/// entrada). Corrupto/truncado ⇒ fallback silencioso a compile. Se
/// desactiva con ZIG_AI_NVRTC_CACHE=0 (A/B/debug).
fn diskCacheDir(buf: *[512]u8) ?[]const u8 {
    // ZIG_AI_NVRTC_CACHE=0 lo desactiva (A/B/debug).
    if (std.c.getenv("ZIG_AI_NVRTC_CACHE")) |v| {
        if (std.mem.eql(u8, std.mem.span(v), "0")) return null;
    }
    const home_env = std.c.getenv("XDG_CACHE_HOME");
    const home = std.c.getenv("HOME") orelse return null;
    const dir_fmt = if (home_env) |x| std.fmt.bufPrint(buf, "{s}/zig-ai/nvrtc", .{std.mem.span(x)}) else std.fmt.bufPrint(buf, "{s}/.cache/zig-ai/nvrtc", .{home});
    const dir = dir_fmt catch return null;
    // mkdir -p vía libc (patrón hf_download.zig:87 — Io.Dir no expone makePath)
    mkdirP(dir);
    return dir;
}

/// unlink vía libc (patrón cpu_executor.zig:53).
extern "c" fn unlink(path: [*:0]const u8) i32;

/// mkdir -p vía libc (copiado del patrón hf_download.zig — Io.Dir 0.16 no
/// expone makePath; libc ya linkada).
fn mkdirP(path: []const u8) void {
    var buf: [512]u8 = undefined;
    if (path.len >= buf.len) return;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    var i: usize = 1;
    while (i <= path.len) : (i += 1) {
        if (buf[i] == '/' or buf[i] == 0) {
            const save = buf[i];
            buf[i] = 0;
            _ = std.c.mkdir(@ptrCast(buf[0..].ptr), 0o755);
            buf[i] = save;
        }
    }
}

/// Lee un cubin cacheado (si existe y pesa >0). Caller libera.
fn diskCacheRead(allocator: std.mem.Allocator, dir: []const u8, name: []const u8, hash8: u64) ?[]u8 {
    var pbuf: [576]u8 = undefined;
    const path = std.fmt.bufPrint(&pbuf, "{s}/{s}-{x:0>8}.cubin", .{ dir, name, hash8 }) catch return null;
    const io = std.Io.Threaded.global_single_threaded.io();
    const data = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited) catch return null;
    if (data.len == 0) {
        allocator.free(data);
        return null;
    }
    return data;
}

/// Escribe el cubin al cache (best-effort; fallo = no-op silencioso).
/// Patrón writer del repo (audit.zig:63): createFile + writer + flush.
fn diskCacheWrite(dir: []const u8, name: []const u8, hash8: u64, cubin: []const u8) void {
    var pbuf: [576]u8 = undefined;
    const path = std.fmt.bufPrint(&pbuf, "{s}/{s}-{x:0>8}.cubin", .{ dir, name, hash8 }) catch return;
    const io = std.Io.Threaded.global_single_threaded.io();
    const f = std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true }) catch return;
    defer f.close(io);
    var wbuf: [4096]u8 = undefined;
    var fw = f.writer(io, &wbuf);
    const w = &fw.interface;
    w.writeAll(cubin) catch return;
    w.flush() catch return;
}

/// UC-1.3: wrapper de integración para los loadModule() existentes.
/// Gate ZIG_AI_NVRTC=1 + fichero fuente presente ⇒ compila el .cu por JIT
/// y devuelve el CUmodule por cuModuleLoadData (los launchers no cambian).
/// Cualquier fallo (sin libnvrtc, error de compile, sin fuente) degrada al
/// CAMINO ORIGINAL transparentemente — el caller pasa el fallback como
/// closure para mantener su lógica de cubin_path/build_options local.
///
/// Uso (patrón layer_kernels.zig):
///   g_module = nvrtc.tryJitOrFallback(allocator, .{
///       .name = "layer_kernels",
///       .src_path = "src/cuda/layer_kernels.cu",
///       .include_dirs = &.{"src/cuda"},
///   }, cubin_path) catch |e| switch (e) {
///       error.CudaUnavailable, error.CudaError => ... // camino original
///   };
pub fn tryJitOrFallback(allocator: std.mem.Allocator, spec: JitModuleSpec, cubin_path: []const u8) !cudaz.CUmodule {
    _ = cubin_path; // solo informativo en el breadcrumb (fallback lo maneja el caller)
    if (!gateEnabled()) return error.NvrtcDisabled;
    if (!available()) return error.NvrtcUnavailable;

    // Cache por nombre: un proceso compila cada módulo UNA vez (los
    // loadModule() se llaman por primera referencia de cada kernel).
    if (g_modules.get(spec.name)) |m| return m;

    const src = readSrcFile(allocator, spec.src_path) catch return error.SourceUnavailable;
    defer allocator.free(src);

    var arch_buf: [16]u8 = undefined;
    const arch = spec.arch orelse blk: {
        // Detección runtime (NUNCA hardcode): capability del device 0 vía
        // la API del repo (cudaz.cuDeviceInfo — misma que gpuArchDetect).
        const dev = try cudaz.cuDeviceGet(0);
        const info = try cudaz.cuDeviceInfo(allocator, dev);
        defer allocator.free(info.name);
        break :blk std.fmt.bufPrint(&arch_buf, "sm_{d}{d}", .{ info.major, info.minor }) catch unreachable;
    };

    // UC-1.6: hash fuente+arch+flags → clave del cache disco. Wyhash del
    // repo (rápido, sin dependencias); incluir arch (mismo fuente en otra
    // GPU = otro cubin) y knobs que afectan código generado.
    const hash8 = std.hash.Wyhash.hash(0, src) ^ (std.hash.Wyhash.hash(1, arch) << 32) ^
        (if (spec.fast_math) @as(u64, 0x5A17) else 0) ^
        (if (spec.maxrregcount) |r| @as(u64, r) << 40 else 0);

    var dir_buf: [512]u8 = undefined;
    if (diskCacheDir(&dir_buf)) |dcache| {
        if (diskCacheRead(allocator, dcache, spec.name, hash8)) |cached| {
            defer allocator.free(cached);
            const module = cudaz.cuModuleLoadData(cached) catch {
                // Cubin corrupto/obsoleto (formato driver cambió): borra y
                // sigue al compile normal. Nunca fatal.
                var pbuf: [576]u8 = undefined;
                if (std.fmt.bufPrint(&pbuf, "{s}/{s}-{x:0>8}.cubin", .{ dcache, spec.name, hash8 })) |p| {
                    var zbuf: [576]u8 = undefined;
                    if (p.len < zbuf.len) {
                        @memcpy(zbuf[0..p.len], p);
                        zbuf[p.len] = 0;
                        _ = unlink(@ptrCast(&zbuf));
                    }
                } else |_| {}
                return tryJitOrFallbackCompile(allocator, spec, src, arch, hash8, dcache);
            };
            const gpa0 = g_modules_gpa orelse allocator;
            const key0 = try gpa0.dupe(u8, spec.name);
            try g_modules.put(gpa0, key0, module);
            if (debugz.dbg.at(.info)) {
                debugz.dbg.printLevel(.info, "[nvrtc] '{s}' cache DISCO hit ({d} bytes, {s}) — 0 ms compile\n", .{ spec.name, cached.len, arch });
            }
            return module;
        }
    }

    return tryJitOrFallbackCompile(allocator, spec, src, arch, hash8, if (diskCacheDir(&dir_buf)) |d| d else "");
}

/// Camino compile (UC-1.3 original + escritura al cache disco UC-1.6).
fn tryJitOrFallbackCompile(
    allocator: std.mem.Allocator,
    spec: JitModuleSpec,
    src: []const u8,
    arch: []const u8,
    hash8: u64,
    dcache: []const u8,
) !cudaz.CUmodule {
    const t0 = timez.Timer.start();
    const cubin = try compileCubin(allocator, src, .{
        .arch = arch,
        .include_dirs = spec.include_dirs,
        .fast_math = spec.fast_math,
        .maxrregcount = spec.maxrregcount,
        // Receta UC-1.4 (validada por sondas de tests/test_nvrtc_gpu.zig):
        //  - -default-device: fuentes nvcc-style tratan lo sin-anotar como
        //    __device__ (sin esto, cuda_fp16.h declara funciones "host").
        //  - -D__x86_64__: glibc multiarch (stdint.h→gnu/stubs.h) elige
        //    stubs-64.h; sin el define pide stubs-32.h (multilib ausente).
        //  - __dp4a lo resuelve nvrtc_shims/sm_61_dp4a_shim.h (causa raíz:
        //    la distro guarda el inline-asm en un .hpp que excluye
        //    __CUDACC_RTC__ ⇒ declaraciones extern sin resolver).
        //    (NOTA: --device-c parchea el síntoma pero produce .o
        //    reubicable — cuModuleLoadData lo rechaza con
        //    ERROR_INVALID_IMAGE. NO volver a usarlo.)
        .extra_flags = &.{ "-default-device", "-D__x86_64__" },
    });
    defer allocator.free(cubin);
    const dt_ms = @divTrunc(t0.read(), std.time.ns_per_ms);

    // UC-1.6: persistir best-effort (fallo de escritura NO rompe nada).
    if (dcache.len > 0) diskCacheWrite(dcache, spec.name, hash8, cubin);

    const module = try cudaz.cuModuleLoadData(cubin);

    // Cache bajo el allocator del proceso; la CUmodule vive hasta exit.
    const gpa = g_modules_gpa orelse allocator;
    const key = try gpa.dupe(u8, spec.name);
    try g_modules.put(gpa, key, module);

    if (debugz.dbg.at(.info)) {
        debugz.dbg.printLevel(.info, "[nvrtc] '{s}' JIT desde {s}: {d} ms, {d} bytes, {s} (cache disco escrito)\n", .{ spec.name, spec.src_path, dt_ms, cubin.len, arch });
    }
    return module;
}

pub const JitModuleSpec = struct {
    /// Nombre lógico del módulo (breadcrumb + clave de cache).
    name: []const u8,
    /// Ruta del .cu a compilar en runtime (relativa al cwd del proceso).
    src_path: []const u8,
    /// Include-dirs para headers locales del .cu (UC-1.4).
    include_dirs: []const []const u8 = &.{},
    /// Arch explícita (tests A/B); null = auto-detect del device actual —
    /// NUNCA hardcodear una arch concreta en código de producción.
    arch: ?[]const u8 = null,
    fast_math: bool = true,
    maxrregcount: ?u32 = null,
};

/// UC-1.4: ruta del fuente JIT de layer_kernels. El engine corre con cwd
/// variable (zig-out/bin, repo root, tests) — la env ZIG_AI_NVRTC_SRC
/// permite fijarla en setups no estándar; default relativo al repo.
pub fn srcPath() []const u8 {
    const S = struct {
        var cached: ?[]const u8 = null;
    };
    if (S.cached) |p| return p;
    const env = std.c.getenv("ZIG_AI_NVRTC_SRC");
    const p: []const u8 = if (env) |e| std.mem.span(e) else "src/cuda/layer_kernels.cu";
    S.cached = p;
    return p;
}

/// UC-1.4: include-dirs que NVRTC necesita para los headers del SDK que
/// nuestros .cu usan (cuda_fp16.h, mma.h...). NVRTC NO trae headers: se
/// inyecta el include-dir del sistema (distro CUDA: /usr/include — el host
/// tiene nvrtc.h/cuda_fp16.h/mma.h ahí) + src/cuda para headers locales
/// (kvarn_desc.cuh, mma_kvarn.cuh). Override: ZIG_AI_NVRTC_INC (CSV).
pub fn sdkIncludeDirs() []const []const u8 {
    const S = struct {
        var dirs: [8][]const u8 = undefined;
        var buffers: [8][512]u8 = undefined; // backing de rutas absolutas
        var n: usize = 0;
        var inited = false;
    };
    if (S.inited) return S.dirs[0..S.n];
    S.n = 0;
    if (std.c.getenv("ZIG_AI_NVRTC_INC")) |env| {
        // CSV de override: "/usr/include,/opt/cuda/include"
        var it = std.mem.splitScalar(u8, std.mem.span(env), ',');
        while (it.next()) |d| {
            if (S.n >= S.dirs.len) break;
            if (d.len > 0) {
                S.dirs[S.n] = d;
                S.n += 1;
            }
        }
    } else {
        // ORDEN = precedencia (NVRTC busca en orden; el PRIMERO gana).
        // RUTAS ABSOLUTAS obligatorias: NVRTC resuelve include-dirs contra
        // SU cwd, no el del proceso (verificado: rutas relativas no se
        // encontraban — sonda UC-1.4 de tests/test_nvrtc_gpu.zig).
        S.dirs[S.n] = absPathOr(&S.buffers[S.n], "src/cuda/nvrtc_shims"); // shims device-safe (stdio.h→printf no-op)
        S.n += 1;
        S.dirs[S.n] = "/usr/include"; // distro CUDA (cuda_fp16.h/mma.h/nvrtc.h)
        S.n += 1;
        S.dirs[S.n] = "/usr/include/x86_64-linux-gnu"; // glibc multiarch: stdint.h→bits/→gnu/stubs.h
        S.n += 1;
        S.dirs[S.n] = "/usr/lib/gcc/x86_64-linux-gnu/13/include"; // headers de compilador: stddef.h (stdio.h lo pide)
        S.n += 1;
        S.dirs[S.n] = absPathOr(&S.buffers[S.n], "src/cuda"); // headers locales del repo
        S.n += 1;
    }
    S.inited = true;
    return S.dirs[0..S.n];
}

/// realpath() de libc: ruta relativa → absoluta canonical. Si falla (no
/// existe/cwd raro), devuelve la relativa original (mejor-esfuerzo — el
/// caller JIT degradará al fallback cubin con log claro).
fn absPathOr(buf: *[512]u8, rel: []const u8) []const u8 {
    if (rel.len >= buf.len) return rel;
    @memcpy(buf[0..rel.len], rel);
    buf[rel.len] = 0;
    const resolved = std.c.realpath(buf[0..rel.len :0], buf);
    if (resolved) |r| return std.mem.span(r);
    return buf[0..rel.len];
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests unitarios (sin GPU): opciones, error-mapping, gate. GPU tests en
// tests/test_nvrtc_gpu.zig (paridad JIT vs cubin — UC-1.5).
// ─────────────────────────────────────────────────────────────────────────────

test "CompileOptions: buildOptions serializa flags esperadas" {
    const alloc = std.testing.allocator;
    // Arch SINTÉTICA: este test valida la serialización de flags, no targeting
    // — la arch REAL se detecta en runtime (cuDeviceComputeCapability →
    // tryJitOrFallback/test helper); nunca hardcodear sm_XX de un host dado.
    const opts = CompileOptions{
        .arch = "sm_TEST",
        .opt_level = 2,
        .fast_math = true,
        .maxrregcount = 64,
        .include_dirs = &.{"kernels"},
        .extra_flags = &.{"--std=c++17"},
    };
    var built = try opts.buildOptions(alloc);
    defer freeOptions(alloc, &built);

    try std.testing.expectEqual(@as(usize, 5), built.items.len);
    try std.testing.expectEqualStrings("--gpu-architecture=sm_TEST", built.items[0]);
    try std.testing.expectEqualStrings("--opt-level=2", built.items[1]);
    try std.testing.expectEqualStrings("--use_fast_math", built.items[2]);
    try std.testing.expectEqualStrings("--maxrregcount=64", built.items[3]);
    try std.testing.expectEqualStrings("-Ikernels", built.items[4]);
    // opt_level 3 (default) NO añade flag; extra_flags va al final.
    const def = CompileOptions{ .arch = null };
    var built2 = try def.buildOptions(alloc);
    defer freeOptions(alloc, &built2);
    try std.testing.expectEqual(@as(usize, 0), built2.items.len);
}

test "toError: mapping nvrtcResult" {
    try std.testing.expectError(error.OutOfMemory, toError(.OUT_OF_MEMORY));
    try std.testing.expectError(error.Compilation, toError(.COMPILATION));
    try std.testing.expectError(error.InternalError, toError(.INTERNAL_ERROR));
}
test "gateEnabled: default OFF sin env" {
    // Sin ZIG_AI_NVRTC en el entorno del test runner ⇒ false. (Si un dev
    // corre los tests con la env puesta, el test NO valida "sin env" —
    // aceptable: es un smoke de wiring, no un contrato.)
    if (std.c.getenv("ZIG_AI_NVRTC") != null) return error.SkipZigTest;
    try std.testing.expect(!gateEnabled());
}
