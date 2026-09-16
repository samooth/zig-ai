//! UC-1.5 (TODO_CUDA.md, lane-cuda): paridad del camino NVRTC JIT vs cubin
//! build-time del MISMO kernel (argmax) + smoke del ErrorFlag device-side
//! (UC-2 pilot en el fichero JIT). Requiere CUDA + libnvrtc; sin GPU o sin
//! libnvrtc sale limpio (SkipZigTest).
//!
//!   zig build test-nvrtc
//!
//! Gate de adopción UC-1 (demo loop iter <10s): el smoke JIT compila
//! nvrtc_kernels.cu en runtime y lo lanza — medir la compilación es parte
//! del test (breadcrumb [nvrtc]).
const std = @import("std");
const cudaz = @import("cudaz");
const matmul = @import("matmul");
const layer_kernels = @import("layer_kernels");
const nvrtc = @import("nvrtc");
const debugz = @import("debug");
const timez = @import("time");

/// Fuente JIT self-contained embebida en el test (mismo kernel que
/// nvrtc_kernels.cu — la copia es deliberada: valida que un fuente EDITADO
/// en runtime compila y produce el MISMO resultado que el cubin frozen).
const argmax_src =
    \\#define WARP 32
    \\__device__ __forceinline__ bool jitArgmaxWins(float val, int col, float maxval, int argmax) {
    \\    if (val > maxval) return true;
    \\    if (val == maxval && (argmax < 0 || col < argmax)) return true;
    \\    return false;
    \\}
    \\extern "C" __global__ void argmaxJitKernel(const float* __restrict__ x, int* __restrict__ dst, int ncols) {
    \\    const int row = blockIdx.x;
    \\    const int tid = threadIdx.x;
    \\    const int block = blockDim.x;
    \\    const float* rowp = x + (size_t)row * ncols;
    \\    float maxval = -1.70141183e38f;
    \\    int   argmax = -1;
    \\    for (int col = tid; col < ncols; col += block) {
    \\        const float val = rowp[col];
    \\        if (jitArgmaxWins(val, col, maxval, argmax)) { maxval = val; argmax = col; }
    \\    }
    \\    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
    \\        const float val = __shfl_xor_sync(0xffffffffu, maxval, offset, WARP);
    \\        const int   col = __shfl_xor_sync(0xffffffffu, argmax, offset, WARP);
    \\        if (jitArgmaxWins(val, col, maxval, argmax)) { maxval = val; argmax = col; }
    \\    }
    \\    const int nwarps = (block + WARP - 1) / WARP;
    \\    if (nwarps == 1) {
    \\        if (tid == 0) dst[row] = argmax;
    \\        return;
    \\    }
    \\    __shared__ float shared_max[32];
    \\    __shared__ int   shared_argmax[32];
    \\    const int warp_id = tid / WARP;
    \\    const int lane_id = tid % WARP;
    \\    if (lane_id == 0) { shared_max[warp_id] = maxval; shared_argmax[warp_id] = argmax; }
    \\    __syncthreads();
    \\    if (warp_id == 0) {
    \\        maxval = (lane_id < nwarps) ? shared_max[lane_id] : -1.70141183e38f;
    \\        argmax = (lane_id < nwarps) ? shared_argmax[lane_id] : -1;
    \\        for (int offset = WARP / 2; offset > 0; offset >>= 1) {
    \\            const float val = __shfl_xor_sync(0xffffffffu, maxval, offset, WARP);
    \\            const int   col = __shfl_xor_sync(0xffffffffu, argmax, offset, WARP);
    \\            if (jitArgmaxWins(val, col, maxval, argmax)) { maxval = val; argmax = col; }
    \\        }
    \\        if (lane_id == 0) dst[row] = argmax;
    \\    }
    \\}
;

fn sharedStream() !cudaz.CUstream {
    return @ptrCast((try matmul.MatmulEngine.sharedCudaStream()).raw);
}

/// Arch del device ACTUAL (detección runtime, nunca hardcode — la arch del
/// host puede cambiar; el repo detecta la plataforma) formateada "sm_XX"
/// para compileCubin.
fn deviceArch(allocator: std.mem.Allocator, buf: []u8) ![]const u8 {
    const dev = try cudaz.cuDeviceGet(0);
    const info = try cudaz.cuDeviceInfo(allocator, dev);
    defer allocator.free(info.name);
    return std.fmt.bufPrint(buf, "sm_{d}{d}", .{ info.major, info.minor }) catch error.OutOfMemory;
}

fn hostArgmax(row: []const f32) i32 {
    var maxval: f32 = -std.math.inf(f32);
    var argmax: i32 = -1;
    for (row, 0..) |v, i| {
        if (v > maxval) {
            maxval = v;
            argmax = @intCast(i);
        }
    }
    return argmax;
}

test "nvrtc: disponible y versión legible" {
    debugz.init();
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    if (!nvrtc.available()) return error.SkipZigTest; // sin libnvrtc: skip limpio
    const v = try nvrtc.getVersion();
    try std.testing.expect(v.major >= 12);
}

// UC-1.4 diagnóstico: qué includes del SDK resuelve NVRTC y con qué
// include-dirs (sondas de compilación aisladas, una por header crítico).
// Documenta las trampas encontradas: (a) nvrtc 12.0 NO trae stdint.h
// builtin — "no directories in search list"; (b) glibc multiarch
// (stdint.h→bits/→gnu/stubs.h) exige stubs-32.h si __x86_64__ no está
// definido — el define va en extra_flags; (c) /usr/include solo basta
// para los headers CUDA (cuda_fp16/mma).
test "nvrtc: sondas de includes del SDK (UC-1.4)" {
    debugz.init();
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    if (!nvrtc.available()) return error.SkipZigTest;

    var arch_buf: [16]u8 = undefined;
    const arch = try deviceArch(std.testing.allocator, &arch_buf);
    const glibc_dirs: []const []const u8 = &.{ "/usr/include", "/usr/include/x86_64-linux-gnu" };
    // Shim dir resuelto en runtime (realpath del árbol actual — el patrón
    // absPathOr de nvrtc.zig; antes hardcoded a un worktree de lane muerto).
    var shim_dir_buf: [512]u8 = undefined;
    {
        const rel = "src/cuda/nvrtc_shims";
        @memcpy(shim_dir_buf[0..rel.len], rel);
        shim_dir_buf[rel.len] = 0;
        if (std.c.realpath(shim_dir_buf[0..rel.len :0], &shim_dir_buf)) |_| {} else {
            @memcpy(shim_dir_buf[0..rel.len], rel);
        }
    }
    const shim_dir_len = std.mem.indexOfScalar(u8, shim_dir_buf[0..], 0) orelse shim_dir_buf.len;

    const Sonda = struct { name: []const u8, src: []const u8, dirs: []const []const u8, define_x86_64: bool = false, want_ok: bool };
    const sondas = [_]Sonda{
        // cuda_fp16 con el include del sistema: el caso base que necesitamos.
        .{ .name = "cuda_fp16 + /usr/include", .src = "#include <cuda_fp16.h>\nextern \"C\" __global__ void k(float* x){x[0]=1.f;}", .dirs = &.{"/usr/include"}, .want_ok = true },
        // stdint requiere glibc multiarch (nvrtc no lo trae builtin) +
        // __x86_64__ para que stubs.h elija stubs-64.h (sin multilib 32-bit
        // instalado, la ausencia del define rompe en stubs-32.h).
        .{ .name = "stdint + multiarch", .src = "#include <stdint.h>\nextern \"C\" __global__ void k(float* x){x[0]=1.f;}", .dirs = glibc_dirs, .define_x86_64 = true, .want_ok = true },
        // Sonda inversa (la trampa sin el define): documenta el fallo exacto.
        .{ .name = "stdint sin __x86_64__ (trampa stubs-32)", .src = "#include <stdint.h>\nextern \"C\" __global__ void k(float* x){x[0]=1.f;}", .dirs = glibc_dirs, .want_ok = false },
        // mma.h con include del sistema: necesario para los kernels WMMA.
        .{ .name = "mma.h + /usr/include", .src = "#include <mma.h>\nextern \"C\" __global__ void k(float* x){x[0]=1.f;}", .dirs = &.{"/usr/include"}, .want_ok = true },
        // stdio con el SHIM nvrtc (device-safe printf no-op) ANTES que
        // glibc — el camino que layer_kernels.cu necesita: sus printf son
        // breadcrumbs gated, no-op en JIT está bien (frontera documentada).
        // RUTA ABSOLUTA: NVRTC resuelve los include-dirs relativos contra
        // SU cwd (que puede diferir del test) — el shim con ruta relativa
        // NO se encontraba (verificado: error glibc stubs-32 persistía).
        // El path del shim se resuelve EN RUNTIME contra el árbol actual
        // (absPathOr de nvrtc.zig — realpath del repo, no hardcode).
        .{ .name = "stdio con shim nvrtc (rutas absolutas)", .src = "#include <stdio.h>\nextern \"C\" __global__ void k(float* x){printf(\"%d\", 1); x[0]=1.f;}", .dirs = &.{ shim_dir_buf[0..shim_dir_len], "/usr/include", "/usr/include/x86_64-linux-gnu", "/usr/lib/gcc/x86_64-linux-gnu/13/include" }, .want_ok = true },
        // extern __shared__ dentro de la función (patrón dynamic smem de
        // layer_kernels.cu:340): legal nvcc; valida que -default-device
        // no lo rompa en NVRTC.
        .{ .name = "extern __shared__ dinámico + -default-device", .src = "extern \"C\" __global__ void k(float* x){ extern __shared__ float sx[]; sx[threadIdx.x]=x[threadIdx.x]; x[threadIdx.x]=sx[threadIdx.x]+1.f; }", .dirs = &.{}, .want_ok = true },
        // SONDA DECISIVA: variable namespace-scope SIN anotación. Verifica
        // que -default-device cubre VARIABLES globales (no solo funciones):
        // compila con el flag — el error genérico de layer_kernels.cu que
        // sugiere el flag es OTRA cosa (ver sonda WMMA/namespace).
        .{ .name = "global sin anotar + -default-device", .src = "int g = 5;\nextern \"C\" __global__ void k(float* x){ x[0] = (float)g; }", .dirs = &.{}, .want_ok = true },
        // HALLAZGO UC-1.4 (destapado por el JIT en layer_kernels.cu):
        // `1e308f` excede el rango de f32 (FLT_MAX≈3.4e38). nvcc lo acepta
        // con warning (→inf); NVRTC lo rechaza. 18 ocurrencias en
        // layer_kernels.cu (mn/mx init de diagnóstico) — fix de lane-a:
        // FLT_MAX (o INFINITY). Sonda que documenta la clase de fallo.
        .{ .name = "1e308f fuera de rango f32 (bug latente)", .src = "extern \"C\" __global__ void k(float* x){ float mn = 1e308f; x[0] = mn; }", .dirs = &.{}, .want_ok = false },
    };
    for (sondas) |s| {
        var flags: [2][]const u8 = undefined;
        var extra: []const []const u8 = &.{"-default-device"};
        if (s.define_x86_64) {
            flags[0] = "-default-device";
            flags[1] = "-D__x86_64__";
            extra = flags[0..2];
        }
        const r = nvrtc.compileCubin(std.testing.allocator, s.src, .{ .arch = arch, .include_dirs = s.dirs, .extra_flags = extra });
        if (r) |cub| {
            defer std.testing.allocator.free(cub);
            if (debugz.dbg.at(.info)) debugz.dbg.printLevel(.info, "[nvrtc] sonda '{s}': OK ({d} bytes)\n", .{ s.name, cub.len });
            if (!s.want_ok) return testFail("sonda '{s}' compiló pero se esperaba fallo (trampa)", .{s.name});
        } else |e| {
            if (debugz.dbg.at(.info)) debugz.dbg.printLevel(.info, "[nvrtc] sonda '{s}': {s}\n", .{ s.name, @errorName(e) });
            if (s.want_ok) return testFail("sonda '{s}' falló inesperadamente: {s}", .{ s.name, @errorName(e) });
        }
    }
}

fn testFail(comptime fmt: []const u8, args: anytype) anyerror {
    debugz.dbg.printLevel(.info, "[nvrtc] FAIL: " ++ fmt ++ "\n", args);
    return error.NvrtcSondaFailed;
}

test "nvrtc: JIT argmax paridad bit-exact vs cubin build-time" {
    debugz.init();
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    if (!nvrtc.available()) return error.SkipZigTest;

    // Arch del device actual (cubin JIT sólo carga en arch match).
    var arch_buf: [16]u8 = undefined;
    const arch = try deviceArch(std.testing.allocator, &arch_buf);

    // Compilación JIT (el tiempo ES el dato del gate de adopción).
    const t0 = timez.Timer.start();
    const cubin = try nvrtc.compileCubin(std.testing.allocator, argmax_src, .{
        .arch = arch,
        .fast_math = true,
    });
    defer std.testing.allocator.free(cubin);
    const dt_compile_ms = @divTrunc(t0.read(), std.time.ns_per_ms);
    debugz.dbg.printLevel(.info, "[nvrtc] JIT compile: {d} ms ({d} bytes cubin, {s})\n", .{ dt_compile_ms, cubin.len, arch });

    // Carga por el MISMO camino que loadModule (cuModuleLoadData).
    try cudaz.ensureContext();
    const jit_module = try cudaz.cuModuleLoadData(cubin);
    defer cudaz.cuModuleUnload(jit_module);
    const jit_func = try cudaz.cuModuleGetFunction(jit_module, "argmaxJitKernel");

    // Referencia build-time: el cubin nvcc del MISMO kernel (argmaxF32).
    var lk = try layer_kernels.LayerKernels.init(try sharedStream());
    defer lk.deinit();

    const shapes = [_][2]usize{
        .{ 1, 1 },    .{ 1, 31 },   .{ 1, 32 },   .{ 1, 33 },
        .{ 1, 1023 }, .{ 1, 1024 }, .{ 1, 1025 }, .{ 1, 5000 },
        .{ 2, 2048 }, .{ 3, 777 },
    };
    var rng = std.Random.Xoshiro256.init(12345);
    const stream = lk.stream;

    for (shapes) |sh| {
        const rows = sh[0];
        const cols = sh[1];
        const host = try std.testing.allocator.alloc(f32, rows * cols);
        defer std.testing.allocator.free(host);
        for (host) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
        // Empates deliberados (desempate índice mínimo — la trampa real).
        for (0..rows) |r| host[r * cols + (cols / 2)] = 7.0;
        for (0..rows) |r| host[r * cols + 0] = 7.0;

        const d_x = try cudaz.cuMemAlloc(host.len * @sizeOf(f32));
        defer cudaz.cuMemFree(d_x);
        try cudaz.cuMemcpyHtoD(d_x, @intFromPtr(host.ptr), host.len * @sizeOf(f32));

        // Salida JIT.
        const d_jit = try cudaz.cuMemAlloc(rows * @sizeOf(i32));
        defer cudaz.cuMemFree(d_jit);
        try cudaz.cuMemsetD8(d_jit, 0, rows * @sizeOf(i32));

        var ncols_c: c_int = @intCast(cols);
        var xd: usize = d_x;
        var jd: usize = d_jit;
        var kp = [_]?*anyopaque{ &xd, &jd, &ncols_c };
        const blocks: c_uint = @intCast(rows);
        try cudaz.cuLaunchKernel(jit_func, blocks, 1, 1, 256, 1, 1, 0, stream, @ptrCast(&kp), null);
        try cudaz.cuStreamSynchronize(stream);
        const jit_out = try std.testing.allocator.alloc(i32, rows);
        defer std.testing.allocator.free(jit_out);
        try cudaz.cuMemcpyDtoH(@intFromPtr(jit_out.ptr), d_jit, rows * @sizeOf(i32));

        // Salida cubin build-time (argmaxF32Kernel).
        const bt_out = try std.testing.allocator.alloc(i32, rows);
        defer std.testing.allocator.free(bt_out);
        try runBuildTimeArgmax(&lk, d_x, bt_out, rows, cols);

        for (0..rows) |r| {
            const want = hostArgmax(host[r * cols ..][0..cols]);
            try std.testing.expectEqual(want, jit_out[r]); // JIT == host
            try std.testing.expectEqual(bt_out[r], jit_out[r]); // JIT == cubin bit-exact
        }
    }
}

/// Lanza argmaxF32Kernel del cubin build-time sobre el MISMO d_x.
fn runBuildTimeArgmax(lk: *layer_kernels.LayerKernels, d_x: cudaz.CUdeviceptr, out: []i32, rows: usize, cols: usize) !void {
    const d_out = try lk.argmaxOut(rows);
    try lk.argmaxF32(d_x, d_out, rows, cols);
    try cudaz.cuStreamSynchronize(lk.stream);
    try cudaz.cuMemcpyDtoH(@intFromPtr(out.ptr), d_out, rows * @sizeOf(i32));
}

test "nvrtc: ErrorFlag — flag limpio en verde, EF_NAN/EF_OOB en malicioso" {
    debugz.init();
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    if (!nvrtc.available()) return error.SkipZigTest;

    var arch_buf: [16]u8 = undefined;
    const arch = try deviceArch(std.testing.allocator, &arch_buf);

    const ef_src =
        \\__device__ __forceinline__ void efSetError(unsigned int* ef, unsigned int code) {
        \\    atomicCAS(ef, 0u, code);
        \\}
        \\__device__ __forceinline__ int efIsNan(float v) { return v != v; }
        \\extern "C" __global__ void efDemoKernel(const float* __restrict__ src, float* __restrict__ dst, int n, unsigned int* __restrict__ err_flag) {
        \\    const int i = blockIdx.x * blockDim.x + threadIdx.x;
        \\    if (i >= n) return;
        \\    const float v = src[i];
        \\    if (efIsNan(v)) { efSetError(err_flag, 2u); dst[i] = 0.0f; return; }
        \\    dst[i] = v * 2.0f;
        \\}
        \\extern "C" __global__ void efOobKernel(float* __restrict__ dst, int work_n, int buf_n, unsigned int* __restrict__ err_flag) {
        \\    const int i = blockIdx.x * blockDim.x + threadIdx.x;
        \\    if (i >= work_n) return;
        \\    if (i >= buf_n) { efSetError(err_flag, 1u); return; }
        \\    dst[i] = (float)i;
        \\}
    ;
    const cubin = try nvrtc.compileCubin(std.testing.allocator, ef_src, .{ .arch = arch });
    defer std.testing.allocator.free(cubin);
    try cudaz.ensureContext();
    const mod = try cudaz.cuModuleLoadData(cubin);
    defer cudaz.cuModuleUnload(mod);
    const demo_f = try cudaz.cuModuleGetFunction(mod, "efDemoKernel");
    const oob_f = try cudaz.cuModuleGetFunction(mod, "efOobKernel");
    const stream = try sharedStream();

    // Buffer del flag: alloc UNA vez, memset por launch (barato — UC-2.2).
    const d_flag = try cudaz.cuMemAlloc(4);
    defer cudaz.cuMemFree(d_flag);

    // ── Verde: sin NaN ⇒ flag queda 0 ──
    const n = 256;
    const src = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(src);
    for (src, 0..) |*v, i| v.* = @floatFromInt(i);
    const d_src = try cudaz.cuMemAlloc(n * @sizeOf(f32));
    defer cudaz.cuMemFree(d_src);
    try cudaz.cuMemcpyHtoD(d_src, @intFromPtr(src.ptr), n * @sizeOf(f32));
    const d_dst = try cudaz.cuMemAlloc(n * @sizeOf(f32));
    defer cudaz.cuMemFree(d_dst);

    try cudaz.cuMemsetD8(d_flag, 0, 4);
    var nc: c_int = n;
    var sd: usize = d_src;
    var dd: usize = d_dst;
    var fd: usize = d_flag;
    var kp = [_]?*anyopaque{ &sd, &dd, &nc, &fd };
    try cudaz.cuLaunchKernel(demo_f, 1, 1, 1, 256, 1, 1, 0, stream, @ptrCast(&kp), null);
    try cudaz.cuStreamSynchronize(stream);
    var flag: u32 = 99;
    try cudaz.cuMemcpyDtoH(@intFromPtr(&flag), d_flag, 4);
    try std.testing.expectEqual(@as(u32, 0), flag); // limpio en verde

    // ── NaN: el kernel graba EF_NAN (2) y no trappea ──
    src[10] = std.math.nan(f32);
    try cudaz.cuMemcpyHtoD(d_src, @intFromPtr(src.ptr), n * @sizeOf(f32));
    try cudaz.cuMemsetD8(d_flag, 0, 4);
    try cudaz.cuLaunchKernel(demo_f, 1, 1, 1, 256, 1, 1, 0, stream, @ptrCast(&kp), null);
    try cudaz.cuStreamSynchronize(stream);
    try cudaz.cuMemcpyDtoH(@intFromPtr(&flag), d_flag, 4);
    try std.testing.expectEqual(@as(u32, 2), flag); // EF_NAN

    // ── OOB malicioso: work_n > buf_n ⇒ EF_OOB (1), sin escritura fuera ──
    var work: c_int = 2048; // intenta tocar 2048
    var buf: c_int = 256; // buffer REAL: 256
    var kp2 = [_]?*anyopaque{ &dd, &work, &buf, &fd };
    try cudaz.cuMemsetD8(d_flag, 0, 4);
    try cudaz.cuLaunchKernel(oob_f, @intCast(@divTrunc(work, 256)), 1, 1, 256, 1, 1, 0, stream, @ptrCast(&kp2), null);
    try cudaz.cuStreamSynchronize(stream);
    try cudaz.cuMemcpyDtoH(@intFromPtr(&flag), d_flag, 4);
    try std.testing.expectEqual(@as(u32, 1), flag); // EF_OOB

    // Breadcrumb host del error (patrón UC-2.2 para los launchers reales).
    if (debugz.dbg.at(.info)) {
        debugz.dbg.printLevel(.info, "[gpu_kernels] ErrorFlag test: verde=0 NaN=2 OOB=1 — todos los códigos correctos\n", .{});
    }
}
