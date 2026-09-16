//! UC-3/UC-5 + UC-2.2 (lane-cuda): tests unitarios SIN GPU de las utilidades
//! nuevas — launch (LaunchConfig), bridge (enums comptime) y error_flag
//! (tabla de códigos host espejo del .cuh). Step: `zig build test-cuda-utils`.
//!
//! Ligero a propósito: NO arrastra cubins nvcc ni el engine — sólo cudaz +
//! debug + las utilidades. Sirve de verificación en máquinas con disco justo.
const std = @import("std");
const cudaz = @import("cudaz");
const launch = @import("launch");
const bridge = @import("bridge");
const error_flag = @import("error_flag");
const nvtx = @import("nvtx");

// Fuerza el análisis de los módulos importados (incluye sus tests internos).
test {
    _ = launch;
    _ = bridge;
    _ = error_flag;
    _ = nvtx;
    _ = cudaz;
}

test "launch: forNumElems ceil-div + defaults" {
    const a = launch.LaunchConfig.forNumElems(512);
    try std.testing.expectEqual(@as(u32, 2), a.grid_dim.x);
    try std.testing.expectEqual(@as(u32, 256), a.block_dim.x);
    try std.testing.expectEqual(@as(u32, 3), launch.LaunchConfig.forNumElems(513).grid_dim.x);
    try std.testing.expectEqual(@as(u32, 1), launch.LaunchConfig.forNumElems(0).grid_dim.x);
    try std.testing.expectEqual(@as(u32, 4), launch.LaunchConfig.forNumElemsCustom(100, 32).grid_dim.x);
}

test "bridge: Fn comptime typo-safe + contains" {
    const LK = bridge.Bridge(&.{ "rmsNormKernel", "argmaxF32Kernel", "qgemmKernel" });
    try std.testing.expectEqual(@as(usize, 3), LK.count);
    try std.testing.expectEqualStrings("qgemmKernel", @tagName(LK.Fn.qgemmKernel));
    try std.testing.expect(LK.contains("argmaxF32Kernel"));
    try std.testing.expect(!LK.contains("argmaxF32Kernell"));
    // getFunction existe y es comptime-typed (no se llama sin CUmodule real).
    try std.testing.expect(@TypeOf(LK.getFunction) != void);
}

test "error_flag: espejo de códigos del .cuh" {
    try std.testing.expectEqual(@as(u32, 0), @intFromEnum(error_flag.Code.no_error));
    try std.testing.expectEqual(@as(u32, 1), @intFromEnum(error_flag.Code.oob));
    try std.testing.expectEqual(@as(u32, 2), @intFromEnum(error_flag.Code.nan));
    try std.testing.expectEqual(@as(u32, 3), @intFromEnum(error_flag.Code.inf));
    try std.testing.expectEqual(@as(u32, 4), @intFromEnum(error_flag.Code.assertion));
    try std.testing.expectEqualStrings("NAN", error_flag.codeName(2));
    try std.testing.expectEqualStrings("CUSTOM", error_flag.codeName(0x100));
}

test "cudaz: tipos CU disponibles (smoke de import)" {
    // El test root importa cudaz — confirma que el módulo ligero linka.
    try std.testing.expect(@sizeOf(cudaz.CUdeviceptr) > 0);
}

test "nvtx: fail-safe — push/pop/mark no crashean con o sin libnvtx" {
    // En este host (noble) hay libnvToolsExt.so.1; en CI puede no haber
    // nada. AMBOS caminos deben ser no-op/barato y NO crashear.
    const avail = nvtx.available();
    nvtx.rangePush("zig-ai-test");
    nvtx.mark("zig-ai-test-mark");
    nvtx.rangePop();
    // Determinismo del cache: segunda llamada = mismo resultado.
    try std.testing.expectEqual(avail, nvtx.available());
    if (avail) {
        // Rango anidado + pop extra benigno (NVTX lo tolera con warning).
        nvtx.rangePush("outer");
        nvtx.rangePush("inner");
        nvtx.rangePop();
        nvtx.rangePop();
        nvtx.rangePop(); // extra
    }
}
