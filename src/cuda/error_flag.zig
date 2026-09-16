//! UC-2.2 (TODO_CUDA.md, lane-cuda): wrapper HOST del ErrorFlag device-side.
//! Complemento de `src/cuda/error_flag.cuh` (device): aquí vive el buffer
//! device persistente + memset por launch (barato) + lectura post-launch
//! con breadcrumb `[gpu_kernels]` gated DEBUG_LEVEL>=1.
//!
//! Para el detector DEDICADO de OOB (buffer con tamaño real + work_n), ver
//! el piloto de `src/cuda/nvrtc_kernels.cu` (efOobKernel) — aquí el wrapper
//! es genérico: el kernel recibe `*u32` y llama zaSetError.
const std = @import("std");
const cudaz = @import("cudaz");
const debugz = @import("debug");

/// Códigos — ESPEJO de los #define ZA_EF_* de error_flag.cuh. Un test
/// unitario (tests/test_cuda_utils.zig) fija los valores para que no
/// diverjan silenciosamente.
pub const Code = enum(u32) {
    no_error = 0,
    oob = 1,
    nan = 2,
    inf = 3,
    assertion = 4,
    custom = 0x100,
};

pub fn codeName(c: u32) []const u8 {
    return switch (c) {
        0 => "NO_ERROR",
        1 => "OOB",
        2 => "NAN",
        3 => "INF",
        4 => "ASSERT",
        else => if (c >= 0x100) "CUSTOM" else "UNKNOWN",
    };
}

/// Buffer device del flag (u32). Persistente: alloc UNA vez (init) y
/// memset a 0 por launch (clear) — capture-safe (async).
pub const Buffer = struct {
    ptr: cudaz.CUdeviceptr = 0,
    stream: cudaz.CUstream = undefined,

    /// Aloca el flag (4 bytes) y lo pone a 0. Requiere contexto CUDA.
    pub fn init(stream: cudaz.CUstream) !Buffer {
        try cudaz.ensureContext();
        const ptr = try cudaz.cuMemAlloc(@sizeOf(u32));
        errdefer cudaz.cuMemFree(ptr);
        const buf = Buffer{ .ptr = ptr, .stream = stream };
        try buf.clear();
        return buf;
    }

    pub fn deinit(self: *Buffer) void {
        if (self.ptr != 0) cudaz.cuMemFree(self.ptr);
        self.ptr = 0;
    }

    /// memset 0 — usar ANTES de cada launch que pueda grabar. Async
    /// (stream-ordered) para ser legal dentro de capture (lección 1.3).
    pub fn clear(self: *const Buffer) !void {
        try cudaz.cuMemsetD8Async(self.ptr, 0, @sizeOf(u32), self.stream);
    }

    /// Lectura SÍNCRONA del flag (sólo tras sincronizar el stream).
    pub fn read(self: *const Buffer) !u32 {
        var v: u32 = 0;
        try cudaz.cuMemcpyDtoH(@intFromPtr(&v), self.ptr, @sizeOf(u32));
        return v;
    }

    /// Lee y, si hay error, emite breadcrumb `[gpu_kernels]` gated
    /// DEBUG_LEVEL>=1. Devuelve el código (0 = verde).
    pub fn check(self: *const Buffer, kernel: []const u8) !u32 {
        const v = try self.read();
        if (v != 0 and debugz.dbg.at(.info)) {
            debugz.dbg.printLevel(.info, "[gpu_kernels] ErrorFlag {s}: {s} ({d}) — primer error grabado\n", .{ kernel, codeName(v), v });
        }
        return v;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Tests unitarios (sin GPU): cubren la tabla de códigos (espejo del .cuh) y
// codeName. Los tests con device real viven en tests/test_nvrtc_gpu.zig
// (ErrorFlag smoke: verde=0/NaN=2/OOB=1).
// ─────────────────────────────────────────────────────────────────────────────

test "error_flag: códigos espejo del .cuh (NO_ERROR/OOB/NAN/INF/ASSERT/CUSTOM)" {
    try std.testing.expectEqual(@as(u32, 0), @intFromEnum(Code.no_error));
    try std.testing.expectEqual(@as(u32, 1), @intFromEnum(Code.oob));
    try std.testing.expectEqual(@as(u32, 2), @intFromEnum(Code.nan));
    try std.testing.expectEqual(@as(u32, 3), @intFromEnum(Code.inf));
    try std.testing.expectEqual(@as(u32, 4), @intFromEnum(Code.assertion));
    try std.testing.expectEqual(@as(u32, 0x100), @intFromEnum(Code.custom));
}

test "error_flag: codeName legible" {
    try std.testing.expectEqualStrings("NO_ERROR", codeName(0));
    try std.testing.expectEqualStrings("OOB", codeName(1));
    try std.testing.expectEqualStrings("NAN", codeName(2));
    try std.testing.expectEqualStrings("CUSTOM", codeName(0x100));
    try std.testing.expectEqualStrings("CUSTOM", codeName(0x1FF));
    try std.testing.expectEqualStrings("UNKNOWN", codeName(99));
}
