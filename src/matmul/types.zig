
//! Tipos y utilidades compartidas del motor matmul

const std = @import("std");

/// Layout de memoria para matrices
pub const Layout = enum {
    RowMajor,   // C-style: A[i][j] = data[i*lda + j]
    ColMajor,   // Fortran-style: A[i][j] = data[j*lda + i]
};

/// Información de dimensión para GEMM
pub const GemmDims = struct {
    M: usize, // filas de A y C
    N: usize, // columnas de B y C
    K: usize, // columnas de A / filas de B
};

/// Configuración de tiling (valores conservativos, ajustables)
pub const TileConfig = struct {
    MC: usize, // macro-panel rows (L2 cache)
    NC: usize, // macro-panel cols (L2 cache)
    KC: usize, // panel depth (L1 cache)
    MR: usize, // micro-kernel rows (registros)
    NR: usize, // micro-kernel cols (registros)

    pub fn default() TileConfig {
        return .{
            .MC = 256,
            .NC = 256,
            .KC = 128,
            .MR = 8,
            .NR = 8,
        };
    }

    pub fn conservative() TileConfig {
        return .{
            .MC = 64,
            .NC = 64,
            .KC = 64,
            .MR = 4,
            .NR = 4,
        };
    }
};

/// Detectar capacidades SIMD en comptime
pub const SimdInfo = struct {
    const builtin = @import("builtin");
    const featureSetHas = std.Target.x86.featureSetHas;

    pub const vec_len_f32 = blk: {
        if (featureSetHas(builtin.cpu.features, .avx512f)) break :blk 16;
        if (featureSetHas(builtin.cpu.features, .avx2)) break :blk 8;
        if (featureSetHas(builtin.cpu.features, .sse2)) break :blk 4;
        break :blk 1;
    };

    pub const vec_len_f64 = blk: {
        if (featureSetHas(builtin.cpu.features, .avx512f)) break :blk 8;
        if (featureSetHas(builtin.cpu.features, .avx2)) break :blk 4;
        if (featureSetHas(builtin.cpu.features, .sse2)) break :blk 2;
        break :blk 1;
    };

    pub const has_simd = vec_len_f32 > 1;
};

/// Timer para benchmarks
pub const Timer = struct {
    start_time: i128,

    pub fn start() Timer {
        return .{ .start_time = std.time.nanoTimestamp() };
    }

    pub fn elapsedNs(self: Timer) u64 {
        return @intCast(std.time.nanoTimestamp() - self.start_time);
    }

    pub fn elapsedMs(self: Timer) f64 {
        return @as(f64, @floatFromInt(self.elapsedNs())) / 1_000_000.0;
    }

    pub fn gflops(self: Timer, ops: f64) f64 {
        const sec = @as(f64, @floatFromInt(self.elapsedNs())) / 1_000_000_000.0;
        return ops / sec / 1_000_000_000.0;
    }
};

/// Comparar tensores con tolerancia
pub fn tensorsApproxEq(comptime T: type, a: anytype, b: anytype, eps: T) bool {
    if (a.shape.len != b.shape.len) return false;
    for (a.shape, b.shape) |sa, sb| {
        if (sa != sb) return false;
    }

    var it_a = a.iterator();
    var it_b = b.iterator();
    while (it_a.next()) |pa| {
        const pb = it_b.next().?;
        const diff = if (pa.* > pb.*) pa.* - pb.* else pb.* - pa.*;
        if (diff > eps) {
            std.log.err("Mismatch: {d} vs {d} (diff {d})", .{ pa.*, pb.*, diff });
            return false;
        }
    }
    return true;
}


