//! UC-5 (TODO_CUDA.md, lane-cuda): util de LaunchConfig — elimina el
//! cálculo manual de grid (ceil-div) repetido en cada launcher. Sin
//! imports: utilidad pura, testeable sin GPU.
//!
//! Patrón adaptado del estudio del repo externo coderonion/zcuda (MIT):
//! src/types.zig:53-75 (LaunchConfig.forNumElems). Atribución.
const std = @import("std");

/// Dimensiones 3D de grid/block. z/y default 1 (linear launch).
pub const Dim3 = struct {
    x: u32 = 1,
    y: u32 = 1,
    z: u32 = 1,

    pub fn init(x: u32, y: u32, z: u32) Dim3 {
        return .{ .x = x, .y = y, .z = z };
    }

    /// Dim3 lineal: sólo x. (equivalente a grid_dim=n, block_dim=n)
    pub fn linear(n: u32) Dim3 {
        return .{ .x = n };
    }
};

/// Config de launch: grid/block/smem. `forNumElems` = caso común
/// (1D, un hilo por elemento, ceil-div).
pub const LaunchConfig = struct {
    grid_dim: Dim3 = .{},
    block_dim: Dim3 = .{},
    shared_mem_bytes: u32 = 0,

    /// 256 threads/block por defecto (el más común en el repo).
    pub const default_threads: u32 = 256;

    /// Config para procesar `num_elems` elementos con 1 hilo/elemento y
    /// 256 threads/block. ceil-div, mínimo 1 bloque (nunca grid 0).
    pub fn forNumElems(num_elems: u32) LaunchConfig {
        return forNumElemsCustom(num_elems, default_threads);
    }

    /// Igual que `forNumElems` con threads/block explícito (p.ej. 32 para
    /// kernels warp-only). ceil-div sin overflow (usa u64 internamente).
    pub fn forNumElemsCustom(num_elems: u32, threads_per_block: u32) LaunchConfig {
        std.debug.assert(threads_per_block > 0);
        const t: u64 = threads_per_block;
        const n: u64 = num_elems;
        const blocks: u32 = @intCast(@max(@as(u64, 1), (n + t - 1) / t));
        return .{
            .grid_dim = Dim3.linear(blocks),
            .block_dim = Dim3.linear(threads_per_block),
        };
    }

    /// Config con smem dinámico (bytes), p.ej. kernels con `extern __shared__`.
    pub fn withShared(self: LaunchConfig, bytes: u32) LaunchConfig {
        var c = self;
        c.shared_mem_bytes = bytes;
        return c;
    }
};

test "LaunchConfig.forNumElems: ceil-div exacto, mínimo 1 bloque" {
    // Múltiplo exacto.
    const a = LaunchConfig.forNumElems(512);
    try std.testing.expectEqual(@as(u32, 2), a.grid_dim.x);
    try std.testing.expectEqual(@as(u32, 256), a.block_dim.x);
    // Cola no múltiplo.
    const b = LaunchConfig.forNumElems(513);
    try std.testing.expectEqual(@as(u32, 3), b.grid_dim.x);
    // Menor que un bloque.
    const c = LaunchConfig.forNumElems(1);
    try std.testing.expectEqual(@as(u32, 1), c.grid_dim.x);
    // Cero elementos: 1 bloque (no grid 0).
    const d = LaunchConfig.forNumElems(0);
    try std.testing.expectEqual(@as(u32, 1), d.grid_dim.x);
}

test "LaunchConfig.forNumElemsCustom: threads explícitos (warp-only)" {
    const a = LaunchConfig.forNumElemsCustom(100, 32);
    try std.testing.expectEqual(@as(u32, 4), a.grid_dim.x); // ceil(100/32)
    try std.testing.expectEqual(@as(u32, 32), a.block_dim.x);
    // Cola exacta: 64/32 = 2 (no 3).
    const b = LaunchConfig.forNumElemsCustom(64, 32);
    try std.testing.expectEqual(@as(u32, 2), b.grid_dim.x);
}

test "LaunchConfig.withShared: smem dinámico" {
    const c = LaunchConfig.forNumElems(256).withShared(19728);
    try std.testing.expectEqual(@as(u32, 19728), c.shared_mem_bytes);
}
