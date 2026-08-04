
//! GEMM con tiling en caché (algoritmo GotoBLAS simplificado)
//! + microkernel SIMD para máximo rendimiento en CPU.

const std = @import("std");
const Tensor = @import("core").Tensor;
const TileConfig = @import("types.zig").TileConfig;
const SimdInfo = @import("types.zig").SimdInfo;

/// GEMM tiled: C = A @ B^T
/// Asume B ya transpuesto para acceso row-major
pub fn gemmTiled(
    comptime T: type,
    A: Tensor(T),
    B: Tensor(T),
    C: *Tensor(T),
    M: usize,
    N: usize,
    K: usize,
    config: TileConfig,
) void {
    if (T != f32) {
        @compileError("gemmTiled solo soporta f32 por ahora");
    }

    @memset(C.data, 0);

    const MC = config.MC;
    const NC = config.NC;
    const KC = config.KC;
    const MR = config.MR;
    const NR = config.NR;

    var i: usize = 0;
    while (i < M) : (i += MC) {
        const i_end = @min(i + MC, M);

        var j: usize = 0;
        while (j < N) : (j += NC) {
            const j_end = @min(j + NC, N);

            var k: usize = 0;
            while (k < K) : (k += KC) {
                const k_end = @min(k + KC, K);

                // Multiplicar panel A[i..i_end, k..k_end] con B[j..j_end, k..k_end]
                gemmPanelF32(A, B, C, i, i_end, j, j_end, k, k_end, MR, NR);
            }
        }
    }
}

fn gemmPanelF32(
    A: Tensor(f32),
    B: Tensor(f32),
    C: *Tensor(f32),
    row0: usize, row1: usize,
    j0: usize, j1: usize,
    k0: usize, k1: usize,
    MR: usize, NR: usize,
) void {
    var i = row0;
    while (i < row1) : (i += MR) {
        const i_end = @min(i + MR, row1);

        var j = j0;
        while (j < j1) : (j += NR) {
            const j_end = @min(j + NR, j1);

            microKernelF32(A, B, C, i, i_end, j, j_end, k0, k1);
        }
    }
}

fn microKernelF32(
    A: Tensor(f32),
    B: Tensor(f32),
    C: *Tensor(f32),
    row0: usize, row1: usize,
    j0: usize, j1: usize,
    k0: usize, k1: usize,
) void {
    const VecLen = SimdInfo.vec_len_f32;
    const Vec = @Vector(VecLen, f32);

    // Acumuladores en registros (pequeño bloque)
    const max_rows = 8;
    const max_cols = 8;
    var accum: [max_rows][max_cols]f32 = undefined;

    const rows = row1 - row0;
    const cols = j1 - j0;

    for (0..rows) |ri| {
        for (0..cols) |cj| {
            accum[ri][cj] = 0;
        }
    }

    var k = k0;
    const k_vec_end = k1 - ((k1 - k0) % VecLen);

    while (k < k_vec_end) : (k += VecLen) {
        for (0..rows) |ri| {
            const i = row0 + ri;
            var a_elems: [VecLen]f32 = undefined;
            for (0..VecLen) |v| {
                a_elems[v] = A.at2(i, k + v);
            }
            const a_vec: Vec = a_elems;

            for (0..cols) |cj| {
                const j = j0 + cj;
                var b_elems: [VecLen]f32 = undefined;
                for (0..VecLen) |v| {
                    b_elems[v] = B.at2(j, k + v);
                }
                const b_vec: Vec = b_elems;

                const prod = a_vec * b_vec;
                accum[ri][cj] += @reduce(.Add, prod);
            }
        }
    }

    // Remainder scalar
    while (k < k1) : (k += 1) {
        for (0..rows) |ri| {
            const i = row0 + ri;
            const a_val = A.at2(i, k);
            for (0..cols) |cj| {
                const j = j0 + cj;
                accum[ri][cj] += a_val * B.at2(j, k);
            }
        }
    }

    // Escribir resultado a C
    for (0..rows) |ri| {
        const i = row0 + ri;
        for (0..cols) |cj| {
            const j = j0 + cj;
            C.ptr2(i, j).* += accum[ri][cj];
        }
    }
}


