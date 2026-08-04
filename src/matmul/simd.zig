
//! Kernels SIMD nativos usando @Vector de Zig
//! Auto-detecta AVX2/AVX-512/SSE2 en comptime.

const std = @import("std");
const Tensor = @import("core").Tensor;
const SimdInfo = @import("types.zig").SimdInfo;

/// GEMM SIMD para matrices contiguas row-major
/// C = A @ B^T (asume B ya está transpuesto para acceso row-major)
/// Optimizado para: A[M,K], B[N,K] (B transpuesto), C[M,N]
pub fn gemmSimd(
    comptime T: type,
    A: Tensor(T),
    B: Tensor(T),
    C: *Tensor(T),
    M: usize,
    N: usize,
    K: usize,
) void {
    if (T == f32) {
        gemmSimdF32(A, B, C, M, N, K);
    } else if (T == f64) {
        gemmSimdF64(A, B, C, M, N, K);
    } else {
        @compileError("SIMD GEMM solo soporta f32 y f64");
    }
}

fn gemmSimdF32(
    A: Tensor(f32),
    B: Tensor(f32),
    C: *Tensor(f32),
    M: usize,
    N: usize,
    K: usize,
) void {
    const VecLen = SimdInfo.vec_len_f32;
    const Vec = @Vector(VecLen, f32);

    @memset(C.data, 0);

    for (0..M) |i| {
        for (0..N) |j| {
            var sum_vec: Vec = @splat(0.0);

            var k: usize = 0;
            const k_vec_end = K - (K % VecLen);

            while (k < k_vec_end) : (k += VecLen) {
                // Cargar A[i, k..k+VecLen]
                var a_elems: [VecLen]f32 = undefined;
                for (0..VecLen) |v| {
                    a_elems[v] = A.at2(i, k + v);
                }
                const a_vec: Vec = a_elems;

                // Cargar B[j, k..k+VecLen] (B está en formato transpuesto)
                var b_elems: [VecLen]f32 = undefined;
                for (0..VecLen) |v| {
                    b_elems[v] = B.at2(j, k + v);
                }
                const b_vec: Vec = b_elems;

                sum_vec += a_vec * b_vec;
            }

            var sum = @reduce(.Add, sum_vec);

            // Remainder scalar
            while (k < K) : (k += 1) {
                sum += A.at2(i, k) * B.at2(j, k);
            }

            C.ptr2(i, j).* = sum;
        }
    }
}

fn gemmSimdF64(
    A: Tensor(f64),
    B: Tensor(f64),
    C: *Tensor(f64),
    M: usize,
    N: usize,
    K: usize,
) void {
    const VecLen = SimdInfo.vec_len_f64;
    const Vec = @Vector(VecLen, f64);

    @memset(C.data, 0);

    for (0..M) |i| {
        for (0..N) |j| {
            var sum_vec: Vec = @splat(0.0);

            var k: usize = 0;
            const k_vec_end = K - (K % VecLen);

            while (k < k_vec_end) : (k += VecLen) {
                var a_elems: [VecLen]f64 = undefined;
                for (0..VecLen) |v| {
                    a_elems[v] = A.at2(i, k + v);
                }
                const a_vec: Vec = a_elems;

                var b_elems: [VecLen]f64 = undefined;
                for (0..VecLen) |v| {
                    b_elems[v] = B.at2(j, k + v);
                }
                const b_vec: Vec = b_elems;

                sum_vec += a_vec * b_vec;
            }

            var sum = @reduce(.Add, sum_vec);

            while (k < K) : (k += 1) {
                sum += A.at2(i, k) * B.at2(j, k);
            }

            C.ptr2(i, j).* = sum;
        }
    }
}

/// GEMV SIMD: y = A @ x
pub fn gemvSimd(
    comptime T: type,
    A: Tensor(T),
    x: Tensor(T),
    y: *Tensor(T),
    M: usize,
    K: usize,
) void {
    if (T == f32) {
        gemvSimdF32(A, x, y, M, K);
    } else {
        @compileError("gemvSimd solo soporta f32");
    }
}

fn gemvSimdF32(
    A: Tensor(f32),
    x: Tensor(f32),
    y: *Tensor(f32),
    M: usize,
    K: usize,
) void {
    const VecLen = SimdInfo.vec_len_f32;
    const Vec = @Vector(VecLen, f32);

    for (0..M) |i| {
        var sum_vec: Vec = @splat(0.0);

        var k: usize = 0;
        const k_vec_end = K - (K % VecLen);

        while (k < k_vec_end) : (k += VecLen) {
            var a_elems: [VecLen]f32 = undefined;
            for (0..VecLen) |v| {
                a_elems[v] = A.at2(i, k + v);
            }
            const a_vec: Vec = a_elems;

            var x_elems: [VecLen]f32 = undefined;
            for (0..VecLen) |v| {
                x_elems[v] = x.at(&[_]usize{k + v});
            }
            const x_vec: Vec = x_elems;

            sum_vec += a_vec * x_vec;
        }

        var sum = @reduce(.Add, sum_vec);

        while (k < K) : (k += 1) {
            sum += A.at2(i, k) * x.at(&[_]usize{k});
        }

        y.ptr(&[_]usize{i}).* = sum;
    }
}


