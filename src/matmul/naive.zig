
//! Implementación naive de GEMM — correctitud de referencia
//! Complejidad: O(M*N*K), sin optimizaciones.

const std = @import("std");
const Tensor = @import("core").Tensor;

/// GEMM naive: C = alpha * A * B + beta * C
/// A: [M, K], B: [K, N], C: [M, N]
/// trans_a/trans_b: si true, A/B se acceden transpuestos
pub fn gemmNaive(
    comptime T: type,
    A: Tensor(T),
    B: Tensor(T),
    C: *Tensor(T),
    M: usize,
    N: usize,
    K: usize,
    trans_a: bool,
    trans_b: bool,
    alpha: T,
    beta: T,
) void {
    std.debug.assert(A.shape.len == 2 and B.shape.len == 2 and C.shape.len == 2);
    std.debug.assert(C.shape[0] == M and C.shape[1] == N);

    // C = beta * C primero
    if (beta != 1.0) {
        var it = C.iterator();
        while (it.next()) |p| p.* *= beta;
    }

    for (0..M) |i| {
        for (0..N) |j| {
            var sum: T = 0;
            for (0..K) |k| {
                const a_val = if (trans_a) A.at2(k, i) else A.at2(i, k);
                const b_val = if (trans_b) B.at2(j, k) else B.at2(k, j);
                sum += a_val * b_val;
            }
            C.ptr2(i, j).* += alpha * sum;
        }
    }
}

/// GEMV naive: y = alpha * A * x + beta * y
pub fn gemvNaive(
    comptime T: type,
    A: Tensor(T),
    x: Tensor(T),
    y: *Tensor(T),
    M: usize,
    K: usize,
    alpha: T,
    beta: T,
) void {
    if (beta != 1.0) {
        var it = y.iterator();
        while (it.next()) |p| p.* *= beta;
    }

    for (0..M) |i| {
        var sum: T = 0;
        for (0..K) |k| {
            sum += A.at2(i, k) * x.at(&[_]usize{k});
        }
        y.ptr(&[_]usize{i}).* += alpha * sum;
    }
}


