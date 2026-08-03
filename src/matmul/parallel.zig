
//! GEMM paralelo usando std.Thread.Pool
//! Particiona por filas (row-wise) para mantener locality.

const std = @import("std");
const Tensor = @import("core").Tensor;
const naive = @import("naive.zig");
const simd = @import("simd.zig");

pub const ParallelConfig = struct {
    num_threads: usize,
    use_simd: bool,
};

/// GEMM paralelo con thread pool
/// C = A @ B^T (B transpuesto)
pub fn gemmParallel(
    comptime T: type,
    pool: *std.Thread.Pool,
    A: Tensor(T),
    B: Tensor(T),
    C: *Tensor(T),
    M: usize,
    N: usize,
    K: usize,
    config: ParallelConfig,
) !void {
    @memset(C.data, 0);

    const num_threads = config.num_threads;
    const rows_per_thread = M / num_threads;
    const remainder = M % num_threads;

    var wg: std.Thread.WaitGroup = .{};

    var start_row: usize = 0;
    for (0..num_threads) |t| {
        const extra: usize = if (t < remainder) 1 else 0;
        const end_row = start_row + rows_per_thread + extra;

        if (end_row > start_row) {
            wg.start();
            const Worker = workerFn(T);
            try pool.spawn(Worker.worker, .{
                A, B, C,
                start_row, end_row,
                N, K,
                config.use_simd,
                &wg,
            });
        }

        start_row = end_row;
    }

    pool.waitAndWork(&wg);
}

fn workerFn(comptime T: type) type {
    return struct {
        fn worker(
            A: Tensor(T),
            B: Tensor(T),
            C: *Tensor(T),
            row_start: usize,
            row_end: usize,
            N: usize,
            K: usize,
            use_simd: bool,
            wg: *std.Thread.WaitGroup,
        ) void {
            defer wg.finish();

            if (use_simd and T == f32) {
                // Usar SIMD por fila
                for (row_start..row_end) |i| {
                    for (0..N) |j| {
                        const VecLen = @import("types.zig").SimdInfo.vec_len_f32;
                        const Vec = @Vector(VecLen, f32);
                        var sum_vec: Vec = @splat(0.0);

                        var k: usize = 0;
                        const k_vec_end = K - (K % VecLen);

                        while (k < k_vec_end) : (k += VecLen) {
                            var a_vec: Vec = undefined;
                            var b_vec: Vec = undefined;
                            for (0..VecLen) |v| {
                                a_vec[v] = A.at2(i, k + v);
                                b_vec[v] = B.at2(j, k + v);
                            }
                            sum_vec += a_vec * b_vec;
                        }

                        var sum = @reduce(.Add, sum_vec);
                        while (k < K) : (k += 1) {
                            sum += A.at2(i, k) * B.at2(j, k);
                        }

                        C.ptr2(i, j).* = sum;
                    }
                }
            } else {
                // Fallback naive por fila
                for (row_start..row_end) |i| {
                    for (0..N) |j| {
                        var sum: T = 0;
                        for (0..K) |k| {
                            sum += A.at2(i, k) * B.at2(j, k);
                        }
                        C.ptr2(i, j).* = sum;
                    }
                }
            }
        }
    };
}


