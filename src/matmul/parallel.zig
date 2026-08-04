
//! GEMM paralelo usando std.Thread.spawn
//! Particiona por filas (row-wise) para mantener locality.

const std = @import("std");
const Tensor = @import("core").Tensor;
const naive = @import("naive.zig");
const simd = @import("simd.zig");

pub const ParallelConfig = struct {
    num_threads: usize,
    use_simd: bool,
};

/// GEMM paralelo
/// C = A @ B^T (B transpuesto)
pub fn gemmParallel(
    comptime T: type,
    allocator: std.mem.Allocator,
    A: Tensor(T),
    B: Tensor(T),
    C: *Tensor(T),
    M: usize,
    N: usize,
    K: usize,
    config: ParallelConfig,
) !void {
    @memset(C.data, 0);

    const num_threads = @max(1, @min(config.num_threads, M));
    const rows_per_thread = M / num_threads;
    const remainder = M % num_threads;

    var threads = try allocator.alloc(std.Thread, num_threads);
    defer allocator.free(threads);
    @memset(threads, undefined);

    const Worker = workerFn(T);

    var spawned: usize = 0;
    var start_row: usize = 0;
    for (0..num_threads) |t| {
        const extra: usize = if (t < remainder) 1 else 0;
        const end_row = start_row + rows_per_thread + extra;

        if (end_row > start_row) {
            threads[spawned] = try std.Thread.spawn(.{}, Worker.worker, .{
                A, B, C,
                start_row, end_row,
                N, K,
                config.use_simd,
            });
            spawned += 1;
        }

        start_row = end_row;
    }

    for (threads[0..spawned]) |thread| thread.join();
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
        ) void {
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
                            var a_elems: [VecLen]f32 = undefined;
                            var b_elems: [VecLen]f32 = undefined;
                            for (0..VecLen) |v| {
                                a_elems[v] = A.at2(i, k + v);
                                b_elems[v] = B.at2(j, k + v);
                            }
                            const a_vec: Vec = a_elems;
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

