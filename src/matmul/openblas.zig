
//! FFI a OpenBLAS / CBLAS
//! Requiere compilar con: zig build -Dopenblas=true

const std = @import("std");
const Tensor = @import("core").Tensor;

// Condicional: solo importar C headers si estamos linkando OpenBLAS
pub const has_openblas = @hasDecl(@This(), "cblas_sgemm");

// Usamos @cImport condicionalmente via declarativo
// En Zig actual, usamos extern declarations directas

extern "c" fn cblas_sgemm(
    order: i32,
    transA: i32,
    transB: i32,
    M: i32,
    N: i32,
    K: i32,
    alpha: f32,
    A: [*]const f32,
    lda: i32,
    B: [*]const f32,
    ldb: i32,
    beta: f32,
    C: [*]f32,
    ldc: i32,
) void;

extern "c" fn cblas_dgemm(
    order: i32,
    transA: i32,
    transB: i32,
    M: i32,
    N: i32,
    K: i32,
    alpha: f64,
    A: [*]const f64,
    lda: i32,
    B: [*]const f64,
    ldb: i32,
    beta: f64,
    C: [*]f64,
    ldc: i32,
) void;

const CblasRowMajor = 101;
const CblasColMajor = 102;
const CblasNoTrans = 111;
const CblasTrans = 112;

/// GEMM via OpenBLAS
/// A: [M, K], B: [K, N], C: [M, N]
pub fn gemmOpenBlas(
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
    std.debug.assert(A.isContiguous() and B.isContiguous() and C.isContiguous());

    const op_a = if (trans_a) CblasTrans else CblasNoTrans;
    const op_b = if (trans_b) CblasTrans else CblasNoTrans;

    const lda: i32 = if (trans_a) @intCast(M) else @intCast(K);
    const ldb: i32 = if (trans_b) @intCast(K) else @intCast(N);
    const ldc: i32 = @intCast(N);

    if (T == f32) {
        cblas_sgemm(
            CblasRowMajor,
            op_a, op_b,
            @intCast(M), @intCast(N), @intCast(K),
            alpha,
            A.data.ptr, lda,
            B.data.ptr, ldb,
            beta,
            C.data.ptr, ldc,
        );
    } else if (T == f64) {
        cblas_dgemm(
            CblasRowMajor,
            op_a, op_b,
            @intCast(M), @intCast(N), @intCast(K),
            alpha,
            A.data.ptr, lda,
            B.data.ptr, ldb,
            beta,
            C.data.ptr, ldc,
        );
    } else {
        @compileError("OpenBLAS solo soporta f32 y f64");
    }
}

/// Detectar si OpenBLAS está disponible en runtime
pub fn isAvailable() bool {
    // En FFI siempre asumimos disponible si se linkó
    return true;
}


