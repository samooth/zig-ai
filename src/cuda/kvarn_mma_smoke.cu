// Smoke A9: mma.m16n8k16 vs CPU — lane-b1 (Dev A).
//
// Un warp (32 threads) computa D[16×8] = A[16×16]·B[16×8] con los
// fragmentos de mma_kvarn.cuh; grid (1), block 32. A se materializa en
// smem row-major (stride 16), B en smem k-major (16 filas de 8, col-major
// lógico). Compara bit-exacto-2-ulp contra el producto f32 de referencia
// (f16 inputs → f32 mul-add; MMA acumula f32, tolerancia rel 1e-3).

#include "mma_kvarn.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void kvarn_mma_smoke_kernel(
    const __half* a_g,   // 16×16 row-major
    const __half* b_g,   // 16×8  k-major (fila k tiene las 8 cols)
    float* d_g,          // 16×8  row-major out
    int n_iter)
{
    __shared__ __half a_s[16 * 16];
    __shared__ __half b_s[16 * 8];
    __shared__ float d_s[16 * 8];

    const int tid = (int)threadIdx.x;
    for (int i = tid; i < 16 * 16; i += 32) a_s[i] = a_g[i];
    for (int i = tid; i < 16 * 8; i += 32) b_s[i] = b_g[i];
    __syncthreads();

    kvarn_mma::TileC acc;
    #pragma unroll
    for (int l = 0; l < kvarn_mma::TileC::ne; ++l) acc.x[l] = 0.0f;

    for (int it = 0; it < n_iter; ++it) {
        // A: 16×16 row-major (stride 16 halfs). B: 16×8 k-major (stride 8).
        kvarn_mma::TileA ta;
        kvarn_mma::load_ldmatrix_a(ta, a_s, 16);
        kvarn_mma::TileB tb;
        kvarn_mma::load_ldmatrix_b(tb, b_s, 8);
        kvarn_mma::mma(acc, ta, tb);
    }

    kvarn_mma::store_c(acc, d_s, 8);
    __syncthreads();
    for (int i = tid; i < 16 * 8; i += 32) d_g[i] = d_s[i];
}
