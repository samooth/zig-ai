// MMA primitives para KVarN FA — lane-b1 A9 (Dev A).
//
// Transcripción de las primitivas de fragmentos Turing+ al estilo zig-ai,
// con la convención canónica del PTX ISA para mma.m16n8k16 f32·f16·f16:
//
//   A (16×16 f16, row-major en smem):  ldmatrix.x4 SIN trans; cada lane
//     aporta la dirección de su fila: lanes 0-15 → filas 0-15 cols 0-7,
//     lanes 16-31 → filas 0-15 cols 8-15.
//   B (16×8 f16, k-major en smem — fila k con 8 cols): ldmatrix.x2.trans;
//     lanes 0-15 aportan las 16 filas k; .trans entrega el fragmento B.
//   C (16×8 f32): layout PTX puro: lane L reg l →
//     fila = L/4 + 8·(l/2), col = 2·(L%4) + l%2.
//
// Los fragmentos A/B guardan pares de f16 packed como bits (uint32).

#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace kvarn_mma {

template <int I_, int J_, typename T>
struct Tile {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = (I_ * J_) / 32; // regs de 32 bits por lane

    T x[ne];
};

using TileA = Tile<16, 8, uint32_t>; // A: 16×16 f16 packed, 4 regs de bits
using TileB = Tile<8, 8, uint32_t>;  // B: 16×8  f16 packed, 2 regs de bits
using TileC = Tile<16, 8, float>;    // C: 16×8  f32,           4 regs

// half2 <-> bits (los fragmentos A/B guardan pares f16 como uint32).
__device__ __forceinline__ uint32_t h2bits(__half2 v)
{
    return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ __half2 bits2h(uint32_t b)
{
    return *reinterpret_cast<const __half2*>(&b);
}

// ---------------------------------------------------------------------------
// ldmatrix
// ---------------------------------------------------------------------------

// A 16×16 row-major desde smem (stride en halfs). Dir por lane:
// (lane%16)·stride + (lane/16)·8.
__device__ __forceinline__ void load_ldmatrix_a(
    TileA& a, const __half* smem, int stride)
{
    const int lane = (int)threadIdx.x & 31;
    const __half* p = smem + (lane % 16) * stride + (lane / 16) * 8;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a.x[0]), "=r"(a.x[1]), "=r"(a.x[2]), "=r"(a.x[3])
        : "r"(addr));
}

// B 16×8 k-major desde smem (fila k con 8 cols, stride en halfs).
// Lanes 0-15 aportan las filas k 0-15; .trans produce el fragmento B.
__device__ __forceinline__ void load_ldmatrix_b(
    TileB& b, const __half* smem, int stride)
{
    const int lane = (int)threadIdx.x & 31;
    const __half* p = smem + (lane % 16) * stride;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(b.x[0]), "=r"(b.x[1])
        : "r"(addr));
}

// ---------------------------------------------------------------------------
// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
// ---------------------------------------------------------------------------

__device__ __forceinline__ void mma(TileC& d, const TileA& a, const TileB& b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d.x[0]), "+f"(d.x[1]), "+f"(d.x[2]), "+f"(d.x[3])
        : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]),
          "r"(b.x[0]), "r"(b.x[1]));
}

// ---------------------------------------------------------------------------
// C helpers (layout PTX puro):
//   lane L, reg l → fila = L/4 + 8·(l/2), col = 2·(L%4) + l%2
// ---------------------------------------------------------------------------

__device__ __forceinline__ void store_c(const TileC& c, float* dst, int ld)
{
    const int lane = (int)threadIdx.x & 31;
    const int row_base = lane / 4;
    const int col0 = 2 * (lane % 4);
    #pragma unroll
    for (int l = 0; l < TileC::ne; ++l) {
        const int row = row_base + 8 * (l / 2);
        const int col = col0 + (l % 2);
        dst[(size_t)row * ld + col] = c.x[l];
    }
}

} // namespace kvarn_mma
