// UC-2.3-prep (fix-1e308f, 2026-09-13, lane-cuda): implementaciones
// __dp4a para NVRTC bajo la distro nvhpc-style (/usr/include).
//
// CAUSA RAÍZ: sm_61_intrinsics.h del SDK distro declara __dp4a pero la
// implementación (inline-asm PTX) vive en sm_61_intrinsics.hpp, incluido
// SOLO si `!defined(__CUDACC_RTC__)` — bajo NVRTC (que define
// __CUDACC_RTC__) las declaraciones quedan extern sin resolver:
//   ptxas fatal: Unresolved extern function '_Z6__dp4aiii'
//
// Este header se incluye desde nvrtc_shims/cuda_runtime.h (que va PRIMERO
// en el include-order de nvrtc.sdkIncludeDirs) ⇒ define las funciones
// ANTES de que cuda_runtime.h del SDK las declare. El asm es el MISMO
// que sm_61_intrinsics.hpp:79-100 (dp4a = SM61+; el engine requiere
// sm_86 — siempre disponible).
#ifndef NVRTC_SHIM_SM61_DP4A_H
#define NVRTC_SHIM_SM61_DP4A_H

#if defined(__CUDACC_RTC__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610

// Sólo las variantes escalares (int/uint) — las .cu del repo usan
// exclusivamente __dp4a(int,int,int) y __dp4a(uint,uint,uint). Las
// variantes char4 del SDK dependen de vector_types.h (incluido
// DESPUÉS por cuda_runtime.h distro): declarar structs propios aquí
// crearía OVERLOADS distintos (tipos diferentes) — confuso e inútil.
static __device__ __inline__ int __dp4a(int srcA, int srcB, int c) {
    int ret;
    asm volatile ("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(ret) : "r"(srcA), "r"(srcB), "r"(c));
    return ret;
}

static __device__ __inline__ unsigned int __dp4a(unsigned int srcA, unsigned int srcB, unsigned int c) {
    unsigned int ret;
    asm volatile ("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(srcA), "r"(srcB), "r"(c));
    return ret;
}

#endif // __CUDACC_RTC__ && __CUDA_ARCH__ >= 610
#endif // NVRTC_SHIM_SM61_DP4A_H
