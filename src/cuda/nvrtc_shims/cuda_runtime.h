// UC-1.4 (lane-cuda): shim de cuda_runtime.h para NVRTC JIT. Ver
// header-comment de stdio.h en este dir. Inventario: los .cu del repo NO
// usan funciones cuda* host (kernels puros driver-API); este header se
// incluye por hábito nvcc. La distro cuda_runtime.h declara host-only.
//
// NOTA: __CUDACC_RTC__ ya la define NVRTC — algunos headers del SDK ya se
// auto-adaptan; este shim cubre el resto del laberinto glibc que sí abre.
//
// UC-2.3-prep (fix-1e308f, 2026-09-13): la distro (/usr/include, estilo
// nvhpc) guarda la IMPLEMENTACIÓN inline-asm de __dp4a en
// sm_61_intrinsics.hpp, que sm_61_intrinsics.h SOLO incluye si
// `!defined(__CUDACC_RTC__)` — bajo NVRTC quedan DECLARACIONES sin cuerpo
// ⇒ ptxas fatal: Unresolved extern '_Z6__dp4aiii'. Este shim se incluye
// ANTES (precedencia de include-dirs) y define las 4 variantes con el
// mismo inline-asm PTX que el .hpp (dp4a = SM61+, sm_86 OK).
#ifndef NVRTC_SHIM_CUDA_RUNTIME_H
#define NVRTC_SHIM_CUDA_RUNTIME_H

#if defined(__CUDACC_RTC__) && defined(__CUDA_ARCH__)
#include "sm_61_dp4a_shim.h"
#endif

// Vacío a propósito: los kernels del repo no consumen la Runtime API.
// cuda_fp16.h / mma.h (intrinsics device) van por /usr/include real —
// esos SÍ compilan device-side (son puros inline __device__).
#endif
