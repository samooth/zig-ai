// UC-2.1 (TODO_CUDA.md, lane-cuda): ErrorFlag device-side — primer error
// grabado, no silencio. Header device-only SIN includes (usable por nvcc
// build-time y por NVRTC JIT: solo builtins de CUDA).
//
// Patrón adaptado del estudio del repo externo coderonion/zcuda (MIT):
// src/kernel/debug.zig:84-130 — ErrorFlag + setError atomicCAS (solo el
// primer error gana). Atribución; cero dependencia.
//
// USO (kernel):
//   __global__ void k(..., unsigned int* ef) {
//       if (i >= n) { zaSetError(ef, ZA_EF_OOB); return; }
//       float v = src[i];
//       zaCheckFloat(v, ef);            // NAN/INF
//       zaAssert(i < out_n, ef);        // ASSERT
//   }
//
// USO (host, patrón UC-2.2 en src/cuda/error_flag.zig):
//   buffer device persistente, memset a 0 por launch, lectura post-launch
//   + breadcrumb [gpu_kernels] gated DEBUG_LEVEL>=1. Sólo el PRIMER error
//   queda grabado (los siguientes CAS fallan al no valer 0 el flag).

#ifndef ZIG_AI_ERROR_FLAG_CUH
#define ZIG_AI_ERROR_FLAG_CUH

// Códigos (u32) — ESPEJO de Code en src/cuda/error_flag.zig (host).
#define ZA_EF_NO_ERROR 0u
#define ZA_EF_OOB 1u
#define ZA_EF_NAN 2u
#define ZA_EF_INF 3u
#define ZA_EF_ASSERT 4u
#define ZA_EF_CUSTOM 0x100u

// Primer error gana: atomicCAS(ef, 0, code) sólo escribe si el valor
// actual es 0 (NO_ERROR). Sin carreras entre threads/bloques.
__device__ __forceinline__ void zaSetError(unsigned int* ef, unsigned int code) {
    atomicCAS(ef, ZA_EF_NO_ERROR, code);
}

// NaN: v != v (IEEE-754). No usa <math.h> (isfinite/isnan) — device-only.
__device__ __forceinline__ int zaIsNan(float v) {
    return v != v;
}

// Inf exacto por bits (0x7f800000/+inf, 0xff800000/-inf) — __int_as_float
// es builtin device, sin header.
__device__ __forceinline__ int zaIsInf(float v) {
    return (v == __int_as_float(0x7f800000)) || (v == __int_as_float(0xff800000));
}

// Chequeo combinado NaN/Inf (lo más común en GEMV/attn): graba el PRIMER
// que aparezca. Devuelve 1 si el valor era no-finito (el kernel puede
// saltarse el acumulado).
__device__ __forceinline__ int zaCheckFloat(float v, unsigned int* ef) {
    if (zaIsNan(v)) { zaSetError(ef, ZA_EF_NAN); return 1; }
    if (zaIsInf(v)) { zaSetError(ef, ZA_EF_INF); return 1; }
    return 0;
}

// Bounds check ON-SPOT: graba OOB y devuelve 1 (el caller hace return).
// Alternativa instrumentada a saltarse el guard en silencio.
__device__ __forceinline__ int zaCheckBounds(unsigned int idx, unsigned int n, unsigned int* ef) {
    if (idx >= n) { zaSetError(ef, ZA_EF_OOB); return 1; }
    return 0;
}

// Assert device con flag (no trappea): graba ASSERT si falla.
__device__ __forceinline__ int zaAssert(int cond, unsigned int* ef) {
    if (!cond) { zaSetError(ef, ZA_EF_ASSERT); return 1; }
    return 0;
}

#endif // ZIG_AI_ERROR_FLAG_CUH
