// UC-1.4 (lane-cuda): shim de math.h para NVRTC JIT. Ver header-comment
// de stdio.h en este dir. Las roundf/fmaxf/fabsf/expf/... que usan los
// .cu son INTRINSICS device de CUDA (declarados en el builtin header de
// NVRTC), NO funciones de math.h glibc — el include es por hábito nvcc.
// La versión glibc declara host + macros __builtin que rompen device-side.
#ifndef NVRTC_SHIM_MATH_H
#define NVRTC_SHIM_MATH_H
// max/min helpers comunes en nuestros .cu (nvcc los trae vía cuda_runtime).
#ifndef __CUDACC_RTC_MAX_MIN_SHIM
#define __CUDACC_RTC_MAX_MIN_SHIM
template <typename T> __device__ inline T max(T a, T b) { return a > b ? a : b; }
template <typename T> __device__ inline T min(T a, T b) { return a < b ? a : b; }
#endif
#endif
