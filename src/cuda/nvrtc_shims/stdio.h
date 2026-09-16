// UC-1.4 (lane-cuda): shims device-safe para el camino JIT NVRTC.
//
// PROBLEMA: NVRTC compila DEVICE code sin los headers del compilador host.
// layer_kernels.cu incluye headers "de confort" nvcc (<stdio.h> por
// printf diagnóstico gated, <float.h> por FLT_MAX, <cuda_runtime.h> por
// hábito) — y las versiones glibc/gcc arrastran construcciones host
// imposibles device-side (verificado en orden):
//   stdio.h → stdarg.h → __builtin_va_list undefined
//   stdint.h → bits/ → gnu/stubs.h → stubs-32.h ausente (sin multilib)
//   float.h → LDBL_MAX constantes out-of-range para el backend device
//   cuda_runtime.h (distro) → variables/funciones host
//
// SOLUCIÓN: este dir va PRIMERO en los include-dirs del JIT (el orden
// NVRTC es de búsqueda secuencial — el primero gana). Los shims exponen
// SOLO lo que los .cu del repo usan de cada header (inventario verificado
// en layer_kernels.cu: FLT_MAX de float.h; printf gated; cero funciones
// cuda* host; las roundf/fmaxf/... son intrinsics device de CUDA que NO
// vienen de math.h glibc). nvcc build-time sigue viendo los headers
// reales — estos shims SOLO existen en el camino JIT.
//
// FRONTERA (no hay magia): un .cu que use algo NO cubierto por los shims
// en JIT-mode fallará con error claro de nvrtc y degradará al cubin
// build-time (tryJitOrFallback); extender el shim correspondiente.

#ifndef NVRTC_SHIM_STDIO_H
#define NVRTC_SHIM_STDIO_H
// printf device → no-op en JIT (breadcrumbs gated en runtime; el diagnóstico
// REAL de esos paths usa el build nvcc, que trae printf completo).
__device__ inline int nvrtc_printf_noop(const char* fmt, ...) { return 0; }
#define printf(...) nvrtc_printf_noop(__VA_ARGS__)
#endif
