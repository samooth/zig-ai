// UC-1.4 (lane-cuda): shim de float.h para NVRTC JIT. Ver header-comment
// de stdio.h en este dir. Inventario: el .cu solo usa FLT_MAX.
#ifndef NVRTC_SHIM_FLOAT_H
#define NVRTC_SHIM_FLOAT_H
#define FLT_MAX 3.402823466e+38f
#define FLT_MIN 1.175494351e-38f
#define FLT_EPSILON 1.19209290e-07f
#define FLT_DIG 6
#define DBL_MAX 1.7976931348623157e+308
#define DBL_MIN 2.2250738585072014e-308
#define DBL_EPSILON 2.2204460492503131e-016
#define DBL_DIG 15
#define FLT_RADIX 2
#endif
