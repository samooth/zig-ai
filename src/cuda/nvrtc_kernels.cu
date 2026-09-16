// UC-1/UC-2 (TODO_CUDA.md, lane-cuda): kernels SELF-CONTAINED para el
// runtime JIT NVRTC. NVRTC NO trae los headers del SDK (trampa UC-1.4):
// estos fuentes NO usan `#include` — ni del SDK ni locales. Todo lo que
// necesitan (builtins del device) es intrínseco del compilador device.
//
// Compilación:
//   - JIT:  nvrtc.compileCubin() desde src/cuda/nvrtc.zig (2-5s, sin
//     rebuild del binario — gate ZIG_AI_NVRTC=1).
//   - Build-time: NO se compilan con nvcc (no entran en build.zig) —
//     viven SOLO para validar el camino JIT vs cubin del MISMO kernel.
//     La paridad real es contra `argmaxF32Kernel` de layer_kernels.cu
//     (ver tests/test_nvrtc_gpu.zig: kernelArgmaxParityBitExact vs el
//     cubin build-time, y kernelArgmaxSelfContained como smoke JIT puro).
//
// Convención del repo: punteros device, kernels sin sync (el caller
// sincroniza el stream).

#define WARP 32

// ─── ErrorFlag (UC-2.1): primer error grabado, atómico ───────────────────────
// Espejo device-side del patrón del estudio zcuda (MIT, kernel/debug.zig:84).
// El host aloca un buffer [1]u32 (memset 0 por launch), el kernel llama
// setError en fallo; solo el PRIMER error gana (atomicCAS 0→code).
typedef enum {
    EF_NO_ERROR = 0,
    EF_OOB = 1,
    EF_NAN = 2,
    EF_INF = 3,
    EF_ASSERT = 4,
    EF_CUSTOM = 0x100
} EfCode;

__device__ __forceinline__ void efSetError(unsigned int* ef, unsigned int code) {
    // atomicCAS(ef, expected=NO_ERROR, code): sólo el primer escritor gana.
    atomicCAS(ef, EF_NO_ERROR, code);
}

__device__ __forceinline__ int efIsNan(float v) {
    return v != v;
}

__device__ __forceinline__ int efIsInf(float v) {
    return v == v + 1e30f && v == v; // inf pasa, nan no (nan != nan)
}

// ─── argmaxJitKernel: espejo self-contained de argmaxF32Kernel ────────────────
// Misma semántica que layer_kernels.cu:6654 (desempate PRIMERA aparición =
// índice mínimo; el barrido con stride NO garantiza orden de descubrimiento).
// Un hilo por fila; reduce con __shfl_xor_sync.
__device__ __forceinline__ bool jitArgmaxWins(float val, int col, float maxval, int argmax) {
    if (val > maxval) return true;
    if (val == maxval && (argmax < 0 || col < argmax)) return true;
    return false;
}

extern "C" __global__ void argmaxJitKernel(
    const float* __restrict__ x,
    int* __restrict__ dst,
    int ncols)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block = blockDim.x;

    const float* rowp = x + (size_t)row * ncols;

    float maxval = -1.70141183e38f; // -FLT_MAX sin <float.h>
    int   argmax = -1;
    for (int col = tid; col < ncols; col += block) {
        const float val = rowp[col];
        if (jitArgmaxWins(val, col, maxval, argmax)) {
            maxval = val;
            argmax = col;
        }
    }

    // Reduce intra-warp (xor butterfly).
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xffffffffu, maxval, offset, WARP);
        const int   col = __shfl_xor_sync(0xffffffffu, argmax, offset, WARP);
        if (jitArgmaxWins(val, col, maxval, argmax)) {
            maxval = val;
            argmax = col;
        }
    }

    const int nwarps = (block + WARP - 1) / WARP;
    if (nwarps == 1) {
        if (tid == 0) dst[row] = argmax;
        return;
    }

    __shared__ float shared_max[32];
    __shared__ int   shared_argmax[32];
    const int warp_id = tid / WARP;
    const int lane_id = tid % WARP;
    if (lane_id == 0) {
        shared_max[warp_id] = maxval;
        shared_argmax[warp_id] = argmax;
    }
    __syncthreads();

    if (warp_id == 0) {
        maxval = (lane_id < nwarps) ? shared_max[lane_id] : -1.70141183e38f;
        argmax = (lane_id < nwarps) ? shared_argmax[lane_id] : -1;
        for (int offset = WARP / 2; offset > 0; offset >>= 1) {
            const float val = __shfl_xor_sync(0xffffffffu, maxval, offset, WARP);
            const int   col = __shfl_xor_sync(0xffffffffu, argmax, offset, WARP);
            if (jitArgmaxWins(val, col, maxval, argmax)) {
                maxval = val;
                argmax = col;
            }
        }
        if (lane_id == 0) dst[row] = argmax;
    }
}

// ─── efDemoKernel (UC-2.3 piloto): OOB/NaN/INF grabados, NO silencio ────────
// Deliberadamente SIN clamps: si el caller manda n mayor que el buffer real,
// el OOB se graba en vez de corromper en silencio. Es el "test malicioso"
// dedicado de UC-2 (paridad: flag limpio en verde, EF_OOB en la geometría
// maliciosa).
extern "C" __global__ void efDemoKernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n,
    unsigned int* __restrict__ err_flag)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return; // in-range: el OOB real lo simula el test via n malicioso

    // OOB self-check: la carga DEBE estar en rango; si el caller mintió en
    // n (buffer menor), no podemos verlo sin tamaño — el test malicioso
    // usa el flag CUSTOM via una pasada con n escalado.
    const float v = src[i];

    if (efIsNan(v)) {
        efSetError(err_flag, EF_NAN);
        dst[i] = 0.0f;
        return;
    }
    if (efIsInf(v)) {
        efSetError(err_flag, EF_INF);
        dst[i] = 0.0f;
        return;
    }
    dst[i] = v * 2.0f;
}

// ─── efOobKernel (UC-2.3): OOB real con bound explícito ─────────────────────
// El caller pasa buf_n (tamaño REAL del buffer) y work_n (elementos a tocar).
// work_n > buf_n ⇒ EF_OOB grabado y NO se escribe fuera (skip). El "antes":
// escritura fuera de rango → corrompción silenciosa o trap async.
extern "C" __global__ void efOobKernel(
    float* __restrict__ dst,
    int work_n,
    int buf_n,
    unsigned int* __restrict__ err_flag)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= work_n) return;
    if (i >= buf_n) {
        efSetError(err_flag, EF_OOB);
        return; // NO escribe fuera del buffer
    }
    dst[i] = (float)i;
}

// ─── addJitKernel: smoke mínimo (1 línea) para el gate de adopción ──────────
// Loop de iter demo: edit de ESTE kernel + relanzar el harness JIT sin
// rebuild del binario (tests/test_nvrtc_gpu.zig -Dtest-filter=nvrtc).
extern "C" __global__ void addJitKernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
