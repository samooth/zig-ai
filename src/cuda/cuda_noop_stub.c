// cuda_noop_stub.c — Stub no-op de CUDA/cuBLAS para builds SIN toolkit.
//
// Se enlaza SOLO cuando build.zig detecta que nvcc no existe (has_cuda=false).
// Con cuInit devolviendo CUDA_ERROR_NOT_INITIALIZED (999), cudaz.isCudaAvailable()
// retorna false y TODOS los tests GPU se saltan por sus guards existentes
// (error.SkipZigTest). Ningún código GPU corre en este modo: es un build
// CPU-puro que compila porque los `extern "c"` de cudaz_stub.zig/launchers
// resuelven contra estas no-ops.
//
// CONVENCIÓN: driver API retorna CUresult!=0 (error), runtime API retorna
// cudaError_t!=0, cublas retorna cublasStatus_t!=0. Los launchers dequant
// son void — sin CUDA nunca deben llamarse (los callers chequean
// isCudaAvailable primero); su cuerpo es unreachable-lite: no hacen nada.

#include <stddef.h>

// ===== Driver API (CUresult: 0=SUCCESS, 999=NOT_INITIALIZED) =====
typedef int CUresult;
#define CU_STUB_ERR 999

CUresult cuInit(unsigned int flags) { (void)flags; return CU_STUB_ERR; }
CUresult cuDeviceGet(int *dev, int ordinal) { (void)dev; (void)ordinal; return CU_STUB_ERR; }
CUresult cuDeviceGetCount(int *count) { (void)count; return CU_STUB_ERR; }
CUresult cuDeviceGetName(char *name, int len, int dev) { (void)name; (void)len; (void)dev; return CU_STUB_ERR; }
CUresult cuDeviceTotalMem_v2(size_t *bytes, int dev) { (void)bytes; (void)dev; return CU_STUB_ERR; }
CUresult cuDeviceComputeCapability(int *major, int *minor, int dev) { (void)major; (void)minor; (void)dev; return CU_STUB_ERR; }
CUresult cuDeviceGetAttribute(int *pi, int attrib, int dev) { (void)pi; (void)attrib; (void)dev; return CU_STUB_ERR; }
CUresult cuCtxCreate_v2(void **pctx, unsigned int flags, int dev) { (void)pctx; (void)flags; (void)dev; return CU_STUB_ERR; }
CUresult cuCtxDestroy_v2(void *ctx) { (void)ctx; return CU_STUB_ERR; }
CUresult cuCtxGetCurrent(void **pctx) { (void)pctx; return CU_STUB_ERR; }
CUresult cuCtxSetCurrent(void *ctx) { (void)ctx; return CU_STUB_ERR; }
CUresult cuCtxGetDevice(int *dev) { (void)dev; return CU_STUB_ERR; }
CUresult cuCtxSynchronize(void) { return CU_STUB_ERR; }
CUresult cuDevicePrimaryCtxRetain(void **pctx, int dev) { (void)pctx; (void)dev; return CU_STUB_ERR; }
CUresult cuModuleLoad(void **mod, const char *fname) { (void)mod; (void)fname; return CU_STUB_ERR; }
CUresult cuModuleLoadData(void **mod, const void *image) { (void)mod; (void)image; return CU_STUB_ERR; } // UC-1.3 lane-cuda (NVRTC JIT)
CUresult cuModuleUnload(void *mod) { (void)mod; return CU_STUB_ERR; }
CUresult cuModuleGetFunction(void **hfunc, void *hmod, const char *name) { (void)hfunc; (void)hmod; (void)name; return CU_STUB_ERR; }
CUresult cuGetErrorString(int err, const char **pstr) { (void)err; if (pstr) *pstr = "cuda_noop_stub (no CUDA toolkit)"; return CU_STUB_ERR; }
CUresult cuFuncSetAttribute(void *hfunc, int attrib, long long value) { (void)hfunc; (void)attrib; (void)value; return CU_STUB_ERR; }
CUresult cuMemAlloc_v2(unsigned long long *dptr, size_t bytesize) { (void)dptr; (void)bytesize; return CU_STUB_ERR; }
CUresult cuMemFree_v2(unsigned long long dptr) { (void)dptr; return CU_STUB_ERR; }
CUresult cuMemAllocHost_v2(void **pp, size_t bytesize) { (void)pp; (void)bytesize; return CU_STUB_ERR; }
CUresult cuMemFreeHost(void *p) { (void)p; return CU_STUB_ERR; }
CUresult cuMemHostRegister_v2(void *p, size_t bytesize, unsigned int flags) { (void)p; (void)bytesize; (void)flags; return CU_STUB_ERR; }
CUresult cuMemHostUnregister(void *p) { (void)p; return CU_STUB_ERR; }
CUresult cuMemAddressReserve(unsigned long long *ptr, size_t size, size_t align, unsigned long long addr, unsigned long long flags) { (void)ptr; (void)size; (void)align; (void)addr; (void)flags; return CU_STUB_ERR; }
CUresult cuMemAddressFree(unsigned long long ptr, size_t size) { (void)ptr; (void)size; return CU_STUB_ERR; }
CUresult cuMemCreate(unsigned long long *handle, size_t size, void *prop, unsigned long long flags) { (void)handle; (void)size; (void)prop; (void)flags; return CU_STUB_ERR; }
CUresult cuMemRelease(unsigned long long handle) { (void)handle; return CU_STUB_ERR; }
CUresult cuMemMap(unsigned long long ptr, size_t size, size_t offset, unsigned long long handle, unsigned long long flags) { (void)ptr; (void)size; (void)offset; (void)handle; (void)flags; return CU_STUB_ERR; }
CUresult cuMemUnmap(unsigned long long ptr, size_t size) { (void)ptr; (void)size; return CU_STUB_ERR; }
CUresult cuMemSetAccess(unsigned long long ptr, size_t size, void *desc, size_t count) { (void)ptr; (void)size; (void)desc; (void)count; return CU_STUB_ERR; }
CUresult cuMemGetAllocationGranularity(size_t *granularity, void *prop, unsigned long long option) { (void)granularity; (void)prop; (void)option; return CU_STUB_ERR; }
CUresult cuPointerGetAttribute(void *data, int attribute, unsigned long long ptr) { (void)data; (void)attribute; (void)ptr; return CU_STUB_ERR; }
CUresult cuMemcpyHtoD_v2(unsigned long long dst, const void *src, size_t bytes) { (void)dst; (void)src; (void)bytes; return CU_STUB_ERR; }
CUresult cuMemcpyDtoH_v2(void *dst, unsigned long long src, size_t bytes) { (void)dst; (void)src; (void)bytes; return CU_STUB_ERR; }
CUresult cuMemcpyDtoD_v2(unsigned long long dst, unsigned long long src, size_t bytes) { (void)dst; (void)src; (void)bytes; return CU_STUB_ERR; }
CUresult cuMemcpyHtoDAsync_v2(unsigned long long dst, const void *src, size_t bytes, void *stream) { (void)dst; (void)src; (void)bytes; (void)stream; return CU_STUB_ERR; }
CUresult cuMemcpyDtoHAsync_v2(void *dst, unsigned long long src, size_t bytes, void *stream) { (void)dst; (void)src; (void)bytes; (void)stream; return CU_STUB_ERR; }
CUresult cuMemcpyDtoDAsync_v2(unsigned long long dst, unsigned long long src, size_t bytes, void *stream) { (void)dst; (void)src; (void)bytes; (void)stream; return CU_STUB_ERR; }
CUresult cuMemsetD8_v2(unsigned long long dst, unsigned char value, size_t count) { (void)dst; (void)value; (void)count; return CU_STUB_ERR; }
CUresult cuMemsetD8Async(unsigned long long dst, unsigned char value, size_t count, void *stream) { (void)dst; (void)value; (void)count; (void)stream; return CU_STUB_ERR; }
CUresult cuStreamCreate(void **phstream, unsigned int flags) { (void)phstream; (void)flags; return CU_STUB_ERR; }
CUresult cuStreamDestroy_v2(void *hstream) { (void)hstream; return CU_STUB_ERR; }
CUresult cuStreamSynchronize(void *hstream) { (void)hstream; return CU_STUB_ERR; }
CUresult cuStreamWaitEvent(void *stream, void *event, unsigned int flags) { (void)stream; (void)event; (void)flags; return CU_STUB_ERR; }
CUresult cuStreamBeginCapture(void *stream, int mode) { (void)stream; (void)mode; return CU_STUB_ERR; }
CUresult cuStreamEndCapture(void *stream, void **phgraph) { (void)stream; (void)phgraph; return CU_STUB_ERR; }
CUresult cuGraphInstantiateWithParams(void **phgraphexec, void *hgraph, void *params) { (void)phgraphexec; (void)hgraph; (void)params; return CU_STUB_ERR; }
CUresult cuGraphLaunch(void *hgraphexec, void *stream) { (void)hgraphexec; (void)stream; return CU_STUB_ERR; }
CUresult cuGraphGetNodes(void *hgraph, void *nodes, size_t *numnodes) { (void)hgraph; (void)nodes; (void)numnodes; return CU_STUB_ERR; }
CUresult cuGraphNodeGetType(void *hnode, int *ptype) { (void)hnode; (void)ptype; return CU_STUB_ERR; }
CUresult cuGraphKernelNodeGetParams(void *hnode, void *params) { (void)hnode; (void)params; return CU_STUB_ERR; }
CUresult cuGraphKernelNodeSetParams(void *hnode, const void *params) { (void)hnode; (void)params; return CU_STUB_ERR; }
CUresult cuGraphDestroy(void *hgraph) { (void)hgraph; return CU_STUB_ERR; }
CUresult cuGraphExecDestroy(void *hgraphexec) { (void)hgraphexec; return CU_STUB_ERR; }
CUresult cuEventCreate(void **pevent, unsigned int flags) { (void)pevent; (void)flags; return CU_STUB_ERR; }
CUresult cuEventDestroy(void *event) { (void)event; return CU_STUB_ERR; }
CUresult cuEventRecord(void *event, void *stream) { (void)event; (void)stream; return CU_STUB_ERR; }
CUresult cuEventSynchronize(void *event) { (void)event; return CU_STUB_ERR; }
CUresult cuEventQuery(void *event) { (void)event; return CU_STUB_ERR; }
CUresult cuEventElapsedTime(float *ms, void *start, void *end) { (void)ms; (void)start; (void)end; return CU_STUB_ERR; }
CUresult cuLaunchKernel(void *f, unsigned int gx, unsigned int gy, unsigned int gz, unsigned int bx, unsigned int by, unsigned int bz, unsigned int sm, void *stream, void *params, void *extra) { (void)f; (void)gx; (void)gy; (void)gz; (void)bx; (void)by; (void)bz; (void)sm; (void)stream; (void)params; (void)extra; return CU_STUB_ERR; }
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numblocks, void *hfunc, int blocksize, unsigned int dynamicSMemSize, unsigned int flags) { (void)numblocks; (void)hfunc; (void)blocksize; (void)dynamicSMemSize; (void)flags; return CU_STUB_ERR; }
CUresult cuMemGetInfo_v2(size_t *free_bytes, size_t *total_bytes) { (void)free_bytes; (void)total_bytes; return CU_STUB_ERR; }

// ===== Runtime API (cudaError_t: 0=Success, 999 también sirve como error opaco) =====
int cudaMalloc(void **devPtr, size_t size) { (void)devPtr; (void)size; return CU_STUB_ERR; }
int cudaFree(void *devPtr) { (void)devPtr; return CU_STUB_ERR; }
int cudaMallocAsync(void **devPtr, size_t size, void *pool) { (void)devPtr; (void)size; (void)pool; return CU_STUB_ERR; }
int cudaFreeAsync(void *devPtr, void *pool) { (void)devPtr; (void)pool; return CU_STUB_ERR; }
int cudaMemcpy(void *dst, const void *src, size_t count, int kind) { (void)dst; (void)src; (void)count; (void)kind; return CU_STUB_ERR; }
int cudaMemcpyAsync(void *dst, const void *src, size_t count, int kind, void *stream) { (void)dst; (void)src; (void)count; (void)kind; (void)stream; return CU_STUB_ERR; }
int cudaDeviceSynchronize(void) { return CU_STUB_ERR; }
int cudaGetDeviceCount(int *count) { (void)count; return CU_STUB_ERR; }
int cudaSetDevice(int dev) { (void)dev; return CU_STUB_ERR; }
const char *cudaGetErrorString(int err) { (void)err; return "cuda_noop_stub (no CUDA toolkit)"; }
int cudaGetLastError(void) { return CU_STUB_ERR; }
int cudaStreamCreate(void **pstream, unsigned int flags) { (void)pstream; (void)flags; return CU_STUB_ERR; }
int cudaStreamDestroy(void *stream) { (void)stream; return CU_STUB_ERR; }
int cudaStreamSynchronize(void *stream) { (void)stream; return CU_STUB_ERR; }

// ===== cuBLAS (cublasStatus_t: 0=SUCCESS, 999 opaco) =====
typedef void *cublasHandle_t;
int cublasCreate_v2(cublasHandle_t *handle) { (void)handle; return CU_STUB_ERR; }
int cublasDestroy_v2(cublasHandle_t handle) { (void)handle; return CU_STUB_ERR; }
int cublasSetStream_v2(cublasHandle_t handle, void *stream) { (void)handle; (void)stream; return CU_STUB_ERR; }
int cublasGetStream_v2(cublasHandle_t handle, void **stream) { (void)handle; (void)stream; return CU_STUB_ERR; }
int cublasSetMathMode(cublasHandle_t handle, int mode) { (void)handle; (void)mode; return CU_STUB_ERR; }
int cublasSgemm_v2(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const float *alpha, const void *A, int lda, const void *B, int ldb, const float *beta, float *C, int ldc) { (void)handle; (void)transa; (void)transb; (void)m; (void)n; (void)k; (void)alpha; (void)A; (void)lda; (void)B; (void)ldb; (void)beta; (void)C; (void)ldc; return CU_STUB_ERR; }
int cublasHgemm(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const void *alpha, const void *A, int lda, const void *B, int ldb, const void *beta, void *C, int ldc) { (void)handle; (void)transa; (void)transb; (void)m; (void)n; (void)k; (void)alpha; (void)A; (void)lda; (void)B; (void)ldb; (void)beta; (void)C; (void)ldc; return CU_STUB_ERR; }
int cublasSgemmStridedBatched(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const float *alpha, const void *A, int lda, long long strideA, const void *B, int ldb, long long strideB, const float *beta, float *C, int ldc, long long strideC, int batchCount) { (void)handle; (void)transa; (void)transb; (void)m; (void)n; (void)k; (void)alpha; (void)A; (void)lda; (void)strideA; (void)B; (void)ldb; (void)strideB; (void)beta; (void)C; (void)ldc; (void)strideC; (void)batchCount; return CU_STUB_ERR; }
int cublasGemmEx(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const void *alpha, const void *A, int atype, int lda, const void *B, int btype, int ldb, const void *beta, void *C, int ctype, int ldc, int computeType, int algo) { (void)handle; (void)transa; (void)transb; (void)m; (void)n; (void)k; (void)alpha; (void)A; (void)atype; (void)lda; (void)B; (void)btype; (void)ldb; (void)beta; (void)C; (void)ctype; (void)ldc; (void)computeType; (void)algo; return CU_STUB_ERR; }
int cublasGemmStridedBatchedEx(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const void *alpha, const void *A, int atype, int lda, long long strideA, const void *B, int btype, int ldb, long long strideB, const void *beta, void *C, int ctype, int ldc, long long strideC, int batchCount, int computeType, int algo) { (void)handle; (void)transa; (void)transb; (void)m; (void)n; (void)k; (void)alpha; (void)A; (void)atype; (void)lda; (void)strideA; (void)B; (void)btype; (void)ldb; (void)strideB; (void)beta; (void)C; (void)ctype; (void)ldc; (void)strideC; (void)batchCount; (void)computeType; (void)algo; return CU_STUB_ERR; }
int cublasSetWorkspace(cublasHandle_t handle, void *workspace, size_t size) { (void)handle; (void)workspace; (void)size; return CU_STUB_ERR; }

// ===== Launchers dequant/quant (void; solo se llaman con CUDA real) =====
#define STUB_LAUNCH4(name) void name(float *out, const unsigned char *in, int n, void *stream) { (void)out; (void)in; (void)n; (void)stream; }
#define STUB_LAUNCH5(name) void name(float *out, const unsigned char *in, int n, int blk, void *stream) { (void)out; (void)in; (void)n; (void)blk; (void)stream; }
STUB_LAUNCH4(dequant_q4_0_launcher)
STUB_LAUNCH4(dequant_q4_1_launcher)
STUB_LAUNCH4(dequant_q5_0_launcher)
STUB_LAUNCH4(dequant_q5_1_launcher)
STUB_LAUNCH4(dequant_q8_0_launcher)
STUB_LAUNCH4(dequant_q8_1_launcher)
STUB_LAUNCH4(dequant_q2_k_launcher)
STUB_LAUNCH4(dequant_q3_k_launcher)
STUB_LAUNCH4(dequant_q4_k_launcher)
STUB_LAUNCH4(dequant_q5_k_launcher)
STUB_LAUNCH4(dequant_q6_k_launcher)
STUB_LAUNCH4(dequant_q8_k_launcher)
STUB_LAUNCH4(dequant_iq2_xxs_launcher)
STUB_LAUNCH4(dequant_iq2_xs_launcher)
STUB_LAUNCH4(dequant_iq2_s_launcher)
STUB_LAUNCH4(dequant_iq3_xxs_launcher)
STUB_LAUNCH4(dequant_iq3_s_launcher)
STUB_LAUNCH4(dequant_iq4_xs_launcher)
STUB_LAUNCH4(dequant_iq4_nl_launcher)
STUB_LAUNCH4(dequant_iq1_s_launcher)
STUB_LAUNCH4(dequant_iq1_m_launcher)
STUB_LAUNCH4(dequant_tq1_0_launcher)
STUB_LAUNCH4(dequant_tq2_0_launcher)
STUB_LAUNCH4(dequant_mxfp4_launcher)
STUB_LAUNCH5(dequant_int4_launcher)
STUB_LAUNCH5(dequant_int8_sym_launcher)
STUB_LAUNCH5(dequant_int8_asym_launcher)
void quant_q8_0_launcher(const float *src, unsigned char *dst, int num_blocks, void *stream) { (void)src; (void)dst; (void)num_blocks; (void)stream; }
void quant_mxfp4_launcher(const float *src, unsigned char *dst, int num_blocks, void *stream) { (void)src; (void)dst; (void)num_blocks; (void)stream; }

// ===== Kernels KVarN FA (fattn_kvarn_*) — declarados extern por kvarn modules =====
// void, igual que los launchers: inalcanzables sin CUDA.
void fattn_kvarn_portable_d128_kernel(void) {}
void fattn_kvarn_portable_d128_tail_kernel(void) {}
void fattn_kvarn_portable_d256_kernel(void) {}
void fattn_kvarn_portable_d512_kernel(void) {}
void fattn_kvarn_vec_d256_k4v4_kernel(void) {}
void hybridSplitKernelLauncher(void) {}
void kvarn_init_descs_kernel(void) {}

// ===== NVRTC JIT (lane-cuda UC-1.2) — sin toolkit: available()==false, el
// módulo degrada a error claro; tryJitOrFallback nunca se activa. =====
int nvrtcVersion(int *major, int *minor) { (void)major; (void)minor; return 1; }
int nvrtcCreateProgram(void **prog, const char *src, const char *name, int num_headers, const char **headers, const char **include_names) { (void)prog; (void)src; (void)name; (void)num_headers; (void)headers; (void)include_names; return 1; }
int nvrtcDestroyProgram(void **prog) { (void)prog; return 1; }
int nvrtcCompileProgram(void *prog, int num_options, const char **options) { (void)prog; (void)num_options; (void)options; return 1; }
const char *nvrtcGetErrorString(int res) { (void)res; return "cuda_noop_stub (no CUDA toolkit)"; }
int nvrtcGetProgramLogSize(void *prog, size_t *size) { (void)prog; (void)size; return 1; }
int nvrtcGetProgramLog(void *prog, char *log) { (void)prog; (void)log; return 1; }
int nvrtcGetPTXSize(void *prog, size_t *size) { (void)prog; (void)size; return 1; }
int nvrtcGetPTX(void *prog, char *ptx) { (void)prog; (void)ptx; return 1; }
int nvrtcGetCUBINSize(void *prog, size_t *size) { (void)prog; (void)size; return 1; }
int nvrtcGetCUBIN(void *prog, char *cubin) { (void)prog; (void)cubin; return 1; }
