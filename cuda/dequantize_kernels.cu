// cuda/dequantize_kernels.cu
// De-cuantizacion on-the-fly de K/V cuantizados a FP16 en GPU
// Soporta: INT8 simetrico/asimetrico, INT4, Q4_0, Q8_0

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// ─── INT8 Simetrico ───
// raw: []i8, scales: []f32 (un scale por bloque), block_size: 32/64/256
__global__ void dequant_int8_symmetric_kernel(
    const int8_t* __restrict__ raw,
    const float* __restrict__ scales,
    __half* __restrict__ out,
    int num_elements,
    int block_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    int block_idx = idx / block_size;
    float scale = scales[block_idx];
    int8_t q = raw[idx];
    out[idx] = __float2half((float)q * scale);
}

// ─── INT8 Asimetrico ───
__global__ void dequant_int8_asymmetric_kernel(
    const uint8_t* __restrict__ raw,
    const float* __restrict__ scales,
    const float* __restrict__ zero_points,
    __half* __restrict__ out,
    int num_elements,
    int block_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    int block_idx = idx / block_size;
    float scale = scales[block_idx];
    float zp = zero_points[block_idx];
    float q = (float)raw[idx];
    out[idx] = __float2half((q - zp) * scale);
}

// ─── INT4 (2 valores por byte) ───
__global__ void dequant_int4_kernel(
    const uint8_t* __restrict__ raw,
    const float* __restrict__ scales,
    const float* __restrict__ zero_points,
    __half* __restrict__ out,
    int num_elements,
    int block_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 2 >= num_elements) return;

    int block_idx = (idx * 2) / block_size;
    float scale = scales[block_idx];
    float zp = zero_points[block_idx];

    uint8_t byte = raw[idx];
    float q0 = (float)(byte & 0x0F);
    float q1 = (float)((byte >> 4) & 0x0F);

    int base = idx * 2;
    out[base] = __float2half((q0 - zp) * scale);
    if (base + 1 < num_elements)
        out[base + 1] = __float2half((q1 - zp) * scale);
}

// ─── Q4_0 (GGUF) ───
// Bloque de 32: [scale_f16: 2 bytes] + [32 nibbles: 16 bytes] = 18 bytes
__global__ void dequant_q4_0_kernel(
    const uint8_t* __restrict__ raw,
    __half* __restrict__ out,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    int block_idx = idx / 32;
    int block_offset = block_idx * 18;  // 18 bytes por bloque

    // Leer scale (f16 en bytes 0-1)
    __half scale_h = *(__half*)(raw + block_offset);
    float scale = __half2float(scale_h);

    int in_block = idx % 32;
    int byte_idx = block_offset + 2 + (in_block / 2);
    uint8_t byte = raw[byte_idx];

    int nibble = (in_block % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
    int q = nibble - 8;  // Q4_0 usa offset 8

    out[idx] = __float2half((float)q * scale);
}

// ─── Q8_0 (GGUF) ───
// Bloque de 32: [scale_f16: 2 bytes] + [32 bytes] = 34 bytes
__global__ void dequant_q8_0_kernel(
    const uint8_t* __restrict__ raw,
    __half* __restrict__ out,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    int block_idx = idx / 32;
    int block_offset = block_idx * 34;  // 34 bytes por bloque

    __half scale_h = *(__half*)(raw + block_offset);
    float scale = __half2float(scale_h);

    int in_block = idx % 32;
    int8_t q = (int8_t)raw[block_offset + 2 + in_block];

    out[idx] = __float2half((float)q * scale);
}

// ─── Launcher FFI ───
extern "C" {

void launch_dequant_int8_sym(
    const void* raw, const void* scales, void* out,
    int num_elements, int block_size, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    dequant_int8_symmetric_kernel<<<blocks, threads, 0, stream>>>(
        (const int8_t*)raw, (const float*)scales, (__half*)out,
        num_elements, block_size
    );
}

void launch_dequant_int8_asym(
    const void* raw, const void* scales, const void* zero_points,
    void* out, int num_elements, int block_size, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    dequant_int8_asymmetric_kernel<<<blocks, threads, 0, stream>>>(
        (const uint8_t*)raw, (const float*)scales, (const float*)zero_points,
        (__half*)out, num_elements, block_size
    );
}

void launch_dequant_int4(
    const void* raw, const void* scales, const void* zero_points,
    void* out, int num_elements, int block_size, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_elements / 2 + threads - 1) / threads;
    dequant_int4_kernel<<<blocks, threads, 0, stream>>>(
        (const uint8_t*)raw, (const float*)scales, (const float*)zero_points,
        (__half*)out, num_elements, block_size
    );
}

void launch_dequant_q4_0(
    const void* raw, void* out, int num_elements, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    dequant_q4_0_kernel<<<blocks, threads, 0, stream>>>(
        (const uint8_t*)raw, (__half*)out, num_elements
    );
}

void launch_dequant_q8_0(
    const void* raw, void* out, int num_elements, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    dequant_q8_0_kernel<<<blocks, threads, 0, stream>>>(
        (const uint8_t*)raw, (__half*)out, num_elements
    );
}

} // extern "C"
