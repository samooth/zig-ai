// FP8 E4M3 intrinsics compatibility layer
// Provides native __nv_fp8_e4m3 on sm_89+ and software emulation on sm_80-86
// Based on FreeToken's e4m3_compat.py patterns

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#if __CUDA_ARCH__ >= 890 || !defined(__CUDA_ARCH__)
    #define HAS_NATIVE_FP8 1
#else
    #define HAS_NATIVE_FP8 0
#endif

// ─── FP8 E4M3 type definition ────────────────────────────────────────────────
#if HAS_NATIVE_FP8
    typedef __nv_fp8_e4m3 fp8_e4m3;
#else
    typedef uint8_t fp8_e4m3;
#endif

// ─── Conversion: fp32 -> fp8_e4m3 ────────────────────────────────────────────
#if HAS_NATIVE_FP8
    __device__ __forceinline__ fp8_e4m3 fp32_to_fp8_e4m3(float x) {
        return __float2e4m3_rn(x);
    }
#else
    // Software emulation for sm_80-86
    __device__ __forceinline__ fp8_e4m3 fp32_to_fp8_e4m3(float x) {
        // E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits (no hidden bit for subnormals)
        // Max normal: 1.75 * 2^7 = 224, Min normal: 1.0 * 2^-6 = 0.015625
        // Max subnormal: 0.875 * 2^-6 = 0.013671875
        constexpr float max_normal = 224.0f;
        constexpr float min_normal = 0.015625f;
        constexpr float max_subnormal = 0.013671875f;
        (void)max_normal; (void)min_normal; (void)max_subnormal;

        uint32_t bits = __float_as_uint(x);
        int sign = (bits >> 31) & 1;
        int exp = (bits >> 23) & 0xFF;
        int mant = bits & 0x7FFFFF;

        if (exp == 0xFF) {  // NaN/Inf
            return (sign << 7) | 0x7C;  // qNaN
        }
        if (exp == 0) {  // Subnormal or zero
            return sign << 7;
        }

        // Convert to unbiased exponent
        int unbiased_exp = exp - 127;

        // Clamp to E4M3 range: max FINITO = 448 = 1.75 * 2^8 (exp-8, mant 6).
        // (2.1 lane-f FIX: el clamp original era exp>7→0x78 y 0x78 caía en
        // el NaN-path del decode — con scale=amax/448 el máximo del grupo
        // SIEMPRE llega a 448 ⇒ NaN garantizado. E4M3 real: exp campo 15
        // válido hasta mant 6; solo 0x7F es NaN.)
        if (unbiased_exp > 8) {
            return (sign << 7) | 0x7E;  // 448
        }
        if (unbiased_exp < -6) {
            // 2.1 lane-f FIX-2: subnormales. El path original aniquilaba
            // TODO valor < 2^-6 a cero — con scale=amax/448 (grupos de 128)
            // el ~30% de una gaussiana caía bajo 1.5% del amax y moría
            // (mean_rel 15-30% en GEMM). E4M3 tiene subnormales mant/8·2^-6:
            // redondear al más cercano (2^-9 de paso).
            // x = m23 * 2^(unbiased-23) → cand = round(x · 2^9)
            const unsigned sub = (unsigned)((float)x * 512.0f + (x < 0 ? -0.5f : 0.5f));
            // ((mant | 0x800000) >> (23 - unbiased - 6 + ... )) vía float es
            // más simple: sub ∈ [0, 8].
            if (sub >= 8) {
                // redondeó a arriba al mínimo normal
                return (sign << 7) | 0x08;  // 1.0·2^-6
            }
            return (unsigned)(sign << 7) | sub;
        }

        // Normal number: E4M3 has no hidden bit in mantissa for normals (uses 3 bits)
        // FP32 mantissa has 23 bits, we need top 3 bits
        int e4m3_exp = unbiased_exp + 7;  // Bias of 7 for E4M3
        int e4m3_mant = (mant >> 20) & 0x7;

        // Round to nearest even
        int round_bit = (mant >> 19) & 1;
        int sticky = (mant & 0x7FFFF) != 0;
        if (round_bit && (e4m3_mant & 1 || sticky)) {
            e4m3_mant++;
            if (e4m3_mant == 8) {
                e4m3_mant = 0;
                e4m3_exp++;
            }
            // Overflow tras redondeo SOLO si supera el máximo finito
            // (exp 15 && mant 7 = 0x7F NaN ⇒ clamp a 448).
            if (e4m3_exp == 15 && e4m3_mant == 7) {
                return (sign << 7) | 0x7E;
            }
        }

        return (sign << 7) | (e4m3_exp << 3) | e4m3_mant;
    }
#endif

// ─── Conversion: fp8_e4m3 -> fp32 ────────────────────────────────────────────
#if HAS_NATIVE_FP8
    __device__ __forceinline__ float fp8_e4m3_to_fp32(fp8_e4m3 x) {
        return __e4m32float(x);
    }
#else
    __device__ __forceinline__ float fp8_e4m3_to_fp32(fp8_e4m3 x) {
        int sign = (x >> 7) & 1;
        int exp = (x >> 3) & 0xF;
        int mant = x & 0x7;

        if (exp == 0) {  // Subnormal or zero
            if (mant == 0) return 0.0f;
            // Subnormal: 0.mant * 2^-6
            float val = ldexpf(mant / 8.0f, -6);
            return sign ? -val : val;
        }
        // Normal: 1.mant * 2^(exp-7). E4M3 (OCP FP8): SOLO 0x7F es NaN —
        // NO hay Inf, y exp=15 con mant 1..6 son números válidos (256..448).
        // (2.1 lane-f FIX: el decode original mandaba TODO exp==0xF a NaN
        // — con el scale/448 del cuantizador el máximo del grupo siempre
        // caía ahí ⇒ NaN en cada GEMM.)
        float val = ldexpf(1.0f + mant / 8.0f, exp - 7);
        if (exp == 0xF && mant == 7) {
            return sign ? -__int_as_float(0x7FC00000) : __int_as_float(0x7FC00000);
        }
        return sign ? -val : val;
    }
#endif

// ─── Vectorized conversions ──────────────────────────────────────────────────
__device__ __forceinline__ void fp32_to_fp8_e4m3_x2(float2 x, fp8_e4m3* out) {
    out[0] = fp32_to_fp8_e4m3(x.x);
    out[1] = fp32_to_fp8_e4m3(x.y);
}

__device__ __forceinline__ void fp32_to_fp8_e4m3_x4(float4 x, fp8_e4m3* out) {
    out[0] = fp32_to_fp8_e4m3(x.x);
    out[1] = fp32_to_fp8_e4m3(x.y);
    out[2] = fp32_to_fp8_e4m3(x.z);
    out[3] = fp32_to_fp8_e4m3(x.w);
}

__device__ __forceinline__ float2 fp8_e4m3_to_fp32_x2(fp8_e4m3* in) {
    return make_float2(fp8_e4m3_to_fp32(in[0]), fp8_e4m3_to_fp32(in[1]));
}

__device__ __forceinline__ float4 fp8_e4m3_to_fp32_x4(fp8_e4m3* in) {
    return make_float4(
        fp8_e4m3_to_fp32(in[0]),
        fp8_e4m3_to_fp32(in[1]),
        fp8_e4m3_to_fp32(in[2]),
        fp8_e4m3_to_fp32(in[3])
    );
}