
//! Tipos y utilidades FP16/BF16 para Zig
//! Zig tiene f16 nativo pero con limitaciones en algunas plataformas.
//! Este módulo proporciona conversiones seguras y kernels básicos.

const std = @import("std");
const Tensor = @import("core").Tensor;

// ═══════════════════════════════════════════════════════════════════════════════
// BF16 (Brain Float 16) — 1 sign, 8 exponent, 7 mantissa
// Almacenado como u16, convertido a/f desde f32
// ═══════════════════════════════════════════════════════════════════════════════

pub const BF16 = extern struct {
    bits: u16,

    pub fn fromF32(f: f32) BF16 {
        const u = @as(u32, @bitCast(f));
        const sign = (u >> 31) & 1;
        const exponent = (u >> 23) & 0xFF;
        var mantissa = (u >> 16) & 0x7F;

        // Redondeo: si el bit descartado (bit 15) es 1, redondear
        const round_bit = (u >> 15) & 1;
        if (round_bit == 1) {
            mantissa += 1;
            if (mantissa > 0x7F) {
                mantissa = 0;
                if (exponent < 255) {
                    // overflow a siguiente exponente
                    // simplificado: dejamos que el cast natural lo maneje
                }
            }
        }

        const bf_bits = @as(u16, @intCast((sign << 15) | (exponent << 7) | mantissa));
        return BF16{ .bits = bf_bits };
    }

    pub fn toF32(self: BF16) f32 {
        const u = @as(u32, self.bits);
        const sign = (u >> 15) & 1;
        const exponent = (u >> 7) & 0xFF;
        const mantissa = u & 0x7F;
        const f32_bits = (sign << 31) | (exponent << 23) | (mantissa << 16);
        return @as(f32, @bitCast(f32_bits));
    }

    pub fn add(a: BF16, b: BF16) BF16 {
        return BF16.fromF32(a.toF32() + b.toF32());
    }

    pub fn mul(a: BF16, b: BF16) BF16 {
        return BF16.fromF32(a.toF32() * b.toF32());
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// FP16 (IEEE 754 half) — Zig tiene f16 nativo en la mayoría de targets
// Pero para compatibilidad máxima, ofrecemos conversiones explícitas
// ═══════════════════════════════════════════════════════════════════════════════

pub const F16 = extern struct {
    bits: u16,

    pub fn fromF32(f: f32) F16 {
        // Usar el cast nativo de Zig cuando esté disponible
        const native = @as(f16, @floatCast(f));
        return F16{ .bits = @as(u16, @bitCast(native)) };
    }

    pub fn toF32(self: F16) f32 {
        const native = @as(f16, @bitCast(self.bits));
        return @as(f32, @floatCast(native));
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// Conversión de tensores f32 -> f16/bf16 y viceversa
// ═══════════════════════════════════════════════════════════════════════════════

pub fn tensorF32ToF16(allocator: std.mem.Allocator, src: Tensor(f32)) !Tensor(f16) {
    var dst = try Tensor(f16).alloc(allocator, src.shape);
    for (src.data, dst.data) |s, *d| {
        d.* = @as(f16, @floatCast(s));
    }
    return dst;
}

pub fn tensorF16ToF32(allocator: std.mem.Allocator, src: Tensor(f16)) !Tensor(f32) {
    var dst = try Tensor(f32).alloc(allocator, src.shape);
    for (src.data, dst.data) |s, *d| {
        d.* = @as(f32, @floatCast(s));
    }
    return dst;
}

pub fn tensorF32ToBF16(allocator: std.mem.Allocator, src: Tensor(f32)) !Tensor(u16) {
    // Almacenamos BF16 como u16
    var dst = try Tensor(u16).alloc(allocator, src.shape);
    for (src.data, dst.data) |s, *d| {
        const bf = BF16.fromF32(s);
        d.* = bf.bits;
    }
    return dst;
}

pub fn tensorBF16ToF32(allocator: std.mem.Allocator, src: Tensor(u16)) !Tensor(f32) {
    var dst = try Tensor(f32).alloc(allocator, src.shape);
    for (src.data, dst.data) |s, *d| {
        const bf = BF16{ .bits = s };
        d.* = bf.toF32();
    }
    return dst;
}

// ═══════════════════════════════════════════════════════════════════════════════
// GEMM naive en BF16 (CPU, para testing)
// ═══════════════════════════════════════════════════════════════════════════════

pub fn gemmNaiveBF16(
    A: Tensor(u16), // BF16 almacenado como u16
    B: Tensor(u16),
    C: *Tensor(f32), // Salida en FP32
    M: usize,
    N: usize,
    K: usize,
) void {
    @memset(C.data, 0);
    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0;
            for (0..K) |k| {
                const a_bf = BF16{ .bits = A.at2(i, k) };
                const b_bf = BF16{ .bits = B.at2(k, j) };
                sum += a_bf.toF32() * b_bf.toF32();
            }
            C.ptr2(i, j).* = sum;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GEMM naive en FP16 (CPU, para testing)
// ═══════════════════════════════════════════════════════════════════════════════

pub fn gemmNaiveF16(
    A: Tensor(f16),
    B: Tensor(f16),
    C: *Tensor(f32), // Salida en FP32
    M: usize,
    N: usize,
    K: usize,
) void {
    @memset(C.data, 0);
    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0;
            for (0..K) |k| {
                const a_f = @as(f32, @floatCast(A.at2(i, k)));
                const b_f = @as(f32, @floatCast(B.at2(k, j)));
                sum += a_f * b_f;
            }
            C.ptr2(i, j).* = sum;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

test "BF16 conversion roundtrip" {
    const test_values = &[_]f32{ 0.0, 1.0, -1.0, 3.14159, -2.71828, 1e-5, 1e5, 255.0 };
    for (test_values) |v| {
        const bf = BF16.fromF32(v);
        const back = bf.toF32();
        // BF16 tiene ~3 dígitos decimales de precisión
        const diff = if (back > v) back - v else v - back;
        const rel_diff = if (v != 0.0) diff / @abs(v) else diff;
        try std.testing.expect(rel_diff < 0.01 or diff < 0.01);
    }
}

test "F16 conversion roundtrip" {
    const test_values = &[_]f32{ 0.0, 1.0, -1.0, 0.5, -0.5, 10.0, -10.0 };
    for (test_values) |v| {
        const f = F16.fromF32(v);
        const back = f.toF32();
        const diff = if (back > v) back - v else v - back;
        try std.testing.expect(diff < 0.01);
    }
}


