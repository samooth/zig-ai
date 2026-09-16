//! bandwidth — utilidades q* auto-split (GPU↔CPU Gap 2, lane-f 4.7).
//!
//! FUENTE CANÓNICA del perfil: `cpu_executor.resolveFetchFracQ16` (Contrato 6
//! vía F4 — benchbw.json con vías overlap + validación del reportado). Este
//! módulo aporta SOLO la aritmética pura del split (Q16.16 + ceil) para
//! tests y para futuros call-sites que necesiten repartir un `missing_count`
//! sin depender del módulo executor (pesado: CUDA + threads).
//!
//! Espejo FreeToken:
//!   bench_profile.py:107-139  q* = pcie/(pcie+cpu)
//!   offload_cache.py:136       fetch_count = ceil(q* × missing_count)
//!
//! Host de referencia (RTX 3080 Laptop, Contrato 6 capturado 2026-09-02):
//!   nominal: cpu 22.64 / pcie 6.60  ⇒ q16 ≈ 14789 (q* ≈ 0.226)
//!   overlap: cpu 18.00 / pcie 6.61  ⇒ q16 = 17590 (q* ≈ 0.269) ← usada
const std = @import("std");

/// Fracción Q16.16: 0..65535. FRAC_MAX/65536 ≈ 1.0.
pub const FRAC_MIN: u32 = 0;
pub const FRAC_MAX: u32 = 65535;

/// q16 = round(q* × 65536) con q* = pcie/(pcie+cpu).
/// · pcie<=0 ⇒ 0 (sin vía PCIe: todo CPU)
/// · cpu==0 ⇒ FRAC_MAX (executor ausente: todo fetch)
/// · cpu<0 (perfil corrupto) ⇒ 0 conservador
pub fn q16Of(pcie_gbs: f64, cpu_gbs: f64) u32 {
    if (pcie_gbs <= 0 or cpu_gbs < 0) return 0;
    if (cpu_gbs == 0) return FRAC_MAX;
    const q = pcie_gbs / (pcie_gbs + cpu_gbs);
    const q16f = q * 65536.0;
    if (q16f <= 0) return FRAC_MIN;
    if (q16f >= @as(f64, FRAC_MAX)) return FRAC_MAX;
    return @intFromFloat(q16f);
}

/// Nº de expertos a fetcheado por PCIe dado un missing_count y una fracción
/// Q16.16: ceil(q* × M) = floor((frac×M + 65535)/65536) — espejo FreeToken
/// offload_cache.py:136 (math.ceil con q* float).
pub fn fetchCount(frac_q16: u32, missing_count: usize) usize {
    if (frac_q16 == 0 or missing_count == 0) return 0;
    const num = @as(u64, frac_q16) * @as(u64, missing_count) + 65535;
    const n = num / 65536;
    return @intCast(@min(n, missing_count));
}

// ── Tests ────────────────────────────────────────────────────────────────────
test "q16Of aritmética Q16.16" {
    try std.testing.expectEqual(@as(u32, 32768), q16Of(10.0, 10.0));
    try std.testing.expectEqual(@as(u32, 0), q16Of(0, 22.64));
    try std.testing.expectEqual(FRAC_MAX, q16Of(6.6, 0));
    try std.testing.expectEqual(@as(u32, 0), q16Of(-1, 10));
    try std.testing.expectEqual(@as(u32, 0), q16Of(10, -1));
}

test "q16Of: perfiles nominales del host de referencia" {
    // vía nominal (22.64/6.60) ≈ 14789
    const nominal = q16Of(6.60, 22.64);
    try std.testing.expect(nominal >= 14700 and nominal <= 14900);
    // vía overlap (18.00/6.61) = 17590 reportado por Contrato 6
    const overlap = q16Of(6.61, 18.00);
    try std.testing.expect(overlap >= 17500 and overlap <= 17700);
    // 50/50 exacto
    try std.testing.expectEqual(@as(u32, 32768), q16Of(5, 5));
}

test "fetchCount: ceil(q*×M) espejo FreeToken" {
    // 25% con M=7 ⇒ ceil(1.75)=2
    try std.testing.expectEqual(@as(usize, 2), fetchCount(16384, 7));
    // 100% ⇒ todos
    try std.testing.expectEqual(@as(usize, 7), fetchCount(65535, 7));
    // 0% ⇒ 0
    try std.testing.expectEqual(@as(usize, 0), fetchCount(0, 7));
    // M=0 ⇒ 0
    try std.testing.expectEqual(@as(usize, 0), fetchCount(32768, 0));
    // perfil overlap del host (17590) con M=8 (top_k gemma): ceil(0.2685×8)=3
    try std.testing.expectEqual(@as(usize, 3), fetchCount(17590, 8));
}
