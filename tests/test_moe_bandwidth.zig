//! test_moe_bandwidth — 4.7 q* auto-split (GPU↔CPU Gap 2, lane-f).
//!
//! Aritmética pura de src/moe/bandwidth.zig (Q16.16 + ceil espejo FreeToken)
//! contra los perfiles reales del Contrato 6 de este host (benchbw.json,
//! RTX 3080 Laptop, capturado 2026-09-02: nominal 22.64/6.60, overlap
//! 18.00/6.61 → fetch_frac_q16=17590). La resolución runtime del perfil
//! vive en cpu_executor.resolveFetchFracQ16 (Contrato 6 vía F4) — ver sus
//! tests en tests/test_hybrid_sync.zig.
const std = @import("std");
const bandwidth = @import("bandwidth");

test "q16Of aritmética Q16.16" {
    try std.testing.expectEqual(@as(u32, 32768), bandwidth.q16Of(10.0, 10.0));
    try std.testing.expectEqual(@as(u32, 0), bandwidth.q16Of(0, 22.64));
    try std.testing.expectEqual(bandwidth.FRAC_MAX, bandwidth.q16Of(6.6, 0));
    try std.testing.expectEqual(@as(u32, 0), bandwidth.q16Of(-1, 10));
    try std.testing.expectEqual(@as(u32, 0), bandwidth.q16Of(10, -1));
}

test "q16Of: perfiles del host de referencia (Contrato 6)" {
    // vía nominal (cpu 22.64 / pcie 6.60) ≈ 14789
    const nominal = bandwidth.q16Of(6.60, 22.64);
    try std.testing.expect(nominal >= 14700 and nominal <= 14900);
    // vía overlap (cpu_ov 18.00 / pcie_ov 6.61) = 17590 (la que reporta el
    // benchbw.json como fetch_frac_q16)
    const overlap = bandwidth.q16Of(6.61, 18.00);
    try std.testing.expect(overlap >= 17500 and overlap <= 17700);
}

test "fetchCount: ceil(q*×M) espejo FreeToken" {
    // 25% con M=7: ceil(0.25×7)=ceil(1.75)=2
    try std.testing.expectEqual(@as(usize, 2), bandwidth.fetchCount(16384, 7));
    // 100% ⇒ todos
    try std.testing.expectEqual(@as(usize, 7), bandwidth.fetchCount(65535, 7));
    // 0% ⇒ 0
    try std.testing.expectEqual(@as(usize, 0), bandwidth.fetchCount(0, 7));
    // M=0 ⇒ 0
    try std.testing.expectEqual(@as(usize, 0), bandwidth.fetchCount(32768, 0));
    // perfil overlap del host (17590 ⇒ q*=0.2684):
    // M=7 → ceil(1.879)=2 · M=8 → ceil(2.147)=3 · M=9 → ceil(2.416)=3
    try std.testing.expectEqual(@as(usize, 2), bandwidth.fetchCount(17590, 7));
    try std.testing.expectEqual(@as(usize, 3), bandwidth.fetchCount(17590, 8));
    try std.testing.expectEqual(@as(usize, 3), bandwidth.fetchCount(17590, 9));
}

test "invariantes de reparto" {
    // fetchCount nunca excede M ni es negativo en extremos
    var m: usize = 0;
    while (m <= 64) : (m += 8) {
        const f = bandwidth.fetchCount(17590, m);
        try std.testing.expect(f <= m);
    }
    // q16Of nunca excede FRAC_MAX
    try std.testing.expect(bandwidth.q16Of(1e9, 1e-9) <= bandwidth.FRAC_MAX);
}
