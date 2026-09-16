//! Adaptive draft-max bench (lane-b3 T6) — micro-bench determinista.
//!
//! Mide el comportamiento del `ProfitController` frente a un workload
//! sintético configurable (acceptance rate variable por fase). El
//! objetivo NO es medir t/s end-to-end del motor (eso requiere modelo
//! real, harness aparte, decisión del absorbedor) sino:
//
//!   1. Confirmar que el controller DEMUESTRA la propiedad de Bee
//!      v0.4.4: cuando el drafter falla, el controller demote/shutdown
//!      temprano y reduce trabajo desperdiciado.
//!   2. Reportar la diferencia de "tokens útiles producidos por unidad
//!      de drafter invocaciones" entre `off` (fijo max) y `profit`
//!      (adaptativo). Esto es un proxy medible del speedup end-to-end.
//!
//! ## Modelo de coste
//!
//! El spec-decoding tiene coste por ronda:
//!   - Drafter cost:  proporcional a `draft_n` (la profundidad activa).
//!   - Verifier cost: constante (1 forward del target).
//!   - Tokens aceptados: hasta `draft_n` (en caso ideal).
//!
//! Workload sintético: el caller define un perfil de acceptance por
//! step (vector de 0..1). En step i, el drafter acepta `profile[i]`
//! tokens esperados de los `n` propuestos. El bench cuenta:
//!   - total_drafter_invocations
//!   - total_tokens_accepted
//!   - avg_active_depth (media del depth activo)
//!   - decisions (count por tipo)
//!
//! El output (proxy del speedup) es:
//!   tokens_accepted / drafter_invocations
//! (más alto = más eficiente). El controller profit debe mantener
//! este ratio cuando el acceptance rate cae, mientras que off lo
//! colapsa hacia abajo.
const std = @import("std");
pub const debugz = @import("debug");

/// Carga de trabajo simulada (acceptance rate por step).
pub const Profile = struct {
    /// Vector de acceptance rate ∈ [0, 1] por step del bench.
    accept: []const f32,
    /// Número total de ciclos a simular (capped a `accept.len`).
    cycles: usize,

    /// Devuelve el acceptance rate del step `i` (clamp 0..1).
    pub fn rateAt(self: Profile, i: usize) f32 {
        if (self.accept.len == 0) return 0.5; // 50% default si vacío
        if (i >= self.accept.len) return 0.0;
        const v = self.accept[i];
        if (v < 0) return 0;
        if (v > 1) return 1;
        return v;
    }
};

/// Modo del bench: `off` (depth fijo) o `profit` (controller activo).
pub const Mode = enum(u8) { off, profit };

/// Resultado del bench para un modo dado.
pub const BenchResult = struct {
    mode: Mode,
    /// Profundidad fija si mode=off.
    fixed_depth: u32,
    /// Total de invocaciones del drafter (cada step = 1 invocación).
    drafter_invocations: u64,
    /// Total de tokens aceptados (suma de acceptance rate × depth activo).
    tokens_accepted: f64,
    /// Suma de depths activos (para avg).
    depth_sum: u64,
    /// Suma de cycle_us simulados (para throughput).
    total_cycle_us: f64,
    /// Conteo de decisiones del controller (por tipo).
    demote_count: u32,
    promote_count: u32,
    shutdown_count: u32,
    probe_count: u32,
    hold_count: u32,

    pub fn avgDepth(self: BenchResult) f64 {
        if (self.drafter_invocations == 0) return 0;
        return @as(f64, @floatFromInt(self.depth_sum)) /
            @as(f64, @floatFromInt(self.drafter_invocations));
    }

    /// Eficiencia: tokens aceptados por µs simulado (throughput efectivo).
    /// Esta es la métrica del gate del LANE prompt: si profit elige bien
    /// las profundidades, throughput >= 5% de off (en modelos reales).
    /// Aquí es un proxy determinista.
    pub fn efficiency(self: BenchResult) f64 {
        if (self.total_cycle_us == 0) return 0;
        return self.tokens_accepted / self.total_cycle_us;
    }
};

/// Resultado comparativo de los dos modos.
pub const CompareResult = struct {
    off: BenchResult,
    profit: BenchResult,
    /// Ratio de throughput (profit / off). >1 = profit gana (menos µs
    /// por token aceptado). Calculado en base a la métrica primaria
    /// (tokens / µs) — es la métrica del gate del LANE prompt.
    pub fn speedup(self: CompareResult) f64 {
        const e_off = self.off.efficiency();
        if (e_off == 0) return if (self.profit.efficiency() > 0) std.math.inf(f64) else 1.0;
        return self.profit.efficiency() / e_off;
    }
};

/// Perfil sintético "drafter colapsa" (estilo Bee v0.4.4): alta
/// acceptance al inicio, luego caída brusca sostenida.
pub fn collapsingProfile(allocator: std.mem.Allocator, cycles: usize, drop_at: usize) !Profile {
    const buf = try allocator.alloc(f32, cycles);
    for (buf, 0..) |*v, i| {
        // 0.9 hasta drop_at, luego 0.1 (cae 80%).
        v.* = if (i < drop_at) 0.9 else 0.1;
    }
    return .{ .accept = buf, .cycles = cycles };
}

/// Perfil "acceptance rebota" (caso mixto): fase mala, fase buena.
pub fn bouncingProfile(allocator: std.mem.Allocator, cycles: usize) !Profile {
    const buf = try allocator.alloc(f32, cycles);
    for (buf, 0..) |*v, i| {
        // Fases alternas de 32 steps: 0.1 / 0.9
        const phase = (i / 32) % 2;
        v.* = if (phase == 0) 0.1 else 0.9;
    }
    return .{ .accept = buf, .cycles = cycles };
}

/// Perfil "estable alto" (control): acceptance siempre 0.9.
pub fn stableHighProfile(allocator: std.mem.Allocator, cycles: usize) !Profile {
    const buf = try allocator.alloc(f32, cycles);
    @memset(buf, 0.9);
    return .{ .accept = buf, .cycles = cycles };
}

/// Core del bench. Simula el spec-decoding con el controller y
/// devuelve el resultado. El caller corre dos veces (off + profit)
/// con el mismo `profile` y compara.
///
/// En `profit`, el controller decide la profundidad del drafter:
///   - Si la profundidad N acepta X tokens, registramos N×X como
///     tokens aceptados.
///   - Si el controller hace demote, la profundidad cae y los
///     siguientes steps aprovechan la nueva profundidad.
/// En `off`, la profundidad es fija.
pub fn runOnce(
    mode: Mode,
    profile: Profile,
    fixed_depth: u32,
    profit_cfg: adaptive.Config,
) BenchResult {
    var result: BenchResult = .{
        .mode = mode,
        .fixed_depth = fixed_depth,
        .drafter_invocations = 0,
        .tokens_accepted = 0,
        .depth_sum = 0,
        .total_cycle_us = 0,
        .demote_count = 0,
        .promote_count = 0,
        .shutdown_count = 0,
        .probe_count = 0,
        .hold_count = 0,
    };

    var profit: adaptive.ProfitController = undefined;
    var active_depth: u32 = undefined;
    const baseline_us: f64 = 800.0; // coste baseline ficticio µs/step (sin spec)
    // Modelo de coste del spec-decoding (escala 1=µs):
    //   verifier ∝ active_depth (1 forward del target por cada token draft)
    //   drafter ∝ active_depth pero MUCHO más barato que el verifier
    //   en motores reales el ratio drafter:verifier por token es ~1:8
    //   (Qwen3.5 DFlash: drafter 0.5B vs verifier 7B).
    //   Cuando el rate de acceptance cae, mantenemos active_depth pero
    //   el verifier COMPRUEBA todos los tokens (coste fijo en tokens
    //   propuestos), así que depth alto con rate bajo ⇒ coste alto.
    const drafter_per_token: f64 = 1.0;
    const verifier_per_token: f64 = 8.0;
    if (mode == .profit) {
        profit = adaptive.ProfitController.init(profit_cfg);
        // Pre-seed: el baseline se reporta desde la fase 0.
        profit.recordBaseline(@intFromFloat(baseline_us));
        active_depth = profit.tick();
    } else {
        active_depth = fixed_depth;
    }

    var i: usize = 0;
    while (i < profile.cycles) : (i += 1) {
        const rate = profile.rateAt(i);
        // Modelo de coste del step:
        //   - el drafter propone `active_depth` tokens (coste proporcional).
        //   - de esos, `rate * active_depth` son aceptados.
        //   - el verifier comprueba todos (coste constante 1 forward).
        result.drafter_invocations += 1;
        result.depth_sum += active_depth;
        result.tokens_accepted += @as(f64, @floatFromInt(active_depth)) * rate;

        // Modelo de coste del step (mismo para off y profit, para fair
        // comparación): cost(drafter) + cost(verifier) por token. El
        // "verifier" simula que el motor REAL comprueba todos los tokens
        // propuestos (constante en depth); el drafter es 8x más barato
        // por token. Cuando el rate cae, mantenemos depth pero el
        // "verifier" gasta lo mismo ⇒ cycle_us se dispara.
        const cycle_us: u64 = @intFromFloat(baseline_us +
            @as(f64, @floatFromInt(active_depth)) * drafter_per_token +
            @as(f64, @floatFromInt(active_depth)) * verifier_per_token);

        result.drafter_invocations += 1;
        result.depth_sum += active_depth;
        result.tokens_accepted += @as(f64, @floatFromInt(active_depth)) * rate;
        result.total_cycle_us += @as(f64, @floatFromInt(cycle_us));

        // En profit: tras el step, reportamos timing y decisión.
        if (mode == .profit) {
            profit.recordAccepted(active_depth, cycle_us);
            profit.recordTiming(cycle_us);
            profit.evaluateProbe();

            // Para el siguiente step, el baseline se actualiza como
            // EMA del cycle_us (el motor lo hace realmente; aquí
            // emulamos para que el controller tenga telemetría
            // coherente entre steps).
            if (i % 32 == 0) profit.recordBaseline(cycle_us);

            const decision = profit.lastDecision();
            switch (decision) {
                .demote => result.demote_count += 1,
                .promote => result.promote_count += 1,
                .shutdown => result.shutdown_count += 1,
                .probe => result.probe_count += 1,
                .hold => result.hold_count += 1,
            }
            // Tick siguiente: el controller decide el nuevo depth.
            active_depth = profit.tick();
        }
    }

    return result;
}

/// Compara off vs profit en el mismo profile y devuelve la tabla.
pub fn runCompare(
    profile: Profile,
    fixed_depth: u32,
    profit_cfg: adaptive.Config,
) CompareResult {
    const off_res = runOnce(.off, profile, fixed_depth, profit_cfg);
    const profit_res = runOnce(.profit, profile, fixed_depth, profit_cfg);
    return .{ .off = off_res, .profit = profit_res };
}

/// Imprime la tabla comparativa (formato ladder Bee) y devuelve el
/// ratio de eficiencia (profit / off).
///
/// IMPORTANTE: este bench es un PROXY determinista del comportamiento
/// del controller. Mide que el controller demotea correctamente cuando
/// la profundidad es cara, y que reduce el coste por token cuando el
/// acceptance cae. NO sustituye al gate end-to-end (≥5% speedup t/s
/// con modelo real), que se mide en `zig-ai-engine --spec-dm-controller
/// {off,profit}` sobre un GGUF. La métrica aquí es de ciclos simulados,
/// no de µs del motor real.
pub fn writeReport(writer: anytype, cmp: CompareResult, label: []const u8) !f64 {
    const speedup = cmp.speedup();
    const fixed = cmp.off.fixed_depth;

    try writer.print("| {s: <24} | fixed={d} | inv={d} | accepted={d:.1} | rate={d:.4} | avg_d={d:.2} |\n", .{
        label,
        fixed,
        cmp.off.drafter_invocations,
        cmp.off.tokens_accepted,
        cmp.off.efficiency(),
        cmp.off.avgDepth(),
    });
    try writer.print("| {s: <24} | adapt   | inv={d} | accepted={d:.1} | rate={d:.4} | avg_d={d:.2} | d={d} p={d} sd={d} |\n", .{
        label,
        cmp.profit.drafter_invocations,
        cmp.profit.tokens_accepted,
        cmp.profit.efficiency(),
        cmp.profit.avgDepth(),
        cmp.profit.demote_count,
        cmp.profit.promote_count,
        cmp.profit.shutdown_count,
    });
    try writer.print("| {s: <24} | ratio (proxy) = {d:.3} (eff_profit / eff_off; ≥1.0 ⇒ controller OK) |\n", .{ label, speedup });
    return speedup;
}

// ─── Tests (suite autoritativa; corre en `zig build test` por estar
//     el módulo en test_files)
test "bench-adaptive: controller respeta min_depth (no demotea por debajo)" {
    const allocator = std.testing.allocator;
    const profile = try collapsingProfile(allocator, 256, 64);
    defer allocator.free(profile.accept);
    const cfg = adaptive.Config{ .max_depth = 8, .min_depth = 1, .probe_interval = 4 };
    const cmp = runCompare(profile, 8, cfg);
    try std.testing.expect(cmp.profit.avgDepth() >= 1.0);
}

test "bench-adaptive: stable_high produce avg_d cercano a off" {
    const allocator = std.testing.allocator;
    const profile = try stableHighProfile(allocator, 256);
    defer allocator.free(profile.accept);
    const cfg = adaptive.Config{ .max_depth = 8, .min_depth = 1, .probe_interval = 4 };
    const cmp = runCompare(profile, 8, cfg);
    // En profile estable con rate=0.9 el controller NO demotea hasta el
    // mínimo — mantiene avg_d en zona media. (Aserción laxa porque el
    // modelo de coste aquí es proxy, no la economía real del motor.)
    try std.testing.expect(cmp.profit.avgDepth() >= 2.0);
    try std.testing.expect(cmp.profit.avgDepth() < cmp.off.avgDepth());
}

test "bench-adaptive: collapsing produce avg_d MENOR que off (controller reacciona)" {
    const allocator = std.testing.allocator;
    const profile = try collapsingProfile(allocator, 256, 64);
    defer allocator.free(profile.accept);
    const cfg = adaptive.Config{ .max_depth = 8, .min_depth = 1, .probe_interval = 4 };
    const cmp = runCompare(profile, 8, cfg);
    // Cuando el rate cae, el controller demotea. avg_d_profit debe ser
    // < avg_d_off (que es 8 fijo).
    try std.testing.expect(cmp.profit.avgDepth() < cmp.off.avgDepth());
}

test "bench-adaptive: demote_count > 0 en profile collapsing" {
    const allocator = std.testing.allocator;
    const profile = try collapsingProfile(allocator, 256, 64);
    defer allocator.free(profile.accept);
    const cfg = adaptive.Config{ .max_depth = 8, .min_depth = 1, .probe_interval = 4 };
    const cmp = runCompare(profile, 8, cfg);
    try std.testing.expect(cmp.profit.demote_count > 0);
}

test "bench-adaptive: speedup es > 0 y finito en los 3 perfiles canónicos" {
    const allocator = std.testing.allocator;
    const cfg = adaptive.Config{ .max_depth = 8, .min_depth = 1, .probe_interval = 4 };

    const p1 = try collapsingProfile(allocator, 256, 64);
    defer allocator.free(p1.accept);
    const p2 = try bouncingProfile(allocator, 256);
    defer allocator.free(p2.accept);
    const p3 = try stableHighProfile(allocator, 256);
    defer allocator.free(p3.accept);

    inline for ([_][]const u8{ "collapsing", "bouncing", "stable_high" }) |label| {
        const p = switch (label.len) {
            10 => p1,
            8 => p2,
            else => p3,
        };
        const cmp = runCompare(p, 8, cfg);
        const s = cmp.speedup();
        try std.testing.expect(s > 0);
        try std.testing.expect(std.math.isFinite(s));
    }
}

// ─── Imports nombrados ──────────────────────────────────────────────────────
const adaptive = @import("adaptive_dm");
