// Metrics module for TUI bridge - wraps time functionality to avoid direct
// std.time dependency in engine_api, plus the engine-side spec/loop-guard
// metrics surface owned by lane-b3 (BeeLlama P1.1/P1.2/T5).
//
// La superficie spec_* la publica el engine vía `SpecMetricsSnapshot`:
//   - adaptive draft-max controller (profit) — profundidad activa,
//     baseline, última decisión
//   - reasoning-loop guard — contador de eventos
// El bridge del TUI la lee igual que MetricsSnapshot (campos planos).
//
// RouteCounters (B1 → B3): B1 publica PERF_KVARN_ROUTE por capa; el struct
// `KvarnRouteCounters` define aquí la superficie de exposición (split/
// vector/MMA/portable). B1 llena los contadores cuando su rama cierre P2.2.

const std = @import("std");
const time = @import("time");

pub const Timer = time.Timer;
pub const ns_per_s = time.ns_per_s;
pub const ns_per_ms = time.ns_per_ms;
pub const ns_per_us = time.ns_per_us;

pub fn sleep(ns: u64) void {
    time.sleep(ns);
}

// ─── Speculative metrics (lane-b3 T5) ────────────────────────────────────────

/// Decisión del adaptive draft-max controller (espejo del enum de
/// adaptive_dm.zig; duplicado aquí para no crear dependencia del módulo
/// speculative en engine_api — el bridge del TUI solo consume planos).
pub const SpecAdaptiveDecision = enum(u8) {
    hold = 0,
    probe = 1,
    demote = 2,
    promote = 3,
    shutdown = 4,
};

/// Snapshot de métricas especulativas del ciclo actual. El pipeline la
/// publica con cada on_metrics; campos planos (f64/u32/u64/bool) para el
/// bridge sin dependencias.
pub const SpecMetricsSnapshot = struct {
    /// Profundidad de draft activa (0 = speculation OFF / shutdown).
    spec_adaptive_depth: u32 = 0,
    /// Último baseline medido (µs/ciclo sin spec). 0 = sin caracterizar.
    spec_adaptive_baseline_us: u64 = 0,
    /// Última decisión del controller.
    spec_adaptive_decision: SpecAdaptiveDecision = .hold,
    /// Eventos de reasoning-loop confirmados (force-close o warn).
    loop_guard_events: u64 = 0,
    /// Tokens/s con speculation activa (para el dashboard en vivo).
    spec_tokens_per_sec: f64 = 0,
    /// Acceptance rate (0..1) del driver especulativo.
    spec_acceptance_rate: f64 = 0,
};

/// Contadores de ruta KVarN por capa (contrato metrics B1→B3).
/// B1 expone la tabla de capability routing (sm_75→portable / sm_80+→MMA)
/// y por-capa qué ruta ejecutó; este struct es la superficie de agregación
/// que el TUI consume. Los campos se llenan desde el PERF_KVARN_ROUTE
/// breadcrumb cuando B1 cierre P2.2 (flags ya reservados en debug.zig).
pub const KvarnRouteCounters = struct {
    /// Veces que cada ruta atendió un forward de attention (global).
    route_split: u64 = 0,
    route_vector: u64 = 0,
    route_mma: u64 = 0,
    route_portable: u64 = 0,
    /// Capas que usan cada ruta en el último forward (por-capa B1).
    layers_using_mma: u32 = 0,
    layers_using_portable: u32 = 0,

    pub fn total(self: KvarnRouteCounters) u64 {
        return self.route_split + self.route_vector + self.route_mma + self.route_portable;
    }

    /// Fracción de forwards atendida por la ruta MMA (0..1).
    pub fn mmaFraction(self: KvarnRouteCounters) f64 {
        const t = self.total();
        if (t == 0) return 0;
        return @as(f64, @floatFromInt(self.route_mma)) / @as(f64, @floatFromInt(t));
    }
};

/// Registro vivo de métricas del engine (singleton-friendly: el pipeline
/// mantiene una instancia y la copia al snapshot por ciclo).
pub const EngineMetrics = struct {
    spec: SpecMetricsSnapshot = .{},
    kvarn_routes: KvarnRouteCounters = .{},

    /// Marca la decisión del controller adaptativo (por tick).
    pub fn recordAdaptive(self: *EngineMetrics, depth: u32, baseline_us: u64, decision: SpecAdaptiveDecision) void {
        self.spec.spec_adaptive_depth = depth;
        self.spec.spec_adaptive_baseline_us = baseline_us;
        self.spec.spec_adaptive_decision = decision;
    }

    /// Incrementa el contador de eventos del loop guard (por detección).
    pub fn recordLoopGuardEvent(self: *EngineMetrics) void {
        self.spec.loop_guard_events += 1;
    }

    /// Copia los contadores de ruta que B1 publica (por forward).
    pub fn recordKvarnRoutes(self: *EngineMetrics, c: KvarnRouteCounters) void {
        self.kvarn_routes = c;
    }
};

// ─── Tests (suite autoritativa del puente) ──────────────────────────────────

test "EngineMetrics: recordAdaptive refleja profundidad y decisión" {
    var m = EngineMetrics{};
    m.recordAdaptive(3, 1000, .demote);
    try std.testing.expectEqual(@as(u32, 3), m.spec.spec_adaptive_depth);
    try std.testing.expectEqual(@as(u64, 1000), m.spec.spec_adaptive_baseline_us);
    try std.testing.expectEqual(SpecAdaptiveDecision.demote, m.spec.spec_adaptive_decision);

    // Shutdown completo: depth=0 + decisión shutdown.
    m.recordAdaptive(0, 1000, .shutdown);
    try std.testing.expectEqual(@as(u32, 0), m.spec.spec_adaptive_depth);
    try std.testing.expectEqual(SpecAdaptiveDecision.shutdown, m.spec.spec_adaptive_decision);
}

test "EngineMetrics: recordLoopGuardEvent acumula" {
    var m = EngineMetrics{};
    try std.testing.expectEqual(@as(u64, 0), m.spec.loop_guard_events);
    m.recordLoopGuardEvent();
    m.recordLoopGuardEvent();
    try std.testing.expectEqual(@as(u64, 2), m.spec.loop_guard_events);
}

test "KvarnRouteCounters: fracciones y totales" {
    var c = KvarnRouteCounters{};
    try std.testing.expectEqual(@as(u64, 0), c.total());
    try std.testing.expectEqual(@as(f64, 0), c.mmaFraction());

    c.route_split = 10;
    c.route_vector = 20;
    c.route_mma = 60;
    c.route_portable = 10;
    try std.testing.expectEqual(@as(u64, 100), c.total());
    try std.testing.expectApproxEqAbs(@as(f64, 0.6), c.mmaFraction(), 1e-12);

    var m = EngineMetrics{};
    m.recordKvarnRoutes(c);
    try std.testing.expectEqual(@as(u64, 60), m.kvarn_routes.route_mma);
}
