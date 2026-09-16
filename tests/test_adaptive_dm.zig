//! Tests del controller adaptativo de draft-max (lane-b3 P1.1).
//! Puros CPU, sin GPU. Verifica la lógica `profit` descrita en
//! `src/speculative/adaptive_dm.zig`.
const std = @import("std");
const spec = @import("speculative");
const adaptive = spec.adaptive_dm;

test "init: cold-start hold max_depth con baseline=null" {
    var c = adaptive.ProfitController.init(.{ .max_depth = 4 });
    try std.testing.expectEqual(@as(u32, 4), c.activeDepth());
    try std.testing.expectEqual(adaptive.Decision.hold, c.lastDecision());
    try std.testing.expect(c.baseline_us == null);
}

test "tick: sin baseline devuelve max_depth y hold" {
    var c = adaptive.ProfitController.init(.{ .max_depth = 8 });
    const d = c.tick();
    try std.testing.expectEqual(@as(u32, 8), d);
    try std.testing.expectEqual(adaptive.Decision.hold, c.lastDecision());
}

test "profit: baseline gana → demote iterativo hasta shutdown" {
    var c = adaptive.ProfitController.init(.{ .max_depth = 4, .min_depth = 1, .probe_interval = 1 });
    c.recordBaseline(1000);

    // Ronda 1: probe a depth=3, peora. Demote 4→3.
    var d = c.tick();
    try std.testing.expectEqual(@as(u32, 3), d);
    try std.testing.expectEqual(adaptive.Decision.probe, c.lastDecision());
    c.recordTiming(1100);
    c.recordAccepted(d, 1100);
    c.evaluateProbe();
    try std.testing.expectEqual(@as(u32, 3), c.activeDepth());
    try std.testing.expectEqual(adaptive.Decision.demote, c.lastDecision());

    // Ronda 2: probe a depth=2, peora. Demote 3→2.
    d = c.tick();
    try std.testing.expectEqual(@as(u32, 2), d);
    c.recordTiming(1100);
    c.recordAccepted(d, 1100);
    c.evaluateProbe();
    try std.testing.expectEqual(@as(u32, 2), c.activeDepth());

    // Ronda 3: probe a depth=1 (current-1). Peora. Demote 2→1.
    d = c.tick();
    try std.testing.expectEqual(@as(u32, 1), d);
    c.recordTiming(1100);
    c.recordAccepted(d, 1100);
    c.evaluateProbe();
    try std.testing.expectEqual(@as(u32, 1), c.activeDepth());

    // Ronda 4: nextProbeDepth devuelve current+1=2 (current==min, no current-1).
    // Probe 2 peora, pd (2) >= current (1) ⇒ shutdown.
    d = c.tick();
    try std.testing.expectEqual(@as(u32, 2), d);
    c.recordTiming(1100);
    c.recordAccepted(d, 1100);
    c.evaluateProbe();
    try std.testing.expectEqual(@as(u32, 0), c.activeDepth());
    try std.testing.expectEqual(adaptive.Decision.shutdown, c.lastDecision());

    // Post-shutdown: depth=0 estable.
    d = c.tick();
    try std.testing.expectEqual(@as(u32, 0), d);
    try std.testing.expectEqual(adaptive.Decision.hold, c.lastDecision());
}

test "profit: probe mejor que baseline → promote" {
    var c = adaptive.ProfitController.init(.{
        .max_depth = 8,
        .min_depth = 2,
        .probe_interval = 1,
        .promote_ratio = 1.05,
    });
    c.recordBaseline(1000);
    const d = c.tick();
    try std.testing.expectEqual(@as(u32, 7), d); // current=8 → probe=7
    c.recordTiming(800); // mejora 25%
    c.recordAccepted(d, 800);
    c.evaluateProbe();
    try std.testing.expectEqual(@as(u32, 7), c.activeDepth());
    try std.testing.expectEqual(adaptive.Decision.promote, c.lastDecision());
}

test "profit: probe marginalmente mejor (ratio < promote_ratio) → hold" {
    var c = adaptive.ProfitController.init(.{ .max_depth = 4, .probe_interval = 1, .promote_ratio = 1.10 });
    c.recordBaseline(1000);
    const d = c.tick();
    try std.testing.expectEqual(@as(u32, 3), d);
    c.recordTiming(950); // mejora 5% < 10% promote_ratio
    c.recordAccepted(d, 950);
    c.evaluateProbe();
    // 950 < 1000 pero ratio 0.95 no llega a promote_ratio=1.10 ⇒ hold current.
    try std.testing.expectEqual(@as(u32, 4), c.activeDepth());
    try std.testing.expectEqual(adaptive.Decision.hold, c.lastDecision());
}

test "reset: limpia estado sin stale (baseline, EMAs, decision)" {
    var c = adaptive.ProfitController.init(.{ .max_depth = 4 });
    c.recordBaseline(1000);
    c.recordAccepted(4, 800);
    _ = c.tick();
    c.evaluateProbe();

    c.reset();
    try std.testing.expectEqual(@as(u32, 4), c.activeDepth()); // cold start
    try std.testing.expect(c.baseline_us == null);
    try std.testing.expectEqual(@as(u64, 0), c.depth_ema[4]);
    try std.testing.expectEqual(adaptive.Decision.hold, c.lastDecision());
}

test "shutdown completo: no crash al re-activar con nuevo baseline" {
    var c = adaptive.ProfitController.init(.{ .max_depth = 2, .probe_interval = 1 });
    c.recordBaseline(500);
    // Ronda 1: probe a depth=1 pierde (600 > 500) → demote 2→1.
    var d = c.tick();
    try std.testing.expectEqual(@as(u32, 1), d);
    c.recordTiming(600);
    c.recordAccepted(d, 600);
    c.evaluateProbe();
    try std.testing.expectEqual(@as(u32, 1), c.activeDepth());
    try std.testing.expectEqual(adaptive.Decision.demote, c.lastDecision());

    // Ronda 2: current==min ⇒ nextProbeDepth devuelve +1 (2). Pierde y
    // pd >= current con current <= min ⇒ shutdown completo.
    d = c.tick();
    try std.testing.expectEqual(@as(u32, 2), d);
    c.recordTiming(600);
    c.recordAccepted(d, 600);
    c.evaluateProbe();
    try std.testing.expectEqual(@as(u32, 0), c.activeDepth());
    try std.testing.expectEqual(adaptive.Decision.shutdown, c.lastDecision());

    // Post-shutdown: tick devuelve 0 estable (speculation OFF).
    d = c.tick();
    try std.testing.expectEqual(@as(u32, 0), d);
    try std.testing.expectEqual(adaptive.Decision.hold, c.lastDecision());

    // Re-activación SIN crash: reset (context-bucket nuevo) + baseline fresco.
    c.reset();
    c.recordBaseline(400);
    d = c.tick();
    // Re-activado: profundidad válida > 0 (probe a current-1=1).
    try std.testing.expect(d > 0);
    try std.testing.expect(d <= 2);
    // Completa el probe con timing MEJOR que baseline: promote sin crash.
    c.recordTiming(300);
    c.recordAccepted(d, 300);
    c.evaluateProbe();
    try std.testing.expect(c.activeDepth() > 0);
    try std.testing.expectEqual(adaptive.Decision.promote, c.lastDecision());
}

test "shutdown vía probe: en min_depth y probe peora" {
    var c = adaptive.ProfitController.init(.{ .max_depth = 2, .min_depth = 1, .probe_interval = 1 });
    c.recordBaseline(500);
    // Ronda 1: probe a depth=1 pierde (600>500). current=2 > min ⇒ demote 2→1.
    var d = c.tick();
    try std.testing.expectEqual(@as(u32, 1), d);
    c.recordTiming(600);
    c.recordAccepted(d, 600);
    c.evaluateProbe();
    try std.testing.expectEqual(@as(u32, 1), c.activeDepth());

    // Ronda 2: current==min=1 ⇒ nextProbeDepth devuelve +1=2 (solo queda
    // subir para explorar). Pierde: pd=2 >= current=1, current <= min
    // ⇒ shutdown completo (ninguna profundidad disponible supera baseline).
    d = c.tick();
    try std.testing.expectEqual(@as(u32, 2), d);
    c.recordTiming(700);
    c.recordAccepted(d, 700);
    c.evaluateProbe();
    try std.testing.expectEqual(@as(u32, 0), c.activeDepth());
    try std.testing.expectEqual(adaptive.Decision.shutdown, c.lastDecision());
}

test "re-baseline: refresca baseline_us sin alterar current_depth" {
    var c = adaptive.ProfitController.init(.{ .max_depth = 4, .baseline_interval = 4 });
    c.recordBaseline(1000);
    const initial_depth = c.activeDepth();
    // 4 ciclos con timing neutro (no probe porque cycles_since_reset % 4 != 0
    // hasta el ciclo 4, donde probe_interval=64 lo evita).
    for (0..4) |_| {
        c.recordTiming(900);
    }
    c.recordBaseline(950); // nuevo baseline
    try std.testing.expectEqual(initial_depth, c.activeDepth());
}

test "recordRejected: API presente, no afecta decisión" {
    var c = adaptive.ProfitController.init(.{ .max_depth = 4 });
    c.recordBaseline(1000);
    c.recordRejected(); // no-op por diseño
    try std.testing.expectEqual(@as(u32, 4), c.activeDepth());
}

test "Decisión expuesta: valores cubren el enum completo" {
    // Compilar-time check: enum values son estables.
    try std.testing.expectEqual(@as(u8, 0), @intFromEnum(adaptive.Decision.hold));
    try std.testing.expectEqual(@as(u8, 4), @intFromEnum(adaptive.Decision.shutdown));
}
