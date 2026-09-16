//! Adaptive draft-max controller — `profit` strategy (lane-b3, P1.1).
//!
//! Ajusta EN RUNTIME la profundidad de draft (`--spec-draft-n-max`) según
//! si cada profundidad gana contra un baseline SIN speculative. Inspirado
//! en Bee `server-adaptive-dm.h` (controlador `profit`).
//!
//! ## Algoritmo
//!
//! 1. **Baseline sin-spec**: se siembra ANTES de probes positivos (lección
//!    Bee v0.1.2: si no, el controller decide desde telemetría draft-only
//!    = sesgo). El caller mide ciclos sin spec y los reporta con
//!    `recordBaseline(us)`.
//! 2. **Probes gated**: cada `probe_interval` ciclos activos, prueba una
//!    profundidad alternativa (decreasing exploration). Sin degradar la
//!    experiencia: durante el probe se sigue usando la profundidad actual
//!    y solo se evalúa la alternativa.
//! 3. **Decisión**: si spec-depth N no supera al baseline → demote a N-1.
//!    Si NINGUNA supera → **shutdown completo** (depth = 0, DFlash off).
//! 4. **Re-baseline periódico**: cada `baseline_interval` ciclos activos,
//!    refresca el baseline. Al re-probar, retoma la profundidad previa
//!    (evita starve del counter).
//! 5. **Cold start**: mientras `baseline == null`, mantiene `max_depth`
//!    mientras caracteriza (Bee: "cold starts hold the maximum useful depth
//!    while lower depths are characterized through gated probes").
//!
//! ## Invariantes (lecciones Bee v0.1.2)
//!
//! - Reset correcto ante cambio de context-bucket/config: telemetry de
//!   baseline limpia NO puede dejar el controller en estado stale
//!   → `reset()` deja `baseline = null` y `current_depth = max_depth`.
//! - Nunca decide desde telemetría draft-only: `tick()` con `baseline == null`
//!   siempre devuelve `max_depth` (cold start hold).
//! - Shutdown completo: `activeDepth() == 0` ⇒ DFlash off. Re-activación
//!   requiere `recordBaseline(us)` que reactive el baseline.
//!
//! Sin allocator: estado embebido en el struct. Pensado para vivir en el
//! `SpecDriver` o como global del engine.
const std = @import("std");
pub const debugz = @import("debug");

/// Controller mode enum — used in CLI params and config.
pub const Controller = enum(u8) {
    off = 0,
    profit = 1,
};

/// Decisión del controller — expuesta vía métricas (`spec_adaptive_decision`).
pub const Decision = enum(u8) {
    /// Profundidad sin cambios (cold-start hold o probe sin telemetría).
    hold = 0,
    /// Probe ejecutado; pendiente de decisión (se acepta telemetría).
    probe = 1,
    /// Demote a profundidad inferior.
    demote = 2,
    /// Promote a profundidad superior (rare, post-shutdown reactivation).
    promote = 3,
    /// Shutdown completo (depth = 0). El caller debe apagar el drafter.
    shutdown = 4,
};

/// Configuración del controller. Construir desde CLI flags:
///   --spec-dm-controller profit|off       (default: profit si spec activo)
///   --spec-dm-profit-baseline-interval <n> (default: 1024 ciclos activos)
pub const Config = struct {
    /// Profundidad máxima permitida (≤ SpecConfig.n_max del drafter).
    max_depth: u32 = 16,
    /// Base n_max for draft depth (used by cli.zig).
    base_n_max: u32 = 0,
    /// Profundidad mínima que se permite explorar (≥1, default 1).
    min_depth: u32 = 1,
    /// Cada cuántos ciclos activos se refresca el baseline.
    baseline_interval: u32 = 1024,
    /// Cada cuántos ciclos activos se ejecuta un probe gated.
    probe_interval: u32 = 64,
    /// Si el candidato de probe mejora el baseline por al menos este ratio,
    /// se acepta. Default 1% (es conservador — la mejora tiende a ser mayor
    /// cuando spec realmente ayuda).
    promote_ratio: f32 = 1.01,
};

/// Estado interno — sin allocator. El struct entero vive en stack/globales.
pub const ProfitController = struct {
    cfg: Config,
    /// Última profundidad activa (0 = speculation OFF).
    current_depth: u32,
    /// EMA del cycle_us sin spec (baseline). `null` ⇒ no caracterizado.
    baseline_us: ?u64,
    /// EMA por profundidad probada (clave = profundidad N, valor = us).
    /// Tamaño = max_depth + 1 (índice 0 ignorado).
    depth_ema: [16]u64,
    /// EMA del cycle_us medido en el ÚLTIMO probe activo (para comparar).
    last_probe_us: ?u64,
    /// Profundidad del probe pendiente de evaluación.
    pending_probe_depth: ?u32,
    /// Ciclos activos desde el último reset.
    cycles_since_reset: u64,
    /// Ciclos activos desde el último re-baseline.
    cycles_since_baseline: u64,
    /// Última decisión (para `spec_adaptive_decision`).
    last_decision: Decision,

    const Self = @This();

    /// Constructor. Estado limpio: cold-start en `max_depth`, baseline=null.
    pub fn init(cfg: Config) Self {
        const depth = if (cfg.base_n_max > 0) cfg.base_n_max else cfg.max_depth;
        std.debug.assert(depth > 0 and depth <= 16);
        std.debug.assert(cfg.min_depth >= 1 and cfg.min_depth <= depth);
        return .{
            .cfg = cfg,
            .current_depth = depth,
            .baseline_us = null,
            .depth_ema = .{0} ** 16,
            .last_probe_us = null,
            .pending_probe_depth = null,
            .cycles_since_reset = 0,
            .cycles_since_baseline = 0,
            .last_decision = .hold,
        };
    }

    /// Reset por cambio de context-bucket/config. Estado limpio, sin stale.
    pub fn reset(self: *Self) void {
        self.current_depth = self.cfg.max_depth;
        self.baseline_us = null;
        self.depth_ema = .{0} ** 16;
        self.last_probe_us = null;
        self.pending_probe_depth = null;
        self.cycles_since_reset = 0;
        self.cycles_since_baseline = 0;
        self.last_decision = .hold;
    }

    /// Profundidad activa (0 = speculation OFF). El caller la usa como
    /// `n_max` del drafter en cada ronda.
    pub fn activeDepth(self: *const Self) u32 {
        return self.current_depth;
    }

    /// Última decisión (gated `DUMP_SPEC` y `metrics.spec_adaptive_decision`).
    pub fn lastDecision(self: *const Self) Decision {
        return self.last_decision;
    }

    /// Reporte del caller: ciclo SIN speculative completado en `cycle_us`
    /// microsegundos. Solo se acepta si `spec_active == false` (caller debe
    /// garantizarlo — el controller no tiene forma de saber).
    pub fn recordBaseline(self: *Self, cycle_us: u64) void {
        self.baseline_us = ema(self.baseline_us orelse cycle_us, cycle_us, 0.2);
        self.cycles_since_baseline = 0;
        if (debugz.dbg.at(.detail)) {
            debugz.dbg.printLevel(.detail, "[adaptive_dm] baseline_us={d}\n", .{self.baseline_us.?});
        }
    }

    /// Reporte del caller: ronda de spec-depth `n` aceptada en `cycle_us`.
    /// Acumula en la EMA de esa profundidad.
    pub fn recordAccepted(self: *Self, n: u32, cycle_us: u64) void {
        if (n == 0 or n > 16) return;
        self.depth_ema[n] = ema(self.depth_ema[n], cycle_us, 0.2);
    }

    /// Reporte del caller: ronda rechazada (no se acepta timing).
    /// Útil para telemetry pero NO afecta decisión (la decisión es sobre
    /// throughput total, no acceptance rate).
    pub fn recordRejected(_: *Self) void {}

    /// Reporte del caller: timing global del ciclo (decode completo,
    /// independiente del resultado de la ronda). Alimenta al probe en
    /// curso si lo hay. El counter `cycles_since_reset` se incrementa en
    /// `tick()` para que probe gate y EMA queden en la misma escala.
    pub fn recordTiming(self: *Self, cycle_us: u64) void {
        self.cycles_since_baseline += 1;
        if (self.pending_probe_depth) |pd| {
            self.last_probe_us = ema(self.last_probe_us orelse cycle_us, cycle_us, 0.3);
            if (debugz.dbg.at(.trace)) {
                debugz.dbg.printLevel(.trace, "[adaptive_dm] probe depth={d} us={d}\n", .{ pd, cycle_us });
            }
        }
        // Re-baseline periódico: hint al caller para que mida sin spec.
        if (self.cycles_since_baseline >= self.cfg.baseline_interval) {
            if (debugz.dbg.at(.detail)) {
                debugz.dbg.printLevel(.detail, "[adaptive_dm] re-baseline hint @ cycles_since_baseline={d}\n", .{self.cycles_since_baseline});
            }
        }
    }

    /// Decisión para el ciclo actual. Llamar UNA VEZ por ciclo ANTES de
    /// ejecutar el spec round. Devuelve la profundidad a usar.
    ///
    /// Si hay un probe pendiente, devuelve la profundidad alternativa
    /// (probe) — el caller DEBE medir timing sincrónicamente y reportar
    /// con `recordTiming`. Tras un número de ciclos suficiente, llamar
    /// `evaluateProbe()` para aplicar la decisión.
    pub fn tick(self: *Self) u32 {
        // Cold start: hold max depth mientras caracteriza.
        if (self.baseline_us == null) {
            self.last_decision = .hold;
            return self.cfg.max_depth;
        }
        // Shutdown previo: seguir en 0 hasta que el caller reactive con
        // un baseline fresco.
        if (self.current_depth == 0) {
            self.last_decision = .hold;
            return 0;
        }
        // Probe gated cada probe_interval ciclos. cycles_since_reset
        // cuenta cuántos ticks han transcurrido: en el tick N, llevamos N
        // ciclos activos; cuando (N+1) % probe_interval == 0, probe.
        const next_cycle = self.cycles_since_reset + 1;
        if (next_cycle % self.cfg.probe_interval == 0) {
            const probe_depth = self.nextProbeDepth();
            if (probe_depth) |pd| {
                self.pending_probe_depth = pd;
                self.last_probe_us = null;
                self.last_decision = .probe;
                if (debugz.dbg.at(.detail)) {
                    debugz.dbg.printLevel(.detail, "[adaptive_dm] probe start depth={d} (current={d})\n", .{ pd, self.current_depth });
                }
                self.cycles_since_reset = next_cycle;
                return pd;
            }
        }
        self.cycles_since_reset = next_cycle;
        self.last_decision = .hold;
        return self.current_depth;
    }

    /// Evalúa el probe en curso. Llamar tras recoger telemetría del ciclo
    /// probe (típicamente 1 ronda). Decide: hold, demote, promote, shutdown.
    pub fn evaluateProbe(self: *Self) void {
        const pd = self.pending_probe_depth orelse return;
        defer self.pending_probe_depth = null;
        const probe_us = self.last_probe_us orelse return;
        const baseline = self.baseline_us orelse return;

        // Ratio probe/baseline. <1 = probe más rápido.
        const ratio = @as(f32, @floatFromInt(probe_us)) / @as(f32, @floatFromInt(baseline));
        if (debugz.dbg.at(.detail)) {
            debugz.dbg.printLevel(.detail, "[adaptive_dm] evaluate probe={d} us={d} baseline={d} ratio={d:.3}\n", .{
                pd, probe_us, baseline, ratio,
            });
        }
        if (ratio < 1.0 / self.cfg.promote_ratio) {
            // Probe mucho mejor que baseline. Adoptamos la profundidad probe
            // (puede ser inferior si el drafter sobre-generaba, o superior
            // si cabe más profundidad útil). En ambos casos = promote.
            self.current_depth = pd;
            self.last_decision = .promote;
            return;
        }
        if (ratio <= 1.0) {
            // Probe marginalmente mejor: hold current.
            self.last_decision = .hold;
            return;
        }
        if (pd < self.current_depth) {
            // Probe peor que baseline. Demote al siguiente safe step.
            if (self.current_depth <= self.cfg.min_depth) {
                // Ya en min y probe pierde: shutdown completo.
                self.current_depth = 0;
                self.last_decision = .shutdown;
                return;
            }
            // Demote a pd (asumimos que si la probada no ayuda, ninguna
            // intermedia ayuda). Si pd > min, eso ya respeta el lower bound.
            self.current_depth = pd;
            self.last_decision = .demote;
            return;
        }
        if (self.current_depth <= self.cfg.min_depth) {
            // Ya estamos en min_depth y el probe no ayuda: shutdown.
            self.current_depth = 0;
            self.last_decision = .shutdown;
            return;
        }
        // Probe peor pero no podemos demote más: shutdown si NINGUNA profundidad
        // conocida ayuda. Aproximación simple: si probe es el más profundo y
        // pierde, shutdown.
        if (pd >= self.current_depth) {
            self.current_depth = 0;
            self.last_decision = .shutdown;
            return;
        }
        self.last_decision = .hold;
    }

    /// Próxima profundidad a probar (None si ya probamos todas o si min=max).
    /// Estrategia simple: alternar entre (current-1) y (current+1) capped.
    fn nextProbeDepth(self: *const Self) ?u32 {
        if (self.cfg.min_depth == self.cfg.max_depth) return null;
        // Intentar una profundidad menor primero (más segura).
        if (self.current_depth > self.cfg.min_depth) {
            return self.current_depth - 1;
        }
        if (self.current_depth < self.cfg.max_depth) {
            return self.current_depth + 1;
        }
        return null;
    }

    fn ema(prev: u64, curr: u64, alpha: f32) u64 {
        const p: f32 = @floatFromInt(prev);
        const c: f32 = @floatFromInt(curr);
        const blended = alpha * c + (1.0 - alpha) * p;
        return @intFromFloat(blended);
    }

    pub const RoundObservation = struct {
        requested_n_max: u32 = 0,
        n_draft: u32 = 0,
        n_accepted: u32 = 0,
        draft_ms: f32 = 0,
        verify_ms: f32 = 0,
        accept_ms: f32 = 0,
        cycle_ms: f32 = 0,
    };

    pub const DecideResult = struct {
        recommended_n_max: i32 = -1,
        reason: []const u8 = "",
    };

    pub fn recordRound(self: *Self, obs: RoundObservation) void {
        if (obs.cycle_ms > 0) {
            self.recordTiming(@intFromFloat(obs.cycle_ms * 1e6));
        }
        if (obs.n_accepted > 0) {
            self.recordAccepted(obs.n_accepted, @intFromFloat(obs.cycle_ms * 1e6));
        } else {
            self.recordRejected();
        }
    }

    pub fn decide(self: *Self) DecideResult {
        _ = self.tick();
        self.evaluateProbe();
        const d = self.activeDepth();
        return .{
            .recommended_n_max = @intCast(d),
            .reason = switch (self.last_decision) {
                .hold => if (self.baseline_us == null) "cold-start" else if (self.current_depth == 0) "shutdown-stable" else "ewma-hold",
                .probe => "probe",
                .demote => "demote",
                .promote => "promote",
                .shutdown => "shutdown",
            },
        };
    }
};

// Tests: la suite autoritativa vive en tests/test_adaptive_dm.zig
// (wired via build.zig con el módulo `debug` disponible).
