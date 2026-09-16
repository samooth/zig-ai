//! Reasoning-loop guard (lane-b3, P1.2 + P3.1).
//!
//! Detecta bucles patológicos de generación:
//!   - `hidden`: dentro del razonamiento del modelo (reasoning tokens)
//!   - `visible`: el output visible repite la MISMA secuencia demasiadas
//!     veces (extensión zig-ai, Bee v0.4.4)
//!   - `both`: ambos detectores activos
//!
//! ## Algoritmo (Bee `server-loop-guard.cpp/h`)
//!
//! 1. **Rolling hash incremental** sobre la ventana de `window` tokens:
//!    hash FNV-1a de bloques alineados por periodo `p ∈ [min_period, max_period]`.
//! 2. **Detección de período**: el MISMO hash de secuencia aparece
//!    `max_repeats` veces (default 3) → loop confirmado.
//! 3. **Intervención** (`mode`):
//!    - `force-close` (default): el caller inyecta la secuencia de fin de
//!      razonamiento por la ruta full-logits (NUNCA por reduced-verify
//!      especulativo — lección Bee v0.1.2). El guard solo REPORTA el evento;
//!      la intervención vive en el pipeline (callback `onLoopDetected`).
//!    - `warn`: solo breadcrumb + métrica (`loop_guard_events`).
//!    - `off`: detector dormido (overhead ~0).
//!
//! ## Exclusiones (lección Bee v0.4.4)
//!
//! "fenced code and embedded marker-like strings are excluded": el caller
//! alimenta los tokens de código delimitado / marcadores tool-call vía
//! `pushExcluded()` para que NO cuenten en ningún detector.
//!
//! ## Overhead
//!
//! O(periods) por token (solo xor+mul, sin allocs, sin branches caros).
//! Objetivo <0.5% en decode (verificar con PERF_SPEC).
const std = @import("std");
pub const debugz = @import("debug");

/// Periodo máximo rastreado por diseño (ventana Bee default 512).
pub const max_tracked_period = 512;

/// Evento de loop detectado — el caller decide la intervención.
pub const LoopEvent = struct {
    /// Periodo detectado (tokens por repetición).
    period: u32 = 0,
    /// Posición absoluta (tokens del canal) donde se confirmó el loop.
    first_seen: u64 = 0,
    /// Canal del loop: reasoning oculto o output visible.
    channel: Channel = .hidden,
    /// Whether a loop was triggered.
    triggered: bool = false,
    /// Confidence score (0.0-1.0).
    score: f32 = 0,
};

pub const Channel = enum(u8) {
    hidden = 0,
    visible = 1,
};

/// Modo de intervención (`--reasoning-loop-mode`).
pub const Mode = enum(u8) {
    /// Inyecta fin de razonamiento vía full-logits.
    force_close = 0,
    /// Solo breadcrumb + métrica.
    warn = 1,
    /// Detector dormido.
    off = 2,

    pub fn fromCli(s: []const u8) ?Mode {
        const map = .{
            .{ "force-close", .force_close },
            .{ "warn", .warn },
            .{ "off", .off },
        };
        inline for (map) |m| {
            if (std.mem.eql(u8, s, m[0])) return m[1];
        }
        return null;
    }
};

/// Qué canales vigila el guard (`--reasoning-loop-channel`, extensión P3.1).
pub const ChannelCfg = enum(u8) {
    hidden = 0,
    visible = 1,
    both = 2,

    pub fn fromCli(s: []const u8) ?ChannelCfg {
        const map = .{
            .{ "hidden", .hidden },
            .{ "visible", .visible },
            .{ "both", .both },
        };
        inline for (map) |m| {
            if (std.mem.eql(u8, s, m[0])) return m[1];
        }
        return null;
    }
};

pub const Config = struct {
    /// Tamaño de ventana de tokens razonados (--reasoning-loop-window).
    window: u32 = 512,
    /// Periodo mínimo a rastrear (evita ruido en periodos cortos).
    min_period: u32 = 8,
    /// Periodo máximo (cap: window y max_tracked_period).
    max_period: u32 = 512,
    /// Repeticiones del mismo hash para confirmar loop
    /// (--reasoning-loop-max-period; default Bee 3).
    max_repeats: u32 = 3,
    /// Modo de intervención (--reasoning-loop-mode).
    mode: Mode = .force_close,
    /// Canales vigilados (--reasoning-loop-channel, P3.1).
    channels: ChannelCfg = .hidden,
    /// Repeticiones extra exigidas al canal visible (párrafo patológico:
    /// Bee v0.4.4 usa 5+ en visible vs 3 en hidden).
    visible_extra_repeats: u32 = 2,
};

/// Estado por (periodo, canal): hash del bloque alineado en curso y del
/// último completo, con contador de repeticiones.
const PeriodTracker = struct {
    /// Hash del último bloque COMPLETO de longitud p.
    last_hash: u64 = 0,
    /// Hash del bloque EN CURSO (incremental FNV-1a).
    cur_hash: u64 = 0,
    /// Tokens acumulados en el bloque en curso.
    fill: u32 = 0,
    /// Bloques consecutivos con el MISMO hash (1 = primer bloque).
    repeat_count: u32 = 0,
};

/// LoopGuard — sin allocator: trackers inline (2 canales × 512 periodos ×
/// 32B ≈ 32KB). Pensado para vivir en el SpecDriver/estado del engine, NO
/// en stack.
pub const LoopGuard = struct {
    cfg: Config,
    /// Posición absoluta de tokens hidden procesados (first_seen).
    hidden_pos: u64 = 0,
    /// Posición absoluta de tokens visible procesados.
    visible_pos: u64 = 0,
    /// Estado reasoning (canal hidden).
    in_reasoning: bool = false,
    /// Trackers hidden, indexados por periodo (índice = p).
    hidden_trackers: [max_tracked_period + 1]PeriodTracker = @splat(.{}),
    /// Trackers visible.
    visible_trackers: [max_tracked_period + 1]PeriodTracker = @splat(.{}),
    /// Contador de eventos (métrica loop_guard_events).
    events: u64 = 0,
    /// Último evento (DUMP_SPEC / callback onLoopDetected).
    last_event: ?LoopEvent = null,

    const Self = @This();

    pub fn init(cfg: Config) Self {
        var c = cfg;
        if (c.window == 0) c.window = 512;
        if (c.min_period == 0) c.min_period = 1;
        if (c.max_period < c.min_period) c.max_period = c.min_period;
        if (c.max_repeats < 2) c.max_repeats = 3;
        if (c.max_period > c.window) c.max_period = c.window;
        if (c.max_period > max_tracked_period) c.max_period = max_tracked_period;
        return .{ .cfg = c };
    }

    /// Feed de un token del canal razonamiento (hidden). El flanco off→on
    /// de `in_reasoning` resetea la fase de los trackers hidden (los bloques
    /// a mitad no comparan contra hashes de segmentos previos).
    pub fn pushToken(self: *Self, tok: u32, in_reasoning: bool) ?LoopEvent {
        if (self.cfg.mode == .off) return null;
        if (in_reasoning and !self.in_reasoning) self.resetPhase(.hidden);
        self.in_reasoning = in_reasoning;
        if (in_reasoning and self.watches(.hidden)) {
            return self.feedHidden(tok);
        }
        return null;
    }

    /// Feed de un token del output visible (extensión P3.1).
    pub fn pushVisibleToken(self: *Self, tok: u32) ?LoopEvent {
        if (self.cfg.mode == .off) return null;
        if (!self.watches(.visible)) return null;
        return self.feedVisible(tok);
    }

    /// Feed de tokens EXCLUIDOS (fenced code, marcadores tool-call): no
    /// alimentan ningún detector (lección Bee v0.4.4), solo avanzan posición.
    pub fn pushExcluded(self: *Self) void {
        self.visible_pos +|= 1;
    }

    fn watches(self: *const Self, ch: Channel) bool {
        return switch (self.cfg.channels) {
            .hidden => ch == .hidden,
            .visible => ch == .visible,
            .both => true,
        };
    }

    fn feedHidden(self: *Self, tok: u32) ?LoopEvent {
        self.hidden_pos +|= 1;
        return self.feedTrackers(&self.hidden_trackers, tok, .hidden);
    }

    fn feedVisible(self: *Self, tok: u32) ?LoopEvent {
        self.visible_pos +|= 1;
        return self.feedTrackers(&self.visible_trackers, tok, .visible);
    }

    /// Núcleo: alimenta los trackers de periodo [min_period, max_period] con
    /// un token y decide. Devuelve el LoopEvent del PRIMER periodo que
    /// confirma loop (el más corto — mayor confianza).
    fn feedTrackers(self: *Self, trackers: *[max_tracked_period + 1]PeriodTracker, tok: u32, ch: Channel) ?LoopEvent {
        const threshold = self.repeatThreshold(ch);
        const lo = self.cfg.min_period;
        const hi = self.cfg.max_period;
        var p: u32 = lo;
        while (p <= hi) : (p += 1) {
            const t = &trackers[p];
            // Hash incremental FNV-1a del bloque en curso (fase alineada
            // al inicio del segmento).
            t.cur_hash = (t.cur_hash ^ tok) *% 0x100000001b3;
            t.fill +|= 1;
            if (t.fill == p) {
                // Bloque completo: comparar contra el anterior.
                if (t.cur_hash == t.last_hash) {
                    t.repeat_count +|= 1;
                } else {
                    t.repeat_count = 1;
                }
                t.last_hash = t.cur_hash;
                t.cur_hash = 0;
                t.fill = 0;
                if (t.repeat_count >= threshold) {
                    return self.report(p, ch);
                }
            }
        }
        return null;
    }

    fn repeatThreshold(self: *const Self, ch: Channel) u32 {
        return switch (ch) {
            .hidden => self.cfg.max_repeats,
            // Visible exige más repeticiones (párrafo patológico 5+).
            .visible => self.cfg.max_repeats + self.cfg.visible_extra_repeats,
        };
    }

    fn report(self: *Self, period: u32, ch: Channel) LoopEvent {
        const ev = LoopEvent{
            .period = period,
            .first_seen = switch (ch) {
                .hidden => self.hidden_pos,
                .visible => self.visible_pos,
            },
            .channel = ch,
        };
        self.events +|= 1;
        self.last_event = ev;
        if (debugz.dbg.at(.info)) {
            debugz.dbg.printLevel(.info, "[loop_guard] loop {s} detectado: period={d} pos={d}\n", .{
                @tagName(ch),
                period,
                ev.first_seen,
            });
        }
        // Reset de fase tras el report: un evento por segmento, no spam.
        self.resetPhase(ch);
        return ev;
    }

    /// Reset de fase de un canal (nuevo segmento o post-evento).
    fn resetPhase(self: *Self, ch: Channel) void {
        const trackers = switch (ch) {
            .hidden => &self.hidden_trackers,
            .visible => &self.visible_trackers,
        };
        inline for (trackers) |*t| t.* = .{};
    }

    /// Reset completo (context-bucket nuevo): limpia posiciones y trackers.
    pub fn reset(self: *Self) void {
        self.hidden_pos = 0;
        self.visible_pos = 0;
        self.in_reasoning = false;
        self.events = 0;
        self.last_event = null;
        self.resetPhase(.hidden);
        self.resetPhase(.visible);
    }

    /// Feed a token (alias for pushToken with in_reasoning=true).
    pub fn feed(self: *Self, tok: u32) void {
        _ = self.pushToken(tok, true);
    }

    /// Check if a loop was detected in the last feed.
    pub fn check(self: *const Self) LoopEvent {
        return self.last_event orelse .{ .triggered = false, .period = 0, .score = 0 };
    }
};
