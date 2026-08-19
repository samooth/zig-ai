//! Instrumentación de diagnóstico centralizada.
//!
//! Toda la instrumentación vive en este módulo como "breadcrumbs": el código de
//! captura/dump está SIEMPRE presente en el árbol (no se borra), pero cada
//! punto solo emite si su flag está activo. Los flags se leen UNA vez de env en
//! `init()`; después son consultas baratas a campos del struct.
//!
//! Niveles (`DEBUG_LEVEL`, jerárquicos):
//!   0 = off      (por defecto: sin salida de diagnóstico)
//!   1 = info     (resumen de eventos relevantes)
//!   2 = detail   (datos por token/capa)
//!   3 = trace    (todo, valores completos)
//!
//! Flags por-dump (activan su breadcrumb independientemente del nivel):
//!   DUMPKV, DUMPNORM, DUMP_LOGITS, CHKSTATE, PERF_STAGE, DUMP_GRAPH
//!
//! Flags de comportamiento (también centralizados):
//!   NOGRAPH, NOGPU_PREFILL, NOQ4, NOQ4ATTN, NOQ4SSM, NOQ4FFN
const std = @import("std");

pub const Level = enum(u8) {
    off = 0,
    info = 1,
    detail = 2,
    trace = 3,

    pub fn fromEnv() Level {
        return parseLevel(std.c.getenv("DEBUG_LEVEL"));
    }
};

fn parseLevel(s: ?[*:0]const u8) Level {
    if (s) |v| {
        if (std.fmt.parseInt(u8, std.mem.span(v), 10)) |n| {
            if (n >= @intFromEnum(Level.off) and n <= @intFromEnum(Level.trace)) {
                return @enumFromInt(n);
            }
        } else |_| {}
    }
    return .off;
}

pub const Debug = struct {
    level: Level = .off,

    // Breadcrumbs de dump (env: DUMPKV, DUMPNORM, ...).
    dump_kv: bool = false,
    dump_norm: bool = false,
    dump_logits: bool = false,
    chk_state: bool = false,
    perf_stage: bool = false,
    dump_graph: bool = false,
    dump_prefill_layers: bool = false,

    // Flags de comportamiento (env: NOGRAPH, NOQ4, ...).
    no_graph: bool = false,
    no_gpu_prefill: bool = false,
    no_q4: bool = false,
    no_q4_attn: bool = false,
    no_q4_ssm: bool = false,
    no_q4_ffn: bool = false,

    pub fn init() Debug {
        return .{
            .level = Level.fromEnv(),
            .dump_kv = envOn("DUMPKV"),
            .dump_norm = envOn("DUMPNORM"),
            .dump_logits = envOn("DUMP_LOGITS"),
            .chk_state = envOn("CHKSTATE"),
            .perf_stage = envOn("PERF_STAGE"),
            .dump_graph = envOn("DUMP_GRAPH"),
            .dump_prefill_layers = envOn("DUMP_PREFILL_LAYERS"),
            .no_graph = envOn("NOGRAPH"),
            .no_gpu_prefill = envOn("NOGPU_PREFILL"),
            .no_q4 = envOn("NOQ4"),
            .no_q4_attn = envOn("NOQ4ATTN"),
            .no_q4_ssm = envOn("NOQ4SSM"),
            .no_q4_ffn = envOn("NOQ4FFN"),
        };
    }

    pub fn at(self: Debug, lvl: Level) bool {
        return @intFromEnum(self.level) >= @intFromEnum(lvl);
    }

    pub fn print(self: Debug, comptime fmt: []const u8, args: anytype) void {
        if (self.level == .off) return;
        std.debug.print(fmt, args);
    }

    pub fn printLevel(self: Debug, lvl: Level, comptime fmt: []const u8, args: anytype) void {
        if (!self.at(lvl)) return;
        std.debug.print(fmt, args);
    }
};

/// Instancia global (se lee de env en `init()`; leerla antes devuelve off).
pub var dbg: Debug = .{};
pub var load_count: usize = 0;

pub fn init() void {
    dbg = Debug.init();
}

fn envOn(name: [:0]const u8) bool {
    return std.c.getenv(name) != null;
}

// ─── Helpers de breadcrumb ──────────────────────────────────────────────────
// Funciones SIEMPRE presentes; el punto de llamada decide con qué flag/nivel
// emitir. Reducen la duplicación de sumas/máximos en los dumps.

pub fn sumAbsF32(slice: []const f32) f64 {
    var s: f64 = 0;
    for (slice) |v| s += @abs(@as(f64, v));
    return s;
}

pub fn maxAbsF32(slice: []const f32) f32 {
    var mx: f32 = 0;
    for (slice) |v| {
        const a = @abs(v);
        if (a > mx) mx = a;
    }
    return mx;
}

/// Suma de valores absolutos de un slice de f16 visto como u16 (KV cache).
pub fn sumAbsF16(u16s: []const u16) f64 {
    var s: f64 = 0;
    for (u16s) |v| s += @abs(@as(f64, @as(f32, @floatFromInt(v))));
    return s;
}

pub fn sumAbsF32Slice(slice: []const f32) f64 {
    return sumAbsF32(slice);
}

pub fn sumAbsF16Slice(u16s: []const u16) f64 {
    return sumAbsF16(u16s);
}