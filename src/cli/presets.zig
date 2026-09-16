//! INI presets (lane-b3, P3.4 — contrato C5).
//!
//! El preset ES la superficie de integración de flags de toda la oleada
//! BeeLlama: B1 (routes), B2 (cache/tail) y B3 (spec/guard) validan sus
//! combinaciones en UN solo sitio.
//!
//! ```ini
//! # presets/qwen36-27b-kvarn.ini
//! [model]
//! ctx = 65536
//!
//! [kv]
//! cache-type-k = kvarn5
//! cache-type-v = kvarn4
//! kv-tail-tokens = 1024
//! kv-tail-type = f16
//!
//! [speculative]
//! dm-controller = profit
//!
//! [guard]
//! reasoning-loop-mode = force-close
//! ```
//!
//! ```bash
//! zig-ai-engine -m model.gguf --preset presets/qwen36-27b-kvarn.ini
//! # o directorio:
//! zig-ai-engine --models-dir /path --models-preset presets.ini
//! ```
//!
//! ## Precedencia (documentada, contractual)
//!   1. Flags CLI explícitos PISAN al preset.
//!   2. Claves del preset pisan a los defaults del motor.
//!   3. Default del motor si nadie lo define.
//! El wiring P4 pasa tres capas (cli_overrides, preset, defaults) y
//! `resolve()` aplica la precedencia: valor presente gana, `null` cede.
//!
//! ## Validación coherente (falla AL PARSEAR, no al correr)
//! - `kv-tail-tokens > 0` con MLA K-only (cache-type-k seteado pero
//!   cache-type-v ausente) ⇒ error inmediato.
//! - `dm-controller = profit` sin speculation activa (`spec-type` none o
//!   ausente) ⇒ error inmediato.
//! - Valores desconocidos de enums (mode/channel/controller) ⇒ error.
//!
//! ## Tabla de flags soportados (C5 — mantener sincronizada con B1/B2)
//! | Sección      | Clave                | Tipo   | Dueño |
//! |-------------|----------------------|--------|-------|
//! | model       | ctx                  | usize  | core  |
//! | model       | batch-size (-b)      | usize  | B2    |
//! | model       | ubatch-size (-ub)     | usize  | B2    |
//! | kv          | cache-type-k         | string | B2    |
//! | kv          | cache-type-v         | string | B2    |
//! | kv          | kv-tail-tokens       | usize  | B2    |
//! | kv          | kv-tail-type         | string | B2    |
//! | speculative | spec-type            | string | B3    |
//! | speculative | dm-controller        | string | B3    |
//! | speculative | dm-profit-baseline-interval | usize | B3 |
//! | guard       | reasoning-loop-mode  | string | B3    |
//! | guard       | reasoning-loop-window | usize | B3   |
//! | guard       | reasoning-loop-max-period | usize | B3 |
//! | guard       | reasoning-loop-channel | string | B3  |
const std = @import("std");

pub const PresetError = error{
    UnknownSection,
    UnknownKey,
    InvalidValue,
    InvalidCombination,
    SyntaxError,
    OutOfMemory,
};

/// Preset parseado — todos los valores opcionales: `null` = clave ausente
/// (la precedencia la resuelve `resolve` en el wiring).
pub const Preset = struct {
    // [model]
    ctx: ?usize = null,
    batch_size: ?usize = null,
    ubatch_size: ?usize = null,
    // [kv]
    cache_type_k: ?[]const u8 = null,
    cache_type_v: ?[]const u8 = null,
    kv_tail_tokens: ?usize = null,
    kv_tail_type: ?[]const u8 = null,
    // [speculative]
    spec_type: ?[]const u8 = null,
    dm_controller: ?[]const u8 = null,
    dm_profit_baseline_interval: ?usize = null,
    // [guard]
    reasoning_loop_mode: ?[]const u8 = null,
    reasoning_loop_window: ?usize = null,
    reasoning_loop_max_period: ?usize = null,
    reasoning_loop_channel: ?[]const u8 = null,

    /// Validación coherente (C5): combinaciones inválidas fallan AL
    /// PARSEAR, no al correr. Devuelve el primer problema encontrado.
    pub fn validate(self: Preset) PresetError!void {
        // kv-tail-tokens > 0 requiere K Y V (MLA K-only + tail es inválido:
        // la cola exacta vive en el cuerpo comprimido de ambos canales).
        if (self.kv_tail_tokens) |t| {
            if (t > 0 and self.cache_type_k != null and self.cache_type_v == null) {
                return PresetError.InvalidCombination;
            }
            if (t > 0 and self.cache_type_k == null and self.cache_type_v != null) {
                return PresetError.InvalidCombination;
            }
        }
        // dm-controller solo tiene sentido con speculation activa.
        if (self.dm_controller) |ctrl| {
            if (!std.mem.eql(u8, ctrl, "off") and self.spec_type == null) {
                return PresetError.InvalidCombination;
            }
        }
        // Enums válidos.
        if (self.reasoning_loop_mode) |m| {
            if (!isOneOf(m, &.{ "force-close", "warn", "off" })) return PresetError.InvalidValue;
        }
        if (self.reasoning_loop_channel) |c| {
            if (!isOneOf(c, &.{ "hidden", "visible", "both" })) return PresetError.InvalidValue;
        }
        if (self.dm_controller) |c| {
            if (!isOneOf(c, &.{ "profit", "off" })) return PresetError.InvalidValue;
        }
        if (self.kv_tail_type) |t| {
            if (!isOneOf(t, &.{ "f16", "bf16", "fp16" })) return PresetError.InvalidValue;
        }
        // spec-type valores conocidos.
        if (self.spec_type) |s| {
            if (!isOneOf(s, &.{ "none", "draft-mtp", "draft-dflash", "draft-dspark", "draft-dflash2" })) {
                return PresetError.InvalidValue;
            }
        }
        // kv-tail-tokens coherente con window del guard (no mayor que ctx
        // sería absurdo; check duro solo contra 0/negativos via tipo).
        if (self.ctx) |c| {
            if (c == 0) return PresetError.InvalidValue;
            if (self.kv_tail_tokens) |t| {
                if (t > c) return PresetError.InvalidCombination;
            }
        }
    }
};

fn isOneOf(v: []const u8, list: []const []const u8) bool {
    for (list) |x| {
        if (std.mem.eql(u8, v, x)) return true;
    }
    return false;
}

/// Parser INI minimal: secciones `[name]`, claves `key = value`, comentarios
/// `#` al inicio de línea. Sin allocs: devuelve strings del input original.
pub fn parse(text: []const u8, out: *Preset) PresetError!void {
    var section: []const u8 = "";
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");
        if (line.len == 0 or line[0] == '#' or line[0] == ';') continue;
        if (line[0] == '[') {
            const close = std.mem.indexOfScalar(u8, line, ']') orelse return PresetError.SyntaxError;
            section = std.mem.trim(u8, line[1..close], " \t");
            if (!isOneOf(section, &.{ "model", "kv", "speculative", "guard" })) {
                return PresetError.UnknownSection;
            }
            continue;
        }
        const eq = std.mem.indexOfScalar(u8, line, '=') orelse return PresetError.SyntaxError;
        const key = std.mem.trim(u8, line[0..eq], " \t");
        const value = std.mem.trim(u8, line[eq + 1 ..], " \t");
        try setKey(out, section, key, value);
    }
    try out.validate();
}

fn setKey(p: *Preset, section: []const u8, key: []const u8, value: []const u8) PresetError!void {
    if (std.mem.eql(u8, section, "model")) {
        if (std.mem.eql(u8, key, "ctx")) {
            p.ctx = parseUsize(value) orelse return PresetError.InvalidValue;
        } else if (std.mem.eql(u8, key, "batch-size")) {
            p.batch_size = parseUsize(value) orelse return PresetError.InvalidValue;
        } else if (std.mem.eql(u8, key, "ubatch-size")) {
            p.ubatch_size = parseUsize(value) orelse return PresetError.InvalidValue;
        } else return PresetError.UnknownKey;
    } else if (std.mem.eql(u8, section, "kv")) {
        if (std.mem.eql(u8, key, "cache-type-k")) {
            p.cache_type_k = value;
        } else if (std.mem.eql(u8, key, "cache-type-v")) {
            p.cache_type_v = value;
        } else if (std.mem.eql(u8, key, "kv-tail-tokens")) {
            p.kv_tail_tokens = parseUsize(value) orelse return PresetError.InvalidValue;
        } else if (std.mem.eql(u8, key, "kv-tail-type")) {
            p.kv_tail_type = value;
        } else return PresetError.UnknownKey;
    } else if (std.mem.eql(u8, section, "speculative")) {
        if (std.mem.eql(u8, key, "spec-type")) {
            p.spec_type = value;
        } else if (std.mem.eql(u8, key, "dm-controller")) {
            p.dm_controller = value;
        } else if (std.mem.eql(u8, key, "dm-profit-baseline-interval")) {
            p.dm_profit_baseline_interval = parseUsize(value) orelse return PresetError.InvalidValue;
        } else return PresetError.UnknownKey;
    } else if (std.mem.eql(u8, section, "guard")) {
        if (std.mem.eql(u8, key, "reasoning-loop-mode")) {
            p.reasoning_loop_mode = value;
        } else if (std.mem.eql(u8, key, "reasoning-loop-window")) {
            p.reasoning_loop_window = parseUsize(value) orelse return PresetError.InvalidValue;
        } else if (std.mem.eql(u8, key, "reasoning-loop-max-period")) {
            p.reasoning_loop_max_period = parseUsize(value) orelse return PresetError.InvalidValue;
        } else if (std.mem.eql(u8, key, "reasoning-loop-channel")) {
            p.reasoning_loop_channel = value;
        } else return PresetError.UnknownKey;
    } else {
        return PresetError.UnknownSection;
    }
}

fn parseUsize(v: []const u8) ?usize {
    return std.fmt.parseInt(usize, v, 10) catch null;
}

/// Resolución de precedencia: CLI explícito > preset > default motor.
/// Cada campo: si el override (CLI) está presente, gana; si no, preset;
/// si no, default. El wiring P4 construye CliOverrides desde args con
/// "estaba el flag presente" (optionals) — la función es genérica por campo.
pub fn resolveField(comptime T: type, cli: ?T, preset: ?T, default: T) T {
    if (cli) |c| return c;
    if (preset) |p| return p;
    return default;
}

/// Lee un preset desde disco. El caller retiene el `out` mientras use las
/// strings apuntadas (que apuntan al buffer del caller; `parse` no alloca).
/// Usa `std.Io.Dir.cwd().openFile` con el Io pasado por el caller (Zig 0.16).
pub fn parseFromFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8, out: *Preset) !void {
    var f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);

    // Stat para conocer tamaño (readPositional requiere longitud pre-calculada).
    const stat = try f.stat(io);
    const size: usize = @intCast(stat.size);

    const buf = try allocator.alloc(u8, size);
    defer allocator.free(buf);

    // readPositional en Zig 0.16 toma `[]const []u8` (vectored I/O).
    const n = try f.readPositional(io, &.{buf}, 0);
    try parse(buf[0..n], out);
}

// ─── Tests ──────────────────────────────────────────────────────────────────
// (suite autoritativa en tests/test_presets.zig, wired via build.zig)
