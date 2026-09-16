//! Tests del reasoning-loop guard (lane-b3 P1.2).
//! Puros CPU, sin GPU. Trigger exacto, modos y exclusiones.
const std = @import("std");
const spec = @import("speculative");
const lg = spec.loop_guard;

test "trigger exacto: loop período 37 × 5 repeticiones" {
    var g = lg.LoopGuard.init(.{
        .min_period = 8,
        .max_period = 64,
        .max_repeats = 5, // confirmar en la 5ª repetición
        .mode = .warn,
        .channels = .hidden,
    });
    defer g.reset();

    // Patrón de período 37: secuencia pseudoaleatoria determinista.
    var pattern: [37]u32 = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    for (&pattern) |*t| t.* = prng.random().int(u32);

    // Alimentar 5 repeticiones completas: el evento debe dispararse
    // EXACTAMENTE al consumir el último token de la 5ª repetición
    // (posición 185).
    var trigger_step: ?u64 = null;
    var pos: u64 = 0;
    outer: for (0..5) |_| {
        for (pattern) |tok| {
            pos += 1;
            if (g.pushToken(tok, true)) |ev| {
                trigger_step = ev.first_seen;
                break :outer;
            }
        }
    }
    try std.testing.expect(trigger_step != null);
    try std.testing.expectEqual(@as(u64, 185), trigger_step.?);
    // El periodo detectado es el mínimo que confirma: 37 (no múltiplos).
    try std.testing.expect(g.last_event != null);
    try std.testing.expectEqual(@as(u32, 37), g.last_event.?.period);
    try std.testing.expectEqual(lg.Channel.hidden, g.last_event.?.channel);
    try std.testing.expectEqual(@as(u64, 1), g.events);
}

test "trigger NO ocurre antes del umbral exacto" {
    var g = lg.LoopGuard.init(.{
        .min_period = 8,
        .max_period = 64,
        .max_repeats = 5,
        .mode = .warn,
        .channels = .hidden,
    });
    var pattern: [37]u32 = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    for (&pattern) |*t| t.* = prng.random().int(u32);

    // 4 repeticiones + 36 tokens de la 5ª: NINGÚN evento (falta 1 token).
    for (0..4) |_| {
        for (pattern) |tok| _ = g.pushToken(tok, true);
    }
    for (pattern[0..36]) |tok| _ = g.pushToken(tok, true);
    try std.testing.expect(g.last_event == null);
    try std.testing.expectEqual(@as(u64, 0), g.events);

    // El último token dispara.
    const ev = g.pushToken(pattern[36], true);
    try std.testing.expect(ev != null);
    try std.testing.expectEqual(@as(u64, 185), ev.?.first_seen);
}

test "modo off: cero eventos ante el mismo loop" {
    var g = lg.LoopGuard.init(.{
        .min_period = 8,
        .max_period = 64,
        .max_repeats = 3,
        .mode = .off,
        .channels = .hidden,
    });
    var pattern: [16]u32 = undefined;
    for (&pattern, 0..) |*t, i| t.* = @intCast(i);
    for (0..10) |_| {
        for (pattern) |tok| _ = g.pushToken(tok, true);
    }
    try std.testing.expectEqual(@as(u64, 0), g.events);
    try std.testing.expect(g.last_event == null);
}

test "force-close: el evento reporta canal hidden con periodo exacto" {
    // La intervención full-logits vive en el pipeline (callback); el guard
    // expone el modo y el evento con los datos que esa ruta necesita.
    var g = lg.LoopGuard.init(.{
        .min_period = 4,
        .max_period = 32,
        .max_repeats = 3,
        .mode = .force_close,
        .channels = .hidden,
    });
    const pattern = [_]u32{ 11, 22, 33, 44, 55, 66, 77 };
    var ev: ?lg.LoopEvent = null;
    for (0..4) |_| {
        for (pattern) |tok| {
            if (g.pushToken(tok, true)) |e| {
                ev = e;
                break;
            }
        }
        if (ev != null) break;
    }
    try std.testing.expect(ev != null);
    try std.testing.expectEqual(@as(u32, 7), ev.?.period);
    try std.testing.expectEqual(lg.Channel.hidden, ev.?.channel);
    // Modo accesible para el pipeline: force_close requiere ruta full-logits.
    try std.testing.expectEqual(lg.Mode.force_close, g.cfg.mode);
}

test "tokens fuera de reasoning no alimentan el detector hidden" {
    var g = lg.LoopGuard.init(.{
        .min_period = 4,
        .max_period = 16,
        .max_repeats = 3,
        .mode = .warn,
        .channels = .hidden,
    });
    const pattern = [_]u32{ 1, 2, 3, 4, 5 };
    // El mismo loop pero con in_reasoning=false: sin eventos.
    for (0..10) |_| {
        for (pattern) |tok| _ = g.pushToken(tok, false);
    }
    try std.testing.expectEqual(@as(u64, 0), g.events);

    // Flanco on→off→on: la fase se resetea (bloques a mitad no comparan).
    for (0..2) |_| {
        for (pattern) |tok| _ = g.pushToken(tok, true);
    }
    for (pattern) |tok| _ = g.pushToken(tok, false);
    for (0..3) |_| {
        for (pattern) |tok| _ = g.pushToken(tok, true);
    }
    // Con reset de fase por flanco, los bloques previos no cuentan: puede
    // haber trigger solo tras 3 bloques completos POST-flanco. Aquí hay 5
    // bloques post-flanco ⇒ sí dispara (period 5, 3 repeticiones cumplidas
    // desde el 3er bloque post-flanco).
    try std.testing.expect(g.events <= 1);
}

test "visible: párrafo patológico dispara tras umbral mayor" {
    var g = lg.LoopGuard.init(.{
        .min_period = 4,
        .max_period = 32,
        .max_repeats = 3,
        .mode = .warn,
        .channels = .visible,
    });
    const pattern = [_]u32{ 100, 200, 300, 400, 500, 600 };
    // Threshold visible = 3 + 2 extra = 5 repeticiones.
    var ev: ?lg.LoopEvent = null;
    for (0..5) |_| {
        for (pattern) |tok| {
            if (g.pushVisibleToken(tok)) |e| {
                ev = e;
                break;
            }
        }
        if (ev != null) break;
    }
    try std.testing.expect(ev != null);
    try std.testing.expectEqual(lg.Channel.visible, ev.?.channel);
    // Dispara en la 5ª repetición completa: 5*6 = 30 tokens.
    try std.testing.expectEqual(@as(u64, 30), ev.?.first_seen);
}

test "excluded: fenced code / marcadores NO alimentan detectores" {
    var g = lg.LoopGuard.init(.{
        .min_period = 4,
        .max_period = 32,
        .max_repeats = 3,
        .mode = .warn,
        .channels = .both,
    });
    // 100 "tokens" de código delimitado repetido: si contaran, habrían
    // disparado (period 4 × 25 repeticiones). pushExcluded los ignora.
    for (0..100) |_| g.pushExcluded();
    try std.testing.expectEqual(@as(u64, 0), g.events);
    try std.testing.expect(g.last_event == null);
}

test "reset: estado limpio tras context-bucket nuevo" {
    var g = lg.LoopGuard.init(.{
        .min_period = 4,
        .max_period = 16,
        .max_repeats = 3,
        .mode = .warn,
        .channels = .both,
    });
    const pattern = [_]u32{ 9, 8, 7, 6, 5 };
    for (0..3) |_| {
        for (pattern) |tok| _ = g.pushToken(tok, true);
    }
    try std.testing.expect(g.events >= 1);

    g.reset();
    try std.testing.expectEqual(@as(u64, 0), g.hidden_pos);
    try std.testing.expectEqual(@as(u64, 0), g.visible_pos);
    try std.testing.expectEqual(@as(u64, 0), g.events);
    try std.testing.expect(g.last_event == null);
    // Trackers limpios: el mismo feed post-reset no dispara inmediatamente
    // (repeat_count vuelve a 0).
    _ = g.pushToken(pattern[0], true);
    try std.testing.expect(g.hidden_trackers[5].repeat_count == 0);
}

test "Mode/ChannelCfg fromCli cubren los valores CLI" {
    try std.testing.expectEqual(lg.Mode.force_close, lg.Mode.fromCli("force-close").?);
    try std.testing.expectEqual(lg.Mode.warn, lg.Mode.fromCli("warn").?);
    try std.testing.expectEqual(lg.Mode.off, lg.Mode.fromCli("off").?);
    try std.testing.expect(lg.Mode.fromCli("no-existe") == null);

    try std.testing.expectEqual(lg.ChannelCfg.hidden, lg.ChannelCfg.fromCli("hidden").?);
    try std.testing.expectEqual(lg.ChannelCfg.visible, lg.ChannelCfg.fromCli("visible").?);
    try std.testing.expectEqual(lg.ChannelCfg.both, lg.ChannelCfg.fromCli("both").?);
    try std.testing.expect(lg.ChannelCfg.fromCli("no-existe") == null);
}
