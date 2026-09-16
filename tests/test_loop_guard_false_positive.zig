//! Tests de falsos positivos del loop guard (lane-b3 P1.2).
//! Prosa legítima con repetición NO dispara: listas numeradas, estribillos,
//! fenced code y marcadores tool-call.
const std = @import("std");
const spec = @import("speculative");
const lg = spec.loop_guard;

/// Generador determinista de "prosa": tokens con deriva (no periódicos).
fn proseToken(i: usize) u32 {
    // LCG con estado que CAMBIA cada llamada: sin período corto.
    var s: u64 = 0x9E3779B97F4A7C15 ^ @as(u64, i);
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return @truncate((s *% 0x2545F4914F6CDD1D) >> 32);
}

test "prosa legítima 4096 tokens: cero eventos" {
    var g = lg.LoopGuard.init(.{ .mode = .warn, .channels = .both });
    for (0..4096) |i| {
        _ = g.pushToken(proseToken(i), true);
        _ = g.pushVisibleToken(proseToken(i * 3 + 1));
    }
    try std.testing.expectEqual(@as(u64, 0), g.events);
}

test "lista numerada (marcador repetido + contenido variable) NO dispara" {
    var g = lg.LoopGuard.init(.{
        .min_period = 8,
        .max_period = 64,
        .max_repeats = 3,
        .mode = .warn,
        .channels = .both,
    });
    // Lista numerada: tokens de marcador IGUALES cada línea pero el
    // CONTENIDO difiere → el hash del bloque completo nunca repite.
    const item_ids = [_]u32{ 10, 11, 12, 13 };
    for (0..200) |line| {
        _ = g.pushToken(item_ids[line % item_ids.len], true);
        for (0..20) |k| {
            _ = g.pushToken(proseToken(line * 100 + k), true);
        }
    }
    try std.testing.expectEqual(@as(u64, 0), g.events);
}

test "estribillo (verso repetido con variación final) NO dispara" {
    var g = lg.LoopGuard.init(.{
        .min_period = 8,
        .max_period = 64,
        .max_repeats = 3,
        .mode = .warn,
        .channels = .both,
    });
    // Estribillo de 30 tokens repetido 10 veces PERO cada repetición lleva
    // 2 tokens de variación al final (número de estrofa) → período real 32,
    // hash distinto por estrofa.
    var refrain: [30]u32 = undefined;
    for (&refrain, 0..) |*t, i| t.* = proseToken(i);
    for (0..10) |stanza| {
        for (refrain) |tok| _ = g.pushToken(tok, true);
        _ = g.pushToken(50 + @as(u32, @intCast(stanza / 10)), true);
        _ = g.pushToken(50 + @as(u32, @intCast(stanza % 10)), true);
    }
    try std.testing.expectEqual(@as(u64, 0), g.events);
}

test "fenced code (bloque de código largo repetido) NO dispara vía exclusiones" {
    var g = lg.LoopGuard.init(.{
        .min_period = 8,
        .max_period = 64,
        .max_repeats = 3,
        .mode = .warn,
        .channels = .both,
    });
    // El pipeline detecta fenced code y deriva a pushExcluded (lección Bee
    // v0.4.4). Simulamos: 3 bloques idénticos de 40 tokens van TODOS por
    // pushExcluded, sin alimentar ningún detector.
    for (0..3) |_| {
        for (0..40) |_| {
            g.pushExcluded();
        }
    }
    try std.testing.expectEqual(@as(u64, 0), g.events);
}

test "marcadores tool-call en el stream NO disparan" {
    var g = lg.LoopGuard.init(.{
        .min_period = 4,
        .max_period = 32,
        .max_repeats = 3,
        .mode = .warn,
        .channels = .both,
    });
    // Marcadores tool-call idénticos repetidos 20 veces: el pipeline los
    // detecta por id y deriva a pushExcluded; el contenido interno varía
    // por llamada. El guard nunca los ve como candidatos de loop.
    for (0..20) |_| {
        g.pushExcluded();
        g.pushExcluded();
    }
    try std.testing.expectEqual(@as(u64, 0), g.events);
}

test "mixto: loop REAL tras prosa legítima sigue detectándose" {
    var g = lg.LoopGuard.init(.{
        .min_period = 8,
        .max_period = 64,
        .max_repeats = 4,
        .mode = .warn,
        .channels = .hidden,
    });
    // 500 tokens de prosa sana.
    for (0..500) |i| _ = g.pushToken(proseToken(i), true);
    try std.testing.expectEqual(@as(u64, 0), g.events);

    // Loop REAL de período 20 × 4 → dispara tras 4 bloques = 80 tokens.
    var pattern: [20]u32 = undefined;
    for (&pattern, 0..) |*t, i| t.* = proseToken(10000 + i);
    var ev: ?lg.LoopEvent = null;
    outer: for (0..4) |_| {
        for (pattern) |tok| {
            if (g.pushToken(tok, true)) |e| {
                ev = e;
                break :outer;
            }
        }
    }
    try std.testing.expect(ev != null);
    try std.testing.expectEqual(@as(u32, 20), ev.?.period);
}
