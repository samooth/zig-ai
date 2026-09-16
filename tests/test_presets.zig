//! Tests de INI presets (lane-b3 P3.4 — contrato C5).
//! Parse, precedencia CLI>preset>default, y validación coherente.
const std = @import("std");
const presets = @import("presets");

const sample_ini =
    \\# presets/qwen36-27b-kvarn.ini
    \\[model]
    \\ctx = 65536
    \\
    \\[kv]
    \\cache-type-k = kvarn5
    \\cache-type-v = kvarn4
    \\kv-tail-tokens = 1024
    \\kv-tail-type = f16
    \\
    \\[speculative]
    \\spec-type = draft-mtp
    \\dm-controller = profit
    \\dm-profit-baseline-interval = 2048
    \\
    \\[guard]
    \\reasoning-loop-mode = force-close
    \\reasoning-loop-window = 512
    \\reasoning-loop-max-period = 3
    \\reasoning-loop-channel = both
;

test "parse: preset completo del plan qwen36-27b" {
    var p = presets.Preset{};
    try presets.parse(sample_ini, &p);
    try std.testing.expectEqual(@as(?usize, 65536), p.ctx);
    try std.testing.expectEqualStrings("kvarn5", p.cache_type_k.?);
    try std.testing.expectEqualStrings("kvarn4", p.cache_type_v.?);
    try std.testing.expectEqual(@as(?usize, 1024), p.kv_tail_tokens);
    try std.testing.expectEqualStrings("f16", p.kv_tail_type.?);
    try std.testing.expectEqualStrings("draft-mtp", p.spec_type.?);
    try std.testing.expectEqualStrings("profit", p.dm_controller.?);
    try std.testing.expectEqual(@as(?usize, 2048), p.dm_profit_baseline_interval);
    try std.testing.expectEqualStrings("force-close", p.reasoning_loop_mode.?);
    try std.testing.expectEqual(@as(?usize, 512), p.reasoning_loop_window.?);
    try std.testing.expectEqual(@as(?usize, 3), p.reasoning_loop_max_period.?);
    try std.testing.expectEqualStrings("both", p.reasoning_loop_channel.?);
}

test "parse: claves ausentes quedan null (no defaults ocultos)" {
    var p = presets.Preset{};
    try presets.parse("[model]\nctx = 4096\n", &p);
    try std.testing.expectEqual(@as(?usize, 4096), p.ctx);
    try std.testing.expect(p.cache_type_k == null);
    try std.testing.expect(p.spec_type == null);
    try std.testing.expect(p.reasoning_loop_mode == null);
}

test "precedencia: CLI explícito PISA al preset" {
    var p = presets.Preset{};
    try presets.parse(sample_ini, &p);
    // CLI trae ctx=1024 explícito: gana sobre preset 65536.
    try std.testing.expectEqual(@as(usize, 1024), presets.resolveField(usize, 1024, p.ctx, 4096));
    // CLI ausente: preset gana.
    try std.testing.expectEqual(@as(usize, 65536), presets.resolveField(usize, null, p.ctx, 4096));
    // Ni CLI ni preset: default.
    try std.testing.expectEqual(@as(usize, 4096), presets.resolveField(usize, null, null, 4096));
}

test "validación: kv-tail-tokens>0 con K-only (MLA) falla al parsear" {
    var p = presets.Preset{};
    const ini =
        \\[kv]
        \\cache-type-k = kvarn5
        \\kv-tail-tokens = 1024
    ;
    try std.testing.expectError(presets.PresetError.InvalidCombination, presets.parse(ini, &p));
}

test "validación: dm-controller profit sin spec-type falla al parsear" {
    var p = presets.Preset{};
    try std.testing.expectError(
        presets.PresetError.InvalidCombination,
        presets.parse("[speculative]\ndm-controller = profit\n", &p),
    );
    // ...pero dm-controller = off sin spec-type es válido (guard apagado).
    var p2 = presets.Preset{};
    try presets.parse("[speculative]\ndm-controller = off\n", &p2);
    try std.testing.expect(p2.spec_type == null);
}

test "validación: valores de enum inválidos fallan al parsear" {
    var p = presets.Preset{};
    try std.testing.expectError(
        presets.PresetError.InvalidValue,
        presets.parse("[guard]\nreasoning-loop-mode = explode\n", &p),
    );
    var p2 = presets.Preset{};
    try std.testing.expectError(
        presets.PresetError.InvalidValue,
        presets.parse("[guard]\nreasoning-loop-channel = infrared\n", &p2),
    );
    var p3 = presets.Preset{};
    try std.testing.expectError(
        presets.PresetError.InvalidValue,
        presets.parse("[speculative]\nspec-type = warp-drive\n", &p3),
    );
}

test "validación: sección desconocida falla al parsear" {
    var p = presets.Preset{};
    try std.testing.expectError(
        presets.PresetError.UnknownSection,
        presets.parse("[warp]\nspeed = 10\n", &p),
    );
}

test "validación: clave desconocida falla al parsear" {
    var p = presets.Preset{};
    try std.testing.expectError(
        presets.PresetError.UnknownKey,
        presets.parse("[model]\nlr = 0.01\n", &p),
    );
}

test "parse: comentarios y líneas en blanco tolerados" {
    var p = presets.Preset{};
    const ini =
        \\
        \\; comentario estilo-ini
        \\# otro comentario
        \\
        \\[model]
        \\  ctx = 8192
    ;
    try presets.parse(ini, &p);
    try std.testing.expectEqual(@as(?usize, 8192), p.ctx);
}

test "validación: kv-tail-tokens > ctx falla al parsear" {
    var p = presets.Preset{};
    const ini =
        \\[model]
        \\ctx = 1024
        \\[kv]
        \\cache-type-k = q8_0
        \\cache-type-v = q8_0
        \\kv-tail-tokens = 4096
    ;
    try std.testing.expectError(presets.PresetError.InvalidCombination, presets.parse(ini, &p));
}

test "parse: string values sin allocs (referencian el input)" {
    var p = presets.Preset{};
    const ini = "[kv]\ncache-type-k = q4_k\n";
    try presets.parse(ini, &p);
    // El valor apunta DENTRO de ini (copia por el caller si sobrevive).
    const v_addr = @intFromPtr(p.cache_type_k.?.ptr);
    const ini_lo = @intFromPtr(ini.ptr);
    try std.testing.expect(v_addr >= ini_lo);
    try std.testing.expect(v_addr < ini_lo + ini.len);
}
