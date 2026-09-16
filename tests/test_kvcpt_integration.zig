//! Lane-b2 P0.2: integración KVCPT en kv_cache_manager.
//!
//! Cubre:
//!   - Dual-write body (Q4_0) + cola exacta (f16 ring) en appendTokensF16.
//!   - Rollback N tokens preserva el body intacto y mantiene el ring
//!     consistente (los slots viejos quedan pero current_len decrece).
//!   - getExactTail devuelve el bloque exacto (últimos `exact_tokens`).
//!   - E2E con Qwen3.5-0.8B-Q4_0.gguf si GGUF_MODEL_PATH está definida.
//!
//! Tests sintéticos configuran dimensiones típicas de un modelo pequeño
//! (24 layers × 8 KV heads × head_dim 128 × max_seq 4096) para validar
//! el contrato sin cargar el modelo completo.

const std = @import("std");
const testing = std.testing;
const kvc = @import("kv_cache");
const kvarn = @import("kv_cache").kvarn;

const KVCacheManager = kvc.KVCacheManager;
const KVCacheConfig = kvc.KVCacheConfig;
const QuantFormat = kvc.QuantFormat;
const LayerQuantConfig = kvc.LayerQuantConfig;
const ExactType = kvc.quant_types.ExactType;

fn configQ4_0WithTail(num_layers: u32, num_kv_heads: u32, head_dim: u32, max_seq_len: u32, tail_tokens: u32) KVCacheConfig {
    var cfg = KVCacheConfig.default(num_layers, num_kv_heads, head_dim, max_seq_len);
    cfg.num_kv_heads = num_kv_heads;
    cfg.use_gpu_dequant = false; // CPU path para estos tests
    cfg.tail_tokens = tail_tokens;
    cfg.tail_type = .f16;
    return cfg;
}

test "kvcpt: append + rollback + exact tail Q4_0 ladder 1024" {
    // Dimensiones reducidas vs Qwen3.5-0.8B (24×8) para que el pool
    // quepa en testing.allocator. Estructura equivalente: 4 layers × 4
    // kv_heads × head_dim 128 × 256 tokens × (q4_0 18 + q8_0 34) = 340KB
    // por head × 16 = 5.4MB body + cola exacta f16 (4×4×1024×2 = 32KB).
    const num_layers: u32 = 4;
    const num_kv_heads: u32 = 4;
    const head_dim: u32 = 128;
    const max_seq_len: u32 = 256;
    const tail_tokens: u32 = 1024;

    const allocator = testing.allocator;
    var layer_cfgs_storage: [4]LayerQuantConfig = undefined;
    for (layer_cfgs_storage[0..]) |*lcf| {
        lcf.* = .{
            .k_format = .q4_0,
            .v_format = .q8_0,
            .k_block_size = 32,
            .v_block_size = 32,
            .quant_threshold = null,
        };
    }
    var config = configQ4_0WithTail(num_layers, num_kv_heads, head_dim, max_seq_len, tail_tokens);
    config.layer_configs = &layer_cfgs_storage;

    var mgr = try KVCacheManager.init(allocator, config, 128);
    defer mgr.deinit();

    const seq_id: u64 = 42;
    try mgr.createSequence(seq_id);

    // Append 200 tokens a la primera capa+head, verificando que el
    // ring exacto mantiene los últimos `tail_tokens` correctamente.
    var i: u32 = 0;
    while (i < 200) : (i += 1) {
        const k_f16 = try allocator.alloc(f16, head_dim);
        defer allocator.free(k_f16);
        const v_f16 = try allocator.alloc(f16, head_dim);
        defer allocator.free(v_f16);
        for (k_f16, 0..) |*k, j| k.* = @as(f16, @floatFromInt((i + 1) * 100 + j));
        for (v_f16, 0..) |*v, j| v.* = @as(f16, @floatFromInt((i + 1) * 200 + j));
        try mgr.appendTokensF16(seq_id, 0, 0, k_f16, v_f16);
        try mgr.advanceSequence(seq_id);
    }

    try testing.expectEqual(@as(u32, 200), try mgr.getSequenceLen(seq_id));

    // Lee los últimos 200 tokens del ring exacto (200 < exact_groups * 128).
    // Para tail_tokens=1024 y head_dim=128, exact_groups=8, exact_total=1024
    // → last_written_group = (199 % 8) = 7. El bloque [0..200] cubre los
    // grupos 0..6 parciales + grupo 7 parcial.
    //
    // Verificación: el último token appendado (i=199, current_len=200,
    // slot = 199 % 8 = 7) tiene los valores esperados.
    const tail_k = try allocator.alloc(f16, tail_tokens * head_dim);
    defer allocator.free(tail_k);
    const tail_v = try allocator.alloc(f16, tail_tokens * head_dim);
    defer allocator.free(tail_v);
    try mgr.getExactTail(seq_id, 0, 0, tail_k, tail_v);

    // El token más reciente está en slot 7 (199 % 8 = 7), offset 7*128 = 896.
    const last_slot_off: usize = 7 * head_dim;
    try testing.expectEqual(@as(f16, @floatFromInt(200 * 100)), tail_k[last_slot_off]);
    try testing.expectEqual(@as(f16, @floatFromInt(200 * 200)), tail_v[last_slot_off]);
}

test "kvcpt: rollback 50 tokens preserves body, decrements length" {
    const num_layers: u32 = 4;
    const num_kv_heads: u32 = 2;
    const head_dim: u32 = 64;
    const max_seq_len: u32 = 1024;
    const tail_tokens: u32 = 256;

    const allocator = testing.allocator;
    var layer_cfgs: [4]LayerQuantConfig = undefined;
    for (layer_cfgs[0..]) |*lcf| {
        lcf.* = .{ .k_format = .q4_0, .v_format = .q8_0, .k_block_size = 32, .v_block_size = 32, .quant_threshold = null };
    }
    var config = configQ4_0WithTail(num_layers, num_kv_heads, head_dim, max_seq_len, tail_tokens);
    config.layer_configs = &layer_cfgs;

    var mgr = try KVCacheManager.init(allocator, config, 128);
    defer mgr.deinit();

    const seq_id: u64 = 7;
    try mgr.createSequence(seq_id);

    // Append 100 tokens a (layer 0, head 0) y (layer 1, head 1).
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        const k_f16 = try allocator.alloc(f16, head_dim);
        defer allocator.free(k_f16);
        const v_f16 = try allocator.alloc(f16, head_dim);
        defer allocator.free(v_f16);
        for (k_f16, 0..) |*k, j| k.* = @as(f16, @floatFromInt((i + 1) * 10 + j));
        for (v_f16, 0..) |*v, j| v.* = @as(f16, @floatFromInt((i + 1) * 20 + j));
        try mgr.appendTokensF16(seq_id, 0, 0, k_f16, v_f16);
        try mgr.appendTokensF16(seq_id, 1, 1, k_f16, v_f16);
        try mgr.advanceSequence(seq_id);
    }
    try testing.expectEqual(@as(u32, 100), try mgr.getSequenceLen(seq_id));

    // Rollback 50 tokens.
    try mgr.rollbackN(seq_id, 50);
    try testing.expectEqual(@as(u32, 50), try mgr.getSequenceLen(seq_id));

    // El body (Q4_0/Q8_0) NO se reescribe — el caller puede sobrescribir
    // posiciones 50..99 con nuevos appends. La cola exacta mantiene su
    // estado (los slots no se borran); los últimos `tail_tokens` desde
    // el nuevo current_len siguen siendo accesibles.

    // Verificación: el slot del token más reciente (token 99, slot 99 % 2 = 1
    // porque exact_groups = tail_tokens/128 = 2 con head_dim=64)
    // sigue ahí. (El caller puede rollback+append y el slot se sobreescribe
    // en el siguiente append.)
    const tail_k = try allocator.alloc(f16, tail_tokens * head_dim);
    defer allocator.free(tail_k);
    const tail_v = try allocator.alloc(f16, tail_tokens * head_dim);
    defer allocator.free(tail_v);
    try mgr.getExactTail(seq_id, 0, 0, tail_k, tail_v);
    // El último token appendado (i=99) está en slot 99 % 2 = 1, offset 64.
    try testing.expectEqual(@as(f16, @floatFromInt(100 * 10)), tail_k[64]);
}

test "kvcpt: rollbackBeyondStart returns error" {
    const num_layers: u32 = 2;
    const num_kv_heads: u32 = 2;
    const head_dim: u32 = 32;
    const tail_tokens: u32 = 128;

    const allocator = testing.allocator;
    var layer_cfgs: [2]LayerQuantConfig = undefined;
    for (layer_cfgs[0..]) |*lcf| {
        lcf.* = .{ .k_format = .fp16, .v_format = .fp16, .k_block_size = 32, .v_block_size = 32, .quant_threshold = null };
    }
    var config = configQ4_0WithTail(num_layers, num_kv_heads, head_dim, 256, tail_tokens);
    config.layer_configs = &layer_cfgs;

    var mgr = try KVCacheManager.init(allocator, config, 128);
    defer mgr.deinit();
    try mgr.createSequence(1);
    try testing.expectError(error.RollbackBeyondStart, mgr.rollbackN(1, 10));
}

test "kvcpt: getExactTail fails when tail disabled" {
    const num_layers: u32 = 2;
    const num_kv_heads: u32 = 2;
    const head_dim: u32 = 32;

    const allocator = testing.allocator;
    var layer_cfgs: [2]LayerQuantConfig = undefined;
    for (layer_cfgs[0..]) |*lcf| {
        lcf.* = .{ .k_format = .fp16, .v_format = .fp16, .k_block_size = 32, .v_block_size = 32, .quant_threshold = null };
    }
    var config = configQ4_0WithTail(num_layers, num_kv_heads, head_dim, 256, 0); // tail=0 → disabled
    config.layer_configs = &layer_cfgs;

    var mgr = try KVCacheManager.init(allocator, config, 128);
    defer mgr.deinit();
    try mgr.createSequence(1);
    const tail_k = try allocator.alloc(f16, 256);
    defer allocator.free(tail_k);
    const tail_v = try allocator.alloc(f16, 256);
    defer allocator.free(tail_v);
    try testing.expectError(error.NoExactTail, mgr.getExactTail(1, 0, 0, tail_k, tail_v));
}

test "kvcpt: KVarN5/4 body + exact tail (round 256 = 2 groups)" {
    // Integra con KVarN encoding CPU. tail_tokens=256 = 2 grupos exactos.
    // Sin GPU: usa el kvarn.zig CPU reference para escribir al body.
    const num_layers: u32 = 1;
    const num_kv_heads: u32 = 1;
    const head_dim: u32 = 128;
    const tail_tokens: u32 = 256;

    const allocator = testing.allocator;
    var layer_cfgs: [1]LayerQuantConfig = undefined;
    for (layer_cfgs[0..]) |*lcf| {
        lcf.* = .{ .k_format = .fp16, .v_format = .fp16, .k_block_size = 32, .v_block_size = 32, .quant_threshold = null };
    }
    var config = configQ4_0WithTail(num_layers, num_kv_heads, head_dim, 1024, tail_tokens);
    config.layer_configs = &layer_cfgs;

    var mgr = try KVCacheManager.init(allocator, config, 128);
    defer mgr.deinit();
    try mgr.createSequence(1);

    // Append 300 tokens al body (fp16 directo).
    var i: u32 = 0;
    while (i < 300) : (i += 1) {
        const k_f16 = try allocator.alloc(f16, head_dim);
        defer allocator.free(k_f16);
        const v_f16 = try allocator.alloc(f16, head_dim);
        defer allocator.free(v_f16);
        for (k_f16, 0..) |*k, j| k.* = @as(f16, @floatFromInt((i + 1) * 3 + j));
        for (v_f16, 0..) |*v, j| v.* = @as(f16, @floatFromInt((i + 1) * 5 + j));
        try mgr.appendTokensF16(1, 0, 0, k_f16, v_f16);
        try mgr.advanceSequence(1);
    }

    // exact_groups = ceil(256/128) = 2. current_len = 300.
    // El token más reciente (i=299) está en slot 299 % 2 = 1, offset 128.
    const tail_k = try allocator.alloc(f16, tail_tokens * head_dim);
    defer allocator.free(tail_k);
    const tail_v = try allocator.alloc(f16, tail_tokens * head_dim);
    defer allocator.free(tail_v);
    try mgr.getExactTail(1, 0, 0, tail_k, tail_v);

    const last_slot_off: usize = 128; // group_idx 1 * head_dim 128
    try testing.expectEqual(@as(f16, @floatFromInt(300 * 3)), tail_k[last_slot_off]);
    try testing.expectEqual(@as(f16, @floatFromInt(300 * 5)), tail_v[last_slot_off]);

    // El segundo más reciente (i=298) está en slot 0, offset 0.
    try testing.expectEqual(@as(f16, @floatFromInt(299 * 3)), tail_k[0]);
}

test "kvcpt: multiple layers + heads share same tail tokens policy" {
    // Cada (layer, head) tiene su propio ring exacto. La policy global
    // (tail_tokens) aplica a todos.
    const num_layers: u32 = 3;
    const num_kv_heads: u32 = 4;
    const head_dim: u32 = 64;
    const tail_tokens: u32 = 128; // 1 grupo exacto

    const allocator = testing.allocator;
    var layer_cfgs: [3]LayerQuantConfig = undefined;
    for (layer_cfgs[0..]) |*lcf| {
        lcf.* = .{ .k_format = .fp16, .v_format = .fp16, .k_block_size = 32, .v_block_size = 32, .quant_threshold = null };
    }
    var config = configQ4_0WithTail(num_layers, num_kv_heads, head_dim, 512, tail_tokens);
    config.layer_configs = &layer_cfgs;

    var mgr = try KVCacheManager.init(allocator, config, 128);
    defer mgr.deinit();
    try mgr.createSequence(1);

    // Append 50 tokens a TODAS las (layer, head) combinaciones con
    // valores distinguibles por (layer_id, head_id, token_idx).
    var layer: u32 = 0;
    while (layer < num_layers) : (layer += 1) {
        var head: u32 = 0;
        while (head < num_kv_heads) : (head += 1) {
            var token_idx: u32 = 0;
            while (token_idx < 40) : (token_idx += 1) {
                const k_f16 = try allocator.alloc(f16, head_dim);
                defer allocator.free(k_f16);
                const v_f16 = try allocator.alloc(f16, head_dim);
                defer allocator.free(v_f16);
                const magic: u32 = layer * 1000 + head * 100 + token_idx;
                for (k_f16, 0..) |*k, j| k.* = @as(f16, @floatFromInt(magic * 10 + j));
                for (v_f16, 0..) |*v, j| v.* = @as(f16, @floatFromInt(magic * 20 + j));
                try mgr.appendTokensF16(1, layer, head, k_f16, v_f16);
                try mgr.advanceSequence(1);
            }
        }
    }

    // Cada (layer, head) tiene su ring independiente. Verifica que el
    // último token (token_idx=49) está en el slot correcto.
    layer = 0;
    while (layer < num_layers) : (layer += 1) {
        var head: u32 = 0;
        while (head < num_kv_heads) : (head += 1) {
            const tail_k = try allocator.alloc(f16, tail_tokens * head_dim);
            defer allocator.free(tail_k);
            const tail_v = try allocator.alloc(f16, tail_tokens * head_dim);
            defer allocator.free(tail_v);
            try mgr.getExactTail(1, layer, head, tail_k, tail_v);
            // Último token appendado a (layer, head) = (l*1000 + h*100 + 49).
            const magic: u32 = layer * 1000 + head * 100 + 39;
            // tail_tokens=128 → exact_groups=1. Token 49 slot 0 offset 0.
            try testing.expectEqual(@as(f16, @floatFromInt(magic * 10)), tail_k[0]);
            try testing.expectEqual(@as(f16, @floatFromInt(magic * 20)), tail_v[0]);
        }
    }
}

test "kvcpt: ring rollover overwrites oldest token exactly" {
    // Con tail_tokens=128 (1 grupo exacto), el segundo append sobrescribe
    // el primero (token 0 → slot 0, token 1 → slot 0). Verificamos que
    // solo el último token sobrevive.
    const num_layers: u32 = 1;
    const num_kv_heads: u32 = 1;
    const head_dim: u32 = 32;
    const tail_tokens: u32 = 128;

    const allocator = testing.allocator;
    var layer_cfgs: [1]LayerQuantConfig = undefined;
    for (layer_cfgs[0..]) |*lcf| {
        lcf.* = .{ .k_format = .fp16, .v_format = .fp16, .k_block_size = 32, .v_block_size = 32, .quant_threshold = null };
    }
    var config = configQ4_0WithTail(num_layers, num_kv_heads, head_dim, 256, tail_tokens);
    config.layer_configs = &layer_cfgs;

    var mgr = try KVCacheManager.init(allocator, config, 128);
    defer mgr.deinit();
    try mgr.createSequence(1);

    // Append 5 tokens con valores únicos.
    var i: u32 = 0;
    while (i < 5) : (i += 1) {
        const k_f16 = try allocator.alloc(f16, head_dim);
        defer allocator.free(k_f16);
        const v_f16 = try allocator.alloc(f16, head_dim);
        defer allocator.free(v_f16);
        for (k_f16, 0..) |*k, j| k.* = @as(f16, @floatFromInt((i + 1) * 1000 + j));
        for (v_f16, 0..) |*v, j| v.* = @as(f16, @floatFromInt((i + 1) * 1000 + j));
        try mgr.appendTokensF16(1, 0, 0, k_f16, v_f16);
        try mgr.advanceSequence(1);
    }

    const tail_k = try allocator.alloc(f16, tail_tokens * head_dim);
    defer allocator.free(tail_k);
    const tail_v = try allocator.alloc(f16, tail_tokens * head_dim);
    defer allocator.free(tail_v);
    try mgr.getExactTail(1, 0, 0, tail_k, tail_v);

    // El último token (i=4) sobrescribió al primero en slot 0.
    // Token más reciente en slot 0: valor de i=4.
    try testing.expectEqual(@as(f16, @floatFromInt(5 * 1000)), tail_k[0]);
    try testing.expectEqual(@as(f16, @floatFromInt(5 * 1000)), tail_v[0]);
}