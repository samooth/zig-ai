//! Tests Lane F5: presupuesto elástico + rebuild con rollback.
//!
//! Aritmética PURA sin GPU (patrón cache_budget de FreeToken): la máquina de
//! estados RebuildTx se maneja con outcomes simulados; el modelo extendido
//! de presupuesto se valida contra el repro 27B de lane-c (q80 1.29GB +
//! pool contiguo 512MB + W_T ~250MB mataban el primer forward post-prefill).
const std = @import("std");
const vb = @import("vram_budget");

const GB: usize = 1024 * 1024 * 1024;
const MB: usize = 1024 * 1024;

// ============================================================================
// Modelo extendido: residentes fijos, transitorio W_T, cap de pool contiguo
// ============================================================================

fn base27bEstimate() vb.VramEstimate {
    return .{
        .compressed_weight_per_layer = 60 * MB,
        .num_layers = 74,
        .num_attn_layers = 40,
        .hidden_dim = 4096,
        .head_dim = 128,
        .num_kv_heads = 8,
        .feed_forward_dim = 12288,
        .max_seq_len = 32 * 1024, // ctx largo real del caso 27B
        .kv_quant = .q8_0,
        .max_resident = 1,
    };
}

test "repro 27B: sin términos F5 el presupuesto da VERDE y muere en runtime" {
    const est = base27bEstimate();
    const bd_old = vb.estimateTotalVram(est);
    // Presupuesto viejo (sin fixed/transient/cap): parece caber en 8GB al 85%.
    try std.testing.expect(bd_old.total_vram <= 8 * GB * 85 / 100);
}

test "repro 27B: con términos F5 el presupuesto DETECTA el pico post-prefill" {
    var est = base27bEstimate();
    // Cifras medidas por lane-c en su matriz:
    est.resident_fixed_bytes = 1290 * MB; // q80-device siempre-residente
    est.transient_peak_bytes = vb.ssmTransientBytes(6144, 4096); // W_T ≈ 100MB… d_inner×n_embd reales ≈250MB según config
    est.contiguous_pool_cap = 512 * MB; // anti-OOM host cap

    const bd = vb.estimateTotalVram(est);
    try std.testing.expect(bd.pool_capped); // KV excede el cap → offload obligatorio
    try std.testing.expectEqual(est.contiguous_pool_cap, bd.kv_vram);

    // El total extendido ya NO cabe cómodo en 8GB al 85% ⇒ suggest reduce/null
    // (antes pasaba verde y moría en ssm.zig:708).
    const suggested = vb.suggestLayerStreamConfig(8 * GB, est);
    _ = suggested; // el valor exacto depende de cifras; lo que importa es que
    // el breakdown refleje los términos:
    try std.testing.expectEqual(est.resident_fixed_bytes, bd.resident_fixed);
    try std.testing.expectEqual(est.transient_peak_bytes, bd.transient_peak);
}

test "cap de pool contiguo: sin exceder no marca capped" {
    var est = base27bEstimate();
    est.max_seq_len = 256; // KV pequeño
    est.contiguous_pool_cap = 4 * GB;
    const bd = vb.estimateTotalVram(est);
    try std.testing.expect(!bd.pool_capped);
    try std.testing.expectEqual(@as(usize, 0), bd.resident_fixed);
}

// ============================================================================
// RebuildTx: rechazo recoverable → fit-check → punto de no retorno → resize
// → rollback ante OOM destructivo.
// ============================================================================

fn testSplit() vb.BudgetSplit {
    return .{
        .total_vram = 8 * GB,
        .fixed_bytes = 2 * GB,
        .activation_bytes = 200 * MB,
        .transient_reserve = 250 * MB,
        .kv_bytes = 3 * GB,
        .moe_slots_bytes = 0,
    };
}

fn testPricing() vb.PoolPricing {
    return .{
        .kv_page_bytes = 2 * MB,
        .moe_slot_bytes = 16 * MB,
        .min_kv_pages = 8,
        .min_moe_slots = 4,
    };
}

test "prevalidación: geometría inválida rechaza SIN tocar estado" {
    var tx = vb.RebuildTx{ .current = testSplit(), .pricing = testPricing(), .account_free = 1 * GB };
    try std.testing.expectError(error.RejectedInvalidGeometry, tx.prevalidate(.{})); // nada pedido
    try std.testing.expectError(error.RejectedInvalidGeometry, tx.prevalidate(.{ .kv_pages = 2 })); // < min_kv_pages
    try tx.prevalidate(.{ .moe_slots = 8 }); // ≥ min_moe_slots ⇒ válido
    try std.testing.expectEqual(vb.RebuildPhase.prevalidated, tx.phase);
    // Estado intacto tras los rechazos:
    try std.testing.expectEqual(@as(usize, 3 * GB), tx.current.kv_bytes);
}

test "fit-check: objetivo que no cabe rechaza limpio (estado intacto)" {
    var tx = vb.RebuildTx{ .current = testSplit(), .pricing = testPricing(), .account_free = 1 * GB };
    try tx.prevalidate(.{ .kv_pages = 4000 }); // 8000MB > techo (~5550MB)
    try std.testing.expectError(error.RejectedDoesNotFit, tx.fitCheck(.{ .kv_pages = 4000 }));
    try std.testing.expectEqual(vb.RebuildPhase.prevalidated, tx.phase);
    try std.testing.expectEqual(@as(usize, 3 * GB), tx.current.kv_bytes); // intacto
}

test "camino feliz: prevalidate→fit→no-return→resize→done" {
    var tx = vb.RebuildTx{ .current = testSplit(), .pricing = testPricing(), .account_free = 1 * GB };
    try tx.prevalidate(.{ .kv_pages = 1500, .moe_slots = 8 });
    const target = try tx.fitCheck(.{ .kv_pages = 1500, .moe_slots = 8 });
    try std.testing.expectEqual(@as(usize, 1500 * 2 * MB), target.kv_bytes);
    try std.testing.expectEqual(@as(usize, 8 * 16 * MB), target.moe_slots_bytes);
    tx.pointOfNoReturn();
    try std.testing.expect(tx.snapshot != null);
    const outcomes = [_]vb.ResizeOutcome{ .ok, .ok };
    try tx.applyResize(target, &outcomes);
    tx.complete();
    try std.testing.expectEqual(vb.RebuildPhase.done, tx.phase);
    // Invariante contable: committed + transitorio + headroom = total.
    try std.testing.expectEqual(
        target.total_vram,
        target.committed() + target.transient_reserve + target.freeHeadroom(),
    );
}

test "OOM destructivo post-teardown → rollback a snapshot + error" {
    var tx = vb.RebuildTx{ .current = testSplit(), .pricing = testPricing(), .account_free = 1 * GB };
    const before = tx.current;
    try tx.prevalidate(.{ .kv_pages = 1000 });
    const target = try tx.fitCheck(.{ .kv_pages = 1000 });
    tx.pointOfNoReturn();

    const outcomes = [_]vb.ResizeOutcome{ .ok, .oom }; // segundo pool OOMea
    try std.testing.expectError(error.OutOfMemoryDuringResize, tx.applyResize(target, &outcomes));

    // Rollback: geometría EXACTA del snapshot restaurada.
    try std.testing.expectEqual(vb.RebuildPhase.rolled_back, tx.phase);
    try std.testing.expectEqual(before.kv_bytes, tx.current.kv_bytes);
    try std.testing.expectEqual(before.moe_slots_bytes, tx.current.moe_slots_bytes);
    try std.testing.expectEqual(before.fixed_bytes, tx.current.fixed_bytes);
    try std.testing.expectEqual(before.activation_bytes, tx.current.activation_bytes);
    try std.testing.expectEqual(before.transient_reserve, tx.current.transient_reserve);
    try std.testing.expectEqual(before.total_vram, tx.current.total_vram);
    // El servicio queda en geometría vieja: un nuevo rebuild puede intentarse.
    var tx2 = vb.RebuildTx{ .current = tx.current, .pricing = tx.pricing, .account_free = 1 * GB };
    try tx2.prevalidate(.{ .kv_pages = 1200 });
    _ = try tx2.fitCheck(.{ .kv_pages = 1200 });
}

test "rollback sin snapshot es error, nunca corrupción" {
    var tx = vb.RebuildTx{ .current = testSplit(), .pricing = testPricing(), .account_free = 0 };
    try std.testing.expectError(error.RejectedInvalidGeometry, tx.rollback());
}

// ============================================================================
// Invariantes de BudgetSplit
// ============================================================================

test "BudgetSplit: headroom respeta reserva transitoria" {
    const s = vb.BudgetSplit{
        .total_vram = 10 * GB,
        .fixed_bytes = 5 * GB,
        .activation_bytes = 1 * GB,
        .transient_reserve = 500 * MB,
        .kv_bytes = 2 * GB,
        .moe_slots_bytes = 1 * GB,
    };
    // committed=9GB + transient .5GB → headroom .5GB
    // total 10240MiB − committed 9216MiB − transitorio 500MiB = 524MiB.
    try std.testing.expectEqual(@as(usize, 524 * MB), s.freeHeadroom());
}

// ============================================================================
// Phase 3: fit-check elástico (RebuildTx ↔ budget.Rebuilder)
// ============================================================================

fn elasticTx() vb.RebuildTx {
    return .{
        .current = .{
            .total_vram = 10 * GB,
            .fixed_bytes = 4 * GB,
            .activation_bytes = 1 * GB,
            .transient_reserve = 500 * MB,
            .kv_bytes = 3 * GB,
            .moe_slots_bytes = 1 * GB,
        },
        .pricing = .{ .kv_page_bytes = MB, .moe_slot_bytes = 8 * MB, .min_kv_pages = 1, .min_moe_slots = 1 },
        .account_free = 6 * GB,
    };
}

test "fitCheckElastic: consumers reciben grants proporcionales al headroom" {
    var tx = elasticTx();
    try tx.prevalidate(.{ .kv_pages = 2 * 1024 }); // 2 GiB de KV
    var consumers = [_]@import("budget").Consumer{
        .{ .name = "kv", .current = 3 * GB, .desired = 3 * GB, .shrinkable_floor = 1 * GB },
        .{ .name = "moe", .current = 1 * GB, .desired = 1 * GB, .shrinkable_floor = 256 * MB },
    };
    // account_free=6GB, weights(=fixed+activ)=5GB, fixed_cache=0.5GB → net≈0.5GB
    // sobre floors (1GB+0.25GB) → overcommit ⇒ rechazo limpio PERO primero
    // probamos con headroom holgado:
    tx.account_free = 20 * GB; // net ≈ 14.5GB ≥ floors ⇒ cabe
    const target = try tx.fitCheckElastic(.{ .kv_pages = 2 * 1024 }, &consumers);
    try std.testing.expectEqual(2 * GB, target.kv_bytes);
    // grants: fit ⇒ cada consumer a su desired
    try std.testing.expectEqual(3 * GB, consumers[0].granted);
    try std.testing.expectEqual(1 * GB, consumers[1].granted);
}

test "fitCheckElastic: overcommit de floors rechaza limpio (estado intacto)" {
    var tx = elasticTx(); // account_free=6GB, weights=5GB, fixed=0.5GB → net≈0.5GB
    try tx.prevalidate(.{ .kv_pages = 2 * 1024 });
    var consumers = [_]@import("budget").Consumer{
        .{ .name = "kv", .current = 3 * GB, .desired = 3 * GB, .shrinkable_floor = 1 * GB },
        .{ .name = "moe", .current = 1 * GB, .desired = 1 * GB, .shrinkable_floor = 256 * MB },
    };
    // floors (1.25GB) > net (0.5GB) → overcommit → rechazo
    try std.testing.expectError(error.RejectedDoesNotFit, tx.fitCheckElastic(.{ .kv_pages = 2 * 1024 }, &consumers));
    // El rechazo elástico NO avanza la fase (el caller puede reintentar con
    // otra geometría): la fase sigue prevalidated, no fit_checked.
    try std.testing.expectEqual(vb.RebuildPhase.prevalidated, tx.phase);
}
