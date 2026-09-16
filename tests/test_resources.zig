//! Tests del presupuesto de recursos central (coordinador, post-freeze
//! 2026-09-13). El módulo raíz lleva sus propios tests unitarios de
//! sanidad (topología/loadavg/mem); aquí los de INTEGRACIÓN:
//! - re-exports de cpu_executor (API histórica intacta para lane-e/tests)
//! - budget: knob explícito honrado, auto nunca 0, cap tests ≤4
//! - determinismo: budget estable entre llamadas sin cambio de load
const std = @import("std");
const resources = @import("resources");
const exec_mod = @import("moe_cpu_executor");

test "re-export cpu_executor: misma topología que resources" {
    const a = std.testing.allocator;
    const r1 = try resources.physicalCoreCpus(a);
    defer a.free(r1);
    const r2 = try exec_mod.physicalCoreCpus(a);
    defer a.free(r2);
    try std.testing.expectEqual(r1.len, r2.len);
    for (r1, r2) |x, y| try std.testing.expectEqual(x, y);
}

test "re-export cpu_executor: loadavg idéntico" {
    const a = resources.ambientLoadAvg1m();
    const b = exec_mod.ambientLoadAvg1m();
    try std.testing.expectEqual(a, b);
}

test "budget: knob ZIG_AI_CPU_WORKERS explícito se honra" {
    // El knob se lee por getenv en cada llamada: inyectar env en un test
    // es frágil (putenv no remueve), así que validamos la FUNCIONALIDAD
    // con el estado actual del proceso:
    const a = std.testing.allocator;
    const budget = resources.computeThreadBudget(a, "test");
    try std.testing.expect(budget >= 1);
    const phys = try resources.physicalCoreCpus(a);
    defer a.free(phys);
    // Sin knob: en tests el cap es ≤4 SIEMPRE (paralelismo de suite).
    const knob = resources.cpuWorkersEnv();
    if (knob == null or knob.? == 0) {
        try std.testing.expect(budget <= @min(phys.len, 4));
    } else {
        try std.testing.expectEqual(knob.?, budget);
    }
}

test "budget: determinista en llamadas consecutivas (load estable)" {
    const a = std.testing.allocator;
    const b1 = resources.computeThreadBudget(a, "test");
    const b2 = resources.computeThreadBudget(a, "test");
    // Sin cambio de load entre dos lecturas adyacentes el budget debe
    // ser idéntico (misma afinidad, mismo knob).
    try std.testing.expectEqual(b1, b2);
}

test "resolveThreadsAndAffinity: N explícito sin duplicar CPUs" {
    const a = std.testing.allocator;
    const ids = try resources.resolveThreadsAndAffinity(a, 3);
    defer a.free(ids);
    try std.testing.expectEqual(@as(usize, 3), ids.len);
    const logical = std.Thread.getCpuCount() catch 1;
    if (logical >= 3) {
        for (0..ids.len) |i| {
            for (0..i) |j| try std.testing.expect(ids[i] != ids[j]);
        }
    }
}

test "logCpuBudget no crashea (breadcrumb gated por DEBUG_LEVEL)" {
    const a = std.testing.allocator;
    resources.logCpuBudgetForced("test_runner", resources.computeThreadBudget(a, "test_runner"), a);
}

test "testRunnerBudget respeta cap duro <=4 y >=1" {
    const a = std.testing.allocator;
    const budget = resources.testRunnerBudget(a);
    try std.testing.expect(budget >= 1);
    try std.testing.expect(budget <= 4);
}
