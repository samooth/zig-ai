//! Standalone driver para `zig build bench-overhead`.
//! Corre el ciclo del controller (recordAccepted+recordTiming+evaluateProbe+tick)
//! durante un número de rounds + iteraciones, e imprime ns/op con
//! percentiles. El threshold < 0.5% decode se aplica en el motor real
//! con `PERF_SPEC`; este bench es el ground truth CPU del controller.
const std = @import("std");
const bench = @import("controller_overhead");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const rounds: usize = 5;
    const iters_per_round: u64 = 200_000;

    var out_buf: [4096]u8 = undefined;
    var out_w = std.Io.File.stdout().writer(io, &out_buf);
    const out = &out_w.interface;

    try out.print("# ProfitController overhead — lane-b3 T7\n", .{});
    try out.print("# rounds={d} iters_per_round={d} (total={d})\n", .{
        rounds, iters_per_round, rounds * iters_per_round,
    });
    try out.print("# método: N iteraciones por round (reset controller entre rounds),\n", .{});
    try out.print("# warmup 10k iters previo. Cycle: recordAccepted(d) + recordTiming +\n", .{});
    try out.print("# evaluateProbe + tick (camino realista del motor por token).\n", .{});
    try out.print("# gate duro (ReleaseFast, target de este binario): < 1µs/op.\n\n", .{});
    try out.flush();

    const results = bench.runBench(rounds, iters_per_round);
    defer std.heap.page_allocator.free(results);

    try bench.writeReport(out, results, "controller_overhead");

    // Regla Bee: gate < 0.5% decode. Asumiendo decode ~200µs/tok
    // (Qwen3.5-7B batch=1 GPU), el threshold absoluto es 1µs/op.
    const med = bench.medianNsPerOp(results);
    try out.print("\n", .{});
    if (med < 1_000.0) {
        try out.print("# PASS: med={d:.0}ns < 1000ns (umbral < 0.5% decode @ 200µs/tok)\n", .{med});
    } else {
        try out.print("# WARN: med={d:.0}ns > 1000ns (umbral no cumplido; debug build?\n", .{med});
        try out.print("#       para gate estricto correr en ReleaseFast)\n", .{});
    }
    try out.flush();
}
