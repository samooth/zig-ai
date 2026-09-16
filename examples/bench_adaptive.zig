//! Standalone driver para `zig build bench-adaptive`.
//! Carga `src/bench/adaptive_bench.zig` como módulo, corre 3 perfiles
//! canónicos (collapsing / bouncing / stable-high) con un par
//! off-vs-profit por perfil, e imprime la tabla comparativa.
//! No requiere modelo: es el micro-bench CPU determinista que valida
//! la propiedad del controller. Para bench end-to-end con modelo
//! real, usar `zig-ai-engine` con `--spec-dm-controller off|profit`
//! y medir t/s con `ggml-bench`-style harness.
const std = @import("std");
const bench = @import("adaptive_bench");

const Profiles = struct {
    collapsing: bench.Profile,
    bouncing: bench.Profile,
    stable_high: bench.Profile,
};

fn buildProfiles(allocator: std.mem.Allocator, cycles: usize) !Profiles {
    return .{
        .collapsing = try bench.collapsingProfile(allocator, cycles, cycles / 2),
        .bouncing = try bench.bouncingProfile(allocator, cycles),
        .stable_high = try bench.stableHighProfile(allocator, cycles),
    };
}

fn freeProfiles(allocator: std.mem.Allocator, p: Profiles) void {
    allocator.free(p.collapsing.accept);
    allocator.free(p.bouncing.accept);
    allocator.free(p.stable_high.accept);
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.gpa;
    const cycles: usize = 1024;
    const fixed_depth: u32 = 8;
    const cfg = adaptive.Config{ .max_depth = 8, .min_depth = 1, .probe_interval = 16, .promote_ratio = 1.05 };

    var out_buf: [4096]u8 = undefined;
    var out_w = std.Io.File.stdout().writer(io, &out_buf);
    const out = &out_w.interface;

    try out.print("# Adaptive draft-max bench — off vs profit (lane-b3)\n", .{});
    try out.print("# cycles={d} fixed_depth={d} profit.min={d} probe_interval={d}\n", .{
        cycles, fixed_depth, cfg.min_depth, cfg.probe_interval,
    });
    try out.print("# (micro-bench CPU determinista; sin modelo — mide la propiedad del controller)\n\n", .{});
    try out.flush();

    const profiles = try buildProfiles(allocator, cycles);
    defer freeProfiles(allocator, profiles);

    var total_speedup: f64 = 0;
    var n_profiles: usize = 0;

    const labels = [_][]const u8{ "collapsing", "bouncing", "stable_high" };
    inline for (labels) |label| {
        const p = switch (label.len) {
            10 => profiles.collapsing, // "collapsing"
            8 => profiles.bouncing, // "bouncing"
            else => profiles.stable_high, // "stable_high"
        };
        const cmp = bench.runCompare(p, fixed_depth, cfg);
        const s = try bench.writeReport(out, cmp, label);
        try out.print("\n", .{});
        total_speedup += s;
        n_profiles += 1;
    }

    try out.print("# avg speedup: {d:.3}x sobre {d} perfiles\n", .{ total_speedup / @as(f64, @floatFromInt(n_profiles)), n_profiles });
    try out.flush();
}

const adaptive = @import("adaptive_dm");
