//! 11.2 p3 + 11.4: medición COLD de neuronas FFN.
//! Cuenta neuronas "frías" (activación == 0) vs total por capa.
//! Activación: env ZIG_AI_MEASURE_COLD=1.
const std = @import("std");
const debugz = @import("debug");

pub const ColdStats = struct {
    total_neurons: usize = 0,
    cold_neurons: usize = 0,
    layers_measured: usize = 0,

    pub fn record(self: *ColdStats, activated: []const f16) void {
        for (activated) |v| {
            self.total_neurons += 1;
            if (@as(f32, @floatCast(v)) == 0.0) self.cold_neurons += 1;
        }
        self.layers_measured += 1;
    }

    pub fn ratio(self: ColdStats) f32 {
        if (self.total_neurons == 0) return 0.0;
        return @as(f32, @floatFromInt(self.cold_neurons)) / @as(f32, @floatFromInt(self.total_neurons));
    }

    pub fn report(self: ColdStats) void {
        debugz.dbg.print("[moe_cold] layers={d} total={d} cold={d} ratio={d:.1}%\n", .{
            self.layers_measured, self.total_neurons, self.cold_neurons, self.ratio() * 100.0
        });
    }
};

var global_stats: ColdStats = .{};

pub fn coldCallback(_layer_idx: usize, activated: []const f16) void {
    global_stats.record(activated);
}

pub fn reportColdStats() void {
    global_stats.report();
}

pub fn resetColdStats() void {
    global_stats = .{};
}
