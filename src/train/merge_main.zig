//! CLI del merge tool RLT: zig build rlt-merge -- base.gguf sidecar.gguf out.gguf
const std = @import("std");
const merge_tool = @import("merge_tool");

pub fn main(init: std.process.Init) !void {
    var gpa_state = std.heap.DebugAllocator(.{}){};
    // A5 (COORD): ZIG_AI_NO_LEAK_REPORT=1 silencia leak checks en gates
    // (ReleaseFast). Default: .deinit() reporta leaks; flag: deinitWithoutLeakChecks().
    if (std.c.getenv("ZIG_AI_NO_LEAK_REPORT") != null) {
        defer gpa_state.deinitWithoutLeakChecks();
    } else {
        defer _ = gpa_state.deinit();
    }
    const gpa = gpa_state.allocator();
    const io = init.io;

    const args: std.process.Args = init.minimal.args;
    var args_it = std.process.Args.Iterator.initAllocator(args, gpa) catch return error.OutOfMemory;
    defer args_it.deinit();
    _ = args_it.next(); // argv[0]

    const base_path = args_it.next() orelse {
        std.debug.print(
            \\Uso: rlt-merge <base.gguf> <rlt_sidecar.gguf> <out.gguf>
            \\
            \\Inyecta los pesos RLT (f32) del sidecar en el GGUF base a nivel
            \\de bytes — sin dequantizar los tensores del base. El resultado
            \\activa el merge recurrente on-load (rlt.feedback_alpha > 0).
            \\
        , .{});
        return error.MissingArgs;
    };
    const sidecar_path = args_it.next() orelse return error.MissingArgs;
    const out_path = args_it.next() orelse return error.MissingArgs;

    _ = try merge_tool.mergeRltIntoBase(io, gpa, base_path, sidecar_path, out_path);
}
