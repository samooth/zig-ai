//! test_expert_bundle — 11.2: integridad del formato bundle contiguo.
//!
//! Requiere BUNDLE_TEST_GGUF (modelo MoE real). Flujo:
//!   1. abrir GGUF fuente + bundle construido (env BUNDLE_TEST_PATH)
//!   2. para capas/kinds/expertos muestreados: bytes útiles del bundle
//!      (slot completo hasta expertBytes) == expertSlice del GGUF
//!   3. readExpertInto (pread a buffer) == expertSlice (mmap)
//!
//! Sin GPU (puro disco). Skip si falta el env.

const std = @import("std");
const gguf = @import("gguf");
const gguf_moe = @import("gguf_moe");
const expert_bundle = @import("expert_bundle");

test "bundle: contenido por-experto idéntico al GGUF" {
    const gpa = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();

    const gguf_path: []const u8 = std.mem.span(std.c.getenv("BUNDLE_TEST_GGUF") orelse return error.SkipZigTest);
    const bundle_path_c = std.c.getenv("BUNDLE_TEST_PATH") orelse return error.SkipZigTest;

    var model = try gguf.GgufFile.fromFileMmap(io, gpa, gguf_path);
    defer model.deinit();

    var src = try expert_bundle.BundleSource.open(gpa, std.mem.span(bundle_path_c));
    defer src.deinit();
    try src.ensureMmap(io);

    try expert_bundle.validateAgainstGguf(&src, &model, gpa);

    // Muestrear: primera + última capa MoE, 3 expertos (0, E/2, E-1), 3 kinds.
    var first_moe: ?u32 = null;
    var last_moe: u32 = 0;
    var n_experts: u32 = 0;
    var il: u32 = 0;
    while (il < 1024) : (il += 1) {
        if (!gguf_moe.isMoeLayer(&model, il)) continue;
        if (first_moe == null) first_moe = il;
        last_moe = il;
        const spec = try gguf_moe.layerSpec(&model, il);
        n_experts = @intCast(spec.n_expert);
    }
    const fl = first_moe orelse return error.SkipZigTest; // sin capas MoE

    const kinds = [_]gguf_moe.BankKind{ .gate, .up, .down };
    var checked: usize = 0;
    for ([_]u32{ fl, last_moe }) |layer| {
        const spec = try gguf_moe.layerSpec(&model, layer);
        for (kinds, 0..) |kind, k| {
            const bank = switch (kind) {
                .gate => spec.gate,
                .up => spec.up,
                .down => spec.down,
            };
            const useful = bank.expertBytes();
            const experts = [_]u32{ 0, n_experts / 2, n_experts - 1 };
            for (experts) |e| {
                const from_bundle = try src.expertSlice(layer, kind, e);
                if (!src.slicesAreBorrowed()) gpa.free(from_bundle);
                const from_gguf = bank.expertSlice(e);
                try std.testing.expectEqual(useful, from_gguf.len);
                try std.testing.expect(from_bundle.len >= useful);
                const expected = if (bank.external_scale) |_| blk: {
                    const composed = try bank.composeCanonical(gpa);
                    defer gpa.free(composed);
                    break :blk composed[e * bank.blockBytes() ..][0..useful];
                } else from_gguf;
                try std.testing.expectEqualSlices(u8, expected, from_bundle[0..useful]);
                checked += 1;
            }
            _ = k;
        }
    }
    std.debug.print("bundle: {d} slots muestreados idénticos GGUF↔bundle (L {d}..{d}, E={d})\n", .{ checked, fl, last_moe, n_experts });
}
