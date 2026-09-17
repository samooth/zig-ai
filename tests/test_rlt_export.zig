//! Roundtrip test del exportador GGUF de RLT — escribe un sidecar con pesos
//! conocidos y lo re-parsea con el PARSER REAL del engine (gguf.GgufFile).
//!
//! Verifica el contrato completo de loadRltWeights (hybrid_layer.zig):
//!   1. El parser acepta magic/version/counts (header de 24 bytes).
//!   2. rlt.feedback_alpha (float32) y rlt.swa_window (uint32) llegan con el
//!      tipo correcto (MetaValueType.float32=6, uint32=4).
//!   3. Los tensores blk.N.rlt.feedback_{gate,state}.weight existen con dims
//!      GGUF [in, out] correctas.
//!   4. La TRANSPOSICIÓN de loadGgufF32 (GGUF [in,out] → engine [out,in])
//!      devuelve EXACTAMENTE los pesos de entrenamiento w_gate[c*2d + r].
const std = @import("std");
const testing = std.testing;
const export_gguf = @import("export_gguf");
const gguf = @import("gguf");

/// Nombre temporal único (std.crypto.random eliminado en 0.16 — timestamp ns).
fn tmpPath(buf: []u8, prefix: []const u8) ![]const u8 {
    const ts: u64 = @intCast(@max(0, @import("time").wallClockSec()));
    const rand: u32 = @truncate(ts ^ (ts >> 32));
    return std.fmt.bufPrint(buf, "/tmp/rlt_{s}_{d}.gguf", .{ prefix, rand });
}

extern "c" fn unlink(path: [*:0]const u8) c_int;

/// std.posix.unlink eliminado en 0.16 — shim C (patrón test_hybrid_sync).
fn rmTmp(path: []const u8) void {
    var zbuf: [128]u8 = undefined;
    if (path.len >= zbuf.len) return;
    @memcpy(zbuf[0..path.len], path);
    zbuf[path.len] = 0;
    _ = unlink(zbuf[0..path.len :0].ptr);
}

test "export gguf: roundtrip parse con el parser del engine" {
    const allocator = testing.allocator;
    const d: usize = 8;
    const alpha: f32 = 0.15;

    // Pesos de entrenamiento con valores distinguibles
    const w_gate = try allocator.alloc(f32, d * 2 * d);
    defer allocator.free(w_gate);
    const w_state = try allocator.alloc(f32, d * d);
    defer allocator.free(w_state);
    for (w_gate, 0..) |*v, i| v.* = @floatFromInt(i + 1); // 1..2d²
    for (w_state, 0..) |*v, i| v.* = @floatFromInt(1000 + i); // 1000..1000+d²

    var tmp_name_buf: [64]u8 = undefined;
    const tmp_path = try tmpPath(&tmp_name_buf, "roundtrip");

    // Roundtrip usa el Io threaded estándar (mismo que main del engine)
    const io = std.Io.Threaded.global_single_threaded.io();
    try export_gguf.writeRltGguf(io, allocator, tmp_path, .{
        .num_layers = 1,
        .d = d,
        .alpha = alpha,
        .swa_window = 64,
    }, &.{.{ .w_gate = w_gate, .w_state = w_state }});
    defer rmTmp(tmp_path);

    // ─── Re-parse con el parser del engine ───
    var file = try gguf.GgufFile.fromFile(io, allocator, tmp_path);
    defer file.deinit();

    // 1. Metadata
    const alpha_meta = file.getMeta("rlt.feedback_alpha") orelse return error.MissingAlpha;
    const alpha_val = alpha_meta.asF32() orelse return error.BadAlphaType;
    try testing.expectApproxEqAbs(alpha, alpha_val, 1e-6);

    const swa_meta = file.getMeta("rlt.swa_window") orelse return error.MissingSwa;
    const swa_val = swa_meta.asU32() orelse return error.BadSwaType;
    try testing.expectEqual(@as(u32, 64), swa_val);

    // 2. Tensor gate: dims GGUF [2d, d] = [in, out]
    const gate_info = file.getTensor("blk.0.rlt.feedback_gate.weight") orelse return error.MissingGateTensor;
    try testing.expectEqual(@as(u32, 2), gate_info.n_dims);
    try testing.expectEqual(@as(u64, 2 * d), gate_info.dims[0]); // in = 2d
    try testing.expectEqual(@as(u64, d), gate_info.dims[1]); // out = d
    try testing.expect(gate_info.dtype == .f32);

    // 3. Dequant + transposición como loadGgufF32: GGUF [in,out] → [out,in]
    const gate_numel: usize = @intCast(gate_info.numel());
    try testing.expectEqual(d * 2 * d, gate_numel);
    const f32buf = try allocator.alloc(f32, gate_numel);
    defer allocator.free(f32buf);
    try gguf.dequantTensor(gate_info, file.tensorData(gate_info), f32buf);

    // w_gate entrenamiento [d, 2d]: w_gate[c*2d + r]. El export escribió
    // GGUF[r*d + c] = w_gate[c*2d + r]; la transposición del load la invierte.
    for (0..d) |c| {
        for (0..2 * d) |r| {
            // f32buf viene en orden GGUF row-major [in=2d][out=d]:
            // elemento (r, c) está en f32buf[r*d + c]
            try testing.expectEqual(w_gate[c * (2 * d) + r], f32buf[r * d + c]);
        }
    }

    // 4. Tensor state: dims GGUF [d, d] — sin transposición efectiva
    const state_info = file.getTensor("blk.0.rlt.feedback_state.weight") orelse return error.MissingStateTensor;
    try testing.expectEqual(@as(u64, d), state_info.dims[0]);
    try testing.expectEqual(@as(u64, d), state_info.dims[1]);
    const state_numel: usize = @intCast(state_info.numel());
    try testing.expectEqual(d * d, state_numel);
    const state_f32 = try allocator.alloc(f32, state_numel);
    defer allocator.free(state_f32);
    try gguf.dequantTensor(state_info, file.tensorData(state_info), state_f32);
    for (w_state, 0..) |expected, i| {
        try testing.expectEqual(expected, state_f32[i]);
    }
}

test "export gguf: header GGUF v3 correcto (magic + version + counts)" {
    const allocator = testing.allocator;
    const d: usize = 4;

    const w_gate = try allocator.alloc(f32, d * 2 * d);
    defer allocator.free(w_gate);
    const w_state = try allocator.alloc(f32, d * d);
    defer allocator.free(w_state);
    @memset(w_gate, 0.1);
    @memset(w_state, 0.2);

    var tmp_name_buf: [64]u8 = undefined;
    const tmp_path = try tmpPath(&tmp_name_buf, "hdr");

    const io = std.Io.Threaded.global_single_threaded.io();
    try export_gguf.writeRltGguf(io, allocator, tmp_path, .{
        .num_layers = 2,
        .d = d,
        .alpha = 0.5,
    }, &.{
        .{ .w_gate = w_gate, .w_state = w_state },
        .{ .w_gate = w_gate, .w_state = w_state },
    });
    defer rmTmp(tmp_path);

    var file = try gguf.GgufFile.fromFile(io, allocator, tmp_path);
    defer file.deinit();

    // 2 capas × 2 tensores = 4; KVs: arch, name, alpha (+swa solo si >0)
    try testing.expectEqual(@as(u64, 4), file.tensors.count());
    try testing.expect(file.getTensor("blk.1.rlt.feedback_gate.weight") != null);
    try testing.expect(file.getTensor("blk.1.rlt.feedback_state.weight") != null);
    // Sin swa_window=0 ⇒ el KV no debe existir
    try testing.expect(file.getMeta("rlt.swa_window") == null);
}
