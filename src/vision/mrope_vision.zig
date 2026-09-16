//! M-RoPE VISION mode — port fiel de `ggml_rope_multi` con
//! `GGML_ROPE_TYPE_VISION` para el encoder CLIP ViT de Qwen2/3-VL.
//!
//! Referencias exactas (llama.cpp):
//!   - ggml_mrope_cache_init: ggml/src/ggml-cpu/ops.cpp:5862-5930
//!   - modo VISION + rotate_pairs: ops.cpp:6015,6078 (rotate_pairs con
//!     n=ne0, n_offset=n_dims — pares INTERLEAVED adyacentes (2i, 2i+1),
//!     NO el half-split NEOX de applyRoPEMultiSection de rope.zig:127)
//!   - positions ViT Qwen-VL: tools/mtmd/clip.cpp:4068-4086 (por patch:
//!     pos = (y+dy, x+dx, y+dy, x+dx) — 4 ids por token)
//!   - sections ViT: models/qwen3vl.cpp:14 ([d_head/4 ×4])
//!
//! Layout de entrada:
//!   Q/K: [n_pos, n_head, head_dim] f32 (pos-major)
const std = @import("std");

/// Cache cos/sin por token: [n_pos][head_dim] (pares interleaved).
/// theta por par i: p_sección(i) · base^(-2·i/n_pairs), con n_pairs=hd/2
/// (el grafo ViT pasa n_dims=d_head/2, ops.cpp:6008).
pub fn buildVisionCache(
    cache: []f32, // [n_pos * head_dim]
    ids: []const [4]i32, // por token: (t, h, w, e) — ViT: (y, x, y, x)
    head_dim: usize,
    sections: [4]usize, // [hd/4, hd/4, hd/4, hd/4]
) void {
    for (ids, 0..) |id, tok| {
        const base_t: f32 = @floatFromInt(id[0]);
        const base_h: f32 = @floatFromInt(id[1]);
        const base_w: f32 = @floatFromInt(id[2]);
        const base_e: f32 = @floatFromInt(id[3]);

        const row = cache[tok * head_dim ..][0..head_dim];
        const n_pairs = head_dim / 2;
        const sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
        const sec_w = sections[0] + sections[1];
        const sec_e = sections[2] + sec_w;

        // Modo VISION: NO imrope — bloques contiguos por sección
        // (ops.cpp:5908-5918, rama else).
        for (0..n_pairs) |ip| {
            const sector = ip % sect_dims;
            const p: f32 = blk: {
                if (sector < sections[0]) break :blk base_t;
                if (sector < sec_w) break :blk base_h;
                if (sector < sec_e) break :blk base_w;
                break :blk base_e;
            };
            const theta = p * std.math.pow(f32, 10000.0, -2.0 * @as(f32, @floatFromInt(ip)) / @as(f32, @floatFromInt(n_pairs)));
            row[ip * 2] = @cos(theta);
            row[ip * 2 + 1] = @sin(theta);
        }
    }
}

/// Aplica M-RoPE VISION in-place sobre Q/K [n_pos, n_head, head_dim].
/// rotate_pairs interleaved: pares adyacentes (2i, 2i+1), TODO head_dim.
pub fn applyMRopeVision(
    comptime T: type,
    Q: []T,
    K: []T,
    ids: []const [4]i32,
    n_head: usize,
    head_dim: usize,
    sections: [4]usize,
    freq_base: f32, // ViT Qwen: 10000 (qwen3vl.cpp:106)
    scratch: []f32, // >= n_pos * head_dim
) void {
    const n_pos = ids.len;
    std.debug.assert(scratch.len >= n_pos * head_dim);
    _ = freq_base; // base fija 10000 en buildVisionCache (fiel al grafo)

    buildVisionCache(scratch, ids, head_dim, sections);

    const hd = head_dim;
    for (0..n_pos) |t| {
        const cache = scratch[t * hd ..][0..hd];
        for (0..n_head) |h| {
            const q = Q[(t * n_head + h) * hd ..][0..hd];
            const k = K[(t * n_head + h) * hd ..][0..hd];
            var i: usize = 0;
            while (i < hd) : (i += 2) {
                const cos_v = cache[i];
                const sin_v = cache[i + 1];

                const q0 = @as(f32, @floatCast(q[i]));
                const q1 = @as(f32, @floatCast(q[i + 1]));
                q[i] = @floatCast(q0 * cos_v - q1 * sin_v);
                q[i + 1] = @floatCast(q0 * sin_v + q1 * cos_v);

                const k0 = @as(f32, @floatCast(k[i]));
                const k1 = @as(f32, @floatCast(k[i + 1]));
                k[i] = @floatCast(k0 * cos_v - k1 * sin_v);
                k[i + 1] = @floatCast(k0 * sin_v + k1 * cos_v);
            }
        }
    }
}

/// Genera los position-ids 4D del ViT Qwen2/3-VL para un grid de `ph×pw`
/// patches. Port exacto de clip.cpp:4068-4086:
///   por merge-block (y,x) [paso merge] × (dy,dx) en [0..2)²:
///     pos = (y+dy, x+dx, y+dy, x+dx)
/// El orden coincide con el pixel-shuffle del input (bloques 2x2
/// consecutivos) — MISMO layout que produce spatialMergePatches.
pub fn visionPosIds(
    out: [][4]i32, // [ph*pw]
    ph: usize, // patches verticales
    pw: usize, // patches horizontales
    merge: usize, // spatial_merge_size (2)
) void {
    std.debug.assert(out.len == ph * pw);
    var ptr: usize = 0;
    var y: usize = 0;
    while (y < ph) : (y += merge) {
        var x: usize = 0;
        while (x < pw) : (x += merge) {
            var dy: usize = 0;
            while (dy < 2) : (dy += 1) {
                var dx: usize = 0;
                while (dx < 2) : (dx += 1) {
                    out[ptr] = .{
                        @intCast(y + dy),
                        @intCast(x + dx),
                        @intCast(y + dy),
                        @intCast(x + dx),
                    };
                    ptr += 1;
                }
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

const testing = std.testing;

test "visionPosIds: grid 4x4, merge 2 → 16 ids en orden pixel-shuffle" {
    var ids: [16][4]i32 = undefined;
    visionPosIds(&ids, 4, 4, 2);
    // Primer bloque 2x2: (0,0),(0,1),(1,0),(1,1)
    try testing.expectEqualSlices(i32, &.{ 0, 0, 0, 0 }, &ids[0]);
    try testing.expectEqualSlices(i32, &.{ 0, 1, 0, 1 }, &ids[1]);
    try testing.expectEqualSlices(i32, &.{ 1, 0, 1, 0 }, &ids[2]);
    try testing.expectEqualSlices(i32, &.{ 1, 1, 1, 1 }, &ids[3]);
    // Segundo bloque (y=0, x=2): (0,2),(0,3),(1,2),(1,3)
    try testing.expectEqualSlices(i32, &.{ 0, 2, 0, 2 }, &ids[4]);
    try testing.expectEqualSlices(i32, &.{ 1, 3, 1, 3 }, &ids[7]);
    // Último bloque (y=2, x=2): (2,2)..(3,3)
    try testing.expectEqualSlices(i32, &.{ 2, 2, 2, 2 }, &ids[12]);
    try testing.expectEqualSlices(i32, &.{ 3, 3, 3, 3 }, &ids[15]);
}

test "visionPosIds: grid 2x2 (1 bloque), merge 2" {
    var ids: [4][4]i32 = undefined;
    visionPosIds(&ids, 2, 2, 2);
    try testing.expectEqualSlices(i32, &.{ 0, 0, 0, 0 }, &ids[0]);
    try testing.expectEqualSlices(i32, &.{ 1, 1, 1, 1 }, &ids[3]);
}

test "applyMRopeVision: pos iguales a 0 → Q/K sin cambio (cos=1, sin=0)" {
    const n_pos = 2;
    const n_head = 1;
    const hd = 8;
    var q = [_]f32{0} ** (n_pos * n_head * hd);
    var k = [_]f32{0} ** (n_pos * n_head * hd);
    for (&q, 0..) |*v, i| v.* = @floatFromInt(i % hd);
    const q_orig = q;
    const ids = [_][4]i32{ .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 } };
    var scratch: [n_pos * hd]f32 = undefined;
    applyMRopeVision(f32, &q, &k, &ids, n_head, hd, .{ 2, 2, 2, 2 }, 10000.0, &scratch);
    try testing.expectApproxEqAbs(q_orig[0], q[0], 1e-6);
    for (k) |v| try testing.expectApproxEqAbs(0, v, 1e-6);
}

test "applyMRopeVision: rotación en pos (1,0) afecta solo sección t" {
    const n_pos = 1;
    const n_head = 1;
    const hd = 8;
    var q = [_]f32{ 1, 0, 1, 0, 1, 0, 1, 0 };
    var k = [_]f32{0} ** 8;
    // sections [2,2,2,2]: pares 0-1 sección t (theta=1·base⁰=1),
    const ids = [_][4]i32{.{ 1, 0, 0, 0 }};
    var scratch: [n_pos * hd]f32 = undefined;
    applyMRopeVision(f32, &q, &k, &ids, n_head, hd, .{ 2, 2, 2, 2 }, 10000.0, &scratch);
    // Primer par (sector 0, p_t=1): theta=1 rad
    try testing.expectApproxEqAbs(@cos(1.0), q[0], 1e-6);
    try testing.expectApproxEqAbs(@sin(1.0), q[1], 1e-6);
    // Sector 1 (p_h=0): theta=0 ⇒ par intacto
    try testing.expectApproxEqAbs(1.0, q[2], 1e-6);
    try testing.expectApproxEqAbs(0.0, q[3], 1e-6);
}
