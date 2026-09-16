//! Preprocess — carga de imagen (stb_image), smart_resize (Qwen-VL),
//! normalización CLIP y conversión a CHW f32.
//!
//! Referencias (llama.cpp mtmd):
//!   - smart_resize (dyn_size): tools/mtmd/mtmd-image.cpp:166-189
//!     (calc_size_preserved_ratio con min/max_pixels)
//!   - resize_bilinear: mtmd-image.cpp:232-275 (align_corners style:
//!     ratio = (src-1)/(dst-1))
//!   - normalize: mtmd-image.cpp:7-27 ((px/255 - mean)/std)
//!   - preprocessor dyn_size (Qwen-VL): mtmd-image.cpp:29-166
//!
//! Pipeline Qwen-VL (Qwen2/3-VL):
//!   RGB u8 → smart_resize(align = patch·merge, min/max px) → f32/255
//!   → normalize(mean, std) → CHW [3, H, W]
const std = @import("std");
const debugz = @import("debug");

// stb_image (vendored, link C — build.zig vision_preprocess_mod)
const stbi = @cImport({
    @cInclude("stb_image.h");
});

pub const PreprocessError = error{
    ImageLoadFailed,
    InvalidDimensions,
    OutOfMemory,
};

pub const RgbImage = struct {
    data: []u8, // [H, W, 3] HWC RGB (ownership: caller frees via deinit)
    width: usize,
    height: usize,

    pub fn deinit(self: *RgbImage, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
        self.data = &.{};
    }
};

/// Carga PNG/JPG/BMP/TGA vía stb_image → RGB888.
/// El caller posee `data` (free con allocator).
pub fn loadImage(allocator: std.mem.Allocator, path: []const u8) PreprocessError!RgbImage {
    const cpath = allocator.alloc(u8, path.len + 1) catch
        return PreprocessError.OutOfMemory;
    defer allocator.free(cpath);
    @memcpy(cpath[0..path.len], path);
    cpath[path.len] = 0;

    var w: c_int = 0;
    var h: c_int = 0;
    var channels: c_int = 0;
    const pixels = stbi.stbi_load(cpath.ptr, &w, &h, &channels, 3);
    if (pixels == null or w <= 0 or h <= 0) {
        if (pixels != null) stbi.stbi_image_free(pixels);
        return PreprocessError.ImageLoadFailed;
    }
    errdefer stbi.stbi_image_free(pixels);

    const nw: usize = @intCast(w);
    const nh: usize = @intCast(h);
    const n = nw * nh * 3;
    const data = allocator.alloc(u8, n) catch {
        stbi.stbi_image_free(pixels);
        return PreprocessError.OutOfMemory;
    };
    @memcpy(data, pixels[0..n]);
    stbi.stbi_image_free(pixels);
    return .{ .data = data, .width = nw, .height = nh };
}

/// smart_resize (transformers Qwen-VL) — port de calc_size_preserved_ratio
/// (mtmd-image.cpp:170-189): tamaño alineado a `align_size` con
/// min_pixels <= W·H <= max_pixels, preservando ratio.
pub fn smartResizeTarget(
    width: usize,
    height: usize,
    align_size: usize, // patch_size · spatial_merge_size
    min_pixels: usize,
    max_pixels: usize,
) struct { w: usize, h: usize } {
    const wf: f32 = @floatFromInt(width);
    const hf: f32 = @floatFromInt(height);
    const align_f: f32 = @floatFromInt(align_size);

    const roundBy = struct {
        fn f(x: f32, a: f32) usize {
            return @intFromFloat(@round(x / a) * a);
        }
    }.f;
    const ceilBy = struct {
        fn f(x: f32, a: f32) usize {
            return @intFromFloat(@ceil(x / a) * a);
        }
    }.f;
    const floorBy = struct {
        fn f(x: f32, a: f32) usize {
            return @intFromFloat(@floor(x / a) * a);
        }
    }.f;

    var h_bar = @max(align_size, roundBy(hf, align_f));
    var w_bar = @max(align_size, roundBy(wf, align_f));

    const total = @as(f32, @floatFromInt(height)) * @as(f32, @floatFromInt(width));
    if (h_bar * w_bar > max_pixels) {
        const beta = @sqrt(total / @as(f32, @floatFromInt(max_pixels)));
        h_bar = @max(align_size, floorBy(hf / beta, align_f));
        w_bar = @max(align_size, floorBy(wf / beta, align_f));
    } else if (h_bar * w_bar < min_pixels) {
        const beta = @sqrt(@as(f32, @floatFromInt(min_pixels)) / total);
        h_bar = ceilBy(hf * beta, align_f);
        w_bar = ceilBy(wf * beta, align_f);
    }

    return .{ .w = w_bar, .h = h_bar };
}

/// Resize bilinear HWC RGB → HWC RGB. Port de mtmd-image.cpp:232-275
/// (align_corners: ratio=(src-1)/(dst-1), clamp bordes).
pub fn resizeBilinear(
    allocator: std.mem.Allocator,
    src: []const u8, // [sh, sw, 3]
    sw: usize,
    sh: usize,
    tw: usize,
    th: usize,
) ![]u8 {
    const dst = try allocator.alloc(u8, tw * th * 3);
    errdefer allocator.free(dst);

    const x_ratio: f32 = if (tw > 1)
        @as(f32, @floatFromInt(sw - 1)) / @as(f32, @floatFromInt(tw - 1))
    else
        0.0;
    const y_ratio: f32 = if (th > 1)
        @as(f32, @floatFromInt(sh - 1)) / @as(f32, @floatFromInt(th - 1))
    else
        0.0;

    for (0..th) |y| {
        for (0..tw) |x| {
            const px = @as(f32, @floatFromInt(x)) * x_ratio;
            const py = @as(f32, @floatFromInt(y)) * y_ratio;

            const x0 = @min(@as(usize, @intFromFloat(px)), sw - 1);
            const y0 = @min(@as(usize, @intFromFloat(py)), sh - 1);
            const x1 = @min(x0 + 1, sw - 1);
            const y1 = @min(y0 + 1, sh - 1);

            const xf = px - @as(f32, @floatFromInt(x0));
            const yf = py - @as(f32, @floatFromInt(y0));

            const di = (y * tw + x) * 3;
            for (0..3) |c| {
                const p00: f32 = @floatFromInt(src[(y0 * sw + x0) * 3 + c]);
                const p10: f32 = @floatFromInt(src[(y0 * sw + x1) * 3 + c]);
                const p01: f32 = @floatFromInt(src[(y1 * sw + x0) * 3 + c]);
                const p11: f32 = @floatFromInt(src[(y1 * sw + x1) * 3 + c]);
                const top = p00 + (p10 - p00) * xf;
                const bottom = p01 + (p11 - p01) * xf;
                const v = top + (bottom - top) * yf;
                dst[di + c] = @intFromFloat(@min(255.0, @max(0.0, @round(v))));
            }
        }
    }
    return dst;
}

pub const PreprocessedImage = struct {
    /// [3, H, W] CHW f32 normalizado — input directo del patch-embed conv.
    data: []f32,
    width: usize,
    height: usize,
    /// Grid de patches (ANTES del merge): H/patch, W/patch
    grid_y: usize,
    grid_x: usize,
    /// Grid final tras spatial merge (tokens de imagen que va a consumir el LLM)
    merge_grid_y: usize,
    merge_grid_x: usize,
};

/// Pipeline completo: load → smartResize → CHW f32 normalizado.
/// `patch_size` y `merge` definen el alignment (patch·merge, qwen vl).
pub fn preprocess(
    allocator: std.mem.Allocator,
    rgb: []const u8, // [H, W, 3] HWC
    width: usize,
    height: usize,
    patch_size: usize,
    merge: usize, // spatial_merge_size
    min_pixels: usize,
    max_pixels: usize,
    mean: [3]f32,
    std_dev: [3]f32,
) !PreprocessedImage {
    const align_size = patch_size * merge;
    const target = smartResizeTarget(width, height, align_size, min_pixels, max_pixels);

    var src = rgb;
    var sw = width;
    var sh = height;
    var resized: ?[]u8 = null;
    defer if (resized) |r| allocator.free(r);
    if (target.w != width or target.h != height) {
        resized = try resizeBilinear(allocator, rgb, width, height, target.w, target.h);
        src = resized.?;
        sw = target.w;
        sh = target.h;
    }

    const n = 3 * sh * sw;
    const data = try allocator.alloc(f32, n);
    errdefer allocator.free(data);

    // HWC u8 → CHW f32 con (px/255 - mean)/std por canal
    for (0..3) |c| {
        const m = mean[c];
        const s = std_dev[c];
        const plane = data[c * sh * sw ..][0 .. sh * sw];
        for (0..sh) |y| {
            for (0..sw) |x| {
                const px: f32 = @floatFromInt(src[(y * sw + x) * 3 + c]);
                plane[y * sw + x] = (px / 255.0 - m) / s;
            }
        }
    }

    const grid_y = sh / patch_size;
    const grid_x = sw / patch_size;
    return .{
        .data = data,
        .width = sw,
        .height = sh,
        .grid_y = grid_y,
        .grid_x = grid_x,
        .merge_grid_y = grid_y / merge,
        .merge_grid_x = grid_x / merge,
    };
}

// ── Tests ──────────────────────────────────────────────────────────────────

const testing = std.testing;

test "smartResizeTarget: cuadrado ya alineado no cambia" {
    const r = smartResizeTarget(448, 448, 28, 3136, 1003520);
    try testing.expectEqual(448, r.w);
    try testing.expectEqual(448, r.h);
}

test "smartResizeTarget: alinea a múltiplo de align" {
    const r = smartResizeTarget(100, 100, 28, 3136, 1003520);
    // 100 → round a múltiplo de 28: 112
    try testing.expectEqual(112, r.w);
    try testing.expectEqual(112, r.h);
}

test "smartResizeTarget: min_pixels amplía" {
    // 28x28 = 784 px < 3136 → beta=sqrt(3136/784)=2 → 56x56=3136
    const r = smartResizeTarget(28, 28, 28, 3136, 1003520);
    try testing.expectEqual(56, r.w);
    try testing.expectEqual(56, r.h);
}

test "smartResizeTarget: max_pixels reduce" {
    // 2000x2000 = 4M px > 1003520 → beta=sqrt(4M/1003520)≈1.996 → floor → 1008
    const r = smartResizeTarget(2000, 2000, 28, 3136, 1003520);
    try testing.expect(r.w * r.h <= 1003520 + 28 * 28); // tolerancia 1 fila
    try testing.expect(r.w % 28 == 0 and r.h % 28 == 0);
}

test "resizeBilinear: identidad cuando tw==sw" {
    const a = std.testing.allocator;
    const src = [_]u8{ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120 }; // 2x2x3
    const dst = try resizeBilinear(a, &src, 2, 2, 2, 2);
    defer a.free(dst);
    try testing.expectEqualSlices(u8, &src, dst);
}

test "resizeBilinear: upscale 2x2→4x4 interpola" {
    const a = std.testing.allocator;
    const src = [_]u8{0} ** 3 ++ [_]u8{100} ** 3 ++ [_]u8{200} ** 3 ++ [_]u8{255} ** 3;
    const dst = try resizeBilinear(a, &src, 2, 2, 4, 4);
    defer a.free(dst);
    // Esquina (0,0) = src(0,0) = 0
    try testing.expectEqual(@as(u8, 0), dst[0]);
    // Centro (1..2, 1..2) ≈ promedio de los 4
    const c = dst[(1 * 4 + 1) * 3];
    try testing.expect(c > 60 and c < 200);
}

test "preprocess: CHW layout + normalización" {
    const a = std.testing.allocator;
    // 1x1 imagen RGB (28x28 sería el mínimo real, pero para el test basta)
    // Nota: smartResize con align=28 ampliaría 1x1... usamos dims 28x28.
    var rgb: [28 * 28 * 3]u8 = undefined;
    for (0..28) |y| {
        for (0..28) |x| {
            rgb[(y * 28 + x) * 3 + 0] = 255; // R full
            rgb[(y * 28 + x) * 3 + 1] = 0; // G 0
            rgb[(y * 28 + x) * 3 + 2] = 128; // B mid
        }
    }
    const img = try preprocess(a, &rgb, 28, 28, 14, 2, 3136, 1003520, .{ 0.5, 0.5, 0.5 }, .{ 0.5, 0.5, 0.5 });
    defer a.free(img.data);
    try testing.expectEqual(28, img.width);
    try testing.expectEqual(2, img.grid_x);
    try testing.expectEqual(2, img.grid_y);
    try testing.expectEqual(1, img.merge_grid_x);
    try testing.expectEqual(1, img.merge_grid_y);
    // R: (255/255 - 0.5)/0.5 = 1.0
    try testing.expectApproxEqAbs(1.0, img.data[0], 1e-6);
    // G: (0 - 0.5)/0.5 = -1.0
    try testing.expectApproxEqAbs(-1.0, img.data[28 * 28], 1e-6);
}
