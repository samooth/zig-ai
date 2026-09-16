//! VIDEO (TODO 10.7): decodificación de vídeo vía ffmpeg/ffprobe subprocess
//! (patrón del oráculo mtmd-helper.cpp:654-949; regla ola 2 — sin deps
//! nuevas, ffmpeg binario en PATH como llama.cpp).
//!
//! Pipeline: ffprobe (w/h/fps/duración) → `ffmpeg -f rawvideo rgb24` a
//! stdout → frames HWC u8 leídas completas (batch v1; vídeos de test son
//! cortos. El streaming lazy del oráculo queda como optimización futura).
//!
//! Temporal merge: frames agrupadas en PARES (Conv3D t=2 del ViT split,
//! unsloth qwenvl.py:111-117). Par impar → repetir última frame
//! (`nz = ceil(n/2)` — mtmd.cpp:99-109).

const std = @import("std");
const debugz = @import("debug");

pub const VideoError = error{
    FfprobeFailed,
    FfmpegFailed,
    BadDimensions,
    NoFrames,
    OutOfMemory,
};

pub const VideoInfo = struct {
    width: usize,
    height: usize,
    fps: f32,
    duration_s: f32,
    n_frames: usize,
};

pub const Frame = struct {
    data: []u8, // [H, W, 3] HWC
    width: usize,
    height: usize,
};

pub const VideoFrames = struct {
    frames: []Frame,
    info: VideoInfo,

    pub fn deinit(self: *VideoFrames, allocator: std.mem.Allocator) void {
        for (self.frames) |*f| allocator.free(f.data);
        allocator.free(self.frames);
        self.frames = &.{};
    }

    /// Pares temporales: frames consecutivas [0,1],[2,3],...; impar →
    /// repetir última. Devuelve slices (f0, f1) SIN ownership.
    pub fn pair(self: *const VideoFrames, i: usize) ?struct { f0: []const u8, f1: []const u8 } {
        const n_pairs = (self.frames.len + 1) / 2;
        if (i >= n_pairs) return null;
        const f0 = self.frames[i * 2].data;
        const f1 = if (i * 2 + 1 < self.frames.len)
            self.frames[i * 2 + 1].data
        else
            self.frames[self.frames.len - 1].data; // repetir última (impar)
        return .{ .f0 = f0, .f1 = f1 };
    }

    pub fn nPairs(self: *const VideoFrames) usize {
        return (self.frames.len + 1) / 2;
    }
};

/// ffprobe -v error -select_streams v:0 -show_entries stream=width,height,
/// r_frame_rate,duration -of csv=p=0 <path>
pub fn probe(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !VideoInfo {
    const argv = [_][]const u8{
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,duration",
        "-of",
        "csv=p=0",
        path,
    };
    const result = std.process.run(allocator, io, .{
        .argv = &argv,
        .stdout_limit = .limited(4 * 1024),
        .stderr_limit = .limited(4 * 1024),
    }) catch |e| {
        debugz.dbg.print("[video] ffprobe spawn error: {s}\n", .{@errorName(e)});
        return VideoError.FfprobeFailed;
    };
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    const ok = switch (result.term) {
        .exited => |code| code == 0,
        else => false,
    };
    if (!ok) return VideoError.FfprobeFailed;

    // csv: "width,height,r_frame_rate,duration" (r_frame_rate "N/D")
    var it = std.mem.splitScalar(u8, result.stdout, ',');
    const w_s = it.next() orelse return VideoError.BadDimensions;
    const h_s = it.next() orelse return VideoError.BadDimensions;
    const fps_s = it.next() orelse return VideoError.BadDimensions;
    const dur_s = it.next() orelse "0";
    const width = std.fmt.parseInt(usize, std.mem.trim(u8, w_s, " \n\r"), 10) catch return VideoError.BadDimensions;
    const height = std.fmt.parseInt(usize, std.mem.trim(u8, h_s, " \n\r"), 10) catch return VideoError.BadDimensions;
    if (width == 0 or height == 0) return VideoError.BadDimensions;

    var fps: f32 = 0;
    {
        var fit = std.mem.splitScalar(u8, std.mem.trim(u8, fps_s, " \n\r"), '/');
        const num = fit.next() orelse "0";
        const den = fit.next() orelse "1";
        const n = std.fmt.parseFloat(f32, num) catch 0;
        const d = std.fmt.parseFloat(f32, den) catch 1;
        if (d != 0) fps = n / d;
    }
    const dur = std.fmt.parseFloat(f32, std.mem.trim(u8, dur_s, " \n\r")) catch 0;
    const n_frames: usize = if (fps > 0 and dur > 0) @intFromFloat(dur * fps) else 0;
    return .{ .width = width, .height = height, .fps = fps, .duration_s = dur, .n_frames = n_frames };
}

/// Decodifica un vídeo a frames RGB24 HWC. `fps_target` remuestrea (0 = fps
/// nativo). El oráculo usa 4.0 default (mtmd-helper.h:127).
pub fn decode(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
    fps_target: f32,
) !VideoFrames {
    const info = try probe(io, allocator, path);

    var fps_buf: [32]u8 = undefined;
    const fps_arg = try std.fmt.bufPrint(&fps_buf, "{d:.3}", .{if (fps_target > 0) fps_target else info.fps});
    var pix_buf: [64]u8 = undefined;
    const pix_fmt = try std.fmt.bufPrint(&pix_buf, "{d}x{d}", .{ info.width, info.height });

    var vf_buf: [48]u8 = undefined;
    const vf_arg = try std.fmt.bufPrint(&vf_buf, "fps={s}", .{fps_arg});
    const full_argv = [_][]const u8{
        "ffmpeg",   "-nostdin",
        "-v",       "error",
        "-i",       path,
        "-vf",      vf_arg,
        "-f",       "rawvideo",
        "-pix_fmt", "rgb24",
        "-s",       pix_fmt,
        "pipe:1",
    };

    const result = std.process.run(allocator, io, .{
        .argv = &full_argv,
        .stdout_limit = .limited(1024 * 1024 * 1024), // 1GB cap
        .stderr_limit = .limited(16 * 1024),
    }) catch |e| {
        debugz.dbg.print("[video] ffmpeg spawn error: {s}\n", .{@errorName(e)});
        return VideoError.FfmpegFailed;
    };
    defer allocator.free(result.stderr);
    const ok = switch (result.term) {
        .exited => |code| code == 0,
        else => false,
    };
    if (!ok) {
        debugz.dbg.print("[video] ffmpeg falló para {s}: {s}\n", .{ path, result.stderr[0..@min(result.stderr.len, 512)] });
        allocator.free(result.stdout);
        return VideoError.FfmpegFailed;
    }

    const frame_bytes = info.width * info.height * 3;
    const n = result.stdout.len / frame_bytes;
    if (n == 0) {
        allocator.free(result.stdout);
        return VideoError.NoFrames;
    }

    // mover cada frame a su propio alloc (ownership limpia)
    const frames = try allocator.alloc(Frame, n);
    errdefer allocator.free(frames);
    var ok_frames: usize = 0;
    errdefer for (frames[0..ok_frames]) |*f| allocator.free(f.data);
    for (0..n) |i| {
        frames[i] = .{
            .data = try allocator.dupe(u8, result.stdout[i * frame_bytes ..][0..frame_bytes]),
            .width = info.width,
            .height = info.height,
        };
        ok_frames += 1;
    }
    allocator.free(result.stdout);

    debugz.dbg.printLevel(.info, "[video] {s}: {d} frames {d}x{d} @ {d:.2}fps ({d} pares)\n", .{ path, n, info.width, info.height, info.fps, (n + 1) / 2 });
    return .{ .frames = frames, .info = info };
}

/// Genera vídeos sintéticos de test (ffmpeg lavfi). Usado por los tests y
/// por el E2E de validación (blanco/transition).
pub fn makeTestVideo(io: std.Io, allocator: std.mem.Allocator, kind: []const u8, dst: []const u8) !void {
    const src_filter = if (std.mem.eql(u8, kind, "white"))
        "color=c=white:s=256x256:d=2:r=2"
    else if (std.mem.eql(u8, kind, "transition"))
        "gradients=s=256x256:d=2:r=2:c0=red:c1=blue"
    else if (std.mem.eql(u8, kind, "black"))
        "color=c=black:s=256x256:d=2:r=2"
    else
        "testsrc=s=256x256:d=2:r=2";
    const argv = [_][]const u8{
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", src_filter, "-pix_fmt", "yuv420p", dst,
    };
    const result = std.process.run(allocator, io, .{
        .argv = &argv,
        .stdout_limit = .limited(4 * 1024),
        .stderr_limit = .limited(4 * 1024),
    }) catch |e| {
        debugz.dbg.print("[video] lavfi spawn error: {s}\n", .{@errorName(e)});
        return VideoError.FfmpegFailed;
    };
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    const ok = switch (result.term) {
        .exited => |code| code == 0,
        else => false,
    };
    if (!ok) return VideoError.FfmpegFailed;
}

// ── Tests (requieren ffmpeg en PATH) ──────────────────────────────────────

const testing = std.testing;

test "video: probe + decode blanco 4 frames" {
    if (std.c.getenv("ZIG_AI_VIDEO_TESTS") == null) {
        debugz.dbg.print("[video] SKIP: ZIG_AI_VIDEO_TESTS=1 para tests ffmpeg\n", .{});
        return error.SkipZigTest;
    }
    const gpa = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    try makeTestVideo(io, gpa, "white", "/tmp/zai-video-white.mp4");
    const info = try probe(io, gpa, "/tmp/zai-video-white.mp4");
    try testing.expect(info.width == 256);
    try testing.expect(info.height == 256);
    try testing.expect(info.fps > 0);

    var vf = try decode(io, gpa, "/tmp/zai-video-white.mp4", 0);
    defer vf.deinit(gpa);
    try testing.expect(vf.frames.len >= 2); // 2s @ 2fps = 4
    // par 0: frames 0,1
    const p0 = vf.pair(0).?;
    try testing.expect(p0.f0.len == 256 * 256 * 3);
    // todas blancas
    try testing.expect(p0.f0[0] == 255);
    // pares == ceil(n/2)
    try testing.expectEqual(vf.nPairs(), (vf.frames.len + 1) / 2);
}
