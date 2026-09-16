const std = @import("std");
const cudaz = @import("cudaz");
const layer_kernels = @import("layer_kernels");

fn parityP06(
    gpa: std.mem.Allocator,
    lk: *layer_kernels.LayerKernels,
    stream: cudaz.CUstream,
    qtype: u8,
    K: usize,
    N: usize,
    w_bytes: []const u8,
) !void {
    const a1 = try gpa.alloc(f32, K);
    defer gpa.free(a1);
    {
        var rng = std.Random.Xoshiro256.init(7032);
        for (a1) |*v| v.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);
    }

    const d_a = try cudaz.cuMemAlloc(K * @sizeOf(f32));
    defer cudaz.cuMemFree(d_a);
    const d_w = try cudaz.cuMemAlloc(w_bytes.len);
    defer cudaz.cuMemFree(d_w);
    const d_c_ref = try cudaz.cuMemAlloc(N * @sizeOf(f32));
    defer cudaz.cuMemFree(d_c_ref);
    const d_c_dp4a = try cudaz.cuMemAlloc(N * @sizeOf(f32));
    defer cudaz.cuMemFree(d_c_dp4a);

    try cudaz.cuMemcpyHtoD(d_a, @intFromPtr(a1.ptr), K * @sizeOf(f32));
    try cudaz.cuMemcpyHtoD(d_w, @intFromPtr(w_bytes.ptr), w_bytes.len);

    try lk.qgemm(d_a, d_w, d_c_ref, 1, K, N, qtype);
    switch (qtype) {
        1 => try lk.q41GemmM1Dp4a(d_a, d_w, d_c_dp4a, K, N),
        7 => try lk.q2kGemmM1Dp4a(d_a, d_w, d_c_dp4a, K, N),
        8 => try lk.iq3sGemmM1Dp4a(d_a, d_w, d_c_dp4a, K, N),
        9 => try lk.iq2sGemmM1Dp4a(d_a, d_w, d_c_dp4a, K, N),
        10 => try lk.iq4nlGemmM1Dp4a(d_a, d_w, d_c_dp4a, K, N),
        12 => try lk.iq3xxsGemmM1Dp4a(d_a, d_w, d_c_dp4a, K, N),
        13 => try lk.iq2xxsGemmM1Dp4a(d_a, d_w, d_c_dp4a, K, N),
        14 => try lk.iq2xsGemmM1Dp4a(d_a, d_w, d_c_dp4a, K, N),
        18 => try lk.iq4xsGemmM1Dp4a(d_a, d_w, d_c_dp4a, K, N),
        else => unreachable,
    }
    try cudaz.cuStreamSynchronize(stream);

    const c_ref = try gpa.alloc(f32, N);
    defer gpa.free(c_ref);
    const c_dp4a = try gpa.alloc(f32, N);
    defer gpa.free(c_dp4a);
    try cudaz.cuMemcpyDtoH(@intFromPtr(c_ref.ptr), d_c_ref, N * @sizeOf(f32));
    try cudaz.cuMemcpyDtoH(@intFromPtr(c_dp4a.ptr), d_c_dp4a, N * @sizeOf(f32));

    var bad: usize = 0;
    var max_rel: f32 = 0;
    var max_abs: f32 = 0;
    for (0..N) |j| {
        const abs_diff = @abs(c_dp4a[j] - c_ref[j]);
        const rel = abs_diff / @max(@abs(c_ref[j]), 1.0);
        max_rel = @max(max_rel, rel);
        max_abs = @max(max_abs, abs_diff);
        if (rel > 1e-2) bad += 1;
    }
    const tag = switch (qtype) {
        1 => "q41", 7 => "q2k", 8 => "iq3s", 9 => "iq2s",
        10 => "iq4nl", 12 => "iq3xxs", 13 => "iq2_xxs", 14 => "iq2xs", 18 => "iq4xs",
        else => "?",
    };
    std.debug.print("{s} dp4a M=1 k={d} n={d}: bad={d}/{d} max_rel={e} max_abs={e}\n", .{ tag, K, N, bad, N, max_rel, max_abs });
    if (bad > 0) return error.Dp4aParityFail;
}

fn makeB_q41(gpa: std.mem.Allocator, K: usize, N: usize) ![]u8 {
    const row_bytes = 20 * (K / 32);
    const bytes = try gpa.alloc(u8, N * row_bytes);
    @memset(bytes, 0);
    var rng = std.Random.Xoshiro256.init(7777);
    for (0..N) |j| {
        const bp = bytes[j * row_bytes ..][0..row_bytes];
        for (0..(K / 32)) |blk| {
            const d: f16 = @floatCast(1.0);
            const m: f16 = @floatCast(0.0);
            const d_bytes = std.mem.toBytes(d);
            @memcpy(bp[blk * 20 ..][0..2], &d_bytes);
            const m_bytes = std.mem.toBytes(m);
            @memcpy(bp[blk * 20 + 2 ..][0..2], &m_bytes);
            for (0..16) |i| {
                bp[blk * 20 + 4 + i] = @intCast(rng.random().int(u8) & 0xF);
            }
        }
    }
    return bytes;
}

fn makeB_q4_0(gpa: std.mem.Allocator, K: usize, N: usize) ![]u8 {
    const row_bytes = 18 * (K / 32);
    const bytes = try gpa.alloc(u8, N * row_bytes);
    @memset(bytes, 0);
    var rng = std.Random.Xoshiro256.init(7777);
    for (0..N) |j| {
        const bp = bytes[j * row_bytes ..][0..row_bytes];
        for (0..(K / 32)) |blk| {
            const d: f16 = @floatCast(1.0);
            const d_bytes18 = std.mem.toBytes(d);
            @memcpy(bp[blk * 18 ..][0..2], &d_bytes18);
            for (0..16) |i| {
                bp[blk * 18 + 2 + i] = @intCast((rng.random().int(u8) & 0xF) + 8);
            }
        }
    }
    return bytes;
}

fn makeB_iq4_nl(gpa: std.mem.Allocator, K: usize, N: usize) ![]u8 {
    const row_bytes = 18 * (K / 32);
    const bytes = try gpa.alloc(u8, N * row_bytes);
    @memset(bytes, 0);
    var rng = std.Random.Xoshiro256.init(7777);
    for (0..N) |j| {
        const bp = bytes[j * row_bytes ..][0..row_bytes];
        for (0..(K / 32)) |blk| {
            const d: f16 = @floatCast(1.0);
            const d_bytes18 = std.mem.toBytes(d);
            @memcpy(bp[blk * 18 ..][0..2], &d_bytes18);
            for (0..16) |i| {
                bp[blk * 18 + 2 + i] = @intCast(rng.random().int(u8) & 0xF);
            }
        }
    }
    return bytes;
}

fn makeB_q2k(gpa: std.mem.Allocator, K: usize, N: usize) ![]u8 {
    const sb_total = K / 256;
    const row_bytes = 84 * sb_total;
    const bytes = try gpa.alloc(u8, N * row_bytes);
    @memset(bytes, 0);
    var rng = std.Random.Xoshiro256.init(7777);
    for (0..N) |j| {
        const bp = bytes[j * row_bytes ..][0..row_bytes];
        for (0..sb_total) |sb| {
            const d: f16 = @floatCast(1.0);
            const mn: f16 = @floatCast(0.0);
            const d_bytes = std.mem.toBytes(d);
            @memcpy(bp[sb * 84 + 80 ..][0..2], &d_bytes);
            const mn_bytes = std.mem.toBytes(mn);
            @memcpy(bp[sb * 84 + 82 ..][0..2], &mn_bytes);
            for (0..64) |i| {
                bp[sb * 84 + 16 + i] = @intCast(rng.random().int(u8) & 0x3);
            }
            for (0..16) |i| {
                bp[sb * 84 + i] = @intCast((rng.random().int(u8) & 0xF) | ((rng.random().int(u8) & 0x3) << 4));
            }
        }
    }
    return bytes;
}

fn makeB_iq3s(gpa: std.mem.Allocator, K: usize, N: usize) ![]u8 {
    const sb_total = K / 256;
    const row_bytes = 110 * sb_total;
    const bytes = try gpa.alloc(u8, N * row_bytes);
    @memset(bytes, 0);
    var rng = std.Random.Xoshiro256.init(7777);
    for (0..N) |j| {
        const bp = bytes[j * row_bytes ..][0..row_bytes];
        for (0..sb_total) |sb| {
            const d: f16 = @floatCast(1.0);
            const d_bytes = std.mem.toBytes(d);
            @memcpy(bp[sb * 110 ..][0..2], &d_bytes);
            for (0..64) |i| {
                bp[sb * 110 + 2 + i] = @intCast(rng.random().int(u8));
            }
            for (0..8) |i| {
                bp[sb * 110 + 66 + i] = @intCast(rng.random().int(u8));
            }
            for (0..32) |i| {
                bp[sb * 110 + 74 + i] = @intCast(rng.random().int(u8));
            }
            for (0..4) |i| {
                bp[sb * 110 + 106 + i] = @intCast(rng.random().int(u8) & 0xF);
            }
        }
    }
    return bytes;
}

fn makeB_iq2s(gpa: std.mem.Allocator, K: usize, N: usize) ![]u8 {
    const sb_total = K / 256;
    const row_bytes = 82 * sb_total;
    const bytes = try gpa.alloc(u8, N * row_bytes);
    @memset(bytes, 0);
    var rng = std.Random.Xoshiro256.init(7777);
    for (0..N) |j| {
        const bp = bytes[j * row_bytes ..][0..row_bytes];
        for (0..sb_total) |sb| {
            const d: f16 = @floatCast(1.0);
            const d_bytes82 = std.mem.toBytes(d);
            @memcpy(bp[sb * 82 ..][0..2], &d_bytes82);
            for (0..32) |i| {
                bp[sb * 82 + 2 + i] = @intCast(rng.random().int(u8));
            }
            for (0..32) |i| {
                bp[sb * 82 + 34 + i] = @intCast(rng.random().int(u8));
            }
            for (0..8) |i| {
                bp[sb * 82 + 66 + i] = @intCast(rng.random().int(u8));
            }
            for (0..8) |i| {
                bp[sb * 82 + 74 + i] = @intCast(rng.random().int(u8));
            }
        }
    }
    return bytes;
}

fn makeB_iq4xs(gpa: std.mem.Allocator, K: usize, N: usize) ![]u8 {
    const sb_total = K / 256;
    const row_bytes = 136 * sb_total;
    const bytes = try gpa.alloc(u8, N * row_bytes);
    @memset(bytes, 0);
    var rng = std.Random.Xoshiro256.init(7777);
    for (0..N) |j| {
        const bp = bytes[j * row_bytes ..][0..row_bytes];
        for (0..sb_total) |sb| {
            const d: f16 = @floatCast(1.0);
            const d_bytes136 = std.mem.toBytes(d);
            @memcpy(bp[sb * 136 ..][0..2], &d_bytes136);
            @memcpy(bp[sb * 136 + 2 ..][0..2], &[_]u8{0, 0});
            for (0..4) |i| {
                bp[sb * 136 + 4 + i] = @intCast(rng.random().int(u8));
            }
            for (0..128) |i| {
                bp[sb * 136 + 8 + i] = @intCast(rng.random().int(u8) & 0xF);
            }
        }
    }
    return bytes;
}

fn makeB_iq3xxs(gpa: std.mem.Allocator, K: usize, N: usize) ![]u8 {
    const sb_total = K / 256;
    const row_bytes = 98 * sb_total;
    const bytes = try gpa.alloc(u8, N * row_bytes);
    @memset(bytes, 0);
    var rng = std.Random.Xoshiro256.init(7777);
    for (0..N) |j| {
        const bp = bytes[j * row_bytes ..][0..row_bytes];
        for (0..sb_total) |sb| {
            const d: f16 = @floatCast(1.0);
            const d_bytes98 = std.mem.toBytes(d);
            @memcpy(bp[sb * 98 ..][0..2], &d_bytes98);
            for (0..64) |i| {
                bp[sb * 98 + 2 + i] = @intCast(rng.random().int(u8));
            }
            for (0..32) |i| {
                bp[sb * 98 + 66 + i] = @intCast(rng.random().int(u8));
            }
        }
    }
    return bytes;
}

fn makeB_iq2xxs(gpa: std.mem.Allocator, K: usize, N: usize) ![]u8 {
    const sb_total = K / 256;
    const row_bytes = 66 * sb_total;
    const bytes = try gpa.alloc(u8, N * row_bytes);
    @memset(bytes, 0);
    var rng = std.Random.Xoshiro256.init(7777);
    for (0..N) |j| {
        const bp = bytes[j * row_bytes ..][0..row_bytes];
        for (0..sb_total) |sb| {
            const d: f16 = @floatCast(1.0);
            const d_bytes66 = std.mem.toBytes(d);
            @memcpy(bp[sb * 66 ..][0..2], &d_bytes66);
            for (0..64) |i| {
                bp[sb * 66 + 2 + i] = @intCast(rng.random().int(u8));
            }
        }
    }
    return bytes;
}

fn makeB_iq2xs(gpa: std.mem.Allocator, K: usize, N: usize) ![]u8 {
    const sb_total = K / 256;
    const row_bytes = 74 * sb_total;
    const bytes = try gpa.alloc(u8, N * row_bytes);
    @memset(bytes, 0);
    var rng = std.Random.Xoshiro256.init(7777);
    for (0..N) |j| {
        const bp = bytes[j * row_bytes ..][0..row_bytes];
        for (0..sb_total) |sb| {
            const d: f16 = @floatCast(1.0);
            const d_bytes74 = std.mem.toBytes(d);
            @memcpy(bp[sb * 74 ..][0..2], &d_bytes74);
            for (0..8) |i| {
                bp[sb * 74 + 66 + i] = @intCast(rng.random().int(u8));
            }
            for (0..64) |i| {
                bp[sb * 74 + 2 + i] = @intCast(rng.random().int(u8));
            }
        }
    }
    return bytes;
}

test "P0-6 dp4a parity" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    @import("debug").init();
    try cudaz.ensureContext();
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);
    var lk = try layer_kernels.LayerKernels.init(stream);
    defer lk.deinit();

    // q4_1 K=32 N=1
    {
        const b = try makeB_q41(gpa, 32, 1);
        defer gpa.free(b);
        try parityP06(gpa, &lk, stream, 1, 32, 1, b);
    }
    // q2_k K=256 N=1
    {
        const b = try makeB_q2k(gpa, 256, 1);
        defer gpa.free(b);
        try parityP06(gpa, &lk, stream, 7, 256, 1, b);
    }
    // iq3_s K=256 N=1
    {
        const b = try makeB_iq3s(gpa, 256, 1);
        defer gpa.free(b);
        try parityP06(gpa, &lk, stream, 8, 256, 1, b);
    }
    // iq2_s K=256 N=1
    {
        const b = try makeB_iq2s(gpa, 256, 1);
        defer gpa.free(b);
        try parityP06(gpa, &lk, stream, 9, 256, 1, b);
    }
    // iq4_xs K=256 N=1
    {
        const b = try makeB_iq4xs(gpa, 256, 1);
        defer gpa.free(b);
        try parityP06(gpa, &lk, stream, 18, 256, 1, b);
    }
    // iq4_nl K=32 N=1
    {
        const b = try makeB_iq4_nl(gpa, 32, 1);
        defer gpa.free(b);
        try parityP06(gpa, &lk, stream, 10, 32, 1, b);
    }
    // iq3_xxs K=256 N=1
    {
        const b = try makeB_iq3xxs(gpa, 256, 1);
        defer gpa.free(b);
        try parityP06(gpa, &lk, stream, 12, 256, 1, b);
    }
    // iq2_xxs K=256 N=1
    {
        const b = try makeB_iq2xxs(gpa, 256, 1);
        defer gpa.free(b);
        try parityP06(gpa, &lk, stream, 13, 256, 1, b);
    }
    // iq2_xs K=256 N=1
    {
        const b = try makeB_iq2xs(gpa, 256, 1);
        defer gpa.free(b);
        try parityP06(gpa, &lk, stream, 14, 256, 1, b);
    }
}
