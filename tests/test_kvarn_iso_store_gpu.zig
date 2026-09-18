// ISO-BISECT: store kvh=2 con LAYOUT REAL + args idénticos al path
// gpucache appendTokens (gps=3, eager=1, 1 chunk de 128). Si hanga ⇒
// bug en kvarn_store_kernel con grid=n_record_heads=2. Pasa ⇒ el hang
// está en init_descs/refresh o en la secuencia del cache.
const std = @import("std");

fn stdoutPrint(io: std.Io, comptime fmt: []const u8, args: anytype) !void {
    var stdout_buf: [256]u8 = undefined;
    const stdout_file = std.Io.File.stdout();
    var stdout_writer = stdout_file.writer(io, &stdout_buf);
    try stdout_writer.interface.print(fmt, args);
}
const testing = std.testing;
const cudaz = @import("cudaz");
const kvk = @import("kvarn_kernels");
const kvarn = @import("kv_cache").kvarn;
const build_options = @import("build_options");

test "iso: store SOLO kvh=2 layout real (bisect hang)" {
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;
    cudaz.ensureContext() catch return error.SkipZigTest;
    const kmod = try cudaz.cuModuleLoad(build_options.kvarn_cubin);
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);
    const a = testing.allocator;

    const kvh: usize = 2;
    const n: usize = 128;
    const layout = try kvarn.KvarnRecordLayout.init(128, 5, 4);
    const rec_bytes: c_int = @intCast(layout.tile_bytes);

    const cur = try a.alloc(f32, n * kvh * 128);
    defer a.free(cur);
    for (cur, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 7)) * 0.1;
    const d_cur = try cudaz.cuMemAlloc(4 * cur.len);
    defer cudaz.cuMemFree(d_cur);
    try cudaz.cuMemcpyHtoD(d_cur, @intFromPtr(cur.ptr), 4 * cur.len);

    const idx = try a.alloc(i64, n);
    defer a.free(idx);
    for (idx, 0..) |*e, i| e.* = @intCast(i);
    const d_idx = try cudaz.cuMemAlloc(8 * n);
    defer cudaz.cuMemFree(d_idx);
    try cudaz.cuMemcpyHtoD(d_idx, @intFromPtr(idx.ptr), 8 * n);

    const stage_len = 4 * 128 * (2 * kvh) * 128;
    const d_stage = try cudaz.cuMemAlloc(2 * stage_len);
    defer cudaz.cuMemFree(d_stage);
    try cudaz.cuMemsetD8(d_stage, 0, 2 * stage_len);

    const n_groups: usize = 3; // como gpucache ctx=384
    const d_rec = try cudaz.cuMemAlloc(@as(usize, @intCast(rec_bytes)) * n_groups * kvh);
    defer cudaz.cuMemFree(d_rec);
    try cudaz.cuMemsetD8(d_rec, 0, @as(usize, @intCast(rec_bytes)) * n_groups * kvh);

    var args: kvk.KvarnStoreArgs = .{
        .current = @ptrFromInt(d_cur),
        .current_v = null,
        .indices = @ptrFromInt(d_idx),
        .stage = @ptrFromInt(d_stage),
        .records = @ptrFromInt(d_rec),
        .n_tokens = @intCast(n),
        .n_record_heads = @intCast(kvh),
        .stream = 0,
        .groups_per_stream = @intCast(n_groups),
        .record_bytes = rec_bytes,
        .k_payload_off = @intCast(layout.k_payload_off),
        .k_s_col_off = @intCast(layout.k_s_col_off),
        .k_zp_off = @intCast(layout.k_zp_off),
        .k_s_row_off = @intCast(layout.k_s_row_off),
        .v_payload_off = @intCast(layout.v_payload_off),
        .v_s_col_off = @intCast(layout.v_s_col_off),
        .v_s_row_off = @intCast(layout.v_s_row_off),
        .v_zp_off = @intCast(layout.v_zp_off),
        .k_bits = 5,
        .v_bits = 4,
        .sinkhorn_iters = 16,
        .stage_groups = 4,
        .tail_groups = 3,
        .swa = 0,
        .eager_records = 1,
    };
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "iso-store v2: lanzando kvh=2 grid=2 (layout real)...\n", .{});
    try kvk.kvarnStoreDevice(kmod, &args, stream);
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "iso-store v2: lanzado; sync...\n", .{});
    try cudaz.cuStreamSynchronize(stream);
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "iso-store v2: COMPLETO SIN HANG\n", .{});

    // === BISECT FASE 2: la SECUENCIA completa del appendTokens ===
    // store + init_descs con los args EXACTOS de refreshDescs.
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "iso2: init_descs tras store (kvh=2)\n", .{});
    const d_descs = try cudaz.cuMemAlloc(2 * kvh * @sizeOf(kvk.KvarnDesc));
    defer cudaz.cuMemFree(d_descs);
    try cudaz.cuMemsetD8(d_descs, 0, 2 * kvh * @sizeOf(kvk.KvarnDesc));
    var ia: kvk.KvarnInitDescsArgs = .{
        .n_stream = 1,
        .n_indices = @intCast(n),
        .d_indices = @ptrFromInt(d_idx),
        .d_descs = @ptrFromInt(d_descs),
        .desc_stride = 2,
        .d_records = @ptrFromInt(d_rec),
        .d_stage = @ptrFromInt(d_stage),
        .n_record_heads = @intCast(kvh),
        .groups_per_stream = @intCast(n_groups),
        .record_bytes = rec_bytes,
        .stage_groups = 4,
        .tail_groups = 3,
        .k_bits = 5,
        .v_bits = 4,
        .head_dim = 128, // BUG A @15a4b5c: campo obligatorio (D=128 aquí)
        .head_slices = 1,
        .eager_records = 1,
        .read_indirect = 0,
        .original_domain = 0,
        .swa = 0,
    };
    try kvk.kvarnInitDescsDevice(kmod, &ia, stream);
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "iso2: sync post-init_descs...\n", .{});
    try cudaz.cuStreamSynchronize(stream);
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "iso2: SIN HANG — leyendo descs...\n", .{});
    const dh = try a.alloc(u8, 2 * kvh * @sizeOf(kvk.KvarnDesc));
    defer a.free(dh);
    try cudaz.cuMemcpyDtoH(@intFromPtr(dh.ptr), d_descs, 2 * kvh * @sizeOf(kvk.KvarnDesc));
    for (0..2 * kvh) |i| {
        const d = @as([*]align(1) const kvk.KvarnDesc, @ptrCast(dh.ptr))[i];
        try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "desc[{d}]: lg={d} lp={d} hb={d} v={d}\n", .{ i, d.live_group, d.live_pos, d.head_base, d.value });
    }
    // Gate: con el fix A3-bis, TODOS los descs deben tener live=(0,127).
    try testing.expectEqual(@as(c_int, 0), @as([*]align(1) const kvk.KvarnDesc, @ptrCast(dh.ptr))[0].live_group);
    try testing.expectEqual(@as(c_int, 127), @as([*]align(1) const kvk.KvarnDesc, @ptrCast(dh.ptr))[0].live_pos);
    try testing.expectEqual(@as(c_int, 127), @as([*]align(1) const kvk.KvarnDesc, @ptrCast(dh.ptr))[3].live_pos);
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "iso2: A3-bis VERIFICADO — descs h1 con live correcto\n", .{});
}
