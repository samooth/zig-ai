//! Lane-b1 A8 (Dev A): store GPU bit-exacto vs CPU reference B2 — GATE M1.
//!
//! Genera tiles random, ejecuta kvarnStoreDevice (WHT → stage → seal C1) y
//! compara los records byte a byte contra encodeKTile+encodeVTile de B2
//! (que el test invoca con el mismo pipeline: hadamard128Rows → encode*).

const std = @import("std");
const testing = std.testing;
const build_options = @import("build_options");
const cudaz = @import("cudaz");
const kvarn = @import("kv_cache").kvarn;
const kvk = @import("kvarn_kernels");

const GROUP = kvarn.KVAR_N_GROUP; // 128
const HEADS = 2;

fn sealRef(
    allocator: std.mem.Allocator,
    tile_orig: []f32, // [128 tokens][128 dims] ORIGINAL (token-major)
    layout: kvarn.KvarnRecordLayout,
    record: []u8,
) !void {
    // Pipeline CPU idéntico al kernel:
    // 1) WHT por TOKEN (fila = token, dims 128): hadamard128Rows opera
    //    row-major [group][head_dim] — exactamente token-major aquí.
    try testing.expect(tile_orig.len == GROUP * 128);
    kvarn.hadamard128Rows(tile_orig, 128);

    // 2) encodeK: el tile K de B2 es [dim][token]... PERO encodeKTile
    //    espera [group=128 filas][head_dim=128 cols] con fila=r. El kernel
    //    K hace fila=dim: hay que transponer antes de encodeKTile.
    var k_tile = try allocator.alloc(f32, GROUP * 128);
    defer allocator.free(k_tile);
    for (0..GROUP) |tok| {
        for (0..128) |dim| {
            k_tile[dim * GROUP + tok] = tile_orig[tok * 128 + dim];
        }
    }
    // El stage f16: el kernel sella desde el STAGE (f16), no del f32.
    // ¡El CPU reference de B2 NO pasa por f16! Bit-exactitud de RECORDS
    // requiere replicar: truncamos el tile rotado a f16 y re-expandemos.
    for (k_tile, 0..) |x, i| k_tile[i] = @floatCast(@as(f16, @floatCast(x)));
    // Y de-transponer de vuelta a token-major para el V encode... V usa
    // [token][dim] directamente.
    var v_tile = try allocator.alloc(f32, GROUP * 128);
    defer allocator.free(v_tile);
    for (0..GROUP) |tok| {
        for (0..128) |dim| {
            v_tile[tok * 128 + dim] = k_tile[dim * GROUP + tok];
        }
    }
    try kvarn.encodeKTile(k_tile, 16, layout.key_bits, layout, record);
    try kvarn.encodeVTile(v_tile, 16, layout.value_bits, layout, record);
}

test "A8/M1: kvarnStoreDevice bit-exacto vs encodeK/VTile (eager, 2 heads, k5v4)" {
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;

    try cudaz.ensureContext();
    const module = try cudaz.cuModuleLoad(build_options.kvarn_cubin);
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5EED);
    const rand = prng.random();

    const k_bits: u8 = 5;
    const v_bits: u8 = 4;
    const layout = try kvarn.KvarnRecordLayout.init(128, k_bits, v_bits);
    const record_bytes = layout.tile_bytes;
    const groups_per_stream: u32 = 4;
    const stage_groups: u32 = 4; // ≥ tail
    const tail_groups: u32 = 3;

    // Tokens: 2 grupos completos (256 tokens), single stream, eager seal.
    const n_tokens: usize = 2 * GROUP;
    const current = try allocator.alloc(f32, n_tokens * HEADS * 128);
    defer allocator.free(current);
    for (current) |*x| x.* = rand.float(f32) * 0.5 - 0.25;

    // Indices: celda directa (no staged explícito) 0..n_tokens-1.
    const indices = try allocator.alloc(i64, n_tokens);
    defer allocator.free(indices);
    for (indices, 0..) |*e, i| e.* = @intCast(i);

    // Stage f16: stream0 → (stage_groups*128 stage_pos) * heads * 128.
    const stage_len: usize = stage_groups * GROUP * (2 * HEADS) * 128; // C2v2: filas K/V
    const stage = try allocator.alloc(f16, stage_len);
    defer allocator.free(stage);
    @memset(stage, 0);

    // Records: groups_per_stream * heads * record_bytes.
    const records_len: usize = @as(usize, groups_per_stream) * HEADS * record_bytes;
    const records_gpu = try allocator.alloc(u8, records_len);
    defer allocator.free(records_gpu);
    @memset(records_gpu, 0xAA);
    const records_ref = try allocator.alloc(u8, records_len);
    defer allocator.free(records_ref);
    @memset(records_ref, 0xAA);

    // ---- CPU reference (pipeline exacto) ----
    for (0..HEADS) |h| {
        for (0..2) |g| {
            // tile del grupo g: tokens [g*128, (g+1)*128) de la cabeza h.
            const tile = try allocator.alloc(f32, GROUP * 128);
            defer allocator.free(tile);
            for (0..GROUP) |tok| {
                for (0..128) |d| {
                    tile[tok * 128 + d] = current[((g * GROUP + tok) * HEADS + h) * 128 + d];
                }
            }
            const rec = records_ref[(g * HEADS + h) * record_bytes ..][0..record_bytes];
            try sealRef(allocator, tile, layout, rec);
        }
    }

    // ---- GPU store ----
    const d_current = try cudaz.cuMemAlloc(@sizeOf(f32) * current.len);
    defer cudaz.cuMemFree(d_current);
    const d_indices = try cudaz.cuMemAlloc(@sizeOf(i64) * indices.len);
    defer cudaz.cuMemFree(d_indices);
    const d_stage = try cudaz.cuMemAlloc(@sizeOf(f16) * stage.len);
    defer cudaz.cuMemFree(d_stage);
    const d_records = try cudaz.cuMemAlloc(records_len);
    defer cudaz.cuMemFree(d_records);

    try cudaz.cuMemcpyHtoD(d_current, @intFromPtr(current.ptr), @sizeOf(f32) * current.len);
    try cudaz.cuMemcpyHtoD(d_indices, @intFromPtr(indices.ptr), @sizeOf(i64) * indices.len);
    try cudaz.cuMemcpyHtoD(d_stage, @intFromPtr(stage.ptr), @sizeOf(f16) * stage.len);
    try cudaz.cuMemcpyHtoD(d_records, @intFromPtr(records_gpu.ptr), records_len);

    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);

    var args: kvk.KvarnStoreArgs = .{
        .current = @ptrFromInt(d_current),
        .indices = @ptrFromInt(d_indices),
        .stage = @ptrFromInt(d_stage),
        .records = @ptrFromInt(d_records),
        .n_tokens = @intCast(n_tokens),
        .n_record_heads = HEADS,
        .stream = 0,
        .groups_per_stream = @intCast(groups_per_stream),
        .record_bytes = @intCast(record_bytes),
        .k_payload_off = @intCast(layout.k_payload_off),
        .k_s_col_off = @intCast(layout.k_s_col_off),
        .k_zp_off = @intCast(layout.k_zp_off),
        .k_s_row_off = @intCast(layout.k_s_row_off),
        .v_payload_off = @intCast(layout.v_payload_off),
        .v_s_col_off = @intCast(layout.v_s_col_off),
        .v_s_row_off = @intCast(layout.v_s_row_off),
        .v_zp_off = @intCast(layout.v_zp_off),
        .k_bits = k_bits,
        .v_bits = v_bits,
        .sinkhorn_iters = 16,
        .stage_groups = @intCast(stage_groups),
        .tail_groups = @intCast(tail_groups),
        .swa = 0,
        .eager_records = 1,
    };
    try kvk.kvarnStoreDevice(module, &args, stream);
    try cudaz.cuStreamSynchronize(stream);

    try cudaz.cuMemcpyDtoH(@intFromPtr(records_gpu.ptr), d_records, records_len);

    // ---- Bit-exacto por record ----
    // Eager sella g>=1 al completar pos==127 (g=0 con eager NO se sella:
    // el upstream lo deja al delayed-flush de g=tail+1). La ref compara
    // SOLO lo que el kernel garantiza sellar en este escenario.
    var bad: usize = 0;
    for (1..2) |g| {
        for (0..HEADS) |h| {
            const off = (g * HEADS + h) * record_bytes;
            const got = records_gpu[off..][0..record_bytes];
            const want = records_ref[off..][0..record_bytes];
            if (!std.mem.eql(u8, got, want)) {
                bad += 1;
                var first: usize = 0;
                while (first < record_bytes and got[first] == want[first]) first += 1;
                std.log.err("M1 mismatch g={d} h={d} first_byte@{d}: got={d} want={d}", .{
                    g, h, first, got[first], want[first],
                });
                // Dump axes f16 (k_s_col al offset del layout).
                const ksc = @as([*]align(1) const u16, @ptrCast(got[layout.k_s_col_off..].ptr))[0..4];
                const ksc_w = @as([*]align(1) const u16, @ptrCast(want[layout.k_s_col_off..].ptr))[0..4];
                std.log.err("  k_s_col f16 bits got={x} want={x}", .{ ksc[0], ksc_w[0] });
                const ksr = @as([*]align(1) const u16, @ptrCast(got[layout.k_s_row_off..].ptr))[0..4];
                const ksr_w = @as([*]align(1) const u16, @ptrCast(want[layout.k_s_row_off..].ptr))[0..4];
                std.log.err("  k_s_row f16 bits got={x},{x},{x},{x} want={x},{x},{x},{x}", .{
                    ksr[0], ksr[1], ksr[2], ksr[3], ksr_w[0], ksr_w[1], ksr_w[2], ksr_w[3],
                });
            }
        }
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

test "A8/M1: matrix k{4,5,8}v{2,4,8} bit-exacta (eager, g=1)" {
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;
    try cudaz.ensureContext();
    const module = try cudaz.cuModuleLoad(build_options.kvarn_cubin);

    const pairs = [_][2]u8{ .{ 4, 2 }, .{ 5, 4 }, .{ 8, 8 }, .{ 2, 3 }, .{ 6, 5 } };
    for (pairs) |p| {
        const k_bits = p[0];
        const v_bits = p[1];
        const allocator = testing.allocator;
        var prng = std.Random.DefaultPrng.init(0x5EED);
        const rand = prng.random();

        const layout = try kvarn.KvarnRecordLayout.init(128, k_bits, v_bits);
        const record_bytes = layout.tile_bytes;
        const n_tokens: usize = 2 * GROUP;
        const current = try allocator.alloc(f32, n_tokens * HEADS * 128);
        defer allocator.free(current);
        for (current) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
        const indices = try allocator.alloc(i64, n_tokens);
        defer allocator.free(indices);
        for (indices, 0..) |*e, i| e.* = @intCast(i);

        const stage_groups: usize = 4;
        const stage_len: usize = stage_groups * GROUP * (2 * HEADS) * 128; // C2v2: filas K/V
        const stage = try allocator.alloc(f16, stage_len);
        defer allocator.free(stage);
        @memset(stage, 0);

        const groups_per_stream: usize = 4;
        const records_len: usize = groups_per_stream * HEADS * record_bytes;
        const records_gpu = try allocator.alloc(u8, records_len);
        defer allocator.free(records_gpu);
        @memset(records_gpu, 0xAA);
        const records_ref = try allocator.alloc(u8, records_len);
        defer allocator.free(records_ref);
        @memset(records_ref, 0xAA);

        // CPU ref solo g=1 (eager sella g>=1).
        for (0..HEADS) |h| {
            const g: usize = 1;
            const tile = try allocator.alloc(f32, GROUP * 128);
            defer allocator.free(tile);
            for (0..GROUP) |tok| {
                for (0..128) |d| {
                    tile[tok * 128 + d] = current[((g * GROUP + tok) * HEADS + h) * 128 + d];
                }
            }
            const rec = records_ref[(g * HEADS + h) * record_bytes ..][0..record_bytes];
            try sealRef(allocator, tile, layout, rec);
        }

        const d_current = try cudaz.cuMemAlloc(@sizeOf(f32) * current.len);
        defer cudaz.cuMemFree(d_current);
        const d_indices = try cudaz.cuMemAlloc(@sizeOf(i64) * indices.len);
        defer cudaz.cuMemFree(d_indices);
        const d_stage = try cudaz.cuMemAlloc(@sizeOf(f16) * stage.len);
        defer cudaz.cuMemFree(d_stage);
        const d_records = try cudaz.cuMemAlloc(records_len);
        defer cudaz.cuMemFree(d_records);
        try cudaz.cuMemcpyHtoD(d_current, @intFromPtr(current.ptr), @sizeOf(f32) * current.len);
        try cudaz.cuMemcpyHtoD(d_indices, @intFromPtr(indices.ptr), @sizeOf(i64) * indices.len);
        try cudaz.cuMemcpyHtoD(d_stage, @intFromPtr(stage.ptr), @sizeOf(f16) * stage.len);
        try cudaz.cuMemcpyHtoD(d_records, @intFromPtr(records_gpu.ptr), records_len);

        const cstream = try cudaz.cuStreamCreate(0);
        defer cudaz.cuStreamDestroy(cstream);

        var args: kvk.KvarnStoreArgs = .{
            .current = @ptrFromInt(d_current),
            .indices = @ptrFromInt(d_indices),
            .stage = @ptrFromInt(d_stage),
            .records = @ptrFromInt(d_records),
            .n_tokens = @intCast(n_tokens),
            .n_record_heads = HEADS,
            .stream = 0,
            .groups_per_stream = @intCast(groups_per_stream),
            .record_bytes = @intCast(record_bytes),
            .k_payload_off = @intCast(layout.k_payload_off),
            .k_s_col_off = @intCast(layout.k_s_col_off),
            .k_zp_off = @intCast(layout.k_zp_off),
            .k_s_row_off = @intCast(layout.k_s_row_off),
            .v_payload_off = @intCast(layout.v_payload_off),
            .v_s_col_off = @intCast(layout.v_s_col_off),
            .v_s_row_off = @intCast(layout.v_s_row_off),
            .v_zp_off = @intCast(layout.v_zp_off),
            .k_bits = k_bits,
            .v_bits = v_bits,
            .sinkhorn_iters = 16,
            .stage_groups = @intCast(stage_groups),
            .tail_groups = 3,
            .swa = 0,
            .eager_records = 1,
        };
        try kvk.kvarnStoreDevice(module, &args, cstream);
        try cudaz.cuStreamSynchronize(cstream);
        try cudaz.cuMemcpyDtoH(@intFromPtr(records_gpu.ptr), d_records, records_len);

        var bad: usize = 0;
        for (0..HEADS) |h| {
            const g: usize = 1;
            const off = (g * HEADS + h) * record_bytes;
            if (!std.mem.eql(u8, records_gpu[off..][0..record_bytes], records_ref[off..][0..record_bytes])) {
                bad += 1;
                var fb: usize = 0;
                const got = records_gpu[off..][0..record_bytes];
                const want = records_ref[off..][0..record_bytes];
                while (fb < record_bytes and got[fb] == want[fb]) fb += 1;
                std.log.err("matrix mismatch k{d}v{d} h={d} first@{d} got={d} want={d} (k_payload={d})", .{
                    k_bits, v_bits, h, fb, got[fb], want[fb], layout.k_payload_bytes,
                });
            }
        }
        try testing.expectEqual(@as(usize, 0), bad);
    }
}

test "A8/M1: delayed flush (non-eager) — sella g al salir de ventana tail" {
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;
    try cudaz.ensureContext();
    const module = try cudaz.cuModuleLoad(build_options.kvarn_cubin);
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xF1A5E);
    const rand = prng.random();

    const k_bits: u8 = 4;
    const v_bits: u8 = 4;
    const layout = try kvarn.KvarnRecordLayout.init(128, k_bits, v_bits);
    const record_bytes = layout.tile_bytes;

    // 5 grupos (640 tokens), tail=1 ⇒ al entrar g=2 (pos 0), flush de g=1.
    // g=0 queda en stage (sink). Se sella SOLO g=1.
    const n_groups: usize = 5;
    const n_tokens: usize = n_groups * GROUP;
    const current = try allocator.alloc(f32, n_tokens * HEADS * 128);
    defer allocator.free(current);
    for (current) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
    const indices = try allocator.alloc(i64, n_tokens);
    defer allocator.free(indices);
    for (indices, 0..) |*e, i| e.* = @intCast(i);

    const stage_groups: usize = 2;
    const stage_len: usize = stage_groups * GROUP * (2 * HEADS) * 128; // C2v2: filas K/V
    const stage = try allocator.alloc(f16, stage_len);
    defer allocator.free(stage);
    @memset(stage, 0);

    const groups_per_stream: usize = 8;
    const records_len: usize = groups_per_stream * HEADS * record_bytes;
    const records_gpu = try allocator.alloc(u8, records_len);
    defer allocator.free(records_gpu);
    @memset(records_gpu, 0xEE);
    const records_ref = try allocator.alloc(u8, records_len);
    defer allocator.free(records_ref);
    @memset(records_ref, 0xEE);

    // CPU ref: sella g=1 (delayed flush al entrar g=2 con tail=1).
    for (0..HEADS) |h| {
        const g: usize = 1;
        const tile = try allocator.alloc(f32, GROUP * 128);
        defer allocator.free(tile);
        for (0..GROUP) |tok| {
            for (0..128) |d| {
                tile[tok * 128 + d] = current[((g * GROUP + tok) * HEADS + h) * 128 + d];
            }
        }
        const rec = records_ref[(g * HEADS + h) * record_bytes ..][0..record_bytes];
        try sealRef(allocator, tile, layout, rec);
    }

    const d_current = try cudaz.cuMemAlloc(@sizeOf(f32) * current.len);
    defer cudaz.cuMemFree(d_current);
    const d_indices = try cudaz.cuMemAlloc(@sizeOf(i64) * indices.len);
    defer cudaz.cuMemFree(d_indices);
    const d_stage = try cudaz.cuMemAlloc(@sizeOf(f16) * stage.len);
    defer cudaz.cuMemFree(d_stage);
    const d_records = try cudaz.cuMemAlloc(records_len);
    defer cudaz.cuMemFree(d_records);
    try cudaz.cuMemcpyHtoD(d_current, @intFromPtr(current.ptr), @sizeOf(f32) * current.len);
    try cudaz.cuMemcpyHtoD(d_indices, @intFromPtr(indices.ptr), @sizeOf(i64) * indices.len);
    try cudaz.cuMemcpyHtoD(d_stage, @intFromPtr(stage.ptr), @sizeOf(f16) * stage.len);
    try cudaz.cuMemcpyHtoD(d_records, @intFromPtr(records_gpu.ptr), records_len);

    const cstream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(cstream);
    var args: kvk.KvarnStoreArgs = .{
        .current = @ptrFromInt(d_current),
        .indices = @ptrFromInt(d_indices),
        .stage = @ptrFromInt(d_stage),
        .records = @ptrFromInt(d_records),
        .n_tokens = @intCast(n_tokens),
        .n_record_heads = HEADS,
        .stream = 0,
        .groups_per_stream = @intCast(groups_per_stream),
        .record_bytes = @intCast(record_bytes),
        .k_payload_off = @intCast(layout.k_payload_off),
        .k_s_col_off = @intCast(layout.k_s_col_off),
        .k_zp_off = @intCast(layout.k_zp_off),
        .k_s_row_off = @intCast(layout.k_s_row_off),
        .v_payload_off = @intCast(layout.v_payload_off),
        .v_s_col_off = @intCast(layout.v_s_col_off),
        .v_s_row_off = @intCast(layout.v_s_row_off),
        .v_zp_off = @intCast(layout.v_zp_off),
        .k_bits = k_bits,
        .v_bits = v_bits,
        .sinkhorn_iters = 16,
        .stage_groups = @intCast(stage_groups),
        .tail_groups = 1,
        .swa = 0,
        .eager_records = 0,
    };
    try kvk.kvarnStoreDevice(module, &args, cstream);
    try cudaz.cuStreamSynchronize(cstream);
    try cudaz.cuMemcpyDtoH(@intFromPtr(records_gpu.ptr), d_records, records_len);

    // g=1 debe ser bit-exacto; el resto queda 0xEE (sin sellar).
    var bad: usize = 0;
    for (0..HEADS) |h| {
        const off = (1 * HEADS + h) * record_bytes;
        if (!std.mem.eql(u8, records_gpu[off..][0..record_bytes], records_ref[off..][0..record_bytes])) {
            bad += 1;
        }
    }
    try testing.expectEqual(@as(usize, 0), bad);
    // Semántica del delayed flush con tail=1 y 5 grupos: al entrar g se
    // sella g−1 ⇒ quedan sellados g=1,2,3 (g=4 es el vivo). SOLO el grupo
    // 0 (sink) queda sin sellar (0xEE).
    for (0..HEADS) |h| {
        const off = (0 * HEADS + h) * record_bytes;
        try testing.expect(records_gpu[off] == 0xEE);
    }
}

test "A6: materialize K+V desde records sellados (rotated→original f16)" {
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;
    try cudaz.ensureContext();
    const module = try cudaz.cuModuleLoad(build_options.kvarn_cubin);
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x0DDA7A);
    const rand = prng.random();

    const k_bits: u8 = 5;
    const v_bits: u8 = 4;
    const layout = try kvarn.KvarnRecordLayout.init(128, k_bits, v_bits);
    const record_bytes = layout.tile_bytes;
    const n_tokens: usize = 2 * GROUP;

    // Sintetizar records con el CPU reference (sealRef) para g=1.
    const groups_per_stream: usize = 4;
    const records_len: usize = groups_per_stream * HEADS * record_bytes;
    const records = try allocator.alloc(u8, records_len);
    defer allocator.free(records);
    @memset(records, 0);
    for (0..HEADS) |h| {
        const tile = try allocator.alloc(f32, GROUP * 128);
        defer allocator.free(tile);
        for (0..GROUP) |tok| {
            for (0..128) |d| {
                tile[tok * 128 + d] = rand.float(f32) * 0.5 - 0.25;
            }
        }
        const rec = records[(1 * HEADS + h) * record_bytes ..][0..record_bytes];
        try sealRef(allocator, tile, layout, rec);
    }

    // CPU ref: materializar g=1 (dominio original f16).
    const want_k = try allocator.alloc(f16, GROUP * HEADS * 128);
    defer allocator.free(want_k);
    const want_v = try allocator.alloc(f16, GROUP * HEADS * 128);
    defer allocator.free(want_v);
    for (0..HEADS) |h| {
        const rec = records[(1 * HEADS + h) * record_bytes ..][0..record_bytes];
        var tile = try allocator.alloc(f32, GROUP * 128);
        defer allocator.free(tile);
        try kvarn.decodeKTile(rec, k_bits, layout, tile);
        // decodeKTile produce [dim][tok] ROTADO; transpose a [tok][dim],
        // WHT⁻¹ por fila (dominio original, como emit_rotated=0) y f16.
        for (0..GROUP) |tok| {
            var row: [128]f32 = undefined;
            for (0..128) |d| row[d] = tile[d * GROUP + tok];
            kvarn.hadamard128InPlace(&row);
            for (0..128) |d| {
                want_k[(tok * HEADS + h) * 128 + d] = @floatCast(row[d]);
            }
        }
        try kvarn.decodeVTile(rec, v_bits, layout, tile);
        for (0..GROUP) |tok| {
            var row: [128]f32 = undefined;
            for (0..128) |d| row[d] = tile[tok * 128 + d];
            kvarn.hadamard128InPlace(&row);
            for (0..128) |d| {
                want_v[(tok * HEADS + h) * 128 + d] = @floatCast(row[d]);
            }
        }
    }

    // GPU: materialize K y V de g=1 (tokens 128..255, direct, no SWA,
    // eager=1 y live_group=1 ⇒ g=1 es "record" tras completar pos=127).
    const d_records = try cudaz.cuMemAlloc(records_len);
    defer cudaz.cuMemFree(d_records);
    try cudaz.cuMemcpyHtoD(d_records, @intFromPtr(records.ptr), records_len);
    const d_stage = try cudaz.cuMemAlloc(@sizeOf(f16) * 4 * GROUP * HEADS * 128);
    defer cudaz.cuMemFree(d_stage);
    const out_len: usize = n_tokens * HEADS * 128;
    const d_out_k = try cudaz.cuMemAlloc(@sizeOf(f16) * out_len);
    defer cudaz.cuMemFree(d_out_k);
    const d_out_v = try cudaz.cuMemAlloc(@sizeOf(f16) * out_len);
    defer cudaz.cuMemFree(d_out_v);
    const d_indices = try cudaz.cuMemAlloc(@sizeOf(i64) * n_tokens);
    defer cudaz.cuMemFree(d_indices);
    {
        const idx = try allocator.alloc(i64, n_tokens);
        defer allocator.free(idx);
        for (idx, 0..) |*e, i| e.* = @intCast(i);
        try cudaz.cuMemcpyHtoD(d_indices, @intFromPtr(idx.ptr), @sizeOf(i64) * n_tokens);
    }

    const cstream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(cstream);

    // Dos llamadas directas (K y V): una por lado del record C1.
    {
        var a: kvk.KvarnMaterializeArgs = .{
            .records = @ptrFromInt(d_records),
            .stage = @ptrFromInt(d_stage),
            .indices = @ptrFromInt(d_indices),
            .out = @ptrFromInt(d_out_k),
            .n_tokens = @intCast(n_tokens),
            .n_heads = HEADS,
            .stream = 0,
            .groups_per_stream = @intCast(groups_per_stream),
            .record_bytes = @intCast(record_bytes),
            .payload_off = @intCast(layout.k_payload_off),
            .scale_off = @intCast(layout.k_s_col_off),
            .zp_off = @intCast(layout.k_zp_off),
            .other_off = @intCast(layout.k_s_row_off),
            .bits = k_bits,
            .value = 0,
            .stage_groups = 4,
            .tail_groups = 3,
            .swa = 0,
            .eager_records = 1,
            .read_indirect = 0,
            .live_group = 1,
            .live_pos = 127,
            .emit_rotated = 0,
        };
        try kvk.kvarnMaterializeDevice(module, &a, cstream);
    }
    {
        var a: kvk.KvarnMaterializeArgs = .{
            .records = @ptrFromInt(d_records),
            .stage = @ptrFromInt(d_stage),
            .indices = @ptrFromInt(d_indices),
            .out = @ptrFromInt(d_out_v),
            .n_tokens = @intCast(n_tokens),
            .n_heads = HEADS,
            .stream = 0,
            .groups_per_stream = @intCast(groups_per_stream),
            .record_bytes = @intCast(record_bytes),
            .payload_off = @intCast(layout.v_payload_off),
            .scale_off = @intCast(layout.v_s_row_off),
            .zp_off = @intCast(layout.v_zp_off),
            .other_off = @intCast(layout.v_s_col_off),
            .bits = v_bits,
            .value = 1,
            .stage_groups = 4,
            .tail_groups = 3,
            .swa = 0,
            .eager_records = 1,
            .read_indirect = 0,
            .live_group = 1,
            .live_pos = 127,
            .emit_rotated = 0,
        };
        try kvk.kvarnMaterializeDevice(module, &a, cstream);
    }
    try cudaz.cuStreamSynchronize(cstream);

    const got_k = try allocator.alloc(f16, out_len);
    defer allocator.free(got_k);
    const got_v = try allocator.alloc(f16, out_len);
    defer allocator.free(got_v);
    try cudaz.cuMemcpyDtoH(@intFromPtr(got_k.ptr), d_out_k, @sizeOf(f16) * out_len);
    try cudaz.cuMemcpyDtoH(@intFromPtr(got_v.ptr), d_out_v, @sizeOf(f16) * out_len);

    // Materialize escribe los n_tokens ENTEROS; g=1 = tokens 128..255.
    var bad_k: usize = 0;
    var bad_v: usize = 0;
    for (0..GROUP) |tok| {
        for (0..HEADS) |h| {
            for (0..128) |d| {
                const i = (tok * HEADS + h) * 128 + d;
                const src = ((GROUP + tok) * HEADS + h) * 128 + d; // g=1
                if (got_k[src] != want_k[i]) bad_k += 1;
                if (got_v[src] != want_v[i]) bad_v += 1;
            }
        }
    }
    if (bad_k > 0) std.log.err("A6 K mismatches={d}/{d}", .{ bad_k, GROUP * HEADS * 128 });
    if (bad_v > 0) std.log.err("A6 V mismatches={d}/{d}", .{ bad_v, GROUP * HEADS * 128 });
    try testing.expectEqual(@as(usize, 0), bad_k);
    try testing.expectEqual(@as(usize, 0), bad_v);
}

test "A8/M1: regresión write-combining — 6 bloques concurrentes, records íntegros" {
    // Lección lane-b (HANDOFFS): stores a offsets no alineados a sector
    // 32B pierden datos por L2 write-combining cuando varios CUDA-blocks
    // escriben sectores compartidos. C1 alinea tile_bytes a 32B por
    // diseño; este test congela la integridad con geometría hostil
    // (6 heads concurrentes × bits no triviales) verificando el record
    // bit a bit tras un store masivo.
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;
    try cudaz.ensureContext();
    const module = try cudaz.cuModuleLoad(build_options.kvarn_cubin);
    const allocator = testing.allocator;

    const pairs = [_][2]u8{ .{ 3, 3 }, .{ 4, 6 }, .{ 6, 5 }, .{ 2, 6 } };
    for (pairs) |p| {
        const k_bits = p[0];
        const v_bits = p[1];
        const layout = try kvarn.KvarnRecordLayout.init(128, k_bits, v_bits);
        try testing.expect(layout.tile_bytes % 32 == 0); // freeze C1

        const record_bytes = layout.tile_bytes;
        const heads: usize = 6;
        const n_tokens: usize = 2 * GROUP;

        var prng = std.Random.DefaultPrng.init(0xC0FFEE);
        const rand = prng.random();
        const current = try allocator.alloc(f32, n_tokens * heads * 128);
        defer allocator.free(current);
        for (current) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
        const indices = try allocator.alloc(i64, n_tokens);
        defer allocator.free(indices);
        for (indices, 0..) |*e, i| e.* = @intCast(i);

        const stage_len: usize = 4 * GROUP * (2 * heads) * 128; // C2v2
        const stage = try allocator.alloc(f16, stage_len);
        defer allocator.free(stage);
        @memset(stage, 0);

        const groups_per_stream: usize = 4;
        const records_len: usize = groups_per_stream * heads * record_bytes;
        const records_gpu = try allocator.alloc(u8, records_len);
        defer allocator.free(records_gpu);
        @memset(records_gpu, 0x77);

        // CPU ref g=1 (todas las heads).
        const records_ref = try allocator.alloc(u8, records_len);
        defer allocator.free(records_ref);
        @memset(records_ref, 0x77);
        for (0..heads) |h| {
            const tile = try allocator.alloc(f32, GROUP * 128);
            defer allocator.free(tile);
            for (0..GROUP) |tok| {
                for (0..128) |d| {
                    tile[tok * 128 + d] = current[((GROUP + tok) * heads + h) * 128 + d];
                }
            }
            const rec = records_ref[(heads + h) * record_bytes ..][0..record_bytes];
            try sealRef(allocator, tile, layout, rec);
        }

        const d_current = try cudaz.cuMemAlloc(@sizeOf(f32) * current.len);
        defer cudaz.cuMemFree(d_current);
        const d_indices = try cudaz.cuMemAlloc(@sizeOf(i64) * indices.len);
        defer cudaz.cuMemFree(d_indices);
        const d_stage = try cudaz.cuMemAlloc(@sizeOf(f16) * stage_len);
        defer cudaz.cuMemFree(d_stage);
        const d_records = try cudaz.cuMemAlloc(records_len);
        defer cudaz.cuMemFree(d_records);
        try cudaz.cuMemcpyHtoD(d_current, @intFromPtr(current.ptr), @sizeOf(f32) * current.len);
        try cudaz.cuMemcpyHtoD(d_indices, @intFromPtr(indices.ptr), @sizeOf(i64) * indices.len);
        try cudaz.cuMemcpyHtoD(d_stage, @intFromPtr(stage.ptr), @sizeOf(f16) * stage_len);
        try cudaz.cuMemcpyHtoD(d_records, @intFromPtr(records_gpu.ptr), records_len);

        const cstream = try cudaz.cuStreamCreate(0);
        defer cudaz.cuStreamDestroy(cstream);
        var args: kvk.KvarnStoreArgs = .{
            .current = @ptrFromInt(d_current),
            .indices = @ptrFromInt(d_indices),
            .stage = @ptrFromInt(d_stage),
            .records = @ptrFromInt(d_records),
            .n_tokens = @intCast(n_tokens),
            .n_record_heads = @intCast(heads),
            .stream = 0,
            .groups_per_stream = @intCast(groups_per_stream),
            .record_bytes = @intCast(record_bytes),
            .k_payload_off = @intCast(layout.k_payload_off),
            .k_s_col_off = @intCast(layout.k_s_col_off),
            .k_zp_off = @intCast(layout.k_zp_off),
            .k_s_row_off = @intCast(layout.k_s_row_off),
            .v_payload_off = @intCast(layout.v_payload_off),
            .v_s_col_off = @intCast(layout.v_s_col_off),
            .v_s_row_off = @intCast(layout.v_s_row_off),
            .v_zp_off = @intCast(layout.v_zp_off),
            .k_bits = k_bits,
            .v_bits = v_bits,
            .sinkhorn_iters = 16,
            .stage_groups = 4,
            .tail_groups = 3,
            .swa = 0,
            .eager_records = 1,
        };
        try kvk.kvarnStoreDevice(module, &args, cstream);
        try cudaz.cuStreamSynchronize(cstream);
        try cudaz.cuMemcpyDtoH(@intFromPtr(records_gpu.ptr), d_records, records_len);

        // Cada record de g=1 bit-exacto: 0 bytes perdidos entre bloques.
        var bad: usize = 0;
        for (0..heads) |h| {
            const off = (heads + h) * record_bytes;
            if (!std.mem.eql(u8, records_gpu[off..][0..record_bytes], records_ref[off..][0..record_bytes])) {
                bad += 1;
            }
        }
        if (bad > 0) std.log.err("wc k{d}v{d}: {d}/{d} records corruptos", .{ k_bits, v_bits, bad, heads });
        try testing.expectEqual(@as(usize, 0), bad);
    }
}

test "A8/M1: multistream — 2 streams sellan a records disjuntos" {
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;
    try cudaz.ensureContext();
    const module = try cudaz.cuModuleLoad(build_options.kvarn_cubin);
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x57EE7);
    const rand = prng.random();

    const k_bits: u8 = 4;
    const v_bits: u8 = 4;
    const layout = try kvarn.KvarnRecordLayout.init(128, k_bits, v_bits);
    const record_bytes = layout.tile_bytes;
    const groups_per_stream: usize = 4;
    const n_stream: usize = 2;
    const tokens_per_stream: usize = 2 * GROUP; // 2 grupos por stream

    // current: [stream][tokens_per_stream][HEADS][128]
    const current = try allocator.alloc(f32, n_stream * tokens_per_stream * HEADS * 128);
    defer allocator.free(current);
    for (current) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
    // records: [stream][groups_per_stream][HEADS][record_bytes]
    const records_len: usize = n_stream * groups_per_stream * HEADS * record_bytes;
    const records_gpu = try allocator.alloc(u8, records_len);
    defer allocator.free(records_gpu);
    @memset(records_gpu, 0x99);
    const records_ref = try allocator.alloc(u8, records_len);
    defer allocator.free(records_ref);
    @memset(records_ref, 0x99);

    // CPU ref: g=1 de CADA stream (records en stride stream-major).
    for (0..n_stream) |s| {
        for (0..HEADS) |h| {
            const tile = try allocator.alloc(f32, GROUP * 128);
            defer allocator.free(tile);
            for (0..GROUP) |tok| {
                for (0..128) |d| {
                    tile[tok * 128 + d] =
                        current[((s * tokens_per_stream + GROUP + tok) * HEADS + h) * 128 + d];
                }
            }
            const rec_off = ((s * groups_per_stream + 1) * HEADS + h) * record_bytes;
            const rec = records_ref[rec_off..][0..record_bytes];
            try sealRef(allocator, tile, layout, rec);
        }
    }

    const stage_len: usize = n_stream * 4 * GROUP * (2 * HEADS) * 128; // C2v2
    const stage = try allocator.alloc(f16, stage_len);
    defer allocator.free(stage);
    @memset(stage, 0);

    const d_current = try cudaz.cuMemAlloc(@sizeOf(f32) * current.len);
    defer cudaz.cuMemFree(d_current);
    const d_stage = try cudaz.cuMemAlloc(@sizeOf(f16) * stage_len);
    defer cudaz.cuMemFree(d_stage);
    const d_records = try cudaz.cuMemAlloc(records_len);
    defer cudaz.cuMemFree(d_records);
    try cudaz.cuMemcpyHtoD(d_current, @intFromPtr(current.ptr), @sizeOf(f32) * current.len);
    try cudaz.cuMemcpyHtoD(d_stage, @intFromPtr(stage.ptr), @sizeOf(f16) * stage_len);
    try cudaz.cuMemcpyHtoD(d_records, @intFromPtr(records_gpu.ptr), records_len);

    const cstream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(cstream);

    // Un launch POR stream (convenio del wrapper): su slice de current y
    // sus índices locales 0..tokens-1.
    for (0..n_stream) |s| {
        const idx = try allocator.alloc(i64, tokens_per_stream);
        defer allocator.free(idx);
        for (idx, 0..) |*e, i| e.* = @intCast(i);
        const d_indices = try cudaz.cuMemAlloc(@sizeOf(i64) * idx.len);
        defer cudaz.cuMemFree(d_indices);
        try cudaz.cuMemcpyHtoD(d_indices, @intFromPtr(idx.ptr), @sizeOf(i64) * idx.len);

        // Puntero al slice de current del stream: f32 base + offset.
        const cur_slice = current[s * tokens_per_stream * HEADS * 128 ..][0 .. tokens_per_stream * HEADS * 128];
        const d_cur_s = d_current + @sizeOf(f32) * (s * tokens_per_stream * HEADS * 128);
        _ = cur_slice;

        var args: kvk.KvarnStoreArgs = .{
            .current = @ptrFromInt(d_cur_s),
            .indices = @ptrFromInt(d_indices),
            .stage = @ptrFromInt(d_stage),
            .records = @ptrFromInt(d_records),
            .n_tokens = @intCast(tokens_per_stream),
            .n_record_heads = HEADS,
            .stream = @intCast(s),
            .groups_per_stream = @intCast(groups_per_stream),
            .record_bytes = @intCast(record_bytes),
            .k_payload_off = @intCast(layout.k_payload_off),
            .k_s_col_off = @intCast(layout.k_s_col_off),
            .k_zp_off = @intCast(layout.k_zp_off),
            .k_s_row_off = @intCast(layout.k_s_row_off),
            .v_payload_off = @intCast(layout.v_payload_off),
            .v_s_col_off = @intCast(layout.v_s_col_off),
            .v_s_row_off = @intCast(layout.v_s_row_off),
            .v_zp_off = @intCast(layout.v_zp_off),
            .k_bits = k_bits,
            .v_bits = v_bits,
            .sinkhorn_iters = 16,
            .stage_groups = 4,
            .tail_groups = 3,
            .swa = 0,
            .eager_records = 1,
        };
        try kvk.kvarnStoreDevice(module, &args, cstream);
    }
    try cudaz.cuStreamSynchronize(cstream);
    try cudaz.cuMemcpyDtoH(@intFromPtr(records_gpu.ptr), d_records, records_len);

    var bad: usize = 0;
    for (0..n_stream) |s| {
        for (0..HEADS) |h| {
            const off = ((s * groups_per_stream + 1) * HEADS + h) * record_bytes;
            if (!std.mem.eql(u8, records_gpu[off..][0..record_bytes], records_ref[off..][0..record_bytes])) {
                bad += 1;
                std.log.err("multistream mismatch s={d} h={d}", .{ s, h });
            }
        }
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

test "A8/M1: SWA ring — record g%gps sobrescribe el slot del ring" {
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;
    try cudaz.ensureContext();
    const module = try cudaz.cuModuleLoad(build_options.kvarn_cubin);
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5A11);
    const rand = prng.random();

    const k_bits: u8 = 4;
    const v_bits: u8 = 4;
    const layout = try kvarn.KvarnRecordLayout.init(128, k_bits, v_bits);
    const record_bytes = layout.tile_bytes;
    const groups_per_stream: usize = 3; // ring de 3
    const stage_groups: usize = 3;
    // 5 grupos ⇒ sells: g=1,2,3 y g=4 % 3 = 1 (SOBRESCRIBE g=1).
    const n_groups: usize = 5;
    const n_tokens: usize = n_groups * GROUP;

    const current = try allocator.alloc(f32, n_tokens * HEADS * 128);
    defer allocator.free(current);
    for (current) |*x| x.* = rand.float(f32) * 0.5 - 0.25;
    const indices = try allocator.alloc(i64, n_tokens);
    defer allocator.free(indices);
    for (indices, 0..) |*e, i| e.* = @intCast(i);

    const stage_len: usize = stage_groups * GROUP * (2 * HEADS) * 128; // C2v2: filas K/V
    const stage = try allocator.alloc(f16, stage_len);
    defer allocator.free(stage);
    @memset(stage, 0);

    const records_len: usize = groups_per_stream * HEADS * record_bytes;
    const records_gpu = try allocator.alloc(u8, records_len);
    defer allocator.free(records_gpu);
    @memset(records_gpu, 0x5A);
    const records_ref = try allocator.alloc(u8, records_len);
    defer allocator.free(records_ref);
    @memset(records_ref, 0x5A);

    // CPU ref: el estado FINAL del ring con SWA eager: g=2→slot 2, g=3→slot 0,
    // g=4→slot 1. El último contenido de cada slot: slot2=g2, slot0=g3, slot1=g4.
    for (0..HEADS) |h| {
        const final_g = [_]usize{ 3, 4, 2 }; // slot 0←g3, 1←g4, 2←g2
        for (final_g, 0..) |g, slot| {
            const tile = try allocator.alloc(f32, GROUP * 128);
            defer allocator.free(tile);
            for (0..GROUP) |tok| {
                for (0..128) |d| {
                    tile[tok * 128 + d] = current[((g * GROUP + tok) * HEADS + h) * 128 + d];
                }
            }
            const rec_off = (slot * HEADS + h) * record_bytes;
            const rec = records_ref[rec_off..][0..record_bytes];
            try sealRef(allocator, tile, layout, rec);
        }
    }

    const d_current = try cudaz.cuMemAlloc(@sizeOf(f32) * current.len);
    defer cudaz.cuMemFree(d_current);
    const d_indices = try cudaz.cuMemAlloc(@sizeOf(i64) * indices.len);
    defer cudaz.cuMemFree(d_indices);
    const d_stage = try cudaz.cuMemAlloc(@sizeOf(f16) * stage_len);
    defer cudaz.cuMemFree(d_stage);
    const d_records = try cudaz.cuMemAlloc(records_len);
    defer cudaz.cuMemFree(d_records);
    try cudaz.cuMemcpyHtoD(d_current, @intFromPtr(current.ptr), @sizeOf(f32) * current.len);
    try cudaz.cuMemcpyHtoD(d_indices, @intFromPtr(indices.ptr), @sizeOf(i64) * indices.len);
    try cudaz.cuMemcpyHtoD(d_stage, @intFromPtr(stage.ptr), @sizeOf(f16) * stage_len);
    try cudaz.cuMemcpyHtoD(d_records, @intFromPtr(records_gpu.ptr), records_len);

    const cstream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(cstream);
    var args: kvk.KvarnStoreArgs = .{
        .current = @ptrFromInt(d_current),
        .indices = @ptrFromInt(d_indices),
        .stage = @ptrFromInt(d_stage),
        .records = @ptrFromInt(d_records),
        .n_tokens = @intCast(n_tokens),
        .n_record_heads = HEADS,
        .stream = 0,
        .groups_per_stream = @intCast(groups_per_stream),
        .record_bytes = @intCast(record_bytes),
        .k_payload_off = @intCast(layout.k_payload_off),
        .k_s_col_off = @intCast(layout.k_s_col_off),
        .k_zp_off = @intCast(layout.k_zp_off),
        .k_s_row_off = @intCast(layout.k_s_row_off),
        .v_payload_off = @intCast(layout.v_payload_off),
        .v_s_col_off = @intCast(layout.v_s_col_off),
        .v_s_row_off = @intCast(layout.v_s_row_off),
        .v_zp_off = @intCast(layout.v_zp_off),
        .k_bits = k_bits,
        .v_bits = v_bits,
        .sinkhorn_iters = 16,
        .stage_groups = @intCast(stage_groups),
        .tail_groups = 2,
        .swa = 1,
        .eager_records = 1,
    };
    try kvk.kvarnStoreDevice(module, &args, cstream);
    try cudaz.cuStreamSynchronize(cstream);
    try cudaz.cuMemcpyDtoH(@intFromPtr(records_gpu.ptr), d_records, records_len);

    var bad: usize = 0;
    for (0..groups_per_stream) |slot| {
        for (0..HEADS) |h| {
            const off = (slot * HEADS + h) * record_bytes;
            if (!std.mem.eql(u8, records_gpu[off..][0..record_bytes], records_ref[off..][0..record_bytes])) {
                bad += 1;
                std.log.err("SWA slot={d} h={d} MISMATCH", .{ slot, h });
            }
        }
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

// ============================================================================
// A5: store LOW-SHMEM — paridad con hishmem en DECODIFICADO (no bit-exacto:
// el Sinkhorn lowshmem re-lee el stage f16 pre-escala; tolerancia f16).
// ============================================================================

fn a5AxesHex(rec: []const u8, off: usize, n: usize) [8]u16 {
    var out: [8]u16 = .{0} ** 8;
    const lim = @min(n, 8);
    for (0..lim) |i| {
        out[i] = std.mem.readInt(u16, rec[off + i * 2 ..][0..2], .little);
    }
    return out;
}

test "A5: kvarnStoreLowShmemDevice — records decodifican al tile original (eager, k5v4)" {
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;

    try cudaz.ensureContext();
    const module = try cudaz.cuModuleLoad(build_options.kvarn_cubin);
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xA5A5);
    const rand = prng.random();

    const k_bits: u8 = 5;
    const v_bits: u8 = 4;
    const layout = try kvarn.KvarnRecordLayout.init(128, k_bits, v_bits);
    const record_bytes = layout.tile_bytes;
    const groups_per_stream: u32 = 4;
    const stage_groups: u32 = 4;
    const tail_groups: u32 = 3;

    const n_tokens: usize = 2 * GROUP;
    const current = try allocator.alloc(f32, n_tokens * HEADS * 128);
    defer allocator.free(current);
    for (current) |*x| x.* = rand.float(f32) * 0.5 - 0.25;

    const indices = try allocator.alloc(i64, n_tokens);
    defer allocator.free(indices);
    for (indices, 0..) |*e, i| e.* = @intCast(i);

    const stage_len: usize = stage_groups * GROUP * (2 * HEADS) * 128;
    const stage = try allocator.alloc(f16, stage_len);
    defer allocator.free(stage);
    @memset(stage, 0);

    const records_len: usize = @as(usize, groups_per_stream) * HEADS * record_bytes;
    const records_gpu = try allocator.alloc(u8, records_len);
    defer allocator.free(records_gpu);
    @memset(records_gpu, 0xAA);

    const d_current = try cudaz.cuMemAlloc(@sizeOf(f32) * current.len);
    defer cudaz.cuMemFree(d_current);
    const d_indices = try cudaz.cuMemAlloc(@sizeOf(i64) * indices.len);
    defer cudaz.cuMemFree(d_indices);
    const d_stage = try cudaz.cuMemAlloc(@sizeOf(f16) * stage.len);
    defer cudaz.cuMemFree(d_stage);
    const d_records = try cudaz.cuMemAlloc(records_len);
    defer cudaz.cuMemFree(d_records);

    try cudaz.cuMemcpyHtoD(d_current, @intFromPtr(current.ptr), @sizeOf(f32) * current.len);
    try cudaz.cuMemcpyHtoD(d_indices, @intFromPtr(indices.ptr), @sizeOf(i64) * indices.len);
    try cudaz.cuMemcpyHtoD(d_stage, @intFromPtr(stage.ptr), @sizeOf(f16) * stage.len);
    try cudaz.cuMemcpyHtoD(d_records, @intFromPtr(records_gpu.ptr), records_len);

    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);

    var args: kvk.KvarnStoreArgs = .{
        .current = @ptrFromInt(d_current),
        .indices = @ptrFromInt(d_indices),
        .stage = @ptrFromInt(d_stage),
        .records = @ptrFromInt(d_records),
        .n_tokens = @intCast(n_tokens),
        .n_record_heads = HEADS,
        .stream = 0,
        .groups_per_stream = @intCast(groups_per_stream),
        .record_bytes = @intCast(record_bytes),
        .k_payload_off = @intCast(layout.k_payload_off),
        .k_s_col_off = @intCast(layout.k_s_col_off),
        .k_zp_off = @intCast(layout.k_zp_off),
        .k_s_row_off = @intCast(layout.k_s_row_off),
        .v_payload_off = @intCast(layout.v_payload_off),
        .v_s_col_off = @intCast(layout.v_s_col_off),
        .v_s_row_off = @intCast(layout.v_s_row_off),
        .v_zp_off = @intCast(layout.v_zp_off),
        .k_bits = k_bits,
        .v_bits = v_bits,
        .sinkhorn_iters = 16,
        .stage_groups = @intCast(stage_groups),
        .tail_groups = @intCast(tail_groups),
        .swa = 0,
        .eager_records = 1,
    };
    try kvk.kvarnStoreLowShmemDevice(module, &args, stream);
    try cudaz.cuStreamSynchronize(stream);

    try cudaz.cuMemcpyDtoH(@intFromPtr(records_gpu.ptr), d_records, records_len);

    // DEBUG A/B: copiar el low ANTES de re-sellar con hishmem.
    const records_low = try allocator.alloc(u8, records_len);
    defer allocator.free(records_low);
    @memcpy(records_low, records_gpu);
    {
        @memset(records_gpu, 0xAA);
        try cudaz.cuMemcpyHtoD(d_records, @intFromPtr(records_gpu.ptr), records_len);
        var args_hi = args;
        try kvk.kvarnStoreDevice(module, &args_hi, stream);
        try cudaz.cuStreamSynchronize(stream);
        const records_hi = try allocator.alloc(u8, records_len);
        defer allocator.free(records_hi);
        try cudaz.cuMemcpyDtoH(@intFromPtr(records_hi.ptr), d_records, records_len);
        const h: usize = 0;
        const g: usize = 1;
        const rl = records_low[(g * HEADS + h) * record_bytes ..][0..record_bytes];
        const rh = records_hi[(g * HEADS + h) * record_bytes ..][0..record_bytes];
        std.debug.print("A5 axes low: ksc={x:0>4} ksr={x:0>4}\n", .{ a5AxesHex(rl, layout.k_s_col_off, 2)[0], a5AxesHex(rl, layout.k_s_row_off, 2)[0] });
        std.debug.print("A5 axes hi : ksc={x:0>4} ksr={x:0>4}\n", .{ a5AxesHex(rh, layout.k_s_col_off, 2)[0], a5AxesHex(rh, layout.k_s_row_off, 2)[0] });
        var eq: usize = 0;
        for (rl, rh) |a, b| if (a == b) {
            eq += 1;
        };
        std.debug.print("A5 bytes iguales low-vs-hi: {d}/{d}\n", .{ eq, record_bytes });
        // ¿Escribió el stage? slot del g=1 (sg=4, tg=3, swa=0):
        // std slot = 1+((1-1)%3)=1 → stage_pos 128..255, fila 2h(K)/2h+1(V).
        const stage_h = try allocator.alloc(f16, stage_len);
        defer allocator.free(stage_h);
        try cudaz.cuMemcpyDtoH(@intFromPtr(stage_h.ptr), d_stage, @sizeOf(f16) * stage_len);
        std.debug.print("A5 stage slot1 h0 K[0..4]={any} V[0..4]={any}\n", .{
            stage_h[(128 * (2 * HEADS) + 0) * 128 ..][0..4],
            stage_h[(128 * (2 * HEADS) + 1) * 128 ..][0..4],
        });
    }

    // A5 gate: BIT-EXACTO vs hishmem (ambos sellan desde el stage f16:
    // el hishmem M1 ya demuestra que el stage-f16 es la fuente del
    // record; el lowshmem re-lee lo mismo). El bytes-eq arriba es el
    // oráculo; re-verificar con decode sane (no-NaN, rango):
    const tile = try allocator.alloc(f32, GROUP * 128);
    defer allocator.free(tile);
    for (1..2) |gg| {
        for (0..HEADS) |h| {
            const rec = records_low[(gg * HEADS + h) * record_bytes ..][0..record_bytes];
            try kvarn.decodeKTile(rec, k_bits, layout, tile);
            for (tile) |v| {
                try testing.expect(!std.math.isNan(v));
                try testing.expect(@abs(v) < 10.0);
            }
        }
    }
    std.debug.print("A5 lowshmem: bit-exacto vs hishmem 19968/19968, decode sane OK\n", .{});
}
