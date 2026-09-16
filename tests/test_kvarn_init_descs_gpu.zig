//! Lane-b1 B3 (Dev-B): test GPU `kvarnInitDescsDevice`.
//!
//! Verifica que el kernel init rellena los `KvarnDesc` correctamente
//! para K y V, computa `live_group/live_pos` consistente con un
//! harness CPU de referencia, y propaga todos los campos del `A0`.
//!
//! Plan de validación (3 niveles):
//!   1. Smoke: launch con índices vacíos (-1) ⇒ live_group=-1, live_pos=-1
//!   2. Secuencial: índices 0..N-1 ⇒ live_group = (N-1)/128, live_pos = (N-1)%128
//!   3. Sparse: mezcla de positivos, -1 (skip) y staged (-2) ⇒ max
//!      lexicográfico del campo cell de los staged/no-skip
//!
//! Comparación CPU: replicar el árbol de reducción en Zig (1 thread ⇒ 4
//! warps ⇒ 1 final), usando exactamente el mismo criterio de max
//! lexicográfico (group, pos) sobre el cell decodificado.

const std = @import("std");
const testing = std.testing;
const build_options = @import("build_options");
const cudaz = @import("cudaz");
const kvarn_k = @import("kvarn_kernels");
const kvarn = @import("kv_cache").kvarn;

const N_STREAMS: u32 = 4;
const N_INDICES: u32 = 512; // 4 grupos de 128 por stream

fn cpuLive(indices: []const i64) struct { group: c_int, pos: c_int } {
    var max_group: c_int = -1;
    var max_pos: c_int = -1;
    for (indices) |enc| {
        if (enc == -1) continue;
        // Decode payload (mismo criterio que kvarn_reduce_max_group_pos).
        const payload: u64 = if (enc < -1) @intCast(@as(i64, -(enc + 2))) else @intCast(enc);
        const cell: u64 = payload & 0xFFFFFFFF;
        if (cell >= (@as(u64, 1) << 63)) continue; // negativos ⇒ skip (no deberían)
        const g: c_int = @intCast(cell / 128);
        const p: c_int = @intCast(cell % 128);
        if (g > max_group or (g == max_group and p > max_pos)) {
            max_group = g;
            max_pos = p;
        }
    }
    return .{ .group = max_group, .pos = max_pos };
}

fn allocIndices(allocator: std.mem.Allocator, n: u32, fill: fn (usize) i64) ![]i64 {
    const out = try allocator.alloc(i64, n);
    for (out, 0..) |*v, i| v.* = fill(i);
    return out;
}

test "B3 init_descs: smoke con índices vacíos (-1) ⇒ live_group=-1" {
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;

    try cudaz.ensureContext();
    const module = try cudaz.cuModuleLoad(build_options.kvarn_cubin);

    const allocator = testing.allocator;
    const indices = try allocIndices(allocator, N_STREAMS * N_INDICES, struct {
        fn f(_: usize) i64 {
            return -1;
        }
    }.f);
    defer allocator.free(indices);

    // Referencia CPU.
    for (0..N_STREAMS) |s| {
        const ref = cpuLive(indices[s * N_INDICES ..][0..N_INDICES]);
        try testing.expectEqual(@as(c_int, -1), ref.group);
        try testing.expectEqual(@as(c_int, -1), ref.pos);
    }

    // GPU: upload, launch, download, compare.
    const d_indices = try cudaz.cuMemAlloc(@sizeOf(i64) * indices.len);
    defer cudaz.cuMemFree(d_indices);
    try cudaz.cuMemcpyHtoD(d_indices, @intFromPtr(indices.ptr), @sizeOf(i64) * indices.len);

    // Layout POR-LADO (9.4 lane-b): [K(h0..h7), V(h0..h7)] por stream —
    // descs[stream*2*n_heads + lado*n_heads + h]. El intercalado A12
    // quedó obsoleto en 9.4 (desc_stride=2 por-lado para el portable).
    const n_heads: usize = 8;
    const n_descs: usize = N_STREAMS * 2 * n_heads;
    const d_descs = try cudaz.cuMemAlloc(@sizeOf(kvarn_k.KvarnDesc) * n_descs);
    defer cudaz.cuMemFree(d_descs);

    // Init cuMemsetD8 a 0xFE (patrón de byte no-cero en campos c_int)
    // para que el device claramente escriba algo distinto.
    try cudaz.cuMemsetD8(d_descs, 0xFE, @sizeOf(kvarn_k.KvarnDesc) * n_descs);

    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);

    var args: kvarn_k.KvarnInitDescsArgs = .{
        .n_stream = @intCast(N_STREAMS),
        .n_indices = @intCast(N_INDICES),
        .d_indices = @ptrFromInt(d_indices),
        .d_descs = @ptrFromInt(d_descs),
        .desc_stride = 2,
        .d_records = @ptrFromInt(0x1000),
        .d_stage = @ptrFromInt(0x2000),
        .n_record_heads = 8,
        .head_dim = 128,
        .groups_per_stream = 16,
        .record_bytes = 19968, // k5v4 hd128
        .stage_groups = 4,
        .tail_groups = 3,
        .k_bits = 5,
        .v_bits = 4,
        .head_slices = 1,
        .eager_records = 0,
        .read_indirect = 1,
        .original_domain = 0,
        .swa = 0,
    };
    try kvarn_k.kvarnInitDescsDevice(module, &args, stream);
    try cudaz.cuStreamSynchronize(stream);

    const out_host = try allocator.alloc(kvarn_k.KvarnDesc, n_descs);
    defer allocator.free(out_host);
    try cudaz.cuMemcpyDtoH(@intFromPtr(out_host.ptr), d_descs, @sizeOf(kvarn_k.KvarnDesc) * n_descs);

    // Verifica: live_group=-1, live_pos=-1, value=0/1, bits K=5 V=4,
    // head_base=kv_head (cada head físico tiene su par).
    for (0..N_STREAMS) |s| {
        for (0..n_heads) |h| {
            for ([_]u8{ 0, 1 }) |side| {
                const d = out_host[s * 2 * n_heads + @as(usize, side) * n_heads + h];
                try testing.expectEqual(@as(c_int, -1), d.live_group);
                try testing.expectEqual(@as(c_int, -1), d.live_pos);
                try testing.expectEqual(@as(c_int, @intCast(s)), d.stream);
                try testing.expectEqual(@as(c_int, if (side == 0) 5 else 4), d.bits);
                try testing.expectEqual(@as(c_int, @intCast(side)), d.value);
                try testing.expectEqual(@as(c_int, @intCast(h)), d.head_base);
                try testing.expectEqual(@as(c_int, 8), d.n_record_heads);
                try testing.expectEqual(@as(c_int, 16), d.groups_per_stream);
                try testing.expectEqual(@as(c_int, 19968), d.record_bytes);
                try testing.expectEqual(@as(c_int, 4), d.stage_groups);
                try testing.expectEqual(@as(c_int, 3), d.tail_groups);
            }
        }
    }
}

test "B3 init_descs: secuencial 0..N-1 ⇒ live_group=(N-1)/128" {
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;

    try cudaz.ensureContext();
    const module = try cudaz.cuModuleLoad(build_options.kvarn_cubin);

    const allocator = testing.allocator;
    const indices = try allocIndices(allocator, N_STREAMS * N_INDICES, struct {
        fn f(i: usize) i64 {
            return @intCast(@mod(i, N_INDICES));
        }
    }.f);
    defer allocator.free(indices);

    // CPU ref.
    for (0..N_STREAMS) |s| {
        const ref = cpuLive(indices[s * N_INDICES ..][0..N_INDICES]);
        const expected_g: c_int = @intCast((N_INDICES - 1) / 128);
        const expected_p: c_int = @intCast((N_INDICES - 1) % 128);
        try testing.expectEqual(expected_g, ref.group);
        try testing.expectEqual(expected_p, ref.pos);
    }

    const d_indices = try cudaz.cuMemAlloc(@sizeOf(i64) * indices.len);
    defer cudaz.cuMemFree(d_indices);
    try cudaz.cuMemcpyHtoD(d_indices, @intFromPtr(indices.ptr), @sizeOf(i64) * indices.len);

    const n_heads: usize = 4;
    const n_descs: usize = N_STREAMS * 2 * n_heads;
    const d_descs = try cudaz.cuMemAlloc(@sizeOf(kvarn_k.KvarnDesc) * n_descs);
    defer cudaz.cuMemFree(d_descs);
    try cudaz.cuMemsetD8(d_descs, 0xAA, @sizeOf(kvarn_k.KvarnDesc) * n_descs);

    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);

    var args: kvarn_k.KvarnInitDescsArgs = .{
        .n_stream = @intCast(N_STREAMS),
        .n_indices = @intCast(N_INDICES),
        .d_indices = @ptrFromInt(d_indices),
        .d_descs = @ptrFromInt(d_descs),
        .desc_stride = 2,
        .d_records = @ptrFromInt(0x1000),
        .d_stage = @ptrFromInt(0x2000),
        .n_record_heads = 4,
        .groups_per_stream = 4,
        .record_bytes = 4352, // k4v4 hd128
        .stage_groups = 2,
        .tail_groups = 1,
        .k_bits = 4,
        .v_bits = 4,
        .head_dim = 128,
        .head_slices = 1,
        .eager_records = 1,
        .read_indirect = 1,
        .original_domain = 0,
        .swa = 0,
    };
    try kvarn_k.kvarnInitDescsDevice(module, &args, stream);
    try cudaz.cuStreamSynchronize(stream);

    const out_host = try allocator.alloc(kvarn_k.KvarnDesc, n_descs);
    defer allocator.free(out_host);
    try cudaz.cuMemcpyDtoH(@intFromPtr(out_host.ptr), d_descs, @sizeOf(kvarn_k.KvarnDesc) * n_descs);

    const expected_g: c_int = @intCast((N_INDICES - 1) / 128);
    const expected_p: c_int = @intCast((N_INDICES - 1) % 128);
    for (0..N_STREAMS) |s| {
        for ([_]u8{ 0, 1 }) |side| {
            const d = out_host[s * 2 * n_heads + side]; // h=0: par (K0,V0)
            try testing.expectEqual(expected_g, d.live_group);
            try testing.expectEqual(expected_p, d.live_pos);
            try testing.expectEqual(@as(c_int, 1), d.eager_records);
        }
    }
}

test "B3 init_descs: sparse con -1 (skip) y staged (-2) ⇒ max del cell" {
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;

    try cudaz.ensureContext();
    const module = try cudaz.cuModuleLoad(build_options.kvarn_cubin);

    const allocator = testing.allocator;
    // Stream 0: 100, 200, 300, -1, -1, ... (max = 300 = grp 2 pos 44)
    // Stream 1: 127 (max group 0, max pos 127)
    // Stream 2: staged de cell 1024 (grp 8 pos 0)
    // Stream 3: vacío (-1)
    var fill_buf = try allocator.alloc(i64, N_STREAMS * N_INDICES);
    defer allocator.free(fill_buf);
    for (0..N_STREAMS) |s| {
        const base = s * N_INDICES;
        for (0..N_INDICES) |i| {
            fill_buf[base + i] = -1;
        }
    }
    fill_buf[0] = 100;
    fill_buf[1] = 200;
    fill_buf[2] = 300;
    fill_buf[N_INDICES] = 127;
    // Stream 2: staged con cell 1024 = grp 8 pos 0
    // enc = -(((slot+1)<<32) | cell) - 2 ⇒ payload = ((slot+1)<<32) | 1024
    // con slot=0: payload = (1<<32) | 1024 = 0x1_00000400
    // enc = -(payload) - 2 = -0x1_00000400 - 2 = -4294968322 - 2 = -4294968324
    // Para quepa en i64: cabe perfectamente.
    const slot1: u64 = 1; // slot+1
    const cell: u64 = 1024;
    const payload: u64 = (slot1 << 32) | cell;
    const staged_enc: i64 = -@as(i64, @intCast(payload)) - 2;
    fill_buf[2 * N_INDICES] = staged_enc;
    // Stream 3: vacío (queda en -1)

    // CPU ref.
    const ref0 = cpuLive(fill_buf[0 * N_INDICES ..][0..N_INDICES]);
    try testing.expectEqual(@as(c_int, 2), ref0.group);
    try testing.expectEqual(@as(c_int, 44), ref0.pos); // 300 / 128 = 2, 300 % 128 = 44
    const ref1 = cpuLive(fill_buf[1 * N_INDICES ..][0..N_INDICES]);
    try testing.expectEqual(@as(c_int, 0), ref1.group);
    try testing.expectEqual(@as(c_int, 127), ref1.pos);
    const ref2 = cpuLive(fill_buf[2 * N_INDICES ..][0..N_INDICES]);
    try testing.expectEqual(@as(c_int, 8), ref2.group);
    try testing.expectEqual(@as(c_int, 0), ref2.pos);
    const ref3 = cpuLive(fill_buf[3 * N_INDICES ..][0..N_INDICES]);
    try testing.expectEqual(@as(c_int, -1), ref3.group);
    try testing.expectEqual(@as(c_int, -1), ref3.pos);

    const d_indices = try cudaz.cuMemAlloc(@sizeOf(i64) * fill_buf.len);
    defer cudaz.cuMemFree(d_indices);
    try cudaz.cuMemcpyHtoD(d_indices, @intFromPtr(fill_buf.ptr), @sizeOf(i64) * fill_buf.len);

    const n_heads: usize = 2;
    const n_descs: usize = N_STREAMS * 2 * n_heads;
    const d_descs = try cudaz.cuMemAlloc(@sizeOf(kvarn_k.KvarnDesc) * n_descs);
    defer cudaz.cuMemFree(d_descs);

    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);

    var args: kvarn_k.KvarnInitDescsArgs = .{
        .n_stream = @intCast(N_STREAMS),
        .n_indices = @intCast(N_INDICES),
        .d_indices = @ptrFromInt(d_indices),
        .d_descs = @ptrFromInt(d_descs),
        .desc_stride = 2,
        .d_records = @ptrFromInt(0x1000),
        .d_stage = @ptrFromInt(0x2000),
        .n_record_heads = 2,
        .groups_per_stream = 16,
        .record_bytes = 4864, // k2v2 hd128
        .stage_groups = 4,
        .tail_groups = 3,
        .k_bits = 2,
        .v_bits = 2,
        .head_dim = 128,
        .head_slices = 1,
        .eager_records = 0,
        .read_indirect = 1,
        .original_domain = 1,
        .swa = 0,
    };
    try kvarn_k.kvarnInitDescsDevice(module, &args, stream);
    try cudaz.cuStreamSynchronize(stream);

    const out_host = try allocator.alloc(kvarn_k.KvarnDesc, n_descs);
    defer allocator.free(out_host);
    try cudaz.cuMemcpyDtoH(@intFromPtr(out_host.ptr), d_descs, @sizeOf(kvarn_k.KvarnDesc) * n_descs);

    // Layout POR-LADO (9.4 lane-b): [K(h0,h1), V(h0,h1)] por stream —
    // head 0 del par K en s*2*n_heads+0, V en s*2*n_heads+n_heads+0.
    // (h=0 aquí; los heads 1 comparten el mismo live por stream.)
    // Stream 0: grp 2 pos 44
    try testing.expectEqual(@as(c_int, 2), out_host[0].live_group);
    try testing.expectEqual(@as(c_int, 44), out_host[0].live_pos);
    try testing.expectEqual(@as(c_int, 2), out_host[1].live_group);
    try testing.expectEqual(@as(c_int, 44), out_host[1].live_pos);
    // Stream 1: grp 0 pos 127
    try testing.expectEqual(@as(c_int, 0), out_host[1 * 2 * n_heads].live_group);
    try testing.expectEqual(@as(c_int, 127), out_host[1 * 2 * n_heads].live_pos);
    // Stream 2: staged cell 1024 ⇒ grp 8 pos 0
    try testing.expectEqual(@as(c_int, 8), out_host[2 * 2 * n_heads].live_group);
    try testing.expectEqual(@as(c_int, 0), out_host[2 * 2 * n_heads].live_pos);
    // Stream 3: vacío ⇒ -1, -1
    try testing.expectEqual(@as(c_int, -1), out_host[3 * 2 * n_heads].live_group);
    try testing.expectEqual(@as(c_int, -1), out_host[3 * 2 * n_heads].live_pos);
    // K / V diferenciados en value + bits (K h0 y V h0 del stream)
    for (0..N_STREAMS) |s| {
        try testing.expectEqual(@as(c_int, 0), out_host[s * 2 * n_heads].value);
        try testing.expectEqual(@as(c_int, 2), out_host[s * 2 * n_heads].bits);
        try testing.expectEqual(@as(c_int, 1), out_host[s * 2 * n_heads + n_heads].value);
        try testing.expectEqual(@as(c_int, 2), out_host[s * 2 * n_heads + n_heads].bits);
        try testing.expectEqual(@as(c_int, 1), out_host[s * 2 * n_heads].original_domain);
    }
}

test "B3 init_descs: pre-condiciones ⇒ errores limpios" {
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;
    try cudaz.ensureContext();
    const module = try cudaz.cuModuleLoad(build_options.kvarn_cubin);
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);

    var args: kvarn_k.KvarnInitDescsArgs = .{
        .n_stream = 0, // inválido
        .n_indices = 100,
        .d_descs = @ptrFromInt(0x1000),
        .desc_stride = 2,
        .head_dim = 128,
    };
    try testing.expectError(error.InvalidStreamCount, kvarn_k.kvarnInitDescsDevice(module, &args, stream));

    args.n_stream = 2;
    args.n_indices = 0; // inválido
    try testing.expectError(error.InvalidIndexCount, kvarn_k.kvarnInitDescsDevice(module, &args, stream));

    args.n_indices = 100;
    args.desc_stride = 0; // inválido
    try testing.expectError(error.InvalidDescStride, kvarn_k.kvarnInitDescsDevice(module, &args, stream));
}
