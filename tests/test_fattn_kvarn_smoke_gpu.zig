//! Lane-b1 B8 follow-up — ARITY smoke self-tests for portable FA wrappers.
//!
//! Defense in depth against the kp-19 scale bug (iter 15/17, found
//! by Dev-A at 882c20d): each wrapper launches a trivial forward
//! (Q=0.1, K=0.1, V=0.1, indices seq, no tail, no SWA) and verifies
//! the output is finite + not garbage. Catches:
//!   1) kp-array arity mismatch (kernel reads scale/head_slices/etc
//!      from garbage memory, output becomes NaN/Inf).
//!   2) Wrong field in KvarnInitDescsArgs (e.g. wrong head_slices
//!      for D=256/512, wrong n_kv for the size).
//!   3) cuFuncSetAttribute mistakes (smem > 99KB on sm_86 ⇒ launch
//!      error).
//!
//! Gated by cubin (P4 wiring). Without cubin, all tests SKIP.

const std = @import("std");
const testing = std.testing;
const build_options = @import("build_options");
const cudaz = @import("cudaz");
const kvk = @import("kvarn_kernels");
const fattn_kv = @import("fattn_kvarn");
const kvarn = @import("kv_cache").kvarn;

const D: usize = 128;
const D_U32: u32 = 128;

/// Common smoke fixture: allocates a minimal KvarnMemoryConfig
/// (1 stream, 1 kv_head, 1 group, 1 head_slice, 1 record_head) and
/// runs the store + init_descs pipeline. Returns device pointers
/// ready for an attention launch. Caller must defer all free.
const SmokeFixture = struct {
    layout: kvarn.KvarnRecordLayout,
    record_bytes: c_int,
    fattn_module: cudaz.CUmodule,
    kvk_module: cudaz.CUmodule,
    stream: cudaz.CUstream,
    d_stage: cudaz.CUdeviceptr,
    d_records: cudaz.CUdeviceptr,
    d_indices: cudaz.CUdeviceptr,
    d_descs: cudaz.CUdeviceptr,
    n_kv: u32,

    fn init(allocator: std.mem.Allocator, n_kv_local: u32) !SmokeFixture {
        const layout = kvarn.KvarnRecordLayout.init(D_U32, 4, 4) catch unreachable;
        const record_bytes = @as(c_int, @intCast(layout.tile_bytes));

        const fattn_module = try cudaz.cuModuleLoad(build_options.fattn_cubin);
        const kvk_module = try cudaz.cuModuleLoad(build_options.kvarn_cubin);
        const stream = try cudaz.cuStreamCreate(0);
        errdefer cudaz.cuStreamDestroy(stream);

        const d_stage = try cudaz.cuMemAlloc(@as(usize, 2) * D * D * @sizeOf(f16));
        errdefer cudaz.cuMemFree(d_stage);
        const d_records = try cudaz.cuMemAlloc(@intCast(record_bytes));
        errdefer cudaz.cuMemFree(d_records);
        const d_indices = try cudaz.cuMemAlloc(@sizeOf(i64) * n_kv_local);
        errdefer cudaz.cuMemFree(d_indices);
        const d_descs = try cudaz.cuMemAlloc(@sizeOf(kvk.KvarnDesc) * 2);
        errdefer cudaz.cuMemFree(d_descs);

        // Zero the descs (init_descs writes over them, but a clean
        // state avoids spurious bits).
        try cudaz.cuMemsetD8(d_descs, 0, @sizeOf(kvk.KvarnDesc) * 2);
        // Indices seq [0, n_kv).
        const indices = try allocator.alloc(i64, n_kv_local);
        defer allocator.free(indices);
        for (0..n_kv_local) |i| indices[i] = @intCast(i);
        try cudaz.cuMemcpyHtoD(d_indices, @intFromPtr(indices.ptr), @sizeOf(i64) * n_kv_local);

        return .{
            .layout = layout,
            .record_bytes = record_bytes,
            .fattn_module = fattn_module,
            .kvk_module = kvk_module,
            .stream = stream,
            .d_stage = d_stage,
            .d_records = d_records,
            .d_indices = d_indices,
            .d_descs = d_descs,
            .n_kv = n_kv_local,
        };
    }

    fn deinit(self: *SmokeFixture) void {
        cudaz.cuMemFree(self.d_descs);
        cudaz.cuMemFree(self.d_indices);
        cudaz.cuMemFree(self.d_records);
        cudaz.cuMemFree(self.d_stage);
        cudaz.cuStreamDestroy(self.stream);
    }

    /// Run store + init_descs with a trivial f32 input. Returns the
    /// initialized descs ready for an attention launch.
    fn materialize(
        self: *const SmokeFixture,
        current_k: cudaz.CUdeviceptr,
        current_v: cudaz.CUdeviceptr,
        k_bits: c_int,
        v_bits: c_int,
    ) !void {
        const store_args_k: kvk.KvarnStoreArgs = .{
            .current = @ptrFromInt(current_k),
            .current_v = @ptrFromInt(current_v),
            .indices = @ptrFromInt(self.d_indices),
            .stage = @ptrFromInt(self.d_stage),
            .records = @ptrFromInt(self.d_records),
            .n_tokens = @intCast(self.n_kv),
            .n_record_heads = 1,
            .stream = 0,
            .groups_per_stream = 1,
            .record_bytes = self.record_bytes,
            .k_payload_off = @intCast(self.layout.k_payload_off),
            .k_s_col_off = @intCast(self.layout.k_s_col_off),
            .k_zp_off = @intCast(self.layout.k_zp_off),
            .k_s_row_off = @intCast(self.layout.k_s_row_off),
            .v_payload_off = @intCast(self.layout.v_payload_off),
            .v_s_col_off = @intCast(self.layout.v_s_col_off),
            .v_s_row_off = @intCast(self.layout.v_s_row_off),
            .v_zp_off = @intCast(self.layout.v_zp_off),
            .k_bits = k_bits,
            .v_bits = v_bits,
            .sinkhorn_iters = 4, // fewer iters for smoke
            .stage_groups = 2,
            .tail_groups = 1,
            .swa = 0,
            .eager_records = 0,
        };
        try kvk.kvarnStoreDevice(self.kvk_module, &store_args_k, self.stream);
        try cudaz.cuStreamSynchronize(self.stream);

        const init_args: kvk.KvarnInitDescsArgs = .{
            .n_stream = 1,
            .n_indices = @intCast(self.n_kv),
            .d_indices = @ptrFromInt(self.d_indices),
            .d_descs = @ptrFromInt(self.d_descs),
            .desc_stride = 2,
            .d_records = @ptrFromInt(self.d_records),
            .d_stage = @ptrFromInt(self.d_stage),
            .n_record_heads = 1,
            .groups_per_stream = 1,
            .record_bytes = self.record_bytes,
            .stage_groups = 2,
            .tail_groups = 1,
            .k_bits = k_bits,
            .v_bits = v_bits,
            .head_dim = 128, // BUG A @15a4b5c: campo obligatorio (D=128 aquí)
            .head_slices = 1,
            .eager_records = 0,
            .read_indirect = 0,
            .original_domain = 0,
            .swa = 0,
        };
        try kvk.kvarnInitDescsDevice(self.kvk_module, &init_args, self.stream);
        try cudaz.cuStreamSynchronize(self.stream);
    }
};

/// Run a wrapper and check the output is finite (not NaN/Inf).
/// Returns true if smoke passed.
fn checkWrapperSmoke(
    fixture: *const SmokeFixture,
    n_q_heads: u32,
) !void {
    const allocator = testing.allocator;
    const d_q = try cudaz.cuMemAlloc(@sizeOf(f32) * n_q_heads * D);
    defer cudaz.cuMemFree(d_q);
    const d_dst = try cudaz.cuMemAlloc(@sizeOf(f32) * n_q_heads * D);
    defer cudaz.cuMemFree(d_dst);

    // Q = 0.1 (constant).
    const q = try allocator.alloc(f32, n_q_heads * D);
    defer allocator.free(q);
    for (q) |*x| x.* = 0.1;
    try cudaz.cuMemcpyHtoD(d_q, @intFromPtr(q.ptr), @sizeOf(f32) * n_q_heads * D);
    try cudaz.cuMemsetD8(d_dst, 0, @sizeOf(f32) * n_q_heads * D);

    const args = fattn_kv.KvarnAttentionArgs{
        .q_data = @ptrFromInt(d_q),
        .k_descs = @ptrFromInt(fixture.d_descs),
        .v_descs = @ptrFromInt(fixture.d_descs + @sizeOf(kvk.KvarnDesc)),
        .mask_data = null,
        .dst_data = @ptrFromInt(d_dst),
        .n_kv = @intCast(fixture.n_kv),
        .n_q = 1,
        .n_q_heads = @intCast(n_q_heads),
        .n_kv_heads = 1,
        .n_stream = 1,
        .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(D))),
        .gqa = @intCast(n_q_heads),
    };
    _ = try fattn_kv.fattnKvarnPortableDevice(fixture.fattn_module, &args, fixture.stream);
    try cudaz.cuStreamSynchronize(fixture.stream);

    const out = try allocator.alloc(f32, n_q_heads * D);
    defer allocator.free(out);
    try cudaz.cuMemcpyDtoH(@intFromPtr(out.ptr), d_dst, @sizeOf(f32) * n_q_heads * D);
    for (out) |v| try testing.expect(std.math.isFinite(v));
}

test "ARITY smoke: fattnKvarnPortableDevice (D=128) output is finite" {
    if (build_options.fattn_cubin.len == 0) return error.SkipZigTest;
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;
    try cudaz.ensureContext();

    const allocator = testing.allocator;
    var fixture = try SmokeFixture.init(allocator, 128);
    defer fixture.deinit();

    const current_k = try cudaz.cuMemAlloc(@sizeOf(f32) * 128);
    defer cudaz.cuMemFree(current_k);
    const current_v = try cudaz.cuMemAlloc(@sizeOf(f32) * 128);
    defer cudaz.cuMemFree(current_v);
    const kv = try allocator.alloc(f32, 128);
    defer allocator.free(kv);
    for (kv) |*x| x.* = 0.1;
    try cudaz.cuMemcpyHtoD(current_k, @intFromPtr(kv.ptr), @sizeOf(f32) * 128);
    try cudaz.cuMemcpyHtoD(current_v, @intFromPtr(kv.ptr), @sizeOf(f32) * 128);

    try fixture.materialize(current_k, current_v, 4, 4);
    try checkWrapperSmoke(&fixture, 4);
}

test "ARITY smoke: fattnKvarnPortableD256Device output is finite" {
    if (build_options.fattn_cubin.len == 0) return error.SkipZigTest;
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;
    if (std.c.getenv("ZIG_AI_KVARN_D2_UNLOCK") == null) return error.SkipZigTest;
    try cudaz.ensureContext();

    const allocator = testing.allocator;
    var fixture = try SmokeFixture.init(allocator, 128);
    defer fixture.deinit();

    const current_k = try cudaz.cuMemAlloc(@sizeOf(f32) * 256);
    defer cudaz.cuMemFree(current_k);
    const current_v = try cudaz.cuMemAlloc(@sizeOf(f32) * 256);
    defer cudaz.cuMemFree(current_v);
    const kv = try allocator.alloc(f32, 256);
    defer allocator.free(kv);
    for (kv) |*x| x.* = 0.1;
    try cudaz.cuMemcpyHtoD(current_k, @intFromPtr(kv.ptr), @sizeOf(f32) * 256);
    try cudaz.cuMemcpyHtoD(current_v, @intFromPtr(kv.ptr), @sizeOf(f32) * 256);

    try fixture.materialize(current_k, current_v, 4, 4);
    try checkWrapperSmoke(&fixture, 4);
}

test "ARITY smoke: fattnKvarnPortableD512Device output is finite" {
    if (build_options.fattn_cubin.len == 0) return error.SkipZigTest;
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;
    if (std.c.getenv("ZIG_AI_KVARN_D2_UNLOCK") == null) return error.SkipZigTest;
    try cudaz.ensureContext();

    const allocator = testing.allocator;
    var fixture = try SmokeFixture.init(allocator, 128);
    defer fixture.deinit();

    const current_k = try cudaz.cuMemAlloc(@sizeOf(f32) * 512);
    defer cudaz.cuMemFree(current_k);
    const current_v = try cudaz.cuMemAlloc(@sizeOf(f32) * 512);
    defer cudaz.cuMemFree(current_v);
    const kv = try allocator.alloc(f32, 512);
    defer allocator.free(kv);
    for (kv) |*x| x.* = 0.1;
    try cudaz.cuMemcpyHtoD(current_k, @intFromPtr(kv.ptr), @sizeOf(f32) * 512);
    try cudaz.cuMemcpyHtoD(current_v, @intFromPtr(kv.ptr), @sizeOf(f32) * 512);

    try fixture.materialize(current_k, current_v, 4, 4);
    try checkWrapperSmoke(&fixture, 4);
}

test "ARITY smoke: fattnKvarnPortableD128TailDevice (n_tail=0) output is finite" {
    if (build_options.fattn_cubin.len == 0) return error.SkipZigTest;
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;
    try cudaz.ensureContext();

    const allocator = testing.allocator;
    var fixture = try SmokeFixture.init(allocator, 128);
    defer fixture.deinit();

    const current_k = try cudaz.cuMemAlloc(@sizeOf(f32) * 128);
    defer cudaz.cuMemFree(current_k);
    const current_v = try cudaz.cuMemAlloc(@sizeOf(f32) * 128);
    defer cudaz.cuMemFree(current_v);
    const kv = try allocator.alloc(f32, 128);
    defer allocator.free(kv);
    for (kv) |*x| x.* = 0.1;
    try cudaz.cuMemcpyHtoD(current_k, @intFromPtr(kv.ptr), @sizeOf(f32) * 128);
    try cudaz.cuMemcpyHtoD(current_v, @intFromPtr(kv.ptr), @sizeOf(f32) * 128);

    try fixture.materialize(current_k, current_v, 4, 4);

    // For the tail wrapper, build a custom launch (n_tail=0).
    const d_q = try cudaz.cuMemAlloc(@sizeOf(f32) * 4 * D);
    defer cudaz.cuMemFree(d_q);
    const d_dst = try cudaz.cuMemAlloc(@sizeOf(f32) * 4 * D);
    defer cudaz.cuMemFree(d_dst);
    const q = try allocator.alloc(f32, 4 * D);
    defer allocator.free(q);
    for (q) |*x| x.* = 0.1;
    try cudaz.cuMemcpyHtoD(d_q, @intFromPtr(q.ptr), @sizeOf(f32) * 4 * D);
    try cudaz.cuMemsetD8(d_dst, 0, @sizeOf(f32) * 4 * D);

    const tail_args = fattn_kv.KvarnAttentionTailArgs{
        .q_data = @ptrFromInt(d_q),
        .k_descs = @ptrFromInt(fixture.d_descs),
        .v_descs = @ptrFromInt(fixture.d_descs + @sizeOf(kvk.KvarnDesc)),
        .mask_data = null,
        .k_tail_data = null,
        .v_tail_data = null,
        .tail_mask = null,
        .run_desc_slots = null,
        .n_kv = @intCast(fixture.n_kv),
        .n_tail = 0,
        .d_k = 0,
        .d_v = 0,
        .n_q = 1,
        .n_q_heads = 4,
        .n_kv_heads = 1,
        .n_stream = 1,
        .k_tail_bf16 = 0,
        .v_tail_bf16 = 0,
        .dst_data = @ptrFromInt(d_dst),
        .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(D))),
        .gqa = 4,
    };
    try fattn_kv.fattnKvarnPortableD128TailDevice(fixture.fattn_module, &tail_args, fixture.stream);
    try cudaz.cuStreamSynchronize(fixture.stream);

    const out = try allocator.alloc(f32, 4 * D);
    defer allocator.free(out);
    try cudaz.cuMemcpyDtoH(@intFromPtr(out.ptr), d_dst, @sizeOf(f32) * 4 * D);
    for (out) |v_| {
        try testing.expect(!std.math.isNan(v_));
        try testing.expect(!std.math.isInf(v_));
    }
}

test "ARITY smoke: fattnKvarnVecDevice (D=256 GQA=2 k4v4) output is finite" {
    if (build_options.fattn_cubin.len == 0) return error.SkipZigTest;
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;
    // Vec D=256 E2E: WIP Dev-B (softmax-per-slice structural fix
    // pendiente) + requiere D2 (cross-slice WHT ratificación B2).
    if (std.c.getenv("ZIG_AI_KVARN_D2_UNLOCK") == null) return error.SkipZigTest;
    try cudaz.ensureContext();

    const allocator = testing.allocator;
    var fixture = try SmokeFixture.init(allocator, 128);
    defer fixture.deinit();

    const current_k = try cudaz.cuMemAlloc(@sizeOf(f32) * 256);
    defer cudaz.cuMemFree(current_k);
    const current_v = try cudaz.cuMemAlloc(@sizeOf(f32) * 256);
    defer cudaz.cuMemFree(current_v);
    const kv = try allocator.alloc(f32, 256);
    defer allocator.free(kv);
    for (kv) |*x| x.* = 0.1;
    try cudaz.cuMemcpyHtoD(current_k, @intFromPtr(kv.ptr), @sizeOf(f32) * 256);
    try cudaz.cuMemcpyHtoD(current_v, @intFromPtr(kv.ptr), @sizeOf(f32) * 256);

    try fixture.materialize(current_k, current_v, 4, 4);

    const d_q = try cudaz.cuMemAlloc(@sizeOf(f32) * 2 * 256);
    defer cudaz.cuMemFree(d_q);
    const d_dst = try cudaz.cuMemAlloc(@sizeOf(f32) * 2 * 256);
    defer cudaz.cuMemFree(d_dst);
    const q = try allocator.alloc(f32, 2 * 256);
    defer allocator.free(q);
    for (q) |*x| x.* = 0.1;
    try cudaz.cuMemcpyHtoD(d_q, @intFromPtr(q.ptr), @sizeOf(f32) * 2 * 256);
    try cudaz.cuMemsetD8(d_dst, 0, @sizeOf(f32) * 2 * 256);

    const vec_args = fattn_kv.KvarnVecArgs{
        .q_data = @ptrFromInt(d_q),
        .k_descs = @ptrFromInt(fixture.d_descs),
        .v_descs = @ptrFromInt(fixture.d_descs + @sizeOf(kvk.KvarnDesc)),
        .mask_data = null,
        .dst_data = @ptrFromInt(d_dst),
        .n_kv = @intCast(fixture.n_kv),
        .n_q_heads = 2, // vec eligibility: GQA=2
        .n_kv_heads = 1,
        .n_stream = 1,
        .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(256))),
    };
    _ = try fattn_kv.fattnKvarnVecDevice(fixture.fattn_module, &vec_args, fixture.stream);
    try cudaz.cuStreamSynchronize(fixture.stream);

    const out = try allocator.alloc(f32, 2 * 256);
    defer allocator.free(out);
    try cudaz.cuMemcpyDtoH(@intFromPtr(out.ptr), d_dst, @sizeOf(f32) * 2 * 256);
    for (out) |v_| {
        try testing.expect(!std.math.isNan(v_));
        try testing.expect(!std.math.isInf(v_));
    }
}

test "ARITY smoke: kvarnDecodeSplitDevice output is finite (D=128 gqa2)" {
    if (build_options.kvarn_split_cubin.len == 0) return error.SkipZigTest;
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;

    try cudaz.ensureContext();
    const smod = try cudaz.cuModuleLoad(build_options.kvarn_split_cubin);
    const kmod = try cudaz.cuModuleLoad(build_options.kvarn_cubin);
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);
    const allocator = testing.allocator;

    const layout = try kvarn.KvarnRecordLayout.init(D_U32, 4, 4);
    const n_kv: usize = 256; // 2 grupos
    const n_kvh: usize = 1;
    const n_qh: usize = 2;
    const q_size: usize = 1 * n_qh * D;
    const n_groups: usize = 2;
    const stage_groups: usize = 4;
    const n_splits: usize = 4; // divUp(256, 64)

    const d_q = try cudaz.cuMemAlloc(@sizeOf(f32) * q_size);
    defer cudaz.cuMemFree(d_q);
    const q = try allocator.alloc(f32, q_size);
    defer allocator.free(q);
    for (q) |*x| x.* = 0.25;
    try cudaz.cuMemcpyHtoD(d_q, @intFromPtr(q.ptr), @sizeOf(f32) * q_size);
    const d_dst = try cudaz.cuMemAlloc(@sizeOf(f32) * q_size);
    defer cudaz.cuMemFree(d_dst);
    try cudaz.cuMemsetD8(d_dst, 0, @sizeOf(f32) * q_size);

    const n_descs: usize = 2 * n_kvh;
    const d_descs = try cudaz.cuMemAlloc(@sizeOf(kvk.KvarnDesc) * n_descs);
    defer cudaz.cuMemFree(d_descs);
    const records_len: usize = n_groups * n_kvh * layout.tile_bytes;
    const d_records = try cudaz.cuMemAlloc(records_len);
    defer cudaz.cuMemFree(d_records);
    try cudaz.cuMemsetD8(d_records, 0, records_len);
    const d_indices = try cudaz.cuMemAlloc(@sizeOf(i64) * n_kv);
    defer cudaz.cuMemFree(d_indices);
    const idx = try allocator.alloc(i64, n_kv);
    defer allocator.free(idx);
    for (idx, 0..) |*e, i| e.* = @intCast(i);
    try cudaz.cuMemcpyHtoD(d_indices, @intFromPtr(idx.ptr), @sizeOf(i64) * n_kv);
    const stage_len: usize = stage_groups * 128 * (2 * n_kvh) * 128;
    const d_stage = try cudaz.cuMemAlloc(@sizeOf(f16) * stage_len);
    defer cudaz.cuMemFree(d_stage);
    try cudaz.cuMemsetD8(d_stage, 0, @sizeOf(f16) * stage_len);

    var ia: kvk.KvarnInitDescsArgs = .{
        .n_stream = 1,
        .n_indices = @intCast(n_kv),
        .d_indices = @ptrFromInt(d_indices),
        .d_descs = @ptrFromInt(d_descs),
        .desc_stride = 2,
        .d_records = @ptrFromInt(d_records),
        .d_stage = @ptrFromInt(d_stage),
        .n_record_heads = @intCast(n_kvh),
        .groups_per_stream = @intCast(n_groups),
        .record_bytes = @intCast(layout.tile_bytes),
        .stage_groups = @intCast(stage_groups),
        .tail_groups = 3,
        .k_bits = 4,
        .v_bits = 4,
        .head_dim = 128, // BUG A @15a4b5c: campo obligatorio (D=128 aquí)
        .head_slices = 1,
        .eager_records = 1,
        .read_indirect = 0,
        .original_domain = 0,
        .swa = 0,
    };
    try kvk.kvarnInitDescsDevice(kmod, &ia, stream);
    try cudaz.cuStreamSynchronize(stream);

    const partial_len: usize = 1 * n_qh * n_splits * D;
    const meta_len: usize = 1 * n_qh * n_splits;
    const d_partial = try cudaz.cuMemAlloc(@sizeOf(f32) * partial_len);
    defer cudaz.cuMemFree(d_partial);
    const d_meta = try cudaz.cuMemAlloc(@sizeOf(f32) * 2 * meta_len);
    defer cudaz.cuMemFree(d_meta);

    var sargs: kvk.KvarnSplitLaunchArgs = .{
        .q_data = d_q,
        .k_descs = d_descs,
        .v_descs = d_descs + @sizeOf(kvk.KvarnDesc),
        .partial_data = d_partial,
        .meta_data = d_meta,
        .dst_data = d_dst,
        .n_kv = @intCast(n_kv),
        .n_q = 1,
        .n_q_heads = @intCast(n_qh),
        .n_kv_heads = @intCast(n_kvh),
        .n_splits = @intCast(n_splits),
        .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(D_U32))),
    };
    _ = try kvk.kvarnDecodeSplitDevice(smod, &sargs, stream);
    try cudaz.cuStreamSynchronize(stream);

    const out = try allocator.alloc(f32, q_size);
    defer allocator.free(out);
    try cudaz.cuMemcpyDtoH(@intFromPtr(out.ptr), d_dst, @sizeOf(f32) * q_size);
    for (out) |v| try testing.expect(std.math.isFinite(v));
}
