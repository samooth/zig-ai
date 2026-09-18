//! Benchmark sintético DFlash2 selector top-K GPU vs CPU.
//! No requiere modelo DFlash2 real; usa scores aleatorios en device.
const std = @import("std");
const cudaz = @import("cudaz");
const layer_kernels = @import("layer_kernels");
const specsampler = @import("speculative").sampler;
const debugz = @import("debug");

test "dflash2 top-k GPU vs CPU parity + throughput" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    cudaz.ensureContext() catch return error.SkipZigTest;
    const stream = try cudaz.cuStreamCreate(0);
    _ = std.c.setenv("ZIG_AI_DFLASH2_TOPK_GPU", "1", 1);
    defer cudaz.cuStreamDestroy(stream);
    var lk = try layer_kernels.LayerKernels.init(stream);
    defer lk.deinit();

    const num_experts = 256;
    const top_k = 8;
    const iters = 1000;

    var rng = std.Random.Xoshiro256.init(42);
    const scores_host = try gpa.alloc(f32, num_experts);
    defer gpa.free(scores_host);
    for (scores_host) |*v| v.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);

    const scores_dev = try cudaz.cuMemAlloc(num_experts * @sizeOf(f32));
    defer cudaz.cuMemFree(scores_dev);
    try cudaz.cuMemcpyHtoD(scores_dev, @intFromPtr(scores_host.ptr), num_experts * @sizeOf(f32));

    const ids_dev = try cudaz.cuMemAlloc(top_k * @sizeOf(i32));
    defer cudaz.cuMemFree(ids_dev);
    const vals_dev = try cudaz.cuMemAlloc(top_k * @sizeOf(f32));
    defer cudaz.cuMemFree(vals_dev);

    // CPU reference
    var cpu_idx: [8]u32 = undefined;
    var cpu_val: [8]f32 = undefined;
    specsampler.topK(scores_host, &cpu_idx, &cpu_val);

    // GPU warmup
    try lk.dflash2TopK(scores_dev, ids_dev, vals_dev, num_experts, top_k);
    try cudaz.cuStreamSynchronize(stream);

    // GPU timed
    const t0: i128 = @import("time").Timer.now();
    for (0..iters) |_| {
        try lk.dflash2TopK(scores_dev, ids_dev, vals_dev, num_experts, top_k);
    }
    try cudaz.cuStreamSynchronize(stream);
    const t1: i128 = @import("time").Timer.now();
    const gpu_ns = t1 - t0;
    const gpu_us = @as(f64, @floatFromInt(gpu_ns)) / @as(f64, iters) / 1000.0;

    // Read back GPU results
    const gpu_ids = try gpa.alloc(i32, top_k);
    defer gpa.free(gpu_ids);
    const gpu_vals = try gpa.alloc(f32, top_k);
    defer gpa.free(gpu_vals);
    try cudaz.cuMemcpyDtoH(@intFromPtr(gpu_ids.ptr), ids_dev, top_k * @sizeOf(i32));
    try cudaz.cuMemcpyDtoH(@intFromPtr(gpu_vals.ptr), vals_dev, top_k * @sizeOf(f32));

    // Parity check (top-k sets may differ by order for equal values; check sets)
    var gpu_set: [8]usize = undefined;
    for (gpu_ids, 0..) |id, i| gpu_set[i] = @intCast(id);
    std.sort.pdq(usize, &gpu_set, {}, struct {
        fn lessThan(_: void, a: usize, b: usize) bool {
            return a < b;
        }
    }.lessThan);
    var cpu_set: [8]usize = undefined;
    for (cpu_idx, 0..) |idx, i| cpu_set[i] = @intCast(idx);
    std.sort.pdq(usize, &cpu_set, {}, struct {
        fn lessThan(_: void, a: usize, b: usize) bool {
            return a < b;
        }
    }.lessThan);
    for (0..top_k) |i| {
        try std.testing.expectEqual(cpu_set[i], gpu_set[i]);
    }

    try std.testing.expect(gpu_us < 100.0);
    debugz.dbg.printLevel(.info, "[dflash2_topk] E={d} K={d} gpu={d:.1}us/iter\n", .{ num_experts, top_k, gpu_us });
}

test "dflash2 tree-walk parity vs selector" {
    if (!cudaz.isCudaAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    cudaz.ensureContext() catch return error.SkipZigTest;
    const stream = try cudaz.cuStreamCreate(0);
    _ = std.c.setenv("ZIG_AI_DFLASH2_TOPK_GPU", "1", 1);
    defer cudaz.cuStreamDestroy(stream);
    var lk = try layer_kernels.LayerKernels.init(stream);
    defer lk.deinit();

    const num_experts = 256;
    const top_k = 8;

    var rng = std.Random.Xoshiro256.init(42);
    const scores_host = try gpa.alloc(f32, num_experts);
    defer gpa.free(scores_host);
    for (scores_host) |*v| v.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);

    const scores_dev = try cudaz.cuMemAlloc(num_experts * @sizeOf(f32));
    defer cudaz.cuMemFree(scores_dev);
    try cudaz.cuMemcpyHtoD(scores_dev, @intFromPtr(scores_host.ptr), num_experts * @sizeOf(f32));

    const ids_dev = try cudaz.cuMemAlloc(top_k * @sizeOf(i32));
    defer cudaz.cuMemFree(ids_dev);
    const vals_dev = try cudaz.cuMemAlloc(top_k * @sizeOf(f32));
    defer cudaz.cuMemFree(vals_dev);

    // Run selector
    try lk.dflash2TopK(scores_dev, ids_dev, vals_dev, num_experts, top_k);
    try cudaz.cuStreamSynchronize(stream);

    const sel_ids = try gpa.alloc(i32, top_k);
    defer gpa.free(sel_ids);
    const sel_vals = try gpa.alloc(f32, top_k);
    defer gpa.free(sel_vals);
    try cudaz.cuMemcpyDtoH(@intFromPtr(sel_ids.ptr), ids_dev, top_k * @sizeOf(i32));
    try cudaz.cuMemcpyDtoH(@intFromPtr(sel_vals.ptr), vals_dev, top_k * @sizeOf(f32));

    // Run tree-walk
    try lk.dflash2TreeWalk(scores_dev, ids_dev, vals_dev, num_experts, top_k);
    try cudaz.cuStreamSynchronize(stream);

    const tw_ids = try gpa.alloc(i32, top_k);
    defer gpa.free(tw_ids);
    const tw_vals = try gpa.alloc(f32, top_k);
    defer gpa.free(tw_vals);
    try cudaz.cuMemcpyDtoH(@intFromPtr(tw_ids.ptr), ids_dev, top_k * @sizeOf(i32));
    try cudaz.cuMemcpyDtoH(@intFromPtr(tw_vals.ptr), vals_dev, top_k * @sizeOf(f32));

    // Both should select the same top-K experts (order may differ)
    var sel_set: [8]usize = undefined;
    for (sel_ids, 0..) |id, i| sel_set[i] = @intCast(id);
    std.sort.pdq(usize, &sel_set, {}, struct {
        fn lessThan(_: void, a: usize, b: usize) bool {
            return a < b;
        }
    }.lessThan);
    var tw_set: [8]usize = undefined;
    for (tw_ids, 0..) |id, i| tw_set[i] = @intCast(id);
    std.sort.pdq(usize, &tw_set, {}, struct {
        fn lessThan(_: void, a: usize, b: usize) bool {
            return a < b;
        }
    }.lessThan);
    for (0..top_k) |i| {
        try std.testing.expectEqual(sel_set[i], tw_set[i]);
    }
}
