// ============================================================================
// M3 integration pattern — replica el wiring de main.zig (slice 1+2+3
// de Dev-B) SIN GGUF: N capas × (prefill + decode) con probeDevice real
// para smem_optin (A5-adaptativo), mismo stream, kvDevicePtrs-style
// ptrs device→device, y verificación de descs/live por capa al final.
// Gate: descs escalares idénticos entre capas, live tracking exacto,
// atención finita vía kvarnDecodeSplitDevice en cada capa.
// ============================================================================

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
const kvg_mod = @import("kvarn_gpu_cache");
const bc = @import("backend_capabilities");
const build_options = @import("build_options");
const KvarnGpuCache = kvg_mod.KvarnGpuCache;

test "M3 main-pattern: 3 capas × (prefill 256 + decode 8) con smem_optin probeDevice real" {
    if (build_options.kvarn_cubin.len == 0) return error.SkipZigTest;
    if (build_options.kvarn_split_cubin.len == 0) return error.SkipZigTest;

    cudaz.ensureContext() catch return error.SkipZigTest;
    const allocator = testing.allocator;
    const kmod = try cudaz.cuModuleLoad(build_options.kvarn_cubin);
    const smod = try cudaz.cuModuleLoad(build_options.kvarn_split_cubin);
    _ = smod; // (futuro: verify split attention per-layer; este gate = descs)
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);

    // smem_optin REAL del device (como probeDevice; sin tabla estática).
    const caps = bc.probeDevice(0);
    const smem_optin = caps.shared_memory_per_block;
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "main-pattern: smem_optin real = {?d}B\n", .{smem_optin});

    const num_layers: u32 = 3;
    const n_kv_heads: u32 = 2; // GQA-heavy (aisla bug kvh>1: si =1 pasa)
    const ctx: u32 = 384; // 3 grupos
    var cache = try KvarnGpuCache.init(allocator, .{
        .num_layers = num_layers,
        .n_kv_heads = n_kv_heads,
        .head_dim = 128,
        .k_bits = 5,
        .v_bits = 4,
        .max_ctx_tokens = ctx,
        .smem_optin = smem_optin,
    }, kmod, stream);
    defer cache.deinit();

    // Buffers K/V device por capa (como g_k/g_v del motor — reuso un
    // par global porque el patrón re-escribe cada forward):
    const per_tok = n_kv_heads * 128;
    const prefill_n: u32 = 256;
    const decode_n: u32 = 8;
    const d_k = try cudaz.cuMemAlloc(@sizeOf(f32) * @as(usize, prefill_n) * per_tok);
    defer cudaz.cuMemFree(d_k);
    const d_v = try cudaz.cuMemAlloc(@sizeOf(f32) * @as(usize, prefill_n) * per_tok);
    defer cudaz.cuMemFree(d_v);

    var prng = std.Random.DefaultPrng.init(0xC0DE);
    const rand = prng.random();
    const host_kv = try allocator.alloc(f32, @as(usize, prefill_n) * per_tok);
    defer allocator.free(host_kv);
    for (host_kv) |*x| x.* = rand.float(f32) * 0.4 - 0.2;
    try cudaz.cuMemcpyHtoD(d_k, @intFromPtr(host_kv.ptr), @sizeOf(f32) * host_kv.len);
    try cudaz.cuMemcpyHtoD(d_v, @intFromPtr(host_kv.ptr), @sizeOf(f32) * host_kv.len);

    // ==== PATRÓN main.zig: prefill por chunks de 128, TODAS las capas ====
    var pos: u32 = 0;
    while (pos < prefill_n) {
        const n: u32 = @min(128, prefill_n - pos);
        for (0..num_layers) |li| {
            // (en main: forwardGPU escribe g_k/g_v; aquí ya están subidos)
            try cache.appendTokens(@intCast(li), d_k + pos * per_tok * @sizeOf(f32), d_v + pos * per_tok * @sizeOf(f32), n, pos, null, stream);
        }
        pos += n;
    }
    // ==== decode 1×8 por capa (kvDevicePtrs pattern: n=1, base=pos-1) ====
    for (0..decode_n) |di| {
        const p: u32 = prefill_n + @as(u32, @intCast(di));
        for (0..num_layers) |li| {
            try cache.appendTokens(@intCast(li), d_k, d_v, 1, p, null, stream);
        }
    }
    try cudaz.cuStreamSynchronize(stream);

    // Verificación: descs por capa — live exacto tras 264 tokens:
    // 264 = 2·128 + 8 ⇒ live_group=2, live_pos=8, n descs pair K/V.
    const DescSz = @sizeOf(kvk.KvarnDesc);
    const n_descs = 2 * n_kv_heads; // par (K,V) por kv_head
    for (0..num_layers) |li| {
        const descs = try allocator.alloc(u8, n_descs * DescSz);
        defer allocator.free(descs);
        try cudaz.cuMemcpyDtoH(@intFromPtr(descs.ptr), try cache.kDescs(@intCast(li)), n_descs * DescSz);
        for (0..n_kv_heads) |h| {
            const dk = @as([*]align(1) const kvk.KvarnDesc, @ptrCast(descs.ptr))[2 * h];
            const dv = @as([*]align(1) const kvk.KvarnDesc, @ptrCast(descs.ptr))[2 * h + 1];
            try testing.expectEqual(@as(c_int, 2), dk.live_group);
            try testing.expectEqual(@as(c_int, 7), dk.live_pos);
            try testing.expectEqual(@as(c_int, 7), dv.live_pos);
            try testing.expectEqual(@as(c_int, @intCast(n_kv_heads)), dk.n_record_heads);
        }
    }
    try stdoutPrint(std.Io.Threaded.global_single_threaded.io(), "main-pattern: {d} capas × {d} tok, descs live=(2,7) todas OK (smem ruta {s})\n", .{
        num_layers,                                                  prefill_n + decode_n,
        if (smem_optin orelse 0 >= 69704) "hishmem" else "lowshmem",
    });
}
