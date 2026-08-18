//! Benchmark de PagedAttention — Fase 5.2.
//! Mide: latencia decode (ms/token) y throughput (tok/s) vs batch size,
//! memoria VRAM vs num_blocks, hit-rate de prefix cache, y latencia de
//! CPU offload (swapToCpu/swapFromCpu). Usa referencia CPU; si CUDA está
//! disponible añade decode GPU y el pool persistente.
const std = @import("std");
const pa = @import("paged_attention");
const cudaz = @import("cudaz");
const debug = @import("debug");

/// Config sintética de decode (head_dim=128, 32Q/8KV, f16).
fn benchConfig(num_blocks: usize, max_batch: usize) pa.PagedConfig {
    return .{
        .block_size = 16,
        .num_blocks = num_blocks,
        .head_dim = 128,
        .num_kv_heads = 8,
        .num_q_heads = 32,
        .dtype = .f16,
        .enable_prefix_cache = true,
        .enable_cpu_offload = false,
        .max_seq_len = 2048,
        .max_batch_size = max_batch,
    };
}

fn fillBlocks(kv: *pa.PagedKVCache, seq_id: u64, seed: u64) !void {
    var rng = std.Random.Xoshiro256.init(seed);
    const bt = kv.getBlockTableMut(seq_id).?;
    const head_dim = kv.config.head_dim;
    const num_kv_heads = kv.config.num_kv_heads;
    const block_size = kv.config.block_size;
    for (bt.table.items) |phys_id| {
        const data = kv.getBlockData(phys_id);
        const fdata = @as([*]f16, @ptrCast(@alignCast(data)))[0 .. block_size * num_kv_heads * head_dim * 2];
        for (fdata) |*v| {
            const r: f32 = rng.random().float(f32);
            v.* = @floatCast((r - 0.5) * 2.0);
        }
    }
}

fn randomQueries(allocator: std.mem.Allocator, n: usize, stride: usize, seed: u64) ![]f32 {
    const q = try allocator.alloc(f32, n * stride);
    var rng = std.Random.Xoshiro256.init(seed);
    for (q) |*v| v.* = (rng.random().float(f32) - 0.5) * 2.0;
    return q;
}

/// Latencia decode (ms/token) y throughput (tok/s) de un batch.
fn benchDecodeBatch(allocator: std.mem.Allocator, batch: usize, seq_len: usize, iters: usize) !struct { cpu_ms: f64, cpu_toks: f64, gpu_ms: f64, gpu_toks: f64 } {
    const config = benchConfig(1024, batch);
    debug.dbg.printLevel(.info, "benchDecodeBatch: creating KV cache with {} blocks\n", .{config.num_blocks});
    var kv = try pa.PagedKVCache.init(allocator, config);
    defer kv.deinit();
    debug.dbg.printLevel(.info, "benchDecodeBatch: KV cache created\n", .{});

    const attn = pa.PagedAttention.init(allocator, config);
    var engine: ?pa.PagedAttentionGpu = null;
    var gpu_stream: cudaz.CUstream = undefined;
    var has_cuda = false;
    defer if (engine) |*e| e.deinit();
    defer if (has_cuda) cudaz.cuStreamDestroy(gpu_stream);
    if (cudaz.isCudaAvailable()) {
        debug.dbg.printLevel(.info, "benchDecodeBatch: CUDA available, ensuring context\n", .{});
        try cudaz.ensureContext();
        debug.dbg.printLevel(.info, "benchDecodeBatch: context ensured, creating stream\n", .{});
        gpu_stream = try cudaz.cuStreamCreate(0);
        debug.dbg.printLevel(.info, "benchDecodeBatch: stream created: {}\n", .{gpu_stream});
        has_cuda = true;
        debug.dbg.printLevel(.info, "benchDecodeBatch: initializing PagedAttentionGpu\n", .{});
        engine = try pa.PagedAttentionGpu.init(allocator, config, gpu_stream);
        debug.dbg.printLevel(.info, "benchDecodeBatch: PagedAttentionGpu initialized successfully\n", .{});
    }

    var seq_ids: [64]u64 = undefined;
    for (0..batch) |i| {
        seq_ids[i] = try kv.createSequence();
        try kv.allocatePrefill(seq_ids[i], seq_len);
        try fillBlocks(&kv, seq_ids[i], 100 + i);
    }

    var bts: [64]*const pa.BlockTable = undefined;
    for (0..batch) |i| bts[i] = kv.getBlockTable(seq_ids[i]).?;

    const q_stride = config.num_q_heads * config.head_dim;
    const queries = try randomQueries(allocator, batch, q_stride, 7);
    defer allocator.free(queries);
    const outs = try allocator.alloc(f32, batch * q_stride);
    defer allocator.free(outs);

    var t = @import("time").Timer.start();
    debug.dbg.printLevel(.info, "benchDecodeBatch: running CPU benchmark loop ({} iters)\n", .{iters});
    for (0..iters) |_| try attn.decodeBatch(queries, outs, bts[0..batch], kv.block_alloc);
    const cpu_ns = t.read();
    debug.dbg.printLevel(.info, "benchDecodeBatch: CPU loop done\n", .{});
    const cpu_ms = @as(f64, @floatFromInt(@divTrunc(cpu_ns, std.time.ns_per_ms))) / @as(f64, @floatFromInt(iters));
    const cpu_toks = @as(f64, @floatFromInt(batch)) / (cpu_ms / 1000.0);

    var gpu_ms: f64 = 0;
    var gpu_toks: f64 = 0;
    if (engine) |*e| {
        debug.dbg.printLevel(.info, "benchDecodeBatch: running GPU benchmark loop ({} iters)\n", .{iters});
        var tg = @import("time").Timer.start();
        for (0..iters) |it| {
            debug.dbg.printLevel(.detail, "benchDecodeBatch: GPU iter {}\n", .{it});
            for (0..batch) |b| {
                debug.dbg.printLevel(.detail, "benchDecodeBatch: GPU iter {} batch {}\n", .{it, b});
                try e.decode(
                    queries[b * q_stride ..][0..q_stride],
                    outs[b * q_stride ..][0..q_stride],
                    bts[b],
                    kv.block_alloc,
                );
            }
        }
        debug.dbg.printLevel(.info, "benchDecodeBatch: GPU loop done\n", .{});
        const gpu_ns = tg.read();
        gpu_ms = @as(f64, @floatFromInt(@divTrunc(gpu_ns, std.time.ns_per_ms))) / @as(f64, @floatFromInt(iters));
        gpu_toks = @as(f64, @floatFromInt(batch)) / (gpu_ms / 1000.0);
    }

    return .{ .cpu_ms = cpu_ms, .cpu_toks = cpu_toks, .gpu_ms = gpu_ms, .gpu_toks = gpu_toks };
}

pub fn main(init: std.process.Init) !void {
    debug.init();
    const io = init.io;
    const allocator = init.gpa;

    var stdout_buffer: [0x200]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print("\n", .{});
    try stdout.print("=================================================\n", .{});
    try stdout.print("   PagedAttention Benchmark — Fase 5.2          \n", .{});
    try stdout.print("=================================================\n", .{});
    try stdout.print("\n", .{});

    debug.dbg.printLevel(.info, "main: starting benchmark\n", .{});
    const config = benchConfig(512, 16);
    try stdout.print("Config: block_size={d}, head_dim={d}, {d}Q/{d}KV, f16, ctx={d}\n", .{
        config.block_size, config.head_dim, config.num_q_heads, config.num_kv_heads, config.max_seq_len,
    });
    try stdout.print("CUDA: {s}\n\n", .{if (cudaz.isCudaAvailable()) "disponible" else "no disponible"});

    // 1. Decode latency / throughput vs batch size (seq_len = 512 tokens = 32 bloques)
    const seq_len: usize = 512;
    const batch_sizes = &[_]usize{ 1, 2, 4, 8, 16 };
    try stdout.print("--- Decode: latencia (ms/token) y throughput (tok/s) vs batch (seq_len={d}) ---\n", .{seq_len});
    for (batch_sizes) |b| {
        const r = try benchDecodeBatch(allocator, b, seq_len, 20);
        try stdout.print("batch={d:>2}  CPU {d:7.3} ms/tok ({d:8.2} tok/s)", .{ b, r.cpu_ms, r.cpu_toks });
        if (r.gpu_ms > 0) {
            try stdout.print("   GPU {d:7.3} ms/tok ({d:8.2} tok/s)", .{ r.gpu_ms, r.gpu_toks });
        }
        try stdout.print("\n", .{});
    }

    // 2. Memoria VRAM / host vs num_blocks
    try stdout.print("\n--- Memoria KV-cache (f16) vs num_blocks ---\n", .{});
    const bytes_per_block = config.block_size * config.num_kv_heads * config.head_dim * 2 * 2;
    const num_blocks_vals = &[_]usize{ 256, 512, 1024, 2048 };
    for (num_blocks_vals) |nb| {
        const mem = nb * bytes_per_block;
        try stdout.print("num_blocks={d:>6}  {d:8.1} MB total  ({d} B/bloque)\n", .{ nb, @as(f64, @floatFromInt(mem)) / (1024.0 * 1024.0), bytes_per_block });
    }

    // 3. Prefix cache hit rate
    try stdout.print("\n--- Prefix cache hit rate ---\n", .{});
    try benchPrefixHitRate(allocator, stdout);

    // 4. CPU offload latency
    try stdout.print("\n--- CPU offload (swapToCpu/swapFromCpu) ---\n", .{});
    try benchCpuOffload(allocator, stdout);

    try stdout.print("\n=================================================\n", .{});
    try stdout.print("              Benchmark completado               \n", .{});
    try stdout.print("=================================================\n", .{});
    try stdout.flush();
}

fn benchPrefixHitRate(allocator: std.mem.Allocator, stdout: anytype) !void {
    const config = benchConfig(512, 16);
    var kv = try pa.PagedKVCache.init(allocator, config);
    defer kv.deinit();

    // Genera n_prefixes prefijos de 2 bloques, los cachea y guarda para probe.
    const n_prefixes: usize = 64;
    const prefix_len: usize = 32; // 2 bloques de 16
    var rng = std.Random.Xoshiro256.init(1234);

    var prefixes = try allocator.alloc([]u32, n_prefixes);
    defer allocator.free(prefixes);
    for (0..n_prefixes) |i| {
        const tokens = try allocator.alloc(u32, prefix_len);
        for (tokens) |*t| t.* = rng.random().uintLessThan(u32, 50000);
        prefixes[i] = tokens;
        const seq_id = try kv.createSequence();
        try kv.allocatePrefill(seq_id, prefix_len);
        try kv.cachePrefix(seq_id, tokens);
    }
    defer for (prefixes) |p| allocator.free(p);

    // Probes: la mitad reutiliza un prefijo cacheador exacto (hit esperado),
    // la otra mitad son tokens nuevos (miss esperado).
    const probes: usize = 400;
    var hits: usize = 0;
    for (0..probes) |p| {
        var buf: std.ArrayList(u32) = .empty;
        defer buf.deinit(allocator);
        if (p % 2 == 0) {
            const src = prefixes[(p / 2) % n_prefixes];
            for (src) |t| buf.append(allocator, t) catch {};
        } else {
            for (0..prefix_len) |_| buf.append(allocator, rng.random().uintLessThan(u32, 50000)) catch {};
        }
        const m = try kv.matchPrefix(buf.items);
        if (m > 0) hits += 1;
    }

    const stats = kv.getStats();
    try stdout.print("prefixes={d}, probes={d}, hits={d} ({d:.1}%), hit_rate_estad={d:.3}\n", .{
        n_prefixes, probes, hits, @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(probes)) * 100.0, stats.prefix_hit_rate,
    });
}

fn benchCpuOffload(allocator: std.mem.Allocator, stdout: anytype) !void {
    const config = benchConfig(512, 16);
    const num_blocks = 512;
    var ba = try pa.BlockAllocator.init(
        allocator, num_blocks, config.block_size,
        config.num_kv_heads, config.head_dim, .f16, true,
    );
    defer ba.deinit();

    var rng = std.Random.Xoshiro256.init(99);
    for (ba.memory_pool) |*b| b.* = rng.random().int(u8);

    const ids = try allocator.alloc(usize, num_blocks);
    defer allocator.free(ids);
    for (0..num_blocks) |i| ids[i] = i;

    const iters: usize = 100;
    var t = @import("time").Timer.start();
    for (0..iters) |_| {
        for (ids) |id| try ba.swapToCpu(id);
        for (ids) |id| try ba.swapFromCpu(id);
    }
    const ns = t.read();
    const ms_total = @as(f64, @floatFromInt(@divTrunc(ns, std.time.ns_per_ms)));
    const us_per_block = ms_total * 1000.0 / @as(f64, @floatFromInt(num_blocks * 2 * iters));
    try stdout.print("{d} bloques x {d} iter (swap-out+swap-in): {d:.2} ms → {d:.2} us/bloque\n", .{
        num_blocks, iters, ms_total, us_per_block,
    });
}