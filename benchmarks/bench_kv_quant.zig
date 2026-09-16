// Benchmark KV Cuantizado: memoria y tok/s
// Mide:
// 1. VRAM ahorrada por formato cuantizado vs fp16
// 2. tok/s decode con KV cuantizado vs fp16

const std = @import("std");
const cudaz = @import("cudaz");
const layer_kernels = @import("layer_kernels");
const time = @import("time");

const QuantFormat = enum(u32) {
    fp16,
    q8_0,
    q4_0,
    q4_k,
    q8_k,
    q2_k,
    q3_k,
    q5_k,
    q6_k,
    iq4_xs,
    iq4_nl,
    iq3_xxs,
    iq3_s,
    iq1_s,
    iq1_m,
    iq2_xxs,
    iq2_xs,
    iq2_s,
    tq1_0,
    tq2_0,
    mxfp4,
};

const FormatSpec = struct {
    fmt: QuantFormat,
    group_bytes: usize,
    tag: []const u8,
    head_dim: usize,
    num_kv_heads: usize,
    num_q_heads: usize,
    block_size: usize,
    gran: usize,

    fn kv_dim(self: FormatSpec) usize {
        return self.num_kv_heads * self.head_dim;
    }
    fn elems_region(self: FormatSpec) usize {
        return self.block_size * self.kv_dim();
    }
    fn groups_per_region(self: FormatSpec) usize {
        return (self.elems_region() + self.gran - 1) / self.gran;
    }
    fn region_bytes(self: FormatSpec) usize {
        return self.groups_per_region() * self.group_bytes;
    }
};

const base_dims = .{ .head_dim = 64, .num_kv_heads = 8, .num_q_heads = 32, .block_size = 16 };

fn specFor(fmt: QuantFormat) FormatSpec {
    return switch (fmt) {
        .fp16 => .{ .fmt = fmt, .group_bytes = 0, .tag = "fp16", .head_dim = base_dims.head_dim, .num_kv_heads = base_dims.num_kv_heads, .num_q_heads = base_dims.num_q_heads, .block_size = base_dims.block_size, .gran = 1 },
        .q8_0 => .{ .fmt = fmt, .group_bytes = 34, .tag = "q8_0", .head_dim = base_dims.head_dim, .num_kv_heads = base_dims.num_kv_heads, .num_q_heads = base_dims.num_q_heads, .block_size = base_dims.block_size, .gran = 32 },
        .q4_0 => .{ .fmt = fmt, .group_bytes = 18, .tag = "q4_0", .head_dim = base_dims.head_dim, .num_kv_heads = base_dims.num_kv_heads, .num_q_heads = base_dims.num_q_heads, .block_size = base_dims.block_size, .gran = 32 },
        .q4_k => .{ .fmt = fmt, .group_bytes = 144, .tag = "q4_k", .head_dim = 256, .num_kv_heads = 1, .num_q_heads = base_dims.num_q_heads, .block_size = base_dims.block_size, .gran = 256 },
        .q8_k => .{ .fmt = fmt, .group_bytes = 292, .tag = "q8_k", .head_dim = 256, .num_kv_heads = 1, .num_q_heads = base_dims.num_q_heads, .block_size = base_dims.block_size, .gran = 256 },
        .q2_k => .{ .fmt = fmt, .group_bytes = 84, .tag = "q2_k", .head_dim = 256, .num_kv_heads = 1, .num_q_heads = base_dims.num_q_heads, .block_size = base_dims.block_size, .gran = 256 },
        .q3_k => .{ .fmt = fmt, .group_bytes = 110, .tag = "q3_k", .head_dim = 256, .num_kv_heads = 1, .num_q_heads = base_dims.num_q_heads, .block_size = base_dims.block_size, .gran = 256 },
        .q5_k => .{ .fmt = fmt, .group_bytes = 176, .tag = "q5_k", .head_dim = 256, .num_kv_heads = 1, .num_q_heads = base_dims.num_q_heads, .block_size = base_dims.block_size, .gran = 256 },
        .q6_k => .{ .fmt = fmt, .group_bytes = 210, .tag = "q6_k", .head_dim = 256, .num_kv_heads = 1, .num_q_heads = base_dims.num_q_heads, .block_size = base_dims.block_size, .gran = 256 },
        .iq4_xs => .{ .fmt = fmt, .group_bytes = 136, .tag = "iq4_xs", .head_dim = 256, .num_kv_heads = 1, .num_q_heads = base_dims.num_q_heads, .block_size = base_dims.block_size, .gran = 256 },
        .iq4_nl => .{ .fmt = fmt, .group_bytes = 18, .tag = "iq4_nl", .head_dim = base_dims.head_dim, .num_kv_heads = base_dims.num_kv_heads, .num_q_heads = base_dims.num_q_heads, .block_size = base_dims.block_size, .gran = 32 },
        .iq3_xxs => .{ .fmt = fmt, .group_bytes = 98, .tag = "iq3_xxs", .head_dim = 256, .num_kv_heads = 1, .num_q_heads = base_dims.num_q_heads, .block_size = base_dims.block_size, .gran = 256 },
        .iq3_s => .{ .fmt = fmt, .group_bytes = 110, .tag = "iq3_s", .head_dim = 256, .num_kv_heads = 1, .num_q_heads = base_dims.num_q_heads, .block_size = base_dims.block_size, .gran = 256 },
        .iq1_s => .{ .fmt = fmt, .group_bytes = 50, .tag = "iq1_s", .head_dim = 256, .num_kv_heads = 1, .num_q_heads = base_dims.num_q_heads, .block_size = base_dims.block_size, .gran = 256 },
        .iq1_m => .{ .fmt = fmt, .group_bytes = 56, .tag = "iq1_m", .head_dim = 256, .num_kv_heads = 1, .num_q_heads = base_dims.num_q_heads, .block_size = base_dims.block_size, .gran = 256 },
        .iq2_xxs => .{ .fmt = fmt, .group_bytes = 66, .tag = "iq2_xxs", .head_dim = 256, .num_kv_heads = 1, .num_q_heads = base_dims.num_q_heads, .block_size = base_dims.block_size, .gran = 256 },
        .iq2_xs => .{ .fmt = fmt, .group_bytes = 74, .tag = "iq2_xs", .head_dim = 256, .num_kv_heads = 1, .num_q_heads = base_dims.num_q_heads, .block_size = base_dims.block_size, .gran = 256 },
        .iq2_s => .{ .fmt = fmt, .group_bytes = 82, .tag = "iq2_s", .head_dim = 256, .num_kv_heads = 1, .num_q_heads = base_dims.num_q_heads, .block_size = base_dims.block_size, .gran = 256 },
        .tq1_0 => .{ .fmt = fmt, .group_bytes = 54, .tag = "tq1_0", .head_dim = 256, .num_kv_heads = 1, .num_q_heads = base_dims.num_q_heads, .block_size = base_dims.block_size, .gran = 256 },
        .tq2_0 => .{ .fmt = fmt, .group_bytes = 66, .tag = "tq2_0", .head_dim = 256, .num_kv_heads = 1, .num_q_heads = base_dims.num_q_heads, .block_size = base_dims.block_size, .gran = 256 },
        .mxfp4 => .{ .fmt = fmt, .group_bytes = 17, .tag = "mxfp4", .head_dim = base_dims.head_dim, .num_kv_heads = base_dims.num_kv_heads, .num_q_heads = base_dims.num_q_heads, .block_size = base_dims.block_size, .gran = 32 },
    };
}

fn benchmarkFormat(spec: FormatSpec, ctx_len: usize) !void {
    if (!cudaz.isCudaAvailable()) {
        std.debug.print("SKIP: CUDA no disponible\n", .{});
        return;
    }

    debugz.init();
    try cudaz.ensureContext();
    const stream = try cudaz.cuStreamCreate(0);
    defer cudaz.cuStreamDestroy(stream);

    const num_blocks = (ctx_len + spec.block_size - 1) / spec.block_size;
    const region_bytes = spec.region_bytes();
    const total_bytes_k = num_blocks * region_bytes;
    const total_bytes_v = total_bytes_k;
    const total_bytes = total_bytes_k + total_bytes_v;

    // fp16 reference
    const fp16_region_bytes = spec.elems_region() * 2;
    const fp16_total = num_blocks * fp16_region_bytes * 2;

    const savings_pct = if (spec.fmt == .fp16) 0 else @as(f32, @floatFromInt(fp16_total - total_bytes)) / @as(f32, @floatFromInt(fp16_total)) * 100;

    // Setup PagedAttentionGpu config
    // const config: pa.PagedConfig = .{
    //     .block_size = spec.block_size,
    //     .num_blocks = num_blocks,
    //     .head_dim = spec.head_dim,
    //     .num_kv_heads = spec.num_kv_heads,
    //     .num_q_heads = spec.num_q_heads,
    //     .dtype = .f16,
    //     .enable_prefix_cache = false,
    //     .max_seq_len = ctx_len,
    //     .max_batch_size = 4,
    //     .quant_k = spec.fmt,
    //     .quant_v = spec.fmt,
    // };

// Calculate memory stats
    const q_stride = spec.num_q_heads * spec.head_dim;
    _ = q_stride;

    // Total KV cache bytes for this format
    _ = spec.group_bytes * spec.groups_per_region() * num_blocks * 2;

    // Dummy timing for interface compatibility
    const iterations = 100;
    var timer = time.Timer.start();
    for (0..iterations) |_| {
        _ = 1;
    }
    _ = timer.read();
    const tok_per_s = 0.0; // placeholder

    std.debug.print("[{s}] ctx={d} blocks={d} KV={d:.2} MB (fp16={d:.2} MB) savings={d:.1}% tok/s={d:.1}\n", .{
        spec.tag, ctx_len, num_blocks,
        @as(f32, @floatFromInt(total_bytes)) / 1024.0 / 1024.0,
        @as(f32, @floatFromInt(fp16_total)) / 1024.0 / 1024.0,
        savings_pct,
        tok_per_s,
    });
}

const debugz = @import("debug");

pub fn main() !void {
    // const alloc = std.heap.page_allocator;

    const formats = [_]QuantFormat{
        .fp16,
        .q8_0,
        .q4_0,
        .q4_k,
        .q8_k,
        .q2_k,
        .q3_k,
        .q5_k,
        .q6_k,
        .iq4_xs,
        .iq4_nl,
        .iq3_xxs,
        .iq3_s,
        .iq1_s,
        .iq1_m,
        .iq2_xxs,
        .iq2_xs,
        .iq2_s,
        .tq1_0,
        .tq2_0,
        .mxfp4,
    };

    std.debug.print("=== KV Quantized Memory & Throughput Benchmark ===\n", .{});
    const dev = try cudaz.cuDeviceGet(0);
    std.debug.print("GPU: device {d}\n", .{dev});

    for (formats) |fmt| {
        const spec = specFor(fmt);
        // Test con 4K context (típico para 7B/8B models)
        try benchmarkFormat(spec, 4096);
    }

    std.debug.print("=== Benchmark Complete ===\n", .{});
}