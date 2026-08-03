const std = @import("std");
const Tensor = @import("core").Tensor;
const matmul = @import("matmul");
const fa = @import("fa");
const transformer = @import("transformer");

const FlashAttention = fa.FlashAttention;
const FlashAttentionCpu = fa.FlashAttentionCpu;
const FlashAttentionConfig = fa.fa_config.FlashAttentionConfig;
const TransformerLayer = transformer.TransformerLayer;
const LayerPrecision = transformer.LayerPrecision;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const stdout = std.io.getStdOut().writer();
    try stdout.print("\n", .{});
    try stdout.print("=================================================\n", .{});
    try stdout.print("     Zig AI Engine — FlashAttention + Matmul     \n", .{});
    try stdout.print("=================================================\n", .{});
    try stdout.print("\n", .{});

    const config = FlashAttentionConfig{
        .N = 512, .d = 128, .num_heads = 8, .batch_size = 1,
        .dtype = .f16, .causal = true,
    };

    try stdout.print("Config: N={d}, d={d}, heads={d}, batch={d}, causal={}\n", .{
        config.N, config.d, config.num_heads, config.batch_size, config.causal,
    });

    // Verificar CUDA
    const cuda_available = @import("cudaz").isCudaAvailable();
    try stdout.print("CUDA disponible: {}\n", .{cuda_available});

    if (!cuda_available) {
        try stdout.print("\n[!] CUDA no disponible. Ejecutando modo CPU...\n", .{});
        try runCpuMode(allocator, config);
        return;
    }

    // Modo GPU completo
    try stdout.print("\n[+] Inicializando TransformerLayer con GPU...\n", .{});

    const precision = LayerPrecision{
        .compute = .f16,
        .weights_on_gpu = false, // por simplicidad en demo
        .use_quantized = false,
    };

    var layer = try TransformerLayer.init(allocator, 0, config, "cuda/flash_attention.ptx", 1024, precision);
    defer layer.deinit();

    try stdout.print("[+] Capa transformer inicializada\n", .{});
    try stdout.print("[+] Backend matmul: {s}\n", .{layer.matmul_engine.backendName()});

    // Crear hidden state de entrada
    var hidden_state = try Tensor(f16).alloc(allocator, &.{ config.batch_size, config.N, 1024 });
    defer hidden_state.deinit();
    var rng = std.Random.Xoshiro256.init(42);
    hidden_state.randUniform(&rng, -0.1, 0.1);

    var output = try Tensor(f16).alloc(allocator, &.{ config.batch_size, config.N, 1024 });
    defer output.deinit();

    // Warmup
    try stdout.print("[*] Warmup...\n", .{});
    try layer.forward(hidden_state, &output);

    // Benchmark
    const iterations: usize = 10;
    var timer = try std.time.Timer.start();
    for (0..iterations) |_| {
        try layer.forward(hidden_state, &output);
    }
    const total_ns = timer.read();
    const avg_ms = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(iterations)) / 1_000_000.0;

    try stdout.print("\n[GPU] {d} iteraciones promedio: {d:.3} ms\n", .{ iterations, avg_ms });

    // Stats del pool si hay cuBLAS
    if (layer.matmul_engine.gpuPoolStats()) |stats| {
        try stdout.print("[GPU] Pool: {d} total, {d} usado, {d} libre\n", .{ stats.total, stats.used, stats.free });
    }

    try stdout.print("\n=================================================\n", .{});
    try stdout.print("              Ejecucion completada               \n", .{});
    try stdout.print("=================================================\n", .{});
}

fn runCpuMode(allocator: std.mem.Allocator, config: FlashAttentionConfig) !void {
    const stdout = std.io.getStdOut().writer();

    var fa_cpu = FlashAttentionCpu.init(allocator, config);

    var Q = try Tensor(f16).alloc(allocator, &.{ 1, 8, 512, 128 });
    defer Q.deinit();
    var K = try Tensor(f16).alloc(allocator, &.{ 1, 8, 512, 128 });
    defer K.deinit();
    var V = try Tensor(f16).alloc(allocator, &.{ 1, 8, 512, 128 });
    defer V.deinit();
    var O = try Tensor(f16).alloc(allocator, &.{ 1, 8, 512, 128 });
    defer O.deinit();

    fa.fa_utils.initUniform(&Q, -0.1, 0.1, 42);
    fa.fa_utils.initUniform(&K, -0.1, 0.1, 43);
    fa.fa_utils.initUniform(&V, -0.1, 0.1, 44);

    var timer = try std.time.Timer.start();
    try fa_cpu.forward(Q, K, V, &O);
    const elapsed = timer.read();

    try stdout.print("[CPU] Forward completado en {d:.2} ms\n", .{@as(f64, @floatFromInt(elapsed)) / 1_000_000.0});
    try stdout.print("[CPU] Output sample: ", .{});
    O.printHead(10);
}
